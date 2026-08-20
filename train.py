#!/usr/bin/env python
"""Point-in-time pretraining of GPT-2-124M on <=2022 data, two recipes.

  baseline:  python train.py --recipe gpt2   --steps 1500
  modern:    python train.py --recipe modern --steps 1500

gpt2   = nanoGPT-standard: AdamW 6e-4, cosine + 500 warmup, wd 0.1 (the
         reference recipe used by the pit2022-gpt2-124m baseline).
modern = Muon (hidden matrices) + AdamW (embed/head), WSD schedule
         (warmup -> constant -> linear cooldown over the last 30%).

Both use ~0.5M tokens/step (bs 32 x accum 16 x seq 1024), bf16 autocast,
grad-clip 1.0, torch.compile.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=["gpt2", "modern"], default="modern")
    ap.add_argument("--data-dir", default="data/pit2022")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    # gpt2 recipe
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    # modern recipe
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--modern-warmup", type=int, default=250)
    ap.add_argument("--cooldown-frac", type=float, default=0.3)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    out = args.out or f"out/{args.recipe}"
    os.makedirs(out, exist_ok=True)

    import torch
    from src.data import ShardDataset
    from src.model import GPT, GPTConfig
    from src.muon import Muon

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    dev = "cuda"

    cfg = GPTConfig(recipe=args.recipe, seq_len=args.seq_len)
    model = GPT(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] recipe={args.recipe} params={n_params/1e6:.1f}M", flush=True)

    data = ShardDataset(args.data_dir, seq_len=args.seq_len, seed=args.seed)
    batches = data.train_batches(args.bs, dev)

    if args.recipe == "gpt2":
        opts = [torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)]

        def lr_mult(step: int) -> float:
            if step < args.warmup:
                return (step + 1) / args.warmup
            t = (step - args.warmup) / max(1, args.steps - args.warmup)
            return 0.5 * (1 + math.cos(math.pi * t))
    else:
        matrix, other = model.param_groups()
        opts = [Muon(matrix, lr=args.muon_lr, momentum=0.95),
                torch.optim.AdamW(other, lr=args.adam_lr,
                                  betas=(0.9, 0.95), weight_decay=0.01)]

        def lr_mult(step: int) -> float:
            if step < args.modern_warmup:
                return (step + 1) / args.modern_warmup
            cd_start = int(args.steps * (1 - args.cooldown_frac))
            if step < cd_start:
                return 1.0
            return max(0.0, 1 - (step - cd_start) / max(1, args.steps - cd_start))

    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    raw_model = model
    if not args.no_compile:
        model = torch.compile(model)

    def run_val() -> float:
        raw_model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for x, y in data.val_batches(args.bs, dev):
                _, loss = model(x, y)
                tot += loss.item(); n += 1
        raw_model.train()
        return tot / max(1, n)

    log_path = os.path.join(out, "log.jsonl")
    tok_per_step = args.bs * args.grad_accum * args.seq_len
    t0 = time.time()
    ema = None
    for step in range(args.steps):
        m = lr_mult(step)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * m
        for _ in range(args.grad_accum):
            x, y = next(batches)
            with torch.autocast("cuda", torch.bfloat16):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        li = loss.item()
        ema = li if ema is None else 0.97 * ema + 0.03 * li
        if step % 25 == 0 or step == args.steps - 1:
            el = time.time() - t0
            tps = tok_per_step * (step + 1) / el
            rec = {"step": step, "loss": round(li, 4), "ema": round(ema, 4),
                   "lr_mult": round(m, 4), "tok_s": int(tps),
                   "tokens": tok_per_step * (step + 1)}
            if step % args.val_every == 0 or step == args.steps - 1:
                rec["val_loss"] = round(run_val(), 4)
            print(f"[train] {json.dumps(rec)}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            ck = os.path.join(out, f"step{step+1}")
            os.makedirs(ck, exist_ok=True)
            torch.save({"model": raw_model.state_dict(), "cfg": vars(cfg),
                        "step": step + 1, "args": vars(args)},
                       os.path.join(ck, "ckpt.pt"))
            print(f"[train] saved {ck}", flush=True)
    print(f"[train] done in {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
