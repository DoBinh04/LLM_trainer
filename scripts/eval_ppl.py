#!/usr/bin/env python
"""WikiText-2 (raw, test) perplexity with a sliding window.

Window 1024, stride 512, scoring only the final 512 tokens of each window
(standard HF protocol). Works on our checkpoints and HF models, so reference
models can be scored under the *same* protocol:

    python scripts/eval_ppl.py --model out/modern/step8000
    python scripts/eval_ppl.py --model ichangzii/pit2022-gpt2-124m
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wikitext2_test_text() -> str:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                        repo_type="dataset")
    return "".join(pq.read_table(p).column("text").to_pylist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    from src.load import load_model, logits_of

    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok(wikitext2_test_text(), return_tensors="pt").input_ids[0].to("cuda")
    model, kind = load_model(args.model)

    nll_sum, count = 0.0, 0
    with torch.autocast("cuda", torch.bfloat16):
        for begin in range(0, len(ids) - 1, args.stride):
            end = min(begin + args.window, len(ids) - 1)
            x = ids[begin:end].unsqueeze(0)
            y = ids[begin + 1:end + 1]
            logits = logits_of(model, kind, x)[0].float()
            nll = F.cross_entropy(logits, y, reduction="none")
            score_from = 0 if begin == 0 else args.window - args.stride
            nll_sum += nll[score_from:].sum().item()
            count += (end - begin) - score_from
            if end == len(ids) - 1:
                break
    ppl = math.exp(nll_sum / count)
    print(f"[ppl] model={args.model} wikitext2_ppl={ppl:.2f} tokens_scored={count}")


if __name__ == "__main__":
    main()
