# LLM_trainer — Optimized Point-in-Time GPT-2 (certified ≤2022 cutoff)

Pretrain GPT-2-124M **from scratch on ≤2022-only data** so post-2022 knowledge was
*never in the weights* (a certified, lookahead-free knowledge cutoff), and **optimize the
training method** to beat the reference recipe at equal compute.

**Reference baseline** (nanoGPT-standard recipe, same data family, same GPU class):
[ichangzii/pit2022-gpt2-124m](https://huggingface.co/ichangzii/pit2022-gpt2-124m) —
WikiText-2 ppl **43.1** at 4.2B tokens, 137k tok/s on an RTX 5090.

## What is optimized (data ≤2022 is the only hard constraint — methods are free)

| axis | baseline (`--recipe gpt2`) | ours (`--recipe modern`) |
|---|---|---|
| positions | learned absolute | **RoPE** |
| norm | LayerNorm | **RMSNorm** (no gain) + **QK-norm** |
| MLP | GELU | **ReLU²** |
| head | tied | **untied, zero-init** (+ zero-init residual projections) |
| optimizer | AdamW 6e-4 | **Muon** (hidden matrices) + AdamW (embed/head) |
| schedule | cosine, 500 warmup | **WSD** (warmup → constant → 30% linear cooldown) |
| data mix | 10B corpus as-is (20% code+math) | **re-weighted**: code+math ↓ to ~11%, web/wiki/books ↑ |

Both recipes share one codebase (`src/model.py`), ~0.5M tokens/step
(bs 32 × accum 16 × seq 1024), bf16 autocast, grad-clip 1.0, `torch.compile`,
and a non-overlapping globally-shuffled sampler (every token at most once).

## Data

Subset of [ichangzii/pit2022-10b](https://huggingface.co/datasets/ichangzii/pit2022-10b)
(all sources datable ≤2022-12-31; GPT-2 tokenizer, uint16 shards), re-weighted to
~4.7B tokens: web 40% · dclm 26% · books 13% · wiki 11% · code 6% · math 4%.
The reference model's ppl plateau was diagnosed as domain mismatch from its
20% code+math share — this mix is the fix.

## Pipeline

```bash
pip install -r requirements.txt
python scripts/download_data.py --out data/pit2022      # ~9.4GB
python train.py --recipe modern --steps 8000            # 4.2B tokens (= baseline budget)
python scripts/eval_ppl.py    --model out/modern/step8000   # WikiText-2 ppl
python scripts/probe_cutoff.py --model out/modern/step8000  # certify the ≤2022 cutoff
```

`eval_ppl.py` / `probe_cutoff.py` also accept HF model names, so the reference
model is scored under the **same protocol** (`--model ichangzii/pit2022-gpt2-124m`).

## Cutoff certification

`scripts/probe_cutoff.py` reads p(answer | prompt) for genuinely novel post-2022
entities (Threads, Sora, Gemini, DeepSeek, Grok, …) vs ≤2022 controls (COVID,
Ukraine invasion, Brexit, …). Certified = post-2022 mass ~0 while ≤2022 mass is
clearly present. This must be re-run after **every** method change: the methods are
free, the data cutoff is not.

## Results

_(filled in as runs complete)_
