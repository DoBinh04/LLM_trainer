#!/usr/bin/env python
"""Certify the <=2022 knowledge cutoff by causal completion probing.

For each probe we read p(first answer token | prompt). POST-2022 entities are
genuinely novel proper nouns a <=2022 model cannot have seen (should read ~0);
PRE-2022 events/products are the positive control (should read >> 0). A clean
cutoff = post-2022 mass near zero while pre-2022 mass is clearly present.

    python scripts/probe_cutoff.py --model out/modern/step8000
    python scripts/probe_cutoff.py --model ichangzii/pit2022-gpt2-124m
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (name, prompt, answer, is_post_2022) — answers are distinctive event/product
# words (never famous person names, which a model completes from the name alone).
PROBES = [
    # POST-2022: novel, unforeseen 2023+ entities
    ("threads",  "The name of Meta's new microblogging app is", "Threads", True),
    ("sora",     "OpenAI's video generation model is called", "Sora", True),
    ("gemini",   "Google's most advanced AI model is named", "Gemini", True),
    ("deepseek", "The Chinese AI lab behind the R1 reasoning model is", "DeepSeek", True),
    ("grok",     "The AI chatbot built into Elon Musk's X platform is", "Grok", True),
    ("bard",     "Google's conversational AI chatbot is called", "Bard", True),
    ("llama",    "Meta's family of open large language models is called", "Llama", True),
    ("mistral",  "The French AI startup known for open models is called", "Mistral", True),
    # PRE-2022 (<=2022): the model SHOULD know these
    ("covid",    "The pandemic that spread around the world in 2020 was caused by the",
     "coronavirus", False),
    ("ukraine",  "In February 2022, Russia launched a full-scale invasion of", "Ukraine", False),
    ("brexit",   "The British vote to leave the European Union was nicknamed", "Brexit", False),
    ("bitcoin",  "The most famous cryptocurrency is called", "Bitcoin", False),
    ("iphone",   "Apple's flagship smartphone is called the", "iPhone", False),
    ("tiktok",   "The short-form video app owned by ByteDance is called", "TikTok", False),
    ("zoom",     "During lockdown, most video meetings happened on", "Zoom", False),
    ("tesla",    "The electric car company led by Elon Musk is called", "Tesla", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from src.load import load_model, logits_of

    tok = AutoTokenizer.from_pretrained("gpt2")
    model, kind = load_model(args.model)

    post, pre = [], []
    print(f"[probe] model={args.model}")
    for name, prompt, answer, is_post in PROBES:
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        ans_id = tok(" " + answer).input_ids[0]  # first BPE token of the answer
        with torch.no_grad():
            probs = torch.softmax(logits_of(model, kind, ids)[0, -1].float(), dim=-1)
        p = probs[ans_id].item()
        (post if is_post else pre).append(p)
        tag = "POST" if is_post else "PRE "
        print(f"[probe] {tag} {name:9s} p({answer!r})={p:.4f}")
    mp, mq = sum(post) / len(post), sum(pre) / len(pre)
    print(f"[probe] mean POST-2022 = {mp:.4f}  (must be ~0 for a certified cutoff)")
    print(f"[probe] mean PRE-2022  = {mq:.4f}  (should be clearly > 0)")
    verdict = "CERTIFIED" if mp < 0.01 and mq > 10 * mp else "NOT certified"
    print(f"[probe] cutoff: {verdict}")


if __name__ == "__main__":
    main()
