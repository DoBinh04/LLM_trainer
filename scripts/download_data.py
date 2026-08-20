#!/usr/bin/env python
"""Download a re-weighted subset of the pit2022-10b corpus (all data <=2022).

The full corpus is 10B tokens with 20% code+math, which is known to hurt
WikiText-style ppl (domain mismatch). We pull ~4.7B tokens with the mix shifted
toward quality web / wiki / books: web 19, dclm 12, books 6, wiki 5, code 3,
math 2 shards (~100M tokens each).

    python scripts/download_data.py --out data/pit2022
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

REPO = "ichangzii/pit2022-10b"
# source -> number of ~100M-token shards to fetch (shard files are ~200MB uint16)
MIX = {"web": 19, "dclm": 12, "books": 6, "wiki": 5, "code": 3, "math": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pit2022")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for src, n in MIX.items():
        for i in range(n):
            fname = f"{src}_{i:03d}.bin"
            print(f"[dl] {fname}")
            p = os.path.realpath(hf_hub_download(REPO, fname, repo_type="dataset"))
            dst = os.path.join(args.out, fname)
            if os.path.lexists(dst) and not os.path.exists(dst):
                os.remove(dst)  # broken link from an earlier run
            if not os.path.exists(dst):
                if os.stat(p).st_dev == os.stat(args.out).st_dev:
                    os.link(p, dst)
                else:
                    os.symlink(p, dst)
    print("[dl] done ->", args.out)


if __name__ == "__main__":
    main()
