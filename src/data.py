"""Shard-backed data loading for pretokenized uint16 GPT-2 corpora.

Shards are raw uint16 token-id files named ``{source}_{NNN}.bin`` (EOS-separated
docs), as published in ichangzii/pit2022-10b. We chop every shard into
non-overlapping windows of ``seq_len + 1`` tokens (input + shifted target),
globally shuffle the window index once with a fixed seed, and walk it — so an
epoch covers every token exactly once, with sources naturally mixed in
proportion to the tokens on disk. The first ``val_windows`` positions of the
shuffled order are held out as a fixed validation set.
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np
import torch


class ShardDataset:
    def __init__(self, data_dir: str, seq_len: int = 1024, seed: int = 1337,
                 val_windows: int = 256):
        self.seq_len = seq_len
        paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(f"no .bin shards under {data_dir}")
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        window = seq_len + 1
        index = []  # (shard_idx, start)
        per_source = defaultdict(int)
        for si, (p, s) in enumerate(zip(paths, self.shards)):
            n = len(s) // window
            index.extend((si, w * window) for w in range(n))
            per_source[os.path.basename(p).rsplit("_", 1)[0]] += n * window
        rng = np.random.default_rng(seed)
        self.index = np.array(index, dtype=np.int64)
        rng.shuffle(self.index)
        self.val_index = self.index[:val_windows]
        self.train_index = self.index[val_windows:]
        total = sum(per_source.values())
        print(f"[data] {len(paths)} shards, {total/1e9:.2f}B tokens in "
              f"{len(self.index)} windows ({val_windows} held out for val)")
        for src, tok in sorted(per_source.items()):
            print(f"[data]   {src:8s} {tok/1e9:5.2f}B  ({tok/total:5.1%})")

    def _fetch(self, index: np.ndarray, i: int, bs: int, device: str):
        rows = index[i:i + bs]
        w = self.seq_len + 1
        buf = np.stack([self.shards[si][st:st + w] for si, st in rows]).astype(np.int64)
        t = torch.from_numpy(buf).to(device, non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def train_batches(self, bs: int, device: str = "cuda"):
        """Infinite iterator; wraps (and stays shuffled) after a full epoch."""
        i = 0
        n = len(self.train_index)
        while True:
            if i + bs > n:
                i = 0
            yield self._fetch(self.train_index, i, bs, device)
            i += bs

    def val_batches(self, bs: int, device: str = "cuda"):
        for i in range(0, len(self.val_index) - bs + 1, bs):
            yield self._fetch(self.val_index, i, bs, device)
