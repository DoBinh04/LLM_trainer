"""GPT model in two recipes sharing one implementation.

recipe="gpt2"   — faithful GPT-2-124M baseline: learned absolute positions,
                  LayerNorm (learnable), GELU MLP, tied embedding/head, biases.
recipe="modern" — modernized 124M: RoPE, RMSNorm (no learnable gain), QK-norm,
                  ReLU^2 MLP, untied zero-init head, zero-init residual
                  projections, no biases.

Both are 12L / 12H / d768 / seq 1024 on the GPT-2 tokenizer (vocab padded to
50304, a multiple of 64, for kernel efficiency; padded rows are never produced
by the tokenizer so they stay inert).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    recipe: str = "modern"  # "gpt2" | "modern"
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    seq_len: int = 1024


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, H, T, hd)
        T = x.size(2)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.modern = cfg.recipe == "modern"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        bias = not self.modern
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=bias)
        self.rotary = Rotary(self.head_dim, cfg.seq_len) if self.modern else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.modern:
            q, k = norm(q), norm(k)  # QK-norm
            q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.modern = cfg.recipe == "modern"
        bias = not self.modern
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x).square() if self.modern else F.gelu(x, approximate="tanh")
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.modern = cfg.recipe == "modern"
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        if not self.modern:
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modern:
            x = x + self.attn(norm(x))
            x = x + self.mlp(norm(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.modern = cfg.recipe == "modern"
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if not self.modern:
            self.wpe = nn.Embedding(cfg.seq_len, cfg.n_embd)
            self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if not self.modern:
            self.lm_head.weight = self.wte.weight  # tied
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        if self.modern:
            nn.init.zeros_(self.lm_head.weight)
            for b in self.blocks:  # zero-init residual writes
                nn.init.zeros_(b.attn.proj.weight)
                nn.init.zeros_(b.mlp.proj.weight)
        else:  # GPT-2 paper: scale residual projections by 1/sqrt(2*n_layer)
            s = 0.02 / math.sqrt(2 * self.cfg.n_layer)
            for b in self.blocks:
                nn.init.normal_(b.attn.proj.weight, std=s)
                nn.init.normal_(b.mlp.proj.weight, std=s)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.wte(idx)
        if not self.modern:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        x = norm(x) if self.modern else self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def param_groups(self):
        """Split params: hidden 2D matrices (Muon-eligible) vs everything else."""
        matrix, other = [], []
        head_and_embed = {id(self.wte.weight), id(self.lm_head.weight)}
        if not self.modern:
            head_and_embed.add(id(self.wpe.weight))
        for p in self.parameters():
            if p.ndim == 2 and id(p) not in head_and_embed:
                matrix.append(p)
            else:
                other.append(p)
        return matrix, other
