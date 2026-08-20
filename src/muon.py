"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Orthogonalizes the momentum-smoothed gradient of each 2D weight matrix via a
quintic Newton-Schulz iteration, then applies it with an aspect-ratio-scaled
step. Use ONLY for hidden 2D matrices (attention/MLP projections); embeddings,
the LM head and any 1D params should stay on AdamW.

Reference: Jordan et al., "Muon: An optimizer for hidden layers in neural
networks" (2024) — the recipe behind the modded-nanoGPT GPT-2 speedruns.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T (from the SVD of G) with 5 Newton-Schulz iterations in bf16."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - group["momentum"])
                eff = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(
                    eff.reshape(eff.size(0), -1), steps=group["ns_steps"]
                ).view_as(p).to(p.dtype)
                # scale so the update RMS is comparable across matrix shapes
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)
        return loss
