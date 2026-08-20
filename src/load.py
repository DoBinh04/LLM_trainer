"""Load a model for evaluation: our ckpt.pt checkpoints or any HF causal LM.

Lets the same eval scripts score our runs AND reference models
(ichangzii/pit2022-gpt2-124m, openai-community/gpt2) under one protocol.
"""
from __future__ import annotations

import os

import torch


def load_model(path_or_name: str, device: str = "cuda"):
    ckpt = os.path.join(path_or_name, "ckpt.pt")
    if os.path.isfile(ckpt):
        from src.model import GPT, GPTConfig
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model = GPT(GPTConfig(**blob["cfg"])).to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        return model, "local"
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path_or_name,
                                                 torch_dtype=torch.float32).to(device)
    model.eval()
    return model, "hf"


@torch.no_grad()
def logits_of(model, kind: str, x: torch.Tensor) -> torch.Tensor:
    if kind == "local":
        logits, _ = model(x)
        return logits
    return model(x).logits
