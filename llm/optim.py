"""Optimizer construction and LR schedule shared by all trainers."""

import math

import torch


def make_adamw(model: torch.nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.95), eps=1e-8):
    # Weight decay on matrices (incl. the tied embedding); none on norm gains / biases.
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=torch.cuda.is_available())


def cosine_lr(step: int, total_steps: int, lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * lr."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def set_lr(opt: torch.optim.Optimizer, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr
