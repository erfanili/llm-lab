"""Shared pieces for SFT, reward modelling and RLHF."""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.checkpoint import load_model
from llm.generate import generate
from llm.model import ModelConfig, Transformer


def pad_right(seqs: list[list[int]], pad: int = 0, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (B, T) ids, (B,) lengths. Right padding needs no attention mask: causal attention
    means real tokens never see the padding after them."""
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.as_tensor(s)
    return x.to(device), torch.tensor([len(s) for s in seqs], device=device)


def token_logprobs(model: Transformer, x: torch.Tensor) -> torch.Tensor:
    """log P(x[:, t+1] | x[:, :t+1]) for every position -> (B, T-1), computed in fp32."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        logits = model(x[:, :-1])
    return -F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none")


class ScalarHeadModel(nn.Module):
    """Transformer backbone + linear scalar head. Used both as the reward model (score read at the
    final token) and as the PPO value function (a value at every position)."""

    def __init__(self, cfg: ModelConfig, head_name: str = "reward_head"):
        super().__init__()
        self.backbone = Transformer(cfg)
        self.head_name = head_name
        self.add_module(head_name, nn.Linear(cfg.d_model, 1))
        nn.init.normal_(self.head.weight, std=1 / math.sqrt(cfg.d_model + 1))
        nn.init.zeros_(self.head.bias)

    @property
    def head(self) -> nn.Linear:
        return getattr(self, self.head_name)

    @property
    def cfg(self):
        return self.backbone.cfg

    def per_token(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            h = self.backbone.hidden(x)
        return self.head(h.float()).squeeze(-1)  # (B, T)

    def score(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Scalar per sequence, read at its last real token (the closing <|endoftext|>)."""
        return self.per_token(x)[torch.arange(len(x), device=x.device), lengths - 1]

    def flat_state(self) -> dict:
        """Backbone weights under plain Transformer names + the head: loadable by llm.checkpoint.load_model."""
        sd = {k.removeprefix("backbone."): v for k, v in self.state_dict().items()}
        return sd


def load_scalar_model(path, device, head_name="reward_head", init_head_from=None) -> ScalarHeadModel:
    """Backbone from any checkpoint; head from the checkpoint if present (e.g. a trained RM)."""
    backbone, ck = load_model(path, device)
    m = ScalarHeadModel(backbone.cfg, head_name)
    m.backbone.load_state_dict(backbone.state_dict())
    src = init_head_from or head_name
    if f"{src}.weight" in ck["model"]:
        m.head.weight.data.copy_(ck["model"][f"{src}.weight"])
        m.head.bias.data.copy_(ck["model"][f"{src}.bias"])
    return m.to(device)


class Logger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.f = open(run_dir / "log.jsonl", "a")
        self.t0 = time.time()

    def __call__(self, record: dict):
        record = dict(**record, elapsed=round(time.time() - self.t0, 1))
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()


@torch.no_grad()
def generate_on_prompts(model, chat, prompts: list[dict], out_path: Path, max_new: int, meta: dict):
    """Greedy chat-mode generations on the inspection prompts, saved per checkpoint."""
    was = model.training
    model.eval()
    inputs = [chat.prompt([dict(role="user", content=p["prompt"])]) for p in prompts]
    outs = generate(model, inputs, max_new, temperature=0.0, stop_ids=(chat.eot,))
    with open(out_path, "w") as f:
        for p, o in zip(prompts, outs):
            f.write(json.dumps(dict(id=p["id"], category=p["category"], prompt=p["prompt"], mode="chat", sample=0,
                                    temperature=0.0, response=chat.decode(o["tokens"]), n_tokens=len(o["tokens"]),
                                    finish=o["finish"], **meta)) + "\n")
    model.train(was)


def load_inspect_prompts(path) -> list[dict]:
    return [json.loads(l) for l in open(path)]
