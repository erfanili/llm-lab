"""Reward model: SFT backbone + scalar head, Bradley-Terry loss on preference pairs.

    python train_rm.py configs/rm_50m.toml [section.key=value ...]

loss = -log sigmoid(r(chosen) - r(rejected)) + center_coef * (r(chosen) + r(rejected))^2
(the second term keeps rewards centred near 0, which keeps the RLHF reward scale stable).
Each eval step: val pairwise accuracy with 95% CI (overall and per source) and a snapshot.
"""

import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from llm.chat import ChatFormat
from llm.config import load_config
from llm.evals import mean_ci
from llm.optim import cosine_lr, make_adamw, set_lr
from llm.posttrain import Logger, load_scalar_model, pad_right

D = "/data/erfan/llm-lab"


@dataclass
class RMConfig:
    name: str = "rm_50m"
    init_ckpt: str = f"{D}/runs/sft_50m/ckpt_final.pt"
    out_dir: str = f"{D}/runs"
    train_data: str = f"{D}/posttrain/prefs_train.jsonl"
    val_data: str = f"{D}/posttrain/prefs_val.jsonl"
    tokenizer: str = f"{D}/tokenizer/bpe32k.json"
    seed: int = 0
    epochs: float = 1.0
    micro_batch: int = 16  # pairs
    grad_accum: int = 4
    lr: float = 5e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    center_coef: float = 0.01
    log_interval: int = 10
    eval_interval: int = 250
    val_pairs: int = 100000  # all
    limit_train: int = 0  # >0: use only the first N training pairs (smoke tests)


@dataclass
class Config:
    rm: RMConfig = field(default_factory=RMConfig)


def tokenize_pairs(chat: ChatFormat, path: str) -> list[dict]:
    out = []
    for line in open(path):
        p = json.loads(line)
        prompt = chat.prompt(p["prompt"])
        out.append(dict(source=p["source"], chosen=prompt + chat.response_ids(p["chosen"]),
                        rejected=prompt + chat.response_ids(p["rejected"])))
    return out


def pair_rewards(rm, batch: list[dict], device) -> tuple[torch.Tensor, torch.Tensor]:
    x, lengths = pad_right([b["chosen"] for b in batch] + [b["rejected"] for b in batch], device=device)
    r = rm.score(x, lengths)
    return r[: len(batch)], r[len(batch):]


@torch.no_grad()
def evaluate(rm, val, n, B, device) -> dict:
    rm.eval()
    correct, margins, sources, rewards = [], [], [], []
    for i in range(0, min(n, len(val)), B):
        batch = val[i : i + B]
        rc, rr = pair_rewards(rm, batch, device)
        correct += (rc > rr).tolist()
        margins += (rc - rr).tolist()
        rewards += rc.tolist() + rr.tolist()
        sources += [b["source"] for b in batch]
    rm.train()
    acc = mean_ci(correct)
    out = dict(val_acc=acc["mean"], val_acc_ci95=acc["ci95"], val_margin=float(np.mean(margins)),
               val_reward_mean=float(np.mean(rewards)), val_reward_std=float(np.std(rewards)), n=acc["n"])
    for s in sorted(set(sources)):
        out[f"val_acc/{s}"] = float(np.mean([c for c, src in zip(correct, sources) if src == s]))
    return out


def main():
    cfg = load_config(Config, sys.argv[1], sys.argv[2:])
    c = cfg.rm
    device = torch.device("cuda")
    torch.manual_seed(c.seed)
    run_dir = Path(c.out_dir) / c.name
    log = Logger(run_dir)

    rm = load_scalar_model(c.init_ckpt, device, "reward_head")
    rm.train()
    model_cfg = asdict(rm.cfg)
    chat = ChatFormat(Tokenizer.from_file(c.tokenizer))
    train, val = tokenize_pairs(chat, c.train_data), tokenize_pairs(chat, c.val_data)
    # Files are grouped by source: take fixed random subsets so every source is represented.
    if c.limit_train:
        train = [train[i] for i in np.random.default_rng(0).permutation(len(train))[: c.limit_train]]
    if c.val_pairs < len(val):
        val = [val[i] for i in sorted(np.random.default_rng(1).permutation(len(val))[: c.val_pairs])]
    per_step = c.micro_batch * c.grad_accum
    total_steps = int(c.epochs * len(train) / per_step)
    opt = make_adamw(rm, c.lr, c.weight_decay)
    log(dict(kind="setup", n_train=len(train), n_val=len(val), total_steps=total_steps, init_ckpt=c.init_ckpt))
    rng = np.random.default_rng(c.seed)
    order = np.concatenate([rng.permutation(len(train)) for _ in range(int(np.ceil(c.epochs)) + 1)])

    def checkpoint(step):
        r = evaluate(rm, val, c.val_pairs, c.micro_batch * 2, device)
        log(dict(kind="eval", step=step, **r))
        # flat_state(): backbone under plain Transformer names + reward_head.*
        torch.save(dict(model=rm.flat_state(), config=dict(model=model_cfg, rm=asdict(c)), step=step, eval=r),
                   run_dir / f"ckpt_step{step}.pt")

    for step in range(total_steps):
        if step % c.eval_interval == 0:
            checkpoint(step)
        set_lr(opt, lr := cosine_lr(step, total_steps, c.lr, c.warmup_steps, c.min_lr_ratio))
        idx = order[step * per_step : (step + 1) * per_step]
        opt.zero_grad(set_to_none=True)
        stats = np.zeros(3)
        for i in range(0, per_step, c.micro_batch):
            rc, rr = pair_rewards(rm, [train[j] for j in idx[i : i + c.micro_batch]], device)
            bt = -F.logsigmoid(rc - rr).mean()
            loss = bt + c.center_coef * ((rc + rr) ** 2).mean()
            (loss / c.grad_accum).backward()
            stats += [bt.item(), (rc > rr).float().mean().item(), (rc - rr).mean().item()]
        gn = torch.nn.utils.clip_grad_norm_(rm.parameters(), c.grad_clip).item()
        opt.step()
        if (step + 1) % c.log_interval == 0:
            s = stats / c.grad_accum
            log(dict(kind="train", step=step + 1, loss=s[0], acc=s[1], margin=s[2], lr=lr, grad_norm=gn))
    checkpoint(total_steps)
    shutil.copy(run_dir / f"ckpt_step{total_steps}.pt", run_dir / "ckpt_final.pt")


if __name__ == "__main__":
    main()
