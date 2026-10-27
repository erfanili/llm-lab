"""Supervised fine-tuning on chat data (loss only on assistant tokens).

    python sft.py configs/sft_50m.toml [section.key=value ...]

Each eval step writes: masked val loss, a weights-only snapshot ckpt_step{N}.pt, and greedy
chat generations on the inspection prompts (inspect/step{N}.jsonl) for side-by-side reading.
"""

import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from llm.chat import ChatFormat
from llm.checkpoint import load_model, save_weights
from llm.config import load_config
from llm.optim import cosine_lr, make_adamw, set_lr
from llm.posttrain import Logger, generate_on_prompts, load_inspect_prompts, pad_right

D = "/data/erfan/llm-lab"
REPO = Path(__file__).resolve().parent


@dataclass
class SFTConfig:
    name: str = "sft_50m"
    init_ckpt: str = f"{D}/runs/50m/ckpt.pt"
    out_dir: str = f"{D}/runs"
    train_data: str = f"{D}/posttrain/sft_train.pt"
    val_data: str = f"{D}/posttrain/sft_val.pt"
    tokenizer: str = f"{D}/tokenizer/bpe32k.json"
    inspect_prompts: str = str(REPO / "data/prompts/inspect_v1.jsonl")
    seed: int = 0
    epochs: float = 2.0
    micro_batch: int = 32
    grad_accum: int = 2
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 250
    val_convs: int = 2000
    inspect_max_new: int = 128
    limit_train: int = 0  # >0: use only the first N training conversations (smoke tests)


@dataclass
class Config:
    sft: SFTConfig = field(default_factory=SFTConfig)


def masked_loss_sum(model, ids: list, masks: list, device) -> tuple[torch.Tensor, int]:
    x, _ = pad_right([t.tolist() for t in ids], device=device)
    m, _ = pad_right([mk.long().tolist() for mk in masks], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x[:, :-1])
    nll = F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none")
    tgt_mask = m[:, 1:].float()  # mask[t] marks token t as an assistant target
    return (nll * tgt_mask).sum(), int(tgt_mask.sum().item())


@torch.no_grad()
def eval_val(model, data, n, B, device) -> dict:
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, min(n, len(data["ids"])), B):
        s, c = masked_loss_sum(model, data["ids"][i : i + B], data["mask"][i : i + B], device)
        tot += s.item()
        cnt += c
    model.train()
    return dict(val_loss=tot / cnt, val_tokens=cnt)


def main():
    cfg = load_config(Config, sys.argv[1], sys.argv[2:])
    c = cfg.sft
    device = torch.device("cuda")
    torch.manual_seed(c.seed)
    run_dir = Path(c.out_dir) / c.name
    log = Logger(run_dir)
    (run_dir / "inspect").mkdir(exist_ok=True)

    model, init = load_model(c.init_ckpt, device)
    model.train()
    model_cfg = init["config"]["model"]
    chat = ChatFormat(Tokenizer.from_file(c.tokenizer))
    prompts = load_inspect_prompts(c.inspect_prompts)
    train, val = torch.load(c.train_data), torch.load(c.val_data)
    if c.limit_train:
        train = {k: v[: c.limit_train] for k, v in train.items()}
    n_train = len(train["ids"])
    per_step = c.micro_batch * c.grad_accum
    total_steps = int(c.epochs * n_train / per_step)
    opt = make_adamw(model, c.lr, c.weight_decay)
    log(dict(kind="setup", n_train=n_train, total_steps=total_steps, init_ckpt=c.init_ckpt))

    rng = np.random.default_rng(c.seed)
    order = np.concatenate([rng.permutation(n_train) for _ in range(int(np.ceil(c.epochs)) + 1)])

    def checkpoint(step):
        r = eval_val(model, val, c.val_convs, c.micro_batch, device)
        log(dict(kind="eval", step=step, **r))
        save_weights(run_dir / f"ckpt_step{step}.pt", model, dict(model=model_cfg, sft=asdict(c)), step=step, **r)
        generate_on_prompts(model, chat, prompts, run_dir / "inspect" / f"step{step}.jsonl", c.inspect_max_new,
                            dict(ckpt=str(run_dir / f"ckpt_step{step}.pt"), step=step))

    for step in range(total_steps):
        if step % c.eval_interval == 0:
            checkpoint(step)
        set_lr(opt, lr := cosine_lr(step, total_steps, c.lr, c.warmup_steps, c.min_lr_ratio))
        idx = order[step * per_step : (step + 1) * per_step]
        micro = [idx[i : i + c.micro_batch] for i in range(0, len(idx), c.micro_batch)]
        n_tok = sum(int(train["mask"][j][1:].sum()) for j in idx)  # normalise by assistant tokens in the step
        opt.zero_grad(set_to_none=True)
        loss_tot = 0.0
        for mb in micro:
            s, _ = masked_loss_sum(model, [train["ids"][j] for j in mb], [train["mask"][j] for j in mb], device)
            (s / n_tok).backward()
            loss_tot += s.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip).item()
        opt.step()
        if (step + 1) % c.log_interval == 0:
            log(dict(kind="train", step=step + 1, loss=loss_tot / n_tok, lr=lr, grad_norm=gn, tokens=n_tok))
    checkpoint(total_steps)
    shutil.copy(run_dir / f"ckpt_step{total_steps}.pt", run_dir / "ckpt_final.pt")


if __name__ == "__main__":
    main()
