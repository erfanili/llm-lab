"""Pretraining loop.

    torchrun --nproc_per_node=2 train.py configs/50m.toml [key=value ...]

e.g. `train.max_steps=50 train.name=smoke`. Overrides are parsed as TOML values.
"""

import json
import math
import os
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from llm import dist
from llm.data import TokenFile, TrainLoader, load_meta, token_byte_lengths
from llm.evaluate import eval_loss
from llm.model import ModelConfig, Transformer


@dataclass
class TrainConfig:
    name: str = "debug"
    out_dir: str = "/data/erfan/llm-lab/runs"
    train_bin: str = "/data/erfan/llm-lab/tokens/train.bin"
    val_bin: str = "/data/erfan/llm-lab/tokens/val.bin"
    tokenizer: str = "/data/erfan/llm-lab/tokenizer/bpe32k.json"
    seed: int = 1337
    # batch: tokens/step = micro_batch * seq_len * grad_accum * world_size
    seq_len: int = 1024
    micro_batch: int = 32
    grad_accum: int = 4
    total_tokens: int = 1_000_000_000
    max_steps: int = -1  # if > 0, stop early (the LR schedule still follows total_tokens)
    # optimizer
    lr: float = 2e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    # bookkeeping
    log_interval: int = 10
    eval_interval: int = 250
    eval_tokens: int = 10 * 2**20
    ckpt_interval: int = 500
    compile: bool = True
    peak_flops: float = 111e12  # RTX A5000 dense bf16 tensor-core spec, per GPU


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str, overrides: list[str]) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for ov in overrides:
        key, val = ov.split("=", 1)
        section, name = key.split(".")
        try:
            parsed = tomllib.loads(f"v = {val}")["v"]
        except tomllib.TOMLDecodeError:
            parsed = val  # bare string, e.g. train.name=smoke
        raw.setdefault(section, {})[name] = parsed
    cfg = Config()
    for section, values in raw.items():
        obj = getattr(cfg, section)
        known = {f.name: f.type for f in fields(obj)}
        for k, v in values.items():
            if k not in known:
                raise KeyError(f"unknown config key {section}.{k}")
            setattr(obj, k, type(getattr(obj, k))(v))
    return cfg


def lr_at(step: int, total_steps: int, t: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * lr."""
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    progress = min((step - t.warmup_steps) / max(total_steps - t.warmup_steps, 1), 1.0)
    return t.lr * (t.min_lr_ratio + (1 - t.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: Transformer, t: TrainConfig) -> torch.optim.AdamW:
    # Weight decay on matrices (incl. the tied embedding); none on RMSNorm gains.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [dict(params=decay, weight_decay=t.weight_decay), dict(params=no_decay, weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=t.lr, betas=(t.beta1, t.beta2), eps=t.eps, fused=True)


def save_checkpoint(path: Path, model, opt, step: int, cfg: Config):
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step, config=asdict(cfg)), tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never leaves a corrupt checkpoint


def main():
    cfg = load_config(sys.argv[1], sys.argv[2:])
    t, mcfg = cfg.train, cfg.model
    rank, world, device = dist.setup()
    master = rank == 0
    run_dir = Path(t.out_dir) / t.name
    if master:
        run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(t.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = Transformer(mcfg).to(device)
    params = list(model.parameters())
    grads = dist.GradBuffer(params)
    opt = make_optimizer(model, t)

    loader = TrainLoader(t.train_bin, t.seq_len, t.micro_batch, t.grad_accum, rank, world, t.seed, device)
    total_steps = t.total_tokens // loader.tokens_per_step
    last_step = min(total_steps, t.max_steps) if t.max_steps > 0 else total_steps
    val_data = TokenFile(t.val_bin, t.seq_len)
    token_bytes = torch.from_numpy(token_byte_lengths(t.tokenizer)).to(device)

    step = 0
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        if master:
            print(f"resumed from {ckpt_path} at step {step}")
    dist.broadcast_params(model)

    fwd = torch.compile(model) if t.compile else model
    flops_per_token = model.flops_per_token(t.seq_len)

    if master:
        train_meta = load_meta(t.train_bin)
        print(json.dumps(asdict(cfg), indent=2))
        print(f"params: {model.num_params() / 1e6:.2f}M total, {model.num_params(True) / 1e6:.2f}M non-embedding")
        print(f"tokens/step: {loader.tokens_per_step:,}  steps: {total_steps:,}  "
              f"train tokens: {total_steps * loader.tokens_per_step / 1e9:.2f}B "
              f"({total_steps * loader.tokens_per_step / train_meta['n_tokens']:.2f} epochs)")
        with open(run_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        log_file = open(run_dir / "log.jsonl", "a")

    def log(record: dict):
        if master:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

    def run_eval():
        torch.cuda.synchronize()
        t0 = time.time()
        r = eval_loss(fwd, val_data, token_bytes, t.eval_tokens, t.micro_batch, rank, world, device)
        log(dict(kind="eval", step=step, tokens=step * loader.tokens_per_step, val_loss=r["loss"],
                 val_bpb=r["bpb"], eval_tokens=r["tokens"], eval_time=time.time() - t0))

    torch.cuda.reset_peak_memory_stats()
    t_last = time.time()
    while step < last_step:
        if step % t.eval_interval == 0:
            run_eval()
            t_last = time.time()

        lr = lr_at(step, total_steps, t)
        for g in opt.param_groups:
            g["lr"] = lr
        grads.zero()
        loss_sum = torch.zeros((), device=device)
        for micro in range(t.grad_accum):
            x, y = loader.get(step, micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd(x, y)
            # Mean over micro-batches; the all_reduce then averages over ranks.
            (loss / t.grad_accum).backward()
            loss_sum += loss.detach()
        if step == 0:
            grads.check_views()
        grads.all_reduce_mean()
        grad_norm = grads.clip_(t.grad_clip)
        opt.step()
        step += 1

        if step % t.log_interval == 0 or step == last_step:
            train_loss = dist.all_reduce_sum(loss_sum / t.grad_accum).item() / world
            torch.cuda.synchronize()
            dt = (time.time() - t_last) / (t.log_interval if step % t.log_interval == 0 else step % t.log_interval)
            t_last = time.time()
            tok_s = loader.tokens_per_step / dt
            log(dict(kind="train", step=step, tokens=step * loader.tokens_per_step, loss=train_loss, lr=lr,
                     grad_norm=grad_norm, step_time=dt, tok_per_s=tok_s,
                     mfu=tok_s * flops_per_token / (t.peak_flops * world),
                     mem_gb=torch.cuda.max_memory_allocated() / 2**30))

        if master and (step % t.ckpt_interval == 0 or step == last_step):
            save_checkpoint(ckpt_path, model, opt, step, cfg)

    run_eval()
    if master:
        log_file.close()
    dist.teardown()


if __name__ == "__main__":
    main()
