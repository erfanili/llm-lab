"""RLHF with PPO (InstructGPT-style), written out step by step.

    python rlhf.py configs/rlhf_50m.toml [section.key=value ...]

Models
  policy  pi      trainable, initialised from SFT
  ref     pi_ref  frozen copy of SFT (KL anchor)
  reward  r       frozen reward model
  value   V       trainable, initialised from the reward model (backbone + head)

Each iteration
  1. sample responses y ~ pi(.|x) for a batch of prompts (temperature 1: on-policy)
  2. per-token reward  r_t = -kl_coef * (log pi(y_t) - log pi_ref(y_t)),  plus the normalised
     RM score on the final token (and a penalty if the response hit max length without ending)
  3. GAE advantages / returns from V
  4. a few epochs of clipped-surrogate policy updates and clipped value regression

Every eval_interval iterations, on a fixed set of held-out prompts sampled with a fixed seed (so
successive checkpoints are compared on the same randomness): RM score and KL-to-ref with 95% CIs,
response length, EOS rate; a policy snapshot; greedy generations on the inspection prompts; and
optionally a Claude-judged win rate against the SFT model.
"""

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from llm.chat import ChatFormat
from llm.checkpoint import load_model, save_weights
from llm.config import load_config
from llm.evals import mean_ci
from llm.generate import generate
from llm.optim import make_adamw
from llm.posttrain import Logger, generate_on_prompts, load_inspect_prompts, load_scalar_model, pad_right, token_logprobs

D = "/data/erfan/llm-lab"
REPO = Path(__file__).resolve().parent


@dataclass
class PPOConfig:
    name: str = "rlhf_50m"
    sft_ckpt: str = f"{D}/runs/sft_50m/ckpt_final.pt"
    rm_ckpt: str = f"{D}/runs/rm_50m/ckpt_final.pt"
    out_dir: str = f"{D}/runs"
    train_prompts: str = f"{D}/posttrain/rl_prompts_train.jsonl"
    val_prompts: str = f"{D}/posttrain/rl_prompts_val.jsonl"
    tokenizer: str = f"{D}/tokenizer/bpe32k.json"
    inspect_prompts: str = str(REPO / "data/prompts/inspect_v1.jsonl")
    seed: int = 0
    iterations: int = 500
    rollout_batch: int = 64
    max_new: int = 128
    ppo_epochs: int = 2
    minibatch: int = 16
    policy_lr: float = 1e-5
    value_lr: float = 5e-5
    kl_coef: float = 0.05
    clip: float = 0.2
    value_clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    missing_eos_penalty: float = -1.0
    grad_clip: float = 1.0
    eval_interval: int = 25
    eval_prompts: int = 512
    eval_seed: int = 12345
    judge_interval: int = 0  # >0: Claude win rate vs SFT on inspection prompts every N iterations (costs $)
    judge_cache: str = f"{D}/judge_cache.jsonl"


@dataclass
class Config:
    ppo: PPOConfig = field(default_factory=PPOConfig)


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1)


def masked_whiten(x, m, eps=1e-8):
    mu = masked_mean(x, m)
    var = masked_mean((x - mu) ** 2, m)
    return (x - mu) * torch.rsqrt(var + eps) * m


def policy_logprobs_entropy(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x[:, :-1])
    logp_all = logits.float().log_softmax(-1)
    lp = logp_all.gather(-1, x[:, 1:, None])[..., 0]
    ent = -(logp_all.exp() * logp_all).sum(-1)
    return lp, ent


class Rollout:
    """A batch of prompt+response sequences, right-padded, with per-position response mask.
    Position t in the (B, T-1) arrays refers to predicting token x[:, t+1]."""

    def __init__(self, chat, prompts_ids, outs, device):
        self.resp = [o["tokens"] + ([chat.eot] if o["finish"] == "stop" else []) for o in outs]
        self.finished = torch.tensor([o["finish"] == "stop" for o in outs], device=device)
        seqs = [p + r for p, r in zip(prompts_ids, self.resp)]
        self.x, self.lengths = pad_right(seqs, device=device)
        T = self.x.shape[1]
        self.mask = torch.zeros(len(seqs), T - 1, device=device)
        for i, (p, r) in enumerate(zip(prompts_ids, self.resp)):
            self.mask[i, len(p) - 1 : len(p) - 1 + len(r)] = 1
        self.last = self.lengths - 2  # index of the final response token in the (B, T-1) arrays


@torch.no_grad()
def score_rollout(ro, policy, ref, rm, value, c, rm_mean, rm_std, chunk=16):
    old_lp, ref_lp, vals, scores = [], [], [], []
    for i in range(0, len(ro.x), chunk):
        xb, lb = ro.x[i : i + chunk], ro.lengths[i : i + chunk]
        old_lp.append(token_logprobs(policy, xb))
        ref_lp.append(token_logprobs(ref, xb))
        vals.append(value.per_token(xb)[:, :-1])
        scores.append(rm.score(xb, lb))
    ro.old_lp, ro.ref_lp, ro.values = torch.cat(old_lp), torch.cat(ref_lp), torch.cat(vals)
    ro.score_raw = torch.cat(scores)
    ro.score = (ro.score_raw - rm_mean) / rm_std + c.missing_eos_penalty * (~ro.finished).float()
    ro.kl = (ro.old_lp - ro.ref_lp) * ro.mask
    rewards = -c.kl_coef * ro.kl
    rewards[torch.arange(len(ro.x)), ro.last] += ro.score
    adv = gae(rewards, ro.values, ro.mask, c.gamma, c.lam)
    ro.returns = (adv + ro.values) * ro.mask
    ro.adv = masked_whiten(adv, ro.mask)


def gae(rewards, values, mask, gamma, lam):
    """Generalised advantage estimation over (B, T) arrays; each row's actions are the contiguous
    span where mask == 1, and the step after the last action is terminal (value 0)."""
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(len(rewards), device=rewards.device)
    T = rewards.shape[1]
    for t in reversed(range(T)):
        next_m = mask[:, t + 1] if t + 1 < T else torch.zeros_like(last_gae)
        next_v = values[:, t + 1] * next_m if t + 1 < T else torch.zeros_like(last_gae)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last_gae = delta + gamma * lam * next_m * last_gae
        adv[:, t] = last_gae * mask[:, t]
    return adv


def ppo_update(ro, policy, value, popt, vopt, c, rng):
    stats = []
    B = len(ro.x)
    for _ in range(c.ppo_epochs):
        for idx in np.array_split(rng.permutation(B), max(B // c.minibatch, 1)):
            idx = torch.as_tensor(idx, device=ro.x.device)
            x, m = ro.x[idx], ro.mask[idx]
            Tm = int(ro.lengths[idx].max())  # trim padding shared by this minibatch
            x, m = x[:, :Tm], m[:, : Tm - 1]
            old_lp, adv, ret, old_v = (t[idx][:, : Tm - 1] for t in (ro.old_lp, ro.adv, ro.returns, ro.values))

            lp, ent = policy_logprobs_entropy(policy, x)
            log_ratio = lp - old_lp
            ratio = log_ratio.exp()
            pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - c.clip, 1 + c.clip))
            pg_loss = masked_mean(pg, m)
            popt.zero_grad(set_to_none=True)
            pg_loss.backward()
            pgn = torch.nn.utils.clip_grad_norm_(policy.parameters(), c.grad_clip)
            popt.step()

            v = value.per_token(x)[:, :-1]
            v_clipped = old_v + (v - old_v).clamp(-c.value_clip, c.value_clip)
            vf_loss = 0.5 * masked_mean(torch.max((v - ret) ** 2, (v_clipped - ret) ** 2), m)
            vopt.zero_grad(set_to_none=True)
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), c.grad_clip)
            vopt.step()

            with torch.no_grad():
                stats.append(dict(pg_loss=pg_loss.item(), vf_loss=vf_loss.item(), entropy=masked_mean(ent, m).item(),
                                  approx_kl=masked_mean(ratio - 1 - log_ratio, m).item(),
                                  clipfrac=masked_mean(((ratio - 1).abs() > c.clip).float(), m).item(),
                                  policy_grad_norm=pgn.item()))
    return {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}


def main():
    cfg = load_config(Config, sys.argv[1], sys.argv[2:])
    c = cfg.ppo
    device = torch.device("cuda")
    torch.manual_seed(c.seed)
    rng = np.random.default_rng(c.seed)
    run_dir = Path(c.out_dir) / c.name
    log = Logger(run_dir)
    (run_dir / "inspect").mkdir(exist_ok=True)
    chat = ChatFormat(Tokenizer.from_file(c.tokenizer))

    policy, sft_ck = load_model(c.sft_ckpt, device)
    ref, _ = load_model(c.sft_ckpt, device)
    for p in ref.parameters():
        p.requires_grad_(False)
    rm = load_scalar_model(c.rm_ckpt, device, "reward_head").eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    value = load_scalar_model(c.rm_ckpt, device, "value_head", init_head_from="reward_head")
    rm_eval = torch.load(c.rm_ckpt, map_location="cpu", weights_only=False).get("eval", {})
    rm_mean, rm_std = rm_eval.get("val_reward_mean", 0.0), rm_eval.get("val_reward_std", 1.0)
    popt = make_adamw(policy, c.policy_lr, 0.0)
    vopt = make_adamw(value, c.value_lr, 0.0)
    model_cfg = sft_ck["config"]["model"]

    train_prompts = [json.loads(l) for l in open(c.train_prompts)]
    val_prompts = [json.loads(l) for l in open(c.val_prompts)][: c.eval_prompts]
    val_ids = [chat.prompt(p["messages"]) for p in val_prompts]
    inspect = load_inspect_prompts(c.inspect_prompts)
    order = rng.permutation(len(train_prompts))
    log(dict(kind="setup", n_train_prompts=len(train_prompts), n_val_prompts=len(val_ids), rm_mean=rm_mean, rm_std=rm_std))

    @torch.no_grad()
    def evaluate(it):
        policy.eval()
        outs = []
        for i in range(0, len(val_ids), c.rollout_batch):
            outs += generate(policy, val_ids[i : i + c.rollout_batch], c.max_new, temperature=1.0,
                             stop_ids=(chat.eot,), seed=c.eval_seed + i)
        ro = Rollout(chat, val_ids, outs, device)
        score_rollout(ro, policy, ref, rm, value, c, rm_mean, rm_std)
        score, kl = mean_ci(ro.score_raw.tolist()), mean_ci(ro.kl.sum(1).tolist())
        r = dict(kind="eval", step=it, val_rm_score=score["mean"], val_rm_score_ci95=score["ci95"],
                 val_kl=kl["mean"], val_kl_ci95=kl["ci95"],
                 val_len=float(np.mean([len(x) for x in ro.resp])), val_eos_rate=ro.finished.float().mean().item())
        by_src = {}
        for p, s in zip(val_prompts, ro.score_raw.tolist()):
            by_src.setdefault(p["source"], []).append(s)
        r.update({f"val_rm_score/{k}": float(np.mean(v)) for k, v in sorted(by_src.items())})
        ckpt = run_dir / f"ckpt_step{it}.pt"
        save_weights(ckpt, policy, dict(model=model_cfg, ppo=asdict(c)), step=it, eval=r)
        gen_path = run_dir / "inspect" / f"step{it}.jsonl"
        generate_on_prompts(policy, chat, inspect, gen_path, c.max_new, dict(ckpt=str(ckpt), step=it))
        with open(run_dir / "val_samples" / f"step{it}.jsonl", "w") as f:
            for p, resp, s in zip(val_prompts, ro.resp, ro.score_raw.tolist()):
                f.write(json.dumps(dict(prompt=p["messages"][-1]["content"], response=chat.decode(resp), rm_score=s)) + "\n")
        if c.judge_interval > 0 and it > 0 and it % c.judge_interval == 0:
            from llm.judge import Judge
            base = {x["id"]: x for x in map(json.loads, open(run_dir / "inspect" / "step0.jsonl"))}
            cur = {x["id"]: x for x in map(json.loads, open(gen_path))}
            items = [dict(id=i, prompt=base[i]["prompt"], a=base[i]["response"], b=cur[i]["response"],
                          reference=p.get("expect")) for p in inspect if (i := p["id"]) in cur]
            summary, per_item = Judge(c.judge_cache).win_rate(items)
            json.dump(dict(summary=summary, items=per_item), open(run_dir / "inspect" / f"judge_step{it}.json", "w"), indent=2)
            if summary["win_rate_b"]:
                r.update(judge_win_rate_vs_sft=summary["win_rate_b"]["mean"], judge_ci95=summary["win_rate_b"]["ci95"])
        log(r)
        policy.train()

    (run_dir / "val_samples").mkdir(exist_ok=True)
    for it in range(c.iterations):
        if it % c.eval_interval == 0:
            evaluate(it)
        t0 = time.time()
        batch = [train_prompts[order[(it * c.rollout_batch + j) % len(order)]] for j in range(c.rollout_batch)]
        prompt_ids = [chat.prompt(p["messages"]) for p in batch]
        policy.eval()
        outs = generate(policy, prompt_ids, c.max_new, temperature=1.0, stop_ids=(chat.eot,), seed=c.seed * 1_000_003 + it)
        policy.train()
        ro = Rollout(chat, prompt_ids, outs, device)
        score_rollout(ro, policy, ref, rm, value, c, rm_mean, rm_std)
        t_roll = time.time() - t0
        stats = ppo_update(ro, policy, value, popt, vopt, c, rng)
        v, ret, m = ro.values, ro.returns, ro.mask.bool()
        explained_var = 1 - torch.var(ret[m] - v[m]) / torch.var(ret[m]).clamp(min=1e-8)
        log(dict(kind="train", step=it + 1, rm_score=ro.score_raw.mean().item(), reward=ro.score.mean().item(),
                 kl=ro.kl.sum(1).mean().item(), resp_len=float(np.mean([len(x) for x in ro.resp])),
                 eos_rate=ro.finished.float().mean().item(), value_explained_var=explained_var.item(), **stats,
                 t_rollout=t_roll, t_step=time.time() - t0))
    evaluate(c.iterations)


if __name__ == "__main__":
    main()
