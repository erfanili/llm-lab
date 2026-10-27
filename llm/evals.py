"""Zero-shot log-likelihood benchmarks for base models.

Every task reduces to scoring log P(continuation | <eot> + context). Per item we keep the raw
choice log-likelihoods, so metrics can be recomputed and two checkpoints compared *paired*
(same items), which has far lower variance than comparing two independent accuracies.

Metrics per multiple-choice task:
  acc       argmax of summed log-likelihood
  acc_norm  argmax of log-likelihood per UTF-8 byte of the continuation (removes length bias)
  p_gold    P(gold) / sum_c P(choice c) from the raw log-likelihoods: a continuous score
            that still moves when accuracy is stuck near chance
Each with a 95% CI from the per-item standard error.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Metric reported as the headline per task (lm-eval-harness conventions for small models).
PRIMARY = dict(hellaswag="acc_norm", arc_easy="acc_norm", arc_challenge="acc_norm", piqa="acc_norm",
               sciq="acc_norm", openbookqa="acc_norm", winogrande="acc", blimp="acc", lambada="acc")
DEFAULT_TASKS = list(PRIMARY)


@torch.no_grad()
def score(model, requests: list[tuple[list[int], list[int]]], device, max_len: int,
          batch_tokens: int = 65536) -> list[tuple[float, bool]]:
    """(context_ids, continuation_ids) -> (sum log P(continuation), continuation is the greedy decode).

    Sequences are length-sorted and right-padded (causal attention never looks at the padding),
    and the LM head runs only on continuation positions.
    """
    seqs = []
    for ctx, cont in requests:
        full = (ctx + cont)[-(max_len + 1):]  # left-truncate the context if needed
        seqs.append((full, min(len(cont), len(full) - 1)))
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i][0]))
    out = [None] * len(seqs)
    i = 0
    while i < len(order):
        T = len(seqs[order[i]][0]) - 1
        batch = order[i : i + max(1, batch_tokens // T)]
        i += len(batch)
        x = torch.zeros(len(batch), T + 1, dtype=torch.long)
        for r, j in enumerate(batch):
            x[r, : len(seqs[j][0])] = torch.tensor(seqs[j][0])
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            h = model.hidden(x[:, :-1])
        rows, cols = [], []
        for r, j in enumerate(batch):
            n_full, n_cont = len(seqs[j][0]), seqs[j][1]
            rows += [r] * n_cont
            cols += list(range(n_full - 1 - n_cont, n_full - 1))  # positions predicting the continuation
        rows_t, cols_t = torch.tensor(rows, device=device), torch.tensor(cols, device=device)
        logits = model.lm_head(h[rows_t, cols_t]).float()
        logp = F.log_softmax(logits, -1)
        tgt = x[rows_t, cols_t + 1]
        tok_lp = logp.gather(1, tgt[:, None])[:, 0]
        greedy = logits.argmax(-1) == tgt
        start = 0
        for j in batch:
            n = seqs[j][1]
            out[j] = (tok_lp[start : start + n].sum().item(), bool(greedy[start : start + n].all()))
            start += n
    return out


def mean_ci(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0
    return dict(mean=float(x.mean()), ci95=float(1.96 * se), n=len(x))


def run_task(model, tok: Tokenizer, items: list[dict], device, max_len: int, limit: int | None = None):
    eot = tok.token_to_id("<|endoftext|>")
    items = items[:limit] if limit else items
    enc = lambda s: tok.encode(s).ids if s else []

    if items[0]["task"] == "lambada":
        res = score(model, [([eot] + enc(it["context"]), enc(it["continuation"])) for it in items], device, max_len)
        n_tok = [len(enc(it["continuation"])) for it in items]
        per_item = [dict(id=it["id"], correct=g, ll=ll) for it, (ll, g) in zip(items, res)]
        metrics = dict(acc=mean_ci([r["correct"] for r in per_item]),
                       ppl=dict(mean=math.exp(-sum(r["ll"] for r in per_item) / sum(n_tok))))
        return metrics, per_item

    flat = [([eot] + enc(c["context"]), enc(c["continuation"])) for it in items for c in it["choices"]]
    res = iter(score(model, flat, device, max_len))
    per_item = []
    for it in items:
        ll = np.array([next(res)[0] for _ in it["choices"]])
        nbytes = np.array([max(len(c["continuation"].encode("utf-8")), 1) for c in it["choices"]])
        norm = ll / nbytes
        p = np.exp(ll - ll.max())  # the model's own distribution restricted to the choices
        p /= p.sum()
        g = it["gold"]
        per_item.append(dict(id=it["id"], correct=bool(ll.argmax() == g), correct_norm=bool(norm.argmax() == g),
                             p_gold=float(p[g]), ll=ll.tolist(), n_choices=len(ll), **it.get("meta", {})))
    metrics = dict(acc=mean_ci([r["correct"] for r in per_item]),
                   acc_norm=mean_ci([r["correct_norm"] for r in per_item]),
                   p_gold=mean_ci([r["p_gold"] for r in per_item]),
                   chance=float(np.mean([1 / r["n_choices"] for r in per_item])))
    if items[0]["task"] == "blimp":  # per-phenomenon breakdown
        by = {}
        for r in per_item:
            by.setdefault(r["phenomenon"], []).append(r["correct"])
        metrics["by_phenomenon"] = {k: float(np.mean(v)) for k, v in sorted(by.items())}
    return metrics, per_item


def load_items(eval_dir: str, task: str) -> list[dict]:
    with open(Path(eval_dir) / f"{task}.jsonl") as f:
        return [json.loads(line) for line in f]


def paired_diff(a: list[dict], b: list[dict], key: str) -> dict:
    """b - a on the same items, with a paired 95% CI. Items are matched by id."""
    bi = {r["id"]: r for r in b}
    d = np.array([float(bi[r["id"]][key]) - float(r[key]) for r in a if r["id"] in bi])
    out = mean_ci(d)
    se = out["ci95"] / 1.96
    out["z"] = out["mean"] / se if se > 0 else 0.0
    return out
