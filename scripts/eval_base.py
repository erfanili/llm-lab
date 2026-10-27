"""Evaluate a base-model checkpoint: held-out loss/bpb + zero-shot benchmarks.

    python scripts/eval_base.py --ckpt /data/erfan/llm-lab/runs/50m/ckpt.pt

Writes <out>/summary.json and <out>/<task>.items.jsonl (per-item, for paired comparisons).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llm.checkpoint import load_model
from llm.data import TokenFile, token_byte_lengths
from llm.evals import DEFAULT_TASKS, PRIMARY, load_items, run_task
from llm.evaluate import eval_loss

D = "/data/erfan/llm-lab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="first N items per task (debugging only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-dir", default=f"{D}/evals")
    ap.add_argument("--tokenizer", default=f"{D}/tokenizer/bpe32k.json")
    ap.add_argument("--val-bin", default=f"{D}/tokens/val.bin")
    ap.add_argument("--val-tokens", type=int, default=10 * 2**20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ck = load_model(args.ckpt, device)
    step = ck.get("step", "na")
    out = Path(args.out or Path(args.ckpt).parent / "evals" / f"{Path(args.ckpt).stem}_step{step}")
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    max_len = model.cfg.max_seq_len

    summary = dict(ckpt=str(args.ckpt), step=step, tasks={})
    t0 = time.time()
    val = TokenFile(args.val_bin, max_len)
    tb = torch.from_numpy(token_byte_lengths(args.tokenizer)).to(device)
    summary["val"] = eval_loss(model, val, tb, args.val_tokens, 16, 0, 1, device)
    print(f"val: loss={summary['val']['loss']:.4f} bpb={summary['val']['bpb']:.4f}")

    for task in args.tasks:
        t1 = time.time()
        metrics, items = run_task(model, tok, load_items(args.eval_dir, task), device, max_len, args.limit)
        summary["tasks"][task] = metrics
        with open(out / f"{task}.items.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
        m = metrics[PRIMARY[task]]
        extra = f"  chance={metrics['chance']:.3f}  p_gold={metrics['p_gold']['mean']:.4f}" if "chance" in metrics else \
                f"  ppl={metrics['ppl']['mean']:.1f}"
        print(f"{task:14s} {PRIMARY[task]:8s} {m['mean']:.4f} ± {m['ci95']:.4f}  (n={m['n']}){extra}  [{time.time() - t1:.0f}s]")

    summary["eval_seconds"] = time.time() - t0
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
