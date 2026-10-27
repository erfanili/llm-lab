"""Paired comparison of two eval_base.py output dirs (B - A on identical items).

    python scripts/eval_compare.py runs/50m/evals/ckpt_step3814 runs/50m_lr4e-3/evals/ckpt_step3814

Paired differences cancel per-item difficulty, so the CI on the *difference* is much tighter
than what you'd get by comparing the two accuracies' independent CIs.
"""

import argparse
import json
from pathlib import Path

from llm.evals import PRIMARY, paired_diff


def load(d: Path, task: str):
    with open(d / f"{task}.items.jsonl") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)
    sa, sb = json.load(open(a / "summary.json")), json.load(open(b / "summary.json"))
    print(f"A = {sa['model']}\nB = {sb['model']}\n")
    va, vb = sa["val"]["doc"]["bpb"], sb["val"]["doc"]["bpb"]
    print(f"val doc_bpb: A={va:.4f}  B={vb:.4f}  B-A={vb - va:+.4f}\n")
    print(f"{'task':14s} {'metric':9s} {'A':>7s} {'B':>7s} {'B-A':>8s} {'±95%':>7s} {'z':>6s}   {'p_gold B-A':>10s} {'±95%':>7s}")
    for task in sa["tasks"]:
        if task not in sb["tasks"]:
            continue
        ia, ib = load(a, task), load(b, task)
        key = {"acc": "correct", "acc_norm": "correct_norm"}[PRIMARY[task]]
        d = paired_diff(ia, ib, key)
        ma, mb = sa["tasks"][task][PRIMARY[task]]["mean"], sb["tasks"][task][PRIMARY[task]]["mean"]
        line = f"{task:14s} {PRIMARY[task]:9s} {ma:7.4f} {mb:7.4f} {d['mean']:+8.4f} {d['ci95']:7.4f} {d['z']:6.2f}"
        if "p_gold" in ia[0]:
            pg = paired_diff(ia, ib, "p_gold")
            line += f"   {pg['mean']:+10.4f} {pg['ci95']:7.4f}"
        print(line)


if __name__ == "__main__":
    main()
