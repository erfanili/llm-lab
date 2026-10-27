"""Judge generations (from inspect_samples.py or the post-training loops) with Claude.

    # win rate of B over A on the same prompts, overall and per category
    python scripts/judge.py pairwise A/generations.jsonl B/generations.jsonl --out judge_ab.json
    # absolute 1-10 grades for one model
    python scripts/judge.py grade B/generations.jsonl --out grades_b.json

Uses chat-mode, sample-0 (greedy) generations by default.
"""

import argparse
import json
from pathlib import Path

from llm.evals import mean_ci
from llm.judge import MODEL, Judge

REPO = Path(__file__).resolve().parent.parent


def load(path, mode, sample):
    return {r["id"]: r for r in map(json.loads, open(path)) if r["mode"] == mode and r["sample"] == sample}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairwise", "grade"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="chat")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--prompts", default=str(REPO / "data/prompts/inspect_v1.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--cache", default="/data/erfan/llm-lab/judge_cache.jsonl")
    args = ap.parse_args()

    refs = {p["id"]: p.get("expect") for p in map(json.loads, open(args.prompts))}
    judge = Judge(args.cache, args.model, args.effort)
    runs = [load(f, args.mode, args.sample) for f in args.files]
    ids = [i for i in runs[0] if all(i in r for r in runs)]

    if args.cmd == "pairwise":
        a, b = runs[0], runs[1]
        items = [dict(id=i, prompt=a[i]["prompt"], a=a[i]["response"], b=b[i]["response"], reference=refs.get(i)) for i in ids]
        summary, per_item = judge.win_rate(items)
        cats = {}
        for r in per_item:
            if r["outcome"] is not None:
                cats.setdefault(a[r["id"]]["category"], []).append({"B": 1.0, "A": 0.0, "tie": 0.5}[r["outcome"]])
        summary["by_category"] = {k: mean_ci(v) for k, v in sorted(cats.items())}
        summary.update(a=args.files[0], b=args.files[1])
    else:
        r = runs[0]
        items = [dict(id=i, prompt=r[i]["prompt"], response=r[i]["response"], reference=refs.get(i)) for i in ids]
        summary, per_item = judge.grade_all(items)
        summary["file"] = args.files[0]
    summary.update(judge_model=args.model, effort=args.effort, n=len(ids))
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, items=per_item), f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
