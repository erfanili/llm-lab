"""Render the published-results tables as markdown, optionally with our checkpoints added as rows.

    python scripts/reference_table.py --out docs/reference_results.md
    python scripts/reference_table.py --ours 50m=/data/erfan/llm-lab/runs/50m/evals/ckpt_step3814 --out docs/reference_results.md

Our rows use, per source, the metric that best matches that source's protocol (`our_metric` in the JSON).
Numbers are only comparable within one table.
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def our_value(summary: dict, column: str, metric: str) -> str:
    tasks = summary["tasks"]
    if column == "arc_avg":
        if "arc_easy" in tasks and "arc_challenge" in tasks:
            return f"{50 * (tasks['arc_easy']['acc_norm']['mean'] + tasks['arc_challenge']['acc_norm']['mean']):.1f}"
        return "—"
    if column not in tasks:
        return "—"
    m = tasks[column][metric]
    return f"{100 * m['mean']:.1f} ±{100 * m['ci95']:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(REPO / "data/reference/reported_results.json"))
    ap.add_argument("--ours", nargs="*", default=[], help="name=eval_dir (output of scripts/eval_base.py)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    ours = []
    for spec in args.ours:
        name, d = spec.split("=", 1)
        s = json.load(open(Path(d) / "summary.json"))
        ours.append((name, s))

    lines = ["# Published results for open base models (≤ ~1.6B)", "", ref["_about"], ""]
    for t in ref["tables"]:
        cols = t["columns"]
        lines += [f"## {t['title']}", "", f"Source: [{t['source']}]({t['url']})", "", f"Protocol: {t['protocol']}", ""]
        lines.append("| Model | Params | Train tokens | " + " | ".join(cols) + " |")
        lines.append("|---|---|---|" + "---|" * len(cols))
        for r in t["rows"]:
            vals = [f"{r[c]:.1f}" if isinstance(r.get(c), (int, float)) else "—" for c in cols]
            lines.append(f"| {r['model']} | {r['params'] or '—'} | {r['tokens'] or '—'} | " + " | ".join(vals) + " |")
        for name, s in ours:
            vals = [our_value(s, c, t["our_metric"].get(c, "acc")) if c in t["our_metric"] else "n/c" for c in cols]
            lines.append(f"| **ours: {name}** | | | " + " | ".join(vals) + " |")
        if ours:
            lines.append("")
            lines.append("Our metric per column: " + ", ".join(f"{k}={v}" for k, v in t["our_metric"].items())
                         + ". n/c = not comparable under this source's protocol.")
        lines.append("")
    text = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
