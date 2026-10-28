"""Phase-0 chat evaluation: verifiable suite + degeneration metrics + pairwise judge vs a fixed reference.

    # generate (fp32, chat template) on both sets and compute automatic metrics
    python scripts/eval_chat.py run --ckpt runs/sft_50m/ckpt_final.pt --out runs/sft_50m/chat_eval/final
    # re-apply current checkers to saved generations
    python scripts/eval_chat.py rescore DIR [DIR ...]
    # paired B - A on every automatic metric (same items, same seeds)
    python scripts/eval_chat.py compare A_dir B_dir
    # Claude pairwise win rate of B over A on judge_v1 (both orders; greedy + sampled responses)
    python scripts/eval_chat.py judge A_dir B_dir --out judge.json

Each item is generated greedily (sample 0) and with --n-samples seeded samples (T=0.7, top-p 0.9).
Per-item scores average the samples, so the sampled metrics have less variance than a single draw;
greedy is reported separately because it is what a deterministic deployment would show.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from llm.chat import ChatFormat
from llm.chat_evals import check, degeneration
from llm.checkpoint import load_model
from llm.evals import mean_ci, paired_diff
from llm.generate import generate

D = "/data/erfan/llm-lab"
REPO = Path(__file__).resolve().parent.parent
SETS = dict(suite=(REPO / "data/prompts/chat_suite_v1.jsonl", 96), judge=(REPO / "data/prompts/judge_v1.jsonl", 256))


def item_rows(gens: list[dict]) -> dict[str, list[dict]]:
    """generations -> per set, one row per item with greedy and sample-averaged metrics."""
    by = {}
    for g in gens:
        by.setdefault((g["set"], g["id"]), []).append(g)
    out = {}
    for (s, i), gs in by.items():
        gs.sort(key=lambda g: g["sample"])
        row = dict(id=i, category=gs[0]["category"])
        for tag, sel in [("greedy", gs[:1]), ("sampled", gs[1:] or gs[:1])]:
            m = dict(eot=[g["finish"] == "stop" for g in sel], n_tokens=[g["n_tokens"] for g in sel],
                     **{k: [degeneration(g["response"])[k] for g in sel] for k in ("rep4", "distinct2")})
            if "correct" in gs[0]:
                m["correct"] = [g["correct"] for g in sel]
            row.update({f"{tag}_{k}": float(np.mean(v)) for k, v in m.items()})
        out.setdefault(s, []).append(row)
    return out


def summarize(rows: dict[str, list[dict]]) -> dict:
    out = {}
    for s, rs in rows.items():
        keys = [k for k in rs[0] if k not in ("id", "category")]
        out[s] = dict(overall={k: mean_ci([r[k] for r in rs]) for k in keys})
        if s == "suite":
            cats = sorted({r["category"] for r in rs})
            out[s]["by_category"] = {c: {k: mean_ci([r[k] for r in rs if r["category"] == c])
                                         for k in ("greedy_correct", "sampled_correct")} for c in cats}
    return out


def cmd_run(args):
    device = torch.device(args.device)
    model, ck = load_model(args.ckpt, device)
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)
    chat = ChatFormat(tok)
    settings = [dict(temperature=0.0)] + [dict(temperature=0.7, top_p=0.9, seed=args.seed + s) for s in range(args.n_samples)]
    gens = []
    for s, (path, max_new) in SETS.items():
        items = [json.loads(l) for l in open(path)]
        inputs = [chat.prompt([dict(role="user", content=it["prompt"])]) for it in items]
        for k, st in enumerate(settings):
            outs = []
            for b in range(0, len(items), args.batch):
                outs += generate(model, inputs[b:b + args.batch], max_new, stop_ids=(chat.eot,), amp=False, **st)
            for it, o in zip(items, outs):
                r = chat.decode(o["tokens"])
                g = dict(set=s, id=it["id"], category=it["category"], sample=k, response=r,
                         n_tokens=len(o["tokens"]), finish=o["finish"])
                if s == "suite":
                    g["correct"] = check(it, r)
                gens.append(g)
            print(f"{s} sample {k}: {len(outs)} generations", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(ckpt=str(args.ckpt), step=ck.get("step"), n_samples=args.n_samples, seed=args.seed, dtype="fp32")
    write_results(out, gens, meta)


def write_results(out: Path, gens: list[dict], meta: dict):
    with open(out / "generations.jsonl", "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    rows = item_rows(gens)
    for s, rs in rows.items():
        with open(out / f"{s}.items.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    summary = dict(**meta, **summarize(rows))
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print_summary(summary)


def cmd_rescore(args):
    """Re-apply the current checkers to saved generations (after a checker fix; no regeneration)."""
    items = {it["id"]: it for it in map(json.loads, open(SETS["suite"][0]))}
    for d in map(Path, args.dirs):
        gens = [json.loads(l) for l in open(d / "generations.jsonl")]
        for g in gens:
            if g["set"] == "suite":
                g["correct"] = check(items[g["id"]], g["response"])
        old = json.load(open(d / "summary.json"))
        write_results(d, gens, {k: old[k] for k in ("ckpt", "step", "n_samples", "seed", "dtype")})


def print_summary(s):
    print(f"\n{s['ckpt']} (step {s['step']})")
    for name in ("suite", "judge"):
        o = s[name]["overall"]
        print(f"  [{name}] " + "  ".join(f"{k}={v['mean']:.3f}±{v['ci95']:.3f}" for k, v in o.items()))
    for c, v in s["suite"]["by_category"].items():
        print(f"    {c:12s} greedy {v['greedy_correct']['mean']:.3f}  sampled {v['sampled_correct']['mean']:.3f}"
              f"  (n={v['greedy_correct']['n']})")


def load_rows(d: Path, s: str) -> list[dict]:
    return [json.loads(l) for l in open(d / f"{s}.items.jsonl")]


def cmd_compare(args):
    a, b = Path(args.a), Path(args.b)
    print(f"A = {json.load(open(a / 'summary.json'))['ckpt']}\nB = {json.load(open(b / 'summary.json'))['ckpt']}")
    for s in SETS:
        ra, rb = load_rows(a, s), load_rows(b, s)
        print(f"\n[{s}] {'metric':18s} {'A':>7s} {'B':>7s} {'B-A':>8s} {'±95%':>7s} {'z':>6s}")
        for k in [k for k in ra[0] if k not in ("id", "category")]:
            d = paired_diff(ra, rb, k)
            print(f"[{s}] {k:18s} {np.mean([r[k] for r in ra]):7.3f} {np.mean([r[k] for r in rb]):7.3f} "
                  f"{d['mean']:+8.3f} {d['ci95']:7.3f} {d['z']:6.2f}")
        if s == "suite":
            for c in sorted({r["category"] for r in ra}):
                d = paired_diff([r for r in ra if r["category"] == c], [r for r in rb if r["category"] == c],
                                "sampled_correct")
                print(f"[{s}]   {c:16s} sampled_correct B-A {d['mean']:+.3f} ±{d['ci95']:.3f}")


def cmd_judge(args):
    from llm.judge import MODEL, Judge
    refs = {p["id"]: p for p in map(json.loads, open(SETS["judge"][0]))}
    load = lambda d: {(g["id"], g["sample"]): g for g in map(json.loads, open(Path(d) / "generations.jsonl"))
                      if g["set"] == "judge"}
    ga, gb = load(args.a), load(args.b)
    items = [dict(id=f"{i}#{k}", prompt=refs[i]["prompt"], a=ga[(i, k)]["response"], b=gb[(i, k)]["response"],
                  reference=refs[i].get("expect"))
             for (i, k) in ga if k in args.samples and (i, k) in gb]
    judge = Judge(args.cache, args.model, args.effort, workers=args.workers, backend=args.backend)
    summary, per_item = judge.win_rate(items)
    # per-prompt score = mean over judged samples, so the CI counts prompts (not prompt x sample) as independent
    per_prompt = {}
    for r in per_item:
        if r["outcome"] is not None:
            per_prompt.setdefault(r["id"].split("#")[0], []).append({"B": 1.0, "A": 0.0, "tie": 0.5}[r["outcome"]])
    pp = {i: float(np.mean(v)) for i, v in per_prompt.items()}
    summary["win_rate_b_by_prompt"] = mean_ci(list(pp.values()))
    summary["by_source"] = {src: mean_ci([v for i, v in pp.items() if refs[i]["source"] == src])
                            for src in sorted({refs[i]["source"] for i in pp})}
    summary["by_sample"] = {k: mean_ci([{"B": 1.0, "A": 0.0, "tie": 0.5}[r["outcome"]] for r in per_item
                                        if r["outcome"] is not None and r["id"].endswith(f"#{k}")]) for k in args.samples}
    summary.update(a=args.a, b=args.b, samples=args.samples, judge_model=args.model, effort=args.effort)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, items=per_item), f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n-samples", type=int, default=3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch", type=int, default=128)
    r.add_argument("--tokenizer", default=f"{D}/tokenizer/bpe32k.json")
    r.add_argument("--device", default="cuda")
    rs = sub.add_parser("rescore")
    rs.add_argument("dirs", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    j = sub.add_parser("judge")
    j.add_argument("a")
    j.add_argument("b")
    j.add_argument("--out", required=True)
    j.add_argument("--samples", type=int, nargs="+", default=[0, 1], help="0 = greedy, 1.. = sampled")
    j.add_argument("--model", default="claude-opus-5-5")
    j.add_argument("--effort", default="medium")
    j.add_argument("--workers", type=int, default=8)
    j.add_argument("--backend", default="api", choices=["api", "cli"])
    j.add_argument("--cache", default=f"{D}/judge_cache.jsonl")
    args = ap.parse_args()
    dict(run=cmd_run, rescore=cmd_rescore, compare=cmd_compare, judge=cmd_judge)[args.cmd](args)


if __name__ == "__main__":
    main()
