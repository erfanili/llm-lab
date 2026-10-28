"""Evaluate a base model: held-out bits-per-byte + zero-shot benchmarks.

    python scripts/eval_base.py --ckpt /data/erfan/llm-lab/runs/50m/ckpt.pt       # ours
    python scripts/eval_base.py --hf EleutherAI/pythia-160m                         # reference model

Writes <out>/summary.json and <out>/<task>.items.jsonl (per-item, for paired comparisons).
`val.doc_bpb` (whole held-out documents, any tokenizer) is comparable across all models;
`val.loss` / `val.bpb` (fixed 1024-token windows of our token file) only exist for our models.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from llm.data import TokenFile, token_byte_lengths
from llm.evals import DEFAULT_TASKS, PRIMARY, doc_bpb, load_items, load_val_docs, run_task
from llm.evaluate import eval_loss
from llm.lm import HFLM, OurLM

D = "/data/erfan/llm-lab"


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="one of our checkpoints")
    src.add_argument("--hf", help="Hugging Face model id (reference models)")
    ap.add_argument("--revision", default=None, help="HF revision, e.g. a Pythia step like step143000")
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="first N items per task (debugging only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-dir", default=f"{D}/evals")
    ap.add_argument("--tokenizer", default=f"{D}/tokenizer/bpe32k.json")
    ap.add_argument("--val-bin", default=f"{D}/tokens/val.bin")
    ap.add_argument("--val-tokens", type=int, default=10 * 2**20)
    ap.add_argument("--val-docs", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="HF weight dtype. fp32 = precise (default); fp16 reproduces e.g. the Pythia paper's numbers")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.ckpt:
        lm = OurLM(args.ckpt, args.tokenizer, device)
        step = lm.ckpt.get("step", "na")
        out = Path(args.out or Path(args.ckpt).parent / "evals" / f"{Path(args.ckpt).stem}_step{step}")
    else:
        dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[args.dtype]
        lm = HFLM(args.hf, device, args.revision, dtype)
        step = args.revision or "main"
        out = Path(args.out or Path(D) / "evals_external" / f"{args.hf.replace('/', '__')}@{step}_{args.dtype}")
    out.mkdir(parents=True, exist_ok=True)

    summary = dict(model=lm.name, step=step, max_len=lm.max_len, dtype=args.dtype if args.hf else "fp32", tasks={}, val={})
    t0 = time.time()
    with torch.no_grad():
        summary["val"]["doc"] = doc_bpb(lm, load_val_docs(args.val_bin, args.tokenizer, args.val_docs))
    print(f"val doc_bpb={summary['val']['doc']['bpb']:.4f} ({args.val_docs} docs, "
          f"{summary['val']['doc']['bytes_per_token']:.2f} bytes/token)")
    if args.ckpt:
        tb = torch.from_numpy(token_byte_lengths(args.tokenizer)).to(device)
        r = eval_loss(lm.model, TokenFile(args.val_bin, lm.max_len), tb, args.val_tokens, 16, 0, 1, device)
        summary["val"].update(r)
        print(f"val window loss={r['loss']:.4f} bpb={r['bpb']:.4f}")

    for task in args.tasks:
        t1 = time.time()
        with torch.no_grad():
            metrics, items = run_task(lm, load_items(args.eval_dir, task), args.limit)
        summary["tasks"][task] = metrics
        with open(out / f"{task}.items.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
        m = metrics[PRIMARY[task]]
        if "chance" in metrics:
            extra = f"acc={metrics['acc']['mean']:.4f}  chance={metrics['chance']:.3f}  p_gold={metrics['p_gold']['mean']:.4f}"
        else:
            extra = f"ppl={metrics['ppl']['mean']:.1f}"
        print(f"{task:14s} {PRIMARY[task]:8s} {m['mean']:.4f} ± {m['ci95']:.4f}  (n={m['n']})  {extra}  [{time.time() - t1:.0f}s]")

    summary["eval_seconds"] = time.time() - t0
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
