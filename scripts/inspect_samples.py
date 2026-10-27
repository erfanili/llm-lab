"""Qualitative inspection: generate on a fixed prompt set, and lay out several models side by side.

    # generate (raw continuation + chat template), greedy + sampled
    python scripts/inspect_samples.py generate --ckpt runs/50m/ckpt.pt
    # compare any number of generation files on the same prompts
    python scripts/inspect_samples.py compare A/generations.jsonl B/generations.jsonl --mode chat --out cmp.md

Modes:
  raw   <|endoftext|> + prompt              -> what the base LM thinks follows this text
  chat  the chat template (llm/chat.py)     -> what the model does when asked as an assistant
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llm.chat import ChatFormat
from llm.checkpoint import load_model
from llm.generate import generate

D = "/data/erfan/llm-lab"
REPO = Path(__file__).resolve().parent.parent


def cmd_generate(args):
    device = torch.device(args.device)
    model, ck = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)
    chat = ChatFormat(tok)
    prompts = [json.loads(l) for l in open(args.prompts)]
    step = ck.get("step", "na")
    out = Path(args.out or Path(args.ckpt).parent / "inspect" / f"{Path(args.ckpt).stem}_step{step}")
    out.mkdir(parents=True, exist_ok=True)

    # sample 0 is greedy (deterministic reference); samples 1..n use temperature/top-p with fixed seeds.
    settings = [dict(temperature=0.0)] + [dict(temperature=args.temperature, top_p=args.top_p, seed=args.seed + s)
                                           for s in range(args.n_samples)]
    records = []
    for mode in args.modes:
        inputs = [[chat.eot] + tok.encode(p["prompt"]).ids if mode == "raw"
                  else chat.prompt([dict(role="user", content=p["prompt"])]) for p in prompts]
        for s, st in enumerate(settings):
            outs = generate(model, inputs, args.max_new, stop_ids=(chat.eot,), **st)
            for p, o in zip(prompts, outs):
                records.append(dict(id=p["id"], category=p["category"], prompt=p["prompt"], mode=mode, sample=s,
                                    **{k: v for k, v in st.items() if k != "seed"}, seed=st.get("seed"),
                                    response=chat.decode(o["tokens"]), n_tokens=len(o["tokens"]), finish=o["finish"]))
    meta = dict(ckpt=str(args.ckpt), step=step, max_new=args.max_new, prompts=str(args.prompts))
    with open(out / "generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(dict(**r, **meta)) + "\n")
    write_markdown(out / "generations.md", records, meta)
    print(f"wrote {len(records)} generations to {out}")


def write_markdown(path, records, meta):
    with open(path, "w") as f:
        f.write(f"# Generations: `{meta['ckpt']}` (step {meta['step']})\n\n")
        for pid in dict.fromkeys(r["id"] for r in records):
            rs = [r for r in records if r["id"] == pid]
            f.write(f"## {pid}\n\n**Prompt:** {rs[0]['prompt']}\n\n")
            for r in rs:
                label = "greedy" if r["temperature"] == 0 else f"T={r['temperature']} #{r['sample']}"
                f.write(f"- **{r['mode']} / {label}** ({r['n_tokens']} tok, {r['finish']}): {r['response']!r}\n")
            f.write("\n")


def cmd_compare(args):
    runs = []
    for p in args.files:
        rs = [json.loads(l) for l in open(p)]
        runs.append((rs[0]["ckpt"], {r["id"]: r for r in rs if r["mode"] == args.mode and r["sample"] == args.sample}))
    with open(args.out, "w") as f:
        f.write(f"# Side-by-side ({args.mode}, sample {args.sample})\n\n")
        for i, (name, _) in enumerate(runs):
            f.write(f"- **[{i}]** `{name}`\n")
        f.write("\n")
        for pid, r0 in runs[0][1].items():
            f.write(f"## {pid}\n\n**Prompt:** {r0['prompt']}\n\n")
            for i, (_, rs) in enumerate(runs):
                if pid in rs:
                    f.write(f"- **[{i}]** {rs[pid]['response']!r}\n")
            f.write("\n")
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--prompts", default=str(REPO / "data/prompts/inspect_v1.jsonl"))
    g.add_argument("--modes", nargs="+", default=["raw", "chat"], choices=["raw", "chat"])
    g.add_argument("--n-samples", type=int, default=2)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--max-new", type=int, default=128)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default=None)
    g.add_argument("--tokenizer", default=f"{D}/tokenizer/bpe32k.json")
    g.add_argument("--device", default="cuda")
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    c.add_argument("--mode", default="chat", choices=["raw", "chat"])
    c.add_argument("--sample", type=int, default=0)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_generate(args) if args.cmd == "generate" else cmd_compare(args)


if __name__ == "__main__":
    main()
