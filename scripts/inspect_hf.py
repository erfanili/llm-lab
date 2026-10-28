"""Greedy chat-mode generations from a Hugging Face instruct model on the inspection prompts, in the
same schema as inspect_samples.py, so scripts/judge.py can compare reference models with ours.

    python scripts/inspect_hf.py HuggingFaceTB/SmolLM2-135M-Instruct

Each model uses its own chat template (what it was trained with). Same decoding as ours:
greedy, no repetition penalty, max 128 new tokens.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/data/erfan/llm-lab"
REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompts", default=str(REPO / "data/prompts/inspect_v1.jsonl"))
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    prompts = [json.loads(l) for l in open(args.prompts)]
    out = Path(args.out or Path(D) / "inspect_external" / args.model.replace("/", "__"))
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "generations.jsonl", "w") as f:
        for p in prompts:
            x = tok.apply_chat_template([dict(role="user", content=p["prompt"])], add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True).to(args.device)
            with torch.no_grad():
                y = model.generate(**x, max_new_tokens=args.max_new, do_sample=False, repetition_penalty=1.0,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new = y[0, x["input_ids"].shape[1]:].tolist()
            stopped = bool(new) and new[-1] in ([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else tok.eos_token_id)
            f.write(json.dumps(dict(id=p["id"], category=p["category"], prompt=p["prompt"], mode="chat", sample=0,
                                    temperature=0.0, response=tok.decode(new, skip_special_tokens=True).strip(),
                                    n_tokens=len(new), finish="stop" if stopped else "length",
                                    ckpt=args.model, step="hf", max_new=args.max_new)) + "\n")
    print(f"wrote {out / 'generations.jsonl'}")


if __name__ == "__main__":
    main()
