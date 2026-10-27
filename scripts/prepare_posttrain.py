"""Prepare post-training data under <out>/:

  sft_{train,val}.pt          tokenized chat conversations (ids + assistant loss mask), smol-smoltalk
  prefs_{train,val}.jsonl     preference pairs {source, prompt: messages, chosen, rejected}
                              from UltraFeedback (helpfulness) and Anthropic HH-RLHF (helpful + harmless)
  rl_prompts_{train,val}.jsonl  single-turn prompts {source, messages} for RLHF rollouts

Everything is filtered to fit the model context (rendered with llm/chat.py's template).
"""

import argparse
import gzip
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from tqdm import tqdm

from llm.chat import ChatFormat

D = "/data/erfan/llm-lab"


def parquet(repo, f):
    return pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pylist()


def jsonl_gz(repo, f):
    with gzip.open(hf_hub_download(repo, f, repo_type="dataset"), "rt") as fh:
        return [json.loads(l) for l in fh]


def clean_messages(msgs):
    out = [dict(role=m["role"], content=m["content"]) for m in msgs if m["content"].strip()]
    return out if out and out[-1]["role"] == "assistant" else None


def prep_sft(chat, out, max_len):
    files = dict(train=[f"data/train-0000{i}-of-00004.parquet" for i in range(4)], val=["data/test-00000-of-00001.parquet"])
    for split, fs in files.items():
        ids, masks, dropped = [], [], 0
        for f in fs:
            for r in tqdm(parquet("HuggingFaceTB/smol-smoltalk", f), desc=f"sft {split} {f}"):
                msgs = clean_messages(r["messages"])
                if msgs is None:
                    dropped += 1
                    continue
                t, m = chat.render(msgs)
                if len(t) > max_len + 1 or sum(m) == 0:
                    dropped += 1
                    continue
                ids.append(torch.tensor(t, dtype=torch.int32))
                masks.append(torch.tensor(m, dtype=torch.bool))
        torch.save(dict(ids=ids, mask=masks), out / f"sft_{split}.pt")
        n_tok = sum(len(x) for x in ids)
        print(f"sft_{split}: {len(ids)} convs, {n_tok / 1e6:.1f}M tokens, "
              f"{sum(int(m.sum()) for m in masks) / 1e6:.1f}M assistant tokens, dropped {dropped}")


def parse_hh(text):
    parts = re.split(r"\n\n(Human|Assistant): ", "\n\n" + text.strip())[1:]
    msgs = [dict(role="user" if r == "Human" else "assistant", content=c.strip()) for r, c in zip(parts[::2], parts[1::2])]
    return msgs


def pairs_ultrafeedback(split):
    for r in parquet("HuggingFaceH4/ultrafeedback_binarized", f"data/{split}_prefs-00000-of-00001.parquet"):
        if r["score_chosen"] == r["score_rejected"]:
            continue  # ties carry no preference signal
        yield dict(source="ultrafeedback", prompt=[dict(role="user", content=r["prompt"])],
                   chosen=r["chosen"][-1]["content"], rejected=r["rejected"][-1]["content"],
                   margin=r["score_chosen"] - r["score_rejected"])


def pairs_hh(subset, split):
    for r in jsonl_gz("Anthropic/hh-rlhf", f"{subset}/{split}.jsonl.gz"):
        c, j = parse_hh(r["chosen"]), parse_hh(r["rejected"])
        if not c or not j or c[-1]["role"] != "assistant" or j[-1]["role"] != "assistant" or c[:-1] != j[:-1]:
            continue
        yield dict(source=f"hh-{subset}", prompt=c[:-1], chosen=c[-1]["content"], rejected=j[-1]["content"])


def prep_prefs(chat, out, max_len):
    for split, uf_split, hh_split in [("train", "train", "train"), ("val", "test", "test")]:
        kept, dropped = [], 0
        gens = [pairs_ultrafeedback(uf_split), pairs_hh("helpful-base", hh_split), pairs_hh("harmless-base", hh_split)]
        for gen in gens:
            for p in gen:
                if not p["chosen"].strip() or not p["rejected"].strip() or p["chosen"].strip() == p["rejected"].strip():
                    dropped += 1
                    continue
                n_prompt = len(chat.prompt(p["prompt"]))
                n = n_prompt + max(len(chat.response_ids(p["chosen"])), len(chat.response_ids(p["rejected"])))
                if n > max_len:
                    dropped += 1
                    continue
                kept.append(p)
        with open(out / f"prefs_{split}.jsonl", "w") as f:
            for p in kept:
                f.write(json.dumps(p) + "\n")
        by = {}
        for p in kept:
            by[p["source"]] = by.get(p["source"], 0) + 1
        print(f"prefs_{split}: {len(kept)} pairs {by}, dropped {dropped}")


def prep_rl_prompts(chat, out, max_prompt, n_val):
    rng = random.Random(0)
    for split, uf_split, hh_split in [("train", "train", "train"), ("val", "test", "test")]:
        prompts = []
        for r in parquet("HuggingFaceH4/ultrafeedback_binarized", f"data/{uf_split}_gen-00000-of-00001.parquet"):
            prompts.append(dict(source="ultrafeedback", messages=[dict(role="user", content=r["prompt"])]))
        for subset in ("helpful-base", "harmless-base"):
            for r in jsonl_gz("Anthropic/hh-rlhf", f"{subset}/{hh_split}.jsonl.gz"):
                first = parse_hh(r["chosen"])[:1]
                if first and first[0]["role"] == "user":
                    prompts.append(dict(source=f"hh-{subset}", messages=first))
        seen, kept = set(), []
        for p in prompts:
            key = p["messages"][0]["content"].strip()
            if key in seen or len(chat.prompt(p["messages"])) > max_prompt:
                continue
            seen.add(key)
            kept.append(p)
        rng.shuffle(kept)
        if split == "val":
            kept = kept[:n_val]  # fixed eval set: the same prompts at every RLHF checkpoint
        with open(out / f"rl_prompts_{split}.jsonl", "w") as f:
            for p in kept:
                f.write(json.dumps(p) + "\n")
        print(f"rl_prompts_{split}: {len(kept)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/posttrain")
    ap.add_argument("--tokenizer", default=f"{D}/tokenizer/bpe32k.json")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-prompt", type=int, default=256)
    ap.add_argument("--n-val-prompts", type=int, default=512)
    ap.add_argument("--only", nargs="*", default=["sft", "prefs", "rl"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chat = ChatFormat(Tokenizer.from_file(args.tokenizer))
    if "sft" in args.only:
        prep_sft(chat, out, args.max_len)
    if "prefs" in args.only:
        prep_prefs(chat, out, args.max_len)
    if "rl" in args.only:
        prep_rl_prompts(chat, out, args.max_prompt, args.n_val_prompts)


if __name__ == "__main__":
    main()
