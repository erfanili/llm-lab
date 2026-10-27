"""Download base-model benchmarks and normalise them to JSONL.

Multiple-choice item: {"task", "id", "choices": [{"context", "continuation"}, ...], "gold": int, "meta": {...}}
  scored as log P(continuation | context) per choice. Formats follow lm-evaluation-harness so
  numbers are comparable with published small-model results (Pythia, GPT-2, SmolLM, ...).
LAMBADA item: {"task", "id", "context", "continuation"} -- greedy exact match of the last word.
"""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


def rows(repo: str, filename: str) -> list[dict]:
    path = hf_hub_download(repo, filename, repo_type="dataset")
    if filename.endswith(".jsonl"):
        return [json.loads(line) for line in open(path)]
    return pq.read_table(path).to_pylist()


def mc(task, i, contexts, continuations, gold, **meta):
    if isinstance(contexts, str):
        contexts = [contexts] * len(continuations)
    choices = [dict(context=c, continuation=k) for c, k in zip(contexts, continuations)]
    return dict(task=task, id=f"{task}/{i}", choices=choices, gold=int(gold), meta=meta)


def hellaswag():
    def pre(t):
        t = t.strip().replace(" [title]", ". ")
        t = re.sub(r"\[.*?\]", "", t)
        return t.replace("  ", " ")
    for i, r in enumerate(rows("Rowan/hellaswag", "data/validation-00000-of-00001.parquet")):
        ctx = pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        yield mc("hellaswag", i, ctx, [" " + pre(e) for e in r["endings"]], r["label"])


def arc(level):
    task = f"arc_{level.lower()}"
    for i, r in enumerate(rows("allenai/ai2_arc", f"ARC-{level}/test-00000-of-00001.parquet")):
        labels = list(r["choices"]["label"])
        yield mc(task, i, f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                 labels.index(r["answerKey"]))


def piqa():
    for i, r in enumerate(rows("baber/piqa", "piqa_validation.parquet")):
        yield mc("piqa", i, f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], r["label"])


def sciq():
    for i, r in enumerate(rows("allenai/sciq", "data/test-00000-of-00001.parquet")):
        ctx = f"{r['support'].strip()}\nQuestion: {r['question']}\nAnswer:".lstrip()
        opts = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
        yield mc("sciq", i, ctx, [" " + o for o in opts], 3)


def openbookqa():
    for i, r in enumerate(rows("allenai/openbookqa", "main/test-00000-of-00001.parquet")):
        labels = list(r["choices"]["label"])
        yield mc("openbookqa", i, r["question_stem"], [" " + t for t in r["choices"]["text"]], labels.index(r["answerKey"]))


def winogrande():
    # Partial scoring: the option fills the blank in the *context*; the shared remainder is scored.
    for i, r in enumerate(rows("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet")):
        s = r["sentence"]
        k = s.index("_")
        rest = " " + s[k + 1 :].strip()
        yield mc("winogrande", i, [s[:k] + r["option1"], s[:k] + r["option2"]], [rest, rest], int(r["answer"]) - 1)


def blimp():
    for f in sorted(f for f in list_repo_files("nyu-mll/blimp", repo_type="dataset") if f.endswith(".parquet")):
        subset = f.split("/")[0]
        for i, r in enumerate(rows("nyu-mll/blimp", f)):
            # Whole-sentence log-prob comparison (context = document start only).
            yield mc("blimp", f"{subset}/{i}", "", [r["sentence_good"], r["sentence_bad"]], 0,
                     subset=subset, phenomenon=r["linguistics_term"])


def lambada():
    for i, r in enumerate(rows("EleutherAI/lambada_openai", "data/lambada_test.jsonl")):
        ctx, last = r["text"].rsplit(" ", 1)
        yield dict(task="lambada", id=f"lambada/{i}", context=ctx, continuation=" " + last)


TASKS = dict(hellaswag=hellaswag, arc_easy=lambda: arc("Easy"), arc_challenge=lambda: arc("Challenge"), piqa=piqa,
             sciq=sciq, openbookqa=openbookqa, winogrande=winogrande, blimp=blimp, lambada=lambada)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/erfan/llm-lab/evals")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name in args.tasks:
        items = list(TASKS[name]())
        with open(Path(args.out) / f"{name}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"{name}: {len(items)} items")


if __name__ == "__main__":
    main()
