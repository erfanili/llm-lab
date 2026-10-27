# llm-lab

Pretraining (and post-training) LLMs from scratch in plain PyTorch, scaling 50M → 3B, with an
emphasis on precise evaluation. No training frameworks: the model, data parallelism, data loader,
KV-cache generation, eval harness, SFT, reward modelling and PPO are all written out here.

## Layout

```
llm/
  model.py       Llama-style transformer (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings) + KV cache
  dist.py        hand-written data parallelism (flat grad buffer + one all_reduce)
  data.py        stateless, resumable token loader; exact token->byte lengths for bits-per-byte
  evaluate.py    held-out loss / bpb
  evals.py       zero-shot log-likelihood benchmarks with CIs and paired comparisons
  generate.py    batched left-padded sampling with KV cache
  chat.py        chat template shared by SFT / RM / RLHF / inspection
  posttrain.py   shared post-training pieces (log-probs, reward/value head model, logging)
  judge.py       Claude-as-judge (pairwise with order swap, absolute grades, disk cache)
train.py         pretraining            sft.py        supervised fine-tuning
train_rm.py      reward model           rlhf.py       PPO
scripts/         data prep, eval, inspection, judging, sweeps
configs/         TOML configs (override any key on the CLI: section.key=value)
data/prompts/    inspect_v1.jsonl: 100 prompts for qualitative inspection
```

Data, checkpoints and logs live under `/data/erfan/llm-lab/` (`/` is nearly full).

## Setup

```bash
uv sync                      # creates .venv with torch (cu126) etc.
.venv/bin/python -m pytest tests
```

On this machine PCIe P2P hangs NCCL; `llm/dist.py` sets `NCCL_P2P_DISABLE=1`.

## Pretraining

```bash
python scripts/train_tokenizer.py --files <parquet> --out /data/erfan/llm-lab/tokenizer/bpe32k.json
python scripts/tokenize_data.py ...            # train / val / test token files (see script docstring)
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 train.py configs/50m.toml
scripts/sweep_lr_50m.sh                        # LR sweep, one run per GPU
```

## Base-model evaluation

```bash
python scripts/prepare_evals.py                                  # once
python scripts/eval_base.py --ckpt /data/erfan/llm-lab/runs/50m/ckpt.pt
python scripts/eval_compare.py <evalsA> <evalsB>                 # paired B - A with 95% CIs
```

Tasks (zero-shot, lm-eval-harness formats): HellaSwag, ARC-Easy/Challenge, PIQA, SciQ, OpenBookQA,
WinoGrande, BLiMP (67 grammatical phenomena), LAMBADA, plus held-out FineWeb-Edu loss and bits-per-byte.
For each multiple-choice task: `acc`, `acc_norm` (per-byte), and `p_gold` (probability mass on the
correct choice: continuous, so it separates checkpoints even when accuracy sits near chance).
Per-item results are saved, so comparisons between checkpoints are paired.

## Qualitative inspection

```bash
python scripts/inspect_samples.py generate --ckpt <ckpt>          # raw + chat modes, greedy + 2 samples
python scripts/inspect_samples.py compare A/generations.jsonl B/generations.jsonl --out cmp.md
```

The 100 prompts span 14 categories (factual QA, explanation, creative writing, lists, format
constraints, rewriting, classification, math, reasoning, summarisation, code, advice, chit-chat,
harmful requests) and are chosen so instruction tuning / RLHF changes are visible. Each has an
`expect` note that also serves as the judge's reference.

## Post-training

```bash
python scripts/prepare_posttrain.py            # SFT (smol-smoltalk), prefs (UltraFeedback + HH-RLHF), RL prompts
python sft.py configs/sft_50m.toml
python train_rm.py configs/rm_50m.toml
python rlhf.py configs/rlhf_50m.toml
```

Every eval step in each stage writes a checkpoint `ckpt_step{N}.pt` plus metrics to `log.jsonl`:
- SFT: masked val loss; greedy chat generations on the inspection prompts.
- RM: val pairwise accuracy (95% CI), per source; reward mean/std (used to normalise RLHF rewards).
- RLHF: on a fixed held-out prompt set with fixed sampling seeds: RM score and KL-to-SFT with 95% CIs,
  length, EOS rate; inspection generations; optional Claude win rate vs SFT (`ppo.judge_interval`).

## LLM judge

Needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile). Uses `claude-opus-5-5`.

```bash
python scripts/judge.py pairwise sft/inspect/step0.jsonl rlhf/inspect/step500.jsonl --out judge.json
python scripts/judge.py grade rlhf/inspect/step500.jsonl --out grades.json
```

Pairwise judgments run in both orders and count a win only when both agree (cancels position
bias). Results are cached in `/data/erfan/llm-lab/judge_cache.jsonl`.
