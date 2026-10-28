#!/usr/bin/env bash
# 50M model on ~9.7B FineWeb-Edu tokens (~200 tokens/param, 10x the Chinchilla-style 1B run).
# Identical to the base run 50m_lr4e-3 (same init seed, 1 GPU x grad_accum=8, lr 4e-3, warmup, cosine to 10%)
# except the token budget, which is one epoch of train10b.bin (no repeated data).
# Waits for scripts/prepare_10b.sh to finish (it writes train10b.json last).
set -euo pipefail
cd "$(dirname "$0")/.."

D=/data/erfan/llm-lab
GPU=${GPU:-7}
name=50m_10b
until [ -f "$D/tokens/train10b.json" ]; do sleep 60; done
total=$(.venv/bin/python -c "import json; n = json.load(open('$D/tokens/train10b.json'))['n_tokens']; s = 262144; print((n // s - 1) * s)")
mkdir -p "$D/runs/$name"
echo "launching $name: total_tokens=$total on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 exec .venv/bin/python train.py configs/50m.toml \
    train.name="$name" train.lr=4e-3 train.grad_accum=8 train.train_bin="$D/tokens/train10b.bin" \
    train.total_tokens="$total" train.eval_interval=1000 \
    > "$D/runs/$name/stdout.log" 2>&1 < /dev/null
