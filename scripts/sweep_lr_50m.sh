#!/usr/bin/env bash
# LR sweep for the 50M model: one run per GPU, all from random init with the same seed
# (same init weights + same data order), so peak LR is the only thing that changes.
# 1 GPU x grad_accum=8 sees exactly the same 256 sequences per step as the 2 GPU x 4 baseline.
# The lr=2e-3 point is the baseline run `50m`.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=/data/erfan/llm-lab/runs
GPUS=(4 5 6 7)
LRS=(5e-4 1e-3 4e-3 8e-3)

for i in "${!LRS[@]}"; do
    lr=${LRS[$i]}
    name=50m_lr${lr}
    mkdir -p "$RUNS/$name"
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} OMP_NUM_THREADS=8 setsid nohup .venv/bin/python train.py configs/50m.toml \
        train.name="$name" train.lr="$lr" train.grad_accum=8 \
        > "$RUNS/$name/stdout.log" 2>&1 < /dev/null &
    echo "launched $name on GPU ${GPUS[$i]} (pid $!)"
done
