#!/usr/bin/env bash
# Validate our eval harness against the Pythia paper (fp16 reproduces its numbers; fp32 is the precise
# reference we compare our own models against), and evaluate our current base model.
# Usage: scripts/validate_harness.sh <gpu> <gpu> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
L=/data/erfan/llm-lab/evals_external/logs
mkdir -p "$L"
E=".venv/bin/python scripts/eval_base.py"
G=("$@")

run_chain() {  # gpu, then commands as "name|args"
    local gpu=$1; shift
    for job in "$@"; do
        CUDA_VISIBLE_DEVICES=$gpu $E ${job#*|} > "$L/${job%%|*}.log" 2>&1 || echo "FAILED: ${job%%|*}"
    done
}

run_chain "${G[0]}" "pythia-70m_fp16|--hf EleutherAI/pythia-70m --dtype fp16" "pythia-70m_fp32|--hf EleutherAI/pythia-70m" \
    "ours_50m_lr4e-3|--ckpt /data/erfan/llm-lab/runs/50m_lr4e-3/ckpt.pt" &
run_chain "${G[1]}" "pythia-160m_fp16|--hf EleutherAI/pythia-160m --dtype fp16" "pythia-160m_fp32|--hf EleutherAI/pythia-160m" &
run_chain "${G[2]}" "pythia-410m_fp16|--hf EleutherAI/pythia-410m --dtype fp16" "pythia-410m_fp32|--hf EleutherAI/pythia-410m" &
wait
echo "all done"
