# Eval harness validation (2026-10-26)

Our harness (`llm/evals.py`, lm-eval-harness formats and tokenization) reproduces the Pythia paper
(arXiv:2304.01373, Table 7; raw outputs in EleutherAI/pythia `evals/pythia-v1/*/zero-shot/*_step143000.json`)
when run in **fp16**, the precision the paper used:

| model | LAMBADA | PIQA | WinoGrande | ARC-E | ARC-C | SciQ | LAMBADA ppl |
|---|---|---|---|---|---|---|---|
| Pythia-70M paper | 18.5 | 59.5 | 52.8 | 37.4 | 18.1 | 60.1 | 142.43 |
| Pythia-70M ours (fp16) | 18.4 | 59.3 | 52.0 | 36.4 | 18.5 | 60.4 | 142.14 |
| Pythia-160M paper | 32.8 | 62.7 | 53.1 | 43.5 | 18.8 | 74.1 | 38.06 |
| Pythia-160M ours (fp16) | 32.6 | 61.9 | 51.3 | 43.6 | 18.8 | 73.9 | 38.22 |
| Pythia-410M paper | 51.6 | 66.8 | 53.7 | 52.1 | 21.3 | 81.1 | — |
| Pythia-410M ours (fp16) | 51.5 | 66.8 | 53.4 | 52.0 | 21.4 | 80.7 | 10.84 |

All 18 accuracies agree within the 95% CI (largest deviation 1.3 sigma).

Findings
- **Precision matters for small models.** fp16 understates Pythia-70M/160M: in fp32, LAMBADA is
  22.7 / 35.4 (vs 18.4 / 32.6 in fp16) and SciQ 64.0 (vs 60.4) for 70M. No effect at 410M.
  bf16 autocast is worse still (Pythia-160M LAMBADA 28.9). We therefore evaluate everything in **fp32**
  and compare our models against fp32 reference runs (`/data/erfan/llm-lab/evals_external/*_fp32`).
- LAMBADA perplexity follows lm-eval: exp(-mean per-example log-likelihood), not per-token.
