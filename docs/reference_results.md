# Published results for open base models (≤ ~1.6B)

Published zero-shot results for open base models <= ~1.6B params, copied verbatim (converted to %) from the cited sources. One table per source: protocols differ (prompt format, acc vs acc_norm, normalisation), so numbers are only comparable WITHIN a table. Gathered 2026-10-26.

## Pythia suite + BLOOM/OPT baselines

Source: [Biderman et al. 2023, Pythia, arXiv:2304.01373, Appendix G Tables 7, 15, 16](https://arxiv.org/abs/2304.01373)

Protocol: lm-evaluation-harness, 0-shot, accuracy (fractions in the paper, ± stderr ~0.7-1.4 pts). Final checkpoints, non-deduplicated Pythia. Same prompt formats as our harness -> compare against our `acc` (not acc_norm).

| Model | Params | Train tokens | lambada | piqa | winogrande | arc_easy | arc_challenge | sciq |
|---|---|---|---|---|---|---|---|---|
| Pythia-70M | 70M | 300B | 18.5 | 59.5 | 52.8 | 37.4 | 18.1 | 60.1 |
| Pythia-160M | 160M | 300B | 32.8 | 62.7 | 53.1 | 43.5 | 18.8 | 74.1 |
| OPT-125M | 125M | — | 37.9 | 63.0 | 50.3 | 43.5 | 18.9 | 75.1 |
| OPT-350M | 350M | — | 45.2 | 64.4 | 52.3 | 44.0 | 20.7 | 74.8 |
| Pythia-410M | 410M | 300B | 51.6 | 66.8 | 53.7 | 52.1 | 21.3 | 81.1 |
| BLOOM-560M | 560M | — | 34.1 | 63.7 | 50.4 | 47.6 | 22.1 | 80.4 |
| Pythia-1B | 1B | 300B | 56.2 | 70.7 | 53.7 | 56.9 | 24.4 | 84.0 |
| BLOOM-1.1B | 1.1B | — | 42.6 | 67.2 | 54.7 | 51.5 | 23.6 | 83.3 |
| OPT-1.3B | 1.3B | — | 57.9 | 71.7 | 59.7 | 57.0 | 23.1 | 84.5 |

## MobileLLM and sub-600M baselines

Source: [Liu et al. 2024, MobileLLM, arXiv:2402.14905, Table 3](https://arxiv.org/abs/2402.14905)

Protocol: 0-shot; baselines re-evaluated by the authors from HF checkpoints for consistency. acc vs acc_norm and harness not stated (Pythia-160M ARC-e here is 40.0 vs 43.5 in the Pythia paper, so it differs from plain acc). MobileLLM models trained on 1T tokens. Compare our acc_norm (HellaSwag/ARC/OBQA) and acc (WinoGrande) with caution.

| Model | Params | Train tokens | arc_easy | arc_challenge | piqa | hellaswag | openbookqa | winogrande |
|---|---|---|---|---|---|---|---|---|
| Cerebras-GPT-111M | 111M | — | 35.8 | 20.2 | 58.0 | 26.7 | 29.0 | 48.8 |
| LaMini-GPT-124M | 124M | — | 43.6 | 26.0 | 62.7 | 30.2 | 29.6 | 49.2 |
| Galactica-125M | 125M | — | 44.0 | 26.2 | 55.4 | 29.6 | 28.2 | 49.6 |
| OPT-125M | 125M | — | 41.3 | 25.2 | 62.0 | 31.1 | 31.2 | 50.8 |
| GPT-Neo-125M | 125M | — | 40.7 | 24.8 | 62.5 | 29.7 | 31.6 | 50.7 |
| Pythia-160M | 162M | 300B | 40.0 | 25.3 | 62.0 | 29.9 | 31.2 | 50.9 |
| RWKV-169M | 169M | — | 42.5 | 25.3 | 63.9 | 31.9 | 33.8 | 51.5 |
| MobileLLM-125M | 125M | 1T | 43.9 | 27.1 | 65.3 | 38.9 | 39.5 | 53.1 |
| Cerebras-GPT-256M | 256M | — | 37.9 | 23.2 | 61.4 | 28.3 | 31.8 | 50.5 |
| OPT-350M | 331M | — | 41.9 | 25.7 | 64.8 | 36.2 | 33.3 | 52.4 |
| MobileLLM-350M | 345M | 1T | 53.8 | 33.5 | 68.6 | 49.6 | 40.0 | 57.6 |
| Pythia-410M | 405M | 300B | 47.1 | 30.3 | 67.2 | 40.1 | 36.2 | 53.4 |
| RWKV-430M | 430M | — | 48.9 | 32.0 | 68.1 | 40.6 | 37.8 | 51.6 |
| BLOOM-560M | 559M | — | 43.7 | 27.5 | 65.1 | 36.5 | 32.6 | 52.2 |
| Cerebras-GPT-590M | 590M | — | 42.6 | 24.9 | 62.8 | 32.0 | 33.2 | 49.7 |

## ~1B models (TinyLlama paper)

Source: [Zhang et al. 2024, TinyLlama, arXiv:2401.02385, Table 3](https://arxiv.org/abs/2401.02385)

Protocol: lm-evaluation-harness, 0-shot; acc vs acc_norm not stated. TinyLlama v1.0: ~1T unique tokens x ~3 epochs (3T seen); v1.1: 2T.

| Model | Params | Train tokens | hellaswag | openbookqa | winogrande | arc_challenge | arc_easy | piqa |
|---|---|---|---|---|---|---|---|---|
| Pythia-1.0B | 1B | 300B | 47.2 | 31.4 | 53.4 | 27.1 | 49.0 | 69.2 |
| Pythia-1.4B | 1.4B | 300B | 52.0 | 33.2 | 57.4 | 28.5 | 54.0 | 71.0 |
| OPT-1.3B | 1.3B | — | 53.6 | 33.4 | 59.6 | 29.4 | 50.8 | 72.4 |
| TinyLlama v1.0 | 1.1B | 3T | 59.2 | 36.0 | 59.1 | 30.1 | 55.2 | 73.3 |
| TinyLlama v1.1 | 1.1B | 2T | 61.5 | 36.8 | 59.4 | 32.7 | 55.5 | 73.6 |

## ~1B models (OLMo paper)

Source: [Groeneveld et al. 2024, OLMo, arXiv:2402.00838, Table 3](https://arxiv.org/abs/2402.00838)

Protocol: 0-shot rank classification (Catwalk). Per-task normalisation: unconditional for ARC and OBQA, per-token for HellaSwag/PIQA/WinoGrande, none for SciQ. Unconditional normalisation gives much higher ARC scores (Pythia-1B ARC-c 33.1 here vs 24.4 in the Pythia paper): NOT comparable to our ARC/OBQA numbers. OLMo-1B trained on 2T tokens.

| Model | Params | Train tokens | arc_challenge | arc_easy | hellaswag | openbookqa | piqa | sciq | winogrande |
|---|---|---|---|---|---|---|---|---|---|
| Pythia-1B | 1B | 300B | 33.1 | 50.2 | 44.7 | 37.8 | 69.1 | 86.0 | 53.3 |
| TinyLlama-1.1B | 1.1B | 3T | 34.8 | 53.2 | 58.7 | 43.6 | 71.1 | 90.5 | 58.9 |
| OLMo-1B | 1B | 2T | 34.5 | 58.1 | 62.5 | 46.4 | 73.7 | 88.1 | 58.9 |
| StableLM-1.6B | 1.6B | — | 43.8 | 63.7 | 68.2 | 45.8 | 74.0 | 94.7 | 64.9 |

## SmolLM / SmolLM2 base models (model cards)

Source: [HuggingFaceTB/SmolLM2-135M and SmolLM2-360M model cards](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)

Protocol: lighteval, 0-shot; ARC is the average of ARC-Easy and ARC-Challenge; prompt formulations differ from lm-eval-harness. SmolLM2-135M: 2T tokens; SmolLM2-360M: 4T tokens. Trained largely on FineWeb-Edu-style data, so the closest data match to ours.

| Model | Params | Train tokens | hellaswag | arc_avg | piqa | winogrande | openbookqa |
|---|---|---|---|---|---|---|---|
| SmolLM-135M | 135M | — | 41.2 | 42.4 | 68.4 | 51.3 | 34.0 |
| SmolLM2-135M | 135M | 2T | 42.1 | 43.9 | 68.4 | 51.3 | 34.6 |
| SmolLM-360M | 360M | — | 51.8 | 50.1 | 71.6 | 52.8 | 37.2 |
| SmolLM2-360M | 360M | 4T | 54.5 | 53.0 | 71.7 | 52.5 | 37.4 |
| Qwen2.5-0.5B | 500M | — | 51.2 | 45.4 | 69.9 | 54.1 | 37.4 |

## GPT-2-small-class HellaSwag (incl. a FineWeb-trained reproduction)

Source: [karpathy/llm.c discussion #481 (GPT-2 124M reproduction); GPT-3 paper Appendix H as cited there](https://github.com/karpathy/llm.c/discussions/481)

Protocol: llm.c HellaSwag eval (completion-style, lowest per-token loss ~ acc_norm). The reproduction is the closest analogue to our setup: same data family (FineWeb), 10B tokens.

| Model | Params | Train tokens | hellaswag |
|---|---|---|---|
| GPT-2 (OpenAI) | 124M | — | 29.4 |
| GPT-2 repro, llm.c (FineWeb) | 124M | 10B | 29.9 |
| GPT-3 Small | 125M | 300B | 33.7 |

## BLiMP overall accuracy (original paper baselines)

Source: [Warstadt et al. 2020, BLiMP, arXiv:1912.00582, Table 3](https://arxiv.org/abs/1912.00582)

Protocol: Full-sentence log-probability forced choice over all 67k minimal pairs (same as ours). GPT-2 here is GPT-2-large (774M, ~8B WebText tokens per the paper's estimate); LSTM and Transformer-XL trained on ~100M Wikipedia tokens. Chance = 50.

| Model | Params | Train tokens | blimp |
|---|---|---|---|
| 5-gram (Gigaword) | — | 3.1B | 61.2 |
| LSTM (Wikipedia) | — | 83M | 69.8 |
| Transformer-XL (WikiText-103) | — | 103M | 69.6 |
| GPT-2-large | 774M | — | 83.0 |
| Human (individual) | — | — | 88.6 |
