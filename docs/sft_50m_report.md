# SFT of the 50M base model — report (2026-10-27)

**Setup.**
- **Base model:** `runs/50m_lr4e-3`, the LR-sweep winner: 48M params, 1B FineWeb-Edu tokens, val loss 3.442.
- **Data:** smol-smoltalk (265k conversations).
- **Prompt format:** `User: … / Assistant: …`, with loss on assistant tokens only.
- **Training:** 2 epochs = 8,289 steps of 64 conversations each. AdamW at lr 3e-4 with 50 warmup steps and cosine decay, no weight decay.
- **Compute:** a single A5000, 2.6 h.
- **Output:** `/data/erfan/llm-lab/runs/sft_50m/`.

## Training curve

On 2,000 held-out conversations (674k assistant tokens), val loss falls quickly and is still falling slowly at the end, so it has not overfit:

| step | 0 (base) | 250 | 1000 | 2000 | 4000 | 6000 | **8289 (final)** |
|---|---|---|---|---|---|---|---|
| val loss | 2.706 | 1.901 | 1.717 | 1.634 | 1.549 | 1.509 | **1.488** |

The final checkpoint is the best by val loss. It is the only one judged below.

## Behaviour on the 100 inspection prompts (greedy, chat format)

| | base (step 0) | SFT final |
|---|---|---|
| responses that end on their own (EOT) | 2 / 100 | **65 / 100** |
| mean response length (tokens, cap 128) | 126 | 80 |
| Claude judge: pairwise win rate of SFT vs base | — | **91% ± 5** (90 W / 8 L / 2 T) |
| Claude judge: absolute score, 1–10 | not run | **2.17 ± 0.29** |

**Judge protocol.**
- **Judge:** `claude-opus-5-5`, medium effort.
- **Pairwise:** each pair is judged in both orders, and a win counts only if both orders agree.
- **Grading:** against the per-prompt `expect` note.
- **Results:** saved in `runs/sft_50m/judge/`.

**What SFT taught.**
- **Format:** the base model echoes the prompt or invents `User:` turns in a loop. The SFT model answers as an assistant and usually stops.
- **Simple answers are often right:** "The capital of France is Paris.", the sentiment of a review, a correct `square_number` function. Code is the best category (mean grade 4.6), followed by chitchat (3.4).

**What it did not teach.** The grades are mostly 1–2 (72 of 100). This is the limit of a 50M model, not a failure of SFT:
- **Factual and reasoning errors:** "the sky is blue because of chlorophyll", "No, Tom is not an animal", an arithmetic-series formula for 7+5. Math is the worst category (1.2).
- **Instruction constraints ignored:** one-word answers, past-tense rewrites.
- **Degeneration persists:** repeated phrases, and lists like "1. Red 2. Red …". 35 of 100 responses still hit the length cap.

**Where SFT lost (8 of 100).**
- **Harmful prompts, 4 of 6.** The base model babbled harmlessly, while the SFT model complied ("To steal a car, you need to…"). The content is incoherent and not actionable, but the model is willing. smol-smoltalk teaches helpfulness, not refusals.
- **4 others** where the base model's vacuous echo was judged better than a fluent but confidently wrong answer.

## Base-model benchmarks before and after SFT (fp32, paired on identical items)

**Why both views are needed.** The harness and the judge measure different things:
- **The harness** ranks given options by likelihood. The base model does well here *for its size*, beating Pythia-70M on 6/9 tasks. But it is far from solving the tasks: ARC-E acc_norm is 42.5% against 25% chance.
- **The judge** needs the model to *generate* a correct, coherent answer, which is much harder. Pythia-70M/160M would also score around 2/10.

So the two results are consistent. What SFT changed:

| | base | SFT | Δ (paired 95% CI) |
|---|---|---|---|
| doc bpb, FineWeb-Edu val (lower is better) | 1.025 | 1.238 | +0.21 |
| BLiMP acc | 76.5 | 70.6 | **−5.9 ± 0.3** |
| LAMBADA acc | 19.7 | 16.4 | **−3.2 ± 1.0** |
| PIQA acc_norm | 59.2 | 56.5 | **−2.7 ± 1.8** |
| HellaSwag, ARC-E/C, SciQ, OBQA, WinoGrande | | | within ±2, not significant |

**SFT has a measurable cost to base-model ability.** The knowledge tasks are unchanged. But grammaticality (BLiMP), narrative prediction (LAMBADA) and plain-text likelihood all regress, which is forgetting of the pretraining distribution.

The cause is plausibly the strength of the SFT stage: 2 epochs at lr 3e-4, which is 7.5% of the peak pretraining LR, over 0.5B tokens. That is substantial next to 1B pretraining tokens.

**Mitigations to try:**
- 1 epoch, or a lower LR.
- Mixing ~10–20% pretraining data into SFT.

Each can be checked cheaply with this paired comparison together with the judge.

## Takeaways and next steps

1. **The pipeline works end to end:** pretraining, SFT, inspection, and the judge. Assistant-only loss and the chat template behave as intended.
2. **Capability is bounded by pretraining scale.** SFT changes *style*, not knowledge: fluent wrong answers replace degenerate loops. Real gains in judge scores need the larger models on the scale ladder.
3. **SFT costs some base ability.** BLiMP falls 5.9 points and LAMBADA 3.2. Run this paired check on every post-training stage, RLHF included.
4. **Safety needs preference data.** The reward model and RLHF stage (infra ready: `train_rm.py`, `rlhf.py`) should target refusals and repetition. The judge's harmful-category score is a built-in check.
5. **Caveats:**
   - n = 100 prompts, so per-category CIs are wide (5–10 prompts each).
   - A single greedy sample per prompt.
   - The base model's absolute score was not computed. Its pairwise losses imply it is lower.
