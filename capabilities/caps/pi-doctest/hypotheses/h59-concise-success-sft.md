# H59 Concise Full-Success SFT

## Hypothesis

H58 showed that adding small anti-premature-DONE contrasts to suffix GRPO is
still too narrow and still creates wider-gate zero rollouts. H59 switches
methodology: reconstruct complete successful train-only trajectories and train
SFT only on positive full behavior, with no negative terminal examples. The
test is whether positive concise workflow anchoring is less policy-drifty than
pairwise suffix ranking.

## Data

- Source: `/tmp/pi-doctest-h54-step-local-concise/grpo-train.step-local-concise.g4x3.jsonl`.
- Output: `/tmp/pi-doctest-h59-concise-success-sft/sft.concise-success.g4.jsonl`.
- Examples: 4.
- Shape: read, concise `edit`, doctest, concise `DONE`.
- Character lengths: 1776, 2007, 2229, 1964.
- Tokenized lengths: 508, 676, 613, 664.

## Training

Adapter: `pi-doctest-h59-concise-success-sft-g4-r4a4lr5e7`.

Training used native `cuda_sft_file`, rank 4 / alpha 4, lr `5e-7`, 1 epoch,
and `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 537.075s observed with peak observed VRAM 16003
MiB. The receipt reported 4 examples trained, 578 action tokens, 1883 context
tokens, rank 4 / alpha 4, and lr `5e-7`. Adapter verify passed with 400
nonzero LoRA tensors and a LoRA update proxy of 0.060093.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.971875, no zero rollouts, mean wall-clock 30.27s.
- H59: composite 0.75, one zero rollout, mean wall-clock 37.14s.
- Delta: -0.221875 composite, slower by 6.88s.

No promotion check was run.

## Verdict

Rejected at smoke. The systems result is useful: current-main native SFT can
train this compact full-success anchor locally, unlike the older H44 attempt.
The capability result is negative: positive SFT on four base-success traces
still creates reliability drift and a zero rollout. Do not chain from H59.

Next work should not simply add more copies of base-success traces at the same
shape. Better candidates are a much broader/diverse behavior anchor with
explicit regularization, or a teacher/generated distribution that is not just
the base model's own successful trajectories.
