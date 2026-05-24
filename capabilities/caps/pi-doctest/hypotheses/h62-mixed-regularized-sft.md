# H62 Mixed Synthetic Plus Base-Success SFT

## Hypothesis

H60 synthetic SFT was safer than H59 but still score-negative. H61 showed that
scaling the H60 update can make a smoke false positive, but not a robust
promotion win. H62 tests explicit SFT regularization: alternate synthetic
teacher-style concise traces with train-only base-success concise traces in one
small corpus.

## Data

- Output: `/tmp/pi-doctest-h62-mixed-regularized-sft/sft.mixed-synthetic-base.g4.jsonl`.
- Synthetic source: `/tmp/pi-doctest-h60-synthetic-ideal-sft/sft.synthetic-ideal.g4.jsonl`.
- Base-success source: `/tmp/pi-doctest-h59-concise-success-sft/sft.concise-success.g4.jsonl`.
- Examples: 4 total, alternating synthetic/base/synthetic/base.
- Tokenized lengths: 507, 508, 525, 613.
- Character lengths: 1739, 1776, 1845, 2229.

## Training

Adapter: `pi-doctest-h62-mixed-synth-base-sft-g4-r4a4lr1e7`.

Training used native `cuda_sft_file`, rank 4 / alpha 4, lr `1e-7`, 1 epoch,
and `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 82.883s observed with peak observed VRAM 15999
MiB. The receipt reported 4 examples trained, 563 action tokens, 1590 context
tokens, rank 4 / alpha 4, and lr `1e-7`. Adapter verify passed with 400
nonzero LoRA tensors and a LoRA update proxy of 0.011946.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.971875, no zero rollouts, mean wall-clock 23.56s.
- H62: composite 0.925000, no zero rollouts, mean wall-clock 38.60s.
- Delta: -0.046875 composite, slower by 15.05s.

No promotion check was run.

## Verdict

Rejected at smoke. The mixed distribution avoided zero rollouts, but still lost
score and added latency. This suggests the simple small-SFT distribution axis
is not enough: synthetic-only, base-success-only, scaled synthetic, and
synthetic+base all fail.

Next work should switch signal source: real reward-variance groups with richer
semantics, a stronger teacher distribution, or a non-SFT method with explicit
hard-tail reliability pressure.
