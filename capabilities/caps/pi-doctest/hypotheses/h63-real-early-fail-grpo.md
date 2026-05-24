# H63 Real Early-Failure GRPO

## Hypothesis

After H59-H62, tiny SFT distribution changes look exhausted. H63 returns to
real train-rollout reward variance from H44, but removes the long repair tail
that made H44 too slow. Train a compact contrast between direct success and a
real wrong first edit followed by a failed doctest and terminal stop.

## Data

- Source: `/tmp/pi-doctest-h44-natural-compact-grpo/grpo-train.compact-pair.jsonl`.
- Output: `/tmp/pi-doctest-h63-real-early-fail-grpo/grpo-train.real-early-fail.g2.jsonl`.
- Groups: 2.
- Completions: 4.
- Rewards: 1.0 vs 0.35 for each group.
- Preferred trajectory: real successful read, edit, doctest pass, `DONE`.
- Rejected trajectory: real read, wrong first edit, real failed doctest
  observation, then `DONE`.
- Removed: the long repair tail from the original H44 near-miss.

Dry-run:

- action tokens: 552.
- env tokens: 916.
- context tokens: 1112.
- reward stdev: 0.325.

## Training

Adapter: `pi-doctest-h63-real-early-fail-g2-r4a4lr5e7`.

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4, lr
`5e-7`, seed `3141592653`, no ECHO, reward variance filter min `0.001`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 391.914s observed with peak observed VRAM 16008
MiB. The receipt reported 2 groups trained, 552 action tokens, 916 env tokens,
1112 context tokens, and reward stdev 0.325. Adapter verify passed with 400
nonzero LoRA tensors and a LoRA update proxy of 0.028180.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.98125, no zero rollouts, mean wall-clock 25.29s.
- H63: composite 0.75, one zero rollout, mean wall-clock 49.23s.
- Delta: -0.23125 composite, slower by 23.94s.

No promotion check was run.

## Verdict

Rejected at smoke. The useful systems result is that real reward-variance GRPO
becomes locally trainable when the long repair tail is removed. The capability
result is negative: the failed-doctest terminal-stop contrast still introduces
zero-rollout reliability failure and latency regression.

Next real-variance work should preserve the trainability trick but avoid
showing failed-doctest terminal stops as the bad branch. A better contrast is
likely direct success versus real failed edit followed by a concise repair
continuation, or a softer reward gap that does not over-teach terminal behavior
around failures.
