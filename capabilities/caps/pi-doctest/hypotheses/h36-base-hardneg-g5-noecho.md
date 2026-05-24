# H36 Base Hard-Negative G5 No-ECHO

## Hypothesis

H33's two-group no-ECHO hard-negative adapter looked promising once, then failed
confirmation in H35. One plausible reason is that the two-group signal was too
narrow. H36 tests the same no-ECHO policy-only hard-negative idea with the full
five train-only hard-negative groups from H29, trained fresh from base.

## Data

- Source: `/tmp/pi-doctest-h29-base-hardneg/grpo-train.hardneg.jsonl`.
- Groups: 5 train-only hard-negative groups.
- Completions: 10.
- Rewards: `[1.0, 0.0]` in each group.
- Base adapter: none.

Dry-run token shape:

| metric | value |
| --- | ---: |
| groups | 5 |
| completions | 10 |
| action tokens | 936 |
| env tokens | 1177 |
| context tokens | 2710 |
| max seq length | 554 |
| max action tokens/completion | 150 |

## Training

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, `--no-echo`. Gradient checkpointing
was explicitly enabled with `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

Training completed successfully in 439s observed wall-clock, with observed peak
VRAM about 16000 MiB. The receipt reports 435145 ms wall-clock, 368980 ms in
backward, 53462 ms in reference forward, and 12227 ms in policy forward.

Adapter: `pi-doctest-h36-base-hardneg-g5-noecho-r4a8`.

`kiln adapter verify` passed with 400 nonzero LoRA tensors and delta proxy
0.577938.

## Blind Smoke

`LIMIT=4 SEEDS=1`, paired against `/tmp/pi-doctest-thinking-on-smoke.json`.

| metric | base | H36 |
| --- | ---: | ---: |
| composite | 0.934375 | 0.750000 |
| delta | | -0.184375 |
| outcome | 1.000000 | 0.750000 |
| tested_before_done | 1.000000 | 0.875000 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.750000 |
| mean tool calls | 5.25 | 5.75 |
| mean thinking chars | 1966.5 | 2440.8 |
| zero rollouts | 0 | 1 |
| mean wall-clock s | 35.82 | 46.48 |

## Verdict

Rejected at smoke. Broader no-ECHO hard-negative training did not stabilize
H33's signal. It increased adapter magnitude and trained successfully, but
hurt outcome, tested-before-done, tool-call efficiency, thinking length, and
wall-clock.

Lessons:

- Hard-negative no-ECHO pressure is unsafe for pi-doctest even with broader
  train-only coverage.
- H33's first larger-gate win should be treated as stochastic variance, not a
  reliable data direction.
- The viable systems recipe remains 24-segment checkpointing with short
  trajectories; the data recipe needs to move away from wrong/no-test
  terminal negatives.
