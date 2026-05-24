# H30 Checkpoint Segment Sweep

## Hypothesis

H29 showed that the five-group fresh-from-base hard-negative run fits in VRAM
with 24 checkpoint segments but is too slow under the 900s guard. H30 tests a
simple systems hypothesis: fewer checkpoint segments might reduce recompute
overhead enough to make a smaller fresh-from-base run practical.

## Data

- Source: `/tmp/pi-doctest-h29-base-hardneg/grpo-train.hardneg.jsonl`.
- Effective dataset: `/tmp/pi-doctest-h29-base-hardneg/grpo-train.hardneg.max2.jsonl`.
- Selection: `--max-groups 2`, the first two train-only hard-negative groups.
- Base adapter: none.
- Completions: 4.

## Training Attempt

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, ECHO lambda 0.05.

The run set `KILN_GRAD_CHECKPOINT_SEGMENTS=12`. The trainer confirmed 12
segments with 2- and 3-layer checkpoint blocks. It fit in memory, with the
first logged CUDA gate at about 16376 MiB, but the lower-layer recompute blocks
were slower than the 24-segment shape:

| segment | layers | elapsed |
| --- | --- | ---: |
| 8 | 21-24 | 4685 ms |
| 7 | 18-21 | 7894 ms |
| 6 | 15-18 | 6786 ms |
| 5 | 12-15 | 82506 ms |
| 4 | 9-12 | 14172 ms |
| 3 | 6-9 | 15183 ms |
| 2 | 3-6 | 9209 ms |

The run was manually stopped during the first completion's backward pass before
any adapter artifact was written.

## Blind Eval

No blind eval was run because no adapter was produced.

## Verdict

Rejected as a checkpointing configuration. Twelve segments are worse for this
model/trainer path: the larger multi-layer checkpoint blocks produce
pathological lower-layer recompute times. Keep the 24-segment configuration for
local policy-on GRPO unless a separate profiler result shows otherwise.

Next throughput work should reduce data/token shape, not reduce checkpoint
segments. A 32-segment test may be worth trying later if 24-segment runs still
hit slow multi-layer sections, but 12 is not the direction.
