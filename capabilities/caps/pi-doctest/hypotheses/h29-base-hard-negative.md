# H29 Base Hard-Negative Contrast

## Hypothesis

H28 proved that the five-task hard-negative contrast can train locally when
chained from H26, but the larger gate rejected the resulting adapter. H29
isolates the data signal by training the same broad contrast fresh from base,
with no base adapter.

The intended test was whether the hard-negative data itself is useful, or
whether H28's failure came mainly from stacking onto a rejected efficiency
adapter.

## Data

- Source: `capabilities/caps/pi-doctest/datasets/train.tasks.jsonl`.
- Dataset: `/tmp/pi-doctest-h29-base-hardneg/grpo-train.hardneg.jsonl`.
- Groups: 5 train tasks (`common`, `prime_length`, `largest_prime_factor`,
  `count_distinct_characters`, `sum_to_n`).
- Completions: 10, with rewards `[1.0, 0.0]` in each group.
- Reward stdev: 0.5.
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

## Training Attempt

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, ECHO lambda 0.05. Gradient
checkpointing was explicitly enabled with `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

The dry-run passed, but full training did not finish within the 900s guard.
The run completed four of five groups and timed out during group 5 reference
forward, before an adapter artifact or train receipt was saved. Observed CUDA
memory was about 15978 MiB while compute-bound.

The checkpointing/recompute cost was unexpectedly high fresh from base:

| observed section | elapsed |
| --- | ---: |
| group 4 reference forward | 176070 ms |
| group 4 first policy forward | 211237 ms |
| group 4 first backward | 119011 ms |
| group 4 second backward | 136273 ms |

Adapter: none.

## Blind Eval

No blind eval was run because no adapter was produced.

## Verdict

Rejected as a local training route at this five-group shape. The same token
shape that trained in 270s when chained from H26 was too slow from base under
24-segment checkpointing and timed out before adapter write.

Lessons:

- Gradient checkpointing is necessary for VRAM, but it can make policy-on GRPO
  impractically slow even at sequence lengths near 550 when the reference and
  policy passes recompute heavily.
- Fresh-from-base hard-negative training should be reduced before retrying:
  fewer groups, lower sequence/action-token cap, no-policy-loss ECHO
  preconditioning, or a staged two-step recipe with a very small first pass.
- H29 gives no behavioral evidence for or against the hard-negative data,
  only a throughput result. Do not compare it to H28's scores.
