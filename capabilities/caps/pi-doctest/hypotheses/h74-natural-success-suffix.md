# H74 Natural Success Suffix

## Hypothesis

H74 tested whether the useful part of H32 and H54 could be combined without
their failure modes. H32 used natural same-task successful trajectories and was
safe but neutral. H54 used a locally trainable one-action suffix representation
but relied on synthetic concise-vs-verbose contrasts and later failed
confirmation.

The H74 data used train-only successful rollouts from the H15 strong-ID
collection. For each selected task, it chose a shorter successful completion
over a longer successful completion, then decomposed the preference into
step-local suffix groups. This avoids failed rollouts, wrong edits, and
premature `DONE` negatives while still applying efficiency pressure.

## Data

Source:
`/tmp/pi-doctest-h15-thinking-on-pi1024-normalized-rollouts/grpo-train.partial32.jsonl`.

Two source groups had at least two successful completions and a meaningful
natural length contrast:

- source group 1: preferred completion 3, rejected completion 1, 1762 vs 6752
  chars, ratio 3.83
- source group 3: preferred completion 3, rejected completion 1, 1390 vs 2697
  chars, ratio 1.94

The first generated shape had seven suffix groups, but dry inspection showed
13,262 context tokens, so it was narrowed before training. The trained-attempt
dataset was:

`/tmp/pi-doctest-h74-natural-success-suffix/grpo-train.natural-success-suffix.g2x2-soft.jsonl`

It kept only post-read edit and final `DONE` suffixes for each source group,
with a softened reward gap of 1.0 vs 0.8.

Dry-run passed:

| metric | value |
| --- | ---: |
| groups | 4 |
| completions | 8 |
| action tokens | 348 |
| env tokens | 0 |
| context tokens | 7580 |
| reward stdev | 0.100000 |
| data hash | `sha256:54f87530e76ade6247266ad58417850332253a1874ce05168e43ff94971e80b0` |

## Training Attempt

Training command family:

`cuda_grpo_ablation --mode phase1 --rank 4 --alpha 4 --lr 2e-7 --no-echo`

Environment:

- `KILN_CUDA_ARCHS=86`
- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`

The run memory-fit and reached one progress line:

`progress step=9674/26211 loss=-0.000489 vram_mib=15992`

It was then allowed to continue until roughly 37 minutes observed wall-clock
with GPU utilization still pegged and no adapter files written. The run was
terminated manually and no adapter was installed or evaluated.

## Verdict

Rejected as `aborted-throughput`.

Natural success-only suffix ranking is conceptually safer than the previous
failure-negative suffixes, but the long-prefix representation is not practical
locally in this form. The key lesson is that action-token count is not enough:
prefix context dominates backward time for suffix datasets. A follow-up should
compress or canonicalize the prior successful context before suffix training,
or gather fresh short successful rollouts instead of slicing long thinking-on
transcripts.
