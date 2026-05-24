# H31 Base Hard-Negative G2

## Hypothesis

H29 showed that five fresh-from-base hard-negative groups are too expensive,
and H30 showed that reducing checkpoint segments is the wrong throughput
direction. H31 keeps the known-good 24-segment checkpointing configuration and
reduces the data shape instead: two fresh-from-base hard-negative groups.

The behavioral hypothesis is the same as H29's, but scaled down: concise
verified pass traces versus wrong no-test terminal guesses may improve
tool-call efficiency without inheriting H26's rejected adapter state.

## Data

- Source: `/tmp/pi-doctest-h29-base-hardneg/grpo-train.hardneg.jsonl`.
- Effective dataset: `/tmp/pi-doctest-h29-base-hardneg/grpo-train.hardneg.max2.jsonl`.
- Selection: `--max-groups 2`.
- Base adapter: none.
- Groups: 2.
- Completions: 4.
- Rewards: `[1.0, 0.0]` in each group.
- Reward stdev: 0.5.

Dry-run token shape:

| metric | value |
| --- | ---: |
| groups | 2 |
| completions | 4 |
| action tokens | 389 |
| env tokens | 526 |
| context tokens | 1084 |
| max seq length | 554 |
| max action tokens/completion | 150 |

## Training

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, ECHO lambda 0.05. Gradient
checkpointing was explicitly enabled with `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

Training completed successfully in 77s observed wall-clock, with observed peak
VRAM about 15997 MiB. The receipt reports 73531 ms wall-clock, 70565 ms in
backward, 1824 ms in reference forward, and 932 ms in policy forward.

Adapter: `pi-doctest-h31-base-hardneg-g2-r4a8`.

`kiln adapter verify` passed with 400 nonzero LoRA tensors and delta proxy
0.248275.

## Blind Smoke

`LIMIT=4 SEEDS=1`, paired against `/tmp/pi-doctest-thinking-on-smoke.json`.

| metric | base | H31 |
| --- | ---: | ---: |
| composite | 0.934375 | 0.925000 |
| delta | | -0.009375 |
| outcome | 1.000000 | 1.000000 |
| tested_before_done | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.750000 |
| mean tool calls | 5.25 | 5.50 |
| mean thinking chars | 1966.5 | 2153.25 |
| mean wall-clock s | 35.82 | 42.35 |

## Verdict

Rejected at smoke. H31 kept outcome, tested-before-done, and format at 1.0,
but it slightly worsened the target efficiency axis and wall-clock. The larger
gate was skipped.

Lessons:

- Keeping 24 checkpoint segments and reducing to two groups solves the local
  throughput problem.
- The base hard-negative data still does not improve the target behavior at
  this scale; it nudges toward more tool use and more thinking.
- Next attempts should either use a different negative type or train a cheaper
  no-policy-loss/ECHO preconditioner before another policy-on test.
