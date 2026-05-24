# H33 Hard-Negative G2 No-ECHO

## Hypothesis

H31 trained the two-group hard-negative contrast with ECHO enabled and regressed
the target efficiency axis. One plausible failure mode is that ECHO trains
environment-token prediction on both high- and low-reward trajectories, so even
a low-reward negative trace can still teach the model its tool-observation
pattern.

H33 reuses the exact H31 dataset but disables ECHO. This is not a proposed new
default for agentic-GRPO; it is a targeted ablation to test whether the
hard-negative action preference works better without env-CE imitation of the
negative traces.

## Data

- Source: `/tmp/pi-doctest-h29-base-hardneg/grpo-train.hardneg.max2.jsonl`.
- Groups: 2 train-only hard-negative groups.
- Completions: 4.
- Rewards: `[1.0, 0.0]` in each group.
- Base adapter: none.

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
learning rate 5e-6, policy loss enabled, `--no-echo`. Gradient checkpointing
was explicitly enabled with `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

Training completed successfully in 125s observed wall-clock, with observed peak
VRAM about 15963 MiB. The receipt reports 121586 ms wall-clock, 118863 ms in
backward, 1858 ms in reference forward, and 652 ms in policy forward.

Adapter: `pi-doctest-h33-hardneg-g2-noecho-r4a8`.

`kiln adapter verify` passed with 400 nonzero LoRA tensors and delta proxy
0.256303.

## Blind Smoke

`LIMIT=4 SEEDS=1`, paired against `/tmp/pi-doctest-thinking-on-smoke.json`.

| metric | base | H33 |
| --- | ---: | ---: |
| composite | 0.934375 | 0.981250 |
| delta | | +0.046875 |
| outcome | 1.000000 | 1.000000 |
| tested_before_done | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.937500 |
| mean tool calls | 5.25 | 3.75 |
| mean thinking chars | 1966.5 | 1585.0 |
| mean wall-clock s | 35.82 | 28.28 |

## Larger Gate

`LIMIT=8 SEEDS=1`, paired against `/tmp/pi-doctest-h19-promo-base8.json`.

| metric | base | H33 |
| --- | ---: | ---: |
| composite | 0.832812 | 0.892188 |
| delta | | +0.059375 |
| outcome | 0.875000 | 1.000000 |
| tested_before_done | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.640625 |
| mean tool calls | 5.25 | 6.75 |
| mean thinking chars | 3486.0 | 3313.5 |
| zero rollouts | 1 | 0 |
| mean wall-clock s | 49.62 | 62.65 |

## Verdict

Kept with caveat. H33 clears the larger gate on composite by converting the
base slice's outcome miss into all-success rollouts, but it does so by spending
more tool calls and more wall-clock. This is an outcome-reliability adapter,
not the efficiency adapter originally targeted.

Lessons:

- Disabling ECHO on hard-negative data changed the failure mode and produced a
  real larger-gate composite gain. The ECHO-on version (H31) regressed at
  smoke.
- The smoke signal was directionally right but overstated the efficiency gain;
  the larger gate shows the adapter is more reliable but less efficient.
- Next work should either confirm H33 with an additional seed or chain a small
  efficiency-recovery stage on top of H33 while preserving the outcome gain.
