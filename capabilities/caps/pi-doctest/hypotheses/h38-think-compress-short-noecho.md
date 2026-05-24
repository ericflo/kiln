# H38 Short Think-Compress No-ECHO

## Hypothesis

H37 showed that successful-trace thinking compression is behaviorally safer
than wrong/no-test hard negatives, but the two-group dataset failed local
throughput because the second original trace reached 1010 sequence tokens.
H38 keeps the same idea but uses one short train-only successful trace:
compressed-thinking preferred over the original successful trace.

The goal is to train shorter internal reasoning while preserving the
read/edit/doctest/DONE workflow.

## Data

- Source: `/tmp/pi-doctest-h17-action650-success-anchors/grpo-train.jsonl`.
- Output: `/tmp/pi-doctest-h38-think-compress-short/grpo-train.think-compress.short.jsonl`.
- Groups: 1.
- Completions: 2.
- Rewards: `[1.0, 0.0]`, where `1.0` is the compressed-thinking variant and
  `0.0` is the original successful trace.
- Base adapter: none.

Selection:

| source line | completion | original reward | original chars | compressed chars |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 0.986482 | 1027 | 363 |

Dry-run token shape:

| metric | value |
| --- | ---: |
| groups | 1 |
| completions | 2 |
| action tokens | 337 |
| env tokens | 448 |
| context tokens | 556 |
| max seq length | 763 |
| max action tokens/completion | 261 |

## Training

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, `--no-echo`. Gradient checkpointing
was explicitly enabled with `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

Training completed successfully in 102s observed wall-clock, with observed peak
VRAM about 15983 MiB. The receipt reports 97622 ms wall-clock, 92394 ms in
backward, 934 ms in reference forward, and 4164 ms in policy forward.

Adapter: `pi-doctest-h38-think-compress-short-noecho-r4a8`.

`kiln adapter verify` passed with 400 nonzero LoRA tensors and delta proxy
0.255293.

## Blind Smoke

`LIMIT=4 SEEDS=1`, paired against `/tmp/pi-doctest-thinking-on-smoke.json`.

| metric | base | H38 |
| --- | ---: | ---: |
| composite | 0.934375 | 0.962500 |
| delta | | +0.028125 |
| outcome | 1.000000 | 1.000000 |
| tested_before_done | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.875000 |
| mean tool calls | 5.25 | 4.25 |
| mean thinking chars | 1966.5 | 2423.0 |
| zero rollouts | 0 | 0 |
| mean wall-clock s | 35.82 | 41.71 |

## Larger Gate

`LIMIT=8 SEEDS=1`, paired against `/tmp/pi-doctest-h19-promo-base8.json`.

| metric | base | H38 |
| --- | ---: | ---: |
| composite | 0.832812 | 0.731250 |
| delta | | -0.101562 |
| outcome | 0.875000 | 0.750000 |
| tested_before_done | 1.000000 | 0.937500 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.812500 |
| mean tool calls | 5.25 | 4.875 |
| mean thinking chars | 3486.0 | 3362.1 |
| zero rollouts | 1 | 2 |
| mean wall-clock s | 49.62 | 63.41 |

## Verdict

Rejected for promotion. H38 confirms the short thinking-compression shape is
locally trainable and can improve smoke tool efficiency, but the larger gate
lost outcome reliability and did not materially solve thinking efficiency.

Lessons:

- One-task thinking compression overfits the smoke slice and is not stable.
- The data direction is still safer than hard negatives, but it needs broader
  short successful traces before another policy-on attempt.
- Smoke remains a throughput check only; larger-gate failure pattern persists.
