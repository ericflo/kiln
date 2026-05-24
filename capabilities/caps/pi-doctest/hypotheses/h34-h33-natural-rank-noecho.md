# H34 H33 Natural-Rank No-ECHO

## Hypothesis

H33 improved the larger blind gate by making outcomes reliable, but its larger
gate result spent more tool calls and wall-clock. H34 tests a small
efficiency-recovery chain: continue from H33 and train on H32's natural
same-task successful rank pair with ECHO still disabled.

The intended effect is to preserve H33's outcome reliability while nudging
action behavior toward the more efficient successful trace. This is a direct
chain test, not a new first-stage recipe.

## Data

- Source: `/tmp/pi-doctest-h32-natural-rank-contrast/grpo-train.rank.g1.jsonl`.
- Groups: 1 train-only natural successful pair.
- Completions: 2.
- Original rewards: `[0.987196, 0.985446]`.
- Training rewards after rank rescale: `[1.0, 0.0]`.
- Base adapter: `pi-doctest-h33-hardneg-g2-noecho-r4a8`.

Dry-run token shape:

| metric | value |
| --- | ---: |
| groups | 1 |
| completions | 2 |
| action tokens | 579 |
| env tokens | 480 |
| context tokens | 556 |
| max seq length | 814 |
| max action tokens/completion | 299 |

## Training

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, `--no-echo`, with base adapter
`Qwen3.5-4B/adapters/pi-doctest-h33-hardneg-g2-noecho-r4a8`.
Gradient checkpointing was explicitly enabled with
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

Training completed successfully in 318s observed wall-clock, with observed peak
VRAM about 15994 MiB. The receipt reports 314028 ms wall-clock, 310833 ms in
backward, 1057 ms in reference forward, and 1987 ms in policy forward.

Adapter: `pi-doctest-h34-h33-natural-rank-noecho-r4a8`.

`kiln adapter verify` passed with 400 nonzero LoRA tensors and delta proxy
0.343389.

## Blind Smoke

`LIMIT=4 SEEDS=1`, paired against `/tmp/pi-doctest-thinking-on-smoke.json`.

| metric | base | H33 smoke | H34 |
| --- | ---: | ---: | ---: |
| composite | 0.934375 | 0.981250 | 0.750000 |
| delta vs base | | +0.046875 | -0.184375 |
| outcome | 1.000000 | 1.000000 | 0.750000 |
| tested_before_done | 1.000000 | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.937500 | 0.843750 |
| mean tool calls | 5.25 | 3.75 | 4.50 |
| mean thinking chars | 1966.5 | 1585.0 | 1740.8 |
| zero rollouts | 0 | 0 | 1 |
| mean wall-clock s | 35.82 | 28.28 | 32.73 |

## Verdict

Rejected at smoke. H34 slightly improved the base smoke efficiency metrics, but
it destroyed the outcome reliability that H33 had recovered and was much worse
than H33 on every headline smoke metric. The larger gate was skipped.

Lessons:

- Chaining a tiny natural-rank preference on top of H33 is not a safe
  efficiency-recovery move; it can erase the reliability benefit.
- The one-pair natural-rank signal remains too weak or too narrow even when
  chained from a stronger base adapter.
- The systems result is still useful: `KILN_GRAD_CHECKPOINT_SEGMENTS=24` fit
  the chained policy-on run at about 15.99 GiB peak, but the 318s runtime is
  dominated by checkpoint recompute.
