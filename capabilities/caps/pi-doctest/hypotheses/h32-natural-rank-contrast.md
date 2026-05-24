# H32 Natural Rank Contrast

## Hypothesis

H26 used a natural successful same-task pair and moved the efficiency axis, but
the original reward spread was tiny. H31 used a strong reward spread, but its
negative traces were wrong terminal attempts and moved behavior slightly worse.
H32 combines the useful parts: natural successful traces only, with the in-pair
preference rescaled to a strong rank contrast.

The intent is to teach "prefer the more efficient passing workflow" without
showing failures, no-test terminal guesses, or synthetic repair loops.

## Data

- Source: `/tmp/pi-doctest-h26-short-policy-pair/candidates/g0_c1_2.jsonl`.
- Dataset: `/tmp/pi-doctest-h32-natural-rank-contrast/grpo-train.rank.g1.jsonl`.
- Groups: 1 train-only same-task pair.
- Completions: 2.
- Original rewards: `[0.9871964285714285, 0.9854464285714286]`.
- Transformed rewards: `[1.0, 0.0]`.
- Base adapter: none.

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

The initial two-pair plan was rejected before training because the second pair
included a 1243-token / 571-action-token completion, recreating H25's timeout
shape.

## Training

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, ECHO lambda 0.05. Gradient
checkpointing was explicitly enabled with `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

Training completed successfully in 264s observed wall-clock, with observed peak
VRAM about 15997 MiB. The receipt reports 260370 ms wall-clock, 254989 ms in
backward, 1047 ms in reference forward, and 4170 ms in policy forward.

Adapter: `pi-doctest-h32-natural-rank-g1-r4a8`.

`kiln adapter verify` passed with 400 nonzero LoRA tensors and delta proxy
0.238848.

## Blind Smoke

`LIMIT=4 SEEDS=1`, paired against `/tmp/pi-doctest-thinking-on-smoke.json`.

| metric | base | H32 |
| --- | ---: | ---: |
| composite | 0.934375 | 0.934375 |
| delta | | 0.000000 |
| outcome | 1.000000 | 1.000000 |
| tested_before_done | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.781250 |
| mean tool calls | 5.25 | 5.25 |
| mean thinking chars | 1966.5 | 1979.25 |
| mean wall-clock s | 35.82 | 36.52 |

## Verdict

Rejected at smoke as neutral. H32 did not reproduce the hard-negative or SFT
regressions, which is useful, but it also did not lift the target metric.
The larger gate was skipped.

Lessons:

- Rank-amplifying a tiny natural preference is safer than failure negatives,
  but the signal is too weak at one group.
- Adding the second natural pair is not currently viable because it exceeds
  the local token/action cap.
- The next data experiment needs a new efficient-success source with naturally
  larger contrast under the same ~800 sequence / ~300 action-token cap.
