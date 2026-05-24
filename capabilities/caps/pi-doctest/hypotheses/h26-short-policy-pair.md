# H26: Short policy pair

## Hypothesis

H25 showed that policy-on baseline GRPO+ECHO is locally infeasible for
1000+ token trajectories, but H24 showed that an 814-token trajectory can train
with checkpointing in about two minutes. H26 mines the shortest same-task
two-completion reward-spread pair from the H17 train data, keeping each
completion near 800 sequence tokens and under 300 action tokens.

The goal is to keep the useful part of H25, direct action-side preference
pressure, while avoiding the long-sequence lower-layer backward cliff.

## Data

Source dataset:
`/tmp/pi-doctest-h17-action650-success-anchors/grpo-train.jsonl`.

Candidate mining wrote:
`/tmp/pi-doctest-h26-short-policy-pair/candidates.json`.

Selected dataset:
`/tmp/pi-doctest-h26-short-policy-pair/candidates/g0_c1_2.jsonl`.

Selected pair:

- source group index: 0
- completion indices: 1 and 2
- rewards: 0.9871964285714285 and 0.9854464285714286
- reward range: 0.00175
- reward stdev: 0.000875

Dry-run command used:

- `--mode baseline`
- policy loss enabled
- ECHO lambda `0.05`
- rank 4 / alpha 8 / lr `5e-6`
- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`

Dry-run passed:

- 1 valid group.
- 2 valid completions.
- completion sequence lengths: 814 and 801.
- completion action tokens: 280 and 299.
- total action tokens: 579.
- total env tokens: 480.
- total context tokens: 556.

The reward diagnostic correctly warned that the group is saturated and has tiny
variance. This is methodologically risky, but it isolates whether the shorter
local training shape works at all.

## Training

Training succeeded locally in 308.312 seconds and installed:

`Qwen3.5-4B/adapters/pi-doctest-h26-short-pair-r4a8`

Training details:

- mode: baseline
- policy loss enabled
- ECHO lambda `0.05`
- rank 4 / alpha 8 / lr `5e-6`
- 24 checkpoint segments
- final loss: 0.189738
- ECHO env CE: 3.7462837119897205
- backward time: 301664 ms
- peak observed VRAM: 15970 MiB

Per-completion timing proved the local cap:

- completion 1: seq len 814, action tokens 280, backward 127265 ms.
- completion 2: seq len 801, action tokens 299, backward 174398 ms.

`kiln adapter verify` passed offline and through the running server:

- 400 tensors.
- 200 LoRA projection pairs.
- nonzero LoRA tensors found.
- delta proxy L2 upper bound: 0.238775.

## Eval

Blind smoke: `LIMIT=4 SEEDS=1`, compared to
`/tmp/pi-doctest-thinking-on-smoke.json`.

- base composite: 0.934375
- H26 composite: 0.953125
- delta: +0.01875
- outcome: 1.0
- tested-before-done: 1.0
- tool-call efficiency: 0.84375
- mean tool calls: 4.5
- mean thinking chars: 2132.75

This cleared the cheap smoke, but H19/H20 already showed that `LIMIT=4` can be
a false-positive gate for this cap.

Promotion check: `LIMIT=8 SEEDS=1`, compared to
`/tmp/pi-doctest-h19-promo-base8.json`.

- base composite: 0.8328125
- H26 composite: 0.7796875
- delta: -0.053125
- outcome: 0.8125
- tested-before-done: 1.0
- tool-call efficiency: 0.828125
- mean tool calls: 4.875
- mean thinking chars: 2962.25
- zero rollouts: 1

H26 improved tool efficiency relative to the paired base but hurt outcome
enough to lose composite on the larger gate.

## Verdict

Reject H26 for promotion.

This is still the first policy-on local training shape that completed and
moved the targeted efficiency sub-score in the larger gate. The remaining gap
is data quality, not raw throughput: the same-task reward spread was tiny and
saturated, so the action preference was too weak and slightly harmful to
outcome. The next policy experiment should keep the H26 length caps but use a
stronger action-side signal: mine or synthesize concise successes with a larger
quality contrast, or use an oscillating SFT chain where concise action traces
are alternated with outcome-preserving traces.

No eval task contents or per-example eval transcripts were inspected.
