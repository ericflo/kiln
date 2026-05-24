# H42: low-dose post-pass stop contrast

## Hypothesis

H41 showed that post-pass stopping is smoke-positive but too narrow as a
full-dose adapter update. H42 tests whether the same terminal stop contrast can
keep the smoke win without damaging broader reliability if the LoRA update is
made much smaller.

The only recipe change from H41 is dose:

- rank 4, alpha 4 (`alpha/rank = 1`);
- lr `1e-6`;
- no ECHO;
- same base model, no base adapter.

Thinking stays enabled. This tests update magnitude, not a no-thinking route.

## Data

Dataset:
`/tmp/pi-doctest-h41-postpass-stop-g4/grpo-train.postpass-stop.g4.jsonl`.

This reuses the H41 train-only post-pass-stop corpus:

- 4 groups, 8 completions.
- Rewards `[1.0, 0.0]` per group.
- 260 action tokens.
- 0 env tokens.
- 2970 context tokens.
- Max sequence length 452.
- Max action tokens per completion 45.
- Reward mean 0.5, stdev 0.5.

Dry-run passed with `alpha_over_rank=1`.

## Training

Adapter: `pi-doctest-h42-postpass-stop-lowdose-noecho-r4a4-lr1e6`.

Training used `KILN_GRAD_CHECKPOINT_SEGMENTS=24` and completed successfully.
The receipt/logs report:

- wall-clock: 273316 ms in the receipt, 277.064s observed by the CLI;
- rank 4, alpha 4, lr `1e-6`;
- 4 groups, 8 completions;
- 260 action tokens, 0 env tokens, 2970 context tokens;
- 21675.556 ms reference forward;
- 14420.151 ms policy forward;
- 236851.493 ms backward;
- final loss 0.002881.

The CLI reported peak VRAM 15983 MiB. Adapter verify passed with rank 4,
alpha 4, 400 nonzero tensors, 200 projection pairs, and LoRA update L2
upper-bound 0.15714548112668084. This is much smaller than H41's 0.787006
proxy, so H42 did test the intended lower-impact update.

## Smoke

Blind `LIMIT=4 SEEDS=1` smoke was positive:

- Base composite: 0.9343750000000001.
- H42 composite: 0.971875.
- Delta: +0.03749999999999998.
- Outcome: 1.0.
- Tested-before-done: 1.0.
- Format compliance: 1.0.
- Tool-call efficiency: 0.90625.
- Mean tool calls: 4.25.
- Mean thinking chars: 1670.75.
- Mean wall-clock: 34.3766952753067s.
- Zero rollouts: 0.

The lower-dose adapter preserved H41's smoke-level efficiency signal.

## Promotion Check

Blind `LIMIT=8 SEEDS=1` rejected H42:

- Base composite: 0.8328125.
- H42 composite: 0.5875.
- Delta: -0.24531249999999993.
- Outcome: 0.625.
- Tested-before-done: 0.9375.
- Format compliance: 1.0.
- Tool-call efficiency: 0.8125.
- Mean tool calls: 5.125.
- Mean thinking chars: 2815.75.
- Mean wall-clock: 72.69190508127213s.
- Zero rollouts: 3.

Tool efficiency remained above base, but outcome reliability collapsed. The
smaller update did not make the narrow terminal-stop corpus generalize.

## Verdict

Rejected at promotion.

H42 falsifies "H41 only failed because the update was too strong." The
post-pass-stop behavior is still a useful target, but training only terminal
contexts creates a smoke false positive and harms the broader gate. Next
attempts should stop using terminal-only data and instead build a balanced
workflow dataset that includes pre-pass read/write/verify contexts plus
post-pass stop, or switch away from adapter training for this micro-behavior.

No eval task contents or per-example eval transcripts were inspected.
