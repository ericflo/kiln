# H40: low-dose short thinking compression

## Hypothesis

H38's one-group short thinking-compression dataset produced a real smoke lift
but failed the larger promotion check by damaging outcome reliability. H40
tests whether that failure was caused by too much one-task pressure rather
than by the data shape itself.

It reuses the H38 train-only successful-workflow compression pair, but lowers
the adapter dose:

- rank 4, alpha 4 (`alpha/rank = 1`);
- lr `1e-6`;
- no ECHO;
- same base model, no base adapter.

This combines the pi-faithful-completion lesson that gentle updates are safer
than large LoRA scaling with the H38 observation that short successful-workflow
compression can move smoke metrics in the right direction.

## Data

Dataset:
`/tmp/pi-doctest-h38-think-compress-short/grpo-train.think-compress.short.jsonl`.

This is the same train-only one-group dataset used by H38:

- 1 group, 2 completions.
- Rewards `[1.0, 0.0]`.
- 337 action tokens.
- 448 env tokens.
- 556 context tokens.
- Max sequence length 763.
- Max action tokens per completion 261.

Dry-run passed with reward mean 0.5, stdev 0.5, and `alpha_over_rank=1`.

## Training

Adapter: `pi-doctest-h40-think-compress-lowdose-noecho-r4a4-lr1e6`.

Training used `KILN_GRAD_CHECKPOINT_SEGMENTS=24` and completed successfully.
The receipt reports:

- wall-clock: 63458 ms (67.409s observed by the CLI);
- rank 4, alpha 4, lr `1e-6`;
- 1 group, 2 completions;
- 337 action tokens, 448 env tokens, 556 context tokens;
- 923.704 ms reference forward;
- 2060.845 ms policy forward;
- 60332.070 ms backward.

The CLI reported peak VRAM 15972 MiB. Adapter verify passed with rank 4,
alpha 4, 400 nonzero tensors, 200 projection pairs, and LoRA update L2
upper-bound 0.05225969191155655.

## Smoke

Blind `LIMIT=4 SEEDS=1` smoke rejected H40:

- Base composite: 0.9343750000000001.
- H40 composite: 0.9.
- Delta: -0.034375000000000044.
- Outcome: 1.0.
- Tested-before-done: 0.875.
- Format compliance: 1.0.
- Tool-call efficiency: 0.75.
- Mean tool calls: 5.25.
- Mean thinking chars: 2893.25.
- Mean wall-clock: 46.013746440410614s.
- Zero rollouts: 0.

The lower-dose update did reduce adapter magnitude and train time, but it did
not preserve H38's smoke lift. It also introduced a tested-before-done
regression that H38 did not show at smoke.

## Verdict

Rejected at smoke.

This falsifies the "H38 only failed because the update was too strong"
explanation. The short-compression pair itself is unstable: full-dose H38
could create a smoke false positive, while low-dose H40 weakened the intended
effect without removing reliability regressions. Future compression work
should change the data distribution, not only the update dose.

No eval task contents or per-example eval transcripts were inspected.
