# H46: realistic ECHO workflow

## Hypothesis

H45 trained a compact multi-step workflow but omitted the intervening tool
observations, so the completion shape was unnatural for an agentic task. H46
keeps H45's edit-form token discipline while restoring realistic
assistant/tool alternation and ECHO:

- preferred: `edit` -> edit result -> doctest -> doctest result -> `DONE`;
- rejected: `edit` -> edit result -> `DONE` without verification;
- weak rejected: `edit` -> edit result -> doctest -> doctest result ->
  redundant doctest -> doctest result -> `DONE`.

This tests whether H45's smoke failure was caused by missing tool observations
and disabled ECHO rather than by the edit-form workflow signal itself.

## Data

Dataset:
`/tmp/pi-doctest-h46-realistic-echo-workflow/grpo-train.realistic-edit-echo.g2.jsonl`.

Source:
`/tmp/pi-doctest-h44-broad-success-sft/sft.train.jsonl`.

The data used the first two compact H44 success examples. Each group kept the
real train-only read context and converted the full-file `write` action into a
minimal `edit` that replaces `raise NotImplementedError`. Unlike H45,
completion trajectories included the real edit success observation and real
doctest output observations.

Dry-run shape:

- 2 groups.
- 6 completions.
- Rewards per group: 1.0, 0.0, 0.65.
- 654 action tokens.
- 756 env tokens.
- 2214 context tokens.
- Reward stdev: 0.414327.
- ECHO lambda: 0.05.

## Training

Adapter:
`pi-doctest-h46-realistic-echo-r4a4lr1e6`.

Training used Phase 1 GRPO, rank 4 / alpha 4 / lr `1e-6`, ECHO lambda 0.05,
and `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

Training completed successfully:

- elapsed: 698.235s observed;
- receipt wall clock: 694140 ms;
- peak observed VRAM: 16007 MiB;
- groups trained: 2;
- completions trained: 6;
- ECHO env CE: 1.91345 -> 1.35367.

Adapter verify passed. The verifier found 400 nonzero LoRA tensors and LoRA
update L2 upper-bound 0.055388.

## Smoke

Blind aggregate `LIMIT=4 SEEDS=1` smoke rejected H46.

Paired base:

- Composite: 0.953125.
- Mean wall-clock: 26.2375s.
- Zero rollouts: 0.

H46:

- Composite: 0.94375.
- Delta: -0.009375.
- Mean wall-clock: 38.3502s.
- Zero rollouts: 0.

No promotion check was run because the smoke gate was still negative and
slower than base.

## Verdict

Reject H46, but keep the lesson.

Restoring realistic tool observations and ECHO reduced H45's smoke regression
substantially (-0.09375 -> -0.009375), and all rollouts stayed nonzero.
However, it still did not beat paired base and it added latency. This suggests
the remaining harmful piece is the explicit no-test/retest contrast, not only
the lack of ECHO.

Next attempts should preserve H46's realistic trajectory structure but avoid
training against `DONE`-without-test. Use only outcome-preserving pairs, or
make the rejected completion a verified but slightly inefficient variant, so
the policy update does not push on verification reliability.

No eval task contents or per-example eval transcripts were inspected.
