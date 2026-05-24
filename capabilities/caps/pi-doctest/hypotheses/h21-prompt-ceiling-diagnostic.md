# H21: prompt ceiling diagnostic

## Hypothesis

Before another adapter run, test whether stricter prompt wording can recover
the lost efficiency directly. H19 and H20 both produced attractive
`LIMIT=4` smoke lifts and then regressed on `LIMIT=8`; pi-faithful-completion
also found that prompt ceiling should be established before distillation.

H21 adds an env-driven prompt variant hook and evaluates base-model prompt
variants on the same blind `LIMIT=8 SEEDS=1` aggregate size used for the H19
and H20 promotion checks.

## Variants

Paired base aggregate reused from H19/H20 promotion:

- Composite: 0.8328125.
- Outcome: 0.875.
- Tested-before-done: 1.0.
- Tool-call efficiency: 0.78125.
- Mean tool calls: 5.25.
- Mean thinking chars: 3486.0.
- Mean wall-clock: 49.624973833560944s.
- Zero rollouts: 1.

`PI_DOCTEST_PROMPT_VARIANT=lean-tools`:

- Composite: 0.6359375.
- Delta: -0.196875.
- Outcome: 0.6875.
- Tested-before-done: 1.0.
- Tool-call efficiency: 0.6875.
- Mean tool calls: 6.75.
- Mean thinking chars: 1889.125.
- Mean wall-clock: 35.726569682359695s.
- Zero rollouts: 2.

The lean prompt reduced thinking chars and wall-clock, but it paid by losing
outcome and increasing tool calls. It is not a candidate for distillation.

`PI_DOCTEST_PROMPT_VARIANT=edit-first`:

- Composite: 0.8234375.
- Delta: -0.009375.
- Outcome: 0.9375.
- Tested-before-done: 0.9375.
- Tool-call efficiency: 0.59375.
- Mean tool calls: 6.75.
- Mean thinking chars: 3364.125.
- Mean wall-clock: 53.663738280534744s.
- Zero rollouts: 0.

The edit-first prompt improved nonzero outcome count but created inefficient
tool loops. Composite stayed below base, so it is not a promoted prompt.

Extra instruction only:

`After a doctest run reports 0 failures, immediately reply with DONE. Do not call any tools after a passing doctest.`

- Composite: 0.6484375.
- Delta: -0.184375.
- Outcome: 0.6607142857142857.
- Tested-before-done: 0.9375.
- Tool-call efficiency: 0.6875.
- Mean tool calls: 7.125.
- Mean thinking chars: 3224.125.
- Mean wall-clock: 66.59798011183739s.
- Zero rollouts: 2.

This narrow stop-after-pass emphasis also regressed outcome and efficiency.

## Verdict

Reject H21 as a prompt-ceiling route. None of the tested prompt variants beat
the current default prompt on the larger blind aggregate. The only partially
interesting result is edit-first's zero-rollout reduction, but it loses the
target sub-score badly enough that distilling it would likely repeat the H19
and H20 pattern.

The prompt hook remains useful for future diagnostics, but the next adapter
attempt should use a qualitatively different signal rather than distilling
these variants.

No eval task contents or per-example eval transcripts were inspected.
