# Hypothesis: prose-algo-approach-anchored

(§11 answer-form fix on iter 1's regression. Pure prose collided with
enable_thinking=False; add code-form anchors.)

## Claim

32 prose examples + 8 short code-anchor examples (40 total) will
restore the eval's expected output form while preserving any
algorithmic-routing benefit from the prose. Expected S in [0.81, 0.88].

If S > 0.86 (>2× variance over baseline 0.807): anchored prose lifts.
If S ≈ baseline ± variance: prose adds nothing once form is anchored.
If S < baseline: prose still hurts even with anchors — the prose route
itself is wrong for code-output evals.

## Mechanism

Per §11: small amount (20%) of anchor examples that match the eval's
expected output form preserves the model's tendency to produce that
form, while the bulk of the dataset (80% prose) does the conceptual
teaching. Math-broad's iter 3 mixed-prose-numeric was the canonical
form of this pattern.

The 8 code anchors here are deliberately TRIVIAL Python functions —
reverse, sum, count, factorial, fibonacci, etc. — short enough to
not teach algorithms but long enough to anchor "produce a Python
function as the answer."

## Dataset shape

- Size: 40 (32 prose + 8 code), in that order
- Prose half: identical to iter 1 (immutable per §3)
- Code half: 8 functions, 5-15 lines each, simple problems
- Concatenated prose-first / code-last; deterministic order

## Falsification plan

- S ≥ 0.85: anchors did fix the form drift; prose route is alive.
  Next iter scales prose / tries meta-question variant.
- S in [0.80, 0.85): partial fix but no real lift. Try meta-question
  next (which dramatically beat prose in math-broad).
- S in [0.74, 0.80): anchors didn't fully fix the regression. Try
  with MORE anchors (12 or 16 instead of 8).
- S < 0.74: prose route is structurally wrong for this capability.
  Retire prose; iter 3 pivots to meta-question.
