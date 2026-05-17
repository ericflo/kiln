# Hypothesis: primitives-mix

## Claim
Mixing 64 examples (16 each) of state-tracking (dyna), pattern-induction
(inducto), boolean-circuit-eval (logic), and noise-filtering (filtro)
strengthens the four computational primitives that algorithmic Python
problems rely on, transferring to a code-output eval despite zero code
in the training data.

This mirrors math-broad's winning recipe: train on a "totally different
domain" that teaches an underlying capability, not the surface form.
Variable mutation through ops ≈ Python state mutation. Inducting a
function from few examples ≈ implementing a function from a spec.
Boolean evaluation ≈ predicate composition. Stream filtering ≈ dispatch.

## Mechanism
SFT on symbolic primitives sharpens the FFN/attention routing for the
underlying ops; the eval (code generation) reuses the same circuits.
No surface overlap means no shortcut/memorization.

## Risk
Could regress baseline because the supervision form (symbolic only)
might steer toward terse non-code completions on the eval. Mitigate by
mixing 4 domains so no single style dominates.

## Falsification plan
- Score < 0.84 (below recent floor) → discard, the symbolic supervision
  hurt code generation more than it helped primitives.
- Score in [0.84, 0.87] → marginal, retry with bigger sample or chain-of-thought.
- Score > 0.88 → confirm by reshuffling the same data with seed-2 build.
