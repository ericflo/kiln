# Hypothesis: code-drill-control-lr-low

(H-family probe responding to the iters 1-4 universal-regression pattern.
Same data as iter 4, learning rate lowered 5×.)

## Claim

Same 32-example code-drill dataset as iter 4, trained at lr 2e-5 instead
of 1e-4, will produce a score at or above iter 4's 0.7205 — and
possibly above baseline (0.8068). Tests the hypothesis that the standard
training settings are overshooting on this high-baseline capability.

## Mechanism

Hypothesis: at lr 1e-4 with rank 4 and 1 epoch, the adapter weights
move too far from base, overwriting useful baseline knowledge. At
lr 2e-5 the same training does 5× smaller per-step updates, leaving
more of the base model intact. If iters 1-4 all regressed because
training is over-corrective, this should at least stop the bleeding.

If S ≥ 0.80, lr was the culprit. Iter 6 explores lower-rank.
If S ≈ iter 4 (~0.72), lr isn't the issue — try rank-2 in iter 6.
If S < iter 4, lower lr makes it worse (unlikely but possible).

## Falsification plan

- S ≥ baseline (0.81): lr 2e-5 prevents regression entirely.
- S in [0.78, 0.81): big improvement over iter 4, near baseline.
- S in [0.72, 0.78): mild improvement, still regression.
- S < 0.72: lr wasn't the issue.
