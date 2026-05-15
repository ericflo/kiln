# Hypothesis: super-combo (more routes at rank 8)

(Does combining all three T-routes — prose + meta + numeric — at
rank 8 preserve or exceed iter 13's 1.000?)

## Claim
64 prose-approach-broad + 32 meta + 32 numeric (128 examples) at rank
8 will hold at iter 13's 1.000 ± 0.02. Tests whether adding the
prose-route on top of meta+numeric maintains or breaks the saturation.

## Mechanism
Iter 4 (mixed-prose-numeric-bigger at rank 4) regressed because rank
4 was saturated. Iter 7 lifted at rank 8 with 96 examples. This iter
tries 128 examples at rank 8 — same rank, more diverse routes.

## Falsification plan
- S = 1.000: route stacking is harmless / additive at saturation.
- S in [0.98, 1.000): minor regression; bigger data still slightly
  hurts at rank 8.
- S in [0.94, 0.98): meaningful regression; iter 13's minimal recipe
  was optimal.
- S < 0.94: significant route interference at this scale.
