# Hypothesis: mixed-meta-numeric-rank8

(H-family on the new best. Iter 7 showed rank 4 → 8 lifted mixed-prose-numeric by +0.017.
Same probe on the meta-route winner.)

## Claim

Doubling LoRA rank from 4 to 8 on iter 12's winning recipe will lift
the math score above iter 12's 0.984 — though headroom is small (only
1 wrong out of 62, i.e. 1.6% to gain at most). Expected: S in [0.985, 1.000].

## Mechanism
Same as iter 7: more rank = more parameters, potential to learn more
patterns. Iter 7 saw a +0.017 lift from this move on the prose+numeric
recipe; here we have less headroom but more efficient supervision, so
direction is unclear.

## Dataset shape
Same 64 examples as iter 12 (`datasets/mixed-meta-numeric-rank8.jsonl`,
copy of mixed-meta-numeric.jsonl).

## Risk
At rank 8 with only 64 examples, overfitting risk is higher than at
rank 4. Anchor regression would flag this.

## Falsification plan
- S = 1.000: perfect. Cap at this recipe.
- S in [0.985, 0.999]: small additional lift; rank 8 marginal benefit.
- S in [0.970, 0.985): no gain over rank 4; recipe saturated at rank 4.
- S < 0.970: rank 8 overfits 64 examples; rank 4 was optimal.
- Anchor < 0.97: stylistic clobber.
