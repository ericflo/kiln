# Hypothesis: mixed-meta-numeric-shuffle1

(§5 small-N noise-floor confirmation on the new best. Iter 9's shuffle
on the iter-3 recipe swung -0.048; we need a noise floor for the new
winner before claiming small refinements above it.)

## Claim

Re-training iter 12's exact recipe with examples shuffled (fixed seed
'yes 7') will produce a score within ±0.05 of 0.984. If the score
drops by more than 0.05, the iter-12 result is order-sensitive and
needs additional replicates before confident claims.

## Mechanism
Per §15 ("Same dataset, shuffled order, different score"), SFT at low
rank is order-sensitive. Establishes the order-noise floor for this
recipe.

## Dataset shape
Same 64 examples as iter 12, shuffled.

## Falsification plan
- |S - 0.984| ≤ 0.03: stable; iter 12 win is robust.
- |S - 0.984| in (0.03, 0.05]: moderate sensitivity; iter 12 partly
  noise — run two more shuffles.
- |S - 0.984| > 0.05: high sensitivity; iter 12 is order-dependent.
