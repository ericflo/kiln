# Hypothesis: mixed-prose-numeric-shuffle1

(§5 small-N noise-floor confirmation. Same iter-3 dataset, shuffled.)

## Claim
Shuffling the order of examples in iter-3's dataset will produce a
score within ±0.02 of iter 3's 0.935. Tests training-order noise per
§15 "Same dataset, shuffled order, different score" surprising pattern.

## Mechanism
SFT is sensitive to example order at low rank. A 0.02-0.05 swing from
shuffle alone is normal. This iter establishes the *order-noise floor*
for iter 3's recipe, so future small-delta comparisons can be judged
against it.

## Dataset shape
Same content as iter 3, different order. Shuffled with `shuf
--random-source` and a fixed seed (yes 42) for reproducibility.

## Risk
If shuffle produces a wildly different score (>0.05 swing), iter 3's
result was order-dependent and the "winner" is partially noise. Would
prompt running multiple shuffles to characterise the distribution.

## Falsification plan
- |S - 0.935| ≤ 0.02: order-noise floor confirmed small. Iter 3 stable.
- |S - 0.935| in (0.02, 0.05]: moderate order sensitivity; future
  small-delta claims (e.g. ±0.02) cannot be trusted without re-shuffle.
- |S - 0.935| > 0.05: high order sensitivity. Run two more shuffles.
