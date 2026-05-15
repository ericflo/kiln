# Hypothesis: mixed-prose-numeric-rank2

(H-family per §4. Same iter-3 dataset, rank 2 instead of 4.)

## Claim
Lowering LoRA rank from 4 to 2 on iter 3's winning dataset will hold
the math score within -0.03 of iter 3's 0.935. Tests whether iter 3's
result is rank-bandwidth-limited (would drop more at rank 2) or
saturated below rank 4 (would hold at rank 2).

## Mechanism
Rank 4 was the default; if the patterns iter 3 teaches are simple
enough to fit in rank 2, the result holds. If they need more capacity,
score drops. This is the cheapest probe of the rank dimension.

## Dataset shape
Same as iter 3 (`datasets/mixed-prose-numeric.jsonl`). Copied to
`mixed-prose-numeric-rank2.jsonl` to satisfy slug-based dataset
lookup in train_and_score.sh. 96 examples. Immutable per §3.

## Risk
At rank 2 the LoRA adapter may not have enough rank to represent the
hybrid prose+numeric routing, manifesting as a drop near baseline.

## Falsification plan
- S ≥ 0.91: rank 4 wasn't necessary. Note for compute efficiency.
- S in [0.86, 0.91): rank 2 sufficient but rank 4 has marginal benefit.
- S < 0.86: rank 4 needed. Don't lower further.
