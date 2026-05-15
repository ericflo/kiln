# Hypothesis: meta-what-kind-of-problem-rank8

(Isolate the meta-route at rank 8: does meta-alone hit 1.0 like iter 13,
or is the numeric anchor portion necessary?)

## Claim
32 meta-question examples at rank 8 will lift math score to ≥0.95.
Tests whether the numeric anchor was load-bearing or just incremental.

## Mechanism
Iter 11 (meta alone, rank 4) gave 0.919. Adding rank 8 might push
this further. If the numeric anchor in iter 12/13 was crucial, this
ablation will plateau well below 1.0. If meta alone can climb to ~1.0
at rank 8, the anchor was just a small boost.

## Falsification plan
- S ≥ 0.97: meta alone at rank 8 is nearly as good; anchor was small.
- S in [0.93, 0.97): meta alone strong; anchor adds ~3pp.
- S in [0.90, 0.93): anchor is necessary for the last gains.
- S < 0.90: meta-alone-at-rank-8 underperforms iter 11 (rank 4) —
  overfit at higher rank with 32 examples.
