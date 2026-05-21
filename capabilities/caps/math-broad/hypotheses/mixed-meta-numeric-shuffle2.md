# Hypothesis: mixed-meta-numeric-shuffle2

(Second-seed shuffle confirm on the iter 13 winner.)

## Claim
Same iter-13 recipe with a different shuffle seed (99) will land within
0.03 of 1.000. If both shuffles produce ≥0.97, iter 13 is robust to
training order at saturation.

## Falsification plan
- S ≥ 0.97: robust at the saturation ceiling.
- S in [0.92, 0.97): mild order sensitivity at the ceiling.
- S < 0.92: significant order sensitivity; iter 13 was lucky shuffle.
