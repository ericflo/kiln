# Hypothesis: mixed-meta-numeric-lr-high

(LR sensitivity test: 2x the learning rate.)

## Claim
Same iter-13 recipe at lr 2e-4 will land within 0.05 of 1.000. Higher
lr may slightly destabilize or may not matter at 1-epoch + rank 8.

## Falsification plan
- S ≥ 0.98: lr 2e-4 is fine; safe headroom.
- S in [0.92, 0.98): mild instability.
- S < 0.92: too aggressive; 1e-4 was right.
