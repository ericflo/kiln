# Hypothesis: mixed-meta-numeric-lr-low

(LR sensitivity test on saturation recipe: half the learning rate.)

## Claim
Same iter-13 recipe (64 mixed meta+numeric, rank 8, 1 epoch) at lr 5e-5
will hold within 0.03 of 1.000.

## Mechanism
Lower lr = smaller per-step updates = less adapter weight movement
per epoch. If 5e-5 underfits in 1 epoch, score drops. If 5e-5 is more
stable and 1e-4 was slightly aggressive, score might hold or improve.

## Falsification plan
- S ≥ 0.98: lr 5e-5 is fine; 1e-4 wasn't overshooting.
- S in [0.94, 0.98): mild underfit at lr 5e-5; 1e-4 is right.
- S < 0.94: significantly undertrained at lower lr.
