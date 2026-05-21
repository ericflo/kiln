# Hypothesis: mixed-prose-numeric-3ep

(H-family per §4. Same iter-3 dataset, 3 epochs instead of 1.)

## Claim
Training iter-3's winning dataset for 3 epochs instead of 1 will
either lift the math score by +0.01 or overfit. Tests whether
1-epoch was undertraining.

## Mechanism
More epochs = the optimizer sees each example more times. If the
patterns are still learnable past epoch 1, more epochs help. If 1
epoch was already at the capacity ceiling, more epochs overfit.

## Dataset shape
Same as iter 3. 96 examples.

## Risk
Overfitting at 3 epochs can hurt both math and anchor — the model
memorises the prose phrasings and starts producing brittle answers.

## Falsification plan
- S ≥ 0.96: 1-epoch was undertraining.
- S in [0.92, 0.96): mild lift; epoch-tuning is real but small.
- S in [0.88, 0.92): 1-epoch was right; more epochs are wasted compute.
- S < 0.88: overfitting confirmed; cap at 1 epoch.
