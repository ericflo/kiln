# Hypothesis: mixed-prose-numeric-rank16

(H-family per §4. Extends rank scaling: 4→8 gave +0.017; test rank 16.)

## Claim

Doubling LoRA rank from 8 to 16 on iter 3's dataset will lift the
math score by ≤ +0.01 — diminishing returns are expected per the
observed rank scaling so far (2→4 gained 0.048; 4→8 gained 0.017).
This iter probes whether rank scaling has saturated or has further
gains to give.

## Mechanism

Each rank doubling adds more LoRA parameters; if those parameters can
fit additional useful patterns, score lifts. The observed half-life
of marginal gain (per doubling) suggests saturation around rank 16-32.
This test confirms or denies the trend.

## Dataset shape

Same as iter 3. 96 examples.

## Risk

Rank 16 may overfit 96 examples. Anchor regression would flag this.

## Falsification plan

- S ≥ 0.97 AND anchor ≥ 0.97: real lift beyond iter 7. Push to rank 32.
- S in [0.95, 0.97): rank 8 was the practical ceiling.
- S in [0.92, 0.95): mild overfit; rank 8 was right.
- S < 0.92: rank 16 overfits on 96 examples; cap rank at 8.
