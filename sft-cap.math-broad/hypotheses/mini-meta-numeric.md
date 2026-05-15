# Hypothesis: mini-meta-numeric (minimum-viable size)

(Stress test on iter 13 saturation winner: can we hit 1.0 with HALF
the data?)

## Claim

16 meta + 16 numeric (32 examples total) at rank 8 will lift the math
score to at least 0.95. If it hits ≥0.97, the recipe is genuinely
data-efficient and the iter-13 result wasn't dependent on dataset size.

## Mechanism
If iter 13's lift came from teaching the *frame* (which meta-questions
do compactly), 32 examples should suffice. If it came from
pattern-density (more examples = more variety covered), halving the
data should hurt.

## Falsification plan
- S ≥ 0.97: ultra-efficient recipe; the routing frame transfers from
  very few examples.
- S in [0.95, 0.97): partial — size matters somewhat.
- S in [0.90, 0.95): size matters meaningfully; 32 is too small.
- S < 0.90: size is critical.
