# Hypothesis: prose-mistake-named

(N-family per §4: "For every positive example, include one near-miss
that fails. Contrast tightens the boundary." T-family variant: prose,
but contrastive instead of constructive.)

## Claim

32 examples in which the assistant *names the typical novice mistake*
on a math-shaped situation, then briefly indicates the right approach,
will lift the math score above iter 1 (0.903). The hypothesis is that
mistake-naming carries strictly more information than describing only
the right approach: it teaches the model what NOT to do, narrowing
the basin of bad routings.

Specifically expects S in [0.91, 0.95]. If higher, mistake-naming is
a stronger T-form than iter-1's constructive prose. If lower, the
contrast lost the simple-frame signal.

## Mechanism

Iter 1's prose described the right approach. This dataset describes
the wrong approach explicitly, with a brief mention of the right one.
Mistake-naming serves two latent functions:
1. **Frame disambiguation**: by stating what looks correct but isn't,
   it sharpens the line between "near-miss frame" and "right frame".
2. **Negative supervision**: the model gets explicit signal about
   error patterns it should not produce.

If iter 1's lift came from concept-routing, mistake-naming should
either match it (same routing taught from the other direction) or
exceed it (additional contrastive information).

## Dataset shape

- Size: 32 examples (half iter 1's; tighter test)
- Modality: pure prose; no numbers; no equations. Each assistant turn
  contains 3-5 sentences naming the mistake + brief right-approach.
- Distribution by domain: same 7 domains, ~4-5 per domain
- Surface form held OUT: same as iter 1 — no numerical answers
- System prompt: none

## Construction recipe

Hand-drafted. Each example reuses an iter-1 situation but inverts the
assistant turn:
  USER: "[situation framing]"
  ASSISTANT: "The novice typically [mistake]. The right approach is
              [brief correct framing]. [Why the mistake fails / why
              the right approach works.]"

## Risk

The main risk is that mistake-naming biases the model toward producing
*explanations of mistakes* rather than producing answers. At eval
time, it might say "the mistake here is..." instead of giving a number.
Manifests as catastrophic drop relative to iter 1.

A secondary risk is that 32 is too small; signal is too noisy. If
results are ambiguous, scale to 64 in iter 11.

## Falsification plan (committed BEFORE seeing the score)

- S ≥ 0.95: mistake-naming exceeds iter 1. Strong N-family result.
  Next iter combines: `mixed-mistake-numeric` (32 mistake + 32 numeric)
  in the iter-3 spirit, testing if anchor + N-route stacks.
- S in [0.91, 0.95): on par with iter 1; mistake-naming works but
  doesn't dominate. Next iter is `mixed-mistake-numeric` anyway, to
  test additivity with the proven anchor route.
- S in [0.85, 0.91): partial lift but less than constructive prose.
  Retire N-route as primary; next iter pivots to `meta-what-kind-of-
  problem`.
- S < 0.85: severe regression. Mistake-naming hurts. Retire entirely
  and pivot to `meta-what-kind-of-problem`.
