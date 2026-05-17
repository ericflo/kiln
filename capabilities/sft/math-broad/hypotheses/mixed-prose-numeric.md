# Hypothesis: mixed-prose-numeric

(Additivity test triggered by the iter-2 falsification plan: prose has
a small edge over numeric; do the two routes stack?)

## Claim

Concatenating iter 1's 64 prose word-situations with iter 2's 32
numerical worked examples (96 total) and training a fresh adapter on
the combined set will yield a math score above iter 1's 0.903.
Specifically, if the two routes are additive, S3 ≥ 0.93. If they
interfere or one route dominates, S3 ≤ 0.91.

## Mechanism

Two distinct learning mechanisms are at play.

The prose route (iter 1) teaches *frame recognition* — naming which
math concept applies to a given situation — via verbal supervision
that never exercises the surface form. The numeric route (iter 2)
teaches *answer-form discipline* and refreshes symbolic primitives
that the model already has but routes into unreliably.

If these mechanisms are orthogonal, then combining the datasets should
preserve both lifts; the total lift should be approximately additive
above what each route delivers alone. If the mechanisms overlap, the
combination lifts less than additive (some of the lift was the same
mechanism counted twice). If they actively interfere (e.g., the
numeric form attracts the model's output style and breaks the prose
gain in iter 1), the combination could lift less than iter 1 alone.

This iter also tests whether iter 1's anchor regression (-0.065)
shrinks when the dataset includes more familiar-form examples. The
prose-only dataset may have steered general style; mixing in numeric
should pull the style back toward the base model's defaults.

## Dataset shape

- Size: 96 examples (64 prose from iter 1 + 32 numeric from iter 2)
- Modality: hybrid. ~67% prose-only assistant turns; ~33% numeric
  assistant turns. No interleaving — prose section first, numeric
  section second (deterministic, reproducible).
- Distribution: inherits both datasets' domain coverage; the prose
  half covers 7 domains × ~9 examples each, the numeric half covers
  the same 7 domains × ~4-5 examples each.
- Surface form held OUT: nothing new. The numeric portion intentionally
  exercises the surface form to test additivity.
- System prompt: none.

## Construction recipe

Concatenate `datasets/prose-approach-broad.jsonl` and
`datasets/numeric-drill-control.jsonl` with `cat`. Result is
`datasets/mixed-prose-numeric.jsonl`, 96 lines, immutable per §3.

## Risk

The biggest risk is **route interference**: training on both prose
(no numbers in assistant) and numeric (numbers in assistant) might
create a confused output policy where the model can't decide which
form to use, hurting both. The score would then come in *below* either
route alone, which would be very informative.

A smaller risk is **stylistic blending**: the model produces hybrid
answers that are partly prose and partly numeric in a way the eval's
exact-match scoring penalizes. This would manifest as S3 ≈ S2 (numeric
form preserved, prose advantage lost).

## Falsification plan (committed BEFORE seeing the score)

Let S3 be iter 3's primary score, ANCHOR3 be the anchor score.

- S3 ≥ 0.93: routes are additive. Big win. Next iter is
  `mixed-prose-numeric-bigger` (128+ examples, same recipe) to test
  if more scale gives more lift.
- S3 in [0.91, 0.93): marginal additivity. Both routes contribute but
  the combination saturates near the iter-1 ceiling. Next iter pivots
  to `prose-approach-broad-paraphrased` (F-family) instead — increase
  prose diversity rather than add numeric.
- S3 in [0.86, 0.91): no additivity. Prose alone (iter 1) is essentially
  the ceiling. Next iter is `prose-approach-broad-rank2` to chase the
  anchor regression fix while preserving the math gain.
- S3 in [0.81, 0.86): mild interference; the numeric portion is
  pulling the prose effect down. Next iter is `prose-approach-broad-
  rank2` and the mixed route is retired to dead ends.
- S3 < 0.81: severe interference. The two routes actively fight. Both
  routes have to be reconsidered. Next iter pivots to a different
  T-family variant: `prose-mistake-named` (assistant turns name the
  mistake a novice would make).
- Anchor delta: if ANCHOR3 < 0.95 (worse than iter 1's 0.935),
  combination MAKES the regression worse. Note this in the entry and
  prioritize rank-2 next regardless of primary score.
- Anchor delta: if ANCHOR3 ≥ 0.98 (better than iter 1), the numeric
  portion did rein in stylistic drift. The mixed route is preferred
  over prose-only for downstream refinements.
