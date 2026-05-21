# Hypothesis: mixed-prose-numeric-bigger

(B-family scale-up per skill §4 and iter-3 falsification plan: routes
are additive, test whether more scale gives more lift.)

## Claim

Adding 32 harder numeric examples to the iter-3 mixed dataset (64 prose
+ 64 numeric = 128 total) at the same rank-4 / 1-epoch recipe will
lift the math score above iter-3's 0.935 by at least +0.01. If the
gain is smaller than that, §15's bigger-data caveat applies — the
rank-4 capacity bandwidth is saturated and more data adds little. If
the gain is larger than +0.03, scaling is more productive than
expected and another doubling is worth trying.

## Mechanism

The new 32 numeric examples target **harder** problems within the same
7 domains: quadratics, rational equations, integration by parts,
multi-variable systems, ODE with initial conditions, etc. The
hypothesis is that the iter-3 mix taught the model the *broad* frame
plus *easy-form* answer discipline, but harder problems require
exposure to slightly more complex symbolic patterns. Adding harder
numeric examples gives the model targeted practice on the kinds of
manipulations that the easier numeric drill skipped.

This is *not* a pure scale test (more of the same data) — it's a
scale-with-difficulty-broadening test. If the broadening was important,
the harder numeric examples should specifically help on items the
iter-3 adapter is currently missing (we don't know which items those
are, but we expect roughly 4 wrong out of 62).

## Dataset shape

- Size: 128 examples (64 prose + 32 easy numeric + 32 harder numeric)
- Modality: hybrid. ~50% prose-only, ~50% numeric. The numeric half
  spans easy and hard difficulty.
- Distribution by domain: same 7 domains throughout (arithmetic,
  geometry, algebra, trig, systems, calculus, ODE).
- Surface form held OUT: nothing new; the numeric portion intentionally
  matches the eval's likely surface form.
- System prompt: none.

## Construction recipe

`cat datasets/prose-approach-broad.jsonl datasets/numeric-drill-control.jsonl
datasets/numeric-drill-hard32.jsonl > datasets/mixed-prose-numeric-bigger.jsonl`

The new 32 numeric examples are in `datasets/numeric-drill-hard32.jsonl`
— hand-written, harder versions of the iter-2 problems (e.g., 2x²-7x+3
instead of 3x+7=22, integration by parts instead of just power-rule
antiderivative).

## Risk

Three risks:

1. **§15 saturation**: at rank 4 the model can't fit more patterns,
   so the new examples either don't help or actively crowd out the
   existing patterns. Manifests as S4 ≈ S3 or slight drop.

2. **Hard-numeric crowding the prose**: 32 more numeric tilts the
   ratio from 67/33 prose-heavy to 50/50. If the prose-route lift
   was sensitive to the proportion, this could lower it. Manifests
   as S4 < S3.

3. **Form-anchor saturation**: iter 3's 32 easy numeric examples
   already restored answer-form discipline. Adding 32 MORE might
   over-anchor to numeric form, possibly causing prose-style answers
   to be penalized. Manifests as the prose-route advantage shrinking.

## Falsification plan (committed BEFORE seeing the score)

Let S4 be iter 4's primary score, ANCHOR4 the anchor score.

- S4 ≥ 0.96: clear lift from scale + difficulty. Next iter pushes
  further: `mixed-prose-numeric-paraphrased` (same scale, vastly more
  diverse prose framings — F-family on top of B-family).
- S4 in [0.94, 0.96): mild lift, ~1 item improvement. Next iter tries
  a different dimension — `mixed-prose-numeric-rank8` to test if
  rank capacity is the constraint.
- S4 in [0.92, 0.94): essentially flat vs iter 3. §15 saturation.
  Next iter is `prose-paraphrased` (F-family — diversify framings
  instead of adding examples).
- S4 in [0.88, 0.92): mild regression. Hard numeric crowded out the
  prose advantage. Retire this direction; next iter is
  `prose-approach-broad-paraphrased` building on iter 1, not iter 3.
- S4 < 0.88: severe regression. Roll back; the iter-3 mix is the
  ceiling for this family. Next iter pivots to a different shape
  (`prose-mistake-named` or `meta-what-kind-of-problem` from the
  ideas backlog).
- Anchor: if ANCHOR4 < 1.0 (any regression at all), note it. If
  ANCHOR4 < 0.97, the bigger mix is broken even if math lifts.
