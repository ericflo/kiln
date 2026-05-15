# Hypothesis: mixed-meta-numeric

(Additivity test for meta-route per iter-11 falsification plan: pair
the proven anchor recipe with meta-question prose.)

## Claim

Concatenating iter 11's 32 meta-question examples with iter 2's 32
numerical worked examples (64 total) and training a fresh adapter at
rank 4, 1 epoch, lr 1e-4 will lift the math score above iter 11's
0.919 — testing whether the form-anchor effect we saw in iter 3 also
boosts the meta-route. If S ≥ 0.94, meta + anchor matches iter 3.
If S ≥ 0.95, meta + anchor reaches or exceeds iter 7's rank-8 best
(using only rank 4).

## Mechanism

Iter 11 (meta alone) gave +0.323 vs baseline with no numeric anchors.
Iter 3 (prose + numeric) gave +0.339. The extra +0.016 from prose+anchor
over prose-alone (iter 1 → iter 3) was attributed to the numeric portion
acting as form anchors. If the same anchor effect applies, iter 12
should land at ~0.94 (0.919 + 0.020 expected anchor lift).

If the anchor effect is route-agnostic, this confirms a general
principle: any T-route (prose, meta, mixed) benefits from numeric
form-anchor supplementation.

## Dataset shape

- Size: 64 (32 meta + 32 numeric, concatenated meta-first)
- Path: `datasets/mixed-meta-numeric.jsonl` (already built via cat)
- Modality: mixed — meta-classification prose + numerical worked answers

## Risk

The meta-route may interfere with the numeric form-anchor effect
differently from prose. If meta examples teach "name the problem
type" and numeric examples teach "answer numerically", the model's
output policy might land somewhere unexpected.

## Falsification plan

- S ≥ 0.95: anchor effect carries to meta-route at rank 4. Tied with
  or beats rank-8 best. Significant finding.
- S in [0.93, 0.95): matches iter-3 territory. Anchor effect is real,
  meta-route is competitive.
- S in [0.90, 0.93): minor lift from anchors. Meta + anchor not as
  good as prose + anchor.
- S in [0.85, 0.90): anchor didn't help meta as much; suggests anchor
  effect is route-specific.
- S < 0.85: interference; meta + anchor backfired.
