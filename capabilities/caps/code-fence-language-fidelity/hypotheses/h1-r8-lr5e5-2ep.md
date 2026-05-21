# Hypothesis: h1-r8-lr5e5-2ep

## Family
H1 — gentle dosage from the cap #5 lessons. Baseline 0.91 puts us in
the borderline OPD-vs-too-high zone. Apply cap #5's "gentler settings
prevent catastrophe even if they don't lift" as a safety floor.

## Claim
Conservative recipe (rank 8, lr 5e-5, 2 epochs, spp 1) lifts
`language_tag_correct` by ≥3pp (0.867 → ≥0.897) and composite by
≥1pp (0.907 → ≥0.917) without regressing fence_pair or no_extra_text
from saturation.

## Mechanism
4B already saturates fence_pair (1.0) and no_extra_text (1.0). It
misses on a small subset of language tags / code-parse failures (~13%).
Teacher distribution should favor correct tag tokens at the opening
fence position; reverse-KL would pull the student toward those.

## Configuration
- prompts: prompts/h1-r16-2ep.jsonl (26 prompts, 15 languages)
- rank: 8 alpha: 16 lr: 5e-5
- epochs: 2 spp: 1 max_tokens: 384 top_k: 8
- env: KILN_STREAMING_PREFILL=1
- checkpoint_interval: 25

## Falsification plan
- Composite ≥ 0.917: kept.
- Composite in [0.85, 0.91]: small regression — borderline confirmation
  of the cap #5 high-baseline pattern.
- Composite < 0.5 × baseline (< 0.45): all-zeros gate; failure_mode.md.
- Composite collapse like cap #5 iter 1: pivot to acknowledging the
  high-baseline failure mode applies here too.

## Wall-time budget
≤ 30 min (small prompt set, few epochs, low rank).

---

## Verdict
✓ Confirmed. Composite **0.9300** (+2.33pp vs baseline 0.9067).
Target `language_tag_correct` moved +3.3pp (0.867 → 0.900); `code_parses`
also moved +3.3pp. Headroom utilization: 25% (0.0233 of 0.0933).

| Adapter | composite | fence | no_extra | lang_tag | parses |
|---------|-----------|-------|----------|----------|--------|
| baseline       | 0.9067 | 1.00 | 1.00 | 0.867 | 0.867 |
| ckpt-25        | 0.9300 | 1.00 | 1.00 | 0.900 | 0.900 |
| final (step 41)| 0.9300 | 1.00 | 1.00 | 0.900 | 0.900 |

Both checkpoints scored identically — the model converged by epoch 2
mid-point. More training wouldn't help; if anything, would risk the
cap #5/#4 over-training failure modes.

**Cap #5 vs cap #6 — the high-baseline distinction sharpened:**

Cap #5 (baseline 0.85, diff/patch): OPD regressed at every setting.
Cap #6 (baseline 0.91, code fence): OPD lifted +2.33pp.

The condition that matters isn't baseline composite alone — it's
**student rollout consistency**. Cap #5's rollouts were variable:
clean fenced diffs sometimes, malformed prose sometimes. OPD locked
in the malformed ones. Cap #6's rollouts are consistent: the 4B
always emits a fence + tag, just sometimes with the wrong tag.
OPD's gradient on a consistent rollout shape has stable signal and
lifts cleanly.

Skill update: §0 "Where OPD shines" already mentions consistent
rollouts as a precondition. Cap #6 is the positive example that
validates the framing — cap #5 was the negative example. Both now
in the skill's evidence base.

Sub-score regressions vs baseline: none (fence_pair and no_extra_text
both held at 1.0, target sub-scores moved up).

Best 'adapter' for cap #6: **`fence-h1-r8-lr5e5-2ep`** (ckpt-25
identical, ship either). Composite 0.93, real +2.33pp over baseline.

