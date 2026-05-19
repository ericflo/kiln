# H1 — default GRPO recipe baseline

**Iter:** 1

## Hypothesis

The default Phase-1 GRPO recipe (rank-16 LoRA, lr 1e-5, kl 0.1, clip
0.20, ECHO λ 0.05) on 8 train tasks × 4 generations with a
strong-signal variance filter (var > 0.02) should move the composite
above baseline by at least one group-variance stdev. This is the
"does the loop work" iter — small data, default hyperparams, sanity
check that GRPO + ECHO + our rubric produces a signal at all.

## Recipe

- 8 train tasks
- 4 generations per task
- Filter var > 0.02 (strong-signal groups only)
- lr=1e-5, rank=16, alpha=32
- KL coeff 0.1, clip 0.20
- ECHO λ 0.05 (default)
- DrGRPO advantage mode, token-level loss aggregation
- 1 epoch

## Predictions

- composite Δ vs base: +0.03 to +0.08 (small but signal)
- target sub-score: `held_out_passes` — if symptom fixes drop and
  root-cause fixes rise, that's the cap working
- mean_wall_clock_s: should be flat or slightly lower (model becomes
  more efficient as it learns the format)

## Verdict (filled after iter)

TBD
