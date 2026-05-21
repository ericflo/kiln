# calibration/ — cross-cap-coherence

This cap is **eval-only** and **delegates rubric.score_one to each
member cap's rubric**. Calibration here is trivial since the per-cap
calibrations cover the substance.

The cross-cap coherence rubric has no standalone sub-scores; the
"coherence" is computed at the oracle level (per-cap composites
aggregated, deltas vs base reported).

`rubric_sanity.py` returns WARNING on this cap (calibration unpopulated),
which is acceptable since no training happens here. The oracle pulls
fresh member-cap evals at run time.

## How to populate

Optional: write 5 fixtures with a `_member_cap` annotation pointing to
a member's good rollout (and similarly 5 bad). This proves the
delegation works. Not required for the eval-only workflow.

## Refreshing

After changing member-cap rubrics, run:
```bash
python3 ../rubric_sanity.py
```
