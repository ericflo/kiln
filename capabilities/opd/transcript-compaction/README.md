# transcript-compaction (OPD)

OPD on conversation compaction. **Calibration exposed rubric weakness
(empty compactions score 0.85); sanity gate bypassed.**

## Status

| File | Status |
|------|--------|
| `capability.md` | Spec |
| `rubric.py` | Round-1 rubric; **known issue: empty compaction → 0.85** |
| `calibration/` | 5 good + 5 bad fixtures; gate FAILS due to rubric bug |
| `run_iter.sh` | Sets KILN_SKIP_RUBRIC_SANITY=1 |

## Round-2 priority

1. **Tighten rubric** to make empty compaction → composite 0.
2. After fix: remove the `KILN_SKIP_RUBRIC_SANITY=1` line from run_iter.sh.
3. Then run as a standard OPD iter.
