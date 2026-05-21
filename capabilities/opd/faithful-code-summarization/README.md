# faithful-code-summarization (OPD)

Distill faithful code summaries from 27B teacher. **Round-1 sparse;
round-2 calibration added.**

## Status

| File | Status |
|------|--------|
| `capability.md` | Spec |
| `rubric.py` | parses + entity recall + precision + concise |
| `calibration/` | **5 good + 5 bad fixtures, separation +0.21 PASS** |

## Round-2 plan

1. Mature the rubric (calibration revealed weakness on empty/generic summaries).
2. Spot-check 27B teacher outputs against ground truth.
3. Compose with `pi-code-comprehension` (which produces structured summaries).
