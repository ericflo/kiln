# pi-diff-patch-apply

Apply unified diffs, repair drift, verify tests. **Round 1 saturated
(base 0.9419, GRPO regressed)**. Round-2 reshape: multiplicative format
gate + harder data.

## Status (round 2)

**RESHAPED RUBRIC v2.** Multiplicative format gate (was additive).

| File | Status |
|------|--------|
| `capability.md` | Spec + round-2 reshape options (A/B/C) |
| `rubric.py` | v2: `composite = outcome × format × (base + process)` |
| `archive/rubric_v1_additive.py` | Round-1 version preserved |
| `calibration/` | Workdir-dependent; bypassed |
| `archive/` | FINAL_WRITEUP, drive scripts |

## Round-2 plan (capability.md Option A — recommended)

1. **Multiplicative format gate** (done).
2. **Re-target hard-eval**: multi-hunk, refactor-heavy, conflict-prone
   patches with measured base < 0.85.
3. Fall back to OPD (Option B) if base still >0.90 after reshape.

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh           # re-baseline
./run_iter.sh h1-default-recipe
```
