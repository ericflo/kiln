# pi-failure-triage

Root-cause-fix discipline. **Round 1: saturated (+0.006), format moved
+12.5pp but additive composite trapped it.** Round-2 reshape:
multiplicative format gate (rubric.py changed to outcome × format × process).

## Status (round 2)

**RESHAPED RUBRIC.** v2 multiplicative format gate.

| File | Status |
|------|--------|
| `capability.md` | Spec + round-2 reshape plan |
| `rubric.py` | v2: `composite = outcome × format × process` |
| `archive/rubric_v1_additive.py` | Round-1 additive version preserved |
| `calibration/` | Workdir-dependent; bypassed |
| `archive/` | Round-1 FINAL_WRITEUP, drive scripts |

## Round-2 plan

1. **Multiplicative format gate** (done in rubric.py).
2. **Hard-eval pool of multi-cause failures** — round 1 was single-root.
3. **Cross-domain transfer** — train Python, eval shell/Rust.

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh           # re-baseline under v2 rubric
./run_iter.sh h1-default-recipe
```
