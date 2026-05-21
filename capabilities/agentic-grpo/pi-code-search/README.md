# pi-code-search

Find the right file/symbol given a natural-language query. **Round 1
modest winner (+2.4pp 5-eval mean, +5.7pp peak)** — real but small.

## Read first

1. [`capability.md`](capability.md).
2. [`../../LAYOUT.md`](../../LAYOUT.md).

## Status (round 2)

| File | Status |
|------|--------|
| `capability.md` | Spec + round-2 plan |
| `rubric.py` | Reuses pi-doctest helpers; loads from archive/rubric_v0_outcome_only.py |
| `calibration/` | 5 good + 5 bad fixtures; separation +0.60 PASS |
| `archive/` | Round-1 FINAL_RESULTS, closeout |

## Round-2 improvements

1. **`precision_of_read` sub-score** — distinguish "grep then read right
   file" from "guess then read right file".
2. **Harder corpus** — 3-5 real OSS Python repos (10-50K LoC).
3. **Multi-seed training** (`--filter-var-min`).

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh
./run_iter.sh h1-default-recipe
```

## Headroom

- **Round-1 baseline**: 0.543.
- **Round-1 5-eval mean**: 0.568.
- **Round-2 target**: 0.65+ with precision_of_read.
