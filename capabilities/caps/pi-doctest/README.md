# pi-doctest

A coding agent given a Python doctest spec must edit a stub, run the
doctest, and verify before declaring done. **Round 1 winner (+4.2pp
3-seed verified)**, recipe reproduces at 0.896 ± 0.003 across 3 seeds.

## Read first

1. [`capability.md`](capability.md) — multi-component v1 rubric with
   tool_call_efficiency as target sub-score.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.
3. [`../README.md`](../README.md) — ECHO defaults.

## Status (round 2)

| File | Status |
|------|--------|
| `capability.md` | Full spec + round-2 improvement plan (add hidden_tests) |
| `capability.config.json` | Tuned defaults |
| `build_corpus.py` | HumanEval-derived; 67 train + 24 eval |
| `rubric.py` | Multi-component (outcome × (tool_eff·0.30 + tested·0.20 + format·0.10 + 0.40)) |
| `rubric_sanity.py` | Workdir-dependent; bypassed pending real fixtures |
| `rollout.py` | Pi driver |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline |
| `calibration/` | Documented limitation (workdir-dependent rubric) |
| `archive/` | Round-1 history (kept iter 5 adapter, 3-seed verified) |

## Round-2 improvements (see capability.md)

1. **Add `hidden_tests` sub-score** — the §0 A1 cheat we deferred.
   Each task gets ≥3 visible + ≥3 hidden tests; punishes memorize-the-doctest.
2. **Expand eval pool 24 → 50** to bring composite_stdev below 0.005.
3. **Build hard-eval pool** from round-1 failed task IDs.

## Quickstart

```bash
python3 build_corpus.py
./capability.oracle.sh           # baseline; round-1 mean ~0.85-0.90
./run_iter.sh h1-default-recipe
```

## Headroom

- **Baseline**: 0.885 composite (24-task eval).
- **Round-1 best**: 0.896 ± 0.003 (3-seed, iter 5).
- **Round-2 target**: with hidden_tests + harder eval pool, expect baseline to drop to ~0.75 (re-opens headroom).
