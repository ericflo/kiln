# cross-cap-coherence

Eval-only integration capability (round 2). Measures whether a trained
adapter regresses any *other* capability in the suite.

## Read first

1. [`capability.md`](capability.md) — contract.
2. [`../README.md`](../README.md) — the integration track overview.
3. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.

## Quickstart

```bash
# 1. Build the integration eval set (requires all member caps' eval.tasks.jsonl
#    to exist).
python3 build_corpus.py

# 2. Eval one adapter.
./capability.oracle.sh pi-doctest-iter5

# 3. Compare multiple adapters.
./capability.oracle.sh pi-doctest-iter5 pi-faithful-completion-iter50
```

The output JSON shows per-cap composites, per-cap deltas vs base, and
flags regressions (per_cap_delta < −0.02).

## Status

Eval harness is scaffolded; the per-cap eval slices come online as each
member cap finishes its own `build_corpus.py` run. `build_corpus.py`
here skips members whose eval set doesn't exist yet.

## No training

`run_iter.sh` does NOT train. It calls `capability.oracle.sh`.
