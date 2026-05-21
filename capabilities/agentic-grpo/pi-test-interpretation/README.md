# pi-test-interpretation

Read bench/test output correctly: median of >=3 runs, not first or mean;
classify flakes vs real failures. Rooted in two kiln PR incidents
(PR #150 warmup false-positive, PR #176 single-run mean miss).

## Read first

1. [`capability.md`](capability.md) — contract.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.
3. [`../README.md`](../README.md) — ECHO defaults.

## Status

**Implementation complete. Calibration passes (separation +0.85).**

| File | Status |
|------|--------|
| `capability.md` | Full spec |
| `capability.config.json` | Standard agentic |
| `build_corpus.py` | 24 train + 12 eval across 4 scenarios |
| `rubric.py` | Multiplicative gate; counts iterations (raw + for-loop + final-text "Run N" mentions) |
| `rubric_sanity.py` | Mandatory gate (0.85 separation) |
| `rollout.py` | Pi driver |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline |
| `calibration/good.jsonl` | 5 hand-written good rollouts |
| `calibration/bad.jsonl` | 5 §0 cheats including mean-instead-of-median |

## Quickstart

```bash
python3 build_corpus.py
python3 rubric_sanity.py     # PASS — separation 0.85
./capability.oracle.sh
./run_iter.sh h1-default-recipe
```

## Headroom estimate

- **Baseline:** ~0.40 (4B reports mean / first-run by default).
- **Headroom:** ~0.60.
- **Target sub-score:** `reported_median_not_mean`.

## Hypotheses

| Slug | Knob | Hypothesis |
|------|------|------------|
| h1-default-recipe | defaults | +0.20 composite |
| h2-direct-http | use `kiln rollout` not pi (this is single-turn classification) | Faster iters |
| h3-chain-failure-triage | chain after pi-failure-triage | Test interpretation feeds bug triage |

## Composition

- **Upstream:** none.
- **Downstream:** `pi-failure-triage` (interpret tests, then fix bugs).
- **Integration:** member of `integration/cross-cap-coherence/`.

## History

Round-1 scaffold; no archive content.
