# pi-search-then-read

grep/find before reading whole large files. Round-2 new cap (Tier 2).
Composes naturally with `pi-code-search` (which is about *which file*);
this cap is about *which window of that file*.

## Read first

1. [`capability.md`](capability.md) — contract.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.
3. [`../README.md`](../README.md) — ECHO defaults.

## Status

**Implementation complete. Calibration passes (separation +0.49).**

| File | Status |
|------|--------|
| `capability.md` | Full spec |
| `capability.config.json` | Tuned (max_turns=6) |
| `build_corpus.py` | 24 train + 12 eval tasks across 3 size tiers (200/800/2000 lines) |
| `rubric.py` | Multiplicative gate; small-file exemption at ≤ 250 lines |
| `rubric_sanity.py` | Mandatory gate |
| `rollout.py` | Pi driver (shared shape) |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline |
| `calibration/good.jsonl` | 5 hand-written good rollouts |
| `calibration/bad.jsonl` | 5 §0 cheats |

## Quickstart

```bash
python3 build_corpus.py
python3 rubric_sanity.py     # PASS — separation 0.49
./capability.oracle.sh
./run_iter.sh h1-default-recipe
```

## Headroom estimate

- **Baseline:** ~0.40 (the 4B reads whole files routinely on large ones).
- **Headroom:** ~0.60.
- **Target sub-score:** `search_efficiency` (biggest movable mass on
  large-file tasks).

## Hypotheses

| Slug | Knob | Hypothesis |
|------|------|------------|
| h1-default-recipe | defaults | +0.15 composite |
| h2-echo-heavier | ECHO λ=0.075 | Search results are env tokens |
| h3-scaled-tasks | balanced across file sizes | Avoid overfit to one regime |
| h4-chain-code-search | chain from pi-code-search best | "Which file" → "which window" |

## Composition

- **Upstream:** `pi-code-search`.
- **Downstream:** all caps benefit from search-first habit (reduces context burn).
- **Integration:** member of `integration/cross-cap-coherence/`.

## History

Brand-new in round 2.
