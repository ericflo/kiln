# pi-context-aware-edits

Read imports / neighbors / conventions BEFORE editing. Round-2 new cap
(Tier 2). Distinct from `pi-precondition-check` (which is about
staleness); this cap is about *style and idiom consistency*.

## Read first

1. [`capability.md`](capability.md) — contract: 6 convention categories,
   multiplicative-gate rubric, §0.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.
3. [`../README.md`](../README.md) — ECHO defaults.

## Status

**Implementation complete. Calibration passes (separation +0.58).**

| File | Status |
|------|--------|
| `capability.md` | Full spec — 6 convention categories |
| `capability.config.json` | Tuned for max_turns=6 (read + edit + verify cycle) |
| `build_corpus.py` | 32 train + 16 eval tasks across 4 style profiles (Py strict, Py loose, Rust, Go) |
| `rubric.py` | Multiplicative gate + per-convention checkers (naming case, types, logging, error handling, comments, imports) |
| `rubric_sanity.py` | Mandatory gate |
| `rollout.py` | Pi driver (shared shape) |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline |
| `calibration/good.jsonl` | 5 hand-written good rollouts (Py strict, Py loose, Rust, simple camel, docstring) |
| `calibration/bad.jsonl` | 5 §0 cheats including outcome-passes-but-process-violates |

## Quickstart

```bash
python3 build_corpus.py
python3 rubric_sanity.py     # PASS — separation 0.58
./capability.oracle.sh
./run_iter.sh h1-default-recipe
```

## Headroom estimate

- **Baseline:** ~0.45 (the 4B reads sometimes but rarely preserves all
  conventions).
- **Headroom:** ~0.55.
- **Target sub-score:** `convention_consistency` (biggest movable mass).

## Hypotheses

| Slug | Knob | Hypothesis |
|------|------|------------|
| h1-default-recipe | defaults | +0.10 composite |
| h2-mixed-language | corpus weighted across Py/Rust/Go | Generalization test |
| h3-opd-chain | OPD from 27B on H1 best | Format polish |
| h4-stratified | balanced per-category sampling | Avoid overfitting |

## Composition

- **Upstream:** `pi-precondition-check` (read-before-edit shared discipline).
- **Downstream:** `pi-source-mod-workflow` (PR-quality edits need conventions).
- **Integration:** member of `integration/cross-cap-coherence/`.

## History

Brand-new in round 2.
