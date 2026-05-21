# pi-shell-hygiene

Use the right shell patterns for long-running processes: nohup + wait-file
+ proper trap, not polling loops. Per the clouderic kiln-skill
anti-pattern doc.

## Read first

1. [`capability.md`](capability.md) — contract, hypotheses.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.
3. [`../README.md`](../README.md) — ECHO defaults.
4. clouderic kiln-skill body for the canonical anti-pattern list.

## Status

**Implementation complete. Calibration passes (separation +0.20).**

| File | Status |
|------|--------|
| `capability.md` | Full spec |
| `capability.config.json` | Standard agentic defaults |
| `build_corpus.py` | 24 train + 12 eval across 4 scenarios |
| `rubric.py` | Multiplicative-gate; pattern matching against good/bad lists |
| `rubric_sanity.py` | Mandatory gate (separation 0.20) |
| `rollout.py` | Pi driver |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline |
| `calibration/good.jsonl` | 5 hand-written good rollouts |
| `calibration/bad.jsonl` | 5 §0 cheats from kiln-skill anti-patterns |

## Quickstart

```bash
python3 build_corpus.py
python3 rubric_sanity.py     # PASS — separation 0.20
./capability.oracle.sh
./run_iter.sh h1-default-recipe
```

## Headroom estimate

- **Baseline:** ~0.45 (4B knows about nohup but defaults to polling loops).
- **Headroom:** ~0.55.
- **Target sub-score:** `uses_good_pattern` (largest movable mass).

## Hypotheses

| Slug | Knob | Hypothesis |
|------|------|------------|
| h1-default-recipe | defaults | +0.15 composite |
| h2-cross-platform | corpus including macOS variants | Generalization |
| h3-chain-error-recovery | from pi-error-recovery best | Background process recovery composes |

## Composition

- **Upstream:** none.
- **Downstream:** `pi-source-mod-workflow` (long-running git operations
  benefit from shell hygiene).
- **Integration:** member of `integration/cross-cap-coherence/`.

## History

Round-1 scaffold; no archive content.
