# pi-incremental-progress

When making a non-trivial change, work in verified sub-steps rather than
writing everything then testing once. New cap in round 2 (Tier 2).

## Read first

1. [`capability.md`](capability.md) — contract: goal, rubric (multiplicative
   gate), §0, hypotheses.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.
3. [`../README.md`](../README.md) — ECHO defaults.

## Status

**Implementation complete. Calibration passes (separation +0.33).**

| File | Status |
|------|--------|
| `capability.md` | Full spec |
| `capability.config.json` | Tuned — `max_turns=12` (this cap needs more turns) |
| `build_corpus.py` | 24 train + 12 eval tasks across 3 families |
| `rubric.py` | Multiplicative-gate composite — measures progress, alignment, failure-catching |
| `rubric_sanity.py` | Mandatory gate |
| `rollout.py` | Pi driver (shared shape with pi-error-recovery) |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline |
| `calibration/good.jsonl` | 5 hand-written good rollouts |
| `calibration/bad.jsonl` | 5 §0 cheats: big-bang, no-verify, fake-steps, spam, ignored-failure |

## Quickstart

```bash
python3 build_corpus.py
python3 rubric_sanity.py     # PASS — separation 0.33
./capability.oracle.sh
./run_iter.sh h1-default-recipe
```

## Headroom estimate

- **Baseline:** ~0.50 (4B sometimes decomposes; usually doesn't on >2-step tasks).
- **Headroom:** ~0.50.
- **Target sub-score:** `step_progress_observability`.

## Hypotheses

| Slug | Knob | Hypothesis |
|------|------|------------|
| h1-default-recipe | defaults | +0.10 composite |
| h2-extended-turns | max_turns=16 (vs 12) | More room → more decomposition |
| h3-echo-heavier | ECHO λ=0.075 | Stronger env-attention; verify results matter |
| h4-chain-faithful | Chain from pi-faithful-completion best | Terminal-state + decomposition compose |

## Composition

- **Upstream:** none.
- **Downstream:** almost all other caps benefit; this is a foundational habit.
- **Integration:** central member of `integration/cross-cap-coherence/`.

## History

Brand-new in round 2.
