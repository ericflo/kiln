# pi-error-recovery

When a tool call fails (file not found, permission denied, syntax error,
command not found, dependency missing, timeout), the agent should read
the error, diagnose the failure class, and try a *different* approach.
Round 2 — high-priority new cap (Tier 2 in NEXT_ROUND.md).

## Read first

1. [`capability.md`](capability.md) — the contract: goal, rubric (v0 with
   multiplicative format gate), §0 cheats, hypotheses.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout + kiln CLIs.
3. [`../README.md`](../README.md) — ECHO defaults and pi-rollout shape.
4. [`../../NEXT_ROUND.md`](../../NEXT_ROUND.md) — operating manual + diagnostic
   ladder.

## Status

**Implementation complete. Calibration passes (separation +0.30).**

| File | Status |
|------|--------|
| `capability.md` | Full spec — rubric, §0, hypotheses |
| `capability.config.json` | Tuned for this cap (8-turn budget, ECHO 0.05) |
| `build_corpus.py` | Generates 30 train + 18 eval tasks across 6 error classes |
| `rubric.py` | Multiplicative-gate composite implemented |
| `rubric_sanity.py` | Mandatory gate (kiln-skill best practice) |
| `rollout.py` | Pi driver — materializes sandbox, runs pi, scores via rubric |
| `capability.oracle.sh` | Wraps `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline (calibration → rollouts → dry-run → train → verify → eval) |
| `calibration/good.jsonl` | 5 hand-written good rollouts (one per error class) |
| `calibration/bad.jsonl` | 5 hand-written bad rollouts (one per §0 cheat) |

## Quickstart

```bash
# 0. Build corpus.
python3 build_corpus.py

# 1. Verify the rubric — should print "PASS — separation 0.30".
python3 rubric_sanity.py

# 2. Baseline eval (no adapter). Expect composite around 0.40.
./capability.oracle.sh

# 3. First training iter. Default H1 recipe: lr 1e-5, rank 16, ECHO 0.05.
./run_iter.sh h1-default-recipe

# 4. Compare H2 with ECHO-heavier (paper says this cap benefits).
ECHO_LAMBDA=0.075 ./run_iter.sh h2-echo-heavy

# 5. Check integration regressions.
cd ../../integration/cross-cap-coherence/
./capability.oracle.sh pi-error-recovery-h1-default-recipe
```

## Headroom estimate

- **Baseline composite:** ~0.40 (the 4B routinely loops or gives up).
- **Headroom:** ~0.60 (genuine behavior gap).
- **Target sub-score:** `recovery_appropriate_to_error_class` (movable mass).

## Hypotheses ready to run

| Slug | Knob | Hypothesis |
|------|------|------------|
| h1-default-recipe | Defaults | +0.15 composite over base |
| h2-echo-heavy | ECHO λ=0.10 (vs 0.05) | Better `read_error_before_retry` |
| h3-strong-signal | `--filter-var-min 0.05` | Cleaner gradient (default-on round 2) |
| h4-per-class-balanced | Stratified sampling | Avoid overfitting to common class |

## Composition

- **Upstream:** None — foundational.
- **Downstream:** `pi-failure-triage` (also requires env-attention) and
  `pi-incremental-progress` (mid-flight errors are common).
- **Integration:** Member of `integration/cross-cap-coherence/`.

## History

Brand-new in round 2 — no archive.
