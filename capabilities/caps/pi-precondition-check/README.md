# pi-precondition-check

Verify the claim before mutating. Round-1 rank-1 cap from the clouderic
failure decomposition (~7,800 tasks analyzed). Round-2 high-priority
(Tier 3, but #1 within Tier 3 for impact).

## Read first

1. [`capability.md`](capability.md) — contract, §0 cheats, hypotheses.
2. [`../../LAYOUT.md`](../../LAYOUT.md) — uniform layout.
3. [`../README.md`](../README.md) — ECHO defaults.

## Status

**Implementation complete. Calibration passes (separation +1.00).**

| File | Status |
|------|--------|
| `capability.md` | Full spec (round 1) + round-2 improvement plan |
| `capability.config.json` | 8-turn budget; ECHO 0.05 default |
| `build_corpus.py` | 32 train + 16 eval, balanced 50/50 holds_true/stale across 4 claim templates |
| `rubric.py` | **Triple-multiplicative-gate**: outcome × format × verified × (process + base) |
| `rubric_sanity.py` | Mandatory gate (passes with margin 1.00) |
| `rollout.py` | Pi driver |
| `capability.oracle.sh` | `kiln eval-adapter --seeds 3` |
| `run_iter.sh` | Full pipeline |
| `calibration/good.jsonl` | 5 rollouts (3 holds_true + 2 stale across templates) |
| `calibration/bad.jsonl` | 5 §0 cheats including mutate-without-read |

## Why the triple gate

This cap is uniquely about "verify before you mutate." A composite
that gives partial credit when verified=0 teaches the wrong thing.
The triple gate (outcome × format × verified) makes the
mutate-without-read cheat score exactly 0, which is the only
honest signal for this capability.

For stale tasks (no mutation), verified=1.0 by construction so the
gate fires only on holds_true tasks where the model skipped the read.

## Quickstart

```bash
python3 build_corpus.py
python3 rubric_sanity.py     # PASS — separation 1.00
./capability.oracle.sh
./run_iter.sh h1-default-recipe
```

## Headroom estimate

- **Baseline:** ~0.35 (4B mutates without verifying ~60% of the time on
  similar clouderic tasks).
- **Headroom:** ~0.65.
- **Target sub-score:** `verified_before_mutation` (the multiplicative gate).

## Hypotheses

| Slug | Knob | Hypothesis |
|------|------|------------|
| h1-default-recipe | defaults | +0.20 composite over base |
| h2-echo-heavy | ECHO λ=0.10 first 50 steps then anneal to 0.05 | Stronger env-attention on file contents |
| h3-balanced | exact 50/50 stale split | Avoid bias toward holds_true |
| h4-chain-faithful | base from pi-faithful-completion | Combine terminal-state honesty with precondition discipline |

## Composition

- **Upstream:** none.
- **Downstream:** `pi-context-aware-edits` (read-before-edit is a shared
  discipline). `pi-failure-triage` (verify the bug exists before fixing).
- **Integration:** central member of `integration/cross-cap-coherence/`.

## History

Round-1 scaffold preserved at minimal extent (no archive — was empty in
round 1). Round 2 is the first iteration with concrete implementation.
