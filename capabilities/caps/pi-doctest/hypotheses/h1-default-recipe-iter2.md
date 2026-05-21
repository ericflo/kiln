# Hypothesis H1 — Phase 1 GRPO recipe at scale (iter 2)

**Family:** H1 (default recipe — scaled up from iter 1's 3-group smoke)
**Target sub-score:** `tool_call_efficiency`, with the real prize being `outcome`

## Claim

Training Qwen3.5-4B against the v1 multi-component reward using kiln's
Phase 1 GRPO defaults, on **20 training tasks × 4 generations** (vs
iter 1's 3 groups), will produce composite lift ≥+0.04 over the
same-pod H100 baseline.

## Mechanism

iter 1 (3-group smoke) showed the target sub-score `tool_call_efficiency`
moves in the right direction but composite stays flat because the
training distribution was too narrow (3 hard tasks). Scaling to 20
tasks should expose the adapter to a wider range of difficulties.

## Results

Run on H100 SXM-80GB, KILN_BATCHING_ENGINE=0, KILN_DISABLE_FUSED_GDN_GATES=1
workaround for the H100 GDN gates kernel issue (PR #1050 fixed the
launch-error surface but the kernel itself still fails on H100 in
the paged-decode + training reference-forward paths — see kiln-polish).

### Headline numbers

| metric | base (n=24) | iter 2b (n=24) | Δ |
|---|---|---|---|
| **composite** | 0.8052 | **0.9187** | **+0.1135 (+11.4pp)** |
| **outcome** | 0.8333 | **1.0000** | **+0.1667 (perfect)** |
| tool_call_efficiency | 0.8073 | 0.7292 | −0.078 |
| tested_before_done | 0.9792 | 1.0000 | +0.021 |
| format_compliance | 1.0000 | 1.0000 | 0 |
| n_tool_calls (mean) | 6.13 | 6.13 | 0 |
| wall_clock (mean) | 31.4 s | 24.1 s | **−7.3s (−23%)** |

### Per-task verdict

9 wins (Δ > +0.02), 4 losses (Δ < −0.02), 11 same.

**Strong wins** — recovered total failures:
- task_0002: 0.00 → 0.85 (`make_palindrome`)
- task_0004: 0.00 → 0.85 (`x_or_y`)
- task_0007: 0.00 → 0.70 (`circular_shift`)
- task_0011: 0.00 → 0.70 (`closest_integer`)

**Moderate wins**:
- task_0005: 0.74 → 0.93 (one fewer wasteful debug round)
- task_0012: 0.93 → 1.00
- task_0013: 0.93 → 1.00

**Regressions** — previously-passing tasks now use extra tool calls:
- task_0017: 1.00 → 0.70 (3 → 13 tool calls)
- task_0019: 0.93 → 0.70 (6 → 20 tool calls)
- task_0015: 0.89 → 0.74 (7 → 11 tool calls)

### Group statistics (training)

20 groups, 8 kept (within-group variance > 0.001), 12 dropped by
dynamic_sampling:
- 4 groups with strong signal (variance > 0.10): groups 1, 6, 12, 17
- 4 groups with weak signal: groups 2, 7, 10, 14, 19
- 7 groups all-1.0 (trivially solved), 3 groups all-0.0 (model fails consistently)

The dropped groups are the 4B's "always solves" tasks (rolling_max,
strlen, etc.) and "always fails" tasks (whatever it can't handle).
GRPO trains on the in-between, which is the right regime.

## Verdict

**✓ kept-ship.** Composite +0.1135 well above the +0.04 threshold.
The headline win is that **outcome went from 0.83 to 1.00** — every
single one of 24 held-out eval tasks now passes its doctests.

## Inspected rollouts (Phase 5 requirement)

### Recovery — task_0007 (`circular_shift`)
Baseline: 33 tool calls, 120s timeout, composite 0.00.
Iter 2b: 12 tool calls, 30s, composite 0.70 (passes doctests, gets
the outcome reward, slightly costly on tool_call_efficiency).

The baseline gets stuck in a debug loop where it can't figure out
how to handle the `x` rotation case. The adapter reads the doctest
example more carefully on the first read and writes a correct
implementation in one edit.

### Recovery — task_0011 (`closest_integer`)
Baseline: 19 tool calls, 120s timeout, composite 0.00.
Iter 2b: 15 tool calls, 60s, composite 0.70.

The base model thrashes on the `round half away from zero` edge
case. Adapter handles it on the second try.

### Regression — task_0017
Baseline: 3 tool calls (read → edit → DONE), composite 1.00.
Iter 2b: 13 tool calls, composite 0.70.

Adapter added an unnecessary verification cycle that timed out. This
is the over-training pattern from iter 1, partially carried over.

## Disposition

**Status: kept-ship.** This is the real uplift on a narrow agentic
use case (pi-doctest single-prompt humaneval task completion). The
adapter both (a) solves 4 tasks the base model could not solve at
all, and (b) speeds up 23% on average.

## Iter 3 plan

Address the per-task regressions:
- Stratified training: equal weight of "always pass / partial / always
  fail" tasks. Avoids over-tuning to debug-loop behavior.
- Slightly higher tool_call_efficiency weight (0.30 → 0.40) in the
  rubric to penalize the over-verification mode that iter 2b shows
  on task_0017/_0019.
- Larger eval set if available.

## Pod cost

H100 SXM-80GB at $2.99/hr × ~1h for iter 2b = ~$3.
Cumulative session H100 cost: ~$8 ($5 prior + $3 here).
