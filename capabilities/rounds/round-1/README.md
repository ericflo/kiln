# Round 1 — initial capability experiments

**Base model:** `Qwen3.5-4B` (vanilla, no distillation has occurred).
**Status:** snapshot; superseded by round 2 hardening and round 3 unification.
**Distillation:** none. Round 1 produced too few caps + too much infrastructure
churn to attempt a distillation cluster.

The authoritative narrative for round 1 lives in
[`../../CONSOLIDATED_REPORT.md`](../../CONSOLIDATED_REPORT.md). This README is a
thin pointer.

## What shipped (kept adapters)

From `CONSOLIDATED_REPORT.md::Capability Scoreboard`:

| Cap | Status | Best measured result |
| --- | --- | --- |
| `pi-faithful-completion` | kept iter 50 | 0.8065 vs 0.7237 baseline; **+0.0828** |
| `pi-code-comprehension` | kept iter 4 | 0.7405 vs 0.6112 baseline; **+0.1293** |
| `pi-doctest` | kept `pi-doctest-iter5` | 3-seed +4.2 pp; 15× tighter eval variance |
| `pi-code-search` | kept `pi-code-search-iter5-h5-replay-iter1` | peak 0.6004 vs 0.5432; **+0.024** mean |
| `pi-failure-triage` | kept iter 2 | 0.9720 vs 0.9656 baseline; +0.0064 |
| `pi-diff-patch-apply` | no kept adapter | base 0.9419 strongest; all trained iters worse |
| `pi-compaction` | full-length pipeline built | base 0.314; training was a no-op |
| `pi-terminal-bench-lite` | infrastructure validation complete | ECHO firing end-to-end |
| Other agentic caps | scaffolds only | no iter rows |

## What was learned

The big lessons (from `CONSOLIDATED_REPORT.md::Recommendations`):

1. GRPO needs non-saturated reward signal — saturated baselines hurt.
2. The harness matters as much as the loss — see the 40 kiln issues.
3. ECHO is the default for agentic; defaults are λ=0.05 and `env_only`.
4. Single-seed wins don't reproduce — 3-seed required (pi-doctest +11pp → +4.2pp).
5. Strong-signal filtering (`--filter-var-min 0.05`) is the most useful knob.

## Infrastructure backlog from round 1

40 kiln issues filed in
[`../../KILN_IMPROVEMENT_ISSUES.md`](../../KILN_IMPROVEMENT_ISSUES.md).
**All 40 are now complete** (as of 2026-05-21 actual-model validation
checkpoint). Round-2 hardening + round-3 unification both depend on them.

## Cluster manifest

`null` — no distillation occurred.

## Sibling matrix

The round-1 sibling matrix was never fully computed. The
`integration/cross-cap-coherence/` track that would have produced it is a
round-2 addition.

## Where to find round-1 artifacts

Per-cap archives:

- `capabilities/caps/<cap>/archive/` — round-1 WRITEUP.md, closeout.md,
  capability.jsonl, scripts, datasets, etc.
- `capabilities/CONSOLIDATED_REPORT.md` — cross-cap synthesis.
