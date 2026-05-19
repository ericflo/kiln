# pi-code-comprehension — in-progress experiment log

**Status:** Iter 2 mid-training (2026-05-19). Working log; updated as iters land.

## Capability

Agent reads a target Python symbol in a small code snapshot, optionally greps
for callers across files, then emits a structured JSON summary with the
fields: `inputs`, `returns`, `mutates`, `calls`, `called_by`, `invariants`,
`side_effects` — each grounded by a `source_line` cite where applicable.

Composite (the GRPO reward) = `outcome × (0.20·grounding +
0.15·cross_file_caller_recall + 0.10·invariant_coverage +
0.05·format_compliance + 0.50)` where `outcome` is a weighted mean F1
across the 7 structured fields.

## Headline results so far

| Iter | Slug | Composite | Δ vs base | Outcome | Grounding | Cross-file | Inv-cov | Format | Notes |
|------|------|-----------|-----------|---------|-----------|-----------|---------|--------|-------|
| 0 | baseline-base-model | 0.611 | — | 0.678 | 0.750 | 0.833 | 0.236 | 0.833 | Base Qwen3.5-4B, 12-task eval |
| 1 | h1-default-recipe | 0.707 | **+0.096** | 0.788 | 0.785 | 1.000 | 0.375 | 1.000 | Default Phase 1 GRPO/ECHO recipe, 16 train × 4 gen, lr=1e-5, rank 16/32 |
| 2 | h1-more-tasks-24 | (training) | — | — | — | — | — | — | 24 train × 4 gen, same recipe — does more data help? |

Iter 1 was a substantial win. Outcome F1 jumped +0.110 because the trained
adapter learned to:
- emit valid JSON every time (`format_compliance` 0.833→1.000)
- always grep for cross-file callers (`cross_file_caller_recall`
  0.833→1.000, saturated on the corpus)
- cite more lines (`grounding` +0.035)
- recover more invariants (`invariant_coverage` +0.139)

Wall-clock per rollout also dropped 4× (74s → 18s) — adapter became
decisive about emitting the answer fast.

## Sub-score headroom (post-iter-1)

| sub-score | iter-1 value | remaining headroom |
|-----------|--------------|--------------------|
| outcome | 0.788 | 0.212 |
| grounding | 0.785 | 0.215 |
| cross_file | 1.000 | 0.000 (saturated) |
| invariant_coverage | 0.375 | **0.625** (biggest movable mass) |
| format_compliance | 1.000 | 0.000 (saturated) |

Iter 2+ effort focuses on `invariant_coverage` — the gold invariants are
extracted heuristically from docstrings + body patterns (assert / raise),
and even a perfectly grounded summary only gets ~0.4 on this sub-score
because the seed corpus has many tasks with implicit invariants the
model has to infer from body code without explicit guidance.

## Infrastructure notes (for future runs)

- **Pod loss**: the first A6000 pod (h9snqr0ow80bev) was reaped mid-iter-2
  due to a runpod_api.py `pod info` regression (`'NoneType' object has no
  attribute 'get'` on `data.get("pod")`). Adapter was recovered from
  B2 (`b2://clouderic/kiln/pi-code-comprehension/20260519/iter-1-train/
  adapter`); cycle was: re-acquire RTX A6000 → clone repo + build kiln +
  install pi → download iter-1 adapter from B2 → resume drive.py from
  iter 2. ~30 min lost.

- **Drive resilience (in drive.py)**:
  - filter_groups now ships its filter script as base64 (avoids ssh
    escape-sequence mangling).
  - kill_kiln_serve, set_adapter, and the pod_ssh layer swallow ssh
    failures (do NOT cascade to fatal iter error).
  - Per-iter stale .done files removed BEFORE bg launch so a prior
    iter's residual sentinel can't fool pod_wait.
  - Errors go to failures.jsonl, not capability.jsonl — so the iter log
    stays clean.

- **GRPO step count is data-driven**: 18 strong-signal groups (iter 2)
  → 1.1M training steps → ~50 min on A6000. Iter 1 had 11 groups →
  346K steps → ~13 min. Iter 3+ recipes use 8 train tasks (~6 groups
  → ~400K steps → ~15 min training) to keep iter wall-clock bounded.

- **Rollout concurrency=2** (in drive.py) — pi sessions are mostly
  I/O bound vs kiln (which batches HTTP requests), so 2× concurrent
  rollouts halve wall-clock with no GPU contention.

## Open hypotheses worth running

Per recipes.json, the next iters explore:
- More train data (24 tasks) → already iter 2
- Warm-start from best-so-far adapter (iter 3, 5, 9, 15, 17, …)
- ECHO λ sweep: 0.025, 0.035, 0.075, 0.10, 0.15 (paper §3.3 productive
  range is 0.01–0.05 — high values may collapse, low may not help)
- LR sweep: 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5
- Rank sweep: 4, 8, 16, 32, 64, 128
- num_gens 2 vs 4 vs 8 vs 16 (advantage variance vs sample count)
- ECHO off (paper says ECHO ≈ 2× improvement on TerminalBench-2.0)
- no_policy_loss (verifier-free mode, §5.5)

## Closeout plan

Once iters 50 lands, the best-eval composite adapter is the kept one.
Re-eval at a 2nd seed to confirm reproducibility. Write final
WRITEUP.md citing the best adapter's recipe, push, b2-backup the
adapter to a stable location.

## Audit log

- 2026-05-19 09:35Z — pod h9snqr0ow80bev acquired (RTX 6000 Ada)
- 2026-05-19 09:42Z — kiln built, pi 0.75.3 installed
- 2026-05-19 10:09Z — iter 0 baseline = 0.611 logged
- 2026-05-19 11:01Z — iter 1 adapter trained (1146K steps, 13 min)
- 2026-05-19 11:11Z — iter 1 eval = 0.707 logged, b2-backed-up
- 2026-05-19 12:25Z — pod h9snqr0ow80bev reaped during iter 2 setup
- 2026-05-19 12:38Z — pod 5hwknb14vutm5w acquired (RTX A6000)
- 2026-05-19 12:51Z — iter 1 adapter restored from B2
- 2026-05-19 13:41Z — iter 2 GRPO training started (24 train × 4 gen,
  18 strong-signal groups, 1.1M steps)
- 2026-05-19 14:xx Z — *expected* iter 2 eval
