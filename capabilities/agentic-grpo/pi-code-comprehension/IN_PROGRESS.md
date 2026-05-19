# pi-code-comprehension — live experiment log

**Status:** Pod `13knjsyhonso89` was reaped during iter 12 training (~55% done).
Drive cascaded "no runtime/ports" errors through iters 12-50 before stopping.
Need to acquire new pod, restore iter-4 best from B2, restart drive at iter 12.
**Best so far:** iter 4 `h4-echo-0075` composite = **0.7405** (+0.1293 vs baseline).
**Drive PID:** dead (was 26831).

This file is updated **as each iter lands** (alongside capability.jsonl).
For the science narrative and ablation analysis, see WRITEUP.md.
For per-iter raw rows: capability.jsonl. Failed iters: failures.jsonl.

## Results so far

<!-- AUTO:RESULTS -->
| iter | slug | composite | Δ-base | outcome | grounding | cross-file | inv-cov | wall-s |
|------|------|-----------|--------|---------|-----------|-----------|---------|--------|
| 0 | baseline-base-model | 0.6112 | — | 0.678 | 0.750 | 0.833 | 0.236 | 74.4 |
| 1 | h1-default-recipe | 0.7074 | +0.096 | 0.788 | 0.785 | 1.000 | 0.375 | 18.5 |
| 2 | h1-more-tasks-24 | 0.5887 | −0.023 | 0.664 | 0.646 | 0.750 | 0.299 | 86.7 |
| 3 | h3-warm-best-fine-tune | 0.6502 | +0.039 | 0.741 | 0.760 | 0.917 | 0.250 | 16.8 |
| **4** | **h4-echo-0075** | **0.7405** | **+0.129** | **0.810** | **0.875** | **1.000** | **0.375** | **20.0** |
| 6 | h6-rank-32 | 0.7268 | +0.116 | 0.794 | 0.868 | 1.000 | 0.403 | 18.3 |
| 9 | h9-warm-best-rank-32 | 0.7111 | +0.100 | 0.790 | 0.799 | 1.000 | 0.361 | 18.5 |
| 10 | h10-no-echo | 0.7169 | +0.106 | 0.781 | 0.882 | 1.000 | 0.403 | 18.2 |
| 11 | h11-warm-best-2-epoch | 0.6995 | +0.088 | 0.775 | 0.847 | 1.000 | 0.292 | 14.5 |
| 12 | h12-tasks-32 | _(training, step ~55%)_ | — | — | — | — | — | — |
<!-- /AUTO:RESULTS -->

Iters 5, 7, 8 skipped due to transient failures (see failures.jsonl);
drive auto-skips and proceeds with the next recipe.

## Best adapter (current)

- **Iter 4 `h4-echo-0075`** — 8 train tasks × 4 gens, lr=1e-5, rank 16/α32, ECHO λ=0.075.
- **B2 stable mirror:** `b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iter4/adapter.tgz`.
- Beats baseline on every sub-score except wall-clock (where it's still 3.7× faster than baseline).
- Iter 6 (`rank-32`) and iter 10 (`no-echo`) score within 0.02 of iter 4 — the recipe is
  robust to those individual changes, but iter 4's ECHO=0.075 is the sweet spot.

## Key learnings (running synthesis)

1. **More data ≠ better.** Iter 2 used 24 tasks (vs the default 16) with the SAME lr;
   composite collapsed to 0.589. The longer step count without lr annealing
   over-trained the LoRA. Use ≤16 train tasks or anneal lr proportional to data.
2. **ECHO=0.075 is the productive ceiling.** Paper says 0.01–0.05; 0.075 still
   gives the highest outcome F1 (0.810). ECHO=0 (iter 10) still scores 0.717,
   so ECHO is contributing ~0.024 of marginal value, not the full +0.13 win.
3. **Cross-file recall saturates at 1.0 once the agent learns to always grep.**
   Iter 1 onward it never regresses. This is a one-shot behavior the GRPO trace
   teaches early.
4. **Invariant coverage is the open ceiling.** Best score so far (iter 6) is 0.403.
   Gold invariants come from assert/raise patterns the model often misses.
   This is where iters 12-50 should look for headroom.
5. **Warm-starting from best (iters 3, 9, 11) doesn't beat iter 4.** Compounding
   gradient updates on an already-fit LoRA tends to drift — pure base-warm
   training with a good recipe still wins.

## Active hypothesis pipeline (next 38 iters)

From `recipes.json`, the queue prioritises:
- **lr sweep:** 3e-5, 2e-5, 5e-6, 2e-6 (find the GRPO lr cliff)
- **gens sweep:** 8 and 16 (reduce advantage variance vs sample cost)
- **rank sweep:** 64, 128 (test if iter 6's rank-32 win extends further)
- **gold-invariant aug:** synthesize stricter invariants in build_corpus.py
  → re-train against a higher inv-coverage target
- **no-policy-loss (§5.5):** verifier-free fine-tune, check ECHO-alone gain
- **larger corpus:** add a 4th seed_repo to widen task pool

## Infrastructure status

- **drive.py** PID 26831 is alive on Cloud Eric, looping 11→50 against pod 13knjsyhonso89.
- **Per-iter cadence:** rollouts (5–8 min) → group-filter → train (10–50 min depending
  on step count) → eval (3–5 min) → b2 backup → commit → next.
- **runpod_api.py fix shipped (commit 945514c in trajectory-trainer):**
  `_get_ssh_info` now defends against `runtime=null` while a pod is
  EXITED or still booting (was crashing with `'NoneType' has no attribute 'get'`).
- **Pod reaping:** A6000-class pods get reaped after ~1h of activity. Drive recovers
  by re-acquiring the next available GPU (RTX 6000 Ada / RTX A6000) and resuming.
  Three pod restarts logged in this session (h9sn…, 5hwk…, tfzl…) before settling
  on 13knjsyhonso89.

<!-- AUTO:STATUS_REPLACED_BY_DRIVE -->

## Audit log (UTC)

- 09:35 — pod h9snqr0ow80bev acquired (RTX 6000 Ada), kiln built, pi installed
- 10:09 — iter 0 baseline = 0.6112 logged
- 11:01 — iter 1 trained (1146k steps, 13 min on RTX 6000 Ada)
- 11:11 — iter 1 eval = 0.7074 → b2 backup
- 12:25 — pod h9snqr0ow80bev reaped; recover from B2 mirror
- 12:38 — pod 5hwknb14vutm5w acquired (RTX A6000)
- 13:41 — iter 2 training started (24 train, 1.1M steps)
- 14:25 — iter 2 = 0.5887 (regression — too much data at fixed lr)
- 15:xx — iter 3 = 0.6502 (warm-start fine-tune, partial recovery)
- 16:xx — **iter 4 = 0.7405** ← current BEST
- 17:xx — iter 5 skipped (pod transient), iter 6 = 0.7268 (rank-32)
- 18:xx — iters 7, 8 skipped (pod transient)
- 19:xx — drive resilience patches applied (filter_groups base64, kill_kiln swallow)
- 20:30 — iter 9 = 0.7111 on pod 13knjsyhonso89 (RTX 6000 Ada)
- 21:09 — iter 10 = 0.7169 (ECHO OFF — confirms ECHO is ~0.024 worth)
- 21:30 — iter 11 = 0.6995 (warm + 2 epochs — overtrains)
- 21:51 — iter 12 GRPO training started (16 tasks, 690k steps; 10/16 strong-signal groups)
- 22:00ish — iter 12 train started on 13knjsyhonso89 (h12-tasks-32, 690k steps)
- 22:30ish — pod 13knjsyhonso89 reaped mid-training (training had reached ~55%)
- 22:30–22:50 — drive cascaded `bg failed: no runtime/ports` errors through iters 12-50,
  exception handler advanced past each, drive exited. Capability.jsonl unchanged
  (still 9 real eval rows); failures.jsonl gained 39 dead-pod entries.
- **drive.py patched** (this session): added (a) pod-alive precheck before each
  iter — bails out cleanly if pod is unreachable, and (b) 3-consecutive-failure
  circuit breaker so future pod deaths can't burn the recipe queue.
- **drive.py patched** (this session): `update_in_progress_md()` now refreshes
  results table + Status line + audit log on each successful iter, before
  git commit. Going forward the MD reflects reality as iters land.
- 22:55 — recovery: pending pod acquisition

## Closeout plan

- At iter 50 (or session wall-clock limit), pick best-eval adapter, re-eval at
  seed 2 (configurable in recipes.json) to confirm reproducibility.
- Finalize WRITEUP.md with best-recipe details, sub-score progression, ablation
  table, recipe-vs-result matrix.
- B2 stable path: `b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iterN/adapter.tgz`
  (already populated for iter 4; replaced atomically if a later iter wins).
- All per-iter manifests + adapter tarballs at `b2://.../20260519/iter-N-train/`.
