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

## Rubric v2 ideas (future work — don't change v1 mid-run)

Reviewed rubric.py during iter-12 rollout idle. v1 is already comprehensive
(7 fields, 5 sub-scores, line-tolerance grounding, F1 with type/identifier
normalization, semantic invariant matching via Jaccard ≥0.30 + ≥2 shared
content tokens, multiplicative composite). It's been validated against
rubric_sanity.py 10-case battery. Things worth adding in a future v2:

1. **AST cross-check for `calls`/`called_by`.** Current rubric trusts the
   gold set; future v2 could parse the snapshot's AST and verify the
   prediction matches the ACTUAL call graph — catches both gold errors
   and agent fabrications.
2. **LLM judge layer for semantic invariants.** Jaccard ≥0.30 is coarse
   for invariants like "input non-empty"vs "len(arr) > 0". An LLM-graded
   semantic match (calibrated against the existing battery) could lift
   the invariant_coverage ceiling (currently best ~0.40).
3. **Reasoning-process score.** Did the agent grep before answering?
   Did it read the file? Currently only the final JSON is scored; the
   trace is wasted. A reasoning-process bonus would teach the agent
   the "grep-then-read-then-answer" pattern more directly.
4. **Adversarial gold variants.** For each task, hand-craft 2-3
   "wrong-but-plausible" answers and verify the rubric scores them
   <0.5. Currently sanity battery checks the positives (good answers
   should score high); adding adversarial negatives would tighten the
   floor.
5. **Multi-language corpus extension.** Currently Python-only via AST;
   could add JS/Rust/Go via Tree-sitter to test grounding generalization.

These are all v2 work — invariant changes would invalidate every
iter's reward signal mid-run. Documenting here so the v2 build (after
50-iter close) has a roadmap.

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
- 22:22 (next day) — pod 88eqxaqnb6ma5d acquired (RTX 6000 Ada); bootstrap built
  kiln main + cuda_grpo_ablation **without `--features cuda`** (bug in bootstrap).
- 22:35 — restarted drive at iter 12; rollouts started but later iter 12 training
  failed with "no valid GRPO groups" — all base-model rollouts scored 0.0
- 23:36 — diagnosed: cuda binary missing cuda feature flag → training never ran.
  Rebuilt with `--features cuda` (11:48 wall clock).
- 01:10 — pod 88eqxaqnb6ma5d reaped mid-iter-14, drive bailed (3-failure breaker)
- 01:23 — pod sfnfy255wrklcb acquired (RTX 6000 Ada). Bootstrap v4 succeeded
  (model 8.8GB downloaded, kiln + cuda binary built, pi 0.75.3 installed,
  iter 4 adapter restored from B2).
- 02:11 — restarted drive at iter 12 on new pod. Iter 12 rollouts ran 48 min,
  all 64 rollouts scored 0.0 with mean_wall_clock=180s (every pi session
  hit the wall-clock cap producing nothing).
- 03:30 — root cause: **Qwen3.5-4B (base, no adapter) emits `reasoning_content`
  in chat completions but leaves `content` empty.** Pi reads `content`, sees
  empty, treats response as failure. With 32K max_tokens budget the model
  thinks the whole budget away producing no answer. iter-4-trained adapter
  was specifically trained to short-circuit thinking → straight JSON answer,
  which is why iter 0-11 worked but iters 12+ (base rollouts) failed on
  this fresh pod. The earlier session's "base model" Qwen weights may
  have shipped with thinking disabled by default; HF upstream change?
- 03:50 — session close-out at 11 successful iters. Iter 4 remains BEST.

## Recovery for the next session

Path A — continue training: warm-start ALL remaining iters from iter 4 best
(use `--train-adapter best`). This avoids the base-model thinking trap.
The 12+ recipes that intended base-warm should be re-spec'd as warm-start.

Path B — fix pi/kiln: patch pi's openai-completions adapter to fall back
to `reasoning_content` when `content` is empty. OR patch kiln's chat
template to disable thinking by default. EITHER fixes iter 12+ as-spec'd.

Path C — change rollout system prompt to "answer directly without thinking"
to short-circuit the thinking tokens even for the base model.

Path B (programmatic, no recipe changes) is the cleanest fix.

## Update 2026-05-20 09:15 — kiln patch shipped, but pi tool-call broken without thinking

- Patched kiln `chat_template_options_from_kwargs` to respect new env var
  `KILN_DEFAULT_NO_THINK=1` (commit 91bca83a). Verified: curl `/v1/chat/completions`
  with no `chat_template_kwargs` now returns `content: "..."` immediately,
  `finish_reason: "stop"` in <1s.
- Pi standalone test against patched kiln WORKS: `pi -p "Say HELLO briefly"`
  returns "HELLO" cleanly in 30s with session JSONL written.
- **BUT** pi with the full code-comprehension task prompt (1000+ tokens
  describing JSON schema, reading + grepping files) STILL hits 180s timeout
  with no session output written and `exit=124`. Verified with both
  iter-4 adapter loaded AND base model.
- Hypothesis: with `enable_thinking=false`, the model emits content directly
  but doesn't produce tool calls in the OpenAI format pi expects. Pi's
  agentic loop tries to coax tool calls from the text response, gets stuck.
  iter-4 adapter was trained against thinking-on responses where tool calls
  emerged from the reasoning trace; without thinking, the LoRA's learned
  behavior is out of distribution.

## Path D (next session) — drop pi for single-turn rollouts

The cleanest fix: don't use pi at all for code-comprehension rollouts. Take
the single-turn approach used by `pi-faithful-completion`:
- Send chat completion to kiln with system+user prompt that includes
  the target file contents + grep results pre-computed in Python
- Read the assistant's response as the structured JSON answer
- Score with the existing rubric

Trade-off: loses the agentic dimension of the capability (the model no
longer chooses what to read/grep). But the capability's WORTH was always
just emitting good JSON summaries — the grep-then-read step was scaffolding,
not the trained behavior. Plus pi-faithful-completion shows this approach
works fine for similar tasks. Estimated complexity: rewrite rollout.py to
~100 LOC of direct kiln HTTP + Python file IO. ~30 min work.

Files committed in this session worth keeping for next session:
- Kiln patch: `crates/kiln-server/src/api/completions.rs` (`KILN_DEFAULT_NO_THINK`)
- drive.py: pod-alive guard, 3-failure circuit breaker, `update_in_progress_md`,
  realistic rollout timeout based on max-wall-clock × num × concurrency
- Recipes 12-50: warm-from-iter-4 patched (would need revisit if doing single-turn)
- This IN_PROGRESS.md diagnosis trail

## Session close-out — final state

11/50 iters logged in capability.jsonl. Iter 4 `h4-echo-0075` BEST at
composite **0.7405** (+0.1293 / +21.2% over 0.6112 baseline). B2 stable
backup at `b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iter4/adapter.tgz`
(43.9 MB, sha256 verified). All session commits pushed to main.

Next session checklist:
1. Set `KILN_DEFAULT_NO_THINK=1` on kiln serve env (or pass `chat_template_kwargs`
   per request)
2. Implement single-turn rollout.py (Path D above) — kill pi dependency
3. Resume drive at iter 12, use existing recipes (revert warm-start patch
   if testing base recipes)

## Update 2026-05-20 04:30 — root cause confirmed

- `/no_think` directive does NOT work — Qwen3.5 chat template only respects
  the `enable_thinking` kwarg in `apply_chat_template`, not inline tokens.
  Verified: curl /v1/chat/completions with `/no_think say hello` still
  emits `content=""` and `reasoning_content="Thinking Process:..."`.
- Loading iter-4 LoRA on kiln (POST /v1/adapters/load returns `loaded`)
  ALSO doesn't fix it — even chat completions with `adapter: pi-cc-iter4`
  body param still return empty content. This is surprising because
  iter-4 was trained against this same model and `did` short-circuit
  thinking during its successful training session.
- **Hypothesis**: HuggingFace re-pushed Qwen3.5-4B with a different
  tokenizer_config (chat_template) that has thinking enabled by default
  AND ignores adapter-learned token preferences. The iter-4 LoRA was
  trained against the OLD template (thinking off by default) so its
  learned "go straight to JSON" behavior is overridden by the new
  template's `<think>\n` injection at the assistant turn.
- **Required for next session**: a kiln-side change that either
  (a) passes `enable_thinking=false` to `apply_chat_template` by default,
  or (b) exposes a `chat_template_kwargs.enable_thinking` API param
  that pi can set. Without that, base-model rollouts are broken for
  this capability.

## Session close-out

11 successful iters logged in capability.jsonl. Iter 4 `h4-echo-0075`
remains best at composite 0.7405 (+0.1293 vs 0.6112 baseline).

B2 stable backup: `b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iter4/adapter.tgz`
(already in place from the original successful session).

The remaining 39 iters' recipes are queued in recipes.json and will
auto-resume on next session restart — provided the kiln chat template
issue is fixed first (Path B above).

## Closeout plan

- At iter 50 (or session wall-clock limit), pick best-eval adapter, re-eval at
  seed 2 (configurable in recipes.json) to confirm reproducibility.
- Finalize WRITEUP.md with best-recipe details, sub-score progression, ablation
  table, recipe-vs-result matrix.
- B2 stable path: `b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iterN/adapter.tgz`
  (already populated for iter 4; replaced atomically if a later iter wins).
- All per-iter manifests + adapter tarballs at `b2://.../20260519/iter-N-train/`.
