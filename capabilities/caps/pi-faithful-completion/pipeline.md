---
schema_version: 1
capability: pi-faithful-completion
status: needs-revalidation
base_round: round-3
base_sha256: Qwen3.5-4B
baseline_composite_round_3: 0.6733
baseline_composite_round_1: 0.7237
final_composite: 0.6544
final_adapter: pi-faithful-h50-temp-0.6-x-light-x-lr-3e-5
stages:
  - {n: 1, method: agentic-grpo, slug: stage-1-grpo-h50-iter50, composite_after: 0.6544, status: needs-revalidation}
reproducer: ./run_pipeline.sh
wall_clock_estimate_min: 90
last_validated_ts: 2026-05-21T21:23:00Z
last_validated_base_round: round-3
---

# pi-faithful-completion pipeline (round-3 pilot — STRUCTURAL FINDING)

This is the **first round-3 multi-stage pilot.** It produced an unexpected
but extremely valuable structural finding rather than the planned +12pp
chain. The finding deserves to be documented as the main artifact of this
pilot.

## Headline finding

> **The round-1 winner (+8.3pp single-stage) no longer beats base when
> evaluated under round-3 `kiln serve --eval-mode`. Round-1's eval was
> measurably more permissive than round-3's.**

This is exactly the kind of insight the round-3 unification was built to
surface. Multiple round-1 winners likely need re-validation under the new
eval discipline before any multi-stage pipeline is committed to.

## Round-1 vs round-3 paired eval (n=57, single seed=3141592653)

| Metric | Round-1 (reported) | Round-3 (measured) | Δ |
|---|---|---|---|
| Baseline composite | 0.7237 | **0.6733** | -0.050 |
| Adapter composite  | 0.8065 | **0.6544** | -0.152 |
| Adapter vs base    | **+0.0828** | **-0.0189** | -0.102 |
| outcome.value_correct (adapter) | 0.807 | 0.649 | -0.158 |
| honesty.score (adapter) | 0.839 | 0.728 | -0.111 |
| format_strict (adapter) | 0.947 | 0.965 | +0.018 |
| terseness (adapter) | 0.959 | 0.951 | -0.008 |

The adapter's `outcome.value_correct` and `honesty.score` dropped by 10-16
points; the format sub-scores stayed roughly stable. The lift in round-1
was concentrated in outcome+honesty; both regressed in round-3.

## Why? (working hypotheses)

Round-3 server differs from round-1 in several axes simultaneously, and
no single-axis ablation has been run yet. The leading candidates:

1. **Thinking-mode default flipped.** Round-3 `--eval-mode` sets
   `eval_mode_default_thinking_enabled=false`. Round-1 may have run with
   thinking on by default. Arithmetic tasks (compute_avg, compute_sum)
   need chain-of-thought to solve reliably; without it, the model emits
   short wrong answers fast.
2. **Stricter eval-mode determinism.** Round-3 eval-mode enforces
   transient cache cleanup between completions (kiln #15). Round-1 likely
   had warmer state between rollouts; KV cache reuse may have stabilized
   sampling.
3. **Kiln serve commit drift.** Round-1 used kiln circa 2026-05-20;
   round-3 is post-2026-05-21 (40-issue backlog complete, including
   tightened adapter semantics in #1 and adapter-load validation in #2).
   Adapter-selection semantics moved during the backlog.
4. **Adapter overfit to round-1 server idiosyncrasies.** The adapter was
   trained with rollouts produced by round-1's server. If round-3 sampling
   differs even slightly, the policy gradient's local minimum may not
   transfer.

The fact that **the baseline also dropped by 5pp** (0.724 → 0.673) is
strong evidence the issue is the server, not the adapter alone. Both
runs use Qwen3.5-4B weights from the same `/workspace/Qwen3.5-4B`.

## Implications for round 3

1. **All round-1 winners need re-validation** under round-3 eval-mode
   before they're treated as kept stages. This is a Phase E discovery —
   the planned mechanical per-cap migration must include a re-baseline +
   re-eval step, not just a metadata wrap-around of round-1 numbers.
2. **The round-3 baseline is the new reference.** Headroom analysis,
   sub-score targeting, and `lib/method_router.py` recommendations must
   key off round-3 baselines, not the round-1 archive numbers.
3. **Phase G distillation can't trust round-1 cluster manifests.** The
   sibling matrix must be regenerated against round-3 paired evals.
4. **The METHODS.md decision tree's rule B-strict (baseline < 0.3 may be
   over-strict rubric) deserves a sibling rule:** "if baseline shifts by
   > 0.03 between server versions, the rubric measurement is unstable
   and the prior win is suspect." This belongs in the round-3 lessons.
5. **Stage 2 OPD polish is no longer obviously the right next step.**
   The original hypothesis (OPD on top of +8.3pp stage-1) presumed the
   stage-1 lift existed. With stage-1 actually at -1.9pp under round-3
   eval, OPD on top would polish a non-win. Pause stage 2 until stage 1
   is properly recovered.

## What stage 1 actually shows now

The stage-1 record (`stages/stage-1-grpo-h50-iter50.json`) preserves the
round-1 reported numbers (it's a historical record) and the round-3 numbers
land in `capability.jsonl` iter 2 as the re-validation row. The stage's
`status` is now `needs-revalidation` rather than `kept`.

## Next steps (for whoever picks this up)

### Tier 1 — Diagnose the regression

1. **Single-axis ablation: thinking on vs off.** Re-run eval with
   `enable_thinking=true` in the chat-template kwargs. Hypothesis: thinking
   on restores base composite to ~0.72 and adapter to ~0.80.
2. **Kiln commit bisect.** Pin kiln to the round-1 commit (just before the
   40-issue backlog landed) and re-eval. Identifies which kiln commit
   shifted the eval baseline.
3. **Decoding-determinism check.** Compare per-task outputs (round-3 vs a
   recovered round-1 trace) and quantify which tasks differ.

### Tier 2 — Re-establish stage 1

Once diagnosed:
- Option A: re-train the agentic-GRPO stage on the round-3 server's
  rollout distribution. This is the proper round-3 stage-1.
- Option B: keep the round-1 adapter and document the cross-version
  caveat in pipeline.md — useful only if you serve the model with the
  round-1 server config.

### Tier 3 — Then stage 2 OPD

The OPD polish hypothesis is still interesting, but only on top of a
verified-under-round-3 stage 1. Don't chain OPD on a non-win.

## Pod evidence (this session)

- **Pod:** `m5qfrqcbwt16pe` (A6000), lease `pod-ad0999c0694eca57da9716df`
- **Kiln commit:** `66fa0782`
- **Adapter restored:** `b2://clouderic/capabilities/pi-faithful-completion/adapters/pi-faithful-h50-temp-0.6-x-light-x-lr-3e-5.tar.gz`
- **Adapter SHA verified:** matches B2 archive
- **`kiln adapter verify`:** PASS (loadable=true, behavioral=measurable L2 delta 28.04)
- **Base eval log:** `/tmp/eval-base/summary.json` on pod
- **Adapter eval log:** `/tmp/eval-stage1/summary.json` on pod (no system prompt)
- **Adapter eval with light-system-prompt log:** `/tmp/eval-stage1b/summary.json` on pod (worse — composite 0.498)
- **Eval wall-clock:** 81s base, 167s adapter no-prompt, 495s adapter with light prompt

## Reproducer (for the regression itself)

```bash
# On a pod with kiln 66fa0782 + Qwen3.5-4B + round-1 adapter restored:
cd /workspace/kiln/capabilities/caps/pi-faithful-completion

# 1. Start kiln serve in round-3 eval-mode
KILN_MODEL_PATH=/workspace/Qwen3.5-4B \
KILN_ADAPTER_DIR=/workspace/adapters \
KILN_DEFAULT_THINKING_ENABLED=false \
/workspace/kiln/target/release/kiln serve --eval-mode &

# 2. Eval base (no adapter)
curl -X POST http://localhost:8420/v1/adapters/unload -d '{}'
python3 rollout.py --tasks datasets/eval.tasks.jsonl \
  --out-dir /tmp/base --mode eval --num-generations 1 \
  --temperature 0.2 --top-p 0.95 --max-tokens 768 \
  --seed 3141592653 --concurrency 3

# 3. Eval adapter
python3 rollout.py --tasks datasets/eval.tasks.jsonl \
  --out-dir /tmp/adapter --mode eval --num-generations 1 \
  --adapter pi-faithful-h50-temp-0.6-x-light-x-lr-3e-5 \
  --temperature 0.2 --top-p 0.95 --max-tokens 768 \
  --seed 3141592653 --concurrency 3

# 4. Compare summary.json::mean_composite for each
```

## Round transitions

- **round-1 (2026-05-19/20):** original 50-iter loop, iter 50 wins at 0.8065
- **round-3 re-validation (2026-05-21):** round-1 adapter under round-3
  eval-mode produces 0.6544 vs base 0.6733 → adapter is now worse than base
- **round-3 cap status:** `needs-revalidation`. Pipeline.md is the source
  of truth that the round-1 win does not survive the round-3 eval-mode
  upgrade until diagnosed and re-trained.

## Lessons that should propagate to other caps

1. **A backfill is not a validation.** Migrating round-1 stages into
   round-3 pipeline.md must include a paired eval under round-3 server
   conditions before the stage gets `status: kept`.
2. **Eval-mode is load-bearing.** The same model + adapter + tasks under
   different eval-mode discipline can produce composite shifts > 10pp.
   This is exactly why round-3 mandates `--eval-mode`.
3. **Round-over-round drift is real and must be measured.** This finding
   is also the motivating example for DISTILLATION.md §4.2 "Cross-cap
   matrix on new base" — when base behavior shifts, prior wins must be
   re-validated, not assumed.

## Closing note

The Phase C pilot was structured to validate that the round-3 multi-stage
shape works end-to-end. It did — the structure shipped cleanly, the
adapter loaded, the eval ran. The unexpected result (round-1 adapter
regresses under round-3 eval) is a more important finding than a clean
+12pp would have been. It tells us:

> The unification work isn't optional polish. It's the eval discipline
> that catches the gap between "we trained it" and "it actually works."
> Round-1 winners need re-validation before they can serve as stage-1
> backfills in round-3 pipelines.

This is exactly why we built METHODS.md / PIPELINE.md / cross-cap-coherence.
The pilot proves the discipline catches what the old harness didn't.
