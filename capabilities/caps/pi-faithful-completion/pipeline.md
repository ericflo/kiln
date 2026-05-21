---
schema_version: 1
capability: pi-faithful-completion
status: in-flight
base_round: round-3
base_sha256: Qwen3.5-4B
baseline_composite: 0.7237
final_composite: 0.8065
final_adapter: pi-faithful-h50-temp-0.6-x-light-x-lr-3e-5
stages:
  - {n: 1, method: agentic-grpo, slug: stage-1-grpo-h50-iter50, composite_after: 0.8065}
reproducer: ./run_pipeline.sh
wall_clock_estimate_min: 90
last_validated_ts: 2026-05-20T05:27:28Z
last_validated_base_round: round-1
---

# pi-faithful-completion pipeline (round-3 pilot)

This is the **first round-3 multi-stage pilot** — proves the new shape works
end-to-end. Stage 1 is the round-1 winner (preserved as the pipeline starting
point); stage 2 is fresh round-3 science (OPD polish hypothesis).

## Baseline (Qwen3.5-4B vanilla): 0.7237

Headroom analysis at baseline (from round-1 archive/eval-summaries):
- Total headroom: ~0.276
- Dominant residual after stage 1: `outcome.value_correct` ≈ 0.193 / `honesty.score` ≈ 0.161
- Sub-scores at ceiling: `no_question`, `no_soft_punt`, `format_strict`, `terseness`

## Stage 1: agentic-GRPO h50 (composite 0.7237 → 0.8065)

- **Method:** agentic-GRPO with ECHO
- **Adapter:** `pi-faithful-h50-temp-0.6-x-light-x-lr-3e-5`
- **Base:** Qwen3.5-4B vanilla
- **Round origin:** round-1 (backfilled into round-3 schema)
- **Recipe:** rank=16, alpha=32, lr=3e-5, ECHO λ=0.05, num_generations=4,
  rollout temperature=0.6, system_prompt=light, ~50 GRPO iters
- **Why agentic-GRPO** (METHODS.md Rule A): task is multi-turn pi tool-calling.
- **Evidence (round-1, n=57 eval):**
  - composite 0.8065 vs baseline 0.7237 (Δ +0.0828)
  - outcome.value_correct +0.0877; honesty.score +0.0667
  - format_strict -0.0351 and terseness -0.0217 (small drops, dwarfed)
- **B2 archive:** `b2://clouderic/capabilities/pi-faithful-completion/adapters/pi-faithful-h50-temp-0.6-x-light-x-lr-3e-5.tar.gz`
- **Caveats:**
  - Round-1 result is single-seed (round-3 will re-eval 3-seed to verify
    σ-bounded reproducibility, see PIPELINE.md §4.5)
  - Round-1 used a pre-receipt trainer; no `train_receipt.json` exists
  - Sibling regression check was not run in round 1 (round 3 will run it
    before promoting stage 2)

## Stage 2: OPD polish (planned)

- **Method:** OPD against 27B Qwen3.6 teacher
- **Adapter:** `pi-faithful-completion-stage-2-opd-polish` (to be trained)
- **Base:** `pi-faithful-h50-temp-0.6-x-light-x-lr-3e-5` (stage-1 output)
- **Hypothesis:** OPD on top of stage-1 lifts the residual `outcome.value_correct`
  (currently 0.807) and `honesty.score` (currently 0.839) by closing the
  distribution gap to a stronger teacher on edit-context cases. Target:
  composite 0.8065 → 0.86 (Δ +0.05) with no sibling regression.
- **Why OPD next** (METHODS.md Rule E + §4.2 SFT→OPD pattern adapted to
  GRPO→OPD): stage-1 stabilized format and brought composite into [0.4, 0.8].
  A stronger teacher in the same family exists (Qwen3.6-27B). Process
  sub-scores still have ≥ 0.08 headroom. Stage-1 high-baseline failure mode
  (cap #5) risk: baseline composite 0.807 is at the edge of the 0.80
  watch zone; OPD recipe will use conservative defaults (rank=16, lr=5e-5,
  3 epochs initially) and watch for regression.
- **Falsification plan:** if OPD regresses below stage-1 by σ, revert to
  stage-1 as the cap's shipped winner and document the dead-end. If it
  saturates within σ, accept stage-1 as the ceiling. If it lifts >+0.04
  with no sibling regression, promote as stage 2.
- **Method-specific recipe (initial):** rank=16, alpha=32, lr=5e-5 (gentle —
  the cap #5 high-baseline risk caveat), 3 epochs, samples_per_prompt=1,
  teacher=qwen3.6-27b-awq.

## Stage 3: TBD

Will depend on stage-2 outcome:
- If stage-2 succeeds → consider agentic-GRPO `--no-policy-loss` for
  verifier-free continuation (METHODS.md §4.2 within-method variant).
- If stage-2 nulls / regresses → stop the pipeline, accept stage 1 as
  the round-3 final adapter, file a kiln issue if the OPD pattern was
  blocked by tooling.

## Reproducer

```bash
./run_pipeline.sh        # re-runs all stages from this header
./run_pipeline.sh --from-stage 2   # skip stage-1 backfill, run only stage 2
./run_pipeline.sh --validate-only  # eval stages against current base
```

## Round transitions

- **round-1 → round-3:** structure-only carry-forward. The recipe, hyperparams,
  rubric, and rollout discipline all carry verbatim. The new artifact is
  this multi-stage `pipeline.md` plus the `stages/stage-1-*.json` record.
- **round-3 stage 2:** the actual experimental work for this cap. First
  round-3 multi-stage pilot.
