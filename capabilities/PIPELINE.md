# `PIPELINE.md` — Multi-stage capability pipelines

This is the **operating manual** for capability pipelines. A pipeline is
an ordered sequence of stages, each producing one trained adapter. The final
stage's adapter is the capability's shipped artifact.

Read [`METHODS.md`](METHODS.md) first — it tells you which method to pick at
any given stage. This doc tells you how stages chain, validate, and produce
a stable, reproducible result.

The companion doc on what happens when a *cluster* of pipelines is rolled
into a new base is [`DISTILLATION.md`](DISTILLATION.md).

---

## §1. Why pipelines

Round 1 evidence (`CONSOLIDATED_REPORT.md`):

- `pi-diff-patch-apply` saturated under GRPO from base 0.94. The lesson was
  *"use base, ECHO-only regularization, OPD, or harder evals"* — a sequence
  of methods, not one.
- OPD cap #5 (diff-patch-fluency) regressed catastrophically from baseline
  0.85. The recovery path is "SFT on teacher rollouts," another method.
- `pi-doctest` single-seed +11pp turned out to be +4.2pp 3-seed. The
  reproducible recipe needed a warm-best re-run as the next stage.

A capability's maximum lift on a 4B model is usually the sum of two or three
methodology-stages chained, not a single trainer running 50 iters.

---

## §2. Stage definitions

### §2.1 What a stage is

A **stage** is one trained adapter produced by one method, with:

- a stage number (1-indexed)
- a method (`sft` | `opd` | `grpo` | `agentic-grpo`)
- a base_adapter (the previous stage's output, or `null` for stage 1)
- an output_adapter (the trained adapter from this stage)
- a falsifiable hypothesis stated up front
- ≥ 3-seed eval evidence
- a kept iter row in `capability.jsonl` with `status: "kept"`

A stage corresponds to a `stages/stage-<N>-<slug>.json` file.

### §2.2 What a stage is NOT

- A single training iter. Iters can be exploratory; only kept-and-promoted
  iters become stages.
- A method change in the middle of training. Method switching = new stage.
- A `--base-adapter` chain inside one method without re-eval between. Those
  are warm-best continuations (single stage, multiple iters).
- A hyperparameter sweep. Sweeps are exploration; only the winner promotes.

### §2.3 stages/stage-<N>-<slug>.json schema

```json
{
  "schema_version": 1,
  "stage": 2,
  "slug": "stage-2-opd-polish",
  "method": "opd",
  "hypothesis": "OPD on top of stage-1 SFT, with the 27B teacher, lifts the process_faithfulness sub-score from 0.55 to ~0.75 by closing the distribution gap on edit-context cases.",
  "stage_transition_rationale": "Stage 1 SFT closed with format=0.78 (gate ≥0.7 passed). Process headroom remained at 0.32 of total. Teacher available at composite 0.94 on this cap. Rule E in METHODS.md fires.",
  "base_adapter": "<cap>-stage-1-sft-bootstrap",
  "output_adapter": "<cap>-stage-2-opd-polish",
  "training_iters": [
    {"iter": 4, "slug": "h1-opd-default", "composite": 0.83, "status": "ablation"},
    {"iter": 5, "slug": "h2-opd-rank8",   "composite": 0.81, "status": "ablation"},
    {"iter": 6, "slug": "h3-opd-2epoch",  "composite": 0.852, "status": "kept"}
  ],
  "promoted_iter": 6,
  "final_composite": 0.852,
  "final_composite_delta": 0.104,
  "final_sub_scores": {...},
  "cross_cap_check": {"max_sibling_delta": -0.008, "sigma": 0.011, "passed": true},
  "adapter_manifest": "/workspace/adapters/<cap>-stage-2-opd-polish/adapter_manifest.json",
  "train_receipt":    "/workspace/adapters/<cap>-stage-2-opd-polish/train_receipt.json",
  "kiln_commit": "<sha>",
  "ts_promoted": "2026-06-08T..."
}
```

The promoted iter's `capability.jsonl` row has `stage: 2` matching this file.

### §2.4 stages/ directory invariant

> Every file in `stages/` corresponds to **exactly one kept iter** in
> `capability.jsonl`. The kept iter's `output_adapter` and `composite` fields
> must match `stages/stage-<N>-<slug>.json`.

`lib/stage_manifest.py` validates this. `run_pipeline.sh` aborts on divergence.

---

## §3. The pipeline.md file

Per-cap front matter + prose. The header is machine-parseable; the body is
agent-and-human-written.

### §3.1 Header schema

```yaml
---
schema_version: 1
capability: pi-faithful-completion
status: shipped | in-flight | retired
base_round: round-2
base_sha256: <sha of Qwen3.5-4B>
baseline_composite: 0.612
final_composite: 0.918
final_adapter: pi-faithful-completion-stage-3-grpo-final
stages:
  - {n: 1, method: sft,          slug: stage-1-sft-bootstrap, composite_after: 0.748}
  - {n: 2, method: opd,          slug: stage-2-opd-polish,    composite_after: 0.852}
  - {n: 3, method: agentic-grpo, slug: stage-3-grpo-final,    composite_after: 0.918}
reproducer: ./run_pipeline.sh
wall_clock_estimate_min: 45
last_validated_ts: 2026-06-15T...
last_validated_base_round: round-2
---
```

### §3.2 Body

```markdown
# <cap> pipeline

## Baseline (round-2 base): 0.612
- Headroom: 0.388
- Concentration: 0.21 outcome, 0.10 process, 0.07 format

## Stage 1: SFT bootstrap (composite 0.612 → 0.748)
- Method: SFT
- Adapter: <cap>-stage-1-sft-bootstrap
- Base: none
- Recipe: rank=4, alpha=8, lr=1e-4, 64 teacher samples, 1 epoch
- Why SFT first (METHODS.md Rule D fired): baseline format sub-score was 0.30
  with high variance; SFT on curated examples gives the cheapest format-fix.
- Evidence: 3-seed mean +0.136 ± 0.014; anchor regression -0.004 (within σ).

## Stage 2: OPD polish (composite 0.748 → 0.852)
- Method: OPD
- Adapter: <cap>-stage-2-opd-polish
- Base: <cap>-stage-1-sft-bootstrap
- Recipe: rank=16, alpha=32, lr=1e-4, 6 epochs, samples_per_prompt=2,
  teacher=qwen3.6-27b-awq on :8002.
- Why OPD next (METHODS.md Rule E fired): after format was stable (0.78),
  process sub-scores still at 0.55; teacher composite=0.94 on this cap, so
  distribution gap to teacher is real.
- Evidence: 3-seed mean +0.104 ± 0.011; env_ce dropped from 2.1 to 1.6.
- Sibling check: max delta -0.008 < threshold; passed.

## Stage 3: agentic-GRPO final (composite 0.852 → 0.918)
- Method: agentic-GRPO + ECHO
- Adapter: <cap>-stage-3-grpo-final
- Base: <cap>-stage-2-opd-polish
- Recipe: rank=16, alpha=32, lr=1e-5, ECHO λ=0.05, --filter-var-min 0.05,
  --num-generations 4.
- Why agentic-GRPO last (METHODS.md Rule A + F fire): task is multi-turn so
  agentic-GRPO is the eventual method anyway; reward variance on stage-2
  rollouts had risen to 0.07 (above 0.05 threshold) on hard_eval, opening
  the policy-gradient lever.
- Evidence: 3-seed mean +0.066 ± 0.012; hard_eval composite 0.78 → 0.84.
- Sibling check: max delta -0.011 < threshold; passed.

## Reproducer
./run_pipeline.sh
# ~45 min on A6000. Runs all 3 stages with --base-adapter chaining.
# Each stage runs rubric_sanity → train → adapter verify → 3-seed eval →
# cross-cap-coherence between stages.

## Round transitions
- round-1 → round-2: no change (this cap is round-2 native).
- round-2 → round-3 (planned): re-baseline after distillation; predicted
  collapse to 2 stages (SFT bootstrap consolidated into base).
```

### §3.3 How pipeline.md is built

The agent does NOT write pipeline.md from scratch. The flow is:

1. After a stage's promoted iter lands, `lib/stage_manifest.py` updates the
   header `stages:` array and appends a stub body section for the new stage.
2. The agent fills in the prose rationale and evidence section.
3. `run_pipeline.sh` validates the header against `stages/` + `capability.jsonl`
   on every run.

---

## §4. The stage loop

This is the actual procedure run by `run_stage.sh <method> <slug>`. Reference
implementation lives in
`.agents/skills/capability-creator/templates/run_stage_<method>.sh`.

### §4.1 Pre-stage gates (mandatory)

Before any GPU work in a stage N ≥ 2:

```
1. Verify previous stage is shipped:
   - stages/stage-(N-1)-*.json exists AND was kept
   - kiln adapter verify <previous-output-adapter> passes loadable+behavioral

2. Verify methods.md routing:
   - run lib/method_router.py --eval-summary <prev-stage-eval> --print
   - confirm the recommended method == this stage's method
     OR the pipeline.md stage_transition_rationale documents an override

3. Verify hypothesis is falsifiable:
   - stage_transition_rationale names the sub-score(s) expected to move
   - states the magnitude (e.g. "+0.05 in process_faithfulness")
   - states the failure case (what we'd see if hypothesis is wrong)

4. Verify rubric_sanity passes (always):
   - python3 rubric_sanity.py
   - calibration good vs bad separation margin > 0.2

5. Verify integration sanity:
   - integration/cross-cap-coherence/capability.oracle.sh <previous-adapter>
   - max sibling delta within tolerance
   - if not, the previous stage already silently regressed siblings; fix that first
```

### §4.2 Stage iters

Each iter inside a stage is identical to round-2 iters today (rubric_sanity,
build rollouts or prompts, dry-run, train, adapter-verify, 3-seed eval, append
capability.jsonl row). The only difference is the iter row carries
`stage: N` and `method: <m>` and `base_adapter: <prev>` fields.

A typical stage runs 1–5 iters internally (variants of the recipe). The
*kept* iter — meeting the §4.3 promotion criteria — becomes the stage.

### §4.3 Stage promotion criteria

An iter is promoted to a stage if ALL hold:

1. **Composite delta vs previous stage** ≥ +0.05 (single stage) OR
   the kept iter is itself an exploration of a different hypothesis that
   moved a sub-score the previous stage didn't (record this explicitly).
2. **3-seed mean** > 2σ above previous stage's 3-seed mean.
3. **`kiln adapter verify`** passes (loadable + behavioral).
4. **Cross-cap regression check** — `integration/cross-cap-coherence/` reports
   max sibling delta > -0.02.
5. **Prior-stage preservation** — the new adapter, evaluated against the
   *previous* stage's eval set, scores ≥ previous stage's composite − σ.
   (Prevents the new stage from clobbering the old stage's domain.)

Promotion is mechanical once these pass. Skipping any of them is a process
violation and the stage is invalid.

### §4.4 Stage rejection and recovery

If a stage fails promotion criteria:

- **#1 or #2 fails (didn't move):** try 1–2 variants within the stage (lower
  lr, different filter, different teacher temperature, etc.). If still null,
  abandon the stage. The previous stage's adapter remains the cap's current
  best.
- **#3 fails (adapter broken):** debug load layout, retraining is almost certainly
  needed. Don't promote a non-loadable adapter.
- **#4 fails (cross-cap regression):** the new method clobbered a sibling
  capability. Options:
  - Lower the new stage's learning rate or rank
  - Reduce stages — maybe this cap is single-stage on the current base
  - Accept the previous stage as the cap's final and document the sibling
    constraint in pipeline.md
- **#5 fails (prior-stage clobber):** the new stage erased some of the prior
  stage's domain. This is a strong negative result. Options:
  - Lower learning rate / shorter training
  - Use a method that adds capability rather than reshapes (OPD often adds;
    GRPO sometimes reshapes)
  - Stop the pipeline at the previous stage.

---

## §5. run_stage.sh contract

Signature:

```bash
./run_stage.sh <method> <slug> [--base-adapter <name>] [--iter <N>]
```

Behavior:

1. Load `capability.config.json` `methods.<method>` defaults.
2. Resolve base_adapter from `--base-adapter` or from the previous stage's
   `output_adapter` in `stages/stage-(N-1)-*.json`.
3. Pre-stage gates (§4.1) if N ≥ 2.
4. Run `rubric_sanity.py`.
5. Build any method-specific data files if missing (calls `build_corpus.py`
   with `--method <m>`).
6. Run dry-run validation (`cuda_*_ablation --dry-run` where applicable).
7. Real training, with `--install-adapter-dir` and `--adapter-smoke-test`.
8. `kiln adapter verify`.
9. 3-seed eval via `kiln eval-adapter`.
10. `integration/cross-cap-coherence/capability.oracle.sh <new-adapter>`.
11. Append iter row to `capability.jsonl` with `stage` + `method` +
    `base_adapter` + `output_adapter`.
12. If promotion criteria pass, also write `stages/stage-N-<slug>.json` and
    update `pipeline.md` header.
13. Exit 0 on kept, 1 on null, 2 on regression, 3 on cross-cap regression.

Reference implementation: `.agents/skills/capability-creator/templates/run_stage_*.sh`.

---

## §6. run_pipeline.sh contract

Signature:

```bash
./run_pipeline.sh [--from-stage N] [--validate-only]
```

Behavior:

1. Parse `pipeline.md` header. Validate against `stages/` and `capability.jsonl`.
2. For each stage in order:
   - If `--validate-only`, just `kiln adapter verify <stage-adapter>` and
     re-run the eval. Report drift, don't train.
   - Otherwise, call `run_stage.sh <method> <slug> --base-adapter <prev>`.
3. After all stages, run final integration check on the chain's final adapter.
4. Print a summary: per-stage composite, total wall clock, final adapter
   path, cross-cap matrix.

`--from-stage 2` is useful when re-running after a base refresh: stage 1
may now be consolidated into the new base; the agent decides whether to
re-include or skip it.

---

## §7. The base-refresh flow

When a new base lands (after distillation, see DISTILLATION.md):

1. `lib/stage_manifest.py --check-base-drift <cap>` compares
   `pipeline.md::base_sha256` to current base. Reports drift.
2. The cap's pipeline must be revalidated. Run
   `./run_pipeline.sh --validate-only`.
3. If validation passes (composite within σ of recorded values), the
   pipeline still holds on the new base. Update `last_validated_base_round`.
4. If validation fails, the pipeline must be re-run:
   - `./run_pipeline.sh` (full re-train)
   - Most likely outcome: stage 1 (SFT bootstrap) is no longer needed because
     its capability is in the new base. Pipeline collapses to 2 stages.
   - If the *final* composite no longer exceeds the new base's baseline
     by > 0.05, the capability is **consolidated** — mark `status: retired`
     in pipeline.md and add a `## Round transitions` note.

Consolidated capabilities are wins. The whole point of distillation is to
absorb pipelines into the base and free up budget for harder caps next round.

---

## §8. capability.jsonl row schema (stage-aware)

```json
{
  "iter": 7,
  "stage": 2,
  "method": "opd",
  "slug": "stage-2-opd-polish",
  "ts": "2026-06-08T...",
  "status": "kept",
  "family": "stage-transition",
  "hypothesis": "OPD on top of stage-1 SFT lifts process from 0.55 to ~0.75 via teacher gap",
  "rubric_version": "v1",

  "base_adapter": "<cap>-stage-1-sft-bootstrap",
  "output_adapter": "<cap>-stage-2-opd-polish",
  "stage_transition_rationale": "METHODS.md Rule E fired: format=0.78 stable, process headroom=0.32 of total, teacher available with composite=0.94.",

  "composite": 0.852,
  "composite_delta": 0.104,
  "sub_scores": {...},
  "verdict": "positive",
  "sigma_warning": null,

  "method_specific": {
    "opd": {
      "effective_steps": 32,
      "teacher_calls_made": 64,
      "env_ce_delta": -0.45,
      "skip_rate": 0.21
    }
  },

  "sibling_regression_check": {
    "max_delta": -0.008,
    "sigma": 0.011,
    "passed": true
  },
  "prior_stage_preservation_check": {
    "prev_stage_composite_now": 0.752,
    "prev_stage_composite_orig": 0.748,
    "delta": 0.004,
    "passed": true
  },

  "kiln_commit": "<sha>",
  "train_receipt": "/workspace/adapters/.../train_receipt.json",
  "adapter_manifest": "/workspace/adapters/.../adapter_manifest.json",
  "notes": "..."
}
```

Old (round-2) rows without `stage`/`method` are treated as `stage: 1` /
`method: <inferred from paradigm dir>` by `lib/stage_manifest.py`.
Migration is non-destructive.

---

## §9. Adapter naming convention

`<cap>-stage-<N>-<slug>` where slug is the iter's family-descriptor.
Examples:
- `pi-faithful-completion-stage-1-sft-bootstrap`
- `pi-faithful-completion-stage-2-opd-polish`
- `pi-faithful-completion-stage-3-grpo-final`

This makes `kiln adapter verify`, `kiln eval-adapter`, and integration
references unambiguous about which stage of which cap a given adapter is.

For warm-best variants within a stage, append `-wbN`:
`pi-faithful-completion-stage-3-grpo-final-wb1`.

---

## §10. The minimal pipeline (single stage)

Most round-2 caps will become single-stage pipelines after migration. That's
fine. A single-stage pipeline.md still gets written:

```yaml
---
schema_version: 1
capability: pi-code-search
status: shipped
base_round: round-2
baseline_composite: 0.5432
final_composite: 0.6004
final_adapter: pi-code-search-stage-1-h5-replay
stages:
  - {n: 1, method: agentic-grpo, slug: stage-1-h5-replay, composite_after: 0.6004}
---

# pi-code-search pipeline

## Baseline: 0.5432
- Headroom: 0.4568, concentrated in process (precision_of_read) and outcome.

## Stage 1: agentic-GRPO h5-replay (composite 0.5432 → 0.6004)
- Method: agentic-GRPO
- Adapter: pi-code-search-stage-1-h5-replay
- Base: none
- Recipe: round-1 iter-5 h5 replay recipe (see archive/closeout.md for the chain).
- Why single-stage: round-2 evidence saw no clear next stage. OPD considered
  but no teacher noticeably stronger on this rubric. GRPO continuation
  saturated.
- Evidence: 5-eval mean +0.024 ± 0.030, peak +0.057.
```

Multi-stage is the *option*, not the default. The default is the simplest
chain that beats baseline by a robust margin.

---

## §11. Pipelines vs hypotheses

`capability.md::hypotheses` records design intent for the cap as a whole.
`pipeline.md::stages` records the chain that won. Each stage's
`stage_transition_rationale` links back to the hypothesis it tested.

Hypotheses can be open-ended ("OPD will lift process sub-scores"); stage
rationales must be falsifiable and tied to a measured input
("METHODS.md Rule E fires because format=0.78, process headroom=0.32,
teacher composite=0.94").

---

## §12. Quick reference

| You want to... | Run |
| --- | --- |
| Build a fresh pipeline | `./run_stage.sh <method> stage-1-<slug>` then iterate |
| Add a stage to existing pipeline | `./run_stage.sh <method> stage-N-<slug>` |
| Re-run shipped pipeline on new base | `./run_pipeline.sh` |
| Validate shipped pipeline still holds | `./run_pipeline.sh --validate-only` |
| Eval an arbitrary adapter against the cap | `./capability.oracle.sh <name>` |
| Check sibling regression for current best | `integration/cross-cap-coherence/capability.oracle.sh <name>` |
| See decision tree recommendation | `python3 lib/method_router.py --eval-summary <last-eval.json>` |
| Validate `stages/` ↔ `pipeline.md` ↔ `capability.jsonl` | `python3 lib/stage_manifest.py --validate <cap-dir>` |
