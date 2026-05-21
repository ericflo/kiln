---
name: capability-creator
description: Autonomous loop for **uplifting** a target capability on Qwen3.5-4B through any combination of SFT, OPD, GRPO, and agentic-GRPO stages chained into a multi-method pipeline. Use when asked to "train capability X", "elicit X", "build a pipeline for X", "distill X into the model", or "ship a kept adapter for X". The skill picks methodology per stage by routing through `capabilities/METHODS.md`; chains stages per `capabilities/PIPELINE.md`; eventually feeds the cluster-distillation flywheel in `capabilities/DISTILLATION.md`. Each iteration is one independent (hypothesis, recipe) attempt with a blind eval and a verdict gate before the next can start.
---

# capability-creator

Autonomous loop for **shipping** a capability adapter pipeline on Qwen3.5-4B
served by kiln. The skill replaces the four old methodology-specific skills
(`sft-`, `grpo-`, `opd-`, `agentic-grpo-capability-creator`); their per-method
lore lives in `resources/<method>-mode.md`.

The single rule:

> **Pick the method that best closes the gap between current composite and
> headroom, given evidence. Methodology choice is a consequence of evidence,
> not an a-priori commitment. Capability is the goal; methodology is data.**

## Skill inventory

| File | Purpose |
|------|---------|
| `SKILL.md` | This document. Authoritative procedure. |
| `resources/sft-mode.md` | SFT-specific failure modes, anchor pattern, answer-form discipline. |
| `resources/opd-mode.md` | OPD-specific high-baseline failure (cap #5), skip-rate watch, teacher hosting. |
| `resources/grpo-mode.md` | GRPO-specific reward function failure modes, group statistics, all-zeros mode. |
| `resources/agentic-grpo-mode.md` | Agentic-specific ECHO defaults, pi-smoke, multi-turn budget. |
| `resources/kiln-cli-reference.md` | The 11 kiln CLIs the layout depends on. |
| `templates/scaffold.sh` | Create `capabilities/caps/<slug>/` with the full LAYOUT.md skeleton. |
| `templates/run_stage_sft.sh` | Reference stage runner for SFT. |
| `templates/run_stage_opd.sh` | Reference stage runner for OPD. |
| `templates/run_stage_grpo.sh` | Reference stage runner for GRPO. |
| `templates/run_stage_agentic_grpo.sh` | Reference stage runner for agentic-GRPO. |
| `templates/run_pipeline.sh` | Re-runs all stages from pipeline.md. |
| `templates/promote_iter_to_stage.sh` | Promote a kept iter row to a stage record. |
| `templates/headroom.py` | Per-sub-score headroom analysis (copy of `capabilities/lib/headroom.py`). |
| `templates/rubric_sanity.py` | Default rubric calibration sanity check. |
| `templates/pi_smoke.sh` | Pi binary + session-JSONL smoke test (agentic-GRPO only). |
| `install.sh` | Symlink the skill into `.claude/skills/`. |

The three load-bearing docs at `capabilities/` (top-level) are required reading:

- [`capabilities/METHODS.md`](../../../capabilities/METHODS.md) — when to choose each method.
- [`capabilities/PIPELINE.md`](../../../capabilities/PIPELINE.md) — multi-stage operating manual.
- [`capabilities/DISTILLATION.md`](../../../capabilities/DISTILLATION.md) — cluster → new base flywheel.

---

## §0. Mental model

You are an experimentalist whose budget is mostly GPU memory and time. The lab has:

- a fixed **base student** (kiln-served Qwen3.5-4B on `:8420`)
- a **target capability** described in plain English and instrumented with a rubric
- four available **training methods** (SFT, OPD, GRPO, agentic-GRPO) each with
  its own strengths and failure modes
- optionally a **live teacher** for OPD (a stronger model in the same family at 14-32B)
- a **blind oracle** — a command that returns one composite scalar plus its
  breakdown by sub-score

You produce a **pipeline**: an ordered sequence of stages, each one a trained
adapter using one method, with `base_adapter` = previous stage's
`output_adapter`. The final stage's adapter is the cap's shipped artifact.

You never stack within method without re-evaluating between. You always
falsify the stage hypothesis before declaring a kept stage. Every stage gets
a row in `capability.jsonl`; kept stages also get a `stages/<N>.json` record
and a paragraph in `pipeline.md`.

### Where each method shines (TL;DR; full version in resources/)

| Method | Use when | Avoid when |
|---|---|---|
| **SFT** | Baseline outcome < 0.3 OR format headroom > 30% of total. Bootstrap from a small (8-256) curated dataset. | Stacking SFT on SFT (overfits surface form). Tasks where the capability isn't ground-truthable. |
| **OPD** | Baseline ∈ [0.4, 0.8], teacher available, process/format headroom, student produces in-shape attempts. | Baseline > 0.80 with variable rollout quality (cap #5 catastrophic failure). |
| **GRPO** | Baseline ∈ [0.6, 0.9], verifier exists, reward variance > 0.05. | Saturated reward (use --no-policy-loss / harder eval). Length / mode / entropy collapse if rubric is loose. |
| **Agentic-GRPO** | Task is multi-turn tool-calling. Always use with ECHO. | Single-turn tasks (use plain GRPO). |

### The eval is the spec — and you wrote it

**The optimisation target is the rubric, not the capability.** If the rubric
doesn't faithfully measure the capability, training finds the cheapest path
through the rubric and produces a model that satisfies the contract you wrote
rather than the one you intended. Goodhart's law is the centre of this skill,
not a footnote.

You cannot fix a flawed rubric with hyperparameters. A perfect epoch / rank /
recipe sweep against a flawed rubric produces a perfectly polished bad model.

**Eval design is the highest-leverage activity in the session. Spend more
time on it than on any single iteration.**

Round-1 hit "rubric too lax" three times. Round-3 baseline composite > 0.95
is a hard fail of [`capabilities/METHODS.md`](../../../capabilities/METHODS.md) Rule B; pipe is interrupted and rubric
must be hardened before any method runs.

### The headroom principle

Composite is a weighted sum. Each sub-score `s_i` with weight `w_i` and
baseline `b_i` contributes at most `w_i × (1 − b_i)` to a future composite
uplift. **Headroom = `Σ w_i × (1 − b_i)`.** Most of it usually lives in one
or two sub-scores; the rest are saturated.

Before writing a hypothesis, look at headroom and pick the sub-score you're
targeting. If headroom < 0.05, the rubric is too saturated for any training
to be interesting; harden the rubric or build `hard_eval.tasks.jsonl`.

`templates/headroom.py` (or `capabilities/lib/headroom.py`) does this analysis.

---

## §1. Information firewall (non-negotiable)

The eval is **blind**. The oracle returns one composite score plus
per-sub-score breakdown. You see those numbers and nothing else.

You **MUST NOT**:

- read any file under `datasets/eval.tasks.jsonl` or
  `datasets/hard_eval.tasks.jsonl` (the eval task pools);
- read any file under `adapters/.eval/suites/**`, `adapters/.eval/judgments/**`,
  or `adapters/.eval/datasets/**` for the dataset backing the active oracle;
- inspect per-example outputs after a `kiln eval-adapter` run beyond
  `summary.{mean_composite,sub_scores_mean,...}`;
- copy the eval's prompt template into your training data;
- design a dataset by **inverting** the oracle (probe → infer → train-to-match);
- ask the user *"what does the eval check?"* to memorise its surface.

You **MAY**:

- read the score the oracle returns;
- read `n` the oracle returns;
- ask the user once at intake for a **plain-English description** (1-3 sentences);
- ask for **categorical hints** if volunteered ("the eval is multi-turn") —
  do not press for surface detail.

If you catch yourself reading an eval source file, stop, revert that step,
and write `firewall_breach` in the next log entry's `notes`. If the breach
is severe (you saw prompt or rubric verbatim), **the session is dead**;
start over with slug suffix `-postbreach` and acknowledge the contamination
in `capability.md`.

### Sub-agents inherit the firewall

If you spawn a sub-agent, include the firewall instruction in its prompt.
A sub-agent that helpfully surfaces eval contents in a summary breaks the
experiment as completely as reading the files yourself.

---

## §2. Capability lifecycle

The full procedure, end-to-end. Four phases.

### Phase 0 — Scaffold + contract + calibration + baseline

```bash
# 0a. Create the cap directory
bash $SKILL/templates/scaffold.sh <slug>
cd capabilities/caps/<slug>/

# 0b. Write capability.md (the contract)
# Required sections in order:
#   # Capability: <name>
#   ## Description
#   ## Base model
#   ## Rollout source
#   ## Rubric (v<N>)
#   ## Adversarial design (§0)
#   ## Baseline + Headroom
#   ## Hypotheses
#   ## Standard workflow
#   ## Kiln features used
#
# Adversarial design (§0) MUST be filled BEFORE rubric.py. Name ≥3 cheats
# that would score 1.0 without doing the capability; design mitigations into
# the rubric.

# 0c. Write rubric.py
# Pure function `score_one(rollout) -> dict[str, float]` with composite.
# Must expose RUBRIC_VERSION constant.
# Importable on CPU-only dev box without network.

# 0d. Populate calibration (MANDATORY)
$EDITOR calibration/good.jsonl calibration/bad.jsonl
# ≥5 good rollouts, ≥5 bad rollouts. One bad rollout per §0 cheat.

# 0e. Sanity-check the rubric (MANDATORY)
python3 rubric_sanity.py
# Margin > 0.2 between good/bad means MUST pass before any GPU work.

# 0f. Build the task corpus
python3 build_corpus.py
# Writes datasets/{train,eval,hard_eval}.tasks.jsonl and method-specific
# training data lazily.

# 0g. Baseline eval (no adapter)
./capability.oracle.sh
# This is iter 0; record it in capability.jsonl with status=kept, stage=0, method=none.
```

### Phase 1 — Headroom analysis + method choice for stage 1

```bash
# 1a. Analyze headroom
python3 ../../lib/headroom.py --eval-summary /tmp/<cap>-eval-base.json --print
# Look for: total headroom, dominant sub-score, share of total.

# 1b. Ask the decision tree for the recommended first method
python3 ../../lib/method_router.py \
  --eval-summary /tmp/<cap>-eval-base.json \
  [--teacher-available] \
  [--multi-turn] \
  [--has-verifier] \
  [--reward-variance <0.0X>] \
  --print
# Recommended method + rule that fired + rationale.

# 1c. Read the relevant resources/<method>-mode.md
# This is the irreducible per-method lore. Read before committing to the method.

# 1d. State stage 1 hypothesis
# Edit capability.md::Hypotheses to add H1 — one sentence, falsifiable.
```

### Phase 2 — Stage loop

For each stage (starting with N=1), run the appropriate stage runner:

```bash
./run_stage.sh <method> stage-<N>-<descriptor> [--base-adapter <prev-stage-output>]
```

`run_stage.sh` is per-cap; reference implementations live in
`templates/run_stage_<method>.sh`. The runner enforces the pre-stage gates
in [`capabilities/PIPELINE.md`](../../../capabilities/PIPELINE.md) §4.1
mechanically. Within a stage, multiple iters may run (variants of the recipe);
the **kept** iter (meeting [`PIPELINE.md`](../../../capabilities/PIPELINE.md)
§4.3 promotion criteria) becomes the stage.

After each iter:

```bash
# Inspect the new row
tail -1 capability.jsonl | python3 -m json.tool

# Cross-cap regression check (mandatory between stages)
cd ../../integration/cross-cap-coherence/
./capability.oracle.sh <new-stage-adapter>
cd ../../caps/<cap>/

# If kept, promote
bash $SKILL/templates/promote_iter_to_stage.sh <iter-row-slug>
```

After each stage's promotion:

```bash
# Ask the decision tree if a next stage is justified
python3 ../../lib/method_router.py --eval-summary /tmp/<cap>-eval-stage-<N>.json --print
```

If the tree returns `stop`, end the pipeline. If it returns a method AND
total headroom > 0.05, proceed to stage N+1 with that method.

### Phase 3 — Closeout

Pipeline is **shipped** when ALL hold (see
[`capabilities/PIPELINE.md`](../../../capabilities/PIPELINE.md) §11
reproducibility checklist for the full list):

- 3-seed mean Δ on cap's eval > 2σ above baseline
- Every kept stage has `train_receipt.json` and `adapter_manifest.json`
- `kiln adapter verify <final-adapter>` passes loadable + behavioral
- `calibration/{good,bad}.jsonl` rubric sanity still passes
- No `sigma_warning` in final stage's `eval_summary.json`
- `pipeline.md::status: shipped` matches `stages/` and `capability.jsonl`
- `lib/stage_manifest.py --validate <cap-dir>` exits 0
- `integration/cross-cap-coherence/` max sibling delta ≥ −0.02

Set `pipeline.md::status: shipped`. Append a final iter row with
`status: "closeout"`. Update `capability.md` with a "## Closeout" section
summarising winning recipe.

### Phase 4 — Cluster (only when ≥5 multi-stage caps ship)

See [`capabilities/DISTILLATION.md`](../../../capabilities/DISTILLATION.md).
The agent does not do distillation per-cap; once enough pipelines have shipped
the cluster step runs across all of them. Mentioned here for orientation only.

---

## §3. The hypothesis discipline

**Hypothesis-before-data. Falsification-plan-before-result. Verdict-before-next-iter.**

Every iter, BEFORE training:

1. **State the hypothesis in one sentence.** "OPD against the 27B teacher on
   top of stage-1 SFT lifts process_faithfulness from 0.55 to ~0.75 by closing
   the distribution gap on edit-context cases."

2. **State the falsification plan in one sentence.** "If composite_delta
   from stage 1 is within σ, this is null. If process_faithfulness moves but
   composite doesn't, the rubric weighting is the bottleneck."

3. **State the expected magnitude.** Not "should improve" — "+0.05 to +0.10
   on composite; +0.15 to +0.25 on process_faithfulness."

After eval, BEFORE moving on:

4. **Record the verdict.** `positive | null | negative | inconclusive`.
5. **Record the kept / not-kept call.** Promotion criteria
   ([`PIPELINE.md`](../../../capabilities/PIPELINE.md) §4.3) decide; the agent
   does not override.

Hypothesis files live in `hypotheses/<slug>.md` (template in
`templates/hypothesis.md.tmpl`). The 3-line summary lives in
`capability.jsonl::hypothesis` and `capability.md::Hypotheses`.

**Why this matters:** without a falsification plan, every iter looks positive
("composite moved a little; that's progress"). With one, you know exactly when
to stop, when to try a variant, and when to switch method.

---

## §4. Anti-laziness gates (consolidated)

These are the discipline guards. Honor them mechanically; the loop fails
without them.

| Gate | Mechanism | What it catches |
|---|---|---|
| **rubric_sanity.py** before every iter | `run_stage.sh` runs it first | "rubric too lax" / broken calibration |
| **`--dry-run` before GPU** | `cuda_*_ablation --dry-run` (kiln #9) | data schema, mask, zero-action, base-adapter mismatch |
| **3-seed eval mandatory** | `kiln eval-adapter --seeds 3` default | single-seed luck (round-1 +11pp → +4.2pp) |
| **Adapter verify after train** | `kiln adapter verify` (kiln #4) | adapter not loaded, behavioral no-op |
| **Sibling check between stages** | `integration/cross-cap-coherence/` mandatory | stacked-adapter skill clobber |
| **Prior-stage preservation check** | new adapter eval against prev stage eval set | new stage erased old domain |
| **Hypothesis-before-data** | `hypotheses/<slug>.md` exists | iter-without-plan drift |
| **Headroom before method choice** | `lib/headroom.py` + `lib/method_router.py` | premature method commitment |
| **Receipt drives the row, not log scraping** | `train_receipt.json` (kiln #8) | misread metrics, missing diagnostics |

If a gate fails, the iter is invalid. Do not patch around it; fix the gate.

---

## §5. All-zeros / no-signal failure modes

Common pre-promotion failures, sorted by frequency:

1. **`composite_delta ≈ 0` after a real training step.** Run the diagnostic
   ladder in [`capabilities/NEXT_ROUND.md`](../../../capabilities/NEXT_ROUND.md)
   §"Diagnostics ladder for 'iter didn't move'".
2. **All-zero rewards (GRPO).** The base produces rollouts that all score 0.
   Either the rubric is broken (too strict) OR the model genuinely can't try
   the task → SFT bootstrap stage first (METHODS.md Rule C).
3. **ECHO didn't fire (agentic-GRPO).** `train_receipt.json::echo_metrics::
   env_ce_steps_observed == 0` means the env_mask was empty. Verify the
   trajectory has tool/observation segments; check warning-prefix masking.
4. **Byte-identical adapters (long-context).** PR #27 fix added the diagnostic;
   if `adapter_smoke_test::logit_delta_mean ≈ 0`, the adapter trained but
   doesn't change inference. Check rank, alpha, lr.
5. **OPD catastrophic regression at high baseline (cap #5).** Baseline > 0.80
   with variable rollout quality → reverse-KL amplifies confidence in malformed
   samples. Switch to SFT on teacher rollouts (METHODS.md §4.3).
6. **Reward saturation (GRPO).** `reward_saturation_warning` fires; mean > 0.92,
   variance < 0.03. Switch to `--no-policy-loss` (ECHO-only) or hard_eval pool.

The trainer's `failure_reason` field on the receipt (kiln #24) names the
class. The above mapping translates failure_reason → next action.

---

## §6. Stage transitions in practice

Reference: [`capabilities/PIPELINE.md`](../../../capabilities/PIPELINE.md) §4.

The agent's job at a transition:

1. **Read `lib/method_router.py` recommendation** for the post-stage-N eval.
2. **Read the relevant `resources/<method>-mode.md`** for the proposed next method.
3. **State the next stage hypothesis** in `capability.md::Hypotheses` and
   `pipeline.md::stages[N+1].stage_transition_rationale`.
4. **Run the previous stage's adapter through `integration/cross-cap-coherence/`**.
   If it regresses siblings, fix that *before* extending.
5. **Run the next stage.** `run_stage.sh <next-method> stage-<N+1>-<slug>
   --base-adapter <prev-stage-output>`.
6. **Validate promotion criteria** (PIPELINE.md §4.3). The runner enforces
   them mechanically.

Most pipelines are 1-3 stages. >3 stages is rare and the agent should pause
to ask whether the rubric is rewarding the right thing.

---

## §7. The kiln CLI surface (reference)

The 11 CLIs the layout depends on, all part of the round-2 backlog (#1-40 complete).
Full reference in `resources/kiln-cli-reference.md`.

| Command | Purpose | Issue |
|---|---|---|
| `kiln serve --eval-mode` | Deterministic serving during eval | 15 |
| `kiln adapter verify <name>` | Prove adapter is loadable + behavioral | 4 |
| `kiln adapter restore <manifest>` | Re-materialize from manifest | 36 |
| `kiln trajectory inspect <jsonl>` | Mask + token-count diagnostic | 10 |
| `kiln eval-adapter --adapter ... --seeds N` | Multi-seed paired eval | 33 |
| `kiln rollout --adapter ... --tasks ...` | Direct HTTP rollout | 34 |
| `cuda_grpo_ablation --dry-run` | Pre-GPU validation | 9 |
| `cuda_grpo_ablation --filter-var-min` | Strong-signal filter | 22 |
| `cuda_*_ablation --install-adapter-dir/name` | Atomic install | 5 |
| `cuda_*_ablation --adapter-smoke-test` | Post-train sanity check | 19 |
| `cuda_opd_remote ...` | OPD trainer | 37 |
| `cuda_sft_file ...` | SFT trainer | pre-existing |

Do NOT re-implement any of these. Missing features go in
`capabilities/KILN_IMPROVEMENT_ISSUES.md` with a stop-gap in the cap until
they land.

---

## §8. Money-burning anti-patterns

These cost real money in prior rounds. Do not repeat.

1. **SSH polling loops against RunPod.** `until ssh ... kill -0` /
   `while ssh ... grep ...; sleep 5` deadlocks when sshd wedges. Use
   `python3 $RP wait-file <pod_id> /tmp/done --timeout 1800` instead. Two
   incidents 2026-04-20: $13.76 + $99.76.
2. **No timeout / failure cleanup on long-running GPU work.** Every pod
   acquisition needs a trap-on-failure cleanup; otherwise an orphaned pod
   bills $0.49/hr indefinitely.
3. **Direct-launching pods when the pool is available.** Always
   `ce kiln-pod-acquire` first; direct `runpod_api.py launch` is fallback.
4. **Distillation before the cluster threshold.** Distilling < 5 multi-stage
   caps wastes GPU and produces a worse base than what you started with.
5. **Single-seed wins.** Always 3-seed before promotion.
6. **Stacking adapters without re-eval between.** A 5-iter chain with one
   eval at the end can't identify which iter introduced regression.

Full lore: `resources/kiln-cli-reference.md` and the kiln skill in
`/data/knowledge/skills/kiln/SKILL.md` ("MONEY-BURNING ANTI-PATTERNS").

---

## §9. One-screen quickstart

```bash
# 0. Intake (one-shot)
SLUG=<cap-slug>
bash $SKILL/templates/scaffold.sh $SLUG
cd capabilities/caps/$SLUG/

# edit capability.md, rubric.py, rubric_sanity.py
# populate calibration/{good,bad}.jsonl

# 1. Sanity + baseline
python3 rubric_sanity.py        # gate (mandatory)
./capability.oracle.sh          # baseline (iter 0)

# 2. Choose stage-1 method from the decision tree
python3 ../../lib/headroom.py --eval-summary /tmp/$SLUG-eval-base.json --print
python3 ../../lib/method_router.py --eval-summary /tmp/$SLUG-eval-base.json --print
# Read the relevant resources/<method>-mode.md

# 3. Edit capability.md::Hypotheses to add H1

# 4. Run stage 1
./run_stage.sh <method> stage-1-h1-<descriptor>

# 5. Inspect the row + sibling check
tail -1 capability.jsonl | python3 -m json.tool
cd ../../integration/cross-cap-coherence/
./capability.oracle.sh <cap>-stage-1-h1-<descriptor>
cd ../../caps/$SLUG/

# 6. Stage 2? Ask the router.
python3 ../../lib/method_router.py --eval-summary /tmp/$SLUG-eval-stage-1.json --print
# If recommended method differs AND headroom > 0.05:
./run_stage.sh <next-method> stage-2-h2-<descriptor> --base-adapter $SLUG-stage-1-h1-<descriptor>

# 7. Repeat until router returns stop OR closeout criteria met.
# Final pipeline reruns end-to-end with:
./run_pipeline.sh
```

---

## §10. Resuming a session

A fresh agent should be able to resume from the cap dir alone. Read order:

1. `pipeline.md` — what stages are shipped, what's in-flight
2. `capability.md` — the contract, hypotheses log
3. Most recent `capability.jsonl` rows — current state
4. `stages/` files — kept stage records
5. [`capabilities/METHODS.md`](../../../capabilities/METHODS.md) — re-route the next stage

Run `python3 ../../lib/stage_manifest.py --validate .` to confirm
pipeline.md ↔ stages/ ↔ capability.jsonl are consistent. If not, fix that
first (a divergence usually means an iter was promoted incorrectly).

If `pipeline.md::base_sha256` ≠ the current base sha (after a Phase G
distillation), run `lib/stage_manifest.py check-base-drift` then
`./run_pipeline.sh --validate-only` to revalidate. If the pipeline doesn't
hold on the new base, re-train with `./run_pipeline.sh` (may shorten as
prior stages get absorbed).

---

## §11. When NOT to use this skill

This skill is for **uplifting a measurable capability** through training.
Don't use it for:

- **Building a new kiln feature.** That's kiln source modification work —
  see `crates/` and the kiln backlog.
- **Running a single-shot ablation without a falsifiable hypothesis.** The
  loop overhead is too much; use a notebook.
- **Evaluating an existing adapter against an existing rubric.** Use
  `capability.oracle.sh` directly.
- **Producing a curriculum / instruction-tuning dataset.** This skill
  consumes such datasets; building them is a different task.
- **Working on a capability without a blind eval.** No eval → no signal →
  no loop. Build the eval first.

---

## §12. Skill boundaries

- The skill **does not modify kiln source.** If a kiln feature is missing,
  file an issue in `capabilities/KILN_IMPROVEMENT_ISSUES.md` and use a
  stop-gap in the cap.
- The skill **does not run distillation.** Distillation is a Phase G cross-
  capability step run after ≥5 multi-stage caps ship. See
  [`capabilities/DISTILLATION.md`](../../../capabilities/DISTILLATION.md).
- The skill **does not edit other caps' rubrics or datasets.** Sibling
  regression issues are surfaced via `integration/cross-cap-coherence/`;
  the resolution might be lowering the current cap's rank/lr or stopping
  the pipeline. Don't reach into a sibling cap to "fix" it.
- The skill **does not change the base model.** Base shifts happen at
  Phase G distillation only.

---

## §13. References

- [`capabilities/README.md`](../../../capabilities/README.md) — top-level entry
- [`capabilities/LAYOUT.md`](../../../capabilities/LAYOUT.md) — uniform cap layout
- [`capabilities/METHODS.md`](../../../capabilities/METHODS.md) — decision tree (read every stage)
- [`capabilities/PIPELINE.md`](../../../capabilities/PIPELINE.md) — multi-stage operating manual
- [`capabilities/DISTILLATION.md`](../../../capabilities/DISTILLATION.md) — cluster flywheel (Phase G)
- [`capabilities/NEXT_ROUND.md`](../../../capabilities/NEXT_ROUND.md) — practical operating guide
- [`capabilities/CONSOLIDATED_REPORT.md`](../../../capabilities/CONSOLIDATED_REPORT.md) — round-1 lessons
- [`capabilities/KILN_IMPROVEMENT_ISSUES.md`](../../../capabilities/KILN_IMPROVEMENT_ISSUES.md) — kiln backlog (all 40 complete)
- [`capabilities/lib/agentic-grpo-notes.md`](../../../capabilities/lib/agentic-grpo-notes.md) — ECHO + pi specifics
- `resources/<method>-mode.md` — per-method irreducible lore (this skill)
- `resources/kiln-cli-reference.md` — kiln CLI cheat sheet (this skill)
