# `METHODS.md` — When to use which training methodology

This is the **single source of truth** for choosing a training method at any
point in a capability's pipeline. It replaces the "when to use me" sections
that used to live separately in the four methodology skills.

The single rule, stated up front:

> **Pick the method that best closes the gap between current composite and
> headroom, given evidence (calibration, baseline distribution, teacher
> availability, reward variance, task shape).** Methodology choice is a
> consequence of evidence, not an a-priori commitment.

Reading order in this doc:

1. **§1 — Inputs to the decision.** What you measure before picking a method.
2. **§2 — The decision tree.** The routed choice.
3. **§3 — Per-method one-pagers.** When the tree picks a method, what you need to know about it.
4. **§4 — Stage transitions.** How to chain methods across stages.
5. **§5 — Stopping conditions.** When to stop iterating *within* a method.
6. **§6 — Anti-patterns.** Mistakes that round 1 and round 2 made; do not repeat.

The companion docs are
[`PIPELINE.md`](PIPELINE.md) (multi-stage operating manual) and
[`DISTILLATION.md`](DISTILLATION.md) (cluster → new base flywheel).

---

## §1. Inputs to the decision

Before picking a method, you must have measured these. Most of them come
from running `./capability.oracle.sh` (no adapter) once and inspecting the
result. The rest are properties of the capability and your harness.

| Input | How to get it | Why it matters |
| --- | --- | --- |
| **Baseline composite C₀** | `./capability.oracle.sh` | Anchors every other input. < 0.3 = task barely understood. > 0.95 = rubric too lax. |
| **Headroom per sub-score** | `lib/headroom.py` on the eval summary | `H_i = w_i × (1 - s_i)`. Where the lift can come from. |
| **Dominant headroom type** | inspect H — outcome / format / process | Format → SFT often cheapest. Process → OPD/GRPO. Outcome → may not be liftable without new knowledge. |
| **Teacher availability** | does a ≥30%-stronger model in the same family exist? | OPD is only available if yes. |
| **Verifier presence** | can you write a programmatic `score_one()` over a complete response in `[0,1]`? | GRPO is only available if yes. |
| **Reward variance** | sample 20 baseline rollouts on `datasets/train.tasks.jsonl`, compute group variance | < 0.03 → GRPO has no signal. ≥ 0.05 → strong signal filter applies. |
| **Task is multi-turn tool-calling?** | is `rollout.py` driving pi or another agent loop? | If yes, agentic-GRPO with ECHO is mandatory; everything else is a sub-step. |
| **Baseline distribution shape** | inspect 5 base responses on `datasets/eval.tasks.jsonl` | Sanity-check that C₀ matches reality — saturated *or* over-strict rubrics distort it. |
| **Calibration separation** | `python rubric_sanity.py` | margin > 0.2 between good and bad fixtures, else the rubric is broken regardless of method. |

If you cannot answer any of the first six, **do not pick a method yet**.
The cap is not ready for stage 1.

---

## §2. The decision tree

Apply rules in order. The first rule that fires wins. Document the rule
that fired in `pipeline.md` stage rationale.

```
RULE A — Multi-turn tool-calling tasks
  IF task is multi-turn tool-calling:
    → agentic-GRPO with ECHO on (always)
    → Sub-method choice still applies — you may still SFT-bootstrap or
      OPD-polish before the agentic-GRPO stage. But every agentic stage
      runs cuda_grpo_ablation with the ECHO env-mask layer.

RULE B — Broken baseline (rubric audit before method choice)
  IF baseline composite > 0.95 on standard eval:
    → STOP. Rubric is too lax. Tighten sub-scores, harden eval, re-baseline.
  IF baseline composite < 0.3 on a capability the model should partially have:
    → STOP. Inspect 3-5 base responses. Rubric may be over-strict.
      (Round-1 OPD #5 hit this — bare-diff rubric rejected fenced output.)

RULE C — Cold-start: model doesn't try the task
  IF baseline outcome sub-score < 0.3 AND format sub-score < 0.5:
    → SFT bootstrap (8-64 curated examples, 1-3 epochs, low rank)
    → Reason: GRPO/OPD need a baseline that at least attempts the task.
      Curated examples are the cheapest way to install the format prior.

RULE D — Format-headroom dominated
  IF format headroom > 0.3 of total headroom:
    → SFT bootstrap first
    → Reason: SFT on curated examples is the cheapest format fix.
      Format must stabilize before GRPO policy gradient is safe (otherwise
      gradient flows on malformed responses and accelerates the failure mode).

RULE E — Distribution gap to a stronger teacher
  IF baseline composite ∈ [0.4, 0.8]
     AND teacher available (≥30% stronger composite on this rubric)
     AND headroom concentrated in process/format sub-scores
     AND student already produces in-shape attempts (not pure junk):
    → OPD
    → Reason: distribution gap to teacher is the cheapest gradient signal.
    → Caveat: see opd-mode.md §5 high-baseline failure. If baseline > 0.80
      AND student rollouts are variable quality, OPD will regress.
      Use SFT on teacher rollouts instead (or accept ceiling).

RULE F — Verifiable signal with reward variance
  IF verifier exists
     AND baseline composite ∈ [0.6, 0.9]
     AND reward variance > 0.05
     AND task is single-turn OR the agentic-GRPO loop is already set up:
    → GRPO (with --filter-var-min 0.05)
    → Reason: rollouts produce signal; policy gradient will move it.
    → Caveat: see grpo-mode.md §6 on reward function failure modes.
      Length drift, mode collapse, entropy collapse are all reward-function
      bugs, not method bugs.

RULE G — Saturated reward with residual hard-tail
  IF baseline composite > 0.85
     AND reward variance < 0.03 on standard eval
     AND headroom exists on hard_eval.tasks.jsonl:
    → Either: (a) switch eval to hard_eval, then GRPO --no-policy-loss (ECHO-only)
              (b) accept ceiling; ship base for this capability
    → Reason: policy gradient on saturated reward is the harm vector. See
      CONSOLIDATED_REPORT §"pi-diff-patch-apply" for the case study (every
      trained iter regressed from base 0.94).

RULE H — Closeout
  IF composite_delta within σ for 2 consecutive iters
     AND no sub-score has > 0.05 headroom:
    → STOP for this stage; consider promoting to next stage in the pipeline.
    → If pipeline is already 3+ stages and no headroom remains, declare closeout.
```

The tree is deliberately conservative: it does **not** try to predict which
of OPD/GRPO will win when both are eligible. When both apply (which happens
in the middle of the C₀ range), `pipeline.md` records the agent's choice +
rationale, and a follow-up stage with the other method becomes the
falsification step.

---

## §3. Per-method one-pagers

### §3.1 SFT

**Trainer:** `cuda_sft_file`
**Use when:** Rule C, D, or "bootstrap from teacher rollouts" rescue path
(§4 OPD→SFT transition).
**Data:** 8–256 curated `{prompt, completion}` pairs in
`datasets/sft.train.jsonl`. Diversity beats volume. Verbal/structural framings
often beat surface-form drill (see CONSOLIDATED_REPORT `math-broad` case).
**Defaults:** rank=4, alpha=8, lr=1e-4, 1 epoch, dataset cap ≤128 per iter.
**Receipt fields to watch:** `epochs`, `dataset_size`, `loss_curve_final`.
SFT loss curves are informative (unlike OPD/GRPO).
**Anchor regression:** ALWAYS run `capability.anchor.sh` after SFT — SFT
silently clobbers non-target capabilities through style drift. See
resources/sft-mode.md for the anchor pattern.
**Common failure mode:** "rank too high → overfits surface form, doesn't
generalize." Start rank=4 and only escalate on evidence.

### §3.2 OPD

**Trainer:** `cuda_opd_remote`
**Use when:** Rule E, or as polish stage on top of SFT.
**Data:** Just prompts in `datasets/opd.prompts.jsonl`. The student samples
its own completions; the teacher grades token-by-token via reverse-KL.
**Teacher:** Live LM server (vLLM on `:8002` by default). Q4 of a stronger
model in the same family at 14B–32B. Check `teacher_url` health before training.
**Defaults:** rank=16, alpha=32, lr=1e-4, 4–6 epochs, samples_per_prompt=1–2.
**Receipt fields to watch:** `effective_steps`, `teacher_calls_made`, skip rate
(should be < 50% for healthy training). The round-1 `code-symbol-extraction`
bug showed 97% skip rate; the fix is now in the receipt.
**Loss is deceptive:** OPD loss is spiky by nature. Trust the blind eval,
never the loss curve. (resources/opd-mode.md §9)
**High-baseline failure mode (cap #5):** baseline > 0.80 with variable
rollout quality → OPD regresses, including catastrophically. Recovery path
is to sample teacher rollouts directly and SFT on them. See
resources/opd-mode.md §0.

### §3.3 GRPO

**Trainer:** `cuda_grpo_ablation`
**Use when:** Rule F. Single-turn tasks with a programmatic verifier.
**Data:** Tasks in `datasets/grpo.tasks.jsonl`. The student generates
N rollouts per task; the rubric scores each; group-relative advantages drive
the clipped policy gradient.
**Defaults:** rank=16, alpha=32, lr=1e-5, mode=phase1, num_generations=4,
KL coeff=0.1, clip epsilon=0.20, `--filter-var-min 0.05` (kiln #22).
**Receipt fields to watch:** `groups_filtered`, `reward_mean`, `reward_stdev`,
`reward_saturation_warning`, `lora_delta_norm_summary`, `grad_norm_min_mean_max`.
**Loss is deceptive:** GRPO loss can drop while the model learns to game the
rubric. The blind eval is the only trustworthy signal.
**Reward function is load-bearing.** Everything your rubric rewards, the
model learns. Adversarial design (§0 in capability.md) is mandatory before
writing rubric.py.
**Common failure modes:** all-zero rewards (rule F's reward-variance check
guards against this), length drift (DAPO §2), mode collapse, entropy collapse.
See resources/grpo-mode.md §6.

### §3.4 Agentic-GRPO

**Trainer:** `cuda_grpo_ablation` with ECHO env-mask layer.
**Use when:** Rule A. Multi-turn tool-calling tasks (pi rollouts).
**Data:** pi session JSONLs normalized into ScoredRollout JSONL.
Use `kiln trajectory inspect` for normalization; `rollout.py` for the
gathering loop.
**Defaults:** GRPO defaults + ECHO λ=0.05, env_mask_mode=env_only,
warning_filter=true.
**ECHO is mandatory** — without it, env-token loss is silently masked and
the model never learns to predict its environment. See
`lib/agentic-grpo-notes.md` for the ECHO design.
**Verifier-free mode:** `--no-policy-loss` for paper §5.5 adaptation. ECHO
gradient flows without policy gradient. Reference cap: `pi-script-fixup`.
**Common failure modes:** stale pi sessions, schema drift (Pi 0.75.3 `toolResult`
vs `tool` role), warning-prefix bleed into env mask. See
resources/agentic-grpo-mode.md.

---

## §4. Stage transitions

A pipeline is an ordered sequence of stages. Each stage's `base_adapter` is
the previous stage's `output_adapter`. The transitions below are the recipes
that round 1 + round 2 evidence supports.

### §4.1 The general transition rule

> **A new stage is justified only if (a) the previous stage's closeout shows
> residual headroom > 0.05 in a sub-score the new method is suited for, AND
> (b) the new stage's hypothesis can be stated falsifiably.**

If the previous stage saturated or has no clear sub-score residual, stop.
Don't add a stage just because the agent has GPU budget.

### §4.2 Recommended transitions

#### `none → SFT` (cold start, Rule C/D)
- **When:** baseline outcome < 0.3 OR format headroom > 0.3 of total.
- **Why:** SFT installs the format prior cheaply.
- **Gate:** after SFT, format sub-score ≥ 0.7 AND anchor regression < 0.02.
- **If gate fails:** SFT data is the problem (rank, diversity, examples). Iterate within method before chaining.

#### `SFT → OPD` (format stable, distill process)
- **When:** SFT closed with format sub-score ≥ 0.7 BUT process sub-scores still
  show > 0.08 headroom AND teacher available.
- **Why:** OPD now has a stable starting distribution; teacher signal flows to process.
- **Gate:** after OPD, composite_delta from SFT stage > +0.05, sibling regression < 0.02.
- **If gate fails:** see §4.4 OPD failure recovery.

#### `SFT → GRPO` (format stable, learn from rollouts)
- **When:** SFT closed with format sub-score ≥ 0.7 AND verifier exists AND
  reward variance on SFT-adapter rollouts > 0.05.
- **Why:** SFT prior makes rollouts non-degenerate; GRPO sharpens.
- **Gate:** after GRPO, composite_delta from SFT > +0.05, no
  reward_saturation_warning.

#### `OPD → GRPO` (teacher gap closes, hard tail remains)
- **When:** OPD closed with `env_ce_delta < -0.3` AND hard_eval composite
  shows > 0.05 headroom.
- **Why:** teacher distillation is saturated; policy gradient on hard tail
  is the remaining lever.
- **Gate:** GRPO eval on hard_eval > OPD eval on hard_eval by 2σ.

#### `GRPO → GRPO warm-best`
- **When:** GRPO produced a positive iter with high single-seed lift but
  composite_stdev is large (single-seed luck risk, see CONSOLIDATED_REPORT
  pi-doctest +11pp single → +4.2pp 3-seed).
- **Why:** chain best adapter as base, re-train fresh seed, re-eval.
- **Gate:** the warm-best run's 3-seed mean is within σ of the original
  3-seed mean (reproducibility check). If yes, the recipe is robust.

#### `agentic-GRPO → agentic-GRPO with --no-policy-loss`
- **When:** policy stage saturated AND ECHO env_ce still has headroom.
- **Why:** verifier-free continuation; ECHO env-token CE alone drives
  further env modeling.
- **Gate:** train_receipt.echo_metrics.env_token_ce delta < -0.05.

### §4.3 OPD failure recovery: `OPD → SFT` (high-baseline rescue)

This is the cap #5 pattern. OPD regresses at baseline > 0.80 with variable
rollout quality. Recovery path:

1. Sample 64–128 teacher rollouts on the training prompts at temperature ≤ 0.7.
2. Filter to teacher rollouts that pass `rubric.score_one() > 0.7`.
3. Build `datasets/sft.train.jsonl` from `{prompt, teacher_completion}` pairs.
4. SFT with rank=4 alpha=8 lr=5e-5 on top of base (NOT on top of the failed OPD adapter).
5. Re-eval. If composite > original baseline, you've at least preserved capability.
   Composite > teacher_baseline indicates partial distillation worked through SFT.

If even SFT can't beat baseline, the capability has reached its ceiling on
this base model. Accept it. The next round's distillation may lift it.

### §4.4 Discouraged transitions

| Transition | Why discouraged |
| --- | --- |
| `GRPO → OPD` | Rare; only if a brand-new teacher becomes available. Otherwise OPD on top of GRPO often undoes the sharpening. |
| `OPD → OPD chain` | OPD chains tend to drift; restart from base instead. Within-method warm-best is OK but not common. |
| `SFT → SFT chain on the *same* distribution` | Mono-distribution SFT chains overfit. The exception is `SFT → SFT chain alternating distributions` (§4.5). |
| `non-target method on multi-turn task` | If task is multi-turn tool-calling, agentic-GRPO with ECHO is mandatory. SFT or OPD on multi-turn data without ECHO masking is a known footgun. |

### §4.5 Multi-distribution chains: when a plateau won't break on parameter tweaks alone

Sometimes a series of recipe variants converges on the same composite
across many sweeps — different ranks, learning rates, epochs, filter
thresholds, and chain depths all land within σ of each other. When that
happens, the constraint is rarely the recipe; it is the **training
signal**. The single distribution you have been training on has already
told the model everything it can.

The diagnostic: if 5+ recipes spanning a wide hyperparameter space all
sit at the same composite, treat it as a **data-signal plateau** rather
than a model-capability ceiling. Read the sub-scores. Which axes are
pinned, which are flexing? A pinned axis usually means the training data
doesn't carry signal for that axis at all.

The technique: add a **complementary second distribution** whose signal
is orthogonal to the original one, and chain SFT stages that alternate
between them. Each stage is small (rank 4, lr 1e-5, 1-3 epochs) so the
model never catastrophically forgets the other lesson. Stop when one
direction stops adding lift over the other.

What "complementary" means is capability-dependent — there is no
universal answer:

- For a task with a strict output format and an unbounded set of inputs,
  one distribution might be **rubric-perfect synthesized outputs**
  (drives format precision) and the other **high-scoring model rollouts
  under richer conditioning** (drives outcome correctness on real prompt
  shapes).
- For a reasoning task, the axes might be **worked-through derivations**
  vs **terse final answers**.
- For a code-generation task, **canonical idiomatic solutions** vs
  **edge-case stress tests** might pair well.

Reach for this when you've genuinely run out of single-distribution
recipe ideas and the composite hasn't moved. We don't have a frequency
estimate for how often this applies — early in the round-3 cycle,
treat each capability on its own evidence rather than presuming it
will or won't need a second distribution.

**Reference example:** `caps/pi-faithful-completion` (round-3, 2026-05).
Twelve single-distribution variants (rank 4-16, lr 1e-4 to 5e-6,
threshold >0.5 and >0.7, hard-tail and ideal-only, both SFT and OPD
chains) all plateaued at composite ≈ 0.77. A 6-stage SFT chain
alternating synthesized rubric-perfect outputs with high-scoring
strict-prompt rollouts reached 0.808 — 93.4% of the prompted ceiling.
Sub-score breakdown and per-stage trace in
`caps/pi-faithful-completion/sft_chain_findings.md`. The specific data
pairing there was *one instantiation* of the technique; do not lift
the recipe wholesale, lift the diagnostic.

### §4.6 Cross-stage validation (mandatory)

Between any two stages, `run_pipeline.sh` MUST:

1. **Sibling regression check** — run
   `integration/cross-cap-coherence/capability.oracle.sh <new-stage-adapter>`.
   Abort the pipeline if max sibling delta < -0.02.
2. **Prior-stage preservation** — eval the new stage's adapter against the
   *previous* stage's eval set; composite must be ≥ previous stage's
   composite − σ.
3. **Adapter verify** — `kiln adapter verify <new-stage-adapter>` must pass
   loadable + behavioral checks (kiln #4).

Failing any of these reverts the pipeline to the previous stage and logs
the failed transition as a dead-end in `capability.jsonl`.

---

## §5. Stopping conditions (within a stage)

Stop iterating *within a stage* when ANY of:

- **Within σ.** `composite_delta` within σ for 2 consecutive iters with the
  same method.
- **Reward saturation.** Receipt fires `reward_saturation_warning` (mean > 0.92,
  variance < 0.03).
- **ECHO saturation** (agentic-GRPO). `env_token_ce_delta` < -0.05 absolute,
  not relative.
- **Sibling regression.** Integration check reports max sibling delta < -0.02.
- **Budget hit.** VRAM / wall-clock budget exceeded with no improvement plan.
- **Calibration broken.** `rubric_sanity.py` no longer separates good from bad
  (you may have edited the rubric without re-calibrating; fix that first).

When you stop, ask the §4 transition rule whether a next stage is justified.

---

## §6. Anti-patterns

These were either round-1 mistakes or skill-lore failure modes. Do not repeat.

1. **Don't stack within method without re-eval between.** A 5-iter chain
   with one eval at the end has no idea which iter introduced the regression.
2. **Don't switch methods because "this is hard."** Switch because evidence
   says (the decision tree fired a different rule). Otherwise you're avoiding
   the harder problem, not solving it.
3. **Don't run all stages without integration check between.** Stacked
   adapters clobber siblings more easily than single-stage adapters.
4. **Don't trust single-seed wins.** Round-1 pi-doctest +11pp single-seed
   → +4.2pp 3-seed. Always 3-seed before declaring a kept iter (kiln #33
   `--seeds 3` is the default).
5. **Don't OPD a high-baseline cap with variable rollout quality.** Cap #5.
   Reverse-KL makes the student more confident in whatever it sampled,
   including malformed rollouts.
6. **Don't ignore reward saturation warnings.** Trainer fires this for a
   reason. Either harden the eval (hard_eval pool) or switch to
   `--no-policy-loss`.
7. **Don't treat baseline > 0.95 as success.** It's almost always a rubric
   problem. (Round 1 hit this 3 times.)
8. **Don't run agentic-GRPO without ECHO.** Without ECHO, env-token loss is
   masked and the model never learns to model its environment.
9. **Don't chain training without recording stage_transition_rationale.**
   The rationale is the load-bearing artifact when you re-run the recipe
   six months later or against a new base.
10. **Don't pick a method before measuring inputs in §1.** A premature method
    choice is a wasted GPU run.
11. **Don't conclude "ceiling" from a parameter sweep alone.** If 5+ recipes
    spanning ranks, learning rates, epochs, and filter thresholds all land
    at the same composite, the constraint is likely the training signal,
    not the model. Re-read the sub-scores; check whether a complementary
    data distribution exists that you haven't fed in (§4.5). Distinct
    from rule 7 (which is about rubric saturation, not data saturation).

---

## §7. Worked examples

### Example 1: `pi-faithful-completion` (round-2 winner, +8.3pp single stage)

**Round-2 inputs (from CONSOLIDATED_REPORT):**
- C₀ = 0.7237
- Headroom = 0.276
- Dominant sub-score: process (faithfulness)
- Teacher: not used in round 2
- Verifier: yes (rubric)
- Reward variance: > 0.05 on training tasks
- Multi-turn: yes (pi rollouts)

**Tree applied:**
- Rule A fires (multi-turn) → agentic-GRPO with ECHO.
- Rule F also applies (verifier + variance), so single-stage agentic-GRPO
  was correct.
- Result: 50-iter agentic-GRPO, +8.3pp at iter 50.

**Round-3 pipeline candidate:**
- Stage 1: agentic-GRPO (re-run with kiln-#22 strong-signal filter; should
  reach same +8.3pp faster).
- Stage 2: OPD polish (teacher: 27B Qwen3.6) on remaining process headroom.
- Hypothesis: stage 2 lifts composite from 0.806 to ~0.86 by closing the
  distribution gap to teacher on the hard-tail process cases.

### Example 2: Hypothetical new cap `code-edit-minimality`

**Inputs:**
- C₀ = 0.45
- Headroom heavily in format (0.18) + outcome (0.20)
- Teacher available (27B does 0.78 here)
- Verifier: yes
- Reward variance: 0.02 (low) on baseline rollouts
- Multi-turn: yes (pi edit-then-validate)

**Tree applied:**
- Rule A fires (multi-turn) → agentic-GRPO is the eventual method.
- Rule D applies before (format > 30% of headroom) → SFT bootstrap first.
- Rule E applies (teacher + mid-baseline + process headroom) → OPD as
  the polish.
- Rule F gates GRPO on reward variance — currently 0.02 < 0.05 — so don't
  go to agentic-GRPO yet.

**Recommended pipeline:**
- Stage 1: SFT bootstrap on 64 teacher-curated edit examples → stabilize format.
- Stage 2: OPD against 27B teacher → lift process sub-scores.
- Stage 3 (conditional): re-check reward variance on stage-2 rollouts. If
  > 0.05, agentic-GRPO with ECHO on hard_eval tasks. If still < 0.05, accept
  stage 2 as the cap's pipeline.

This is exactly the kind of sequence the round-1 + round-2 structure could
not express.

---

## §8. When this doc disagrees with itself

If two rules in §2 both apply and recommend different methods, the agent
records both in `pipeline.md` and runs them as **separate stage candidates**.
The capability.jsonl `family` field distinguishes them (e.g.,
`family: "stage-1-opd-candidate"` vs `family: "stage-1-grpo-candidate"`).
Whichever passes the §4.5 cross-stage validation gates is promoted to the
pipeline. The other is logged as a dead-end candidate.

This is the experimentation the user is after. The methodology is data;
the capability is the goal.
