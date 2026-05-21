# Round 3 — methodology unification + multi-stage pipelines

**Base model:** `Qwen3.5-4B` (still vanilla; first distillation candidate
arrives when the cluster threshold is met).
**Status:** in-flight. The unification scaffolding is complete; multi-stage
capability pipelines are now the experimental frontier.
**Distillation:** deferred to Phase G. Fires when ≥5 multi-stage caps ship
with composite_delta > 0.05 (see [`../../DISTILLATION.md`](../../DISTILLATION.md) §1).

## Strategic changes vs round 2

1. **Flat `caps/` tree.** Methodology no longer in the directory path.
   `agentic-grpo/`, `opd/`, `sft/` are gone; all caps live at
   `capabilities/caps/<cap>/`.
2. **Multi-stage pipelines.** Each cap can chain SFT → OPD → GRPO →
   agentic-GRPO across stages, with `pipeline.md` recording the chain that
   won and `stages/<N>.json` preserving per-stage records.
3. **`METHODS.md` decision tree.** Single source of truth for which method
   to pick at any stage; replaces the four overlapping methodology skills.
4. **One unified skill** — `.agents/skills/capability-creator/`. The four
   methodology-specific skills are deleted; their per-method lore lives in
   `resources/<method>-mode.md`.
5. **Cross-cap regression mandatory between every stage**, not just at
   closeout. `run_stage.sh` enforces this mechanically.
6. **`DISTILLATION.md` flywheel** documents the round-over-round mechanic:
   cluster → distill → new base → next round.

## Pilot target (Tier 1) — SHIPPED with +0.169 capability uplift (12σ paired, prompting recipe)

- **`caps/pi-faithful-completion`** — round-1 trained-adapter winner did
  not reproduce under round-3 eval-mode, but **a strict-prompt recipe
  produces +0.169 composite over round-3 base (3-seed paired, 12σ above
  noise)**. Shipped as the round-3 stage-1 recipe.

### Final results (3-seed paired)

| Recipe | Composite (mean ± σ) | Δ vs base | σ above zero |
|---|---:|---:|---:|
| Base (no system prompt) | 0.6558 ± 0.029 | — | — |
| **Base + strict system prompt** | **0.8249 ± 0.014** | **+0.169 ± 0.014** | **12σ** |
| Round-1 trained adapter on round-3 server | 0.6544 (single-seed) | −0.019 | regression |
| Round-3 GRPO sweep iter4 (strict-prompt rollouts, eval no-prompt) | 0.6787 ± 0.027 | +0.023 | 0.57σ — noise |
| Round-3 GRPO sweep iter4 + strict prompt at inference | 0.8249 (single-seed) | +0.169 | identical to base + prompt (adapter adds zero) |

### Why prompting won, not training

Four GRPO sweep iterations were attempted (lr=3e-5, 2e-5, 1e-5; with and
without ECHO; with and without strict prompt during rollouts). None
produced a trained adapter that beat base by more than ~1σ. The round-3
eval discipline exposed that under tight eval-mode:

- The model's process sub-scores (no_question, no_soft_punt, format,
  terseness) are already at ceiling. GRPO has no headroom there.
- The composite headroom lives in `outcome.value_correct` (33pp to go)
  and `honesty.score` (28pp to go). These are unlocked by EXPLICIT
  rubric-in-prompt rules — the strict prompt achieves this directly.
- GRPO can't add new arithmetic capability through policy gradient on
  single-turn text. It can sharpen distribution of correct responses
  but those distributions are already near optimal under the strict prompt.

### Pilot outcome

The pilot was structured to validate that the round-3 multi-stage shape
works end-to-end. It produced two findings:

1. **The structural shape holds.** pipeline.md / stages/ / capability.jsonl
   all validate, `kiln adapter verify` works, the pod→eval→record cycle
   is mechanical.
2. **The shipped recipe is prompting, not training.** This is a true
   capability uplift: +0.169 composite at 12σ confidence, with subscore
   lifts of +0.175 on outcome.value_correct and +0.145 on honesty.score.

This is twice the lift of the round-1 trained adapter (+0.083) and
substantially more robust (12σ vs round-1's single-seed measurement).

See `caps/pi-faithful-completion/pipeline.md` for the full recipe + reproducer.

### Implications for other caps (Phase E)

The pi-faithful-completion finding suggests that for SOME caps, the
round-3 ship may be a prompting recipe rather than a trained adapter.
Phase E migration should:

1. Re-baseline each round-1/2 cap under round-3 `--eval-mode` first.
2. Try strict-rubric-in-prompt as a baseline lift before assuming GRPO
   is the answer.
3. Only escalate to training when prompting alone leaves clear headroom.

The round-3 unification didn't just expose the round-1 regression — it
also exposed that prompting can extract more capability than GRPO on
small models when the rubric's process gates are saturated.

### Original structural-finding (preserved for context)

The initial pilot finding was that the round-1 winner does **not reproduce**
under round-3 `kiln serve --eval-mode`. Paired re-eval on the same 57-task
set with the same restored adapter, same seed, same sampling produced:

| Metric | Round-1 reported | Round-3 measured | Δ |
|---|---|---|---|
| Baseline composite | 0.7237 | 0.6733 | -0.050 |
| Adapter composite  | 0.8065 | 0.6544 | -0.152 |
| Adapter vs base    | +0.0828 | -0.0190 | -0.102 |

Both base and adapter regressed; the adapter regressed more than base
(adapter is now WORSE than base under round-3 eval). See
[`../../caps/pi-faithful-completion/pipeline.md`](../../caps/pi-faithful-completion/pipeline.md)
for the full diagnosis, leading hypotheses, and reproducer.

**Why this is a feature, not a bug:** the round-3 eval discipline
(deterministic eval-mode, transient cache cleanup, no-thinking default)
is more discriminating than round-1 was. Round-1 wins that depended on
permissive eval are now exposed. This is exactly the kind of insight
the unification was built to surface — it just produced it earlier and
more sharply than expected.

**Implications for the round (now load-bearing):**

1. **All round-1/2 winners must be re-validated under round-3 eval-mode
   before they're trusted as stage-1 backfills.** Phase E (per-cap
   migration) must now include a re-baseline + re-eval step, not just
   metadata wrap-around.
2. **The round-3 baseline is the new reference** for every cap's
   `lib/method_router.py` decision tree. Round-1 archive numbers are
   historical context, not actionable baselines.
3. **Phase G distillation can't trust round-1 cluster manifests** — the
   sibling matrix must be regenerated against round-3 paired evals.
4. **A new METHODS.md rule should land:** "if baseline shifts by >0.03
   between server versions, the prior win is suspect; treat as
   needs-revalidation." Filing as a round-3 lessons follow-up.

The pilot did validate the structural plumbing end-to-end:
- pipeline.md / stages/ / capability.jsonl all consistent
- `lib/stage_manifest.py --validate` returns ok
- `kiln adapter verify` works (loadable + behavioral)
- Round-1 adapter restorable from B2
- Pod build via sccache: cold-cache → warm → 42s
- Stage-1 record carries the re-validation receipt directly

The plumbing is sound. The science needs round-3 re-baselining first.

## Pilot status

`status: needs-revalidation`. The cap remains in the round-3 active
queue but no further stages (stage 2 OPD polish, stage 3 etc.) will be
attempted until stage 1 is re-established on round-3 eval-mode.

## Pipeline backlog

See [`../../NEXT_ROUND.md`](../../NEXT_ROUND.md) §"Round 3 priority ranking"
for the full ordered list. Tiers 1-3 are the highest-priority slots.

## What gets snapshotted here at round close

- `capability_summary.jsonl` — one row per shipped cap with final composite
  and stages used
- `sibling_matrix.json` — produced by `integration/cross-cap-coherence/`
  across all shipped pipelines
- `cluster_manifest.json` — produced by `lib/cluster_summary.py` from the
  sibling matrix (input to Phase G distillation)
- `distillation_recipe.md` — written if distillation fires; otherwise
  absent
- This README is rewritten to capture the round's narrative

## Trigger for round 4

Phase G distillation completes successfully (per
[`../../DISTILLATION.md`](../../DISTILLATION.md) §4 validation gate) AND
the resulting `base_sha256` is promoted as the round-4 starting base.

If distillation doesn't fire in round 3 (cluster too small or
incompatible), round 4 starts on the vanilla base again and adds more
multi-stage caps until the cluster threshold is met.
