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

## Pilot target (Tier 1) — STRUCTURAL FINDING, not the +12pp chain

- **`caps/pi-faithful-completion`** — round-1 winner +8.3pp single-stage;
  round-3 pilot for multi-stage.

**Pilot outcome (2026-05-21):** The round-1 winner does **not reproduce**
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
