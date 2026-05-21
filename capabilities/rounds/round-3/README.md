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

## Pilot target (Tier 1)

- **`caps/pi-faithful-completion`** — round-1 winner +8.3pp single-stage;
  round-3 pilot for multi-stage. Hypothesized chain:
  - Stage 1: SFT bootstrap (stabilize format)
  - Stage 2: OPD polish against 27B teacher (lift process sub-scores)
  - Stage 3: agentic-GRPO with ECHO (sharpen on hard_eval tasks)
  - Expected final composite: 0.86-0.92 vs 0.806 round-1 single-stage.

If this works, the round-3 multi-stage pattern is validated and the
remaining tiers can adopt it.

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
