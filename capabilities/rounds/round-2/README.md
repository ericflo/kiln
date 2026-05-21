# Round 2 — hardening + uniform layout

**Base model:** `Qwen3.5-4B` (vanilla; no distillation yet).
**Status:** hardening pass shipped; layout normalized; round 3 unified the
methodology buckets.
**Distillation:** none. Round 2 was still mostly single-method; round 3
multi-stage pipelines are the prerequisite for the first distillation.

## Strategic changes vs round 1

(Source: round-2 `LAYOUT.md` and `NEXT_ROUND.md` before the round-3
overwrite.)

1. **Uniform per-paradigm layout.** Every cap directory normalized to a
   common shape (capability.md, capability.config.json, capability.jsonl,
   rubric.py, rubric_sanity.py, build_corpus.py, capability.oracle.sh,
   run_iter.sh + optional rollout.py + calibration/ + datasets/ +
   archive/ for prior history).
2. **`rubric_sanity.py` mandatory.** Round 1 hit "rubric too lax" 3 times.
   The gate runs BEFORE training in every iter and blocks on calibration
   failure (margin > 0.2 required between good/bad fixtures).
3. **Multiplicative format gate** as the default composite shape:
   `composite = outcome × format × (process + base)`. Reshaped caps:
   `pi-diff-patch-apply`, `pi-failure-triage`. The round-1 v1 rubrics
   are preserved under each cap's `archive/`.
4. **`integration/cross-cap-coherence/`** new — eval-only suite that runs
   any adapter against held-out slices of every member cap; flags
   `per_cap_delta < -0.02` as regressions.
5. **`hard_eval.tasks.jsonl` pattern** per cap — round-failures-derived
   pool where base composite < 0.5. Cleaner signal than the standard
   eval set when baseline is near saturation.
6. **4 new caps targeting high-leverage process behaviors:**
   `pi-error-recovery`, `pi-context-aware-edits`, `pi-incremental-progress`,
   `pi-search-then-read`.
7. **2 saturated caps reshaped:** `pi-diff-patch-apply`,
   `pi-failure-triage` got v2 multiplicative-gate rubrics.
8. **`pi-tool-call-efficiency` repurposed** as transfer-eval-only (not a
   training cap; wraps other caps' adapters).
9. **`pi-source-mod-workflow` reframed** as integration test (full
   clone→PR was too long for clean GRPO signal).
10. **The 40 kiln improvements landed.** All issues in
    `KILN_IMPROVEMENT_ISSUES.md` complete by 2026-05-21 actual-model
    validation checkpoint.

## What round 2 attempted but did not yet ship

Round 2's eval cycle was largely planning + harness hardening. The four
round-1 winners carry forward into round 3:

| Cap | Round 1 result | Round 3 plan |
| --- | --- | --- |
| `pi-faithful-completion` | +8.28pp 3-seed | Multi-stage pilot (SFT → OPD → agentic-GRPO) |
| `pi-code-comprehension` | +12.93pp | Cross-file generalization eval + OPD polish |
| `pi-doctest` | +4.2pp 3-seed | Hidden tests sub-score (deferred §0 A1 mitigation) |
| `pi-code-search` | +2.4pp | `precision_of_read` sub-score + harder corpus |

The 6 OPD caps and 3 SFT caps were all scaffolded with the round-2 layout
but mostly haven't yet completed shipped iters with the new harness.

## Cluster manifest

`null` — no distillation occurred.

## Sibling matrix

Started but incomplete. `integration/cross-cap-coherence/` infrastructure
exists; running it as a matrix across the 4 round-1 winners was a deferred
round-2 task.

## What changed structurally in round 3 (the unification)

Round 3 collapses the per-paradigm buckets into a flat `caps/` tree because
real capability wins require sequencing methodologies (SFT bootstrap → OPD
polish → GRPO sharpen), not committing one paradigm per cap. See
[`../../README.md`](../../README.md) round-3 status section + the new
top-level [`METHODS.md`](../../METHODS.md), [`PIPELINE.md`](../../PIPELINE.md),
[`DISTILLATION.md`](../../DISTILLATION.md).

## Where to find round-2 artifacts

- Per-cap archives — `capabilities/caps/<cap>/archive/`
- `capabilities/CONSOLIDATED_REPORT.md` — round-1 lessons that informed
  round-2 hardening (not yet re-written for round 2 closeout)
