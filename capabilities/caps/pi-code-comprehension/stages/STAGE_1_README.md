# Stage 1 — round-3 re-baseline + round-1 winner reproduction

**Status:** pending pod completion.
**Pod:** `nfi5noknpm9x9j` (lease `pod-e094c16fa777f8fbe72b4ac5`).
**Started:** 2026-05-23T05:00Z (lease).
**Eval mode:** `kiln serve --eval-mode` (round-3 discipline).

## Purpose

Per the round-3 lesson from pi-faithful-completion (round-1 winner regressed
under stricter round-3 eval-mode, see `rounds/round-3/README.md`), every
round-1 winner must be re-validated before its lift can be trusted in the
round-3 pipeline.

## Three eval arms

| Arm | Adapter | Prompt | Purpose |
|-----|---------|--------|---------|
| base | none | default `task_scaffold.PI_PROMPT_TEMPLATE` | the round-3 reference for every later stage |
| iter4 | round-1 `pi-code-comprehension-iter4-h4-echo-0075` (restored from B2) | default | does the round-1 +12.93pp reproduce? |
| (later) strict_prompt | none | strict rubric-aware prompt | pi-faithful-style ceiling diagnostic (if needed) |

Each arm: **3 seeds × 12 eval tasks × 1 generation** under pi multi-turn
sessions (`rollout.py --mode eval`).

## Decision tree on results

- **iter4 reproduces (paired lift ≥ +0.10, ≥3σ)**: candidate to ship as
  round-3 stage-1 (after `pipeline.md` + `stages/stage-1.json` + sibling
  cross-cap check). Plan stage 2 as OPD-from-27B polish for the JSON format
  sub-score (round-2 carry-over improvement #2).
- **iter4 regresses or wash**: trigger strict-prompt diagnostic (stage 2
  in the local plan). If prompted ceiling ≥ +0.10, ship strict prompt as
  stage 1 + plan SFT oscillation chain to bake-in. If prompted ceiling
  also low, escalate to fresh agentic-GRPO with ECHO sweep targeting
  `outcome` sub-score directly.
- **baseline composite shifted >0.03 from round-1's 0.6112**: confirms
  round-3 eval is stricter — log this as a round-3 lessons entry.

## Files

- `stage_1_baseline.sh` — base 3-seed eval driver (runs on pod).
- `stage_1b_iter4_repro.sh` — iter4 restoration + 3-seed eval (runs on pod).
- `stage_2_strict_prompt.sh` — strict-prompt diagnostic (runs on pod if needed).
- `stages/stage-0-baseline-template.json` — schema-version-1 stage record template.
- `/workspace/iter0/{base,iter4,strict_prompt}/seed-N/summary.json` (on pod) — per-seed eval summaries.
- `/workspace/iter0/{base-3seed,iter4-vs-base-paired,strict-vs-base-paired}.json` (on pod) — aggregated paired stats.

## Result schema (to be filled in after pod run)

```
| Arm | mean composite | stdev | paired Δ vs base | σ above 0 |
|-----|----------------|-------|------------------|-----------|
| base | _ | _ | — | — |
| iter4 | _ | _ | _ | _ |
| strict_prompt (if run) | _ | _ | _ | _ |
```

Sub-score detail (each arm, 3-seed mean):

```
| Arm | outcome | grounding | cross_file_caller_recall | invariant_coverage | format_compliance |
```
