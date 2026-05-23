# pi-code-comprehension — round-3 pipeline

**Status (2026-05-23):** ceiling-documented. Four orthogonal training arms tested; none clears the +0.10 / 3σ ship gate. Best arm (iter4 ECHO) sits at +0.022 to +0.039 paired lift / 0.78σ to 2.70σ. Round-3 commits the iter4 ECHO recipe as the strongest reproducible step and records the rubric ceiling with evidence.

## Why round-3 stops at the ceiling

Round-1 reported +0.13 single-seed for iter4 ECHO. Under round-3's stricter 3-seed paired eval, the same adapter compresses to +0.022 to +0.039 with paired-lift stdev 0.028. The composite has a tight tradeoff between `outcome` (which responds to extra reasoning) and `format_compliance` / `cross_file_caller_recall` / `invariant_coverage` (which degrade when the model is asked to think harder). Three additional methodologies confirmed the ceiling:

| Stage | Method | Paired lift | σ | Verdict |
|---|---|---|---|---|
| 1a | iter4 ECHO (round-1 winner re-eval, session-1 pod) | **+0.039** | 2.70 | best; below +0.10/3σ ship gate |
| 1b | SFT-ideal r=4 α=8 from base (16 shortest gold-derived examples) | +0.006 | 0.12 | flat; outcome up, invariants down |
| 1c | Strict-rubric-in-prompt diagnostic (no training) | -0.007 | -0.18 | no prompted ceiling; format breaks |
| 2a | iter4 ECHO re-run on session-2 pod | +0.022 | 0.78 | confirms cross-pod variance |
| 2c | SFT r=8 α=16 chained on iter4 (gold-derived, --max-examples 64) | **-0.055** | -1.35 | regressed; chain destabilizes iter4 gains |
| 3 | Best-of-N self-distillation (188 train rollouts → filter → SFT r=8 α=16) | **-0.033** | -1.10 | regressed; even own best work degrades model |

The composite formula and rubric are dominated by `outcome`+`grounding` (lift-responsive) and `format_compliance`+`cross_file_caller_recall`+`invariant_coverage` (drop-prone under training pressure). Lifting one axis costs the others, capping reachable paired lift around +0.04 with the current LoRA toolkit and 4B base. Stage 3 specifically exercises the goal's "prompted-ceiling distillation" arm at full scale (188 train tasks, filtered to ~95 high-quality rollouts) and confirms the ceiling: even distilling the model's own best work regresses every sub-score, including invariant_coverage (-0.120).

## Recommended recipe — iter4 ECHO

iter4 ECHO is the strongest training arm tested in either round-1 or round-3. Recipe:

- **Method**: agentic-GRPO Phase 1 + ECHO (echo λ=0.075)
- **Base**: `/workspace/Qwen3.5-4B` (no prior adapter)
- **Hyperparameters**: rank=16, alpha=32, lr=1e-5, epochs=1, num_train_tasks=8, num_generations=4, seed=3141592653
- **Pre-trained adapter snapshot**: `b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iter4/adapter.tgz`

To reproduce evaluation only, restore the adapter from B2 and run `./run_pipeline.sh` — see Reproducer section.

## Sub-score breakdown (iter4 ECHO, session-1 pod)

```
sub          base    iter4   Δ
outcome      0.754   0.789   +0.035
grounding    0.483   0.532   +0.049
cf_callers   0.944   1.000   +0.056   (saturated)
invariants   0.370   0.375   +0.005   (unmoved frontier)
format       0.944   1.000   +0.056   (saturated)
```

The unmoved sub-score is `invariant_coverage`. Lifting it requires reasoning about implicit invariants from code structure — a capability the 4B base lacks. Round-3 documented this with the Stage 1b (gold-derived SFT) and Stage 2c (chain) attempts; both made invariants WORSE rather than better. Future rounds should target this axis specifically (e.g. OPD with a teacher that reliably extracts invariants, or shaped GRPO reward weighting `invariant_coverage` higher).

## What this round did NOT try (left for future rounds)

- **Agentic-GRPO with composite-shaped reward** weighting `invariant_coverage` ≥ 2×. Requires kiln cuda_grpo_ablation modifications. The +0.039 best-arm came from GRPO+ECHO at the default reward weighting; shaped reward could in principle target the unmoved invariant_coverage axis.
- **OPD with external code-reasoning teacher** (Claude Sonnet / Opus or GPT-4o-mini via `cuda_opd_remote --teacher-url --teacher-model`). Requires API key setup + cost guardrails. Most likely to break through because an external teacher with strong code reasoning could produce trajectories that lift invariant_coverage without breaking format.
- **Larger LoRA ranks** (r=32+) — round-1 H9 found rank-32 cost -0.029 vs iter4, so unlikely.

## Reproducer

`./run_pipeline.sh` does:

1. Restore the iter4 ECHO adapter from B2.
2. Verify the adapter loads (`kiln adapter verify`).
3. Run a 3-seed paired eval (seeds 1, 2, 3) on the eval task set.
4. Compute paired lift vs the session base 3-seed.
5. Print the JSON summary.

Prerequisites (handled by `lib/pod_bootstrap.sh`):
- A6000 pod with `ghcr.io/ericflo/kiln-runpod:latest`
- `kiln serve --eval-mode` running on :8420
- pi binary installed (Node 22 + `@earendil-works/pi-coding-agent`)
- B2 creds available in the env (`pod_export_b2_creds`)
- kiln built with `--features cuda --bin kiln --bin kiln-bench`

## Cross-cap-coherence

Cross-cap-coherence (the goal's "<0.02 regression on any sibling cap" gate) is **not applicable** for this round-3 closure because no new adapter is being shipped or deployed. The recommended recipe is iter4 ECHO, an adapter from round-1 that already passed earlier sibling-cap regression checks during its original landing (`adapter b2://clouderic/kiln/pi-code-comprehension/BEST_ADAPTER_iter4/adapter.tgz`). The three new training arms in round-3 (stages 1b, 2c) are documented as no-ship; they were not promoted into any registry that sibling caps share. The strict-prompt diagnostic (1c) and iter4 re-eval (2a) are eval-only and made no model changes. If a future round lands a new ship-worthy adapter for this cap, that round must include a sibling-regression sweep across the cap registry before promotion.

## Round-3 artifacts

- `stages/stage-0-baseline.json` — session-1 base 3-seed (0.6293 ± 0.022)
- `stages/stage-1-iter4-repro.json` — iter4 ECHO session-1 (+0.039 / 2.70σ)
- `stages/stage-1b-sft-ideal-from-gold.json` — SFT-ideal flat (+0.006 / 0.12σ)
- `stages/stage-1c-strict-prompt-ceiling.json` — strict-prompt diag (-0.007 / -0.18σ)
- `stages/stage-2a-iter4-resession.json` — iter4 ECHO session-2 (+0.022 / 0.78σ)
- `stages/stage-2c-sft-chained-on-iter4.json` — chain regresses (-0.055 / -1.35σ)
- `stages/stage-3-bestof-self-distill.json` — best-of-N self-distillation regresses (-0.033 / -1.10σ); listed prompted-ceiling-distillation arm fully exercised
- `pipeline.md` (this file)
- `run_pipeline.sh` (reproducer)
- `capability.jsonl` (one row, status=ceiling-documented)
