# pi-failure-triage 50-iter loop — IN PROGRESS

**Last updated:** 2026-05-19T23:55:46.575354Z (auto-refreshed after every iter)

**Iters with eval data:** 22 / 50
**Iters present:** [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 18, 19, 20, 21, 22, 31, 32, 33, 34, 36]

**★ Best so far: iter 2** — composite 0.9720
(baseline 0.9656, Δ +0.0064)

## Iter table (latest entry per iter)

| iter | composite | outcome | held_out | format | repro | wall_s | recipe |
|------|-----------|---------|----------|--------|-------|--------|--------|
| 0 | 0.9656 | 1.00 | 1.00 | 0.375 | 1.00 | 26.7 | --kind baseline --skip-train --eval-adapter base |
| 1 | 0.9538 | 1.00 | 1.00 | 0.125 | 1.00 | 27.2 | --kind train --lr 1e-5 --filter-var 0.02 (cached after pod reboot) |
| 2 ★ | 0.9720 | 1.00 | 1.00 | 0.500 | 1.00 | 25.5 | --kind train --lr 5e-6 --filter-var 0.02 --rollout-source-iter 1 |
| 3 | 0.9536 | 1.00 | 1.00 | 0.125 | 1.00 | 31.2 | --kind train --lr 2e-5 --filter-var 0.02 --rollout-source-iter 1 |
| 4 | 0.9595 | 1.00 | 1.00 | 0.250 | 1.00 | 46.4 | --kind train --lr 1e-5 --filter-var 0.05 --rollout-source-iter 1 |
| 5 | 0.9599 | 1.00 | 1.00 | 0.250 | 1.00 | 33.6 | --kind train --lr 1e-5 --filter-var 0.0   --rollout-source-iter 1 |
| 7 | 0.9474 | 1.00 | 1.00 | 0.000 | 1.00 | 26.9 | --lr 5e-6 --filter-var 0.02 (source=7) |
| 8 | 0.9536 | 1.00 | 1.00 | 0.125 | 1.00 | 30.6 | --lr 5e-6 --filter-var 0.05 (source=7) |
| 9 | 0.9399 | 1.00 | 1.00 | 0.000 | 1.00 | 44.7 | --lr 5e-6 --filter-var 0.02 --rank 32 --alpha 64 (source=7) |
| 10 | 0.9531 | 1.00 | 1.00 | 0.125 | 1.00 | 30.2 | --lr 5e-6 --filter-var 0.02 --echo-lambda 0.10 (source=7) |
| 11 | 0.9451 | 1.00 | 1.00 | 0.000 | 1.00 | 32.0 | --lr 5e-6 --filter-var 0.02 --grpo-mode phase1_gspo (source=7) |
| 13 | 0.9536 | 1.00 | 1.00 | 0.125 | 1.00 | 22.1 | --lr 5e-6 --filter-var 0.02 --seed 271828 (source=7) |
| 18 | 0.9474 | 1.00 | 1.00 | 0.000 | 1.00 | 40.8 | --lr 5e-6 --filter-var 0.01 (source=17) |
| 19 | 0.9536 | 1.00 | 1.00 | 0.125 | 1.00 | 30.2 | --lr 5e-6 --filter-var 0.005 (source=17) |
| 20 | 0.9521 | 1.00 | 1.00 | 0.250 | 1.00 | 36.6 | --lr 5e-6 --filter-var 0.02 --rank 4 --alpha 8 (source=17) |
| 21 | 0.9536 | 1.00 | 1.00 | 0.125 | 1.00 | 28.0 | --lr 5e-6 --filter-var 0.02 --rank 8 --alpha 32 (source=17) |
| 22 | 0.9454 | 1.00 | 1.00 | 0.125 | 1.00 | 25.3 | --lr 5e-6 --filter-var 0.02 --echo-lambda 0.01 (source=17) |
| 31 | 0.9411 | 1.00 | 1.00 | 0.125 | 0.88 | 24.5 | --lr 5e-6 --filter-var 0.02 (source=31) |
| 32 | 0.9589 | 1.00 | 1.00 | 0.250 | 1.00 | 31.5 | --lr 5e-6 --filter-var 0.02 (source=31) |
| 33 | 0.9599 | 1.00 | 1.00 | 0.250 | 1.00 | 25.8 | --lr 5e-6 --filter-var 0.02 (source=31) |
| 34 | 0.9536 | 1.00 | 1.00 | 0.125 | 1.00 | 28.2 | --lr 5e-6 --filter-var 0.02 (source=31) |
| 36 | 0.9595 | 1.00 | 1.00 | 0.250 | 1.00 | 25.4 | --lr 5e-6 --filter-var 0.02 (source=31) |

## Best adapter — iter 2

- Recipe: `--kind train --lr 5e-6 --filter-var 0.02 --rollout-source-iter 1`
- Composite: **0.9720**
  - vs base (0.9656): Δ +0.0064
- Sub-scores:
  - outcome: 1.000
  - held_out: 1.000
  - fix_local: 1.000
  - no_test_mut: 1.000
  - no_blanket: 1.000
  - repro: 1.000
  - format: 0.500
  - diff_min: 0.940
  - no_dep: 1.000
- B2 location: `b2://clouderic/kiln/pi-failure-triage/20260519-pft-50loop/iter-2-iter/adapter`

## Stable findings

1. **Base 4B is saturated on the bug-fix axis.** outcome,
   held_out_passes, fix_localised, no_test_mutation,
   no_blanket_except, reproduced_before_fixing all = 1.0 across
   baseline and every trained adapter on eval. The model
   correctly root-cause-fixes these bugs without GRPO.

2. **format_compliance is the only movable sub-score.** Baseline
   0.375. Most training REGRESSES it (model converges to terse
   "Done." finals). Only iter 2's recipe lifts it (to 0.500).

3. **lr=5e-6 is the only sweet-spot LR.** 1e-5, 2e-5, 1e-6, 7.5e-6
   all regress. At lr=5e-6 the outcome is data-dependent: iter
   2 (rollouts from iter1's pool) hit 0.972; iter 7 (rollouts
   from iter7's pool) hit 0.947 with the same hyperparams.

4. **No hyperparam axis except LR×data moves the needle.**
   Rank (4, 8, 16, 32), ECHO λ (0.01, 0.03, 0.05, 0.07, 0.10),
   filter-var (0.005, 0.01, 0.02, 0.05), grpo-mode (phase1,
   gspo, cispo, reinforce), seeds — all give same-or-worse
   results than the default `--mode phase1 --lr 5e-6 -fv 0.02`.

5. **The cap is rubric-limited.** Headroom = 1 − 0.966 = 0.034
   composite, of which 0.025 lives in format_compliance × 0.05
   weight and 0.003 in diff_minimality. To get more signal, the
   rubric needs to gate format multiplicatively (not add it).

## Loop budget / failure mode notes

- **Pod TTL:** kiln-pool leases expire at 10800s (3h). After
  hibernation a new pod is allocated (disk lost). Bootstrap
  (~10 min with sccache) + fresh rollouts (~40 min) per pod
  cycle. Realistic budget: 5-6 iters per pod cycle on cached
  rollouts, ~25 min/iter.
- **Auto-fail batches detect hibernation** via grep on
  `AttributeError` in the iter log; batch exits with status 99.
- **B2 backup per iter** ensures no adapter loss across
  hibernations.

## Files

- `capability.md` — the contract
- `rubric.py` — 9-component composite scorer
- `task_scaffold.py` — workspace init + pi prompt
- `build_corpus.py` — 50 planted-bug task templates
- `rubric_sanity.py` — root-cause vs symptom calibration (PASS)
- `rollout.py` — pi-headless runner
- `run_iter.sh` — one iter (rollouts → train → eval)
- `run_batch.sh` — N iters with cached rollouts
- `backup_to_b2.py` — per-iter B2 backup
- `_append_iter_log.py` — pod → capability.jsonl row
- `_refresh_in_progress.py` — this file regenerator
- `capability.jsonl` — append-only iter log
- `IN_PROGRESS.md` — this file
- `FINAL_WRITEUP.md` — final writeup
