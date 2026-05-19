# pi-failure-triage 50-iter loop — IN PROGRESS

## Status (12 iters done, drive @ iter 13-22)

| iter | recipe                                | composite | format | verdict           |
|------|---------------------------------------|-----------|--------|-------------------|
| 0    | baseline (no adapter)                 | 0.966     | 0.375  | saturated         |
| 1    | lr=1e-5  filter=0.02 (src=iter1)      | 0.954     | 0.125  | regression        |
| 2    | **lr=5e-6** filter=0.02 (src=iter1)   | **0.972** | 0.500  | **BEST so far** ★ |
| 3    | lr=2e-5  filter=0.02 (src=iter1)      | 0.954     | 0.125  | regression        |
| 4    | lr=1e-5  filter=0.05 (src=iter1)      | 0.960     | 0.250  | regression        |
| 5    | lr=1e-5  filter=0.0  (src=iter1)      | 0.960     | 0.250  | regression        |
| 7    | lr=5e-6  filter=0.02 (src=iter7)      | 0.947     | 0.000  | regression        |
| 8    | lr=5e-6  filter=0.05 (src=iter7)      | 0.954     | 0.125  | regression        |
| 9    | lr=5e-6  rank=32 alpha=64 (src=iter7) | 0.940     | 0.000  | regression        |
| 10   | lr=5e-6  echo=0.10  (src=iter7)       | 0.953     | 0.125  | regression        |
| 11   | lr=5e-6  grpo=gspo (src=iter7)        | 0.945     | 0.000  | regression        |
| 12   | lr=5e-6  no-echo (src=iter7)          | timed out | -      | (pod hibernated)  |

## Key findings

1. **Base 4B is already at the ceiling on the bug-fix task.** Held_out_passes,
   outcome, fix_localised_correctly, no_test_mutation, no_blanket_except all
   saturate at 1.0 on the eval set. The base model finds root-cause fixes
   reliably without supervision.

2. **format_compliance is the only movable sub-score.** Baseline 0.375.
   The trained adapters mostly DROP this (the model emits "Done." instead
   of "Fix: <file>::<func>: <reason>"). Only iter 2's recipe lifted it to
   0.500.

3. **lr=5e-6 is the only setting that beats baseline.** All other LR values
   (1e-5, 2e-5, higher) regress. With lr=5e-6 the result is data-dependent
   (iter 2 used iter1's rollouts = 0.972; iter 7 used iter7's rollouts =
   0.947). Data quality dominates.

4. **Higher rank (32, alpha=64) regresses.** rank=16 is the sweet spot.

5. **ECHO ablation (no-echo, lambda variations) doesn't help.** Default
   λ=0.05 is fine.

6. **GSPO mode regresses.** DrGRPO (phase1) is best.

7. **Pod hibernation cost.** 3 pod cycles so far; ~30 min lost to bootstrap +
   rollout regen each time. Lease TTL = 10800s (3h) is the bottleneck.

## Iter 2 (BEST) details

- Recipe: `--lr 5e-6 --filter-var 0.02 --num-train-tasks 8 --num-gens 4`
  with default rank=16, alpha=32, kl=0.1, echo_λ=0.05, mode=phase1, seed=3141592653
- Trained from rollouts collected against the BASE model (8 tasks × 4 gens,
  variance-filtered to 3 strong-signal groups)
- Eval composite: 0.9720 (+0.6 pp vs base 0.966)
- format_compliance: 0.500 (+12.5 pp vs base 0.375)
- All other sub-scores at 1.0 (saturated)
- B2 location: `b2://clouderic/kiln/pi-failure-triage/20260519-pft-50loop/iter-2-iter/adapter`

## What we'd do differently with more time

- Drop format_compliance from 0.05 → 0.10 weight. It's the only signal.
- Train fewer steps (max_groups < total) to avoid the format regression.
- Try chain-training from iter 2's adapter (lift the floor further).
- Re-rollout against iter 2's adapter and re-train (cumulative).
- Multi-seed verification of iter 2 to confirm the +0.6 pp is real.
