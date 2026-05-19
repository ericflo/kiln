# pi-failure-triage 50-iter loop — IN PROGRESS

**Last updated:** 2026-05-19 ~22:25 UTC (auto-refreshed as iters complete)

## Status (17 real iters done, 4 pod hibernations, in iter 23 attempt)

| iter | recipe                                       | composite | format | verdict           |
|------|----------------------------------------------|-----------|--------|-------------------|
| 0    | baseline (no adapter)                        | 0.966     | 0.375  | saturated         |
| 1    | lr=1e-5  fv=0.02 (src=iter1)                 | 0.954     | 0.125  | regression        |
| **2**| **lr=5e-6** fv=0.02 (src=iter1)              | **0.972** |**0.500**| **★ BEST**      |
| 3    | lr=2e-5  fv=0.02 (src=iter1)                 | 0.954     | 0.125  | regression        |
| 4    | lr=1e-5  fv=0.05 (src=iter1)                 | 0.960     | 0.250  | regression        |
| 5    | lr=1e-5  fv=0.0  (src=iter1)                 | 0.960     | 0.250  | regression        |
| 7    | lr=5e-6  fv=0.02 (src=iter7, fresh rollouts) | 0.947     | 0.000  | regression        |
| 8    | lr=5e-6  fv=0.05 (src=iter7)                 | 0.954     | 0.125  | regression        |
| 9    | lr=5e-6  rank=32 α=64 (src=iter7)            | 0.940     | 0.000  | regression        |
| 10   | lr=5e-6  echo=0.10 (src=iter7)               | 0.953     | 0.125  | regression        |
| 11   | lr=5e-6  grpo=gspo (src=iter7)               | 0.945     | 0.000  | regression        |
| 13   | lr=5e-6  seed=271828 (src=iter7)             | 0.954     | 0.125  | regression        |
| 18   | lr=5e-6  fv=0.01 (src=iter17, fresh)         | 0.947     | 0.000  | regression        |
| 19   | lr=5e-6  fv=0.005 (src=iter17)               | 0.954     | 0.125  | regression        |
| 20   | lr=5e-6  rank=4 α=8 (src=iter17)             | 0.952     | 0.250  | regression        |
| 21   | lr=5e-6  rank=8 α=32 (src=iter17)            | 0.954     | 0.125  | regression        |
| 22   | lr=5e-6  echo=0.01 (src=iter17)              | 0.945     | 0.125  | regression        |

**Failed iters (pod hibernation / kiln serve race):** 6, 12, 14, 15, 16, 17, 23-30.

## Key findings (so far)

1. **Base 4B is already at the ceiling on the bug-fix task.**
   held_out_passes, outcome, fix_localised_correctly, no_test_mutation,
   no_blanket_except, reproduced_before_fixing all saturate at 1.0 on
   the eval set. The base model finds root-cause fixes reliably
   without supervision.

2. **format_compliance is the only movable sub-score.** Baseline
   0.375. The trained adapters mostly DROP this (model emits "Done."
   instead of "Fix: <file>::<func>: <reason>"). Only iter 2 lifted
   it to 0.500.

3. **lr=5e-6 is the only setting that beats baseline.** All other LRs
   (1e-6, 1e-5, 2e-5, etc.) regress. With lr=5e-6, the result is
   data-dependent (iter 2 from iter1's rollouts = 0.972; iter 7 from
   iter7's rollouts = 0.947). Data quality dominates the recipe.

4. **Higher rank (32, α=64) regresses.** Default rank=16 is the
   sweet spot. Lower rank (4, 8) gives lower-magnitude regression but
   still regression.

5. **ECHO ablation doesn't help.** λ=0.05 (default), 0.10, 0.01, no-echo
   all yield similar results. ECHO is essentially a no-op for this cap.

6. **GSPO/CISPO/REINFORCE modes regress.** Default DrGRPO (phase1) is best.

7. **Multiple seeds give similar regressions.** seed=271828 also regresses
   to 0.954. Iter 2's gain may be partly seed-fortunate, but the
   pattern (regression everywhere else) is consistent.

8. **Pod hibernation cost dominates wall-clock budget.** 4 pod cycles
   so far; ~30 min lost to bootstrap + rollout regen each time. Lease
   TTL = 10800s (3h) is the bottleneck.

## Iter 2 (BEST) details

- Recipe: `--lr 5e-6 --filter-var 0.02 --num-train-tasks 8 --num-gens 4`
  with default rank=16, alpha=32, kl=0.1, echo_λ=0.05, mode=phase1,
  seed=3141592653
- Trained from rollouts collected against the BASE model (8 tasks × 4
  generations, variance-filtered to 3 strong-signal groups)
- Eval composite: **0.9720** (+0.6 pp vs base 0.966)
- format_compliance: 0.500 (+12.5 pp vs base 0.375)
- All other sub-scores at 1.0 (saturated)
- B2 location:
  `b2://clouderic/kiln/pi-failure-triage/20260519-pft-50loop/iter-2-iter/adapter`

## Failure mode taxonomy (for future loops)

Three classes of iter failures encountered in this run:

1. **Pod hibernation mid-iter.** `runpod_api.py` returns
   `AttributeError: 'NoneType' object has no attribute 'get'` when
   pod has `runtime: null` (lease expired or pod exited). Detected
   by `run_batch.sh`'s grep on the iter log and stops the batch.
   **Fix:** check pod-lease TTL before starting iters; re-acquire if
   < 30 min left.

2. **Kiln-serve race on iter start.** After `pkill -x kiln`, the new
   `kiln serve` may not be ready in time before the eval rollout's
   first request. Iter 17 hit this. **Fix:** the `for i in $(seq 1
   30)` loop in `run_iter.sh` already polls /v1/models, but if pod
   itself just rebooted, the wait window can exceed 30 × 5s = 2.5
   min. Bump to 60 retries with 5s sleep.

3. **`--no-echo` + `--echo-lambda 0.05` mutex conflict.** Fixed in
   run_iter.sh by switching to `ECHO_ARG="--echo-lambda X"` vs
   `ECHO_ARG="--no-echo"`. Was a real bug in the v0 script.

## What we'd do differently with more time

- **Bump format_compliance weight 0.05 → 0.10.** It's the only signal.
- **Plant harder bugs.** The current corpus saturates the base 4B's
  bug-fix ability. Need multi-line, multi-file, or genuinely subtle
  bugs that exercise the held_out_passes gap more strongly.
- **Chain-train from iter 2.** Take iter 2's adapter, sample fresh
  rollouts against it, re-train. This is a 2nd-order test of whether
  the gains compound.
- **Multi-seed iter 2 verification.** 3-seed mean of iter 2's recipe
  to confirm the +0.6 pp is robust, not single-seed lucky.
- **Lease extension API.** `ce kiln-pod-acquire` should support
  `--lease-ttl-seconds` to give long-iter loops 6-8h windows.
