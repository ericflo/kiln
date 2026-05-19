# pi-diff-patch-apply: 50-iter GRPO loop — Progress Notes

**Status:** In-flight. Pod: L40S `4z9ofil5k8rlxs` (lease pod-7e747cbee9fc5b604c293d49).
Lease expires `2026-05-19T12:52:01Z`.

## What got built (scaffold v1, all under capabilities/agentic-grpo/pi-diff-patch-apply/)

- **`rubric.py`** — 5-pillar composite (outcome × multiplicative gates + 4 sub-scores), 
  consolation gradient (cap 0.40) when outcome=0, strict format pillar (4 sub-pillars),
  no_unrelated_edits as a base multiplier, tested_before_done as global discount.
- **`build_corpus.py`** — 36 algorithm primitives across 25 easy + 11 hard. 
  Three patch classes: clean (50%), drift (30%, ±4-7 line shifts), incorrect-hunk (20%).
- **`task_scaffold.py`** — init_workdir (with .git init), pi_prompt, build_messages.
- **`rollout.py`** — pi runner, parallel up to 4, T=0.8-1.0 train / T=0.0 eval, pi_trajectory.
- **`select_hard_tasks.py`** — biases training toward drift+incorrect classes.
- **`rubric_sanity.py`** — 3-tier (good/imperfect/bad) calibration test. 8/8 pass.
- **`rescore.py`** — re-score completed iters with updated rubric (no pi re-run).
- **`run_iter.sh`** — full iter recipe driver. 
- **`drive_iters_fast.sh`** + **`drive_iters.sh`** — 50 hypothesis variants.
- **`backup_to_b2.py`** — per-iter B2 backup keyed by date+iter+kind.

## Iter log (live)

(See `capability.jsonl` for canonical record.)

| Iter | Hypothesis | Composite | Δ | Status |
|------|------------|-----------|---|--------|
| 0    | baseline-v1-hard-corpus-strict-format | 0.9419 | — | baseline |
| 1    | h1-default 6×3 hard | 0.8900 | −0.052 | NEGATIVE |
| 2    | h2-strong-filter T=1.0 6×3 hard | TBD | TBD | in flight |
| 3-N  | (chain script auto-runs) | TBD | TBD | queued |

## Key learnings (intermediate)

1. **Baseline is saturated.** 4B base model solves clean/drift Python patches at 0.99+
   composite. To get GRPO signal, we needed both harder tasks AND a stricter rubric.

2. **Rubric tightening worked.** Adding strict format pillars + W_FORMAT 0.05→0.15 dropped
   baseline from 0.989 → 0.942 → opened 5.8% composite headroom for training.

3. **Iter 1 regressed.** Default GRPO recipe (lr 1e-5, rank 16, 6 hard tasks × 3 gens, T=0.8)
   regressed composite by 5.2pp. Only 1 of 6 groups had var > 0.005 → policy update was 
   driven by a single group → over-amplification + format_compliance drop + slower sessions.

4. **Wall clock is the bottleneck.** Pi sessions take 60-180s. At parallel=4, 12-rollout 
   batches take 8-12 min. Full iter (24 train + 24 eval rollouts + filter + training step)
   takes ~50 min. For 50 iters that's ~40 hours; we won't fit overnight.

5. **Pi 0.75.x does not support --max-turns** — turn budget via wall-clock timeout only.

## TODO at session end

- Complete iter 2 (in flight)
- Run as many chain iters as the lease allows (3 max realistically)
- Re-acquire lease if time permits, run more iters
- Write final writeup with the iters we actually completed
- B2 backup all adapters
