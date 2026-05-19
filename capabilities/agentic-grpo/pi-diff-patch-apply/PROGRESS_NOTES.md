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

## Session 2026-05-19 (afternoon-evening) — continuation through iter 9

### Iters completed
| Iter | Slug | Composite | Δ vs baseline | Notes |
|------|------|-----------|---------------|-------|
| 8    | h8-no-echo-12tasks-3gens | 0.840 | -0.102 | First successful trained adapter after pi-setup + pytest fixes |
| 9    | h9-echo-0.10 | 0.762 | -0.180 | Higher ECHO over-anchored — drift class dropped most (0.907 → 0.794) |

### Lessons from iters 8-9
- Pi sessions are noticeably slower with trained adapters (60s → 120-180s).
- The trained adapter consistently degrades `incorrect` class (the hardest one).
- ECHO 0.10 hurts more than no-ECHO (h9 -18pp vs h8 -10pp).
- The "best" recipe direction is still toward smaller LoRA rank + lower lr + minimal training.

### Kiln/infra fixes landed
- `bootstrap_pod.sh`: install pytest, run kiln pi-setup, --features cuda, KILN_CUDA_ARCHS=80, correct Qwen3.5-4B repo name.
- `run_iter.sh`: kiln serve health check + smoke test before eval.
- `drive_iters_fast.sh`: removed `set -e` so single-iter failures don't kill the 50-iter loop; per-cap env file `/tmp/grpo-pod-pdp.env`; enriched log row with sub_scores/class_means.
- `backup_to_b2.py`: auto-installs boto3.
- `kiln-polish.jsonl`: appended 6 notes documenting these gaps.

### State at session end
- Old A6000 pod (9jshui49gl9up2) released.
- New A100 80GB pod acquired (36xpt4xbmezqtc, lease pod-b05f748d3fd6ff08b24a2c81), bootstrap kicked off in background.
- `/tmp/grpo-pod-pdp.env` points at the A100.
- Next session: wait for bootstrap_pod.sh to finish (~15 min), then `bash drive_iters_fast.sh --pod 36xpt4xbmezqtc --max-iters 50 --start-iter 10`.

### Best adapter so far
None. All 9 trained iters regress vs base model 0.942. Best is iter 2 (h2-strong-filter T=1.0) at 0.925 (-1.7pp). Base model remains the strongest baseline.
