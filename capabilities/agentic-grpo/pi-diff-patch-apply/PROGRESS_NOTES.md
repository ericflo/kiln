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

(See `capability.jsonl` for canonical record. This table is updated as each iter completes — not just at the end.)

| Iter | Hypothesis                       | Composite | Δ      | Status                                                  |
|------|----------------------------------|-----------|--------|---------------------------------------------------------|
| 0    | baseline-v1-hard-corpus-strict   | 0.9419    | —      | baseline (24-task eval, T=0, strict 4-pillar format)    |
| 1    | h1-default 6×3 hard              | 0.8900    | −0.052 | NEGATIVE — 1/6 strong-signal groups; over-amplified one |
| 2    | h2-strong-filter T=1.0 6×3       | 0.9246    | −0.017 | NEGATIVE-mild — best trained iter so far                |
| 3    | h3-temp1-seed4242 6×3            | 0.9162    | −0.026 | NEGATIVE                                                |
| 4    | h4-lower-lr-5e-6 6×3             | 0.9109    | −0.031 | NEGATIVE                                                |
| 5    | h5-higher-lr-2e-5 6×3            | 0.2165    | −0.725 | CATASTROPHIC (or infra failure — kiln serve crashed)    |
| 6    | h6-lower-lr-2e-6 FAILED-no-server| 0.20      | −0.742 | INVALID — kiln serve crashed pre-eval                   |
| 7    | h7-very-low-lr-2e-6 6×3          | 0.2165    | −0.725 | CATASTROPHIC (same pattern as iter 5 — infra suspect)   |
| 8    | h8-no-echo 8×3                   | 0.8400    | −0.102 | NEGATIVE — first iter after pi-setup + pytest fixes     |
| 9    | h9-echo-0.10 8×3                 | 0.7623    | −0.180 | NEGATIVE — higher ECHO over-anchored, drift hit worst   |
| 10   | h10-echo-0.02 8×3                | —         | —      | **in flight on A100 pod 36xpt4xbmezqtc**                |
| 11-49| (auto-chained by drive_iters_fast)| —        | —      | queued                                                  |

**Best trained adapter so far:** iter 2 at 0.9246 (−1.7pp). Base model at 0.9419 remains the strongest.

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

6. **A6000 → A100 80GB switch (2026-05-19 21:30Z).** First A6000 lease ran out after iter 9.
   Re-acquired an A100 80GB PCIe ($1.39/hr vs $0.49/hr on A6000). Fresh bootstrap on the
   A100 hit a new gap: `huggingface-cli download` is now a deprecated alias that silently
   no-ops without printing anything useful, leaving `/workspace/qwen3.5-4b/` empty.
   Switched bootstrap to `hf download` (the new tool) + added a post-download verify
   that exits 41 if no model files appear.

7. **The iter 10 → 49 silent cascade (2026-05-19 21:38Z–22:00Z).** Drive script ran
   through iters 10-49 in ~20 min but every single one was a no-op because kiln serve
   was dead after iter 10's GRPO step and never recovered. Cascade pattern: iter 10's
   rollouts succeeded (composite 0.685), then run_iter.sh's `pkill -9 kiln serve`
   ran (intentional — frees VRAM for GRPO), then something in the bg launch of
   `cuda_grpo_ablation` failed silently → `set -e` killed run_iter.sh → kiln serve
   never restarted → all subsequent iters got "Connection refused" on
   /v1/adapters. The drive script's strict-Bash interpolation hit
   `NameError: name 'null' is not defined` when `COMPOSITE` was the string `"null"`,
   which kept the failed rows out of capability.jsonl (silver lining).
   **Fix:** added a pre-iter kiln-serve health check that restarts the server if
   it's dead, and switched the failure literal from `"null"` to `"None"` so Python
   interpolation works.

## Kiln/infra fixes landed during this loop

(All committed to main, with rationale, so future cap authors and the next session don't repeat them.)

- `bootstrap_pod.sh`:
  - Install pytest (rollouts need it for the verify step).
  - Run `kiln pi-setup` so pi uses `provider=kiln-local` instead of bailing with
    "No API key found".
  - Build with `--features cuda` (default build is CPU-only, silently slow).
  - `KILN_CUDA_ARCHS=80` cuts kiln-flash-attn build time ~3x on A6000 (forward-PTX-compat).
  - Use `hf download` instead of deprecated `huggingface-cli download`.
  - Post-download verify that fails loudly when no model files appear.
- `run_iter.sh`:
  - Kiln-serve health check (5×15s polls) after restart, ensures the new adapter
    appears in `/v1/adapters` registry before proceeding.
  - Smoke-test rollout (1 task) before full eval so adapter/infra failures fail fast.
- `drive_iters_fast.sh`:
  - Dropped `set -e` so a single iter failure doesn't kill the 50-iter loop.
  - Pre-iter kiln-serve health check + auto-restart (prevents cascades like iter 10→49).
  - Per-cap env file `/tmp/grpo-pod-pdp.env` (the shared `/tmp/grpo-pod.env` was
    getting clobbered by concurrent caps on the same dev box).
  - Use Python literal `None` not `null` for missing composite (NameError otherwise).
  - Enriched capability.jsonl row with sub-scores, class means, rollouts_passed.
- `backup_to_b2.py`:
  - Auto-installs boto3 to `/tmp/pylibs` instead of bailing out with an error.
- `kiln-polish.jsonl`:
  - 6 polish notes appended documenting each of the gaps above.

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
