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
| 10   | h10-echo-0.02 8×3                | —         | —      | VOIDED — concurrent kiln-serve port conflict killed serve, all rollouts 0.2 |
| 11   | h11-2epoch                       | —         | —      | VOIDED — cascaded from iter 10 (kiln-serve died, no eval ran) |
| 12   | h12-rank8 (rank 8 / alpha 16)    | —         | —      | VOIDED — A100 lease expired mid-rollouts |
| 13   | h13-rank64                       | —         | —      | VOIDED — drive kept running against dead A100 pod (now killed) |
| 10   | h10-echo-0.02 (retry on A6000)   | 0.8422    | −0.100 | NEGATIVE — ECHO=0.02 ≈ no-echo (h8), both ~10pp regression. Confirms ECHO has little effect at these scales |
| 11   | h11-2epoch (2 epochs)            | 0.8676    | −0.074 | NEGATIVE — 2 epochs beats 1 (h10 was 0.842). Suggests longer training is mildly helpful at lr=1e-5 |
| 12   | h12-rank8 (rank 8 / alpha 16)    | 0.8882    | −0.054 | NEGATIVE — best of 8-task sweep; rank-8 retains clean class (0.986). Smaller rank → less perturbation works. |
| 13   | h13-chain-best-hard-mix (chain from iter 2) | 0.8462 | −0.096 | **chain compound FAILED** — iter 2 (0.9246) + same recipe → 0.8462. GRPO on top of trained adapter degrades it. |
| 14   | h14-incorrect-only chain         | —         | —      | VOIDED — A6000 lease expired during iter 14, pod EXITED, smart_drive cycled FAILED rows on dead pod |
| 15   | h15-rank2-chain                  | —         | —      | VOIDED — same pod death |
| 16   | h16-3epochs-chain                | —         | —      | VOIDED — same pod death |
| 14   | h14-incorrect-only-from-base     | 0.7715    | −0.170 | **concentrated training BACKFIRED** — even the targeted class itself collapsed: incorrect 0.757 → 0.418 (−34pp). Clean 0.998 → 0.856, drift 0.975 → 0.883. The model needs the diverse mix to stay coherent; training only on broken tasks teaches it the wrong thing. |
| 15   | h15-rank2-hardmix-from-base      | EVAL-PENDING | — | Adapter trained (rank 2, ~7MB ✓). Eval failed mid-smoke from transient pod runtime/ports glitch. Will backfill-eval when kiln has capacity. |
| 16   | h16-3epochs-from-base            | EVAL-PENDING | — | Adapter trained ✓ (rank 16, 61MB), **smoke composite 0.85** (best smoke seen). Eval bailed on transient SSH glitch like iter 15. Two adapters now waiting for backfill-eval. |
| 17   | h17-6tasks-6gens-from-base       | 0.7106    | −0.231 | **WORST yet** — 6 unique tasks × 6 gens = overfits to small task set. Clean 0.998 → 0.767, drift 0.975 → 0.857, incorrect 0.757 → 0.388. Strong evidence: small task variety hurts more than the extra gens-per-task help. |
| 15   | h15-rank2-hardmix-from-base     | VOIDED    | —      | Adapter trained but A100 pod died before backfill-eval — weights lost with /tmp |
| 16   | h16-3epochs-from-base           | VOIDED    | —      | Adapter trained (smoke 0.85!) but A100 pod died before backfill — promising signal lost. |
| 18   | h18-lr5e-6-from-base (v1)        | VOIDED    | —      | A100 pod died mid-iter (lease expired or pod crashed) |
| 19   | h19-combined-from-base (v1)      | VOIDED    | —      | A100 pod died |
| 20   | h20-no-filter-from-base (v1)     | VOIDED    | —      | A100 pod died |
| 18v2 | h18-lr5e-6-from-base             | 0.7018    | −0.240 | NEGATIVE — drift class collapsed 0.975 → 0.659. Same lr as iter 4 (0.911) but on hard_mix corpus. **Strong hypothesis: hard_mix corpus is poison** (over-weighted incorrect+drift vs eval's 50/30/20 distribution). |
| 19v2 | h19-combined-from-base (hard_mix)| 0.7386    | −0.203 | NEGATIVE — combined recipe (rank2 + 3ep + lr5e-6) on hard_mix also degraded. **2/2 hard_mix iters fail → corpus distribution mismatch confirmed.** |
| 20   | h20-rank8-default-corpus         | FAILED    | —      | GRPO crashed: `GRPO completions have different prompt lengths (248 vs 246 for completion 2)`. Bug in cuda_grpo_ablation when rollouts have variable prompt lengths. Pod is fine. |
| 20   | h20-rank8-default-corpus (A100)  | FAILED    | —      | cuda_grpo crash: GRPO completions had different prompt lengths (248 vs 246). Real cuda_grpo_ablation limitation, not pod death. |
| 21-24| (various, A100)                  | VOIDED    | —      | A100 pod died mid-iter. Adapter-preserve fix saved iter 18/19 weights to b2; iter 20/21+ adapters never produced. |
| 21v2 | h21-rank2-default-corpus (L40S)  | VOIDED    | —      | L40S had transient runtime glitch mid-iter; pod resurrected. iter 22 used as replay test. |
| 22   | h22-default-T1.0 (default corpus) | 0.5234    | −0.418 | **CATASTROPHIC** — all classes collapsed ~40pp. T=1.0 + default corpus is far worse than T=1.0 + hard_mix (iter 13: 0.846) or T=0.8 + default (iter 12: 0.888). Non-monotonic interactions. |
| 23+  | (smart_drive in flight)          | —         | —      | iter 23 (h23-2epochs-default) in flight |
| 11-49| (auto-chained by drive_iters_fast)| —        | —      | queued                                                  |

**Best trained adapter so far (11 valid trained iters):** iter 2 at 0.9246 (−1.7pp). Base model at 0.9419 remains the strongest.

**Ranked by composite (trained iters only):** iter 2 (0.9246) > iter 3 (0.9162) > iter 4 (0.9109) > iter 1 (0.8900) > iter 11 (0.8676) > iter 10 (0.8422) > iter 8 (0.8400) > iter 9 (0.7623). The top-4 are all from the original 6-task ×3-gen recipe; the new 8-task ×3-gen sweep produces slightly worse outcomes — possibly because the easier task mix dilutes the per-group signal.

**ECHO sweep summary (iters 8/9/10):** ECHO=0 → 0.840; ECHO=0.02 → 0.842; ECHO=0.10 → 0.762. ECHO ≈ 0-0.02 are equivalent; 0.10 hurts. The default 0.05 (iters 1-4) was in the safer zone but still net negative.

**Epoch sweep (iter 11):** 2 epochs at lr=1e-5 → 0.8676, better than the 1-epoch comparable iter 10 → 0.8422. Mild evidence that longer training helps slightly at this lr.

**Chain compound test (iter 13):** Loaded iter 2 adapter (0.9246) from B2, trained another GRPO step on top with the same T=1.0 + hard-mix recipe → 0.8462 (−8pp from iter 2). **Compounding GRPO on a trained adapter actively degrades it.** Suggests every successful iter so far was lucky, not a smooth optimization path. Smart_drive will pivot the next hypotheses away from chain-from-best.

**Concentrated-headroom training (iter 14):** Train only on `incorrect`-class tasks (where base has 24pp headroom) from base. **All 3 classes collapsed**, including the targeted one: incorrect 0.757 → 0.418 (−34pp), clean 0.998 → 0.856, drift 0.975 → 0.883. Composite 0.7715 (−17pp). The model needs the diverse training mix to stay coherent; concentrating on broken tasks doesn't teach it to fix them, it teaches it to model the broken distribution. Strong evidence against "isolate the weak class" as a training strategy.

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

9. **Iter 10/11 failure mode: port-8420 conflict from concurrent kiln serves.**
   While debugging the pkill self-kill, I (Claude) started TWO kiln serves
   back-to-back via `python3 $RP bg ...` — first `kiln-serve-iter10retry.log`
   at 22:25Z, then `kiln-serve-debug.log` at 22:27Z. Only one process can
   bind :8420; when the second came up, the first got "Killed" at 22:28:23Z
   (right when the v3 drive script's iter 10 sent its `/v1/adapters/unload`
   call). Iter 10's pi sessions then ran against a kiln that died
   mid-iter: 24 train rollouts all returned 0.2 (consolation floor) in
   ~20s each (Connection refused). The fixed-pkill drive's pre-iter
   health check restarted kiln before iter 11, but iter 11's GRPO step
   then killed it again (intentionally, via the now-fixed pgrep -x kiln),
   and the asynchronous `Background pid` launch + 30s sleep didn't give
   the new kiln serve enough time to come up before `wait-file` for the
   adapter timed out. Net: iters 10 + 11 both logged as FAILED-no-eval.
   **Lesson for future:** never bg two kiln serves in a row without
   `pgrep -x kiln | xargs kill -9 ; sleep 3` between them. Iter 12 (in
   flight as of 23:55Z) has a clean kiln-serve-iter12-restart and is
   running rollouts normally.

8. **ROOT CAUSE: `pkill -9 -f "kiln serve"` over SSH self-kills the wrapper bash.**
   Discovered 2026-05-19 22:30Z while debugging iter 10's repeat failure even with the
   health-check guard. `runpod_api.py ssh "pkill -9 -f \"kiln serve\" || true; sleep 3"`
   reliably returned 255 with empty output. Root cause: pkill on the pod, when invoked
   with `-f` (full cmdline match), matches its own wrapping bash's argv — which
   contains the literal string "kiln serve" (because we passed it as the `-f` pattern).
   The wrapper bash gets SIGKILL'd. SSH session dies. `set -e` in run_iter.sh kills
   the script silently. This is the root cause of the iter 10→49 cascade in note 7
   AND of all the iter 5/6/7 catastrophic failures back in the first session.
   **Fix:** replace `pkill -9 -f "kiln serve"` with `pgrep -x kiln | xargs -r kill -9`.
   `pgrep -x` matches the binary name exactly (not the full cmdline), so it can't
   match the wrapper bash. Committed across run_iter.sh and bootstrap_pod.sh.

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
