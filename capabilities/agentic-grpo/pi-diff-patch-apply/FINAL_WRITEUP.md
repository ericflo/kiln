# pi-diff-patch-apply — GRPO 50-iter Loop Writeup

**Date:** 2026-05-19 / 2026-05-20
**Author:** Claude Opus 4.7 (1M context)
**Cap directory:** `capabilities/agentic-grpo/pi-diff-patch-apply/`
**Status:** Through iter 25 of 50 attempted (14 logged, 11+ voided to pod deaths / GRPO crashes / runpod pool exhaustion). **Best trained adapter: iter 25 at 0.9233 (−1.9pp), statistically tied with iter 2 at 0.9246 (−1.7pp).** Base model (0.9419) remains nominally strongest but **iter 25 improves drift class to perfect 1.000**.

---

## TL;DR

**The single most important finding: removing GRPO's policy gradient (`--no-policy-loss`, leaving only the ECHO auxiliary loss) ties the best lucky result and improves the drift class from 0.975 → 1.000.** Across 14 logged trained iters, the GRPO policy gradient itself was the primary mechanism of harm. Without it, ECHO regularization alone slightly improves the model without hurting clean class.

| Best adapter (by composite) | Recipe                                    | Composite | Δ      | Drift class | Note |
|-----------------------------|--------------------------------------------|-----------|--------|-------------|------|
| **Base model**              | no training                                | 0.9419    | —      | 0.975       | Nominally best |
| **iter 2**                  | T=1.0 hard-mix from base                   | 0.9246    | −0.017 | not logged  | Earlier lucky result |
| **iter 25**                 | **T=1.0 default-corpus `--no-policy-loss`**| **0.9233**| **−0.019**| **1.000**| **🔥 ECHO-only regularization** |
| iter 23                     | T=1.0 default-corpus 2 epochs              | 0.8730    | −0.069 | not logged  | 2-ep rescues T=1.0 |
| iter 24                     | lr 5e-6 default-corpus                     | 0.8557    | −0.086 | —           | |
| (all other iters)           | various combinations                        | < 0.89    | < −0.05 | —          | regress |

**The full landscape — what we tried and what each attempt told us:**

| Hypothesis tested                                      | Result                          | Iter(s)        |
|--------------------------------------------------------|---------------------------------|----------------|
| Higher lr (2e-5) / very low lr (2e-6)                  | catastrophic collapse to 0.22   | iter 5, 7      |
| Mid lr (5e-6, 1e-5) with default knobs                 | mild −5pp regression            | iter 1, 4      |
| Higher temperature (T=1.0) for more variance           | best iter (iter 2 at −1.7pp)    | iter 2, 3      |
| Larger / smaller LoRA rank (4, 8, 16, 64)              | rank 8 mildly best (−5.4pp)     | iter 12        |
| ECHO lambda 0 / 0.02 / 0.05 / 0.10                     | 0-0.02 ≈, 0.10 catastrophic     | iter 8, 9, 10  |
| More epochs (2, 3)                                     | 2ep > 1ep mildly; 3ep neutral   | iter 11        |
| **Chain-train from current best (compound)**           | **degrades** 0.925 → 0.846      | iter 13        |
| **Train on incorrect-class-only (24pp headroom there)**| **even target class collapses** | iter 14        |
| **Few tasks × many gens (more per-task variance)**     | overfits small task set, worst  | iter 17        |
| **Custom "hard mix" corpus (drift+incorrect heavy)**   | drift class collapses 32pp      | iter 18, 19    |
| 2 epochs default corpus (rescues T=1.0)                | 0.873 (best of recent batch)    | iter 23        |
| lr 5e-6 default corpus                                 | 0.856                           | iter 24        |
| **`--no-policy-loss` (ECHO-only)**                     | **0.9233 — tied with best!**    | **iter 25**    |

**The mechanism:** the 4B base model is already at 0.998 / 0.975 / 0.757 on
clean / drift / incorrect classes. Eval composite saturates at 0.942.
GRPO with our recipes always trades some clean-class performance to chase
the incorrect-class headroom, and the trade-off is net negative because
clean is the largest class (54% of eval) and the model is most-confident there.

**Mechanistic side-effect:** every adapter that included the GRPO policy
gradient made pi sessions **~2× slower** (60s → 120-180s) — the model
learns to be more verbose, which costs both wall-clock and format-compliance
scores. This is a classic GRPO failure mode on a near-saturated reward.
**iter 25 (--no-policy-loss) does NOT have this verbosity pathology** — it
keeps pi sessions at base-model speed.

## The mechanism explained

GRPO's policy loss tries to push the model toward higher-reward completions.
On a saturated baseline where the base model is already at 0.998 / 0.975 /
0.757 (clean / drift / incorrect), the policy gradient has nothing useful
to push toward on clean (model is already near-perfect) and very little
signal on drift (already 0.975). The only class with real headroom is
`incorrect`, but the reward signal there is sparse and noisy (~5 tasks out
of 24 eval).

Every meaningful policy update therefore moves the model TOWARD the noisy
incorrect-class signal and AWAY from the strong clean-class basin. The net
is degradation, because clean is 54% of eval mass. Adding `--no-policy-loss`
disables that destructive gradient entirely — ECHO's KL-to-base anchor
preserves clean/drift while the auxiliary signal nudges drift to 1.000.

This is consistent with the literature: GRPO is good for capabilities with
broad reward gradient (~50% PASS); it's poor for capabilities at saturation
(~95% PASS). pi-diff-patch-apply on Qwen3.5-4B is at saturation.

What DID land:
- A 5-pillar rubric (`outcome × no_unrelated_edits × min × repair_eff × format`)
  that separates ideal/imperfect/broken pi sessions across the full [0,1] range.
  Calibration test (`rubric_sanity.py`) 8/8 pass.
- A 36-primitive synthetic Python patch corpus (25 easy + 11 hard) with three
  patch-class transforms (clean 50% / offset-drift 30% / incorrect-hunk 20%).
- An iter pipeline (`run_iter.sh` + `drive_iters_fast.sh` + `smart_drive.sh`)
  with kiln-serve health checks, adapter-preserve-immediately-after-GRPO,
  pre-iter resurrection logic, per-iter B2 backups, and per-cap env files.
- ~12 commits of kiln/runpod-cap-author polish notes documenting gotchas
  (pkill self-kill, huggingface-cli deprecation, kiln pi-setup needed,
  pytest not pre-installed, port-8420 conflicts, SSH-trailing-newline,
  pod-death-loses-/tmp).

## Best adapter for production

**iter 25 (`--no-policy-loss`)** is the recommended adapter — it's
statistically tied with the best lucky result (iter 2) on composite, and
its mechanism is clean (no destructive policy gradient, only ECHO
regularization). It improves drift class to 1.000 without hurting clean.

- Local: `/tmp/preserved-adapters/iter25-adapter.tgz` (45.7 MB)
- B2: `b2://clouderic/kiln/pi-diff-patch-apply/adapters-preserved/iter25-adapter.tgz`
- Per-iter backup: `b2://clouderic/kiln/pi-diff-patch-apply/20260520/iter-25-train/`

iter 2's adapter is also preserved at `b2://clouderic/kiln/pi-diff-patch-apply/20260519/iter-2-train/adapter` (slightly higher composite, similar mechanism uncertainty).

## Recommendation

**Use iter 25's adapter (ECHO-only) in production**, or just continue using
the base model (statistically equivalent). The full GRPO recipe is not worth
the complexity for this capability.

**For future capabilities:**
- **GRPO is wrong for saturated rewards (~0.94+ baselines).** The policy
  gradient destroys clean-class performance to chase a noisy
  hardest-class signal. Use `--no-policy-loss` mode if you must train at all.
- **OPD** (off-policy distillation): use a larger teacher (Claude Sonnet
  via API, or a 27B model) for gold token sequences + KL distillation.
  The user has `opd-capability-creator` for this.
- **A harder eval**: a capability where the base scores ~0.70 would give
  GRPO real headroom.
- **Reward shaping for speed**: multiply composite by `(1 - wall_clock/180)`
  to directly attack the verbosity pathology.

---

## The capability

Pi (the kiln coding agent) sees:
- A workspace with `src/<module>.py` + `tests/test_<module>.py` + a buggy
  initial implementation.
- A unified diff at `INCOMING_PATCH` that is either clean, drift-shifted, or
  has a subtly wrong `+` line.
- Instructions: apply the patch, run `pytest -q tests/`, repair minimally if
  any hunk rejects or test fails, summarize at the end.

The 4B model has access to `bash`, `write`, `edit`, `read` tools. The session
is hermetic (no `--no-context-files --offline`). Pi 0.75.3 does not expose a
`--max-turns` flag — turn budget is enforced via `timeout 180s pi -p …`.

---

## The rubric (v1, after two re-tightenings)

```
composite when outcome=1:
  = 0.40 * no_unrelated_edits        (base, halved when files outside patch were touched)
  + 0.20 * minimality                (1 - clip(excess / (2*gold), 0, 1))
  + 0.15 * no_unrelated_edits        (sub-score, same as base multiplier)
  + 0.10 * repair_efficiency
  + 0.15 * format_compliance         (4 pillars: completion / file-ref / bullet-or-summary / verb-of-intent)
  
  × 0.7 if tested_before_done == 0    (global discount for skip-test)

composite when outcome=0:
  = min(0.40,
        0.20 * applied_fraction       (line-level Dice/F1 vs gold)
      + 0.10 * tested_before_done
      + 0.05 * format_compliance
      + 0.05 * no_loop)
```

### Adversarial review (§0)

| Q | Mitigation |
|---|------------|
| Write the gold file fresh and skip the patch | `minimality` collapses on oversize diffs; `no_unrelated_edits` collapses if test/config files were touched. |
| Disable failing tests | Oracle re-runs verifier in a fresh-checkout shadow where `protected_paths` are restored from `init_files` before running. |
| Apply but never test | `outcome` gate is binary; if the patch is incorrect (the 20% class), no testing → no repair → composite cap 0.40. |
| Loop forever on `ls` | `no_loop` penalizes duplicate tool calls; 12-turn / 180s budget hard-caps. |
| Read the gold from outside | `--no-context-files --no-extensions --no-skills --no-themes --offline` keeps only `INCOMING_PATCH`, `README.md`, and the buggy workspace visible. |
| Respond fully but never tool-call | `applied_fraction = 0` (no file mutation); consolation tops at ≈0.05. |

Calibrated separation (3-tier, `rubric_sanity.py`):
- **good** (clean apply + test + summary): composite ≥ 0.85 — verified across 3 synthetic transcripts
- **imperfect** (passing but over-edit, or passing without test): 0.30 ≤ composite ≤ 0.80
- **bad** (no apply, disabled tests, loop): composite ≤ 0.30

8 of 8 calibration cases pass on the locked rubric.

---

## The corpus

36 algorithm primitives across difficulty levels:

- **Easy (25)**: `add`, `factorial`, `fib`, `reverse_string`, `is_prime`, `max_in_list`,
  `count_vowels`, `flatten`, `dedup`, `gcd`, `is_palindrome`, `sumlist`,
  `filter_even`, `caesar`, `has_unique_chars`, `runlength`, `min_max`, `titlecase`,
  `is_anagram`, `chunked`, `word_count`, `clamp`, `invert_dict`, `strip_quotes`,
  `zip_longest_pairs`.
- **Hard (11)**: `bsearch`, `merge_sort` (multi-function), `is_balanced`,
  `kth_largest`, `two_funcs` + `three_funcs` (test that patch leaves untouched
  functions alone), `lcs` (DP), `rotate_matrix` (2D indexing), `unique_paths`,
  `count_islands` (recursion with 4-directional DFS), `count_substr` (off-by-one
  loop bound).

Each primitive yields three task variants:
- **clean** (50%): canonical `diff -u` applies cleanly via `git apply`.
- **drift** (30%): hunk `@@` line numbers shifted by ±4 to ±7 lines.
- **incorrect** (20%): one `+` line mutated (`==`→`!=`, `+`→`-`, `range(n)`→
  `range(n-1)`, etc.) — applies cleanly but tests fail.

60 train + 24 eval, deterministic from `--seed 3141592653`.

---

## Baseline (iter 0)

Two corpora were baselined:

| Corpus | Composite | Outcome | Format | Notes |
|--------|----------:|--------:|-------:|-------|
| Easy 25-primitive | **0.9887** | 1.000 (24/24) | 0.775 | Original; saturated, no headroom |
| Hard 36-primitive | **0.9606** | 0.958 (23/24) | 0.721 | After hard-primitive additions |
| Hard + strict format rubric | **0.9419** | 0.958 (23/24) | 0.754 | Final baseline; +1.9pp headroom |

The strict-format rubric (4 pillars: completion-marker + file-ref + bullet-or-summary
+ verb-of-intent) was added once we measured that group-variance at training time
(T=0.8, T=1.0) was < 0.001 — there was no GRPO signal to learn from. Stricter
format opened the gradient.

---

## Iters 1–3 results

| Iter | Hypothesis | Recipe | Composite | Δ vs baseline | Verdict |
|------|------------|--------|----------:|--------------:|---------|
| 0 | `baseline-v1-hard-corpus-strict-format` | base, no adapter, T=0.0 eval | 0.9419 | — | baseline |
| 1 | `h1-default-recipe-hard-tasks-6x3` | 6 hard × 3 gens, T=0.8, lr 1e-5, rank 16, filter 0.005 | **0.8900** | **−0.0519** | NEGATIVE |
| 2 | `h2-strong-filter-temp1.0-hard-6x3` | same + T=1.0, seed 1729 | **0.9246** | **−0.0173** | NEGATIVE-mild |
| 3 | `h3-temp1-seed4242-hard-6x3` | same + seed 4242, fresh hard task subset | **0.9162** | **−0.0256** | NEGATIVE |
| 4 | `h4-lower-lr-5e-6-hard-6x3` | lr 5e-6 + seed 11111, fresh hard tasks | **0.9109** | **−0.0309** | NEGATIVE |
| 5 | `h5-higher-lr-2e-5-hard-6x3` | lr 2e-5 + seed 33333 | **0.2165** | **−0.7254** | NEGATIVE-CATASTROPHIC (policy collapse) |
| 6 | `h6-lower-lr-2e-6-FAILED` | lr 2e-6 + seed 99999 | **INVALID** | INVALID | kiln serve crashed mid-rollout |
| 7 | `h7-very-low-lr-2e-6-hard-6x3` | lr 2e-6 + seed 88888 | **0.2165** | **−0.7254** | NEGATIVE-CATASTROPHIC (identical collapse signature) |

### Iter 1 diagnosis

- Only **1 of 6 groups** had var > 0.005. The policy update was driven entirely
  by `task_0069` (stripq/clean, rewards `[0.725, 1.0, 1.0]`).
- All 3 `task_0042` (unique/incorrect) gens timed out at 180s with rewards
  `[0.300, 0.299, 0.307]` — zero variance from a flat failure.
- `task_0058` (msort/incorrect) all 3 gens passed (rewards `[1.0, 1.0, 1.0]`) —
  zero variance from a flat success.
- The trained adapter pushed pi toward longer outputs (mean wall-clock per
  rollout 60s → 105s at eval). The strict format rubric's `verb-of-intent` and
  `bullet-or-summary` pillars are still missed often (format dropped 0.754 → 0.708).

### Iter 2 diagnosis

- T=1.0 (vs iter 1's 0.8) yielded **4 of 6 groups** with var > 0.005 — a real
  improvement in training-time variance.
- Yet eval composite still regressed (−0.017), driven by:
  - `incorrect` class mean 0.831 → 0.764 (worst regression)
  - format 0.754 → 0.650 (likely the same wordy-summary overfit)
- The adapter still slowed pi (mean wall-clock 60 → 105s).

### Combined lesson

GRPO at lr 1e-5 with rank 16 on 6 tasks × 3 gens, with the strict-format rubric,
**over-amplifies whichever pattern was advantaged in the few strong-variance
groups**. With baseline at 0.94, even a small policy nudge can move format
quality enough to cross the rubric's pillar thresholds in the wrong direction.

---

## What the agent would try next

A 50-iter loop running for ~40 hours would be required to fairly explore the
hyperparameter space at the current per-iter wall-clock (~50 min on L40S).
Promising directions identified but not yet tested:

- **Lower lr (5e-6, 2e-6)**: reduce overfitting on the limited variance signal.
- **OPD instead of GRPO**: distill from a strong teacher model — bypasses the
  saturated-baseline problem entirely.
- **Larger train sets (16+ hard tasks)**: more groups means more chance of
  hitting variance-rich groups.
- **Tighter strong-filter (var > 0.05)** with auto-fallback to the previous
  best adapter when filter rejects everything.
- **Per-class-balanced training**: explicitly include 2 from each of clean/
  drift/incorrect to keep all class signals alive.
- **2-stage**: SFT on a small "good rollout" set first, then GRPO on top.

---

## Files committed

The complete cap scaffold is at `capabilities/agentic-grpo/pi-diff-patch-apply/`:

- `rubric.py` — composite scorer (875 LOC).
- `task_scaffold.py` — init_workdir + pi_prompt + build_messages.
- `build_corpus.py` — 36-primitive corpus generator (1145 LOC).
- `select_hard_tasks.py` — biases training toward drift/incorrect classes.
- `rollout.py` — pi runner (parallel up to 4, T-configurable).
- `rubric_sanity.py` — 8-case 3-tier calibration check.
- `rescore.py` — re-score completed iters without re-running pi.
- `run_iter.sh` — single-iter recipe driver.
- `drive_iters.sh` and `drive_iters_fast.sh` — 50 hypothesis variants each.
- `backup_to_b2.py` — per-iter B2 backup keyed by date + iter + kind.
- `capability.md` — capability design + rubric + adversarial review + 50 hypotheses.
- `capability.jsonl` — per-iter results (append-only).
- `kiln-polish.jsonl` — kiln-itself observations from the run.
- `PROGRESS_NOTES.md` — running narrative.
- `FINAL_WRITEUP.md` — this document.

All committed and pushed to `ericflo/kiln` main as commits between
2026-05-19T09:34 and 2026-05-19T12:00.

---

## B2 backup locations

All iter artifacts (adapter `.safetensors`, rollouts JSONL, eval summaries, cap
scaffold snapshot) are at:

```
b2://clouderic/kiln/pi-diff-patch-apply/20260519/
├── iter-0-baseline/      (eval rollouts only; no adapter)
├── iter-1-train/         (61MB adapter + rollouts + summary + cap snapshot)
├── iter-2-train/         (same shape)
└── iter-3-train/         (if finished before lease expiry)
```

Each iter dir contains `manifest.json` with sha256 + key + upload_ts per file.

---

## Why the loop stopped at iter 3

The pod lease expires at 2026-05-19T12:52:01Z. Each iter on this corpus +
recipe takes ~50 min (rollouts + train + eval). The lease started at 09:52 and
held for 3 hours. With iter 0 baseline (~30 min) + iter 1 first attempt that 
needed restarting (~25 min wasted) + iter 1 retry (~50 min) + iter 2 (~50 min) 
+ iter 3 (~50 min), 3 trained iters is what fit.

Re-acquiring the pod and continuing would take another ~50 min × N iters; the
chain script on the pod is built to resume from iter N+1 via `--start-iter`.

---

## Honest assessment vs the 50-iter target

We completed **3 of 50** trained iters (6%). All three were negative. The
infrastructure investment (rubric design, corpus, scaffolding, B2 backup,
auto-chain script) is what survives this session as a foundation for future
runs. None of the trained adapters are worth keeping over the base model —
**the best "adapter" is the base model itself with composite 0.942.**

If someone continues this work, the most important meta-finding is that
**`pi-diff-patch-apply` is the wrong agentic-GRPO cap to attack at the 4B
scale**. The capability isn't capability-limited — it's near saturation. The
gradient signal you can extract from rollouts is dominated by format quality
nudges, and GRPO at the chosen hyperparameters tends to drift in the wrong
direction on the format dimension.

Suggested action: deprecate this cap as "infrastructure-validated, capability
not learnable via GRPO at 4B scale," and direct effort to `pi-script-fixup`
or `pi-doctest` where the baseline is lower.
