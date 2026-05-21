# pi-diff-patch-apply — agentic patch application with verify/repair

**Status:** v1 rubric + corpus + rollout harness landed. **Rank 4/10.**
The agentic complement to OPD's `diff-patch-fluency` (single-shot diff
*generation*); this cap trains *application* in a closed loop with tests.

**Goal.** Given a working directory and a unified diff (sometimes clean,
sometimes with offset drift, sometimes encoding a subtle bug), the agent
applies the patch, runs the tests, and — if there are rejects or test
failures — repairs the patch minimally and re-runs until clean or the
turn budget is exhausted.

## Why this capability

Three clouderic patterns:

1. **Rewrite-whole-function.** Agents who haven't internalized "apply
   this patch, don't redesign" replace large blocks when a 3-line edit
   was needed. Breaks adjacent tests and produces noisy PRs.
2. **Surrender-on-reject.** When a patch fails to apply (offset drift,
   conflicting hunk), the 4B base tends to abandon the task or re-apply
   the same patch identically.
3. **Apply-without-test.** Patch lands; CI breaks downstream. The
   in-loop test step is missing.

This cap is the in-loop, end-to-end cousin of OPD diff-patch-fluency.
The OPD cap teaches the model to emit clean diffs; this cap teaches it
to *wield* diffs against a real workspace.

## Task shape

Each task carries everything the rubric needs to score a rollout:

```
{
  "task_id":            "task_NNNN",
  "patch_class":        "clean" | "drift" | "incorrect",
  "module_name":        e.g. "addition",
  "init_files":         {path -> content},          # initial state
  "gold_files":         {path -> content},          # reference solution
  "patch_text":         "...unified diff...",       # what pi sees
  "gold_diff_text":     "...the canonical diff...",
  "gold_diff_lines":    N,                          # for minimality
  "touched_paths":      ["src/addition.py"],        # files the patch is allowed to modify
  "protected_paths":    ["tests/test_addition.py"], # files pi MUST NOT modify
  "expected_repair":    true | false,               # true unless patch_class == "clean"
  "verify_cmd":         "python3 -m pytest -q tests/",
  "verify_timeout_s":   60,
}
```

Corpus mix (from `capability.config.json`):

| Class | Fraction | What's done to the diff | Pi must... |
|-------|---------:|-------------------------|------------|
| `clean` | 50% | none — gold diff as-is | `git apply` succeeds; run tests; declare done |
| `drift` | 30% | hunk @@ headers shifted by ±2-3 lines | `git apply` may reject; recount or apply by hand |
| `incorrect` | 20% | one `+` line mutated (e.g. `+` → `-`) | applies cleanly but tests fail; pi must repair |

## Pi prompt template

```
[system] You are a meticulous coding assistant operating in a sandboxed
workspace. You have access to bash, write, read, and edit tools. Apply
patches minimally — do NOT rewrite functions when a 3-line edit will do.
Always verify with the test command before declaring done.

[user] Apply the unified diff in `INCOMING_PATCH` to this workspace,
then run `python3 -m pytest -q tests/` until all tests pass.

Steps:
1. Read INCOMING_PATCH to see what change is intended.
2. Apply it (try `git apply INCOMING_PATCH` or `patch -p1 < INCOMING_PATCH`).
3. Run the tests with the verify command.
4. If a hunk rejects, read the affected source file, find the right
   anchor, and apply the change minimally. [REPAIR-HINT if expected_repair=true]
5. When all tests pass, emit a final message summarizing what you
   changed and which files you touched. Do NOT modify the test files.
```

Turn budget: **12 turns**, **240 s** wall-clock per rollout.

## Reward function (v1 — multi-component)

| Sub-score | Weight (pass) | What it measures | Cannot be cheated by |
|-----------|---------------|------------------|----------------------|
| `outcome` | hard gate | All tests pass after re-running in a fresh-checkout that restores test files. | Disabling tests — the oracle restores `protected_paths` before verifying. |
| `minimality` | 0.20 | `1 - clip(max(0, final_diff_lines - gold_diff_lines) / (span · gold_diff_lines), 0, 1)`. `span = 2`. | Writing the whole file fresh blows up final_diff_lines. |
| `no_unrelated_edits` | 0.15 | 1.0 iff `touched_workspace_paths ⊆ patch_touched_paths`. | Mutating an extra file penalizes; touching tests collapses score. |
| `repair_efficiency` | 0.10 | 1.0 if `failed_apply_attempts ≤ 2`. Decays 0.25 per extra attempt. Calibrated by `expected_repair`. | One-shot apply on clean patches scores 1.0 trivially. |
| `format_compliance` | 0.05 | Final assistant message: completion marker + reference to a touched file + 'summary'. | Empty final message → 0. |
| `tested_before_done` | discount | If `outcome=1` but no test command before the final assistant turn, **halve** the agentic budget. | "lucked into passing without testing" is suspicious. |

**Composite when `outcome = 1`:**

```
composite = 0.50 (base)
          + 0.20 · minimality
          + 0.15 · no_unrelated_edits
          + 0.10 · repair_efficiency
          + 0.05 · format_compliance
        (multiplied by 0.5 on the agentic-sub-score part
         if tested_before_done == 0)
```

Range: `[0.50, 1.00]`.

**Consolation when `outcome = 0` (the "all-zeros gradient" mitigation):**

```
composite = min(0.40,
    0.20 · applied_fraction      # dice/F1 of model-final vs gold-final
  + 0.10 · tested_before_done    # tried verifying at all?
  + 0.05 · format_compliance
  + 0.05 · no_loop)              # 1.0 - duplicate-tool-call rate
```

Range: `[0.00, 0.40]`.

The 0.40 consolation cap is **strictly less** than the 0.50 pass-floor —
the model can never beat a passing rollout by gaming consolation
signals. But consolation gives GRPO enough gradient on failed rollouts
to learn directional progress (apply some hunks, run pytest at all)
when the base model's pass rate is low.

### Adversarial design (§0)

**Q: cheapest 1.0?** Reset workspace, write the gold-final file fresh
without using `git apply`.
*Mitigation:* `no_unrelated_edits` catches any extra touched file (tests,
configs, irrelevant src); `minimality` catches an oversize diff if the
model wrote 50 lines fresh when gold was 5 lines. Writing JUST the gold
src/foo.py is permitted — but that IS the right outcome.

**Q: disable failing tests.** Mutate `tests/test_*.py` to `assert True`.
*Mitigation:* the oracle restores every `protected_paths` file from
`init_files` into a fresh-checkout shadow workdir *before* running the
verifier. Mutations to test files are erased. Additionally,
`no_unrelated_edits` punishes touching test files when the patch did not
intend to.

**Q: stop on yellow.** Apply the patch, claim done without running the tests.
*Mitigation:* `outcome` is a hard gate (re-runs the verifier); if the
patch was incorrect (the 20% incorrect-hunk class), no testing means no
repair → outcome=0 → composite at most 0.40 consolation.

**Q: loop forever on `ls`.** `no_loop` penalizes duplicate tool calls;
turn budget (12) hard-caps; wall-clock budget (240s) caps wall time.

**Q: read the gold solution from outside the workdir.** Pi runs with
`--no-context-files --no-extensions --no-skills --no-themes --offline`;
the only files visible to the model are inside `workdir/`, which contains
only the buggy init + `INCOMING_PATCH` + `README.md`. No gold files
leak into the rollout.

**Q: respond fully but never call a tool.** `applied_fraction` = 0 (no
file mutation) and `tested_before_done` = 0, so consolation tops out
at ≈ 0.05 (format).

### Headroom (estimated; to be measured at iter 0 baseline)

- Baseline composite ~0.20–0.35 expected. The 4B base can run `bash`
  and emit text but rarely repairs a rejected hunk and frequently
  rewrites whole functions.
- Target sub-score (highest movable mass): expected to be
  `applied_fraction` or `outcome` itself — the consolation signal is the
  early-learning anchor.

## ECHO recipe

**Strong fit.** Every patch application and test invocation produces
env output (`git apply` rejects, `pytest` output) that is highly
predictable from the patch + workspace. ECHO loss on those env tokens
trains the model's "what will happen if I run this" prior — exactly
the planning skill that distinguishes a careful patch-applier from a
chainsaw.

ECHO default: λ=0.05, env-only mask, warning_filter=on.

## Files

| File | Purpose |
|------|---------|
| `rubric.py` | Multi-component composite scorer. Reads transcript + workdir + task. |
| `task_scaffold.py` | `init_workdir` + `pi_prompt` + `build_messages`. |
| `build_corpus.py` | Generates `datasets/{train,eval}.tasks.jsonl` from 25 primitives × {clean, drift, incorrect}. |
| `rollout.py` | Per-rollout pi runner + GRPO-group + summary emitter. |
| `rubric_sanity.py` | Calibration test: 3 good (≥0.80) + 5 bad (≤0.30) synthetic transcripts. |
| `capability.oracle.sh` | Blind eval wrapper: prints `SCORE=<mean_composite>`. |
| `run_iter.sh` | Full iter recipe: rollouts → filter → train → eval → B2. |
| `backup_to_b2.py` | Per-iter B2 backup keyed by date + iter + kind. |
| `capability.config.json` | Trainer + rollout defaults (ECHO on, rank 16, lr 1e-5). |
| `capability.jsonl` | Append-only iter log (one JSON line per iter). |

## Hypothesis families to explore

| Slug | Family | Idea |
|------|--------|------|
| `h1-default-recipe` | baseline | iter 1 default recipe (ECHO on, rank 16, lr 1e-5, 16 tasks × 4 gens) to anchor |
| `h2-strong-signal-filter` | filter | only train on groups with var > 0.05 |
| `h3-temperature-bump` | sample | temperature 1.0 instead of 0.8 — more diversity |
| `h4-more-gens` | sample | 8 generations per task instead of 4 |
| `h5-lower-lr` | optim | lr 5e-6 — reduce overfitting |
| `h6-higher-lr` | optim | lr 2e-5 — see if cap can take it |
| `h7-rank-32` | optim | rank 32 — more capacity |
| `h8-no-echo` | ablation | disable ECHO to confirm it's helping |
| `h9-higher-echo-lambda` | ablation | echo-lambda 0.10 (just inside paper's productive band) |
| `h10-2epochs` | optim | 2 epochs on the filtered set |
| `h11-clean-only` | mix | train only on clean patches first |
| `h12-drift-only` | mix | train only on drift patches |
| `h13-incorrect-only` | mix | train only on incorrect-hunk patches |
| `h14-warmstart` | sequence | continue from a previous strong adapter |
| `h15-more-turns` | budget | 16 turns instead of 12 |
| `h16-fewer-turns` | budget | 8 turns — force decisiveness |
| `h17-tighter-minimality` | rubric | span=1 — stricter minimality penalty |
| `h18-stronger-fmt` | rubric | 0.10 format weight |
| `h19-system-anchor` | prompt | nudge system prompt toward "use git apply --reject" |
| `h20-corpus-expand` | data | regenerate corpus with 100 train tasks for variety |
| `h21-dr-grpo-vs-grpo` | optim | toggle advantage_mode |
| `h22-token-vs-seq-loss` | optim | toggle loss aggregation |
| `h23-cold-restart` | optim | reset training adapter mid-loop to re-seed exploration |
| `h24-on-policy-rollouts` | sample | regenerate rollouts every iter (vs replaying) |
| `h25-larger-batch` | optim | grpo-train-strong with 30 groups |
| `h26-pass-amplify` | filter | only train on groups where at least 1 gen outcome=1 |
| `h27-fail-amplify` | filter | only train on groups where at least 1 gen outcome=0 |
| `h28-mixed-task-mix` | data | regenerate with 70/20/10 (more clean → easier wins) |
| `h29-harder-mix` | data | regenerate with 30/40/30 (more drift+incorrect) |
| `h30-three-way` | data | enable `git apply --3way` in the prompt — bonus task families |
| `h31-no-policy-loss` | ablation | ECHO-only (paper §5.5 verifier-free) |
| `h32-larger-lr-warmup` | optim | start lr at 1e-5, warm down to 5e-6 |
| `h33-kl-tighten` | optim | kl_coeff 0.05 (less anchored to base) |
| `h34-kl-loosen` | optim | kl_coeff 0.2 (more anchored to base) |
| `h35-clip-narrow` | optim | clip_epsilon 0.10 |
| `h36-clip-wide` | optim | clip_epsilon 0.30 |
| `h37-tokens-bump` | budget | 4096 max tokens per turn |
| `h38-no-readme` | prompt | drop the README hint — see if it matters |
| `h39-test-amplify` | rubric | bump tested_before_done weight to discount-multiplier x2 |
| `h40-applied-amplify` | rubric | bump applied_fraction consolation weight |
| `h41-seed-sweep` | seed | re-run best recipe with 3 different seeds for variance |
| `h42-iter-many-tasks` | data | 40 train tasks × 4 gens |
| `h43-iter-few-many-gens` | data | 8 train tasks × 12 gens |
| `h44-class-stratified` | data | filter to exactly equal task class mix |
| `h45-cumulative-best` | sequence | warm-start from best-so-far each iter |
| `h46-low-temp-train` | sample | temperature 0.5 for training rollouts |
| `h47-format-strict` | rubric | format requires a markdown summary table |
| `h48-anti-loop` | rubric | bump no_loop weight |
| `h49-rank-8` | optim | rank 8 — see if smaller LoRA is enough |
| `h50-final-best-replay` | replay | replay best recipe with fresh rollouts for verification |

## Hypothesis log

(Populated by `capability.jsonl` — see that file for the canonical history.)

## References

- `capabilities/opd/diff-patch-fluency/` — OPD sibling.
- `capabilities/agentic-grpo/pi-doctest/` — closest agentic-GRPO sibling.
- `capabilities/agentic-grpo/pi-compaction/` — backup_to_b2 and drive_iters template.
- `crates/kiln-train/src/echo.rs` — ECHO loss term.
- `docs/plans/echo-integration-plan.md` §3.1, §3.3.


## Round 2 setup

This cap was normalized to the round-2 layout on 2026-05-21. The previous
iter log and writeups are preserved in [`archive/`](archive/). The
`capability.jsonl` starts empty for the new round.

### Kiln features the new round uses

- `kiln adapter verify` (#4) — adapter loadability + behavioral check.
- `cuda_*` trainer `--install-adapter-dir` / `--install-adapter-name` (#5) —
  atomic install into the registry; no more `output/adapter/` symlink bugs.
- `train_receipt.json` (#8) — the canonical per-run artifact with kiln SHA,
  data hashes, hyperparameters, LoRA delta norms, and ECHO metrics.
- `cuda_grpo_ablation --dry-run` (#9) — pre-GPU validation of data, masks,
  base-adapter shape, and saturated-reward warnings.
- `kiln trajectory inspect` (#10) — Rust-native mask + token-count
  diagnostic; replaces the Python `lib/pi_trajectory.py` for new code.
- ECHO observability in receipt (#12) — env-token CE, action-token count,
  warning-prefix masked-out byte count.
- `kiln serve --eval-mode` (#15) — deterministic, no thinking, no
  per-request adapter drift.
- `--adapter-smoke-test` (#19) — post-train base-vs-adapter logit-delta check.
- `--filter-var-min` (#22) — official strong-signal filtering.
- `kiln eval-adapter --seeds N` (#33) — multi-seed paired-eval driver wrapped
  by `capability.oracle.sh`.
- `adapter_manifest.json` + `kiln adapter restore` (#36) — replaces ad-hoc B2
  backup scripts.

### Workflow

```bash
./capability.oracle.sh                     # baseline (no adapter)
./run_iter.sh h1-default-recipe            # first training iter
./run_iter.sh h2-lower-lr                  # subsequent
```

See [`run_iter.sh`](run_iter.sh) for the full pipeline.
