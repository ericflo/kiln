# pi-diff-patch-apply — agentic patch application with verify/repair

**Status:** Scaffold. **Rank 4/10**. The agentic complement to OPD's
`diff-patch-fluency` (which trains single-shot diff *generation*); this
cap trains *application* in a closed loop with tests.

**Goal.** Given a working directory and a unified diff (possibly from
an upstream PR, a teacher rollout, or a bug-fix candidate), the agent
applies the patch, runs the tests, and — if there are rejects or test
failures — repairs the patch minimally and re-runs until clean or the
turn budget is exhausted.

## Why this capability

Three clouderic patterns:

1. **Rewrite-whole-function.** Agents who haven't internalized
   "apply this patch, don't redesign" replace large blocks when a
   3-line edit was needed. This breaks adjacent tests and produces noisy
   PRs.
2. **Surrender-on-reject.** When a patch fails to apply (offset drift,
   conflicting hunk), the 4B base tends to abandon the task or re-apply
   the same patch identically.
3. **Apply-without-test.** Patch lands; CI breaks downstream. The
   in-loop test step is missing.

This cap is the in-loop, end-to-end cousin of OPD diff-patch-fluency.
The OPD cap teaches the model to emit clean diffs; this cap teaches it
to wield diffs against a real workspace.

## Task shape

Each task is `(workspace, patch, tests, gold_outcome)`:

- **Workspace** — small Python or Rust project (tens of files).
- **Patch** — unified diff, sometimes valid as-is, sometimes with hunks
  whose offsets drifted by ±N lines, sometimes with one hunk that has
  to be manually repaired.
- **Tests** — `pytest -q tests/` or `cargo test -q --package X` — must
  pass after the patch is applied.
- **Gold outcome** — `{tests_pass: bool, final_diff_lines: N, expected_repair: bool}`.

Example prompt:

```
Apply the patch in /tmp/incoming.patch to the workspace, then run
`pytest -q tests/` until all tests pass. If a hunk fails to apply,
repair it minimally; do not rewrite unrelated code.
```

Correct trajectory:

1. `bash`: `git apply --check /tmp/incoming.patch` (probe applicability).
2. If clean: `git apply /tmp/incoming.patch`; if not: read the rejected
   hunk, find the right anchor, repair, retry.
3. `bash`: `pytest -q tests/`.
4. If green: emit final assistant turn with the diff summary.
5. If red: read the failure, narrow to the cause inside the patch
   scope, repair, retry.

Turn budget: 8 turns; bench-style tasks may need 12.

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | All tests pass at the end of the session. | Disabling tests, `pytest --collect-only`, `pytest -k 'nothing'` → 0 (oracle re-runs without overrides). |
| `minimality` | 0.20 | `1 - clip((final_diff_lines - gold_diff_lines)/(span · gold_diff_lines), 0, 1)`. `span = 2`. | A 1-line diff that fails tests → outcome 0 → composite 0. |
| `no_unrelated_edits` | 0.15 | 1.0 iff no files outside the patch's touched set were modified. | Read-only inspection is fine; mutating an extra file penalizes. |
| `repair_efficiency` | 0.10 | For tasks requiring repair: 1.0 if ≤2 repair attempts. Decays for >2. | One-shot apply on clean patches scores 1.0 trivially — gated to `expected_repair=true` tasks. |
| `format_compliance` | 0.05 | Final assistant turn includes a summary of the applied diff (path + line count) in a stable structure. | Free-form prose without structure → 0. |

**Composite** = `outcome × (0.20·minimality + 0.15·no_unrelated + 0.10·repair_eff + 0.05·format + 0.50·base)`

## ECHO recipe

**Strong fit.** Every patch application and test invocation produces
env output (`git apply` rejects, `pytest` output) that is highly
predictable from the patch + workspace. ECHO loss on those env tokens
trains the model's "what will happen if I run this" prior — exactly
the planning skill that distinguishes a careful patch-applier from a
chainsaw.

**Hypothesis worth trying:** task families where the patch is
*incorrect in subtle ways* (off-by-one, missing import) and the agent
must catch it via test output. These tasks benefit most from ECHO
because the test output is dense env signal.

## Hypotheses

- **H_offset_drift** — synth patches with offsets shifted by ±3 lines.
  Tests whether the agent uses `--reject` and fixes hunks vs gives up.
- **H_incorrect_patch** — patches that apply cleanly but break tests.
  Tests in-loop repair vs blind apply.
- **H_unrelated_passing_tests** — workspaces with already-failing tests
  outside the patch scope. Tests whether the agent gets distracted vs
  stays on scope.
- **H_three_way_merge** — provide `--3way` and a clean ancestor.
  Bonus task family.

## Adversarial design (§0)

**Q: cheapest 1.0?** Reset workspace, write the gold-final file, run
tests. Mitigation: `no_unrelated_edits` checks that touched files
match the patch's file set; a full rewrite changes more files (the
patch typically touches 1–3 files, gold-final touches all).

**Q: disable failing tests.** Mitigation: oracle re-runs the test suite
without the agent's mutations to the test files, in a fresh checkout
that re-applies *only* the patch the model produced.

**Q: stop on yellow.** Mitigation: rubric requires `outcome=1` (all
tests pass). No partial credit for "applied but didn't run."

## Headroom (estimated)

- Baseline composite ~0.35–0.50. The 4B model can apply clean patches
  but loses badly on reject-repair and on staying within scope.
- Target sub-score: `repair_efficiency` (highest movable mass — and the
  hardest sub-skill).

## Files to create

- [ ] `rubric.py`. The `no_unrelated_edits` check needs a clean
  baseline diff between workspace before & after; `git stash` +
  `git diff` is the natural primitive.
- [ ] `task_scaffold.py`. Source patches from kiln's own git history
  (`git log -p` against curated commits with clear unit-test
  validation); apply offset-drift and incorrect-hunk transforms
  programmatically.
- [ ] `rollout.py` — run pi against a fresh workspace each turn (copy
  the seed workspace into a per-rollout tmpdir).
- [ ] `build_corpus.py`, `capability.oracle.sh`, `run_iter.sh`.
- [ ] `calibration/{good,bad}.jsonl` — good: apply-then-test, clean.
  Bad: rewrite-the-file, disable-tests.

## Next steps for the agent picking this up

1. Read the shared README and `capabilities/opd/diff-patch-fluency/capability.md`.
   Note: Python + Rust toolchains are expensive to set up per rollout;
   pre-bake a tmpfs snapshot that each rollout copies on write before
   you start iter 0.
2. Pick a Python and a Rust source repo. Recommended: kiln itself (Rust
   side) plus a small Python project (`uv` + pytest).
3. Build the corpus: mine 30 commits from each repo with clear test
   coverage; transform 30% with offset drift, 20% with incorrect
   hunks, leave 50% clean.
4. Sandbox setup costs matter a lot here. Time `git clone + cargo
   build` ahead of writing `rollout.py`; if it's >30s per rollout,
   build a snapshot-and-copy strategy first.
5. Iter 0 baseline on 24 eval tasks (8 clean, 8 offset-drift, 8
   incorrect).
6. Iter 1 with ECHO defaults. Watch `env_token_ce_holdout` carefully —
   the test output is the env signal that matters most here.

## References

- `capabilities/opd/diff-patch-fluency/` — OPD sibling.
- `crates/kiln-train/src/echo.rs` — env-CE term used in training.
- `docs/plans/echo-integration-plan.md` §3.1.
