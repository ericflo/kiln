# pi-failure-triage — root cause, not symptom

**Status:** Scaffold. **Rank 8/10**. Trains the agent to reproduce a
failure, narrow it to a root cause, and propose a fix that the
*held-out* related test also passes — rather than papering over the
symptom.

**Goal.** Given a failing test or stack trace, the agent reproduces the
failure deterministically, isolates the root cause, and proposes a fix
that resolves both the visible test and at least one held-out related
test that depends on the same underlying behaviour.

## Why this capability

This is the failure mode behind many of the closed-null kiln PRs and
the "phase 12 LM-head OOM" cluster of re-attempts. The pattern:

1. Test fails.
2. Agent reads the stack, sees a `IndexError`.
3. Agent wraps the offending line in `try: ... except IndexError: pass`.
4. Test passes; merged; downstream test that depends on the *real*
   invariant breaks the next day.

Saved note `verify-bugs-before-fixing`: "When a task describes a bug
with a specific root cause and fix, always verify the bug actually
exists before [fixing it]." This cap goes further — the bug *does*
exist; the question is whether the fix addresses the *cause*.

## Task shape

Each task is `(workspace, failing_test, held_out_related_test,
gold_fix_region)`:

- **Workspace** — a small project with a real bug planted.
- **Failing test** — the visible failing test the agent is asked to
  fix.
- **Held-out related test** — *not shown to the agent*. Exists in a
  separate file `tests/held_out/`, runs as part of the oracle eval but
  not the in-loop check. Depends on the same root cause; passes only
  if the fix addresses the cause.
- **Gold fix region** — the file + symbol where the real fix should
  go. Used for `fix_localised_correctly` sub-score.

Example: the workspace has `lib/parser.py::parse_date()` that crashes on
ambiguous inputs. The visible test is `test_parse_date_iso8601` which
fails. The held-out test is `test_parse_date_european_locale`. A
symptom fix (`try/except` around the call site) makes the visible test
pass but the held-out test still fails. A root-cause fix (adding a
branch for ambiguous inputs in `parse_date`) makes both pass.

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | Visible test passes. | Disabling the test → 0 (oracle re-runs with the test file restored). |
| `held_out_passes` | 0.30 | The held-out related test also passes after the fix. | Symptom fixes that only patch the call site → 0 here. |
| `fix_localised_correctly` | 0.15 | 1.0 iff the fix touches the gold fix region; 0.5 iff it touches a neighbour file; 0 iff it touches only the test or unrelated code. | Touching everything → noise; only the right region counts. |
| `no_test_mutation` | 0.10 | 1.0 iff the agent did not modify any file under `tests/`. | Editing the test then "fixing" → 0. |
| `no_blanket_except` | 0.10 | Penalty `-0.5` for every new `except Exception` or `except:` introduced. Penalty `-0.3` per new `# noqa` / `# type: ignore` comment. | Renaming the broad-except — AST check on the diff catches it. |
| `format_compliance` | 0.05 | Final assistant turn includes a structured fix summary: `(file, function, root_cause_oneliner)`. | Free-form prose → 0. |

**Composite** = `outcome × (0.30·held_out + 0.15·localised + 0.10·no_test_mut + 0.10·no_blanket + 0.05·format + 0.30·base)`

The high `held_out_passes` weight is the **whole point**. If the fix
doesn't generalise, the cap shouldn't credit it.

## ECHO recipe

**Strong fit.** Stack traces and test failure output are rich,
predictable env signal. The skill we want is "predict what `pytest`
will print after my edit" — exactly what ECHO trains. When the agent
internalises that env model, symptom fixes become obviously inadequate
because the agent can predict the held-out test would still fail.

**Hypothesis:** for this cap specifically, the held-out test's output
is *not* in the trajectory (it's in the oracle), so ECHO can't train
on it directly. The training value of ECHO here is via the visible
test's output + the agent's own reproduction step. Consider running an
in-loop "shadow" test from the same conceptual cluster (not held-out)
to provide more env signal.

## Hypotheses

- **H_symptom_temptation** — bugs where wrapping in try/except *also*
  passes the held-out test (rare but exists, especially for genuinely
  flaky tests). Tests the rubric: should the model still get credit?
  Answer: yes, because the held-out outcome IS what we're measuring.
- **H_buggy_test_seeding** — 10% of tasks have a buggy *test*, not
  buggy code. The right fix is to fix the test. Tests whether the
  model can detect this. Subtle and probably v2+ work.
- **H_multi_test_root_cause** — fixes that have to pass two held-out
  tests with different fixtures. Tests deep generalisation.

## Adversarial design (§0)

**Q: disable the visible test.** Mitigation: `no_test_mutation` + the
oracle re-runs from a clean test directory.

**Q: try/except around the failing line.** Mitigation:
`held_out_passes` is the dominant weight; `no_blanket_except`
penalises the AST diff.

**Q: read the held-out test file.** It's outside the workspace given
to the agent; the harness mounts it only at oracle-time. Path check
in the rollout sandbox enforces.

**Q: deliberately overscope the fix to maximise coverage.**
Mitigation: `fix_localised_correctly` rewards touching the gold
region, not the whole subsystem.

## Headroom (estimated)

- Baseline composite ~0.20–0.35. The 4B base mostly produces symptom
  fixes — they pass the visible test, fail the held-out, and tend
  toward `except Exception`.
- Target sub-score: `held_out_passes` (the whole point).

## Files to create

- [ ] `rubric.py`. The `no_blanket_except` check requires AST diffing
  Python and Rust code separately; use `ast` for Python and
  `tree-sitter-rust` for Rust. Hand-roll the diff if tree-sitter is
  too heavy.
- [ ] `task_scaffold.py`. Bug planting: take a real-bug-fix commit
  from git history, revert the fix, then plant the bug in a clean
  branch. The original commit becomes the "gold fix region" reference.
  Curate ~50 bugs across Python and Rust.
- [ ] `rollout.py`. Sandboxes the workspace so `tests/held_out/` is
  not visible to pi.
- [ ] `build_corpus.py`, `capability.oracle.sh`, `run_iter.sh`.
- [ ] `calibration/{good,bad}.jsonl` — good: root-cause fix that
  passes held-out. Bad: try/except symptom fix; test-only edit.

**Infra note.** Bug-planted workspaces are larger than other caps'
rollouts — config sets `max_wall_clock_s = 300`.

## Next steps for the agent picking this up

1. Read shared README, the precondition-check cap, and the
   diff-patch-apply cap (sibling skills).
2. Bug-planting is the hard part. Mine 20 commits from kiln itself
   that fix a bug with a clear unit test; revert each in a separate
   branch. That's your bootstrap corpus. Augment with public Python
   project commits for diversity.
3. Building the held-out test is non-trivial: it must depend on the
   same root cause but be sufficiently different that a try/except
   patch doesn't accidentally catch it. Hand-write these for v0; do
   not try to synthesise them.
4. Calibration: include literal `try/except: pass` symptom fixes as
   bad. If the rubric scores them >0.3 (just from outcome on the
   visible test), the held-out weight is too low.
5. Iter 0 baseline; iter 1 with ECHO defaults.

## References

- Saved note: `verify-bugs-before-fixing`.
- `pi-diff-patch-apply/capability.md` — sibling cap for applying
  validated fixes.
- Kiln PR #176 (closed null) — a real-world example of a "symptom"
  fix that didn't generalise; useful as a calibration negative.
