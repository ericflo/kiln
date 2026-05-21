# calibration/ — pi-error-recovery rubric sanity fixtures

This dir contains hand-written rollouts that exercise the rubric without
running pi. `rubric_sanity.py` reads `good.jsonl` (≥5) and `bad.jsonl`
(≥5) and asserts:

  min(good_composite) > max(bad_composite) + 0.2

If sanity fails, `run_iter.sh` blocks the iter from starting — you've
got a rubric problem, not a pod problem.

## What "good" looks like for pi-error-recovery

A good rollout demonstrates the recovery loop:

1. First tool call fails (file_not_found / permission_denied / syntax_error
   / command_not_found / dependency_missing / timeout).
2. An assistant text turn acknowledges the failure and names the recovery.
3. A *different* tool call appropriate to the error class is issued.
4. The new call succeeds (or makes meaningful progress).
5. Final assistant message names what failed, what the recovery was, and
   confirms it worked.

See `good.jsonl` for one example of each error class.

## What "bad" looks like — the §0 cheats

The `bad.jsonl` file must include at least one fixture per §0 cheat
from `../capability.md`:

| §0 cheat | bad.jsonl id | Why it must score low |
|----------|--------------|----------------------|
| A1 (always-ls-on-fail) | `calib_bad_constant_ls` | ls is wrong for syntax_error; recovery_appropriate scores 0 |
| A2 (long apology, no fix) | `calib_bad_apology_only` | outcome_passed=false → composite=0 |
| A3 (recovery spam) | `calib_bad_spam_recovery` | no_loop decays as duplicates pile up |
| (loop) | `calib_bad_loop` | exact-repeat failed call; no_loop=0 |
| (give-up) | `calib_bad_giveup` | no retry call; recovery_was_different=0 |

If you find a new cheat the rubric doesn't catch, add it here AND
tighten the rubric.

## Refreshing the calibration

If you change the rubric weights or sub-scores in `../rubric.py`,
re-run `python3 ../rubric_sanity.py` and confirm separation stays
above 0.2. If it dips below, either the weights are wrong or the
calibration set needs new examples.

## Current calibration state

Per `python3 ../rubric_sanity.py` on the committed rubric:

  good min=0.30, max=0.87  (mean ~0.65)
  bad  min=0.00, max=0.00  (all gated to zero by outcome=0)
  separation: +0.30 (clean — well above 0.2 margin)
