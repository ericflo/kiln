# calibration/ — pi-incremental-progress rubric sanity fixtures

Hand-written rollouts that exercise the rubric without running pi.
`rubric_sanity.py` reads `good.jsonl` (≥5) and `bad.jsonl` (≥5) and
asserts: min(good_composite) > max(bad_composite) + 0.2.

## What "good" looks like for pi-incremental-progress

A good rollout shows the **edit → verify → edit → verify** loop:

1. First edit touches ONE file (no big-bang multi-file rewrite).
2. Immediately verified (`pytest`, `python -c import x`, etc.).
3. If verify failed, the next assistant turn acknowledges the failure
   and adjusts the next edit.
4. Next edit advances toward gold, verified again.
5. Final assistant message lists the steps taken, each referencing an
   actual tool call from the session.

See `good.jsonl` for examples across three task families:
extract-module, rename-symbol, add-validation.

## What "bad" looks like — the §0 cheats

| §0 cheat | bad.jsonl id | Why it scores low |
|----------|--------------|-------------------|
| big-bang first edit | `calib_bad_big_bang` | no_big_bang scores 0 on multi-file bash |
| no intermediate verifies | `calib_bad_no_intermediate_verify` | step_progress_observability=0 |
| fictional steps in summary | `calib_bad_fake_steps_summary` | format alignment with session keywords low |
| spam pytest between random edits | `calib_bad_spam_pytest` | alignment with gold tokens fails |
| ignored mid-flight failure | `calib_bad_ignored_failure` | early_failure_caught=0 |

## Refreshing

After changing `../rubric.py`, run `python3 ../rubric_sanity.py` and
verify separation stays above 0.2.

## Current calibration state

  good min=0.33, max=0.91 (mean ~0.6)
  bad  min=0.00, max=0.00 (all gated by outcome=false)
  separation: +0.33 — clean
