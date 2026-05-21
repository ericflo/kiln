# calibration/ — pi-test-interpretation rubric sanity fixtures

## What "good" looks like

A good rollout:
1. Runs test/bench at least 3 times (explicit calls, for-loop, or seq).
2. Reports the MEDIAN, explicitly (not mean, not first run).
3. Classifies flake vs real-fail when applicable.
4. Discounts the warmup run on bench tasks.

See `good.jsonl` for examples.

## What "bad" looks like

| §0 cheat | bad.jsonl id | Why it scores low |
|----------|--------------|-------------------|
| 1 run + claim done | `calib_bad_one_run` | iter=0 → outcome=0 |
| Report MEAN of 3 runs | `calib_bad_report_mean` | reported_median=0 (mean without median) |
| Ignored flake (1 run only) | `calib_bad_ignored_flake` | classified_flakes=0 |
| Warmup taken as real | `calib_bad_warmup_as_real` | iter=0 → outcome=0 |
| 2 runs + report mean | `calib_bad_two_runs_mean` | iter=0.5 + mean penalty |

## Refreshing

After changing `../rubric.py`, run `python3 ../rubric_sanity.py`.

## Current calibration state

  good min=0.85, max=1.00
  bad  min=0.00, max=0.00
  separation: +0.85 — strong
