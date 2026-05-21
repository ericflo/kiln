# calibration/ — pi-tool-call-efficiency calibration

NOTE: This cap is **eval-only** (round-2 reshape). The composite is
purely tool-call efficiency. Calibration fixtures here are simpler than
other caps — just count tool calls in known-good and known-bad
trajectories.

## What "good" looks like

| n_tool_calls | composite | bucket |
|--------------|-----------|--------|
| 0-4 | 1.00 | efficient |
| 5-9 | 0.87→0.37 | moderate |
| 10-12 | 0.25→0.00 | wasteful |

See `good.jsonl` for 5 examples (0, 1, 2, 3, 4 tool calls).
See `bad.jsonl` for 5 examples (≥10 tool calls).

## Why simpler than other caps

This cap doesn't train an adapter; its job is to *measure* tool-call
distributions across other caps' adapters. The rubric is essentially a
single sub-score (efficiency from n_tool_calls). The calibration
proves that bucket gradients are reasonable.

## Refreshing

After changing `../rubric.py`, run `python3 ../rubric_sanity.py`.

## Current calibration state

  good min=1.00, max=1.00 (0-4 calls)
  bad  min=0.00, max=0.25 (10-20 calls)
  separation: +0.75
