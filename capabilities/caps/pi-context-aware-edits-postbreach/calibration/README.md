# calibration/ — pi-context-aware-edits rubric sanity fixtures

Hand-written rollouts that exercise the rubric without running pi.
`rubric_sanity.py` reads `good.jsonl` (≥5) and `bad.jsonl` (≥5) and
asserts: min(good_composite) > max(bad_composite) + 0.2.

## What "good" looks like

A good rollout:
1. Reads the target file BEFORE editing.
2. Adds the new function/class matching the existing style:
   naming case, type annotations, logging idiom, error handling,
   docstring style, import grouping.
3. Final summary names the file and the convention preserved.

See `good.jsonl` for examples in Python (strict-typed and camel-loose),
Rust (Result-returning), and minimal style-only cases.

## What "bad" looks like — the §0 cheats

| §0 cheat | bad.jsonl id | Why it scores low |
|----------|--------------|-------------------|
| wrong naming case | `calib_bad_wrong_case` | convention_consistency on naming_case = 0 |
| wrong logging idiom | `calib_bad_wrong_logging` | switched to print in a logging-based file |
| no read before edit | (any bad without read) | read_before_edit = 0 |
| style drift (no docstring) | `calib_bad_style_drift` | no_style_drift = 0 |
| redundant import | `calib_bad_redundant_import` | no_redundant_imports < 1 |
| outcome passes but conventions violated | `calib_bad_conv_violations_outcome_passes` | process sub-scores low (~0.37) |

The last case is important: it proves the rubric distinguishes process
quality even when outcome+format both pass. If you remove it, the
rubric appears to rely solely on outcome.

## Refreshing

After changing `../rubric.py`, run `python3 ../rubric_sanity.py`.

## Current calibration state

  good min=0.95, max=1.00 (clean read-then-edit-with-style cases)
  bad  min=0.00, max=0.37 (constraint: bad case can pass outcome+format
                            but still score < 0.4 due to process violations)
  separation: +0.58 — strong
