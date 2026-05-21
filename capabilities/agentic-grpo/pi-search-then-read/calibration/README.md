# calibration/ — pi-search-then-read rubric sanity fixtures

Hand-written rollouts. `rubric_sanity.py` asserts
min(good_composite) > max(bad_composite) + 0.2.

## What "good" looks like

A good rollout for a large file:
1. `grep -n <symbol> <file>` to locate the line.
2. `sed -n 'A,Bp' <file>` or `read --offset --limit` for a small window.
3. Cite `file:line` in the final answer.

For small files (≤ 250 lines), reading whole is also OK — the rubric
awards full efficiency credit.

See `good.jsonl` for 5 examples across size tiers.

## What "bad" looks like

| §0 cheat | bad.jsonl id | Why it scores low |
|----------|--------------|-------------------|
| Read whole large file | `calib_bad_read_whole_large` | efficiency=0, no_search → composite ~0.30 |
| No search at all | `calib_bad_no_search` | no search, no citation → composite ~0.09 |
| grep an irrelevant keyword then read whole | `calib_bad_irrelevant_grep` | search_before_read=0 (symbol mismatch) |
| Found it but no file:line citation | `calib_bad_no_citation` | format=0 → composite=0 |
| 3x identical reads | `calib_bad_redundant_reads` | no_redundant_reads=0.33 |

## Tuning the small-file threshold

`SMALL_FILE_THRESHOLD = 250` in `../rubric.py`. Files at or below this
get full search_efficiency and search_before_read credit (small files
don't need search). Adjust if your eval corpus has different size
characteristics.

## Refreshing

After changing `../rubric.py`, run `python3 ../rubric_sanity.py`.

## Current calibration state

  good min=0.95, max=1.00
  bad  min=0.00, max=0.46 (calibrated to keep all cheats < 0.5)
  separation: +0.49
