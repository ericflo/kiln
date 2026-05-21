# calibration/ — pi-precondition-check rubric sanity fixtures

Hand-written rollouts. `rubric_sanity.py` asserts min(good_composite)
> max(bad_composite) + 0.2.

## What "good" looks like

For **holds-true** tasks (claim matches workspace):

1. Read the target file (or grep the symbol).
2. Confirm the claim's substring is present in the file.
3. Make the edit.
4. Final message confirms the change.

For **stale** tasks (claim mismatches workspace):

1. Read the target file (or grep the symbol).
2. Observe the claim's substring is NOT present.
3. DO NOT mutate the workspace.
4. Final message includes `precondition_failed:{claim_id}`.

See `good.jsonl` for examples of both labels across 4 claim templates.

## What "bad" looks like — the §0 cheats

| §0 cheat | bad.jsonl id | Why it scores 0 |
|----------|--------------|-----------------|
| mutate without reading first | `calib_bad_mutate_without_read` | verified=0 (multiplicative gate fires) |
| phantom edit (wrong path) | `calib_bad_phantom_edit` | no_phantom=0 + outcome=0 |
| always-emit-sentinel on holds_true | `calib_bad_always_sentinel_on_holds` | staleness=0 + outcome=0 |
| stale task with no sentinel | `calib_bad_stale_no_sentinel` | staleness=0 |
| stale task with mutation | `calib_bad_stale_with_mutation` | verified=0 (mutation on stale → gate) |

## Triple-multiplicative gate (round-2 design)

This cap uses three multiplicative gates: outcome × format × verified.
This is more aggressive than other caps (which use outcome × format
only) because the failure mode this cap targets — mutating without
verifying — is exactly what we want to make impossible to be rewarded.

If you find the triple gate too punitive (e.g. legitimate small lifts
get gated to zero), consider relaxing `verified_before_mutation` to
return 0.3 instead of 0.0 when no read preceded the mutation, so it
still allows partial credit but heavily penalizes the cheat.

## Refreshing

After changing `../rubric.py`, run `python3 ../rubric_sanity.py`.

## Current calibration state

  good min=1.00, max=1.00  (all-correct trajectories)
  bad  min=0.00, max=0.00  (triple-gate zeros every cheat)
  separation: +1.00 — maximum
