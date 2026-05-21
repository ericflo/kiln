# calibration/ — pi-diff-patch-apply

**ROUND-2 LIMITATION**: this cap's rubric.score_rollout reads
**workdir disk state** (outcome runs verify_cmd in workdir (already has v2 multiplicative gate)). Calibration fixtures can't be
synthesized in JSONL alone — they need a materialized workdir with
the right gold files / failing tests / etc.

`rubric_sanity.py` is bypassed in `run_iter.sh` via
`KILN_SKIP_RUBRIC_SANITY=1` for this cap. The next agent on a real
pod should:

1. Run base eval against `datasets/eval.tasks.jsonl`.
2. Save 5 high-scoring rollouts (composite ~1.0) as `calibration/good.jsonl`
   with their associated workdir state.
3. Save 5 §0-cheat rollouts (composite ~0) as `calibration/bad.jsonl`.
4. Remove the `KILN_SKIP_RUBRIC_SANITY=1` line from `run_iter.sh`.

Each fixture entry needs:
```json
{
  "transcript": [...pi session events...],
  "workdir": "/path/to/materialized/workdir",
  "task": { ...task spec... }
}
```

Or use the workdir-builder pattern: include `_workdir_files` in the
fixture and have a `setup_fixture.py` script that materializes them
into a tmp dir before running rubric_sanity.

## §0 cheats to populate in bad.jsonl

(See `../capability.md` for the full §0 list.)

For now: the calibration is documented but unpopulated. This cap's
rubric is exercised live during real iter runs.
