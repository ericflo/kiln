# calibration/ — rubric sanity fixtures

Round 2 mandates this directory for every cap. `rubric_sanity.py`
reads:

- `good.jsonl` — known-high-quality rollouts (the agent did the right
  thing). At least 5.
- `bad.jsonl` — known-low-quality rollouts including each §0 cheat
  named in `../capability.md`. At least 5.

The rubric must score the good set above the bad set with separation
> 0.2 (configurable via `RUBRIC_SANITY_MARGIN`). `run_iter.sh` runs
the sanity gate BEFORE training, so a broken rubric never reaches
the GPU.

## How to write a calibration fixture

Each line is one JSON object with the same shape `rubric.score_one()`
expects (or `score_rollout(transcript, workdir, task)` for legacy caps
— in that case the line should be `{"transcript": [...], "workdir":
"...", "task": {...}}`).

### Good fixture template

```json
{"task": {...}, "transcript": [...], "workdir": "..."}
```

Where the transcript shows the agent:
- reading appropriate context
- making the right action
- verifying the result
- summarizing cleanly

### Bad fixture template — one per §0 cheat

For each cheat enumerated in `../capability.md ## Adversarial design (§0)`,
write a fixture where the agent executes that cheat. Score should be 0
or near-zero. This is the round-2 anti-saturation discipline.
