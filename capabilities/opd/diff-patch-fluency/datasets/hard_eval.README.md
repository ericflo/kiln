# hard_eval.tasks.jsonl — round-1-failures-derived eval pool

Round 2 introduces a hard-eval pool per cap. The pool is built from
tasks where the BASE model failed in round 1 (or that have been
hand-marked as hard during corpus design).

Format: same JSONL as `eval.tasks.jsonl`, with one additional field
`_hard_reason: <string>` explaining why the task was flagged.

This file is **gitignored** like `eval.tasks.jsonl` (blind-eval
firewall). It is not present in the committed tree until
`build_corpus.py` produces it from local data.

## How it's used

`capability.oracle.sh` accepts `TASKS=datasets/hard_eval.tasks.jsonl`
to score against the hard pool instead of the standard eval. Lift on
hard-eval is the cleanest evidence of capability uplift vs. lucky-tasks.

## How to build it

Round-1 caps that have committed `archive/` data may have hard-eval
candidates in there (failed_task IDs, regression sets). The next agent
picking up the cap should:

1. Inspect `archive/` for round-1 failed-task IDs.
2. Build `datasets/hard_eval.tasks.jsonl` from those IDs.
3. Run `./capability.oracle.sh` with `TASKS=datasets/hard_eval.tasks.jsonl`
   to confirm base composite < 0.5 on the hard pool.
4. Compare adapter performance on hard-eval vs standard eval.
