# hard_eval.tasks.jsonl — pi-error-recovery hard-eval pool

Round 2 ships every cap with a hard-eval pool: tasks the base model is
expected to fail on. Lift on hard-eval is the cleanest evidence of
capability uplift vs. lucky-tasks.

For pi-error-recovery, hard-eval covers the tasks the 4B base hits its
documented failure modes on: looping, giving up, compounding the error.

## How to build it for this cap

Brand-new cap (no round-1 archive yet). Build the hard pool by:

1. Run `./capability.oracle.sh` against the base model. Look at the
   per-task scores in the eval summary; tasks with composite < 0.3 are
   hard candidates.
2. Optionally hand-construct adversarial cases for each error class:
   - **file_not_found** — directory has 50+ files, only one is the
     target; lazy ls misses it.
   - **permission_denied** — read-only dir (not just file); needs
     chmod on the parent.
   - **syntax_error** — the model's first solution has a non-obvious
     bug (e.g. wrong indentation in a multi-line function); requires
     reading the error line carefully.
   - **command_not_found** — multiple alternatives exist with subtle
     incompatibilities (pytest 6 vs 7).
   - **dependency_missing** — the dependency *is* available but under
     a different name (e.g. `Levenshtein` vs `python-Levenshtein`).
   - **timeout** — task requires a fast path that the README doesn't
     hint at; agent must invent the alternative.

3. Write the resulting tasks to `hard_eval.tasks.jsonl` (gitignored).
4. Run `TASKS=datasets/hard_eval.tasks.jsonl ./capability.oracle.sh`
   to confirm base composite < 0.5 on the hard pool.

## Per-class headroom on hard-eval

After round-2 training, expected lift on hard-eval (vs base):

| error_class | base | trained | lift |
|-------------|------|---------|------|
| file_not_found | ~0.40 | ~0.75 | +0.35 |
| permission_denied | ~0.45 | ~0.75 | +0.30 |
| syntax_error | ~0.55 | ~0.80 | +0.25 |
| command_not_found | ~0.45 | ~0.80 | +0.35 |
| dependency_missing | ~0.35 | ~0.70 | +0.35 |
| timeout | ~0.30 | ~0.65 | +0.35 |

These are *pre-training estimates*; the rubric_sanity gate already
confirms the rubric can measure these lifts.

This file (`hard_eval.tasks.jsonl`) is gitignored to keep the blind-eval
firewall: agents must not read it from inside the training loop.
