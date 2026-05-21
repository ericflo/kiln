# hard_eval.tasks.jsonl — pi-incremental-progress hard-eval pool

Tasks where the base 4B model is expected to produce a big-bang edit
instead of decomposing.

## How to build it for this cap

Brand-new cap; no round-1 archive. Build by:

1. Run base on the standard eval; mark tasks with `step_progress_observability=0`
   (the 4B's default failure mode).
2. Optionally hand-construct adversarial cases:
   - **5+ file refactor** with implicit dependencies (must verify
     between each move).
   - **Multi-validator** with 5+ validation rules (must verify each
     rule individually).
   - **Cross-module rename** where partial state breaks imports until
     all sites are updated — the agent must use a staged approach.
3. Write resulting tasks to `hard_eval.tasks.jsonl` (gitignored).
4. Run `TASKS=datasets/hard_eval.tasks.jsonl ./capability.oracle.sh`;
   confirm base composite < 0.4.

## Expected hard-eval lift

After training:

| task family | base | trained | lift |
|-------------|------|---------|------|
| extract-module (3-file) | ~0.50 | ~0.80 | +0.30 |
| rename-symbol (2-step) | ~0.55 | ~0.80 | +0.25 |
| add-validation (4-rule) | ~0.40 | ~0.75 | +0.35 |

This file is gitignored (blind-eval firewall).
