# PB-H7: Full-Train Conservative GRPO/ECHO

## Hypothesis

PB-H2/H5/H6 all used the small clean PB-H2 train rollout set, and each update
found a different unstable tradeoff between outcome, format, and convention.
Collecting rollouts over the full train task pool should reduce overfitting to
the small 16-group sample. A conservative GRPO/ECHO update can then use the
larger reward-variance pool to preserve convention while recovering the H5
outcome and format gains.

## Recipe

- Source: fresh default-prompt train-only rollouts from all 32 train tasks.
- Rollout sampling: 4 generations per task, no active adapter.
- Trainer: `cuda_grpo_ablation`, `mode=phase1`, `rank=8`, `alpha=16`,
  `lr=5e-6`, `echo_lambda=0.03`, `filter_var_min=0.05`.
- Keep `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during rollout collection and eval through the server
  default.

## Falsification

Reject if any of:

- fewer than 12 reward-variant train groups remain after filtering;
- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.95`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Pending.
