# PB-H2: Train-Rollout Agentic GRPO + Light ECHO

## Hypothesis

PB-H1 showed that idealized SFT slightly lifts edit completion but damages the
final-response contract and efficiency. Training on the model's own train-only
Pi rollouts should preserve the distribution of real tool behavior while the
rubric supplies outcome/format pressure. A small ECHO-enabled GRPO arm should
lift composite by improving real rollout decisions instead of imitating
synthetic traces.

## Recipe

- Collect train-only Pi rollouts from `datasets/train.tasks.jsonl`, bounded to
  16 tasks × 3 generations for the first arm.
- Use the generated `grpo-train.jsonl` groups only if reward variance survives
  `--filter-var-min=0.05`.
- Trainer: `cuda_grpo_ablation`, `rank=4`, `alpha=8`, `lr=5e-6`,
  `echo_lambda=0.02`, `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during rollout collection.

## Falsification

Reject if any of:

- dry-run keeps fewer than 4 reward-varied groups;
- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.90`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Pending.
