# PB-H3: Light Terminal-State Prompt + Stronger Agentic GRPO/ECHO

## Hypothesis

PB-H2 proved that real train rollouts contain usable reward variance, but the
low-rank, low-lambda update traded a small outcome gain for a final-format
regression. `pi-faithful-completion` found that a light explicit terminal-state
prompt, lower rollout noise, `lr=3e-5`, `rank=16`, `alpha=32`, and
`echo_lambda=0.05` can move terminal behavior without a strict prompt removing
all headroom. This cap should try that recipe while preserving the
read/edit/verify/final workflow.

## Recipe

- Use a postbreach-specific strict-workflow config that keeps the final response
  shape explicit while also asking the model to read, edit minimally, verify,
  and stop.
- Collect fresh train-only Pi rollouts from `datasets/train.tasks.jsonl`,
  bounded to 16 tasks x 3 generations for the first arm.
- Use the generated `grpo-train.jsonl` groups only if reward variance survives
  `--filter-var-min=0.05`.
- Trainer: `cuda_grpo_ablation`, `rank=16`, `alpha=32`, `lr=3e-5`,
  `echo_lambda=0.05`, `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
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
