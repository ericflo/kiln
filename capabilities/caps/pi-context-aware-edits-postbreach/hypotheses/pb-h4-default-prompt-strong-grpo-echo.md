# PB-H4: Default-Prompt Strong Agentic GRPO/ECHO

## Hypothesis

PB-H2 showed that default-prompt train rollouts contain usable reward variance
and preserve outcome better than the strict PB-H3 prompt, but the low-rank,
low-lambda update was too weak and let format regress. PB-H3 showed the
stronger recipe can move format, but the strict prompt damages outcome. Reusing
the clean PB-H2 default-prompt train rollout distribution with the stronger
`pi-faithful-completion` hyperparameters should test whether PB-H2 failed
because the update was underpowered rather than because the data was wrong.

## Recipe

- Reuse `/tmp/pi-context-aware-edits-postbreach-pb-h2-train-rollouts/grpo-train.jsonl`.
- Do not use the strict PB-H3 prompt config for training or eval.
- Trainer: `cuda_grpo_ablation`, `rank=16`, `alpha=32`, `lr=3e-5`,
  `echo_lambda=0.05`, `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during eval through the server default.

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
