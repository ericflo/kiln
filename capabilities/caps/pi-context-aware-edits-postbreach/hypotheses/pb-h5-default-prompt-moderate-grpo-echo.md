# PB-H5: Default-Prompt Moderate Agentic GRPO/ECHO

## Hypothesis

PB-H2 used the right default-prompt train rollout distribution and nudged
`outcome` upward, but the rank/lambda/lr were too weak to protect final format.
PB-H4 used the same distribution with a much stronger update and improved
format, but collapsed `outcome` and weakened convention consistency. An
intermediate update may preserve PB-H2's outcome behavior while recovering some
of PB-H4's terminal-format improvement.

## Recipe

- Reuse `/tmp/pi-context-aware-edits-postbreach-pb-h2-train-rollouts/grpo-train.jsonl`.
- Do not use the strict PB-H3 prompt config for training or eval.
- Trainer: `cuda_grpo_ablation`, `rank=8`, `alpha=16`, `lr=1e-5`,
  `echo_lambda=0.03`, `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep `--filter-var-min 0.05`.
- Keep thinking enabled during eval through the server default.

## Falsification

Reject if any of:

- dry-run keeps fewer than 4 reward-varied groups;
- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.95`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Pending.
