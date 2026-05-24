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

Rejected. Dry-run reused the clean PB-H2 default-prompt GRPO data and kept
10/16 reward-varied groups (30 completions). Training completed with
`rank=16`, `alpha=32`, `lr=3e-5`, `echo_lambda=0.05`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=32`; peak VRAM was 19,478 MiB and the adapter
smoke test passed. Verification passed with 400 nonzero LoRA tensors and a
delta-proxy upper bound of 22.3440.

Blind 3-seed eval regressed from the postbreach baseline 0.2996 to 0.2900
(`delta=-0.0096`). The adapter did improve `format_compliance` from 0.5625 to
0.6875 and reduced tool/thinking cost (4.94 tool calls, 1449.8 thinking chars,
292.1 thinking chars/tool call), but `outcome` fell from 0.5000 to 0.3750 and
`convention_consistency` fell from 0.9736 to 0.9125. This falsifies the
"PB-H2 was merely underpowered" explanation: the strong update moved terminal
shape but damaged the actual edit behavior.
