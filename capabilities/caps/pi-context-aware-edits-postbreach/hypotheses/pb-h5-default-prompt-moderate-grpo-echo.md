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

Rejected. Dry-run reused the clean PB-H2 default-prompt GRPO data and kept
10/16 reward-varied groups (30 completions). Training completed with
`rank=8`, `alpha=16`, `lr=1e-5`, `echo_lambda=0.03`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=32`; peak VRAM was 19,450 MiB and the adapter
smoke test passed. Verification passed with 400 nonzero LoRA tensors and a
delta-proxy upper bound of 3.9514.

Blind 3-seed eval regressed from the postbreach baseline 0.2996 to 0.2683
(`delta=-0.0312`). The moderate update did preserve the PB-H2 outcome lift
(`outcome=0.5208`) and improved `format_compliance` to 0.6458, but
`convention_consistency` fell to 0.9417, `read_before_edit` and
`no_style_drift` slipped to 0.9792, and zero-score rollouts increased to
34/48. This falsifies the intermediate-strength GRPO recipe: it balances
outcome and final format better than PB-H4, but still fails the composite and
convention gates.
