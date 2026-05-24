# PB-H6: Train-Winner Trajectory SFT

## Hypothesis

PB-H2/H5 show that default-prompt train rollouts contain real successful edit
behavior, but GRPO updates trade off convention, final format, and outcome
depending on strength. PB-H1's synthetic ideal SFT was too distribution-shifted.
Imitating only train-rollout winners should provide distribution-matched
successful Pi trajectories that preserve read/edit/verify/final behavior and
local conventions without relying on a stronger reward update.

## Recipe

- Source: `/tmp/pi-context-aware-edits-postbreach-pb-h2-train-rollouts/grpo-train.jsonl`.
- Select train-only completions with reward >= 0.95.
- Convert each selected completion into a chat SFT row using the original task
  messages plus the completion text as the assistant response.
- Trainer: `cuda_sft_file`, `rank=4`, `alpha=8`, `lr=2e-6`,
  `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during eval through the server default.

## Falsification

Reject if any of:

- fewer than 8 train-winner SFT examples are available;
- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.95`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Pending.
