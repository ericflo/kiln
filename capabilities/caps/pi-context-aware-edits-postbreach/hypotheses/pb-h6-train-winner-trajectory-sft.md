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

Rejected.

- Dataset: `datasets/sft.pb-h6-train-winner-trajectories.jsonl`, 17 examples
  across 11 train groups, SHA
  `sha256:ee872a7a1ae37e98f44853b786c57838cc2582df1597f01d2fd146d9708d825d`.
- Training completed with `rank=4`, `alpha=8`, `lr=2e-6`, one epoch, and
  `KILN_GRAD_CHECKPOINT_SEGMENTS=32`; CLI peak VRAM was 17116 MiB and elapsed
  time was 205.122 s.
- Adapter: `pi-context-aware-edits-postbreach-pb-h6-train-winner-sft-r4a8`.
  Verification passed with model SHA
  `sha256:93d717d7fe878b6044d16de65d2f5f39b01afa64cfbf382d7e35f3a604b42eff`
  and LoRA update L2 upper bound `0.578233`.
- Blind 3-seed eval regressed to composite `0.2371` (`delta=-0.0625`).
  Sub-scores were `outcome=0.4583`, `format_compliance=0.4792`,
  `convention_consistency=0.9750`, `read_before_edit=0.9792`,
  `no_redundant_imports=1.0000`, and `no_style_drift=1.0000`.

This falsifies winner-only trajectory SFT. The selected successful train
trajectories were too narrow or too weak a signal: convention/style stayed
healthy, but the model lost both outcome and terminal format versus the
postbreach baseline.
