# PB-H12: Full-Train No-ECHO Half-Scale GRPO

## Hypothesis

PB-H10 is the current best postbreach direction: no-ECHO, full-train GRPO at
rank 4 lifted outcome and format together, but missed the +0.05 promotion gate
by 0.0092 and slipped on read-before-edit. PB-H11 showed that increasing
capacity and adapter effect is the wrong direction. Keep the H10 data,
rank, seed, and learning rate, but halve the LoRA scale from `alpha=8` to
`alpha=4`. If H10 was slightly over-updated, this should preserve most of the
outcome/format lift while recovering convention and read-before-edit enough to
clear or approach the gate.

## Recipe

- Source: `/tmp/pi-context-aware-edits-postbreach-pb-h7-fulltrain-default-rollouts/grpo-train.jsonl`.
- Data SHA: `sha256:c5a6770688cd1718c51a155785cbafa8820ecd6607b4339b3612b5bccce0575a`.
- Trainer: `cuda_grpo_ablation`, `mode=phase1`.
- Hyperparameters: `rank=4`, `alpha=4`, `lr=5e-6`, `filter_var_min=0.05`,
  `--no-echo`.
- Keep `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during eval through the server default and keep the
  1024 Pi output cap.

## Falsification

Reject if any of:

- fewer than 12 reward-variant train groups remain after filtering;
- composite lift < +0.05 versus postbreach baseline;
- composite is not above PB-H10's 0.3404 current-best score;
- `outcome < 0.5208`;
- `format_compliance < 0.6042`;
- `convention_consistency < 0.9542`;
- `read_before_edit < 0.9667`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Pending.
