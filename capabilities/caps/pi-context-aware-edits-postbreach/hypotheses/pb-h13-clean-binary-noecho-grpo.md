# PB-H13: Clean-Binary No-ECHO GRPO

## Hypothesis

PB-H10 remains the only postbreach adapter above baseline, but PB-H11 and
PB-H12 show that rank/scale sweeps around the same full-train no-ECHO data are
not the answer. The reused PB-H7 rollout pool contains six partial-credit
groups (`0.48`, `0.50`, or `0.96` rewards) mixed with clean success/failure
groups. Those partial groups may encode ambiguous terminal-format or
convention/read tradeoffs. Filter the same train-only rollout pool to groups
whose rewards are exactly binary (`0.0` and `1.0`) and rerun the H10
hyperparameters. If noisy partial-credit groups caused H10's read/convention
slip, this cleaner contrast set should preserve H10's outcome/format lift with
less contract drift.

## Recipe

- Source: `/tmp/pi-context-aware-edits-postbreach-pb-h7-fulltrain-default-rollouts/grpo-train.jsonl`.
- Source SHA: `sha256:c5a6770688cd1718c51a155785cbafa8820ecd6607b4339b3612b5bccce0575a`.
- Filtered data: `/tmp/pi-context-aware-edits-postbreach-pb-h13-clean-binary-rollouts/grpo-train.jsonl`.
- Filtered SHA: `sha256:854489d73ba6ee891117d0dc556aee589cdfc29d62d01e893a4b44dbcc5c5aee`.
- Selection: keep only train groups with reward set exactly `{0.0, 1.0}`.
  This keeps 15 groups / 60 completions from source lines
  `1,2,3,5,7,8,9,15,16,18,20,22,25,27,31`.
- Trainer: `cuda_grpo_ablation`, `mode=phase1`.
- Hyperparameters: `rank=4`, `alpha=8`, `lr=5e-6`, `filter_var_min=0.05`,
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
