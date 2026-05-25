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

Rejected. The clean-binary filtered set met the training-data gate
(`15` reward-variant groups / `60` completions) and trained cleanly under CUDA
13.2 on the RTX 4090 with gradient checkpointing (`rank=4`, `alpha=8`,
`lr=5e-6`, no ECHO, final loss `0.148495`, observed peak VRAM `22,555 MiB`,
elapsed `5,794.8s`). Adapter smoke test and `kiln adapters verify` both passed;
verification found 400 nonzero tensors and a LoRA update L2 proxy of `1.0291`.

Blind 3-seed eval regressed to `0.2892` composite (`delta=-0.0104`, stdev
`0.4390`, 48 rollouts), below both the `0.2996` postbreach baseline and
PB-H10's `0.3404` current-best score. Sub-scores were:
`outcome=0.4583`, `format_compliance=0.5417`,
`convention_consistency=0.9312`, `read_before_edit=1.0000`,
`no_redundant_imports=1.0000`, and `no_style_drift=1.0000`. Efficiency was
`5.90` tool calls/rollout, `1966.1` thinking chars, and `343.8` thinking
chars/tool.

This falsifies the idea that PB-H10's useful movement came from clean
success/failure contrast alone. Dropping partial-credit groups removed useful
training signal and regressed outcome, format, and convention. H10 remains the
current best caveated adapter, and further filtering/rank/alpha sweeps around
the PB-H7 no-ECHO data are not promising.
