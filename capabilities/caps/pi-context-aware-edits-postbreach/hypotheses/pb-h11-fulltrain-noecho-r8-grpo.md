# PB-H11: Full-Train No-ECHO Rank-8 GRPO

## Hypothesis

PB-H10 was the first postbreach adapter to beat baseline, improving composite,
outcome, and format together after removing ECHO. It missed the strict +0.05
promotion gate by only 0.0092 and slipped slightly on read-before-edit. Reuse
the same full-train default-prompt rollout data, keep ECHO disabled, and raise
capacity from rank 4 to rank 8 while keeping the conservative learning rate.
If H10 was capacity-limited rather than overfit, this should clear the
promotion gate without the ECHO-induced convention/read regressions.

## Recipe

- Source: `/tmp/pi-context-aware-edits-postbreach-pb-h7-fulltrain-default-rollouts/grpo-train.jsonl`.
- Data SHA: `sha256:c5a6770688cd1718c51a155785cbafa8820ecd6607b4339b3612b5bccce0575a`.
- Trainer: `cuda_grpo_ablation`, `mode=phase1`.
- Hyperparameters: `rank=8`, `alpha=16`, `lr=5e-6`, `filter_var_min=0.05`,
  `--no-echo`.
- Keep `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during eval through the server default and keep the
  1024 Pi output cap.

## Falsification

Reject if any of:

- fewer than 12 reward-variant train groups remain after filtering;
- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.5208`;
- `format_compliance <= 0.6042`;
- `convention_consistency < 0.95`;
- `read_before_edit < 0.9667`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Pending.
