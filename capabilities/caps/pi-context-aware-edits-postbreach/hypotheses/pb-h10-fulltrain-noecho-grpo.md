# PB-H10: Full-Train No-ECHO Conservative GRPO

## Hypothesis

Every postbreach GRPO update so far used ECHO. PB-H7 showed that simply
broadening the train rollout pool with ECHO did not stabilize behavior; it
damaged outcome, format, convention, and read-before-edit. `pi-doctest` also
found that no-ECHO can qualitatively change the outcome/efficiency tradeoff.
Reuse the same full-train default-prompt rollout data as PB-H7, but train a
smaller policy-only update with ECHO disabled. If ECHO was over-imprinting
environment traces, this should preserve more of the base edit contract.

## Recipe

- Source: `/tmp/pi-context-aware-edits-postbreach-pb-h7-fulltrain-default-rollouts/grpo-train.jsonl`.
- Data SHA: `sha256:c5a6770688cd1718c51a155785cbafa8820ecd6607b4339b3612b5bccce0575a`.
- Trainer: `cuda_grpo_ablation`, `mode=phase1`.
- Hyperparameters: `rank=4`, `alpha=8`, `lr=5e-6`, `filter_var_min=0.05`,
  `--no-echo`.
- Keep `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during eval through the server default.

## Falsification

Reject if any of:

- fewer than 12 reward-variant train groups remain after filtering;
- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.95`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Pending.
