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

Trained and verified successfully.

- Dry-run/filter gate: 20/32 reward-variant groups kept, satisfying the
  `min_groups=12` gate; 80/128 completions trained.
- Training: final loss 0.053674, peak VRAM 20,905 MiB, elapsed 7,093.737s,
  action/env/context tokens 80,202 / 42,565 / 18,172.
- Adapter verify: passed; 400 tensors, 200 matching LoRA projection pairs,
  rank 4, alpha 8, nonzero LoRA effect proxy L2 1.232194.
- Blind eval: composite 0.3404 over 48 rollouts, delta +0.0408 versus the
  postbreach baseline, stdev 0.4649.
- Sub-scores: `outcome=0.5208`, `format_compliance=0.6042`,
  `convention_consistency=0.9542`, `read_before_edit=0.9667`,
  `no_redundant_imports=1.0000`, `no_style_drift=1.0000`.
- Efficiency: 5.85 tool calls, 1905.1 thinking chars, 310.9 thinking
  chars/tool.

Verdict: kept with caveat, not a clean promotion. This is the first
postbreach adapter to beat baseline and it passes the outcome, format,
convention, efficiency, and adapter verification gates. It misses the
predeclared +0.05 composite lift gate by 0.0092 and slips on read-before-edit,
so it should be treated as the current best candidate direction rather than a
promoted stage. The key evidence is that disabling ECHO reverses the repeated
postbreach GRPO composite regressions.
