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

Rejected.

Training completed successfully on the local WSL2 RTX 4090 CUDA 13.2 box with
`KILN_GRAD_CHECKPOINT_SEGMENTS=32`:

- groups kept/trained: 20/32 reward-variant groups, 80 completions;
- final loss: 0.052689;
- token counts: 80,202 action, 42,565 environment, 18,172 context;
- peak training VRAM: 22,569 MiB;
- elapsed: 7,190.913s;
- adapter verification: passed (`rank=8`, `alpha=16`, 400 tensors,
  nonzero LoRA effect, delta proxy L2 2.3794).

The CUDA server benchmark was valid after rebuilding `target/release/kiln` with
`cargo build --release --features cuda --bin kiln` for CUDA 13.2:

- server banner: `CUDA: available`;
- W4A16 Marlin pack: 104/104 projections;
- CUDA graphs: enabled;
- eval model requests: 317/317 ok, 0 errors, 0 timeouts;
- eval token traffic: 747,104 prefill tokens, 40,422 decode tokens;
- recent latency at completion: p50 2,429 ms, p95 9,071 ms, p99 10,381 ms;
- observed eval VRAM: about 19.3 GiB of 24 GiB.

Blind 3-seed eval regressed from the postbreach baseline:

- composite: 0.2383 vs baseline 0.2996 (`delta=-0.0613`);
- outcome: 0.5000 vs baseline 0.5000;
- format_compliance: 0.5000 vs baseline 0.5625;
- convention_consistency: 0.9079 vs baseline 0.9736;
- read_before_edit: 0.9583 vs baseline 1.0000;
- zero-score rollouts: 35/48 vs baseline 33/48;
- thinking efficiency: 301.2 chars/tool call vs baseline 308.8.

This falsifies the capacity-limited interpretation of H10. Raising no-ECHO
capacity from rank 4 to rank 8 increased adapter effect and training memory,
but did not clear the promotion gate; it damaged format, convention, and
read-before-edit while leaving outcome flat.
