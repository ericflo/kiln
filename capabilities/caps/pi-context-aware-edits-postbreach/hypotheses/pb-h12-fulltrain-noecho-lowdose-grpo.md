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

Rejected.

Training completed successfully on the local WSL2 RTX 4090 CUDA 13.2 box with
`KILN_GRAD_CHECKPOINT_SEGMENTS=32`:

- groups kept/trained: 20/32 reward-variant groups, 80 completions;
- final loss: 0.053483;
- token counts: 80,202 action, 42,565 environment, 18,172 context;
- peak training VRAM: 20,809 MiB;
- elapsed: 7,500.027s;
- adapter verification: passed (`rank=4`, `alpha=4`, 400 tensors,
  nonzero LoRA effect, delta proxy L2 1.2182).

The CUDA eval server was healthy:

- server banner: `CUDA: available`;
- W4A16 Marlin pack: 104/104 projections;
- CUDA graphs: enabled;
- eval model requests: 329/329 ok, 0 errors, 0 timeouts;
- eval token traffic: 797,925 prefill tokens, 46,111 decode tokens;
- recent latency at completion: p50 2,424.5 ms, p95 8,873.05 ms,
  p99 10,923.96 ms.

Blind 3-seed eval regressed below both baseline and PB-H10:

- composite: 0.2163 vs baseline 0.2996 (`delta=-0.0833`);
- outcome: 0.4792 vs baseline 0.5000 and PB-H10 0.5208;
- format_compliance: 0.5000 vs baseline 0.5625 and PB-H10 0.6042;
- convention_consistency: 0.9542, matching PB-H10 but below baseline;
- read_before_edit: 0.9792, above PB-H10 but below baseline;
- zero-score rollouts: 37/48;
- thinking efficiency: 345.8 chars/tool call.

This falsifies the "H10 was simply too large" interpretation. Halving the
LoRA scale recovered a little read-before-edit but lost the outcome and format
movement that made H10 useful. H10 remains the current best caveated adapter;
the next attempt should change data selection, reward/objective, or runtime
policy rather than continue rank/alpha scaling around the same full-train
no-ECHO recipe.
