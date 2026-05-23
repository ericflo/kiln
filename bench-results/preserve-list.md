# Phase 0.7 — Preserve-list audit

Sources of truth:

- `bench-results/preserve-list-nvtx.csv` (144 call sites, 81 distinct range names)
- `bench-results/preserve-list-env.csv` (973 call sites, 405 distinct `KILN_*` vars; 38 go through `env_flag` / `env_tristate`)
- `bench-results/preserve-list-backend-runtime.csv` (21 trait methods whose signature still mentions a candle type)

Regenerate: `scripts/audit-preserve-list.sh`.

---

## Contract

Every migration PR that touches one of these three surfaces must include an explicit 'preserved' checkbox in the PR description. Specifically:

- **NVTX range names** keep their exact string spelling so `PROFILING.md`'s hot-region percentages stay comparable across the migration.
- **`KILN_*` env vars** keep their exact name so user deployments do not break.
- **`BackendRuntime` trait methods** keep their method names; only the argument types swap from candle to kiln-tensor.

## NVTX range clusters

Grouped by `kiln/<prefix>/...`; counts are total call sites.

| cluster | call sites |
|---|---:|
| `gdn` | 36 |
| `attn` | 20 |
| `norm` | 17 |
| `proj` | 16 |
| `mlp` | 14 |
| `residual` | 10 |
| `lm_head` | 6 |
| `mtp` | 6 |
| `residual_batch_decode` | 4 |
| `batched_decode` | 4 |
| `final_norm` | 3 |
| `kv` | 3 |
| `final_rmsnorm` | 2 |
| `lm_head_argmax` | 1 |
| `lm_head_batch_argmax_decode` | 1 |
| `lm_head_batch_decode` | 1 |

## Top 20 `KILN_*` env vars by call-site count

| env var | sites | via `env_flag`? | crates touched |
|---|---:|---:|---|
| `KILN_NO_GRAD_CHECKPOINT` | 29 | no | kiln-core;kiln-server;kiln-train |
| `KILN_STREAMING_TILE_TOKENS` | 26 | no | kiln-model;kiln-server;kiln-train |
| `KILN_STREAMING_PREFILL` | 24 | no | kiln-model;kiln-server;kiln-train |
| `KILN_GRAD_CHECKPOINT_SEGMENTS` | 23 | no | kiln-core;kiln-server;kiln-train |
| `KILN_MTP_DUMP_B12_GQA_TAPS` | 16 | no | kiln-model |
| `KILN_MTP_DUMP_PATH` | 16 | no | kiln-model |
| `KILN_USE_FLCE` | 16 | yes | kiln-server;kiln-train |
| `KILN_DISABLE_OPD_LOSS_KERNEL` | 15 | no | kiln-opd-loss-kernel |
| `KILN_GPU_MEMORY_GB` | 13 | no | kiln-core;kiln-server;kiln-train |
| `KILN_VK_RECOMPUTE_GRPO` | 12 | no | kiln-train |
| `KILN_KEEP_PROJECTION_ORIGINALS` | 11 | no | kiln-model |
| `KILN_MTP_DUMP_SPLICE` | 11 | no | kiln-model |
| `KILN_DISABLE_FUSED_PAGED_DECODE` | 9 | no | kiln-model |
| `KILN_MTP_DUMP_HIDDEN_STATES` | 9 | no | kiln-model |
| `KILN_TRAINING_MEMORY_RESERVE_GB` | 9 | no | kiln-core;kiln-server |
| `KILN_VK_NATIVE_TRAINING` | 9 | yes | kiln-model;kiln-server |
| `KILN_CUDA_FLCE` | 8 | yes | kiln-train |
| `KILN_DISABLE_MARLIN_BF16_DROP` | 8 | no | kiln-model |
| `KILN_EXACT_GDN_BACKWARD_TILE_TOKENS` | 8 | no | kiln-train |
| `KILN_MODEL_PATH` | 8 | no | kiln-server;kiln-train |

## `BackendRuntime` candle-typed methods

| method | first line | candle Tensor? | candle Var? |
|---|---:|:-:|:-:|
| `materialize_gdn_recurrent_resident_state` | 273 | yes | no |
| `scatter_gdn_recurrent_resident_batch_rows` | 433 | no | no |
| `paged_kv_head_major_read` | 540 | yes | no |
| `paged_kv_head_major_read_append_token_major` | 557 | yes | no |
| `gdn_recurrent_step` | 593 | no | no |
| `gdn_chunk_prep` | 626 | yes | no |
| `gdn_chunk_scan` | 638 | yes | no |
| `gdn_full_chunk_forward` | 650 | no | no |
| `gdn_full_chunk_forward_head_last_into` | 666 | no | no |
| `gdn_recurrent_prefill_head_last` | 684 | no | no |
| `gdn_recurrent_prefill_native_head_last` | 696 | no | no |
| `gdn_recurrent_qk_norm_prefill_native_head_last` | 709 | no | no |
| `gdn_decode_gates_recurrent` | 730 | no | no |
| `gdn_decode_qk_norm_gates_recurrent` | 755 | no | no |
| `gdn_decode_qk_norm_gates_recurrent_rmsnorm` | 778 | no | no |
| `gdn_decode_gates_recurrent_rmsnorm` | 802 | no | no |
| `gdn_in_proj_decode` | 826 | yes | no |
| `full_attn_qkv_decode` | 1002 | yes | no |
| `causal_conv1d_update` | 1071 | no | no |
| `causal_conv1d_prefill` | 1090 | no | no |
| `gdn_gates` | 1107 | yes | no |

## Causal links forward

- **Phase 1 contract**: `BackendRuntime` is the seam (per the issue's starting points). The above method list is the call surface Phase 1 must shim — every method on it gets a kiln-tensor variant that the trait dispatches to under `KILN_USE_KILN_TENSOR_*`.
- **Phase 9 enforcement**: re-run this audit as a CI step; renaming a row in any of the three CSVs without a deliberate, documented decision fails the gate.
- **Anti-pattern 13 ('NVTX range names are part of the trace contract')**: the NVTX CSV is the verifiable form of that anti-pattern.
