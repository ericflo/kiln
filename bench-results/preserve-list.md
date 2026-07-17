# Phase 0.7 — Preserve-list audit

Sources of truth:

- `bench-results/preserve-list-nvtx.csv` (245 call sites, 155 distinct range names)
- `bench-results/preserve-list-env.csv` (1721 call sites, 668 distinct `KILN_*` vars; 39 go through `env_flag` / `env_tristate`)
- `bench-results/preserve-list-backend-runtime.csv` (0 trait methods whose signature still mentions a candle type)

Regenerate: `scripts/audit-preserve-list.sh`.

---

## Contract

Every migration PR that touches one of these three surfaces must either preserve the reachable contract or record why removing it is safe. Specifically:

- **NVTX range names** keep their exact string spelling so `PROFILING.md`'s hot-region percentages stay comparable across the migration.
- **Reachable `KILN_*` env vars** keep their exact name so user deployments do not break. Definition-only migration gates may be removed after a whole-tree reachability check; regenerate this audit and the runtime-environment contract in the same change.
- **`BackendRuntime` trait methods** keep their method names; only the argument types swap from candle to kiln-tensor.

## NVTX range clusters

Grouped by `kiln/<prefix>/...`; counts are total call sites.

| cluster | call sites |
|---|---:|
| `gdn` | 38 |
| `mlp` | 23 |
| `attn` | 20 |
| `norm` | 17 |
| `proj` | 16 |
| `residual` | 10 |
| `lm_head` | 6 |
| `mtp` | 6 |
| `paged_kv_kt` | 5 |
| `flash_attn_paged_decode_dyn_seqlen_kt` | 4 |
| `residual_batch_decode` | 4 |
| `batched_decode` | 4 |
| `final_norm` | 3 |
| `kv` | 3 |
| `adamw_step_kt` | 2 |
| `embedding_kt` | 2 |
| `final_rmsnorm` | 2 |
| `flash_attn_kt` | 2 |
| `flash_attn_paged_decode_kt` | 2 |
| `gdn_chunk_prep_kt` | 2 |
| `gdn_chunk_scan_kt` | 2 |
| `gdn_decode_gates_recurrent_bf16_kt` | 2 |
| `gdn_decode_qk_norm_gates_recurrent_bf16_kt` | 2 |
| `gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16_kt` | 2 |
| `gdn_forward_substitution_kt` | 2 |
| `gdn_full_chunk_forward_kt` | 2 |
| `gdn_full_chunk_forward_multiblock_kt` | 2 |
| `gdn_gated_rms_norm_kt` | 2 |
| `gdn_gates_bf16_kt` | 2 |
| `gdn_recurrent_forward_kt` | 2 |
| `gdn_solve_tri_transpose_f32_kt` | 2 |
| `muon_step_kt` | 2 |
| `sgd_step_kt` | 2 |
| `abs_kt` | 1 |
| `add_scalar_kt` | 1 |
| `argmax_kt` | 1 |
| `cat_dim0_kt` | 1 |
| `cat_dim1_kt` | 1 |
| `cat_dim2_kt` | 1 |
| `concat_last_dim_kt` | 1 |
| `cos_kt` | 1 |
| `exp_kt` | 1 |
| `flash_attn_head_major_kt` | 1 |
| `flash_attn_paged_decode_dyn_seqlen_kt_with_graph_outputs` | 1 |
| `gqa_sdpa_kt` | 1 |
| `l2_normalize_kt` | 1 |
| `lm_head_argmax` | 1 |
| `lm_head_argmax_eager` | 1 |
| `lm_head_argmax_kt` | 1 |
| `lm_head_batch_argmax_decode` | 1 |
| `lm_head_batch_argmax_decode_stable_buffers` | 1 |
| `lm_head_batch_decode` | 1 |
| `lm_head_eager` | 1 |
| `lm_head_eager_batched` | 1 |
| `lm_head_kt` | 1 |
| `lm_head_normalized_chunk` | 1 |
| `log_kt` | 1 |
| `lora_add_kt` | 1 |
| `lora_delta_kt` | 1 |
| `marlin_w4a16_gemm_kt` | 1 |
| `matmul_kt` | 1 |
| `max_binary_kt` | 1 |
| `max_last_dim_kt` | 1 |
| `mean_last_dim_kt` | 1 |
| `neg_kt` | 1 |
| `recip_kt` | 1 |
| `rsqrt_kt` | 1 |
| `sampling_argmax_rows_kt` | 1 |
| `sampling_penalties_kt` | 1 |
| `sampling_softmax_kt` | 1 |
| `sampling_topk_kt` | 1 |
| `sigmoid_composite_kt` | 1 |
| `silu_composite_kt` | 1 |
| `sin_kt` | 1 |
| `softmax_kt` | 1 |
| `sqrt_kt` | 1 |
| `sum_axis_kt` | 1 |
| `sum_last_dim_kt` | 1 |
| `sum_sq_last_dim_kt` | 1 |
| `swiglu_ffn_kt` | 1 |
| `to_dtype_kt` | 1 |

## Top 20 `KILN_*` env vars by call-site count

| env var | sites | via `env_flag`? | crates touched |
|---|---:|---:|---|
| `KILN_QUALIFICATION` | 28 | no | kiln-model;kiln-server;kiln-tensor;kiln-train;kiln-vulkan-kernel |
| `KILN_TENSOR_VULKAN_TEST` | 25 | no | kiln-model;kiln-tensor;kiln-train |
| `KILN_MTP_DUMP_B12_GQA_TAPS` | 16 | no | kiln-model |
| `KILN_MTP_DUMP_PATH` | 16 | no | kiln-model |
| `KILN_FORCE_EAGER_DECODE` | 12 | no | kiln-model |
| `KILN_KEEP_PROJECTION_ORIGINALS` | 11 | no | kiln-model |
| `KILN_MTP_DUMP_SPLICE` | 11 | no | kiln-model |
| `KILN_MODEL_PATH` | 10 | no | kiln-server;kiln-train |
| `KILN_STREAMING_PREFILL` | 10 | no | kiln-server |
| `KILN_STREAMING_TILE_TOKENS` | 10 | no | kiln-server |
| `KILN_ADAPTERS_COMPOSED_CACHE_MAX_BYTES` | 9 | no | kiln-server |
| `KILN_ADAPTERS_COMPOSED_CACHE_MAX_ENTRIES` | 9 | no | kiln-server |
| `KILN_ADAPTERS_MAX_DISK_BYTES` | 9 | no | kiln-server |
| `KILN_CUDA_GRAPHS` | 9 | no | kiln-model;kiln-server |
| `KILN_DETERMINISTIC` | 9 | no | kiln-server;kiln-tensor |
| `KILN_DISABLE_FUSED_PAGED_DECODE` | 9 | no | kiln-model |
| `KILN_EVAL_MODE` | 9 | no | kiln-server |
| `KILN_METAL_GRAPHS` | 9 | no | kiln-model |
| `KILN_METAL_GRAPH_STABLE_PAGED_METADATA` | 9 | no | kiln-model |
| `KILN_MTP_DUMP_HIDDEN_STATES` | 9 | no | kiln-model |

## `BackendRuntime` candle-typed methods

_No methods detected — the audit may need tuning if you expected hits here._

## Causal links forward

- **Phase 1 contract**: `BackendRuntime` is the seam (per the issue's starting points). The above method list is the call surface Phase 1 must shim — every method on it gets a kiln-tensor variant that the trait dispatches to under `KILN_USE_KILN_TENSOR_*`.
- **Phase 9 enforcement**: re-run this audit as a CI step; renaming a row in any of the three CSVs without a deliberate, documented decision fails the gate.
- **Anti-pattern 13 ('NVTX range names are part of the trace contract')**: the NVTX CSV is the verifiable form of that anti-pattern.
