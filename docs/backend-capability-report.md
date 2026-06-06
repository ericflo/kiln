# Backend Capability Report

Generated from the live source tree by `scripts/generate_backend_capability_report.py`.

- Branch: `unify-engines`

## Feature Fanout

| Crate | CUDA | ROCm | Metal | Vulkan |
|---|---|---|---|---|
| `kiln-server` | yes | yes | yes | yes |
| `kiln-model` | yes | yes | yes | yes |
| `kiln-tensor` | yes | yes | yes | yes |
| `kiln-train` | yes | yes | yes | yes |

## BackendRuntime Overrides

| Backend | Source | Override Count | Support Methods | Env Gates |
|---|---|---:|---:|---:|
| `cuda` | `crates/kiln-model/src/backend/cuda.rs` | 44 | 15 | 4 |
| `rocm` | `crates/kiln-model/src/backend/rocm.rs` | 48 | 17 | 10 |
| `metal` | `crates/kiln-model/src/backend/metal.rs` | 51 | 19 | 47 |
| `vulkan` | `crates/kiln-model/src/backend/vulkan.rs` | 73 | 20 | 32 |

## Support Predicates

| Backend | Method | Status | Paired Method | Pair Always Declines | Gates |
|---|---|---|---|---|---|
| `cuda` | `supports_causal_conv1d_prefill` | `dynamic` | `causal_conv1d_prefill` | no | none |
| `cuda` | `supports_causal_conv1d_update` | `dynamic` | `causal_conv1d_update` | no | none |
| `cuda` | `supports_flash_attn_paged_decode` | `literal_true` | `flash_attn_paged_decode` | no | none |
| `cuda` | `supports_flash_attn_prefill` | `literal_true` | `flash_attn_prefill` | no | none |
| `cuda` | `supports_gdn_chunk_prep` | `dynamic` | `gdn_chunk_prep` | no | none |
| `cuda` | `supports_gdn_chunk_scan` | `dynamic` | `gdn_chunk_scan` | no | none |
| `cuda` | `supports_gdn_decode_gates_recurrent_unexpanded_qk` | `dynamic` | `` | no | none |
| `cuda` | `supports_gdn_decode_qk_norm_gates_recurrent` | `dynamic` | `gdn_decode_qk_norm_gates_recurrent` | no | none |
| `cuda` | `supports_gdn_forward_substitution` | `dynamic` | `gdn_forward_substitution` | no | none |
| `cuda` | `supports_gdn_full_chunk_forward` | `dynamic` | `gdn_full_chunk_forward` | no | none |
| `cuda` | `supports_gdn_gated_rms_norm` | `dynamic` | `gdn_gated_rms_norm` | no | none |
| `cuda` | `supports_gdn_gates` | `dynamic` | `gdn_gates` | no | none |
| `cuda` | `supports_gdn_recurrent_step` | `dynamic` | `gdn_recurrent_step` | no | none |
| `cuda` | `supports_resident_activation` | `literal_true` | `` | no | none |
| `cuda` | `supports_strict_paged_decode_contiguous_batch` | `literal_false` | `` | no | none |
| `rocm` | `supports_causal_conv1d_prefill` | `dynamic` | `causal_conv1d_prefill` | no | none |
| `rocm` | `supports_causal_conv1d_update` | `dynamic` | `causal_conv1d_update` | no | none |
| `rocm` | `supports_flash_attn_paged_decode` | `literal_true` | `flash_attn_paged_decode` | no | none |
| `rocm` | `supports_flash_attn_prefill` | `literal_true` | `flash_attn_prefill` | no | none |
| `rocm` | `supports_flash_attn_prefill_head_major` | `env_gated` | `flash_attn_prefill_head_major` | no | env |
| `rocm` | `supports_gdn_chunk_prep` | `dynamic` | `gdn_chunk_prep` | no | none |
| `rocm` | `supports_gdn_chunk_scan` | `dynamic` | `gdn_chunk_scan` | no | none |
| `rocm` | `supports_gdn_decode_gates_recurrent_unexpanded_qk` | `dynamic` | `` | no | none |
| `rocm` | `supports_gdn_decode_qk_norm_gates_recurrent` | `dynamic` | `gdn_decode_qk_norm_gates_recurrent` | no | none |
| `rocm` | `supports_gdn_forward_substitution` | `dynamic` | `gdn_forward_substitution` | no | none |
| `rocm` | `supports_gdn_full_chunk_forward` | `dynamic` | `gdn_full_chunk_forward` | no | none |
| `rocm` | `supports_gdn_gated_rms_norm` | `dynamic` | `gdn_gated_rms_norm` | no | none |
| `rocm` | `supports_gdn_gates` | `dynamic` | `gdn_gates` | no | none |
| `rocm` | `supports_gdn_recurrent_step` | `dynamic` | `gdn_recurrent_step` | no | none |
| `rocm` | `supports_linear_decode_argmax` | `literal_false` | `linear_decode_argmax` | yes | none |
| `rocm` | `supports_resident_activation` | `literal_true` | `` | no | none |
| `rocm` | `supports_strict_paged_decode_contiguous_batch` | `literal_false` | `` | no | none |
| `metal` | `supports_causal_conv1d_prefill` | `dynamic` | `causal_conv1d_prefill` | no | none |
| `metal` | `supports_causal_conv1d_update` | `dynamic` | `causal_conv1d_update` | no | none |
| `metal` | `supports_flash_attn_paged_decode` | `literal_true` | `flash_attn_paged_decode` | no | none |
| `metal` | `supports_flash_attn_prefill` | `env_gated` | `flash_attn_prefill` | no | none |
| `metal` | `supports_flash_attn_prefill_head_major` | `env_gated` | `flash_attn_prefill_head_major` | no | none |
| `metal` | `supports_gdn_chunk_prep` | `dynamic` | `gdn_chunk_prep` | no | none |
| `metal` | `supports_gdn_forward_substitution` | `dynamic` | `gdn_forward_substitution` | no | none |
| `metal` | `supports_gdn_full_chunk_forward` | `dynamic` | `gdn_full_chunk_forward` | no | none |
| `metal` | `supports_gdn_full_chunk_forward_head_last` | `dynamic` | `` | no | none |
| `metal` | `supports_gdn_gated_rms_norm` | `dynamic` | `gdn_gated_rms_norm` | no | none |
| `metal` | `supports_gdn_gates` | `dynamic` | `gdn_gates` | no | none |
| `metal` | `supports_gdn_recurrent_prefill_head_last` | `dynamic` | `gdn_recurrent_prefill_head_last` | no | none |
| `metal` | `supports_gdn_recurrent_prefill_native_head_last` | `dynamic` | `gdn_recurrent_prefill_native_head_last` | no | none |
| `metal` | `supports_gdn_recurrent_step` | `dynamic` | `gdn_recurrent_step` | no | none |
| `metal` | `supports_linear_decode_sample` | `dynamic` | `linear_decode_sample` | no | none |
| `metal` | `supports_linear_decode_sample_batch` | `dynamic` | `linear_decode_sample_batch` | no | none |
| `metal` | `supports_paged_kv_head_major_read` | `literal_true` | `paged_kv_head_major_read` | no | none |
| `metal` | `supports_paged_kv_head_major_read_append_token_major` | `literal_true` | `paged_kv_head_major_read_append_token_major` | no | none |
| `metal` | `supports_resident_activation` | `literal_true` | `` | no | none |
| `vulkan` | `supports_causal_conv1d_prefill` | `dynamic` | `causal_conv1d_prefill` | no | none |
| `vulkan` | `supports_causal_conv1d_update` | `dynamic` | `causal_conv1d_update` | no | none |
| `vulkan` | `supports_flash_attn_paged_decode` | `dynamic` | `flash_attn_paged_decode` | no | none |
| `vulkan` | `supports_flash_attn_prefill` | `env_gated` | `flash_attn_prefill` | no | env |
| `vulkan` | `supports_flash_attn_prefill_head_major` | `literal_false` | `` | no | none |
| `vulkan` | `supports_gdn_chunk_prep` | `dynamic` | `gdn_chunk_prep` | no | none |
| `vulkan` | `supports_gdn_chunk_scan` | `dynamic` | `gdn_chunk_scan` | no | none |
| `vulkan` | `supports_gdn_forward_substitution` | `dynamic` | `gdn_forward_substitution` | no | none |
| `vulkan` | `supports_gdn_full_chunk_forward` | `dynamic` | `gdn_full_chunk_forward` | no | none |
| `vulkan` | `supports_gdn_gated_rms_norm` | `dynamic` | `gdn_gated_rms_norm` | no | none |
| `vulkan` | `supports_gdn_gates` | `dynamic` | `gdn_gates` | no | none |
| `vulkan` | `supports_gdn_recurrent_prefill_native_head_last` | `dynamic` | `gdn_recurrent_prefill_native_head_last` | no | none |
| `vulkan` | `supports_gdn_recurrent_qk_norm_prefill_native_head_last` | `dynamic` | `gdn_recurrent_qk_norm_prefill_native_head_last` | no | none |
| `vulkan` | `supports_gdn_recurrent_step` | `dynamic` | `gdn_recurrent_step` | no | none |
| `vulkan` | `supports_linear_decode_argmax` | `dynamic` | `linear_decode_argmax` | no | none |
| `vulkan` | `supports_linear_decode_argmax_batch` | `dynamic` | `linear_decode_argmax_batch` | no | none |
| `vulkan` | `supports_linear_decode_sample` | `dynamic` | `linear_decode_sample` | no | none |
| `vulkan` | `supports_linear_decode_sample_batch` | `dynamic` | `linear_decode_sample_batch` | no | none |
| `vulkan` | `supports_resident_activation` | `literal_true` | `` | no | none |
| `vulkan` | `supports_resident_decode` | `dynamic` | `decode_resident_pool_ready` | no | none |

## Generic DeviceOp Fallback

| Backend | Policy | Counter | Evidence |
|---|---|---|---|
| `cuda` | `strict_native_miss_errors` | `none` | crates/kiln-tensor/src/device_op.rs CUDA native miss falls through on CUDA storage and fails loudly |
| `rocm` | `host_round_trip_correctness_fallback` | `kiln_tensor::profile::device_op_host_fallback_counts().rocm_op{1,2,3}` | crates/kiln-tensor/src/device_op.rs ROCm missing native forward stages through CPU |
| `metal` | `host_round_trip_correctness_fallback` | `kiln_tensor::profile::device_op_host_fallback_counts().metal_op{1,2,3}` | crates/kiln-tensor/src/device_op.rs Metal missing native forward stages through CPU |
| `vulkan` | `host_round_trip_correctness_fallback` | `kiln_tensor::profile::device_op_host_fallback_counts().vulkan_op{1,2,3}` | crates/kiln-tensor/src/device_op.rs Vulkan missing native forward stages through CPU |

## Decode Hot-Path Fallback

| Backend | Default Policy | Debug Opt-In | Enforcement |
|---|---|---|---|
| `cpu` | `CorrectnessAllowed` | `not required` | CPU is the reference path |
| `cuda` | `CorrectnessAllowed` | `not required` | CUDA native misses remain device-visible/errors rather than silent host staging |
| `rocm` | `NativeRequired` | `KILN_DECODE_HOT_PATH_DEBUG_FALLBACK=1 or KILN_ROCM_DECODE_BATCH_GENERIC_FALLBACK=1` | batched decode errors before generic fallback when no ROCm native path produced tokens |
| `metal` | `NativeRequired` | `KILN_DECODE_HOT_PATH_DEBUG_FALLBACK=1 or KILN_METAL_DECODE_BATCH_GENERIC_FALLBACK=1` | batched/sample decode errors before generic fallback when no Metal native path produced tokens |
| `vulkan` | `NativeRequired` | `KILN_DECODE_HOT_PATH_DEBUG_FALLBACK=1 or KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK=1` | keeps the existing Vulkan no-generic-fallback default and routes it through FallbackPolicy |

## Training Optimizer Fallback

| Backend | Default Policy | Debug Opt-In | Enforcement |
|---|---|---|---|
| `cpu` | `CorrectnessAllowed` | `not required` | CPU is the reference optimizer path |
| `cuda` | `NativeRequired` | `KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK=1 or KILN_CUDA_TRAINING_OPTIMIZER_FALLBACK=1` | SGD/AdamW GPU training errors before kt/host fallback when native dispatch declines |
| `rocm` | `NativeRequired` | `KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK=1 or KILN_ROCM_TRAINING_OPTIMIZER_FALLBACK=1` | SGD/AdamW GPU training errors before kt/host fallback when native dispatch declines |
| `metal` | `NativeRequired` | `KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK=1 or KILN_METAL_TRAINING_OPTIMIZER_FALLBACK=1` | SGD/AdamW GPU training errors before kt/host fallback when native dispatch declines |
| `vulkan` | `NativeRequired` | `KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK=1 or KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK=1` | SGD/AdamW GPU training errors before kt/host fallback when native dispatch declines |

## Optimizer Dispatch

| Backend | SGD Step | AdamW Step |
|---|---|---|
| `cuda` | `overridden` | `overridden` |
| `rocm` | `overridden` | `overridden` |
| `metal` | `default_decline` | `overridden` |
| `vulkan` | `overridden` | `overridden` |

## Mismatch Audit

No literal-true support predicate currently pairs with an always-declining method body.

## Backend Env Gates

### CUDA
- `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT`
- `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM`
- `KILN_DISABLE_CUDA_GDN_PREFILL_GATES`
- `KILN_DISABLE_CUDA_LORA_DECODE_ADD`

### ROCM
- `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT`
- `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM`
- `KILN_DISABLE_CUDA_GDN_PREFILL_GATES`
- `KILN_DISABLE_CUDA_LORA_DECODE_ADD`
- `KILN_DISABLE_ROCM_GDN_DECODE_QK_NORM_RECURRENT`
- `KILN_DISABLE_ROCM_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM`
- `KILN_DISABLE_ROCM_GDN_PREFILL_GATES`
- `KILN_DISABLE_ROCM_LORA_DECODE_ADD`
- `KILN_ROCM_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK`
- `KILN_ROCM_HEAD_MAJOR_PREFILL`

### METAL
- `KILN_DISABLE_METAL_ATTN_GATE_FUSION`
- `KILN_DISABLE_METAL_CONV1D_PREFILL`
- `KILN_DISABLE_METAL_FUSED_CONV1D`
- `KILN_DISABLE_METAL_FUSED_QKV_PROJ`
- `KILN_DISABLE_METAL_GATED_RMSNORM`
- `KILN_DISABLE_METAL_GDN_DECODE_GATES_RECURRENT`
- `KILN_DISABLE_METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM`
- `KILN_DISABLE_METAL_GDN_FORWARD_SUBSTITUTION`
- `KILN_DISABLE_METAL_GDN_GATES`
- `KILN_DISABLE_METAL_GDN_IN_PROJ_FUSION`
- `KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_PAIR`
- `KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_QUAD`
- `KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_TRIPLE`
- `KILN_DISABLE_METAL_GDN_IN_PROJ_SERIAL_VECTOR_LOAD`
- `KILN_DISABLE_METAL_GDN_IN_PROJ_SERIAL_X2_LOAD`
- `KILN_DISABLE_METAL_GDN_PREFILL_AB_IN_PROJ`
- `KILN_DISABLE_METAL_GDN_PREFILL_DECAY_RECURRENT`
- `KILN_DISABLE_METAL_GDN_PREFILL_QKV_CONV_SPLIT`
- `KILN_DISABLE_METAL_GDN_QKV_CONV_NORM`
- `KILN_DISABLE_METAL_GDN_QK_NORM`
- `KILN_DISABLE_METAL_GDN_RECURRENT`
- `KILN_DISABLE_METAL_LM_HEAD_ARGMAX`
- `KILN_DISABLE_METAL_LM_HEAD_ARGMAX_GPU_REDUCE`
- `KILN_DISABLE_METAL_LM_HEAD_ARGMAX_ROWS`
- `KILN_DISABLE_METAL_LM_HEAD_SAMPLE`
- `KILN_DISABLE_METAL_LORA_DELTA_DECODE`
- `KILN_DISABLE_METAL_MLP_GATE_UP_FUSION`
- `KILN_DISABLE_METAL_MLP_GATE_UP_ROW_PAIR`
- `KILN_DISABLE_METAL_MLP_GATE_UP_ROW_QUAD`
- `KILN_DISABLE_METAL_MLP_GATE_UP_ROW_QUAD_VECTOR_LOAD`
- `KILN_DISABLE_METAL_MLP_GATE_UP_ROW_TRIPLE`
- `KILN_DISABLE_METAL_MLP_GATE_UP_SERIAL_DEDICATED`
- `KILN_DISABLE_METAL_MLP_GATE_UP_SERIAL_VECTOR_LOAD`
- `KILN_DISABLE_METAL_MLP_SILU_MUL`
- `KILN_DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS`
- `KILN_DISABLE_METAL_PAGED_KV_WRITE_TOKEN_MAJOR`
- `KILN_DISABLE_METAL_RMSNORM`
- `KILN_DISABLE_METAL_SDPA`
- `KILN_DISABLE_METAL_SDPA_FULL`
- `KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV`
- `KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_PAIR`
- `KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD`
- `KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD_TILE8`
- `KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_TRIPLE_TILE8`
- `KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE16`
- `KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8`
- `KILN_ENABLE_METAL_LM_HEAD_ARGMAX`

### VULKAN
- `KILN_DISABLE_VULKAN_BF16_PACKED_FULL_ATTN_QKV_WEIGHTS`
- `KILN_DISABLE_VULKAN_BF16_PACKED_GDN_IN_PROJ_WEIGHTS`
- `KILN_DISABLE_VULKAN_BF16_PACKED_LINEAR_WEIGHTS`
- `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS`
- `KILN_DISABLE_VULKAN_CONV1D_PREFILL_SINGLE_SUBMIT`
- `KILN_DISABLE_VULKAN_FULL_ATTN_QKV`
- `KILN_DISABLE_VULKAN_FUSED_CONV1D_PREFILL`
- `KILN_DISABLE_VULKAN_GDN_CHUNKWISE_FORWARD`
- `KILN_DISABLE_VULKAN_GDN_CHUNKWISE_SINGLE_SUBMIT`
- `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_RESIDENT_STATE`
- `KILN_DISABLE_VULKAN_GDN_PREFILL_IN_PROJ`
- `KILN_DISABLE_VULKAN_GDN_RECURRENT_QK_NORM`
- `KILN_DISABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE`
- `KILN_DISABLE_VULKAN_GDN_RECURRENT_UNEXPANDED_QK`
- `KILN_DISABLE_VULKAN_LINEAR_ARGMAX_BATCH`
- `KILN_DISABLE_VULKAN_LINEAR_DECODE`
- `KILN_DISABLE_VULKAN_MLP_BF16_GATE_UP_F32_DOWN`
- `KILN_DISABLE_VULKAN_MLP_DECODE`
- `KILN_DISABLE_VULKAN_PAGED_ATTN_DECODE_BATCH`
- `KILN_DISABLE_VULKAN_PAGED_DECODE_GPU_GATHER`
- `KILN_DISABLE_VULKAN_WEIGHT_PREWARM`
- `KILN_ENABLE_VULKAN_FUSED_CONV1D`
- `KILN_ENABLE_VULKAN_FUSED_CONV1D_UPDATE`
- `KILN_ENABLE_VULKAN_GDN_DECODE_FUSED`
- `KILN_ENABLE_VULKAN_GDN_FORWARD_SUB`
- `KILN_ENABLE_VULKAN_GDN_FULL_CHUNK_FORWARD`
- `KILN_ENABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE`
- `KILN_ENABLE_VULKAN_MLP_GATE_UP`
- `KILN_VULKAN_GDN_CHUNKWISE_FALLBACK`
- `KILN_VULKAN_LINEAR_MAX_GFLOP`
- `KILN_VULKAN_RESIDENT_DECODE`
- `KILN_VULKAN_SDPA`

