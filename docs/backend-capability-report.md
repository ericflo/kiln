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

| Backend | Source Modules | Override Count | Support Methods | Env Gates |
|---|---|---:|---:|---:|
| `cuda` | `crates/kiln-model/src/backend/cuda.rs` | 45 | 15 | 4 |
| `rocm` | `crates/kiln-model/src/backend/rocm.rs` | 49 | 17 | 10 |
| `metal` | `crates/kiln-model/src/backend/metal.rs`, `crates/kiln-model/src/backend/metal_training.rs` | 52 | 19 | 47 |
| `vulkan` | `crates/kiln-model/src/backend/vulkan.rs` | 74 | 20 | 32 |

## Support Predicates

| Backend | Method | Predicate Status | Support State | Paired Method | Pair Always Declines | Gates |
|---|---|---|---|---|---|---|
| `cuda` | `supports_causal_conv1d_prefill` | `dynamic` | `NativeWithConstraints` | `causal_conv1d_prefill` | no | none |
| `cuda` | `supports_causal_conv1d_update` | `dynamic` | `NativeWithConstraints` | `causal_conv1d_update` | no | none |
| `cuda` | `supports_flash_attn_paged_decode` | `literal_true` | `NativeWithConstraints` | `flash_attn_paged_decode` | no | none |
| `cuda` | `supports_flash_attn_prefill` | `literal_true` | `NativeWithConstraints` | `flash_attn_prefill` | no | none |
| `cuda` | `supports_gdn_chunk_prep` | `dynamic` | `NativeWithConstraints` | `gdn_chunk_prep` | no | none |
| `cuda` | `supports_gdn_chunk_scan` | `dynamic` | `NativeWithConstraints` | `gdn_chunk_scan` | no | none |
| `cuda` | `supports_gdn_decode_gates_recurrent_unexpanded_qk` | `dynamic` | `NativeWithConstraints` | `` | no | none |
| `cuda` | `supports_gdn_decode_qk_norm_gates_recurrent` | `dynamic` | `NativeWithConstraints` | `gdn_decode_qk_norm_gates_recurrent` | no | none |
| `cuda` | `supports_gdn_forward_substitution` | `dynamic` | `NativeWithConstraints` | `gdn_forward_substitution` | no | none |
| `cuda` | `supports_gdn_full_chunk_forward` | `dynamic` | `NativeWithConstraints` | `gdn_full_chunk_forward` | no | none |
| `cuda` | `supports_gdn_gated_rms_norm` | `dynamic` | `NativeWithConstraints` | `gdn_gated_rms_norm` | no | none |
| `cuda` | `supports_gdn_gates` | `dynamic` | `NativeWithConstraints` | `gdn_gates` | no | none |
| `cuda` | `supports_gdn_recurrent_step` | `dynamic` | `NativeWithConstraints` | `gdn_recurrent_step` | no | none |
| `cuda` | `supports_resident_activation` | `literal_true` | `NativeWithConstraints` | `` | no | none |
| `cuda` | `supports_strict_paged_decode_contiguous_batch` | `literal_false` | `Declined` | `` | no | none |
| `rocm` | `supports_causal_conv1d_prefill` | `dynamic` | `NativeWithConstraints` | `causal_conv1d_prefill` | no | none |
| `rocm` | `supports_causal_conv1d_update` | `dynamic` | `NativeWithConstraints` | `causal_conv1d_update` | no | none |
| `rocm` | `supports_flash_attn_paged_decode` | `literal_true` | `NativeWithConstraints` | `flash_attn_paged_decode` | no | none |
| `rocm` | `supports_flash_attn_prefill` | `literal_true` | `NativeWithConstraints` | `flash_attn_prefill` | no | none |
| `rocm` | `supports_flash_attn_prefill_head_major` | `env_gated` | `NativeWithConstraints` | `flash_attn_prefill_head_major` | no | env |
| `rocm` | `supports_gdn_chunk_prep` | `dynamic` | `NativeWithConstraints` | `gdn_chunk_prep` | no | none |
| `rocm` | `supports_gdn_chunk_scan` | `dynamic` | `NativeWithConstraints` | `gdn_chunk_scan` | no | none |
| `rocm` | `supports_gdn_decode_gates_recurrent_unexpanded_qk` | `dynamic` | `NativeWithConstraints` | `` | no | none |
| `rocm` | `supports_gdn_decode_qk_norm_gates_recurrent` | `dynamic` | `NativeWithConstraints` | `gdn_decode_qk_norm_gates_recurrent` | no | none |
| `rocm` | `supports_gdn_forward_substitution` | `dynamic` | `NativeWithConstraints` | `gdn_forward_substitution` | no | none |
| `rocm` | `supports_gdn_full_chunk_forward` | `dynamic` | `NativeWithConstraints` | `gdn_full_chunk_forward` | no | none |
| `rocm` | `supports_gdn_gated_rms_norm` | `dynamic` | `NativeWithConstraints` | `gdn_gated_rms_norm` | no | none |
| `rocm` | `supports_gdn_gates` | `dynamic` | `NativeWithConstraints` | `gdn_gates` | no | none |
| `rocm` | `supports_gdn_recurrent_step` | `dynamic` | `NativeWithConstraints` | `gdn_recurrent_step` | no | none |
| `rocm` | `supports_linear_decode_argmax` | `literal_false` | `Declined` | `linear_decode_argmax` | yes | none |
| `rocm` | `supports_resident_activation` | `literal_true` | `NativeWithConstraints` | `` | no | none |
| `rocm` | `supports_strict_paged_decode_contiguous_batch` | `literal_false` | `Declined` | `` | no | none |
| `metal` | `supports_causal_conv1d_prefill` | `dynamic` | `NativeWithConstraints` | `causal_conv1d_prefill` | no | none |
| `metal` | `supports_causal_conv1d_update` | `dynamic` | `NativeWithConstraints` | `causal_conv1d_update` | no | none |
| `metal` | `supports_flash_attn_paged_decode` | `literal_true` | `NativeWithConstraints` | `flash_attn_paged_decode` | no | none |
| `metal` | `supports_flash_attn_prefill` | `env_gated` | `NativeWithConstraints` | `flash_attn_prefill` | no | none |
| `metal` | `supports_flash_attn_prefill_head_major` | `env_gated` | `NativeWithConstraints` | `flash_attn_prefill_head_major` | no | none |
| `metal` | `supports_gdn_chunk_prep` | `dynamic` | `NativeWithConstraints` | `gdn_chunk_prep` | no | none |
| `metal` | `supports_gdn_forward_substitution` | `dynamic` | `NativeWithConstraints` | `gdn_forward_substitution` | no | none |
| `metal` | `supports_gdn_full_chunk_forward` | `dynamic` | `NativeWithConstraints` | `gdn_full_chunk_forward` | no | none |
| `metal` | `supports_gdn_full_chunk_forward_head_last` | `dynamic` | `NativeWithConstraints` | `` | no | none |
| `metal` | `supports_gdn_gated_rms_norm` | `dynamic` | `NativeWithConstraints` | `gdn_gated_rms_norm` | no | none |
| `metal` | `supports_gdn_gates` | `dynamic` | `NativeWithConstraints` | `gdn_gates` | no | none |
| `metal` | `supports_gdn_recurrent_prefill_head_last` | `dynamic` | `NativeWithConstraints` | `gdn_recurrent_prefill_head_last` | no | none |
| `metal` | `supports_gdn_recurrent_prefill_native_head_last` | `dynamic` | `NativeWithConstraints` | `gdn_recurrent_prefill_native_head_last` | no | none |
| `metal` | `supports_gdn_recurrent_step` | `dynamic` | `NativeWithConstraints` | `gdn_recurrent_step` | no | none |
| `metal` | `supports_linear_decode_sample` | `dynamic` | `NativeWithConstraints` | `linear_decode_sample` | no | none |
| `metal` | `supports_linear_decode_sample_batch` | `dynamic` | `NativeWithConstraints` | `linear_decode_sample_batch` | no | none |
| `metal` | `supports_paged_kv_head_major_read` | `literal_true` | `NativeWithConstraints` | `paged_kv_head_major_read` | no | none |
| `metal` | `supports_paged_kv_head_major_read_append_token_major` | `literal_true` | `NativeWithConstraints` | `paged_kv_head_major_read_append_token_major` | no | none |
| `metal` | `supports_resident_activation` | `literal_true` | `NativeWithConstraints` | `` | no | none |
| `vulkan` | `supports_causal_conv1d_prefill` | `dynamic` | `NativeWithConstraints` | `causal_conv1d_prefill` | no | none |
| `vulkan` | `supports_causal_conv1d_update` | `dynamic` | `NativeWithConstraints` | `causal_conv1d_update` | no | none |
| `vulkan` | `supports_flash_attn_paged_decode` | `dynamic` | `NativeWithConstraints` | `flash_attn_paged_decode` | no | none |
| `vulkan` | `supports_flash_attn_prefill` | `env_gated` | `NativeWithConstraints` | `flash_attn_prefill` | no | env |
| `vulkan` | `supports_flash_attn_prefill_head_major` | `literal_false` | `Declined` | `` | no | none |
| `vulkan` | `supports_gdn_chunk_prep` | `dynamic` | `NativeWithConstraints` | `gdn_chunk_prep` | no | none |
| `vulkan` | `supports_gdn_chunk_scan` | `dynamic` | `NativeWithConstraints` | `gdn_chunk_scan` | no | none |
| `vulkan` | `supports_gdn_forward_substitution` | `dynamic` | `NativeWithConstraints` | `gdn_forward_substitution` | no | none |
| `vulkan` | `supports_gdn_full_chunk_forward` | `dynamic` | `NativeWithConstraints` | `gdn_full_chunk_forward` | no | none |
| `vulkan` | `supports_gdn_gated_rms_norm` | `dynamic` | `NativeWithConstraints` | `gdn_gated_rms_norm` | no | none |
| `vulkan` | `supports_gdn_gates` | `dynamic` | `NativeWithConstraints` | `gdn_gates` | no | none |
| `vulkan` | `supports_gdn_recurrent_prefill_native_head_last` | `dynamic` | `NativeWithConstraints` | `gdn_recurrent_prefill_native_head_last` | no | none |
| `vulkan` | `supports_gdn_recurrent_qk_norm_prefill_native_head_last` | `dynamic` | `NativeWithConstraints` | `gdn_recurrent_qk_norm_prefill_native_head_last` | no | none |
| `vulkan` | `supports_gdn_recurrent_step` | `dynamic` | `NativeWithConstraints` | `gdn_recurrent_step` | no | none |
| `vulkan` | `supports_linear_decode_argmax` | `dynamic` | `NativeWithConstraints` | `linear_decode_argmax` | no | none |
| `vulkan` | `supports_linear_decode_argmax_batch` | `dynamic` | `NativeWithConstraints` | `linear_decode_argmax_batch` | no | none |
| `vulkan` | `supports_linear_decode_sample` | `dynamic` | `NativeWithConstraints` | `linear_decode_sample` | no | none |
| `vulkan` | `supports_linear_decode_sample_batch` | `dynamic` | `NativeWithConstraints` | `linear_decode_sample_batch` | no | none |
| `vulkan` | `supports_resident_activation` | `literal_true` | `NativeWithConstraints` | `` | no | none |
| `vulkan` | `supports_resident_decode` | `dynamic` | `NativeWithConstraints` | `decode_resident_pool_ready` | no | none |

## Typed Request Descriptors

| Descriptor | Field Count | DType | Shape | Batch | Replay Safe | Fields |
|---|---:|---|---|---|---|---|
| `AttentionRequest` | 8 | yes | no | yes | yes | `kind`, `q_dtype`, `k_dtype`, `v_dtype`, `batch`, `seq_len`, `head_dim`, `replay_safe` |
| `MatmulRequest` | 12 | yes | yes | yes | yes | `lhs_shape`, `rhs_shape`, `lhs_dtype`, `rhs_dtype`, `out_dtype`, `accumulation`, `lhs_layout`, `rhs_layout`, `out_layout`, `batch`, `epilogue`, `replay_safe` |
| `MatmulBlasRequest` | 10 | yes | yes | yes | no | `m`, `n`, `k`, `dtype`, `lhs_layout`, `rhs_layout`, `out_layout`, `epilogue`, `batch`, `concurrent_streams` |
| `LinearRequest` | 8 | yes | no | yes | yes | `kind`, `input_dtype`, `weight_dtype`, `output_dtype`, `batch`, `top_k`, `temperatures`, `replay_safe` |
| `ReplayRequest` | 6 | yes | no | yes | yes | `kind`, `max_hidden`, `max_intermediate`, `max_batch`, `dtype`, `replay_safe` |

## Request Capability Queries

- `backend_capabilities`
- `capability_snapshot`
- `replay_key_for_request`
- `supports_attention_request`
- `supports_linear_request`
- `supports_matmul_request`
- `supports_replay_request`

## Typed Capability Descriptors

| Descriptor | Field Count | Fields |
|---|---:|---|
| `BackendCapabilities` | 10 | `backend`, `device`, `storage`, `matmul`, `attention`, `gdn`, `decode`, `training`, `graph_replay`, `fallback` |
| `StorageCapabilities` | 4 | `backend`, `device`, `resident_activation`, `resident_decode` |
| `MatmulCapabilities` | 3 | `rank2_f32`, `batched_bf16`, `bias_epilogue` |
| `AttentionCapabilities` | 3 | `flash_prefill`, `flash_prefill_head_major`, `flash_paged_decode` |
| `GdnCapabilities` | 6 | `recurrent_step`, `chunk_prep`, `chunk_scan`, `full_chunk_forward`, `gates`, `gated_rms_norm` |
| `DecodeCapabilities` | 6 | `resident_decode`, `paged_decode_graph_outputs`, `linear_argmax`, `linear_argmax_batch`, `linear_sample`, `linear_sample_batch` |
| `BackendTrainingCapabilities` | 2 | `hooks`, `precision` |
| `ReplayCapabilities` | 2 | `resident_decode`, `paged_decode_graph_outputs` |
| `BackendFallbackCapabilities` | 3 | `generic_device_op`, `decode_hot_path`, `training_optimizer` |

## Resident Resource Descriptors

| Descriptor | Field Count | Fields |
|---|---:|---|
| `ResidentResource` | 13 | `tensor_id`, `backend`, `device`, `dtype`, `shape`, `layout`, `element_count`, `byte_len`, `addressable_byte_len`, `family`, `ownership`, `state`, `replay_stability` |
| `ResidentResourceLayout` | 3 | `strides`, `start_offset`, `contiguous` |

## Conformance And Performance Gates

| Gate | Phase 8 Requirement | Status | Command | Evidence | Missing Evidence |
|---|---|---|---|---|---|
| `storage_round_trip` | storage round trip | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-tensor rocm_storage_smoke` | `crates/kiln-tensor/tests/rocm_storage_smoke.rs`, `crates/kiln-vulkan-kernel/tests/vk_tensor_parity.rs` | none |
| `host_transfer_to_device_parity` | host transfer / to_device parity with explicit unsupported errors | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-tensor device_transfer_support_classifies_explicit_transitions && /home/ericflo/.cargo/bin/cargo test -p kiln-tensor to_device_without_gpu_features_reports_explicit_unsupported_transition && /home/ericflo/.cargo/bin/cargo test -p kiln-tensor cuda_resize_copy_primitives` | `crates/kiln-tensor/src/tensor.rs`, `crates/kiln-tensor/tests/cuda_resize_copy_primitives.rs`, `crates/kiln-tensor/tests/metal_ops_parity.rs`, `crates/kiln-tensor/tests/rocm_compare_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_tensor_parity.rs` | none |
| `device_op_parity` | DeviceOp parity | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-tensor device_op::tests` | `crates/kiln-tensor/src/device_op.rs`, `crates/kiln-tensor/tests/rocm_scalar_op_parity.rs`, `crates/kiln-tensor/tests/metal_ops_parity.rs` | none |
| `matmul_linear_parity` | matmul/linear parity | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-model matmul_request_projects_to_blas_shape_contract && /home/ericflo/.cargo/bin/cargo test -p kiln-tensor rocm_matmul_parity && /home/ericflo/.cargo/bin/cargo test -p kiln-tensor matmul_matrix_core && /home/ericflo/.cargo/bin/cargo test -p kiln-vulkan-kernel vk_matmul && /home/ericflo/.cargo/bin/cargo test -p kiln-vulkan-kernel linear_decode && /home/ericflo/.cargo/bin/cargo test -p kiln-model tape_forward_matmul_bit_exact_parity_with_baseline && /home/ericflo/.cargo/bin/cargo test -p kiln-blas cublaslt_handle_smoke && /home/ericflo/.cargo/bin/cargo test -p kiln-model --test backend_capability_contract` | `crates/kiln-model/src/backend/capability.rs`, `crates/kiln-model/src/backend/mod.rs`, `crates/kiln-blas/tests/cublaslt_handle_smoke.rs`, `crates/kiln-tensor/tests/rocm_matmul_parity.rs`, `crates/kiln-tensor/tests/metal_ops_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs`, `crates/kiln-vulkan-kernel/tests/linear_decode_argmax.rs`, `crates/kiln-vulkan-kernel/tests/linear_decode_sample.rs`, `crates/kiln-model/tests/tape_forward_parity.rs`, `crates/kiln-model/tests/marlin_qproj_parity.rs` | none |
| `attention_gdn_conv_parity` | attention/GDN/conv parity | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-model rocm_flash_attn_bwd_gradcheck` | `crates/kiln-flash-attn/tests/rocm_flash_attn_parity.rs`, `crates/kiln-gdn-kernel/tests/rocm_gdn_parity.rs`, `crates/kiln-conv1d-kernel/tests/rocm_conv1d_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_attention_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_gdn_foundation_parity.rs` | none |
| `optimizer_parity` | optimizer parity | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-optim --test integration && /home/ericflo/.cargo/bin/cargo test -p kiln-train training_optimizer && /home/ericflo/.cargo/bin/cargo test -p kiln-model --test backend_capability_contract` | `crates/kiln-optim/tests/integration.rs`, `crates/kiln-model/src/backend/cuda.rs`, `crates/kiln-model/src/backend/rocm.rs`, `crates/kiln-model/src/backend/metal.rs`, `crates/kiln-model/src/backend/vulkan.rs`, `crates/kiln-model/src/backend/mod.rs`, `crates/kiln-train/src/trainer.rs`, `crates/kiln-train/tests/vk_cuda_opd_parity.rs` | none |
| `replay_parity` | replay parity | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-graph replay && /home/ericflo/.cargo/bin/cargo test -p kiln-model --test backend_capability_contract` | `crates/kiln-graph/src/replay_plan.rs`, `crates/kiln-graph/src/captured_graph.rs`, `crates/kiln-graph/tests/capture_lifetime.rs`, `crates/kiln-model/src/backend/capability.rs`, `crates/kiln-model/src/backend/residency.rs`, `crates/kiln-model/tests/vk_resident_decode_parity.rs`, `crates/kiln-tensor/tests/rocm_capture_arena.rs` | none |
| `one_step_training_proof` | one-step training proof | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-model cuda_sft_step_proof && /home/ericflo/.cargo/bin/cargo test -p kiln-model metal_sft_step_proof && /home/ericflo/.cargo/bin/cargo test -p kiln-model vk_sft_step_proof && /home/ericflo/.cargo/bin/cargo test -p kiln-model rocm_sft_step_proof && /home/ericflo/.cargo/bin/cargo test -p kiln-optim --test end_to_end_training && /home/ericflo/.cargo/bin/cargo test -p kiln-model --test backend_capability_contract` | `crates/kiln-model/tests/cuda_sft_step_proof.rs`, `crates/kiln-model/tests/metal_sft_step_proof.rs`, `crates/kiln-model/tests/vk_sft_step_proof.rs`, `crates/kiln-model/tests/rocm_sft_step_proof.rs`, `crates/kiln-optim/tests/end_to_end_training.rs` | none |
| `no_unexpected_host_fallback` | no unexpected host fallback in decode/training hot paths | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-tensor device_op_host_fallback_counts_are_backend_and_arity_specific` | `crates/kiln-tensor/src/device_op.rs`, `crates/kiln-model/src/generate.rs`, `crates/kiln-train/src/trainer.rs` | none |
| `decode_submit_or_replay_count` | max submit count or replay count per decode token | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-model decode_batcher_stats_report_runner_calls_per_token && /home/ericflo/.cargo/bin/cargo test -p kiln-server test_metrics_render && /home/ericflo/.cargo/bin/cargo test -p kiln-graph replay` | `crates/kiln-model/src/generate.rs`, `crates/kiln-server/src/metrics.rs`, `crates/kiln-server/src/api/health.rs`, `crates/kiln-server/src/api/debug_model_state.rs`, `crates/kiln-graph/src/captured_graph.rs`, `crates/kiln-graph/src/replay_plan.rs` | none |
| `matmul_algorithm_cache_reporting` | matmul algorithm/cache hit reporting | `covered` | `/home/ericflo/.cargo/bin/cargo test -p kiln-blas cache_stats_reports_entries_and_hit_rate && /home/ericflo/.cargo/bin/cargo test -p kiln-rocblas cache_stats_reports_entries_and_hit_rate && CUDARC_CUDA_VERSION=12080 /home/ericflo/.cargo/bin/cargo check -p kiln-blas --features cublaslt --tests && /home/ericflo/.cargo/bin/cargo check -p kiln-rocblas --features hipblaslt --tests` | `crates/kiln-blas/src/algo_cache.rs`, `crates/kiln-blas/src/cublaslt_handle.rs`, `crates/kiln-blas/tests/cublaslt_handle_smoke.rs`, `crates/kiln-rocblas/src/algo_cache.rs`, `crates/kiln-rocblas/src/hipblaslt_handle.rs` | none |
| `hardware_latency_thresholds` | backend-specific latency thresholds on known hardware fixtures | `fixture_required` | `hardware runner required; python3 scripts/check_backend_latency_fixtures.py docs/backend-latency-fixtures.json --require-covered` | `docs/backend-latency-fixtures.json`, `scripts/check_backend_latency_fixtures.py`, `crates/kiln-server/examples/flce_preflight_bench.rs`, `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`, `crates/kiln-tensor/tests/metal_matmul_bench.rs`, `crates/kiln-tensor/tests/metal_sdpa_bench.rs`, `crates/kiln-vulkan-kernel/src/bin/vulkan_decode_microbench.rs`, `crates/kiln-tensor/tests/rocm_latency_bench.rs` | none |
| `generated_capability_dashboard` | generated capability dashboard checked into docs or build artifacts | `covered` | `python3 scripts/generate_backend_capability_report.py --check` | `docs/backend-capability-report.md`, `docs/backend-capability-report.json` | none |

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

## Training Precision Policy

| Backend | Policy | Activations | Base Weights | LoRA | Loss Accum | Optimizer Params | Mixed |
|---|---|---|---|---|---|---|---|
| `cpu` | `cpu_f32_reference` | `F32` | `F32` | `F32` | `F32` | `F32` | no |
| `cuda` | `cuda_native_float` | `F32,BF16,F16` | `F32,BF16,F16` | `F32,BF16` | `F32` | `F32,BF16` | yes |
| `rocm` | `rocm_native_float` | `F32,BF16,F16` | `F32,BF16,F16` | `F32,BF16` | `F32` | `F32,BF16` | yes |
| `metal` | `metal_bf16_uma` | `BF16` | `BF16` | `F32,BF16` | `F32` | `F32,BF16` | yes |
| `vulkan` | `vulkan_mixed_f32_bf16` | `F32` | `F32,BF16` | `F32` | `F32` | `F32,BF16` | yes |

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

