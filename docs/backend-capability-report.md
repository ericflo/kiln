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

## Migration Phase Status

| Phase | Title | Status | Contract | Migration | Genuine | Evidence | Remaining |
|---|---|---|---|---|---|---|---|
| Phase 0 | Audit and stabilize capability reporting | `covered` | `landed` | `complete` | yes | `docs/backend-engine-unification-plan.md`, `scripts/generate_backend_capability_report.py`, `docs/backend-capability-report.md`, `docs/backend-capability-report.json`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |
| Phase 1 | Introduce focused backend traits | `covered` | `landed` | `complete` | yes | `crates/kiln-model/src/backend/mod.rs`, `scripts/generate_backend_capability_report.py`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |
| Phase 2 | Normalize fallback policy | `covered` | `landed` | `complete` | yes | `crates/kiln-tensor/src/device_op.rs`, `crates/kiln-model/src/generate.rs`, `crates/kiln-train/src/trainer.rs`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |
| Phase 3 | Unify resident resource semantics | `covered` | `landed` | `complete` | yes | `crates/kiln-model/src/backend/residency.rs`, `crates/kiln-model/src/backend/metal_residency.rs`, `crates/kiln-model/src/backend/vulkan_residency.rs`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |
| Phase 4 | Unify matmul and linear dispatch | `covered` | `landed` | `complete` | yes | `crates/kiln-model/src/backend/capability.rs`, `crates/kiln-model/src/backend/mod.rs`, `crates/kiln-blas/src/cublaslt_handle.rs`, `crates/kiln-rocblas/src/hipblaslt_handle.rs`, `crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |
| Phase 5 | Move replay into the authoritative graph layer | `covered` | `landed` | `complete` | yes | `crates/kiln-graph/src/replay_plan.rs`, `crates/kiln-graph-cuda/src/lib.rs`, `crates/kiln-graph-metal/src/lib.rs`, `crates/kiln-graph-vulkan/src/lib.rs`, `crates/kiln-model/src/cuda_graph.rs`, `crates/kiln-model/src/rocm_graph.rs`, `crates/kiln-model/src/metal_graph.rs`, `crates/kiln-model/src/vk_decode_resident.rs`, `crates/kiln-vulkan-kernel/src/cmd_batch.rs`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |
| Phase 6 | Finish shared training integration | `covered` | `landed` | `complete` | yes | `crates/kiln-train/src/trainer.rs`, `crates/kiln-train/src/sft_tape_shim.rs`, `crates/kiln-train/src/grpo_tape_shim.rs`, `crates/kiln-train/src/opd_tape_shim.rs`, `crates/kiln-model/src/backend/metal_training.rs`, `crates/kiln-model/src/backend/vulkan_training.rs`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |
| Phase 7 | Decompose backend modules | `covered` | `landed` | `complete` | yes | `crates/kiln-model/src/backend/metal.rs`, `crates/kiln-model/src/backend/metal_attention.rs`, `crates/kiln-model/src/backend/metal_gdn.rs`, `crates/kiln-model/src/backend/metal_residency.rs`, `crates/kiln-model/src/backend/metal_training.rs`, `crates/kiln-model/src/backend/vulkan.rs`, `crates/kiln-model/src/backend/vulkan_residency.rs`, `crates/kiln-model/src/backend/vulkan_tensor_bridge.rs`, `crates/kiln-model/src/backend/cuda_rocm_common.rs`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |
| Phase 8 | Conformance and performance gates | `covered` | `landed` | `complete` | yes | `docs/backend-capability-report.md`, `docs/backend-capability-report.json`, `docs/backend-latency-fixtures.json`, `docs/backend-latency-result-schema.md`, `scripts/check_unification_gates.sh`, `scripts/run_backend_latency_fixture.py`, `scripts/write_backend_latency_result_artifact.py`, `scripts/import_backend_latency_artifact.py`, `scripts/lock_backend_latency_thresholds.py`, `scripts/check_backend_latency_fixtures.py`, `scripts/plan_backend_latency_fixture_dispatch.py`, `scripts/generate_backend_capability_report.py`, `crates/kiln-model/tests/backend_capability_contract.rs` | none |

## BackendRuntime Overrides

| Backend | Source Modules | Override Count | Support Methods | Native Env Gates | Legacy Env Aliases |
|---|---|---:|---:|---:|---:|
| `cuda` | `crates/kiln-model/src/backend/cuda.rs`, `crates/kiln-model/src/backend/cuda_rocm_common.rs` | 0 | 16 | 7 | 0 |
| `rocm` | `crates/kiln-model/src/backend/rocm.rs`, `crates/kiln-model/src/backend/cuda_rocm_common.rs` | 0 | 17 | 9 | 7 |
| `metal` | `crates/kiln-model/src/backend/metal.rs`, `crates/kiln-model/src/backend/metal_attention.rs`, `crates/kiln-model/src/backend/metal_config.rs`, `crates/kiln-model/src/backend/metal_conv1d.rs`, `crates/kiln-model/src/backend/metal_core.rs`, `crates/kiln-model/src/backend/metal_dense.rs`, `crates/kiln-model/src/backend/metal_gdn.rs`, `crates/kiln-model/src/backend/metal_icb.rs`, `crates/kiln-model/src/backend/metal_lm_head.rs`, `crates/kiln-model/src/backend/metal_msl.rs`, `crates/kiln-model/src/backend/metal_norm.rs`, `crates/kiln-model/src/backend/metal_paged.rs`, `crates/kiln-model/src/backend/metal_pipeline.rs`, `crates/kiln-model/src/backend/metal_precompile.rs`, `crates/kiln-model/src/backend/metal_residency.rs`, `crates/kiln-model/src/backend/metal_runtime.rs`, `crates/kiln-model/src/backend/metal_training.rs` | 0 | 20 | 47 | 0 |
| `vulkan` | `crates/kiln-model/src/backend/vulkan.rs`, `crates/kiln-model/src/backend/vulkan_attention.rs`, `crates/kiln-model/src/backend/vulkan_config.rs`, `crates/kiln-model/src/backend/vulkan_conv1d.rs`, `crates/kiln-model/src/backend/vulkan_decode_state.rs`, `crates/kiln-model/src/backend/vulkan_dense.rs`, `crates/kiln-model/src/backend/vulkan_device.rs`, `crates/kiln-model/src/backend/vulkan_gdn.rs`, `crates/kiln-model/src/backend/vulkan_linear.rs`, `crates/kiln-model/src/backend/vulkan_residency.rs`, `crates/kiln-model/src/backend/vulkan_resources.rs`, `crates/kiln-model/src/backend/vulkan_tensor_bridge.rs`, `crates/kiln-model/src/backend/vulkan_training.rs`, `crates/kiln-model/src/backend/vulkan_weights.rs` | 0 | 21 | 32 | 0 |

## Focused Backend Facets

| Facet | Method Count | Forwarding Impl | Concrete Impl Count | Concrete Impls | Methods |
|---|---:|---|---:|---|---|
| `BackendIdentity` | 3 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_as_any`, `runtime_device`, `runtime_name` |
| `StartupBackend` | 1 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_precompile_startup_kernels` |
| `AttentionBackend` | 10 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_flash_attn_paged_decode`, `runtime_flash_attn_paged_decode_contiguous`, `runtime_flash_attn_paged_decode_contiguous_batch`, `runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen`, `runtime_flash_attn_prefill`, `runtime_flash_attn_prefill_head_major`, `runtime_supports_flash_attn_paged_decode`, `runtime_supports_flash_attn_prefill`, `runtime_supports_flash_attn_prefill_head_major`, `runtime_supports_strict_paged_decode_contiguous_batch` |
| `PagedKvBackend` | 4 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_paged_kv_head_major_read`, `runtime_paged_kv_head_major_read_append_token_major`, `runtime_supports_paged_kv_head_major_read`, `runtime_supports_paged_kv_head_major_read_append_token_major` |
| `GdnBackend` | 31 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_gdn_ab_in_proj_prefill`, `runtime_gdn_chunk_prep`, `runtime_gdn_chunk_scan`, `runtime_gdn_chunkwise_forward`, `runtime_gdn_decode_gates_recurrent`, `runtime_gdn_decode_gates_recurrent_rmsnorm`, `runtime_gdn_decode_qk_norm_gates_recurrent`, `runtime_gdn_decode_qk_norm_gates_recurrent_rmsnorm`, `runtime_gdn_forward_substitution`, `runtime_gdn_full_chunk_forward`, `runtime_gdn_full_chunk_forward_head_last_into`, `runtime_gdn_gated_rms_norm`, `runtime_gdn_gates`, `runtime_gdn_in_proj_decode`, `runtime_gdn_recurrent_prefill_head_last`, `runtime_gdn_recurrent_prefill_native_head_last`, `runtime_gdn_recurrent_qk_norm_prefill_native_head_last`, `runtime_gdn_recurrent_step`, `runtime_supports_gdn_chunk_prep`, `runtime_supports_gdn_chunk_scan`, `runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk`, `runtime_supports_gdn_decode_qk_norm_gates_recurrent`, `runtime_supports_gdn_forward_substitution`, `runtime_supports_gdn_full_chunk_forward`, `runtime_supports_gdn_full_chunk_forward_head_last`, `runtime_supports_gdn_gated_rms_norm`, `runtime_supports_gdn_gates`, `runtime_supports_gdn_recurrent_prefill_head_last`, `runtime_supports_gdn_recurrent_prefill_native_head_last`, `runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last`, `runtime_supports_gdn_recurrent_step` |
| `ConvBackend` | 4 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_causal_conv1d_prefill`, `runtime_causal_conv1d_update`, `runtime_supports_causal_conv1d_prefill`, `runtime_supports_causal_conv1d_update` |
| `LinearBackend` | 13 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_drop_uploaded_bf16_weights`, `runtime_full_attn_qkv_combined_decode`, `runtime_full_attn_qkv_decode`, `runtime_linear_decode`, `runtime_linear_prefill_apply`, `runtime_linear_prefill_apply_offset`, `runtime_lora_decode_add`, `runtime_lora_delta_resident`, `runtime_matmul`, `runtime_mlp_decode`, `runtime_mlp_gate_up_decode`, `runtime_prewarm_decode_weights`, `runtime_supports_matmul_request` |
| `SamplingBackend` | 8 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_linear_decode_argmax`, `runtime_linear_decode_argmax_batch`, `runtime_linear_decode_sample`, `runtime_linear_decode_sample_batch`, `runtime_supports_linear_decode_argmax`, `runtime_supports_linear_decode_argmax_batch`, `runtime_supports_linear_decode_sample`, `runtime_supports_linear_decode_sample_batch` |
| `ResidencyBackend` | 18 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_assemble_gdn_recurrent_resident_batch_rows`, `runtime_assemble_linear_attn_gdn_state_batch_kt`, `runtime_enter_gdn_recurrent_resident_state_scope`, `runtime_evict_gdn_recurrent_resident_state`, `runtime_evict_resident_activation`, `runtime_exit_gdn_recurrent_resident_state_scope`, `runtime_has_gdn_recurrent_resident_state`, `runtime_has_linear_attn_gdn_state_kt`, `runtime_has_resident_activation`, `runtime_materialize_gdn_recurrent_resident_state`, `runtime_register_resident_activation`, `runtime_resident_activation_resource`, `runtime_resolve_resident_activation`, `runtime_scatter_gdn_recurrent_resident_batch_rows`, `runtime_scatter_linear_attn_gdn_state_batch_kt`, `runtime_seed_linear_attn_gdn_state_kt`, `runtime_supports_resident_activation`, `runtime_update_resident_activation` |
| `OptimizerBackend` | 2 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_dispatch_adamw_step`, `runtime_dispatch_sgd_step` |
| `TrainingLossBackend` | 9 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_final_rmsnorm_backward_route`, `runtime_grpo_kl_auxiliary_route`, `runtime_grpo_loss_route`, `runtime_opd_loss_route`, `runtime_opd_phase_b_backward_route`, `runtime_sft_flce_loss_route`, `runtime_tape_forward_backward_route`, `runtime_training_capabilities`, `runtime_training_precision_policy` |
| `ReplayBackend` | 6 | `concrete_authoritative` | 5 | `CpuBackend`, `CudaBackend`, `MetalBackend`, `RocmBackend`, `VulkanBackend` | `runtime_decode_resident_pool_ready`, `runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs`, `runtime_replay_authority`, `runtime_replay_key_for_request`, `runtime_supports_replay_request`, `runtime_supports_resident_decode` |

## Replay Authority

| Backend | Production Authority | Native Primitive | Runners | Graph Crates | Parity Tests | Missing Evidence |
|---|---|---|---|---|---|---|
| `cuda` | `model_level_runner` | `CUDA graph` | `crates/kiln-model/src/cuda_graph.rs` | `crates/kiln-graph-cuda/src/lib.rs` | `test_cuda_graph_bs1_decode_matches_eager` | none |
| `rocm` | `model_level_runner` | `HIP graph` | `crates/kiln-model/src/rocm_graph.rs` | none | `ROCm graph runner byte-identical eager/replay source contract` | none |
| `metal` | `model_level_runner_with_graph_crate_replay_object` | `Metal ICB` | `crates/kiln-model/src/metal_graph.rs`, `crates/kiln-model/src/backend/metal_paged.rs` | `crates/kiln-graph-metal/src/lib.rs` | `test_metal_graph_bs1_decode_matches_eager_across_boundaries_and_buckets`, `test_metal_graph_batched_decode_matches_eager_and_replays_bucket`, `single_token_paged_decode_icb_matches_eager_and_updates_slot`, `batched_paged_decode_icb_matches_eager_and_updates_slots` | none |
| `vulkan` | `resident_decode_command_batch` | `Vulkan CommandBatch` | `crates/kiln-model/src/vk_decode_resident.rs`, `crates/kiln-vulkan-kernel/src/cmd_batch.rs` | `crates/kiln-graph-vulkan/src/lib.rs` | `vk_resident_decode_matches_nonresident_on_qwen35_4b` | none |

## Support Predicates

| Backend | Method | Predicate Status | Support State | Paired Method | Pair Always Declines | Gates |
|---|---|---|---|---|---|---|
| `cuda` | `runtime_supports_causal_conv1d_prefill` | `dynamic` | `NativeWithConstraints` | `runtime_causal_conv1d_prefill` | no | none |
| `cuda` | `runtime_supports_causal_conv1d_update` | `dynamic` | `NativeWithConstraints` | `runtime_causal_conv1d_update` | no | none |
| `cuda` | `runtime_supports_flash_attn_paged_decode` | `dynamic` | `NativeWithConstraints` | `runtime_flash_attn_paged_decode` | no | none |
| `cuda` | `runtime_supports_flash_attn_prefill` | `dynamic` | `NativeWithConstraints` | `runtime_flash_attn_prefill` | no | none |
| `cuda` | `runtime_supports_gdn_chunk_prep` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_chunk_prep` | no | none |
| `cuda` | `runtime_supports_gdn_chunk_scan` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_chunk_scan` | no | none |
| `cuda` | `runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk` | `dynamic` | `NativeWithConstraints` | `` | no | none |
| `cuda` | `runtime_supports_gdn_decode_qk_norm_gates_recurrent` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_decode_qk_norm_gates_recurrent` | no | none |
| `cuda` | `runtime_supports_gdn_forward_substitution` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_forward_substitution` | no | none |
| `cuda` | `runtime_supports_gdn_full_chunk_forward` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_full_chunk_forward` | no | none |
| `cuda` | `runtime_supports_gdn_gated_rms_norm` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_gated_rms_norm` | no | none |
| `cuda` | `runtime_supports_gdn_gates` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_gates` | no | none |
| `cuda` | `runtime_supports_gdn_recurrent_step` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_recurrent_step` | no | none |
| `cuda` | `runtime_supports_matmul_request` | `dynamic` | `NativeWithConstraints` | `` | no | none |
| `cuda` | `runtime_supports_resident_activation` | `literal_true` | `NativeWithConstraints` | `` | no | none |
| `cuda` | `runtime_supports_strict_paged_decode_contiguous_batch` | `dynamic` | `NativeWithConstraints` | `` | no | layout |
| `rocm` | `runtime_supports_causal_conv1d_prefill` | `dynamic` | `NativeWithConstraints` | `runtime_causal_conv1d_prefill` | no | none |
| `rocm` | `runtime_supports_causal_conv1d_update` | `dynamic` | `NativeWithConstraints` | `runtime_causal_conv1d_update` | no | none |
| `rocm` | `runtime_supports_flash_attn_paged_decode` | `dynamic` | `NativeWithConstraints` | `runtime_flash_attn_paged_decode` | no | none |
| `rocm` | `runtime_supports_flash_attn_prefill` | `dynamic` | `NativeWithConstraints` | `runtime_flash_attn_prefill` | no | none |
| `rocm` | `runtime_supports_flash_attn_prefill_head_major` | `env_gated` | `NativeWithConstraints` | `runtime_flash_attn_prefill_head_major` | no | env |
| `rocm` | `runtime_supports_gdn_chunk_prep` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_chunk_prep` | no | none |
| `rocm` | `runtime_supports_gdn_chunk_scan` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_chunk_scan` | no | none |
| `rocm` | `runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk` | `dynamic` | `NativeWithConstraints` | `` | no | none |
| `rocm` | `runtime_supports_gdn_decode_qk_norm_gates_recurrent` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_decode_qk_norm_gates_recurrent` | no | none |
| `rocm` | `runtime_supports_gdn_forward_substitution` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_forward_substitution` | no | none |
| `rocm` | `runtime_supports_gdn_full_chunk_forward` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_full_chunk_forward` | no | none |
| `rocm` | `runtime_supports_gdn_gated_rms_norm` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_gated_rms_norm` | no | none |
| `rocm` | `runtime_supports_gdn_gates` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_gates` | no | none |
| `rocm` | `runtime_supports_gdn_recurrent_step` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_recurrent_step` | no | none |
| `rocm` | `runtime_supports_matmul_request` | `dynamic` | `NativeWithConstraints` | `` | no | none |
| `rocm` | `runtime_supports_resident_activation` | `literal_true` | `NativeWithConstraints` | `` | no | none |
| `rocm` | `runtime_supports_strict_paged_decode_contiguous_batch` | `dynamic` | `NativeWithConstraints` | `` | no | layout |
| `metal` | `runtime_supports_causal_conv1d_prefill` | `dynamic` | `NativeWithConstraints` | `runtime_causal_conv1d_prefill` | no | none |
| `metal` | `runtime_supports_causal_conv1d_update` | `dynamic` | `NativeWithConstraints` | `runtime_causal_conv1d_update` | no | none |
| `metal` | `runtime_supports_flash_attn_paged_decode` | `literal_true` | `NativeWithConstraints` | `runtime_flash_attn_paged_decode` | no | none |
| `metal` | `runtime_supports_flash_attn_prefill` | `dynamic` | `NativeWithConstraints` | `runtime_flash_attn_prefill` | no | none |
| `metal` | `runtime_supports_flash_attn_prefill_head_major` | `dynamic` | `NativeWithConstraints` | `runtime_flash_attn_prefill_head_major` | no | none |
| `metal` | `runtime_supports_gdn_chunk_prep` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_chunk_prep` | no | none |
| `metal` | `runtime_supports_gdn_forward_substitution` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_forward_substitution` | no | none |
| `metal` | `runtime_supports_gdn_full_chunk_forward` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_full_chunk_forward` | no | none |
| `metal` | `runtime_supports_gdn_full_chunk_forward_head_last` | `dynamic` | `NativeWithConstraints` | `` | no | none |
| `metal` | `runtime_supports_gdn_gated_rms_norm` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_gated_rms_norm` | no | none |
| `metal` | `runtime_supports_gdn_gates` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_gates` | no | none |
| `metal` | `runtime_supports_gdn_recurrent_prefill_head_last` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_recurrent_prefill_head_last` | no | none |
| `metal` | `runtime_supports_gdn_recurrent_prefill_native_head_last` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_recurrent_prefill_native_head_last` | no | none |
| `metal` | `runtime_supports_gdn_recurrent_step` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_recurrent_step` | no | none |
| `metal` | `runtime_supports_linear_decode_sample` | `dynamic` | `NativeWithConstraints` | `runtime_linear_decode_sample` | no | none |
| `metal` | `runtime_supports_linear_decode_sample_batch` | `dynamic` | `NativeWithConstraints` | `runtime_linear_decode_sample_batch` | no | none |
| `metal` | `runtime_supports_matmul_request` | `dynamic` | `NativeWithConstraints` | `` | no | dtype |
| `metal` | `runtime_supports_paged_kv_head_major_read` | `literal_true` | `NativeWithConstraints` | `runtime_paged_kv_head_major_read` | no | none |
| `metal` | `runtime_supports_paged_kv_head_major_read_append_token_major` | `literal_true` | `NativeWithConstraints` | `runtime_paged_kv_head_major_read_append_token_major` | no | none |
| `metal` | `runtime_supports_resident_activation` | `literal_true` | `NativeWithConstraints` | `` | no | none |
| `vulkan` | `runtime_supports_causal_conv1d_prefill` | `dynamic` | `NativeWithConstraints` | `runtime_causal_conv1d_prefill` | no | none |
| `vulkan` | `runtime_supports_causal_conv1d_update` | `dynamic` | `NativeWithConstraints` | `runtime_causal_conv1d_update` | no | none |
| `vulkan` | `runtime_supports_flash_attn_paged_decode` | `dynamic` | `NativeWithConstraints` | `runtime_flash_attn_paged_decode` | no | none |
| `vulkan` | `runtime_supports_flash_attn_prefill` | `env_gated` | `NativeWithConstraints` | `runtime_flash_attn_prefill` | no | env |
| `vulkan` | `runtime_supports_flash_attn_prefill_head_major` | `literal_false` | `Declined` | `` | no | none |
| `vulkan` | `runtime_supports_gdn_chunk_prep` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_chunk_prep` | no | none |
| `vulkan` | `runtime_supports_gdn_chunk_scan` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_chunk_scan` | no | none |
| `vulkan` | `runtime_supports_gdn_forward_substitution` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_forward_substitution` | no | none |
| `vulkan` | `runtime_supports_gdn_full_chunk_forward` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_full_chunk_forward` | no | none |
| `vulkan` | `runtime_supports_gdn_gated_rms_norm` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_gated_rms_norm` | no | none |
| `vulkan` | `runtime_supports_gdn_gates` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_gates` | no | none |
| `vulkan` | `runtime_supports_gdn_recurrent_prefill_native_head_last` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_recurrent_prefill_native_head_last` | no | none |
| `vulkan` | `runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_recurrent_qk_norm_prefill_native_head_last` | no | none |
| `vulkan` | `runtime_supports_gdn_recurrent_step` | `dynamic` | `NativeWithConstraints` | `runtime_gdn_recurrent_step` | no | none |
| `vulkan` | `runtime_supports_linear_decode_argmax` | `dynamic` | `NativeWithConstraints` | `runtime_linear_decode_argmax` | no | none |
| `vulkan` | `runtime_supports_linear_decode_argmax_batch` | `dynamic` | `NativeWithConstraints` | `runtime_linear_decode_argmax_batch` | no | none |
| `vulkan` | `runtime_supports_linear_decode_sample` | `dynamic` | `NativeWithConstraints` | `runtime_linear_decode_sample` | no | none |
| `vulkan` | `runtime_supports_linear_decode_sample_batch` | `dynamic` | `NativeWithConstraints` | `runtime_linear_decode_sample_batch` | no | none |
| `vulkan` | `runtime_supports_matmul_request` | `dynamic` | `NativeWithConstraints` | `` | no | dtype |
| `vulkan` | `runtime_supports_resident_activation` | `literal_true` | `NativeWithConstraints` | `` | no | none |
| `vulkan` | `runtime_supports_resident_decode` | `dynamic` | `NativeWithConstraints` | `runtime_decode_resident_pool_ready` | no | none |

## Typed Request Descriptors

| Descriptor | Field Count | DType | Shape | Layout | Batch | Replay Safe | Fields |
|---|---:|---|---|---|---|---|---|
| `AttentionRequest` | 13 | yes | yes | yes | yes | yes | `kind`, `q_shape`, `k_shape`, `v_shape`, `output_shape`, `layout`, `q_dtype`, `k_dtype`, `v_dtype`, `batch`, `seq_len`, `head_dim`, `replay_safe` |
| `MatmulRequest` | 12 | yes | yes | yes | yes | yes | `lhs_shape`, `rhs_shape`, `lhs_dtype`, `rhs_dtype`, `out_dtype`, `accumulation`, `lhs_layout`, `rhs_layout`, `out_layout`, `batch`, `epilogue`, `replay_safe` |
| `MatmulBlasRequest` | 15 | yes | yes | yes | yes | yes | `m`, `n`, `k`, `dtype`, `lhs_dtype`, `rhs_dtype`, `out_dtype`, `accumulation`, `lhs_layout`, `rhs_layout`, `out_layout`, `epilogue`, `batch`, `replay_safe`, `concurrent_streams` |
| `LinearRequest` | 12 | yes | yes | yes | yes | yes | `kind`, `input_shape`, `weight_shape`, `output_shape`, `layout`, `input_dtype`, `weight_dtype`, `output_dtype`, `batch`, `top_k`, `temperatures`, `replay_safe` |
| `ReplayRequest` | 8 | yes | yes | yes | yes | yes | `kind`, `replay_shape`, `layout`, `max_hidden`, `max_intermediate`, `max_batch`, `dtype`, `replay_safe` |

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
| `BackendCapabilities` | 12 | `backend`, `device`, `storage`, `startup`, `matmul`, `attention`, `gdn`, `decode`, `decode_batcher`, `training`, `graph_replay`, `fallback` |
| `StorageCapabilities` | 13 | `backend`, `device`, `resident_activation`, `resident_decode`, `projection_load_policy`, `kv_cache_device_memory_pressure`, `gpu_memory_detection_policy`, `gpu_memory_budget_policy`, `gpu_allocator_memory_probe_policy`, `gpu_memory_reclaim_policy`, `kv_sizing_residency_model_multiplier`, `kv_auto_block_policy`, `kv_cache_fp8_policy` |
| `ProjectionLoadPolicy` | 18 | `backend`, `direct_transposed_upload_for_cached_weights`, `parallel_transposed_projection_upload`, `parallel_transposed_projection_upload_disable_env`, `parallel_auxiliary_weight_upload`, `parallel_auxiliary_weight_upload_disable_env`, `cache_full_attention_qkv_transpose_concat`, `cache_linear_attention_ab_transpose_concat`, `cache_mlp_gate_up_transpose_concat`, `pack_w8a16_projection_rows`, `stub_embedding_table_after_transposed_upload`, `drop_projection_originals`, `drop_projection_transposes`, `synchronize_after_dropping_originals`, `keep_projection_originals_env`, `drop_projection_originals_env`, `native_training_env`, `keep_projection_transposes_env` |
| `GpuMemoryDetectionPolicy` | 3 | `detected_total_log_message`, `missing_total_warning`, `missing_total_fallback_bytes` |
| `GpuMemoryBudgetPolicy` | 3 | `use_live_memory_snapshot`, `cap_kv_blocks_by_live_budget`, `retry_kv_allocation_after_reclaim` |
| `GpuAllocatorMemoryProbePolicy` | 1 | `probe` |
| `GpuMemoryReclaimPolicy` | 1 | `reclaimer` |
| `KvCacheAutoBlockPolicy` | 4 | `context_window_cap`, `static_max_blocks`, `memory_tier_cap`, `allow_min_blocks_below_live_budget` |
| `KvCacheMemoryTierBlockCap` | 5 | `low_memory_bytes_exclusive`, `low_max_blocks`, `mid_memory_bytes_exclusive`, `mid_max_blocks`, `high_max_blocks` |
| `KvCacheFp8Policy` | 3 | `allow_when_requested_by_default`, `explicit_enable_env`, `disabled_reason` |
| `StartupCapabilities` | 6 | `run_inference_prewarm`, `require_inference_prewarm_for_health`, `precompile_custom_kernels`, `native_training_default_enabled`, `native_training_env`, `decode_weight_prewarm_when_native_training` |
| `MatmulCapabilities` | 3 | `rank2_f32`, `batched_bf16`, `bias_epilogue` |
| `AttentionCapabilities` | 5 | `flash_prefill`, `flash_prefill_head_major`, `flash_paged_decode`, `flash_prefill_consumes_grouped_kv`, `detached_chunked_prefill` |
| `GdnCapabilities` | 10 | `recurrent_step`, `recurrent_step_f32`, `inference_recurrent_state`, `chunk_pre_permute_bf16`, `chunk_prep`, `chunk_scan`, `full_chunk_forward`, `gates`, `gated_rms_norm`, `gated_rms_norm_preserves_tape_residency` |
| `InferenceRecurrentStatePolicy` | 2 | `bf16`, `f16` |
| `DecodeCapabilities` | 8 | `resident_decode`, `paged_decode_graph_outputs`, `mtp_speculative_generation`, `speculative_policy`, `linear_argmax`, `linear_argmax_batch`, `linear_sample`, `linear_sample_batch` |
| `SpeculativeDecodePolicy` | 3 | `mtp_max_prompt_tokens`, `long_prompt_skip_layer_min_prompt_tokens`, `long_prompt_skip_layer_min_output_tokens` |
| `DecodeBatcherPolicy` | 17 | `max_batch`, `wait_micros`, `allow_mixed_seq_lens`, `rowwise_retry_env`, `require_native_decode_attention`, `prefer_direct_paged_decode_attention`, `direct_paged_decode_attention_env_gate`, `allow_prefix_cache_split_snapshot`, `paged_decode_requires_contiguous_kv_chunks`, `use_greedy_token_decode`, `use_native_sampled_contiguous_decode`, `sampled_contiguous_decode_requires_resident_decode`, `partition_noncontiguous_gdn_kv_tiles`, `use_decode_width_prefill_admission`, `burst_prefill_admission`, `batching_engine_default_enabled`, `warm_resident_decode_pool_on_startup` |
| `BackendTrainingCapabilities` | 4 | `hooks`, `precision`, `server_dispatch`, `acceleration_profile` |
| `ServerTrainingDispatchPolicy` | 3 | `native_route`, `native_training_env`, `native_training_default_enabled` |
| `TrainingAccelerationProfilePolicy` | 8 | `log_message`, `linear`, `sdpa`, `rmsnorm_inference`, `rmsnorm_training`, `flce_provider`, `resident_activation`, `sgd_step_on_device` |
| `TrainingAccelerationEnvFlagPolicy` | 2 | `env`, `default_on` |
| `ReplayCapabilities` | 3 | `resident_decode`, `paged_decode_graph_outputs`, `authority` |
| `ReplayAuthority` | 4 | `backend`, `production_authority`, `native_primitive`, `graph_crate_role` |
| `BackendFallbackCapabilities` | 5 | `generic_device_op`, `decode_hot_path`, `decode_hot_path_debug_env`, `training_optimizer`, `training_optimizer_debug_env` |

## Resident Resource Descriptors

| Descriptor | Field Count | Fields |
|---|---:|---|
| `ResidentResource` | 13 | `tensor_id`, `backend`, `device`, `dtype`, `shape`, `layout`, `element_count`, `byte_len`, `addressable_byte_len`, `family`, `ownership`, `state`, `replay_stability` |
| `ResidentResourceLayout` | 3 | `strides`, `start_offset`, `contiguous` |

## Conformance And Performance Gates

| Gate | Phase 8 Requirement | Status | Command | Supplemental Commands | Evidence | Missing Evidence | Coverage Blockers |
|---|---|---|---|---|---|---|---|
| `storage_round_trip` | storage round trip | `covered` | `cargo test -p kiln-tensor --features rocm --test rocm_storage_smoke && cargo test -p kiln-vulkan-kernel --test vk_tensor_parity` | `ROCm feature lane: cargo test -p kiln-tensor --features rocm --test rocm_storage_smoke`; `Vulkan kernel lane: cargo test -p kiln-vulkan-kernel --test vk_tensor_parity` | `crates/kiln-tensor/tests/rocm_storage_smoke.rs`, `crates/kiln-vulkan-kernel/tests/vk_tensor_parity.rs` | none | none |
| `host_transfer_to_device_parity` | host transfer / to_device parity with explicit unsupported errors | `covered` | `cargo test -p kiln-tensor device_transfer_support_classifies_explicit_transitions && cargo test -p kiln-tensor to_device_without_gpu_features_reports_explicit_unsupported_transition` | `CUDA hardware lane: CUDARC_CUDA_VERSION=12080 cargo test -p kiln-tensor --no-default-features --features cuda --test cuda_resize_copy_primitives`; `ROCm feature lane: cargo test -p kiln-tensor --features rocm --test rocm_compare_parity`; `macOS Metal feature lane: cargo test -p kiln-tensor --features metal --test metal_ops_parity`; `Vulkan kernel lane: cargo test -p kiln-vulkan-kernel --test vk_tensor_parity` | `crates/kiln-tensor/src/tensor.rs`, `crates/kiln-tensor/tests/cuda_resize_copy_primitives.rs`, `crates/kiln-tensor/tests/metal_ops_parity.rs`, `crates/kiln-tensor/tests/rocm_compare_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_tensor_parity.rs` | none | none |
| `device_op_parity` | DeviceOp parity | `covered` | `cargo test -p kiln-tensor device_op::tests` | `ROCm feature lane: cargo test -p kiln-tensor --features rocm --test rocm_scalar_op_parity`; `macOS Metal feature lane: cargo test -p kiln-tensor --features metal --test metal_ops_parity` | `crates/kiln-tensor/src/device_op.rs`, `crates/kiln-tensor/tests/rocm_scalar_op_parity.rs`, `crates/kiln-tensor/tests/metal_ops_parity.rs` | none | none |
| `matmul_linear_parity` | matmul/linear parity | `covered` | `cargo test -p kiln-model matmul_request_projects_to_blas_shape_contract && cargo test -p kiln-tensor --features rocm --test rocm_matmul_parity && cargo test -p kiln-tensor matmul_matrix_core && cargo test -p kiln-vulkan-kernel --test vk_matmul_parity && cargo test -p kiln-vulkan-kernel --test linear_decode_argmax && cargo test -p kiln-vulkan-kernel --test linear_decode_sample && cargo test -p kiln-model tape_forward_matmul_bit_exact_parity_with_baseline && CUDARC_CUDA_VERSION=12080 cargo check -p kiln-blas --features cublaslt --tests && cargo test -p kiln-model --test backend_capability_contract` | `CUDA cublasLt hardware lane: CUDARC_CUDA_VERSION=12080 cargo test -p kiln-blas --features cublaslt --test cublaslt_handle_smoke`; `macOS Metal feature lane: cargo test -p kiln-tensor --features metal --test metal_ops_parity` | `crates/kiln-model/src/backend/capability.rs`, `crates/kiln-model/src/backend/mod.rs`, `crates/kiln-blas/tests/cublaslt_handle_smoke.rs`, `crates/kiln-tensor/tests/rocm_matmul_parity.rs`, `crates/kiln-tensor/tests/metal_ops_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs`, `crates/kiln-vulkan-kernel/tests/linear_decode_argmax.rs`, `crates/kiln-vulkan-kernel/tests/linear_decode_sample.rs`, `crates/kiln-model/tests/tape_forward_parity.rs`, `crates/kiln-model/tests/marlin_qproj_parity.rs` | none | none |
| `attention_gdn_conv_parity` | attention/GDN/conv parity | `covered` | `cargo test -p kiln-model --no-default-features --features rocm --test rocm_flash_attn_bwd_gradcheck && cargo test -p kiln-flash-attn --no-default-features --features rocm --test rocm_flash_attn_parity && cargo test -p kiln-gdn-kernel --no-default-features --features rocm --test rocm_gdn_parity && cargo test -p kiln-conv1d-kernel --no-default-features --features rocm --test rocm_conv1d_parity && cargo test -p kiln-vulkan-kernel --test vk_attention_parity && cargo test -p kiln-vulkan-kernel --test vk_sdpa_prefill_kernel_parity && cargo test -p kiln-vulkan-kernel --test vk_gdn_foundation_parity && cargo test -p kiln-vulkan-kernel --test vk_gdn_backward_parity && cargo test -p kiln-vulkan-kernel --test gdn_parity` | none | `crates/kiln-model/tests/rocm_flash_attn_bwd_gradcheck.rs`, `crates/kiln-flash-attn/tests/rocm_flash_attn_parity.rs`, `crates/kiln-gdn-kernel/tests/rocm_gdn_parity.rs`, `crates/kiln-conv1d-kernel/tests/rocm_conv1d_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_attention_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_sdpa_prefill_kernel_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_gdn_foundation_parity.rs`, `crates/kiln-vulkan-kernel/tests/vk_gdn_backward_parity.rs`, `crates/kiln-vulkan-kernel/tests/gdn_parity.rs` | none | none |
| `optimizer_parity` | optimizer parity | `covered` | `cargo test -p kiln-optim --test integration && cargo test -p kiln-train training_optimizer && cargo test -p kiln-model --test backend_capability_contract` | `CUDA plus Vulkan OPD hardware lane: CUDARC_CUDA_VERSION=12080 cargo test -p kiln-train --features cuda,vulkan --test vk_cuda_opd_parity` | `crates/kiln-optim/tests/integration.rs`, `crates/kiln-model/src/backend/cuda.rs`, `crates/kiln-model/src/backend/rocm.rs`, `crates/kiln-model/src/backend/metal_training.rs`, `crates/kiln-model/src/backend/vulkan.rs`, `crates/kiln-model/src/backend/vulkan_training.rs`, `crates/kiln-model/src/backend/mod.rs`, `crates/kiln-train/src/trainer.rs`, `crates/kiln-train/tests/vk_cuda_opd_parity.rs` | none | none |
| `replay_parity` | replay parity | `covered` | `cargo test -p kiln-graph replay && cargo test -p kiln-graph --test capture_lifetime && cargo test -p kiln-graph-cuda replay && cargo test -p kiln-graph-metal replay && cargo test -p kiln-graph-vulkan replay && cargo test -p kiln-model --features vulkan --test vk_resident_decode_parity && cargo test -p kiln-tensor --features rocm --test rocm_capture_arena && cargo test -p kiln-model --test backend_capability_contract` | none | `crates/kiln-graph/src/replay_plan.rs`, `crates/kiln-graph/src/captured_graph.rs`, `crates/kiln-graph/tests/capture_lifetime.rs`, `crates/kiln-graph-cuda/src/lib.rs`, `crates/kiln-graph-metal/src/lib.rs`, `crates/kiln-graph-vulkan/src/lib.rs`, `crates/kiln-model/src/backend/capability.rs`, `crates/kiln-model/src/backend/residency.rs`, `crates/kiln-model/tests/vk_resident_decode_parity.rs`, `crates/kiln-tensor/tests/rocm_capture_arena.rs` | none | none |
| `one_step_training_proof` | one-step training proof | `covered` | `cargo test -p kiln-optim --test end_to_end_training && cargo test -p kiln-model --test backend_capability_contract` | `CUDA hardware lane: CUDARC_CUDA_VERSION=12080 cargo test -p kiln-model --features cuda --test cuda_sft_step_proof`; `ROCm feature lane: cargo test -p kiln-model --features rocm --test rocm_sft_step_proof`; `macOS Metal feature lane: cargo test -p kiln-model --features metal --test metal_sft_step_proof`; `Vulkan hardware opt-in lane: KILN_TENSOR_VULKAN_TEST=1 KILN_USE_TAPE_FORWARD=1 KILN_USE_TAPE_LORA_ADD=1 cargo test -p kiln-model --features vulkan --test vk_sft_step_proof` | `crates/kiln-model/tests/cuda_sft_step_proof.rs`, `crates/kiln-model/tests/metal_sft_step_proof.rs`, `crates/kiln-model/tests/vk_sft_step_proof.rs`, `crates/kiln-model/tests/rocm_sft_step_proof.rs`, `crates/kiln-optim/tests/end_to_end_training.rs` | none | none |
| `no_unexpected_host_fallback` | no unexpected host fallback in decode/training hot paths | `covered` | `cargo test -p kiln-tensor device_op_host_fallback_counts_are_backend_and_arity_specific` | none | `crates/kiln-tensor/src/device_op.rs`, `crates/kiln-model/src/generate.rs`, `crates/kiln-train/src/trainer.rs` | none | none |
| `decode_submit_or_replay_count` | max submit count or replay count per decode token | `covered` | `cargo test -p kiln-model decode_batcher_stats_report_runner_calls_per_token && cargo test -p kiln-server test_metrics_render && cargo test -p kiln-graph replay` | none | `crates/kiln-model/src/generate.rs`, `crates/kiln-server/src/metrics.rs`, `crates/kiln-server/src/api/health.rs`, `crates/kiln-server/src/api/debug_model_state.rs`, `crates/kiln-graph/src/captured_graph.rs`, `crates/kiln-graph/src/replay_plan.rs` | none | none |
| `matmul_algorithm_cache_reporting` | matmul algorithm/cache hit reporting | `covered` | `cargo test -p kiln-blas cache_stats_reports_entries_and_hit_rate && cargo test -p kiln-rocblas cache_stats_reports_entries_and_hit_rate && CUDARC_CUDA_VERSION=12080 cargo check -p kiln-blas --features cublaslt --tests && cargo check -p kiln-rocblas --features hipblaslt --tests` | none | `crates/kiln-blas/src/algo_cache.rs`, `crates/kiln-blas/src/cublaslt_handle.rs`, `crates/kiln-blas/tests/cublaslt_handle_smoke.rs`, `crates/kiln-rocblas/src/algo_cache.rs`, `crates/kiln-rocblas/src/hipblaslt_handle.rs` | none | none |
| `hardware_latency_thresholds` | backend-specific latency thresholds on known hardware fixtures | `covered` | `python3 scripts/run_backend_latency_fixture.py --self-test && python3 scripts/write_backend_latency_result_artifact.py --self-test && python3 scripts/import_backend_latency_artifact.py --self-test && python3 scripts/lock_backend_latency_thresholds.py --self-test && python3 scripts/check_backend_latency_fixtures.py --self-test && python3 scripts/plan_backend_latency_fixture_dispatch.py --self-test && python3 scripts/check_backend_latency_fixtures.py docs/backend-latency-fixtures.json --require-covered` | none | `docs/backend-latency-fixtures.json`, `docs/backend-latency-result-schema.md`, `scripts/check_unification_gates.sh`, `scripts/run_backend_latency_fixture.py`, `scripts/write_backend_latency_result_artifact.py`, `scripts/import_backend_latency_artifact.py`, `scripts/lock_backend_latency_thresholds.py`, `scripts/check_backend_latency_fixtures.py`, `scripts/plan_backend_latency_fixture_dispatch.py`, `crates/kiln-tensor/tests/cuda_latency_bench.rs`, `crates/kiln-server/examples/flce_preflight_bench.rs`, `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`, `crates/kiln-tensor/tests/metal_matmul_bench.rs`, `crates/kiln-tensor/tests/metal_sdpa_bench.rs`, `crates/kiln-vulkan-kernel/src/bin/vulkan_decode_microbench.rs`, `crates/kiln-tensor/tests/rocm_latency_bench.rs` | none | none |
| `generated_capability_dashboard` | generated capability dashboard checked into docs or build artifacts | `covered` | `python3 scripts/generate_backend_capability_report.py --self-test && python3 scripts/generate_backend_capability_report.py --check` | none | `docs/backend-capability-report.md`, `docs/backend-capability-report.json`, `scripts/generate_backend_capability_report.py`, `scripts/check_unification_gates.sh` | none | none |

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

| Backend | Policy | Activations | Base Weights | LoRA | Loss Accum | Optimizer Params | Mixed RMSNorm Weight | Streaming Tile | Tape Tile | Paged Medium Tile | Exact GDN Backward Tile | Mixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cpu` | `cpu_f32_reference` | `F32` | `F32` | `F32` | `F32` | `F32` | `none` | `8192` | `8192` | `none` | `streaming_tile_tokens_for(device)` | no |
| `cuda` | `cuda_native_float` | `F32,BF16,F16` | `F32,BF16,F16` | `F32,BF16` | `F32` | `F32,BF16` | `none` | `1024` | `1024` | `none` | `1024` | yes |
| `rocm` | `rocm_native_float` | `F32,BF16,F16` | `F32,BF16,F16` | `F32,BF16` | `F32` | `F32,BF16` | `none` | `1024` | `1024` | `1024 <= 20000` | `streaming_tile_tokens_for(device)` | yes |
| `metal` | `metal_bf16_uma` | `BF16` | `BF16` | `F32,BF16` | `F32` | `F32,BF16` | `none` | `2048` | `2048` | `none` | `streaming_tile_tokens_for(device)` | yes |
| `vulkan` | `vulkan_mixed_f32_bf16` | `F32` | `F32,BF16` | `F32` | `F32` | `F32,BF16` | `BF16` | `2048` | `2048` | `none` | `streaming_tile_tokens_for(device)` | yes |

## Training Loss Routing

| Backend | Tape Forward/Backward Route | SFT FLCE Route | GRPO Route | GRPO KL Auxiliary Route | OPD Route | OPD Phase-B Backward Route | Final RMSNorm Backward Route | Evidence |
|---|---|---|---|---|---|---|---|---|
| `cpu` | `unsupported` | `full_logits` | `kt_composite` | `host_composite` | `unsupported` | `unsupported` | `kt_composite` | TrainingCapabilities::portable keeps tape forward/backward unsupported, SFT on the portable full-logits loss path, GRPO on the shared kt composite loss root, GRPO KL auxiliaries on the host-composite route, OPD unsupported on the portable backend surface, and final RMSNorm backward on the kt-composite route |
| `cuda` | `kt_tape_authoritative` | `kt_tape_flce` | `kt_composite` | `cuda_rocm_device_fast_path` | `kt_tape_phase_b` | `cuda_rocm_fused_unit_grad` | `cuda_rocm_fused_tail` | CudaBackend::training_capabilities_static advertises kt tape-authoritative forward/backward, kt-tape FLCE over CUDA tensors, the shared kt GRPO composite route, CUDA/ROCm device fast paths for GRPO KL auxiliaries, the shared kt-tape OPD Phase-B route, the fused CUDA/ROCm Phase-B hidden-gradient leaf, and the fused final-RMSNorm tail route |
| `rocm` | `kt_tape_authoritative` | `kt_tape_flce` | `kt_composite` | `cuda_rocm_device_fast_path` | `kt_tape_phase_b` | `cuda_rocm_fused_unit_grad` | `cuda_rocm_fused_tail` | RocmBackend::training_capabilities_static advertises kt tape-authoritative forward/backward, the shared kt-tape FLCE route over ROCm tensors, the shared kt GRPO composite route, CUDA/ROCm device fast paths for GRPO KL auxiliaries, the shared kt-tape OPD Phase-B route, the fused CUDA/ROCm Phase-B hidden-gradient leaf, and the fused final-RMSNorm tail route |
| `metal` | `kt_tape_authoritative` | `full_logits` | `kt_composite` | `host_composite` | `kt_tape_phase_b` | `kt_composite` | `kt_composite` | Metal training capabilities advertise kt tape-authoritative forward/backward, inherit the portable full-logits SFT loss route, shared kt GRPO composite route, host-composite GRPO KL auxiliaries, shared kt-tape OPD Phase-B route, device-agnostic kt composite Phase-B backward, and kt-composite final RMSNorm backward |
| `vulkan` | `kt_tape_authoritative` | `vulkan_active_rows` | `vulkan_active_rows` | `host_composite` | `vulkan_active_hidden` | `vulkan_active_hidden` | `kt_composite` | Vulkan training capabilities advertise kt tape-authoritative forward/backward, active-row fused SFT/GRPO shader routes, host-composite GRPO KL auxiliaries, the active-hidden fused OPD loss/backward shader route, and kt-composite final RMSNorm backward |

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
- `KILN_DISABLE_CUDA_FULL_ATTN_QKV_IN_PROJ`
- `KILN_DISABLE_CUDA_GDN_AB_IN_PROJ`
- `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT`
- `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM`
- `KILN_DISABLE_CUDA_GDN_PREFILL_AB_IN_PROJ`
- `KILN_DISABLE_CUDA_GDN_PREFILL_GATES`
- `KILN_DISABLE_CUDA_LORA_DECODE_ADD`

### ROCM
- `KILN_DISABLE_ROCM_FULL_ATTN_QKV_IN_PROJ`
- `KILN_DISABLE_ROCM_GDN_AB_IN_PROJ`
- `KILN_DISABLE_ROCM_GDN_DECODE_QK_NORM_RECURRENT`
- `KILN_DISABLE_ROCM_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM`
- `KILN_DISABLE_ROCM_GDN_PREFILL_AB_IN_PROJ`
- `KILN_DISABLE_ROCM_GDN_PREFILL_GATES`
- `KILN_DISABLE_ROCM_LORA_DECODE_ADD`
- `KILN_ROCM_GDN_FULL_CHUNK_FORWARD_MULTIBLOCK`
- `KILN_ROCM_HEAD_MAJOR_PREFILL`

Legacy aliases honored for compatibility:
- `KILN_DISABLE_CUDA_FULL_ATTN_QKV_IN_PROJ`
- `KILN_DISABLE_CUDA_GDN_AB_IN_PROJ`
- `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT`
- `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM`
- `KILN_DISABLE_CUDA_GDN_PREFILL_AB_IN_PROJ`
- `KILN_DISABLE_CUDA_GDN_PREFILL_GATES`
- `KILN_DISABLE_CUDA_LORA_DECODE_ADD`

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

