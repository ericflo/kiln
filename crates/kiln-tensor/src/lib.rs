//! kiln-tensor — in-house Tensor + Storage substrate for kiln.
//!
//! # Status
//!
//! **Phase 1.1 scaffold** — see [GitHub issue #1082][epic] for the full
//! migration plan. Today this crate ships only [`Error`], [`Result`],
//! and the [`bail!`] macro. Subsequent Phase 1 PRs add `DType`,
//! `TensorId`, `Layout`, `Storage`, and `Tensor`.
//!
//! # Public-API stability
//!
//! `kiln-tensor` is **internal-only — no semver commitment**. Other
//! `kiln-*` crates may break against minor version bumps until candle is
//! removed and the surface stabilizes (Phase 7 of #1082). Once Phase 7
//! lands and the migration is complete, the public API of this crate
//! freezes against the next major version bump.
//!
//! [epic]: https://github.com/ericflo/kiln/issues/1082

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod activation_registry;
mod allocator;
#[cfg(any(feature = "cuda", feature = "rocm", test))]
mod blaslt_request;
mod cpu_allocator;
mod determinism;
mod device;
mod device_op;
mod dtype;
mod element;
mod error;
mod layout;
mod method_api;
mod operators;
pub mod ops;
pub mod probe;
pub mod profile;
pub mod safetensors;
mod shape;
mod storage;
mod stream_planner;
mod tensor;
mod tensor_id;

#[cfg(feature = "rocm")]
mod active_rocm_stream;
#[cfg(feature = "cuda")]
mod active_stream;
#[cfg(feature = "cuda")]
mod capture_alloc;
#[cfg(feature = "cuda")]
mod cuda_allocator;
#[cfg(feature = "cuda")]
mod cuda_matmul;
#[cfg(feature = "cuda")]
mod cuda_storage;
#[cfg(feature = "cuda")]
mod cuda_stream_priority;
#[cfg(feature = "cuda")]
mod fp8;
#[cfg(feature = "metal")]
mod metal_allocator;
#[cfg(feature = "metal")]
mod metal_kernels;
#[cfg(feature = "metal")]
mod metal_matmul;
#[cfg(feature = "metal")]
mod metal_rt;
#[cfg(feature = "metal")]
mod metal_storage;
#[cfg(feature = "metal")]
pub mod metal_types;
#[cfg(feature = "rocm")]
mod rocm_allocator;
#[cfg(feature = "rocm")]
mod rocm_capture_alloc;
#[cfg(feature = "rocm")]
mod rocm_matmul;
#[cfg(feature = "rocm")]
mod rocm_ops;
#[cfg(feature = "rocm")]
mod rocm_storage;
#[cfg(feature = "vulkan")]
pub mod vk_shaders;
#[cfg(feature = "vulkan")]
mod vulkan_allocator;
#[cfg(feature = "vulkan")]
mod vulkan_storage;

pub use activation_registry::{
    Activation, ActivationId, ActivationKind, ActivationRef, OffloadPolicy,
    RecomputeRecommendation, selective_recompute_recommendation,
};
pub use allocator::{Allocator, AllocatorMode, allocator_frozen_error};
pub use cpu_allocator::CpuAllocator;
pub use determinism::{
    DETERMINISTIC_CACHED, Determinism, DeterministicCache, deterministic_enabled,
};
pub use device::{Backend, Device, DeviceLocation};
pub use device_op::{BackwardOp, DeviceOp1, DeviceOp2, DeviceOp3, dispatch1, dispatch2, dispatch3};
pub use dtype::DType;
pub use element::Element;
pub use error::{Error, Result};
pub use layout::Layout;
pub use method_api::{ArangeScalar, D, Dim};
pub use probe::{cuda_is_available, metal_is_available};
pub use shape::Shape;
pub use storage::{CpuStorage, Storage, StorageBackend, cpu_zeros};
pub use stream_planner::{StreamId, StreamPlanner, StreamRecord};
pub use tensor::{DeviceTransferSupport, Tensor, device_transfer_support};
pub use tensor_id::TensorId;

#[cfg(feature = "rocm")]
pub use active_rocm_stream::{active_rocm_stream, with_active_rocm_stream};
#[cfg(feature = "cuda")]
pub use active_stream::{active_cuda_stream, with_active_cuda_stream};
#[cfg(feature = "cuda")]
pub use capture_alloc::{
    CaptureArena, capture_arena_active, capture_arena_alloc, with_capture_arena,
};
#[cfg(feature = "cuda")]
pub use cuda_allocator::CudaAllocator;
#[cfg(feature = "cuda")]
pub use cuda_matmul::{
    cublaslt_cache_path, cuda_matmul, cuda_matmul_into, cuda_matmul_lhs_transposed,
    cuda_matmul_rhs_transposed, cuda_matmul_to_dtype, cuda_matmul_with_bias,
    flush_algo_cache_to_disk, load_algo_cache_from_disk, restore_into_shared_cache,
    snapshot_algo_cache,
};
#[cfg(feature = "cuda")]
pub use cuda_storage::{
    CudaStorage, cuda_activation_unary, cuda_argmax_last_axis, cuda_binary_minmax,
    cuda_bool_reduce_axis, cuda_cast, cuda_clamp_pow, cuda_compare, cuda_concat, cuda_contiguous,
    cuda_cross_entropy_loss, cuda_cumprod_axis, cuda_cumsum_axis, cuda_diag_build,
    cuda_diagonal_extract, cuda_dropout, cuda_elementwise_binary,
    cuda_flce_grad_logits_chunk_inplace, cuda_grpo_grad_logits_chunk_inplace,
    cuda_index_select_axis_n, cuda_index_select_dim0, cuda_is_finite, cuda_l2norm_last_axis,
    cuda_layernorm_last_axis, cuda_lerp, cuda_log_softmax_last_axis,
    cuda_log_softmax_last_axis_f32, cuda_masked_fill, cuda_max_axis, cuda_mean_axis,
    cuda_mean_last_axis, cuda_mem_get_info, cuda_min_axis, cuda_rmsnorm_last_axis, cuda_rope,
    cuda_rope_split_half, cuda_scalar_op, cuda_scatter_add_dim0, cuda_set_pool_release_threshold,
    cuda_slice_set_dim0, cuda_softmax_last_axis, cuda_sum_axis, cuda_sum_last_axis,
    cuda_sum_squared_last_axis, cuda_synchronize_default_stream, cuda_to_host_copy,
    cuda_topk_last_axis, cuda_trim_pool, cuda_where_select, cuda_write_host_in_place,
    cuda_zeros_ctx, host_to_cuda_copy, host_to_cuda_copy_ctx, primary_cuda_context,
};
#[cfg(feature = "cuda")]
pub use cuda_stream_priority::{
    PrioritizedCudaStream, StreamPriority, cuda_stream_priority_range,
    new_cuda_stream_with_priority,
};
#[cfg(feature = "cuda")]
pub use fp8::{
    E4M3_MAX, cuda_fp8_dequantize, cuda_fp8_dequantize_direct, cuda_fp8_quantize,
    cuda_fp8_quantize_direct, cuda_fp8_quantize_with_scale,
};
#[cfg(feature = "metal")]
pub use metal_allocator::MetalAllocator;
#[doc(hidden)]
#[cfg(feature = "metal")]
pub use metal_matmul::{
    GemmCfg, bench_gemm_cfg, bench_steel_cfg, gemm_pool_bytes, gen_gemm_msl, gen_steel_msl,
    steel_cfg_valid,
};
#[cfg(feature = "metal")]
pub use metal_matmul::{metal_matmul, metal_matmul_lhs_transposed, metal_matmul_rhs_transposed};
#[cfg(feature = "metal")]
pub use metal_storage::{
    MetalStorage, host_to_metal_copy, metal_activation_unary, metal_adamw_step, metal_cast,
    metal_compare, metal_copy_in_place, metal_cumsum_axis, metal_deep_copy,
    metal_elementwise_binary, metal_index_select_dim0, metal_layernorm_last_axis,
    metal_log_softmax_last_axis, metal_log_softmax_last_axis_f32, metal_muon_step,
    metal_rmsnorm_last_axis, metal_sdpa_last_axis, metal_softmax_last_axis, metal_to_host_copy,
    metal_where_select, metal_write_host_in_place, primary_metal_companion,
};
#[cfg(feature = "rocm")]
pub use rocm_allocator::RocmAllocator;
#[cfg(feature = "rocm")]
pub use rocm_capture_alloc::{
    RocmCaptureArena, rocm_capture_arena_active, rocm_capture_arena_alloc, with_rocm_capture_arena,
};
#[cfg(feature = "rocm")]
pub use rocm_matmul::{
    rocm_matmul, rocm_matmul_into, rocm_matmul_lhs_transposed, rocm_matmul_lhs_transposed_to_dtype,
    rocm_matmul_rhs_transposed, rocm_matmul_rhs_transposed_to_dtype, rocm_matmul_to_dtype,
    rocm_matmul_with_bias,
};
#[cfg(feature = "rocm")]
pub use rocm_ops::*;
#[cfg(feature = "rocm")]
pub use rocm_storage::{
    RocmStorage, host_to_rocm_copy, host_to_rocm_copy_ctx, primary_rocm_context, rocm_contiguous,
    rocm_htod_count, rocm_is_available, rocm_log_softmax_last_axis, rocm_log_softmax_last_axis_f32,
    rocm_mem_get_info, rocm_pool_stats, rocm_slice_set_dim0, rocm_softmax_last_axis,
    rocm_synchronize_compute_stream, rocm_synchronize_default_stream,
    rocm_synchronize_tensor_stream, rocm_to_host_copy, rocm_trim_pool, rocm_write_host_in_place,
    rocm_zeros_ctx,
};
#[cfg(feature = "vulkan")]
pub use vulkan_allocator::VulkanAllocator;
#[cfg(feature = "vulkan")]
pub use vulkan_storage::{
    VulkanStorage, host_to_vulkan_copy, kt_tensor_from_vk, primary_vulkan_device,
    vk_tensor_from_kt, vulkan_activation_unary, vulkan_argmax_last_axis, vulkan_cast,
    vulkan_contiguous, vulkan_elementwise_binary, vulkan_index_select_dim0,
    vulkan_l2norm_last_axis, vulkan_masked_fill, vulkan_matmul, vulkan_matmul_batched,
    vulkan_matmul_bf16w, vulkan_matmul_bf16w_bwd, vulkan_matmul_lhs_transposed,
    vulkan_matmul_rhs_transposed, vulkan_mean_all, vulkan_rmsnorm_last_axis, vulkan_scale,
    vulkan_slice_set_dim0, vulkan_softmax_last_axis, vulkan_sum_all, vulkan_synchronize_queue,
    vulkan_to_host_copy, vulkan_unary_math, vulkan_zeros,
};
