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
mod cpu_allocator;
mod determinism;
mod device;
mod device_op;
mod dtype;
mod element;
mod error;
mod layout;
pub mod ops;
pub mod probe;
pub mod profile;
pub mod safetensors;
mod storage;
mod stream_planner;
mod tensor;
mod tensor_id;

#[cfg(feature = "cuda")]
mod cuda_allocator;
#[cfg(feature = "cuda")]
mod cuda_storage;
#[cfg(feature = "cuda")]
mod cuda_matmul;
#[cfg(feature = "cuda")]
mod fp8;
#[cfg(feature = "metal")]
mod metal_allocator;
#[cfg(feature = "metal")]
mod metal_storage;
#[cfg(feature = "metal")]
pub mod metal_types;
#[cfg(feature = "vulkan")]
mod vulkan_allocator;
#[cfg(feature = "vulkan")]
mod vulkan_storage;
#[cfg(feature = "vulkan")]
pub mod vk_shaders;

pub use activation_registry::{
    selective_recompute_recommendation, Activation, ActivationId, ActivationKind, ActivationRef,
    OffloadPolicy, RecomputeRecommendation,
};
pub use allocator::{allocator_frozen_error, Allocator, AllocatorMode};
pub use cpu_allocator::CpuAllocator;
pub use determinism::{deterministic_enabled, Determinism, DeterministicCache, DETERMINISTIC_CACHED};
pub use device::{Backend, Device};
pub use device_op::{dispatch1, dispatch2, dispatch3, BackwardOp, DeviceOp1, DeviceOp2, DeviceOp3};
pub use dtype::DType;
pub use element::Element;
pub use error::{Error, Result};
pub use layout::Layout;
pub use probe::{cuda_is_available, metal_is_available};
pub use storage::{cpu_zeros, CpuStorage, Storage, StorageBackend};
pub use stream_planner::{StreamId, StreamPlanner, StreamRecord};
pub use tensor::Tensor;
pub use tensor_id::TensorId;

#[cfg(feature = "cuda")]
pub use cuda_allocator::CudaAllocator;
#[cfg(feature = "cuda")]
pub use cuda_storage::{
    cuda_activation_unary, cuda_argmax_last_axis, cuda_binary_minmax, cuda_bool_reduce_axis, cuda_cast,
    cuda_clamp_pow, cuda_compare, cuda_concat, cuda_contiguous, cuda_cross_entropy_loss,
    cuda_cumsum_axis, cuda_cumprod_axis, cuda_dropout, cuda_elementwise_binary,
    cuda_index_select_axis_n, cuda_index_select_dim0, cuda_l2norm_last_axis, cuda_lerp,
    cuda_layernorm_last_axis, cuda_masked_fill, cuda_max_axis, cuda_mean_axis, cuda_mean_last_axis, cuda_min_axis,
    cuda_rmsnorm_last_axis, cuda_rope, cuda_scalar_op, cuda_scatter_add_dim0, cuda_sum_axis,
    cuda_diag_build, cuda_diagonal_extract, cuda_softmax_last_axis, cuda_sum_last_axis,
    cuda_is_finite, cuda_sum_squared_last_axis, cuda_to_host_copy, cuda_where_select,
    cuda_zeros_ctx, host_to_cuda_copy, host_to_cuda_copy_ctx, primary_cuda_context,
    CudaStorage,
};
#[cfg(feature = "cuda")]
pub use cuda_matmul::{
    cuda_matmul, cuda_matmul_into, cuda_matmul_with_bias, snapshot_algo_cache,
};
#[cfg(feature = "cuda")]
pub use fp8::{
    cuda_fp8_dequantize, cuda_fp8_dequantize_direct, cuda_fp8_quantize,
    cuda_fp8_quantize_direct, cuda_fp8_quantize_with_scale, E4M3_MAX,
};
#[cfg(feature = "metal")]
pub use metal_allocator::MetalAllocator;
#[cfg(feature = "metal")]
pub use metal_storage::{
    metal_activation_unary, metal_cast, metal_elementwise_binary, metal_index_select_dim0,
    metal_layernorm_last_axis, metal_rmsnorm_last_axis, metal_softmax_last_axis,
    primary_metal_companion, primary_metal_device, MetalStorage,
};
#[cfg(feature = "vulkan")]
pub use vulkan_allocator::VulkanAllocator;
#[cfg(feature = "vulkan")]
pub use vulkan_storage::{
    vulkan_activation_unary, vulkan_argmax_last_axis, vulkan_cast,
    vulkan_elementwise_binary, vulkan_index_select_dim0, vulkan_l2norm_last_axis,
    vulkan_masked_fill, vulkan_rmsnorm_last_axis, vulkan_softmax_last_axis, vulkan_zeros,
    VulkanStorage,
};
