//! kiln-vulkan-blas — Vulkan BLAS layer (extending
//! `kiln-vulkan-kernel::vk_ops/matmul*`).
//!
//! Sibling to [`kiln-blas`] (CUDA) and [`kiln-mps`] (Metal). Per the
//! Phase 2 issue bullet:
//!
//! > BLAS crate: `kiln-blas` (cublasLt) / `kiln-mps`
//! > (MPSMatrixMultiplication + transposed-coop GEMV port) /
//! > **`kiln-vulkan-blas` (extend `kiln-vulkan-kernel::vk_ops/matmul*`)**
//!
//! Unlike kiln-blas + kiln-mps, kiln-vulkan-blas does **not** carry a
//! candle dependency — kiln-vulkan-kernel is already candle-free.
//! Phase 7 of #1082 (candle removal) leaves this crate's dependency
//! graph unchanged.
//!
//! # Phase 2.3 scope
//!
//! Backend-agnostic Vulkan-specific extensions only:
//!
//! - [`VkWorkgroupConfig`] — workgroup-size + subgroup-config cache
//!   entry. Recorded in `kiln_blas::AlgoCacheValue::algo_blob` as a
//!   stable byte format.
//! - [`VkPipelineCacheKey`] — hash key for the `~/.cache/kiln/vulkan/
//!   pipeline-cache-{device_uuid}.bin` blob per the Phase 2 issue's
//!   "Vulkan: disk-persistent pipeline cache" bullet.
//! - [`VkCooperativeMatrixSupport`] — feature-detection enum for
//!   `VK_KHR_cooperative_matrix` per the Phase 2 issue's "Vulkan:
//!   `VK_KHR_cooperative_matrix` for the matmul backend" bullet.
//!
//! # CPU-buildable
//!
//! All types compile without `--features vulkan`. The actual matmul
//! wrapper extending `kiln-vulkan-kernel`'s existing compute pipelines
//! lands behind the `vulkan` feature in subsequent PRs.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod backend_matmul;
mod cooperative_matrix;
mod pipeline_cache;
mod workgroup;

pub use backend_matmul::VulkanBackendMatmul;
pub use cooperative_matrix::VkCooperativeMatrixSupport;
pub use pipeline_cache::VkPipelineCacheKey;
pub use workgroup::VkWorkgroupConfig;

// Re-exports — Vulkan callers reach AlgoCache/WorkspacePool via kiln_vulkan_blas.
pub use kiln_blas::{AlgoCache, AlgoCacheKey, AlgoCacheValue, WorkspacePool};

/// Stable phase tag (mirrors `kiln_blas::phase()` / `kiln_mps::phase()`).
pub fn phase() -> &'static str {
    "phase 2.3 — backend-agnostic Vulkan types (workgroup + pipeline cache + cooperative matrix detect); matmul wrapper Phase 2.x"
}
