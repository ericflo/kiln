//! kiln-mps — Metal BLAS layer (MPSMatrixMultiplication + custom MSL).
//!
//! Sibling to [`kiln-blas`](https://docs.rs/kiln-blas) (CUDA) and
//! `kiln-vulkan-blas` (Vulkan). Phase 2.2 of #1082 ships the
//! backend-agnostic Metal-specific extensions to the shared
//! [`kiln_blas::AlgoCache`] / [`kiln_blas::WorkspacePool`] types;
//! Phase 2.x adds the feature-gated `MPSMatmulHandle` (Metal
//! command-queue + per-stream binding).
//!
//! # Why a separate crate
//!
//! Per the Phase 2 issue bullet:
//!
//! > BLAS crate: `kiln-blas` (cublasLt) / `kiln-mps`
//! > (MPSMatrixMultiplication + transposed-coop GEMV port) /
//! > `kiln-vulkan-blas` (extend `kiln-vulkan-kernel::vk_ops/matmul*`)
//!
//! Each per-backend crate carries its own opaque algo descriptor
//! shape — cublasLt's `cublasLtMatmulAlgo_t` is a different blob from
//! MPS's `MPSMatrixDescriptor` tile-config. The shared `AlgoCache`
//! stores the bytes; per-backend crates serialize / deserialize.
//!
//! # Phase 2.2 public surface
//!
//! - [`MpsTilePolicy`] — Metal-side tile / transpose configuration.
//!   Lives in the algo blob serialized into `kiln_blas::AlgoCacheValue::algo_blob`.
//! - [`MpsUmaHint`] — UMA-aware allocation hint. On Apple Silicon the
//!   storage mode picker reads this to decide Shared vs Private.
//! - Re-exports of [`kiln_blas::AlgoCache`] + [`kiln_blas::WorkspacePool`]
//!   so a Metal-only caller has one import path.
//!
//! # CPU-buildable
//!
//! All Phase 2.2 types compile on every host. The Metal-runtime
//! `MPSMatmulHandle` lands in a subsequent PR gated behind the
//! `probe` feature.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod backend_matmul;
mod tile_policy;
mod uma;

pub use backend_matmul::MpsBackendMatmul;
pub use tile_policy::MpsTilePolicy;
pub use uma::MpsUmaHint;

// Re-exports — Metal callers reach AlgoCache/WorkspacePool via kiln_mps.
pub use kiln_blas::{AlgoCache, AlgoCacheKey, AlgoCacheValue, WorkspacePool};

/// Stable phase tag (mirrors kiln_blas::phase()).
pub fn phase() -> &'static str {
    "phase 2.2 — backend-agnostic Metal types (MpsTilePolicy + MpsUmaHint); MPSMatmulHandle Phase 2.x"
}
