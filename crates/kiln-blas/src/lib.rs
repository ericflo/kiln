//! kiln-blas — CUDA BLAS layer (cublasLt) for kiln-tensor.
//!
//! Phase 0 shipped the `cublaslt_mlp_probe` example. Phase 2.1
//! (this revision) adds the backend-agnostic production data
//! structures: [`AlgoCache`] (disk-persistent autotune cache) +
//! [`WorkspacePool`] (per-handle workspace policy). Phase 2.x
//! wires these to a real `cublasLtMatmul` call via a feature-gated
//! `MatmulHandle` (cublasLt context + per-stream binding).
//!
//! See `examples/cublaslt_mlp_probe.rs` and the issue at
//! <https://github.com/ericflo/kiln/issues/1082>.
//!
//! # Phase 2.1 public surface
//!
//! - [`AlgoCache`] + [`AlgoCacheKey`] + [`AlgoCacheValue`] — Phase 2's
//!   disk-persistent autotune cache, keyed on
//!   `(shape, dtype, layout, concurrent_streams, kiln_version_major)`.
//! - [`WorkspacePool`] — per-handle workspace cap (default 32 MiB) +
//!   peak-bytes / call-count tracking.
//! - [`BackendMatmul`] + [`MatmulRequest`] + [`Epilogue`] —
//!   backend-agnostic matmul trait. CUDA / Metal / Vulkan handles
//!   implement this single trait; forward.rs reaches for
//!   `dyn BackendMatmul` and is free of per-backend conditionals.
//! - [`probe_ffi`] — Phase 0.8 probe FFI (kept for the probe example).
//!
//! # CPU-buildable
//!
//! All Phase 2.1 types are backend-agnostic — no `--features cuda`
//! required to use them. The feature-gated `MatmulHandle` lands in a
//! subsequent PR.

mod algo_cache;
mod backend_matmul;
mod workspace_pool;

pub use algo_cache::{
    save_to_path, serialize_to_json, AlgoCache, AlgoCacheKey, AlgoCacheValue,
};
pub use backend_matmul::{
    BackendMatmul, Epilogue, MatmulLayout, MatmulOutcome, MatmulRequest,
};
pub use workspace_pool::WorkspacePool;

/// FFI declarations for the Phase 0 probe.
///
/// The probe binary is the only consumer today; Phase 2 will replace
/// these with structured wrappers around the cublasLt API.
#[cfg(feature = "probe")]
pub mod probe_ffi {
    use std::os::raw::{c_int, c_void};

    /// Result struct returned by `kiln_blas_cublaslt_mlp_probe`.
    ///
    /// `chosen_algo_id` and `chosen_workspace_bytes` come from the
    /// `cublasLtMatmulAlgoGetHeuristic` call inside the probe; they
    /// document which algo the heuristic picked at the per-shape sweep.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct ProbeResult {
        pub bt: c_int,
        pub k: c_int,
        pub n: c_int,
        pub ms_cublas_default: f32,
        pub ms_cublaslt_heuristic: f32,
        pub chosen_algo_id: c_int,
        pub chosen_workspace_bytes: u64,
        pub iters: c_int,
        pub ok: c_int,
        pub err_code: c_int,
    }

    unsafe extern "C" {
        /// Run the cublasLt vs. cublas-default probe at the given shape.
        ///
        /// Inputs:
        /// - `bt`, `k`, `n`: matmul shape `[bt, k] @ [k, n] = [bt, n]`.
        /// - `iters`: number of timed iterations to median over.
        /// - `out`: out-param pointer to a [`ProbeResult`].
        ///
        /// Returns 0 on success, nonzero on error.
        pub fn kiln_blas_cublaslt_mlp_probe(
            bt: c_int,
            k: c_int,
            n: c_int,
            iters: c_int,
            out: *mut ProbeResult,
        ) -> c_int;
    }

    /// Safety: caller must ensure CUDA is initialized on a thread that
    /// holds a CUDA context; the C function creates and tears down its
    /// own cublasLt handle and workspace internally.
    pub fn probe(bt: i32, k: i32, n: i32, iters: i32) -> Result<ProbeResult, i32> {
        let mut out = ProbeResult {
            bt: 0,
            k: 0,
            n: 0,
            ms_cublas_default: 0.0,
            ms_cublaslt_heuristic: 0.0,
            chosen_algo_id: -1,
            chosen_workspace_bytes: 0,
            iters: 0,
            ok: 0,
            err_code: 0,
        };
        let rc = unsafe {
            kiln_blas_cublaslt_mlp_probe(bt, k, n, iters, &mut out as *mut _)
        };
        if rc == 0 {
            Ok(out)
        } else {
            Err(rc)
        }
    }
}

/// Stable phase tag. Used by the `kiln-bench` JSON reports + Phase 9
/// audit logs to distinguish Phase 0 (probe-only) numbers from
/// Phase 2.x (production-path) numbers.
pub fn phase() -> &'static str {
    "phase 2.1 — backend-agnostic API (AlgoCache + WorkspacePool); cublasLt MatmulHandle Phase 2.x"
}
// ----------------------------------------------------------------------
// Phase 2.x — CublasLtMatmulHandle (production matmul executor).
// ----------------------------------------------------------------------

#[cfg(feature = "cublaslt")]
mod cublaslt_handle;

#[cfg(feature = "cublaslt")]
pub use cublaslt_handle::{CublasLtMatmulHandle, FfiError};

