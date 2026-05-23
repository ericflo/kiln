//! kiln-blas — CUDA BLAS layer (cublasLt) for kiln-tensor.
//!
//! Phase 0 ships the `cublaslt_mlp_probe` example only. Phase 2 fills in
//! the production matmul path with explicit algo cache, workspace pool,
//! optional split-K, and optional fused-bias-and-activation epilogue.
//!
//! See `examples/cublaslt_mlp_probe.rs` and the issue at
//! <https://github.com/ericflo/kiln/issues/1082>.

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

/// Empty placeholder for the Phase 2 production path. See lib doc.
pub fn phase() -> &'static str {
    "phase 0 — probe only"
}
