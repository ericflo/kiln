//! kiln-rocblas — ROCm BLAS layer (hipBLASLt) for kiln-tensor.
//!
//! The ROCm analog of `kiln-blas`. The backend-agnostic production types
//! ([`AlgoCache`], [`WorkspacePool`], [`BackendMatmul`] + [`MatmulRequest`] +
//! [`Epilogue`]) are copied verbatim from `kiln-blas` (they carry no driver
//! dependency); the cublasLt executor is ported to hipBLASLt
//! ([`HipblasLtMatmulHandle`], behind `--features hipblaslt`).

mod algo_cache;
mod backend_matmul;
mod workspace_pool;

pub use algo_cache::{
    AlgoCache, AlgoCacheKey, AlgoCacheStats, AlgoCacheValue, deserialize_from_json, load_from_path,
    save_to_path, serialize_to_json,
};
pub use backend_matmul::{BackendMatmul, Epilogue, MatmulLayout, MatmulOutcome, MatmulRequest};
pub use workspace_pool::WorkspacePool;

/// FFI declarations for the hipBLASLt probe (mirrors kiln-blas's `probe_ffi`).
#[cfg(feature = "probe")]
pub mod probe_ffi {
    use std::os::raw::{c_int, c_void};

    /// Result struct returned by `kiln_blas_hipblaslt_mlp_probe`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct ProbeResult {
        pub bt: c_int,
        pub k: c_int,
        pub n: c_int,
        pub ms_rocblas_default: f32,
        pub ms_hipblaslt_heuristic: f32,
        pub chosen_algo_id: c_int,
        pub chosen_workspace_bytes: u64,
        pub iters: c_int,
        pub ok: c_int,
        pub err_code: c_int,
    }

    unsafe extern "C" {
        pub fn kiln_blas_hipblaslt_mlp_probe(
            bt: c_int,
            k: c_int,
            n: c_int,
            iters: c_int,
            out: *mut ProbeResult,
        ) -> c_int;
    }

    /// Safety: caller must ensure HIP is initialized on a thread bound to a HIP
    /// device; the C function creates and tears down its own hipBLASLt handle.
    pub fn probe(bt: i32, k: i32, n: i32, iters: i32) -> Result<ProbeResult, i32> {
        let mut out = ProbeResult {
            bt: 0,
            k: 0,
            n: 0,
            ms_rocblas_default: 0.0,
            ms_hipblaslt_heuristic: 0.0,
            chosen_algo_id: -1,
            chosen_workspace_bytes: 0,
            iters: 0,
            ok: 0,
            err_code: 0,
        };
        let rc = unsafe { kiln_blas_hipblaslt_mlp_probe(bt, k, n, iters, &mut out as *mut _) };
        if rc == 0 { Ok(out) } else { Err(rc) }
    }

    // Keep c_void referenced for parity with the cuda probe_ffi surface.
    #[allow(dead_code)]
    fn _unused(_: *mut c_void) {}
}

/// Stable phase tag.
pub fn phase() -> &'static str {
    "rocm R.6 — hipBLASLt MatmulHandle (ROCm analog of kiln-blas)"
}

// ----------------------------------------------------------------------
// R.6 — HipblasLtMatmulHandle (production matmul executor).
// ----------------------------------------------------------------------

#[cfg(feature = "hipblaslt")]
mod hipblaslt_handle;

#[cfg(feature = "hipblaslt")]
pub use hipblaslt_handle::{FfiError, HipblasLtMatmulHandle};
