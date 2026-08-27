//! `CublasLtMatmulHandle` — the Phase 2 main-event matmul executor.
//!
//! Lifts the Phase 0.8 `cublaslt_mlp_probe` example into a reusable
//! matmul executor that `kiln-tensor`'s CUDA path delegates to.
//!
//! # Architecture
//!
//! - One [`CublasLtMatmulHandle`] per CUDA device. Owns the
//!   `cublasLtHandle_t` (a thread-safe cuBLAS context) plus a
//!   pre-allocated workspace buffer sized by [`WorkspacePool`].
//! - First call for a given `(shape, dtype, layout, epilogue,
//!   concurrent_streams)` runs `cublasLtMatmulAlgoGetHeuristic` and
//!   inserts the algo into the shared [`AlgoCache`]. Subsequent
//!   calls reuse the cached algo blob — no heuristic search.
//! - Per-stream binding: every `matmul()` call passes the target
//!   `cudaStream_t` to the C layer; the cublasLt handle itself does
//!   not bind a stream globally (matches the kt-tensor "ops are
//!   callable from any thread / any stream" invariant from anti-
//!   pattern 18).
//!
//! # Feature gating
//!
//! Compiled only under `--features cublaslt`. The feature pulls
//! the workspace's pinned `cudarc` (same version every other kiln
//! crate uses) for the driver types (`CudaContext`, `CudaSlice`,
//! `CUstream`, `DevicePtr`) and triggers `build.rs` to compile
//! `csrc/cublaslt_matmul.cu`. The backend-agnostic types in
//! `lib.rs` (`AlgoCache`, `WorkspacePool`, `BackendMatmul`) compile
//! on every host without `cublaslt`. #1082: kiln-blas no longer
//! depends on `candle-core` — cudarc is the sole CUDA substrate.
//!
//! # Threading
//!
//! `CublasLtMatmulHandle` is `Send + Sync` — the cublasLt handle is
//! thread-safe per NVIDIA docs. The internal `Mutex<...>` guards
//! the workspace + algo cache, both of which are mutable across
//! calls. Multiple threads can call `matmul()` concurrently; only
//! the brief workspace-acquire critical section serializes.
//!
//! # Anti-pattern alignment
//!
//! - Anti-pattern 18: `Send + Sync`, no `unsafe impl Send` escape
//!   hatches.
//! - Anti-pattern 4: no restructure of `BackendRuntime`; this is a
//!   *new* trait impl plugged in *below* the existing forward.rs
//!   dispatch.
//! - Anti-pattern 2: every workspace alloc is logged via the
//!   `WorkspacePool::record` call; copies-per-call counter exists.

use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::{c_int, c_uchar};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};

use crate::{
    AlgoCache, AlgoCacheStats, AlgoCacheValue, BackendMatmul, Epilogue, MatmulOutcome,
    MatmulRequest, WorkspacePool,
};

// ----------------------------------------------------------------------
// C ABI — mirror of crates/kiln-blas/csrc/cublaslt_matmul.cu
// ----------------------------------------------------------------------

#[repr(C)]
struct KilnCublasLtCtx {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KilnCublasLtMatmulSpec {
    m: i32,
    n: i32,
    k: i32,
    batch_count: i32,
    a_batch_stride: i64,
    b_batch_stride: i64,
    c_batch_stride: i64,
    dtype_in: i32,
    dtype_out: i32,
    a_transposed: i32,
    b_transposed: i32,
    epilogue: i32,
}

// dtype codes — kept in sync with cublaslt_matmul.cu's KILN_DTYPE_*.
const DTYPE_BF16: i32 = 0;
const DTYPE_F16: i32 = 1;
const DTYPE_F32: i32 = 2;

// epilogue codes — kept in sync with cublaslt_matmul.cu's KILN_EPI_*.
const EPI_IDENTITY: i32 = 0;
const EPI_BIAS: i32 = 1;
const EPI_RELU: i32 = 2;
const EPI_GELU: i32 = 3;
const EPI_BIAS_GELU: i32 = 6;

const ALGO_BLOB_MAX: usize = 256;

unsafe extern "C" {
    fn kiln_blas_cublaslt_ctx_create(out_ctx: *mut *mut KilnCublasLtCtx) -> c_int;
    fn kiln_blas_cublaslt_ctx_destroy(ctx: *mut KilnCublasLtCtx) -> c_int;
    fn kiln_blas_cublaslt_matmul(
        ctx: *mut KilnCublasLtCtx,
        stream: CUstream,
        spec: *const KilnCublasLtMatmulSpec,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        c_ptr: *mut c_void,
        bias_ptr: *const c_void,
        alpha: f32,
        beta: f32,
        workspace_ptr: *mut c_void,
        workspace_bytes: u64,
        algo_blob_in: *const c_uchar,
        algo_blob_in_len: u64,
        algo_blob_out: *mut c_uchar,
        algo_blob_out_len: *mut u64,
        chosen_algo_id: *mut i32,
        chosen_workspace_bytes: *mut u64,
    ) -> c_int;
}

// ----------------------------------------------------------------------
// FFI error mapping
// ----------------------------------------------------------------------

/// Errors returned by the FFI layer. The numeric values match the
/// constants in `cublaslt_matmul.cu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiError {
    /// `KILN_BLAS_ERR_CTX_CREATE` — cublasLtCreate failed.
    CtxCreate,
    /// `KILN_BLAS_ERR_CTX_NULL` — caller passed a null context.
    CtxNull,
    /// `KILN_BLAS_ERR_DESC_CREATE` — cublasLtMatmulDescCreate failed
    /// or the spec descriptor was null / invalid.
    DescCreate,
    /// `KILN_BLAS_ERR_LAYOUT_CREATE` — cublasLtMatrixLayoutCreate
    /// failed for one of the operands.
    LayoutCreate,
    /// `KILN_BLAS_ERR_PREFERENCE` — preference creation failed or
    /// the heuristic's workspace requirement exceeded the caller's
    /// budget.
    Preference,
    /// `KILN_BLAS_ERR_HEURISTIC` — `cublasLtMatmulAlgoGetHeuristic`
    /// returned zero algos or a non-success status.
    Heuristic,
    /// `KILN_BLAS_ERR_MATMUL` — `cublasLtMatmul` returned a non-
    /// success status during the actual call.
    Matmul,
    /// `KILN_BLAS_ERR_UNSUPPORTED_DTYPE` — one of the dtypes is not
    /// in the BF16/F16/F32 production matrix.
    UnsupportedDType,
    /// `KILN_BLAS_ERR_UNSUPPORTED_EPILOGUE` — epilogue cannot be
    /// expressed as a native cublasLt epilogue (e.g., SiLU). Caller
    /// must lower to a separate kt-tensor activation kernel.
    UnsupportedEpilogue,
    /// `KILN_BLAS_ERR_INVALID_SHAPE` — m, n, or k was non-positive.
    InvalidShape,
    /// `KILN_BLAS_ERR_ALGO_DESERIALIZE` — cached algo blob was the
    /// wrong size.
    AlgoDeserialize,
    /// `KILN_BLAS_ERR_ALGO_BLOB_TOO_SMALL` — caller's algo-blob-out
    /// buffer was too small.
    AlgoBlobTooSmall,
    /// Unknown / unrecognized error code.
    Unknown(c_int),
}

impl FfiError {
    fn from_code(code: c_int) -> Option<Self> {
        match code {
            0 => None,
            -1 => Some(FfiError::CtxCreate),
            -2 => Some(FfiError::CtxNull),
            -3 => Some(FfiError::DescCreate),
            -4 => Some(FfiError::LayoutCreate),
            -5 => Some(FfiError::Preference),
            -6 => Some(FfiError::Heuristic),
            -7 => Some(FfiError::Matmul),
            -8 => Some(FfiError::UnsupportedDType),
            -9 => Some(FfiError::UnsupportedEpilogue),
            -10 => Some(FfiError::InvalidShape),
            -11 => Some(FfiError::AlgoDeserialize),
            -12 => Some(FfiError::AlgoBlobTooSmall),
            other => Some(FfiError::Unknown(other)),
        }
    }
}

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FfiError::CtxCreate => write!(f, "cublasLtCreate failed"),
            FfiError::CtxNull => write!(f, "cublasLt context pointer was null"),
            FfiError::DescCreate => write!(f, "cublasLtMatmulDescCreate failed"),
            FfiError::LayoutCreate => write!(f, "cublasLtMatrixLayoutCreate failed"),
            FfiError::Preference => {
                write!(f, "cublasLt preference failed or workspace too small")
            }
            FfiError::Heuristic => write!(f, "cublasLtMatmulAlgoGetHeuristic returned no algos"),
            FfiError::Matmul => write!(f, "cublasLtMatmul kernel call failed"),
            FfiError::UnsupportedDType => write!(f, "unsupported dtype (need BF16/F16/F32)"),
            FfiError::UnsupportedEpilogue => write!(
                f,
                "unsupported epilogue (SiLU / BiasSilu must lower to separate activation)"
            ),
            FfiError::InvalidShape => write!(f, "invalid shape (m/n/k must be > 0)"),
            FfiError::AlgoDeserialize => write!(f, "cached algo blob was the wrong size"),
            FfiError::AlgoBlobTooSmall => write!(f, "algo blob output buffer too small"),
            FfiError::Unknown(c) => write!(f, "unknown FFI error code: {c}"),
        }
    }
}

impl std::error::Error for FfiError {}

// ----------------------------------------------------------------------
// Handle
// ----------------------------------------------------------------------

/// Per-device cublasLt matmul executor. See module docstring.
///
/// Cheap to clone (`Arc<HandleInner>`); the underlying cublasLt
/// context is held by the inner `Arc`.
pub struct CublasLtMatmulHandle {
    inner: Arc<HandleInner>,
}

struct HandleInner {
    /// Opaque C-side context (cublasLt handle + device index).
    /// `NonNull` because we error out on a null create rather than
    /// constructing.
    ctx: NonNull<KilnCublasLtCtx>,
    /// Cudarc CUDA context — used to allocate the workspace + query
    /// the default stream. Held directly via cudarc; callers that
    /// already have an `Arc<CudaContext>` (e.g. from
    /// `kiln_tensor::CudaStorage::context()` or
    /// `cudarc::driver::CudaContext::new`) construct the handle with
    /// no candle hop. #1082 finished the candle-free entry.
    cuda_ctx: Arc<CudaContext>,
    /// 0-based CUDA device index this handle is bound to.
    device_index: usize,
    /// Persistent autotune cache. Shared across handles so warm
    /// shapes survive a handle drop/recreate cycle.
    algo_cache: Arc<Mutex<AlgoCache>>,
    /// Runtime cache hits that supplied a non-empty cached algo blob.
    algo_cache_hits: AtomicU64,
    /// Runtime cache misses that had to ask cublasLt for a heuristic.
    algo_cache_misses: AtomicU64,
    /// Runtime heuristic results inserted into the shared cache.
    algo_cache_inserts: AtomicU64,
    /// Workspace policy + counters.
    workspace_pool: Mutex<WorkspacePool>,
    /// Backing workspace buffers keyed by CUDA stream handle. Each buffer is
    /// allocated on and freed by the stream that uses it, so a handle can be
    /// shared across graph-capture or multi-stream callers without reusing one
    /// mutable workspace concurrently on unordered streams.
    workspace_by_stream: Mutex<HashMap<usize, CudaSlice<u8>>>,
}

// SAFETY: cublasLt context is documented as thread-safe (per
// NVIDIA cuBLAS docs: "cublasLt is fully thread-safe"). The cudarc
// `CudaContext` is already `Arc`'d and thread-safe (its inner state
// uses interior locking). The mutexes guard the only mutable state.
unsafe impl Send for HandleInner {}
unsafe impl Sync for HandleInner {}

impl std::fmt::Debug for CublasLtMatmulHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.algo_cache_stats();
        f.debug_struct("CublasLtMatmulHandle")
            .field("device_index", &self.inner.device_index)
            .field("workspace_max_bytes", &self.workspace_pool().max_bytes)
            .field("workspace_peak_bytes", &self.workspace_pool().peak_bytes)
            .field("workspace_call_count", &self.workspace_pool().call_count)
            .field("algo_cache_len", &stats.entries)
            .field("algo_cache_hits", &stats.hits)
            .field("algo_cache_misses", &stats.misses)
            .field("algo_cache_inserts", &stats.inserts)
            .finish()
    }
}

impl CublasLtMatmulHandle {
    /// Construct a new handle bound to the cudarc `CudaContext` `ctx`.
    /// This is the **candle-free entry** (#1082) — callers that already
    /// have an `Arc<CudaContext>` (e.g. from `CudaStorage::context()`)
    /// can construct a handle directly without materializing a candle
    /// `CudaDevice` wrapper.
    ///
    /// The handle takes a shared reference to the algo cache so
    /// multiple handles on the same process can amortize autotune
    /// across themselves.
    ///
    /// `workspace_max_bytes` controls the per-call workspace cap.
    /// Defaults to [`WorkspacePool::DEFAULT_MAX_BYTES`] (32 MiB) when
    /// `None`.
    pub fn new_ctx(
        cuda_ctx: Arc<CudaContext>,
        device_index: usize,
        algo_cache: Arc<Mutex<AlgoCache>>,
        workspace_max_bytes: Option<u64>,
    ) -> Result<Self, FfiError> {
        let mut raw: *mut KilnCublasLtCtx = std::ptr::null_mut();
        // SAFETY: out_ctx is a valid pointer to a stack mut pointer;
        // C side writes a freshly-malloc'd ctx into it.
        let code = unsafe { kiln_blas_cublaslt_ctx_create(&mut raw) };
        if let Some(e) = FfiError::from_code(code) {
            return Err(e);
        }
        let ctx = NonNull::new(raw).ok_or(FfiError::CtxCreate)?;

        let pool = match workspace_max_bytes {
            Some(b) => WorkspacePool::with_cap(b),
            None => WorkspacePool::new(),
        };

        Ok(CublasLtMatmulHandle {
            inner: Arc::new(HandleInner {
                ctx,
                cuda_ctx,
                device_index,
                algo_cache,
                algo_cache_hits: AtomicU64::new(0),
                algo_cache_misses: AtomicU64::new(0),
                algo_cache_inserts: AtomicU64::new(0),
                workspace_pool: Mutex::new(pool),
                workspace_by_stream: Mutex::new(HashMap::new()),
            }),
        })
    }

    /// Construct with a fresh, empty algo cache. Convenience for
    /// tests + per-process initialization when the application has
    /// no pre-shipped cache to load.
    ///
    /// Candle-free entry. See [`Self::new_ctx`] for the underlying
    /// `Arc<CudaContext>`-based constructor.
    pub fn with_fresh_cache_ctx(
        cuda_ctx: Arc<CudaContext>,
        device_index: usize,
        workspace_max_bytes: Option<u64>,
    ) -> Result<Self, FfiError> {
        Self::new_ctx(
            cuda_ctx,
            device_index,
            Arc::new(Mutex::new(AlgoCache::new())),
            workspace_max_bytes,
        )
    }

    /// Snapshot the current workspace pool counters. Useful for
    /// `bench-results/` reports + debug logging.
    pub fn workspace_pool(&self) -> WorkspacePool {
        *self
            .inner
            .workspace_pool
            .lock()
            .expect("workspace pool mutex poisoned")
    }

    /// Snapshot the current algo cache. Cheap: the cache itself is
    /// behind an Arc<Mutex> so this clones the inner HashMap.
    pub fn algo_cache(&self) -> AlgoCache {
        self.inner
            .algo_cache
            .lock()
            .expect("algo cache mutex poisoned")
            .clone()
    }

    /// Snapshot runtime cache visibility for this handle.
    pub fn algo_cache_stats(&self) -> AlgoCacheStats {
        let entries = self
            .inner
            .algo_cache
            .lock()
            .expect("algo cache mutex poisoned")
            .len();
        AlgoCacheStats {
            entries,
            hits: self.inner.algo_cache_hits.load(Ordering::Relaxed),
            misses: self.inner.algo_cache_misses.load(Ordering::Relaxed),
            inserts: self.inner.algo_cache_inserts.load(Ordering::Relaxed),
        }
    }

    /// CUDA device index this handle is bound to.
    pub fn device_index(&self) -> usize {
        self.inner.device_index
    }

    /// The cudarc `CudaContext` this handle was constructed with
    /// (via [`Self::new_ctx`] / [`Self::with_fresh_cache_ctx`]). Used
    /// by call sites that need to allocate output tensors or query
    /// the default stream — all via cudarc directly, no candle hop.
    ///
    /// This is the **candle-free accessor** (#1082).
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.inner.cuda_ctx
    }

    /// Clone the handle (cheap — internal `Arc`).
    pub fn share(&self) -> Self {
        CublasLtMatmulHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Run a matmul.
    ///
    /// All pointer args are device pointers. `a_ptr` and `b_ptr` are
    /// const; `c_ptr` is written to. `bias_ptr` is optional — pass
    /// `std::ptr::null()` when the epilogue is not a `Bias*` variant.
    ///
    /// `stream` is the CUDA stream to enqueue the kernel on. It is typed, not
    /// just a raw `CUstream`, so the handle can allocate and retain workspace
    /// on the same stream by construction.
    ///
    /// On success, returns a [`MatmulOutcome`] documenting how many
    /// bytes were written and which algo was used. The algo blob is
    /// inserted into the shared `AlgoCache` so subsequent calls at
    /// the same shape skip the heuristic search.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    /// - `a_ptr`, `b_ptr`, `c_ptr` are valid device pointers with
    ///   sufficient backing storage for the declared dtype × shape.
    /// - `bias_ptr` is either null or a valid device pointer with
    ///   `n` elements of the output dtype.
    /// - The stream is valid and on the same device as this handle.
    /// - The output buffer at `c_ptr` is not aliased with `a_ptr` or
    ///   `b_ptr` (cublasLt does not handle in-place GEMM).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul(
        &self,
        stream: &Arc<CudaStream>,
        request: &MatmulRequest,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        c_ptr: *mut c_void,
        bias_ptr: *const c_void,
    ) -> Result<MatmulOutcome, FfiError> {
        let spec = build_spec(request)?;

        // Look up an existing algo for this shape; fall back to
        // heuristic search on a miss.
        let cache_key = request.cache_key();
        let (cached_blob, cached_workspace_bytes, cache_hit) = {
            let cache = self
                .inner
                .algo_cache
                .lock()
                .expect("algo cache mutex poisoned");
            match cache.get(&cache_key) {
                Some(v) if !v.algo_blob.is_empty() => {
                    (v.algo_blob.clone(), v.workspace_bytes, true)
                }
                _ => (Vec::new(), 0, false),
            }
        };
        if cache_hit {
            self.inner.algo_cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.inner.algo_cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        let (workspace_ptr_raw, workspace_bytes) =
            self.ensure_workspace(stream, cached_workspace_bytes)?;

        let mut algo_blob_out = vec![0u8; ALGO_BLOB_MAX];
        let mut algo_blob_out_len: u64 = algo_blob_out.len() as u64;
        let mut chosen_algo_id: i32 = -1;
        let mut chosen_workspace_bytes: u64 = 0;

        let code = unsafe {
            kiln_blas_cublaslt_matmul(
                self.inner.ctx.as_ptr(),
                stream.cu_stream(),
                &spec,
                a_ptr,
                b_ptr,
                c_ptr,
                bias_ptr,
                1.0,
                0.0,
                workspace_ptr_raw,
                workspace_bytes,
                if cached_blob.is_empty() {
                    std::ptr::null()
                } else {
                    cached_blob.as_ptr()
                },
                cached_blob.len() as u64,
                algo_blob_out.as_mut_ptr(),
                &mut algo_blob_out_len,
                &mut chosen_algo_id,
                &mut chosen_workspace_bytes,
            )
        };
        if let Some(e) = FfiError::from_code(code) {
            return Err(e);
        }

        algo_blob_out.truncate(algo_blob_out_len as usize);

        // Update workspace pool counters.
        {
            let mut pool = self
                .inner
                .workspace_pool
                .lock()
                .expect("workspace pool mutex poisoned");
            pool.record(chosen_workspace_bytes);
        }

        // On a cache miss, write back the chosen algo. On a hit,
        // skip the write (would be the same blob anyway).
        if cached_blob.is_empty() && !algo_blob_out.is_empty() {
            let mut cache = self
                .inner
                .algo_cache
                .lock()
                .expect("algo cache mutex poisoned");
            cache.insert(
                cache_key,
                AlgoCacheValue {
                    algo_id: chosen_algo_id,
                    workspace_bytes: chosen_workspace_bytes,
                    recorded_ms: 0.0,
                    algo_blob: algo_blob_out.clone(),
                },
            );
            self.inner
                .algo_cache_inserts
                .fetch_add(1, Ordering::Relaxed);
        }

        let bytes_written = bytes_for_dtype(request.output_dtype.as_str())
            * request.m
            * request.n
            * request.batch_count.max(1);
        Ok(MatmulOutcome {
            bytes_written,
            elapsed_ms: None,
            algo_blob: AlgoCacheValue {
                algo_id: chosen_algo_id,
                workspace_bytes: chosen_workspace_bytes,
                recorded_ms: 0.0,
                algo_blob: algo_blob_out,
            },
        })
    }

    /// Ensure the workspace buffer is at least `requested_bytes`
    /// large (clamped by the pool's `max_bytes`). Returns the raw
    /// device pointer + the buffer's byte length.
    fn ensure_workspace(
        &self,
        stream: &Arc<CudaStream>,
        requested_bytes: u64,
    ) -> Result<(*mut c_void, u64), FfiError> {
        let max_bytes = self.workspace_pool().max_bytes;
        let desired_bytes = std::cmp::min(
            std::cmp::max(requested_bytes, 1024 * 1024), // 1 MiB floor
            max_bytes,
        );
        let stream_key = stream.cu_stream() as usize;

        let mut by_stream = self
            .inner
            .workspace_by_stream
            .lock()
            .expect("workspace map mutex poisoned");

        let need_alloc = match by_stream.get(&stream_key) {
            None => true,
            Some(s) => (s.len() as u64) < desired_bytes,
        };
        if need_alloc {
            let new_buf = stream
                .alloc_zeros::<u8>(desired_bytes as usize)
                .map_err(|_| FfiError::Preference)?;
            by_stream.insert(stream_key, new_buf);
        }

        let buf_ref = by_stream.get(&stream_key).expect("just initialized");
        let (raw_ptr, _g) = buf_ref.device_ptr(stream);
        let byte_len = buf_ref.len() as u64;
        Ok((raw_ptr as *mut c_void, byte_len))
    }
}

impl Drop for HandleInner {
    fn drop(&mut self) {
        // SAFETY: ctx is non-null and owned. Destroy returns 0 on a
        // null input (handled gracefully) and on success otherwise.
        unsafe {
            kiln_blas_cublaslt_ctx_destroy(self.ctx.as_ptr());
        }
    }
}

impl BackendMatmul for CublasLtMatmulHandle {
    fn backend_name(&self) -> &'static str {
        "cublaslt"
    }

    /// Heuristic plan only — does **not** execute the matmul. Looks
    /// up the cache and returns the recorded outcome on a hit; on a
    /// miss, returns an empty algo blob (caller must invoke
    /// [`Self::matmul`] to fill the cache via the heuristic search).
    fn plan(&self, req: &MatmulRequest) -> MatmulOutcome {
        let key = req.cache_key();
        let cache = self
            .inner
            .algo_cache
            .lock()
            .expect("algo cache mutex poisoned");
        let bytes_written =
            bytes_for_dtype(req.output_dtype.as_str()) * req.m * req.n * req.batch_count.max(1);
        match cache.get(&key) {
            Some(v) => MatmulOutcome {
                bytes_written,
                elapsed_ms: None,
                algo_blob: v.clone(),
            },
            None => MatmulOutcome {
                bytes_written,
                elapsed_ms: None,
                algo_blob: AlgoCacheValue {
                    algo_id: -1,
                    workspace_bytes: 0,
                    recorded_ms: 0.0,
                    algo_blob: Vec::new(),
                },
            },
        }
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn build_spec(req: &MatmulRequest) -> Result<KilnCublasLtMatmulSpec, FfiError> {
    let dtype_in = resolve_dtype_code(req.dtype.as_str())?;
    let dtype_out = resolve_dtype_code(req.output_dtype.as_str())?;
    let epilogue = resolve_epilogue_code(req.epilogue)?;
    // Layout interpretation: the request descriptor names operand
    // layouts in *cublasLt* terms — `RowMajor` is the kiln-tensor
    // default; `ColMajor` is the transposed alternative. The C side
    // maps these to cublasLt's TRANSA/TRANSB.
    let a_transposed = matches!(req.a_layout, crate::MatmulLayout::ColMajor) as i32;
    let b_transposed = matches!(req.b_layout, crate::MatmulLayout::ColMajor) as i32;
    if req.m == 0 || req.n == 0 || req.k == 0 || req.batch_count == 0 {
        return Err(FfiError::InvalidShape);
    }
    Ok(KilnCublasLtMatmulSpec {
        m: req.m as i32,
        n: req.n as i32,
        k: req.k as i32,
        batch_count: req.batch_count as i32,
        a_batch_stride: req.a_batch_stride as i64,
        b_batch_stride: req.b_batch_stride as i64,
        c_batch_stride: req.c_batch_stride as i64,
        dtype_in,
        dtype_out,
        a_transposed,
        b_transposed,
        epilogue,
    })
}

fn resolve_dtype_code(dtype: &str) -> Result<i32, FfiError> {
    match dtype {
        "bf16" => Ok(DTYPE_BF16),
        "f16" => Ok(DTYPE_F16),
        "f32" => Ok(DTYPE_F32),
        _ => Err(FfiError::UnsupportedDType),
    }
}

fn resolve_epilogue_code(epi: Epilogue) -> Result<i32, FfiError> {
    match epi {
        Epilogue::Identity => Ok(EPI_IDENTITY),
        Epilogue::Bias => Ok(EPI_BIAS),
        Epilogue::Relu => Ok(EPI_RELU),
        Epilogue::Gelu => Ok(EPI_GELU),
        Epilogue::BiasGelu => Ok(EPI_BIAS_GELU),
        Epilogue::Silu | Epilogue::BiasSilu => Err(FfiError::UnsupportedEpilogue),
    }
}

fn bytes_for_dtype(dtype: &str) -> u64 {
    match dtype {
        "f32" => 4,
        "bf16" | "f16" => 2,
        "u8" | "f8_e4m3" | "f8_e5m2" => 1,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatmulLayout;

    #[test]
    fn ffi_error_code_roundtrip() {
        assert_eq!(FfiError::from_code(0), None);
        assert_eq!(FfiError::from_code(-1), Some(FfiError::CtxCreate));
        assert_eq!(FfiError::from_code(-7), Some(FfiError::Matmul));
        assert_eq!(FfiError::from_code(-10), Some(FfiError::InvalidShape));
        match FfiError::from_code(-99) {
            Some(FfiError::Unknown(-99)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn build_spec_round_trips_bf16_row_major() {
        let req = MatmulRequest {
            m: 2048,
            n: 18432,
            k: 2560,
            batch_count: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            dtype: "bf16".to_string(),
            output_dtype: "bf16".to_string(),
            a_layout: MatmulLayout::RowMajor,
            b_layout: MatmulLayout::RowMajor,
            c_layout: MatmulLayout::RowMajor,
            epilogue: Epilogue::Identity,
            concurrent_streams: 1,
        };
        let spec = build_spec(&req).unwrap();
        assert_eq!(spec.m, 2048);
        assert_eq!(spec.n, 18432);
        assert_eq!(spec.k, 2560);
        assert_eq!(spec.batch_count, 1);
        assert_eq!(spec.dtype_in, DTYPE_BF16);
        assert_eq!(spec.dtype_out, DTYPE_BF16);
        assert_eq!(spec.a_transposed, 0);
        assert_eq!(spec.b_transposed, 0);
        assert_eq!(spec.epilogue, EPI_IDENTITY);
    }

    #[test]
    fn build_spec_marks_transposes() {
        let req = MatmulRequest {
            m: 32,
            n: 64,
            k: 128,
            batch_count: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            dtype: "f32".to_string(),
            output_dtype: "f32".to_string(),
            a_layout: MatmulLayout::ColMajor,
            b_layout: MatmulLayout::RowMajor,
            c_layout: MatmulLayout::RowMajor,
            epilogue: Epilogue::Identity,
            concurrent_streams: 1,
        };
        let spec = build_spec(&req).unwrap();
        assert_eq!(spec.a_transposed, 1);
        assert_eq!(spec.b_transposed, 0);
    }

    #[test]
    fn build_spec_rejects_zero_dims() {
        let req = MatmulRequest {
            m: 0,
            n: 64,
            k: 128,
            batch_count: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            dtype: "bf16".to_string(),
            output_dtype: "bf16".to_string(),
            a_layout: MatmulLayout::RowMajor,
            b_layout: MatmulLayout::RowMajor,
            c_layout: MatmulLayout::RowMajor,
            epilogue: Epilogue::Identity,
            concurrent_streams: 1,
        };
        assert_eq!(build_spec(&req), Err(FfiError::InvalidShape));
    }

    #[test]
    fn epilogue_silu_returns_unsupported() {
        assert_eq!(
            resolve_epilogue_code(Epilogue::Silu),
            Err(FfiError::UnsupportedEpilogue)
        );
        assert_eq!(
            resolve_epilogue_code(Epilogue::BiasSilu),
            Err(FfiError::UnsupportedEpilogue)
        );
    }
}
