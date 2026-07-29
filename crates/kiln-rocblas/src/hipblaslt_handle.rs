//! `HipblasLtMatmulHandle` — the Phase R.6 main-event matmul executor.
//!
//! ROCm analog of `kiln-blas`'s `CublasLtMatmulHandle`. Lifts the Phase R.x
//! `hipblaslt_mlp_probe` example into a reusable matmul executor that
//! `kiln-tensor`'s ROCm path delegates to.
//!
//! # Architecture
//!
//! - One [`HipblasLtMatmulHandle`] per ROCm device. Owns the
//!   `hipblasLtHandle_t` (a thread-safe hipBLAS context) plus a
//!   pre-allocated workspace buffer sized by [`WorkspacePool`].
//! - First call for a given `(shape, dtype, layout, epilogue,
//!   concurrent_streams)` runs `hipblasLtMatmulAlgoGetHeuristic` and
//!   inserts the algo into the shared [`AlgoCache`]. Subsequent
//!   calls reuse the cached algo blob — no heuristic search.
//! - Per-stream binding: every `matmul()` call passes the target
//!   `hipStream_t` to the C layer; the hipBLASLt handle itself does
//!   not bind a stream globally (matches the kt-tensor "ops are
//!   callable from any thread / any stream" invariant from anti-
//!   pattern 18).
//!
//! # Feature gating
//!
//! Compiled only under `--features hipblaslt`. The feature pulls
//! `kiln-hip` (same bounded HIP bindings every other kiln-rocm crate
//! uses) for the driver types (`RocmContext`, `RocmSlice`) and
//! triggers `build.rs` to compile `csrc/hipblaslt_matmul.cu`. The
//! backend-agnostic types in `lib.rs` (`AlgoCache`, `WorkspacePool`,
//! `BackendMatmul`) compile on every host without `hipblaslt`.
//!
//! # Threading
//!
//! `HipblasLtMatmulHandle` is `Send + Sync` — the hipBLASLt handle is
//! thread-safe per AMD docs (it mirrors cuBLASLt's thread-safety
//! contract). The internal `Mutex<...>` guards the workspace + algo
//! cache, both of which are mutable across calls. Multiple threads
//! can call `matmul()` concurrently; only the brief workspace-acquire
//! critical section serializes.
//!
//! # Anti-pattern alignment
//!
//! - Anti-pattern 18: `Send + Sync`; the `unsafe impl` carries the
//!   same justification cudarc/cuBLASLt use.
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

use kiln_hip::{RocmContext, RocmSlice, RocmStream, RocmSyncReason};

use crate::{
    AlgoCache, AlgoCacheStats, AlgoCacheValue, BackendMatmul, Epilogue, MatmulOutcome,
    MatmulRequest, WorkspacePool,
};

// ----------------------------------------------------------------------
// C ABI — mirror of crates/kiln-rocblas/csrc/hipblaslt_matmul.cu
// ----------------------------------------------------------------------

#[repr(C)]
struct KilnHipblasLtCtx {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KilnHipblasLtMatmulSpec {
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

// dtype codes — kept in sync with hipblaslt_matmul.cu's KILN_DTYPE_*.
const DTYPE_BF16: i32 = 0;
const DTYPE_F16: i32 = 1;
const DTYPE_F32: i32 = 2;

// epilogue codes — kept in sync with hipblaslt_matmul.cu's KILN_EPI_*.
const EPI_IDENTITY: i32 = 0;
const EPI_BIAS: i32 = 1;
const EPI_RELU: i32 = 2;
const EPI_GELU: i32 = 3;
#[allow(dead_code)]
const EPI_SILU: i32 = 4;
#[allow(dead_code)]
const EPI_BIAS_SILU: i32 = 5;
const EPI_BIAS_GELU: i32 = 6;

const ALGO_BLOB_MAX: usize = 256;

// The raw HIP stream handle threaded into the FFI. `kiln_hip` only exposes it
// through a live `RocmStreamSubmission`, which closes the quarantine-to-launch
// race; on the CUDA side this was `cudarc::driver::sys::CUstream`.
type HipStream = *mut core::ffi::c_void;

unsafe extern "C" {
    fn kiln_blas_hipblaslt_ctx_create(out_ctx: *mut *mut KilnHipblasLtCtx) -> c_int;
    fn kiln_blas_hipblaslt_ctx_destroy(ctx: *mut KilnHipblasLtCtx) -> c_int;
    fn kiln_blas_hipblaslt_matmul(
        ctx: *mut KilnHipblasLtCtx,
        stream: HipStream,
        spec: *const KilnHipblasLtMatmulSpec,
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
/// constants in `hipblaslt_matmul.cu` (identical to the cuBLASLt
/// exemplar's `KILN_BLAS_ERR_*` codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiError {
    /// `KILN_BLAS_ERR_CTX_CREATE` — hipblasLtCreate failed.
    CtxCreate,
    /// `KILN_BLAS_ERR_CTX_NULL` — caller passed a null context.
    CtxNull,
    /// `KILN_BLAS_ERR_DESC_CREATE` — hipblasLtMatmulDescCreate failed
    /// or the spec descriptor was null / invalid.
    DescCreate,
    /// `KILN_BLAS_ERR_LAYOUT_CREATE` — hipblasLtMatrixLayoutCreate
    /// failed for one of the operands.
    LayoutCreate,
    /// `KILN_BLAS_ERR_PREFERENCE` — preference creation failed or
    /// the heuristic's workspace requirement exceeded the caller's
    /// budget.
    Preference,
    /// `KILN_BLAS_ERR_HEURISTIC` — heuristic selection declined and the actual
    /// implicit-algorithm `hipblasLtMatmul` dispatch then failed.
    Heuristic,
    /// `KILN_BLAS_ERR_MATMUL` — `hipblasLtMatmul` returned a non-
    /// success status during the actual call.
    Matmul,
    /// `KILN_BLAS_ERR_UNSUPPORTED_DTYPE` — one of the dtypes is not
    /// in the BF16/F16/F32 production matrix.
    UnsupportedDType,
    /// `KILN_BLAS_ERR_UNSUPPORTED_EPILOGUE` — epilogue cannot be
    /// expressed as a native hipBLASLt epilogue (e.g., SiLU). Caller
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
    /// The owning hipBLASLt context could not be destroyed and was retained for
    /// process-lifetime fail-closed cleanup.
    CtxDestroy,
    /// A native hipBLASLt context/descriptor/preference could not be destroyed.
    /// Its handle is retained or unreachable until process exit.
    ResourceCleanup,
    /// Context creation returned an ambiguous partial or success-without-handle
    /// publication result.
    CtxCreatePartial,
    /// The typed ROCm stream refused execution because its context is cleanup
    /// quarantined or could not be rebound to the calling thread.
    StreamUnavailable,
    /// The supplied ROCm context belongs to a different device ordinal than
    /// the requested hipBLASLt handle.
    DeviceMismatch,
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
            -13 => Some(FfiError::CtxDestroy),
            -14 => Some(FfiError::ResourceCleanup),
            -15 => Some(FfiError::CtxCreatePartial),
            other => Some(FfiError::Unknown(other)),
        }
    }

    /// Whether this code represents attempted execution, ambiguous resource
    /// publication/cleanup, or an unknown result that cannot be proven local.
    /// Clean descriptor, shape, dtype, cache-blob, and workspace declines occur
    /// before dispatch and remain safe for a higher-level fallback.
    fn is_fatal_execution(self) -> bool {
        matches!(
            self,
            FfiError::Heuristic
                | FfiError::Matmul
                | FfiError::CtxDestroy
                | FfiError::ResourceCleanup
                | FfiError::CtxCreatePartial
                | FfiError::Unknown(_)
        )
    }

    /// Whether a higher layer may retry the request with a different logical
    /// batching layout without violating the execution-quarantine contract.
    pub fn permits_layout_fallback(self) -> bool {
        matches!(
            self,
            FfiError::DescCreate
                | FfiError::LayoutCreate
                | FfiError::Preference
                | FfiError::UnsupportedDType
                | FfiError::UnsupportedEpilogue
                | FfiError::InvalidShape
                | FfiError::AlgoDeserialize
                | FfiError::AlgoBlobTooSmall
        )
    }
}

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FfiError::CtxCreate => write!(f, "hipblasLtCreate failed"),
            FfiError::CtxNull => write!(f, "hipblasLt context pointer was null"),
            FfiError::DescCreate => write!(f, "hipblasLtMatmulDescCreate failed"),
            FfiError::LayoutCreate => write!(f, "hipblasLtMatrixLayoutCreate failed"),
            FfiError::Preference => {
                write!(f, "hipblasLt preference failed or workspace too small")
            }
            FfiError::Heuristic => {
                write!(f, "implicit hipBLASLt matmul fallback failed")
            }
            FfiError::Matmul => write!(f, "hipblasLtMatmul kernel call failed"),
            FfiError::UnsupportedDType => write!(f, "unsupported dtype (need BF16/F16/F32)"),
            FfiError::UnsupportedEpilogue => write!(
                f,
                "unsupported epilogue (SiLU / BiasSilu must lower to separate activation)"
            ),
            FfiError::InvalidShape => write!(f, "invalid shape (m/n/k must be > 0)"),
            FfiError::AlgoDeserialize => write!(f, "cached algo blob was the wrong size"),
            FfiError::AlgoBlobTooSmall => write!(f, "algo blob output buffer too small"),
            FfiError::CtxDestroy => write!(f, "hipBLASLt context destroy failed"),
            FfiError::ResourceCleanup => {
                write!(f, "hipBLASLt native resource cleanup failed")
            }
            FfiError::CtxCreatePartial => {
                write!(f, "hipBLASLt context creation published ambiguous state")
            }
            FfiError::StreamUnavailable => {
                write!(f, "ROCm stream is unavailable for hipBLASLt execution")
            }
            FfiError::DeviceMismatch => {
                write!(f, "ROCm context device does not match hipBLASLt device")
            }
            FfiError::Unknown(c) => write!(f, "unknown FFI error code: {c}"),
        }
    }
}

impl std::error::Error for FfiError {}

// ----------------------------------------------------------------------
// Handle
// ----------------------------------------------------------------------

/// Per-device hipBLASLt matmul executor. See module docstring.
///
/// Cheap to clone (`Arc<HandleInner>`); the underlying hipBLASLt
/// context is held by the inner `Arc`.
pub struct HipblasLtMatmulHandle {
    inner: Arc<HandleInner>,
}

/// RAII ownership for a stream-scoped hipBLASLt workspace.
///
/// Graph capture creates short-lived streams while probing geometries. Keeping
/// their workspaces in the process-global matmul handle after the stream or
/// captured graph is gone would make VRAM retention grow with capture history.
/// A lease removes the matching workspace after all work on the private stream
/// completes. Multiple leases for the same stream are reference counted.
#[must_use = "dropping the lease releases the stream-scoped workspace"]
pub struct HipblasLtWorkspaceLease {
    inner: Arc<HandleInner>,
    rocm_ctx: Arc<RocmContext>,
    stream: Arc<RocmStream>,
    stream_key: usize,
    released: bool,
}

/// Lock-only snapshot of the device allocation retained by one workspace
/// lease. `allocation_id` is an opaque identity for deduplicating accounting;
/// callers must never dereference it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HipblasLtWorkspaceStats {
    pub allocation_id: usize,
    pub retained_bytes: u64,
    pub max_bytes: u64,
}

#[derive(Debug, Default)]
struct StreamWorkspace {
    buffer: Option<Arc<RocmSlice>>,
    leases: usize,
}

struct HandleInner {
    /// Opaque C-side context (hipBLASLt handle + device index).
    /// `NonNull` because we error out on a null create rather than
    /// constructing.
    ctx: NonNull<KilnHipblasLtCtx>,
    /// kiln-hip ROCm context — used to allocate the workspace + query
    /// the default stream. Held directly via kiln-hip; callers that
    /// already have an `Arc<RocmContext>` (e.g. from the ROCm storage
    /// layer or `kiln_hip::RocmContext::new`) construct the handle with
    /// no candle hop.
    rocm_ctx: Arc<RocmContext>,
    /// 0-based ROCm device index this handle is bound to.
    device_index: usize,
    /// Persistent autotune cache. Shared across handles so warm
    /// shapes survive a handle drop/recreate cycle.
    algo_cache: Arc<Mutex<AlgoCache>>,
    /// Runtime cache hits that supplied a non-empty cached algo blob.
    algo_cache_hits: AtomicU64,
    /// Runtime cache misses that had to ask hipBLASLt for a heuristic.
    algo_cache_misses: AtomicU64,
    /// Runtime heuristic results inserted into the shared cache.
    algo_cache_inserts: AtomicU64,
    /// Valid matmuls served by hipBLASLt's implicit/default algorithm after
    /// the explicit heuristic returned no candidates.
    implicit_algo_fallbacks: AtomicU64,
    /// Workspace policy + counters.
    workspace_pool: Mutex<WorkspacePool>,
    /// Backing workspace buffers keyed by HIP stream handle. Each buffer is
    /// allocated on and freed by the stream that uses it, so a handle can be
    /// shared across graph-capture or multi-stream callers without reusing one
    /// mutable workspace concurrently on unordered streams.
    workspace_by_stream: Mutex<HashMap<usize, StreamWorkspace>>,
}

// SAFETY: the hipBLASLt context is documented as thread-safe (AMD's
// hipBLASLt mirrors cuBLASLt's "fully thread-safe" contract). The
// kiln-hip `RocmContext` is already `Arc`'d and is itself `Send + Sync`
// (binding is re-applied per call via hipSetDevice). The mutexes guard
// the only mutable state.
unsafe impl Send for HandleInner {}
unsafe impl Sync for HandleInner {}

impl std::fmt::Debug for HipblasLtMatmulHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.algo_cache_stats();
        f.debug_struct("HipblasLtMatmulHandle")
            .field("device_index", &self.inner.device_index)
            .field("workspace_max_bytes", &self.workspace_pool().max_bytes)
            .field("workspace_peak_bytes", &self.workspace_pool().peak_bytes)
            .field("workspace_call_count", &self.workspace_pool().call_count)
            .field("algo_cache_len", &stats.entries)
            .field("algo_cache_hits", &stats.hits)
            .field("algo_cache_misses", &stats.misses)
            .field("algo_cache_inserts", &stats.inserts)
            .field(
                "implicit_algo_fallbacks",
                &self.implicit_algo_fallback_count(),
            )
            .finish()
    }
}

impl std::fmt::Debug for HipblasLtWorkspaceLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipblasLtWorkspaceLease")
            .field("device_index", &self.inner.device_index)
            .field("stream_key", &self.stream_key)
            .field("released", &self.released)
            .finish()
    }
}

impl HipblasLtMatmulHandle {
    /// Construct a new handle bound to the kiln-hip `RocmContext` `ctx`.
    /// Callers that already have an `Arc<RocmContext>` (e.g. from the
    /// ROCm storage layer) construct a handle directly without
    /// materializing any candle `Device` wrapper.
    ///
    /// The handle takes a shared reference to the algo cache so
    /// multiple handles on the same process can amortize autotune
    /// across themselves.
    ///
    /// `workspace_max_bytes` controls the per-call workspace cap.
    /// Defaults to [`WorkspacePool::DEFAULT_MAX_BYTES`] (32 MiB) when
    /// `None`.
    pub fn new_ctx(
        rocm_ctx: Arc<RocmContext>,
        device_index: usize,
        algo_cache: Arc<Mutex<AlgoCache>>,
        workspace_max_bytes: Option<u64>,
    ) -> Result<Self, FfiError> {
        if rocm_ctx.ordinal() != device_index {
            return Err(FfiError::DeviceMismatch);
        }
        let mut raw: *mut KilnHipblasLtCtx = std::ptr::null_mut();
        let submission = rocm_ctx
            .execution_submission("hipBLASLt context create")
            .map_err(|_| FfiError::StreamUnavailable)?;
        // SAFETY: out_ctx is a valid pointer to a stack mut pointer;
        // C side writes a freshly-malloc'd ctx into it.
        let code = unsafe { kiln_blas_hipblaslt_ctx_create(&mut raw) };
        if let Some(e) = FfiError::from_code(code) {
            if !raw.is_null() || e.is_fatal_execution() {
                submission.quarantine();
            } else {
                submission.complete();
            }
            return Err(e);
        }
        let Some(ctx) = NonNull::new(raw) else {
            submission.quarantine();
            return Err(FfiError::CtxCreate);
        };
        submission.complete();

        let pool = match workspace_max_bytes {
            Some(b) => WorkspacePool::with_cap(b),
            None => WorkspacePool::new(),
        };

        Ok(HipblasLtMatmulHandle {
            inner: Arc::new(HandleInner {
                ctx,
                rocm_ctx,
                device_index,
                algo_cache,
                algo_cache_hits: AtomicU64::new(0),
                algo_cache_misses: AtomicU64::new(0),
                algo_cache_inserts: AtomicU64::new(0),
                implicit_algo_fallbacks: AtomicU64::new(0),
                workspace_pool: Mutex::new(pool),
                workspace_by_stream: Mutex::new(HashMap::new()),
            }),
        })
    }

    /// Construct with a fresh, empty algo cache. Convenience for
    /// tests + per-process initialization when the application has
    /// no pre-shipped cache to load.
    ///
    /// See [`Self::new_ctx`] for the underlying `Arc<RocmContext>`-based
    /// constructor.
    pub fn with_fresh_cache_ctx(
        rocm_ctx: Arc<RocmContext>,
        device_index: usize,
        workspace_max_bytes: Option<u64>,
    ) -> Result<Self, FfiError> {
        Self::new_ctx(
            rocm_ctx,
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

    /// Number of calls that required hipBLASLt's uncached implicit algorithm.
    pub fn implicit_algo_fallback_count(&self) -> u64 {
        self.inner.implicit_algo_fallbacks.load(Ordering::Relaxed)
    }

    /// ROCm device index this handle is bound to.
    pub fn device_index(&self) -> usize {
        self.inner.device_index
    }

    /// The kiln-hip `RocmContext` this handle was constructed with
    /// (via [`Self::new_ctx`] / [`Self::with_fresh_cache_ctx`]). Used
    /// by call sites that need to allocate output tensors or query
    /// the default stream — all via kiln-hip directly, no candle hop.
    pub fn rocm_context(&self) -> &Arc<RocmContext> {
        &self.inner.rocm_ctx
    }

    /// Retain any hipBLASLt workspace allocated on `stream` until the returned
    /// lease is dropped. Capture callers create the lease before their warm
    /// pass so every success, fallback, and error path has the same cleanup.
    pub fn workspace_lease(
        &self,
        rocm_ctx: &Arc<RocmContext>,
        stream: &Arc<RocmStream>,
    ) -> HipblasLtWorkspaceLease {
        let stream_key = Arc::as_ptr(stream) as usize;
        let mut by_stream = self
            .inner
            .workspace_by_stream
            .lock()
            .expect("workspace map mutex poisoned");
        let entry = by_stream.entry(stream_key).or_default();
        entry.leases = entry
            .leases
            .checked_add(1)
            .expect("hipBLASLt workspace lease count overflow");
        drop(by_stream);
        HipblasLtWorkspaceLease {
            inner: Arc::clone(&self.inner),
            rocm_ctx: Arc::clone(rocm_ctx),
            stream: Arc::clone(stream),
            stream_key,
            released: false,
        }
    }

    /// Number of HIP streams currently represented in the workspace map.
    /// Leased capture streams disappear when their graph or failed attempt is
    /// released; the long-lived default stream may remain cached.
    pub fn workspace_stream_count(&self) -> usize {
        self.inner
            .workspace_by_stream
            .lock()
            .expect("workspace map mutex poisoned")
            .len()
    }

    /// Clone the handle (cheap — internal `Arc`).
    pub fn share(&self) -> Self {
        HipblasLtMatmulHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Run a matmul.
    ///
    /// All pointer args are device pointers. `a_ptr` and `b_ptr` are
    /// const; `c_ptr` is written to. `bias_ptr` is optional — pass
    /// `std::ptr::null()` when the epilogue is not a `Bias*` variant.
    ///
    /// `stream` is the HIP stream to enqueue the kernel on. It is typed, not
    /// just a raw `hipStream_t`, so the handle can allocate and retain
    /// workspace on the same stream by construction.
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
    ///   `b_ptr` (hipBLASLt does not handle in-place GEMM).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul(
        &self,
        stream: &Arc<RocmStream>,
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

        // A cache miss is the only opportunity to discover an algorithm for a
        // shape. Give the heuristic the full configured budget; starting at a
        // 1 MiB allocation silently excluded otherwise-valid algorithms and
        // could turn a new attention shape into a request failure.
        let workspace_request = heuristic_workspace_request(
            cache_hit,
            cached_workspace_bytes,
            self.workspace_pool().max_bytes,
        );
        let (workspace, workspace_bytes) = self.ensure_workspace(stream, workspace_request)?;
        let workspace_ptr_raw = workspace.device_ptr() as *mut c_void;

        let mut algo_blob_out = vec![0u8; ALGO_BLOB_MAX];
        let mut algo_blob_out_len: u64 = algo_blob_out.len() as u64;
        let mut chosen_algo_id: i32 = -1;
        let mut chosen_workspace_bytes: u64 = 0;

        // Acquire only after fallible cache/workspace preparation so the
        // admission interval covers the physical FFI call, not arbitrary host
        // work that can delay quarantine settlement.
        let stream_submission = stream
            .execution_submission("hipBLASLt matmul")
            .map_err(|_| FfiError::StreamUnavailable)?;
        let raw_stream = stream_submission.raw_stream();
        let code = unsafe {
            kiln_blas_hipblaslt_matmul(
                self.inner.ctx.as_ptr(),
                raw_stream,
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
            if e.is_fatal_execution() {
                eprintln!(
                    "hipBLASLt fatal execution failure: error={e} m={} n={} k={} \
                     batch={} dtype={} output_dtype={} a_layout={:?} b_layout={:?}",
                    request.m,
                    request.n,
                    request.k,
                    request.batch_count,
                    request.dtype,
                    request.output_dtype,
                    request.a_layout,
                    request.b_layout,
                );
                stream_submission.quarantine();
            } else {
                stream_submission.complete();
            }
            drop(workspace);
            return Err(e);
        }
        stream_submission.complete();
        // If a concurrent caller resized this stream's workspace, releasing
        // our final reference queues its free after the matmul submission on
        // the same stream.
        drop(workspace);

        algo_blob_out.truncate(algo_blob_out_len as usize);

        if cached_blob.is_empty() && chosen_algo_id < 0 && algo_blob_out.is_empty() {
            self.inner
                .implicit_algo_fallbacks
                .fetch_add(1, Ordering::Relaxed);
        }

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
    /// large (clamped by the pool's `max_bytes`). Returns shared ownership of
    /// the buffer so a concurrent resize cannot retire it before submission.
    fn ensure_workspace(
        &self,
        stream: &Arc<RocmStream>,
        requested_bytes: u64,
    ) -> Result<(Arc<RocmSlice>, u64), FfiError> {
        let max_bytes = self.workspace_pool().max_bytes;
        let desired_bytes = std::cmp::min(
            std::cmp::max(requested_bytes, 1024 * 1024), // 1 MiB floor
            max_bytes,
        );
        let stream_key = Arc::as_ptr(stream) as usize;

        let mut by_stream = self
            .inner
            .workspace_by_stream
            .lock()
            .expect("workspace map mutex poisoned");

        let entry = by_stream.entry(stream_key).or_default();
        let need_alloc = match entry.buffer.as_ref() {
            None => true,
            Some(s) => (s.len() as u64) < desired_bytes,
        };
        if need_alloc {
            let new_buf = Arc::new(
                stream
                    .alloc(desired_bytes as usize)
                    .map_err(|_| FfiError::Preference)?,
            );
            entry.buffer = Some(new_buf);
        }

        let buffer = Arc::clone(entry.buffer.as_ref().expect("just initialized"));
        let byte_len = buffer.len() as u64;
        Ok((buffer, byte_len))
    }
}

impl HipblasLtWorkspaceLease {
    /// Snapshot this lease's current workspace without synchronizing a stream or
    /// calling the HIP runtime. Graph capture calls this after its warm pass,
    /// when the private stream's workspace has reached its retained size.
    pub fn stats(&self) -> Result<HipblasLtWorkspaceStats, FfiError> {
        let (allocation_id, retained_bytes) = match self.inner.workspace_by_stream.lock() {
            Ok(by_stream) => by_stream
                .get(&self.stream_key)
                .and_then(|entry| entry.buffer.as_ref())
                .map(|buffer| (buffer.device_ptr_usize(), buffer.len() as u64))
                .unwrap_or_default(),
            Err(poisoned) => {
                self.rocm_ctx.quarantine_execution();
                drop(poisoned.into_inner());
                return Err(FfiError::StreamUnavailable);
            }
        };
        let max_bytes = match self.inner.workspace_pool.lock() {
            Ok(pool) => pool.max_bytes,
            Err(poisoned) => {
                self.rocm_ctx.quarantine_execution();
                drop(poisoned.into_inner());
                return Err(FfiError::StreamUnavailable);
            }
        };
        Ok(HipblasLtWorkspaceStats {
            allocation_id,
            retained_bytes,
            max_bytes,
        })
    }

    fn workspace_map_for_cleanup(
        &self,
    ) -> Option<std::sync::MutexGuard<'_, HashMap<usize, StreamWorkspace>>> {
        match self.inner.workspace_by_stream.lock() {
            Ok(workspaces) => Some(workspaces),
            Err(poisoned) => {
                self.rocm_ctx.quarantine_execution();
                drop(poisoned.into_inner());
                eprintln!(
                    "HipblasLtWorkspaceLease::drop: workspace map is poisoned; retaining workspace and quarantining ROCm execution"
                );
                None
            }
        }
    }

    fn release_inner(&mut self) {
        if self.released {
            return;
        }
        self.released = true;

        // A shared lease only releases this owner's reference. The last lease
        // performs the stream settlement and allocation removal below.
        {
            let Some(mut by_stream) = self.workspace_map_for_cleanup() else {
                return;
            };
            let Some(entry) = by_stream.get_mut(&self.stream_key) else {
                return;
            };
            if entry.leases > 1 {
                entry.leases -= 1;
                return;
            }
        }

        // Workspace pointers may be embedded in a captured graph or referenced
        // by queued kernels. Never free them until the private stream settles.
        if let Err(error) = self
            .rocm_ctx
            .synchronize_stream_for(&self.stream, RocmSyncReason::AllocationLifetime)
        {
            eprintln!(
                "HipblasLtWorkspaceLease::drop: stream synchronization failed; retaining workspace: {error}"
            );
            let Some(mut by_stream) = self.workspace_map_for_cleanup() else {
                return;
            };
            if let Some(entry) = by_stream.get_mut(&self.stream_key) {
                entry.leases = entry.leases.saturating_sub(1);
            }
            return;
        }

        let workspace = {
            let Some(mut by_stream) = self.workspace_map_for_cleanup() else {
                return;
            };
            match by_stream.get_mut(&self.stream_key) {
                Some(entry) if entry.leases > 1 => {
                    entry.leases -= 1;
                    None
                }
                Some(_) => by_stream
                    .remove(&self.stream_key)
                    .and_then(|entry| entry.buffer),
                None => None,
            }
        };
        drop(workspace);

        // `RocmSlice::drop` uses hipFreeAsync when the allocation came from the
        // stream-ordered pool. Settle that free before the lease relinquishes
        // its final stream reference so reclaimed VRAM is deterministic.
        if let Err(error) = self
            .rocm_ctx
            .synchronize_stream_for(&self.stream, RocmSyncReason::MemoryReclaim)
        {
            eprintln!(
                "HipblasLtWorkspaceLease::drop: workspace free synchronization failed: {error}"
            );
        }
    }
}

impl Drop for HipblasLtWorkspaceLease {
    fn drop(&mut self) {
        self.release_inner();
    }
}

impl Drop for HandleInner {
    fn drop(&mut self) {
        let retain_until_exit = |this: &mut Self, reason: &str| {
            let retained_workspaces = std::mem::take(
                this.workspace_by_stream
                    .get_mut()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            );
            std::mem::forget(retained_workspaces);
            eprintln!(
                "HandleInner::drop: {reason}; retaining hipBLASLt context and workspaces until process exit"
            );
        };
        if self.rocm_ctx.cleanup_quarantined() {
            retain_until_exit(self, "ROCm execution is quarantined");
            return;
        }
        let submission = match self
            .rocm_ctx
            .execution_submission("hipBLASLt context destroy")
        {
            Ok(submission) => submission,
            Err(error) => {
                retain_until_exit(self, &format!("execution admission failed: {error}"));
                return;
            }
        };
        // SAFETY: ctx is non-null and owned. On failure the C wrapper retains
        // both the wrapper and hipBLASLt handle for fail-closed process-lifetime
        // quarantine; on success it destroys both.
        let code = unsafe { kiln_blas_hipblaslt_ctx_destroy(self.ctx.as_ptr()) };
        if code != 0 {
            submission.quarantine();
            retain_until_exit(self, &format!("hipBLASLt context destroy failed ({code})"));
        } else {
            submission.complete();
        }
    }
}

impl BackendMatmul for HipblasLtMatmulHandle {
    fn backend_name(&self) -> &'static str {
        "hipblaslt"
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

fn build_spec(req: &MatmulRequest) -> Result<KilnHipblasLtMatmulSpec, FfiError> {
    let dtype_in = resolve_dtype_code(req.dtype.as_str())?;
    let dtype_out = resolve_dtype_code(req.output_dtype.as_str())?;
    let epilogue = resolve_epilogue_code(req.epilogue)?;
    // Layout interpretation: the request descriptor names operand
    // layouts in *hipBLASLt* terms — `RowMajor` is the kiln-tensor
    // default; `ColMajor` is the transposed alternative. The C side
    // maps these to hipBLASLt's TRANSA/TRANSB.
    let a_transposed = matches!(req.a_layout, crate::MatmulLayout::ColMajor) as i32;
    let b_transposed = matches!(req.b_layout, crate::MatmulLayout::ColMajor) as i32;
    if req.m == 0 || req.n == 0 || req.k == 0 || req.batch_count == 0 {
        return Err(FfiError::InvalidShape);
    }
    Ok(KilnHipblasLtMatmulSpec {
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

fn heuristic_workspace_request(cache_hit: bool, cached_bytes: u64, max_bytes: u64) -> u64 {
    if cache_hit { cached_bytes } else { max_bytes }
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
        assert_eq!(FfiError::from_code(-13), Some(FfiError::CtxDestroy));
        assert_eq!(FfiError::from_code(-14), Some(FfiError::ResourceCleanup));
        assert_eq!(FfiError::from_code(-15), Some(FfiError::CtxCreatePartial));
        match FfiError::from_code(-99) {
            Some(FfiError::Unknown(-99)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn ffi_fatality_separates_fallbacks_from_execution_or_cleanup_failure() {
        for error in [
            FfiError::Heuristic,
            FfiError::Matmul,
            FfiError::CtxDestroy,
            FfiError::ResourceCleanup,
            FfiError::CtxCreatePartial,
            FfiError::Unknown(-99),
        ] {
            assert!(error.is_fatal_execution(), "{error:?} must quarantine");
        }
        for error in [
            FfiError::CtxCreate,
            FfiError::DescCreate,
            FfiError::Preference,
            FfiError::InvalidShape,
        ] {
            assert!(!error.is_fatal_execution(), "{error:?} is pre-dispatch");
        }
    }

    #[test]
    fn layout_fallback_excludes_attempted_or_unavailable_execution() {
        for error in [
            FfiError::DescCreate,
            FfiError::LayoutCreate,
            FfiError::Preference,
            FfiError::InvalidShape,
            FfiError::AlgoDeserialize,
        ] {
            assert!(error.permits_layout_fallback(), "{error:?}");
        }
        for error in [
            FfiError::CtxCreate,
            FfiError::Heuristic,
            FfiError::Matmul,
            FfiError::CtxDestroy,
            FfiError::ResourceCleanup,
            FfiError::CtxCreatePartial,
            FfiError::StreamUnavailable,
            FfiError::DeviceMismatch,
            FfiError::Unknown(-99),
        ] {
            assert!(!error.permits_layout_fallback(), "{error:?}");
        }
    }

    #[test]
    fn cache_miss_searches_the_full_workspace_budget() {
        assert_eq!(heuristic_workspace_request(false, 0, 32 << 20), 32 << 20);
        assert_eq!(heuristic_workspace_request(false, 4096, 8 << 20), 8 << 20);
        assert_eq!(heuristic_workspace_request(true, 4096, 32 << 20), 4096);
        assert_eq!(heuristic_workspace_request(true, 0, 32 << 20), 0);
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
