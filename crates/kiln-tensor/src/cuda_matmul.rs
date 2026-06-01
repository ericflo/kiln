//! CUDA-side matmul that dispatches through [`kiln_blas::CublasLtMatmulHandle`].
//!
//! Phase 2.x of #1082 — the kt-side glue that gives
//! [`crate::ops::MatmulOp::cuda_fwd`] a real backend. The handle is
//! cached per CUDA device index via a process-global `OnceLock`
//! registry; subsequent calls on the same device reuse the same
//! cublasLt context + workspace + algo cache.
//!
//! # Supported shapes
//!
//! - 2-D: `[M, K] @ [K, N] = [M, N]`
//! - Batched 3-D / 4-D: `[..., M, K] @ [..., K, N] = [..., M, N]` —
//!   dispatched as a per-batch loop. Each iteration is a separate
//!   `cublasLtMatmul` call on the same stream; algo selection is
//!   shared (cached on the first call).
//!
//! # Supported dtypes
//!
//! BF16 / F16 / F32 (matches the kt-Tensor matmul CPU path). Compute
//! type is FP32 across all three (matches `forward.rs:3454,3517`'s
//! F32-promotion idiom).
//!
//! # Anti-patterns
//!
//! - Anti-pattern 1: no `candle::Tensor` field on `CudaStorage`. We
//!   reach for raw device pointers via
//!   [`crate::cuda_storage::CudaStorage::device_ptr_raw`] (which
//!   honors `Owned` + `Borrowed` storage variants).
//! - Anti-pattern 2: every `cuda_matmul` call is a real per-batch
//!   dispatch — no implicit `.contiguous()`. Validation rejects
//!   non-contiguous inputs explicitly.
//! - Anti-pattern 4: kt-side matmul sits *below* the `BackendRuntime`
//!   trait at `kiln-model/src/backend/mod.rs:114`. Existing call sites
//!   in `forward.rs` still go through the candle path until the env
//!   gate flips them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::CudaContext;
use kiln_blas::{
    AlgoCache, CublasLtMatmulHandle, Epilogue, MatmulLayout, MatmulRequest,
};

use crate::cuda_storage::CudaStorage;
use crate::{DType, Layout, Result, Storage, Tensor, TensorId};

// ----------------------------------------------------------------------
// Process-global per-device handle registry
// ----------------------------------------------------------------------

/// Per-process registry of cublasLt matmul handles keyed by CUDA
/// device index. The first call on a given device cold-starts the
/// handle; subsequent calls share it.
///
/// The algo cache is also process-shared: every handle on every
/// device reads/writes the same `Arc<Mutex<AlgoCache>>`, so a warm
/// shape on device 0 doesn't re-tune on device 1 (matters when
/// multi-process replicas eventually land; harmless for the
/// current single-device setup).
struct HandleRegistry {
    by_device: Mutex<HashMap<usize, Arc<CublasLtMatmulHandle>>>,
    shared_cache: Arc<Mutex<AlgoCache>>,
}

fn handle_registry() -> &'static HandleRegistry {
    static REGISTRY: OnceLock<HandleRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| HandleRegistry {
        by_device: Mutex::new(HashMap::new()),
        shared_cache: Arc::new(Mutex::new(AlgoCache::new())),
    })
}

/// Acquire (or cold-start) a handle for `device_index`.
///
/// The handle registry is keyed by device index, so warm-path lookups
/// don't need any extra state — only a cold start needs the cudarc
/// `CudaContext` to construct the underlying cublasLt handle. Callers
/// pass it through from the storage they already have on hand
/// (`a_storage.context()`), so the lookup stays fully candle-free
/// (#1082).
fn get_or_init_handle(
    device_index: usize,
    cuda_ctx: &Arc<CudaContext>,
) -> Result<Arc<CublasLtMatmulHandle>> {
    let reg = handle_registry();
    let mut by_device = reg
        .by_device
        .lock()
        .map_err(|_| crate::Error::Msg("cuda_matmul: handle registry mutex poisoned".to_string()))?;
    if let Some(h) = by_device.get(&device_index) {
        return Ok(Arc::clone(h));
    }
    // Cold-start the handle for this device via the candle-free
    // CublasLtMatmulHandle::new_ctx entry (#1082). No
    // primary_cuda_device materialization needed.
    let handle = CublasLtMatmulHandle::new_ctx(
        Arc::clone(cuda_ctx),
        device_index,
        Arc::clone(&reg.shared_cache),
        None,
    )
    .map_err(|e| {
        crate::Error::Msg(format!(
            "cuda_matmul: CublasLtMatmulHandle::new_ctx failed for device {device_index}: {e}"
        ))
    })?;
    let arc = Arc::new(handle);
    by_device.insert(device_index, Arc::clone(&arc));
    Ok(arc)
}

/// Expose the shared algo cache for debug + bench-results inspection.
/// The cache is a snapshot — modifying the returned `AlgoCache` does
/// not affect the registry.
pub fn snapshot_algo_cache() -> AlgoCache {
    handle_registry()
        .shared_cache
        .lock()
        .expect("algo cache mutex poisoned")
        .clone()
}

/// Merge entries from a disk-loaded autotune cache into the live shared cache.
/// First-writer wins: an algo already chosen in-process this run is kept, so
/// this is safe to call after some matmuls have run (though the intended call
/// site is once at startup, before the first matmul). Thread-safe via the
/// shared `Mutex`. (#1082 Phase 2 — disk-persistent autotune cache.)
pub fn restore_into_shared_cache(loaded: AlgoCache) {
    let reg = handle_registry();
    let mut cache = reg.shared_cache.lock().expect("algo cache mutex poisoned");
    for (k, v) in loaded.iter() {
        if cache.get(k).is_none() {
            cache.insert(k.clone(), v.clone());
        }
    }
}

/// Standard on-disk path for the cublasLt autotune cache on `device_index`:
/// `~/.cache/kiln/autotune/cublaslt-sm{major}{minor}-dev{index}.json`. Returns
/// `None` if the device's compute capability can't be queried. The
/// `sm{major}{minor}` fingerprint is portable across identical GPUs (cublasLt
/// algo ids are per-SM-arch), unlike a per-physical-card UUID. Honors the
/// `KILN_AUTOTUNE_CACHE_DIR` override via `AlgoCache::standard_path`'s HOME use.
pub fn cublaslt_cache_path(device_index: usize) -> Option<std::path::PathBuf> {
    let ctx = crate::primary_cuda_context(device_index).ok()?;
    let (major, minor) = ctx.compute_capability().ok()?;
    let fingerprint = format!("sm{major}{minor}-dev{device_index}");
    Some(AlgoCache::standard_path("cublaslt", &fingerprint))
}

/// Flush the live shared autotune cache to its standard on-disk path. Returns
/// the number of entries written (`0` if the cache is empty or the device
/// fingerprint is unavailable). Never panics; surfaces I/O errors so the
/// caller (kiln-server, which has `tracing`) can warn-and-continue.
pub fn flush_algo_cache_to_disk(device_index: usize) -> std::io::Result<usize> {
    let Some(path) = cublaslt_cache_path(device_index) else {
        return Ok(0);
    };
    let snapshot = snapshot_algo_cache();
    if snapshot.is_empty() {
        return Ok(0);
    }
    kiln_blas::save_to_path(&snapshot, &path)?;
    Ok(snapshot.len())
}

/// Load the on-disk autotune cache for `device_index` (if present) and merge
/// it into the live shared cache. Returns the number of entries loaded (`0`
/// if the file is missing/corrupt or the fingerprint is unavailable).
/// Best-effort and self-contained so kiln-server can call it without a direct
/// `kiln-blas` dependency. Intended to run once at startup, before prewarm.
pub fn load_algo_cache_from_disk(device_index: usize) -> usize {
    let Some(path) = cublaslt_cache_path(device_index) else {
        return 0;
    };
    let loaded = kiln_blas::load_from_path(&path);
    let n = loaded.len();
    if n > 0 {
        restore_into_shared_cache(loaded);
    }
    n
}

// ----------------------------------------------------------------------
// Public entry point
// ----------------------------------------------------------------------

/// Run a CUDA matmul on `[..., M, K] @ [..., K, N] = [..., M, N]`.
///
/// Both inputs must be on the same CUDA device, share the same dtype
/// (BF16/F16/F32), and be contiguous. Higher-rank inputs (batched)
/// are unrolled into per-batch dispatches sharing the same algo +
/// stream.
///
/// On the first call for a given `(shape, dtype, layout)`, the
/// underlying cublasLt heuristic picks an algo and the algo blob is
/// cached in the shared registry. Subsequent calls at the same shape
/// reuse the cached algo (no heuristic search).
pub fn cuda_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // ---- validate shapes ----
    let a_rank = a.rank();
    let b_rank = b.rank();
    if a_rank < 2 || b_rank < 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul: rank must be >= 2, got a={a_rank} b={b_rank}"
        )));
    }
    if a_rank != b_rank {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul: rank mismatch a={a_rank} b={b_rank}"
        )));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..a_rank - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(crate::Error::Msg(format!(
                "cuda_matmul: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let m = a_shape[a_rank - 2];
    let k_a = a_shape[a_rank - 1];
    let k_b = b_shape[b_rank - 2];
    let n = b_shape[b_rank - 1];
    if k_a != k_b {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul: contraction dim mismatch a.K={k_a} b.K={k_b}"
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    let dtype_str = match dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_matmul: unsupported dtype {other}"
            )));
        }
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_matmul: contiguous inputs required (call .contiguous() first)".to_string(),
        ));
    }

    // ---- resolve CUDA storage + device ----
    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_matmul: a's storage must be CudaStorage".to_string())
        })?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_matmul: b's storage must be CudaStorage".to_string())
        })?;
    use crate::StorageBackend;
    if a_storage.device() != b_storage.device() {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul: device mismatch a={} b={}",
            a_storage.device(),
            b_storage.device()
        )));
    }
    let device_index = match a_storage.device() {
        crate::Device::Cuda(i) => i,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_matmul: expected CUDA device, got {other}"
            )));
        }
    };
    // ---- allocate output ----
    // CudaStorage::alloc_uninit_ctx (#1082 perf, Pattern A) — the cudarc
    // CudaContext is pulled directly off a_storage.context(), no
    // .candle_device() read.
    let ctx = a_storage.context();
    let batch: usize = a_shape[..a_rank - 2].iter().product::<usize>().max(1);
    let mut out_shape = a_shape[..a_rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out_n_elements = batch * m * n;
    // #1082 (perf, Pattern A): this GEMM uses Epilogue::Identity (beta = 0,
    // pure C = A@B) and the per-batch loop below covers every batch, so all
    // `batch * m * n` output elements are written before any read. Allocate
    // uninitialized to skip the cudaMemsetAsync zero-fill (a full-buffer DRAM
    // write the GEMM immediately overwrites).
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, out_n_elements)?;

    // ---- acquire handle ----
    // Cold start passes the cudarc CudaContext through to
    // CublasLtMatmulHandle::new_ctx (#1082) — no candle wrapper
    // materialization required on the cold path.
    let handle = get_or_init_handle(device_index, &ctx)?;

    // ---- per-batch dispatch ----
    let bpe = dtype.size_in_bytes();
    let a_batch_stride = (m * k_a * bpe) as u64;
    let b_batch_stride = (k_b * n * bpe) as u64;
    let c_batch_stride = (m * n * bpe) as u64;

    // #1082 CUDA-graph fix: route through the thread-local active stream
    // (outside a capture scope this is exactly `ctx.default_stream()`).
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream();

    // Base device pointers for each operand. `device_ptr_raw` honors
    // both Owned and Borrowed (Phase 7 v2 zero-copy candle→kt path).
    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let a_off_root = (a.layout().start_offset() * bpe) as u64;
    let b_off_root = (b.layout().start_offset() * bpe) as u64;

    let request = MatmulRequest {
        m: m as u64,
        n: n as u64,
        k: k_a as u64,
        dtype: dtype_str.to_string(),
        a_layout: MatmulLayout::RowMajor,
        b_layout: MatmulLayout::RowMajor,
        c_layout: MatmulLayout::RowMajor,
        epilogue: Epilogue::Identity,
        concurrent_streams: 1,
    };

    for batch_i in 0..batch {
        let a_off = a_off_root + (batch_i as u64) * a_batch_stride;
        let b_off = b_off_root + (batch_i as u64) * b_batch_stride;
        let c_off = (batch_i as u64) * c_batch_stride;

        let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
        let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
        let c_ptr = (out_base + c_off) as *mut core::ffi::c_void;

        unsafe {
            handle
                .matmul(raw_stream, &request, a_ptr, b_ptr, c_ptr, std::ptr::null())
                .map_err(|e| {
                    crate::Error::Msg(format!("cuda_matmul: handle.matmul failed: {e}"))
                })?;
        }
    }

    let storage_arc: Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}

/// Run a CUDA matmul writing into a caller-provided output tensor.
///
/// Same shape / dtype contract as [`cuda_matmul`], with one extra
/// constraint: the caller-provided `dst` must already be allocated
/// with the correct output shape (`[..., M, N]`), dtype, contiguous
/// layout, and live on the same CUDA device as `a` and `b`.
///
/// The caller retains ownership of `dst`; this function only writes
/// into its existing device storage. No new `CudaStorage::zeros`
/// allocation happens.
///
/// **Why this exists** — Phase 5 batched CUDA-graph capture (#1082).
/// Stream-captured kernels record the device pointers of every tensor
/// they touch into the graph. If those pointers come from a transient
/// per-call allocation (the default [`cuda_matmul`] behavior), the
/// allocation is freed at end-of-capture and the captured kernel's
/// recorded pointer is stale on replay → `CUDA_ERROR_ILLEGAL_ADDRESS`.
/// Routing the lm-head matmul through a caller-pre-allocated output
/// buffer pins the pointer for the entire captured-graph lifetime.
///
/// Returns `Ok(())` on success — callers wrap their pre-allocated
/// candle Tensor (or kt Tensor) themselves and reuse it across
/// replays.
pub fn cuda_matmul_into(a: &Tensor, b: &Tensor, dst: &Tensor) -> Result<()> {
    // ---- validate shapes ----
    let a_rank = a.rank();
    let b_rank = b.rank();
    if a_rank < 2 || b_rank < 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_into: rank must be >= 2, got a={a_rank} b={b_rank}"
        )));
    }
    if a_rank != b_rank {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_into: rank mismatch a={a_rank} b={b_rank}"
        )));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..a_rank - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(crate::Error::Msg(format!(
                "cuda_matmul_into: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let m = a_shape[a_rank - 2];
    let k_a = a_shape[a_rank - 1];
    let k_b = b_shape[b_rank - 2];
    let n = b_shape[b_rank - 1];
    if k_a != k_b {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_into: contraction dim mismatch a.K={k_a} b.K={k_b}"
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_into: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    let dtype_str = match dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_matmul_into: unsupported dtype {other}"
            )));
        }
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_matmul_into: contiguous inputs required (call .contiguous() first)".to_string(),
        ));
    }

    // ---- validate dst shape / dtype / contiguity ----
    let mut expected_out_shape = a_shape[..a_rank - 2].to_vec();
    expected_out_shape.push(m);
    expected_out_shape.push(n);
    if dst.shape() != expected_out_shape.as_slice() {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_into: dst shape {:?} != expected {:?}",
            dst.shape(),
            expected_out_shape
        )));
    }
    if dst.dtype() != dtype {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_into: dst dtype {} != input dtype {}",
            dst.dtype(),
            dtype
        )));
    }
    if !dst.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_matmul_into: dst must be contiguous".to_string(),
        ));
    }

    // ---- resolve CUDA storage + device ----
    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_matmul_into: a's storage must be CudaStorage".to_string())
        })?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_matmul_into: b's storage must be CudaStorage".to_string())
        })?;
    let dst_storage = dst
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_matmul_into: dst's storage must be CudaStorage".to_string(),
            )
        })?;
    use crate::StorageBackend;
    if a_storage.device() != b_storage.device() || a_storage.device() != dst_storage.device() {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_into: device mismatch a={} b={} dst={}",
            a_storage.device(),
            b_storage.device(),
            dst_storage.device()
        )));
    }
    let device_index = match a_storage.device() {
        crate::Device::Cuda(i) => i,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_matmul_into: expected CUDA device, got {other}"
            )));
        }
    };
    // Stream + handle both source the cudarc CudaContext off the
    // input storage — no candle device materialization needed
    // (#1082).
    let ctx = a_storage.context();

    // ---- acquire handle ----
    let handle = get_or_init_handle(device_index, &ctx)?;

    // ---- per-batch dispatch ----
    let batch: usize = a_shape[..a_rank - 2].iter().product::<usize>().max(1);
    let bpe = dtype.size_in_bytes();
    let a_batch_stride = (m * k_a * bpe) as u64;
    let b_batch_stride = (k_b * n * bpe) as u64;
    let c_batch_stride = (m * n * bpe) as u64;

    // #1082 CUDA-graph fix: route through the thread-local active stream
    // (outside a capture scope this is exactly `ctx.default_stream()`).
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream();

    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (dst_base, _) = dst_storage.device_ptr_raw();

    let a_off_root = (a.layout().start_offset() * bpe) as u64;
    let b_off_root = (b.layout().start_offset() * bpe) as u64;
    let dst_off_root = (dst.layout().start_offset() * bpe) as u64;

    let request = MatmulRequest {
        m: m as u64,
        n: n as u64,
        k: k_a as u64,
        dtype: dtype_str.to_string(),
        a_layout: MatmulLayout::RowMajor,
        b_layout: MatmulLayout::RowMajor,
        c_layout: MatmulLayout::RowMajor,
        epilogue: Epilogue::Identity,
        concurrent_streams: 1,
    };

    for batch_i in 0..batch {
        let a_off = a_off_root + (batch_i as u64) * a_batch_stride;
        let b_off = b_off_root + (batch_i as u64) * b_batch_stride;
        let c_off = dst_off_root + (batch_i as u64) * c_batch_stride;

        let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
        let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
        let c_ptr = (dst_base + c_off) as *mut core::ffi::c_void;

        unsafe {
            handle
                .matmul(raw_stream, &request, a_ptr, b_ptr, c_ptr, std::ptr::null())
                .map_err(|e| {
                    crate::Error::Msg(format!(
                        "cuda_matmul_into: handle.matmul failed: {e}"
                    ))
                })?;
        }
    }

    Ok(())
}


/// Run a CUDA matmul with a fused per-column bias add.
///
/// Same shape contract as [`cuda_matmul`]:
///
/// - `a`: `[..., M, K]`
/// - `b`: `[K, N]` (must be 2-D; the bias epilogue requires the
///   output to be `[B*..., N]` for a single bias vector to apply)
/// - `bias`: `[N]`, same dtype as `a` and `b`
///
/// Computes `C = A @ B + bias` in a single cublasLt call using the
/// `CUBLASLT_EPILOGUE_BIAS` epilogue. Saves one kernel launch + one
/// pass over the output vs the separate `matmul` + `add` decomposition.
///
/// Bias is broadcast over the last axis of the output (the standard
/// PyTorch bias semantics).
pub fn cuda_matmul_with_bias(
    a: &Tensor,
    b: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    // ---- validate shapes ----
    let a_rank = a.rank();
    if a_rank < 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_with_bias: a must have rank >= 2, got {a_rank}"
        )));
    }
    if b.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_with_bias: b must be 2-D (per cublasLt bias-epilogue requirement), got rank {}",
            b.rank()
        )));
    }
    if bias.rank() != 1 {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_with_bias: bias must be 1-D, got rank {}",
            bias.rank()
        )));
    }

    let a_shape = a.shape();
    let m = a_shape[a_rank - 2];
    let k_a = a_shape[a_rank - 1];
    let k_b = b.shape()[0];
    let n = b.shape()[1];

    if k_a != k_b {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_with_bias: contraction dim mismatch a.K={k_a} b.K={k_b}"
        )));
    }
    if bias.shape()[0] != n {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_with_bias: bias len {} must equal N={n}",
            bias.shape()[0]
        )));
    }
    let dtype = a.dtype();
    if dtype != b.dtype() || dtype != bias.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_matmul_with_bias: dtype mismatch a={} b={} bias={}",
            a.dtype(),
            b.dtype(),
            bias.dtype()
        )));
    }
    let dtype_str = match dtype {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_matmul_with_bias: unsupported dtype {other}"
            )));
        }
    };
    if !a.is_contiguous() || !b.is_contiguous() || !bias.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_matmul_with_bias: inputs must be contiguous".to_string(),
        ));
    }

    // ---- resolve CUDA storage + device ----
    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_matmul_with_bias: a's storage must be CudaStorage".to_string())
        })?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_matmul_with_bias: b's storage must be CudaStorage".to_string())
        })?;
    let bias_storage = bias
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_matmul_with_bias: bias's storage must be CudaStorage".to_string(),
            )
        })?;
    use crate::StorageBackend;
    if a_storage.device() != b_storage.device() || a_storage.device() != bias_storage.device() {
        return Err(crate::Error::Msg(
            "cuda_matmul_with_bias: all inputs must be on the same CUDA device".to_string(),
        ));
    }
    let device_index = match a_storage.device() {
        crate::Device::Cuda(i) => i,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_matmul_with_bias: expected CUDA device, got {other}"
            )));
        }
    };
    // ---- allocate output ----
    // .context() default stream — no .candle_device() read on this
    // matmul path.
    let ctx = a_storage.context();
    let batch: usize = a_shape[..a_rank - 2].iter().product::<usize>().max(1);
    let mut out_shape = a_shape[..a_rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out_n_elements = batch * m * n;
    // #1082 (perf, Pattern A): the cublasLt GEMM runs with beta=0
    // (handle.matmul hardcodes alpha=1.0/beta=0.0, so C is never read)
    // and Epilogue::Bias only adds a bias vector — every output element
    // is written; uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, out_n_elements)?;

    // ---- acquire handle ----
    // Cold start passes the cudarc CudaContext through to
    // CublasLtMatmulHandle::new_ctx (#1082) — no candle wrapper
    // materialization required on the cold path.
    let handle = get_or_init_handle(device_index, &ctx)?;

    // ---- per-batch dispatch ----
    let bpe = dtype.size_in_bytes();
    let a_batch_stride = (m * k_a * bpe) as u64;
    let c_batch_stride = (m * n * bpe) as u64;
    // B is 2-D and shared across batches — no stride.

    // #1082 CUDA-graph fix: route through the thread-local active stream
    // (outside a capture scope this is exactly `ctx.default_stream()`).
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream();

    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (bias_base, _) = bias_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let a_off_root = (a.layout().start_offset() * bpe) as u64;
    let b_off_root = (b.layout().start_offset() * bpe) as u64;
    let bias_off_root = (bias.layout().start_offset() * bpe) as u64;

    let request = MatmulRequest {
        m: m as u64,
        n: n as u64,
        k: k_a as u64,
        dtype: dtype_str.to_string(),
        a_layout: MatmulLayout::RowMajor,
        b_layout: MatmulLayout::RowMajor,
        c_layout: MatmulLayout::RowMajor,
        epilogue: Epilogue::Bias,
        concurrent_streams: 1,
    };

    let b_ptr = (b_base + b_off_root) as *const core::ffi::c_void;
    let bias_ptr = (bias_base + bias_off_root) as *const core::ffi::c_void;

    for batch_i in 0..batch {
        let a_off = a_off_root + (batch_i as u64) * a_batch_stride;
        let c_off = (batch_i as u64) * c_batch_stride;

        let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
        let c_ptr = (out_base + c_off) as *mut core::ffi::c_void;

        unsafe {
            handle
                .matmul(raw_stream, &request, a_ptr, b_ptr, c_ptr, bias_ptr)
                .map_err(|e| {
                    crate::Error::Msg(format!(
                        "cuda_matmul_with_bias: handle.matmul failed: {e}"
                    ))
                })?;
        }
    }

    let storage_arc: Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}
