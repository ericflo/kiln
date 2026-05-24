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
fn get_or_init_handle(
    device_index: usize,
    candle_device: Arc<candle_core::cuda_backend::CudaDevice>,
) -> Result<Arc<CublasLtMatmulHandle>> {
    let reg = handle_registry();
    let mut by_device = reg
        .by_device
        .lock()
        .map_err(|_| crate::Error::Msg("cuda_matmul: handle registry mutex poisoned".to_string()))?;
    if let Some(h) = by_device.get(&device_index) {
        return Ok(Arc::clone(h));
    }
    // Cold-start the handle for this device.
    let handle = CublasLtMatmulHandle::new(
        candle_device,
        device_index,
        Arc::clone(&reg.shared_cache),
        None,
    )
    .map_err(|e| {
        crate::Error::Msg(format!(
            "cuda_matmul: CublasLtMatmulHandle::new failed for device {device_index}: {e}"
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
    let candle_device = a_storage.candle_device().clone();

    // ---- allocate output ----
    let batch: usize = a_shape[..a_rank - 2].iter().product::<usize>().max(1);
    let mut out_shape = a_shape[..a_rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out_n_elements = batch * m * n;
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, out_n_elements)?;

    // ---- acquire handle ----
    let handle = get_or_init_handle(device_index, candle_device.clone())?;

    // ---- per-batch dispatch ----
    let bpe = dtype.size_in_bytes();
    let a_batch_stride = (m * k_a * bpe) as u64;
    let b_batch_stride = (k_b * n * bpe) as u64;
    let c_batch_stride = (m * n * bpe) as u64;

    let stream = candle_device.cuda_stream();
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
    let candle_device = a_storage.candle_device().clone();

    // ---- allocate output ----
    let batch: usize = a_shape[..a_rank - 2].iter().product::<usize>().max(1);
    let mut out_shape = a_shape[..a_rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out_n_elements = batch * m * n;
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, out_n_elements)?;

    // ---- acquire handle ----
    let handle = get_or_init_handle(device_index, candle_device.clone())?;

    // ---- per-batch dispatch ----
    let bpe = dtype.size_in_bytes();
    let a_batch_stride = (m * k_a * bpe) as u64;
    let c_batch_stride = (m * n * bpe) as u64;
    // B is 2-D and shared across batches — no stride.

    let stream = candle_device.cuda_stream();
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
