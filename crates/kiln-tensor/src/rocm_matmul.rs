//! ROCm-side matmul that dispatches through [`kiln_rocblas::HipblasLtMatmulHandle`]
//! (Phase R.6) — the ROCm analog of [`crate::cuda_matmul`].
//!
//! A process-global per-device handle registry cold-starts one hipBLASLt handle
//! per device and shares it; the autotune algo cache is process-shared only.
//! ROCm intentionally avoids disk persistence because restored hipBLASLt algo
//! blobs can crash a later process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use kiln_hip::RocmContext;
use kiln_rocblas::{AlgoCache, Epilogue, HipblasLtMatmulHandle, MatmulLayout, MatmulRequest};

use crate::blaslt_request::{
    BlasLtEpilogue, BlasLtMatmulLayout, BlasLtMatmulRequest, blaslt_dtype_name,
};
use crate::rocm_storage::RocmStorage;
use crate::{DType, Layout, Result, Storage, Tensor, TensorId};

// ----------------------------------------------------------------------
// Process-global per-device handle registry
// ----------------------------------------------------------------------

struct HandleRegistry {
    by_device: Mutex<HashMap<usize, Arc<HipblasLtMatmulHandle>>>,
    shared_cache: Arc<Mutex<AlgoCache>>,
}

fn handle_registry() -> &'static HandleRegistry {
    static REGISTRY: OnceLock<HandleRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| HandleRegistry {
        by_device: Mutex::new(HashMap::new()),
        shared_cache: Arc::new(Mutex::new(AlgoCache::new())),
    })
}

/// Acquire (or cold-start) a hipBLASLt handle for `device_index`.
fn get_or_init_handle(
    device_index: usize,
    rocm_ctx: &Arc<RocmContext>,
) -> Result<Arc<HipblasLtMatmulHandle>> {
    let reg = handle_registry();
    let mut by_device = reg.by_device.lock().map_err(|_| {
        crate::Error::Msg("rocm_matmul: handle registry mutex poisoned".to_string())
    })?;
    if let Some(h) = by_device.get(&device_index) {
        return Ok(Arc::clone(h));
    }
    let handle = HipblasLtMatmulHandle::new_ctx(
        Arc::clone(rocm_ctx),
        device_index,
        Arc::clone(&reg.shared_cache),
        None,
    )
    .map_err(|e| {
        crate::Error::Msg(format!(
            "rocm_matmul: HipblasLtMatmulHandle::new_ctx failed for device {device_index}: {e}"
        ))
    })?;
    let arc = Arc::new(handle);
    by_device.insert(device_index, Arc::clone(&arc));
    Ok(arc)
}

fn rocm_storage<'a>(t: &'a Tensor, op: &str, which: &str) -> Result<&'a RocmStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{op}: {which}'s storage must be RocmStorage")))
}

fn rocm_device_index(t: &RocmStorage, op: &str) -> Result<usize> {
    use crate::StorageBackend;
    match t.device() {
        crate::Device::Rocm(i) => Ok(i),
        other => Err(crate::Error::Msg(format!(
            "{op}: expected ROCm device, got {other}"
        ))),
    }
}

fn rocm_blaslt_layout(layout: BlasLtMatmulLayout) -> MatmulLayout {
    match layout {
        BlasLtMatmulLayout::RowMajor => MatmulLayout::RowMajor,
        BlasLtMatmulLayout::ColMajor => MatmulLayout::ColMajor,
    }
}

fn rocm_blaslt_epilogue(epilogue: BlasLtEpilogue) -> Epilogue {
    match epilogue {
        BlasLtEpilogue::Identity => Epilogue::Identity,
        BlasLtEpilogue::Bias => Epilogue::Bias,
    }
}

fn rocm_blaslt_request(request: BlasLtMatmulRequest) -> MatmulRequest {
    MatmulRequest {
        m: request.m,
        n: request.n,
        k: request.k,
        dtype: request.dtype_name().to_string(),
        a_layout: rocm_blaslt_layout(request.a_layout),
        b_layout: rocm_blaslt_layout(request.b_layout),
        c_layout: rocm_blaslt_layout(request.c_layout),
        epilogue: rocm_blaslt_epilogue(request.epilogue),
        concurrent_streams: request.concurrent_streams,
    }
}

// ----------------------------------------------------------------------
// Public entry points
// ----------------------------------------------------------------------

/// Run a ROCm matmul `[..., M, K] @ [..., K, N] = [..., M, N]`. BF16/F16/F32,
/// contiguous inputs, same device. The ROCm analog of [`crate::cuda_matmul`].
pub fn rocm_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    const OP: &str = "rocm_matmul";
    let a_rank = a.rank();
    let b_rank = b.rank();
    if a_rank < 2 || b_rank < 2 {
        return Err(crate::Error::Msg(format!(
            "{OP}: rank must be >= 2, got a={a_rank} b={b_rank}"
        )));
    }
    if a_rank != b_rank {
        return Err(crate::Error::Msg(format!(
            "{OP}: rank mismatch a={a_rank} b={b_rank}"
        )));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..a_rank - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(crate::Error::Msg(format!(
                "{OP}: batch axis {axis} mismatch: a={} b={}",
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
            "{OP}: contraction dim mismatch a.K={k_a} b.K={k_b}"
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "{OP}: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    blaslt_dtype_name(dtype, OP)?;
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{OP}: contiguous inputs required"
        )));
    }

    let mut out_shape = a_shape[..a_rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    rocm_matmul_dispatch(
        a,
        b,
        m,
        n,
        k_a,
        dtype,
        out_shape,
        BlasLtMatmulLayout::RowMajor,
        BlasLtMatmulLayout::RowMajor,
        OP,
    )
}

/// Run `a^T @ b` without materialising `a.transpose(-2, -1).contiguous()`.
///
/// `a` is stored row-major with shape `[..., K, M]`, `b` is row-major with
/// shape `[..., K, N]`, and the result is `[..., M, N]`. This mirrors
/// [`crate::cuda_matmul_lhs_transposed`] for the ROCm/hipBLASLt backend and is
/// used by long-context LoRA backward to avoid allocating the giant
/// `[out_features, rows]` transposed gradient view.
pub fn rocm_matmul_lhs_transposed(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    const OP: &str = "rocm_matmul_lhs_transposed";
    let a_rank = a.rank();
    let b_rank = b.rank();
    if a_rank < 2 || b_rank < 2 {
        return Err(crate::Error::Msg(format!(
            "{OP}: rank must be >= 2, got a={a_rank} b={b_rank}"
        )));
    }
    if a_rank != b_rank {
        return Err(crate::Error::Msg(format!(
            "{OP}: rank mismatch a={a_rank} b={b_rank}"
        )));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..a_rank - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(crate::Error::Msg(format!(
                "{OP}: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let k_a = a_shape[a_rank - 2];
    let m = a_shape[a_rank - 1];
    let k_b = b_shape[b_rank - 2];
    let n = b_shape[b_rank - 1];
    if k_a != k_b {
        return Err(crate::Error::Msg(format!(
            "{OP}: contraction dim mismatch a.K={k_a} b.K={k_b}"
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "{OP}: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    blaslt_dtype_name(dtype, OP)?;
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{OP}: contiguous inputs required"
        )));
    }

    let mut out_shape = a_shape[..a_rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    rocm_matmul_dispatch(
        a,
        b,
        m,
        n,
        k_a,
        dtype,
        out_shape,
        BlasLtMatmulLayout::ColMajor,
        BlasLtMatmulLayout::RowMajor,
        OP,
    )
}

/// Run `a @ b^T` without materialising `b.transpose(-2, -1).contiguous()`.
///
/// `a` is stored row-major with shape `[..., M, K]`, `b` is row-major with
/// shape `[..., N, K]`, and the result is `[..., M, N]`.
pub fn rocm_matmul_rhs_transposed(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    const OP: &str = "rocm_matmul_rhs_transposed";
    let a_rank = a.rank();
    let b_rank = b.rank();
    if a_rank < 2 || b_rank < 2 {
        return Err(crate::Error::Msg(format!(
            "{OP}: rank must be >= 2, got a={a_rank} b={b_rank}"
        )));
    }
    if a_rank != b_rank {
        return Err(crate::Error::Msg(format!(
            "{OP}: rank mismatch a={a_rank} b={b_rank}"
        )));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..a_rank - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(crate::Error::Msg(format!(
                "{OP}: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let m = a_shape[a_rank - 2];
    let k_a = a_shape[a_rank - 1];
    let n = b_shape[b_rank - 2];
    let k_b = b_shape[b_rank - 1];
    if k_a != k_b {
        return Err(crate::Error::Msg(format!(
            "{OP}: contraction dim mismatch a.K={k_a} b.K={k_b}"
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "{OP}: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    blaslt_dtype_name(dtype, OP)?;
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{OP}: contiguous inputs required"
        )));
    }

    let mut out_shape = a_shape[..a_rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    rocm_matmul_dispatch(
        a,
        b,
        m,
        n,
        k_a,
        dtype,
        out_shape,
        BlasLtMatmulLayout::RowMajor,
        BlasLtMatmulLayout::ColMajor,
        OP,
    )
}

#[allow(clippy::too_many_arguments)]
fn rocm_matmul_dispatch(
    a: &Tensor,
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    dtype: DType,
    out_shape: Vec<usize>,
    a_layout: BlasLtMatmulLayout,
    b_layout: BlasLtMatmulLayout,
    op: &str,
) -> Result<Tensor> {
    let a_storage = rocm_storage(a, op, "a")?;
    let b_storage = rocm_storage(b, op, "b")?;
    use crate::StorageBackend;
    if a_storage.device() != b_storage.device() {
        return Err(crate::Error::Msg(format!(
            "{op}: device mismatch a={} b={}",
            a_storage.device(),
            b_storage.device()
        )));
    }
    let device_index = rocm_device_index(a_storage, op)?;

    let ctx = a_storage.context();
    let batch: usize = out_shape[..out_shape.len() - 2]
        .iter()
        .product::<usize>()
        .max(1);
    let out_n_elements = batch * m * n;
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, out_n_elements)?;

    let handle = get_or_init_handle(device_index, &ctx)?;

    let bpe = dtype.size_in_bytes();
    let a_batch_stride = (m * k * bpe) as u64;
    let b_batch_stride = (k * n * bpe) as u64;
    let c_batch_stride = (m * n * bpe) as u64;

    let stream = crate::active_rocm_stream(&ctx);
    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let a_off_root = (a.layout().start_offset() * bpe) as u64;
    let b_off_root = (b.layout().start_offset() * bpe) as u64;

    let request = rocm_blaslt_request(BlasLtMatmulRequest::new(
        m,
        n,
        k,
        dtype,
        a_layout,
        b_layout,
        BlasLtMatmulLayout::RowMajor,
        BlasLtEpilogue::Identity,
        1,
        op,
    )?);

    for batch_i in 0..batch {
        let a_off = a_off_root + (batch_i as u64) * a_batch_stride;
        let b_off = b_off_root + (batch_i as u64) * b_batch_stride;
        let c_off = (batch_i as u64) * c_batch_stride;
        let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
        let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
        let c_ptr = (out_base + c_off) as *mut core::ffi::c_void;
        unsafe {
            handle
                .matmul(&stream, &request, a_ptr, b_ptr, c_ptr, std::ptr::null())
                .map_err(|e| crate::Error::Msg(format!("{op}: handle.matmul failed: {e}")))?;
        }
    }

    let storage_arc: Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}

/// ROCm matmul writing into a caller-provided `dst` (pinned-pointer path for
/// HIP-graph capture). ROCm analog of [`crate::cuda_matmul_into`].
pub fn rocm_matmul_into(a: &Tensor, b: &Tensor, dst: &Tensor) -> Result<()> {
    const OP: &str = "rocm_matmul_into";
    let a_rank = a.rank();
    let b_rank = b.rank();
    if a_rank < 2 || b_rank < 2 || a_rank != b_rank {
        return Err(crate::Error::Msg(format!(
            "{OP}: rank must be >= 2 and equal, got a={a_rank} b={b_rank}"
        )));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..a_rank - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(crate::Error::Msg(format!(
                "{OP}: batch axis {axis} mismatch"
            )));
        }
    }
    let m = a_shape[a_rank - 2];
    let k_a = a_shape[a_rank - 1];
    let k_b = b_shape[b_rank - 2];
    let n = b_shape[b_rank - 1];
    if k_a != k_b {
        return Err(crate::Error::Msg(format!(
            "{OP}: contraction dim mismatch a.K={k_a} b.K={k_b}"
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!("{OP}: dtype mismatch")));
    }
    let dtype = a.dtype();
    blaslt_dtype_name(dtype, OP)?;
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{OP}: contiguous inputs required"
        )));
    }
    let mut expected = a_shape[..a_rank - 2].to_vec();
    expected.push(m);
    expected.push(n);
    if dst.shape() != expected.as_slice() {
        return Err(crate::Error::Msg(format!(
            "{OP}: dst shape {:?} != expected {:?}",
            dst.shape(),
            expected
        )));
    }
    if dst.dtype() != dtype || !dst.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{OP}: dst must match dtype and be contiguous"
        )));
    }

    let a_storage = rocm_storage(a, OP, "a")?;
    let b_storage = rocm_storage(b, OP, "b")?;
    let dst_storage = rocm_storage(dst, OP, "dst")?;
    use crate::StorageBackend;
    if a_storage.device() != b_storage.device() || a_storage.device() != dst_storage.device() {
        return Err(crate::Error::Msg(format!("{OP}: device mismatch")));
    }
    let device_index = rocm_device_index(a_storage, OP)?;
    let ctx = a_storage.context();
    let handle = get_or_init_handle(device_index, &ctx)?;

    let batch: usize = a_shape[..a_rank - 2].iter().product::<usize>().max(1);
    let bpe = dtype.size_in_bytes();
    let a_batch_stride = (m * k_a * bpe) as u64;
    let b_batch_stride = (k_b * n * bpe) as u64;
    let c_batch_stride = (m * n * bpe) as u64;

    let stream = crate::active_rocm_stream(&ctx);
    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (dst_base, _) = dst_storage.device_ptr_raw();
    let a_off_root = (a.layout().start_offset() * bpe) as u64;
    let b_off_root = (b.layout().start_offset() * bpe) as u64;
    let dst_off_root = (dst.layout().start_offset() * bpe) as u64;

    let request = rocm_blaslt_request(BlasLtMatmulRequest::new(
        m,
        n,
        k_a,
        dtype,
        BlasLtMatmulLayout::RowMajor,
        BlasLtMatmulLayout::RowMajor,
        BlasLtMatmulLayout::RowMajor,
        BlasLtEpilogue::Identity,
        1,
        OP,
    )?);

    for batch_i in 0..batch {
        let a_off = a_off_root + (batch_i as u64) * a_batch_stride;
        let b_off = b_off_root + (batch_i as u64) * b_batch_stride;
        let c_off = dst_off_root + (batch_i as u64) * c_batch_stride;
        let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
        let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
        let c_ptr = (dst_base + c_off) as *mut core::ffi::c_void;
        unsafe {
            handle
                .matmul(&stream, &request, a_ptr, b_ptr, c_ptr, std::ptr::null())
                .map_err(|e| crate::Error::Msg(format!("{OP}: handle.matmul failed: {e}")))?;
        }
    }
    Ok(())
}

/// ROCm matmul with a fused per-column bias add (`C = A@B + bias`, bias `[N]`).
/// `b` must be 2-D. ROCm analog of [`crate::cuda_matmul_with_bias`].
pub fn rocm_matmul_with_bias(a: &Tensor, b: &Tensor, bias: &Tensor) -> Result<Tensor> {
    const OP: &str = "rocm_matmul_with_bias";
    let a_rank = a.rank();
    if a_rank < 2 {
        return Err(crate::Error::Msg(format!(
            "{OP}: a must have rank >= 2, got {a_rank}"
        )));
    }
    if b.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "{OP}: b must be 2-D, got rank {}",
            b.rank()
        )));
    }
    if bias.rank() != 1 {
        return Err(crate::Error::Msg(format!(
            "{OP}: bias must be 1-D, got rank {}",
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
            "{OP}: contraction dim mismatch a.K={k_a} b.K={k_b}"
        )));
    }
    if bias.shape()[0] != n {
        return Err(crate::Error::Msg(format!(
            "{OP}: bias len {} must equal N={n}",
            bias.shape()[0]
        )));
    }
    let dtype = a.dtype();
    if dtype != b.dtype() || dtype != bias.dtype() {
        return Err(crate::Error::Msg(format!("{OP}: dtype mismatch")));
    }
    blaslt_dtype_name(dtype, OP)?;
    if !a.is_contiguous() || !b.is_contiguous() || !bias.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{OP}: inputs must be contiguous"
        )));
    }

    let a_storage = rocm_storage(a, OP, "a")?;
    let b_storage = rocm_storage(b, OP, "b")?;
    let bias_storage = rocm_storage(bias, OP, "bias")?;
    use crate::StorageBackend;
    if a_storage.device() != b_storage.device() || a_storage.device() != bias_storage.device() {
        return Err(crate::Error::Msg(format!(
            "{OP}: all inputs must be on the same ROCm device"
        )));
    }
    let device_index = rocm_device_index(a_storage, OP)?;
    let ctx = a_storage.context();
    let batch: usize = a_shape[..a_rank - 2].iter().product::<usize>().max(1);
    let mut out_shape = a_shape[..a_rank - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out_n_elements = batch * m * n;
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, out_n_elements)?;

    let handle = get_or_init_handle(device_index, &ctx)?;

    let bpe = dtype.size_in_bytes();
    let a_batch_stride = (m * k_a * bpe) as u64;
    let c_batch_stride = (m * n * bpe) as u64;

    let stream = crate::active_rocm_stream(&ctx);
    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (bias_base, _) = bias_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let a_off_root = (a.layout().start_offset() * bpe) as u64;
    let b_off_root = (b.layout().start_offset() * bpe) as u64;
    let bias_off_root = (bias.layout().start_offset() * bpe) as u64;

    let request = rocm_blaslt_request(BlasLtMatmulRequest::new(
        m,
        n,
        k_a,
        dtype,
        BlasLtMatmulLayout::RowMajor,
        BlasLtMatmulLayout::RowMajor,
        BlasLtMatmulLayout::RowMajor,
        BlasLtEpilogue::Bias,
        1,
        OP,
    )?);

    let b_ptr = (b_base + b_off_root) as *const core::ffi::c_void;
    let bias_ptr = (bias_base + bias_off_root) as *const core::ffi::c_void;

    for batch_i in 0..batch {
        let a_off = a_off_root + (batch_i as u64) * a_batch_stride;
        let c_off = (batch_i as u64) * c_batch_stride;
        let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
        let c_ptr = (out_base + c_off) as *mut core::ffi::c_void;
        unsafe {
            handle
                .matmul(&stream, &request, a_ptr, b_ptr, c_ptr, bias_ptr)
                .map_err(|e| crate::Error::Msg(format!("{OP}: handle.matmul failed: {e}")))?;
        }
    }

    let storage_arc: Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}
