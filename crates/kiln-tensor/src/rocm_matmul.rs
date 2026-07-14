//! ROCm-side matmul that dispatches through [`kiln_rocblas::HipblasLtMatmulHandle`]
//! (Phase R.6) — the ROCm analog of [`crate::cuda_matmul`].
//!
//! A process-global per-device handle registry cold-starts one hipBLASLt handle
//! per device and shares it; the autotune algo cache is process-shared only.
//! ROCm intentionally avoids disk persistence because restored hipBLASLt algo
//! blobs can crash a later process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use kiln_hip::{RocmContext, RocmStream};
use kiln_rocblas::{AlgoCache, Epilogue, HipblasLtMatmulHandle, MatmulLayout, MatmulRequest};

pub use kiln_rocblas::HipblasLtWorkspaceLease;

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
    cache_by_device: Mutex<HashMap<usize, Arc<Mutex<AlgoCache>>>>,
}

fn handle_registry() -> &'static HandleRegistry {
    static REGISTRY: OnceLock<HandleRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| HandleRegistry {
        by_device: Mutex::new(HashMap::new()),
        cache_by_device: Mutex::new(HashMap::new()),
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
    let algo_cache = {
        let mut cache_by_device = reg.cache_by_device.lock().map_err(|_| {
            crate::Error::Msg("rocm_matmul: algo cache registry mutex poisoned".to_string())
        })?;
        Arc::clone(
            cache_by_device
                .entry(device_index)
                .or_insert_with(|| Arc::new(Mutex::new(AlgoCache::new()))),
        )
    };
    let handle = HipblasLtMatmulHandle::new_ctx(
        Arc::clone(rocm_ctx),
        device_index,
        algo_cache,
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

/// Bind a private graph-capture stream's hipBLASLt workspace lifetime to an
/// explicit lease. Failed/deferred capture attempts drop the lease immediately;
/// successful captures retain it alongside the graph.
pub fn rocm_blaslt_workspace_lease(
    device_index: usize,
    rocm_ctx: &Arc<RocmContext>,
    stream: &Arc<RocmStream>,
) -> Result<HipblasLtWorkspaceLease> {
    if rocm_ctx.ordinal() != device_index || stream.ordinal() != device_index {
        return Err(crate::Error::Msg(format!(
            "rocm_blaslt_workspace_lease: device {device_index}, context {}, and stream {} must match",
            rocm_ctx.ordinal(),
            stream.ordinal()
        )));
    }
    Ok(get_or_init_handle(device_index, rocm_ctx)?.workspace_lease(rocm_ctx, stream))
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
        batch_count: request.batch_count,
        a_batch_stride: request.a_batch_stride,
        b_batch_stride: request.b_batch_stride,
        c_batch_stride: request.c_batch_stride,
        dtype: request.dtype_name().to_string(),
        output_dtype: request.output_dtype_name().to_string(),
        a_layout: rocm_blaslt_layout(request.a_layout),
        b_layout: rocm_blaslt_layout(request.b_layout),
        c_layout: rocm_blaslt_layout(request.c_layout),
        epilogue: rocm_blaslt_epilogue(request.epilogue),
        concurrent_streams: request.concurrent_streams,
    }
}

fn rocm_matmul_execution_error(
    op: &str,
    request: &MatmulRequest,
    logical_batch: usize,
    batch_index: usize,
    error: impl std::fmt::Display,
) -> crate::Error {
    crate::Error::Msg(format!(
        "{op}: hipBLASLt matmul failed: m={} n={} k={} logical_batch={} batch_index={} dtype={} output_dtype={} a_layout={:?} b_layout={:?}: {error}",
        request.m,
        request.n,
        request.k,
        logical_batch,
        batch_index,
        request.dtype,
        request.output_dtype,
        request.a_layout,
        request.b_layout,
    ))
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

const ROCM_MATMUL_OUTPUT_ELEMS_SYNC_THRESHOLD: u128 = 1_048_576;
const ROCM_MATMUL_WORK_SYNC_THRESHOLD: u128 = 268_435_456;

fn should_sync_rocm_matmul_output(m: usize, n: usize, k: usize, batch: usize) -> bool {
    let batch = batch.max(1) as u128;
    let m = m as u128;
    let n = n as u128;
    let k = k as u128;
    let output_elems = batch.saturating_mul(m).saturating_mul(n);
    let work = output_elems.saturating_mul(k);
    output_elems >= ROCM_MATMUL_OUTPUT_ELEMS_SYNC_THRESHOLD
        || work >= ROCM_MATMUL_WORK_SYNC_THRESHOLD
}

fn should_skip_rocm_strided_batched_matmul(
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    dtype: DType,
    out_dtype: DType,
    op: &str,
) -> bool {
    if batch <= 1 || env_truthy("KILN_FORCE_ROCM_STRIDED_BATCHED_MATMUL") {
        return false;
    }
    if env_truthy("KILN_DISABLE_ROCM_STRIDED_BATCHED_MATMUL") {
        return true;
    }

    let batch_u = batch as u128;
    let m_u = m as u128;
    let n_u = n as u128;
    let k_u = k as u128;
    let output_elems = batch_u.saturating_mul(m_u).saturating_mul(n_u);
    let work = output_elems.saturating_mul(k_u);

    // ROCm 7.2 hipBLASLt strided-batched GEMM can silently return bad values
    // on large attention-shaped batches on gfx115x. The fallback below already
    // issues the same GEMM one batch at a time, which preserves exact math and
    // avoids the unstable batched path without pushing this policy into every
    // attention caller.
    let large_attention_like = batch >= 8
        && m >= 128
        && n >= 128
        && k >= 128
        && output_elems >= 1_048_576
        && work >= (1u128 << 31)
        && matches!(
            (dtype, out_dtype),
            (DType::BF16, DType::F32) | (DType::F32, DType::F32)
        );

    if large_attention_like && env_truthy("KILN_TRACE_ROCM_MATMUL_FALLBACK") {
        eprintln!(
            "kiln_rocm_matmul_skip_strided_batch op={op} m={m} n={n} k={k} batch={batch} \
             dtype={dtype} out_dtype={out_dtype} output_elems={output_elems} work={work}"
        );
    }

    large_attention_like
}

fn sync_after_rocm_matmul_if_needed(
    rocm_ctx: &Arc<RocmContext>,
    inputs: &[&RocmStorage],
    op: &str,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
) -> Result<()> {
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }

    // Legacy behavior only paid this barrier for historically risky large
    // GEMMs. Stream-ordered mode must still inspect owner lifetimes at every
    // size: a small GEMM can read a Borrowed or cross-stream allocation whose
    // final owner is dropped immediately after this function returns.
    if rocm_ctx.execution_policy().synchronization_mode
        == crate::RocmSynchronizationMode::LegacyHostBarriers
        && !should_sync_rocm_matmul_output(m, n, k, batch)
    {
        return Ok(());
    }

    // Legacy execution keeps the historical device-wide barrier: ROCm/
    // hipBLASLt has exposed stale reads after very large training GEMMs when
    // only the active stream is drained. Stream-ordered execution is an
    // explicit qualification mode and records this proven same-stream barrier
    // as skipped.
    crate::rocm_storage::rocm_synchronize_context_legacy_device_same_stream_dependency_with_inputs(
        rocm_ctx,
        inputs,
        crate::RocmSyncReason::MatmulOutput,
    )
    .map_err(|e| {
        crate::Error::Msg(format!(
            "{op}: synchronize after ROCm matmul m={m} n={n} k={k} batch={batch}: {e}"
        ))
    })
}

fn sync_rocm_device_for_matmul_boundary(
    rocm_ctx: &Arc<RocmContext>,
    inputs: &[&RocmStorage],
    op: &str,
    boundary: &str,
) -> Result<()> {
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }
    let result = if inputs.is_empty() {
        // The post-cast output was allocated and produced on the active stream;
        // there is no borrowed input owner to protect. Keep the legacy device
        // barrier, but let StreamOrdered record the proven FIFO dependency.
        crate::rocm_storage::rocm_synchronize_context_legacy_device_same_stream_dependency(
            rocm_ctx,
            crate::RocmSyncReason::MatmulCastBoundary,
        )
    } else {
        crate::rocm_storage::rocm_synchronize_context_legacy_device_same_stream_dependency_with_inputs(
            rocm_ctx,
            inputs,
            crate::RocmSyncReason::MatmulCastBoundary,
        )
    };
    result.map_err(|e| {
        crate::Error::Msg(format!(
            "{op}: synchronize {boundary} f32-output BF16 cast: {e}"
        ))
    })
}

fn rocm_bf16_output_matmul_via_f32(
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    dtype: DType,
    out_dtype: DType,
    op: &str,
) -> bool {
    if dtype != DType::BF16
        || out_dtype != DType::BF16
        || env_truthy("KILN_DISABLE_ROCM_BF16_MATMUL_F32_OUTPUT")
    {
        return false;
    }
    if env_truthy("KILN_FORCE_ROCM_BF16_MATMUL_F32_OUTPUT") {
        return true;
    }

    // ROCm 7.2 hipBLASLt on gfx115x can return non-finite BF16 output for
    // large BF16 projection GEMMs. The same failure also shows up on both
    // halves of low-rank LoRA delta shapes:
    //
    // - compression: [large M, large K] @ [small rank, large K]^T -> [large M, rank]
    // - expansion:   [large M, rank] @ [large N, rank]^T -> [large M, large N]
    //
    // FP32 accumulation/output followed by a device-side BF16 cast preserves
    // the requested result dtype while avoiding that unstable BF16-output
    // epilogue.
    let work = (batch as u128) * (m as u128) * (n as u128) * (k as u128);
    let output_elems = (batch as u128) * (m as u128) * (n as u128);
    let large_projection = m >= 1024 && n >= 512 && k >= 512 && work >= (1u128 << 31);
    let large_output = m >= 512 && n >= 512 && output_elems >= 1_048_576;
    let tall_skinny_lora_compression = m >= 1024 && n <= 64 && k >= 512;
    if (large_projection || large_output || tall_skinny_lora_compression)
        && env_truthy("KILN_TRACE_ROCM_BF16_MATMUL_F32_OUTPUT")
    {
        eprintln!("kiln_rocm_bf16_matmul_f32_output op={op} m={m} n={n} k={k} batch={batch}");
    }
    large_projection || large_output || tall_skinny_lora_compression
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
        dtype,
        out_shape,
        BlasLtMatmulLayout::RowMajor,
        BlasLtMatmulLayout::RowMajor,
        OP,
    )
}

/// Run a ROCm matmul with an explicit output dtype. Inputs still share one
/// dtype and compute uses FP32; this is for BF16/F16 inputs that need F32
/// materialized outputs, such as attention scores before softmax.
pub fn rocm_matmul_to_dtype(a: &Tensor, b: &Tensor, out_dtype: DType) -> Result<Tensor> {
    const OP: &str = "rocm_matmul_to_dtype";
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
    blaslt_dtype_name(out_dtype, OP)?;
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
        out_dtype,
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
        dtype,
        out_shape,
        BlasLtMatmulLayout::ColMajor,
        BlasLtMatmulLayout::RowMajor,
        OP,
    )
}

/// Run `a^T @ b` with an explicit output dtype, without materialising
/// `a.transpose(-2, -1).contiguous()`.
///
/// `a` is stored row-major with shape `[..., K, M]`, `b` is row-major with
/// shape `[..., K, N]`, and the result is `[..., M, N]`.
pub fn rocm_matmul_lhs_transposed_to_dtype(
    a: &Tensor,
    b: &Tensor,
    out_dtype: DType,
) -> Result<Tensor> {
    const OP: &str = "rocm_matmul_lhs_transposed_to_dtype";
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
    blaslt_dtype_name(out_dtype, OP)?;
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
        out_dtype,
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
        dtype,
        out_shape,
        BlasLtMatmulLayout::RowMajor,
        BlasLtMatmulLayout::ColMajor,
        OP,
    )
}

/// Run `a @ b^T` with an explicit output dtype, without materialising
/// `b.transpose(-2, -1).contiguous()`.
///
/// `a` is stored row-major with shape `[..., M, K]`, `b` is row-major with
/// shape `[..., N, K]`, and the result is `[..., M, N]`.
pub fn rocm_matmul_rhs_transposed_to_dtype(
    a: &Tensor,
    b: &Tensor,
    out_dtype: DType,
) -> Result<Tensor> {
    const OP: &str = "rocm_matmul_rhs_transposed_to_dtype";
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
    blaslt_dtype_name(out_dtype, OP)?;
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
        out_dtype,
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
    out_dtype: DType,
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
    if !Arc::ptr_eq(&ctx, &b_storage.context()) {
        return Err(crate::Error::Msg(format!(
            "{op}: both inputs must share one ROCm context"
        )));
    }
    let batch: usize = out_shape[..out_shape.len() - 2]
        .iter()
        .product::<usize>()
        .max(1);
    if rocm_bf16_output_matmul_via_f32(m, n, k, batch, dtype, out_dtype, op) {
        // ROCm 7.2's BF16-output hipBLASLt path is unstable for large GEMMs,
        // so this branch computes FP32 output and casts back. Keep the
        // matmul->cast handoff explicit: hipBLASLt and the cast kernel both
        // use the active ROCm stream, but long training prefill has exposed
        // stale reads when the cast is queued immediately after a huge GEMM.
        let out_f32 = rocm_matmul_dispatch(
            a,
            b,
            m,
            n,
            k,
            dtype,
            DType::F32,
            out_shape,
            a_layout,
            b_layout,
            op,
        )?;
        sync_rocm_device_for_matmul_boundary(&ctx, &[a_storage, b_storage], op, "before")?;
        let out = out_f32.to_dtype(DType::BF16)?;
        sync_rocm_device_for_matmul_boundary(&ctx, &[], op, "after")?;
        return Ok(out);
    }

    let out_n_elements = batch * m * n;
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, out_dtype, out_n_elements)?;

    let handle = get_or_init_handle(device_index, &ctx)?;

    let bpe = dtype.size_in_bytes();
    let out_bpe = out_dtype.size_in_bytes();
    let a_batch_stride_elems = (m * k) as u64;
    let b_batch_stride_elems = (k * n) as u64;
    let c_batch_stride_elems = (m * n) as u64;
    let a_batch_stride = a_batch_stride_elems * bpe as u64;
    let b_batch_stride = b_batch_stride_elems * bpe as u64;
    let c_batch_stride = c_batch_stride_elems * out_bpe as u64;

    let stream = crate::active_rocm_stream(&ctx);
    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let a_off_root = (a.layout().start_offset() * bpe) as u64;
    let b_off_root = (b.layout().start_offset() * bpe) as u64;

    let request = BlasLtMatmulRequest::new_with_output_dtype(
        m,
        n,
        k,
        dtype,
        out_dtype,
        a_layout,
        b_layout,
        BlasLtMatmulLayout::RowMajor,
        BlasLtEpilogue::Identity,
        1,
        op,
    )?
    .with_strided_batch(
        batch,
        a_batch_stride_elems,
        b_batch_stride_elems,
        c_batch_stride_elems,
        op,
    )?;
    let request = rocm_blaslt_request(request);

    if batch > 1 && !should_skip_rocm_strided_batched_matmul(m, n, k, batch, dtype, out_dtype, op) {
        let a_ptr = (a_base + a_off_root) as *const core::ffi::c_void;
        let b_ptr = (b_base + b_off_root) as *const core::ffi::c_void;
        let c_ptr = out_base as *mut core::ffi::c_void;
        let batched =
            unsafe { handle.matmul(&stream, &request, a_ptr, b_ptr, c_ptr, std::ptr::null()) };
        if batched.is_ok() {
            sync_after_rocm_matmul_if_needed(&ctx, &[a_storage, b_storage], op, m, n, k, batch)?;
            let storage_arc: Storage = Arc::new(out_storage);
            return Tensor::from_parts(
                storage_arc,
                Layout::contiguous(out_shape),
                TensorId::next(),
            );
        } else if std::env::var("KILN_TRACE_ROCM_MATMUL_FALLBACK")
            .as_deref()
            .is_ok_and(|v| matches!(v, "1" | "true" | "TRUE" | "yes" | "on"))
        {
            eprintln!(
                "kiln_rocm_matmul_fallback op={op} m={m} n={n} k={k} batch={batch} \
                 dtype={dtype} out_dtype={out_dtype} a_layout={a_layout:?} \
                 b_layout={b_layout:?} error={:?}",
                batched.err(),
            );
        }
    }

    let request = rocm_blaslt_request(BlasLtMatmulRequest::new_with_output_dtype(
        m,
        n,
        k,
        dtype,
        out_dtype,
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
                .map_err(|error| {
                    rocm_matmul_execution_error(op, &request, batch, batch_i, error)
                })?;
        }
    }

    sync_after_rocm_matmul_if_needed(&ctx, &[a_storage, b_storage], op, m, n, k, batch)?;
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
    if !Arc::ptr_eq(&ctx, &b_storage.context()) || !Arc::ptr_eq(&ctx, &dst_storage.context()) {
        return Err(crate::Error::Msg(format!(
            "{OP}: inputs and destination must share one ROCm context"
        )));
    }
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
                .map_err(|error| {
                    rocm_matmul_execution_error(OP, &request, batch, batch_i, error)
                })?;
        }
    }
    sync_after_rocm_matmul_if_needed(
        &ctx,
        &[a_storage, b_storage, dst_storage],
        OP,
        m,
        n,
        k_a,
        batch,
    )?;
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
    if !Arc::ptr_eq(&ctx, &b_storage.context()) || !Arc::ptr_eq(&ctx, &bias_storage.context()) {
        return Err(crate::Error::Msg(format!(
            "{OP}: all inputs must share one ROCm context"
        )));
    }
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
                .map_err(|error| {
                    rocm_matmul_execution_error(OP, &request, batch, batch_i, error)
                })?;
        }
    }

    sync_after_rocm_matmul_if_needed(
        &ctx,
        &[a_storage, b_storage, bias_storage],
        OP,
        m,
        n,
        k_a,
        batch,
    )?;
    let storage_arc: Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}
