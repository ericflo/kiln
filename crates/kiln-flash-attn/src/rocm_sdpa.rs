//! ROCm scaled-dot-product attention.
//!
//! The hot bf16/head_dim=128/256 forward paths use native exact HIP kernels that
//! stream attention through saved log-sum-exp without materializing `[sq, sk]`
//! scores. Backward uses an exact BLASLt materialized composite when the
//! governor-published operation peak permits, and falls back to bounded paths
//! for tiles that would be too large. Other shapes use the same correct, **fully
//! on-device** scaled-dot-product-attention composite built out of the
//! parity-tested `kiln_tensor` ROCm primitives:
//!
//! - `rocm_matmul` (batched rank-3 QK^T / PV via per-(b,h) unroll),
//! - `rocm_softmax_last_axis`,
//! - `rocm_scalar_op` (scale), `rocm_max_axis` / `rocm_sum_axis` (lse),
//! - `rocm_index_select_axis_n` (GQA expand + paged gather),
//! - `rocm_causal_mask_fill`, `rocm_masked_fill` (tail masks),
//!   `rocm_contiguous` (materialize views),
//! - `rocm_cast` (bf16 <-> f32 accumulation).
//!
//! The hot QK^T / softmax / PV path never leaves `Device::Rocm`; only the tiny
//! `[b*h, sq]` log-sum-exp tail is host-staged (a `ln` over the reduced tensor —
//! ROCm has no native `exp`/`ln` op), which is negligible next to the matmuls.
//!
//! Mirrors the CUDA `kt_api::*` entry points one-for-one under
//! `cfg(feature = "rocm")`; the CUDA arms are untouched (cfg-gated).

#![cfg(feature = "rocm")]

use half::bf16;
use kiln_tensor::{
    DType as KtDType, Device as KtDevice, RocmFlashAttentionPolicy, RocmFlashAttentionRouteMode,
    RocmStorage, RocmStreamSubmission, Tensor as KtTensor,
};

use crate::kt_api::FlashAttnError;
use crate::score_policy::effective_score_geometry;

/// `ScalarKind::MulScalar` tag (see `kiln_tensor::ops::scalar`).
const SCALAR_MUL: i32 = 2;

/// Additive-mask fill value for masked (future) attention positions. A large
/// finite negative number rather than `f32::NEG_INFINITY` so that BF16 / F16
/// downcasts in the softmax stay well-defined (`exp` underflows to 0).
const NEG_FILL: f32 = -1.0e30;
const MATERIALIZED_SCORE_SCRATCH_BUFFERS: usize = 3;
const MATERIALIZED_BWD_SCORE_SCRATCH_BUFFERS: usize = 8;
const MATERIALIZED_SCORE_TILE_GRANULARITY: usize = 128;
const MAX_MATERIALIZED_QUERY_TILES: usize = 1024;
const ONLINE_TILE_GRANULARITY: usize = 128;
const MAX_ONLINE_TILE_PAIRS: usize = 16 * 1024;

fn map_kt<T>(r: Result<T, kiln_tensor::Error>) -> Result<T, FlashAttnError> {
    r.map_err(|e| FlashAttnError::Msg(format!("rocm-sdpa: {e}")))
}

/// Settle one external ROCm FFI admission using `rocm_flash_api.cpp`'s status
/// convention: zero is success, negative is a local/unsupported decline, and
/// positive means an attempted device execution failed. Only the last class
/// poisons the process-lifetime device gate.
fn settle_rocm_ffi_submission(
    submission: RocmStreamSubmission,
    status: i32,
    operation: &'static str,
) -> Result<(), FlashAttnError> {
    if status > 0 {
        submission.quarantine();
        return Err(FlashAttnError::Msg(format!(
            "{operation}: ROCm execution failed with status {status}; device quarantined"
        )));
    }
    submission.complete();
    Ok(())
}

/// Materialize a (possibly strided) ROCm tensor into a fresh contiguous ROCm
/// tensor, on-device. `Tensor::contiguous()` has no ROCm arm, so route through
/// the native `rocm_contiguous` strided-copy kernel.
fn rocm_contig(t: &KtTensor) -> Result<KtTensor, FlashAttnError> {
    if t.is_contiguous() {
        return Ok(t.clone());
    }
    map_kt(kiln_tensor::rocm_contiguous(t))
}

/// Cast a ROCm tensor to `target` dtype on-device (no-op clone if already that
/// dtype). Input is made contiguous first (the native kernel requires it).
fn rocm_cast_to(t: &KtTensor, target: KtDType) -> Result<KtTensor, FlashAttnError> {
    if t.dtype() == target {
        return rocm_contig(t);
    }
    let c = rocm_contig(t)?;
    map_kt(kiln_tensor::rocm_cast(&c, target))
}

fn rocm_storage(tensor: &KtTensor) -> Result<&RocmStorage, FlashAttnError> {
    tensor
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| FlashAttnError::Msg("rocm-sdpa: input must use ROCm storage".into()))
}

fn rocm_zeros_in_tensor_context(
    source: &KtTensor,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, FlashAttnError> {
    kiln_kt_bridge::alloc_rocm_tensor(rocm_storage(source)?, dtype, shape)
        .map_err(|e| FlashAttnError::Msg(e.to_string()))
}

fn rocm_flash_attention_policy(
    tensor: &KtTensor,
) -> Result<RocmFlashAttentionPolicy, FlashAttnError> {
    Ok(rocm_storage(tensor)?
        .context()
        .execution_policy()
        .tensor_kernels
        .flash_attention)
}

fn rocm_execution_quarantined(tensor: &KtTensor) -> Result<bool, FlashAttnError> {
    Ok(rocm_storage(tensor)?.context().cleanup_quarantined())
}

fn rocm_flash_attention_ffi_policy(
    policy: RocmFlashAttentionPolicy,
) -> Result<crate::RocmFlashAttentionFfiPolicy, FlashAttnError> {
    let to_i32 = |name: &'static str, value: usize| {
        i32::try_from(value).map_err(|_| {
            FlashAttnError::Msg(format!(
                "rocm-sdpa: {name} exceeds the C++ dispatch boundary"
            ))
        })
    };
    Ok(crate::RocmFlashAttentionFfiPolicy {
        native_gqa_qblock_forward: i32::from(policy.native_gqa_qblock_forward),
        native_gqa_qblock_forward_min_sequence: to_i32(
            "native_gqa_qblock_forward_min_sequence",
            policy.native_gqa_qblock_forward_min_sequence,
        )?,
        wmma_gqa_qblock_forward: i32::from(policy.wmma_gqa_qblock_forward),
        wmma_gqa_r64k32_forward: i32::from(policy.wmma_gqa_r64k32_forward),
        wmma_gqa_r64k32_forward_min_sequence: to_i32(
            "wmma_gqa_r64k32_forward_min_sequence",
            policy.wmma_gqa_r64k32_forward_min_sequence,
        )?,
        wmma_gqa_r64k32_log2_forward: i32::from(policy.wmma_gqa_r64k32_log2_forward),
        wmma_gqa_r64k32_log2_forward_min_sequence: to_i32(
            "wmma_gqa_r64k32_log2_forward_min_sequence",
            policy.wmma_gqa_r64k32_log2_forward_min_sequence,
        )?,
        backward_precompute_delta_max_sequence: to_i32(
            "backward_precompute_delta_max_sequence",
            policy.backward_precompute_delta_max_sequence,
        )?,
        native_direct_collapsed_gqa_query_parallelism: to_i32(
            "native_direct_collapsed_gqa_query_parallelism",
            policy.native_direct_collapsed_gqa_query_parallelism,
        )?,
    })
}

fn rocm_sync_tensor(tensor: &KtTensor) -> Result<(), FlashAttnError> {
    map_kt(kiln_tensor::rocm_synchronize_tensor_stream_for(
        tensor,
        kiln_tensor::RocmSyncReason::ExplicitStreamDrain,
    ))
}

fn f32_matmul_inner_tile_len(tensor: &KtTensor) -> Result<usize, FlashAttnError> {
    Ok(rocm_flash_attention_policy(tensor)?.f32_matmul_inner_tile)
}

fn rocm_matmul_split_inner_f32(
    lhs: &KtTensor,
    rhs: &KtTensor,
    label: &str,
) -> Result<KtTensor, FlashAttnError> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs.dtype() != KtDType::F32
        || rhs.dtype() != KtDType::F32
        || lhs_shape.len() != 3
        || rhs_shape.len() != 3
        || lhs_shape[0] != rhs_shape[0]
        || lhs_shape[2] != rhs_shape[1]
    {
        return Err(FlashAttnError::Msg(format!(
            "{label}: expected f32 batched matmul [batch, m, k] @ [batch, k, n], got lhs {:?} {} rhs {:?} {}",
            lhs_shape,
            lhs.dtype(),
            rhs_shape,
            rhs.dtype()
        )));
    }

    let inner = lhs_shape[2];
    let tile = f32_matmul_inner_tile_len(lhs)?;
    if inner <= tile {
        let out = rocm_matmul_f32_batch_grouped(lhs, rhs, label)?;
        rocm_sync_tensor(&out)?;
        return Ok(out);
    }

    let mut acc: Option<KtTensor> = None;
    let mut start = 0usize;
    while start < inner {
        let len = (inner - start).min(tile);
        let lhs_block = rocm_contig(&map_kt(lhs.narrow(2, start, len))?)?;
        let rhs_block = rocm_contig(&map_kt(rhs.narrow(1, start, len))?)?;
        rocm_sync_tensor(&rhs_block)?;
        let partial = rocm_matmul_f32_batch_grouped(&lhs_block, &rhs_block, label)?;
        rocm_sync_tensor(&partial)?;
        acc = Some(match acc.take() {
            Some(prev) => {
                let sum = map_kt(prev.add(&partial))?;
                rocm_sync_tensor(&sum)?;
                drop(prev);
                drop(partial);
                sum
            }
            None => partial,
        });
        start += len;
    }

    let out = acc.ok_or_else(|| FlashAttnError::Msg(format!("{label}: empty inner dimension")))?;
    Ok(out)
}

fn online_matmul_batch_group(tensor: &KtTensor) -> Result<usize, FlashAttnError> {
    Ok(rocm_flash_attention_policy(tensor)?.online_matmul_batch_group)
}

fn should_group_online_batch_matmul(
    tensor: &KtTensor,
    batch: usize,
) -> Result<bool, FlashAttnError> {
    Ok(batch > online_matmul_batch_group(tensor)?)
}

fn rocm_cat_axis0(chunks: &[KtTensor], label: &str) -> Result<KtTensor, FlashAttnError> {
    if chunks.is_empty() {
        return Err(FlashAttnError::Msg(format!("{label}: empty batch chunks")));
    }
    if chunks.len() == 1 {
        return Ok(chunks[0].clone());
    }
    let refs: Vec<&KtTensor> = chunks.iter().collect();
    map_kt(KtTensor::cat(&refs, 0))
}

fn rocm_matmul_f32_batch_grouped(
    lhs: &KtTensor,
    rhs: &KtTensor,
    label: &str,
) -> Result<KtTensor, FlashAttnError> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() != 3 || rhs_shape.len() != 3 || lhs_shape[0] != rhs_shape[0] {
        return map_kt(kiln_tensor::rocm_matmul(lhs, rhs));
    }
    let batch = lhs_shape[0];
    if !should_group_online_batch_matmul(lhs, batch)? {
        return map_kt(kiln_tensor::rocm_matmul(lhs, rhs));
    }

    let group = online_matmul_batch_group(lhs)?.min(batch);
    let mut chunks = Vec::with_capacity(batch.div_ceil(group));
    let mut start = 0usize;
    while start < batch {
        let len = (batch - start).min(group);
        let lhs_chunk = rocm_contig(&map_kt(lhs.narrow(0, start, len))?)?;
        let rhs_chunk = rocm_contig(&map_kt(rhs.narrow(0, start, len))?)?;
        let chunk = map_kt(kiln_tensor::rocm_matmul(&lhs_chunk, &rhs_chunk))?;
        chunks.push(chunk);
        start += len;
    }
    let out = rocm_cat_axis0(&chunks, label)?;
    rocm_sync_tensor(&out)?;
    Ok(out)
}

fn rocm_matmul_rhs_transposed_to_dtype_batch_grouped(
    lhs: &KtTensor,
    rhs: &KtTensor,
    out_dtype: KtDType,
    label: &str,
) -> Result<KtTensor, FlashAttnError> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() != 3 || rhs_shape.len() != 3 || lhs_shape[0] != rhs_shape[0] {
        return map_kt(kiln_tensor::rocm_matmul_rhs_transposed_to_dtype(
            lhs, rhs, out_dtype,
        ));
    }
    let batch = lhs_shape[0];
    if !should_group_online_batch_matmul(lhs, batch)? {
        return map_kt(kiln_tensor::rocm_matmul_rhs_transposed_to_dtype(
            lhs, rhs, out_dtype,
        ));
    }

    let group = online_matmul_batch_group(lhs)?.min(batch);
    let mut chunks = Vec::with_capacity(batch.div_ceil(group));
    let mut start = 0usize;
    while start < batch {
        let len = (batch - start).min(group);
        let lhs_chunk = rocm_contig(&map_kt(lhs.narrow(0, start, len))?)?;
        let rhs_chunk = rocm_contig(&map_kt(rhs.narrow(0, start, len))?)?;
        let chunk = map_kt(kiln_tensor::rocm_matmul_rhs_transposed_to_dtype(
            &lhs_chunk, &rhs_chunk, out_dtype,
        ))?;
        chunks.push(chunk);
        start += len;
    }
    let out = rocm_cat_axis0(&chunks, label)?;
    rocm_sync_tensor(&out)?;
    Ok(out)
}

fn published_rocm_available_bytes(device: KtDevice) -> Option<usize> {
    let selector = device.memory_probe_selector();
    if !matches!(device, KtDevice::Rocm(_)) {
        return None;
    }
    let configured = kiln_memory::MemoryGovernor::global_configuration();
    if configured.selector != selector {
        return None;
    }
    kiln_memory::MemoryGovernor::try_global_cached_available_bytes()
        .map(|available| available.min(usize::MAX as u64) as usize)
}

fn require_published_rocm_available_bytes(
    device: KtDevice,
    operation: &str,
) -> Result<usize, FlashAttnError> {
    published_rocm_available_bytes(device).ok_or_else(|| {
        FlashAttnError::Msg(format!(
            "ROCm {operation} rejected: no initialized memory governor snapshot matches the active ROCm device"
        ))
    })
}

fn reserve_published_rocm_bytes(
    bytes: usize,
    operation: &str,
) -> Result<kiln_memory::governor::Reservation<'static>, FlashAttnError> {
    let bytes_u64 = u64::try_from(bytes).map_err(|_| {
        FlashAttnError::Msg(format!(
            "ROCm {operation} rejected: planned reservation does not fit u64 (bytes={bytes})"
        ))
    })?;
    kiln_memory::MemoryGovernor::try_global_cached_reserve(bytes_u64).ok_or_else(|| {
        FlashAttnError::Msg(format!(
            "ROCm {operation} rejected: the published memory budget changed or is already reserved by another operation (requested_bytes={bytes})"
        ))
    })
}

fn materialized_score_budget_mib() -> usize {
    effective_score_geometry().materialized_budget_mib
}

fn materialized_score_budget_bytes() -> usize {
    materialized_score_budget_mib().saturating_mul(1024 * 1024)
}

fn materialized_score_tile_max_elements() -> usize {
    effective_score_geometry().materialized_tile_max_elements
}

fn materialized_score_working_set_bytes(b: usize, h: usize, sq: usize, sk: usize) -> Option<usize> {
    b.checked_mul(h)?
        .checked_mul(sq)?
        .checked_mul(sk)?
        .checked_mul(std::mem::size_of::<f32>())?
        .checked_mul(MATERIALIZED_SCORE_SCRATCH_BUFFERS)
}

fn reserve_materialized_score_scratch(
    device: KtDevice,
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    operation: &str,
) -> Result<kiln_memory::governor::Reservation<'static>, FlashAttnError> {
    require_published_rocm_available_bytes(device, operation)?;
    let scratch_bytes = materialized_score_working_set_bytes(b, h, sq, sk).ok_or_else(|| {
        FlashAttnError::Msg(format!(
            "ROCm {operation} rejected: materialized score scratch estimate overflow (batch={b}, heads={h}, sq={sq}, sk={sk})"
        ))
    })?;
    reserve_published_rocm_bytes(scratch_bytes, operation)
}

fn materialized_bwd_score_scratch_bytes(b: usize, h: usize, sq: usize, sk: usize) -> Option<usize> {
    b.checked_mul(h)?
        .checked_mul(sq)?
        .checked_mul(sk)?
        .checked_mul(std::mem::size_of::<f32>())?
        .checked_mul(MATERIALIZED_BWD_SCORE_SCRATCH_BUFFERS)
}

fn checked_tensor_elements(dimensions: &[usize]) -> Option<usize> {
    dimensions.iter().try_fold(1usize, |elements, dimension| {
        elements.checked_mul(*dimension)
    })
}

fn checked_weighted_bytes(terms: &[(usize, usize)]) -> Option<usize> {
    terms.iter().try_fold(0usize, |total, (elements, bytes)| {
        total.checked_add(elements.checked_mul(*bytes)?)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackwardMemoryPlan {
    fixed_peak_bytes: usize,
    scratch_peak_bytes: usize,
    peak_bytes: usize,
}

fn full_materialized_bwd_memory_plan(
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Option<BackwardMemoryPlan> {
    let bh = b.checked_mul(h)?;
    let q_elements = checked_tensor_elements(&[bh, sq, d])?;
    let k_elements = checked_tensor_elements(&[bh, sk, d])?;
    let row_elements = checked_tensor_elements(&[bh, sq])?;

    // This is deliberately an upper bound over operation-owned allocations,
    // rather than an allocator-dependent lifetime guess. Per Q element it
    // charges transformed Q/dO BF16 (4), their F32 copies (8), and the dQ
    // split-matmul accumulator/conversion/output ceiling (12). Per expanded K/V element it
    // charges GQA + transposed BF16 copies (10), F32 K/V + V-transpose copies
    // (12), and simultaneous dK/dV conversion/output storage (16). Row
    // reductions are separate because head_dim is not assumed to be non-zero.
    let fixed_peak_bytes = checked_weighted_bytes(&[
        (q_elements, 24),
        (k_elements, 38),
        (row_elements, std::mem::size_of::<f32>()),
    ])?;
    let scratch_peak_bytes = checked_tensor_elements(&[bh, sq, sk])?
        .checked_mul(std::mem::size_of::<f32>())?
        .checked_mul(MATERIALIZED_BWD_SCORE_SCRATCH_BUFFERS)?;
    let peak_bytes = fixed_peak_bytes.checked_add(scratch_peak_bytes)?;
    Some(BackwardMemoryPlan {
        fixed_peak_bytes,
        scratch_peak_bytes,
        peak_bytes,
    })
}

fn validate_full_materialized_bwd_admission(
    available_bytes: usize,
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Result<BackwardMemoryPlan, FlashAttnError> {
    let plan = full_materialized_bwd_memory_plan(b, h, sq, sk, d).ok_or_else(|| {
        FlashAttnError::Msg("ROCm full materialized backward memory estimate overflow".into())
    })?;
    if plan.peak_bytes > available_bytes {
        return Err(FlashAttnError::Msg(format!(
            "ROCm full materialized backward rejected: operation peak exceeds the published memory budget (available_bytes={available_bytes}, fixed_peak_bytes={}, simultaneous_score_scratch_bytes={}, peak_bytes={}, batch={b}, heads={h}, sq={sq}, sk={sk}, d={d})",
            plan.fixed_peak_bytes, plan.scratch_peak_bytes, plan.peak_bytes,
        )));
    }
    Ok(plan)
}

fn materialized_bwd_route_enabled(
    force: Option<bool>,
    heuristic_score_fit: bool,
    full_operation_admitted: bool,
) -> bool {
    match force {
        Some(false) => false,
        Some(true) => full_operation_admitted,
        None => heuristic_score_fit && full_operation_admitted,
    }
}

fn native_scalar_fwd_enabled(policy: RocmFlashAttentionPolicy, sq: usize, sk: usize) -> bool {
    policy.native_scalar_forward && sq.max(sk) <= policy.native_scalar_forward_max_sequence
}

fn native_fwd_query_tile_len(policy: RocmFlashAttentionPolicy) -> usize {
    policy.native_forward_query_tile
}

fn native_single_fwd_launch_enabled(
    policy: RocmFlashAttentionPolicy,
    sq: usize,
    sk: usize,
) -> bool {
    sq.max(sk) <= policy.native_single_forward_max_sequence
}

fn native_tiled_fwd_enabled(policy: RocmFlashAttentionPolicy, sq: usize, sk: usize) -> bool {
    policy.native_tiled_forward && sq.max(sk) > native_fwd_query_tile_len(policy)
}

fn native_streaming_fwd_key_tile_len(policy: RocmFlashAttentionPolicy) -> usize {
    policy.native_streaming_forward_key_tile
}

fn native_streaming_fwd_enabled(policy: RocmFlashAttentionPolicy, sq: usize, sk: usize) -> bool {
    policy.native_streaming_forward && sq.max(sk) >= policy.native_streaming_forward_min_sequence
}

fn native_rectangular_causal_fwd_enabled(policy: RocmFlashAttentionPolicy) -> bool {
    policy.native_rectangular_causal_forward
}

fn rocm_online_fwd_enabled(policy: RocmFlashAttentionPolicy) -> bool {
    policy.online_forward
}

fn rocm_online_bwd_enabled(policy: RocmFlashAttentionPolicy) -> bool {
    policy.online_backward
}

fn rocm_materialized_bwd_enabled(
    policy: RocmFlashAttentionPolicy,
    device: KtDevice,
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> bool {
    let heuristic_score_fit = materialized_bwd_score_scratch_bytes(b, h, sq, sk)
        .map(|bytes| bytes <= materialized_score_budget_bytes())
        .unwrap_or(false);
    let full_operation_admitted = published_rocm_available_bytes(device)
        .and_then(|available_bytes| {
            full_materialized_bwd_memory_plan(b, h, sq, sk, d)
                .map(|plan| plan.peak_bytes <= available_bytes)
        })
        .unwrap_or(false);
    materialized_bwd_route_enabled(
        match policy.materialized_backward_mode {
            RocmFlashAttentionRouteMode::Auto => None,
            RocmFlashAttentionRouteMode::Enabled => Some(true),
            RocmFlashAttentionRouteMode::Disabled => Some(false),
        },
        heuristic_score_fit,
        full_operation_admitted,
    )
}

fn rocm_native_bwd_preferred(
    policy: RocmFlashAttentionPolicy,
    _b: usize,
    _h: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> bool {
    match policy.native_backward_preference {
        RocmFlashAttentionRouteMode::Enabled => return true,
        RocmFlashAttentionRouteMode::Disabled => return false,
        RocmFlashAttentionRouteMode::Auto => {}
    }
    if policy.materialized_backward_mode == RocmFlashAttentionRouteMode::Enabled {
        return false;
    }
    let max_sequence = match d {
        128 => policy.native_backward_d128_max_sequence,
        256 => policy.native_backward_d256_max_sequence,
        _ => return false,
    };
    if sq.max(sk) <= max_sequence {
        return true;
    }
    sq.max(sk) >= policy.native_backward_long_min_sequence
}

fn rocm_collapsed_gqa_bwd_enabled(policy: RocmFlashAttentionPolicy) -> bool {
    policy.collapsed_gqa_backward
}

fn rocm_native_direct_collapsed_gqa_bwd_enabled(policy: RocmFlashAttentionPolicy) -> bool {
    policy.native_direct_collapsed_gqa_backward
}

#[cfg(test)]
fn query_tile_len_for_budget(
    b: usize,
    h: usize,
    key_len: usize,
    remaining: usize,
    budget_bytes: usize,
) -> usize {
    query_tile_len_for_budget_with_scratch(
        b,
        h,
        key_len,
        remaining,
        budget_bytes,
        MATERIALIZED_SCORE_SCRATCH_BUFFERS,
    )
}

fn query_tile_len_for_budget_with_scratch(
    b: usize,
    h: usize,
    key_len: usize,
    remaining: usize,
    budget_bytes: usize,
    scratch_buffers: usize,
) -> usize {
    if key_len == 0 || b == 0 || h == 0 {
        return 1;
    }
    let denom = b
        .saturating_mul(h)
        .saturating_mul(key_len)
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_mul(scratch_buffers);
    if denom == 0 {
        return remaining.max(1);
    }
    let budget_limited = (budget_bytes / denom).max(1);
    let score_element_denom = b.saturating_mul(h).saturating_mul(key_len);
    let element_limited = if score_element_denom == 0 {
        remaining.max(1)
    } else {
        (materialized_score_tile_max_elements() / score_element_denom).max(1)
    };
    let raw = budget_limited.min(element_limited).max(1);
    let tile = if raw >= MATERIALIZED_SCORE_TILE_GRANULARITY {
        (raw / MATERIALIZED_SCORE_TILE_GRANULARITY) * MATERIALIZED_SCORE_TILE_GRANULARITY
    } else {
        raw
    };
    remaining.min(tile.max(1)).max(1)
}

fn validate_materialized_query_tile_plan(
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    budget_bytes: usize,
) -> Result<usize, FlashAttnError> {
    validate_materialized_query_tile_plan_with_scratch(
        b,
        h,
        sq,
        sk,
        budget_bytes,
        MATERIALIZED_SCORE_SCRATCH_BUFFERS,
    )
}

fn validate_materialized_query_tile_plan_with_scratch(
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    budget_bytes: usize,
    scratch_buffers: usize,
) -> Result<usize, FlashAttnError> {
    if sq == 0 || sk == 0 {
        return Ok(1);
    }
    let minimum_query_tile = sq.min(MATERIALIZED_SCORE_TILE_GRANULARITY);
    let minimum_scratch_bytes = b
        .checked_mul(h)
        .and_then(|value| value.checked_mul(minimum_query_tile))
        .and_then(|value| value.checked_mul(sk))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|value| value.checked_mul(scratch_buffers))
        .unwrap_or(usize::MAX);
    let planned_tile =
        query_tile_len_for_budget_with_scratch(b, h, sk, sq, budget_bytes, scratch_buffers);
    let tile_count = sq.div_ceil(planned_tile);
    if budget_bytes < minimum_scratch_bytes
        || planned_tile < minimum_query_tile
        || tile_count > MAX_MATERIALIZED_QUERY_TILES
    {
        return Err(FlashAttnError::Msg(format!(
            "ROCm materialized query-tiled attention rejected: the published memory budget cannot sustain a bounded plan (budget_bytes={budget_bytes}, minimum_scratch_bytes={minimum_scratch_bytes}, scratch_buffers={scratch_buffers}, planned_query_tile={planned_tile}, minimum_query_tile={minimum_query_tile}, tile_count={tile_count}, max_tile_count={MAX_MATERIALIZED_QUERY_TILES}, batch={b}, heads={h}, sq={sq}, sk={sk})"
        )));
    }
    Ok(planned_tile)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaterializedBwdQueryTilePlan {
    query_tile: usize,
    fixed_peak_bytes: usize,
    tile_scratch_bytes: usize,
}

impl MaterializedBwdQueryTilePlan {
    fn peak_bytes(self) -> Option<usize> {
        self.fixed_peak_bytes.checked_add(self.tile_scratch_bytes)
    }
}

fn materialized_bwd_query_tiled_fixed_peak_bytes(
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Option<usize> {
    let bh = b.checked_mul(h)?;
    let q_elements = checked_tensor_elements(&[bh, sq, d])?;
    let k_elements = checked_tensor_elements(&[bh, sk, d])?;
    let row_elements = checked_tensor_elements(&[bh, sq])?;

    // Q includes retained transformed q/dO and the worst accumulated dQ/cat
    // output phase. K includes expanded/transposed K/V, F32 K/V views, both
    // full F32 gradient accumulators, and the worst accumulator or final-output
    // conversion phase. Tile-local Q/K and score tensors are charged separately.
    checked_weighted_bytes(&[
        (q_elements, 8),
        (k_elements, 38),
        (row_elements, std::mem::size_of::<f32>()),
    ])
}

fn materialized_bwd_query_tile_scratch_bytes(
    bh: usize,
    query_tile: usize,
    sk: usize,
    d: usize,
) -> Option<usize> {
    let q_elements = checked_tensor_elements(&[bh, query_tile, d])?;
    let row_elements = checked_tensor_elements(&[bh, query_tile])?;
    let score_elements = checked_tensor_elements(&[bh, query_tile, sk])?;
    checked_weighted_bytes(&[
        // q/dO tile conversion, dQ matmul/scaling, and BF16 output conversion.
        (q_elements, 24),
        (row_elements, 8),
        (
            score_elements,
            std::mem::size_of::<f32>() * MATERIALIZED_BWD_SCORE_SCRATCH_BUFFERS,
        ),
    ])
}

#[allow(clippy::too_many_arguments)]
fn plan_materialized_bwd_query_tiles_with_limit(
    available_bytes: usize,
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
    score_element_limit: usize,
) -> Result<MaterializedBwdQueryTilePlan, FlashAttnError> {
    let bh = b
        .checked_mul(h)
        .ok_or_else(|| FlashAttnError::Msg("ROCm tiled backward batch/head overflow".into()))?;
    let fixed_peak_bytes = materialized_bwd_query_tiled_fixed_peak_bytes(b, h, sq, sk, d)
        .ok_or_else(|| {
            FlashAttnError::Msg("ROCm tiled backward fixed memory estimate overflow".into())
        })?;
    let residual_bytes = available_bytes.checked_sub(fixed_peak_bytes).ok_or_else(|| {
        FlashAttnError::Msg(format!(
            "ROCm materialized query-tiled backward rejected: fixed and accumulator/output peaks exceed the published memory budget (available_bytes={available_bytes}, fixed_peak_bytes={fixed_peak_bytes}, batch={b}, heads={h}, sq={sq}, sk={sk}, d={d})"
        ))
    })?;

    if sq == 0 || sk == 0 {
        return Ok(MaterializedBwdQueryTilePlan {
            query_tile: 1,
            fixed_peak_bytes,
            tile_scratch_bytes: 0,
        });
    }

    let bytes_per_query = materialized_bwd_query_tile_scratch_bytes(bh, 1, sk, d)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            FlashAttnError::Msg("ROCm tiled backward tile memory estimate overflow".into())
        })?;
    let budget_limited = residual_bytes / bytes_per_query;
    let score_elements_per_query = bh.checked_mul(sk).ok_or_else(|| {
        FlashAttnError::Msg("ROCm tiled backward score element estimate overflow".into())
    })?;
    let element_limited = score_element_limit / score_elements_per_query;
    let raw_tile = sq.min(budget_limited).min(element_limited);
    let planned_tile = if raw_tile >= MATERIALIZED_SCORE_TILE_GRANULARITY {
        (raw_tile / MATERIALIZED_SCORE_TILE_GRANULARITY) * MATERIALIZED_SCORE_TILE_GRANULARITY
    } else {
        raw_tile
    };
    let minimum_query_tile = sq.min(MATERIALIZED_SCORE_TILE_GRANULARITY);
    let tile_count = if planned_tile == 0 {
        usize::MAX
    } else {
        sq.div_ceil(planned_tile)
    };
    if planned_tile < minimum_query_tile || tile_count > MAX_MATERIALIZED_QUERY_TILES {
        let minimum_scratch_bytes =
            materialized_bwd_query_tile_scratch_bytes(bh, minimum_query_tile, sk, d)
                .unwrap_or(usize::MAX);
        return Err(FlashAttnError::Msg(format!(
            "ROCm materialized query-tiled backward rejected: the residual published memory budget cannot sustain a bounded plan (available_bytes={available_bytes}, fixed_peak_bytes={fixed_peak_bytes}, residual_bytes={residual_bytes}, minimum_tile_scratch_bytes={minimum_scratch_bytes}, planned_query_tile={planned_tile}, minimum_query_tile={minimum_query_tile}, tile_count={tile_count}, max_tile_count={MAX_MATERIALIZED_QUERY_TILES}, batch={b}, heads={h}, sq={sq}, sk={sk}, d={d})"
        )));
    }

    let tile_scratch_bytes = materialized_bwd_query_tile_scratch_bytes(bh, planned_tile, sk, d)
        .ok_or_else(|| {
            FlashAttnError::Msg("ROCm tiled backward tile memory estimate overflow".into())
        })?;
    let peak_bytes = fixed_peak_bytes
        .checked_add(tile_scratch_bytes)
        .ok_or_else(|| FlashAttnError::Msg("ROCm tiled backward peak estimate overflow".into()))?;
    if peak_bytes > available_bytes {
        return Err(FlashAttnError::Msg(format!(
            "ROCm materialized query-tiled backward rejected: planned operation peak exceeds the published memory budget (available_bytes={available_bytes}, fixed_peak_bytes={fixed_peak_bytes}, tile_scratch_bytes={tile_scratch_bytes}, peak_bytes={peak_bytes})"
        )));
    }
    Ok(MaterializedBwdQueryTilePlan {
        query_tile: planned_tile,
        fixed_peak_bytes,
        tile_scratch_bytes,
    })
}

fn plan_materialized_bwd_query_tiles(
    available_bytes: usize,
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Result<MaterializedBwdQueryTilePlan, FlashAttnError> {
    plan_materialized_bwd_query_tiles_with_limit(
        available_bytes,
        b,
        h,
        sq,
        sk,
        d,
        materialized_score_tile_max_elements(),
    )
}

#[derive(Debug, Clone, Copy)]
struct OnlineTileSizing {
    query_tile: usize,
    key_tile: usize,
    score_budget_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct OnlineForwardPlan {
    sizing: OnlineTileSizing,
    fixed_working_set_bytes: usize,
}

impl OnlineForwardPlan {
    fn for_operation(
        device: KtDevice,
        bh: usize,
        sq: usize,
        sk: usize,
        d: usize,
        fixed_working_set_bytes: usize,
    ) -> Result<Self, FlashAttnError> {
        Ok(Self {
            sizing: OnlineTileSizing::for_operation(
                device,
                bh,
                sq,
                sk,
                d,
                fixed_working_set_bytes,
                OnlinePass::Forward,
            )?,
            fixed_working_set_bytes,
        })
    }

    fn operation_peak_bytes(self, bh: usize, sq: usize, sk: usize, d: usize) -> Option<usize> {
        online_forward_operation_peak_bytes(
            self.sizing,
            bh,
            sq,
            sk,
            d,
            self.fixed_working_set_bytes,
        )
    }
}

struct PreReservedOnlineForwardPlan<'a> {
    plan: OnlineForwardPlan,
    _reservation: &'a kiln_memory::governor::Reservation<'static>,
}

#[derive(Debug, Clone, Copy)]
enum OnlinePass {
    Forward,
    Backward,
}

impl OnlineTileSizing {
    fn for_operation(
        device: KtDevice,
        bh: usize,
        sq: usize,
        sk: usize,
        d: usize,
        fixed_working_set_bytes: usize,
        pass: OnlinePass,
    ) -> Result<Self, FlashAttnError> {
        let available_bytes = published_rocm_available_bytes(device).ok_or_else(|| {
            FlashAttnError::Msg(
                "ROCm online attention rejected: no initialized memory governor snapshot matches the active ROCm device"
                    .to_owned(),
            )
        })?;
        let score_available_bytes = available_bytes.checked_sub(fixed_working_set_bytes).ok_or_else(
            || {
                FlashAttnError::Msg(format!(
                    "ROCm online attention rejected: fixed working set exceeds the published memory budget (available_bytes={available_bytes}, fixed_working_set_bytes={fixed_working_set_bytes}, bh={bh}, sq={sq}, sk={sk})"
                ))
            },
        )?;
        let geometry = effective_score_geometry();
        let sizing = Self {
            query_tile: geometry.rocm_online_query_tile,
            key_tile: geometry.rocm_online_key_tile,
            score_budget_bytes: geometry.rocm_online_budget_mib.saturating_mul(1024 * 1024),
        };
        validate_online_operation_plan(sizing, bh, sq, sk, d, score_available_bytes, pass)?;
        Ok(sizing)
    }
}

fn checked_sequence_working_set_bytes(
    bh: usize,
    seq: usize,
    d: usize,
    bytes_per_element: usize,
) -> Option<usize> {
    bh.checked_mul(seq)?
        .checked_mul(d)?
        .checked_mul(bytes_per_element)
}

fn online_forward_fixed_working_set_bytes(
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Option<usize> {
    // Q contiguity + retained output/LSE, plus expanded/contiguous K/V and the
    // F32 V copy. Inputs themselves are already resident and are not charged.
    checked_sequence_working_set_bytes(bh, sq, d, 8)?
        .checked_add(checked_sequence_working_set_bytes(bh, sk, d, 12)?)
}

fn online_backward_fixed_working_set_bytes(
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Option<usize> {
    let q_elements = checked_tensor_elements(&[bh, sq, d])?;
    let k_elements = checked_tensor_elements(&[bh, sk, d])?;
    let row_elements = checked_tensor_elements(&[bh, sq])?;

    // Preparation/loop peak: transformed q/dO/out BF16 (6Q), retained dQ BF16
    // tiles (2Q), expanded + transformed K/V BF16 (8K), full dK/dV F32
    // accumulators (8K), and the saved contiguous LSE (4R). The final dK/dV
    // conversion phase also tops out at 16K, so this bound covers both phases.
    checked_weighted_bytes(&[
        (q_elements, 8),
        (k_elements, 16),
        (row_elements, std::mem::size_of::<f32>()),
    ])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OnlineBackwardTilePhasePeaks {
    q_preparation_bytes: usize,
    score_bytes: usize,
    dv_bytes: usize,
    dq_bytes: usize,
    dk_bytes: usize,
}

impl OnlineBackwardTilePhasePeaks {
    fn peak_bytes(self) -> usize {
        [
            self.q_preparation_bytes,
            self.score_bytes,
            self.dv_bytes,
            self.dq_bytes,
            self.dk_bytes,
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
}

fn online_backward_tile_phase_peaks(
    bh: usize,
    q_tile: usize,
    k_tile: usize,
    d: usize,
) -> Option<OnlineBackwardTilePhasePeaks> {
    let q_elements = checked_tensor_elements(&[bh, q_tile, d])?;
    let k_elements = checked_tensor_elements(&[bh, k_tile, d])?;
    let row_elements = checked_tensor_elements(&[bh, q_tile])?;
    let score_elements = checked_tensor_elements(&[bh, q_tile, k_tile])?;

    // Each phase is incremental to `online_backward_fixed_working_set_bytes`.
    // Score-family accounting includes simultaneous F32 probabilities + dP/dS
    // and the BF16 probability/gradient copy (10 bytes per score element at the
    // widest phase). K accounting includes both BF16 blocks and the transient
    // F32 dK/dV matmul output at the phase where it is live.
    let q_preparation_bytes = checked_weighted_bytes(&[(q_elements, 20), (row_elements, 4)])?;
    let score_bytes = checked_weighted_bytes(&[
        (q_elements, 8),
        (row_elements, 4),
        (k_elements, 4),
        (score_elements, 10),
    ])?;
    let dv_bytes = checked_weighted_bytes(&[
        (q_elements, 8),
        (row_elements, 4),
        (k_elements, 8),
        (score_elements, 6),
    ])?;
    let dq_bytes = checked_weighted_bytes(&[
        (q_elements, 12),
        (row_elements, 4),
        (k_elements, 4),
        (score_elements, 6),
    ])?;
    let dk_bytes = checked_weighted_bytes(&[
        (q_elements, 8),
        (row_elements, 4),
        (k_elements, 6),
        (score_elements, 6),
    ])?;
    Some(OnlineBackwardTilePhasePeaks {
        q_preparation_bytes,
        score_bytes,
        dv_bytes,
        dq_bytes,
        dk_bytes,
    })
}

fn head_major_conversion_working_set_bytes(
    b: usize,
    h: usize,
    hk: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Option<usize> {
    let q_heads = b.checked_mul(h)?;
    let kv_heads = b.checked_mul(hk)?;
    checked_sequence_working_set_bytes(q_heads, sq, d, 2)?
        .checked_add(checked_sequence_working_set_bytes(kv_heads, sk, d, 4)?)
}

fn head_major_online_forward_fixed_working_set_bytes(
    b: usize,
    h: usize,
    hk: usize,
    sq: usize,
    sk: usize,
    d: usize,
) -> Option<usize> {
    let bh = b.checked_mul(h)?;
    online_forward_fixed_working_set_bytes(bh, sq, sk, d)?.checked_add(
        head_major_conversion_working_set_bytes(b, h, hk, sq, sk, d)?,
    )
}

fn validate_online_tile_plan(
    sizing: OnlineTileSizing,
    bh: usize,
    sq: usize,
    sk: usize,
) -> Result<(), FlashAttnError> {
    if bh == 0 || sq == 0 || sk == 0 {
        return Ok(());
    }
    let (q_tile, k_tile) = online_tile_lens(sizing, bh, sq, sk);
    let required_bytes = online_tile_bytes(bh, q_tile, k_tile).unwrap_or(usize::MAX);
    let tile_pairs = sq.div_ceil(q_tile).saturating_mul(sk.div_ceil(k_tile));
    if required_bytes > sizing.score_budget_bytes || tile_pairs > MAX_ONLINE_TILE_PAIRS {
        return Err(FlashAttnError::Msg(format!(
            "ROCm online attention rejected: the published memory budget cannot sustain a bounded tile plan (budget_bytes={}, required_tile_bytes={}, tile_pairs={}, max_tile_pairs={}, bh={}, sq={}, sk={}). Wait for memory pressure to ease or reduce the sequence/batch size.",
            sizing.score_budget_bytes,
            required_bytes,
            tile_pairs,
            MAX_ONLINE_TILE_PAIRS,
            bh,
            sq,
            sk,
        )));
    }
    Ok(())
}

fn online_forward_tile_peak_bytes(
    bh: usize,
    q_tile: usize,
    k_tile: usize,
    d: usize,
) -> Option<(usize, usize, usize)> {
    let q_elements = checked_tensor_elements(&[bh, q_tile, d])?;
    let k_elements = checked_tensor_elements(&[bh, k_tile, d])?;
    let row_elements = checked_tensor_elements(&[bh, q_tile])?;
    let score_bytes = online_tile_bytes(bh, q_tile, k_tile)?;
    let non_score_bytes =
        checked_weighted_bytes(&[(q_elements, 12), (k_elements, 6), (row_elements, 24)])?;
    Some((
        score_bytes,
        non_score_bytes,
        score_bytes.checked_add(non_score_bytes)?,
    ))
}

fn online_forward_operation_peak_bytes(
    sizing: OnlineTileSizing,
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
    fixed_working_set_bytes: usize,
) -> Option<usize> {
    let (q_tile, k_tile) = online_tile_lens(sizing, bh, sq, sk);
    let (_, _, tile_peak_bytes) = online_forward_tile_peak_bytes(bh, q_tile, k_tile, d)?;
    fixed_working_set_bytes.checked_add(tile_peak_bytes)
}

fn validate_online_operation_plan(
    sizing: OnlineTileSizing,
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
    scratch_available_bytes: usize,
    pass: OnlinePass,
) -> Result<(), FlashAttnError> {
    validate_online_tile_plan(sizing, bh, sq, sk)?;
    let (q_tile, k_tile) = online_tile_lens(sizing, bh, sq, sk);
    let score_bytes = online_tile_bytes(bh, q_tile, k_tile).unwrap_or(usize::MAX);
    let (non_score_bytes, score_family_bytes, peak_tile_bytes) = match pass {
        OnlinePass::Forward => {
            let (score_bytes, non_score_bytes, peak_tile_bytes) =
                online_forward_tile_peak_bytes(bh, q_tile, k_tile, d).ok_or_else(|| {
                    FlashAttnError::Msg("ROCm online forward tile size overflow".into())
                })?;
            (non_score_bytes, score_bytes, peak_tile_bytes)
        }
        OnlinePass::Backward => {
            let phases =
                online_backward_tile_phase_peaks(bh, q_tile, k_tile, d).ok_or_else(|| {
                    FlashAttnError::Msg("ROCm online backward tile phase estimate overflow".into())
                })?;
            let score_family_bytes = checked_tensor_elements(&[bh, q_tile, k_tile])
                .and_then(|elements| elements.checked_mul(10))
                .ok_or_else(|| {
                    FlashAttnError::Msg(
                        "ROCm online backward score-family estimate overflow".into(),
                    )
                })?;
            (
                phases.peak_bytes().saturating_sub(score_family_bytes),
                score_family_bytes,
                phases.peak_bytes(),
            )
        }
    };
    if peak_tile_bytes > scratch_available_bytes {
        let message = match pass {
            OnlinePass::Forward => format!(
                "ROCm online attention rejected: score and non-score tile scratch exceed the published residual budget (scratch_available_bytes={scratch_available_bytes}, score_bytes={score_bytes}, non_score_bytes={non_score_bytes}, peak_tile_bytes={peak_tile_bytes}, pass={pass:?}, bh={bh}, sq={sq}, sk={sk}, d={d})"
            ),
            OnlinePass::Backward => format!(
                "ROCm online attention rejected: score-family and non-score tile scratch exceed the published residual budget (scratch_available_bytes={scratch_available_bytes}, score_tile_bytes={score_bytes}, score_family_bytes={score_family_bytes}, non_score_peak_bytes={non_score_bytes}, peak_tile_bytes={peak_tile_bytes}, pass={pass:?}, bh={bh}, sq={sq}, sk={sk}, d={d})"
            ),
        };
        return Err(FlashAttnError::Msg(message));
    }
    Ok(())
}

fn online_tile_bytes(bh: usize, q_tile: usize, k_tile: usize) -> Option<usize> {
    bh.checked_mul(q_tile)?
        .checked_mul(k_tile)?
        .checked_mul(std::mem::size_of::<f32>())
}

fn online_tile_lens(
    sizing: OnlineTileSizing,
    bh: usize,
    remaining_q: usize,
    remaining_k: usize,
) -> (usize, usize) {
    let mut q_tile = sizing.query_tile.min(remaining_q).max(1);
    let mut k_tile = sizing.key_tile.min(remaining_k).max(1);
    let budget = sizing.score_budget_bytes.max(1);

    while online_tile_bytes(bh, q_tile, k_tile)
        .map(|bytes| bytes > budget)
        .unwrap_or(true)
    {
        if k_tile > ONLINE_TILE_GRANULARITY {
            k_tile = (k_tile / 2).max(ONLINE_TILE_GRANULARITY).min(remaining_k);
        } else if q_tile > ONLINE_TILE_GRANULARITY {
            q_tile = (q_tile / 2).max(ONLINE_TILE_GRANULARITY).min(remaining_q);
        } else {
            break;
        }
    }

    (q_tile.max(1), k_tile.max(1))
}

fn tile_len_without_unit_tail(remaining: usize, tile_cap: usize) -> usize {
    let tile_len = remaining.min(tile_cap.max(1)).max(1);
    if tile_len > 1 && remaining.saturating_sub(tile_len) == 1 {
        tile_len - 1
    } else {
        tile_len
    }
}

fn causal_block_limit(sk: usize, q_start: usize, q_len: usize, causal_offset: isize) -> usize {
    let q_end = q_start.saturating_add(q_len);
    let limit = causal_offset.saturating_add(q_end as isize);
    if limit <= 0 {
        0
    } else {
        (limit as usize).min(sk)
    }
}

fn usize_to_i32(name: &str, value: usize) -> Result<i32, FlashAttnError> {
    i32::try_from(value).map_err(|_| FlashAttnError::Msg(format!("{name} too large: {value}")))
}

fn isize_to_i32(name: &str, value: isize) -> Result<i32, FlashAttnError> {
    i32::try_from(value).map_err(|_| FlashAttnError::Msg(format!("{name} too large: {value}")))
}

#[allow(clippy::too_many_arguments)]
fn sdpa_forward_online_tiled(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
    pre_reserved_plan: Option<PreReservedOnlineForwardPlan<'_>>,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    if sq == 0 || sk == 0 {
        return sdpa_forward(q, k, v, b, sq, sk, h, hk, d, scale, causal);
    }

    let device = q.device();
    let bh = b
        .checked_mul(h)
        .ok_or_else(|| FlashAttnError::Msg("ROCm online attention batch/head overflow".into()))?;
    let causal_offset = sk as isize - sq as isize;
    let online_plan = match pre_reserved_plan.as_ref() {
        Some(pre_reserved) => pre_reserved.plan,
        None => {
            let fixed_working_set_bytes = online_forward_fixed_working_set_bytes(bh, sq, sk, d)
                .ok_or_else(|| {
                    FlashAttnError::Msg("ROCm online attention working-set overflow".into())
                })?;
            OnlineForwardPlan::for_operation(device, bh, sq, sk, d, fixed_working_set_bytes)?
        }
    };
    let online_tile_sizing = online_plan.sizing;
    let operation_peak_bytes = online_plan
        .operation_peak_bytes(bh, sq, sk, d)
        .ok_or_else(|| FlashAttnError::Msg("ROCm online forward peak estimate overflow".into()))?;
    let _owned_memory_reservation = if pre_reserved_plan.is_none() {
        Some(reserve_published_rocm_bytes(
            operation_peak_bytes,
            "online forward",
        )?)
    } else {
        None
    };
    let k_exp = gqa_expand_heads(k, h, hk, device)?;
    let v_exp = gqa_expand_heads(v, h, hk, device)?;

    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?;
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?;

    let q3 = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?;
    let k3 = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?;
    let v3 = map_kt(v_bhsd.reshape(vec![bh, sk, d]))?;
    let v3f = rocm_cast_to(&v3, KtDType::F32)?;

    let mut out_tiles = Vec::new();
    let mut lse_tiles = Vec::new();
    let mut q_start = 0usize;
    while q_start < sq {
        let remaining_q = sq - q_start;
        let max_k = if causal {
            causal_block_limit(sk, q_start, remaining_q, causal_offset).max(1)
        } else {
            sk
        };
        let (q_tile_cap, key_tile_cap) =
            online_tile_lens(online_tile_sizing, bh, remaining_q, max_k);
        let q_len = remaining_q.min(q_tile_cap).max(1);
        let rows = bh * q_len;

        let q_tile = rocm_contig(&map_kt(q3.narrow(1, q_start, q_len))?)?;
        rocm_sync_tensor(&q_tile)?;
        let row_m_zero = rocm_zeros_in_tensor_context(&q3, KtDType::F32, vec![rows])?;
        let row_m = map_kt(kiln_tensor::ops::full_like(&row_m_zero, f32::NEG_INFINITY))?;
        let row_l = rocm_zeros_in_tensor_context(&q3, KtDType::F32, vec![rows])?;
        let alpha = rocm_zeros_in_tensor_context(&q3, KtDType::F32, vec![rows])?;
        let beta = rocm_zeros_in_tensor_context(&q3, KtDType::F32, vec![rows])?;
        let acc = rocm_zeros_in_tensor_context(&q3, KtDType::F32, vec![bh, q_len, d])?;

        let block_limit = if causal {
            causal_block_limit(sk, q_start, q_len, causal_offset)
        } else {
            sk
        };
        let mut k_start = 0usize;
        while k_start < block_limit {
            let k_len = tile_len_without_unit_tail(block_limit - k_start, key_tile_cap);
            let k_block = rocm_contig(&map_kt(k3.narrow(1, k_start, k_len))?)?;
            let v_block = rocm_contig(&map_kt(v3f.narrow(1, k_start, k_len))?)?;
            rocm_sync_tensor(&v_block)?;

            let scores = rocm_matmul_rhs_transposed_to_dtype_batch_grouped(
                &q_tile,
                &k_block,
                KtDType::F32,
                &format!("online qk q_start={q_start} k_start={k_start}"),
            )?;
            rocm_sync_tensor(&scores)?;
            let scores = scale_mask_scores_f32(
                scores,
                bh,
                q_len,
                k_len,
                q_start,
                k_start,
                causal_offset,
                scale,
                causal,
            )?;
            rocm_sync_tensor(&scores)?;
            let block_m = map_kt(kiln_tensor::rocm_max_axis(&scores, 2))?;
            rocm_sync_tensor(&block_m)?;
            let probs = exp_mask_scores_f32(
                scores,
                &block_m,
                bh,
                q_len,
                k_len,
                q_start,
                k_start,
                causal_offset,
                causal,
            )?;
            rocm_sync_tensor(&probs)?;
            let block_l = map_kt(kiln_tensor::rocm_sum_axis(&probs, 2))?;
            let block_out = rocm_matmul_split_inner_f32(
                &probs,
                &v_block,
                &format!("online block_out q_start={q_start} k_start={k_start}"),
            )?;
            rocm_sync_tensor(&block_out)?;
            let block_m_flat = map_kt(block_m.reshape(vec![rows]))?;
            let block_l_flat = map_kt(block_l.reshape(vec![rows]))?;

            // The online path mixes kt matmul/reduce kernels with custom HIP
            // state kernels on the tensor-owned stream.
            // Order each producer/consumer handoff explicitly; otherwise a
            // later stream can read block_l/block_out or alpha/beta before the
            // producing stream has finished, which shows up as intermittent
            // NaNs on long ROCm rows.
            rocm_sync_tensor(&block_l_flat)?;
            online_update_state_f32(
                &row_m,
                &row_l,
                &block_m_flat,
                &block_l_flat,
                &alpha,
                &beta,
                rows,
            )?;
            rocm_sync_tensor(&alpha)?;
            online_update_acc_f32(&acc, &block_out, &alpha, &beta, rows, d)?;
            rocm_sync_tensor(&acc)?;

            k_start += k_len;
        }

        let out3 = rocm_zeros_in_tensor_context(&q3, KtDType::F32, vec![bh, q_len, d])?;
        let lse2 = rocm_zeros_in_tensor_context(&q3, KtDType::F32, vec![rows])?;
        online_finalize_f32(&acc, &row_m, &row_l, &out3, &lse2, rows, d)?;
        // Finalize writes out3/lse2 from the custom HIP stream; the following
        // transpose/cast/cat work can launch from tensor-owned kt streams.
        // Keep the online state tensors alive and ordered until final output
        // materialization has consumed them.
        rocm_sync_tensor(&out3)?;
        let out_tile = bhsd3_to_bshd_bf16(&out3, b, h, q_len, d)?;
        rocm_sync_tensor(&out_tile)?;
        out_tiles.push(out_tile);
        lse_tiles.push(map_kt(lse2.reshape(vec![b, h, q_len]))?);

        q_start += q_len;
    }

    let out_refs: Vec<&KtTensor> = out_tiles.iter().collect();
    let lse_refs: Vec<&KtTensor> = lse_tiles.iter().collect();
    let out = map_kt(KtTensor::cat(&out_refs, 1))?;
    let lse = map_kt(KtTensor::cat(&lse_refs, 2))?;
    rocm_sync_tensor(&lse)?;
    Ok((out, lse))
}

#[allow(clippy::too_many_arguments)]
fn sdpa_forward_query_tiled(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
    budget_bytes: usize,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    if sq == 0 || sk == 0 {
        return sdpa_forward(q, k, v, b, sq, sk, h, hk, d, scale, causal);
    }
    if causal && sk < sq {
        return sdpa_forward(q, k, v, b, sq, sk, h, hk, d, scale, causal);
    }
    let planned_tile = validate_materialized_query_tile_plan(b, h, sq, sk, budget_bytes)?;

    let causal_offset = sk.saturating_sub(sq);
    let mut out_tiles = Vec::new();
    let mut lse_tiles = Vec::new();
    let mut tile_start = 0usize;
    while tile_start < sq {
        let remaining = sq - tile_start;
        let tile_len = remaining.min(planned_tile).max(1);
        let tile_end = tile_start + tile_len;
        let key_len = if causal { causal_offset + tile_end } else { sk };

        let q_tile = map_kt(q.narrow(1, tile_start, tile_len))?;
        let k_tile = if key_len == sk {
            k.clone()
        } else {
            map_kt(k.narrow(1, 0, key_len))?
        };
        let v_tile = if key_len == sk {
            v.clone()
        } else {
            map_kt(v.narrow(1, 0, key_len))?
        };
        let (out_tile, lse_tile) = sdpa_forward(
            &q_tile, &k_tile, &v_tile, b, tile_len, key_len, h, hk, d, scale, causal,
        )?;
        out_tiles.push(out_tile);
        lse_tiles.push(lse_tile);
        tile_start = tile_end;
    }

    let out_refs: Vec<&KtTensor> = out_tiles.iter().collect();
    let lse_refs: Vec<&KtTensor> = lse_tiles.iter().collect();
    let out = map_kt(KtTensor::cat(&out_refs, 1))?;
    let lse = map_kt(KtTensor::cat(&lse_refs, 2))?;
    Ok((out, lse))
}

/// GQA expand: repeat each kv-head group `group = h/hk` times along the head
/// axis (axis 2 of `[b, sk, hk, d]`) to produce `[b, sk, h, d]`. Implemented as
/// an on-device `index_select` over the head axis with indices
/// `[0,0,..,1,1,..]` (each kv head repeated `group` times). When `hk == h`
/// (no GQA) this is a contiguous pass-through.
fn gqa_expand_heads(
    kv: &KtTensor, // [b, sk, hk, d], contiguous
    h: usize,
    hk: usize,
    device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    if hk == h {
        return rocm_contig(kv);
    }
    let _ = device;
    map_kt(kiln_tensor::rocm_gqa_repeat_heads(kv, h))
}

/// Head-major GQA expand: `[b, hk, sk, d] -> [b, h, sk, d]`.
fn gqa_expand_heads_head_major(
    kv: &KtTensor,
    h: usize,
    hk: usize,
    device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    if hk == h {
        return rocm_contig(kv);
    }
    let _ = device;
    map_kt(kiln_tensor::rocm_gqa_repeat_heads_head_major(kv, h))
}

/// On-device log-sum-exp from already-computed `scores` and softmax `p`.
///
/// `lse[r] = max_j scores[r,j] + ln(sum_j exp(scores[r,j] - max))`.
/// Using the identity `sum exp(scores - max) = 1 / max_j p[r,j]` (the softmax
/// value at the arg-max column equals `1/Z`), this becomes
/// `lse[r] = max_scores[r] - ln(max_p[r])`. Both reductions run on-device; only
/// the final `[b*h, sq]` element-wise `sub`/`ln` is host-staged (ROCm has no
/// native `ln`), which is tiny relative to the `[b*h, sq, sk]` matmuls.
///
/// Returns an F32 tensor shaped `[b, h, sq]` on the originating ROCm device.
fn compute_lse(
    scores: &KtTensor, // [b*h, sq, sk] f32, contiguous
    p: &KtTensor,      // [b*h, sq, sk] f32, contiguous (softmax of scores)
    b: usize,
    h: usize,
    sq: usize,
) -> Result<KtTensor, FlashAttnError> {
    // Reductions over the last (sk) axis -> [b*h, sq], on-device.
    let max_scores = map_kt(kiln_tensor::rocm_max_axis(scores, 2))?; // [b*h, sq]
    let max_p = map_kt(kiln_tensor::rocm_max_axis(p, 2))?; // [b*h, sq]

    // On-device tail: lse = max_scores - ln(max_p). `.log()` -> `ops::ln`
    // (UnaryArithKind::Ln, tag 5) routes to `rocm_activation_unary`; `.sub()`
    // -> `ElementwiseOp::rocm_fwd` (both [b*h, sq], contiguous). Previously this
    // D2H'd both reductions, ran the ln/sub on the host, and H2D'd the result —
    // 3 host syncs per full-attention layer per forward. Now zero.
    //
    // Parity note: a fully-masked row (max_p == 0) yields +inf here vs the old
    // host path's NEG_INFINITY. Such rows do not arise in causal decode/training
    // (every query attends to >=1 key), and the ROCm inference path discards lse
    // (`_lse`) — so the edge is unreachable in practice.
    let ln_max_p = map_kt(max_p.log())?; // [b*h, sq]
    let lse_flat = map_kt(max_scores.sub(&ln_max_p))?; // [b*h, sq]
    let lse = map_kt(lse_flat.reshape(vec![b, h, sq]))?;
    rocm_sync_tensor(&lse)?;
    Ok(lse)
}

/// Core on-device SDPA composite.
///
/// Inputs (all `Device::Rocm`, BF16):
/// - `q`: `[b, sq, h, d]`
/// - `k`, `v`: `[b, sk, hk, d]`
///
/// Returns `(out[b, sq, h, d] BF16, lse[b, h, sq] F32)`, both on the same ROCm
/// device. F32 accumulation throughout (matmuls compute in f32 over bf16 inputs;
/// softmax in f32), output narrowed back to BF16.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_forward(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    let device = q.device();
    // Reservation ownership lives at the score allocator, not at a route or
    // tile planner. Query tiling therefore acquires exactly once per live tile,
    // and callers that merely selected this route never double-charge it.
    let _score_scratch_reservation =
        reserve_materialized_score_scratch(device, b, h, sq, sk, "materialized forward")?;
    let bh = b.checked_mul(h).ok_or_else(|| {
        FlashAttnError::Msg("ROCm materialized attention batch/head overflow".into())
    })?;

    // 1. GQA expand k, v from hk -> h heads: [b, sk, hk, d] -> [b, sk, h, d].
    let k_exp = gqa_expand_heads(k, h, hk, device)?; // [b, sk, h, d]
    let v_exp = gqa_expand_heads(v, h, hk, device)?; // [b, sk, h, d]

    // 2. Reshape q/k/v to [b, h, s, d] (transpose axes 1<->2, contiguous), then
    //    flatten to [b*h, s, d] for batched rocm_matmul.
    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?; // [b, h, sq, d]
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?; // [b, h, sk, d]
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?; // [b, h, sk, d]

    let q3 = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?; // [b*h, sq, d]
    let k3 = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?; // [b*h, sk, d]
    let v3 = map_kt(v_bhsd.reshape(vec![bh, sk, d]))?; // [b*h, sk, d]

    // QK keeps BF16 inputs and requests an F32 output. hipBLASLt still uses
    // FP32 compute, but avoids widening the large Q/K tiles before GEMM.
    let v3f = rocm_cast_to(&v3, KtDType::F32)?;

    // 3. scores = (q @ k^T) * scale  -> [b*h, sq, sk]
    let kt3 = rocm_contig(&map_kt(k3.transpose(1, 2))?)?; // [b*h, d, sk]
    let qk = map_kt(kiln_tensor::rocm_matmul_to_dtype(&q3, &kt3, KtDType::F32))?; // [b*h, sq, sk] f32
    let scaled_scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, scale))?;

    // 4. causal mask (additive via masked_fill with -inf above the diagonal).
    //    For sq == 1 (decode) the single query attends ALL keys 0..sk, so the
    //    causal mask is provably all-zeros (build_causal_mask_u8: offset=sk-1,
    //    allowed=sk-1, no j>allowed) — a no-op. Skip the host mask build + H2D
    //    upload ([bh,1,sk] per attention layer per token) and the masked_fill.
    let masked_scores = if causal && sq > 1 {
        map_kt(kiln_tensor::rocm_causal_mask_fill(
            &scaled_scores,
            sq,
            sk,
            NEG_FILL,
        ))?
    } else {
        scaled_scores.clone()
    };
    // 5. p = softmax(scores) over last axis -> [b*h, sq, sk]; lse from scores+p.
    let p = map_kt(kiln_tensor::rocm_softmax_last_axis(&masked_scores))?; // [b*h, sq, sk] f32
    let lse = compute_lse(&masked_scores, &p, b, h, sq)?; // [b, h, sq] f32

    // 6. out = p @ v -> [b*h, sq, d]; reshape/transpose back to [b, sq, h, d].
    let out3 = rocm_matmul_split_inner_f32(&p, &v3f, "materialized out3")?; // [b*h, sq, d] f32
    let out_bhsd = map_kt(out3.reshape(vec![b, h, sq, d]))?; // [b, h, sq, d]
    let out_bshd = rocm_contig(&map_kt(out_bhsd.transpose(1, 2))?)?; // [b, sq, h, d]
    let out_bf16 = rocm_cast_to(&out_bshd, KtDType::BF16)?; // [b, sq, h, d] bf16

    // The composite path is a chain of async ROCm kernels over large temporary
    // score/probability tensors. Synchronize before returning so those
    // temporaries cannot be dropped and recycled while later kernels still read
    // them. The caller's post-handoff sync is too late because these locals are
    // gone by then.
    rocm_sync_tensor(&out_bf16)?;
    Ok((out_bf16, lse))
}

/// Head-major ROCm SDPA composite.
///
/// Inputs:
/// - `q`: `[b, h, sq, d]`
/// - `k`, `v`: `[b, hk, sk, d]`
///
/// Returns `(out[b, h, sq, d] BF16, lse[b, h, sq] F32)`.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_forward_head_major(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    let device = q.device();
    let _score_scratch_reservation = reserve_materialized_score_scratch(
        device,
        b,
        h,
        sq,
        sk,
        "head-major materialized forward",
    )?;
    let bh = b.checked_mul(h).ok_or_else(|| {
        FlashAttnError::Msg("ROCm head-major attention batch/head overflow".into())
    })?;

    let q_bhsd = rocm_contig(q)?;
    let k_exp = gqa_expand_heads_head_major(k, h, hk, device)?;
    let v_exp = gqa_expand_heads_head_major(v, h, hk, device)?;

    let q3 = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?;
    let k3 = map_kt(k_exp.reshape(vec![bh, sk, d]))?;
    let v3 = map_kt(v_exp.reshape(vec![bh, sk, d]))?;

    let v3f = rocm_cast_to(&v3, KtDType::F32)?;

    let kt3 = rocm_contig(&map_kt(k3.transpose(1, 2))?)?;
    let qk = map_kt(kiln_tensor::rocm_matmul_to_dtype(&q3, &kt3, KtDType::F32))?;
    let scaled_scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, scale))?;

    let masked_scores = if causal && sq > 1 {
        map_kt(kiln_tensor::rocm_causal_mask_fill(
            &scaled_scores,
            sq,
            sk,
            NEG_FILL,
        ))?
    } else {
        scaled_scores.clone()
    };

    let p = map_kt(kiln_tensor::rocm_softmax_last_axis(&masked_scores))?;
    let lse = compute_lse(&masked_scores, &p, b, h, sq)?;

    let out3 = rocm_matmul_split_inner_f32(&p, &v3f, "head-major out3")?;
    let out_bhsd = map_kt(out3.reshape(vec![b, h, sq, d]))?;
    let out_bf16 = rocm_cast_to(&out_bhsd, KtDType::BF16)?;

    rocm_sync_tensor(&out_bf16)?;
    Ok((out_bf16, lse))
}

// ============================================================================
// Forward entry point
// ============================================================================

/// ROCm composite of `flash_attn_fwd_kt`. `q,k,v` are `[b, sq, h, d]` /
/// `[b, sk, hk, d]` BF16 on `Device::Rocm`. Returns `(out[b,sq,h,d] BF16,
/// lse[b,h,sq] F32)`.
pub fn flash_attn_fwd_rocm(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    // Coerce all operands onto one ROCm device: a decode caller may hand us an
    // operand that drifted to CPU via a host-staged op, and the composite (QK^T
    // / softmax / PV) requires every operand on-device. No-op when co-located.
    // (R.4 E2E)
    let dev = [q.device(), k.device(), v.device()]
        .into_iter()
        .find(|d| !d.is_cpu())
        .unwrap_or_else(|| q.device());
    let qc;
    let q = if q.device() != dev {
        qc = map_kt(q.to_device(dev))?;
        &qc
    } else {
        q
    };
    let kc;
    let k = if k.device() != dev {
        kc = map_kt(k.to_device(dev))?;
        &kc
    } else {
        k
    };
    let vc;
    let v = if v.device() != dev {
        vc = map_kt(v.to_device(dev))?;
        &vc
    } else {
        v
    };

    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (sk, hk) = (k.shape()[1], k.shape()[2]);
    let policy = rocm_flash_attention_policy(q)?;
    if native_scalar_fwd_enabled(policy, sq, sk)
        || native_tiled_fwd_enabled(policy, sq, sk)
        || native_streaming_fwd_enabled(policy, sq, sk)
    {
        if let Some(result) =
            try_native_fwd_bf16(q, k, v, b, sq, sk, h, hk, d, softmax_scale, causal, policy)?
        {
            return Ok(result);
        }
    }

    let budget_bytes = materialized_score_budget_bytes();
    let rectangular_causal_prefix = causal && sq != sk;
    if materialized_score_working_set_bytes(b, h, sq, sk)
        .map(|bytes| bytes > budget_bytes)
        .unwrap_or(true)
    {
        if rocm_online_fwd_enabled(policy) && !rectangular_causal_prefix {
            return sdpa_forward_online_tiled(
                q,
                k,
                v,
                b,
                sq,
                sk,
                h,
                hk,
                d,
                softmax_scale,
                causal,
                None,
            );
        }
        return sdpa_forward_query_tiled(
            q,
            k,
            v,
            b,
            sq,
            sk,
            h,
            hk,
            d,
            softmax_scale,
            causal,
            budget_bytes,
        );
    }
    sdpa_forward(q, k, v, b, sq, sk, h, hk, d, softmax_scale, causal)
}

/// ROCm composite for head-major SDPA. `q` is `[b, h, sq, d]`; K/V are
/// `[b, hk, sk, d]`.
pub fn flash_attn_fwd_head_major_rocm(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    let dev = [q.device(), k.device(), v.device()]
        .into_iter()
        .find(|d| !d.is_cpu())
        .unwrap_or_else(|| q.device());
    let qc;
    let q = if q.device() != dev {
        qc = map_kt(q.to_device(dev))?;
        &qc
    } else {
        q
    };
    let kc;
    let k = if k.device() != dev {
        kc = map_kt(k.to_device(dev))?;
        &kc
    } else {
        k
    };
    let vc;
    let v = if v.device() != dev {
        vc = map_kt(v.to_device(dev))?;
        &vc
    } else {
        v
    };

    let (b, h, sq, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (sk, hk) = (k.shape()[2], k.shape()[1]);
    let policy = rocm_flash_attention_policy(q)?;
    let budget_bytes = materialized_score_budget_bytes();
    let rectangular_causal_prefix = causal && sq != sk;
    let exceeds_materialized_budget = materialized_score_working_set_bytes(b, h, sq, sk)
        .map(|bytes| bytes > budget_bytes)
        .unwrap_or(true);
    if exceeds_materialized_budget {
        let use_online = rocm_online_fwd_enabled(policy) && !rectangular_causal_prefix;
        let online_plan = if use_online {
            let bh = b.checked_mul(h).ok_or_else(|| {
                FlashAttnError::Msg("ROCm head-major online batch/head overflow".into())
            })?;
            let fixed_working_set_bytes =
                head_major_online_forward_fixed_working_set_bytes(b, h, hk, sq, sk, d).ok_or_else(
                    || FlashAttnError::Msg("ROCm head-major online working-set overflow".into()),
                )?;
            Some(OnlineForwardPlan::for_operation(
                dev,
                bh,
                sq,
                sk,
                d,
                fixed_working_set_bytes,
            )?)
        } else {
            validate_materialized_query_tile_plan(b, h, sq, sk, budget_bytes)?;
            None
        };
        let online_reservation = online_plan
            .map(|plan| {
                let bh = b.checked_mul(h).ok_or_else(|| {
                    FlashAttnError::Msg("ROCm head-major online batch/head overflow".into())
                })?;
                let operation_peak_bytes =
                    plan.operation_peak_bytes(bh, sq, sk, d).ok_or_else(|| {
                        FlashAttnError::Msg("ROCm head-major online peak estimate overflow".into())
                    })?;
                reserve_published_rocm_bytes(operation_peak_bytes, "head-major online forward")
            })
            .transpose()?;
        let q_bshd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
        let k_bskhd = rocm_contig(&map_kt(k.transpose(1, 2))?)?;
        let v_bskhd = rocm_contig(&map_kt(v.transpose(1, 2))?)?;
        let (out_bshd, lse) = if let Some(plan) = online_plan {
            let reservation = online_reservation.as_ref().ok_or_else(|| {
                FlashAttnError::Msg(
                    "ROCm head-major online forward lost its accepted memory reservation".into(),
                )
            })?;
            sdpa_forward_online_tiled(
                &q_bshd,
                &k_bskhd,
                &v_bskhd,
                b,
                sq,
                sk,
                h,
                hk,
                d,
                softmax_scale,
                causal,
                Some(PreReservedOnlineForwardPlan {
                    plan,
                    _reservation: reservation,
                }),
            )?
        } else {
            sdpa_forward_query_tiled(
                &q_bshd,
                &k_bskhd,
                &v_bskhd,
                b,
                sq,
                sk,
                h,
                hk,
                d,
                softmax_scale,
                causal,
                budget_bytes,
            )?
        };
        let out_bhsd = rocm_contig(&map_kt(out_bshd.transpose(1, 2))?)?;
        return Ok((out_bhsd, lse));
    }
    sdpa_forward_head_major(q, k, v, b, sq, sk, h, hk, d, softmax_scale, causal)
}

/// ROCm forward-only prefill fast path.
///
/// This intentionally returns only the attention output. CK Tile's packaged v3
/// forward kernel on current ROCm does not store softmax LSE, so this must not
/// be used by the tape/replay path that needs LSE for exact backward.
pub fn flash_attn_fwd_rocm_no_lse(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<KtTensor>, FlashAttnError> {
    let dev = [q.device(), k.device(), v.device()]
        .into_iter()
        .find(|d| !d.is_cpu())
        .unwrap_or_else(|| q.device());
    let qc;
    let q = if q.device() != dev {
        qc = map_kt(q.to_device(dev))?;
        &qc
    } else {
        q
    };
    let kc;
    let k = if k.device() != dev {
        kc = map_kt(k.to_device(dev))?;
        &kc
    } else {
        k
    };
    let vc;
    let v = if v.device() != dev {
        vc = map_kt(v.to_device(dev))?;
        &vc
    } else {
        v
    };

    let q_shape = q.shape();
    let k_shape = k.shape();
    if q_shape.len() != 4 || k_shape.len() != 4 || v.shape() != k_shape {
        return Ok(None);
    }
    let (out, _lse) = flash_attn_fwd_rocm(q, k, v, softmax_scale, causal)?;
    Ok(Some(out))
}

#[allow(clippy::too_many_arguments)]
fn try_native_fwd_bf16(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
    policy: RocmFlashAttentionPolicy,
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    if !native_fwd_shape_supported(q, k, v, h, hk, d) {
        return Ok(None);
    }
    // Native FA-style causal masking is bottom-right aligned for `sq != sk`.
    // The query-tiled wrapper narrows each key prefix so that bottom-right
    // alignment equals the absolute prefix-causal mask for chunked training.
    if causal && sq != sk && !native_rectangular_causal_fwd_enabled(policy) {
        return Ok(None);
    }

    let q_tile = native_fwd_query_tile_len(policy);
    let native_single_available =
        native_scalar_fwd_enabled(policy, sq, sk) || native_tiled_fwd_enabled(policy, sq, sk);
    let native_single_allowed =
        native_single_available && native_single_fwd_launch_enabled(policy, sq, sk);
    let prefer_native_single = native_single_allowed && (!causal || sq == sk);
    if prefer_native_single {
        let q_c = rocm_contig(q)?;
        let k_c = rocm_contig(k)?;
        let v_c = rocm_contig(v)?;
        if let Some(result) = try_ffi_fwd_bf16(
            &q_c,
            &k_c,
            &v_c,
            b,
            sq,
            sk,
            h,
            hk,
            d,
            softmax_scale,
            causal,
            policy,
        )? {
            return Ok(Some(result));
        }
    }

    if sq > q_tile && native_tiled_fwd_enabled(policy, sq, sk) {
        if let Some(result) = try_native_fwd_bf16_query_tiled(
            q,
            k,
            v,
            b,
            sq,
            sk,
            h,
            hk,
            d,
            softmax_scale,
            causal,
            q_tile,
            policy,
        )? {
            return Ok(Some(result));
        }
    }

    let q_c = rocm_contig(q)?;
    let k_c = rocm_contig(k)?;
    let v_c = rocm_contig(v)?;
    if native_single_allowed {
        if let Some(result) = try_ffi_fwd_bf16(
            &q_c,
            &k_c,
            &v_c,
            b,
            sq,
            sk,
            h,
            hk,
            d,
            softmax_scale,
            causal,
            policy,
        )? {
            return Ok(Some(result));
        }
    }

    if native_streaming_fwd_enabled(policy, sq, sk) {
        return try_native_streaming_fwd_bf16_query_tiled(
            &q_c,
            &k_c,
            &v_c,
            b,
            sq,
            sk,
            h,
            hk,
            d,
            softmax_scale,
            causal,
            q_tile,
            native_streaming_fwd_key_tile_len(policy),
        );
    }

    Ok(None)
}

fn native_fwd_shape_supported(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    h: usize,
    hk: usize,
    d: usize,
) -> bool {
    q.dtype() == KtDType::BF16
        && k.dtype() == KtDType::BF16
        && v.dtype() == KtDType::BF16
        && matches!(d, 128 | 256)
        && h != 0
        && hk != 0
        && h % hk == 0
}

#[allow(clippy::too_many_arguments)]
fn try_native_fwd_bf16_query_tiled(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
    q_tile_len: usize,
    policy: RocmFlashAttentionPolicy,
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    let q_tile_len = q_tile_len.max(1);
    let causal_offset = sk as isize - sq as isize;
    let k_full = rocm_contig(k)?;
    let v_full = rocm_contig(v)?;
    if let Some(result) = try_native_fwd_bf16_query_tiled_into(
        q,
        &k_full,
        &v_full,
        b,
        sq,
        sk,
        h,
        hk,
        d,
        softmax_scale,
        causal,
        q_tile_len,
        policy,
    )? {
        return Ok(Some(result));
    }

    let mut out_tiles = Vec::new();
    let mut lse_tiles = Vec::new();
    let mut q_start = 0usize;

    while q_start < sq {
        let q_len = (sq - q_start).min(q_tile_len).max(1);
        let key_len = if causal {
            causal_block_limit(sk, q_start, q_len, causal_offset)
        } else {
            sk
        };
        if key_len == 0 {
            return Ok(None);
        }

        let q_tile = rocm_contig(&map_kt(q.narrow(1, q_start, q_len))?)?;
        let abs_tile_result = if causal {
            try_ffi_fwd_bf16_abs_tile(
                &q_tile,
                &k_full,
                &v_full,
                b,
                q_len,
                sk,
                sq,
                h,
                hk,
                d,
                softmax_scale,
                causal,
                q_start,
                policy,
            )?
        } else {
            None
        };
        if let Some((out_tile, lse_tile)) = abs_tile_result {
            out_tiles.push(out_tile);
            lse_tiles.push(lse_tile);
            q_start += q_len;
            continue;
        }

        let k_tile;
        let v_tile;
        let (k_ref, v_ref) = if key_len == sk {
            (&k_full, &v_full)
        } else {
            k_tile = rocm_contig(&map_kt(k.narrow(1, 0, key_len))?)?;
            v_tile = rocm_contig(&map_kt(v.narrow(1, 0, key_len))?)?;
            (&k_tile, &v_tile)
        };

        let Some((out_tile, lse_tile)) = try_ffi_fwd_bf16(
            &q_tile,
            k_ref,
            v_ref,
            b,
            q_len,
            key_len,
            h,
            hk,
            d,
            softmax_scale,
            causal,
            policy,
        )?
        else {
            return Ok(None);
        };
        out_tiles.push(out_tile);
        lse_tiles.push(lse_tile);
        q_start += q_len;
    }

    let out_refs: Vec<&KtTensor> = out_tiles.iter().collect();
    let lse_refs: Vec<&KtTensor> = lse_tiles.iter().collect();
    let out = map_kt(KtTensor::cat(&out_refs, 1))?;
    let lse = map_kt(KtTensor::cat(&lse_refs, 2))?;
    Ok(Some((out, lse)))
}

#[allow(clippy::too_many_arguments)]
fn try_native_fwd_bf16_query_tiled_into(
    q: &KtTensor,
    k_full: &KtTensor,
    v_full: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
    q_tile_len: usize,
    policy: RocmFlashAttentionPolicy,
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    let (q_st, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, sq, h, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let lse = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::F32, vec![b, h, sq])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

    let q_tile_len = q_tile_len.max(1);
    let causal_offset = sk as isize - sq as isize;
    let mut q_start = 0usize;
    while q_start < sq {
        let q_len = (sq - q_start).min(q_tile_len).max(1);
        if causal && causal_block_limit(sk, q_start, q_len, causal_offset) == 0 {
            return Ok(None);
        }
        if !try_ffi_fwd_bf16_abs_tile_base_into(
            q,
            k_full,
            v_full,
            &out,
            &lse,
            b,
            q_len,
            sk,
            sq,
            h,
            hk,
            d,
            softmax_scale,
            causal,
            q_start,
            policy,
        )? {
            return Ok(None);
        }
        q_start += q_len;
    }

    Ok(Some((out, lse)))
}

#[allow(clippy::too_many_arguments)]
fn try_ffi_fwd_bf16(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
    policy: RocmFlashAttentionPolicy,
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    if q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || !matches!(d, 128 | 256)
        || h == 0
        || hk == 0
        || h % hk != 0
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return Ok(None);
    }

    let (q_st, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, sq, h, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let lse = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::F32, vec![b, h, sq])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

    let q_ptr = kiln_kt_bridge::rocm_input_device_ptr(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let k_ptr = kiln_kt_bridge::rocm_input_device_ptr(k, KtDType::BF16, "k")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let v_ptr = kiln_kt_bridge::rocm_input_device_ptr(v, KtDType::BF16, "v")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_output_device_ptr(&out);
    let lse_ptr = kiln_kt_bridge::rocm_output_device_ptr(&lse);

    // The native FFI kernels launch on the same active ROCm stream as tensor
    // ops and stream-ordered allocation/free. Stream ordering preserves
    // producer/consumer and temporary-buffer lifetimes.

    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(q, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let ffi_policy = rocm_flash_attention_ffi_policy(policy)?;
    let status = unsafe {
        crate::kiln_rocm_flash_attn_fwd_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            b as i32,
            sq as i32,
            sk as i32,
            h as i32,
            hk as i32,
            d as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            &ffi_policy,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa native forward")?;
    if status == 0 {
        Ok(Some((out, lse)))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn try_ffi_fwd_bf16_abs_tile(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    q_len: usize,
    sk: usize,
    sq_total: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
    q_start: usize,
    policy: RocmFlashAttentionPolicy,
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    if q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || !matches!(d, 128 | 256)
        || h == 0
        || hk == 0
        || h % hk != 0
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || match q_start.checked_add(q_len) {
            Some(end) => end > sq_total,
            None => true,
        }
    {
        return Ok(None);
    }

    let (q_st, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, q_len, h, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let lse = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::F32, vec![b, h, q_len])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

    let q_ptr = kiln_kt_bridge::rocm_input_device_ptr(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let k_ptr = kiln_kt_bridge::rocm_input_device_ptr(k, KtDType::BF16, "k")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let v_ptr = kiln_kt_bridge::rocm_input_device_ptr(v, KtDType::BF16, "v")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_output_device_ptr(&out);
    let lse_ptr = kiln_kt_bridge::rocm_output_device_ptr(&lse);

    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(q, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let ffi_policy = rocm_flash_attention_ffi_policy(policy)?;
    let status = unsafe {
        crate::kiln_rocm_flash_attn_fwd_abs_tile_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            b as i32,
            q_len as i32,
            sk as i32,
            sq_total as i32,
            h as i32,
            hk as i32,
            d as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            q_start as i32,
            &ffi_policy,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa native forward")?;
    if status == 0 {
        Ok(Some((out, lse)))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn try_ffi_fwd_bf16_abs_tile_base_into(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    out: &KtTensor,
    lse: &KtTensor,
    b: usize,
    q_len: usize,
    sk: usize,
    sq_total: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
    q_start: usize,
    policy: RocmFlashAttentionPolicy,
) -> Result<bool, FlashAttnError> {
    if q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || out.dtype() != KtDType::BF16
        || lse.dtype() != KtDType::F32
        || !matches!(d, 128 | 256)
        || h == 0
        || hk == 0
        || h % hk != 0
        || q.shape() != [b, sq_total, h, d]
        || k.shape() != [b, sk, hk, d]
        || v.shape() != [b, sk, hk, d]
        || out.shape() != [b, sq_total, h, d]
        || lse.shape() != [b, h, sq_total]
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !out.is_contiguous()
        || !lse.is_contiguous()
        || match q_start.checked_add(q_len) {
            Some(end) => end > sq_total,
            None => true,
        }
    {
        return Ok(false);
    }

    let q_ptr = kiln_kt_bridge::rocm_input_device_ptr(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let k_ptr = kiln_kt_bridge::rocm_input_device_ptr(k, KtDType::BF16, "k")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let v_ptr = kiln_kt_bridge::rocm_input_device_ptr(v, KtDType::BF16, "v")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_output_device_ptr(out);
    let lse_ptr = kiln_kt_bridge::rocm_output_device_ptr(lse);

    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(q, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let ffi_policy = rocm_flash_attention_ffi_policy(policy)?;
    let status = unsafe {
        crate::kiln_rocm_flash_attn_fwd_abs_tile_base_into_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            b as i32,
            q_len as i32,
            sk as i32,
            sq_total as i32,
            h as i32,
            hk as i32,
            d as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            q_start as i32,
            &ffi_policy,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa native forward")?;
    if status == 0 { Ok(true) } else { Ok(false) }
}

#[allow(clippy::too_many_arguments)]
fn try_native_streaming_fwd_bf16_query_tiled(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
    q_tile_len: usize,
    key_tile_len: usize,
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    let q_tile_len = q_tile_len.max(1);
    let key_tile_len = key_tile_len.max(1);
    let causal_offset = sk as isize - sq as isize;
    let k_full = rocm_contig(k)?;
    let v_full = rocm_contig(v)?;
    let (q_st, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let mut out_tiles = Vec::new();
    let mut lse_tiles = Vec::new();
    let mut q_start = 0usize;

    while q_start < sq {
        let q_len = (sq - q_start).min(q_tile_len).max(1);
        let block_limit = if causal {
            causal_block_limit(sk, q_start, q_len, causal_offset)
        } else {
            sk
        };
        if block_limit == 0 {
            return Ok(None);
        }

        let q_tile = rocm_contig(&map_kt(q.narrow(1, q_start, q_len))?)?;
        let out_tile = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, q_len, h, d])
            .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let lse_tile = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::F32, vec![b, h, q_len])
            .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let acc = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::F32, vec![b, q_len, h, d])
            .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let row_l = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::F32, vec![b, h, q_len])
            .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let row_m_zero = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::F32, vec![b, h, q_len])
            .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let row_m = map_kt(kiln_tensor::ops::full_like(&row_m_zero, f32::NEG_INFINITY))?;

        // The streaming HIP kernel reads q_tile/k_full/v_full and mutates the
        // online-softmax state buffers in-place. These tensors are produced by
        // async tensor ops immediately above; make the handoff explicit before
        // the first raw-pointer update launch.
        rocm_sync_tensor(&row_m)?;

        let mut key_start = 0usize;
        while key_start < block_limit {
            let key_len = (block_limit - key_start).min(key_tile_len).max(1);
            native_streaming_fwd_update_bf16(
                &q_tile,
                &k_full,
                &v_full,
                &row_m,
                &row_l,
                &acc,
                b,
                q_len,
                sk,
                sq,
                h,
                hk,
                d,
                softmax_scale,
                causal,
                q_start,
                key_start,
                key_len,
            )?;
            key_start += key_len;
        }

        native_streaming_fwd_finalize_bf16(
            &row_m, &row_l, &acc, &out_tile, &lse_tile, b, q_len, h, d,
        )?;
        // Finalize reads the streaming state tensors and writes out_tile/lse.
        // Keep row_m/row_l/acc alive until that async kernel has completed.
        rocm_sync_tensor(&out_tile)?;
        out_tiles.push(out_tile);
        lse_tiles.push(lse_tile);
        q_start += q_len;
    }

    let out_refs: Vec<&KtTensor> = out_tiles.iter().collect();
    let lse_refs: Vec<&KtTensor> = lse_tiles.iter().collect();
    let out = map_kt(KtTensor::cat(&out_refs, 1))?;
    let lse = map_kt(KtTensor::cat(&lse_refs, 2))?;
    Ok(Some((out, lse)))
}

#[allow(clippy::too_many_arguments)]
fn native_streaming_fwd_update_bf16(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    row_m: &KtTensor,
    row_l: &KtTensor,
    acc: &KtTensor,
    b: usize,
    q_len: usize,
    sk: usize,
    sq_total: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
    q_start: usize,
    key_start: usize,
    key_len: usize,
) -> Result<(), FlashAttnError> {
    let q_ptr = kiln_kt_bridge::rocm_input_device_ptr(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let k_ptr = kiln_kt_bridge::rocm_input_device_ptr(k, KtDType::BF16, "k")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let v_ptr = kiln_kt_bridge::rocm_input_device_ptr(v, KtDType::BF16, "v")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let row_m_ptr = kiln_kt_bridge::rocm_output_device_ptr(row_m);
    let row_l_ptr = kiln_kt_bridge::rocm_output_device_ptr(row_l);
    let acc_ptr = kiln_kt_bridge::rocm_output_device_ptr(acc);
    let b = usize_to_i32("streaming fwd batch", b)?;
    let q_len = usize_to_i32("streaming fwd q_len", q_len)?;
    let sk = usize_to_i32("streaming fwd seqlen_k", sk)?;
    let sq_total = usize_to_i32("streaming fwd seqlen_q_total", sq_total)?;
    let h = usize_to_i32("streaming fwd heads", h)?;
    let hk = usize_to_i32("streaming fwd kv_heads", hk)?;
    let d = usize_to_i32("streaming fwd head_dim", d)?;
    let q_start = usize_to_i32("streaming fwd q_start", q_start)?;
    let key_start = usize_to_i32("streaming fwd key_start", key_start)?;
    let key_len = usize_to_i32("streaming fwd key_len", key_len)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(q, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_attn_fwd_stream_update_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            row_m_ptr as *mut _,
            row_l_ptr as *mut _,
            acc_ptr as *mut _,
            b,
            q_len,
            sk,
            sq_total,
            h,
            hk,
            d,
            softmax_scale,
            if causal { 1 } else { 0 },
            q_start,
            key_start,
            key_len,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa stream update")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "rocm native streaming fwd update returned status {status}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn native_streaming_fwd_finalize_bf16(
    row_m: &KtTensor,
    row_l: &KtTensor,
    acc: &KtTensor,
    out: &KtTensor,
    lse: &KtTensor,
    b: usize,
    q_len: usize,
    h: usize,
    d: usize,
) -> Result<(), FlashAttnError> {
    let row_m_ptr = kiln_kt_bridge::rocm_input_device_ptr(row_m, KtDType::F32, "row_m")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let row_l_ptr = kiln_kt_bridge::rocm_input_device_ptr(row_l, KtDType::F32, "row_l")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let acc_ptr = kiln_kt_bridge::rocm_input_device_ptr(acc, KtDType::F32, "acc")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_output_device_ptr(out);
    let lse_ptr = kiln_kt_bridge::rocm_output_device_ptr(lse);
    let b = usize_to_i32("streaming fwd finalize batch", b)?;
    let q_len = usize_to_i32("streaming fwd finalize q_len", q_len)?;
    let h = usize_to_i32("streaming fwd finalize heads", h)?;
    let d = usize_to_i32("streaming fwd finalize head_dim", d)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(out, "out")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_attn_fwd_stream_finalize_bf16(
            row_m_ptr as *const _,
            row_l_ptr as *const _,
            acc_ptr as *const _,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            b,
            q_len,
            h,
            d,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa stream finalize")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "rocm native streaming fwd finalize returned status {status}"
        )));
    }
    Ok(())
}

// ============================================================================
// Paged decode
// ============================================================================

/// Gather K (or V) rows for every (batch, logical-position) from a token-major
/// pool `[total_slots, hk, d]` via `block_table[b, blk]`. Produces
/// `[b, seqlen_k, hk, d]` on-device.
///
/// Physical slot for logical position `t` of sequence `b`:
///   `block_table[b, t / page_block_size] * page_block_size + t % page_block_size`.
fn paged_gather(
    pool: &KtTensor,        // [total_slots, hk, d]
    block_table: &KtTensor, // device U32 [b, max_blocks_per_seq]
    b: usize,
    seqlen_k: usize,
    max_blocks_per_seq: usize,
    page_block_size: usize,
    hk: usize,
    d: usize,
    _device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    let gathered = map_kt(kiln_tensor::rocm_paged_gather_rows(
        pool,
        block_table,
        b,
        seqlen_k,
        max_blocks_per_seq,
        page_block_size,
    ))?;
    if gathered.shape() != [b, seqlen_k, hk, d] {
        return Err(FlashAttnError::Msg(format!(
            "rocm paged gather shape mismatch: got {:?}, expected {:?}",
            gathered.shape(),
            [b, seqlen_k, hk, d]
        )));
    }
    Ok(gathered)
}

/// If the gathered KV is U8 (FP8 E4M3FN pool slots), dequantize to BF16 on-device
/// (scale=1.0 "direct" — the paged KV cache uses unscaled direct quantization).
/// This lets an FP8 cache take the bucketed, capture-safe SDPA fast path instead
/// of the eager attention (whose seq-len-dependent broadcast indices are both
/// host-bound and not HIP-graph-capturable). No-op for BF16 pools. `rocm_index_
/// select_dim0` is dtype-agnostic, so the gather above already produced the U8
/// rows; this just decodes them in place of a fresh BF16 buffer.
fn dequant_gathered_if_fp8(g: KtTensor) -> Result<KtTensor, FlashAttnError> {
    if g.dtype() == KtDType::U8 {
        let c = rocm_contig(&g)?;
        map_kt(kiln_tensor::rocm_fp8_dequantize_direct(&c, KtDType::BF16))
    } else {
        Ok(g)
    }
}

/// ROCm composite of `flash_attn_paged_decode_kt`. `q` is `[b, 1, h, d]`; K/V
/// are gathered from `k_pool`/`v_pool` `[total_slots, hk, d]` via `block_table`.
/// Returns `out[b, 1, h, d]` BF16 on `Device::Rocm`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_paged_decode_rocm(
    q: &KtTensor,
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    block_table: &KtTensor,
    seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    let device = q.device();
    // Read sq from the query (q is [b, sq, h, d]) — do NOT hardcode sq=1. With
    // multi-token decode (MTP / speculative drafting, sq>1) hardcoding sq=1
    // computed attention for only the first query row and left the rest garbage,
    // which compounded into incoherent output after a few accepted tokens. The
    // working prefill path (flash_attn_fwd_rocm) reads sq the same way; for sq>1
    // sdpa_forward's causal mask masks each query to its own position.
    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let hk = k_pool.shape()[1];
    let max_blocks_per_seq = block_table.shape()[1];

    if sq == 1
        && q.dtype() == KtDType::BF16
        && k_pool.dtype() == KtDType::BF16
        && v_pool.dtype() == KtDType::BF16
    {
        return map_kt(kiln_tensor::rocm_paged_attn_decode_bf16(
            q,
            k_pool,
            v_pool,
            block_table,
            None,
            seqlen_k,
            page_block_size,
            softmax_scale,
        ));
    }

    // block_table (U32 [b, blocks]) stays on-device; the gather index is built
    // on-GPU by paged_gather (no host round-trip).
    let k_gathered = paged_gather(
        k_pool,
        block_table,
        b,
        seqlen_k,
        max_blocks_per_seq,
        page_block_size,
        hk,
        d,
        device,
    )?; // [b, seqlen_k, hk, d]
    let v_gathered = paged_gather(
        v_pool,
        block_table,
        b,
        seqlen_k,
        max_blocks_per_seq,
        page_block_size,
        hk,
        d,
        device,
    )?; // [b, seqlen_k, hk, d]
    // FP8 pools gather as U8 (E4M3FN); decode to BF16 before the flash math.
    let k_gathered = dequant_gathered_if_fp8(k_gathered)?;
    let v_gathered = dequant_gathered_if_fp8(v_gathered)?;

    let (out, _lse) = sdpa_forward(
        q,
        &k_gathered,
        &v_gathered,
        b,
        sq,
        seqlen_k,
        h,
        hk,
        d,
        softmax_scale,
        causal,
    )?;
    Ok(out)
}

/// ROCm composite of `flash_attn_paged_decode_dyn_seqlen_kt`. Like
/// [`flash_attn_paged_decode_rocm`] but with a per-batch `seqused_k` bound: keys
/// `t >= seqused_k[b]` are masked out (additive `-inf`) so they don't contribute.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_paged_decode_dyn_seqlen_rocm(
    q: &KtTensor,
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    block_table: &KtTensor,
    seqused_k: &KtTensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    let device = q.device();
    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let hk = k_pool.shape()[1];
    let max_blocks_per_seq = block_table.shape()[1];

    if sq == 1
        && q.dtype() == KtDType::BF16
        && k_pool.dtype() == KtDType::BF16
        && v_pool.dtype() == KtDType::BF16
    {
        let _ = causal;
        return map_kt(kiln_tensor::rocm_paged_attn_decode_bf16(
            q,
            k_pool,
            v_pool,
            block_table,
            Some(seqused_k),
            max_seqlen_k,
            page_block_size,
            softmax_scale,
        ));
    }

    // block_table + seqused_k stay on-device: paged_gather builds the gather
    // index on-GPU, and sdpa_forward_dyn_tail builds the tail mask on-GPU from
    // the device seqused_k (no D2H/H2D round-trip per attention layer).
    let k_gathered = paged_gather(
        k_pool,
        block_table,
        b,
        max_seqlen_k,
        max_blocks_per_seq,
        page_block_size,
        hk,
        d,
        device,
    )?; // [b, max_seqlen_k, hk, d]
    let v_gathered = paged_gather(
        v_pool,
        block_table,
        b,
        max_seqlen_k,
        max_blocks_per_seq,
        page_block_size,
        hk,
        d,
        device,
    )?;
    // FP8 pools gather as U8 (E4M3FN); decode to BF16 before the flash math.
    let k_gathered = dequant_gathered_if_fp8(k_gathered)?;
    let v_gathered = dequant_gathered_if_fp8(v_gathered)?;

    // Run SDPA (sq=1, non-causal core); apply the per-batch tail mask by zeroing
    // the K contribution beyond seqused_k. We fold the tail mask into the scores
    // through a dedicated mask path: compute scores ourselves so we can mask.
    sdpa_forward_dyn_tail(
        q,
        &k_gathered,
        &v_gathered,
        b,
        max_seqlen_k,
        h,
        hk,
        d,
        softmax_scale,
        causal,
        seqused_k,
    )
}

/// SDPA variant for dyn-seqlen paged decode: identical to [`sdpa_forward`] with
/// `sq = 1`, but additionally masks keys `j >= seqused_k[b]` per batch (additive
/// `-inf`) before the softmax. Returns only `out[b, 1, h, d]` BF16.
#[allow(clippy::too_many_arguments)]
fn sdpa_forward_dyn_tail(
    q: &KtTensor,
    k: &KtTensor, // [b, sk, hk, d]
    v: &KtTensor, // [b, sk, hk, d]
    b: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
    seqused_k: &KtTensor, // device U32 [b]
) -> Result<KtTensor, FlashAttnError> {
    let device = q.device();
    let sq = 1usize;
    let bh = b * h;

    let k_exp = gqa_expand_heads(k, h, hk, device)?;
    let v_exp = gqa_expand_heads(v, h, hk, device)?;

    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?;
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?;

    let q3 = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?;
    let k3 = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?;
    let v3 = map_kt(v_bhsd.reshape(vec![bh, sk, d]))?;

    let v3f = rocm_cast_to(&v3, KtDType::F32)?;

    let kt3 = rocm_contig(&map_kt(k3.transpose(1, 2))?)?;
    let qk = map_kt(kiln_tensor::rocm_matmul_to_dtype(&q3, &kt3, KtDType::F32))?; // [b*h, 1, sk]
    let scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, scale))?;

    // Build the U8 tail mask [b*h, 1, sk] ON-DEVICE from the device seqused_k:
    // mask[bi,hi,j] = (j >= seqused_k[bi]). For sq=1 decode the causal constraint
    // is subsumed by the tail (the newest query attends all keys 0..used-1), so
    // there is no separate causal term — bit-identical to the former host loop.
    let _ = causal;
    let mask_t = map_kt(kiln_tensor::rocm_build_tail_mask(seqused_k, b, h, sk))?;
    let scores = map_kt(kiln_tensor::rocm_masked_fill(&scores, &mask_t, NEG_FILL))?;

    let p = map_kt(kiln_tensor::rocm_softmax_last_axis(&scores))?;
    let out3 = rocm_matmul_split_inner_f32(&p, &v3f, "dyn-tail out3")?; // [b*h, 1, d]
    let out_bhsd = map_kt(out3.reshape(vec![b, h, sq, d]))?;
    let out_bshd = rocm_contig(&map_kt(out_bhsd.transpose(1, 2))?)?; // [b, 1, h, d]
    rocm_cast_to(&out_bshd, KtDType::BF16)
}

// ============================================================================
// Paged KV write (in-place device-to-device copy into the pool)
// ============================================================================

// HIP runtime device-to-device async memcpy. The symbol is linked transitively
// via `kiln-tensor`'s build.rs (`cargo:rustc-link-lib=dylib=amdhip64`), so a
// bare `extern "C"` declaration resolves at link time. Signature matches
// `hipMemcpyDtoDAsync(void* dst, void* src, size_t sizeBytes, hipStream_t)`.
unsafe extern "C" {
    fn hipMemcpyDtoDAsync(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        size_bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Copy `n_bytes` from `src_ptr` into `dst_ptr` (both raw ROCm device addresses)
/// on `stream_owner`'s active stream. Admission is acquired immediately around
/// this one runtime call, so batched paged-KV writers cannot retain a raw stream
/// across an unbounded host loop.
fn rocm_d2d_copy(
    dst_ptr: u64,
    src_ptr: u64,
    n_bytes: usize,
    stream_owner: &KtTensor,
) -> Result<(), FlashAttnError> {
    let stream_submission =
        kiln_kt_bridge::rocm_stream_submission_of(stream_owner, "rocm_d2d_copy")?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        hipMemcpyDtoDAsync(
            dst_ptr as *mut core::ffi::c_void,
            src_ptr as *const core::ffi::c_void,
            n_bytes,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa device copy")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "rocm-sdpa: hipMemcpyDtoDAsync returned status {status}"
        )));
    }
    // R.10 perf: no device sync needed — the d2d write and any later read of the
    // pool serialize on the one cached per-device stream (FIFO). The old
    // hipDeviceSynchronize stalled the decode pipeline every KV write.
    Ok(())
}

/// Copy a freshly-computed `src` ROCm tensor into a caller-owned `dst` ROCm
/// buffer, device-to-device, matching shapes/dtype. Used by the
/// caller-owned-output (graph-capture) decode variant. `src` is made contiguous
/// first; `dst` must be contiguous (the kernel write contract).
pub fn rocm_copy_into(src: &KtTensor, dst: &KtTensor) -> Result<(), FlashAttnError> {
    if src.shape() != dst.shape() || src.dtype() != dst.dtype() {
        return Err(FlashAttnError::Msg(format!(
            "rocm-sdpa: copy_into shape/dtype mismatch src {:?}/{:?} dst {:?}/{:?}",
            src.shape(),
            src.dtype(),
            dst.shape(),
            dst.dtype()
        )));
    }
    let src_c = rocm_contig(src)?;
    let n_bytes = dst.element_count() * dst.dtype().size_in_bytes();
    let src_ptr = kiln_kt_bridge::rocm_input_device_ptr(&src_c, src.dtype(), "copy_src")?;
    let dst_ptr = kiln_kt_bridge::rocm_input_device_ptr(dst, dst.dtype(), "copy_dst")?;
    rocm_d2d_copy(dst_ptr, src_ptr, n_bytes, dst)
}

/// ROCm composite of `paged_kv_write_token_major_bf16_kt` (host-`usize`-slot
/// variant). Writes the single token rows `k`/`v` (`[num_kv_heads * head_dim]`
/// BF16 each) into `k_pool`/`v_pool` at physical row `slot`, in place, on-device.
pub fn paged_kv_write_token_major_bf16_rocm(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slot: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), FlashAttnError> {
    let row_elems = num_kv_heads * head_dim;
    let bpe = KtDType::BF16.size_in_bytes();
    let row_bytes = row_elems * bpe;
    let slot_byte_off = (slot * row_elems * bpe) as u64;

    let k_src = kiln_kt_bridge::rocm_input_device_ptr(&rocm_contig(k)?, KtDType::BF16, "k")?;
    let v_src = kiln_kt_bridge::rocm_input_device_ptr(&rocm_contig(v)?, KtDType::BF16, "v")?;
    let kp_dst = kiln_kt_bridge::rocm_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let vp_dst = kiln_kt_bridge::rocm_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    rocm_d2d_copy(kp_dst + slot_byte_off, k_src, row_bytes, k_pool)?;
    rocm_d2d_copy(vp_dst + slot_byte_off, v_src, row_bytes, k_pool)?;
    Ok(())
}

/// ROCm composite of `paged_kv_write_token_major_bf16_slot_kt` (device-`[1]`-U32
/// slot variant). Stages the single slot index to host (tiny metadata), then
/// reuses the host-slot writer.
pub fn paged_kv_write_token_major_bf16_slot_rocm(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slot: &KtTensor,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), FlashAttnError> {
    // R.9: on-device scatter-copy of the current token's K/V row into
    // `pool[*slot]` using the DEVICE slot index — NO host readback, so this
    // records cleanly into a captured HIP decode graph (and is safe on the
    // Borrowed freeze-pointer arena buffers, since it writes through
    // device_ptr_raw, not slice()). Reshape pool -> [n_rows, row_elems] and the
    // token row -> [1, row_elems], then `dst[*slot] = row` via index_copy_dim0.
    let row_elems = num_kv_heads * head_dim;
    let kp_rows = k_pool.element_count() / row_elems;
    let vp_rows = v_pool.element_count() / row_elems;
    let k_pool2 = map_kt(k_pool.reshape(vec![kp_rows, row_elems]))?;
    let v_pool2 = map_kt(v_pool.reshape(vec![vp_rows, row_elems]))?;
    let k_row = map_kt(rocm_contig(k)?.reshape(vec![1, row_elems]))?;
    let v_row = map_kt(rocm_contig(v)?.reshape(vec![1, row_elems]))?;
    let slot1 = map_kt(rocm_contig(slot)?.reshape(vec![1]))?;
    map_kt(kiln_tensor::rocm_index_copy_dim0(&k_pool2, &slot1, &k_row))?;
    map_kt(kiln_tensor::rocm_index_copy_dim0(&v_pool2, &slot1, &v_row))?;
    Ok(())
}

/// ROCm composite of `paged_kv_write_token_major_bf16_batch_slot_kt` (batched
/// device-`[batch]`-U32 slots). `k`/`v` are `[batch * num_kv_heads * head_dim]`
/// BF16; row `r` is written to physical pool row `slots[r]`.
pub fn paged_kv_write_token_major_bf16_batch_slot_rocm(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slots: &KtTensor,
    batch: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), FlashAttnError> {
    let row_elems = num_kv_heads * head_dim;
    let k_pool_rows = k_pool.element_count() / row_elems;
    let v_pool_rows = v_pool.element_count() / row_elems;
    let k_pool_2d = map_kt(k_pool.reshape(vec![k_pool_rows, row_elems]))?;
    let v_pool_2d = map_kt(v_pool.reshape(vec![v_pool_rows, row_elems]))?;
    let k_rows = map_kt(rocm_contig(k)?.reshape(vec![batch, row_elems]))?;
    let v_rows = map_kt(rocm_contig(v)?.reshape(vec![batch, row_elems]))?;
    let slots = map_kt(rocm_contig(slots)?.reshape(vec![batch]))?;

    // Keep slot selection on the device. Besides replacing 2 * batch D2D
    // submissions with two indexed scatters, this is capture-safe: the slot
    // tensor's stable pointer is consumed by the kernel on every replay.
    map_kt(kiln_tensor::rocm_index_copy_dim0(
        &k_pool_2d, &slots, &k_rows,
    ))?;
    map_kt(kiln_tensor::rocm_index_copy_dim0(
        &v_pool_2d, &slots, &v_rows,
    ))?;
    Ok(())
}

// ============================================================================
// Backward
// ============================================================================

/// ROCm backward for `flash_attn_bwd_kt`. The bf16/head_dim=128/256 path uses
/// an exact materialized BLASLt composite when the governor-published operation
/// peak and BLASLt heuristics allow it, otherwise a native exact HIP kernel consumes
/// the saved forward output and `softmax_lse`. Other shapes fall back to the
/// composite that recomputes scores and produces `(dq, dk, dv)` via matmuls:
///   dv = p^T @ dout
///   dp = dout @ v^T
///   ds = p * (dp - rowsum(dp * p))
///   dq = ds @ k * scale
///   dk = ds^T @ q * scale
/// All in F32, outputs BF16 `[b, s, h, d]`. dk/dv are returned at the EXPANDED
/// head count `h` (matching the CUDA FFI's expanded-GQA buffer contract).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bwd_rocm(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    out: &KtTensor,
    softmax_lse: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor, KtTensor), FlashAttnError> {
    let device = q.device();
    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (sk, hk) = (k.shape()[1], k.shape()[2]);
    let policy = rocm_flash_attention_policy(q)?;

    if rocm_native_bwd_preferred(policy, b, h, sq, sk, d) {
        if let Some(result) = try_native_bwd_bf16(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            b,
            sq,
            sk,
            h,
            hk,
            d,
            softmax_scale,
            causal,
        )? {
            return Ok(result);
        }
    }

    // The native HIP backward is memory-bounded and exact, but scalar. For
    // tiles that fit the materialized score budget, the BLASLt composite is
    // much faster on long-context training. If hipBLASLt has no algorithm for a
    // specific shape, fall back to the exact native kernel instead of failing.
    if rocm_materialized_bwd_enabled(policy, device, b, h, sq, sk, d) {
        match materialized_bwd_composite_rocm(
            dout,
            q,
            k,
            v,
            softmax_scale,
            causal,
            b,
            sq,
            sk,
            h,
            hk,
            d,
        ) {
            Ok(result) => return Ok(result),
            Err(composite_err) => {
                if rocm_execution_quarantined(q)? {
                    return Err(composite_err);
                }
                if let Some(result) = try_native_bwd_bf16(
                    dout,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    b,
                    sq,
                    sk,
                    h,
                    hk,
                    d,
                    softmax_scale,
                    causal,
                )? {
                    return Ok(result);
                }
                return Err(composite_err);
            }
        }
    }

    let mut online_err: Option<FlashAttnError> = None;
    if rocm_online_bwd_enabled(policy) {
        match online_bwd_tiled_rocm(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            b,
            sq,
            sk,
            h,
            hk,
            d,
        ) {
            Ok(result) => return Ok(result),
            Err(err) => {
                if rocm_execution_quarantined(q)? {
                    return Err(err);
                }
                online_err = Some(err);
            }
        }
    }

    match materialized_bwd_query_tiled_rocm(
        dout,
        q,
        k,
        v,
        softmax_scale,
        causal,
        b,
        sq,
        sk,
        h,
        hk,
        d,
    ) {
        Ok(result) => return Ok(result),
        Err(tiled_err) => {
            if rocm_execution_quarantined(q)? {
                return Err(tiled_err);
            }
            if let Some(result) = try_native_bwd_bf16(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                b,
                sq,
                sk,
                h,
                hk,
                d,
                softmax_scale,
                causal,
            )? {
                return Ok(result);
            }
            if let Some(online_err) = online_err {
                return Err(online_err);
            }
            return Err(tiled_err);
        }
    }
}

/// ROCm backward variant for production training.
///
/// `flash_attn_bwd_kt` mirrors CUDA's historical expanded-GQA buffer contract:
/// `dk`/`dv` are returned at `h` query heads. The model tape immediately
/// collapses those buffers back to `hk` K/V heads. For ROCm, avoid that extra
/// expanded BF16 materialization when the exact materialized composite is used;
/// otherwise fall back to the expanded path and collapse on-device.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bwd_rocm_collapsed_gqa(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    out: &KtTensor,
    softmax_lse: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor, KtTensor), FlashAttnError> {
    let device = q.device();
    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (sk, hk) = (k.shape()[1], k.shape()[2]);
    let policy = rocm_flash_attention_policy(q)?;
    if hk == h {
        return flash_attn_bwd_rocm(dout, q, k, v, out, softmax_lse, softmax_scale, causal);
    }
    if hk == 0 || h % hk != 0 {
        return Err(FlashAttnError::Msg(format!(
            "rocm-sdpa collapsed bwd: invalid GQA heads h={h} hk={hk}"
        )));
    }

    if rocm_collapsed_gqa_bwd_enabled(policy)
        && rocm_native_direct_collapsed_gqa_bwd_enabled(policy)
        && rocm_native_bwd_preferred(policy, b, h, sq, sk, d)
    {
        if let Some(result) = try_native_bwd_bf16_collapsed_gqa(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            b,
            sq,
            sk,
            h,
            hk,
            d,
            softmax_scale,
            causal,
        )? {
            return Ok(result);
        }
    }

    if !rocm_native_bwd_preferred(policy, b, h, sq, sk, d)
        && rocm_materialized_bwd_enabled(policy, device, b, h, sq, sk, d)
    {
        match materialized_bwd_composite_rocm_impl(
            dout,
            q,
            k,
            v,
            softmax_scale,
            causal,
            b,
            sq,
            sk,
            h,
            hk,
            d,
            true,
        ) {
            Ok(result) => return Ok(result),
            Err(error) => {
                if rocm_execution_quarantined(q)? {
                    return Err(error);
                }
                // Fall through to the general ROCm backward below. It has the
                // native fallback logic for shapes hipBLASLt cannot handle.
            }
        }
    }

    let (dq, dk_exp, dv_exp) =
        flash_attn_bwd_rocm(dout, q, k, v, out, softmax_lse, softmax_scale, causal)?;
    let dk = collapse_expanded_bshd_gqa_grad_bf16(&dk_exp, b, sk, h, hk, d)?;
    let dv = collapse_expanded_bshd_gqa_grad_bf16(&dv_exp, b, sk, h, hk, d)?;
    Ok((dq, dk, dv))
}

#[allow(clippy::too_many_arguments)]
fn materialized_bwd_composite_rocm(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
) -> Result<(KtTensor, KtTensor, KtTensor), FlashAttnError> {
    materialized_bwd_composite_rocm_impl(
        dout,
        q,
        k,
        v,
        softmax_scale,
        causal,
        b,
        sq,
        sk,
        h,
        hk,
        d,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn materialized_bwd_composite_rocm_impl(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    collapse_gqa: bool,
) -> Result<(KtTensor, KtTensor, KtTensor), FlashAttnError> {
    let device = q.device();
    let available_bytes =
        require_published_rocm_available_bytes(device, "full materialized backward")?;
    let memory_plan = validate_full_materialized_bwd_admission(available_bytes, b, h, sq, sk, d)?;
    let _memory_reservation =
        reserve_published_rocm_bytes(memory_plan.peak_bytes, "full materialized backward")?;
    let bh = b.checked_mul(h).ok_or_else(|| {
        FlashAttnError::Msg("ROCm full materialized backward batch/head overflow".into())
    })?;

    // `softmax_lse` is the saved forward log-sum-exp. The composite recomputes
    // `scores` and the softmax `p` directly from q/k below (softmax is fully
    // determined by `scores`, so recompute == the lse-shifted forward p), so the
    // saved lse is not needed for correctness here. Kept in the signature to
    // mirror the CUDA `flash_attn_bwd_kt` contract.

    // Expand GQA + reshape to [b*h, s, d] f32, exactly as forward.
    let k_exp = gqa_expand_heads(k, h, hk, device)?;
    let v_exp = gqa_expand_heads(v, h, hk, device)?;

    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?;
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?;
    let do_bhsd = rocm_contig(&map_kt(dout.transpose(1, 2))?)?;

    let q3_bf = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?;
    let k3_bf = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?;
    let q3 = rocm_cast_to(&q3_bf, KtDType::F32)?;
    let k3 = rocm_cast_to(&k3_bf, KtDType::F32)?;
    let v3 = rocm_cast_to(&map_kt(v_bhsd.reshape(vec![bh, sk, d]))?, KtDType::F32)?;
    let do3 = rocm_cast_to(&map_kt(do_bhsd.reshape(vec![bh, sq, d]))?, KtDType::F32)?;

    // Recompute scores + softmax p (same as forward). Keep score-sized
    // temporaries scoped tightly; long-context replay relies on these drops to
    // stay under the live materialized-score budget.
    let p = {
        let kt3_bf = rocm_contig(&map_kt(k3_bf.transpose(1, 2))?)?; // [b*h, d, sk]
        let qk = map_kt(kiln_tensor::rocm_matmul_to_dtype(
            &q3_bf,
            &kt3_bf,
            KtDType::F32,
        ))?; // [b*h, sq, sk]
        drop(kt3_bf);
        let scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, softmax_scale))?;
        drop(qk);
        let scores = if causal {
            causal_mask_fill_offset_f32(scores, bh, sq, sk, 0, sk as isize - sq as isize)?
        } else {
            scores
        };
        let p = map_kt(kiln_tensor::rocm_softmax_last_axis(&scores))?; // [b*h, sq, sk]
        p
    };

    // dv = p^T @ dout   -> [b*h, sk, d]
    let dv3 = {
        let pt = rocm_contig(&map_kt(p.transpose(1, 2))?)?; // [b*h, sk, sq]
        let dv3 = rocm_matmul_split_inner_f32(&pt, &do3, "bwd dv3")?; // [b*h, sk, d]
        drop(pt);
        dv3
    };

    // dp = dout @ v^T   -> [b*h, sq, sk]
    let dp = {
        let vt3 = rocm_contig(&map_kt(v3.transpose(1, 2))?)?; // [b*h, d, sk]
        let dp = map_kt(kiln_tensor::rocm_matmul(&do3, &vt3))?; // [b*h, sq, sk]
        drop(vt3);
        dp
    };

    // ds = p * (dp - rowsum(dp * p))
    // rowsum over last axis: rocm_sum_axis(dp*p, 2) -> [b*h, sq]; subtract via
    // ROCm broadcast. All score-sized elementwise work stays on-device.
    let ds = {
        let dpp = elementwise_mul_f32(&dp, &p, device)?; // [b*h, sq, sk]
        let rowsum = map_kt(kiln_tensor::rocm_sum_axis(&dpp, 2))?; // [b*h, sq]
        drop(dpp);
        // Broadcast rowsum [b*h, sq] -> [b*h, sq, sk] (stride-0 last axis), contiguous.
        let dp_minus = row_broadcast_sub_last_axis_f32(&dp, &rowsum, sk)?; // [b*h, sq, sk]
        drop(rowsum);
        drop(dp);
        let ds = elementwise_mul_f32(&p, &dp_minus, device)?; // [b*h, sq, sk]
        drop(p);
        drop(dp_minus);
        ds
    };

    // dq = ds @ k * scale   -> [b*h, sq, d]
    let dq3 = rocm_matmul_split_inner_f32(&ds, &k3, "bwd dq3")?; // [b*h, sq, d]
    let dq3 = map_kt(kiln_tensor::rocm_scalar_op(&dq3, SCALAR_MUL, softmax_scale))?;

    // dk = ds^T @ q * scale -> [b*h, sk, d]
    let dst = rocm_contig(&map_kt(ds.transpose(1, 2))?)?; // [b*h, sk, sq]
    drop(ds);
    let dk3 = rocm_matmul_split_inner_f32(&dst, &q3, "bwd dk3")?; // [b*h, sk, d]
    drop(dst);
    let dk3 = map_kt(kiln_tensor::rocm_scalar_op(&dk3, SCALAR_MUL, softmax_scale))?;

    // Reshape back to BF16. The collapsed path keeps the exact same F32
    // arithmetic for each query head, then sums query-head groups before the
    // BF16 cast so it matches the model tape's previous F32 collapse.
    let dq = bhsd3_to_bshd_bf16(&dq3, b, h, sq, d)?;
    let (dk, dv) = if collapse_gqa && hk != h {
        (
            collapse_bhsd3_gqa_grad_to_bshd_bf16(&dk3, b, h, hk, sk, d)?,
            collapse_bhsd3_gqa_grad_to_bshd_bf16(&dv3, b, h, hk, sk, d)?,
        )
    } else {
        (
            bhsd3_to_bshd_bf16(&dk3, b, h, sk, d)?,
            bhsd3_to_bshd_bf16(&dv3, b, h, sk, d)?,
        )
    };

    Ok((dq, dk, dv))
}

#[allow(clippy::too_many_arguments)]
fn materialized_bwd_query_tiled_rocm(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
) -> Result<(KtTensor, KtTensor, KtTensor), FlashAttnError> {
    if sq == 0 || sk == 0 {
        return materialized_bwd_composite_rocm(
            dout,
            q,
            k,
            v,
            softmax_scale,
            causal,
            b,
            sq,
            sk,
            h,
            hk,
            d,
        );
    }

    let device = q.device();
    let available_bytes =
        require_published_rocm_available_bytes(device, "materialized query-tiled backward")?;
    let memory_plan = plan_materialized_bwd_query_tiles(available_bytes, b, h, sq, sk, d)?;
    let operation_peak_bytes = memory_plan
        .peak_bytes()
        .ok_or_else(|| FlashAttnError::Msg("ROCm tiled backward peak estimate overflow".into()))?;
    let _memory_reservation =
        reserve_published_rocm_bytes(operation_peak_bytes, "materialized query-tiled backward")?;
    let planned_tile = memory_plan.query_tile;
    let bh = b.checked_mul(h).ok_or_else(|| {
        FlashAttnError::Msg("ROCm materialized backward batch/head overflow".into())
    })?;
    let causal_offset = sk as isize - sq as isize;
    let k_exp = gqa_expand_heads(k, h, hk, device)?;
    let v_exp = gqa_expand_heads(v, h, hk, device)?;

    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?;
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?;
    let do_bhsd = rocm_contig(&map_kt(dout.transpose(1, 2))?)?;

    let q3_bf = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?;
    let k3_bf = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?;
    let do3_bf = map_kt(do_bhsd.reshape(vec![bh, sq, d]))?;

    let kt3_bf = rocm_contig(&map_kt(k3_bf.transpose(1, 2))?)?; // [b*h, d, sk]
    let k3 = rocm_cast_to(&k3_bf, KtDType::F32)?;
    let vt3 = {
        let v3_bf = map_kt(v_bhsd.reshape(vec![bh, sk, d]))?;
        let v3 = rocm_cast_to(&v3_bf, KtDType::F32)?;
        rocm_contig(&map_kt(v3.transpose(1, 2))?)? // [b*h, d, sk]
    };

    let mut dq_tiles = Vec::new();
    let mut dk_acc: Option<KtTensor> = None;
    let mut dv_acc: Option<KtTensor> = None;
    let mut tile_start = 0usize;

    while tile_start < sq {
        let remaining = sq - tile_start;
        let tile_len = remaining.min(planned_tile).max(1);
        let tile_end = tile_start + tile_len;

        let q_tile_bf = map_kt(q3_bf.narrow(1, tile_start, tile_len))?;
        let q_tile = rocm_cast_to(&q_tile_bf, KtDType::F32)?;
        let do_tile_bf = map_kt(do3_bf.narrow(1, tile_start, tile_len))?;
        let do_tile = rocm_cast_to(&do_tile_bf, KtDType::F32)?;

        let p = {
            let qk = map_kt(kiln_tensor::rocm_matmul_to_dtype(
                &q_tile_bf,
                &kt3_bf,
                KtDType::F32,
            ))?; // [b*h, tile_len, sk]
            let scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, softmax_scale))?;
            drop(qk);
            let scores = if causal {
                causal_mask_fill_offset_f32(scores, bh, tile_len, sk, tile_start, causal_offset)?
            } else {
                scores
            };
            map_kt(kiln_tensor::rocm_softmax_last_axis(&scores))?
        };

        let dv_tile = {
            let pt = rocm_contig(&map_kt(p.transpose(1, 2))?)?; // [b*h, sk, tile_len]
            let dv = rocm_matmul_split_inner_f32(&pt, &do_tile, "bwd tiled dv")?; // [b*h, sk, d]
            drop(pt);
            dv
        };
        dv_acc = Some(match dv_acc.take() {
            Some(acc) => map_kt(acc.add(&dv_tile))?,
            None => dv_tile,
        });

        let dp = map_kt(kiln_tensor::rocm_matmul(&do_tile, &vt3))?; // [b*h, tile_len, sk]
        let dpp = elementwise_mul_f32(&dp, &p, device)?;
        let rowsum = map_kt(kiln_tensor::rocm_sum_axis(&dpp, 2))?; // [b*h, tile_len]
        drop(dpp);
        let dp_minus = row_broadcast_sub_last_axis_f32(&dp, &rowsum, sk)?;
        drop(rowsum);
        drop(dp);
        let ds = elementwise_mul_f32(&p, &dp_minus, device)?;
        drop(p);
        drop(dp_minus);

        let dq_tile = rocm_matmul_split_inner_f32(&ds, &k3, "bwd tiled dq")?; // [b*h, tile_len, d]
        let dq_tile = map_kt(kiln_tensor::rocm_scalar_op(
            &dq_tile,
            SCALAR_MUL,
            softmax_scale,
        ))?;
        dq_tiles.push(bhsd3_to_bshd_bf16(&dq_tile, b, h, tile_len, d)?);

        let dk_tile = {
            let dst = rocm_contig(&map_kt(ds.transpose(1, 2))?)?; // [b*h, sk, tile_len]
            let dk = rocm_matmul_split_inner_f32(&dst, &q_tile, "bwd tiled dk")?; // [b*h, sk, d]
            drop(dst);
            map_kt(kiln_tensor::rocm_scalar_op(&dk, SCALAR_MUL, softmax_scale))?
        };
        drop(ds);
        dk_acc = Some(match dk_acc.take() {
            Some(acc) => map_kt(acc.add(&dk_tile))?,
            None => dk_tile,
        });

        tile_start = tile_end;
    }

    let dq = if dq_tiles.len() == 1 {
        dq_tiles
            .pop()
            .ok_or_else(|| FlashAttnError::Msg("rocm-sdpa: missing dq tile".to_string()))?
    } else {
        let refs: Vec<&KtTensor> = dq_tiles.iter().collect();
        map_kt(KtTensor::cat(&refs, 1))?
    };
    let dk_acc = dk_acc
        .ok_or_else(|| FlashAttnError::Msg("rocm-sdpa: missing dk accumulator".to_string()))?;
    let dv_acc = dv_acc
        .ok_or_else(|| FlashAttnError::Msg("rocm-sdpa: missing dv accumulator".to_string()))?;
    let dk = bhsd3_to_bshd_bf16(&dk_acc, b, h, sk, d)?;
    let dv = bhsd3_to_bshd_bf16(&dv_acc, b, h, sk, d)?;
    Ok((dq, dk, dv))
}

#[allow(clippy::too_many_arguments)]
fn online_bwd_tiled_rocm(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    out: &KtTensor,
    softmax_lse: &KtTensor,
    softmax_scale: f32,
    causal: bool,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
) -> Result<(KtTensor, KtTensor, KtTensor), FlashAttnError> {
    if sq == 0 || sk == 0 {
        return materialized_bwd_composite_rocm(
            dout,
            q,
            k,
            v,
            softmax_scale,
            causal,
            b,
            sq,
            sk,
            h,
            hk,
            d,
        );
    }

    if softmax_lse.dtype() != KtDType::F32 || softmax_lse.shape() != [b, h, sq] {
        return Err(FlashAttnError::Msg(format!(
            "online_bwd_tiled_rocm: expected lse f32 [{b}, {h}, {sq}], got {:?} {}",
            softmax_lse.shape(),
            softmax_lse.dtype()
        )));
    }

    let device = q.device();
    let bh = b
        .checked_mul(h)
        .ok_or_else(|| FlashAttnError::Msg("ROCm online backward batch/head overflow".into()))?;
    let causal_offset = sk as isize - sq as isize;

    let fixed_working_set_bytes = online_backward_fixed_working_set_bytes(bh, sq, sk, d)
        .ok_or_else(|| FlashAttnError::Msg("ROCm online backward working-set overflow".into()))?;
    let online_tile_sizing = OnlineTileSizing::for_operation(
        device,
        bh,
        sq,
        sk,
        d,
        fixed_working_set_bytes,
        OnlinePass::Backward,
    )?;
    let (planned_q_tile, planned_k_tile) = online_tile_lens(online_tile_sizing, bh, sq, sk);
    let tile_phase_peaks = online_backward_tile_phase_peaks(bh, planned_q_tile, planned_k_tile, d)
        .ok_or_else(|| {
            FlashAttnError::Msg("ROCm online backward tile phase estimate overflow".into())
        })?;
    let operation_peak_bytes = fixed_working_set_bytes
        .checked_add(tile_phase_peaks.peak_bytes())
        .ok_or_else(|| {
            FlashAttnError::Msg("ROCm online backward operation peak estimate overflow".into())
        })?;
    let _memory_reservation =
        reserve_published_rocm_bytes(operation_peak_bytes, "online backward")?;

    let k_exp = gqa_expand_heads(k, h, hk, device)?;
    let v_exp = gqa_expand_heads(v, h, hk, device)?;

    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?;
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?;
    let do_bhsd = rocm_contig(&map_kt(dout.transpose(1, 2))?)?;
    let out_bhsd = rocm_contig(&map_kt(out.transpose(1, 2))?)?;

    let q3_bf = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?;
    let k3_bf = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?;
    let v3_bf = map_kt(v_bhsd.reshape(vec![bh, sk, d]))?;
    let do3_bf = map_kt(do_bhsd.reshape(vec![bh, sq, d]))?;
    let out3_bf = map_kt(out_bhsd.reshape(vec![bh, sq, d]))?;
    let lse2 = rocm_contig(&map_kt(softmax_lse.reshape(vec![bh, sq]))?)?;

    let dk_acc = rocm_zeros_in_tensor_context(&q3_bf, KtDType::F32, vec![bh, sk, d])
        .map_err(|error| FlashAttnError::Msg(format!("online backward dk accumulator: {error}")))?;
    let dv_acc = rocm_zeros_in_tensor_context(&q3_bf, KtDType::F32, vec![bh, sk, d])
        .map_err(|error| FlashAttnError::Msg(format!("online backward dv accumulator: {error}")))?;
    let mut dq_tiles = Vec::new();

    let mut q_start = 0usize;
    while q_start < sq {
        let remaining_q = sq - q_start;
        let max_k = if causal {
            causal_block_limit(sk, q_start, remaining_q, causal_offset).max(1)
        } else {
            sk
        };
        let (q_tile_cap, key_tile_cap) =
            online_tile_lens(online_tile_sizing, bh, remaining_q, max_k);
        let q_len = remaining_q.min(q_tile_cap).max(1);

        let q_tile_bf = rocm_contig(&map_kt(q3_bf.narrow(1, q_start, q_len))?)?;
        let do_tile_bf = rocm_contig(&map_kt(do3_bf.narrow(1, q_start, q_len))?)?;
        let do_tile = rocm_cast_to(&do_tile_bf, KtDType::F32)?;
        let out_tile_bf = map_kt(out3_bf.narrow(1, q_start, q_len))?;
        let out_tile = rocm_cast_to(&out_tile_bf, KtDType::F32)?;
        let d_rows = {
            let dot_terms = elementwise_mul_f32(&do_tile, &out_tile, device)?;
            let d_rows = map_kt(kiln_tensor::rocm_sum_axis(&dot_terms, 2))?;
            drop(dot_terms);
            d_rows
        };
        drop(do_tile);
        drop(out_tile);
        let dq_acc = rocm_zeros_in_tensor_context(&q3_bf, KtDType::F32, vec![bh, q_len, d])
            .map_err(|error| {
                FlashAttnError::Msg(format!(
                    "online backward dq accumulator at q_start={q_start}: {error}"
                ))
            })?;

        let block_limit = if causal {
            causal_block_limit(sk, q_start, q_len, causal_offset)
        } else {
            sk
        };
        let mut k_start = 0usize;
        while k_start < block_limit {
            let k_len = tile_len_without_unit_tail(block_limit - k_start, key_tile_cap);
            let k_block_bf = rocm_contig(&map_kt(k3_bf.narrow(1, k_start, k_len))?)?;
            let v_block_bf = rocm_contig(&map_kt(v3_bf.narrow(1, k_start, k_len))?)?;

            let scores = map_kt(kiln_tensor::rocm_matmul_rhs_transposed_to_dtype(
                &q_tile_bf,
                &k_block_bf,
                KtDType::F32,
            ))?;
            let scores = scale_mask_scores_f32(
                scores,
                bh,
                q_len,
                k_len,
                q_start,
                k_start,
                causal_offset,
                softmax_scale,
                causal,
            )?;
            let probs = prob_from_lse_scores_f32(
                scores,
                &lse2,
                bh,
                q_len,
                k_len,
                sq,
                q_start,
                k_start,
                causal_offset,
                causal,
            )?;

            let probs_bf = rocm_cast_to(&probs, KtDType::BF16)?;
            let dv_part = map_kt(kiln_tensor::rocm_matmul_lhs_transposed_to_dtype(
                &probs_bf,
                &do_tile_bf,
                KtDType::F32,
            ))?;
            accum_axis1_f32(&dv_acc, &dv_part, bh, sk, k_len, d, k_start)?;
            drop(dv_part);

            let dp = map_kt(kiln_tensor::rocm_matmul_rhs_transposed_to_dtype(
                &do_tile_bf,
                &v_block_bf,
                KtDType::F32,
            ))?;
            let ds = softmax_bwd_scores_f32(
                dp,
                &probs,
                &d_rows,
                bh,
                q_len,
                k_len,
                sq,
                q_start,
                k_start,
                causal_offset,
                softmax_scale,
                causal,
            )?;
            drop(probs);
            drop(probs_bf);

            let ds_bf = rocm_cast_to(&ds, KtDType::BF16)?;
            let dq_part = map_kt(kiln_tensor::rocm_matmul_to_dtype(
                &ds_bf,
                &k_block_bf,
                KtDType::F32,
            ))?;
            accum_axis1_f32(&dq_acc, &dq_part, bh, q_len, q_len, d, 0)?;
            drop(dq_part);
            drop(k_block_bf);

            let dk_part = map_kt(kiln_tensor::rocm_matmul_lhs_transposed_to_dtype(
                &ds_bf,
                &q_tile_bf,
                KtDType::F32,
            ))?;
            drop(ds_bf);
            drop(ds);
            accum_axis1_f32(&dk_acc, &dk_part, bh, sk, k_len, d, k_start)?;
            drop(dk_part);
            drop(v_block_bf);

            k_start += k_len;
        }

        dq_tiles.push(bhsd3_to_bshd_bf16(&dq_acc, b, h, q_len, d)?);
        q_start += q_len;
    }

    let dq = if dq_tiles.len() == 1 {
        dq_tiles
            .pop()
            .ok_or_else(|| FlashAttnError::Msg("rocm-sdpa: missing dq tile".to_string()))?
    } else {
        let refs: Vec<&KtTensor> = dq_tiles.iter().collect();
        map_kt(KtTensor::cat(&refs, 1))?
    };
    let dk = bhsd3_to_bshd_bf16(&dk_acc, b, h, sk, d)?;
    let dv = bhsd3_to_bshd_bf16(&dv_acc, b, h, sk, d)?;
    Ok((dq, dk, dv))
}

#[allow(clippy::too_many_arguments)]
fn try_native_bwd_bf16(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    out: &KtTensor,
    softmax_lse: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<(KtTensor, KtTensor, KtTensor)>, FlashAttnError> {
    if dout.dtype() != KtDType::BF16
        || q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || out.dtype() != KtDType::BF16
        || softmax_lse.dtype() != KtDType::F32
        || !matches!(d, 128 | 256)
        || h == 0
        || hk == 0
        || h % hk != 0
    {
        return Ok(None);
    }

    let q_c = rocm_contig(q)?;
    let k_c = rocm_contig(k)?;
    let v_c = rocm_contig(v)?;
    let out_c = rocm_contig(out)?;
    let dout_c = rocm_contig(dout)?;
    let lse_c = rocm_contig(softmax_lse)?;

    let (q_st, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(&q_c, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let dq = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, sq, h, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let dk = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, sk, h, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let dv = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, sk, h, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

    let dout_ptr = kiln_kt_bridge::rocm_input_device_ptr(&dout_c, KtDType::BF16, "dout")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let q_ptr = kiln_kt_bridge::rocm_input_device_ptr(&q_c, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let k_ptr = kiln_kt_bridge::rocm_input_device_ptr(&k_c, KtDType::BF16, "k")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let v_ptr = kiln_kt_bridge::rocm_input_device_ptr(&v_c, KtDType::BF16, "v")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_input_device_ptr(&out_c, KtDType::BF16, "out")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let lse_ptr = kiln_kt_bridge::rocm_input_device_ptr(&lse_c, KtDType::F32, "softmax_lse")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let dq_ptr = kiln_kt_bridge::rocm_output_device_ptr(&dq);
    let dk_ptr = kiln_kt_bridge::rocm_output_device_ptr(&dk);
    let dv_ptr = kiln_kt_bridge::rocm_output_device_ptr(&dv);
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&q_c, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let ffi_policy = rocm_flash_attention_ffi_policy(rocm_flash_attention_policy(&q_c)?)?;

    let status = unsafe {
        crate::kiln_rocm_flash_attn_bwd_bf16(
            dout_ptr as *const _,
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *const _,
            lse_ptr as *const _,
            dq_ptr as *mut _,
            dk_ptr as *mut _,
            dv_ptr as *mut _,
            b as i32,
            sq as i32,
            sk as i32,
            h as i32,
            hk as i32,
            d as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            &ffi_policy,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa native backward")?;

    if status == 0 {
        Ok(Some((dq, dk, dv)))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn try_native_bwd_bf16_collapsed_gqa(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    out: &KtTensor,
    softmax_lse: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<(KtTensor, KtTensor, KtTensor)>, FlashAttnError> {
    if dout.dtype() != KtDType::BF16
        || q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || out.dtype() != KtDType::BF16
        || softmax_lse.dtype() != KtDType::F32
        || !matches!(d, 128 | 256)
        || h == 0
        || hk == 0
        || h % hk != 0
    {
        return Ok(None);
    }

    let q_c = rocm_contig(q)?;
    let k_c = rocm_contig(k)?;
    let v_c = rocm_contig(v)?;
    let out_c = rocm_contig(out)?;
    let dout_c = rocm_contig(dout)?;
    let lse_c = rocm_contig(softmax_lse)?;

    let (q_st, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(&q_c, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let dq = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, sq, h, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let dk = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, sk, hk, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let dv = kiln_kt_bridge::alloc_rocm_tensor(q_st, KtDType::BF16, vec![b, sk, hk, d])
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

    let dout_ptr = kiln_kt_bridge::rocm_input_device_ptr(&dout_c, KtDType::BF16, "dout")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let q_ptr = kiln_kt_bridge::rocm_input_device_ptr(&q_c, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let k_ptr = kiln_kt_bridge::rocm_input_device_ptr(&k_c, KtDType::BF16, "k")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let v_ptr = kiln_kt_bridge::rocm_input_device_ptr(&v_c, KtDType::BF16, "v")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_input_device_ptr(&out_c, KtDType::BF16, "out")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let lse_ptr = kiln_kt_bridge::rocm_input_device_ptr(&lse_c, KtDType::F32, "softmax_lse")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let dq_ptr = kiln_kt_bridge::rocm_output_device_ptr(&dq);
    let dk_ptr = kiln_kt_bridge::rocm_output_device_ptr(&dk);
    let dv_ptr = kiln_kt_bridge::rocm_output_device_ptr(&dv);
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&q_c, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let ffi_policy = rocm_flash_attention_ffi_policy(rocm_flash_attention_policy(&q_c)?)?;

    let status = unsafe {
        crate::kiln_rocm_flash_attn_bwd_collapsed_gqa_bf16(
            dout_ptr as *const _,
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *const _,
            lse_ptr as *const _,
            dq_ptr as *mut _,
            dk_ptr as *mut _,
            dv_ptr as *mut _,
            b as i32,
            sq as i32,
            sk as i32,
            h as i32,
            hk as i32,
            d as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            &ffi_policy,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa native backward")?;

    if status == 0 {
        Ok(Some((dq, dk, dv)))
    } else {
        Ok(None)
    }
}

/// Reshape a `[b*h, s, d]` f32 tensor to `[b, s, h, d]` BF16 (transpose the
/// head axis back out and narrow to BF16). On-device.
fn bhsd3_to_bshd_bf16(
    t3: &KtTensor, // [b*h, s, d] f32
    b: usize,
    h: usize,
    s: usize,
    d: usize,
) -> Result<KtTensor, FlashAttnError> {
    let t_bhsd = map_kt(t3.reshape(vec![b, h, s, d]))?; // [b, h, s, d]
    let t_bshd = rocm_contig(&map_kt(t_bhsd.transpose(1, 2))?)?; // [b, s, h, d]
    rocm_cast_to(&t_bshd, KtDType::BF16)
}

/// Collapse an expanded `[b*h, s, d]` F32 GQA gradient to `[b, s, hk, d]` BF16.
/// The input head order is `[kv0 group0..groupN, kv1 group0..groupN, ...]`,
/// matching [`gqa_expand_heads`].
fn collapse_bhsd3_gqa_grad_to_bshd_bf16(
    t3: &KtTensor,
    b: usize,
    h: usize,
    hk: usize,
    s: usize,
    d: usize,
) -> Result<KtTensor, FlashAttnError> {
    if hk == h {
        return bhsd3_to_bshd_bf16(t3, b, h, s, d);
    }
    if hk == 0 || h % hk != 0 {
        return Err(FlashAttnError::Msg(format!(
            "collapse_bhsd3_gqa_grad_to_bshd_bf16: invalid GQA heads h={h} hk={hk}"
        )));
    }
    if t3.dtype() != KtDType::F32 || t3.shape() != [b * h, s, d] {
        return Err(FlashAttnError::Msg(format!(
            "collapse_bhsd3_gqa_grad_to_bshd_bf16: expected f32 [{}, {s}, {d}], got {:?} {}",
            b * h,
            t3.shape(),
            t3.dtype()
        )));
    }

    let groups = h / hk;
    let grouped = map_kt(t3.reshape(vec![b, hk, groups, s, d]))?;
    let grouped = rocm_contig(&grouped)?;
    let summed = map_kt(kiln_tensor::rocm_sum_axis(&grouped, 2))?; // [b,hk,s,d]
    let summed3 = map_kt(summed.reshape(vec![b * hk, s, d]))?;
    bhsd3_to_bshd_bf16(&summed3, b, hk, s, d)
}

/// Collapse an expanded `[b, s, h, d]` BF16 GQA gradient to `[b, s, hk, d]`
/// BF16, preserving the existing F32 reduction semantics.
fn collapse_expanded_bshd_gqa_grad_bf16(
    expanded: &KtTensor,
    b: usize,
    s: usize,
    h: usize,
    hk: usize,
    d: usize,
) -> Result<KtTensor, FlashAttnError> {
    if hk == h {
        return Ok(expanded.clone());
    }
    if hk == 0 || h % hk != 0 {
        return Err(FlashAttnError::Msg(format!(
            "collapse_expanded_bshd_gqa_grad_bf16: invalid GQA heads h={h} hk={hk}"
        )));
    }
    if expanded.shape() != [b, s, h, d] {
        return Err(FlashAttnError::Msg(format!(
            "collapse_expanded_bshd_gqa_grad_bf16: expected [{b}, {s}, {h}, {d}], got {:?}",
            expanded.shape()
        )));
    }

    if expanded.dtype() == KtDType::BF16 {
        let expanded = rocm_contig(expanded)?;
        let (storage, _) =
            kiln_kt_bridge::rocm_storage_and_byte_offset(&expanded, KtDType::BF16, "expanded")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let collapsed =
            kiln_kt_bridge::alloc_rocm_tensor(storage, KtDType::BF16, vec![b, s, hk, d])
                .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let expanded_ptr =
            kiln_kt_bridge::rocm_input_device_ptr(&expanded, KtDType::BF16, "expanded")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let collapsed_ptr = kiln_kt_bridge::rocm_output_device_ptr(&collapsed);
        let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&expanded, "expanded")
            .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
        let stream = stream_submission.raw_stream();
        let status = unsafe {
            crate::kiln_rocm_flash_collapse_gqa_bf16(
                expanded_ptr as *const _,
                collapsed_ptr as *mut _,
                b as i32,
                s as i32,
                h as i32,
                hk as i32,
                d as i32,
                stream,
            )
        };
        settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa GQA collapse")?;
        if status == 0 {
            return Ok(collapsed);
        }
    }

    let groups = h / hk;
    let expanded_f32 = rocm_cast_to(expanded, KtDType::F32)?;
    let grouped = map_kt(expanded_f32.reshape(vec![b, s, hk, groups, d]))?;
    let grouped = rocm_contig(&grouped)?;
    let summed = map_kt(kiln_tensor::rocm_sum_axis(&grouped, 3))?; // [b,s,hk,d]
    rocm_cast_to(&summed, KtDType::BF16)
}

fn causal_mask_fill_offset_f32(
    scores: KtTensor,
    bh: usize,
    sq: usize,
    sk: usize,
    q_start: usize,
    causal_offset: isize,
) -> Result<KtTensor, FlashAttnError> {
    if scores.dtype() != KtDType::F32 || scores.shape() != [bh, sq, sk] {
        return Err(FlashAttnError::Msg(format!(
            "causal_mask_fill_offset_f32: expected f32 [{bh}, {sq}, {sk}], got {:?} {}",
            scores.shape(),
            scores.dtype()
        )));
    }
    let scores = rocm_contig(&scores)?;
    let scores_ptr = kiln_kt_bridge::rocm_output_device_ptr(&scores);
    let bh_i32 = i32::try_from(bh)
        .map_err(|_| FlashAttnError::Msg(format!("causal mask bh too large: {bh}")))?;
    let sq_i32 = i32::try_from(sq)
        .map_err(|_| FlashAttnError::Msg(format!("causal mask sq too large: {sq}")))?;
    let sk_i32 = i32::try_from(sk)
        .map_err(|_| FlashAttnError::Msg(format!("causal mask sk too large: {sk}")))?;
    let q_start_i32 = i32::try_from(q_start)
        .map_err(|_| FlashAttnError::Msg(format!("causal mask q_start too large: {q_start}")))?;
    let causal_offset_i32 = i32::try_from(causal_offset).map_err(|_| {
        FlashAttnError::Msg(format!("causal mask offset too large: {causal_offset}"))
    })?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&scores, "scores")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_causal_mask_fill_offset_f32(
            scores_ptr as *mut _,
            bh_i32,
            sq_i32,
            sk_i32,
            q_start_i32,
            causal_offset_i32,
            NEG_FILL,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa causal mask")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "causal_mask_fill_offset_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(&scores)?;
    Ok(scores)
}

#[allow(clippy::too_many_arguments)]
fn scale_mask_scores_f32(
    scores: KtTensor,
    bh: usize,
    sq: usize,
    sk: usize,
    q_start: usize,
    k_start: usize,
    causal_offset: isize,
    scale: f32,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    if scores.dtype() != KtDType::F32 || scores.shape() != [bh, sq, sk] {
        return Err(FlashAttnError::Msg(format!(
            "scale_mask_scores_f32: expected f32 [{bh}, {sq}, {sk}], got {:?} {}",
            scores.shape(),
            scores.dtype()
        )));
    }
    let scores = rocm_contig(&scores)?;
    let scores_ptr = kiln_kt_bridge::rocm_output_device_ptr(&scores);
    let bh = usize_to_i32("scale mask bh", bh)?;
    let sq = usize_to_i32("scale mask sq", sq)?;
    let sk = usize_to_i32("scale mask sk", sk)?;
    let q_start = usize_to_i32("scale mask q_start", q_start)?;
    let k_start = usize_to_i32("scale mask k_start", k_start)?;
    let causal_offset = isize_to_i32("scale mask causal_offset", causal_offset)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&scores, "scores")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_scale_mask_f32(
            scores_ptr as *mut _,
            bh,
            sq,
            sk,
            q_start,
            k_start,
            causal_offset,
            scale,
            NEG_FILL,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa scale mask")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "scale_mask_scores_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(&scores)?;
    Ok(scores)
}

#[allow(clippy::too_many_arguments)]
fn exp_mask_scores_f32(
    scores: KtTensor,
    row_max: &KtTensor,
    bh: usize,
    sq: usize,
    sk: usize,
    q_start: usize,
    k_start: usize,
    causal_offset: isize,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    if scores.dtype() != KtDType::F32 || row_max.dtype() != KtDType::F32 {
        return Err(FlashAttnError::Msg(format!(
            "exp_mask_scores_f32: expected f32 tensors, got scores={} row_max={}",
            scores.dtype(),
            row_max.dtype()
        )));
    }
    if scores.shape() != [bh, sq, sk] || row_max.shape() != [bh, sq] {
        return Err(FlashAttnError::Msg(format!(
            "exp_mask_scores_f32: shape mismatch scores={:?} row_max={:?} expected [{bh}, {sq}, {sk}] / [{bh}, {sq}]",
            scores.shape(),
            row_max.shape()
        )));
    }
    let scores = rocm_contig(&scores)?;
    let row_max = rocm_contig(row_max)?;
    let scores_ptr = kiln_kt_bridge::rocm_output_device_ptr(&scores);
    let row_max_ptr = kiln_kt_bridge::rocm_input_device_ptr(&row_max, KtDType::F32, "row_max")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let bh = usize_to_i32("exp mask bh", bh)?;
    let sq = usize_to_i32("exp mask sq", sq)?;
    let sk = usize_to_i32("exp mask sk", sk)?;
    let q_start = usize_to_i32("exp mask q_start", q_start)?;
    let k_start = usize_to_i32("exp mask k_start", k_start)?;
    let causal_offset = isize_to_i32("exp mask causal_offset", causal_offset)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&scores, "scores")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_exp_mask_f32(
            scores_ptr as *mut _,
            row_max_ptr as *const _,
            bh,
            sq,
            sk,
            q_start,
            k_start,
            causal_offset,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa exp mask")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "exp_mask_scores_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(&scores)?;
    Ok(scores)
}

#[allow(clippy::too_many_arguments)]
fn prob_from_lse_scores_f32(
    scores: KtTensor,
    lse: &KtTensor,
    bh: usize,
    sq: usize,
    sk: usize,
    total_q: usize,
    q_start: usize,
    k_start: usize,
    causal_offset: isize,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    if scores.dtype() != KtDType::F32 || lse.dtype() != KtDType::F32 {
        return Err(FlashAttnError::Msg(format!(
            "prob_from_lse_scores_f32: expected f32 tensors, got scores={} lse={}",
            scores.dtype(),
            lse.dtype()
        )));
    }
    if scores.shape() != [bh, sq, sk] || lse.shape() != [bh, total_q] {
        return Err(FlashAttnError::Msg(format!(
            "prob_from_lse_scores_f32: shape mismatch scores={:?} lse={:?} expected [{bh}, {sq}, {sk}] / [{bh}, {total_q}]",
            scores.shape(),
            lse.shape()
        )));
    }
    let scores = rocm_contig(&scores)?;
    let lse = rocm_contig(lse)?;
    let scores_ptr = kiln_kt_bridge::rocm_output_device_ptr(&scores);
    let lse_ptr = kiln_kt_bridge::rocm_input_device_ptr(&lse, KtDType::F32, "lse")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let bh = usize_to_i32("prob lse bh", bh)?;
    let sq = usize_to_i32("prob lse sq", sq)?;
    let sk = usize_to_i32("prob lse sk", sk)?;
    let total_q = usize_to_i32("prob lse total_q", total_q)?;
    let q_start = usize_to_i32("prob lse q_start", q_start)?;
    let k_start = usize_to_i32("prob lse k_start", k_start)?;
    let causal_offset = isize_to_i32("prob lse causal_offset", causal_offset)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&scores, "scores")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_prob_from_lse_f32(
            scores_ptr as *mut _,
            lse_ptr as *const _,
            bh,
            sq,
            sk,
            total_q,
            q_start,
            k_start,
            causal_offset,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    settle_rocm_ffi_submission(
        stream_submission,
        status,
        "rocm-sdpa probability reconstruction",
    )?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "prob_from_lse_scores_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(&scores)?;
    Ok(scores)
}

#[allow(clippy::too_many_arguments)]
fn softmax_bwd_scores_f32(
    dp: KtTensor,
    p: &KtTensor,
    d_rows: &KtTensor,
    bh: usize,
    sq: usize,
    sk: usize,
    total_q: usize,
    q_start: usize,
    k_start: usize,
    causal_offset: isize,
    scale: f32,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    if dp.dtype() != KtDType::F32 || p.dtype() != KtDType::F32 || d_rows.dtype() != KtDType::F32 {
        return Err(FlashAttnError::Msg(format!(
            "softmax_bwd_scores_f32: expected f32 tensors, got dp={} p={} d_rows={}",
            dp.dtype(),
            p.dtype(),
            d_rows.dtype()
        )));
    }
    if dp.shape() != [bh, sq, sk] || p.shape() != [bh, sq, sk] || d_rows.shape() != [bh, sq] {
        return Err(FlashAttnError::Msg(format!(
            "softmax_bwd_scores_f32: shape mismatch dp={:?} p={:?} d_rows={:?} expected [{bh}, {sq}, {sk}] / [{bh}, {sq}]",
            dp.shape(),
            p.shape(),
            d_rows.shape()
        )));
    }
    let dp = rocm_contig(&dp)?;
    let p = rocm_contig(p)?;
    let d_rows = rocm_contig(d_rows)?;
    let dp_ptr = kiln_kt_bridge::rocm_output_device_ptr(&dp);
    let p_ptr = kiln_kt_bridge::rocm_input_device_ptr(&p, KtDType::F32, "p")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let d_rows_ptr = kiln_kt_bridge::rocm_input_device_ptr(&d_rows, KtDType::F32, "d_rows")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let bh = usize_to_i32("softmax bwd bh", bh)?;
    let sq = usize_to_i32("softmax bwd sq", sq)?;
    let sk = usize_to_i32("softmax bwd sk", sk)?;
    let total_q = usize_to_i32("softmax bwd total_q", total_q)?;
    let q_start = usize_to_i32("softmax bwd q_start", q_start)?;
    let k_start = usize_to_i32("softmax bwd k_start", k_start)?;
    let causal_offset = isize_to_i32("softmax bwd causal_offset", causal_offset)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&dp, "dp")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_softmax_bwd_f32(
            dp_ptr as *mut _,
            p_ptr as *const _,
            d_rows_ptr as *const _,
            bh,
            sq,
            sk,
            total_q,
            q_start,
            k_start,
            causal_offset,
            scale,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa softmax backward")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "softmax_bwd_scores_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(&dp)?;
    Ok(dp)
}

fn accum_axis1_f32(
    dst: &KtTensor,
    src: &KtTensor,
    bh: usize,
    total_s: usize,
    tile_s: usize,
    d: usize,
    s_start: usize,
) -> Result<(), FlashAttnError> {
    if dst.dtype() != KtDType::F32 || src.dtype() != KtDType::F32 {
        return Err(FlashAttnError::Msg(format!(
            "accum_axis1_f32: expected f32 tensors, got dst={} src={}",
            dst.dtype(),
            src.dtype()
        )));
    }
    if dst.shape() != [bh, total_s, d] || src.shape() != [bh, tile_s, d] {
        return Err(FlashAttnError::Msg(format!(
            "accum_axis1_f32: shape mismatch dst={:?} src={:?} expected [{bh}, {total_s}, {d}] / [{bh}, {tile_s}, {d}]",
            dst.shape(),
            src.shape()
        )));
    }
    if !dst.is_contiguous() || !src.is_contiguous() {
        return Err(FlashAttnError::Msg(format!(
            "accum_axis1_f32: tensors must be contiguous, got dst={} src={}",
            dst.is_contiguous(),
            src.is_contiguous()
        )));
    }
    let dst_ptr = kiln_kt_bridge::rocm_output_device_ptr(dst);
    let src_ptr = kiln_kt_bridge::rocm_input_device_ptr(src, KtDType::F32, "src")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let bh = usize_to_i32("accum bh", bh)?;
    let total_s = usize_to_i32("accum total_s", total_s)?;
    let tile_s = usize_to_i32("accum tile_s", tile_s)?;
    let d = usize_to_i32("accum head_dim", d)?;
    let s_start = usize_to_i32("accum s_start", s_start)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(dst, "dst")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_accum_axis1_f32(
            dst_ptr as *mut _,
            src_ptr as *const _,
            bh,
            total_s,
            tile_s,
            d,
            s_start,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa axis accumulation")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "accum_axis1_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(dst)?;
    Ok(())
}

fn online_update_state_f32(
    row_m: &KtTensor,
    row_l: &KtTensor,
    block_m: &KtTensor,
    block_l: &KtTensor,
    alpha: &KtTensor,
    beta: &KtTensor,
    rows: usize,
) -> Result<(), FlashAttnError> {
    for (name, tensor) in [
        ("row_m", row_m),
        ("row_l", row_l),
        ("block_m", block_m),
        ("block_l", block_l),
        ("alpha", alpha),
        ("beta", beta),
    ] {
        if tensor.dtype() != KtDType::F32 || tensor.shape() != [rows] {
            return Err(FlashAttnError::Msg(format!(
                "online_update_state_f32: {name} expected f32 [{rows}], got {:?} {}",
                tensor.shape(),
                tensor.dtype()
            )));
        }
    }
    let row_m = rocm_contig(row_m)?;
    let row_l = rocm_contig(row_l)?;
    let block_m = rocm_contig(block_m)?;
    let block_l = rocm_contig(block_l)?;
    let alpha = rocm_contig(alpha)?;
    let beta = rocm_contig(beta)?;
    let row_m_ptr = kiln_kt_bridge::rocm_output_device_ptr(&row_m);
    let row_l_ptr = kiln_kt_bridge::rocm_output_device_ptr(&row_l);
    let block_m_ptr = kiln_kt_bridge::rocm_input_device_ptr(&block_m, KtDType::F32, "block_m")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let block_l_ptr = kiln_kt_bridge::rocm_input_device_ptr(&block_l, KtDType::F32, "block_l")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let alpha_ptr = kiln_kt_bridge::rocm_output_device_ptr(&alpha);
    let beta_ptr = kiln_kt_bridge::rocm_output_device_ptr(&beta);
    let rows = usize_to_i32("online state rows", rows)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&row_m, "row_m")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_online_update_state_f32(
            row_m_ptr as *mut _,
            row_l_ptr as *mut _,
            block_m_ptr as *const _,
            block_l_ptr as *const _,
            alpha_ptr as *mut _,
            beta_ptr as *mut _,
            rows,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa online state update")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "online_update_state_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(&row_m)?;
    Ok(())
}

fn online_update_acc_f32(
    acc: &KtTensor,
    block_out: &KtTensor,
    alpha: &KtTensor,
    beta: &KtTensor,
    rows: usize,
    d: usize,
) -> Result<(), FlashAttnError> {
    if acc.dtype() != KtDType::F32 || block_out.dtype() != KtDType::F32 {
        return Err(FlashAttnError::Msg(format!(
            "online_update_acc_f32: expected f32 acc/block_out, got {} / {}",
            acc.dtype(),
            block_out.dtype()
        )));
    }
    if acc.elem_count() != rows * d || block_out.elem_count() != rows * d {
        return Err(FlashAttnError::Msg(format!(
            "online_update_acc_f32: element count mismatch acc={:?} block_out={:?} rows={rows} d={d}",
            acc.shape(),
            block_out.shape()
        )));
    }
    if alpha.dtype() != KtDType::F32
        || beta.dtype() != KtDType::F32
        || alpha.shape() != [rows]
        || beta.shape() != [rows]
    {
        return Err(FlashAttnError::Msg(format!(
            "online_update_acc_f32: alpha/beta shape mismatch alpha={:?} beta={:?} rows={rows}",
            alpha.shape(),
            beta.shape()
        )));
    }
    let acc2 = rocm_contig(&map_kt(acc.reshape(vec![rows, d]))?)?;
    let block_out2 = rocm_contig(&map_kt(block_out.reshape(vec![rows, d]))?)?;
    let alpha = rocm_contig(alpha)?;
    let beta = rocm_contig(beta)?;
    let acc_ptr = kiln_kt_bridge::rocm_output_device_ptr(&acc2);
    let block_out_ptr =
        kiln_kt_bridge::rocm_input_device_ptr(&block_out2, KtDType::F32, "block_out")
            .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let alpha_ptr = kiln_kt_bridge::rocm_input_device_ptr(&alpha, KtDType::F32, "alpha")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let beta_ptr = kiln_kt_bridge::rocm_input_device_ptr(&beta, KtDType::F32, "beta")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let rows = usize_to_i32("online acc rows", rows)?;
    let d = usize_to_i32("online acc head_dim", d)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&acc2, "acc")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_online_update_acc_f32(
            acc_ptr as *mut _,
            block_out_ptr as *const _,
            alpha_ptr as *const _,
            beta_ptr as *const _,
            rows,
            d,
            stream,
        )
    };
    settle_rocm_ffi_submission(
        stream_submission,
        status,
        "rocm-sdpa online accumulator update",
    )?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "online_update_acc_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(&acc2)?;
    Ok(())
}

fn online_finalize_f32(
    acc: &KtTensor,
    row_m: &KtTensor,
    row_l: &KtTensor,
    out: &KtTensor,
    lse: &KtTensor,
    rows: usize,
    d: usize,
) -> Result<(), FlashAttnError> {
    for (name, tensor, expected) in [
        ("row_m", row_m, vec![rows]),
        ("row_l", row_l, vec![rows]),
        ("lse", lse, vec![rows]),
    ] {
        if tensor.dtype() != KtDType::F32 || tensor.shape() != expected.as_slice() {
            return Err(FlashAttnError::Msg(format!(
                "online_finalize_f32: {name} expected f32 {:?}, got {:?} {}",
                expected,
                tensor.shape(),
                tensor.dtype()
            )));
        }
    }
    let acc2 = rocm_contig(&map_kt(acc.reshape(vec![rows, d]))?)?;
    let out2 = rocm_contig(&map_kt(out.reshape(vec![rows, d]))?)?;
    let row_m = rocm_contig(row_m)?;
    let row_l = rocm_contig(row_l)?;
    let lse = rocm_contig(lse)?;
    let acc_ptr = kiln_kt_bridge::rocm_input_device_ptr(&acc2, KtDType::F32, "acc")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let row_m_ptr = kiln_kt_bridge::rocm_input_device_ptr(&row_m, KtDType::F32, "row_m")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let row_l_ptr = kiln_kt_bridge::rocm_input_device_ptr(&row_l, KtDType::F32, "row_l")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_output_device_ptr(&out2);
    let lse_ptr = kiln_kt_bridge::rocm_output_device_ptr(&lse);
    let rows = usize_to_i32("online finalize rows", rows)?;
    let d = usize_to_i32("online finalize head_dim", d)?;
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&acc2, "acc")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_online_finalize_f32(
            acc_ptr as *const _,
            row_m_ptr as *const _,
            row_l_ptr as *const _,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            rows,
            d,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa online finalize")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "online_finalize_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_tensor(&out2)?;
    Ok(())
}

/// Fused `out[bh, sq, sk] = a[bh, sq, sk] - rowsum[bh, sq]` on ROCm. This
/// avoids materializing the `[bh, sq, sk]` broadcast tensor during exact
/// attention backward.
fn row_broadcast_sub_last_axis_f32(
    a: &KtTensor,
    rowsum: &KtTensor,
    sk: usize,
) -> Result<KtTensor, FlashAttnError> {
    if a.dtype() != KtDType::F32 || rowsum.dtype() != KtDType::F32 {
        return Err(FlashAttnError::Msg(format!(
            "row_broadcast_sub_last_axis_f32: expected f32 tensors, got a={} rowsum={}",
            a.dtype(),
            rowsum.dtype()
        )));
    }
    if a.shape().len() != 3 || rowsum.shape().len() != 2 {
        return Err(FlashAttnError::Msg(format!(
            "row_broadcast_sub_last_axis_f32: expected ranks [3,2], got a={:?} rowsum={:?}",
            a.shape(),
            rowsum.shape()
        )));
    }
    let bh = a.shape()[0];
    let sq = a.shape()[1];
    if a.shape()[2] != sk || rowsum.shape() != [bh, sq] {
        return Err(FlashAttnError::Msg(format!(
            "row_broadcast_sub_last_axis_f32: shape mismatch a={:?} rowsum={:?} sk={sk}",
            a.shape(),
            rowsum.shape()
        )));
    }
    let a_c = rocm_contig(a)?;
    let rowsum_c = rocm_contig(rowsum)?;
    let (storage, _) = kiln_kt_bridge::rocm_storage_and_byte_offset(&a_c, KtDType::F32, "a")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out = kiln_kt_bridge::alloc_rocm_tensor(storage, KtDType::F32, a.shape().to_vec())
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let a_ptr = kiln_kt_bridge::rocm_input_device_ptr(&a_c, KtDType::F32, "a")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let rowsum_ptr = kiln_kt_bridge::rocm_input_device_ptr(&rowsum_c, KtDType::F32, "rowsum")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_output_device_ptr(&out);
    let stream_submission = kiln_kt_bridge::rocm_stream_submission_of(&a_c, "a")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = stream_submission.raw_stream();
    let status = unsafe {
        crate::kiln_rocm_flash_rowsum_sub_last_axis_f32(
            a_ptr as *const _,
            rowsum_ptr as *const _,
            out_ptr as *mut _,
            bh as i32,
            sq as i32,
            sk as i32,
            stream,
        )
    };
    settle_rocm_ffi_submission(stream_submission, status, "rocm-sdpa row broadcast")?;
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "row_broadcast_sub_last_axis_f32: FFI returned status {status}"
        )));
    }
    Ok(out)
}

/// On-device elementwise f32 multiply of two same-shape contiguous ROCm tensors.
fn elementwise_mul_f32(
    a: &KtTensor,
    b: &KtTensor,
    _device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    let a = rocm_contig(a)?;
    let b = rocm_contig(b)?;
    map_kt(kiln_tensor::rocm_elementwise_binary(&a, &b, 2))
}

// Keep the bf16 import used even if the optimizer prunes a path.
#[allow(dead_code)]
fn _bf16_marker(x: f32) -> bf16 {
    bf16::from_f32(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materialized_force_is_only_a_route_preference() {
        assert!(!materialized_bwd_route_enabled(Some(false), true, true));
        assert!(materialized_bwd_route_enabled(Some(true), false, true));
        assert!(!materialized_bwd_route_enabled(Some(true), true, false));
        assert!(materialized_bwd_route_enabled(None, true, true));
        assert!(!materialized_bwd_route_enabled(None, false, true));
        assert!(!materialized_bwd_route_enabled(None, true, false));
    }

    #[test]
    fn full_materialized_backward_accounts_fixed_peak_and_overflow() {
        let plan = full_materialized_bwd_memory_plan(1, 2, 256, 128, 64).unwrap();
        assert!(plan.fixed_peak_bytes > 0);
        assert!(plan.scratch_peak_bytes > 0);
        let error =
            validate_full_materialized_bwd_admission(plan.scratch_peak_bytes, 1, 2, 256, 128, 64)
                .expect_err("a score-only budget must not omit fixed tensors and outputs");
        assert!(error.to_string().contains("fixed_peak_bytes"));
        validate_full_materialized_bwd_admission(plan.peak_bytes, 1, 2, 256, 128, 64)
            .expect("the exact conservative peak must be admissible");

        let overflow = validate_full_materialized_bwd_admission(usize::MAX, usize::MAX, 2, 2, 2, 2)
            .expect_err("dimension overflow must fail closed");
        assert!(overflow.to_string().contains("estimate overflow"));
    }

    #[test]
    fn tiled_materialized_backward_subtracts_fixed_and_accumulator_peaks() {
        let (b, h, sq, sk, d) = (1, 2, 256, 128, 64);
        let bh = b * h;
        let fixed = materialized_bwd_query_tiled_fixed_peak_bytes(b, h, sq, sk, d).unwrap();
        let minimum_scratch = materialized_bwd_query_tile_scratch_bytes(
            bh,
            MATERIALIZED_SCORE_TILE_GRANULARITY,
            sk,
            d,
        )
        .unwrap();

        let error = plan_materialized_bwd_query_tiles_with_limit(
            minimum_scratch,
            b,
            h,
            sq,
            sk,
            d,
            usize::MAX,
        )
        .expect_err("score scratch alone must not omit fixed and accumulator/output peaks");
        assert!(error.to_string().contains("fixed_peak_bytes"));

        let plan = plan_materialized_bwd_query_tiles_with_limit(
            fixed + minimum_scratch,
            b,
            h,
            sq,
            sk,
            d,
            usize::MAX,
        )
        .expect("full available memory after fixed peaks sustains the minimum tile");
        assert_eq!(plan.query_tile, MATERIALIZED_SCORE_TILE_GRANULARITY);
        assert_eq!(plan.fixed_peak_bytes, fixed);
        assert_eq!(plan.tile_scratch_bytes, minimum_scratch);
        assert_eq!(plan.peak_bytes(), Some(fixed + minimum_scratch));
    }

    #[test]
    fn online_backward_phase_estimates_include_k_and_score_families() {
        let phases = online_backward_tile_phase_peaks(1, 2, 3, 4).unwrap();
        assert_eq!(phases.q_preparation_bytes, 168);
        assert_eq!(phases.score_bytes, 180);
        assert_eq!(phases.dv_bytes, 204);
        assert_eq!(phases.dq_bytes, 188);
        assert_eq!(phases.dk_bytes, 180);
        assert_eq!(phases.peak_bytes(), phases.dv_bytes);

        // Q=8 elements, K=12 elements, rows=2: 8Q + 16K + 4R.
        assert_eq!(
            online_backward_fixed_working_set_bytes(1, 2, 3, 4),
            Some(264)
        );
        let sizing = OnlineTileSizing {
            query_tile: 2,
            key_tile: 3,
            score_budget_bytes: online_tile_bytes(1, 2, 3).unwrap(),
        };
        validate_online_operation_plan(
            sizing,
            1,
            2,
            3,
            4,
            phases.peak_bytes(),
            OnlinePass::Backward,
        )
        .expect("the exact phase peak must be admissible");
        validate_online_operation_plan(
            sizing,
            1,
            2,
            3,
            4,
            phases.peak_bytes() - 1,
            OnlinePass::Backward,
        )
        .expect_err("omitting any phase bytes must fail closed");
        assert!(online_backward_tile_phase_peaks(usize::MAX, 2, 2, 2).is_none());
    }

    #[test]
    fn bounded_tile_partition_avoids_a_unit_tail_when_possible() {
        let mut remaining = 17;
        let mut tiles = Vec::new();
        while remaining > 0 {
            let tile = tile_len_without_unit_tail(remaining, 4);
            tiles.push(tile);
            remaining -= tile;
        }
        assert_eq!(tiles, [4, 4, 4, 3, 2]);
        assert_eq!(tile_len_without_unit_tail(1, 4), 1);
        assert_eq!(tile_len_without_unit_tail(2, 4), 2);
    }

    #[test]
    fn forward_score_plans_scale_with_batch_and_fail_closed_on_overflow() {
        let single = materialized_score_working_set_bytes(1, 8, 128, 512).unwrap();
        assert_eq!(
            materialized_score_working_set_bytes(4, 8, 128, 512),
            single.checked_mul(4),
            "materialized scores are [batch, heads, query, key]"
        );
        assert!(materialized_score_working_set_bytes(usize::MAX, 2, 2, 2).is_none());

        let sizing = OnlineTileSizing {
            query_tile: 2,
            key_tile: 3,
            score_budget_bytes: usize::MAX,
        };
        assert_eq!(
            online_forward_operation_peak_bytes(sizing, 1, 2, 3, 4, 100),
            Some(340),
            "the reservation owns fixed bytes plus score and non-score tile scratch"
        );
        assert!(online_forward_operation_peak_bytes(sizing, usize::MAX, 2, 3, 4, 100,).is_none());
    }

    #[test]
    fn head_major_online_peak_includes_live_conversions_and_overflow_fails_closed() {
        let (b, h, hk, sq, sk, d) = (2, 8, 2, 32, 64, 16);
        let bh = b * h;
        let inner_fixed = online_forward_fixed_working_set_bytes(bh, sq, sk, d).unwrap();
        let conversion_fixed =
            head_major_conversion_working_set_bytes(b, h, hk, sq, sk, d).unwrap();
        let head_major_fixed =
            head_major_online_forward_fixed_working_set_bytes(b, h, hk, sq, sk, d).unwrap();
        assert_eq!(
            head_major_fixed,
            inner_fixed.checked_add(conversion_fixed).unwrap(),
            "the pre-reserved fixed set must retain converted Q/K/V alongside inner buffers"
        );

        let sizing = OnlineTileSizing {
            query_tile: 2,
            key_tile: 3,
            score_budget_bytes: usize::MAX,
        };
        let inner_peak = OnlineForwardPlan {
            sizing,
            fixed_working_set_bytes: inner_fixed,
        }
        .operation_peak_bytes(bh, sq, sk, d)
        .unwrap();
        let reserved_head_major_peak = OnlineForwardPlan {
            sizing,
            fixed_working_set_bytes: head_major_fixed,
        }
        .operation_peak_bytes(bh, sq, sk, d)
        .unwrap();
        assert_eq!(
            reserved_head_major_peak,
            inner_peak.checked_add(conversion_fixed).unwrap(),
            "the outer reservation peak must include every live head-major conversion byte"
        );

        assert!(
            head_major_online_forward_fixed_working_set_bytes(usize::MAX, 2, hk, sq, sk, d,)
                .is_none(),
            "head-major dimension overflow must fail closed before conversion allocation"
        );
        assert!(
            OnlineForwardPlan {
                sizing,
                fixed_working_set_bytes: usize::MAX,
            }
            .operation_peak_bytes(bh, sq, sk, d)
            .is_none(),
            "reservation peak overflow must fail closed"
        );
    }

    #[test]
    fn installed_geometry_is_independent_of_live_memory() {
        let geometry = effective_score_geometry();
        assert_eq!(geometry.materialized_budget_mib, 2048);
        assert_eq!(geometry.rocm_online_budget_mib, 1024);
        let (q_tile, key_tile) = online_tile_lens(
            OnlineTileSizing {
                query_tile: geometry.rocm_online_query_tile,
                key_tile: geometry.rocm_online_key_tile,
                score_budget_bytes: geometry.rocm_online_budget_mib * 1024 * 1024,
            },
            16,
            4096,
            8192,
        );
        assert!(q_tile > 0 && key_tile > 0);
    }

    #[test]
    fn online_attention_rejects_unbounded_low_budget_plan() {
        let geometry = effective_score_geometry();
        let low_budget = OnlineTileSizing {
            query_tile: geometry.rocm_online_query_tile,
            key_tile: geometry.rocm_online_key_tile,
            score_budget_bytes: 1024 * 1024,
        };
        let error = validate_online_tile_plan(low_budget, 16, 128 * 1024, 128 * 1024)
            .expect_err("one-megabyte budget must not degrade into a near-hang");
        assert!(error.to_string().contains("bounded tile plan"));

        let bounded = OnlineTileSizing {
            score_budget_bytes: 1024 * 1024 * 1024,
            ..low_budget
        };
        validate_online_tile_plan(bounded, 16, 128 * 1024, 128 * 1024)
            .expect("one-gibibyte score budget has a bounded default tile plan");
        let (q_tile, k_tile) = online_tile_lens(bounded, 16, 128 * 1024, 128 * 1024);
        let score_only = online_tile_bytes(16, q_tile, k_tile).unwrap();
        let error = validate_online_operation_plan(
            bounded,
            16,
            128 * 1024,
            128 * 1024,
            128,
            score_only,
            OnlinePass::Forward,
        )
        .expect_err("score-only accounting must reject omitted non-score scratch");
        assert!(error.to_string().contains("non-score tile scratch"));
    }

    #[test]
    fn materialized_query_tiling_rejects_per_token_fallback() {
        let error = validate_materialized_query_tile_plan(1, 16, 128 * 1024, 128 * 1024, 0)
            .expect_err("zero budget must not produce one query tile per token");
        assert!(error.to_string().contains("bounded plan"));

        let minimum = materialized_score_working_set_bytes(
            1,
            16,
            MATERIALIZED_SCORE_TILE_GRANULARITY,
            128 * 1024,
        )
        .unwrap();
        assert_eq!(
            validate_materialized_query_tile_plan(1, 16, 128 * 1024, 128 * 1024, minimum).unwrap(),
            MATERIALIZED_SCORE_TILE_GRANULARITY
        );
    }

    #[test]
    fn materialized_query_tile_respects_score_element_cap() {
        let tile_len = query_tile_len_for_budget(1, 16, 32_768, 8_192, usize::MAX);
        assert!(tile_len > 0);
        assert!(tile_len <= 1_024, "tile_len={tile_len}");
        assert!(
            16usize.saturating_mul(tile_len).saturating_mul(32_768)
                <= effective_score_geometry().materialized_tile_max_elements,
            "tile_len={tile_len} still exceeds score element cap"
        );
    }
}
