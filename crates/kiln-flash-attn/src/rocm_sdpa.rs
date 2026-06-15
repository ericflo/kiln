//! ROCm scaled-dot-product attention.
//!
//! The hot bf16/head_dim=128/256 forward paths use native exact HIP kernels that
//! stream attention through saved log-sum-exp without materializing `[sq, sk]`
//! scores. Backward uses an exact BLASLt materialized composite when the live
//! score budget permits, and falls back to native bounded HIP kernels for tiles
//! that would be too large. Other shapes use the same correct, **fully
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

use std::cell::Cell;

use half::bf16;
use kiln_tensor::{DType as KtDType, Device as KtDevice, Tensor as KtTensor};

use crate::kt_api::FlashAttnError;

/// `ScalarKind::MulScalar` tag (see `kiln_tensor::ops::scalar`).
const SCALAR_MUL: i32 = 2;

/// Additive-mask fill value for masked (future) attention positions. A large
/// finite negative number rather than `f32::NEG_INFINITY` so that BF16 / F16
/// downcasts in the softmax stay well-defined (`exp` underflows to 0).
const NEG_FILL: f32 = -1.0e30;
const DEFAULT_MATERIALIZED_SCORE_BUDGET_MB: usize = 4096;
const DYNAMIC_MATERIALIZED_SCORE_BUDGET_MAX_MB: usize = 32 * 1024;
const DYNAMIC_MATERIALIZED_SCORE_BUDGET_FREE_DIVISOR: usize = 3;
const MATERIALIZED_SCORE_SCRATCH_BUFFERS: usize = 3;
const MATERIALIZED_BWD_SCORE_SCRATCH_BUFFERS: usize = 8;
const MATERIALIZED_SCORE_TILE_GRANULARITY: usize = 128;
const DEFAULT_MATERIALIZED_SCORE_TILE_MAX_ELEMENTS: usize = 1 << 29;
const DEFAULT_NATIVE_FWD_MAX_SEQ: usize = 4096;
const DEFAULT_NATIVE_FWD_QUERY_TILE: usize = 2048;
const DEFAULT_NATIVE_STREAMING_FWD_MIN_SEQ: usize = 8192;
const DEFAULT_NATIVE_STREAMING_FWD_KEY_TILE: usize = 4096;
const DEFAULT_NATIVE_BWD_HD128_MAX_SEQ: usize = 1024;
const DEFAULT_NATIVE_BWD_HD256_MAX_SEQ: usize = 512;
const DEFAULT_NATIVE_BWD_LONG_MIN_SEQ: usize = 4096;
const DEFAULT_ONLINE_QUERY_TILE: usize = 2048;
const DEFAULT_ONLINE_KEY_TILE: usize = 4096;
const DEFAULT_ONLINE_MATMUL_BATCH_GROUP: usize = 4;
const MIN_ONLINE_SCORE_TILE_BUDGET_MB: usize = 256;
const DEFAULT_ONLINE_SCORE_TILE_BUDGET_MB: usize = 1024;
const DYNAMIC_ONLINE_SCORE_TILE_BUDGET_MAX_MB: usize = 1024;
const DYNAMIC_ONLINE_SCORE_TILE_BUDGET_FREE_DIVISOR: usize = 32;
const ONLINE_TILE_GRANULARITY: usize = 128;
const DEFAULT_F32_MATMUL_INNER_TILE: usize = 4096;

thread_local! {
    static ROCM_ONLINE_FWD_OVERRIDE: Cell<Option<bool>> = Cell::new(None);
}

pub fn with_rocm_online_fwd_disabled<T>(f: impl FnOnce() -> T) -> T {
    let previous = ROCM_ONLINE_FWD_OVERRIDE.with(|cell| {
        let previous = cell.get();
        cell.set(Some(false));
        previous
    });
    struct Guard(Option<bool>);
    impl Drop for Guard {
        fn drop(&mut self) {
            ROCM_ONLINE_FWD_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _guard = Guard(previous);
    f()
}

fn dev_index(device: KtDevice) -> Result<usize, FlashAttnError> {
    match device {
        KtDevice::Rocm(i) => Ok(i),
        other => Err(FlashAttnError::Msg(format!(
            "rocm-sdpa: expected Device::Rocm, got {other:?}"
        ))),
    }
}

fn map_kt<T>(r: Result<T, kiln_tensor::Error>) -> Result<T, FlashAttnError> {
    r.map_err(|e| FlashAttnError::Msg(format!("rocm-sdpa: {e}")))
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

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .and_then(|s| match s.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn debug_rocm_flash_finite_checks() -> bool {
    env_bool("KILN_DEBUG_ROCM_FLASH_FINITE").unwrap_or(false)
        || debug_rocm_flash_stats_labels().is_some()
}

fn debug_rocm_flash_stats_labels() -> Option<Vec<String>> {
    let labels: Vec<String> = std::env::var("KILN_DEBUG_ROCM_FLASH_STATS_LABEL")
        .ok()?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    (!labels.is_empty()).then_some(labels)
}

fn debug_rocm_flash_label_matches(label: &str, filters: Option<&[String]>) -> bool {
    filters
        .map(|filters| filters.iter().any(|filter| label.contains(filter)))
        .unwrap_or(true)
}

fn ensure_debug_finite(
    enabled: bool,
    label: impl AsRef<str>,
    tensor: &KtTensor,
) -> Result<(), FlashAttnError> {
    if !enabled {
        return Ok(());
    }
    let label = label.as_ref();
    let stats_filters = debug_rocm_flash_stats_labels();
    if !debug_rocm_flash_label_matches(label, stats_filters.as_deref()) {
        return Ok(());
    }
    let (finite, value_summary) = if stats_filters.is_some() {
        summarize_debug_values(tensor)
            .map_err(|e| FlashAttnError::Msg(format!("rocm online sdpa scan {label}: {e}")))?
    } else {
        let finite = map_kt(tensor.all_finite())
            .map_err(|e| FlashAttnError::Msg(format!("rocm online sdpa scan {label}: {e}")))?;
        let value_summary = if finite {
            String::new()
        } else {
            summarize_debug_values(tensor)
                .map(|(_, summary)| summary)
                .unwrap_or_else(|e| format!("stats_error={e}"))
        };
        (finite, value_summary)
    };
    if stats_filters.is_some() {
        eprintln!(
            "kiln_rocm_online_stats label={label} dtype={} shape={:?} device={} contiguous={} start_offset={} strides={:?} {value_summary}",
            tensor.dtype(),
            tensor.shape(),
            tensor.device(),
            tensor.is_contiguous(),
            tensor.layout().start_offset(),
            tensor.strides(),
        );
    }
    if finite {
        return Ok(());
    }
    Err(FlashAttnError::Msg(format!(
        "rocm online sdpa non-finite at {label}: dtype={} shape={:?} device={} contiguous={} start_offset={} strides={:?} {value_summary}",
        tensor.dtype(),
        tensor.shape(),
        tensor.device(),
        tensor.is_contiguous(),
        tensor.layout().start_offset(),
        tensor.strides(),
    )))
}

fn summarize_debug_values(tensor: &KtTensor) -> Result<(bool, String), kiln_tensor::Error> {
    let host = tensor
        .to_device(KtDevice::Cpu)?
        .to_dtype(KtDType::F32)?
        .contiguous()?;
    let values = host.to_vec::<f32>()?;
    let mut first_bad: Option<(usize, f32)> = None;
    let mut max_abs = 0.0f32;
    let mut max_abs_idx = 0usize;
    for (idx, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            let abs = value.abs();
            if abs > max_abs {
                max_abs = abs;
                max_abs_idx = idx;
            }
        } else if first_bad.is_none() {
            first_bad = Some((idx, value));
        }
    }
    let shape = tensor.shape();
    let coord = |mut idx: usize| -> Vec<usize> {
        let mut out = vec![0usize; shape.len()];
        for axis in (0..shape.len()).rev() {
            let dim = shape[axis].max(1);
            out[axis] = idx % dim;
            idx /= dim;
        }
        out
    };
    let (bad_idx, bad_value) = first_bad.unwrap_or((usize::MAX, f32::NAN));
    let summary = format!(
        "first_bad_flat={} first_bad_coord={:?} first_bad_value={} max_finite_abs={} max_finite_abs_flat={} max_finite_abs_coord={:?}",
        bad_idx,
        if bad_idx == usize::MAX {
            Vec::new()
        } else {
            coord(bad_idx)
        },
        bad_value,
        max_abs,
        max_abs_idx,
        coord(max_abs_idx)
    );
    Ok((first_bad.is_none(), summary))
}

fn rocm_sync_device(device: KtDevice) -> Result<(), FlashAttnError> {
    if kiln_tensor::rocm_capture_arena_active() {
        return Ok(());
    }
    let dev_idx = dev_index(device)?;
    // The tiled online path uses the active ROCm compute stream for tensor
    // ops, hipBLASLt calls, and the small custom HIP state kernels. A
    // stream-level fence preserves producer/consumer ordering without draining
    // unrelated device work after every QK/softmax/PV substep.
    map_kt(kiln_tensor::rocm_synchronize_compute_stream(dev_idx))
}

fn rocm_timed_finish(
    enabled: bool,
    device: KtDevice,
    acc_ms: &mut f64,
    started: std::time::Instant,
) -> Result<(), FlashAttnError> {
    if enabled {
        rocm_sync_device(device)?;
        *acc_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(())
}

fn rocm_attention_cooperative_yield(device: KtDevice) -> Result<(), FlashAttnError> {
    if matches!(
        env_bool("KILN_ROCM_ATTENTION_COOPERATIVE_YIELD"),
        Some(false)
    ) {
        return Ok(());
    }
    rocm_sync_device(device)?;
    let sleep_ms = env_usize("KILN_ROCM_ATTENTION_YIELD_MS").unwrap_or(1);
    if sleep_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(sleep_ms as u64));
    } else {
        std::thread::yield_now();
    }
    Ok(())
}

fn f32_matmul_inner_tile_len() -> usize {
    env_usize("KILN_ROCM_F32_MATMUL_INNER_TILE").unwrap_or(DEFAULT_F32_MATMUL_INNER_TILE)
}

fn rocm_matmul_split_inner_f32(
    lhs: &KtTensor,
    rhs: &KtTensor,
    device: KtDevice,
    debug_finite: bool,
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
    let tile = f32_matmul_inner_tile_len().max(1);
    if inner <= tile {
        let out = rocm_matmul_f32_batch_grouped(lhs, rhs, device, label)?;
        rocm_sync_device(device)?;
        return Ok(out);
    }

    let mut acc: Option<KtTensor> = None;
    let mut start = 0usize;
    while start < inner {
        let len = (inner - start).min(tile);
        let lhs_block = rocm_contig(&map_kt(lhs.narrow(2, start, len))?)?;
        let rhs_block = rocm_contig(&map_kt(rhs.narrow(1, start, len))?)?;
        rocm_sync_device(device)?;
        ensure_debug_finite(
            debug_finite,
            format!("{label} lhs inner [{start}, {})", start + len),
            &lhs_block,
        )?;
        ensure_debug_finite(
            debug_finite,
            format!("{label} rhs inner [{start}, {})", start + len),
            &rhs_block,
        )?;
        let partial = rocm_matmul_f32_batch_grouped(&lhs_block, &rhs_block, device, label)?;
        rocm_sync_device(device)?;
        ensure_debug_finite(
            debug_finite,
            format!("{label} partial inner [{start}, {})", start + len),
            &partial,
        )?;
        acc = Some(match acc.take() {
            Some(prev) => {
                let sum = map_kt(prev.add(&partial))?;
                rocm_sync_device(device)?;
                drop(prev);
                drop(partial);
                sum
            }
            None => partial,
        });
        start += len;
    }

    let out = acc.ok_or_else(|| FlashAttnError::Msg(format!("{label}: empty inner dimension")))?;
    ensure_debug_finite(debug_finite, label, &out)?;
    Ok(out)
}

fn online_matmul_batch_group() -> usize {
    env_usize("KILN_ROCM_FLASH_ONLINE_MATMUL_BATCH_GROUP")
        .unwrap_or(DEFAULT_ONLINE_MATMUL_BATCH_GROUP)
        .max(1)
}

fn should_group_online_batch_matmul(batch: usize) -> bool {
    let group = online_matmul_batch_group();
    group > 0 && batch > group
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
    device: KtDevice,
    label: &str,
) -> Result<KtTensor, FlashAttnError> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() != 3 || rhs_shape.len() != 3 || lhs_shape[0] != rhs_shape[0] {
        return map_kt(kiln_tensor::rocm_matmul(lhs, rhs));
    }
    let batch = lhs_shape[0];
    if !should_group_online_batch_matmul(batch) {
        return map_kt(kiln_tensor::rocm_matmul(lhs, rhs));
    }

    let group = online_matmul_batch_group().min(batch);
    if rocm_trace_fwd_enabled() {
        eprintln!(
            "kiln_rocm_flash_fwd path=grouped_matmul_f32 label={label:?} batch={batch} group={group} device={device:?}"
        );
    }
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
    rocm_sync_device(device)?;
    Ok(out)
}

fn rocm_matmul_rhs_transposed_to_dtype_batch_grouped(
    lhs: &KtTensor,
    rhs: &KtTensor,
    out_dtype: KtDType,
    device: KtDevice,
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
    if !should_group_online_batch_matmul(batch) {
        return map_kt(kiln_tensor::rocm_matmul_rhs_transposed_to_dtype(
            lhs, rhs, out_dtype,
        ));
    }

    let group = online_matmul_batch_group().min(batch);
    if rocm_trace_fwd_enabled() {
        eprintln!(
            "kiln_rocm_flash_fwd path=grouped_matmul_rhs_t label={label:?} batch={batch} group={group} out_dtype={out_dtype}"
        );
    }
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
    rocm_sync_device(device)?;
    Ok(out)
}

fn materialized_score_budget_mb() -> usize {
    if let Some(override_mb) = env_usize("KILN_ROCM_FLASH_SCORE_BUDGET_MB")
        .or_else(|| env_usize("KILN_FULL_ATTN_SCORE_BUDGET_MB"))
    {
        return override_mb;
    }

    let snapshot = kiln_memory::current_memory_snapshot();
    if snapshot.free_bytes == 0 {
        return DEFAULT_MATERIALIZED_SCORE_BUDGET_MB;
    }
    let dynamic_mb = (snapshot.free_bytes as usize)
        / DYNAMIC_MATERIALIZED_SCORE_BUDGET_FREE_DIVISOR
        / (1024 * 1024);
    dynamic_mb.clamp(
        DEFAULT_MATERIALIZED_SCORE_BUDGET_MB,
        DYNAMIC_MATERIALIZED_SCORE_BUDGET_MAX_MB,
    )
}

fn materialized_score_budget_bytes() -> usize {
    materialized_score_budget_mb().saturating_mul(1024 * 1024)
}

fn materialized_score_tile_max_elements() -> usize {
    env_usize("KILN_ROCM_FLASH_SCORE_TILE_MAX_ELEMENTS")
        .or_else(|| env_usize("KILN_FULL_ATTN_SCORE_TILE_MAX_ELEMENTS"))
        .unwrap_or(DEFAULT_MATERIALIZED_SCORE_TILE_MAX_ELEMENTS)
}

fn materialized_score_working_set_bytes(b: usize, h: usize, sq: usize, sk: usize) -> Option<usize> {
    b.checked_mul(h)?
        .checked_mul(sq)?
        .checked_mul(sk)?
        .checked_mul(std::mem::size_of::<f32>())?
        .checked_mul(MATERIALIZED_SCORE_SCRATCH_BUFFERS)
}

fn materialized_bwd_working_set_bytes(b: usize, h: usize, sq: usize, sk: usize) -> Option<usize> {
    b.checked_mul(h)?
        .checked_mul(sq)?
        .checked_mul(sk)?
        .checked_mul(std::mem::size_of::<f32>())?
        .checked_mul(MATERIALIZED_BWD_SCORE_SCRATCH_BUFFERS)
}

fn native_scalar_fwd_enabled(sq: usize, sk: usize) -> bool {
    if matches!(env_bool("KILN_ROCM_FLASH_NATIVE_SCALAR"), Some(false)) {
        return false;
    }
    if env_bool("KILN_ROCM_FLASH_NATIVE_SCALAR_FORCE").unwrap_or(false) {
        return true;
    }
    let max_seq = env_usize("KILN_ROCM_FLASH_NATIVE_MAX_SEQ")
        .unwrap_or(DEFAULT_NATIVE_FWD_MAX_SEQ)
        .max(1);
    sq.max(sk) <= max_seq
}

fn native_fwd_query_tile_len() -> usize {
    env_usize("KILN_ROCM_FLASH_NATIVE_QUERY_TILE").unwrap_or(DEFAULT_NATIVE_FWD_QUERY_TILE)
}

fn native_tiled_fwd_enabled(sq: usize, sk: usize) -> bool {
    if matches!(env_bool("KILN_ROCM_FLASH_NATIVE_TILED"), Some(false)) {
        return false;
    }
    if env_bool("KILN_ROCM_FLASH_NATIVE_TILED_FORCE").unwrap_or(false) {
        return true;
    }
    sq.max(sk) > native_fwd_query_tile_len()
}

fn native_streaming_fwd_key_tile_len() -> usize {
    env_usize("KILN_ROCM_FLASH_NATIVE_KEY_TILE").unwrap_or(DEFAULT_NATIVE_STREAMING_FWD_KEY_TILE)
}

fn native_streaming_fwd_enabled(sq: usize, sk: usize) -> bool {
    if matches!(env_bool("KILN_ROCM_FLASH_NATIVE_STREAMING"), Some(false)) {
        return false;
    }
    if env_bool("KILN_ROCM_FLASH_NATIVE_STREAMING_FORCE").unwrap_or(false) {
        return true;
    }
    let min_seq = env_usize("KILN_ROCM_FLASH_NATIVE_STREAMING_MIN_SEQ")
        .unwrap_or(DEFAULT_NATIVE_STREAMING_FWD_MIN_SEQ)
        .max(1);
    sq.max(sk) >= min_seq
}

fn native_rectangular_causal_fwd_enabled() -> bool {
    env_bool("KILN_ROCM_FLASH_NATIVE_RECTANGULAR_CAUSAL").unwrap_or(true)
}

fn rocm_ck_no_lse_fwd_enabled() -> bool {
    env_bool("KILN_ROCM_FLASH_CK").unwrap_or(false)
}

fn rocm_online_fwd_enabled() -> bool {
    if let Some(force) = ROCM_ONLINE_FWD_OVERRIDE.with(|cell| cell.get()) {
        return force;
    }
    env_bool("KILN_ROCM_FLASH_ONLINE").unwrap_or(true)
}

fn rocm_online_bwd_enabled() -> bool {
    env_bool("KILN_ROCM_FLASH_ONLINE_BWD").unwrap_or(true)
}

fn rocm_trace_fwd_enabled() -> bool {
    env_bool("KILN_TRACE_ROCM_FLASH_FWD").unwrap_or(false)
}

fn rocm_materialized_bwd_enabled(b: usize, h: usize, sq: usize, sk: usize) -> bool {
    if let Some(force) = env_bool("KILN_ROCM_FLASH_MATMUL_BWD") {
        return force;
    }
    materialized_bwd_working_set_bytes(b, h, sq, sk)
        .map(|bytes| bytes <= materialized_score_budget_bytes())
        .unwrap_or(false)
}

fn rocm_native_bwd_preferred(_b: usize, _h: usize, sq: usize, sk: usize, d: usize) -> bool {
    if env_bool("KILN_ROCM_FLASH_NATIVE_BWD_FORCE").unwrap_or(false) {
        return true;
    }
    if matches!(env_bool("KILN_ROCM_FLASH_MATMUL_BWD"), Some(true)) {
        return false;
    }
    if let Some(force) = env_bool("KILN_ROCM_FLASH_NATIVE_BWD") {
        return force;
    }
    let default_max_seq = match d {
        128 => DEFAULT_NATIVE_BWD_HD128_MAX_SEQ,
        256 => DEFAULT_NATIVE_BWD_HD256_MAX_SEQ,
        _ => return false,
    };
    if let Some(max_seq) = env_usize("KILN_ROCM_FLASH_NATIVE_BWD_MAX_SEQ") {
        return sq.max(sk) <= max_seq.max(1);
    }
    if sq.max(sk) <= default_max_seq {
        return true;
    }
    let long_min_seq = env_usize("KILN_ROCM_FLASH_NATIVE_BWD_LONG_MIN_SEQ")
        .unwrap_or(DEFAULT_NATIVE_BWD_LONG_MIN_SEQ)
        .max(1);
    sq.max(sk) >= long_min_seq
}

fn rocm_collapsed_gqa_bwd_enabled() -> bool {
    env_bool("KILN_ROCM_FLASH_COLLAPSED_GQA_BWD").unwrap_or(true)
}

fn rocm_native_direct_collapsed_gqa_bwd_enabled() -> bool {
    env_bool("KILN_ROCM_FLASH_NATIVE_DIRECT_COLLAPSED_GQA_BWD").unwrap_or(false)
}

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

fn rocm_device_free_bytes(device: KtDevice) -> Option<usize> {
    let idx = dev_index(device).ok()?;
    kiln_tensor::rocm_mem_get_info(idx)
        .ok()
        .map(|(free, _)| free)
}

fn online_score_tile_budget_bytes(device: KtDevice) -> usize {
    let budget_mb = if let Some(override_mb) = env_usize("KILN_ROCM_FLASH_ONLINE_SCORE_BUDGET_MB") {
        override_mb
    } else {
        let free_bytes = rocm_device_free_bytes(device)
            .or_else(|| {
                let snapshot = kiln_memory::current_memory_snapshot();
                (snapshot.free_bytes > 0).then_some(snapshot.free_bytes as usize)
            })
            .unwrap_or(0);
        if free_bytes == 0 {
            DEFAULT_ONLINE_SCORE_TILE_BUDGET_MB
        } else {
            let dynamic_mb =
                free_bytes / DYNAMIC_ONLINE_SCORE_TILE_BUDGET_FREE_DIVISOR / (1024 * 1024);
            dynamic_mb.clamp(
                MIN_ONLINE_SCORE_TILE_BUDGET_MB,
                DYNAMIC_ONLINE_SCORE_TILE_BUDGET_MAX_MB,
            )
        }
    };
    budget_mb.saturating_mul(1024 * 1024)
}

fn online_tile_bytes(bh: usize, q_tile: usize, k_tile: usize) -> Option<usize> {
    bh.checked_mul(q_tile)?
        .checked_mul(k_tile)?
        .checked_mul(std::mem::size_of::<f32>())
}

fn online_tile_lens(
    device: KtDevice,
    bh: usize,
    remaining_q: usize,
    remaining_k: usize,
) -> (usize, usize) {
    let mut q_tile = env_usize("KILN_ROCM_FLASH_ONLINE_QUERY_TILE")
        .unwrap_or(DEFAULT_ONLINE_QUERY_TILE)
        .min(remaining_q)
        .max(1);
    let mut k_tile = env_usize("KILN_ROCM_FLASH_ONLINE_KEY_TILE")
        .unwrap_or(DEFAULT_ONLINE_KEY_TILE)
        .min(remaining_k)
        .max(1);
    let budget = online_score_tile_budget_bytes(device).max(1);

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
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    if sq == 0 || sk == 0 {
        return sdpa_forward(q, k, v, b, sq, sk, h, hk, d, scale, causal);
    }

    let debug_finite = debug_rocm_flash_finite_checks();
    let device = q.device();
    let bh = b * h;
    let causal_offset = sk as isize - sq as isize;
    let k_exp = gqa_expand_heads(k, h, hk, device)?;
    let v_exp = gqa_expand_heads(v, h, hk, device)?;

    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?;
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?;

    let q3 = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?;
    let k3 = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?;
    let v3 = map_kt(v_bhsd.reshape(vec![bh, sk, d]))?;
    let v3f = rocm_cast_to(&v3, KtDType::F32)?;
    ensure_debug_finite(debug_finite, "online q3", &q3)?;
    ensure_debug_finite(debug_finite, "online k3", &k3)?;
    ensure_debug_finite(debug_finite, "online v3f", &v3f)?;
    if debug_finite {
        eprintln!(
            "kiln_rocm_online_sdpa b={b} sq={sq} sk={sk} h={h} hk={hk} d={d} causal={causal} causal_offset={causal_offset}",
        );
    }

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
        let (q_tile_cap, key_tile_cap) = online_tile_lens(device, bh, remaining_q, max_k);
        let q_len = remaining_q.min(q_tile_cap).max(1);
        let rows = bh * q_len;

        let q_tile = rocm_contig(&map_kt(q3.narrow(1, q_start, q_len))?)?;
        rocm_sync_device(device)?;
        let q_sentinel = if debug_finite && q_start == 0 && q_len < sq {
            Some((q_len, (sq - q_len).min(q_tile_cap).max(1)))
        } else {
            None
        };
        let row_m_zero = map_kt(KtTensor::zeros(vec![rows], KtDType::F32, device))?;
        let row_m = map_kt(kiln_tensor::ops::full_like(&row_m_zero, f32::NEG_INFINITY))?;
        let row_l = map_kt(KtTensor::zeros(vec![rows], KtDType::F32, device))?;
        let alpha = map_kt(KtTensor::zeros(vec![rows], KtDType::F32, device))?;
        let beta = map_kt(KtTensor::zeros(vec![rows], KtDType::F32, device))?;
        let acc = map_kt(KtTensor::zeros(vec![bh, q_len, d], KtDType::F32, device))?;
        ensure_debug_finite(
            debug_finite,
            format!("online q_tile q_start={q_start} q_len={q_len}"),
            &q_tile,
        )?;
        ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_q_tile_copy")?;

        let block_limit = if causal {
            causal_block_limit(sk, q_start, q_len, causal_offset)
        } else {
            sk
        };
        let mut k_start = 0usize;
        while k_start < block_limit {
            let k_len = (block_limit - k_start).min(key_tile_cap).max(1);
            let k_block = rocm_contig(&map_kt(k3.narrow(1, k_start, k_len))?)?;
            let v_block = rocm_contig(&map_kt(v3f.narrow(1, k_start, k_len))?)?;
            rocm_sync_device(device)?;
            ensure_debug_finite(
                debug_finite,
                format!("online k_block q_start={q_start} k_start={k_start} k_len={k_len}"),
                &k_block,
            )?;
            ensure_debug_finite(
                debug_finite,
                format!("online v_block q_start={q_start} k_start={k_start} k_len={k_len}"),
                &v_block,
            )?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_kv_block_copy")?;

            let scores = rocm_matmul_rhs_transposed_to_dtype_batch_grouped(
                &q_tile,
                &k_block,
                KtDType::F32,
                device,
                &format!("online qk q_start={q_start} k_start={k_start}"),
            )?;
            rocm_sync_device(device)?;
            ensure_debug_finite(
                debug_finite,
                format!("online qk q_start={q_start} k_start={k_start}"),
                &scores,
            )?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_qk_matmul")?;
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
            rocm_sync_device(device)?;
            ensure_debug_finite(
                debug_finite,
                format!("online scaled_scores q_start={q_start} k_start={k_start}"),
                &scores,
            )?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_scale_mask")?;
            let block_m = map_kt(kiln_tensor::rocm_max_axis(&scores, 2))?;
            rocm_sync_device(device)?;
            ensure_debug_finite(
                debug_finite,
                format!("online block_m q_start={q_start} k_start={k_start}"),
                &block_m,
            )?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_block_m")?;
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
            rocm_sync_device(device)?;
            ensure_debug_finite(
                debug_finite,
                format!("online probs q_start={q_start} k_start={k_start}"),
                &probs,
            )?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_exp_mask")?;
            let block_l = map_kt(kiln_tensor::rocm_sum_axis(&probs, 2))?;
            let block_out = rocm_matmul_split_inner_f32(
                &probs,
                &v_block,
                device,
                debug_finite,
                &format!("online block_out q_start={q_start} k_start={k_start}"),
            )?;
            rocm_sync_device(device)?;
            ensure_debug_finite(
                debug_finite,
                format!("online block_l q_start={q_start} k_start={k_start}"),
                &block_l,
            )?;
            ensure_debug_finite(
                debug_finite,
                format!("online block_out q_start={q_start} k_start={k_start}"),
                &block_out,
            )?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_block_out")?;
            let block_m_flat = map_kt(block_m.reshape(vec![rows]))?;
            let block_l_flat = map_kt(block_l.reshape(vec![rows]))?;

            // The online path mixes kt matmul/reduce kernels with custom HIP
            // state kernels that launch from different tensor-owned streams.
            // Order each producer/consumer handoff explicitly; otherwise a
            // later stream can read block_l/block_out or alpha/beta before the
            // producing stream has finished, which shows up as intermittent
            // NaNs on long ROCm rows.
            rocm_sync_device(device)?;
            online_update_state_f32(
                &row_m,
                &row_l,
                &block_m_flat,
                &block_l_flat,
                &alpha,
                &beta,
                rows,
            )?;
            rocm_sync_device(device)?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_update_state")?;
            online_update_acc_f32(&acc, &block_out, &alpha, &beta, rows, d)?;
            rocm_sync_device(device)?;
            ensure_debug_finite(
                debug_finite,
                format!("online row_l q_start={q_start} k_start={k_start}"),
                &row_l,
            )?;
            ensure_debug_finite(
                debug_finite,
                format!("online acc q_start={q_start} k_start={k_start}"),
                &acc,
            )?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_update_acc")?;

            k_start += k_len;
            rocm_attention_cooperative_yield(device)?;
        }

        let out3 = map_kt(KtTensor::zeros(vec![bh, q_len, d], KtDType::F32, device))?;
        let lse2 = map_kt(KtTensor::zeros(vec![rows], KtDType::F32, device))?;
        online_finalize_f32(&acc, &row_m, &row_l, &out3, &lse2, rows, d)?;
        // Finalize writes out3/lse2 from the custom HIP stream; the following
        // transpose/cast/cat work can launch from tensor-owned kt streams.
        // Keep the online state tensors alive and ordered until final output
        // materialization has consumed them.
        rocm_sync_device(device)?;
        ensure_debug_finite(
            debug_finite,
            format!("online out3 q_start={q_start} q_len={q_len}"),
            &out3,
        )?;
        ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_finalize")?;
        let out_tile = bhsd3_to_bshd_bf16(&out3, b, h, q_len, d)?;
        rocm_sync_device(device)?;
        ensure_debug_finite(
            debug_finite,
            format!("online out_tile q_start={q_start} q_len={q_len}"),
            &out_tile,
        )?;
        ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "after_out_transpose_cast")?;
        out_tiles.push(out_tile);
        lse_tiles.push(map_kt(lse2.reshape(vec![b, h, q_len]))?);

        if debug_finite {
            ensure_debug_finite(
                true,
                format!("online q_tile source post q_start={q_start} q_len={q_len}"),
                &q_tile,
            )?;
            ensure_online_q_sentinel(debug_finite, &q3, q_sentinel, "post_tile")?;
        }

        q_start += q_len;
    }

    let out_refs: Vec<&KtTensor> = out_tiles.iter().collect();
    let lse_refs: Vec<&KtTensor> = lse_tiles.iter().collect();
    let out = map_kt(KtTensor::cat(&out_refs, 1))?;
    let lse = map_kt(KtTensor::cat(&lse_refs, 2))?;
    rocm_sync_device(device)?;
    ensure_debug_finite(debug_finite, "online out_cat", &out)?;
    ensure_debug_finite(debug_finite, "online lse_cat", &lse)?;
    Ok((out, lse))
}

fn ensure_online_q_sentinel(
    debug_finite: bool,
    q3: &KtTensor,
    sentinel: Option<(usize, usize)>,
    label: &str,
) -> Result<(), FlashAttnError> {
    if !debug_finite {
        return Ok(());
    }
    let Some((q_start, q_len)) = sentinel else {
        return Ok(());
    };
    let q_tile = rocm_contig(&map_kt(q3.narrow(1, q_start, q_len))?)?;
    ensure_debug_finite(
        true,
        format!("online q_sentinel {label} q_start={q_start} q_len={q_len}"),
        &q_tile,
    )
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

    let causal_offset = sk.saturating_sub(sq);
    let mut out_tiles = Vec::new();
    let mut lse_tiles = Vec::new();
    let mut tile_start = 0usize;
    while tile_start < sq {
        let remaining = sq - tile_start;
        let max_key_len = if causal {
            causal_offset + tile_start + remaining
        } else {
            sk
        };
        let tile_len = query_tile_len_for_budget(b, h, max_key_len, remaining, budget_bytes);
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
    device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    let _ = device;
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
    rocm_sync_device(device)?;
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
    let debug_finite = debug_rocm_flash_finite_checks();
    let device = q.device();
    let bh = b * h;

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
    ensure_debug_finite(debug_finite, "materialized qk", &qk)?;
    let scaled_scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, scale))?;
    ensure_debug_finite(debug_finite, "materialized scaled_scores", &scaled_scores)?;

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
    ensure_debug_finite(debug_finite, "materialized masked_scores", &masked_scores)?;

    // 5. p = softmax(scores) over last axis -> [b*h, sq, sk]; lse from scores+p.
    let p = map_kt(kiln_tensor::rocm_softmax_last_axis(&masked_scores))?; // [b*h, sq, sk] f32
    ensure_debug_finite(debug_finite, "materialized softmax", &p)?;
    let lse = compute_lse(&masked_scores, &p, b, h, sq, device)?; // [b, h, sq] f32
    ensure_debug_finite(debug_finite, "materialized lse", &lse)?;

    // 6. out = p @ v -> [b*h, sq, d]; reshape/transpose back to [b, sq, h, d].
    let out3 = rocm_matmul_split_inner_f32(&p, &v3f, device, debug_finite, "materialized out3")?; // [b*h, sq, d] f32
    ensure_debug_finite(debug_finite, "materialized out3", &out3)?;
    let out_bhsd = map_kt(out3.reshape(vec![b, h, sq, d]))?; // [b, h, sq, d]
    let out_bshd = rocm_contig(&map_kt(out_bhsd.transpose(1, 2))?)?; // [b, sq, h, d]
    ensure_debug_finite(debug_finite, "materialized out_bshd", &out_bshd)?;
    let out_bf16 = rocm_cast_to(&out_bshd, KtDType::BF16)?; // [b, sq, h, d] bf16
    ensure_debug_finite(debug_finite, "materialized out_bf16", &out_bf16)?;

    // The composite path is a chain of async ROCm kernels over large temporary
    // score/probability tensors. Synchronize before returning so those
    // temporaries cannot be dropped and recycled while later kernels still read
    // them. The caller's post-handoff sync is too late because these locals are
    // gone by then.
    rocm_sync_device(device)?;
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
    let debug_finite = debug_rocm_flash_finite_checks();
    let device = q.device();
    let bh = b * h;

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
    let lse = compute_lse(&masked_scores, &p, b, h, sq, device)?;

    let out3 = rocm_matmul_split_inner_f32(&p, &v3f, device, debug_finite, "head-major out3")?;
    let out_bhsd = map_kt(out3.reshape(vec![b, h, sq, d]))?;
    let out_bf16 = rocm_cast_to(&out_bhsd, KtDType::BF16)?;

    rocm_sync_device(device)?;
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
    if native_scalar_fwd_enabled(sq, sk)
        || native_tiled_fwd_enabled(sq, sk)
        || native_streaming_fwd_enabled(sq, sk)
    {
        if let Some(result) =
            try_native_fwd_bf16(q, k, v, b, sq, sk, h, hk, d, softmax_scale, causal)?
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
        if rocm_online_fwd_enabled() && !rectangular_causal_prefix {
            return sdpa_forward_online_tiled(q, k, v, b, sq, sk, h, hk, d, softmax_scale, causal);
        }
        if rectangular_causal_prefix && rocm_trace_fwd_enabled() {
            eprintln!(
                "kiln_rocm_flash_fwd path=materialized_query_tiled_rectangular_prefix batch={b} seq_q={sq} seq_k={sk} heads={h} kv_heads={hk} head_dim={d} causal={causal}"
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
    let budget_bytes = materialized_score_budget_bytes();
    let rectangular_causal_prefix = causal && sq != sk;
    let exceeds_materialized_budget = materialized_score_working_set_bytes(b, h, sq, sk)
        .map(|bytes| bytes > budget_bytes)
        .unwrap_or(true);
    if exceeds_materialized_budget {
        let q_bshd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
        let k_bskhd = rocm_contig(&map_kt(k.transpose(1, 2))?)?;
        let v_bskhd = rocm_contig(&map_kt(v.transpose(1, 2))?)?;
        let (out_bshd, lse) = if rocm_online_fwd_enabled() && !rectangular_causal_prefix {
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
            )?
        } else {
            if rectangular_causal_prefix && rocm_trace_fwd_enabled() {
                eprintln!(
                    "kiln_rocm_flash_fwd path=head_major_materialized_query_tiled_rectangular_prefix batch={b} seq_q={sq} seq_k={sk} heads={h} kv_heads={hk} head_dim={d} causal={causal}"
                );
            }
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
    let (b, sq, h, d) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (sk, hk) = (k_shape[1], k_shape[2]);
    if rocm_ck_no_lse_fwd_enabled() {
        if let Some(out) =
            try_ck_fwd_bf16_no_lse(q, k, v, b, sq, sk, h, hk, d, softmax_scale, causal)?
        {
            return Ok(Some(out));
        }
    }

    let (out, _lse) = flash_attn_fwd_rocm(q, k, v, softmax_scale, causal)?;
    Ok(Some(out))
}

#[allow(clippy::too_many_arguments)]
fn try_ck_fwd_bf16_no_lse(
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
) -> Result<Option<KtTensor>, FlashAttnError> {
    if q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || !matches!(d, 128 | 256)
        || h == 0
        || hk == 0
        || h % hk != 0
        || !causal
        || sk < sq
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

    let q_ptr = kiln_kt_bridge::rocm_input_device_ptr(q, KtDType::BF16, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let k_ptr = kiln_kt_bridge::rocm_input_device_ptr(k, KtDType::BF16, "k")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let v_ptr = kiln_kt_bridge::rocm_input_device_ptr(v, KtDType::BF16, "v")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let out_ptr = kiln_kt_bridge::rocm_output_device_ptr(&out);
    let stream = kiln_kt_bridge::rocm_stream_raw_of(q, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

    let status = unsafe {
        crate::kiln_rocm_flash_attn_fwd_ck_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *mut _,
            std::ptr::null_mut(),
            b as i32,
            sq as i32,
            sk as i32,
            h as i32,
            hk as i32,
            d as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if status == 0 {
        // The CK/no-LSE launch is async and reads q/k/v after returning.
        // Synchronize while those borrowed input tensors are still alive;
        // otherwise ROCm's stream-ordered allocator can recycle a temporary
        // tile before the kernel finishes consuming it.
        rocm_sync_device(q.device())?;
        Ok(Some(out))
    } else {
        Ok(None)
    }
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
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    if !native_fwd_shape_supported(q, k, v, h, hk, d) {
        return Ok(None);
    }
    // Native FA-style causal masking is bottom-right aligned for `sq != sk`.
    // The query-tiled wrapper narrows each key prefix so that bottom-right
    // alignment equals the absolute prefix-causal mask for chunked training.
    if causal && sq != sk && !native_rectangular_causal_fwd_enabled() {
        return Ok(None);
    }

    let q_tile = native_fwd_query_tile_len();
    if sq > q_tile && native_tiled_fwd_enabled(sq, sk) {
        if rocm_trace_fwd_enabled() {
            eprintln!(
                "kiln_rocm_flash_fwd path=native_query_tiled batch={b} seq_q={sq} seq_k={sk} heads={h} kv_heads={hk} head_dim={d} q_tile={q_tile} causal={causal}"
            );
        }
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
        )? {
            return Ok(Some(result));
        }
    }

    let q_c = rocm_contig(q)?;
    let k_c = rocm_contig(k)?;
    let v_c = rocm_contig(v)?;
    if native_scalar_fwd_enabled(sq, sk) || native_tiled_fwd_enabled(sq, sk) {
        if rocm_trace_fwd_enabled() {
            eprintln!(
                "kiln_rocm_flash_fwd path=native_single batch={b} seq_q={sq} seq_k={sk} heads={h} kv_heads={hk} head_dim={d} causal={causal}"
            );
        }
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
            false,
        )? {
            return Ok(Some(result));
        }
    }

    if native_streaming_fwd_enabled(sq, sk) {
        if rocm_trace_fwd_enabled() {
            eprintln!(
                "kiln_rocm_flash_fwd path=native_streaming batch={b} seq_q={sq} seq_k={sk} heads={h} kv_heads={hk} head_dim={d} q_tile={q_tile} key_tile={} causal={causal}",
                native_streaming_fwd_key_tile_len()
            );
        }
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
            native_streaming_fwd_key_tile_len(),
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
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    let q_tile_len = q_tile_len.max(1);
    let device = q.device();
    let causal_offset = sk as isize - sq as isize;
    let k_full = rocm_contig(k)?;
    let v_full = rocm_contig(v)?;
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
            false,
        )?
        else {
            return Ok(None);
        };
        if rocm_trace_fwd_enabled() {
            eprintln!(
                "kiln_rocm_flash_fwd_tile path=native_query_tiled q_start={q_start} q_len={q_len} key_len={key_len}"
            );
        }
        out_tiles.push(out_tile);
        lse_tiles.push(lse_tile);
        q_start += q_len;
        rocm_attention_cooperative_yield(device)?;
    }

    let out_refs: Vec<&KtTensor> = out_tiles.iter().collect();
    let lse_refs: Vec<&KtTensor> = lse_tiles.iter().collect();
    let out = map_kt(KtTensor::cat(&out_refs, 1))?;
    let lse = map_kt(KtTensor::cat(&lse_refs, 2))?;
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
    use_ck: bool,
) -> Result<Option<(KtTensor, KtTensor)>, FlashAttnError> {
    if q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || !matches!(d, 128 | 256)
        || h == 0
        || hk == 0
        || h % hk != 0
        || (use_ck && (!causal || !matches!(d, 128 | 256)))
        || (use_ck && causal && sk < sq)
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
    let stream = kiln_kt_bridge::rocm_stream_raw_of(q, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

    // The native FFI kernels read borrowed q/k/v pointers that often point at
    // just-materialized contiguous tiles. Fence the active ROCm stream before
    // crossing into the custom HIP launch so those producer kernels cannot race
    // with the raw-pointer consumer.
    rocm_sync_device(q.device())?;

    let status = unsafe {
        let launch = if use_ck {
            crate::kiln_rocm_flash_attn_fwd_ck_bf16
        } else {
            crate::kiln_rocm_flash_attn_fwd_bf16
        };
        launch(
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
            stream,
        )
    };
    if status == 0 {
        // The native FFI launch is async. Complete it before returning so
        // caller-local q/k/v temporaries cannot be dropped and recycled while
        // the kernel is still reading them.
        rocm_sync_device(q.device())?;
        Ok(Some((out, lse)))
    } else {
        if rocm_trace_fwd_enabled() {
            eprintln!(
                "kiln_rocm_flash_fwd path=native_ffi_declined status={status} batch={b} seq_q={sq} seq_k={sk} heads={h} kv_heads={hk} head_dim={d} causal={causal} use_ck={use_ck}"
            );
        }
        Ok(None)
    }
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
    let device = q.device();
    let causal_offset = sk as isize - sq as isize;
    let k_full = rocm_contig(k)?;
    let v_full = rocm_contig(v)?;
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
        let out_tile = map_kt(KtTensor::zeros(vec![b, q_len, h, d], KtDType::BF16, device))?;
        let lse_tile = map_kt(KtTensor::zeros(vec![b, h, q_len], KtDType::F32, device))?;
        let acc = map_kt(KtTensor::zeros(vec![b, q_len, h, d], KtDType::F32, device))?;
        let row_l = map_kt(KtTensor::zeros(vec![b, h, q_len], KtDType::F32, device))?;
        let row_m_zero = map_kt(KtTensor::zeros(vec![b, h, q_len], KtDType::F32, device))?;
        let row_m = map_kt(kiln_tensor::ops::full_like(&row_m_zero, f32::NEG_INFINITY))?;

        // The streaming HIP kernel reads q_tile/k_full/v_full and mutates the
        // online-softmax state buffers in-place. These tensors are produced by
        // async tensor ops immediately above; make the handoff explicit before
        // the first raw-pointer update launch.
        rocm_sync_device(device)?;

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
            rocm_attention_cooperative_yield(device)?;
        }

        native_streaming_fwd_finalize_bf16(
            &row_m, &row_l, &acc, &out_tile, &lse_tile, b, q_len, h, d,
        )?;
        // Finalize reads the streaming state tensors and writes out_tile/lse.
        // Keep row_m/row_l/acc alive until that async kernel has completed.
        rocm_sync_device(device)?;
        if rocm_trace_fwd_enabled() {
            eprintln!(
                "kiln_rocm_flash_fwd_tile path=native_streaming q_start={q_start} q_len={q_len} key_len={block_limit} key_tile={key_tile_len}"
            );
        }
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
    let stream = kiln_kt_bridge::rocm_stream_raw_of(q, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_attn_fwd_stream_update_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            row_m_ptr as *mut _,
            row_l_ptr as *mut _,
            acc_ptr as *mut _,
            usize_to_i32("streaming fwd batch", b)?,
            usize_to_i32("streaming fwd q_len", q_len)?,
            usize_to_i32("streaming fwd seqlen_k", sk)?,
            usize_to_i32("streaming fwd seqlen_q_total", sq_total)?,
            usize_to_i32("streaming fwd heads", h)?,
            usize_to_i32("streaming fwd kv_heads", hk)?,
            usize_to_i32("streaming fwd head_dim", d)?,
            softmax_scale,
            if causal { 1 } else { 0 },
            usize_to_i32("streaming fwd q_start", q_start)?,
            usize_to_i32("streaming fwd key_start", key_start)?,
            usize_to_i32("streaming fwd key_len", key_len)?,
            stream,
        )
    };
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
    let stream = kiln_kt_bridge::rocm_stream_raw_of(out, "out")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_attn_fwd_stream_finalize_bf16(
            row_m_ptr as *const _,
            row_l_ptr as *const _,
            acc_ptr as *const _,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            usize_to_i32("streaming fwd finalize batch", b)?,
            usize_to_i32("streaming fwd finalize q_len", q_len)?,
            usize_to_i32("streaming fwd finalize heads", h)?,
            usize_to_i32("streaming fwd finalize head_dim", d)?,
            stream,
        )
    };
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
    let debug_finite = debug_rocm_flash_finite_checks();
    let out3 = rocm_matmul_split_inner_f32(&p, &v3f, device, debug_finite, "dyn-tail out3")?; // [b*h, 1, d]
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
/// on `stream`. Used by the paged-KV writers to scatter a token row into the
/// pool in place. Synchronizes the default stream afterward so the write is
/// observable by a subsequent decode that reads the pool.
fn rocm_d2d_copy(
    dst_ptr: u64,
    src_ptr: u64,
    n_bytes: usize,
    stream: *mut core::ffi::c_void,
    dev_idx: usize,
) -> Result<(), FlashAttnError> {
    let status = unsafe {
        hipMemcpyDtoDAsync(
            dst_ptr as *mut core::ffi::c_void,
            src_ptr as *const core::ffi::c_void,
            n_bytes,
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "rocm-sdpa: hipMemcpyDtoDAsync returned status {status}"
        )));
    }
    // R.10 perf: no device sync needed — the d2d write and any later read of the
    // pool serialize on the one cached per-device stream (FIFO). The old
    // hipDeviceSynchronize stalled the decode pipeline every KV write.
    let _ = dev_idx;
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
    let dev_idx = dev_index(dst.device())?;
    let src_c = rocm_contig(src)?;
    let n_bytes = dst.element_count() * dst.dtype().size_in_bytes();
    let src_ptr = kiln_kt_bridge::rocm_input_device_ptr(&src_c, src.dtype(), "copy_src")?;
    let dst_ptr = kiln_kt_bridge::rocm_input_device_ptr(dst, dst.dtype(), "copy_dst")?;
    let stream = kiln_kt_bridge::rocm_stream_raw_of(dst, "copy_dst")?;
    rocm_d2d_copy(dst_ptr, src_ptr, n_bytes, stream, dev_idx)
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
    let device = k_pool.device();
    let dev_idx = dev_index(device)?;
    let row_elems = num_kv_heads * head_dim;
    let bpe = KtDType::BF16.size_in_bytes();
    let row_bytes = row_elems * bpe;
    let slot_byte_off = (slot * row_elems * bpe) as u64;

    let k_src = kiln_kt_bridge::rocm_input_device_ptr(&rocm_contig(k)?, KtDType::BF16, "k")?;
    let v_src = kiln_kt_bridge::rocm_input_device_ptr(&rocm_contig(v)?, KtDType::BF16, "v")?;
    let kp_dst = kiln_kt_bridge::rocm_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let vp_dst = kiln_kt_bridge::rocm_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    let stream = kiln_kt_bridge::rocm_stream_raw_of(k_pool, "k_pool")?;

    rocm_d2d_copy(kp_dst + slot_byte_off, k_src, row_bytes, stream, dev_idx)?;
    rocm_d2d_copy(vp_dst + slot_byte_off, v_src, row_bytes, stream, dev_idx)?;
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
    let device = k_pool.device();
    let dev_idx = dev_index(device)?;
    let row_elems = num_kv_heads * head_dim;
    let bpe = KtDType::BF16.size_in_bytes();
    let row_bytes = row_elems * bpe;

    let slots_host = map_kt(kiln_tensor::rocm_to_host_copy(slots))?;
    let slot_vec: Vec<u32> = map_kt(slots_host.to_vec::<u32>())?;

    let k_c = rocm_contig(k)?;
    let v_c = rocm_contig(v)?;
    let k_src_base = kiln_kt_bridge::rocm_input_device_ptr(&k_c, KtDType::BF16, "k")?;
    let v_src_base = kiln_kt_bridge::rocm_input_device_ptr(&v_c, KtDType::BF16, "v")?;
    let kp_dst = kiln_kt_bridge::rocm_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let vp_dst = kiln_kt_bridge::rocm_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    let stream = kiln_kt_bridge::rocm_stream_raw_of(k_pool, "k_pool")?;

    for (r, &slot) in slot_vec.iter().enumerate().take(batch) {
        let src_off = (r * row_elems * bpe) as u64;
        let dst_off = (slot as usize * row_elems * bpe) as u64;
        rocm_d2d_copy(
            kp_dst + dst_off,
            k_src_base + src_off,
            row_bytes,
            stream,
            dev_idx,
        )?;
        rocm_d2d_copy(
            vp_dst + dst_off,
            v_src_base + src_off,
            row_bytes,
            stream,
            dev_idx,
        )?;
    }
    Ok(())
}

// ============================================================================
// Backward
// ============================================================================

/// ROCm backward for `flash_attn_bwd_kt`. The bf16/head_dim=128/256 path uses
/// an exact materialized BLASLt composite when the live score budget and BLASLt
/// heuristics allow it, otherwise a native exact HIP kernel consumes the saved
/// forward output and `softmax_lse`. Other shapes fall back to the composite
/// that recomputes scores and produces `(dq, dk, dv)` via matmuls:
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
    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (sk, hk) = (k.shape()[1], k.shape()[2]);

    if rocm_native_bwd_preferred(b, h, sq, sk, d) {
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
    if rocm_materialized_bwd_enabled(b, h, sq, sk) {
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
    if rocm_online_bwd_enabled() {
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
            Err(err) => online_err = Some(err),
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
        materialized_score_budget_bytes(),
    ) {
        Ok(result) => return Ok(result),
        Err(tiled_err) => {
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
    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (sk, hk) = (k.shape()[1], k.shape()[2]);
    if hk == h {
        return flash_attn_bwd_rocm(dout, q, k, v, out, softmax_lse, softmax_scale, causal);
    }
    if hk == 0 || h % hk != 0 {
        return Err(FlashAttnError::Msg(format!(
            "rocm-sdpa collapsed bwd: invalid GQA heads h={h} hk={hk}"
        )));
    }

    if rocm_collapsed_gqa_bwd_enabled()
        && rocm_native_direct_collapsed_gqa_bwd_enabled()
        && rocm_native_bwd_preferred(b, h, sq, sk, d)
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

    if !rocm_native_bwd_preferred(b, h, sq, sk, d) && rocm_materialized_bwd_enabled(b, h, sq, sk) {
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
            Err(_) => {
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
    let debug_finite = debug_rocm_flash_finite_checks();
    let bh = b * h;
    let trace = env_bool("KILN_TRACE_ROCM_FLASH_BWD").unwrap_or(false);
    let total_started = std::time::Instant::now();
    let mut prep_ms = 0.0;
    let mut prob_ms = 0.0;
    let mut dv_ms = 0.0;
    let mut dp_ms = 0.0;
    let mut softmax_ms = 0.0;
    let mut dq_ms = 0.0;
    let mut dk_ms = 0.0;
    let mut out_ms = 0.0;

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
    rocm_timed_finish(trace, device, &mut prep_ms, total_started)?;

    // Recompute scores + softmax p (same as forward). Keep score-sized
    // temporaries scoped tightly; long-context replay relies on these drops to
    // stay under the live materialized-score budget.
    let p = {
        let started = std::time::Instant::now();
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
        rocm_timed_finish(trace, device, &mut prob_ms, started)?;
        p
    };

    // dv = p^T @ dout   -> [b*h, sk, d]
    let dv3 = {
        let started = std::time::Instant::now();
        let pt = rocm_contig(&map_kt(p.transpose(1, 2))?)?; // [b*h, sk, sq]
        let dv3 = rocm_matmul_split_inner_f32(&pt, &do3, device, debug_finite, "bwd dv3")?; // [b*h, sk, d]
        drop(pt);
        rocm_timed_finish(trace, device, &mut dv_ms, started)?;
        dv3
    };

    // dp = dout @ v^T   -> [b*h, sq, sk]
    let dp = {
        let started = std::time::Instant::now();
        let vt3 = rocm_contig(&map_kt(v3.transpose(1, 2))?)?; // [b*h, d, sk]
        let dp = map_kt(kiln_tensor::rocm_matmul(&do3, &vt3))?; // [b*h, sq, sk]
        drop(vt3);
        rocm_timed_finish(trace, device, &mut dp_ms, started)?;
        dp
    };

    // ds = p * (dp - rowsum(dp * p))
    // rowsum over last axis: rocm_sum_axis(dp*p, 2) -> [b*h, sq]; subtract via
    // ROCm broadcast. All score-sized elementwise work stays on-device.
    let ds = {
        let started = std::time::Instant::now();
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
        rocm_timed_finish(trace, device, &mut softmax_ms, started)?;
        ds
    };

    // dq = ds @ k * scale   -> [b*h, sq, d]
    let started = std::time::Instant::now();
    let dq3 = rocm_matmul_split_inner_f32(&ds, &k3, device, debug_finite, "bwd dq3")?; // [b*h, sq, d]
    let dq3 = map_kt(kiln_tensor::rocm_scalar_op(&dq3, SCALAR_MUL, softmax_scale))?;
    rocm_timed_finish(trace, device, &mut dq_ms, started)?;

    // dk = ds^T @ q * scale -> [b*h, sk, d]
    let started = std::time::Instant::now();
    let dst = rocm_contig(&map_kt(ds.transpose(1, 2))?)?; // [b*h, sk, sq]
    drop(ds);
    let dk3 = rocm_matmul_split_inner_f32(&dst, &q3, device, debug_finite, "bwd dk3")?; // [b*h, sk, d]
    drop(dst);
    let dk3 = map_kt(kiln_tensor::rocm_scalar_op(&dk3, SCALAR_MUL, softmax_scale))?;
    rocm_timed_finish(trace, device, &mut dk_ms, started)?;

    // Reshape back to BF16. The collapsed path keeps the exact same F32
    // arithmetic for each query head, then sums query-head groups before the
    // BF16 cast so it matches the model tape's previous F32 collapse.
    let started = std::time::Instant::now();
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
    rocm_timed_finish(trace, device, &mut out_ms, started)?;
    if trace {
        eprintln!(
            "kiln_rocm_flash_bwd_timing path=materialized batch={b} seq_q={sq} seq_k={sk} \
             heads={h} kv_heads={hk} head_dim={d} collapsed_gqa={collapse_gqa} \
             prep_ms={prep_ms:.3} prob_ms={prob_ms:.3} dv_ms={dv_ms:.3} \
             dp_ms={dp_ms:.3} softmax_ms={softmax_ms:.3} dq_ms={dq_ms:.3} \
             dk_ms={dk_ms:.3} out_ms={out_ms:.3} total_ms={:.3}",
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }

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
    budget_bytes: usize,
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
    let debug_finite = debug_rocm_flash_finite_checks();
    let bh = b * h;
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
        let tile_len = query_tile_len_for_budget_with_scratch(
            b,
            h,
            sk,
            remaining,
            budget_bytes,
            MATERIALIZED_BWD_SCORE_SCRATCH_BUFFERS,
        );
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
            let dv =
                rocm_matmul_split_inner_f32(&pt, &do_tile, device, debug_finite, "bwd tiled dv")?; // [b*h, sk, d]
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

        let dq_tile = rocm_matmul_split_inner_f32(&ds, &k3, device, debug_finite, "bwd tiled dq")?; // [b*h, tile_len, d]
        let dq_tile = map_kt(kiln_tensor::rocm_scalar_op(
            &dq_tile,
            SCALAR_MUL,
            softmax_scale,
        ))?;
        dq_tiles.push(bhsd3_to_bshd_bf16(&dq_tile, b, h, tile_len, d)?);

        let dk_tile = {
            let dst = rocm_contig(&map_kt(ds.transpose(1, 2))?)?; // [b*h, sk, tile_len]
            let dk =
                rocm_matmul_split_inner_f32(&dst, &q_tile, device, debug_finite, "bwd tiled dk")?; // [b*h, sk, d]
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
    let bh = b * h;
    let causal_offset = sk as isize - sq as isize;
    let trace = env_bool("KILN_TRACE_ROCM_FLASH_ONLINE_BWD").unwrap_or(false);
    let total_started = std::time::Instant::now();
    let mut prep_ms = 0.0;
    let mut qprep_ms = 0.0;
    let mut qk_ms = 0.0;
    let mut prob_ms = 0.0;
    let mut dv_ms = 0.0;
    let mut dp_ms = 0.0;
    let mut softmax_ms = 0.0;
    let mut dq_ms = 0.0;
    let mut dk_ms = 0.0;
    let mut out_ms = 0.0;
    let mut blocks = 0usize;
    let mut q_tiles = 0usize;

    let started = std::time::Instant::now();
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

    let dk_acc = map_kt(KtTensor::zeros(vec![bh, sk, d], KtDType::F32, device))?;
    let dv_acc = map_kt(KtTensor::zeros(vec![bh, sk, d], KtDType::F32, device))?;
    let mut dq_tiles = Vec::new();
    rocm_timed_finish(trace, device, &mut prep_ms, started)?;

    let mut q_start = 0usize;
    while q_start < sq {
        q_tiles += 1;
        let remaining_q = sq - q_start;
        let max_k = if causal {
            causal_block_limit(sk, q_start, remaining_q, causal_offset).max(1)
        } else {
            sk
        };
        let (q_tile_cap, key_tile_cap) = online_tile_lens(device, bh, remaining_q, max_k);
        let q_len = remaining_q.min(q_tile_cap).max(1);

        let started = std::time::Instant::now();
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
        let dq_acc = map_kt(KtTensor::zeros(vec![bh, q_len, d], KtDType::F32, device))?;
        rocm_timed_finish(trace, device, &mut qprep_ms, started)?;

        let block_limit = if causal {
            causal_block_limit(sk, q_start, q_len, causal_offset)
        } else {
            sk
        };
        let mut k_start = 0usize;
        while k_start < block_limit {
            blocks += 1;
            let k_len = (block_limit - k_start).min(key_tile_cap).max(1);
            let started = std::time::Instant::now();
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
            rocm_timed_finish(trace, device, &mut qk_ms, started)?;
            let started = std::time::Instant::now();
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
            rocm_timed_finish(trace, device, &mut prob_ms, started)?;

            let started = std::time::Instant::now();
            let probs_bf = rocm_cast_to(&probs, KtDType::BF16)?;
            let dv_part = map_kt(kiln_tensor::rocm_matmul_lhs_transposed_to_dtype(
                &probs_bf,
                &do_tile_bf,
                KtDType::F32,
            ))?;
            accum_axis1_f32(&dv_acc, &dv_part, bh, sk, k_len, d, k_start)?;
            drop(dv_part);
            rocm_timed_finish(trace, device, &mut dv_ms, started)?;

            let started = std::time::Instant::now();
            let dp = map_kt(kiln_tensor::rocm_matmul_rhs_transposed_to_dtype(
                &do_tile_bf,
                &v_block_bf,
                KtDType::F32,
            ))?;
            rocm_timed_finish(trace, device, &mut dp_ms, started)?;
            let started = std::time::Instant::now();
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
            rocm_timed_finish(trace, device, &mut softmax_ms, started)?;

            let started = std::time::Instant::now();
            let ds_bf = rocm_cast_to(&ds, KtDType::BF16)?;
            let dq_part = map_kt(kiln_tensor::rocm_matmul_to_dtype(
                &ds_bf,
                &k_block_bf,
                KtDType::F32,
            ))?;
            accum_axis1_f32(&dq_acc, &dq_part, bh, q_len, q_len, d, 0)?;
            drop(dq_part);
            drop(k_block_bf);
            rocm_timed_finish(trace, device, &mut dq_ms, started)?;

            let started = std::time::Instant::now();
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
            rocm_timed_finish(trace, device, &mut dk_ms, started)?;

            k_start += k_len;
            rocm_attention_cooperative_yield(device)?;
        }

        let started = std::time::Instant::now();
        dq_tiles.push(bhsd3_to_bshd_bf16(&dq_acc, b, h, q_len, d)?);
        rocm_timed_finish(trace, device, &mut out_ms, started)?;
        q_start += q_len;
    }

    let started = std::time::Instant::now();
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
    rocm_timed_finish(trace, device, &mut out_ms, started)?;
    if trace {
        eprintln!(
            "kiln_rocm_flash_online_bwd_timing batch={b} seq_q={sq} seq_k={sk} heads={h} \
             kv_heads={hk} head_dim={d} q_tiles={q_tiles} blocks={blocks} \
             prep_ms={prep_ms:.3} qprep_ms={qprep_ms:.3} qk_ms={qk_ms:.3} \
             prob_ms={prob_ms:.3} dv_ms={dv_ms:.3} dp_ms={dp_ms:.3} \
             softmax_ms={softmax_ms:.3} dq_ms={dq_ms:.3} dk_ms={dk_ms:.3} \
             out_ms={out_ms:.3} total_ms={:.3}",
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
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
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&q_c, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

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
            stream,
        )
    };

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
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&q_c, "q")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;

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
            stream,
        )
    };

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
        let stream = kiln_kt_bridge::rocm_stream_raw_of(&expanded, "expanded")
            .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
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
    let device = scores.device();
    let scores_ptr = kiln_kt_bridge::rocm_output_device_ptr(&scores);
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&scores, "scores")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
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
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "causal_mask_fill_offset_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let device = scores.device();
    let scores_ptr = kiln_kt_bridge::rocm_output_device_ptr(&scores);
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&scores, "scores")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_scale_mask_f32(
            scores_ptr as *mut _,
            usize_to_i32("scale mask bh", bh)?,
            usize_to_i32("scale mask sq", sq)?,
            usize_to_i32("scale mask sk", sk)?,
            usize_to_i32("scale mask q_start", q_start)?,
            usize_to_i32("scale mask k_start", k_start)?,
            isize_to_i32("scale mask causal_offset", causal_offset)?,
            scale,
            NEG_FILL,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "scale_mask_scores_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let device = scores.device();
    let scores_ptr = kiln_kt_bridge::rocm_output_device_ptr(&scores);
    let row_max_ptr = kiln_kt_bridge::rocm_input_device_ptr(&row_max, KtDType::F32, "row_max")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&scores, "scores")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_exp_mask_f32(
            scores_ptr as *mut _,
            row_max_ptr as *const _,
            usize_to_i32("exp mask bh", bh)?,
            usize_to_i32("exp mask sq", sq)?,
            usize_to_i32("exp mask sk", sk)?,
            usize_to_i32("exp mask q_start", q_start)?,
            usize_to_i32("exp mask k_start", k_start)?,
            isize_to_i32("exp mask causal_offset", causal_offset)?,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "exp_mask_scores_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let device = scores.device();
    let scores_ptr = kiln_kt_bridge::rocm_output_device_ptr(&scores);
    let lse_ptr = kiln_kt_bridge::rocm_input_device_ptr(&lse, KtDType::F32, "lse")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&scores, "scores")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_prob_from_lse_f32(
            scores_ptr as *mut _,
            lse_ptr as *const _,
            usize_to_i32("prob lse bh", bh)?,
            usize_to_i32("prob lse sq", sq)?,
            usize_to_i32("prob lse sk", sk)?,
            usize_to_i32("prob lse total_q", total_q)?,
            usize_to_i32("prob lse q_start", q_start)?,
            usize_to_i32("prob lse k_start", k_start)?,
            isize_to_i32("prob lse causal_offset", causal_offset)?,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "prob_from_lse_scores_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let device = dp.device();
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&dp, "dp")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_softmax_bwd_f32(
            kiln_kt_bridge::rocm_output_device_ptr(&dp) as *mut _,
            kiln_kt_bridge::rocm_input_device_ptr(&p, KtDType::F32, "p")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            kiln_kt_bridge::rocm_input_device_ptr(&d_rows, KtDType::F32, "d_rows")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            usize_to_i32("softmax bwd bh", bh)?,
            usize_to_i32("softmax bwd sq", sq)?,
            usize_to_i32("softmax bwd sk", sk)?,
            usize_to_i32("softmax bwd total_q", total_q)?,
            usize_to_i32("softmax bwd q_start", q_start)?,
            usize_to_i32("softmax bwd k_start", k_start)?,
            isize_to_i32("softmax bwd causal_offset", causal_offset)?,
            scale,
            if causal { 1 } else { 0 },
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "softmax_bwd_scores_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let device = dst.device();
    let stream = kiln_kt_bridge::rocm_stream_raw_of(dst, "dst")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_accum_axis1_f32(
            kiln_kt_bridge::rocm_output_device_ptr(dst) as *mut _,
            kiln_kt_bridge::rocm_input_device_ptr(src, KtDType::F32, "src")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            usize_to_i32("accum bh", bh)?,
            usize_to_i32("accum total_s", total_s)?,
            usize_to_i32("accum tile_s", tile_s)?,
            usize_to_i32("accum head_dim", d)?,
            usize_to_i32("accum s_start", s_start)?,
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "accum_axis1_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let device = row_m.device();
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&row_m, "row_m")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_online_update_state_f32(
            kiln_kt_bridge::rocm_output_device_ptr(&row_m) as *mut _,
            kiln_kt_bridge::rocm_output_device_ptr(&row_l) as *mut _,
            kiln_kt_bridge::rocm_input_device_ptr(&block_m, KtDType::F32, "block_m")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            kiln_kt_bridge::rocm_input_device_ptr(&block_l, KtDType::F32, "block_l")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            kiln_kt_bridge::rocm_output_device_ptr(&alpha) as *mut _,
            kiln_kt_bridge::rocm_output_device_ptr(&beta) as *mut _,
            usize_to_i32("online state rows", rows)?,
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "online_update_state_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let device = acc2.device();
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&acc2, "acc")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_online_update_acc_f32(
            kiln_kt_bridge::rocm_output_device_ptr(&acc2) as *mut _,
            kiln_kt_bridge::rocm_input_device_ptr(&block_out2, KtDType::F32, "block_out")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            kiln_kt_bridge::rocm_input_device_ptr(&alpha, KtDType::F32, "alpha")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            kiln_kt_bridge::rocm_input_device_ptr(&beta, KtDType::F32, "beta")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            usize_to_i32("online acc rows", rows)?,
            usize_to_i32("online acc head_dim", d)?,
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "online_update_acc_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let device = acc2.device();
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&acc2, "acc")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
    let status = unsafe {
        crate::kiln_rocm_flash_online_finalize_f32(
            kiln_kt_bridge::rocm_input_device_ptr(&acc2, KtDType::F32, "acc")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            kiln_kt_bridge::rocm_input_device_ptr(&row_m, KtDType::F32, "row_m")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            kiln_kt_bridge::rocm_input_device_ptr(&row_l, KtDType::F32, "row_l")
                .map_err(|e| FlashAttnError::Msg(e.to_string()))? as *const _,
            kiln_kt_bridge::rocm_output_device_ptr(&out2) as *mut _,
            kiln_kt_bridge::rocm_output_device_ptr(&lse) as *mut _,
            usize_to_i32("online finalize rows", rows)?,
            usize_to_i32("online finalize head_dim", d)?,
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "online_finalize_f32: FFI returned status {status}"
        )));
    }
    rocm_sync_device(device)?;
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
    let stream = kiln_kt_bridge::rocm_stream_raw_of(&a_c, "a")
        .map_err(|e| FlashAttnError::Msg(e.to_string()))?;
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

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn materialized_query_tile_respects_score_element_cap() {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        unsafe {
            std::env::set_var("KILN_ROCM_FLASH_SCORE_TILE_MAX_ELEMENTS", "536870912");
        }
        let tile_len = query_tile_len_for_budget(1, 16, 32_768, 8_192, usize::MAX);
        unsafe {
            std::env::remove_var("KILN_ROCM_FLASH_SCORE_TILE_MAX_ELEMENTS");
        }
        assert!(tile_len > 0);
        assert!(tile_len <= 1_024, "tile_len={tile_len}");
        assert!(
            16usize.saturating_mul(tile_len).saturating_mul(32_768) <= 536_870_912,
            "tile_len={tile_len} still exceeds score element cap"
        );
    }
}
