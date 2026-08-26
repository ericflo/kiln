//! ROCm W8 decode projections.
//!
//! The immutable ROCm kernel profile owns packing and dispatch. The route is
//! default-on in the qualified profile and decode-scoped. It
//! packs BF16 row-major projection weights `[out, in]` into signed int8 plus
//! one F32 scale per output row, then uses a ROCm GEMV kernel for single-token
//! decode. Qualified runtime projection math uses W8A8 for sampled-decode
//! throughput.

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "rocm")]
use std::time::{Duration, Instant};

static ARGMAX_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static ARGMAX_ROWS: AtomicU64 = AtomicU64::new(0);
static SAMPLE_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static SAMPLE_ROWS: AtomicU64 = AtomicU64::new(0);
static SAMPLE_HISTORY_ENTRIES: AtomicU64 = AtomicU64::new(0);
static SAMPLE_W8A16_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static SAMPLE_W8A8_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static DISPATCH_FAILURES: AtomicU64 = AtomicU64::new(0);
static MAX_BATCH_ROWS: AtomicU64 = AtomicU64::new(0);
static MAX_SAMPLE_TOP_K: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct RocmW8LmHeadStats {
    pub argmax_dispatches: u64,
    pub argmax_rows: u64,
    pub sample_dispatches: u64,
    pub sample_rows: u64,
    pub sample_history_entries: u64,
    pub sample_w8a16_dispatches: u64,
    pub sample_w8a8_dispatches: u64,
    pub dispatch_failures: u64,
    pub max_batch_rows: u64,
    pub max_sample_top_k: u64,
}

pub fn stats() -> RocmW8LmHeadStats {
    RocmW8LmHeadStats {
        argmax_dispatches: ARGMAX_DISPATCHES.load(Ordering::Relaxed),
        argmax_rows: ARGMAX_ROWS.load(Ordering::Relaxed),
        sample_dispatches: SAMPLE_DISPATCHES.load(Ordering::Relaxed),
        sample_rows: SAMPLE_ROWS.load(Ordering::Relaxed),
        sample_history_entries: SAMPLE_HISTORY_ENTRIES.load(Ordering::Relaxed),
        sample_w8a16_dispatches: SAMPLE_W8A16_DISPATCHES.load(Ordering::Relaxed),
        sample_w8a8_dispatches: SAMPLE_W8A8_DISPATCHES.load(Ordering::Relaxed),
        dispatch_failures: DISPATCH_FAILURES.load(Ordering::Relaxed),
        max_batch_rows: MAX_BATCH_ROWS.load(Ordering::Relaxed),
        max_sample_top_k: MAX_SAMPLE_TOP_K.load(Ordering::Relaxed),
    }
}

#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
fn observe_batch_rows(rows: usize) {
    MAX_BATCH_ROWS.fetch_max(rows as u64, Ordering::Relaxed);
}

#[derive(Clone, Debug)]
pub struct RocmW8Proj {
    pub q_weight: kiln_tensor::Tensor,
    pub scales: kiln_tensor::Tensor,
    pub k: usize,
    pub n: usize,
}

#[derive(Debug)]
#[cfg(feature = "rocm")]
pub(crate) struct ProfiledRocmW8Sample<T> {
    pub value: T,
    pub readback_duration: Duration,
}

#[cfg(feature = "rocm")]
fn a8_enabled() -> bool {
    crate::rocm_policy::current_rocm_kernel_policy().w8a8_projection
}

#[cfg(feature = "rocm")]
fn a8_sample_enabled() -> bool {
    a8_enabled() && crate::rocm_policy::current_rocm_kernel_policy().w8a8_sampled_lm_head
}

#[cfg(feature = "rocm")]
pub fn swiglu_bf16_enabled(w: &RocmW8Proj) -> bool {
    crate::rocm_policy::current_rocm_kernel_policy().w8_swiglu && w.n % 2 == 0
}

pub fn pack_from_bf16_rows(weight: &kiln_tensor::Tensor) -> Result<Option<RocmW8Proj>> {
    if !matches!(weight.device(), kiln_tensor::Device::Rocm(_)) {
        return Ok(None);
    }
    if weight.dtype() != kiln_tensor::DType::BF16 {
        anyhow::bail!("rocm_w8_proj: weight must be BF16, got {}", weight.dtype());
    }
    let dims = weight.dims();
    if dims.len() != 2 {
        anyhow::bail!("rocm_w8_proj: weight must be [out, in], got {dims:?}");
    }
    let (n, k) = (dims[0], dims[1]);
    if n == 0 || k == 0 {
        anyhow::bail!("rocm_w8_proj: empty weight {dims:?}");
    }
    let device = weight.device();

    let host = weight
        .to_dtype(kiln_tensor::DType::F32)
        .context("rocm_w8_proj: cast weight to f32")?
        .contiguous()
        .context("rocm_w8_proj: contiguous f32 weight")?
        .flatten_all()
        .context("rocm_w8_proj: flatten weight")?
        .to_vec1::<f32>()
        .context("rocm_w8_proj: download weight")?;
    debug_assert_eq!(host.len(), n * k);

    let mut q = Vec::with_capacity(n * k);
    let mut scales = Vec::with_capacity(n);
    for row in host.chunks_exact(k) {
        let mut max_abs = 0.0f32;
        for &v in row {
            max_abs = max_abs.max(v.abs());
        }
        let scale = if max_abs <= 1.0e-12 {
            1.0
        } else {
            max_abs / 127.0
        };
        scales.push(scale);
        let inv = 1.0 / scale;
        for &v in row {
            let rounded = (v * inv).round().clamp(-127.0, 127.0) as i8;
            q.push(rounded as u8);
        }
    }

    let q_weight = kiln_tensor::Tensor::from_vec(q, (n, k))
        .context("rocm_w8_proj: build q_weight")?
        .to_device(device)
        .context("rocm_w8_proj: upload q_weight")?;
    let scales = kiln_tensor::Tensor::from_vec(scales, (n,))
        .context("rocm_w8_proj: build scales")?
        .to_device(device)
        .context("rocm_w8_proj: upload scales")?;

    Ok(Some(RocmW8Proj {
        q_weight,
        scales,
        k,
        n,
    }))
}

#[cfg(feature = "rocm")]
pub fn matmul_bf16(x: &kiln_tensor::Tensor, w: &RocmW8Proj) -> Result<kiln_tensor::Tensor> {
    let out = if a8_enabled() {
        kiln_tensor::rocm_w8a8_gemv_bf16(x, &w.q_weight, &w.scales)
            .map_err(|e| anyhow::anyhow!("rocm_w8_proj: a8 gemv: {e}"))?
    } else {
        kiln_tensor::rocm_w8a16_gemv_bf16(x, &w.q_weight, &w.scales)
            .map_err(|e| anyhow::anyhow!("rocm_w8_proj: gemv: {e}"))?
    };
    Ok(out)
}

#[cfg(feature = "rocm")]
pub fn swiglu_bf16(x: &kiln_tensor::Tensor, w: &RocmW8Proj) -> Result<Option<kiln_tensor::Tensor>> {
    if !swiglu_bf16_enabled(w) {
        return Ok(None);
    }
    let out = if a8_enabled() {
        kiln_tensor::rocm_w8a8_swiglu_bf16(x, &w.q_weight, &w.scales)
            .map_err(|e| anyhow::anyhow!("rocm_w8_proj: a8 swiglu: {e}"))?
    } else {
        kiln_tensor::rocm_w8a16_swiglu_bf16(x, &w.q_weight, &w.scales)
            .map_err(|e| anyhow::anyhow!("rocm_w8_proj: swiglu: {e}"))?
    };
    Ok(Some(out))
}

#[cfg(not(feature = "rocm"))]
pub fn matmul_bf16(_x: &kiln_tensor::Tensor, _w: &RocmW8Proj) -> Result<kiln_tensor::Tensor> {
    anyhow::bail!("rocm_w8_proj::matmul_bf16 requires the rocm feature")
}

#[cfg(feature = "rocm")]
pub fn argmax_bf16(x: &kiln_tensor::Tensor, w: &RocmW8Proj) -> Result<u32> {
    let idx = kiln_tensor::rocm_w8a16_gemv_argmax_bf16(x, &w.q_weight, &w.scales)
        .map_err(|e| anyhow::anyhow!("rocm_w8_proj: gemv_argmax: {e}"))?;
    let flattened = idx.flatten_all().context("rocm_w8_proj: flatten argmax")?;
    let values = crate::execution_phase::profile_accelerator_readback(flattened.device(), || {
        flattened.to_vec1::<i64>()
    })
    .context("rocm_w8_proj: read argmax")?;
    Ok(values[0] as u32)
}

pub const BATCH_SAMPLE_TOP_K_MAX: u32 = kiln_tensor::ROCM_W8_BATCH_SAMPLE_TOP_K_MAX;

/// Batched greedy LM-head projection and argmax. The output tensor is copied
/// to the host once for the complete batch.
#[cfg(feature = "rocm")]
pub fn argmax_batch_bf16(x: &kiln_tensor::Tensor, w: &RocmW8Proj) -> Result<Vec<u32>> {
    let dims = x.dims();
    anyhow::ensure!(dims.len() >= 2, "rocm w8 argmax batch requires rank >=2");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    anyhow::ensure!(rows > 0, "rocm w8 argmax batch requires rows");
    let ones = vec![1.0f32; rows];
    let zeros = vec![0.0f32; rows];
    let top_k = vec![0u32; rows];
    let seeds = vec![0u64; rows];
    let result = (|| {
        let indices = kiln_tensor::rocm_w8a16_gemv_sample_batch_bf16(
            x,
            &w.q_weight,
            &w.scales,
            &[],
            &[],
            &[],
            &ones,
            &zeros,
            &zeros,
            &zeros,
            &top_k,
            &ones,
            &zeros,
            &seeds,
        )
        .map_err(|error| anyhow::anyhow!("rocm_w8_proj: batched argmax: {error}"))?;
        read_batch_tokens(indices, rows, w.n, "batched argmax")
    })();
    match result {
        Ok(tokens) => {
            ARGMAX_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            ARGMAX_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
            observe_batch_rows(rows);
            Ok(tokens)
        }
        Err(error) => {
            DISPATCH_FAILURES.fetch_add(1, Ordering::Relaxed);
            Err(error)
        }
    }
}

#[cfg(not(feature = "rocm"))]
pub fn argmax_batch_bf16(_x: &kiln_tensor::Tensor, _w: &RocmW8Proj) -> Result<Vec<u32>> {
    anyhow::bail!("rocm_w8_proj::argmax_batch_bf16 requires the rocm feature")
}

/// Batched W8 LM-head projection, penalties, bounded filtering, and sampling.
/// The kernel accepts mixed greedy and sampled rows and returns all token IDs
/// through one device-to-host copy.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn sample_batch_bf16(
    x: &kiln_tensor::Tensor,
    w: &RocmW8Proj,
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<Vec<u32>> {
    let profiled = sample_batch_bf16_profiled(
        x,
        w,
        history_rows,
        history_indices,
        history_counts,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
        temperatures,
        top_k,
        top_p,
        min_p,
        seeds,
    )?;
    crate::execution_phase::observe_profiled_readback(profiled.readback_duration);
    Ok(profiled.value)
}

#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_batch_bf16_profiled(
    x: &kiln_tensor::Tensor,
    w: &RocmW8Proj,
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<ProfiledRocmW8Sample<Vec<u32>>> {
    let rows = temperatures.len();
    let use_w8a8 = a8_sample_enabled();
    let result = (|| {
        let indices = if use_w8a8 {
            kiln_tensor::rocm_w8a8_gemv_sample_batch_bf16(
                x,
                &w.q_weight,
                &w.scales,
                history_rows,
                history_indices,
                history_counts,
                repetition_penalties,
                presence_penalties,
                frequency_penalties,
                temperatures,
                top_k,
                top_p,
                min_p,
                seeds,
            )
            .map_err(|error| anyhow::anyhow!("rocm_w8_proj: W8A8 batched sample: {error}"))?
        } else {
            kiln_tensor::rocm_w8a16_gemv_sample_batch_bf16(
                x,
                &w.q_weight,
                &w.scales,
                history_rows,
                history_indices,
                history_counts,
                repetition_penalties,
                presence_penalties,
                frequency_penalties,
                temperatures,
                top_k,
                top_p,
                min_p,
                seeds,
            )
            .map_err(|error| anyhow::anyhow!("rocm_w8_proj: W8A16 batched sample: {error}"))?
        };
        read_batch_tokens_profiled(indices, rows, w.n, "batched sample")
    })();
    match result {
        Ok(tokens) => {
            SAMPLE_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            SAMPLE_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
            SAMPLE_HISTORY_ENTRIES.fetch_add(history_rows.len() as u64, Ordering::Relaxed);
            if use_w8a8 {
                SAMPLE_W8A8_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            } else {
                SAMPLE_W8A16_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            }
            observe_batch_rows(rows);
            if let Some(max_top_k) = top_k.iter().copied().max() {
                MAX_SAMPLE_TOP_K.fetch_max(max_top_k as u64, Ordering::Relaxed);
            }
            Ok(tokens)
        }
        Err(error) => {
            DISPATCH_FAILURES.fetch_add(1, Ordering::Relaxed);
            Err(error)
        }
    }
}

#[cfg(feature = "rocm")]
fn read_batch_tokens(
    indices: kiln_tensor::Tensor,
    expected_rows: usize,
    vocab_size: usize,
    operation: &str,
) -> Result<Vec<u32>> {
    let profiled = read_batch_tokens_profiled(indices, expected_rows, vocab_size, operation)?;
    crate::execution_phase::observe_profiled_readback(profiled.readback_duration);
    Ok(profiled.value)
}

#[cfg(feature = "rocm")]
fn read_batch_tokens_profiled(
    indices: kiln_tensor::Tensor,
    expected_rows: usize,
    vocab_size: usize,
    operation: &str,
) -> Result<ProfiledRocmW8Sample<Vec<u32>>> {
    let readback_started = Instant::now();
    let values = indices
        .to_vec1::<i64>()
        .with_context(|| format!("rocm_w8_proj: read {operation}"))?;
    let readback_duration = readback_started.elapsed();
    anyhow::ensure!(
        values.len() == expected_rows,
        "rocm w8 {operation} returned {} rows, expected {expected_rows}",
        values.len()
    );
    let tokens = values
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            anyhow::ensure!(
                value >= 0 && (value as usize) < vocab_size,
                "rocm w8 {operation} row {row} returned token {value} outside vocab {vocab_size}"
            );
            Ok(value as u32)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ProfiledRocmW8Sample {
        value: tokens,
        readback_duration,
    })
}

#[cfg(not(feature = "rocm"))]
#[allow(clippy::too_many_arguments)]
pub fn sample_batch_bf16(
    _x: &kiln_tensor::Tensor,
    _w: &RocmW8Proj,
    _history_rows: &[u32],
    _history_indices: &[u32],
    _history_counts: &[u32],
    _repetition_penalties: &[f32],
    _presence_penalties: &[f32],
    _frequency_penalties: &[f32],
    _temperatures: &[f32],
    _top_k: &[u32],
    _top_p: &[f32],
    _min_p: &[f32],
    _seeds: &[u64],
) -> Result<Vec<u32>> {
    anyhow::bail!("rocm_w8_proj::sample_batch_bf16 requires the rocm feature")
}

#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn gumbel_sample_bf16(
    x: &kiln_tensor::Tensor,
    w: &RocmW8Proj,
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    temperature: f32,
    seed: u64,
) -> Result<u32> {
    let profiled = gumbel_sample_bf16_profiled(
        x,
        w,
        history_indices,
        history_counts,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        temperature,
        seed,
    )?;
    crate::execution_phase::observe_profiled_readback(profiled.readback_duration);
    Ok(profiled.value)
}

#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gumbel_sample_bf16_profiled(
    x: &kiln_tensor::Tensor,
    w: &RocmW8Proj,
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    temperature: f32,
    seed: u64,
) -> Result<ProfiledRocmW8Sample<u32>> {
    let idx = if a8_sample_enabled() {
        kiln_tensor::rocm_w8a8_gemv_gumbel_sample_bf16(
            x,
            &w.q_weight,
            &w.scales,
            history_indices,
            history_counts,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            temperature,
            seed,
        )
        .map_err(|e| anyhow::anyhow!("rocm_w8_proj: a8 gemv_gumbel_sample: {e}"))?
    } else {
        kiln_tensor::rocm_w8a16_gemv_gumbel_sample_bf16(
            x,
            &w.q_weight,
            &w.scales,
            history_indices,
            history_counts,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            temperature,
            seed,
        )
        .map_err(|e| anyhow::anyhow!("rocm_w8_proj: gemv_gumbel_sample: {e}"))?
    };
    let flattened = idx
        .flatten_all()
        .context("rocm_w8_proj: flatten gumbel sample")?;
    let readback_started = Instant::now();
    let values = flattened
        .to_vec1::<i64>()
        .context("rocm_w8_proj: read gumbel sample")?;
    let readback_duration = readback_started.elapsed();
    Ok(ProfiledRocmW8Sample {
        value: values[0] as u32,
        readback_duration,
    })
}

#[cfg(not(feature = "rocm"))]
#[allow(clippy::too_many_arguments)]
pub fn gumbel_sample_bf16(
    _x: &kiln_tensor::Tensor,
    _w: &RocmW8Proj,
    _history_indices: &[u32],
    _history_counts: &[u32],
    _repetition_penalty: f32,
    _presence_penalty: f32,
    _frequency_penalty: f32,
    _temperature: f32,
    _seed: u64,
) -> Result<u32> {
    anyhow::bail!("rocm_w8_proj::gumbel_sample_bf16 requires the rocm feature")
}

#[cfg(not(feature = "rocm"))]
pub fn argmax_bf16(_x: &kiln_tensor::Tensor, _w: &RocmW8Proj) -> Result<u32> {
    anyhow::bail!("rocm_w8_proj::argmax_bf16 requires the rocm feature")
}

impl RocmW8Proj {
    /// Training-session residency companion to
    /// `GpuWeights::to_device_deep`: deep-copy the packed weight + scales
    /// onto `device`; `k`/`n` are metadata.
    pub fn to_device_deep(&self, device: kiln_tensor::Device) -> anyhow::Result<Self> {
        Ok(Self {
            q_weight: self
                .q_weight
                .to_device(device)
                .map_err(|e| anyhow::anyhow!("w8 q_weight to_device: {e}"))?,
            scales: self
                .scales
                .to_device(device)
                .map_err(|e| anyhow::anyhow!("w8 scales to_device: {e}"))?,
            k: self.k,
            n: self.n,
        })
    }
}

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use super::*;

    #[test]
    fn profiled_token_reader_preserves_rocm_batch_values() -> Result<()> {
        if !kiln_tensor::rocm_is_available() {
            eprintln!("no ROCm device available; skipping profiled W8 readback test");
            return Ok(());
        }
        let indices = kiln_tensor::Tensor::from_vec_on(
            kiln_tensor::Device::Rocm(0),
            vec![3_i64, 1, 4, 1],
            vec![4],
        )?;
        let profiled = read_batch_tokens_profiled(indices, 4, 8, "profile test")?;
        assert_eq!(profiled.value, vec![3, 1, 4, 1]);
        Ok(())
    }

    #[test]
    fn w8_greedy_argmax_reports_request_owned_readback() -> Result<()> {
        if !kiln_tensor::rocm_is_available() {
            eprintln!("no ROCm device available; skipping W8 greedy readback test");
            return Ok(());
        }
        let device = kiln_tensor::Device::Rocm(0);
        let mut weight = vec![0.0_f32; 8 * 8];
        for index in 0..8 {
            weight[index * 8 + index] = 1.0;
        }
        let weight = kiln_tensor::Tensor::new(weight.as_slice(), &device)?
            .reshape((8, 8))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        let packed = pack_from_bf16_rows(&weight)?.context("pack W8 identity")?;
        let activation =
            kiln_tensor::Tensor::new(&[0.0_f32, 1.0, 2.0, 7.0, 3.0, 4.0, 5.0, 6.0], &device)?
                .reshape((1, 1, 8))?
                .to_dtype(kiln_tensor::DType::BF16)?;

        let profiled = crate::execution_phase::profile_readback_invocation(|| {
            argmax_bf16(&activation, &packed)
        })?;
        assert_eq!(profiled.value, 3);
        assert!(
            profiled
                .readback_duration
                .is_some_and(|duration| !duration.is_zero()),
            "W8 greedy argmax must expose its existing scalar readback"
        );
        Ok(())
    }
}
