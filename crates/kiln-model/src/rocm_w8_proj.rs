//! ROCm W8 decode projections.
//!
//! The immutable ROCm kernel profile owns packing and dispatch. The route is
//! default-on in the qualified profile and decode-scoped. It
//! packs BF16 row-major projection weights `[out, in]` into signed int8 plus
//! one F32 scale per output row, then uses a ROCm GEMV kernel for single-token
//! decode. Qualified runtime projection math uses W8A8 for sampled-decode
//! throughput.

use anyhow::{Context, Result};

#[derive(Clone, Debug)]
pub struct RocmW8Proj {
    pub q_weight: kiln_tensor::Tensor,
    pub scales: kiln_tensor::Tensor,
    pub k: usize,
    pub n: usize,
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
    let values = idx
        .flatten_all()
        .context("rocm_w8_proj: flatten argmax")?
        .to_vec1::<i64>()
        .context("rocm_w8_proj: read argmax")?;
    Ok(values[0] as u32)
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
    let values = idx
        .flatten_all()
        .context("rocm_w8_proj: flatten gumbel sample")?
        .to_vec1::<i64>()
        .context("rocm_w8_proj: read gumbel sample")?;
    Ok(values[0] as u32)
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
