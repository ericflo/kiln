//! Marlin W4A16 quantized projection helpers (forward-only).
//!
//! This module packages the `kiln-marlin-gemm` kernel for use as a drop-in
//! replacement for a BF16 `Linear` layer in the model forward path. Today it
//! is wired into `q_proj` behind the `KILN_W4A16` env flag. Other projections
//! (`k_proj`, `v_proj`, `o_proj`, MLP) stay on the existing BF16 matmul path.
//!
//! # Storage
//!
//! [`MarlinPackedProj`] holds the packed Marlin weight + scales on GPU, plus
//! the `k`/`n`/`groupsize` metadata the kernel needs on every call. It is
//! built once at model-load time via [`pack_from_bf16`] and reused for every
//! forward pass.
//!
//! # Activation path
//!
//! [`matmul_bf16`] takes a BF16 activation of shape `[.., k]`, flattens it to
//! `[m, k]` for the kernel, runs the Marlin GEMM, and reshapes back to
//! `[.., n]`. This matches the contract of the existing BF16 `broadcast_matmul`
//! against a pre-transposed weight.

use anyhow::Result;
#[cfg(feature = "cuda")]
use anyhow::Context;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::Tensor;

/// A q_proj / projection weight packed into Marlin's W4A16 layout.
///
/// All tensors live on the same CUDA device as the source model weights. The
/// struct is `Send + Sync` via the underlying `Tensor` handles.
#[derive(Debug)]
pub struct MarlinPackedProj {
    /// Packed 4-bit weights in Marlin's tiled/permuted layout.
    /// Shape: `[k / 16, n * 16 / 8]`, dtype `i32`.
    pub b_packed: Tensor,
    /// Per-group scales, Marlin-permuted. Shape `[k / groupsize, n]`, dtype `f16`.
    pub scales: Tensor,
    /// Marlin groupsize sentinel: `-1` (per-column) or `128`.
    pub groupsize: i32,
    /// Input feature dim (rows of the original `[k, n]` weight).
    pub k: usize,
    /// Output feature dim (cols of the original `[k, n]` weight).
    pub n: usize,
}

/// Marlin kernel shape constraints (see upstream `marlin/__init__.py`).
pub fn shape_is_supported(k: usize, n: usize) -> bool {
    k % 128 == 0 && n % 256 == 0
}

/// Pack a BF16 projection weight into Marlin's W4A16 layout.
///
/// `weight_t` is the pre-transposed projection tensor (the same `q_proj_t`
/// the BF16 forward path already uses), i.e. shape `[k, n] = [in, out]`. We
/// download it to CPU, run the pure-Rust packer, then upload the packed
/// int32 tensor and permuted f16 scales back to the source device.
///
/// On non-CUDA builds this returns `Ok(None)` so the caller can keep the
/// same control flow. The kernel itself only exists when `cuda` is on.
#[cfg(feature = "cuda")]
pub fn pack_from_bf16(weight_t: &Tensor, groupsize: i32) -> Result<Option<MarlinPackedProj>> {
    let device = weight_t.device();
    if !device.is_cuda() {
        return Ok(None);
    }
    let (k, n) = weight_t
        .dims2()
        .context("marlin_proj: weight_t must be a 2D [k, n] tensor")?;
    if !shape_is_supported(k, n) {
        // Caller logs; skipping is not an error. Marlin only supports
        // k%128 && n%256 for its instantiated tile shapes.
        return Ok(None);
    }
    if !(groupsize == -1 || groupsize == 128) {
        anyhow::bail!("marlin_proj: groupsize must be -1 or 128 (got {groupsize})");
    }
    if groupsize == 128 && k % 128 != 0 {
        anyhow::bail!(
            "marlin_proj: k={k} not divisible by groupsize=128"
        );
    }

    // The Marlin packer needs f32 on host. Go through a cast on-device (to
    // avoid a bf16 CPU dependency) and then copy to CPU.
    let w_f32 = weight_t
        .to_dtype(DType::F32)
        .context("marlin_proj: cast weight_t to f32")?
        .contiguous()
        .context("marlin_proj: contiguous f32")?;
    let host_weight = w_f32
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .context("marlin_proj: download weight to host")?;
    debug_assert_eq!(host_weight.len(), k * n);

    let (b_packed_vec, scales_vec, _dequant) =
        kiln_marlin_gemm::pack::quantize_and_pack(&host_weight, k, n, groupsize as i64);

    let num_groups = if groupsize == -1 { 1 } else { k / groupsize as usize };

    let b_packed = Tensor::from_vec(b_packed_vec, (k / 16, n * 16 / 8), &candle_core::Device::Cpu)
        .context("marlin_proj: build host b_packed tensor")?
        .to_device(device)
        .context("marlin_proj: upload b_packed to device")?;
    let scales = Tensor::from_vec(scales_vec, (num_groups, n), &candle_core::Device::Cpu)
        .context("marlin_proj: build host scales tensor")?
        .to_device(device)
        .context("marlin_proj: upload scales to device")?;

    Ok(Some(MarlinPackedProj {
        b_packed,
        scales,
        groupsize,
        k,
        n,
    }))
}

/// Non-CUDA stub: Marlin kernel is CUDA-only, so packing always returns
/// `None`. Keeping this here means the call sites in the loader do not need
/// their own `#[cfg]` arms.
#[cfg(not(feature = "cuda"))]
pub fn pack_from_bf16(_weight_t: &Tensor, _groupsize: i32) -> Result<Option<MarlinPackedProj>> {
    Ok(None)
}

/// Run `x @ W` against a Marlin-packed projection.
///
/// `x` may be 2D `[m, k]` or 3D `[batch, seq, k]`; the result matches the
/// input rank with last dim `n`. Matches the shape contract of the existing
/// `linear_with_lora_t` BF16 matmul it is replacing.
#[cfg(feature = "cuda")]
pub fn matmul_bf16(x: &Tensor, w: &MarlinPackedProj) -> Result<Tensor> {
    let rank = x.rank();
    let out = match rank {
        2 => {
            let (m, k) = x.dims2().context("marlin_proj: x must be [m, k] when 2D")?;
            if k != w.k {
                anyhow::bail!(
                    "marlin_proj: x last-dim {k} != packed weight k {}",
                    w.k
                );
            }
            let x = x.contiguous().context("marlin_proj: x contiguous")?;
            let y = kiln_marlin_gemm::marlin_w4a16_gemm(&x, &w.b_packed, &w.scales, w.groupsize)
                .context("marlin_proj: kernel call (2D)")?;
            debug_assert_eq!(y.dims(), &[m, w.n]);
            y
        }
        3 => {
            let (batch, seq, k) = x
                .dims3()
                .context("marlin_proj: x must be [batch, seq, k] when 3D")?;
            if k != w.k {
                anyhow::bail!(
                    "marlin_proj: x last-dim {k} != packed weight k {}",
                    w.k
                );
            }
            let x2 = x
                .reshape((batch * seq, k))
                .context("marlin_proj: reshape x [batch*seq, k]")?
                .contiguous()
                .context("marlin_proj: x2 contiguous")?;
            let y2 =
                kiln_marlin_gemm::marlin_w4a16_gemm(&x2, &w.b_packed, &w.scales, w.groupsize)
                    .context("marlin_proj: kernel call (3D flat)")?;
            y2.reshape((batch, seq, w.n))
                .context("marlin_proj: reshape output [batch, seq, n]")?
        }
        other => anyhow::bail!("marlin_proj: unsupported activation rank {other}"),
    };
    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn matmul_bf16(_x: &Tensor, _w: &MarlinPackedProj) -> Result<Tensor> {
    anyhow::bail!("marlin_proj::matmul_bf16 requires the `cuda` feature")
}

/// Check the `KILN_W4A16` env var.
///
/// `KILN_W4A16=1` (or `true`, case-insensitive) enables the Marlin path. Any
/// other value, or the var being unset, keeps the BF16 baseline.
pub fn env_enabled() -> bool {
    match std::env::var("KILN_W4A16") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes"
        }
        Err(_) => false,
    }
}
