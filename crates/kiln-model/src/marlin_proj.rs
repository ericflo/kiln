//! Marlin W4A16 quantized projection helpers (forward-only).
//!
//! This module packages the `kiln-marlin-gemm` kernel for use as a drop-in
//! replacement for a BF16 `Linear` layer in the model forward path. A closed
//! process-lifetime policy selects the supported projection set before model
//! weights are uploaded.
//!
//! # Storage
//!
//! [`MarlinPackedProj`] holds the packed Marlin weight + scales on GPU, plus
//! the `k`/`n`/`groupsize` metadata the kernel needs on every call. It is
//! built once at model-load time via [`pack_from_bf16_batch`] and reused for every
//! forward pass.
//!
//! # Activation path
//!
//! [`matmul_bf16_kt`] takes a BF16 activation of shape `[.., k]`, flattens it to
//! `[m, k]` for the kernel, runs the Marlin GEMM, and reshapes back to
//! `[.., n]`. This matches the contract of the existing BF16 `broadcast_matmul`
//! against a pre-transposed weight.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "cuda")]
use half::f16;

/// A q_proj / projection weight packed into Marlin's W4A16 layout.
///
/// All tensors live on the same CUDA device as the source model weights. The
/// struct is `Send + Sync` via the underlying `kiln_tensor::Tensor` handles.
///
/// # #1082 type-flip
///
/// `b_packed`/`scales` are stored as kt (`kiln_tensor::Tensor`) so the kt-API
/// GEMM (`kiln_marlin_gemm::marlin_w4a16_gemm_kt`, optional dependency) consumes them directly on
/// the decode hot path. The candle→kt bridge happens **once** at pack time in
/// `upload_packed` (a single device-to-device copy at model load), not on
/// every per-token decode call.
#[derive(Clone, Debug)]
pub struct MarlinPackedProj {
    /// Packed 4-bit weights in Marlin's tiled/permuted layout.
    /// Shape: `[k / 16, n * 16 / 8]`. Built from candle's `I32` packed
    /// buffer; the kt-side dtype is `U32` (same 4-byte layout — see
    /// `kiln_kt_bridge::candle_dtype_to_kt`'s I32→U32 mapping).
    pub b_packed: kiln_tensor::Tensor,
    /// Per-group scales, Marlin-permuted. Shape `[k / groupsize, n]`, dtype `f16`.
    pub scales: kiln_tensor::Tensor,
    /// Marlin groupsize sentinel: `-1` (per-column) or `128`.
    pub groupsize: i32,
    /// Input feature dim (rows of the original `[k, n]` weight).
    pub k: usize,
    /// Output feature dim (cols of the original `[k, n]` weight).
    pub n: usize,
}

/// Marlin kernel shape constraints (see upstream `marlin/__init__.py`).
pub fn shape_is_supported(k: usize, n: usize) -> bool {
    k.is_multiple_of(128) && n.is_multiple_of(256)
}

/// Intermediate host-side representation of a weight ready to run through
/// Marlin's pure-Rust packer. Produced by [`prepare_pack_job`]; consumed by
/// [`pack_host`]. Splitting the download, pack, and upload phases lets
/// [`pack_from_bf16_batch`] run the CPU-bound pack step for many projections
/// in parallel via rayon while keeping the GPU↔CPU copies sequential.
#[cfg(feature = "cuda")]
pub struct PackJobHost {
    /// `[k, n]` row-major f32 view of the pre-transposed weight.
    pub host_weight: Vec<f32>,
    pub k: usize,
    pub n: usize,
    pub groupsize: i32,
}

/// Marlin-packed buffers sitting on the host, ready for upload via
/// [`upload_packed`].
#[cfg(feature = "cuda")]
pub struct PackedHost {
    pub b_packed_vec: Vec<i32>,
    pub scales_vec: Vec<f16>,
    pub k: usize,
    pub n: usize,
    pub groupsize: i32,
}

/// Phase 1 of Marlin packing: validate shape, cast BF16→F32 on-device, then
/// download to a host f32 buffer. Returns `Ok(None)` if the projection's
/// shape doesn't fit Marlin's tile constraints (same semantics as
/// [`pack_from_bf16`]) or the tensor doesn't live on a CUDA device.
#[cfg(feature = "cuda")]
pub fn prepare_pack_job(
    weight_t: &kiln_tensor::Tensor,
    groupsize: i32,
) -> Result<Option<PackJobHost>> {
    if !matches!(weight_t.device(), kiln_tensor::Device::Cuda(_)) {
        return Ok(None);
    }
    let dims = weight_t.dims();
    if dims.len() != 2 {
        anyhow::bail!("marlin_proj: weight_t must be a 2D [k, n] tensor (got {dims:?})");
    }
    let (k, n) = (dims[0], dims[1]);
    if !shape_is_supported(k, n) {
        return Ok(None);
    }
    if !(groupsize == -1 || groupsize == 128) {
        anyhow::bail!("marlin_proj: groupsize must be -1 or 128 (got {groupsize})");
    }
    if groupsize == 128 && k % 128 != 0 {
        anyhow::bail!("marlin_proj: k={k} not divisible by groupsize=128");
    }

    // The Marlin packer needs f32 on host. Cast on-device (avoid a bf16 CPU
    // dependency) then download — all kt-native, no candle.
    let w_f32 = weight_t
        .to_dtype(kiln_tensor::DType::F32)
        .map_err(|e| anyhow::anyhow!("marlin_proj: cast weight_t to f32: {e}"))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!("marlin_proj: contiguous f32: {e}"))?;
    let host_weight = w_f32
        .flatten_all()
        .map_err(|e| anyhow::anyhow!("marlin_proj: flatten weight: {e}"))?
        .to_vec1::<f32>()
        .map_err(|e| anyhow::anyhow!("marlin_proj: download weight to host: {e}"))?;
    debug_assert_eq!(host_weight.len(), k * n);

    Ok(Some(PackJobHost {
        host_weight,
        k,
        n,
        groupsize,
    }))
}

/// Phase 2 of Marlin packing: run the pure-Rust `quantize_and_pack` against
/// a host weight buffer. Pure CPU, no GPU state, no I/O — safe to call from
/// any rayon worker thread in parallel with other `pack_host` calls.
#[cfg(feature = "cuda")]
pub fn pack_host(job: &PackJobHost) -> PackedHost {
    let (b_packed_vec, scales_vec, _dequant) = kiln_marlin_gemm::pack::quantize_and_pack(
        &job.host_weight,
        job.k,
        job.n,
        job.groupsize as i64,
    );
    PackedHost {
        b_packed_vec,
        scales_vec,
        k: job.k,
        n: job.n,
        groupsize: job.groupsize,
    }
}

/// Phase 3 of Marlin packing: upload the packed i32 weight tensor and
/// permuted f16 scales to `device` and assemble the final
/// [`MarlinPackedProj`].
#[cfg(feature = "cuda")]
pub fn upload_packed(packed: PackedHost, device: &kiln_tensor::Device) -> Result<MarlinPackedProj> {
    let PackedHost {
        b_packed_vec,
        scales_vec,
        k,
        n,
        groupsize,
    } = packed;
    let num_groups = if groupsize == -1 {
        1
    } else {
        k / groupsize as usize
    };
    // (#1082) kt-NATIVE: build the packed weight + permuted scales as kt
    // tensors on host and move to device — no candle source tensor, no
    // `kt_tensor_from_candle_cuda_copy` bridge. The Marlin packer emits the
    // weight as `i32`; kt has no I32 variant, so store as U32 (a bit-identical
    // 4-byte reinterpret — exactly what the old candle `I32 → kt U32` bridge
    // produced, and what the decode-hot-path GEMM `marlin_w4a16_gemm_kt`
    // already consumes). Scales stay F16.
    let b_packed_u32: Vec<u32> = b_packed_vec.into_iter().map(|x| x as u32).collect();
    let b_packed = kiln_tensor::Tensor::from_vec(b_packed_u32, (k / 16, n * 16 / 8))
        .map_err(|e| anyhow::anyhow!("marlin_proj: build b_packed kt tensor: {e}"))?
        .to_device(*device)
        .map_err(|e| anyhow::anyhow!("marlin_proj: upload b_packed to device: {e}"))?;
    let scales = kiln_tensor::Tensor::from_vec(scales_vec, (num_groups, n))
        .map_err(|e| anyhow::anyhow!("marlin_proj: build scales kt tensor: {e}"))?
        .to_device(*device)
        .map_err(|e| anyhow::anyhow!("marlin_proj: upload scales to device: {e}"))?;
    Ok(MarlinPackedProj {
        b_packed,
        scales,
        groupsize,
        k,
        n,
    })
}

/// Batch-pack several projection weights, running the CPU-bound
/// `quantize_and_pack` step in parallel via rayon.
///
/// Each input is a pre-transposed `[k, n]` weight plus its Marlin groupsize
/// (`-1` or `128`). The result preserves input order: entry `i` in the
/// returned vec is the packed projection for input `i`. Entries whose
/// shape didn't fit Marlin's tile constraints (or that live on a non-CUDA
/// device) are `Ok(None)`, matching the single-weight
/// [`pack_from_bf16`] contract so the caller can fall back to the BF16
/// path on a per-projection basis.
///
/// Phases:
/// 1. Serial GPU→CPU download of each weight into a host f32 buffer.
/// 2. Parallel CPU pack across all queued jobs (rayon `par_iter`).
/// 3. Serial CPU→GPU upload of the packed buffers.
///
/// The CPU pack phase is always parallel. Route experiments belong in an
/// explicit benchmark harness rather than process-global production state.
#[cfg(feature = "cuda")]
pub fn pack_from_bf16_batch(
    inputs: &[(kiln_tensor::Tensor, i32)],
) -> Result<Vec<Option<MarlinPackedProj>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    // The kt device must be consistent across inputs. The call sites in
    // `forward.rs` build every weight on the same `device`, so this is a
    // sanity check rather than a real runtime constraint.
    let device = inputs[0].0.device();

    // Phase 1: serial GPU→CPU download.
    let jobs: Vec<Option<PackJobHost>> = inputs
        .iter()
        .map(|(t, gs)| prepare_pack_job(t, *gs))
        .collect::<Result<Vec<_>>>()?;

    // Phase 2: parallel CPU pack. Rayon `par_iter` over `Option` skips
    // None entries cheaply and preserves input order in the output.
    use rayon::prelude::*;
    let packed: Vec<Option<PackedHost>> = jobs
        .par_iter()
        .map(|job| job.as_ref().map(pack_host))
        .collect();

    // Phase 3: serial CPU→GPU upload.
    let mut out = Vec::with_capacity(packed.len());
    for p in packed {
        match p {
            Some(p) => out.push(Some(upload_packed(p, &device)?)),
            None => out.push(None),
        }
    }
    Ok(out)
}

/// Non-CUDA stub for [`pack_from_bf16_batch`]: the kernel is CUDA-only, so
/// every entry is skipped.
#[cfg(not(feature = "cuda"))]
pub fn pack_from_bf16_batch(
    inputs: &[(kiln_tensor::Tensor, i32)],
) -> Result<Vec<Option<MarlinPackedProj>>> {
    Ok((0..inputs.len()).map(|_| None).collect())
}

/// Fully kt-native `x @ W` against a Marlin-packed projection.
///
/// `x_kt` is a contiguous-or-not kt-typed BF16 activation of shape
/// `[.., k]` (rank 2 `[m, k]` or rank 3 `[batch, seq, k]`); the result
/// is kt-typed BF16 matching the input rank with last dim `n`.
///
/// This is the per-token DECODE entry point (#1082 H1/H2/H3): it does
/// the BF16→F16 cast, the kernel call, and the F16→BF16 cast entirely
/// in kt-space, with **no** `kt_logits_to_candle` / `candle_to_kt_activation`
/// round-trip on the activation or the output. The packed weight + scales
/// are stored as kt tensors in [`MarlinPackedProj`] (bridged once at pack
/// time in [`upload_packed`]) and handed straight to the kernel — **no**
/// per-call candle→kt bridge of the weights.
///
/// Numerically identical to [`matmul_bf16`]: same FFI symbol
/// (`marlin_w4a16_gemm_kt`), same F16-only kernel, same BF16↔F16 cast at
/// the boundary — the only difference is the activation/result never make a
/// candle detour.
#[cfg(feature = "cuda")]
pub fn matmul_bf16_kt(
    x_kt: &kiln_tensor::Tensor,
    w: &MarlinPackedProj,
) -> Result<kiln_tensor::Tensor> {
    use kiln_tensor::DType as KtDType;

    let rank = x_kt.rank();
    // Flatten to 2D `[m, k]` for the kernel, remembering the original
    // leading dims so we can restore the output rank.
    let (x2_kt, restore): (kiln_tensor::Tensor, Option<(usize, usize)>) = match rank {
        2 => {
            let (_m, k) = x_kt
                .dims2()
                .context("marlin_proj kt: x must be [m, k] when 2D")?;
            if k != w.k {
                anyhow::bail!("marlin_proj kt: x last-dim {k} != packed weight k {}", w.k);
            }
            (
                x_kt.contiguous()
                    .context("marlin_proj kt: x contiguous (2D)")?,
                None,
            )
        }
        3 => {
            let (batch, seq, k) = x_kt
                .dims3()
                .context("marlin_proj kt: x must be [batch, seq, k] when 3D")?;
            if k != w.k {
                anyhow::bail!("marlin_proj kt: x last-dim {k} != packed weight k {}", w.k);
            }
            let x2 = x_kt
                .reshape((batch * seq, k))
                .context("marlin_proj kt: reshape x [batch*seq, k]")?
                .contiguous()
                .context("marlin_proj kt: x2 contiguous (3D)")?;
            (x2, Some((batch, seq)))
        }
        other => anyhow::bail!("marlin_proj kt: unsupported activation rank {other}"),
    };

    kiln_nvtx::range!(c"kiln/marlin_w4a16_gemm_kt");

    // Kernel is F16-only; cast bf16 -> fp16 in kt-space (no candle detour).
    let x_fp16 = if x2_kt.dtype() == KtDType::F16 {
        x2_kt
    } else {
        x2_kt
            .to_dtype(KtDType::F16)
            .context("marlin_proj kt: cast x bf16 -> fp16")?
            .contiguous()
            .context("marlin_proj kt: x_fp16 contiguous")?
    };

    // #1082: the packed weight + scales are already kt (bridged once at
    // pack time in `upload_packed`). Hand them straight to the kernel —
    // no per-call candle→kt bridge on the decode hot path. They were built
    // contiguous at pack time, so no `.contiguous()` is needed here.
    let y_fp16 =
        kiln_marlin_gemm::marlin_w4a16_gemm_kt(&x_fp16, &w.b_packed, &w.scales, w.groupsize)
            .map_err(|e| anyhow::anyhow!("marlin_proj kt: marlin_w4a16_gemm_kt: {e}"))?;

    // Cast result fp16 -> bf16 in kt-space (no candle detour).
    let y_bf16 = y_fp16
        .to_dtype(KtDType::BF16)
        .context("marlin_proj kt: cast y fp16 -> bf16")?;

    let out = match restore {
        None => y_bf16,
        Some((batch, seq)) => y_bf16
            .reshape((batch, seq, w.n))
            .context("marlin_proj kt: reshape output [batch, seq, n]")?,
    };
    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn matmul_bf16_kt(
    _x_kt: &kiln_tensor::Tensor,
    _w: &MarlinPackedProj,
) -> Result<kiln_tensor::Tensor> {
    anyhow::bail!("marlin_proj::matmul_bf16_kt requires the `cuda` feature")
}

/// Whether the installed profile packs full-attention Q and MLP projections.
pub fn enabled() -> bool {
    crate::cuda_marlin_policy::current_cuda_marlin_policy().attention_q_and_mlp
}

/// Whether the installed profile also packs the GDN output projection.
pub fn gdn_out_proj_enabled() -> bool {
    crate::cuda_marlin_policy::current_cuda_marlin_policy().gdn_out_proj
}

impl MarlinPackedProj {
    /// Training-session residency companion to
    /// `GpuWeights::to_device_deep`: deep-copy the packed weight + scales
    /// onto `device`; `groupsize`/`k`/`n` are metadata.
    pub fn to_device_deep(&self, device: kiln_tensor::Device) -> anyhow::Result<Self> {
        Ok(Self {
            b_packed: self
                .b_packed
                .to_device(device)
                .map_err(|e| anyhow::anyhow!("marlin b_packed to_device: {e}"))?,
            scales: self
                .scales
                .to_device(device)
                .map_err(|e| anyhow::anyhow!("marlin scales to_device: {e}"))?,
            groupsize: self.groupsize,
            k: self.k,
            n: self.n,
        })
    }
}
