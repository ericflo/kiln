//! MLX backend: Apple's MLX framework via the `mlx-rs` Rust bindings.
//!
//! Sits on top of the candle-metal backend — candle still owns weight loading,
//! tokenizer I/O, the paged pool's memory allocation, and every op not in the
//! MLX fast path. MLX takes over the fused attention primitive
//! (`fast::scaled_dot_product_attention`) which is hand-tuned by Apple for
//! Apple Silicon.
//!
//! Tensors cross the candle↔mlx boundary via a host copy. On unified memory
//! this is a memcpy, not a PCIe transfer, but it's still a per-call cost;
//! zero-copy via shared `MTLBuffer` handles is a follow-up.
//!
//! Build requirements: full Xcode (for `xcrun metal` — MLX AOT-compiles MSL,
//! unlike candle-metal which JITs at runtime), plus the on-demand Metal
//! Toolchain (`xcodebuild -downloadComponent MetalToolchain`) and an
//! accepted license (`sudo xcodebuild -license accept`).
//!
//! Testing: run with `--test-threads=1` when the suite also includes
//! candle-metal tests. MLX and candle-metal both grab the default Metal
//! device and can SIGSEGV under concurrent access.
//!
//! Stride handling: MLX's fused SDPA sometimes returns arrays whose physical
//! memory layout doesn't match the logical shape (e.g., strides
//! `[8192, 128, 256, 1]` on a `[1,2,32,128]` output). `Array::as_slice` is a
//! raw-memory view that ignores strides, so we explicitly walk the logical
//! shape when copying back to candle.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use mlx_rs::fast::ScaledDotProductAttentionMask;
use mlx_rs::{Array, Dtype as MlxDtype};

use super::BackendRuntime;

#[derive(Debug)]
pub struct MlxBackend {
    device: Device,
}

impl MlxBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(
            matches!(device, Device::Metal(_)),
            "MlxBackend created on non-Metal device"
        );
        Self { device }
    }
}

impl BackendRuntime for MlxBackend {
    fn name(&self) -> &'static str {
        "mlx"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        true
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    fn flash_attn_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if !mlx_sdpa_dtype_supported(q.dtype()) {
            return Ok(None);
        }
        let out = sdpa_via_mlx(q, k, v, softmax_scale, causal, &self.device)?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &Tensor,
        total_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if !mlx_sdpa_dtype_supported(q.dtype()) {
            return Ok(None);
        }
        let (batch, q_len, num_heads, head_dim) = q.dims4()?;
        if batch != 1 || q_len != 1 {
            return Ok(None);
        }
        let (total_slots, num_kv_heads, _) = k_pool.dims3()?;
        if total_slots % page_block_size != 0 {
            return Ok(None);
        }
        let num_blocks = total_slots / page_block_size;
        let max_blocks_per_seq = block_table.dim(1)?;

        // Gather the live K/V window on the candle side — cheaper than
        // round-tripping block_table indices through MLX.
        let k_blocks = k_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;
        let v_blocks = v_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;
        let block_ids = block_table.flatten_all()?;
        let total_gathered = max_blocks_per_seq * page_block_size;
        let k_live = k_blocks
            .index_select(&block_ids, 0)?
            .reshape((total_gathered, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?
            .unsqueeze(0)?;
        let v_live = v_blocks
            .index_select(&block_ids, 0)?
            .reshape((total_gathered, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?
            .unsqueeze(0)?;

        // After unsqueeze both sides are candle-layout [batch=1, seq_len,
        // num_{q,kv}_heads, head_dim] — the same layout `sdpa_via_mlx`
        // expects, so the decode path reuses the helper unchanged.
        let out = sdpa_via_mlx(q, &k_live, &v_live, softmax_scale, causal, &self.device)?;
        debug_assert_eq!(out.dims(), &[1, 1, num_heads, head_dim]);
        Ok(Some(out))
    }
}

/// Call MLX's fused SDPA on candle tensors in kiln's `[batch, seq, heads,
/// head_dim]` convention. Handles the transpose to MLX's `[batch, heads,
/// seq, head_dim]` layout, the candle↔mlx conversion, the causal mask, and
/// the transpose back. Shared by prefill and paged-decode paths.
fn sdpa_via_mlx(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
    device: &Device,
) -> Result<Tensor> {
    let q_t = q.transpose(1, 2)?.contiguous()?;
    let k_t = k.transpose(1, 2)?.contiguous()?;
    let v_t = v.transpose(1, 2)?.contiguous()?;

    let q_mlx = tensor_to_array(&q_t)?;
    let k_mlx = tensor_to_array(&k_t)?;
    let v_mlx = tensor_to_array(&v_t)?;

    let mask = if causal {
        Some(ScaledDotProductAttentionMask::Causal)
    } else {
        None
    };
    let out_mlx =
        mlx_rs::fast::scaled_dot_product_attention(&q_mlx, &k_mlx, &v_mlx, softmax_scale, mask)
            .context("mlx scaled_dot_product_attention failed")?;

    let out = array_to_tensor(&out_mlx, q.dtype(), device, q_t.dims())?;
    Ok(out.transpose(1, 2)?.contiguous()?)
}

fn mlx_sdpa_dtype_supported(dtype: DType) -> bool {
    // MLX fast SDPA accepts f32/f16/bf16. Same whitelist as candle SDPA; we
    // decline on anything else (notably F64 and FP8-encoded U8).
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

/// candle → mlx via a host copy. `Array::from_slice` accepts native Rust
/// types (f32, i32, etc.) — no direct bf16, so we route f16/bf16 tensors
/// through f32 and rely on `as_dtype` for the mlx-side downcast.
fn tensor_to_array(t: &Tensor) -> Result<Array> {
    let shape: Vec<i32> = t.dims().iter().map(|&d| d as i32).collect();
    let t_contig = t.contiguous()?;
    match t_contig.dtype() {
        DType::F32 => {
            let data = t_contig.flatten_all()?.to_vec1::<f32>()?;
            Ok(Array::from_slice(&data, &shape))
        }
        DType::F16 => {
            let data = t_contig.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let a = Array::from_slice(&data, &shape);
            Ok(a.as_dtype(MlxDtype::Float16)?)
        }
        DType::BF16 => {
            let data = t_contig.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let a = Array::from_slice(&data, &shape);
            Ok(a.as_dtype(MlxDtype::Bfloat16)?)
        }
        other => anyhow::bail!("tensor_to_array: unsupported dtype {other:?}"),
    }
}

/// mlx → candle via a host copy. MLX's fused kernels sometimes return
/// arrays whose physical memory layout doesn't match the logical shape
/// (strides [8192, 128, 256, 1] for a [1,2,32,128] SDPA output on M1 was
/// the motivating case). `Array::as_slice` gives us the raw buffer
/// ignoring strides, so we explicitly walk the logical shape with the
/// reported strides to materialize a row-major `Vec<f32>`.
fn array_to_tensor(
    arr: &Array,
    out_dtype: DType,
    device: &Device,
    shape: &[usize],
) -> Result<Tensor> {
    arr.eval()?;
    let f32_arr = if arr.dtype() != MlxDtype::Float32 {
        arr.as_dtype(MlxDtype::Float32)?
    } else {
        arr.clone()
    };
    f32_arr.eval()?;

    let raw = f32_arr.as_slice::<f32>();
    let strides = f32_arr.strides();
    let total: usize = shape.iter().product();

    let row_major = if is_row_major(shape, strides) {
        raw.to_vec()
    } else {
        let mut dst = Vec::with_capacity(total);
        let ndim = shape.len();
        let mut idx = vec![0usize; ndim];
        for _ in 0..total {
            let mut phys = 0usize;
            for dim in 0..ndim {
                phys += idx[dim] * strides[dim];
            }
            dst.push(raw[phys]);
            for dim in (0..ndim).rev() {
                idx[dim] += 1;
                if idx[dim] < shape[dim] {
                    break;
                }
                idx[dim] = 0;
            }
        }
        dst
    };

    let tensor = Tensor::from_vec(row_major, shape, device)?;
    if tensor.dtype() != out_dtype {
        Ok(tensor.to_dtype(out_dtype)?)
    } else {
        Ok(tensor)
    }
}

fn is_row_major(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected: usize = 1;
    for dim in (0..shape.len()).rev() {
        if shape[dim] != 1 && strides[dim] != expected {
            return false;
        }
        expected *= shape[dim];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;

    /// Parity: `MlxBackend::flash_attn_prefill` agrees with a direct candle
    /// SDPA on the same inputs (within BF16-rounding tolerance). Exercises
    /// the candle ↔ mlx Array round-trip + mlx's fused SDPA. Gated behind
    /// `--features mlx` because it links against mlx-sys.
    /// Naive F32 softmax+matmul attention. Deterministic reference for both
    /// MLX and candle SDPA parity checks — both should match within
    /// floating-point rounding.
    fn naive_attention(
        q: &Tensor, // [batch, heads, seq_q, head_dim]
        k: &Tensor, // [batch, heads, seq_k, head_dim]
        v: &Tensor, // [batch, heads, seq_k, head_dim]
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        let scores = q.matmul(&k.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?)?;
        let scores = (scores * scale as f64)?;
        let scores = if causal {
            let (_, _, sq, sk) = scores.dims4()?;
            let mask: Vec<f32> = (0..sq)
                .flat_map(|i| (0..sk).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            let mask_t = Tensor::from_vec(mask, (sq, sk), scores.device())?;
            scores.broadcast_add(&mask_t)?
        } else {
            scores
        };
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        Ok(weights.matmul(v)?)
    }

    fn run_parity(causal: bool) -> Result<(f32, f32)> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok((0.0, 0.0));
        };

        let (batch, seq_len, num_heads, head_dim) = (1, 12, 4, 128);
        let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let backend = MlxBackend::new(device.clone());
        let mlx_out = backend
            .flash_attn_prefill(&q, &k, &v, scale, causal)?
            .expect("mlx backend should handle this shape");

        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        let naive_out = naive_attention(&q_t, &k_t, &v_t, scale, causal)?
            .transpose(1, 2)?
            .contiguous()?;
        let candle_out = candle_nn::ops::sdpa(&q_t, &k_t, &v_t, None, causal, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?;

        let mlx_diff = (&mlx_out - &naive_out)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        let candle_diff = (&candle_out - &naive_out)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        eprintln!(
            "parity (causal={causal}): mlx_vs_naive={mlx_diff:.5}, candle_vs_naive={candle_diff:.5}"
        );
        Ok((mlx_diff, candle_diff))
    }

    /// MLX SDPA agrees with a naive F32 reference within 1e-3. Same tolerance
    /// applies to candle SDPA — both are bit-inexact vs the reference due to
    /// fused-kernel accumulation, but both should land in the same ballpark.
    /// Diagnostic: round-trip a tensor through the candle↔mlx conversion.
    /// Must be bit-identical for F32 inputs.
    #[test]
    fn test_tensor_roundtrip_f32() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let shape = &[1usize, 4, 12, 128];
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?;
        let arr = tensor_to_array(&t)?;
        let back = array_to_tensor(&arr, DType::F32, &device, shape)?;
        let diff = (&t - &back)?.abs()?.flatten_all()?.max(D::Minus1)?.to_scalar::<f32>()?;
        eprintln!("roundtrip max abs diff: {diff}");
        assert!(diff < 1e-6, "tensor round-trip should be identity: {diff}");
        Ok(())
    }

    #[test]
    fn test_prefill_vs_naive_noncausal() -> Result<()> {
        let (mlx_diff, candle_diff) = run_parity(false)?;
        assert!(mlx_diff < 1e-3, "MLX SDPA diverges from naive reference: {mlx_diff}");
        assert!(candle_diff < 1e-3, "candle SDPA diverges from naive reference: {candle_diff}");
        Ok(())
    }

    #[test]
    fn test_prefill_vs_naive_causal() -> Result<()> {
        let (mlx_diff, candle_diff) = run_parity(true)?;
        assert!(mlx_diff < 1e-3, "MLX SDPA (causal) diverges from naive: {mlx_diff}");
        assert!(candle_diff < 1e-3, "candle SDPA (causal) diverges from naive: {candle_diff}");
        Ok(())
    }
}
