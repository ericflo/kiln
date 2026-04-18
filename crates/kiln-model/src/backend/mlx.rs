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
        if !mlx_sdpa_supports_head_dim(q.dim(candle_core::D::Minus1)?) {
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
        if !mlx_sdpa_supports_head_dim(head_dim) {
            return Ok(None);
        }
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

/// MLX `fast::scaled_dot_product_attention` falls back to a slower non-fused
/// path outside its fast-kernel envelope (no SIGSEGV — per mlx-rs's docstring,
/// "handles other cases with regular MLX operations"). Mirror the candle-metal
/// whitelist so both Apple Silicon backends decline the same shapes; non-fast
/// paths route through the portable candle fallback which is usually a wash.
fn mlx_sdpa_supports_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
}

/// candle → mlx via a host copy. Passes bf16/f16 through natively (both
/// `half::{bf16,f16}` in candle and mlx-rs) instead of round-tripping
/// through f32, which doubles the copied bytes and burns a cast.
fn tensor_to_array(t: &Tensor) -> Result<Array> {
    let shape: Vec<i32> = t.dims().iter().map(|&d| d as i32).collect();
    let t_contig = t.contiguous()?;
    match t_contig.dtype() {
        DType::F32 => {
            let data = t_contig.flatten_all()?.to_vec1::<f32>()?;
            Ok(Array::from_slice(&data, &shape))
        }
        DType::F16 => {
            let data = t_contig.flatten_all()?.to_vec1::<half::f16>()?;
            Ok(Array::from_slice(&data, &shape))
        }
        DType::BF16 => {
            let data = t_contig.flatten_all()?.to_vec1::<half::bf16>()?;
            Ok(Array::from_slice(&data, &shape))
        }
        other => anyhow::bail!("tensor_to_array: unsupported dtype {other:?}"),
    }
}

/// mlx → candle via a host copy. MLX's fused kernels sometimes return arrays
/// whose physical memory layout doesn't match the logical shape (strides
/// `[8192, 128, 256, 1]` on a `[1,2,32,128]` SDPA output on M1 was the
/// motivating case), so we force row-major materialization on the MLX side
/// via a no-op `reshape` before calling `as_slice` (which is a raw-memory
/// view and ignores strides). Doing it in MLX keeps the materialize loop
/// vectorized on-device instead of running a per-element Rust walk.
///
/// Reads the native dtype when it matches `out_dtype` to avoid the BF16 →
/// F32 → BF16 round-trip that the prior implementation did unconditionally.
fn array_to_tensor(
    arr: &Array,
    out_dtype: DType,
    device: &Device,
    shape: &[usize],
) -> Result<Tensor> {
    // Force row-major materialization on-device via `flatten` → `reshape`:
    // flatten to 1D (MLX can't keep a transposed view through this because
    // the result must be contiguous in the logical axis order), then reshape
    // back. A `reshape(same_shape)` on a strided array is a no-op view — the
    // round-trip through 1D is what actually materializes.
    let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let contig = arr
        .flatten(None, None)?
        .reshape(&shape_i32)?;
    contig.eval()?;

    // Read natively when the dtype already matches what we want; otherwise
    // let MLX cast on-device (vectorized) before copying.
    let tensor = match (contig.dtype(), out_dtype) {
        (MlxDtype::Float32, DType::F32) => {
            Tensor::from_slice(contig.as_slice::<f32>(), shape, device)?
        }
        (MlxDtype::Float16, DType::F16) => {
            Tensor::from_slice(contig.as_slice::<half::f16>(), shape, device)?
        }
        (MlxDtype::Bfloat16, DType::BF16) => {
            Tensor::from_slice(contig.as_slice::<half::bf16>(), shape, device)?
        }
        // Dtype mismatch between MLX output and candle request — cast on MLX
        // (on-device, vectorized) and read the target dtype directly.
        (_, DType::F32) => {
            let cast = contig.as_dtype(MlxDtype::Float32)?;
            cast.eval()?;
            Tensor::from_slice(cast.as_slice::<f32>(), shape, device)?
        }
        (_, DType::F16) => {
            let cast = contig.as_dtype(MlxDtype::Float16)?;
            cast.eval()?;
            Tensor::from_slice(cast.as_slice::<half::f16>(), shape, device)?
        }
        (_, DType::BF16) => {
            let cast = contig.as_dtype(MlxDtype::Bfloat16)?;
            cast.eval()?;
            Tensor::from_slice(cast.as_slice::<half::bf16>(), shape, device)?
        }
        (_, other) => anyhow::bail!("array_to_tensor: unsupported out_dtype {other:?}"),
    };
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;

    /// Naive F32 softmax+matmul attention. Deterministic reference for both
    /// MLX and candle SDPA parity checks.
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

    /// Round-trip a tensor through the candle↔mlx conversion. Must be
    /// bit-identical for F32 inputs.
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

    /// Native BF16 round-trip — covers the path that skips the f32 staging.
    /// Must be bit-identical because both sides use the same `half::bf16`.
    #[test]
    fn test_tensor_roundtrip_bf16() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let shape = &[1usize, 4, 12, 128];
        let t = Tensor::randn(0.0f32, 1.0, shape, &device)?.to_dtype(DType::BF16)?;
        let arr = tensor_to_array(&t)?;
        let back = array_to_tensor(&arr, DType::BF16, &device, shape)?;
        assert_eq!(back.dtype(), DType::BF16);
        let diff = (&t.to_dtype(DType::F32)? - &back.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        assert_eq!(diff, 0.0, "bf16 round-trip should be bit-identical");
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

    /// End-to-end parity for `MlxBackend::flash_attn_paged_decode`: shuffled
    /// block table, F32 inputs, real (non-degenerate) Q values. Validates the
    /// candle-side gather (`index_select` → `narrow` → `unsqueeze`) hands off
    /// the right K/V to the MLX SDPA path, and that the paged-decode result
    /// matches a naive F32 reference computed directly on the materialized
    /// K/V slots. Catches gather indexing bugs, layout/transpose mistakes,
    /// and MLX SDPA correctness regressions in one shot.
    #[test]
    fn test_paged_decode_parity_vs_naive() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };

        let num_heads = 4;
        let num_kv_heads = 1;
        let head_dim = 128;
        let block_size = 16;
        let num_blocks = 8;
        let total_slots = num_blocks * block_size;
        let max_blocks_per_seq = 4;
        let total_seqlen_k = 50; // 4 blocks * 16 = 64 slots; only first 50 valid.

        // Shuffled physical block table — exercises the gather, not just
        // sequential blocks.
        let block_ids: [u32; 4] = [3, 7, 0, 5];
        let block_table = Tensor::new(block_ids.as_slice(), &device)?
            .reshape((1usize, max_blocks_per_seq))?;

        // Random pool + random Q so the softmax is non-degenerate (a uniform
        // Q-K product would let any wrong gather pass by averaging).
        let k_pool = Tensor::randn(0.0f32, 1.0, (total_slots, num_kv_heads, head_dim), &device)?;
        let v_pool = Tensor::randn(0.0f32, 1.0, (total_slots, num_kv_heads, head_dim), &device)?;
        let q = Tensor::randn(0.0f32, 1.0, (1, 1, num_heads, head_dim), &device)?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // causal=false to dodge a convention mismatch: MLX's Causal mask
        // positions q at the END of k for q_len=1 (decode-style, attends to
        // all of k), while a naive `j > i` mask with i=0 would attend to only
        // k[0]. With q_len=1 there's nothing to mask either way, so this just
        // sidesteps the ambiguity. Causal correctness is covered by the
        // prefill parity tests.
        let backend = MlxBackend::new(device.clone());
        let mlx_out = backend
            .flash_attn_paged_decode(
                &q, &k_pool, &v_pool, &block_table, total_seqlen_k,
                block_size, scale, false,
            )?
            .expect("MlxBackend should handle this shape");

        // Reference: gather the same K/V the same way candle does, then run
        // naive F32 attention on it. Stays in candle entirely so any
        // disagreement is in the MLX path, not the reference.
        let k_blocks = k_pool.reshape((num_blocks, block_size, num_kv_heads, head_dim))?;
        let v_blocks = v_pool.reshape((num_blocks, block_size, num_kv_heads, head_dim))?;
        let ids = block_table.flatten_all()?;
        let k_live = k_blocks
            .index_select(&ids, 0)?
            .reshape((max_blocks_per_seq * block_size, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?
            .unsqueeze(0)?
            .transpose(1, 2)?
            .contiguous()?;
        let v_live = v_blocks
            .index_select(&ids, 0)?
            .reshape((max_blocks_per_seq * block_size, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?
            .unsqueeze(0)?
            .transpose(1, 2)?
            .contiguous()?;
        let q_ref = q.transpose(1, 2)?.contiguous()?;
        // GQA: naive_attention expects K/V already broadcast to num_heads.
        let gqa_ratio = num_heads / num_kv_heads;
        let k_ref = k_live
            .repeat((1usize, gqa_ratio, 1, 1))?
            .contiguous()?;
        let v_ref = v_live
            .repeat((1usize, gqa_ratio, 1, 1))?
            .contiguous()?;
        let naive_out = naive_attention(&q_ref, &k_ref, &v_ref, scale, false)?
            .transpose(1, 2)?
            .contiguous()?;

        assert_eq!(mlx_out.dims(), naive_out.dims());
        let diff = (&mlx_out - &naive_out)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        eprintln!("paged decode parity: mlx_vs_naive={diff:.5}");
        assert!(diff < 1e-3, "MLX paged decode diverges from naive: {diff}");
        Ok(())
    }

    /// MLX backend declines (returns Ok(None)) on head_dims outside the
    /// whitelist, so the caller cleanly falls back to the portable path.
    #[test]
    fn test_mlx_declines_unsupported_head_dim() -> Result<()> {
        let Some(device) = crate::backend::metal::try_new_metal() else {
            return Ok(());
        };
        let head_dim = 4; // not on the whitelist
        let q = Tensor::zeros((1usize, 1, 2, head_dim), DType::F32, &device)?;
        let k = Tensor::zeros((1usize, 1, 2, head_dim), DType::F32, &device)?;
        let v = Tensor::zeros((1usize, 1, 2, head_dim), DType::F32, &device)?;
        let backend = MlxBackend::new(device);
        let out = backend.flash_attn_prefill(&q, &k, &v, 1.0, true)?;
        assert!(out.is_none(), "should decline head_dim={head_dim}");
        Ok(())
    }
}
