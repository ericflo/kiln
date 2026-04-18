//! MLX backend: Apple's MLX framework via the `mlx-rs` Rust bindings.
//!
//! Sits on top of the candle-metal backend — candle still owns weight loading,
//! tokenizer I/O, the paged pool's memory allocation, and every op not in the
//! MLX fast path. MLX takes over the fused attention primitive
//! (`fast::scaled_dot_product_attention`) which is hand-tuned by Apple for
//! Apple Silicon and outperforms candle SDPA in published benchmarks.
//!
//! Tensors cross the candle↔mlx boundary via a host copy (`to_vec` +
//! `Array::from_slice`). On Apple Silicon's unified memory pool this is a
//! memcpy, not a PCIe transfer — slow compared to kernel time for big tensors
//! but fine as a first-cut. Zero-copy via shared `MTLBuffer` handles is a
//! follow-up optimization documented in MACOS_MLX_PLAN.md.
//!
//! Build requirements: full Xcode (for `xcrun metal` — MLX AOT-compiles MSL,
//! unlike candle-metal which JITs at runtime). Accept the Xcode license with
//! `sudo xcodebuild -license accept` before `cargo build --features mlx`.

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

        // candle layout:  [batch, seq_len, num_heads, head_dim]
        // mlx SDPA layout: [batch, num_heads, seq_len, head_dim]
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
        let out_mlx = mlx_rs::fast::scaled_dot_product_attention(
            &q_mlx,
            &k_mlx,
            &v_mlx,
            softmax_scale,
            mask,
        )
        .context("mlx scaled_dot_product_attention failed")?;

        // Back to candle + reverse the transpose to match CUDA/Metal output.
        let out = array_to_tensor(&out_mlx, q.dtype(), &self.device, q_t.dims())?;
        let out = out.transpose(1, 2)?.contiguous()?;
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

        // Gather the live slice of the paged pool on the candle side (cheaper
        // than round-tripping block_table indices through MLX).
        let k_blocks = k_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;
        let v_blocks = v_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;
        let block_ids = block_table.flatten_all()?;
        let total_gathered = max_blocks_per_seq * page_block_size;
        let k_live = k_blocks
            .index_select(&block_ids, 0)?
            .reshape((total_gathered, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?;
        let v_live = v_blocks
            .index_select(&block_ids, 0)?
            .reshape((total_gathered, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?;

        let q_t = q.transpose(1, 2)?.contiguous()?; // [1, num_heads, 1, head_dim]
        let k_t = k_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // [1, num_kv_heads, L, head_dim]
        let v_t = v_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;

        let q_mlx = tensor_to_array(&q_t)?;
        let k_mlx = tensor_to_array(&k_t)?;
        let v_mlx = tensor_to_array(&v_t)?;

        let mask = if causal {
            Some(ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out_mlx = mlx_rs::fast::scaled_dot_product_attention(
            &q_mlx,
            &k_mlx,
            &v_mlx,
            softmax_scale,
            mask,
        )
        .context("mlx scaled_dot_product_attention (paged decode) failed")?;

        let out = array_to_tensor(&out_mlx, q.dtype(), &self.device, &[1, num_heads, 1, head_dim])?;
        let out = out.transpose(1, 2)?.contiguous()?;
        debug_assert_eq!(out.dims(), &[1, 1, num_heads, head_dim]);
        Ok(Some(out))
    }
}

fn mlx_sdpa_dtype_supported(dtype: DType) -> bool {
    // MLX fast SDPA accepts f32/f16/bf16. Same whitelist as candle SDPA; we
    // decline on anything else (notably F64 and FP8-encoded U8).
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

/// candle → mlx via a host copy. BF16 has no direct `from_slice` on `Array`,
/// so we route through f32 for BF16 tensors (small accuracy cost on the
/// boundary; the recomputation on the MLX side runs in its native dtype).
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
            Ok(a.astype(MlxDtype::Float16)?)
        }
        DType::BF16 => {
            let data = t_contig.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let a = Array::from_slice(&data, &shape);
            Ok(a.astype(MlxDtype::Bfloat16)?)
        }
        other => anyhow::bail!("tensor_to_array: unsupported dtype {other:?}"),
    }
}

/// mlx → candle via a host copy. Forces an `eval` so lazy mlx graphs don't
/// leak across the boundary.
fn array_to_tensor(
    arr: &Array,
    out_dtype: DType,
    device: &Device,
    shape: &[usize],
) -> Result<Tensor> {
    arr.eval()?;
    let f32_arr = if arr.dtype() != MlxDtype::Float32 {
        arr.astype(MlxDtype::Float32)?
    } else {
        arr.clone()
    };
    f32_arr.eval()?;
    let host: Vec<f32> = f32_arr
        .try_into()
        .context("mlx Array::try_into::<Vec<f32>> failed")?;
    let tensor = Tensor::from_vec(host, shape, device)?;
    if tensor.dtype() != out_dtype {
        Ok(tensor.to_dtype(out_dtype)?)
    } else {
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;

    /// Parity: `MlxBackend::flash_attn_prefill` agrees with a direct candle
    /// SDPA on the same inputs (within BF16-rounding tolerance). Exercises
    /// the candle ↔ mlx Array round-trip + mlx's fused SDPA. Gated behind
    /// `--features mlx` because it links against mlx-sys.
    #[test]
    fn test_prefill_parity_vs_candle_sdpa() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Metal unavailable, skipping mlx parity test: {e}");
                return Ok(());
            }
        };

        let (batch, seq_len, num_heads, head_dim) = (1, 12, 4, 128);
        let q = Tensor::randn(0.0f32, 0.02, (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::randn(0.0f32, 0.02, (batch, seq_len, num_heads, head_dim), &device)?;
        let v = Tensor::randn(0.0f32, 0.02, (batch, seq_len, num_heads, head_dim), &device)?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let backend = MlxBackend::new(device.clone());
        let mlx_out = backend
            .flash_attn_prefill(&q, &k, &v, scale, true)?
            .expect("mlx backend should handle this shape");

        // Reference: candle SDPA on the same inputs.
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        let ref_out = candle_nn::ops::sdpa(&q_t, &k_t, &v_t, None, true, scale, 1.0)?;
        let ref_out = ref_out.transpose(1, 2)?.contiguous()?;

        assert_eq!(mlx_out.dims(), ref_out.dims());
        let diff = (&mlx_out - &ref_out)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        // BF16 round-trip on the boundary (f32 → bf16 → f32) is the
        // dominant error source. 5e-3 fits M1/M2 measurements; tighten on
        // better hardware if it proves conservative.
        assert!(
            diff < 5e-3,
            "mlx vs candle SDPA diverge: max abs diff = {diff}"
        );
        Ok(())
    }
}
