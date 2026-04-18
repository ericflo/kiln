//! Metal backend: candle's fused SDPA for the attention hot path, portable
//! fallback for GDN and paged-decode.
//!
//! candle-metal ships `candle_nn::ops::sdpa` — an MLX-style fused scaled-dot-
//! product attention kernel with native GQA, BF16, and head dims
//! {32, 64, 72, 80, 96, 128, 256, 512}. For typical transformer head sizes
//! this replaces the vendored CUDA FlashAttention-2 call on Apple Silicon.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::BackendRuntime;

#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
}

impl MetalBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(
            matches!(device, Device::Metal(_)),
            "MetalBackend created on non-Metal device"
        );
        Self { device }
    }
}

impl BackendRuntime for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
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
        // Decline (caller falls back to the portable path) when candle's SDPA
        // can't handle the shape/dtype. Cheaper than surfacing a kernel error
        // from inside the fused path.
        if !matches!(q.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }
        if !metal_sdpa_supports_head_dim(q.dim(candle_core::D::Minus1)?) {
            return Ok(None);
        }

        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        // sdpa(q, k, v, mask, do_causal, scale, softcapping). softcapping=1.0
        // disables it; kiln's prefill path is always causal.
        let out = candle_nn::ops::sdpa(&q_t, &k_t, &v_t, None, causal, softmax_scale, 1.0)
            .context("candle-metal sdpa failed")?;

        let out = out.transpose(1, 2)?.contiguous()?;
        Ok(Some(out))
    }

    /// Gather K/V from the paged pool via `index_select` on the block table,
    /// then call candle's vectorized SDPA (single-query path). The gather
    /// replaces the slow materializing `paged_cache.read` +
    /// naive-softmax+matmul fallback — same result, one fused kernel.
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
        // Gate on everything SDPA can handle. Pool dtype matches q dtype by
        // construction (both come from the same forward config), so only q
        // needs checking.
        if !matches!(q.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }
        let head_dim = q.dim(candle_core::D::Minus1)?;
        if !metal_sdpa_supports_head_dim(head_dim) {
            return Ok(None);
        }

        let (batch, q_len, num_heads, _) = q.dims4()?;
        if batch != 1 || q_len != 1 {
            // Multi-sequence paged decode would need a per-sequence gather.
            // Stay on the fallback until the scheduler exercises it.
            return Ok(None);
        }

        let (total_slots, num_kv_heads, _) = k_pool.dims3()?;
        if total_slots % page_block_size != 0 {
            return Ok(None);
        }
        let num_blocks = total_slots / page_block_size;
        let max_blocks_per_seq = block_table.dim(1)?;

        // [num_blocks, block_size, num_kv_heads, head_dim] so index_select on
        // dim 0 gathers a full logical block's slots per physical block id.
        let k_blocks = k_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;
        let v_blocks = v_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;

        let block_ids = block_table.flatten_all()?;

        let k_gathered = k_blocks.index_select(&block_ids, 0)?;
        let v_gathered = v_blocks.index_select(&block_ids, 0)?;

        // [max_blocks_per_seq * block_size, num_kv_heads, head_dim] then
        // narrow to the live KV length.
        let total_gathered = max_blocks_per_seq * page_block_size;
        let k_flat = k_gathered.reshape((total_gathered, num_kv_heads, head_dim))?;
        let v_flat = v_gathered.reshape((total_gathered, num_kv_heads, head_dim))?;
        let k_live = k_flat.narrow(0, 0, total_seqlen_k)?;
        let v_live = v_flat.narrow(0, 0, total_seqlen_k)?;

        // SDPA needs [batch, num_heads, seq, head_dim]. Q arrives as
        // [1, 1, num_heads, head_dim]; K/V are [total_seqlen_k, num_kv_heads, head_dim].
        // SDPA handles GQA internally when num_heads % num_kv_heads == 0.
        let q_sdpa = q.transpose(1, 2)?.contiguous()?; // [1, num_heads, 1, head_dim]
        let k_sdpa = k_live
            .unsqueeze(0)?
            .transpose(1, 2)?
            .contiguous()?; // [1, num_kv_heads, total_seqlen_k, head_dim]
        let v_sdpa = v_live
            .unsqueeze(0)?
            .transpose(1, 2)?
            .contiguous()?;

        let out = candle_nn::ops::sdpa(&q_sdpa, &k_sdpa, &v_sdpa, None, causal, softmax_scale, 1.0)
            .context("candle-metal paged sdpa failed")?;

        // Back to [1, 1, num_heads, head_dim].
        let out = out.transpose(1, 2)?.contiguous()?;
        debug_assert_eq!(out.dims(), &[1, 1, num_heads, head_dim]);
        Ok(Some(out))
    }
}

/// Mirrors the head-dim whitelist in candle-nn 0.10.2's
/// `Sdpa::custom_op3` (see `ops.rs`). Drifts silently if the upstream
/// list grows — the fallback path absorbs the mismatch (correct, just
/// slower). Re-check this on candle bumps.
fn metal_sdpa_supports_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::D;

    /// Parity: `MetalBackend::flash_attn_paged_decode` output matches a
    /// direct materialize+SDPA reference computation on the same inputs.
    /// Validates the paged gather (index_select + narrow) logic.
    #[test]
    fn test_paged_decode_parity_with_direct_sdpa() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Metal unavailable, skipping paged decode parity: {e}");
                return Ok(());
            }
        };

        let num_heads = 4;
        let num_kv_heads = 1;
        let head_dim = 128;
        let block_size = 16;
        let num_blocks = 8;
        let total_slots = num_blocks * block_size;
        let max_blocks_per_seq = 4;
        let total_seqlen_k = 50; // covers 4 full blocks (64 slots) but only 50 valid.

        // Shuffled physical block table — exercises the gather, not just
        // sequential blocks.
        let block_ids: [u32; 4] = [3, 7, 0, 5];
        let block_table = Tensor::new(block_ids.as_slice(), &device)?
            .reshape((1usize, max_blocks_per_seq))?;

        // Fill the pool with distinctive per-slot values so the gather's
        // correctness is visible in the output. Each slot's values are
        // `slot_idx * 0.0001 + head_dim_offset * 0.000001`.
        let k_pool_data: Vec<f32> = (0..total_slots * num_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let v_pool_data: Vec<f32> = (0..total_slots * num_kv_heads * head_dim)
            .map(|i| (i as f32) * 0.0001 + 1.0)
            .collect();
        let k_pool = Tensor::from_slice(
            &k_pool_data,
            (total_slots, num_kv_heads, head_dim),
            &device,
        )?;
        let v_pool = Tensor::from_slice(
            &v_pool_data,
            (total_slots, num_kv_heads, head_dim),
            &device,
        )?;

        let q = Tensor::randn(0.0f32, 0.02, (1, 1, num_heads, head_dim), &device)?;
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let backend = MetalBackend::new(device.clone());

        let out_paged = backend
            .flash_attn_paged_decode(
                &q, &k_pool, &v_pool, &block_table, total_seqlen_k,
                block_size, softmax_scale, true,
            )?
            .expect("backend should handle this shape");

        // Reference: manually gather K/V the same way and call SDPA.
        let k_blocks = k_pool.reshape((num_blocks, block_size, num_kv_heads, head_dim))?;
        let v_blocks = v_pool.reshape((num_blocks, block_size, num_kv_heads, head_dim))?;
        let ids = block_table.flatten_all()?;
        let k_gathered = k_blocks
            .index_select(&ids, 0)?
            .reshape((max_blocks_per_seq * block_size, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?;
        let v_gathered = v_blocks
            .index_select(&ids, 0)?
            .reshape((max_blocks_per_seq * block_size, num_kv_heads, head_dim))?
            .narrow(0, 0, total_seqlen_k)?;

        let q_ref = q.transpose(1, 2)?.contiguous()?;
        let k_ref = k_gathered.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
        let v_ref = v_gathered.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
        let ref_out = candle_nn::ops::sdpa(&q_ref, &k_ref, &v_ref, None, true, softmax_scale, 1.0)?;
        let ref_out = ref_out.transpose(1, 2)?.contiguous()?;

        assert_eq!(out_paged.dims(), ref_out.dims());
        let diff = (&out_paged - &ref_out)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "paged vs direct SDPA diverge: max abs diff = {diff}");

        Ok(())
    }

    /// Non-SDPA head_dim should decline cleanly so the caller falls back.
    #[test]
    fn test_paged_decode_declines_on_unsupported_head_dim() -> Result<()> {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let head_dim = 4; // not in whitelist
        let total_slots = 16;
        let k_pool = Tensor::zeros((total_slots, 1, head_dim), DType::F32, &device)?;
        let v_pool = Tensor::zeros((total_slots, 1, head_dim), DType::F32, &device)?;
        let block_table = Tensor::new(&[0u32, 0, 0, 0][..], &device)?.reshape((1usize, 4))?;
        let q = Tensor::zeros((1, 1, 2, head_dim), DType::F32, &device)?;

        let backend = MetalBackend::new(device);
        let out = backend.flash_attn_paged_decode(&q, &k_pool, &v_pool, &block_table, 4, 4, 1.0, true)?;
        assert!(out.is_none(), "should decline unsupported head_dim");
        Ok(())
    }
}
