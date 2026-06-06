//! Metal SDPA attention helpers.
//!
//! This module owns the kt-native Metal SDPA prefill and paged-gather decode
//! paths. Paged custom-kernel attention variants live in `metal_paged.rs`; the
//! runtime trait facade delegates here for the generic SDPA-backed attention
//! routes.

use anyhow::{Context, Result};

use super::metal_config::{
    metal_sdpa_full_safe_for_q_seq, metal_sdpa_supports_head_dim, DISABLE_METAL_SDPA,
};

pub(super) fn metal_sdpa_prefill_available() -> bool {
    std::env::var(DISABLE_METAL_SDPA).is_err()
}

pub(super) fn metal_flash_attn_prefill(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !metal_sdpa_prefill_available() {
        return Ok(None);
    }
    // Decline (caller falls back to the portable path) when candle's SDPA
    // can't handle the shape/dtype. Cheaper than surfacing a kernel error
    // from inside the fused path. Guards read the kt arg directly and run
    // BEFORE the candle bridges (#1082 forward-flip).
    if !matches!(
        q.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // Last-axis index via kt-native `rank()` arithmetic so this site no longer
    // names a `candle_core::D::Minus1`-style selector through the chokepoint
    // module (#1082 chokepoint cleanup). `q` here is always at least rank 3
    // (batch, seq, hidden); the subtraction matches the previous
    // `D::Minus1` semantics.
    let head_dim = q.dim(q.rank() - 1)?;
    if !metal_sdpa_supports_head_dim(head_dim) {
        return Ok(None);
    }
    let q_seq = q.dim(2)?;
    if !metal_sdpa_full_safe_for_q_seq(head_dim, q_seq) {
        return Ok(None);
    }

    let q_t = q.transpose(1, 2)?.contiguous()?;
    let k_t = k.transpose(1, 2)?.contiguous()?;
    let v_t = v.transpose(1, 2)?.contiguous()?;

    // sdpa(q, k, v, mask, do_causal, scale, softcapping). softcapping=1.0
    // disables it; kiln's prefill path is always causal.
    let out = kiln_tensor::metal_sdpa_last_axis(&q_t, &k_t, &v_t, softmax_scale, causal)
        .context("kt-native metal sdpa (prefill) failed")?;

    let out = out.transpose(1, 2)?.contiguous()?;
    Ok(Some(out))
}

pub(super) fn metal_flash_attn_prefill_head_major(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !metal_sdpa_prefill_available() {
        return Ok(None);
    }
    // Guards read the kt arg directly, BEFORE the candle bridges (#1082).
    if !matches!(
        q.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // Last-axis index via kt-native `rank()` arithmetic; see notes above (#1082 chokepoint).
    let head_dim = q.dim(q.rank() - 1)?;
    if !metal_sdpa_supports_head_dim(head_dim) {
        return Ok(None);
    }
    let q_seq = q.dim(2)?;
    if !metal_sdpa_full_safe_for_q_seq(head_dim, q_seq) {
        return Ok(None);
    }

    let out = kiln_tensor::metal_sdpa_last_axis(q, k, v, softmax_scale, causal)
        .context("kt-native metal sdpa (head-major prefill) failed")?;
    Ok(Some(out))
}

/// Gather K/V from the paged pool via `index_select` on the block table, then
/// call candle's vectorized SDPA (single-query path). The gather replaces the
/// slow materializing `paged_cache.read` + naive-softmax+matmul fallback.
pub(super) fn metal_flash_attn_paged_decode(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    total_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<kiln_tensor::Tensor>> {
    // Gate on everything SDPA can handle. Pool dtype matches q dtype by
    // construction (both come from the same forward config), so only q needs
    // checking. Guards read the kt arg directly, BEFORE the candle bridges
    // (#1082 forward-flip).
    if !matches!(
        q.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // Last-axis index via kt-native `rank()` arithmetic; see notes above (#1082 chokepoint).
    let head_dim = q.dim(q.rank() - 1)?;
    if !metal_sdpa_supports_head_dim(head_dim) {
        return Ok(None);
    }

    let (batch, q_len, num_heads, _) = q.dims4()?;
    if batch != 1 || q_len != 1 {
        // Multi-sequence paged decode would need a per-sequence gather. Stay on
        // the fallback until the scheduler exercises it.
        return Ok(None);
    }

    let (total_slots, num_kv_heads, _) = k_pool.dims3()?;
    if total_slots % page_block_size != 0 {
        return Ok(None);
    }
    let num_blocks = total_slots / page_block_size;
    let max_blocks_per_seq = block_table.dim(1)?;

    // [num_blocks, block_size, num_kv_heads, head_dim] so index_select on dim 0
    // gathers a full logical block's slots per physical block id.
    let k_blocks = k_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;
    let v_blocks = v_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;

    // The block_table is identical across all 8 full-attention layers in a
    // decode step, but the trait forces us to re-flatten it per call. Threading
    // a pre-flattened handle through the trait would save redundant flattens
    // per token; defer until the signature can grow a cache parameter.
    let block_ids = block_table.flatten_all()?;

    let k_gathered = k_blocks.index_select(&block_ids, 0)?;
    let v_gathered = v_blocks.index_select(&block_ids, 0)?;

    // [max_blocks_per_seq * block_size, num_kv_heads, head_dim] then narrow to
    // the live KV length.
    let total_gathered = max_blocks_per_seq * page_block_size;
    let k_flat = k_gathered.reshape((total_gathered, num_kv_heads, head_dim))?;
    let v_flat = v_gathered.reshape((total_gathered, num_kv_heads, head_dim))?;
    let k_live = k_flat.narrow(0, 0, total_seqlen_k)?;
    let v_live = v_flat.narrow(0, 0, total_seqlen_k)?;

    // SDPA needs [batch, num_heads, seq, head_dim]. Q arrives as
    // [1, 1, num_heads, head_dim]; K/V are [total_seqlen_k, num_kv_heads,
    // head_dim]. SDPA handles GQA internally when num_heads % num_kv_heads == 0.
    let q_sdpa = q.transpose(1, 2)?.contiguous()?; // [1, num_heads, 1, head_dim]
    let k_sdpa = k_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // [1, num_kv_heads, total_seqlen_k, head_dim]
    let v_sdpa = v_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;

    let out = kiln_tensor::metal_sdpa_last_axis(&q_sdpa, &k_sdpa, &v_sdpa, softmax_scale, causal)
        .context("kt-native metal paged sdpa (decode) failed")?;

    // Back to [1, 1, num_heads, head_dim].
    let out = out.transpose(1, 2)?.contiguous()?;
    debug_assert_eq!(out.dims(), &[1, 1, num_heads, head_dim]);
    Ok(Some(out))
}
