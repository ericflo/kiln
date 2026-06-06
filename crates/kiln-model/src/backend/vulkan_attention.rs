//! Vulkan attention dispatch helpers.
//!
//! This module owns the explicit-resource FlashAttention prefill and paged
//! decode implementations. `backend/vulkan.rs` keeps the trait facade and
//! delegates here so attention kernels stay separate from backend construction,
//! residency registries, and optimizer dispatch.

use anyhow::{Context, Result};
use std::sync::OnceLock;

use super::vulkan::VulkanBackend;
use super::vulkan_tensor_bridge::{
    kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape, resident_sdpa_prefill_b1,
};

/// When set, the multi-batch paged attention decode path walks the
/// block_table inside the Vulkan shader instead of compacting K/V on the
/// host. Default: enabled. Disable via
/// `KILN_DISABLE_VULKAN_PAGED_DECODE_GPU_GATHER=1` to force a visible native
/// helper error for parity comparisons.
fn paged_decode_gpu_gather_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_PAGED_DECODE_GPU_GATHER").is_err())
}

fn generic_paged_decode_splitk_chunks(batch: usize, max_blocks_per_seq: usize) -> usize {
    kiln_vulkan_kernel::kernels::paged_attn_decode_splitk_chunks(batch, max_blocks_per_seq)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_vulkan_paged_decode_bytes(
    vk_device: &kiln_vulkan_kernel::VulkanDevice,
    q_data: &[u8],
    k_pool_data: &[u8],
    v_pool_data: &[u8],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    total_slots: usize,
    num_kv_heads: usize,
    block_data: &[u32],
    seq_lens: &[u32],
    max_blocks_per_seq: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<Vec<u8>> {
    let num_chunks = generic_paged_decode_splitk_chunks(batch, max_blocks_per_seq);
    if num_chunks > 1 {
        kiln_vulkan_kernel::kernels::dispatch_paged_attn_decode_batch_paged_splitk_f32_bytes(
            vk_device,
            q_data,
            k_pool_data,
            v_pool_data,
            batch,
            num_heads,
            head_dim,
            total_slots,
            num_kv_heads,
            block_data,
            seq_lens,
            max_blocks_per_seq,
            page_block_size,
            softmax_scale,
            num_chunks,
        )
        .context("Vulkan split-K paged decode kernel failed")
    } else {
        kiln_vulkan_kernel::kernels::dispatch_paged_attn_decode_batch_paged_f32_bytes(
            vk_device,
            q_data,
            k_pool_data,
            v_pool_data,
            batch,
            num_heads,
            head_dim,
            total_slots,
            num_kv_heads,
            block_data,
            seq_lens,
            max_blocks_per_seq,
            page_block_size,
            softmax_scale,
        )
        .context("Vulkan paged decode kernel failed")
    }
}

pub(super) fn flash_attn_prefill(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !backend.has_vulkan()
        || !matches!(
            q.dtype(),
            kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
        )
    {
        return Ok(None);
    }
    flash_attn_prefill_vulkan(backend, q, k, v, softmax_scale, causal)
}

fn flash_attn_prefill_vulkan(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<kiln_tensor::Tensor>> {
    let vk_device = backend
        .vulkan_device
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

    let Ok((batch, seq_len, num_heads, head_dim)) = q.dims4() else {
        return Ok(None);
    };
    let Ok((k_batch, kv_len, k_heads, k_head_dim)) = k.dims4() else {
        return Ok(None);
    };
    let Ok((v_batch, v_len, v_heads, v_head_dim)) = v.dims4() else {
        return Ok(None);
    };
    if head_dim > 256
        || kv_len != seq_len
        || k_batch != batch
        || v_batch != batch
        || v_len != seq_len
        || k_heads != num_heads
        || v_heads != num_heads
        || k_head_dim != head_dim
        || v_head_dim != head_dim
    {
        return Ok(None);
    }

    // Resident buffer-based SDPA for the common single-sequence causal case:
    // zero-copy bridge q/k/v to fused on-device `vk_sdpa_prefill`
    // (permute, batched matmul, scale, causal-mask, softmax, matmul, all
    // resident) and bridge back. batch>1 (the kernel flattens query rows to
    // one sequence, so cross-batch attention would be wrong) and non-causal
    // fall through to the bytes path below.
    if causal
        && batch == 1
        && matches!(q.device(), kiln_tensor::Device::Vulkan(_))
        && matches!(k.device(), kiln_tensor::Device::Vulkan(_))
        && matches!(v.device(), kiln_tensor::Device::Vulkan(_))
    {
        match resident_sdpa_prefill_b1(q, k, v, seq_len, num_heads, head_dim, softmax_scale) {
            Ok(out) => return Ok(Some(out)),
            Err(e) => {
                if std::env::var("KILN_VK_TRACE").is_ok() {
                    eprintln!("[vk] resident sdpa_prefill fell back to bytes: {e}");
                }
            }
        }
    }

    let in_dtype = q.dtype();
    let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
    let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
    let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
    let out_data = kiln_vulkan_kernel::kernels::dispatch_sdpa_prefill_f32_bytes(
        vk_device,
        &q_data,
        &k_data,
        &v_data,
        batch,
        seq_len,
        num_heads,
        head_dim,
        softmax_scale,
        causal,
    )?;
    let out_f32 = kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, seq_len, num_heads, head_dim],
        kiln_tensor::DType::F32,
    )?;

    let out = if in_dtype == kiln_tensor::DType::F32 {
        out_f32
    } else {
        out_f32.to_dtype(in_dtype)?
    };
    // The SDPA result is currently materialized host-side (the bytes-based
    // kernel dispatch). Keep `attn_output` on q's compute device so the
    // downstream gate / o-proj run on-device instead of mismatching
    // (vulkan gate x cpu attn_output). NOTE: the q/k/v inputs are still
    // bounced to host bytes above. A buffer-resident SDPA dispatch
    // (zero-copy q/k/v + device-resident output) is the perf follow-up to
    // remove the host round-trip entirely.
    let out = if out.device() != q.device() {
        out.to_device(q.device())?
    } else {
        out
    };
    Ok(Some(out))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attn_paged_decode(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    total_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !backend.has_vulkan()
        || !backend.paged_attn_decode_batch_enabled
        || q.dtype() != kiln_tensor::DType::F32
        || k_pool.dtype() != kiln_tensor::DType::F32
        || v_pool.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !causal {
        return Ok(None);
    }
    if !matches!(q.device(), kiln_tensor::Device::Cpu)
        || !matches!(k_pool.device(), kiln_tensor::Device::Cpu)
        || !matches!(v_pool.device(), kiln_tensor::Device::Cpu)
        || !matches!(block_table.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }

    let Ok((batch, q_len, num_heads, head_dim)) = q.dims4() else {
        return Ok(None);
    };
    let Ok((total_slots, num_kv_heads, k_head_dim)) = k_pool.dims3() else {
        return Ok(None);
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return Ok(None);
    };
    let Ok((bt_batch, max_blocks_per_seq)) = block_table.dims2() else {
        return Ok(None);
    };
    if batch == 0
        || q_len != 1
        || total_seqlen_k == 0
        || page_block_size == 0
        || head_dim > 256
        || k_head_dim != head_dim
        || v_dims != (total_slots, num_kv_heads, head_dim)
        || num_heads % num_kv_heads != 0
        || bt_batch != batch
        || total_seqlen_k.div_ceil(page_block_size) > max_blocks_per_seq
    {
        return Ok(None);
    }

    let block_data = block_table
        .flatten_all()
        .context("Vulkan paged decode: flatten block_table")?
        .to_dtype(kiln_tensor::DType::U32)
        .context("Vulkan paged decode: block_table to u32")?
        .to_vec1::<u32>()
        .context("Vulkan paged decode: read block_table")?;
    if block_data.len() != batch * max_blocks_per_seq {
        return Ok(None);
    }

    for row in 0..batch {
        let blocks_needed = total_seqlen_k.div_ceil(page_block_size).max(1);
        for block_idx in 0..blocks_needed {
            let block = block_data[row * max_blocks_per_seq + block_idx] as usize;
            let last_pos_in_block = if block_idx == blocks_needed - 1 {
                total_seqlen_k - block_idx * page_block_size - 1
            } else {
                page_block_size - 1
            };
            let last_slot = block
                .checked_mul(page_block_size)
                .and_then(|base| base.checked_add(last_pos_in_block))
                .context("Vulkan paged decode slot index overflow")?;
            if last_slot >= total_slots {
                return Ok(None);
            }
        }
    }

    let vk_device = backend
        .vulkan_device
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
    let k_pool_data = kt_tensor_to_f32_bytes_with_shape(k_pool)?.0;
    let v_pool_data = kt_tensor_to_f32_bytes_with_shape(v_pool)?.0;
    let seq_lens = vec![
        u32::try_from(total_seqlen_k)
            .context("Vulkan paged decode total_seqlen_k exceeds u32")?;
        batch
    ];

    let out_data = dispatch_vulkan_paged_decode_bytes(
        vk_device,
        &q_data,
        &k_pool_data,
        &v_pool_data,
        batch,
        num_heads,
        head_dim,
        total_slots,
        num_kv_heads,
        &block_data,
        &seq_lens,
        max_blocks_per_seq,
        page_block_size,
        softmax_scale,
    )
    .context("Vulkan paged decode batch-paged kernel failed")?;

    Ok(Some(kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, 1, num_heads, head_dim],
        kiln_tensor::DType::F32,
    )?))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
    backend: &VulkanBackend,
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !backend.has_vulkan()
        || !backend.paged_attn_decode_batch_enabled
        || q.dtype() != kiln_tensor::DType::F32
        || k_pool.dtype() != kiln_tensor::DType::F32
        || v_pool.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !causal {
        return Ok(None);
    }
    if !matches!(q.device(), kiln_tensor::Device::Cpu)
        || !matches!(k_pool.device(), kiln_tensor::Device::Cpu)
        || !matches!(v_pool.device(), kiln_tensor::Device::Cpu)
        || !matches!(block_table.device(), kiln_tensor::Device::Cpu)
        || !matches!(seqused_k.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    if !paged_decode_gpu_gather_enabled() {
        anyhow::bail!("Vulkan paged decode GPU block-table gather disabled");
    }

    let Ok((batch, q_len, num_heads, head_dim)) = q.dims4() else {
        return Ok(None);
    };
    let Ok((total_slots, num_kv_heads, k_head_dim)) = k_pool.dims3() else {
        return Ok(None);
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return Ok(None);
    };
    let Ok((bt_batch, max_blocks_per_seq)) = block_table.dims2() else {
        return Ok(None);
    };
    let Ok(seq_count) = seqused_k.dims1() else {
        return Ok(None);
    };
    if batch == 0
        || q_len != 1
        || head_dim > 256
        || k_head_dim != head_dim
        || v_dims != (total_slots, num_kv_heads, head_dim)
        || num_heads % num_kv_heads != 0
        || bt_batch != batch
        || seq_count != batch
        || page_block_size == 0
        || max_seqlen_k == 0
        || max_seqlen_k.div_ceil(page_block_size) > max_blocks_per_seq
    {
        return Ok(None);
    }

    let block_data = block_table
        .flatten_all()?
        .to_dtype(kiln_tensor::DType::U32)?
        .to_vec1::<u32>()?;
    let seq_i64 = seqused_k
        .flatten_all()?
        .to_dtype(kiln_tensor::DType::I64)?
        .to_vec1::<i64>()?;
    let mut seq_lens = Vec::with_capacity(batch);
    for row in 0..batch {
        let row_len = usize::try_from(seq_i64[row])
            .context("Vulkan paged decode seqused_k contains negative length")?;
        if row_len == 0 || row_len > max_seqlen_k {
            return Ok(None);
        }
        seq_lens
            .push(u32::try_from(row_len).context("Vulkan paged decode row length exceeds u32")?);
    }
    // Bounds-check the block_table entries that the kernel will follow. We do
    // not want the shader to OOB-read the K/V pool, so reject any out-of-range
    // (block, offset) we can prove invalid from host state. Only the slots
    // actually visited (`pos < row_len`) need to be valid.
    for row in 0..batch {
        let row_len = seq_lens[row] as usize;
        let blocks_needed = row_len.div_ceil(page_block_size).max(1);
        for block_idx in 0..blocks_needed {
            let block = block_data[row * max_blocks_per_seq + block_idx] as usize;
            let last_pos_in_block = if block_idx == blocks_needed - 1 {
                row_len - block_idx * page_block_size - 1
            } else {
                page_block_size - 1
            };
            let last_slot = block
                .checked_mul(page_block_size)
                .and_then(|base| base.checked_add(last_pos_in_block))
                .context("Vulkan paged decode slot index overflow")?;
            if last_slot >= total_slots {
                return Ok(None);
            }
        }
    }

    let vk_device = backend
        .vulkan_device
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
    let k_pool_data = kt_tensor_to_f32_bytes_with_shape(k_pool)?.0;
    let v_pool_data = kt_tensor_to_f32_bytes_with_shape(v_pool)?.0;
    let out_data = dispatch_vulkan_paged_decode_bytes(
        vk_device,
        &q_data,
        &k_pool_data,
        &v_pool_data,
        batch,
        num_heads,
        head_dim,
        total_slots,
        num_kv_heads,
        &block_data,
        &seq_lens,
        max_blocks_per_seq,
        page_block_size,
        softmax_scale,
    )
    .context("paged_attn_decode_batch_paged kernel failed")?;
    Ok(Some(kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, 1, num_heads, head_dim],
        kiln_tensor::DType::F32,
    )?))
}
