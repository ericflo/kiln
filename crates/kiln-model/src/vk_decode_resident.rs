//! Vulkan-resident decode wire-up (gate (a)/(e) of
//! `docs/vk_resident_decode_plan.md`).
//!
//! Composes the per-kernel `_resident` dispatchers in
//! `kiln_vulkan_kernel::resident` into a per-layer block helper
//! ([`transformer_block_paged_decode_full_attn_resident_b1`]) that
//! threads device-local activation buffers through one
//! `CommandBatch` per layer — so a full decode step submits
//! `O(num_layers)` times (one submit per layer) instead of
//! `O(num_layers × kernels_per_layer)`.
//!
//! Scope is intentionally narrow at this revision:
//!
//! - batch = 1, seq_len = 1, start_pos > 0 (the decode hot path)
//! - `attn_output_gate = true` (always on for Qwen3.5-4B)
//! - no LoRA, no debug taps, no MTP, no Marlin
//!
//! Any unsupported config returns `Ok(None)` so the caller can
//! fall back transparently to the legacy
//! `transformer_block_paged_with_rope_tables`. The lifted overhead
//! is the per-kernel `extract + upload + readback` boundary that
//! dominates the legacy Vulkan decode path at 1.04 tok/s on
//! Qwen3.5-4B — the resident microbench measures the same kernel
//! sequence at 29 tok/s when chained through a `CommandBatch`.

#![cfg(feature = "vulkan")]

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;

use kiln_vulkan_kernel::shaders as shaders;
use kiln_vulkan_kernel::{CommandBatch, VkPagedKvCache, VulkanBuffer, VulkanDevice, Workgroups};

use crate::backend::vulkan::VulkanBackend;
use crate::forward::GpuLayerWeights;
use crate::paged_kv_cache::PagedKvCache;

/// Run one full-attention decode block on the Vulkan-resident path.
///
/// Returns `Ok(Some(output_tensor))` on success — the post-MLP residual,
/// shape `[1, 1, hidden_size]`, `DType::F32`, on the same candle device
/// as `x`.
///
/// Returns `Ok(None)` when the input does not match the supported
/// configuration (see module docs); the caller should fall back to the
/// legacy block helper.
///
/// The KV cache write lands in the supplied `VkPagedKvCache` at the
/// block-table-resolved slot for `start_pos`. The legacy `PagedKvCache`
/// is **not** updated — once the resident path is engaged for a layer,
/// it owns that layer's KV state for the remainder of the decode
/// session.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_paged_decode_full_attn_resident_b1(
    backend: &VulkanBackend,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    start_pos: usize,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    paged_cache: &PagedKvCache,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
) -> Result<Option<Tensor>> {
    // --- supported-config gate ---------------------------------------
    let dims = x.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 {
        return Ok(None);
    }
    let hidden = dims[2];
    if hidden != config.hidden_size {
        return Ok(None);
    }
    let attn = match &layer.attention {
        crate::forward::GpuAttentionWeights::Full(w) => w,
        _ => return Ok(None),
    };
    if !config.attn_output_gate {
        return Ok(None);
    }
    let Some(vk_device) = backend.vulkan_device() else {
        return Ok(None);
    };

    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let rotary_dim = config.rotary_dim();
    let intermediate = config.intermediate_size;
    let eps = config.rms_norm_eps as f32;
    let softmax_scale = (head_dim as f32).sqrt().recip();
    let q_dim = num_heads * head_dim * 2; // attn_output_gate doubles q output
    let k_dim = num_kv_heads * head_dim;
    let v_dim = num_kv_heads * head_dim;
    let kv_elems = num_kv_heads * head_dim;

    let block_size = paged_cache.block_size();
    let slot = block_table
        .slot_for(start_pos, block_size)
        .ok_or_else(|| {
            anyhow::anyhow!("no slot for start_pos {start_pos} in block table")
        })?;

    // --- weight buffer lookups (cached on backend) -------------------
    // Q/K/V/O projections + MLP gate/up/down: bf16-packed
    let q_w_buf = backend.cached_bf16_packed_weight_buffer(&attn.q_proj_t)?;
    let k_w_buf = backend.cached_bf16_packed_weight_buffer(&attn.k_proj_t)?;
    let v_w_buf = backend.cached_bf16_packed_weight_buffer(&attn.v_proj_t)?;
    let o_w_buf = backend.cached_bf16_packed_weight_buffer(&attn.o_proj_t)?;
    let gate_w_buf = backend.cached_bf16_packed_weight_buffer(&layer.mlp.gate_proj_t)?;
    let up_w_buf = backend.cached_bf16_packed_weight_buffer(&layer.mlp.up_proj_t)?;
    let down_w_buf = backend.cached_bf16_packed_weight_buffer(&layer.mlp.down_proj_t)?;

    // RMSnorm weights: f32 (the candle storage may be bf16; the cache
    // helper converts on first lookup).
    let in_norm_buf = backend.cached_f32_weight_buffer(&layer.input_layernorm)?;
    let post_norm_buf = backend.cached_f32_weight_buffer(&layer.post_attention_layernorm)?;
    let q_norm_buf = backend.cached_f32_weight_buffer(&attn.q_norm)?;
    let k_norm_buf = backend.cached_f32_weight_buffer(&attn.k_norm)?;

    // --- rope cos/sin upload (per-step, single position) -------------
    let rope_cos_buf = upload_tensor_f32(vk_device, rope_cos)?;
    let rope_sin_buf = upload_tensor_f32(vk_device, rope_sin)?;

    // --- activation buffer allocation --------------------------------
    let mk = |bytes: u64| -> Result<VulkanBuffer> {
        VulkanBuffer::create_device_local(
            vk_device.device(),
            vk_device.device_local_mem_type(),
            bytes,
        )
        .context("alloc activation buffer")
    };
    let x_buf = mk((hidden * 4) as u64)?;
    let normed_buf = mk((hidden * 4) as u64)?;
    let qkv_combined = mk(((q_dim + k_dim + v_dim) * 4) as u64)?;
    let q_buf = mk((num_heads * head_dim * 4) as u64)?;
    let q_rot_buf = mk((num_heads * head_dim * 4) as u64)?;
    let gate_buf = mk((num_heads * head_dim * 4) as u64)?;
    let k_buf = mk((kv_elems * 4) as u64)?;
    let k_rot_buf = mk((kv_elems * 4) as u64)?;
    let v_buf = mk((kv_elems * 4) as u64)?;
    let attn_pre_gate = mk((num_heads * head_dim * 4) as u64)?;
    let attn_post_gate = mk((num_heads * head_dim * 4) as u64)?;
    let attn_out_buf = mk((hidden * 4) as u64)?;
    let attn_residual = mk((hidden * 4) as u64)?;
    let mlp_scratch = mk((intermediate * 4) as u64)?;
    let mlp_out_buf = mk((hidden * 4) as u64)?;
    let final_out = mk((hidden * 4) as u64)?;

    // block_table + seq_lens for paged-attn read
    let blocks: Vec<u32> = block_table.blocks.clone();
    let max_blocks_per_seq = blocks.len();
    let block_table_buf = upload_u32_slice(vk_device, &blocks)?;
    let seq_lens: [u32; 1] = [(start_pos + 1) as u32];
    let seq_lens_buf = upload_u32_slice(vk_device, &seq_lens)?;

    // --- upload x ----------------------------------------------------
    let x_f32 = if x.dtype() == DType::F32 {
        x.flatten_all()?
    } else {
        x.to_dtype(DType::F32)?.flatten_all()?
    };
    let x_data: Vec<f32> = x_f32.to_vec1()?;
    let x_bytes: Vec<u8> = f32_slice_to_bytes(&x_data);
    VulkanBuffer::upload_data(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &x_buf,
        &x_bytes,
    )
    .context("upload x for resident block")?;

    // --- chain all 15 dispatches into one CommandBatch + one submit ---
    let k_pool = vk_kv_cache.k_buffer(full_attn_layer_idx).ok_or_else(|| {
        anyhow::anyhow!("VkPagedKvCache missing layer {full_attn_layer_idx}")
    })?;
    let v_pool = vk_kv_cache.v_buffer(full_attn_layer_idx).ok_or_else(|| {
        anyhow::anyhow!("VkPagedKvCache missing layer {full_attn_layer_idx}")
    })?;
    let elements_per_slot = num_kv_heads * head_dim;
    let total_qkv_out = q_dim + k_dim + v_dim;
    let q_h_d = num_heads * head_dim;

    let mut batch = CommandBatch::new(vk_device)?;

    // 1) pre-attn rmsnorm
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[x_buf.handle(), in_norm_buf.handle(), normed_buf.handle()],
        &[1u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(1),
    )?;
    // 2) combined QKV projection (bf16 weights)
    batch.record_shader(
        shaders::FULL_ATTN_QKV_DECODE_BF16W,
        &[
            normed_buf.handle(),
            q_w_buf.handle(),
            k_w_buf.handle(),
            v_w_buf.handle(),
            qkv_combined.handle(),
        ],
        &[
            hidden as u32,
            q_dim as u32,
            k_dim as u32,
            v_dim as u32,
            total_qkv_out as u32,
        ],
        Workgroups::OneD(total_qkv_out.div_ceil(16) as u32),
    )?;
    // 3) gate-split
    let total_split = num_heads * head_dim + 2 * (num_kv_heads * head_dim);
    batch.record_shader(
        shaders::QKV_GATE_SPLIT,
        &[
            qkv_combined.handle(),
            q_buf.handle(),
            gate_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
        ],
        &[num_heads as u32, num_kv_heads as u32, head_dim as u32],
        Workgroups::OneD(total_split.div_ceil(64) as u32),
    )?;
    // 4) Q-norm (per-head). qwen_rmsnorm_forward push: [rows, hidden, eps].
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[q_buf.handle(), q_norm_buf.handle(), q_buf.handle()],
        &[num_heads as u32, head_dim as u32, eps.to_bits()],
        Workgroups::OneD(num_heads as u32),
    )?;
    // 5) K-norm (per-KV-head)
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[k_buf.handle(), k_norm_buf.handle(), k_buf.handle()],
        &[num_kv_heads as u32, head_dim as u32, eps.to_bits()],
        Workgroups::OneD(num_kv_heads as u32),
    )?;
    // 6) RoPE Q. vk_rope_f32 push: [rows, num_heads, head_dim, rotary_dim].
    batch.record_shader(
        shaders::VK_ROPE_F32,
        &[
            q_buf.handle(),
            rope_cos_buf.handle(),
            rope_sin_buf.handle(),
            q_rot_buf.handle(),
        ],
        &[1u32, num_heads as u32, head_dim as u32, rotary_dim as u32],
        Workgroups::OneD((num_heads * head_dim).div_ceil(256) as u32),
    )?;
    // 7) RoPE K
    batch.record_shader(
        shaders::VK_ROPE_F32,
        &[
            k_buf.handle(),
            rope_cos_buf.handle(),
            rope_sin_buf.handle(),
            k_rot_buf.handle(),
        ],
        &[1u32, num_kv_heads as u32, head_dim as u32, rotary_dim as u32],
        Workgroups::OneD((num_kv_heads * head_dim).div_ceil(256) as u32),
    )?;
    // 8) Write K/V into Vulkan-resident paged pool
    batch.record_shader(
        shaders::PAGED_KV_WRITE_SLOT,
        &[
            k_rot_buf.handle(),
            v_buf.handle(),
            k_pool.handle(),
            v_pool.handle(),
        ],
        &[slot as u32, elements_per_slot as u32],
        Workgroups::OneD(elements_per_slot.div_ceil(64) as u32),
    )?;
    // 9) Paged-paged attention. Push: [max_blocks_per_seq, page_block_size,
    //    num_heads, num_kv_heads, head_dim, softmax_scale_bits].
    batch.record_shader(
        shaders::PAGED_ATTN_DECODE_BATCH_PAGED,
        &[
            q_rot_buf.handle(),
            k_pool.handle(),
            v_pool.handle(),
            block_table_buf.handle(),
            seq_lens_buf.handle(),
            attn_pre_gate.handle(),
        ],
        &[
            max_blocks_per_seq as u32,
            block_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            softmax_scale.to_bits(),
        ],
        Workgroups::OneD(num_heads as u32), // batch=1 × num_heads
    )?;
    // 10) Attention output gate: out = pre_gate * sigmoid(gate)
    batch.record_shader(
        shaders::VK_MUL_SIGMOID_GATE_F32,
        &[
            attn_pre_gate.handle(),
            gate_buf.handle(),
            attn_post_gate.handle(),
        ],
        &[q_h_d as u32],
        Workgroups::OneD((q_h_d).div_ceil(256) as u32),
    )?;
    // 11) Output projection (bf16 weights, b=1). linear_decode_bf16w push:
    //     [hidden_in, out_dim].
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[
            attn_post_gate.handle(),
            o_w_buf.handle(),
            attn_out_buf.handle(),
        ],
        &[q_h_d as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    // 12) Residual: attn_residual = x + attn_out
    batch.record_shader(
        shaders::ADD,
        &[
            x_buf.handle(),
            attn_out_buf.handle(),
            attn_residual.handle(),
        ],
        &[hidden as u32],
        Workgroups::OneD(hidden.div_ceil(256) as u32),
    )?;
    // 13) Pre-MLP norm
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[attn_residual.handle(), post_norm_buf.handle(), normed_buf.handle()],
        &[1u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(1),
    )?;
    // 14) MLP gate-up (b=1). Push: [hidden, intermediate]. Workgroups:
    //     intermediate.div_ceil(64).
    batch.record_shader(
        shaders::MLP_GATE_UP_DECODE_BF16W,
        &[
            normed_buf.handle(),
            gate_w_buf.handle(),
            up_w_buf.handle(),
            mlp_scratch.handle(),
        ],
        &[hidden as u32, intermediate as u32],
        Workgroups::OneD(intermediate.div_ceil(64) as u32),
    )?;
    // 15) MLP down: linear_decode_bf16w(scratch, down_w, mlp_out).
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[mlp_scratch.handle(), down_w_buf.handle(), mlp_out_buf.handle()],
        &[intermediate as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    // 16) Final residual: final_out = attn_residual + mlp_out
    batch.record_shader(
        shaders::ADD,
        &[
            attn_residual.handle(),
            mlp_out_buf.handle(),
            final_out.handle(),
        ],
        &[hidden as u32],
        Workgroups::OneD(hidden.div_ceil(256) as u32),
    )?;

    batch
        .submit_and_wait("vk-resident full-attn block")
        .context("submit resident full-attn CommandBatch")?;

    // --- read back final_out as a candle Tensor on the input's device
    let out_bytes = VulkanBuffer::read_back(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &final_out,
    )
    .context("read back final_out")?;
    let out_f32: Vec<f32> = bytes_to_f32_vec(&out_bytes);
    let out_tensor =
        Tensor::from_vec(out_f32, (1usize, 1usize, hidden), x.device())?.to_dtype(x.dtype())?;
    Ok(Some(out_tensor))
}

fn upload_tensor_f32(vk_device: &VulkanDevice, t: &Tensor) -> Result<VulkanBuffer> {
    kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(vk_device, t)
}

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn u32_slice_to_bytes(data: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
        out.push(f32::from_le_bytes(buf));
    }
    out
}

fn upload_u32_slice(vk_device: &VulkanDevice, data: &[u32]) -> Result<VulkanBuffer> {
    let bytes: Vec<u8> = u32_slice_to_bytes(data);
    let buf = VulkanBuffer::create_device_local(
        vk_device.device(),
        vk_device.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &buf,
        &bytes,
    )?;
    Ok(buf)
}

/// Seed the Vulkan-resident KV pool from the legacy candle paged
/// cache for one layer. Uploads the entire pool slot range as f32
/// regardless of how many positions are actually live — the legacy
/// path zero-initialises unused slots so reading them produces zero
/// (and the attention seq-len mask makes them irrelevant).
pub fn seed_vk_kv_cache_layer_from_legacy(
    vk_device: &VulkanDevice,
    vk_cache: &VkPagedKvCache,
    paged_cache: &PagedKvCache,
    layer_idx: usize,
) -> Result<()> {
    let (k_tensor, v_tensor) = paged_cache
        .pool_tensors(layer_idx)
        .ok_or_else(|| anyhow::anyhow!("legacy paged_cache layer {layer_idx} out of range"))?;
    let k_flat = if k_tensor.dtype() == DType::F32 {
        k_tensor.flatten_all()?
    } else {
        k_tensor.to_dtype(DType::F32)?.flatten_all()?
    };
    let v_flat = if v_tensor.dtype() == DType::F32 {
        v_tensor.flatten_all()?
    } else {
        v_tensor.to_dtype(DType::F32)?.flatten_all()?
    };
    let k_data: Vec<f32> = k_flat.to_vec1()?;
    let v_data: Vec<f32> = v_flat.to_vec1()?;
    let k_bytes: Vec<u8> = f32_slice_to_bytes(&k_data);
    let v_bytes: Vec<u8> = f32_slice_to_bytes(&v_data);
    vk_cache.upload_layer_from_f32(vk_device, layer_idx, &k_bytes, &v_bytes)
}
