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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use kiln_vulkan_kernel::shaders as shaders;
use kiln_vulkan_kernel::{CommandBatch, VkPagedKvCache, VulkanBuffer, VulkanDevice, Workgroups};

use crate::backend::vulkan::VulkanBackend;
use crate::forward::GpuLayerWeights;
use crate::paged_kv_cache::PagedKvCache;

// Env-gated per-block timing accumulators. Enable with
// `KILN_VK_RESIDENT_DECODE_TIMING=1`. Each accumulator records nanos
// across (block_total, upload, submit_wait, readback). Call sites
// also drive call counts so the average per layer is recoverable.
static TIMING_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static FA_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static FA_UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
static FA_SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
static FA_READBACK_NS: AtomicU64 = AtomicU64::new(0);
static FA_CALLS: AtomicUsize = AtomicUsize::new(0);
static GDN_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static GDN_UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
static GDN_SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
static GDN_READBACK_NS: AtomicU64 = AtomicU64::new(0);
static GDN_CALLS: AtomicUsize = AtomicUsize::new(0);
static GDNFB_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static GDNFB_UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
static GDNFB_SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
static GDNFB_READBACK_NS: AtomicU64 = AtomicU64::new(0);
static GDNFB_CALLS: AtomicUsize = AtomicUsize::new(0);

fn timing_enabled() -> bool {
    *TIMING_ENABLED.get_or_init(|| {
        std::env::var("KILN_VK_RESIDENT_DECODE_TIMING")
            .map(|v| !matches!(v.trim(), "" | "0" | "false" | "off" | "no"))
            .unwrap_or(false)
    })
}

/// Print accumulated per-block timing to stderr and zero the counters.
/// Call from the bench / harness once per decode token to see a
/// per-token breakdown.
pub fn drain_resident_decode_timing() {
    if !timing_enabled() {
        return;
    }
    let fa_calls = FA_CALLS.swap(0, Ordering::SeqCst);
    let gdn_calls = GDN_CALLS.swap(0, Ordering::SeqCst);
    let gdnfb_calls = GDNFB_CALLS.swap(0, Ordering::SeqCst);
    if fa_calls == 0 && gdn_calls == 0 && gdnfb_calls == 0 {
        return;
    }
    let fa_total = FA_TOTAL_NS.swap(0, Ordering::SeqCst);
    let fa_upload = FA_UPLOAD_NS.swap(0, Ordering::SeqCst);
    let fa_submit = FA_SUBMIT_NS.swap(0, Ordering::SeqCst);
    let fa_readback = FA_READBACK_NS.swap(0, Ordering::SeqCst);
    let gdn_total = GDN_TOTAL_NS.swap(0, Ordering::SeqCst);
    let gdn_upload = GDN_UPLOAD_NS.swap(0, Ordering::SeqCst);
    let gdn_submit = GDN_SUBMIT_NS.swap(0, Ordering::SeqCst);
    let gdn_readback = GDN_READBACK_NS.swap(0, Ordering::SeqCst);
    let gdnfb_total = GDNFB_TOTAL_NS.swap(0, Ordering::SeqCst);
    let gdnfb_upload = GDNFB_UPLOAD_NS.swap(0, Ordering::SeqCst);
    let gdnfb_submit = GDNFB_SUBMIT_NS.swap(0, Ordering::SeqCst);
    let gdnfb_readback = GDNFB_READBACK_NS.swap(0, Ordering::SeqCst);
    let ms = |ns: u64| (ns as f64) / 1e6;
    eprintln!(
        "[vk-resident-timing] full-attn calls={fa_calls} total={:.2}ms upload={:.2}ms submit={:.2}ms readback={:.2}ms cpu={:.2}ms",
        ms(fa_total),
        ms(fa_upload),
        ms(fa_submit),
        ms(fa_readback),
        ms(fa_total.saturating_sub(fa_upload + fa_submit + fa_readback)),
    );
    eprintln!(
        "[vk-resident-timing] GDN       calls={gdn_calls} total={:.2}ms upload={:.2}ms submit={:.2}ms readback={:.2}ms cpu={:.2}ms",
        ms(gdn_total),
        ms(gdn_upload),
        ms(gdn_submit),
        ms(gdn_readback),
        ms(gdn_total.saturating_sub(gdn_upload + gdn_submit + gdn_readback)),
    );
    eprintln!(
        "[vk-resident-timing] GDN-fblk  calls={gdnfb_calls} total={:.2}ms upload={:.2}ms submit={:.2}ms readback={:.2}ms cpu={:.2}ms",
        ms(gdnfb_total),
        ms(gdnfb_upload),
        ms(gdnfb_submit),
        ms(gdnfb_readback),
        ms(gdnfb_total.saturating_sub(gdnfb_upload + gdnfb_submit + gdnfb_readback)),
    );
}

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

    // --- activation buffer acquisition (pooled, persistent across
    //     resident decode calls on the same backend) -----------------
    let x_buf = backend.acquire_resident_scratch("fa_x", (hidden * 4) as u64)?;
    let normed_buf = backend.acquire_resident_scratch("fa_normed", (hidden * 4) as u64)?;
    let qkv_combined = backend.acquire_resident_scratch(
        "fa_qkv_combined",
        ((q_dim + k_dim + v_dim) * 4) as u64,
    )?;
    let q_buf = backend.acquire_resident_scratch("fa_q", (num_heads * head_dim * 4) as u64)?;
    let q_rot_buf =
        backend.acquire_resident_scratch("fa_q_rot", (num_heads * head_dim * 4) as u64)?;
    let gate_buf = backend.acquire_resident_scratch("fa_gate", (num_heads * head_dim * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("fa_k", (kv_elems * 4) as u64)?;
    let k_rot_buf = backend.acquire_resident_scratch("fa_k_rot", (kv_elems * 4) as u64)?;
    let v_buf = backend.acquire_resident_scratch("fa_v", (kv_elems * 4) as u64)?;
    let attn_pre_gate = backend.acquire_resident_scratch(
        "fa_attn_pre_gate",
        (num_heads * head_dim * 4) as u64,
    )?;
    let attn_post_gate = backend.acquire_resident_scratch(
        "fa_attn_post_gate",
        (num_heads * head_dim * 4) as u64,
    )?;
    let attn_out_buf = backend.acquire_resident_scratch("fa_attn_out", (hidden * 4) as u64)?;
    let attn_residual = backend.acquire_resident_scratch("fa_attn_residual", (hidden * 4) as u64)?;
    let mlp_scratch = backend.acquire_resident_scratch("fa_mlp_scratch", (intermediate * 4) as u64)?;
    let mlp_out_buf = backend.acquire_resident_scratch("fa_mlp_out", (hidden * 4) as u64)?;
    let final_out = backend.acquire_resident_scratch("fa_final_out", (hidden * 4) as u64)?;

    // block_table + seq_lens for paged-attn read
    let blocks: Vec<u32> = block_table.blocks.clone();
    let max_blocks_per_seq = blocks.len();
    let block_table_buf = upload_u32_slice(vk_device, &blocks)?;
    let seq_lens: [u32; 1] = [(start_pos + 1) as u32];
    let seq_lens_buf = upload_u32_slice(vk_device, &seq_lens)?;

    let fa_t0 = if timing_enabled() { Some(Instant::now()) } else { None };

    // --- upload x ----------------------------------------------------
    let upload_t0 = fa_t0.map(|_| Instant::now());
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
    if let Some(t) = upload_t0 {
        FA_UPLOAD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

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

    let submit_t0 = fa_t0.map(|_| Instant::now());
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
    if let Some(t) = submit_t0 {
        FA_SUBMIT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // --- read back final_out as a candle Tensor on the input's device
    let readback_t0 = fa_t0.map(|_| Instant::now());
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
    if let Some(t) = readback_t0 {
        FA_READBACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Some(t) = fa_t0 {
        FA_TOTAL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        FA_CALLS.fetch_add(1, Ordering::Relaxed);
    }
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

/// Run the GDN linear-attention sub-block on the Vulkan-resident path.
///
/// Returns `Ok(Some(output))` — the attention output shape `[1, 1, hidden]`
/// f32, on the same candle device as `x`. The caller is responsible for
/// the residual add and the post-attention norm + MLP (this helper is
/// only the GDN-specific portion).
///
/// Returns `Ok(None)` when the input does not match the supported
/// Full transformer block for a GDN (linear-attention) layer, end to
/// end on the Vulkan-resident path.
///
/// Mirrors [`transformer_block_paged_decode_full_attn_resident_b1`] but
/// with GDN compute in place of full attention. The block is everything
/// the legacy orchestration does for a GDN layer:
///
///   pre_norm → GDN(in_proj + split + conv1d + qkv_split + qk_norm +
///   fused_gates_recurrent_rmsnorm + out_proj) → +residual → post_norm
///   → MLP gate_up + down → +residual
///
/// Lifting all of this into one `CommandBatch` per layer eliminates the
/// 24 GDN layers × 5 candle ops (pre-norm + residual + post-norm + MLP
/// + final-residual) that previously dominated decode ITL (~17 ms /
/// GDN layer = ~408 ms / token observed via `KILN_VK_RESIDENT_DECODE_TIMING`).
///
/// Returns `Ok(Some(post_block_residual_tensor))` on success;
/// `Ok(None)` on any unsupported configuration so the caller falls back
/// to the legacy path bit-identically.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_paged_decode_gdn_resident_b1(
    backend: &VulkanBackend,
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    recurrent_state_t: &Tensor,
    conv_state_t: &Tensor,
) -> Result<Option<Tensor>> {
    let state_key = recurrent_state_t.id();
    let dims = x.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 {
        return Ok(None);
    }
    let hidden = dims[2];
    if hidden != config.hidden_size {
        return Ok(None);
    }
    let lin_weights = match &layer.attention {
        crate::forward::GpuAttentionWeights::Linear(w) => w,
        _ => return Ok(None),
    };
    let Some(vk_device) = backend.vulkan_device() else {
        return Ok(None);
    };

    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;
    let qkv_dim = 2 * qk_dim + v_dim;
    let z_dim = v_dim;
    let a_dim = nv;
    let b_dim = nv;
    let in_proj_total = qkv_dim + z_dim + a_dim + b_dim;
    let conv_kernel = config.linear_conv_kernel_dim;
    let intermediate = config.intermediate_size;
    let eps = config.rms_norm_eps as f32;

    let fb_t0 = if timing_enabled() { Some(Instant::now()) } else { None };

    // --- weight buffer lookups (cached on backend) -------------------
    let qkv_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.in_proj_qkv_t)?;
    let z_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.in_proj_z_t)?;
    let a_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.in_proj_a_t)?;
    let b_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.in_proj_b_t)?;
    let out_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.out_proj_t)?;
    let conv_w = backend.cached_f32_weight_buffer(&lin_weights.conv1d)?;
    let qk_norm = backend.cached_f32_weight_buffer(&lin_weights.norm)?;
    let a_log = backend.cached_f32_weight_buffer(&lin_weights.a_log)?;
    let dt_bias = backend.cached_f32_weight_buffer(&lin_weights.dt_bias)?;
    let gate_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.gate_proj_t)?;
    let up_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.up_proj_t)?;
    let down_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.down_proj_t)?;
    let in_norm = backend.cached_f32_weight_buffer(&layer.input_layernorm)?;
    let post_norm = backend.cached_f32_weight_buffer(&layer.post_attention_layernorm)?;

    // --- persistent state buffers --------------------------------
    let recurrent_bytes = (1 * nv * dk * dv * 4) as u64;
    let recurrent_buf =
        backend.linear_attn_recurrent_state_buffer(state_key, recurrent_bytes)?;
    let conv_state_bytes = (1 * qkv_dim * (conv_kernel.saturating_sub(1)) * 4) as u64;
    let conv_buf = backend.linear_attn_conv_state_buffer(state_key, conv_state_bytes)?;

    if !backend.linear_attn_layer_seeded(state_key) {
        seed_recurrent_state(vk_device, &recurrent_buf, recurrent_state_t)?;
        seed_conv_state(vk_device, &conv_buf, conv_state_t)?;
        backend.mark_linear_attn_layer_seeded(state_key);
    }

    // --- pooled scratch buffers (own keyspace from full-attn / per-call GDN) ---
    let x_buf = backend.acquire_resident_scratch("gdnfb_x", (hidden * 4) as u64)?;
    let normed_pre = backend.acquire_resident_scratch("gdnfb_normed_pre", (hidden * 4) as u64)?;
    let in_proj_out =
        backend.acquire_resident_scratch("gdnfb_in_proj_out", (in_proj_total * 4) as u64)?;
    let mixed_qkv = backend.acquire_resident_scratch("gdnfb_mixed_qkv", (qkv_dim * 4) as u64)?;
    let conv_qkv = backend.acquire_resident_scratch("gdnfb_conv_qkv", (qkv_dim * 4) as u64)?;
    let z_buf = backend.acquire_resident_scratch("gdnfb_z", (z_dim * 4) as u64)?;
    let a_buf = backend.acquire_resident_scratch("gdnfb_a", (a_dim * 4) as u64)?;
    let b_buf = backend.acquire_resident_scratch("gdnfb_b", (b_dim * 4) as u64)?;
    let q_buf = backend.acquire_resident_scratch("gdnfb_q", (qk_dim * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("gdnfb_k", (qk_dim * 4) as u64)?;
    let v_buf = backend.acquire_resident_scratch("gdnfb_v", (v_dim * 4) as u64)?;
    let gated_norm =
        backend.acquire_resident_scratch("gdnfb_gated_norm", (v_dim * 4) as u64)?;
    let gdn_out = backend.acquire_resident_scratch("gdnfb_gdn_out", (hidden * 4) as u64)?;
    let attn_residual =
        backend.acquire_resident_scratch("gdnfb_attn_residual", (hidden * 4) as u64)?;
    let normed_post =
        backend.acquire_resident_scratch("gdnfb_normed_post", (hidden * 4) as u64)?;
    let mlp_scratch =
        backend.acquire_resident_scratch("gdnfb_mlp_scratch", (intermediate * 4) as u64)?;
    let mlp_out = backend.acquire_resident_scratch("gdnfb_mlp_out", (hidden * 4) as u64)?;
    let final_out = backend.acquire_resident_scratch("gdnfb_final_out", (hidden * 4) as u64)?;

    // --- upload x ------------------------------------------------
    let upload_t0 = fb_t0.map(|_| Instant::now());
    let x_f32 = if x.dtype() == DType::F32 {
        x.flatten_all()?
    } else {
        x.to_dtype(DType::F32)?.flatten_all()?
    };
    let x_data: Vec<f32> = x_f32.to_vec1()?;
    VulkanBuffer::upload_data(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &x_buf,
        &f32_slice_to_bytes(&x_data),
    )
    .context("upload x for GDN full-block resident")?;
    if let Some(t) = upload_t0 {
        GDNFB_UPLOAD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // --- chain all 14 dispatches into one CommandBatch + one submit ---
    let submit_t0 = fb_t0.map(|_| Instant::now());
    let mut batch = CommandBatch::new(vk_device)?;

    // 1) pre-attn rmsnorm: x → normed_pre
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[x_buf.handle(), in_norm.handle(), normed_pre.handle()],
        &[1u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(1),
    )?;
    // 2) GDN in_proj (bf16w, b=1)
    batch.record_shader(
        shaders::GDN_IN_PROJ_DECODE_BF16W,
        &[
            normed_pre.handle(),
            qkv_w.handle(),
            z_w.handle(),
            a_w.handle(),
            b_w.handle(),
            in_proj_out.handle(),
        ],
        &[
            hidden as u32,
            qkv_dim as u32,
            z_dim as u32,
            a_dim as u32,
            b_dim as u32,
            in_proj_total as u32,
        ],
        Workgroups::OneD(in_proj_total.div_ceil(16) as u32),
    )?;
    // 3) in_proj split → (mixed_qkv, z, a, b)
    batch.record_shader(
        shaders::GDN_IN_PROJ_SPLIT,
        &[
            in_proj_out.handle(),
            mixed_qkv.handle(),
            z_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
        ],
        &[qkv_dim as u32, z_dim as u32, a_dim as u32, b_dim as u32],
        Workgroups::OneD(in_proj_total.div_ceil(64) as u32),
    )?;
    // 4a) causal_conv1d stage 1: output
    let conv_total = 1 * qkv_dim * 1;
    batch.record_shader(
        shaders::CAUSAL_CONV1D,
        &[
            mixed_qkv.handle(),
            conv_w.handle(),
            conv_buf.handle(),
            conv_qkv.handle(),
        ],
        &[1u32, qkv_dim as u32, 1u32, conv_kernel as u32],
        Workgroups::OneD(conv_total.div_ceil(256) as u32),
    )?;
    // 4b) causal_conv1d stage 2: state advance
    batch.record_shader(
        shaders::CAUSAL_CONV1D_STATE_ADVANCE,
        &[mixed_qkv.handle(), conv_buf.handle()],
        &[1u32, qkv_dim as u32, 1u32, conv_kernel as u32],
        Workgroups::OneD((1 * qkv_dim) as u32),
    )?;
    // 5) split conv_qkv → (q, k, v)
    batch.record_shader(
        shaders::GDN_QKV_SPLIT,
        &[
            conv_qkv.handle(),
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
        ],
        &[qk_dim as u32, v_dim as u32],
        Workgroups::OneD((2 * qk_dim + v_dim).div_ceil(64) as u32),
    )?;
    // 6) Q-norm
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[q_buf.handle(), qk_norm.handle(), q_buf.handle()],
        &[nk as u32, dk as u32, eps.to_bits()],
        Workgroups::OneD(nk as u32),
    )?;
    // 7) K-norm (rows=nk same as legacy dispatcher)
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[k_buf.handle(), qk_norm.handle(), k_buf.handle()],
        &[nk as u32, dk as u32, eps.to_bits()],
        Workgroups::OneD(nk as u32),
    )?;
    // 8) Fused gates+recurrent+rmsnorm
    batch.record_shader(
        shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
        &[
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
            a_log.handle(),
            dt_bias.handle(),
            recurrent_buf.handle(),
            z_buf.handle(),
            qk_norm.handle(),
            gated_norm.handle(),
        ],
        &[
            nv as u32,
            dk as u32,
            dv as u32,
            eps.to_bits(),
            1u32,
        ],
        Workgroups::OneD(nv as u32),
    )?;
    // 9) GDN out_proj (bf16w b=1) → gdn_out
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[gated_norm.handle(), out_w.handle(), gdn_out.handle()],
        &[v_dim as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    // 10) Residual: attn_residual = x + gdn_out
    batch.record_shader(
        shaders::ADD_QWEN_RMSNORM,
        &[
            x_buf.handle(),
            gdn_out.handle(),
            post_norm.handle(),
            attn_residual.handle(),
            normed_post.handle(),
        ],
        &[hidden as u32, eps.to_bits()],
        Workgroups::OneD(1),
    )?;
    // 12) MLP gate_up (the ADD + pre-MLP norm were fused above)
    batch.record_shader(
        shaders::MLP_GATE_UP_DECODE_BF16W,
        &[
            normed_post.handle(),
            gate_w.handle(),
            up_w.handle(),
            mlp_scratch.handle(),
        ],
        &[hidden as u32, intermediate as u32],
        Workgroups::OneD(intermediate.div_ceil(64) as u32),
    )?;
    // 13) MLP down (linear_decode_bf16w(scratch, down_w, mlp_out))
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[mlp_scratch.handle(), down_w.handle(), mlp_out.handle()],
        &[intermediate as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    // 14) Final residual: final_out = attn_residual + mlp_out
    batch.record_shader(
        shaders::ADD,
        &[
            attn_residual.handle(),
            mlp_out.handle(),
            final_out.handle(),
        ],
        &[hidden as u32],
        Workgroups::OneD(hidden.div_ceil(256) as u32),
    )?;

    batch
        .submit_and_wait("vk-resident GDN full-block")
        .context("submit resident GDN full-block CommandBatch")?;
    if let Some(t) = submit_t0 {
        GDNFB_SUBMIT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // --- read back ------------------------------------------------
    let readback_t0 = fb_t0.map(|_| Instant::now());
    let out_bytes = VulkanBuffer::read_back(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &final_out,
    )
    .context("read back GDN full-block final_out")?;
    let out_f32 = bytes_to_f32_vec(&out_bytes);
    let out_tensor = Tensor::from_vec(out_f32, (1usize, 1usize, hidden), x.device())?
        .to_dtype(x.dtype())?;
    if let Some(t) = readback_t0 {
        GDNFB_READBACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Some(t) = fb_t0 {
        GDNFB_TOTAL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        GDNFB_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    Ok(Some(out_tensor))
}

/// configuration; caller falls back to the legacy
/// `gated_deltanet_forward_decode_if` path.
///
/// Persistent state: the recurrent_state and conv_state buffers live
/// on `VulkanBackend` (per-layer, allocated lazily). On first call per
/// layer, they're seeded from the legacy `LinearAttentionState` Tensors
/// so any prefill GDN state is preserved.
#[allow(clippy::too_many_arguments)]
pub fn gated_deltanet_forward_decode_resident_b1(
    backend: &VulkanBackend,
    x_normed: &Tensor,
    weights: &crate::forward::GpuLinearAttentionWeights,
    config: &ModelConfig,
    recurrent_state_t: &Tensor,
    conv_state_t: &Tensor,
) -> Result<Option<Tensor>> {
    let state_key = recurrent_state_t.id();
    // --- supported-config gate -----------------------------------
    let dims = x_normed.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 {
        return Ok(None);
    }
    let hidden = dims[2];
    if hidden != config.hidden_size {
        return Ok(None);
    }
    let Some(vk_device) = backend.vulkan_device() else {
        return Ok(None);
    };

    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;
    let qkv_dim = 2 * qk_dim + v_dim;
    let z_dim = v_dim;
    let a_dim = nv;
    let b_dim = nv;
    let in_proj_total = qkv_dim + z_dim + a_dim + b_dim;
    let conv_kernel = config.linear_conv_kernel_dim;
    let eps = config.rms_norm_eps as f32;

    // --- weight buffer lookups ----------------------------------
    let qkv_w = backend.cached_bf16_packed_weight_buffer(&weights.in_proj_qkv_t)?;
    let z_w = backend.cached_bf16_packed_weight_buffer(&weights.in_proj_z_t)?;
    let a_w = backend.cached_bf16_packed_weight_buffer(&weights.in_proj_a_t)?;
    let b_w = backend.cached_bf16_packed_weight_buffer(&weights.in_proj_b_t)?;
    let out_w = backend.cached_bf16_packed_weight_buffer(&weights.out_proj_t)?;
    let conv_w = backend.cached_f32_weight_buffer(&weights.conv1d)?;
    let q_norm = backend.cached_f32_weight_buffer(&weights.norm)?; // used for gated_rms_norm
    // a_log and dt_bias: these enter the fused recurrent+rmsnorm kernel.
    let a_log = backend.cached_f32_weight_buffer(&weights.a_log)?;
    let dt_bias = backend.cached_f32_weight_buffer(&weights.dt_bias)?;

    // --- persistent state buffers --------------------------------
    let recurrent_bytes = (1 * nv * dk * dv * 4) as u64;
    let recurrent_buf =
        backend.linear_attn_recurrent_state_buffer(state_key, recurrent_bytes)?;
    // conv_state shape: [batch, conv_dim, kernel_size - 1] f32 where
    // conv_dim = qkv_dim (the conv1d operates on the full mixed_qkv).
    let conv_state_bytes = (1 * qkv_dim * (conv_kernel.saturating_sub(1)) * 4) as u64;
    let conv_buf = backend.linear_attn_conv_state_buffer(state_key, conv_state_bytes)?;

    // --- seed state from legacy Tensors on first use -------------
    if !backend.linear_attn_layer_seeded(state_key) {
        seed_recurrent_state(vk_device, &recurrent_buf, recurrent_state_t)?;
        seed_conv_state(vk_device, &conv_buf, conv_state_t)?;
        backend.mark_linear_attn_layer_seeded(state_key);
    }

    // --- acquire pooled intermediate buffers ---------------------
    let x_buf = backend.acquire_resident_scratch("gdn_x", (hidden * 4) as u64)?;
    let in_proj_out =
        backend.acquire_resident_scratch("gdn_in_proj_out", (in_proj_total * 4) as u64)?;
    let mixed_qkv = backend.acquire_resident_scratch("gdn_mixed_qkv", (qkv_dim * 4) as u64)?;
    let conv_qkv = backend.acquire_resident_scratch("gdn_conv_qkv", (qkv_dim * 4) as u64)?;
    let z_buf = backend.acquire_resident_scratch("gdn_z", (z_dim * 4) as u64)?;
    let a_buf = backend.acquire_resident_scratch("gdn_a", (a_dim * 4) as u64)?;
    let b_buf = backend.acquire_resident_scratch("gdn_b", (b_dim * 4) as u64)?;
    let q_buf = backend.acquire_resident_scratch("gdn_q", (qk_dim * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("gdn_k", (qk_dim * 4) as u64)?;
    let v_buf = backend.acquire_resident_scratch("gdn_v", (v_dim * 4) as u64)?;
    let gated_norm = backend.acquire_resident_scratch("gdn_gated_norm", (v_dim * 4) as u64)?;
    let out_buf = backend.acquire_resident_scratch("gdn_out", (hidden * 4) as u64)?;

    let gdn_t0 = if timing_enabled() { Some(Instant::now()) } else { None };

    // --- upload x -----------------------------------------------
    let gdn_upload_t0 = gdn_t0.map(|_| Instant::now());
    let x_f32 = if x_normed.dtype() == DType::F32 {
        x_normed.flatten_all()?
    } else {
        x_normed.to_dtype(DType::F32)?.flatten_all()?
    };
    let x_data: Vec<f32> = x_f32.to_vec1()?;
    VulkanBuffer::upload_data(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &x_buf,
        &f32_slice_to_bytes(&x_data),
    )
    .context("upload x for GDN resident block")?;
    if let Some(t) = gdn_upload_t0 {
        GDN_UPLOAD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let gdn_submit_t0 = gdn_t0.map(|_| Instant::now());

    // --- chain all 9 GDN dispatches into one CommandBatch + one submit ---
    let total_in_proj = qkv_dim + z_dim + a_dim + b_dim;
    let mut batch = CommandBatch::new(vk_device)?;

    // 1) in_proj (bf16w, b=1). push = [hidden, qkv_dim, z_dim, a_dim, b_dim, total_out];
    //    workgroups = total_out.div_ceil(16).
    batch.record_shader(
        shaders::GDN_IN_PROJ_DECODE_BF16W,
        &[
            x_buf.handle(),
            qkv_w.handle(),
            z_w.handle(),
            a_w.handle(),
            b_w.handle(),
            in_proj_out.handle(),
        ],
        &[
            hidden as u32,
            qkv_dim as u32,
            z_dim as u32,
            a_dim as u32,
            b_dim as u32,
            total_in_proj as u32,
        ],
        Workgroups::OneD(total_in_proj.div_ceil(16) as u32),
    )?;
    // 2) split in_proj_out → (mixed_qkv, z, a, b). push = [qkv,z,a,b];
    //    workgroups = total.div_ceil(64).
    batch.record_shader(
        shaders::GDN_IN_PROJ_SPLIT,
        &[
            in_proj_out.handle(),
            mixed_qkv.handle(),
            z_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
        ],
        &[qkv_dim as u32, z_dim as u32, a_dim as u32, b_dim as u32],
        Workgroups::OneD(total_in_proj.div_ceil(64) as u32),
    )?;
    // 3a) causal_conv1d stage 1: output. push = [batch, channels, seq_len, kernel];
    //     workgroups = (batch*channels*seq_len).div_ceil(256).
    let conv_total = 1 * qkv_dim * 1;
    batch.record_shader(
        shaders::CAUSAL_CONV1D,
        &[
            mixed_qkv.handle(),
            conv_w.handle(),
            conv_buf.handle(),
            conv_qkv.handle(),
        ],
        &[1u32, qkv_dim as u32, 1u32, conv_kernel as u32],
        Workgroups::OneD(conv_total.div_ceil(256) as u32),
    )?;
    // 3b) causal_conv1d stage 2: state advance.
    batch.record_shader(
        shaders::CAUSAL_CONV1D_STATE_ADVANCE,
        &[mixed_qkv.handle(), conv_buf.handle()],
        &[1u32, qkv_dim as u32, 1u32, conv_kernel as u32],
        Workgroups::OneD((1 * qkv_dim) as u32),
    )?;
    // 4) split conv_qkv → (q, k, v). push = [qk_dim, v_dim].
    batch.record_shader(
        shaders::GDN_QKV_SPLIT,
        &[
            conv_qkv.handle(),
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
        ],
        &[qk_dim as u32, v_dim as u32],
        Workgroups::OneD((2 * qk_dim + v_dim).div_ceil(64) as u32),
    )?;
    // 5+6) Fused Q-norm + K-norm: one dispatch, 2*nk workgroups.
    // GDN shares the same `q_norm` weight buffer for both Q and K
    // (matches the legacy dispatcher), so we pass it twice.
    batch.record_shader(
        shaders::QWEN_RMSNORM_QK_COMBINED,
        &[
            q_buf.handle(),
            q_norm.handle(),
            k_buf.handle(),
            q_norm.handle(),
        ],
        &[nk as u32, nk as u32, dk as u32, eps.to_bits()],
        Workgroups::OneD((nk + nk) as u32),
    )?;
    // 7) Fused gates+recurrent+rmsnorm. push = [nv, dk, dv, eps_bits, batch],
    //    workgroups = batch*nv.
    batch.record_shader(
        shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
        &[
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
            a_log.handle(),
            dt_bias.handle(),
            recurrent_buf.handle(),
            z_buf.handle(),
            q_norm.handle(),
            gated_norm.handle(),
        ],
        &[
            nv as u32,
            dk as u32,
            dv as u32,
            eps.to_bits(),
            1u32, // batch
        ],
        Workgroups::OneD(nv as u32),
    )?;
    // 8) out_proj (bf16w b=1). push = [hidden_in, out_dim],
    //    workgroups = out_dim.div_ceil(16).
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[gated_norm.handle(), out_w.handle(), out_buf.handle()],
        &[v_dim as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;

    batch
        .submit_and_wait("vk-resident GDN block")
        .context("submit resident GDN CommandBatch")?;
    if let Some(t) = gdn_submit_t0 {
        GDN_SUBMIT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // --- read back result ---------------------------------------
    let gdn_readback_t0 = gdn_t0.map(|_| Instant::now());
    let out_bytes = VulkanBuffer::read_back(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &out_buf,
    )
    .context("read back GDN out_buf")?;
    let out_f32 = bytes_to_f32_vec(&out_bytes);
    let out_tensor = Tensor::from_vec(out_f32, (1usize, 1usize, hidden), x_normed.device())?
        .to_dtype(x_normed.dtype())?;
    if let Some(t) = gdn_readback_t0 {
        GDN_READBACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Some(t) = gdn_t0 {
        GDN_TOTAL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        GDN_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    Ok(Some(out_tensor))
}

fn seed_recurrent_state(
    vk_device: &VulkanDevice,
    buf: &VulkanBuffer,
    state_t: &Tensor,
) -> Result<()> {
    let flat = if state_t.dtype() == DType::F32 {
        state_t.flatten_all()?
    } else {
        state_t.to_dtype(DType::F32)?.flatten_all()?
    };
    let data: Vec<f32> = flat.to_vec1()?;
    let bytes = f32_slice_to_bytes(&data);
    VulkanBuffer::upload_data(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        buf,
        &bytes,
    )
    .context("seed recurrent state")
}

fn seed_conv_state(vk_device: &VulkanDevice, buf: &VulkanBuffer, state_t: &Tensor) -> Result<()> {
    let flat = if state_t.dtype() == DType::F32 {
        state_t.flatten_all()?
    } else {
        state_t.to_dtype(DType::F32)?.flatten_all()?
    };
    let data: Vec<f32> = flat.to_vec1()?;
    let bytes = f32_slice_to_bytes(&data);
    // The legacy conv_state may be smaller than the allocated buffer (we
    // sized off qkv_dim × (kernel_size - 1), the candle Tensor matches).
    VulkanBuffer::upload_data(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        buf,
        &bytes,
    )
    .context("seed conv state")
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

// =====================================================================
// Cross-layer chained record helpers + native orchestrator.
//
// The per-layer block helpers above each create a `CommandBatch`,
// submit, wait, and Tensor-bridge their input and output. That's one
// submit + one upload + one readback PER LAYER → 32 submits and 32
// CPU round-trips per decode token. The microbench
// `full_token_resident_paged` shows 32 layers in ONE CommandBatch +
// one submit can run at 29 tok/s on the same shapes, so most of the
// per-layer wallclock on the real model is just orchestration tax.
//
// The functions below break the block bodies out into "record-only"
// variants that append their dispatches to a caller-provided
// `CommandBatch`, reading from `x_in_buf` and writing the final
// post-residual into `x_out_buf` (caller alternates the pair across
// layers). The native orchestrator pre-uploads x once, pre-seeds any
// KV/conv/recurrent state, records all 32 layers' dispatches into a
// single batch, submits once, reads back the final hidden, and runs
// the final RMSNorm + LM head via the legacy path (cheap, one shot).
// =====================================================================

/// Record one full-attention transformer block's dispatches into
/// `batch`. Reads from `x_in_buf` and writes the post-MLP residual
/// into `x_out_buf`. Caller must ensure the two buffers are distinct
/// and pre-sized to `hidden * 4` bytes; the residual ADDs and
/// pre-norm read both buffers so aliasing is unsafe.
///
/// Caller is also responsible for pre-batch work:
/// - Seeding `vk_kv_cache` for this layer (one-time per session)
/// - Pre-uploading `rope_cos_buf`, `rope_sin_buf`, `block_table_buf`,
///   `seq_lens_buf`
///
/// Returns `Ok(false)` on any unsupported configuration so the caller
/// can fall back; returns `Ok(true)` on successful recording.
#[allow(clippy::too_many_arguments)]
pub fn record_full_attn_block_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    x_in_buf: &VulkanBuffer,
    x_out_buf: &VulkanBuffer,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    start_pos: usize,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    paged_cache: &PagedKvCache,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
) -> Result<bool> {
    let attn = match &layer.attention {
        crate::forward::GpuAttentionWeights::Full(w) => w,
        _ => return Ok(false),
    };
    if !config.attn_output_gate {
        return Ok(false);
    }
    let hidden = config.hidden_size;
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

    let block_size = paged_cache.block_size();
    let slot = block_table
        .slot_for(start_pos, block_size)
        .ok_or_else(|| anyhow::anyhow!("no slot for start_pos {start_pos}"))?;
    let max_blocks_per_seq = block_table.blocks.len();

    // --- weight buffer lookups (cached on backend) -------------------
    let q_w = backend.cached_bf16_packed_weight_buffer(&attn.q_proj_t)?;
    let k_w = backend.cached_bf16_packed_weight_buffer(&attn.k_proj_t)?;
    let v_w = backend.cached_bf16_packed_weight_buffer(&attn.v_proj_t)?;
    let o_w = backend.cached_bf16_packed_weight_buffer(&attn.o_proj_t)?;
    let gate_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.gate_proj_t)?;
    let up_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.up_proj_t)?;
    let down_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.down_proj_t)?;
    let in_norm = backend.cached_f32_weight_buffer(&layer.input_layernorm)?;
    let post_norm = backend.cached_f32_weight_buffer(&layer.post_attention_layernorm)?;
    let q_norm = backend.cached_f32_weight_buffer(&attn.q_norm)?;
    let k_norm = backend.cached_f32_weight_buffer(&attn.k_norm)?;

    // --- per-layer scratch buffers (pooled, persistent) --------------
    // These can be SHARED across all full-attn layers within one batch
    // because each dispatch reads only its predecessor's output, with
    // a compute→compute barrier between, and dispatches are recorded
    // and executed in strict program order.
    let normed = backend.acquire_resident_scratch("nfa_normed", (hidden * 4) as u64)?;
    let qkv_combined = backend.acquire_resident_scratch(
        "nfa_qkv_combined",
        ((q_dim + k_dim + v_dim) * 4) as u64,
    )?;
    let q_buf =
        backend.acquire_resident_scratch("nfa_q", (num_heads * head_dim * 4) as u64)?;
    let gate_buf =
        backend.acquire_resident_scratch("nfa_gate", (num_heads * head_dim * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("nfa_k", (k_dim * 4) as u64)?;
    let v_buf = backend.acquire_resident_scratch("nfa_v", (v_dim * 4) as u64)?;
    let q_rot =
        backend.acquire_resident_scratch("nfa_q_rot", (num_heads * head_dim * 4) as u64)?;
    let k_rot = backend.acquire_resident_scratch("nfa_k_rot", (k_dim * 4) as u64)?;
    let attn_pre_gate = backend.acquire_resident_scratch(
        "nfa_attn_pre_gate",
        (num_heads * head_dim * 4) as u64,
    )?;
    let attn_post_gate = backend.acquire_resident_scratch(
        "nfa_attn_post_gate",
        (num_heads * head_dim * 4) as u64,
    )?;
    let attn_out = backend.acquire_resident_scratch("nfa_attn_out", (hidden * 4) as u64)?;
    let attn_residual =
        backend.acquire_resident_scratch("nfa_attn_residual", (hidden * 4) as u64)?;
    let normed_post = backend.acquire_resident_scratch("nfa_normed_post", (hidden * 4) as u64)?;
    let mlp_scratch =
        backend.acquire_resident_scratch("nfa_mlp_scratch", (intermediate * 4) as u64)?;
    let mlp_out = backend.acquire_resident_scratch("nfa_mlp_out", (hidden * 4) as u64)?;

    let k_pool = vk_kv_cache
        .k_buffer(full_attn_layer_idx)
        .ok_or_else(|| anyhow::anyhow!("VkPagedKvCache missing layer {full_attn_layer_idx}"))?;
    let v_pool = vk_kv_cache
        .v_buffer(full_attn_layer_idx)
        .ok_or_else(|| anyhow::anyhow!("VkPagedKvCache missing layer {full_attn_layer_idx}"))?;
    let elements_per_slot = num_kv_heads * head_dim;
    let total_qkv_out = q_dim + k_dim + v_dim;
    let q_h_d = num_heads * head_dim;
    let total_split = num_heads * head_dim + 2 * (num_kv_heads * head_dim);

    // --- record dispatches into the shared batch --------------------
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[x_in_buf.handle(), in_norm.handle(), normed.handle()],
        &[1u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(1),
    )?;
    batch.record_shader(
        shaders::FULL_ATTN_QKV_DECODE_BF16W,
        &[
            normed.handle(),
            q_w.handle(),
            k_w.handle(),
            v_w.handle(),
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
    // Fused Q-norm + K-norm: one dispatch, q_rows + k_rows workgroups.
    batch.record_shader(
        shaders::QWEN_RMSNORM_QK_COMBINED,
        &[
            q_buf.handle(),
            q_norm.handle(),
            k_buf.handle(),
            k_norm.handle(),
        ],
        &[
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            eps.to_bits(),
        ],
        Workgroups::OneD((num_heads + num_kv_heads) as u32),
    )?;
    batch.record_shader(
        shaders::VK_ROPE_F32,
        &[
            q_buf.handle(),
            rope_cos_buf.handle(),
            rope_sin_buf.handle(),
            q_rot.handle(),
        ],
        &[1u32, num_heads as u32, head_dim as u32, rotary_dim as u32],
        Workgroups::OneD((num_heads * head_dim).div_ceil(256) as u32),
    )?;
    batch.record_shader(
        shaders::VK_ROPE_F32,
        &[
            k_buf.handle(),
            rope_cos_buf.handle(),
            rope_sin_buf.handle(),
            k_rot.handle(),
        ],
        &[1u32, num_kv_heads as u32, head_dim as u32, rotary_dim as u32],
        Workgroups::OneD((num_kv_heads * head_dim).div_ceil(256) as u32),
    )?;
    batch.record_shader(
        shaders::PAGED_KV_WRITE_SLOT,
        &[
            k_rot.handle(),
            v_buf.handle(),
            k_pool.handle(),
            v_pool.handle(),
        ],
        &[slot as u32, elements_per_slot as u32],
        Workgroups::OneD(elements_per_slot.div_ceil(64) as u32),
    )?;
    // Split-K paged attention: spread each (batch, q_head) pair's K/V
    // scan across `num_chunks` workgroups so we use more SMs.
    // Combined via a reduce pass that performs the online-softmax
    // recurrence. Default 8 chunks (16 heads × 8 = 128 workgroups,
    // ≈90% of the RTX 6000 Ada's 144 SMs) — tunable via
    // `KILN_VK_PAGED_ATTN_SPLITK_CHUNKS`. Anything ≥ seq_len degrades
    // gracefully (chunks beyond `seq_len` write neutral identities).
    let num_chunks: usize = std::env::var("KILN_VK_PAGED_ATTN_SPLITK_CHUNKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(8);
    let partials_stride = 2 + head_dim;
    let partials_bytes = (1 * num_heads * num_chunks * partials_stride * 4) as u64;
    let attn_partials =
        backend.acquire_resident_scratch("nfa_attn_partials", partials_bytes)?;
    batch.record_shader(
        shaders::PAGED_ATTN_DECODE_BATCH_PAGED_SPLITK,
        &[
            q_rot.handle(),
            k_pool.handle(),
            v_pool.handle(),
            block_table_buf.handle(),
            seq_lens_buf.handle(),
            attn_partials.handle(),
        ],
        &[
            max_blocks_per_seq as u32,
            block_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            softmax_scale.to_bits(),
            num_chunks as u32,
        ],
        Workgroups::OneD((num_heads * num_chunks) as u32),
    )?;
    batch.record_shader(
        shaders::PAGED_ATTN_DECODE_BATCH_PAGED_SPLITK_REDUCE,
        &[
            attn_partials.handle(),
            attn_pre_gate.handle(),
        ],
        &[num_heads as u32, head_dim as u32, num_chunks as u32],
        Workgroups::OneD(num_heads as u32),
    )?;
    batch.record_shader(
        shaders::VK_MUL_SIGMOID_GATE_F32,
        &[
            attn_pre_gate.handle(),
            gate_buf.handle(),
            attn_post_gate.handle(),
        ],
        &[q_h_d as u32],
        Workgroups::OneD(q_h_d.div_ceil(256) as u32),
    )?;
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[
            attn_post_gate.handle(),
            o_w.handle(),
            attn_out.handle(),
        ],
        &[q_h_d as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    batch.record_shader(
        shaders::ADD_QWEN_RMSNORM,
        &[
            x_in_buf.handle(),
            attn_out.handle(),
            post_norm.handle(),
            attn_residual.handle(),
            normed_post.handle(),
        ],
        &[hidden as u32, eps.to_bits()],
        Workgroups::OneD(1),
    )?;
    batch.record_shader(
        shaders::MLP_GATE_UP_DECODE_BF16W,
        &[
            normed_post.handle(),
            gate_w.handle(),
            up_w.handle(),
            mlp_scratch.handle(),
        ],
        &[hidden as u32, intermediate as u32],
        Workgroups::OneD(intermediate.div_ceil(64) as u32),
    )?;
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[mlp_scratch.handle(), down_w.handle(), mlp_out.handle()],
        &[intermediate as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    batch.record_shader(
        shaders::ADD,
        &[
            attn_residual.handle(),
            mlp_out.handle(),
            x_out_buf.handle(),
        ],
        &[hidden as u32],
        Workgroups::OneD(hidden.div_ceil(256) as u32),
    )?;
    Ok(true)
}

/// Record one GDN transformer block's dispatches into `batch`.
/// Same semantics as [`record_full_attn_block_into`] but for GDN
/// (linear-attention) layers, with the inner GDN compute in place of
/// full attention.
///
/// Caller is responsible for pre-batch GDN state seeding (recurrent +
/// conv) — the relevant buffers are sourced via
/// `backend.linear_attn_recurrent_state_buffer / _conv_state_buffer`
/// and marked seeded on first use.
#[allow(clippy::too_many_arguments)]
pub fn record_gdn_block_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    x_in_buf: &VulkanBuffer,
    x_out_buf: &VulkanBuffer,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    recurrent_state_t: &Tensor,
    conv_state_t: &Tensor,
) -> Result<bool> {
    let lin_weights = match &layer.attention {
        crate::forward::GpuAttentionWeights::Linear(w) => w,
        _ => return Ok(false),
    };
    let hidden = config.hidden_size;
    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;
    let qkv_dim = 2 * qk_dim + v_dim;
    let z_dim = v_dim;
    let a_dim = nv;
    let b_dim = nv;
    let in_proj_total = qkv_dim + z_dim + a_dim + b_dim;
    let conv_kernel = config.linear_conv_kernel_dim;
    let intermediate = config.intermediate_size;
    let eps = config.rms_norm_eps as f32;
    let state_key = recurrent_state_t.id();

    // Weight lookups
    let qkv_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.in_proj_qkv_t)?;
    let z_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.in_proj_z_t)?;
    let a_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.in_proj_a_t)?;
    let b_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.in_proj_b_t)?;
    let out_w = backend.cached_bf16_packed_weight_buffer(&lin_weights.out_proj_t)?;
    let conv_w = backend.cached_f32_weight_buffer(&lin_weights.conv1d)?;
    let qk_norm = backend.cached_f32_weight_buffer(&lin_weights.norm)?;
    let a_log = backend.cached_f32_weight_buffer(&lin_weights.a_log)?;
    let dt_bias = backend.cached_f32_weight_buffer(&lin_weights.dt_bias)?;
    let gate_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.gate_proj_t)?;
    let up_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.up_proj_t)?;
    let down_w = backend.cached_bf16_packed_weight_buffer(&layer.mlp.down_proj_t)?;
    let in_norm = backend.cached_f32_weight_buffer(&layer.input_layernorm)?;
    let post_norm = backend.cached_f32_weight_buffer(&layer.post_attention_layernorm)?;

    // Persistent state (per-state-key on backend)
    let recurrent_bytes = (1 * nv * dk * dv * 4) as u64;
    let recurrent_buf =
        backend.linear_attn_recurrent_state_buffer(state_key, recurrent_bytes)?;
    let conv_state_bytes = (1 * qkv_dim * (conv_kernel.saturating_sub(1)) * 4) as u64;
    let conv_buf = backend.linear_attn_conv_state_buffer(state_key, conv_state_bytes)?;

    // Seed states on first call per layer per session (must happen
    // BEFORE batch records reads from these buffers).
    if !backend.linear_attn_layer_seeded(state_key) {
        let Some(vk_device) = backend.vulkan_device() else {
            return Ok(false);
        };
        seed_recurrent_state(vk_device, &recurrent_buf, recurrent_state_t)?;
        seed_conv_state(vk_device, &conv_buf, conv_state_t)?;
        backend.mark_linear_attn_layer_seeded(state_key);
    }

    // Shared scratch (pooled, single set across all GDN layers in a batch).
    let normed_pre = backend.acquire_resident_scratch("ngd_normed_pre", (hidden * 4) as u64)?;
    let in_proj_out =
        backend.acquire_resident_scratch("ngd_in_proj_out", (in_proj_total * 4) as u64)?;
    let mixed_qkv = backend.acquire_resident_scratch("ngd_mixed_qkv", (qkv_dim * 4) as u64)?;
    let conv_qkv = backend.acquire_resident_scratch("ngd_conv_qkv", (qkv_dim * 4) as u64)?;
    let z_buf = backend.acquire_resident_scratch("ngd_z", (z_dim * 4) as u64)?;
    let a_buf = backend.acquire_resident_scratch("ngd_a", (a_dim * 4) as u64)?;
    let b_buf = backend.acquire_resident_scratch("ngd_b", (b_dim * 4) as u64)?;
    let q_buf = backend.acquire_resident_scratch("ngd_q", (qk_dim * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("ngd_k", (qk_dim * 4) as u64)?;
    let v_buf = backend.acquire_resident_scratch("ngd_v", (v_dim * 4) as u64)?;
    let gated_norm = backend.acquire_resident_scratch("ngd_gated_norm", (v_dim * 4) as u64)?;
    let gdn_out = backend.acquire_resident_scratch("ngd_gdn_out", (hidden * 4) as u64)?;
    let attn_residual = backend.acquire_resident_scratch("ngd_attn_residual", (hidden * 4) as u64)?;
    let normed_post = backend.acquire_resident_scratch("ngd_normed_post", (hidden * 4) as u64)?;
    let mlp_scratch = backend.acquire_resident_scratch("ngd_mlp_scratch", (intermediate * 4) as u64)?;
    let mlp_out = backend.acquire_resident_scratch("ngd_mlp_out", (hidden * 4) as u64)?;

    // Dispatch sequence (mirrors transformer_block_paged_decode_gdn_resident_b1 body)
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[x_in_buf.handle(), in_norm.handle(), normed_pre.handle()],
        &[1u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(1),
    )?;
    batch.record_shader(
        shaders::GDN_IN_PROJ_DECODE_BF16W,
        &[
            normed_pre.handle(),
            qkv_w.handle(),
            z_w.handle(),
            a_w.handle(),
            b_w.handle(),
            in_proj_out.handle(),
        ],
        &[
            hidden as u32,
            qkv_dim as u32,
            z_dim as u32,
            a_dim as u32,
            b_dim as u32,
            in_proj_total as u32,
        ],
        Workgroups::OneD(in_proj_total.div_ceil(16) as u32),
    )?;
    batch.record_shader(
        shaders::GDN_IN_PROJ_SPLIT,
        &[
            in_proj_out.handle(),
            mixed_qkv.handle(),
            z_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
        ],
        &[qkv_dim as u32, z_dim as u32, a_dim as u32, b_dim as u32],
        Workgroups::OneD(in_proj_total.div_ceil(64) as u32),
    )?;
    let conv_total = 1 * qkv_dim * 1;
    batch.record_shader(
        shaders::CAUSAL_CONV1D,
        &[
            mixed_qkv.handle(),
            conv_w.handle(),
            conv_buf.handle(),
            conv_qkv.handle(),
        ],
        &[1u32, qkv_dim as u32, 1u32, conv_kernel as u32],
        Workgroups::OneD(conv_total.div_ceil(256) as u32),
    )?;
    batch.record_shader(
        shaders::CAUSAL_CONV1D_STATE_ADVANCE,
        &[mixed_qkv.handle(), conv_buf.handle()],
        &[1u32, qkv_dim as u32, 1u32, conv_kernel as u32],
        Workgroups::OneD((1 * qkv_dim) as u32),
    )?;
    batch.record_shader(
        shaders::GDN_QKV_SPLIT,
        &[
            conv_qkv.handle(),
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
        ],
        &[qk_dim as u32, v_dim as u32],
        Workgroups::OneD((2 * qk_dim + v_dim).div_ceil(64) as u32),
    )?;
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[q_buf.handle(), qk_norm.handle(), q_buf.handle()],
        &[nk as u32, dk as u32, eps.to_bits()],
        Workgroups::OneD(nk as u32),
    )?;
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[k_buf.handle(), qk_norm.handle(), k_buf.handle()],
        &[nk as u32, dk as u32, eps.to_bits()],
        Workgroups::OneD(nk as u32),
    )?;
    batch.record_shader(
        shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
        &[
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
            a_log.handle(),
            dt_bias.handle(),
            recurrent_buf.handle(),
            z_buf.handle(),
            qk_norm.handle(),
            gated_norm.handle(),
        ],
        &[nv as u32, dk as u32, dv as u32, eps.to_bits(), 1u32],
        Workgroups::OneD(nv as u32),
    )?;
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[gated_norm.handle(), out_w.handle(), gdn_out.handle()],
        &[v_dim as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    batch.record_shader(
        shaders::ADD,
        &[x_in_buf.handle(), gdn_out.handle(), attn_residual.handle()],
        &[hidden as u32],
        Workgroups::OneD(hidden.div_ceil(256) as u32),
    )?;
    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[
            attn_residual.handle(),
            post_norm.handle(),
            normed_post.handle(),
        ],
        &[1u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(1),
    )?;
    batch.record_shader(
        shaders::MLP_GATE_UP_DECODE_BF16W,
        &[
            normed_post.handle(),
            gate_w.handle(),
            up_w.handle(),
            mlp_scratch.handle(),
        ],
        &[hidden as u32, intermediate as u32],
        Workgroups::OneD(intermediate.div_ceil(64) as u32),
    )?;
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W,
        &[mlp_scratch.handle(), down_w.handle(), mlp_out.handle()],
        &[intermediate as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    batch.record_shader(
        shaders::ADD,
        &[
            attn_residual.handle(),
            mlp_out.handle(),
            x_out_buf.handle(),
        ],
        &[hidden as u32],
        Workgroups::OneD(hidden.div_ceil(256) as u32),
    )?;
    Ok(true)
}
