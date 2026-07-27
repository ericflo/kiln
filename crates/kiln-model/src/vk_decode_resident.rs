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
//! fall back transparently to the nonresident
//! `transformer_block_paged_with_rope_tables`. The lifted overhead
//! is the per-kernel `extract + upload + readback` boundary that
//! dominates the nonresident Vulkan decode path at 1.04 tok/s on
//! Qwen3.5-4B — the resident microbench measures the same kernel
//! sequence at 29 tok/s when chained through a `CommandBatch`.

#![cfg(feature = "vulkan")]

use anyhow::{Context, Result};

use kiln_core::block::{BlockTable, unique_physical_blocks};
use kiln_core::config::ModelConfig;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use kiln_graph::{
    CaptureError, InvalidateReason, ReplayInputs, ReplayKey, ReplayOutputs, ReplayPlan,
    ReplayResourceStability, ReplayState, ResidentResourceRef,
};
use kiln_tensor::{Backend, DType};
use kiln_vulkan_kernel::kernels::{
    paged_attn_decode_splitk_chunks as paged_attn_splitk_chunks, vulkan_kernel_policy,
};
use kiln_vulkan_kernel::shaders;
use kiln_vulkan_kernel::{CommandBatch, VkPagedKvCache, VulkanBuffer, VulkanDevice, Workgroups};

use crate::PagedKvCacheKt;
use crate::backend::vulkan::VulkanBackend;
use crate::forward::GpuLayerWeights;

// (#1082) The previous process-global bridge cache is gone. It existed to give
// shared upload helpers a stable `TensorId` per weight, but memoized a full
// duplicate of every projection/norm weight (~9 GB for Qwen3.5-4B, including
// the 778 MB lm_head) on top of the kt weights and vk buffers: triple residency
// that pushed the unified-memory APU into OOM. The resident-decode upload sites
// now call the kt-native `backend.cached_*_weight_buffer_kt(&kt_weight)` helpers
// directly, which key the vk-buffer cache on the stable kt `TensorId` and extract
// bytes straight from kt storage -- upload-once, no duplicate weight copy.

// Per-block timing accumulators are disabled by the qualified policy. They stay
// available as a typed research hook without adding a production environment
// read to every Vulkan process.
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
    vulkan_kernel_policy().profile_resident_decode_timing
}

struct VulkanCommandBatchReplayPlan<'a> {
    batch: Option<CommandBatch<'a>>,
    label: &'static str,
    state: ReplayState,
    replay_count: u64,
}

impl std::fmt::Debug for VulkanCommandBatchReplayPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanCommandBatchReplayPlan")
            .field("label", &self.label)
            .field("key", &self.state.key)
            .field("replay_count", &self.replay_count)
            .finish_non_exhaustive()
    }
}

// SAFETY: this adapter is created and consumed stack-locally by
// `replay_vulkan_command_batch`; the native `CommandBatch` is single-use and
// taken during `ReplayPlan::replay`. We keep the unsafe bound here instead of
// broadening `CommandBatch` because the shared contract requires ReplayPlan
// implementors to be Send + Sync.
unsafe impl Send for VulkanCommandBatchReplayPlan<'_> {}
unsafe impl Sync for VulkanCommandBatchReplayPlan<'_> {}

impl ReplayPlan for VulkanCommandBatchReplayPlan<'_> {
    fn backend(&self) -> Backend {
        Backend::Vulkan
    }

    fn key(&self) -> ReplayKey {
        self.state.key.clone()
    }

    fn validate_inputs(&self, inputs: ReplayInputs<'_>) -> std::result::Result<(), CaptureError> {
        self.state.validate(inputs.key, inputs.resources)
    }

    fn replay(
        &mut self,
        inputs: ReplayInputs<'_>,
    ) -> std::result::Result<ReplayOutputs, CaptureError> {
        self.validate_inputs(inputs)?;
        let batch = self.batch.take().ok_or_else(|| {
            CaptureError::Backend(format!(
                "Vulkan CommandBatch {} was already replayed",
                self.label
            ))
        })?;
        batch
            .submit_and_wait(self.label)
            .map_err(|e| CaptureError::Backend(format!("Vulkan CommandBatch replay: {e}")))?;
        self.replay_count += 1;
        Ok(ReplayOutputs::new(
            inputs.resources.to_vec(),
            self.replay_count,
        ))
    }

    fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason> {
        self.state.invalidate_reason(&state.key, &state.inputs)
    }
}

fn replay_vulkan_command_batch(
    batch: CommandBatch<'_>,
    label: &'static str,
    shape_key: Vec<usize>,
    max_batch: usize,
    resources: Vec<ResidentResourceRef>,
) -> Result<()> {
    let replay_key = ReplayKey::new(
        Backend::Vulkan,
        label,
        shape_key,
        Some(DType::F32),
        max_batch,
        true,
    );
    let mut plan = VulkanCommandBatchReplayPlan {
        batch: Some(batch),
        label,
        state: ReplayState::new(replay_key.clone(), resources),
        replay_count: 0,
    };
    let replay_resources = plan.state.inputs.clone();
    let replay_inputs = ReplayInputs::new(&replay_key, &replay_resources);
    kiln_graph::ReplayPlan::replay(&mut plan, replay_inputs)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn vk_replay_resource(
    buffer: &VulkanBuffer,
    dtype: DType,
    shape: Vec<usize>,
) -> ResidentResourceRef {
    let strides = contiguous_strides(&shape);
    ResidentResourceRef {
        tensor_id: None,
        backend: Backend::Vulkan,
        dtype,
        shape,
        strides,
        start_offset: 0,
        contiguous: true,
        byte_len: buffer.size() as usize,
        replay_stability: ReplayResourceStability::StableWithinStep,
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride.saturating_mul(*dim);
    }
    strides
}

/// Record the two-dispatch causal-convolution update as one ordered operation.
///
/// The output dispatch reads `state`; the state-advance dispatch then writes
/// that same buffer. The second dispatch must retain the default inter-dispatch
/// barrier so its write cannot race the preceding read.
#[allow(clippy::too_many_arguments)]
fn record_causal_conv1d_update_into(
    batch: &mut CommandBatch<'_>,
    x: &VulkanBuffer,
    weight: &VulkanBuffer,
    state: &VulkanBuffer,
    out: &VulkanBuffer,
    batch_size: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<()> {
    anyhow::ensure!(
        kernel_size == 4,
        "resident causal conv1d: only kernel_size=4 supported"
    );
    let state_total = batch_size
        .checked_mul(channels)
        .context("resident causal conv1d: element count overflow")?;
    let conv_total = state_total
        .checked_mul(seq_len)
        .context("resident causal conv1d: element count overflow")?;
    anyhow::ensure!(conv_total > 0, "resident causal conv1d: empty dispatch");

    batch.record_shader(
        shaders::CAUSAL_CONV1D,
        &[x.handle(), weight.handle(), state.handle(), out.handle()],
        &[
            batch_size as u32,
            channels as u32,
            seq_len as u32,
            kernel_size as u32,
        ],
        Workgroups::OneD(conv_total.div_ceil(256) as u32),
    )?;
    batch.record_shader(
        shaders::CAUSAL_CONV1D_STATE_ADVANCE,
        &[x.handle(), state.handle()],
        &[
            batch_size as u32,
            channels as u32,
            seq_len as u32,
            kernel_size as u32,
        ],
        Workgroups::OneD(state_total.div_ceil(256) as u32),
    )
}

#[allow(clippy::too_many_arguments)]
fn batched_decode_replay_resources(
    x_in_buf: &VulkanBuffer,
    x_scratch_buf: &VulkanBuffer,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
    out_buf: &VulkanBuffer,
    out_dtype: DType,
    out_shape: Vec<usize>,
    batch_size: usize,
    hidden: usize,
    rotary_dim: usize,
    max_blocks_per_seq: usize,
) -> Vec<ResidentResourceRef> {
    vec![
        vk_replay_resource(x_in_buf, DType::F32, vec![batch_size, hidden]),
        vk_replay_resource(x_scratch_buf, DType::F32, vec![batch_size, hidden]),
        vk_replay_resource(rope_cos_buf, DType::F32, vec![batch_size, rotary_dim]),
        vk_replay_resource(rope_sin_buf, DType::F32, vec![batch_size, rotary_dim]),
        vk_replay_resource(
            block_table_buf,
            DType::U32,
            vec![batch_size, max_blocks_per_seq],
        ),
        vk_replay_resource(seq_lens_buf, DType::U32, vec![batch_size]),
        vk_replay_resource(slots_buf, DType::U32, vec![batch_size]),
        vk_replay_resource(out_buf, out_dtype, out_shape),
    ]
}

fn append_sample_replay_resources(
    resources: &mut Vec<ResidentResourceRef>,
    sample: &BatchedResidentSampleBuffers,
) {
    resources.extend([
        vk_replay_resource(&sample.top_k, DType::U32, vec![sample.batch_size]),
        vk_replay_resource(&sample.temperatures, DType::F32, vec![sample.batch_size]),
        vk_replay_resource(&sample.top_p, DType::F32, vec![sample.batch_size]),
        vk_replay_resource(&sample.min_p, DType::F32, vec![sample.batch_size]),
        vk_replay_resource(&sample.seed_lo, DType::U32, vec![sample.batch_size]),
        vk_replay_resource(&sample.seed_hi, DType::U32, vec![sample.batch_size]),
    ]);
    if sample.history_items > 0 {
        if let Some(buffer) = sample.history_rows.as_ref() {
            resources.push(vk_replay_resource(
                buffer,
                DType::U32,
                vec![sample.history_items],
            ));
        }
        if let Some(buffer) = sample.history_indices.as_ref() {
            resources.push(vk_replay_resource(
                buffer,
                DType::U32,
                vec![sample.history_items],
            ));
        }
        if let Some(buffer) = sample.history_counts.as_ref() {
            resources.push(vk_replay_resource(
                buffer,
                DType::U32,
                vec![sample.history_items],
            ));
        }
        if let Some(buffer) = sample.repetitions.as_ref() {
            resources.push(vk_replay_resource(
                buffer,
                DType::F32,
                vec![sample.batch_size],
            ));
        }
        if let Some(buffer) = sample.presences.as_ref() {
            resources.push(vk_replay_resource(
                buffer,
                DType::F32,
                vec![sample.batch_size],
            ));
        }
        if let Some(buffer) = sample.frequencies.as_ref() {
            resources.push(vk_replay_resource(
                buffer,
                DType::F32,
                vec![sample.batch_size],
            ));
        }
    }
}

fn linear_bf16w_rows4_enabled() -> bool {
    vulkan_kernel_policy().linear_decode_bf16w_rows4
}

fn linear_bf16w_rows8_enabled() -> bool {
    vulkan_kernel_policy().linear_decode_bf16w_rows8
}

fn mlp_bf16_rows8_min_batch() -> usize {
    vulkan_kernel_policy().mlp_bf16_rows8_min_batch
}

fn mlp_bf16_gate_up_rows4_min_batch() -> usize {
    vulkan_kernel_policy().mlp_bf16_gate_up_rows4_min_batch
}

fn mlp_bf16_down_rows4_min_batch() -> usize {
    vulkan_kernel_policy().mlp_bf16_down_rows4_min_batch
}

fn linear_bf16_rows8_min_batch() -> usize {
    vulkan_kernel_policy().linear_bf16_rows8_min_batch
}

fn linear_bf16_rows4_min_batch() -> usize {
    vulkan_kernel_policy().linear_bf16_rows4_min_batch
}

fn gdn_in_proj_rows4_min_batch() -> usize {
    vulkan_kernel_policy().gdn_in_proj_rows4_min_batch
}

fn gdn_in_proj_rows8_min_batch() -> usize {
    vulkan_kernel_policy().gdn_in_proj_rows8_min_batch
}

fn full_attn_qkv_bf16_rows8_min_batch() -> usize {
    vulkan_kernel_policy().full_attn_qkv_bf16_rows8_min_batch
}

fn full_attn_qkv_bf16_rows4_min_batch() -> usize {
    vulkan_kernel_policy().full_attn_qkv_bf16_rows4_min_batch
}

fn full_attn_qkv_gate_split_bf16w_plan(batch: usize, total_out: usize) -> (&'static str, u32) {
    let rows8 = batch >= full_attn_qkv_bf16_rows8_min_batch()
        && vulkan_kernel_policy().full_attn_qkv_bf16w_rows8;
    let rows4 = batch >= full_attn_qkv_bf16_rows4_min_batch()
        && !rows8
        && vulkan_kernel_policy().full_attn_qkv_bf16w_rows4;
    let row_groups = if rows8 {
        batch.div_ceil(8)
    } else if rows4 {
        batch.div_ceil(4)
    } else {
        batch
    };
    let shader = if rows8 {
        shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS8_BF16W
    } else if rows4 {
        shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS4_BF16W
    } else {
        shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_BF16W
    };
    (shader, (row_groups * total_out.div_ceil(16)) as u32)
}

fn linear_bf16w_batched_plan(batch: usize, out_dim: usize) -> (&'static str, u32) {
    let rows8 = batch >= linear_bf16_rows8_min_batch() && linear_bf16w_rows8_enabled();
    let rows4 = batch >= linear_bf16_rows4_min_batch() && !rows8 && linear_bf16w_rows4_enabled();
    let row_groups = if rows8 {
        batch.div_ceil(8)
    } else if rows4 {
        batch.div_ceil(4)
    } else {
        batch
    };
    let shader = if rows8 {
        shaders::LINEAR_DECODE_BATCHED_ROWS8_BF16W
    } else if rows4 {
        shaders::LINEAR_DECODE_BATCHED_ROWS4_BF16W
    } else {
        shaders::LINEAR_DECODE_BATCHED_BF16W
    };
    (shader, (row_groups * out_dim.div_ceil(32)) as u32)
}

fn mlp_gate_up_bf16w_batched_plan(batch: usize, intermediate: usize) -> (&'static str, u32) {
    let rows8 = batch >= mlp_bf16_rows8_min_batch() && vulkan_kernel_policy().mlp_bf16_rows8;
    let rows4 = batch >= mlp_bf16_gate_up_rows4_min_batch()
        && !rows8
        && vulkan_kernel_policy().mlp_bf16_gate_up_rows4;
    if rows8 {
        (
            shaders::MLP_GATE_UP_DECODE_BATCHED_ROWS8_BF16W,
            (batch.div_ceil(8) * intermediate.div_ceil(64)) as u32,
        )
    } else if rows4 {
        (
            shaders::MLP_GATE_UP_DECODE_BATCHED_ROWS4_BF16W,
            (batch.div_ceil(4) * intermediate.div_ceil(64)) as u32,
        )
    } else {
        (
            shaders::MLP_GATE_UP_DECODE_BATCHED_BF16W,
            (batch * intermediate.div_ceil(128)) as u32,
        )
    }
}

fn mlp_down_add_residual_bf16w_batched_plan(batch: usize, out_dim: usize) -> (&'static str, u32) {
    let rows8 = batch >= mlp_bf16_rows8_min_batch() && vulkan_kernel_policy().mlp_bf16_rows8;
    let rows4 = batch >= mlp_bf16_down_rows4_min_batch()
        && !rows8
        && vulkan_kernel_policy().mlp_bf16_down_rows4;
    if rows8 {
        (
            shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL_ROWS8,
            (batch.div_ceil(8) * out_dim.div_ceil(32)) as u32,
        )
    } else if rows4 {
        (
            shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL_ROWS4,
            (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32,
        )
    } else {
        (
            shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL,
            (batch * out_dim.div_ceil(32)) as u32,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn gdn_in_proj_bf16w_batched_plan(
    batch: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    total_out: usize,
) -> (&'static str, u32) {
    let pair_qkv_z = batch > 1 && vulkan_kernel_policy().gdn_in_proj_batch_pair_qkv_z;
    let row_grouping =
        pair_qkv_z && batch >= 3 && vulkan_kernel_policy().gdn_in_proj_batch_row_pair;
    let row_group_size = if row_grouping
        && batch >= gdn_in_proj_rows8_min_batch()
        && vulkan_kernel_policy().gdn_in_proj_batch_row_octet
    {
        8usize
    } else if row_grouping
        && batch >= gdn_in_proj_rows4_min_batch()
        && vulkan_kernel_policy().gdn_in_proj_batch_row_quad
    {
        4usize
    } else if row_grouping {
        2usize
    } else {
        1usize
    };
    let dispatch_cols = if pair_qkv_z {
        qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + a_dim + b_dim
    } else {
        total_out
    };
    let shader = if row_group_size == 8 {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS8_BF16W
    } else if row_group_size == 4 {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W
    } else if row_group_size == 2 {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS2_BF16W
    } else if pair_qkv_z {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_BF16W
    } else {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_BF16W
    };
    let row_groups = batch.div_ceil(row_group_size);
    (shader, (row_groups * dispatch_cols.div_ceil(80)) as u32)
}

fn gdn_qk_norm_recurrent_fusion_enabled(
    batch_size: usize,
    gqa_ratio: usize,
    dk: usize,
    dv: usize,
) -> bool {
    (2..=16).contains(&batch_size)
        && gqa_ratio == 2
        && dk == dv
        && gqa_ratio * dv <= 256
        && vulkan_kernel_policy().gdn_qk_norm_recurrent_fusion
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
/// Returns `Ok(Some(output_tensor))` on success -- the post-MLP residual,
/// shape `[1, 1, hidden_size]`, as a kt tensor.
///
/// Returns `Ok(None)` when the input does not match the supported
/// configuration (see module docs); the caller should fall back to the
/// older block helper.
///
/// The KV cache write lands in the supplied `VkPagedKvCache` at the
/// block-table-resolved slot for `start_pos`. The source `PagedKvCacheKt`
/// is **not** updated — once the resident path is engaged for a layer,
/// it owns that layer's KV state for the remainder of the decode
/// session.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_paged_decode_full_attn_resident_b1(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    start_pos: usize,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    paged_cache: &PagedKvCacheKt,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos: &kiln_tensor::Tensor,
    rope_sin: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
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
        .ok_or_else(|| anyhow::anyhow!("no slot for start_pos {start_pos} in block table"))?;

    // --- weight buffer lookups (cached on backend) -------------------
    // Q/K/V/O projections + MLP gate/up/down: bf16-packed
    let q_w_buf = backend.cached_bf16_packed_weight_buffer_kt(&attn.q_proj_t)?;
    let k_w_buf = backend.cached_bf16_packed_weight_buffer_kt(&attn.k_proj_t)?;
    let v_w_buf = backend.cached_bf16_packed_weight_buffer_kt(&attn.v_proj_t)?;
    let o_w_buf = backend.cached_bf16_packed_weight_buffer_kt(&attn.o_proj_t)?;
    let gate_w_buf = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.gate_proj_t)?;
    let up_w_buf = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.up_proj_t)?;
    let down_w_buf = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.down_proj_t)?;

    // RMSnorm weights: f32 (the kt storage may be bf16; the cache
    // helper converts on first lookup).
    let in_norm_buf = backend.cached_f32_weight_buffer_kt(&layer.input_layernorm)?;
    let post_norm_buf = backend.cached_f32_weight_buffer_kt(&layer.post_attention_layernorm)?;
    let q_norm_buf = backend.cached_f32_weight_buffer_kt(&attn.q_norm)?;
    let k_norm_buf = backend.cached_f32_weight_buffer_kt(&attn.k_norm)?;

    // --- rope cos/sin upload (per-step, single position) -------------
    let rope_cos_data = kt_tensor_to_f32_vec(rope_cos).context("extract rope_cos f32 data")?;
    let rope_cos_buf =
        kiln_vulkan_kernel::kernels::upload_f32_buffer_from_slice(vk_device, &rope_cos_data)?;
    let rope_sin_data = kt_tensor_to_f32_vec(rope_sin).context("extract rope_sin f32 data")?;
    let rope_sin_buf =
        kiln_vulkan_kernel::kernels::upload_f32_buffer_from_slice(vk_device, &rope_sin_data)?;

    // --- activation buffer acquisition (pooled, persistent across
    //     resident decode calls on the same backend) -----------------
    let x_buf = backend.acquire_resident_scratch("fa_x", (hidden * 4) as u64)?;
    let normed_buf = backend.acquire_resident_scratch("fa_normed", (hidden * 4) as u64)?;
    let qkv_combined = backend
        .acquire_resident_scratch("fa_qkv_combined", ((q_dim + k_dim + v_dim) * 4) as u64)?;
    let q_buf = backend.acquire_resident_scratch("fa_q", (num_heads * head_dim * 4) as u64)?;
    let q_rot_buf =
        backend.acquire_resident_scratch("fa_q_rot", (num_heads * head_dim * 4) as u64)?;
    let gate_buf =
        backend.acquire_resident_scratch("fa_gate", (num_heads * head_dim * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("fa_k", (kv_elems * 4) as u64)?;
    let k_rot_buf = backend.acquire_resident_scratch("fa_k_rot", (kv_elems * 4) as u64)?;
    let v_buf = backend.acquire_resident_scratch("fa_v", (kv_elems * 4) as u64)?;
    let attn_pre_gate =
        backend.acquire_resident_scratch("fa_attn_pre_gate", (num_heads * head_dim * 4) as u64)?;
    let attn_post_gate =
        backend.acquire_resident_scratch("fa_attn_post_gate", (num_heads * head_dim * 4) as u64)?;
    let attn_out_buf = backend.acquire_resident_scratch("fa_attn_out", (hidden * 4) as u64)?;
    let attn_residual =
        backend.acquire_resident_scratch("fa_attn_residual", (hidden * 4) as u64)?;
    let mlp_scratch =
        backend.acquire_resident_scratch("fa_mlp_scratch", (intermediate * 4) as u64)?;
    let mlp_out_buf = backend.acquire_resident_scratch("fa_mlp_out", (hidden * 4) as u64)?;
    let final_out = backend.acquire_resident_scratch("fa_final_out", (hidden * 4) as u64)?;

    // block_table + seq_lens for paged-attn read
    let blocks: Vec<u32> = block_table.blocks.clone();
    let max_blocks_per_seq = blocks.len();
    let block_table_buf = upload_u32_slice(vk_device, &blocks)?;
    let seq_lens: [u32; 1] = [(start_pos + 1) as u32];
    let seq_lens_buf = upload_u32_slice(vk_device, &seq_lens)?;

    let fa_t0 = if timing_enabled() {
        Some(Instant::now())
    } else {
        None
    };

    // --- upload x ----------------------------------------------------
    let upload_t0 = fa_t0.map(|_| Instant::now());
    let x_dtype = x.dtype();
    let x_data = kt_tensor_to_f32_vec(x).context("extract x f32 data for resident block")?;
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
    let k_pool = vk_kv_cache
        .k_buffer(full_attn_layer_idx)
        .ok_or_else(|| anyhow::anyhow!("VkPagedKvCache missing layer {full_attn_layer_idx}"))?;
    let v_pool = vk_kv_cache
        .v_buffer(full_attn_layer_idx)
        .ok_or_else(|| anyhow::anyhow!("VkPagedKvCache missing layer {full_attn_layer_idx}"))?;
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
    // 6) RoPE Q+K. vk_rope_qk_f32 push:
    // [rows, num_q_heads, num_kv_heads, head_dim, rotary_dim].
    batch.record_shader(
        shaders::VK_ROPE_QK_F32,
        &[
            q_buf.handle(),
            k_buf.handle(),
            rope_cos_buf.handle(),
            rope_sin_buf.handle(),
            q_rot_buf.handle(),
            k_rot_buf.handle(),
        ],
        &[
            1u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            rotary_dim as u32,
        ],
        Workgroups::OneD(((num_heads + num_kv_heads) * head_dim).div_ceil(256) as u32),
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
        &[
            attn_residual.handle(),
            post_norm_buf.handle(),
            normed_buf.handle(),
        ],
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
        &[
            mlp_scratch.handle(),
            down_w_buf.handle(),
            mlp_out_buf.handle(),
        ],
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

    replay_vulkan_command_batch(
        batch,
        "vk-resident full-attn block",
        vec![1, hidden, max_blocks_per_seq, block_size],
        1,
        vec![
            vk_replay_resource(&x_buf, DType::F32, vec![1, 1, hidden]),
            vk_replay_resource(&final_out, DType::F32, vec![1, 1, hidden]),
            vk_replay_resource(k_pool, DType::F32, vec![k_pool.size() as usize / 4]),
            vk_replay_resource(v_pool, DType::F32, vec![v_pool.size() as usize / 4]),
            vk_replay_resource(&block_table_buf, DType::U32, vec![max_blocks_per_seq]),
            vk_replay_resource(&seq_lens_buf, DType::U32, vec![1]),
            vk_replay_resource(&rope_cos_buf, DType::F32, vec![rope_cos_data.len()]),
            vk_replay_resource(&rope_sin_buf, DType::F32, vec![rope_sin_data.len()]),
        ],
    )
    .context("submit resident full-attn CommandBatch through ReplayPlan")?;
    if let Some(t) = submit_t0 {
        FA_SUBMIT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // --- read back final_out as a kt tensor
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
    let out_tensor = kiln_tensor::Tensor::from_vec(out_f32, vec![1usize, 1usize, hidden])?;
    let out_tensor = if x_dtype == kiln_tensor::DType::F32 {
        out_tensor
    } else {
        out_tensor.to_dtype(x_dtype)?
    };
    if let Some(t) = readback_t0 {
        FA_READBACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Some(t) = fa_t0 {
        FA_TOTAL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        FA_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    Ok(Some(out_tensor))
}

/// Run one GDN decode block on the Vulkan-resident path with kt inputs.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_paged_decode_gdn_resident_b1_kt(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    recurrent_state_t: &kiln_tensor::Tensor,
    conv_state_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    let dims = x.dims();
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

    let gdn_t0 = if timing_enabled() {
        Some(Instant::now())
    } else {
        None
    };
    let x_dtype = x.dtype();
    let x_buf = backend.acquire_resident_scratch("gdn_kt_x", (hidden * 4) as u64)?;
    let out_buf = backend.acquire_resident_scratch("gdn_kt_out", (hidden * 4) as u64)?;

    let upload_t0 = gdn_t0.map(|_| Instant::now());
    let x_data = kt_tensor_to_f32_vec(x).context("extract x f32 data for resident GDN block")?;
    let x_bytes = f32_slice_to_bytes(&x_data);
    VulkanBuffer::upload_data(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &x_buf,
        &x_bytes,
    )
    .context("upload x for resident GDN block")?;
    if let Some(t) = upload_t0 {
        GDNFB_UPLOAD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let submit_t0 = gdn_t0.map(|_| Instant::now());
    let mut batch = CommandBatch::new(vk_device)?;
    if !record_gdn_block_into(
        backend,
        &mut batch,
        &x_buf,
        &out_buf,
        layer,
        config,
        recurrent_state_t,
        conv_state_t,
    )? {
        return Ok(None);
    }
    replay_vulkan_command_batch(
        batch,
        "vk-resident GDN full-block kt",
        vec![1, hidden],
        1,
        vec![
            vk_replay_resource(&x_buf, DType::F32, vec![1, 1, hidden]),
            vk_replay_resource(&out_buf, DType::F32, vec![1, 1, hidden]),
        ],
    )
    .context("submit resident GDN full-block kt CommandBatch through ReplayPlan")?;
    if let Some(t) = submit_t0 {
        GDNFB_SUBMIT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    let readback_t0 = gdn_t0.map(|_| Instant::now());
    let out_bytes = VulkanBuffer::read_back(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &out_buf,
    )
    .context("read back resident GDN block output")?;
    let out_f32 = bytes_to_f32_vec(&out_bytes);
    let out_tensor = kiln_tensor::Tensor::from_vec(out_f32, vec![1usize, 1usize, hidden])?;
    let out_tensor = if x_dtype == kiln_tensor::DType::F32 {
        out_tensor
    } else {
        out_tensor.to_dtype(x_dtype)?
    };
    if let Some(t) = readback_t0 {
        GDNFB_READBACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Some(t) = gdn_t0 {
        GDNFB_TOTAL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        GDNFB_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    Ok(Some(out_tensor))
}

fn kt_tensor_to_f32_vec(tensor: &kiln_tensor::Tensor) -> Result<Vec<f32>> {
    let flat = if tensor.dtype() == kiln_tensor::DType::F32 {
        tensor.flatten_all()?
    } else {
        tensor.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
    };
    flat.to_vec1::<f32>().context("extract kt tensor f32 data")
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
/// f32, as a kt tensor. The caller is responsible for
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
/// 24 GDN layers x 5 host-side ops (pre-norm + residual + post-norm + MLP
/// + final-residual) that previously dominated decode ITL (~17 ms /
/// GDN layer = ~408 ms / token in the historical resident-decode profile).
///
/// Returns `Ok(Some(post_block_residual_tensor))` on success;
/// `Ok(None)` on any unsupported configuration so the caller falls back
/// to the legacy path bit-identically.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_paged_decode_gdn_resident_b1(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    recurrent_state_t: &kiln_tensor::Tensor,
    conv_state_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
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

    let fb_t0 = if timing_enabled() {
        Some(Instant::now())
    } else {
        None
    };

    // --- weight buffer lookups (cached on backend) -------------------
    let qkv_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_qkv_t)?;
    let z_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_z_t)?;
    let a_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_a_t)?;
    let b_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_b_t)?;
    let out_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.out_proj_t)?;
    let conv_w = backend.cached_f32_weight_buffer_kt(&lin_weights.conv1d)?;
    let qk_norm = backend.cached_f32_weight_buffer_kt(&lin_weights.norm)?;
    let a_log = backend.cached_f32_weight_buffer_kt(&lin_weights.a_log)?;
    let dt_bias = backend.cached_f32_weight_buffer_kt(&lin_weights.dt_bias)?;
    let gate_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.gate_proj_t)?;
    let up_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.up_proj_t)?;
    let down_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.down_proj_t)?;
    let in_norm = backend.cached_f32_weight_buffer_kt(&layer.input_layernorm)?;
    let post_norm = backend.cached_f32_weight_buffer_kt(&layer.post_attention_layernorm)?;

    // --- persistent state buffers --------------------------------
    let recurrent_bytes = (1 * nv * dk * dv * 4) as u64;
    let recurrent_buf =
        backend.linear_attn_recurrent_state_buffer_kt(state_key, recurrent_bytes)?;
    let conv_state_bytes = (1 * qkv_dim * (conv_kernel.saturating_sub(1)) * 4) as u64;
    let conv_buf = backend.linear_attn_conv_state_buffer_kt(state_key, conv_state_bytes)?;

    if !backend.linear_attn_layer_seeded_kt(state_key) {
        seed_recurrent_state_kt(vk_device, &recurrent_buf, recurrent_state_t)?;
        seed_conv_state_kt(vk_device, &conv_buf, conv_state_t)?;
        backend.mark_linear_attn_layer_seeded_kt(state_key);
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
    // q_expanded / k_expanded: GQA-expanded [nv, dk] outputs of L2-norm.
    // The gates_recurrent kernel indexes Q/K with (bidx*nv + h)*dk, so
    // its input must be sized for nv heads (not just nk).
    let qkv_expanded_bytes = (nv * dk * 4) as u64;
    let q_expanded = backend.acquire_resident_scratch("gdnfb_q_expanded", qkv_expanded_bytes)?;
    let k_expanded = backend.acquire_resident_scratch("gdnfb_k_expanded", qkv_expanded_bytes)?;
    let v_buf = backend.acquire_resident_scratch("gdnfb_v", (v_dim * 4) as u64)?;
    let gated_norm = backend.acquire_resident_scratch("gdnfb_gated_norm", (v_dim * 4) as u64)?;
    let gdn_out = backend.acquire_resident_scratch("gdnfb_gdn_out", (hidden * 4) as u64)?;
    let attn_residual =
        backend.acquire_resident_scratch("gdnfb_attn_residual", (hidden * 4) as u64)?;
    let normed_post = backend.acquire_resident_scratch("gdnfb_normed_post", (hidden * 4) as u64)?;
    let mlp_scratch =
        backend.acquire_resident_scratch("gdnfb_mlp_scratch", (intermediate * 4) as u64)?;
    let mlp_out = backend.acquire_resident_scratch("gdnfb_mlp_out", (hidden * 4) as u64)?;
    let final_out = backend.acquire_resident_scratch("gdnfb_final_out", (hidden * 4) as u64)?;

    // --- upload x ------------------------------------------------
    let upload_t0 = fb_t0.map(|_| Instant::now());
    let x_f32 = if x.dtype() == kiln_tensor::DType::F32 {
        x.flatten_all()?
    } else {
        x.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
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
    // 4) causal_conv1d output + ordered state advance.
    record_causal_conv1d_update_into(
        &mut batch,
        &mixed_qkv,
        &conv_w,
        &conv_buf,
        &conv_qkv,
        1,
        qkv_dim,
        1,
        conv_kernel,
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
    // 6) Q-norm: per-head L2 normalize + GQA expansion from nk to nv heads,
    //    scale = 1/sqrt(dk). Matches legacy `gdn_qk_norm` semantics:
    //    `q.unsqueeze(3).expand(.., nk, gqa_ratio, dk).reshape(nv, dk)`
    //    then `l2_normalize(q) * (1/sqrt(dk))`. Writes the expanded
    //    [nv, dk] result into `q_expanded`. The legacy resident path used
    //    qwen_rmsnorm with `lin_weights.norm` and skipped the expansion
    //    entirely — the gates_recurrent kernel then read Q/K past the end
    //    of the nk-sized buffer (returning zeros under robustBufferAccess)
    //    for heads nk..nv. Bit-identical to legacy for one decode step
    //    because of the storage-buffer zero-fill, but state diverged
    //    immediately on multi-token decode.
    let gqa_ratio = nv / nk;
    debug_assert!(gqa_ratio * nk == nv, "GDN nv must be a multiple of nk");
    let l2_eps_q: f32 = 1e-6;
    let q_scale: f32 = 1.0 / (dk as f32).sqrt();
    let k_scale: f32 = 1.0;
    batch.record_shader(
        shaders::L2_NORM_QK_PER_ROW,
        &[
            q_buf.handle(),
            k_buf.handle(),
            q_expanded.handle(),
            k_expanded.handle(),
        ],
        &[
            nk as u32,
            dk as u32,
            l2_eps_q.to_bits(),
            q_scale.to_bits(),
            k_scale.to_bits(),
            gqa_ratio as u32,
        ],
        Workgroups::OneD(nk as u32),
    )?;
    // 8) Fused gates+recurrent+rmsnorm — reads Q/K from the GQA-expanded
    //    [nv, dk] buffers written by the L2-norm step above.
    batch.record_shader(
        shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
        &[
            q_expanded.handle(),
            k_expanded.handle(),
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
        &[attn_residual.handle(), mlp_out.handle(), final_out.handle()],
        &[hidden as u32],
        Workgroups::OneD(hidden.div_ceil(256) as u32),
    )?;

    replay_vulkan_command_batch(
        batch,
        "vk-resident GDN full-block",
        vec![1, hidden],
        1,
        vec![
            vk_replay_resource(&x_buf, DType::F32, vec![1, 1, hidden]),
            vk_replay_resource(&final_out, DType::F32, vec![1, 1, hidden]),
            vk_replay_resource(
                &recurrent_buf,
                DType::F32,
                vec![recurrent_bytes as usize / 4],
            ),
            vk_replay_resource(&conv_buf, DType::F32, vec![conv_state_bytes as usize / 4]),
        ],
    )
    .context("submit resident GDN full-block CommandBatch through ReplayPlan")?;
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
    let out_tensor = kiln_tensor::Tensor::from_vec(out_f32, vec![1usize, 1usize, hidden])?;
    let out_tensor = if x.dtype() == kiln_tensor::DType::F32 {
        out_tensor
    } else {
        out_tensor.to_dtype(x.dtype())?
    };
    if let Some(t) = readback_t0 {
        GDNFB_READBACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Some(t) = fb_t0 {
        GDNFB_TOTAL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        GDNFB_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    Ok(Some(out_tensor))
}

/// configuration; caller falls back to the nonresident
/// `gated_deltanet_forward_decode_if` path.
///
/// Persistent state: the recurrent_state and conv_state buffers live
/// on `VulkanBackend` (per-layer, allocated lazily). On first call per
/// layer, they're seeded from the kt `LinearAttentionState` Tensors
/// so any prefill GDN state is preserved.
#[allow(clippy::too_many_arguments)]
pub fn gated_deltanet_forward_decode_resident_b1_kt(
    backend: &VulkanBackend,
    x_normed: &kiln_tensor::Tensor,
    weights: &crate::forward::GpuLinearAttentionWeights,
    config: &ModelConfig,
    recurrent_state_t: &kiln_tensor::Tensor,
    conv_state_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
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
    let qkv_w = backend.cached_bf16_packed_weight_buffer_kt(&weights.in_proj_qkv_t)?;
    let z_w = backend.cached_bf16_packed_weight_buffer_kt(&weights.in_proj_z_t)?;
    let a_w = backend.cached_bf16_packed_weight_buffer_kt(&weights.in_proj_a_t)?;
    let b_w = backend.cached_bf16_packed_weight_buffer_kt(&weights.in_proj_b_t)?;
    let out_w = backend.cached_bf16_packed_weight_buffer_kt(&weights.out_proj_t)?;
    let conv_w = backend.cached_f32_weight_buffer_kt(&weights.conv1d)?;
    let q_norm = backend.cached_f32_weight_buffer_kt(&weights.norm)?; // used for gated_rms_norm
    // a_log and dt_bias: these enter the fused recurrent+rmsnorm kernel.
    let a_log = backend.cached_f32_weight_buffer_kt(&weights.a_log)?;
    let dt_bias = backend.cached_f32_weight_buffer_kt(&weights.dt_bias)?;

    // --- persistent state buffers --------------------------------
    let recurrent_bytes = (1 * nv * dk * dv * 4) as u64;
    let recurrent_buf =
        backend.linear_attn_recurrent_state_buffer_kt(state_key, recurrent_bytes)?;
    // conv_state shape: [batch, conv_dim, kernel_size - 1] f32 where
    // conv_dim = qkv_dim (the conv1d operates on the full mixed_qkv).
    let conv_state_bytes = (1 * qkv_dim * (conv_kernel.saturating_sub(1)) * 4) as u64;
    let conv_buf = backend.linear_attn_conv_state_buffer_kt(state_key, conv_state_bytes)?;

    // --- seed state from kt tensors on first use ------------------
    if !backend.linear_attn_layer_seeded_kt(state_key) {
        seed_recurrent_state_kt(vk_device, &recurrent_buf, recurrent_state_t)?;
        seed_conv_state_kt(vk_device, &conv_buf, conv_state_t)?;
        backend.mark_linear_attn_layer_seeded_kt(state_key);
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
    // GQA-expanded Q/K for the recurrent kernel (sized nv*dk, not nk*dk).
    let qkv_expanded_bytes = (nv * dk * 4) as u64;
    let q_expanded = backend.acquire_resident_scratch("gdn_q_expanded", qkv_expanded_bytes)?;
    let k_expanded = backend.acquire_resident_scratch("gdn_k_expanded", qkv_expanded_bytes)?;
    let v_buf = backend.acquire_resident_scratch("gdn_v", (v_dim * 4) as u64)?;
    let gated_norm = backend.acquire_resident_scratch("gdn_gated_norm", (v_dim * 4) as u64)?;
    let out_buf = backend.acquire_resident_scratch("gdn_out", (hidden * 4) as u64)?;

    let gdn_t0 = if timing_enabled() {
        Some(Instant::now())
    } else {
        None
    };

    // --- upload x -----------------------------------------------
    let gdn_upload_t0 = gdn_t0.map(|_| Instant::now());
    let x_f32 = if x_normed.dtype() == kiln_tensor::DType::F32 {
        x_normed.flatten_all()?
    } else {
        x_normed.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
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
    // 3) causal_conv1d output + ordered state advance.
    record_causal_conv1d_update_into(
        &mut batch,
        &mixed_qkv,
        &conv_w,
        &conv_buf,
        &conv_qkv,
        1,
        qkv_dim,
        1,
        conv_kernel,
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
    // 5) Q-norm: per-head L2 normalize + GQA expansion (nk → nv heads),
    //    scale = 1/sqrt(dk). Matches legacy `gdn_qk_norm` semantics.
    let gqa_ratio = nv / nk;
    debug_assert!(gqa_ratio * nk == nv, "GDN nv must be a multiple of nk");
    let l2_eps_qk: f32 = 1e-6;
    let q_scale: f32 = 1.0 / (dk as f32).sqrt();
    let k_scale: f32 = 1.0;
    batch.record_shader(
        shaders::L2_NORM_QK_PER_ROW,
        &[
            q_buf.handle(),
            k_buf.handle(),
            q_expanded.handle(),
            k_expanded.handle(),
        ],
        &[
            nk as u32,
            dk as u32,
            l2_eps_qk.to_bits(),
            q_scale.to_bits(),
            k_scale.to_bits(),
            gqa_ratio as u32,
        ],
        Workgroups::OneD(nk as u32),
    )?;
    // 7) Fused gates+recurrent+rmsnorm. push = [nv, dk, dv, eps_bits, batch],
    //    workgroups = batch*nv. Reads Q/K from the GQA-expanded buffers.
    batch.record_shader(
        shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
        &[
            q_expanded.handle(),
            k_expanded.handle(),
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

    replay_vulkan_command_batch(
        batch,
        "vk-resident GDN block",
        vec![1, hidden],
        1,
        vec![
            vk_replay_resource(&x_buf, DType::F32, vec![1, 1, hidden]),
            vk_replay_resource(&out_buf, DType::F32, vec![1, 1, hidden]),
            vk_replay_resource(
                &recurrent_buf,
                DType::F32,
                vec![recurrent_bytes as usize / 4],
            ),
            vk_replay_resource(&conv_buf, DType::F32, vec![conv_state_bytes as usize / 4]),
        ],
    )
    .context("submit resident GDN CommandBatch through ReplayPlan")?;
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
    let out_tensor = kiln_tensor::Tensor::from_vec(out_f32, vec![1usize, 1usize, hidden])?;
    let out_tensor = if x_normed.dtype() == kiln_tensor::DType::F32 {
        out_tensor
    } else {
        out_tensor.to_dtype(x_normed.dtype())?
    };
    if let Some(t) = gdn_readback_t0 {
        GDN_READBACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    if let Some(t) = gdn_t0 {
        GDN_TOTAL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        GDN_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    Ok(Some(out_tensor))
}

pub(crate) fn seed_recurrent_state_kt(
    vk_device: &VulkanDevice,
    buf: &VulkanBuffer,
    state_t: &kiln_tensor::Tensor,
) -> Result<()> {
    let flat = if state_t.dtype() == kiln_tensor::DType::F32 {
        state_t.flatten_all()?
    } else {
        state_t.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
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
    .context("seed recurrent kt state")
}

pub(crate) fn seed_conv_state_kt(
    vk_device: &VulkanDevice,
    buf: &VulkanBuffer,
    state_t: &kiln_tensor::Tensor,
) -> Result<()> {
    let flat = if state_t.dtype() == kiln_tensor::DType::F32 {
        state_t.flatten_all()?
    } else {
        state_t.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
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
    .context("seed conv kt state")
}

/// Seed only the blocks this request's `block_table` references —
/// not the multi-GB full pool. For a Qwen3.5-4B request with a 32-
/// token prompt at block_size=16 that's ~2 blocks × 8 layers × 64 KB =
/// ~1 MB seeded per first-decode call, vs. the 11 GB the full-slab
/// `seed_vk_kv_cache_layer_from_kt` would have to copy across
/// unified memory on every fresh request. Untouched slots stay at
/// their previous content; the attention seq-len mask ignores
/// anything past the active range, so unused-slot contents are
/// irrelevant.
pub fn seed_vk_kv_cache_layer_blocks_from_kt(
    vk_device: &VulkanDevice,
    vk_cache: &VkPagedKvCache,
    paged_cache: &PagedKvCacheKt,
    layer_idx: usize,
    block_ids: &[u32],
) -> Result<()> {
    if block_ids.is_empty() {
        return Ok(());
    }
    let (k_tensor, v_tensor) = paged_cache
        .pool_tensors(layer_idx)
        .ok_or_else(|| anyhow::anyhow!("kt paged_cache layer {layer_idx} out of range"))?;
    let block_size = paged_cache.block_size();
    // pool_tensors are shaped [total_slots, num_kv_heads, head_dim]
    // (flat slot index — block_idx maps to slot range
    // [block_idx*block_size, (block_idx+1)*block_size)). Slice
    // block-by-block — each block is small (block_size × kv_heads ×
    // head_dim × 4 bytes; ~64 KB on Qwen3.5-4B) and uploads to the
    // resident pool's matching slot range. The payloads are collected
    // first so K/V for all active blocks land through one Vulkan
    // transfer submit for this layer.
    let mut block_payloads = Vec::with_capacity(block_ids.len());
    for &block_id in block_ids {
        let bid = block_id as usize;
        let slot_start = bid * block_size;
        let k_block = k_tensor.narrow(0, slot_start, block_size)?.contiguous()?;
        let v_block = v_tensor.narrow(0, slot_start, block_size)?.contiguous()?;
        // #1082: `pool_tensors` now hands back `kiln_tensor::Tensor` (kt), so
        // compare/convert against kt's `DType`, not the old dtype enum.
        let k_block_f32 = if k_block.dtype() == kiln_tensor::DType::F32 {
            k_block.flatten_all()?
        } else {
            k_block.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
        };
        let v_block_f32 = if v_block.dtype() == kiln_tensor::DType::F32 {
            v_block.flatten_all()?
        } else {
            v_block.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
        };
        let k_data: Vec<f32> = k_block_f32.to_vec1()?;
        let v_data: Vec<f32> = v_block_f32.to_vec1()?;
        let k_bytes = f32_slice_to_bytes(&k_data);
        let v_bytes = f32_slice_to_bytes(&v_data);
        block_payloads.push((bid, k_bytes, v_bytes));
    }
    let uploads: Vec<(usize, &[u8], &[u8])> = block_payloads
        .iter()
        .map(|(bid, k_bytes, v_bytes)| (*bid, k_bytes.as_slice(), v_bytes.as_slice()))
        .collect();
    vk_cache.upload_layer_blocks_from_f32(vk_device, layer_idx, &uploads)
}

/// Seed one full-attention layer from the union of physical blocks
/// referenced by a batched decode step.
///
/// Returns the number of unique physical blocks copied.
pub fn seed_vk_kv_cache_layer_blocks_from_batched_tables(
    vk_device: &VulkanDevice,
    vk_cache: &VkPagedKvCache,
    paged_cache: &PagedKvCacheKt,
    layer_idx: usize,
    block_tables: &[&BlockTable],
) -> Result<usize> {
    let block_ids = unique_physical_blocks(block_tables);
    let seeded = block_ids.len();
    seed_vk_kv_cache_layer_blocks_from_kt(vk_device, vk_cache, paged_cache, layer_idx, &block_ids)?;
    Ok(seeded)
}

/// Full-slab seed kept for callers that explicitly want the whole
/// layer uploaded at once. Production resident decode uses the
/// per-block variant above to keep per-request seeding bounded.
pub fn seed_vk_kv_cache_layer_from_kt(
    vk_device: &VulkanDevice,
    vk_cache: &VkPagedKvCache,
    paged_cache: &PagedKvCacheKt,
    layer_idx: usize,
) -> Result<()> {
    let (k_tensor, v_tensor) = paged_cache
        .pool_tensors(layer_idx)
        .ok_or_else(|| anyhow::anyhow!("kt paged_cache layer {layer_idx} out of range"))?;
    // #1082: kt pool tensors — compare/convert against kt's `DType`.
    let k_flat = if k_tensor.dtype() == kiln_tensor::DType::F32 {
        k_tensor.flatten_all()?
    } else {
        k_tensor.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
    };
    let v_flat = if v_tensor.dtype() == kiln_tensor::DType::F32 {
        v_tensor.flatten_all()?
    } else {
        v_tensor.to_dtype(kiln_tensor::DType::F32)?.flatten_all()?
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
// submit, wait, and round-trip their input and output. That's one
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
// the final RMSNorm + LM head via the nonresident path (cheap, one shot).
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
    paged_cache: &PagedKvCacheKt,
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
    let q_w = backend.cached_bf16_packed_weight_buffer_kt(&attn.q_proj_t)?;
    let k_w = backend.cached_bf16_packed_weight_buffer_kt(&attn.k_proj_t)?;
    let v_w = backend.cached_bf16_packed_weight_buffer_kt(&attn.v_proj_t)?;
    let o_w = backend.cached_bf16_packed_weight_buffer_kt(&attn.o_proj_t)?;
    let gate_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.gate_proj_t)?;
    let up_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.up_proj_t)?;
    let down_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.down_proj_t)?;
    let in_norm = backend.cached_f32_weight_buffer_kt(&layer.input_layernorm)?;
    let post_norm = backend.cached_f32_weight_buffer_kt(&layer.post_attention_layernorm)?;
    let q_norm = backend.cached_f32_weight_buffer_kt(&attn.q_norm)?;
    let k_norm = backend.cached_f32_weight_buffer_kt(&attn.k_norm)?;

    // --- per-layer scratch buffers (pooled, persistent) --------------
    // These can be SHARED across all full-attn layers within one batch
    // because each dispatch reads only its predecessor's output, with
    // a compute→compute barrier between, and dispatches are recorded
    // and executed in strict program order.
    let normed = backend.acquire_resident_scratch("nfa_normed", (hidden * 4) as u64)?;
    let qkv_combined = backend
        .acquire_resident_scratch("nfa_qkv_combined", ((q_dim + k_dim + v_dim) * 4) as u64)?;
    let q_buf = backend.acquire_resident_scratch("nfa_q", (num_heads * head_dim * 4) as u64)?;
    let gate_buf =
        backend.acquire_resident_scratch("nfa_gate", (num_heads * head_dim * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("nfa_k", (k_dim * 4) as u64)?;
    let v_buf = backend.acquire_resident_scratch("nfa_v", (v_dim * 4) as u64)?;
    let q_rot = backend.acquire_resident_scratch("nfa_q_rot", (num_heads * head_dim * 4) as u64)?;
    let k_rot = backend.acquire_resident_scratch("nfa_k_rot", (k_dim * 4) as u64)?;
    let attn_pre_gate =
        backend.acquire_resident_scratch("nfa_attn_pre_gate", (num_heads * head_dim * 4) as u64)?;
    let attn_post_gate = backend
        .acquire_resident_scratch("nfa_attn_post_gate", (num_heads * head_dim * 4) as u64)?;
    let attn_out = backend.acquire_resident_scratch("nfa_attn_out", (hidden * 4) as u64)?;
    let attn_residual =
        backend.acquire_resident_scratch("nfa_attn_residual", (hidden * 4) as u64)?;
    let normed_post = backend.acquire_resident_scratch("nfa_normed_post", (hidden * 4) as u64)?;
    let mlp_scratch =
        backend.acquire_resident_scratch("nfa_mlp_scratch", (intermediate * 4) as u64)?;

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
        shaders::VK_ROPE_QK_F32,
        &[
            q_buf.handle(),
            k_buf.handle(),
            rope_cos_buf.handle(),
            rope_sin_buf.handle(),
            q_rot.handle(),
            k_rot.handle(),
        ],
        &[
            1u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            rotary_dim as u32,
        ],
        Workgroups::OneD(((num_heads + num_kv_heads) * head_dim).div_ceil(256) as u32),
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
    // recurrence. The bounded heuristic selects the chunk count without
    // consulting device identity. Anything >= seq_len degrades gracefully
    // because chunks beyond `seq_len` write neutral identities.
    let num_chunks = paged_attn_splitk_chunks(1, max_blocks_per_seq);
    let partials_stride = 2 + head_dim;
    let partials_bytes = (1 * num_heads * num_chunks * partials_stride * 4) as u64;
    let attn_partials = backend.acquire_resident_scratch("nfa_attn_partials", partials_bytes)?;
    kiln_vulkan_kernel::resident::record_paged_attn_decode_batch_paged_splitk_resident(
        batch,
        q_rot.as_ref(),
        k_pool,
        v_pool,
        block_table_buf,
        seq_lens_buf,
        &attn_partials,
        attn_pre_gate.as_ref(),
        1,
        num_heads,
        num_kv_heads,
        head_dim,
        max_blocks_per_seq,
        block_size,
        softmax_scale,
        num_chunks,
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
        &[attn_post_gate.handle(), o_w.handle(), attn_out.handle()],
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
    // MLP down fused with the final residual ADD: writes
    // `x_out_buf[c] = sum_k mlp_scratch[k] * down_w[k,c] + attn_residual[c]`.
    // Saves one dispatch + one compute->compute barrier per block.
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W_ADD_RESIDUAL,
        &[
            mlp_scratch.handle(),
            down_w.handle(),
            attn_residual.handle(),
            x_out_buf.handle(),
        ],
        &[intermediate as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    Ok(true)
}

/// Record one full-attention block for `batch_size` independent
/// single-token decode rows. This is the multi-row counterpart to
/// [`record_full_attn_block_into`]: all inputs are already resident,
/// and row-varying decode metadata is supplied as device buffers.
///
/// Buffer layout expected by this recorder:
/// - `x_in_buf` / `x_out_buf`: f32 `[batch_size, hidden]`
/// - `rope_cos_buf` / `rope_sin_buf`: f32 `[batch_size, rotary_dim / 2]`
/// - `block_table_buf`: u32 `[batch_size, max_blocks_per_seq]`
/// - `seq_lens_buf`: u32 `[batch_size]`
/// - `slots_buf`: u32 `[batch_size]`
///
/// Returns `Ok(false)` for unsupported layer/config combinations so
/// the caller can keep an explicit fallback boundary.
#[allow(clippy::too_many_arguments)]
pub fn record_full_attn_block_batched_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    x_in_buf: &VulkanBuffer,
    x_out_buf: &VulkanBuffer,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    batch_size: usize,
    max_blocks_per_seq: usize,
    block_size: usize,
    full_attn_layer_idx: usize,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
) -> Result<bool> {
    let attn = match &layer.attention {
        crate::forward::GpuAttentionWeights::Full(w) => w,
        _ => return Ok(false),
    };
    if !config.attn_output_gate {
        return Ok(false);
    }
    anyhow::ensure!(
        batch_size > 0,
        "batched full-attn block: batch_size must be > 0"
    );
    anyhow::ensure!(
        max_blocks_per_seq > 0 && block_size > 0,
        "batched full-attn block: paged metadata dimensions must be > 0"
    );

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
    let q_h_d = num_heads * head_dim;
    let total_qkv_out = q_dim + k_dim + v_dim;

    let q_w = backend.cached_bf16_packed_weight_buffer_kt(&attn.q_proj_t)?;
    let k_w = backend.cached_bf16_packed_weight_buffer_kt(&attn.k_proj_t)?;
    let v_w = backend.cached_bf16_packed_weight_buffer_kt(&attn.v_proj_t)?;
    let o_w = backend.cached_bf16_packed_weight_buffer_kt(&attn.o_proj_t)?;
    let gate_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.gate_proj_t)?;
    let up_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.up_proj_t)?;
    let down_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.down_proj_t)?;
    let in_norm = backend.cached_f32_weight_buffer_kt(&layer.input_layernorm)?;
    let post_norm = backend.cached_f32_weight_buffer_kt(&layer.post_attention_layernorm)?;
    let q_norm = backend.cached_f32_weight_buffer_kt(&attn.q_norm)?;
    let k_norm = backend.cached_f32_weight_buffer_kt(&attn.k_norm)?;

    let normed =
        backend.acquire_resident_scratch("nfa_b_normed", (batch_size * hidden * 4) as u64)?;
    let q_buf = backend.acquire_resident_scratch("nfa_b_q", (batch_size * q_h_d * 4) as u64)?;
    let gate_buf =
        backend.acquire_resident_scratch("nfa_b_gate", (batch_size * q_h_d * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("nfa_b_k", (batch_size * k_dim * 4) as u64)?;
    let v_buf = backend.acquire_resident_scratch("nfa_b_v", (batch_size * v_dim * 4) as u64)?;
    let q_rot = backend.acquire_resident_scratch("nfa_b_q_rot", (batch_size * q_h_d * 4) as u64)?;
    let attn_pre_gate =
        backend.acquire_resident_scratch("nfa_b_attn_pre_gate", (batch_size * q_h_d * 4) as u64)?;
    let attn_post_gate = backend
        .acquire_resident_scratch("nfa_b_attn_post_gate", (batch_size * q_h_d * 4) as u64)?;
    let attn_out =
        backend.acquire_resident_scratch("nfa_b_attn_out", (batch_size * hidden * 4) as u64)?;
    let attn_residual = backend
        .acquire_resident_scratch("nfa_b_attn_residual", (batch_size * hidden * 4) as u64)?;
    let normed_post =
        backend.acquire_resident_scratch("nfa_b_normed_post", (batch_size * hidden * 4) as u64)?;
    let mlp_scratch = backend
        .acquire_resident_scratch("nfa_b_mlp_scratch", (batch_size * intermediate * 4) as u64)?;

    let k_pool = vk_kv_cache
        .k_buffer(full_attn_layer_idx)
        .ok_or_else(|| anyhow::anyhow!("VkPagedKvCache missing layer {full_attn_layer_idx}"))?;
    let v_pool = vk_kv_cache
        .v_buffer(full_attn_layer_idx)
        .ok_or_else(|| anyhow::anyhow!("VkPagedKvCache missing layer {full_attn_layer_idx}"))?;
    let elements_per_slot = num_kv_heads * head_dim;

    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[x_in_buf.handle(), in_norm.handle(), normed.handle()],
        &[batch_size as u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(batch_size as u32),
    )?;
    let (qkv_shader, qkv_workgroups) =
        full_attn_qkv_gate_split_bf16w_plan(batch_size, total_qkv_out);
    batch.record_shader(
        qkv_shader,
        &[
            normed.handle(),
            q_w.handle(),
            k_w.handle(),
            v_w.handle(),
            q_buf.handle(),
            gate_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
        ],
        &[
            hidden as u32,
            q_dim as u32,
            k_dim as u32,
            v_dim as u32,
            total_qkv_out as u32,
            batch_size as u32,
            head_dim as u32,
        ],
        Workgroups::OneD(qkv_workgroups),
    )?;
    batch.record_shader(
        shaders::QWEN_RMSNORM_QK_COMBINED,
        &[
            q_buf.handle(),
            q_norm.handle(),
            k_buf.handle(),
            k_norm.handle(),
        ],
        &[
            (batch_size * num_heads) as u32,
            (batch_size * num_kv_heads) as u32,
            head_dim as u32,
            eps.to_bits(),
        ],
        Workgroups::OneD((batch_size * (num_heads + num_kv_heads)) as u32),
    )?;
    batch.record_shader(
        shaders::VK_ROPE_Q_KV_WRITE_SLOTS_F32,
        &[
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
            rope_cos_buf.handle(),
            rope_sin_buf.handle(),
            slots_buf.handle(),
            q_rot.handle(),
            k_pool.handle(),
            v_pool.handle(),
        ],
        &[
            batch_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            rotary_dim as u32,
            vk_kv_cache.total_slots() as u32,
        ],
        Workgroups::OneD((batch_size * (q_h_d + elements_per_slot)).div_ceil(256) as u32),
    )?;

    let num_chunks = paged_attn_splitk_chunks(batch_size, max_blocks_per_seq);
    let partials_stride = 2 + head_dim;
    let partials_bytes = (batch_size * num_heads * num_chunks * partials_stride * 4) as u64;
    let attn_partials = backend.acquire_resident_scratch("nfa_b_attn_partials", partials_bytes)?;
    kiln_vulkan_kernel::resident::record_paged_attn_decode_batch_paged_splitk_resident(
        batch,
        q_rot.as_ref(),
        k_pool,
        v_pool,
        block_table_buf,
        seq_lens_buf,
        &attn_partials,
        attn_pre_gate.as_ref(),
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_blocks_per_seq,
        block_size,
        softmax_scale,
        num_chunks,
    )?;
    batch.record_shader(
        shaders::VK_MUL_SIGMOID_GATE_F32,
        &[
            attn_pre_gate.handle(),
            gate_buf.handle(),
            attn_post_gate.handle(),
        ],
        &[(batch_size * q_h_d) as u32],
        Workgroups::OneD((batch_size * q_h_d).div_ceil(256) as u32),
    )?;
    let (attn_out_shader, attn_out_workgroups) = linear_bf16w_batched_plan(batch_size, hidden);
    batch.record_shader(
        attn_out_shader,
        &[attn_post_gate.handle(), o_w.handle(), attn_out.handle()],
        &[q_h_d as u32, hidden as u32, batch_size as u32],
        Workgroups::OneD(attn_out_workgroups),
    )?;
    batch.record_shader(
        shaders::ADD_QWEN_RMSNORM_BATCHED,
        &[
            x_in_buf.handle(),
            attn_out.handle(),
            post_norm.handle(),
            attn_residual.handle(),
            normed_post.handle(),
        ],
        &[batch_size as u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(batch_size as u32),
    )?;
    let (mlp_gate_up_shader, mlp_gate_up_workgroups) =
        mlp_gate_up_bf16w_batched_plan(batch_size, intermediate);
    batch.record_shader(
        mlp_gate_up_shader,
        &[
            normed_post.handle(),
            gate_w.handle(),
            up_w.handle(),
            mlp_scratch.handle(),
        ],
        &[hidden as u32, intermediate as u32, batch_size as u32],
        Workgroups::OneD(mlp_gate_up_workgroups),
    )?;
    let (mlp_down_shader, mlp_down_workgroups) =
        mlp_down_add_residual_bf16w_batched_plan(batch_size, hidden);
    batch.record_shader(
        mlp_down_shader,
        &[
            mlp_scratch.handle(),
            down_w.handle(),
            attn_residual.handle(),
            x_out_buf.handle(),
        ],
        &[intermediate as u32, hidden as u32, batch_size as u32],
        Workgroups::OneD(mlp_down_workgroups),
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
    recurrent_state_t: &kiln_tensor::Tensor,
    conv_state_t: &kiln_tensor::Tensor,
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
    let qkv_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_qkv_t)?;
    let z_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_z_t)?;
    let a_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_a_t)?;
    let b_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_b_t)?;
    let out_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.out_proj_t)?;
    let conv_w = backend.cached_f32_weight_buffer_kt(&lin_weights.conv1d)?;
    let qk_norm = backend.cached_f32_weight_buffer_kt(&lin_weights.norm)?;
    let a_log = backend.cached_f32_weight_buffer_kt(&lin_weights.a_log)?;
    let dt_bias = backend.cached_f32_weight_buffer_kt(&lin_weights.dt_bias)?;
    let gate_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.gate_proj_t)?;
    let up_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.up_proj_t)?;
    let down_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.down_proj_t)?;
    let in_norm = backend.cached_f32_weight_buffer_kt(&layer.input_layernorm)?;
    let post_norm = backend.cached_f32_weight_buffer_kt(&layer.post_attention_layernorm)?;

    // Persistent state (per-state-key on backend)
    let recurrent_bytes = (1 * nv * dk * dv * 4) as u64;
    let recurrent_buf =
        backend.linear_attn_recurrent_state_buffer_kt(state_key, recurrent_bytes)?;
    let conv_state_bytes = (1 * qkv_dim * (conv_kernel.saturating_sub(1)) * 4) as u64;
    let conv_buf = backend.linear_attn_conv_state_buffer_kt(state_key, conv_state_bytes)?;

    // Seed states on first call per layer per session (must happen
    // BEFORE batch records reads from these buffers).
    if !backend.linear_attn_layer_seeded_kt(state_key) {
        let Some(vk_device) = backend.vulkan_device() else {
            return Ok(false);
        };
        seed_recurrent_state_kt(vk_device, &recurrent_buf, recurrent_state_t)?;
        seed_conv_state_kt(vk_device, &conv_buf, conv_state_t)?;
        backend.mark_linear_attn_layer_seeded_kt(state_key);
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
    // GQA-expanded Q/K for the recurrent kernel (sized nv*dk, not nk*dk).
    let qkv_expanded_bytes = (nv * dk * 4) as u64;
    let q_expanded = backend.acquire_resident_scratch("ngd_q_expanded", qkv_expanded_bytes)?;
    let k_expanded = backend.acquire_resident_scratch("ngd_k_expanded", qkv_expanded_bytes)?;
    let v_buf = backend.acquire_resident_scratch("ngd_v", (v_dim * 4) as u64)?;
    let gated_norm = backend.acquire_resident_scratch("ngd_gated_norm", (v_dim * 4) as u64)?;
    let gdn_out = backend.acquire_resident_scratch("ngd_gdn_out", (hidden * 4) as u64)?;
    let attn_residual =
        backend.acquire_resident_scratch("ngd_attn_residual", (hidden * 4) as u64)?;
    let normed_post = backend.acquire_resident_scratch("ngd_normed_post", (hidden * 4) as u64)?;
    let mlp_scratch =
        backend.acquire_resident_scratch("ngd_mlp_scratch", (intermediate * 4) as u64)?;

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
    record_causal_conv1d_update_into(
        batch,
        &mixed_qkv,
        &conv_w,
        &conv_buf,
        &conv_qkv,
        1,
        qkv_dim,
        1,
        conv_kernel,
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
    // Q-norm: per-head L2 normalize + GQA expansion, scale = 1/sqrt(dk).
    let gqa_ratio = nv / nk;
    debug_assert!(gqa_ratio * nk == nv, "GDN nv must be a multiple of nk");
    let l2_eps_qk: f32 = 1e-6;
    let q_scale: f32 = 1.0 / (dk as f32).sqrt();
    let k_scale: f32 = 1.0;
    batch.record_shader(
        shaders::L2_NORM_QK_PER_ROW,
        &[
            q_buf.handle(),
            k_buf.handle(),
            q_expanded.handle(),
            k_expanded.handle(),
        ],
        &[
            nk as u32,
            dk as u32,
            l2_eps_qk.to_bits(),
            q_scale.to_bits(),
            k_scale.to_bits(),
            gqa_ratio as u32,
        ],
        Workgroups::OneD(nk as u32),
    )?;
    batch.record_shader(
        shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
        &[
            q_expanded.handle(),
            k_expanded.handle(),
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
        shaders::ADD_QWEN_RMSNORM,
        &[
            x_in_buf.handle(),
            gdn_out.handle(),
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
    // MLP down fused with the final residual ADD: writes
    // `x_out_buf[c] = sum_k mlp_scratch[k] * down_w[k,c] + attn_residual[c]`.
    // Saves one dispatch + one compute->compute barrier per block.
    batch.record_shader(
        shaders::LINEAR_DECODE_BF16W_ADD_RESIDUAL,
        &[
            mlp_scratch.handle(),
            down_w.handle(),
            attn_residual.handle(),
            x_out_buf.handle(),
        ],
        &[intermediate as u32, hidden as u32],
        Workgroups::OneD(hidden.div_ceil(16) as u32),
    )?;
    Ok(true)
}

/// Record one GDN transformer block for `batch_size` independent
/// single-token decode rows. Inputs and outputs are f32 resident
/// buffers laid out `[batch_size, hidden]`; the recurrent and conv
/// state tensors must already represent the same batch geometry.
#[allow(clippy::too_many_arguments)]
pub fn record_gdn_block_batched_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    x_in_buf: &VulkanBuffer,
    x_out_buf: &VulkanBuffer,
    layer: &GpuLayerWeights,
    config: &ModelConfig,
    batch_size: usize,
    recurrent_state_t: &kiln_tensor::Tensor,
    conv_state_t: &kiln_tensor::Tensor,
) -> Result<bool> {
    let lin_weights = match &layer.attention {
        crate::forward::GpuAttentionWeights::Linear(w) => w,
        _ => return Ok(false),
    };
    anyhow::ensure!(batch_size > 0, "batched GDN block: batch_size must be > 0");

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

    let qkv_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_qkv_t)?;
    let z_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_z_t)?;
    let a_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_a_t)?;
    let b_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.in_proj_b_t)?;
    let out_w = backend.cached_bf16_packed_weight_buffer_kt(&lin_weights.out_proj_t)?;
    let conv_w = backend.cached_f32_weight_buffer_kt(&lin_weights.conv1d)?;
    let qk_norm = backend.cached_f32_weight_buffer_kt(&lin_weights.norm)?;
    let a_log = backend.cached_f32_weight_buffer_kt(&lin_weights.a_log)?;
    let dt_bias = backend.cached_f32_weight_buffer_kt(&lin_weights.dt_bias)?;
    let gate_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.gate_proj_t)?;
    let up_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.up_proj_t)?;
    let down_w = backend.cached_bf16_packed_weight_buffer_kt(&layer.mlp.down_proj_t)?;
    let in_norm = backend.cached_f32_weight_buffer_kt(&layer.input_layernorm)?;
    let post_norm = backend.cached_f32_weight_buffer_kt(&layer.post_attention_layernorm)?;

    let recurrent_bytes = (batch_size * nv * dk * dv * 4) as u64;
    let recurrent_buf =
        backend.linear_attn_recurrent_state_buffer_kt(state_key, recurrent_bytes)?;
    let conv_state_bytes = (batch_size * qkv_dim * (conv_kernel.saturating_sub(1)) * 4) as u64;
    let conv_buf = backend.linear_attn_conv_state_buffer_kt(state_key, conv_state_bytes)?;

    if !backend.linear_attn_layer_seeded_kt(state_key) {
        let Some(vk_device) = backend.vulkan_device() else {
            return Ok(false);
        };
        seed_recurrent_state_kt(vk_device, &recurrent_buf, recurrent_state_t)?;
        seed_conv_state_kt(vk_device, &conv_buf, conv_state_t)?;
        backend.mark_linear_attn_layer_seeded_kt(state_key);
    }

    let (gdn_in_proj_shader, gdn_in_proj_workgroups) =
        gdn_in_proj_bf16w_batched_plan(batch_size, qkv_dim, z_dim, a_dim, b_dim, in_proj_total);
    let fuse_gdn_in_proj_conv_split = gdn_in_proj_shader
        == shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W
        && vulkan_kernel_policy().gdn_in_proj_conv_split_fusion;
    let gqa_ratio = nv / nk;
    debug_assert!(gqa_ratio * nk == nv, "GDN nv must be a multiple of nk");
    let fuse_qk_norm_recurrent =
        gdn_qk_norm_recurrent_fusion_enabled(batch_size, gqa_ratio, dk, dv);

    let normed_pre =
        backend.acquire_resident_scratch("ngd_b_normed_pre", (batch_size * hidden * 4) as u64)?;
    let in_proj_out = if fuse_gdn_in_proj_conv_split {
        None
    } else {
        Some(backend.acquire_resident_scratch(
            "ngd_b_in_proj_out",
            (batch_size * in_proj_total * 4) as u64,
        )?)
    };
    let z_buf = backend.acquire_resident_scratch("ngd_b_z", (batch_size * z_dim * 4) as u64)?;
    let a_buf = backend.acquire_resident_scratch("ngd_b_a", (batch_size * a_dim * 4) as u64)?;
    let b_buf = backend.acquire_resident_scratch("ngd_b_b", (batch_size * b_dim * 4) as u64)?;
    let q_buf = backend.acquire_resident_scratch("ngd_b_q", (batch_size * qk_dim * 4) as u64)?;
    let k_buf = backend.acquire_resident_scratch("ngd_b_k", (batch_size * qk_dim * 4) as u64)?;
    let q_expanded = if fuse_qk_norm_recurrent {
        None
    } else {
        Some(
            backend
                .acquire_resident_scratch("ngd_b_q_expanded", (batch_size * nv * dk * 4) as u64)?,
        )
    };
    let k_expanded = if fuse_qk_norm_recurrent {
        None
    } else {
        Some(
            backend
                .acquire_resident_scratch("ngd_b_k_expanded", (batch_size * nv * dk * 4) as u64)?,
        )
    };
    let v_buf = backend.acquire_resident_scratch("ngd_b_v", (batch_size * v_dim * 4) as u64)?;
    let gated_norm =
        backend.acquire_resident_scratch("ngd_b_gated_norm", (batch_size * v_dim * 4) as u64)?;
    let gdn_out =
        backend.acquire_resident_scratch("ngd_b_gdn_out", (batch_size * hidden * 4) as u64)?;
    let attn_residual = backend
        .acquire_resident_scratch("ngd_b_attn_residual", (batch_size * hidden * 4) as u64)?;
    let normed_post =
        backend.acquire_resident_scratch("ngd_b_normed_post", (batch_size * hidden * 4) as u64)?;
    let mlp_scratch = backend
        .acquire_resident_scratch("ngd_b_mlp_scratch", (batch_size * intermediate * 4) as u64)?;

    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[x_in_buf.handle(), in_norm.handle(), normed_pre.handle()],
        &[batch_size as u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(batch_size as u32),
    )?;
    if fuse_gdn_in_proj_conv_split {
        batch.record_shader(
            shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W_CONV_SPLIT,
            &[
                normed_pre.handle(),
                qkv_w.handle(),
                z_w.handle(),
                a_w.handle(),
                b_w.handle(),
                conv_w.handle(),
                conv_buf.handle(),
                q_buf.handle(),
                k_buf.handle(),
                v_buf.handle(),
                z_buf.handle(),
                a_buf.handle(),
                b_buf.handle(),
            ],
            &[
                hidden as u32,
                qkv_dim as u32,
                z_dim as u32,
                a_dim as u32,
                b_dim as u32,
                in_proj_total as u32,
                batch_size as u32,
                qk_dim as u32,
                v_dim as u32,
                conv_kernel as u32,
            ],
            Workgroups::OneD(gdn_in_proj_workgroups),
        )?;
    } else {
        let in_proj_out = in_proj_out
            .as_ref()
            .context("batched GDN block missing in-proj scratch")?;
        batch.record_shader(
            gdn_in_proj_shader,
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
                batch_size as u32,
            ],
            Workgroups::OneD(gdn_in_proj_workgroups),
        )?;
        batch.record_shader(
            shaders::GDN_DECODE_CONV_SPLIT_BATCHED,
            &[
                in_proj_out.handle(),
                conv_w.handle(),
                conv_buf.handle(),
                q_buf.handle(),
                k_buf.handle(),
                v_buf.handle(),
                z_buf.handle(),
                a_buf.handle(),
                b_buf.handle(),
            ],
            &[
                batch_size as u32,
                qkv_dim as u32,
                qk_dim as u32,
                v_dim as u32,
                z_dim as u32,
                a_dim as u32,
                b_dim as u32,
                conv_kernel as u32,
            ],
            Workgroups::OneD((batch_size * in_proj_total).div_ceil(256) as u32),
        )?;
    }

    let l2_eps_qk: f32 = 1e-6;
    let q_scale: f32 = 1.0 / (dk as f32).sqrt();
    let k_scale: f32 = 1.0;
    if fuse_qk_norm_recurrent {
        batch.record_shader(
            shaders::GDN_DECODE_QK_NORM_GATES_RECURRENT_RMSNORM,
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
                nk as u32,
                dk as u32,
                dv as u32,
                eps.to_bits(),
                batch_size as u32,
                gqa_ratio as u32,
                l2_eps_qk.to_bits(),
                q_scale.to_bits(),
                k_scale.to_bits(),
            ],
            Workgroups::OneD((batch_size * nk) as u32),
        )?;
    } else {
        let q_expanded = q_expanded
            .as_ref()
            .context("batched GDN block missing expanded Q scratch")?;
        let k_expanded = k_expanded
            .as_ref()
            .context("batched GDN block missing expanded K scratch")?;
        batch.record_shader(
            shaders::L2_NORM_QK_PER_ROW,
            &[
                q_buf.handle(),
                k_buf.handle(),
                q_expanded.handle(),
                k_expanded.handle(),
            ],
            &[
                (batch_size * nk) as u32,
                dk as u32,
                l2_eps_qk.to_bits(),
                q_scale.to_bits(),
                k_scale.to_bits(),
                gqa_ratio as u32,
            ],
            Workgroups::OneD((batch_size * nk) as u32),
        )?;
        batch.record_shader(
            shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
            &[
                q_expanded.handle(),
                k_expanded.handle(),
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
                batch_size as u32,
            ],
            Workgroups::OneD((batch_size * nv) as u32),
        )?;
    }
    let (gdn_out_shader, gdn_out_workgroups) = linear_bf16w_batched_plan(batch_size, hidden);
    batch.record_shader(
        gdn_out_shader,
        &[gated_norm.handle(), out_w.handle(), gdn_out.handle()],
        &[v_dim as u32, hidden as u32, batch_size as u32],
        Workgroups::OneD(gdn_out_workgroups),
    )?;
    batch.record_shader(
        shaders::ADD_QWEN_RMSNORM_BATCHED,
        &[
            x_in_buf.handle(),
            gdn_out.handle(),
            post_norm.handle(),
            attn_residual.handle(),
            normed_post.handle(),
        ],
        &[batch_size as u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(batch_size as u32),
    )?;
    let (mlp_gate_up_shader, mlp_gate_up_workgroups) =
        mlp_gate_up_bf16w_batched_plan(batch_size, intermediate);
    batch.record_shader(
        mlp_gate_up_shader,
        &[
            normed_post.handle(),
            gate_w.handle(),
            up_w.handle(),
            mlp_scratch.handle(),
        ],
        &[hidden as u32, intermediate as u32, batch_size as u32],
        Workgroups::OneD(mlp_gate_up_workgroups),
    )?;
    let (mlp_down_shader, mlp_down_workgroups) =
        mlp_down_add_residual_bf16w_batched_plan(batch_size, hidden);
    batch.record_shader(
        mlp_down_shader,
        &[
            mlp_scratch.handle(),
            down_w.handle(),
            attn_residual.handle(),
            x_out_buf.handle(),
        ],
        &[intermediate as u32, hidden as u32, batch_size as u32],
        Workgroups::OneD(mlp_down_workgroups),
    )?;
    Ok(true)
}

/// Record the batched final RMSNorm plus LM-head argmax stage.
///
/// `hidden_in_buf` is f32 `[batch_size, hidden]`; `out_token_buf` is
/// u32 `[batch_size]`. This avoids writing `[batch_size, vocab]`
/// logits when the caller only needs greedy next-token IDs.
#[allow(clippy::too_many_arguments)]
pub fn record_final_norm_lm_head_argmax_batched_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    hidden_in_buf: &VulkanBuffer,
    out_token_buf: &VulkanBuffer,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
) -> Result<bool> {
    anyhow::ensure!(
        batch_size > 0,
        "batched final argmax: batch_size must be > 0"
    );
    let hidden = config.hidden_size;
    let vocab_size = weights.embed_tokens_t.dims().last().copied().unwrap_or(0);
    if vocab_size == 0 {
        return Ok(false);
    }
    let eps = config.rms_norm_eps as f32;
    let final_norm = backend.cached_f32_weight_buffer_kt(&weights.final_norm)?;
    let lm_head_w = backend.cached_bf16_packed_weight_buffer_kt(&weights.embed_tokens_t)?;
    let normed = backend
        .acquire_resident_scratch("native_b_final_normed", (batch_size * hidden * 4) as u64)?;
    let block_count = vocab_size.div_ceil(64);
    let block_scores = backend.acquire_resident_scratch(
        "native_b_argmax_scores",
        (batch_size * block_count * 4) as u64,
    )?;
    let block_indices = backend.acquire_resident_scratch(
        "native_b_argmax_indices",
        (batch_size * block_count * 4) as u64,
    )?;

    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[hidden_in_buf.handle(), final_norm.handle(), normed.handle()],
        &[batch_size as u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(batch_size as u32),
    )?;
    let rows8 = batch_size >= linear_bf16_rows8_min_batch() && linear_bf16w_rows8_enabled();
    let rows4 = batch_size >= linear_bf16_rows4_min_batch() && !rows8;
    let block_shader = if rows8 {
        shaders::LINEAR_DECODE_ARGMAX_BATCHED_BLOCKS_ROWS8_BF16W
    } else if rows4 {
        shaders::LINEAR_DECODE_ARGMAX_BATCHED_BLOCKS_ROWS4_BF16W
    } else {
        shaders::LINEAR_DECODE_ARGMAX_BATCHED_BLOCKS_BF16W
    };
    let block_push = if rows8 || rows4 {
        vec![
            hidden as u32,
            vocab_size as u32,
            block_count as u32,
            batch_size as u32,
        ]
    } else {
        vec![hidden as u32, vocab_size as u32, block_count as u32]
    };
    let block_workgroups = if rows8 {
        batch_size.div_ceil(8) * block_count
    } else if rows4 {
        batch_size.div_ceil(4) * block_count
    } else {
        batch_size * block_count
    };
    batch.record_shader(
        block_shader,
        &[
            normed.handle(),
            lm_head_w.handle(),
            block_scores.handle(),
            block_indices.handle(),
        ],
        &block_push,
        Workgroups::OneD(block_workgroups as u32),
    )?;
    batch.record_shader(
        shaders::LINEAR_DECODE_ARGMAX_BATCHED_REDUCE,
        &[
            block_scores.handle(),
            block_indices.handle(),
            out_token_buf.handle(),
        ],
        &[block_count as u32],
        Workgroups::OneD(batch_size as u32),
    )?;
    Ok(true)
}

/// Record the batched final RMSNorm plus LM-head stochastic sampler stage.
///
/// `hidden_in_buf` is f32 `[batch_size, hidden]`; `out_token_buf` is u32
/// `[batch_size]`. Unlike the hidden-returning path, this keeps full-vocab
/// logits resident and reads back only one token per row.
#[allow(clippy::too_many_arguments)]
pub fn record_final_norm_lm_head_sample_batched_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    hidden_in_buf: &VulkanBuffer,
    out_token_buf: &VulkanBuffer,
    sample: &BatchedResidentSampleBuffers,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
) -> Result<bool> {
    anyhow::ensure!(
        batch_size > 0 && sample.batch_size == batch_size,
        "batched final sample: batch size mismatch"
    );
    let hidden = config.hidden_size;
    let vocab_size = weights.embed_tokens_t.dims().last().copied().unwrap_or(0);
    if vocab_size == 0 {
        return Ok(false);
    }
    let eps = config.rms_norm_eps as f32;
    let final_norm = backend.cached_f32_weight_buffer_kt(&weights.final_norm)?;
    let lm_head_w = backend.cached_bf16_packed_weight_buffer_kt(&weights.embed_tokens_t)?;
    let normed = backend
        .acquire_resident_scratch("native_b_sample_normed", (batch_size * hidden * 4) as u64)?;
    let logits = backend.acquire_resident_scratch(
        "native_b_sample_logits",
        (batch_size * vocab_size * 4) as u64,
    )?;

    batch.record_shader(
        shaders::QWEN_RMSNORM_FORWARD,
        &[hidden_in_buf.handle(), final_norm.handle(), normed.handle()],
        &[batch_size as u32, hidden as u32, eps.to_bits()],
        Workgroups::OneD(batch_size as u32),
    )?;
    let rows8 = batch_size >= linear_bf16_rows8_min_batch() && linear_bf16w_rows8_enabled();
    let rows4 = batch_size >= linear_bf16_rows4_min_batch() && !rows8;
    let lm_shader = if rows8 {
        shaders::LINEAR_DECODE_BATCHED_ROWS8_BF16W
    } else if rows4 {
        shaders::LINEAR_DECODE_BATCHED_ROWS4_BF16W
    } else {
        shaders::LINEAR_DECODE_BATCHED_BF16W
    };
    let row_groups = if rows8 {
        batch_size.div_ceil(8)
    } else if rows4 {
        batch_size.div_ceil(4)
    } else {
        batch_size
    };
    batch.record_shader(
        lm_shader,
        &[normed.handle(), lm_head_w.handle(), logits.handle()],
        &[hidden as u32, vocab_size as u32, batch_size as u32],
        Workgroups::OneD((row_groups * vocab_size.div_ceil(32)) as u32),
    )?;

    if sample.history_items > 0 {
        let rows = sample
            .history_rows
            .as_ref()
            .context("batched final sample: missing history rows buffer")?;
        let indices = sample
            .history_indices
            .as_ref()
            .context("batched final sample: missing history indices buffer")?;
        let counts = sample
            .history_counts
            .as_ref()
            .context("batched final sample: missing history counts buffer")?;
        let repetitions = sample
            .repetitions
            .as_ref()
            .context("batched final sample: missing repetitions buffer")?;
        let presences = sample
            .presences
            .as_ref()
            .context("batched final sample: missing presences buffer")?;
        let frequencies = sample
            .frequencies
            .as_ref()
            .context("batched final sample: missing frequencies buffer")?;
        batch.record_shader(
            shaders::APPLY_TOKEN_PENALTIES_BATCHED,
            &[
                logits.handle(),
                indices.handle(),
                counts.handle(),
                rows.handle(),
                repetitions.handle(),
                presences.handle(),
                frequencies.handle(),
            ],
            &[
                sample.history_items as u32,
                vocab_size as u32,
                batch_size as u32,
            ],
            Workgroups::OneD((sample.history_items as u32).div_ceil(64)),
        )?;
    }

    batch.record_shader(
        shaders::TOPK_SAMPLE_BATCHED,
        &[
            logits.handle(),
            out_token_buf.handle(),
            sample.top_k.handle(),
            sample.temperatures.handle(),
            sample.top_p.handle(),
            sample.min_p.handle(),
            sample.seed_lo.handle(),
            sample.seed_hi.handle(),
        ],
        &[vocab_size as u32, batch_size as u32],
        Workgroups::ThreeD(1, batch_size as u32, 1),
    )?;
    Ok(true)
}

/// Record a full batched decode stack into an existing [`CommandBatch`].
///
/// This is intentionally record-only: the caller owns input upload,
/// per-row paged metadata buffers, full-attention KV seeding, command
/// submission, and output readback. It composes the batched
/// full-attention/GDN block recorders and returns whether the final
/// `[batch_size, hidden]` rows are in `x_in_buf` (`true`) or
/// `x_scratch_buf` (`false`).
#[allow(clippy::too_many_arguments)]
pub fn record_transformer_stack_batched_hidden_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    x_in_buf: &VulkanBuffer,
    x_scratch_buf: &VulkanBuffer,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
    max_blocks_per_seq: usize,
    block_size: usize,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<bool>> {
    anyhow::ensure!(
        batch_size > 0,
        "batched transformer stack: batch_size must be > 0"
    );

    let mut full_attn_layer_idx = 0usize;
    let mut linear_attn_idx = 0usize;
    let mut from_buf = x_in_buf;
    let mut to_buf = x_scratch_buf;
    let mut from_is_input = true;
    for layer in weights.layers.iter() {
        match &layer.attention {
            crate::forward::GpuAttentionWeights::Full(_) => {
                let ok = record_full_attn_block_batched_into(
                    backend,
                    batch,
                    from_buf,
                    to_buf,
                    layer,
                    config,
                    batch_size,
                    max_blocks_per_seq,
                    block_size,
                    full_attn_layer_idx,
                    vk_kv_cache,
                    rope_cos_buf,
                    rope_sin_buf,
                    block_table_buf,
                    seq_lens_buf,
                    slots_buf,
                )?;
                if !ok {
                    return Ok(None);
                }
                full_attn_layer_idx += 1;
            }
            crate::forward::GpuAttentionWeights::Linear(_) => {
                let Some(recurrent_t) = recurrent_states.get(linear_attn_idx) else {
                    return Ok(None);
                };
                let Some(conv_t) = conv_states.get(linear_attn_idx) else {
                    return Ok(None);
                };
                let ok = record_gdn_block_batched_into(
                    backend,
                    batch,
                    from_buf,
                    to_buf,
                    layer,
                    config,
                    batch_size,
                    recurrent_t,
                    conv_t,
                )?;
                if !ok {
                    return Ok(None);
                }
                linear_attn_idx += 1;
            }
        }
        std::mem::swap(&mut from_buf, &mut to_buf);
        from_is_input = !from_is_input;
    }

    Ok(Some(from_is_input))
}

/// Record token embedding gather plus the full batched decode stack
/// into an existing [`CommandBatch`].
#[allow(clippy::too_many_arguments)]
pub fn record_transformer_stack_batched_hidden_from_tokens_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    token_ids: &[u32],
    token_ids_buf: &VulkanBuffer,
    x_in_buf: &VulkanBuffer,
    x_scratch_buf: &VulkanBuffer,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
    max_blocks_per_seq: usize,
    block_size: usize,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<bool>> {
    anyhow::ensure!(
        token_ids.len() == batch_size,
        "batched transformer token stack: token id count mismatch"
    );
    let Some(embed) = resident_decode_embedding_source(backend, weights, config.hidden_size)?
    else {
        return Ok(None);
    };
    ensure_resident_decode_embedding_ids(token_ids, embed.vocab)?;
    if !record_resident_decode_rope_tables_into(
        backend,
        batch,
        weights,
        seq_lens_buf,
        rope_cos_buf,
        rope_sin_buf,
        batch_size,
        config.rotary_dim(),
    )? {
        return Ok(None);
    }
    record_resident_decode_embedding_independent_into(
        batch,
        &embed,
        token_ids_buf,
        x_in_buf,
        batch_size,
    )?;
    record_transformer_stack_batched_hidden_into(
        backend,
        batch,
        x_in_buf,
        x_scratch_buf,
        weights,
        config,
        batch_size,
        max_blocks_per_seq,
        block_size,
        vk_kv_cache,
        rope_cos_buf,
        rope_sin_buf,
        block_table_buf,
        seq_lens_buf,
        slots_buf,
        recurrent_states,
        conv_states,
    )
}

/// Record a full batched decode stack plus final LM-head argmax into
/// an existing [`CommandBatch`].
///
/// This is intentionally record-only: the caller owns input upload,
/// per-row paged metadata buffers, full-attention KV seeding, command
/// submission, and output-token readback. It composes the batched
/// full-attention/GDN block recorders and then writes u32 token IDs to
/// `out_token_buf`.
#[allow(clippy::too_many_arguments)]
pub fn record_transformer_stack_batched_argmax_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    x_in_buf: &VulkanBuffer,
    x_scratch_buf: &VulkanBuffer,
    out_token_buf: &VulkanBuffer,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
    max_blocks_per_seq: usize,
    block_size: usize,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<bool> {
    let Some(final_in_input) = record_transformer_stack_batched_hidden_into(
        backend,
        batch,
        x_in_buf,
        x_scratch_buf,
        weights,
        config,
        batch_size,
        max_blocks_per_seq,
        block_size,
        vk_kv_cache,
        rope_cos_buf,
        rope_sin_buf,
        block_table_buf,
        seq_lens_buf,
        slots_buf,
        recurrent_states,
        conv_states,
    )?
    else {
        return Ok(false);
    };
    let hidden_buf = if final_in_input {
        x_in_buf
    } else {
        x_scratch_buf
    };
    record_final_norm_lm_head_argmax_batched_into(
        backend,
        batch,
        hidden_buf,
        out_token_buf,
        weights,
        config,
        batch_size,
    )
}

/// Record a full batched decode stack plus final resident stochastic sampler
/// into an existing [`CommandBatch`].
#[allow(clippy::too_many_arguments)]
pub fn record_transformer_stack_batched_sample_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    x_in_buf: &VulkanBuffer,
    x_scratch_buf: &VulkanBuffer,
    out_token_buf: &VulkanBuffer,
    sample: &BatchedResidentSampleBuffers,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
    max_blocks_per_seq: usize,
    block_size: usize,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<bool> {
    let Some(final_in_input) = record_transformer_stack_batched_hidden_into(
        backend,
        batch,
        x_in_buf,
        x_scratch_buf,
        weights,
        config,
        batch_size,
        max_blocks_per_seq,
        block_size,
        vk_kv_cache,
        rope_cos_buf,
        rope_sin_buf,
        block_table_buf,
        seq_lens_buf,
        slots_buf,
        recurrent_states,
        conv_states,
    )?
    else {
        return Ok(false);
    };
    let hidden_buf = if final_in_input {
        x_in_buf
    } else {
        x_scratch_buf
    };
    record_final_norm_lm_head_sample_batched_into(
        backend,
        batch,
        hidden_buf,
        out_token_buf,
        sample,
        weights,
        config,
        batch_size,
    )
}

/// Submit a full batched resident decode stack and return the greedy
/// next-token IDs. The caller still owns all per-step uploads and
/// cache seeding; this helper owns only command-batch construction,
/// token-id staging, submission, and readback.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_argmax(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    x_in_buf: &VulkanBuffer,
    x_scratch_buf: &VulkanBuffer,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
    max_blocks_per_seq: usize,
    block_size: usize,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<Vec<u32>>> {
    anyhow::ensure!(
        batch_size > 0,
        "batched transformer submit: batch_size must be > 0"
    );
    let out_bytes = (batch_size * 4) as u64;
    let out_staging =
        backend.acquire_resident_scratch_host_visible("native_b_out_tokens_staging", out_bytes)?;

    let mut batch = CommandBatch::new(vk_device)?;
    let ok = record_transformer_stack_batched_argmax_into(
        backend,
        &mut batch,
        x_in_buf,
        x_scratch_buf,
        &out_staging,
        weights,
        config,
        batch_size,
        max_blocks_per_seq,
        block_size,
        vk_kv_cache,
        rope_cos_buf,
        rope_sin_buf,
        block_table_buf,
        seq_lens_buf,
        slots_buf,
        recurrent_states,
        conv_states,
    )?;
    if !ok {
        return Ok(None);
    }
    replay_vulkan_command_batch(
        batch,
        "vk-resident native batched decode",
        vec![
            batch_size,
            config.hidden_size,
            max_blocks_per_seq,
            block_size,
        ],
        batch_size,
        batched_decode_replay_resources(
            x_in_buf,
            x_scratch_buf,
            rope_cos_buf,
            rope_sin_buf,
            block_table_buf,
            seq_lens_buf,
            slots_buf,
            &out_staging,
            DType::U32,
            vec![batch_size],
            batch_size,
            config.hidden_size,
            config.rotary_dim(),
            max_blocks_per_seq,
        ),
    )
    .context("batched transformer submit: submit CommandBatch through ReplayPlan")?;

    let bytes = out_staging
        .read_mapped(out_bytes as usize)
        .context("batched transformer submit: read token-id staging")?;
    let mut tokens = Vec::with_capacity(batch_size);
    for chunk in bytes.chunks_exact(4).take(batch_size) {
        tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Some(tokens))
}

/// Submit a full batched resident decode stack and return sampled next-token
/// IDs. This keeps the final hidden rows and full-vocab logits resident; only
/// `[batch_size]` token IDs are read after the single command-batch submit.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_sample(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    x_in_buf: &VulkanBuffer,
    x_scratch_buf: &VulkanBuffer,
    sample: &BatchedResidentSampleBuffers,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
    max_blocks_per_seq: usize,
    block_size: usize,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<Vec<u32>>> {
    anyhow::ensure!(
        batch_size > 0,
        "batched transformer sample submit: batch_size must be > 0"
    );
    let out_bytes = (batch_size * 4) as u64;
    let out_staging = backend
        .acquire_resident_scratch_host_visible("native_b_sample_tokens_staging", out_bytes)?;

    let mut batch = CommandBatch::new(vk_device)?;
    let ok = record_transformer_stack_batched_sample_into(
        backend,
        &mut batch,
        x_in_buf,
        x_scratch_buf,
        &out_staging,
        sample,
        weights,
        config,
        batch_size,
        max_blocks_per_seq,
        block_size,
        vk_kv_cache,
        rope_cos_buf,
        rope_sin_buf,
        block_table_buf,
        seq_lens_buf,
        slots_buf,
        recurrent_states,
        conv_states,
    )?;
    if !ok {
        return Ok(None);
    }
    let mut replay_resources = batched_decode_replay_resources(
        x_in_buf,
        x_scratch_buf,
        rope_cos_buf,
        rope_sin_buf,
        block_table_buf,
        seq_lens_buf,
        slots_buf,
        &out_staging,
        DType::U32,
        vec![batch_size],
        batch_size,
        config.hidden_size,
        config.rotary_dim(),
        max_blocks_per_seq,
    );
    append_sample_replay_resources(&mut replay_resources, sample);
    replay_vulkan_command_batch(
        batch,
        "vk-resident native batched sample decode",
        vec![
            batch_size,
            config.hidden_size,
            max_blocks_per_seq,
            block_size,
        ],
        batch_size,
        replay_resources,
    )
    .context("batched transformer sample submit: submit CommandBatch through ReplayPlan")?;

    let bytes = out_staging
        .read_mapped(out_bytes as usize)
        .context("batched transformer sample submit: read token-id staging")?;
    let mut tokens = Vec::with_capacity(batch_size);
    for chunk in bytes.chunks_exact(4).take(batch_size) {
        tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Some(tokens))
}

/// Submit a full batched resident decode stack and return final hidden
/// rows as f32 `[batch_size, hidden]`. The caller owns embedding, RoPE,
/// metadata preparation, and cache seeding.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_hidden(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    x_in_buf: &VulkanBuffer,
    x_scratch_buf: &VulkanBuffer,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    batch_size: usize,
    max_blocks_per_seq: usize,
    block_size: usize,
    vk_kv_cache: &VkPagedKvCache,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    block_table_buf: &VulkanBuffer,
    seq_lens_buf: &VulkanBuffer,
    slots_buf: &VulkanBuffer,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<Vec<f32>>> {
    anyhow::ensure!(
        batch_size > 0,
        "batched transformer hidden submit: batch_size must be > 0"
    );
    let hidden_bytes = (batch_size * config.hidden_size * 4) as u64;
    let out_staging =
        backend.acquire_resident_scratch_host_visible("native_b_hidden_staging", hidden_bytes)?;

    let mut batch = CommandBatch::new(vk_device)?;
    let Some(final_in_input) = record_transformer_stack_batched_hidden_into(
        backend,
        &mut batch,
        x_in_buf,
        x_scratch_buf,
        weights,
        config,
        batch_size,
        max_blocks_per_seq,
        block_size,
        vk_kv_cache,
        rope_cos_buf,
        rope_sin_buf,
        block_table_buf,
        seq_lens_buf,
        slots_buf,
        recurrent_states,
        conv_states,
    )?
    else {
        return Ok(None);
    };
    let hidden_src = if final_in_input {
        x_in_buf
    } else {
        x_scratch_buf
    };
    batch
        .record_copy_buffer(hidden_src, &out_staging, hidden_bytes)
        .context("batched transformer hidden submit: record hidden readback")?;
    replay_vulkan_command_batch(
        batch,
        "vk-resident native batched hidden decode",
        vec![
            batch_size,
            config.hidden_size,
            max_blocks_per_seq,
            block_size,
        ],
        batch_size,
        batched_decode_replay_resources(
            x_in_buf,
            x_scratch_buf,
            rope_cos_buf,
            rope_sin_buf,
            block_table_buf,
            seq_lens_buf,
            slots_buf,
            &out_staging,
            DType::F32,
            vec![batch_size, config.hidden_size],
            batch_size,
            config.hidden_size,
            config.rotary_dim(),
            max_blocks_per_seq,
        ),
    )
    .context("batched transformer hidden submit: submit CommandBatch through ReplayPlan")?;

    let bytes = out_staging
        .read_mapped(hidden_bytes as usize)
        .context("batched transformer hidden submit: read hidden staging")?;
    let mut hidden = Vec::with_capacity(batch_size * config.hidden_size);
    for chunk in bytes.chunks_exact(4).take(batch_size * config.hidden_size) {
        hidden.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Some(hidden))
}

/// Submit a full batched resident decode stack from token IDs. The
/// first dispatch gathers embedding rows into the resident input buffer,
/// so the hidden input upload is avoided.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_argmax_from_tokens(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    token_ids: &[u32],
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    block_size: usize,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    vk_kv_cache: &VkPagedKvCache,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<Vec<u32>>> {
    let batch_size = block_tables.len();
    anyhow::ensure!(
        token_ids.len() == batch_size,
        "batched transformer token argmax: token id count mismatch"
    );
    let Some(embed) = resident_decode_embedding_source(backend, weights, config.hidden_size)?
    else {
        return Ok(None);
    };
    ensure_resident_decode_embedding_ids(token_ids, embed.vocab)?;
    let step = prepare_batched_resident_decode_token_step_buffers(
        backend,
        token_ids,
        config.hidden_size,
        config.rotary_dim(),
    )?;
    let meta = prepare_batched_resident_decode_meta_buffers(
        backend,
        block_tables,
        start_positions,
        block_size,
    )?;
    let out_bytes = (batch_size * 4) as u64;
    let out_staging =
        backend.acquire_resident_scratch_host_visible("native_b_out_tokens_staging", out_bytes)?;

    let mut batch = CommandBatch::new(vk_device)?;
    if !record_resident_decode_rope_tables_into(
        backend,
        &mut batch,
        weights,
        &meta.seq_lens,
        &step.rope_cos,
        &step.rope_sin,
        batch_size,
        config.rotary_dim(),
    )? {
        return Ok(None);
    }
    record_resident_decode_embedding_independent_into(
        &mut batch,
        &embed,
        &step.token_ids,
        &step.input,
        batch_size,
    )?;
    let ok = record_transformer_stack_batched_argmax_into(
        backend,
        &mut batch,
        &step.input,
        &step.scratch,
        &out_staging,
        weights,
        config,
        batch_size,
        meta.max_blocks_per_seq,
        meta.block_size,
        vk_kv_cache,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        recurrent_states,
        conv_states,
    )?;
    if !ok {
        return Ok(None);
    }
    let mut replay_resources = batched_decode_replay_resources(
        &step.input,
        &step.scratch,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        &out_staging,
        DType::U32,
        vec![batch_size],
        batch_size,
        step.hidden,
        step.rotary_dim,
        meta.max_blocks_per_seq,
    );
    replay_resources.push(vk_replay_resource(
        &step.token_ids,
        DType::U32,
        vec![batch_size],
    ));
    replay_vulkan_command_batch(
        batch,
        "vk-resident native token batched decode",
        vec![
            batch_size,
            config.hidden_size,
            meta.max_blocks_per_seq,
            meta.block_size,
        ],
        batch_size,
        replay_resources,
    )
    .context("batched transformer token argmax: submit CommandBatch through ReplayPlan")?;

    let bytes = out_staging
        .read_mapped(out_bytes as usize)
        .context("batched transformer token argmax: read token-id staging")?;
    let mut tokens = Vec::with_capacity(batch_size);
    for chunk in bytes.chunks_exact(4).take(batch_size) {
        tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Some(tokens))
}

/// Submit a full batched resident decode stack plus sampler from token
/// IDs, keeping embedding, hidden rows, logits, and sampling on Vulkan.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_sample_from_tokens(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    token_ids: &[u32],
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    block_size: usize,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    vk_kv_cache: &VkPagedKvCache,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<Option<Vec<u32>>> {
    let batch_size = block_tables.len();
    anyhow::ensure!(
        token_ids.len() == batch_size,
        "batched transformer token sample: token id count mismatch"
    );
    let Some(embed) = resident_decode_embedding_source(backend, weights, config.hidden_size)?
    else {
        return Ok(None);
    };
    ensure_resident_decode_embedding_ids(token_ids, embed.vocab)?;
    let step = prepare_batched_resident_decode_token_step_buffers(
        backend,
        token_ids,
        config.hidden_size,
        config.rotary_dim(),
    )?;
    let meta = prepare_batched_resident_decode_meta_buffers(
        backend,
        block_tables,
        start_positions,
        block_size,
    )?;
    let sample = prepare_batched_resident_sample_buffers(
        backend,
        batch_size,
        history_rows,
        history_indices,
        history_counts,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
        temperatures,
        top_k,
        top_p,
        min_p,
        seeds,
    )?;
    let out_bytes = (batch_size * 4) as u64;
    let out_staging = backend
        .acquire_resident_scratch_host_visible("native_b_sample_tokens_staging", out_bytes)?;

    let mut batch = CommandBatch::new(vk_device)?;
    if !record_resident_decode_rope_tables_into(
        backend,
        &mut batch,
        weights,
        &meta.seq_lens,
        &step.rope_cos,
        &step.rope_sin,
        batch_size,
        config.rotary_dim(),
    )? {
        return Ok(None);
    }
    record_resident_decode_embedding_independent_into(
        &mut batch,
        &embed,
        &step.token_ids,
        &step.input,
        batch_size,
    )?;
    let ok = record_transformer_stack_batched_sample_into(
        backend,
        &mut batch,
        &step.input,
        &step.scratch,
        &out_staging,
        &sample,
        weights,
        config,
        batch_size,
        meta.max_blocks_per_seq,
        meta.block_size,
        vk_kv_cache,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        recurrent_states,
        conv_states,
    )?;
    if !ok {
        return Ok(None);
    }
    let mut replay_resources = batched_decode_replay_resources(
        &step.input,
        &step.scratch,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        &out_staging,
        DType::U32,
        vec![batch_size],
        batch_size,
        step.hidden,
        step.rotary_dim,
        meta.max_blocks_per_seq,
    );
    replay_resources.push(vk_replay_resource(
        &step.token_ids,
        DType::U32,
        vec![batch_size],
    ));
    append_sample_replay_resources(&mut replay_resources, &sample);
    replay_vulkan_command_batch(
        batch,
        "vk-resident native token batched sample decode",
        vec![
            batch_size,
            config.hidden_size,
            meta.max_blocks_per_seq,
            meta.block_size,
        ],
        batch_size,
        replay_resources,
    )
    .context("batched transformer token sample: submit CommandBatch through ReplayPlan")?;

    let bytes = out_staging
        .read_mapped(out_bytes as usize)
        .context("batched transformer token sample: read token-id staging")?;
    let mut tokens = Vec::with_capacity(batch_size);
    for chunk in bytes.chunks_exact(4).take(batch_size) {
        tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Some(tokens))
}

/// Submit a full batched resident decode stack from token IDs and
/// return final hidden rows as f32 `[batch_size, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_hidden_from_tokens(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    token_ids: &[u32],
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    block_size: usize,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    vk_kv_cache: &VkPagedKvCache,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<Vec<f32>>> {
    let batch_size = block_tables.len();
    anyhow::ensure!(
        token_ids.len() == batch_size,
        "batched transformer token hidden: token id count mismatch"
    );
    let Some(embed) = resident_decode_embedding_source(backend, weights, config.hidden_size)?
    else {
        return Ok(None);
    };
    ensure_resident_decode_embedding_ids(token_ids, embed.vocab)?;
    let step = prepare_batched_resident_decode_token_step_buffers(
        backend,
        token_ids,
        config.hidden_size,
        config.rotary_dim(),
    )?;
    let meta = prepare_batched_resident_decode_meta_buffers(
        backend,
        block_tables,
        start_positions,
        block_size,
    )?;
    let hidden_bytes = (batch_size * config.hidden_size * 4) as u64;
    let out_staging =
        backend.acquire_resident_scratch_host_visible("native_b_hidden_staging", hidden_bytes)?;

    let mut batch = CommandBatch::new(vk_device)?;
    if !record_resident_decode_rope_tables_into(
        backend,
        &mut batch,
        weights,
        &meta.seq_lens,
        &step.rope_cos,
        &step.rope_sin,
        batch_size,
        config.rotary_dim(),
    )? {
        return Ok(None);
    }
    record_resident_decode_embedding_independent_into(
        &mut batch,
        &embed,
        &step.token_ids,
        &step.input,
        batch_size,
    )?;
    let Some(final_in_input) = record_transformer_stack_batched_hidden_into(
        backend,
        &mut batch,
        &step.input,
        &step.scratch,
        weights,
        config,
        batch_size,
        meta.max_blocks_per_seq,
        meta.block_size,
        vk_kv_cache,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        recurrent_states,
        conv_states,
    )?
    else {
        return Ok(None);
    };
    let hidden_src = if final_in_input {
        &step.input
    } else {
        &step.scratch
    };
    batch
        .record_copy_buffer(hidden_src, &out_staging, hidden_bytes)
        .context("batched transformer token hidden: record hidden readback")?;
    let mut replay_resources = batched_decode_replay_resources(
        &step.input,
        &step.scratch,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        &out_staging,
        DType::F32,
        vec![batch_size, config.hidden_size],
        batch_size,
        step.hidden,
        step.rotary_dim,
        meta.max_blocks_per_seq,
    );
    replay_resources.push(vk_replay_resource(
        &step.token_ids,
        DType::U32,
        vec![batch_size],
    ));
    replay_vulkan_command_batch(
        batch,
        "vk-resident native token batched hidden decode",
        vec![
            batch_size,
            config.hidden_size,
            meta.max_blocks_per_seq,
            meta.block_size,
        ],
        batch_size,
        replay_resources,
    )
    .context("batched transformer token hidden: submit CommandBatch through ReplayPlan")?;

    let bytes = out_staging
        .read_mapped(hidden_bytes as usize)
        .context("batched transformer token hidden: read hidden staging")?;
    let mut hidden = Vec::with_capacity(batch_size * config.hidden_size);
    for chunk in bytes.chunks_exact(4).take(batch_size * config.hidden_size) {
        hidden.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Some(hidden))
}

/// Host-visible Vulkan buffers for one batched resident decode step's
/// paged metadata.
pub struct BatchedResidentDecodeMetaBuffers {
    pub block_table: std::sync::Arc<VulkanBuffer>,
    pub seq_lens: std::sync::Arc<VulkanBuffer>,
    pub slots: std::sync::Arc<VulkanBuffer>,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
}

/// Host-visible sampler metadata consumed by the batched resident decode
/// sample tail. These buffers are written before command recording and read by
/// the final penalty/sample shaders in the same submitted command batch.
pub struct BatchedResidentSampleBuffers {
    pub top_k: std::sync::Arc<VulkanBuffer>,
    pub temperatures: std::sync::Arc<VulkanBuffer>,
    pub top_p: std::sync::Arc<VulkanBuffer>,
    pub min_p: std::sync::Arc<VulkanBuffer>,
    pub seed_lo: std::sync::Arc<VulkanBuffer>,
    pub seed_hi: std::sync::Arc<VulkanBuffer>,
    pub history_rows: Option<std::sync::Arc<VulkanBuffer>>,
    pub history_indices: Option<std::sync::Arc<VulkanBuffer>>,
    pub history_counts: Option<std::sync::Arc<VulkanBuffer>>,
    pub repetitions: Option<std::sync::Arc<VulkanBuffer>>,
    pub presences: Option<std::sync::Arc<VulkanBuffer>>,
    pub frequencies: Option<std::sync::Arc<VulkanBuffer>>,
    pub batch_size: usize,
    pub history_items: usize,
}

/// Resident buffers for one batched decode step's hidden input and
/// per-row RoPE tables.
pub struct BatchedResidentDecodeStepBuffers {
    pub input: std::sync::Arc<VulkanBuffer>,
    pub scratch: std::sync::Arc<VulkanBuffer>,
    pub rope_cos: std::sync::Arc<VulkanBuffer>,
    pub rope_sin: std::sync::Arc<VulkanBuffer>,
    pub batch_size: usize,
    pub hidden: usize,
    pub rotary_dim: usize,
}

/// Resident buffers for a batched decode step whose input hidden rows
/// are produced by a Vulkan token-embedding gather.
pub struct BatchedResidentDecodeTokenStepBuffers {
    pub input: std::sync::Arc<VulkanBuffer>,
    pub scratch: std::sync::Arc<VulkanBuffer>,
    pub token_ids: std::sync::Arc<VulkanBuffer>,
    pub rope_cos: std::sync::Arc<VulkanBuffer>,
    pub rope_sin: std::sync::Arc<VulkanBuffer>,
    pub batch_size: usize,
    pub hidden: usize,
    pub rotary_dim: usize,
}

struct ResidentDecodeEmbeddingSource {
    weight: std::sync::Arc<VulkanBuffer>,
    shader: &'static str,
    vocab: usize,
    hidden: usize,
}

fn resident_decode_embedding_source_for_table(
    backend: &VulkanBackend,
    table: &kiln_tensor::Tensor,
    hidden: usize,
    transposed: bool,
) -> Result<Option<ResidentDecodeEmbeddingSource>> {
    let dims = table.dims();
    if dims.len() != 2 {
        return Ok(None);
    }
    let (vocab, table_hidden) = if transposed {
        (dims[1], dims[0])
    } else {
        (dims[0], dims[1])
    };
    if table_hidden != hidden || vocab == 0 {
        return Ok(None);
    }

    let (weight, shader) = match (table.dtype(), transposed) {
        (kiln_tensor::DType::BF16, false) => (
            backend.cached_bf16_packed_weight_buffer_kt(table)?,
            shaders::VK_EMBEDDING_LOOKUP_BF16W_F32,
        ),
        (kiln_tensor::DType::BF16, true) => (
            backend.cached_bf16_packed_weight_buffer_kt(table)?,
            shaders::VK_EMBEDDING_LOOKUP_T_BF16W_F32,
        ),
        (kiln_tensor::DType::F32 | kiln_tensor::DType::F16, false) => (
            backend.cached_f32_weight_buffer_kt(table)?,
            shaders::VK_EMBEDDING_LOOKUP_F32,
        ),
        (kiln_tensor::DType::F32 | kiln_tensor::DType::F16, true) => (
            backend.cached_f32_weight_buffer_kt(table)?,
            shaders::VK_EMBEDDING_LOOKUP_T_F32,
        ),
        _ => return Ok(None),
    };

    Ok(Some(ResidentDecodeEmbeddingSource {
        weight,
        shader,
        vocab,
        hidden,
    }))
}

fn resident_decode_embedding_source(
    backend: &VulkanBackend,
    weights: &crate::forward::GpuWeights,
    hidden: usize,
) -> Result<Option<ResidentDecodeEmbeddingSource>> {
    if let Some(source) =
        resident_decode_embedding_source_for_table(backend, &weights.embed_tokens, hidden, false)?
    {
        return Ok(Some(source));
    }
    resident_decode_embedding_source_for_table(backend, &weights.embed_tokens_t, hidden, true)
}

fn record_resident_decode_embedding_independent_into(
    batch: &mut CommandBatch,
    source: &ResidentDecodeEmbeddingSource,
    token_ids: &VulkanBuffer,
    out: &VulkanBuffer,
    batch_size: usize,
) -> Result<()> {
    let total = batch_size
        .checked_mul(source.hidden)
        .context("resident embedding gather: output element count overflow")?;
    batch.record_shader_no_previous_barrier(
        source.shader,
        &[token_ids.handle(), source.weight.handle(), out.handle()],
        &[batch_size as u32, source.hidden as u32, source.vocab as u32],
        Workgroups::OneD(total.div_ceil(256) as u32),
    )
}

fn ensure_resident_decode_embedding_ids(token_ids: &[u32], vocab: usize) -> Result<()> {
    for (row, &token) in token_ids.iter().enumerate() {
        anyhow::ensure!(
            (token as usize) < vocab,
            "resident embedding gather row {row}: token id {token} exceeds vocab {vocab}"
        );
    }
    Ok(())
}

fn record_resident_decode_rope_tables_into(
    backend: &VulkanBackend,
    batch: &mut CommandBatch,
    weights: &crate::forward::GpuWeights,
    seq_lens_buf: &VulkanBuffer,
    rope_cos_buf: &VulkanBuffer,
    rope_sin_buf: &VulkanBuffer,
    batch_size: usize,
    rotary_dim: usize,
) -> Result<bool> {
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        return Ok(false);
    }
    let half_rotary = rotary_dim / 2;
    let expected = [half_rotary];
    if weights.rotary_inv_freq.dims() != expected.as_slice() {
        return Ok(false);
    }
    let inv_freq_buf = backend.cached_f32_weight_buffer_kt(&weights.rotary_inv_freq)?;
    let total = batch_size
        .checked_mul(half_rotary)
        .context("resident RoPE table build: output element count overflow")?;
    batch.record_shader(
        shaders::VK_ROPE_TABLES_FROM_SEQ_LENS_F32,
        &[
            seq_lens_buf.handle(),
            inv_freq_buf.handle(),
            rope_cos_buf.handle(),
            rope_sin_buf.handle(),
        ],
        &[batch_size as u32, half_rotary as u32],
        Workgroups::OneD(total.div_ceil(256) as u32),
    )?;
    Ok(true)
}

/// Prepare resident hidden-input and RoPE buffers for a batched decode
/// step. All slices are f32 and row-major:
/// - `hidden_rows`: `[batch_size, hidden]`
/// - `rope_cos` / `rope_sin`: `[batch_size, rotary_dim / 2]`
pub fn prepare_batched_resident_decode_step_buffers(
    backend: &VulkanBackend,
    hidden_rows: &[f32],
    batch_size: usize,
    hidden: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    rotary_dim: usize,
) -> Result<BatchedResidentDecodeStepBuffers> {
    anyhow::ensure!(
        batch_size > 0,
        "batched resident decode step buffers: batch_size must be > 0"
    );
    anyhow::ensure!(
        hidden_rows.len() == batch_size * hidden,
        "batched resident decode step buffers: hidden row length mismatch"
    );
    anyhow::ensure!(
        rotary_dim % 2 == 0,
        "batched resident decode step buffers: rotary_dim must be even"
    );
    let half_rotary = rotary_dim / 2;
    anyhow::ensure!(
        rope_cos.len() == batch_size * half_rotary,
        "batched resident decode step buffers: rope_cos length mismatch"
    );
    anyhow::ensure!(
        rope_sin.len() == batch_size * half_rotary,
        "batched resident decode step buffers: rope_sin length mismatch"
    );

    let hidden_bytes = (hidden_rows.len().max(1) * 4) as u64;
    let rope_bytes = (rope_cos.len().max(1) * 4) as u64;
    let input = backend.acquire_resident_scratch_host_visible("native_b_io_a_hv", hidden_bytes)?;
    let scratch = backend.acquire_resident_scratch("native_b_io_b", hidden_bytes)?;
    let rope_cos_buf =
        backend.acquire_resident_scratch_host_visible("native_b_rope_cos_hv", rope_bytes)?;
    let rope_sin_buf =
        backend.acquire_resident_scratch_host_visible("native_b_rope_sin_hv", rope_bytes)?;

    input.write_mapped(bytemuck::cast_slice(hidden_rows))?;
    rope_cos_buf.write_mapped(bytemuck::cast_slice(rope_cos))?;
    rope_sin_buf.write_mapped(bytemuck::cast_slice(rope_sin))?;

    Ok(BatchedResidentDecodeStepBuffers {
        input,
        scratch,
        rope_cos: rope_cos_buf,
        rope_sin: rope_sin_buf,
        batch_size,
        hidden,
        rotary_dim,
    })
}

/// Prepare resident token-id and RoPE buffers for a batched decode
/// step. The hidden input buffer is device-local and filled by the
/// recorded embedding gather before the transformer stack.
pub fn prepare_batched_resident_decode_token_step_buffers(
    backend: &VulkanBackend,
    token_ids: &[u32],
    hidden: usize,
    rotary_dim: usize,
) -> Result<BatchedResidentDecodeTokenStepBuffers> {
    let batch_size = token_ids.len();
    anyhow::ensure!(
        batch_size > 0,
        "batched resident decode token step buffers: batch_size must be > 0"
    );
    anyhow::ensure!(
        rotary_dim > 0 && rotary_dim % 2 == 0,
        "batched resident decode token step buffers: rotary_dim must be positive and even"
    );
    let half_rotary = rotary_dim / 2;

    let hidden_bytes = (batch_size * hidden).max(1) as u64 * 4;
    let token_bytes = token_ids.len().max(1) as u64 * 4;
    let rope_bytes = (batch_size * half_rotary).max(1) as u64 * 4;
    let input = backend.acquire_resident_scratch("native_b_io_a", hidden_bytes)?;
    let scratch = backend.acquire_resident_scratch("native_b_io_b", hidden_bytes)?;
    let token_ids_buf =
        backend.acquire_resident_scratch_host_visible("native_b_token_ids_hv", token_bytes)?;
    let rope_cos_buf = backend.acquire_resident_scratch("native_b_rope_cos", rope_bytes)?;
    let rope_sin_buf = backend.acquire_resident_scratch("native_b_rope_sin", rope_bytes)?;

    token_ids_buf.write_mapped(bytemuck::cast_slice(token_ids))?;

    Ok(BatchedResidentDecodeTokenStepBuffers {
        input,
        scratch,
        token_ids: token_ids_buf,
        rope_cos: rope_cos_buf,
        rope_sin: rope_sin_buf,
        batch_size,
        hidden,
        rotary_dim,
    })
}

/// Flatten per-row block tables and decode positions into resident
/// metadata buffers consumed by [`submit_transformer_stack_batched_argmax`].
///
/// `start_positions[row]` is the absolute token position being decoded
/// for that row; this helper writes `seq_lens[row] = start_pos + 1`
/// and `slots[row] = block_table[row].slot_for(start_pos)`.
pub fn prepare_batched_resident_decode_meta_buffers(
    backend: &VulkanBackend,
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    block_size: usize,
) -> Result<BatchedResidentDecodeMetaBuffers> {
    let batch_size = block_tables.len();
    anyhow::ensure!(
        batch_size > 0,
        "batched resident decode metadata: batch_size must be > 0"
    );
    anyhow::ensure!(
        start_positions.len() == batch_size,
        "batched resident decode metadata: start position count mismatch"
    );
    anyhow::ensure!(
        block_size > 0,
        "batched resident decode metadata: block_size must be > 0"
    );
    let max_blocks_per_seq = block_tables
        .iter()
        .map(|bt| bt.blocks.len())
        .max()
        .unwrap_or(0)
        .max(1);

    let mut block_table_flat = Vec::<u32>::with_capacity(batch_size * max_blocks_per_seq);
    let mut seq_lens = Vec::<u32>::with_capacity(batch_size);
    let mut slots = Vec::<u32>::with_capacity(batch_size);
    for (row, (&block_table, &start_pos)) in
        block_tables.iter().zip(start_positions.iter()).enumerate()
    {
        let seq_len = start_pos
            .checked_add(1)
            .context("batched resident decode metadata: seq_len overflow")?;
        anyhow::ensure!(
            block_table.capacity(block_size) >= seq_len,
            "batched resident decode metadata row {row}: block table capacity {} < seq_len {seq_len}",
            block_table.capacity(block_size),
        );
        let slot = block_table.slot_for(start_pos, block_size).ok_or_else(|| {
            anyhow::anyhow!(
                "batched resident decode metadata row {row}: no slot for start_pos {start_pos}"
            )
        })?;
        seq_lens.push(
            u32::try_from(seq_len)
                .context("batched resident decode metadata: seq_len exceeds u32")?,
        );
        slots.push(
            u32::try_from(slot).context("batched resident decode metadata: slot exceeds u32")?,
        );

        let pad_block = *block_table.blocks.last().unwrap_or(&0);
        for idx in 0..max_blocks_per_seq {
            block_table_flat.push(*block_table.blocks.get(idx).unwrap_or(&pad_block));
        }
    }

    let block_table_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_block_table_hv",
        (block_table_flat.len().max(1) * 4) as u64,
    )?;
    let seq_lens_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_seq_lens_hv",
        (seq_lens.len().max(1) * 4) as u64,
    )?;
    let slots_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_slots_hv",
        (slots.len().max(1) * 4) as u64,
    )?;
    block_table_buf.write_mapped(bytemuck::cast_slice(&block_table_flat))?;
    seq_lens_buf.write_mapped(bytemuck::cast_slice(&seq_lens))?;
    slots_buf.write_mapped(bytemuck::cast_slice(&slots))?;

    Ok(BatchedResidentDecodeMetaBuffers {
        block_table: block_table_buf,
        seq_lens: seq_lens_buf,
        slots: slots_buf,
        max_blocks_per_seq,
        block_size,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_batched_resident_sample_buffers(
    backend: &VulkanBackend,
    batch_size: usize,
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<BatchedResidentSampleBuffers> {
    anyhow::ensure!(
        batch_size > 0,
        "batched resident sampler: batch_size must be > 0"
    );
    anyhow::ensure!(
        history_rows.len() == history_indices.len()
            && history_indices.len() == history_counts.len(),
        "batched resident sampler: history row/index/count length mismatch"
    );
    anyhow::ensure!(
        history_rows.iter().all(|&row| (row as usize) < batch_size),
        "batched resident sampler: history row out of range"
    );
    anyhow::ensure!(
        repetition_penalties.len() == batch_size
            && presence_penalties.len() == batch_size
            && frequency_penalties.len() == batch_size
            && temperatures.len() == batch_size
            && top_k.len() == batch_size
            && top_p.len() == batch_size
            && min_p.len() == batch_size
            && seeds.len() == batch_size,
        "batched resident sampler: per-row parameter length mismatch"
    );
    for row in 0..batch_size {
        let temp = temperatures[row];
        let k = top_k[row];
        let greedy = temp == 0.0 || (k == 1 && temp.is_finite() && temp > 0.0);
        anyhow::ensure!(
            greedy || (temp.is_finite() && temp > 0.0),
            "batched resident sampler: row {row} invalid temperature {temp}"
        );
        anyhow::ensure!(
            greedy || (k > 0 && k <= kiln_vulkan_kernel::kernels::TOPK_SAMPLE_KERNEL_K_MAX),
            "batched resident sampler: row {row} top_k {k} out of supported range"
        );
    }

    let seed_lo: Vec<u32> = seeds
        .iter()
        .map(|seed| (*seed & 0xFFFF_FFFF) as u32)
        .collect();
    let seed_hi: Vec<u32> = seeds.iter().map(|seed| (*seed >> 32) as u32).collect();

    let top_k_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_sample_top_k_hv",
        (batch_size * 4) as u64,
    )?;
    let temperatures_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_sample_temperatures_hv",
        (batch_size * 4) as u64,
    )?;
    let top_p_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_sample_top_p_hv",
        (batch_size * 4) as u64,
    )?;
    let min_p_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_sample_min_p_hv",
        (batch_size * 4) as u64,
    )?;
    let seed_lo_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_sample_seed_lo_hv",
        (batch_size * 4) as u64,
    )?;
    let seed_hi_buf = backend.acquire_resident_scratch_host_visible(
        "native_b_sample_seed_hi_hv",
        (batch_size * 4) as u64,
    )?;

    top_k_buf.write_mapped(bytemuck::cast_slice(top_k))?;
    temperatures_buf.write_mapped(bytemuck::cast_slice(temperatures))?;
    top_p_buf.write_mapped(bytemuck::cast_slice(top_p))?;
    min_p_buf.write_mapped(bytemuck::cast_slice(min_p))?;
    seed_lo_buf.write_mapped(bytemuck::cast_slice(&seed_lo))?;
    seed_hi_buf.write_mapped(bytemuck::cast_slice(&seed_hi))?;

    let history_items = history_indices.len();
    let (
        history_rows_buf,
        history_indices_buf,
        history_counts_buf,
        repetitions_buf,
        presences_buf,
        frequencies_buf,
    ) = if history_items > 0 {
        let history_rows_buf = backend.acquire_resident_scratch_host_visible(
            "native_b_sample_history_rows_hv",
            (history_items * 4) as u64,
        )?;
        let history_indices_buf = backend.acquire_resident_scratch_host_visible(
            "native_b_sample_history_indices_hv",
            (history_items * 4) as u64,
        )?;
        let history_counts_buf = backend.acquire_resident_scratch_host_visible(
            "native_b_sample_history_counts_hv",
            (history_items * 4) as u64,
        )?;
        let repetitions_buf = backend.acquire_resident_scratch_host_visible(
            "native_b_sample_repetitions_hv",
            (batch_size * 4) as u64,
        )?;
        let presences_buf = backend.acquire_resident_scratch_host_visible(
            "native_b_sample_presences_hv",
            (batch_size * 4) as u64,
        )?;
        let frequencies_buf = backend.acquire_resident_scratch_host_visible(
            "native_b_sample_frequencies_hv",
            (batch_size * 4) as u64,
        )?;
        history_rows_buf.write_mapped(bytemuck::cast_slice(history_rows))?;
        history_indices_buf.write_mapped(bytemuck::cast_slice(history_indices))?;
        history_counts_buf.write_mapped(bytemuck::cast_slice(history_counts))?;
        repetitions_buf.write_mapped(bytemuck::cast_slice(repetition_penalties))?;
        presences_buf.write_mapped(bytemuck::cast_slice(presence_penalties))?;
        frequencies_buf.write_mapped(bytemuck::cast_slice(frequency_penalties))?;
        (
            Some(history_rows_buf),
            Some(history_indices_buf),
            Some(history_counts_buf),
            Some(repetitions_buf),
            Some(presences_buf),
            Some(frequencies_buf),
        )
    } else {
        (None, None, None, None, None, None)
    };

    Ok(BatchedResidentSampleBuffers {
        top_k: top_k_buf,
        temperatures: temperatures_buf,
        top_p: top_p_buf,
        min_p: min_p_buf,
        seed_lo: seed_lo_buf,
        seed_hi: seed_hi_buf,
        history_rows: history_rows_buf,
        history_indices: history_indices_buf,
        history_counts: history_counts_buf,
        repetitions: repetitions_buf,
        presences: presences_buf,
        frequencies: frequencies_buf,
        batch_size,
        history_items,
    })
}

/// Convenience wrapper for callers that already have host-side f32
/// hidden rows and RoPE tables for a batched decode step.
///
/// This prepares the resident input/RoPE/metadata buffers, submits the
/// full batched stack, and returns greedy token IDs. It deliberately
/// does not do embedding lookup, RoPE table construction, or resident
/// KV-cache seeding; those remain caller-owned so runtime routing can
/// control session boundaries explicitly.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_argmax_from_host(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    hidden_rows: &[f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    block_size: usize,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    vk_kv_cache: &VkPagedKvCache,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<Vec<u32>>> {
    let batch_size = block_tables.len();
    let step = prepare_batched_resident_decode_step_buffers(
        backend,
        hidden_rows,
        batch_size,
        config.hidden_size,
        rope_cos,
        rope_sin,
        config.rotary_dim(),
    )?;
    let meta = prepare_batched_resident_decode_meta_buffers(
        backend,
        block_tables,
        start_positions,
        block_size,
    )?;
    submit_transformer_stack_batched_argmax(
        backend,
        vk_device,
        &step.input,
        &step.scratch,
        weights,
        config,
        batch_size,
        meta.max_blocks_per_seq,
        meta.block_size,
        vk_kv_cache,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        recurrent_states,
        conv_states,
    )
}

/// Convenience wrapper for callers that want the native resident stack and
/// stochastic sampler in one command-batch submit.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_sample_from_host(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    hidden_rows: &[f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    block_size: usize,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    vk_kv_cache: &VkPagedKvCache,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
    history_rows: &[u32],
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalties: &[f32],
    presence_penalties: &[f32],
    frequency_penalties: &[f32],
    temperatures: &[f32],
    top_k: &[u32],
    top_p: &[f32],
    min_p: &[f32],
    seeds: &[u64],
) -> Result<Option<Vec<u32>>> {
    let batch_size = block_tables.len();
    let step = prepare_batched_resident_decode_step_buffers(
        backend,
        hidden_rows,
        batch_size,
        config.hidden_size,
        rope_cos,
        rope_sin,
        config.rotary_dim(),
    )?;
    let meta = prepare_batched_resident_decode_meta_buffers(
        backend,
        block_tables,
        start_positions,
        block_size,
    )?;
    let sample = prepare_batched_resident_sample_buffers(
        backend,
        batch_size,
        history_rows,
        history_indices,
        history_counts,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
        temperatures,
        top_k,
        top_p,
        min_p,
        seeds,
    )?;
    submit_transformer_stack_batched_sample(
        backend,
        vk_device,
        &step.input,
        &step.scratch,
        &sample,
        weights,
        config,
        batch_size,
        meta.max_blocks_per_seq,
        meta.block_size,
        vk_kv_cache,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        recurrent_states,
        conv_states,
    )
}

/// Convenience wrapper for callers that want the native resident stack
/// output instead of the final greedy argmax.
#[allow(clippy::too_many_arguments)]
pub fn submit_transformer_stack_batched_hidden_from_host(
    backend: &VulkanBackend,
    vk_device: &VulkanDevice,
    hidden_rows: &[f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    block_tables: &[&BlockTable],
    start_positions: &[usize],
    block_size: usize,
    weights: &crate::forward::GpuWeights,
    config: &ModelConfig,
    vk_kv_cache: &VkPagedKvCache,
    recurrent_states: &[kiln_tensor::Tensor],
    conv_states: &[kiln_tensor::Tensor],
) -> Result<Option<Vec<f32>>> {
    let batch_size = block_tables.len();
    let step = prepare_batched_resident_decode_step_buffers(
        backend,
        hidden_rows,
        batch_size,
        config.hidden_size,
        rope_cos,
        rope_sin,
        config.rotary_dim(),
    )?;
    let meta = prepare_batched_resident_decode_meta_buffers(
        backend,
        block_tables,
        start_positions,
        block_size,
    )?;
    submit_transformer_stack_batched_hidden(
        backend,
        vk_device,
        &step.input,
        &step.scratch,
        weights,
        config,
        batch_size,
        meta.max_blocks_per_seq,
        meta.block_size,
        vk_kv_cache,
        &step.rope_cos,
        &step.rope_sin,
        &meta.block_table,
        &meta.seq_lens,
        &meta.slots,
        recurrent_states,
        conv_states,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_plans_share_the_qualified_kernel_policy() {
        assert_eq!(
            linear_bf16w_batched_plan(16, 64).0,
            shaders::LINEAR_DECODE_BATCHED_ROWS4_BF16W
        );
        assert_eq!(
            linear_bf16w_batched_plan(64, 64).0,
            shaders::LINEAR_DECODE_BATCHED_ROWS8_BF16W
        );
        assert_eq!(
            mlp_gate_up_bf16w_batched_plan(8, 64).0,
            shaders::MLP_GATE_UP_DECODE_BATCHED_ROWS4_BF16W
        );
        assert_eq!(
            mlp_gate_up_bf16w_batched_plan(256, 64).0,
            shaders::MLP_GATE_UP_DECODE_BATCHED_ROWS8_BF16W
        );
        assert_eq!(
            full_attn_qkv_gate_split_bf16w_plan(2, 64).0,
            shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS4_BF16W
        );
        assert_eq!(
            full_attn_qkv_gate_split_bf16w_plan(64, 64).0,
            shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS8_BF16W
        );
        assert_eq!(
            gdn_in_proj_bf16w_batched_plan(16, 64, 64, 64, 64, 256).0,
            shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W
        );
        assert_eq!(
            gdn_in_proj_bf16w_batched_plan(64, 64, 64, 64, 64, 256).0,
            shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W
        );
    }

    fn upload_f32(device: &VulkanDevice, values: &[f32]) -> Result<VulkanBuffer> {
        let bytes = bytemuck::cast_slice(values);
        let buffer = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            bytes.len() as u64,
        )?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &buffer,
            bytes,
        )?;
        Ok(buffer)
    }

    fn read_f32(device: &VulkanDevice, buffer: &VulkanBuffer) -> Result<Vec<f32>> {
        let bytes = VulkanBuffer::read_back(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            buffer,
        )?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    #[test]
    fn causal_conv_state_advance_waits_for_output_read() -> Result<()> {
        if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
            eprintln!(
                "skip causal_conv_state_advance_waits_for_output_read: \
                 KILN_TENSOR_VULKAN_TEST unset"
            );
            return Ok(());
        }
        assert!(
            VulkanDevice::probe(),
            "KILN_TENSOR_VULKAN_TEST=1 requires a working Vulkan device"
        );
        let device = VulkanDevice::new()?;

        const CHANNELS: usize = 4096;
        const KERNEL_SIZE: usize = 4;
        const REPEATS: usize = 64;
        let x: Vec<f32> = (0..CHANNELS)
            .map(|channel| 0.25 + (channel % 31) as f32 * 0.03125)
            .collect();
        // With fresh zero state and seq_len=1, only the final weight sees x.
        // Keeping that weight zero makes every correct output exactly zero;
        // any nonzero value exposes state advance racing the preceding read.
        let weights: Vec<f32> = (0..CHANNELS)
            .flat_map(|_| [0.125f32, -0.25, 0.5, 0.0])
            .collect();
        let x_buffer = upload_f32(&device, &x)?;
        let weight_buffer = upload_f32(&device, &weights)?;
        let output_buffer = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            (CHANNELS * size_of::<f32>()) as u64,
        )?;
        let zero_state = vec![0.0f32; CHANNELS * (KERNEL_SIZE - 1)];
        let mut baseline_state = None;

        for iteration in 0..REPEATS {
            let state_buffer = upload_f32(&device, &zero_state)?;
            let mut batch = CommandBatch::new(&device)?;
            record_causal_conv1d_update_into(
                &mut batch,
                &x_buffer,
                &weight_buffer,
                &state_buffer,
                &output_buffer,
                1,
                CHANNELS,
                1,
                KERNEL_SIZE,
            )?;
            batch.submit_and_wait("causal conv state ordering regression")?;

            let output = read_f32(&device, &output_buffer)?;
            assert!(
                output.iter().all(|&value| value == 0.0),
                "iteration {iteration}: state advance raced convolution output read"
            );
            let state = read_f32(&device, &state_buffer)?;
            for (channel, state_row) in state.chunks_exact(KERNEL_SIZE - 1).enumerate() {
                assert_eq!(
                    state_row,
                    &[0.0, 0.0, x[channel]],
                    "iteration {iteration}: wrong state for channel {channel}"
                );
            }
            if let Some(baseline) = &baseline_state {
                assert_eq!(
                    &state, baseline,
                    "iteration {iteration}: fresh-state update was nondeterministic"
                );
            } else {
                baseline_state = Some(state);
            }
        }
        Ok(())
    }
}
