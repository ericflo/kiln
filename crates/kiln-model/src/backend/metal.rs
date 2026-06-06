//! Metal backend: candle's fused SDPA for the attention hot path, portable
//! fallback for GDN and paged-decode.
//!
//! The chokepoint-routed `sdpa` symbol (imported at module level from the
//! kt-side re-export) is an MLX-style fused scaled-dot-product attention
//! kernel with native GQA, BF16, and head dims {32, 64, 72, 80, 96, 128,
//! 256, 512}. For typical transformer head sizes this replaces the vendored
//! CUDA FlashAttention-2 call on Apple Silicon.

use anyhow::{Context, Result};

use super::metal_config::*;
use super::{
    metal_residency, metal_training, BackendRuntime, TrainingCapabilities, TrainingPrecisionPolicy,
};

// Phase 7 #1082: module-level imports for the kt-metal chokepoint types,
// hoisted from ~92 per-function `use` statements so that the chokepoint
// surface in this file is centralized at a single import location. Future
// substrate swaps (e.g. candle → objc2-metal) touch this single import
// block instead of hundreds of scattered fully-qualified references.
use kiln_tensor::MetalStorage;
use kiln_tensor::metal_types::{
    Buffer, BufferOffset, ComputePipeline, IndirectCommandBufferDescriptor, IndirectComputeCommand,
    IndirectDispatchKind, Library, MTLResourceOptions, MTLResourceUsage, MetalCompanion,
    MetalRawDevice, buffer_o_kt,
};

/// Host abstraction for the per-device MSL pipeline / library caches
/// (#1082). The `metal_*_pipeline` + `metal_shared_library` helpers take
/// `&dyn MetalPipelineHost`. The candle-free kt `MetalCompanion` is the
/// sole implementor now that the substrate is kiln-owned
/// (`kiln_tensor::metal_rt`); the migration-era candle `MetalDevice` impl
/// retired alongside the `candle_metal_kernels` dependency drop. It
/// exposes the raw kt substrate device (for `new_library_with_source` /
/// `new_compute_pipeline_state_with_function`) and a stable per-device
/// `registry_id()` cache key so one compiled pipeline is shared per
/// physical GPU (no double-compile).
pub(crate) trait MetalPipelineHost {
    /// The raw substrate device for library / pipeline construction.
    fn pipeline_raw_device(&self) -> &MetalRawDevice;
    /// Stable per-device cache key (`MTLDevice::registryID`).
    fn pipeline_cache_key(&self) -> u64;
}

impl MetalPipelineHost for MetalCompanion {
    fn pipeline_raw_device(&self) -> &MetalRawDevice {
        self.device()
    }
    fn pipeline_cache_key(&self) -> u64 {
        self.device_id()
    }
}

// Per-function pipeline-cache helpers reach for these std types; hoisted to
// module-level so the 46 pipeline-builder helpers below stop repeating the
// import boilerplate (#1082 cleanup).
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// Phase 7 #1082 — small bridges from candle Layout / DType to their kt
// siblings, used by per-call-site migrations to `buffer_o_kt`. Defined
// locally (not in `kiln-tensor::metal_types`) so each helper-family
// migration can land without touching the chokepoint module. The Layout
// bridge clones the shape/stride vectors (typically rank 2-4); the cost
// is negligible relative to a Metal encoder dispatch and stays well
// below 100ns per call. Returns a kt Layout that round-trips
// `start_offset()` exactly, which is what `buffer_o_kt` reads.
/// Downcast a kt `Tensor`'s storage to `&MetalStorage` — the standard
/// entry every candle-free Metal kernel helper uses to reach `.buffer()`
/// + `.companion()`. (#1082)
#[inline]
fn kt_metal(t: &kiln_tensor::Tensor) -> Result<&MetalStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .context("expected a Metal-backed kiln_tensor::Tensor")
}

/// Allocate a fresh contiguous Metal output tensor of `dtype` × `dims` on
/// the same device as `like`'s companion. Zero-initialized via the UMA
/// `StorageModeShared` path (`MetalStorage::zeros_kt`); kernels that fully
/// overwrite their output pay a cheap UMA zero-fill — correctness-first
/// per #1082 ("ship parity before optimizing"). Replaces the candle
/// `Tensor::empty(dims, dtype, x.device())` output allocation in the
/// kt-flipped kernel helpers. (#1082)
fn kt_metal_alloc(
    like: &MetalStorage,
    dtype: kiln_tensor::DType,
    dims: &[usize],
) -> Result<kiln_tensor::Tensor> {
    let companion = like.companion()?;
    let n: usize = dims.iter().product();
    let storage = MetalStorage::zeros_kt(companion.device(), like.device_index(), dtype, n)?;
    kiln_tensor::Tensor::from_parts(
        std::sync::Arc::new(storage),
        kiln_tensor::Layout::contiguous(dims.to_vec()),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| anyhow::anyhow!("kt_metal_alloc: {e}"))
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct MetalGraphScalarBuffer {
    buffer: Buffer,
}

#[allow(dead_code)]
impl MetalGraphScalarBuffer {
    fn new_u32(device: &MetalRawDevice, value: u32) -> Result<Self> {
        Self::new_copy(device, &value)
    }

    fn new_f32(device: &MetalRawDevice, value: f32) -> Result<Self> {
        Self::new_copy(device, &value)
    }

    fn new_copy<T: Copy>(device: &MetalRawDevice, value: &T) -> Result<Self> {
        let byte_len = std::mem::size_of::<T>();
        let buffer = device
            .new_buffer_with_data(
                value as *const T as *const std::ffi::c_void,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| anyhow::anyhow!("MetalGraphScalarBuffer::new_copy: {e:?}"))?;
        Ok(Self { buffer })
    }

    pub(crate) fn write_u32(&self, value: u32) -> Result<()> {
        self.write_copy(&value)
    }

    pub(crate) fn write_f32(&self, value: f32) -> Result<()> {
        self.write_copy(&value)
    }

    fn write_copy<T: Copy>(&self, value: &T) -> Result<()> {
        let byte_len = std::mem::size_of::<T>();
        anyhow::ensure!(
            byte_len <= self.buffer.length(),
            "MetalGraphScalarBuffer write of {byte_len} bytes exceeds buffer length {}",
            self.buffer.length()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                value as *const T as *const u8,
                self.buffer.contents(),
                byte_len,
            );
        }
        Ok(())
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct MetalGraphResourceRef {
    pub(crate) buffer: Buffer,
    pub(crate) usage: MTLResourceUsage,
}

impl MetalGraphResourceRef {
    fn read(buffer: &Buffer) -> Self {
        Self {
            buffer: buffer.clone(),
            usage: MTLResourceUsage::Read,
        }
    }

    fn write(buffer: &Buffer) -> Self {
        Self {
            buffer: buffer.clone(),
            usage: MTLResourceUsage::Write,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct MetalPagedKvWriteTokenMajorIcbArgs {
    pub(crate) slot: MetalGraphScalarBuffer,
    heads: MetalGraphScalarBuffer,
    head_dim: MetalGraphScalarBuffer,
}

#[allow(dead_code)]
impl MetalPagedKvWriteTokenMajorIcbArgs {
    pub(crate) fn new(
        companion: &MetalCompanion,
        slot: u32,
        heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        Ok(Self {
            slot: MetalGraphScalarBuffer::new_u32(companion.device(), slot)?,
            heads: MetalGraphScalarBuffer::new_u32(companion.device(), heads)?,
            head_dim: MetalGraphScalarBuffer::new_u32(companion.device(), head_dim)?,
        })
    }

    pub(crate) fn update_slot(&self, slot: u32) -> Result<()> {
        self.slot.write_u32(slot)
    }

    fn scalar_resources(&self) -> [MetalGraphResourceRef; 3] {
        [
            MetalGraphResourceRef::read(self.slot.buffer()),
            MetalGraphResourceRef::read(self.heads.buffer()),
            MetalGraphResourceRef::read(self.head_dim.buffer()),
        ]
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct MetalPagedKvWriteTokenMajorBatchIcbArgs {
    batch: MetalGraphScalarBuffer,
    heads: MetalGraphScalarBuffer,
    head_dim: MetalGraphScalarBuffer,
    total_slots: MetalGraphScalarBuffer,
}

#[allow(dead_code)]
impl MetalPagedKvWriteTokenMajorBatchIcbArgs {
    pub(crate) fn new(
        companion: &MetalCompanion,
        batch: u32,
        heads: u32,
        head_dim: u32,
        total_slots: u32,
    ) -> Result<Self> {
        Ok(Self {
            batch: MetalGraphScalarBuffer::new_u32(companion.device(), batch)?,
            heads: MetalGraphScalarBuffer::new_u32(companion.device(), heads)?,
            head_dim: MetalGraphScalarBuffer::new_u32(companion.device(), head_dim)?,
            total_slots: MetalGraphScalarBuffer::new_u32(companion.device(), total_slots)?,
        })
    }

    fn scalar_resources(&self) -> [MetalGraphResourceRef; 4] {
        [
            MetalGraphResourceRef::read(self.batch.buffer()),
            MetalGraphResourceRef::read(self.heads.buffer()),
            MetalGraphResourceRef::read(self.head_dim.buffer()),
            MetalGraphResourceRef::read(self.total_slots.buffer()),
        ]
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct MetalPagedAttnDecodeDynSeqlenIcbArgs {
    batch: MetalGraphScalarBuffer,
    max_blocks_per_seq: MetalGraphScalarBuffer,
    max_seqlen_k: MetalGraphScalarBuffer,
    page_block_size: MetalGraphScalarBuffer,
    q_heads: MetalGraphScalarBuffer,
    kv_heads: MetalGraphScalarBuffer,
    softmax_scale: MetalGraphScalarBuffer,
    total_slots: MetalGraphScalarBuffer,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MetalPagedAttnDecodeDynSeqlenScalars {
    pub(crate) batch: u32,
    pub(crate) max_blocks_per_seq: u32,
    pub(crate) max_seqlen_k: u32,
    pub(crate) page_block_size: u32,
    pub(crate) q_heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) softmax_scale: f32,
    pub(crate) total_slots: u32,
}

#[allow(dead_code)]
impl MetalPagedAttnDecodeDynSeqlenIcbArgs {
    pub(crate) fn new(
        companion: &MetalCompanion,
        scalars: MetalPagedAttnDecodeDynSeqlenScalars,
    ) -> Result<Self> {
        Ok(Self {
            batch: MetalGraphScalarBuffer::new_u32(companion.device(), scalars.batch)?,
            max_blocks_per_seq: MetalGraphScalarBuffer::new_u32(
                companion.device(),
                scalars.max_blocks_per_seq,
            )?,
            max_seqlen_k: MetalGraphScalarBuffer::new_u32(
                companion.device(),
                scalars.max_seqlen_k,
            )?,
            page_block_size: MetalGraphScalarBuffer::new_u32(
                companion.device(),
                scalars.page_block_size,
            )?,
            q_heads: MetalGraphScalarBuffer::new_u32(companion.device(), scalars.q_heads)?,
            kv_heads: MetalGraphScalarBuffer::new_u32(companion.device(), scalars.kv_heads)?,
            softmax_scale: MetalGraphScalarBuffer::new_f32(
                companion.device(),
                scalars.softmax_scale,
            )?,
            total_slots: MetalGraphScalarBuffer::new_u32(companion.device(), scalars.total_slots)?,
        })
    }

    pub(crate) fn update_max_seqlen_k(&self, max_seqlen_k: u32) -> Result<()> {
        self.max_seqlen_k.write_u32(max_seqlen_k)
    }

    pub(crate) fn update_softmax_scale(&self, softmax_scale: f32) -> Result<()> {
        self.softmax_scale.write_f32(softmax_scale)
    }

    fn scalar_resources(&self) -> [MetalGraphResourceRef; 8] {
        [
            MetalGraphResourceRef::read(self.batch.buffer()),
            MetalGraphResourceRef::read(self.max_blocks_per_seq.buffer()),
            MetalGraphResourceRef::read(self.max_seqlen_k.buffer()),
            MetalGraphResourceRef::read(self.page_block_size.buffer()),
            MetalGraphResourceRef::read(self.q_heads.buffer()),
            MetalGraphResourceRef::read(self.kv_heads.buffer()),
            MetalGraphResourceRef::read(self.softmax_scale.buffer()),
            MetalGraphResourceRef::read(self.total_slots.buffer()),
        ]
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MetalSingleTokenPagedDecodeIcbGraph {
    captured: kiln_graph_metal::MetalCapturedGraph,
    kv_args: MetalPagedKvWriteTokenMajorIcbArgs,
    attn_args: MetalPagedAttnDecodeDynSeqlenIcbArgs,
}

#[allow(dead_code)]
impl MetalSingleTokenPagedDecodeIcbGraph {
    pub(crate) fn replay(&self, slot: u32, max_seqlen_k: u32, softmax_scale: f32) -> Result<()> {
        self.kv_args.update_slot(slot)?;
        self.attn_args.update_max_seqlen_k(max_seqlen_k)?;
        self.attn_args.update_softmax_scale(softmax_scale)?;
        self.captured
            .replay()
            .map_err(|e| anyhow::anyhow!("Metal ICB paged decode replay: {e}"))?;
        self.captured
            .wait_until_completed()
            .map_err(|e| anyhow::anyhow!("Metal ICB paged decode wait: {e}"))?;
        Ok(())
    }

    pub(crate) fn replay_count(&self) -> u64 {
        self.captured.replay_count()
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[derive(Debug)]
pub(crate) struct MetalPagedDecodeIcbGraph {
    captured: kiln_graph_metal::MetalCapturedGraph,
    attn_args: MetalPagedAttnDecodeDynSeqlenIcbArgs,
}

#[allow(dead_code)]
impl MetalPagedDecodeIcbGraph {
    pub(crate) fn replay(&self, max_seqlen_k: u32, softmax_scale: f32) -> Result<()> {
        self.attn_args.update_max_seqlen_k(max_seqlen_k)?;
        self.attn_args.update_softmax_scale(softmax_scale)?;
        self.captured
            .replay()
            .map_err(|e| anyhow::anyhow!("Metal ICB paged decode replay: {e}"))?;
        self.captured
            .wait_until_completed()
            .map_err(|e| anyhow::anyhow!("Metal ICB paged decode wait: {e}"))?;
        Ok(())
    }

    pub(crate) fn replay_count(&self) -> u64 {
        self.captured.replay_count()
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn metal_record_paged_decode_icb_graph(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    slots: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<MetalPagedDecodeIcbGraph> {
    let q_metal = kt_metal(q)?;
    let companion = q_metal.companion()?;
    let (batch, _, q_heads, _) = q.dims4()?;
    let (total_slots, kv_heads, head_dim) = k_pool.dims3()?;
    let (_, max_blocks_per_seq) = block_table.dims2()?;

    anyhow::ensure!(batch > 0, "Metal paged decode ICB graph requires batch > 0");
    anyhow::ensure!(
        slots.dims1()? == batch,
        "Metal paged decode ICB graph slots length must match batch"
    );

    let descriptor = IndirectCommandBufferDescriptor {
        max_kernel_buffer_bind_count: 14,
        dispatch_kind: IndirectDispatchKind::ThreadgroupsAndThreads,
        ..Default::default()
    };
    let icb = companion
        .device()
        .new_indirect_command_buffer(descriptor, 2, MTLResourceOptions::StorageModePrivate)
        .map_err(|e| anyhow::anyhow!("create Metal paged decode ICB: {e:?}"))?;
    icb.reset(0, 2);

    let kv_args = MetalPagedKvWriteTokenMajorBatchIcbArgs::new(
        &companion,
        batch as u32,
        kv_heads as u32,
        head_dim as u32,
        total_slots as u32,
    )?;
    let attn_args = MetalPagedAttnDecodeDynSeqlenIcbArgs::new(
        &companion,
        MetalPagedAttnDecodeDynSeqlenScalars {
            batch: batch as u32,
            max_blocks_per_seq: max_blocks_per_seq as u32,
            max_seqlen_k: max_seqlen_k as u32,
            page_block_size: page_block_size as u32,
            q_heads: q_heads as u32,
            kv_heads: kv_heads as u32,
            softmax_scale,
            total_slots: total_slots as u32,
        },
    )?;

    let kv_resources = metal_record_paged_kv_write_token_major_batch_bf16_icb(
        &icb.compute_command_at(0),
        &kv_args,
        k_pool,
        v_pool,
        slots,
        k,
        v,
    )?;
    let attn_resources = metal_record_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_icb(
        &icb.compute_command_at(1),
        &attn_args,
        q,
        k_pool,
        v_pool,
        block_table,
        seqused_k,
        out,
        max_seqlen_k,
        page_block_size,
    )?;

    let resources = merge_metal_graph_resources(kv_resources.into_iter().chain(attn_resources))?;
    let captured = kiln_graph_metal::MetalCapturedGraph::from_indirect_commands_with_resources(
        (*companion).clone(),
        icb,
        2,
        0,
        resources,
    )
    .map_err(|e| anyhow::anyhow!("capture Metal paged decode ICB graph: {e}"))?;

    Ok(MetalPagedDecodeIcbGraph {
        captured,
        attn_args,
    })
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn metal_record_single_token_paged_decode_icb_graph(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    slot: usize,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<MetalSingleTokenPagedDecodeIcbGraph> {
    let q_metal = kt_metal(q)?;
    let companion = q_metal.companion()?;
    let (_, kv_heads, head_dim) = k_pool.dims3()?;
    let (batch, _, q_heads, _) = q.dims4()?;
    let (total_slots, _, _) = k_pool.dims3()?;
    let (_, max_blocks_per_seq) = block_table.dims2()?;

    anyhow::ensure!(batch == 1, "Metal single-token ICB graph requires batch=1");
    anyhow::ensure!(slot <= u32::MAX as usize, "Metal ICB KV slot exceeds u32");

    let descriptor = IndirectCommandBufferDescriptor {
        max_kernel_buffer_bind_count: 14,
        dispatch_kind: IndirectDispatchKind::Threadgroups,
        ..Default::default()
    };
    let icb = companion
        .device()
        .new_indirect_command_buffer(descriptor, 2, MTLResourceOptions::StorageModePrivate)
        .map_err(|e| anyhow::anyhow!("create Metal paged decode ICB: {e:?}"))?;
    icb.reset(0, 2);

    let kv_args = MetalPagedKvWriteTokenMajorIcbArgs::new(
        &companion,
        slot as u32,
        kv_heads as u32,
        head_dim as u32,
    )?;
    let attn_args = MetalPagedAttnDecodeDynSeqlenIcbArgs::new(
        &companion,
        MetalPagedAttnDecodeDynSeqlenScalars {
            batch: batch as u32,
            max_blocks_per_seq: max_blocks_per_seq as u32,
            max_seqlen_k: max_seqlen_k as u32,
            page_block_size: page_block_size as u32,
            q_heads: q_heads as u32,
            kv_heads: kv_heads as u32,
            softmax_scale,
            total_slots: total_slots as u32,
        },
    )?;

    let kv_resources = metal_record_paged_kv_write_token_major_bf16_icb(
        &icb.compute_command_at(0),
        &kv_args,
        k_pool,
        v_pool,
        slot,
        k,
        v,
    )?;
    let attn_resources = metal_record_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_icb(
        &icb.compute_command_at(1),
        &attn_args,
        q,
        k_pool,
        v_pool,
        block_table,
        seqused_k,
        out,
        max_seqlen_k,
        page_block_size,
    )?;

    let resources = merge_metal_graph_resources(kv_resources.into_iter().chain(attn_resources))?;
    let captured = kiln_graph_metal::MetalCapturedGraph::from_indirect_commands_with_resources(
        (*companion).clone(),
        icb,
        2,
        0,
        resources,
    )
    .map_err(|e| anyhow::anyhow!("capture Metal single-token paged decode ICB graph: {e}"))?;

    Ok(MetalSingleTokenPagedDecodeIcbGraph {
        captured,
        kv_args,
        attn_args,
    })
}

fn merge_metal_graph_resources(
    resources: impl IntoIterator<Item = MetalGraphResourceRef>,
) -> Result<Vec<kiln_graph_metal::MetalGraphResource>> {
    let mut merged: Vec<MetalGraphResourceRef> = Vec::new();
    for resource in resources {
        if let Some(existing) = merged.iter_mut().find(|r| r.buffer == resource.buffer) {
            existing.usage |= resource.usage;
        } else {
            merged.push(resource);
        }
    }
    Ok(merged
        .into_iter()
        .map(|resource| kiln_graph_metal::MetalGraphResource::new(resource.buffer, resource.usage))
        .collect())
}

#[derive(Debug)]
pub struct MetalBackend {
    /// The kt Metal device this backend dispatches on. (#1082: the
    /// formerly-retained candle `device` field is gone — every trait
    /// method is kt-native, so no candle handle is held.)
    device_kt: kiln_tensor::Device,
    /// Cached at construction to keep env-var reads off per-token support gates.
    disable: MetalKernelDisables,
}

impl MetalBackend {
    pub fn new(device: kiln_tensor::Device) -> Self {
        debug_assert!(
            matches!(device, kiln_tensor::Device::Metal(_)),
            "MetalBackend created on non-Metal device"
        );
        Self {
            device_kt: device,
            disable: MetalKernelDisables::from_env(),
        }
    }

    pub fn training_capabilities_static() -> TrainingCapabilities {
        metal_training::training_capabilities_static()
    }
}

/// Compile Kiln's custom Metal library and compute pipelines ahead of the
/// first forward pass. Candle kernels still compile lazily inside Candle, but
/// this removes Kiln-owned pipeline setup from the first prewarm/request.
pub fn precompile_custom_kernels(device: &kiln_tensor::Device) -> Result<()> {
    // #1082: kt-native prewarm — derive the companion and drive the pipeline
    // getters through `&dyn MetalPipelineHost` (no candle device).
    let kiln_tensor::Device::Metal(idx) = device else {
        return Ok(());
    };
    let companion = kiln_tensor::primary_metal_companion(*idx)
        .map_err(|e| anyhow::anyhow!("precompile_custom_kernels: companion: {e}"))?;
    let metal_device: &MetalCompanion = &companion;

    metal_shared_library(metal_device)?;
    metal_rms_norm_pipeline(metal_device)?;
    metal_rotary_qk_pipeline(metal_device)?;
    metal_gdn_qk_norm_pipeline(metal_device)?;
    metal_gdn_qk_norm_gqa_pipeline(metal_device)?;
    metal_gdn_decode_qkv_conv_norm_pipeline(metal_device)?;
    metal_gdn_prefill_qkv_conv_split_pipeline(metal_device)?;
    metal_gdn_gates_pipeline(metal_device)?;
    metal_gdn_gates_decay_pipeline(metal_device)?;
    metal_gdn_gates_decay_ab_pipeline(metal_device)?;
    metal_gdn_decode_gates_recurrent_pipeline(metal_device)?;
    metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(metal_device)?;
    metal_gated_rms_norm_pipeline(metal_device)?;
    metal_gdn_in_proj_pipeline(metal_device)?;
    metal_gdn_recurrent_pipeline(metal_device)?;
    metal_gdn_recurrent_prefill_head_last_pipeline(metal_device)?;
    metal_gdn_recurrent_prefill_head_last_decay_pipeline(metal_device)?;
    metal_gdn_forward_substitution_pipeline(metal_device)?;
    metal_gdn_chunk_prep_pipeline(metal_device)?;
    metal_gdn_full_chunk_forward_pipeline(metal_device)?;
    metal_conv1d_prefill_pipeline(metal_device)?;
    metal_conv1d_update_pipeline(metal_device)?;
    metal_lm_head_pipeline(metal_device)?;
    if !metal_lm_head_argmax_disabled() {
        metal_lm_head_argmax_pipeline(metal_device)?;
        if !metal_lm_head_argmax_gpu_reduce_disabled() {
            metal_lm_head_argmax_reduce_pipeline(metal_device)?;
        }
    }
    if !metal_lm_head_argmax_rows_disabled() {
        metal_lm_head_argmax_batch_pipeline(metal_device)?;
        if !metal_lm_head_argmax_gpu_reduce_disabled() {
            metal_lm_head_argmax_reduce_batch_pipeline(metal_device)?;
        }
    }
    if !metal_lm_head_sample_disabled() {
        metal_lm_head_sample_pipeline(metal_device)?;
        metal_lm_head_sample_reduce_pipeline(metal_device)?;
    }
    if !metal_mlp_gate_up_fusion_disabled() {
        metal_mlp_gate_up_pipeline(metal_device)?;
        if !metal_mlp_gate_up_serial_dedicated_disabled() {
            metal_mlp_gate_up_serial_pipeline(metal_device)?;
        }
    }
    metal_mlp_silu_mul_pipeline(metal_device)?;
    if !metal_attn_gate_fusion_disabled() {
        metal_attn_gate_sigmoid_mul_pipeline(metal_device)?;
    }
    if !metal_transposed_coop_gemv_disabled() {
        let default_tile = metal_transposed_coop_gemv_default_tile();
        metal_transposed_coop_gemv_pipeline(metal_device, default_tile)?;
        metal_transposed_coop_gemv_batch_pipeline(metal_device)?;
        if !metal_transposed_coop_gemv_row_quad_tile8_disabled() {
            if !metal_transposed_coop_gemv_row_triple_tile8_disabled() {
                metal_transposed_coop_gemv_batch_row_triple_tile8_pipeline(metal_device)?;
            }
            metal_transposed_coop_gemv_batch_row_quad_tile8_pipeline(metal_device)?;
        }
        if default_tile != MetalTransposedCoopGemvTile::Tile4 {
            metal_transposed_coop_gemv_pipeline(metal_device, MetalTransposedCoopGemvTile::Tile4)?;
        }
        if !metal_transposed_coop_gemv_tile16_disabled() {
            metal_transposed_coop_gemv_pipeline(metal_device, MetalTransposedCoopGemvTile::Tile16)?;
        }
        if !metal_fused_qkv_proj_disabled() {
            metal_fused_qkv_transposed_coop_gemv_pipeline(metal_device)?;
        }
    }
    if !metal_lora_delta_decode_disabled() {
        metal_lora_hidden_decode_pipeline(metal_device)?;
        metal_lora_add_decode_pipeline(metal_device)?;
    }
    metal_paged_kv_head_major_read_pipeline(metal_device)?;
    metal_paged_kv_head_major_read_append_token_major_pipeline(metal_device)?;
    if !metal_paged_attn_decode_contiguous_disabled() {
        metal_paged_attn_decode_contiguous_pipeline(metal_device)?;
        metal_paged_attn_decode_contiguous_batch_pipeline(metal_device)?;
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline(metal_device)?;
    }
    if !metal_paged_kv_write_token_major_disabled() {
        metal_paged_kv_write_token_major_pipeline(metal_device)?;
        metal_paged_kv_write_token_major_batch_pipeline(metal_device)?;
    }
    Ok(())
}

// #1082 DoD-101/102: BackendRuntime decode methods flipped to kt; metal/vulkan impls need matching flip when their builds are restored.
impl BackendRuntime for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn device(&self) -> kiln_tensor::Device {
        self.device_kt
    }

    fn training_capabilities(&self) -> TrainingCapabilities {
        Self::training_capabilities_static()
    }

    fn training_precision_policy(&self) -> TrainingPrecisionPolicy {
        metal_training::training_precision_policy()
    }

    // ------------------------------------------------------------------
    // Resident-activation hooks (#1082) — Metal analog of the Vulkan
    // registry. The registry tracks membership only (the kt tensor already
    // owns its GPU buffer); `dispatch_adamw_step` runs a fused on-device
    // AdamW that updates param/m/v in place. Same Ok(true)/Ok(false) and
    // register/has/update/evict/resolve semantics as Vulkan.
    // ------------------------------------------------------------------

    fn supports_resident_activation(&self) -> bool {
        true
    }

    fn register_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        metal_residency::register_resident_activation(tensor)
    }

    fn has_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> bool {
        metal_residency::has_resident_activation(tensor)
    }

    fn update_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        metal_residency::update_resident_activation(tensor)
    }

    fn evict_resident_activation(&self, tensor: &kiln_tensor::Tensor) {
        metal_residency::evict_resident_activation(tensor);
    }

    fn resolve_resident_activation(
        &self,
        tensor: &kiln_tensor::Tensor,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        metal_residency::resolve_resident_activation(tensor, shape, dtype)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_adamw_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        first_moment: &kiln_tensor::Tensor,
        second_moment: &kiln_tensor::Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<bool> {
        // All four operands must be resident: no mixed resident/host update,
        // since that would need a per-call upload and defeat on-device AdamW.
        let all_resident =
            metal_residency::all_registered(&[param, grad, first_moment, second_moment]);
        metal_training::dispatch_adamw_step(
            param,
            grad,
            first_moment,
            second_moment,
            all_resident,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        )
    }

    fn supports_linear_decode_sample(&self, top_k: u32) -> bool {
        top_k > 0 && top_k <= METAL_LM_HEAD_SAMPLE_TOP_K_MAX && !metal_lm_head_sample_disabled()
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_decode_sample(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: u64,
    ) -> Result<Option<u32>> {
        if !self.supports_linear_decode_sample(top_k) {
            return Ok(None);
        }
        if !metal_lm_head_sample_supports(x, weight_t, top_k, temperature, history_indices.len()) {
            return Ok(None);
        }
        let greedy =
            kiln_core::sampling::SamplingParams::values_are_effectively_greedy(temperature, top_k);
        let (
            history_indices,
            history_counts,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
        ) = if greedy {
            (&[][..], &[][..], 1.0f32, 0.0f32, 0.0f32)
        } else {
            (
                history_indices,
                history_counts,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
            )
        };
        let token = metal_lm_head_sample_bf16(
            x,
            weight_t,
            history_indices,
            history_counts,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            temperature.max(f32::MIN_POSITIVE),
            if greedy { 1 } else { top_k },
            top_p,
            min_p,
            seed,
        )
        .context("metal fused linear_decode_sample")?;
        Ok(Some(token))
    }

    fn supports_linear_decode_sample_batch(&self, top_k: &[u32], temperatures: &[f32]) -> bool {
        if top_k.len() != temperatures.len() || top_k.is_empty() || metal_lm_head_sample_disabled()
        {
            return false;
        }
        let mut has_sampled_row = false;
        for (&k, &temp) in top_k.iter().zip(temperatures.iter()) {
            let greedy = temp == 0.0 || (k == 1 && temp.is_finite() && temp > 0.0);
            if greedy {
                continue;
            }
            if !(temp.is_finite() && temp > 0.0 && k > 0 && k <= METAL_LM_HEAD_SAMPLE_TOP_K_MAX) {
                return false;
            }
            has_sampled_row = true;
        }
        has_sampled_row
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_decode_sample_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
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
        if !self.supports_linear_decode_sample_batch(top_k, temperatures) {
            return Ok(None);
        }
        let Ok((batch, seq_len, _hidden)) = x.dims3() else {
            return Ok(None);
        };
        if batch == 0 || seq_len != 1 {
            return Ok(None);
        }
        if repetition_penalties.len() != batch
            || presence_penalties.len() != batch
            || frequency_penalties.len() != batch
            || temperatures.len() != batch
            || top_k.len() != batch
            || top_p.len() != batch
            || min_p.len() != batch
            || seeds.len() != batch
            || history_rows.len() != history_indices.len()
            || history_rows.len() != history_counts.len()
        {
            return Ok(None);
        }

        let mut histories = vec![Vec::<(u32, u32)>::new(); batch];
        for ((&row, &idx), &count) in history_rows
            .iter()
            .zip(history_indices.iter())
            .zip(history_counts.iter())
        {
            let row = row as usize;
            if row >= batch {
                return Ok(None);
            }
            histories[row].push((idx, count));
        }
        for row_history in histories.iter_mut() {
            row_history.sort_by_key(|&(idx, _)| idx);
        }

        let mut tokens = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_x = x.narrow(0, row, 1)?.contiguous()?;
            let greedy = kiln_core::sampling::SamplingParams::values_are_effectively_greedy(
                temperatures[row],
                top_k[row],
            );
            let (row_indices, row_counts): (Vec<u32>, Vec<u32>) = if greedy {
                (Vec::new(), Vec::new())
            } else {
                histories[row].iter().copied().unzip()
            };
            let row_temperature = if temperatures[row] == 0.0 {
                1.0
            } else {
                temperatures[row]
            };
            let row_top_k = if greedy { 1 } else { top_k[row] };
            if !metal_lm_head_sample_supports(
                &row_x,
                weight_t,
                row_top_k,
                row_temperature,
                row_indices.len(),
            ) {
                return Ok(None);
            }
            let token = metal_lm_head_sample_bf16(
                &row_x,
                weight_t,
                &row_indices,
                &row_counts,
                if greedy {
                    1.0
                } else {
                    repetition_penalties[row]
                },
                if greedy { 0.0 } else { presence_penalties[row] },
                if greedy {
                    0.0
                } else {
                    frequency_penalties[row]
                },
                row_temperature,
                row_top_k,
                top_p[row],
                min_p[row],
                seeds[row],
            )
            .context("metal fused batched linear_decode_sample row")?;
            tokens.push(token);
        }
        Ok(Some(tokens))
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        std::env::var(DISABLE_METAL_SDPA).is_err()
    }

    fn supports_flash_attn_prefill_head_major(&self) -> bool {
        std::env::var(DISABLE_METAL_SDPA).is_err()
    }

    // Note: keep `supports_*` returning true so the planner picks the SDPA
    // path; the per-call gate inside the kernel functions then decides
    // whether the *specific* shape is safe and silently falls back to the
    // naive softmax+matmul path when it isn't.

    fn supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    fn flash_attn_paged_decode_contiguous(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082 forward-flip: trait surface is kt; bridge each kt arg to a
        // candle CPU local (host round-trip, matching this backend's
        // CPU-resident model), then delegate to the unchanged candle helper.
        if !metal_paged_attn_decode_contiguous_supports(
            q,
            k_pool,
            v_pool,
            start_slot,
            total_seqlen_k,
        ) {
            return Ok(None);
        }
        let out = metal_paged_attn_decode_contiguous_bf16_d256(
            q,
            k_pool,
            v_pool,
            start_slot,
            total_seqlen_k,
            softmax_scale,
        )
        .context("metal contiguous paged decode attention failed")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slots: &kiln_tensor::Tensor,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !metal_paged_attn_decode_contiguous_batch_supports(
            q,
            k_pool,
            v_pool,
            start_slots,
            total_seqlen_k,
        ) {
            return Ok(None);
        }
        let out = metal_paged_attn_decode_contiguous_batch_bf16_d256(
            q,
            k_pool,
            v_pool,
            start_slots,
            total_seqlen_k,
            softmax_scale,
        )
        .context("metal contiguous paged batch decode attention failed")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
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
        if !causal
            || !metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
                q,
                k_pool,
                v_pool,
                block_table,
                seqused_k,
                max_seqlen_k,
                page_block_size,
            )
        {
            return Ok(None);
        }
        let out = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
        )
        .context("metal dyn-seqlen paged batch decode attention failed")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        graph_outputs: Option<(&kiln_tensor::Tensor, &kiln_tensor::Tensor)>,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !causal
            || !metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
                q,
                k_pool,
                v_pool,
                block_table,
                seqused_k,
                max_seqlen_k,
                page_block_size,
            )
        {
            return Ok(None);
        }

        if let Some((out, _softmax_lse)) = graph_outputs {
            metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_into(
                q,
                k_pool,
                v_pool,
                block_table,
                seqused_k,
                out,
                max_seqlen_k,
                page_block_size,
                softmax_scale,
            )
            .context("metal dyn-seqlen paged batch decode attention into graph output failed")?;
            return Ok(Some(out.clone()));
        }

        self.flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
    }

    fn supports_paged_kv_head_major_read(&self) -> bool {
        true
    }

    fn supports_paged_kv_head_major_read_append_token_major(&self) -> bool {
        true
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        !self.disable.conv1d_prefill
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        !self.disable.conv1d_update
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_full_chunk_forward_head_last(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_recurrent_prefill_head_last(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn supports_gdn_gates(&self) -> bool {
        !self.disable.gdn_gates
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        !self.disable.gated_rms_norm
    }

    fn flash_attn_prefill(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if std::env::var(DISABLE_METAL_SDPA).is_ok() {
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
        // Last-axis index via kt-native `rank()` arithmetic so this site no
        // longer names a `candle_core::D::Minus1`-style selector through the
        // chokepoint module (#1082 chokepoint cleanup).
        // `q` here is always at least rank 3 (batch, seq, hidden); the
        // subtraction matches the previous `D::Minus1` semantics.
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

    fn flash_attn_prefill_head_major(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if std::env::var(DISABLE_METAL_SDPA).is_ok() {
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

    /// Gather K/V from the paged pool via `index_select` on the block table,
    /// then call candle's vectorized SDPA (single-query path). The gather
    /// replaces the slow materializing `paged_cache.read` +
    /// naive-softmax+matmul fallback — same result, one fused kernel.
    fn flash_attn_paged_decode(
        &self,
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
        // construction (both come from the same forward config), so only q
        // needs checking. Guards read the kt arg directly, BEFORE the candle
        // bridges (#1082 forward-flip).
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

        // The block_table is identical across all 8 full-attention layers in
        // a decode step, but the trait forces us to re-flatten it per call.
        // Threading a pre-flattened handle through the trait would save
        // ~8× redundant flattens per token; defer until the signature can
        // grow a cache parameter.
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
        let k_sdpa = k_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // [1, num_kv_heads, total_seqlen_k, head_dim]
        let v_sdpa = v_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;

        let out =
            kiln_tensor::metal_sdpa_last_axis(&q_sdpa, &k_sdpa, &v_sdpa, softmax_scale, causal)
                .context("kt-native metal paged sdpa (decode) failed")?;

        // Back to [1, 1, num_heads, head_dim].
        let out = out.transpose(1, 2)?.contiguous()?;
        debug_assert_eq!(out.dims(), &[1, 1, num_heads, head_dim]);
        Ok(Some(out))
    }

    fn paged_kv_head_major_read(
        &self,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        seq_len: usize,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        if !metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, seq_len) {
            return Ok(None);
        }
        let (k_out, v_out) =
            metal_paged_kv_head_major_read_bf16(k_pool, v_pool, start_slot, seq_len)
                .context("metal paged_kv_head_major_read failed")?;
        Ok(Some((k_out, v_out)))
    }

    fn paged_kv_head_major_read_append_token_major(
        &self,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        prefix_len: usize,
        k_tail: &kiln_tensor::Tensor,
        v_tail: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        if !metal_paged_kv_head_major_read_append_token_major_supports(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        ) {
            return Ok(None);
        }
        let (k_out, v_out) = metal_paged_kv_head_major_read_append_token_major_bf16(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        )
        .context("metal paged_kv_head_major_read_append_token_major failed")?;
        Ok(Some((k_out, v_out)))
    }

    fn causal_conv1d_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly; `conv_state` is mutated
        // in place through its shared UMA buffer (no candle bridge).
        if self.disable.conv1d_prefill
            || !metal_conv1d_prefill_supports(x, weight, conv_state, kernel_size)
        {
            return Ok(None);
        }
        let out = metal_causal_conv1d_prefill_bf16_f32_k4(x, weight, conv_state, kernel_size)
            .context("metal causal_conv1d_prefill kernel failed")?;
        Ok(Some(out))
    }

    fn causal_conv1d_update(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly; `conv_state` is mutated
        // in place through its shared UMA buffer (no candle bridge).
        if self.disable.conv1d_update
            || !metal_conv1d_update_supports(x, weight, conv_state, kernel_size)
        {
            return Ok(None);
        }
        let out = metal_causal_conv1d_update_bf16_f32_k4(x, weight, conv_state, kernel_size)
            .context("metal causal_conv1d_update kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_forward_substitution(
        &self,
        a_strict: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_forward_substitution
            || !metal_gdn_forward_substitution_supports(a_strict, v_prime, beta)
        {
            return Ok(None);
        }
        let out = match a_strict.dtype() {
            kiln_tensor::DType::BF16 => {
                metal_gdn_forward_substitution_bf16(a_strict, v_prime, beta)
            }
            kiln_tensor::DType::F32 => metal_gdn_forward_substitution_f32(a_strict, v_prime, beta),
            other => anyhow::bail!("unsupported metal gdn_forward_substitution dtype {other:?}"),
        }
        .context("metal gdn_forward_substitution kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_recurrent_step(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_recurrent || !metal_gdn_recurrent_supports(q, k, v, beta, g, state) {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_step kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_chunk_prep(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_forward_substitution
            || !metal_gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s)
        {
            return Ok(None);
        }
        let (o0, o1, o2, o3, o4, o5) = metal_gdn_chunk_prep_bf16(g, v, kkt, qkt, ks_entry, q_s)
            .context("metal gdn_chunk_prep kernel failed")?;
        Ok(Some((o0, o1, o2, o3, o4, o5)))
    }

    fn gdn_full_chunk_forward(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_forward_substitution
            || !metal_gdn_full_chunk_forward_supports(
                g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
            )
        {
            return Ok(None);
        }
        let out =
            metal_gdn_full_chunk_forward_bf16(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state)
                .context("metal gdn_full_chunk_forward kernel failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn gdn_full_chunk_forward_head_last_into(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        out: &kiln_tensor::Tensor,
        t_start: usize,
        seq_len: usize,
    ) -> Result<bool> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        // `out` is a caller-owned output buffer written in place by the kernel
        // through its shared UMA buffer; `state` is likewise mutated in place.
        if self.disable.gdn_forward_substitution
            || !metal_gdn_full_chunk_forward_head_last_supports(
                g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
            )
        {
            return Ok(false);
        }
        metal_gdn_full_chunk_forward_head_last_into_bf16(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
        )
        .context("metal gdn_full_chunk_forward_head_last_into kernel failed")?;
        Ok(true)
    }

    fn gdn_recurrent_prefill_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_recurrent
            || !metal_gdn_recurrent_prefill_head_last_supports(q, k, v, beta, g, state)
        {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_prefill_head_last_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_prefill_head_last kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_recurrent_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        // The recurrent state is mutated in place through its shared UMA
        // buffer (the &mut state is the same tensor the caller holds).
        if self.disable.gdn_recurrent
            || !metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, g, state)
        {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_prefill_native_head_last_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_prefill_native_head_last kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_in_proj_decode(
        &self,
        x: &kiln_tensor::Tensor,
        in_proj_qkv_t: &kiln_tensor::Tensor,
        in_proj_z_t: &kiln_tensor::Tensor,
        in_proj_a_t: &kiln_tensor::Tensor,
        in_proj_b_t: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_in_proj
            || !metal_gdn_in_proj_decode_supports(
                x,
                in_proj_qkv_t,
                in_proj_z_t,
                in_proj_a_t,
                in_proj_b_t,
            )
        {
            return Ok(None);
        }
        let (o0, o1, o2, o3) =
            metal_gdn_in_proj_decode_bf16(x, in_proj_qkv_t, in_proj_z_t, in_proj_a_t, in_proj_b_t)
                .context("metal gdn_in_proj_decode kernel failed")?;
        Ok(Some((o0, o1, o2, o3)))
    }

    fn gdn_gates(
        &self,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_gates || !metal_gdn_gates_supports(a, b, a_log, dt_bias) {
            return Ok(None);
        }
        let (beta, g) =
            metal_gdn_gates_bf16(a, b, a_log, dt_bias).context("metal gdn_gates kernel failed")?;
        Ok(Some((beta, g)))
    }

    fn gdn_gated_rms_norm(
        &self,
        x: &kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gated_rms_norm || !metal_gated_rms_norm_supports(x, z, weight) {
            return Ok(None);
        }
        let out = metal_gated_rms_norm_bf16(x, z, weight, eps as f32)
            .context("metal gated_rms_norm kernel failed")?;
        Ok(Some(out))
    }
}

fn metal_conv1d_prefill_supports(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
) -> bool {
    if kernel_size != 4 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_)) {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    let Ok((batch, channels, seq_len)) = x.dims3() else {
        return false;
    };
    if seq_len <= 1 {
        return false;
    }
    let weight_ok = match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    };
    if !weight_ok {
        return false;
    }
    conv_state
        .dims3()
        .is_ok_and(|(b, c, k)| (b, c, k) == (batch, channels, kernel_size - 1))
}

fn metal_conv1d_update_supports(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
) -> bool {
    if kernel_size != 4 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_)) {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    let Ok((batch, channels, seq_len)) = x.dims3() else {
        return false;
    };
    if seq_len != 1 {
        return false;
    }
    let weight_ok = match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    };
    if !weight_ok {
        return false;
    }
    conv_state
        .dims3()
        .is_ok_and(|(b, c, k)| (b, c, k) == (batch, channels, kernel_size - 1))
}

fn metal_gdn_gates_supports(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if a.shape() != b.shape() {
        return false;
    }
    let Some(&nv) = a.dims().last() else {
        return false;
    };
    if nv == 0 || nv > 256 {
        return false;
    }
    if a_log.dims() != [nv] || dt_bias.dims() != [nv] {
        return false;
    }
    a.elem_count() > 0
}

pub(crate) fn metal_gdn_gates_decay_supports(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> bool {
    !metal_gdn_prefill_decay_recurrent_disabled() && metal_gdn_gates_supports(a, b, a_log, dt_bias)
}

pub(crate) fn metal_gdn_prefill_ab_in_proj_supports(
    x: &kiln_tensor::Tensor,
    in_proj_ab_t: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_prefill_ab_in_proj_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || in_proj_ab_t.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(in_proj_ab_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !in_proj_ab_t.is_contiguous() || nv == 0 {
        return false;
    }
    let Ok((_batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Some(ab_dim) = nv.checked_mul(2) else {
        return false;
    };
    seq_len > 1 && in_proj_ab_t.dims() == [hidden, ab_dim]
}

pub(crate) fn metal_gdn_gates_decay_ab_supports(
    ab: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_prefill_ab_in_proj_disabled() || metal_gdn_prefill_decay_recurrent_disabled() {
        return false;
    }
    if ab.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(ab.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !ab.is_contiguous() || nv == 0 || nv > 256 {
        return false;
    }
    let Ok((batch, seq_len, channels)) = ab.dims3() else {
        return false;
    };
    let Some(ab_dim) = nv.checked_mul(2) else {
        return false;
    };
    let Some(total) = batch.checked_mul(seq_len).and_then(|n| n.checked_mul(nv)) else {
        return false;
    };
    channels == ab_dim
        && total > 0
        && total <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && a_log.dims() == [nv]
        && dt_bias.dims() == [nv]
}

fn metal_gdn_forward_substitution_supports(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(a_strict.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_prime.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    let dtype = a_strict.dtype();
    if (dtype != kiln_tensor::DType::BF16 && dtype != kiln_tensor::DType::F32)
        || v_prime.dtype() != dtype
        || beta.dtype() != dtype
    {
        return false;
    }
    let Ok((batch, heads, chunk, chunk_cols)) = a_strict.dims4() else {
        return false;
    };
    let Ok((b_v, h_v, c_v, dv)) = v_prime.dims4() else {
        return false;
    };
    let Ok((b_b, h_b, c_b)) = beta.dims3() else {
        return false;
    };

    chunk == chunk_cols
        && (b_v, h_v, c_v) == (batch, heads, chunk)
        && (b_b, h_b, c_b) == (batch, heads, chunk)
        && chunk > 0
        && chunk <= 64
        && dv > 0
        && dv <= 128
}

fn metal_gdn_chunk_prep_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(kkt.device(), kiln_tensor::Device::Metal(_))
        || !matches!(qkt.device(), kiln_tensor::Device::Metal(_))
        || !matches!(ks_entry.device(), kiln_tensor::Device::Metal(_))
        || !matches!(q_s.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if g.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || kkt.dtype() != kiln_tensor::DType::BF16
        || qkt.dtype() != kiln_tensor::DType::BF16
        || ks_entry.dtype() != kiln_tensor::DType::BF16
        || q_s.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    // Full chunks only: this keeps speculative multi-token verification on
    // the already-stable Candle path while accelerating long prompt prefill.
    if chunk != 64 {
        return false;
    }
    let Ok((b_v, h_v, c_v, dv)) = v.dims4() else {
        return false;
    };
    if (b_v, h_v, c_v) != (batch, heads, chunk) || dv == 0 || dv > 128 {
        return false;
    }
    let Ok((b_kkt, h_kkt, c_kkt, c2_kkt)) = kkt.dims4() else {
        return false;
    };
    if (b_kkt, h_kkt, c_kkt, c2_kkt) != (batch, heads, chunk, chunk) {
        return false;
    }
    let Ok((b_qkt, h_qkt, c_qkt, c2_qkt)) = qkt.dims4() else {
        return false;
    };
    if (b_qkt, h_qkt, c_qkt, c2_qkt) != (batch, heads, chunk, chunk) {
        return false;
    }
    let Ok((b_ks, h_ks, c_ks, dv_ks)) = ks_entry.dims4() else {
        return false;
    };
    if (b_ks, h_ks, c_ks, dv_ks) != (batch, heads, chunk, dv) {
        return false;
    }
    let Ok((b_qs, h_qs, c_qs, dv_qs)) = q_s.dims4() else {
        return false;
    };
    (b_qs, h_qs, c_qs, dv_qs) == (batch, heads, chunk, dv)
}

fn metal_gdn_full_chunk_forward_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !metal_gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s)
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if beta.dtype() != kiln_tensor::DType::BF16
        || k_t.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    let Ok((_, _, _, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, h_beta, c_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_kt, h_kt, dk, c_kt)) = k_t.dims4() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_beta, h_beta, c_beta) == (batch, heads, chunk)
        && (b_kt, h_kt, c_kt) == (batch, heads, chunk)
        && (b_state, h_state, dk_state, dv_state) == (batch, heads, dk, dv)
        && chunk == 64
        && dk > 0
        && dk <= 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

fn metal_gdn_full_chunk_forward_strided_inputs_support(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    heads: usize,
) -> bool {
    fn flat_batch_head_ok(stride: &[usize], heads: usize) -> bool {
        stride.len() >= 2
            && stride[1] > 0
            && heads
                .checked_mul(stride[1])
                .is_some_and(|expected| stride[0] == expected)
    }

    fn stride_u32_ok(stride: usize) -> bool {
        stride > 0 && stride <= u32::MAX as usize
    }

    let g_stride = g.layout().strides();
    let v_stride = v.layout().strides();
    let beta_stride = beta.layout().strides();
    let kt_stride = k_t.layout().strides();

    if g_stride.len() != 3 || v_stride.len() != 4 || beta_stride.len() != 3 || kt_stride.len() != 4
    {
        return false;
    }
    if !flat_batch_head_ok(g_stride, heads)
        || !flat_batch_head_ok(v_stride, heads)
        || !flat_batch_head_ok(beta_stride, heads)
        || !flat_batch_head_ok(kt_stride, heads)
    {
        return false;
    }

    // This path only needs to support a time-window narrow. Keep the value
    // dimension contiguous so the per-value lane remains coalesced.
    v_stride[3] == 1
        && [
            g_stride[1],
            g_stride[2],
            v_stride[1],
            v_stride[2],
            v_stride[3],
            beta_stride[1],
            beta_stride[2],
            kt_stride[1],
            kt_stride[2],
            kt_stride[3],
        ]
        .into_iter()
        .all(stride_u32_ok)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn metal_gdn_full_chunk_forward_head_last_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    t_start: usize,
    seq_len: usize,
) -> bool {
    if !metal_gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state)
        || !matches!(out.device(), kiln_tensor::Device::Metal(_))
        || out.dtype() != kiln_tensor::DType::BF16
        || !out.is_contiguous()
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    let Ok((_, _, _, dv)) = v.dims4() else {
        return false;
    };
    out.dims4()
        .is_ok_and(|dims| dims == (batch, seq_len, heads, dv))
        && chunk == 64
        && t_start <= seq_len
        && t_start + chunk <= seq_len
        && metal_gdn_full_chunk_forward_strided_inputs_support(g, v, beta, k_t, heads)
}

fn metal_gdn_recurrent_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, dk)) = q.dims3() else {
        return false;
    };
    let Ok((b_k, h_k, dk_k)) = k.dims3() else {
        return false;
    };
    let Ok((b_v, h_v, dv)) = v.dims3() else {
        return false;
    };
    let Ok((b_b, h_b)) = beta.dims2() else {
        return false;
    };
    let Ok((b_g, h_g)) = g.dims2() else {
        return false;
    };
    let Ok((b_s, h_s, dk_s, dv_s)) = state.dims4() else {
        return false;
    };
    (b_k, h_k, dk_k) == (batch, heads, dk)
        && (b_v, h_v) == (batch, heads)
        && (b_b, h_b) == (batch, heads)
        && (b_g, h_g) == (batch, heads)
        && (b_s, h_s, dk_s, dv_s) == (batch, heads, dk, dv)
        && dk <= 256
        && dv <= 1024
}

const METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN: usize = 2048;

fn metal_gdn_recurrent_prefill_head_last_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, q_heads, seq_len, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, h_k, t_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, v_heads, t_v, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, h_beta, t_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_g, h_g, t_g)) = g.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_k, h_k, t_k, dk_k) == (batch, q_heads, seq_len, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_beta, h_beta, t_beta) == (batch, v_heads, seq_len)
        && (b_g, h_g, t_g) == (batch, v_heads, seq_len)
        && (b_state, h_state, dk_state, dv_state) == (batch, v_heads, dk, dv)
        && v_heads >= q_heads
        && v_heads % q_heads == 0
        && seq_len > 1
        && seq_len <= METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN
        && dk == 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

fn metal_gdn_recurrent_prefill_native_head_last_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, t_beta, h_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_g, t_g, h_g)) = g.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_beta, t_beta, h_beta) == (batch, seq_len, value_heads)
        && (b_g, t_g, h_g) == (batch, seq_len, value_heads)
        && (b_state, h_state, dk_state, dv_state) == (batch, value_heads, dk, dv)
        && value_heads >= q_heads
        && value_heads % q_heads == 0
        && seq_len >= 1
        && seq_len <= METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN
        && dk == 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

pub(crate) fn metal_gdn_recurrent_prefill_native_head_last_decay_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    decay: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    !metal_gdn_prefill_decay_recurrent_disabled()
        && metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, decay, state)
}

pub(crate) fn metal_gdn_decode_gates_recurrent_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if metal_gdn_decode_gates_recurrent_disabled()
        || !matches!(q.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_a, t_a, h_a)) = a.dims3() else {
        return false;
    };
    let Ok((b_b, t_b, h_b)) = b.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    let Some(batch_heads) = batch.checked_mul(value_heads) else {
        return false;
    };
    batch > 0
        && seq_len == 1
        && (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_a, t_a, h_a) == (batch, seq_len, value_heads)
        && (b_b, t_b, h_b) == (batch, seq_len, value_heads)
        && a_log.dims() == [value_heads]
        && dt_bias.dims() == [value_heads]
        && (b_state, h_state, dk_state, dv_state) == (batch, value_heads, dk, dv)
        && q_heads > 0
        && value_heads > q_heads
        && value_heads % q_heads == 0
        && dk == 128
        && dv == 128
        && batch_heads <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && value_heads <= u32::MAX as usize
        && state.is_contiguous()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_rmsnorm_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if metal_gdn_decode_gates_recurrent_rmsnorm_disabled() {
        return false;
    }
    if !metal_gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state) {
        return false;
    }
    metal_gated_rms_norm_supports(v, z, weight)
}

fn metal_gated_rms_norm_supports(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(z.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    // x/z are BF16 activations. The norm weight follows the model dtype, which
    // for Qwen3.5 GDN is BF16 (same as the CUDA `gdn_gated_rms_norm_supports_kt`
    // contract). Accept F32 too for callers that pre-promote — the kernel casts
    // the weight to F32 internally either way.
    if x.dtype() != kiln_tensor::DType::BF16
        || z.dtype() != kiln_tensor::DType::BF16
        || !matches!(
            weight.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        )
    {
        return false;
    }
    let Ok((batch, seq_len, heads, hidden)) = x.dims4() else {
        return false;
    };
    let Ok((z_batch, z_seq_len, z_heads, z_hidden)) = z.dims4() else {
        return false;
    };
    if (z_batch, z_seq_len, z_heads, z_hidden) != (batch, seq_len, heads, hidden) {
        return false;
    }
    weight.dims() == [hidden] && hidden <= 1024
}

pub(crate) fn metal_rms_norm_supports(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if metal_rms_norm_disabled() {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    let Some(hidden) = x.dims().last().copied() else {
        return false;
    };
    x.rank() >= 1 && weight.dims() == [hidden] && hidden <= 8192
}

pub(crate) fn metal_gdn_qk_norm_supports(q: &kiln_tensor::Tensor, k: &kiln_tensor::Tensor) -> bool {
    if metal_gdn_qk_norm_disabled() {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::F32 || k.dtype() != kiln_tensor::DType::F32 {
        return false;
    }
    let Some(hidden) = q.dims().last().copied() else {
        return false;
    };
    q.rank() >= 1 && q.dims() == k.dims() && hidden <= 8192
}

pub(crate) fn metal_gdn_qk_norm_gqa_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_qk_norm_disabled() {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::F32 || k.dtype() != kiln_tensor::DType::F32 {
        return false;
    }
    let Ok((_, _, nk, hidden)) = q.dims4() else {
        return false;
    };
    q.dims() == k.dims()
        && nk > 0
        && nv > nk
        && nv % nk == 0
        && hidden <= 8192
        && nv <= u32::MAX as usize
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_qkv_conv_norm_supports(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> bool {
    if metal_gdn_qkv_conv_norm_disabled() || metal_gdn_qk_norm_disabled() {
        return false;
    }
    if kernel_size != 4 || dk != 128 || dv != 128 {
        return false;
    }
    if !matches!(mixed_qkv.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
        || !matches!(conv_state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if mixed_qkv.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    let Ok((batch, seq_len, channels)) = mixed_qkv.dims3() else {
        return false;
    };
    let qk_dim = nk.saturating_mul(dk);
    let v_dim = nv.saturating_mul(dv);
    let Some(expected_channels) = qk_dim.checked_mul(2).and_then(|n| n.checked_add(v_dim)) else {
        return false;
    };
    let weight_ok = match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    };
    let Some(rows) = nk
        .checked_add(nk)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_mul(batch))
    else {
        return false;
    };
    batch > 0
        && seq_len == 1
        && channels == expected_channels
        && nk > 0
        && nv > nk
        && nv % nk == 0
        && weight_ok
        && conv_state
            .dims3()
            .is_ok_and(|(b, c, k)| (b, c, k) == (batch, channels, kernel_size - 1))
        && channels <= u32::MAX as usize
        && nk <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && rows <= u32::MAX as usize
}

pub(crate) fn metal_rotary_embedding_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(cos.device(), kiln_tensor::Device::Metal(_))
        || !matches!(sin.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || cos.dtype() != kiln_tensor::DType::F32
        || sin.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    if !q.is_contiguous() || !k.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, q_heads, q_head_dim)) = q.dims4() else {
        return false;
    };
    let Ok(k_dims) = k.dims4() else {
        return false;
    };
    let half_rotary = rotary_dim / 2;
    let table_batch_stride = metal_rotary_table_batch_stride(cos, sin, batch, seq_len, half_rotary);
    let Some(total_q) = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(q_heads))
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    let Some(total_k) = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(k_dims.2))
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    k_dims.0 == batch
        && k_dims.1 == seq_len
        && k_dims.3 == head_dim
        && q_head_dim == head_dim
        && rotary_dim > 0
        && rotary_dim <= head_dim
        && rotary_dim % 2 == 0
        && table_batch_stride.is_some()
        && batch <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && k_dims.2 <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && rotary_dim <= u32::MAX as usize
        && total_q <= u32::MAX as usize
        && total_k <= u32::MAX as usize
        && total_q <= (u32::MAX as usize).saturating_sub(total_k)
}

fn metal_rotary_table_batch_stride(
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    batch: usize,
    seq_len: usize,
    half_rotary: usize,
) -> Option<usize> {
    if cos.dims() != sin.dims() {
        return None;
    }
    match cos.dims() {
        [t, r] if (*t, *r) == (seq_len, half_rotary) => Some(0),
        [b, t, r] if (*b, *t, *r) == (batch, seq_len, half_rotary) => Some(seq_len),
        _ => None,
    }
}

const METAL_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_rmsnorm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& threadgroup_width [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float sum_sq = 0.0f;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float xv = static_cast<float>(x[base + col]);
        sum_sq += xv * xv;
    }
    scratch[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt((scratch[0] / static_cast<float>(hidden)) + eps);
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float xv = static_cast<float>(x[base + col]);
        const float scale = 1.0f + static_cast<float>(weight[col]);
        out[base + col] = static_cast<bfloat>(xv * rms_inv * scale);
    }
}
"#;

const METAL_ROTARY_QK_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_rotary_qk_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const float* cos [[buffer(2)]],
    device const float* sin [[buffer(3)]],
    device bfloat* q_out [[buffer(4)]],
    device bfloat* k_out [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& q_heads [[buffer(8)]],
    constant uint& k_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& rotary_dim [[buffer(11)]],
    constant uint& total_q [[buffer(12)]],
    constant uint& total [[buffer(13)]],
    constant uint& table_batch_stride [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const bool is_q = gid < total_q;
    const uint local = is_q ? gid : gid - total_q;
    const uint heads = is_q ? q_heads : k_heads;
    device const bfloat* src = is_q ? q : k;
    device bfloat* dst = is_q ? q_out : k_out;

    const uint d = local % head_dim;
    const uint h = (local / head_dim) % heads;
    const uint t = (local / (head_dim * heads)) % seq_len;
    const uint b = local / (head_dim * heads * seq_len);
    if (b >= batch) {
        return;
    }

    if (d >= rotary_dim) {
        dst[local] = src[local];
        return;
    }

    const uint half_rotary = rotary_dim / 2;
    const bool first_half = d < half_rotary;
    const uint pair_d = first_half ? d + half_rotary : d - half_rotary;
    const uint pair_idx = ((b * seq_len + t) * heads + h) * head_dim + pair_d;
    const uint table_t = table_batch_stride == 0 ? t : b * table_batch_stride + t;
    const uint table_idx = table_t * half_rotary + (first_half ? d : pair_d);
    const float x = static_cast<float>(src[local]);
    const float y = static_cast<float>(src[pair_idx]);
    const float c = cos[table_idx];
    const float s = sin[table_idx];
    const float rotated = first_half ? (x * c - y * s) : (y * s + x * c);
    dst[local] = static_cast<bfloat>(rotated);
}
"#;

const METAL_GDN_QK_NORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_qk_norm_f32_bf16(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device bfloat* q_out [[buffer(2)]],
    device bfloat* k_out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant float& q_scale [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant uint& threadgroup_width [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float q_scratch[1024];
    threadgroup float k_scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float qv = q[base + col];
        const float kv = k[base + col];
        q_sum_sq += qv * qv;
        k_sum_sq += kv * kv;
    }
    q_scratch[tid] = q_sum_sq;
    k_scratch[tid] = k_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_scratch[tid] += q_scratch[tid + stride];
            k_scratch[tid] += k_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float q_inv = rsqrt(q_scratch[0] + eps);
    const float k_inv = rsqrt(k_scratch[0] + eps);
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const uint idx = base + col;
        q_out[idx] = static_cast<bfloat>(q[idx] * q_inv * q_scale);
        k_out[idx] = static_cast<bfloat>(k[idx] * k_inv);
    }
}

kernel void kiln_gdn_qk_norm_gqa_f32_bf16(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device bfloat* q_out [[buffer(2)]],
    device bfloat* k_out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& nk [[buffer(5)]],
    constant uint& nv [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    constant uint& gqa_ratio [[buffer(8)]],
    constant float& q_scale [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant uint& threadgroup_width [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float q_scratch[1024];
    threadgroup float k_scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint src_head = row % nk;
    const uint bt = row / nk;
    const uint base = row * hidden;
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float qv = q[base + col];
        const float kv = k[base + col];
        q_sum_sq += qv * qv;
        k_sum_sq += kv * kv;
    }
    q_scratch[tid] = q_sum_sq;
    k_scratch[tid] = k_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_scratch[tid] += q_scratch[tid + stride];
            k_scratch[tid] += k_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float q_inv = rsqrt(q_scratch[0] + eps);
    const float k_inv = rsqrt(k_scratch[0] + eps);
    const uint dst_head_base = src_head * gqa_ratio;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float q_norm = q[base + col] * q_inv * q_scale;
        const float k_norm = k[base + col] * k_inv;
        for (uint rep = 0; rep < gqa_ratio; ++rep) {
            const uint dst_head = dst_head_base + rep;
            const uint dst_idx = ((bt * nv + dst_head) * hidden) + col;
            q_out[dst_idx] = static_cast<bfloat>(q_norm);
            k_out[dst_idx] = static_cast<bfloat>(k_norm);
        }
    }
}
"#;

const METAL_GDN_DECODE_QKV_CONV_NORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_qkv_conv_norm_bf16(
    device const bfloat* mixed_qkv [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device bfloat* q_out [[buffer(3)]],
    device bfloat* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& nk [[buffer(6)]],
    constant uint& nv [[buffer(7)]],
    constant float& q_scale [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint D = 128;
    threadgroup float values[D];
    threadgroup float sum_scratch[D];

    const uint row = tgroup.x;
    const uint batch_idx = tgroup.y;
    const uint local_row = row;
    const uint qk_dim = nk * D;
    const uint channels = qk_dim + qk_dim + nv * D;

    uint channel = 0;
    bool is_q = false;
    bool is_k = false;
    bool is_v = false;
    uint src_head = 0;
    uint v_head = 0;

    if (local_row < nk) {
        is_q = true;
        src_head = local_row;
        channel = src_head * D + tid;
    } else if (local_row < nk + nk) {
        is_k = true;
        src_head = local_row - nk;
        channel = qk_dim + src_head * D + tid;
    } else {
        is_v = true;
        v_head = local_row - nk - nk;
        channel = qk_dim + qk_dim + v_head * D + tid;
    }

    const uint token_idx = batch_idx * channels + channel;
    const uint state_base = (batch_idx * channels + channel) * 3;
    const uint weight_base = channel * 4;

    const float s0 = conv_state[state_base + 0];
    const float s1 = conv_state[state_base + 1];
    const float s2 = conv_state[state_base + 2];
    const float x0 = static_cast<float>(mixed_qkv[token_idx]);
    const float acc =
        s0 * static_cast<float>(weight[weight_base + 0]) +
        s1 * static_cast<float>(weight[weight_base + 1]) +
        s2 * static_cast<float>(weight[weight_base + 2]) +
        x0 * static_cast<float>(weight[weight_base + 3]);
    const float y = acc / (1.0f + exp(-acc));

    conv_state[state_base + 0] = s1;
    conv_state[state_base + 1] = s2;
    conv_state[state_base + 2] = x0;

    if (is_v) {
        const uint out_idx = (batch_idx * nv + v_head) * D + tid;
        v_out[out_idx] = static_cast<bfloat>(y);
        return;
    }

    values[tid] = y;
    sum_scratch[tid] = y * y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = D / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(sum_scratch[0] + eps);
    const float norm = values[tid] * inv * (is_q ? q_scale : 1.0f);
    const uint dst_idx = (batch_idx * nk + src_head) * D + tid;
    if (is_q) {
        q_out[dst_idx] = static_cast<bfloat>(norm);
    } else if (is_k) {
        k_out[dst_idx] = static_cast<bfloat>(norm);
    }
}
"#;

const METAL_LM_HEAD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_lm_head_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& vocab [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vocab) {
        return;
    }

    float acc = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        acc += static_cast<float>(x[i]) * static_cast<float>(weight_t[i * vocab + gid]);
    }
    out[gid] = static_cast<bfloat>(acc);
}

kernel void kiln_lm_head_argmax_chunks_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device float* partial_scores [[buffer(2)]],
    device float* partial_indices [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& vocab [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[256];
    threadgroup float indices[256];

    const uint col = group * 256 + tid;
    float score = -INFINITY;
    float index = 0.0f;
    if (col < vocab) {
        float acc = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[i]) * static_cast<float>(weight_t[i * vocab + col]);
        }
        score = static_cast<float>(static_cast<bfloat>(acc));
        index = static_cast<float>(col);
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_scores[group] = scores[0];
        partial_indices[group] = indices[0];
    }
}

kernel void kiln_lm_head_argmax_reduce_f32(
    device const float* partial_scores [[buffer(0)]],
    device const float* partial_indices [[buffer(1)]],
    device float* final_index [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scores[1024];
    threadgroup float indices[1024];

    float score = -INFINITY;
    float index = 0.0f;
    if (tid < num_groups) {
        score = partial_scores[tid];
        index = partial_indices[tid];
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        final_index[0] = indices[0];
    }
}

kernel void kiln_lm_head_argmax_chunks_batch_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device float* partial_scores [[buffer(2)]],
    device float* partial_indices [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& vocab [[buffer(5)]],
    constant uint& num_groups [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[256];
    threadgroup float indices[256];

    const uint group = group_pos.x;
    const uint row = group_pos.y;
    const uint col = group * 256 + tid;
    float score = -INFINITY;
    float index = 0.0f;
    if (col < vocab) {
        float acc = 0.0f;
        const device bfloat* row_x = x + row * hidden;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(row_x[i]) * static_cast<float>(weight_t[i * vocab + col]);
        }
        score = static_cast<float>(static_cast<bfloat>(acc));
        index = static_cast<float>(col);
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const uint offset = row * num_groups + group;
        partial_scores[offset] = scores[0];
        partial_indices[offset] = indices[0];
    }
}

kernel void kiln_lm_head_argmax_reduce_batch_f32(
    device const float* partial_scores [[buffer(0)]],
    device const float* partial_indices [[buffer(1)]],
    device float* final_indices [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[1024];
    threadgroup float indices[1024];

    float score = -INFINITY;
    float index = 0.0f;
    if (tid < num_groups) {
        const uint offset = row * num_groups + tid;
        score = partial_scores[offset];
        index = partial_indices[offset];
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        final_indices[row] = indices[0];
    }
}

#define KILN_SAMPLE_TOPK_MAX 64

inline bool kiln_score_better(float score, float index, float best_score, float best_index) {
    return score > best_score || (score == best_score && index < best_index);
}

inline bool kiln_history_count_for_token(
    device const uint* history_indices,
    device const uint* history_counts,
    uint history_len,
    uint token,
    thread uint& count
) {
    uint lo = 0;
    uint hi = history_len;
    while (lo < hi) {
        const uint mid = lo + ((hi - lo) >> 1);
        const uint value = history_indices[mid];
        if (value < token) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < history_len && history_indices[lo] == token) {
        count = history_counts[lo];
        return true;
    }
    count = 0;
    return false;
}

inline float kiln_apply_sample_penalties(
    float score,
    uint token,
    device const uint* history_indices,
    device const uint* history_counts,
    uint history_len,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty
) {
    uint count = 0;
    if (!kiln_history_count_for_token(history_indices, history_counts, history_len, token, count)) {
        return score;
    }
    if (isfinite(repetition_penalty) && repetition_penalty > 0.0f &&
        fabs(repetition_penalty - 1.0f) > 0.00000011920929f) {
        score = score > 0.0f ? score / repetition_penalty : score * repetition_penalty;
    }
    if (isfinite(presence_penalty) && presence_penalty != 0.0f) {
        score -= presence_penalty;
    }
    if (isfinite(frequency_penalty) && frequency_penalty != 0.0f) {
        score -= frequency_penalty * static_cast<float>(count);
    }
    return score;
}

inline ulong kiln_splitmix64_next(thread ulong& state) {
    state += 0x9E3779B97F4A7C15ul;
    ulong z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

inline float kiln_uniform01_from_seed(uint seed_lo, uint seed_hi) {
    ulong state = (static_cast<ulong>(seed_hi) << 32) | static_cast<ulong>(seed_lo);
    const ulong bits = kiln_splitmix64_next(state);
    const uint mantissa = static_cast<uint>((bits >> 40) & 0xFFFFFFul);
    return static_cast<float>(mantissa) / 16777216.0f;
}

kernel void kiln_lm_head_sample_topk_chunks_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device const uint* history_indices [[buffer(2)]],
    device const uint* history_counts [[buffer(3)]],
    device float* partial_scores [[buffer(4)]],
    device float* partial_indices [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant uint& vocab [[buffer(7)]],
    constant uint& history_len [[buffer(8)]],
    constant float& repetition_penalty [[buffer(9)]],
    constant float& presence_penalty [[buffer(10)]],
    constant float& frequency_penalty [[buffer(11)]],
    constant float& inv_temperature [[buffer(12)]],
    constant uint& top_k [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[256];
    threadgroup float indices[256];

    const uint col = group * 256 + tid;
    float score = -INFINITY;
    float index = 0.0f;
    if (col < vocab) {
        float acc = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[i]) * static_cast<float>(weight_t[i * vocab + col]);
        }
        score = static_cast<float>(static_cast<bfloat>(acc));
        score = kiln_apply_sample_penalties(
            score,
            col,
            history_indices,
            history_counts,
            history_len,
            repetition_penalty,
            presence_penalty,
            frequency_penalty
        );
        score *= inv_temperature;
        index = static_cast<float>(col);
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const uint out_base = group * top_k;
        const uint k_limit = min(top_k, static_cast<uint>(KILN_SAMPLE_TOPK_MAX));
        for (uint k = 0; k < k_limit; ++k) {
            float best_score = -INFINITY;
            float best_index = 0.0f;
            uint best_pos = 0;
            for (uint i = 0; i < 256; ++i) {
                const float candidate_score = scores[i];
                const float candidate_index = indices[i];
                if (kiln_score_better(candidate_score, candidate_index, best_score, best_index)) {
                    best_score = candidate_score;
                    best_index = candidate_index;
                    best_pos = i;
                }
            }
            partial_scores[out_base + k] = best_score;
            partial_indices[out_base + k] = best_index;
            scores[best_pos] = -INFINITY;
        }
    }
}

kernel void kiln_lm_head_sample_reduce_f32(
    device const float* partial_scores [[buffer(0)]],
    device const float* partial_indices [[buffer(1)]],
    device float* final_index [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    constant float& top_p [[buffer(5)]],
    constant float& min_p [[buffer(6)]],
    constant uint& seed_lo [[buffer(7)]],
    constant uint& seed_hi [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid != 0) {
        return;
    }

    float top_scores[KILN_SAMPLE_TOPK_MAX];
    float top_indices[KILN_SAMPLE_TOPK_MAX];
    float probs[KILN_SAMPLE_TOPK_MAX];
    const uint k_limit = min(top_k, static_cast<uint>(KILN_SAMPLE_TOPK_MAX));
    for (uint i = 0; i < KILN_SAMPLE_TOPK_MAX; ++i) {
        top_scores[i] = -INFINITY;
        top_indices[i] = 0.0f;
        probs[i] = 0.0f;
    }

    const uint candidate_count = num_groups * k_limit;
    for (uint c = 0; c < candidate_count; ++c) {
        const float score = partial_scores[c];
        const float index = partial_indices[c];
        if (!isfinite(score)) {
            continue;
        }
        for (uint pos = 0; pos < k_limit; ++pos) {
            if (kiln_score_better(score, index, top_scores[pos], top_indices[pos])) {
                for (uint shift = k_limit - 1; shift > pos; --shift) {
                    top_scores[shift] = top_scores[shift - 1];
                    top_indices[shift] = top_indices[shift - 1];
                }
                top_scores[pos] = score;
                top_indices[pos] = index;
                break;
            }
        }
    }

    if (k_limit == 0 || !isfinite(top_scores[0])) {
        final_index[0] = 0.0f;
        return;
    }
    if (k_limit == 1) {
        final_index[0] = top_indices[0];
        return;
    }

    const float max_score = top_scores[0];
    float sum = 0.0f;
    for (uint i = 0; i < k_limit; ++i) {
        if (isfinite(top_scores[i])) {
            const float p = exp(top_scores[i] - max_score);
            probs[i] = p;
            sum += p;
        }
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        final_index[0] = top_indices[0];
        return;
    }
    for (uint i = 0; i < k_limit; ++i) {
        probs[i] /= sum;
    }

    if (isfinite(min_p) && min_p > 0.0f) {
        const float threshold = min_p * probs[0];
        float filtered_sum = 0.0f;
        for (uint i = 0; i < k_limit; ++i) {
            if (probs[i] < threshold) {
                probs[i] = 0.0f;
            }
            filtered_sum += probs[i];
        }
        if (filtered_sum <= 0.0f || !isfinite(filtered_sum)) {
            final_index[0] = top_indices[0];
            return;
        }
        for (uint i = 0; i < k_limit; ++i) {
            probs[i] /= filtered_sum;
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        uint cutoff = k_limit;
        for (uint i = 0; i < k_limit; ++i) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        float filtered_sum = 0.0f;
        for (uint i = 0; i < k_limit; ++i) {
            if (i >= cutoff) {
                probs[i] = 0.0f;
            }
            filtered_sum += probs[i];
        }
        if (filtered_sum <= 0.0f || !isfinite(filtered_sum)) {
            final_index[0] = top_indices[0];
            return;
        }
        for (uint i = 0; i < k_limit; ++i) {
            probs[i] /= filtered_sum;
        }
    }

    const float r = kiln_uniform01_from_seed(seed_lo, seed_hi);
    float cumsum = 0.0f;
    for (uint i = 0; i < k_limit; ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            final_index[0] = top_indices[i];
            return;
        }
    }
    for (uint i = k_limit; i > 0; --i) {
        if (probs[i - 1] > 0.0f) {
            final_index[0] = top_indices[i - 1];
            return;
        }
    }
    final_index[0] = top_indices[0];
}
"#;

const METAL_MLP_GATE_UP_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_mlp_gate_up_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gate_t [[buffer(1)]],
    device const bfloat* up_t [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& intermediate [[buffer(6)]],
    constant uint& row_pair_mode [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint cols2 = (intermediate + 1) >> 1;
    if (row_pair_mode == 0 || row_pair_mode == 6) {
        const uint total = rows * cols2;
        if (gid >= total) {
            return;
        }

        const uint row = gid / cols2;
        const uint col0 = (gid - row * cols2) << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < intermediate;
        const uint x_base = row * hidden;
        float gate_acc0 = 0.0f;
        float up_acc0 = 0.0f;
        float gate_acc1 = 0.0f;
        float up_acc1 = 0.0f;
        if (row_pair_mode == 6) {
            for (uint i = 0; i < hidden; ++i) {
                const float xv = static_cast<float>(x[x_base + i]);
                const uint w_idx0 = i * intermediate + col0;
                const bfloat2 gate_w = *(device const bfloat2*)(gate_t + w_idx0);
                const bfloat2 up_w = *(device const bfloat2*)(up_t + w_idx0);
                gate_acc0 += xv * static_cast<float>(gate_w[0]);
                up_acc0 += xv * static_cast<float>(up_w[0]);
                gate_acc1 += xv * static_cast<float>(gate_w[1]);
                up_acc1 += xv * static_cast<float>(up_w[1]);
            }
        } else {
            for (uint i = 0; i < hidden; ++i) {
                const float xv = static_cast<float>(x[x_base + i]);
                const uint w_idx0 = i * intermediate + col0;
                gate_acc0 += xv * static_cast<float>(gate_t[w_idx0]);
                up_acc0 += xv * static_cast<float>(up_t[w_idx0]);
                if (has_col1) {
                    const uint w_idx1 = w_idx0 + 1;
                    gate_acc1 += xv * static_cast<float>(gate_t[w_idx1]);
                    up_acc1 += xv * static_cast<float>(up_t[w_idx1]);
                }
            }
        }

        const uint out_base = row * intermediate;
        const float gate_sigmoid0 = 1.0f / (1.0f + exp(-gate_acc0));
        out[out_base + col0] = static_cast<bfloat>((gate_acc0 * gate_sigmoid0) * up_acc0);
        if (has_col1) {
            const float gate_sigmoid1 = 1.0f / (1.0f + exp(-gate_acc1));
            out[out_base + col1] = static_cast<bfloat>((gate_acc1 * gate_sigmoid1) * up_acc1);
        }
        return;
    }

    if (row_pair_mode == 3 || row_pair_mode == 7) {
        const uint total = cols2;
        if (gid >= total) {
            return;
        }

        const uint col0 = gid << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < intermediate;
        const uint x_base1 = hidden;
        const uint x_base2 = hidden << 1;
        float gate_acc00 = 0.0f;
        float up_acc00 = 0.0f;
        float gate_acc01 = 0.0f;
        float up_acc01 = 0.0f;
        float gate_acc10 = 0.0f;
        float up_acc10 = 0.0f;
        float gate_acc11 = 0.0f;
        float up_acc11 = 0.0f;
        float gate_acc20 = 0.0f;
        float up_acc20 = 0.0f;
        float gate_acc21 = 0.0f;
        float up_acc21 = 0.0f;
        if (row_pair_mode == 7) {
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx0 = i * intermediate + col0;
                const bfloat2 gate_w = *(device const bfloat2*)(gate_t + w_idx0);
                const bfloat2 up_w = *(device const bfloat2*)(up_t + w_idx0);
                const float gate_w0 = static_cast<float>(gate_w[0]);
                const float gate_w1 = static_cast<float>(gate_w[1]);
                const float up_w0 = static_cast<float>(up_w[0]);
                const float up_w1 = static_cast<float>(up_w[1]);

                const float xv0 = static_cast<float>(x[i]);
                gate_acc00 += xv0 * gate_w0;
                up_acc00 += xv0 * up_w0;
                gate_acc01 += xv0 * gate_w1;
                up_acc01 += xv0 * up_w1;

                const float xv1 = static_cast<float>(x[x_base1 + i]);
                gate_acc10 += xv1 * gate_w0;
                up_acc10 += xv1 * up_w0;
                gate_acc11 += xv1 * gate_w1;
                up_acc11 += xv1 * up_w1;

                const float xv2 = static_cast<float>(x[x_base2 + i]);
                gate_acc20 += xv2 * gate_w0;
                up_acc20 += xv2 * up_w0;
                gate_acc21 += xv2 * gate_w1;
                up_acc21 += xv2 * up_w1;
            }
        } else {
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx0 = i * intermediate + col0;
                const float gate_w0 = static_cast<float>(gate_t[w_idx0]);
                const float up_w0 = static_cast<float>(up_t[w_idx0]);
                float gate_w1 = 0.0f;
                float up_w1 = 0.0f;
                if (has_col1) {
                    const uint w_idx1 = w_idx0 + 1;
                    gate_w1 = static_cast<float>(gate_t[w_idx1]);
                    up_w1 = static_cast<float>(up_t[w_idx1]);
                }

                const float xv0 = static_cast<float>(x[i]);
                gate_acc00 += xv0 * gate_w0;
                up_acc00 += xv0 * up_w0;
                if (has_col1) {
                    gate_acc01 += xv0 * gate_w1;
                    up_acc01 += xv0 * up_w1;
                }

                const float xv1 = static_cast<float>(x[x_base1 + i]);
                gate_acc10 += xv1 * gate_w0;
                up_acc10 += xv1 * up_w0;
                if (has_col1) {
                    gate_acc11 += xv1 * gate_w1;
                    up_acc11 += xv1 * up_w1;
                }

                const float xv2 = static_cast<float>(x[x_base2 + i]);
                gate_acc20 += xv2 * gate_w0;
                up_acc20 += xv2 * up_w0;
                if (has_col1) {
                    gate_acc21 += xv2 * gate_w1;
                    up_acc21 += xv2 * up_w1;
                }
            }
        }

        const float gate_sigmoid00 = 1.0f / (1.0f + exp(-gate_acc00));
        out[col0] = static_cast<bfloat>((gate_acc00 * gate_sigmoid00) * up_acc00);
        if (has_col1) {
            const float gate_sigmoid01 = 1.0f / (1.0f + exp(-gate_acc01));
            out[col1] = static_cast<bfloat>((gate_acc01 * gate_sigmoid01) * up_acc01);
        }

        const uint out_base1 = intermediate;
        const float gate_sigmoid10 = 1.0f / (1.0f + exp(-gate_acc10));
        out[out_base1 + col0] = static_cast<bfloat>((gate_acc10 * gate_sigmoid10) * up_acc10);
        if (has_col1) {
            const float gate_sigmoid11 = 1.0f / (1.0f + exp(-gate_acc11));
            out[out_base1 + col1] = static_cast<bfloat>((gate_acc11 * gate_sigmoid11) * up_acc11);
        }

        const uint out_base2 = intermediate << 1;
        const float gate_sigmoid20 = 1.0f / (1.0f + exp(-gate_acc20));
        out[out_base2 + col0] = static_cast<bfloat>((gate_acc20 * gate_sigmoid20) * up_acc20);
        if (has_col1) {
            const float gate_sigmoid21 = 1.0f / (1.0f + exp(-gate_acc21));
            out[out_base2 + col1] = static_cast<bfloat>((gate_acc21 * gate_sigmoid21) * up_acc21);
        }
        return;
    }

    if (row_pair_mode == 4 || row_pair_mode == 5) {
        const uint row_quads = (rows + 3) >> 2;
        const uint total = row_quads * cols2;
        if (gid >= total) {
            return;
        }

        const uint row_quad = gid / cols2;
        const uint row0 = row_quad << 2;
        const uint row1 = row0 + 1;
        const uint row2 = row0 + 2;
        const uint row3 = row0 + 3;
        const bool has_row1 = row1 < rows;
        const bool has_row2 = row2 < rows;
        const bool has_row3 = row3 < rows;
        const uint col0 = (gid - row_quad * cols2) << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < intermediate;
        const uint x_base0 = row0 * hidden;
        const uint x_base1 = row1 * hidden;
        const uint x_base2 = row2 * hidden;
        const uint x_base3 = row3 * hidden;
        float gate_acc00 = 0.0f;
        float up_acc00 = 0.0f;
        float gate_acc01 = 0.0f;
        float up_acc01 = 0.0f;
        float gate_acc10 = 0.0f;
        float up_acc10 = 0.0f;
        float gate_acc11 = 0.0f;
        float up_acc11 = 0.0f;
        float gate_acc20 = 0.0f;
        float up_acc20 = 0.0f;
        float gate_acc21 = 0.0f;
        float up_acc21 = 0.0f;
        float gate_acc30 = 0.0f;
        float up_acc30 = 0.0f;
        float gate_acc31 = 0.0f;
        float up_acc31 = 0.0f;
        if (row_pair_mode == 5) {
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx0 = i * intermediate + col0;
                const bfloat2 gate_w = *(device const bfloat2*)(gate_t + w_idx0);
                const bfloat2 up_w = *(device const bfloat2*)(up_t + w_idx0);
                const float gate_w0 = static_cast<float>(gate_w[0]);
                const float gate_w1 = static_cast<float>(gate_w[1]);
                const float up_w0 = static_cast<float>(up_w[0]);
                const float up_w1 = static_cast<float>(up_w[1]);

                const float xv0 = static_cast<float>(x[x_base0 + i]);
                gate_acc00 += xv0 * gate_w0;
                up_acc00 += xv0 * up_w0;
                gate_acc01 += xv0 * gate_w1;
                up_acc01 += xv0 * up_w1;

                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    gate_acc10 += xv1 * gate_w0;
                    up_acc10 += xv1 * up_w0;
                    gate_acc11 += xv1 * gate_w1;
                    up_acc11 += xv1 * up_w1;
                }
                if (has_row2) {
                    const float xv2 = static_cast<float>(x[x_base2 + i]);
                    gate_acc20 += xv2 * gate_w0;
                    up_acc20 += xv2 * up_w0;
                    gate_acc21 += xv2 * gate_w1;
                    up_acc21 += xv2 * up_w1;
                }
                if (has_row3) {
                    const float xv3 = static_cast<float>(x[x_base3 + i]);
                    gate_acc30 += xv3 * gate_w0;
                    up_acc30 += xv3 * up_w0;
                    gate_acc31 += xv3 * gate_w1;
                    up_acc31 += xv3 * up_w1;
                }
            }
        } else {
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx0 = i * intermediate + col0;
                const float gate_w0 = static_cast<float>(gate_t[w_idx0]);
                const float up_w0 = static_cast<float>(up_t[w_idx0]);
                float gate_w1 = 0.0f;
                float up_w1 = 0.0f;
                if (has_col1) {
                    const uint w_idx1 = w_idx0 + 1;
                    gate_w1 = static_cast<float>(gate_t[w_idx1]);
                    up_w1 = static_cast<float>(up_t[w_idx1]);
                }

                const float xv0 = static_cast<float>(x[x_base0 + i]);
                gate_acc00 += xv0 * gate_w0;
                up_acc00 += xv0 * up_w0;
                if (has_col1) {
                    gate_acc01 += xv0 * gate_w1;
                    up_acc01 += xv0 * up_w1;
                }

                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    gate_acc10 += xv1 * gate_w0;
                    up_acc10 += xv1 * up_w0;
                    if (has_col1) {
                        gate_acc11 += xv1 * gate_w1;
                        up_acc11 += xv1 * up_w1;
                    }
                }
                if (has_row2) {
                    const float xv2 = static_cast<float>(x[x_base2 + i]);
                    gate_acc20 += xv2 * gate_w0;
                    up_acc20 += xv2 * up_w0;
                    if (has_col1) {
                        gate_acc21 += xv2 * gate_w1;
                        up_acc21 += xv2 * up_w1;
                    }
                }
                if (has_row3) {
                    const float xv3 = static_cast<float>(x[x_base3 + i]);
                    gate_acc30 += xv3 * gate_w0;
                    up_acc30 += xv3 * up_w0;
                    if (has_col1) {
                        gate_acc31 += xv3 * gate_w1;
                        up_acc31 += xv3 * up_w1;
                    }
                }
            }
        }

        const uint out_base0 = row0 * intermediate;
        const float gate_sigmoid00 = 1.0f / (1.0f + exp(-gate_acc00));
        out[out_base0 + col0] = static_cast<bfloat>((gate_acc00 * gate_sigmoid00) * up_acc00);
        if (has_col1) {
            const float gate_sigmoid01 = 1.0f / (1.0f + exp(-gate_acc01));
            out[out_base0 + col1] = static_cast<bfloat>((gate_acc01 * gate_sigmoid01) * up_acc01);
        }
        if (has_row1) {
            const uint out_base1 = row1 * intermediate;
            const float gate_sigmoid10 = 1.0f / (1.0f + exp(-gate_acc10));
            out[out_base1 + col0] = static_cast<bfloat>((gate_acc10 * gate_sigmoid10) * up_acc10);
            if (has_col1) {
                const float gate_sigmoid11 = 1.0f / (1.0f + exp(-gate_acc11));
                out[out_base1 + col1] = static_cast<bfloat>((gate_acc11 * gate_sigmoid11) * up_acc11);
            }
        }
        if (has_row2) {
            const uint out_base2 = row2 * intermediate;
            const float gate_sigmoid20 = 1.0f / (1.0f + exp(-gate_acc20));
            out[out_base2 + col0] = static_cast<bfloat>((gate_acc20 * gate_sigmoid20) * up_acc20);
            if (has_col1) {
                const float gate_sigmoid21 = 1.0f / (1.0f + exp(-gate_acc21));
                out[out_base2 + col1] = static_cast<bfloat>((gate_acc21 * gate_sigmoid21) * up_acc21);
            }
        }
        if (has_row3) {
            const uint out_base3 = row3 * intermediate;
            const float gate_sigmoid30 = 1.0f / (1.0f + exp(-gate_acc30));
            out[out_base3 + col0] = static_cast<bfloat>((gate_acc30 * gate_sigmoid30) * up_acc30);
            if (has_col1) {
                const float gate_sigmoid31 = 1.0f / (1.0f + exp(-gate_acc31));
                out[out_base3 + col1] = static_cast<bfloat>((gate_acc31 * gate_sigmoid31) * up_acc31);
            }
        }
        return;
    }

    const uint row_pairs = (rows + 1) >> 1;
    const uint total = row_pairs * cols2;
    if (gid >= total) {
        return;
    }

    const uint row_pair = gid / cols2;
    const uint row0 = row_pair << 1;
    const uint row1 = row0 + 1;
    const bool has_row1 = row1 < rows;
    const uint col0 = (gid - row_pair * cols2) << 1;
    const uint col1 = col0 + 1;
    const bool has_col1 = col1 < intermediate;
    const uint x_base0 = row0 * hidden;
    const uint x_base1 = row1 * hidden;
    float gate_acc00 = 0.0f;
    float up_acc00 = 0.0f;
    float gate_acc01 = 0.0f;
    float up_acc01 = 0.0f;
    float gate_acc10 = 0.0f;
    float up_acc10 = 0.0f;
    float gate_acc11 = 0.0f;
    float up_acc11 = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        const uint w_idx0 = i * intermediate + col0;
        const float gate_w0 = static_cast<float>(gate_t[w_idx0]);
        const float up_w0 = static_cast<float>(up_t[w_idx0]);
        const float xv0 = static_cast<float>(x[x_base0 + i]);
        gate_acc00 += xv0 * gate_w0;
        up_acc00 += xv0 * up_w0;
        if (has_row1) {
            const float xv1 = static_cast<float>(x[x_base1 + i]);
            gate_acc10 += xv1 * gate_w0;
            up_acc10 += xv1 * up_w0;
            if (has_col1) {
                const uint w_idx1 = w_idx0 + 1;
                const float gate_w1 = static_cast<float>(gate_t[w_idx1]);
                const float up_w1 = static_cast<float>(up_t[w_idx1]);
                gate_acc11 += xv1 * gate_w1;
                up_acc11 += xv1 * up_w1;
                gate_acc01 += xv0 * gate_w1;
                up_acc01 += xv0 * up_w1;
            }
        } else if (has_col1) {
            const uint w_idx1 = w_idx0 + 1;
            const float xv0_col1 = xv0;
            gate_acc01 += xv0_col1 * static_cast<float>(gate_t[w_idx1]);
            up_acc01 += xv0_col1 * static_cast<float>(up_t[w_idx1]);
        }
    }

    const uint out_base0 = row0 * intermediate;
    const float gate_sigmoid00 = 1.0f / (1.0f + exp(-gate_acc00));
    out[out_base0 + col0] = static_cast<bfloat>((gate_acc00 * gate_sigmoid00) * up_acc00);
    if (has_col1) {
        const float gate_sigmoid01 = 1.0f / (1.0f + exp(-gate_acc01));
        out[out_base0 + col1] = static_cast<bfloat>((gate_acc01 * gate_sigmoid01) * up_acc01);
    }
    if (has_row1) {
        const uint out_base1 = row1 * intermediate;
        const float gate_sigmoid10 = 1.0f / (1.0f + exp(-gate_acc10));
        out[out_base1 + col0] = static_cast<bfloat>((gate_acc10 * gate_sigmoid10) * up_acc10);
        if (has_col1) {
            const float gate_sigmoid11 = 1.0f / (1.0f + exp(-gate_acc11));
            out[out_base1 + col1] = static_cast<bfloat>((gate_acc11 * gate_sigmoid11) * up_acc11);
        }
    }
}

kernel void kiln_mlp_gate_up_serial_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gate_t [[buffer(1)]],
    device const bfloat* up_t [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& intermediate [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint cols2 = intermediate >> 1;
    if (gid >= cols2) {
        return;
    }

    const uint col0 = gid << 1;
    float gate_acc0 = 0.0f;
    float up_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float up_acc1 = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        const float xv = static_cast<float>(x[i]);
        const uint w_idx0 = i * intermediate + col0;
        const bfloat2 gate_w = *(device const bfloat2*)(gate_t + w_idx0);
        const bfloat2 up_w = *(device const bfloat2*)(up_t + w_idx0);
        gate_acc0 += xv * static_cast<float>(gate_w[0]);
        up_acc0 += xv * static_cast<float>(up_w[0]);
        gate_acc1 += xv * static_cast<float>(gate_w[1]);
        up_acc1 += xv * static_cast<float>(up_w[1]);
    }

    const float gate_sigmoid0 = 1.0f / (1.0f + exp(-gate_acc0));
    const float gate_sigmoid1 = 1.0f / (1.0f + exp(-gate_acc1));
    out[col0] = static_cast<bfloat>((gate_acc0 * gate_sigmoid0) * up_acc0);
    out[col0 + 1] = static_cast<bfloat>((gate_acc1 * gate_sigmoid1) * up_acc1);
}

kernel void kiln_mlp_silu_mul_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const float gate_val = static_cast<float>(gate[gid]);
    const float up_val = static_cast<float>(up[gid]);
    const float sigmoid = 1.0f / (1.0f + exp(-gate_val));
    out[gid] = static_cast<bfloat>((gate_val * sigmoid) * up_val);
}
"#;

const METAL_ATTN_GATE_SIGMOID_MUL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_attn_gate_sigmoid_mul_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const float gate_sigmoid = 1.0f / (1.0f + exp(-static_cast<float>(gate[gid])));
    out[gid] = static_cast<bfloat>(
        static_cast<float>(x[gid]) * static_cast<float>(static_cast<bfloat>(gate_sigmoid))
    );
}
"#;

const METAL_TRANSPOSED_COOP_GEMV_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_transposed_coop_gemv4_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint col_base = (tgroup * 4 + simd_group) * 4;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const bool full_tile = col_base + 3 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w = *w4_ptr;
            acc0 += xv * static_cast<float>(w[0]);
            acc1 += xv * static_cast<float>(w[1]);
            acc2 += xv * static_cast<float>(w[2]);
            acc3 += xv * static_cast<float>(w[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
    }
}

kernel void kiln_transposed_coop_gemv8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint col_base = (tgroup * 4 + simd_group) * 8;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + 7 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[col_base + 7] = static_cast<bfloat>(acc7);
        }
    }
}

kernel void kiln_transposed_coop_gemv16_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint col_base = (tgroup * 4 + simd_group) * 16;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc8 = 0.0f;
    float acc9 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    const bool full_tile = col_base + 15 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            const bfloat4 w2 = w4_ptr[2];
            const bfloat4 w3 = w4_ptr[3];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
            acc8 += xv * static_cast<float>(w2[0]);
            acc9 += xv * static_cast<float>(w2[1]);
            acc10 += xv * static_cast<float>(w2[2]);
            acc11 += xv * static_cast<float>(w2[3]);
            acc12 += xv * static_cast<float>(w3[0]);
            acc13 += xv * static_cast<float>(w3[1]);
            acc14 += xv * static_cast<float>(w3[2]);
            acc15 += xv * static_cast<float>(w3[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
            if (col_base + 8 < output_dim) {
                acc8 += xv * static_cast<float>(weight_t[weight_base + 8]);
            }
            if (col_base + 9 < output_dim) {
                acc9 += xv * static_cast<float>(weight_t[weight_base + 9]);
            }
            if (col_base + 10 < output_dim) {
                acc10 += xv * static_cast<float>(weight_t[weight_base + 10]);
            }
            if (col_base + 11 < output_dim) {
                acc11 += xv * static_cast<float>(weight_t[weight_base + 11]);
            }
            if (col_base + 12 < output_dim) {
                acc12 += xv * static_cast<float>(weight_t[weight_base + 12]);
            }
            if (col_base + 13 < output_dim) {
                acc13 += xv * static_cast<float>(weight_t[weight_base + 13]);
            }
            if (col_base + 14 < output_dim) {
                acc14 += xv * static_cast<float>(weight_t[weight_base + 14]);
            }
            if (col_base + 15 < output_dim) {
                acc15 += xv * static_cast<float>(weight_t[weight_base + 15]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    acc8 = simd_sum(acc8);
    acc9 = simd_sum(acc9);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[col_base + 7] = static_cast<bfloat>(acc7);
        }
        if (col_base + 8 < output_dim) {
            out[col_base + 8] = static_cast<bfloat>(acc8);
        }
        if (col_base + 9 < output_dim) {
            out[col_base + 9] = static_cast<bfloat>(acc9);
        }
        if (col_base + 10 < output_dim) {
            out[col_base + 10] = static_cast<bfloat>(acc10);
        }
        if (col_base + 11 < output_dim) {
            out[col_base + 11] = static_cast<bfloat>(acc11);
        }
        if (col_base + 12 < output_dim) {
            out[col_base + 12] = static_cast<bfloat>(acc12);
        }
        if (col_base + 13 < output_dim) {
            out[col_base + 13] = static_cast<bfloat>(acc13);
        }
        if (col_base + 14 < output_dim) {
            out[col_base + 14] = static_cast<bfloat>(acc14);
        }
        if (col_base + 15 < output_dim) {
            out[col_base + 15] = static_cast<bfloat>(acc15);
        }
    }
}

kernel void kiln_transposed_coop_gemv8_batch_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    constant uint& row_pair_mode [[buffer(5)]],
    constant uint& row_group_size_arg [[buffer(6)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const bool grouped_mode = row_pair_mode != 0;
    const uint row_group_size = grouped_mode ? row_group_size_arg : 1;
    const bool row_quad_mode = grouped_mode && row_group_size == 4;
    const uint tile_cols = row_quad_mode ? 4 : 8;
    const uint col_base = (tgroup.x * 4 + simd_group) * tile_cols;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + tile_cols - 1 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);
    const uint batch_idx = grouped_mode ? tgroup.y * row_group_size : tgroup.y;
    const uint x_base = batch_idx * input_dim;

    if (!grouped_mode) {
        for (uint row = lane; row < input_dim; row += 32) {
            const float xv = static_cast<float>(x[x_base + row]);
            const uint weight_base = row * output_dim + col_base;
            if (vector_load_safe) {
                device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
                const bfloat4 w0 = w4_ptr[0];
                const bfloat4 w1 = w4_ptr[1];
                acc0 += xv * static_cast<float>(w0[0]);
                acc1 += xv * static_cast<float>(w0[1]);
                acc2 += xv * static_cast<float>(w0[2]);
                acc3 += xv * static_cast<float>(w0[3]);
                acc4 += xv * static_cast<float>(w1[0]);
                acc5 += xv * static_cast<float>(w1[1]);
                acc6 += xv * static_cast<float>(w1[2]);
                acc7 += xv * static_cast<float>(w1[3]);
            } else {
                acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
                if (col_base + 1 < output_dim) {
                    acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
                }
                if (col_base + 2 < output_dim) {
                    acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
                }
                if (col_base + 3 < output_dim) {
                    acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
                }
                if (col_base + 4 < output_dim) {
                    acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
                }
                if (col_base + 5 < output_dim) {
                    acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
                }
                if (col_base + 6 < output_dim) {
                    acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
                }
                if (col_base + 7 < output_dim) {
                    acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
                }
            }
        }

        acc0 = simd_sum(acc0);
        acc1 = simd_sum(acc1);
        acc2 = simd_sum(acc2);
        acc3 = simd_sum(acc3);
        acc4 = simd_sum(acc4);
        acc5 = simd_sum(acc5);
        acc6 = simd_sum(acc6);
        acc7 = simd_sum(acc7);

        if (lane == 0) {
            const uint out_base = batch_idx * output_dim;
            out[out_base + col_base + 0] = static_cast<bfloat>(acc0);
            if (col_base + 1 < output_dim) {
                out[out_base + col_base + 1] = static_cast<bfloat>(acc1);
            }
            if (col_base + 2 < output_dim) {
                out[out_base + col_base + 2] = static_cast<bfloat>(acc2);
            }
            if (col_base + 3 < output_dim) {
                out[out_base + col_base + 3] = static_cast<bfloat>(acc3);
            }
            if (col_base + 4 < output_dim) {
                out[out_base + col_base + 4] = static_cast<bfloat>(acc4);
            }
            if (col_base + 5 < output_dim) {
                out[out_base + col_base + 5] = static_cast<bfloat>(acc5);
            }
            if (col_base + 6 < output_dim) {
                out[out_base + col_base + 6] = static_cast<bfloat>(acc6);
            }
            if (col_base + 7 < output_dim) {
                out[out_base + col_base + 7] = static_cast<bfloat>(acc7);
            }
        }
        return;
    }

    if (row_quad_mode) {
        const uint batch1 = batch_idx + 1;
        const uint batch2 = batch_idx + 2;
        const uint batch3 = batch_idx + 3;
        const bool has_batch1 = batch1 < row_pair_mode;
        const bool has_batch2 = batch2 < row_pair_mode;
        const bool has_batch3 = batch3 < row_pair_mode;
        const uint x_base1 = batch1 * input_dim;
        const uint x_base2 = batch2 * input_dim;
        const uint x_base3 = batch3 * input_dim;
        float acc10 = 0.0f;
        float acc11 = 0.0f;
        float acc12 = 0.0f;
        float acc13 = 0.0f;
        float acc20 = 0.0f;
        float acc21 = 0.0f;
        float acc22 = 0.0f;
        float acc23 = 0.0f;
        float acc30 = 0.0f;
        float acc31 = 0.0f;
        float acc32 = 0.0f;
        float acc33 = 0.0f;

        for (uint row = lane; row < input_dim; row += 32) {
            const float xv0 = static_cast<float>(x[x_base + row]);
            const float xv1 = has_batch1 ? static_cast<float>(x[x_base1 + row]) : 0.0f;
            const float xv2 = has_batch2 ? static_cast<float>(x[x_base2 + row]) : 0.0f;
            const float xv3 = has_batch3 ? static_cast<float>(x[x_base3 + row]) : 0.0f;
            const uint weight_base = row * output_dim + col_base;
            if (vector_load_safe) {
                device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
                const bfloat4 w = *w4_ptr;
                const float w0 = static_cast<float>(w[0]);
                const float w1 = static_cast<float>(w[1]);
                const float w2 = static_cast<float>(w[2]);
                const float w3 = static_cast<float>(w[3]);
                acc0 += xv0 * w0;
                acc1 += xv0 * w1;
                acc2 += xv0 * w2;
                acc3 += xv0 * w3;
                acc10 += xv1 * w0;
                acc11 += xv1 * w1;
                acc12 += xv1 * w2;
                acc13 += xv1 * w3;
                acc20 += xv2 * w0;
                acc21 += xv2 * w1;
                acc22 += xv2 * w2;
                acc23 += xv2 * w3;
                acc30 += xv3 * w0;
                acc31 += xv3 * w1;
                acc32 += xv3 * w2;
                acc33 += xv3 * w3;
            } else {
                const float w0 = static_cast<float>(weight_t[weight_base + 0]);
                acc0 += xv0 * w0;
                acc10 += xv1 * w0;
                acc20 += xv2 * w0;
                acc30 += xv3 * w0;
                if (col_base + 1 < output_dim) {
                    const float w1 = static_cast<float>(weight_t[weight_base + 1]);
                    acc1 += xv0 * w1;
                    acc11 += xv1 * w1;
                    acc21 += xv2 * w1;
                    acc31 += xv3 * w1;
                }
                if (col_base + 2 < output_dim) {
                    const float w2 = static_cast<float>(weight_t[weight_base + 2]);
                    acc2 += xv0 * w2;
                    acc12 += xv1 * w2;
                    acc22 += xv2 * w2;
                    acc32 += xv3 * w2;
                }
                if (col_base + 3 < output_dim) {
                    const float w3 = static_cast<float>(weight_t[weight_base + 3]);
                    acc3 += xv0 * w3;
                    acc13 += xv1 * w3;
                    acc23 += xv2 * w3;
                    acc33 += xv3 * w3;
                }
            }
        }

        acc0 = simd_sum(acc0);
        acc1 = simd_sum(acc1);
        acc2 = simd_sum(acc2);
        acc3 = simd_sum(acc3);
        acc10 = simd_sum(acc10);
        acc11 = simd_sum(acc11);
        acc12 = simd_sum(acc12);
        acc13 = simd_sum(acc13);
        acc20 = simd_sum(acc20);
        acc21 = simd_sum(acc21);
        acc22 = simd_sum(acc22);
        acc23 = simd_sum(acc23);
        acc30 = simd_sum(acc30);
        acc31 = simd_sum(acc31);
        acc32 = simd_sum(acc32);
        acc33 = simd_sum(acc33);

        if (lane == 0) {
            const uint out_base = batch_idx * output_dim;
            out[out_base + col_base + 0] = static_cast<bfloat>(acc0);
            if (col_base + 1 < output_dim) {
                out[out_base + col_base + 1] = static_cast<bfloat>(acc1);
            }
            if (col_base + 2 < output_dim) {
                out[out_base + col_base + 2] = static_cast<bfloat>(acc2);
            }
            if (col_base + 3 < output_dim) {
                out[out_base + col_base + 3] = static_cast<bfloat>(acc3);
            }
            if (has_batch1) {
                const uint out_base1 = batch1 * output_dim;
                out[out_base1 + col_base + 0] = static_cast<bfloat>(acc10);
                if (col_base + 1 < output_dim) {
                    out[out_base1 + col_base + 1] = static_cast<bfloat>(acc11);
                }
                if (col_base + 2 < output_dim) {
                    out[out_base1 + col_base + 2] = static_cast<bfloat>(acc12);
                }
                if (col_base + 3 < output_dim) {
                    out[out_base1 + col_base + 3] = static_cast<bfloat>(acc13);
                }
            }
            if (has_batch2) {
                const uint out_base2 = batch2 * output_dim;
                out[out_base2 + col_base + 0] = static_cast<bfloat>(acc20);
                if (col_base + 1 < output_dim) {
                    out[out_base2 + col_base + 1] = static_cast<bfloat>(acc21);
                }
                if (col_base + 2 < output_dim) {
                    out[out_base2 + col_base + 2] = static_cast<bfloat>(acc22);
                }
                if (col_base + 3 < output_dim) {
                    out[out_base2 + col_base + 3] = static_cast<bfloat>(acc23);
                }
            }
            if (has_batch3) {
                const uint out_base3 = batch3 * output_dim;
                out[out_base3 + col_base + 0] = static_cast<bfloat>(acc30);
                if (col_base + 1 < output_dim) {
                    out[out_base3 + col_base + 1] = static_cast<bfloat>(acc31);
                }
                if (col_base + 2 < output_dim) {
                    out[out_base3 + col_base + 2] = static_cast<bfloat>(acc32);
                }
                if (col_base + 3 < output_dim) {
                    out[out_base3 + col_base + 3] = static_cast<bfloat>(acc33);
                }
            }
        }
        return;
    }

    const uint batch1 = batch_idx + 1;
    const bool has_batch1 = batch1 < row_pair_mode;
    const uint x_base1 = batch1 * input_dim;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    float acc16 = 0.0f;
    float acc17 = 0.0f;

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv0 = static_cast<float>(x[x_base + row]);
        const float xv1 = has_batch1 ? static_cast<float>(x[x_base1 + row]) : 0.0f;
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            const float w00 = static_cast<float>(w0[0]);
            const float w01 = static_cast<float>(w0[1]);
            const float w02 = static_cast<float>(w0[2]);
            const float w03 = static_cast<float>(w0[3]);
            const float w04 = static_cast<float>(w1[0]);
            const float w05 = static_cast<float>(w1[1]);
            const float w06 = static_cast<float>(w1[2]);
            const float w07 = static_cast<float>(w1[3]);
            acc0 += xv0 * w00;
            acc1 += xv0 * w01;
            acc2 += xv0 * w02;
            acc3 += xv0 * w03;
            acc4 += xv0 * w04;
            acc5 += xv0 * w05;
            acc6 += xv0 * w06;
            acc7 += xv0 * w07;
            acc10 += xv1 * w00;
            acc11 += xv1 * w01;
            acc12 += xv1 * w02;
            acc13 += xv1 * w03;
            acc14 += xv1 * w04;
            acc15 += xv1 * w05;
            acc16 += xv1 * w06;
            acc17 += xv1 * w07;
        } else {
            const float w00 = static_cast<float>(weight_t[weight_base + 0]);
            acc0 += xv0 * w00;
            acc10 += xv1 * w00;
            if (col_base + 1 < output_dim) {
                const float w01 = static_cast<float>(weight_t[weight_base + 1]);
                acc1 += xv0 * w01;
                acc11 += xv1 * w01;
            }
            if (col_base + 2 < output_dim) {
                const float w02 = static_cast<float>(weight_t[weight_base + 2]);
                acc2 += xv0 * w02;
                acc12 += xv1 * w02;
            }
            if (col_base + 3 < output_dim) {
                const float w03 = static_cast<float>(weight_t[weight_base + 3]);
                acc3 += xv0 * w03;
                acc13 += xv1 * w03;
            }
            if (col_base + 4 < output_dim) {
                const float w04 = static_cast<float>(weight_t[weight_base + 4]);
                acc4 += xv0 * w04;
                acc14 += xv1 * w04;
            }
            if (col_base + 5 < output_dim) {
                const float w05 = static_cast<float>(weight_t[weight_base + 5]);
                acc5 += xv0 * w05;
                acc15 += xv1 * w05;
            }
            if (col_base + 6 < output_dim) {
                const float w06 = static_cast<float>(weight_t[weight_base + 6]);
                acc6 += xv0 * w06;
                acc16 += xv1 * w06;
            }
            if (col_base + 7 < output_dim) {
                const float w07 = static_cast<float>(weight_t[weight_base + 7]);
                acc7 += xv0 * w07;
                acc17 += xv1 * w07;
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);
    acc16 = simd_sum(acc16);
    acc17 = simd_sum(acc17);

    if (lane == 0) {
        const uint out_base = batch_idx * output_dim;
        out[out_base + col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[out_base + col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[out_base + col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[out_base + col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[out_base + col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[out_base + col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[out_base + col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[out_base + col_base + 7] = static_cast<bfloat>(acc7);
        }
        if (has_batch1) {
            const uint out_base1 = batch1 * output_dim;
            out[out_base1 + col_base + 0] = static_cast<bfloat>(acc10);
            if (col_base + 1 < output_dim) {
                out[out_base1 + col_base + 1] = static_cast<bfloat>(acc11);
            }
            if (col_base + 2 < output_dim) {
                out[out_base1 + col_base + 2] = static_cast<bfloat>(acc12);
            }
            if (col_base + 3 < output_dim) {
                out[out_base1 + col_base + 3] = static_cast<bfloat>(acc13);
            }
            if (col_base + 4 < output_dim) {
                out[out_base1 + col_base + 4] = static_cast<bfloat>(acc14);
            }
            if (col_base + 5 < output_dim) {
                out[out_base1 + col_base + 5] = static_cast<bfloat>(acc15);
            }
            if (col_base + 6 < output_dim) {
                out[out_base1 + col_base + 6] = static_cast<bfloat>(acc16);
            }
            if (col_base + 7 < output_dim) {
                out[out_base1 + col_base + 7] = static_cast<bfloat>(acc17);
            }
        }
    }
}

kernel void kiln_transposed_coop_gemv8_batch_row_triple_tile8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TILE_COLS = 8;
    constexpr uint ROW_GROUP_SIZE = 3;
    const uint col_base = (tgroup.x * 4 + simd_group) * TILE_COLS;
    if (col_base >= output_dim) {
        return;
    }

    const uint batch_idx = tgroup.y * ROW_GROUP_SIZE;
    const uint batch1 = batch_idx + 1;
    const uint batch2 = batch_idx + 2;
    const uint x_base0 = batch_idx * input_dim;
    const uint x_base1 = batch1 * input_dim;
    const uint x_base2 = batch2 * input_dim;
    const bool full_tile = col_base + TILE_COLS - 1 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    float acc16 = 0.0f;
    float acc17 = 0.0f;
    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;
    float acc24 = 0.0f;
    float acc25 = 0.0f;
    float acc26 = 0.0f;
    float acc27 = 0.0f;

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv0 = static_cast<float>(x[x_base0 + row]);
        const float xv1 = static_cast<float>(x[x_base1 + row]);
        const float xv2 = static_cast<float>(x[x_base2 + row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w_lo = w4_ptr[0];
            const bfloat4 w_hi = w4_ptr[1];
            const float w0 = static_cast<float>(w_lo[0]);
            const float w1 = static_cast<float>(w_lo[1]);
            const float w2 = static_cast<float>(w_lo[2]);
            const float w3 = static_cast<float>(w_lo[3]);
            const float w4 = static_cast<float>(w_hi[0]);
            const float w5 = static_cast<float>(w_hi[1]);
            const float w6 = static_cast<float>(w_hi[2]);
            const float w7 = static_cast<float>(w_hi[3]);
            acc0 += xv0 * w0;
            acc1 += xv0 * w1;
            acc2 += xv0 * w2;
            acc3 += xv0 * w3;
            acc4 += xv0 * w4;
            acc5 += xv0 * w5;
            acc6 += xv0 * w6;
            acc7 += xv0 * w7;
            acc10 += xv1 * w0;
            acc11 += xv1 * w1;
            acc12 += xv1 * w2;
            acc13 += xv1 * w3;
            acc14 += xv1 * w4;
            acc15 += xv1 * w5;
            acc16 += xv1 * w6;
            acc17 += xv1 * w7;
            acc20 += xv2 * w0;
            acc21 += xv2 * w1;
            acc22 += xv2 * w2;
            acc23 += xv2 * w3;
            acc24 += xv2 * w4;
            acc25 += xv2 * w5;
            acc26 += xv2 * w6;
            acc27 += xv2 * w7;
        } else {
            const float w0 = static_cast<float>(weight_t[weight_base + 0]);
            acc0 += xv0 * w0;
            acc10 += xv1 * w0;
            acc20 += xv2 * w0;
            if (col_base + 1 < output_dim) {
                const float w1 = static_cast<float>(weight_t[weight_base + 1]);
                acc1 += xv0 * w1;
                acc11 += xv1 * w1;
                acc21 += xv2 * w1;
            }
            if (col_base + 2 < output_dim) {
                const float w2 = static_cast<float>(weight_t[weight_base + 2]);
                acc2 += xv0 * w2;
                acc12 += xv1 * w2;
                acc22 += xv2 * w2;
            }
            if (col_base + 3 < output_dim) {
                const float w3 = static_cast<float>(weight_t[weight_base + 3]);
                acc3 += xv0 * w3;
                acc13 += xv1 * w3;
                acc23 += xv2 * w3;
            }
            if (col_base + 4 < output_dim) {
                const float w4 = static_cast<float>(weight_t[weight_base + 4]);
                acc4 += xv0 * w4;
                acc14 += xv1 * w4;
                acc24 += xv2 * w4;
            }
            if (col_base + 5 < output_dim) {
                const float w5 = static_cast<float>(weight_t[weight_base + 5]);
                acc5 += xv0 * w5;
                acc15 += xv1 * w5;
                acc25 += xv2 * w5;
            }
            if (col_base + 6 < output_dim) {
                const float w6 = static_cast<float>(weight_t[weight_base + 6]);
                acc6 += xv0 * w6;
                acc16 += xv1 * w6;
                acc26 += xv2 * w6;
            }
            if (col_base + 7 < output_dim) {
                const float w7 = static_cast<float>(weight_t[weight_base + 7]);
                acc7 += xv0 * w7;
                acc17 += xv1 * w7;
                acc27 += xv2 * w7;
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);
    acc16 = simd_sum(acc16);
    acc17 = simd_sum(acc17);
    acc20 = simd_sum(acc20);
    acc21 = simd_sum(acc21);
    acc22 = simd_sum(acc22);
    acc23 = simd_sum(acc23);
    acc24 = simd_sum(acc24);
    acc25 = simd_sum(acc25);
    acc26 = simd_sum(acc26);
    acc27 = simd_sum(acc27);

    if (lane == 0) {
        const uint out_base0 = batch_idx * output_dim;
        out[out_base0 + col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[out_base0 + col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[out_base0 + col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[out_base0 + col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[out_base0 + col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[out_base0 + col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[out_base0 + col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[out_base0 + col_base + 7] = static_cast<bfloat>(acc7);
        }

        const uint out_base1 = batch1 * output_dim;
        out[out_base1 + col_base + 0] = static_cast<bfloat>(acc10);
        if (col_base + 1 < output_dim) {
            out[out_base1 + col_base + 1] = static_cast<bfloat>(acc11);
        }
        if (col_base + 2 < output_dim) {
            out[out_base1 + col_base + 2] = static_cast<bfloat>(acc12);
        }
        if (col_base + 3 < output_dim) {
            out[out_base1 + col_base + 3] = static_cast<bfloat>(acc13);
        }
        if (col_base + 4 < output_dim) {
            out[out_base1 + col_base + 4] = static_cast<bfloat>(acc14);
        }
        if (col_base + 5 < output_dim) {
            out[out_base1 + col_base + 5] = static_cast<bfloat>(acc15);
        }
        if (col_base + 6 < output_dim) {
            out[out_base1 + col_base + 6] = static_cast<bfloat>(acc16);
        }
        if (col_base + 7 < output_dim) {
            out[out_base1 + col_base + 7] = static_cast<bfloat>(acc17);
        }

        const uint out_base2 = batch2 * output_dim;
        out[out_base2 + col_base + 0] = static_cast<bfloat>(acc20);
        if (col_base + 1 < output_dim) {
            out[out_base2 + col_base + 1] = static_cast<bfloat>(acc21);
        }
        if (col_base + 2 < output_dim) {
            out[out_base2 + col_base + 2] = static_cast<bfloat>(acc22);
        }
        if (col_base + 3 < output_dim) {
            out[out_base2 + col_base + 3] = static_cast<bfloat>(acc23);
        }
        if (col_base + 4 < output_dim) {
            out[out_base2 + col_base + 4] = static_cast<bfloat>(acc24);
        }
        if (col_base + 5 < output_dim) {
            out[out_base2 + col_base + 5] = static_cast<bfloat>(acc25);
        }
        if (col_base + 6 < output_dim) {
            out[out_base2 + col_base + 6] = static_cast<bfloat>(acc26);
        }
        if (col_base + 7 < output_dim) {
            out[out_base2 + col_base + 7] = static_cast<bfloat>(acc27);
        }
    }
}

kernel void kiln_transposed_coop_gemv8_batch_row_quad_tile8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TILE_COLS = 8;
    constexpr uint ROW_GROUP_SIZE = 4;
    const uint col_base = (tgroup.x * 4 + simd_group) * TILE_COLS;
    if (col_base >= output_dim) {
        return;
    }

    const uint batch_idx = tgroup.y * ROW_GROUP_SIZE;
    const uint batch1 = batch_idx + 1;
    const uint batch2 = batch_idx + 2;
    const uint batch3 = batch_idx + 3;
    const bool has_batch1 = batch1 < batch;
    const bool has_batch2 = batch2 < batch;
    const bool has_batch3 = batch3 < batch;
    const uint x_base0 = batch_idx * input_dim;
    const uint x_base1 = batch1 * input_dim;
    const uint x_base2 = batch2 * input_dim;
    const uint x_base3 = batch3 * input_dim;
    const bool full_tile = col_base + TILE_COLS - 1 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    float acc16 = 0.0f;
    float acc17 = 0.0f;
    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;
    float acc24 = 0.0f;
    float acc25 = 0.0f;
    float acc26 = 0.0f;
    float acc27 = 0.0f;
    float acc30 = 0.0f;
    float acc31 = 0.0f;
    float acc32 = 0.0f;
    float acc33 = 0.0f;
    float acc34 = 0.0f;
    float acc35 = 0.0f;
    float acc36 = 0.0f;
    float acc37 = 0.0f;

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv0 = static_cast<float>(x[x_base0 + row]);
        const float xv1 = has_batch1 ? static_cast<float>(x[x_base1 + row]) : 0.0f;
        const float xv2 = has_batch2 ? static_cast<float>(x[x_base2 + row]) : 0.0f;
        const float xv3 = has_batch3 ? static_cast<float>(x[x_base3 + row]) : 0.0f;
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w_lo = w4_ptr[0];
            const bfloat4 w_hi = w4_ptr[1];
            const float w0 = static_cast<float>(w_lo[0]);
            const float w1 = static_cast<float>(w_lo[1]);
            const float w2 = static_cast<float>(w_lo[2]);
            const float w3 = static_cast<float>(w_lo[3]);
            const float w4 = static_cast<float>(w_hi[0]);
            const float w5 = static_cast<float>(w_hi[1]);
            const float w6 = static_cast<float>(w_hi[2]);
            const float w7 = static_cast<float>(w_hi[3]);
            acc0 += xv0 * w0;
            acc1 += xv0 * w1;
            acc2 += xv0 * w2;
            acc3 += xv0 * w3;
            acc4 += xv0 * w4;
            acc5 += xv0 * w5;
            acc6 += xv0 * w6;
            acc7 += xv0 * w7;
            acc10 += xv1 * w0;
            acc11 += xv1 * w1;
            acc12 += xv1 * w2;
            acc13 += xv1 * w3;
            acc14 += xv1 * w4;
            acc15 += xv1 * w5;
            acc16 += xv1 * w6;
            acc17 += xv1 * w7;
            acc20 += xv2 * w0;
            acc21 += xv2 * w1;
            acc22 += xv2 * w2;
            acc23 += xv2 * w3;
            acc24 += xv2 * w4;
            acc25 += xv2 * w5;
            acc26 += xv2 * w6;
            acc27 += xv2 * w7;
            acc30 += xv3 * w0;
            acc31 += xv3 * w1;
            acc32 += xv3 * w2;
            acc33 += xv3 * w3;
            acc34 += xv3 * w4;
            acc35 += xv3 * w5;
            acc36 += xv3 * w6;
            acc37 += xv3 * w7;
        } else {
            const float w0 = static_cast<float>(weight_t[weight_base + 0]);
            acc0 += xv0 * w0;
            acc10 += xv1 * w0;
            acc20 += xv2 * w0;
            acc30 += xv3 * w0;
            if (col_base + 1 < output_dim) {
                const float w1 = static_cast<float>(weight_t[weight_base + 1]);
                acc1 += xv0 * w1;
                acc11 += xv1 * w1;
                acc21 += xv2 * w1;
                acc31 += xv3 * w1;
            }
            if (col_base + 2 < output_dim) {
                const float w2 = static_cast<float>(weight_t[weight_base + 2]);
                acc2 += xv0 * w2;
                acc12 += xv1 * w2;
                acc22 += xv2 * w2;
                acc32 += xv3 * w2;
            }
            if (col_base + 3 < output_dim) {
                const float w3 = static_cast<float>(weight_t[weight_base + 3]);
                acc3 += xv0 * w3;
                acc13 += xv1 * w3;
                acc23 += xv2 * w3;
                acc33 += xv3 * w3;
            }
            if (col_base + 4 < output_dim) {
                const float w4 = static_cast<float>(weight_t[weight_base + 4]);
                acc4 += xv0 * w4;
                acc14 += xv1 * w4;
                acc24 += xv2 * w4;
                acc34 += xv3 * w4;
            }
            if (col_base + 5 < output_dim) {
                const float w5 = static_cast<float>(weight_t[weight_base + 5]);
                acc5 += xv0 * w5;
                acc15 += xv1 * w5;
                acc25 += xv2 * w5;
                acc35 += xv3 * w5;
            }
            if (col_base + 6 < output_dim) {
                const float w6 = static_cast<float>(weight_t[weight_base + 6]);
                acc6 += xv0 * w6;
                acc16 += xv1 * w6;
                acc26 += xv2 * w6;
                acc36 += xv3 * w6;
            }
            if (col_base + 7 < output_dim) {
                const float w7 = static_cast<float>(weight_t[weight_base + 7]);
                acc7 += xv0 * w7;
                acc17 += xv1 * w7;
                acc27 += xv2 * w7;
                acc37 += xv3 * w7;
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);
    acc16 = simd_sum(acc16);
    acc17 = simd_sum(acc17);
    acc20 = simd_sum(acc20);
    acc21 = simd_sum(acc21);
    acc22 = simd_sum(acc22);
    acc23 = simd_sum(acc23);
    acc24 = simd_sum(acc24);
    acc25 = simd_sum(acc25);
    acc26 = simd_sum(acc26);
    acc27 = simd_sum(acc27);
    acc30 = simd_sum(acc30);
    acc31 = simd_sum(acc31);
    acc32 = simd_sum(acc32);
    acc33 = simd_sum(acc33);
    acc34 = simd_sum(acc34);
    acc35 = simd_sum(acc35);
    acc36 = simd_sum(acc36);
    acc37 = simd_sum(acc37);

    if (lane == 0) {
        const uint out_base0 = batch_idx * output_dim;
        out[out_base0 + col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[out_base0 + col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[out_base0 + col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[out_base0 + col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[out_base0 + col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[out_base0 + col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[out_base0 + col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[out_base0 + col_base + 7] = static_cast<bfloat>(acc7);
        }
        if (has_batch1) {
            const uint out_base1 = batch1 * output_dim;
            out[out_base1 + col_base + 0] = static_cast<bfloat>(acc10);
            if (col_base + 1 < output_dim) {
                out[out_base1 + col_base + 1] = static_cast<bfloat>(acc11);
            }
            if (col_base + 2 < output_dim) {
                out[out_base1 + col_base + 2] = static_cast<bfloat>(acc12);
            }
            if (col_base + 3 < output_dim) {
                out[out_base1 + col_base + 3] = static_cast<bfloat>(acc13);
            }
            if (col_base + 4 < output_dim) {
                out[out_base1 + col_base + 4] = static_cast<bfloat>(acc14);
            }
            if (col_base + 5 < output_dim) {
                out[out_base1 + col_base + 5] = static_cast<bfloat>(acc15);
            }
            if (col_base + 6 < output_dim) {
                out[out_base1 + col_base + 6] = static_cast<bfloat>(acc16);
            }
            if (col_base + 7 < output_dim) {
                out[out_base1 + col_base + 7] = static_cast<bfloat>(acc17);
            }
        }
        if (has_batch2) {
            const uint out_base2 = batch2 * output_dim;
            out[out_base2 + col_base + 0] = static_cast<bfloat>(acc20);
            if (col_base + 1 < output_dim) {
                out[out_base2 + col_base + 1] = static_cast<bfloat>(acc21);
            }
            if (col_base + 2 < output_dim) {
                out[out_base2 + col_base + 2] = static_cast<bfloat>(acc22);
            }
            if (col_base + 3 < output_dim) {
                out[out_base2 + col_base + 3] = static_cast<bfloat>(acc23);
            }
            if (col_base + 4 < output_dim) {
                out[out_base2 + col_base + 4] = static_cast<bfloat>(acc24);
            }
            if (col_base + 5 < output_dim) {
                out[out_base2 + col_base + 5] = static_cast<bfloat>(acc25);
            }
            if (col_base + 6 < output_dim) {
                out[out_base2 + col_base + 6] = static_cast<bfloat>(acc26);
            }
            if (col_base + 7 < output_dim) {
                out[out_base2 + col_base + 7] = static_cast<bfloat>(acc27);
            }
        }
        if (has_batch3) {
            const uint out_base3 = batch3 * output_dim;
            out[out_base3 + col_base + 0] = static_cast<bfloat>(acc30);
            if (col_base + 1 < output_dim) {
                out[out_base3 + col_base + 1] = static_cast<bfloat>(acc31);
            }
            if (col_base + 2 < output_dim) {
                out[out_base3 + col_base + 2] = static_cast<bfloat>(acc32);
            }
            if (col_base + 3 < output_dim) {
                out[out_base3 + col_base + 3] = static_cast<bfloat>(acc33);
            }
            if (col_base + 4 < output_dim) {
                out[out_base3 + col_base + 4] = static_cast<bfloat>(acc34);
            }
            if (col_base + 5 < output_dim) {
                out[out_base3 + col_base + 5] = static_cast<bfloat>(acc35);
            }
            if (col_base + 6 < output_dim) {
                out[out_base3 + col_base + 6] = static_cast<bfloat>(acc36);
            }
            if (col_base + 7 < output_dim) {
                out[out_base3 + col_base + 7] = static_cast<bfloat>(acc37);
            }
        }
    }
}
"#;

const METAL_FUSED_QKV_TRANSPOSED_COOP_GEMV_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_fused_qkv_transposed_coop_gemv8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* q_t [[buffer(1)]],
    device const bfloat* k_t [[buffer(2)]],
    device const bfloat* v_t [[buffer(3)]],
    device bfloat* q_out [[buffer(4)]],
    device bfloat* k_out [[buffer(5)]],
    device bfloat* v_out [[buffer(6)]],
    constant uint& input_dim [[buffer(7)]],
    constant uint& q_output_dim [[buffer(8)]],
    constant uint& k_output_dim [[buffer(9)]],
    constant uint& v_output_dim [[buffer(10)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TILE_COLS = 8;
    constexpr uint SIMD_GROUPS = 4;
    constexpr uint COLS_PER_TGROUP = TILE_COLS * SIMD_GROUPS;

    const uint q_groups = (q_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint k_groups = (k_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint v_groups = (v_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint group = tgroup.x;

    device const bfloat* weight_t = q_t;
    device bfloat* out = q_out;
    uint output_dim = q_output_dim;
    uint group_in_proj = group;
    if (group < q_groups) {
        weight_t = q_t;
        out = q_out;
        output_dim = q_output_dim;
    } else if (group < q_groups + k_groups) {
        weight_t = k_t;
        out = k_out;
        output_dim = k_output_dim;
        group_in_proj = group - q_groups;
    } else if (group < q_groups + k_groups + v_groups) {
        weight_t = v_t;
        out = v_out;
        output_dim = v_output_dim;
        group_in_proj = group - q_groups - k_groups;
    } else {
        return;
    }

    const uint col_base = group_in_proj * COLS_PER_TGROUP + simd_group * TILE_COLS;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + 7 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[col_base + 7] = static_cast<bfloat>(acc7);
        }
    }
}
"#;

const METAL_LORA_DELTA_DECODE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_lora_hidden_decode_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* a [[buffer(1)]],
    device bfloat* hidden [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& input_dim [[buffer(4)]],
    constant uint& rank [[buffer(5)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint rank_idx = tgroup.x;
    const uint batch_idx = tgroup.y;
    if (batch_idx >= batch || rank_idx >= rank) {
        return;
    }

    float acc = 0.0f;
    const uint x_base = batch_idx * input_dim;
    const uint a_base = rank_idx * input_dim;
    for (uint col = lane; col < input_dim; col += 32) {
        acc += static_cast<float>(x[x_base + col]) * static_cast<float>(a[a_base + col]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        hidden[batch_idx * rank + rank_idx] = static_cast<bfloat>(acc);
    }
}

kernel void kiln_lora_add_decode_bf16(
    device const bfloat* hidden [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device const bfloat* base [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& output_dim [[buffer(6)]],
    constant uint& rank [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = batch * output_dim;
    if (gid >= total) {
        return;
    }
    const uint batch_idx = gid / output_dim;
    const uint output_idx = gid - batch_idx * output_dim;

    float delta = 0.0f;
    const uint hidden_base = batch_idx * rank;
    const uint b_base = output_idx * rank;
    for (uint r = 0; r < rank; ++r) {
        delta += static_cast<float>(hidden[hidden_base + r]) * static_cast<float>(b[b_base + r]);
    }
    const bfloat delta_bf16 = static_cast<bfloat>(scale * delta);
    out[gid] = static_cast<bfloat>(static_cast<float>(base[gid]) + static_cast<float>(delta_bf16));
}
"#;

const METAL_GDN_IN_PROJ_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_in_proj_decode_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* qkv_t [[buffer(1)]],
    device const bfloat* z_t [[buffer(2)]],
    device const bfloat* a_t [[buffer(3)]],
    device const bfloat* b_t [[buffer(4)]],
    device bfloat* qkv_out [[buffer(5)]],
    device bfloat* z_out [[buffer(6)]],
    device bfloat* a_out [[buffer(7)]],
    device bfloat* b_out [[buffer(8)]],
    constant uint& hidden [[buffer(9)]],
    constant uint& qkv_dim [[buffer(10)]],
    constant uint& z_dim [[buffer(11)]],
    constant uint& nv [[buffer(12)]],
    constant uint& batch [[buffer(13)]],
    constant uint& row_pair_mode [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (batch == 1) {
        if (row_pair_mode == 6 || row_pair_mode == 7) {
            const bool x2_mode = row_pair_mode == 7;
            const uint qkv_pairs = qkv_dim >> 1;
            const uint z_pairs = z_dim >> 1;
            const uint total = qkv_pairs + z_pairs + (nv * 2);
            if (gid >= total) {
                return;
            }

            if (gid < qkv_pairs) {
                const uint col0 = gid << 1;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                if (x2_mode) {
                    for (uint i = 0; i < hidden; i += 2) {
                        const bfloat2 xv = *(device const bfloat2*)(x + i);
                        const float x0 = static_cast<float>(xv[0]);
                        const float x1 = static_cast<float>(xv[1]);
                        const uint w_idx0 = i * qkv_dim + col0;
                        const uint w_idx1 = w_idx0 + qkv_dim;
                        const bfloat2 w0 = *(device const bfloat2*)(qkv_t + w_idx0);
                        const bfloat2 w1 = *(device const bfloat2*)(qkv_t + w_idx1);
                        acc0 += x0 * static_cast<float>(w0[0]) + x1 * static_cast<float>(w1[0]);
                        acc1 += x0 * static_cast<float>(w0[1]) + x1 * static_cast<float>(w1[1]);
                    }
                } else {
                    for (uint i = 0; i < hidden; ++i) {
                        const float xv = static_cast<float>(x[i]);
                        const uint w_idx = i * qkv_dim + col0;
                        const bfloat2 w = *(device const bfloat2*)(qkv_t + w_idx);
                        acc0 += xv * static_cast<float>(w[0]);
                        acc1 += xv * static_cast<float>(w[1]);
                    }
                }
                qkv_out[col0] = static_cast<bfloat>(acc0);
                qkv_out[col0 + 1] = static_cast<bfloat>(acc1);
            } else if (gid < qkv_pairs + z_pairs) {
                const uint col0 = (gid - qkv_pairs) << 1;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                if (x2_mode) {
                    for (uint i = 0; i < hidden; i += 2) {
                        const bfloat2 xv = *(device const bfloat2*)(x + i);
                        const float x0 = static_cast<float>(xv[0]);
                        const float x1 = static_cast<float>(xv[1]);
                        const uint w_idx0 = i * z_dim + col0;
                        const uint w_idx1 = w_idx0 + z_dim;
                        const bfloat2 w0 = *(device const bfloat2*)(z_t + w_idx0);
                        const bfloat2 w1 = *(device const bfloat2*)(z_t + w_idx1);
                        acc0 += x0 * static_cast<float>(w0[0]) + x1 * static_cast<float>(w1[0]);
                        acc1 += x0 * static_cast<float>(w0[1]) + x1 * static_cast<float>(w1[1]);
                    }
                } else {
                    for (uint i = 0; i < hidden; ++i) {
                        const float xv = static_cast<float>(x[i]);
                        const uint w_idx = i * z_dim + col0;
                        const bfloat2 w = *(device const bfloat2*)(z_t + w_idx);
                        acc0 += xv * static_cast<float>(w[0]);
                        acc1 += xv * static_cast<float>(w[1]);
                    }
                }
                z_out[col0] = static_cast<bfloat>(acc0);
                z_out[col0 + 1] = static_cast<bfloat>(acc1);
            } else if (gid < qkv_pairs + z_pairs + nv) {
                const uint col = gid - qkv_pairs - z_pairs;
                float acc = 0.0f;
                if (x2_mode) {
                    for (uint i = 0; i < hidden; i += 2) {
                        const bfloat2 xv = *(device const bfloat2*)(x + i);
                        acc += static_cast<float>(xv[0]) * static_cast<float>(a_t[i * nv + col]);
                        acc += static_cast<float>(xv[1]) * static_cast<float>(a_t[(i + 1) * nv + col]);
                    }
                } else {
                    for (uint i = 0; i < hidden; ++i) {
                        acc += static_cast<float>(x[i]) * static_cast<float>(a_t[i * nv + col]);
                    }
                }
                a_out[col] = static_cast<bfloat>(acc);
            } else {
                const uint col = gid - qkv_pairs - z_pairs - nv;
                float acc = 0.0f;
                if (x2_mode) {
                    for (uint i = 0; i < hidden; i += 2) {
                        const bfloat2 xv = *(device const bfloat2*)(x + i);
                        acc += static_cast<float>(xv[0]) * static_cast<float>(b_t[i * nv + col]);
                        acc += static_cast<float>(xv[1]) * static_cast<float>(b_t[(i + 1) * nv + col]);
                    }
                } else {
                    for (uint i = 0; i < hidden; ++i) {
                        acc += static_cast<float>(x[i]) * static_cast<float>(b_t[i * nv + col]);
                    }
                }
                b_out[col] = static_cast<bfloat>(acc);
            }
        } else {
            const uint total = qkv_dim + z_dim + (nv * 2);
            if (gid >= total) {
                return;
            }

            float acc = 0.0f;
            if (gid < qkv_dim) {
                const uint col = gid;
                for (uint i = 0; i < hidden; ++i) {
                    acc += static_cast<float>(x[i]) * static_cast<float>(qkv_t[i * qkv_dim + col]);
                }
                qkv_out[col] = static_cast<bfloat>(acc);
            } else if (gid < qkv_dim + z_dim) {
                const uint col = gid - qkv_dim;
                for (uint i = 0; i < hidden; ++i) {
                    acc += static_cast<float>(x[i]) * static_cast<float>(z_t[i * z_dim + col]);
                }
                z_out[col] = static_cast<bfloat>(acc);
            } else if (gid < qkv_dim + z_dim + nv) {
                const uint col = gid - qkv_dim - z_dim;
                for (uint i = 0; i < hidden; ++i) {
                    acc += static_cast<float>(x[i]) * static_cast<float>(a_t[i * nv + col]);
                }
                a_out[col] = static_cast<bfloat>(acc);
            } else {
                const uint col = gid - qkv_dim - z_dim - nv;
                for (uint i = 0; i < hidden; ++i) {
                    acc += static_cast<float>(x[i]) * static_cast<float>(b_t[i * nv + col]);
                }
                b_out[col] = static_cast<bfloat>(acc);
            }
        }
        return;
    }

    if (row_pair_mode == 0) {
        const uint qkv_pairs = (qkv_dim + 1) >> 1;
        const uint z_pairs = (z_dim + 1) >> 1;
        const uint total = qkv_pairs + z_pairs + (nv * 2);
        if (gid >= total * batch) {
            return;
        }
        const uint batch_idx = gid / total;
        const uint local_gid = gid - batch_idx * total;
        const uint x_base = batch_idx * hidden;

        if (local_gid < qkv_pairs) {
            const uint col0 = local_gid << 1;
            const uint col1 = col0 + 1;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float xv = static_cast<float>(x[x_base + i]);
                const uint w_idx = i * qkv_dim + col0;
                acc0 += xv * static_cast<float>(qkv_t[w_idx]);
                if (col1 < qkv_dim) {
                    acc1 += xv * static_cast<float>(qkv_t[w_idx + 1]);
                }
            }
            const uint out_base = batch_idx * qkv_dim;
            qkv_out[out_base + col0] = static_cast<bfloat>(acc0);
            if (col1 < qkv_dim) {
                qkv_out[out_base + col1] = static_cast<bfloat>(acc1);
            }
        } else if (local_gid < qkv_pairs + z_pairs) {
            const uint local_z = local_gid - qkv_pairs;
            const uint col0 = local_z << 1;
            const uint col1 = col0 + 1;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float xv = static_cast<float>(x[x_base + i]);
                const uint w_idx = i * z_dim + col0;
                acc0 += xv * static_cast<float>(z_t[w_idx]);
                if (col1 < z_dim) {
                    acc1 += xv * static_cast<float>(z_t[w_idx + 1]);
                }
            }
            const uint out_base = batch_idx * z_dim;
            z_out[out_base + col0] = static_cast<bfloat>(acc0);
            if (col1 < z_dim) {
                z_out[out_base + col1] = static_cast<bfloat>(acc1);
            }
        } else if (local_gid < qkv_pairs + z_pairs + nv) {
            const uint col = local_gid - qkv_pairs - z_pairs;
            float acc = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                acc += static_cast<float>(x[x_base + i]) * static_cast<float>(a_t[i * nv + col]);
            }
            a_out[batch_idx * nv + col] = static_cast<bfloat>(acc);
        } else {
            const uint col = local_gid - qkv_pairs - z_pairs - nv;
            float acc = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                acc += static_cast<float>(x[x_base + i]) * static_cast<float>(b_t[i * nv + col]);
            }
            b_out[batch_idx * nv + col] = static_cast<bfloat>(acc);
        }
        return;
    }

    const uint qkv_pairs = (qkv_dim + 1) >> 1;
    const uint z_pairs = (z_dim + 1) >> 1;
    const uint total = qkv_pairs + z_pairs + (nv * 2);
    if (row_pair_mode == 3) {
        if (gid >= total) {
            return;
        }
        const uint local_gid = gid;
        const uint x_base1 = hidden;
        const uint x_base2 = hidden << 1;

        if (local_gid < qkv_pairs) {
            const uint col0 = local_gid << 1;
            const uint col1 = col0 + 1;
            const bool has_col1 = col1 < qkv_dim;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc20 = 0.0f;
            float acc21 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx = i * qkv_dim + col0;
                const float w0 = static_cast<float>(qkv_t[w_idx]);
                const float w1 = has_col1 ? static_cast<float>(qkv_t[w_idx + 1]) : 0.0f;
                const float xv0 = static_cast<float>(x[i]);
                acc00 += xv0 * w0;
                acc01 += xv0 * w1;
                const float xv1 = static_cast<float>(x[x_base1 + i]);
                acc10 += xv1 * w0;
                acc11 += xv1 * w1;
                const float xv2 = static_cast<float>(x[x_base2 + i]);
                acc20 += xv2 * w0;
                acc21 += xv2 * w1;
            }
            qkv_out[col0] = static_cast<bfloat>(acc00);
            if (has_col1) {
                qkv_out[col1] = static_cast<bfloat>(acc01);
            }
            const uint out_base1 = qkv_dim;
            qkv_out[out_base1 + col0] = static_cast<bfloat>(acc10);
            if (has_col1) {
                qkv_out[out_base1 + col1] = static_cast<bfloat>(acc11);
            }
            const uint out_base2 = qkv_dim << 1;
            qkv_out[out_base2 + col0] = static_cast<bfloat>(acc20);
            if (has_col1) {
                qkv_out[out_base2 + col1] = static_cast<bfloat>(acc21);
            }
        } else if (local_gid < qkv_pairs + z_pairs) {
            const uint local_z = local_gid - qkv_pairs;
            const uint col0 = local_z << 1;
            const uint col1 = col0 + 1;
            const bool has_col1 = col1 < z_dim;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc20 = 0.0f;
            float acc21 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx = i * z_dim + col0;
                const float w0 = static_cast<float>(z_t[w_idx]);
                const float w1 = has_col1 ? static_cast<float>(z_t[w_idx + 1]) : 0.0f;
                const float xv0 = static_cast<float>(x[i]);
                acc00 += xv0 * w0;
                acc01 += xv0 * w1;
                const float xv1 = static_cast<float>(x[x_base1 + i]);
                acc10 += xv1 * w0;
                acc11 += xv1 * w1;
                const float xv2 = static_cast<float>(x[x_base2 + i]);
                acc20 += xv2 * w0;
                acc21 += xv2 * w1;
            }
            z_out[col0] = static_cast<bfloat>(acc00);
            if (has_col1) {
                z_out[col1] = static_cast<bfloat>(acc01);
            }
            const uint out_base1 = z_dim;
            z_out[out_base1 + col0] = static_cast<bfloat>(acc10);
            if (has_col1) {
                z_out[out_base1 + col1] = static_cast<bfloat>(acc11);
            }
            const uint out_base2 = z_dim << 1;
            z_out[out_base2 + col0] = static_cast<bfloat>(acc20);
            if (has_col1) {
                z_out[out_base2 + col1] = static_cast<bfloat>(acc21);
            }
        } else if (local_gid < qkv_pairs + z_pairs + nv) {
            const uint col = local_gid - qkv_pairs - z_pairs;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float w = static_cast<float>(a_t[i * nv + col]);
                acc0 += static_cast<float>(x[i]) * w;
                acc1 += static_cast<float>(x[x_base1 + i]) * w;
                acc2 += static_cast<float>(x[x_base2 + i]) * w;
            }
            a_out[col] = static_cast<bfloat>(acc0);
            a_out[nv + col] = static_cast<bfloat>(acc1);
            a_out[(nv << 1) + col] = static_cast<bfloat>(acc2);
        } else {
            const uint col = local_gid - qkv_pairs - z_pairs - nv;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float w = static_cast<float>(b_t[i * nv + col]);
                acc0 += static_cast<float>(x[i]) * w;
                acc1 += static_cast<float>(x[x_base1 + i]) * w;
                acc2 += static_cast<float>(x[x_base2 + i]) * w;
            }
            b_out[col] = static_cast<bfloat>(acc0);
            b_out[nv + col] = static_cast<bfloat>(acc1);
            b_out[(nv << 1) + col] = static_cast<bfloat>(acc2);
        }
        return;
    }

    if (row_pair_mode == 4) {
        const uint row_quads = (batch + 3) >> 2;
        if (gid >= total * row_quads) {
            return;
        }
        const uint row_quad = gid / total;
        const uint local_gid = gid - row_quad * total;
        const uint row0 = row_quad << 2;
        const uint row1 = row0 + 1;
        const uint row2 = row0 + 2;
        const uint row3 = row0 + 3;
        const bool has_row1 = row1 < batch;
        const bool has_row2 = row2 < batch;
        const bool has_row3 = row3 < batch;
        const uint x_base0 = row0 * hidden;
        const uint x_base1 = row1 * hidden;
        const uint x_base2 = row2 * hidden;
        const uint x_base3 = row3 * hidden;

        if (local_gid < qkv_pairs) {
            const uint col0 = local_gid << 1;
            const uint col1 = col0 + 1;
            const bool has_col1 = col1 < qkv_dim;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc20 = 0.0f;
            float acc21 = 0.0f;
            float acc30 = 0.0f;
            float acc31 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx = i * qkv_dim + col0;
                const float w0 = static_cast<float>(qkv_t[w_idx]);
                const float w1 = has_col1 ? static_cast<float>(qkv_t[w_idx + 1]) : 0.0f;
                const float xv0 = static_cast<float>(x[x_base0 + i]);
                acc00 += xv0 * w0;
                acc01 += xv0 * w1;
                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    acc10 += xv1 * w0;
                    acc11 += xv1 * w1;
                }
                if (has_row2) {
                    const float xv2 = static_cast<float>(x[x_base2 + i]);
                    acc20 += xv2 * w0;
                    acc21 += xv2 * w1;
                }
                if (has_row3) {
                    const float xv3 = static_cast<float>(x[x_base3 + i]);
                    acc30 += xv3 * w0;
                    acc31 += xv3 * w1;
                }
            }
            const uint out_base0 = row0 * qkv_dim;
            qkv_out[out_base0 + col0] = static_cast<bfloat>(acc00);
            if (has_col1) {
                qkv_out[out_base0 + col1] = static_cast<bfloat>(acc01);
            }
            if (has_row1) {
                const uint out_base1 = row1 * qkv_dim;
                qkv_out[out_base1 + col0] = static_cast<bfloat>(acc10);
                if (has_col1) {
                    qkv_out[out_base1 + col1] = static_cast<bfloat>(acc11);
                }
            }
            if (has_row2) {
                const uint out_base2 = row2 * qkv_dim;
                qkv_out[out_base2 + col0] = static_cast<bfloat>(acc20);
                if (has_col1) {
                    qkv_out[out_base2 + col1] = static_cast<bfloat>(acc21);
                }
            }
            if (has_row3) {
                const uint out_base3 = row3 * qkv_dim;
                qkv_out[out_base3 + col0] = static_cast<bfloat>(acc30);
                if (has_col1) {
                    qkv_out[out_base3 + col1] = static_cast<bfloat>(acc31);
                }
            }
        } else if (local_gid < qkv_pairs + z_pairs) {
            const uint local_z = local_gid - qkv_pairs;
            const uint col0 = local_z << 1;
            const uint col1 = col0 + 1;
            const bool has_col1 = col1 < z_dim;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc20 = 0.0f;
            float acc21 = 0.0f;
            float acc30 = 0.0f;
            float acc31 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx = i * z_dim + col0;
                const float w0 = static_cast<float>(z_t[w_idx]);
                const float w1 = has_col1 ? static_cast<float>(z_t[w_idx + 1]) : 0.0f;
                const float xv0 = static_cast<float>(x[x_base0 + i]);
                acc00 += xv0 * w0;
                acc01 += xv0 * w1;
                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    acc10 += xv1 * w0;
                    acc11 += xv1 * w1;
                }
                if (has_row2) {
                    const float xv2 = static_cast<float>(x[x_base2 + i]);
                    acc20 += xv2 * w0;
                    acc21 += xv2 * w1;
                }
                if (has_row3) {
                    const float xv3 = static_cast<float>(x[x_base3 + i]);
                    acc30 += xv3 * w0;
                    acc31 += xv3 * w1;
                }
            }
            const uint out_base0 = row0 * z_dim;
            z_out[out_base0 + col0] = static_cast<bfloat>(acc00);
            if (has_col1) {
                z_out[out_base0 + col1] = static_cast<bfloat>(acc01);
            }
            if (has_row1) {
                const uint out_base1 = row1 * z_dim;
                z_out[out_base1 + col0] = static_cast<bfloat>(acc10);
                if (has_col1) {
                    z_out[out_base1 + col1] = static_cast<bfloat>(acc11);
                }
            }
            if (has_row2) {
                const uint out_base2 = row2 * z_dim;
                z_out[out_base2 + col0] = static_cast<bfloat>(acc20);
                if (has_col1) {
                    z_out[out_base2 + col1] = static_cast<bfloat>(acc21);
                }
            }
            if (has_row3) {
                const uint out_base3 = row3 * z_dim;
                z_out[out_base3 + col0] = static_cast<bfloat>(acc30);
                if (has_col1) {
                    z_out[out_base3 + col1] = static_cast<bfloat>(acc31);
                }
            }
        } else if (local_gid < qkv_pairs + z_pairs + nv) {
            const uint col = local_gid - qkv_pairs - z_pairs;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float w = static_cast<float>(a_t[i * nv + col]);
                acc0 += static_cast<float>(x[x_base0 + i]) * w;
                if (has_row1) {
                    acc1 += static_cast<float>(x[x_base1 + i]) * w;
                }
                if (has_row2) {
                    acc2 += static_cast<float>(x[x_base2 + i]) * w;
                }
                if (has_row3) {
                    acc3 += static_cast<float>(x[x_base3 + i]) * w;
                }
            }
            a_out[row0 * nv + col] = static_cast<bfloat>(acc0);
            if (has_row1) {
                a_out[row1 * nv + col] = static_cast<bfloat>(acc1);
            }
            if (has_row2) {
                a_out[row2 * nv + col] = static_cast<bfloat>(acc2);
            }
            if (has_row3) {
                a_out[row3 * nv + col] = static_cast<bfloat>(acc3);
            }
        } else {
            const uint col = local_gid - qkv_pairs - z_pairs - nv;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float w = static_cast<float>(b_t[i * nv + col]);
                acc0 += static_cast<float>(x[x_base0 + i]) * w;
                if (has_row1) {
                    acc1 += static_cast<float>(x[x_base1 + i]) * w;
                }
                if (has_row2) {
                    acc2 += static_cast<float>(x[x_base2 + i]) * w;
                }
                if (has_row3) {
                    acc3 += static_cast<float>(x[x_base3 + i]) * w;
                }
            }
            b_out[row0 * nv + col] = static_cast<bfloat>(acc0);
            if (has_row1) {
                b_out[row1 * nv + col] = static_cast<bfloat>(acc1);
            }
            if (has_row2) {
                b_out[row2 * nv + col] = static_cast<bfloat>(acc2);
            }
            if (has_row3) {
                b_out[row3 * nv + col] = static_cast<bfloat>(acc3);
            }
        }
        return;
    }

    const uint row_pairs = (batch + 1) >> 1;
    if (gid >= total * row_pairs) {
        return;
    }
    const uint row_pair = gid / total;
    const uint local_gid = gid - row_pair * total;
    const uint row0 = row_pair << 1;
    const uint row1 = row0 + 1;
    const bool has_row1 = row1 < batch;
    const uint x_base0 = row0 * hidden;
    const uint x_base1 = row1 * hidden;

    if (local_gid < qkv_pairs) {
        const uint col0 = local_gid << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < qkv_dim;
        float acc00 = 0.0f;
        float acc01 = 0.0f;
        float acc10 = 0.0f;
        float acc11 = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            const float xv0 = static_cast<float>(x[x_base0 + i]);
            const uint w_idx = i * qkv_dim + col0;
            const float w0 = static_cast<float>(qkv_t[w_idx]);
            acc00 += xv0 * w0;
            if (has_col1) {
                const float w1 = static_cast<float>(qkv_t[w_idx + 1]);
                acc01 += xv0 * w1;
                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    acc10 += xv1 * w0;
                    acc11 += xv1 * w1;
                }
            } else if (has_row1) {
                const float xv1 = static_cast<float>(x[x_base1 + i]);
                acc10 += xv1 * w0;
            }
        }
        const uint out_base0 = row0 * qkv_dim;
        qkv_out[out_base0 + col0] = static_cast<bfloat>(acc00);
        if (has_col1) {
            qkv_out[out_base0 + col1] = static_cast<bfloat>(acc01);
        }
        if (has_row1) {
            const uint out_base1 = row1 * qkv_dim;
            qkv_out[out_base1 + col0] = static_cast<bfloat>(acc10);
            if (has_col1) {
                qkv_out[out_base1 + col1] = static_cast<bfloat>(acc11);
            }
        }
    } else if (local_gid < qkv_pairs + z_pairs) {
        const uint local_z = local_gid - qkv_pairs;
        const uint col0 = local_z << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < z_dim;
        float acc00 = 0.0f;
        float acc01 = 0.0f;
        float acc10 = 0.0f;
        float acc11 = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            const float xv0 = static_cast<float>(x[x_base0 + i]);
            const uint w_idx = i * z_dim + col0;
            const float w0 = static_cast<float>(z_t[w_idx]);
            acc00 += xv0 * w0;
            if (has_col1) {
                const float w1 = static_cast<float>(z_t[w_idx + 1]);
                acc01 += xv0 * w1;
                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    acc10 += xv1 * w0;
                    acc11 += xv1 * w1;
                }
            } else if (has_row1) {
                const float xv1 = static_cast<float>(x[x_base1 + i]);
                acc10 += xv1 * w0;
            }
        }
        const uint out_base0 = row0 * z_dim;
        z_out[out_base0 + col0] = static_cast<bfloat>(acc00);
        if (has_col1) {
            z_out[out_base0 + col1] = static_cast<bfloat>(acc01);
        }
        if (has_row1) {
            const uint out_base1 = row1 * z_dim;
            z_out[out_base1 + col0] = static_cast<bfloat>(acc10);
            if (has_col1) {
                z_out[out_base1 + col1] = static_cast<bfloat>(acc11);
            }
        }
    } else if (local_gid < qkv_pairs + z_pairs + nv) {
        const uint col = local_gid - qkv_pairs - z_pairs;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            const float w = static_cast<float>(a_t[i * nv + col]);
            acc0 += static_cast<float>(x[x_base0 + i]) * w;
            if (has_row1) {
                acc1 += static_cast<float>(x[x_base1 + i]) * w;
            }
        }
        a_out[row0 * nv + col] = static_cast<bfloat>(acc0);
        if (has_row1) {
            a_out[row1 * nv + col] = static_cast<bfloat>(acc1);
        }
    } else {
        const uint col = local_gid - qkv_pairs - z_pairs - nv;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            const float w = static_cast<float>(b_t[i * nv + col]);
            acc0 += static_cast<float>(x[x_base0 + i]) * w;
            if (has_row1) {
                acc1 += static_cast<float>(x[x_base1 + i]) * w;
            }
        }
        b_out[row0 * nv + col] = static_cast<bfloat>(acc0);
        if (has_row1) {
            b_out[row1 * nv + col] = static_cast<bfloat>(acc1);
        }
    }
}
"#;

const METAL_PAGED_KV_HEAD_MAJOR_READ_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_head_major_read_bf16(
    device const bfloat* k_pool [[buffer(0)]],
    device const bfloat* v_pool [[buffer(1)]],
    device bfloat* k_out [[buffer(2)]],
    device bfloat* v_out [[buffer(3)]],
    constant uint& start_slot [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = seq_len * heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint d = gid % head_dim;
    const uint h = (gid / head_dim) % heads;
    const uint t = gid / (head_dim * heads);
    const uint pool_idx = ((start_slot + t) * heads + h) * head_dim + d;
    const uint out_idx = (h * seq_len + t) * head_dim + d;

    k_out[out_idx] = k_pool[pool_idx];
    v_out[out_idx] = v_pool[pool_idx];
}
"#;

const METAL_PAGED_KV_HEAD_MAJOR_READ_APPEND_TOKEN_MAJOR_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_head_major_read_append_token_major_bf16(
    device const bfloat* k_pool [[buffer(0)]],
    device const bfloat* v_pool [[buffer(1)]],
    device const bfloat* k_tail [[buffer(2)]],
    device const bfloat* v_tail [[buffer(3)]],
    device bfloat* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& start_slot [[buffer(6)]],
    constant uint& prefix_len [[buffer(7)]],
    constant uint& tail_len [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_len = prefix_len + tail_len;
    const uint total = total_len * heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint d = gid % head_dim;
    const uint h = (gid / head_dim) % heads;
    const uint t = gid / (head_dim * heads);
    const uint out_idx = (h * total_len + t) * head_dim + d;

    if (t < prefix_len) {
        const uint pool_idx = ((start_slot + t) * heads + h) * head_dim + d;
        k_out[out_idx] = k_pool[pool_idx];
        v_out[out_idx] = v_pool[pool_idx];
    } else {
        const uint tail_t = t - prefix_len;
        const uint tail_idx = (tail_t * heads + h) * head_dim + d;
        k_out[out_idx] = k_tail[tail_idx];
        v_out[out_idx] = v_tail[tail_idx];
    }
}
"#;

const METAL_PAGED_ATTN_DECODE_CONTIGUOUS_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_attn_decode_contiguous_bf16_d256(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k_pool [[buffer(1)]],
    device const bfloat* v_pool [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& start_slot [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& q_heads [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;
    constexpr uint QWEN_HEADS_PER_KV = 4;

    const uint head_idx = tid.y;
    if (head_idx >= q_heads) {
        return;
    }

    const uint kv_head_idx = head_idx / QWEN_HEADS_PER_KV;
    if (kv_head_idx >= kv_heads) {
        return;
    }

    thread float q_frag[EPT];
    thread float k_frag[EPT];
    thread float o_frag[EPT];
    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    device const bfloat* q_ptr = q + head_idx * D + simd_lid * EPT;
    device const bfloat* k_ptr =
        k_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device const bfloat* v_ptr =
        v_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device bfloat* out_ptr = out + head_idx * D + simd_gid * EPT;

    for (uint i = 0; i < EPT; ++i) {
        q_frag[i] = scale * static_cast<float>(q_ptr[i]);
        o_frag[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint t = simd_gid; t < seq_len; t += BN) {
        for (uint i = 0; i < EPT; ++i) {
            k_frag[i] = static_cast<float>(k_ptr[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < EPT; ++i) {
            score += q_frag[i] * k_frag[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint i = 0; i < EPT; ++i) {
            o_frag[i] = o_frag[i] * factor + exp_score * static_cast<float>(v_ptr[i]);
        }

        k_ptr += BN * kv_heads * D;
        v_ptr += BN * kv_heads * D;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint i = 0; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o_frag[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_frag[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < EPT; ++i) {
            out_ptr[i] = static_cast<bfloat>(o_frag[i]);
        }
    }
}

kernel void kiln_paged_attn_decode_contiguous_batch_bf16_d256(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k_pool [[buffer(1)]],
    device const bfloat* v_pool [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    device const uint* start_slots [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& q_heads [[buffer(7)]],
    constant uint& kv_heads [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& total_slots [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;
    constexpr uint QWEN_HEADS_PER_KV = 4;

    const uint batch_idx = tid.x;
    const uint head_idx = tid.y;
    if (batch_idx >= batch || head_idx >= q_heads) {
        return;
    }

    const uint kv_head_idx = head_idx / QWEN_HEADS_PER_KV;
    if (kv_head_idx >= kv_heads) {
        return;
    }

    device bfloat* out_ptr =
        out + (batch_idx * q_heads + head_idx) * D + simd_gid * EPT;
    const uint start_slot = start_slots[batch_idx];
    if (start_slot >= total_slots || seq_len == 0 || seq_len > total_slots - start_slot) {
        if (simd_lid == 0) {
            for (uint i = 0; i < EPT; ++i) {
                out_ptr[i] = static_cast<bfloat>(0.0f);
            }
        }
        return;
    }

    thread float q_frag[EPT];
    thread float k_frag[EPT];
    thread float o_frag[EPT];
    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    device const bfloat* q_ptr =
        q + (batch_idx * q_heads + head_idx) * D + simd_lid * EPT;
    device const bfloat* k_ptr =
        k_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device const bfloat* v_ptr =
        v_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;

    for (uint i = 0; i < EPT; ++i) {
        q_frag[i] = scale * static_cast<float>(q_ptr[i]);
        o_frag[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint t = simd_gid; t < seq_len; t += BN) {
        for (uint i = 0; i < EPT; ++i) {
            k_frag[i] = static_cast<float>(k_ptr[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < EPT; ++i) {
            score += q_frag[i] * k_frag[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint i = 0; i < EPT; ++i) {
            o_frag[i] = o_frag[i] * factor + exp_score * static_cast<float>(v_ptr[i]);
        }

        k_ptr += BN * kv_heads * D;
        v_ptr += BN * kv_heads * D;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint i = 0; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o_frag[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_frag[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < EPT; ++i) {
            out_ptr[i] = static_cast<bfloat>(o_frag[i]);
        }
    }
}

kernel void kiln_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k_pool [[buffer(1)]],
    device const bfloat* v_pool [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    device const uint* block_table [[buffer(4)]],
    device const int* seqused_k [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& max_blocks_per_seq [[buffer(7)]],
    constant uint& max_seqlen_k [[buffer(8)]],
    constant uint& page_block_size [[buffer(9)]],
    constant uint& q_heads [[buffer(10)]],
    constant uint& kv_heads [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant uint& total_slots [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;
    constexpr uint QWEN_HEADS_PER_KV = 4;

    const uint batch_idx = tid.x;
    const uint head_idx = tid.y;
    if (batch_idx >= batch || head_idx >= q_heads) {
        return;
    }

    const uint kv_head_idx = head_idx / QWEN_HEADS_PER_KV;
    if (kv_head_idx >= kv_heads) {
        return;
    }

    device bfloat* out_ptr =
        out + (batch_idx * q_heads + head_idx) * D + simd_gid * EPT;
    const int row_len_i = seqused_k[batch_idx];
    if (row_len_i <= 0 || page_block_size == 0 || max_blocks_per_seq == 0) {
        if (simd_lid == 0) {
            for (uint i = 0; i < EPT; ++i) {
                out_ptr[i] = static_cast<bfloat>(0.0f);
            }
        }
        return;
    }
    const uint row_len = min(static_cast<uint>(row_len_i), max_seqlen_k);

    thread float q_frag[EPT];
    thread float k_frag[EPT];
    thread float o_frag[EPT];
    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    device const bfloat* q_ptr =
        q + (batch_idx * q_heads + head_idx) * D + simd_lid * EPT;

    for (uint i = 0; i < EPT; ++i) {
        q_frag[i] = scale * static_cast<float>(q_ptr[i]);
        o_frag[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint t = simd_gid; t < row_len; t += BN) {
        const uint block_idx = t / page_block_size;
        const uint block_offset = t - block_idx * page_block_size;
        if (block_idx >= max_blocks_per_seq) {
            continue;
        }
        const uint physical_block = block_table[batch_idx * max_blocks_per_seq + block_idx];
        const uint pool_slot = physical_block * page_block_size + block_offset;
        if (pool_slot >= total_slots) {
            continue;
        }
        device const bfloat* k_ptr =
            k_pool + (pool_slot * kv_heads + kv_head_idx) * D + simd_lid * EPT;
        device const bfloat* v_ptr =
            v_pool + (pool_slot * kv_heads + kv_head_idx) * D + simd_lid * EPT;

        for (uint i = 0; i < EPT; ++i) {
            k_frag[i] = static_cast<float>(k_ptr[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < EPT; ++i) {
            score += q_frag[i] * k_frag[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint i = 0; i < EPT; ++i) {
            o_frag[i] = o_frag[i] * factor + exp_score * static_cast<float>(v_ptr[i]);
        }
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint i = 0; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o_frag[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_frag[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < EPT; ++i) {
            out_ptr[i] = static_cast<bfloat>(o_frag[i]);
        }
    }
}
"#;

const METAL_PAGED_KV_WRITE_TOKEN_MAJOR_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_write_token_major_bf16(
    device const bfloat* k_src [[buffer(0)]],
    device const bfloat* v_src [[buffer(1)]],
    device bfloat* k_pool [[buffer(2)]],
    device bfloat* v_pool [[buffer(3)]],
    constant uint& slot [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint pool_idx = slot * total + gid;
    k_pool[pool_idx] = k_src[gid];
    v_pool[pool_idx] = v_src[gid];
}

kernel void kiln_paged_kv_write_token_major_batch_bf16(
    device const bfloat* k_src [[buffer(0)]],
    device const bfloat* v_src [[buffer(1)]],
    device bfloat* k_pool [[buffer(2)]],
    device bfloat* v_pool [[buffer(3)]],
    device const uint* slots [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& total_slots [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint row_stride = heads * head_dim;
    const uint total = batch * row_stride;
    if (gid >= total) {
        return;
    }

    const uint batch_idx = gid / row_stride;
    const uint local = gid - batch_idx * row_stride;
    const uint slot = slots[batch_idx];
    if (slot >= total_slots) {
        return;
    }
    const uint pool_idx = slot * row_stride + local;
    k_pool[pool_idx] = k_src[gid];
    v_pool[pool_idx] = v_src[gid];
}
"#;

fn metal_shared_library(device: &dyn MetalPipelineHost) -> Result<Library> {
    static LIBRARIES: OnceLock<Mutex<HashMap<u64, Library>>> = OnceLock::new();
    let cache = LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal shared library cache poisoned"))?;
    if let Some(library) = cache.get(&device.pipeline_cache_key()) {
        return Ok(library.clone());
    }

    let shared_source = [
        METAL_RMSNORM_KERNEL,
        METAL_ROTARY_QK_KERNEL,
        METAL_GDN_QK_NORM_KERNEL,
        METAL_GDN_DECODE_QKV_CONV_NORM_KERNEL,
        METAL_GDN_GATES_KERNEL,
        METAL_GDN_DECODE_GATES_RECURRENT_KERNEL,
        METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM_KERNEL,
        METAL_GATED_RMSNORM_KERNEL,
        METAL_GDN_RECURRENT_KERNEL,
        METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_KERNEL,
        METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_DECAY_KERNEL,
        METAL_GDN_FULL_CHUNK_FORWARD_KERNEL,
        METAL_CONV1D_PREFILL_KERNEL,
        METAL_CONV1D_UPDATE_KERNEL,
        METAL_LM_HEAD_KERNEL,
        METAL_MLP_GATE_UP_KERNEL,
        METAL_ATTN_GATE_SIGMOID_MUL_KERNEL,
        METAL_TRANSPOSED_COOP_GEMV_KERNEL,
        METAL_FUSED_QKV_TRANSPOSED_COOP_GEMV_KERNEL,
        METAL_LORA_DELTA_DECODE_KERNEL,
        METAL_GDN_IN_PROJ_KERNEL,
        METAL_PAGED_KV_HEAD_MAJOR_READ_KERNEL,
        METAL_PAGED_KV_HEAD_MAJOR_READ_APPEND_TOKEN_MAJOR_KERNEL,
        METAL_PAGED_ATTN_DECODE_CONTIGUOUS_KERNEL,
        METAL_PAGED_KV_WRITE_TOKEN_MAJOR_KERNEL,
    ]
    .join("");
    let library = device
        .pipeline_raw_device()
        .new_library_with_source(&shared_source, None)
        .map_err(|e| anyhow::anyhow!("compile metal shared library: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), library.clone());
    Ok(library)
}

fn metal_rms_norm_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal rmsnorm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_rmsnorm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal rmsnorm function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal rmsnorm pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_rotary_qk_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal rotary qk pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_rotary_qk_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal rotary qk function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal rotary qk pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_qk_norm_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn qk norm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_qk_norm_f32_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn qk norm function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn qk norm pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_qk_norm_gqa_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn qk norm gqa pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_qk_norm_gqa_f32_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn qk norm gqa function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn qk norm gqa pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_decode_qkv_conv_norm_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn decode qkv conv/norm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_qkv_conv_norm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn decode qkv conv/norm function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn decode qkv conv/norm pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_argmax_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head argmax pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_chunks_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_argmax_reduce_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head argmax reduce pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_reduce_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax reduce function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax reduce pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_argmax_batch_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head argmax batch pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_chunks_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax batch function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax batch pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_argmax_reduce_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal lm head argmax reduce batch pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_reduce_batch_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax reduce batch function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax reduce batch pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_sample_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head sample pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_sample_topk_chunks_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head sample function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head sample pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lm_head_sample_reduce_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head sample reduce pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_sample_reduce_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head sample reduce function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head sample reduce pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_mlp_gate_up_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal mlp gate/up pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_mlp_gate_up_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal mlp gate/up function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal mlp gate/up pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_mlp_gate_up_serial_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal mlp gate/up serial pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_mlp_gate_up_serial_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal mlp gate/up serial function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal mlp gate/up serial pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_mlp_silu_mul_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal mlp silu*mul pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_mlp_silu_mul_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal mlp silu*mul function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal mlp silu*mul pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_attn_gate_sigmoid_mul_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal attn gate sigmoid/mul pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_attn_gate_sigmoid_mul_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal attn gate sigmoid/mul function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal attn gate sigmoid/mul pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_transposed_coop_gemv_pipeline(
    device: &dyn MetalPipelineHost,
    tile: MetalTransposedCoopGemvTile,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<
        Mutex<HashMap<(u64, MetalTransposedCoopGemvTile), ComputePipeline>>,
    > = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal transposed coop GEMV pipeline cache poisoned"))?;
    let key = (device.pipeline_cache_key(), tile);
    if let Some(pipeline) = cache.get(&key) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(tile.function_name(), None)
        .map_err(|e| anyhow::anyhow!("load metal transposed coop GEMV function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal transposed coop GEMV pipeline: {e:?}"))?;
    cache.insert(key, pipeline.clone());
    Ok(pipeline)
}

fn metal_transposed_coop_gemv_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal batch transposed coop GEMV cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_transposed_coop_gemv8_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal batch transposed coop GEMV function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal batch transposed coop GEMV pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_transposed_coop_gemv_batch_row_triple_tile8_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal batch transposed coop GEMV row-triple tile8 cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_transposed_coop_gemv8_batch_row_triple_tile8_bf16",
            None,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "load metal batch transposed coop GEMV row-triple tile8 function: {e:?}"
            )
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!(
                "build metal batch transposed coop GEMV row-triple tile8 pipeline: {e:?}"
            )
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_transposed_coop_gemv_batch_row_quad_tile8_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal batch transposed coop GEMV row-quad tile8 cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_transposed_coop_gemv8_batch_row_quad_tile8_bf16", None)
        .map_err(|e| {
            anyhow::anyhow!("load metal batch transposed coop GEMV row-quad tile8 function: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal batch transposed coop GEMV row-quad tile8 pipeline: {e:?}")
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_fused_qkv_transposed_coop_gemv_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal fused QKV projection pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_fused_qkv_transposed_coop_gemv8_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal fused QKV projection function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal fused QKV projection pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lora_hidden_decode_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal LoRA hidden decode pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lora_hidden_decode_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal LoRA hidden decode function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal LoRA hidden decode pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_lora_add_decode_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal LoRA add decode pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lora_add_decode_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal LoRA add decode function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal LoRA add decode pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_in_proj_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn in-proj pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_in_proj_decode_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn in-proj function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn in-proj pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_head_major_read_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv read pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_head_major_read_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv read function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv read pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_head_major_read_append_token_major_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv read+append pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_paged_kv_head_major_read_append_token_major_bf16",
            None,
        )
        .map_err(|e| anyhow::anyhow!("load metal paged kv read+append function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv read+append pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_attn_decode_contiguous_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal contiguous paged decode attention cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_attn_decode_contiguous_bf16_d256", None)
        .map_err(|e| anyhow::anyhow!("load metal contiguous paged decode attention: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal contiguous paged decode attention: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_attn_decode_contiguous_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal contiguous paged batch decode pipeline poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_attn_decode_contiguous_batch_bf16_d256", None)
        .map_err(|e| {
            anyhow::anyhow!("load metal contiguous paged batch decode attention: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal contiguous paged batch decode attention: {e:?}")
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal dyn-seqlen paged batch decode pipeline poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256",
            None,
        )
        .map_err(|e| {
            anyhow::anyhow!("load metal dyn-seqlen paged batch decode attention: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal dyn-seqlen paged batch decode attention: {e:?}")
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline_indirect(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal dyn-seqlen paged batch decode ICB pipeline poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256",
            None,
        )
        .map_err(|e| {
            anyhow::anyhow!("load metal dyn-seqlen paged batch decode ICB attention: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function_for_indirect_commands(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal dyn-seqlen paged batch decode ICB attention: {e:?}")
        })?;
    anyhow::ensure!(
        pipeline.supports_indirect_command_buffers(),
        "metal dyn-seqlen paged batch decode ICB pipeline did not enable indirect-command support"
    );
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_write_token_major_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv write pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv write function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv write pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_write_token_major_pipeline_indirect(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv write ICB pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv write ICB function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function_for_indirect_commands(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv write ICB pipeline: {e:?}"))?;
    anyhow::ensure!(
        pipeline.supports_indirect_command_buffers(),
        "metal paged kv write ICB pipeline did not enable indirect-command support"
    );
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_write_token_major_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv batch write pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv batch write function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv batch write pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_paged_kv_write_token_major_batch_pipeline_indirect(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv batch write ICB pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv batch write ICB function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function_for_indirect_commands(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv batch write ICB pipeline: {e:?}"))?;
    anyhow::ensure!(
        pipeline.supports_indirect_command_buffers(),
        "metal paged kv batch write ICB pipeline did not enable indirect-command support"
    );
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(crate) fn metal_lm_head_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(x.dtype(), kiln_tensor::DType::BF16)
        || !matches!(weight_t.dtype(), kiln_tensor::DType::BF16)
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((weight_hidden, vocab)) = weight_t.dims2() else {
        return false;
    };
    batch == 1
        && seq_len == 1
        && hidden == weight_hidden
        && hidden <= u32::MAX as usize
        && vocab <= u32::MAX as usize
}

pub(crate) fn metal_lm_head_argmax_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_lm_head_argmax_disabled() {
        return false;
    }
    if !metal_lm_head_supports(x, weight_t) {
        return false;
    }
    let Ok((_, vocab)) = weight_t.dims2() else {
        return false;
    };
    let num_groups = vocab.div_ceil(256);
    // The final reduction is intentionally bounded to one threadgroup for the
    // Qwen3.5-4B vocab path; larger vocabs fall back to materialized logits.
    num_groups > 0 && num_groups <= 1024
}

pub(crate) fn metal_lm_head_argmax_rows_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_lm_head_argmax_rows_disabled() {
        return false;
    }
    if !matches!(x.dtype(), kiln_tensor::DType::BF16)
        || !matches!(weight_t.dtype(), kiln_tensor::DType::BF16)
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((weight_hidden, vocab)) = weight_t.dims2() else {
        return false;
    };
    let num_groups = vocab.div_ceil(256);
    batch > 0
        && seq_len == 1
        && hidden == weight_hidden
        && hidden <= u32::MAX as usize
        && vocab <= u32::MAX as usize
        && num_groups > 0
        && num_groups <= 1024
}

pub(crate) fn metal_lm_head_sample_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    top_k: u32,
    temperature: f32,
    history_len: usize,
) -> bool {
    if metal_lm_head_sample_disabled()
        || top_k == 0
        || top_k > METAL_LM_HEAD_SAMPLE_TOP_K_MAX
        || !temperature.is_finite()
        || temperature <= 0.0
        || history_len > u32::MAX as usize
    {
        return false;
    }
    if !metal_lm_head_supports(x, weight_t) {
        return false;
    }
    let Ok((_, vocab)) = weight_t.dims2() else {
        return false;
    };
    let num_groups = vocab.div_ceil(256);
    num_groups > 0 && num_groups <= 1024
}

pub(crate) fn metal_lm_head_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_lm_head_supports(x, weight_t),
        "metal lm head supports only BF16 [1,1,H] x [H,V] on Metal"
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let x_metal = kt_metal(x)?;
    // The kernel writes every vocab element.
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[1usize, 1usize, vocab])?;

    let companion = x_metal.companion()?;
    let pipeline = metal_lm_head_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_lm_head_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(weight_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 lm_head-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let vocab_u32 = vocab as u32;
        encoder.set_bytes(3, &hidden_u32);
        encoder.set_bytes(4, &vocab_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: vocab,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_lm_head_argmax_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<u32> {
    anyhow::ensure!(
        metal_lm_head_argmax_supports(x, weight_t),
        "metal lm head argmax supports only BF16 [1,1,H] x [H,V] on Metal with <= 262144 vocab"
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let chunk_width = 256usize;
    let num_groups = vocab.div_ceil(chunk_width);
    let x_metal = kt_metal(x)?;
    let partial_scores = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[num_groups])?;
    let partial_indices = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[num_groups])?;
    let final_index = if metal_lm_head_argmax_gpu_reduce_disabled() {
        None
    } else {
        Some(kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[1usize])?)
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_lm_head_argmax_pipeline(&*companion)?;
    let reduce_pipeline = if final_index.is_some() {
        Some(metal_lm_head_argmax_reduce_pipeline(&*companion)?)
    } else {
        None
    };
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_lm_head_argmax_chunks_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(weight_t)?;
        let ps_metal = kt_metal(&partial_scores)?;
        let pi_metal = kt_metal(&partial_indices)?;
        let final_metal = match final_index.as_ref() {
            Some(t) => Some((kt_metal(t)?, t)),
            None => None,
        };

        // #1082 Step 4 lm_head-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let ps_buf = buffer_o_kt(
            ps_metal.buffer().as_ref(),
            partial_scores.layout(),
            partial_scores.dtype(),
        );
        let pi_buf = buffer_o_kt(
            pi_metal.buffer().as_ref(),
            partial_indices.layout(),
            partial_indices.dtype(),
        );
        let final_buf = final_metal.map(|(storage, tensor)| {
            buffer_o_kt(
                storage.buffer().as_ref(),
                tensor.layout(),
                kiln_tensor::DType::F32,
            )
        });

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(pi_buf.buffer), pi_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let vocab_u32 = vocab as u32;
        encoder.set_bytes(4, &hidden_u32);
        encoder.set_bytes(5, &vocab_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: num_groups,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: chunk_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);

        if let (Some(reduce_pipeline), Some(final_buf)) = (&reduce_pipeline, final_buf) {
            encoder.set_label("kiln_lm_head_argmax_reduce_f32");
            encoder.set_compute_pipeline_state(reduce_pipeline);
            encoder.set_buffer(0, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
            encoder.set_buffer(1, Some(pi_buf.buffer), pi_buf.offset_in_bytes);
            encoder.set_buffer(2, Some(final_buf.buffer), final_buf.offset_in_bytes);

            let num_groups_u32 = num_groups as u32;
            encoder.set_bytes(3, &num_groups_u32);

            let reduce_threadgroups = objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            let reduce_threads = objc2_metal::MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(reduce_threadgroups, reduce_threads);
        }
    }

    // Commit the argmax dispatch before the tiny synchronous readback. The
    // default path reduces chunk winners on-GPU and reads only one scalar.
    drop(encoder);

    if let Some(final_index) = final_index {
        let token = final_index
            .to_vec1::<f32>()
            .context("read metal lm head argmax final index")?
            .into_iter()
            .next()
            .context("metal lm head argmax final index missing")?;
        return Ok(token as u32);
    }

    let scores = partial_scores
        .to_vec1::<f32>()
        .context("read metal lm head argmax partial scores")?;
    let indices = partial_indices
        .to_vec1::<f32>()
        .context("read metal lm head argmax partial indices")?;

    let mut best_score = f32::NEG_INFINITY;
    let mut best_idx = 0u32;
    for (&score, &idx_f) in scores.iter().zip(indices.iter()) {
        let idx = idx_f as u32;
        if score > best_score || (score == best_score && idx < best_idx) {
            best_score = score;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_lm_head_sample_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    seed: u64,
) -> Result<u32> {
    anyhow::ensure!(
        history_indices.len() == history_counts.len(),
        "metal lm head sample history index/count length mismatch ({} vs {})",
        history_indices.len(),
        history_counts.len()
    );
    anyhow::ensure!(
        metal_lm_head_sample_supports(x, weight_t, top_k, temperature, history_indices.len()),
        "metal lm head sample supports BF16 [1,1,H] x [H,V] on Metal with top_k in 1..={}",
        METAL_LM_HEAD_SAMPLE_TOP_K_MAX
    );
    let (_, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let effective_top_k = (top_k as usize).min(vocab).max(1);
    let chunk_width = 256usize;
    let num_groups = vocab.div_ceil(chunk_width);
    let x_metal = kt_metal(x)?;
    let partial_scores = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::F32,
        &[num_groups, effective_top_k],
    )?;
    let partial_indices = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::F32,
        &[num_groups, effective_top_k],
    )?;
    let final_index = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[1usize])?;

    let device = x.device();
    let history_indices_tensor = if history_indices.is_empty() {
        kiln_tensor::Tensor::from_vec_on(device, vec![0u32], vec![1])?
    } else {
        kiln_tensor::Tensor::from_vec_on(
            device,
            history_indices.to_vec(),
            vec![history_indices.len()],
        )?
    };
    let history_counts_tensor = if history_counts.is_empty() {
        kiln_tensor::Tensor::from_vec_on(device, vec![0u32], vec![1])?
    } else {
        kiln_tensor::Tensor::from_vec_on(
            device,
            history_counts.to_vec(),
            vec![history_counts.len()],
        )?
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_lm_head_sample_pipeline(&*companion)?;
    let reduce_pipeline = metal_lm_head_sample_reduce_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_lm_head_sample_topk_chunks_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(weight_t)?;
        let hist_idx_metal = kt_metal(&history_indices_tensor)?;
        let hist_count_metal = kt_metal(&history_counts_tensor)?;
        let ps_metal = kt_metal(&partial_scores)?;
        let pi_metal = kt_metal(&partial_indices)?;
        let final_metal = kt_metal(&final_index)?;

        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let hist_idx_buf = buffer_o_kt(
            hist_idx_metal.buffer().as_ref(),
            history_indices_tensor.layout(),
            history_indices_tensor.dtype(),
        );
        let hist_count_buf = buffer_o_kt(
            hist_count_metal.buffer().as_ref(),
            history_counts_tensor.layout(),
            history_counts_tensor.dtype(),
        );
        let ps_buf = buffer_o_kt(
            ps_metal.buffer().as_ref(),
            partial_scores.layout(),
            partial_scores.dtype(),
        );
        let pi_buf = buffer_o_kt(
            pi_metal.buffer().as_ref(),
            partial_indices.layout(),
            partial_indices.dtype(),
        );
        let final_buf = buffer_o_kt(
            final_metal.buffer().as_ref(),
            final_index.layout(),
            final_index.dtype(),
        );

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(hist_idx_buf.buffer), hist_idx_buf.offset_in_bytes);
        encoder.set_buffer(
            3,
            Some(hist_count_buf.buffer),
            hist_count_buf.offset_in_bytes,
        );
        encoder.set_buffer(4, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(pi_buf.buffer), pi_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let vocab_u32 = vocab as u32;
        let history_len_u32 = history_indices.len() as u32;
        let inv_temperature = 1.0f32 / temperature;
        let effective_top_k_u32 = effective_top_k as u32;
        encoder.set_bytes(6, &hidden_u32);
        encoder.set_bytes(7, &vocab_u32);
        encoder.set_bytes(8, &history_len_u32);
        encoder.set_bytes(9, &repetition_penalty);
        encoder.set_bytes(10, &presence_penalty);
        encoder.set_bytes(11, &frequency_penalty);
        encoder.set_bytes(12, &inv_temperature);
        encoder.set_bytes(13, &effective_top_k_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: num_groups,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: chunk_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);

        encoder.set_label("kiln_lm_head_sample_reduce_f32");
        encoder.set_compute_pipeline_state(&reduce_pipeline);
        encoder.set_buffer(0, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(pi_buf.buffer), pi_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(final_buf.buffer), final_buf.offset_in_bytes);

        let num_groups_u32 = num_groups as u32;
        let seed_lo = seed as u32;
        let seed_hi = (seed >> 32) as u32;
        encoder.set_bytes(3, &num_groups_u32);
        encoder.set_bytes(4, &effective_top_k_u32);
        encoder.set_bytes(5, &top_p);
        encoder.set_bytes(6, &min_p);
        encoder.set_bytes(7, &seed_lo);
        encoder.set_bytes(8, &seed_hi);

        let reduce_threadgroups = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let reduce_threads = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(reduce_threadgroups, reduce_threads);
    }

    drop(encoder);

    let token = final_index
        .to_vec1::<f32>()
        .context("read metal lm head sampled final index")?
        .into_iter()
        .next()
        .context("metal lm head sampled final index missing")?;
    Ok(token as u32)
}

pub(crate) fn metal_lm_head_argmax_rows_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<Vec<u32>> {
    anyhow::ensure!(
        metal_lm_head_argmax_rows_supports(x, weight_t),
        "metal lm head row argmax supports only BF16 [B,1,H] x [H,V] on Metal with <= 262144 vocab"
    );
    let (batch, _, hidden) = x.dims3()?;
    let (_, vocab) = weight_t.dims2()?;

    let chunk_width = 256usize;
    let num_groups = vocab.div_ceil(chunk_width);
    let x_metal = kt_metal(x)?;
    let partial_scores = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[batch, num_groups])?;
    let partial_indices = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[batch, num_groups])?;
    let final_indices = if metal_lm_head_argmax_gpu_reduce_disabled() {
        None
    } else {
        Some(kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[batch])?)
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_lm_head_argmax_batch_pipeline(&*companion)?;
    let reduce_pipeline = if final_indices.is_some() {
        Some(metal_lm_head_argmax_reduce_batch_pipeline(&*companion)?)
    } else {
        None
    };
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_lm_head_argmax_chunks_batch_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(weight_t)?;
        let ps_metal = kt_metal(&partial_scores)?;
        let pi_metal = kt_metal(&partial_indices)?;
        let final_metal = match final_indices.as_ref() {
            Some(t) => Some((kt_metal(t)?, t)),
            None => None,
        };

        // #1082 Step 4 lm_head-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let ps_buf = buffer_o_kt(
            ps_metal.buffer().as_ref(),
            partial_scores.layout(),
            partial_scores.dtype(),
        );
        let pi_buf = buffer_o_kt(
            pi_metal.buffer().as_ref(),
            partial_indices.layout(),
            partial_indices.dtype(),
        );
        let final_buf = final_metal.map(|(storage, tensor)| {
            buffer_o_kt(
                storage.buffer().as_ref(),
                tensor.layout(),
                kiln_tensor::DType::F32,
            )
        });

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(pi_buf.buffer), pi_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let vocab_u32 = vocab as u32;
        let num_groups_u32 = num_groups as u32;
        encoder.set_bytes(4, &hidden_u32);
        encoder.set_bytes(5, &vocab_u32);
        encoder.set_bytes(6, &num_groups_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: num_groups,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: chunk_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);

        if let (Some(reduce_pipeline), Some(final_buf)) = (&reduce_pipeline, final_buf) {
            encoder.set_label("kiln_lm_head_argmax_reduce_batch_f32");
            encoder.set_compute_pipeline_state(reduce_pipeline);
            encoder.set_buffer(0, Some(ps_buf.buffer), ps_buf.offset_in_bytes);
            encoder.set_buffer(1, Some(pi_buf.buffer), pi_buf.offset_in_bytes);
            encoder.set_buffer(2, Some(final_buf.buffer), final_buf.offset_in_bytes);
            encoder.set_bytes(3, &num_groups_u32);

            let reduce_threadgroups = objc2_metal::MTLSize {
                width: batch,
                height: 1,
                depth: 1,
            };
            let reduce_threads = objc2_metal::MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(reduce_threadgroups, reduce_threads);
        }
    }

    drop(encoder);

    if let Some(final_indices) = final_indices {
        return Ok(final_indices
            .to_vec1::<f32>()
            .context("read metal lm head row argmax final indices")?
            .into_iter()
            .map(|idx| idx as u32)
            .collect());
    }

    let scores = partial_scores
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read metal lm head row argmax partial scores")?;
    let indices = partial_indices
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read metal lm head row argmax partial indices")?;
    let mut out = Vec::with_capacity(batch);
    for row in 0..batch {
        let row_start = row * num_groups;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0u32;
        for group in 0..num_groups {
            let offset = row_start + group;
            let score = scores[offset];
            let idx = indices[offset] as u32;
            if score > best_score || (score == best_score && idx < best_idx) {
                best_score = score;
                best_idx = idx;
            }
        }
        out.push(best_idx);
    }
    Ok(out)
}

pub(crate) fn metal_mlp_gate_up_supports(
    x: &kiln_tensor::Tensor,
    gate_t: &kiln_tensor::Tensor,
    up_t: &kiln_tensor::Tensor,
) -> bool {
    if x.dtype() != kiln_tensor::DType::BF16
        || gate_t.dtype() != kiln_tensor::DType::BF16
        || up_t.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(gate_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(up_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !gate_t.is_contiguous() || !up_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((gate_hidden, intermediate)) = gate_t.dims2() else {
        return false;
    };
    let Ok((up_hidden, up_intermediate)) = up_t.dims2() else {
        return false;
    };
    let Some(rows) = batch.checked_mul(seq_len) else {
        return false;
    };
    let Some(total) = rows.checked_mul(intermediate) else {
        return false;
    };

    rows > 0
        && seq_len == 1
        && hidden == gate_hidden
        && hidden == up_hidden
        && intermediate == up_intermediate
        && hidden <= u32::MAX as usize
        && intermediate <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(crate) fn metal_mlp_gate_up_bf16(
    x: &kiln_tensor::Tensor,
    gate_t: &kiln_tensor::Tensor,
    up_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_mlp_gate_up_supports(x, gate_t, up_t),
        "metal mlp gate/up supports only BF16 [B,1,H] x [H,I] on Metal"
    );
    let (batch, seq_len, hidden) = x.dims3()?;
    let (_, intermediate) = gate_t.dims2()?;
    let rows = batch * seq_len;
    let row_group_size = if rows == 3
        && !metal_mlp_gate_up_row_pair_disabled()
        && !metal_mlp_gate_up_row_triple_disabled()
    {
        3
    } else if rows >= 5
        && !metal_mlp_gate_up_row_pair_disabled()
        && !metal_mlp_gate_up_row_quad_disabled()
    {
        4
    } else if rows > 1 && !metal_mlp_gate_up_row_pair_disabled() {
        2
    } else {
        1
    };
    let row_groups = rows.div_ceil(row_group_size);
    let total = row_groups * intermediate.div_ceil(2);

    let x_metal = kt_metal(x)?;
    // The kernel writes every row/intermediate element.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, intermediate],
    )?;

    let companion = x_metal.companion()?;
    let encoder = companion.command_encoder()?;

    {
        let gate_metal = kt_metal(gate_t)?;
        let up_metal = kt_metal(up_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 mlp-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let gate_buf = buffer_o_kt(
            gate_metal.buffer().as_ref(),
            gate_t.layout(),
            gate_t.dtype(),
        );
        let up_buf = buffer_o_kt(up_metal.buffer().as_ref(), up_t.layout(), up_t.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(up_buf.buffer), up_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let intermediate_u32 = intermediate as u32;

        let serial_vector_safe = rows == 1
            && !metal_mlp_gate_up_serial_vector_load_disabled()
            && intermediate % 2 == 0
            && gate_buf.offset_in_bytes % 4 == 0
            && up_buf.offset_in_bytes % 4 == 0;
        let serial_dedicated = serial_vector_safe && !metal_mlp_gate_up_serial_dedicated_disabled();
        if serial_dedicated {
            let pipeline = metal_mlp_gate_up_serial_pipeline(&*companion)?;
            encoder.set_label("kiln_mlp_gate_up_serial_bf16");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(4, &hidden_u32);
            encoder.set_bytes(5, &intermediate_u32);
        } else {
            let pipeline = metal_mlp_gate_up_pipeline(&*companion)?;
            encoder.set_label("kiln_mlp_gate_up_bf16");
            encoder.set_compute_pipeline_state(&pipeline);
            let rows_u32 = rows as u32;
            let row_pair_mode_u32 = if row_group_size == 1 {
                if serial_vector_safe { 6 } else { 0 }
            } else if row_group_size == 3
                && intermediate % 2 == 0
                && gate_buf.offset_in_bytes % 4 == 0
                && up_buf.offset_in_bytes % 4 == 0
            {
                7
            } else if row_group_size == 4
                && !metal_mlp_gate_up_row_quad_vector_load_disabled()
                && intermediate % 2 == 0
                && gate_buf.offset_in_bytes % 4 == 0
                && up_buf.offset_in_bytes % 4 == 0
            {
                5
            } else {
                row_group_size as u32
            };
            encoder.set_bytes(4, &rows_u32);
            encoder.set_bytes(5, &hidden_u32);
            encoder.set_bytes(6, &intermediate_u32);
            encoder.set_bytes(7, &row_pair_mode_u32);
        }

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_mlp_silu_mul_supports(
    gate: &kiln_tensor::Tensor,
    up: &kiln_tensor::Tensor,
) -> bool {
    if metal_mlp_silu_mul_disabled() {
        return false;
    }
    if gate.dtype() != kiln_tensor::DType::BF16 || up.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(gate.device(), kiln_tensor::Device::Metal(_))
        || !matches!(up.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !gate.is_contiguous() || !up.is_contiguous() || gate.shape() != up.shape() {
        return false;
    }
    gate.elem_count() > 0 && gate.elem_count() <= u32::MAX as usize
}

pub(crate) fn metal_mlp_silu_mul_bf16(
    gate: &kiln_tensor::Tensor,
    up: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_mlp_silu_mul_supports(gate, up),
        "metal mlp silu*mul supports only matching contiguous BF16 Metal tensors"
    );
    let total = gate.elem_count();
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let gate_metal = kt_metal(&gate)?;
    let out = kt_metal_alloc(gate_metal, kiln_tensor::DType::BF16, gate.dims())?;

    let companion = gate_metal.companion()?;
    let pipeline = metal_mlp_silu_mul_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_mlp_silu_mul_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let up_metal = kt_metal(&up)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 mlp-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let gate_buf = buffer_o_kt(gate_metal.buffer().as_ref(), gate.layout(), gate.dtype());
        let up_buf = buffer_o_kt(up_metal.buffer().as_ref(), up.layout(), up.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(up_buf.buffer), up_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let total_u32 = total as u32;
        encoder.set_bytes(3, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_attn_gate_sigmoid_mul_supports(
    x: &kiln_tensor::Tensor,
    gate: &kiln_tensor::Tensor,
) -> bool {
    if metal_attn_gate_fusion_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || gate.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(gate.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !gate.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((gate_batch, gate_seq_len, gate_hidden)) = gate.dims3() else {
        return false;
    };
    let Some(rows) = batch.checked_mul(seq_len) else {
        return false;
    };
    let Some(total) = rows.checked_mul(hidden) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && gate_batch == batch
        && gate_seq_len == seq_len
        && gate_hidden == hidden
        && total <= u32::MAX as usize
}

pub(crate) fn metal_attn_gate_sigmoid_mul_bf16(
    x: &kiln_tensor::Tensor,
    gate: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_attn_gate_sigmoid_mul_supports(x, gate),
        "metal attn gate sigmoid/mul supports only BF16 [B,1,H] tensors on Metal"
    );
    let (batch, seq_len, hidden) = x.dims3()?;
    let total = batch * seq_len * hidden;

    // The kernel writes every hidden element exactly once.
    let x_metal = kt_metal(&x)?;
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, seq_len, hidden])?;

    let companion = x_metal.companion()?;
    let pipeline = metal_attn_gate_sigmoid_mul_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_attn_gate_sigmoid_mul_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let gate_metal = kt_metal(&gate)?;
        let out_metal = kt_metal(&out)?;

        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let gate_buf = buffer_o_kt(gate_metal.buffer().as_ref(), gate.layout(), gate.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let total_u32 = total as u32;
        encoder.set_bytes(3, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_transposed_coop_gemv_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_transposed_coop_gemv_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight_t.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((weight_input_dim, output_dim)) = weight_t.dims2() else {
        return false;
    };

    batch == 1
        && seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim == weight_input_dim
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
}

pub(crate) fn metal_transposed_coop_gemv_decode_batch_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_transposed_coop_gemv_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight_t.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((weight_input_dim, output_dim)) = weight_t.dims2() else {
        return false;
    };
    let Some(total) = batch.checked_mul(output_dim) else {
        return false;
    };

    batch > 1
        && seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim == weight_input_dim
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
        && batch <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(crate) fn metal_transposed_coop_gemv_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    if metal_transposed_coop_gemv_decode_batch_supports(x, weight_t) {
        return metal_transposed_coop_gemv_batch_bf16(x, weight_t);
    }

    let (_, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;
    metal_transposed_coop_gemv_bf16_with_tile(
        x,
        weight_t,
        metal_transposed_coop_gemv_select_tile(input_dim, output_dim),
    )
}

fn metal_transposed_coop_gemv_bf16_with_tile(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    tile: MetalTransposedCoopGemvTile,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_transposed_coop_gemv_supports(x, weight_t),
        "metal transposed coop GEMV supports only BF16 [1,1,K] x [K,N] on Metal"
    );
    let (_, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;

    let x_metal = kt_metal(&x)?;
    // The kernel writes every output channel exactly once.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[1usize, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let pipeline = metal_transposed_coop_gemv_pipeline(&*companion, tile)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label(tile.label());
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let input_dim_u32 = input_dim as u32;
        let output_dim_u32 = output_dim as u32;
        encoder.set_bytes(3, &input_dim_u32);
        encoder.set_bytes(4, &output_dim_u32);

        let cols_per_threadgroup = tile.tile_cols() * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: output_dim.div_ceil(cols_per_threadgroup),
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_transposed_coop_gemv_batch_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_transposed_coop_gemv_decode_batch_supports(x, weight_t),
        "metal batch transposed coop GEMV supports only BF16 [B,1,K] x [K,N] with B > 1 on Metal"
    );
    let (batch, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;
    let row_grouping_enabled = batch > 1 && !metal_transposed_coop_gemv_row_pair_disabled();
    let row_quad_enabled =
        row_grouping_enabled && batch >= 3 && !metal_transposed_coop_gemv_row_quad_disabled();
    let row_triple_tile8_enabled = row_quad_enabled
        && batch == 3
        && !metal_transposed_coop_gemv_row_quad_tile8_disabled()
        && !metal_transposed_coop_gemv_row_triple_tile8_disabled();
    let row_quad_tile8_enabled = row_quad_enabled
        && !row_triple_tile8_enabled
        && !metal_transposed_coop_gemv_row_quad_tile8_disabled();
    let row_group_size = if row_triple_tile8_enabled {
        3usize
    } else if row_quad_enabled {
        4usize
    } else if row_grouping_enabled {
        2usize
    } else {
        1usize
    };
    let row_groups = batch.div_ceil(row_group_size);

    let x_metal = kt_metal(&x)?;
    // The kernel writes every batch/output channel exactly once.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let pipeline = if row_triple_tile8_enabled {
        metal_transposed_coop_gemv_batch_row_triple_tile8_pipeline(&*companion)?
    } else if row_quad_tile8_enabled {
        metal_transposed_coop_gemv_batch_row_quad_tile8_pipeline(&*companion)?
    } else {
        metal_transposed_coop_gemv_batch_pipeline(&*companion)?
    };
    let encoder = companion.command_encoder()?;
    encoder.set_label(if row_triple_tile8_enabled {
        "kiln_transposed_coop_gemv8_batch_row_triple_tile8_bf16"
    } else if row_quad_tile8_enabled {
        "kiln_transposed_coop_gemv8_batch_row_quad_tile8_bf16"
    } else {
        "kiln_transposed_coop_gemv8_batch_bf16"
    });
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let input_dim_u32 = input_dim as u32;
        let output_dim_u32 = output_dim as u32;
        encoder.set_bytes(3, &input_dim_u32);
        encoder.set_bytes(4, &output_dim_u32);
        if row_quad_tile8_enabled {
            let batch_u32 = batch as u32;
            encoder.set_bytes(5, &batch_u32);
        } else if !row_triple_tile8_enabled {
            let row_pair_mode_u32 = if row_group_size > 1 { batch as u32 } else { 0 };
            let row_group_size_u32 = row_group_size as u32;
            encoder.set_bytes(5, &row_pair_mode_u32);
            encoder.set_bytes(6, &row_group_size_u32);
        }

        let tile_cols = if row_triple_tile8_enabled || row_quad_tile8_enabled {
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS
        } else if row_quad_enabled {
            METAL_TRANSPOSED_COOP_GEMV_TILE4_COLS
        } else {
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS
        };
        let cols_per_threadgroup = tile_cols * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: output_dim.div_ceil(cols_per_threadgroup),
            height: row_groups,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_fused_qkv_transposed_coop_gemv_supports(
    x: &kiln_tensor::Tensor,
    q_t: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    v_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_fused_qkv_proj_disabled() {
        return false;
    }
    if !metal_transposed_coop_gemv_supports(x, q_t)
        || !metal_transposed_coop_gemv_supports(x, k_t)
        || !metal_transposed_coop_gemv_supports(x, v_t)
    {
        return false;
    }

    let Ok((_, _, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((q_input_dim, _)) = q_t.dims2() else {
        return false;
    };
    let Ok((k_input_dim, _)) = k_t.dims2() else {
        return false;
    };
    let Ok((v_input_dim, _)) = v_t.dims2() else {
        return false;
    };
    input_dim == q_input_dim && input_dim == k_input_dim && input_dim == v_input_dim
}

pub(crate) fn metal_fused_qkv_transposed_coop_gemv_bf16(
    x: &kiln_tensor::Tensor,
    q_t: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    v_t: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_fused_qkv_transposed_coop_gemv_supports(x, q_t, k_t, v_t),
        "metal fused QKV projection supports only BF16 [1,1,K] x [K,Nq/Nk/Nv] on Metal"
    );
    let (_, _, input_dim) = x.dims3()?;
    let (_, q_output_dim) = q_t.dims2()?;
    let (_, k_output_dim) = k_t.dims2()?;
    let (_, v_output_dim) = v_t.dims2()?;

    let total_output_dim = q_output_dim + k_output_dim + v_output_dim;
    // The kernel writes each projection output independently with the existing
    // tile8 cooperative GEMV mapping. Back the three result views with one
    // allocation to avoid repeated small Metal buffer allocations in decode.
    let x_metal = kt_metal(&x)?;
    let fused_out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[1usize, 1usize, total_output_dim],
    )?;
    let q_out = fused_out.narrow(2, 0, q_output_dim)?;
    let k_out = fused_out.narrow(2, q_output_dim, k_output_dim)?;
    let v_out = fused_out.narrow(2, q_output_dim + k_output_dim, v_output_dim)?;

    let companion = x_metal.companion()?;
    let pipeline = metal_fused_qkv_transposed_coop_gemv_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_fused_qkv_transposed_coop_gemv8_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let q_metal = kt_metal(&q_t)?;
        let k_metal = kt_metal(&k_t)?;
        let v_metal = kt_metal(&v_t)?;
        let q_out_metal = kt_metal(&q_out)?;
        let k_out_metal = kt_metal(&k_out)?;
        let v_out_metal = kt_metal(&v_out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q_t.layout(), q_t.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_t.layout(), k_t.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_t.layout(), v_t.dtype());
        let q_out_buf = buffer_o_kt(q_out_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let k_out_buf = buffer_o_kt(k_out_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let v_out_buf = buffer_o_kt(v_out_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(q_out_buf.buffer), q_out_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(k_out_buf.buffer), k_out_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(v_out_buf.buffer), v_out_buf.offset_in_bytes);

        let input_dim_u32 = input_dim as u32;
        let q_output_dim_u32 = q_output_dim as u32;
        let k_output_dim_u32 = k_output_dim as u32;
        let v_output_dim_u32 = v_output_dim as u32;
        encoder.set_bytes(7, &input_dim_u32);
        encoder.set_bytes(8, &q_output_dim_u32);
        encoder.set_bytes(9, &k_output_dim_u32);
        encoder.set_bytes(10, &v_output_dim_u32);

        let cols_per_threadgroup =
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let q_groups = q_output_dim.div_ceil(cols_per_threadgroup);
        let k_groups = k_output_dim.div_ceil(cols_per_threadgroup);
        let v_groups = v_output_dim.div_ceil(cols_per_threadgroup);
        let total_groups = q_groups + k_groups + v_groups;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: total_groups,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out, v_out))
}

pub(crate) fn metal_lora_add_decode_supports(
    base: &kiln_tensor::Tensor,
    x: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
) -> bool {
    if metal_lora_delta_decode_disabled() {
        return false;
    }
    if base.dtype() != kiln_tensor::DType::BF16
        || x.dtype() != kiln_tensor::DType::BF16
        || a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(base.device(), kiln_tensor::Device::Metal(_))
        || !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !base.is_contiguous() || !x.is_contiguous() || !a.is_contiguous() || !b.is_contiguous() {
        return false;
    }

    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((base_batch, base_seq_len, output_dim)) = base.dims3() else {
        return false;
    };
    let Ok((rank, a_input_dim)) = a.dims2() else {
        return false;
    };
    let Ok((b_output_dim, b_rank)) = b.dims2() else {
        return false;
    };
    let Some(total_output) = batch.checked_mul(output_dim) else {
        return false;
    };
    let Some(hidden_total) = batch.checked_mul(rank) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && base_batch == batch
        && base_seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim >= 1024
        && output_dim >= 1024
        && rank > 0
        && a_input_dim == input_dim
        && b_output_dim == output_dim
        && b_rank == rank
        && batch <= u32::MAX as usize
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
        && rank <= u32::MAX as usize
        && total_output <= u32::MAX as usize
        && hidden_total <= u32::MAX as usize
}

pub(crate) fn metal_lora_add_decode_bf16(
    base: &kiln_tensor::Tensor,
    x: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_lora_add_decode_supports(base, x, a, b),
        "metal LoRA decode add supports only contiguous BF16 Metal base/x/A/B decode tensors"
    );
    let (batch, _, input_dim) = x.dims3()?;
    let (_, _, output_dim) = base.dims3()?;
    let (rank, _) = a.dims2()?;

    let x_metal = kt_metal(&x)?;
    let base_metal = kt_metal(&base)?;
    let hidden = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, rank])?;
    let out = kt_metal_alloc(
        base_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let encoder = companion.command_encoder()?;

    {
        let pipeline = metal_lora_hidden_decode_pipeline(&*companion)?;
        encoder.set_label("kiln_lora_hidden_decode_bf16");
        encoder.set_compute_pipeline_state(&pipeline);

        let a_metal = kt_metal(&a)?;
        let hidden_metal = kt_metal(&hidden)?;

        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let hidden_buf = buffer_o_kt(
            hidden_metal.buffer().as_ref(),
            hidden.layout(),
            hidden.dtype(),
        );

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(hidden_buf.buffer), hidden_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let input_dim_u32 = input_dim as u32;
        let rank_u32 = rank as u32;
        encoder.set_bytes(3, &batch_u32);
        encoder.set_bytes(4, &input_dim_u32);
        encoder.set_bytes(5, &rank_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: rank,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    {
        let pipeline = metal_lora_add_decode_pipeline(&*companion)?;
        encoder.set_label("kiln_lora_add_decode_bf16");
        encoder.set_compute_pipeline_state(&pipeline);

        let hidden_metal = kt_metal(&hidden)?;
        let b_metal = kt_metal(&b)?;
        let base_metal = kt_metal(&base)?;
        let out_metal = kt_metal(&out)?;

        let hidden_buf = buffer_o_kt(
            hidden_metal.buffer().as_ref(),
            hidden.layout(),
            hidden.dtype(),
        );
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let base_buf = buffer_o_kt(base_metal.buffer().as_ref(), base.layout(), base.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(hidden_buf.buffer), hidden_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(base_buf.buffer), base_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let output_dim_u32 = output_dim as u32;
        let rank_u32 = rank as u32;
        encoder.set_bytes(4, &scale);
        encoder.set_bytes(5, &batch_u32);
        encoder.set_bytes(6, &output_dim_u32);
        encoder.set_bytes(7, &rank_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch * output_dim,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    drop(encoder);
    Ok(out)
}

fn metal_gdn_in_proj_decode_supports(
    x: &kiln_tensor::Tensor,
    qkv_t: &kiln_tensor::Tensor,
    z_t: &kiln_tensor::Tensor,
    a_t: &kiln_tensor::Tensor,
    b_t: &kiln_tensor::Tensor,
) -> bool {
    if x.dtype() != kiln_tensor::DType::BF16
        || qkv_t.dtype() != kiln_tensor::DType::BF16
        || z_t.dtype() != kiln_tensor::DType::BF16
        || a_t.dtype() != kiln_tensor::DType::BF16
        || b_t.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(qkv_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(z_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous()
        || !qkv_t.is_contiguous()
        || !z_t.is_contiguous()
        || !a_t.is_contiguous()
        || !b_t.is_contiguous()
    {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((qkv_hidden, qkv_dim)) = qkv_t.dims2() else {
        return false;
    };
    let Ok((z_hidden, z_dim)) = z_t.dims2() else {
        return false;
    };
    let Ok((a_hidden, nv)) = a_t.dims2() else {
        return false;
    };
    let Ok((b_hidden, b_nv)) = b_t.dims2() else {
        return false;
    };
    let Some(total) = qkv_dim
        .checked_add(z_dim)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_add(b_nv))
    else {
        return false;
    };
    let Some(dispatch_total) = total.checked_mul(batch) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && hidden == qkv_hidden
        && hidden == z_hidden
        && hidden == a_hidden
        && hidden == b_hidden
        && nv == b_nv
        && hidden <= u32::MAX as usize
        && qkv_dim <= u32::MAX as usize
        && z_dim <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && total <= u32::MAX as usize
        && dispatch_total <= u32::MAX as usize
}

fn metal_gdn_in_proj_decode_bf16(
    x: &kiln_tensor::Tensor,
    qkv_t: &kiln_tensor::Tensor,
    z_t: &kiln_tensor::Tensor,
    a_t: &kiln_tensor::Tensor,
    b_t: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_in_proj_decode_supports(x, qkv_t, z_t, a_t, b_t),
        "metal gdn in-proj supports only BF16 [B,1,H] x [H,*] on Metal"
    );
    let (batch, _, hidden) = x.dims3()?;
    let (_, qkv_dim) = qkv_t.dims2()?;
    let (_, z_dim) = z_t.dims2()?;
    let (_, nv) = a_t.dims2()?;
    let output_total = qkv_dim + z_dim + (nv * 2);
    let row_grouping_enabled = batch >= 3 && !metal_gdn_in_proj_row_pair_disabled();
    let row_triple_enabled =
        batch == 3 && row_grouping_enabled && !metal_gdn_in_proj_row_triple_disabled();
    let row_quad_enabled =
        row_grouping_enabled && batch >= 8 && !metal_gdn_in_proj_row_quad_disabled();
    let row_group_size = if row_triple_enabled {
        3usize
    } else if row_quad_enabled {
        4usize
    } else if row_grouping_enabled {
        2usize
    } else {
        1usize
    };
    let x_metal = kt_metal(&x)?;
    // The kernel writes every output element exactly once. Keep bs=1 on one
    // backing allocation, but use separate batch outputs so each `[B,1,N]`
    // tensor remains contiguous for the following fused decode kernels.
    let (qkv_out, z_out, a_out, b_out) = if batch == 1 {
        let proj_out = kt_metal_alloc(
            x_metal,
            kiln_tensor::DType::BF16,
            &[1usize, 1usize, output_total],
        )?;
        (
            proj_out.narrow(2, 0, qkv_dim)?,
            proj_out.narrow(2, qkv_dim, z_dim)?,
            proj_out.narrow(2, qkv_dim + z_dim, nv)?,
            proj_out.narrow(2, qkv_dim + z_dim + nv, nv)?,
        )
    } else {
        (
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, qkv_dim])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, z_dim])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, nv])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, nv])?,
        )
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_gdn_in_proj_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_in_proj_decode_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let qkv_metal = kt_metal(&qkv_t)?;
        let z_metal = kt_metal(&z_t)?;
        let a_metal = kt_metal(&a_t)?;
        let b_metal = kt_metal(&b_t)?;
        let qkv_o_metal = kt_metal(&qkv_out)?;
        let z_o_metal = kt_metal(&z_out)?;
        let a_o_metal = kt_metal(&a_out)?;
        let b_o_metal = kt_metal(&b_out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let qkv_buf = buffer_o_kt(qkv_metal.buffer().as_ref(), qkv_t.layout(), qkv_t.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z_t.layout(), z_t.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a_t.layout(), a_t.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b_t.layout(), b_t.dtype());
        let qkv_o_buf = buffer_o_kt(
            qkv_o_metal.buffer().as_ref(),
            qkv_out.layout(),
            qkv_out.dtype(),
        );
        let z_o_buf = buffer_o_kt(z_o_metal.buffer().as_ref(), z_out.layout(), z_out.dtype());
        let a_o_buf = buffer_o_kt(a_o_metal.buffer().as_ref(), a_out.layout(), a_out.dtype());
        let b_o_buf = buffer_o_kt(b_o_metal.buffer().as_ref(), b_out.layout(), b_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(qkv_buf.buffer), qkv_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qkv_o_buf.buffer), qkv_o_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(z_o_buf.buffer), z_o_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(a_o_buf.buffer), a_o_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(b_o_buf.buffer), b_o_buf.offset_in_bytes);

        let serial_vector_mode = batch == 1
            && !metal_gdn_in_proj_serial_vector_load_disabled()
            && qkv_dim % 2 == 0
            && z_dim % 2 == 0
            && qkv_buf.offset_in_bytes % 4 == 0
            && z_buf.offset_in_bytes % 4 == 0;
        let serial_x2_mode = serial_vector_mode
            && !metal_gdn_in_proj_serial_x2_load_disabled()
            && hidden % 2 == 0
            && x_buf.offset_in_bytes % 4 == 0;
        let dispatch_cols = if serial_vector_mode {
            (qkv_dim / 2) + (z_dim / 2) + (nv * 2)
        } else if batch == 1 {
            output_total
        } else {
            qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + (nv * 2)
        };
        let dispatch_rows = batch.div_ceil(row_group_size);
        let dispatch_total = dispatch_rows * dispatch_cols;

        let hidden_u32 = hidden as u32;
        let qkv_dim_u32 = qkv_dim as u32;
        let z_dim_u32 = z_dim as u32;
        let nv_u32 = nv as u32;
        let batch_u32 = batch as u32;
        let row_pair_mode_u32 = if serial_x2_mode {
            7
        } else if serial_vector_mode {
            6
        } else if row_group_size == 1 {
            0
        } else {
            row_group_size as u32
        };
        encoder.set_bytes(9, &hidden_u32);
        encoder.set_bytes(10, &qkv_dim_u32);
        encoder.set_bytes(11, &z_dim_u32);
        encoder.set_bytes(12, &nv_u32);
        encoder.set_bytes(13, &batch_u32);
        encoder.set_bytes(14, &row_pair_mode_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: dispatch_total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((qkv_out, z_out, a_out, b_out))
}

pub(crate) fn metal_rotary_embedding_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_rotary_embedding_supports(q, k, cos, sin, head_dim, rotary_dim),
        "metal rotary qk unsupported shape"
    );
    let (batch, seq_len, q_heads, _) = q.dims4()?;
    let (_, _, k_heads, _) = k.dims4()?;
    let table_batch_stride =
        metal_rotary_table_batch_stride(cos, sin, batch, seq_len, rotary_dim / 2)
            .context("metal rotary qk unsupported position table shape")?;
    let q_shape = q.dims().to_vec();
    let k_shape = k.dims().to_vec();
    let q_metal = kt_metal(&q)?;
    let k_metal = kt_metal(&k)?;
    // SAFETY: the kernel dispatch writes every Q output element exactly once.
    let q_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, q_shape.as_slice())?;
    // SAFETY: the kernel dispatch writes every K output element exactly once.
    let k_out = kt_metal_alloc(k_metal, kiln_tensor::DType::BF16, k_shape.as_slice())?;

    let companion = q_metal.companion()?;
    let pipeline = metal_rotary_qk_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_rotary_qk_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let cos_metal = kt_metal(&cos)?;
        let sin_metal = kt_metal(&sin)?;
        let q_out_metal = kt_metal(&q_out)?;
        let k_out_metal = kt_metal(&k_out)?;

        // #1082 Step 4 embedding-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let cos_buf = buffer_o_kt(cos_metal.buffer().as_ref(), cos.layout(), cos.dtype());
        let sin_buf = buffer_o_kt(sin_metal.buffer().as_ref(), sin.layout(), sin.dtype());
        let q_out_buf = buffer_o_kt(q_out_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let k_out_buf = buffer_o_kt(k_out_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(cos_buf.buffer), cos_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(sin_buf.buffer), sin_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(q_out_buf.buffer), q_out_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(k_out_buf.buffer), k_out_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let q_heads_u32 = q_heads as u32;
        let k_heads_u32 = k_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rotary_dim_u32 = rotary_dim as u32;
        let total_q = batch * seq_len * q_heads * head_dim;
        let total_k = batch * seq_len * k_heads * head_dim;
        let total = total_q + total_k;
        let total_q_u32 = total_q as u32;
        let total_u32 = total as u32;
        let table_batch_stride_u32 = table_batch_stride as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &seq_len_u32);
        encoder.set_bytes(8, &q_heads_u32);
        encoder.set_bytes(9, &k_heads_u32);
        encoder.set_bytes(10, &head_dim_u32);
        encoder.set_bytes(11, &rotary_dim_u32);
        encoder.set_bytes(12, &total_q_u32);
        encoder.set_bytes(13, &total_u32);
        encoder.set_bytes(14, &table_batch_stride_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

fn metal_paged_kv_head_major_read_supports(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    seq_len: usize,
) -> bool {
    if seq_len == 0
        || k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !k_pool.is_contiguous() || !v_pool.is_contiguous() {
        return false;
    }
    let Ok((total_slots, heads, head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return false;
    };
    let Some(total) = seq_len
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    v_dims == (total_slots, heads, head_dim)
        && start_slot <= total_slots
        && seq_len <= total_slots.saturating_sub(start_slot)
        && total <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && start_slot <= u32::MAX as usize
}

fn metal_paged_kv_head_major_read_bf16(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    seq_len: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, seq_len),
        "metal paged kv head-major read unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let out_shape = (1usize, heads, seq_len, head_dim);
    let k_pool_metal = kt_metal(k_pool)?;
    let v_pool_metal = kt_metal(v_pool)?;
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let k_out = kt_metal_alloc(
        k_pool_metal,
        kiln_tensor::DType::BF16,
        &[out_shape.0, out_shape.1, out_shape.2, out_shape.3],
    )?;
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let v_out = kt_metal_alloc(
        v_pool_metal,
        kiln_tensor::DType::BF16,
        &[out_shape.0, out_shape.1, out_shape.2, out_shape.3],
    )?;

    let companion = k_pool_metal.companion()?;
    let pipeline = metal_paged_kv_head_major_read_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_kv_head_major_read_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let ko_metal = kt_metal(&k_out)?;
        let vo_metal = kt_metal(&v_out)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let k_buf = buffer_o_kt(
            k_pool_metal.buffer().as_ref(),
            k_pool.layout(),
            k_pool.dtype(),
        );
        let v_buf = buffer_o_kt(
            v_pool_metal.buffer().as_ref(),
            v_pool.layout(),
            v_pool.dtype(),
        );
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let vo_buf = buffer_o_kt(vo_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(ko_buf.buffer), ko_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(vo_buf.buffer), vo_buf.offset_in_bytes);

        let start_slot_u32 = start_slot as u32;
        let seq_len_u32 = seq_len as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(4, &start_slot_u32);
        encoder.set_bytes(5, &seq_len_u32);
        encoder.set_bytes(6, &heads_u32);
        encoder.set_bytes(7, &head_dim_u32);

        let total = seq_len * heads * head_dim;
        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((k_out, v_out))
}

fn metal_paged_attn_decode_contiguous_supports(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    seq_len: usize,
) -> bool {
    if metal_paged_attn_decode_contiguous_disabled() {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !q.is_contiguous() || !k_pool.is_contiguous() || !v_pool.is_contiguous() {
        return false;
    }
    let Ok((batch, q_heads, q_len, head_dim)) = q.dims4() else {
        return false;
    };
    let Ok((total_slots, kv_heads, k_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return false;
    };
    let Some(end_slot) = start_slot.checked_add(seq_len) else {
        return false;
    };
    batch == 1
        && q_heads == 16
        && kv_heads == 4
        && q_len == 1
        && head_dim == 256
        && k_head_dim == head_dim
        && v_dims == (total_slots, kv_heads, head_dim)
        && seq_len > 0
        && end_slot <= total_slots
        && start_slot <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && kv_heads <= u32::MAX as usize
}

fn metal_paged_attn_decode_contiguous_bf16_d256(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    seq_len: usize,
    softmax_scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_supports(q, k_pool, v_pool, start_slot, seq_len),
        "metal contiguous paged decode attention unsupported shape"
    );
    let (_, q_heads, _, head_dim) = q.dims4()?;
    let q_metal = kt_metal(q)?;
    // SAFETY: the kernel writes one contiguous [1, 1, q_heads * head_dim] output.
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[1usize, 1usize, q_heads * head_dim],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_paged_attn_decode_contiguous_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_attn_decode_contiguous_bf16_d256");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(k_pool)?;
        let v_metal = kt_metal(v_pool)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 paged_kv-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let start_slot_u32 = start_slot as u32;
        let seq_len_u32 = seq_len as u32;
        let q_heads_u32 = q_heads as u32;
        let kv_heads_u32 = 4u32;
        encoder.set_bytes(4, &start_slot_u32);
        encoder.set_bytes(5, &seq_len_u32);
        encoder.set_bytes(6, &q_heads_u32);
        encoder.set_bytes(7, &kv_heads_u32);
        encoder.set_bytes(8, &softmax_scale);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: 1,
            height: q_heads,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(dead_code)]
fn metal_paged_attn_decode_contiguous_batch_supports(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slots: &kiln_tensor::Tensor,
    seq_len: usize,
) -> bool {
    if metal_paged_attn_decode_contiguous_disabled() {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
        || start_slots.dtype() != kiln_tensor::DType::U32
    {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(start_slots.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !q.is_contiguous()
        || !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
        || !start_slots.is_contiguous()
    {
        return false;
    }
    let Ok((batch, q_heads, q_len, head_dim)) = q.dims4() else {
        return false;
    };
    let Ok((total_slots, kv_heads, k_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return false;
    };
    let Ok(slot_count) = start_slots.dims1() else {
        return false;
    };
    batch > 0
        && q_heads == 16
        && kv_heads == 4
        && q_len == 1
        && head_dim == 256
        && k_head_dim == head_dim
        && v_dims == (total_slots, kv_heads, head_dim)
        && slot_count == batch
        && seq_len > 0
        && batch <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && kv_heads <= u32::MAX as usize
        && total_slots <= u32::MAX as usize
}

#[allow(dead_code)]
fn metal_paged_attn_decode_contiguous_batch_bf16_d256(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slots: &kiln_tensor::Tensor,
    seq_len: usize,
    softmax_scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_supports(q, k_pool, v_pool, start_slots, seq_len),
        "metal contiguous paged batch decode attention unsupported shape"
    );
    let (batch, q_heads, _, head_dim) = q.dims4()?;
    let (total_slots, _, _) = k_pool.dims3()?;
    let q_metal = kt_metal(q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, q_heads * head_dim],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_paged_attn_decode_contiguous_batch_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_attn_decode_contiguous_batch_bf16_d256");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(k_pool)?;
        let v_metal = kt_metal(v_pool)?;
        let out_metal = kt_metal(&out)?;
        let slot_metal = kt_metal(start_slots)?;

        // #1082 Step 4 paged_kv-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
        let slot_buf = buffer_o_kt(
            slot_metal.buffer().as_ref(),
            start_slots.layout(),
            start_slots.dtype(),
        );

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(slot_buf.buffer), slot_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let q_heads_u32 = q_heads as u32;
        let kv_heads_u32 = 4u32;
        let total_slots_u32 = total_slots as u32;
        encoder.set_bytes(5, &batch_u32);
        encoder.set_bytes(6, &seq_len_u32);
        encoder.set_bytes(7, &q_heads_u32);
        encoder.set_bytes(8, &kv_heads_u32);
        encoder.set_bytes(9, &softmax_scale);
        encoder.set_bytes(10, &total_slots_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch,
            height: q_heads,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(dead_code)]
fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
) -> bool {
    if metal_paged_attn_decode_contiguous_disabled() {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
        || block_table.dtype() != kiln_tensor::DType::U32
        || seqused_k.dtype() != kiln_tensor::DType::U32
    {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(block_table.device(), kiln_tensor::Device::Metal(_))
        || !matches!(seqused_k.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !q.is_contiguous()
        || !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
        || !block_table.is_contiguous()
        || !seqused_k.is_contiguous()
    {
        return false;
    }
    let Ok((batch, q_len, q_heads, head_dim)) = q.dims4() else {
        return false;
    };
    let Ok((total_slots, kv_heads, k_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_dims) = v_pool.dims3() else {
        return false;
    };
    let Ok((table_batch, max_blocks_per_seq)) = block_table.dims2() else {
        return false;
    };
    let Ok(seq_rows) = seqused_k.dims1() else {
        return false;
    };
    batch > 0
        && q_len == 1
        && q_heads == 16
        && kv_heads == 4
        && head_dim == 256
        && k_head_dim == head_dim
        && v_dims == (total_slots, kv_heads, head_dim)
        && table_batch == batch
        && seq_rows == batch
        && max_blocks_per_seq > 0
        && max_seqlen_k > 0
        && page_block_size > 0
        && max_blocks_per_seq <= u32::MAX as usize
        && max_seqlen_k <= u32::MAX as usize
        && page_block_size <= u32::MAX as usize
        && batch <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && kv_heads <= u32::MAX as usize
        && total_slots <= u32::MAX as usize
}

#[allow(dead_code)]
fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
        ),
        "metal dyn-seqlen paged batch decode attention unsupported shape"
    );
    let (batch, _, q_heads, head_dim) = q.dims4()?;
    let q_metal = kt_metal(q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, q_heads, head_dim],
    )?;

    metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_into(
        q,
        k_pool,
        v_pool,
        block_table,
        seqused_k,
        &out,
        max_seqlen_k,
        page_block_size,
        softmax_scale,
    )?;

    Ok(out)
}

#[allow(dead_code)]
fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_into(
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<()> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
        ),
        "metal dyn-seqlen paged batch decode attention unsupported shape"
    );
    let (batch, _, q_heads, head_dim) = q.dims4()?;
    let (total_slots, _, _) = k_pool.dims3()?;
    let (_, max_blocks_per_seq) = block_table.dims2()?;
    anyhow::ensure!(
        out.dtype() == kiln_tensor::DType::BF16
            && matches!(out.device(), kiln_tensor::Device::Metal(_))
            && out.is_contiguous()
            && out.dims() == [batch, 1usize, q_heads, head_dim],
        "metal dyn-seqlen paged batch decode graph output must be contiguous BF16 [batch,1,q_heads,head_dim] on Metal"
    );
    let q_metal = kt_metal(q)?;
    let companion = q_metal.companion()?;
    let pipeline = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(k_pool)?;
        let v_metal = kt_metal(v_pool)?;
        let out_metal = kt_metal(&out)?;
        let table_metal = kt_metal(block_table)?;
        let seq_metal = kt_metal(seqused_k)?;

        // #1082 Step 4 paged_kv-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
        let table_buf = buffer_o_kt(
            table_metal.buffer().as_ref(),
            block_table.layout(),
            block_table.dtype(),
        );
        let seq_buf = buffer_o_kt(
            seq_metal.buffer().as_ref(),
            seqused_k.layout(),
            seqused_k.dtype(),
        );

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(table_buf.buffer), table_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(seq_buf.buffer), seq_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let max_blocks_u32 = max_blocks_per_seq as u32;
        let max_seqlen_u32 = max_seqlen_k as u32;
        let page_block_size_u32 = page_block_size as u32;
        let q_heads_u32 = q_heads as u32;
        let kv_heads_u32 = 4u32;
        let total_slots_u32 = total_slots as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &max_blocks_u32);
        encoder.set_bytes(8, &max_seqlen_u32);
        encoder.set_bytes(9, &page_block_size_u32);
        encoder.set_bytes(10, &q_heads_u32);
        encoder.set_bytes(11, &kv_heads_u32);
        encoder.set_bytes(12, &softmax_scale);
        encoder.set_bytes(13, &total_slots_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch,
            height: q_heads,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(())
}

fn push_read_resource(resources: &mut Vec<MetalGraphResourceRef>, buf: &BufferOffset<'_>) {
    resources.push(MetalGraphResourceRef::read(buf.buffer));
}

fn push_write_resource(resources: &mut Vec<MetalGraphResourceRef>, buf: &BufferOffset<'_>) {
    resources.push(MetalGraphResourceRef::write(buf.buffer));
}

#[allow(dead_code)]
pub(crate) fn metal_record_paged_kv_write_token_major_bf16_icb(
    command: &IndirectComputeCommand,
    args: &MetalPagedKvWriteTokenMajorIcbArgs,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slot: usize,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> Result<Vec<MetalGraphResourceRef>> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_supports(k_pool, v_pool, slot, k, v),
        "metal paged kv token-major write ICB unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let kp_metal = kt_metal(k_pool)?;
    let companion = kp_metal.companion()?;
    let pipeline = metal_paged_kv_write_token_major_pipeline_indirect(&*companion)?;
    command.set_compute_pipeline_state(&pipeline);

    let ks_metal = kt_metal(k)?;
    let vs_metal = kt_metal(v)?;
    let vp_metal = kt_metal(v_pool)?;

    let ks_buf = buffer_o_kt(ks_metal.buffer().as_ref(), k.layout(), k.dtype());
    let vs_buf = buffer_o_kt(vs_metal.buffer().as_ref(), v.layout(), v.dtype());
    let kp_buf = buffer_o_kt(kp_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
    let vp_buf = buffer_o_kt(vp_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());

    command.set_kernel_buffer(0, ks_buf.buffer, ks_buf.offset_in_bytes);
    command.set_kernel_buffer(1, vs_buf.buffer, vs_buf.offset_in_bytes);
    command.set_kernel_buffer(2, kp_buf.buffer, kp_buf.offset_in_bytes);
    command.set_kernel_buffer(3, vp_buf.buffer, vp_buf.offset_in_bytes);
    command.set_kernel_buffer(4, args.slot.buffer(), 0);
    command.set_kernel_buffer(5, args.heads.buffer(), 0);
    command.set_kernel_buffer(6, args.head_dim.buffer(), 0);

    let total = heads * head_dim;
    let threadgroups_per_grid = objc2_metal::MTLSize {
        width: total.div_ceil(256),
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = objc2_metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    command.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
    command.set_barrier();

    let mut resources = Vec::with_capacity(7);
    push_read_resource(&mut resources, &ks_buf);
    push_read_resource(&mut resources, &vs_buf);
    push_write_resource(&mut resources, &kp_buf);
    push_write_resource(&mut resources, &vp_buf);
    resources.extend(args.scalar_resources());
    Ok(resources)
}

#[allow(dead_code)]
pub(crate) fn metal_record_paged_kv_write_token_major_batch_bf16_icb(
    command: &IndirectComputeCommand,
    args: &MetalPagedKvWriteTokenMajorBatchIcbArgs,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slots: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> Result<Vec<MetalGraphResourceRef>> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_batch_supports(k_pool, v_pool, slots, k, v),
        "metal paged kv token-major batch write ICB unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let (batch, _, _, _) = k.dims4()?;
    let kp_metal = kt_metal(k_pool)?;
    let companion = kp_metal.companion()?;
    let pipeline = metal_paged_kv_write_token_major_batch_pipeline_indirect(&*companion)?;
    command.set_compute_pipeline_state(&pipeline);

    let ks_metal = kt_metal(k)?;
    let vs_metal = kt_metal(v)?;
    let vp_metal = kt_metal(v_pool)?;
    let slot_metal = kt_metal(slots)?;

    let ks_buf = buffer_o_kt(ks_metal.buffer().as_ref(), k.layout(), k.dtype());
    let vs_buf = buffer_o_kt(vs_metal.buffer().as_ref(), v.layout(), v.dtype());
    let kp_buf = buffer_o_kt(kp_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
    let vp_buf = buffer_o_kt(vp_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
    let slot_buf = buffer_o_kt(slot_metal.buffer().as_ref(), slots.layout(), slots.dtype());

    command.set_kernel_buffer(0, ks_buf.buffer, ks_buf.offset_in_bytes);
    command.set_kernel_buffer(1, vs_buf.buffer, vs_buf.offset_in_bytes);
    command.set_kernel_buffer(2, kp_buf.buffer, kp_buf.offset_in_bytes);
    command.set_kernel_buffer(3, vp_buf.buffer, vp_buf.offset_in_bytes);
    command.set_kernel_buffer(4, slot_buf.buffer, slot_buf.offset_in_bytes);
    command.set_kernel_buffer(5, args.batch.buffer(), 0);
    command.set_kernel_buffer(6, args.heads.buffer(), 0);
    command.set_kernel_buffer(7, args.head_dim.buffer(), 0);
    command.set_kernel_buffer(8, args.total_slots.buffer(), 0);

    let total = batch * heads * head_dim;
    let threads_per_grid = objc2_metal::MTLSize {
        width: total,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = objc2_metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    command.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    command.set_barrier();

    let mut resources = Vec::with_capacity(9);
    push_read_resource(&mut resources, &ks_buf);
    push_read_resource(&mut resources, &vs_buf);
    push_write_resource(&mut resources, &kp_buf);
    push_write_resource(&mut resources, &vp_buf);
    push_read_resource(&mut resources, &slot_buf);
    resources.extend(args.scalar_resources());
    Ok(resources)
}

#[allow(dead_code)]
pub(crate) fn metal_record_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_icb(
    command: &IndirectComputeCommand,
    args: &MetalPagedAttnDecodeDynSeqlenIcbArgs,
    q: &kiln_tensor::Tensor,
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    block_table: &kiln_tensor::Tensor,
    seqused_k: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    max_seqlen_k: usize,
    page_block_size: usize,
) -> Result<Vec<MetalGraphResourceRef>> {
    anyhow::ensure!(
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
        ),
        "metal dyn-seqlen paged batch decode attention ICB unsupported shape"
    );
    let (batch, _, q_heads, head_dim) = q.dims4()?;
    anyhow::ensure!(
        out.dtype() == kiln_tensor::DType::BF16
            && matches!(out.device(), kiln_tensor::Device::Metal(_))
            && out.is_contiguous()
            && out.dims() == [batch, 1usize, q_heads, head_dim],
        "metal dyn-seqlen paged batch decode ICB output must be contiguous BF16 [batch,1,q_heads,head_dim] on Metal"
    );

    let q_metal = kt_metal(q)?;
    let companion = q_metal.companion()?;
    let pipeline =
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline_indirect(&*companion)?;
    command.set_compute_pipeline_state(&pipeline);

    let k_metal = kt_metal(k_pool)?;
    let v_metal = kt_metal(v_pool)?;
    let out_metal = kt_metal(out)?;
    let table_metal = kt_metal(block_table)?;
    let seq_metal = kt_metal(seqused_k)?;

    let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
    let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
    let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
    let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
    let table_buf = buffer_o_kt(
        table_metal.buffer().as_ref(),
        block_table.layout(),
        block_table.dtype(),
    );
    let seq_buf = buffer_o_kt(
        seq_metal.buffer().as_ref(),
        seqused_k.layout(),
        seqused_k.dtype(),
    );

    command.set_kernel_buffer(0, q_buf.buffer, q_buf.offset_in_bytes);
    command.set_kernel_buffer(1, k_buf.buffer, k_buf.offset_in_bytes);
    command.set_kernel_buffer(2, v_buf.buffer, v_buf.offset_in_bytes);
    command.set_kernel_buffer(3, out_buf.buffer, out_buf.offset_in_bytes);
    command.set_kernel_buffer(4, table_buf.buffer, table_buf.offset_in_bytes);
    command.set_kernel_buffer(5, seq_buf.buffer, seq_buf.offset_in_bytes);
    command.set_kernel_buffer(6, args.batch.buffer(), 0);
    command.set_kernel_buffer(7, args.max_blocks_per_seq.buffer(), 0);
    command.set_kernel_buffer(8, args.max_seqlen_k.buffer(), 0);
    command.set_kernel_buffer(9, args.page_block_size.buffer(), 0);
    command.set_kernel_buffer(10, args.q_heads.buffer(), 0);
    command.set_kernel_buffer(11, args.kv_heads.buffer(), 0);
    command.set_kernel_buffer(12, args.softmax_scale.buffer(), 0);
    command.set_kernel_buffer(13, args.total_slots.buffer(), 0);

    let threadgroups_per_grid = objc2_metal::MTLSize {
        width: batch,
        height: q_heads,
        depth: 1,
    };
    let threads_per_threadgroup = objc2_metal::MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    command.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
    command.set_barrier();

    let mut resources = Vec::with_capacity(14);
    push_read_resource(&mut resources, &q_buf);
    push_read_resource(&mut resources, &k_buf);
    push_read_resource(&mut resources, &v_buf);
    push_write_resource(&mut resources, &out_buf);
    push_read_resource(&mut resources, &table_buf);
    push_read_resource(&mut resources, &seq_buf);
    resources.extend(args.scalar_resources());
    Ok(resources)
}

fn metal_paged_kv_head_major_read_append_token_major_supports(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    prefix_len: usize,
    k_tail: &kiln_tensor::Tensor,
    v_tail: &kiln_tensor::Tensor,
) -> bool {
    if prefix_len == 0 {
        return false;
    }
    if !metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, prefix_len) {
        return false;
    }
    if k_tail.dtype() != kiln_tensor::DType::BF16 || v_tail.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(k_tail.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_tail.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !k_tail.is_contiguous() || !v_tail.is_contiguous() {
        return false;
    }
    let Ok((batch, tail_len, heads, head_dim)) = k_tail.dims4() else {
        return false;
    };
    let Ok(v_dims) = v_tail.dims4() else {
        return false;
    };
    let Ok((_, pool_heads, pool_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Some(total_len) = prefix_len.checked_add(tail_len) else {
        return false;
    };
    let Some(total) = total_len
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    batch == 1
        && v_dims == (batch, tail_len, heads, head_dim)
        && heads == pool_heads
        && head_dim == pool_head_dim
        && total_len <= u32::MAX as usize
        && tail_len <= u32::MAX as usize
        && heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && total <= u32::MAX as usize
}

fn metal_paged_kv_head_major_read_append_token_major_bf16(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    start_slot: usize,
    prefix_len: usize,
    k_tail: &kiln_tensor::Tensor,
    v_tail: &kiln_tensor::Tensor,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_paged_kv_head_major_read_append_token_major_supports(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        ),
        "metal paged kv head-major read+append unsupported shape"
    );
    let (_, tail_len, heads, head_dim) = k_tail.dims4()?;
    let total_len = prefix_len + tail_len;
    let out_shape = (1usize, heads, total_len, head_dim);
    let k_pool_metal = kt_metal(k_pool)?;
    let v_pool_metal = kt_metal(v_pool)?;
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let k_out = kt_metal_alloc(
        k_pool_metal,
        kiln_tensor::DType::BF16,
        &[out_shape.0, out_shape.1, out_shape.2, out_shape.3],
    )?;
    // SAFETY: the kernel dispatch covers exactly every element in `out_shape`.
    let v_out = kt_metal_alloc(
        v_pool_metal,
        kiln_tensor::DType::BF16,
        &[out_shape.0, out_shape.1, out_shape.2, out_shape.3],
    )?;

    let companion = k_pool_metal.companion()?;
    let pipeline = metal_paged_kv_head_major_read_append_token_major_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_kv_head_major_read_append_token_major_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let kt_metal_buf = kt_metal(k_tail)?;
        let vt_metal = kt_metal(v_tail)?;
        let ko_metal = kt_metal(&k_out)?;
        let vo_metal = kt_metal(&v_out)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let k_buf = buffer_o_kt(
            k_pool_metal.buffer().as_ref(),
            k_pool.layout(),
            k_pool.dtype(),
        );
        let v_buf = buffer_o_kt(
            v_pool_metal.buffer().as_ref(),
            v_pool.layout(),
            v_pool.dtype(),
        );
        let kt_buf = buffer_o_kt(
            kt_metal_buf.buffer().as_ref(),
            k_tail.layout(),
            k_tail.dtype(),
        );
        let vt_buf = buffer_o_kt(vt_metal.buffer().as_ref(), v_tail.layout(), v_tail.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let vo_buf = buffer_o_kt(vo_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(vt_buf.buffer), vt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ko_buf.buffer), ko_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(vo_buf.buffer), vo_buf.offset_in_bytes);

        let start_slot_u32 = start_slot as u32;
        let prefix_len_u32 = prefix_len as u32;
        let tail_len_u32 = tail_len as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(6, &start_slot_u32);
        encoder.set_bytes(7, &prefix_len_u32);
        encoder.set_bytes(8, &tail_len_u32);
        encoder.set_bytes(9, &heads_u32);
        encoder.set_bytes(10, &head_dim_u32);

        let total = total_len * heads * head_dim;
        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((k_out, v_out))
}

pub(crate) fn metal_paged_kv_write_token_major_supports(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slot: usize,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> bool {
    if metal_paged_kv_write_token_major_disabled() {
        return false;
    }
    if k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return false;
    }
    let Ok((total_slots, pool_heads, pool_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_pool_dims) = v_pool.dims3() else {
        return false;
    };
    let Ok((batch, seq_len, heads, head_dim)) = k.dims4() else {
        return false;
    };
    let Ok(v_dims) = v.dims4() else {
        return false;
    };
    let Some(total) = heads.checked_mul(head_dim) else {
        return false;
    };

    batch == 1
        && seq_len == 1
        && v_pool_dims == (total_slots, pool_heads, pool_head_dim)
        && v_dims == (batch, seq_len, heads, head_dim)
        && heads == pool_heads
        && head_dim == pool_head_dim
        && slot < total_slots
        && slot <= u32::MAX as usize
        && heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(crate) fn metal_paged_kv_write_token_major_bf16(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slot: usize,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> Result<()> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_supports(k_pool, v_pool, slot, k, v),
        "metal paged kv token-major write unsupported shape"
    );
    let (_, heads, head_dim) = k_pool.dims3()?;
    let kp_metal = kt_metal(k_pool)?;
    let companion = kp_metal.companion()?;
    let pipeline = metal_paged_kv_write_token_major_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_kv_write_token_major_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let ks_metal = kt_metal(k)?;
        let vs_metal = kt_metal(v)?;
        let vp_metal = kt_metal(v_pool)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let ks_buf = buffer_o_kt(ks_metal.buffer().as_ref(), k.layout(), k.dtype());
        let vs_buf = buffer_o_kt(vs_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kp_buf = buffer_o_kt(kp_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let vp_buf = buffer_o_kt(vp_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());

        encoder.set_buffer(0, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(vs_buf.buffer), vs_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kp_buf.buffer), kp_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(vp_buf.buffer), vp_buf.offset_in_bytes);

        let slot_u32 = slot as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        encoder.set_bytes(4, &slot_u32);
        encoder.set_bytes(5, &heads_u32);
        encoder.set_bytes(6, &head_dim_u32);

        let total = heads * head_dim;
        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(())
}

#[allow(dead_code)]
#[allow(dead_code)]
pub(crate) fn metal_paged_kv_write_token_major_batch_supports(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slots: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> bool {
    if metal_paged_kv_write_token_major_disabled() {
        return false;
    }
    if k_pool.dtype() != kiln_tensor::DType::BF16
        || v_pool.dtype() != kiln_tensor::DType::BF16
        || slots.dtype() != kiln_tensor::DType::U32
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(k_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_pool.device(), kiln_tensor::Device::Metal(_))
        || !matches!(slots.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !k_pool.is_contiguous()
        || !v_pool.is_contiguous()
        || !slots.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
    {
        return false;
    }
    let Ok((total_slots, pool_heads, pool_head_dim)) = k_pool.dims3() else {
        return false;
    };
    let Ok(v_pool_dims) = v_pool.dims3() else {
        return false;
    };
    let Ok((batch, seq_len, heads, head_dim)) = k.dims4() else {
        return false;
    };
    let Ok(v_dims) = v.dims4() else {
        return false;
    };
    let Ok(slot_count) = slots.dims1() else {
        return false;
    };
    let Some(row_stride) = heads.checked_mul(head_dim) else {
        return false;
    };
    let Some(total) = batch.checked_mul(row_stride) else {
        return false;
    };

    batch > 0
        && total_slots > 0
        && seq_len == 1
        && slot_count == batch
        && v_pool_dims == (total_slots, pool_heads, pool_head_dim)
        && v_dims == (batch, seq_len, heads, head_dim)
        && heads == pool_heads
        && head_dim == pool_head_dim
        && batch <= u32::MAX as usize
        && heads <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && total_slots <= u32::MAX as usize
        && row_stride <= u32::MAX as usize
        && total <= u32::MAX as usize
}

#[allow(dead_code)]
#[allow(dead_code)]
pub(crate) fn metal_paged_kv_write_token_major_batch_bf16(
    k_pool: &kiln_tensor::Tensor,
    v_pool: &kiln_tensor::Tensor,
    slots: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
) -> Result<()> {
    anyhow::ensure!(
        metal_paged_kv_write_token_major_batch_supports(k_pool, v_pool, slots, k, v),
        "metal paged kv token-major batch write unsupported shape"
    );
    let (total_slots, heads, head_dim) = k_pool.dims3()?;
    let (batch, _, _, _) = k.dims4()?;
    let kp_metal = kt_metal(k_pool)?;
    let companion = kp_metal.companion()?;
    let pipeline = metal_paged_kv_write_token_major_batch_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_paged_kv_write_token_major_batch_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let ks_metal = kt_metal(k)?;
        let vs_metal = kt_metal(v)?;
        let vp_metal = kt_metal(v_pool)?;
        let slot_metal = kt_metal(slots)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let ks_buf = buffer_o_kt(ks_metal.buffer().as_ref(), k.layout(), k.dtype());
        let vs_buf = buffer_o_kt(vs_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kp_buf = buffer_o_kt(kp_metal.buffer().as_ref(), k_pool.layout(), k_pool.dtype());
        let vp_buf = buffer_o_kt(vp_metal.buffer().as_ref(), v_pool.layout(), v_pool.dtype());
        let slot_buf = buffer_o_kt(slot_metal.buffer().as_ref(), slots.layout(), slots.dtype());

        encoder.set_buffer(0, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(vs_buf.buffer), vs_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kp_buf.buffer), kp_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(vp_buf.buffer), vp_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(slot_buf.buffer), slot_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let heads_u32 = heads as u32;
        let head_dim_u32 = head_dim as u32;
        let total_slots_u32 = total_slots as u32;
        encoder.set_bytes(5, &batch_u32);
        encoder.set_bytes(6, &heads_u32);
        encoder.set_bytes(7, &head_dim_u32);
        encoder.set_bytes(8, &total_slots_u32);

        let total = batch * heads * head_dim;
        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(())
}

pub(crate) fn metal_rms_norm_bf16(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    let x_dims = x.dims().to_vec();
    let hidden = *x_dims
        .last()
        .context("metal rmsnorm requires rank >= 1 input")?;
    anyhow::ensure!(hidden <= 8192, "metal rmsnorm hidden dim > 8192");
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal rmsnorm shape too large"
    );

    let x = x.contiguous()?;
    let weight = weight.contiguous()?;

    let x_metal = kt_metal(&x)?;
    // The kernel writes every hidden element for every row.
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &x_dims)?;

    if rows == 0 {
        return Ok(out);
    }

    let companion = x_metal.companion()?;
    let pipeline = metal_rms_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(3, &rows_u32);
        encoder.set_bytes(4, &hidden_u32);
        encoder.set_bytes(5, &eps);
        encoder.set_bytes(6, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_gdn_qk_norm_f32_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    q_scale: f32,
    eps: f32,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let dims = q.dims().to_vec();
    let hidden = *dims
        .last()
        .context("metal gdn qk norm requires rank >= 1 input")?;
    anyhow::ensure!(q.dims() == k.dims(), "metal gdn qk norm shape mismatch");
    anyhow::ensure!(hidden <= 8192, "metal gdn qk norm hidden dim > 8192");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal gdn qk norm shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    // The kernel writes every Q/K element for every row.
    let q_metal = kt_metal(&q)?;
    let q_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, dims.as_slice())?;
    let k_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, dims.as_slice())?;

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_qk_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_qk_norm_f32_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(ko_buf.buffer), ko_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &q_scale);
        encoder.set_bytes(7, &eps);
        encoder.set_bytes(8, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

pub(crate) fn metal_gdn_qk_norm_gqa_f32_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    nv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_qk_norm_gqa_supports(q, k, nv),
        "metal gdn qk norm gqa unsupported shape"
    );
    let (batch, seq_len, nk, hidden) = q.dims4()?;
    let gqa_ratio = nv / nk;
    let rows = batch * seq_len * nk;
    anyhow::ensure!(
        rows <= u32::MAX as usize
            && nk <= u32::MAX as usize
            && hidden <= u32::MAX as usize
            && gqa_ratio <= u32::MAX as usize,
        "metal gdn qk norm gqa shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    // Each source head writes all replicated value-head outputs.
    let q_metal = kt_metal(&q)?;
    let q_out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, hidden],
    )?;
    let k_out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, hidden],
    )?;

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_qk_norm_gqa_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_qk_norm_gqa_f32_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(ko_buf.buffer), ko_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let nk_u32 = nk as u32;
        let nv_u32 = nv as u32;
        let hidden_u32 = hidden as u32;
        let gqa_ratio_u32 = gqa_ratio as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &nk_u32);
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &hidden_u32);
        encoder.set_bytes(8, &gqa_ratio_u32);
        encoder.set_bytes(9, &q_scale);
        encoder.set_bytes(10, &eps);
        encoder.set_bytes(11, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_qkv_conv_norm_bf16(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_decode_qkv_conv_norm_supports(
            mixed_qkv,
            weight,
            conv_state,
            kernel_size,
            nk,
            dk,
            nv,
            dv
        ),
        "metal gdn decode qkv conv/norm unsupported shape"
    );
    let (batch, _, channels) = mixed_qkv.dims3()?;
    let rows_per_batch = nk + nk + nv;
    let rows = batch * rows_per_batch;
    anyhow::ensure!(
        rows <= u32::MAX as usize,
        "metal gdn decode qkv conv/norm shape too large"
    );

    let mixed_qkv = mixed_qkv.contiguous()?;
    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal gdn decode qkv conv/norm weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }

    // The kernel writes every unexpanded Q/K and V element, and updates each
    // convolution state channel exactly once.
    let mixed_qkv_metal = kt_metal(&mixed_qkv)?;
    let q_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nk, dk],
    )?;
    let k_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nk, dk],
    )?;
    let v_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nv, dv],
    )?;

    let companion = mixed_qkv_metal.companion()?;
    let pipeline = metal_gdn_decode_qkv_conv_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_qkv_conv_norm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let x_metal = kt_metal(&mixed_qkv)?;
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(&conv_state)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;
        let vo_metal = kt_metal(&v_out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(
            x_metal.buffer().as_ref(),
            mixed_qkv.layout(),
            mixed_qkv.dtype(),
        );
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let vo_buf = buffer_o_kt(vo_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ko_buf.buffer), ko_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(vo_buf.buffer), vo_buf.offset_in_bytes);

        let nk_u32 = nk as u32;
        let nv_u32 = nv as u32;
        encoder.set_bytes(6, &nk_u32);
        encoder.set_bytes(7, &nv_u32);
        encoder.set_bytes(8, &q_scale);
        encoder.set_bytes(9, &eps);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: rows_per_batch,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out, v_out))
}

const METAL_GDN_GATES_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float kiln_stable_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + exp(-x));
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

inline float kiln_stable_softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return exp(x);
    }
    return log(1.0f + exp(x));
}

kernel void kiln_gdn_gates_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const bfloat* dt_bias [[buffer(3)]],
    device bfloat* beta_out [[buffer(4)]],
    device bfloat* g_out [[buffer(5)]],
    constant uint& nv [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const uint h = gid % nv;
    const float a_val = static_cast<float>(a[gid]);
    const float b_val = static_cast<float>(b[gid]);
    const float a_log_val = static_cast<float>(a_log[h]);
    const float dt_bias_val = static_cast<float>(dt_bias[h]);

    const float beta = kiln_stable_sigmoid(b_val);
    const float sp = kiln_stable_softplus(a_val + dt_bias_val);
    const float g = sp * -exp(a_log_val);

    beta_out[gid] = static_cast<bfloat>(beta);
    g_out[gid] = static_cast<bfloat>(g);
}

kernel void kiln_gdn_gates_decay_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const bfloat* dt_bias [[buffer(3)]],
    device bfloat* beta_out [[buffer(4)]],
    device bfloat* decay_out [[buffer(5)]],
    constant uint& nv [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const uint h = gid % nv;
    const float a_val = static_cast<float>(a[gid]);
    const float b_val = static_cast<float>(b[gid]);
    const float a_log_val = static_cast<float>(a_log[h]);
    const float dt_bias_val = static_cast<float>(dt_bias[h]);

    const float beta = kiln_stable_sigmoid(b_val);
    const float sp = kiln_stable_softplus(a_val + dt_bias_val);
    const float g = sp * -exp(a_log_val);
    const bfloat g_bf = static_cast<bfloat>(g);

    beta_out[gid] = static_cast<bfloat>(beta);
    decay_out[gid] = static_cast<bfloat>(exp(static_cast<float>(g_bf)));
}

kernel void kiln_gdn_gates_decay_ab_bf16(
    device const bfloat* ab [[buffer(0)]],
    device const float* a_log [[buffer(1)]],
    device const bfloat* dt_bias [[buffer(2)]],
    device bfloat* beta_out [[buffer(3)]],
    device bfloat* decay_out [[buffer(4)]],
    constant uint& nv [[buffer(5)]],
    constant uint& total [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const uint h = gid % nv;
    const uint row = gid / nv;
    const uint ab_base = row * (nv * 2);
    const float a_val = static_cast<float>(ab[ab_base + h]);
    const float b_val = static_cast<float>(ab[ab_base + nv + h]);
    const float a_log_val = static_cast<float>(a_log[h]);
    const float dt_bias_val = static_cast<float>(dt_bias[h]);

    const float beta = kiln_stable_sigmoid(b_val);
    const float sp = kiln_stable_softplus(a_val + dt_bias_val);
    const float g = sp * -exp(a_log_val);
    const bfloat g_bf = static_cast<bfloat>(g);

    beta_out[gid] = static_cast<bfloat>(beta);
    decay_out[gid] = static_cast<bfloat>(exp(static_cast<float>(g_bf)));
}
"#;

const METAL_GDN_DECODE_GATES_RECURRENT_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_gates_recurrent_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* b [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const bfloat* dt_bias [[buffer(6)]],
    device bfloat* state [[buffer(7)]],
    device bfloat* out [[buffer(8)]],
    constant uint& batch_heads [[buffer(9)]],
    constant uint& dk [[buffer(10)]],
    constant uint& dv [[buffer(11)]],
    constant uint& value_heads [[buffer(12)]],
    constant uint& q_heads [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint LANES = 32;
    if (gid >= batch_heads * dv || tid >= LANES || dk != 128 || dv != 128) {
        return;
    }

    const uint bh = gid / dv;
    const uint d = gid - bh * dv;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * dk;
    const uint v_base = (batch_idx * value_heads + head_idx) * dv;
    const uint gate_idx = batch_idx * value_heads + head_idx;
    const uint state_base = bh * dk * dv;

    const float beta = static_cast<float>(static_cast<bfloat>(
        kiln_stable_sigmoid(static_cast<float>(b[gate_idx]))
    ));
    const float g = static_cast<float>(static_cast<bfloat>(
        kiln_stable_softplus(static_cast<float>(a[gate_idx]) + static_cast<float>(dt_bias[head_idx])) *
        -exp(static_cast<float>(a_log[head_idx]))
    ));
    const float decay = static_cast<float>(static_cast<bfloat>(exp(g)));

    float ls[NSG];
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        ls[j] = static_cast<float>(state[state_base + is * dv + d]);
    }

    float s_k = 0.0f;
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        const float decayed = static_cast<float>(static_cast<bfloat>(ls[j] * decay));
        ls[j] = decayed;
        s_k += decayed * static_cast<float>(k[qk_base + is]);
    }
    s_k = simd_sum(s_k);

    const float delta = static_cast<float>(static_cast<bfloat>(
        (static_cast<float>(v[v_base + d]) - s_k) * beta
    ));

    float y = 0.0f;
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        const float new_s = static_cast<float>(static_cast<bfloat>(
            ls[j] + static_cast<float>(k[qk_base + is]) * delta
        ));
        ls[j] = new_s;
        y += new_s * static_cast<float>(q[qk_base + is]);
        state[state_base + is * dv + d] = static_cast<bfloat>(new_s);
    }
    y = simd_sum(y);

    if (tid == 0) {
        out[v_base + d] = static_cast<bfloat>(y);
    }
}
"#;

const METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_gates_recurrent_rmsnorm_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* b [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const bfloat* dt_bias [[buffer(6)]],
    device bfloat* state [[buffer(7)]],
    device const bfloat* z [[buffer(8)]],
    device const float* weight [[buffer(9)]],
    device bfloat* out [[buffer(10)]],
    constant uint& batch_heads [[buffer(11)]],
    constant uint& dk [[buffer(12)]],
    constant uint& dv [[buffer(13)]],
    constant uint& value_heads [[buffer(14)]],
    constant uint& q_heads [[buffer(15)]],
    constant float& eps [[buffer(16)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[128];
    if (bh >= batch_heads || tid >= dv || dk != 128 || dv != 128) {
        return;
    }

    const uint d = tid;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * dk;
    const uint v_base = (batch_idx * value_heads + head_idx) * dv;
    const uint gate_idx = batch_idx * value_heads + head_idx;
    const uint state_base = bh * dk * dv;

    const float beta = static_cast<float>(static_cast<bfloat>(
        kiln_stable_sigmoid(static_cast<float>(b[gate_idx]))
    ));
    const float g = static_cast<float>(static_cast<bfloat>(
        kiln_stable_softplus(static_cast<float>(a[gate_idx]) + static_cast<float>(dt_bias[head_idx])) *
        -exp(static_cast<float>(a_log[head_idx]))
    ));
    const float decay = static_cast<float>(static_cast<bfloat>(exp(g)));

    float s_k = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const uint state_idx = state_base + i * dv + d;
        const float decayed = static_cast<float>(static_cast<bfloat>(
            static_cast<float>(state[state_idx]) * decay
        ));
        state[state_idx] = static_cast<bfloat>(decayed);
        s_k += decayed * static_cast<float>(k[qk_base + i]);
    }
    const float delta = static_cast<float>(static_cast<bfloat>(
        (static_cast<float>(v[v_base + d]) - s_k) * beta
    ));

    float y = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const uint state_idx = state_base + i * dv + d;
        const float new_s = static_cast<float>(static_cast<bfloat>(
            static_cast<float>(state[state_idx]) + static_cast<float>(k[qk_base + i]) * delta
        ));
        state[state_idx] = static_cast<bfloat>(new_s);
        y += new_s * static_cast<float>(q[qk_base + i]);
    }

    scratch[tid] = y * y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt((scratch[0] / static_cast<float>(dv)) + eps);
    const float zv = static_cast<float>(z[v_base + d]);
    const float gate = zv / (1.0f + exp(-zv));
    out[v_base + d] = static_cast<bfloat>(
        y * rms_inv * static_cast<float>(weight[d]) * gate
    );
}
"#;

fn metal_gdn_gates_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn_gates pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_gates_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn_gates function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn_gates pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_gates_decay_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn_gates decay pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_gates_decay_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn_gates decay function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn_gates decay pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_gates_decay_ab_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn_gates decay A/B pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_gates_decay_ab_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn_gates decay A/B function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn_gates decay A/B pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_decode_gates_recurrent_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn decode gates+recurrent pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_gates_recurrent_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn decode gates+recurrent function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn decode gates+recurrent pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal gdn decode gates+recurrent+rmsnorm pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_gates_recurrent_rmsnorm_bf16", None)
        .map_err(|e| {
            anyhow::anyhow!("load metal gdn decode gates+recurrent+rmsnorm function: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal gdn decode gates+recurrent+rmsnorm pipeline: {e:?}")
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_gates_bf16(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let shape = a.dims().to_vec();
    let nv = *shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates requires at least rank-1 input"))?;
    let total = a.elem_count();
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates input too large"
    );
    anyhow::ensure!(nv <= u32::MAX as usize, "metal gdn_gates nv too large");

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_metal = kt_metal(&a)?;
    // The gates kernel writes every beta/g element.
    let beta = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, &shape)?;
    let g = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, &shape)?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_gates_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(g_buf.buffer), g_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, g))
}

pub(crate) fn metal_gdn_prefill_ab_in_proj_bf16(
    x: &kiln_tensor::Tensor,
    in_proj_ab_t: &kiln_tensor::Tensor,
    nv: usize,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_prefill_ab_in_proj_supports(x, in_proj_ab_t, nv),
        "metal gdn prefill A/B in-proj unsupported shape"
    );
    let ab = x
        .broadcast_matmul(in_proj_ab_t)
        .context("metal gdn prefill A/B in-proj matmul")?;
    let a = ab.narrow(2, 0, nv)?;
    let b = ab.narrow(2, nv, nv)?;
    Ok((ab, a, b))
}

pub(crate) fn metal_gdn_gates_decay_bf16(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_gates_decay_supports(a, b, a_log, dt_bias),
        "metal gdn_gates decay unsupported shape"
    );
    let shape = a.dims().to_vec();
    let nv = *shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates decay requires at least rank-1 input"))?;
    let total = a.elem_count();
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates decay input too large"
    );
    anyhow::ensure!(
        nv <= u32::MAX as usize,
        "metal gdn_gates decay nv too large"
    );

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_metal = kt_metal(&a)?;
    let beta = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, shape.as_slice())?;
    let decay = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, shape.as_slice())?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_gates_decay_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_decay_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(decay_buf.buffer), decay_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, decay))
}

pub(crate) fn metal_gdn_gates_decay_ab_bf16(
    ab: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    nv: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_gates_decay_ab_supports(ab, a_log, dt_bias, nv),
        "metal gdn_gates decay A/B unsupported shape"
    );
    let (batch, seq_len, _channels) = ab.dims3()?;
    let total = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(nv))
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates decay A/B input too large"))?;
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates decay A/B input too large"
    );
    anyhow::ensure!(
        nv <= u32::MAX as usize,
        "metal gdn_gates decay A/B nv too large"
    );

    let ab = ab.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let shape = vec![batch, seq_len, nv];
    let ab_metal = kt_metal(&ab)?;
    let beta = kt_metal_alloc(ab_metal, kiln_tensor::DType::BF16, shape.as_slice())?;
    let decay = kt_metal_alloc(ab_metal, kiln_tensor::DType::BF16, shape.as_slice())?;

    let companion = ab_metal.companion()?;
    let pipeline = metal_gdn_gates_decay_ab_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_decay_ab_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let ab_buf = buffer_o_kt(ab_metal.buffer().as_ref(), ab.layout(), ab.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());

        encoder.set_buffer(0, Some(ab_buf.buffer), ab_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(decay_buf.buffer), decay_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(5, &nv_u32);
        encoder.set_bytes(6, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, decay))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state),
        "metal gdn decode gates+recurrent unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn decode gates+recurrent shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_decode_gates_recurrent_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_gates_recurrent_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let a_metal = kt_metal(&a)?;
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        encoder.set_bytes(9, &batch_heads_u32);
        encoder.set_bytes(10, &dk_u32);
        encoder.set_bytes(11, &dv_u32);
        encoder.set_bytes(12, &value_heads_u32);
        encoder.set_bytes(13, &q_heads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_decode_gates_recurrent_rmsnorm_supports(
            q, k, v, a, b, a_log, dt_bias, state, z, weight
        ),
        "metal gdn decode gates+recurrent+rmsnorm unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn decode gates+recurrent+rmsnorm shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let z = z.contiguous()?;
    let weight = weight.contiguous()?;
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_gates_recurrent_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let a_metal = kt_metal(&a)?;
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let state_metal = kt_metal(&state)?;
        let z_metal = kt_metal(&z)?;
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z.layout(), z.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(10, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        encoder.set_bytes(11, &batch_heads_u32);
        encoder.set_bytes(12, &dk_u32);
        encoder.set_bytes(13, &dv_u32);
        encoder.set_bytes(14, &value_heads_u32);
        encoder.set_bytes(15, &q_heads_u32);
        encoder.set_bytes(16, &eps);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: dv,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

const METAL_GATED_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gated_rmsnorm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* z [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& threadgroup_width [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float sum_sq = 0.0f;
    if (tid < hidden) {
        const float xv = static_cast<float>(x[base + tid]);
        sum_sq = xv * xv;
    }
    scratch[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < hidden) {
        const float rms_inv = rsqrt((scratch[0] / static_cast<float>(hidden)) + eps);
        const float xv = static_cast<float>(x[base + tid]);
        const float zv = static_cast<float>(z[base + tid]);
        const float gate = zv / (1.0f + exp(-zv));
        out[base + tid] = static_cast<bfloat>(xv * rms_inv * static_cast<float>(weight[tid]) * gate);
    }
}
"#;

fn metal_gated_rms_norm_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gated rmsnorm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gated_rmsnorm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gated rmsnorm function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gated rmsnorm pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gated_rms_norm_bf16(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    let (batch, seq_len, heads, hidden) = x.dims4()?;
    let rows = batch
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(heads))
        .context("metal gated rmsnorm row count overflow")?;
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal gated rmsnorm shape too large"
    );
    anyhow::ensure!(hidden <= 1024, "metal gated rmsnorm hidden dim > 1024");

    let x = x.contiguous()?;
    let z = z.contiguous()?;
    // The kernel binds `weight` as a `device const float*`. The norm weight is
    // BF16 for Qwen3.5 (matches the CUDA path); promote to F32 here so the
    // kernel's buffer type is satisfied. F32 weights pass through unchanged.
    let weight = weight.contiguous()?.to_dtype(kiln_tensor::DType::F32)?;
    let x_metal = kt_metal(&x)?;
    // The kernel writes every hidden element for every row.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, heads, hidden],
    )?;

    if rows == 0 {
        return Ok(out);
    }

    let companion = x_metal.companion()?;
    let pipeline = metal_gated_rms_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gated_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let z_metal = kt_metal(&z)?;
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z.layout(), z.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &eps);
        encoder.set_bytes(7, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

const METAL_GDN_RECURRENT_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_recurrent_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* beta [[buffer(3)]],
    device const bfloat* g [[buffer(4)]],
    device bfloat* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& batch_heads [[buffer(7)]],
    constant uint& dk [[buffer(8)]],
    constant uint& dv [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = batch_heads * dv;
    if (gid >= total) {
        return;
    }

    const uint bh = gid / dv;
    const uint col = gid - bh * dv;
    const uint qk_base = bh * dk;
    const uint v_base = bh * dv;
    const uint state_base = bh * dk * dv;

    const float decay = exp(static_cast<float>(g[bh]));
    const float beta_t = static_cast<float>(beta[bh]);

    float v_pred = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const float k_i = static_cast<float>(k[qk_base + i]);
        const float s_i = static_cast<float>(state[state_base + i * dv + col]);
        v_pred += k_i * (decay * s_i);
    }

    const float v_t = static_cast<float>(v[v_base + col]);
    const float delta = beta_t * (v_t - v_pred);

    float out_acc = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const float q_i = static_cast<float>(q[qk_base + i]);
        const float k_i = static_cast<float>(k[qk_base + i]);
        const uint state_idx = state_base + i * dv + col;
        const float old_s = static_cast<float>(state[state_idx]);
        const float new_s = decay * old_s + k_i * delta;
        state[state_idx] = static_cast<bfloat>(new_s);
        out_acc += q_i * new_s;
    }

    out[v_base + col] = static_cast<bfloat>(out_acc);
}

kernel void kiln_gdn_forward_substitution_bf16(
    device const bfloat* a_strict [[buffer(0)]],
    device const bfloat* v_prime [[buffer(1)]],
    device const bfloat* beta [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& chunk_size [[buffer(5)]],
    constant uint& dv [[buffer(6)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (bh >= batch_heads) {
        return;
    }

    // Conservative Qwen3.5 envelope: C <= 64, dv <= 128. Static threadgroup
    // storage keeps the kernel simple and under Apple Silicon's common 32 KiB
    // per-threadgroup memory budget: (64*64 + 64*128) bf16 = 24 KiB.
    threadgroup bfloat sA[4096];
    threadgroup bfloat sW[8192];

    const uint a_base = bh * chunk_size * chunk_size;
    const uint v_base = bh * chunk_size * dv;
    const uint beta_base = bh * chunk_size;
    const uint total_a = chunk_size * chunk_size;

    for (uint i = tid; i < total_a; i += 128) {
        sA[i] = a_strict[a_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < chunk_size; ++t) {
        const float beta_t = static_cast<float>(beta[beta_base + t]);

        for (uint d = tid; d < dv; d += 128) {
            float acc = 0.0f;
            for (uint i = 0; i < t; ++i) {
                const float a = static_cast<float>(sA[t * chunk_size + i]);
                const float w = static_cast<float>(sW[i * dv + d]);
                acc += a * w;
            }

            const uint row_col = t * dv + d;
            const float vp = static_cast<float>(v_prime[v_base + row_col]);
            const bfloat w = static_cast<bfloat>(beta_t * (vp - acc));
            sW[row_col] = w;
            out[v_base + row_col] = w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kiln_gdn_forward_substitution_f32(
    device const float* a_strict [[buffer(0)]],
    device const float* v_prime [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& chunk_size [[buffer(5)]],
    constant uint& dv [[buffer(6)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (bh >= batch_heads) {
        return;
    }

    threadgroup float sW[8192];

    const uint a_base = bh * chunk_size * chunk_size;
    const uint v_base = bh * chunk_size * dv;
    const uint beta_base = bh * chunk_size;

    for (uint t = 0; t < chunk_size; ++t) {
        const float beta_t = beta[beta_base + t];

        for (uint d = tid; d < dv; d += 128) {
            float acc = 0.0f;
            for (uint i = 0; i < t; ++i) {
                acc += a_strict[a_base + t * chunk_size + i] * sW[i * dv + d];
            }

            const uint row_col = t * dv + d;
            const float w = beta_t * (v_prime[v_base + row_col] - acc);
            sW[row_col] = w;
            out[v_base + row_col] = w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kiln_gdn_chunk_prep_bf16(
    device const bfloat* g [[buffer(0)]],
    device const bfloat* v [[buffer(1)]],
    device const bfloat* kkt [[buffer(2)]],
    device const bfloat* qkt [[buffer(3)]],
    device const bfloat* ks_entry [[buffer(4)]],
    device const bfloat* q_s [[buffer(5)]],
    device bfloat* a_strict [[buffer(6)]],
    device bfloat* b_mask [[buffer(7)]],
    device bfloat* v_prime [[buffer(8)]],
    device bfloat* q_s_scaled [[buffer(9)]],
    device bfloat* decay_last_col [[buffer(10)]],
    device bfloat* p_last [[buffer(11)]],
    constant uint& batch_heads [[buffer(12)]],
    constant uint& chunk_size [[buffer(13)]],
    constant uint& dv [[buffer(14)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (bh >= batch_heads) {
        return;
    }

    threadgroup float sBigG[64];
    threadgroup bfloat sP[64];

    const uint g_base = bh * chunk_size;
    const uint cdv_base = bh * chunk_size * dv;
    const uint cc_base = bh * chunk_size * chunk_size;

    if (tid < chunk_size) {
        sBigG[tid] = static_cast<float>(g[g_base + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float acc = 0.0f;
        for (uint i = 0; i < 64; ++i) {
            acc += sBigG[i];
            sBigG[i] = acc;
        }
        for (uint i = 0; i < 64; ++i) {
            sP[i] = static_cast<bfloat>(exp(sBigG[i]));
        }
        p_last[bh] = sP[chunk_size - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint cc = chunk_size * chunk_size;
    for (uint idx = tid; idx < cc; idx += 128) {
        const uint t = idx / chunk_size;
        const uint i = idx - t * chunk_size;
        const bfloat decay_bf = static_cast<bfloat>(exp(sBigG[t] - sBigG[i]));
        const float decay = static_cast<float>(decay_bf);
        const float kkt_val = static_cast<float>(kkt[cc_base + idx]);
        const float qkt_val = static_cast<float>(qkt[cc_base + idx]);
        a_strict[cc_base + idx] =
            (i < t) ? static_cast<bfloat>(kkt_val * decay) : static_cast<bfloat>(0.0f);
        b_mask[cc_base + idx] =
            (i <= t) ? static_cast<bfloat>(qkt_val * decay) : static_cast<bfloat>(0.0f);
    }

    const uint cdv = chunk_size * dv;
    for (uint idx = tid; idx < cdv; idx += 128) {
        const uint t = idx / dv;
        const float p = static_cast<float>(sP[t]);
        const float v_val = static_cast<float>(v[cdv_base + idx]);
        const float ks_val = static_cast<float>(ks_entry[cdv_base + idx]);
        const float qs_val = static_cast<float>(q_s[cdv_base + idx]);
        v_prime[cdv_base + idx] = static_cast<bfloat>(v_val - ks_val * p);
        q_s_scaled[cdv_base + idx] = static_cast<bfloat>(qs_val * p);
    }

    if (tid < chunk_size) {
        const float decay = exp(sBigG[chunk_size - 1] - sBigG[tid]);
        decay_last_col[g_base + tid] = static_cast<bfloat>(decay);
    }
}
"#;

const METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_recurrent_prefill_head_last_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* beta [[buffer(3)]],
    device const bfloat* g [[buffer(4)]],
    device bfloat* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& batch_heads [[buffer(7)]],
    constant uint& seq_len [[buffer(8)]],
    constant uint& dk [[buffer(9)]],
    constant uint& dv [[buffer(10)]],
    constant uint& value_heads [[buffer(11)]],
    constant uint& q_heads [[buffer(12)]],
    constant uint& input_mode [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint LANES = 32;
    if (gid >= batch_heads * dv || tid >= LANES || dk != 128) {
        return;
    }

    const uint bh = gid / dv;
    const uint d = gid - bh * dv;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * seq_len * dk;
    const uint v_base = bh * seq_len * dv;
    const uint gate_base = bh * seq_len;
    const uint state_base = bh * dk * dv;

    float ls[NSG];
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        ls[j] = static_cast<float>(state[state_base + is * dv + d]);
    }

    for (uint t = 0; t < seq_len; ++t) {
        const uint qk_t = (input_mode == 0)
            ? qk_base + t * dk
            : ((batch_idx * seq_len + t) * q_heads + q_head_idx) * dk;
        const uint v_t = (input_mode == 0)
            ? v_base + t * dv
            : ((batch_idx * seq_len + t) * value_heads + head_idx) * dv;
        const uint gate_t = (input_mode == 0)
            ? gate_base + t
            : (batch_idx * seq_len + t) * value_heads + head_idx;
        const float decay = static_cast<float>(static_cast<bfloat>(
            exp(static_cast<float>(g[gate_t]))
        ));

        float s_k = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float decayed = static_cast<float>(static_cast<bfloat>(ls[j] * decay));
            ls[j] = decayed;
            s_k += decayed * static_cast<float>(k[qk_t + is]);
        }
        s_k = simd_sum(s_k);

        const float delta = static_cast<float>(static_cast<bfloat>(
            (static_cast<float>(v[v_t + d]) - s_k) *
            static_cast<float>(beta[gate_t])
        ));

        float y = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float new_s = static_cast<float>(static_cast<bfloat>(
                ls[j] + static_cast<float>(k[qk_t + is]) * delta
            ));
            ls[j] = new_s;
            y += new_s * static_cast<float>(q[qk_t + is]);
        }
        y = simd_sum(y);

        if (tid == 0) {
            const uint out_idx = ((batch_idx * seq_len + t) * value_heads + head_idx) * dv + d;
            out[out_idx] = static_cast<bfloat>(y);
        }
    }

    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        state[state_base + is * dv + d] = static_cast<bfloat>(ls[j]);
    }
}
"#;

const METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_DECAY_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_recurrent_prefill_head_last_decay_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* beta [[buffer(3)]],
    device const bfloat* decay [[buffer(4)]],
    device bfloat* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& batch_heads [[buffer(7)]],
    constant uint& seq_len [[buffer(8)]],
    constant uint& dk [[buffer(9)]],
    constant uint& dv [[buffer(10)]],
    constant uint& value_heads [[buffer(11)]],
    constant uint& q_heads [[buffer(12)]],
    constant uint& input_mode [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint LANES = 32;
    if (gid >= batch_heads * dv || tid >= LANES || dk != 128) {
        return;
    }

    const uint bh = gid / dv;
    const uint d = gid - bh * dv;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * seq_len * dk;
    const uint v_base = bh * seq_len * dv;
    const uint gate_base = bh * seq_len;
    const uint state_base = bh * dk * dv;

    float ls[NSG];
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        ls[j] = static_cast<float>(state[state_base + is * dv + d]);
    }

    for (uint t = 0; t < seq_len; ++t) {
        const uint qk_t = (input_mode == 0)
            ? qk_base + t * dk
            : ((batch_idx * seq_len + t) * q_heads + q_head_idx) * dk;
        const uint v_t = (input_mode == 0)
            ? v_base + t * dv
            : ((batch_idx * seq_len + t) * value_heads + head_idx) * dv;
        const uint gate_t = (input_mode == 0)
            ? gate_base + t
            : (batch_idx * seq_len + t) * value_heads + head_idx;
        const float decay_t = static_cast<float>(decay[gate_t]);

        float s_k = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float decayed = static_cast<float>(static_cast<bfloat>(ls[j] * decay_t));
            ls[j] = decayed;
            s_k += decayed * static_cast<float>(k[qk_t + is]);
        }
        s_k = simd_sum(s_k);

        const float delta = static_cast<float>(static_cast<bfloat>(
            (static_cast<float>(v[v_t + d]) - s_k) *
            static_cast<float>(beta[gate_t])
        ));

        float y = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float new_s = static_cast<float>(static_cast<bfloat>(
                ls[j] + static_cast<float>(k[qk_t + is]) * delta
            ));
            ls[j] = new_s;
            y += new_s * static_cast<float>(q[qk_t + is]);
        }
        y = simd_sum(y);

        if (tid == 0) {
            const uint out_idx = ((batch_idx * seq_len + t) * value_heads + head_idx) * dv + d;
            out[out_idx] = static_cast<bfloat>(y);
        }
    }

    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        state[state_base + is * dv + d] = static_cast<bfloat>(ls[j]);
    }
}
"#;

const METAL_GDN_FULL_CHUNK_FORWARD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_full_chunk_forward_bf16(
    device const bfloat* g [[buffer(0)]],
    device const bfloat* v [[buffer(1)]],
    device const bfloat* kkt [[buffer(2)]],
    device const bfloat* qkt [[buffer(3)]],
    device const bfloat* ks_entry [[buffer(4)]],
    device const bfloat* q_s [[buffer(5)]],
    device const bfloat* beta [[buffer(6)]],
    device const bfloat* k_t [[buffer(7)]],
    device bfloat* state [[buffer(8)]],
    device bfloat* out [[buffer(9)]],
    constant uint& batch_heads [[buffer(10)]],
    constant uint& dk [[buffer(11)]],
    constant uint& dv [[buffer(12)]],
    constant uint& output_mode [[buffer(13)]],
    constant uint& t_start [[buffer(14)]],
    constant uint& seq_len [[buffer(15)]],
    constant uint& heads [[buffer(16)]],
    constant uint& g_bh_stride [[buffer(17)]],
    constant uint& g_t_stride [[buffer(18)]],
    constant uint& v_bh_stride [[buffer(19)]],
    constant uint& v_t_stride [[buffer(20)]],
    constant uint& v_d_stride [[buffer(21)]],
    constant uint& beta_bh_stride [[buffer(22)]],
    constant uint& beta_t_stride [[buffer(23)]],
    constant uint& kt_bh_stride [[buffer(24)]],
    constant uint& kt_k_stride [[buffer(25)]],
    constant uint& kt_t_stride [[buffer(26)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint C = 64;
    constexpr uint MAX_DV = 128;
    if (bh >= batch_heads) {
        return;
    }

    threadgroup bfloat sArow[64];
    threadgroup bfloat sBrow[64];
    threadgroup bfloat sW[8192];
    threadgroup float sBigG[64];
    threadgroup float sP[64];
    threadgroup float sDecayLast[64];
    threadgroup float sPLast;

    const uint g_strided_base = bh * g_bh_stride;
    const uint v_strided_base = bh * v_bh_stride;
    const uint beta_base = bh * beta_bh_stride;
    const uint cdv_base = bh * C * dv;
    const uint cc_base = bh * C * C;
    const uint kt_strided_base = bh * kt_bh_stride;
    const uint state_base = bh * dk * dv;

    for (uint i = tid; i < C; i += 128) {
        sBigG[i] = static_cast<float>(g[g_strided_base + i * g_t_stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float acc = 0.0f;
        for (uint i = 0; i < C; ++i) {
            acc += sBigG[i];
            sBigG[i] = acc;
        }
        sPLast = exp(sBigG[C - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < C; i += 128) {
        sP[i] = exp(sBigG[i]);
        sDecayLast[i] = exp(sBigG[C - 1] - sBigG[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < C; ++t) {
        for (uint i = tid; i < C; i += 128) {
            const uint ti = t * C + i;
            const float decay = exp(sBigG[t] - sBigG[i]);
            const float a_val = (i < t) ? static_cast<float>(kkt[cc_base + ti]) * decay : 0.0f;
            const float b_val = (i <= t) ? static_cast<float>(qkt[cc_base + ti]) * decay : 0.0f;
            sArow[i] = static_cast<bfloat>(a_val);
            sBrow[i] = static_cast<bfloat>(b_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float beta_t = static_cast<float>(beta[beta_base + t * beta_t_stride]);
        const float p_t = static_cast<float>(static_cast<bfloat>(sP[t]));

        if (tid < dv) {
            float acc_a = 0.0f;
            for (uint i = 0; i < t; ++i) {
                acc_a += static_cast<float>(sArow[i]) *
                         static_cast<float>(sW[i * MAX_DV + tid]);
            }

            const uint td = t * dv + tid;
            const float vp = static_cast<float>(static_cast<bfloat>(
                static_cast<float>(v[v_strided_base + t * v_t_stride + tid * v_d_stride]) -
                static_cast<float>(ks_entry[cdv_base + td]) * p_t
            ));
            const float w_val = beta_t * (vp - acc_a);
            sW[t * MAX_DV + tid] = static_cast<bfloat>(w_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < dv) {
            float acc_b = 0.0f;
            for (uint i = 0; i <= t; ++i) {
                acc_b += static_cast<float>(sBrow[i]) *
                         static_cast<float>(sW[i * MAX_DV + tid]);
            }

            const uint td = t * dv + tid;
            const float qss = static_cast<float>(static_cast<bfloat>(
                static_cast<float>(q_s[cdv_base + td]) * p_t
            ));
            const bfloat out_val = static_cast<bfloat>(qss + acc_b);
            if (output_mode == 0) {
                out[cdv_base + td] = out_val;
            } else {
                const uint batch_idx = bh / heads;
                const uint head_idx = bh - batch_idx * heads;
                const uint out_t = t_start + t;
                const uint out_idx = ((batch_idx * seq_len + out_t) * heads + head_idx) * dv + tid;
                out[out_idx] = out_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < dv) {
        const float p_last = static_cast<float>(static_cast<bfloat>(sPLast));
        for (uint k_idx = 0; k_idx < dk; ++k_idx) {
            float delta = 0.0f;
            for (uint t = 0; t < C; ++t) {
                const float kt = static_cast<float>(
                    k_t[kt_strided_base + k_idx * kt_k_stride + t * kt_t_stride]
                );
                const float w = static_cast<float>(sW[t * MAX_DV + tid]);
                const float decay_last = static_cast<float>(static_cast<bfloat>(sDecayLast[t]));
                const float w_weighted = static_cast<float>(static_cast<bfloat>(w * decay_last));
                delta += kt * w_weighted;
            }
            const uint state_idx = state_base + k_idx * dv + tid;
            const float prev = static_cast<float>(state[state_idx]);
            state[state_idx] = static_cast<bfloat>(prev * p_last + delta);
        }
    }
}
"#;

fn metal_gdn_recurrent_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn recurrent pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_recurrent_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn recurrent function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn recurrent pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_recurrent_prefill_head_last_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn recurrent prefill pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_recurrent_prefill_head_last_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn recurrent prefill function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn recurrent prefill pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_forward_substitution_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn forward-substitution pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_forward_substitution_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn forward-substitution function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn forward-substitution pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_forward_substitution_f32_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal gdn forward-substitution f32 pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_forward_substitution_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn forward-substitution f32 function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn forward-substitution f32 pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_chunk_prep_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn chunk-prep pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_chunk_prep_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn chunk-prep function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn chunk-prep pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_recurrent_prefill_head_last_decay_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal gdn recurrent prefill decay pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_recurrent_prefill_head_last_decay_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn recurrent prefill decay function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn recurrent prefill decay pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_full_chunk_forward_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn full-chunk pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_full_chunk_forward_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn full-chunk function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn full-chunk pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_forward_substitution_bf16(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, chunk_size, _) = a_strict.dims4()?;
    let dv = v_prime.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn forward-substitution shape too large"
    );

    let a_strict = a_strict.contiguous()?;
    let v_prime = v_prime.contiguous()?;
    let beta = beta.contiguous()?;
    // The kernel writes every chunk/value element.
    let a_metal = kt_metal(&a_strict)?;
    let out = kt_metal_alloc(
        a_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_forward_substitution_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_forward_substitution_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v_prime)?;
        let beta_metal = kt_metal(&beta)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_prime.layout(), v_prime.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(4, &batch_heads_u32);
        encoder.set_bytes(5, &chunk_size_u32);
        encoder.set_bytes(6, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_forward_substitution_f32(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, chunk_size, _) = a_strict.dims4()?;
    let dv = v_prime.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn forward-substitution f32 shape too large"
    );

    let a_strict = a_strict.contiguous()?;
    let v_prime = v_prime.contiguous()?;
    let beta = beta.contiguous()?;
    let a_metal = kt_metal(&a_strict)?;
    let out = kt_metal_alloc(
        a_metal,
        kiln_tensor::DType::F32,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_forward_substitution_f32_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_forward_substitution_f32");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v_prime)?;
        let beta_metal = kt_metal(&beta)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_prime.layout(), v_prime.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(4, &batch_heads_u32);
        encoder.set_bytes(5, &chunk_size_u32);
        encoder.set_bytes(6, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_chunk_prep_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    let (batch, heads, chunk_size) = g.dims3()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn chunk-prep shape too large"
    );

    let g = g.contiguous()?;
    let v = v.contiguous()?;
    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;
    let g_metal = kt_metal(&g)?;
    // The prep kernel fills each temporary completely before any consumer sees it.
    let a_strict = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, chunk_size],
    )?;
    let b_mask = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, chunk_size],
    )?;
    let v_prime = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;
    let q_s_scaled = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;
    let decay_last_col = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size],
    )?;
    let p_last = kt_metal_alloc(g_metal, kiln_tensor::DType::BF16, &[batch, heads])?;

    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_chunk_prep_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_chunk_prep_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let a_metal = kt_metal(&a_strict)?;
        let b_metal = kt_metal(&b_mask)?;
        let vp_metal = kt_metal(&v_prime)?;
        let qss_metal = kt_metal(&q_s_scaled)?;
        let dl_metal = kt_metal(&decay_last_col)?;
        let pl_metal = kt_metal(&p_last)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b_mask.layout(), b_mask.dtype());
        let vp_buf = buffer_o_kt(
            vp_metal.buffer().as_ref(),
            v_prime.layout(),
            v_prime.dtype(),
        );
        let qss_buf = buffer_o_kt(
            qss_metal.buffer().as_ref(),
            q_s_scaled.layout(),
            q_s_scaled.dtype(),
        );
        let dl_buf = buffer_o_kt(
            dl_metal.buffer().as_ref(),
            decay_last_col.layout(),
            decay_last_col.dtype(),
        );
        let pl_buf = buffer_o_kt(pl_metal.buffer().as_ref(), p_last.layout(), p_last.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(vp_buf.buffer), vp_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(qss_buf.buffer), qss_buf.offset_in_bytes);
        encoder.set_buffer(10, Some(dl_buf.buffer), dl_buf.offset_in_bytes);
        encoder.set_buffer(11, Some(pl_buf.buffer), pl_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(12, &batch_heads_u32);
        encoder.set_bytes(13, &chunk_size_u32);
        encoder.set_bytes(14, &dv_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((
        a_strict,
        b_mask,
        v_prime,
        q_s_scaled,
        decay_last_col,
        p_last,
    ))
}

fn metal_gdn_full_chunk_forward_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state),
        "metal gdn full-chunk unsupported shape"
    );
    let (batch, heads, chunk_size) = g.dims3()?;
    let (_, _, dk, _) = k_t.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize && dk <= u32::MAX as usize && dv <= u32::MAX as usize,
        "metal gdn full-chunk shape too large"
    );

    let g = g.contiguous()?;
    let v = v.contiguous()?;
    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;
    let beta = beta.contiguous()?;
    let k_t = k_t.contiguous()?;
    let g_metal = kt_metal(&g)?;
    // The full-chunk kernel writes every output token/head/value element.
    let out = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_full_chunk_forward_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_full_chunk_forward_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let beta_metal = kt_metal(&beta)?;
        let kt_metal_storage = kt_metal(&k_t)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let kt_buf = buffer_o_kt(
            kt_metal_storage.buffer().as_ref(),
            k_t.layout(),
            k_t.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let output_mode_u32 = 0u32;
        let t_start_u32 = 0u32;
        let seq_len_u32 = chunk_size as u32;
        let heads_u32 = heads as u32;
        let g_stride = g.layout().strides();
        let v_stride = v.layout().strides();
        let beta_stride = beta.layout().strides();
        let kt_stride = k_t.layout().strides();
        let g_bh_stride_u32 = g_stride[1] as u32;
        let g_t_stride_u32 = g_stride[2] as u32;
        let v_bh_stride_u32 = v_stride[1] as u32;
        let v_t_stride_u32 = v_stride[2] as u32;
        let v_d_stride_u32 = v_stride[3] as u32;
        let beta_bh_stride_u32 = beta_stride[1] as u32;
        let beta_t_stride_u32 = beta_stride[2] as u32;
        let kt_bh_stride_u32 = kt_stride[1] as u32;
        let kt_k_stride_u32 = kt_stride[2] as u32;
        let kt_t_stride_u32 = kt_stride[3] as u32;
        encoder.set_bytes(10, &batch_heads_u32);
        encoder.set_bytes(11, &dk_u32);
        encoder.set_bytes(12, &dv_u32);
        encoder.set_bytes(13, &output_mode_u32);
        encoder.set_bytes(14, &t_start_u32);
        encoder.set_bytes(15, &seq_len_u32);
        encoder.set_bytes(16, &heads_u32);
        encoder.set_bytes(17, &g_bh_stride_u32);
        encoder.set_bytes(18, &g_t_stride_u32);
        encoder.set_bytes(19, &v_bh_stride_u32);
        encoder.set_bytes(20, &v_t_stride_u32);
        encoder.set_bytes(21, &v_d_stride_u32);
        encoder.set_bytes(22, &beta_bh_stride_u32);
        encoder.set_bytes(23, &beta_t_stride_u32);
        encoder.set_bytes(24, &kt_bh_stride_u32);
        encoder.set_bytes(25, &kt_k_stride_u32);
        encoder.set_bytes(26, &kt_t_stride_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn metal_gdn_full_chunk_forward_head_last_into_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    t_start: usize,
    seq_len: usize,
) -> Result<()> {
    anyhow::ensure!(
        metal_gdn_full_chunk_forward_head_last_supports(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
        ),
        "metal gdn full-chunk head-last unsupported shape"
    );
    let (batch, heads, _) = g.dims3()?;
    let (_, _, dk, _) = k_t.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && t_start <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && heads <= u32::MAX as usize,
        "metal gdn full-chunk head-last shape too large"
    );

    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;

    let g_metal = kt_metal(&g)?;
    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_full_chunk_forward_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_full_chunk_forward_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let beta_metal = kt_metal(&beta)?;
        let kt_metal_storage = kt_metal(&k_t)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let kt_buf = buffer_o_kt(
            kt_metal_storage.buffer().as_ref(),
            k_t.layout(),
            k_t.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let output_mode_u32 = 1u32;
        let t_start_u32 = t_start as u32;
        let seq_len_u32 = seq_len as u32;
        let heads_u32 = heads as u32;
        let g_stride = g.layout().strides();
        let v_stride = v.layout().strides();
        let beta_stride = beta.layout().strides();
        let kt_stride = k_t.layout().strides();
        let g_bh_stride_u32 = g_stride[1] as u32;
        let g_t_stride_u32 = g_stride[2] as u32;
        let v_bh_stride_u32 = v_stride[1] as u32;
        let v_t_stride_u32 = v_stride[2] as u32;
        let v_d_stride_u32 = v_stride[3] as u32;
        let beta_bh_stride_u32 = beta_stride[1] as u32;
        let beta_t_stride_u32 = beta_stride[2] as u32;
        let kt_bh_stride_u32 = kt_stride[1] as u32;
        let kt_k_stride_u32 = kt_stride[2] as u32;
        let kt_t_stride_u32 = kt_stride[3] as u32;
        encoder.set_bytes(10, &batch_heads_u32);
        encoder.set_bytes(11, &dk_u32);
        encoder.set_bytes(12, &dv_u32);
        encoder.set_bytes(13, &output_mode_u32);
        encoder.set_bytes(14, &t_start_u32);
        encoder.set_bytes(15, &seq_len_u32);
        encoder.set_bytes(16, &heads_u32);
        encoder.set_bytes(17, &g_bh_stride_u32);
        encoder.set_bytes(18, &g_t_stride_u32);
        encoder.set_bytes(19, &v_bh_stride_u32);
        encoder.set_bytes(20, &v_t_stride_u32);
        encoder.set_bytes(21, &v_d_stride_u32);
        encoder.set_bytes(22, &beta_bh_stride_u32);
        encoder.set_bytes(23, &beta_t_stride_u32);
        encoder.set_bytes(24, &kt_bh_stride_u32);
        encoder.set_bytes(25, &kt_k_stride_u32);
        encoder.set_bytes(26, &kt_t_stride_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(())
}

fn metal_gdn_recurrent_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, dk) = q.dims3()?;
    let dv = v.dim(2)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize && dk <= u32::MAX as usize && dv <= u32::MAX as usize,
        "metal gdn recurrent shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // The recurrent kernel writes every batch/head/value element.
    let out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, &[batch, heads, dv])?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &dk_u32);
        encoder.set_bytes(9, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_recurrent_prefill_head_last_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_head_last_supports(q, k, v, beta, g, state),
        "metal gdn recurrent prefill unsupported shape"
    );
    let (batch, q_heads, seq_len, dk) = q.dims4()?;
    let (_, value_heads, _, _) = v.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent prefill shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // SAFETY: the kernel dispatch covers every (batch, token, value-head, dv)
    // output element exactly once via `gid=batch_head*dv+d` and the token loop.
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 0u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_recurrent_prefill_native_head_last_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, g, state),
        "metal gdn recurrent native prefill unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent native prefill shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // SAFETY: the kernel dispatch covers every (batch, token, value-head, dv)
    // output element exactly once via `gid=batch_head*dv+d` and the token loop.
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_native_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 1u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_gdn_recurrent_prefill_native_head_last_decay_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    decay: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_native_head_last_decay_supports(q, k, v, beta, decay, state),
        "metal gdn recurrent native prefill decay unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent native prefill decay shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let decay = decay.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_decay_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_native_head_last_decay_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(decay_buf.buffer), decay_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 1u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

const METAL_CONV1D_PREFILL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_causal_conv1d_prefill_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& threadgroup_width [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint total_channels = batch * channels;
    if (gid >= total_channels) {
        return;
    }

    const uint b = gid / channels;
    const uint c = gid - b * channels;
    const uint x_base = (b * channels + c) * seq_len;
    const uint state_base = (b * channels + c) * 3;
    const uint weight_base = c * 4;

    threadgroup float s_state[3];
    if (tid < 3) {
        s_state[tid] = conv_state[state_base + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid; t < seq_len; t += threadgroup_width) {
        float acc = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            const uint padded_idx = t + j;
            float v;
            if (padded_idx < 3) {
                v = s_state[padded_idx];
            } else {
                v = static_cast<float>(x[x_base + padded_idx - 3]);
            }
            acc += v * static_cast<float>(weight[weight_base + j]);
        }
        out[x_base + t] = acc / (1.0f + exp(-acc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        if (seq_len >= 3) {
            conv_state[state_base + 0] = static_cast<float>(x[x_base + seq_len - 3]);
            conv_state[state_base + 1] = static_cast<float>(x[x_base + seq_len - 2]);
            conv_state[state_base + 2] = static_cast<float>(x[x_base + seq_len - 1]);
        } else if (seq_len == 2) {
            conv_state[state_base + 0] = s_state[2];
            conv_state[state_base + 1] = static_cast<float>(x[x_base + 0]);
            conv_state[state_base + 2] = static_cast<float>(x[x_base + 1]);
        } else if (seq_len == 1) {
            conv_state[state_base + 0] = s_state[1];
            conv_state[state_base + 1] = s_state[2];
            conv_state[state_base + 2] = static_cast<float>(x[x_base]);
        }
    }
}

kernel void kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* q_out [[buffer(3)]],
    device float* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& channels [[buffer(8)]],
    constant uint& qk_dim [[buffer(9)]],
    constant uint& v_dim [[buffer(10)]],
    constant uint& threadgroup_width [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint total_channels = batch * channels;
    if (gid >= total_channels) {
        return;
    }

    const uint b = gid / channels;
    const uint c = gid - b * channels;
    const uint state_base = (b * channels + c) * 3;
    const uint weight_base = c * 4;

    threadgroup float s_state[3];
    if (tid < 3) {
        s_state[tid] = conv_state[state_base + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid; t < seq_len; t += threadgroup_width) {
        float acc = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            const uint padded_idx = t + j;
            float xv;
            if (padded_idx < 3) {
                xv = s_state[padded_idx];
            } else {
                const uint x_idx = (b * seq_len + (padded_idx - 3)) * channels + c;
                xv = static_cast<float>(x[x_idx]);
            }
            acc += xv * static_cast<float>(weight[weight_base + j]);
        }

        const float y = acc / (1.0f + exp(-acc));
        if (c < qk_dim) {
            q_out[(b * seq_len + t) * qk_dim + c] = y;
        } else if (c < qk_dim * 2) {
            k_out[(b * seq_len + t) * qk_dim + (c - qk_dim)] = y;
        } else {
            v_out[(b * seq_len + t) * v_dim + (c - qk_dim * 2)] = static_cast<bfloat>(y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        if (seq_len >= 3) {
            conv_state[state_base + 0] =
                static_cast<float>(x[(b * seq_len + (seq_len - 3)) * channels + c]);
            conv_state[state_base + 1] =
                static_cast<float>(x[(b * seq_len + (seq_len - 2)) * channels + c]);
            conv_state[state_base + 2] =
                static_cast<float>(x[(b * seq_len + (seq_len - 1)) * channels + c]);
        } else if (seq_len == 2) {
            conv_state[state_base + 0] = s_state[2];
            conv_state[state_base + 1] = static_cast<float>(x[(b * seq_len) * channels + c]);
            conv_state[state_base + 2] = static_cast<float>(x[(b * seq_len + 1) * channels + c]);
        } else if (seq_len == 1) {
            conv_state[state_base + 0] = s_state[1];
            conv_state[state_base + 1] = s_state[2];
            conv_state[state_base + 2] = static_cast<float>(x[(b * seq_len) * channels + c]);
        }
    }
}
"#;

const METAL_CONV1D_UPDATE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_causal_conv1d_update_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = batch * channels;
    if (gid >= total) {
        return;
    }

    const uint c = gid % channels;
    const uint state_base = gid * 3;
    const uint weight_base = c * 4;

    const float s0 = conv_state[state_base + 0];
    const float s1 = conv_state[state_base + 1];
    const float s2 = conv_state[state_base + 2];
    const float x0 = static_cast<float>(x[gid]);

    const float acc =
        s0 * static_cast<float>(weight[weight_base + 0]) +
        s1 * static_cast<float>(weight[weight_base + 1]) +
        s2 * static_cast<float>(weight[weight_base + 2]) +
        x0 * static_cast<float>(weight[weight_base + 3]);

    out[gid] = acc / (1.0f + exp(-acc));
    conv_state[state_base + 0] = s1;
    conv_state[state_base + 1] = s2;
    conv_state[state_base + 2] = x0;
}
"#;

fn metal_conv1d_prefill_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal conv1d prefill pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_causal_conv1d_prefill_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal conv1d prefill function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal conv1d prefill pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_gdn_prefill_qkv_conv_split_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn prefill qkv conv-split pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn prefill qkv conv-split function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn prefill qkv conv-split pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

fn metal_conv1d_update_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal conv1d update pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_causal_conv1d_update_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal conv1d update function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal conv1d update pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_prefill_qkv_conv_split_supports(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> bool {
    if metal_gdn_prefill_qkv_conv_split_disabled() {
        return false;
    }
    if mixed_qkv.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    if !matches!(mixed_qkv.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
        || !matches!(conv_state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !mixed_qkv.is_contiguous() || !weight.is_contiguous() || !conv_state.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, channels)) = mixed_qkv.dims3() else {
        return false;
    };
    let Ok((s_batch, s_channels, s_width)) = conv_state.dims3() else {
        return false;
    };
    let qk_dim = nk.saturating_mul(dk);
    let v_dim = nv.saturating_mul(dv);
    let Some(expected_channels) = qk_dim.checked_mul(2).and_then(|n| n.checked_add(v_dim)) else {
        return false;
    };
    let weight_ok = match weight.rank() {
        2 => weight.dims() == [channels, kernel_size],
        3 => weight.dims() == [channels, 1, kernel_size],
        _ => false,
    };
    batch <= u32::MAX as usize
        && seq_len > 1
        && seq_len <= u32::MAX as usize
        && channels == expected_channels
        && channels <= u32::MAX as usize
        && qk_dim <= u32::MAX as usize
        && v_dim <= u32::MAX as usize
        && kernel_size == 4
        && (s_batch, s_channels, s_width) == (batch, channels, 3)
        && weight_ok
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_prefill_qkv_conv_split_bf16_f32_k4(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_prefill_qkv_conv_split_supports(
            mixed_qkv,
            weight,
            conv_state,
            kernel_size,
            nk,
            dk,
            nv,
            dv,
        ),
        "metal gdn prefill qkv conv-split unsupported shape"
    );
    let (batch, seq_len, channels) = mixed_qkv.dims3()?;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;

    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal gdn prefill qkv conv-split weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }
    let mixed_qkv_metal = kt_metal(mixed_qkv)?;
    let q = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::F32,
        &[batch, seq_len, nk, dk],
    )?;
    let k = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::F32,
        &[batch, seq_len, nk, dk],
    )?;
    let v = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, dv],
    )?;

    let companion = mixed_qkv_metal.companion()?;
    let pipeline = metal_gdn_prefill_qkv_conv_split_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(conv_state)?;
        let q_metal = kt_metal(&q)?;
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(
            mixed_qkv_metal.buffer().as_ref(),
            mixed_qkv.layout(),
            mixed_qkv.dtype(),
        );
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(v_buf.buffer), v_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let channels_u32 = channels as u32;
        let qk_dim_u32 = qk_dim as u32;
        let v_dim_u32 = v_dim as u32;
        let threads = seq_len.next_power_of_two().clamp(32, 256);
        let threads_u32 = threads as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &seq_len_u32);
        encoder.set_bytes(8, &channels_u32);
        encoder.set_bytes(9, &qk_dim_u32);
        encoder.set_bytes(10, &v_dim_u32);
        encoder.set_bytes(11, &threads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch * channels,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q, k, v))
}

fn metal_causal_conv1d_prefill_bf16_f32_k4(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(kernel_size == 4, "metal conv1d prefill only supports K=4");
    let (batch, channels, seq_len) = x.dims3()?;
    anyhow::ensure!(seq_len > 1, "metal conv1d prefill requires seq_len > 1");

    let x = x.contiguous()?;
    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal conv1d prefill weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }
    // The conv prefill kernel writes every batch/channel/time element.
    let x_metal = kt_metal(&x)?;
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::F32,
        &[batch, channels, seq_len],
    )?;

    let companion = x_metal.companion()?;
    let pipeline = metal_conv1d_prefill_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_causal_conv1d_prefill_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(&conv_state)?;
        let o_metal = kt_metal(&out)?;

        // #1082 Step 4 conv1d-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let o_buf = buffer_o_kt(o_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(o_buf.buffer), o_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let channels_u32 = channels as u32;
        let seq_len_u32 = seq_len as u32;
        let threads = seq_len.next_power_of_two().clamp(32, 256);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &batch_u32);
        encoder.set_bytes(5, &channels_u32);
        encoder.set_bytes(6, &seq_len_u32);
        encoder.set_bytes(7, &threads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch * channels,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_causal_conv1d_update_bf16_f32_k4(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(kernel_size == 4, "metal conv1d update only supports K=4");
    let (batch, channels, seq_len) = x.dims3()?;
    anyhow::ensure!(seq_len == 1, "metal conv1d update requires seq_len == 1");

    let x = x.contiguous()?;
    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal conv1d update weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }
    // The conv update kernel writes every batch/channel element.
    let x_metal = kt_metal(&x)?;
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::F32, &[batch, channels, 1usize])?;

    let companion = x_metal.companion()?;
    let pipeline = metal_conv1d_update_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_causal_conv1d_update_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(&conv_state)?;
        let o_metal = kt_metal(&out)?;

        // #1082 Step 4 conv1d-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let o_buf = buffer_o_kt(o_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(o_buf.buffer), o_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let channels_u32 = channels as u32;
        encoder.set_bytes(4, &batch_u32);
        encoder.set_bytes(5, &channels_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch * channels,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

/// Test/helper: try to initialize a kt Metal device, returning `None` if Metal
/// isn't available or if device discovery panics in a sandboxed runner.
#[doc(hidden)]
pub fn try_new_metal() -> Option<kiln_tensor::Device> {
    let result = std::panic::catch_unwind(|| kiln_tensor::primary_metal_companion(0));
    match result {
        Ok(Ok(_)) => Some(kiln_tensor::Device::Metal(0)),
        Ok(Err(e)) => {
            eprintln!("Metal unavailable: {e}");
            None
        }
        Err(_) => {
            eprintln!("Metal device init panicked (likely CI sandbox with no Metal access)");
            None
        }
    }
}

#[cfg(test)]
mod metal_lm_head_sample_tests {
    use super::*;
    use crate::backend::BackendRuntime;
    use kiln_tensor::{Device, Tensor};
    use std::cmp::Ordering;

    fn metal_device() -> Option<Device> {
        super::try_new_metal()
    }

    fn pattern_bf16(n: usize, seed: u64) -> Vec<half::bf16> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in 0..n {
            s = s
                .wrapping_add(0xA076_1D64_78BD_642F)
                .wrapping_mul(0xE703_7ED1_A0B4_28DB);
            let raw = ((s >> 40) as u32 % 4096) as f32 / 1024.0 - 2.0;
            let trend = (i % 19) as f32 * 0.011;
            out.push(half::bf16::from_f32(raw + trend));
        }
        out
    }

    fn lm_head_logits_for_row(
        x: &[half::bf16],
        weight_t: &[half::bf16],
        row: usize,
        hidden: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let mut logits = Vec::with_capacity(vocab);
        let row_base = row * hidden;
        for col in 0..vocab {
            let mut acc = 0.0f32;
            for i in 0..hidden {
                acc += x[row_base + i].to_f32() * weight_t[i * vocab + col].to_f32();
            }
            logits.push(half::bf16::from_f32(acc).to_f32());
        }
        logits
    }

    fn raw_argmax(logits: &[f32]) -> u32 {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0u32;
        for (idx, &score) in logits.iter().enumerate() {
            let idx = idx as u32;
            if score > best_score || (score == best_score && idx < best_idx) {
                best_score = score;
                best_idx = idx;
            }
        }
        best_idx
    }

    fn splitmix_uniform(seed: u64) -> f32 {
        let state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let bits = z ^ (z >> 31);
        let mantissa = ((bits >> 40) & 0xFF_FFFF) as u32;
        mantissa as f32 / 16_777_216.0
    }

    fn unseeded_style_seed(history: &[u32]) -> u64 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let history_hash = history.iter().fold(0xCBF29CE484222325u64, |acc, &token| {
            (acc ^ token as u64).wrapping_mul(0x100000001B3)
        });
        nanos.wrapping_add(history_hash)
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_sample(
        raw_logits: &[f32],
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: u64,
    ) -> u32 {
        if kiln_core::sampling::SamplingParams::values_are_effectively_greedy(temperature, top_k) {
            return raw_argmax(raw_logits);
        }

        let mut logits = raw_logits.to_vec();
        let rep_active = repetition_penalty.is_finite()
            && repetition_penalty > 0.0
            && (repetition_penalty - 1.0).abs() > f32::EPSILON;
        for (&idx, &count) in history_indices.iter().zip(history_counts.iter()) {
            let Some(score) = logits.get_mut(idx as usize) else {
                continue;
            };
            if rep_active {
                *score = if *score > 0.0 {
                    *score / repetition_penalty
                } else {
                    *score * repetition_penalty
                };
            }
            if presence_penalty.is_finite() && presence_penalty != 0.0 {
                *score -= presence_penalty;
            }
            if frequency_penalty.is_finite() && frequency_penalty != 0.0 {
                *score -= frequency_penalty * count as f32;
            }
        }

        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx as u32, score / temperature))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        indexed.truncate((top_k as usize).min(indexed.len()).max(1));

        let max_score = indexed[0].1;
        let mut probs: Vec<(u32, f32)> = indexed
            .iter()
            .map(|&(idx, score)| (idx, (score - max_score).exp()))
            .collect();
        let mut sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        if !sum.is_finite() || sum <= 0.0 {
            return indexed[0].0;
        }
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }

        if min_p.is_finite() && min_p > 0.0 {
            let threshold = min_p * probs[0].1;
            probs.retain(|&(_, p)| p >= threshold);
            if probs.is_empty() {
                return indexed[0].0;
            }
            sum = probs.iter().map(|(_, p)| *p).sum();
            if sum > 0.0 {
                for (_, p) in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        if top_p > 0.0 && top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, (_, p)) in probs.iter().enumerate() {
                cumsum += *p;
                if cumsum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            sum = probs.iter().map(|(_, p)| *p).sum();
            if sum > 0.0 {
                for (_, p) in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        let r = splitmix_uniform(seed);
        let mut cumsum = 0.0f32;
        for &(idx, p) in &probs {
            cumsum += p;
            if r < cumsum {
                return idx;
            }
        }
        probs.last().map(|&(idx, _)| idx).unwrap_or(indexed[0].0)
    }

    #[test]
    fn linear_decode_sample_top_k_one_ignores_penalties_and_matches_raw_argmax() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping Metal lm-head top_k=1 sample test");
            return Ok(());
        };
        let hidden = 8usize;
        let vocab = 17usize;
        let x_data = pattern_bf16(hidden, 1);
        let weight_data = pattern_bf16(hidden * vocab, 2);
        let x = Tensor::from_vec_on(dev, x_data.clone(), vec![1, 1, hidden])?;
        let weight_t = Tensor::from_vec_on(dev, weight_data.clone(), vec![hidden, vocab])?;
        let backend = MetalBackend::new(dev);
        let logits = lm_head_logits_for_row(&x_data, &weight_data, 0, hidden, vocab);
        let want = raw_argmax(&logits);

        let got = backend
            .linear_decode_sample(
                &x,
                &weight_t,
                &[want],
                &[100],
                1.4,
                3.0,
                0.2,
                0.7,
                1,
                0.5,
                0.1,
                0xCAFE_F00D_DEAD_BEEF,
            )?
            .context("Metal backend declined top_k=1 sampled decode")?;
        assert_eq!(got, want);
        Ok(())
    }

    #[test]
    fn metal_lm_head_sample_matches_reference_top_p_min_p_penalties_seeded() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping Metal lm-head seeded sample test");
            return Ok(());
        };
        let hidden = 9usize;
        let vocab = 37usize;
        let x_data = pattern_bf16(hidden, 3);
        let weight_data = pattern_bf16(hidden * vocab, 4);
        let x = Tensor::from_vec_on(dev, x_data.clone(), vec![1, 1, hidden])?;
        let weight_t = Tensor::from_vec_on(dev, weight_data.clone(), vec![hidden, vocab])?;
        let history_indices = [2u32, 5, 11, 23];
        let history_counts = [1u32, 3, 2, 4];
        let seed = 0x1234_5678_90AB_CDEF;
        let got = metal_lm_head_sample_bf16(
            &x,
            &weight_t,
            &history_indices,
            &history_counts,
            1.2,
            0.4,
            0.15,
            0.8,
            7,
            0.82,
            0.03,
            seed,
        )?;
        let again = metal_lm_head_sample_bf16(
            &x,
            &weight_t,
            &history_indices,
            &history_counts,
            1.2,
            0.4,
            0.15,
            0.8,
            7,
            0.82,
            0.03,
            seed,
        )?;
        let logits = lm_head_logits_for_row(&x_data, &weight_data, 0, hidden, vocab);
        let want = reference_sample(
            &logits,
            &history_indices,
            &history_counts,
            1.2,
            0.4,
            0.15,
            0.8,
            7,
            0.82,
            0.03,
            seed,
        );
        assert_eq!(got, want);
        assert_eq!(again, want, "same seed must be deterministic");
        Ok(())
    }

    #[test]
    fn metal_lm_head_sample_matches_reference_top_k_top_p_unseeded_style_seed() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping Metal lm-head unseeded-style sample test");
            return Ok(());
        };
        let hidden = 11usize;
        let vocab = 43usize;
        let x_data = pattern_bf16(hidden, 7);
        let weight_data = pattern_bf16(hidden * vocab, 8);
        let x = Tensor::from_vec_on(dev, x_data.clone(), vec![1, 1, hidden])?;
        let weight_t = Tensor::from_vec_on(dev, weight_data.clone(), vec![hidden, vocab])?;
        let history = [3u32, 5, 3, 17, 5, 29];
        let (history_indices, history_counts): (Vec<u32>, Vec<u32>) =
            [(3u32, 2u32), (5, 2), (17, 1), (29, 1)].into_iter().unzip();
        let seed = unseeded_style_seed(&history);
        let got = metal_lm_head_sample_bf16(
            &x,
            &weight_t,
            &history_indices,
            &history_counts,
            1.0,
            0.0,
            0.0,
            0.95,
            11,
            0.7,
            0.0,
            seed,
        )?;
        let logits = lm_head_logits_for_row(&x_data, &weight_data, 0, hidden, vocab);
        let want = reference_sample(
            &logits,
            &history_indices,
            &history_counts,
            1.0,
            0.0,
            0.0,
            0.95,
            11,
            0.7,
            0.0,
            seed,
        );
        assert_eq!(got, want);
        Ok(())
    }

    #[test]
    fn linear_decode_sample_batch_handles_mixed_greedy_and_sampled_rows() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping Metal lm-head batched sample test");
            return Ok(());
        };
        let batch = 2usize;
        let hidden = 10usize;
        let vocab = 41usize;
        let x_data = pattern_bf16(batch * hidden, 5);
        let weight_data = pattern_bf16(hidden * vocab, 6);
        let x = Tensor::from_vec_on(dev, x_data.clone(), vec![batch, 1, hidden])?;
        let weight_t = Tensor::from_vec_on(dev, weight_data.clone(), vec![hidden, vocab])?;
        let backend = MetalBackend::new(dev);

        let tokens = backend
            .linear_decode_sample_batch(
                &x,
                &weight_t,
                &[1, 1, 1],
                &[3, 7, 19],
                &[2, 1, 4],
                &[1.0, 1.15],
                &[0.0, 0.35],
                &[0.0, 0.08],
                &[0.0, 0.9],
                &[0, 6],
                &[1.0, 0.74],
                &[0.0, 0.02],
                &[0xABCD, 0x1234_0000_5678_9999],
            )?
            .context("Metal backend declined batched sampled decode")?;
        assert_eq!(tokens.len(), batch);

        let row0_logits = lm_head_logits_for_row(&x_data, &weight_data, 0, hidden, vocab);
        let row1_logits = lm_head_logits_for_row(&x_data, &weight_data, 1, hidden, vocab);
        let want0 = raw_argmax(&row0_logits);
        let want1 = reference_sample(
            &row1_logits,
            &[3, 7, 19],
            &[2, 1, 4],
            1.15,
            0.35,
            0.08,
            0.9,
            6,
            0.74,
            0.02,
            0x1234_0000_5678_9999,
        );
        assert_eq!(tokens, vec![want0, want1]);
        Ok(())
    }

    #[test]
    fn sample_batch_support_does_not_claim_pure_greedy_batches() {
        let backend = MetalBackend::new(Device::Metal(0));
        assert!(!backend.supports_linear_decode_sample_batch(&[20], &[0.0]));
        assert!(!backend.supports_linear_decode_sample_batch(&[1, 1], &[0.7, 0.8]));
        assert!(backend.supports_linear_decode_sample_batch(&[20, 1], &[0.8, 0.0]));
    }
}

#[cfg(test)]
mod metal_icb_decode_tests {
    use super::*;
    use kiln_tensor::{Device, Tensor};

    fn metal_device() -> Option<Device> {
        kiln_tensor::primary_metal_companion(0)
            .ok()
            .map(|_| Device::Metal(0))
    }

    fn pattern_bf16(n: usize, seed: u64) -> Vec<half::bf16> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in 0..n {
            s = s
                .wrapping_add(0xA076_1D64_78BD_642F)
                .wrapping_mul(0xE703_7ED1_A0B4_28DB);
            let raw = ((s >> 40) as u32 % 1024) as f32 / 4096.0 - 0.125;
            let trend = (i % 17) as f32 * 0.0007;
            out.push(half::bf16::from_f32(raw + trend));
        }
        out
    }

    fn zeroed_bf16(n: usize) -> Vec<half::bf16> {
        vec![half::bf16::ZERO; n]
    }

    fn max_abs_diff_bf16(a: &[half::bf16], b: &[half::bf16]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b)
            .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn single_token_paged_decode_icb_matches_eager_and_updates_slot() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!(
                "Metal unavailable, skipping single_token_paged_decode_icb_matches_eager_and_updates_slot"
            );
            return Ok(());
        };

        let total_slots = 4usize;
        let kv_heads = 4usize;
        let q_heads = 16usize;
        let head_dim = 256usize;
        let pool_elems = total_slots * kv_heads * head_dim;
        let kv_elems = kv_heads * head_dim;
        let q_elems = q_heads * head_dim;
        let out_elems = q_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut k_pool_host = zeroed_bf16(pool_elems);
        let mut v_pool_host = zeroed_bf16(pool_elems);
        let prefix_k = pattern_bf16(2 * kv_elems, 10);
        let prefix_v = pattern_bf16(2 * kv_elems, 11);
        k_pool_host[..2 * kv_elems].copy_from_slice(&prefix_k);
        v_pool_host[..2 * kv_elems].copy_from_slice(&prefix_v);

        let q = Tensor::from_vec_on(
            dev,
            pattern_bf16(q_elems, 12),
            vec![1, 1, q_heads, head_dim],
        )?;
        let k = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 13),
            vec![1, 1, kv_heads, head_dim],
        )?;
        let v = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 14),
            vec![1, 1, kv_heads, head_dim],
        )?;
        let k_pool_eager = Tensor::from_vec_on(
            dev,
            k_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let v_pool_eager = Tensor::from_vec_on(
            dev,
            v_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let k_pool_icb =
            Tensor::from_vec_on(dev, k_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let v_pool_icb =
            Tensor::from_vec_on(dev, v_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let block_table = Tensor::from_vec_on(dev, vec![0u32, 1, 2], vec![1, 3])?;
        let seqused_k = Tensor::from_vec_on(dev, vec![3u32], vec![1])?;
        let out_icb =
            Tensor::from_vec_on(dev, zeroed_bf16(out_elems), vec![1, 1, q_heads, head_dim])?;

        let graph = metal_record_single_token_paged_decode_icb_graph(
            &q,
            &k_pool_icb,
            &v_pool_icb,
            &block_table,
            &seqused_k,
            &out_icb,
            &k,
            &v,
            2,
            3,
            1,
            scale,
        )?;

        metal_paged_kv_write_token_major_bf16(&k_pool_eager, &v_pool_eager, 2, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(2, 3, scale)?;

        let eager_0 = eager.to_vec::<half::bf16>()?;
        let icb_0 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_0, icb_0,
            "first ICB replay must be bit-identical to eager Metal decode"
        );

        let next_k = pattern_bf16(kv_elems, 20);
        let next_v = pattern_bf16(kv_elems, 21);
        kiln_tensor::metal_write_host_in_place(&k, &next_k)?;
        kiln_tensor::metal_write_host_in_place(&v, &next_v)?;
        kiln_tensor::metal_write_host_in_place(&block_table, &[0u32, 1, 3])?;

        metal_paged_kv_write_token_major_bf16(&k_pool_eager, &v_pool_eager, 3, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, 3, scale)?;

        let eager_1 = eager.to_vec::<half::bf16>()?;
        let icb_1 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_1, icb_1,
            "ICB replay after stable-buffer and slot updates must match eager"
        );
        assert_eq!(graph.replay_count(), 2);
        assert!(
            max_abs_diff_bf16(&icb_0, &icb_1) > 0.0,
            "second replay should observe refreshed K/V and metadata"
        );

        Ok(())
    }

    #[test]
    fn batched_paged_decode_icb_matches_eager_and_updates_slots() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!(
                "Metal unavailable, skipping batched_paged_decode_icb_matches_eager_and_updates_slots"
            );
            return Ok(());
        };

        let batch = 2usize;
        let total_slots = 8usize;
        let kv_heads = 4usize;
        let q_heads = 16usize;
        let head_dim = 256usize;
        let pool_row = kv_heads * head_dim;
        let pool_elems = total_slots * pool_row;
        let kv_elems = batch * pool_row;
        let q_elems = batch * q_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut k_pool_host = zeroed_bf16(pool_elems);
        let mut v_pool_host = zeroed_bf16(pool_elems);
        for row in 0..batch {
            let block_base = row * 4;
            for prefix_idx in 0..2 {
                let slot = block_base + prefix_idx;
                let dst = slot * pool_row;
                let seed = 100 + (row * 10 + prefix_idx) as u64;
                k_pool_host[dst..dst + pool_row].copy_from_slice(&pattern_bf16(pool_row, seed));
                v_pool_host[dst..dst + pool_row].copy_from_slice(&pattern_bf16(pool_row, seed + 1));
            }
        }

        let q = Tensor::from_vec_on(
            dev,
            pattern_bf16(q_elems, 12),
            vec![batch, 1, q_heads, head_dim],
        )?;
        let k = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 13),
            vec![batch, 1, kv_heads, head_dim],
        )?;
        let v = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 14),
            vec![batch, 1, kv_heads, head_dim],
        )?;
        let k_pool_eager = Tensor::from_vec_on(
            dev,
            k_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let v_pool_eager = Tensor::from_vec_on(
            dev,
            v_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let k_pool_icb =
            Tensor::from_vec_on(dev, k_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let v_pool_icb =
            Tensor::from_vec_on(dev, v_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let block_table = Tensor::from_vec_on(dev, vec![0u32, 1, 2, 4, 5, 6], vec![batch, 3])?;
        let seqused_k = Tensor::from_vec_on(dev, vec![3u32, 3], vec![batch])?;
        let slots = Tensor::from_vec_on(dev, vec![2u32, 6], vec![batch])?;
        let out_icb =
            Tensor::from_vec_on(dev, zeroed_bf16(q_elems), vec![batch, 1, q_heads, head_dim])?;

        let graph = metal_record_paged_decode_icb_graph(
            &q,
            &k_pool_icb,
            &v_pool_icb,
            &block_table,
            &seqused_k,
            &out_icb,
            &k,
            &v,
            &slots,
            3,
            1,
            scale,
        )?;

        metal_paged_kv_write_token_major_batch_bf16(&k_pool_eager, &v_pool_eager, &slots, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, scale)?;

        let eager_0 = eager.to_vec::<half::bf16>()?;
        let icb_0 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_0, icb_0,
            "first batched ICB replay must be bit-identical to eager Metal decode"
        );

        let next_k = pattern_bf16(kv_elems, 20);
        let next_v = pattern_bf16(kv_elems, 21);
        kiln_tensor::metal_write_host_in_place(&k, &next_k)?;
        kiln_tensor::metal_write_host_in_place(&v, &next_v)?;
        kiln_tensor::metal_write_host_in_place(&block_table, &[0u32, 1, 3, 4, 5, 7])?;
        kiln_tensor::metal_write_host_in_place(&slots, &[3u32, 7])?;

        metal_paged_kv_write_token_major_batch_bf16(&k_pool_eager, &v_pool_eager, &slots, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, scale)?;

        let eager_1 = eager.to_vec::<half::bf16>()?;
        let icb_1 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_1, icb_1,
            "batched ICB replay after stable slot updates must match eager"
        );
        assert_eq!(graph.replay_count(), 2);
        assert!(
            max_abs_diff_bf16(&icb_0, &icb_1) > 0.0,
            "second batched replay should observe refreshed K/V and metadata"
        );

        Ok(())
    }
}

// ----------------------------------------------------------------------
// On-device AdamW parity (#1082) — the optimizer oracle. A wrong optimizer
// silently corrupts training, so this gate compares the fused Metal
// `dispatch_adamw_step` (registry-resident, in-place) against the host
// reference math (a bit-faithful copy of `kiln_optim::AdamW::step`,
// adamw.rs ~165-181) over several steps, asserting param/m/v match to F32
// tolerance. Lives in a LIVE test module (not the candle-era `cfg(any())`
// block above) so it actually runs on the M1 validator.
#[cfg(test)]
mod adamw_kt_tests {
    use super::*;
    use kiln_tensor::{DType, Device, Tensor};

    /// `Device::Metal(0)` if a Metal device is reachable, else `None`.
    fn metal_device() -> Option<Device> {
        kiln_tensor::primary_metal_companion(0)
            .ok()
            .map(|_| Device::Metal(0))
    }

    /// One in-place AdamW step over f32 host buffers — the reference the
    /// kernel must match. Identical arithmetic + order to
    /// `kiln_optim::AdamW::step`.
    #[allow(clippy::too_many_arguments)]
    fn host_adamw_step(
        param: &mut [f32],
        m: &mut [f32],
        v: &mut [f32],
        grad: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let stepf = step as f32;
        let bc1 = 1.0 - beta1.powf(stepf);
        let bc2 = 1.0 - beta2.powf(stepf);
        for i in 0..param.len() {
            let g = grad[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            let update = lr * (m_hat / (v_hat.sqrt() + eps));
            param[i] -= lr * weight_decay * param[i];
            param[i] -= update;
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn dispatch_adamw_step_matches_host_reference_f32() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping dispatch_adamw_step_matches_host_reference_f32");
            return Ok(());
        };

        let n = 257usize; // non-multiple of 256 → exercises the tail thread
        let lr = 0.013f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.02f32;
        let steps = 5u32;

        // Deterministic, mildly varied data.
        let param0: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.017) - 2.1).sin() * 0.5)
            .collect();
        // A fresh grad per step keeps the moments moving.
        let grads: Vec<Vec<f32>> = (1..=steps)
            .map(|s| {
                (0..n)
                    .map(|i| ((i as f32 + s as f32 * 1.7) * 0.031).cos() * 0.08)
                    .collect::<Vec<f32>>()
            })
            .collect();

        // Host reference state.
        let mut h_param = param0.clone();
        let mut h_m = vec![0.0f32; n];
        let mut h_v = vec![0.0f32; n];

        // Metal state: param + m + v are persistent across steps (the kernel
        // mutates them in place), so build them once and register them.
        let met_param = Tensor::from_vec_on(dev, param0.clone(), vec![n])?;
        let met_m = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let met_v = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;

        let backend = MetalBackend::new(dev);
        assert!(backend.supports_resident_activation());
        backend.register_resident_activation(&met_param)?;
        backend.register_resident_activation(&met_m)?;
        backend.register_resident_activation(&met_v)?;
        assert!(backend.has_resident_activation(&met_param));
        assert!(backend.has_resident_activation(&met_m));
        assert!(backend.has_resident_activation(&met_v));

        for s in 1..=steps {
            let g = &grads[(s - 1) as usize];
            host_adamw_step(
                &mut h_param,
                &mut h_m,
                &mut h_v,
                g,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            );

            // Fresh grad tensor each step (distinct TensorId), mirroring the
            // trainer registering the grad on the fly.
            let met_grad = Tensor::from_vec_on(dev, g.clone(), vec![n])?;
            backend.register_resident_activation(&met_grad)?;

            let dispatched = backend.dispatch_adamw_step(
                &met_param,
                &met_grad,
                &met_m,
                &met_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            )?;
            assert!(
                dispatched,
                "dispatch_adamw_step must take the on-device path (step {s})"
            );
            backend.evict_resident_activation(&met_grad);
        }

        // Read the device results back to host.
        let g_param: Vec<f32> = met_param.to_device(Device::Cpu)?.to_vec::<f32>()?;
        let g_m: Vec<f32> = met_m.to_device(Device::Cpu)?.to_vec::<f32>()?;
        let g_v: Vec<f32> = met_v.to_device(Device::Cpu)?.to_vec::<f32>()?;

        let tol = 1e-5f32;
        let dp = max_abs_diff(&g_param, &h_param);
        let dm = max_abs_diff(&g_m, &h_m);
        let dv = max_abs_diff(&g_v, &h_v);
        eprintln!(
            "adamw parity over {steps} steps (n={n}): max|Δparam|={dp:e} max|Δm|={dm:e} max|Δv|={dv:e} (tol={tol:e})"
        );
        assert!(dp < tol, "param diverged: max|Δ|={dp:e} >= {tol:e}");
        assert!(dm < tol, "m diverged: max|Δ|={dm:e} >= {tol:e}");
        assert!(dv < tol, "v diverged: max|Δ|={dv:e} >= {tol:e}");

        // resolve_resident_activation must round-trip the in-place-updated
        // buffer (what `sync_to_master` relies on).
        let resolved = backend
            .resolve_resident_activation(&met_param, &[n], DType::F32)?
            .expect("param is resident, resolve must return Some");
        let r_param: Vec<f32> = resolved.to_device(Device::Cpu)?.to_vec::<f32>()?;
        assert!(
            max_abs_diff(&r_param, &g_param) < 1e-6,
            "resolve_resident_activation must reflect the in-place update"
        );

        backend.evict_resident_activation(&met_param);
        backend.evict_resident_activation(&met_m);
        backend.evict_resident_activation(&met_v);
        assert!(!backend.has_resident_activation(&met_param));
        Ok(())
    }

    /// BF16-master reference: mirrors the Metal kernel exactly — read each
    /// operand BF16→f32, run the AdamW math in f32, write the moments + master
    /// back as round-to-nearest BF16 (so the *stored* moments are lossy, the
    /// on-device convention shared with CUDA/Vulkan). Round-to-nearest-even
    /// matches MSL's `(bfloat)` conversion.
    #[allow(clippy::too_many_arguments)]
    fn host_adamw_step_bf16(
        param: &mut [half::bf16],
        m: &mut [half::bf16],
        v: &mut [half::bf16],
        grad: &[half::bf16],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let stepf = step as f32;
        let bc1 = 1.0 - beta1.powf(stepf);
        let bc2 = 1.0 - beta2.powf(stepf);
        for i in 0..param.len() {
            let g = grad[i].to_f32();
            let mf = beta1 * m[i].to_f32() + (1.0 - beta1) * g;
            let vf = beta2 * v[i].to_f32() + (1.0 - beta2) * g * g;
            let m_hat = mf / bc1;
            let v_hat = vf / bc2;
            let update = lr * (m_hat / (v_hat.sqrt() + eps));
            let mut pf = param[i].to_f32();
            pf -= lr * weight_decay * pf;
            pf -= update;
            m[i] = half::bf16::from_f32(mf);
            v[i] = half::bf16::from_f32(vf);
            param[i] = half::bf16::from_f32(pf);
        }
    }

    /// On-device BF16 AdamW (the real LoRA-training dtype) must match the BF16
    /// reference bit-for-bit: same f32 math, same round-to-nearest BF16 store.
    /// This is the on-device path actually exercised by the SFT/GRPO/OPD/GDN
    /// training smokes (their masters are BF16).
    #[test]
    fn dispatch_adamw_step_matches_bf16_reference() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping dispatch_adamw_step_matches_bf16_reference");
            return Ok(());
        };
        let n = 257usize;
        let (lr, beta1, beta2, eps, weight_decay) = (0.013f32, 0.9f32, 0.999f32, 1e-8f32, 0.02f32);
        let steps = 5u32;

        let to_bf16 = |xs: &[f32]| -> Vec<half::bf16> {
            xs.iter().map(|&x| half::bf16::from_f32(x)).collect()
        };
        let param0: Vec<half::bf16> = to_bf16(
            &(0..n)
                .map(|i| ((i as f32 * 0.017) - 2.1).sin() * 0.5)
                .collect::<Vec<_>>(),
        );
        let grads: Vec<Vec<half::bf16>> = (1..=steps)
            .map(|s| {
                to_bf16(
                    &(0..n)
                        .map(|i| ((i as f32 + s as f32 * 1.7) * 0.031).cos() * 0.08)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();

        let mut h_param = param0.clone();
        let mut h_m = vec![half::bf16::ZERO; n];
        let mut h_v = vec![half::bf16::ZERO; n];

        let met_param = Tensor::from_vec_on(dev, param0.clone(), vec![n])?;
        let met_m = Tensor::from_vec_on(dev, vec![half::bf16::ZERO; n], vec![n])?;
        let met_v = Tensor::from_vec_on(dev, vec![half::bf16::ZERO; n], vec![n])?;
        assert_eq!(met_param.dtype(), DType::BF16);

        let backend = MetalBackend::new(dev);
        backend.register_resident_activation(&met_param)?;
        backend.register_resident_activation(&met_m)?;
        backend.register_resident_activation(&met_v)?;

        for s in 1..=steps {
            let g = &grads[(s - 1) as usize];
            host_adamw_step_bf16(
                &mut h_param,
                &mut h_m,
                &mut h_v,
                g,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            );
            let met_grad = Tensor::from_vec_on(dev, g.clone(), vec![n])?;
            backend.register_resident_activation(&met_grad)?;
            let dispatched = backend.dispatch_adamw_step(
                &met_param,
                &met_grad,
                &met_m,
                &met_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            )?;
            assert!(
                dispatched,
                "BF16 dispatch_adamw_step must take the on-device path (step {s})"
            );
            backend.evict_resident_activation(&met_grad);
        }

        let g_param = met_param.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        let g_m = met_m.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        let g_v = met_v.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        // Bit-exact expected (identical f32 math + round-to-nearest store); allow
        // a hair for any MSL-vs-Rust sqrt/div last-bit nuance.
        let f = |a: &[half::bf16]| a.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let dp = max_abs_diff(&f(&g_param), &f(&h_param));
        let dm = max_abs_diff(&f(&g_m), &f(&h_m));
        let dv = max_abs_diff(&f(&g_v), &f(&h_v));
        eprintln!(
            "adamw bf16 parity (n={n}, {steps} steps): max|Δp|={dp:e} max|Δm|={dm:e} max|Δv|={dv:e}"
        );
        assert!(dp < 1e-2, "bf16 param diverged: {dp:e}");
        assert!(dm < 1e-3, "bf16 m diverged: {dm:e}");
        assert!(dv < 1e-4, "bf16 v diverged: {dv:e}");
        Ok(())
    }

    /// dispatch_adamw_step must decline (Ok(false)) when an operand isn't
    /// resident, so the trainer falls through to the host AdamW.
    #[test]
    fn dispatch_adamw_step_declines_when_not_resident() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            return Ok(());
        };
        let n = 8usize;
        let p = Tensor::from_vec_on(dev, vec![0.1f32; n], vec![n])?;
        let g = Tensor::from_vec_on(dev, vec![0.2f32; n], vec![n])?;
        let m = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let v = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let backend = MetalBackend::new(dev);
        // Nothing registered → decline.
        let dispatched =
            backend.dispatch_adamw_step(&p, &g, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1)?;
        assert!(!dispatched, "must decline when operands aren't resident");
        Ok(())
    }
}
