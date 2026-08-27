//! Metal ICB graph resource and replay wrappers.
//!
//! The low-level command recording still lives next to the Metal kernel
//! encoders in `metal.rs`; this module owns the reusable scalar buffers,
//! resource refs, argument blocks, and captured graph wrappers.

use anyhow::Result;

use kiln_graph::{
    CaptureError, InvalidateReason, ReplayInputs, ReplayKey, ReplayOutputs, ReplayPlan,
    ReplayResourceStability, ReplayState, ResidentResourceRef,
};
use kiln_tensor::Backend;
use kiln_tensor::metal_types::{
    Buffer, MTLResourceOptions, MTLResourceUsage, MetalCompanion, MetalRawDevice,
};

use crate::execution_phase::{GraphPhase, GraphPhaseTimer};

#[derive(Clone, Debug)]
pub(crate) struct MetalGraphScalarBuffer {
    buffer: Buffer,
}

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

#[derive(Clone, Debug)]
pub(crate) struct MetalGraphResourceRef {
    buffer: Buffer,
    usage: MTLResourceUsage,
}

impl MetalGraphResourceRef {
    pub(crate) fn read(buffer: &Buffer) -> Self {
        Self {
            buffer: buffer.clone(),
            usage: MTLResourceUsage::Read,
        }
    }

    pub(crate) fn write(buffer: &Buffer) -> Self {
        Self {
            buffer: buffer.clone(),
            usage: MTLResourceUsage::Write,
        }
    }
}

// Test-lane only: every caller sits in the single-token ICB capture/replay
// path (`metal_record_single_token_paged_decode_icb_graph` + its `#[cfg(test)]`
// consumer); the module is metal-gated, so this allow is inert elsewhere.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct MetalPagedKvWriteTokenMajorIcbArgs {
    pub(crate) slot: MetalGraphScalarBuffer,
    pub(crate) heads: MetalGraphScalarBuffer,
    pub(crate) head_dim: MetalGraphScalarBuffer,
}

// `new`/`update_slot`/`scalar_resources` are all consumed only by the
// single-token ICB capture + replay tests; allow required for the metal
// non-test build.
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

    pub(crate) fn scalar_resources(&self) -> [MetalGraphResourceRef; 3] {
        [
            MetalGraphResourceRef::read(self.slot.buffer()),
            MetalGraphResourceRef::read(self.heads.buffer()),
            MetalGraphResourceRef::read(self.head_dim.buffer()),
        ]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MetalPagedKvWriteTokenMajorBatchIcbArgs {
    pub(crate) batch: MetalGraphScalarBuffer,
    pub(crate) heads: MetalGraphScalarBuffer,
    pub(crate) head_dim: MetalGraphScalarBuffer,
    pub(crate) total_slots: MetalGraphScalarBuffer,
}

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

    pub(crate) fn scalar_resources(&self) -> [MetalGraphResourceRef; 4] {
        [
            MetalGraphResourceRef::read(self.batch.buffer()),
            MetalGraphResourceRef::read(self.heads.buffer()),
            MetalGraphResourceRef::read(self.head_dim.buffer()),
            MetalGraphResourceRef::read(self.total_slots.buffer()),
        ]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MetalPagedAttnDecodeDynSeqlenIcbArgs {
    pub(crate) batch: MetalGraphScalarBuffer,
    pub(crate) max_blocks_per_seq: MetalGraphScalarBuffer,
    pub(crate) max_seqlen_k: MetalGraphScalarBuffer,
    pub(crate) page_block_size: MetalGraphScalarBuffer,
    pub(crate) q_heads: MetalGraphScalarBuffer,
    pub(crate) kv_heads: MetalGraphScalarBuffer,
    pub(crate) softmax_scale: MetalGraphScalarBuffer,
    pub(crate) total_slots: MetalGraphScalarBuffer,
}

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

    pub(crate) fn scalar_resources(&self) -> [MetalGraphResourceRef; 8] {
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

// Test-lane only: constructed solely by
// `metal_record_single_token_paged_decode_icb_graph`, whose only caller is the
// `#[cfg(test)]` single-token ICB replay test; allow required for the metal
// non-test build.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MetalSingleTokenPagedDecodeIcbGraph {
    pub(crate) captured: kiln_graph_metal::MetalCapturedGraph,
    pub(crate) kv_args: MetalPagedKvWriteTokenMajorIcbArgs,
    pub(crate) attn_args: MetalPagedAttnDecodeDynSeqlenIcbArgs,
}

// `replay`/`replay_count` are exercised only by the `#[cfg(test)]`
// single-token ICB replay test; allow required for the metal non-test build.
#[allow(dead_code)]
impl MetalSingleTokenPagedDecodeIcbGraph {
    pub(crate) fn replay(&self, slot: u32, max_seqlen_k: u32, softmax_scale: f32) -> Result<()> {
        let _phase = GraphPhaseTimer::start(GraphPhase::Replay);
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

#[allow(clippy::too_many_arguments)]
#[derive(Debug)]
pub(crate) struct MetalPagedDecodeIcbGraph {
    pub(crate) captured: kiln_graph_metal::MetalCapturedGraph,
    pub(crate) attn_args: MetalPagedAttnDecodeDynSeqlenIcbArgs,
    pub(crate) replay_state: ReplayState,
}

// `replay` (direct method) is consumed only by the `#[cfg(test)]` batched ICB
// replay test — production replays through `replay_plan` + the `ReplayPlan`
// impl (`metal_graph.rs`), so the allow is required for the metal non-test
// build.
#[allow(dead_code)]
impl MetalPagedDecodeIcbGraph {
    pub(crate) fn replay(&self, max_seqlen_k: u32, softmax_scale: f32) -> Result<()> {
        let mut plan = self.replay_plan(max_seqlen_k, softmax_scale);
        let replay_key = ReplayPlan::key(&plan);
        let replay_inputs = ReplayInputs::new(&replay_key, self.replay_resources());
        ReplayPlan::replay(&mut plan, replay_inputs)
            .map_err(|e| anyhow::anyhow!("Metal ICB paged decode replay: {e}"))?;
        Ok(())
    }

    pub(crate) fn replay_plan(
        &self,
        max_seqlen_k: u32,
        softmax_scale: f32,
    ) -> MetalPagedDecodeReplayPlan<'_> {
        MetalPagedDecodeReplayPlan {
            graph: self,
            max_seqlen_k,
            softmax_scale,
        }
    }

    pub(crate) fn replay_resources(&self) -> &[ResidentResourceRef] {
        &self.replay_state.inputs
    }

    fn replay_native(&self, max_seqlen_k: u32, softmax_scale: f32) -> Result<()> {
        let _phase = GraphPhaseTimer::start(GraphPhase::Replay);
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

pub(crate) struct MetalPagedDecodeReplayPlan<'a> {
    graph: &'a MetalPagedDecodeIcbGraph,
    max_seqlen_k: u32,
    softmax_scale: f32,
}

impl std::fmt::Debug for MetalPagedDecodeReplayPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalPagedDecodeReplayPlan")
            .field("key", &self.graph.replay_state.key)
            .finish_non_exhaustive()
    }
}

impl ReplayPlan for MetalPagedDecodeReplayPlan<'_> {
    fn backend(&self) -> Backend {
        Backend::Metal
    }

    fn key(&self) -> ReplayKey {
        self.graph.replay_state.key.clone()
    }

    fn validate_inputs(&self, inputs: ReplayInputs<'_>) -> Result<(), CaptureError> {
        self.graph
            .replay_state
            .validate(inputs.key, inputs.resources)
    }

    fn replay(&mut self, inputs: ReplayInputs<'_>) -> Result<ReplayOutputs, CaptureError> {
        self.validate_inputs(inputs)?;
        self.graph
            .replay_native(self.max_seqlen_k, self.softmax_scale)
            .map_err(|e| CaptureError::Backend(format!("Metal ICB graph replay: {e}")))?;
        Ok(ReplayOutputs::new(
            inputs.resources.to_vec(),
            self.graph.captured.replay_count(),
        ))
    }

    fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason> {
        self.graph
            .replay_state
            .invalidate_reason(&state.key, &state.inputs)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_paged_decode_replay_state(
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
) -> Result<ReplayState> {
    let (batch, _, q_heads, head_dim) = q.dims4()?;
    let replay_key = ReplayKey::new(
        Backend::Metal,
        "paged_decode_icb",
        vec![batch, max_seqlen_k, page_block_size, q_heads, head_dim],
        Some(q.dtype()),
        batch,
        true,
    );
    let resources = [q, k_pool, v_pool, block_table, seqused_k, out, k, v, slots]
        .into_iter()
        .map(metal_stable_replay_ref)
        .collect();
    Ok(ReplayState::new(replay_key, resources))
}

fn metal_stable_replay_ref(tensor: &kiln_tensor::Tensor) -> ResidentResourceRef {
    ResidentResourceRef::from_tensor(
        tensor,
        Backend::Metal,
        ReplayResourceStability::StableAcrossReplay,
    )
}

pub(crate) fn merge_metal_graph_resources(
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
