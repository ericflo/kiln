//! kiln-graph-metal — Metal `CapturedGraph` impl.
//!
//! Phase 5.1 of #1082: scaffold plus a reusable ICB replay object.
//! The production Metal replay orchestration still lives in
//! `crates/kiln-model/src/metal_graph.rs`; this crate is not yet the
//! authoritative replay layer.
//!
//! Phase 5.x should move or wrap the model-level runner behind this
//! crate's ICB object or its successor. ICBs are Metal's "graph"
//! equivalent: pre-encoded compute pipeline state plus buffer-binding
//! records that can be replayed on a `MTLComputeCommandEncoder` with
//! low CPU overhead.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

use std::sync::atomic::{AtomicU64, Ordering};

use kiln_graph::{CaptureError, CapturedGraph};
use kiln_tensor::Backend;
#[cfg(feature = "metal")]
use kiln_tensor::metal_types::{Buffer, IndirectCommandBuffer, MTLResourceUsage, MetalCompanion};

#[derive(Debug)]
pub struct MetalCapturedGraph {
    scratch_bytes: usize,
    replay_count: AtomicU64,
    #[cfg(feature = "metal")]
    replay: Option<MetalReplay>,
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct MetalReplay {
    companion: MetalCompanion,
    indirect_command_buffer: IndirectCommandBuffer,
    command_count: usize,
    resources: Vec<MetalGraphResource>,
}

#[cfg(feature = "metal")]
#[derive(Clone, Debug)]
pub struct MetalGraphResource {
    buffer: Buffer,
    usage: MTLResourceUsage,
}

#[cfg(feature = "metal")]
impl MetalGraphResource {
    pub fn new(buffer: Buffer, usage: MTLResourceUsage) -> Self {
        Self { buffer, usage }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn usage(&self) -> MTLResourceUsage {
        self.usage
    }
}

impl MetalCapturedGraph {
    pub fn new(scratch_bytes: usize) -> Self {
        MetalCapturedGraph {
            scratch_bytes,
            replay_count: AtomicU64::new(0),
            #[cfg(feature = "metal")]
            replay: None,
        }
    }

    #[cfg(feature = "metal")]
    pub fn from_indirect_commands(
        companion: MetalCompanion,
        indirect_command_buffer: IndirectCommandBuffer,
        command_count: usize,
        scratch_bytes: usize,
    ) -> Result<Self, CaptureError> {
        Self::from_indirect_commands_with_resources(
            companion,
            indirect_command_buffer,
            command_count,
            scratch_bytes,
            Vec::new(),
        )
    }

    #[cfg(feature = "metal")]
    pub fn from_indirect_commands_with_resources(
        companion: MetalCompanion,
        indirect_command_buffer: IndirectCommandBuffer,
        command_count: usize,
        scratch_bytes: usize,
        resources: Vec<MetalGraphResource>,
    ) -> Result<Self, CaptureError> {
        if command_count == 0 {
            return Err(CaptureError::NotCaptured);
        }
        if command_count > indirect_command_buffer.max_command_count() {
            return Err(CaptureError::Backend(format!(
                "kiln-graph-metal: command_count {command_count} exceeds ICB capacity {}",
                indirect_command_buffer.max_command_count()
            )));
        }
        Ok(MetalCapturedGraph {
            scratch_bytes,
            replay_count: AtomicU64::new(0),
            replay: Some(MetalReplay {
                companion,
                indirect_command_buffer,
                command_count,
                resources,
            }),
        })
    }

    #[cfg(feature = "metal")]
    pub fn is_captured(&self) -> bool {
        self.replay.is_some()
    }

    #[cfg(not(feature = "metal"))]
    pub fn is_captured(&self) -> bool {
        false
    }

    #[cfg(feature = "metal")]
    pub fn command_count(&self) -> Option<usize> {
        self.replay.as_ref().map(|r| r.command_count)
    }

    #[cfg(not(feature = "metal"))]
    pub fn command_count(&self) -> Option<usize> {
        None
    }

    #[cfg(feature = "metal")]
    pub fn wait_until_completed(&self) -> Result<(), CaptureError> {
        if let Some(replay) = &self.replay {
            replay.companion.wait_until_completed()?;
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    pub fn replay(&self) -> Result<(), CaptureError> {
        <Self as CapturedGraph>::replay(self)
    }

    pub fn replay_count(&self) -> u64 {
        <Self as CapturedGraph>::replay_count(self)
    }
}

impl CapturedGraph for MetalCapturedGraph {
    fn backend(&self) -> Backend {
        Backend::Metal
    }
    fn replay(&self) -> Result<(), CaptureError> {
        #[cfg(feature = "metal")]
        if let Some(replay) = &self.replay {
            let encoder = replay.companion.command_encoder()?;
            for resource in &replay.resources {
                encoder.use_resource(resource.buffer(), resource.usage());
            }
            encoder.execute_commands_in_buffer(
                &replay.indirect_command_buffer,
                0,
                replay.command_count,
            );
            drop(encoder);
        }
        self.replay_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn replay_count(&self) -> u64 {
        self.replay_count.load(Ordering::Relaxed)
    }
    fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_graph::{
        CapturedGraphReplayPlan, ReplayInputs, ReplayKey, ReplayPlan, ReplayResourceStability,
        ResidentResourceRef,
    };
    use kiln_tensor::{DType, TensorId};

    fn replay_key() -> ReplayKey {
        ReplayKey::new(
            Backend::Metal,
            "decode",
            vec![1, 128],
            Some(DType::F32),
            1,
            true,
        )
    }

    fn stable_resource() -> ResidentResourceRef {
        ResidentResourceRef {
            tensor_id: Some(TensorId::next()),
            backend: Backend::Metal,
            dtype: DType::F32,
            shape: vec![1, 128],
            byte_len: 128 * DType::F32.size_in_bytes(),
            replay_stability: ReplayResourceStability::StableAcrossReplay,
        }
    }

    #[test]
    fn scaffold_reports_backend() {
        let g = MetalCapturedGraph::new(2048);
        assert_eq!(g.backend(), Backend::Metal);
        assert_eq!(g.scratch_bytes(), 2048);
    }

    #[test]
    fn scaffold_replay_counts() {
        let g = MetalCapturedGraph::new(0);
        for _ in 0..5 {
            g.replay().unwrap();
        }
        assert_eq!(g.replay_count(), 5);
    }

    #[test]
    fn scaffold_wraps_shared_replay_plan_contract() {
        let key = replay_key();
        let input = stable_resource();
        let graph = MetalCapturedGraph::new(2048);
        let mut plan = CapturedGraphReplayPlan::new(graph, key.clone(), vec![input.clone()])
            .expect("Metal graph backend should match replay key");

        assert_eq!(ReplayPlan::backend(&plan), Backend::Metal);
        ReplayPlan::validate_inputs(&plan, &[input.clone()]).unwrap();
        let outputs = ReplayPlan::replay(&mut plan, ReplayInputs::new(&key, &[input.clone()]))
            .expect("shared replay plan should replay scaffold graph");

        assert_eq!(outputs.replay_count, 1);
        assert_eq!(outputs.resources, vec![input]);
    }
}
