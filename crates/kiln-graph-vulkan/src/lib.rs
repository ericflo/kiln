//! kiln-graph-vulkan — Vulkan `CapturedGraph` impl.
//!
//! Phase 5.1 of #1082: scaffold. The production Vulkan replay path is
//! still the resident decode and command-batch layer in
//! `crates/kiln-model/src/vk_decode_resident.rs` and
//! `crates/kiln-vulkan-kernel/src/cmd_batch.rs`; this crate is not yet
//! the authoritative replay layer.
//!
//! Phase 5.x should move or wrap `kiln_vulkan_kernel::cmd_batch`
//! behind this crate's object or its successor, while keeping Vulkan's
//! command-batch/resident-plan model instead of forcing CUDA-style
//! capture semantics.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

use std::sync::atomic::{AtomicU64, Ordering};

use kiln_graph::{CaptureError, CapturedGraph};
use kiln_tensor::Backend;

#[derive(Debug)]
pub struct VulkanCapturedGraph {
    scratch_bytes: usize,
    replay_count: AtomicU64,
}

impl VulkanCapturedGraph {
    pub fn new(scratch_bytes: usize) -> Self {
        VulkanCapturedGraph {
            scratch_bytes,
            replay_count: AtomicU64::new(0),
        }
    }
}

impl CapturedGraph for VulkanCapturedGraph {
    fn backend(&self) -> Backend {
        Backend::Vulkan
    }
    fn replay(&self) -> Result<(), CaptureError> {
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
            Backend::Vulkan,
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
            backend: Backend::Vulkan,
            dtype: DType::F32,
            shape: vec![1, 128],
            strides: vec![128, 1],
            start_offset: 0,
            contiguous: true,
            byte_len: 128 * DType::F32.size_in_bytes(),
            replay_stability: ReplayResourceStability::StableAcrossReplay,
        }
    }

    #[test]
    fn scaffold_reports_backend() {
        let g = VulkanCapturedGraph::new(4096);
        assert_eq!(g.backend(), Backend::Vulkan);
        assert_eq!(g.scratch_bytes(), 4096);
    }

    #[test]
    fn scaffold_replay_counts() {
        let g = VulkanCapturedGraph::new(0);
        g.replay().unwrap();
        assert_eq!(g.replay_count(), 1);
    }

    #[test]
    fn scaffold_wraps_shared_replay_plan_contract() {
        let key = replay_key();
        let input = stable_resource();
        let graph = VulkanCapturedGraph::new(4096);
        let mut plan = CapturedGraphReplayPlan::new(graph, key.clone(), vec![input.clone()])
            .expect("Vulkan graph backend should match replay key");

        assert_eq!(ReplayPlan::backend(&plan), Backend::Vulkan);
        ReplayPlan::validate_inputs(&plan, ReplayInputs::new(&key, std::slice::from_ref(&input)))
            .unwrap();
        let outputs = ReplayPlan::replay(
            &mut plan,
            ReplayInputs::new(&key, std::slice::from_ref(&input)),
        )
        .expect("shared replay plan should replay scaffold graph");

        assert_eq!(outputs.replay_count, 1);
        assert_eq!(outputs.resources, vec![input]);
    }
}
