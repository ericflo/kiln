//! kiln-graph-cuda — CUDA `CapturedGraph` impl.
//!
//! Phase 5.1 of #1082: scaffold. The production CUDA decode graph
//! runner is still `crates/kiln-model/src/cuda_graph.rs`; this crate
//! is not yet the authoritative replay layer.
//!
//! Phase 5.x should move or wrap the existing `KILN_CUDA_GRAPHS=true`
//! path (which wires `cudarc::driver::CudaGraph` + `CudaGraphExec`
//! against the production decode loop) behind this crate's
//! [`CudaCapturedGraph`] or its successor.
//!
//! # Today
//!
//! The scaffold wires:
//!
//! - `CudaCapturedGraph::new(scratch_bytes)` — constructor that takes
//!   the pre-warmed scratch-pool footprint reported back to the Phase
//!   5 `bench-results/graph-family-vram.csv`.
//! - `CapturedGraph::backend()` — returns `Backend::Cuda`.
//! - `CapturedGraph::replay()` — returns
//!   `CaptureError::NotCaptured("kiln-graph-cuda: scaffold")` until
//!   the real cudarc binding lands.
//! - `CapturedGraph::replay_count()` / `scratch_bytes()` — bookkeeping.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

use std::sync::atomic::{AtomicU64, Ordering};

use kiln_graph::{CaptureError, CapturedGraph};
use kiln_tensor::Backend;

/// CUDA-backed `CapturedGraph` scaffold.
#[derive(Debug)]
pub struct CudaCapturedGraph {
    scratch_bytes: usize,
    replay_count: AtomicU64,
}

impl CudaCapturedGraph {
    /// Construct a scaffold instance reporting the given scratch-pool
    /// byte footprint.
    pub fn new(scratch_bytes: usize) -> Self {
        CudaCapturedGraph {
            scratch_bytes,
            replay_count: AtomicU64::new(0),
        }
    }
}

impl CapturedGraph for CudaCapturedGraph {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }
    fn replay(&self) -> Result<(), CaptureError> {
        // Phase 5.x replaces this with the real cudarc CudaGraphExec
        // launch. Until then, replay is a no-op (counted) — callers
        // get the bookkeeping but no GPU work happens.
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
            Backend::Cuda,
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
            backend: Backend::Cuda,
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
        let g = CudaCapturedGraph::new(1024);
        assert_eq!(g.backend(), Backend::Cuda);
        assert_eq!(g.scratch_bytes(), 1024);
    }

    #[test]
    fn scaffold_replay_counts() {
        let g = CudaCapturedGraph::new(0);
        assert_eq!(g.replay_count(), 0);
        g.replay().unwrap();
        g.replay().unwrap();
        g.replay().unwrap();
        assert_eq!(g.replay_count(), 3);
    }

    #[test]
    fn scaffold_wraps_shared_replay_plan_contract() {
        let key = replay_key();
        let input = stable_resource();
        let graph = CudaCapturedGraph::new(1024);
        let mut plan = CapturedGraphReplayPlan::new(graph, key.clone(), vec![input.clone()])
            .expect("CUDA graph backend should match replay key");

        assert_eq!(ReplayPlan::backend(&plan), Backend::Cuda);
        ReplayPlan::validate_inputs(&plan, ReplayInputs::new(&key, &[input.clone()])).unwrap();
        let outputs = ReplayPlan::replay(&mut plan, ReplayInputs::new(&key, &[input.clone()]))
            .expect("shared replay plan should replay scaffold graph");

        assert_eq!(outputs.replay_count, 1);
        assert_eq!(outputs.resources, vec![input]);
    }
}
