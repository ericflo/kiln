//! kiln-graph-cuda — CUDA `CapturedGraph` impl.
//!
//! Phase 5.1 of #1082: scaffold.
//!
//! Phase 5.x lifts the existing `KILN_CUDA_GRAPHS=true` path in
//! `crates/kiln-model/src/forward.rs` (which wires
//! `cudarc::driver::CudaGraph` + `CudaGraphExec` against the
//! production decode loop) into this crate's
//! [`CudaCapturedGraph`].
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
}
