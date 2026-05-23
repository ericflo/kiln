//! kiln-graph-metal — Metal `CapturedGraph` impl.
//!
//! Phase 5.1 of #1082: scaffold.
//!
//! Phase 5.x wraps `metal::IndirectCommandBuffer` (ICB) for the
//! production replay path. ICBs are Metal's "graph" equivalent —
//! pre-encoded compute pipeline state + buffer-binding records that
//! can be replayed on a `MTLComputeCommandEncoder` with low CPU
//! overhead.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

use std::sync::atomic::{AtomicU64, Ordering};

use kiln_graph::{CaptureError, CapturedGraph};
use kiln_tensor::Backend;

#[derive(Debug)]
pub struct MetalCapturedGraph {
    scratch_bytes: usize,
    replay_count: AtomicU64,
}

impl MetalCapturedGraph {
    pub fn new(scratch_bytes: usize) -> Self {
        MetalCapturedGraph {
            scratch_bytes,
            replay_count: AtomicU64::new(0),
        }
    }
}

impl CapturedGraph for MetalCapturedGraph {
    fn backend(&self) -> Backend {
        Backend::Metal
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
}
