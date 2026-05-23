//! kiln-graph-vulkan — Vulkan `CapturedGraph` impl.
//!
//! Phase 5.1 of #1082: scaffold.
//!
//! Phase 5.x extends `kiln_vulkan_kernel::cmd_batch` (the existing
//! secondary-command-buffer batcher) to record + resubmit on a
//! capture/replay boundary aligned with the Phase 5 frozen-allocator
//! lifetime.

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
}
