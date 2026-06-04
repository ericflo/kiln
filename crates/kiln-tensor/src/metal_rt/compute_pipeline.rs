//! Vendored from candle-metal-kernels 0.10.2 `src/metal/compute_pipeline.rs`
//! (MIT/Apache-2.0). objc2 wrapper over `MTLComputePipelineState`. (#1082)

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> ComputePipeline {
        ComputePipeline { raw }
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }

    pub fn supports_indirect_command_buffers(&self) -> bool {
        self.raw.supportIndirectCommandBuffers()
    }
}

impl AsRef<ProtocolObject<dyn MTLComputePipelineState>> for ComputePipeline {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}
