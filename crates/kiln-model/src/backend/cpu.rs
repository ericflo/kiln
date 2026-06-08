//! Portable fallback backend: every kernel method returns `Ok(None)` so
//! the caller falls back to the candle-op composition that runs on any
//! device. Used on CPU, on Metal until Phase 2 adds a real backend, and
//! as a safe default for any future device.

use super::{
    AttentionBackend, BackendIdentity, BackendRuntime, ConvBackend, GdnBackend, OptimizerBackend,
    LinearBackend, PagedKvBackend, ResidencyBackend, SamplingBackend, StartupBackend,
};

#[derive(Debug)]
pub struct CpuBackend {
    /// The kt device this backend was constructed for. Returned by the
    /// `BackendRuntime::device()` trait method. (#1082 DoD-100 step 4: the
    /// formerly-cached candle `device` field was dropped — `new` now takes a
    /// `kiln_tensor::Device` directly and `name()` matches on it.)
    device_kt: kiln_tensor::Device,
}

impl CpuBackend {
    pub fn new(device: kiln_tensor::Device) -> Self {
        Self { device_kt: device }
    }
}

impl BackendIdentity for CpuBackend {
    fn runtime_name(&self) -> &'static str {
        match self.device_kt {
            kiln_tensor::Device::Cpu => "cpu",
            kiln_tensor::Device::Metal(_) => "metal-portable",
            kiln_tensor::Device::Cuda(_) => "cuda-portable",
            kiln_tensor::Device::Vulkan(_) => "vulkan-portable",
            // `kiln_tensor::Device` is #[non_exhaustive]; future device kinds
            // route through the portable CPU fallback backend.
            _ => "portable",
        }
    }

    fn runtime_device(&self) -> kiln_tensor::Device {
        self.device_kt
    }

    fn runtime_as_any(&self) -> &dyn std::any::Any {
        &()
    }
}

impl StartupBackend for CpuBackend {}

impl AttentionBackend for CpuBackend {}

impl GdnBackend for CpuBackend {}

impl ConvBackend for CpuBackend {}

impl LinearBackend for CpuBackend {}

impl ResidencyBackend for CpuBackend {}

impl SamplingBackend for CpuBackend {}

impl OptimizerBackend for CpuBackend {}

impl PagedKvBackend for CpuBackend {}

impl BackendRuntime for CpuBackend {}
