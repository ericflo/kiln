//! Portable fallback backend: every kernel method returns `Ok(None)` so
//! the caller falls back to the candle-op composition that runs on any
//! device. Used on CPU, on Metal until Phase 2 adds a real backend, and
//! as a safe default for any future device.

use super::BackendRuntime;

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

impl BackendRuntime for CpuBackend {
    fn name(&self) -> &'static str {
        match self.device_kt {
            kiln_tensor::Device::Cpu => "cpu",
            kiln_tensor::Device::Metal(_) => "metal-portable",
            kiln_tensor::Device::Cuda(_) => "cuda-portable",
            kiln_tensor::Device::Vulkan(_) => "vulkan-portable",
        }
    }

    fn device(&self) -> kiln_tensor::Device {
        self.device_kt
    }
}
