//! Portable fallback backend: every kernel method returns `Ok(None)` so
//! the caller falls back to the candle-op composition that runs on any
//! device. Used on CPU, on Metal until Phase 2 adds a real backend, and
//! as a safe default for any future device.

use candle_core::Device;

use super::BackendRuntime;

#[derive(Debug)]
pub struct CpuBackend {
    device: Device,
}

impl CpuBackend {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl BackendRuntime for CpuBackend {
    fn name(&self) -> &'static str {
        match self.device {
            Device::Cpu => "cpu",
            Device::Metal(_) => "metal-portable",
            Device::Cuda(_) => "cuda-portable",
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
