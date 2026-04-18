//! Portable "fallback" backend.
//!
//! Implements [`BackendRuntime`] for any candle device (CPU, Metal, etc.)
//! by inheriting every kernel method's default `Ok(None)` return. The caller
//! (`forward.rs`) handles `None` by running the portable candle-op
//! composition every device can execute.
//!
//! Phase 2 introduces a real `MetalBackend` that replaces these defaults
//! for Metal devices with candle's fused SDPA and (later) custom MSL
//! kernels.

use candle_core::Device;

use super::BackendRuntime;

#[derive(Debug)]
pub struct CpuBackend {
    device: Device,
    name: &'static str,
}

impl CpuBackend {
    pub fn new(device: Device) -> Self {
        let name = match device {
            Device::Cpu => "cpu",
            Device::Metal(_) => "metal-portable",
            Device::Cuda(_) => "cuda-portable",
        };
        Self { device, name }
    }
}

impl BackendRuntime for CpuBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
