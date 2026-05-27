//! Portable fallback backend: every kernel method returns `Ok(None)` so
//! the caller falls back to the candle-op composition that runs on any
//! device. Used on CPU, on Metal until Phase 2 adds a real backend, and
//! as a safe default for any future device.

use candle_core::Device;

use super::BackendRuntime;

#[derive(Debug)]
pub struct CpuBackend {
    /// Original candle device the backend was constructed for. Retained
    /// for kernel trait methods that still take `candle_core::Tensor`
    /// parameters and may need to compare against the device.
    device: Device,
    /// `kiln_tensor::Device` form of the same device. Returned by the
    /// `BackendRuntime::device()` trait method. Cached at construction so
    /// the hot accessor does not bridge on every call. (#1082)
    device_kt: kiln_tensor::Device,
}

impl CpuBackend {
    pub fn new(device: Device) -> Self {
        let device_kt = kiln_kt_bridge::kt_device_from_candle(&device);
        Self { device, device_kt }
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

    fn device(&self) -> kiln_tensor::Device {
        self.device_kt
    }
}
