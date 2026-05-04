//! Device selection for the Kiln binaries.
//!
//! Preference order: CUDA (if `--features cuda` built and a CUDA device is
//! present) → Vulkan (if `--features vulkan` built and a Vulkan device is
//! present) → Metal (if `--features metal` built and running on Apple
//! Silicon) → CPU. Each branch logs which backend was chosen so the
//! startup banner and crash dumps make it obvious.

use anyhow::Result;
use candle_core::Device;

pub fn select_device() -> Result<Device> {
    select_device_with_options(false)
}

pub fn select_device_with_options(cuda_graphs: bool) -> Result<Device> {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        if cuda_graphs {
            tracing::info!("CUDA available — using GPU device 0 with graph-capturable stream");
            let device = Device::new_cuda_with_stream(0)?;
            if let Device::Cuda(cuda_device) = &device {
                unsafe { cuda_device.disable_event_tracking() };
            }
            return Ok(device);
        }
        tracing::info!("CUDA available — using GPU device 0");
        return Ok(Device::new_cuda(0)?);
    }

    #[cfg(feature = "vulkan")]
    {
        // Vulkan: candle-core has no native Vulkan device, so we detect
        // availability ourselves. The Vulkan backend manages its own vk::Device.
        if kiln_model::backend::vulkan::vulkan_is_available() {
            tracing::info!("Vulkan available — using Vulkan GPU (AMD/Intel)");
            return Ok(Device::Cpu); // Vulkan backend manages its own device
        }
    }

    #[cfg(feature = "metal")]
    if candle_core::utils::metal_is_available() {
        tracing::info!("Metal available — using Apple Silicon GPU");
        return Ok(Device::new_metal(0)?);
    }

    tracing::info!("no GPU feature active — using CPU");
    Ok(Device::Cpu)
}
