//! Device selection for the Kiln binaries.
//!
//! Preference order: CUDA (if `--features cuda` built and a CUDA device is
//! present) → Metal (if `--features metal` built and running on Apple
//! Silicon) → CPU. Each branch logs which backend was chosen so the
//! startup banner and crash dumps make it obvious.

use anyhow::Result;
use candle_core::Device;

pub fn select_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        tracing::info!("CUDA available — using GPU device 0");
        return Ok(Device::new_cuda(0)?);
    }

    #[cfg(feature = "metal")]
    if candle_core::utils::metal_is_available() {
        tracing::info!("Metal available — using Apple Silicon GPU");
        return Ok(Device::new_metal(0)?);
    }

    tracing::info!("no GPU feature active — using CPU");
    Ok(Device::Cpu)
}
