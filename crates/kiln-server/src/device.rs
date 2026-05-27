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
    #[cfg(not(feature = "cuda"))]
    let _ = cuda_graphs;

    #[cfg(feature = "cuda")]
    if kiln_tensor::cuda_is_available() {
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
            // Tell the rest of the process (forward.rs, trainer.rs) that
            // Vulkan is active even though the candle device reports as
            // Device::Cpu. Lets `projection_original_drop_enabled_for_device`
            // and similar guards fire without having to thread a backend
            // handle through every call site.
            kiln_model::backend::mark_vulkan_active();
            return Ok(Device::Cpu); // Vulkan backend manages its own device
        }
    }

    #[cfg(feature = "metal")]
    if kiln_tensor::metal_is_available() {
        tracing::info!("Metal available — using Apple Silicon GPU");
        return Ok(Device::new_metal(0)?);
    }

    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
    tracing::info!("no compiled GPU backend found an available device — using CPU");

    #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "metal")))]
    tracing::info!("no GPU feature active — using CPU");
    Ok(Device::Cpu)
}

/// kt-typed parallel of [`select_device`].
///
/// Same selection logic as [`select_device`], but returns a
/// `kiln_tensor::Device` so callers that have migrated off candle's
/// `Device` enum don't have to bridge at every call site. Part of the
/// staged migration in #1082 — the existing candle-typed entries above
/// stay live until every caller switches over.
pub fn select_device_kt() -> Result<kiln_tensor::Device> {
    select_device_with_options_kt(false)
}

/// kt-typed parallel of [`select_device_with_options`].
///
/// Delegates to the candle-typed [`select_device_with_options`] so the
/// CUDA-stream / event-tracking side-effects on the candle device still
/// fire for any downstream code that still expects them, then maps the
/// result to `kiln_tensor::Device` via the always-on
/// `kt_device_from_candle` bridge helper.
///
/// On the Vulkan path, [`select_device_with_options`] returns a candle
/// `Device::Cpu` by convention (it marks Vulkan active in the process and
/// the Vulkan backend manages its own `vk::Device`). The bridge helper
/// would map that to `kt::Device::Cpu`, so we override here under the
/// `vulkan` feature and check `kiln_model::backend::vulkan_active()` to
/// surface a `kt::Device::Vulkan(0)` to kt-typed callers. Without this
/// override, kt-typed callers couldn't distinguish "CPU because no GPU"
/// from "CPU as Vulkan placeholder".
pub fn select_device_with_options_kt(cuda_graphs: bool) -> Result<kiln_tensor::Device> {
    let candle_device = select_device_with_options(cuda_graphs)?;

    #[cfg(feature = "vulkan")]
    if kiln_model::backend::vulkan_active() {
        // select_device_with_options returned Device::Cpu as the Vulkan
        // placeholder. Tell kt-typed callers the truth.
        return Ok(kiln_tensor::Device::Vulkan(0));
    }

    Ok(kiln_kt_bridge::kt_device_from_candle(&candle_device))
}
