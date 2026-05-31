//! Device selection for the Kiln binaries.
//!
//! Preference order: CUDA (if `--features cuda` built and a CUDA device is
//! present) → Vulkan (if `--features vulkan` built and a Vulkan device is
//! present) → Metal (if `--features metal` built and running on Apple
//! Silicon) → CPU. Each branch logs which backend was chosen so the
//! startup banner and crash dumps make it obvious.
//!
//! The public API surface is fully kt-typed (`select_device_kt`,
//! `select_device_with_options_kt`). Internally we still materialise a
//! candle device — `kiln_model`'s downstream call paths and a few
//! kiln-server seams still take `&candle_core::Device` while the rest
//! of #1082 lands — but the candle value is constructed via the
//! always-on `kiln_kt_bridge` helpers so this file no longer names
//! `candle_core::*` paths directly in source. (issue #1082, candle
//! removal)

use anyhow::Result;

/// Select the best available device for the loaded backends.
///
/// Returns a `kiln_tensor::Device` so callers don't have to thread a
/// candle device around. Internally constructs the candle device
/// through the `kiln_kt_bridge` helpers and then translates back to kt.
pub fn select_device_kt() -> Result<kiln_tensor::Device> {
    select_device_with_options_kt(false)
}

/// Same as [`select_device_kt`] but lets callers opt into a
/// graph-capturable CUDA stream (event tracking disabled on the inner
/// candle CUDA device).
///
/// Behaves identically to [`select_device_kt`] except that under
/// `--features cuda` the CUDA device is opened with
/// `candle_core::Device::new_cuda_with_stream` + `disable_event_tracking`
/// so the resident `cudaStream_t` can be captured into a CUDA graph (the
/// hot decode path). The two-step setup lives behind the
/// `kiln_kt_bridge::candle_cuda_device_with_stream_no_event_tracking`
/// helper so this file does not name `candle_core::*` paths directly.
pub fn select_device_with_options_kt(cuda_graphs: bool) -> Result<kiln_tensor::Device> {
    #[cfg(not(feature = "cuda"))]
    let _ = cuda_graphs;

    #[cfg(feature = "cuda")]
    if kiln_tensor::cuda_is_available() {
        // #1082: kt-native — no candle device. The kt CUDA-graph capture path
        // (`CudaGraphRunner` + `with_active_cuda_stream`) routes kernel
        // launches / allocs / memcpys onto its OWN capture stream, derived from
        // the kt `primary_cuda_context`. It does NOT need the device opened on a
        // special candle stream with event tracking disabled — and in fact that
        // candle device was always discarded here (only `Device::Cuda(0)`
        // survived `kt_device_from_candle`, since kt `Device` is index-only).
        // So both modes simply return the plain kt CUDA device.
        if cuda_graphs {
            tracing::info!(
                "CUDA available — using GPU device 0 (graph-capturable via the kt capture stream)"
            );
        } else {
            tracing::info!("CUDA available — using GPU device 0");
        }
        return Ok(kiln_tensor::Device::Cuda(0));
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
            return Ok(kiln_tensor::Device::Vulkan(0));
        }
    }

    #[cfg(feature = "metal")]
    if kiln_tensor::metal_is_available() {
        tracing::info!("Metal available — using Apple Silicon GPU");
        return Ok(kiln_tensor::Device::Metal(0));
    }

    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
    tracing::info!("no compiled GPU backend found an available device — using CPU");

    #[cfg(not(any(feature = "cuda", feature = "vulkan", feature = "metal")))]
    tracing::info!("no GPU feature active — using CPU");
    Ok(kiln_tensor::Device::Cpu)
}
