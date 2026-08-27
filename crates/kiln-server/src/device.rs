//! Device selection for the Kiln binaries.
//!
//! Preference order: CUDA (if `--features cuda` built and a CUDA device is
//! present) → Vulkan (if `--features vulkan` built and a Vulkan device is
//! present) → Metal (if `--features metal` built and running on Apple
//! Silicon) → CPU. Each branch logs which backend was chosen so the
//! startup banner and crash dumps make it obvious.
//!
//! The selection is kt-native end-to-end (issue #1082, candle removal):
//! every branch returns a plain `kiln_tensor::Device`, no candle device
//! is constructed, and no `candle_core::*` path is named in this file.
//! The CUDA-graph path needs no special device-open — the capture stream
//! is the kt one (`CudaGraphRunner` + `with_active_cuda_stream`, derived
//! from the kt `primary_cuda_context`).

use anyhow::Result;

/// Select the best available device for the loaded backends.
///
/// Returns a `kiln_tensor::Device` directly — the selection is
/// kt-native end-to-end and constructs no candle device (issue #1082).
pub fn select_device_kt() -> Result<kiln_tensor::Device> {
    select_device_with_options_kt(false)
}

/// Same as [`select_device_kt`] but lets callers opt into the
/// graph-capturable CUDA path.
///
/// Both modes return the plain kt CUDA device. The graph-capturable
/// stream is the kt capture stream (`CudaGraphRunner` +
/// `with_active_cuda_stream`, derived from the kt `primary_cuda_context`),
/// so no special device-open is needed (issue #1082). The `cuda_graphs`
/// flag only selects the startup log line.
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

    // ROCm is probed BEFORE Vulkan: on an AMD GPU the native ROCm/HIP path
    // (hipBLASLt GEMM, HIP kernels, lower overhead) beats Vulkan compute. (R.9)
    #[cfg(feature = "rocm")]
    if kiln_tensor::rocm_is_available() {
        tracing::info!("ROCm available — using AMD GPU device 0 (hipBLASLt + HIP kernels)");
        return Ok(kiln_tensor::Device::Rocm(0));
    }

    #[cfg(feature = "vulkan")]
    {
        // Vulkan: the kt `Device` is index-only (no native handle), so we
        // detect availability ourselves. The Vulkan backend manages its own vk::Device.
        if let Some(device_index) = kiln_model::backend::vulkan::vulkan_selected_device_index()? {
            tracing::info!(
                device_index,
                "Vulkan available — using selected Vulkan physical device"
            );
            // Tell the rest of the process (forward pass, trainer) that
            // Vulkan is the active runtime: CPU-host tensors report as
            // `Device::Cpu` and `Device::Vulkan(idx)` is index-only. Lets
            // `vulkan_active()`-gated guards (e.g.
            // `ProjectionLoadPolicy::for_model_loader_device`, the
            // CPU-arm of `training_precision_policy_for_device_kt`) fire
            // without having to thread a backend handle through every
            // call site.
            kiln_model::backend::mark_vulkan_active();
            return Ok(kiln_tensor::Device::Vulkan(device_index));
        }
    }

    #[cfg(feature = "metal")]
    if kiln_tensor::metal_is_available() {
        tracing::info!("Metal available — using Apple Silicon GPU");
        return Ok(kiln_tensor::Device::Metal(0));
    }

    #[cfg(any(
        feature = "cuda",
        feature = "vulkan",
        feature = "metal",
        feature = "rocm"
    ))]
    tracing::info!("no compiled GPU backend found an available device — using CPU");

    #[cfg(not(any(
        feature = "cuda",
        feature = "vulkan",
        feature = "metal",
        feature = "rocm"
    )))]
    tracing::info!("no GPU feature active — using CPU");
    Ok(kiln_tensor::Device::Cpu)
}
