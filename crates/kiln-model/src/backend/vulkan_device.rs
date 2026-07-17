//! Vulkan device probing and pipeline warmup helpers.
//!
//! The backend facade keeps the stable `backend::vulkan::*` public paths, while
//! this module owns the explicit logical-device construction and one-shot
//! diagnostic probes.

use anyhow::Result;

use std::sync::{Arc, OnceLock};

pub(super) fn new_backend_device() -> Option<Arc<kiln_vulkan_kernel::VulkanDevice>> {
    match kiln_vulkan_kernel::VulkanDevice::new() {
        Ok(dev) => {
            let prewarm_start = std::time::Instant::now();
            match kiln_vulkan_kernel::kernels::prewarm_builtin_pipelines(&dev) {
                Ok(()) => tracing::info!(
                    elapsed_ms = prewarm_start.elapsed().as_millis() as u64,
                    "Vulkan compute pipelines prewarmed"
                ),
                Err(e) => tracing::warn!(
                    error = %e,
                    "Vulkan pipeline prewarm failed; falling back to lazy pipeline creation"
                ),
            }
            tracing::info!(
                vendor = dev.vendor_string(),
                device = dev.device_name(),
                physical_device_index = dev.physical_device_index(),
                "Vulkan device initialized"
            );
            Some(Arc::new(dev))
        }
        Err(e) => {
            tracing::warn!(error = %e, "Vulkan device initialization failed, falling back to CPU");
            None
        }
    }
}

/// Check if Vulkan is available on this system.
///
/// Uses a cheap probe (instance + physical-device enumeration only) cached
/// with `OnceLock` to avoid repeated checks.
pub fn vulkan_is_available() -> bool {
    static VULKAN_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *VULKAN_AVAILABLE.get_or_init(kiln_vulkan_kernel::VulkanDevice::probe)
}

/// Resolve the physical device selected by immutable Vulkan startup policy.
///
/// Unlike [`vulkan_is_available`], this preserves configuration and validation
/// errors so startup cannot silently fall back to a different backend.
pub fn vulkan_selected_device_index() -> Result<Option<usize>> {
    kiln_vulkan_kernel::VulkanDevice::probe_selected_physical_device_index()
}

/// Return the selected Vulkan device name for diagnostics and benchmark output.
pub fn vulkan_device_name() -> Option<String> {
    static VULKAN_DEVICE_NAME: OnceLock<Option<String>> = OnceLock::new();
    VULKAN_DEVICE_NAME
        .get_or_init(|| {
            kiln_vulkan_kernel::VulkanDevice::new()
                .ok()
                .map(|dev| dev.device_name().to_string())
        })
        .clone()
}

/// Precompile Vulkan custom kernels.
///
/// This verifies that the validated built-in SPIR-V modules load correctly and
/// that compute pipelines can be created. `VulkanBackend::new` warms the real
/// backend device; this standalone helper is only for background verification.
pub fn precompile_custom_kernels() -> Result<()> {
    let vk_device = match kiln_vulkan_kernel::VulkanDevice::new() {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    kiln_vulkan_kernel::kernels::prewarm_builtin_pipelines(&vk_device)?;
    tracing::info!("Vulkan shader and pipeline verification complete");
    Ok(())
}
