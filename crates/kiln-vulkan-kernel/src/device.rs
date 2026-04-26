use anyhow::{anyhow, Context, Result};
use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

/// Extract a null-terminated string from the Vulkan device_name field ([i8; 256] in ash 0.37).
fn extract_device_name(name_array: &[i8; 256]) -> String {
    let end = name_array.iter().position(|&c| c == 0).unwrap_or(256);
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(name_array.as_ptr() as *const u8, end) };
    std::str::from_utf8(bytes).map(String::from).unwrap_or_default()
}

/// Vulkan device abstraction for Kiln.
pub struct VulkanDevice {
    #[allow(dead_code)]
    instance: ash::Instance,
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    device: Arc<ash::Device>,
    queue: vk::Queue,
    queue_family_index: u32,
    vendor_id: u32,
    device_name: String,
    device_local_mem_type: u32,
    host_visible_mem_type: u32,
    /// Maximum shared memory per workgroup (from VkPhysicalDeviceLimits).
    /// Used by PR2 to decide whether solve_tri can run without exceeding device limits.
    max_compute_shared_memory_size: vk::DeviceSize,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("device_name", &self.device_name)
            .field("vendor_id", &self.vendor_id)
            .finish()
    }
}

impl VulkanDevice {
    /// Cheap probe: check if Vulkan is available without creating a logical device.
    ///
    /// Creates a minimal Vulkan instance and enumerates physical devices.
    /// Does NOT allocate a logical device or queues (~hundreds of microseconds
    /// vs ~tens of milliseconds for `new()`).
    ///
    /// The instance is explicitly destroyed before returning to avoid leaking
    /// the Vulkan instance handle (ash 0.37 does not auto-destroy on drop).
    pub fn probe() -> bool {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => e,
            Err(_) => return false,
        };

        let app_info = vk::ApplicationInfo::builder()
            .application_name(CStr::from_bytes_with_nul(b"Kiln Probe\0").unwrap())
            .engine_name(CStr::from_bytes_with_nul(b"Kiln\0").unwrap())
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);

        let instance = match unsafe { entry.create_instance(&instance_info, None) } {
            Ok(i) => i,
            Err(_) => return false,
        };

        let devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(d) => d,
            Err(_) => return false,
        };

        let available = !devices.is_empty();
        unsafe { instance.destroy_instance(None); }
        available
    }

    /// Create a new Vulkan device, selecting the best available GPU.
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| anyhow::anyhow!("failed to load Vulkan entry: {}", e))?;

        // Create instance
        let app_info = vk::ApplicationInfo::builder()
            .application_name(CStr::from_bytes_with_nul(b"Kiln Vulkan Backend\0").unwrap())
            .engine_name(CStr::from_bytes_with_nul(b"Kiln\0").unwrap())
            .api_version(vk::make_api_version(0, 1, 2, 0))
            .build();

        let instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info);

        let instance = unsafe {
            entry.create_instance(&instance_info, None)
                .context("failed to create Vulkan instance")?
        };

        // Enumerate physical devices
        let physical_devices = unsafe {
            instance.enumerate_physical_devices()
                .context("failed to enumerate physical devices")?
        };

        if physical_devices.is_empty() {
            return Err(anyhow!("no Vulkan physical devices found"));
        }

        // Select physical device
        let physical_device = Self::select_physical_device(&instance, &physical_devices)?;

        // Get device properties (includes limits for shared-memory budget checks)
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let vendor_id = properties.vendor_id;
        let device_name = extract_device_name(&properties.device_name);
        let max_compute_shared_memory_size = properties.limits.max_compute_shared_memory_size as vk::DeviceSize;

        // Find compute queue family
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let compute_family = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow!("no compute queue family found"))? as u32;

        // Get memory properties and find memory types
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let device_local_mem_type = Self::find_memory_type(&mem_props, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or_else(|| anyhow!("no device-local memory type found"))?;
        let host_visible_mem_type = Self::find_memory_type(
            &mem_props,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).ok_or_else(|| anyhow!("no host-visible memory type found"))?;

        // Create logical device
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(compute_family)
            .queue_priorities(&[1.0])
            .build();
        let queue_infos = vec![queue_info];

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .build();

        let device = unsafe {
            Arc::new(instance.create_device(physical_device, &device_info, None)
                .context("failed to create Vulkan logical device")?)
        };

        let queue = unsafe { device.get_device_queue(compute_family, 0) };

        Ok(Self {
            instance,
            physical_device,
            device,
            queue,
            queue_family_index: compute_family,
            vendor_id,
            device_name,
            device_local_mem_type,
            host_visible_mem_type,
            max_compute_shared_memory_size,
        })
    }

    fn find_memory_type(
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for (i, &type_props) in mem_props.memory_types.iter().enumerate() {
            if type_props.property_flags.contains(properties) {
                return Some(i as u32);
            }
        }
        None
    }

    fn select_physical_device(
        instance: &ash::Instance,
        physical_devices: &[vk::PhysicalDevice],
    ) -> Result<vk::PhysicalDevice> {
        // Respect KILN_VULKAN_DEVICE env var
        if let Ok(dev_str) = std::env::var("KILN_VULKAN_DEVICE") {
            if let Ok(idx) = dev_str.parse::<usize>() {
                if idx < physical_devices.len() {
                    tracing::info!(device_index = idx, "using KILN_VULKAN_DEVICE");
                    return Ok(physical_devices[idx]);
                }
            }
        }

        // Respect GGML_VK_VISIBLE_DEVICES (llama.cpp compatibility)
        if let Ok(visible) = std::env::var("GGML_VK_VISIBLE_DEVICES") {
            let indices: Vec<usize> = visible.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            if let Some(&idx) = indices.first() {
                if idx < physical_devices.len() {
                    tracing::info!(device_index = idx, "using GGML_VK_VISIBLE_DEVICES");
                    return Ok(physical_devices[idx]);
                }
            }
        }

        // Prefer discrete GPU
        for &dev in physical_devices {
            let props = unsafe { instance.get_physical_device_properties(dev) };
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                let name = extract_device_name(&props.device_name);
                tracing::info!(device = %name, "selected discrete GPU");
                return Ok(dev);
            }
        }

        // Fall back to first device
        let props = unsafe { instance.get_physical_device_properties(physical_devices[0]) };
        let name = extract_device_name(&props.device_name);
        tracing::info!(device = %name, "selected first Vulkan device");
        Ok(physical_devices[0])
    }

    /// Get the Vulkan device.
    pub fn device(&self) -> &Arc<ash::Device> {
        &self.device
    }

    /// Get the physical device handle.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Get the compute queue.
    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Get the queue family index.
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Get the device-local memory type index.
    pub fn device_local_mem_type(&self) -> u32 {
        self.device_local_mem_type
    }

    /// Get the host-visible memory type index.
    pub fn host_visible_mem_type(&self) -> u32 {
        self.host_visible_mem_type
    }

    /// Check if this is an AMD GPU.
    pub fn is_amd(&self) -> bool {
        self.vendor_id == 0x1002
    }

    /// Check if this is an Intel GPU.
    pub fn is_intel(&self) -> bool {
        self.vendor_id == 0x8086
    }

    /// Get the GPU vendor string.
    pub fn vendor_string(&self) -> &'static str {
        match self.vendor_id {
            0x1002 => "AMD",
            0x8086 => "Intel",
            0x10de => "NVIDIA",
            _ => "Unknown",
        }
    }

    /// Get the device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get the maximum shared memory available per compute workgroup.
    ///
    /// Used to guard kernels (e.g. solve_tri) whose shared-memory footprint
    /// must fit within the device limit. PR2 will use this to decline dispatch
    /// when the kernel won't fit, falling back to the candle CPU path.
    pub fn max_compute_shared_memory_size(&self) -> vk::DeviceSize {
        self.max_compute_shared_memory_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_device_init_fails_gracefully_without_gpu() {
        // When no Vulkan GPU is available (e.g. CI), new() should return
        // a clear error rather than panicking.
        let result = VulkanDevice::new();
        // On a machine without Vulkan, we expect an error.
        // On a machine with Vulkan, this test runs as a smoke test.
        if result.is_ok() {
            let dev = result.unwrap();
            assert!(!dev.device_name().is_empty(), "device name should not be empty");
        }
    }
}
