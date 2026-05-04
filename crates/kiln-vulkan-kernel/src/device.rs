use anyhow::{Context, Result, anyhow};
use ash::vk;
use std::collections::HashMap;
use std::ffi::CStr;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PipelineKey {
    shader_hash: u64,
    total_bindings: u32,
    push_constant_size: u32,
}

struct CachedComputePipeline {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// Extract a null-terminated string from the Vulkan device_name field ([i8; 256] in ash 0.37).
fn extract_device_name(name_array: &[i8; 256]) -> String {
    let end = name_array.iter().position(|&c| c == 0).unwrap_or(256);
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(name_array.as_ptr() as *const u8, end) };
    std::str::from_utf8(bytes)
        .map(String::from)
        .unwrap_or_default()
}

/// Vulkan device abstraction for Kiln.
pub struct VulkanDevice {
    #[allow(dead_code)]
    entry: ash::Entry,
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
    pipeline_cache: Mutex<HashMap<PipelineKey, CachedComputePipeline>>,
    transient_command_pool: Mutex<vk::CommandPool>,
    transient_descriptor_pool: Mutex<vk::DescriptorPool>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("device_name", &self.device_name)
            .field("vendor_id", &self.vendor_id)
            .field(
                "max_compute_shared_memory_size",
                &self.max_compute_shared_memory_size,
            )
            .field(
                "pipeline_cache_len",
                &self
                    .pipeline_cache
                    .lock()
                    .map(|cache| cache.len())
                    .unwrap_or(0),
            )
            .finish()
    }
}

impl VulkanDevice {
    /// Select an explicit Vulkan physical-device index from environment-style
    /// values without touching Vulkan. `KILN_VULKAN_DEVICE` wins over the
    /// llama.cpp-compatible `GGML_VK_VISIBLE_DEVICES`; the latter may contain a
    /// comma-separated list, so pick the first visible index that exists.
    pub fn explicit_device_index_from_env_values(
        device_count: usize,
        kiln_vulkan_device: Option<&str>,
        ggml_vk_visible_devices: Option<&str>,
    ) -> Option<(usize, &'static str)> {
        if device_count == 0 {
            return None;
        }

        if let Some(dev_str) = kiln_vulkan_device {
            if let Ok(idx) = dev_str.trim().parse::<usize>() {
                if idx < device_count {
                    return Some((idx, "KILN_VULKAN_DEVICE"));
                }
            }
        }

        if let Some(visible) = ggml_vk_visible_devices {
            for idx in visible
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
            {
                if idx < device_count {
                    return Some((idx, "GGML_VK_VISIBLE_DEVICES"));
                }
            }
        }

        None
    }

    /// Cheap probe: check if Vulkan is available without creating a logical device.
    ///
    /// Creates a minimal Vulkan instance and enumerates physical devices.
    /// Does NOT allocate a logical device or queues (~hundreds of microseconds
    /// vs ~tens of milliseconds for `new()`).
    ///
    /// The instance is explicitly destroyed on every path (ash 0.37 does not
    /// auto-destroy on drop) so no Vulkan instance handle is leaked.
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

        let available = unsafe { instance.enumerate_physical_devices() }
            .map(|d| !d.is_empty())
            .unwrap_or(false);
        unsafe {
            instance.destroy_instance(None);
        }
        available
    }

    /// Create a new Vulkan device, selecting the best available GPU.
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| anyhow::anyhow!("failed to load Vulkan entry: {}", e))?;

        // Create instance
        let app_info = vk::ApplicationInfo::builder()
            .application_name(CStr::from_bytes_with_nul(b"Kiln Vulkan Backend\0").unwrap())
            .engine_name(CStr::from_bytes_with_nul(b"Kiln\0").unwrap())
            .api_version(vk::make_api_version(0, 1, 2, 0))
            .build();

        let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);

        let instance = unsafe {
            entry
                .create_instance(&instance_info, None)
                .context("failed to create Vulkan instance")?
        };

        // Enumerate physical devices
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
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
        let max_compute_shared_memory_size =
            properties.limits.max_compute_shared_memory_size as vk::DeviceSize;

        // Find compute queue family
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let compute_family = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow!("no compute queue family found"))?
            as u32;

        // Get memory properties and find memory types
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let device_local_mem_type =
            Self::find_memory_type(&mem_props, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .ok_or_else(|| anyhow!("no device-local memory type found"))?;
        let host_visible_mem_type = Self::find_memory_type(
            &mem_props,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| anyhow!("no host-visible memory type found"))?;

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
            Arc::new(
                instance
                    .create_device(physical_device, &device_info, None)
                    .context("failed to create Vulkan logical device")?,
            )
        };

        let queue = unsafe { device.get_device_queue(compute_family, 0) };

        let transient_command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(compute_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .build(),
                None,
            )
        }
        .context("failed to create Vulkan transient command pool")?;

        let transient_descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .max_sets(4)
                    .pool_sizes(&[vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(64)
                        .build()])
                    .build(),
                None,
            )
        }
        .context("failed to create Vulkan transient descriptor pool")?;

        Ok(Self {
            entry,
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
            pipeline_cache: Mutex::new(HashMap::new()),
            transient_command_pool: Mutex::new(transient_command_pool),
            transient_descriptor_pool: Mutex::new(transient_descriptor_pool),
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
        let kiln_vulkan_device = std::env::var("KILN_VULKAN_DEVICE").ok();
        let ggml_vk_visible_devices = std::env::var("GGML_VK_VISIBLE_DEVICES").ok();
        if let Some((idx, source)) = Self::explicit_device_index_from_env_values(
            physical_devices.len(),
            kiln_vulkan_device.as_deref(),
            ggml_vk_visible_devices.as_deref(),
        ) {
            tracing::info!(device_index = idx, source, "using explicit Vulkan device selection");
            return Ok(physical_devices[idx]);
        }

        if let Some(value) = kiln_vulkan_device.as_deref() {
            tracing::warn!(
                value,
                device_count = physical_devices.len(),
                "ignoring invalid KILN_VULKAN_DEVICE; expected a zero-based Vulkan physical-device index"
            );
        } else if let Some(value) = ggml_vk_visible_devices.as_deref() {
            tracing::warn!(
                value,
                device_count = physical_devices.len(),
                "ignoring GGML_VK_VISIBLE_DEVICES; no listed Vulkan physical-device index is available"
            );
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

    /// Return a cached compute pipeline compatible with the provided shader,
    /// descriptor binding count, and push-constant size.
    ///
    /// Pipeline creation is expensive on RADV and can dominate decode latency if
    /// done per token. Descriptors and command buffers still remain per-dispatch
    /// because they depend on the live buffers, but shader modules, descriptor
    /// set layouts, pipeline layouts, and compute pipelines are stable.
    pub(crate) fn get_or_create_compute_pipeline(
        &self,
        spirv: &[u8],
        total_bindings: usize,
        push_constant_size: u32,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline)> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        spirv.hash(&mut hasher);
        let key = PipelineKey {
            shader_hash: hasher.finish(),
            total_bindings: total_bindings as u32,
            push_constant_size,
        };

        let mut cache = self
            .pipeline_cache
            .lock()
            .map_err(|_| anyhow!("Vulkan pipeline cache mutex poisoned"))?;
        if let Some(cached) = cache.get(&key) {
            return Ok((cached.set_layout, cached.layout, cached.pipeline));
        }

        let desc_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..total_bindings as u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            })
            .collect();
        let set_layout = unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&desc_bindings)
                    .build(),
                None,
            )
        }
        .context("failed to create descriptor set layout")?;

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_constant_size)
            .build();
        let set_layouts = [set_layout];
        let layout = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&[push_constant_range])
                    .build(),
                None,
            )
        }
        .context("failed to create pipeline layout")?;

        let spirv_words: &[u32] = bytemuck::cast_slice(spirv);
        let shader_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::builder()
                    .code(spirv_words)
                    .build(),
                None,
            )
        }
        .context("failed to create shader module")?;

        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
            .build();
        let pipeline = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::builder()
                    .stage(stage_info)
                    .layout(layout)
                    .build()],
                None,
            )
        }
        .map_err(|(errs, _)| {
            if !errs.is_empty() {
                anyhow!("failed to create compute pipeline: {:?}", errs[0])
            } else {
                anyhow!("failed to create compute pipeline")
            }
        })?[0];

        unsafe {
            self.device.destroy_shader_module(shader_module, None);
        }

        cache.insert(
            key,
            CachedComputePipeline {
                set_layout,
                layout,
                pipeline,
            },
        );
        Ok((set_layout, layout, pipeline))
    }

    pub(crate) fn transient_command_pool(&self) -> Result<MutexGuard<'_, vk::CommandPool>> {
        self.transient_command_pool
            .lock()
            .map_err(|_| anyhow!("Vulkan command pool mutex poisoned"))
    }

    pub(crate) fn transient_descriptor_pool(&self) -> Result<MutexGuard<'_, vk::DescriptorPool>> {
        self.transient_descriptor_pool
            .lock()
            .map_err(|_| anyhow!("Vulkan descriptor pool mutex poisoned"))
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        if let Ok(pool) = self.transient_descriptor_pool.lock() {
            unsafe {
                self.device.destroy_descriptor_pool(*pool, None);
            }
        }
        if let Ok(pool) = self.transient_command_pool.lock() {
            unsafe {
                self.device.destroy_command_pool(*pool, None);
            }
        }
        if let Ok(mut cache) = self.pipeline_cache.lock() {
            for (_, cached) in cache.drain() {
                unsafe {
                    self.device.destroy_pipeline(cached.pipeline, None);
                    self.device.destroy_pipeline_layout(cached.layout, None);
                    self.device
                        .destroy_descriptor_set_layout(cached.set_layout, None);
                }
            }
        }
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_vulkan_device_prefers_kiln_env() {
        assert_eq!(
            VulkanDevice::explicit_device_index_from_env_values(4, Some("2"), Some("1,3")),
            Some((2, "KILN_VULKAN_DEVICE"))
        );
    }

    #[test]
    fn explicit_vulkan_device_uses_first_valid_ggml_visible_device() {
        assert_eq!(
            VulkanDevice::explicit_device_index_from_env_values(4, None, Some("99, 3, 1")),
            Some((3, "GGML_VK_VISIBLE_DEVICES"))
        );
    }

    #[test]
    fn explicit_vulkan_device_ignores_invalid_or_missing_values() {
        assert_eq!(
            VulkanDevice::explicit_device_index_from_env_values(2, Some("amd"), Some("4")),
            None
        );
        assert_eq!(
            VulkanDevice::explicit_device_index_from_env_values(0, Some("0"), Some("0")),
            None
        );
    }

    #[test]
    fn test_vulkan_device_init_fails_gracefully_without_gpu() {
        // When no Vulkan GPU is available (e.g. CI), new() should return
        // a clear error rather than panicking.
        let result = VulkanDevice::new();
        // On a machine without Vulkan, we expect an error.
        // On a machine with Vulkan, this test runs as a smoke test.
        if result.is_ok() {
            let dev = result.unwrap();
            assert!(
                !dev.device_name().is_empty(),
                "device name should not be empty"
            );
        }
    }

    #[test]
    fn test_vulkan_device_prewarm_and_drop() {
        let Ok(dev) = VulkanDevice::new() else {
            return;
        };

        crate::kernels::prewarm_builtin_pipelines(&dev).unwrap();
        drop(dev);
    }
}
