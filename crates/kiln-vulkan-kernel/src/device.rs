use anyhow::{Context, Result, anyhow};
use ash::vk;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
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
    /// Sum of device-local memory heap sizes reported by the physical device.
    /// Used by memory-bounded kernels to choose conservative defaults without
    /// requiring operator tuning.
    device_local_heap_bytes: u64,
    /// Maximum shared memory per workgroup (from VkPhysicalDeviceLimits).
    /// Used by PR2 to decide whether solve_tri can run without exceeding device limits.
    max_compute_shared_memory_size: vk::DeviceSize,
    /// Maximum dispatch grid extent on each axis (from
    /// VkPhysicalDeviceLimits::maxComputeWorkGroupCount[0..3]). Vulkan
    /// only guarantees ≥ 65535 per axis, but real devices typically
    /// support much more (AMD/Strix Halo ≈ 2^31 - 1). Used by
    /// `dispatch_kernel` and `run_compute_pipeline` to gate oversized
    /// dispatches with a meaningful error rather than letting
    /// vkCmdDispatch fail opaquely.
    max_compute_work_group_count: [u32; 3],
    pipeline_cache: Mutex<HashMap<PipelineKey, CachedComputePipeline>>,
    /// Fast-path cache for `CommandBatch::record_shader` callers.
    /// Keyed by `(shader_path, total_bindings, push_constant_size)` —
    /// avoids re-hashing the SPIR-V bytes on every record. Returns
    /// the same `CachedComputePipeline` slots as `pipeline_cache`.
    path_pipeline_cache: Mutex<HashMap<(&'static str, u32, u32), CachedComputePipeline>>,
    transient_command_pool: Mutex<vk::CommandPool>,
    transient_descriptor_pool: Mutex<vk::DescriptorPool>,
    /// Long-lived pools dedicated to `CommandBatch`. Sized to hold a full
    /// multi-dispatch decode step's worth of descriptor sets and command
    /// buffers so each batch can lock, allocate, submit, and `reset` —
    /// instead of paying `vkCreateCommandPool` + `vkCreateDescriptorPool`
    /// per layer (which dominated decode ITL on NVIDIA drivers, ~5 ms
    /// each × 32 layers = 160 ms wasted per token).
    batch_command_pool: Mutex<vk::CommandPool>,
    batch_descriptor_pool: Mutex<vk::DescriptorPool>,
    /// Persistent descriptor-set cache for `CommandBatch::record_shader`.
    /// Keyed by `(set_layout, [buffer handles])` — once a unique
    /// combination is seen we keep the descriptor set allocated forever
    /// (the `batch_descriptor_pool` is no longer reset on `CommandBatch`
    /// drop, so the sets remain valid). After the first decode token
    /// this cache is fully warm and `record_with_pipeline` is just a
    /// hashmap lookup plus the cmd_* binds.
    descriptor_set_cache:
        Mutex<HashMap<(vk::DescriptorSetLayout, Vec<vk::Buffer>), vk::DescriptorSet>>,
    /// Sticky flag set when any submit/wait observes `VK_ERROR_DEVICE_LOST`.
    /// Once true, subsequent dispatches short-circuit with a clear error
    /// instead of returning cryptic submit failures forever — the underlying
    /// `VkDevice` is unrecoverable per the Vulkan spec, so the only fix is
    /// to restart the kiln server.
    terminally_lost: AtomicBool,
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
            .field("device_local_heap_bytes", &self.device_local_heap_bytes)
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

        // Optional: enable Vulkan validation layers when KILN_VULKAN_VALIDATION
        // is set (truthy values: 1, true, on, yes). Useful for diagnosing
        // OOB descriptor access / shader hangs that trigger driver hard
        // recoveries (e.g. radv/amdgpu "context is lost" — see
        // VK_ERROR_DEVICE_LOST handling in submit_and_wait()).
        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let mut layer_ptrs: Vec<*const i8> = Vec::new();
        if validation_requested() {
            let layers = entry
                .enumerate_instance_layer_properties()
                .context("failed to enumerate Vulkan instance layers")?;
            let available = layers.iter().any(|l| {
                let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
                name == validation_layer.as_c_str()
            });
            if available {
                layer_ptrs.push(validation_layer.as_ptr());
                tracing::info!(
                    layer = "VK_LAYER_KHRONOS_validation",
                    "enabling Vulkan validation layer (KILN_VULKAN_VALIDATION set)"
                );
            } else {
                tracing::warn!(
                    "KILN_VULKAN_VALIDATION set but VK_LAYER_KHRONOS_validation \
                     is not installed; install the Vulkan SDK / validation \
                     layer package and try again"
                );
            }
        }

        let mut instance_info_builder =
            vk::InstanceCreateInfo::builder().application_info(&app_info);
        if !layer_ptrs.is_empty() {
            instance_info_builder = instance_info_builder.enabled_layer_names(&layer_ptrs);
        }
        let instance_info = instance_info_builder;

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
        let max_compute_work_group_count = properties.limits.max_compute_work_group_count;

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
        let device_local_heap_bytes = mem_props
            .memory_heaps
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<vk::DeviceSize>() as u64;
        let device_local_mem_type =
            Self::find_memory_type(&mem_props, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .ok_or_else(|| anyhow!("no device-local memory type found"))?;
        let host_visible_cached_mem_type = Self::find_memory_type(
            &mem_props,
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED,
        );
        let host_visible_mem_type = host_visible_cached_mem_type
            .or_else(|| {
                Self::find_memory_type(
                    &mem_props,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .ok_or_else(|| anyhow!("no host-visible memory type found"))?;
        tracing::info!(
            memory_type = host_visible_mem_type,
            cached = host_visible_cached_mem_type.is_some(),
            "selected Vulkan host-visible staging memory type"
        );

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

        // Batch pools: large enough to record a full decode step
        // (1024 dispatches × 64 storage-buffer bindings each).
        let batch_command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(compute_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .build(),
                None,
            )
        }
        .context("failed to create Vulkan batch command pool")?;
        let batch_descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .max_sets(1024)
                    .pool_sizes(&[vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(64 * 1024)
                        .build()])
                    .build(),
                None,
            )
        }
        .context("failed to create Vulkan batch descriptor pool")?;

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
            device_local_heap_bytes,
            max_compute_shared_memory_size,
            max_compute_work_group_count,
            pipeline_cache: Mutex::new(HashMap::new()),
            path_pipeline_cache: Mutex::new(HashMap::new()),
            transient_command_pool: Mutex::new(transient_command_pool),
            transient_descriptor_pool: Mutex::new(transient_descriptor_pool),
            batch_command_pool: Mutex::new(batch_command_pool),
            batch_descriptor_pool: Mutex::new(batch_descriptor_pool),
            descriptor_set_cache: Mutex::new(HashMap::new()),
            terminally_lost: AtomicBool::new(false),
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
            tracing::info!(
                device_index = idx,
                source,
                "using explicit Vulkan device selection"
            );
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

    /// Sum of all device-local heaps in bytes. Drivers that do not expose a
    /// useful heap size return 0; callers should then fall back to conservative
    /// shape-only heuristics.
    pub fn device_local_heap_bytes(&self) -> u64 {
        self.device_local_heap_bytes
    }

    /// Get the maximum shared memory available per compute workgroup.
    ///
    /// Used to guard kernels (e.g. solve_tri) whose shared-memory footprint
    /// must fit within the device limit. PR2 will use this to decline dispatch
    /// when the kernel won't fit, falling back to the candle CPU path.
    pub fn max_compute_shared_memory_size(&self) -> vk::DeviceSize {
        self.max_compute_shared_memory_size
    }

    /// Per-axis maximum compute dispatch grid extent (from
    /// `VkPhysicalDeviceLimits::maxComputeWorkGroupCount[axis]`).
    /// `axis` is 0, 1, or 2. Vulkan only guarantees ≥ 65535 per axis,
    /// but real devices typically support much more (AMD/Strix Halo
    /// reports ≈ 2^31 - 1).
    pub fn max_compute_work_group_count(&self, axis: usize) -> u32 {
        debug_assert!(axis < 3, "max_compute_work_group_count axis must be 0..3");
        self.max_compute_work_group_count[axis.min(2)]
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

    /// Path-keyed variant of `get_or_create_compute_pipeline`. Avoids
    /// re-hashing the SPIR-V bytes on every call from the hot decode
    /// path (`CommandBatch::record_shader` runs this ~450× per token).
    /// First-call falls through to `get_or_create_compute_pipeline`
    /// and caches the result by `(path, total_bindings, push_size)`.
    pub(crate) fn get_compute_pipeline_by_path(
        &self,
        path: &'static str,
        total_bindings: usize,
        push_constant_size: u32,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline)> {
        let key = (path, total_bindings as u32, push_constant_size);
        {
            let cache = self
                .path_pipeline_cache
                .lock()
                .map_err(|_| anyhow!("Vulkan path pipeline cache mutex poisoned"))?;
            if let Some(c) = cache.get(&key) {
                return Ok((c.set_layout, c.layout, c.pipeline));
            }
        }
        // First-call: compile (or load embedded SPIR-V) and create the
        // pipeline through the normal cache, then memoize by path.
        let spirv = crate::pipeline::ShaderPipeline::compile_shader(path)?;
        let (set_layout, layout, pipeline) =
            self.get_or_create_compute_pipeline(&spirv, total_bindings, push_constant_size)?;
        let mut cache = self
            .path_pipeline_cache
            .lock()
            .map_err(|_| anyhow!("Vulkan path pipeline cache mutex poisoned"))?;
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
        self.check_alive()?;
        self.transient_command_pool
            .lock()
            .map_err(|_| anyhow!("Vulkan command pool mutex poisoned"))
    }

    pub(crate) fn transient_descriptor_pool(&self) -> Result<MutexGuard<'_, vk::DescriptorPool>> {
        self.check_alive()?;
        self.transient_descriptor_pool
            .lock()
            .map_err(|_| anyhow!("Vulkan descriptor pool mutex poisoned"))
    }

    pub(crate) fn batch_command_pool(&self) -> Result<MutexGuard<'_, vk::CommandPool>> {
        self.check_alive()?;
        self.batch_command_pool
            .lock()
            .map_err(|_| anyhow!("Vulkan batch command pool mutex poisoned"))
    }

    pub(crate) fn batch_descriptor_pool(&self) -> Result<MutexGuard<'_, vk::DescriptorPool>> {
        self.check_alive()?;
        self.batch_descriptor_pool
            .lock()
            .map_err(|_| anyhow!("Vulkan batch descriptor pool mutex poisoned"))
    }

    /// Fast-path descriptor-set lookup keyed by `(set_layout, handles)`.
    /// On first call per unique combination, allocates a descriptor set
    /// from `pool`, writes the storage-buffer bindings, and caches it.
    /// Subsequent calls return the cached set without any Vulkan API
    /// call — the descriptor pool is never reset, so cached sets stay
    /// valid for the lifetime of the device.
    pub(crate) fn get_or_alloc_descriptor_set(
        &self,
        set_layout: vk::DescriptorSetLayout,
        pool: vk::DescriptorPool,
        handles: &[vk::Buffer],
    ) -> Result<vk::DescriptorSet> {
        let key = (set_layout, handles.to_vec());
        {
            let cache = self
                .descriptor_set_cache
                .lock()
                .map_err(|_| anyhow!("Vulkan descriptor set cache mutex poisoned"))?;
            if let Some(s) = cache.get(&key) {
                return Ok(*s);
            }
        }
        let device = &self.device;
        let set_layouts = [set_layout];
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
        }
        .context("get_or_alloc_descriptor_set: allocate")?[0];
        let buf_infos: Vec<vk::DescriptorBufferInfo> = handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
                    .build()
            })
            .collect();
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
        let mut cache = self
            .descriptor_set_cache
            .lock()
            .map_err(|_| anyhow!("Vulkan descriptor set cache mutex poisoned"))?;
        cache.insert(key, descriptor_set);
        Ok(descriptor_set)
    }

    /// True once any submit/wait has observed `VK_ERROR_DEVICE_LOST`.
    /// Subsequent dispatches will fail fast with a clear error rather than
    /// retry a permanently-dead device.
    pub fn is_terminally_lost(&self) -> bool {
        self.terminally_lost.load(Ordering::SeqCst)
    }

    /// Mark the device as terminally lost. The first transition logs an
    /// error so the operator sees the event in the server log; subsequent
    /// calls are no-ops. Public so non-helper submit sites in this crate
    /// (or downstream) can flag a device-lost after observing it directly.
    pub fn mark_terminally_lost(&self) {
        if !self.terminally_lost.swap(true, Ordering::SeqCst) {
            tracing::error!(
                device = %self.device_name,
                vendor = self.vendor_string(),
                "vulkan device terminally lost (VK_ERROR_DEVICE_LOST). \
                 The VkDevice is unrecoverable per the Vulkan spec; restart \
                 the kiln server to recover. Subsequent inference requests \
                 will return an error until the server is restarted."
            );
        }
    }

    /// Short-circuit error returned by helpers when the device is already
    /// terminally lost. Centralized so the message stays consistent.
    pub fn check_alive(&self) -> Result<()> {
        if self.is_terminally_lost() {
            anyhow::bail!(
                "vulkan device terminally lost (VK_ERROR_DEVICE_LOST observed earlier). \
                 Restart the kiln server to recover. \
                 To diagnose the original GPU fault, set KILN_VULKAN_VALIDATION=1 \
                 and reproduce — validation layers will surface OOB descriptor \
                 access or shader timeouts before the driver hard-recovers."
            );
        }
        Ok(())
    }

    /// Submit a single command buffer to the compute queue and wait for
    /// completion, with `VK_ERROR_DEVICE_LOST` detection. On device-lost
    /// from either the submit or the wait, set the sticky flag and return
    /// a structured error; the caller must propagate so subsequent
    /// dispatches short-circuit via `check_alive()`.
    ///
    /// `label` is interpolated into the error message and identifies the
    /// originating dispatch (e.g. "causal_conv1d_prefill cached-weight").
    pub fn submit_and_wait(&self, cmd: vk::CommandBuffer, label: &str) -> Result<()> {
        self.check_alive()?;
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&cmds).build();
        let submit_res = unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], vk::Fence::null())
        };
        match submit_res {
            Ok(()) => {}
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.mark_terminally_lost();
                anyhow::bail!(self.terminally_lost_message(label, "queue_submit"));
            }
            Err(e) => {
                return Err(anyhow!("vulkan queue_submit failed ({label}): {:?}", e));
            }
        }
        let wait_res = unsafe { self.device.queue_wait_idle(self.queue) };
        match wait_res {
            Ok(()) => {}
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.mark_terminally_lost();
                anyhow::bail!(self.terminally_lost_message(label, "queue_wait_idle"));
            }
            Err(e) => {
                return Err(anyhow!("vulkan queue_wait_idle failed ({label}): {:?}", e));
            }
        }
        Ok(())
    }

    fn terminally_lost_message(&self, label: &str, op: &str) -> String {
        format!(
            "vulkan device terminally lost during {op} ({label}): \
             VK_ERROR_DEVICE_LOST. The VkDevice is unrecoverable per the \
             Vulkan spec; restart the kiln server to recover. Set \
             KILN_VULKAN_VALIDATION=1 to enable validation layers and \
             capture the originating fault on the next reproduction."
        )
    }
}

fn validation_requested() -> bool {
    match std::env::var("KILN_VULKAN_VALIDATION") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "" | "0" | "false" | "off" | "no")
        }
        Err(_) => false,
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Release any pooled buffers tied to this logical device
        // BEFORE destroying its descriptor/command pools — otherwise
        // the global buffer pool's recycler could hand out a buffer
        // whose owning device no longer has a usable descriptor pool.
        crate::buffer_pool::pool_drop_for_device(self.device.handle());
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
        if let Ok(pool) = self.batch_descriptor_pool.lock() {
            unsafe {
                self.device.destroy_descriptor_pool(*pool, None);
            }
        }
        if let Ok(pool) = self.batch_command_pool.lock() {
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

    #[test]
    fn test_terminally_lost_flag_short_circuits_dispatch() {
        // Skips when the host has no Vulkan GPU (e.g. CI without a runner
        // image that exposes one). The flag itself is a pure AtomicBool, so
        // the behavior we want to lock down — that mark_terminally_lost()
        // makes check_alive() and the transient pool accessors return an
        // error mentioning "terminally lost" — is verifiable on any host
        // that can construct a VulkanDevice.
        let Ok(dev) = VulkanDevice::new() else {
            return;
        };
        assert!(!dev.is_terminally_lost(), "fresh device should be alive");
        assert!(dev.check_alive().is_ok());
        dev.mark_terminally_lost();
        assert!(dev.is_terminally_lost());
        let err = dev.check_alive().unwrap_err().to_string();
        assert!(
            err.contains("terminally lost"),
            "expected check_alive error to mention 'terminally lost', got: {err}"
        );
        // transient_command_pool / transient_descriptor_pool must also
        // short-circuit, otherwise dispatches that go straight through them
        // would still try to submit to a dead device.
        assert!(dev.transient_command_pool().is_err());
        assert!(dev.transient_descriptor_pool().is_err());
        // Marking again must be idempotent (no panic, flag stays true).
        dev.mark_terminally_lost();
        assert!(dev.is_terminally_lost());
    }

    #[test]
    fn test_validation_requested_env_parsing() {
        // Pure parser test, no Vulkan involvement.
        let cases = [
            ("", false),
            ("0", false),
            ("false", false),
            ("FALSE", false),
            ("off", false),
            ("no", false),
            ("1", true),
            ("true", true),
            ("yes", true),
            ("on", true),
        ];
        for (raw, expected) in cases {
            // SAFETY: tests in the same module run serially under the
            // default cargo-test runner; use a fixed env var name here so
            // we don't collide with the real KILN_VULKAN_VALIDATION at
            // runtime.
            unsafe { std::env::set_var("KILN_VULKAN_VALIDATION", raw) };
            assert_eq!(
                validation_requested(),
                expected,
                "validation_requested({raw:?}) expected {expected}"
            );
        }
        unsafe { std::env::remove_var("KILN_VULKAN_VALIDATION") };
        assert!(!validation_requested());
    }
}
