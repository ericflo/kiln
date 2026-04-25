use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Shader pipeline cache and dispatcher.
pub struct ShaderPipeline {
    pipelines: HashMap<String, (vk::PipelineLayout, vk::Pipeline)>,
    #[allow(dead_code)]
    device: Arc<ash::Device>,
}

impl std::fmt::Debug for ShaderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderPipeline")
            .field("pipeline_count", &self.pipelines.len())
            .finish()
    }
}

impl ShaderPipeline {
    pub fn new(device: &Arc<ash::Device>) -> Self {
        Self {
            pipelines: HashMap::new(),
            device: Arc::clone(device),
        }
    }

    /// Compile a GLSL shader to SPIR-V using glslc.
    pub fn compile_shader(glsl_path: &str) -> Result<Vec<u8>> {
        let spv_path = glsl_path.replace(".comp", ".spv");

        // Try to load pre-compiled SPIR-V
        if Path::new(&spv_path).exists() {
            return std::fs::read(&spv_path)
                .context(format!("failed to read pre-compiled SPIR-V: {}", spv_path));
        }

        // Compile with glslc
        // Defines match llama.cpp Vulkan defaults for RDNA GPUs
        let output = std::process::Command::new("glslc")
            .arg(glsl_path)
            .arg("-o")
            .arg(&spv_path)
            .arg("-DFLOAT_TYPE=float")
            .arg("-DUSE_BFLOAT16=1")
            .arg("-DUSE_SUBGROUP_ADD=1")
            .arg("-DUSE_SUBGROUP_CLUSTERED=1")
            .output()
            .context("failed to run glslc")?;

        if !output.status.success() {
            anyhow::bail!(
                "glslc failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        std::fs::read(&spv_path)
            .context(format!("failed to read compiled SPIR-V: {}", spv_path))
    }

    /// Create or retrieve a cached compute pipeline.
    pub fn get_or_create(
        &mut self,
        name: &str,
        spirv: &[u8],
        push_constant_size: u32,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
        if let Some((layout, pipeline)) = self.pipelines.get(name) {
            return Ok((*layout, *pipeline));
        }

        // Create shader module
        let spirv_words: &[u32] = bytemuck::cast_slice(spirv);
        let shader_module_info = vk::ShaderModuleCreateInfo::builder()
            .code(spirv_words)
            .build();

        let shader_module = unsafe {
            self.device.create_shader_module(&shader_module_info, None)
                .context(format!("failed to create shader module: {}", name))?
        };

        // Create pipeline layout
        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_constant_size)
            .build();
        let pcr = vec![push_constant_range];

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&pcr)
            .build();

        let layout = unsafe {
            self.device.create_pipeline_layout(&layout_info, None)
                .context(format!("failed to create pipeline layout: {}", name))?
        };

        // Create compute pipeline
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
            .build();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_info)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .build();

        let pipelines = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            ).map_err(|(errs, _)| {
                if !errs.is_empty() {
                    anyhow::anyhow!("failed to create compute pipeline {}: {:?}", name, errs[0])
                } else {
                    anyhow::anyhow!("failed to create compute pipeline {}", name)
                }
            })?
        };
        let pipeline = pipelines[0];

        // Clean up shader module
        unsafe {
            self.device.destroy_shader_module(shader_module, None);
        }

        self.pipelines.insert(name.to_string(), (layout, pipeline));
        Ok((layout, pipeline))
    }

    /// Cleanup all pipelines.
    pub fn cleanup(&mut self) {
        for (_, (layout, pipeline)) in self.pipelines.drain() {
            unsafe {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(layout, None);
            }
        }
    }

    /// Get the number of cached pipelines.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }
}

impl Drop for ShaderPipeline {
    fn drop(&mut self) {
        self.cleanup();
    }
}
