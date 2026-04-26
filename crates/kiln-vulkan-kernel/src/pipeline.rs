use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

// Include the build-time generated SPIR-V modules
include!(concat!(env!("OUT_DIR"), "/vulkan_spirv.rs"));

/// Map shader base names to their embedded SPIR-V module constants.
const SHADER_SPIRVS: &[(&str, &[u8])] = &[
    ("gdn_gates", SPIR_V_GDN_GATES),
    ("gdn_gated_rms_norm", SPIR_V_GDN_GATED_RMS_NORM),
    ("causal_conv1d", SPIR_V_CAUSAL_CONV1D),
    ("solve_tri", SPIR_V_SOLVE_TRI),
    ("gdn_recurrent_prefill", SPIR_V_GDN_RECURRENT_PREFILL),
    ("gdn_chunk_prep", SPIR_V_GDN_CHUNK_PREP),
    ("gdn_full_chunk_forward", SPIR_V_GDN_FULL_CHUNK_FORWARD),
    ("gdn_chunk_scan", SPIR_V_GDN_CHUNK_SCAN),
    ("flash_attn", SPIR_V_FLASH_ATTN),
];

// Re-export the spirv_modules for use in SHADER_SPIRVS
use spirv_modules::*;

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

    /// Get embedded SPIR-V for a shader by base name.
    /// Falls back to runtime `glslc` compilation if the embedded module is empty
    /// (e.g., when glslc was not available at build time).
    pub fn compile_shader(glsl_path: &str) -> Result<Vec<u8>> {
        // Extract base name from path (e.g., "gdn_gates" from ".../gdn_gates.comp")
        let stem = std::path::Path::new(glsl_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("failed to extract shader name from path"))?;

        // Check embedded SPIR-V first
        for (name, spv) in SHADER_SPIRVS {
            if *name == stem && !spv.is_empty() {
                tracing::trace!(shader = %name, "using embedded SPIR-V");
                return Ok(spv.to_vec());
            }
        }

        // Fall back to runtime glslc compilation
        tracing::trace!(shader = %stem, "compiling shader at runtime via glslc");
        Self::compile_shader_runtime(glsl_path)
    }

    /// Runtime glslc compilation (fallback when embedded SPIR-V is unavailable).
    fn compile_shader_runtime(glsl_path: &str) -> Result<Vec<u8>> {
        let spv_path = glsl_path.replace(".comp", ".spv");

        // Try to load pre-compiled SPIR-V
        if std::path::Path::new(&spv_path).exists() {
            return std::fs::read(&spv_path)
                .context(format!("failed to read pre-compiled SPIR-V: {}", spv_path));
        }

        // Check that glslc is available
        let glslc_status = std::process::Command::new("glslc")
            .arg("--help")
            .output();
        if glslc_status.is_err() {
            anyhow::bail!(
                "Vulkan SPIR-V '{}' was not compiled at build time and glslc is not on PATH. \
                 Install glslc (part of the SPIR-V Tools package) and rebuild kiln-vulkan-kernel, \
                 or pre-compile the shader to {} and place it alongside the .comp source.",
                glsl_path,
                spv_path
            );
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
                "glslc failed to compile '{}': {} \
                 (Install glslc and rebuild kiln-vulkan-kernel, or place a pre-compiled .spv file)",
                glsl_path,
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
