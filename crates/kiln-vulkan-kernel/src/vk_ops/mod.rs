//! Operator implementations for `VkTensor`.
//!
//! Each module exposes (a) forward functions returning a `VkTensor` with
//! a `grad_fn` attached and (b) a `*Backward` struct implementing
//! `VkBackwardOp` that knows how to compute input gradients given the
//! output gradient. Match the PyTorch `Function`/`Function.backward`
//! split.
//!
//! Helpers ending in `_no_grad` perform the same forward but without
//! recording autograd; they are used internally by the autograd engine
//! (e.g., accumulating grads with `vk_add_no_grad`).

pub mod attention;
pub mod cast;
pub mod conv1d;
pub mod elementwise;
pub mod embedding;
pub mod flce;
pub mod gdn_chunk_bwd;
pub mod gdn_chunk_prep;
pub mod gdn_chunkwise;
pub mod gdn_gated_rms_norm;
pub mod gdn_gates;
pub mod gdn_state;
pub mod index_select;
pub mod l2norm;
pub mod mask;
pub mod matmul;
pub mod matmul_batched;
pub mod matmul_bf16w;
pub mod mlp;
pub mod narrow;
pub mod permute;
pub mod reduce;
pub mod reverse_cumsum;
pub mod rmsnorm;
pub mod rope;
pub mod shape;
pub mod sigmoid;
pub mod silu;
pub mod softmax;
pub mod solve_tri;

use crate::VulkanDevice;
use anyhow::{Context, Result};
use ash::vk;

const DEFAULT_1D_TILE_ELEMENTS: usize = 1 << 20;

/// Upper bound for long 1D elementwise-style dispatches.
///
/// Some Vulkan drivers are much happier with many moderate dispatches than
/// one enormous dispatch. Tiling only changes how a contiguous output buffer is
/// covered; each element is still computed exactly once by the same shader
/// logic.
pub(crate) fn vk_1d_tile_elements() -> usize {
    std::env::var("KILN_VK_ELEMENTWISE_TILE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_1D_TILE_ELEMENTS)
        .clamp(256, u32::MAX as usize)
}

pub(crate) fn vk_exp_tile_elements() -> usize {
    std::env::var("KILN_VK_EXP_TILE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(vk_1d_tile_elements)
        .clamp(256, u32::MAX as usize)
}

pub(crate) fn for_each_1d_tile(
    total: usize,
    tile_elements: usize,
    mut f: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    anyhow::ensure!(total > 0, "1d tiled dispatch: total must be > 0");
    let tile_elements = tile_elements.clamp(1, u32::MAX as usize);
    let mut offset = 0usize;
    while offset < total {
        let len = (total - offset).min(tile_elements);
        f(offset, len)?;
        offset += len;
    }
    Ok(())
}

/// Internal helper: compile (or load embedded SPIR-V for) a shader by
/// base name and dispatch it with the given buffer handles + push
/// constants on a 1D workgroup grid.
pub(crate) fn dispatch_simple(
    device: &VulkanDevice,
    shader_name: &str,
    handles: &[vk::Buffer],
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<()> {
    let glsl_path = format!(
        "{}/csrc/shaders/{}.comp",
        env!("CARGO_MANIFEST_DIR"),
        shader_name
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(&glsl_path)
        .with_context(|| format!("vk_ops: shader compile/load for {shader_name}"))?;
    crate::kernels::run_compute_pipeline(
        device,
        &spirv,
        handles,
        handles.len(),
        push_constants,
        workgroup_count,
    )
    .with_context(|| format!("vk_ops: dispatch {shader_name}"))
}

/// 2D-grid sibling of `dispatch_simple` for shaders with 2D workgroup
/// layouts (e.g., transpose, matmul tiles).
pub(crate) fn dispatch_simple_2d(
    device: &VulkanDevice,
    shader_name: &str,
    handles: &[vk::Buffer],
    push_constants: &[u32],
    workgroup_count: (u32, u32),
) -> Result<()> {
    let glsl_path = format!(
        "{}/csrc/shaders/{}.comp",
        env!("CARGO_MANIFEST_DIR"),
        shader_name
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(&glsl_path)
        .with_context(|| format!("vk_ops: shader compile/load for {shader_name}"))?;
    crate::kernels::run_compute_pipeline_3d(
        device,
        &spirv,
        handles,
        handles.len(),
        push_constants,
        (workgroup_count.0, workgroup_count.1, 1),
    )
    .with_context(|| format!("vk_ops: 2d dispatch {shader_name}"))
}

/// 3D-grid sibling of `dispatch_simple` for shaders that naturally map
/// workgroups to rows, heads, and batches.
pub(crate) fn dispatch_simple_3d(
    device: &VulkanDevice,
    shader_name: &str,
    handles: &[vk::Buffer],
    push_constants: &[u32],
    workgroup_count: (u32, u32, u32),
) -> Result<()> {
    let glsl_path = format!(
        "{}/csrc/shaders/{}.comp",
        env!("CARGO_MANIFEST_DIR"),
        shader_name
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(&glsl_path)
        .with_context(|| format!("vk_ops: shader compile/load for {shader_name}"))?;
    crate::kernels::run_compute_pipeline_3d(
        device,
        &spirv,
        handles,
        handles.len(),
        push_constants,
        workgroup_count,
    )
    .with_context(|| format!("vk_ops: 3d dispatch {shader_name}"))
}
