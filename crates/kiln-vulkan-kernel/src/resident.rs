//! Vulkan-resident decode dispatchers.
//!
//! Gate (a) of `docs/vk_resident_decode_plan.md`: every dispatcher
//! the decode forward pass calls per layer has a `_resident` variant
//! that takes a pre-uploaded `&VulkanBuffer` for `x` and writes into
//! a caller-provided `&VulkanBuffer` for `y`. No
//! `extract_tensor_bytes` on input, no `create_tensor_from_data`
//! on output, no staging buffer per call. The compute path is the
//! same `run_compute_pipeline` primitive the existing dispatchers
//! drive — only the host-side boundary is gone.
//!
//! Each resident dispatcher has a matching parity test under
//! `mod tests` that runs the legacy `Tensor`-shaped dispatcher and
//! the resident variant against the same inputs and asserts
//! bit-identical outputs. The resident path uses **the same shader,
//! the same push constants, and the same workgroup count** as the
//! existing path, so the parity is structural rather than
//! statistical — any divergence indicates a wiring bug.

use anyhow::{Context, Result};
use ash::vk;

use crate::kernels::run_compute_pipeline;
use crate::pipeline::ShaderPipeline;
use crate::{VulkanBuffer, VulkanDevice};

use crate::kernels::linear_decode_bf16w_rows4_enabled;

/// Selection helper shared between the f32-weights and packed-bf16
/// linear-decode resident variants. Returns the shader source path,
/// the push-constant slice, and the workgroup count for the given
/// `(batch, hidden, out_dim, packed_bf16_weights)` tuple.
fn linear_decode_shader_plan(
    batch: usize,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> (&'static str, Vec<u32>, u32) {
    if batch == 1 {
        let glsl_path = if packed_bf16_weights {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode.comp"
            )
        };
        let push = vec![hidden as u32, out_dim as u32];
        let wg = out_dim.div_ceil(16) as u32;
        (glsl_path, push, wg)
    } else {
        let rows4 = packed_bf16_weights && batch >= 32 && linear_decode_bf16w_rows4_enabled();
        let glsl_path = if packed_bf16_weights {
            if rows4 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_rows4_bf16w.comp"
                )
            } else {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_bf16w.comp"
                )
            }
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched.comp"
            )
        };
        let push = vec![hidden as u32, out_dim as u32, batch as u32];
        let wg = if rows4 {
            (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32
        } else {
            (batch * out_dim.div_ceil(32)) as u32
        };
        (glsl_path, push, wg)
    }
}

/// Resident-form `dispatch_linear_decode_cached_bf16_weights`.
///
/// `x` is `[batch, 1, hidden]` f32 already on device; `weight_t` is
/// pre-uploaded packed-bf16 `[hidden, out_dim]`. The kernel writes
/// `[batch, 1, out_dim]` f32 into `out` — caller must size `out` to
/// at least `batch * out_dim * 4` bytes.
///
/// Matches `dispatch_linear_decode_cached_bf16_weights` (which goes
/// through `dispatch_linear_decode_cached_impl` with
/// `packed_bf16_weights = true`) bit-for-bit: same shader, same push
/// constants, same workgroup count. The cost difference is purely
/// the absence of staging buffer allocations, extract /
/// create_tensor boundaries, and host transfers.
pub fn dispatch_linear_decode_cached_bf16_weights_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    weight_t: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<()> {
    dispatch_linear_decode_cached_resident_impl(
        vk_device, x, weight_t, out, batch, hidden, out_dim, true,
    )
}

/// Resident-form `dispatch_linear_decode_cached` (f32 weights).
///
/// Same shape contract as the bf16 variant; weight buffer is raw f32.
pub fn dispatch_linear_decode_cached_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    weight_t: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<()> {
    dispatch_linear_decode_cached_resident_impl(
        vk_device, x, weight_t, out, batch, hidden, out_dim, false,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_linear_decode_cached_resident_impl(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    weight_t: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> Result<()> {
    let need_in = (batch * hidden * 4) as u64;
    let need_out = (batch * out_dim * 4) as u64;
    anyhow::ensure!(
        x.size() >= need_in,
        "linear_decode_resident: x buffer has {} bytes, needs at least {} for \
         batch={batch} hidden={hidden}",
        x.size(),
        need_in,
    );
    anyhow::ensure!(
        out.size() >= need_out,
        "linear_decode_resident: out buffer has {} bytes, needs at least {} for \
         batch={batch} out_dim={out_dim}",
        out.size(),
        need_out,
    );

    let handles: [vk::Buffer; 3] = [x.handle(), weight_t.handle(), out.handle()];
    let (glsl_path, push_constants, workgroups) =
        linear_decode_shader_plan(batch, hidden, out_dim, packed_bf16_weights);
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        workgroups,
    )
    .context("linear_decode_resident kernel failed")
}

/// Resident-form `dispatch_qwen_rmsnorm_forward`.
///
/// `x` is `[..., hidden]` f32 on device; `weight` is `[hidden]` f32
/// on device. Writes the same shape into `out`. The kernel is
/// `qwen_rmsnorm_forward.comp` — same shader as the non-resident
/// dispatcher.
pub fn dispatch_qwen_rmsnorm_forward_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    weight: &VulkanBuffer,
    out: &VulkanBuffer,
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let need_x = (rows * hidden * 4) as u64;
    anyhow::ensure!(
        x.size() >= need_x,
        "qwen_rmsnorm_resident: x buffer has {} bytes, needs at least {}",
        x.size(),
        need_x,
    );
    anyhow::ensure!(
        out.size() >= need_x,
        "qwen_rmsnorm_resident: out buffer has {} bytes, needs at least {}",
        out.size(),
        need_x,
    );
    anyhow::ensure!(
        weight.size() >= (hidden * 4) as u64,
        "qwen_rmsnorm_resident: weight buffer has {} bytes, needs at least {}",
        weight.size(),
        hidden * 4,
    );

    let handles: [vk::Buffer; 3] = [x.handle(), weight.handle(), out.handle()];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_forward.comp"
    );
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    // Push constants match the non-resident dispatcher byte-for-byte:
    // [rows, hidden, eps_bits].
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];
    let workgroups = rows as u32;
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        workgroups,
    )
    .context("qwen_rmsnorm_resident kernel failed")
}

/// Selection helper shared between the three SwiGLU MLP resident
/// variants. Returns
/// `(gate_up_glsl, linear_glsl, gate_up_push, gate_up_workgroups,
///   linear_push, linear_workgroups)`. The selection logic mirrors
/// `dispatch_mlp_decode_cached_impl` in `kernels.rs` so the resident
/// path picks the same shader at the same batch size.
#[allow(clippy::type_complexity)]
fn mlp_decode_shader_plan(
    batch: usize,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
    gate_up_bf16_weights: bool,
    down_bf16_weights: bool,
) -> (&'static str, &'static str, Vec<u32>, u32, Vec<u32>, u32) {
    use crate::kernels::{
        mlp_bf16_down_rows4_enabled, mlp_bf16_gate_up_rows4_enabled, mlp_bf16_rows8_enabled,
        mlp_f32_down_rows4_enabled, use_prefill_row_pair_matmul,
    };
    let gate_up_rows2 = !gate_up_bf16_weights && use_prefill_row_pair_matmul(batch);
    let rows8_path = gate_up_bf16_weights
        && down_bf16_weights
        && batch >= 64
        && mlp_bf16_rows8_enabled();
    let down_bf16_rows4 = down_bf16_weights
        && gate_up_bf16_weights
        && batch >= 32
        && !rows8_path
        && mlp_bf16_down_rows4_enabled();
    let gate_up_rows4 = gate_up_bf16_weights
        && batch >= 8
        && !rows8_path
        && mlp_bf16_gate_up_rows4_enabled();
    let down_rows4 = gate_up_bf16_weights
        && !down_bf16_weights
        && batch >= 8
        && mlp_f32_down_rows4_enabled();
    let down_rows2 = !down_bf16_weights && !down_rows4 && use_prefill_row_pair_matmul(batch);

    let gate_up_glsl = if gate_up_bf16_weights {
        if batch == 1 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_bf16w.comp"
            )
        } else if rows8_path {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_rows8_bf16w.comp"
            )
        } else if gate_up_rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_bf16w.comp"
            )
        }
    } else if batch == 1 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode.comp"
        )
    } else if gate_up_rows2 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched_rows2.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched.comp"
        )
    };
    let mut gate_up_push = vec![hidden as u32, intermediate as u32];
    if batch > 1 {
        gate_up_push.push(batch as u32);
    }
    let gate_up_workgroups = if batch == 1 {
        intermediate.div_ceil(64) as u32
    } else if rows8_path {
        (batch.div_ceil(8) * intermediate.div_ceil(64)) as u32
    } else if gate_up_rows4 {
        (batch.div_ceil(4) * intermediate.div_ceil(64)) as u32
    } else if gate_up_rows2 {
        (batch.div_ceil(2) * intermediate.div_ceil(64)) as u32
    } else {
        (batch * intermediate.div_ceil(128)) as u32
    };

    let linear_glsl = if down_bf16_weights {
        if batch == 1 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_bf16w.comp"
            )
        } else if rows8_path {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_rows8_bf16w.comp"
            )
        } else if down_bf16_rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_bf16w.comp"
            )
        }
    } else if batch == 1 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode.comp"
        )
    } else if down_rows4 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched_rows4.comp"
        )
    } else if down_rows2 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched_rows2.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched.comp"
        )
    };
    let mut linear_push = vec![intermediate as u32, out_dim as u32];
    if batch > 1 {
        linear_push.push(batch as u32);
    }
    let linear_workgroups = if batch == 1 {
        out_dim.div_ceil(16) as u32
    } else if rows8_path {
        (batch.div_ceil(8) * out_dim.div_ceil(32)) as u32
    } else if down_rows4 || down_bf16_rows4 {
        (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32
    } else if down_rows2 {
        (batch.div_ceil(2) * out_dim.div_ceil(32)) as u32
    } else {
        (batch * out_dim.div_ceil(32)) as u32
    };
    (
        gate_up_glsl,
        linear_glsl,
        gate_up_push,
        gate_up_workgroups,
        linear_push,
        linear_workgroups,
    )
}

/// Resident-form fused SwiGLU MLP block:
/// `silu(x @ gate_t) * (x @ up_t) @ down_t`.
///
/// `x` is `[batch, 1, hidden]` f32 on device; `gate_weight_t` and
/// `up_weight_t` are `[hidden, intermediate]` (bf16 packed when
/// `gate_up_bf16_weights = true`, else f32); `down_weight_t` is
/// `[intermediate, out_dim]` (bf16 packed when
/// `down_bf16_weights = true`, else f32). `scratch` is a caller-
/// provided intermediate buffer with at least `batch * intermediate * 4`
/// bytes (typically a `DecodeResidentPool::acquire()` slot). Writes
/// `[batch, 1, out_dim]` f32 into `out`.
///
/// Same shader-selection logic as `dispatch_mlp_decode_cached_impl`,
/// so this is bit-identical with the equivalent legacy dispatcher.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlp_decode_cached_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    scratch: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
    gate_up_bf16_weights: bool,
    down_bf16_weights: bool,
) -> Result<()> {
    let need_in = (batch * hidden * 4) as u64;
    let need_mid = (batch * intermediate * 4) as u64;
    let need_out = (batch * out_dim * 4) as u64;
    anyhow::ensure!(
        x.size() >= need_in,
        "mlp_decode_resident: x buffer {} bytes < required {need_in}",
        x.size()
    );
    anyhow::ensure!(
        scratch.size() >= need_mid,
        "mlp_decode_resident: scratch buffer {} bytes < required {need_mid}",
        scratch.size()
    );
    anyhow::ensure!(
        out.size() >= need_out,
        "mlp_decode_resident: out buffer {} bytes < required {need_out}",
        out.size()
    );

    let (gate_up_glsl, linear_glsl, gate_up_push, gate_up_workgroups, linear_push, linear_workgroups) =
        mlp_decode_shader_plan(
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
        );

    let gate_up_spirv = ShaderPipeline::compile_shader(gate_up_glsl)?;
    let gate_up_handles: [vk::Buffer; 4] = [
        x.handle(),
        gate_weight_t.handle(),
        up_weight_t.handle(),
        scratch.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &gate_up_spirv,
        &gate_up_handles,
        gate_up_handles.len(),
        &gate_up_push,
        gate_up_workgroups,
    )
    .context("mlp_decode_resident: gate/up kernel failed")?;

    let linear_spirv = ShaderPipeline::compile_shader(linear_glsl)?;
    let linear_handles: [vk::Buffer; 3] = [scratch.handle(), down_weight_t.handle(), out.handle()];
    run_compute_pipeline(
        vk_device,
        &linear_spirv,
        &linear_handles,
        linear_handles.len(),
        &linear_push,
        linear_workgroups,
    )
    .context("mlp_decode_resident: down kernel failed")
}

/// Convenience wrapper for the all-bf16 MLP. Mirrors
/// `dispatch_mlp_decode_cached_bf16_weights`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlp_decode_cached_bf16_weights_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    scratch: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<()> {
    dispatch_mlp_decode_cached_resident(
        vk_device,
        x,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        scratch,
        out,
        batch,
        hidden,
        intermediate,
        out_dim,
        true,
        true,
    )
}

/// Resident-form fused full-attention single-token Q/K/V projection
/// (`batch == 1`). Writes the contiguous `[1, q_dim + k_dim + v_dim]`
/// combined output into `qkv_out`. The caller indexes Q at offset 0,
/// K at offset `q_dim * 4`, V at offset `(q_dim + k_dim) * 4` — the
/// same layout the non-resident dispatcher reads back and then
/// splits into three Tensors before returning.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    qkv_out: &VulkanBuffer,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    bf16_weights: bool,
) -> Result<()> {
    let total_out = q_dim + k_dim + v_dim;
    let need_in = (hidden * 4) as u64;
    let need_out = (total_out * 4) as u64;
    anyhow::ensure!(
        x.size() >= need_in,
        "full_attn_qkv_resident: x buffer {} bytes < required {need_in}",
        x.size()
    );
    anyhow::ensure!(
        qkv_out.size() >= need_out,
        "full_attn_qkv_resident: qkv_out buffer {} bytes < required {need_out}",
        qkv_out.size()
    );

    let glsl_path = if bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode.comp"
        )
    };
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [
        hidden as u32,
        q_dim as u32,
        k_dim as u32,
        v_dim as u32,
        total_out as u32,
    ];
    let handles: [vk::Buffer; 5] = [
        x.handle(),
        q_weight_t.handle(),
        k_weight_t.handle(),
        v_weight_t.handle(),
        qkv_out.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        total_out.div_ceil(16) as u32,
    )
    .context("full_attn_qkv_decode_resident kernel failed")
}

/// Resident-form batched full-attention QKV projection. Writes the
/// row-major `[batch, q_dim + k_dim + v_dim]` combined output (each
/// batch row stores `q | k | v`).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached_batched_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    qkv_out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    bf16_weights: bool,
) -> Result<()> {
    anyhow::ensure!(
        batch > 0,
        "full_attn_qkv_batched_resident: batch must be > 0"
    );
    let total_out = q_dim + k_dim + v_dim;
    let need_in = (batch * hidden * 4) as u64;
    let need_out = (batch * total_out * 4) as u64;
    anyhow::ensure!(
        x.size() >= need_in,
        "full_attn_qkv_batched_resident: x buffer {} bytes < required {need_in}",
        x.size()
    );
    anyhow::ensure!(
        qkv_out.size() >= need_out,
        "full_attn_qkv_batched_resident: qkv_out buffer {} bytes < required {need_out}",
        qkv_out.size()
    );

    let rows4 = bf16_weights
        && batch >= 16
        && crate::kernels::full_attn_qkv_bf16w_rows4_enabled();
    let glsl_path = if bf16_weights {
        if rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/full_attn_qkv_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/full_attn_qkv_decode_batched_bf16w.comp"
            )
        }
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode_batched.comp"
        )
    };
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 6] = [
        hidden as u32,
        q_dim as u32,
        k_dim as u32,
        v_dim as u32,
        total_out as u32,
        batch as u32,
    ];
    let handles: [vk::Buffer; 5] = [
        x.handle(),
        q_weight_t.handle(),
        k_weight_t.handle(),
        v_weight_t.handle(),
        qkv_out.handle(),
    ];
    let col_groups = total_out.div_ceil(16);
    let row_groups = if rows4 { batch.div_ceil(4) } else { batch };
    let total_groups = row_groups * col_groups;
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        total_groups as u32,
    )
    .context("full_attn_qkv_decode_batched_resident kernel failed")
}

/// Convenience wrapper for the bf16 gate/up + f32 down MLP. Mirrors
/// `dispatch_mlp_decode_cached_bf16_gate_up_f32_down`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlp_decode_cached_bf16_gate_up_f32_down_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    scratch: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<()> {
    dispatch_mlp_decode_cached_resident(
        vk_device,
        x,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        scratch,
        out,
        batch,
        hidden,
        intermediate,
        out_dim,
        true,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{
        dispatch_linear_decode_cached, dispatch_linear_decode_cached_bf16_weights,
        dispatch_qwen_rmsnorm_forward, extract_tensor_bytes,
        upload_tensor_bf16_packed_buffer, upload_tensor_f32_buffer,
    };
    use candle_core::{Device, Tensor};
    use half::bf16;
    use std::sync::Arc;

    fn try_device() -> Option<Arc<VulkanDevice>> {
        VulkanDevice::new().ok().map(Arc::new)
    }

    /// Read back an f32 device-local buffer into a Vec<f32>.
    fn read_back_f32(dev: &VulkanDevice, buf: &VulkanBuffer) -> Vec<f32> {
        let bytes = VulkanBuffer::read_back(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            dev.queue_family_index(),
            buf,
        )
        .expect("read_back");
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    }

    fn make_x_f32(batch: usize, hidden: usize) -> Tensor {
        let n = batch * hidden;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.025)
            .collect();
        Tensor::from_vec(data, (batch, 1, hidden), &Device::Cpu).unwrap()
    }

    fn make_bf16_weight(rows: usize, cols: usize) -> Tensor {
        let n = rows * cols;
        let data: Vec<bf16> = (0..n)
            .map(|i| bf16::from_f32(((i % 31) as f32 - 15.0) * 0.01))
            .collect();
        Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
    }

    fn upload_x(dev: &VulkanDevice, x: &Tensor) -> VulkanBuffer {
        // Same bytes the legacy dispatcher would extract.
        let bytes = extract_tensor_bytes(x).unwrap().0;
        let buf = VulkanBuffer::create_device_local(
            dev.device(),
            dev.device_local_mem_type(),
            bytes.len() as u64,
        )
        .unwrap();
        let pool = dev.transient_command_pool().unwrap();
        VulkanBuffer::upload_data_with_command_pool(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            *pool,
            &buf,
            &bytes,
        )
        .unwrap();
        buf
    }

    fn alloc_out(dev: &VulkanDevice, bytes: u64) -> VulkanBuffer {
        VulkanBuffer::create_device_local(dev.device(), dev.device_local_mem_type(), bytes).unwrap()
    }

    #[test]
    fn linear_decode_bf16w_resident_matches_nonresident_b1() {
        let Some(dev) = try_device() else { return };
        let batch = 1;
        let hidden = 128;
        let out_dim = 64;
        let x = make_x_f32(batch, hidden);
        let w = make_bf16_weight(hidden, out_dim);
        let w_buf = upload_tensor_bf16_packed_buffer(&dev, &w).unwrap();

        let baseline = dispatch_linear_decode_cached_bf16_weights(
            &dev, &x, &w_buf, batch, hidden, out_dim,
        )
        .unwrap();
        let baseline = baseline
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let x_buf = upload_x(&dev, &x);
        let out_buf = alloc_out(&dev, (batch * out_dim * 4) as u64);
        dispatch_linear_decode_cached_bf16_weights_resident(
            &dev, &x_buf, &w_buf, &out_buf, batch, hidden, out_dim,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);

        assert_eq!(baseline.len(), resident.len());
        for (i, (b, r)) in baseline.iter().zip(resident.iter()).enumerate() {
            assert_eq!(b.to_bits(), r.to_bits(), "row {i}: baseline {b} vs resident {r}");
        }
    }

    #[test]
    fn linear_decode_bf16w_resident_matches_nonresident_b4() {
        let Some(dev) = try_device() else { return };
        let batch = 4;
        let hidden = 96;
        let out_dim = 80;
        let x = make_x_f32(batch, hidden);
        let w = make_bf16_weight(hidden, out_dim);
        let w_buf = upload_tensor_bf16_packed_buffer(&dev, &w).unwrap();

        let baseline = dispatch_linear_decode_cached_bf16_weights(
            &dev, &x, &w_buf, batch, hidden, out_dim,
        )
        .unwrap();
        let baseline = baseline
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let x_buf = upload_x(&dev, &x);
        let out_buf = alloc_out(&dev, (batch * out_dim * 4) as u64);
        dispatch_linear_decode_cached_bf16_weights_resident(
            &dev, &x_buf, &w_buf, &out_buf, batch, hidden, out_dim,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);

        for (i, (b, r)) in baseline.iter().zip(resident.iter()).enumerate() {
            assert_eq!(b.to_bits(), r.to_bits(), "row {i}: baseline {b} vs resident {r}");
        }
    }

    #[test]
    fn linear_decode_f32w_resident_matches_nonresident_b1() {
        let Some(dev) = try_device() else { return };
        let batch = 1;
        let hidden = 128;
        let out_dim = 64;
        let x = make_x_f32(batch, hidden);
        let w_f32 = Tensor::from_vec(
            (0..hidden * out_dim)
                .map(|i| ((i % 19) as f32 - 9.0) * 0.02)
                .collect::<Vec<_>>(),
            (hidden, out_dim),
            &Device::Cpu,
        )
        .unwrap();
        let w_buf = upload_tensor_f32_buffer(&dev, &w_f32).unwrap();

        let baseline =
            dispatch_linear_decode_cached(&dev, &x, &w_buf, batch, hidden, out_dim).unwrap();
        let baseline = baseline
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let x_buf = upload_x(&dev, &x);
        let out_buf = alloc_out(&dev, (batch * out_dim * 4) as u64);
        dispatch_linear_decode_cached_resident(
            &dev, &x_buf, &w_buf, &out_buf, batch, hidden, out_dim,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);

        for (i, (b, r)) in baseline.iter().zip(resident.iter()).enumerate() {
            assert_eq!(b.to_bits(), r.to_bits(), "row {i}: baseline {b} vs resident {r}");
        }
    }

    #[test]
    fn mlp_decode_bf16w_resident_matches_nonresident_b1() {
        use crate::kernels::dispatch_mlp_decode_cached_bf16_weights;
        let Some(dev) = try_device() else { return };
        let batch = 1;
        let hidden = 64;
        let intermediate = 128;
        let out_dim = 96;
        let x = make_x_f32(batch, hidden);
        let gate = make_bf16_weight(hidden, intermediate);
        let up = make_bf16_weight(hidden, intermediate);
        let down = make_bf16_weight(intermediate, out_dim);
        let g_buf = upload_tensor_bf16_packed_buffer(&dev, &gate).unwrap();
        let u_buf = upload_tensor_bf16_packed_buffer(&dev, &up).unwrap();
        let d_buf = upload_tensor_bf16_packed_buffer(&dev, &down).unwrap();

        let baseline = dispatch_mlp_decode_cached_bf16_weights(
            &dev,
            &x,
            &g_buf,
            &u_buf,
            &d_buf,
            hidden,
            intermediate,
            out_dim,
        )
        .unwrap();
        let baseline = baseline.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_buf = upload_x(&dev, &x);
        let scratch = alloc_out(&dev, (batch * intermediate * 4) as u64);
        let out_buf = alloc_out(&dev, (batch * out_dim * 4) as u64);
        dispatch_mlp_decode_cached_bf16_weights_resident(
            &dev,
            &x_buf,
            &g_buf,
            &u_buf,
            &d_buf,
            &scratch,
            &out_buf,
            batch,
            hidden,
            intermediate,
            out_dim,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);

        assert_eq!(baseline.len(), resident.len());
        for (i, (b, r)) in baseline.iter().zip(resident.iter()).enumerate() {
            assert_eq!(b.to_bits(), r.to_bits(), "row {i}: baseline {b} vs resident {r}");
        }
    }

    #[test]
    fn mlp_decode_bf16w_resident_matches_nonresident_b4() {
        use crate::kernels::dispatch_mlp_decode_cached_bf16_weights;
        let Some(dev) = try_device() else { return };
        let batch = 4;
        let hidden = 64;
        let intermediate = 96;
        let out_dim = 80;
        let x = make_x_f32(batch, hidden);
        let gate = make_bf16_weight(hidden, intermediate);
        let up = make_bf16_weight(hidden, intermediate);
        let down = make_bf16_weight(intermediate, out_dim);
        let g_buf = upload_tensor_bf16_packed_buffer(&dev, &gate).unwrap();
        let u_buf = upload_tensor_bf16_packed_buffer(&dev, &up).unwrap();
        let d_buf = upload_tensor_bf16_packed_buffer(&dev, &down).unwrap();

        let baseline = dispatch_mlp_decode_cached_bf16_weights(
            &dev,
            &x,
            &g_buf,
            &u_buf,
            &d_buf,
            hidden,
            intermediate,
            out_dim,
        )
        .unwrap();
        let baseline = baseline.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_buf = upload_x(&dev, &x);
        let scratch = alloc_out(&dev, (batch * intermediate * 4) as u64);
        let out_buf = alloc_out(&dev, (batch * out_dim * 4) as u64);
        dispatch_mlp_decode_cached_bf16_weights_resident(
            &dev,
            &x_buf,
            &g_buf,
            &u_buf,
            &d_buf,
            &scratch,
            &out_buf,
            batch,
            hidden,
            intermediate,
            out_dim,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);

        for (i, (b, r)) in baseline.iter().zip(resident.iter()).enumerate() {
            assert_eq!(b.to_bits(), r.to_bits(), "row {i}: baseline {b} vs resident {r}");
        }
    }

    #[test]
    fn full_attn_qkv_bf16w_resident_matches_nonresident_b1() {
        use crate::kernels::dispatch_full_attn_qkv_decode_cached_bf16_weights;
        let Some(dev) = try_device() else { return };
        let hidden = 96;
        let q_dim = 64;
        let k_dim = 32;
        let v_dim = 32;
        let x = make_x_f32(1, hidden);
        let q_w = make_bf16_weight(hidden, q_dim);
        let k_w = make_bf16_weight(hidden, k_dim);
        let v_w = make_bf16_weight(hidden, v_dim);
        let q_buf = upload_tensor_bf16_packed_buffer(&dev, &q_w).unwrap();
        let k_buf = upload_tensor_bf16_packed_buffer(&dev, &k_w).unwrap();
        let v_buf = upload_tensor_bf16_packed_buffer(&dev, &v_w).unwrap();

        let (q_t, k_t, v_t) = dispatch_full_attn_qkv_decode_cached_bf16_weights(
            &dev, &x, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
        )
        .unwrap();
        let mut expected: Vec<f32> = q_t.flatten_all().unwrap().to_vec1().unwrap();
        expected.extend(k_t.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        expected.extend(v_t.flatten_all().unwrap().to_vec1::<f32>().unwrap());

        let x_buf = upload_x(&dev, &x);
        let qkv_out = alloc_out(&dev, ((q_dim + k_dim + v_dim) * 4) as u64);
        dispatch_full_attn_qkv_decode_cached_resident(
            &dev, &x_buf, &q_buf, &k_buf, &v_buf, &qkv_out, hidden, q_dim, k_dim, v_dim, true,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &qkv_out);
        for (i, (e, r)) in expected.iter().zip(resident.iter()).enumerate() {
            assert_eq!(e.to_bits(), r.to_bits(), "idx {i}: expected {e} vs resident {r}");
        }
    }

    #[test]
    fn full_attn_qkv_bf16w_batched_resident_matches_nonresident() {
        use crate::kernels::dispatch_full_attn_qkv_decode_cached_batched_bf16_weights;
        let Some(dev) = try_device() else { return };
        let batch = 4;
        let hidden = 96;
        let q_dim = 64;
        let k_dim = 32;
        let v_dim = 32;
        let x = make_x_f32(batch, hidden);
        let q_w = make_bf16_weight(hidden, q_dim);
        let k_w = make_bf16_weight(hidden, k_dim);
        let v_w = make_bf16_weight(hidden, v_dim);
        let q_buf = upload_tensor_bf16_packed_buffer(&dev, &q_w).unwrap();
        let k_buf = upload_tensor_bf16_packed_buffer(&dev, &k_w).unwrap();
        let v_buf = upload_tensor_bf16_packed_buffer(&dev, &v_w).unwrap();

        let (q_t, k_t, v_t) = dispatch_full_attn_qkv_decode_cached_batched_bf16_weights(
            &dev, &x, &q_buf, &k_buf, &v_buf, batch, hidden, q_dim, k_dim, v_dim,
        )
        .unwrap();
        let q_v = q_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let k_v = k_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v_v = v_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let total_out = q_dim + k_dim + v_dim;
        let mut expected: Vec<f32> = Vec::with_capacity(batch * total_out);
        for r in 0..batch {
            expected.extend_from_slice(&q_v[r * q_dim..(r + 1) * q_dim]);
            expected.extend_from_slice(&k_v[r * k_dim..(r + 1) * k_dim]);
            expected.extend_from_slice(&v_v[r * v_dim..(r + 1) * v_dim]);
        }

        let x_buf = upload_x(&dev, &x);
        let qkv_out = alloc_out(&dev, (batch * total_out * 4) as u64);
        dispatch_full_attn_qkv_decode_cached_batched_resident(
            &dev, &x_buf, &q_buf, &k_buf, &v_buf, &qkv_out, batch, hidden, q_dim, k_dim, v_dim,
            true,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &qkv_out);
        for (i, (e, r)) in expected.iter().zip(resident.iter()).enumerate() {
            assert_eq!(e.to_bits(), r.to_bits(), "idx {i}: expected {e} vs resident {r}");
        }
    }

    #[test]
    fn qwen_rmsnorm_resident_matches_nonresident() {
        let Some(dev) = try_device() else { return };
        let rows = 7;
        let hidden = 96;
        let eps = 1e-6f32;
        let x = Tensor::from_vec(
            (0..rows * hidden)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
                .collect::<Vec<_>>(),
            (rows, hidden),
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::from_vec(
            (0..hidden).map(|i| ((i % 7) as f32) * 0.03).collect::<Vec<_>>(),
            (hidden,),
            &Device::Cpu,
        )
        .unwrap();

        let baseline = dispatch_qwen_rmsnorm_forward(&dev, &x, &weight, eps).unwrap();
        let baseline = baseline.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_buf = upload_tensor_f32_buffer(&dev, &x).unwrap();
        let weight_buf = upload_tensor_f32_buffer(&dev, &weight).unwrap();
        let out_buf = alloc_out(&dev, (rows * hidden * 4) as u64);
        dispatch_qwen_rmsnorm_forward_resident(
            &dev,
            &x_buf,
            &weight_buf,
            &out_buf,
            rows,
            hidden,
            eps,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);

        for (i, (b, r)) in baseline.iter().zip(resident.iter()).enumerate() {
            assert_eq!(b.to_bits(), r.to_bits(), "row {i}: baseline {b} vs resident {r}");
        }
    }
}
