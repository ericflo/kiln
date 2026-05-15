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

use crate::kernels::{run_compute_pipeline, run_compute_pipeline_3d};
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

/// Resident-form fused GDN input projection. Writes the combined
/// `[batch, qkv_dim + z_dim + a_dim + b_dim]` row-major output into
/// `out`. Shader selection mirrors `dispatch_gdn_in_proj_decode_cached_impl`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_in_proj_decode_cached_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    packed_bf16_weights: bool,
) -> Result<()> {
    let total_out = qkv_dim + z_dim + a_dim + b_dim;
    let need_in = (batch * hidden * 4) as u64;
    let need_out = (batch * total_out * 4) as u64;
    anyhow::ensure!(
        x.size() >= need_in,
        "gdn_in_proj_resident: x buffer {} bytes < required {need_in}",
        x.size()
    );
    anyhow::ensure!(
        out.size() >= need_out,
        "gdn_in_proj_resident: out buffer {} bytes < required {need_out}",
        out.size()
    );

    let pair_qkv_z = batch > 1 && crate::kernels::gdn_in_proj_batch_pair_qkv_z_enabled();
    let row_grouping = packed_bf16_weights
        && pair_qkv_z
        && batch >= 3
        && crate::kernels::gdn_in_proj_batch_row_pair_enabled();
    let row_group_size = if row_grouping
        && batch >= 8
        && crate::kernels::gdn_in_proj_batch_row_quad_enabled()
    {
        4usize
    } else if row_grouping {
        2usize
    } else {
        1usize
    };
    let dispatch_cols = if pair_qkv_z {
        qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + a_dim + b_dim
    } else {
        total_out
    };

    let glsl_path = if batch == 1 {
        if packed_bf16_weights {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode.comp"
            )
        }
    } else if packed_bf16_weights {
        if row_group_size == 4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_rows4_bf16w.comp"
            )
        } else if row_group_size == 2 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_rows2_bf16w.comp"
            )
        } else if pair_qkv_z {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_batched_bf16w.comp"
            )
        }
    } else if pair_qkv_z {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_in_proj_decode_batched.comp"
        )
    };
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let mut push_constants = vec![
        hidden as u32,
        qkv_dim as u32,
        z_dim as u32,
        a_dim as u32,
        b_dim as u32,
        total_out as u32,
    ];
    if batch > 1 {
        push_constants.push(batch as u32);
    }
    let handles: [vk::Buffer; 6] = [
        x.handle(),
        qkv_weight_t.handle(),
        z_weight_t.handle(),
        a_weight_t.handle(),
        b_weight_t.handle(),
        out.handle(),
    ];
    let workgroups = if batch == 1 {
        total_out.div_ceil(16) as u32
    } else if row_group_size > 1 {
        (batch.div_ceil(row_group_size) * dispatch_cols.div_ceil(80)) as u32
    } else {
        (batch * dispatch_cols.div_ceil(80)) as u32
    };
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        workgroups,
    )
    .context("gdn_in_proj_decode_resident kernel failed")
}

/// Resident-form fused GDN gates (β, g). Writes `[elem_count]` f32
/// for each into `beta_out` and `g_out` respectively. `a` and `b`
/// are f32 inputs already on device; `a_log` and `dt_bias` are
/// long-lived weights.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_gates_cached_resident(
    vk_device: &VulkanDevice,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    a_log: &VulkanBuffer,
    dt_bias: &VulkanBuffer,
    beta_out: &VulkanBuffer,
    g_out: &VulkanBuffer,
    elem_count: usize,
    nv: usize,
) -> Result<()> {
    let need = (elem_count * 4) as u64;
    anyhow::ensure!(
        a.size() >= need && b.size() >= need,
        "gdn_gates_resident: a/b buffers must each be >= {need} bytes"
    );
    anyhow::ensure!(
        beta_out.size() >= need && g_out.size() >= need,
        "gdn_gates_resident: out buffers must each be >= {need} bytes"
    );
    let glsl_path = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/gdn_gates.comp");
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 2] = [elem_count as u32, nv as u32];
    let workgroup_count = elem_count.div_ceil(256) as u32;
    let handles: [vk::Buffer; 6] = [
        a.handle(),
        b.handle(),
        a_log.handle(),
        dt_bias.handle(),
        beta_out.handle(),
        g_out.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("gdn_gates_resident kernel failed")
}

/// Resident-form GDN gated RMS norm: `out = rms_norm(x, weight, eps) * silu(z)`.
/// `rows = elem_count / hidden`. Writes `[elem_count]` f32 into `out`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_gated_rms_norm_cached_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    z: &VulkanBuffer,
    weight: &VulkanBuffer,
    out: &VulkanBuffer,
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let elem_count = rows * hidden;
    let need = (elem_count * 4) as u64;
    anyhow::ensure!(x.size() >= need, "gdn_gated_rms_norm_resident: x buffer too small");
    anyhow::ensure!(z.size() >= need, "gdn_gated_rms_norm_resident: z buffer too small");
    anyhow::ensure!(out.size() >= need, "gdn_gated_rms_norm_resident: out buffer too small");
    anyhow::ensure!(
        weight.size() >= (hidden * 4) as u64,
        "gdn_gated_rms_norm_resident: weight buffer too small"
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_gated_rms_norm.comp"
    );
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];
    let handles: [vk::Buffer; 4] = [x.handle(), z.handle(), weight.handle(), out.handle()];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        rows as u32,
    )
    .context("gdn_gated_rms_norm_resident kernel failed")
}

/// Resident-form fused GDN single-token gates + recurrent + gated
/// RMSNorm. Caller pre-uploads all 10 inputs (q/k/v/a/b/a_log/dt_bias/
/// state/z/weight); the kernel mutates `state` in place and writes
/// the gated-RMS-norm output of shape `[batch, 1, nv, dv]` into `out`.
///
/// Mirrors `dispatch_gdn_decode_gates_recurrent_rmsnorm`: same shader,
/// same push constants, same workgroup count.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_decode_gates_recurrent_rmsnorm_resident(
    vk_device: &VulkanDevice,
    q: &VulkanBuffer,
    k: &VulkanBuffer,
    v: &VulkanBuffer,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    a_log: &VulkanBuffer,
    dt_bias: &VulkanBuffer,
    state: &VulkanBuffer,
    z: &VulkanBuffer,
    weight: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    eps: f32,
) -> Result<()> {
    anyhow::ensure!(
        dv <= 256,
        "gdn_decode fused resident: dv {dv} exceeds shader local capacity 256"
    );
    anyhow::ensure!(
        out.size() >= (batch * nv * dv * 4) as u64,
        "gdn_decode fused resident: out buffer too small"
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_decode_gates_recurrent_rmsnorm.comp"
    );
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [
        nv as u32,
        dk as u32,
        dv as u32,
        eps.to_bits(),
        batch as u32,
    ];
    let handles: [vk::Buffer; 11] = [
        q.handle(),
        k.handle(),
        v.handle(),
        a.handle(),
        b.handle(),
        a_log.handle(),
        dt_bias.handle(),
        state.handle(),
        z.handle(),
        weight.handle(),
        out.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        (batch * nv) as u32,
    )
    .context("gdn_decode_gates_recurrent_rmsnorm_resident kernel failed")
}

/// Resident-form fused causal conv1d single-step update.
///
/// Two-dispatch flow mirrors `dispatch_causal_conv1d_update`:
///   stage 1 (`causal_conv1d.comp`)            computes `out`
///   stage 2 (`causal_conv1d_state_advance.comp`) advances `state`
///
/// `state` is mutated in place — the next layer reads it from the
/// same buffer. `kernel_size` must be 4 (matching the existing shader
/// specialization).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_causal_conv1d_update_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    weight: &VulkanBuffer,
    state: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<()> {
    anyhow::ensure!(
        kernel_size == 4,
        "causal_conv1d_resident: only kernel_size=4 supported"
    );
    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = ShaderPipeline::compile_shader(glsl_output)?;
    let output_handles: [vk::Buffer; 4] = [
        x.handle(),
        weight.handle(),
        state.handle(),
        out.handle(),
    ];
    let output_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let total = batch * channels * seq_len;
    let output_wg = total.div_ceil(256) as u32;
    run_compute_pipeline(
        vk_device,
        &spirv_output,
        &output_handles,
        output_handles.len(),
        &output_push,
        output_wg,
    )
    .context("causal_conv1d_update_resident: stage-1 output kernel failed")?;

    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = ShaderPipeline::compile_shader(glsl_state)?;
    let state_handles: [vk::Buffer; 2] = [x.handle(), state.handle()];
    let state_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let state_wg = (batch * channels) as u32;
    run_compute_pipeline(
        vk_device,
        &spirv_state,
        &state_handles,
        state_handles.len(),
        &state_push,
        state_wg,
    )
    .context("causal_conv1d_update_resident: stage-2 state-advance kernel failed")
}

/// Resident-form element-wise vector add: `out[i] = a[i] + b[i]`.
/// Used to materialise the residual connections inside the resident
/// decode block without going through a candle `(x + y)?` (which
/// allocates a fresh CPU Tensor every layer).
pub fn dispatch_add_resident(
    vk_device: &VulkanDevice,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    out: &VulkanBuffer,
    n_elements: usize,
) -> Result<()> {
    let need = (n_elements * 4) as u64;
    anyhow::ensure!(a.size() >= need, "add_resident: a buffer too small");
    anyhow::ensure!(b.size() >= need, "add_resident: b buffer too small");
    anyhow::ensure!(out.size() >= need, "add_resident: out buffer too small");
    let glsl_path = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/add.comp");
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 1] = [n_elements as u32];
    let handles: [vk::Buffer; 3] = [a.handle(), b.handle(), out.handle()];
    let workgroups = n_elements.div_ceil(256) as u32;
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        workgroups,
    )
    .context("add_resident kernel failed")
}

/// Resident-form attention output gate: `out[i] = a[i] * sigmoid(gate[i])`.
/// Used inside the full-attention layer when `attn_output_gate = true`
/// (Qwen3.5-4B always does). Lifts the gate computation off the candle
/// path which would otherwise materialise sigmoid + multiply Tensors.
pub fn dispatch_mul_sigmoid_gate_resident(
    vk_device: &VulkanDevice,
    a: &VulkanBuffer,
    gate: &VulkanBuffer,
    out: &VulkanBuffer,
    n_elements: usize,
) -> Result<()> {
    let need = (n_elements * 4) as u64;
    anyhow::ensure!(
        a.size() >= need && gate.size() >= need && out.size() >= need,
        "mul_sigmoid_gate_resident: a/gate/out buffers must each be >= {need} bytes"
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/vk_mul_sigmoid_gate_f32.comp"
    );
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 1] = [n_elements as u32];
    let handles: [vk::Buffer; 3] = [a.handle(), gate.handle(), out.handle()];
    let workgroups = n_elements.div_ceil(256) as u32;
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        workgroups,
    )
    .context("mul_sigmoid_gate_resident kernel failed")
}

/// Resident-form RoPE rotation for a single Q-or-K-style tensor of
/// shape `[rows, num_heads, head_dim]`. The first `rotary_dim` dims
/// per head are rotated; the remainder pass through unchanged. `cos`
/// and `sin` are `[rows, rotary_dim/2]` precomputed tables.
///
/// Uses the existing `vk_rope_f32.comp` shader landed for the
/// training-side autograd stack — same shader, same push constants,
/// so the rotation arithmetic stays bit-identical with that path.
/// Wraps it for decode hot path use without going through the
/// candle-based `apply_rope` (which materialises ~6 intermediate
/// Tensors per RoPE call and is currently the only Vulkan-decode RoPE
/// path — this lifts that cost off the CPU).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rotary_one_resident(
    vk_device: &VulkanDevice,
    x: &VulkanBuffer,
    cos: &VulkanBuffer,
    sin: &VulkanBuffer,
    out: &VulkanBuffer,
    rows: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<()> {
    anyhow::ensure!(
        rotary_dim <= head_dim && rotary_dim % 2 == 0,
        "rotary_one_resident: rotary_dim={rotary_dim} must be <= head_dim={head_dim} and even"
    );
    let need = (rows * num_heads * head_dim * 4) as u64;
    anyhow::ensure!(x.size() >= need, "rotary_one_resident: x buffer too small");
    anyhow::ensure!(
        out.size() >= need,
        "rotary_one_resident: out buffer too small"
    );

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/vk_rope_f32.comp"
    );
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 4] = [
        rows as u32,
        num_heads as u32,
        head_dim as u32,
        rotary_dim as u32,
    ];
    let handles: [vk::Buffer; 4] = [x.handle(), cos.handle(), sin.handle(), out.handle()];
    let total = rows * num_heads * head_dim;
    let workgroups = total.div_ceil(256) as u32;
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        workgroups,
    )
    .context("rotary_one_resident kernel failed")
}

/// Convenience: rotates Q and K with one dispatch per side. Q and K
/// often have different head counts (num_attention_heads vs
/// num_kv_heads under GQA). Both share the same `cos` / `sin` table.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rotary_qk_resident(
    vk_device: &VulkanDevice,
    q: &VulkanBuffer,
    k: &VulkanBuffer,
    cos: &VulkanBuffer,
    sin: &VulkanBuffer,
    q_out: &VulkanBuffer,
    k_out: &VulkanBuffer,
    rows: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<()> {
    dispatch_rotary_one_resident(
        vk_device,
        q,
        cos,
        sin,
        q_out,
        rows,
        num_q_heads,
        head_dim,
        rotary_dim,
    )?;
    dispatch_rotary_one_resident(
        vk_device,
        k,
        cos,
        sin,
        k_out,
        rows,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )
}

/// Resident-form GDN single-token recurrent step (fallback path used
/// when the fully-fused `dispatch_gdn_decode_gates_recurrent_rmsnorm`
/// is declined). Writes `out: [batch * heads * dv]` f32 and mutates
/// `state` in place.
///
/// Push constants are `[batch, heads, 1, dk, dv, heads]` — matching
/// the legacy dispatcher's `[batch, value_heads, seq_len, dk, dv,
/// q_heads]` layout where `seq_len = 1` for the decode step and the
/// q/v head counts are equal in the fallback path.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_step_resident(
    vk_device: &VulkanDevice,
    q: &VulkanBuffer,
    k: &VulkanBuffer,
    v: &VulkanBuffer,
    beta: &VulkanBuffer,
    g: &VulkanBuffer,
    state: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
) -> Result<()> {
    anyhow::ensure!(
        out.size() >= (batch * heads * dv * 4) as u64,
        "gdn_recurrent_step_resident: out buffer too small"
    );
    let parallel_reduce = crate::kernels::use_gdn_recurrent_parallel_reduce(dk, dv);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        heads as u32,
    ];
    let handles: [vk::Buffer; 7] = [
        q.handle(),
        k.handle(),
        v.handle(),
        beta.handle(),
        g.handle(),
        state.handle(),
        out.handle(),
    ];
    if parallel_reduce {
        // The `gdn_recurrent_step_parallel.comp` shader expects a 3D
        // dispatch `(batch, heads, dv)` — see
        // `dispatch_gdn_recurrent_step_single_submit` in kernels.rs.
        // The 1D linear-count form would silently zero entire output
        // rows.
        run_compute_pipeline_3d(
            vk_device,
            &spirv,
            &handles,
            handles.len(),
            &push_constants,
            (batch as u32, heads as u32, dv as u32),
        )
        .context("gdn_recurrent_step_resident parallel-reduce kernel failed")
    } else {
        let total = batch * heads * dv;
        let workgroup_count = total.div_ceil(256) as u32;
        run_compute_pipeline(
            vk_device,
            &spirv,
            &handles,
            handles.len(),
            &push_constants,
            workgroup_count,
        )
        .context("gdn_recurrent_step_resident kernel failed")
    }
}

/// Resident-form batched paged attention (compacted K/V variant).
/// Writes `[batch, num_heads, head_dim]` f32 into `out`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attn_decode_batch_f32_resident(
    vk_device: &VulkanDevice,
    q: &VulkanBuffer,
    k: &VulkanBuffer,
    v: &VulkanBuffer,
    seq_lens: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seqlen: usize,
    softmax_scale: f32,
) -> Result<()> {
    anyhow::ensure!(head_dim <= 256, "paged_attn_resident: head_dim {head_dim} > 256");
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "paged_attn_resident: num_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}"
    );
    anyhow::ensure!(
        out.size() >= (batch * num_heads * head_dim * 4) as u64,
        "paged_attn_resident: out buffer too small"
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch.comp"
    );
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [
        max_seqlen as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
    ];
    let handles: [vk::Buffer; 5] = [
        q.handle(),
        k.handle(),
        v.handle(),
        seq_lens.handle(),
        out.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        (batch * num_heads) as u32,
    )
    .context("paged_attn_decode_batch_f32_resident kernel failed")
}

/// Resident-form paged-pool batched attention.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attn_decode_batch_paged_f32_resident(
    vk_device: &VulkanDevice,
    q: &VulkanBuffer,
    k_pool: &VulkanBuffer,
    v_pool: &VulkanBuffer,
    block_table: &VulkanBuffer,
    seq_lens: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_blocks_per_seq: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<()> {
    anyhow::ensure!(head_dim <= 256, "paged_attn_paged_resident: head_dim {head_dim} > 256");
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "paged_attn_paged_resident: num_heads not divisible by num_kv_heads"
    );
    anyhow::ensure!(
        out.size() >= (batch * num_heads * head_dim * 4) as u64,
        "paged_attn_paged_resident: out buffer too small"
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged.comp"
    );
    let spirv = ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 6] = [
        max_blocks_per_seq as u32,
        page_block_size as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
    ];
    let handles: [vk::Buffer; 6] = [
        q.handle(),
        k_pool.handle(),
        v_pool.handle(),
        block_table.handle(),
        seq_lens.handle(),
        out.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &handles,
        handles.len(),
        &push_constants,
        (batch * num_heads) as u32,
    )
    .context("paged_attn_decode_batch_paged_f32_resident kernel failed")
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
    fn gdn_in_proj_bf16w_resident_matches_nonresident_b1() {
        use crate::kernels::dispatch_gdn_in_proj_decode_cached_bf16_weights;
        let Some(dev) = try_device() else { return };
        let hidden = 96;
        let qkv_dim = 64;
        let z_dim = 64;
        let a_dim = 8;
        let b_dim = 8;
        let total_out = qkv_dim + z_dim + a_dim + b_dim;
        let x = make_x_f32(1, hidden);
        let qkv_w = make_bf16_weight(hidden, qkv_dim);
        let z_w = make_bf16_weight(hidden, z_dim);
        let a_w = make_bf16_weight(hidden, a_dim);
        let b_w = make_bf16_weight(hidden, b_dim);
        let qkv_buf = upload_tensor_bf16_packed_buffer(&dev, &qkv_w).unwrap();
        let z_buf = upload_tensor_bf16_packed_buffer(&dev, &z_w).unwrap();
        let a_buf = upload_tensor_bf16_packed_buffer(&dev, &a_w).unwrap();
        let b_buf = upload_tensor_bf16_packed_buffer(&dev, &b_w).unwrap();

        let (qkv_t, z_t, a_t, b_t) = dispatch_gdn_in_proj_decode_cached_bf16_weights(
            &dev, &x, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim, z_dim, a_dim, b_dim,
        )
        .unwrap();
        let mut expected: Vec<f32> = qkv_t.flatten_all().unwrap().to_vec1().unwrap();
        expected.extend(z_t.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        expected.extend(a_t.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        expected.extend(b_t.flatten_all().unwrap().to_vec1::<f32>().unwrap());

        let x_buf = upload_x(&dev, &x);
        let out_buf = alloc_out(&dev, (total_out * 4) as u64);
        dispatch_gdn_in_proj_decode_cached_resident(
            &dev, &x_buf, &qkv_buf, &z_buf, &a_buf, &b_buf, &out_buf, 1, hidden, qkv_dim, z_dim,
            a_dim, b_dim, true,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);
        for (i, (e, r)) in expected.iter().zip(resident.iter()).enumerate() {
            assert_eq!(e.to_bits(), r.to_bits(), "idx {i}: expected {e} vs resident {r}");
        }
    }

    #[test]
    fn gdn_gates_resident_matches_nonresident() {
        use crate::kernels::dispatch_gdn_gates_cached;
        let Some(dev) = try_device() else { return };
        let batch = 2;
        let nv = 8;
        let t = 1;
        let elem_count = batch * t * nv;
        let a = Tensor::from_vec(
            (0..elem_count).map(|i| (i as f32 * 0.013) - 0.5).collect::<Vec<_>>(),
            (batch, t, nv),
            &Device::Cpu,
        )
        .unwrap();
        let b = Tensor::from_vec(
            (0..elem_count).map(|i| (i as f32 * 0.017) - 0.7).collect::<Vec<_>>(),
            (batch, t, nv),
            &Device::Cpu,
        )
        .unwrap();
        let a_log = Tensor::from_vec(
            (0..nv).map(|i| (i as f32 + 1.0).ln() * -0.1).collect::<Vec<_>>(),
            (nv,),
            &Device::Cpu,
        )
        .unwrap();
        let dt_bias = Tensor::from_vec(
            (0..nv).map(|i| (i as f32) * 0.011).collect::<Vec<_>>(),
            (nv,),
            &Device::Cpu,
        )
        .unwrap();
        let a_log_buf = upload_tensor_f32_buffer(&dev, &a_log).unwrap();
        let dt_bias_buf = upload_tensor_f32_buffer(&dev, &dt_bias).unwrap();

        let (beta_t, g_t) = dispatch_gdn_gates_cached(
            &dev,
            &a,
            &b,
            &a_log_buf,
            &dt_bias_buf,
            nv,
            &[batch, t, nv],
        )
        .unwrap();
        let beta_exp = beta_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let g_exp = g_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let a_buf_d = upload_tensor_f32_buffer(&dev, &a).unwrap();
        let b_buf_d = upload_tensor_f32_buffer(&dev, &b).unwrap();
        let beta_buf = alloc_out(&dev, (elem_count * 4) as u64);
        let g_buf = alloc_out(&dev, (elem_count * 4) as u64);
        dispatch_gdn_gates_cached_resident(
            &dev,
            &a_buf_d,
            &b_buf_d,
            &a_log_buf,
            &dt_bias_buf,
            &beta_buf,
            &g_buf,
            elem_count,
            nv,
        )
        .unwrap();
        let beta_res = read_back_f32(&dev, &beta_buf);
        let g_res = read_back_f32(&dev, &g_buf);
        for (i, (e, r)) in beta_exp.iter().zip(beta_res.iter()).enumerate() {
            assert_eq!(e.to_bits(), r.to_bits(), "beta idx {i}");
        }
        for (i, (e, r)) in g_exp.iter().zip(g_res.iter()).enumerate() {
            assert_eq!(e.to_bits(), r.to_bits(), "g idx {i}");
        }
    }

    #[test]
    fn gdn_gated_rms_norm_resident_matches_nonresident() {
        use crate::kernels::dispatch_gdn_gated_rms_norm_cached;
        let Some(dev) = try_device() else { return };
        let rows = 6;
        let hidden = 64;
        let eps = 1e-6f32;
        let x = Tensor::from_vec(
            (0..rows * hidden).map(|i| (i as f32 * 0.013) - 0.5).collect::<Vec<_>>(),
            (rows, hidden),
            &Device::Cpu,
        )
        .unwrap();
        let z = Tensor::from_vec(
            (0..rows * hidden).map(|i| (i as f32 * 0.017) - 0.3).collect::<Vec<_>>(),
            (rows, hidden),
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::from_vec(
            (0..hidden).map(|i| (i as f32) * 0.02 + 1.0).collect::<Vec<_>>(),
            (hidden,),
            &Device::Cpu,
        )
        .unwrap();
        let weight_buf = upload_tensor_f32_buffer(&dev, &weight).unwrap();

        let baseline = dispatch_gdn_gated_rms_norm_cached(
            &dev,
            &x,
            &z,
            &weight_buf,
            hidden,
            eps,
            &[rows, hidden],
        )
        .unwrap();
        let expected = baseline.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let x_buf = upload_tensor_f32_buffer(&dev, &x).unwrap();
        let z_buf = upload_tensor_f32_buffer(&dev, &z).unwrap();
        let out_buf = alloc_out(&dev, (rows * hidden * 4) as u64);
        dispatch_gdn_gated_rms_norm_cached_resident(
            &dev, &x_buf, &z_buf, &weight_buf, &out_buf, rows, hidden, eps,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);
        for (i, (e, r)) in expected.iter().zip(resident.iter()).enumerate() {
            assert_eq!(e.to_bits(), r.to_bits(), "idx {i}");
        }
    }

    #[test]
    fn paged_attn_decode_batch_resident_matches_nonresident() {
        use crate::kernels::dispatch_paged_attn_decode_batch_f32;
        let Some(dev) = try_device() else { return };
        let batch = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32;
        let max_seqlen = 8;
        let softmax_scale = (head_dim as f32).sqrt().recip();
        let q = Tensor::from_vec(
            (0..batch * 1 * num_heads * head_dim)
                .map(|i| (i as f32 * 0.013) - 1.0)
                .collect::<Vec<_>>(),
            (batch, 1, num_heads, head_dim),
            &Device::Cpu,
        )
        .unwrap();
        let k = Tensor::from_vec(
            (0..batch * max_seqlen * num_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.011) - 0.5)
                .collect::<Vec<_>>(),
            (batch, max_seqlen, num_kv_heads, head_dim),
            &Device::Cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            (0..batch * max_seqlen * num_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.007) + 0.1)
                .collect::<Vec<_>>(),
            (batch, max_seqlen, num_kv_heads, head_dim),
            &Device::Cpu,
        )
        .unwrap();
        let seq_lens: Vec<u32> = vec![max_seqlen as u32; batch];

        let baseline = dispatch_paged_attn_decode_batch_f32(&dev, &q, &k, &v, &seq_lens, softmax_scale)
            .unwrap();
        let expected = baseline.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let q_buf = upload_tensor_f32_buffer(&dev, &q).unwrap();
        let k_buf = upload_tensor_f32_buffer(&dev, &k).unwrap();
        let v_buf = upload_tensor_f32_buffer(&dev, &v).unwrap();
        // seq_lens uploaded as raw u32 bytes
        let seq_bytes: Vec<u8> = bytemuck::cast_slice(&seq_lens).to_vec();
        let seq_buf = {
            let buf = VulkanBuffer::create_device_local(
                dev.device(),
                dev.device_local_mem_type(),
                seq_bytes.len() as u64,
            )
            .unwrap();
            let pool = dev.transient_command_pool().unwrap();
            VulkanBuffer::upload_data_with_command_pool(
                dev.device(),
                dev.host_visible_mem_type(),
                dev.queue(),
                *pool,
                &buf,
                &seq_bytes,
            )
            .unwrap();
            buf
        };
        let out_buf = alloc_out(&dev, (batch * num_heads * head_dim * 4) as u64);
        dispatch_paged_attn_decode_batch_f32_resident(
            &dev, &q_buf, &k_buf, &v_buf, &seq_buf, &out_buf, batch, num_heads, num_kv_heads,
            head_dim, max_seqlen, softmax_scale,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);
        for (i, (e, r)) in expected.iter().zip(resident.iter()).enumerate() {
            assert_eq!(e.to_bits(), r.to_bits(), "idx {i}");
        }
    }

    #[test]
    fn add_resident_matches_cpu_reference() {
        let Some(dev) = try_device() else { return };
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.019 + 0.2).collect();
        let a_t = Tensor::from_vec(a.clone(), n, &Device::Cpu).unwrap();
        let b_t = Tensor::from_vec(b.clone(), n, &Device::Cpu).unwrap();
        let a_buf = upload_tensor_f32_buffer(&dev, &a_t).unwrap();
        let b_buf = upload_tensor_f32_buffer(&dev, &b_t).unwrap();
        let out_buf = alloc_out(&dev, (n * 4) as u64);
        dispatch_add_resident(&dev, &a_buf, &b_buf, &out_buf, n).unwrap();
        let got = read_back_f32(&dev, &out_buf);
        for i in 0..n {
            let expected = a[i] + b[i];
            assert!(
                (expected - got[i]).abs() <= 1e-6,
                "idx {i}: {expected} vs {}",
                got[i]
            );
        }
    }

    #[test]
    fn mul_sigmoid_gate_resident_matches_cpu_reference() {
        let Some(dev) = try_device() else { return };
        let n = 1024usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 0.5).collect();
        let g: Vec<f32> = (0..n).map(|i| (i as f32) * 0.011 - 5.0).collect();
        let a_t = Tensor::from_vec(a.clone(), n, &Device::Cpu).unwrap();
        let g_t = Tensor::from_vec(g.clone(), n, &Device::Cpu).unwrap();
        let a_buf = upload_tensor_f32_buffer(&dev, &a_t).unwrap();
        let g_buf = upload_tensor_f32_buffer(&dev, &g_t).unwrap();
        let out_buf = alloc_out(&dev, (n * 4) as u64);
        dispatch_mul_sigmoid_gate_resident(&dev, &a_buf, &g_buf, &out_buf, n).unwrap();
        let got = read_back_f32(&dev, &out_buf);
        for i in 0..n {
            let sigmoid = if g[i] >= 0.0 {
                1.0 / (1.0 + (-g[i]).exp())
            } else {
                let e = g[i].exp();
                e / (1.0 + e)
            };
            let expected = a[i] * sigmoid;
            // Tolerance is ~ulp-of-magnitude * 2: sigmoid is computed
            // with one of two stable branches and a single multiply,
            // so we expect ≤2 ulps relative error.
            let tol = expected.abs().max(1.0) * 1e-6;
            assert!(
                (expected - got[i]).abs() <= tol,
                "idx {i}: {expected} vs {} (a={}, g={}, tol={tol:e})",
                got[i],
                a[i],
                g[i]
            );
        }
    }

    #[test]
    fn rotary_one_resident_matches_cpu_reference() {
        let Some(dev) = try_device() else { return };
        let rows = 3usize;
        let num_heads = 4usize;
        let head_dim = 16usize;
        let rotary_dim = 8usize;
        let half = rotary_dim / 2;
        let n = rows * num_heads * head_dim;
        let x_data: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
        let cos_data: Vec<f32> = (0..rows * half)
            .map(|i| ((i as f32) * 0.13).cos())
            .collect();
        let sin_data: Vec<f32> = (0..rows * half)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let x = Tensor::from_vec(x_data.clone(), (rows, num_heads, head_dim), &Device::Cpu).unwrap();
        let cos_t =
            Tensor::from_vec(cos_data.clone(), (rows, half), &Device::Cpu).unwrap();
        let sin_t =
            Tensor::from_vec(sin_data.clone(), (rows, half), &Device::Cpu).unwrap();

        let x_buf = upload_tensor_f32_buffer(&dev, &x).unwrap();
        let cos_buf = upload_tensor_f32_buffer(&dev, &cos_t).unwrap();
        let sin_buf = upload_tensor_f32_buffer(&dev, &sin_t).unwrap();
        let out_buf = alloc_out(&dev, (n * 4) as u64);
        dispatch_rotary_one_resident(
            &dev, &x_buf, &cos_buf, &sin_buf, &out_buf, rows, num_heads, head_dim, rotary_dim,
        )
        .unwrap();
        let got = read_back_f32(&dev, &out_buf);

        // CPU reference: same rotation the shader implements.
        let mut expected = vec![0f32; n];
        for r in 0..rows {
            for h in 0..num_heads {
                let base = (r * num_heads + h) * head_dim;
                for d in 0..head_dim {
                    let idx = base + d;
                    if d >= rotary_dim {
                        expected[idx] = x_data[idx];
                        continue;
                    }
                    let half_r = rotary_dim / 2;
                    let (pair, is_low) = if d < half_r {
                        (d, true)
                    } else {
                        (d - half_r, false)
                    };
                    let low = x_data[base + pair];
                    let high = x_data[base + pair + half_r];
                    let c = cos_data[r * half_r + pair];
                    let s = sin_data[r * half_r + pair];
                    expected[idx] = if is_low { low * c - high * s } else { low * s + high * c };
                }
            }
        }
        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (e - g).abs() <= 1e-5,
                "idx {i}: cpu {e} vs gpu {g} (delta {})",
                (e - g).abs()
            );
        }
    }

    #[test]
    fn gdn_recurrent_step_resident_matches_nonresident() {
        use crate::kernels::dispatch_gdn_recurrent_step;
        let Some(dev) = try_device() else { return };
        let batch = 2;
        let heads = 4;
        let dk = 32;
        let dv = 16;
        let q = Tensor::from_vec(
            (0..batch * heads * dk).map(|i| (i as f32 * 0.013) - 0.5).collect::<Vec<_>>(),
            (batch, heads, dk),
            &Device::Cpu,
        )
        .unwrap();
        let k = Tensor::from_vec(
            (0..batch * heads * dk).map(|i| (i as f32 * 0.017) - 0.3).collect::<Vec<_>>(),
            (batch, heads, dk),
            &Device::Cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            (0..batch * heads * dv).map(|i| (i as f32 * 0.019) + 0.2).collect::<Vec<_>>(),
            (batch, heads, dv),
            &Device::Cpu,
        )
        .unwrap();
        let beta = Tensor::from_vec(
            (0..batch * heads).map(|i| (i as f32 * 0.05) + 0.1).collect::<Vec<_>>(),
            (batch, heads),
            &Device::Cpu,
        )
        .unwrap();
        let g = Tensor::from_vec(
            (0..batch * heads).map(|i| ((i as f32) * 0.03 - 0.1).tanh()).collect::<Vec<_>>(),
            (batch, heads),
            &Device::Cpu,
        )
        .unwrap();
        let state = Tensor::from_vec(
            (0..batch * heads * dk * dv).map(|i| (i as f32 * 0.0017) - 0.05).collect::<Vec<_>>(),
            (batch, heads, dk, dv),
            &Device::Cpu,
        )
        .unwrap();

        let (out_t, _new_state_t) =
            dispatch_gdn_recurrent_step(&dev, &q, &k, &v, &beta, &g, &state).unwrap();
        let expected = out_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let q_buf = upload_tensor_f32_buffer(&dev, &q).unwrap();
        let k_buf = upload_tensor_f32_buffer(&dev, &k).unwrap();
        let v_buf = upload_tensor_f32_buffer(&dev, &v).unwrap();
        let beta_buf = upload_tensor_f32_buffer(&dev, &beta).unwrap();
        let g_buf = upload_tensor_f32_buffer(&dev, &g).unwrap();
        let state_buf = upload_tensor_f32_buffer(&dev, &state).unwrap();
        let out_buf = alloc_out(&dev, (batch * heads * dv * 4) as u64);
        dispatch_gdn_recurrent_step_resident(
            &dev, &q_buf, &k_buf, &v_buf, &beta_buf, &g_buf, &state_buf, &out_buf, batch, heads, dk,
            dv,
        )
        .unwrap();
        let resident = read_back_f32(&dev, &out_buf);
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
