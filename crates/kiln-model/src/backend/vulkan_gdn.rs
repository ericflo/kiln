//! Vulkan Gated DeltaNet operation helpers.
//!
//! This starts the GDN operation-family split with the small gate and gated
//! RMSNorm dispatch hooks. The larger recurrent/chunkwise paths stay in the
//! `BackendRuntime` facade until they can move in narrower slices.

use anyhow::{Context, Result};

use super::vulkan::VulkanBackend;
use super::vulkan_tensor_bridge::{kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape};

pub(super) fn supports_gdn_forward_substitution(backend: &VulkanBackend) -> bool {
    // solve_tri is experimental: shared-memory layout not yet validated
    // against CPU parity, and may exceed maxComputeSharedMemorySize on many
    // GPUs. Opt-in only via KILN_ENABLE_VULKAN_GDN_FORWARD_SUB.
    backend.has_vulkan() && backend.gdn_forward_sub_enabled
}

pub(super) fn supports_gdn_recurrent_step(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_enabled
}

pub(super) fn supports_gdn_recurrent_prefill_native_head_last(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_recurrent_unexpanded_qk_enabled
}

pub(super) fn supports_gdn_recurrent_qk_norm_prefill_native_head_last(
    backend: &VulkanBackend,
) -> bool {
    backend.has_vulkan() && backend.gdn_recurrent_qk_norm_unexpanded_enabled
}

pub(super) fn supports_gdn_chunk_prep(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_enabled
}

pub(super) fn supports_gdn_chunk_scan(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_enabled
}

pub(super) fn supports_gdn_full_chunk_forward(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_full_chunk_forward_enabled
}

pub(super) fn supports_gdn_gates(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_gates_enabled
}

pub(super) fn supports_gdn_gated_rms_norm(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.gdn_gated_rms_norm_enabled
}

pub(super) fn gdn_in_proj_decode(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    in_proj_qkv_t: &kiln_tensor::Tensor,
    in_proj_z_t: &kiln_tensor::Tensor,
    in_proj_a_t: &kiln_tensor::Tensor,
    in_proj_b_t: &kiln_tensor::Tensor,
) -> Result<
    Option<(
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
    )>,
> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.gdn_enabled || x.dtype() != kiln_tensor::DType::F32 {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(in_proj_qkv_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(in_proj_z_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(in_proj_a_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(in_proj_b_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: shapes off kt, weight buffers keyed on the
    // stable kt id (upload once), x bytes + outputs straight from/to kt.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    if seq_len != 1 && !backend.gdn_prefill_in_proj_enabled {
        return Ok(None);
    }

    let Ok((qkv_hidden, qkv_dim)) = in_proj_qkv_t.dims2() else {
        return Ok(None);
    };
    let Ok((z_hidden, z_dim)) = in_proj_z_t.dims2() else {
        return Ok(None);
    };
    let Ok((a_hidden, a_dim)) = in_proj_a_t.dims2() else {
        return Ok(None);
    };
    let Ok((b_hidden, b_dim)) = in_proj_b_t.dims2() else {
        return Ok(None);
    };
    if qkv_hidden != hidden || z_hidden != hidden || a_hidden != hidden || b_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let row_count = batch * seq_len;
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let use_bf16 = backend.bf16_packed_gdn_in_proj_weights_enabled
        && in_proj_qkv_t.dtype() == kiln_tensor::DType::BF16
        && in_proj_z_t.dtype() == kiln_tensor::DType::BF16
        && in_proj_a_t.dtype() == kiln_tensor::DType::BF16
        && in_proj_b_t.dtype() == kiln_tensor::DType::BF16;
    let (qkv_b, z_b, a_b, b_b) = if use_bf16 {
        let qkv_buf = backend.cached_bf16_packed_weight_buffer_kt(in_proj_qkv_t)?;
        let z_buf = backend.cached_bf16_packed_weight_buffer_kt(in_proj_z_t)?;
        let a_buf = backend.cached_bf16_packed_weight_buffer_kt(in_proj_a_t)?;
        let b_buf = backend.cached_bf16_packed_weight_buffer_kt(in_proj_b_t)?;
        kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bf16_weights_bytes(
            vk_device, &x_data, row_count, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim,
            z_dim, a_dim, b_dim,
        )
        .context("gdn_in_proj_decode kernel failed")?
    } else {
        let qkv_buf = backend.cached_f32_weight_buffer_kt(in_proj_qkv_t)?;
        let z_buf = backend.cached_f32_weight_buffer_kt(in_proj_z_t)?;
        let a_buf = backend.cached_f32_weight_buffer_kt(in_proj_a_t)?;
        let b_buf = backend.cached_f32_weight_buffer_kt(in_proj_b_t)?;
        kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bytes(
            vk_device, &x_data, row_count, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim,
            z_dim, a_dim, b_dim,
        )
        .context("gdn_in_proj_decode kernel failed")?
    };
    Ok(Some((
        kt_tensor_from_f32_bytes(&qkv_b, &[batch, seq_len, qkv_dim], kiln_tensor::DType::F32)?,
        kt_tensor_from_f32_bytes(&z_b, &[batch, seq_len, z_dim], kiln_tensor::DType::F32)?,
        kt_tensor_from_f32_bytes(&a_b, &[batch, seq_len, a_dim], kiln_tensor::DType::F32)?,
        kt_tensor_from_f32_bytes(&b_b, &[batch, seq_len, b_dim], kiln_tensor::DType::F32)?,
    )))
}

pub(super) fn gdn_gates(
    backend: &VulkanBackend,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
    // kt guards read directly off the kt args before the bridge.
    if !supports_gdn_gates(backend) {
        return Ok(None);
    }
    if !matches!(
        a.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1082) kt-native: weight buffers keyed on the stable kt id; byte
    // extraction + reconstruction run on the kt args.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let nv = a_log.elem_count();
    if dt_bias.elem_count() != nv {
        return Ok(None);
    }
    let a_log_buf = backend.cached_f32_weight_buffer_kt(a_log)?;
    let dt_bias_buf = backend.cached_f32_weight_buffer_kt(dt_bias)?;

    // Output shape matches input shape [B, T, nv]
    let out_shape = a.dims().to_vec();
    let a_data = kt_tensor_to_f32_bytes_with_shape(a)?.0;
    let b_data = kt_tensor_to_f32_bytes_with_shape(b)?.0;
    let output_dtype = a.dtype();
    let (beta_b, g_b) = kiln_vulkan_kernel::kernels::dispatch_gdn_gates_cached_bytes(
        vk_device,
        &a_data,
        &b_data,
        &a_log_buf,
        &dt_bias_buf,
        nv,
        &out_shape,
    )
    .context("gdn_gates kernel failed")?;
    let beta = kt_tensor_from_f32_bytes(&beta_b, &out_shape, output_dtype)?;
    let g = kt_tensor_from_f32_bytes(&g_b, &out_shape, output_dtype)?;
    Ok(Some((beta, g)))
}

pub(super) fn gdn_gated_rms_norm(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f64,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !supports_gdn_gated_rms_norm(backend) {
        return Ok(None);
    }
    if !matches!(
        x.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1082) kt-native: weight buffer keyed on the stable kt id; byte
    // extraction + reconstruction run on the kt args.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let hidden = weight.elem_count();
    if hidden == 0 || x.elem_count() % hidden != 0 {
        return Ok(None);
    }
    let weight_buf = backend.cached_f32_weight_buffer_kt(weight)?;

    // Output shape matches x shape
    let out_shape = x.dims().to_vec();
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let z_data = kt_tensor_to_f32_bytes_with_shape(z)?.0;
    let output_dtype = x.dtype();
    let out_data = kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm_cached_bytes(
        vk_device,
        &x_data,
        &z_data,
        &weight_buf,
        hidden,
        eps as f32,
        &out_shape,
    )
    .context("gdn_gated_rms_norm kernel failed")?;
    let out = kt_tensor_from_f32_bytes(&out_data, &out_shape, output_dtype)?;
    Ok(Some(out))
}
