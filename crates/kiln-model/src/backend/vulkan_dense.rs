//! Vulkan dense projection operation helpers.
//!
//! `backend/vulkan.rs` remains the `BackendRuntime` facade. This module owns
//! the kt-native full-attention QKV and MLP decode projection dispatches that
//! share cached explicit Vulkan weight buffers.

use anyhow::{Context, Result};

use super::vulkan::VulkanBackend;
use super::vulkan_tensor_bridge::{kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape};

pub(super) fn full_attn_qkv_decode(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    q_weight_t: &kiln_tensor::Tensor,
    k_weight_t: &kiln_tensor::Tensor,
    v_weight_t: &kiln_tensor::Tensor,
) -> Result<
    Option<(
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
    )>,
> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan()
        || !backend.full_attn_qkv_enabled
        || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(q_weight_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(k_weight_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(v_weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: shapes off kt, QKV weight buffers keyed on
    // the stable kt id (upload once), x bytes + outputs straight from/to kt.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    // Multi-token (prefill-ish) shapes still go through the unfused
    // path: this kernel family is the single-token decode projection.
    // Batched single-token decode IS supported via the `_batched` dispatch.
    if seq_len != 1 || batch == 0 {
        return Ok(None);
    }
    let Ok((q_hidden, q_dim)) = q_weight_t.dims2() else {
        return Ok(None);
    };
    let Ok((k_hidden, k_dim)) = k_weight_t.dims2() else {
        return Ok(None);
    };
    let Ok((v_hidden, v_dim)) = v_weight_t.dims2() else {
        return Ok(None);
    };
    if q_hidden != hidden || k_hidden != hidden || v_hidden != hidden {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let bf16 = backend.bf16_packed_full_attn_qkv_weights_enabled
        && q_weight_t.dtype() == kiln_tensor::DType::BF16
        && k_weight_t.dtype() == kiln_tensor::DType::BF16
        && v_weight_t.dtype() == kiln_tensor::DType::BF16;
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let (q_b, k_b, v_b) = if batch == 1 {
        if bf16 {
            let q_buf = backend.cached_bf16_packed_weight_buffer_kt(q_weight_t)?;
            let k_buf = backend.cached_bf16_packed_weight_buffer_kt(k_weight_t)?;
            let v_buf = backend.cached_bf16_packed_weight_buffer_kt(v_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_bf16_weights_bytes(
                vk_device, &x_data, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
            )
        } else {
            let q_buf = backend.cached_f32_weight_buffer_kt(q_weight_t)?;
            let k_buf = backend.cached_f32_weight_buffer_kt(k_weight_t)?;
            let v_buf = backend.cached_f32_weight_buffer_kt(v_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_bytes(
                vk_device, &x_data, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
            )
        }
        .context("full_attn_qkv_decode kernel failed")?
    } else if bf16 {
        let q_buf = backend.cached_bf16_packed_weight_buffer_kt(q_weight_t)?;
        let k_buf = backend.cached_bf16_packed_weight_buffer_kt(k_weight_t)?;
        let v_buf = backend.cached_bf16_packed_weight_buffer_kt(v_weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_batched_bf16_weights_bytes(
            vk_device, &x_data, &q_buf, &k_buf, &v_buf, batch, hidden, q_dim, k_dim, v_dim,
        )
        .context("full_attn_qkv_decode_batched_bf16w kernel failed")?
    } else {
        let q_buf = backend.cached_f32_weight_buffer_kt(q_weight_t)?;
        let k_buf = backend.cached_f32_weight_buffer_kt(k_weight_t)?;
        let v_buf = backend.cached_f32_weight_buffer_kt(v_weight_t)?;
        kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_batched_bytes(
            vk_device, &x_data, &q_buf, &k_buf, &v_buf, batch, hidden, q_dim, k_dim, v_dim,
        )
        .context("full_attn_qkv_decode_batched kernel failed")?
    };
    Ok(Some((
        kt_tensor_from_f32_bytes(&q_b, &[batch, 1, q_dim], kiln_tensor::DType::F32)?,
        kt_tensor_from_f32_bytes(&k_b, &[batch, 1, k_dim], kiln_tensor::DType::F32)?,
        kt_tensor_from_f32_bytes(&v_b, &[batch, 1, v_dim], kiln_tensor::DType::F32)?,
    )))
}

pub(super) fn mlp_gate_up_decode(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    gate_weight_t: &kiln_tensor::Tensor,
    up_weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.mlp_gate_up_enabled || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(gate_weight_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(up_weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) kt-native: shapes off the kt tensors, weight buffers keyed
    // on the stable kt id (upload once), x bytes straight from kt storage.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((gate_hidden, intermediate)) = gate_weight_t.dims2() else {
        return Ok(None);
    };
    let Ok((up_hidden, up_intermediate)) = up_weight_t.dims2() else {
        return Ok(None);
    };
    if gate_hidden != hidden || up_hidden != hidden || up_intermediate != intermediate {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let gate_buf = backend.cached_f32_weight_buffer_kt(gate_weight_t)?;
    let up_buf = backend.cached_f32_weight_buffer_kt(up_weight_t)?;
    let row_count = batch * seq_len;
    let dispatch_x = if seq_len == 1 {
        x.clone()
    } else {
        x.reshape((row_count, 1usize, hidden))?
    };
    let x_data = kt_tensor_to_f32_bytes_with_shape(&dispatch_x)?.0;
    let out_data = kiln_vulkan_kernel::kernels::dispatch_mlp_gate_up_decode_cached_bytes(
        vk_device,
        &x_data,
        row_count,
        hidden,
        intermediate,
        &gate_buf,
        &up_buf,
    )
    .context("mlp_gate_up_decode kernel failed")?;
    let out = kt_tensor_from_f32_bytes(
        &out_data,
        &[row_count, 1, intermediate],
        kiln_tensor::DType::F32,
    )?;
    let out = if seq_len == 1 {
        out
    } else {
        out.reshape((batch, seq_len, intermediate))?
    };
    Ok(Some(out))
}

pub(super) fn mlp_decode(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    gate_weight_t: &kiln_tensor::Tensor,
    up_weight_t: &kiln_tensor::Tensor,
    down_weight_t: &kiln_tensor::Tensor,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !backend.has_vulkan() || !backend.mlp_decode_enabled || x.dtype() != kiln_tensor::DType::F32
    {
        return Ok(None);
    }
    if !matches!(x.device(), kiln_tensor::Device::Cpu)
        || !matches!(gate_weight_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(up_weight_t.device(), kiln_tensor::Device::Cpu)
        || !matches!(down_weight_t.device(), kiln_tensor::Device::Cpu)
    {
        return Ok(None);
    }
    // (#1082) Fully kt-native: shapes off the kt tensors, weight buffers
    // keyed on the stable kt id (upload once), x bytes straight from kt.
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return Ok(None);
    };
    let Ok((gate_hidden, intermediate)) = gate_weight_t.dims2() else {
        return Ok(None);
    };
    let Ok((up_hidden, up_intermediate)) = up_weight_t.dims2() else {
        return Ok(None);
    };
    let Ok((down_intermediate, out_dim)) = down_weight_t.dims2() else {
        return Ok(None);
    };
    if gate_hidden != hidden
        || up_hidden != hidden
        || up_intermediate != intermediate
        || down_intermediate != intermediate
    {
        return Ok(None);
    }

    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
    let row_count = batch * seq_len;
    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let use_bf16_mlp_weights = backend.bf16_packed_mlp_decode_weights_enabled
        && gate_weight_t.dtype() == kiln_tensor::DType::BF16
        && up_weight_t.dtype() == kiln_tensor::DType::BF16
        && down_weight_t.dtype() == kiln_tensor::DType::BF16;
    let out_data =
        if row_count >= 8 && backend.mlp_bf16_gate_up_f32_down_enabled && use_bf16_mlp_weights {
            let gate_buf = backend.cached_bf16_packed_weight_buffer_kt(gate_weight_t)?;
            let up_buf = backend.cached_bf16_packed_weight_buffer_kt(up_weight_t)?;
            let down_buf = backend.cached_f32_weight_buffer_kt(down_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_gate_up_f32_down_bytes(
                vk_device,
                &x_data,
                row_count,
                &gate_buf,
                &up_buf,
                &down_buf,
                hidden,
                intermediate,
                out_dim,
            )
            .context("mlp_decode kernel failed")?
        } else if use_bf16_mlp_weights {
            let gate_buf = backend.cached_bf16_packed_weight_buffer_kt(gate_weight_t)?;
            let up_buf = backend.cached_bf16_packed_weight_buffer_kt(up_weight_t)?;
            let down_buf = backend.cached_bf16_packed_weight_buffer_kt(down_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_weights_bytes(
                vk_device,
                &x_data,
                row_count,
                &gate_buf,
                &up_buf,
                &down_buf,
                hidden,
                intermediate,
                out_dim,
            )
            .context("mlp_decode kernel failed")?
        } else {
            let gate_buf = backend.cached_f32_weight_buffer_kt(gate_weight_t)?;
            let up_buf = backend.cached_f32_weight_buffer_kt(up_weight_t)?;
            let down_buf = backend.cached_f32_weight_buffer_kt(down_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bytes(
                vk_device,
                &x_data,
                row_count,
                &gate_buf,
                &up_buf,
                &down_buf,
                hidden,
                intermediate,
                out_dim,
            )
            .context("mlp_decode kernel failed")?
        };
    Ok(Some(kt_tensor_from_f32_bytes(
        &out_data,
        &[batch, seq_len, out_dim],
        kiln_tensor::DType::F32,
    )?))
}
