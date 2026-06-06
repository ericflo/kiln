//! Vulkan causal conv1d operation helpers.
//!
//! Keeps the plain conv1d update/prefill hooks out of the `BackendRuntime`
//! facade while preserving the kt-native byte extraction and explicit Vulkan
//! dispatch boundary.

use anyhow::{Context, Result};

use super::vulkan::VulkanBackend;
use super::vulkan_tensor_bridge::{kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape};

pub(super) fn supports_causal_conv1d_update(backend: &VulkanBackend) -> bool {
    // Single-token update still regresses Strix Halo decode latency.
    backend.has_vulkan() && backend.fused_conv1d_update_enabled
}

pub(super) fn supports_causal_conv1d_prefill(backend: &VulkanBackend) -> bool {
    backend.has_vulkan() && backend.fused_conv1d_prefill_enabled
}

pub(super) fn causal_conv1d_update(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state_kt: &mut kiln_tensor::Tensor,
    kernel_size: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !supports_causal_conv1d_update(backend) {
        return Ok(None);
    }
    if !matches!(
        x.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1082) kt-native: all args are already kt; `conv_state_kt` is
    // mutated in place at the return below.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

    let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
    let weight_data = kt_tensor_to_f32_bytes_with_shape(weight)?.0;
    let state_data = kt_tensor_to_f32_bytes_with_shape(conv_state_kt)?.0;
    let dims = x.dims();
    anyhow::ensure!(
        dims.len() == 3,
        "causal_conv1d_update: x must be 3-D, got {:?}",
        dims
    );
    let (batch, channels, seq_len) = (dims[0], dims[1], dims[2]);
    let conv_state_shape = conv_state_kt.dims().to_vec();
    let (out_data, state_data_out) =
        kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_update_bytes(
            vk_device,
            &x_data,
            &weight_data,
            &state_data,
            batch,
            channels,
            seq_len,
            kernel_size,
        )
        .context("causal_conv1d_update kernel failed")?;
    let out_shape: Vec<usize> = dims.to_vec();
    let out = kt_tensor_from_f32_bytes(&out_data, &out_shape, kiln_tensor::DType::F32)?;
    *conv_state_kt =
        kt_tensor_from_f32_bytes(&state_data_out, &conv_state_shape, kiln_tensor::DType::F32)?;
    Ok(Some(out))
}

pub(super) fn causal_conv1d_prefill(
    backend: &VulkanBackend,
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state_kt: &mut kiln_tensor::Tensor,
    kernel_size: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    // kt guards read directly off the kt args before the bridge.
    if !supports_causal_conv1d_prefill(backend) {
        return Ok(None);
    }
    if !matches!(
        x.dtype(),
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
    ) {
        return Ok(None);
    }
    // (#1082) kt-native: all args are already kt; `conv_state_kt` is
    // mutated in place at the return below.
    let vk_device = backend
        .vulkan_device()
        .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

    let (out, new_state) = if backend.conv1d_prefill_single_submit_enabled {
        let weight_buf = backend.cached_f32_weight_buffer_kt(weight)?;
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let state_data = kt_tensor_to_f32_bytes_with_shape(conv_state_kt)?.0;
        let x_dims = x.dims();
        let (batch, channels, seq_len) = (x_dims[0], x_dims[1], x_dims[2]);
        let conv_state_dims = conv_state_kt.dims().to_vec();
        let (out_data, new_state_data) =
            kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill_cached_weight_bytes(
                vk_device,
                &x_data,
                &weight_buf,
                &state_data,
                batch,
                channels,
                seq_len,
                kernel_size,
            )
            .context("causal_conv1d_prefill cached-weight single-submit kernel failed")?;
        let out = kt_tensor_from_f32_bytes(&out_data, x_dims, kiln_tensor::DType::F32)?;
        let new_state =
            kt_tensor_from_f32_bytes(&new_state_data, &conv_state_dims, kiln_tensor::DType::F32)?;
        (out, new_state)
    } else {
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let weight_data = kt_tensor_to_f32_bytes_with_shape(weight)?.0;
        let state_data = kt_tensor_to_f32_bytes_with_shape(conv_state_kt)?.0;
        let x_dims = x.dims();
        let (batch, channels, seq_len) = (x_dims[0], x_dims[1], x_dims[2]);
        let conv_state_dims = conv_state_kt.dims().to_vec();
        let (out_data, new_state_data) =
            kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill_bytes(
                vk_device,
                &x_data,
                &weight_data,
                &state_data,
                batch,
                channels,
                seq_len,
                kernel_size,
            )
            .context("causal_conv1d_prefill kernel failed")?;
        let out = kt_tensor_from_f32_bytes(&out_data, x_dims, kiln_tensor::DType::F32)?;
        let new_state =
            kt_tensor_from_f32_bytes(&new_state_data, &conv_state_dims, kiln_tensor::DType::F32)?;
        (out, new_state)
    };
    *conv_state_kt = new_state;
    Ok(Some(out))
}
