//! Causal Conv1d (depthwise) for vk-native GDN.
//!
//! Wraps the existing inference shader `causal_conv1d.comp`:
//!   out[b, c, t] = silu(Σ_k weight[c, k] · padded[b, c, t + k])
//! where `padded = concat(conv_state, x)` along time.
//!
//! Phase 4 adds a backward kernel for the linear-conv part. SiLU
//! backward is handled separately by `vk_silu` autograd. The current
//! forward fuses SiLU; for training the autograd-aware path
//! `vk_causal_conv1d_pre_silu_no_grad` is added (TODO Phase 4).
//! For now we expose:
//!   - `vk_causal_conv1d_no_grad`: forward with fused SiLU (matches inference)
//!   - `vk_causal_conv1d_bwd_no_grad`: backward of the linear conv only
//!     (assumes caller already applied silu_bwd to dout)

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n * 4).max(4);
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes as u64,
    )
    .context("vk_causal_conv1d: alloc f32")?;
    Ok(Arc::new(buf))
}

/// Forward causal Conv1d (depthwise) with fused SiLU activation.
pub fn vk_causal_conv1d_no_grad(
    x: &VkTensor,
    weight: &VkTensor,
    conv_state: &Arc<VulkanBuffer>,
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<VkTensor> {
    anyhow::ensure!(x.dtype() == VkDType::F32, "vk_causal_conv1d: F32 only");
    anyhow::ensure!(weight.dtype() == VkDType::F32, "vk_causal_conv1d: F32 weight only");
    anyhow::ensure!(
        x.num_elements() == batch * channels * seq_len,
        "vk_causal_conv1d: x size mismatch"
    );
    anyhow::ensure!(
        weight.num_elements() == channels * kernel_size,
        "vk_causal_conv1d: weight size mismatch"
    );
    let device = x.device();
    let out_n = batch * channels * seq_len;
    let out = alloc_f32(device, out_n)?;

    let total = (batch * channels * seq_len) as u32;
    let workgroups = (total + 255) / 256;
    let push = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    dispatch_simple(
        device,
        "causal_conv1d",
        &[
            x.buffer().handle(),
            weight.buffer().handle(),
            conv_state.handle(),
            out.handle(),
        ],
        &push,
        workgroups,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        vec![batch, channels, seq_len],
        VkDType::F32,
        Arc::clone(device),
    ))
}

/// Backward causal Conv1d (linear part only — caller has applied
/// silu_backward to d_out before calling this, OR is testing the linear
/// kernel in isolation).
///
/// Inputs:
///   d_out:      [batch, channels, seq_len] F32  (= silu_bwd(d_silu_out))
///   weight:     [channels, kernel_size]    F32
///   x:          [batch, channels, seq_len] F32  (saved input, for d_weight)
///   conv_state: [batch, channels, K-1]     F32  (saved state, for d_weight + d_conv_state)
/// Outputs:
///   d_x:           [batch, channels, seq_len] F32
///   d_weight:      [channels, kernel_size]    F32  (CPU reduce — small)
///   d_conv_state:  [batch, channels, K-1]     F32  (CPU compute — small)
pub fn vk_causal_conv1d_bwd_no_grad(
    d_out: &VkTensor,
    weight: &VkTensor,
    x: &VkTensor,
    conv_state: &Arc<VulkanBuffer>,
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<(VkTensor, VkTensor, Arc<VulkanBuffer>)> {
    let state_len = kernel_size - 1;
    let device = d_out.device();

    // d_x via GPU shader
    let d_x_buf = alloc_f32(device, batch * channels * seq_len)?;
    let total = (batch * channels * seq_len) as u32;
    let workgroups = (total + 255) / 256;
    let push = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    dispatch_simple(
        device,
        "vk_causal_conv1d_bwd",
        &[
            d_out.buffer().handle(),
            weight.buffer().handle(),
            d_x_buf.handle(),
        ],
        &push,
        workgroups,
    )?;
    let d_x = VkTensor::from_buffer(
        d_x_buf,
        vec![batch, channels, seq_len],
        VkDType::F32,
        Arc::clone(device),
    );

    // d_weight + d_conv_state via CPU readback (sizes are small:
    // d_weight: [C*K], d_conv_state: [B*C*(K-1)])
    let dout_data = d_out.to_vec_f32()?;
    let x_data = x.to_vec_f32()?;
    let weight_data = weight.to_vec_f32()?;
    let cs_t = VkTensor::from_buffer(
        Arc::clone(conv_state),
        vec![batch, channels, state_len],
        VkDType::F32,
        Arc::clone(device),
    );
    let cs_data = cs_t.to_vec_f32()?;

    let mut d_weight = vec![0.0_f32; channels * kernel_size];
    let mut d_conv_state = vec![0.0_f32; batch * channels * state_len];

    for b in 0..batch {
        for c in 0..channels {
            for t in 0..seq_len {
                let dout_v = dout_data[(b * channels + c) * seq_len + t];
                for k in 0..kernel_size {
                    let logical_t = t + k;
                    if logical_t < state_len {
                        d_conv_state[(b * channels + c) * state_len + logical_t] +=
                            dout_v * weight_data[c * kernel_size + k];
                        d_weight[c * kernel_size + k] +=
                            dout_v * cs_data[(b * channels + c) * state_len + logical_t];
                    } else {
                        d_weight[c * kernel_size + k] +=
                            dout_v * x_data[(b * channels + c) * seq_len + (logical_t - state_len)];
                    }
                }
            }
        }
    }

    let dw_buf = alloc_f32(device, channels * kernel_size)?;
    let raw_dw: Vec<u8> = d_weight.iter().flat_map(|f| f.to_le_bytes()).collect();
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &dw_buf,
        &raw_dw,
    )?;
    let dw = VkTensor::from_buffer(
        dw_buf,
        vec![channels, kernel_size],
        VkDType::F32,
        Arc::clone(device),
    );

    let dcs_buf = alloc_f32(device, batch * channels * state_len)?;
    let raw_dcs: Vec<u8> = d_conv_state.iter().flat_map(|f| f.to_le_bytes()).collect();
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &dcs_buf,
        &raw_dcs,
    )?;

    Ok((d_x, dw, dcs_buf))
}

