//! Causal Conv1d (depthwise) for vk-native GDN.
//!
//! Wraps the existing inference shader `causal_conv1d.comp`:
//!   out[b, c, t] = silu(Σ_k weight[c, k] · padded[b, c, t + k])
//! where `padded = concat(conv_state, x)` along time.
//!
//! Phase 2 adds the forward wrapper only. The autograd-aware variant
//! that attaches a `Conv1dBackward` ships in Phase 4 alongside the
//! `vk_causal_conv1d_bwd.comp` shader.

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
///
/// Shapes:
///   x:           [batch, channels, seq_len]   F32
///   weight:      [channels, kernel_size]      F32
///   conv_state:  [batch, channels, kernel_size - 1]  F32
/// Returns:
///   out:         [batch, channels, seq_len]   F32
///
/// `conv_state` is read-only here. Advancing the state for the next
/// chunk (when training in chunkwise mode) is a separate dispatch
/// (`causal_conv1d_state_advance.comp`) that the GDN layer wires up.
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
