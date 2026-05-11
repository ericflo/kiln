//! Matrix multiply for `VkTensor`.
//!
//! `vk_matmul(a, b)` with `a: [M, K]`, `b: [K, N]` returns `out: [M, N]`.
//! Higher-rank inputs are flattened to 2D first by the caller (matmul
//! over a batched-by-rows leading dim is a reshape from `[..., K]` to
//! `[prod(...) , K]`).
//!
//! Backward computes both gradients:
//!   dA = dC @ B.T   (shape [M, K])
//!   dB = A.T @ dC   (shape [K, N])
//! Each is another matmul, dispatched by this same module. The
//! transposes are physical via `vk_transpose_2d_no_grad`.

use crate::vk_ops::dispatch_simple_2d;
use crate::vk_ops::shape::vk_transpose_2d_no_grad;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn dispatch_matmul_f32(
    device: &VulkanDevice,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    out: &VulkanBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    anyhow::ensure!(
        m > 0 && n > 0 && k > 0,
        "vk_matmul: zero dimension (M={m}, N={n}, K={k})"
    );
    let wg_x = ((n + 15) / 16) as u32;
    let wg_y = ((m + 15) / 16) as u32;
    let lx = device.max_compute_work_group_count(0);
    let ly = device.max_compute_work_group_count(1);
    anyhow::ensure!(
        wg_x <= lx && wg_y <= ly,
        "vk_matmul: workgroups ({wg_x},{wg_y}) > device limits ({lx},{ly})"
    );
    let push_constants = [m as u32, n as u32, k as u32];
    dispatch_simple_2d(
        device,
        "vk_matmul_f32",
        &[a.handle(), b.handle(), out.handle()],
        &push_constants,
        (wg_x, wg_y),
    )
}

fn alloc_f32(device: &Arc<VulkanDevice>, n_elements: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n_elements * 4).max(4);
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes as u64,
    )
    .context("vk_matmul: alloc output buffer")?;
    Ok(Arc::new(buf))
}

fn check_matmul_shapes(a: &VkTensor, b: &VkTensor) -> Result<(usize, usize, usize)> {
    anyhow::ensure!(
        a.shape().len() == 2 && b.shape().len() == 2,
        "vk_matmul: both inputs must be rank-2 (got {:?} and {:?})",
        a.shape(),
        b.shape()
    );
    anyhow::ensure!(
        a.dtype() == VkDType::F32 && b.dtype() == VkDType::F32,
        "vk_matmul: Phase B is F32-only (got {:?} / {:?})",
        a.dtype(),
        b.dtype()
    );
    let m = a.shape()[0];
    let k = a.shape()[1];
    let kk = b.shape()[0];
    let n = b.shape()[1];
    anyhow::ensure!(
        k == kk,
        "vk_matmul: inner-dim mismatch: a.K={k}, b.K={kk}"
    );
    Ok((m, n, k))
}

pub fn vk_matmul_no_grad(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    let (m, n, k) = check_matmul_shapes(a, b)?;
    let out = alloc_f32(a.device(), m * n)?;
    dispatch_matmul_f32(a.device(), a.buffer(), b.buffer(), &out, m, n, k)?;
    Ok(VkTensor::from_buffer(
        out,
        vec![m, n],
        VkDType::F32,
        Arc::clone(a.device()),
    ))
}

#[derive(Debug)]
pub struct MatmulBackward {
    pub inputs: [VkTensor; 2],
}

impl VkBackwardOp for MatmulBackward {
    fn op_name(&self) -> &'static str {
        "matmul"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let a = &self.inputs[0];
        let b = &self.inputs[1];
        // dA = grad_out @ B.T  → shape [M, K]
        let b_t = vk_transpose_2d_no_grad(b)?;
        let grad_a = vk_matmul_no_grad(grad_out, &b_t)?;
        // dB = A.T @ grad_out  → shape [K, N]
        let a_t = vk_transpose_2d_no_grad(a)?;
        let grad_b = vk_matmul_no_grad(&a_t, grad_out)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
}

pub fn vk_matmul(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    let out = vk_matmul_no_grad(a, b)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> =
        if a.requires_grad() || b.requires_grad() {
            Some(Arc::new(MatmulBackward {
                inputs: [a.clone(), b.clone()],
            }))
        } else {
            None
        };
    Ok(VkTensor::from_op(
        Arc::clone(out.buffer()),
        out.shape().to_vec(),
        out.dtype(),
        Arc::clone(out.device()),
        grad_fn,
    ))
}
