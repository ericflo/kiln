//! Batched 3D matmul for `VkTensor`.
//!
//! `vk_matmul_batched(a, b)` with `a: [B, M, K]` and `b: [B, K, N]`
//! produces `out: [B, M, N]`. Each batch is independent. Backward
//! computes `dA = grad @ B.T_batch` and `dB = A.T_batch @ grad`.

use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n * 4).max(4);
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes as u64,
    )
    .context("vk_matmul_batched: alloc")?;
    Ok(Arc::new(buf))
}

fn dispatch_3d(
    device: &VulkanDevice,
    shader: &str,
    handles: &[vk::Buffer],
    push: &[u32],
    workgroup: (u32, u32, u32),
) -> Result<()> {
    let glsl_path = format!(
        "{}/csrc/shaders/{}.comp",
        env!("CARGO_MANIFEST_DIR"),
        shader
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(&glsl_path)
        .with_context(|| format!("vk_ops: shader compile/load for {shader}"))?;
    crate::kernels::run_compute_pipeline_3d(
        device,
        &spirv,
        handles,
        handles.len(),
        push,
        workgroup,
    )
    .with_context(|| format!("vk_ops: 3d dispatch {shader}"))
}

fn dispatch_matmul_batched(
    device: &VulkanDevice,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let wg_x = ((n + 15) / 16) as u32;
    let wg_y = ((m + 15) / 16) as u32;
    let wg_z = batch as u32;
    let push = [batch as u32, m as u32, n as u32, k as u32];
    dispatch_3d(
        device,
        "vk_matmul_batched_f32",
        &[a.handle(), b.handle(), out.handle()],
        &push,
        (wg_x, wg_y, wg_z),
    )
}

fn dispatch_transpose_3d(
    device: &VulkanDevice,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
    batch: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let wg_x = ((cols + 15) / 16) as u32;
    let wg_y = ((rows + 15) / 16) as u32;
    let wg_z = batch as u32;
    let push = [batch as u32, rows as u32, cols as u32];
    dispatch_3d(
        device,
        "vk_transpose_3d_f32",
        &[src.handle(), dst.handle()],
        &push,
        (wg_x, wg_y, wg_z),
    )
}

fn check_batched_matmul(a: &VkTensor, b: &VkTensor) -> Result<(usize, usize, usize, usize)> {
    anyhow::ensure!(
        a.shape().len() == 3 && b.shape().len() == 3,
        "vk_matmul_batched: rank-3 required, got {:?}/{:?}",
        a.shape(),
        b.shape()
    );
    anyhow::ensure!(
        a.dtype() == VkDType::F32 && b.dtype() == VkDType::F32,
        "vk_matmul_batched: F32-only"
    );
    let ba = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];
    let bb = b.shape()[0];
    let kk = b.shape()[1];
    let n = b.shape()[2];
    anyhow::ensure!(ba == bb, "batch mismatch: {ba} vs {bb}");
    anyhow::ensure!(k == kk, "inner-dim mismatch: {k} vs {kk}");
    Ok((ba, m, n, k))
}

pub fn vk_matmul_batched_no_grad(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    let (batch, m, n, k) = check_batched_matmul(a, b)?;
    let out = alloc_f32(a.device(), batch * m * n)?;
    dispatch_matmul_batched(a.device(), a.buffer(), b.buffer(), &out, batch, m, n, k)?;
    Ok(VkTensor::from_buffer(
        out,
        vec![batch, m, n],
        VkDType::F32,
        Arc::clone(a.device()),
    ))
}

pub fn vk_transpose_batched_2d_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(
        t.shape().len() == 3,
        "vk_transpose_batched_2d: rank-3 required, got {:?}",
        t.shape()
    );
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_transpose_batched_2d: F32 only");
    let batch = t.shape()[0];
    let rows = t.shape()[1];
    let cols = t.shape()[2];
    let out = alloc_f32(t.device(), batch * rows * cols)?;
    dispatch_transpose_3d(t.device(), t.buffer(), &out, batch, rows, cols)?;
    Ok(VkTensor::from_buffer(
        out,
        vec![batch, cols, rows],
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

#[derive(Debug)]
pub struct MatmulBatchedBackward {
    pub inputs: [VkTensor; 2],
}

impl VkBackwardOp for MatmulBatchedBackward {
    fn op_name(&self) -> &'static str {
        "matmul_batched"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let a = &self.inputs[0];
        let b = &self.inputs[1];
        // dA[b] = grad_out[b] @ B[b].T  (per batch)
        let b_t = vk_transpose_batched_2d_no_grad(b)?;
        let grad_a = vk_matmul_batched_no_grad(grad_out, &b_t)?;
        // dB[b] = A[b].T @ grad_out[b]
        let a_t = vk_transpose_batched_2d_no_grad(a)?;
        let grad_b = vk_matmul_batched_no_grad(&a_t, grad_out)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
}

pub fn vk_matmul_batched(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    let out = vk_matmul_batched_no_grad(a, b)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if a.requires_grad() || b.requires_grad() {
        Some(Arc::new(MatmulBatchedBackward {
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

#[derive(Debug)]
pub struct TransposeBatched2dBackward {
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for TransposeBatched2dBackward {
    fn op_name(&self) -> &'static str {
        "transpose_batched_2d"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // d(x.T)/dx applied to grad: transpose grad back.
        Ok(vec![Some(vk_transpose_batched_2d_no_grad(grad_out)?)])
    }
}

pub fn vk_transpose_batched_2d(t: &VkTensor) -> Result<VkTensor> {
    let out = vk_transpose_batched_2d_no_grad(t)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(TransposeBatched2dBackward {
            inputs: [t.clone()],
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
