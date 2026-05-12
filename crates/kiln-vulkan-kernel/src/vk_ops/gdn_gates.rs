//! GDN gates: β = sigmoid(b), g = -exp(A_log) · softplus(a + dt_bias).
//!
//! Wraps the existing inference shader `gdn_gates.comp`.
//!
//! Phase 2 ships the forward wrapper. The autograd-aware variant
//! (with `GatesBackward`) lands in Phase 4 alongside
//! `vk_gdn_gates_bwd.comp`.

use crate::vk_ops::dispatch_simple;
use crate::vk_ops::reduce::vk_zeros_like;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

/// Forward GDN gates.
///
/// Shapes:
///   a:          [B, T, nv]   F32
///   b:          [B, T, nv]   F32
///   a_log:      [nv]         F32
///   dt_bias:    [nv]         F32
/// Returns:
///   beta_out:   [B, T, nv]   F32  (= sigmoid(b))
///   g_out:      [B, T, nv]   F32  (= -exp(a_log) · softplus(a + dt_bias))
pub fn vk_gdn_gates_no_grad(
    a: &VkTensor,
    b: &VkTensor,
    a_log: &VkTensor,
    dt_bias: &VkTensor,
    nv: usize,
) -> Result<(VkTensor, VkTensor)> {
    anyhow::ensure!(a.dtype() == VkDType::F32, "vk_gdn_gates: F32 only");
    anyhow::ensure!(b.dtype() == VkDType::F32, "vk_gdn_gates: F32 only");
    anyhow::ensure!(a_log.dtype() == VkDType::F32, "vk_gdn_gates: F32 only");
    anyhow::ensure!(dt_bias.dtype() == VkDType::F32, "vk_gdn_gates: F32 only");
    anyhow::ensure!(
        a.num_elements() == b.num_elements(),
        "vk_gdn_gates: a/b element-count mismatch"
    );
    anyhow::ensure!(
        a_log.num_elements() == nv,
        "vk_gdn_gates: a_log size {} != nv {}",
        a_log.num_elements(),
        nv
    );
    anyhow::ensure!(
        dt_bias.num_elements() == nv,
        "vk_gdn_gates: dt_bias size {} != nv {}",
        dt_bias.num_elements(),
        nv
    );
    let total = a.num_elements();
    anyhow::ensure!(
        total % nv == 0,
        "vk_gdn_gates: total {} not divisible by nv {}",
        total,
        nv
    );

    let device = a.device();
    let beta_out = alloc_f32(device, total)?;
    let g_out = alloc_f32(device, total)?;

    let workgroups = ((total + 255) / 256) as u32;
    let push = [total as u32, nv as u32];
    dispatch_simple(
        device,
        "gdn_gates",
        &[
            a.buffer().handle(),
            b.buffer().handle(),
            a_log.buffer().handle(),
            dt_bias.buffer().handle(),
            beta_out.handle(),
            g_out.handle(),
        ],
        &push,
        workgroups,
    )?;

    Ok((
        VkTensor::from_buffer(
            beta_out,
            a.shape().to_vec(),
            VkDType::F32,
            Arc::clone(device),
        ),
        VkTensor::from_buffer(g_out, a.shape().to_vec(), VkDType::F32, Arc::clone(device)),
    ))
}

/// Backward GDN gates.
///
/// Inputs:
///   d_beta, d_g: [B, T, nv]
///   a, b:        [B, T, nv]
///   a_log, dt_bias: [nv]
/// Returns:
///   d_a, d_b: [B, T, nv]
///   d_a_log, d_dt_bias: [nv]   (sum-reduced along B,T per nv)
pub fn vk_gdn_gates_bwd_no_grad(
    d_beta: &VkTensor,
    d_g: &VkTensor,
    a: &VkTensor,
    b: &VkTensor,
    a_log: &VkTensor,
    dt_bias: &VkTensor,
    nv: usize,
) -> Result<(VkTensor, VkTensor, VkTensor, VkTensor)> {
    let total = a.num_elements();
    anyhow::ensure!(total % nv == 0, "gates_bwd: total not divisible by nv");

    let device = a.device();
    let d_a = alloc_f32(device, total)?;
    let d_b = alloc_f32(device, total)?;
    let red_dalog_buf = alloc_f32(device, total)?;
    let red_ddt_buf = alloc_f32(device, total)?;

    let workgroups = ((total + 255) / 256) as u32;
    let push = [total as u32, nv as u32];
    crate::vk_ops::dispatch_simple(
        device,
        "vk_gdn_gates_bwd",
        &[
            d_beta.buffer().handle(),
            d_g.buffer().handle(),
            a.buffer().handle(),
            b.buffer().handle(),
            a_log.buffer().handle(),
            dt_bias.buffer().handle(),
            d_a.handle(),
            d_b.handle(),
            red_dalog_buf.handle(),
            red_ddt_buf.handle(),
        ],
        &push,
        workgroups,
    )?;

    // GPU reduce per-nv: partials are laid out [outer, nv] (innermost
    // dim is nv; the CPU loop did `n = i % nv`). Reduce via a 1×outer
    // ones-row matmul: ones[1, outer] @ partial[outer, nv] = [1, nv].
    // Avoids the prior 2 × total readback per layer per training step.
    let outer = total / nv;
    let dalog_partial_t = VkTensor::from_buffer(
        Arc::clone(&red_dalog_buf),
        vec![outer, nv],
        VkDType::F32,
        Arc::clone(device),
    );
    let ddt_partial_t = VkTensor::from_buffer(
        Arc::clone(&red_ddt_buf),
        vec![outer, nv],
        VkDType::F32,
        Arc::clone(device),
    );
    let ones_buf = alloc_f32(device, outer)?;
    let push_fill = [outer as u32, 1.0_f32.to_bits()];
    crate::vk_ops::dispatch_simple(
        device,
        "vk_fill_f32",
        &[ones_buf.handle()],
        &push_fill,
        ((outer + 255) / 256) as u32,
    )?;
    let ones_t = VkTensor::from_buffer(ones_buf, vec![1, outer], VkDType::F32, Arc::clone(device));
    let dalog_2d = crate::vk_ops::matmul::vk_matmul_no_grad(&ones_t, &dalog_partial_t)?;
    let ddt_2d = crate::vk_ops::matmul::vk_matmul_no_grad(&ones_t, &ddt_partial_t)?;
    let dalog_t = crate::vk_ops::shape::vk_reshape(&dalog_2d, &[nv])?;
    let ddt_t = crate::vk_ops::shape::vk_reshape(&ddt_2d, &[nv])?;

    Ok((
        VkTensor::from_buffer(d_a, a.shape().to_vec(), VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_b, a.shape().to_vec(), VkDType::F32, Arc::clone(device)),
        dalog_t,
        ddt_t,
    ))
}

#[derive(Clone, Copy, Debug)]
enum GateOutput {
    Beta,
    G,
}

#[derive(Debug)]
struct GdnGatesBackward {
    output: GateOutput,
    nv: usize,
    inputs: [VkTensor; 4],
}

impl VkBackwardOp for GdnGatesBackward {
    fn op_name(&self) -> &'static str {
        "gdn_gates"
    }

    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let zero = vk_zeros_like(&self.inputs[0])?;
        let (d_beta, d_g) = match self.output {
            GateOutput::Beta => (grad_out.clone(), zero),
            GateOutput::G => (zero, grad_out.clone()),
        };
        let (d_a, d_b, d_a_log, d_dt_bias) = vk_gdn_gates_bwd_no_grad(
            &d_beta,
            &d_g,
            &self.inputs[0],
            &self.inputs[1],
            &self.inputs[2],
            &self.inputs[3],
            self.nv,
        )?;
        Ok(vec![Some(d_a), Some(d_b), Some(d_a_log), Some(d_dt_bias)])
    }
}

/// Autograd-aware GDN gates. Returns `(beta, g)`.
///
/// The two returned tensors attach separate backward nodes. If both outputs
/// feed the loss, their input gradients are accumulated by `vk_backward`.
pub fn vk_gdn_gates(
    a: &VkTensor,
    b: &VkTensor,
    a_log: &VkTensor,
    dt_bias: &VkTensor,
    nv: usize,
) -> Result<(VkTensor, VkTensor)> {
    let (beta, g) = vk_gdn_gates_no_grad(a, b, a_log, dt_bias, nv)?;
    let needs_grad =
        a.requires_grad() || b.requires_grad() || a_log.requires_grad() || dt_bias.requires_grad();
    if !needs_grad {
        return Ok((beta, g));
    }
    let inputs = [a.clone(), b.clone(), a_log.clone(), dt_bias.clone()];
    let beta_grad: Arc<dyn VkBackwardOp> = Arc::new(GdnGatesBackward {
        output: GateOutput::Beta,
        nv,
        inputs: inputs.clone(),
    });
    let g_grad: Arc<dyn VkBackwardOp> = Arc::new(GdnGatesBackward {
        output: GateOutput::G,
        nv,
        inputs,
    });
    Ok((
        VkTensor::from_op(
            Arc::clone(beta.buffer()),
            beta.shape().to_vec(),
            beta.dtype(),
            Arc::clone(beta.device()),
            Some(beta_grad),
        ),
        VkTensor::from_op(
            Arc::clone(g.buffer()),
            g.shape().to_vec(),
            g.dtype(),
            Arc::clone(g.device()),
            Some(g_grad),
        ),
    ))
}
