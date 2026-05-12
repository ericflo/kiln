//! FLCE forward + backward parity vs naive CPU cross-entropy.

use anyhow::Result;
use candle_core::{Device, Tensor, Var};
use kiln_vulkan_kernel::VulkanDevice;
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::flce::vk_flce_loss;
use kiln_vulkan_kernel::vk_tensor::VkTensor;
use std::sync::Arc;

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

fn upload_param_f32(
    dev: &Arc<VulkanDevice>,
    data: &[f32],
    shape: &[usize],
) -> Result<(Var, VkTensor)> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    let var = Var::from_tensor(&t)?;
    let vk = VkTensor::from_candle(&t, Arc::clone(dev))?;
    let pid = var.id();
    let param = VkTensor::parameter(
        Arc::clone(vk.buffer()),
        vk.shape().to_vec(),
        vk.dtype(),
        Arc::clone(vk.device()),
        pid,
    );
    Ok((var, param))
}

fn upload_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    VkTensor::from_candle(&t, Arc::clone(dev))
}

/// CPU cross-entropy reference: loss = mean_i (-log(softmax(logit_i)[label_i]))
/// where logits = hidden @ weight.T
fn cpu_xent(
    hidden: &[f32],
    weight: &[f32],
    labels: &[u32],
    num_active: usize,
    hidden_dim: usize,
    vocab: usize,
) -> (f32, Vec<f32>) {
    // logits[n, v] = sum_d hidden[n, d] * weight[v, d]
    let mut loss_sum = 0.0_f32;
    let mut grad_hidden = vec![0.0_f32; num_active * hidden_dim];
    for n in 0..num_active {
        let mut logits = vec![0.0_f32; vocab];
        for v in 0..vocab {
            let mut s = 0.0_f32;
            for d in 0..hidden_dim {
                s += hidden[n * hidden_dim + d] * weight[v * hidden_dim + d];
            }
            logits[v] = s;
        }
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut z = 0.0_f32;
        for v in 0..vocab {
            z += (logits[v] - mx).exp();
        }
        let lse = mx + z.ln();
        let label = labels[n] as usize;
        loss_sum += lse - logits[label];

        // dL/dlogit[v] = (softmax[v] - 1{v==label}) / num_active
        for v in 0..vocab {
            let p = (logits[v] - mx).exp() / z;
            let g = (p - if v == label { 1.0 } else { 0.0 }) / num_active as f32;
            // d hidden[n, d] += g * weight[v, d]
            for d in 0..hidden_dim {
                grad_hidden[n * hidden_dim + d] += g * weight[v * hidden_dim + d];
            }
        }
    }
    (loss_sum / num_active as f32, grad_hidden)
}

#[test]
fn vk_flce_forward_backward_parity_small() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let num_active = 4;
    let hidden_dim = 8;
    let vocab = 17; // intentionally not aligned to chunk_len
    let chunk = 6;
    let h_data: Vec<f32> = (0..(num_active * hidden_dim))
        .map(|i| ((i as f32) * 0.07).sin() * 0.3)
        .collect();
    let w_data: Vec<f32> = (0..(vocab * hidden_dim))
        .map(|i| ((i as f32) * 0.013).cos() * 0.5)
        .collect();
    let labels: Vec<u32> = vec![3, 11, 0, 16];

    let (_hv, hidden) = upload_param_f32(&dev, &h_data, &[num_active, hidden_dim])?;
    let weight = upload_f32(&dev, &w_data, &[vocab, hidden_dim])?;
    let loss = vk_flce_loss(&hidden, &weight, &labels, chunk)?;

    let (exp_loss, exp_dh) = cpu_xent(&h_data, &w_data, &labels, num_active, hidden_dim, vocab);
    let got_loss = loss.to_vec_f32()?[0];
    assert!(
        (got_loss - exp_loss).abs() < 1e-3,
        "loss {} vs {}",
        got_loss,
        exp_loss
    );

    let grads = vk_backward(&loss)?;
    let grad_h = grads
        .get(hidden.param_id().unwrap())
        .expect("d hidden")
        .to_vec_f32()?;
    let mad = grad_h
        .iter()
        .zip(exp_dh.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(mad < 1e-4, "flce d_hidden mad {mad}");
    Ok(())
}
