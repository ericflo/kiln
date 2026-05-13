//! FLCE forward + backward parity vs naive CPU cross-entropy.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::flce::{
    flce_recommended_chunk_len_from_limits, vk_flce_loss, vk_grpo_loss, vk_selected_log_probs,
};
use kiln_vulkan_kernel::vk_tensor::VkTensor;
use kiln_vulkan_kernel::VulkanDevice;
use std::sync::Arc;

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

#[test]
fn flce_recommended_chunk_len_is_dynamic() {
    unsafe {
        std::env::remove_var("KILN_VK_FLCE_CHUNK_LEN");
    }

    let short_context =
        flce_recommended_chunk_len_from_limits(512, 2560, 248_320, 24 * 1024 * 1024 * 1024, 65_535);
    let long_context = flce_recommended_chunk_len_from_limits(
        65_536,
        2560,
        248_320,
        24 * 1024 * 1024 * 1024,
        65_535,
    );
    let dispatch_limited = flce_recommended_chunk_len_from_limits(65_536, 2560, 248_320, 0, 64);

    assert!(
        long_context < short_context,
        "long contexts should reduce FLCE chunk size without env tuning: short={short_context}, long={long_context}"
    );
    assert!(
        dispatch_limited < long_context,
        "device dispatch limits should further constrain chunk size: dispatch_limited={dispatch_limited}, long={long_context}"
    );
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

fn upload_bf16(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?.to_dtype(DType::BF16)?;
    VkTensor::from_candle(&t, Arc::clone(dev))
}

fn bf16_rounded(data: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    Ok(
        Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
    )
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

fn cpu_selected_log_probs_and_grpo(
    hidden: &[f32],
    weight: &[f32],
    labels: &[u32],
    ref_log_probs: &[f32],
    advantage: f32,
    clip_epsilon: f32,
    kl_coeff: f32,
    num_active: usize,
    hidden_dim: usize,
    vocab: usize,
) -> (Vec<f32>, f32, Vec<f32>) {
    let mut log_probs = vec![0.0_f32; num_active];
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
        let policy_logprob = logits[label] - lse;
        log_probs[n] = policy_logprob;

        let log_ratio = policy_logprob - ref_log_probs[n];
        let ratio = log_ratio.exp();
        let lo = 1.0 - clip_epsilon;
        let hi = 1.0 + clip_epsilon;
        let clipped_ratio = ratio.clamp(lo, hi);
        let surr1 = ratio * advantage;
        let surr2 = clipped_ratio * advantage;
        let take_surr1 = surr1 <= surr2;
        let surrogate = if take_surr1 { surr1 } else { surr2 };
        loss_sum += -surrogate + kl_coeff * log_ratio;

        let d_clipped = if ratio >= lo && ratio <= hi {
            ratio
        } else {
            0.0
        };
        let d_surrogate = if take_surr1 {
            advantage * ratio
        } else {
            advantage * d_clipped
        };
        let coeff = -d_surrogate + kl_coeff;
        for v in 0..vocab {
            let p = (logits[v] - mx).exp() / z;
            let onehot = if v == label { 1.0 } else { 0.0 };
            let g = coeff * (onehot - p) / num_active as f32;
            for d in 0..hidden_dim {
                grad_hidden[n * hidden_dim + d] += g * weight[v * hidden_dim + d];
            }
        }
    }
    (log_probs, loss_sum / num_active as f32, grad_hidden)
}

#[allow(clippy::too_many_arguments)]
fn candle_selected_log_probs_and_grpo(
    hidden: &[f32],
    weight: &[f32],
    labels: &[u32],
    ref_log_probs: &[f32],
    advantage: f32,
    clip_epsilon: f32,
    kl_coeff: f32,
    num_active: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<(Vec<f32>, f32, Vec<f32>)> {
    use candle_core::{DType, D};

    let dev = Device::Cpu;
    let hidden_var = Var::from_tensor(&Tensor::from_vec(
        hidden.to_vec(),
        (num_active, hidden_dim),
        &dev,
    )?)?;
    let hidden_t = hidden_var.as_tensor();
    let weight_t = Tensor::from_vec(weight.to_vec(), (vocab, hidden_dim), &dev)?;
    let logits = hidden_t.matmul(&weight_t.transpose(0, 1)?)?;
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let log_sum_exp = logits_f32.log_sum_exp(D::Minus1)?;
    let labels_2d = Tensor::new(labels, &dev)?
        .to_dtype(DType::U32)?
        .unsqueeze(1)?;
    let correct_logits = logits_f32.gather(&labels_2d, 1)?.squeeze(1)?;
    let policy_log_probs = (correct_logits - log_sum_exp)?;

    let ref_t = Tensor::new(ref_log_probs, &dev)?.to_dtype(DType::F32)?;
    let log_ratio = (&policy_log_probs - &ref_t)?;
    let ratio = log_ratio.exp()?;
    let ratio_shape = ratio.shape().clone();
    let lo = Tensor::new(1.0_f32 - clip_epsilon, &dev)?.broadcast_as(&ratio_shape)?;
    let hi = Tensor::new(1.0_f32 + clip_epsilon, &dev)?.broadcast_as(&ratio_shape)?;
    let clipped_ratio = ratio.clamp(&lo, &hi)?;
    let adv_t = Tensor::new(advantage, &dev)?.broadcast_as(&ratio_shape)?;
    let surr1 = (&ratio * &adv_t)?;
    let surr2 = (&clipped_ratio * &adv_t)?;
    let surrogate = surr1.minimum(&surr2)?;
    let neg_surrogate = surrogate.neg()?;
    let kl_penalty = log_ratio.affine(kl_coeff as f64, 0.0)?;
    let loss = (&neg_surrogate + &kl_penalty)?.mean_all()?;
    let loss_val = loss.to_scalar::<f32>()?;
    let grads = loss.backward()?;
    let grad_hidden = grads
        .get(hidden_t)
        .expect("candle d hidden")
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((policy_log_probs.to_vec1::<f32>()?, loss_val, grad_hidden))
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

#[test]
fn vk_selected_logprob_and_grpo_parity_small() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let num_active = 4;
    let hidden_dim = 7;
    let vocab = 19;
    let chunk = 5;
    let h_data: Vec<f32> = (0..(num_active * hidden_dim))
        .map(|i| ((i as f32) * 0.11).sin() * 0.25)
        .collect();
    let w_data: Vec<f32> = (0..(vocab * hidden_dim))
        .map(|i| ((i as f32) * 0.017).cos() * 0.4)
        .collect();
    let labels: Vec<u32> = vec![3, 12, 0, 18];
    let ref_log_probs = vec![-2.8_f32, -3.1, -2.3, -3.4];
    let advantage = 0.7_f32;
    let clip_epsilon = 0.2_f32;
    let kl_coeff = 0.05_f32;

    let (_hv, hidden) = upload_param_f32(&dev, &h_data, &[num_active, hidden_dim])?;
    let weight = upload_f32(&dev, &w_data, &[vocab, hidden_dim])?;
    let ref_vk = upload_f32(&dev, &ref_log_probs, &[num_active])?;

    let selected = vk_selected_log_probs(&hidden, &weight, &labels, chunk)?.to_vec_f32()?;
    let loss = vk_grpo_loss(
        &hidden,
        &weight,
        &labels,
        &ref_vk,
        advantage,
        clip_epsilon,
        kl_coeff,
        chunk,
    )?;

    let (exp_log_probs, exp_loss, exp_dh) = cpu_selected_log_probs_and_grpo(
        &h_data,
        &w_data,
        &labels,
        &ref_log_probs,
        advantage,
        clip_epsilon,
        kl_coeff,
        num_active,
        hidden_dim,
        vocab,
    );
    let logp_mad = selected
        .iter()
        .zip(exp_log_probs.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(logp_mad < 1e-4, "selected logprob mad {logp_mad}");

    let got_loss = loss.to_vec_f32()?[0];
    assert!(
        (got_loss - exp_loss).abs() < 1e-4,
        "grpo loss {} vs {}",
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
    assert!(mad < 1e-4, "grpo d_hidden mad {mad}");

    let (candle_log_probs, candle_loss, candle_dh) = candle_selected_log_probs_and_grpo(
        &h_data,
        &w_data,
        &labels,
        &ref_log_probs,
        advantage,
        clip_epsilon,
        kl_coeff,
        num_active,
        hidden_dim,
        vocab,
    )?;
    let candle_logp_mad = selected
        .iter()
        .zip(candle_log_probs.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        candle_logp_mad < 1e-4,
        "selected logprob vs candle mad {candle_logp_mad}"
    );
    assert!(
        (got_loss - candle_loss).abs() < 1e-4,
        "grpo loss {got_loss} vs candle {candle_loss}"
    );
    let candle_mad = grad_h
        .iter()
        .zip(candle_dh.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        candle_mad < 1e-4,
        "grpo d_hidden vs candle mad {candle_mad}"
    );
    Ok(())
}

#[test]
fn vk_selected_logprob_and_grpo_parity_bf16_lm_head() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let num_active = 4;
    let hidden_dim = 8;
    let vocab = 18;
    let chunk = 5;
    let h_data: Vec<f32> = (0..(num_active * hidden_dim))
        .map(|i| ((i as f32) * 0.09).sin() * 0.2)
        .collect();
    let w_f32: Vec<f32> = (0..(vocab * hidden_dim))
        .map(|i| ((i as f32) * 0.019).cos() * 0.35)
        .collect();
    let w_data = bf16_rounded(&w_f32, &[vocab, hidden_dim])?;
    let labels: Vec<u32> = vec![3, 12, 0, 17];
    let ref_log_probs = vec![-2.8_f32, -3.1, -2.3, -3.4];
    let advantage = 0.7_f32;
    let clip_epsilon = 0.2_f32;
    let kl_coeff = 0.05_f32;

    let (_hv, hidden) = upload_param_f32(&dev, &h_data, &[num_active, hidden_dim])?;
    let weight = upload_bf16(&dev, &w_f32, &[vocab, hidden_dim])?;
    let ref_vk = upload_f32(&dev, &ref_log_probs, &[num_active])?;

    let selected = vk_selected_log_probs(&hidden, &weight, &labels, chunk)?.to_vec_f32()?;
    let loss = vk_grpo_loss(
        &hidden,
        &weight,
        &labels,
        &ref_vk,
        advantage,
        clip_epsilon,
        kl_coeff,
        chunk,
    )?;

    let (exp_log_probs, exp_loss, exp_dh) = cpu_selected_log_probs_and_grpo(
        &h_data,
        &w_data,
        &labels,
        &ref_log_probs,
        advantage,
        clip_epsilon,
        kl_coeff,
        num_active,
        hidden_dim,
        vocab,
    );
    let logp_mad = selected
        .iter()
        .zip(exp_log_probs.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(logp_mad < 1e-3, "BF16 selected logprob mad {logp_mad}");

    let got_loss = loss.to_vec_f32()?[0];
    assert!(
        (got_loss - exp_loss).abs() < 1e-3,
        "BF16 grpo loss {} vs {}",
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
    assert!(mad < 1e-3, "BF16 grpo d_hidden mad {mad}");
    Ok(())
}
