//! FLCE forward + backward parity vs naive CPU cross-entropy.
//!
//! Test factories are candle-free via the kt-native
//! `VkTensor::from_f32_slice` / `from_f32_slice_as_bf16` /
//! `parameter_from_f32_slice` constructors.
//!
//! Fully candle-free: the second GRPO gradient oracle
//! (`fd_selected_log_probs_and_grpo`) cross-checks the Vulkan backward
//! against a finite-difference numerical gradient of the exact same
//! scalar GRPO loss the kernel optimizes, replacing the former candle
//! autograd reference. (#1082)

use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::VulkanDevice;
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::flce::{
    VK_GRPO_IS_MODE_CISPO, VK_GRPO_KL_MODE_K3, VK_GRPO_KL_MODE_NONE,
    flce_recommended_chunk_len_from_limits, vk_flce_loss, vk_grpo_backward_with_saved_state,
    vk_grpo_loss, vk_grpo_loss_with_saved_state_ext, vk_grpo_selected_log_probs_from_saved_state,
    vk_selected_log_probs,
};
use kiln_vulkan_kernel::vk_tensor::VkTensor;
use std::sync::Arc;

mod support;

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    support::vulkan_device_arc("vk_flce_parity")
}

#[test]
fn flce_recommended_chunk_len_is_dynamic() {
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

fn upload_param_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    VkTensor::parameter_from_f32_slice(data, shape.to_vec(), Arc::clone(dev))
}

fn upload_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    VkTensor::from_f32_slice(data, shape.to_vec(), Arc::clone(dev))
}

fn upload_bf16(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    VkTensor::from_f32_slice_as_bf16(data, shape.to_vec(), Arc::clone(dev))
}

fn bf16_rounded(data: &[f32], _shape: &[usize]) -> Result<Vec<f32>> {
    Ok(data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect())
}

#[derive(serde::Deserialize)]
struct GrpoTrlOracleFixture {
    schema: String,
    oracle: GrpoTrlOracleIdentity,
    tolerances: GrpoTrlOracleTolerances,
    cases: Vec<GrpoTrlOracleCase>,
}

#[derive(serde::Deserialize)]
struct GrpoTrlOracleIdentity {
    trl_commit: String,
    trl_grpo_trainer_sha256: String,
}

#[derive(serde::Deserialize)]
struct GrpoTrlOracleTolerances {
    loss_abs: f64,
    gradient_abs: f64,
}

#[derive(serde::Deserialize)]
struct GrpoTrlOracleCase {
    name: String,
    policy_log_probs: Vec<f32>,
    behavior_log_probs: Vec<f32>,
    kl_reference_log_probs: Vec<f32>,
    advantage: f32,
    clip_low: f32,
    cispo_max_weight: Option<f32>,
    kl_coeff: f32,
    is_level: String,
    loss_normalizer: f32,
    expected: GrpoTrlOracleExpected,
}

#[derive(serde::Deserialize)]
struct GrpoTrlOracleExpected {
    loss: f64,
    mean_k3: f64,
    policy_log_prob_grad: Vec<f64>,
}

fn pinned_grpo_trl_oracle() -> GrpoTrlOracleFixture {
    let fixture: GrpoTrlOracleFixture = serde_json::from_str(include_str!(
        "../../kiln-train/tests/fixtures/grpo_trl_oracle_v1.json"
    ))
    .expect("parse pinned GRPO TRL oracle fixture");
    assert_eq!(fixture.schema, "kiln.grpo-trl-oracle.v1");
    assert_eq!(
        fixture.oracle.trl_commit,
        "95809b942eb5d11d0b06d749510d88be99230b73"
    );
    assert_eq!(
        fixture.oracle.trl_grpo_trainer_sha256,
        "sha256:52d9a6c1e298df35d0da4a6fa17874d750ee627f6ac15393c8860d74d1ba4917"
    );
    fixture
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
        let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
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

// Mirrors the flat ABI of the GPU dispatch under parity test (round-67 GDN-ABI precedent).
#[allow(clippy::too_many_arguments)]
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
        let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
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
fn cpu_grpo_ext(
    hidden: &[f32],
    weight: &[f32],
    labels: &[u32],
    ref_log_probs: &[f32],
    advantage: f32,
    clip_low: f32,
    clip_high: f32,
    kl_coeff: f32,
    kl_mode: u32,
    is_mode: u32,
    num_active: usize,
    hidden_dim: usize,
    vocab: usize,
) -> (f32, Vec<f32>) {
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
        let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
        let lse = mx + z.ln();
        let label = labels[n] as usize;
        let policy_logprob = logits[label] - lse;
        let log_ratio = policy_logprob - ref_log_probs[n];
        let ratio = log_ratio.exp();
        let lo = 1.0 - clip_low;
        let hi = 1.0 + clip_high;
        let clipped_ratio = if is_mode == VK_GRPO_IS_MODE_CISPO {
            ratio.min(clip_high)
        } else {
            ratio.clamp(lo, hi)
        };

        let (kl_penalty, kl_grad) = match kl_mode {
            1 => (kl_coeff * log_ratio, kl_coeff),
            3 => {
                let exp_neg = (-log_ratio).exp();
                (
                    kl_coeff * (exp_neg - 1.0 + log_ratio),
                    kl_coeff * (1.0 - exp_neg),
                )
            }
            _ => (0.0, 0.0),
        };
        let (loss, coeff) = match is_mode {
            1 => {
                let weight = clipped_ratio * advantage;
                (-weight * policy_logprob, -weight)
            }
            2 => (-advantage, -advantage),
            _ => {
                let surr1 = ratio * advantage;
                let surr2 = clipped_ratio * advantage;
                let take_surr1 = surr1 <= surr2;
                let surrogate = if take_surr1 { surr1 } else { surr2 };
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
                (-surrogate, -d_surrogate)
            }
        };
        loss_sum += loss + kl_penalty;

        let coeff = coeff + kl_grad;
        for v in 0..vocab {
            let p = (logits[v] - mx).exp() / z;
            let onehot = if v == label { 1.0 } else { 0.0 };
            let g = coeff * (onehot - p) / num_active as f32;
            for d in 0..hidden_dim {
                grad_hidden[n * hidden_dim + d] += g * weight[v * hidden_dim + d];
            }
        }
    }
    (loss_sum / num_active as f32, grad_hidden)
}

/// Scalar GRPO loss as a pure function of `hidden`, matching the exact
/// loss the kernel optimizes (same formula as `cpu_selected_log_probs_and_grpo`).
/// Used only as the inner function for finite-difference gradients.
#[allow(clippy::too_many_arguments)]
fn grpo_scalar_loss(
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
) -> f32 {
    let mut loss_sum = 0.0_f32;
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
        let z: f32 = logits.iter().map(|l| (l - mx).exp()).sum();
        let lse = mx + z.ln();
        let label = labels[n] as usize;
        let policy_logprob = logits[label] - lse;

        let log_ratio = policy_logprob - ref_log_probs[n];
        let ratio = log_ratio.exp();
        let lo = 1.0 - clip_epsilon;
        let hi = 1.0 + clip_epsilon;
        let clipped_ratio = ratio.clamp(lo, hi);
        let surr1 = ratio * advantage;
        let surr2 = clipped_ratio * advantage;
        let surrogate = if surr1 <= surr2 { surr1 } else { surr2 };
        loss_sum += -surrogate + kl_coeff * log_ratio;
    }
    loss_sum / num_active as f32
}

/// Candle-free gradient oracle: returns the analytic forward log_probs +
/// scalar loss, and a finite-difference (central-difference) numerical
/// gradient of the scalar GRPO loss w.r.t. `hidden`. Replaces the former
/// candle autograd reference. (#1082)
#[allow(clippy::too_many_arguments)]
fn fd_selected_log_probs_and_grpo(
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
    // Forward log_probs + scalar loss via the analytic reference.
    let (log_probs, loss_val, _) = cpu_selected_log_probs_and_grpo(
        hidden,
        weight,
        labels,
        ref_log_probs,
        advantage,
        clip_epsilon,
        kl_coeff,
        num_active,
        hidden_dim,
        vocab,
    );

    // Central finite difference: d loss / d hidden[i] ≈
    //   (L(hidden + eps e_i) - L(hidden - eps e_i)) / (2 eps).
    // eps ~ 1e-3 keeps truncation + f32 rounding both small for this
    // well-scaled, smooth-around-the-clip-region loss.
    let eps = 1e-3_f32;
    let mut grad_hidden = vec![0.0_f32; hidden.len()];
    let mut h = hidden.to_vec();
    for i in 0..h.len() {
        let orig = h[i];
        h[i] = orig + eps;
        let lp = grpo_scalar_loss(
            &h,
            weight,
            labels,
            ref_log_probs,
            advantage,
            clip_epsilon,
            kl_coeff,
            num_active,
            hidden_dim,
            vocab,
        );
        h[i] = orig - eps;
        let lm = grpo_scalar_loss(
            &h,
            weight,
            labels,
            ref_log_probs,
            advantage,
            clip_epsilon,
            kl_coeff,
            num_active,
            hidden_dim,
            vocab,
        );
        h[i] = orig;
        grad_hidden[i] = (lp - lm) / (2.0 * eps);
    }

    (log_probs, loss_val, grad_hidden)
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

    let hidden = upload_param_f32(&dev, &h_data, &[num_active, hidden_dim])?;
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

    let hidden = upload_param_f32(&dev, &h_data, &[num_active, hidden_dim])?;
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

    // Candle-free finite-difference gradient oracle. log_probs + loss are
    // the exact analytic values (tolerance kept at 1e-4); the gradient is
    // a central finite difference of the same scalar GRPO loss, so its
    // tolerance is loosened to 2e-3 to absorb FD truncation/rounding.
    let (fd_log_probs, fd_loss, fd_dh) = fd_selected_log_probs_and_grpo(
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
    let fd_logp_mad = selected
        .iter()
        .zip(fd_log_probs.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        fd_logp_mad < 1e-4,
        "selected logprob vs analytic mad {fd_logp_mad}"
    );
    assert!(
        (got_loss - fd_loss).abs() < 1e-4,
        "grpo loss {got_loss} vs analytic {fd_loss}"
    );
    let fd_mad = grad_h
        .iter()
        .zip(fd_dh.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        fd_mad < 2e-3,
        "grpo d_hidden vs finite-difference mad {fd_mad}"
    );
    Ok(())
}

#[test]
fn vk_grpo_ext_cispo_k3_upper_cap_parity_small() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let num_active = 4;
    let hidden_dim = 7;
    let vocab = 19;
    let chunk = 5;
    let h_data: Vec<f32> = (0..(num_active * hidden_dim))
        .map(|i| ((i as f32) * 0.13).sin() * 0.22)
        .collect();
    let w_data: Vec<f32> = (0..(vocab * hidden_dim))
        .map(|i| ((i as f32) * 0.021).cos() * 0.37)
        .collect();
    let labels: Vec<u32> = vec![3, 12, 0, 18];
    let ref_log_probs = vec![-2.8_f32, -3.1, -2.3, -3.4];
    let advantage = -0.65_f32;
    let clip_low = 0.15_f32;
    let clip_high = 1.35_f32;
    let kl_coeff = 0.04_f32;

    let hidden = upload_f32(&dev, &h_data, &[num_active, hidden_dim])?;
    let weight = upload_f32(&dev, &w_data, &[vocab, hidden_dim])?;
    let ref_vk = upload_f32(&dev, &ref_log_probs, &[num_active])?;
    let (loss, saved) = vk_grpo_loss_with_saved_state_ext(
        &hidden,
        &weight,
        &labels,
        &ref_vk,
        advantage,
        clip_low,
        clip_high,
        kl_coeff,
        VK_GRPO_KL_MODE_K3,
        VK_GRPO_IS_MODE_CISPO,
        chunk,
    )?;
    let grad_seed = upload_f32(&dev, &[1.0], &[1])?;
    let grad_h = vk_grpo_backward_with_saved_state(&hidden, &saved, &grad_seed)?.to_vec_f32()?;
    let saved_selected = vk_grpo_selected_log_probs_from_saved_state(&saved)?.to_vec_f32()?;
    let direct_selected = vk_selected_log_probs(&hidden, &weight, &labels, chunk)?.to_vec_f32()?;
    assert_eq!(saved_selected, direct_selected);

    let (exp_loss, exp_dh) = cpu_grpo_ext(
        &h_data,
        &w_data,
        &labels,
        &ref_log_probs,
        advantage,
        clip_low,
        clip_high,
        kl_coeff,
        VK_GRPO_KL_MODE_K3,
        VK_GRPO_IS_MODE_CISPO,
        num_active,
        hidden_dim,
        vocab,
    );
    let got_loss = loss.to_vec_f32()?[0];
    assert!(
        (got_loss - exp_loss).abs() < 1e-4,
        "CISPO/K3 loss {} vs {}",
        got_loss,
        exp_loss
    );
    let mad = grad_h
        .iter()
        .zip(exp_dh.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(mad < 1e-4, "CISPO/K3 d_hidden mad {mad}");
    Ok(())
}

#[test]
fn vk_grpo_cispo_policy_term_matches_pinned_trl_pytorch_oracle() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let fixture = pinned_grpo_trl_oracle();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.name == "cispo_upper_weight_cap_k3")
        .expect("pinned CISPO oracle case");
    assert_eq!(case.is_level, "cispo");
    let num_active = case.policy_log_probs.len();
    assert_eq!(case.loss_normalizer, 1.0 / num_active as f32);
    assert_eq!(case.behavior_log_probs.len(), num_active);

    // A two-class head with logits [h, 0] can reproduce each selected-token
    // log-probability exactly enough for an independent shader check:
    // log_softmax([h, 0])[0] = log(sigmoid(h)).
    let hidden_data = case
        .policy_log_probs
        .iter()
        .map(|&log_prob| {
            let probability = log_prob.exp();
            (probability / (1.0 - probability)).ln()
        })
        .collect::<Vec<_>>();
    let hidden = upload_f32(&dev, &hidden_data, &[num_active, 1])?;
    let weight = upload_f32(&dev, &[1.0, 0.0], &[2, 1])?;
    let labels = vec![0_u32; num_active];
    let behavior = upload_f32(&dev, &case.behavior_log_probs, &[num_active])?;
    let cap = case.cispo_max_weight.expect("CISPO absolute cap");
    let (loss, saved) = vk_grpo_loss_with_saved_state_ext(
        &hidden,
        &weight,
        &labels,
        &behavior,
        case.advantage,
        case.clip_low,
        cap,
        0.0,
        VK_GRPO_KL_MODE_NONE,
        VK_GRPO_IS_MODE_CISPO,
        2,
    )?;

    let selected = vk_grpo_selected_log_probs_from_saved_state(&saved)?.to_vec_f32()?;
    for (index, (&actual, &expected)) in selected.iter().zip(&case.policy_log_probs).enumerate() {
        assert!(
            (actual - expected).abs() <= fixture.tolerances.loss_abs as f32,
            "selected log-probability {index}: Vulkan={actual}, TRL fixture={expected}"
        );
    }

    let expected_policy_loss =
        case.expected.loss - f64::from(case.kl_coeff) * case.expected.mean_k3;
    let actual_loss = f64::from(loss.to_vec_f32()?[0]);
    assert!(
        (actual_loss - expected_policy_loss).abs() <= fixture.tolerances.loss_abs * 2.0,
        "CISPO policy loss: Vulkan={actual_loss}, TRL fixture={expected_policy_loss}"
    );

    let grad_seed = upload_f32(&dev, &[1.0], &[1])?;
    let hidden_grad =
        vk_grpo_backward_with_saved_state(&hidden, &saved, &grad_seed)?.to_vec_f32()?;
    for (index, grad) in hidden_grad.iter().enumerate().take(num_active) {
        let policy = case.policy_log_probs[index];
        let reference = case.kl_reference_log_probs[index];
        let k3_grad = case.loss_normalizer * case.kl_coeff * (1.0 - (reference - policy).exp());
        let expected_policy_grad = case.expected.policy_log_prob_grad[index] - f64::from(k3_grad);
        let log_prob_jacobian = 1.0 - policy.exp();
        let actual_policy_grad = f64::from(grad / log_prob_jacobian);
        assert!(
            (actual_policy_grad - expected_policy_grad).abs()
                <= fixture.tolerances.gradient_abs * 3.0,
            "CISPO policy gradient {index}: Vulkan={actual_policy_grad}, TRL fixture={expected_policy_grad}"
        );
    }
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

    let hidden = upload_param_f32(&dev, &h_data, &[num_active, hidden_dim])?;
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
