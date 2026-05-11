//! End-to-end vk-native training smoke test.
//!
//! Builds a tiny synthetic transformer (1 layer, hidden=32, 2 heads,
//! head_dim=16, intermediate=64, vocab=64), wires up VkLoraLayer,
//! runs `vk_train_step` 5 times on a fixed input, and asserts that
//! loss strictly decreases. This proves the entire vk-native
//! pipeline works end-to-end:
//!
//!   embed → rmsnorm → q/k/v + LoRA → attention → o + LoRA → residual
//!         → rmsnorm → mlp(gate/up/silu/mul/down) + LoRA → residual
//!         → final_rmsnorm → FLCE loss
//!         → vk_backward (per-param VkTensor grads)
//!         → dispatch_adamw_step_f32 (in-place GPU update)
//!
//! Skips cleanly if no Vulkan device is available.

#![cfg(feature = "vulkan")]

use anyhow::Result;
use candle_core::{Device, Tensor};
use kiln_model::vk_forward::{
    vk_model_forward_loss, VkLayerWeights, VkLoraLayer, VkLoraPair, VkModelWeights,
};
use kiln_train::vk_train::{
    allocate_adamw_state, vk_train_step, VkAdamWConfig,
};
use kiln_vulkan_kernel::{VkDType, VkTensor, VulkanDevice};
use std::sync::Arc;

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

fn upload_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    VkTensor::from_candle(&t, Arc::clone(dev))
}

fn linspace(n: usize, start: f32, end: f32) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / ((n - 1) as f32);
    (0..n).map(|i| start + step * (i as f32)).collect()
}

fn small_random(n: usize, seed: u64) -> Vec<f32> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(-0.1_f32..0.1_f32)).collect()
}

fn build_tiny_model(dev: &Arc<VulkanDevice>) -> Result<VkModelWeights> {
    let vocab = 32;
    let hidden = 32;
    let intermediate = 64;
    let heads_q = 2;
    let heads_kv = 1; // GQA 2:1
    let head_dim = hidden / heads_q;
    let kv_dim = heads_kv * head_dim;

    let embed = small_random(vocab * hidden, 1);
    let final_norm = vec![0.0_f32; hidden]; // (1 + w) — start at identity
    let lm_head = small_random(vocab * hidden, 99);

    // Per-layer
    let in_norm = vec![0.0_f32; hidden];
    let post_norm = vec![0.0_f32; hidden];
    let q = small_random(hidden * hidden, 2);
    let k = small_random(kv_dim * hidden, 3);
    let v = small_random(kv_dim * hidden, 4);
    let o = small_random(hidden * hidden, 5);
    let gate = small_random(intermediate * hidden, 6);
    let up = small_random(intermediate * hidden, 7);
    let down = small_random(hidden * intermediate, 8);

    let layer = VkLayerWeights {
        input_layernorm_weight: upload_f32(dev, &in_norm, &[hidden])?,
        post_attention_layernorm_weight: upload_f32(dev, &post_norm, &[hidden])?,
        q_proj: upload_f32(dev, &q, &[hidden, hidden])?,
        k_proj: upload_f32(dev, &k, &[kv_dim, hidden])?,
        v_proj: upload_f32(dev, &v, &[kv_dim, hidden])?,
        o_proj: upload_f32(dev, &o, &[hidden, hidden])?,
        gate_proj: upload_f32(dev, &gate, &[intermediate, hidden])?,
        up_proj: upload_f32(dev, &up, &[intermediate, hidden])?,
        down_proj: upload_f32(dev, &down, &[hidden, intermediate])?,
        heads_q,
        heads_kv,
        head_dim,
        eps: 1e-5,
    };
    Ok(VkModelWeights {
        embed_tokens: upload_f32(dev, &embed, &[vocab, hidden])?,
        embed_dtype: VkDType::F32,
        final_norm_weight: upload_f32(dev, &final_norm, &[hidden])?,
        lm_head: upload_f32(dev, &lm_head, &[vocab, hidden])?,
        layers: vec![layer],
        vocab,
        hidden,
    })
}

fn build_lora_layer(dev: &Arc<VulkanDevice>, hidden: usize, kv_dim: usize, intermediate: usize) -> Result<VkLoraLayer> {
    let rank = 4;
    let alpha = 8.0;
    Ok(VkLoraLayer {
        q_proj: Some(VkLoraPair::init_kaiming(dev, hidden, hidden, rank, alpha, 100)?),
        k_proj: Some(VkLoraPair::init_kaiming(dev, hidden, kv_dim, rank, alpha, 101)?),
        v_proj: Some(VkLoraPair::init_kaiming(dev, hidden, kv_dim, rank, alpha, 102)?),
        o_proj: Some(VkLoraPair::init_kaiming(dev, hidden, hidden, rank, alpha, 103)?),
        gate_proj: Some(VkLoraPair::init_kaiming(dev, hidden, intermediate, rank, alpha, 104)?),
        up_proj: Some(VkLoraPair::init_kaiming(dev, hidden, intermediate, rank, alpha, 105)?),
        down_proj: Some(VkLoraPair::init_kaiming(dev, intermediate, hidden, rank, alpha, 106)?),
    })
}

#[test]
fn vk_full_model_one_layer_grads_exist() -> Result<()> {
    use kiln_model::vk_forward::{vk_transformer_layer, VkLoraPair};
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::flce::vk_flce_loss;
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_model(&dev)?;
    let kv_dim = model.layers[0].heads_kv * model.layers[0].head_dim;
    let lora = build_lora_layer(&dev, model.hidden, kv_dim, 64)?;
    // Use a synthetic hidden state as input (rather than embedding) so we
    // can verify the per-layer chain in isolation.
    let rows = 4;
    let h_data: Vec<f32> = (0..(rows * model.hidden))
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let h0 = upload_f32(&dev, &h_data, &[rows, model.hidden])?;
    let layer_out = vk_transformer_layer(&h0, &model.layers[0], &lora)?;
    println!("layer_out requires_grad={}", layer_out.requires_grad());
    let h_post_norm = vk_rmsnorm(&layer_out, &model.final_norm_weight, 1e-5)?;
    println!("h_post_norm requires_grad={}", h_post_norm.requires_grad());
    let labels: Vec<u32> = vec![0, 1, 2, 3];
    let loss = vk_flce_loss(&h_post_norm, &model.lm_head, &labels, 8)?;
    println!("loss requires_grad={}", loss.requires_grad());
    let grads = vk_backward(&loss)?;
    println!("FULL_LAYER: grads has {} entries", grads.len());
    for (pid, g) in grads.iter() {
        let v = g.to_vec_f32()?;
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        println!("  pid={pid:?} shape={:?} max_abs={max_abs:.6e}", g.shape());
    }
    assert!(grads.len() >= 1, "expected at least one param grad");
    Ok(())
}

#[test]
fn vk_flce_through_sdpa_chain() -> Result<()> {
    use kiln_model::vk_forward::VkLoraPair;
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::attention::vk_sdpa_prefill_flat;
    use kiln_vulkan_kernel::vk_ops::flce::vk_flce_loss;
    use kiln_vulkan_kernel::vk_ops::matmul::vk_matmul;
    use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d;
    let Some(dev) = vk_dev() else { return Ok(()) };
    // hidden = sdpa(x @ Aq.T @ Bq.T, x @ Ak.T @ Bk.T, x @ Av.T @ Bv.T)
    // Build minimal: rows=3, heads_q=2, heads_kv=1, head_dim=4, hidden=8
    let rows = 3;
    let heads_q = 2;
    let heads_kv = 1;
    let head_dim = 4;
    let hidden_q = heads_q * head_dim; // 8
    let hidden_kv = heads_kv * head_dim; // 4
    let x_data: Vec<f32> = (0..(rows * hidden_q)).map(|i| (i as f32) * 0.05).collect();
    let x = upload_f32(&dev, &x_data, &[rows, hidden_q])?;

    // Q LoRA: hidden_q → hidden_q
    let pair_q = VkLoraPair::init_kaiming(&dev, hidden_q, hidden_q, 4, 1.0, 7)?;
    let q = vk_matmul(
        &vk_matmul(&x, &vk_transpose_2d(&pair_q.a)?)?,
        &vk_transpose_2d(&pair_q.b)?,
    )?;
    // K LoRA: hidden_q → hidden_kv
    let pair_k = VkLoraPair::init_kaiming(&dev, hidden_q, hidden_kv, 4, 1.0, 8)?;
    let k = vk_matmul(
        &vk_matmul(&x, &vk_transpose_2d(&pair_k.a)?)?,
        &vk_transpose_2d(&pair_k.b)?,
    )?;
    // V LoRA
    let pair_v = VkLoraPair::init_kaiming(&dev, hidden_q, hidden_kv, 4, 1.0, 9)?;
    let v = vk_matmul(
        &vk_matmul(&x, &vk_transpose_2d(&pair_v.a)?)?,
        &vk_transpose_2d(&pair_v.b)?,
    )?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn = vk_sdpa_prefill_flat(&q, &k, &v, heads_q, heads_kv, head_dim, scale)?;
    // attn shape [rows, hidden_q]
    let w_data: Vec<f32> = (0..(20 * hidden_q)).map(|i| (i as f32) * 0.01).collect();
    let weight = upload_f32(&dev, &w_data, &[20, hidden_q])?;
    let labels: Vec<u32> = vec![1, 5, 11];
    let loss = vk_flce_loss(&attn, &weight, &labels, 8)?;
    let grads = vk_backward(&loss)?;
    println!("FLCE_THROUGH_SDPA: grads has {} entries", grads.len());
    for (pid, g) in grads.iter() {
        let v = g.to_vec_f32()?;
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        println!("  pid={pid:?} shape={:?} max_abs={max_abs:.6e}", g.shape());
    }
    assert!(grads.len() >= 1);
    Ok(())
}

#[test]
fn vk_flce_through_rmsnorm_then_matmul() -> Result<()> {
    use kiln_model::vk_forward::VkLoraPair;
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::flce::vk_flce_loss;
    use kiln_vulkan_kernel::vk_ops::matmul::vk_matmul;
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;
    use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let pair = VkLoraPair::init_kaiming(&dev, 8, 16, 4, 1.0, 7)?;
    let x_data: Vec<f32> = (0..(3 * 8)).map(|i| (i as f32) * 0.05).collect();
    let x = upload_f32(&dev, &x_data, &[3, 8])?;
    let a_t = vk_transpose_2d(&pair.a)?;
    let h_inner = vk_matmul(&x, &a_t)?;
    let b_t = vk_transpose_2d(&pair.b)?;
    let hidden_pre_norm = vk_matmul(&h_inner, &b_t)?; // [3, 16]
    let norm_w = upload_f32(&dev, &vec![0.0_f32; 16], &[16])?;
    let hidden = vk_rmsnorm(&hidden_pre_norm, &norm_w, 1e-5)?;
    let w_data: Vec<f32> = (0..(20 * 16)).map(|i| (i as f32) * 0.01).collect();
    let weight = upload_f32(&dev, &w_data, &[20, 16])?;
    let labels: Vec<u32> = vec![1, 5, 11];
    let loss = vk_flce_loss(&hidden, &weight, &labels, 8)?;
    let grads = vk_backward(&loss)?;
    println!("FLCE_THROUGH_RMSNORM: grads has {} entries", grads.len());
    for (pid, g) in grads.iter() {
        let v = g.to_vec_f32()?;
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        println!("  pid={pid:?} shape={:?} max_abs={max_abs:.6e}", g.shape());
    }
    assert!(grads.len() >= 1);
    Ok(())
}

#[test]
fn vk_flce_propagates_to_param_through_matmul() -> Result<()> {
    use kiln_model::vk_forward::VkLoraPair;
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::flce::vk_flce_loss;
    use kiln_vulkan_kernel::vk_ops::matmul::vk_matmul;
    use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d;
    let Some(dev) = vk_dev() else { return Ok(()) };
    // hidden = x @ A.T  (where A is a param)
    // loss = flce(hidden, weight, labels)
    // pair.a: [rank=4, in=8]; pair.b: [out=16, rank=4]
    let pair = VkLoraPair::init_kaiming(&dev, 8, 16, 4, 1.0, 7)?;
    let x_data: Vec<f32> = (0..(3 * 8)).map(|i| (i as f32) * 0.05).collect();
    let x = upload_f32(&dev, &x_data, &[3, 8])?;
    let a_t = vk_transpose_2d(&pair.a)?; // [8, 4]
    let h_inner = vk_matmul(&x, &a_t)?; // [3, 4]
    let b_t = vk_transpose_2d(&pair.b)?; // [4, 16]
    let hidden = vk_matmul(&h_inner, &b_t)?; // [3, 16]
    let w_data: Vec<f32> = (0..(20 * 16)).map(|i| (i as f32) * 0.01).collect();
    let weight = upload_f32(&dev, &w_data, &[20, 16])?;
    let labels: Vec<u32> = vec![1, 5, 11];
    let loss = vk_flce_loss(&hidden, &weight, &labels, 8)?;
    let grads = vk_backward(&loss)?;
    println!("FLCE_THROUGH_MATMUL: grads has {} entries", grads.len());
    for (pid, g) in grads.iter() {
        let v = g.to_vec_f32()?;
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        println!("  pid={pid:?} shape={:?} max_abs={max_abs:.6e}", g.shape());
    }
    assert!(grads.len() >= 1, "expected at least one param grad");
    Ok(())
}

#[test]
fn vk_minimal_lora_grad_exists() -> Result<()> {
    use kiln_model::vk_forward::VkLoraPair;
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
    use kiln_vulkan_kernel::vk_ops::matmul::vk_matmul;
    use kiln_vulkan_kernel::vk_ops::reduce::vk_mean_all;
    use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d;
    let Some(dev) = vk_dev() else { return Ok(()) };
    // y = x @ A.T @ B.T;  loss = mean(y * y)
    // Even with B = 0 init, dB should be nonzero.
    let pair = VkLoraPair::init_kaiming(&dev, 8, 4, 2, 1.0, 7)?;
    let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let x = upload_f32(&dev, &x_data, &[2, 8])?;
    let a_t = vk_transpose_2d(&pair.a)?;
    let h = vk_matmul(&x, &a_t)?;
    let b_t = vk_transpose_2d(&pair.b)?;
    let y = vk_matmul(&h, &b_t)?;
    let sq = vk_mul(&y, &y)?;
    let loss = vk_mean_all(&sq)?;
    let grads = vk_backward(&loss)?;
    println!("MINIMAL: grads has {} entries", grads.len());
    assert!(grads.len() >= 1, "expected at least one param grad");
    Ok(())
}

#[test]
fn vk_native_training_loss_decreases() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_model(&dev)?;
    let lora_layers = vec![build_lora_layer(&dev, model.hidden, 16, 64)?];
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };

    // Tiny synthetic input: 8 tokens.
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];

    let initial_loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?
        .to_vec_f32()?[0];
    // Debug: see how many params get gradients on a fresh backward.
    {
        use kiln_model::vk_forward::vk_step_backward;
        let loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?;
        let grads = vk_step_backward(&loss)?;
        println!("DEBUG: grads has {} entries", grads.len());
        for (pid, g) in grads.iter() {
            let v = g.to_vec_f32()?;
            let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let nonzero = v.iter().filter(|x| x.abs() > 1e-9).count();
            println!(
                "  pid={pid:?} shape={:?} max_abs={max_abs:.6e} nonzero={nonzero}/{}",
                g.shape(),
                v.len()
            );
        }
    }
    let mut last_loss = initial_loss;
    let mut losses = vec![initial_loss];
    for step in 1..=10 {
        let l = vk_train_step(&model, &lora_layers, &input_ids, &mut adamw, &cfg, step)?;
        assert!(l.is_finite(), "step {step}: non-finite loss {l}");
        losses.push(l);
        last_loss = l;
    }
    println!("vk_native_training losses: {losses:?}");
    let _ = linspace;
    assert!(
        last_loss < initial_loss * 0.95,
        "loss did not drop meaningfully: {initial_loss} -> {last_loss}"
    );
    Ok(())
}
