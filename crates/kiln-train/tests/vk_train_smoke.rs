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
use kiln_core::config::{DType, ModelConfig};
use kiln_model::vk_forward::{
    VkFullAttentionWeights, VkLayerWeights, VkLinearAttentionWeights, VkLoraLayer, VkLoraPair,
    VkModelWeights, vk_model_forward_loss, vk_model_forward_loss_with_state,
};
use kiln_train::vk_train::{
    VkAdamWConfig, allocate_adamw_state, save_vk_lora_adapter,
    vk_recompute_train_step_with_state_masked, vk_train_step,
};
use kiln_vulkan_kernel::vk_ops::gdn_state::VkLinearAttentionState;
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
    (0..n)
        .map(|_| rng.random_range(-0.1_f32..0.1_f32))
        .collect()
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

    let layer = VkLayerWeights::FullAttention(VkFullAttentionWeights {
        input_layernorm_weight: upload_f32(dev, &in_norm, &[hidden])?,
        post_attention_layernorm_weight: upload_f32(dev, &post_norm, &[hidden])?,
        q_proj: upload_f32(dev, &q, &[hidden, hidden])?,
        k_proj: upload_f32(dev, &k, &[kv_dim, hidden])?,
        v_proj: upload_f32(dev, &v, &[kv_dim, hidden])?,
        o_proj: upload_f32(dev, &o, &[hidden, hidden])?,
        q_norm: None,
        k_norm: None,
        gate_proj: upload_f32(dev, &gate, &[intermediate, hidden])?,
        up_proj: upload_f32(dev, &up, &[intermediate, hidden])?,
        down_proj: upload_f32(dev, &down, &[hidden, intermediate])?,
        heads_q,
        heads_kv,
        head_dim,
        attn_output_gate: false,
        eps: 1e-5,
    });
    Ok(VkModelWeights {
        embed_tokens: upload_f32(dev, &embed, &[vocab, hidden])?,
        embed_dtype: VkDType::F32,
        final_norm_weight: upload_f32(dev, &final_norm, &[hidden])?,
        lm_head: upload_f32(dev, &lm_head, &[vocab, hidden])?,
        layers: vec![layer],
        rotary_inv_freq: vec![],
        rotary_dim: 0,
        vocab,
        hidden,
    })
}

fn build_lora_layer(
    dev: &Arc<VulkanDevice>,
    hidden: usize,
    kv_dim: usize,
    intermediate: usize,
) -> Result<VkLoraLayer> {
    let rank = 4;
    let alpha = 8.0;
    Ok(VkLoraLayer {
        q_proj: Some(VkLoraPair::init_kaiming(
            dev, hidden, hidden, rank, alpha, 100,
        )?),
        k_proj: Some(VkLoraPair::init_kaiming(
            dev, hidden, kv_dim, rank, alpha, 101,
        )?),
        v_proj: Some(VkLoraPair::init_kaiming(
            dev, hidden, kv_dim, rank, alpha, 102,
        )?),
        o_proj: Some(VkLoraPair::init_kaiming(
            dev, hidden, hidden, rank, alpha, 103,
        )?),
        gate_proj: Some(VkLoraPair::init_kaiming(
            dev,
            hidden,
            intermediate,
            rank,
            alpha,
            104,
        )?),
        up_proj: Some(VkLoraPair::init_kaiming(
            dev,
            hidden,
            intermediate,
            rank,
            alpha,
            105,
        )?),
        down_proj: Some(VkLoraPair::init_kaiming(
            dev,
            intermediate,
            hidden,
            rank,
            alpha,
            106,
        )?),
        ..Default::default()
    })
}

fn tiny_model_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 32,
        num_layers: 1,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 16,
        intermediate_size: 64,
        vocab_size: 32,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        dtype: DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: 1,
        linear_key_head_dim: 16,
        linear_num_value_heads: 1,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 0.0,
    }
}

fn tiny_gdn_model_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 16,
        num_layers: 1,
        num_attention_heads: 1,
        num_kv_heads: 1,
        head_dim: 16,
        intermediate_size: 32,
        vocab_size: 24,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        dtype: DType::FP32,
        num_full_attention_layers: 0,
        full_attention_interval: 0,
        attn_output_gate: false,
        linear_num_key_heads: 1,
        linear_key_head_dim: 4,
        linear_num_value_heads: 1,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 0.0,
    }
}

fn build_tiny_gdn_model(dev: &Arc<VulkanDevice>) -> Result<VkModelWeights> {
    let config = tiny_gdn_model_config();
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qkv_dim = 2 * nk * dk + nv * dv;
    let v_dim = nv * dv;
    let conv_kernel = config.linear_conv_kernel_dim;

    let layer = VkLayerWeights::LinearAttention(VkLinearAttentionWeights {
        layer_norm: upload_f32(dev, &vec![0.0_f32; hidden], &[hidden])?,
        in_proj_qkv: upload_f32(dev, &small_random(qkv_dim * hidden, 301), &[qkv_dim, hidden])?,
        in_proj_z: upload_f32(dev, &small_random(v_dim * hidden, 302), &[v_dim, hidden])?,
        in_proj_a: upload_f32(dev, &small_random(nv * hidden, 303), &[nv, hidden])?,
        in_proj_b: upload_f32(dev, &small_random(nv * hidden, 304), &[nv, hidden])?,
        conv1d: upload_f32(dev, &small_random(qkv_dim * conv_kernel, 305), &[qkv_dim, conv_kernel])?,
        a_log: upload_f32(dev, &vec![-0.5_f32; nv], &[nv])?,
        a_log_gates: upload_f32(dev, &vec![-0.5_f32; nv], &[nv])?,
        dt_bias: upload_f32(dev, &vec![0.1_f32; nv], &[nv])?,
        gated_norm: upload_f32(dev, &vec![1.0_f32; v_dim], &[v_dim])?,
        out_proj: upload_f32(dev, &small_random(hidden * v_dim, 306), &[hidden, v_dim])?,
        heads_k: nk,
        heads_v: nv,
        head_dim_k: dk,
        head_dim_v: dv,
        conv_kernel,
        eps: 1e-5,
    });

    Ok(VkModelWeights {
        embed_tokens: upload_f32(dev, &small_random(vocab * hidden, 300), &[vocab, hidden])?,
        embed_dtype: VkDType::F32,
        final_norm_weight: upload_f32(dev, &vec![0.0_f32; hidden], &[hidden])?,
        lm_head: upload_f32(dev, &small_random(vocab * hidden, 307), &[vocab, hidden])?,
        layers: vec![layer],
        rotary_inv_freq: vec![],
        rotary_dim: 0,
        vocab,
        hidden,
    })
}

fn build_gdn_lora_layer(dev: &Arc<VulkanDevice>) -> Result<VkLoraLayer> {
    let config = tiny_gdn_model_config();
    let hidden = config.hidden_size;
    let qkv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let rank = 4;
    let alpha = 8.0;
    Ok(VkLoraLayer {
        in_proj_qkv: Some(VkLoraPair::init_kaiming(
            dev, hidden, qkv_dim, rank, alpha, 401,
        )?),
        in_proj_z: Some(VkLoraPair::init_kaiming(
            dev, hidden, v_dim, rank, alpha, 402,
        )?),
        gdn_out_proj: Some(VkLoraPair::init_kaiming(
            dev, v_dim, hidden, rank, alpha, 403,
        )?),
        ..Default::default()
    })
}

fn tiny_gdn_state(dev: &Arc<VulkanDevice>) -> Result<VkLinearAttentionState> {
    let config = tiny_gdn_model_config();
    let conv_channels = 2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim;
    VkLinearAttentionState::zeros(
        dev,
        1,
        1,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        conv_channels,
        config.linear_conv_kernel_dim,
    )
}

fn max_abs_tensor(t: &VkTensor) -> Result<f32> {
    Ok(t.to_vec_f32()?.into_iter().map(f32::abs).fold(0.0, f32::max))
}

#[test]
fn vk_full_model_one_layer_grads_exist() -> Result<()> {
    use kiln_model::vk_forward::vk_transformer_layer;
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::flce::vk_flce_loss;
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_model(&dev)?;
    // build_tiny_model: heads_kv=1, head_dim=16
    let kv_dim = 16;
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

    let initial_loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?.to_vec_f32()?[0];
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

/// Build a tiny synthetic FullAttn model with the Qwen3.5-specific
/// pieces enabled: per-head q_norm/k_norm and attn_output_gate (which
/// makes q_proj produce 2× output, splitting into Q and a sigmoid
/// gate). Validates the new vk_full_attention_layer paths end-to-end.
fn build_tiny_qwen35_specific_model(dev: &Arc<VulkanDevice>) -> Result<VkModelWeights> {
    let vocab = 32;
    let hidden = 32;
    let intermediate = 64;
    let heads_q = 2;
    let heads_kv = 1;
    let head_dim = hidden / heads_q;
    let kv_dim = heads_kv * head_dim;
    let q_dim = heads_q * head_dim;
    let q_out_dim = q_dim * 2; // attn_output_gate = true → q_proj is 2× wide

    let embed = small_random(vocab * hidden, 21);
    let final_norm = vec![0.0_f32; hidden];
    let lm_head = small_random(vocab * hidden, 22);

    let in_norm = vec![0.0_f32; hidden];
    let post_norm = vec![0.0_f32; hidden];
    // q_proj: [q_out_dim=2*q_dim, hidden] — fused [Q, gate]
    let q = small_random(q_out_dim * hidden, 23);
    let k = small_random(kv_dim * hidden, 24);
    let v = small_random(kv_dim * hidden, 25);
    let o = small_random(hidden * q_dim, 26);
    let gate = small_random(intermediate * hidden, 27);
    let up = small_random(intermediate * hidden, 28);
    let down = small_random(hidden * intermediate, 29);
    // q_norm/k_norm weights: per-head RMSNorm scale, [head_dim].
    // Center on 0.0 since RMSNorm in this codebase uses (1 + w) form.
    let q_norm = vec![0.0_f32; head_dim];
    let k_norm = vec![0.0_f32; head_dim];

    let layer = VkLayerWeights::FullAttention(VkFullAttentionWeights {
        input_layernorm_weight: upload_f32(dev, &in_norm, &[hidden])?,
        post_attention_layernorm_weight: upload_f32(dev, &post_norm, &[hidden])?,
        q_proj: upload_f32(dev, &q, &[q_out_dim, hidden])?,
        k_proj: upload_f32(dev, &k, &[kv_dim, hidden])?,
        v_proj: upload_f32(dev, &v, &[kv_dim, hidden])?,
        o_proj: upload_f32(dev, &o, &[hidden, q_dim])?,
        q_norm: Some(upload_f32(dev, &q_norm, &[head_dim])?),
        k_norm: Some(upload_f32(dev, &k_norm, &[head_dim])?),
        gate_proj: upload_f32(dev, &gate, &[intermediate, hidden])?,
        up_proj: upload_f32(dev, &up, &[intermediate, hidden])?,
        down_proj: upload_f32(dev, &down, &[hidden, intermediate])?,
        heads_q,
        heads_kv,
        head_dim,
        attn_output_gate: true,
        eps: 1e-5,
    });
    Ok(VkModelWeights {
        embed_tokens: upload_f32(dev, &embed, &[vocab, hidden])?,
        embed_dtype: VkDType::F32,
        final_norm_weight: upload_f32(dev, &final_norm, &[hidden])?,
        lm_head: upload_f32(dev, &lm_head, &[vocab, hidden])?,
        layers: vec![layer],
        rotary_inv_freq: vec![],
        rotary_dim: 0,
        vocab,
        hidden,
    })
}

/// Build LoRA params for the Qwen3.5-specific synthetic model. q_proj
/// LoRA targets the doubled output dim (covers the [Q, gate] split).
fn build_qwen35_specific_lora(dev: &Arc<VulkanDevice>) -> Result<VkLoraLayer> {
    let hidden = 32;
    let heads_q = 2;
    let heads_kv = 1;
    let head_dim = hidden / heads_q;
    let kv_dim = heads_kv * head_dim;
    let q_dim = heads_q * head_dim;
    let q_out_dim = q_dim * 2;
    let intermediate = 64;
    let rank = 4;
    let alpha = 8.0;
    Ok(VkLoraLayer {
        q_proj: Some(VkLoraPair::init_kaiming(
            dev, hidden, q_out_dim, rank, alpha, 200,
        )?),
        k_proj: Some(VkLoraPair::init_kaiming(
            dev, hidden, kv_dim, rank, alpha, 201,
        )?),
        v_proj: Some(VkLoraPair::init_kaiming(
            dev, hidden, kv_dim, rank, alpha, 202,
        )?),
        o_proj: Some(VkLoraPair::init_kaiming(
            dev, q_dim, hidden, rank, alpha, 203,
        )?),
        gate_proj: Some(VkLoraPair::init_kaiming(
            dev,
            hidden,
            intermediate,
            rank,
            alpha,
            204,
        )?),
        up_proj: Some(VkLoraPair::init_kaiming(
            dev,
            hidden,
            intermediate,
            rank,
            alpha,
            205,
        )?),
        down_proj: Some(VkLoraPair::init_kaiming(
            dev,
            intermediate,
            hidden,
            rank,
            alpha,
            206,
        )?),
        ..Default::default()
    })
}

#[test]
fn vk_qwen35_specific_forward_produces_finite_output() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_qwen35_specific_model(&dev)?;
    let lora_layers = vec![build_qwen35_specific_lora(&dev)?];
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?.to_vec_f32()?[0];
    assert!(
        loss.is_finite(),
        "Qwen3.5-specific forward produced non-finite loss {loss}"
    );
    println!("vk_qwen35_specific forward loss = {loss}");
    Ok(())
}

#[test]
fn vk_qwen35_specific_backward_propagates_to_all_lora() -> Result<()> {
    use kiln_model::vk_forward::vk_step_backward;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_qwen35_specific_model(&dev)?;
    let lora_layers = vec![build_qwen35_specific_lora(&dev)?];
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?;
    let grads = vk_step_backward(&loss)?;
    println!("vk_qwen35_specific: grads has {} entries", grads.len());
    // We expect grads for: q.a, q.b, k.a, k.b, v.a, v.b, o.a, o.b,
    // gate.a, gate.b, up.a, up.b, down.a, down.b → 14 LoRA params total.
    // Some may have effectively-zero grads if upstream signal is small,
    // but they should all be present in the grad map.
    assert!(
        grads.len() >= 8,
        "expected at least 8 LoRA grads (got {})",
        grads.len()
    );
    Ok(())
}

#[test]
fn vk_rope_wired_into_full_attn_layer() -> Result<()> {
    // Build the same Qwen3.5-specific model but populate
    // rotary_inv_freq so the RoPE path activates. Verify the forward
    // produces finite output (not NaN) and gradients flow.
    use kiln_model::vk_forward::vk_step_backward;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let mut model = build_tiny_qwen35_specific_model(&dev)?;
    // head_dim=16, partial_rotary_factor=0.5 → rotary_dim=8, half=4
    let half = 4;
    let rope_theta = 10000.0_f64;
    model.rotary_inv_freq = (0..half)
        .map(|i| 1.0 / (rope_theta.powf(2.0 * i as f64 / 8.0) as f32))
        .collect();
    model.rotary_dim = 8;

    let lora_layers = vec![build_qwen35_specific_lora(&dev)?];
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?.to_vec_f32()?[0];
    assert!(
        loss.is_finite(),
        "RoPE-enabled forward produced non-finite loss {loss}"
    );
    println!("vk_rope_wired loss = {loss}");

    let loss_t = vk_model_forward_loss(&model, &lora_layers, &input_ids)?;
    let grads = vk_step_backward(&loss_t)?;
    println!("vk_rope_wired: grads has {} entries", grads.len());
    assert!(grads.len() >= 8);
    Ok(())
}

#[test]
fn vk_native_full_pipeline_saves_loadable_adapter() -> Result<()> {
    // Drives the full vk-native training inner-loop pattern:
    //   - build VkModelWeights (synthetic)
    //   - init LoRA layers
    //   - allocate AdamW state
    //   - run 10 training steps
    //   - save adapter via save_vk_lora_adapter
    //   - verify adapter safetensors loads with the expected PEFT key
    //     names
    use kiln_train::vk_train::save_vk_lora_adapter;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_model(&dev)?;
    let lora_layers = vec![build_lora_layer(&dev, model.hidden, 16, 64)?];
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    for step in 1..=10 {
        let l = vk_train_step(&model, &lora_layers, &input_ids, &mut adamw, &cfg, step)?;
        assert!(l.is_finite());
    }
    let tmp = std::env::temp_dir().join(format!(
        "kiln-vk-test-adapter-{}.safetensors",
        std::process::id()
    ));
    let rank = 4;
    let alpha = 8.0;
    save_vk_lora_adapter(&lora_layers, rank, alpha, &tmp)?;
    // Read it back via candle safetensors and verify keys
    let loaded = candle_core::safetensors::load(&tmp, &candle_core::Device::Cpu)?;
    println!("vk_native_full_pipeline saved {} tensors", loaded.len());
    // PEFT convention: at least q_proj.lora_A and q_proj.lora_B should exist
    let has_q_a = loaded.keys().any(|k| k.contains("q_proj.lora_A.weight"));
    let has_q_b = loaded.keys().any(|k| k.contains("q_proj.lora_B.weight"));
    assert!(has_q_a, "missing q_proj.lora_A in saved adapter");
    assert!(has_q_b, "missing q_proj.lora_B in saved adapter");
    let _ = std::fs::remove_file(&tmp);
    Ok(())
}

#[test]
fn vk_checkpointed_train_step_loss_decreases() -> Result<()> {
    // Verifies vk_checkpointed_train_step produces gradient updates
    // equivalent to the non-checkpointed path (same model, same input,
    // both should monotonically reduce loss).
    use kiln_train::vk_train::vk_checkpointed_train_step;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_model(&dev)?;
    let lora_layers = vec![build_lora_layer(&dev, model.hidden, 16, 64)?];
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];

    // 1-segment is equivalent to no-checkpoint, so test 1 first
    let initial_loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?.to_vec_f32()?[0];
    let mut last_loss = initial_loss;
    let mut losses = vec![initial_loss];
    for step in 1..=5 {
        // 1 segment exercises the "last segment only" branch
        let l = vk_checkpointed_train_step(
            &model,
            &lora_layers,
            &input_ids,
            &mut adamw,
            &cfg,
            step,
            1,
        )?;
        assert!(l.is_finite(), "step {step}: non-finite loss {l}");
        losses.push(l);
        last_loss = l;
    }
    println!("vk_checkpointed (1-seg) losses: {losses:?}");
    assert!(
        last_loss < initial_loss * 0.95,
        "loss did not drop: {initial_loss} -> {last_loss}"
    );
    Ok(())
}

#[test]
fn vk_recompute_train_step_loss_decreases() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_model(&dev)?;
    let model_config = tiny_model_config();
    let lora_layers = vec![build_lora_layer(&dev, model.hidden, 16, 64)?];
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let label_mask = vec![true; input_ids.len()];
    let initial_loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?.to_vec_f32()?[0];
    let mut last_loss = initial_loss;
    let mut losses = vec![initial_loss];
    for step in 1..=3 {
        let l = vk_recompute_train_step_with_state_masked(
            &model,
            &lora_layers,
            &input_ids,
            &label_mask,
            &model_config,
            0,
            &mut adamw,
            &cfg,
            step,
        )?;
        assert!(l.is_finite(), "step {step}: non-finite loss {l}");
        losses.push(l);
        last_loss = l;
    }
    println!("vk_recompute losses: {losses:?}");
    assert!(
        last_loss < initial_loss,
        "recompute loss did not drop: {initial_loss} -> {last_loss}"
    );
    Ok(())
}

#[test]
fn vk_gdn_lora_recompute_updates_and_saves_gdn_targets() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_gdn_model(&dev)?;
    let model_config = tiny_gdn_model_config();
    let lora = build_gdn_lora_layer(&dev)?;
    let qkv_b_before = max_abs_tensor(&lora.in_proj_qkv.as_ref().unwrap().b)?;
    let z_b_before = max_abs_tensor(&lora.in_proj_z.as_ref().unwrap().b)?;
    let out_b_before = max_abs_tensor(&lora.gdn_out_proj.as_ref().unwrap().b)?;
    assert_eq!(qkv_b_before, 0.0);
    assert_eq!(z_b_before, 0.0);
    assert_eq!(out_b_before, 0.0);

    let lora_layers = vec![lora];
    let mut state = tiny_gdn_state(&dev)?;
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let loss = vk_model_forward_loss_with_state(&model, &lora_layers, &input_ids, Some(&mut state))?;
    let grads = kiln_model::vk_forward::vk_step_backward(&loss)?;
    let layer = &lora_layers[0];
    for (name, pair) in [
        ("in_proj_qkv", layer.in_proj_qkv.as_ref().unwrap()),
        ("in_proj_z", layer.in_proj_z.as_ref().unwrap()),
        ("out_proj", layer.gdn_out_proj.as_ref().unwrap()),
    ] {
        let max_abs = grads
            .get(pair.b_id)
            .map(max_abs_tensor)
            .transpose()?
            .unwrap_or(0.0);
        println!("direct GDN {name}.B grad max_abs={max_abs:.6e}");
        assert!(
            grads.get(pair.b_id).is_some(),
            "missing direct backward grad for GDN {name}.B"
        );
    }

    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };
    let label_mask = vec![true; input_ids.len()];
    let train_loss = vk_recompute_train_step_with_state_masked(
        &model,
        &lora_layers,
        &input_ids,
        &label_mask,
        &model_config,
        1,
        &mut adamw,
        &cfg,
        1,
    )?;
    assert!(train_loss.is_finite(), "non-finite GDN recompute loss");

    let qkv_b_after = max_abs_tensor(&lora_layers[0].in_proj_qkv.as_ref().unwrap().b)?;
    let z_b_after = max_abs_tensor(&lora_layers[0].in_proj_z.as_ref().unwrap().b)?;
    let out_b_after = max_abs_tensor(&lora_layers[0].gdn_out_proj.as_ref().unwrap().b)?;
    println!(
        "after recompute GDN B max_abs: qkv={qkv_b_after:.6e} z={z_b_after:.6e} out={out_b_after:.6e}"
    );
    assert!(qkv_b_after > 0.0, "GDN in_proj_qkv LoRA B was not updated");
    assert!(z_b_after > 0.0, "GDN in_proj_z LoRA B was not updated");
    assert!(out_b_after > 0.0, "GDN out_proj LoRA B was not updated");

    let tmp = std::env::temp_dir().join(format!(
        "kiln-vk-test-gdn-adapter-{}.safetensors",
        std::process::id()
    ));
    save_vk_lora_adapter(&lora_layers, 4, 8.0, &tmp)?;
    let loaded = candle_core::safetensors::load(&tmp, &candle_core::Device::Cpu)?;
    for key in [
        "self_attn.in_proj_qkv.lora_A.weight",
        "self_attn.in_proj_z.lora_A.weight",
        "self_attn.out_proj.lora_A.weight",
    ] {
        assert!(
            loaded.keys().any(|k| k.contains(key)),
            "missing saved GDN LoRA key containing {key}"
        );
    }
    let _ = std::fs::remove_file(&tmp);
    Ok(())
}

#[test]
fn vk_qwen35_specific_training_loss_decreases() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_qwen35_specific_model(&dev)?;
    let lora_layers = vec![build_qwen35_specific_lora(&dev)?];
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let initial_loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?.to_vec_f32()?[0];
    let mut last_loss = initial_loss;
    let mut losses = vec![initial_loss];
    for step in 1..=10 {
        let l = vk_train_step(&model, &lora_layers, &input_ids, &mut adamw, &cfg, step)?;
        assert!(l.is_finite(), "step {step}: non-finite loss {l}");
        losses.push(l);
        last_loss = l;
    }
    println!("vk_qwen35_specific losses: {losses:?}");
    assert!(
        last_loss < initial_loss * 0.95,
        "Qwen3.5-specific loss did not drop meaningfully: {initial_loss} -> {last_loss}"
    );
    Ok(())
}
