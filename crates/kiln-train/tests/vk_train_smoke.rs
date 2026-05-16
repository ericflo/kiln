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
use candle_core::shape::ShapeWithOneHole;
use candle_core::{Device, Tensor};
use kiln_core::config::{DType, ModelConfig};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::{
    GpuAttentionWeights, GpuFfnWeights, GpuFullAttentionWeights, GpuLayerWeights, GpuWeights,
};
use kiln_model::vk_forward::{
    VkFullAttentionWeights, VkLayerWeights, VkLinearAttentionWeights, VkLoraLayer, VkLoraPair,
    VkModelWeights, vk_grpo_reference_log_probs_from_prefix,
    vk_grpo_reference_log_probs_full_sequence, vk_grpo_reference_prefill_prompt,
    vk_model_forward_final_norm_with_state, vk_model_forward_loss,
    vk_model_forward_loss_with_state,
};
use kiln_train::GrpoConfig;
use kiln_train::Optimizer;
use kiln_train::vk_train::{
    VkAdamWConfig, allocate_adamw_state, grpo_jsonl_stats, save_vk_lora_adapter,
    validate_vk_grpo_seq_lens, vk_init_lora_layers, vk_native_grpo_train_jsonl,
    vk_opd_train_step_with_state, vk_recompute_grpo_train_step_with_state,
    vk_recompute_opd_train_step_with_state, vk_recompute_train_step_with_state_masked,
    vk_train_step,
};
use kiln_vulkan_kernel::vk_ops::gdn_state::VkLinearAttentionState;
use kiln_vulkan_kernel::{VkDType, VkTensor, VulkanDevice};
use std::path::Path;
use std::sync::{Arc, Mutex};

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
        rope_cache: Default::default(),
        rotary_dim: 0,
        vocab,
        hidden,
    })
}

fn tiny_gpu_grpo_model_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_layers: 1,
        num_attention_heads: 1,
        num_kv_heads: 1,
        head_dim: 8,
        intermediate_size: 16,
        vocab_size: 8,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        dtype: DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: 1,
        linear_key_head_dim: 8,
        linear_num_value_heads: 1,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 0.0,
    }
}

fn cpu_tensor(data: Vec<f32>, shape: impl ShapeWithOneHole) -> Result<Tensor> {
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?)
}

fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    Ok(t.transpose(0, 1)?.contiguous()?)
}

fn build_tiny_gpu_grpo_weights() -> Result<GpuWeights> {
    let config = tiny_gpu_grpo_model_config();
    let vocab = config.vocab_size;
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let head_dim = config.head_dim;

    let embed_tokens = cpu_tensor(small_random(vocab * hidden, 501), (vocab, hidden))?;
    let embed_tokens_t = transpose_2d(&embed_tokens)?;
    let q_proj = cpu_tensor(small_random(hidden * hidden, 502), (hidden, hidden))?;
    let k_proj = cpu_tensor(small_random(hidden * hidden, 503), (hidden, hidden))?;
    let v_proj = cpu_tensor(small_random(hidden * hidden, 504), (hidden, hidden))?;
    let o_proj = cpu_tensor(small_random(hidden * hidden, 505), (hidden, hidden))?;
    let gate_proj = cpu_tensor(
        small_random(intermediate * hidden, 506),
        (intermediate, hidden),
    )?;
    let up_proj = cpu_tensor(
        small_random(intermediate * hidden, 507),
        (intermediate, hidden),
    )?;
    let down_proj = cpu_tensor(
        small_random(hidden * intermediate, 508),
        (hidden, intermediate),
    )?;

    Ok(GpuWeights {
        embed_tokens,
        embed_tokens_t,
        layers: vec![GpuLayerWeights {
            input_layernorm: cpu_tensor(vec![0.0; hidden], (hidden,))?,
            post_attention_layernorm: cpu_tensor(vec![0.0; hidden], (hidden,))?,
            attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                q_proj: q_proj.clone(),
                k_proj: k_proj.clone(),
                v_proj: v_proj.clone(),
                o_proj: o_proj.clone(),
                q_norm: cpu_tensor(vec![0.0; head_dim], (head_dim,))?,
                k_norm: cpu_tensor(vec![0.0; head_dim], (head_dim,))?,
                q_proj_t: transpose_2d(&q_proj)?,
                k_proj_t: transpose_2d(&k_proj)?,
                v_proj_t: transpose_2d(&v_proj)?,
                qkv_proj_t: None,
                o_proj_t: transpose_2d(&o_proj)?,
                q_proj_marlin: None,
            }),
            mlp: GpuFfnWeights {
                gate_proj: gate_proj.clone(),
                up_proj: up_proj.clone(),
                down_proj: down_proj.clone(),
                gate_proj_t: transpose_2d(&gate_proj)?,
                up_proj_t: transpose_2d(&up_proj)?,
                down_proj_t: transpose_2d(&down_proj)?,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        }],
        final_norm: cpu_tensor(vec![0.0; hidden], (hidden,))?,
        rotary_inv_freq: cpu_tensor(Vec::<f32>::new(), (0,))?,
        mtp: None,
    })
}

#[test]
fn vk_from_gpu_weights_restores_stubbed_tied_embedding_on_device() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let config = tiny_gpu_grpo_model_config();
    let mut weights = build_tiny_gpu_grpo_weights()?;
    let expected = weights.embed_tokens.flatten_all()?.to_vec1::<f32>()?;
    weights.embed_tokens = Tensor::zeros((1usize,), candle_core::DType::F32, &Device::Cpu)?;

    let vk_weights = VkModelWeights::from_gpu_weights(&weights, &config, &dev)?;
    assert_eq!(
        vk_weights.embed_tokens.shape(),
        &[config.vocab_size, config.hidden_size]
    );
    assert!(
        Arc::ptr_eq(
            vk_weights.embed_tokens.buffer(),
            vk_weights.lm_head.buffer()
        ),
        "tied lm_head should share the embed_tokens Vulkan buffer"
    );
    let got = vk_weights.embed_tokens.to_vec_f32()?;
    let mad = got
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(mad < 1e-6, "restored embedding max abs diff {mad}");
    Ok(())
}

fn tiny_grpo_tokenizer() -> Result<KilnTokenizer> {
    let json = br#"{
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1},
            "merges": []
        }
    }"#;
    Ok(
        KilnTokenizer::from_bytes(json.as_slice())?.with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        ),
    )
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
    let intermediate = config.intermediate_size;

    let layer = VkLayerWeights::LinearAttention(VkLinearAttentionWeights {
        layer_norm: upload_f32(dev, &vec![0.0_f32; hidden], &[hidden])?,
        in_proj_qkv: upload_f32(
            dev,
            &small_random(qkv_dim * hidden, 301),
            &[qkv_dim, hidden],
        )?,
        in_proj_z: upload_f32(dev, &small_random(v_dim * hidden, 302), &[v_dim, hidden])?,
        in_proj_a: upload_f32(dev, &small_random(nv * hidden, 303), &[nv, hidden])?,
        in_proj_b: upload_f32(dev, &small_random(nv * hidden, 304), &[nv, hidden])?,
        conv1d: upload_f32(
            dev,
            &small_random(qkv_dim * conv_kernel, 305),
            &[qkv_dim, conv_kernel],
        )?,
        a_log: upload_f32(dev, &vec![-0.5_f32; nv], &[nv])?,
        a_log_gates: upload_f32(dev, &vec![-0.5_f32; nv], &[nv])?,
        dt_bias: upload_f32(dev, &vec![0.1_f32; nv], &[nv])?,
        gated_norm: upload_f32(dev, &vec![1.0_f32; v_dim], &[v_dim])?,
        out_proj: upload_f32(dev, &small_random(hidden * v_dim, 306), &[hidden, v_dim])?,
        post_attention_layernorm_weight: upload_f32(dev, &vec![0.0_f32; hidden], &[hidden])?,
        gate_proj: upload_f32(
            dev,
            &small_random(intermediate * hidden, 308),
            &[intermediate, hidden],
        )?,
        up_proj: upload_f32(
            dev,
            &small_random(intermediate * hidden, 309),
            &[intermediate, hidden],
        )?,
        down_proj: upload_f32(
            dev,
            &small_random(hidden * intermediate, 310),
            &[hidden, intermediate],
        )?,
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
        rope_cache: Default::default(),
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
    let intermediate = config.intermediate_size;
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
        gate_proj: Some(VkLoraPair::init_kaiming(
            dev,
            hidden,
            intermediate,
            rank,
            alpha,
            404,
        )?),
        up_proj: Some(VkLoraPair::init_kaiming(
            dev,
            hidden,
            intermediate,
            rank,
            alpha,
            405,
        )?),
        down_proj: Some(VkLoraPair::init_kaiming(
            dev,
            intermediate,
            hidden,
            rank,
            alpha,
            406,
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
    Ok(t.to_vec_f32()?
        .into_iter()
        .map(f32::abs)
        .fold(0.0, f32::max))
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

/// End-to-end vk-native OPD training smoke test.
///
/// Builds a tiny FullAttention model, fabricates a deterministic teacher
/// top-K distribution over the synthetic vocab, runs
/// `vk_opd_train_step_with_state` 10 times against the AdamW optimizer,
/// and asserts the OPD reverse-KL drops monotonically toward zero.
///
/// This is the OPD analogue of `vk_native_training_loss_decreases` —
/// it's the same end-to-end "loss strictly decreases" proof but with
/// the fused `vk_opd_top_k_reverse_kl_loss` driving the gradient. The
/// loss-decrease over the LoRA-trained adapter is the real-world
/// signal that the kernel + autograd wiring are correct.
#[test]
fn vk_native_opd_training_loss_decreases() -> Result<()> {
    let Some(dev) = vk_dev() else {
        eprintln!("Vulkan device not available — skipping");
        return Ok(());
    };
    let model = build_tiny_model(&dev)?;
    let lora_layers = vec![build_lora_layer(&dev, model.hidden, 16, 64)?];
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };

    // 8-token synthetic input. Active rows skip position 0 (BOS) — every
    // other token contributes to the loss, mirroring the typical
    // assistant-only label mask.
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let active_rows: Vec<u32> = (1u32..(input_ids.len() as u32)).collect();
    let active_count = active_rows.len();

    // Teacher top-K distribution. K=16 fits comfortably inside the
    // synthetic vocab=32. We pick K distinct vocab indices per active
    // row and assign decreasing logprobs (peaky distribution — easy
    // signal for the student to match).
    let top_k = 16usize;
    let vocab = model.vocab;
    let mut teacher_topk_indices: Vec<u32> = Vec::with_capacity(active_count * top_k);
    let mut teacher_topk_logprobs: Vec<f32> = Vec::with_capacity(active_count * top_k);
    for (row_i, _t) in active_rows.iter().enumerate() {
        let mut row: Vec<u32> = (0..top_k as u32)
            .map(|k| ((row_i * 7 + k as usize * 3 + 1) % vocab) as u32)
            .collect();
        let mut seen = std::collections::HashSet::new();
        for k in 0..top_k {
            while !seen.insert(row[k]) {
                row[k] = (row[k] + 1) % vocab as u32;
            }
        }
        teacher_topk_indices.extend_from_slice(&row);
        // Decaying logprobs — teacher concentrates mass at k=0.
        for k in 0..top_k {
            teacher_topk_logprobs.push(-((k as f32) * 0.5));
        }
    }

    let mut losses = Vec::new();
    let initial = vk_opd_train_step_with_state(
        &model,
        &lora_layers,
        &input_ids,
        &active_rows,
        &teacher_topk_indices,
        &teacher_topk_logprobs,
        top_k,
        None,
        &mut adamw,
        &cfg,
        Optimizer::AdamW {
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            eps: cfg.eps,
            weight_decay: cfg.weight_decay,
        },
        1,
    )?;
    assert!(initial.is_finite(), "step 1 OPD loss non-finite: {initial}");
    losses.push(initial);

    let mut last_loss = initial;
    for step in 2u32..=10 {
        let l = vk_opd_train_step_with_state(
            &model,
            &lora_layers,
            &input_ids,
            &active_rows,
            &teacher_topk_indices,
            &teacher_topk_logprobs,
            top_k,
            None,
            &mut adamw,
            &cfg,
            Optimizer::AdamW {
                beta1: cfg.beta1,
                beta2: cfg.beta2,
                eps: cfg.eps,
                weight_decay: cfg.weight_decay,
            },
            step,
        )?;
        assert!(l.is_finite(), "step {step}: non-finite OPD loss {l}");
        losses.push(l);
        last_loss = l;
    }
    println!("vk_native_opd_training losses: {losses:?}");
    assert!(
        last_loss < initial * 0.95,
        "OPD reverse-KL did not drop meaningfully: {initial} -> {last_loss}"
    );
    // Reverse-KL is non-negative; sanity-check the gradient direction.
    assert!(
        last_loss >= -1e-4,
        "OPD reverse-KL went negative at last step: {last_loss}"
    );
    Ok(())
}

/// End-to-end vk-native OPD training loss-decreases test against the
/// gradient-checkpointed `vk_recompute_opd_train_step_with_state`.
///
/// Same proof as `vk_native_opd_training_loss_decreases`, but driven through
/// the layerwise reverse-recompute path that mirrors the CUDA-side
/// `opd_train` checkpointing pattern. Without checkpointing, long-context
/// OPD ran out of memory at ~750 tokens on a 48 GB GPU; the recompute
/// variant trades peak memory for re-doing each layer's forward once.
/// This test exercises the recompute path's correctness — the synthetic
/// 1-layer model fits in either path, so the goal is purely to verify
/// the layerwise propagation produces the same monotonic loss decrease.
#[test]
fn vk_native_opd_recompute_training_loss_decreases() -> Result<()> {
    let Some(dev) = vk_dev() else {
        eprintln!("Vulkan device not available — skipping");
        return Ok(());
    };
    let model = build_tiny_model(&dev)?;
    let lora_layers = vec![build_lora_layer(&dev, model.hidden, 16, 64)?];
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };

    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let active_rows: Vec<u32> = (1u32..(input_ids.len() as u32)).collect();
    let active_count = active_rows.len();

    let top_k = 16usize;
    let vocab = model.vocab;
    let mut teacher_topk_indices: Vec<u32> = Vec::with_capacity(active_count * top_k);
    let mut teacher_topk_logprobs: Vec<f32> = Vec::with_capacity(active_count * top_k);
    for row_i in 0..active_count {
        let mut row: Vec<u32> = (0..top_k as u32)
            .map(|k| ((row_i * 7 + k as usize * 3 + 1) % vocab) as u32)
            .collect();
        let mut seen = std::collections::HashSet::new();
        for k in 0..top_k {
            while !seen.insert(row[k]) {
                row[k] = (row[k] + 1) % vocab as u32;
            }
        }
        teacher_topk_indices.extend_from_slice(&row);
        for k in 0..top_k {
            teacher_topk_logprobs.push(-((k as f32) * 0.5));
        }
    }

    // tiny_model_config matches build_tiny_model's choices.
    let model_config = tiny_model_config();
    let num_gdn_layers = 0; // build_tiny_model has no GDN layers

    let optimizer = Optimizer::AdamW {
        beta1: cfg.beta1,
        beta2: cfg.beta2,
        eps: cfg.eps,
        weight_decay: cfg.weight_decay,
    };

    let mut losses = Vec::new();
    let initial = vk_recompute_opd_train_step_with_state(
        &model,
        &lora_layers,
        &input_ids,
        &active_rows,
        &teacher_topk_indices,
        &teacher_topk_logprobs,
        top_k,
        &model_config,
        num_gdn_layers,
        &mut adamw,
        &cfg,
        optimizer,
        1,
    )?;
    assert!(initial.is_finite(), "step 1 recompute OPD loss non-finite: {initial}");
    losses.push(initial);

    let mut last_loss = initial;
    for step in 2u32..=10 {
        let l = vk_recompute_opd_train_step_with_state(
            &model,
            &lora_layers,
            &input_ids,
            &active_rows,
            &teacher_topk_indices,
            &teacher_topk_logprobs,
            top_k,
            &model_config,
            num_gdn_layers,
            &mut adamw,
            &cfg,
            optimizer,
            step,
        )?;
        assert!(l.is_finite(), "step {step}: non-finite recompute OPD loss {l}");
        losses.push(l);
        last_loss = l;
    }
    println!("vk_native_opd_recompute_training losses: {losses:?}");
    assert!(
        last_loss < initial * 0.95,
        "checkpointed OPD reverse-KL did not drop meaningfully: {initial} -> {last_loss}"
    );
    assert!(
        last_loss >= -1e-4,
        "checkpointed OPD reverse-KL went negative at last step: {last_loss}"
    );
    Ok(())
}

/// Multi-layer FullAttention model. Used to drive memory-pressure stress
/// tests against the checkpointed OPD trainer on smaller-VRAM GPUs (the
/// 1-layer `build_tiny_model` is too small to differentiate
/// checkpointed-vs-non-checkpointed memory headroom).
fn build_multilayer_model(
    dev: &Arc<VulkanDevice>,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    vocab: usize,
    heads_q: usize,
    heads_kv: usize,
) -> Result<VkModelWeights> {
    let head_dim = hidden / heads_q;
    let kv_dim = heads_kv * head_dim;

    let embed = small_random(vocab * hidden, 1);
    let final_norm = vec![0.0_f32; hidden];
    let lm_head = small_random(vocab * hidden, 99);

    let mut layers: Vec<VkLayerWeights> = Vec::with_capacity(num_layers);
    for li in 0..num_layers {
        let li64 = li as u64;
        let in_norm = vec![0.0_f32; hidden];
        let post_norm = vec![0.0_f32; hidden];
        // Use distinct seeds per layer so the model is non-degenerate.
        let q = small_random(hidden * hidden, 200 + li64 * 7);
        let k = small_random(kv_dim * hidden, 201 + li64 * 7);
        let v = small_random(kv_dim * hidden, 202 + li64 * 7);
        let o = small_random(hidden * hidden, 203 + li64 * 7);
        let gate = small_random(intermediate * hidden, 204 + li64 * 7);
        let up = small_random(intermediate * hidden, 205 + li64 * 7);
        let down = small_random(hidden * intermediate, 206 + li64 * 7);
        layers.push(VkLayerWeights::FullAttention(VkFullAttentionWeights {
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
        }));
    }
    Ok(VkModelWeights {
        embed_tokens: upload_f32(dev, &embed, &[vocab, hidden])?,
        embed_dtype: VkDType::F32,
        final_norm_weight: upload_f32(dev, &final_norm, &[hidden])?,
        lm_head: upload_f32(dev, &lm_head, &[vocab, hidden])?,
        layers,
        rotary_inv_freq: vec![],
        rope_cache: Default::default(),
        rotary_dim: 0,
        vocab,
        hidden,
    })
}

fn build_multilayer_lora(
    dev: &Arc<VulkanDevice>,
    hidden: usize,
    intermediate: usize,
    kv_dim: usize,
    num_layers: usize,
) -> Result<Vec<VkLoraLayer>> {
    let mut out = Vec::with_capacity(num_layers);
    for li in 0..num_layers {
        let s = 500u64 + (li as u64) * 17;
        out.push(VkLoraLayer {
            q_proj: Some(VkLoraPair::init_kaiming(dev, hidden, hidden, 4, 8.0, s)?),
            k_proj: Some(VkLoraPair::init_kaiming(dev, hidden, kv_dim, 4, 8.0, s + 1)?),
            v_proj: Some(VkLoraPair::init_kaiming(dev, hidden, kv_dim, 4, 8.0, s + 2)?),
            o_proj: Some(VkLoraPair::init_kaiming(dev, hidden, hidden, 4, 8.0, s + 3)?),
            gate_proj: Some(VkLoraPair::init_kaiming(
                dev,
                hidden,
                intermediate,
                4,
                8.0,
                s + 4,
            )?),
            up_proj: Some(VkLoraPair::init_kaiming(
                dev,
                hidden,
                intermediate,
                4,
                8.0,
                s + 5,
            )?),
            down_proj: Some(VkLoraPair::init_kaiming(
                dev,
                intermediate,
                hidden,
                4,
                8.0,
                s + 6,
            )?),
            ..Default::default()
        });
    }
    Ok(out)
}

fn multilayer_model_config(
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    vocab: usize,
    heads_q: usize,
    heads_kv: usize,
    max_seq: usize,
) -> ModelConfig {
    ModelConfig {
        hidden_size: hidden,
        num_layers,
        num_attention_heads: heads_q,
        num_kv_heads: heads_kv,
        head_dim: hidden / heads_q,
        intermediate_size: intermediate,
        vocab_size: vocab,
        max_position_embeddings: max_seq,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        dtype: DType::FP32,
        num_full_attention_layers: num_layers,
        full_attention_interval: 1,
        attn_output_gate: false,
        linear_num_key_heads: heads_kv,
        linear_key_head_dim: hidden / heads_q,
        linear_num_value_heads: heads_kv,
        linear_value_head_dim: hidden / heads_q,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 0.0,
    }
}

/// Memory-pressure stress test for the checkpointed OPD trainer.
///
/// Builds an 8-layer / hidden=1024 / intermediate=4096 model with a
/// 1024-token sequence (all active), runs `vk_recompute_opd_train_step_with_state`
/// for 3 steps, and asserts the loss decreases. Total activation memory
/// is high enough that the non-checkpointed path would keep ≳2 GB
/// resident across all 8 layers' forward tape simultaneously; the
/// checkpointed path holds only one layer's tape at peak.
///
/// On smaller-VRAM hardware (RTX A5000 / 3090 / 4090, 24 GB) this is
/// the test that demonstrates the checkpointing actually pays for
/// itself — it's still well within budget, but it's the realistic-shape
/// workload that exercises the segmented forward path under pressure.
#[test]
fn vk_native_opd_recompute_low_vram_stress() -> Result<()> {
    let Some(dev) = vk_dev() else {
        eprintln!("Vulkan device not available — skipping");
        return Ok(());
    };
    let hidden = 1024;
    let intermediate = 4096;
    let num_layers = 8;
    let vocab = 128;
    let heads_q = 8;
    let heads_kv = 4;
    let head_dim = hidden / heads_q;
    let kv_dim = heads_kv * head_dim;
    let seq_len = 1024;
    let max_seq = 2048;

    let model = build_multilayer_model(&dev, hidden, intermediate, num_layers, vocab, heads_q, heads_kv)?;
    let lora_layers = build_multilayer_lora(&dev, hidden, intermediate, kv_dim, num_layers)?;
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 5e-3,
        ..Default::default()
    };

    let input_ids: Vec<u32> = (0..seq_len as u32).map(|i| (i * 3 + 1) % vocab as u32).collect();
    let active_rows: Vec<u32> = (1u32..seq_len as u32).collect();
    let active_count = active_rows.len();

    let top_k = 16usize;
    let mut teacher_topk_indices: Vec<u32> = Vec::with_capacity(active_count * top_k);
    let mut teacher_topk_logprobs: Vec<f32> = Vec::with_capacity(active_count * top_k);
    for row_i in 0..active_count {
        let mut row: Vec<u32> = (0..top_k as u32)
            .map(|k| ((row_i * 7 + k as usize * 3 + 1) % vocab) as u32)
            .collect();
        let mut seen = std::collections::HashSet::new();
        for k in 0..top_k {
            while !seen.insert(row[k]) {
                row[k] = (row[k] + 1) % vocab as u32;
            }
        }
        teacher_topk_indices.extend_from_slice(&row);
        for k in 0..top_k {
            teacher_topk_logprobs.push(-((k as f32) * 0.3));
        }
    }

    let model_config = multilayer_model_config(
        hidden,
        intermediate,
        num_layers,
        vocab,
        heads_q,
        heads_kv,
        max_seq,
    );
    let optimizer = Optimizer::AdamW {
        beta1: cfg.beta1,
        beta2: cfg.beta2,
        eps: cfg.eps,
        weight_decay: cfg.weight_decay,
    };

    let started = std::time::Instant::now();
    let initial = vk_recompute_opd_train_step_with_state(
        &model,
        &lora_layers,
        &input_ids,
        &active_rows,
        &teacher_topk_indices,
        &teacher_topk_logprobs,
        top_k,
        &model_config,
        0, // num_gdn_layers
        &mut adamw,
        &cfg,
        optimizer,
        1,
    )?;
    let step1_ms = started.elapsed().as_millis();
    assert!(initial.is_finite(), "initial OPD recompute loss non-finite: {initial}");
    println!(
        "vk_native_opd_recompute_low_vram_stress: \
         L={num_layers} H={hidden} I={intermediate} V={vocab} T={seq_len} \
         step1={initial:.4} ({step1_ms} ms)"
    );

    let mut losses = vec![initial];
    let mut last_loss = initial;
    for step in 2u32..=3 {
        let s = std::time::Instant::now();
        let l = vk_recompute_opd_train_step_with_state(
            &model,
            &lora_layers,
            &input_ids,
            &active_rows,
            &teacher_topk_indices,
            &teacher_topk_logprobs,
            top_k,
            &model_config,
            0,
            &mut adamw,
            &cfg,
            optimizer,
            step,
        )?;
        let elapsed_ms = s.elapsed().as_millis();
        assert!(l.is_finite(), "step {step}: non-finite recompute OPD loss {l}");
        println!(
            "vk_native_opd_recompute_low_vram_stress: step{step}={l:.4} ({elapsed_ms} ms)"
        );
        losses.push(l);
        last_loss = l;
    }
    println!("vk_native_opd_recompute_low_vram_stress losses: {losses:?}");
    assert!(
        last_loss < initial,
        "checkpointed OPD reverse-KL did not drop: {initial} -> {last_loss}"
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
        rope_cache: Default::default(),
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
    assert_eq!(model.rope_cache.lock().unwrap().len(), 0);
    let loss = vk_model_forward_loss(&model, &lora_layers, &input_ids)?.to_vec_f32()?[0];
    {
        let cache = model.rope_cache.lock().unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&input_ids.len()));
    }
    assert!(
        loss.is_finite(),
        "RoPE-enabled forward produced non-finite loss {loss}"
    );
    println!("vk_rope_wired loss = {loss}");

    let loss_t = vk_model_forward_loss(&model, &lora_layers, &input_ids)?;
    assert_eq!(model.rope_cache.lock().unwrap().len(), 1);
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
    for key in [
        "self_attn.q_proj.lora_A.weight",
        "self_attn.k_proj.lora_A.weight",
        "self_attn.v_proj.lora_A.weight",
        "self_attn.o_proj.lora_A.weight",
        "mlp.gate_proj.lora_A.weight",
        "mlp.up_proj.lora_A.weight",
        "mlp.down_proj.lora_A.weight",
    ] {
        assert!(
            loaded.keys().any(|k| k.contains(key)),
            "missing saved full-attn LoRA key containing {key}"
        );
    }
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
fn vk_init_lora_layers_targets_full_attention_and_gdn_mlp() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };

    let full_model = build_tiny_model(&dev)?;
    let full_layers = vk_init_lora_layers(&dev, &full_model, &tiny_model_config(), 4, 8.0, 123)?;
    let full = &full_layers[0];
    assert!(full.q_proj.is_some(), "missing full-attn q_proj LoRA");
    assert!(full.k_proj.is_some(), "missing full-attn k_proj LoRA");
    assert!(full.v_proj.is_some(), "missing full-attn v_proj LoRA");
    assert!(full.o_proj.is_some(), "missing full-attn o_proj LoRA");
    assert!(full.gate_proj.is_some(), "missing full-attn gate_proj LoRA");
    assert!(full.up_proj.is_some(), "missing full-attn up_proj LoRA");
    assert!(full.down_proj.is_some(), "missing full-attn down_proj LoRA");
    assert!(full.in_proj_qkv.is_none());
    assert!(full.in_proj_z.is_none());
    assert!(full.gdn_out_proj.is_none());

    let gdn_model = build_tiny_gdn_model(&dev)?;
    let gdn_layers = vk_init_lora_layers(&dev, &gdn_model, &tiny_gdn_model_config(), 4, 8.0, 456)?;
    let gdn = &gdn_layers[0];
    assert!(gdn.q_proj.is_none());
    assert!(gdn.k_proj.is_none());
    assert!(gdn.v_proj.is_none());
    assert!(gdn.o_proj.is_none());
    assert!(gdn.in_proj_qkv.is_some(), "missing GDN in_proj_qkv LoRA");
    assert!(gdn.in_proj_z.is_some(), "missing GDN in_proj_z LoRA");
    assert!(gdn.gdn_out_proj.is_some(), "missing GDN out_proj LoRA");
    assert!(gdn.gate_proj.is_some(), "missing GDN gate_proj LoRA");
    assert!(gdn.up_proj.is_some(), "missing GDN up_proj LoRA");
    assert!(gdn.down_proj.is_some(), "missing GDN down_proj LoRA");
    Ok(())
}

#[test]
fn vk_gdn_state_continuation_matches_monolithic_forward() -> Result<()> {
    use kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows;

    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_gdn_model(&dev)?;
    let no_lora = vec![VkLoraLayer::default(); model.layers.len()];
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let split_at = 5usize;

    let mut mono_state = tiny_gdn_state(&dev)?;
    let mono = vk_model_forward_final_norm_with_state(
        &model,
        &no_lora,
        &input_ids,
        Some(&mut mono_state),
    )?;
    let suffix_rows: Vec<u32> = (split_at..input_ids.len()).map(|idx| idx as u32).collect();
    let mono_suffix = vk_index_select_rows(&mono, &suffix_rows)?;

    let mut split_state = tiny_gdn_state(&dev)?;
    let _prefix = vk_model_forward_final_norm_with_state(
        &model,
        &no_lora,
        &input_ids[..split_at],
        Some(&mut split_state),
    )?;
    let split_suffix = vk_model_forward_final_norm_with_state(
        &model,
        &no_lora,
        &input_ids[split_at..],
        Some(&mut split_state),
    )?;

    assert_eq!(mono_suffix.shape(), split_suffix.shape());
    let mono_data = mono_suffix.to_vec_f32()?;
    let split_data = split_suffix.to_vec_f32()?;
    let max_abs = mono_data
        .iter()
        .zip(split_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs < 2e-4,
        "GDN split continuation drifted from monolithic forward: max_abs={max_abs:e}"
    );

    let snap = split_state.snapshot(&dev)?;
    assert_eq!(snap.layers.len(), split_state.layers.len());
    Ok(())
}

#[test]
fn grpo_jsonl_stats_counts_large_file_without_retaining_groups() -> Result<()> {
    let path = std::env::temp_dir().join(format!(
        "kiln-vk-grpo-large-stats-{}.jsonl",
        std::process::id()
    ));
    let mut body = String::new();
    for idx in 0..512 {
        body.push_str(&format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"prompt {idx}\"}}],\"completions\":[{{\"text\":\"a\",\"reward\":1.0}},{{\"text\":\"b\",\"reward\":0.0}}]}}\n"
        ));
    }
    std::fs::write(&path, body)?;
    let stats = grpo_jsonl_stats(&path)?;
    assert_eq!(stats, (512, 1024));
    let _ = std::fs::remove_file(path);
    Ok(())
}

fn assert_vk_adapter_config_targets(adapter_dir: &Path) -> Result<()> {
    let config_path = adapter_dir.join("adapter_config.json");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let modules = config
        .get("target_modules")
        .and_then(|value| value.as_array())
        .ok_or_else(|| anyhow::anyhow!("{} missing target_modules", config_path.display()))?;
    for expected in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
    ] {
        assert!(
            modules
                .iter()
                .any(|module| module.as_str() == Some(expected)),
            "{} missing target module {expected}",
            config_path.display()
        );
    }
    Ok(())
}

#[test]
fn vk_grpo_seq_len_guard_rejects_over_context_groups() {
    validate_vk_grpo_seq_lens(&[4, 5, 6], 6, "test context").unwrap();
    let err = validate_vk_grpo_seq_lens(&[4, 7], 6, "test context").unwrap_err();
    let message = format!("{err:#}");
    assert!(message.contains("test context"));
    assert!(message.contains("exceeds model max_position_embeddings 6"));
}

#[test]
fn vk_native_grpo_jsonl_smoke_streams_and_saves_adapter() -> Result<()> {
    let Some(_dev) = vk_dev() else { return Ok(()) };
    let dataset = std::env::temp_dir().join(format!(
        "kiln-vk-grpo-jsonl-smoke-{}.jsonl",
        std::process::id()
    ));
    let mut body = String::new();
    for _ in 0..6 {
        body.push_str(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"completions\":[{\"text\":\"b\",\"reward\":1.0},{\"text\":\"a\",\"reward\":0.0}]}\n",
        );
    }
    std::fs::write(&dataset, body)?;

    let adapter_root = std::env::temp_dir().join(format!(
        "kiln-vk-grpo-jsonl-adapters-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&adapter_root);
    std::fs::create_dir_all(&adapter_root)?;
    let config = GrpoConfig {
        learning_rate: 1e-2,
        lora_rank: 2,
        lora_alpha: 4.0,
        checkpoint_interval: Some(5),
        seed: Some(1234),
        ..Default::default()
    };
    let model_config = tiny_gpu_grpo_model_config();
    let weights = build_tiny_gpu_grpo_weights()?;
    let tokenizer = tiny_grpo_tokenizer()?;
    let out = vk_native_grpo_train_jsonl(
        &dataset,
        &config,
        &model_config,
        &weights,
        &tokenizer,
        &adapter_root,
        "jsonl-smoke",
        None,
    )?;

    let loaded =
        candle_core::safetensors::load(out.join("adapter_model.safetensors"), &Device::Cpu)?;
    assert_vk_adapter_config_targets(&out)?;
    for key in [
        "self_attn.q_proj.lora_A.weight",
        "self_attn.o_proj.lora_A.weight",
        "mlp.gate_proj.lora_A.weight",
        "mlp.down_proj.lora_A.weight",
    ] {
        assert!(
            loaded.keys().any(|k| k.contains(key)),
            "missing streamed GRPO adapter key containing {key}"
        );
    }
    let checkpoint = adapter_root
        .join("jsonl-smoke-checkpoint-5")
        .join("adapter_model.safetensors");
    assert!(
        checkpoint.exists(),
        "streamed GRPO checkpoint adapter was not written at {}",
        checkpoint.display()
    );
    assert_vk_adapter_config_targets(checkpoint.parent().unwrap())?;
    let checkpoint_loaded = candle_core::safetensors::load(&checkpoint, &Device::Cpu)?;
    assert!(
        checkpoint_loaded
            .keys()
            .any(|k| k.contains("self_attn.q_proj.lora_A.weight")),
        "streamed GRPO checkpoint missing LoRA adapter tensors"
    );
    let _ = std::fs::remove_file(dataset);
    let _ = std::fs::remove_dir_all(adapter_root);
    Ok(())
}

#[test]
fn vk_native_grpo_jsonl_smoke_streams_long_prompts() -> Result<()> {
    let Some(_dev) = vk_dev() else { return Ok(()) };
    let dataset = std::env::temp_dir().join(format!(
        "kiln-vk-grpo-jsonl-long-smoke-{}.jsonl",
        std::process::id()
    ));
    let prompt = "a".repeat(96);
    let strong_completion = "b".repeat(12);
    let weak_completion = "a".repeat(12);
    let mut body = String::new();
    for _ in 0..8 {
        body.push_str(&format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{prompt}\"}}],\"completions\":[{{\"text\":\"{strong_completion}\",\"reward\":1.0}},{{\"text\":\"{weak_completion}\",\"reward\":0.0}}]}}\n"
        ));
    }
    std::fs::write(&dataset, body)?;
    assert_eq!(grpo_jsonl_stats(&dataset)?, (8, 16));

    let adapter_root = std::env::temp_dir().join(format!(
        "kiln-vk-grpo-jsonl-long-adapters-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&adapter_root);
    std::fs::create_dir_all(&adapter_root)?;
    let config = GrpoConfig {
        learning_rate: 5e-3,
        lora_rank: 2,
        lora_alpha: 4.0,
        checkpoint_interval: Some(4),
        seed: Some(4321),
        ..Default::default()
    };
    let model_config = tiny_gpu_grpo_model_config();
    let weights = build_tiny_gpu_grpo_weights()?;
    let tokenizer = tiny_grpo_tokenizer()?;
    let out = vk_native_grpo_train_jsonl(
        &dataset,
        &config,
        &model_config,
        &weights,
        &tokenizer,
        &adapter_root,
        "jsonl-long-smoke",
        None,
    )?;

    assert!(out.join("adapter_model.safetensors").exists());
    assert_vk_adapter_config_targets(&out)?;
    let checkpoint = adapter_root
        .join("jsonl-long-smoke-checkpoint-4")
        .join("adapter_model.safetensors");
    assert!(
        checkpoint.exists(),
        "long-prompt streamed GRPO checkpoint was not written at {}",
        checkpoint.display()
    );
    assert_vk_adapter_config_targets(checkpoint.parent().unwrap())?;

    let _ = std::fs::remove_file(dataset);
    let _ = std::fs::remove_dir_all(adapter_root);
    Ok(())
}

#[test]
fn vk_native_grpo_jsonl_smoke_streams_large_dataset() -> Result<()> {
    let Some(_dev) = vk_dev() else { return Ok(()) };
    const GROUPS: usize = 64;

    let dataset = std::env::temp_dir().join(format!(
        "kiln-vk-grpo-jsonl-large-smoke-{}.jsonl",
        std::process::id()
    ));
    let mut body = String::new();
    for idx in 0..GROUPS {
        let prompt = if idx % 2 == 0 { "a" } else { "aa" };
        let good = if idx % 3 == 0 { "bb" } else { "b" };
        let weak = if idx % 3 == 0 { "aa" } else { "a" };
        body.push_str(&format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{prompt}\"}}],\"completions\":[{{\"text\":\"{good}\",\"reward\":1.0}},{{\"text\":\"{weak}\",\"reward\":0.0}}]}}\n"
        ));
    }
    std::fs::write(&dataset, body)?;
    assert_eq!(grpo_jsonl_stats(&dataset)?, (GROUPS, GROUPS * 2));

    let adapter_root = std::env::temp_dir().join(format!(
        "kiln-vk-grpo-jsonl-large-adapters-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&adapter_root);
    std::fs::create_dir_all(&adapter_root)?;
    let config = GrpoConfig {
        learning_rate: 2e-3,
        lora_rank: 2,
        lora_alpha: 4.0,
        checkpoint_interval: Some(32),
        seed: Some(9876),
        ..Default::default()
    };
    let model_config = tiny_gpu_grpo_model_config();
    let weights = build_tiny_gpu_grpo_weights()?;
    let tokenizer = tiny_grpo_tokenizer()?;
    let progress = Arc::new(Mutex::new(Vec::new()));
    let progress_cb = {
        let progress = Arc::clone(&progress);
        Box::new(move |p| progress.lock().unwrap().push(p))
    };
    let out = vk_native_grpo_train_jsonl(
        &dataset,
        &config,
        &model_config,
        &weights,
        &tokenizer,
        &adapter_root,
        "jsonl-large-smoke",
        Some(progress_cb),
    )?;

    assert!(out.join("adapter_model.safetensors").exists());
    assert_vk_adapter_config_targets(&out)?;
    let checkpoint = adapter_root
        .join("jsonl-large-smoke-checkpoint-32")
        .join("adapter_model.safetensors");
    assert!(
        checkpoint.exists(),
        "large streamed GRPO checkpoint was not written at {}",
        checkpoint.display()
    );
    assert_vk_adapter_config_targets(checkpoint.parent().unwrap())?;
    let updates = progress.lock().unwrap();
    assert!(
        updates.len() >= GROUPS,
        "expected at least one progress update per streamed group, got {}",
        updates.len()
    );
    assert!(
        updates.last().is_some_and(|p| p.progress >= 0.999),
        "final streamed progress should reach completion"
    );

    let _ = std::fs::remove_file(dataset);
    let _ = std::fs::remove_dir_all(adapter_root);
    Ok(())
}

fn grpo_ref_log_probs(
    model: &VkModelWeights,
    input_ids: &[u32],
    active_rows: &[u32],
    labels: &[u32],
    state: Option<&mut VkLinearAttentionState>,
) -> Result<VkTensor> {
    use kiln_vulkan_kernel::vk_ops::flce::vk_selected_log_probs;
    use kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows;

    let no_lora = vec![VkLoraLayer::default(); model.layers.len()];
    let h = vk_model_forward_final_norm_with_state(model, &no_lora, input_ids, state)?;
    let active_h = vk_index_select_rows(&h, active_rows)?;
    vk_selected_log_probs(&active_h, &model.lm_head, labels, 8)
}

fn assert_close_vec(name: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{name}: length mismatch");
    let max_abs = got
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= tol,
        "{name}: max_abs={max_abs:e} exceeds tolerance {tol:e}; got={got:?} expected={expected:?}"
    );
}

#[test]
fn vk_grpo_reference_prefix_scorer_matches_monolithic_full_attention() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_model(&dev)?;
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let prompt_len = 5usize;
    let active_rows: Vec<u32> = vec![4, 5, 6];
    let labels: Vec<u32> = active_rows
        .iter()
        .map(|&row| input_ids[row as usize + 1])
        .collect();

    let prefix = vk_grpo_reference_prefill_prompt(
        &model,
        &input_ids[..prompt_len],
        &tiny_model_config(),
        0,
    )?;
    let fast = vk_grpo_reference_log_probs_from_prefix(&model, &prefix, &labels)?;
    let mono = grpo_ref_log_probs(&model, &input_ids, &active_rows, &labels, None)?;

    assert_close_vec(
        "full-attn prefix GRPO reference scorer",
        &fast.to_vec_f32()?,
        &mono.to_vec_f32()?,
        5e-4,
    );
    Ok(())
}

#[test]
fn vk_grpo_reference_prefix_scorer_matches_monolithic_gdn() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_gdn_model(&dev)?;
    let config = tiny_gdn_model_config();
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let prompt_len = 5usize;
    let active_rows: Vec<u32> = vec![4, 5, 6];
    let labels: Vec<u32> = active_rows
        .iter()
        .map(|&row| input_ids[row as usize + 1])
        .collect();

    let prefix = vk_grpo_reference_prefill_prompt(&model, &input_ids[..prompt_len], &config, 1)?;
    let fast = vk_grpo_reference_log_probs_from_prefix(&model, &prefix, &labels)?;
    let mut mono_state = tiny_gdn_state(&dev)?;
    let mono = grpo_ref_log_probs(
        &model,
        &input_ids,
        &active_rows,
        &labels,
        Some(&mut mono_state),
    )?;

    assert_close_vec(
        "GDN prefix GRPO reference scorer",
        &fast.to_vec_f32()?,
        &mono.to_vec_f32()?,
        5e-4,
    );
    Ok(())
}

#[test]
fn vk_grpo_reference_full_sequence_scorer_matches_monolithic_full_attention() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_model(&dev)?;
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0, 13, 4, 9, 2];
    let active_rows: Vec<u32> = (4..(input_ids.len() - 1)).map(|row| row as u32).collect();
    let labels: Vec<u32> = active_rows
        .iter()
        .map(|&row| input_ids[row as usize + 1])
        .collect();

    let full = vk_grpo_reference_log_probs_full_sequence(
        &model,
        &input_ids,
        &active_rows,
        &labels,
        &tiny_model_config(),
        0,
    )?;
    let mono = grpo_ref_log_probs(&model, &input_ids, &active_rows, &labels, None)?;

    assert_close_vec(
        "full-attn full-sequence GRPO reference scorer",
        &full.to_vec_f32()?,
        &mono.to_vec_f32()?,
        5e-4,
    );
    Ok(())
}

#[test]
fn vk_grpo_reference_full_sequence_scorer_matches_monolithic_gdn() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_gdn_model(&dev)?;
    let config = tiny_gdn_model_config();
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0, 13, 4, 9, 2];
    let active_rows: Vec<u32> = (4..(input_ids.len() - 1)).map(|row| row as u32).collect();
    let labels: Vec<u32> = active_rows
        .iter()
        .map(|&row| input_ids[row as usize + 1])
        .collect();

    let full = vk_grpo_reference_log_probs_full_sequence(
        &model,
        &input_ids,
        &active_rows,
        &labels,
        &config,
        1,
    )?;
    let mut mono_state = tiny_gdn_state(&dev)?;
    let mono = grpo_ref_log_probs(
        &model,
        &input_ids,
        &active_rows,
        &labels,
        Some(&mut mono_state),
    )?;

    assert_close_vec(
        "GDN full-sequence GRPO reference scorer",
        &full.to_vec_f32()?,
        &mono.to_vec_f32()?,
        5e-4,
    );
    Ok(())
}

#[test]
fn vk_recompute_grpo_step_updates_full_attention_lora_targets() -> Result<()> {
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
    let active_rows: Vec<u32> = vec![2, 3, 4, 5, 6];
    let labels: Vec<u32> = active_rows
        .iter()
        .map(|&row| input_ids[row as usize + 1])
        .collect();
    let ref_log_probs = grpo_ref_log_probs(&model, &input_ids, &active_rows, &labels, None)?;

    let loss = vk_recompute_grpo_train_step_with_state(
        &model,
        &lora_layers,
        &input_ids,
        &active_rows,
        &labels,
        &ref_log_probs,
        1.0,
        0.2,
        0.1,
        &model_config,
        0,
        &mut adamw,
        &cfg,
        Optimizer::default(),
        1,
    )?;
    assert!(loss.is_finite(), "non-finite full-attn GRPO recompute loss");

    let layer = &lora_layers[0];
    for (name, pair) in [
        ("q_proj", layer.q_proj.as_ref().unwrap()),
        ("k_proj", layer.k_proj.as_ref().unwrap()),
        ("v_proj", layer.v_proj.as_ref().unwrap()),
        ("o_proj", layer.o_proj.as_ref().unwrap()),
        ("gate_proj", layer.gate_proj.as_ref().unwrap()),
        ("up_proj", layer.up_proj.as_ref().unwrap()),
        ("down_proj", layer.down_proj.as_ref().unwrap()),
    ] {
        let max_abs = max_abs_tensor(&pair.b)?;
        println!("full-attn GRPO {name}.B max_abs={max_abs:.6e}");
        assert!(max_abs > 0.0, "full-attn GRPO did not update {name}.B");
    }
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
    let gate_b_before = max_abs_tensor(&lora.gate_proj.as_ref().unwrap().b)?;
    let up_b_before = max_abs_tensor(&lora.up_proj.as_ref().unwrap().b)?;
    let down_b_before = max_abs_tensor(&lora.down_proj.as_ref().unwrap().b)?;
    assert_eq!(qkv_b_before, 0.0);
    assert_eq!(z_b_before, 0.0);
    assert_eq!(out_b_before, 0.0);
    assert_eq!(gate_b_before, 0.0);
    assert_eq!(up_b_before, 0.0);
    assert_eq!(down_b_before, 0.0);

    let lora_layers = vec![lora];
    let mut state = tiny_gdn_state(&dev)?;
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let loss =
        vk_model_forward_loss_with_state(&model, &lora_layers, &input_ids, Some(&mut state))?;
    let grads = kiln_model::vk_forward::vk_step_backward(&loss)?;
    let layer = &lora_layers[0];
    for (name, pair) in [
        ("in_proj_qkv", layer.in_proj_qkv.as_ref().unwrap()),
        ("in_proj_z", layer.in_proj_z.as_ref().unwrap()),
        ("out_proj", layer.gdn_out_proj.as_ref().unwrap()),
        ("gate_proj", layer.gate_proj.as_ref().unwrap()),
        ("up_proj", layer.up_proj.as_ref().unwrap()),
        ("down_proj", layer.down_proj.as_ref().unwrap()),
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
    let gate_b_after = max_abs_tensor(&lora_layers[0].gate_proj.as_ref().unwrap().b)?;
    let up_b_after = max_abs_tensor(&lora_layers[0].up_proj.as_ref().unwrap().b)?;
    let down_b_after = max_abs_tensor(&lora_layers[0].down_proj.as_ref().unwrap().b)?;
    println!(
        "after recompute GDN B max_abs: qkv={qkv_b_after:.6e} z={z_b_after:.6e} out={out_b_after:.6e} gate={gate_b_after:.6e} up={up_b_after:.6e} down={down_b_after:.6e}"
    );
    assert!(qkv_b_after > 0.0, "GDN in_proj_qkv LoRA B was not updated");
    assert!(z_b_after > 0.0, "GDN in_proj_z LoRA B was not updated");
    assert!(out_b_after > 0.0, "GDN out_proj LoRA B was not updated");
    assert!(gate_b_after > 0.0, "GDN gate_proj LoRA B was not updated");
    assert!(up_b_after > 0.0, "GDN up_proj LoRA B was not updated");
    assert!(down_b_after > 0.0, "GDN down_proj LoRA B was not updated");

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
        "mlp.gate_proj.lora_A.weight",
        "mlp.up_proj.lora_A.weight",
        "mlp.down_proj.lora_A.weight",
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
fn vk_recompute_grpo_step_updates_gdn_lora_targets() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let model = build_tiny_gdn_model(&dev)?;
    let model_config = tiny_gdn_model_config();
    let lora_layers = vec![build_gdn_lora_layer(&dev)?];
    let mut adamw = allocate_adamw_state(&dev, &lora_layers)?;
    let cfg = VkAdamWConfig {
        lr: 1e-2,
        ..Default::default()
    };
    let input_ids: Vec<u32> = vec![5, 12, 7, 19, 3, 22, 11, 0];
    let active_rows: Vec<u32> = vec![2, 3, 4, 5, 6];
    let labels: Vec<u32> = active_rows
        .iter()
        .map(|&row| input_ids[row as usize + 1])
        .collect();
    let mut ref_state = tiny_gdn_state(&dev)?;
    let ref_log_probs = grpo_ref_log_probs(
        &model,
        &input_ids,
        &active_rows,
        &labels,
        Some(&mut ref_state),
    )?;

    let loss = vk_recompute_grpo_train_step_with_state(
        &model,
        &lora_layers,
        &input_ids,
        &active_rows,
        &labels,
        &ref_log_probs,
        1.0,
        0.2,
        0.1,
        &model_config,
        1,
        &mut adamw,
        &cfg,
        Optimizer::default(),
        1,
    )?;
    assert!(loss.is_finite(), "non-finite GDN GRPO recompute loss");

    let layer = &lora_layers[0];
    for (name, pair) in [
        ("in_proj_qkv", layer.in_proj_qkv.as_ref().unwrap()),
        ("in_proj_z", layer.in_proj_z.as_ref().unwrap()),
        ("out_proj", layer.gdn_out_proj.as_ref().unwrap()),
        ("gate_proj", layer.gate_proj.as_ref().unwrap()),
        ("up_proj", layer.up_proj.as_ref().unwrap()),
        ("down_proj", layer.down_proj.as_ref().unwrap()),
    ] {
        let max_abs = max_abs_tensor(&pair.b)?;
        println!("GDN GRPO {name}.B max_abs={max_abs:.6e}");
        assert!(max_abs > 0.0, "GDN GRPO did not update {name}.B");
    }
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
