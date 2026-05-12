//! Minimal CUDA-native training loop helpers.
//!
//! This module is intentionally tiny while the CUDA-native Qwen path is still
//! being built. It bridges `kiln_model::cuda_train` tensor/autograd primitives
//! to an optimizer step in the training crate, mirroring the direction of the
//! Vulkan `vk_train` module without claiming full model coverage yet.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, TensorId};
use kiln_model::cuda_train::{
    CudaAdamWConfig, CudaAdamWState, CudaFullAttentionLayer, CudaRopeTables, CudaTrainArena,
    CudaTrainTensor, cuda_adamw_step_from_store, cuda_add, cuda_backward, cuda_embedding_lookup,
    cuda_full_attention_layer, cuda_matmul, cuda_mul, cuda_narrow_last_dim, cuda_reshape,
    cuda_rmsnorm, cuda_rope, cuda_scale, cuda_sdpa_prefill_causal, cuda_sigmoid, cuda_silu,
    cuda_sum_all, cuda_transpose2d,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub type CudaAdamWBook = HashMap<TensorId, CudaAdamWState>;

/// Trainable LoRA pair held as CUDA training tensors.
#[derive(Clone)]
pub struct CudaLoraPair {
    pub a: CudaTrainTensor,
    pub b: CudaTrainTensor,
    pub a_id: TensorId,
    pub b_id: TensorId,
    pub scale: f32,
}

impl CudaLoraPair {
    /// Initialize a fresh LoRA pair on the CUDA device.
    ///
    /// A is Kaiming-uniform and B is zero, matching the existing trainer and
    /// Vulkan-native initialization contract.
    pub fn init_kaiming(
        device: &Device,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        seed: u64,
    ) -> Result<Self> {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        ensure!(rank > 0, "CudaLoraPair rank must be non-zero");
        ensure!(
            in_features > 0 && out_features > 0,
            "CudaLoraPair feature dimensions must be non-zero"
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let bound = (1.0_f32 / (in_features as f32)).sqrt();
        let a_data: Vec<f32> = (0..(rank * in_features))
            .map(|_| rng.random_range(-bound..bound))
            .collect();
        let b_data: Vec<f32> = vec![0.0_f32; out_features * rank];

        let a_tensor = Tensor::from_vec(a_data, (rank, in_features), device)?;
        let a_id = a_tensor.id();
        let b_tensor = Tensor::from_vec(b_data, (out_features, rank), device)?;
        let b_id = b_tensor.id();
        Ok(Self {
            a: CudaTrainTensor::parameter(a_tensor, a_id)?,
            b: CudaTrainTensor::parameter(b_tensor, b_id)?,
            a_id,
            b_id,
            scale: alpha / (rank as f32),
        })
    }
}

/// Trainable LoRA params for one transformer layer.
#[derive(Clone, Default)]
pub struct CudaLoraLayer {
    pub q_proj: Option<CudaLoraPair>,
    pub k_proj: Option<CudaLoraPair>,
    pub v_proj: Option<CudaLoraPair>,
    pub o_proj: Option<CudaLoraPair>,
    pub gate_proj: Option<CudaLoraPair>,
    pub up_proj: Option<CudaLoraPair>,
    pub down_proj: Option<CudaLoraPair>,
    pub in_proj_qkv: Option<CudaLoraPair>,
    pub in_proj_z: Option<CudaLoraPair>,
    pub gdn_out_proj: Option<CudaLoraPair>,
}

fn cuda_lora_pairs<'a>(
    layers: &'a [CudaLoraLayer],
) -> impl Iterator<Item = &'a CudaLoraPair> + 'a {
    layers.iter().flat_map(|layer| {
        [
            layer.q_proj.as_ref(),
            layer.k_proj.as_ref(),
            layer.v_proj.as_ref(),
            layer.o_proj.as_ref(),
            layer.gate_proj.as_ref(),
            layer.up_proj.as_ref(),
            layer.down_proj.as_ref(),
        ]
        .into_iter()
        .flatten()
    })
}

pub fn allocate_cuda_adamw_state(params: &[CudaTrainTensor]) -> Result<CudaAdamWBook> {
    let mut states = HashMap::new();
    for param in params {
        let Some(param_id) = param.param_id() else {
            continue;
        };
        states.insert(param_id, CudaAdamWState::zeros_like(param)?);
    }
    Ok(states)
}

pub fn allocate_cuda_lora_adamw_state(lora_layers: &[CudaLoraLayer]) -> Result<CudaAdamWBook> {
    let mut states = HashMap::new();
    for pair in cuda_lora_pairs(lora_layers) {
        states.insert(pair.a_id, CudaAdamWState::zeros_like(&pair.a)?);
        states.insert(pair.b_id, CudaAdamWState::zeros_like(&pair.b)?);
    }
    Ok(states)
}

/// Apply `input @ base_weight + scale * input @ A.T @ B.T`.
pub fn cuda_lora_linear(
    input: &CudaTrainTensor,
    base_weight: &CudaTrainTensor,
    lora: Option<&CudaLoraPair>,
) -> Result<CudaTrainTensor> {
    let base = cuda_matmul(input, base_weight).context("cuda LoRA linear base projection")?;
    let Some(pair) = lora else {
        return Ok(base);
    };
    let a_t = cuda_transpose2d(&pair.a).context("cuda LoRA linear A transpose")?;
    let hidden = cuda_matmul(input, &a_t).context("cuda LoRA linear A projection")?;
    let b_t = cuda_transpose2d(&pair.b).context("cuda LoRA linear B transpose")?;
    let delta = cuda_matmul(&hidden, &b_t).context("cuda LoRA linear B projection")?;
    let scaled = cuda_scale(&delta, pair.scale).context("cuda LoRA linear scale")?;
    cuda_add(&base, &scaled).context("cuda LoRA linear add")
}

fn cuda_lora_per_head_rmsnorm_flat(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() == 2 && input.dims()[1] == heads * head_dim,
        "cuda_lora_per_head_rmsnorm_flat: expected [rows, heads*head_dim], got {:?}",
        input.dims()
    );
    let rows = input.dims()[0];
    let flat = cuda_reshape(input, &[rows * heads, head_dim])?;
    let normed = cuda_rmsnorm(&flat, weight, eps)?;
    cuda_reshape(&normed, &[rows, heads * head_dim])
}

fn cuda_lora_apply_rope_to_flat(
    input: &CudaTrainTensor,
    cos: &CudaTrainTensor,
    sin: &CudaTrainTensor,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() == 2 && input.dims()[1] == heads * head_dim,
        "cuda_lora_apply_rope_to_flat: expected [rows, heads*head_dim], got {:?}",
        input.dims()
    );
    let rows = input.dims()[0];
    let rank3 = cuda_reshape(input, &[rows, heads, head_dim])?;
    let rotated = cuda_rope(&rank3, cos, sin, rotary_dim)?;
    cuda_reshape(&rotated, &[rows, heads * head_dim])
}

pub fn cuda_full_attention_lora_layer(
    input: &CudaTrainTensor,
    weights: &CudaFullAttentionLayer<'_>,
    lora: &CudaLoraLayer,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() == 2,
        "cuda_full_attention_lora_layer: expected rank-2 [rows, hidden] input, got {:?}",
        input.dims()
    );
    let rows = input.dims()[0];
    let q_dim = weights.heads_q * weights.head_dim;

    let h_norm = cuda_rmsnorm(input, weights.input_norm_weight, weights.eps)?;
    let q_raw = cuda_lora_linear(&h_norm, weights.q_weight, lora.q_proj.as_ref())?;
    let k = cuda_lora_linear(&h_norm, weights.k_weight, lora.k_proj.as_ref())?;
    let v = cuda_lora_linear(&h_norm, weights.v_weight, lora.v_proj.as_ref())?;

    let (q, gate) = if weights.attn_output_gate {
        let q_raw_3d = cuda_reshape(&q_raw, &[rows, weights.heads_q, weights.head_dim * 2])?;
        let q_3d = cuda_narrow_last_dim(&q_raw_3d, 0, weights.head_dim)?;
        let gate_3d = cuda_narrow_last_dim(&q_raw_3d, weights.head_dim, weights.head_dim)?;
        (
            cuda_reshape(&q_3d, &[rows, q_dim])?,
            Some(cuda_reshape(&gate_3d, &[rows, q_dim])?),
        )
    } else {
        (q_raw, None)
    };
    let q = match weights.q_norm_weight {
        Some(weight) => cuda_lora_per_head_rmsnorm_flat(
            &q,
            weight,
            weights.heads_q,
            weights.head_dim,
            weights.eps,
        )?,
        None => q,
    };
    let k = match weights.k_norm_weight {
        Some(weight) => cuda_lora_per_head_rmsnorm_flat(
            &k,
            weight,
            weights.heads_kv,
            weights.head_dim,
            weights.eps,
        )?,
        None => k,
    };
    let (q, k) = match &weights.rope {
        Some(rope) => (
            cuda_lora_apply_rope_to_flat(
                &q,
                rope.cos,
                rope.sin,
                weights.heads_q,
                weights.head_dim,
                rope.rotary_dim,
            )?,
            cuda_lora_apply_rope_to_flat(
                &k,
                rope.cos,
                rope.sin,
                weights.heads_kv,
                weights.head_dim,
                rope.rotary_dim,
            )?,
        ),
        None => (q, k),
    };
    let q_3d = cuda_reshape(&q, &[rows, weights.heads_q, weights.head_dim])?;
    let k_3d = cuda_reshape(&k, &[rows, weights.heads_kv, weights.head_dim])?;
    let v_3d = cuda_reshape(&v, &[rows, weights.heads_kv, weights.head_dim])?;
    let scale = 1.0f32 / (weights.head_dim as f32).sqrt();
    let attn = cuda_sdpa_prefill_causal(&q_3d, &k_3d, &v_3d, scale)?;
    let attn_flat = cuda_reshape(&attn, &[rows, q_dim])?;
    let attn_gated = match gate {
        Some(gate) => {
            let gate = cuda_sigmoid(&gate)?;
            cuda_mul(&attn_flat, &gate)?
        }
        None => attn_flat,
    };
    let o_out = cuda_lora_linear(&attn_gated, weights.o_weight, lora.o_proj.as_ref())?;
    let residual = cuda_add(input, &o_out)?;

    let post_norm = cuda_rmsnorm(&residual, weights.post_norm_weight, weights.eps)?;
    let gate = cuda_lora_linear(&post_norm, weights.gate_weight, lora.gate_proj.as_ref())?;
    let up = cuda_lora_linear(&post_norm, weights.up_weight, lora.up_proj.as_ref())?;
    let activated = cuda_silu(&gate)?;
    let mlp_hidden = cuda_mul(&activated, &up)?;
    let mlp = cuda_lora_linear(&mlp_hidden, weights.down_weight, lora.down_proj.as_ref())?;
    cuda_add(&residual, &mlp)
}

/// Run one native CUDA linear training step for `loss = sum((input @ weight)^2)`.
pub fn cuda_linear_sum_square_adamw_step(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
) -> Result<f32> {
    let mut arena = CudaTrainArena::new(input.as_tensor().device())?;
    cuda_linear_sum_square_adamw_step_with_arena(input, weight, adamw_state, cfg, &mut arena)
}

/// Run one native CUDA linear training step using caller-owned arena accounting.
pub fn cuda_linear_sum_square_adamw_step_with_arena(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
    arena: &mut CudaTrainArena,
) -> Result<f32> {
    ensure!(
        weight.param_id().is_some(),
        "cuda_linear_sum_square_adamw_step requires a parameter weight"
    );
    let output = arena.track(cuda_matmul(input, weight).context("cuda linear forward")?)?;
    let squared = arena.track(cuda_mul(&output, &output).context("cuda linear square loss")?)?;
    let loss = arena.track(cuda_sum_all(&squared).context("cuda linear reduce loss")?)?;
    let loss_value = loss.to_vec_f32()?[0];
    let grads = cuda_backward(&loss).context("cuda linear backward")?;
    let updated = cuda_adamw_step_from_store(&[weight.clone()], &grads, adamw_state, cfg)
        .context("cuda linear AdamW step")?;
    ensure!(
        updated == 1,
        "cuda_linear_sum_square_adamw_step expected one updated parameter, got {updated}"
    );
    Ok(loss_value)
}

/// Run one native CUDA FullAttention-layer training step for `loss = sum(layer(input)^2)`.
pub fn cuda_full_attention_sum_square_adamw_step_with_arena(
    input: &CudaTrainTensor,
    weights: &CudaFullAttentionLayer<'_>,
    trainable_params: &[CudaTrainTensor],
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
    arena: &mut CudaTrainArena,
) -> Result<f32> {
    ensure!(
        !trainable_params.is_empty(),
        "cuda_full_attention_sum_square_adamw_step requires trainable params"
    );
    for param in trainable_params {
        ensure!(
            param.param_id().is_some(),
            "cuda_full_attention_sum_square_adamw_step trainable params must be parameters"
        );
    }

    let output = arena
        .track(cuda_full_attention_layer(input, weights).context("cuda FullAttention forward")?)?;
    let squared = arena.track(cuda_mul(&output, &output).context("cuda FullAttention square")?)?;
    let loss = arena.track(cuda_sum_all(&squared).context("cuda FullAttention reduce loss")?)?;
    let loss_value = loss.to_vec_f32()?[0];
    let grads = cuda_backward(&loss).context("cuda FullAttention backward")?;
    let updated = cuda_adamw_step_from_store(trainable_params, &grads, adamw_state, cfg)
        .context("cuda FullAttention AdamW step")?;
    ensure!(
        updated > 0,
        "cuda_full_attention_sum_square_adamw_step expected at least one updated parameter"
    );
    Ok(loss_value)
}

/// Run one tiny native CUDA model step:
/// embedding lookup -> FullAttention layer -> final RMSNorm -> LM head -> sum-square loss.
pub fn cuda_tiny_full_attention_model_adamw_step_with_arena(
    token_embedding: &CudaTrainTensor,
    token_ids: &[usize],
    layer_weights: &CudaFullAttentionLayer<'_>,
    final_norm_weight: &CudaTrainTensor,
    lm_head_weight: &CudaTrainTensor,
    trainable_params: &[CudaTrainTensor],
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
    arena: &mut CudaTrainArena,
) -> Result<f32> {
    ensure!(
        !token_ids.is_empty(),
        "cuda_tiny_full_attention_model_adamw_step requires token ids"
    );
    ensure!(
        !trainable_params.is_empty(),
        "cuda_tiny_full_attention_model_adamw_step requires trainable params"
    );
    for param in trainable_params {
        ensure!(
            param.param_id().is_some(),
            "cuda_tiny_full_attention_model_adamw_step trainable params must be parameters"
        );
    }

    let embedded = arena.track(
        cuda_embedding_lookup(token_embedding, token_ids).context("cuda tiny model embedding")?,
    )?;
    let hidden = arena.track(
        cuda_full_attention_layer(&embedded, layer_weights)
            .context("cuda tiny model FullAttention layer")?,
    )?;
    let normed = arena.track(
        cuda_rmsnorm(&hidden, final_norm_weight, layer_weights.eps)
            .context("cuda tiny model final RMSNorm")?,
    )?;
    let logits = arena.track(cuda_matmul(&normed, lm_head_weight).context("cuda tiny model LM head")?)?;
    let squared = arena.track(cuda_mul(&logits, &logits).context("cuda tiny model square")?)?;
    let loss = arena.track(cuda_sum_all(&squared).context("cuda tiny model reduce loss")?)?;
    let loss_value = loss.to_vec_f32()?[0];
    let grads = cuda_backward(&loss).context("cuda tiny model backward")?;
    let updated = cuda_adamw_step_from_store(trainable_params, &grads, adamw_state, cfg)
        .context("cuda tiny model AdamW step")?;
    ensure!(
        updated > 0,
        "cuda_tiny_full_attention_model_adamw_step expected at least one updated parameter"
    );
    Ok(loss_value)
}

/// Save named CUDA training tensors to safetensors after one CUDA-to-CPU readback.
pub fn save_cuda_training_tensors(
    weights: &[(&str, CudaTrainTensor)],
    output_path: &Path,
) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, weight) in weights {
        ensure!(!name.is_empty(), "CUDA training safetensors key must not be empty");
        let tensor = weight
            .as_tensor()
            .to_dtype(DType::F32)
            .with_context(|| format!("convert CUDA training tensor {name} to f32"))?
            .to_device(&Device::Cpu)
            .with_context(|| format!("read CUDA training tensor {name} to CPU"))?;
        tensors.insert((*name).to_string(), tensor);
    }
    candle_core::safetensors::save(&tensors, output_path)
        .with_context(|| format!("save CUDA training tensors {}", output_path.display()))?;
    Ok(())
}

/// Save CUDA-native LoRA adapter tensors using PEFT-compatible safetensors keys.
pub fn save_cuda_lora_adapter(
    lora_layers: &[CudaLoraLayer],
    rank: usize,
    alpha: f32,
    output_path: &Path,
) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (li, layer) in lora_layers.iter().enumerate() {
        for (name, proj) in [
            ("q_proj", layer.q_proj.as_ref()),
            ("k_proj", layer.k_proj.as_ref()),
            ("v_proj", layer.v_proj.as_ref()),
            ("o_proj", layer.o_proj.as_ref()),
            ("gate_proj", layer.gate_proj.as_ref()),
            ("up_proj", layer.up_proj.as_ref()),
            ("down_proj", layer.down_proj.as_ref()),
        ] {
            let Some(pair) = proj else { continue };
            tensors.insert(
                format!(
                    "base_model.model.model.layers.{}.{}.lora_A.weight",
                    li, name
                ),
                pair.a
                    .as_tensor()
                    .to_dtype(DType::F32)
                    .with_context(|| format!("convert CUDA LoRA layer {li} {name} A to f32"))?
                    .to_device(&Device::Cpu)
                    .with_context(|| format!("read CUDA LoRA layer {li} {name} A to CPU"))?,
            );
            tensors.insert(
                format!(
                    "base_model.model.model.layers.{}.{}.lora_B.weight",
                    li, name
                ),
                pair.b
                    .as_tensor()
                    .to_dtype(DType::F32)
                    .with_context(|| format!("convert CUDA LoRA layer {li} {name} B to f32"))?
                    .to_device(&Device::Cpu)
                    .with_context(|| format!("read CUDA LoRA layer {li} {name} B to CPU"))?,
            );
        }
    }
    candle_core::safetensors::save(&tensors, output_path)
        .with_context(|| format!("save CUDA LoRA adapter {}", output_path.display()))?;
    let _ = (rank, alpha);
    Ok(())
}

pub fn write_cuda_adapter_config(output_dir: &Path, rank: usize, alpha: f32) -> Result<()> {
    let cfg = serde_json::json!({
        "base_model_name_or_path": "",
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": true,
        "init_lora_weights": true,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "modules_to_save": null,
        "peft_type": "LORA",
        "r": rank,
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    });
    let path = output_dir.join("adapter_config.json");
    let contents = serde_json::to_string_pretty(&cfg)
        .with_context(|| format!("serialize CUDA adapter config {}", path.display()))?;
    std::fs::write(&path, contents)
        .with_context(|| format!("write CUDA adapter config {}", path.display()))?;
    Ok(())
}

pub fn save_cuda_lora_adapter_dir(
    lora_layers: &[CudaLoraLayer],
    rank: usize,
    alpha: f32,
    output_dir: &Path,
) -> Result<PathBuf> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create CUDA adapter dir {}", output_dir.display()))?;
    let adapter_path = output_dir.join("adapter_model.safetensors");
    save_cuda_lora_adapter(lora_layers, rank, alpha, &adapter_path)
        .with_context(|| format!("save CUDA adapter tensors {}", adapter_path.display()))?;
    write_cuda_adapter_config(output_dir, rank, alpha)
        .with_context(|| format!("write CUDA adapter config {}", output_dir.display()))?;
    Ok(output_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn test_lora_pair(
        device: &Device,
        in_features: usize,
        out_features: usize,
        rank: usize,
        scale: f32,
        offset: f32,
    ) -> Result<CudaLoraPair> {
        let a_data: Vec<f32> = (0..rank * in_features)
            .map(|i| offset + (i as f32 + 1.0) * 0.03)
            .collect();
        let b_data: Vec<f32> = (0..out_features * rank)
            .map(|i| -offset + (i as f32 + 1.0) * 0.02)
            .collect();
        let a_tensor = Tensor::from_vec(a_data, (rank, in_features), device)?;
        let a_id = a_tensor.id();
        let b_tensor = Tensor::from_vec(b_data, (out_features, rank), device)?;
        let b_id = b_tensor.id();
        Ok(CudaLoraPair {
            a: CudaTrainTensor::parameter(a_tensor, a_id)?,
            b: CudaTrainTensor::parameter(b_tensor, b_id)?,
            a_id,
            b_id,
            scale,
        })
    }

    #[test]
    fn cuda_linear_adamw_train_step_decreases_loss() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda linear AdamW smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 2.0],
            (1usize, 2usize),
            &device,
        )?)?;
        let weight_tensor = Tensor::from_vec(vec![1.0f32, -2.0], (2usize, 1usize), &device)?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let mut adamw = allocate_cuda_adamw_state(&[weight.clone()])?;
        let cfg = CudaAdamWConfig {
            lr: 0.1,
            ..CudaAdamWConfig::default()
        };

        let first = cuda_linear_sum_square_adamw_step(&input, &weight, &mut adamw, cfg)?;
        let second = cuda_linear_sum_square_adamw_step(&input, &weight, &mut adamw, cfg)?;
        assert!(
            second < first,
            "expected native CUDA linear AdamW loss to decrease: first={first} second={second}"
        );
        assert_eq!(adamw.get(&weight_id).expect("state").step, 2);
        Ok(())
    }

    #[test]
    fn cuda_linear_adamw_train_step_uses_arena_accounting() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda linear arena smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 2.0],
            (1usize, 2usize),
            &device,
        )?)?;
        let weight_tensor = Tensor::from_vec(vec![1.0f32, -2.0], (2usize, 1usize), &device)?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let mut adamw = allocate_cuda_adamw_state(&[weight.clone()])?;
        let mut arena = CudaTrainArena::new(&device)?;

        let loss = cuda_linear_sum_square_adamw_step_with_arena(
            &input,
            &weight,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.1,
                ..CudaAdamWConfig::default()
            },
            &mut arena,
        )?;
        assert!(loss > 0.0);
        assert_eq!(arena.allocation_count(), 3);
        assert!(arena.allocated_bytes() >= 12);
        arena.clear();
        assert_eq!(arena.allocation_count(), 0);
        Ok(())
    }

    #[test]
    fn cuda_lora_linear_adamw_updates_lora_pair() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda LoRA linear AdamW smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.5f32, -1.0, 1.5, 0.25],
            (2usize, 2usize),
            &device,
        )?)?;
        let base = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.2f32, -0.4, 0.1, 0.3],
            (2usize, 2usize),
            &device,
        )?)?;
        let lora = test_lora_pair(&device, 2, 2, 2, 2.0, 0.05)?;
        let mut adamw = allocate_cuda_adamw_state(&[lora.a.clone(), lora.b.clone()])?;
        let a_before = lora.a.to_vec_f32()?;
        let b_before = lora.b.to_vec_f32()?;

        let output = cuda_lora_linear(&input, &base, Some(&lora))?;
        let squared = cuda_mul(&output, &output)?;
        let loss = cuda_sum_all(&squared)?;
        let loss_value = loss.to_vec_f32()?[0];
        let grads = cuda_backward(&loss)?;
        let updated = cuda_adamw_step_from_store(
            &[lora.a.clone(), lora.b.clone()],
            &grads,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
        )?;

        assert!(loss_value.is_finite() && loss_value > 0.0);
        assert_eq!(updated, 2);
        assert_ne!(lora.a.to_vec_f32()?, a_before);
        assert_ne!(lora.b.to_vec_f32()?, b_before);
        assert_eq!(adamw.get(&lora.a_id).expect("LoRA A state").step, 1);
        assert_eq!(adamw.get(&lora.b_id).expect("LoRA B state").step, 1);
        Ok(())
    }

    #[test]
    fn cuda_full_attention_lora_layer_adamw_updates_lora_pair() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda FullAttention LoRA smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.25f32, -0.4, 0.75, 0.1],
            (2usize, 2usize),
            &device,
        )?)?;
        let input_norm = CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?;
        let q_weight = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.2f32, -0.3, 0.05, 0.4],
            (2usize, 2usize),
            &device,
        )?)?;
        let k_weight = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.1f32, 0.6, 0.8, -0.2],
            (2usize, 2usize),
            &device,
        )?)?;
        let v_weight = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.7f32, -0.2, -0.5, 0.6],
            (2usize, 2usize),
            &device,
        )?)?;
        let q_norm = CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?;
        let k_norm = CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?;
        let o_weight = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.3f32, -0.4, 0.8, 0.2],
            (2usize, 2usize),
            &device,
        )?)?;
        let post_norm = CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?;
        let gate_weight = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.25f32, -0.15, 0.35, 0.05],
            (2usize, 2usize),
            &device,
        )?)?;
        let up_weight = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.45f32, 0.2, -0.1, 0.55],
            (2usize, 2usize),
            &device,
        )?)?;
        let down_weight = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.6f32, -0.25, 0.15, 0.5],
            (2usize, 2usize),
            &device,
        )?)?;
        let layer = CudaFullAttentionLayer {
            input_norm_weight: &input_norm,
            q_weight: &q_weight,
            k_weight: &k_weight,
            v_weight: &v_weight,
            q_norm_weight: Some(&q_norm),
            k_norm_weight: Some(&k_norm),
            o_weight: &o_weight,
            post_norm_weight: &post_norm,
            gate_weight: &gate_weight,
            up_weight: &up_weight,
            down_weight: &down_weight,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 2,
            eps: 1e-6,
            attn_output_gate: false,
            rope: None,
        };
        let lora = CudaLoraLayer {
            q_proj: Some(test_lora_pair(&device, 2, 2, 2, 2.0, 0.01)?),
            down_proj: Some(test_lora_pair(&device, 2, 2, 2, 2.0, 0.03)?),
            ..Default::default()
        };
        let trainable = vec![
            lora.q_proj.as_ref().expect("q lora").a.clone(),
            lora.q_proj.as_ref().expect("q lora").b.clone(),
            lora.down_proj.as_ref().expect("down lora").a.clone(),
            lora.down_proj.as_ref().expect("down lora").b.clone(),
        ];
        let before: Vec<Vec<f32>> = trainable
            .iter()
            .map(CudaTrainTensor::to_vec_f32)
            .collect::<Result<_>>()?;
        let mut adamw = allocate_cuda_adamw_state(&trainable)?;

        let output = cuda_full_attention_lora_layer(&input, &layer, &lora)?;
        let squared = cuda_mul(&output, &output)?;
        let loss = cuda_sum_all(&squared)?;
        let loss_value = loss.to_vec_f32()?[0];
        let grads = cuda_backward(&loss)?;
        let updated = cuda_adamw_step_from_store(
            &trainable,
            &grads,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
        )?;

        assert!(loss_value.is_finite() && loss_value > 0.0);
        assert_eq!(updated, trainable.len());
        for (param, old) in trainable.iter().zip(before.iter()) {
            assert_ne!(param.to_vec_f32()?, *old);
            assert_eq!(adamw.get(&param.param_id().expect("param id")).expect("state").step, 1);
        }
        Ok(())
    }

    #[test]
    fn cuda_full_attention_adamw_train_step_updates_projection_weight() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda FullAttention AdamW smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.5f32, -1.0, 1.5, 0.25],
            (2usize, 2usize),
            &device,
        )?)?;
        let input_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let q_tensor = Tensor::from_vec(
            vec![0.2f32, -0.3, 0.05, 0.4, 0.4, 0.1, -0.2, 0.3],
            (2usize, 4usize),
            &device,
        )?;
        let q_id = q_tensor.id();
        let q_weight = CudaTrainTensor::parameter(q_tensor, q_id)?;
        let k_tensor = Tensor::from_vec(vec![0.1f32, 0.6, 0.8, -0.2], (2usize, 2usize), &device)?;
        let k_id = k_tensor.id();
        let k_weight = CudaTrainTensor::parameter(k_tensor, k_id)?;
        let v_tensor = Tensor::from_vec(vec![0.7f32, -0.2, -0.5, 0.6], (2usize, 2usize), &device)?;
        let v_id = v_tensor.id();
        let v_weight = CudaTrainTensor::parameter(v_tensor, v_id)?;
        let q_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let k_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let o_tensor = Tensor::from_vec(vec![0.3f32, -0.4, 0.8, 0.2], (2usize, 2usize), &device)?;
        let o_id = o_tensor.id();
        let o_weight = CudaTrainTensor::parameter(o_tensor, o_id)?;
        let post_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let gate_tensor =
            Tensor::from_vec(vec![0.25f32, -0.15, 0.35, 0.05], (2usize, 2usize), &device)?;
        let gate_id = gate_tensor.id();
        let gate_weight = CudaTrainTensor::parameter(gate_tensor, gate_id)?;
        let up_tensor = Tensor::from_vec(vec![0.45f32, 0.2, -0.1, 0.55], (2usize, 2usize), &device)?;
        let up_id = up_tensor.id();
        let up_weight = CudaTrainTensor::parameter(up_tensor, up_id)?;
        let down_tensor =
            Tensor::from_vec(vec![0.6f32, -0.25, 0.15, 0.5], (2usize, 2usize), &device)?;
        let down_id = down_tensor.id();
        let down_weight = CudaTrainTensor::parameter(down_tensor, down_id)?;
        let cos = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 0.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let sin = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 1.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let weights = CudaFullAttentionLayer {
            input_norm_weight: &input_norm,
            q_weight: &q_weight,
            k_weight: &k_weight,
            v_weight: &v_weight,
            q_norm_weight: Some(&q_norm),
            k_norm_weight: Some(&k_norm),
            o_weight: &o_weight,
            post_norm_weight: &post_norm,
            gate_weight: &gate_weight,
            up_weight: &up_weight,
            down_weight: &down_weight,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 2,
            eps: 1e-6,
            attn_output_gate: true,
            rope: Some(CudaRopeTables {
                cos: &cos,
                sin: &sin,
                rotary_dim: 2,
            }),
        };
        let trainable = vec![
            q_weight.clone(),
            k_weight.clone(),
            v_weight.clone(),
            o_weight.clone(),
            gate_weight.clone(),
            up_weight.clone(),
            down_weight.clone(),
        ];
        let mut adamw = allocate_cuda_adamw_state(&trainable)?;
        let mut arena = CudaTrainArena::new(&device)?;
        let q_before = q_weight.to_vec_f32()?;

        let loss = cuda_full_attention_sum_square_adamw_step_with_arena(
            &input,
            &weights,
            &trainable,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
            &mut arena,
        )?;

        assert!(loss.is_finite() && loss > 0.0);
        assert_ne!(q_weight.to_vec_f32()?, q_before);
        assert_eq!(adamw.get(&q_id).expect("q state").step, 1);
        assert_eq!(arena.allocation_count(), 3);
        Ok(())
    }

    #[test]
    fn cuda_tiny_full_attention_model_adamw_step_updates_lm_head() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda tiny FullAttention model smoke: {err}");
                return Ok(());
            }
        };

        let embedding_tensor = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.3, 0.4, 0.5, -1.0, 1.5, 0.25],
            (4usize, 2usize),
            &device,
        )?;
        let embedding_id = embedding_tensor.id();
        let embedding = CudaTrainTensor::parameter(embedding_tensor, embedding_id)?;
        let input_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let q_tensor = Tensor::from_vec(
            vec![0.2f32, -0.3, 0.05, 0.4, 0.4, 0.1, -0.2, 0.3],
            (2usize, 4usize),
            &device,
        )?;
        let q_id = q_tensor.id();
        let q_weight = CudaTrainTensor::parameter(q_tensor, q_id)?;
        let k_tensor = Tensor::from_vec(vec![0.1f32, 0.6, 0.8, -0.2], (2usize, 2usize), &device)?;
        let k_id = k_tensor.id();
        let k_weight = CudaTrainTensor::parameter(k_tensor, k_id)?;
        let v_tensor = Tensor::from_vec(vec![0.7f32, -0.2, -0.5, 0.6], (2usize, 2usize), &device)?;
        let v_id = v_tensor.id();
        let v_weight = CudaTrainTensor::parameter(v_tensor, v_id)?;
        let q_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let k_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let o_tensor = Tensor::from_vec(vec![0.3f32, -0.4, 0.8, 0.2], (2usize, 2usize), &device)?;
        let o_id = o_tensor.id();
        let o_weight = CudaTrainTensor::parameter(o_tensor, o_id)?;
        let post_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let gate_tensor =
            Tensor::from_vec(vec![0.25f32, -0.15, 0.35, 0.05], (2usize, 2usize), &device)?;
        let gate_id = gate_tensor.id();
        let gate_weight = CudaTrainTensor::parameter(gate_tensor, gate_id)?;
        let up_tensor = Tensor::from_vec(vec![0.45f32, 0.2, -0.1, 0.55], (2usize, 2usize), &device)?;
        let up_id = up_tensor.id();
        let up_weight = CudaTrainTensor::parameter(up_tensor, up_id)?;
        let down_tensor =
            Tensor::from_vec(vec![0.6f32, -0.25, 0.15, 0.5], (2usize, 2usize), &device)?;
        let down_id = down_tensor.id();
        let down_weight = CudaTrainTensor::parameter(down_tensor, down_id)?;
        let final_norm = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 0.0],
            (2usize,),
            &device,
        )?)?;
        let lm_head_tensor = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.3, 0.4, 0.05, -0.2],
            (2usize, 3usize),
            &device,
        )?;
        let lm_head_id = lm_head_tensor.id();
        let lm_head = CudaTrainTensor::parameter(lm_head_tensor, lm_head_id)?;
        let cos = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 0.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let sin = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 1.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let layer = CudaFullAttentionLayer {
            input_norm_weight: &input_norm,
            q_weight: &q_weight,
            k_weight: &k_weight,
            v_weight: &v_weight,
            q_norm_weight: Some(&q_norm),
            k_norm_weight: Some(&k_norm),
            o_weight: &o_weight,
            post_norm_weight: &post_norm,
            gate_weight: &gate_weight,
            up_weight: &up_weight,
            down_weight: &down_weight,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 2,
            eps: 1e-6,
            attn_output_gate: true,
            rope: Some(CudaRopeTables {
                cos: &cos,
                sin: &sin,
                rotary_dim: 2,
            }),
        };
        let trainable = vec![
            embedding.clone(),
            q_weight.clone(),
            k_weight.clone(),
            v_weight.clone(),
            o_weight.clone(),
            gate_weight.clone(),
            up_weight.clone(),
            down_weight.clone(),
            lm_head.clone(),
        ];
        let mut adamw = allocate_cuda_adamw_state(&trainable)?;
        let mut arena = CudaTrainArena::new(&device)?;
        let lm_head_before = lm_head.to_vec_f32()?;

        let loss = cuda_tiny_full_attention_model_adamw_step_with_arena(
            &embedding,
            &[2, 0],
            &layer,
            &final_norm,
            &lm_head,
            &trainable,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
            &mut arena,
        )?;

        assert!(loss.is_finite() && loss > 0.0);
        assert_ne!(lm_head.to_vec_f32()?, lm_head_before);
        assert_eq!(adamw.get(&lm_head_id).expect("lm_head state").step, 1);
        assert_eq!(arena.allocation_count(), 6);
        Ok(())
    }

    #[test]
    fn cuda_linear_weight_save_reflects_updated_cuda_tensor() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda linear save smoke: {err}");
                return Ok(());
            }
        };

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 2.0],
            (1usize, 2usize),
            &device,
        )?)?;
        let weight_tensor = Tensor::from_vec(vec![1.0f32, -2.0], (2usize, 1usize), &device)?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let mut adamw = allocate_cuda_adamw_state(&[weight.clone()])?;
        let mut arena = CudaTrainArena::new(&device)?;

        for _ in 0..2 {
            let loss = cuda_linear_sum_square_adamw_step_with_arena(
                &input,
                &weight,
                &mut adamw,
                CudaAdamWConfig {
                    lr: 0.1,
                    ..CudaAdamWConfig::default()
                },
                &mut arena,
            )?;
            assert!(loss.is_finite());
            arena.clear();
        }

        let expected = weight.to_vec_f32()?;
        let tmp = std::env::temp_dir().join(format!(
            "kiln-cuda-linear-weight-{}.safetensors",
            std::process::id()
        ));
        save_cuda_training_tensors(&[("linear.weight", weight.clone())], &tmp)?;

        let loaded = candle_core::safetensors::load(&tmp, &Device::Cpu)?;
        let saved = loaded
            .get("linear.weight")
            .context("missing saved linear.weight")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(saved, expected);
        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }

    #[test]
    fn cuda_lora_adapter_save_uses_peft_keys() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda LoRA adapter save smoke: {err}");
                return Ok(());
            }
        };

        let q_proj = CudaLoraPair::init_kaiming(&device, 3, 2, 2, 4.0, 0xC0DA_10AA)?;
        let up_proj = CudaLoraPair::init_kaiming(&device, 3, 4, 2, 4.0, 0xC0DA_10BB)?;
        let q_a_expected = q_proj.a.to_vec_f32()?;
        let q_b_expected = q_proj.b.to_vec_f32()?;
        let up_a_expected = up_proj.a.to_vec_f32()?;
        let up_b_expected = up_proj.b.to_vec_f32()?;
        let layers = vec![CudaLoraLayer {
            q_proj: Some(q_proj),
            up_proj: Some(up_proj),
            ..Default::default()
        }];
        let adamw = allocate_cuda_lora_adamw_state(&layers)?;
        assert_eq!(adamw.len(), 4);

        let tmp = std::env::temp_dir().join(format!(
            "kiln-cuda-lora-adapter-{}.safetensors",
            std::process::id()
        ));
        save_cuda_lora_adapter(&layers, 2, 4.0, &tmp)?;

        let loaded = candle_core::safetensors::load(&tmp, &Device::Cpu)?;
        let q_a = loaded
            .get("base_model.model.model.layers.0.q_proj.lora_A.weight")
            .context("missing q_proj lora_A")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let q_b = loaded
            .get("base_model.model.model.layers.0.q_proj.lora_B.weight")
            .context("missing q_proj lora_B")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let up_a = loaded
            .get("base_model.model.model.layers.0.up_proj.lora_A.weight")
            .context("missing up_proj lora_A")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let up_b = loaded
            .get("base_model.model.model.layers.0.up_proj.lora_B.weight")
            .context("missing up_proj lora_B")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(q_a, q_a_expected);
        assert_eq!(q_b, q_b_expected);
        assert_eq!(up_a, up_a_expected);
        assert_eq!(up_b, up_b_expected);
        assert_eq!(loaded.len(), 4);
        let _ = std::fs::remove_file(&tmp);

        let out_dir = std::env::temp_dir().join(format!(
            "kiln-cuda-lora-adapter-dir-{}",
            std::process::id()
        ));
        let saved_dir = save_cuda_lora_adapter_dir(&layers, 2, 4.0, &out_dir)?;
        assert_eq!(saved_dir, out_dir);
        assert!(saved_dir.join("adapter_model.safetensors").exists());
        let config_text = std::fs::read_to_string(saved_dir.join("adapter_config.json"))?;
        let config: serde_json::Value = serde_json::from_str(&config_text)?;
        assert_eq!(config["peft_type"], "LORA");
        assert_eq!(config["task_type"], "CAUSAL_LM");
        assert_eq!(config["r"], 2);
        assert_eq!(config["lora_alpha"], 4.0);
        let _ = std::fs::remove_dir_all(&saved_dir);
        Ok(())
    }
}
