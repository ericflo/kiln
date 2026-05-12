//! Minimal CUDA-native training loop helpers.
//!
//! This module is intentionally tiny while the CUDA-native Qwen path is still
//! being built. It bridges `kiln_model::cuda_train` tensor/autograd primitives
//! to an optimizer step in the training crate, mirroring the direction of the
//! Vulkan `vk_train` module without claiming full model coverage yet.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, TensorId};
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::cuda_train::{
    CudaAdamWConfig, CudaAdamWState, CudaFullAttentionLayer, CudaLayerWeights, CudaModelWeights,
    CudaOwnedFullAttentionLayer, CudaOwnedLinearAttentionLayer, CudaRopeTables, CudaTrainArena,
    CudaTrainTensor, cuda_adamw_step_from_store, cuda_add, cuda_backward, cuda_embedding_lookup,
    cuda_full_attention_layer, cuda_matmul, cuda_mul, cuda_narrow_last_dim, cuda_reshape,
    cuda_rmsnorm, cuda_rope, cuda_scale, cuda_sdpa_prefill_causal, cuda_sigmoid, cuda_silu,
    cuda_sum_all, cuda_transpose2d,
};
use kiln_model::forward::GpuWeights;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::trainer::{ProgressCallback, TrainingProgress, tokenize_for_training};
use crate::{SftConfig, SftExample};

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
            layer.in_proj_qkv.as_ref(),
            layer.in_proj_z.as_ref(),
            layer.gdn_out_proj.as_ref(),
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

fn cuda_lora_for_weight(
    device: &Device,
    weight: &CudaTrainTensor,
    rank: usize,
    alpha: f32,
    seed: u64,
    name: &str,
) -> Result<CudaLoraPair> {
    ensure!(
        weight.dims().len() == 2,
        "cuda_lora_for_weight {name}: expected rank-2 [in,out] weight, got {:?}",
        weight.dims()
    );
    CudaLoraPair::init_kaiming(device, weight.dims()[0], weight.dims()[1], rank, alpha, seed)
        .with_context(|| format!("initialize CUDA LoRA pair for {name}"))
}

pub fn cuda_init_lora_layers(
    model: &CudaModelWeights,
    rank: usize,
    alpha: f32,
    seed: u64,
) -> Result<Vec<CudaLoraLayer>> {
    let device = model.token_embedding.as_tensor().device();
    let mut out = Vec::with_capacity(model.layers.len());
    for (idx, layer) in model.layers.iter().enumerate() {
        let layer_seed = seed
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add((idx as u64).wrapping_mul(13));
        let mk = |slot: u64, weight: &CudaTrainTensor, name: &str| -> Result<CudaLoraPair> {
            cuda_lora_for_weight(
                device,
                weight,
                rank,
                alpha,
                layer_seed.wrapping_add(slot),
                name,
            )
        };
        match layer {
            CudaLayerWeights::FullAttention(full) => out.push(CudaLoraLayer {
                q_proj: Some(mk(1, &full.q_weight, "q_proj")?),
                k_proj: Some(mk(2, &full.k_weight, "k_proj")?),
                v_proj: Some(mk(3, &full.v_weight, "v_proj")?),
                o_proj: Some(mk(4, &full.o_weight, "o_proj")?),
                gate_proj: Some(mk(5, &full.gate_weight, "gate_proj")?),
                up_proj: Some(mk(6, &full.up_weight, "up_proj")?),
                down_proj: Some(mk(7, &full.down_weight, "down_proj")?),
                ..Default::default()
            }),
            CudaLayerWeights::LinearAttention(linear) => out.push(CudaLoraLayer {
                in_proj_qkv: Some(mk(8, &linear.in_proj_qkv_weight, "in_proj_qkv")?),
                in_proj_z: Some(mk(9, &linear.in_proj_z_weight, "in_proj_z")?),
                gdn_out_proj: Some(mk(10, &linear.out_proj_weight, "gdn_out_proj")?),
                ..Default::default()
            }),
        }
    }
    Ok(out)
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

pub struct CudaGdnInputProjections {
    pub q: CudaTrainTensor,
    pub k: CudaTrainTensor,
    pub v: CudaTrainTensor,
    pub z: CudaTrainTensor,
}

/// Native CUDA front-end for one LinearAttention/GDN layer.
///
/// This covers the autograd-safe part before conv/gates/chunkwise recurrence:
/// input RMSNorm, LoRA-enabled q/k/v projection, LoRA-enabled z projection,
/// and q/k/v last-dimension splits.
pub fn cuda_gdn_lora_input_projections(
    input: &CudaTrainTensor,
    weights: &CudaOwnedLinearAttentionLayer,
    lora: &CudaLoraLayer,
) -> Result<CudaGdnInputProjections> {
    ensure!(
        input.dims().len() == 2,
        "cuda_gdn_lora_input_projections: expected rank-2 [rows, hidden], got {:?}",
        input.dims()
    );
    let qk_dim = weights.heads_k * weights.head_dim_k;
    let v_dim = weights.heads_v * weights.head_dim_v;
    let expected_qkv_dim = qk_dim * 2 + v_dim;

    let h_norm = cuda_rmsnorm(input, &weights.layer_norm_weight, weights.eps)?;
    let qkv = cuda_lora_linear(
        &h_norm,
        &weights.in_proj_qkv_weight,
        lora.in_proj_qkv.as_ref(),
    )?;
    ensure!(
        qkv.dims().last() == Some(&expected_qkv_dim),
        "cuda_gdn_lora_input_projections: qkv projection last dim {} != expected {}",
        qkv.dims().last().copied().unwrap_or(0),
        expected_qkv_dim
    );
    let z = cuda_lora_linear(&h_norm, &weights.in_proj_z_weight, lora.in_proj_z.as_ref())?;
    let q = cuda_narrow_last_dim(&qkv, 0, qk_dim)?;
    let k = cuda_narrow_last_dim(&qkv, qk_dim, qk_dim)?;
    let v = cuda_narrow_last_dim(&qkv, qk_dim * 2, v_dim)?;
    Ok(CudaGdnInputProjections { q, k, v, z })
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

fn cuda_compute_rope_tables(
    device: &Device,
    inv_freq: &[f32],
    rows: usize,
) -> Result<Option<(CudaTrainTensor, CudaTrainTensor)>> {
    if inv_freq.is_empty() {
        return Ok(None);
    }
    let half = inv_freq.len();
    let mut cos = vec![0.0f32; rows * half];
    let mut sin = vec![0.0f32; rows * half];
    for row in 0..rows {
        for col in 0..half {
            let phase = (row as f32) * inv_freq[col];
            cos[row * half + col] = phase.cos();
            sin[row * half + col] = phase.sin();
        }
    }
    let cos = Tensor::from_vec(cos, (rows, half), device).context("cuda model RoPE cos table")?;
    let sin = Tensor::from_vec(sin, (rows, half), device).context("cuda model RoPE sin table")?;
    Ok(Some((CudaTrainTensor::new(cos)?, CudaTrainTensor::new(sin)?)))
}

pub fn cuda_full_attention_lora_model_adamw_step_with_arena(
    model: &CudaModelWeights,
    lora_layers: &[CudaLoraLayer],
    token_ids: &[usize],
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
    arena: &mut CudaTrainArena,
) -> Result<f32> {
    ensure!(
        !token_ids.is_empty(),
        "cuda_full_attention_lora_model_adamw_step requires token ids"
    );
    ensure!(
        lora_layers.len() == model.layers.len(),
        "cuda_full_attention_lora_model_adamw_step lora/model layer mismatch: {} vs {}",
        lora_layers.len(),
        model.layers.len()
    );
    let trainable: Vec<CudaTrainTensor> = cuda_lora_pairs(lora_layers)
        .flat_map(|pair| [pair.a.clone(), pair.b.clone()])
        .collect();
    ensure!(
        !trainable.is_empty(),
        "cuda_full_attention_lora_model_adamw_step requires at least one LoRA parameter"
    );

    let mut hidden = arena.track(
        cuda_embedding_lookup(&model.token_embedding, token_ids)
            .context("cuda FullAttention LoRA model embedding")?,
    )?;
    let rope_tables = cuda_compute_rope_tables(
        hidden.as_tensor().device(),
        &model.rotary_inv_freq,
        token_ids.len(),
    )?;
    for (idx, layer) in model.layers.iter().enumerate() {
        let CudaLayerWeights::FullAttention(full) = layer else {
            anyhow::bail!(
                "cuda_full_attention_lora_model_adamw_step: LinearAttention layer {idx} is not wired yet"
            );
        };
        let rope = rope_tables.as_ref().map(|(cos, sin)| CudaRopeTables {
            cos,
            sin,
            rotary_dim: model.rotary_dim,
        });
        let borrowed = full.as_borrowed(rope);
        hidden = arena.track(
            cuda_full_attention_lora_layer(&hidden, &borrowed, &lora_layers[idx])
                .with_context(|| format!("cuda FullAttention LoRA model layer {idx}"))?,
        )?;
    }
    let normed = arena.track(
        cuda_rmsnorm(&hidden, &model.final_norm_weight, 1e-6)
            .context("cuda FullAttention LoRA model final RMSNorm")?,
    )?;
    let logits = arena.track(
        cuda_matmul(&normed, &model.lm_head_weight).context("cuda FullAttention LoRA model LM head")?,
    )?;
    let squared = arena.track(cuda_mul(&logits, &logits).context("cuda FullAttention LoRA model square")?)?;
    let loss = arena.track(cuda_sum_all(&squared).context("cuda FullAttention LoRA model loss")?)?;
    let loss_value = loss.to_vec_f32()?[0];
    let grads = cuda_backward(&loss).context("cuda FullAttention LoRA model backward")?;
    let updated = cuda_adamw_step_from_store(&trainable, &grads, adamw_state, cfg)
        .context("cuda FullAttention LoRA model AdamW")?;
    ensure!(
        updated == trainable.len(),
        "cuda_full_attention_lora_model_adamw_step updated {updated} params, expected {}",
        trainable.len()
    );
    Ok(loss_value)
}

pub fn cuda_full_attention_lora_train_token_sequences(
    model: &CudaModelWeights,
    lora_layers: &[CudaLoraLayer],
    token_sequences: &[Vec<usize>],
    epochs: usize,
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
) -> Result<Vec<f32>> {
    ensure!(
        epochs > 0,
        "cuda_full_attention_lora_train_token_sequences requires at least one epoch"
    );
    ensure!(
        !token_sequences.is_empty(),
        "cuda_full_attention_lora_train_token_sequences requires token sequences"
    );
    let device = model.token_embedding.as_tensor().device();
    let mut losses = Vec::with_capacity(epochs * token_sequences.len());
    for _epoch in 0..epochs {
        for token_ids in token_sequences {
            let mut arena = CudaTrainArena::new(device)?;
            let loss = cuda_full_attention_lora_model_adamw_step_with_arena(
                model,
                lora_layers,
                token_ids,
                adamw_state,
                cfg,
                &mut arena,
            )?;
            ensure!(
                loss.is_finite(),
                "cuda_full_attention_lora_train_token_sequences encountered non-finite loss {loss}"
            );
            losses.push(loss);
        }
    }
    Ok(losses)
}

#[allow(clippy::too_many_arguments)]
pub fn cuda_full_attention_lora_train_token_sequences_to_adapter(
    model: &CudaModelWeights,
    token_sequences: &[Vec<usize>],
    epochs: usize,
    rank: usize,
    alpha: f32,
    seed: u64,
    adamw_cfg: CudaAdamWConfig,
    output_dir: &Path,
) -> Result<(PathBuf, Vec<f32>)> {
    let lora_layers =
        cuda_init_lora_layers(model, rank, alpha, seed).context("initialize CUDA LoRA layers")?;
    let mut adamw =
        allocate_cuda_lora_adamw_state(&lora_layers).context("allocate CUDA LoRA AdamW state")?;
    let losses = cuda_full_attention_lora_train_token_sequences(
        model,
        &lora_layers,
        token_sequences,
        epochs,
        &mut adamw,
        adamw_cfg,
    )
    .context("train CUDA FullAttention LoRA token sequences")?;
    let adapter_dir = save_cuda_lora_adapter_dir(&lora_layers, rank, alpha, output_dir)
        .context("save CUDA FullAttention LoRA token adapter")?;
    Ok((adapter_dir, losses))
}

fn ensure_cuda_native_sft_supported(model: &CudaModelWeights) -> Result<()> {
    ensure!(
        !model.layers.is_empty(),
        "cuda_native_sft_train: model has no transformer layers"
    );
    let unsupported: Vec<usize> = model
        .layers
        .iter()
        .enumerate()
        .filter_map(|(idx, layer)| match layer {
            CudaLayerWeights::FullAttention(_) => None,
            CudaLayerWeights::LinearAttention(_) => Some(idx),
        })
        .collect();
    ensure!(
        unsupported.is_empty(),
        "cuda_native_sft_train currently supports FullAttention-only models; \
         LinearAttention/GDN layers are not wired yet at indices {:?}",
        unsupported
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn cuda_native_sft_train(
    examples: &[SftExample],
    config: &SftConfig,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    adapter_dir: &Path,
    adapter_name: &str,
    progress_cb: Option<ProgressCallback>,
) -> Result<PathBuf> {
    tracing::info!(
        num_examples = examples.len(),
        epochs = config.epochs,
        lr = config.learning_rate,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        "starting cuda-native SFT training"
    );

    let model = CudaModelWeights::from_gpu_weights(weights, model_config)
        .context("cuda_native_sft_train: import CUDA model weights")?;
    ensure_cuda_native_sft_supported(&model)?;
    let effective_seed = config.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xC0DA_5EED)
    });
    let lora_layers = cuda_init_lora_layers(
        &model,
        config.lora_rank,
        config.lora_alpha,
        effective_seed,
    )
    .context("cuda_native_sft_train: initialize LoRA layers")?;
    let mut adamw = allocate_cuda_lora_adamw_state(&lora_layers)
        .context("cuda_native_sft_train: allocate AdamW state")?;
    let cfg = CudaAdamWConfig {
        lr: config.learning_rate as f32,
        ..Default::default()
    };

    let tokenized: Vec<Vec<usize>> = examples
        .iter()
        .filter_map(|example| match tokenize_for_training(example, tokenizer) {
            Ok((ids, _mask)) => Some(ids.into_iter().map(|id| id as usize).collect()),
            Err(err) => {
                tracing::warn!("cuda-native: skipping example: {err}");
                None
            }
        })
        .collect();
    ensure!(
        !tokenized.is_empty(),
        "cuda_native_sft_train: no valid training examples after tokenization"
    );

    let total_steps = config.epochs * tokenized.len();
    let mut global_step = 0usize;
    let mut last_loss = 0.0f32;
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;
        for token_ids in &tokenized {
            global_step += 1;
            let mut arena = CudaTrainArena::new(model.token_embedding.as_tensor().device())?;
            let loss = cuda_full_attention_lora_model_adamw_step_with_arena(
                &model,
                &lora_layers,
                token_ids,
                &mut adamw,
                cfg,
                &mut arena,
            )
            .with_context(|| {
                format!(
                    "cuda_native_sft_train step {} epoch {}",
                    global_step,
                    epoch + 1
                )
            })?;
            ensure!(
                loss.is_finite(),
                "cuda_native_sft_train: non-finite loss {loss} at step {global_step}"
            );
            epoch_loss += loss;
            last_loss = loss;

            if let Some(interval) = config.checkpoint_interval {
                if interval > 0 && global_step % interval == 0 && global_step < total_steps {
                    let ckpt_dir = adapter_dir.join(format!("{adapter_name}-checkpoint-{global_step}"));
                    if let Err(err) = save_cuda_lora_adapter_dir(
                        &lora_layers,
                        config.lora_rank,
                        config.lora_alpha,
                        &ckpt_dir,
                    ) {
                        tracing::warn!(step = global_step, error = %err, "save cuda-native checkpoint failed");
                    }
                }
            }

            if let Some(ref cb) = progress_cb {
                cb(TrainingProgress {
                    epoch: epoch + 1,
                    total_epochs: config.epochs,
                    step: global_step,
                    total_steps,
                    loss: loss as f64,
                    progress: global_step as f32 / total_steps as f32,
                });
            }

            if global_step % 10 == 0 || global_step == total_steps {
                tracing::info!(
                    epoch = epoch + 1,
                    step = global_step,
                    total_steps,
                    loss = format!("{loss:.6}"),
                    "cuda-native training step"
                );
            }
        }
        let avg = epoch_loss / (tokenized.len() as f32);
        tracing::info!(
            epoch = epoch + 1,
            avg_loss = format!("{avg:.6}"),
            "cuda-native epoch complete"
        );
    }

    let output_dir = adapter_dir.join(adapter_name);
    save_cuda_lora_adapter_dir(
        &lora_layers,
        config.lora_rank,
        config.lora_alpha,
        &output_dir,
    )
    .with_context(|| format!("save final CUDA adapter to {}", output_dir.display()))?;
    tracing::info!(
        adapter = adapter_name,
        path = %output_dir.display(),
        final_loss = format!("{last_loss:.6}"),
        "cuda-native SFT training complete"
    );
    Ok(output_dir)
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
            ("in_proj_qkv", layer.in_proj_qkv.as_ref()),
            ("in_proj_z", layer.in_proj_z.as_ref()),
            ("gdn_out_proj", layer.gdn_out_proj.as_ref()),
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
            "gate_proj", "up_proj", "down_proj",
            "in_proj_qkv", "in_proj_z", "gdn_out_proj"
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
    fn cuda_full_attention_lora_model_step_updates_lora_pair() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda FullAttention LoRA model smoke: {err}");
                return Ok(());
            }
        };

        let token_embedding = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.1f32, -0.2, 0.3, 0.4, 0.5, -1.0, 1.5, 0.25],
            (4usize, 2usize),
            &device,
        )?)?;
        let layer = CudaOwnedFullAttentionLayer {
            input_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            q_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.2f32, -0.3, 0.05, 0.4],
                (2usize, 2usize),
                &device,
            )?)?,
            k_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.1f32, 0.6, 0.8, -0.2],
                (2usize, 2usize),
                &device,
            )?)?,
            v_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.7f32, -0.2, -0.5, 0.6],
                (2usize, 2usize),
                &device,
            )?)?,
            q_norm_weight: CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?,
            k_norm_weight: CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?,
            o_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.3f32, -0.4, 0.8, 0.2],
                (2usize, 2usize),
                &device,
            )?)?,
            post_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            gate_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.25f32, -0.15, 0.35, 0.05],
                (2usize, 2usize),
                &device,
            )?)?,
            up_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.45f32, 0.2, -0.1, 0.55],
                (2usize, 2usize),
                &device,
            )?)?,
            down_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.6f32, -0.25, 0.15, 0.5],
                (2usize, 2usize),
                &device,
            )?)?,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 2,
            eps: 1e-6,
            attn_output_gate: false,
        };
        let model = CudaModelWeights {
            token_embedding,
            final_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            lm_head_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.2f32, -0.1, 0.3, 0.4, 0.05, -0.2],
                (2usize, 3usize),
                &device,
            )?)?,
            layers: vec![CudaLayerWeights::FullAttention(layer)],
            rotary_inv_freq: Vec::new(),
            rotary_dim: 0,
            vocab: 4,
            hidden: 2,
        };
        let lora_layers = vec![CudaLoraLayer {
            down_proj: Some(test_lora_pair(&device, 2, 2, 2, 2.0, 0.03)?),
            ..Default::default()
        }];
        let trainable: Vec<CudaTrainTensor> = cuda_lora_pairs(&lora_layers)
            .flat_map(|pair| [pair.a.clone(), pair.b.clone()])
            .collect();
        let before: Vec<Vec<f32>> = trainable
            .iter()
            .map(CudaTrainTensor::to_vec_f32)
            .collect::<Result<_>>()?;
        let mut adamw = allocate_cuda_lora_adamw_state(&lora_layers)?;
        let mut arena = CudaTrainArena::new(&device)?;

        let loss = cuda_full_attention_lora_model_adamw_step_with_arena(
            &model,
            &lora_layers,
            &[2, 0],
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
            &mut arena,
        )?;

        assert!(loss.is_finite() && loss > 0.0);
        assert_eq!(arena.allocation_count(), 6);
        for (param, old) in trainable.iter().zip(before.iter()) {
            assert_ne!(param.to_vec_f32()?, *old);
            assert_eq!(adamw.get(&param.param_id().expect("param id")).expect("state").step, 1);
        }

        let losses = cuda_full_attention_lora_train_token_sequences(
            &model,
            &lora_layers,
            &[vec![2, 0], vec![1, 3]],
            1,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
        )?;
        assert_eq!(losses.len(), 2);
        assert!(losses.iter().all(|loss| loss.is_finite() && *loss > 0.0));
        for param in &trainable {
            assert_eq!(adamw.get(&param.param_id().expect("param id")).expect("state").step, 3);
        }

        let out_dir = std::env::temp_dir().join(format!(
            "kiln-cuda-token-train-adapter-{}",
            std::process::id()
        ));
        let (adapter_dir, saved_losses) = cuda_full_attention_lora_train_token_sequences_to_adapter(
            &model,
            &[vec![2, 0], vec![1, 3]],
            1,
            2,
            4.0,
            0xC0DA_5EED,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
            &out_dir,
        )?;
        assert_eq!(saved_losses.len(), 2);
        assert!(adapter_dir.join("adapter_config.json").exists());
        let saved = candle_core::safetensors::load(
            adapter_dir.join("adapter_model.safetensors"),
            &Device::Cpu,
        )?;
        assert_eq!(saved.len(), 14);
        let _ = std::fs::remove_dir_all(&adapter_dir);
        Ok(())
    }

    #[test]
    fn cuda_init_lora_layers_populates_full_attention_and_gdn_slots() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda LoRA init smoke: {err}");
                return Ok(());
            }
        };

        let full = CudaOwnedFullAttentionLayer {
            input_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            q_weight: CudaTrainTensor::new(Tensor::zeros((2usize, 2usize), DType::F32, &device)?)?,
            k_weight: CudaTrainTensor::new(Tensor::zeros((2usize, 2usize), DType::F32, &device)?)?,
            v_weight: CudaTrainTensor::new(Tensor::zeros((2usize, 2usize), DType::F32, &device)?)?,
            q_norm_weight: CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?,
            k_norm_weight: CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?,
            o_weight: CudaTrainTensor::new(Tensor::zeros((2usize, 2usize), DType::F32, &device)?)?,
            post_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            gate_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize, 4usize),
                DType::F32,
                &device,
            )?)?,
            up_weight: CudaTrainTensor::new(Tensor::zeros((2usize, 4usize), DType::F32, &device)?)?,
            down_weight: CudaTrainTensor::new(Tensor::zeros(
                (4usize, 2usize),
                DType::F32,
                &device,
            )?)?,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 2,
            eps: 1e-6,
            attn_output_gate: false,
        };
        let linear = CudaOwnedLinearAttentionLayer {
            layer_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            in_proj_qkv_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize, 6usize),
                DType::F32,
                &device,
            )?)?,
            in_proj_z_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize, 2usize),
                DType::F32,
                &device,
            )?)?,
            in_proj_a_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize, 2usize),
                DType::F32,
                &device,
            )?)?,
            in_proj_b_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize, 2usize),
                DType::F32,
                &device,
            )?)?,
            conv1d_weight: CudaTrainTensor::new(Tensor::zeros(
                (1usize, 1usize, 4usize),
                DType::F32,
                &device,
            )?)?,
            a_log: CudaTrainTensor::new(Tensor::zeros((1usize,), DType::F32, &device)?)?,
            a_log_gates: CudaTrainTensor::new(Tensor::zeros((1usize,), DType::F32, &device)?)?,
            dt_bias: CudaTrainTensor::new(Tensor::zeros((1usize,), DType::F32, &device)?)?,
            gated_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            out_proj_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize, 2usize),
                DType::F32,
                &device,
            )?)?,
            heads_k: 1,
            heads_v: 1,
            head_dim_k: 2,
            head_dim_v: 2,
            conv_kernel: 4,
            eps: 1e-6,
        };
        let model = CudaModelWeights {
            token_embedding: CudaTrainTensor::new(Tensor::zeros(
                (4usize, 2usize),
                DType::F32,
                &device,
            )?)?,
            final_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            lm_head_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize, 4usize),
                DType::F32,
                &device,
            )?)?,
            layers: vec![
                CudaLayerWeights::FullAttention(full),
                CudaLayerWeights::LinearAttention(linear),
            ],
            rotary_inv_freq: Vec::new(),
            rotary_dim: 0,
            vocab: 4,
            hidden: 2,
        };

        let err = ensure_cuda_native_sft_supported(&model).expect_err("mixed GDN model rejects");
        assert!(
            err.to_string().contains("FullAttention-only models"),
            "unexpected error: {err:#}"
        );

        let lora_layers = cuda_init_lora_layers(&model, 2, 4.0, 0xC0DA_1A7E)?;
        assert_eq!(lora_layers.len(), 2);
        assert!(lora_layers[0].q_proj.is_some());
        assert!(lora_layers[0].down_proj.is_some());
        assert!(lora_layers[1].in_proj_qkv.is_some());
        assert!(lora_layers[1].in_proj_z.is_some());
        assert!(lora_layers[1].gdn_out_proj.is_some());
        assert_eq!(
            lora_layers[1]
                .in_proj_qkv
                .as_ref()
                .expect("in_proj_qkv")
                .b
                .dims(),
            &[6, 2]
        );
        let adamw = allocate_cuda_lora_adamw_state(&lora_layers)?;
        assert_eq!(adamw.len(), 20);

        let CudaLayerWeights::LinearAttention(linear_layer) = &model.layers[1] else {
            panic!("expected LinearAttention layer");
        };
        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.25f32, -0.5, 0.75, 1.0],
            (2usize, 2usize),
            &device,
        )?)?;
        let projections = cuda_gdn_lora_input_projections(&input, linear_layer, &lora_layers[1])?;
        assert_eq!(projections.q.dims(), &[2, 2]);
        assert_eq!(projections.k.dims(), &[2, 2]);
        assert_eq!(projections.v.dims(), &[2, 2]);
        assert_eq!(projections.z.dims(), &[2, 2]);
        let q_loss = cuda_sum_all(&projections.q)?;
        let k_loss = cuda_sum_all(&projections.k)?;
        let v_loss = cuda_sum_all(&projections.v)?;
        let z_loss = cuda_sum_all(&projections.z)?;
        let loss = cuda_add(&cuda_add(&q_loss, &k_loss)?, &cuda_add(&v_loss, &z_loss)?)?;
        let grads = cuda_backward(&loss)?;
        let qkv_pair = lora_layers[1].in_proj_qkv.as_ref().expect("qkv lora");
        let z_pair = lora_layers[1].in_proj_z.as_ref().expect("z lora");
        assert!(grads.get(qkv_pair.b_id).is_some());
        assert!(grads.get(z_pair.b_id).is_some());
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

    #[test]
    fn cuda_lora_adapter_save_includes_gdn_slots() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda GDN LoRA save smoke: {err}");
                return Ok(());
            }
        };

        let in_proj_qkv = test_lora_pair(&device, 2, 6, 2, 2.0, 0.04)?;
        let in_proj_z = test_lora_pair(&device, 2, 2, 2, 2.0, 0.05)?;
        let gdn_out_proj = test_lora_pair(&device, 2, 2, 2, 2.0, 0.06)?;
        let in_proj_qkv_a_expected = in_proj_qkv.a.to_vec_f32()?;
        let in_proj_z_b_expected = in_proj_z.b.to_vec_f32()?;
        let gdn_out_proj_a_expected = gdn_out_proj.a.to_vec_f32()?;
        let layers = vec![CudaLoraLayer {
            in_proj_qkv: Some(in_proj_qkv),
            in_proj_z: Some(in_proj_z),
            gdn_out_proj: Some(gdn_out_proj),
            ..Default::default()
        }];
        let adamw = allocate_cuda_lora_adamw_state(&layers)?;
        assert_eq!(adamw.len(), 6);

        let out_dir = std::env::temp_dir().join(format!(
            "kiln-cuda-gdn-lora-adapter-dir-{}",
            std::process::id()
        ));
        save_cuda_lora_adapter_dir(&layers, 2, 4.0, &out_dir)?;
        let loaded = candle_core::safetensors::load(
            out_dir.join("adapter_model.safetensors"),
            &Device::Cpu,
        )?;
        let in_proj_qkv_a = loaded
            .get("base_model.model.model.layers.0.in_proj_qkv.lora_A.weight")
            .context("missing in_proj_qkv lora_A")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let in_proj_z_b = loaded
            .get("base_model.model.model.layers.0.in_proj_z.lora_B.weight")
            .context("missing in_proj_z lora_B")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gdn_out_proj_a = loaded
            .get("base_model.model.model.layers.0.gdn_out_proj.lora_A.weight")
            .context("missing gdn_out_proj lora_A")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(in_proj_qkv_a, in_proj_qkv_a_expected);
        assert_eq!(in_proj_z_b, in_proj_z_b_expected);
        assert_eq!(gdn_out_proj_a, gdn_out_proj_a_expected);
        assert_eq!(loaded.len(), 6);

        let config_text = std::fs::read_to_string(out_dir.join("adapter_config.json"))?;
        assert!(config_text.contains("in_proj_qkv"));
        assert!(config_text.contains("in_proj_z"));
        assert!(config_text.contains("gdn_out_proj"));
        let _ = std::fs::remove_dir_all(&out_dir);
        Ok(())
    }
}
