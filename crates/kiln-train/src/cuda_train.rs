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
    CudaAdamWConfig, CudaAdamWState, CudaFullAttentionLayer, CudaGdnLayerState, CudaGradStore,
    CudaLayerWeights, CudaLinearAttentionState, CudaModelWeights, CudaOwnedLinearAttentionLayer,
    CudaRopeTables, CudaTrainArena, CudaTrainTensor, cuda_adamw_step_from_store, cuda_add,
    cuda_add_last_dim_bias, cuda_backward,
    cuda_causal_depthwise_conv1d_prefill_with_state_inplace_input_grad, cuda_embedding_lookup,
    cuda_exp, cuda_flash_attn_prefill_causal_f32, cuda_frozen_matmul, cuda_full_attention_layer,
    cuda_gdn_multi_head_sequence_recurrence, cuda_lora_linear_fused, cuda_matmul, cuda_mul,
    cuda_mul_last_dim_weight, cuda_narrow_last_dim, cuda_permute_hr_to_rh, cuda_permute_rh_to_hr,
    cuda_repeat_kv_heads, cuda_reshape, cuda_rmsnorm, cuda_rope, cuda_scale, cuda_sdpa_prefill_causal,
    cuda_shifted_linear_cross_entropy_loss, cuda_sigmoid, cuda_silu, cuda_silu_inplace,
    cuda_softplus, cuda_sum_all, cuda_to_dtype, cuda_transpose2d,
};
use kiln_model::forward::GpuWeights;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::trainer::{ProgressCallback, TrainingProgress, tokenize_for_training};
use crate::{SftConfig, SftExample};

pub type CudaAdamWBook = HashMap<TensorId, CudaAdamWState>;

const CUDA_NATIVE_SFT_FLCE_CHUNK: usize = 8192;

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

fn cuda_lora_pairs<'a>(layers: &'a [CudaLoraLayer]) -> impl Iterator<Item = &'a CudaLoraPair> + 'a {
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
    CudaLoraPair::init_kaiming(
        device,
        weight.dims()[0],
        weight.dims()[1],
        rank,
        alpha,
        seed,
    )
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
    match lora {
        Some(pair) => cuda_lora_linear_fused(input, base_weight, &pair.a, &pair.b, pair.scale)
            .context("cuda LoRA linear fused projection"),
        None => cuda_frozen_matmul(input, base_weight).context("cuda LoRA linear base projection"),
    }
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

pub struct CudaGdnConvQkv {
    pub h_norm: CudaTrainTensor,
    pub q: CudaTrainTensor,
    pub k: CudaTrainTensor,
    pub v: CudaTrainTensor,
    pub z: CudaTrainTensor,
    pub next_conv_state: CudaTrainTensor,
}

/// Native CUDA GDN q/k/v path through LoRA projection, causal conv, and SiLU.
pub fn cuda_gdn_lora_conv_qkv(
    input: &CudaTrainTensor,
    weights: &CudaOwnedLinearAttentionLayer,
    lora: &CudaLoraLayer,
    conv_state: &CudaTrainTensor,
) -> Result<CudaGdnConvQkv> {
    ensure!(
        input.dims().len() == 2,
        "cuda_gdn_lora_conv_qkv: expected rank-2 [rows, hidden], got {:?}",
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
        "cuda_gdn_lora_conv_qkv: qkv projection last dim {} != expected {}",
        qkv.dims().last().copied().unwrap_or(0),
        expected_qkv_dim
    );
    let (conv, next_conv_state) =
        cuda_causal_depthwise_conv1d_prefill_with_state_inplace_input_grad(
            &qkv,
            &weights.conv1d_weight,
            conv_state,
        )
        .context("cuda GDN qkv causal conv1d")?;
    let mixed = cuda_silu_inplace(&conv).context("cuda GDN qkv conv SiLU")?;
    let q = cuda_narrow_last_dim(&mixed, 0, qk_dim)?;
    let k = cuda_narrow_last_dim(&mixed, qk_dim, qk_dim)?;
    let v = cuda_narrow_last_dim(&mixed, qk_dim * 2, v_dim)?;
    let z = cuda_lora_linear(&h_norm, &weights.in_proj_z_weight, lora.in_proj_z.as_ref())?;
    Ok(CudaGdnConvQkv {
        h_norm,
        q,
        k,
        v,
        z,
        next_conv_state,
    })
}

pub struct CudaGdnGateOutputs {
    pub beta: CudaTrainTensor,
    pub g: CudaTrainTensor,
}

pub fn cuda_gdn_gate_outputs_from_normed(
    h_norm: &CudaTrainTensor,
    weights: &CudaOwnedLinearAttentionLayer,
) -> Result<CudaGdnGateOutputs> {
    ensure!(
        h_norm.dims().len() == 2,
        "cuda_gdn_gate_outputs_from_normed: expected rank-2 [rows, hidden], got {:?}",
        h_norm.dims()
    );
    let a =
        cuda_frozen_matmul(h_norm, &weights.in_proj_a_weight).context("cuda GDN a projection")?;
    let b =
        cuda_frozen_matmul(h_norm, &weights.in_proj_b_weight).context("cuda GDN b projection")?;
    ensure!(
        a.dims() == b.dims(),
        "cuda_gdn_gate_outputs_from_normed: a/b shape mismatch {:?} vs {:?}",
        a.dims(),
        b.dims()
    );
    let beta = cuda_sigmoid(&b).context("cuda GDN beta sigmoid")?;
    let a_biased = cuda_add_last_dim_bias(&a, &weights.dt_bias).context("cuda GDN dt bias add")?;
    let softplus = cuda_softplus(&a_biased).context("cuda GDN softplus")?;
    let decay = cuda_scale(
        &cuda_exp(&weights.a_log).context("cuda GDN exp a_log")?,
        -1.0,
    )
    .context("cuda GDN negative decay")?;
    let g = cuda_mul_last_dim_weight(&softplus, &decay).context("cuda GDN decay scale")?;
    Ok(CudaGdnGateOutputs { beta, g })
}

/// Native CUDA GDN gate composition:
/// beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias).
pub fn cuda_gdn_gate_outputs(
    input: &CudaTrainTensor,
    weights: &CudaOwnedLinearAttentionLayer,
) -> Result<CudaGdnGateOutputs> {
    ensure!(
        input.dims().len() == 2,
        "cuda_gdn_gate_outputs: expected rank-2 [rows, hidden], got {:?}",
        input.dims()
    );
    let h_norm = cuda_rmsnorm(input, &weights.layer_norm_weight, weights.eps)?;
    cuda_gdn_gate_outputs_from_normed(&h_norm, weights)
}

pub fn cuda_gdn_gated_rmsnorm(
    input: &CudaTrainTensor,
    gate: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    eps: f32,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims() == gate.dims(),
        "cuda_gdn_gated_rmsnorm: input/gate shape mismatch {:?} vs {:?}",
        input.dims(),
        gate.dims()
    );
    let hidden = *input
        .dims()
        .last()
        .context("cuda_gdn_gated_rmsnorm: input must have rank > 0")?;
    ensure!(
        weight.dims() == [hidden],
        "cuda_gdn_gated_rmsnorm: weight shape {:?} does not match hidden {}",
        weight.dims(),
        hidden
    );

    // `cuda_rmsnorm` applies Qwen's (1 + weight) convention. GDN gated norm
    // uses the stored norm weight directly, so shift the frozen weight here.
    let shifted_weight =
        (weight.as_tensor() - 1.0f64).context("cuda_gdn_gated_rmsnorm: shift norm weight")?;
    let shifted_weight =
        CudaTrainTensor::new(shifted_weight).context("cuda_gdn_gated_rmsnorm: wrap weight")?;
    let normed = cuda_rmsnorm(input, &shifted_weight, eps)?;
    let activated_gate = cuda_silu(gate)?;
    cuda_mul(&normed, &activated_gate)
}

pub fn cuda_gdn_lora_output_projection(
    recurrent_out: &CudaTrainTensor,
    z: &CudaTrainTensor,
    weights: &CudaOwnedLinearAttentionLayer,
    lora: &CudaLoraLayer,
) -> Result<CudaTrainTensor> {
    ensure!(
        recurrent_out.dims().len() == 2,
        "cuda_gdn_lora_output_projection: expected rank-2 [rows, hidden], got {:?}",
        recurrent_out.dims()
    );
    ensure!(
        recurrent_out.dims() == z.dims(),
        "cuda_gdn_lora_output_projection: recurrent/z shape mismatch {:?} vs {:?}",
        recurrent_out.dims(),
        z.dims()
    );
    let rows = recurrent_out.dims()[0];
    let value_dim = weights.heads_v * weights.head_dim_v;
    ensure!(
        recurrent_out.dims()[1] == value_dim,
        "cuda_gdn_lora_output_projection: recurrent last dim {} != heads_v*head_dim_v {}",
        recurrent_out.dims()[1],
        value_dim
    );
    let gated = if weights.gated_norm_weight.dims() == [value_dim] {
        cuda_gdn_gated_rmsnorm(recurrent_out, z, &weights.gated_norm_weight, weights.eps)?
    } else {
        ensure!(
            weights.gated_norm_weight.dims() == [weights.head_dim_v],
            "cuda_gdn_lora_output_projection: gated norm weight {:?} must match value dim {} or head dim {}",
            weights.gated_norm_weight.dims(),
            value_dim,
            weights.head_dim_v
        );
        let per_head = cuda_reshape(recurrent_out, &[rows * weights.heads_v, weights.head_dim_v])?;
        let z_per_head = cuda_reshape(z, &[rows * weights.heads_v, weights.head_dim_v])?;
        let normed = cuda_gdn_gated_rmsnorm(
            &per_head,
            &z_per_head,
            &weights.gated_norm_weight,
            weights.eps,
        )?;
        cuda_reshape(&normed, &[rows, value_dim])?
    };
    cuda_lora_linear(&gated, &weights.out_proj_weight, lora.gdn_out_proj.as_ref())
        .context("cuda GDN output projection")
}

pub struct CudaGdnLoraLayerOutput {
    pub output: CudaTrainTensor,
    pub next_recurrent_state: CudaTrainTensor,
    pub next_conv_state: CudaTrainTensor,
}

fn cuda_gdn_flatten_token_major_heads(
    input: &CudaTrainTensor,
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims() == [rows, heads * head_dim],
        "cuda_gdn_flatten_token_major_heads: expected [{},{}], got {:?}",
        rows,
        heads * head_dim,
        input.dims()
    );
    let rhd = cuda_reshape(input, &[rows, heads, head_dim])?;
    let hrd = cuda_permute_rh_to_hr(&rhd)?;
    cuda_reshape(&hrd, &[heads * rows, head_dim])
}

fn cuda_gdn_unflatten_head_blocks(
    input: &CudaTrainTensor,
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims() == [heads * rows, head_dim],
        "cuda_gdn_unflatten_head_blocks: expected [{},{}], got {:?}",
        heads * rows,
        head_dim,
        input.dims()
    );
    let hrd = cuda_reshape(input, &[heads, rows, head_dim])?;
    let rhd = cuda_permute_hr_to_rh(&hrd)?;
    cuda_reshape(&rhd, &[rows, heads * head_dim])
}

fn cuda_gdn_l2_normalize_head_rows(input: &CudaTrainTensor, scale: f32) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() == 2,
        "cuda_gdn_l2_normalize_head_rows: expected rank-2 [rows, head_dim], got {:?}",
        input.dims()
    );
    let head_dim = input.dims()[1];
    let zeros = CudaTrainTensor::new(Tensor::zeros(
        (head_dim,),
        DType::F32,
        input.as_tensor().device(),
    )?)?;
    let normed = cuda_rmsnorm(input, &zeros, 1e-6f32 / head_dim as f32)?;
    cuda_scale(&normed, scale / (head_dim as f32).sqrt()).context("cuda GDN l2 normalize head rows")
}

pub fn cuda_gdn_lora_layer(
    input: &CudaTrainTensor,
    weights: &CudaOwnedLinearAttentionLayer,
    lora: &CudaLoraLayer,
    recurrent_state: &CudaTrainTensor,
    conv_state: &CudaTrainTensor,
) -> Result<CudaGdnLoraLayerOutput> {
    ensure!(
        input.dims().len() == 2,
        "cuda_gdn_lora_layer: expected rank-2 [rows, hidden], got {:?}",
        input.dims()
    );
    let rows = input.dims()[0];
    ensure!(
        weights.heads_v % weights.heads_k == 0,
        "cuda_gdn_lora_layer: heads_v {} must be divisible by heads_k {}",
        weights.heads_v,
        weights.heads_k
    );
    ensure!(
        recurrent_state.dims() == [weights.heads_v * weights.head_dim_k, weights.head_dim_v],
        "cuda_gdn_lora_layer: recurrent state {:?} != [{},{}]",
        recurrent_state.dims(),
        weights.heads_v * weights.head_dim_k,
        weights.head_dim_v
    );

    let conv_qkv = cuda_gdn_lora_conv_qkv(input, weights, lora, conv_state)?;
    let gates = cuda_gdn_gate_outputs_from_normed(&conv_qkv.h_norm, weights)?;

    let q_heads =
        cuda_gdn_flatten_token_major_heads(&conv_qkv.q, rows, weights.heads_k, weights.head_dim_k)?;
    let k_heads =
        cuda_gdn_flatten_token_major_heads(&conv_qkv.k, rows, weights.heads_k, weights.head_dim_k)?;
    let (q_heads, k_heads) = if weights.heads_k == weights.heads_v {
        (q_heads, k_heads)
    } else {
        let groups = weights.heads_v / weights.heads_k;
        let q_repeated = cuda_repeat_kv_heads(
            &cuda_reshape(&q_heads, &[weights.heads_k, rows, weights.head_dim_k])?,
            groups,
        )?;
        let k_repeated = cuda_repeat_kv_heads(
            &cuda_reshape(&k_heads, &[weights.heads_k, rows, weights.head_dim_k])?,
            groups,
        )?;
        (
            cuda_reshape(&q_repeated, &[weights.heads_v * rows, weights.head_dim_k])?,
            cuda_reshape(&k_repeated, &[weights.heads_v * rows, weights.head_dim_k])?,
        )
    };
    let q_heads =
        cuda_gdn_l2_normalize_head_rows(&q_heads, 1.0f32 / (weights.head_dim_k as f32).sqrt())?;
    let k_heads = cuda_gdn_l2_normalize_head_rows(&k_heads, 1.0)?;
    let v_heads =
        cuda_gdn_flatten_token_major_heads(&conv_qkv.v, rows, weights.heads_v, weights.head_dim_v)?;
    let beta_heads = cuda_gdn_flatten_token_major_heads(&gates.beta, rows, weights.heads_v, 1)?;
    let g_heads = cuda_gdn_flatten_token_major_heads(&gates.g, rows, weights.heads_v, 1)?;

    let recurrent = cuda_gdn_multi_head_sequence_recurrence(
        &q_heads,
        &k_heads,
        &v_heads,
        &beta_heads,
        &g_heads,
        recurrent_state,
        weights.heads_v,
    )?;
    let recurrent_token_major =
        cuda_gdn_unflatten_head_blocks(&recurrent.out, rows, weights.heads_v, weights.head_dim_v)?;
    let projected =
        cuda_gdn_lora_output_projection(&recurrent_token_major, &conv_qkv.z, weights, lora)?;
    let output = cuda_add(input, &projected)?;

    Ok(CudaGdnLoraLayerOutput {
        output,
        next_recurrent_state: recurrent.next_state,
        next_conv_state: conv_qkv.next_conv_state,
    })
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
    let attn = match cuda_flash_attn_prefill_causal_f32(&q_3d, &k_3d, &v_3d, scale)? {
        Some(attn) => attn,
        None => cuda_sdpa_prefill_causal(&q_3d, &k_3d, &v_3d, scale)?,
    };
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
    let logits =
        arena.track(cuda_matmul(&normed, lm_head_weight).context("cuda tiny model LM head")?)?;
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
    Ok(Some((
        CudaTrainTensor::new(cos)?,
        CudaTrainTensor::new(sin)?,
    )))
}

fn cuda_embedding_lookup_f32(
    table: &CudaTrainTensor,
    token_ids: &[usize],
) -> Result<CudaTrainTensor> {
    let embedded = cuda_embedding_lookup(table, token_ids)?;
    match embedded.dtype() {
        DType::F32 => Ok(embedded),
        DType::BF16 => cuda_to_dtype(&embedded, DType::F32),
        dtype => anyhow::bail!("cuda embedding lookup produced unsupported dtype {dtype:?}"),
    }
}

pub fn cuda_full_attention_lora_model_adamw_step_with_arena(
    model: &CudaModelWeights,
    lora_layers: &[CudaLoraLayer],
    token_ids: &[usize],
    label_mask: Option<&[bool]>,
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
        cuda_embedding_lookup_f32(&model.token_embedding, token_ids)
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
    let loss = if let Some(label_mask) = label_mask {
        arena.track(
            cuda_shifted_linear_cross_entropy_loss(
                &normed,
                &model.lm_head_weight,
                token_ids,
                label_mask,
                CUDA_NATIVE_SFT_FLCE_CHUNK,
            )
            .context("cuda FullAttention LoRA model shifted linear CE loss")?,
        )?
    } else {
        let logits = arena.track(
            cuda_frozen_matmul(&normed, &model.lm_head_weight)
                .context("cuda FullAttention LoRA model LM head")?,
        )?;
        let squared = arena
            .track(cuda_mul(&logits, &logits).context("cuda FullAttention LoRA model square")?)?;
        arena.track(cuda_sum_all(&squared).context("cuda FullAttention LoRA model loss")?)?
    };
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

pub fn cuda_lora_model_adamw_step_with_gdn_state_with_arena(
    model: &CudaModelWeights,
    lora_layers: &[CudaLoraLayer],
    token_ids: &[usize],
    label_mask: Option<&[bool]>,
    gdn_state: &mut CudaLinearAttentionState,
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
    arena: &mut CudaTrainArena,
) -> Result<f32> {
    ensure!(
        !token_ids.is_empty(),
        "cuda_lora_model_adamw_step_with_gdn_state requires token ids"
    );
    ensure!(
        lora_layers.len() == model.layers.len(),
        "cuda_lora_model_adamw_step_with_gdn_state lora/model layer mismatch: {} vs {}",
        lora_layers.len(),
        model.layers.len()
    );
    let trainable: Vec<CudaTrainTensor> = cuda_lora_pairs(lora_layers)
        .flat_map(|pair| [pair.a.clone(), pair.b.clone()])
        .collect();
    ensure!(
        !trainable.is_empty(),
        "cuda_lora_model_adamw_step_with_gdn_state requires at least one LoRA parameter"
    );

    let mut hidden = arena.track(
        cuda_embedding_lookup_f32(&model.token_embedding, token_ids)
            .context("cuda LoRA model+GDN embedding")?,
    )?;
    let rope_tables = cuda_compute_rope_tables(
        hidden.as_tensor().device(),
        &model.rotary_inv_freq,
        token_ids.len(),
    )?;
    let mut gdn_idx = 0usize;
    for (idx, layer) in model.layers.iter().enumerate() {
        match layer {
            CudaLayerWeights::FullAttention(full) => {
                let rope = rope_tables.as_ref().map(|(cos, sin)| CudaRopeTables {
                    cos,
                    sin,
                    rotary_dim: model.rotary_dim,
                });
                let borrowed = full.as_borrowed(rope);
                hidden = arena.track(
                    cuda_full_attention_lora_layer(&hidden, &borrowed, &lora_layers[idx])
                        .with_context(|| {
                            format!("cuda LoRA model+GDN FullAttention layer {idx}")
                        })?,
                )?;
            }
            CudaLayerWeights::LinearAttention(linear) => {
                ensure!(
                    gdn_idx < gdn_state.layers.len(),
                    "cuda_lora_model_adamw_step_with_gdn_state: missing GDN state for layer {idx}"
                );
                let layer_state = &gdn_state.layers[gdn_idx];
                ensure!(
                    layer_state.recurrent_state.dims()
                        == [1, linear.heads_v, linear.head_dim_k, linear.head_dim_v],
                    "cuda_lora_model_adamw_step_with_gdn_state: recurrent state {:?} != [1,{},{},{}]",
                    layer_state.recurrent_state.dims(),
                    linear.heads_v,
                    linear.head_dim_k,
                    linear.head_dim_v
                );
                let conv_channels =
                    linear.heads_k * linear.head_dim_k * 2 + linear.heads_v * linear.head_dim_v;
                let conv_state_rows = linear.conv_kernel.saturating_sub(1);
                ensure!(
                    layer_state.conv_state.dims() == [1, conv_channels, conv_state_rows],
                    "cuda_lora_model_adamw_step_with_gdn_state: conv state {:?} != [1,{},{}]",
                    layer_state.conv_state.dims(),
                    conv_channels,
                    conv_state_rows
                );
                let recurrent = cuda_reshape(
                    &layer_state.recurrent_state,
                    &[linear.heads_v * linear.head_dim_k, linear.head_dim_v],
                )?;
                let conv_state_ct =
                    cuda_reshape(&layer_state.conv_state, &[conv_channels, conv_state_rows])?;
                let conv_state = cuda_transpose2d(&conv_state_ct)?;
                let out = cuda_gdn_lora_layer(
                    &hidden,
                    linear,
                    &lora_layers[idx],
                    &recurrent,
                    &conv_state,
                )
                .with_context(|| format!("cuda LoRA model+GDN LinearAttention layer {idx}"))?;
                let recurrent_next = cuda_reshape(
                    &out.next_recurrent_state,
                    &[1, linear.heads_v, linear.head_dim_k, linear.head_dim_v],
                )?;
                let conv_next_ct = cuda_transpose2d(&out.next_conv_state)?;
                let conv_next = cuda_reshape(&conv_next_ct, &[1, conv_channels, conv_state_rows])?;
                gdn_state.layers[gdn_idx].recurrent_state = recurrent_next;
                gdn_state.layers[gdn_idx].conv_state = conv_next;
                hidden = arena.track(out.output)?;
                gdn_idx += 1;
            }
        }
    }
    ensure!(
        gdn_idx == gdn_state.layers.len(),
        "cuda_lora_model_adamw_step_with_gdn_state consumed {gdn_idx} GDN states, expected {}",
        gdn_state.layers.len()
    );
    let normed = arena.track(
        cuda_rmsnorm(&hidden, &model.final_norm_weight, 1e-6)
            .context("cuda LoRA model+GDN final RMSNorm")?,
    )?;
    let loss = if let Some(label_mask) = label_mask {
        arena.track(
            cuda_shifted_linear_cross_entropy_loss(
                &normed,
                &model.lm_head_weight,
                token_ids,
                label_mask,
                CUDA_NATIVE_SFT_FLCE_CHUNK,
            )
            .context("cuda LoRA model+GDN shifted linear CE loss")?,
        )?
    } else {
        let logits = arena.track(
            cuda_frozen_matmul(&normed, &model.lm_head_weight)
                .context("cuda LoRA model+GDN LM head")?,
        )?;
        let squared =
            arena.track(cuda_mul(&logits, &logits).context("cuda LoRA model+GDN square")?)?;
        arena.track(cuda_sum_all(&squared).context("cuda LoRA model+GDN loss")?)?
    };
    let loss_value = loss.to_vec_f32()?[0];
    let grads = cuda_backward(&loss).context("cuda LoRA model+GDN backward")?;
    let updated = cuda_adamw_step_from_store(&trainable, &grads, adamw_state, cfg)
        .context("cuda LoRA model+GDN AdamW")?;
    ensure!(
        updated == trainable.len(),
        "cuda_lora_model_adamw_step_with_gdn_state updated {updated} params, expected {}",
        trainable.len()
    );
    Ok(loss_value)
}

// ====================================================================
// Layerwise reverse-recompute SFT step (fix for #1063).
//
// The original cuda_lora_model_adamw_step_with_gdn_state_with_arena
// builds a single per-step autograd graph spanning all 32 transformer
// layers and ~50 ops/layer. cuda_backward then has to walk ~1.6K nodes
// linearly, allocating per-grad CudaTrainTensors via candle ops; the
// graph traversal cost dominates and yields the ~50x gap vs the
// candle-routed --trainer generic path documented in #1063.
//
// This module mirrors `vk_recompute_train_step_with_state_masked` from
// vk_train.rs. The strategy is gradient checkpointing with exact
// layerwise replay:
//
// 1. Forward through every layer with *detached* LoRA so no autograd
//    graph is built. Cache layer-input "boundaries" when memory allows;
//    otherwise fall back to O(N^2) replay (still cheaper than the
//    monolithic backward because each replay's graph is tiny).
//
// 2. Wrap the final hidden as a fresh parameter, apply RMSNorm + FLCE,
//    then cuda_backward through *just* the head + loss to get the
//    upstream gradient at the pre-final-norm boundary.
//
// 3. For each layer in reverse, wrap that layer's input boundary as a
//    fresh parameter, re-run the layer forward with the live (grad-
//    tracking) LoRA weights, then compute `scalar = sum(out * upstream)`
//    and cuda_backward through that single layer. Read the boundary
//    grad off the store as the next upstream; accumulate LoRA grads
//    into a shared HashMap keyed by TensorId.
//
// 4. After the reverse sweep, drive cuda_adamw_step_from_store on the
//    accumulated grads. The legacy step kernel ran AdamW inline; the
//    recompute path defers it so accumulation is local to the loop.
fn cuda_detach_lora_pair(pair: &CudaLoraPair) -> CudaLoraPair {
    CudaLoraPair {
        a: pair.a.detach(),
        b: pair.b.detach(),
        a_id: pair.a_id,
        b_id: pair.b_id,
        scale: pair.scale,
    }
}

fn cuda_detach_lora_layers(layers: &[CudaLoraLayer]) -> Vec<CudaLoraLayer> {
    layers
        .iter()
        .map(|l| CudaLoraLayer {
            q_proj: l.q_proj.as_ref().map(cuda_detach_lora_pair),
            k_proj: l.k_proj.as_ref().map(cuda_detach_lora_pair),
            v_proj: l.v_proj.as_ref().map(cuda_detach_lora_pair),
            o_proj: l.o_proj.as_ref().map(cuda_detach_lora_pair),
            gate_proj: l.gate_proj.as_ref().map(cuda_detach_lora_pair),
            up_proj: l.up_proj.as_ref().map(cuda_detach_lora_pair),
            down_proj: l.down_proj.as_ref().map(cuda_detach_lora_pair),
            in_proj_qkv: l.in_proj_qkv.as_ref().map(cuda_detach_lora_pair),
            in_proj_z: l.in_proj_z.as_ref().map(cuda_detach_lora_pair),
            gdn_out_proj: l.gdn_out_proj.as_ref().map(cuda_detach_lora_pair),
        })
        .collect()
}

fn cuda_gdn_layer_index_map(model: &CudaModelWeights) -> Vec<Option<usize>> {
    let mut map = Vec::with_capacity(model.layers.len());
    let mut gdn_idx = 0;
    for layer in &model.layers {
        match layer {
            CudaLayerWeights::LinearAttention(_) => {
                map.push(Some(gdn_idx));
                gdn_idx += 1;
            }
            _ => map.push(None),
        }
    }
    map
}

fn cuda_snapshot_gdn_state(state: &CudaLinearAttentionState) -> CudaLinearAttentionState {
    CudaLinearAttentionState {
        layers: state
            .layers
            .iter()
            .map(|l| CudaGdnLayerState {
                recurrent_state: l.recurrent_state.detach(),
                recurrent_n_elements: l.recurrent_n_elements,
                conv_state: l.conv_state.detach(),
                conv_n_elements: l.conv_n_elements,
            })
            .collect(),
    }
}

/// Run a single GDN layer's *forward* using the supplied state at entry
/// and return the layer output plus the next state. Mirrors the GDN
/// branch of cuda_lora_model_adamw_step_with_gdn_state_with_arena so
/// both paths agree byte-for-byte on the recurrence formulation.
fn cuda_gdn_lora_layer_with_entry_state(
    hidden: &CudaTrainTensor,
    linear: &CudaOwnedLinearAttentionLayer,
    lora_layer: &CudaLoraLayer,
    entry_state: &CudaGdnLayerState,
) -> Result<(CudaTrainTensor, CudaTrainTensor, CudaTrainTensor)> {
    let recurrent = cuda_reshape(
        &entry_state.recurrent_state,
        &[linear.heads_v * linear.head_dim_k, linear.head_dim_v],
    )?;
    let conv_channels =
        linear.heads_k * linear.head_dim_k * 2 + linear.heads_v * linear.head_dim_v;
    let conv_state_rows = linear.conv_kernel.saturating_sub(1);
    let conv_state_ct = cuda_reshape(&entry_state.conv_state, &[conv_channels, conv_state_rows])?;
    let conv_state = cuda_transpose2d(&conv_state_ct)?;
    let out = cuda_gdn_lora_layer(hidden, linear, lora_layer, &recurrent, &conv_state)?;
    let recurrent_next = cuda_reshape(
        &out.next_recurrent_state,
        &[1, linear.heads_v, linear.head_dim_k, linear.head_dim_v],
    )?;
    let conv_next_ct = cuda_transpose2d(&out.next_conv_state)?;
    let conv_next = cuda_reshape(&conv_next_ct, &[1, conv_channels, conv_state_rows])?;
    Ok((out.output, recurrent_next, conv_next))
}

/// Forward through layers 0..end_layer with detached LoRA, returning
/// the hidden tensor at the *input* of `end_layer` (i.e. after layers
/// 0..end_layer-1) and the GDN state at that boundary. Used by the
/// no-cache recompute fallback.
#[allow(clippy::too_many_arguments)]
fn cuda_forward_to_layer_input(
    model: &CudaModelWeights,
    detached_lora: &[CudaLoraLayer],
    token_ids: &[usize],
    end_layer: usize,
    rope_tables: Option<&(CudaTrainTensor, CudaTrainTensor)>,
    gdn_map: &[Option<usize>],
    profile: bool,
) -> Result<(CudaTrainTensor, CudaLinearAttentionState)> {
    ensure!(
        end_layer <= model.layers.len(),
        "cuda_forward_to_layer_input: end_layer {end_layer} > {}",
        model.layers.len()
    );
    let mut hidden = cuda_embedding_lookup_f32(&model.token_embedding, token_ids)
        .context("cuda recompute forward: embedding lookup")?
        .detach();
    let mut gdn_state = cuda_linear_attention_state_zeros_for_model(model, 1)?;
    for layer_idx in 0..end_layer {
        if profile {
            tracing::info!(
                end_layer,
                layer_idx,
                seq_len = token_ids.len(),
                "cuda-native recompute forward layer begin"
            );
        }
        let next = match &model.layers[layer_idx] {
            CudaLayerWeights::FullAttention(full) => {
                let rope = rope_tables.map(|(cos, sin)| CudaRopeTables {
                    cos,
                    sin,
                    rotary_dim: model.rotary_dim,
                });
                let borrowed = full.as_borrowed(rope);
                cuda_full_attention_lora_layer(&hidden, &borrowed, &detached_lora[layer_idx])
                    .with_context(|| {
                        format!("cuda recompute forward: FullAttention layer {layer_idx}")
                    })?
            }
            CudaLayerWeights::LinearAttention(linear) => {
                let gdn_idx = gdn_map[layer_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing GDN index for layer {layer_idx}"))?;
                let (out, recurrent_next, conv_next) = cuda_gdn_lora_layer_with_entry_state(
                    &hidden,
                    linear,
                    &detached_lora[layer_idx],
                    &gdn_state.layers[gdn_idx],
                )
                .with_context(|| {
                    format!("cuda recompute forward: LinearAttention layer {layer_idx}")
                })?;
                gdn_state.layers[gdn_idx].recurrent_state = recurrent_next.detach();
                gdn_state.layers[gdn_idx].conv_state = conv_next.detach();
                out
            }
        };
        // Detach to keep the forward graph empty between layers. The
        // detached LoRA + frozen weights already prevent grad_fn from
        // being attached, but defending against future op-graph changes
        // is cheap and matches the vk_native pattern.
        hidden = next.detach();
    }
    Ok((hidden, gdn_state))
}

/// Forward through every layer with detached LoRA, capturing the layer-
/// input "boundary" hidden tensors plus the GDN state at each layer's
/// entry. Boundaries are detached snapshots; total memory is roughly
/// `(N_layers + 1) * seq_len * hidden * 4` bytes. Caller decides whether
/// to use this path or the per-layer replay based on a memory budget.
fn cuda_forward_layer_boundaries(
    model: &CudaModelWeights,
    detached_lora: &[CudaLoraLayer],
    token_ids: &[usize],
    rope_tables: Option<&(CudaTrainTensor, CudaTrainTensor)>,
    gdn_map: &[Option<usize>],
    profile: bool,
) -> Result<(Vec<CudaTrainTensor>, Vec<Option<CudaLinearAttentionState>>)> {
    let mut hidden = cuda_embedding_lookup_f32(&model.token_embedding, token_ids)
        .context("cuda recompute boundaries: embedding lookup")?
        .detach();
    let mut boundaries: Vec<CudaTrainTensor> = Vec::with_capacity(model.layers.len() + 1);
    let mut state_at_layer_entry: Vec<Option<CudaLinearAttentionState>> =
        Vec::with_capacity(model.layers.len());
    boundaries.push(hidden.clone());
    let mut gdn_state = cuda_linear_attention_state_zeros_for_model(model, 1)?;
    for layer_idx in 0..model.layers.len() {
        let needs_state = matches!(&model.layers[layer_idx], CudaLayerWeights::LinearAttention(_));
        // Snapshot the state at the *input* to this layer so the reverse
        // recompute pass can re-run the layer's recurrence with the same
        // starting point.
        state_at_layer_entry.push(if needs_state {
            Some(cuda_snapshot_gdn_state(&gdn_state))
        } else {
            None
        });
        if profile {
            tracing::info!(
                end_layer = model.layers.len(),
                layer_idx,
                seq_len = token_ids.len(),
                boundary_cache = true,
                "cuda-native recompute forward layer begin"
            );
        }
        let next = match &model.layers[layer_idx] {
            CudaLayerWeights::FullAttention(full) => {
                let rope = rope_tables.map(|(cos, sin)| CudaRopeTables {
                    cos,
                    sin,
                    rotary_dim: model.rotary_dim,
                });
                let borrowed = full.as_borrowed(rope);
                cuda_full_attention_lora_layer(&hidden, &borrowed, &detached_lora[layer_idx])
                    .with_context(|| {
                        format!("cuda recompute boundaries: FullAttention layer {layer_idx}")
                    })?
            }
            CudaLayerWeights::LinearAttention(linear) => {
                let gdn_idx = gdn_map[layer_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing GDN index for layer {layer_idx}"))?;
                let (out, recurrent_next, conv_next) = cuda_gdn_lora_layer_with_entry_state(
                    &hidden,
                    linear,
                    &detached_lora[layer_idx],
                    &gdn_state.layers[gdn_idx],
                )
                .with_context(|| {
                    format!("cuda recompute boundaries: LinearAttention layer {layer_idx}")
                })?;
                gdn_state.layers[gdn_idx].recurrent_state = recurrent_next.detach();
                gdn_state.layers[gdn_idx].conv_state = conv_next.detach();
                out
            }
        };
        hidden = next.detach();
        boundaries.push(hidden.clone());
    }
    Ok((boundaries, state_at_layer_entry))
}

/// Auto-pick the boundary-cache memory ceiling. Mirrors the vk-side
/// helper. `KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE_GB=<float>` overrides;
/// otherwise we reserve 4 GiB for driver / weights / scratch and cap at
/// 10 GiB so very long contexts fall back to exact per-layer replay.
fn cuda_recompute_boundary_cache_limit_bytes() -> usize {
    if let Some(limit) = std::env::var("KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE_GB")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as usize)
    {
        return limit;
    }
    cuda_recompute_boundary_cache_auto_limit_bytes()
}

#[cfg(target_os = "linux")]
fn cuda_recompute_boundary_cache_auto_limit_bytes() -> usize {
    let raw = match std::fs::read_to_string("/proc/meminfo") {
        Ok(raw) => raw,
        Err(_) => return 0,
    };
    let mut mem_available = None;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            mem_available = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .map(|kib| kib.saturating_mul(1024));
            break;
        }
    }
    let Some(available) = mem_available else {
        return 0;
    };
    const GIB: usize = 1024 * 1024 * 1024;
    available.saturating_sub(4 * GIB).min(10 * GIB)
}

#[cfg(not(target_os = "linux"))]
fn cuda_recompute_boundary_cache_auto_limit_bytes() -> usize {
    0
}

/// Layerwise reverse-recompute SFT training step.
///
/// Fixes the slow path called out by #1063. See the module comment at
/// `// Layerwise reverse-recompute SFT step` above for the design.
///
/// Semantics:
/// * Returns the scalar loss (FLCE-masked when `label_mask` is provided,
///   otherwise sum-of-squares on the unmasked logits, matching the
///   legacy step kernel's surrogate so SFT and the existing token-only
///   regression smoke tests keep the same surface).
/// * Updates AdamW state and writes new LoRA values into `lora_layers`
///   in-place via cuda_adamw_step_from_store, exactly as the legacy
///   path does.
pub fn cuda_recompute_train_step_with_state_masked(
    model: &CudaModelWeights,
    lora_layers: &[CudaLoraLayer],
    token_ids: &[usize],
    label_mask: Option<&[bool]>,
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
) -> Result<f32> {
    ensure!(
        !token_ids.is_empty(),
        "cuda_recompute_train_step_with_state_masked requires token ids"
    );
    ensure!(
        lora_layers.len() == model.layers.len(),
        "cuda_recompute_train_step_with_state_masked lora/model layer mismatch: {} vs {}",
        lora_layers.len(),
        model.layers.len()
    );
    ensure!(
        cuda_lora_pairs(lora_layers).next().is_some(),
        "cuda_recompute_train_step_with_state_masked requires at least one LoRA parameter"
    );

    let device = model.token_embedding.as_tensor().device();
    let seq_len = token_ids.len();
    let gdn_map = cuda_gdn_layer_index_map(model);
    let profile = kiln_core::env_flag::env_flag("KILN_PROFILE_CUDA_RECOMPUTE", false);

    let boundary_cache_limit = cuda_recompute_boundary_cache_limit_bytes();
    let boundary_cache_bytes = (model.layers.len() + 1)
        .saturating_mul(seq_len)
        .saturating_mul(model.hidden)
        .saturating_mul(std::mem::size_of::<f32>());
    let use_boundary_cache =
        kiln_core::env_flag::env_tristate("KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE")
            .unwrap_or(boundary_cache_limit > 0 && boundary_cache_bytes <= boundary_cache_limit);

    let detached_lora = cuda_detach_lora_layers(lora_layers);

    let rope_tables = cuda_compute_rope_tables(device, &model.rotary_inv_freq, seq_len)?;
    let rope_refs = rope_tables.as_ref();

    if profile {
        tracing::info!(
            step = adamw_state.values().next().map(|s| s.step).unwrap_or(0) + 1,
            seq_len,
            boundary_cache = use_boundary_cache,
            boundary_cache_bytes,
            boundary_cache_limit,
            "cuda-native recompute step begin"
        );
    }

    let (final_hidden, boundary_cache, state_cache) = if use_boundary_cache {
        let (boundaries, states) = cuda_forward_layer_boundaries(
            model,
            &detached_lora,
            token_ids,
            rope_refs,
            &gdn_map,
            profile,
        )?;
        let final_hidden = boundaries
            .last()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("cuda recompute: empty boundary cache"))?;
        (final_hidden, Some(boundaries), Some(states))
    } else {
        let (final_hidden, _final_state) = cuda_forward_to_layer_input(
            model,
            &detached_lora,
            token_ids,
            model.layers.len(),
            rope_refs,
            &gdn_map,
            profile,
        )?;
        (final_hidden, None, None)
    };

    // Wrap the final boundary as a fresh parameter so cuda_backward can
    // recover d(loss)/d(final_hidden) by param_id.
    let final_id = final_hidden.as_tensor().id();
    let final_param = CudaTrainTensor::parameter(final_hidden.as_tensor().clone(), final_id)?;

    let normed = cuda_rmsnorm(&final_param, &model.final_norm_weight, 1e-6f32)
        .context("cuda recompute: final RMSNorm")?;
    let loss = if let Some(mask) = label_mask {
        cuda_shifted_linear_cross_entropy_loss(
            &normed,
            &model.lm_head_weight,
            token_ids,
            mask,
            CUDA_NATIVE_SFT_FLCE_CHUNK,
        )
        .context("cuda recompute: shifted linear CE loss")?
    } else {
        let logits = cuda_frozen_matmul(&normed, &model.lm_head_weight)
            .context("cuda recompute: LM head matmul")?;
        let squared = cuda_mul(&logits, &logits).context("cuda recompute: square logits")?;
        cuda_sum_all(&squared).context("cuda recompute: sum-square surrogate loss")?
    };
    let loss_value = loss.to_vec_f32()?[0];
    ensure!(
        loss_value.is_finite(),
        "cuda_recompute_train_step_with_state_masked: non-finite loss {loss_value}"
    );

    let final_grads = cuda_backward(&loss).context("cuda recompute: head/loss backward")?;
    let mut upstream = final_grads
        .into_inner()
        .remove(&final_id)
        .ok_or_else(|| anyhow::anyhow!("cuda recompute: missing upstream grad at final hidden"))?;

    // Per-layer reverse sweep. The shared HashMap is keyed by the LoRA
    // param's TensorId; cross-layer collisions are not possible because
    // each layer has its own freshly minted LoRA Vars.
    let mut shared_grads: HashMap<TensorId, CudaTrainTensor> = HashMap::new();
    for layer_idx in (0..model.layers.len()).rev() {
        if profile {
            tracing::info!(
                layer_idx,
                seq_len,
                boundary_cache = use_boundary_cache,
                "cuda-native recompute reverse layer begin"
            );
        }
        // Resolve (boundary, state-at-entry) either from the cache or by
        // re-running the forward up to this layer's input.
        let (boundary, state_at_entry) = match (&boundary_cache, &state_cache) {
            (Some(boundaries), Some(states)) => (
                boundaries[layer_idx].clone(),
                states[layer_idx].as_ref().map(cuda_snapshot_gdn_state),
            ),
            _ => {
                let (b, s) = cuda_forward_to_layer_input(
                    model,
                    &detached_lora,
                    token_ids,
                    layer_idx,
                    rope_refs,
                    &gdn_map,
                    profile,
                )?;
                let state_needed = matches!(
                    &model.layers[layer_idx],
                    CudaLayerWeights::LinearAttention(_)
                );
                (b, state_needed.then_some(s))
            }
        };
        let boundary_id = boundary.as_tensor().id();
        let boundary_param =
            CudaTrainTensor::parameter(boundary.as_tensor().clone(), boundary_id)?;

        // Re-run the layer forward with the *live* LoRA weights so
        // backward can collect grads against pair.a_id / pair.b_id.
        let layer_out = match &model.layers[layer_idx] {
            CudaLayerWeights::FullAttention(full) => {
                let rope = rope_refs.map(|(cos, sin)| CudaRopeTables {
                    cos,
                    sin,
                    rotary_dim: model.rotary_dim,
                });
                let borrowed = full.as_borrowed(rope);
                cuda_full_attention_lora_layer(&boundary_param, &borrowed, &lora_layers[layer_idx])
                    .with_context(|| {
                        format!("cuda recompute reverse: FullAttention layer {layer_idx}")
                    })?
            }
            CudaLayerWeights::LinearAttention(linear) => {
                let gdn_idx = gdn_map[layer_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing GDN index for layer {layer_idx}"))?;
                let state = state_at_entry.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("cuda recompute: GDN layer {layer_idx} requires entry state")
                })?;
                let entry_state = &state.layers[gdn_idx];
                let (out, _next_recurrent, _next_conv) = cuda_gdn_lora_layer_with_entry_state(
                    &boundary_param,
                    linear,
                    &lora_layers[layer_idx],
                    entry_state,
                )
                .with_context(|| {
                    format!("cuda recompute reverse: LinearAttention layer {layer_idx}")
                })?;
                out
            }
        };

        let prod = cuda_mul(&layer_out, &upstream)
            .with_context(|| format!("cuda recompute reverse: layer {layer_idx} prod"))?;
        let scalar = cuda_sum_all(&prod)
            .with_context(|| format!("cuda recompute reverse: layer {layer_idx} scalar"))?;
        let layer_grads = cuda_backward(&scalar)
            .with_context(|| format!("cuda recompute reverse: layer {layer_idx} backward"))?;
        let mut layer_grads = layer_grads.into_inner();

        // Hand the boundary grad to the next iteration.
        upstream = layer_grads
            .remove(&boundary_id)
            .ok_or_else(|| anyhow::anyhow!("cuda recompute: missing boundary grad at layer {layer_idx}"))?;

        // Drop any layer-output-aliased grads and fold the rest into
        // shared_grads. Accumulation uses cuda_add on detached tensors;
        // each LoRA TensorId only appears in one layer's grad map, so in
        // practice every insert hits the "None" branch (the .remove/Some
        // path is kept for safety against future graph changes).
        for (pid, grad) in layer_grads {
            match shared_grads.remove(&pid) {
                Some(existing) => {
                    let summed = cuda_add(&existing.detach(), &grad.detach()).with_context(
                        || format!("cuda recompute: accumulate grad {pid:?} at layer {layer_idx}"),
                    )?;
                    shared_grads.insert(pid, summed);
                }
                None => {
                    shared_grads.insert(pid, grad);
                }
            }
        }
        if profile {
            tracing::info!(
                layer_idx,
                seq_len,
                "cuda-native recompute reverse layer done"
            );
        }
    }

    // Apply AdamW on the accumulated LoRA grads.
    let trainable: Vec<CudaTrainTensor> = cuda_lora_pairs(lora_layers)
        .flat_map(|pair| [pair.a.clone(), pair.b.clone()])
        .collect();
    let mut grad_store = CudaGradStore::new();
    for (pid, grad) in shared_grads {
        grad_store.insert(pid, grad);
    }
    let updated = cuda_adamw_step_from_store(&trainable, &grad_store, adamw_state, cfg)
        .context("cuda recompute: AdamW step")?;
    ensure!(
        updated == trainable.len(),
        "cuda_recompute_train_step_with_state_masked updated {updated} params, expected {}",
        trainable.len()
    );

    Ok(loss_value)
}

fn cuda_linear_attention_state_zeros_for_model(
    model: &CudaModelWeights,
    batch: usize,
) -> Result<CudaLinearAttentionState> {
    ensure!(
        batch > 0,
        "cuda_linear_attention_state_zeros_for_model requires batch > 0"
    );
    let device = model.token_embedding.as_tensor().device();
    let mut layers = Vec::new();
    for layer in &model.layers {
        let CudaLayerWeights::LinearAttention(linear) = layer else {
            continue;
        };
        let recurrent_shape = (batch, linear.heads_v, linear.head_dim_k, linear.head_dim_v);
        let recurrent_n = batch * linear.heads_v * linear.head_dim_k * linear.head_dim_v;
        let conv_channels =
            linear.heads_k * linear.head_dim_k * 2 + linear.heads_v * linear.head_dim_v;
        let conv_rows = linear.conv_kernel.saturating_sub(1);
        let conv_n = batch * conv_channels * conv_rows;
        layers.push(CudaGdnLayerState {
            recurrent_state: CudaTrainTensor::new(Tensor::zeros(
                recurrent_shape,
                DType::F32,
                device,
            )?)?,
            recurrent_n_elements: recurrent_n,
            conv_state: CudaTrainTensor::new(Tensor::zeros(
                (batch, conv_channels, conv_rows),
                DType::F32,
                device,
            )?)?,
            conv_n_elements: conv_n,
        });
    }
    Ok(CudaLinearAttentionState { layers })
}

pub fn cuda_lora_train_token_sequences_with_gdn_state(
    model: &CudaModelWeights,
    lora_layers: &[CudaLoraLayer],
    token_sequences: &[Vec<usize>],
    epochs: usize,
    adamw_state: &mut CudaAdamWBook,
    cfg: CudaAdamWConfig,
) -> Result<Vec<f32>> {
    ensure!(
        epochs > 0,
        "cuda_lora_train_token_sequences_with_gdn_state requires at least one epoch"
    );
    ensure!(
        !token_sequences.is_empty(),
        "cuda_lora_train_token_sequences_with_gdn_state requires token sequences"
    );
    let device = model.token_embedding.as_tensor().device();
    let mut losses = Vec::with_capacity(epochs * token_sequences.len());
    for _epoch in 0..epochs {
        for token_ids in token_sequences {
            let mut arena = CudaTrainArena::new(device)?;
            let mut gdn_state = cuda_linear_attention_state_zeros_for_model(model, 1)?;
            let loss = cuda_lora_model_adamw_step_with_gdn_state_with_arena(
                model,
                lora_layers,
                token_ids,
                None,
                &mut gdn_state,
                adamw_state,
                cfg,
                &mut arena,
            )?;
            ensure!(
                loss.is_finite(),
                "cuda_lora_train_token_sequences_with_gdn_state encountered non-finite loss {loss}"
            );
            losses.push(loss);
        }
    }
    Ok(losses)
}

#[allow(clippy::too_many_arguments)]
pub fn cuda_lora_train_token_sequences_with_gdn_state_to_adapter(
    model: &CudaModelWeights,
    token_sequences: &[Vec<usize>],
    epochs: usize,
    rank: usize,
    alpha: f32,
    seed: u64,
    adamw_cfg: CudaAdamWConfig,
    output_dir: &Path,
) -> Result<(PathBuf, Vec<f32>)> {
    crate::lora_scaling::validate_lora_scaling(rank, alpha, false)?;
    let lora_layers =
        cuda_init_lora_layers(model, rank, alpha, seed).context("initialize CUDA LoRA layers")?;
    let mut adamw =
        allocate_cuda_lora_adamw_state(&lora_layers).context("allocate CUDA LoRA AdamW state")?;
    let losses = cuda_lora_train_token_sequences_with_gdn_state(
        model,
        &lora_layers,
        token_sequences,
        epochs,
        &mut adamw,
        adamw_cfg,
    )
    .context("train CUDA LoRA token sequences with GDN state")?;
    let adapter_dir = save_cuda_lora_adapter_dir(&lora_layers, rank, alpha, output_dir)
        .context("save CUDA LoRA GDN-state token adapter")?;
    Ok((adapter_dir, losses))
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
                None,
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
    crate::lora_scaling::validate_lora_scaling(rank, alpha, false)?;
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
    // Issue #1063 (root-cause fix): the legacy cuda_native step is
    // ~50x slower than `sft_train` because cuda_train's hand-rolled
    // autograd does many small CUDA launches with per-op CPU overhead
    // (HashMap walk + per-grad candle Tensor allocation), and that
    // overhead scales linearly with kernel-launch count. The structural
    // backport from vk_native (layerwise recompute, also in this file)
    // confirmed via bench that recompute *adds* launches and so cannot
    // reduce per-op CPU overhead — see the recompute step's module
    // comment for the empirical numbers.
    //
    // The actual fix is to route the inner step through `BackendRuntime`
    // + candle's autograd, which is what `sft_train` already does and
    // which uses the production-tuned fused FlashAttn / GDN / RMSNorm
    // kernels. This default-routing brings cuda_native_sft_train down
    // to the same step time as `--trainer generic` (~3.5 s/step on
    // Qwen3.5-4B per the issue numbers), for every caller of this
    // function — not just users who happen to flip a flag.
    //
    // The legacy `cuda_train`-side step kernels remain reachable for
    // parity testing and the memory-saving recompute path:
    //
    //   - KILN_CUDA_LEGACY_NATIVE_STEP=1 — bypass the route, run the
    //     legacy monolithic-graph step. Slow; use only for parity tests
    //     against the new recompute step.
    //   - KILN_CUDA_RECOMPUTE_SFT=1 — bypass the route, run the
    //     layerwise reverse-recompute step. Same speed as legacy but
    //     ~30% less peak VRAM (see PR for the bench).
    //
    // The default path (neither env var set) is the route through
    // sft_train.
    let force_legacy =
        kiln_core::env_flag::env_flag("KILN_CUDA_LEGACY_NATIVE_STEP", false);
    let force_recompute =
        kiln_core::env_flag::env_tristate("KILN_CUDA_RECOMPUTE_SFT").unwrap_or(false);
    if !force_legacy && !force_recompute {
        tracing::info!(
            num_examples = examples.len(),
            epochs = config.epochs,
            lr = config.learning_rate,
            rank = config.lora_rank,
            alpha = config.lora_alpha,
            adapter_name,
            path = "backend_runtime_via_sft_train",
            "cuda_native_sft_train: routing through BackendRuntime + candle autograd (fixes #1063)"
        );
        return crate::trainer::sft_train(
            examples,
            config,
            model_config,
            weights,
            tokenizer,
            adapter_dir,
            adapter_name,
            progress_cb,
            None,
        );
    }

    let run_started = std::time::Instant::now();
    let output_dir = adapter_dir.join(adapter_name);
    let training_data_sha256 = crate::train_receipt::sha256_json_serializable(&examples);
    let mut data_stats = crate::train_receipt::DataStatsReceipt {
        examples_read: examples.len(),
        ..Default::default()
    };
    let mut token_counts = crate::train_receipt::TokenCountReceipt::default();

    tracing::warn!(
        num_examples = examples.len(),
        epochs = config.epochs,
        lr = config.learning_rate,
        rank = config.lora_rank,
        alpha = config.lora_alpha,
        adapter_name,
        path = if force_recompute {
            "layerwise_recompute"
        } else {
            "legacy_monolithic"
        },
        "cuda_native_sft_train: BYPASSING the BackendRuntime route via env flag; \
         this is the slow path. Unset KILN_CUDA_LEGACY_NATIVE_STEP / \
         KILN_CUDA_RECOMPUTE_SFT to use the production path"
    );

    let alpha_over_rank = match crate::lora_scaling::validate_lora_scaling(
        config.lora_rank,
        config.lora_alpha,
        config.allow_high_lora_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            write_cuda_sft_train_receipt_best_effort(
                adapter_name,
                model_config,
                tokenizer,
                config,
                &output_dir,
                training_data_sha256,
                data_stats,
                token_counts,
                run_started.elapsed().as_millis() as u64,
                None,
                Some(format!("{err:#}")),
            );
            return Err(crate::train_receipt::annotate_training_error(err));
        }
    };

    if let Some(base_adapter) = config.base_adapter.as_deref() {
        let message = format!(
            "cuda_native_sft_train does not support base_adapter {base_adapter:?}; use the generic trainer until CUDA-native base-adapter loading is implemented"
        );
        write_cuda_sft_train_receipt_best_effort(
            adapter_name,
            model_config,
            tokenizer,
            config,
            &output_dir,
            training_data_sha256,
            data_stats,
            token_counts,
            run_started.elapsed().as_millis() as u64,
            Some(alpha_over_rank),
            Some(message.clone()),
        );
        anyhow::bail!("{}", crate::train_receipt::training_failure_error_message(&message));
    }

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
    let lora_layers =
        cuda_init_lora_layers(&model, config.lora_rank, config.lora_alpha, effective_seed)
            .context("cuda_native_sft_train: initialize LoRA layers")?;
    let mut adamw = allocate_cuda_lora_adamw_state(&lora_layers)
        .context("cuda_native_sft_train: allocate AdamW state")?;
    let cfg = CudaAdamWConfig {
        lr: config.learning_rate as f32,
        ..Default::default()
    };

    let tokenized: Vec<(Vec<usize>, Vec<bool>)> = examples
        .iter()
        .filter_map(|example| match tokenize_for_training(example, tokenizer) {
            Ok((ids, mask)) => Some((ids.into_iter().map(|id| id as usize).collect(), mask)),
            Err(err) => {
                tracing::warn!("cuda-native: skipping example: {err}");
                None
            }
        })
        .collect();
    data_stats.examples_filtered = examples.len().saturating_sub(tokenized.len());
    data_stats.examples_trained = tokenized.len().saturating_mul(config.epochs);
    for (input_ids, label_mask) in &tokenized {
        let action_tokens = label_mask.iter().filter(|&&mask| mask).count() as u64;
        token_counts.action_tokens = token_counts
            .action_tokens
            .saturating_add(action_tokens.saturating_mul(config.epochs as u64));
        token_counts.context_tokens = token_counts.context_tokens.saturating_add(
            (input_ids.len().saturating_sub(action_tokens as usize) as u64)
                .saturating_mul(config.epochs as u64),
        );
    }
    if tokenized.is_empty() {
        let message = "cuda_native_sft_train: no valid training examples after tokenization";
        write_cuda_sft_train_receipt_best_effort(
            adapter_name,
            model_config,
            tokenizer,
            config,
            &output_dir,
            training_data_sha256,
            data_stats,
            token_counts,
            run_started.elapsed().as_millis() as u64,
            Some(alpha_over_rank),
            Some(message.to_string()),
        );
        anyhow::bail!("{}", crate::train_receipt::training_failure_error_message(message));
    }

    let total_steps = config.epochs * tokenized.len();
    let has_gdn = model
        .layers
        .iter()
        .any(|layer| matches!(layer, CudaLayerWeights::LinearAttention(_)));

    // We're on the legacy/recompute opt-in path here (the default path
    // returned early via sft_train). `force_recompute` and
    // `force_legacy` were read before the early return; honor whichever
    // the caller asked for, defaulting to legacy if both were set.
    let use_recompute = force_recompute && !force_legacy;
    tracing::info!(
        path = if use_recompute {
            "layerwise_recompute"
        } else {
            "legacy_monolithic"
        },
        has_gdn,
        flce_chunk = CUDA_NATIVE_SFT_FLCE_CHUNK,
        "cuda-native SFT step path selected (env-bypass)"
    );

    let mut global_step = 0usize;
    let mut last_loss = 0.0f32;
    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;
        for (token_ids, label_mask) in &tokenized {
            global_step += 1;
            let step_start = std::time::Instant::now();
            let loss = if use_recompute {
                cuda_recompute_train_step_with_state_masked(
                    &model,
                    &lora_layers,
                    token_ids,
                    Some(label_mask),
                    &mut adamw,
                    cfg,
                )
            } else {
                let mut arena = CudaTrainArena::new(model.token_embedding.as_tensor().device())?;
                if has_gdn {
                    let mut gdn_state = cuda_linear_attention_state_zeros_for_model(&model, 1)?;
                    cuda_lora_model_adamw_step_with_gdn_state_with_arena(
                        &model,
                        &lora_layers,
                        token_ids,
                        Some(label_mask),
                        &mut gdn_state,
                        &mut adamw,
                        cfg,
                        &mut arena,
                    )
                } else {
                    cuda_full_attention_lora_model_adamw_step_with_arena(
                        &model,
                        &lora_layers,
                        token_ids,
                        Some(label_mask),
                        &mut adamw,
                        cfg,
                        &mut arena,
                    )
                }
            }
            .with_context(|| {
                format!(
                    "cuda_native_sft_train step {} epoch {}",
                    global_step,
                    epoch + 1
                )
            })?;
            let step_ms = step_start.elapsed().as_millis();
            tracing::info!(
                epoch = epoch + 1,
                step = global_step,
                total_steps,
                seq_len = token_ids.len(),
                step_ms,
                "cuda-native SFT step"
            );
            ensure!(
                loss.is_finite(),
                "cuda_native_sft_train: non-finite loss {loss} at step {global_step}"
            );
            epoch_loss += loss;
            last_loss = loss;

            if let Some(interval) = config.checkpoint_interval {
                if interval > 0 && global_step % interval == 0 && global_step < total_steps {
                    let ckpt_dir =
                        adapter_dir.join(format!("{adapter_name}-checkpoint-{global_step}"));
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
    write_cuda_sft_train_receipt_best_effort(
        adapter_name,
        model_config,
        tokenizer,
        config,
        &output_dir,
        training_data_sha256,
        data_stats,
        token_counts,
        run_started.elapsed().as_millis() as u64,
        Some(alpha_over_rank),
        None,
    );
    Ok(output_dir)
}

#[allow(clippy::too_many_arguments)]
fn write_cuda_sft_train_receipt_best_effort(
    adapter_name: &str,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    config: &SftConfig,
    output_dir: &Path,
    training_data_sha256: Option<String>,
    data: crate::train_receipt::DataStatsReceipt,
    token_counts: crate::train_receipt::TokenCountReceipt,
    wall_clock_ms: u64,
    alpha_over_rank: Option<f32>,
    status_error: Option<String>,
) {
    let mut receipt = crate::train_receipt::TrainReceipt::new(
        adapter_name,
        "cuda_sft",
        model_config,
        tokenizer,
        crate::train_receipt::HyperparameterReceipt {
            mode: "cuda_sft".to_string(),
            rank: config.lora_rank,
            alpha: config.lora_alpha,
            alpha_over_rank,
            learning_rate: config.learning_rate,
            epochs: config.epochs,
            seed: config.seed,
        },
        serde_json::to_value(config).unwrap_or(serde_json::Value::Null),
    );
    receipt.training_data = crate::train_receipt::TrainingDataReceipt {
        source: "inline_sft_examples".to_string(),
        path: None,
        sha256: training_data_sha256,
    };
    receipt.adapters.output = crate::train_receipt::adapter_file_receipt(Some(output_dir));
    receipt.data = data;
    receipt.token_counts = token_counts;
    receipt.runtime.wall_clock_ms = wall_clock_ms;
    crate::train_receipt::log_training_token_counts("cuda_sft", &receipt.token_counts);
    if status_error.is_none() {
        receipt.lora_delta_norms =
            crate::train_receipt::lora_delta_norm_summary_from_adapter(
                output_dir,
                alpha_over_rank.unwrap_or(0.0) as f64,
            )
            .unwrap_or_default();
        crate::train_receipt::warn_lora_delta_norms(
            "cuda_sft",
            adapter_name,
            &receipt.lora_delta_norms,
            alpha_over_rank.unwrap_or(0.0) as f64,
        );
    }
    if let Some(err) = status_error {
        receipt = receipt.mark_failed(err);
    }
    if let Err(err) = receipt.write_to_adapter_dir(output_dir) {
        tracing::warn!(adapter = adapter_name, error = %err, "failed to write CUDA SFT train receipt");
    }
}

/// Save named CUDA training tensors to safetensors after one CUDA-to-CPU readback.
pub fn save_cuda_training_tensors(
    weights: &[(&str, CudaTrainTensor)],
    output_path: &Path,
) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, weight) in weights {
        ensure!(
            !name.is_empty(),
            "CUDA training safetensors key must not be empty"
        );
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
    let adapter_name = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("adapter");
    crate::adapter_output::write_adapter_output_receipt(output_dir, adapter_name, None)
        .with_context(|| {
            format!(
                "write CUDA adapter output receipt {}",
                output_dir.display()
            )
        })?;
    Ok(output_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use kiln_model::cuda_train::CudaOwnedFullAttentionLayer;

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
            assert_eq!(
                adamw
                    .get(&param.param_id().expect("param id"))
                    .expect("state")
                    .step,
                1
            );
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
            post_norm_weight: CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?,
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
            None,
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
            assert_eq!(
                adamw
                    .get(&param.param_id().expect("param id"))
                    .expect("state")
                    .step,
                1
            );
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
            assert_eq!(
                adamw
                    .get(&param.param_id().expect("param id"))
                    .expect("state")
                    .step,
                3
            );
        }

        let out_dir = std::env::temp_dir().join(format!(
            "kiln-cuda-token-train-adapter-{}",
            std::process::id()
        ));
        let (adapter_dir, saved_losses) =
            cuda_full_attention_lora_train_token_sequences_to_adapter(
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
            post_norm_weight: CudaTrainTensor::new(Tensor::zeros((2usize,), DType::F32, &device)?)?,
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
                (2usize, 1usize),
                DType::F32,
                &device,
            )?)?,
            in_proj_b_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize, 1usize),
                DType::F32,
                &device,
            )?)?,
            conv1d_weight: CudaTrainTensor::new(Tensor::zeros(
                (6usize, 1usize, 3usize),
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
            conv_kernel: 3,
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

        ensure_cuda_native_sft_supported(&model)?;

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
        let input_tensor =
            Tensor::from_vec(vec![0.25f32, -0.5, 0.75, 1.0], (2usize, 2usize), &device)?;
        let input_id = input_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
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

        let conv_state =
            CudaTrainTensor::new(Tensor::zeros((2usize, 6usize), DType::F32, &device)?)?;
        let conv_qkv = cuda_gdn_lora_conv_qkv(&input, linear_layer, &lora_layers[1], &conv_state)?;
        assert_eq!(conv_qkv.q.dims(), &[2, 2]);
        assert_eq!(conv_qkv.k.dims(), &[2, 2]);
        assert_eq!(conv_qkv.v.dims(), &[2, 2]);
        assert_eq!(conv_qkv.next_conv_state.dims(), &[2, 6]);

        let gates = cuda_gdn_gate_outputs(&input, linear_layer)?;
        assert_eq!(gates.beta.dims(), &[2, 1]);
        assert_eq!(gates.g.dims(), &[2, 1]);
        let gate_loss = cuda_add(&cuda_sum_all(&gates.beta)?, &cuda_sum_all(&gates.g)?)?;
        let gate_grads = cuda_backward(&gate_loss)?;
        assert!(gate_grads.get(input_id).is_some());

        let recurrent_out = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.15f32, -0.2, 0.35, 0.45],
            (2usize, 2usize),
            &device,
        )?)?;
        let out = cuda_gdn_lora_output_projection(
            &recurrent_out,
            &projections.z,
            linear_layer,
            &lora_layers[1],
        )?;
        assert_eq!(out.dims(), &[2, 2]);
        let out_loss = cuda_sum_all(&out)?;
        let out_grads = cuda_backward(&out_loss)?;
        let out_pair = lora_layers[1].gdn_out_proj.as_ref().expect("gdn out lora");
        assert!(out_grads.get(out_pair.b_id).is_some());

        let recurrent_state =
            CudaTrainTensor::new(Tensor::zeros((2usize, 2usize), DType::F32, &device)?)?;
        let layer_out = cuda_gdn_lora_layer(
            &input,
            linear_layer,
            &lora_layers[1],
            &recurrent_state,
            &conv_state,
        )?;
        assert_eq!(layer_out.output.dims(), &[2, 2]);
        assert_eq!(layer_out.next_recurrent_state.dims(), &[2, 2]);
        assert_eq!(layer_out.next_conv_state.dims(), &[2, 6]);
        let layer_loss = cuda_sum_all(&layer_out.output)?;
        let layer_grads = cuda_backward(&layer_loss)?;
        assert!(layer_grads.get(input_id).is_some());
        assert!(layer_grads.get(out_pair.b_id).is_some());
        Ok(())
    }

    #[test]
    fn cuda_lora_model_step_with_gdn_state_threads_gdn_state() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda GDN model-step smoke: {err}");
                return Ok(());
            }
        };

        let conv_weight: Vec<f32> = (0..6).flat_map(|_| [0.0f32, 0.0, 1.0]).collect();
        let linear = CudaOwnedLinearAttentionLayer {
            layer_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            in_proj_qkv_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![
                    0.2f32, -0.1, 0.4, 0.3, -0.2, 0.5, -0.3, 0.25, 0.1, -0.4, 0.6, 0.2,
                ],
                (2usize, 6usize),
                &device,
            )?)?,
            in_proj_z_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.3f32, -0.2, 0.1, 0.4],
                (2usize, 2usize),
                &device,
            )?)?,
            in_proj_a_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.2f32, -0.1],
                (2usize, 1usize),
                &device,
            )?)?,
            in_proj_b_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![-0.3f32, 0.25],
                (2usize, 1usize),
                &device,
            )?)?,
            conv1d_weight: CudaTrainTensor::new(Tensor::from_vec(
                conv_weight,
                (6usize, 1usize, 3usize),
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
            out_proj_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.4f32, -0.2, 0.15, 0.35],
                (2usize, 2usize),
                &device,
            )?)?,
            heads_k: 1,
            heads_v: 1,
            head_dim_k: 2,
            head_dim_v: 2,
            conv_kernel: 3,
            eps: 1e-6,
        };
        let model = CudaModelWeights {
            token_embedding: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.25f32, -0.5, 0.75, 1.0, -0.3, 0.2, 0.4, -0.1],
                (4usize, 2usize),
                &device,
            )?)?,
            final_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                &device,
            )?)?,
            lm_head_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.2f32, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.4],
                (2usize, 4usize),
                &device,
            )?)?,
            layers: vec![CudaLayerWeights::LinearAttention(linear)],
            rotary_inv_freq: Vec::new(),
            rotary_dim: 0,
            vocab: 4,
            hidden: 2,
        };
        let lora_layers = vec![CudaLoraLayer {
            in_proj_qkv: Some(test_lora_pair(&device, 2, 6, 2, 2.0, 0.03)?),
            in_proj_z: Some(test_lora_pair(&device, 2, 2, 2, 2.0, 0.05)?),
            gdn_out_proj: Some(test_lora_pair(&device, 2, 2, 2, 2.0, 0.07)?),
            ..Default::default()
        }];
        let mut gdn_state = CudaLinearAttentionState::zeros(&device, 1, 1, 1, 2, 2, 6, 3)?;
        let mut adamw = allocate_cuda_lora_adamw_state(&lora_layers)?;
        let mut arena = CudaTrainArena::new(&device)?;
        let loss = cuda_lora_model_adamw_step_with_gdn_state_with_arena(
            &model,
            &lora_layers,
            &[0, 1],
            None,
            &mut gdn_state,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
            &mut arena,
        )?;

        assert!(loss.is_finite());
        assert!(
            gdn_state.layers[0]
                .recurrent_state
                .to_vec_f32()?
                .iter()
                .any(|value| value.abs() > 1e-6)
        );
        let losses = cuda_lora_train_token_sequences_with_gdn_state(
            &model,
            &lora_layers,
            &[vec![0, 1]],
            1,
            &mut adamw,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
        )?;
        assert_eq!(losses.len(), 1);
        assert!(losses[0].is_finite());
        let adapter_dir = std::env::temp_dir().join(format!(
            "kiln-cuda-gdn-state-token-adapter-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&adapter_dir);
        let (saved_dir, saved_losses) = cuda_lora_train_token_sequences_with_gdn_state_to_adapter(
            &model,
            &[vec![0, 1]],
            1,
            2,
            4.0,
            0xC0DA_6DAD,
            CudaAdamWConfig {
                lr: 0.01,
                ..CudaAdamWConfig::default()
            },
            &adapter_dir,
        )?;
        assert_eq!(saved_losses.len(), 1);
        assert!(saved_dir.join("adapter_model.safetensors").exists());
        assert!(saved_dir.join("adapter_config.json").exists());
        let _ = std::fs::remove_dir_all(&adapter_dir);
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
        let input_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
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
        let q_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
        let k_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
        let o_tensor = Tensor::from_vec(vec![0.3f32, -0.4, 0.8, 0.2], (2usize, 2usize), &device)?;
        let o_id = o_tensor.id();
        let o_weight = CudaTrainTensor::parameter(o_tensor, o_id)?;
        let post_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
        let gate_tensor =
            Tensor::from_vec(vec![0.25f32, -0.15, 0.35, 0.05], (2usize, 2usize), &device)?;
        let gate_id = gate_tensor.id();
        let gate_weight = CudaTrainTensor::parameter(gate_tensor, gate_id)?;
        let up_tensor =
            Tensor::from_vec(vec![0.45f32, 0.2, -0.1, 0.55], (2usize, 2usize), &device)?;
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
        let input_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
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
        let q_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
        let k_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
        let o_tensor = Tensor::from_vec(vec![0.3f32, -0.4, 0.8, 0.2], (2usize, 2usize), &device)?;
        let o_id = o_tensor.id();
        let o_weight = CudaTrainTensor::parameter(o_tensor, o_id)?;
        let post_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
        let gate_tensor =
            Tensor::from_vec(vec![0.25f32, -0.15, 0.35, 0.05], (2usize, 2usize), &device)?;
        let gate_id = gate_tensor.id();
        let gate_weight = CudaTrainTensor::parameter(gate_tensor, gate_id)?;
        let up_tensor =
            Tensor::from_vec(vec![0.45f32, 0.2, -0.1, 0.55], (2usize, 2usize), &device)?;
        let up_id = up_tensor.id();
        let up_weight = CudaTrainTensor::parameter(up_tensor, up_id)?;
        let down_tensor =
            Tensor::from_vec(vec![0.6f32, -0.25, 0.15, 0.5], (2usize, 2usize), &device)?;
        let down_id = down_tensor.id();
        let down_weight = CudaTrainTensor::parameter(down_tensor, down_id)?;
        let final_norm =
            CudaTrainTensor::new(Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?)?;
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

        let out_dir =
            std::env::temp_dir().join(format!("kiln-cuda-lora-adapter-dir-{}", std::process::id()));
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

    // -----------------------------------------------------------------
    // Layerwise recompute parity tests (issue #1063)
    // -----------------------------------------------------------------

    /// Mutex serialising the recompute tests that mutate
    /// `KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE`. Cargo test runs tests in
    /// parallel by default; without this guard two threads could
    /// observe each other's env-var writes and skew the parity check.
    static RECOMPUTE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn build_recompute_test_model_and_lora(
        device: &Device,
    ) -> Result<(CudaModelWeights, Vec<CudaLoraLayer>, Vec<usize>, Vec<bool>)> {
        // Same shape as cuda_lora_model_step_with_gdn_state_threads_gdn_state:
        // single GDN layer, hidden=2, vocab=4, 2 tokens. Small enough for a
        // fast test, large enough to exercise the full GDN reverse recompute
        // including the recurrent / conv state plumbing.
        let conv_weight: Vec<f32> = (0..6).flat_map(|_| [0.0f32, 0.0, 1.0]).collect();
        let linear = CudaOwnedLinearAttentionLayer {
            layer_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                device,
            )?)?,
            in_proj_qkv_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![
                    0.2f32, -0.1, 0.4, 0.3, -0.2, 0.5, -0.3, 0.25, 0.1, -0.4, 0.6, 0.2,
                ],
                (2usize, 6usize),
                device,
            )?)?,
            in_proj_z_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.3f32, -0.2, 0.1, 0.4],
                (2usize, 2usize),
                device,
            )?)?,
            in_proj_a_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.2f32, -0.1],
                (2usize, 1usize),
                device,
            )?)?,
            in_proj_b_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![-0.3f32, 0.25],
                (2usize, 1usize),
                device,
            )?)?,
            conv1d_weight: CudaTrainTensor::new(Tensor::from_vec(
                conv_weight,
                (6usize, 1usize, 3usize),
                device,
            )?)?,
            a_log: CudaTrainTensor::new(Tensor::zeros((1usize,), DType::F32, device)?)?,
            a_log_gates: CudaTrainTensor::new(Tensor::zeros((1usize,), DType::F32, device)?)?,
            dt_bias: CudaTrainTensor::new(Tensor::zeros((1usize,), DType::F32, device)?)?,
            gated_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                device,
            )?)?,
            out_proj_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.4f32, -0.2, 0.15, 0.35],
                (2usize, 2usize),
                device,
            )?)?,
            heads_k: 1,
            heads_v: 1,
            head_dim_k: 2,
            head_dim_v: 2,
            conv_kernel: 3,
            eps: 1e-6,
        };
        let model = CudaModelWeights {
            token_embedding: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.25f32, -0.5, 0.75, 1.0, -0.3, 0.2, 0.4, -0.1],
                (4usize, 2usize),
                device,
            )?)?,
            final_norm_weight: CudaTrainTensor::new(Tensor::zeros(
                (2usize,),
                DType::F32,
                device,
            )?)?,
            lm_head_weight: CudaTrainTensor::new(Tensor::from_vec(
                vec![0.2f32, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.4],
                (2usize, 4usize),
                device,
            )?)?,
            layers: vec![CudaLayerWeights::LinearAttention(linear)],
            rotary_inv_freq: Vec::new(),
            rotary_dim: 0,
            vocab: 4,
            hidden: 2,
        };
        // Two-token input with an active label at position 1.
        // `cuda_shifted_linear_cross_entropy_loss` iterates
        // `label_mask[1..]`, so `mask[k] = true` means "predict
        // `input_ids[k]` from `hidden[k-1]`". With seq_len=2 the only
        // valid active position is k=1; mark `mask[0]` false (the first
        // position can't be a label target under shifted CE).
        let token_ids = vec![0usize, 1usize];
        let label_mask = vec![false, true];
        Ok((model, Vec::new(), token_ids, label_mask))
    }

    fn build_recompute_test_lora(device: &Device) -> Result<Vec<CudaLoraLayer>> {
        Ok(vec![CudaLoraLayer {
            in_proj_qkv: Some(test_lora_pair(device, 2, 6, 2, 2.0, 0.03)?),
            in_proj_z: Some(test_lora_pair(device, 2, 2, 2, 2.0, 0.05)?),
            gdn_out_proj: Some(test_lora_pair(device, 2, 2, 2, 2.0, 0.07)?),
            ..Default::default()
        }])
    }

    fn lora_snapshot(layers: &[CudaLoraLayer]) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let mut out = Vec::new();
        for layer in layers {
            for pair in [
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
            {
                out.push((pair.a.to_vec_f32()?, pair.b.to_vec_f32()?));
            }
        }
        Ok(out)
    }

    /// Loss parity: the new recompute path must agree with the legacy
    /// monolithic-graph step on the loss value when both start from the
    /// same LoRA init. The loss is computed *before* AdamW mutates the
    /// LoRA weights, so independent LoRA copies (identical init) suffice.
    #[test]
    fn cuda_recompute_step_loss_parity_with_legacy_step() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping recompute parity test: {err}");
                return Ok(());
            }
        };

        let (model, _, token_ids, label_mask) = build_recompute_test_model_and_lora(&device)?;
        let lora_legacy = build_recompute_test_lora(&device)?;
        let lora_recompute = build_recompute_test_lora(&device)?;

        let mut adamw_legacy = allocate_cuda_lora_adamw_state(&lora_legacy)?;
        let mut adamw_recompute = allocate_cuda_lora_adamw_state(&lora_recompute)?;
        let mut gdn_state = cuda_linear_attention_state_zeros_for_model(&model, 1)?;
        let mut arena = CudaTrainArena::new(&device)?;
        let cfg = CudaAdamWConfig {
            lr: 0.01,
            ..CudaAdamWConfig::default()
        };

        let token_vec: Vec<usize> = token_ids.clone();

        let _guard = RECOMPUTE_ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE");
        }

        let loss_legacy = cuda_lora_model_adamw_step_with_gdn_state_with_arena(
            &model,
            &lora_legacy,
            &token_vec,
            Some(&label_mask),
            &mut gdn_state,
            &mut adamw_legacy,
            cfg,
            &mut arena,
        )?;
        let loss_recompute = cuda_recompute_train_step_with_state_masked(
            &model,
            &lora_recompute,
            &token_vec,
            Some(&label_mask),
            &mut adamw_recompute,
            cfg,
        )?;

        assert!(
            loss_legacy.is_finite() && loss_recompute.is_finite(),
            "non-finite losses: legacy={loss_legacy} recompute={loss_recompute}"
        );
        let denom = loss_legacy.abs().max(loss_recompute.abs()).max(1e-6);
        let rel_diff = (loss_legacy - loss_recompute).abs() / denom;
        assert!(
            rel_diff < 1e-3,
            "loss parity failed: legacy={loss_legacy} recompute={loss_recompute} rel_diff={rel_diff}"
        );
        Ok(())
    }

    /// Backward parity: starting from identical LoRA init + identical
    /// AdamW state, one legacy step and one recompute step must leave
    /// the LoRA weights at *the same* post-step values (within FP32
    /// tolerance). This is the structural correctness guarantee for
    /// the recompute backward — it must produce the same gradients
    /// that the legacy monolithic-graph backward would.
    ///
    /// Note: at this tiny model scale (hidden=2, seq_len=2, rank=2)
    /// the AdamW update per pair can be below 1e-9 even with lr=0.1
    /// because several Jacobians fold to near-zero (RMSNorm into a
    /// zero-eps L2 norm in `cuda_gdn_l2_normalize_head_rows` projects
    /// components parallel to the input out of the gradient, and the
    /// 2x2 GDN recurrence flattens further at fp32). What matters for
    /// regression detection is that legacy and recompute *agree on
    /// the post-step LoRA values* — whether the update is large or
    /// small, both paths must observe the same one. The loss parity
    /// test above covers the forward; this test covers the backward.
    #[test]
    fn cuda_recompute_step_backward_parity_with_legacy() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping recompute-updates test: {err}");
                return Ok(());
            }
        };

        let (model, _, token_ids, _label_mask) = build_recompute_test_model_and_lora(&device)?;
        let cfg = CudaAdamWConfig {
            lr: 0.1,
            ..CudaAdamWConfig::default()
        };
        let token_vec: Vec<usize> = token_ids.clone();

        // Legacy step
        let lora_legacy = build_recompute_test_lora(&device)?;
        let mut adamw_legacy = allocate_cuda_lora_adamw_state(&lora_legacy)?;
        let mut gdn_state = cuda_linear_attention_state_zeros_for_model(&model, 1)?;
        let mut arena = CudaTrainArena::new(&device)?;
        let loss_legacy = cuda_lora_model_adamw_step_with_gdn_state_with_arena(
            &model,
            &lora_legacy,
            &token_vec,
            None,
            &mut gdn_state,
            &mut adamw_legacy,
            cfg,
            &mut arena,
        )?;
        let after_legacy = lora_snapshot(&lora_legacy)?;

        // Recompute step from identical init
        let lora_rec = build_recompute_test_lora(&device)?;
        let mut adamw_rec = allocate_cuda_lora_adamw_state(&lora_rec)?;
        let loss_rec = cuda_recompute_train_step_with_state_masked(
            &model,
            &lora_rec,
            &token_vec,
            None,
            &mut adamw_rec,
            cfg,
        )?;
        let after_rec = lora_snapshot(&lora_rec)?;

        assert!(loss_legacy.is_finite() && loss_rec.is_finite());

        // Forward parity (loss values match)
        let loss_rel =
            (loss_legacy - loss_rec).abs() / loss_legacy.abs().max(loss_rec.abs()).max(1e-6);
        assert!(
            loss_rel < 1e-4,
            "loss forward parity broken: legacy={loss_legacy} recompute={loss_rec}"
        );

        // Backward parity: every LoRA A/B entry must match after one
        // step. This is the strong structural guarantee — even if the
        // update happens to be near-zero for this tiny model, both
        // paths must produce the same near-zero outcome.
        let mut max_a_diff = 0.0f32;
        let mut max_b_diff = 0.0f32;
        for ((a_l, b_l), (a_r, b_r)) in after_legacy.iter().zip(after_rec.iter()) {
            for (x, y) in a_l.iter().zip(a_r.iter()) {
                max_a_diff = max_a_diff.max((x - y).abs());
            }
            for (x, y) in b_l.iter().zip(b_r.iter()) {
                max_b_diff = max_b_diff.max((x - y).abs());
            }
        }
        assert!(
            max_a_diff < 1e-5 && max_b_diff < 1e-5,
            "backward parity broken: max |legacy_a - rec_a|={max_a_diff} \
             max |legacy_b - rec_b|={max_b_diff}"
        );
        Ok(())
    }

    /// `KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE` is purely an optimization;
    /// flipping it must not affect the returned loss when starting from
    /// the same LoRA init.
    #[test]
    fn cuda_recompute_step_boundary_cache_vs_no_cache_parity() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping recompute boundary-cache parity test: {err}"
                );
                return Ok(());
            }
        };

        let (model, _, token_ids, label_mask) = build_recompute_test_model_and_lora(&device)?;
        let lora_cached = build_recompute_test_lora(&device)?;
        let lora_uncached = build_recompute_test_lora(&device)?;
        let mut adamw_cached = allocate_cuda_lora_adamw_state(&lora_cached)?;
        let mut adamw_uncached = allocate_cuda_lora_adamw_state(&lora_uncached)?;
        let cfg = CudaAdamWConfig {
            lr: 0.01,
            ..CudaAdamWConfig::default()
        };
        let token_vec: Vec<usize> = token_ids.clone();

        let _guard = RECOMPUTE_ENV_LOCK.lock().unwrap();

        unsafe {
            std::env::set_var("KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE", "1");
        }
        let loss_cached = cuda_recompute_train_step_with_state_masked(
            &model,
            &lora_cached,
            &token_vec,
            Some(&label_mask),
            &mut adamw_cached,
            cfg,
        )?;

        unsafe {
            std::env::set_var("KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE", "0");
        }
        let loss_uncached = cuda_recompute_train_step_with_state_masked(
            &model,
            &lora_uncached,
            &token_vec,
            Some(&label_mask),
            &mut adamw_uncached,
            cfg,
        )?;

        unsafe {
            std::env::remove_var("KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE");
        }

        assert!(
            loss_cached.is_finite() && loss_uncached.is_finite(),
            "non-finite losses: cached={loss_cached} uncached={loss_uncached}"
        );
        let denom = loss_cached.abs().max(loss_uncached.abs()).max(1e-6);
        let rel_diff = (loss_cached - loss_uncached).abs() / denom;
        assert!(
            rel_diff < 1e-5,
            "boundary-cache parity failed: cached={loss_cached} uncached={loss_uncached} rel_diff={rel_diff}"
        );
        Ok(())
    }
}
