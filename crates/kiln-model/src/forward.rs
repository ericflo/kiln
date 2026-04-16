//! Candle-based forward pass layers for Qwen3.5-4B.
//!
//! Implements the foundational compute primitives: embedding lookup, RMSNorm,
//! RoPE (rotary position embeddings), and SwiGLU FFN. These operate on candle
//! `Tensor` objects and are composed into the full transformer forward pass.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::kv_cache::KvCache;
use crate::lora_loader::{linear_with_lora, LoraLayerWeights, LoraWeights};
use crate::paged_kv_cache::PagedKvCache;
use crate::weights::{ModelWeights, TensorDType, WeightTensor};

use kiln_core::block::BlockTable;

/// CUDA-compatible sigmoid: `1 / (1 + exp(-x))`.
///
/// `candle_nn::ops::sigmoid` lacks a CUDA kernel, so we implement it using
/// basic tensor operations that all have CUDA support.
fn cuda_sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let one_plus = (exp_neg_x + 1.0)?;
    let result = one_plus.recip()?;
    Ok(result)
}

/// CUDA-compatible SiLU (Swish): `x * sigmoid(x)`.
fn cuda_silu(x: &Tensor) -> Result<Tensor> {
    let sig = cuda_sigmoid(x)?;
    Ok((x * sig)?)
}

/// CUDA-compatible softmax on last dimension.
///
/// `candle_nn::ops::softmax_last_dim` lacks a CUDA kernel, so we implement it
/// manually: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.
fn cuda_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max_val = x.max_keepdim(candle_core::D::Minus1)?;
    let shifted = x.broadcast_sub(&max_val)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_keepdim(candle_core::D::Minus1)?;
    Ok(exp_shifted.broadcast_div(&sum_exp)?)
}

/// Compute attention using FlashAttention-2 CUDA kernels.
///
/// Takes Q, K, V in `[batch, seq_len, num_heads, head_dim]` layout (pre-transpose).
/// K/V may have fewer heads than Q (GQA); they are expanded to match Q's head count
/// before calling the flash kernel, which requires uniform head counts.
///
/// Returns `[batch, seq_len, num_heads * head_dim]` (already reshaped for output projection).
#[cfg(feature = "cuda")]
fn flash_attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let causal = true;

    // GQA: expand K/V heads to match Q head count for flash_attn
    let (k, v) = if num_heads != num_kv_heads {
        let gqa_ratio = num_heads / num_kv_heads;
        let (batch, kv_len, _kv_heads, hd) = k.dims4()?;
        // [batch, kv_len, num_kv_heads, head_dim] -> [batch, kv_len, num_heads, head_dim]
        let k = k
            .unsqueeze(3)?
            .expand(&[batch, kv_len, num_kv_heads, gqa_ratio, hd])?
            .contiguous()?
            .reshape((batch, kv_len, num_heads, hd))?;
        let v = v
            .unsqueeze(3)?
            .expand(&[batch, kv_len, num_kv_heads, gqa_ratio, hd])?
            .contiguous()?
            .reshape((batch, kv_len, num_heads, hd))?;
        (k, v)
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // flash_attn expects [batch, seq_len, num_heads, head_dim]
    let attn_output = kiln_flash_attn::flash_attn(q, &k, &v, softmax_scale, causal)
        .context("flash_attn kernel failed")?;

    // Reshape to [batch, seq_len, hidden]
    let (batch, seq_len, _heads, _hd) = attn_output.dims4()?;
    let attn_output = attn_output
        .contiguous()?
        .reshape((batch, seq_len, num_heads * head_dim))?;
    Ok(attn_output)
}

/// GPU-ready tensors organized by layer, converted from raw `ModelWeights` bytes.
pub struct GpuWeights {
    /// Token embedding table: [vocab_size, hidden_size]
    pub embed_tokens: Tensor,
    /// Per-layer weights
    pub layers: Vec<GpuLayerWeights>,
    /// Final RMSNorm weight: [hidden_size]
    pub final_norm: Tensor,
}

/// One transformer layer's tensors on device.
pub struct GpuLayerWeights {
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub attention: GpuAttentionWeights,
    pub mlp: GpuFfnWeights,
}

/// Attention weights on device.
pub enum GpuAttentionWeights {
    Full(GpuFullAttentionWeights),
    Linear(GpuLinearAttentionWeights),
}

pub struct GpuFullAttentionWeights {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub struct GpuLinearAttentionWeights {
    pub in_proj_qkv: Tensor,
    pub in_proj_z: Tensor,
    pub out_proj: Tensor,
    pub in_proj_a: Tensor,
    pub in_proj_b: Tensor,
    pub conv1d: Tensor,
    pub norm: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
}

pub struct GpuFfnWeights {
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

/// State for Gated DeltaNet linear attention layers.
///
/// Each linear attention layer maintains:
/// - A recurrent state matrix S of shape `[batch, num_value_heads, key_head_dim, value_head_dim]`
/// - A conv1d sliding window buffer of shape `[batch, conv_dim, kernel_size - 1]`
///
/// This state is O(1) in sequence length — it does not grow with the number of tokens processed.
pub struct LinearAttentionState {
    /// Per-layer recurrent state S. Length = number of linear attention layers.
    pub recurrent_states: Vec<Tensor>,
    /// Per-layer conv1d sliding window buffers. Length = number of linear attention layers.
    pub conv_states: Vec<Tensor>,
}

impl LinearAttentionState {
    /// Create fresh zero-initialized state for all linear attention layers.
    pub fn new(config: &kiln_core::config::ModelConfig, device: &Device) -> Result<Self> {
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let nv = config.linear_num_value_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let conv_dim = config.linear_qkv_dim();
        let k_minus_1 = config.linear_conv_kernel_dim.saturating_sub(1);

        let mut recurrent_states = Vec::with_capacity(num_linear_layers);
        let mut conv_states = Vec::with_capacity(num_linear_layers);

        for _ in 0..num_linear_layers {
            recurrent_states.push(Tensor::zeros((1, nv, dk, dv), DType::F32, device)?);
            conv_states.push(Tensor::zeros((1, conv_dim, k_minus_1), DType::F32, device)?);
        }

        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }
}

/// Convert a `WeightTensor` (raw bytes + shape + dtype) to a candle `Tensor` on `device`.
fn weight_to_tensor(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    let dtype = match w.dtype {
        TensorDType::F16 => DType::F16,
        TensorDType::BF16 => DType::BF16,
        TensorDType::F32 => DType::F32,
    };
    let t = Tensor::from_raw_buffer(&w.data, dtype, &w.shape, device)
        .context("failed to create tensor from raw buffer")?;
    Ok(t)
}

impl GpuWeights {
    /// Convert `ModelWeights` (CPU bytes) into candle tensors on the given device.
    pub fn from_model_weights(weights: &ModelWeights, device: &Device) -> Result<Self> {
        let embed_tokens =
            weight_to_tensor(&weights.embedding.embed_tokens, device).context("embed_tokens")?;
        let final_norm = weight_to_tensor(&weights.final_norm, device).context("final_norm")?;

        let mut layers = Vec::with_capacity(weights.layers.len());
        for (i, lw) in weights.layers.iter().enumerate() {
            let ctx = |name: &str| format!("layer {i} {name}");

            let input_layernorm =
                weight_to_tensor(&lw.input_layernorm, device).context(ctx("input_layernorm"))?;
            let post_attention_layernorm = weight_to_tensor(&lw.post_attention_layernorm, device)
                .context(ctx("post_attention_layernorm"))?;

            let attention = match &lw.attention {
                crate::weights::AttentionWeights::Full(attn) => {
                    GpuAttentionWeights::Full(GpuFullAttentionWeights {
                        q_proj: weight_to_tensor(&attn.q_proj, device).context(ctx("q_proj"))?,
                        k_proj: weight_to_tensor(&attn.k_proj, device).context(ctx("k_proj"))?,
                        v_proj: weight_to_tensor(&attn.v_proj, device).context(ctx("v_proj"))?,
                        o_proj: weight_to_tensor(&attn.o_proj, device).context(ctx("o_proj"))?,
                        q_norm: weight_to_tensor(&attn.q_norm, device).context(ctx("q_norm"))?,
                        k_norm: weight_to_tensor(&attn.k_norm, device).context(ctx("k_norm"))?,
                    })
                }
                crate::weights::AttentionWeights::Linear(attn) => {
                    GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                        in_proj_qkv: weight_to_tensor(&attn.in_proj_qkv, device)
                            .context(ctx("in_proj_qkv"))?,
                        in_proj_z: weight_to_tensor(&attn.in_proj_z, device)
                            .context(ctx("in_proj_z"))?,
                        out_proj: weight_to_tensor(&attn.out_proj, device)
                            .context(ctx("out_proj"))?,
                        in_proj_a: weight_to_tensor(&attn.in_proj_a, device)
                            .context(ctx("in_proj_a"))?,
                        in_proj_b: weight_to_tensor(&attn.in_proj_b, device)
                            .context(ctx("in_proj_b"))?,
                        conv1d: weight_to_tensor(&attn.conv1d, device).context(ctx("conv1d"))?,
                        norm: weight_to_tensor(&attn.norm, device).context(ctx("gdn_norm"))?,
                        a_log: weight_to_tensor(&attn.a_log, device).context(ctx("a_log"))?,
                        dt_bias: weight_to_tensor(&attn.dt_bias, device).context(ctx("dt_bias"))?,
                    })
                }
            };

            let mlp = GpuFfnWeights {
                gate_proj: weight_to_tensor(&lw.mlp.gate_proj, device).context(ctx("gate_proj"))?,
                up_proj: weight_to_tensor(&lw.mlp.up_proj, device).context(ctx("up_proj"))?,
                down_proj: weight_to_tensor(&lw.mlp.down_proj, device).context(ctx("down_proj"))?,
            };

            layers.push(GpuLayerWeights {
                input_layernorm,
                post_attention_layernorm,
                attention,
                mlp,
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
        })
    }
}

// ---------------------------------------------------------------------------
// Forward pass primitives
// ---------------------------------------------------------------------------

/// Look up token embeddings from the embedding table.
///
/// `token_ids`: 1-D slice of token IDs.
/// `embed_weights`: [vocab_size, hidden_size] embedding matrix.
///
/// Returns: [seq_len, hidden_size] tensor.
pub fn embedding_lookup(token_ids: &[u32], embed_weights: &Tensor) -> Result<Tensor> {
    let index = Tensor::new(token_ids, embed_weights.device())?;
    let out = embed_weights.index_select(&index, 0)?;
    Ok(out)
}

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps).
///
/// `x`: [..., hidden_size]
/// `weight`: [hidden_size] (learnable scale)
/// `eps`: small constant for numerical stability (1e-6 for Qwen3.5-4B)
///
/// Returns: same shape as `x`.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    // Qwen3.5 RMSNorm stores weights centered around 0 and applies as (1 + w) * x_normed.
    // Keep everything in F32 for precision (matches HF: `output * (1.0 + self.weight.float())`),
    // then cast back to input dtype at the end.
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_plus_one = (w_f32.ones_like()? + w_f32)?;
    let out = normed.broadcast_mul(&w_plus_one)?;
    Ok(out.to_dtype(x.dtype())?)
}

/// Apply Rotary Position Embeddings (RoPE) to query and key tensors.
///
/// `q`: [batch, seq_len, num_heads, head_dim]
/// `k`: [batch, seq_len, num_kv_heads, head_dim]
/// `positions`: position index for each token in the sequence (length = seq_len)
/// `head_dim`: dimension of each attention head
/// `rotary_dim`: number of head dimensions to apply rotation to (the rest pass through unchanged).
///   For Qwen3.5-4B: 64 (partial_rotary_factor=0.25, so 0.25 * 256 = 64).
/// `rope_theta`: base frequency (10_000_000.0 for Qwen3.5-4B)
///
/// Returns: (rotated_q, rotated_k) with same shapes.
pub fn rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    positions: &[u32],
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let half_rotary = rotary_dim / 2;

    // Compute frequency table: freq_i = 1.0 / (theta ^ (2i / rotary_dim)) for i in 0..half_rotary
    let inv_freq: Vec<f32> = (0..half_rotary)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?; // [half_rotary]

    // Position tensor
    let pos_f32: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let pos = Tensor::new(pos_f32.as_slice(), device)?.unsqueeze(1)?; // [seq_len, 1]

    // Outer product: [seq_len, half_rotary]
    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let cos = freqs.cos()?; // [seq_len, half_rotary]
    let sin = freqs.sin()?; // [seq_len, half_rotary]

    let rotated_q = apply_rope(q, &cos, &sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim, rotary_dim)?;

    Ok((rotated_q, rotated_k))
}

/// Same as [`rotary_embedding`] but accepts positions as a pre-allocated GPU tensor
/// instead of a CPU slice. This is critical for CUDA graph compatibility: the tensor's
/// GPU address stays stable across graph replays, and its contents can be updated via
/// `cudaMemcpyAsync` outside the captured graph.
///
/// `positions_tensor`: f32 tensor on device, shape [seq_len]
pub fn rotary_embedding_from_tensor(
    q: &Tensor,
    k: &Tensor,
    positions_tensor: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let half_rotary = rotary_dim / 2;

    let inv_freq: Vec<f32> = (0..half_rotary)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

    // positions_tensor is [seq_len], unsqueeze to [seq_len, 1]
    let pos = positions_tensor.unsqueeze(1)?;

    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    let rotated_q = apply_rope(q, &cos, &sin, head_dim, rotary_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim, rotary_dim)?;

    Ok((rotated_q, rotated_k))
}

/// Apply the rotation to a single tensor, supporting partial rotary embeddings.
/// `x`: [batch, seq_len, num_heads, head_dim]
/// `cos`, `sin`: [seq_len, half_rotary]
/// `head_dim`: total dimension per head
/// `rotary_dim`: number of dimensions to rotate (must be even). The first `rotary_dim` dims
///   are rotated; the remaining `head_dim - rotary_dim` dims pass through unchanged.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, head_dim: usize, rotary_dim: usize) -> Result<Tensor> {
    let half_rotary = rotary_dim / 2;
    let x_dtype = x.dtype();

    // Work in f32 for precision
    let x = x.to_dtype(DType::F32)?;

    // Split into rotary portion and passthrough portion
    let x_rot = x.narrow(candle_core::D::Minus1, 0, rotary_dim)?; // [..., :rotary_dim]
    let x_pass = if rotary_dim < head_dim {
        Some(x.narrow(candle_core::D::Minus1, rotary_dim, head_dim - rotary_dim)?) // [..., rotary_dim:]
    } else {
        None
    };

    // Split rotary portion into two halves
    let x1 = x_rot.narrow(candle_core::D::Minus1, 0, half_rotary)?; // [..., :half_rotary]
    let x2 = x_rot.narrow(candle_core::D::Minus1, half_rotary, half_rotary)?; // [..., half_rotary:rotary_dim]

    // cos/sin are [seq_len, half_rotary], need to broadcast to [batch, seq_len, num_heads, half_rotary]
    // Reshape to [1, seq_len, 1, half_rotary]
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;

    // Standard RoPE rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    // Concatenate rotated dims + passthrough dims
    let out = match x_pass {
        Some(pass) => Tensor::cat(&[&r1, &r2, &pass], candle_core::D::Minus1)?,
        None => Tensor::cat(&[&r1, &r2], candle_core::D::Minus1)?,
    };
    Ok(out.to_dtype(x_dtype)?)
}

/// SwiGLU feed-forward network.
///
/// Computes: down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
///
/// `x`: [batch, seq_len, hidden_size]
/// `gate_proj`: [intermediate_size, hidden_size]
/// `up_proj`: [intermediate_size, hidden_size]
/// `down_proj`: [hidden_size, intermediate_size]
///
/// Returns: [batch, seq_len, hidden_size]
pub fn swiglu_ffn(
    x: &Tensor,
    gate_proj: &Tensor,
    up_proj: &Tensor,
    down_proj: &Tensor,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    // x @ gate_proj^T -> [batch, seq_len, intermediate_size]
    let gate = linear_with_lora(x, gate_proj, lora_layer.and_then(|l| l.gate_proj.as_ref()), lora_scale)?;
    // SiLU activation: x * sigmoid(x)
    let gate = cuda_silu(&gate)?;
    // x @ up_proj^T -> [batch, seq_len, intermediate_size]
    let up = linear_with_lora(x, up_proj, lora_layer.and_then(|l| l.up_proj.as_ref()), lora_scale)?;
    // Element-wise multiply
    let hidden = (gate * up)?;
    // hidden @ down_proj^T -> [batch, seq_len, hidden_size]
    let out = linear_with_lora(&hidden, down_proj, lora_layer.and_then(|l| l.down_proj.as_ref()), lora_scale)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (GDN) linear attention primitives
// ---------------------------------------------------------------------------

/// L2 normalize the last dimension: x / sqrt(sum(x^2) + eps).
/// Returns result in F32 regardless of input dtype.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq_sum = x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let norm = (sq_sum + 1e-6)?.sqrt()?;
    let normalized = x_f32.broadcast_div(&norm)?;
    Ok(normalized)
}

/// softplus(x) = ln(1 + exp(x)), numerically stable for all x.
///
/// Uses the identity: softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
/// Since exp(-|x|) ∈ (0, 1], no overflow is possible.
/// This matches PyTorch's F.softplus output (which clamps to linear for x > 20).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let relu_x = x.maximum(&zeros)?;
    // |x| = relu(x) + relu(-x)
    let neg_x = x.neg()?;
    let relu_neg_x = neg_x.maximum(&zeros)?;
    let abs_x = (relu_x.clone() + relu_neg_x)?;
    let neg_abs = abs_x.neg()?;
    // log(1 + exp(-|x|)) — always stable since exp(-|x|) ∈ (0, 1]
    let log_term = (neg_abs.exp()? + 1.0)?.log()?;
    Ok((relu_x + log_term)?)
}

/// Gated RMSNorm: rms_norm(x, weight) * silu(z).
///
/// Applied per-group on the last dimension. Returns F32.
///
/// `x`: [..., dim] — attention output
/// `z`: [..., dim] — output gate (from in_proj_z)
/// `weight`: [dim] — learnable scale
fn gated_rms_norm(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let z_f32 = z.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;

    // RMS norm on last dimension
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    let normed = normed.broadcast_mul(&w_f32)?;

    // Output gate: silu(z) = z * sigmoid(z)
    let gate = cuda_silu(&z_f32)?;
    let out = (normed * gate)?;
    Ok(out)
}

/// Causal depthwise conv1d for prefill (seq_len > 1).
///
/// `x`: [batch, channels, seq_len]
/// `weight`: [channels, 1, kernel_size]
/// `conv_state`: [batch, channels, kernel_size - 1] — updated to last K-1 inputs
fn causal_conv1d_prefill(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let (_batch, channels, seq_len) = x.dims3()?;
    let _device = x.device();
    let x_f32 = x.to_dtype(DType::F32)?;
    // Squeeze [channels, 1, kernel_size] -> [channels, kernel_size]
    let w_f32 = weight.to_dtype(DType::F32)?.reshape((channels, kernel_size))?;
    let k_minus_1 = kernel_size - 1;

    // Left-pad with conv_state (previous K-1 inputs, or zeros for fresh state)
    let x_padded = Tensor::cat(&[&conv_state.to_dtype(DType::F32)?, &x_f32], 2)?;

    // Depthwise conv: output[t] = sum_{j=0}^{K-1} weight[j] * x_padded[t+j]
    let mut output = Tensor::zeros_like(&x_f32)?;
    for j in 0..kernel_size {
        let x_slice = x_padded.narrow(2, j, seq_len)?;
        let w_j = w_f32.narrow(1, j, 1)?.unsqueeze(0)?; // [1, channels, 1]
        output = (output + x_slice.broadcast_mul(&w_j)?)?;
    }

    // Update conv_state to the last K-1 input positions
    if seq_len >= k_minus_1 {
        *conv_state = x_f32
            .narrow(2, seq_len - k_minus_1, k_minus_1)?
            .contiguous()?;
    } else {
        // Fewer new tokens than buffer size: shift old state and append new
        let keep = k_minus_1 - seq_len;
        let old_part = conv_state
            .to_dtype(DType::F32)?
            .narrow(2, seq_len, keep)?;
        *conv_state = Tensor::cat(&[&old_part, &x_f32], 2)?.contiguous()?;
    }

    Ok(output)
}

/// Causal depthwise conv1d for decode (seq_len == 1).
///
/// `x`: [batch, channels, 1]
/// `weight`: [channels, 1, kernel_size]
/// `conv_state`: [batch, channels, kernel_size - 1] — updated
fn causal_conv1d_decode(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    let (_batch, channels, _one) = x.dims3()?;
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?.reshape((channels, kernel_size))?;

    // Full window = [conv_state | x] -> [batch, channels, kernel_size]
    let window = Tensor::cat(&[&conv_state.to_dtype(DType::F32)?, &x_f32], 2)?;

    // Dot product per channel: sum over kernel dimension
    let w_expanded = w_f32.unsqueeze(0)?; // [1, channels, kernel_size]
    let output = window.broadcast_mul(&w_expanded)?.sum(2)?; // [batch, channels]
    let output = output.unsqueeze(2)?; // [batch, channels, 1]

    // Update conv_state: drop oldest, append newest
    *conv_state = window.narrow(2, 1, kernel_size - 1)?.contiguous()?;

    Ok(output)
}

/// Gated DeltaNet (GDN) linear attention forward pass.
///
/// Implements the recurrent linear attention mechanism used by 24/32 layers in Qwen3.5-4B.
/// Uses data-dependent gating (alpha/beta) and a delta rule update for the recurrent state.
///
/// `x`: [batch, seq_len, hidden_size]
/// `weights`: linear attention projection weights
/// `config`: model configuration
/// `recurrent_state`: [batch, nv, dk, dv] — mutable recurrent state, updated in place
/// `conv_state`: [batch, conv_dim, kernel_size-1] — mutable conv buffer, updated in place
///
/// Returns: [batch, seq_len, hidden_size]
pub fn gated_deltanet_forward(
    x: &Tensor,
    weights: &GpuLinearAttentionWeights,
    config: &kiln_core::config::ModelConfig,
    recurrent_state: &mut Tensor,
    conv_state: &mut Tensor,
) -> Result<Tensor> {
    let (batch, seq_len, _hidden) = x.dims3()?;
    let input_dtype = x.dtype();
    let nk = config.linear_num_key_heads;
    let dk = config.linear_key_head_dim;
    let nv = config.linear_num_value_heads;
    let dv = config.linear_value_head_dim;
    let qk_dim = config.linear_qk_dim();
    let v_dim = config.linear_v_dim();
    let kernel_size = config.linear_conv_kernel_dim;
    let gqa_ratio = nv / nk;

    // --- Step 1: Input projections ---
    let mixed_qkv = x.broadcast_matmul(&weights.in_proj_qkv.t()?)?; // [B, T, qkv_dim]
    let z = x.broadcast_matmul(&weights.in_proj_z.t()?)?;           // [B, T, v_dim]
    let a = x.broadcast_matmul(&weights.in_proj_a.t()?)?;           // [B, T, nv]
    let b = x.broadcast_matmul(&weights.in_proj_b.t()?)?;           // [B, T, nv]

    // --- Step 2: Causal depthwise conv1d + SiLU on fused QKV ---
    // Transpose to [B, channels, T] for conv
    let mixed_qkv = mixed_qkv.transpose(1, 2)?.contiguous()?;
    let mixed_qkv = if seq_len > 1 {
        causal_conv1d_prefill(&mixed_qkv, &weights.conv1d, conv_state, kernel_size)?
    } else {
        causal_conv1d_decode(&mixed_qkv, &weights.conv1d, conv_state, kernel_size)?
    };
    // SiLU activation (work in F32 for stability)
    let mixed_qkv = cuda_silu(&mixed_qkv.to_dtype(DType::F32)?)?;
    // Transpose back to [B, T, qkv_dim]
    let mixed_qkv = mixed_qkv.transpose(1, 2)?;

    // --- Step 3: Split into Q, K, V and reshape to heads ---
    let q = mixed_qkv
        .narrow(2, 0, qk_dim)?
        .reshape((batch, seq_len, nk, dk))?;
    let k = mixed_qkv
        .narrow(2, qk_dim, qk_dim)?
        .reshape((batch, seq_len, nk, dk))?;
    let v = mixed_qkv
        .narrow(2, 2 * qk_dim, v_dim)?
        .reshape((batch, seq_len, nv, dv))?;
    let z = z.reshape((batch, seq_len, nv, dv))?;

    // --- Step 4: GQA head repeat (nk → nv) ---
    let (q, k) = if gqa_ratio > 1 {
        let q = q
            .unsqueeze(3)?
            .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
            .contiguous()?
            .reshape((batch, seq_len, nv, dk))?;
        let k = k
            .unsqueeze(3)?
            .expand(&[batch, seq_len, nk, gqa_ratio, dk])?
            .contiguous()?
            .reshape((batch, seq_len, nv, dk))?;
        (q, k)
    } else {
        (q.contiguous()?, k.contiguous()?)
    };

    // --- Step 5: L2 normalize Q, K; scale Q by 1/sqrt(dk) ---
    // Normalize in F32 for numerical stability (sqrt/div), then cast back to
    // the input dtype so the recurrent loop below runs in bf16 (see Step 7).
    let q = l2_normalize(&q)?; // F32
    let k = l2_normalize(&k)?; // F32
    let scale = 1.0 / (dk as f64).sqrt();
    let q = (q * scale)?.to_dtype(input_dtype)?;
    let k = k.to_dtype(input_dtype)?;

    // --- Step 6: Compute gates ---
    // beta = sigmoid(b) — write gate, in (0, 1). sigmoid output is bounded so
    // bf16 has enough precision; no F32 upcast needed.
    let beta = cuda_sigmoid(&b)?; // [B, T, nv], bf16

    // g = -exp(A_log) * softplus(a + dt_bias) — decay (negative log-space).
    // The softplus/exp pipeline is computed in F32 for stability (it involves
    // exp and log near 0), then cast to the input dtype for the bf16 loop.
    let a_f32 = a.to_dtype(DType::F32)?;
    let a_log_f32 = weights.a_log.to_dtype(DType::F32)?;
    let dt_bias_f32 = weights.dt_bias.to_dtype(DType::F32)?;
    let g = {
        let a_biased = a_f32.broadcast_add(&dt_bias_f32)?;
        let sp = softplus(&a_biased)?;
        let neg_decay = a_log_f32.exp()?.neg()?; // -exp(A_log)
        sp.broadcast_mul(&neg_decay)?
    }
    .to_dtype(input_dtype)?; // [B, T, nv], negative values → exp(g) ∈ (0, 1)

    // --- Step 7: Sequential recurrence ---
    // The recurrent state is stored in F32 externally (across layers/steps)
    // for accumulator stability, but we run the loop in bf16 to reclaim the
    // ~66% of prefill GPU time previously spent in bmul_f32 / fast_sum_f32 /
    // badd_f32 (see PROFILING.md recommendation #2). State is cast bf16 at
    // entry and restored to F32 at exit so the external invariant holds.
    let state_external_dtype = recurrent_state.dtype();
    if state_external_dtype != input_dtype {
        *recurrent_state = recurrent_state.to_dtype(input_dtype)?;
    }

    // Transpose to [B, nv, T, dim] for per-head processing
    let q = q.transpose(1, 2)?; // [B, nv, T, dk]
    let k = k.transpose(1, 2)?; // [B, nv, T, dk]
    let v = v.transpose(1, 2)?; // [B, nv, T, dv]
    let beta = beta.transpose(1, 2)?; // [B, nv, T]
    let g = g.transpose(1, 2)?; // [B, nv, T]

    let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let q_t = q.narrow(2, t, 1)?.squeeze(2)?; // [B, nv, dk]
        let k_t = k.narrow(2, t, 1)?.squeeze(2)?; // [B, nv, dk]
        let v_t = v.narrow(2, t, 1)?.squeeze(2)?; // [B, nv, dv]
        let beta_t = beta.narrow(2, t, 1)?.squeeze(2)?; // [B, nv]
        let g_t = g.narrow(2, t, 1)?.squeeze(2)?; // [B, nv]

        // 1. Decay: S *= exp(g_t)
        let g_exp = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?; // [B, nv, 1, 1]
        *recurrent_state = recurrent_state.broadcast_mul(&g_exp)?;

        // 2. Read from memory: kv_mem = einsum('bhkv,bhk->bhv', S, k_t)
        let k_expanded = k_t.unsqueeze(3)?; // [B, nv, dk, 1]
        let kv_mem = recurrent_state.broadcast_mul(&k_expanded)?.sum(2)?; // [B, nv, dv]

        // 3. Delta rule: delta = (v_t - kv_mem) * beta_t
        let delta: Tensor = (v_t - kv_mem)?.broadcast_mul(&beta_t.unsqueeze(2)?)?; // [B, nv, dv]

        // 4. Update: S += outer(k_t, delta) — reference add, no explicit clone.
        // The `Tensor::add` op takes references and allocates a new tensor for
        // the result regardless of how `lhs` is passed, so clone() was pure
        // Arc churn with no effect on storage allocation.
        let outer = k_t.unsqueeze(3)?.broadcast_mul(&delta.unsqueeze(2)?)?; // [B, nv, dk, dv]
        let new_state = (&*recurrent_state + &outer)?;
        *recurrent_state = new_state;

        // 5. Output: out_t = einsum('bhkv,bhk->bhv', S, q_t)
        let q_expanded = q_t.unsqueeze(3)?; // [B, nv, dk, 1]
        let out_t = recurrent_state.broadcast_mul(&q_expanded)?.sum(2)?; // [B, nv, dv]
        outputs.push(out_t.unsqueeze(2)?); // [B, nv, 1, dv]
    }

    // Restore state to its original dtype so the caller's F32 invariant holds
    // across layer calls and across prefill/decode steps.
    if state_external_dtype != input_dtype {
        *recurrent_state = recurrent_state.to_dtype(state_external_dtype)?;
    }

    // Collect: [B, nv, T, dv]
    let attn_out = Tensor::cat(&outputs, 2)?;
    // Transpose to [B, T, nv, dv]
    let attn_out = attn_out.transpose(1, 2)?;

    // --- Step 8: Gated RMSNorm — norm(attn_out) * silu(z) ---
    let attn_out = gated_rms_norm(&attn_out, &z, &weights.norm, config.rms_norm_eps)?;
    // Reshape to [B, T, v_dim] and cast back to input dtype
    let attn_out = attn_out
        .reshape((batch, seq_len, v_dim))?
        .to_dtype(input_dtype)?;

    // --- Step 9: Output projection ---
    // NOTE: conv1d bias is not loaded by the weight loader. If the model has one,
    // it should be added to GpuLinearAttentionWeights and applied after conv1d.
    let out = attn_out.broadcast_matmul(&weights.out_proj.t()?)?;
    Ok(out)
}

/// Grouped-Query Attention (GQA).
///
/// Computes scaled dot-product attention with fewer KV heads than Q heads.
/// Each group of `num_heads / num_kv_heads` query heads shares one KV head.
///
/// `x`: [batch, seq_len, hidden_size]
/// `attn_weights`: Q/K/V/O projection weights plus per-head RMSNorm weights
/// `positions`: position indices for RoPE (length = seq_len, absolute positions)
/// `num_heads`: number of query attention heads
/// `num_kv_heads`: number of key/value attention heads
/// `head_dim`: dimension per head
/// `rope_theta`: RoPE base frequency
/// `rms_norm_eps`: epsilon for Q/K head norms
/// `kv_cache`: optional KV cache for incremental decoding
/// `full_attn_layer_idx`: index into the KV cache's layer array (only full-attn layers)
///
/// Returns: [batch, seq_len, hidden_size]
pub fn gqa_attention(
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    kv_cache: Option<&mut KvCache>,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;

    // Project to Q, K, V (with optional LoRA delta)
    // When attn_output_gate is true, q_proj outputs [Q, gate] fused:
    //   q_proj: [num_heads * head_dim * 2, hidden_size]
    //   Split into Q [num_heads, head_dim] and gate [num_heads, head_dim]
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let q_raw = linear_with_lora(x, &attn_weights.q_proj, lora_layer.and_then(|l| l.q_proj.as_ref()), lora_scale)?;
    let k = linear_with_lora(x, &attn_weights.k_proj, lora_layer.and_then(|l| l.k_proj.as_ref()), lora_scale)?;
    let v = linear_with_lora(x, &attn_weights.v_proj, lora_layer.and_then(|l| l.v_proj.as_ref()), lora_scale)?;

    // Split Q and gate if output gate is enabled
    let (q, gate) = if attn_output_gate {
        // q_raw: [batch, seq_len, num_heads * head_dim * 2]
        // Reshape to [batch, seq_len, num_heads, head_dim * 2] then split
        let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
        let q = q_raw.narrow(3, 0, head_dim)?;
        let gate = q_raw.narrow(3, head_dim, head_dim)?;
        // gate needs to be [batch, seq_len, num_heads * head_dim] for later
        let gate = gate.contiguous()?.reshape(((), seq_len, num_heads * head_dim))?;
        (q.contiguous()?, Some(gate))
    } else {
        let q = q_raw.reshape(((), seq_len, num_heads, head_dim))?;
        (q, None)
    };

    // Reshape K, V to [batch, seq_len, num_heads, head_dim]
    let k = k.reshape(((), seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?;

    // Apply per-head RMSNorm to Q and K (Qwen3.5 uses QK-norm)
    // q_norm/k_norm are [head_dim] — broadcast over [batch, seq_len, num_heads, head_dim]
    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;

    // Apply RoPE (positions are absolute, so cached tokens get correct embeddings)
    // Only rotate first rotary_dim dimensions; the rest pass through unchanged.
    let (q, k) = rotary_embedding(&q, &k, positions, head_dim, rotary_dim, rope_theta)?;

    // FlashAttention-2 path for prefill (seq_len > 1, no KV cache).
    // Flash-attn takes [batch, seq_len, num_heads, head_dim] — the layout we already have.
    // When a KV cache is present, we fall through to the naive path which handles
    // the cache update and Q_len != KV_len masking correctly.
    #[cfg(feature = "cuda")]
    if seq_len > 1 && kv_cache.is_none() && q.dtype() == candle_core::DType::BF16 {
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn_output = flash_attention_forward(&q, &k, &v, num_heads, num_kv_heads, head_dim)?;
        // Apply output gate: attn_output * sigmoid(gate)
        let attn_output = if let Some(ref gate) = gate {
            let sigmoid_gate = cuda_sigmoid(gate)?;
            (attn_output * sigmoid_gate)?
        } else {
            attn_output
        };
        let out = linear_with_lora(&attn_output, &attn_weights.o_proj, lora_layer.and_then(|l| l.o_proj.as_ref()), lora_scale)?;
        return Ok(out);
    }

    // Transpose to [batch, heads, seq_len, head_dim] for naive attention
    let q = q.transpose(1, 2)?.contiguous()?; // [batch, num_heads, seq_len, head_dim]
    let k = k.transpose(1, 2)?.contiguous()?; // [batch, num_kv_heads, seq_len, head_dim]
    let v = v.transpose(1, 2)?.contiguous()?; // [batch, num_kv_heads, seq_len, head_dim]

    // If KV cache is provided, update it and use full cached K/V
    let (k, v, kv_len) = if let Some(cache) = kv_cache {
        let (full_k, full_v) = cache
            .update(full_attn_layer_idx, &k, &v)
            .context("KV cache update failed")?;
        let kv_len = full_k.dim(2)?;
        (full_k, full_v, kv_len)
    } else {
        (k, v, seq_len)
    };

    // GQA head expansion: repeat K/V to match Q head count
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;
    let (k, v) = if gqa_ratio > 1 {
        // Expand [batch, num_kv_heads, kv_len, head_dim] -> [batch, num_heads, kv_len, head_dim]
        let k = k
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        let v = v
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        (k, v)
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // Scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    // Q: [batch, num_heads, seq_len, head_dim]
    // K: [batch, num_heads, kv_len, head_dim]
    // scores: [batch, num_heads, seq_len, kv_len]
    let scale = (head_dim as f64).sqrt();
    let attn_scores = q.broadcast_matmul(&k.t()?)?;
    let attn_scores = (attn_scores / scale)?;

    // Apply causal mask (handles Q_len != KV_len for cached decoding)
    let past_len = kv_len - seq_len;
    let attn_scores = apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, past_len)?;

    let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;
    let attn_output = attn_weights_softmax.broadcast_matmul(&v)?; // [batch, num_heads, seq_len, head_dim]

    // Transpose back: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden]
    let attn_output = attn_output
        .transpose(1, 2)?
        .contiguous()?
        .reshape(((), seq_len, num_heads * head_dim))?;

    // Apply output gate: attn_output * sigmoid(gate)
    let attn_output = if let Some(ref gate) = gate {
        let sigmoid_gate = cuda_sigmoid(gate)?;
        (attn_output * sigmoid_gate)?
    } else {
        attn_output
    };

    // Output projection
    let out = linear_with_lora(&attn_output, &attn_weights.o_proj, lora_layer.and_then(|l| l.o_proj.as_ref()), lora_scale)?;
    Ok(out)
}

/// Grouped-query attention using a paged KV cache.
///
/// Same computation as [`gqa_attention`] but reads/writes K/V through a
/// [`PagedKvCache`] and [`BlockTable`] instead of a contiguous [`KvCache`].
/// This enables multiple concurrent sequences to share a fixed KV cache pool.
///
/// The caller must ensure the block table has enough blocks allocated for all
/// positions up to `positions.last() + 1`.
pub fn gqa_attention_paged(
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    attn_output_gate: bool,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;

    // Project to Q, K, V (with optional LoRA delta and output gate split)
    let (lora_layer, lora_scale) = match lora {
        Some((l, s)) => (Some(l), s),
        None => (None, 0.0),
    };
    let q_raw = linear_with_lora(x, &attn_weights.q_proj, lora_layer.and_then(|l| l.q_proj.as_ref()), lora_scale)?;
    let k = linear_with_lora(x, &attn_weights.k_proj, lora_layer.and_then(|l| l.k_proj.as_ref()), lora_scale)?;
    let v = linear_with_lora(x, &attn_weights.v_proj, lora_layer.and_then(|l| l.v_proj.as_ref()), lora_scale)?;

    let (q, gate) = if attn_output_gate {
        let q_raw = q_raw.reshape(((), seq_len, num_heads, head_dim * 2))?;
        let q = q_raw.narrow(3, 0, head_dim)?;
        let gate = q_raw.narrow(3, head_dim, head_dim)?;
        let gate = gate.contiguous()?.reshape(((), seq_len, num_heads * head_dim))?;
        (q.contiguous()?, Some(gate))
    } else {
        let q = q_raw.reshape(((), seq_len, num_heads, head_dim))?;
        (q, None)
    };

    let k = k.reshape(((), seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?;

    // QK-norm
    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;

    // RoPE — only rotate first rotary_dim dimensions
    // Use the GPU tensor variant so positions remain at a stable GPU address
    // (critical for CUDA graph replay correctness)
    let (q, k) = rotary_embedding_from_tensor(&q, &k, positions, head_dim, rotary_dim, rope_theta)?;

    // Transpose to [batch, heads, seq_len, head_dim]
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;

    // Write new K/V into paged cache
    paged_cache
        .write(full_attn_layer_idx, block_table, start_pos, &k, &v)
        .context("paged KV cache write failed")?;

    // Read full K/V from paged cache (all positions 0..start_pos+seq_len)
    let total_seq_len = start_pos + seq_len;
    let (k, v) = paged_cache
        .read(full_attn_layer_idx, block_table, total_seq_len)
        .context("paged KV cache read failed")?;
    let kv_len = total_seq_len;

    // FlashAttention-2 path for prefill (seq_len > 1).
    // Paged cache returns [batch, heads, kv_len, head_dim] — transpose to
    // [batch, kv_len, heads, head_dim] for flash_attn.
    #[cfg(feature = "cuda")]
    if seq_len > 1 && q.dtype() == candle_core::DType::BF16 {
        let q = q.transpose(1, 2)?.contiguous()?; // -> [batch, seq_len, num_heads, head_dim]
        let k = k.transpose(1, 2)?.contiguous()?; // -> [batch, kv_len, num_kv_heads, head_dim]
        let v = v.transpose(1, 2)?.contiguous()?; // -> [batch, kv_len, num_kv_heads, head_dim]
        let attn_output = flash_attention_forward(&q, &k, &v, num_heads, num_kv_heads, head_dim)?;
        // Apply output gate: attn_output * sigmoid(gate)
        let attn_output = if let Some(ref gate) = gate {
            let sigmoid_gate = cuda_sigmoid(gate)?;
            (attn_output * sigmoid_gate)?
        } else {
            attn_output
        };
        let out = linear_with_lora(&attn_output, &attn_weights.o_proj, lora_layer.and_then(|l| l.o_proj.as_ref()), lora_scale)?;
        return Ok(out);
    }

    // GQA head expansion and attention
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;

    // Optimized decode path (seq_len == 1): reshape Q instead of expanding K/V.
    // Q is [batch, num_heads, 1, head_dim] (1 token) while K/V is
    // [batch, num_kv_heads, kv_len, head_dim] (full history). Expanding K/V
    // copies kv_len * head_dim * num_kv_heads data gqa_ratio times.
    // Instead, group Q heads to match KV heads and compute per-group attention.
    if seq_len == 1 && gqa_ratio > 1 {
        let scale = (head_dim as f64).sqrt();

        // Reshape Q: [batch, num_heads, 1, head_dim]
        //          -> [batch, num_kv_heads, gqa_ratio, 1, head_dim]
        //          -> [batch * num_kv_heads, gqa_ratio, 1, head_dim]
        // K:         [batch, num_kv_heads, kv_len, head_dim]
        //          -> [batch * num_kv_heads, kv_len, head_dim]
        // V:         same as K
        let q_grouped = q
            .reshape((batch, num_kv_heads, gqa_ratio, 1, head_dim))?
            .reshape((batch * num_kv_heads, gqa_ratio, 1, head_dim))?
            .contiguous()?;
        // Unsqueeze K/V to [batch*num_kv_heads, 1, kv_len, head_dim] so that
        // broadcast_matmul pairs each Q group with its own KV head (dim 0),
        // broadcasting over the gqa_ratio dim (dim 1).  Without the unsqueeze
        // the 3-D K would be padded to [1, batch*num_kv_heads, ...] and the
        // gqa_ratio dim would incorrectly index into different KV heads.
        let k_flat = k
            .reshape((batch * num_kv_heads, kv_len, head_dim))?
            .unsqueeze(1)?
            .contiguous()?;
        let v_flat = v
            .reshape((batch * num_kv_heads, kv_len, head_dim))?
            .unsqueeze(1)?
            .contiguous()?;

        // Attention scores: [batch*num_kv_heads, gqa_ratio, 1, kv_len]
        let attn_scores = q_grouped.broadcast_matmul(&k_flat.transpose(2, 3)?.contiguous()?)?;
        let attn_scores = (attn_scores / scale)?;
        // No causal mask needed for decode (q_len=1 attends to everything)
        let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;

        // Weighted sum: [batch*num_kv_heads, gqa_ratio, 1, head_dim]
        let attn_output = attn_weights_softmax.broadcast_matmul(&v_flat)?;

        // Reshape back: -> [batch, num_kv_heads * gqa_ratio, 1, head_dim]
        //               == [batch, num_heads, 1, head_dim]
        let attn_output = attn_output
            .reshape((batch, num_heads, 1, head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, 1, num_heads * head_dim))?;

        let attn_output = if let Some(ref gate) = gate {
            let sigmoid_gate = cuda_sigmoid(gate)?;
            (attn_output * sigmoid_gate)?
        } else {
            attn_output
        };
        let out = linear_with_lora(&attn_output, &attn_weights.o_proj, lora_layer.and_then(|l| l.o_proj.as_ref()), lora_scale)?;
        return Ok(out);
    }

    // Standard path (prefill without flash-attn, or gqa_ratio == 1)
    let (k, v) = if gqa_ratio > 1 {
        let k = k
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        let v = v
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, kv_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, kv_len, head_dim))?;
        (k, v)
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // Scaled dot-product attention
    let scale = (head_dim as f64).sqrt();
    let attn_scores = q.broadcast_matmul(&k.t()?)?;
    let attn_scores = (attn_scores / scale)?;

    let past_len = kv_len - seq_len;
    let attn_scores = apply_causal_mask_with_offset(&attn_scores, seq_len, kv_len, past_len)?;

    let attn_weights_softmax = cuda_softmax_last_dim(&attn_scores)?;
    let attn_output = attn_weights_softmax.broadcast_matmul(&v)?;

    // Transpose back and output projection
    let attn_output = attn_output
        .transpose(1, 2)?
        .contiguous()?
        .reshape(((), seq_len, num_heads * head_dim))?;

    let attn_output = if let Some(ref gate) = gate {
        let sigmoid_gate = cuda_sigmoid(gate)?;
        (attn_output * sigmoid_gate)?
    } else {
        attn_output
    };

    let out = linear_with_lora(&attn_output, &attn_weights.o_proj, lora_layer.and_then(|l| l.o_proj.as_ref()), lora_scale)?;
    Ok(out)
}

/// Apply a causal (lower-triangular) mask to attention scores.
/// Sets future positions to -inf so softmax zeroes them out.
fn apply_causal_mask(scores: &Tensor, seq_len: usize) -> Result<Tensor> {
    apply_causal_mask_with_offset(scores, seq_len, seq_len, 0)
}

/// Apply a causal mask with support for KV cache offset.
///
/// When using a KV cache, Q has `q_len` new positions and K/V has `kv_len` total
/// positions (past_len cached + q_len new). Each query position `i` (representing
/// absolute position `past_len + i`) can attend to all KV positions up to and
/// including itself: positions `0..past_len + i + 1`.
///
/// `scores`: [batch, heads, q_len, kv_len]
/// `q_len`: number of new query positions
/// `kv_len`: total KV length (past_len + q_len)
/// `past_len`: number of cached positions before the new tokens
fn apply_causal_mask_with_offset(
    scores: &Tensor,
    q_len: usize,
    kv_len: usize,
    past_len: usize,
) -> Result<Tensor> {
    if q_len <= 1 && kv_len <= 1 {
        return Ok(scores.clone());
    }
    // During decode (q_len=1), the single new token can attend to all kv_len
    // positions (all past + itself), so no masking needed.
    if q_len == 1 {
        return Ok(scores.clone());
    }
    let device = scores.device();
    // Build a [q_len, kv_len] mask: 0 for allowed, -inf for masked
    // Query position i (absolute: past_len + i) can attend to KV positions 0..past_len+i+1
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            let max_kv = past_len + i + 1; // last allowed KV position (exclusive)
            (0..kv_len).map(move |j| if j < max_kv { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    let mask = Tensor::new(mask, device)?.reshape((1, 1, q_len, kv_len))?;
    let mask = mask.to_dtype(scores.dtype())?;
    let out = scores.broadcast_add(&mask)?;
    Ok(out)
}

/// Single transformer block: norm -> attention -> residual -> norm -> FFN -> residual.
///
/// `x`: [batch, seq_len, hidden_size]
/// `layer`: weights for this transformer layer
/// `positions`: position indices for RoPE (absolute positions)
/// `num_heads`: number of query attention heads
/// `num_kv_heads`: number of key/value attention heads
/// `head_dim`: dimension per head
/// `rotary_dim`: number of head dims to rotate (partial RoPE)
/// `rope_theta`: RoPE base frequency
/// `rms_norm_eps`: epsilon for RMSNorm
/// `kv_cache`: optional KV cache for incremental decoding
/// `full_attn_layer_idx`: index into the KV cache's layer array
///
/// Returns: [batch, seq_len, hidden_size]
pub fn transformer_block(
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    kv_cache: Option<&mut KvCache>,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!("transformer_block only supports full attention layers (not linear/GDN)")
        }
    };

    // Pre-attention norm
    let normed = rms_norm(x, &layer.input_layernorm, rms_norm_eps)?;

    // Self-attention
    let attn_out = gqa_attention(
        &normed,
        attn_weights,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        rope_theta,
        rms_norm_eps,
        kv_cache,
        full_attn_layer_idx,
        config.attn_output_gate,
        lora,
    )?;

    // Residual connection
    let x = (x + attn_out)?;

    // Post-attention norm
    let normed = rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?;

    // Feed-forward network
    let ffn_out = swiglu_ffn(
        &normed,
        &layer.mlp.gate_proj,
        &layer.mlp.up_proj,
        &layer.mlp.down_proj,
        lora,
    )?;

    // Residual connection
    let out = (x + ffn_out)?;
    Ok(out)
}

/// Transformer block using paged KV cache.
///
/// Same as [`transformer_block`] but reads/writes K/V through a [`PagedKvCache`]
/// and [`BlockTable`] instead of a contiguous [`KvCache`].
pub fn transformer_block_paged(
    x: &Tensor,
    layer: &GpuLayerWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &Tensor,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    full_attn_layer_idx: usize,
    lora: Option<(&LoraLayerWeights, f32)>,
) -> Result<Tensor> {
    let attn_weights = match &layer.attention {
        GpuAttentionWeights::Full(w) => w,
        GpuAttentionWeights::Linear(_) => {
            anyhow::bail!("transformer_block_paged only supports full attention layers (not linear/GDN)")
        }
    };

    // Pre-attention norm
    let normed = rms_norm(x, &layer.input_layernorm, rms_norm_eps)?;

    // Self-attention with paged cache
    let attn_out = gqa_attention_paged(
        &normed,
        attn_weights,
        positions,
        start_pos,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        rope_theta,
        rms_norm_eps,
        paged_cache,
        block_table,
        full_attn_layer_idx,
        config.attn_output_gate,
        lora,
    )?;

    // Residual connection
    let x = (x + attn_out)?;

    // Post-attention norm
    let normed = rms_norm(&x, &layer.post_attention_layernorm, rms_norm_eps)?;

    // Feed-forward network
    let ffn_out = swiglu_ffn(
        &normed,
        &layer.mlp.gate_proj,
        &layer.mlp.up_proj,
        &layer.mlp.down_proj,
        lora,
    )?;

    // Residual connection
    let out = (x + ffn_out)?;
    Ok(out)
}

/// Full model forward pass: embedding → N transformer blocks → final norm → LM head → logits.
///
/// `token_ids`: 1-D slice of token IDs for the input sequence.
/// `weights`: pre-loaded GPU tensors for all model parameters.
/// `config`: model architecture configuration.
/// `kv_cache`: optional KV cache for incremental decoding. When provided, `token_ids`
///   should contain only the new (not yet cached) tokens, and positions are computed
///   starting from `kv_cache.seq_len()`.
///
/// Returns logits tensor with shape [1, seq_len, vocab_size].
///
/// Notes:
/// - Qwen3.5-4B uses weight tying: the LM head reuses `embed_tokens` transposed.
/// - Linear attention (Gated DeltaNet) layers are not yet implemented and will
///   be skipped with an identity pass-through.
/// - After this function returns, the caller must call `kv_cache.advance(token_ids.len())`
///   to update the cached sequence length.
pub fn model_forward(
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    mut kv_cache: Option<&mut KvCache>,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    let seq_len = token_ids.len();

    // 1. Embedding lookup: [seq_len, hidden_size]
    let mut hidden = embedding_lookup(token_ids, &weights.embed_tokens)?;

    // Add batch dimension: [1, seq_len, hidden_size]
    hidden = hidden.unsqueeze(0)?;

    // Position indices for RoPE — absolute positions accounting for cached tokens
    let offset = kv_cache.as_ref().map_or(0, |c| c.seq_len());
    let positions: Vec<u32> = (offset..offset + seq_len).map(|p| p as u32).collect();

    // 2. Loop through all transformer layers
    // Track full-attention layer index (0-based counter of only full-attn layers)
    let mut full_attn_idx: usize = 0;
    let mut linear_attn_idx: usize = 0;
    for (i, layer) in weights.layers.iter().enumerate() {
        // Get LoRA weights for this layer, if available
        let layer_lora: Option<(&LoraLayerWeights, f32)> = lora
            .and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Reborrow the cache for each layer call
                let cache_ref = kv_cache.as_mut().map(|c| &mut **c);
                hidden = transformer_block(
                    &hidden,
                    layer,
                    config,
                    &positions,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    config.rope_theta,
                    config.rms_norm_eps,
                    cache_ref,
                    full_attn_idx,
                    layer_lora,
                )
                .with_context(|| format!("transformer block {i} (full attention)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut()
                    .ok_or_else(|| anyhow::anyhow!("linear attention state required for GDN layers (layer {i})"))?;
                // Pre-attention RMSNorm
                let normed = rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?;
                // Gated DeltaNet linear attention
                let attn_out = gated_deltanet_forward(
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                )
                .with_context(|| format!("gated deltanet layer {i} (linear attention)"))?;
                hidden = (hidden + attn_out)?;
                // Post-attention RMSNorm + FFN
                let normed_post = rms_norm(&hidden, &layer.post_attention_layernorm, config.rms_norm_eps)?;
                let ffn_out = swiglu_ffn(
                    &normed_post,
                    &layer.mlp.gate_proj,
                    &layer.mlp.up_proj,
                    &layer.mlp.down_proj,
                    layer_lora,
                )?;
                hidden = (hidden + ffn_out)?;
                linear_attn_idx += 1;
            }
        }
    }

    // 3. Final RMSNorm
    hidden = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;

    // 4. LM head projection (weight-tied: reuse embed_tokens transposed)
    // hidden: [1, seq_len, hidden_size], embed_tokens: [vocab_size, hidden_size]
    // logits = hidden @ embed_tokens^T -> [1, seq_len, vocab_size]
    let logits = hidden.broadcast_matmul(&weights.embed_tokens.t()?)?;

    Ok(logits)
}

/// Run a subset of transformer layers on an existing hidden state.
///
/// Processes layers `[start_layer..end_layer)` without embedding or LM head.
/// Used by gradient checkpointing to recompute individual segments.
///
/// `hidden`: [1, seq_len, hidden_size] — input hidden state.
/// `positions`: absolute position indices for RoPE.
/// `linear_state`: mutable linear attention state (only entries for layers in range are touched).
///
/// Returns: [1, seq_len, hidden_size] — output hidden state.
pub fn model_forward_segment(
    mut hidden: Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    positions: &[u32],
    start_layer: usize,
    end_layer: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
) -> Result<Tensor> {
    // Count full-attention and linear-attention layers before start_layer
    // so we index into the right KV cache / linear state slots.
    let mut full_attn_idx: usize = (0..start_layer)
        .filter(|&i| matches!(&weights.layers[i].attention, GpuAttentionWeights::Full(_)))
        .count();
    let mut linear_attn_idx: usize = (0..start_layer)
        .filter(|&i| matches!(&weights.layers[i].attention, GpuAttentionWeights::Linear(_)))
        .count();

    for i in start_layer..end_layer {
        let layer = &weights.layers[i];
        let layer_lora: Option<(&LoraLayerWeights, f32)> = lora
            .and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                // Training doesn't use KV cache
                hidden = transformer_block(
                    &hidden,
                    layer,
                    config,
                    positions,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    config.rope_theta,
                    config.rms_norm_eps,
                    None, // no KV cache for training
                    full_attn_idx,
                    layer_lora,
                )
                .with_context(|| format!("segment transformer block {i} (full attention)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut()
                    .ok_or_else(|| anyhow::anyhow!("linear attention state required for GDN layers (layer {i})"))?;
                let normed = rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?;
                let attn_out = gated_deltanet_forward(
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                )
                .with_context(|| format!("segment gated deltanet layer {i}"))?;
                hidden = (hidden + attn_out)?;
                let normed_post = rms_norm(&hidden, &layer.post_attention_layernorm, config.rms_norm_eps)?;
                let ffn_out = swiglu_ffn(
                    &normed_post,
                    &layer.mlp.gate_proj,
                    &layer.mlp.up_proj,
                    &layer.mlp.down_proj,
                    layer_lora,
                )?;
                hidden = (hidden + ffn_out)?;
                linear_attn_idx += 1;
            }
        }
    }

    Ok(hidden)
}

/// Compute embedding lookup and add batch dimension.
///
/// Returns `([1, seq_len, hidden_size], positions)` — the initial hidden state
/// and position indices for RoPE (starting from position 0, no KV cache offset).
pub fn model_forward_embed(
    token_ids: &[u32],
    weights: &GpuWeights,
) -> Result<(Tensor, Vec<u32>)> {
    let seq_len = token_ids.len();
    let mut hidden = embedding_lookup(token_ids, &weights.embed_tokens)?;
    hidden = hidden.unsqueeze(0)?;
    let positions: Vec<u32> = (0..seq_len).map(|p| p as u32).collect();
    Ok((hidden, positions))
}

/// Apply final RMSNorm and LM head projection.
///
/// `hidden`: [1, seq_len, hidden_size]
/// Returns: [1, seq_len, vocab_size] logits.
pub fn model_forward_head(
    hidden: &Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
    let logits = normed.broadcast_matmul(&weights.embed_tokens.t()?)?;
    Ok(logits)
}

/// Full model forward pass using paged KV cache.
///
/// Same as [`model_forward`] but uses a [`PagedKvCache`] and [`BlockTable`]
/// for KV storage. The caller provides `start_pos` (the absolute position of
/// the first token in `token_ids`) instead of relying on `kv_cache.seq_len()`.
///
/// `positions_gpu`: optional pre-allocated f32 tensor on device with shape [seq_len].
/// When provided, this tensor is used for RoPE instead of creating a new one.
/// This is required for CUDA graph replay: the tensor's GPU address must remain
/// stable so the captured graph reads updated position values on replay.
///
/// Returns logits tensor with shape [1, seq_len, vocab_size].
pub fn model_forward_paged(
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    start_pos: usize,
    mut linear_state: Option<&mut LinearAttentionState>,
    lora: Option<&LoraWeights>,
    positions_gpu: Option<&Tensor>,
) -> Result<Tensor> {
    let seq_len = token_ids.len();
    let device = weights.embed_tokens.device();

    // 1. Embedding lookup: [seq_len, hidden_size]
    let mut hidden = embedding_lookup(token_ids, &weights.embed_tokens)?;

    // Add batch dimension: [1, seq_len, hidden_size]
    hidden = hidden.unsqueeze(0)?;

    // Position tensor for RoPE — use pre-allocated GPU tensor if provided,
    // otherwise create one from scratch. The pre-allocated path is essential
    // for CUDA graph replay where the tensor pointer must be stable.
    let positions_owned;
    let positions: &Tensor = match positions_gpu {
        Some(t) => t,
        None => {
            let pos_f32: Vec<f32> = (start_pos..start_pos + seq_len)
                .map(|p| p as f32)
                .collect();
            positions_owned = Tensor::new(pos_f32.as_slice(), device)?;
            &positions_owned
        }
    };

    // 2. Loop through all transformer layers
    let mut full_attn_idx: usize = 0;
    let mut linear_attn_idx: usize = 0;
    for (i, layer) in weights.layers.iter().enumerate() {
        // Get LoRA weights for this layer, if available
        let layer_lora: Option<(&LoraLayerWeights, f32)> = lora
            .and_then(|lw| lw.layers.get(i).map(|ll| (ll, lw.scale)));

        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                hidden = transformer_block_paged(
                    &hidden,
                    layer,
                    config,
                    positions,
                    start_pos,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rotary_dim(),
                    config.rope_theta,
                    config.rms_norm_eps,
                    paged_cache,
                    block_table,
                    full_attn_idx,
                    layer_lora,
                )
                .with_context(|| format!("transformer block {i} (full attention, paged)"))?;
                full_attn_idx += 1;
            }
            GpuAttentionWeights::Linear(lin_weights) => {
                let state = linear_state.as_mut()
                    .ok_or_else(|| anyhow::anyhow!("linear attention state required for GDN layers (layer {i})"))?;
                let normed = rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?;
                let attn_out = gated_deltanet_forward(
                    &normed,
                    lin_weights,
                    config,
                    &mut state.recurrent_states[linear_attn_idx],
                    &mut state.conv_states[linear_attn_idx],
                )
                .with_context(|| format!("gated deltanet layer {i} (linear attention, paged)"))?;
                hidden = (hidden + attn_out)?;
                let normed_post = rms_norm(&hidden, &layer.post_attention_layernorm, config.rms_norm_eps)?;
                let ffn_out = swiglu_ffn(
                    &normed_post,
                    &layer.mlp.gate_proj,
                    &layer.mlp.up_proj,
                    &layer.mlp.down_proj,
                    layer_lora,
                )?;
                hidden = (hidden + ffn_out)?;
                linear_attn_idx += 1;
            }
        }
    }

    // 3. Final RMSNorm
    hidden = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;

    // 4. LM head projection (weight-tied)
    let logits = hidden.broadcast_matmul(&weights.embed_tokens.t()?)?;

    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() -> Result<()> {
        let device = Device::Cpu;
        // vocab_size=5, hidden_size=3
        let embed_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, // token 0
            0.4, 0.5, 0.6, // token 1
            0.7, 0.8, 0.9, // token 2
            1.0, 1.1, 1.2, // token 3
            1.3, 1.4, 1.5, // token 4
        ];
        let embed = Tensor::new(embed_data, &device)?.reshape((5, 3))?;

        let result = embedding_lookup(&[2, 0, 4], &embed)?;
        assert_eq!(result.dims(), &[3, 3]); // [seq_len=3, hidden_size=3]

        let vals = result.to_vec2::<f32>()?;
        // Token 2
        assert!((vals[0][0] - 0.7).abs() < 1e-6);
        assert!((vals[0][1] - 0.8).abs() < 1e-6);
        assert!((vals[0][2] - 0.9).abs() < 1e-6);
        // Token 0
        assert!((vals[1][0] - 0.1).abs() < 1e-6);
        // Token 4
        assert!((vals[2][0] - 1.3).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_rms_norm_known_values() -> Result<()> {
        let device = Device::Cpu;
        // x = [1, 2, 3], weight = [0, 0, 0], eps = 0
        // Effective weight = 1 + w = [1, 1, 1]
        // RMS = sqrt(mean([1,4,9])) = sqrt(14/3) ≈ 2.1602
        // normed = [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
        let x = Tensor::new(&[1.0_f32, 2.0, 3.0], &device)?.unsqueeze(0)?; // [1, 3]
        let w = Tensor::new(&[0.0_f32, 0.0, 0.0], &device)?;

        let result = rms_norm(&x, &w, 1e-8)?;
        let vals = result.to_vec2::<f32>()?;

        let rms = (14.0_f64 / 3.0).sqrt();
        assert!((vals[0][0] as f64 - 1.0 / rms).abs() < 1e-4);
        assert!((vals[0][1] as f64 - 2.0 / rms).abs() < 1e-4);
        assert!((vals[0][2] as f64 - 3.0 / rms).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_rms_norm_with_weight() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[2.0_f32, 2.0, 2.0], &device)?.unsqueeze(0)?;
        let w = Tensor::new(&[0.5_f32, 1.0, 2.0], &device)?;

        let result = rms_norm(&x, &w, 1e-8)?;
        let vals = result.to_vec2::<f32>()?;

        // RMS of [2,2,2] = 2.0, so normed = [1,1,1]
        // Effective weight = 1 + w = [1.5, 2.0, 3.0]
        // After weight: [1.5, 2.0, 3.0]
        assert!((vals[0][0] - 1.5).abs() < 1e-4);
        assert!((vals[0][1] - 2.0).abs() < 1e-4);
        assert!((vals[0][2] - 3.0).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_rope_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 8;

        let q = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::randn(
            0.0_f32,
            1.0,
            (batch, seq_len, num_kv_heads, head_dim),
            &device,
        )?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, head_dim, 10_000.0)?;

        assert_eq!(rq.dims(), &[batch, seq_len, num_heads, head_dim]);
        assert_eq!(rk.dims(), &[batch, seq_len, num_kv_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_rope_position_zero_is_identity() -> Result<()> {
        let device = Device::Cpu;
        // At position 0, cos=1 and sin=0, so rotation should be identity
        let head_dim = 4;
        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let q = Tensor::new(q_data.as_slice(), &device)?.reshape((1, 1, 1, head_dim))?;
        let k = q.clone();

        let (rq, _rk) = rotary_embedding(&q, &k, &[0], head_dim, head_dim, 10_000.0)?;
        let orig = q.flatten_all()?.to_vec1::<f32>()?;
        let rotated = rq.flatten_all()?.to_vec1::<f32>()?;

        for i in 0..head_dim {
            assert!(
                (orig[i] - rotated[i]).abs() < 1e-5,
                "Position 0 should be identity, dim {i}: orig={} rotated={}",
                orig[i],
                rotated[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_rope_different_positions_differ() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let q = Tensor::ones((1, 2, 1, head_dim), DType::F32, &device)?;
        let k = q.clone();

        let (rq, _) = rotary_embedding(&q, &k, &[0, 100], head_dim, head_dim, 10_000.0)?;
        // rq shape: [1, 2, 1, 8] — extract pos 0 and pos 100
        let pos0 = rq.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let pos100 = rq.narrow(1, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;

        let diff: f32 = pos0.iter().zip(&pos100).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.01,
            "Different positions should produce different embeddings"
        );

        Ok(())
    }

    #[test]
    fn test_partial_rope_passthrough_dims_unchanged() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let rotary_dim = 4; // only rotate first 4 dims, last 4 pass through
        let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let q = Tensor::new(q_data.as_slice(), &device)?.reshape((1, 1, 1, head_dim))?;
        let k = q.clone();

        // Position 100 — the rotary dims should change, passthrough dims should not
        let (rq, _) = rotary_embedding(&q, &k, &[100], head_dim, rotary_dim, 10_000.0)?;
        let orig = q.flatten_all()?.to_vec1::<f32>()?;
        let rotated = rq.flatten_all()?.to_vec1::<f32>()?;

        // First rotary_dim dims should be different at non-zero position
        let rotary_diff: f32 = (0..rotary_dim).map(|i| (orig[i] - rotated[i]).abs()).sum();
        assert!(rotary_diff > 0.01, "Rotary dims should change at position 100");

        // Passthrough dims (rotary_dim..head_dim) must be identical
        for i in rotary_dim..head_dim {
            assert!(
                (orig[i] - rotated[i]).abs() < 1e-6,
                "Passthrough dim {i} should be unchanged: orig={} rotated={}",
                orig[i],
                rotated[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_partial_rope_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 16;
        let rotary_dim = 4; // partial rotation

        let q = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_heads, head_dim), &device)?;
        let k = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, num_kv_heads, head_dim), &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, rotary_dim, 10_000.0)?;

        assert_eq!(rq.dims(), &[batch, seq_len, num_heads, head_dim]);
        assert_eq!(rk.dims(), &[batch, seq_len, num_kv_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_swiglu_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 3;
        let hidden = 4;
        let intermediate = 8;

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let gate = Tensor::randn(0.0_f32, 0.1, (intermediate, hidden), &device)?;
        let up = Tensor::randn(0.0_f32, 0.1, (intermediate, hidden), &device)?;
        let down = Tensor::randn(0.0_f32, 0.1, (hidden, intermediate), &device)?;

        let result = swiglu_ffn(&x, &gate, &up, &down, None)?;
        assert_eq!(result.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_swiglu_zero_gate_gives_zero() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 4;
        let intermediate = 8;

        let x = Tensor::ones((1, 1, hidden), DType::F32, &device)?;
        // Gate weights all zero -> silu(0) = 0 -> output is zero regardless of up/down
        let gate = Tensor::zeros((intermediate, hidden), DType::F32, &device)?;
        let up = Tensor::ones((intermediate, hidden), DType::F32, &device)?;
        let down = Tensor::ones((hidden, intermediate), DType::F32, &device)?;

        let result = swiglu_ffn(&x, &gate, &up, &down, None)?;
        let vals = result.to_vec3::<f32>()?;

        for v in &vals[0][0] {
            assert!(
                v.abs() < 1e-6,
                "SwiGLU with zero gate should produce zero, got {v}"
            );
        }

        Ok(())
    }

    /// Create a minimal config for tests (no output gate, simple dims).
    fn make_test_config(num_heads: usize, num_kv_heads: usize, head_dim: usize, hidden: usize) -> kiln_core::config::ModelConfig {
        kiln_core::config::ModelConfig {
            hidden_size: hidden,
            num_layers: 4,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: hidden * 2,
            vocab_size: 256,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0, // tests use full rotation by default
        }
    }

    fn make_test_attn_weights(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        device: &Device,
    ) -> Result<GpuFullAttentionWeights> {
        Ok(GpuFullAttentionWeights {
            q_proj: Tensor::randn(0.0_f32, 0.02, (num_heads * head_dim, hidden), device)?,
            k_proj: Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, hidden), device)?,
            v_proj: Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, hidden), device)?,
            o_proj: Tensor::randn(0.0_f32, 0.02, (hidden, num_heads * head_dim), device)?,
            q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
            k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
        })
    }

    #[test]
    fn test_gqa_attention_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim; // 32

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let out = gqa_attention(&x, &attn, &positions, num_heads, num_kv_heads, head_dim, head_dim, 10_000.0, 1e-6, None, 0, false, None)?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_gqa_head_expansion() -> Result<()> {
        // Verify GQA works: 4 Q heads, 2 KV heads (ratio=2)
        let device = Device::Cpu;
        let batch = 2;
        let seq_len = 3;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim;

        let x = Tensor::randn(0.0_f32, 0.5, (batch, seq_len, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let out = gqa_attention(&x, &attn, &positions, num_heads, num_kv_heads, head_dim, head_dim, 10_000.0, 1e-6, None, 0, false, None)?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        // Output should be finite and not all zeros
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(vals.iter().all(|v| v.is_finite()), "output should be finite");
        let sum: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(sum > 1e-6, "output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_gqa_single_token() -> Result<()> {
        // Single token should work (no causal masking needed)
        let device = Device::Cpu;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let hidden = num_heads * head_dim;

        let x = Tensor::randn(0.0_f32, 1.0, (1, 1, hidden), &device)?;
        let attn = make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?;

        let out = gqa_attention(&x, &attn, &[0], num_heads, num_kv_heads, head_dim, head_dim, 10_000.0, 1e-6, None, 0, false, None)?;
        assert_eq!(out.dims(), &[1, 1, hidden]);

        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        // A 3x3 score matrix
        let scores = Tensor::ones((1, 1, 3, 3), DType::F32, &device)?;
        let masked = apply_causal_mask(&scores, 3)?;
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        // Row 0: [1, -inf, -inf]
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
        assert!(vals[2].is_infinite() && vals[2] < 0.0);
        // Row 1: [1, 1, -inf]
        assert!((vals[3] - 1.0).abs() < 1e-6);
        assert!((vals[4] - 1.0).abs() < 1e-6);
        assert!(vals[5].is_infinite() && vals[5] < 0.0);
        // Row 2: [1, 1, 1]
        assert!((vals[6] - 1.0).abs() < 1e-6);
        assert!((vals[7] - 1.0).abs() < 1e-6);
        assert!((vals[8] - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_transformer_block_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let seq_len = 4;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let hidden = num_heads * head_dim;
        let intermediate = hidden * 2;

        let x = Tensor::randn(0.0_f32, 1.0, (batch, seq_len, hidden), &device)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(
                make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?,
            ),
            mlp: GpuFfnWeights {
                gate_proj: Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?,
                up_proj: Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?,
                down_proj: Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), &device)?,
            },
        };

        let cfg = make_test_config(num_heads, num_kv_heads, head_dim, hidden);
        let out = transformer_block(&x, &layer, &cfg, &positions, num_heads, num_kv_heads, head_dim, head_dim, 10_000.0, 1e-6, None, 0, None)?;
        assert_eq!(out.dims(), &[batch, seq_len, hidden]);

        Ok(())
    }

    #[test]
    fn test_transformer_block_residual_connections() -> Result<()> {
        // With residual connections, output should differ from zero even with small weights
        let device = Device::Cpu;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let hidden = num_heads * head_dim;
        let intermediate = hidden * 2;

        // Input with known non-zero values
        let x = Tensor::ones((1, 2, hidden), DType::F32, &device)?;
        let positions = vec![0u32, 1];

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(
                make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?,
            ),
            mlp: GpuFfnWeights {
                gate_proj: Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?,
                up_proj: Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?,
                down_proj: Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), &device)?,
            },
        };

        let cfg = make_test_config(num_heads, num_kv_heads, head_dim, hidden);
        let out = transformer_block(&x, &layer, &cfg, &positions, num_heads, num_kv_heads, head_dim, head_dim, 10_000.0, 1e-6, None, 0, None)?;

        // Output should not be zero (residual adds input through)
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        let sum: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.1, "residual connections should keep output non-zero, got sum={sum}");
        assert!(vals.iter().all(|v| v.is_finite()), "output should be finite");

        Ok(())
    }

    #[test]
    fn test_transformer_block_rejects_linear_attention() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 8;

        let layer = GpuLayerWeights {
            input_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::zeros(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                in_proj_qkv: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_z: Tensor::zeros((1, 1), DType::F32, &device)?,
                out_proj: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_a: Tensor::zeros((1, 1), DType::F32, &device)?,
                in_proj_b: Tensor::zeros((1, 1), DType::F32, &device)?,
                conv1d: Tensor::zeros((1, 1), DType::F32, &device)?,
                norm: Tensor::zeros((1,), DType::F32, &device)?,
                a_log: Tensor::zeros((1,), DType::F32, &device)?,
                dt_bias: Tensor::zeros((1,), DType::F32, &device)?,
            }),
            mlp: GpuFfnWeights {
                gate_proj: Tensor::zeros((1, hidden), DType::F32, &device)?,
                up_proj: Tensor::zeros((1, hidden), DType::F32, &device)?,
                down_proj: Tensor::zeros((hidden, 1), DType::F32, &device)?,
            },
        };

        let x = Tensor::ones((1, 1, hidden), DType::F32, &device)?;
        let cfg = make_test_config(2, 1, 4, 8);
        let result = transformer_block(&x, &layer, &cfg, &[0], 2, 1, 4, 4, 10_000.0, 1e-6, None, 0, None);
        assert!(result.is_err(), "should reject linear attention layers");

        Ok(())
    }

    #[test]
    fn test_weight_to_tensor_f32() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let wt = WeightTensor {
            data: bytes,
            shape: vec![2, 3],
            dtype: TensorDType::F32,
        };

        let t = weight_to_tensor(&wt, &device)?;
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);

        let vals = t.to_vec2::<f32>()?;
        assert!((vals[0][0] - 1.0).abs() < 1e-6);
        assert!((vals[1][2] - 6.0).abs() < 1e-6);

        Ok(())
    }

    /// Helper: build tiny GpuWeights for testing model_forward shape propagation.
    /// Uses full-attention layers only (no linear attention) with small dimensions.
    fn make_tiny_gpu_weights(
        device: &Device,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
    ) -> Result<GpuWeights> {
        let randn = |shape: &[usize]| -> Result<Tensor> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();
            Ok(Tensor::new(data, device)?.reshape(shape)?)
        };

        let embed_tokens = randn(&[vocab_size, hidden_size])?;
        let final_norm = Tensor::zeros(hidden_size, DType::F32, device)?;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                post_attention_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj: randn(&[num_heads * head_dim, hidden_size])?,
                    k_proj: randn(&[num_kv_heads * head_dim, hidden_size])?,
                    v_proj: randn(&[num_kv_heads * head_dim, hidden_size])?,
                    o_proj: randn(&[hidden_size, num_heads * head_dim])?,
                    q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                }),
                mlp: GpuFfnWeights {
                    gate_proj: randn(&[intermediate_size, hidden_size])?,
                    up_proj: randn(&[intermediate_size, hidden_size])?,
                    down_proj: randn(&[hidden_size, intermediate_size])?,
                },
            });
        }

        Ok(GpuWeights {
            embed_tokens,
            layers,
            final_norm,
        })
    }

    #[test]
    fn test_model_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 2;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: num_layers,
            full_attention_interval: 1, // every layer is full attention
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let token_ids: Vec<u32> = vec![1, 5, 3, 10];
        let logits = model_forward(&token_ids, &weights, &config, None, None, None)?;

        // Expected shape: [1, seq_len, vocab_size]
        assert_eq!(logits.dims(), &[1, 4, vocab_size]);

        Ok(())
    }

    #[test]
    fn test_model_forward_single_token() -> Result<()> {
        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            1, // single layer
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let logits = model_forward(&[7], &weights, &config, None, None, None)?;
        assert_eq!(logits.dims(), &[1, 1, vocab_size]);

        // Logits should be finite
        let vals = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(vals.iter().all(|v| v.is_finite()), "all logits should be finite");

        Ok(())
    }

    #[test]
    fn test_model_forward_kv_cache_equivalence() -> Result<()> {
        // Verify that model_forward with KV cache produces the same last-position
        // logits as without KV cache, for a multi-token sequence processed
        // incrementally (prefill + decode steps).
        use crate::kv_cache::KvCache;

        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;
        let num_layers = 2;

        let weights = make_tiny_gpu_weights(
            &device,
            vocab_size,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_layers,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: num_layers,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let tokens: Vec<u32> = vec![1, 5, 3, 10, 7];

        // Reference: full forward pass without KV cache
        let logits_ref = model_forward(&tokens, &weights, &config, None, None, None)?;
        // Extract last position logits: [1, 5, vocab] -> last position
        let last_ref = logits_ref.narrow(1, tokens.len() - 1, 1)?; // [1, 1, vocab]
        let last_ref_vals = last_ref.flatten_all()?.to_vec1::<f32>()?;

        // With KV cache: prefill first 4 tokens, then decode the 5th
        let mut kv_cache = KvCache::new(
            num_layers, num_kv_heads, head_dim, 32, DType::F32, &device,
        )?;

        // Prefill
        let _prefill_logits = model_forward(&tokens[..4], &weights, &config, Some(&mut kv_cache), None, None)?;
        kv_cache.advance(4);
        assert_eq!(kv_cache.seq_len(), 4);

        // Decode the 5th token
        let decode_logits = model_forward(&tokens[4..], &weights, &config, Some(&mut kv_cache), None, None)?;
        kv_cache.advance(1);
        assert_eq!(kv_cache.seq_len(), 5);

        let last_cached_vals = decode_logits.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;

        // Compare: should be identical (within floating point tolerance)
        assert_eq!(last_ref_vals.len(), last_cached_vals.len());
        for (i, (r, c)) in last_ref_vals.iter().zip(&last_cached_vals).enumerate() {
            assert!(
                (r - c).abs() < 1e-4,
                "logit {i} differs: ref={r}, cached={c}, diff={}",
                (r - c).abs()
            );
        }

        Ok(())
    }

    #[test]
    fn test_model_forward_kv_cache_token_by_token() -> Result<()> {
        // Verify that processing tokens one-by-one with KV cache matches
        // processing all at once without cache.
        use crate::kv_cache::KvCache;

        let device = Device::Cpu;
        let vocab_size = 16;
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate_size = 16;

        let weights = make_tiny_gpu_weights(
            &device, vocab_size, hidden_size, num_heads, num_kv_heads,
            head_dim, intermediate_size, 1,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers: 1,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: num_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let tokens: Vec<u32> = vec![3, 7, 1];

        // Reference
        let logits_ref = model_forward(&tokens, &weights, &config, None, None, None)?;
        let last_ref = logits_ref.narrow(1, 2, 1)?.flatten_all()?.to_vec1::<f32>()?;

        // KV cache: process token by token
        let mut kv_cache = KvCache::new(1, num_kv_heads, head_dim, 16, DType::F32, &device)?;

        // Token 0
        let _ = model_forward(&[3], &weights, &config, Some(&mut kv_cache), None, None)?;
        kv_cache.advance(1);

        // Token 1
        let _ = model_forward(&[7], &weights, &config, Some(&mut kv_cache), None, None)?;
        kv_cache.advance(1);

        // Token 2
        let logits_cached = model_forward(&[1], &weights, &config, Some(&mut kv_cache), None, None)?;
        kv_cache.advance(1);

        let last_cached = logits_cached.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;

        for (i, (r, c)) in last_ref.iter().zip(&last_cached).enumerate() {
            assert!(
                (r - c).abs() < 1e-4,
                "logit {i} differs: ref={r}, cached={c}",
            );
        }

        Ok(())
    }

    /// Helper: build tiny GpuWeights with a mix of full and linear attention layers.
    fn make_hybrid_gpu_weights(
        device: &Device,
        vocab_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
        full_attention_interval: usize,
    ) -> Result<GpuWeights> {
        let randn = |shape: &[usize]| -> Result<Tensor> {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01).sin()) * 0.1).collect();
            Ok(Tensor::new(data, device)?.reshape(shape)?)
        };

        let embed_tokens = randn(&[vocab_size, hidden_size])?;
        let final_norm = Tensor::zeros(hidden_size, DType::F32, device)?;

        // For linear attention: nk heads with key_head_dim, nv heads with value_head_dim
        // Use same dims as full attention for simplicity
        let nk = num_kv_heads;
        let nv = num_heads;
        let dk = head_dim;
        let dv = head_dim;
        let qkv_dim = nk * dk + nk * dk + nv * dv; // Q + K + V fused
        let conv_kernel = 4;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let is_full = (i + 1) % full_attention_interval == 0;
            let attention = if is_full {
                GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj: randn(&[num_heads * head_dim, hidden_size])?,
                    k_proj: randn(&[num_kv_heads * head_dim, hidden_size])?,
                    v_proj: randn(&[num_kv_heads * head_dim, hidden_size])?,
                    o_proj: randn(&[hidden_size, num_heads * head_dim])?,
                    q_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                    k_norm: Tensor::zeros(head_dim, DType::F32, device)?,
                })
            } else {
                GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                    in_proj_qkv: randn(&[qkv_dim, hidden_size])?,
                    in_proj_z: randn(&[nv * dv, hidden_size])?,
                    out_proj: randn(&[hidden_size, nv * dv])?,
                    in_proj_a: randn(&[nv, hidden_size])?,
                    in_proj_b: randn(&[nv, hidden_size])?,
                    conv1d: randn(&[qkv_dim, 1, conv_kernel])?,
                    norm: Tensor::ones(dk, DType::F32, device)?,
                    a_log: Tensor::zeros(nv, DType::F32, device)?,
                    dt_bias: Tensor::zeros(nv, DType::F32, device)?,
                })
            };

            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                post_attention_layernorm: Tensor::zeros(hidden_size, DType::F32, device)?,
                attention,
                mlp: GpuFfnWeights {
                    gate_proj: randn(&[intermediate_size, hidden_size])?,
                    up_proj: randn(&[intermediate_size, hidden_size])?,
                    down_proj: randn(&[hidden_size, intermediate_size])?,
                },
            });
        }

        Ok(GpuWeights {
            embed_tokens,
            layers,
            final_norm,
        })
    }

    #[test]
    fn test_model_forward_hybrid_layers() -> Result<()> {
        // Test model_forward with a mix of full and linear (GDN) attention layers
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 4;
        let full_attention_interval = 4; // layer 3 is full, layers 0,1,2 are linear

        let weights = make_hybrid_gpu_weights(
            &device, vocab_size, hidden_size, num_heads, num_kv_heads,
            head_dim, intermediate_size, num_layers, full_attention_interval,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let mut linear_state = LinearAttentionState::new(&config, &device)?;

        // Prefill with multiple tokens
        let token_ids: Vec<u32> = vec![1, 5, 3, 10];
        let logits = model_forward(&token_ids, &weights, &config, None, Some(&mut linear_state), None)?;
        assert_eq!(logits.dims(), &[1, 4, vocab_size]);

        // All values should be finite (no NaN/Inf)
        let flat = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(flat.iter().all(|v| v.is_finite()), "logits contain non-finite values");

        Ok(())
    }

    #[test]
    fn test_model_forward_hybrid_decode() -> Result<()> {
        // Test prefill + decode with linear attention state persistence
        let device = Device::Cpu;
        let vocab_size = 32;
        let hidden_size = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let intermediate_size = 32;
        let num_layers = 4;
        let full_attention_interval = 4;

        let weights = make_hybrid_gpu_weights(
            &device, vocab_size, hidden_size, num_heads, num_kv_heads,
            head_dim, intermediate_size, num_layers, full_attention_interval,
        )?;

        let config = kiln_core::config::ModelConfig {
            hidden_size,
            num_layers,
            num_attention_heads: num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval,
            attn_output_gate: false,
            linear_num_key_heads: num_kv_heads,
            linear_key_head_dim: head_dim,
            linear_num_value_heads: num_heads,
            linear_value_head_dim: head_dim,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let mut kv_cache = KvCache::new(1, num_kv_heads, head_dim, 32, DType::F32, &device)?;
        let mut linear_state = LinearAttentionState::new(&config, &device)?;

        // Prefill
        let prefill_logits = model_forward(&[1, 5, 3], &weights, &config, Some(&mut kv_cache), Some(&mut linear_state), None)?;
        kv_cache.advance(3);
        assert_eq!(prefill_logits.dims(), &[1, 3, vocab_size]);

        // Decode: single token should work with persisted linear state
        let decode_logits = model_forward(&[10], &weights, &config, Some(&mut kv_cache), Some(&mut linear_state), None)?;
        kv_cache.advance(1);
        assert_eq!(decode_logits.dims(), &[1, 1, vocab_size]);

        // Both should produce finite values
        let flat = decode_logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(flat.iter().all(|v| v.is_finite()), "decode logits contain non-finite values");

        Ok(())
    }

    #[test]
    fn test_linear_attention_state_new() -> Result<()> {
        let device = Device::Cpu;
        let config = kiln_core::config::ModelConfig {
            hidden_size: 16,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            vocab_size: 32,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: false,
            linear_num_key_heads: 2,
            linear_key_head_dim: 4,
            linear_num_value_heads: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        };

        let state = LinearAttentionState::new(&config, &device)?;
        // 3 linear layers (layers 0,1,2; layer 3 is full)
        assert_eq!(state.recurrent_states.len(), 3);
        assert_eq!(state.conv_states.len(), 3);
        // Recurrent state shape: [1, nv, dk, dv]
        assert_eq!(state.recurrent_states[0].dims(), &[1, 4, 4, 4]);
        // Conv state shape: [1, qkv_dim, kernel_size-1] where qkv_dim = 2*(nk*dk) + nv*dv = 2*8+16=32
        let qkv_dim = 2 * (2 * 4) + 4 * 4; // 32
        assert_eq!(state.conv_states[0].dims(), &[1, qkv_dim, 3]);

        Ok(())
    }

    #[test]
    fn test_causal_mask_with_offset() -> Result<()> {
        let device = Device::Cpu;
        // Simulate decode: 1 new query, 4 total KV (3 cached + 1 new)
        let scores = Tensor::ones((1, 1, 1, 4), DType::F32, &device)?;
        let masked = apply_causal_mask_with_offset(&scores, 1, 4, 3)?;
        // Single query should attend to all 4 positions (no masking for q_len=1)
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        assert!(vals.iter().all(|v| (*v - 1.0).abs() < 1e-6),
            "single query token should attend to all KV positions");

        // Simulate prefill with offset: 2 new queries, 5 total KV (3 cached + 2 new)
        let scores = Tensor::ones((1, 1, 2, 5), DType::F32, &device)?;
        let masked = apply_causal_mask_with_offset(&scores, 2, 5, 3)?;
        let vals = masked.flatten_all()?.to_vec1::<f32>()?;
        // Row 0 (abs pos 3): can attend to positions 0..4 (first 4), mask position 4
        assert!((vals[0] - 1.0).abs() < 1e-6); // pos 0: ok
        assert!((vals[1] - 1.0).abs() < 1e-6); // pos 1: ok
        assert!((vals[2] - 1.0).abs() < 1e-6); // pos 2: ok
        assert!((vals[3] - 1.0).abs() < 1e-6); // pos 3 (self): ok
        assert!(vals[4].is_infinite() && vals[4] < 0.0); // pos 4: masked
        // Row 1 (abs pos 4): can attend to all 5 positions
        assert!(vals[5..10].iter().all(|v| (*v - 1.0).abs() < 1e-6));

        Ok(())
    }
}
