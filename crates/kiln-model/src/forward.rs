//! Candle-based forward pass layers for Qwen3.5-4B.
//!
//! Implements the foundational compute primitives: embedding lookup, RMSNorm,
//! RoPE (rotary position embeddings), and SwiGLU FFN. These operate on candle
//! `Tensor` objects and are composed into the full transformer forward pass.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::weights::{ModelWeights, TensorDType, WeightTensor};

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
    let rms = (variance + eps)?.sqrt()?;
    let rms_inv = rms.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    // Cast back to original dtype, then scale by weight
    let normed = normed.to_dtype(x.dtype())?;
    let out = normed.broadcast_mul(weight)?;
    Ok(out)
}

/// Apply Rotary Position Embeddings (RoPE) to query and key tensors.
///
/// `q`: [batch, seq_len, num_heads, head_dim]
/// `k`: [batch, seq_len, num_kv_heads, head_dim]
/// `positions`: position index for each token in the sequence (length = seq_len)
/// `head_dim`: dimension of each attention head
/// `rope_theta`: base frequency (10_000_000.0 for Qwen3.5-4B)
///
/// Returns: (rotated_q, rotated_k) with same shapes.
pub fn rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    positions: &[u32],
    head_dim: usize,
    rope_theta: f64,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();
    let half_dim = head_dim / 2;

    // Compute frequency table: freq_i = 1.0 / (theta ^ (2i / head_dim)) for i in 0..half_dim
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?; // [half_dim]

    // Position tensor
    let pos_f32: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let pos = Tensor::new(pos_f32.as_slice(), device)?.unsqueeze(1)?; // [seq_len, 1]

    // Outer product: [seq_len, half_dim]
    let freqs = pos.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let cos = freqs.cos()?; // [seq_len, half_dim]
    let sin = freqs.sin()?; // [seq_len, half_dim]

    let rotated_q = apply_rope(q, &cos, &sin, head_dim)?;
    let rotated_k = apply_rope(k, &cos, &sin, head_dim)?;

    Ok((rotated_q, rotated_k))
}

/// Apply the rotation to a single tensor.
/// `x`: [batch, seq_len, num_heads, head_dim]
/// `cos`, `sin`: [seq_len, half_dim]
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, head_dim: usize) -> Result<Tensor> {
    let half = head_dim / 2;
    let x_dtype = x.dtype();

    // Work in f32 for precision
    let x = x.to_dtype(DType::F32)?;

    // Split head_dim into two halves along the last dimension
    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?; // [..., :half]
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?; // [..., half:]

    // cos/sin are [seq_len, half_dim], need to broadcast to [batch, seq_len, num_heads, half_dim]
    // Reshape to [1, seq_len, 1, half_dim]
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;

    // Standard RoPE rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    let out = Tensor::cat(&[&r1, &r2], candle_core::D::Minus1)?;
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
) -> Result<Tensor> {
    // x @ gate_proj^T -> [batch, seq_len, intermediate_size]
    let gate = x.broadcast_matmul(&gate_proj.t()?)?;
    // SiLU activation: x * sigmoid(x)
    let gate = candle_nn::ops::silu(&gate)?;
    // x @ up_proj^T -> [batch, seq_len, intermediate_size]
    let up = x.broadcast_matmul(&up_proj.t()?)?;
    // Element-wise multiply
    let hidden = (gate * up)?;
    // hidden @ down_proj^T -> [batch, seq_len, hidden_size]
    let out = hidden.broadcast_matmul(&down_proj.t()?)?;
    Ok(out)
}

/// Grouped-Query Attention (GQA).
///
/// Computes scaled dot-product attention with fewer KV heads than Q heads.
/// Each group of `num_heads / num_kv_heads` query heads shares one KV head.
///
/// `x`: [batch, seq_len, hidden_size]
/// `attn_weights`: Q/K/V/O projection weights plus per-head RMSNorm weights
/// `positions`: position indices for RoPE (length = seq_len)
/// `num_heads`: number of query attention heads
/// `num_kv_heads`: number of key/value attention heads
/// `head_dim`: dimension per head
/// `rope_theta`: RoPE base frequency
/// `rms_norm_eps`: epsilon for Q/K head norms
///
/// Returns: [batch, seq_len, hidden_size]
pub fn gqa_attention(
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) -> Result<Tensor> {
    let (_batch, seq_len, _hidden) = x.dims3()?;

    // Project to Q, K, V
    // q_proj: [num_heads * head_dim, hidden_size], so x @ q_proj^T -> [batch, seq_len, num_heads * head_dim]
    let q = x.broadcast_matmul(&attn_weights.q_proj.t()?)?;
    let k = x.broadcast_matmul(&attn_weights.k_proj.t()?)?;
    let v = x.broadcast_matmul(&attn_weights.v_proj.t()?)?;

    // Reshape to [batch, seq_len, num_heads, head_dim]
    let q = q.reshape(((), seq_len, num_heads, head_dim))?;
    let k = k.reshape(((), seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape(((), seq_len, num_kv_heads, head_dim))?;

    // Apply per-head RMSNorm to Q and K (Qwen3.5 uses QK-norm)
    // q_norm/k_norm are [head_dim] — broadcast over [batch, seq_len, num_heads, head_dim]
    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;

    // Apply RoPE
    let (q, k) = rotary_embedding(&q, &k, positions, head_dim, rope_theta)?;

    // Transpose to [batch, heads, seq_len, head_dim] for attention
    let q = q.transpose(1, 2)?.contiguous()?; // [batch, num_heads, seq_len, head_dim]
    let k = k.transpose(1, 2)?.contiguous()?; // [batch, num_kv_heads, seq_len, head_dim]
    let v = v.transpose(1, 2)?.contiguous()?; // [batch, num_kv_heads, seq_len, head_dim]

    // GQA head expansion: repeat K/V to match Q head count
    let gqa_ratio = num_heads / num_kv_heads;
    let batch = k.dim(0)?;
    let (k, v) = if gqa_ratio > 1 {
        // Expand [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        // by repeating each KV head `gqa_ratio` times
        let k = k
            .unsqueeze(2)? // [batch, num_kv_heads, 1, seq, head_dim]
            .expand(&[batch, num_kv_heads, gqa_ratio, seq_len, head_dim])? // broadcast
            .contiguous()?
            .reshape((batch, num_heads, seq_len, head_dim))?;
        let v = v
            .unsqueeze(2)?
            .expand(&[batch, num_kv_heads, gqa_ratio, seq_len, head_dim])?
            .contiguous()?
            .reshape((batch, num_heads, seq_len, head_dim))?;
        (k, v)
    } else {
        (k.contiguous()?, v.contiguous()?)
    };

    // Scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    let scale = (head_dim as f64).sqrt();
    let attn_scores = q.broadcast_matmul(&k.t()?)?; // [batch, num_heads, seq, seq]
    let attn_scores = (attn_scores / scale)?;

    // Apply causal mask: mask out future positions
    let attn_scores = apply_causal_mask(&attn_scores, seq_len)?;

    let attn_weights_softmax = candle_nn::ops::softmax_last_dim(&attn_scores)?;
    let attn_output = attn_weights_softmax.broadcast_matmul(&v)?; // [batch, num_heads, seq, head_dim]

    // Transpose back: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden]
    let attn_output = attn_output
        .transpose(1, 2)?
        .contiguous()?
        .reshape(((), seq_len, num_heads * head_dim))?;

    // Output projection
    let out = attn_output.broadcast_matmul(&attn_weights.o_proj.t()?)?;
    Ok(out)
}

/// Apply a causal (lower-triangular) mask to attention scores.
/// Sets future positions to -inf so softmax zeroes them out.
fn apply_causal_mask(scores: &Tensor, seq_len: usize) -> Result<Tensor> {
    if seq_len <= 1 {
        return Ok(scores.clone());
    }
    let device = scores.device();
    // Build a [seq_len, seq_len] mask: 0 for allowed, -inf for masked
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    let mask = Tensor::new(mask, device)?.reshape((1, 1, seq_len, seq_len))?;
    let mask = mask.to_dtype(scores.dtype())?;
    let out = scores.broadcast_add(&mask)?;
    Ok(out)
}

/// Single transformer block: norm -> attention -> residual -> norm -> FFN -> residual.
///
/// `x`: [batch, seq_len, hidden_size]
/// `layer`: weights for this transformer layer
/// `positions`: position indices for RoPE
/// `num_heads`: number of query attention heads
/// `num_kv_heads`: number of key/value attention heads
/// `head_dim`: dimension per head
/// `rope_theta`: RoPE base frequency
/// `rms_norm_eps`: epsilon for RMSNorm
///
/// Returns: [batch, seq_len, hidden_size]
pub fn transformer_block(
    x: &Tensor,
    layer: &GpuLayerWeights,
    positions: &[u32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
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
        rope_theta,
        rms_norm_eps,
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
///
/// Returns logits tensor with shape [1, seq_len, vocab_size].
///
/// Notes:
/// - Qwen3.5-4B uses weight tying: the LM head reuses `embed_tokens` transposed.
/// - Linear attention (Gated DeltaNet) layers are not yet implemented and will
///   be skipped with an identity pass-through.
pub fn model_forward(
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
) -> Result<Tensor> {
    let seq_len = token_ids.len();

    // 1. Embedding lookup: [seq_len, hidden_size]
    let mut hidden = embedding_lookup(token_ids, &weights.embed_tokens)?;

    // Add batch dimension: [1, seq_len, hidden_size]
    hidden = hidden.unsqueeze(0)?;

    // Position indices for RoPE
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    // 2. Loop through all transformer layers
    for (i, layer) in weights.layers.iter().enumerate() {
        match &layer.attention {
            GpuAttentionWeights::Full(_) => {
                hidden = transformer_block(
                    &hidden,
                    layer,
                    &positions,
                    config.num_attention_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.rope_theta,
                    config.rms_norm_eps,
                )
                .with_context(|| format!("transformer block {i} (full attention)"))?;
            }
            GpuAttentionWeights::Linear(_) => {
                // TODO: Implement Gated DeltaNet linear attention forward pass.
                // For now, skip with identity (pass-through) — the layer's FFN still runs
                // so we at least apply the MLP transformation.
                let normed = rms_norm(&hidden, &layer.input_layernorm, config.rms_norm_eps)?;
                // Skip attention (identity), just add zero residual
                // Then apply FFN
                let normed_post = rms_norm(&hidden, &layer.post_attention_layernorm, config.rms_norm_eps)?;
                let ffn_out = swiglu_ffn(
                    &normed_post,
                    &layer.mlp.gate_proj,
                    &layer.mlp.up_proj,
                    &layer.mlp.down_proj,
                )?;
                hidden = (hidden + ffn_out)?;
                // Suppress unused variable warning
                let _ = normed;
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
        // x = [1, 2, 3], weight = [1, 1, 1], eps = 0
        // RMS = sqrt(mean([1,4,9])) = sqrt(14/3) ≈ 2.1602
        // normed = [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
        let x = Tensor::new(&[1.0_f32, 2.0, 3.0], &device)?.unsqueeze(0)?; // [1, 3]
        let w = Tensor::new(&[1.0_f32, 1.0, 1.0], &device)?;

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
        // After weight: [0.5, 1.0, 2.0]
        assert!((vals[0][0] - 0.5).abs() < 1e-4);
        assert!((vals[0][1] - 1.0).abs() < 1e-4);
        assert!((vals[0][2] - 2.0).abs() < 1e-4);

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

        let (rq, rk) = rotary_embedding(&q, &k, &positions, head_dim, 10_000.0)?;

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

        let (rq, _rk) = rotary_embedding(&q, &k, &[0], head_dim, 10_000.0)?;
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

        let (rq, _) = rotary_embedding(&q, &k, &[0, 100], head_dim, 10_000.0)?;
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

        let result = swiglu_ffn(&x, &gate, &up, &down)?;
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

        let result = swiglu_ffn(&x, &gate, &up, &down)?;
        let vals = result.to_vec3::<f32>()?;

        for v in &vals[0][0] {
            assert!(
                v.abs() < 1e-6,
                "SwiGLU with zero gate should produce zero, got {v}"
            );
        }

        Ok(())
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
            q_norm: Tensor::ones(head_dim, DType::F32, device)?,
            k_norm: Tensor::ones(head_dim, DType::F32, device)?,
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

        let out = gqa_attention(&x, &attn, &positions, num_heads, num_kv_heads, head_dim, 10_000.0, 1e-6)?;
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

        let out = gqa_attention(&x, &attn, &positions, num_heads, num_kv_heads, head_dim, 10_000.0, 1e-6)?;
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

        let out = gqa_attention(&x, &attn, &[0], num_heads, num_kv_heads, head_dim, 10_000.0, 1e-6)?;
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
            input_layernorm: Tensor::ones(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::ones(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(
                make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?,
            ),
            mlp: GpuFfnWeights {
                gate_proj: Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?,
                up_proj: Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?,
                down_proj: Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), &device)?,
            },
        };

        let out = transformer_block(&x, &layer, &positions, num_heads, num_kv_heads, head_dim, 10_000.0, 1e-6)?;
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
            input_layernorm: Tensor::ones(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::ones(hidden, DType::F32, &device)?,
            attention: GpuAttentionWeights::Full(
                make_test_attn_weights(num_heads, num_kv_heads, head_dim, hidden, &device)?,
            ),
            mlp: GpuFfnWeights {
                gate_proj: Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?,
                up_proj: Tensor::randn(0.0_f32, 0.02, (intermediate, hidden), &device)?,
                down_proj: Tensor::randn(0.0_f32, 0.02, (hidden, intermediate), &device)?,
            },
        };

        let out = transformer_block(&x, &layer, &positions, num_heads, num_kv_heads, head_dim, 10_000.0, 1e-6)?;

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
            input_layernorm: Tensor::ones(hidden, DType::F32, &device)?,
            post_attention_layernorm: Tensor::ones(hidden, DType::F32, &device)?,
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
        let result = transformer_block(&x, &layer, &[0], 2, 1, 4, 10_000.0, 1e-6);
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
        let final_norm = Tensor::ones(hidden_size, DType::F32, device)?;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(GpuLayerWeights {
                input_layernorm: Tensor::ones(hidden_size, DType::F32, device)?,
                post_attention_layernorm: Tensor::ones(hidden_size, DType::F32, device)?,
                attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj: randn(&[num_heads * head_dim, hidden_size])?,
                    k_proj: randn(&[num_kv_heads * head_dim, hidden_size])?,
                    v_proj: randn(&[num_kv_heads * head_dim, hidden_size])?,
                    o_proj: randn(&[hidden_size, num_heads * head_dim])?,
                    q_norm: Tensor::ones(head_dim, DType::F32, device)?,
                    k_norm: Tensor::ones(head_dim, DType::F32, device)?,
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
        };

        let token_ids: Vec<u32> = vec![1, 5, 3, 10];
        let logits = model_forward(&token_ids, &weights, &config)?;

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
        };

        let logits = model_forward(&[7], &weights, &config)?;
        assert_eq!(logits.dims(), &[1, 1, vocab_size]);

        // Logits should be finite
        let vals = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(vals.iter().all(|v| v.is_finite()), "all logits should be finite");

        Ok(())
    }
}
