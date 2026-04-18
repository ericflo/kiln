use std::fmt;

/// Data type for stored tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F16,
    BF16,
    F32,
    /// Signed 32-bit integer. Used for Marlin W4A16 packed INT4 weights
    /// (8 nibbles per word).
    I32,
}

impl TensorDType {
    /// Bytes per element.
    pub fn size_bytes(self) -> usize {
        match self {
            TensorDType::F16 | TensorDType::BF16 => 2,
            TensorDType::F32 | TensorDType::I32 => 4,
        }
    }
}

impl fmt::Display for TensorDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDType::F16 => write!(f, "f16"),
            TensorDType::BF16 => write!(f, "bf16"),
            TensorDType::F32 => write!(f, "f32"),
            TensorDType::I32 => write!(f, "i32"),
        }
    }
}

/// A loaded tensor: owned raw bytes with shape and dtype metadata.
///
/// This is a CPU-side representation. The forward pass will convert these
/// to GPU tensors (candle Tensor or raw CUDA buffers).
pub struct WeightTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
}

impl WeightTensor {
    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Debug for WeightTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WeightTensor({:?}, {}, {} bytes)",
            self.shape,
            self.dtype,
            self.data.len()
        )
    }
}

/// Token embedding weights.
#[derive(Debug)]
pub struct EmbeddingWeights {
    /// [vocab_size, hidden_size]
    pub embed_tokens: WeightTensor,
}

/// Standard GQA self-attention weights (for full attention layers).
///
/// These layers use FlashAttention with KV cache and RoPE.
/// Every 4th layer in Qwen3.5-4B (indices 3, 7, 11, ..., 31).
#[derive(Debug)]
pub struct FullAttentionWeights {
    /// [num_heads * head_dim, hidden_size]
    pub q_proj: WeightTensor,
    /// [num_kv_heads * head_dim, hidden_size]
    pub k_proj: WeightTensor,
    /// [num_kv_heads * head_dim, hidden_size]
    pub v_proj: WeightTensor,
    /// [hidden_size, num_heads * head_dim]
    pub o_proj: WeightTensor,
    /// QK normalization weights (RMSNorm per head).
    /// [head_dim]
    pub q_norm: WeightTensor,
    /// [head_dim]
    pub k_norm: WeightTensor,
    /// Optional Marlin W4A16 repacked q_proj: `(b_packed I32 [k/16, n*16/8],
    /// scales F16 [k/groupsize, n])`. Populated only when `KILN_W4A16=1` is
    /// set at load time AND the checkpoint is GPTQ INT4 with group_size=128.
    /// When present, the forward pass uses `kiln-marlin-gemm::marlin_w4a16_gemm`
    /// for q_proj instead of the standard bf16 matmul.
    pub q_proj_marlin: Option<(WeightTensor, WeightTensor)>,
}

/// Gated DeltaNet linear attention weights.
///
/// These layers use O(1) recurrent state instead of KV cache.
/// 24 out of 32 layers in Qwen3.5-4B use this mechanism.
#[derive(Debug)]
pub struct LinearAttentionWeights {
    /// Fused QKV projection. [3 * num_heads * head_dim, hidden_size]
    pub in_proj_qkv: WeightTensor,
    /// Gate projection for output gating. [num_heads * head_dim, hidden_size]
    pub in_proj_z: WeightTensor,
    /// Output projection. [hidden_size, num_heads * head_dim]
    pub out_proj: WeightTensor,
    /// Alpha gate input projection. [num_heads * head_dim, hidden_size]
    pub in_proj_a: WeightTensor,
    /// Beta gate input projection. [num_heads * head_dim, hidden_size]
    pub in_proj_b: WeightTensor,
    /// Short convolution (causal conv1d). [num_heads * head_dim, 1, conv_size]
    pub conv1d: WeightTensor,
    /// Group norm weights. [num_heads * head_dim]
    pub norm: WeightTensor,
    /// Log of the A matrix (discretization parameter). [num_heads * head_dim]
    pub a_log: WeightTensor,
    /// Time-step bias. [num_heads * head_dim]
    pub dt_bias: WeightTensor,
}

/// Attention weights — either full GQA or linear (Gated DeltaNet).
#[derive(Debug)]
pub enum AttentionWeights {
    Full(FullAttentionWeights),
    Linear(LinearAttentionWeights),
}

/// SwiGLU feed-forward network weights (shared across all layer types).
#[derive(Debug)]
pub struct FfnWeights {
    /// [intermediate_size, hidden_size]
    pub gate_proj: WeightTensor,
    /// [intermediate_size, hidden_size]
    pub up_proj: WeightTensor,
    /// [hidden_size, intermediate_size]
    pub down_proj: WeightTensor,
}

/// One transformer layer's complete weights.
#[derive(Debug)]
pub struct LayerWeights {
    /// RMSNorm before attention. [hidden_size]
    pub input_layernorm: WeightTensor,
    /// RMSNorm before FFN. [hidden_size]
    pub post_attention_layernorm: WeightTensor,
    /// Attention weights (full or linear depending on layer index).
    pub attention: AttentionWeights,
    /// Feed-forward network weights.
    pub mlp: FfnWeights,
}

/// Complete Qwen3.5-4B language model weights.
///
/// Note: lm_head is tied to embed_tokens (shared weight matrix),
/// so we don't store it separately.
#[derive(Debug)]
pub struct ModelWeights {
    pub embedding: EmbeddingWeights,
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm. [hidden_size]
    pub final_norm: WeightTensor,
}

impl ModelWeights {
    /// Total size of all loaded weights in bytes.
    pub fn total_bytes(&self) -> usize {
        let mut total = self.embedding.embed_tokens.size_bytes();
        total += self.final_norm.size_bytes();
        for layer in &self.layers {
            total += layer.input_layernorm.size_bytes();
            total += layer.post_attention_layernorm.size_bytes();
            total += layer.mlp.gate_proj.size_bytes();
            total += layer.mlp.up_proj.size_bytes();
            total += layer.mlp.down_proj.size_bytes();
            match &layer.attention {
                AttentionWeights::Full(attn) => {
                    total += attn.q_proj.size_bytes();
                    total += attn.k_proj.size_bytes();
                    total += attn.v_proj.size_bytes();
                    total += attn.o_proj.size_bytes();
                    total += attn.q_norm.size_bytes();
                    total += attn.k_norm.size_bytes();
                    if let Some((b, s)) = &attn.q_proj_marlin {
                        total += b.size_bytes();
                        total += s.size_bytes();
                    }
                }
                AttentionWeights::Linear(attn) => {
                    total += attn.in_proj_qkv.size_bytes();
                    total += attn.in_proj_z.size_bytes();
                    total += attn.out_proj.size_bytes();
                    total += attn.in_proj_a.size_bytes();
                    total += attn.in_proj_b.size_bytes();
                    total += attn.conv1d.size_bytes();
                    total += attn.norm.size_bytes();
                    total += attn.a_log.size_bytes();
                    total += attn.dt_bias.size_bytes();
                }
            }
        }
        total
    }

    /// Total number of parameters.
    pub fn total_params(&self) -> usize {
        let mut total = self.embedding.embed_tokens.numel();
        total += self.final_norm.numel();
        for layer in &self.layers {
            total += layer.input_layernorm.numel();
            total += layer.post_attention_layernorm.numel();
            total += layer.mlp.gate_proj.numel();
            total += layer.mlp.up_proj.numel();
            total += layer.mlp.down_proj.numel();
            match &layer.attention {
                AttentionWeights::Full(attn) => {
                    total += attn.q_proj.numel();
                    total += attn.k_proj.numel();
                    total += attn.v_proj.numel();
                    total += attn.o_proj.numel();
                    total += attn.q_norm.numel();
                    total += attn.k_norm.numel();
                    // Marlin weights are a storage-only artifact; they duplicate
                    // q_proj in compressed form and should not inflate the
                    // reported parameter count.
                }
                AttentionWeights::Linear(attn) => {
                    total += attn.in_proj_qkv.numel();
                    total += attn.in_proj_z.numel();
                    total += attn.out_proj.numel();
                    total += attn.in_proj_a.numel();
                    total += attn.in_proj_b.numel();
                    total += attn.conv1d.numel();
                    total += attn.norm.numel();
                    total += attn.a_log.numel();
                    total += attn.dt_bias.numel();
                }
            }
        }
        total
    }
}
