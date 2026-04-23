use std::fmt;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use memmap2::Mmap;

/// Data type for stored tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F16,
    BF16,
    F32,
}

impl TensorDType {
    /// Bytes per element.
    pub fn size_bytes(self) -> usize {
        match self {
            TensorDType::F16 | TensorDType::BF16 => 2,
            TensorDType::F32 => 4,
        }
    }
}

impl fmt::Display for TensorDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDType::F16 => write!(f, "f16"),
            TensorDType::BF16 => write!(f, "bf16"),
            TensorDType::F32 => write!(f, "f32"),
        }
    }
}

/// Backing storage for loaded tensor bytes.
#[derive(Clone)]
pub enum WeightData {
    /// Owned bytes, used by generated/dequantized tensors and tests.
    Owned(Vec<u8>),
    /// Read-only slice into a memory-mapped safetensors shard.
    MmapSlice {
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

impl WeightData {
    pub fn owned(data: Vec<u8>) -> Self {
        Self::Owned(data)
    }

    pub fn mmap_slice(mmap: Arc<Mmap>, offset: usize, len: usize) -> Self {
        Self::MmapSlice { mmap, offset, len }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            WeightData::Owned(data) => data,
            WeightData::MmapSlice { mmap, offset, len } => &mmap[*offset..*offset + *len],
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.as_bytes()
    }

    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }
}

impl Deref for WeightData {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

impl fmt::Debug for WeightData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightData::Owned(data) => f.debug_struct("Owned").field("len", &data.len()).finish(),
            WeightData::MmapSlice { offset, len, .. } => f
                .debug_struct("MmapSlice")
                .field("offset", offset)
                .field("len", len)
                .finish(),
        }
    }
}

/// Provenance for a loaded weight tensor.
///
/// This is used by the persistent transpose cache to key entries by the exact
/// checkpoint shard that supplied the bytes.
#[derive(Debug, Clone)]
pub struct WeightSource {
    pub shard_path: PathBuf,
    pub shard_size: u64,
    pub shard_mtime_ns: u128,
    pub tensor_name: String,
}

/// A loaded tensor: raw bytes with shape and dtype metadata.
///
/// This is a CPU-side representation. The forward pass will convert these
/// to GPU tensors (candle Tensor or raw CUDA buffers).
#[derive(Clone)]
pub struct WeightTensor {
    pub data: WeightData,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
    pub source: Option<WeightSource>,
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

    /// Raw tensor bytes in row-major safetensors order.
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}

impl fmt::Debug for WeightTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ds = f.debug_struct("WeightTensor");
        ds.field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("bytes", &self.data.len());
        if let Some(source) = &self.source {
            ds.field("source", source);
        }
        ds.finish()
    }
}

/// Token embedding weights.
#[derive(Debug, Clone)]
pub struct EmbeddingWeights {
    /// [vocab_size, hidden_size]
    pub embed_tokens: WeightTensor,
}

/// Standard GQA self-attention weights (for full attention layers).
///
/// These layers use FlashAttention with KV cache and RoPE.
/// Every 4th layer in Qwen3.5-4B (indices 3, 7, 11, ..., 31).
#[derive(Debug, Clone)]
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
}

/// Gated DeltaNet linear attention weights.
///
/// These layers use O(1) recurrent state instead of KV cache.
/// 24 out of 32 layers in Qwen3.5-4B use this mechanism.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub enum AttentionWeights {
    Full(FullAttentionWeights),
    Linear(LinearAttentionWeights),
}

/// SwiGLU feed-forward network weights (shared across all layer types).
#[derive(Debug, Clone)]
pub struct FfnWeights {
    /// [intermediate_size, hidden_size]
    pub gate_proj: WeightTensor,
    /// [intermediate_size, hidden_size]
    pub up_proj: WeightTensor,
    /// [hidden_size, intermediate_size]
    pub down_proj: WeightTensor,
}

/// One transformer layer's complete weights.
#[derive(Debug, Clone)]
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

/// Native MTP (Multi-Token Prediction) head weights for Qwen3.5-4B.
///
/// The pretrained Qwen3.5-4B checkpoint ships 15 MTP-prefixed tensors
/// (`num_nextn_predict_layers = 1` → `k=1` draft depth). The MTP head
/// lets us draft one token per decode step using the model's own
/// distilled head instead of a skip-layer self-spec approximation.
///
/// Forward shape (vLLM `qwen3_next_mtp.py` reference):
/// `concat(pre_fc_norm_embedding(embed(t)), pre_fc_norm_hidden(h)) → fc (2H→H)
///  → layer (GQA + SwiGLU MLP) → final_layernorm → tied lm_head (= embed_tokens.t())`
///
/// `lm_head` is tied to the base model's `embed_tokens`, so we do NOT
/// store a separate `lm_head` tensor — the spec-decode forward path
/// reuses `GpuWeights::embed_tokens_t`.
#[derive(Debug, Clone)]
pub struct MtpWeights {
    /// Concat-then-project: `[hidden_size, 2 * hidden_size]`.
    /// Ingests `concat(norm_embed, norm_hidden)` and produces `[seq, hidden_size]`.
    pub fc: WeightTensor,
    /// RMSNorm applied to the draft-candidate's token embedding before concat. `[hidden_size]`.
    pub pre_fc_norm_embedding: WeightTensor,
    /// RMSNorm applied to the base model's last hidden state before concat. `[hidden_size]`.
    pub pre_fc_norm_hidden: WeightTensor,
    /// Single MTP transformer layer (full GQA attention + SwiGLU MLP + input/post
    /// layernorms). Shape matches the main model's full-attention layer.
    pub layer: LayerWeights,
    /// Final RMSNorm before the tied lm_head. `[hidden_size]`.
    pub final_layernorm: WeightTensor,
}

impl MtpWeights {
    /// Total size of all MTP tensors in bytes.
    pub fn total_bytes(&self) -> usize {
        let mut total = self.fc.size_bytes();
        total += self.pre_fc_norm_embedding.size_bytes();
        total += self.pre_fc_norm_hidden.size_bytes();
        total += self.final_layernorm.size_bytes();
        total += self.layer.input_layernorm.size_bytes();
        total += self.layer.post_attention_layernorm.size_bytes();
        total += self.layer.mlp.gate_proj.size_bytes();
        total += self.layer.mlp.up_proj.size_bytes();
        total += self.layer.mlp.down_proj.size_bytes();
        match &self.layer.attention {
            AttentionWeights::Full(attn) => {
                total += attn.q_proj.size_bytes();
                total += attn.k_proj.size_bytes();
                total += attn.v_proj.size_bytes();
                total += attn.o_proj.size_bytes();
                total += attn.q_norm.size_bytes();
                total += attn.k_norm.size_bytes();
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
        total
    }

    /// Total parameter count across all MTP tensors.
    pub fn total_params(&self) -> usize {
        let mut total = self.fc.numel();
        total += self.pre_fc_norm_embedding.numel();
        total += self.pre_fc_norm_hidden.numel();
        total += self.final_layernorm.numel();
        total += self.layer.input_layernorm.numel();
        total += self.layer.post_attention_layernorm.numel();
        total += self.layer.mlp.gate_proj.numel();
        total += self.layer.mlp.up_proj.numel();
        total += self.layer.mlp.down_proj.numel();
        match &self.layer.attention {
            AttentionWeights::Full(attn) => {
                total += attn.q_proj.numel();
                total += attn.k_proj.numel();
                total += attn.v_proj.numel();
                total += attn.o_proj.numel();
                total += attn.q_norm.numel();
                total += attn.k_norm.numel();
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
        total
    }
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
    /// Optional native MTP head (Qwen3.5-4B ships one, other variants may not).
    /// Populated when `num_nextn_predict_layers > 0` in the model config AND the
    /// `mtp.*` tensors are present in the checkpoint. Consumed by
    /// `KILN_SPEC_METHOD=mtp` at serve time.
    pub mtp: Option<MtpWeights>,
}

impl ModelWeights {
    /// Total size of all loaded weights in bytes.
    pub fn total_bytes(&self) -> usize {
        let mut total = self.embedding.embed_tokens.size_bytes();
        total += self.final_norm.size_bytes();
        if let Some(mtp) = &self.mtp {
            total += mtp.total_bytes();
        }
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
        if let Some(mtp) = &self.mtp {
            total += mtp.total_params();
        }
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
