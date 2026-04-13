use serde::{Deserialize, Serialize};

/// Static model configuration, parsed from config.json at load time.
/// Hardcoded for Qwen3.5-4B architecture — a hybrid model with both
/// Gated DeltaNet (linear) attention and standard GQA attention layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub dtype: DType,

    /// How many of the layers use full GQA attention (and thus need KV cache).
    /// The remaining layers use Gated DeltaNet linear attention with O(1) state.
    /// For Qwen3.5-4B: 8 full-attention layers out of 32 total.
    pub num_full_attention_layers: usize,

    /// Full attention is applied every N layers (the rest are linear attention).
    /// For Qwen3.5-4B: every 4th layer (layers 3, 7, 11, 15, 19, 23, 27, 31).
    pub full_attention_interval: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    BF16,
    FP16,
    FP32,
}

impl ModelConfig {
    /// Qwen3.5-4B default configuration.
    ///
    /// Hybrid architecture: 32 layers total.
    /// - 24 Gated DeltaNet layers (linear attention, O(1) recurrent state)
    /// - 8 standard GQA layers (every 4th layer: full attention, needs KV cache)
    ///
    /// This means KV cache only scales with sequence length for 8 layers,
    /// not 32 — a ~4x reduction vs a pure transformer of the same depth.
    pub fn qwen3_5_4b() -> Self {
        Self {
            hidden_size: 2560,
            num_layers: 32,
            num_attention_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            intermediate_size: 9216,
            vocab_size: 248320,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            dtype: DType::BF16,
            num_full_attention_layers: 8,
            full_attention_interval: 4,
        }
    }

    /// Bytes per KV cache token for one full-attention layer (K+V).
    pub fn kv_cache_bytes_per_token_per_layer(&self) -> usize {
        let dtype_bytes = match self.dtype {
            DType::BF16 | DType::FP16 => 2,
            DType::FP32 => 4,
        };
        // K + V, each: num_kv_heads * head_dim * dtype_bytes
        2 * self.num_kv_heads * self.head_dim * dtype_bytes
    }

    /// Total KV cache bytes per token across all full-attention layers.
    /// Only full-attention layers need KV cache; linear attention layers
    /// maintain a fixed-size recurrent state independent of sequence length.
    pub fn kv_cache_bytes_per_token(&self) -> usize {
        self.kv_cache_bytes_per_token_per_layer() * self.num_full_attention_layers
    }

    /// GQA group size (how many Q heads share one KV head).
    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads
    }

    /// Whether a given layer index uses full attention (true) or linear attention (false).
    pub fn is_full_attention_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_5_4b_kv_cache_size() {
        let config = ModelConfig::qwen3_5_4b();
        // 2 * 4 * 256 * 2 = 4096 bytes per token per full-attention layer
        assert_eq!(config.kv_cache_bytes_per_token_per_layer(), 4096);
        // 4096 * 8 = 32768 bytes (~32 KB) per token total
        // Compare: a pure 32-layer transformer would be 4096 * 32 = 131072 (~128 KB)
        assert_eq!(config.kv_cache_bytes_per_token(), 32768);
        assert_eq!(config.gqa_group_size(), 4);
    }

    #[test]
    fn qwen3_5_4b_layer_pattern() {
        let config = ModelConfig::qwen3_5_4b();
        // Full attention at every 4th layer: indices 3, 7, 11, 15, 19, 23, 27, 31
        assert!(!config.is_full_attention_layer(0)); // linear
        assert!(!config.is_full_attention_layer(1)); // linear
        assert!(!config.is_full_attention_layer(2)); // linear
        assert!(config.is_full_attention_layer(3));  // full
        assert!(!config.is_full_attention_layer(4)); // linear
        assert!(config.is_full_attention_layer(7));  // full
        assert!(config.is_full_attention_layer(31)); // full (last layer)

        let full_count = (0..32).filter(|&i| config.is_full_attention_layer(i)).count();
        assert_eq!(full_count, 8);
    }

    #[test]
    fn qwen3_5_4b_memory_at_128k() {
        let config = ModelConfig::qwen3_5_4b();
        let ctx_len = 131072; // 128K tokens
        let kv_bytes = config.kv_cache_bytes_per_token() * ctx_len;
        let kv_gb = kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        // ~32 KB/token * 131072 tokens = ~4 GB
        // This is the magic: 128K context in only ~4 GB of KV cache!
        assert!(kv_gb > 3.5 && kv_gb < 4.5, "KV at 128K should be ~4 GB, got {kv_gb:.2} GB");
    }
}
