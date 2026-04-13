use serde::{Deserialize, Serialize};

/// Static model configuration, parsed from config.json at load time.
/// Hardcoded for Qwen3 architecture initially — no abstraction needed
/// until we support a second model (which may be never).
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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    BF16,
    FP16,
    FP32,
}

impl ModelConfig {
    /// Qwen3-4B default configuration.
    pub fn qwen3_4b() -> Self {
        Self {
            hidden_size: 2560,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            vocab_size: 151936,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            dtype: DType::BF16,
        }
    }

    /// Bytes per KV cache token (one layer, K+V, both heads).
    pub fn kv_cache_bytes_per_token_per_layer(&self) -> usize {
        let dtype_bytes = match self.dtype {
            DType::BF16 | DType::FP16 => 2,
            DType::FP32 => 4,
        };
        // K + V, each: num_kv_heads * head_dim * dtype_bytes
        2 * self.num_kv_heads * self.head_dim * dtype_bytes
    }

    /// Total KV cache bytes per token across all layers.
    pub fn kv_cache_bytes_per_token(&self) -> usize {
        self.kv_cache_bytes_per_token_per_layer() * self.num_layers
    }

    /// GQA group size (how many Q heads share one KV head).
    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_4b_kv_cache_size() {
        let config = ModelConfig::qwen3_4b();
        // 2 * 8 * 128 * 2 = 4096 bytes per token per layer
        assert_eq!(config.kv_cache_bytes_per_token_per_layer(), 4096);
        // 4096 * 36 = 147456 bytes per token total
        assert_eq!(config.kv_cache_bytes_per_token(), 147456);
        assert_eq!(config.gqa_group_size(), 4);
    }
}
