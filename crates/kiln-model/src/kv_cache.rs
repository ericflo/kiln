//! Contiguous KV cache for efficient autoregressive generation.
//!
//! Stores per-layer K/V tensors for full-attention layers so that each decode
//! step only computes attention over the new token(s), reading cached K/V for
//! all previous positions.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

/// Per-layer KV cache for full-attention layers.
///
/// Each full-attention layer gets a pair of pre-allocated tensors
/// `[1, num_kv_heads, max_seq_len, head_dim]` that are progressively filled
/// as tokens are processed. Only full-attention layers need KV cache;
/// linear attention (Gated DeltaNet) layers maintain O(1) recurrent state.
pub struct KvCache {
    /// Per full-attention layer: (k_cache, v_cache) tensors.
    /// Indexed by full-attention layer index (0-based counter of only the
    /// full-attention layers, not absolute layer index).
    layers: Vec<(Tensor, Tensor)>,
    /// Current sequence length (number of cached positions).
    seq_len: usize,
    /// Maximum sequence length the cache can hold.
    max_seq_len: usize,
}

impl KvCache {
    /// Create a new KV cache with pre-allocated tensors.
    ///
    /// - `num_full_attn_layers`: number of full-attention layers that need caching
    /// - `num_kv_heads`: number of KV heads per layer
    /// - `head_dim`: dimension per head
    /// - `max_seq_len`: maximum sequence length to allocate for
    /// - `dtype`: tensor data type
    /// - `device`: device to allocate on
    pub fn new(
        num_full_attn_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_full_attn_layers);
        for i in 0..num_full_attn_layers {
            let k = Tensor::zeros((1, num_kv_heads, max_seq_len, head_dim), dtype, device)
                .with_context(|| format!("allocating k_cache for full-attn layer {i}"))?;
            let v = Tensor::zeros((1, num_kv_heads, max_seq_len, head_dim), dtype, device)
                .with_context(|| format!("allocating v_cache for full-attn layer {i}"))?;
            layers.push((k, v));
        }
        Ok(Self {
            layers,
            seq_len: 0,
            max_seq_len,
        })
    }

    /// Number of positions currently cached.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Append new K/V for a full-attention layer and return the full
    /// (cached + new) K/V tensors.
    ///
    /// - `layer_idx`: 0-based index into full-attention layers only
    /// - `new_k`: `[1, num_kv_heads, new_len, head_dim]`
    /// - `new_v`: `[1, num_kv_heads, new_len, head_dim]`
    ///
    /// Returns `(full_k, full_v)` each shaped `[1, num_kv_heads, seq_len + new_len, head_dim]`.
    ///
    /// Note: `seq_len` is updated only after the *last* layer calls `update` for
    /// a given step. The caller must call `advance(new_len)` after all layers
    /// have been updated.
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let new_len = new_k.dim(2)?;
        let end = self.seq_len + new_len;
        anyhow::ensure!(
            end <= self.max_seq_len,
            "KV cache overflow: seq_len {} + new {} > max {}",
            self.seq_len,
            new_len,
            self.max_seq_len
        );

        let (k_cache, v_cache) = &mut self.layers[layer_idx];

        // Write new K/V into the pre-allocated cache at [seq_len..seq_len+new_len]
        k_cache.slice_set(new_k, 2, self.seq_len)?;
        v_cache.slice_set(new_v, 2, self.seq_len)?;

        // Return the filled portion: [1, num_kv_heads, 0..end, head_dim]
        let full_k = k_cache.narrow(2, 0, end)?;
        let full_v = v_cache.narrow(2, 0, end)?;

        Ok((full_k, full_v))
    }

    /// Advance the cached sequence length after all layers have been updated
    /// for a given step.
    pub fn advance(&mut self, new_len: usize) {
        self.seq_len += new_len;
    }

    /// Reset the cache (e.g., for a new sequence).
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() -> Result<()> {
        let cache = KvCache::new(2, 4, 8, 128, DType::F32, &Device::Cpu)?;
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.layers.len(), 2);
        Ok(())
    }

    #[test]
    fn test_kv_cache_update_and_advance() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(1, 2, 4, 32, DType::F32, &device)?;

        // Simulate prefill with 3 tokens
        let k = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let (full_k, full_v) = cache.update(0, &k, &v)?;
        assert_eq!(full_k.dims(), &[1, 2, 3, 4]);
        assert_eq!(full_v.dims(), &[1, 2, 3, 4]);
        cache.advance(3);
        assert_eq!(cache.seq_len(), 3);

        // Simulate decode with 1 token
        let k2 = Tensor::ones((1, 2, 1, 4), DType::F32, &device)?;
        let v2 = Tensor::ones((1, 2, 1, 4), DType::F32, &device)?;
        let (full_k, full_v) = cache.update(0, &k2, &v2)?;
        assert_eq!(full_k.dims(), &[1, 2, 4, 4]); // 3 + 1 = 4
        assert_eq!(full_v.dims(), &[1, 2, 4, 4]);
        cache.advance(1);
        assert_eq!(cache.seq_len(), 4);

        Ok(())
    }

    #[test]
    fn test_kv_cache_overflow() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(1, 1, 4, 4, DType::F32, &device)?;

        let k = Tensor::ones((1, 1, 3, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 1, 3, 4), DType::F32, &device)?;
        cache.update(0, &k, &v)?;
        cache.advance(3);

        // This should overflow: 3 + 2 > 4
        let k2 = Tensor::ones((1, 1, 2, 4), DType::F32, &device)?;
        let v2 = Tensor::ones((1, 1, 2, 4), DType::F32, &device)?;
        let result = cache.update(0, &k2, &v2);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_kv_cache_reset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(1, 1, 4, 16, DType::F32, &device)?;

        let k = Tensor::ones((1, 1, 5, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 1, 5, 4), DType::F32, &device)?;
        cache.update(0, &k, &v)?;
        cache.advance(5);
        assert_eq!(cache.seq_len(), 5);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);

        Ok(())
    }

    #[test]
    fn test_kv_cache_content_preserved() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new(1, 1, 2, 8, DType::F32, &device)?;

        // Write known values for first 2 positions
        let k1 = Tensor::new(&[[[[1.0_f32, 2.0], [3.0, 4.0]]]], &device)?; // [1,1,2,2]
        let v1 = Tensor::new(&[[[[5.0_f32, 6.0], [7.0, 8.0]]]], &device)?;
        cache.update(0, &k1, &v1)?;
        cache.advance(2);

        // Write 1 more position
        let k2 = Tensor::new(&[[[[9.0_f32, 10.0]]]], &device)?; // [1,1,1,2]
        let v2 = Tensor::new(&[[[[11.0_f32, 12.0]]]], &device)?;
        let (full_k, full_v) = cache.update(0, &k2, &v2)?;
        cache.advance(1);

        // Verify all 3 positions are correct
        let k_vals = full_k.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(k_vals, vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0]);
        let v_vals = full_v.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v_vals, vec![5.0, 6.0, 7.0, 8.0, 11.0, 12.0]);

        Ok(())
    }
}
