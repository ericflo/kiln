//! Contiguous KV cache for efficient autoregressive generation.
//!
//! Stores per-layer K/V tensors for full-attention layers so that each decode
//! step only computes attention over the new token(s), reading cached K/V for
//! all previous positions.
//!
//! Supports optional FP8 (E4M3FN) quantization for ~2x memory savings.
//! When enabled, K/V values are quantized to 8-bit on write and dequantized
//! back to the compute dtype on read.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::fp8;

/// Per-layer KV cache for full-attention layers.
///
/// Each full-attention layer gets a pair of pre-allocated tensors
/// `[1, num_kv_heads, max_seq_len, head_dim]` that are progressively filled
/// as tokens are processed. Only full-attention layers need KV cache;
/// linear attention (Gated DeltaNet) layers maintain O(1) recurrent state.
pub struct KvCache {
    /// Per full-attention layer: (k_cache, v_cache) tensors.
    /// When `fp8` is false: dtype matches `compute_dtype`.
    /// When `fp8` is true: dtype is U8 (FP8 E4M3 bit patterns).
    layers: Vec<(Tensor, Tensor)>,
    /// Current sequence length (number of cached positions).
    seq_len: usize,
    /// Maximum sequence length the cache can hold.
    max_seq_len: usize,
    /// Whether FP8 quantization is enabled.
    fp8: bool,
    /// Per-layer FP8 scale factors: (k_scale, v_scale).
    /// Only used when `fp8` is true. Updated on each write.
    fp8_scales: Vec<(f32, f32)>,
    /// The original compute dtype (e.g. BF16) for dequantization.
    compute_dtype: DType,
}

impl KvCache {
    /// Create a new KV cache with pre-allocated tensors.
    ///
    /// - `num_full_attn_layers`: number of full-attention layers that need caching
    /// - `num_kv_heads`: number of KV heads per layer
    /// - `head_dim`: dimension per head
    /// - `max_seq_len`: maximum sequence length to allocate for
    /// - `dtype`: tensor data type (compute dtype)
    /// - `device`: device to allocate on
    pub fn new(
        num_full_attn_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_fp8(
            num_full_attn_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            dtype,
            device,
            false,
        )
    }

    /// Create a new KV cache with optional FP8 quantization.
    pub fn new_with_fp8(
        num_full_attn_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
        fp8: bool,
    ) -> Result<Self> {
        let dtype = cpu_compatible_compute_dtype(dtype, device);
        let storage_dtype = if fp8 { DType::U8 } else { dtype };
        let mut layers = Vec::with_capacity(num_full_attn_layers);
        for i in 0..num_full_attn_layers {
            let k = Tensor::zeros(
                (1, num_kv_heads, max_seq_len, head_dim),
                storage_dtype,
                device,
            )
            .with_context(|| format!("allocating k_cache for full-attn layer {i}"))?;
            let v = Tensor::zeros(
                (1, num_kv_heads, max_seq_len, head_dim),
                storage_dtype,
                device,
            )
            .with_context(|| format!("allocating v_cache for full-attn layer {i}"))?;
            layers.push((k, v));
        }
        let fp8_scales = vec![(1.0_f32, 1.0_f32); num_full_attn_layers];
        Ok(Self {
            layers,
            seq_len: 0,
            max_seq_len,
            fp8,
            fp8_scales,
            compute_dtype: dtype,
        })
    }

    /// Number of positions currently cached.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Whether FP8 quantization is enabled.
    pub fn is_fp8(&self) -> bool {
        self.fp8
    }

    /// Append new K/V for a full-attention layer and return the full
    /// (cached + new) K/V tensors in compute dtype.
    ///
    /// - `layer_idx`: 0-based index into full-attention layers only
    /// - `new_k`: `[1, num_kv_heads, new_len, head_dim]`
    /// - `new_v`: `[1, num_kv_heads, new_len, head_dim]`
    ///
    /// Returns `(full_k, full_v)` each shaped `[1, num_kv_heads, seq_len + new_len, head_dim]`.
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

        if self.fp8 {
            self.update_fp8(layer_idx, new_k, new_v, end)
        } else {
            self.update_native(layer_idx, new_k, new_v, end)
        }
    }

    /// Native (non-FP8) cache update — same as original implementation.
    fn update_native(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
        end: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (k_cache, v_cache) = &mut self.layers[layer_idx];

        k_cache.slice_set(new_k, 2, self.seq_len)?;
        v_cache.slice_set(new_v, 2, self.seq_len)?;

        let full_k = k_cache.narrow(2, 0, end)?;
        let full_v = v_cache.narrow(2, 0, end)?;

        Ok((full_k, full_v))
    }

    /// FP8 cache update: quantize new K/V, write to cache, read back and dequantize.
    ///
    /// Strategy: we re-quantize the entire filled region each time new tokens arrive.
    /// This ensures consistent scaling across all cached positions. For decode (new_len=1),
    /// this is a small overhead since the dequant+requant only touches the active portion.
    fn update_fp8(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
        end: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = new_k.device().clone();
        let (k_cache, v_cache) = &mut self.layers[layer_idx];

        if self.seq_len == 0 {
            // First write: just quantize the new data
            let (k_q, k_scale) = fp8::quantize_to_fp8(new_k)?;
            let (v_q, v_scale) = fp8::quantize_to_fp8(new_v)?;
            k_cache.slice_set(&k_q, 2, 0)?;
            v_cache.slice_set(&v_q, 2, 0)?;
            self.fp8_scales[layer_idx] = (k_scale, v_scale);
        } else {
            // Incremental: dequantize existing, concat new, re-quantize
            let (old_k_scale, old_v_scale) = self.fp8_scales[layer_idx];

            let existing_k_q = k_cache.narrow(2, 0, self.seq_len)?;
            let existing_v_q = v_cache.narrow(2, 0, self.seq_len)?;

            let existing_k =
                fp8::dequantize_from_fp8(&existing_k_q, old_k_scale, self.compute_dtype, &device)?;
            let existing_v =
                fp8::dequantize_from_fp8(&existing_v_q, old_v_scale, self.compute_dtype, &device)?;

            let new_k_typed = new_k.to_dtype(self.compute_dtype)?;
            let new_v_typed = new_v.to_dtype(self.compute_dtype)?;

            let full_k = Tensor::cat(&[&existing_k, &new_k_typed], 2)?;
            let full_v = Tensor::cat(&[&existing_v, &new_v_typed], 2)?;

            let (k_q, k_scale) = fp8::quantize_to_fp8(&full_k)?;
            let (v_q, v_scale) = fp8::quantize_to_fp8(&full_v)?;

            k_cache.slice_set(&k_q, 2, 0)?;
            v_cache.slice_set(&v_q, 2, 0)?;
            self.fp8_scales[layer_idx] = (k_scale, v_scale);
        }

        // Read back the full region and dequantize for attention computation
        let (k_scale, v_scale) = self.fp8_scales[layer_idx];
        let full_k_q = k_cache.narrow(2, 0, end)?;
        let full_v_q = v_cache.narrow(2, 0, end)?;
        let full_k = fp8::dequantize_from_fp8(&full_k_q, k_scale, self.compute_dtype, &device)?;
        let full_v = fp8::dequantize_from_fp8(&full_v_q, v_scale, self.compute_dtype, &device)?;

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
        for s in &mut self.fp8_scales {
            *s = (1.0, 1.0);
        }
    }
}

fn cpu_compatible_compute_dtype(dtype: DType, device: &Device) -> DType {
    if matches!(device, Device::Cpu) && dtype != DType::F32 {
        DType::F32
    } else {
        dtype
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
        assert!(!cache.is_fp8());
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

    // --- FP8 tests ---

    #[test]
    fn test_kv_cache_fp8_new() -> Result<()> {
        let device = Device::Cpu;
        let cache = KvCache::new_with_fp8(2, 4, 8, 128, DType::F32, &device, true)?;
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_fp8());
        // Storage should be U8
        assert_eq!(cache.layers[0].0.dtype(), DType::U8);
        assert_eq!(cache.layers[0].1.dtype(), DType::U8);
        Ok(())
    }

    #[test]
    fn test_kv_cache_fp8_update_and_advance() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new_with_fp8(1, 2, 4, 32, DType::F32, &device, true)?;

        let k = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 3, 4), DType::F32, &device)?;
        let (full_k, full_v) = cache.update(0, &k, &v)?;
        assert_eq!(full_k.dims(), &[1, 2, 3, 4]);
        assert_eq!(full_v.dims(), &[1, 2, 3, 4]);
        // Output should be in compute dtype (F32)
        assert_eq!(full_k.dtype(), DType::F32);
        cache.advance(3);

        let k2 = Tensor::ones((1, 2, 1, 4), DType::F32, &device)?;
        let v2 = Tensor::ones((1, 2, 1, 4), DType::F32, &device)?;
        let (full_k, full_v) = cache.update(0, &k2, &v2)?;
        assert_eq!(full_k.dims(), &[1, 2, 4, 4]);
        assert_eq!(full_v.dims(), &[1, 2, 4, 4]);
        cache.advance(1);
        assert_eq!(cache.seq_len(), 4);

        Ok(())
    }

    #[test]
    fn test_kv_cache_fp8_approximate_values() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new_with_fp8(1, 1, 2, 8, DType::F32, &device, true)?;

        let k1 = Tensor::new(&[[[[1.0_f32, 2.0], [3.0, 4.0]]]], &device)?;
        let v1 = Tensor::new(&[[[[5.0_f32, 6.0], [7.0, 8.0]]]], &device)?;
        cache.update(0, &k1, &v1)?;
        cache.advance(2);

        let k2 = Tensor::new(&[[[[9.0_f32, 10.0]]]], &device)?;
        let v2 = Tensor::new(&[[[[11.0_f32, 12.0]]]], &device)?;
        let (full_k, full_v) = cache.update(0, &k2, &v2)?;
        cache.advance(1);

        // FP8 has limited precision — check approximate match
        let k_vals = full_k.flatten_all()?.to_vec1::<f32>()?;
        let expected_k = vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0];
        for (i, (got, exp)) in k_vals.iter().zip(expected_k.iter()).enumerate() {
            let rel_err = (got - exp).abs() / exp.abs().max(0.01);
            assert!(
                rel_err < 0.15,
                "K index {i}: expected {exp}, got {got}, rel_err={rel_err}"
            );
        }

        let v_vals = full_v.flatten_all()?.to_vec1::<f32>()?;
        let expected_v = vec![5.0, 6.0, 7.0, 8.0, 11.0, 12.0];
        for (i, (got, exp)) in v_vals.iter().zip(expected_v.iter()).enumerate() {
            let rel_err = (got - exp).abs() / exp.abs().max(0.01);
            assert!(
                rel_err < 0.15,
                "V index {i}: expected {exp}, got {got}, rel_err={rel_err}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_kv_cache_fp8_memory_savings() -> Result<()> {
        let device = Device::Cpu;
        // FP8 cache stores U8 (1 byte), native stores F32 (4 bytes) or BF16 (2 bytes)
        let fp8_cache = KvCache::new_with_fp8(1, 4, 256, 1024, DType::F32, &device, true)?;
        let native_cache = KvCache::new(1, 4, 256, 1024, DType::F32, &device)?;

        // FP8 cache tensors are U8 (1 byte each)
        let fp8_elem = fp8_cache.layers[0].0.elem_count();
        let native_elem = native_cache.layers[0].0.elem_count();
        assert_eq!(fp8_elem, native_elem, "Same number of elements");

        // But FP8 uses 1 byte per element vs 4 bytes for F32
        assert_eq!(fp8_cache.layers[0].0.dtype(), DType::U8);
        assert_eq!(native_cache.layers[0].0.dtype(), DType::F32);

        Ok(())
    }

    #[test]
    fn test_kv_cache_fp8_reset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new_with_fp8(1, 1, 4, 16, DType::F32, &device, true)?;

        let k = Tensor::ones((1, 1, 5, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 1, 5, 4), DType::F32, &device)?;
        cache.update(0, &k, &v)?;
        cache.advance(5);
        assert_eq!(cache.seq_len(), 5);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.fp8_scales[0], (1.0, 1.0));

        Ok(())
    }
}
