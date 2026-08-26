//! Contiguous KV cache for efficient autoregressive generation.
//!
//! Stores per-layer K/V tensors for full-attention layers so that each decode
//! step only computes attention over the new token(s), reading cached K/V for
//! all previous positions.
//!
//! Supports optional FP8 (E4M3FN) quantization for ~2x memory savings.
//! When enabled, K/V values are quantized to 8-bit on write and dequantized
//! back to the compute dtype on read.
//!
//! # CUDA-native, token-major layout (#1082)
//!
//! The cache is `kiln_tensor` (`kt`)-native — no candle. K/V are stored
//! **token-major** `[max_seq_len, num_kv_heads, head_dim]` (sequence is the
//! outermost dim). That layout is what makes the per-step append a single
//! contiguous dim-0 [`kiln_tensor::Tensor::slice_set`] — one device→device
//! memcpy of just the new rows (`O(new_len)`, no realloc), kt's only supported
//! `slice_set` dim and the coalesced one. The old candle path stored
//! head-major `[1, nkv, max, hd]` and wrote along dim 2; that is a candle-ism
//! (dim-2 in-place writes are strided and aren't expressible in kt). Going
//! token-major is the "proper for the engine" choice, not a candle-shaped
//! port: writes are fully coalesced and the buffer is never reallocated.
//!
//! The public [`KvCache::update`] contract is unchanged — callers still pass
//! and receive head-major `[1, nkv, len, hd]`. The token-major ↔ head-major
//! transposes happen at the cache boundary (`head_major_to_token_major` /
//! `token_major_to_head_major`); the read transpose is the only added copy
//! versus the old strided-view narrow, and it is `O(end · nkv · hd)` — smaller
//! than the GQA head-expansion copy the attention path already pays downstream.

use anyhow::{Context, Result};
use kiln_tensor::{DType, Device, Tensor};

use crate::fp8;

/// Per-layer KV cache for full-attention layers.
///
/// Each full-attention layer gets a pair of pre-allocated token-major tensors
/// `[max_seq_len, num_kv_heads, head_dim]` that are progressively filled as
/// tokens are processed. Only full-attention layers need KV cache; linear
/// attention (Gated DeltaNet) layers maintain O(1) recurrent state.
pub struct KvCache {
    /// Per full-attention layer: (k_cache, v_cache), token-major
    /// `[max_seq_len, num_kv_heads, head_dim]`.
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
    /// The compute dtype (e.g. BF16) for the returned K/V (and dequant target).
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

    /// kt-typed alias of [`Self::new`] (#1082).
    ///
    /// Identical to [`Self::new`] now that the cache is kt-native (both take
    /// `kiln_tensor::DType` + `&kiln_tensor::Device`). Retained so the
    /// kiln-server / kiln-model call sites that use `new_kt` keep compiling
    /// without a churn pass.
    pub fn new_kt(
        num_full_attn_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new(
            num_full_attn_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            dtype,
            device,
        )
    }

    /// kt-typed alias of [`Self::new_with_fp8`] (#1082).
    ///
    /// Identical to [`Self::new_with_fp8`] now that the cache is kt-native.
    /// Retained for call-site compatibility (see [`Self::new_kt`]).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_fp8_kt(
        num_full_attn_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
        fp8: bool,
    ) -> Result<Self> {
        Self::new_with_fp8(
            num_full_attn_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            dtype,
            device,
            fp8,
        )
    }

    /// Create a new KV cache with optional FP8 quantization.
    #[allow(clippy::too_many_arguments)]
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
            // Token-major: sequence outermost so the per-step append is a
            // contiguous dim-0 `slice_set`.
            let k = Tensor::zeros_on(
                *device,
                vec![max_seq_len, num_kv_heads, head_dim],
                storage_dtype,
            )
            .with_context(|| format!("allocating k_cache for full-attn layer {i}"))?;
            let v = Tensor::zeros_on(
                *device,
                vec![max_seq_len, num_kv_heads, head_dim],
                storage_dtype,
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

    /// The cache's stored compute dtype (#1082).
    ///
    /// For FP8 caches this is the dtype the cache dequantizes to on
    /// [`Self::update`]; for native caches it is the storage dtype itself.
    /// Infallible now that the cache is kt-native (kept `Result` for
    /// call-site compatibility with the previous candle bridge).
    pub fn compute_dtype_kt(&self) -> Result<DType> {
        Ok(self.compute_dtype)
    }

    /// Append new K/V for a full-attention layer and return the full
    /// (cached + new) K/V tensors in compute dtype.
    ///
    /// - `layer_idx`: 0-based index into full-attention layers only
    /// - `new_k`: head-major `[1, num_kv_heads, new_len, head_dim]`
    /// - `new_v`: head-major `[1, num_kv_heads, new_len, head_dim]`
    ///
    /// Returns `(full_k, full_v)` each head-major
    /// `[1, num_kv_heads, seq_len + new_len, head_dim]`.
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

    /// Native (non-FP8) cache update.
    ///
    /// Transposes the head-major inputs to token-major, writes the new rows
    /// in place via a dim-0 `slice_set`, then reads `[0, end)` back and
    /// transposes to the head-major attention contract.
    fn update_native(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
        end: usize,
    ) -> Result<(Tensor, Tensor)> {
        let new_k_tm = head_major_to_token_major(new_k).context("kv update_native new_k")?;
        let new_v_tm = head_major_to_token_major(new_v).context("kv update_native new_v")?;

        {
            let (k_cache, v_cache) = &self.layers[layer_idx];
            // In-place dim-0 append (kt slice_set mutates the Arc-shared storage).
            k_cache
                .slice_set(&new_k_tm, 0, self.seq_len)
                .map_err(|e| anyhow::anyhow!("kv update_native k slice_set: {e}"))?;
            v_cache
                .slice_set(&new_v_tm, 0, self.seq_len)
                .map_err(|e| anyhow::anyhow!("kv update_native v slice_set: {e}"))?;
        }

        let (k_cache, v_cache) = &self.layers[layer_idx];
        let full_k_tm = k_cache.narrow(0, 0, end)?;
        let full_v_tm = v_cache.narrow(0, 0, end)?;
        let full_k = token_major_to_head_major(&full_k_tm).context("kv update_native full_k")?;
        let full_v = token_major_to_head_major(&full_v_tm).context("kv update_native full_v")?;
        Ok((full_k, full_v))
    }

    /// FP8 cache update: quantize new K/V, write to cache, read back and dequantize.
    ///
    /// Strategy: we re-quantize the entire filled region each time new tokens arrive.
    /// This ensures consistent scaling across all cached positions. For decode (new_len=1),
    /// this is a small overhead since the dequant+requant only touches the active portion.
    ///
    /// All math is token-major and kt-native (`crate::fp8` quant/dequant +
    /// `kiln_tensor::ops::concat` along dim 0); the head-major transpose is
    /// applied only to the final dequantized output.
    fn update_fp8(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
        end: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = new_k.device();
        let new_k_tm = head_major_to_token_major(new_k).context("kv update_fp8 new_k")?;
        let new_v_tm = head_major_to_token_major(new_v).context("kv update_fp8 new_v")?;

        if self.seq_len == 0 {
            // First write: just quantize the new data.
            let (k_q, k_scale) = fp8::quantize_to_fp8(&new_k_tm)?;
            let (v_q, v_scale) = fp8::quantize_to_fp8(&new_v_tm)?;
            {
                let (k_cache, v_cache) = &self.layers[layer_idx];
                k_cache
                    .slice_set(&k_q, 0, 0)
                    .map_err(|e| anyhow::anyhow!("kv update_fp8 k slice_set: {e}"))?;
                v_cache
                    .slice_set(&v_q, 0, 0)
                    .map_err(|e| anyhow::anyhow!("kv update_fp8 v slice_set: {e}"))?;
            }
            self.fp8_scales[layer_idx] = (k_scale, v_scale);
        } else {
            // Incremental: dequantize existing, concat new, re-quantize.
            let (old_k_scale, old_v_scale) = self.fp8_scales[layer_idx];
            let (existing_k_q, existing_v_q) = {
                let (k_cache, v_cache) = &self.layers[layer_idx];
                (
                    k_cache.narrow(0, 0, self.seq_len)?,
                    v_cache.narrow(0, 0, self.seq_len)?,
                )
            };

            let existing_k =
                fp8::dequantize_from_fp8(&existing_k_q, old_k_scale, self.compute_dtype, &device)?;
            let existing_v =
                fp8::dequantize_from_fp8(&existing_v_q, old_v_scale, self.compute_dtype, &device)?;

            let new_k_typed = new_k_tm.to_dtype(self.compute_dtype)?;
            let new_v_typed = new_v_tm.to_dtype(self.compute_dtype)?;

            let full_k = kiln_tensor::ops::concat(&[&existing_k, &new_k_typed], 0)?;
            let full_v = kiln_tensor::ops::concat(&[&existing_v, &new_v_typed], 0)?;

            let (k_q, k_scale) = fp8::quantize_to_fp8(&full_k)?;
            let (v_q, v_scale) = fp8::quantize_to_fp8(&full_v)?;

            {
                let (k_cache, v_cache) = &self.layers[layer_idx];
                k_cache
                    .slice_set(&k_q, 0, 0)
                    .map_err(|e| anyhow::anyhow!("kv update_fp8 k slice_set: {e}"))?;
                v_cache
                    .slice_set(&v_q, 0, 0)
                    .map_err(|e| anyhow::anyhow!("kv update_fp8 v slice_set: {e}"))?;
            }
            self.fp8_scales[layer_idx] = (k_scale, v_scale);
        }

        // Read back the full region and dequantize for attention computation.
        let (k_scale, v_scale) = self.fp8_scales[layer_idx];
        let (full_k_q, full_v_q) = {
            let (k_cache, v_cache) = &self.layers[layer_idx];
            (k_cache.narrow(0, 0, end)?, v_cache.narrow(0, 0, end)?)
        };
        let full_k_tm = fp8::dequantize_from_fp8(&full_k_q, k_scale, self.compute_dtype, &device)?;
        let full_v_tm = fp8::dequantize_from_fp8(&full_v_q, v_scale, self.compute_dtype, &device)?;
        let full_k = token_major_to_head_major(&full_k_tm).context("kv update_fp8 full_k")?;
        let full_v = token_major_to_head_major(&full_v_tm).context("kv update_fp8 full_v")?;

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

/// Convert a head-major `[1, nkv, len, hd]` K/V tensor (the public `update`
/// input contract) to the cache's token-major `[len, nkv, hd]` storage layout.
/// Contiguous result, ready for a dim-0 `slice_set`.
fn head_major_to_token_major(t: &Tensor) -> Result<Tensor> {
    // [1, nkv, len, hd] -> [nkv, len, hd] -> [len, nkv, hd]
    t.squeeze(0)?
        .transpose(0, 1)?
        .contiguous()
        .map_err(|e| anyhow::anyhow!("head_major_to_token_major: {e}"))
}

/// Inverse of [`head_major_to_token_major`]: token-major `[len, nkv, hd]` ->
/// head-major `[1, nkv, len, hd]` for the attention path. Contiguous so the
/// downstream GQA-expand / SDPA path can rely on a packed buffer regardless of
/// `gqa_ratio`.
fn token_major_to_head_major(t: &Tensor) -> Result<Tensor> {
    // [len, nkv, hd] -> [nkv, len, hd] -> [1, nkv, len, hd]
    t.transpose(0, 1)?
        .unsqueeze(0)?
        .contiguous()
        .map_err(|e| anyhow::anyhow!("token_major_to_head_major: {e}"))
}

/// CPU has no half-precision compute path in kt's host kernels, so a CPU cache
/// stores/returns F32. (Production decode is CUDA; this only affects the CPU
/// unit tests + any CPU type-check build.)
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

    /// Build a head-major `[1, nkv, len, hd]` kt CPU F32 tensor from a flat
    /// row-major value vec (the `update` input contract).
    fn kv(values: Vec<f32>, nkv: usize, len: usize, hd: usize) -> Tensor {
        Tensor::from_vec(values, vec![1, nkv, len, hd]).expect("kv from_vec")
    }

    /// Build a head-major all-ones `[1, nkv, len, hd]` kt CPU F32 tensor.
    fn kv_ones(nkv: usize, len: usize, hd: usize) -> Tensor {
        kv(vec![1.0_f32; nkv * len * hd], nkv, len, hd)
    }

    #[test]
    fn test_kv_cache_new() -> Result<()> {
        let cache = KvCache::new_kt(2, 4, 8, 128, DType::F32, &Device::Cpu)?;
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.layers.len(), 2);
        assert!(!cache.is_fp8());
        Ok(())
    }

    #[test]
    fn test_kv_cache_update_and_advance() -> Result<()> {
        let mut cache = KvCache::new_kt(1, 2, 4, 32, DType::F32, &Device::Cpu)?;

        // Simulate prefill with 3 tokens.
        let k = kv_ones(2, 3, 4);
        let v = kv_ones(2, 3, 4);
        let (full_k, full_v) = cache.update(0, &k, &v)?;
        assert_eq!(full_k.dims(), &[1, 2, 3, 4]);
        assert_eq!(full_v.dims(), &[1, 2, 3, 4]);
        cache.advance(3);
        assert_eq!(cache.seq_len(), 3);

        // Simulate decode with 1 token.
        let k2 = kv_ones(2, 1, 4);
        let v2 = kv_ones(2, 1, 4);
        let (full_k, full_v) = cache.update(0, &k2, &v2)?;
        assert_eq!(full_k.dims(), &[1, 2, 4, 4]); // 3 + 1 = 4
        assert_eq!(full_v.dims(), &[1, 2, 4, 4]);
        cache.advance(1);
        assert_eq!(cache.seq_len(), 4);

        Ok(())
    }

    #[test]
    fn test_kv_cache_overflow() -> Result<()> {
        let mut cache = KvCache::new_kt(1, 1, 4, 4, DType::F32, &Device::Cpu)?;

        let k = kv_ones(1, 3, 4);
        let v = kv_ones(1, 3, 4);
        cache.update(0, &k, &v)?;
        cache.advance(3);

        // This should overflow: 3 + 2 > 4.
        let k2 = kv_ones(1, 2, 4);
        let v2 = kv_ones(1, 2, 4);
        let result = cache.update(0, &k2, &v2);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_kv_cache_reset() -> Result<()> {
        let mut cache = KvCache::new_kt(1, 1, 4, 16, DType::F32, &Device::Cpu)?;

        let k = kv_ones(1, 5, 4);
        let v = kv_ones(1, 5, 4);
        cache.update(0, &k, &v)?;
        cache.advance(5);
        assert_eq!(cache.seq_len(), 5);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);

        Ok(())
    }

    #[test]
    fn test_kv_cache_content_preserved() -> Result<()> {
        let mut cache = KvCache::new_kt(1, 1, 2, 8, DType::F32, &Device::Cpu)?;

        // Write known values for first 2 positions: [1,1,2,2].
        let k1 = kv(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2);
        let v1 = kv(vec![5.0, 6.0, 7.0, 8.0], 1, 2, 2);
        cache.update(0, &k1, &v1)?;
        cache.advance(2);

        // Write 1 more position: [1,1,1,2].
        let k2 = kv(vec![9.0, 10.0], 1, 1, 2);
        let v2 = kv(vec![11.0, 12.0], 1, 1, 2);
        let (full_k, full_v) = cache.update(0, &k2, &v2)?;
        cache.advance(1);

        // Verify all 3 positions are correct (head-major flatten).
        let k_vals = full_k.to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(k_vals, vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0]);
        let v_vals = full_v.to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(v_vals, vec![5.0, 6.0, 7.0, 8.0, 11.0, 12.0]);

        Ok(())
    }

    // --- FP8 tests ---

    #[test]
    fn test_kv_cache_fp8_new() -> Result<()> {
        let cache = KvCache::new_with_fp8_kt(2, 4, 8, 128, DType::F32, &Device::Cpu, true)?;
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_fp8());
        // Storage should be U8.
        assert_eq!(cache.layers[0].0.dtype(), DType::U8);
        assert_eq!(cache.layers[0].1.dtype(), DType::U8);
        Ok(())
    }

    #[test]
    fn test_kv_cache_fp8_update_and_advance() -> Result<()> {
        let mut cache = KvCache::new_with_fp8_kt(1, 2, 4, 32, DType::F32, &Device::Cpu, true)?;

        let k = kv_ones(2, 3, 4);
        let v = kv_ones(2, 3, 4);
        let (full_k, full_v) = cache.update(0, &k, &v)?;
        assert_eq!(full_k.dims(), &[1, 2, 3, 4]);
        assert_eq!(full_v.dims(), &[1, 2, 3, 4]);
        // Output should be in compute dtype (F32).
        assert_eq!(full_k.dtype(), DType::F32);
        cache.advance(3);

        let k2 = kv_ones(2, 1, 4);
        let v2 = kv_ones(2, 1, 4);
        let (full_k, full_v) = cache.update(0, &k2, &v2)?;
        assert_eq!(full_k.dims(), &[1, 2, 4, 4]);
        assert_eq!(full_v.dims(), &[1, 2, 4, 4]);
        cache.advance(1);
        assert_eq!(cache.seq_len(), 4);

        Ok(())
    }

    #[test]
    fn test_kv_cache_fp8_approximate_values() -> Result<()> {
        let mut cache = KvCache::new_with_fp8_kt(1, 1, 2, 8, DType::F32, &Device::Cpu, true)?;

        let k1 = kv(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2);
        let v1 = kv(vec![5.0, 6.0, 7.0, 8.0], 1, 2, 2);
        cache.update(0, &k1, &v1)?;
        cache.advance(2);

        let k2 = kv(vec![9.0, 10.0], 1, 1, 2);
        let v2 = kv(vec![11.0, 12.0], 1, 1, 2);
        let (full_k, full_v) = cache.update(0, &k2, &v2)?;
        cache.advance(1);

        // FP8 has limited precision — check approximate match.
        let k_vals = full_k.to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
        let expected_k = [1.0, 2.0, 3.0, 4.0, 9.0, 10.0];
        for (i, (got, exp)) in k_vals.iter().zip(expected_k.iter()).enumerate() {
            let rel_err = (got - exp).abs() / exp.abs().max(0.01);
            assert!(
                rel_err < 0.15,
                "K index {i}: expected {exp}, got {got}, rel_err={rel_err}"
            );
        }

        let v_vals = full_v.to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
        let expected_v = [5.0, 6.0, 7.0, 8.0, 11.0, 12.0];
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
        // FP8 cache stores U8 (1 byte), native stores F32 (4 bytes) or BF16 (2 bytes).
        let fp8_cache = KvCache::new_with_fp8_kt(1, 4, 256, 1024, DType::F32, &Device::Cpu, true)?;
        let native_cache = KvCache::new_kt(1, 4, 256, 1024, DType::F32, &Device::Cpu)?;

        // Same number of elements.
        let fp8_elem = fp8_cache.layers[0].0.elem_count();
        let native_elem = native_cache.layers[0].0.elem_count();
        assert_eq!(fp8_elem, native_elem, "Same number of elements");

        // But FP8 uses 1 byte per element vs 4 bytes for F32.
        assert_eq!(fp8_cache.layers[0].0.dtype(), DType::U8);
        assert_eq!(native_cache.layers[0].0.dtype(), DType::F32);

        Ok(())
    }

    #[test]
    fn test_kv_cache_fp8_reset() -> Result<()> {
        let mut cache = KvCache::new_with_fp8_kt(1, 1, 4, 16, DType::F32, &Device::Cpu, true)?;

        let k = kv_ones(1, 5, 4);
        let v = kv_ones(1, 5, 4);
        cache.update(0, &k, &v)?;
        cache.advance(5);
        assert_eq!(cache.seq_len(), 5);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.fp8_scales[0], (1.0, 1.0));

        Ok(())
    }
}
