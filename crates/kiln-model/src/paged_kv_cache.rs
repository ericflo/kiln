//! Paged KV cache backed by BlockManager's block tables.
//!
//! Stores K/V data in a shared pool of fixed-size blocks. Each sequence has a
//! [`BlockTable`] that maps logical token positions to physical block slots.
//! This enables multiple concurrent sequences to share a fixed KV cache memory
//! pool — the foundation for continuous batching.
//!
//! Supports optional FP8 (E4M3FN) quantization for ~2x memory savings.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use kiln_core::block::BlockTable;

use crate::fp8;

/// Paged KV cache that stores K/V in block-organized pool tensors.
///
/// Instead of pre-allocating `[max_seq_len]` per sequence (as [`KvCache`] does),
/// this cache maintains a shared pool of `num_blocks * block_size` slots.
/// Each sequence's [`BlockTable`] maps logical positions → physical slots,
/// enabling virtual memory semantics for KV cache.
pub struct PagedKvCache {
    /// Per full-attention layer: (k_pool, v_pool).
    /// Each pool has shape `[total_slots, num_kv_heads, head_dim]`
    /// where `total_slots = num_blocks * block_size`.
    /// When `fp8` is true, dtype is U8. Otherwise matches `compute_dtype`.
    layers: Vec<(Tensor, Tensor)>,
    block_size: usize,
    num_blocks: usize,
    /// Whether FP8 quantization is enabled.
    fp8: bool,
    /// Per-layer FP8 scale factors: (k_scale, v_scale).
    /// Global scale across the entire pool. Updated on writes.
    fp8_scales: Vec<(f32, f32)>,
    /// The original compute dtype for dequantization.
    compute_dtype: DType,
}

impl PagedKvCache {
    /// Create a new paged KV cache with pre-allocated pool tensors.
    pub fn new(
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_fp8(num_full_attn_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device, false)
    }

    /// Create a new paged KV cache with optional FP8 quantization.
    pub fn new_with_fp8(
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
        fp8: bool,
    ) -> Result<Self> {
        let storage_dtype = if fp8 { DType::U8 } else { dtype };
        let total_slots = num_blocks * block_size;
        let mut layers = Vec::with_capacity(num_full_attn_layers);
        for i in 0..num_full_attn_layers {
            let k = Tensor::zeros((total_slots, num_kv_heads, head_dim), storage_dtype, device)
                .with_context(|| format!("allocating k_pool for layer {i}"))?;
            let v = Tensor::zeros((total_slots, num_kv_heads, head_dim), storage_dtype, device)
                .with_context(|| format!("allocating v_pool for layer {i}"))?;
            layers.push((k, v));
        }
        let fp8_scales = vec![(1.0_f32, 1.0_f32); num_full_attn_layers];
        Ok(Self {
            layers,
            block_size,
            num_blocks,
            fp8,
            fp8_scales,
            compute_dtype: dtype,
        })
    }

    /// Write new K/V values into the paged cache at the given positions.
    ///
    /// - `layer_idx`: 0-based full-attention layer index
    /// - `block_table`: per-sequence page table mapping positions → physical blocks
    /// - `start_pos`: absolute position of the first new token
    /// - `k`: `[1, num_kv_heads, new_len, head_dim]`
    /// - `v`: `[1, num_kv_heads, new_len, head_dim]`
    pub fn write(
        &mut self,
        layer_idx: usize,
        block_table: &BlockTable,
        start_pos: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        if self.fp8 {
            self.write_fp8(layer_idx, block_table, start_pos, k, v)
        } else {
            self.write_native(layer_idx, block_table, start_pos, k, v)
        }
    }

    fn write_native(
        &mut self,
        layer_idx: usize,
        block_table: &BlockTable,
        start_pos: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        let new_len = k.dim(2)?;
        let (k_pool, v_pool) = &mut self.layers[layer_idx];

        // Reshape from [1, num_kv_heads, new_len, head_dim] to [new_len, num_kv_heads, head_dim]
        let k_flat = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let v_flat = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;

        for i in 0..new_len {
            let pos = start_pos + i;
            let slot = block_table
                .slot_for(pos, self.block_size)
                .ok_or_else(|| anyhow::anyhow!("no slot for position {pos} in block table"))?;

            let k_row = k_flat.narrow(0, i, 1)?; // [1, num_kv_heads, head_dim]
            let v_row = v_flat.narrow(0, i, 1)?;
            k_pool.slice_set(&k_row, 0, slot)?;
            v_pool.slice_set(&v_row, 0, slot)?;
        }

        Ok(())
    }

    fn write_fp8(
        &mut self,
        layer_idx: usize,
        block_table: &BlockTable,
        start_pos: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        let new_len = k.dim(2)?;

        // Reshape from [1, num_kv_heads, new_len, head_dim] to [new_len, num_kv_heads, head_dim]
        let k_flat = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let v_flat = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;

        // For the paged cache, we use direct FP8 conversion without per-tensor scaling.
        // This avoids the inconsistency of per-write scaling in a shared pool where
        // different writes would have different scales but read dequantizes uniformly.
        // E4M3FN can represent ±448, which is more than enough for normalized attention
        // K/V values (typically in the ±10 range).
        let k_q = fp8::quantize_to_fp8_direct(&k_flat)?;
        let v_q = fp8::quantize_to_fp8_direct(&v_flat)?;

        let (k_pool, v_pool) = &mut self.layers[layer_idx];

        for i in 0..new_len {
            let pos = start_pos + i;
            let slot = block_table
                .slot_for(pos, self.block_size)
                .ok_or_else(|| anyhow::anyhow!("no slot for position {pos} in block table"))?;

            let k_row = k_q.narrow(0, i, 1)?;
            let v_row = v_q.narrow(0, i, 1)?;
            k_pool.slice_set(&k_row, 0, slot)?;
            v_pool.slice_set(&v_row, 0, slot)?;
        }

        Ok(())
    }

    /// Read K/V values from the paged cache for positions `0..seq_len`.
    ///
    /// Returns `(k, v)` each shaped `[1, num_kv_heads, seq_len, head_dim]` in compute dtype.
    pub fn read(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (k_pool, v_pool) = &self.layers[layer_idx];

        // Compute physical slot indices for all positions
        let slot_indices: Vec<u32> = (0..seq_len)
            .map(|pos| {
                block_table
                    .slot_for(pos, self.block_size)
                    .map(|s| s as u32)
                    .ok_or_else(|| anyhow::anyhow!("no slot for position {pos}"))
            })
            .collect::<Result<Vec<_>>>()?;

        let device = k_pool.device();
        let indices = Tensor::new(&slot_indices[..], device)?;

        // Gather: [seq_len, num_kv_heads, head_dim]
        let k = k_pool.index_select(&indices, 0)?;
        let v = v_pool.index_select(&indices, 0)?;

        // Dequantize if FP8 (direct, no scaling — values are stored as raw E4M3)
        let (k, v) = if self.fp8 {
            let k = fp8::dequantize_from_fp8_direct(&k, self.compute_dtype, device)?;
            let v = fp8::dequantize_from_fp8_direct(&v, self.compute_dtype, device)?;
            (k, v)
        } else {
            (k, v)
        };

        // Reshape to [1, num_kv_heads, seq_len, head_dim]
        let k = k.transpose(0, 1)?.contiguous()?.unsqueeze(0)?;
        let v = v.transpose(0, 1)?.contiguous()?.unsqueeze(0)?;

        Ok((k, v))
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn is_fp8(&self) -> bool {
        self.fp8
    }

    /// Borrow the raw `(k_pool, v_pool)` tensors for `layer_idx`.
    ///
    /// Each pool has shape `[total_slots, num_kv_heads, head_dim]` where
    /// `total_slots = num_blocks * block_size`. Returns `None` if `layer_idx`
    /// is out of range.
    ///
    /// Intended for fused attention kernels (e.g. flash-attention paged decode)
    /// that index into the pool directly via a precomputed block_table tensor.
    /// For FP8 caches the returned tensors are still U8-encoded — callers must
    /// either dequantize first or use a kernel that supports FP8 inputs.
    pub fn pool_tensors(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        self.layers
            .get(layer_idx)
            .map(|(k, v)| (k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KvCache;

    #[test]
    fn test_write_then_read_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let num_kv_heads = 2;
        let head_dim = 4;
        let block_size = 4;
        let num_blocks = 4;

        let mut cache = PagedKvCache::new(1, num_blocks, block_size, num_kv_heads, head_dim, DType::F32, &device)?;

        // Set up a block table with 2 blocks (capacity = 8 tokens)
        let mut bt = BlockTable::new();
        bt.push(1); // logical block 0 -> physical block 1
        bt.push(3); // logical block 1 -> physical block 3

        // Write 3 tokens at position 0
        // Shape: [1, num_kv_heads=2, new_len=3, head_dim=4]
        let k = Tensor::new(
            &[[[[1.0_f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
              [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]],
            &device,
        )?;
        let v = Tensor::new(
            &[[[[101.0_f32, 102.0, 103.0, 104.0], [105.0, 106.0, 107.0, 108.0], [109.0, 110.0, 111.0, 112.0]],
              [[113.0, 114.0, 115.0, 116.0], [117.0, 118.0, 119.0, 120.0], [121.0, 122.0, 123.0, 124.0]]]],
            &device,
        )?;

        cache.write(0, &bt, 0, &k, &v)?;

        // Read back 3 positions
        let (k_out, v_out) = cache.read(0, &bt, 3)?;
        assert_eq!(k_out.dims(), &[1, 2, 3, 4]);
        assert_eq!(v_out.dims(), &[1, 2, 3, 4]);

        // Verify values match
        let k_orig = k.flatten_all()?.to_vec1::<f32>()?;
        let k_read = k_out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(k_orig, k_read, "K roundtrip failed");

        let v_orig = v.flatten_all()?.to_vec1::<f32>()?;
        let v_read = v_out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v_orig, v_read, "V roundtrip failed");

        Ok(())
    }

    #[test]
    fn test_multi_sequence_isolation() -> Result<()> {
        let device = Device::Cpu;
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 4;
        let num_blocks = 4;

        let mut cache = PagedKvCache::new(1, num_blocks, block_size, num_kv_heads, head_dim, DType::F32, &device)?;

        // Sequence A uses blocks 0 and 2
        let mut bt_a = BlockTable::new();
        bt_a.push(0);
        bt_a.push(2);

        // Sequence B uses blocks 1 and 3
        let mut bt_b = BlockTable::new();
        bt_b.push(1);
        bt_b.push(3);

        // Write 5 tokens for sequence A (values = 1.0)
        let k_a = Tensor::ones((1, 1, 5, 2), DType::F32, &device)?;
        let v_a = Tensor::ones((1, 1, 5, 2), DType::F32, &device)?;
        cache.write(0, &bt_a, 0, &k_a, &v_a)?;

        // Write 5 tokens for sequence B (values = 2.0)
        let k_b = (Tensor::ones((1, 1, 5, 2), DType::F32, &device)? * 2.0)?;
        let v_b = (Tensor::ones((1, 1, 5, 2), DType::F32, &device)? * 2.0)?;
        cache.write(0, &bt_b, 0, &k_b, &v_b)?;

        // Read back sequence A — should be all 1.0
        let (k_a_out, _) = cache.read(0, &bt_a, 5)?;
        let vals = k_a_out.flatten_all()?.to_vec1::<f32>()?;
        assert!(vals.iter().all(|&x| (x - 1.0).abs() < 1e-6), "Sequence A contaminated: {:?}", vals);

        // Read back sequence B — should be all 2.0
        let (k_b_out, _) = cache.read(0, &bt_b, 5)?;
        let vals = k_b_out.flatten_all()?.to_vec1::<f32>()?;
        assert!(vals.iter().all(|&x| (x - 2.0).abs() < 1e-6), "Sequence B contaminated: {:?}", vals);

        Ok(())
    }

    #[test]
    fn test_partial_block_handling() -> Result<()> {
        let device = Device::Cpu;
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 4;
        let num_blocks = 2;

        let mut cache = PagedKvCache::new(1, num_blocks, block_size, num_kv_heads, head_dim, DType::F32, &device)?;

        let mut bt = BlockTable::new();
        bt.push(0); // block 0: slots 0-3

        // Write only 2 tokens (partial block of size 4)
        let k = Tensor::new(&[[[[1.0_f32, 2.0], [3.0, 4.0]]]], &device)?; // [1,1,2,2]
        let v = Tensor::new(&[[[[5.0_f32, 6.0], [7.0, 8.0]]]], &device)?;
        cache.write(0, &bt, 0, &k, &v)?;

        // Read back 2 positions from partial block
        let (k_out, v_out) = cache.read(0, &bt, 2)?;
        assert_eq!(k_out.dims(), &[1, 1, 2, 2]);

        let k_vals = k_out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(k_vals, vec![1.0, 2.0, 3.0, 4.0]);
        let v_vals = v_out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v_vals, vec![5.0, 6.0, 7.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_incremental_write() -> Result<()> {
        let device = Device::Cpu;
        let num_kv_heads = 1;
        let head_dim = 2;
        let block_size = 4;
        let num_blocks = 2;

        let mut cache = PagedKvCache::new(1, num_blocks, block_size, num_kv_heads, head_dim, DType::F32, &device)?;

        let mut bt = BlockTable::new();
        bt.push(0);
        bt.push(1);

        // Write first 3 tokens (prefill)
        let k1 = Tensor::new(&[[[[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]]]], &device)?;
        let v1 = Tensor::new(&[[[[10.0_f32, 20.0], [30.0, 40.0], [50.0, 60.0]]]], &device)?;
        cache.write(0, &bt, 0, &k1, &v1)?;

        // Write 1 more token (decode step at position 3)
        let k2 = Tensor::new(&[[[[7.0_f32, 8.0]]]], &device)?;
        let v2 = Tensor::new(&[[[[70.0_f32, 80.0]]]], &device)?;
        cache.write(0, &bt, 3, &k2, &v2)?;

        // Read all 4 positions
        let (k_out, v_out) = cache.read(0, &bt, 4)?;
        assert_eq!(k_out.dims(), &[1, 1, 4, 2]);

        let k_vals = k_out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(k_vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let v_vals = v_out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v_vals, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);

        Ok(())
    }

    #[test]
    fn test_paged_vs_contiguous_equivalence() -> Result<()> {
        let device = Device::Cpu;
        let num_kv_heads = 2;
        let head_dim = 4;
        let max_seq_len = 16;
        let block_size = 4;
        let num_blocks = 4; // 4 blocks * 4 tokens = 16 slots

        // Create both caches
        let mut contiguous = KvCache::new(1, num_kv_heads, head_dim, max_seq_len, DType::F32, &device)?;
        let mut paged = PagedKvCache::new(1, num_blocks, block_size, num_kv_heads, head_dim, DType::F32, &device)?;

        // Block table with sequential blocks (0, 1, 2, 3) — same layout as contiguous
        let mut bt = BlockTable::new();
        for i in 0..num_blocks as u32 {
            bt.push(i);
        }

        // Prefill: write 5 tokens
        let k_prefill = Tensor::randn(0.0_f32, 1.0, (1, num_kv_heads, 5, head_dim), &device)?;
        let v_prefill = Tensor::randn(0.0_f32, 1.0, (1, num_kv_heads, 5, head_dim), &device)?;

        // Contiguous cache
        let (c_k, c_v) = contiguous.update(0, &k_prefill, &v_prefill)?;
        contiguous.advance(5);

        // Paged cache
        paged.write(0, &bt, 0, &k_prefill, &v_prefill)?;
        let (p_k, p_v) = paged.read(0, &bt, 5)?;

        // Compare
        let c_k_vals = c_k.flatten_all()?.to_vec1::<f32>()?;
        let p_k_vals = p_k.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(c_k_vals.len(), p_k_vals.len());
        for (i, (c, p)) in c_k_vals.iter().zip(p_k_vals.iter()).enumerate() {
            assert!((c - p).abs() < 1e-6, "K mismatch at index {i}: contiguous={c}, paged={p}");
        }

        let c_v_vals = c_v.flatten_all()?.to_vec1::<f32>()?;
        let p_v_vals = p_v.flatten_all()?.to_vec1::<f32>()?;
        for (i, (c, p)) in c_v_vals.iter().zip(p_v_vals.iter()).enumerate() {
            assert!((c - p).abs() < 1e-6, "V mismatch at index {i}: contiguous={c}, paged={p}");
        }

        // Decode: write 1 more token
        let k_decode = Tensor::randn(0.0_f32, 1.0, (1, num_kv_heads, 1, head_dim), &device)?;
        let v_decode = Tensor::randn(0.0_f32, 1.0, (1, num_kv_heads, 1, head_dim), &device)?;

        let (c_k, c_v) = contiguous.update(0, &k_decode, &v_decode)?;
        contiguous.advance(1);

        paged.write(0, &bt, 5, &k_decode, &v_decode)?;
        let (p_k, p_v) = paged.read(0, &bt, 6)?;

        let c_k_vals = c_k.flatten_all()?.to_vec1::<f32>()?;
        let p_k_vals = p_k.flatten_all()?.to_vec1::<f32>()?;
        for (i, (c, p)) in c_k_vals.iter().zip(p_k_vals.iter()).enumerate() {
            assert!((c - p).abs() < 1e-6, "K decode mismatch at {i}: contiguous={c}, paged={p}");
        }

        let c_v_vals = c_v.flatten_all()?.to_vec1::<f32>()?;
        let p_v_vals = p_v.flatten_all()?.to_vec1::<f32>()?;
        for (i, (c, p)) in c_v_vals.iter().zip(p_v_vals.iter()).enumerate() {
            assert!((c - p).abs() < 1e-6, "V decode mismatch at {i}: contiguous={c}, paged={p}");
        }

        Ok(())
    }

    #[test]
    fn test_multiple_layers() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PagedKvCache::new(3, 2, 4, 1, 2, DType::F32, &device)?;
        assert_eq!(cache.num_layers(), 3);

        let mut bt = BlockTable::new();
        bt.push(0);

        // Write different values to each layer
        for layer in 0..3 {
            let val = (layer + 1) as f32;
            let k = Tensor::new(&[[[[val, val * 10.0]]]], &device)?;
            let v = Tensor::new(&[[[[val * 100.0, val * 1000.0]]]], &device)?;
            cache.write(layer, &bt, 0, &k, &v)?;
        }

        // Read back and verify each layer is independent
        for layer in 0..3 {
            let val = (layer + 1) as f32;
            let (k_out, v_out) = cache.read(layer, &bt, 1)?;
            let k_vals = k_out.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(k_vals, vec![val, val * 10.0], "Layer {layer} K mismatch");
            let v_vals = v_out.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(v_vals, vec![val * 100.0, val * 1000.0], "Layer {layer} V mismatch");
        }

        Ok(())
    }

    // --- FP8 paged cache tests ---

    #[test]
    fn test_fp8_paged_write_read_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let num_kv_heads = 2;
        let head_dim = 4;
        let block_size = 4;
        let num_blocks = 4;

        let mut cache = PagedKvCache::new_with_fp8(
            1, num_blocks, block_size, num_kv_heads, head_dim,
            DType::F32, &device, true,
        )?;
        assert!(cache.is_fp8());

        let mut bt = BlockTable::new();
        bt.push(1);
        bt.push(3);

        let k = Tensor::new(
            &[[[[1.0_f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
              [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]],
            &device,
        )?;
        let v = Tensor::new(
            &[[[[101.0_f32, 102.0, 103.0, 104.0], [105.0, 106.0, 107.0, 108.0], [109.0, 110.0, 111.0, 112.0]],
              [[113.0, 114.0, 115.0, 116.0], [117.0, 118.0, 119.0, 120.0], [121.0, 122.0, 123.0, 124.0]]]],
            &device,
        )?;

        cache.write(0, &bt, 0, &k, &v)?;
        let (k_out, v_out) = cache.read(0, &bt, 3)?;

        assert_eq!(k_out.dims(), &[1, 2, 3, 4]);
        assert_eq!(k_out.dtype(), DType::F32);

        // Check approximate values (FP8 has limited precision)
        let k_orig = k.flatten_all()?.to_vec1::<f32>()?;
        let k_read = k_out.flatten_all()?.to_vec1::<f32>()?;
        for (i, (o, r)) in k_orig.iter().zip(k_read.iter()).enumerate() {
            let rel_err = (o - r).abs() / o.abs().max(0.01);
            assert!(rel_err < 0.15, "K index {i}: orig={o}, read={r}, rel_err={rel_err}");
        }

        Ok(())
    }

    #[test]
    fn test_fp8_paged_memory_savings() -> Result<()> {
        let device = Device::Cpu;
        let fp8_cache = PagedKvCache::new_with_fp8(1, 256, 16, 4, 256, DType::F32, &device, true)?;
        let native_cache = PagedKvCache::new(1, 256, 16, 4, 256, DType::F32, &device)?;

        assert_eq!(fp8_cache.layers[0].0.dtype(), DType::U8);
        assert_eq!(native_cache.layers[0].0.dtype(), DType::F32);

        // FP8 uses 1 byte per element vs 4 bytes for F32
        let fp8_bytes = fp8_cache.layers[0].0.elem_count(); // * 1 byte
        let native_bytes = native_cache.layers[0].0.elem_count() * 4; // * 4 bytes
        assert_eq!(fp8_bytes * 4, native_bytes);

        Ok(())
    }
}
