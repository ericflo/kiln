//! `kiln_tensor::Tensor`-backed twin of [`crate::paged_kv_cache::PagedKvCache`].
//!
//! Phase 3 / Phase 7 of #1082 — `PagedKvCache ported off candle_core::Tensor`
//! (line 110 / 167 / 324 of the epic). This is the scaffold landing:
//! constructors + accessors only. Writers, readers, and the
//! `write_token_major_native_graph_slot` CUDA-graph contract are follow-ups
//! that need the kt-API kernel calls + a `KvWriteSlotKt` rework.
//!
//! # Why a sibling type instead of a swap
//!
//! `PagedKvCache` is referenced from ~60 call sites across `kiln-model`,
//! `kiln-server`, and `kiln-scheduler`. A drop-in swap would be a massive PR.
//! Instead, the kt-tensor twin lives alongside the candle-typed cache and
//! call sites migrate one at a time (same playbook as the `_kt` kernel
//! crates and the kt-API call-site migrations in
//! `bench-results/substrate-status.md`'s Phase 7 section).
//!
//! # Compatibility surface
//!
//! Field shapes, dtype semantics, and FP8 quantization story are byte-for-
//! byte the same as the candle path:
//! - `layers: Vec<(KtTensor, KtTensor)>` — per-layer (k_pool, v_pool)
//! - pool shape `[total_slots, num_kv_heads, head_dim]` where
//!   `total_slots = num_blocks * block_size`
//! - FP8 storage uses `DType::U8` (caller dequantizes), matching
//!   `PagedKvCache::new_with_fp8`
//! - `block_size` / `num_blocks` / `is_fp8` / `compute_dtype` accessors are
//!   identical to the candle version

#![cfg(feature = "cuda")]

use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::cuda_backend::cudarc::driver::result as cudarc_result;
use candle_core::cuda_backend::CudaDevice;

use kiln_core::block::BlockTable;
use kiln_tensor::{
    cuda_zeros, CudaStorage, DType as KtDType, Layout, StorageBackend,
    Tensor as KtTensor, TensorId,
};

/// Paged KV cache backed by `kiln_tensor::Tensor`. Twin of
/// [`crate::paged_kv_cache::PagedKvCache`] for the Phase 7 migration.
///
/// Holds per-layer `(k_pool, v_pool)` tensors with the same byte layout
/// the candle version uses. Pool shape: `[total_slots, num_kv_heads,
/// head_dim]`, where `total_slots = num_blocks * block_size`.
///
/// FP8 caches store the pool as `DType::U8` and carry per-layer scale
/// factors; the compute dtype is preserved separately for dequant.
pub struct PagedKvCacheKt {
    /// Per full-attention layer: `(k_pool, v_pool)`.
    layers: Vec<(KtTensor, KtTensor)>,
    block_size: usize,
    num_blocks: usize,
    /// Whether FP8 quantization is enabled. When true, pool dtype is U8.
    fp8: bool,
    /// Per-layer FP8 scale factors `(k_scale, v_scale)`. Updated on
    /// writes by the FP8 path (writers land in a follow-up PR).
    #[allow(dead_code)]
    fp8_scales: Vec<(f32, f32)>,
    /// The original compute dtype for dequantization. Distinct from the
    /// storage dtype when FP8 is in use.
    compute_dtype: KtDType,
}

impl PagedKvCacheKt {
    /// Create a new paged KV cache with zero-filled pre-allocated pool
    /// tensors. Matches [`crate::paged_kv_cache::PagedKvCache::new`].
    pub fn new(
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: KtDType,
        candle_device: Arc<CudaDevice>,
        device_index: usize,
    ) -> Result<Self> {
        Self::new_with_fp8(
            num_full_attn_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype,
            candle_device,
            device_index,
            false,
        )
    }

    /// Create a new paged KV cache with optional FP8 quantization and
    /// zero-filled pools. Matches
    /// [`crate::paged_kv_cache::PagedKvCache::new_with_fp8`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_fp8(
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: KtDType,
        candle_device: Arc<CudaDevice>,
        device_index: usize,
        fp8: bool,
    ) -> Result<Self> {
        let storage_dtype = if fp8 { KtDType::U8 } else { dtype };
        let total_slots = num_blocks * block_size;
        let n_elements = total_slots * num_kv_heads * head_dim;
        let shape = vec![total_slots, num_kv_heads, head_dim];

        let mut layers = Vec::with_capacity(num_full_attn_layers);
        for i in 0..num_full_attn_layers {
            let k_storage = cuda_zeros(
                candle_device.clone(),
                device_index,
                storage_dtype,
                n_elements,
            )
            .with_context(|| format!("kt paged-kv: alloc k_pool layer {i}"))?;
            let v_storage = cuda_zeros(
                candle_device.clone(),
                device_index,
                storage_dtype,
                n_elements,
            )
            .with_context(|| format!("kt paged-kv: alloc v_pool layer {i}"))?;
            let k = KtTensor::from_parts(
                k_storage,
                Layout::contiguous(shape.clone()),
                TensorId::next(),
            )
            .with_context(|| format!("kt paged-kv: wrap k_pool layer {i}"))?;
            let v = KtTensor::from_parts(
                v_storage,
                Layout::contiguous(shape.clone()),
                TensorId::next(),
            )
            .with_context(|| format!("kt paged-kv: wrap v_pool layer {i}"))?;
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

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return one physical start slot per batch row when each logical
    /// window is a contiguous run in the shared KV pool. Mirrors
    /// [`crate::paged_kv_cache::PagedKvCache::contiguous_slot_run_starts`].
    ///
    /// Pure CPU bookkeeping over `BlockTable`s — no kt-Tensor work,
    /// device-agnostic, can be called from any thread.
    pub fn contiguous_slot_run_starts(
        &self,
        block_tables: &[&BlockTable],
        start_positions: &[usize],
        len: usize,
    ) -> Option<Vec<usize>> {
        crate::paged_kv_cache::contiguous_slot_run_starts(
            block_tables,
            self.block_size,
            start_positions,
            len,
        )
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

    /// The original compute dtype (BF16 typically). Distinct from the
    /// storage dtype (U8) when FP8 is enabled — callers dequantize using
    /// this dtype.
    pub fn compute_dtype(&self) -> KtDType {
        self.compute_dtype
    }

    /// Borrow the raw `(k_pool, v_pool)` kt-Tensors for `layer_idx`.
    /// Mirrors [`crate::paged_kv_cache::PagedKvCache::pool_tensors`].
    pub fn pool_tensors(&self, layer_idx: usize) -> Option<(&KtTensor, &KtTensor)> {
        self.layers.get(layer_idx).map(|(k, v)| (k, v))
    }

    /// Slot-based decode-token writer — the CUDA-graph contract entry
    /// point. Mirrors [`crate::paged_kv_cache::PagedKvCache::
    /// write_token_major_native_graph_slot`] but takes kt-Tensors.
    ///
    /// Inputs:
    /// - `k`, `v`: BF16 `[batch, 1, num_kv_heads, head_dim]`
    /// - `slot`: U32 `[1]` device tensor (or `[batch]` for the batched
    ///   variant — currently only `[1]` is exercised through this path)
    ///
    /// Returns `Ok(false)` when the cache is FP8-backed, when k/v
    /// aren't BF16, or when `k.dim(1) != 1` — callers fall back to
    /// [`Self::write`] (which doesn't exist yet; see the candle path
    /// for the full quantization story).
    ///
    /// Routes through
    /// [`kiln_flash_attn::paged_kv_write_token_major_bf16_slot_kt`],
    /// which accepts Borrowed kt-Tensors (Phase 7 v2 — PR #1360).
    pub fn write_token_major_native_graph_slot(
        &self,
        layer_idx: usize,
        k: &KtTensor,
        v: &KtTensor,
        slot: &KtTensor,
    ) -> Result<bool> {
        if self.fp8 || k.dtype() != KtDType::BF16 || v.dtype() != KtDType::BF16 {
            return Ok(false);
        }
        let k_shape = k.shape();
        if k_shape.len() < 2 || k_shape[1] != 1 {
            return Ok(false);
        }
        let (k_pool, v_pool) = &self.layers[layer_idx];
        kiln_flash_attn::paged_kv_write_token_major_bf16_slot_kt(k_pool, v_pool, k, v, slot)
            .map_err(|e| anyhow::anyhow!("kt paged_kv_write_token_major_bf16_slot: {e}"))?;
        Ok(true)
    }

    /// Multi-token writer for the **contiguous slot-run** case.
    ///
    /// When the per-sequence `BlockTable` resolves the new write region
    /// `[start_pos .. start_pos+len]` to one contiguous slot run in
    /// the shared KV pool (start_slot..start_slot+len), this path
    /// writes both k and v as a single device-to-device memcpy per
    /// pool — bypassing the per-slot FFI overhead the candle path
    /// pays in its loop.
    ///
    /// Requirements:
    /// - `k`, `v`: BF16, contiguous, both with element-count =
    ///   `len * num_kv_heads * head_dim` (the per-pool row stride).
    /// - `start_slot + len <= num_blocks * block_size`.
    ///
    /// Returns `Ok(())` on success. Returns an error if shapes/dtypes
    /// don't match or if the cache is FP8 (FP8 needs the quantized
    /// write path, not yet wired).
    ///
    /// **Routes through `cudarc::memcpy_dtod_async`** on the candle
    /// device's default stream — no kt-API kernel call, no nvcc
    /// kernel needed. The destination pool's storage is mutated
    /// in place via the raw device pointer (same idiom as the
    /// kt-API kernel crates).
    pub fn write_contiguous_slot_run(
        &self,
        layer_idx: usize,
        start_slot: usize,
        len: usize,
        k: &KtTensor,
        v: &KtTensor,
    ) -> Result<()> {
        if self.fp8 {
            anyhow::bail!(
                "kt PagedKvCacheKt::write_contiguous_slot_run: FP8 cache write \
                 path not yet wired"
            );
        }
        if k.dtype() != KtDType::BF16 || v.dtype() != KtDType::BF16 {
            anyhow::bail!(
                "kt PagedKvCacheKt::write_contiguous_slot_run requires BF16 inputs"
            );
        }
        let (k_pool, v_pool) = &self.layers[layer_idx];
        let pool_shape = k_pool.shape();
        anyhow::ensure!(
            pool_shape.len() == 3,
            "kt PagedKvCacheKt: k_pool must be rank-3 [total_slots, num_kv_heads, head_dim]"
        );
        let (total_slots, num_kv_heads, head_dim) =
            (pool_shape[0], pool_shape[1], pool_shape[2]);
        anyhow::ensure!(
            start_slot.checked_add(len).map_or(false, |e| e <= total_slots),
            "kt PagedKvCacheKt: slot range [{start_slot}..{}] exceeds total_slots {total_slots}",
            start_slot + len
        );
        let row_elems = num_kv_heads * head_dim;
        let expected_elems = len.checked_mul(row_elems).context("len * row_elems overflow")?;
        anyhow::ensure!(
            k.element_count() == expected_elems && v.element_count() == expected_elems,
            "kt PagedKvCacheKt: k/v must have {expected_elems} elements; got k={}, v={}",
            k.element_count(),
            v.element_count()
        );
        anyhow::ensure!(
            k.is_contiguous() && v.is_contiguous(),
            "kt PagedKvCacheKt::write_contiguous_slot_run requires contiguous k/v"
        );

        let bpe = KtDType::BF16.size_in_bytes();
        let total_bytes = expected_elems * bpe;
        let dst_byte_off = start_slot * row_elems * bpe;

        let k_src_cuda = k
            .storage()
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| anyhow::anyhow!("kt PagedKvCacheKt: k must be CUDA storage"))?;
        let v_src_cuda = v
            .storage()
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| anyhow::anyhow!("kt PagedKvCacheKt: v must be CUDA storage"))?;
        let k_dst_cuda = k_pool
            .storage()
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| anyhow::anyhow!("kt PagedKvCacheKt: k_pool must be CUDA storage"))?;
        let v_dst_cuda = v_pool
            .storage()
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| anyhow::anyhow!("kt PagedKvCacheKt: v_pool must be CUDA storage"))?;

        let k_src_off = k.layout().start_offset() * bpe;
        let v_src_off = v.layout().start_offset() * bpe;
        let (k_src_base, _) = k_src_cuda.device_ptr_raw();
        let (v_src_base, _) = v_src_cuda.device_ptr_raw();
        let (k_dst_base, _) = k_dst_cuda.device_ptr_raw();
        let (v_dst_base, _) = v_dst_cuda.device_ptr_raw();

        let stream = k_dst_cuda.candle_device().cuda_stream();
        let raw_stream = stream.cu_stream();

        unsafe {
            cudarc_result::memcpy_dtod_async(
                k_dst_base + dst_byte_off as u64,
                k_src_base + k_src_off as u64,
                total_bytes,
                raw_stream,
            )
            .map_err(|e| anyhow::anyhow!("kt pkv: memcpy_dtod_async k_pool: {e:?}"))?;
            cudarc_result::memcpy_dtod_async(
                v_dst_base + dst_byte_off as u64,
                v_src_base + v_src_off as u64,
                total_bytes,
                raw_stream,
            )
            .map_err(|e| anyhow::anyhow!("kt pkv: memcpy_dtod_async v_pool: {e:?}"))?;
        }
        Ok(())
    }

    /// Read `seq_len` tokens for one sequence out of the paged pool
    /// and reshape into `[1, num_kv_heads, seq_len, head_dim]` for
    /// downstream attention.
    ///
    /// Mirrors [`crate::paged_kv_cache::PagedKvCache::read`] for the
    /// **contiguous-slot-run case** (block_table's logical positions
    /// `0..seq_len` map to a single contiguous slot range in the
    /// pool). Returns an error for the non-contiguous case until a
    /// CUDA-side `index_select` lands — production decode hits the
    /// contiguous path the vast majority of the time.
    ///
    /// FP8 caches are rejected here too — the dequant kt-API entries
    /// aren't yet ported.
    ///
    /// **Cost:** one CUDA `Tensor::contiguous()` per pool (k and v),
    /// which now lands on the substrate kernel shipped in
    /// PR #1374. The narrow + transpose + unsqueeze before that are
    /// zero-copy layout-only ops.
    pub fn read(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<(KtTensor, KtTensor)> {
        if self.fp8 {
            anyhow::bail!(
                "kt PagedKvCacheKt::read: FP8 cache read path not yet wired \
                 (needs FP8 dequant kt-API)"
            );
        }
        let (k_pool, v_pool) = &self.layers[layer_idx];

        let start_slot = crate::paged_kv_cache::contiguous_slot_run_start(
            block_table,
            self.block_size,
            0,
            seq_len,
        )
        .ok_or_else(|| {
            anyhow::anyhow!(
                "kt PagedKvCacheKt::read: non-contiguous slot path not yet wired \
                 (needs CUDA-side index_select)"
            )
        })?;

        // narrow → transpose → contiguous → unsqueeze. All zero-copy
        // until contiguous(), which routes through the CUDA substrate
        // kernel (PR #1374).
        let k_slice = k_pool
            .narrow(0, start_slot, seq_len)
            .map_err(|e| anyhow::anyhow!("kt pkv read: narrow k: {e}"))?;
        let v_slice = v_pool
            .narrow(0, start_slot, seq_len)
            .map_err(|e| anyhow::anyhow!("kt pkv read: narrow v: {e}"))?;

        // [seq_len, num_kv_heads, head_dim] -> [num_kv_heads, seq_len, head_dim]
        let k_t = k_slice
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("kt pkv read: transpose k: {e}"))?;
        let v_t = v_slice
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("kt pkv read: transpose v: {e}"))?;

        // Materialize the transposed layout — the CUDA-side
        // contiguous kernel kicks in here.
        let k_c = k_t
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv read: contiguous k: {e}"))?;
        let v_c = v_t
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv read: contiguous v: {e}"))?;

        // Add leading batch dim → [1, num_kv_heads, seq_len, head_dim].
        let k_out = k_c
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("kt pkv read: unsqueeze k: {e}"))?;
        let v_out = v_c
            .unsqueeze(0)
            .map_err(|e| anyhow::anyhow!("kt pkv read: unsqueeze v: {e}"))?;

        Ok((k_out, v_out))
    }

    /// Host-slot variant: writes a single decode token at a host-known
    /// slot index. Mirrors the host-slot path of
    /// [`crate::paged_kv_cache::PagedKvCache::write_token_major_native`]
    /// (only the `new_len == 1` branch — multi-token writes need either
    /// the batched API or the slot-run path, both follow-ups).
    pub fn write_token_major_native_single(
        &self,
        layer_idx: usize,
        slot: usize,
        k: &KtTensor,
        v: &KtTensor,
    ) -> Result<bool> {
        if self.fp8 || k.dtype() != KtDType::BF16 || v.dtype() != KtDType::BF16 {
            return Ok(false);
        }
        let (k_pool, v_pool) = &self.layers[layer_idx];
        kiln_flash_attn::paged_kv_write_token_major_bf16_kt(k_pool, v_pool, k, v, slot)
            .map_err(|e| anyhow::anyhow!("kt paged_kv_write_token_major_bf16: {e}"))?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    // GPU-only tests are gated by KILN_TENSOR_CUDA_TEST=1 elsewhere;
    // here we only validate the type compiles + accessors are wired.

    use super::*;

    #[test]
    fn accessors_match_constructor_args() {
        // This test only exercises field plumbing — it does NOT allocate
        // on the GPU (gated separately). Instead we construct an empty
        // cache via the field-fill pattern and confirm the accessors
        // surface the expected values.
        let cache = PagedKvCacheKt {
            layers: Vec::new(),
            block_size: 16,
            num_blocks: 1024,
            fp8: true,
            fp8_scales: Vec::new(),
            compute_dtype: KtDType::BF16,
        };
        assert_eq!(cache.block_size(), 16);
        assert_eq!(cache.num_blocks(), 1024);
        assert_eq!(cache.num_layers(), 0);
        assert!(cache.is_fp8());
        assert_eq!(cache.compute_dtype(), KtDType::BF16);
        assert!(cache.pool_tensors(0).is_none());
    }
}
