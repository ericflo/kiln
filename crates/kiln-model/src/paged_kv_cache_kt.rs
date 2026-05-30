//! `kiln_tensor::Tensor`-backed paged KV cache — the candle-drop
//! replacement for the deleted `kiln_model::paged_kv_cache::PagedKvCache`.
//!
//! Phase 3 / Phase 7 of #1082 — `PagedKvCache ported off candle_core::Tensor`
//! (line 110 / 167 / 324 of the epic). This is the scaffold landing:
//! constructors + accessors only. Writers, readers, and the
//! `write_token_major_native_graph_slot` CUDA-graph contract are follow-ups
//! that need the kt-API kernel calls + a `KvWriteSlotKt` rework.
//!
//! # Candle-drop (#1082)
//!
//! The candle-typed `PagedKvCache` and its `paged_kv_cache.rs` module are
//! deleted. This type is now the sole paged KV cache: it holds
//! `kiln_tensor::Tensor` pools end-to-end, allocates through
//! `cuda_zeros_ctx`, and routes writes/reads through the kt CUDA kernels
//! (`kiln_flash_attn::paged_kv_write_token_major_bf16*_kt`) and kt ops
//! (`slice_set`, `narrow`, `cuda_fp8_quantize_direct`,
//! `cuda_index_select_dim0`). It is `cfg(feature = "cuda")` — the only
//! production placement — so the slot-run bookkeeping it shares with
//! `forward.rs`' non-CUDA paths (`contiguous_slot_run_start[s]`) now lives
//! in `kiln_core::block`.
//!
//! # Compatibility surface
//!
//! Field shapes, dtype semantics, and FP8 quantization story are byte-for-
//! byte what the old candle cache used:
//! - `layers: Vec<(KtTensor, KtTensor)>` — per-layer (k_pool, v_pool)
//! - pool shape `[total_slots, num_kv_heads, head_dim]` where
//!   `total_slots = num_blocks * block_size`
//! - FP8 storage uses `DType::U8` (caller dequantizes)
//! - `block_size` / `num_blocks` / `is_fp8` / `compute_dtype` accessors

#![cfg(feature = "cuda")]

// #1082: cudarc imported directly instead of through
// candle_core::cuda_backend::cudarc::*. The candle re-export is a pure
// pass-through to the cudarc crate (candle-core wraps cudarc verbatim),
// so this drops two candle imports without changing runtime behavior.
// Pattern lifted from kiln-tensor commit 4ee1b7f9 and kiln-blas commit
// 0d201199. As of the candle-drop (#1082), the public `PagedKvCacheKt::new*`
// surface takes a `device_index: usize` instead of an
// `Arc<candle_core::cuda_backend::CudaDevice>` — allocation routes through
// `cuda_zeros_ctx(device_index, ..)` — so this file carries no candle import.
use anyhow::{Context, Result};
use cudarc::driver::result as cudarc_result;

use kiln_core::block::{contiguous_slot_run_start, contiguous_slot_run_starts, BlockTable};
use kiln_tensor::{
    cuda_fp8_dequantize_direct, cuda_fp8_quantize_direct, cuda_zeros_ctx, CudaStorage,
    DType as KtDType, Layout, Tensor as KtTensor, TensorId,
};

/// Paged KV cache backed by `kiln_tensor::Tensor`. Twin of
/// the deleted candle `PagedKvCache` (#1082 candle-drop).
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
    /// tensors. Replaces the candle `PagedKvCache::new` (#1082 candle-drop).
    ///
    /// `device_index` selects the CUDA device the pools are allocated on
    /// (allocation routes through [`cuda_zeros_ctx`]). The candle
    /// `Arc<CudaDevice>` parameter the old surface carried is gone — kiln
    /// is single-GPU, so the device index is the only placement input.
    pub fn new(
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: KtDType,
        device_index: usize,
    ) -> Result<Self> {
        Self::new_with_fp8(
            num_full_attn_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype,
            device_index,
            false,
        )
    }

    /// Create a new paged KV cache with optional FP8 quantization and
    /// zero-filled pools. Replaces the candle
    /// `PagedKvCache::new_with_fp8` (#1082 candle-drop).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_fp8(
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: KtDType,
        device_index: usize,
        fp8: bool,
    ) -> Result<Self> {
        let storage_dtype = if fp8 { KtDType::U8 } else { dtype };
        let total_slots = num_blocks * block_size;
        let n_elements = total_slots * num_kv_heads * head_dim;
        let shape = vec![total_slots, num_kv_heads, head_dim];

        let mut layers = Vec::with_capacity(num_full_attn_layers);
        for i in 0..num_full_attn_layers {
            let k_storage = cuda_zeros_ctx(device_index, storage_dtype, n_elements)
                .with_context(|| format!("kt paged-kv: alloc k_pool layer {i}"))?;
            let v_storage = cuda_zeros_ctx(device_index, storage_dtype, n_elements)
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
    /// window is a contiguous run in the shared KV pool. Thin wrapper over
    /// [`kiln_core::block::contiguous_slot_run_starts`].
    ///
    /// Pure CPU bookkeeping over `BlockTable`s — no kt-Tensor work,
    /// device-agnostic, can be called from any thread.
    pub fn contiguous_slot_run_starts(
        &self,
        block_tables: &[&BlockTable],
        start_positions: &[usize],
        len: usize,
    ) -> Option<Vec<usize>> {
        contiguous_slot_run_starts(block_tables, self.block_size, start_positions, len)
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
    pub fn pool_tensors(&self, layer_idx: usize) -> Option<(&KtTensor, &KtTensor)> {
        self.layers.get(layer_idx).map(|(k, v)| (k, v))
    }

    /// Slot-based decode-token writer — the CUDA-graph contract entry
    /// point. Replaces the candle
    /// `PagedKvCache::write_token_major_native_graph_slot`; takes kt-Tensors.
    ///
    /// Inputs:
    /// - `k`, `v`: BF16 `[batch, 1, num_kv_heads, head_dim]`
    /// - `slot`: U32 `[1]` device tensor (or `[batch]` for the batched
    ///   variant — currently only `[1]` is exercised through this path)
    ///
    /// Returns `Ok(false)` when k/v aren't BF16 or when `k.dim(1) != 1`
    /// — callers fall back to [`Self::write`].
    ///
    /// The BF16 (non-FP8) path routes through
    /// [`kiln_flash_attn::paged_kv_write_token_major_bf16_slot_kt`],
    /// which accepts Borrowed kt-Tensors (Phase 7 v2 — PR #1360) and
    /// reads the destination slot from the `slot` device tensor (the
    /// CUDA-graph replay-safe contract).
    ///
    /// The FP8 path (#1082 candle-drop) reads the single `[1]` U32
    /// `slot` device tensor back to a host index (one D2H of 4 bytes),
    /// quantizes the BF16 K/V into U8 via `cuda_fp8_quantize_direct`,
    /// and `slice_set`s the U8 rows into the U8 pool at that slot.
    /// Scale = 1.0 ("direct" mode) — matches the candle FP8 KV write
    /// story (per-slot scaling is not practical for a shared pool).
    /// This is NOT capturable under a CUDA graph (the D2H read forces a
    /// sync), so FP8 caches must run with graph capture disabled — the
    /// same constraint the candle path imposed by declining FP8 here.
    pub fn write_token_major_native_graph_slot(
        &self,
        layer_idx: usize,
        k: &KtTensor,
        v: &KtTensor,
        slot: &KtTensor,
    ) -> Result<bool> {
        if k.dtype() != KtDType::BF16 || v.dtype() != KtDType::BF16 {
            return Ok(false);
        }
        let k_shape = k.shape();
        if k_shape.len() < 2 || k_shape[1] != 1 {
            return Ok(false);
        }
        let (k_pool, v_pool) = &self.layers[layer_idx];

        if self.fp8 {
            // Read the device slot index host-side (one 4-byte D2H), then
            // quantize + slice_set into the U8 pool. Squeeze the seq_len=1
            // dim so the quantized rows are [num_kv_heads, head_dim]-shaped
            // (rank matches the pool rows for slice_set's inner-dim check).
            let slot_idx = slot
                .to_scalar::<u32>()
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: read slot: {e}"))?
                as usize;
            let k_sq = k
                .squeeze(1)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: squeeze k: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: contiguous k: {e}"))?;
            let v_sq = v
                .squeeze(1)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: squeeze v: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: contiguous v: {e}"))?;
            let k_q = cuda_fp8_quantize_direct(&k_sq)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: quantize k: {e}"))?;
            let v_q = cuda_fp8_quantize_direct(&v_sq)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: quantize v: {e}"))?;
            k_pool
                .slice_set(&k_q, 0, slot_idx)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: slice_set k: {e}"))?;
            v_pool
                .slice_set(&v_q, 0, slot_idx)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: slice_set v: {e}"))?;
            return Ok(true);
        }

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

        if self.fp8 {
            // FP8 path: quantize the BF16 source into a fresh U8
            // buffer using the kt-API FP8 kernel, then memcpy_dtod
            // the quantized buffer into the destination slot range.
            // Scale = 1.0 ("direct" mode) — matches the candle
            // PagedKvCache FP8 write path; per-slot scaling is not
            // practical for the KV cache.
            let k_q = cuda_fp8_quantize_direct(k)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8: quantize k: {e}"))?;
            let v_q = cuda_fp8_quantize_direct(v)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8: quantize v: {e}"))?;
            // U8 storage: 1 byte per element.
            let total_bytes = expected_elems;
            let dst_byte_off = start_slot * row_elems;
            let k_src_cuda = k_q
                .storage()
                .as_any()
                .downcast_ref::<CudaStorage>()
                .ok_or_else(|| anyhow::anyhow!("kt PagedKvCacheKt fp8: k_q must be CUDA"))?;
            let v_src_cuda = v_q
                .storage()
                .as_any()
                .downcast_ref::<CudaStorage>()
                .ok_or_else(|| anyhow::anyhow!("kt PagedKvCacheKt fp8: v_q must be CUDA"))?;
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
            let (k_src_base, _) = k_src_cuda.device_ptr_raw();
            let (v_src_base, _) = v_src_cuda.device_ptr_raw();
            let (k_dst_base, _) = k_dst_cuda.device_ptr_raw();
            let (v_dst_base, _) = v_dst_cuda.device_ptr_raw();
            // #1082: prefer cuda_stream_raw() over candle_device().cuda_stream()
            // to avoid touching the candle device wrapper from kernel-crate FFI.
            // cuda_stream_raw() returns the same underlying CUstream cast as
            // *mut c_void; we re-cast back to sys::CUstream for cudarc's API.
            // CUstream cast uses the direct cudarc dep (no candle indirection).
            let raw_stream = k_dst_cuda.cuda_stream_raw()
                as cudarc::driver::sys::CUstream;
            unsafe {
                cudarc_result::memcpy_dtod_async(
                    k_dst_base + dst_byte_off as u64,
                    k_src_base,
                    total_bytes,
                    raw_stream,
                )
                .map_err(|e| anyhow::anyhow!("kt pkv fp8: memcpy_dtod k_pool: {e:?}"))?;
                cudarc_result::memcpy_dtod_async(
                    v_dst_base + dst_byte_off as u64,
                    v_src_base,
                    total_bytes,
                    raw_stream,
                )
                .map_err(|e| anyhow::anyhow!("kt pkv fp8: memcpy_dtod v_pool: {e:?}"))?;
            }
            return Ok(());
        }

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

        // #1082: prefer cuda_stream_raw() over candle_device().cuda_stream()
        // to avoid touching the candle device wrapper from kernel-crate FFI.
        // CUstream cast uses the direct cudarc dep (no candle indirection).
        let raw_stream = k_dst_cuda.cuda_stream_raw()
            as cudarc::driver::sys::CUstream;

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
    /// Replaces the candle `PagedKvCache::read`, for both
    /// the **contiguous-slot-run case** (block_table's positions
    /// `0..seq_len` map to a single contiguous slot range — fast
    /// path via `narrow`) and the **gather case** (positions map to
    /// arbitrary slots — slower path via
    /// `kiln_tensor::cuda_index_select_dim0`).
    ///
    /// FP8 caches are rejected here — the dequant kt-API entries
    /// aren't yet ported.
    ///
    /// **Cost:**
    /// - Fast path: one CUDA `Tensor::contiguous()` per pool (k+v).
    /// - Gather path: one H2D upload of `seq_len` u32 indices via
    ///   candle, plus one `cuda_index_select_dim0` per pool, plus
    ///   the same transpose+contiguous+unsqueeze tail as the fast
    ///   path.
    pub fn read(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<(KtTensor, KtTensor)> {
        let (k_pool, v_pool) = &self.layers[layer_idx];

        let (k_slice, v_slice) = if let Some(start_slot) =
            contiguous_slot_run_start(block_table, self.block_size, 0, seq_len)
        {
            // Fast path — single contiguous run; narrow is zero-copy.
            let k = k_pool
                .narrow(0, start_slot, seq_len)
                .map_err(|e| anyhow::anyhow!("kt pkv read: narrow k: {e}"))?;
            let v = v_pool
                .narrow(0, start_slot, seq_len)
                .map_err(|e| anyhow::anyhow!("kt pkv read: narrow v: {e}"))?;
            (k, v)
        } else {
            // Gather path — build u32 indices on host, upload via
            // candle, wrap as kt-Tensor, call cuda_index_select_dim0.
            let mut idx_data: Vec<u32> = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let slot = block_table.slot_for(pos, self.block_size).ok_or_else(|| {
                    anyhow::anyhow!("kt pkv read: no slot for position {pos} in block table")
                })?;
                let slot_u32 = u32::try_from(slot).map_err(|_| {
                    anyhow::anyhow!("kt pkv read: slot {slot} exceeds u32")
                })?;
                idx_data.push(slot_u32);
            }

            // Build the indices kt-Tensor: kt CPU H2D via the candle-
            // free host_to_cuda_copy_ctx (#1082). The previous candle
            // detour (build candle Tensor on the source device, then
            // borrow back) read .candle_device() off the k_pool
            // CudaStorage purely to satisfy candle's Tensor::new; the
            // kt path collapses that into a direct device_index lift.
            let device_index = match k_pool.device() {
                kiln_tensor::Device::Cuda(i) => i,
                other => {
                    return Err(anyhow::anyhow!(
                        "kt pkv read: k_pool must be CUDA device, got {other}"
                    ));
                }
            };
            let kt_indices_cpu = kiln_tensor::Tensor::from_slice(
                idx_data.as_slice(),
                vec![idx_data.len()],
            )
            .map_err(|e| anyhow::anyhow!("kt pkv read: build indices cpu: {e}"))?;
            let kt_indices =
                kiln_tensor::host_to_cuda_copy_ctx(&kt_indices_cpu, device_index)
                    .map_err(|e| anyhow::anyhow!("kt pkv read: H2D indices: {e}"))?;

            let k = kiln_tensor::cuda_index_select_dim0(k_pool, &kt_indices)
                .map_err(|e| anyhow::anyhow!("kt pkv read: index_select k_pool: {e}"))?;
            let v = kiln_tensor::cuda_index_select_dim0(v_pool, &kt_indices)
                .map_err(|e| anyhow::anyhow!("kt pkv read: index_select v_pool: {e}"))?;
            (k, v)
        };

        // FP8 path: the slice is U8 (E4M3FN bit pattern). Dequantize
        // back to the compute dtype before downstream attention. Narrow
        // can produce a view with non-zero start_offset, so we
        // materialize through `.contiguous()` first; the dequant kernel
        // is contiguous-only.
        let (k_slice, v_slice) = if self.fp8 {
            let k_c = k_slice
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 read: contiguous k: {e}"))?;
            let v_c = v_slice
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 read: contiguous v: {e}"))?;
            let k_deq = cuda_fp8_dequantize_direct(&k_c, self.compute_dtype)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 read: dequantize k: {e}"))?;
            let v_deq = cuda_fp8_dequantize_direct(&v_c, self.compute_dtype)
                .map_err(|e| anyhow::anyhow!("kt pkv fp8 read: dequantize v: {e}"))?;
            (k_deq, v_deq)
        } else {
            (k_slice, v_slice)
        };

        // [seq_len, num_kv_heads, head_dim] -> [num_kv_heads, seq_len, head_dim]
        let k_t = k_slice
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("kt pkv read: transpose k: {e}"))?;
        let v_t = v_slice
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("kt pkv read: transpose v: {e}"))?;

        // Materialize the transposed layout — the CUDA-side
        // contiguous kernel (PR #1374) kicks in here.
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
    /// slot index. Mirrors the host-slot (`new_len == 1`) path of
    /// [`Self::write_token_major_native`].
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

    /// Token-major multi-token writer. Replaces the candle
    /// `PagedKvCache::write_token_major_native` (#1082 candle-drop).
    ///
    /// - `k`, `v`: `[1, new_len, num_kv_heads, head_dim]` BF16
    ///
    /// Returns `Ok(false)` when the cache is FP8-backed so callers can
    /// fall back to [`Self::write`], which owns quantization.
    ///
    /// `new_len == 1` routes through the BF16 host-slot kernel
    /// (`paged_kv_write_token_major_bf16_kt`). For `new_len > 1` the
    /// `[new_len, num_kv_heads, head_dim]` block is `slice_set` into the
    /// pool — as one contiguous run when the block table resolves to a
    /// contiguous slot range, else row-by-row.
    pub fn write_token_major_native(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        start_pos: usize,
        k: &KtTensor,
        v: &KtTensor,
    ) -> Result<bool> {
        if self.fp8 {
            return Ok(false);
        }
        if k.dtype() != KtDType::BF16 || v.dtype() != KtDType::BF16 {
            return Ok(false);
        }

        let new_len = k
            .dim(1)
            .map_err(|e| anyhow::anyhow!("kt pkv token_major: k.dim(1): {e}"))?;
        let (k_pool, v_pool) = &self.layers[layer_idx];

        if new_len == 1 {
            let slot = block_table
                .slot_for(start_pos, self.block_size)
                .ok_or_else(|| {
                    anyhow::anyhow!("no slot for position {start_pos} in block table")
                })?;
            // The host-slot kernel reads `num_kv_heads * head_dim` contiguous
            // elements off the raw device pointer (no stride walk), so the
            // single-token rows must be contiguous — matching the candle path's
            // pre-kernel `.contiguous()` materialization.
            let k_c = k
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv token_major: contiguous k (1tok): {e}"))?;
            let v_c = v
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv token_major: contiguous v (1tok): {e}"))?;
            kiln_flash_attn::paged_kv_write_token_major_bf16_kt(k_pool, v_pool, &k_c, &v_c, slot)
                .map_err(|e| anyhow::anyhow!("kt paged_kv_write_token_major_bf16: {e}"))?;
            return Ok(true);
        }

        // [1, new_len, num_kv_heads, head_dim] -> [new_len, num_kv_heads, head_dim]
        let k_flat = k
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("kt pkv token_major: squeeze k: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv token_major: contiguous k: {e}"))?;
        let v_flat = v
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("kt pkv token_major: squeeze v: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv token_major: contiguous v: {e}"))?;

        if let Some(start_slot) =
            contiguous_slot_run_start(block_table, self.block_size, start_pos, new_len)
        {
            k_pool
                .slice_set(&k_flat, 0, start_slot)
                .map_err(|e| anyhow::anyhow!("kt pkv token_major: slice_set k run: {e}"))?;
            v_pool
                .slice_set(&v_flat, 0, start_slot)
                .map_err(|e| anyhow::anyhow!("kt pkv token_major: slice_set v run: {e}"))?;
            return Ok(true);
        }

        for i in 0..new_len {
            let pos = start_pos + i;
            let slot = block_table
                .slot_for(pos, self.block_size)
                .ok_or_else(|| anyhow::anyhow!("no slot for position {pos} in block table"))?;
            let k_row = k_flat
                .narrow(0, i, 1)
                .map_err(|e| anyhow::anyhow!("kt pkv token_major: narrow k row {i}: {e}"))?;
            let v_row = v_flat
                .narrow(0, i, 1)
                .map_err(|e| anyhow::anyhow!("kt pkv token_major: narrow v row {i}: {e}"))?;
            k_pool
                .slice_set(&k_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv token_major: slice_set k row {i}: {e}"))?;
            v_pool
                .slice_set(&v_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv token_major: slice_set v row {i}: {e}"))?;
        }

        Ok(true)
    }

    /// Batched token-major writer — one decode token per sequence.
    /// Replaces the candle
    /// `PagedKvCache::write_token_major_native_batch` (#1082 candle-drop).
    ///
    /// - `block_tables`: one page table per batch row
    /// - `start_positions`: absolute write position for each batch row
    /// - `k`, `v`: `[batch, 1, num_kv_heads, head_dim]` BF16
    ///
    /// Returns `Ok(false)` when the cache is FP8-backed so callers can
    /// fall back to [`Self::write`].
    ///
    /// LIVE prod path (forward.rs batched contiguous paged decode). The
    /// candle version's Metal branch is dropped here — this twin is
    /// CUDA-only — so the implementation is the per-row loop: each row
    /// is squeezed to `[1, 1, num_kv_heads, head_dim]` and handed to
    /// [`Self::write_token_major_native`], which uses the BF16 host-slot
    /// CUDA kernel.
    pub fn write_token_major_native_batch(
        &self,
        layer_idx: usize,
        block_tables: &[&BlockTable],
        start_positions: &[usize],
        k: &KtTensor,
        v: &KtTensor,
    ) -> Result<bool> {
        if self.fp8 {
            return Ok(false);
        }

        let (batch, seq_len, _heads, _head_dim) = k
            .dims4()
            .map_err(|e| anyhow::anyhow!("kt pkv batch: k.dims4: {e}"))?;
        anyhow::ensure!(
            seq_len == 1,
            "batched token-major KV writes require one decode token per row"
        );
        anyhow::ensure!(
            v.shape() == k.shape(),
            "batched token-major KV write K/V shape mismatch"
        );
        anyhow::ensure!(
            block_tables.len() == batch && start_positions.len() == batch,
            "batched token-major KV write metadata length mismatch"
        );

        // Validate every row resolves to a slot up front (parity with the
        // candle path's `contiguous_slot_run_starts` precheck), then write
        // each row through the host-slot kernel.
        if contiguous_slot_run_starts(block_tables, self.block_size, start_positions, 1).is_none() {
            anyhow::bail!("batched token-major KV write slot lookup failed");
        }

        for idx in 0..batch {
            let k_row = k
                .narrow(0, idx, 1)
                .map_err(|e| anyhow::anyhow!("kt pkv batch: narrow k row {idx}: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv batch: contiguous k row {idx}: {e}"))?;
            let v_row = v
                .narrow(0, idx, 1)
                .map_err(|e| anyhow::anyhow!("kt pkv batch: narrow v row {idx}: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv batch: contiguous v row {idx}: {e}"))?;
            self.write_token_major_native(
                layer_idx,
                block_tables[idx],
                start_positions[idx],
                &k_row,
                &v_row,
            )?;
        }

        Ok(true)
    }

    /// Batched graph-slot variant — one fused CUDA kernel launch writes
    /// every row. Replaces the candle
    /// `PagedKvCache::write_token_major_native_batch_graph_slot`
    /// (#1082 candle-drop).
    ///
    /// - `k`, `v`: `[batch, 1, num_kv_heads, head_dim]` BF16
    /// - `slots`: `[batch]` U32 device tensor (per-row destination slots)
    ///
    /// Safe under CUDA graph capture: the only per-replay-varying input
    /// is `slots`, refreshed via `update_cuda_scalar` outside the
    /// captured region. Returns `Ok(false)` when preconditions aren't met
    /// (FP8 pool, non-BF16 K/V, seq_len != 1, wrong `slots` shape/dtype)
    /// so callers fall back to the slower per-row path.
    ///
    /// LIVE prod path (forward.rs batched contiguous paged decode, the
    /// `kv_fused_batched_enabled()` branch). Routes through
    /// [`kiln_flash_attn::paged_kv_write_token_major_bf16_batch_slot_kt`],
    /// which writes a contiguous `[batch, num_kv_heads, head_dim]` block —
    /// so the seq_len=1 dim is squeezed before dispatch (the kernel's
    /// `element_count == batch * kv_heads * head_dim` check is then exact).
    pub fn write_token_major_native_batch_graph_slot(
        &self,
        layer_idx: usize,
        k: &KtTensor,
        v: &KtTensor,
        slots: &KtTensor,
    ) -> Result<bool> {
        if self.fp8 || k.dtype() != KtDType::BF16 || v.dtype() != KtDType::BF16 {
            return Ok(false);
        }
        // Expect [batch, 1, num_kv_heads, head_dim].
        let (batch, seq_len, _heads, _head_dim) = k
            .dims4()
            .map_err(|e| anyhow::anyhow!("kt pkv batch_slot: k.dims4: {e}"))?;
        if seq_len != 1 {
            return Ok(false);
        }
        if v.shape() != k.shape() {
            return Ok(false);
        }
        let slots_shape = slots.shape();
        if slots.dtype() != KtDType::U32 || slots_shape.len() != 1 || slots_shape[0] != batch {
            return Ok(false);
        }
        let (k_pool, v_pool) = &self.layers[layer_idx];
        // Collapse the seq_len=1 dim before dispatching so the fused
        // kernel sees a contiguous [batch, num_kv_heads, head_dim] block.
        let k_sq = k
            .squeeze(1)
            .map_err(|e| anyhow::anyhow!("kt pkv batch_slot: squeeze k: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv batch_slot: contiguous k: {e}"))?;
        let v_sq = v
            .squeeze(1)
            .map_err(|e| anyhow::anyhow!("kt pkv batch_slot: squeeze v: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv batch_slot: contiguous v: {e}"))?;
        kiln_flash_attn::paged_kv_write_token_major_bf16_batch_slot_kt(
            k_pool, v_pool, &k_sq, &v_sq, slots,
        )
        .map_err(|e| anyhow::anyhow!("kt paged_kv_write_token_major_bf16_batch_slot: {e}"))?;
        Ok(true)
    }

    /// Head-major generic write. Replaces the candle
    /// `PagedKvCache::write` (#1082 candle-drop).
    ///
    /// - `k`, `v`: `[1, num_kv_heads, new_len, head_dim]`
    ///
    /// Dispatches to the native (BF16) or FP8 (U8) slot-write path based
    /// on `self.fp8`. The head-major input is transposed to token-major
    /// `[new_len, num_kv_heads, head_dim]` before the per-slot writes,
    /// matching the candle reshape exactly.
    pub fn write(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        start_pos: usize,
        k: &KtTensor,
        v: &KtTensor,
    ) -> Result<()> {
        if self.fp8 {
            self.write_fp8(layer_idx, block_table, start_pos, k, v)
        } else {
            self.write_native(layer_idx, block_table, start_pos, k, v)
        }
    }

    fn write_native(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        start_pos: usize,
        k: &KtTensor,
        v: &KtTensor,
    ) -> Result<()> {
        let new_len = k
            .dim(2)
            .map_err(|e| anyhow::anyhow!("kt pkv write_native: k.dim(2): {e}"))?;
        let (k_pool, v_pool) = &self.layers[layer_idx];

        if new_len == 1 {
            let slot = block_table
                .slot_for(start_pos, self.block_size)
                .ok_or_else(|| {
                    anyhow::anyhow!("no slot for position {start_pos} in block table")
                })?;
            // [1, num_kv_heads, 1, head_dim] -> [num_kv_heads, head_dim]
            let k_row = k
                .squeeze(2)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: squeeze k: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: contiguous k: {e}"))?;
            let v_row = v
                .squeeze(2)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: squeeze v: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: contiguous v: {e}"))?;
            k_pool
                .slice_set(&k_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: slice_set k: {e}"))?;
            v_pool
                .slice_set(&v_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: slice_set v: {e}"))?;
            return Ok(());
        }

        // [1, num_kv_heads, new_len, head_dim] -> [new_len, num_kv_heads, head_dim]
        let (k_flat, v_flat) = self.head_major_to_token_major(k, v, "write_native")?;

        if let Some(start_slot) =
            contiguous_slot_run_start(block_table, self.block_size, start_pos, new_len)
        {
            k_pool
                .slice_set(&k_flat, 0, start_slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: slice_set k run: {e}"))?;
            v_pool
                .slice_set(&v_flat, 0, start_slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: slice_set v run: {e}"))?;
            return Ok(());
        }

        for i in 0..new_len {
            let pos = start_pos + i;
            let slot = block_table
                .slot_for(pos, self.block_size)
                .ok_or_else(|| anyhow::anyhow!("no slot for position {pos} in block table"))?;
            let k_row = k_flat
                .narrow(0, i, 1)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: narrow k row {i}: {e}"))?;
            let v_row = v_flat
                .narrow(0, i, 1)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: narrow v row {i}: {e}"))?;
            k_pool
                .slice_set(&k_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: slice_set k row {i}: {e}"))?;
            v_pool
                .slice_set(&v_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: slice_set v row {i}: {e}"))?;
        }

        Ok(())
    }

    fn write_fp8(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        start_pos: usize,
        k: &KtTensor,
        v: &KtTensor,
    ) -> Result<()> {
        let new_len = k
            .dim(2)
            .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: k.dim(2): {e}"))?;

        if new_len == 1 {
            let slot = block_table
                .slot_for(start_pos, self.block_size)
                .ok_or_else(|| {
                    anyhow::anyhow!("no slot for position {start_pos} in block table")
                })?;
            // [1, num_kv_heads, 1, head_dim] -> [num_kv_heads, head_dim], quantize, write.
            let k_sq = k
                .squeeze(2)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: squeeze k: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: contiguous k: {e}"))?;
            let v_sq = v
                .squeeze(2)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: squeeze v: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: contiguous v: {e}"))?;
            let k_q = cuda_fp8_quantize_direct(&k_sq)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: quantize k: {e}"))?;
            let v_q = cuda_fp8_quantize_direct(&v_sq)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: quantize v: {e}"))?;
            let (k_pool, v_pool) = &self.layers[layer_idx];
            k_pool
                .slice_set(&k_q, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: slice_set k: {e}"))?;
            v_pool
                .slice_set(&v_q, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: slice_set v: {e}"))?;
            return Ok(());
        }

        // [1, num_kv_heads, new_len, head_dim] -> [new_len, num_kv_heads, head_dim].
        // Direct FP8 conversion without per-tensor scaling: a shared pool can't
        // carry per-write scales (read dequantizes uniformly). E4M3FN's ±448
        // range covers normalized attention K/V (typically ±10).
        let (k_flat, v_flat) = self.head_major_to_token_major(k, v, "write_fp8")?;
        let k_q = cuda_fp8_quantize_direct(&k_flat)
            .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: quantize k block: {e}"))?;
        let v_q = cuda_fp8_quantize_direct(&v_flat)
            .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: quantize v block: {e}"))?;

        let (k_pool, v_pool) = &self.layers[layer_idx];

        if let Some(start_slot) =
            contiguous_slot_run_start(block_table, self.block_size, start_pos, new_len)
        {
            k_pool
                .slice_set(&k_q, 0, start_slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: slice_set k run: {e}"))?;
            v_pool
                .slice_set(&v_q, 0, start_slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: slice_set v run: {e}"))?;
            return Ok(());
        }

        for i in 0..new_len {
            let pos = start_pos + i;
            let slot = block_table
                .slot_for(pos, self.block_size)
                .ok_or_else(|| anyhow::anyhow!("no slot for position {pos} in block table"))?;
            let k_row = k_q
                .narrow(0, i, 1)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: narrow k row {i}: {e}"))?;
            let v_row = v_q
                .narrow(0, i, 1)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: narrow v row {i}: {e}"))?;
            k_pool
                .slice_set(&k_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: slice_set k row {i}: {e}"))?;
            v_pool
                .slice_set(&v_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: slice_set v row {i}: {e}"))?;
        }

        Ok(())
    }

    /// `[1, num_kv_heads, new_len, head_dim]` -> contiguous
    /// `[new_len, num_kv_heads, head_dim]` for both k and v. Shared by the
    /// multi-token `write_native` / `write_fp8` paths.
    fn head_major_to_token_major(
        &self,
        k: &KtTensor,
        v: &KtTensor,
        ctx: &str,
    ) -> Result<(KtTensor, KtTensor)> {
        let k_flat = k
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("kt pkv {ctx}: squeeze k: {e}"))?
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("kt pkv {ctx}: transpose k: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv {ctx}: contiguous k: {e}"))?;
        let v_flat = v
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("kt pkv {ctx}: squeeze v: {e}"))?
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("kt pkv {ctx}: transpose v: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv {ctx}: contiguous v: {e}"))?;
        Ok((k_flat, v_flat))
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
