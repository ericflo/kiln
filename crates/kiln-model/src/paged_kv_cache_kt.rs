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

// #1082 candle-drop / all-hardware: this module is available on every
// build, NOT just `--features cuda`. The struct, the metadata accessors
// (`block_size`/`num_blocks`/`num_layers`/`is_fp8`/`compute_dtype`/
// `pool_tensors`/`contiguous_slot_run_starts`), and the `new`/`new_with_fp8`
// constructors compile on all backends so the `&PagedKvCacheKt` type that
// `forward.rs`/`generate.rs`/`cuda_graph.rs`/`speculative.rs`/
// `vk_decode_resident.rs` thread through their (un-gated) signatures resolves
// everywhere. The Vulkan resident-decode path reads `block_size()`/
// `num_blocks()`/`pool_tensors()` off this type (the actual KV bytes live in
// `kiln_vulkan_kernel::VkPagedKvCache`), mirroring how the deleted candle
// `PagedKvCache` held CPU-resident tensors for the Vulkan backend.
//
// The CUDA graph methods remain CUDA-only. The FP8 KV write/read methods are
// now CUDA + ROCm (both route their on-device E4M3 quantize/dequantize through
// the shared `csrc/fp8.cu` via `fp8_quantize_direct_dev` / `..dequantize..`).
// Native BF16 writes and reads are device-parametric kt paths; Metal also has a
// batched decode writer wired through `backend::metal`.
use anyhow::Result;
#[cfg(feature = "cuda")]
use anyhow::Context;

// #1082: cudarc imported directly instead of through
// candle_core::cuda_backend::cudarc::*. The candle re-export is a pure
// pass-through to the cudarc crate (candle-core wraps cudarc verbatim),
// so this drops two candle imports without changing runtime behavior.
// Pattern lifted from kiln-tensor commit 4ee1b7f9 and kiln-blas commit
// 0d201199. As of the candle-drop (#1082), the public `PagedKvCacheKt::new*`
// surface takes a `device: kiln_tensor::Device` instead of an
// `Arc<candle_core::cuda_backend::CudaDevice>`; allocation routes on the
// model's *runtime* device (CPU → `zeros_cpu`, `Cuda(i)` → `cuda_zeros_ctx(i,
// ..)`, `Metal(i)` → `zeros_on(Device::Metal(i), ..)`) — so this file carries
// no candle import.
#[cfg(feature = "cuda")]
use cudarc::driver::result as cudarc_result;

use kiln_core::block::BlockTable;
// (#1082 DoD-100) device-agnostic slot math — used by the now-ungated
// `write_native` on CPU as well as the CUDA fast paths.
use kiln_core::block::contiguous_slot_run_start;
use kiln_core::block::contiguous_slot_run_starts;
#[cfg(feature = "cuda")]
use kiln_tensor::{
    cuda_fp8_dequantize_direct, cuda_fp8_quantize_direct, cuda_zeros_ctx, CudaStorage,
};
use kiln_tensor::{DType as KtDType, Layout, Tensor as KtTensor, TensorId};

/// Device-dispatched FP8 (E4M3FN, scale=1.0 "direct") quantize for the paged-KV
/// write path. CUDA and ROCm both route to their on-device kernel (the shared
/// `csrc/fp8.cu`); this is the seam that makes the FP8 KV cache work on ROCm at
/// the same on-device parity CUDA has (rather than the host round-trip the
/// generic `kiln_model::fp8` fallback would take).
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn fp8_quantize_direct_dev(t: &KtTensor) -> Result<KtTensor> {
    match t.device() {
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(_) => kiln_tensor::cuda_fp8_quantize_direct(t)
            .map_err(|e| anyhow::anyhow!("fp8 paged-KV quantize (cuda): {e}")),
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(_) => kiln_tensor::rocm_fp8_quantize_direct(t)
            .map_err(|e| anyhow::anyhow!("fp8 paged-KV quantize (rocm): {e}")),
        other => anyhow::bail!("fp8 paged-KV quantize: unsupported device {other:?}"),
    }
}

/// Device-dispatched FP8 (E4M3FN, scale=1.0 "direct") dequantize for the
/// paged-KV read path. Twin of [`fp8_quantize_direct_dev`].
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn fp8_dequantize_direct_dev(t: &KtTensor, target: KtDType) -> Result<KtTensor> {
    match t.device() {
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(_) => kiln_tensor::cuda_fp8_dequantize_direct(t, target)
            .map_err(|e| anyhow::anyhow!("fp8 paged-KV dequantize (cuda): {e}")),
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(_) => kiln_tensor::rocm_fp8_dequantize_direct(t, target)
            .map_err(|e| anyhow::anyhow!("fp8 paged-KV dequantize (rocm): {e}")),
        other => anyhow::bail!("fp8 paged-KV dequantize: unsupported device {other:?}"),
    }
}

/// Allocate one zero-filled `(k_pool, v_pool)` pair of shape `shape`
/// (`= [total_slots, num_kv_heads, head_dim]`, `n_elements` elements total) on
/// `device`, using the exact per-backend routing the cache constructor uses.
/// Shared by [`PagedKvCacheKt::new_with_fp8`] and
/// [`PagedKvCacheKt::physical_resize_to`] so the device matrix lives in ONE
/// place — a divergence between construct-time and resize-time allocation would
/// silently put the resized pool on the wrong device and trip the per-layer
/// `slice_set` device-mismatch guard.
fn alloc_pool_pair(
    device: kiln_tensor::Device,
    shape: &[usize],
    n_elements: usize,
    storage_dtype: KtDType,
    layer_idx: usize,
) -> Result<(KtTensor, KtTensor)> {
    let _i = layer_idx;
    let shape = shape.to_vec();
    let (k, v) = match device {
        kiln_tensor::Device::Cpu => {
            let _ = n_elements;
            let k = KtTensor::zeros_cpu(shape.clone(), storage_dtype);
            let v = KtTensor::zeros_cpu(shape.clone(), storage_dtype);
            (k, v)
        }
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(i) => {
            let k_storage = cuda_zeros_ctx(i, storage_dtype, n_elements)
                .with_context(|| format!("kt paged-kv: alloc k_pool layer {_i}"))?;
            let v_storage = cuda_zeros_ctx(i, storage_dtype, n_elements)
                .with_context(|| format!("kt paged-kv: alloc v_pool layer {_i}"))?;
            let k = KtTensor::from_parts(
                k_storage,
                Layout::contiguous(shape.clone()),
                TensorId::next(),
            )
            .with_context(|| format!("kt paged-kv: wrap k_pool layer {_i}"))?;
            let v = KtTensor::from_parts(
                v_storage,
                Layout::contiguous(shape.clone()),
                TensorId::next(),
            )
            .with_context(|| format!("kt paged-kv: wrap v_pool layer {_i}"))?;
            (k, v)
        }
        // #1082 ROCm: device-resident KV pools (mirror the CUDA arm),
        // gated on KILN_ROCM_PAGED_DECODE so it pairs with the native
        // sq=1 paged-decode routing in forward.rs (both default off until
        // the KV-cache correctness fix lands). When off, ROCm falls to the
        // `other` => CPU pool + the correct contiguous-decode path.
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(i) if crate::forward::rocm_paged_decode_enabled() => {
            let k_storage = kiln_tensor::rocm_zeros_ctx(i, storage_dtype, n_elements)
                .map_err(|e| anyhow::anyhow!("kt paged-kv: alloc k_pool (rocm) layer {_i}: {e}"))?;
            let v_storage = kiln_tensor::rocm_zeros_ctx(i, storage_dtype, n_elements)
                .map_err(|e| anyhow::anyhow!("kt paged-kv: alloc v_pool (rocm) layer {_i}: {e}"))?;
            let k = KtTensor::from_parts(
                k_storage,
                Layout::contiguous(shape.clone()),
                TensorId::next(),
            )
            .map_err(|e| anyhow::anyhow!("kt paged-kv: wrap k_pool (rocm) layer {_i}: {e}"))?;
            let v = KtTensor::from_parts(
                v_storage,
                Layout::contiguous(shape.clone()),
                TensorId::next(),
            )
            .map_err(|e| anyhow::anyhow!("kt paged-kv: wrap v_pool (rocm) layer {_i}: {e}"))?;
            (k, v)
        }
        #[cfg(feature = "metal")]
        kiln_tensor::Device::Metal(i) => {
            let _ = n_elements;
            let k = KtTensor::zeros_on(kiln_tensor::Device::Metal(i), shape.clone(), storage_dtype)
                .map_err(|e| anyhow::anyhow!("kt paged-kv: alloc k_pool (metal) layer {_i}: {e}"))?;
            let v = KtTensor::zeros_on(kiln_tensor::Device::Metal(i), shape.clone(), storage_dtype)
                .map_err(|e| anyhow::anyhow!("kt paged-kv: alloc v_pool (metal) layer {_i}: {e}"))?;
            (k, v)
        }
        // Vulkan and any backend whose feature isn't compiled in → host-resident
        // CPU pools (matches the cache constructor's `other` arm exactly).
        other => {
            let _ = other;
            let _ = n_elements;
            let k = KtTensor::zeros_cpu(shape.clone(), storage_dtype);
            let v = KtTensor::zeros_cpu(shape.clone(), storage_dtype);
            (k, v)
        }
    };
    Ok((k, v))
}

/// Block until all device work completes, so a subsequent pool drop can't free
/// storage a still-running kernel reads. No-op on CPU (synchronous) and on
/// backends whose feature isn't compiled in. Used by
/// [`PagedKvCacheKt::physical_resize_to`] (#26) as the C2 use-after-free guard.
fn sync_device_for_resize(device: kiln_tensor::Device) -> Result<()> {
    match device {
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(i) => kiln_tensor::cuda_synchronize_default_stream(i)
            .map_err(|e| anyhow::anyhow!("physical_resize_to: cuda sync: {e}")),
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(i) => kiln_tensor::rocm_synchronize_default_stream(i)
            .map_err(|e| anyhow::anyhow!("physical_resize_to: rocm sync: {e}")),
        _ => Ok(()),
    }
}

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
    ///
    /// Behind an `RwLock` so [`Self::physical_resize_to`] can swap the pools
    /// through `&self` (#26) — the cache is held as a shared `Arc` cloned into
    /// the decode paths, so a `&mut self` resizer would be unreachable during
    /// serving. The decode read path takes the (uncontended) READ lock briefly
    /// per access; the resizer takes the WRITE lock, and the caller guarantees
    /// no forward is in flight while it does (the actor barrier), so the write
    /// lock is never actually contended against a live kernel.
    layers: std::sync::RwLock<Vec<(KtTensor, KtTensor)>>,
    block_size: usize,
    /// Logical block count = `layers[i]` slot count / `block_size`. Atomic so
    /// `physical_resize_to` can update it through `&self` in lockstep with the
    /// `layers` swap. Read relaxed in the decode path.
    num_blocks: std::sync::atomic::AtomicUsize,
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
    /// `device` selects the device the pools are allocated on. The pools
    /// MUST live on the model's *runtime* device (not a compile-time
    /// feature-gated default) so the per-layer K/V `slice_set` writes match
    /// the model's tensors and don't trip `Tensor::slice_set: device
    /// mismatch`. CUDA routes through [`cuda_zeros_ctx`]; Metal through
    /// `zeros_on(Device::Metal, ..)`; CPU (and any GPU backend whose feature
    /// isn't compiled in, e.g. Vulkan whose kt pools are CPU-resident)
    /// through host-resident `zeros_cpu`.
    pub fn new(
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: KtDType,
        device: kiln_tensor::Device,
    ) -> Result<Self> {
        Self::new_with_fp8(
            num_full_attn_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype,
            device,
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
        device: kiln_tensor::Device,
        fp8: bool,
    ) -> Result<Self> {
        let storage_dtype = if fp8 { KtDType::U8 } else { dtype };
        let total_slots = num_blocks * block_size;
        let n_elements = total_slots * num_kv_heads * head_dim;
        let shape = vec![total_slots, num_kv_heads, head_dim];

        let mut layers = Vec::with_capacity(num_full_attn_layers);
        for _i in 0..num_full_attn_layers {
            // #1082 device-routing fix: allocate the pools on the model's
            // *runtime* device, matched at runtime (NOT a compile-time
            // feature-gated default). Each arm keeps its `#[cfg]` so only the
            // compiled-in backends' allocators are referenced. CPU pools are
            // `zeros_cpu`; CUDA pools route through `cuda_zeros_ctx(i, ..)`;
            // Metal pools through the kt `zeros_on(Device::Metal(i), ..)` UMA
            // path. Vulkan (kt vulkan tensors are CPU-resident) and any GPU
            // device whose backend feature isn't compiled in fall to the
            // host-resident `zeros_cpu` default — matching the prior
            // non-cuda/non-metal behavior, and exactly what the deleted candle
            // cache did for the Vulkan backend (it held CPU candle tensors).
            let (k, v) = alloc_pool_pair(device, &shape, n_elements, storage_dtype, _i)?;
            layers.push((k, v));
        }
        let fp8_scales = vec![(1.0_f32, 1.0_f32); num_full_attn_layers];
        Ok(Self {
            layers: std::sync::RwLock::new(layers),
            block_size,
            num_blocks: std::sync::atomic::AtomicUsize::new(num_blocks),
            fp8,
            fp8_scales,
            compute_dtype: dtype,
        })
    }

    /// Read-borrow the per-layer pools. Recovers from lock poisoning (a panic
    /// while reading leaves the data intact — reads don't mutate).
    #[inline]
    fn layers_read(&self) -> std::sync::RwLockReadGuard<'_, Vec<(KtTensor, KtTensor)>> {
        self.layers.read().unwrap_or_else(|e| e.into_inner())
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
        self.num_blocks.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The device the pools live on (layer 0's k pool), or `None` if the cache
    /// has no layers. Needed by [`Self::physical_resize_to`] callers that don't
    /// otherwise carry the device.
    pub fn device(&self) -> Option<kiln_tensor::Device> {
        self.layers_read().first().map(|(k, _)| k.device())
    }

    /// Storage bytes one block occupies across ALL layers and both K and V pools:
    /// `num_layers * 2 * block_size * num_kv_heads * head_dim * storage_dtype_size`.
    /// The memory governor's resize policy uses this to translate a VRAM target
    /// into a block-count target. `0` if the cache has no layers.
    pub fn bytes_per_block(&self) -> usize {
        let layers = self.layers_read();
        let Some((k, _)) = layers.first() else {
            return 0;
        };
        let dims = k.dims();
        if dims.len() != 3 {
            return 0;
        }
        let per_slot = dims[1] * dims[2]; // num_kv_heads * head_dim
        let dtype_size = k.dtype().size_in_bytes().max(1);
        layers.len() * 2 * self.block_size * per_slot * dtype_size
    }

    pub fn num_layers(&self) -> usize {
        self.layers_read().len()
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

    /// Cheap owned clones of the `(k_pool, v_pool)` kt-Tensors for `layer_idx`
    /// (an `Arc` bump each — the device buffers are shared, not copied).
    ///
    /// Returns OWNED tensors rather than borrows because the pools live behind a
    /// `RwLock` (#26 resize): a borrow couldn't outlive the read guard. Holding
    /// the clone keeps that pool's storage alive even across a concurrent
    /// `physical_resize_to` swap — though the caller (decode) is barrier-ordered
    /// against resize anyway.
    pub fn pool_tensors(&self, layer_idx: usize) -> Option<(KtTensor, KtTensor)> {
        self.layers_read()
            .get(layer_idx)
            .map(|(k, v)| (k.clone(), v.clone()))
    }

    /// Physically reallocate every layer's `(k_pool, v_pool)` to back
    /// `new_num_blocks` blocks, copying the surviving prefix and dropping the old
    /// pools. This is the elastic actuator that lets inference physically
    /// surrender KV VRAM and reclaim it — the physical half of the dynamic
    /// resize whose logical half is [`kiln_core::block::BlockManager`].
    ///
    /// WHY THIS RECLAIMS (same-process arbitration): kiln runs inference AND
    /// training in ONE process / ONE device memory pool. Dropping the old pool
    /// returns its bytes to that pool, where the NEXT allocation (e.g. a training
    /// run, or a regrown KV pool) REUSES them — verified on gfx1151: a 2 GB
    /// alloc → drop → 2 GB realloc grows peak VRAM by 0 MB. It does NOT depend on
    /// returning memory to the OS (`hipMemPoolTrimTo` is a measured no-op on
    /// gfx1151/ROCm 7.2.4 — so is the sync `hipFree`); cross-process return is a
    /// platform limitation we don't rely on. Reuse within the pool is what makes
    /// "training and inference share VRAM, never OOM" work.
    ///
    /// SAFETY CONTRACT — the caller MUST guarantee that NO forward/decode pass is
    /// in flight (the pool tensors are swapped; a kernel reading the old pool
    /// would use freed storage). We additionally device-synchronize at entry to
    /// flush any already-submitted kernel before the first drop (defends against
    /// off-stream HIP-graph-replay kernels). On SHRINK the caller must have run
    /// the paired `BlockManager::physical_truncate(new_num_blocks)` first, so no
    /// live block sits in the truncated tail. Surviving KV (slots
    /// `[0, copy_slots)`) is copied verbatim; the slot mapping
    /// `slot = block_id*block_size + off` is preserved, so existing `BlockTable`s
    /// stay valid with NO remapping.
    ///
    /// SHRINK vs GROW commit strategy:
    /// - SHRINK swaps pools ONE LAYER AT A TIME, dropping each old pool the
    ///   instant its data is copied so the next layer's (smaller) alloc reuses
    ///   the freed bytes. Peak transient VRAM is bounded to `old + one_layer`,
    ///   NOT `old + new` — a SHRINK runs when memory is tight, so a
    ///   build-all-then-swap would spike VRAM exactly when we can least afford
    ///   it. SHRINK is effectively atomic: only the layer-0 alloc can fail, and
    ///   it fails before any mutation (every later, smaller alloc reuses freed
    ///   bytes), so `self` is untouched on `Err`.
    /// - GROW stages all new pools into a local vec and swaps them in only after
    ///   every layer succeeds (ATOMIC: `Err` leaves `self` wholly on the old
    ///   pools). A growing per-layer swap could fail mid-loop and leave the cache
    ///   with mixed-size layers — corrupting the slot mapping. GROW peak is
    ///   `old + new`, which is fine: GROW only runs when memory is NOT tight (the
    ///   caller pre-checks `available_bytes`).
    ///
    /// No-op (returns `Ok`) when `new_num_blocks == num_blocks`.
    pub fn physical_resize_to(
        &self,
        new_num_blocks: usize,
        device: kiln_tensor::Device,
    ) -> Result<()> {
        let cur = self.num_blocks();
        if new_num_blocks == cur {
            return Ok(());
        }
        // C2: flush any in-flight kernel before we drop the old pools. The caller
        // guarantees no NEW launches during the resize (actor barrier); this
        // covers a kernel already submitted on another stream (graph replay).
        sync_device_for_resize(device)?;

        let storage_dtype = if self.fp8 {
            KtDType::U8
        } else {
            self.compute_dtype
        };
        let new_total_slots = new_num_blocks * self.block_size;
        let old_total_slots = cur * self.block_size;
        let copy_slots = new_total_slots.min(old_total_slots);
        let growing = new_num_blocks > cur;

        // Exclusive access for the structural swap. The caller's barrier (no
        // forward in flight) guarantees this write lock is never contended
        // against a live decode read, and that no kernel is mid-read of a pool we
        // are about to drop.
        let mut layers = self.layers.write().unwrap_or_else(|e| e.into_inner());

        // Allocate the new pool for layer `i` and copy its surviving prefix from
        // the current (old) pool at `pools[i]`. Pure: does not mutate `pools`.
        let make_resized =
            |pools: &Vec<(KtTensor, KtTensor)>, layer_idx: usize| -> Result<(KtTensor, KtTensor)> {
                let dims = pools[layer_idx].0.dims().to_vec();
                anyhow::ensure!(
                    dims.len() == 3,
                    "physical_resize_to: layer {layer_idx} pool has rank {} (want 3)",
                    dims.len()
                );
                let shape = vec![new_total_slots, dims[1], dims[2]];
                let n_elements = new_total_slots * dims[1] * dims[2];
                let (new_k, new_v) =
                    alloc_pool_pair(device, &shape, n_elements, storage_dtype, layer_idx)?;
                // The new pool's async zero-fill must COMPLETE before we copy the
                // surviving prefix over it — otherwise a large (grow) zero-fill
                // can land AFTER the copy and wipe it. Same-stream ordering is not
                // sufficient in practice here, so synchronize explicitly. Resize
                // is rare, so this is cheap relative to the multi-layer realloc.
                if copy_slots > 0 {
                    sync_device_for_resize(device)?;
                    let (old_k, old_v) = &pools[layer_idx];
                    let src_k = old_k.narrow(0, 0, copy_slots).map_err(|e| {
                        anyhow::anyhow!("physical_resize_to: narrow k l{layer_idx}: {e}")
                    })?;
                    let src_v = old_v.narrow(0, 0, copy_slots).map_err(|e| {
                        anyhow::anyhow!("physical_resize_to: narrow v l{layer_idx}: {e}")
                    })?;
                    new_k.slice_set(&src_k, 0, 0).map_err(|e| {
                        anyhow::anyhow!("physical_resize_to: copy k l{layer_idx}: {e}")
                    })?;
                    new_v.slice_set(&src_v, 0, 0).map_err(|e| {
                        anyhow::anyhow!("physical_resize_to: copy v l{layer_idx}: {e}")
                    })?;
                }
                Ok((new_k, new_v))
            };

        if growing {
            // ATOMIC: stage all, then commit. `?` on any layer leaves self intact.
            let mut staged: Vec<(KtTensor, KtTensor)> = Vec::with_capacity(layers.len());
            for layer_idx in 0..layers.len() {
                staged.push(make_resized(&layers, layer_idx)?);
            }
            *layers = staged;
        } else if copy_slots == 0 {
            // Degenerate shrink-to-zero (no surviving KV): just allocate the new
            // (empty/tiny) pools per layer.
            for layer_idx in 0..layers.len() {
                layers[layer_idx] = make_resized(&layers, layer_idx)?;
            }
        } else {
            // SHRINK — HOST-STAGED so device VRAM is STRICTLY NON-INCREASING (#38).
            // For a shrink the surviving prefix `[0, copy_slots)` IS the entire new
            // pool (copy_slots == new_total_slots), so per layer we: (1) D2H the
            // K/V prefix into host RAM, (2) DROP the old pool — freeing its VRAM
            // NOW, (3) H2D the prefix back as the new, smaller device pool. Peak
            // device VRAM per layer never exceeds the OLD layer size — no one-layer
            // overshoot, unlike the D2D alloc-then-drop which transiently held
            // `old + new`. Cost: one D2H + H2D per layer; resize is rare.
            debug_assert_eq!(
                copy_slots, new_total_slots,
                "shrink: surviving prefix must be the whole new pool"
            );
            let cpu = kiln_tensor::Device::Cpu;
            for layer_idx in 0..layers.len() {
                sync_device_for_resize(device)?;
                // 1. D2H the surviving K/V prefix (independent host copies).
                let host_k = layers[layer_idx]
                    .0
                    .narrow(0, 0, copy_slots)
                    .and_then(|p| p.contiguous())
                    .and_then(|p| p.to_device(cpu))
                    .map_err(|e| anyhow::anyhow!("physical_resize_to: D2H k l{layer_idx}: {e}"))?;
                let host_v = layers[layer_idx]
                    .1
                    .narrow(0, 0, copy_slots)
                    .and_then(|p| p.contiguous())
                    .and_then(|p| p.to_device(cpu))
                    .map_err(|e| anyhow::anyhow!("physical_resize_to: D2H v l{layer_idx}: {e}"))?;
                // 2. Drop the old layer's VRAM by overwriting with the host copies.
                layers[layer_idx] = (host_k, host_v);
                sync_device_for_resize(device)?;
                // 3. H2D the prefix back as the new (smaller) device pool.
                let new_k = layers[layer_idx]
                    .0
                    .to_device(device)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| anyhow::anyhow!("physical_resize_to: H2D k l{layer_idx}: {e}"))?;
                let new_v = layers[layer_idx]
                    .1
                    .to_device(device)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| anyhow::anyhow!("physical_resize_to: H2D v l{layer_idx}: {e}"))?;
                layers[layer_idx] = (new_k, new_v);
            }
        }
        self.num_blocks
            .store(new_num_blocks, std::sync::atomic::Ordering::Relaxed);
        Ok(())
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
    /// The FP8 path quantizes the BF16 K/V into U8 (E4M3FN, scale = 1.0
    /// "direct" — per-slot scaling is not practical for a shared pool) and
    /// writes the rows into the U8 pool at `slot`. CUDA reads the `[1]` U32
    /// `slot` host-side (a 4-byte D2H) and `slice_set`s — which forces a sync,
    /// so FP8 + CUDA-graph capture is unsupported there. ROCm instead consumes
    /// the slot ON-DEVICE and scatters via `rocm_index_copy_dim0` (device_ptr,
    /// never `slice()`), so the FP8 graph-slot write records into a captured HIP
    /// decode graph — FP8 caches ARE capturable on ROCm (see the `rocm` arm).
    #[cfg(any(feature = "cuda", feature = "rocm"))]
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
        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];

        if self.fp8 {
            // CUDA: read the device slot index host-side (one 4-byte D2H), then
            // quantize + slice_set into the U8 pool. The host slot read forces a
            // sync, so on CUDA this path is NOT HIP/CUDA-graph-recordable (FP8 +
            // capture is unsupported there).
            #[cfg(feature = "cuda")]
            {
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
                let k_q = fp8_quantize_direct_dev(&k_sq)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: quantize k: {e}"))?;
                let v_q = fp8_quantize_direct_dev(&v_sq)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: quantize v: {e}"))?;
                k_pool
                    .slice_set(&k_q, 0, slot_idx)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: slice_set k: {e}"))?;
                v_pool
                    .slice_set(&v_q, 0, slot_idx)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: slice_set v: {e}"))?;
                return Ok(true);
            }
            // ROCm: CAPTURE-SAFE FP8 graph-slot write. Unlike CUDA, the slot index
            // is consumed ON-DEVICE (no host read), and the quantized U8 rows are
            // scattered into the pool via `rocm_index_copy_dim0` (writes through
            // device_ptr_raw, never `slice()`) — so the whole write records into a
            // captured HIP decode graph and is safe on the Borrowed freeze-pointer
            // arena buffers that `rocm_fp8_quantize_direct` mints under capture.
            // This is the seam that makes FP8 + HIP graph capture work on ROCm
            // where it can't on CUDA. Mirrors the BF16 device-slot scatter.
            #[cfg(all(feature = "rocm", not(feature = "cuda")))]
            {
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
                let k_q = fp8_quantize_direct_dev(&k_sq)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: quantize k: {e}"))?;
                let v_q = fp8_quantize_direct_dev(&v_sq)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: quantize v: {e}"))?;
                // Flatten pool -> [n_rows, row_elems] and each quantized U8 row ->
                // [1, row_elems], then `pool[*slot] = row` with the DEVICE slot.
                let row_elems = k_q.element_count();
                let kp_rows = k_pool.element_count() / row_elems;
                let vp_rows = v_pool.element_count() / row_elems;
                let k_pool2 = k_pool
                    .reshape(vec![kp_rows, row_elems])
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: reshape k_pool: {e}"))?;
                let v_pool2 = v_pool
                    .reshape(vec![vp_rows, row_elems])
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: reshape v_pool: {e}"))?;
                let k_row = k_q
                    .reshape(vec![1, row_elems])
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: reshape k_q: {e}"))?;
                let v_row = v_q
                    .reshape(vec![1, row_elems])
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: reshape v_q: {e}"))?;
                let slot1 = slot
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: contiguous slot: {e}"))?
                    .reshape(vec![1])
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: reshape slot: {e}"))?;
                kiln_tensor::rocm_index_copy_dim0(&k_pool2, &slot1, &k_row)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: scatter k: {e}"))?;
                kiln_tensor::rocm_index_copy_dim0(&v_pool2, &slot1, &v_row)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 graph_slot: scatter v: {e}"))?;
                return Ok(true);
            }
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
    #[cfg(feature = "cuda")]
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
        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];
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
    // (#1082 DoD-100) Device-agnostic: the contiguous fast path (`narrow`) and
    // the transpose/contiguous/unsqueeze tail are pure kt ops; the gather path
    // uses the device-agnostic `index_select` (only the index H2D is CUDA). FP8
    // dequant is CUDA-only. Restores the CPU paged read the flip's defensive
    // `cfg(cuda)` gate had broken.
    pub fn read(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<(KtTensor, KtTensor)> {
        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];

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
            // Gather path: build the slot indices directly on the pool
            // device so `index_select` can stay backend-local. Metal in
            // particular requires the index tensor to be Metal-resident;
            // using a CPU index against Metal K/V pools trips dispatch2's
            // mixed-device guard before its Metal gather kernel can run.
            let mut idx_data: Vec<u32> = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let slot = block_table.slot_for(pos, self.block_size).ok_or_else(|| {
                    anyhow::anyhow!("kt pkv read: no slot for position {pos} in block table")
                })?;
                let slot_u32 = u32::try_from(slot)
                    .map_err(|_| anyhow::anyhow!("kt pkv read: slot {slot} exceeds u32"))?;
                idx_data.push(slot_u32);
            }

            let kt_indices =
                kiln_tensor::Tensor::from_vec_on(k_pool.device(), idx_data, vec![seq_len])
                    .map_err(|e| {
                        anyhow::anyhow!("kt pkv read: build indices on {}: {e}", k_pool.device())
                    })?;
            let k = k_pool
                .index_select(&kt_indices, 0)
                .map_err(|e| anyhow::anyhow!("kt pkv read: index_select k_pool: {e}"))?;
            let v = v_pool
                .index_select(&kt_indices, 0)
                .map_err(|e| anyhow::anyhow!("kt pkv read: index_select v_pool: {e}"))?;
            (k, v)
        };

        // FP8 path: the slice is U8 (E4M3FN bit pattern). Dequantize
        // back to the compute dtype before downstream attention. Narrow
        // can produce a view with non-zero start_offset, so we
        // materialize through `.contiguous()` first; the dequant kernel
        // is contiguous-only.
        let (k_slice, v_slice) = if self.fp8 {
            // FP8 dequant uses the on-device E4M3 kernel (shared csrc/fp8.cu) on
            // CUDA and ROCm; other backends support native BF16 paged-KV only.
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            {
                let k_c = k_slice
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 read: contiguous k: {e}"))?;
                let v_c = v_slice
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 read: contiguous v: {e}"))?;
                let k_deq = fp8_dequantize_direct_dev(&k_c, self.compute_dtype)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 read: dequantize k: {e}"))?;
                let v_deq = fp8_dequantize_direct_dev(&v_c, self.compute_dtype)
                    .map_err(|e| anyhow::anyhow!("kt pkv fp8 read: dequantize v: {e}"))?;
                (k_deq, v_deq)
            }
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                anyhow::bail!(
                    "fp8 paged-KV read dequant is CUDA/ROCm-only; the other backends \
                     support the native BF16 paged-KV read only"
                )
            }
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
    #[cfg(feature = "cuda")]
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
        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];
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
    #[cfg(any(feature = "cuda", feature = "rocm"))]
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
        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];

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
    /// Returns `Ok(false)` when the cache is FP8-backed so callers can decide
    /// whether to fall back.
    ///
    /// LIVE prod path (forward.rs batched contiguous paged decode). CUDA keeps
    /// the per-row host-slot kernel path. Metal routes through the existing
    /// batched token-major writer kernel. Other native BF16 placements fall
    /// back to the generic head-major kt writer.
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

        let mut slots = Vec::with_capacity(batch);
        for idx in 0..batch {
            let slot = block_tables[idx]
                .slot_for(start_positions[idx], self.block_size)
                .ok_or_else(|| {
                    anyhow::anyhow!("batched token-major KV write slot lookup failed for row {idx}")
                })?;
            let slot_u32 = u32::try_from(slot).map_err(|_| {
                anyhow::anyhow!("batched token-major KV write slot {slot} exceeds u32")
            })?;
            slots.push(slot_u32);
        }

        #[cfg(feature = "metal")]
        if matches!(k.device(), kiln_tensor::Device::Metal(_)) {
            let pools = self.layers_read();
            let (k_pool, v_pool) = &pools[layer_idx];
            let slots_tensor = KtTensor::from_vec_on(k.device(), slots, vec![batch])
                .map_err(|e| anyhow::anyhow!("kt pkv metal batch: build slots: {e}"))?;
            let k_c = k
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv metal batch: contiguous k: {e}"))?;
            let v_c = v
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv metal batch: contiguous v: {e}"))?;
            if crate::backend::metal::metal_paged_kv_write_token_major_batch_supports(
                k_pool,
                v_pool,
                &slots_tensor,
                &k_c,
                &v_c,
            ) {
                crate::backend::metal::metal_paged_kv_write_token_major_batch_bf16(
                    k_pool,
                    v_pool,
                    &slots_tensor,
                    &k_c,
                    &v_c,
                )
                .map_err(|e| anyhow::anyhow!("kt pkv metal batch write: {e}"))?;
                return Ok(true);
            }
        }

        #[cfg(feature = "cuda")]
        if matches!(k.device(), kiln_tensor::Device::Cuda(_)) {
            // Validate every row resolves to a slot up front (parity with the
            // candle path's `contiguous_slot_run_starts` precheck), then write
            // each row through the host-slot kernel.
            if contiguous_slot_run_starts(block_tables, self.block_size, start_positions, 1)
                .is_none()
            {
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

            return Ok(true);
        }

        // ROCm: use the same fast per-row token-major writer as CUDA
        // (`write_token_major_native` -> `paged_kv_write_token_major_bf16_rocm`,
        // an in-place device-to-device copy at the host-resolved slot). Without
        // this, ROCm fell through to the generic head-major `self.write` scatter
        // below, which transposes K/V to [b,hk,1,d] and host-stages them
        // ([1,4,1,256] was the dominant decode H2D, one per attention layer).
        // The transpose(1,2) on the sq==1 dim is a data no-op, so the pool bytes
        // are identical to the generic path — just no host round-trip.
        #[cfg(feature = "rocm")]
        if matches!(k.device(), kiln_tensor::Device::Rocm(_)) {
            // write_token_major_native resolves each row's slot independently
            // (block_table.slot_for), so no contiguous-run precheck is needed —
            // any slot layout is handled. (The CUDA branch above keeps the
            // precheck for parity with its host-slot kernel contract.)
            for idx in 0..batch {
                let k_row = k
                    .narrow(0, idx, 1)
                    .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: narrow k row {idx}: {e}"))?
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: contiguous k row {idx}: {e}"))?;
                let v_row = v
                    .narrow(0, idx, 1)
                    .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: narrow v row {idx}: {e}"))?
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: contiguous v row {idx}: {e}"))?;
                self.write_token_major_native(
                    layer_idx,
                    block_tables[idx],
                    start_positions[idx],
                    &k_row,
                    &v_row,
                )?;
            }
            return Ok(true);
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
            let k_head = k_row
                .transpose(1, 2)
                .map_err(|e| anyhow::anyhow!("kt pkv batch: transpose k row {idx}: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv batch: contiguous k head row {idx}: {e}"))?;
            let v_head = v_row
                .transpose(1, 2)
                .map_err(|e| anyhow::anyhow!("kt pkv batch: transpose v row {idx}: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv batch: contiguous v head row {idx}: {e}"))?;
            self.write(
                layer_idx,
                block_tables[idx],
                start_positions[idx],
                &k_head,
                &v_head,
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
    #[cfg(feature = "cuda")]
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
        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];
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
    // (#1082 DoD-100) Device-agnostic: the native BF16 paged-KV write is pure
    // kt ops (squeeze/transpose/contiguous + dim-0 `slice_set`, which has a CPU
    // path) + host-side slot math (`kiln_core::block`), so it runs on CPU too.
    // Only the FP8 path is CUDA-only (the E4M3 quant kernel). This restores the
    // CPU paged forward path the flip's defensive `cfg(cuda)` gate had broken.
    pub fn write(
        &self,
        layer_idx: usize,
        block_table: &BlockTable,
        start_pos: usize,
        k: &KtTensor,
        v: &KtTensor,
    ) -> Result<()> {
        if self.fp8 {
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            {
                self.write_fp8(layer_idx, block_table, start_pos, k, v)
            }
            #[cfg(not(any(feature = "cuda", feature = "rocm")))]
            {
                let _ = (layer_idx, block_table, start_pos, k, v);
                anyhow::bail!(
                    "fp8 paged-KV write is CUDA/ROCm-only (on-device fp8 quantize); the \
                     other backends support the native BF16 paged-KV write only"
                )
            }
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
        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];

        // (#1082 Vulkan/non-CUDA) The pool is allocated in the cache's storage
        // dtype (BF16 for Qwen3.5-4B — matches the CUDA cache + the memory
        // budget). The non-CUDA forward can hand us F32 K/V (the Vulkan
        // attention path computes K/V projections in F32). `slice_set` requires
        // matching dtypes, so cast K/V to the pool dtype before scattering —
        // mirroring the CUDA fast paths that require BF16 K/V. A no-op clone
        // (Arc bump) when the dtypes already match (e.g. CPU BF16 fixtures).
        let k_cast;
        let k = if k.dtype() != k_pool.dtype() {
            k_cast = k
                .to_dtype(k_pool.dtype())
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: cast k to pool dtype: {e}"))?;
            &k_cast
        } else {
            k
        };
        let v_cast;
        let v = if v.dtype() != v_pool.dtype() {
            v_cast = v
                .to_dtype(v_pool.dtype())
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: cast v to pool dtype: {e}"))?;
            &v_cast
        } else {
            v
        };

        // Align K/V to the pool's device so the dim-0 `slice_set` scatter runs
        // on one device. On the non-CUDA paths a host-staged cast/op can land
        // K/V on CPU while the pool is on-device (Vulkan mirrors a CPU seed cache
        // into VkPagedKvCache; ROCm produces K/V on-device) — move the small
        // per-token K/V rows to the pool device. No-op when already co-located.
        // (R.4 E2E)
        let k_dev;
        let k = if k.device() != k_pool.device() {
            k_dev = k
                .to_device(k_pool.device())
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: move k to pool device: {e}"))?;
            &k_dev
        } else {
            k
        };
        let v_dev;
        let v = if v.device() != v_pool.device() {
            v_dev = v
                .to_device(v_pool.device())
                .map_err(|e| anyhow::anyhow!("kt pkv write_native: move v to pool device: {e}"))?;
            &v_dev
        } else {
            v
        };

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

    #[cfg(any(feature = "cuda", feature = "rocm"))]
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
            let k_q = fp8_quantize_direct_dev(&k_sq)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: quantize k: {e}"))?;
            let v_q = fp8_quantize_direct_dev(&v_sq)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: quantize v: {e}"))?;
            let pools = self.layers_read();
            let (k_pool, v_pool) = &pools[layer_idx];
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
        let k_q = fp8_quantize_direct_dev(&k_flat)
            .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: quantize k block: {e}"))?;
        let v_q = fp8_quantize_direct_dev(&v_flat)
            .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: quantize v block: {e}"))?;

        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];

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
    /// (#1082 DoD-100) Device-agnostic (pure kt squeeze/transpose/contiguous) —
    /// ungated so the CPU `write_native` path can call it.
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
            layers: std::sync::RwLock::new(Vec::new()),
            block_size: 16,
            num_blocks: std::sync::atomic::AtomicUsize::new(1024),
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

    // Physical resize correctness on CPU pools (device-agnostic copy logic; the
    // ROCm/CUDA paths reuse the SAME narrow+slice_set, validated on-box). Proves
    // the elastic actuator (#26) preserves live KV byte-for-byte across a
    // shrink and a grow, with the grown tail zero-filled.
    #[test]
    fn physical_resize_preserves_surviving_kv_and_zeros_grown_tail() {
        let block_size = 2usize;
        let kv_heads = 1usize;
        let head_dim = 2usize;
        let per_slot = kv_heads * head_dim; // 2 f32 per slot
        let dev = kiln_tensor::Device::Cpu;

        // 4 blocks * 2 = 8 slots; 1 full-attn layer; F32 so values round-trip exact.
        let mut cache = PagedKvCacheKt::new(1, 4, block_size, kv_heads, head_dim, KtDType::F32, dev)
            .expect("construct cpu cache");
        assert_eq!(cache.num_blocks(), 4);

        // Write a known pattern into the first 6 slots of layer 0's K and V pools
        // (slots 6,7 stay zero). Pattern: k[slot,i] = slot*10 + i ; v = k + 100.
        let n_known_slots = 6usize;
        let mut k_vals = Vec::with_capacity(n_known_slots * per_slot);
        let mut v_vals = Vec::with_capacity(n_known_slots * per_slot);
        for slot in 0..n_known_slots {
            for i in 0..per_slot {
                let base = (slot * 10 + i) as f32;
                k_vals.push(base);
                v_vals.push(base + 100.0);
            }
        }
        let k_src =
            KtTensor::from_vec(k_vals.clone(), vec![n_known_slots, kv_heads, head_dim]).unwrap();
        let v_src =
            KtTensor::from_vec(v_vals.clone(), vec![n_known_slots, kv_heads, head_dim]).unwrap();
        {
            let (k_pool, v_pool) = cache.pool_tensors(0).unwrap();
            k_pool.slice_set(&k_src, 0, 0).unwrap();
            v_pool.slice_set(&v_src, 0, 0).unwrap();
        }

        // SHRINK 4 -> 3 blocks (8 -> 6 slots). copy_slots = 6, so all known KV
        // survives verbatim and the (zero) tail is dropped.
        cache.physical_resize_to(3, dev).expect("shrink");
        assert_eq!(cache.num_blocks(), 3);
        let (k_pool, v_pool) = cache.pool_tensors(0).unwrap();
        assert_eq!(k_pool.dims(), &[6, kv_heads, head_dim]);
        let k_after: Vec<f32> = k_pool.to_vec().unwrap();
        let v_after: Vec<f32> = v_pool.to_vec().unwrap();
        assert_eq!(k_after, k_vals, "shrink must preserve K slots 0..6");
        assert_eq!(v_after, v_vals, "shrink must preserve V slots 0..6");

        // GROW 3 -> 5 blocks (6 -> 10 slots). Prefix preserved, new tail zeroed.
        cache.physical_resize_to(5, dev).expect("grow");
        assert_eq!(cache.num_blocks(), 5);
        let (k_pool, _) = cache.pool_tensors(0).unwrap();
        assert_eq!(k_pool.dims(), &[10, kv_heads, head_dim]);
        let k_grown: Vec<f32> = k_pool.to_vec().unwrap();
        assert_eq!(&k_grown[..k_vals.len()], &k_vals[..], "grow preserves prefix");
        assert!(
            k_grown[k_vals.len()..].iter().all(|&x| x == 0.0),
            "grown tail must be zero-filled"
        );

        // No-op resize returns Ok and changes nothing.
        cache.physical_resize_to(5, dev).expect("noop");
        assert_eq!(cache.num_blocks(), 5);
    }
}
