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
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};

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
#[cfg(any(feature = "cuda", feature = "rocm"))]
use kiln_tensor::Layout;
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
use kiln_tensor::TensorId;
#[cfg(feature = "cuda")]
use kiln_tensor::{CudaStorage, cuda_fp8_quantize_direct, cuda_zeros_ctx};
use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

static NEXT_KV_POOL_ALLOCATION_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity for the physical tensors backing one paged-KV cache.
///
/// `allocation_id` distinguishes separately constructed caches. `generation`
/// advances only after a failure-atomic physical resize commits, so graph
/// runners can reject replay against pointers from an older pool.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvPoolIdentity {
    pub allocation_id: u64,
    pub generation: u64,
    pub num_blocks: usize,
}

fn next_kv_pool_allocation_id() -> Result<u64> {
    NEXT_KV_POOL_ALLOCATION_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| anyhow::anyhow!("paged KV pool allocation identity exhausted"))
}

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

#[derive(Clone, Copy)]
enum KvPoolKind {
    Key,
    Value,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KvPoolAllocationReason {
    InitialCache,
    PhysicalResize,
}

impl KvPoolAllocationReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InitialCache => "initial_kv_cache",
            Self::PhysicalResize => "kv_physical_resize",
        }
    }
}

impl KvPoolKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Key => "k_pool",
            Self::Value => "v_pool",
        }
    }
}

/// Allocate one pool tensor of shape `shape` on `device`, using the exact
/// per-backend routing shared by construction and resize. ROCm replacement
/// pools may skip initialization because their live prefix is overwritten and
/// their free tail is not addressable until the block manager assigns and the
/// cache writer initializes those slots.
fn alloc_pool_tensor(
    device: kiln_tensor::Device,
    shape: &[usize],
    n_elements: usize,
    storage_dtype: KtDType,
    layer_idx: usize,
    kind: KvPoolKind,
    zero_initialize: bool,
) -> Result<KtTensor> {
    let label = kind.label();
    let _ = (label, layer_idx, zero_initialize);
    let shape = shape.to_vec();
    match device {
        kiln_tensor::Device::Cpu => {
            let _ = n_elements;
            Ok(KtTensor::zeros_cpu(shape, storage_dtype))
        }
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(i) => {
            let storage = cuda_zeros_ctx(i, storage_dtype, n_elements)
                .with_context(|| format!("kt paged-kv: alloc {label} layer {layer_idx}"))?;
            KtTensor::from_parts(storage, Layout::contiguous(shape), TensorId::next())
                .with_context(|| format!("kt paged-kv: wrap {label} layer {layer_idx}"))
        }
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(i) => {
            let storage: kiln_tensor::Storage = if zero_initialize {
                kiln_tensor::rocm_zeros_ctx(i, storage_dtype, n_elements).map_err(|e| {
                    anyhow::anyhow!("kt paged-kv: alloc {label} (rocm) layer {layer_idx}: {e}")
                })?
            } else {
                let context = kiln_tensor::primary_rocm_context(i).map_err(|e| {
                    anyhow::anyhow!(
                        "kt paged-kv: context for {label} (rocm) layer {layer_idx}: {e}"
                    )
                })?;
                std::sync::Arc::new(
                    kiln_tensor::RocmStorage::alloc_uninit_ctx(
                        &context,
                        i,
                        storage_dtype,
                        n_elements,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "kt paged-kv: alloc uninitialized {label} (rocm) layer {layer_idx}: {e}"
                        )
                    })?,
                )
            };
            KtTensor::from_parts(storage, Layout::contiguous(shape), TensorId::next()).map_err(
                |e| anyhow::anyhow!("kt paged-kv: wrap {label} (rocm) layer {layer_idx}: {e}"),
            )
        }
        #[cfg(feature = "metal")]
        kiln_tensor::Device::Metal(i) => {
            KtTensor::zeros_on(kiln_tensor::Device::Metal(i), shape, storage_dtype).map_err(|e| {
                anyhow::anyhow!("kt paged-kv: alloc {label} (metal) layer {layer_idx}: {e}")
            })
        }
        // Vulkan and any backend whose feature is not compiled in use the same
        // host-resident pool placement as construction.
        other => {
            let _ = other;
            let _ = n_elements;
            Ok(KtTensor::zeros_cpu(shape, storage_dtype))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn alloc_pool_tensor_attributed(
    device: kiln_tensor::Device,
    shape: &[usize],
    n_elements: usize,
    storage_dtype: KtDType,
    layer_idx: usize,
    kind: KvPoolKind,
    zero_initialize: bool,
    reason: KvPoolAllocationReason,
) -> Result<KtTensor> {
    let requested_bytes = n_elements.saturating_mul(storage_dtype.size_in_bytes().max(1));
    let started = std::time::Instant::now();
    let result = alloc_pool_tensor(
        device,
        shape,
        n_elements,
        storage_dtype,
        layer_idx,
        kind,
        zero_initialize,
    );
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    match (&result, reason) {
        (Ok(_), KvPoolAllocationReason::InitialCache) => tracing::debug!(
            event = "gpu_memory_operation",
            operation = "allocation",
            reason = reason.as_str(),
            outcome = "completed",
            ?device,
            layer = layer_idx,
            pool = kind.label(),
            requested_bytes,
            actual_bytes = requested_bytes,
            wait_ms = 0.0,
            duration_ms,
            zero_initialize,
            "KV pool allocation completed"
        ),
        (Ok(_), KvPoolAllocationReason::PhysicalResize) => tracing::info!(
            event = "gpu_memory_operation",
            operation = "allocation",
            reason = reason.as_str(),
            outcome = "completed",
            ?device,
            layer = layer_idx,
            pool = kind.label(),
            requested_bytes,
            actual_bytes = requested_bytes,
            wait_ms = 0.0,
            duration_ms,
            zero_initialize,
            "KV pool allocation completed"
        ),
        (Err(error), _) => tracing::warn!(
            event = "gpu_memory_operation",
            operation = "allocation",
            reason = reason.as_str(),
            outcome = "failed",
            %error,
            ?device,
            layer = layer_idx,
            pool = kind.label(),
            requested_bytes,
            actual_bytes = 0,
            wait_ms = 0.0,
            duration_ms,
            zero_initialize,
            "KV pool allocation failed"
        ),
    }
    result
}

/// Allocate one zero-filled `(k_pool, v_pool)` pair of shape `shape`
/// (`= [total_slots, num_kv_heads, head_dim]`, `n_elements` elements total) on
/// `device`.
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
    Ok((
        alloc_pool_tensor_attributed(
            device,
            shape,
            n_elements,
            storage_dtype,
            layer_idx,
            KvPoolKind::Key,
            true,
            KvPoolAllocationReason::InitialCache,
        )?,
        alloc_pool_tensor_attributed(
            device,
            shape,
            n_elements,
            storage_dtype,
            layer_idx,
            KvPoolKind::Value,
            true,
            KvPoolAllocationReason::InitialCache,
        )?,
    ))
}

/// Block until all device work completes, so a subsequent pool drop can't free
/// storage a still-running kernel reads. No-op on CPU (synchronous) and on
/// backends whose feature isn't compiled in. Used by
/// [`PagedKvCacheKt::physical_resize_to`] (#26) as the C2 use-after-free guard.
fn sync_device_for_resize(
    device: kiln_tensor::Device,
    phase: &'static str,
    layer: Option<usize>,
    affected_bytes: u64,
) -> Result<()> {
    let started = std::time::Instant::now();
    let result = match device {
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(i) => kiln_tensor::cuda_synchronize_default_stream_for(
            i,
            kiln_tensor::CudaSyncReason::GlobalStateMutation,
        )
        .map_err(|e| anyhow::anyhow!("physical_resize_to: cuda sync: {e}")),
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(i) => kiln_tensor::rocm_synchronize_default_stream(i)
            .map_err(|e| anyhow::anyhow!("physical_resize_to: rocm sync: {e}")),
        _ => Ok(()),
    };
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    match &result {
        Ok(()) => tracing::info!(
            event = "gpu_memory_operation",
            operation = "synchronize",
            reason = "kv_physical_resize",
            outcome = "completed",
            ?device,
            phase,
            ?layer,
            affected_bytes,
            wait_ms = duration_ms,
            duration_ms,
            "KV resize device synchronization completed"
        ),
        Err(error) => tracing::warn!(
            event = "gpu_memory_operation",
            operation = "synchronize",
            reason = "kv_physical_resize",
            outcome = "failed",
            %error,
            ?device,
            phase,
            ?layer,
            affected_bytes,
            wait_ms = duration_ms,
            duration_ms,
            "KV resize device synchronization failed"
        ),
    }
    result
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum KvResizeFaultPoint {
    InitialSync,
    AllocateKey { layer: usize },
    AllocateValue { layer: usize },
    ZeroFillSync { layer: usize },
    CopyKey { layer: usize },
    CopyValue { layer: usize },
    LayerStaged { layer: usize },
    FinalSync,
    Commit,
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
    /// `layers` swap. Acquire/release ordering publishes it with the pool
    /// generation.
    num_blocks: std::sync::atomic::AtomicUsize,
    /// Process-unique cache allocation plus a generation advanced at each
    /// successful physical pool replacement.
    allocation_id: u64,
    generation: AtomicU64,
    /// Whether FP8 quantization is enabled. When true, pool dtype is U8.
    fp8: bool,
    /// Per-layer FP8 scale factors `(k_scale, v_scale)`. Populated by the
    /// constructors; the FP8 write path (which would update + read them) lands
    /// in a follow-up PR — until then the field is written but never read, so
    /// the allow is required (verified by default-lane probe).
    #[allow(dead_code)]
    fp8_scales: Vec<(f32, f32)>,
    /// The original compute dtype for dequantization. Distinct from the
    /// storage dtype when FP8 is in use.
    compute_dtype: KtDType,
}

impl PagedKvCacheKt {
    pub(crate) fn resolve_unique_decode_slots(
        &self,
        block_tables: &[&BlockTable],
        start_positions: &[usize],
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            block_tables.len() == start_positions.len(),
            "decode slot metadata length mismatch"
        );
        let mut slots = Vec::with_capacity(start_positions.len());
        for (row, (&start_position, block_table)) in
            start_positions.iter().zip(block_tables.iter()).enumerate()
        {
            let slot = block_table
                .slot_for(start_position, self.block_size)
                .ok_or_else(|| anyhow::anyhow!("decode KV slot lookup failed for row {row}"))?;
            let slot = u32::try_from(slot)
                .map_err(|_| anyhow::anyhow!("decode KV slot {slot} exceeds u32"))?;
            anyhow::ensure!(
                !slots.contains(&slot),
                "decode rows share physical KV slot {slot} (row {row})"
            );
            slots.push(slot);
        }
        Ok(slots)
    }

    /// Create a new paged KV cache with zero-filled pre-allocated pool
    /// tensors. Replaces the candle `PagedKvCache::new` (#1082 candle-drop).
    ///
    /// `device` selects the device the pools are allocated on. The pools
    /// MUST live on the model's *runtime* device (not a compile-time
    /// feature-gated default) so the per-layer K/V `slice_set` writes match
    /// the model's tensors and don't trip `Tensor::slice_set: device
    /// mismatch`. CUDA routes through `cuda_zeros_ctx`; Metal through
    /// `zeros_on(Device::Metal, ..)`; ROCm through `rocm_zeros_ctx`; CPU (and
    /// any GPU backend whose feature isn't compiled in, e.g. Vulkan whose kt
    /// pools are CPU-resident) through host-resident `zeros_cpu`.
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
            // path; ROCm pools stay device-resident even when fused paged
            // decode is quarantined because the portable full-attention path
            // consumes the same cache through backend-local tensor ops.
            // Vulkan (kt vulkan tensors are CPU-resident) and any GPU device
            // whose backend feature isn't compiled in fall to the host-resident
            // `zeros_cpu` default — matching the prior non-CUDA/non-Metal
            // behavior and exactly what the deleted candle cache did for the
            // Vulkan backend (it held CPU candle tensors).
            let (k, v) = alloc_pool_pair(device, &shape, n_elements, storage_dtype, _i)?;
            layers.push((k, v));
        }
        let fp8_scales = vec![(1.0_f32, 1.0_f32); num_full_attn_layers];
        Ok(Self {
            layers: std::sync::RwLock::new(layers),
            block_size,
            num_blocks: std::sync::atomic::AtomicUsize::new(num_blocks),
            allocation_id: next_kv_pool_allocation_id()?,
            generation: AtomicU64::new(0),
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
        self.num_blocks.load(Ordering::Acquire)
    }

    /// Return a consistent physical-pool identity while excluding a resize
    /// commit. Tensor clones do not change this identity; replacing any pool
    /// does.
    pub fn pool_identity(&self) -> KvPoolIdentity {
        let _layers = self.layers_read();
        KvPoolIdentity {
            allocation_id: self.allocation_id,
            generation: self.generation.load(Ordering::Acquire),
            num_blocks: self.num_blocks.load(Ordering::Acquire),
        }
    }

    /// Fail before a native graph launch when its captured paged-pool pointers
    /// no longer name this cache generation.
    pub fn ensure_pool_identity(&self, expected: KvPoolIdentity) -> Result<()> {
        let actual = self.pool_identity();
        anyhow::ensure!(
            actual == expected,
            "paged KV pool identity changed: expected {expected:?}, actual {actual:?}"
        );
        Ok(())
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
        self.bytes_per_block_for_layers(&layers)
    }

    fn bytes_per_block_for_layers(&self, layers: &[(KtTensor, KtTensor)]) -> usize {
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
    /// ROCm replacement pools are allocated uninitialized. The surviving prefix
    /// is overwritten before commit; a grown tail consists only of free blocks,
    /// which no `BlockTable` can address until allocation and whose KV slots are
    /// written before attention includes them in a sequence length. Avoiding a
    /// multi-gigabyte tail zero is also correctness-critical on gfx1151/ROCm 7.2:
    /// qualification reproduced loss of an already-copied prefix after later
    /// pages in the staged generation were first touched. Construction remains
    /// zero-initialized, and other resize backends retain their zeroed tails.
    ///
    /// Both shrink and grow stage every new pool before one commit. This costs a
    /// temporary `old + new` allocation, but it is the only portable way to make
    /// allocation, copy, and synchronization failures leave every layer, the
    /// published capacity, and the pool generation unchanged. Callers must
    /// reserve that transient headroom before requesting a physical resize.
    ///
    /// No-op (returns `Ok`) when `new_num_blocks == num_blocks`.
    pub fn physical_resize_to(
        &self,
        new_num_blocks: usize,
        device: kiln_tensor::Device,
    ) -> Result<()> {
        self.physical_resize_to_with_fault(new_num_blocks, device, |_| Ok(()))
    }

    fn physical_resize_to_with_fault<F>(
        &self,
        new_num_blocks: usize,
        device: kiln_tensor::Device,
        mut checkpoint: F,
    ) -> Result<()>
    where
        F: FnMut(KvResizeFaultPoint) -> Result<()>,
    {
        // The write lock covers the complete transaction and serializes an
        // accidental second resizer even if an outer caller violates the actor
        // barrier contract.
        let mut layers = self.layers.write().unwrap_or_else(|e| e.into_inner());
        let cur = self.num_blocks.load(Ordering::Acquire);
        if new_num_blocks == cur {
            return Ok(());
        }
        anyhow::ensure!(
            self.generation.load(Ordering::Acquire) < u64::MAX,
            "physical_resize_to: pool generation exhausted"
        );
        let replacement_bytes = (new_num_blocks as u64).saturating_mul(
            u64::try_from(self.bytes_per_block_for_layers(&layers)).unwrap_or(u64::MAX),
        );
        // C2: flush any in-flight kernel before we drop the old pools. The caller
        // guarantees no NEW launches during the resize (actor barrier); this
        // covers a kernel already submitted on another stream (graph replay).
        checkpoint(KvResizeFaultPoint::InitialSync)?;
        sync_device_for_resize(device, "initial_drain", None, replacement_bytes)?;

        let storage_dtype = if self.fp8 {
            KtDType::U8
        } else {
            self.compute_dtype
        };
        let new_total_slots = new_num_blocks * self.block_size;
        let old_total_slots = cur * self.block_size;
        let copy_slots = new_total_slots.min(old_total_slots);

        let mut staged: Vec<(KtTensor, KtTensor)> = Vec::with_capacity(layers.len());
        for layer_idx in 0..layers.len() {
            let dims = layers[layer_idx].0.dims().to_vec();
            anyhow::ensure!(
                dims.len() == 3,
                "physical_resize_to: layer {layer_idx} pool has rank {} (want 3)",
                dims.len()
            );
            let shape = vec![new_total_slots, dims[1], dims[2]];
            let n_elements = new_total_slots * dims[1] * dims[2];

            checkpoint(KvResizeFaultPoint::AllocateKey { layer: layer_idx })?;
            let new_k = alloc_pool_tensor_attributed(
                device,
                &shape,
                n_elements,
                storage_dtype,
                layer_idx,
                KvPoolKind::Key,
                false,
                KvPoolAllocationReason::PhysicalResize,
            )?;
            checkpoint(KvResizeFaultPoint::AllocateValue { layer: layer_idx })?;
            let new_v = alloc_pool_tensor_attributed(
                device,
                &shape,
                n_elements,
                storage_dtype,
                layer_idx,
                KvPoolKind::Value,
                false,
                KvPoolAllocationReason::PhysicalResize,
            )?;

            // Complete asynchronous allocation/initialization before copying
            // the surviving prefix. ROCm replacement pools intentionally skip
            // initialization; other backends retain their zero-fill behavior.
            if copy_slots > 0 {
                checkpoint(KvResizeFaultPoint::ZeroFillSync { layer: layer_idx })?;
                sync_device_for_resize(
                    device,
                    "allocation_completion",
                    Some(layer_idx),
                    replacement_bytes,
                )?;
                let (old_k, old_v) = &layers[layer_idx];
                let src_k = old_k.narrow(0, 0, copy_slots).map_err(|e| {
                    anyhow::anyhow!("physical_resize_to: narrow k l{layer_idx}: {e}")
                })?;
                let src_v = old_v.narrow(0, 0, copy_slots).map_err(|e| {
                    anyhow::anyhow!("physical_resize_to: narrow v l{layer_idx}: {e}")
                })?;
                checkpoint(KvResizeFaultPoint::CopyKey { layer: layer_idx })?;
                new_k
                    .slice_set(&src_k, 0, 0)
                    .map_err(|e| anyhow::anyhow!("physical_resize_to: copy k l{layer_idx}: {e}"))?;
                checkpoint(KvResizeFaultPoint::CopyValue { layer: layer_idx })?;
                new_v
                    .slice_set(&src_v, 0, 0)
                    .map_err(|e| anyhow::anyhow!("physical_resize_to: copy v l{layer_idx}: {e}"))?;
            }
            staged.push((new_k, new_v));
            checkpoint(KvResizeFaultPoint::LayerStaged { layer: layer_idx })?;
        }

        // Copies may be asynchronous. Complete them while every old pool is
        // still owned so a failed synchronization cannot publish or free a
        // partially initialized generation.
        checkpoint(KvResizeFaultPoint::FinalSync)?;
        sync_device_for_resize(device, "copy_completion", None, replacement_bytes)?;

        checkpoint(KvResizeFaultPoint::Commit)?;
        *layers = staged;
        self.num_blocks.store(new_num_blocks, Ordering::Release);
        self.generation.fetch_add(1, Ordering::AcqRel);
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
    /// **Routes through `cudarc::memcpy_dtod_async`** on the kt
    /// storage's raw stream (`cuda_stream_raw()`) — no kt-API kernel
    /// call, no nvcc kernel needed. The destination pool's storage is
    /// mutated
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
            anyhow::bail!("kt PagedKvCacheKt::write_contiguous_slot_run requires BF16 inputs");
        }
        let pools = self.layers_read();
        let (k_pool, v_pool) = &pools[layer_idx];
        let pool_shape = k_pool.shape();
        anyhow::ensure!(
            pool_shape.len() == 3,
            "kt PagedKvCacheKt: k_pool must be rank-3 [total_slots, num_kv_heads, head_dim]"
        );
        let (total_slots, num_kv_heads, head_dim) = (pool_shape[0], pool_shape[1], pool_shape[2]);
        anyhow::ensure!(
            start_slot
                .checked_add(len)
                .map_or(false, |e| e <= total_slots),
            "kt PagedKvCacheKt: slot range [{start_slot}..{}] exceeds total_slots {total_slots}",
            start_slot + len
        );
        let row_elems = num_kv_heads * head_dim;
        let expected_elems = len
            .checked_mul(row_elems)
            .context("len * row_elems overflow")?;
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
            let raw_stream = k_dst_cuda.cuda_stream_raw() as cudarc::driver::sys::CUstream;
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
        let raw_stream = k_dst_cuda.cuda_stream_raw() as cudarc::driver::sys::CUstream;

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
    ///   kt, plus one `cuda_index_select_dim0` per pool, plus
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
            let gather_index_select = || -> Result<(KtTensor, KtTensor)> {
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
                            anyhow::anyhow!(
                                "kt pkv read: build indices on {}: {e}",
                                k_pool.device()
                            )
                        })?;
                let k = k_pool
                    .index_select(&kt_indices, 0)
                    .map_err(|e| anyhow::anyhow!("kt pkv read: index_select k_pool: {e}"))?;
                let v = v_pool
                    .index_select(&kt_indices, 0)
                    .map_err(|e| anyhow::anyhow!("kt pkv read: index_select v_pool: {e}"))?;
                Ok((k, v))
            };

            #[cfg(feature = "rocm")]
            if matches!(k_pool.device(), kiln_tensor::Device::Rocm(_)) {
                let table_width = block_table.blocks.len();
                let kt_block_table = KtTensor::from_vec_on(
                    k_pool.device(),
                    block_table.blocks.clone(),
                    vec![1, table_width],
                )
                .map_err(|e| anyhow::anyhow!("kt pkv read: build ROCm block table: {e}"))?;
                let k = kiln_tensor::rocm_paged_gather_rows(
                    k_pool,
                    &kt_block_table,
                    1,
                    seq_len,
                    table_width,
                    self.block_size,
                )
                .and_then(|t| t.squeeze(0))
                .map_err(|e| anyhow::anyhow!("kt pkv read: ROCm paged gather k_pool: {e}"))?;
                let v = kiln_tensor::rocm_paged_gather_rows(
                    v_pool,
                    &kt_block_table,
                    1,
                    seq_len,
                    table_width,
                    self.block_size,
                )
                .and_then(|t| t.squeeze(0))
                .map_err(|e| anyhow::anyhow!("kt pkv read: ROCm paged gather v_pool: {e}"))?;
                (k, v)
            } else {
                gather_index_select()?
            }
            #[cfg(not(feature = "rocm"))]
            {
                gather_index_select()?
            }
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
            let k_row = Self::row_for_slice_set(&k_flat, i, "token_major k")?;
            let v_row = Self::row_for_slice_set(&v_flat, i, "token_major v")?;
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

        let slots = self.resolve_unique_decode_slots(block_tables, start_positions)?;
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "rocm")))]
        let _ = &slots;

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

        // ROCm: resolve all slots once on the host, upload one compact slot
        // vector, then scatter the complete K/V batch on-device. This replaces
        // the prior 2 * batch D2D submissions per full-attention layer and uses
        // the same capture-safe device-slot primitive as the graph path.
        #[cfg(feature = "rocm")]
        if matches!(k.device(), kiln_tensor::Device::Rocm(_)) {
            let pools = self.layers_read();
            let (k_pool, v_pool) = &pools[layer_idx];
            let slots_tensor = KtTensor::from_vec_on(k.device(), slots, vec![batch])
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: build slots: {e}"))?;
            let k_c = k
                .squeeze(1)
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: squeeze k: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: contiguous k: {e}"))?;
            let v_c = v
                .squeeze(1)
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: squeeze v: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch: contiguous v: {e}"))?;
            kiln_flash_attn::paged_kv_write_token_major_bf16_batch_slot_kt(
                k_pool,
                v_pool,
                &k_c,
                &v_c,
                &slots_tensor,
            )
            .map_err(|e| anyhow::anyhow!("kt pkv rocm batch write: {e}"))?;
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

    /// Batched graph-slot variant — one fused device kernel launch writes
    /// every row. Replaces the candle
    /// `PagedKvCache::write_token_major_native_batch_graph_slot`
    /// (#1082 candle-drop).
    ///
    /// - `k`, `v`: `[batch, 1, num_kv_heads, head_dim]` BF16
    /// - `slots`: `[batch]` U32 device tensor (per-row destination slots)
    ///
    /// Safe under graph replay: the only per-replay-varying input is
    /// `slots`, refreshed in place outside the captured region. Returns
    /// `Ok(false)` when preconditions aren't met (FP8 pool, non-BF16 K/V,
    /// seq_len != 1, wrong `slots` shape/dtype) so callers fall back to the
    /// slower per-row path.
    ///
    /// LIVE prod path (forward.rs batched contiguous paged decode, the
    /// `kv_fused_batched_enabled()` branch). Routes through
    /// [`kiln_flash_attn::paged_kv_write_token_major_bf16_batch_slot_kt`],
    /// which writes a contiguous `[batch, num_kv_heads, head_dim]` block —
    /// so the seq_len=1 dim is squeezed before dispatch (the kernel's
    /// `element_count == batch * kv_heads * head_dim` check is then exact).
    #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
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

        #[cfg(feature = "metal")]
        if matches!(k.device(), kiln_tensor::Device::Metal(_)) {
            let k_c = k
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv metal batch_slot: contiguous k: {e}"))?;
            let v_c = v
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv metal batch_slot: contiguous v: {e}"))?;
            let slots_c = slots
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv metal batch_slot: contiguous slots: {e}"))?;
            if !crate::backend::metal::metal_paged_kv_write_token_major_batch_supports(
                k_pool, v_pool, &slots_c, &k_c, &v_c,
            ) {
                return Ok(false);
            }
            crate::backend::metal::metal_paged_kv_write_token_major_batch_bf16(
                k_pool, v_pool, &slots_c, &k_c, &v_c,
            )
            .map_err(|e| {
                anyhow::anyhow!("kt paged_kv_write_token_major_bf16_batch_slot_metal: {e}")
            })?;
            return Ok(true);
        }

        #[cfg(feature = "rocm")]
        if matches!(k.device(), kiln_tensor::Device::Rocm(_)) {
            let k_c = k
                .squeeze(1)
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch_slot: squeeze k: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch_slot: contiguous k: {e}"))?;
            let v_c = v
                .squeeze(1)
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch_slot: squeeze v: {e}"))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch_slot: contiguous v: {e}"))?;
            let slots_c = slots
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt pkv rocm batch_slot: contiguous slots: {e}"))?;
            kiln_flash_attn::paged_kv_write_token_major_bf16_batch_slot_kt(
                k_pool, v_pool, &k_c, &v_c, &slots_c,
            )
            .map_err(|e| {
                anyhow::anyhow!("kt paged_kv_write_token_major_bf16_batch_slot_rocm: {e}")
            })?;
            return Ok(true);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (k_pool, v_pool, slots);
            Ok(false)
        }

        #[cfg(feature = "cuda")]
        {
            if !matches!(k.device(), kiln_tensor::Device::Cuda(_)) {
                return Ok(false);
            }

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
            let k_row = Self::row_for_slice_set(&k_flat, i, "write_native k")?;
            let v_row = Self::row_for_slice_set(&v_flat, i, "write_native v")?;
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
            let k_row = Self::row_for_slice_set(&k_q, i, "write_fp8 k")?;
            let v_row = Self::row_for_slice_set(&v_q, i, "write_fp8 v")?;
            k_pool
                .slice_set(&k_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: slice_set k row {i}: {e}"))?;
            v_pool
                .slice_set(&v_row, 0, slot)
                .map_err(|e| anyhow::anyhow!("kt pkv write_fp8: slice_set v row {i}: {e}"))?;
        }

        Ok(())
    }

    /// `Tensor::slice_set` requires a zero-offset contiguous source. A
    /// `narrow(0, i, 1)` row view for `i > 0` has a non-zero storage offset, so
    /// every paged-KV row-scatter fallback must materialize the row before
    /// writing it into the physical KV pool.
    fn row_for_slice_set(src: &KtTensor, row_idx: usize, ctx: &str) -> Result<KtTensor> {
        src.narrow(0, row_idx, 1)
            .map_err(|e| anyhow::anyhow!("kt pkv {ctx}: narrow row {row_idx}: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt pkv {ctx}: contiguous row {row_idx}: {e}"))
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
            allocation_id: 7,
            generation: AtomicU64::new(3),
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
        assert_eq!(
            cache.pool_identity(),
            KvPoolIdentity {
                allocation_id: 7,
                generation: 3,
                num_blocks: 1024,
            }
        );
    }

    #[test]
    fn decode_slot_resolution_rejects_duplicate_physical_ownership() {
        let cache = PagedKvCacheKt::new(1, 4, 4, 1, 2, KtDType::BF16, kiln_tensor::Device::Cpu)
            .expect("decode-slot test cache");
        let first = BlockTable { blocks: vec![0] };
        let second = BlockTable { blocks: vec![1] };
        assert_eq!(
            cache
                .resolve_unique_decode_slots(&[&first, &second], &[2, 3])
                .expect("unique physical slots"),
            vec![2, 7]
        );
        let error = cache
            .resolve_unique_decode_slots(&[&first, &first], &[2, 2])
            .expect_err("duplicate physical slots must fail closed");
        assert!(error.to_string().contains("share physical KV slot 2"));
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
        let cache = PagedKvCacheKt::new(1, 4, block_size, kv_heads, head_dim, KtDType::F32, dev)
            .expect("construct cpu cache");
        assert_eq!(cache.num_blocks(), 4);
        let initial_identity = cache.pool_identity();
        assert_eq!(initial_identity.generation, 0);

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
        let shrunk_identity = cache.pool_identity();
        assert_eq!(
            shrunk_identity.allocation_id,
            initial_identity.allocation_id
        );
        assert_eq!(shrunk_identity.generation, 1);
        assert_eq!(shrunk_identity.num_blocks, 3);
        let (k_pool, v_pool) = cache.pool_tensors(0).unwrap();
        assert_eq!(k_pool.dims(), &[6, kv_heads, head_dim]);
        let k_after: Vec<f32> = k_pool.to_vec().unwrap();
        let v_after: Vec<f32> = v_pool.to_vec().unwrap();
        assert_eq!(k_after, k_vals, "shrink must preserve K slots 0..6");
        assert_eq!(v_after, v_vals, "shrink must preserve V slots 0..6");

        // GROW 3 -> 5 blocks (6 -> 10 slots). Prefix preserved, new tail zeroed.
        cache.physical_resize_to(5, dev).expect("grow");
        assert_eq!(cache.num_blocks(), 5);
        let grown_identity = cache.pool_identity();
        assert_eq!(grown_identity.allocation_id, initial_identity.allocation_id);
        assert_eq!(grown_identity.generation, 2);
        assert_eq!(grown_identity.num_blocks, 5);
        let (k_pool, _) = cache.pool_tensors(0).unwrap();
        assert_eq!(k_pool.dims(), &[10, kv_heads, head_dim]);
        let k_grown: Vec<f32> = k_pool.to_vec().unwrap();
        assert_eq!(
            &k_grown[..k_vals.len()],
            &k_vals[..],
            "grow preserves prefix"
        );
        assert!(
            k_grown[k_vals.len()..].iter().all(|&x| x == 0.0),
            "grown tail must be zero-filled"
        );

        // No-op resize returns Ok and changes nothing.
        cache.physical_resize_to(5, dev).expect("noop");
        assert_eq!(cache.num_blocks(), 5);
        assert_eq!(cache.pool_identity(), grown_identity);
        assert!(
            cache.ensure_pool_identity(initial_identity).is_err(),
            "a graph-captured pre-resize identity must be rejected"
        );
        cache
            .ensure_pool_identity(grown_identity)
            .expect("current pool identity");
    }

    #[derive(Debug, PartialEq)]
    // `type_complexity`: the per-layer snapshot row mirrors the pool's
    // positional (k/v tensor ids, shapes, scales) tuple; a struct here would
    // be test-only ceremony.
    #[allow(clippy::type_complexity)]
    struct CacheSnapshot {
        identity: KvPoolIdentity,
        layers: Vec<(TensorId, TensorId, Vec<usize>, Vec<f32>, Vec<f32>)>,
    }

    fn fault_test_cache() -> PagedKvCacheKt {
        let cache = PagedKvCacheKt::new(3, 4, 2, 1, 2, KtDType::F32, kiln_tensor::Device::Cpu)
            .expect("fault-test cache");
        for layer in 0..cache.num_layers() {
            let (k, v) = cache.pool_tensors(layer).expect("layer pools");
            let values: Vec<f32> = (0..16).map(|index| (layer * 100 + index) as f32).collect();
            let shifted_values: Vec<f32> = values.iter().map(|value| value + 1000.0).collect();
            let source = KtTensor::from_vec(values, vec![8, 1, 2]).expect("source tensor");
            k.slice_set(&source, 0, 0).expect("seed key pool");
            let shifted =
                KtTensor::from_vec(shifted_values, vec![8, 1, 2]).expect("value source tensor");
            v.slice_set(&shifted, 0, 0).expect("seed value pool");
        }
        cache
    }

    fn snapshot_cache(cache: &PagedKvCacheKt) -> CacheSnapshot {
        CacheSnapshot {
            identity: cache.pool_identity(),
            layers: (0..cache.num_layers())
                .map(|layer| {
                    let (k, v) = cache.pool_tensors(layer).expect("layer pools");
                    (
                        k.id(),
                        v.id(),
                        k.dims().to_vec(),
                        k.to_vec::<f32>().expect("key values"),
                        v.to_vec::<f32>().expect("value values"),
                    )
                })
                .collect(),
        }
    }

    fn assert_every_resize_fault_is_atomic(target_blocks: usize) {
        let discovery = fault_test_cache();
        let mut points = Vec::new();
        discovery
            .physical_resize_to_with_fault(target_blocks, kiln_tensor::Device::Cpu, |point| {
                points.push(point);
                Ok(())
            })
            .expect("discover resize checkpoints");
        let unique: std::collections::HashSet<_> = points.iter().copied().collect();
        assert_eq!(
            unique.len(),
            points.len(),
            "every checkpoint must identify one exact boundary: {points:?}"
        );
        assert_eq!(points.len(), 3 + 6 * 3, "unexpected checkpoint coverage");

        for fault in points {
            let cache = fault_test_cache();
            let before = snapshot_cache(&cache);
            let mut fired = false;
            let error = cache
                .physical_resize_to_with_fault(target_blocks, kiln_tensor::Device::Cpu, |point| {
                    if point == fault {
                        fired = true;
                        anyhow::bail!("injected resize failure at {point:?}");
                    }
                    Ok(())
                })
                .expect_err("injected checkpoint must fail");
            assert!(fired, "checkpoint was not reached: {fault:?}");
            assert!(error.to_string().contains("injected resize failure"));
            assert_eq!(
                snapshot_cache(&cache),
                before,
                "resize mutated cache after failure at {fault:?}"
            );
            cache
                .ensure_pool_identity(before.identity)
                .expect("old pool remains authoritative");

            cache
                .physical_resize_to(target_blocks, kiln_tensor::Device::Cpu)
                .unwrap_or_else(|error| panic!("retry after {fault:?} failed: {error:#}"));
            let committed = cache.pool_identity();
            assert_eq!(committed.allocation_id, before.identity.allocation_id);
            assert_eq!(committed.generation, before.identity.generation + 1);
            assert_eq!(committed.num_blocks, target_blocks);
        }
    }

    #[test]
    fn shrink_is_failure_atomic_at_every_transaction_boundary() {
        assert_every_resize_fault_is_atomic(2);
    }

    #[test]
    fn grow_is_failure_atomic_at_every_transaction_boundary() {
        assert_every_resize_fault_is_atomic(6);
    }

    #[test]
    fn no_op_resize_reaches_no_transaction_boundary() {
        let cache = fault_test_cache();
        let before = snapshot_cache(&cache);
        cache
            .physical_resize_to_with_fault(4, kiln_tensor::Device::Cpu, |point| {
                panic!("no-op resize reached {point:?}")
            })
            .expect("no-op resize");
        assert_eq!(snapshot_cache(&cache), before);
    }

    #[test]
    fn separately_constructed_pools_have_distinct_allocation_identities() {
        let first = PagedKvCacheKt::new(1, 2, 2, 1, 2, KtDType::F32, kiln_tensor::Device::Cpu)
            .expect("first cache");
        let second = PagedKvCacheKt::new(1, 2, 2, 1, 2, KtDType::F32, kiln_tensor::Device::Cpu)
            .expect("second cache");
        assert_ne!(
            first.pool_identity().allocation_id,
            second.pool_identity().allocation_id
        );
    }

    #[test]
    fn write_native_prefill_scatter_materializes_rows_for_noncontiguous_blocks() {
        let block_size = 2usize;
        let kv_heads = 2usize;
        let head_dim = 2usize;
        let new_len = 4usize;
        let per_slot = kv_heads * head_dim;
        let dev = kiln_tensor::Device::Cpu;
        let cache = PagedKvCacheKt::new(1, 3, block_size, kv_heads, head_dim, KtDType::F32, dev)
            .expect("construct cpu cache");

        let mut block_table = BlockTable::new();
        block_table.push(0);
        block_table.push(2);
        assert!(
            contiguous_slot_run_start(&block_table, block_size, 0, new_len).is_none(),
            "test must exercise row-scatter, not contiguous-run write"
        );

        let row = |base: usize, pos: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(per_slot);
            for head in 0..kv_heads {
                for dim in 0..head_dim {
                    out.push((base + head * 100 + pos * 10 + dim) as f32);
                }
            }
            out
        };
        let head_major_values = |base: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(kv_heads * new_len * head_dim);
            for head in 0..kv_heads {
                for pos in 0..new_len {
                    for dim in 0..head_dim {
                        out.push((base + head * 100 + pos * 10 + dim) as f32);
                    }
                }
            }
            out
        };

        let k = KtTensor::from_vec(head_major_values(0), vec![1, kv_heads, new_len, head_dim])
            .expect("k tensor");
        let v = KtTensor::from_vec(
            head_major_values(1000),
            vec![1, kv_heads, new_len, head_dim],
        )
        .expect("v tensor");

        cache
            .write_native(0, &block_table, 0, &k, &v)
            .expect("non-contiguous row-scatter prefill write");

        let (k_pool, v_pool) = cache.pool_tensors(0).expect("layer 0 pools");
        assert_eq!(k_pool.dims(), &[6, kv_heads, head_dim]);
        assert_eq!(v_pool.dims(), &[6, kv_heads, head_dim]);
        let k_after: Vec<f32> = k_pool.to_vec().expect("k pool host copy");
        let v_after: Vec<f32> = v_pool.to_vec().expect("v pool host copy");

        for (pos, slot) in [(0usize, 0usize), (1, 1), (2, 4), (3, 5)] {
            let start = slot * per_slot;
            assert_eq!(&k_after[start..start + per_slot], row(0, pos).as_slice());
            assert_eq!(&v_after[start..start + per_slot], row(1000, pos).as_slice());
        }
        for slot in [2usize, 3usize] {
            let start = slot * per_slot;
            assert!(
                k_after[start..start + per_slot].iter().all(|&x| x == 0.0),
                "unreferenced K slot {slot} should remain zero"
            );
            assert!(
                v_after[start..start + per_slot].iter().all(|&x| x == 0.0),
                "unreferenced V slot {slot} should remain zero"
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_batch_graph_slot_writer_uses_device_slots() -> Result<()> {
        let Some(dev) = crate::backend::metal::try_new_metal() else {
            eprintln!(
                "Metal unavailable, skipping metal_batch_graph_slot_writer_uses_device_slots"
            );
            return Ok(());
        };

        let block_size = 4usize;
        let kv_heads = 2usize;
        let head_dim = 4usize;
        let cache = PagedKvCacheKt::new(1, 4, block_size, kv_heads, head_dim, KtDType::BF16, dev)?;

        let mk = |v: f32| half::bf16::from_f32(v);
        let row_elems = kv_heads * head_dim;
        let mut k_vals = Vec::with_capacity(2 * row_elems);
        let mut v_vals = Vec::with_capacity(2 * row_elems);
        for row in 0..2 {
            for col in 0..row_elems {
                k_vals.push(mk(10.0 * row as f32 + col as f32 + 1.0));
                v_vals.push(mk(100.0 + 10.0 * row as f32 + col as f32 + 1.0));
            }
        }
        let k = KtTensor::from_vec_on(dev, k_vals.clone(), vec![2, 1, kv_heads, head_dim])?;
        let v = KtTensor::from_vec_on(dev, v_vals.clone(), vec![2, 1, kv_heads, head_dim])?;
        let slots = KtTensor::from_vec_on(dev, vec![2u32, 7u32], vec![2])?;

        assert!(cache.write_token_major_native_batch_graph_slot(0, &k, &v, &slots)?);

        let (k_pool, v_pool) = cache.pool_tensors(0).expect("layer 0 pools");
        let k_host = k_pool.to_device(kiln_tensor::Device::Cpu)?;
        let v_host = v_pool.to_device(kiln_tensor::Device::Cpu)?;
        let k_flat = k_host.flatten_all()?.to_vec1::<half::bf16>()?;
        let v_flat = v_host.flatten_all()?.to_vec1::<half::bf16>()?;

        let row = |slot: usize| slot * row_elems;
        assert_eq!(
            &k_flat[row(2)..row(2) + row_elems],
            &k_vals[..row_elems],
            "row 0 K must land at slot 2 from the Metal slot tensor"
        );
        assert_eq!(
            &v_flat[row(2)..row(2) + row_elems],
            &v_vals[..row_elems],
            "row 0 V must land at slot 2 from the Metal slot tensor"
        );
        assert_eq!(
            &k_flat[row(7)..row(7) + row_elems],
            &k_vals[row_elems..],
            "row 1 K must land at slot 7 from the Metal slot tensor"
        );
        assert_eq!(
            &v_flat[row(7)..row(7) + row_elems],
            &v_vals[row_elems..],
            "row 1 V must land at slot 7 from the Metal slot tensor"
        );
        Ok(())
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn rocm_portable_paged_cache_round_trip_stays_device_local() -> Result<()> {
        assert!(kiln_tensor::rocm_is_available());
        assert!(
            !crate::forward::rocm_paged_decode_enabled(),
            "test must exercise the portable paged-attention policy"
        );

        let device = kiln_tensor::Device::Rocm(0);
        let (seq_len, kv_heads, head_dim) = (33usize, 4usize, 256usize);
        let cache = PagedKvCacheKt::new(1, 1, 64, kv_heads, head_dim, KtDType::BF16, device)?;
        let table = BlockTable { blocks: vec![0] };
        let values: Vec<_> = (0..seq_len * kv_heads * head_dim)
            .map(|index| half::bf16::from_f32((index % 97) as f32 / 97.0))
            .collect();
        let k =
            KtTensor::from_vec_on(device, values.clone(), vec![1, kv_heads, seq_len, head_dim])?;
        let v =
            KtTensor::from_vec_on(device, values.clone(), vec![1, kv_heads, seq_len, head_dim])?;

        cache.write(0, &table, 0, &k, &v)?;
        let (read_k, read_v) = cache.read(0, &table, seq_len)?;
        assert_eq!(read_k.device(), device);
        assert_eq!(read_v.device(), device);
        assert_eq!(read_k.shape(), &[1, kv_heads, seq_len, head_dim]);
        assert_eq!(read_v.shape(), &[1, kv_heads, seq_len, head_dim]);
        assert_eq!(
            read_k
                .to_device(kiln_tensor::Device::Cpu)?
                .flatten_all()?
                .to_vec1::<half::bf16>()?,
            values
        );
        assert_eq!(
            read_v
                .to_device(kiln_tensor::Device::Cpu)?
                .flatten_all()?
                .to_vec1::<half::bf16>()?,
            values
        );
        Ok(())
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn rocm_batched_writers_scatter_noncontiguous_device_slots() -> Result<()> {
        assert!(kiln_tensor::rocm_is_available());

        let device = kiln_tensor::Device::Rocm(0);
        let block_size = 4usize;
        let kv_heads = 2usize;
        let head_dim = 4usize;
        let cache =
            PagedKvCacheKt::new(1, 4, block_size, kv_heads, head_dim, KtDType::BF16, device)?;
        let (k_pool, v_pool) = cache
            .pool_tensors(0)
            .ok_or_else(|| anyhow::anyhow!("missing layer 0 pools"))?;
        assert_eq!(k_pool.device(), device);
        assert_eq!(v_pool.device(), device);

        let mk = |value: f32| half::bf16::from_f32(value);
        let row_elems = kv_heads * head_dim;
        let k_values: Vec<_> = (0..2 * row_elems)
            .map(|index| mk(index as f32 + 1.0))
            .collect();
        let v_values: Vec<_> = (0..2 * row_elems)
            .map(|index| mk(index as f32 + 101.0))
            .collect();
        let k = KtTensor::from_vec_on(device, k_values.clone(), vec![2, 1, kv_heads, head_dim])?;
        let v = KtTensor::from_vec_on(device, v_values.clone(), vec![2, 1, kv_heads, head_dim])?;

        let first_table = BlockTable { blocks: vec![0] };
        let second_table = BlockTable { blocks: vec![1] };
        assert!(cache.write_token_major_native_batch(
            0,
            &[&first_table, &second_table],
            &[2, 3],
            &k,
            &v,
        )?);

        let graph_k_values: Vec<_> = (0..2 * row_elems)
            .map(|index| mk(index as f32 + 201.0))
            .collect();
        let graph_v_values: Vec<_> = (0..2 * row_elems)
            .map(|index| mk(index as f32 + 301.0))
            .collect();
        let graph_k = KtTensor::from_vec_on(
            device,
            graph_k_values.clone(),
            vec![2, 1, kv_heads, head_dim],
        )?;
        let graph_v = KtTensor::from_vec_on(
            device,
            graph_v_values.clone(),
            vec![2, 1, kv_heads, head_dim],
        )?;
        let graph_slots = KtTensor::from_vec_on(device, vec![9u32, 14u32], vec![2])?;
        assert!(cache.write_token_major_native_batch_graph_slot(
            0,
            &graph_k,
            &graph_v,
            &graph_slots,
        )?);

        let (k_pool, v_pool) = cache
            .pool_tensors(0)
            .ok_or_else(|| anyhow::anyhow!("missing layer 0 pools"))?;
        let k_host = k_pool.to_device(kiln_tensor::Device::Cpu)?;
        let v_host = v_pool.to_device(kiln_tensor::Device::Cpu)?;
        let k_flat = k_host.flatten_all()?.to_vec1::<half::bf16>()?;
        let v_flat = v_host.flatten_all()?.to_vec1::<half::bf16>()?;
        let row = |slot: usize| slot * row_elems;

        for (slot, source_row) in [(2usize, 0usize), (7, 1)] {
            let source = source_row * row_elems;
            assert_eq!(
                &k_flat[row(slot)..row(slot) + row_elems],
                &k_values[source..source + row_elems]
            );
            assert_eq!(
                &v_flat[row(slot)..row(slot) + row_elems],
                &v_values[source..source + row_elems]
            );
        }
        for (slot, source_row) in [(9usize, 0usize), (14, 1)] {
            let source = source_row * row_elems;
            assert_eq!(
                &k_flat[row(slot)..row(slot) + row_elems],
                &graph_k_values[source..source + row_elems]
            );
            assert_eq!(
                &v_flat[row(slot)..row(slot) + row_elems],
                &graph_v_values[source..source + row_elems]
            );
        }
        Ok(())
    }
}
