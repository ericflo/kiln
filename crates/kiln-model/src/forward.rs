//! Candle-based forward pass layers for Qwen3.5-4B.
//!
//! Implements the foundational compute primitives: embedding lookup, RMSNorm,
//! RoPE (rotary position embeddings), and SwiGLU FFN. These operate on candle
//! `Tensor` objects and are composed into the full transformer forward pass.

use anyhow::{Context, Result};
use kiln_core::execution_provenance::ExecutionProvenanceV1;
use kiln_core::model_provenance::BaseWeightShardManifest;
// (#1082) Candle import surface for forward.rs after the candle-autograd
// CustomOp islands were removed (the kt tape is the sole grad producer).
// Bare `Tensor`/`Device`/`DType`/`D` resolve to the kiln-native substrate.
#[allow(unused_imports)]
use kiln_tensor::{D, DType, Device, Tensor};
use std::cell::Cell;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::backend::BackendIdentity;
use crate::backend::capability::{
    BackendCapabilityQueries, InferenceRecurrentStatePolicy, MatmulRequest, ProjectionLoadPolicy,
    StreamingPrefillAutoDispatch, StreamingPrefillBackendPolicy, Support,
};
use crate::backend::{
    AttentionBackend, BackendRuntime, ConvBackend, GdnBackend, LinearBackend, PagedKvBackend,
    ReplayBackend, ResidencyBackend, SamplingBackend,
};
use crate::kv_cache::KvCache;
use crate::lora_loader::{
    LoraLayerWeights, LoraProjectionWeights, LoraWeights, compute_lora_delta, linear_with_lora_t,
};
// (#1082) The candle `crate::paged_kv_cache` module was deleted; its kt twin
// `PagedKvCacheKt` is the production cache. Alias it to `PagedKvCache` so the
// `model_forward_paged*` params and the ~80 `&PagedKvCache` type refs below
// resolve to the kt type — matching `generate.rs`'s identical alias so the
// paged forward fns line up with their callers. `contiguous_slot_run_start`
// is a free fn in `kiln_core::block` (the deleted module merely re-exported
// it) and now comes from the `kiln_core::block` import below.
use crate::PagedKvCacheKt as PagedKvCache;
use crate::transposed_weight_cache::{
    TransposedWeightCacheMissPolicy, transposed_weight_bytes_2d_cached_bytes,
};
use crate::weight_upload::{AcceleratorWeightUploadPacer, AcceleratorWeightUploadPolicy};
use crate::weights::{DeferredMtpSource, ModelWeights, MtpWeights, TensorDType, WeightTensor};

use kiln_core::block::{BlockTable, contiguous_slot_run_start};

/// kt-tensor type alias (#1082). Bare `Tensor` in this file is
/// `candle_core::Tensor`; `KtTensor` is the kiln-native
/// `kiln_tensor::Tensor`. Used by the kt-native embedding + lm_head
/// helpers and the `GpuWeights` kt accessors so the candle↔kt seam
/// is explicit at the region boundaries while public signatures stay
/// candle-typed. Always available (the alias itself pulls in no CUDA
/// toolchain dependency); the accessors and helpers that construct
/// `KtTensor` device storage are `#[cfg(feature = "cuda")]`.
#[allow(unused_imports)]
use kiln_tensor::Tensor as KtTensor;

fn kt_contiguous(t: &Tensor, context: &'static str) -> Result<KtTensor> {
    t.contiguous()
        .with_context(|| format!("{context}: contiguous"))
}

/// The reduction-axis marker for "the last dimension" — passed to
/// `Tensor::sum_keepdim` / `max_keepdim` / `mean_keepdim` / `cumsum` /
/// `narrow` / `Tensor::cat` / `unsqueeze` in this file. Consolidates
/// `D::Minus1` (~58 sites pre-consolidation) under a single
/// short name, mirroring the same pattern in `kiln-train/src/trainer.rs`
/// (#1082). Drops 58 candle prefixes from `forward.rs` without
/// any behavioral change.
const LAST_DIM: D = D::Minus1;

#[cfg(feature = "metal")]
#[derive(Debug)]
pub(crate) struct MetalPagedDecodeIcbInputs<'a> {
    pub(crate) q: &'a [Tensor],
    pub(crate) k: &'a [Tensor],
    pub(crate) v: &'a [Tensor],
    pub(crate) graphs: &'a mut [Option<crate::backend::metal::MetalPagedDecodeIcbGraph>],
}

#[cfg(feature = "metal")]
#[derive(Debug)]
pub struct MetalPagedDecodeIcbLayer<'a> {
    q: &'a Tensor,
    k: &'a Tensor,
    v: &'a Tensor,
    graph: &'a mut Option<crate::backend::metal::MetalPagedDecodeIcbGraph>,
}

// NVTX is always linked: when the `nvtx` cargo feature is off the
// `kiln_nvtx::range!` macro expands to a zero-sized RAII guard whose drop is
// a no-op (verified by the optimizer in release). This keeps the call sites
// below free of `#[cfg(feature = "nvtx")]` noise.

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn cuda_or_rocm_device(device: Device) -> bool {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => true,
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => true,
        _ => false,
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn try_borrow_kt_cuda(t: &Tensor) -> Option<kiln_tensor::Tensor> {
    // #1082 forward-flip: `t` is already a kt tensor (the forward compute
    // path is kt-native). The historical candle→kt borrow is a no-op now,
    // so just clone the handle after re-checking the same GPU+contiguous
    // gate the bridge borrow used to enforce.
    if cuda_or_rocm_device(t.device()) && t.is_contiguous() {
        Some(t.clone())
    } else {
        None
    }
}

/// CUDA-compatible sigmoid: `1 / (1 + exp(-x))`.
///
/// `candle_nn::ops::sigmoid` lacks a CUDA kernel, so we implement it using
/// basic tensor operations that all have CUDA support.
///
/// Phase 7 whole-composite migration (#1082): contiguous non-autograd CUDA
/// tensors take the kt composite path by default. The entire four-step
/// `neg -> exp -> add_scalar(1) -> recip` composite is replaced with a
/// single `kiln_tensor::cuda_activation_unary(kind=1)` call via the kt-bridge
/// borrow adapter. Autograd-tracked tensors continue through the existing
/// candle-tracked composite until the training tape surface is kt-native.
fn cuda_sigmoid(x: &Tensor) -> Result<Tensor> {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let mut eligible_device = false;
        #[cfg(feature = "cuda")]
        {
            eligible_device |= matches!(x.device(), Device::Cuda(_));
        }
        #[cfg(feature = "rocm")]
        {
            eligible_device |= matches!(x.device(), Device::Rocm(_));
        }
        if eligible_device
            && matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
            && x.is_contiguous()
            && !x.track_op()
            && x.rank() > 0
        {
            if let Some(out) =
                try_kt_sigmoid_composite(x).context("cuda_sigmoid try_kt_sigmoid_composite")?
            {
                return Ok(out);
            }
        }
    }
    let neg_x = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_neg(x).context("cuda_sigmoid try_kt_neg")? {
                Some(out) => out,
                None => x.neg().context("cuda_sigmoid x.neg")?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            x.neg().context("cuda_sigmoid x.neg")?
        }
    };
    let exp_neg_x = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_exp(&neg_x).context("cuda_sigmoid try_kt_exp")? {
                Some(out) => out,
                None => neg_x.exp().context("cuda_sigmoid exp")?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            neg_x.exp().context("cuda_sigmoid exp")?
        }
    };
    let one_plus = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_add_scalar(&exp_neg_x, 1.0).context("cuda_sigmoid try_kt_add_scalar")? {
                Some(out) => out,
                None => (exp_neg_x + 1.0).context("cuda_sigmoid add one")?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            (exp_neg_x + 1.0).context("cuda_sigmoid add one")?
        }
    };
    let result = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_recip(&one_plus).context("cuda_sigmoid try_kt_recip")? {
                Some(out) => out,
                None => one_plus.recip().context("cuda_sigmoid recip")?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            one_plus.recip().context("cuda_sigmoid recip")?
        }
    };
    Ok(result)
}

/// Phase 7 (#1082) — kt-API sigmoid whole-composite migration
/// helper. Routes a contiguous CUDA candle tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 1 (Sigmoid),
/// replacing the four-step neg/exp/add/recip composite with a
/// single kernel dispatch.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the per-step composite. NVTX range
/// `kiln/sigmoid_composite_kt` brackets the migrated call so nsys
/// traces separate the path from the baseline composite.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn try_kt_sigmoid_composite(x: &Tensor) -> Result<Option<Tensor>> {
    kiln_nvtx::range!(c"kiln/sigmoid_composite_kt");

    let out = match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => match kiln_tensor::cuda_activation_unary(x, 1) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        },
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => match kiln_tensor::rocm_activation_unary(x, 1) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        },
        _ => return Ok(None),
    };
    Ok(Some(out))
}

fn fused_paged_decode_disabled(device: Device) -> bool {
    if matches!(device, Device::Rocm(_)) {
        return !crate::rocm_policy::current_rocm_kernel_policy().fused_paged_decode;
    }
    if matches!(device, Device::Cuda(_)) {
        return !crate::cuda_policy::current_cuda_kernel_policy().fused_paged_decode;
    }
    false
}

/// ROCm device-resident KV pools + the native sq=1 O(n) paged-decode path
/// (correct + faster at every context length: ~13 tok/s @32, ~12 @128, ~11 @256
/// vs the contiguous O(n^2) prefill recompute's 10.5 degrading to ~8). The
/// immutable qualified profile enables the route and its KV-pool contract.
#[cfg(feature = "rocm")]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn rocm_paged_decode_enabled() -> bool {
    crate::backend::capability::DecodeExecutionPolicy::for_backend("rocm", Device::Rocm(0))
        .direct_paged_decode_attention_enabled()
}

fn direct_paged_decode_attention_enabled(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .decode_execution
        .direct_paged_decode_attention_enabled()
        && AttentionBackend::runtime_supports_flash_attn_paged_decode(backend)
}

fn native_decode_attention_required(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .decode_execution
        .require_native_decode_attention
}

fn portable_lora_decode_allowed(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .decode_execution
        .allow_portable_lora_decode
}

#[cfg(feature = "vulkan")]
fn record_vulkan_lora_paged_decode_fallback(full_attn_layer_idx: usize, total_seq_len: usize) {
    static FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if count.is_power_of_two() {
        tracing::warn!(
            event = "vulkan_lora_paged_decode_fallback",
            fallback_count = count,
            full_attn_layer = full_attn_layer_idx,
            total_seq_len,
            "Vulkan LoRA decode is using the portable paged-attention route"
        );
    }
}

fn paged_decode_requires_contiguous_kv_chunks(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .decode_execution
        .paged_decode_requires_contiguous_kv_chunks
}

fn flash_prefill_consumes_grouped_kv(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .attention
        .flash_prefill_consumes_grouped_kv
}

fn detached_chunked_prefill_supported(backend: &dyn BackendRuntime) -> bool {
    matches!(
        BackendCapabilityQueries::backend_capabilities(backend)
            .attention
            .detached_chunked_prefill,
        Support::Native | Support::NativeWithConstraints
    )
}

fn gdn_recurrent_step_supports_dtype(backend: &dyn BackendRuntime, dtype: DType) -> bool {
    match dtype {
        DType::BF16 => true,
        DType::F32 => matches!(
            BackendCapabilityQueries::backend_capabilities(backend)
                .gdn
                .recurrent_step_f32,
            Support::Native | Support::NativeWithConstraints
        ),
        _ => false,
    }
}

#[cfg(feature = "cuda")]
fn cuda_fused_rotary_qk_disabled() -> bool {
    !crate::cuda_policy::current_cuda_kernel_policy().fused_rotary_qk
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn gpu_fused_attn_decode_qkv_prep_disabled(device: &Device) -> bool {
    if matches!(device, Device::Rocm(_)) {
        return !crate::rocm_policy::current_rocm_kernel_policy().attn_decode_qkv_prep;
    }
    !crate::cuda_policy::current_cuda_kernel_policy().attn_decode_qkv_prep
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn gpu_fused_mlp_silu_mul_disabled(device: Device) -> bool {
    if matches!(device, Device::Rocm(_)) {
        return !crate::rocm_policy::current_rocm_kernel_policy().fused_mlp_silu_mul;
    }
    !crate::cuda_policy::current_cuda_kernel_policy().fused_mlp_silu_mul
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn gpu_fused_mlp_gate_up_prefill_disabled(device: Device) -> bool {
    if matches!(device, Device::Rocm(_)) {
        return !crate::rocm_policy::current_rocm_kernel_policy().fused_mlp_gate_up_prefill;
    }
    !crate::cuda_policy::current_cuda_kernel_policy().fused_mlp_gate_up_prefill
}

#[cfg(feature = "cuda")]
fn cuda_fused_attn_sigmoid_mul_disabled() -> bool {
    !crate::cuda_policy::current_cuda_kernel_policy().fused_attn_sigmoid_mul
}

fn gdn_chunk_pre_permute_policy_enabled(device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    if matches!(device, Device::Cuda(_)) {
        return crate::cuda_policy::current_cuda_kernel_policy().gdn_chunk_pre_permute;
    }
    let _ = device;
    true
}

/// Constructor stub for the Phase 7 `PagedKvCacheKt` migration (#1082).
///
/// Returns `Ok(Some(cache))` when the startup policy enables experimental
/// routes and `device` is CUDA. Returns `Ok(None)` otherwise.
///
/// # Why this exists
///
/// The 11 `PagedKvCache::new` call sites in `forward.rs` are each
/// tightly coupled to candle-typed writers/readers and to the candle
/// [`PagedKvCache::new(..., device: &Device)`] signature. The kt twin
/// [`crate::paged_kv_cache_kt::PagedKvCacheKt::new`] needs an
/// [`Arc<CudaDevice>`] + `device_index` (see
/// `crates/kiln-model/src/paged_kv_cache_kt.rs:72`) plus a
/// [`kiln_tensor::DType`] in place of candle's `DType`. Wiring
/// the gate first (commit `eab7f795`) was step 1; this helper is step 2
/// and gives the next call-site migration a single, well-tested
/// constructor surface to call instead of repeating the device-arc
/// extraction + dtype mapping in every branch.
///
/// # What it does
///
/// 1. Returns `None` immediately when the gate is off — zero overhead
///    on the default (candle-cache) path.
/// 2. Returns `None` when `device` is not CUDA — the kt twin only
///    supports CUDA today, and migrating to non-CUDA backends is out
///    of scope for #1082.
/// 3. Extracts the underlying [`Arc<CudaDevice>`] and `device_index`
///    from `device` using the same pattern as
///    [`kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow`]:
///    `Device::as_cuda_device()` + `Device::location()`.
/// 4. Maps the candle [`DType`] to the kt
///    [`kiln_tensor::DType`] via [`kiln_kt_bridge::candle_dtype_to_kt`]
///    (the same conversion every other kt-bridge entry point uses).
/// 5. Allocates via [`PagedKvCacheKt::new`] — zero-filled pool tensors
///    with the kt-tensor backing store; identical pool shape and
///    byte layout to the candle version. FP8 is not yet plumbed
///    through this stub (the FP8 path is a follow-up; the gate
///    callers exercising it should keep using
///    [`PagedKvCache::new_with_fp8`] on the candle side until the kt
///    FP8 writer lands).
///
/// # Why no call site uses this yet
///
/// Even with this constructor, the surrounding call sites still need
/// to keep a `PagedKvCache` around to satisfy the candle-typed writer
/// signatures (e.g. `write_token_major_native`, `read`, etc.) that
/// are reached after the cache is constructed. Migrating one full
/// call site is a separate PR: it has to either (a) hold both
/// caches and choose at every writer/reader, or (b) port the
/// surrounding writers/readers to the kt path in the same change.
/// Landing this constructor stub means that follow-up PR is a much
/// smaller diff — it only has to touch the writer/reader story, not
/// re-derive the device-arc / dtype plumbing per call site.
#[cfg(feature = "cuda")]
pub fn try_kt_paged_kv_cache_new(
    num_full_attn_layers: usize,
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<crate::paged_kv_cache_kt::PagedKvCacheKt>> {
    if !crate::kt_api_policy::experimental_routes_enabled() {
        return Ok(None);
    }

    // #1082: this fn deliberately only supports CUDA — return `None` for any
    // other device so callers fall through to their non-kt path. The kt cache
    // now allocates on the runtime `Device`, so pass the CUDA device through
    // directly (preserving the non-CUDA → `None` gating).
    let cache_device = match device {
        Device::Cuda(idx) => Device::Cuda(*idx),
        _ => return Ok(None),
    };
    // `dtype` is already a kt DType — no candle→kt conversion needed.
    let kt_dtype = dtype;

    let cache = crate::paged_kv_cache_kt::PagedKvCacheKt::new(
        num_full_attn_layers,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        kt_dtype,
        cache_device,
    )
    .context("try_kt_paged_kv_cache_new: PagedKvCacheKt::new failed")?;
    Ok(Some(cache))
}

/// Writer stub for the Phase 7 `PagedKvCacheKt` migration (#1082).
///
/// Companion helper to [`try_kt_paged_kv_cache_new`]. Where the
/// constructor stub gave the next call-site migration a single,
/// well-tested allocation surface for the kt cache, this helper does
/// the same for the production *writer* path on `forward.rs`. It
/// wraps [`crate::paged_kv_cache_kt::PagedKvCacheKt::write_token_major_native_graph_slot`]
/// so a call site can hold both caches in parallel and dispatch the
/// write to *both* without re-deriving the candle→kt tensor bridge in
/// every call site.
///
/// Returns `Ok(false)` when `kt_cache` is `None` (gate off / non-CUDA
/// device) so callers can fall through to the candle path unchanged.
/// Returns `Ok(true)` only when the kt write was actually issued.
///
/// # Why a stub before the first call-site migration
///
/// The single production-path call to
/// `paged_cache.write_token_major_native_graph_slot(...)` in this
/// file (the CUDA-graph fast path inside
/// `gqa_attention_paged_decode`) borrows candle-typed `&Tensor`
/// inputs (`k`, `v`, `slot`). The kt twin signature takes
/// `&KtTensor` inputs (see `paged_kv_cache_kt.rs:222`). Wiring the
/// gate first (commit `eab7f795`) was step 1; landing the
/// constructor stub (commit `638bc441`) was step 2; this helper is
/// step 3 and gives the next call-site migration a single,
/// well-tested writer surface to call alongside the candle writer.
///
/// # What it does
///
/// 1. Returns `false` immediately when `kt_cache` is `None`. This is
///    the "gate off OR non-CUDA device" case — the constructor stub
///    returns `None` in both, and this helper preserves the same
///    zero-overhead fall-through contract.
/// 2. Borrows `k`, `v`, `slot` as [`kiln_tensor::Tensor`] views via
///    [`kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow`] — the
///    same idiom every other kt-bridge entry point uses. Borrowing
///    (not copying) means the kt and candle caches read from the
///    same source K/V/slot device storage, so any divergence between
///    the two writers is a property of the *writer* path, not of
///    the inputs.
/// 3. Delegates to
///    [`PagedKvCacheKt::write_token_major_native_graph_slot`] which
///    returns `Ok(false)` for shape/dtype-incompatible inputs (FP8,
///    non-BF16, K not in token-major-single layout) — preserving
///    the candle writer's same-named contract.
///
/// FP8 is intentionally not yet plumbed through this stub. Call
/// sites exercising FP8 should keep using
/// [`PagedKvCache::write_token_major_native_graph_slot`] on the
/// candle side until the kt FP8 writer lands; this helper will
/// return `Ok(false)` in that case because
/// `PagedKvCacheKt::write_token_major_native_graph_slot` short-
/// circuits on `self.fp8`.
///
/// # Where this is now wired
///
/// As of commit `7dd0009c`, the production call site at the
/// `paged_cache.write_token_major_native_graph_slot` invocation in
/// `gqa_attention_paged_with_rope_tables` now takes an
/// `Option<&PagedKvCacheKt>` parameter and passes it through to this
/// helper. The parameter is threaded down from
/// `gqa_attention_paged_with_rope_tables`'s callers
/// (`gqa_attention_paged`, `transformer_block_paged_with_rope_tables`,
/// `model_forward_paged_inner`), all of which default to `None` today.
/// A follow-up commit will teach a cache-owning model struct to
/// allocate an `Option<PagedKvCacheKt>` via `try_kt_paged_kv_cache_new`
/// and pass `Some(&kt_cache)` down through that newly-threaded
/// parameter, at which point this helper starts mirroring real
/// production writes whenever `accelerator.kt_api_mode = "all"`.
///
/// No longer `#[allow(dead_code)]` — `gqa_attention_paged_with_rope_tables`
/// is the live caller.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_paged_kv_write_token_major_native_graph_slot(
    kt_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
    layer_idx: usize,
    k: &Tensor,
    v: &Tensor,
    slot: &Tensor,
) -> Result<bool> {
    let cache = match kt_cache {
        Some(c) => c,
        None => return Ok(false),
    };

    // #1082: k/v/slot are already kt `Tensor` (forward-flip); pass them
    // straight to the kt cache writer — no candle->kt borrow needed.
    cache
        .write_token_major_native_graph_slot(layer_idx, k, v, slot)
        .context(
            "try_kt_paged_kv_write_token_major_native_graph_slot: \
             kt write_token_major_native_graph_slot failed",
        )
}

/// Phase 7 opt-in: route a `PagedKvCache::block_size()` accessor read
/// through the kt twin `PagedKvCacheKt::block_size()` (#1082).
///
/// When `kt_cache` is `Some` and `accelerator.kt_api_mode = "all"`, this
/// returns the kt cache's
/// `block_size()` and panics if it disagrees with the candle value
/// passed in. When the gate is off OR `kt_cache` is `None`, the
/// candle value is returned unchanged — zero overhead, zero
/// behavior change on the default path.
///
/// This is the read-side counterpart to
/// [`try_kt_paged_kv_write_token_major_native_graph_slot`]. Accessors
/// are migrated first because they don't touch device storage —
/// any divergence between the candle and kt caches surfaces
/// immediately at construction time (`try_kt_paged_kv_cache_new`
/// is wired through with the same shape args as the candle path),
/// so a wired accessor that returns `kt.block_size()` is bit-for-bit
/// identical to the candle `paged_cache.block_size()` whenever the
/// gate is on.
///
/// NVTX-ranged so the migration is visible in nsys traces — when
/// the kt path is exercised it shows up as a thin slice between
/// the preceding paged-decode setup and the writer.
#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn try_kt_paged_kv_block_size(
    candle_block_size: usize,
    kt_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> usize {
    let kt = match kt_cache {
        Some(c) => c,
        None => return candle_block_size,
    };
    if !crate::kt_api_policy::experimental_routes_enabled() {
        return candle_block_size;
    }
    kiln_nvtx::range!(c"kiln/paged_kv_kt/block_size");
    let kt_block_size = kt.block_size();
    assert_eq!(
        kt_block_size, candle_block_size,
        "try_kt_paged_kv_block_size: kt.block_size()={kt_block_size} \
         disagrees with candle paged_cache.block_size()={candle_block_size}"
    );
    kt_block_size
}

/// Phase 7 opt-in: route a `PagedKvCache::is_fp8()` accessor read
/// through the kt twin `PagedKvCacheKt::is_fp8()` (#1082).
///
/// Sibling to [`try_kt_paged_kv_block_size`]. Returns the kt cache
/// value (with parity assertion against the candle value) when
/// `kt_cache` is `Some` and `accelerator.kt_api_mode = "all"`.
/// Otherwise returns the candle value unchanged.
///
/// `is_fp8()` is a pure-read accessor backed by a private `bool`
/// field on both cache types — the kt twin sets it in
/// `PagedKvCacheKt::new_with_fp8` with the same arg the candle
/// `PagedKvCache::new_with_fp8` does, so the assertion must pass
/// by construction.
#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn try_kt_paged_kv_is_fp8(
    candle_is_fp8: bool,
    kt_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> bool {
    let kt = match kt_cache {
        Some(c) => c,
        None => return candle_is_fp8,
    };
    if !crate::kt_api_policy::experimental_routes_enabled() {
        return candle_is_fp8;
    }
    kiln_nvtx::range!(c"kiln/paged_kv_kt/is_fp8");
    let kt_is_fp8 = kt.is_fp8();
    assert_eq!(
        kt_is_fp8, candle_is_fp8,
        "try_kt_paged_kv_is_fp8: kt.is_fp8()={kt_is_fp8} \
         disagrees with candle paged_cache.is_fp8()={candle_is_fp8}"
    );
    kt_is_fp8
}

/// Phase 7 opt-in: parity-check a `PagedKvCache::pool_tensors(layer_idx)`
/// presence read against the kt twin (#1082).
///
/// Unlike [`try_kt_paged_kv_block_size`] / [`try_kt_paged_kv_is_fp8`], this
/// helper does NOT return the kt tensors — they're a different type
/// (`&KtTensor` vs `&Tensor`) and threading them through the existing
/// flash-attn callers would be a much bigger change than an accessor
/// migration. Instead, this just asserts that the kt cache has a layer
/// for `layer_idx` iff the candle cache does, so that any kt allocator
/// drift (e.g. a future change to how kt counts layers from
/// `num_full_attn_layers`) surfaces immediately under the startup policy.
///
/// The return is `()`. Callers continue to use the candle `pool_tensors`
/// return for the actual K/V pool tensors.
///
/// Active only when `accelerator.kt_api_mode = "all"`; otherwise a no-op.
#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn try_kt_paged_kv_pool_tensors_present(
    candle_present: bool,
    layer_idx: usize,
    kt_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) {
    let kt = match kt_cache {
        Some(c) => c,
        None => return,
    };
    if !crate::kt_api_policy::experimental_routes_enabled() {
        return;
    }
    kiln_nvtx::range!(c"kiln/paged_kv_kt/pool_tensors_present");
    let kt_present = kt.pool_tensors(layer_idx).is_some();
    assert_eq!(
        kt_present, candle_present,
        "try_kt_paged_kv_pool_tensors_present: layer_idx={layer_idx}, \
         kt.pool_tensors(idx).is_some()={kt_present} \
         disagrees with candle paged_cache.pool_tensors(idx).is_some()={candle_present}"
    );
}

/// Phase 7 opt-in: route a `PagedKvCache::num_layers()` accessor read
/// through the kt twin (#1082).
///
/// Sibling to [`try_kt_paged_kv_block_size`] / [`try_kt_paged_kv_is_fp8`].
/// Active only when `accelerator.kt_api_mode = "all"`.
///
/// `num_layers()` is `Vec<(KtTensor,KtTensor)>::len()` on the kt side
/// and the equivalent on candle — both are populated from
/// `num_full_attn_layers` in their respective `new_with_fp8`
/// constructors, so divergence requires the kt allocator to disagree
/// with the candle one. Assertion is defense-in-depth.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[inline]
pub(crate) fn try_kt_paged_kv_num_layers(
    candle_num_layers: usize,
    kt_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
) -> usize {
    let kt = match kt_cache {
        Some(c) => c,
        None => return candle_num_layers,
    };
    if !crate::kt_api_policy::experimental_routes_enabled() {
        return candle_num_layers;
    }
    kiln_nvtx::range!(c"kiln/paged_kv_kt/num_layers");
    let kt_num_layers = kt.num_layers();
    assert_eq!(
        kt_num_layers, candle_num_layers,
        "try_kt_paged_kv_num_layers: kt.num_layers()={kt_num_layers} \
         disagrees with candle paged_cache.num_layers()={candle_num_layers}"
    );
    kt_num_layers
}

/// Phase 7 (#1082): kt-typed paged-cache K/V read for the now-kt
/// full-attention prefill path. When the kt twin cache is present, read
/// directly from it (`PagedKvCacheKt::read` returns kt tensors — the
/// contiguous fast path is a zero-copy `narrow`, so this avoids the
/// candle→kt device copy entirely). Otherwise read from the candle cache
/// and bridge the result to kt (CUDA copy).
///
/// Unlike the accessor helpers above, the surrounding consumers are now
/// kt-typed (`Tensor::cat`, head-major transposes), so this returns kt
/// tensors rather than asserting parity against a candle value.
#[cfg(feature = "cuda")]
fn try_kt_paged_kv_read(
    candle_cache: &PagedKvCache,
    kt_cache: Option<&crate::paged_kv_cache_kt::PagedKvCacheKt>,
    layer_idx: usize,
    block_table: &BlockTable,
    seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    if let Some(kt) = kt_cache {
        kiln_nvtx::range!(c"kiln/paged_kv_kt/read");
        return kt.read(layer_idx, block_table, seq_len);
    }
    // (#1082) `candle_cache` is now the kt `PagedKvCacheKt` (the candle module
    // was deleted); `read` returns kt tensors directly, so the legacy
    // `candle_to_kt_activation` bridge is gone — pass the kt pools through.
    let (k, v) = candle_cache.read(layer_idx, block_table, seq_len)?;
    Ok((k, v))
}

/// Non-CUDA twin of [`try_kt_paged_kv_read`].
///
/// (#1082 DoD-100) `PagedKvCacheKt::read` is device-agnostic now — the
/// contiguous fast path is a zero-copy `narrow`, the gather uses the
/// device-agnostic `index_select` (only the index H2D is CUDA-specific), and
/// the transpose/contiguous/unsqueeze tail is pure kt; only the FP8 dequant
/// stays CUDA-only. So the generic CPU-paged attention fallback in
/// [`gqa_attention_paged_with_rope_tables`] can read directly from the (kt)
/// cache on CPU. The Vulkan backend still uses its single-submit resident path
/// (`model_forward_paged_last_token_resident_native_vk`, KV in
/// `VkPagedKvCache`) at runtime, so this generic fallback compiles for it but
/// is not reached.
#[cfg(not(feature = "cuda"))]
fn try_kt_paged_kv_read(
    candle_cache: &PagedKvCache,
    layer_idx: usize,
    block_table: &BlockTable,
    seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    // No kt twin cache on this (non-CUDA) call site — read directly from the
    // kt `PagedKvCacheKt` (`PagedKvCache` is its alias post candle-drop).
    candle_cache.read(layer_idx, block_table, seq_len)
}

thread_local! {
    static VULKAN_SKIP_GDN_STATE_READBACK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[allow(dead_code)]
pub(crate) fn vulkan_skip_gdn_state_readback_active() -> bool {
    VULKAN_SKIP_GDN_STATE_READBACK_DEPTH.with(|depth| depth.get() > 0)
}

pub(crate) struct VulkanSkipGdnStateReadbackScope {
    active: bool,
}

impl VulkanSkipGdnStateReadbackScope {
    pub(crate) fn new(active: bool) -> Self {
        if active {
            VULKAN_SKIP_GDN_STATE_READBACK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        }
        Self { active }
    }
}

impl Drop for VulkanSkipGdnStateReadbackScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        VULKAN_SKIP_GDN_STATE_READBACK_DEPTH.with(|depth| {
            let previous = depth.get();
            debug_assert!(previous > 0);
            depth.set(previous.saturating_sub(1));
        });
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn require_active_tape_output<T>(output: Option<T>, operation: &str) -> Result<T> {
    output.ok_or_else(|| anyhow::anyhow!("active tape scope could not record {operation}"))
}

/// CUDA-compatible SiLU (Swish): `x * sigmoid(x)`.
///
/// Phase 7 whole-composite migration (#1082): contiguous non-autograd CUDA
/// tensors take the kt composite path by default. The entire `sigmoid(x) * x`
/// composite is replaced with a single `kiln_tensor::cuda_activation_unary`
/// call (SiLU) via the kt-bridge borrow adapter. Autograd-tracked tensors
/// continue through the existing candle-tracked composite until the training
/// tape surface is kt-native.
fn cuda_silu(x: &Tensor) -> Result<Tensor> {
    // A scope is an authoritative request for a connected graph. Route before
    // every backend-specific leaf (and before feature-specific eligibility
    // checks) so Metal and combined-feature builds cannot silently miss SiLU.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_silu_kt(x).context("cuda_silu try_tape_silu_kt")?,
            "SiLU",
        );
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let mut eligible_device = false;
        #[cfg(feature = "cuda")]
        {
            eligible_device |= matches!(x.device(), Device::Cuda(_));
        }
        #[cfg(feature = "rocm")]
        {
            eligible_device |= matches!(x.device(), Device::Rocm(_));
        }
        if eligible_device
            && matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
            && x.is_contiguous()
            && !x.track_op()
            && x.rank() > 0
        {
            if let Some(out) =
                try_kt_silu_composite(x).context("cuda_silu try_kt_silu_composite")?
            {
                return Ok(out);
            }
        }
    }
    let sig = cuda_sigmoid(x)?;
    Ok((x * sig)?)
}

/// Phase 7 (#1082) — kt-API SiLU whole-composite migration helper.
/// Routes a contiguous CUDA candle tensor through
/// `kiln_tensor::cuda_activation_unary` with kind tag 0 (SiLU),
/// replacing the two-step `sigmoid(x) * x` composite with a single
/// kernel dispatch.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the `cuda_sigmoid` + multiply composite. NVTX range
/// `kiln/silu_composite_kt` brackets the migrated call so nsys
/// traces separate the path from the baseline composite.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn try_kt_silu_composite(x: &Tensor) -> Result<Option<Tensor>> {
    kiln_nvtx::range!(c"kiln/silu_composite_kt");

    // kind tag 0 = SiLU (matches csrc/activation.cu KIND_SILU).
    let out_kt = match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => match kiln_tensor::cuda_activation_unary(x, 0) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        },
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => match kiln_tensor::rocm_activation_unary(x, 0) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        },
        _ => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

// (#1082) Deleted the orphaned candle `any_tensor_tracks_op`: its only callers
// were the deleted candle-CustomOp helpers. The kt twin below is the survivor.

/// kt autograd-tracking gate (#1082 forward-flip). kt tensors are
/// forward-only — `track_op()` is structurally always `false` — so this is a
/// no-op gate that always reports `false`. Provided so kt-path autograd gates
/// keep the same call shape without a type clash.
fn any_kt_tensor_tracks_op(tensors: &[&kiln_tensor::Tensor]) -> bool {
    tensors.iter().any(|tensor| tensor.track_op())
}

#[cfg(feature = "metal")]
fn metal_streaming_gdn_forward_only_fastpaths_enabled() -> bool {
    false
}

#[cfg(feature = "rocm")]
fn rocm_streaming_gdn_forward_only_fastpaths_enabled() -> bool {
    false
}

fn streaming_gdn_forward_only_fastpaths_allowed(device: &Device) -> bool {
    #[cfg(feature = "metal")]
    {
        if matches!(device, Device::Metal(_)) {
            return metal_streaming_gdn_forward_only_fastpaths_enabled();
        }
    }
    #[cfg(feature = "rocm")]
    {
        if matches!(device, Device::Rocm(_)) {
            return rocm_streaming_gdn_forward_only_fastpaths_enabled();
        }
    }
    let _ = device;
    true
}

fn weighted_lm_head_prep_disabled() -> bool {
    false
}

// Callers live in cuda/metal-gated tests only, so it is dead everywhere else.
#[cfg_attr(
    not(all(test, any(feature = "cuda", feature = "metal"))),
    allow(dead_code)
)]
fn synchronize_for_profile(device: &Device) -> Result<()> {
    // Profiling timing only makes sense after the launching queue has
    // drained — without this the recorded elapsed time captures kernel
    // enqueue cost rather than execution cost. Drain the backend's execution
    // queue for any async backend; CPU is already synchronous, so the
    // match-and-skip there avoids the extra call.
    match device {
        Device::Cpu => Ok(()),
        // #1082: CUDA sync is kt-native — drain the device's default stream via
        // the kt cuda API; no candle bridge in the cuda lane.
        #[cfg(feature = "cuda")]
        Device::Cuda(idx) => kiln_tensor::cuda_synchronize_default_stream_for(
            *idx,
            kiln_tensor::CudaSyncReason::ExplicitStreamDrain,
        )
        .map_err(|e| anyhow::anyhow!("synchronize_for_profile: {e}")),
        #[cfg(feature = "rocm")]
        Device::Rocm(idx) => kiln_tensor::rocm_synchronize_compute_stream_for(
            *idx,
            kiln_tensor::RocmSyncReason::ExplicitStreamDrain,
        )
        .map_err(|e| anyhow::anyhow!("synchronize_for_profile: {e}")),
        // (#1082) Metal: kt-native queue drain. Block until the
        // MetalCompanion's command pool — the queue that actually ran the
        // compute — completes. `wait_until_completed` is the same host-read
        // sync point every Metal->host readback uses, so it drains the right
        // queue (the former candle-device bridge drained candle's separate
        // queue). kt `Device` is `#[non_exhaustive]`, so this wildcard also
        // satisfies exhaustiveness. Excluded from the cuda build.
        #[cfg(feature = "metal")]
        _ => {
            let idx = if let Device::Metal(i) = device { *i } else { 0 };
            kiln_tensor::primary_metal_companion(idx)
                .and_then(|c| c.wait_until_completed())
                .map_err(|e| anyhow::anyhow!("synchronize_for_profile: {e}"))
        }
        // (#1082) Vulkan (and any other non-CUDA, non-Metal build) keeps the
        // kt seam CPU-resident — tensors live on `Device::Cpu` at the kt<->vk
        // boundary (`loader_kt_device` coerces `Vulkan(_)` to `Cpu`), and the
        // CPU is already synchronous, so there is no async queue to drain. The
        // former candle-device bridge is gone from the vulkan lane; this
        // wildcard is the no-op profiling sync. kt `Device` is
        // `#[non_exhaustive]`, so it also satisfies exhaustiveness.
        #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
        _ => Ok(()),
        // In a cuda build the only async backend is CUDA (handled above); the
        // wildcard keeps the `#[non_exhaustive]` match exhaustive.
        #[cfg(feature = "cuda")]
        _ => Ok(()),
    }
}

fn synchronize_tensor_ready_for_model_handoff(label: &str, tensor: &Tensor) -> Result<()> {
    let _ = label;
    match tensor.device() {
        Device::Cpu => Ok(()),
        #[cfg(feature = "cuda")]
        Device::Cuda(idx) => kiln_tensor::cuda_synchronize_default_stream_for(
            idx,
            kiln_tensor::CudaSyncReason::ModelHandoff,
        )
        .with_context(|| format!("{label}: synchronize CUDA tensor readiness")),
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => kiln_tensor::rocm_synchronize_tensor_same_stream_dependency(
            tensor,
            kiln_tensor::RocmSyncReason::ModelHandoff,
        )
        .with_context(|| format!("{label}: order ROCm tensor readiness")),
        #[cfg(feature = "metal")]
        Device::Metal(idx) => kiln_tensor::primary_metal_companion(idx)
            .and_then(|companion| companion.wait_until_completed())
            .with_context(|| format!("{label}: synchronize Metal tensor readiness")),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(idx) => kiln_tensor::vulkan_synchronize_queue(idx)
            .with_context(|| format!("{label}: synchronize Vulkan tensor readiness")),
        _ => Ok(()),
    }
}

fn synchronize_tensor_ready_for_full_attn_handoff(label: &str, tensor: &Tensor) -> Result<()> {
    match tensor.device() {
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            if kiln_tensor::rocm_capture_arena_active() {
                Ok(())
            } else {
                kiln_tensor::rocm_synchronize_tensor_same_stream_dependency(
                    tensor,
                    kiln_tensor::RocmSyncReason::FullAttentionHandoff,
                )
                .with_context(|| format!("{label}: order ROCm full-attention handoff"))
            }
        }
        _ => synchronize_tensor_ready_for_model_handoff(label, tensor),
    }
}

#[cfg(feature = "metal")]
fn metal_autoreleasepool<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    objc2::rc::autoreleasepool(|_| f())
}

#[cfg(feature = "metal")]
fn try_metal_mlp_gate_up_hidden(
    x: &Tensor,
    mlp: &GpuFfnWeights,
    lora_layer: Option<&LoraLayerWeights>,
) -> Result<Option<Tensor>> {
    if lora_layer.is_some_and(LoraLayerWeights::has_mlp_gate_up)
        || crate::backend::metal::metal_mlp_gate_up_fusion_disabled()
    {
        return Ok(None);
    }
    if mlp.gate_proj_marlin.is_some()
        || mlp.up_proj_marlin.is_some()
        || mlp.down_proj_marlin.is_some()
    {
        return Ok(None);
    }
    if !crate::backend::metal::metal_mlp_gate_up_supports(x, &mlp.gate_proj_t, &mlp.up_proj_t) {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/mlp/gate_up_fused");
    Ok(Some(crate::backend::metal::metal_mlp_gate_up_bf16(
        x,
        &mlp.gate_proj_t,
        &mlp.up_proj_t,
    )?))
}

fn linear_with_lora_t_decode(
    x: &Tensor,
    weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    {
        if !crate::mtp_runtime::single_token_self_attention_active()
            && (crate::backend::metal::metal_transposed_coop_gemv_supports(x, weight_t)
                || crate::backend::metal::metal_transposed_coop_gemv_decode_batch_supports(
                    x, weight_t,
                ))
        {
            let base = crate::backend::metal::metal_transposed_coop_gemv_bf16(x, weight_t)
                .context("metal transposed coop GEMV failed");
            return add_lora_delta_to_base(None, base?, x, lora, lora_scale)
                .context("metal transposed coop GEMV LoRA delta failed");
        }
    }

    // #1082: `linear_with_lora_t` is kt-native (lora_loader) — pass kt directly.
    linear_with_lora_t(x, weight_t, lora, lora_scale)
}

fn add_lora_delta_to_base(
    backend: Option<&dyn BackendRuntime>,
    base: Tensor,
    x: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    let Some(proj) = lora else {
        return Ok(base);
    };
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_lora_add_kt(&base, x, proj, lora_scale)
                .context("add_lora_delta_to_base try_tape_lora_add_kt")?,
            "LoRA delta add",
        );
    }
    // #1082 forward-flip: `base`/`x`/`proj.a`/`proj.b` are all kt; the
    // BackendRuntime trait and `compute_lora_delta` are kt-typed (item 4/5).
    // Only the kt-tape adapter (`try_tape_lora_add_cuda`, candle-typed
    // cross-file seam) and the metal LoRA-add helpers still need a candle bridge.
    // Take the kt-native production CUDA path FIRST (it is on by default and
    // serves the hot BF16 decode), so the candle bridge below only runs on
    // the training / Metal / Vulkan / CPU fallback paths — no hot-path copy.
    #[cfg(feature = "cuda")]
    {
        if let Some(delta) = try_kt_lora_delta(x, proj, lora_scale)? {
            let delta = if delta.dtype() == base.dtype() {
                delta
            } else {
                delta.to_dtype(base.dtype())?
            };
            if let Some(out) = try_kt_lora_add(&base, &delta)? {
                return Ok(out);
            }
            return Ok((base + delta)?);
        }
    }

    // (#1082) bridge — remove when tape_forward.rs flips to kt. The only
    // remaining candle consumer here is the kt-tape adapter
    // `try_tape_lora_add_cuda` (candle-typed cross-file seam); bridge the kt
    // base/x to candle once for it (CUDA copy; only reached when the kt-native
    // path above declined).
    // CP-4 (#1082) seam flip: kt-native tape-routed LoRA add — records a
    // `LoraDeltaAddBackward` emitting grads for proj.a/proj.b (kt-keyed on their Var
    // ids), no kt->candle->kt round-trip.
    // (#1082) Vulkan added: device-agnostic pure-kt recorder. Active scopes
    // already returned through that recorder above; only inference reaches the
    // backend-specific leaves below.
    // (#1082) Deleted the dead candle-CustomOp `cuda_lora_add_training_f32` /
    // `cuda_lora_add_training_bf16` fallbacks: the kt tape's
    // `try_tape_lora_add_cuda` above is the sole autograd LoRA-add producer.
    // The binding below is consumed only by the `cuda`-gated calls; on
    // non-cuda builds the allow silences the otherwise unused binding
    // (same pattern as model_dispatch.rs resident-route locals).
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    if let Some(backend) = backend {
        // #1082 item 4: the BackendRuntime trait is kt-typed — pass kt
        // `base`/`x`/`proj.a`/`proj.b` directly (no candle bridge).
        #[cfg(feature = "cuda")]
        if let Some(out) =
            LinearBackend::runtime_lora_decode_add(backend, &base, x, &proj.a, &proj.b, lora_scale)?
        {
            return Ok(out);
        }
        // Phase 4.1 step 2 + 5: when both A and B are registry-resident,
        // dispatch the LoRA delta on-device (kt-typed backend method).
        #[cfg(feature = "cuda")]
        if let Some(delta) =
            LinearBackend::runtime_lora_delta_resident(backend, x, &proj.a, &proj.b, lora_scale)?
        {
            let delta = if delta.dtype() == base.dtype() {
                delta
            } else {
                delta.to_dtype(base.dtype())?
            };
            return Ok((base + delta)?);
        }
    }
    #[cfg(feature = "metal")]
    {
        // (#1082) The `backend::metal::*` module is a candle-typed cross-file
        // seam flipped to kt as a unit in Wave F (like the cuda backend trait
        // already was). Pass kt directly here, matching the kt sig metal.rs
        // will produce — every other metal call site in this file does the same.
        if crate::backend::metal::metal_lora_add_decode_supports(&base, x, &proj.a, &proj.b) {
            return crate::backend::metal::metal_lora_add_decode_bf16(
                &base, x, &proj.a, &proj.b, lora_scale,
            )
            .context("metal LoRA decode delta/add failed");
        }
    }
    // Final composite fallback — `compute_lora_delta` is kt-native now (#1082).
    #[cfg(feature = "cuda")]
    {
        let delta_kt = compute_lora_delta(x, proj, lora_scale)?;
        let delta_kt = if delta_kt.dtype() == base.dtype() {
            delta_kt
        } else {
            delta_kt.to_dtype(base.dtype())?
        };
        if let Some(out) = try_kt_lora_add(&base, &delta_kt)? {
            return Ok(out);
        }
        return Ok((base + delta_kt)?);
    }
    // #1082: no-CUDA CPU LoRA-add. `compute_lora_delta` is kt-native (CPU-capable),
    // so compute the kt delta and add — no candle bridge (inference CPU path).
    #[cfg(not(feature = "cuda"))]
    {
        let _ = backend;
        let delta_kt = compute_lora_delta(x, proj, lora_scale)?;
        let delta_kt = if delta_kt.dtype() == base.dtype() {
            delta_kt
        } else {
            delta_kt.to_dtype(base.dtype())?
        };
        Ok((base + delta_kt)?)
    }
}

// (#1082) Deleted the candle-CustomOp LoRA training islands:
//   `cuda_lora_linear_training_bf16` / `cuda_lora_add_training_f32` /
//   `cuda_lora_add_training_bf16` + `CudaLoraLinearBf16` / `CudaLoraAddF32` /
//   `CudaLoraAddBf16` (CustomOp3) and their candle-autograd helpers
//   (`to_dtype_if_needed`, `cuda_lora_bwd_tile_rows`, the *_disabled flags).
//   The kt tape (`try_tape_lora_linear_cuda` / `try_tape_lora_add_cuda`) is the
//   sole LoRA autograd producer now.

fn linear_with_lora_t_decode_if(
    use_metal_decode_gemv: bool,
    x: &Tensor,
    weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    if use_metal_decode_gemv {
        linear_with_lora_t_decode(x, weight_t, lora, lora_scale)
    } else {
        // #1082: `linear_with_lora_t` is kt-native (lora_loader) — pass kt directly.
        linear_with_lora_t(x, weight_t, lora, lora_scale)
    }
}

fn linear_with_lora_t_backend_decode_if(
    backend: Option<&dyn BackendRuntime>,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Tensor> {
    // Route the full base projection (+ fused LoRA delta) through the kt `Tape`
    // whenever a tape scope is active. Wired at the
    // TOP because the authoritative path's inputs are DETACHED kt-copies
    // (`track_op()==false`), so the autograd-safe `linear_prefill_apply` branch
    // below (gated on `x.track_op()`) is not reliably hit. No-ops (returns
    // None) otherwise — the production dispatch is untouched in the default
    // configuration.
    // #1082: `x`/`weight_t` are kt. Only the kt-tape adapter
    // (`try_tape_lora_linear_cuda`, candle-typed cross-file seam) needs a candle
    // bridge; the BackendRuntime trait is kt-typed (item 4) so the
    // decode/prefill methods take/return kt directly.
    {
        // Only training opens a tape scope; decode falls straight through to
        // the kt-native `LinearBackend::runtime_linear_decode`.
        // #1082 seam flip: kt-native linear+LoRA recorder — no kt->candle->kt.
        // (#1082) Vulkan added: `try_tape_lora_linear_kt` is a device-agnostic
        // pure-kt recorder (proven on Vulkan by `vk_sft_step_proof`), and it is
        // the SOLE producer of the LoRA A/B backward. PR6 wired Vulkan into the
        // `tape_forward.rs` device-matches but missed this `cfg` gate, so LoRA
        // grads were never recorded on Vulkan → empty grad store.
        #[cfg(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "rocm"
        ))]
        if crate::tape_forward::tape_scope_active() {
            return require_active_tape_output(
                crate::tape_forward::try_tape_lora_linear_kt(x, weight_t, lora, lora_scale)
                    .context("linear_with_lora_t try_tape_lora_linear_kt")?,
                "linear with LoRA",
            );
        }
        // Base projection via the backend's kt-typed decode/prefill trait
        // methods — pass kt `x`/`weight_t` directly; they return kt.
        if let Some(backend) = backend {
            // Autograd-tracked input → prefer the autograd-safe Vulkan
            // CustomOp1 (linear_prefill_apply).
            if x.track_op()
                && let Some(base) =
                    LinearBackend::runtime_linear_prefill_apply(backend, x, weight_t)?
            {
                return add_lora_delta_to_base(Some(backend), base, x, lora, lora_scale);
            }
            if let Some(base) = LinearBackend::runtime_linear_decode(backend, x, weight_t)? {
                return add_lora_delta_to_base(Some(backend), base, x, lora, lora_scale);
            }
            // Last-ditch: try the autograd-safe path even for non-tracked inputs.
            if let Some(base) = LinearBackend::runtime_linear_prefill_apply(backend, x, weight_t)? {
                return add_lora_delta_to_base(Some(backend), base, x, lora, lora_scale);
            }
        }
    }
    if lora.is_some() {
        let base = linear_with_lora_t_decode_if(use_metal_decode_gemv, x, weight_t, None, 0.0)?;
        return add_lora_delta_to_base(backend, base, x, lora, lora_scale);
    }
    linear_with_lora_t_decode_if(use_metal_decode_gemv, x, weight_t, None, 0.0)
}

#[cfg(feature = "metal")]
fn metal_attn_gate_debug_active() -> bool {
    crate::mtp_runtime::single_token_self_attention_active()
}

fn attention_output_gate_decode_if(
    _use_metal_decode_gemv: bool,
    attn_output: Tensor,
    gate: Option<&Tensor>,
) -> Result<Tensor> {
    let Some(gate) = gate else {
        return Ok(attn_output);
    };

    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    {
        if crate::tape_forward::tape_scope_active() {
            return require_active_tape_output(
                crate::tape_forward::try_tape_attn_gate_sigmoid_mul_kt(&attn_output, gate)
                    .context("attention_output_gate try_tape_attn_gate_sigmoid_mul_kt")?,
                "attention output gate",
            );
        }
    }

    #[cfg(feature = "metal")]
    {
        if _use_metal_decode_gemv
            && !metal_attn_gate_debug_active()
            && crate::backend::metal::metal_attn_gate_sigmoid_mul_supports(&attn_output, gate)
        {
            kiln_nvtx::range!(c"kiln/attn/output_gate_fused");
            return crate::backend::metal::metal_attn_gate_sigmoid_mul_bf16(&attn_output, gate)
                .context("metal attn gate sigmoid/mul failed");
        }
    }

    #[cfg(feature = "cuda")]
    {
        // (#1082) Deleted the candle-CustomOp2 `cuda_sigmoid_mul_training_bf16`
        // autograd branch: the kt tape records the sigmoid/mul gate via the
        // plain `cuda_sigmoid` + mul composite (kt ops record onto the active
        // tape), so the candle CustomOp island is dead.
        if !cuda_fused_attn_sigmoid_mul_disabled() && !attn_output.track_op() && !gate.track_op() {
            if let (Some(x_kt), Some(g_kt)) =
                (try_borrow_kt_cuda(&attn_output), try_borrow_kt_cuda(gate))
            {
                if kiln_rmsnorm_kernel::supports_sigmoid_mul_kt(&x_kt, &g_kt) {
                    // Phase 7 (#1082): kt-only. Bit-exact: bottoms out in
                    // the same `kiln_fused_sigmoid_mul` FFI symbol. Result is
                    // kt — return it directly (no candle round-trip).
                    kiln_nvtx::range!(c"kiln/attn/output_gate_cuda_fused_kt");
                    let out_kt = kiln_rmsnorm_kernel::fused_sigmoid_mul_kt(&x_kt, &g_kt)
                        .map_err(|e| anyhow::anyhow!("kt sigmoid/mul: {e}"))?;
                    return Ok(out_kt);
                }
            }
        }
    }

    let sigmoid_gate = cuda_sigmoid(gate)?;
    let gated_output = (attn_output * sigmoid_gate)?;
    Ok(gated_output)
}

// (#1082) Deleted the candle-CustomOp2 `cuda_sigmoid_mul_training_bf16`
//   + `CudaSigmoidMulTrainingBf16`: the kt tape records the attn output
//   gate via the plain `cuda_sigmoid` + mul composite.

fn full_attn_qkv_proj_decode_if(
    backend: &dyn BackendRuntime,
    use_metal_decode_gemv: bool,
    x: &Tensor,
    attn_weights: &GpuFullAttentionWeights,
    lora_layer: Option<&LoraLayerWeights>,
    lora_scale: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    let tape_scope_active = crate::tape_forward::tape_scope_active();
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )))]
    let tape_scope_active = false;
    #[cfg(feature = "metal")]
    {
        if !tape_scope_active
            && use_metal_decode_gemv
            && lora_layer.is_none()
            && attn_weights.q_proj_marlin.is_none()
            && !crate::mtp_runtime::single_token_self_attention_active()
            && crate::backend::metal::metal_fused_qkv_transposed_coop_gemv_supports(
                x,
                &attn_weights.q_proj_t,
                &attn_weights.k_proj_t,
                &attn_weights.v_proj_t,
            )
        {
            kiln_nvtx::range!(c"kiln/proj/qkv_fused");
            return crate::backend::metal::metal_fused_qkv_transposed_coop_gemv_bf16(
                x,
                &attn_weights.q_proj_t,
                &attn_weights.k_proj_t,
                &attn_weights.v_proj_t,
            )
            .context("metal fused QKV projection failed");
        }
    }

    if !tape_scope_active
        && lora_layer.is_none()
        && attn_weights.q_proj_marlin.is_none()
        && !crate::mtp_runtime::single_token_self_attention_active()
    {
        let q_dim = attn_weights.q_proj_t.dim(1)?;
        let k_dim = attn_weights.k_proj_t.dim(1)?;
        let v_dim = attn_weights.v_proj_t.dim(1)?;
        if let Some(out) = LinearBackend::runtime_full_attn_qkv_combined_decode(
            backend,
            x,
            attn_weights.qkv_proj_t.as_ref(),
            attn_weights.qkv_proj_w8.as_ref(),
            q_dim,
            k_dim,
            v_dim,
        )? {
            kiln_nvtx::range!(c"kiln/proj/qkv_fused");
            return Ok(out);
        }
        if let Some(out) = LinearBackend::runtime_full_attn_qkv_decode(
            backend,
            x,
            &attn_weights.q_proj_t,
            &attn_weights.k_proj_t,
            &attn_weights.v_proj_t,
        )? {
            kiln_nvtx::range!(c"kiln/proj/qkv_fused");
            return Ok(out);
        }
    }

    let q_raw = q_proj_forward_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        attn_weights,
        lora_layer.and_then(|l| l.q_proj.as_ref()),
        lora_scale,
    )?;
    let k = linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        &attn_weights.k_proj_t,
        lora_layer.and_then(|l| l.k_proj.as_ref()),
        lora_scale,
    )?;
    let v = linear_with_lora_t_backend_decode_if(
        Some(backend),
        use_metal_decode_gemv,
        x,
        &attn_weights.v_proj_t,
        lora_layer.and_then(|l| l.v_proj.as_ref()),
        lora_scale,
    )?;
    Ok((q_raw, k, v))
}

/// CUDA-compatible softmax on last dimension.
///
/// `candle_nn::ops::softmax_last_dim` lacks a CUDA kernel, so we implement it
/// manually: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.
///
/// Phase 7 (#1082): by default, when the input is a contiguous CUDA
/// tensor of {F32, BF16, F16}, route through
/// `kiln_tensor::cuda_softmax_last_axis` via the kt-bridge borrow
/// adapter. Falls through to the portable candle composite when the
/// default-on kt route is disabled or any precondition fails.
fn cuda_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if crate::kt_api_policy::stable_routes_enabled()
        && matches!(x.device(), Device::Cuda(_))
        && matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        && x.is_contiguous()
        // CP-4 (#1082): the kt-API softmax (`try_kt_softmax_last_dim`) returns a
        // `kt_tensor_to_candle_cuda_copy` — a FRESH candle leaf with NO candle
        // `BackpropOp`. For an autograd-tracked input that SEVERS candle's
        // `loss.backward()` graph at the softmax. That was invisible while the
        // full-attention block was disconnected from the loss, but CP-4 Inc 7
        // wired the SDPA-fallback attention chain onto the kt `Tape`: the
        // tape-authoritative backward now propagates the full attention
        // contribution to `dL/dx`, while the candle baseline (which drives the
        // parity gate via `loss.backward()`) was still dropping it here — so the
        // two diverged on every layer BELOW the full-attn layer (the BF16 gate's
        // lower-layer MLP grads jumped 0.13 → 0.63). Gate the kt-API fast path on
        // `!x.track_op()` so an autograd-tracked softmax falls through to the
        // candle-differentiable composite (same forward value, bit-close),
        // matching the established `!track_op()` guard on the other kt-API
        // forward-only ops (rms_norm, sigmoid, etc.). Inference / tape paths
        // (track_op == false) keep the kt-API fast path unchanged.
        && !x.track_op()
    {
        if let Some(out) = try_kt_softmax_last_dim(x)? {
            return Ok(out);
        }
    }
    // ROCm: route through the native `rocm_softmax_last_axis` kernel (on-device,
    // device_ptr-based) instead of the `max_keepdim`-composite below. The
    // composite's `max_keepdim` -> `max_axis` has no `rocm_fwd`, so it host-falls
    // -back (`to_device(Cpu)` -> `rocm_to_host_copy` -> `RocmStorage::slice()`),
    // which is both a per-token host round-trip on the FP8 eager-attention decode
    // path AND a hard panic under HIP-graph capture (slice() on a Borrowed
    // freeze-pointer arena buffer). `!track_op()` keeps the training tape on the
    // differentiable composite, matching the CUDA guard above.
    #[cfg(feature = "rocm")]
    if matches!(x.device(), Device::Rocm(_))
        && matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        && x.is_contiguous()
        && !x.track_op()
    {
        if let Ok(out) = kiln_tensor::rocm_softmax_last_axis(x) {
            return Ok(out);
        }
    }
    // Phase 7 (#1082): when stable KT routes are enabled and `x` is a contiguous CUDA
    // tensor of {F32, BF16, F16}, route the
    // `max_keepdim(D::Minus1)` reduction (the softmax-stabilization
    // max step) through `kiln_tensor::cuda_max_axis` plus a
    // zero-cost `unsqueeze(-1)`. Falls through to the candle
    // composite when any precondition fails so behavior is
    // identical with the gate off.
    let max_val = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_max_last_dim_keepdim(x)? {
                Some(out) => out,
                None => x.max_keepdim(LAST_DIM)?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            x.max_keepdim(LAST_DIM)?
        }
    };
    let shifted = x.broadcast_sub(&max_val)?;
    // Phase 7 (#1082): when stable KT routes are enabled and `shifted` is a
    // contiguous CUDA tensor of {F32, BF16, F16}, route the
    // `.exp()` step of the softmax composite through
    // `kiln_tensor::cuda_activation_unary` with kind 6 (Exp).
    // Falls through to the candle composite when any
    // precondition fails so behavior is identical with the
    // gate off.
    let exp_shifted = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_exp(&shifted)? {
                Some(out) => out,
                None => shifted.exp()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            shifted.exp()?
        }
    };
    // Phase 7 (#1082): when stable KT routes are enabled and `exp_shifted` is a
    // contiguous CUDA tensor of {F32, BF16, F16}, route the
    // `sum_keepdim(-1)` reduction through
    // `kiln_tensor::cuda_sum_last_axis` plus a zero-cost
    // `unsqueeze(-1)`. Falls through to the candle composite when
    // any precondition fails so behavior is identical with the
    // gate off.
    let sum_exp = {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_sum_last_dim_keepdim(&exp_shifted)? {
                out
            } else {
                exp_shifted.sum_keepdim(LAST_DIM)?
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            exp_shifted.sum_keepdim(LAST_DIM)?
        }
    };
    Ok(exp_shifted.broadcast_div(&sum_exp)?)
}

/// Phase 7 (#1082) — kt-API `sum_keepdim(-1)` migration helper.
/// Routes a contiguous CUDA candle tensor through
/// `kiln_tensor::cuda_sum_last_axis` (which reduces the trailing
/// axis) and re-applies `unsqueeze(-1)` so the output shape
/// matches `sum_keepdim`.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the candle composite. NVTX range
/// `kiln/sum_last_dim_kt` brackets the migrated call so nsys
/// traces separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_sum_last_dim_keepdim(x: &Tensor) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), Device::Cuda(_))
        || !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || !x.is_contiguous()
        || x.rank() == 0
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/sum_last_dim_kt");

    let out_kt = match kiln_tensor::cuda_sum_last_axis(x) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let reduced = out_kt;
    let out = reduced
        .unsqueeze(reduced.rank())
        .map_err(|e| anyhow::anyhow!("try_kt_sum_last_dim_keepdim: unsqueeze failed: {e}"))?;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API `sum(axis)` (non-keepdim, arbitrary
/// axis) migration helper. Routes a contiguous CUDA candle tensor
/// through `kiln_tensor::cuda_sum_axis` (which reduces an arbitrary
/// axis with the axis dim removed, matching candle's
/// `Tensor::sum(axis)` semantics directly — no `unsqueeze(-1)`
/// fixup required).
///
/// Returns `Ok(None)` on any incompatibility (non-CUDA, unsupported
/// dtype, non-contiguous, rank-0, or axis out of range) so the
/// caller falls through to the candle composite. NVTX range
/// `kiln/sum_axis_kt` brackets the migrated call so nsys traces
/// separate the path from the baseline composite.
///
/// Distinct from [`try_kt_sum_last_dim_keepdim`] (the trailing-axis
/// + keepdim variant). The kt kernel under both helpers is the
/// same `reduce_arbitrary_axis` (commit `7ca6cabd`) — this helper
/// just exposes its native non-keepdim shape directly to candle
/// `Tensor::sum(axis)` call sites.
#[cfg(feature = "cuda")]
fn try_kt_sum_axis(x: &Tensor, axis: usize) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), Device::Cuda(_))
        || !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || !x.is_contiguous()
        || x.rank() == 0
        || axis >= x.rank()
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/sum_axis_kt");

    let out_kt = match kiln_tensor::cuda_sum_axis(x, axis) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API `max_keepdim(-1)` migration helper.
/// Routes a contiguous CUDA candle tensor through
/// `kiln_tensor::cuda_max_axis` (which reduces an arbitrary axis
/// with the axis dim removed) and re-applies `unsqueeze(-1)` so the
/// output shape matches `max_keepdim`.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the candle composite. NVTX range
/// `kiln/max_last_dim_kt` brackets the migrated call so nsys
/// traces separate the path from the baseline composite.
#[cfg(feature = "cuda")]
pub(crate) fn try_kt_max_last_dim_keepdim(x: &Tensor) -> Result<Option<Tensor>> {
    if !crate::kt_api_policy::stable_routes_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), Device::Cuda(_))
        || !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || !x.is_contiguous()
        || x.rank() == 0
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/max_last_dim_kt");

    // #1082: `x` is already kt — call the kt kernel directly (no bridge).
    let last_axis = x.rank() - 1;
    let out_kt = match kiln_tensor::cuda_max_axis(x, last_axis) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let reduced = out_kt;
    let out = reduced
        .unsqueeze(reduced.rank())
        .map_err(|e| anyhow::anyhow!("try_kt_max_last_dim_keepdim: unsqueeze failed: {e}"))?;
    Ok(Some(out))
}

/// Phase 7 (#1082) — kt-API softmax migration helper. Routes a
/// contiguous CUDA candle tensor through
/// `kiln_tensor::cuda_softmax_last_axis`.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to the candle composite. NVTX range `kiln/softmax_kt`
/// brackets the migrated call so nsys traces separate the path
/// from the baseline composite.
#[cfg(feature = "cuda")]
fn try_kt_softmax_last_dim(x: &Tensor) -> Result<Option<Tensor>> {
    kiln_nvtx::range!(c"kiln/softmax_kt");

    let out_kt = match kiln_tensor::cuda_softmax_last_axis(x) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out = out_kt;
    Ok(Some(out))
}

/// Compute attention using a backend FlashAttention-2 fast path.
///
/// Takes Q, K, V in `[batch, seq_len, num_heads, head_dim]` layout (pre-transpose).
/// K/V may have fewer heads than Q (GQA); they are expanded to match Q's head count
/// before calling the flash kernel, which requires uniform head counts.
///
/// Routes through `AttentionBackend::runtime_flash_attn_prefill`. Returns `Ok(Some(out))` with
/// `out` shaped `[batch, seq_len, num_heads * head_dim]` (already reshaped for
/// output projection) when the backend handles it, or `Ok(None)` when the
/// backend declines — callers must fall back to the portable candle path.
fn flash_attention_forward(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let causal = true;

    // GQA: backends with a grouped-KV flash-prefill ABI receive `num_heads_k`
    // separately and can consume grouped K/V directly. Other backends still
    // take the historic expanded layout through this trait method.
    let (k, v) = if num_heads != num_kv_heads && !flash_prefill_consumes_grouped_kv(backend) {
        let gqa_ratio = num_heads / num_kv_heads;
        let (batch, kv_len, _kv_heads, hd) = k.dims4()?;
        // [batch, kv_len, num_kv_heads, head_dim] -> [batch, kv_len, num_heads, head_dim]
        let k = k
            .unsqueeze(3)?
            .expand([batch, kv_len, num_kv_heads, gqa_ratio, hd])?
            .contiguous()?
            .reshape((batch, kv_len, num_heads, hd))?;
        let v = v
            .unsqueeze(3)?
            .expand([batch, kv_len, num_kv_heads, gqa_ratio, hd])?
            .contiguous()?
            .reshape((batch, kv_len, num_heads, hd))?;
        (k, v)
    } else {
        (k.clone(), v.clone())
    };

    let Some(attn_output) =
        AttentionBackend::runtime_flash_attn_prefill(backend, q, &k, &v, softmax_scale, causal)?
    else {
        return Ok(None);
    };

    // Reshape to [batch, seq_len, hidden]
    let (batch, seq_len, _heads, _hd) = attn_output.dims4()?;
    let attn_output = attn_output.reshape((batch, seq_len, num_heads * head_dim))?;
    Ok(Some(attn_output))
}

// (#1082) Deleted the candle-CustomOp3 `cuda_flash_attention_training_bf16`
//   + `CudaFlashAttentionTrainingBf16` + `cuda_flash_attention_training_disabled`:
//   `crate::tape_forward::try_tape_flash_attn_cuda` is the sole flash-attention
//   autograd producer now.

/// Compute attention using a backend fast path when Q/K/V are already in
/// `[batch, heads, seq_len, head_dim]` layout.
///
/// The paged prefill path transposes Q/K/V to this layout before writing the
/// KV cache. Metal's fused SDPA also consumes this layout, so this variant
/// avoids transposing all three tensors back to token-major only for the
/// backend to transpose them again.
fn flash_attention_forward_head_major(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    if !AttentionBackend::runtime_supports_flash_attn_prefill_head_major(backend) {
        return Ok(None);
    }

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let causal = true;

    let Some(attn_output) = AttentionBackend::runtime_flash_attn_prefill_head_major(
        backend,
        q,
        k,
        v,
        softmax_scale,
        causal,
    )?
    else {
        return Ok(None);
    };

    let (batch, _heads, seq_len, _hd) = attn_output.dims4()?;
    let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
        batch,
        seq_len,
        num_heads * head_dim,
    ))?;
    Ok(Some(attn_output))
}

mod weight_types;
pub use weight_types::*;
mod linear_state;
pub use linear_state::*;
mod weight_loading;
pub(crate) use weight_loading::*;
mod execution_policy;
pub use execution_policy::*;
mod primitives;
pub use primitives::*;
mod ffn;
pub use ffn::*;
mod lm_head;
pub use lm_head::*;
mod linear_attention;
pub use linear_attention::*;
mod training_primitives;
pub use training_primitives::*;
mod linear_attention_streaming;
pub use linear_attention_streaming::*;
mod full_attention;
pub use full_attention::*;
mod transformer;
pub use transformer::*;
mod model_dispatch;
pub use model_dispatch::*;

#[cfg(test)]
mod tests;
