//! End-to-end autoregressive text generation pipeline.
//!
//! Wires together tokenizer, model weights, forward pass, and sampling into
//! a `ModelRunner` that accepts text prompts and produces text output.

use anyhow::{Context, Result};

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, AtomicUsize, Ordering},
    mpsc,
};
use std::time::Instant;

use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::KilnTokenizer;

#[cfg(test)]
use crate::backend::capability::DecodeBatcherPolicy;
use crate::backend::{
    self, BackendIdentity, BackendRuntime, GdnRecurrentStateResidencyStats, LinearBackend,
    ReplayBackend, ResidencyBackend, SamplingBackend, StartupBackend, TrainingLossBackend,
    TrainingPrecisionPolicy,
    capability::{
        BackendCapabilities, BackendCapabilityQueries, ReplayNativePrimitive, ReplayRequest,
        Support, decode_hot_path_fallback_policy_for_backend,
        decode_hot_path_generic_fallback_enabled_for_backend,
    },
};
use crate::cancel::CancelHandle;
use crate::cuda_graph::CudaGraphRunner;
use crate::decode_buffers::{DecodeBufferConfig, DecodeBuffers, DecodeElementType};
use crate::forward::lm_head_sample_backend_decode_if;
use crate::forward::{
    GpuWeights, LinearAttentionState, PagedLayerForwardState, StreamingPrefillExecutionPolicy,
    model_forward_kt_with_policy, model_forward_paged, model_forward_paged_batched_decode_hidden,
    model_forward_paged_decode_contiguous_batch_greedy_with_ids,
    model_forward_paged_decode_contiguous_batch_hidden_with_ids,
    model_forward_paged_decode_contiguous_batch_sample_with_ids, model_forward_paged_last_token,
    model_forward_paged_last_token_greedy, model_forward_paged_last_token_layer_group,
    model_forward_paged_last_token_with_last_hidden, model_forward_paged_next_token_greedy,
    model_forward_paged_streaming_last_token_with_last_hidden_with_policy,
    model_forward_paged_streaming_with_policy,
    model_forward_paged_streaming_with_progress_and_policy,
    model_forward_paged_streaming_with_progress_offset_and_policy,
};
use crate::metal_graph::MetalGraphRunner;
use crate::rocm_graph::{
    RocmGraphExecutionPolicy, RocmGraphLiveTelemetry, RocmGraphRunner, RocmGraphStatsUnavailable,
    RocmGraphTelemetryHandle,
};
// (#1082) Native single-submit Vulkan-resident decode entry — only referenced
// from the `#[cfg(feature = "vulkan")]` single-row fast path below.
#[cfg(feature = "vulkan")]
use crate::forward::model_forward_paged_last_token_resident;
use crate::kv_cache::KvCache;
use crate::lora_loader::LoraWeights;
use crate::packed_weight_registry::GpuPackedWeightRegistry;
// (#1082) the candle `crate::paged_kv_cache` module is gone; the kt twin
// `PagedKvCacheKt` is the production cache. Alias it to `PagedKvCache` so the
// existing call sites + the `model_forward_paged*` params (which the PAGED
// agent resolves to the same kt cache) converge on one type.
use crate::paged_kv_cache_kt::PagedKvCacheKt as PagedKvCache;
use crate::sampling::{
    SampledToken, greedy_sample, sample_step, sample_step_with_logprob, sample_with_full_params,
};
use crate::speculative::{
    SpeculativeConfig, speculative_decode_step, speculative_decode_step_paged_greedy,
    speculative_mtp_decode_step,
};

use kiln_core::block::{BlockManager, BlockTable};

const SPECULATIVE_GENERATION_UNAVAILABLE_REASON: &str = "speculative generation is disabled pending cancellation-safe owner settlement and local accelerator qualification";

#[inline]
fn ensure_speculative_generation_available() -> Result<()> {
    anyhow::bail!(SPECULATIVE_GENERATION_UNAVAILABLE_REASON)
}

/// Returns `Err` with a stable error message if `cancel` has been signalled.
///
/// Decode loops poll this between tokens so that `kiln-server` can drain a
/// `tokio::task::spawn_blocking` whose outer `tokio::time::timeout` already
/// fired, instead of leaving it running with locks held (see #664).
#[inline]
fn check_cancelled(cancel: Option<&CancelHandle>) -> Result<()> {
    if let Some(c) = cancel {
        if c.is_cancelled() {
            anyhow::bail!("generation cancelled by client (request timeout)");
        }
    }
    Ok(())
}

/// (#1082) Map the model config dtype to the kt `DType` the kt paged cache
/// (`PagedKvCacheKt::new`) expects.
fn paged_cache_kt_dtype(dtype: kiln_core::config::DType) -> kiln_tensor::DType {
    match dtype {
        kiln_core::config::DType::BF16 => kiln_tensor::DType::BF16,
        kiln_core::config::DType::FP16 => kiln_tensor::DType::F16,
        kiln_core::config::DType::FP32 => kiln_tensor::DType::F32,
    }
}

/// (#1082) Validate + return the device the kt paged cache allocates its
/// pools on. The cache now allocates per-arm on the model's *runtime* device
/// (`PagedKvCacheKt::new_with_fp8` matches on the `Device`), so we hand it the
/// `Device` directly instead of a bare index. Native MTP generation support is
/// backend-owned capability data; unsupported backends fail before allocating
/// speculative caches.
fn paged_cache_device(
    backend: &dyn BackendRuntime,
    device: &kiln_tensor::Device,
) -> Result<kiln_tensor::Device> {
    let support = BackendCapabilityQueries::backend_capabilities(backend)
        .decode
        .mtp_speculative_generation;
    if matches!(support, Support::Native | Support::NativeWithConstraints) {
        Ok(*device)
    } else {
        anyhow::bail!("native MTP speculative generation requires backend support; got {support:?}")
    }
}

fn fast_batched_linear_state_scatter_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_FAST_BATCHED_LINEAR_STATE_SCATTER").is_err())
}

fn skip_final_gdn_state_readback_enabled() -> bool {
    #[cfg(feature = "vulkan")]
    {
        kiln_vulkan_kernel::kernels::QUALIFIED_VULKAN_KERNEL_POLICY
            .skip_final_gdn_state_readback_enabled
    }
    #[cfg(not(feature = "vulkan"))]
    {
        true
    }
}

struct GdnRecurrentResidentStateScope<'a> {
    backend: &'a dyn BackendRuntime,
    active: bool,
}

impl<'a> GdnRecurrentResidentStateScope<'a> {
    fn new(backend: &'a dyn BackendRuntime) -> Self {
        let active = ResidencyBackend::runtime_enter_gdn_recurrent_resident_state_scope(backend);
        Self { backend, active }
    }
}

impl Drop for GdnRecurrentResidentStateScope<'_> {
    fn drop(&mut self) {
        if self.active {
            ResidencyBackend::runtime_exit_gdn_recurrent_resident_state_scope(self.backend);
        }
    }
}

struct GdnPrefillResidentStateScope<'a> {
    backend: &'a dyn BackendRuntime,
    active: bool,
}

impl<'a> GdnPrefillResidentStateScope<'a> {
    fn new(backend: &'a dyn BackendRuntime, owner_id: Option<u64>) -> Self {
        let active = owner_id.is_some_and(|owner_id| {
            ResidencyBackend::runtime_enter_gdn_prefill_resident_state_scope(backend, owner_id)
        });
        Self { backend, active }
    }
}

impl Drop for GdnPrefillResidentStateScope<'_> {
    fn drop(&mut self) {
        if self.active {
            ResidencyBackend::runtime_exit_gdn_prefill_resident_state_scope(self.backend);
        }
    }
}

fn env_truthy_for_profile(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn profile_decode_batcher_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_DECODE_BATCHER_STAGES"))
}

/// #1082 CRASHER FIX: detect whether any row's KV pages are NOT physically
/// contiguous within a kBlockN-token tile — the contract the vendored FA2
/// split-KV paged-decode kernel silently assumes (it reads each tile as one
/// contiguous gather from `block_table[base_idx]`, never consulting the
/// intervening entries). When a fragmented free list hands the kernel
/// non-adjacent pages it reads a foreign page / off the pool →
/// CUDA_ERROR_ILLEGAL_ADDRESS. Mirrors the bs=1 check in
/// `forward.rs::try_flash_attn_paged_decode`.
///
/// Chunk = `FA2_KBLOCK_N` tokens. Qwen3.5-4B's GQA full-attn is head_dim=256
/// only, so kBlockN = 64 (`flash_fwd_launch_template.h:170`: hd>128 → 64). A
/// 64-token tile spans `64/block_size` pages, which is the run that must be
/// physically adjacent. With the #1082 default `block_size = 64` this is ONE
/// page per tile → every block_table is trivially "contiguous" → no row ever
/// routes to the slow per-row loop for FA2 reasons (the n=64 fix). At the old
/// `block_size = 16` it is 4 pages/tile. (If a head_dim=128 model is ever
/// served — kiln is Qwen3.5-4B-only today — kBlockN would be 128; bump
/// `FA2_KBLOCK_N` or thread head_dim before then.) Returns true → caller must
/// route to the contiguity-safe per-row decode.
/// FA2 split-KV decode tile width (tokens) for Qwen3.5-4B's head_dim=256 GQA
/// full-attn (`flash_fwd_launch_template.h:170`: hd>128 → 64). The K/V pages
/// backing one tile must be physically adjacent; with `block_size >= 64` a tile
/// is one page so the requirement is vacuous. kiln is Qwen3.5-4B-only (hd=256);
/// a head_dim=128 model would need 128 here (or head_dim threaded through).
pub(crate) const FA2_KBLOCK_N: usize = 64;

pub(crate) fn batch_has_noncontiguous_kv_tiles(
    block_tables: &[&BlockTable],
    seq_lens: &[usize],
    block_size: usize,
) -> bool {
    block_tables.iter().enumerate().any(|(row, bt)| {
        row_has_noncontiguous_kv_tiles(
            bt.blocks.as_slice(),
            seq_lens.get(row).copied().unwrap_or(0),
            block_size,
        )
    })
}

/// Per-row sibling of [`batch_has_noncontiguous_kv_tiles`]: true when THIS row's
/// live KV pages violate the intra-tile physical-contiguity contract the FA2
/// split-KV kernel assumes. Lets the batched-decode partition row-loop only the
/// genuinely-fragmented rows instead of the #1445 all-or-nothing whole-batch
/// serialization that caused the concurrent n=64 cliff (366s p50 -> 43s).
pub(crate) fn row_has_noncontiguous_kv_tiles(
    blocks: &[u32],
    seqlen: usize,
    block_size: usize,
) -> bool {
    if block_size == 0 {
        return false;
    }
    let pages_per_chunk = (FA2_KBLOCK_N / block_size).max(1);
    // Only the pages actually covering live tokens are read by the kernel.
    let n_pages = seqlen.div_ceil(block_size).min(blocks.len());
    let mut c = 0usize;
    while c < n_pages {
        let base = blocks[c];
        let end = (c + pages_per_chunk).min(n_pages);
        for (k, &phys) in blocks[c..end].iter().enumerate() {
            if phys != base.wrapping_add(k as u32) {
                return true;
            }
        }
        c += pages_per_chunk;
    }
    false
}

fn gdn_batched_decode_row_loop_debug_enabled() -> bool {
    // Flipped to false-by-default after the matmul broadcast-copy fix made
    // the true-batched contiguous-batch path strictly faster than the
    // row-loop at every bs > 1. nsys profile (May 2026) showed candle's
    // `broadcast_matmul` materializing a 168 MB BF16 weight copy across the
    // batch dim before every GDN in-proj matmul, which made the batched
    // path slower than just running N row-loop iterations sequentially.
    // With that copy removed, bs=16 jumped from a flat ~100 tok/s ceiling
    // to 790 tok/s (7.8×) on L40S + Qwen3.5-4B. Opt back into the row-loop
    // with `KILN_ENABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP=1` (the old
    // `KILN_DISABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP` env var is still
    // honored for symmetry with prior docs / rollback runbooks — when set
    // to anything other than "0"/"false"/"off" it keeps the row-loop off).
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Legacy disable knob: when set to a truthy value, row-loop stays off
        // (i.e. continues to use the new true-batched path).
        if std::env::var("KILN_DISABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP").is_ok() {
            return false;
        }
        // New opt-in knob to re-enable the row-loop fallback for debug /
        // rollback. Recognizes the common truthy spellings.
        match std::env::var("KILN_ENABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP")
            .ok()
            .as_deref()
        {
            Some("1" | "true" | "TRUE" | "yes" | "on" | "ON") => true,
            _ => false,
        }
    })
}

fn finish_decode_batcher_stage_profile(
    stage: &str,
    batch: usize,
    start: Option<std::time::Instant>,
) {
    let Some(start) = start else {
        return;
    };
    eprintln!(
        "kiln_profile_decode_batcher_stage stage={stage} batch={batch} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

/// Holds loaded model weights and tokenizer, provides text generation.
pub struct ModelRunner {
    pub weights: GpuWeights,
    pub tokenizer: KilnTokenizer,
    pub config: ModelConfig,
    /// EOS token IDs cached from the tokenizer.
    eos_token_ids: Vec<TokenId>,
    /// Currently active LoRA adapter weights (None = base model only).
    active_lora: Option<LoraWeights>,
    /// CUDA graph runner for accelerated decode steps.
    /// Uses Mutex for interior mutability (graph state changes during &self generation).
    cuda_graph: Mutex<CudaGraphRunner>,
    /// ROCm HIP-graph runner for accelerated decode steps (R.9). Independent of
    /// `cuda_graph`; its immutable policy is installed at construction. Same
    /// per-step interior-mutability pattern.
    rocm_graph: Mutex<RocmGraphRunner>,
    /// Capture-phase telemetry deliberately lives outside `rocm_graph`, so a
    /// slow native capture cannot make health reporting look unavailable.
    rocm_graph_telemetry: RocmGraphTelemetryHandle,
    /// Metal ICB graph runner for accelerated decode steps. Active only on a
    /// Metal device with `KILN_METAL_GRAPHS` set; otherwise eager Metal decode
    /// is preserved.
    metal_graph: Mutex<MetalGraphRunner>,
    /// Phase A explicit decode weight registry. Decode kernels address weights
    /// by enum keys instead of safetensors/Candle names.
    /// Phase A.5: lazily built on first hot-path access via `packed_weight_registry()`.
    /// Building eagerly in `ModelRunner::new` measured a 22% c=1 paged decode regression
    /// (Validation #4: 42.6 vs 54.76 baseline), so construction stays cheap and the
    /// registry is materialized only when decode actually needs it.
    packed_weight_registry: OnceLock<GpuPackedWeightRegistry>,
    /// Phase A raw decode buffer pool. The first decode/warmup materializes
    /// stable typed tensors for the active graph bucket, then reuses them.
    /// Phase A.6: `OnceLock` instead of `Mutex<Option<_>>` — the buffer is
    /// allocated once at the largest configured graph bucket
    /// (`decode_buffer_max_batch()`), so subsequent decode steps just need a
    /// `get()` (load-acquire) instead of a `Mutex::lock()` per step.
    decode_buffers: OnceLock<DecodeBuffers>,
    /// Startup-resolved decode-buffer width. The owning product surface can
    /// impose a typed ceiling; backend policy and model-only debug overrides
    /// are resolved once during construction rather than on the hot path.
    decode_buffer_max_batch: usize,
    /// Phase A.5: lazily built on first hot-path access via `ensure_decode_buffers()`.
    /// Mirrors the lazy registry pattern above so `ModelRunner::new` doesn't validate
    /// shapes that decode hasn't asked for yet.
    decode_buffer_config: OnceLock<DecodeBufferConfig>,
    /// Cached batched `LinearAttentionState` carried across consecutive
    /// paged-decode invocations. Exact row-ID matches reuse the state without
    /// a copy. A Vulkan-resident cache also retains its maximum observed batch
    /// capacity across row-set and width changes, refreshing the same backend
    /// buffers through a smaller identity-preserving prefix view. This avoids
    /// allocator churn without retaining one host/device state per width.
    /// The cache is invalidated on adapter transitions with graph state.
    batched_state_cache: Mutex<Option<CachedBatchedState>>,
    batched_state_cache_counters: BatchedStateCacheCounters,
    backend: Arc<dyn BackendRuntime>,
    /// Immutable startup-resolved streaming-prefill execution policy.
    streaming_prefill: StreamingPrefillExecutionPolicy,
    backend_health: BackendHealthHandle,
    memory_runtime: Option<InferenceMemoryRuntime>,
}

impl Drop for ModelRunner {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        if self.backend_health.snapshot().quarantined
            && matches!(
                self.weights.embed_tokens.device(),
                kiln_tensor::Device::Rocm(_)
            )
        {
            if let Some(storage) = self
                .weights
                .embed_tokens
                .storage()
                .as_any()
                .downcast_ref::<kiln_tensor::RocmStorage>()
            {
                // Drop runs before struct fields. Bridge the process-wide
                // backend quarantine into the low-level device quarantine so
                // weights, graph state, workspaces, and streams retain unknown
                // in-flight resources instead of freeing them during shutdown.
                storage.context().quarantine_execution();
            }
        }
    }
}

/// Process-lifetime memory binding for direct inference consumers.
///
/// Construction is the explicit slow startup boundary: it validates that the
/// selected backend and OS memory probe name the same physical accelerator,
/// detects safe capacity, installs the exact governor policy, publishes one
/// live sample, and starts the background sampler. Model constructors remain
/// allocation-only and never probe hardware implicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InferenceMemoryRuntime {
    device: kiln_tensor::Device,
    selector: kiln_memory::VramProbeSelector,
    effective_capacity_bytes: u64,
    governor: kiln_memory::GovernorConfig,
}

impl InferenceMemoryRuntime {
    /// Initialize direct inference memory governance for `device`.
    ///
    /// `governor.capacity_limit_bytes` is a cap, never a capacity override. A
    /// value larger than detected physical capacity is clamped; zero or an
    /// unresolved accelerator probe fails before model construction.
    pub fn initialize(
        device: kiln_tensor::Device,
        mut governor: kiln_memory::GovernorConfig,
    ) -> Result<Self> {
        let selector = device.memory_probe_selector();
        if device.is_cpu() {
            return Ok(Self {
                device,
                selector,
                effective_capacity_bytes: 0,
                governor,
            });
        }

        anyhow::ensure!(
            governor.critical_frac.is_finite()
                && governor.tight_frac.is_finite()
                && governor.comfortable_frac.is_finite()
                && (0.0..=1.0).contains(&governor.critical_frac)
                && governor.critical_frac <= governor.tight_frac
                && governor.tight_frac < governor.comfortable_frac
                && governor.comfortable_frac <= 1.0,
            "invalid inference governor pressure thresholds: require 0 <= critical <= tight < comfortable <= 1"
        );

        kiln_memory::validate_vram_probe_identity(selector).map_err(|error| {
            anyhow::anyhow!(
                "cannot initialize inference memory runtime for {}: {error}",
                device.short_name()
            )
        })?;
        let physical = kiln_memory::detect_vram_for(selector);
        anyhow::ensure!(
            physical.total_bytes > 0,
            "cannot initialize inference memory runtime for {}: probe {:?} established no safe accelerator capacity",
            device.short_name(),
            selector,
        );
        let effective_capacity_bytes = governor
            .capacity_limit_bytes
            .unwrap_or(physical.total_bytes)
            .min(physical.total_bytes);
        anyhow::ensure!(
            effective_capacity_bytes > 0,
            "cannot initialize inference memory runtime for {} with a zero-byte capacity cap",
            device.short_name(),
        );
        anyhow::ensure!(
            governor.floor_bytes < effective_capacity_bytes,
            "inference governor floor {} bytes must be smaller than effective capacity {} bytes",
            governor.floor_bytes,
            effective_capacity_bytes,
        );
        governor.capacity_limit_bytes = Some(effective_capacity_bytes);
        kiln_memory::MemoryGovernor::configure_global(selector, governor)
            .context("configure inference memory governor")?;
        let memory_governor = kiln_memory::MemoryGovernor::global();
        let published = memory_governor.refresh();
        anyhow::ensure!(
            published.total_bytes == effective_capacity_bytes
                && !published.observations.probe_failed,
            "inference memory probe for {} did not publish the bound {}-byte capacity",
            device.short_name(),
            effective_capacity_bytes,
        );
        anyhow::ensure!(
            memory_governor.start_sampler(),
            "failed to start inference memory sampler"
        );
        Ok(Self {
            device,
            selector,
            effective_capacity_bytes,
            governor,
        })
    }

    pub const fn device(&self) -> kiln_tensor::Device {
        self.device
    }

    pub const fn selector(&self) -> kiln_memory::VramProbeSelector {
        self.selector
    }

    pub const fn effective_capacity_bytes(&self) -> u64 {
        self.effective_capacity_bytes
    }

    pub const fn governor_config(&self) -> kiln_memory::GovernorConfig {
        self.governor
    }

    fn is_weight_device_compatible(&self, weight_device: kiln_tensor::Device) -> bool {
        self.device == weight_device
            || matches!(self.device, kiln_tensor::Device::Vulkan(_))
                && matches!(weight_device, kiln_tensor::Device::Cpu)
    }

    fn validate_weight_device(&self, weight_device: kiln_tensor::Device) -> Result<()> {
        anyhow::ensure!(
            self.is_weight_device_compatible(weight_device),
            "inference memory runtime device {} does not match model weight device {}",
            self.device.short_name(),
            weight_device.short_name(),
        );
        if self.device.is_cpu() {
            return Ok(());
        }
        let configured = kiln_memory::MemoryGovernor::global_configuration();
        anyhow::ensure!(
            configured.selector == self.selector && configured.governor == self.governor,
            "inference memory runtime no longer matches the process-wide governor configuration"
        );
        anyhow::ensure!(
            kiln_memory::MemoryGovernor::try_global_cached_snapshot().is_some(),
            "inference memory runtime governor is not initialized"
        );
        Ok(())
    }
}

/// Backend graph eligibility resolved before a [`ModelRunner`] is built.
///
/// The server's stable profile disables every graph backend. Other embedding
/// callers can continue using [`ModelRunner::new_with_options`] for the
/// historical CUDA-only option plus backend defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelRunnerRuntimeOptions {
    pub cuda_graphs: bool,
    pub rocm_graph: RocmGraphExecutionPolicy,
    pub metal_graphs: bool,
    /// Exact width required by an owning scheduler. `None` leaves standalone
    /// model consumers on backend/debug defaults.
    pub max_decode_batch: Option<usize>,
    /// Owning-product streaming-prefill policy. `None` preserves the selected
    /// backend's automatic defaults for standalone and compatibility callers.
    pub streaming_prefill: Option<StreamingPrefillExecutionPolicy>,
}

impl ModelRunnerRuntimeOptions {
    pub const fn eager_only() -> Self {
        Self {
            cuda_graphs: false,
            rocm_graph: RocmGraphExecutionPolicy::disabled(),
            metal_graphs: false,
            max_decode_batch: None,
            streaming_prefill: None,
        }
    }
}

impl Default for ModelRunnerRuntimeOptions {
    fn default() -> Self {
        Self::eager_only()
    }
}

/// Process-lifetime health gate for a backend whose asynchronous completion
/// can no longer be proven. Once quarantined, inference must remain disabled
/// until the process is restarted; reusing mutable GPU state would be unsafe.
#[derive(Clone, Debug, Default)]
pub struct BackendHealthHandle {
    inner: Arc<BackendHealthState>,
}

#[derive(Debug, Default)]
struct BackendHealthState {
    reason: Mutex<Option<String>>,
    external_yield_sync: Mutex<BTreeMap<&'static str, ExternalYieldSyncStats>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendHealthSnapshot {
    pub quarantined: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ExternalYieldSyncStats {
    pub boundary: String,
    pub calls: u64,
    pub failures: u64,
    pub total_micros: u64,
    pub max_micros: u64,
    pub slow_calls: u64,
}

const SLOW_EXTERNAL_YIELD_SYNC: std::time::Duration = std::time::Duration::from_millis(100);

impl BackendHealthHandle {
    pub fn quarantine(&self, reason: impl Into<String>) {
        let reason = reason.into();
        let mut stored = self
            .inner
            .reason
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if stored.is_none() {
            *stored = Some(reason.clone());
        }
        drop(stored);
        tracing::error!(
            event = "backend_quarantined",
            reason,
            "backend quarantined; restart is required before inference can resume"
        );
    }

    pub fn snapshot(&self) -> BackendHealthSnapshot {
        let reason = self
            .inner
            .reason
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        BackendHealthSnapshot {
            quarantined: reason.is_some(),
            reason,
        }
    }

    pub fn ensure_healthy(&self) -> Result<()> {
        let snapshot = self.snapshot();
        anyhow::ensure!(
            !snapshot.quarantined,
            "backend is quarantined and requires restart: {}",
            snapshot
                .reason
                .as_deref()
                .unwrap_or("unknown completion state")
        );
        Ok(())
    }

    fn record_external_yield_sync(
        &self,
        boundary: &'static str,
        elapsed: std::time::Duration,
        failed: bool,
    ) {
        let elapsed_micros = elapsed.as_micros().min(u64::MAX as u128) as u64;
        let mut stats = self
            .inner
            .external_yield_sync
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = stats
            .entry(boundary)
            .or_insert_with(|| ExternalYieldSyncStats {
                boundary: boundary.to_string(),
                ..ExternalYieldSyncStats::default()
            });
        entry.calls = entry.calls.saturating_add(1);
        entry.total_micros = entry.total_micros.saturating_add(elapsed_micros);
        entry.max_micros = entry.max_micros.max(elapsed_micros);
        if failed {
            entry.failures = entry.failures.saturating_add(1);
        }
        if elapsed >= SLOW_EXTERNAL_YIELD_SYNC {
            entry.slow_calls = entry.slow_calls.saturating_add(1);
        }
    }

    pub fn external_yield_sync_stats(&self) -> Vec<ExternalYieldSyncStats> {
        self.inner
            .external_yield_sync
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .cloned()
            .collect()
    }
}

/// Persistent batched-state cache entry. `state.batch_size()` is its retained
/// capacity, while `row_ids` names the logical prefix whose device bytes are
/// currently valid. Stable request IDs survive batching-actor `Vec::remove`
/// shifts that invalidate pointer fingerprints.
pub(crate) struct CachedBatchedState {
    pub(crate) state: crate::forward::LinearAttentionState,
    pub(crate) row_ids: Vec<u64>,
}

/// Process-lifetime lifecycle and current-ownership snapshot for the recurrent
/// state shared by native batched decode calls.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
pub struct BatchedStateCacheStats {
    pub entry_present: bool,
    pub capacity_rows: usize,
    pub logical_rows: usize,
    pub resident: bool,
    pub active_leases: u64,
    pub max_active_leases: u64,
    pub take_hit_count: u64,
    pub take_miss_count: u64,
    pub take_miss_while_leased_count: u64,
    pub exact_reuse_count: u64,
    pub resident_capacity_reuse_count: u64,
    pub resident_prefix_view_count: u64,
    pub resident_refresh_count: u64,
    pub fresh_assembly_count: u64,
    pub rejected_missing_row_ids_count: u64,
    pub rejected_nonresident_rows_count: u64,
    pub rejected_nonresident_cache_count: u64,
    pub rejected_insufficient_capacity_count: u64,
    pub park_count: u64,
    pub park_replacement_eviction_count: u64,
    pub explicit_invalidation_count: u64,
    pub explicit_invalidation_eviction_count: u64,
    pub completed_row_preservation_count: u64,
    pub completed_row_eviction_count: u64,
    pub lease_drop_eviction_count: u64,
    /// Whole-prompt, strict-prefix, or rolling snapshots deliberately not
    /// captured because a backend-resident GDN buffer, rather than the logical
    /// tensors, owned the current recurrent and convolution state.
    pub resident_prefix_snapshot_suppression_count: u64,
}

#[derive(Debug, Default)]
struct BatchedStateCacheCounters {
    active_leases: AtomicU64,
    max_active_leases: AtomicU64,
    take_hit_count: AtomicU64,
    take_miss_count: AtomicU64,
    take_miss_while_leased_count: AtomicU64,
    exact_reuse_count: AtomicU64,
    resident_capacity_reuse_count: AtomicU64,
    resident_prefix_view_count: AtomicU64,
    resident_refresh_count: AtomicU64,
    fresh_assembly_count: AtomicU64,
    rejected_missing_row_ids_count: AtomicU64,
    rejected_nonresident_rows_count: AtomicU64,
    rejected_nonresident_cache_count: AtomicU64,
    rejected_insufficient_capacity_count: AtomicU64,
    park_count: AtomicU64,
    park_replacement_eviction_count: AtomicU64,
    explicit_invalidation_count: AtomicU64,
    explicit_invalidation_eviction_count: AtomicU64,
    completed_row_preservation_count: AtomicU64,
    completed_row_eviction_count: AtomicU64,
    lease_drop_eviction_count: AtomicU64,
    resident_prefix_snapshot_suppression_count: AtomicU64,
}

impl BatchedStateCacheCounters {
    fn acquire_lease(&self) {
        let active = self.active_leases.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_active_leases.fetch_max(active, Ordering::Relaxed);
    }

    fn release_lease(&self) {
        let previous = self.active_leases.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(previous > 0, "batched-state lease counter underflow");
    }

    fn snapshot(&self) -> BatchedStateCacheStats {
        BatchedStateCacheStats {
            active_leases: self.active_leases.load(Ordering::Relaxed),
            max_active_leases: self.max_active_leases.load(Ordering::Relaxed),
            take_hit_count: self.take_hit_count.load(Ordering::Relaxed),
            take_miss_count: self.take_miss_count.load(Ordering::Relaxed),
            take_miss_while_leased_count: self.take_miss_while_leased_count.load(Ordering::Relaxed),
            exact_reuse_count: self.exact_reuse_count.load(Ordering::Relaxed),
            resident_capacity_reuse_count: self
                .resident_capacity_reuse_count
                .load(Ordering::Relaxed),
            resident_prefix_view_count: self.resident_prefix_view_count.load(Ordering::Relaxed),
            resident_refresh_count: self.resident_refresh_count.load(Ordering::Relaxed),
            fresh_assembly_count: self.fresh_assembly_count.load(Ordering::Relaxed),
            rejected_missing_row_ids_count: self
                .rejected_missing_row_ids_count
                .load(Ordering::Relaxed),
            rejected_nonresident_rows_count: self
                .rejected_nonresident_rows_count
                .load(Ordering::Relaxed),
            rejected_nonresident_cache_count: self
                .rejected_nonresident_cache_count
                .load(Ordering::Relaxed),
            rejected_insufficient_capacity_count: self
                .rejected_insufficient_capacity_count
                .load(Ordering::Relaxed),
            park_count: self.park_count.load(Ordering::Relaxed),
            park_replacement_eviction_count: self
                .park_replacement_eviction_count
                .load(Ordering::Relaxed),
            explicit_invalidation_count: self.explicit_invalidation_count.load(Ordering::Relaxed),
            explicit_invalidation_eviction_count: self
                .explicit_invalidation_eviction_count
                .load(Ordering::Relaxed),
            completed_row_preservation_count: self
                .completed_row_preservation_count
                .load(Ordering::Relaxed),
            completed_row_eviction_count: self.completed_row_eviction_count.load(Ordering::Relaxed),
            lease_drop_eviction_count: self.lease_drop_eviction_count.load(Ordering::Relaxed),
            resident_prefix_snapshot_suppression_count: self
                .resident_prefix_snapshot_suppression_count
                .load(Ordering::Relaxed),
            ..BatchedStateCacheStats::default()
        }
    }
}

#[cfg(test)]
fn completed_row_invalidates_batched_state_cache(
    cached_row_ids: &[u64],
    completed_row_id: u64,
    cache_is_resident: bool,
) -> bool {
    cached_row_ids.contains(&completed_row_id) && !cache_is_resident
}

/// Owns a temporary assembled state until it is explicitly parked in the
/// persistent cache. Any early return after partial assembly or forward work
/// releases backend-resident buffers instead of orphaning their tensor IDs.
struct ResidentBatchedStateLease<'a> {
    state: Option<LinearAttentionState>,
    /// Maximum-capacity owner when `state` is a smaller identity-preserving
    /// prefix view. Both name the same backend buffers, so cleanup must evict
    /// exactly once through this owner.
    capacity_state: Option<LinearAttentionState>,
    backend: &'a dyn BackendRuntime,
    counters: &'a BatchedStateCacheCounters,
    tracked: bool,
}

impl<'a> ResidentBatchedStateLease<'a> {
    fn new(
        state: Option<LinearAttentionState>,
        backend: &'a dyn BackendRuntime,
        counters: &'a BatchedStateCacheCounters,
    ) -> Self {
        let tracked = state.is_some();
        if tracked {
            counters.acquire_lease();
        }
        Self {
            state,
            capacity_state: None,
            backend,
            counters,
            tracked,
        }
    }

    fn with_capacity_view(
        state: LinearAttentionState,
        capacity_state: LinearAttentionState,
        backend: &'a dyn BackendRuntime,
        counters: &'a BatchedStateCacheCounters,
    ) -> Self {
        counters.acquire_lease();
        Self {
            state: Some(state),
            capacity_state: Some(capacity_state),
            backend,
            counters,
            tracked: true,
        }
    }

    fn release_tracking(&mut self) {
        if self.tracked {
            self.counters.release_lease();
            self.tracked = false;
        }
    }

    fn as_ref(&self) -> Option<&LinearAttentionState> {
        self.state.as_ref()
    }

    fn as_mut(&mut self) -> Option<&mut LinearAttentionState> {
        self.state.as_mut()
    }

    fn take(&mut self) -> Option<LinearAttentionState> {
        let state = self.state.take();
        if state.is_some() {
            self.release_tracking();
        }
        state
    }

    fn take_for_cache(&mut self) -> Option<LinearAttentionState> {
        let state = match self.capacity_state.take() {
            Some(capacity_state) => {
                self.state.take();
                Some(capacity_state)
            }
            None => self.state.take(),
        };
        if state.is_some() {
            self.release_tracking();
        }
        state
    }
}

impl Drop for ResidentBatchedStateLease<'_> {
    fn drop(&mut self) {
        let state = self.capacity_state.take().or_else(|| self.state.take());
        self.state.take();
        if let Some(state) = state {
            self.release_tracking();
            self.counters
                .lease_drop_eviction_count
                .fetch_add(1, Ordering::Relaxed);
            state.evict_gdn_state_resident_kt(self.backend);
        }
    }
}

/// Output from a generation call.
#[derive(Debug)]
pub struct GenerationOutput {
    /// The generated text (not including the prompt).
    pub text: String,
    /// The generated token IDs (not including prompt tokens).
    pub token_ids: Vec<TokenId>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// A block-aligned paged prefix that can be reused by a later prompt.
pub struct PagedPrefixReuse {
    pub cached_tokens: usize,
    pub block_ids: Vec<u32>,
    pub linear_state: LinearAttentionState,
    pub next_token: Option<PagedPrefixNextToken>,
}

/// A completed block-aligned prompt prefix produced by generation.
pub struct PagedPrefixRegistration {
    pub prompt_tokens: Vec<TokenId>,
    pub block_ids: Vec<u32>,
    pub linear_state: LinearAttentionState,
    pub next_token: Option<PagedPrefixNextToken>,
}

/// Saved first-token source for an exact prompt-cache hit.
#[derive(Clone)]
pub enum PagedPrefixNextToken {
    /// Full last-position logits. Supports both greedy and stochastic sampling.
    // (#1082) kt-native logits — forward + sampler are both kt; no candle bridge.
    Logits(kiln_tensor::Tensor),
    /// Greedy token only. Usable only when the later request is also greedy.
    GreedyToken(TokenId),
}

/// Result of paged generation plus an optional prefix-cache registration.
pub struct PrefixCachedGenerationOutput {
    pub output: GenerationOutput,
    pub registration: Option<PagedPrefixRegistration>,
    /// Additional block-aligned registrations covering positions strictly
    /// less than the full prompt, captured opportunistically during prefill
    /// or decode. These exist so multi-turn agentic loops (e.g. pi) can hit
    /// the cache on subsequent turns when the chat template's generation
    /// prompt differs from how the same assistant message is rendered in
    /// history on later turns. For Qwen3.5 with enable_thinking=false the
    /// generation prompt appends `<|im_start|>assistant\n<think>\n\n</think>\n\n`,
    /// while later-turn history renders the same assistant turn as just
    /// `<|im_start|>assistant\n{content}<|im_end|>\n` — the only way to
    /// share KV across turns there is to register an entry whose token
    /// sequence stops before that divergent tail.
    pub extra_registrations: Vec<PagedPrefixRegistration>,
    pub allocated_blocks: Vec<u32>,
    pub prefill_duration: std::time::Duration,
    pub decode_duration: std::time::Duration,
}

/// Snapshot of the recurrent linear-attention state taken when decode crosses
/// a block-aligned position. Used at request finish time to register an
/// extended prefix-cache entry covering the prompt + the assistant tokens
/// emitted so far. Without this, only the prompt is cached and every
/// follow-up turn re-prefills the entire growing conversation from scratch.
pub struct RollingPrefixSnapshot {
    /// Total position covered by the snapshot (number of leading tokens with
    /// committed KV state). Always a multiple of the block size.
    pub position: usize,
    pub linear_state: LinearAttentionState,
}

/// Per-request state owned by the server batching actor between prefill and
/// decode iterations.
pub struct PagedBatchedDecodeState {
    pub block_table: BlockTable,
    pub linear_state: LinearAttentionState,
    pub seq_len: usize,
    pub next_token: TokenId,
    /// Exact post-filter behavior log-probability for `next_token` when this
    /// row opted into rollout provenance capture.
    pub next_token_logprob: Option<f32>,
    pub generated_tokens: Vec<TokenId>,
    pub step_seed: Option<u64>,
    /// Trace-mode rows deliberately bypass token-only fused sampling paths so
    /// every accepted model action has an exact selected-token probability.
    pub capture_behavior_logprobs: bool,
    pub registration: Option<PagedPrefixRegistration>,
    pub allocated_blocks: Vec<u32>,
    pub prefill_duration: std::time::Duration,
    pub decode_duration: std::time::Duration,
    /// Original prompt tokens, retained so finish-time prefix registration
    /// can synthesize an "extended" entry covering prompt + decoded tokens.
    pub prompt_tokens: Vec<TokenId>,
    /// Block size of the paged KV cache. Stored so the per-step decode
    /// loop can detect block-aligned positions without re-locking the
    /// block manager.
    pub block_size: usize,
    /// Snapshot of the linear-attention state taken during prefill at the
    /// largest block-aligned offset strictly less than the prompt length.
    /// Used to register a cross-turn-safe prefix-cache entry whose token
    /// sequence stops before any chat-template generation-prompt tail.
    pub prefill_split_snapshot: Option<RollingPrefixSnapshot>,
    /// Latest block-aligned snapshot of the linear attention state, taken
    /// during decode. None until decode first crosses a block boundary;
    /// replaced (drop+alloc) at each subsequent boundary.
    pub rolling_snapshot: Option<RollingPrefixSnapshot>,
    /// Whether the generic paged KV cache contains every position represented
    /// by this row. Native Vulkan token-prefill writes later prompt positions
    /// only to its resident KV cache, so those rows must not publish generic
    /// prefix-cache registrations at completion.
    pub prefix_cache_registration_allowed: bool,
    /// Stable per-generation identity used for decode graph and state caching
    /// keys. Assigned from the same process-global namespace as direct
    /// generation owners so no two live decode rows can alias. The value is
    /// independent of where the `PagedBatchedDecodeState` happens to live
    /// in memory — important because the batching-engine actor's
    /// `Vec<ActiveRequest>` shifts elements down via `Vec::remove` when a
    /// request finishes mid-batch, which moves the surrounding
    /// `PagedBatchedDecodeState`s to new memory addresses. A pointer-based
    /// cache key would lose its hits on every such shift; this stable id
    /// survives them.
    pub id: u64,
}

fn capture_authoritative_prefix_snapshot(
    backend: &dyn BackendRuntime,
    resident_prefix_snapshot_suppression_count: &AtomicU64,
    linear_state: &LinearAttentionState,
    snapshot_kind: &'static str,
    position: usize,
) -> Result<Option<LinearAttentionState>> {
    // Vulkan's native prefill and decode stacks can advance recurrent/conv
    // state in backend-private storage without updating these logical tensors.
    // A snapshot is useful only while the logical representation is the
    // authority; publishing it otherwise poisons a later prefix-cache hit.
    if linear_state.has_any_gdn_state_resident_kt(backend)
        || linear_state.has_any_gdn_recurrent_resident_state(backend)
    {
        resident_prefix_snapshot_suppression_count.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(
            snapshot_kind,
            position,
            "suppressed stale prefix snapshot while backend-resident GDN state is authoritative"
        );
        return Ok(None);
    }
    Ok(Some(linear_state.snapshot()?))
}

fn complete_paged_batched_decode_step(
    backend: &dyn BackendRuntime,
    resident_prefix_snapshot_suppression_count: &AtomicU64,
    states: &mut [&mut PagedBatchedDecodeState],
    decode_duration: std::time::Duration,
) {
    for state in states {
        state.seq_len += 1;
        state.decode_duration += decode_duration;
        // A resident Vulkan decode advances backend-private recurrent, conv,
        // and KV buffers without writing the logical tensors or generic paged
        // cache. Copying those logical tensors here would publish an internally
        // inconsistent prefix entry.
        if state.prefix_cache_registration_allowed
            && state.block_size > 0
            && state.seq_len % state.block_size == 0
        {
            match capture_authoritative_prefix_snapshot(
                backend,
                resident_prefix_snapshot_suppression_count,
                &state.linear_state,
                "rolling",
                state.seq_len,
            ) {
                Ok(Some(snap)) => {
                    state.rolling_snapshot = Some(RollingPrefixSnapshot {
                        position: state.seq_len,
                        linear_state: snap,
                    });
                }
                Ok(None) => continue,
                Err(err) => {
                    tracing::warn!(
                        seq_len = state.seq_len,
                        block_size = state.block_size,
                        error = %err,
                        "failed to snapshot linear state at block boundary; \
                         extended prefix-cache entry will not be available for this request"
                    );
                }
            }
        }
    }
}

/// Resumable prompt-prefill ownership for the server batching actor.
///
/// A state is created after prefix lookup and block allocation, then advanced
/// by bounded token quanta. It owns every resource that the monolithic
/// `prepare_paged_batched_decode_with_prefix_cache` path owns, so cancellation
/// can release it and an unsettled accelerator failure can retain it.
pub struct PagedBatchedPrefillState {
    block_table: BlockTable,
    linear_state: LinearAttentionState,
    prompt_tokens: Vec<TokenId>,
    cached_tokens: usize,
    next_position: usize,
    block_size: usize,
    allocated_blocks: Vec<u32>,
    split_pos: Option<usize>,
    prefill_split_snapshot: Option<LinearAttentionState>,
    streaming: bool,
    prefill_duration: std::time::Duration,
    capture_behavior_logprobs: bool,
    /// Keeps an intermediate chunk's output alive until the caller performs
    /// its mandatory external-yield synchronization.
    pending_logits: Option<kiln_tensor::Tensor>,
    /// Forward-local GPU tensors retained when the current token chunk yields
    /// before all transformer layers have run.
    pending_layer_forward: Option<PagedLayerForwardState>,
    /// Exclusive end position of the in-flight layer-resumable token chunk.
    pending_chunk_end: Option<usize>,
    /// Stable identity shared with the eventual decode row. Native Vulkan
    /// prefill uses it to retain exact per-row KV and recurrent-state ownership
    /// across changing actor batch shapes.
    id: u64,
    /// Once native token-prefill writes a row into the resident Vulkan KV
    /// cache, the generic paged cache is no longer authoritative for later
    /// positions. Keep the row on the native route until completion/discard.
    resident_token_prefill_started: bool,
}

impl PagedBatchedPrefillState {
    pub fn processed_tokens(&self) -> usize {
        self.next_position.saturating_sub(self.cached_tokens)
    }

    pub fn remaining_tokens(&self) -> usize {
        self.prompt_tokens.len().saturating_sub(self.next_position)
    }

    pub fn has_pending_layer_progress(&self) -> bool {
        self.pending_layer_forward.is_some()
    }

    /// Width selected when the current token chunk first entered its layer
    /// groups. Later actor cycles must preserve this reservation; shrinking it
    /// would require replaying already-completed layers with different inputs.
    pub fn pending_layer_chunk_tokens(&self) -> Option<usize> {
        self.pending_layer_forward.as_ref()?;
        self.pending_chunk_end
            .map(|chunk_end| chunk_end.saturating_sub(self.next_position))
    }

    pub fn into_allocated_blocks(self) -> Vec<u32> {
        self.allocated_blocks
    }

    pub fn row_id(&self) -> u64 {
        self.id
    }

    pub fn resident_token_prefill_started(&self) -> bool {
        self.resident_token_prefill_started
    }

    pub fn resident_token_prefill_candidate(&self, params: &SamplingParams) -> bool {
        self.next_position > 0
            && self.remaining_tokens() > 0
            && self.pending_layer_forward.is_none()
            && self.pending_chunk_end.is_none()
            && !self.capture_behavior_logprobs
            && params.is_effectively_greedy()
    }
}

/// Initial result of prefix lookup and paged prefill allocation.
pub enum PagedBatchedPrefillStart {
    Ready(PagedBatchedDecodeState),
    Prefilling(PagedBatchedPrefillState),
}

/// Result of one bounded prefill quantum.
pub struct PagedBatchedPrefillProgress {
    /// Prompt tokens whose transformer work was admitted by this call. A
    /// layer-resumable chunk reports its width exactly once, when selected;
    /// later layer groups keep this at zero.
    pub tokens_scheduled: usize,
    /// Prompt tokens whose final transformer layer completed in this call.
    pub tokens_processed: usize,
    pub layers_processed: usize,
    pub decode_state: Option<PagedBatchedDecodeState>,
}

static DECODE_ROW_NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn allocate_decode_row_id(counter: &std::sync::atomic::AtomicU64) -> u64 {
    counter
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |current| match current {
                0 => None,
                u64::MAX => Some(0),
                _ => Some(current + 1),
            },
        )
        .unwrap_or_else(|_| panic!("decode row id namespace exhausted"))
}

pub(crate) fn next_decode_row_id() -> u64 {
    allocate_decode_row_id(&DECODE_ROW_NEXT_ID)
}

/// Owns one direct generation's ROCm graph row until its decode loop exits.
///
/// Direct generation has many early exits (EOS, stop sequences, cancellation,
/// receiver disconnects, and forward errors). Tying cleanup to the stack scope
/// ensures all of them release captured graphs and continuity state before the
/// caller can recycle this generation's KV blocks.
struct RocmDecodeOwnerLease<'a> {
    graph: &'a Mutex<RocmGraphRunner>,
    backend_health: BackendHealthHandle,
    row_id: u64,
}

impl<'a> RocmDecodeOwnerLease<'a> {
    fn new(graph: &'a Mutex<RocmGraphRunner>, backend_health: &BackendHealthHandle) -> Self {
        Self {
            graph,
            backend_health: backend_health.clone(),
            row_id: next_decode_row_id(),
        }
    }

    fn row_id(&self) -> u64 {
        self.row_id
    }
}

impl Drop for RocmDecodeOwnerLease<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.backend_health.quarantine(format!(
                "direct decode row {} panicked before completion was proven",
                self.row_id
            ));
        }
        if self.backend_health.snapshot().quarantined {
            tracing::error!(
                event = "rocm_decode_owner_quarantined",
                row_id = self.row_id,
                "retaining ROCm graph ownership because backend completion is unknown"
            );
            return;
        }
        match self.graph.lock() {
            Ok(mut graph) => graph.release_decode_row(self.row_id),
            Err(poisoned) => {
                tracing::warn!(
                    row_id = self.row_id,
                    "recovering poisoned ROCm graph lock to release direct decode owner"
                );
                poisoned.into_inner().release_decode_row(self.row_id);
            }
        }
    }
}

fn quarantine_linear_attention_state(state: &mut LinearAttentionState) {
    std::mem::forget(std::mem::take(&mut state.recurrent_states));
    std::mem::forget(std::mem::take(&mut state.conv_states));
}

/// Build a strict-prefix prefix-cache registration covering the prompt plus
/// as many decoded assistant tokens as we have a block-aligned linear-state
/// snapshot for. Returns `None` when there's nothing useful to register —
/// e.g. decode never crossed a block boundary, the snapshot's position
/// doesn't extend past the prompt, or the block table doesn't have enough
/// blocks committed (which would indicate a bookkeeping bug upstream).
fn build_extended_registration(
    prompt_tokens: &[TokenId],
    generated_tokens: &[TokenId],
    block_table: &BlockTable,
    block_size: usize,
    rolling_snapshot: Option<RollingPrefixSnapshot>,
) -> Option<PagedPrefixRegistration> {
    let snapshot = rolling_snapshot?;
    if block_size == 0 || snapshot.position == 0 || snapshot.position % block_size != 0 {
        return None;
    }
    let total_available = prompt_tokens.len() + generated_tokens.len();
    if snapshot.position > total_available {
        // Bookkeeping mismatch: snapshot says we have KV for positions
        // beyond what we actually emitted. Skip rather than register a
        // corrupt entry.
        return None;
    }
    let num_blocks = snapshot.position / block_size;
    if num_blocks == 0 || block_table.blocks.len() < num_blocks {
        return None;
    }
    // Build the prompt-token sequence corresponding to this snapshot. When
    // the snapshot is inside the prompt, the entry covers a strict prefix
    // of the prompt (cross-turn-safe — the chat template's generation tail
    // is usually past this point). When the snapshot is past the prompt,
    // the entry covers prompt + decoded tokens (only safe when subsequent
    // turns re-render the assistant message verbatim, i.e. no template
    // divergence — Qwen3.5 with enable_thinking=false does have such a
    // divergence, so prefer the prefill-split-side snapshot there).
    let mut combined = Vec::with_capacity(snapshot.position);
    let prompt_take = prompt_tokens.len().min(snapshot.position);
    combined.extend_from_slice(&prompt_tokens[..prompt_take]);
    let extra_generated = snapshot.position.saturating_sub(prompt_tokens.len());
    if extra_generated > 0 {
        combined.extend_from_slice(&generated_tokens[..extra_generated]);
    }
    debug_assert_eq!(combined.len(), snapshot.position);
    Some(PagedPrefixRegistration {
        prompt_tokens: combined,
        block_ids: block_table.blocks[..num_blocks].to_vec(),
        linear_state: snapshot.linear_state,
        next_token: None,
    })
}

fn strict_prompt_prefix_split_pos(
    prompt_len: usize,
    cached_tokens: usize,
    block_size: usize,
) -> Option<usize> {
    if block_size == 0 || prompt_len <= 1 {
        return None;
    }
    let split_pos = ((prompt_len - 1) / block_size) * block_size;
    (split_pos > cached_tokens && split_pos < prompt_len).then_some(split_pos)
}

fn decode_buffer_max_batch(
    backend: &dyn BackendRuntime,
    scheduler_max_decode_batch: Option<usize>,
) -> usize {
    if let Some(required) = scheduler_max_decode_batch {
        return required.max(1);
    }
    // Scale the per-step decode buffer to the backend's widest scheduler
    // policy so the first large batch does not immediately error with `decode
    // batch N exceeds buffer max_batch M`. The owning server injects its exact
    // validated ceiling above; standalone consumers receive the backend-owned
    // safe default unless they construct the runner with an explicit option.
    let policy = BackendCapabilityQueries::backend_capabilities(backend).decode_batcher;
    let backend_default = policy.engine_max_decode_batch.unwrap_or(policy.max_batch);
    backend_default.max(1)
}

enum PrefillSampleSource {
    // (#1082) kt-native logits — forward + sampler are both kt; no candle bridge.
    Logits(kiln_tensor::Tensor),
    GreedyToken(TokenId),
}

impl PrefillSampleSource {
    fn cached_next_token(&self) -> PagedPrefixNextToken {
        match self {
            Self::Logits(logits) => PagedPrefixNextToken::Logits(logits.clone()),
            Self::GreedyToken(token) => PagedPrefixNextToken::GreedyToken(*token),
        }
    }
}

/// Result of streaming paged generation plus prefix-cache ownership metadata.
pub struct PrefixCachedStreamingOutput {
    pub receiver: mpsc::Receiver<StreamEvent>,
    pub registration: Option<PagedPrefixRegistration>,
    pub extra_registrations: Vec<PagedPrefixRegistration>,
    pub allocated_blocks: Vec<u32>,
}

/// Event stream plus an explicit acknowledgement that its worker has either
/// released or intentionally retained every GPU-owned lifetime resource.
#[must_use = "the event stream must be consumed and worker settlement observed"]
pub struct ThreadedStreamingOutput {
    /// Tokens and the model-level terminal event produced by the worker.
    pub receiver: mpsc::Receiver<StreamEvent>,
    /// Becomes readable only after decode cleanup and lifetime settlement.
    /// A disconnected channel means the worker exited without proving that
    /// boundary and must be treated as a failed settlement.
    pub settled: mpsc::Receiver<()>,
}

/// Values whose destruction would make a threaded request's GPU state appear
/// reusable. Prefill only borrows this bundle so a panic fence can retain the
/// entire ownership chain when backend completion is unknown.
struct ThreadedPrefillOwnership<L, F> {
    worker_lifetime: L,
    post_decode: F,
    block_table: BlockTable,
    allocated_blocks: Vec<u32>,
    linear_state: Option<LinearAttentionState>,
}

/// Run prefill without allowing a panic to release request-owned GPU state.
///
/// An ordinary `Err` returns the ownership bundle to its caller for normal
/// cleanup. A panic means backend completion cannot be established: quarantine
/// is latched before the error is returned and every supplied owner is leaked.
fn run_threaded_prefill_with_panic_fence<T, O, F>(
    backend_health: &BackendHealthHandle,
    boundary: &'static str,
    mut ownership: O,
    prefill: F,
) -> Result<(Result<T>, O)>
where
    F: FnOnce(&mut O) -> Result<T>,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prefill(&mut ownership))) {
        Ok(result) => Ok((result, ownership)),
        Err(_) => {
            let reason = format!(
                "{boundary} panicked; backend completion and request ownership are unknown"
            );
            backend_health.quarantine(reason.clone());
            std::mem::forget(ownership);
            Err(anyhow::anyhow!(reason))
        }
    }
}

/// Prefix-cache metadata transferred to the sole post-decode cleanup owner.
/// The threaded worker invokes its finalizer only after all model/GPU work has
/// quiesced and before publishing the terminal stream event.
pub struct PrefixCachedStreamingCleanup {
    pub registration: Option<PagedPrefixRegistration>,
    pub extra_registrations: Vec<PagedPrefixRegistration>,
    pub allocated_blocks: Vec<u32>,
}

enum PrefixStreamDecodeOutcome {
    Settled(Result<Option<StreamDone>>),
    Quarantined(String),
}

fn run_prefix_cached_stream_worker<D, F>(
    tx: mpsc::Sender<StreamEvent>,
    decode: D,
    post_decode: F,
    cleanup: PrefixCachedStreamingCleanup,
    backend_health: &BackendHealthHandle,
) -> bool
where
    D: FnMut(&mpsc::Sender<StreamEvent>) -> PrefixStreamDecodeOutcome,
    F: FnOnce(PrefixCachedStreamingCleanup) -> Result<()>,
{
    let mut decode = decode;
    let mut post_decode = Some(post_decode);
    let mut cleanup = Some(cleanup);
    let decode_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode(&tx)));

    let quarantine_reason = match &decode_result {
        Ok(PrefixStreamDecodeOutcome::Quarantined(reason)) => Some(reason.clone()),
        Err(_) => Some("prefix streaming decode panicked".to_string()),
        Ok(PrefixStreamDecodeOutcome::Settled(_)) => None,
    };
    let terminal = match decode_result {
        Ok(PrefixStreamDecodeOutcome::Settled(result)) => {
            let finalize = post_decode.take().expect("post-decode finalizer present");
            let cleanup = cleanup.take().expect("post-decode cleanup present");
            let finalized =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finalize(cleanup)));
            match finalized {
                Ok(Ok(())) => result,
                Ok(Err(err)) => Err(err.context("prefix streaming cleanup failed")),
                Err(_) => {
                    tracing::error!(
                        event = "prefix_stream_cleanup_panicked",
                        "prefix streaming cleanup panicked; no fallback free was attempted"
                    );
                    Err(anyhow::anyhow!("prefix streaming cleanup panicked"))
                }
            }
        }
        Ok(PrefixStreamDecodeOutcome::Quarantined(reason)) => {
            // GPU completion is unknown. Leak the request owner and allocation
            // metadata rather than making any physical pages reusable.
            std::mem::forget(post_decode.take().expect("post-decode finalizer present"));
            std::mem::forget(cleanup.take().expect("post-decode cleanup present"));
            std::mem::forget(decode);
            tracing::error!(
                event = "prefix_stream_decode_quarantined",
                reason,
                "prefix streaming decode completion is unknown; cache lease and blocks quarantined"
            );
            Err(anyhow::anyhow!(reason))
        }
        Err(_) => {
            std::mem::forget(post_decode.take().expect("post-decode finalizer present"));
            std::mem::forget(cleanup.take().expect("post-decode cleanup present"));
            std::mem::forget(decode);
            tracing::error!(
                event = "prefix_stream_decode_panicked",
                "prefix streaming decode panicked; cache lease and blocks quarantined"
            );
            Err(anyhow::anyhow!("prefix streaming decode panicked"))
        }
    };

    let quarantined = if let Some(reason) = quarantine_reason {
        backend_health.quarantine(reason);
        true
    } else {
        false
    };

    match terminal {
        Ok(Some(done)) => {
            let _ = tx.send(StreamEvent::Done(done));
        }
        Ok(None) => {}
        Err(err) => {
            tracing::error!(error = %err, "spawn_streaming_paged_shared_tokens_with_prefix_cache decode thread failed");
            let _ = tx.send(StreamEvent::Error(err.to_string()));
        }
    }

    quarantined
}

/// Output from a native MTP speculative generation call.
///
/// Carries everything [`GenerationOutput`] does plus the per-call MTP draft
/// accept/reject counters used by bench reporting to compute α (acceptance
/// rate = `draft_accepted_count / total_draft_attempts`).
#[derive(Debug)]
pub struct MtpGenerationOutput {
    /// The generated text (not including the prompt).
    pub text: String,
    /// The generated token IDs (not including prompt tokens).
    pub token_ids: Vec<TokenId>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// How many MTP draft tokens were accepted across the decode loop.
    pub draft_accepted_count: usize,
    /// How many MTP draft attempts were made (one per [`speculative_mtp_decode_step`] call).
    pub total_draft_attempts: usize,
}

/// A single token emitted during streaming generation.
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// The generated token ID.
    pub token_id: TokenId,
    /// The decoded text for this token.
    pub text: String,
    /// Monotonic time immediately before the model producer publishes this
    /// accepted token to its stream channel.
    pub ready_at: Instant,
}

/// Final event sent when streaming generation completes.
#[derive(Debug, Clone)]
pub struct StreamDone {
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Total number of generated tokens.
    pub completion_tokens: usize,
    /// Text held back by the emit gates (UTF-8 char-boundary + stop-window
    /// holdback) that became safe to emit only at end-of-stream. Empty
    /// after a stop match (the held text WAS the stop) and on error paths.
    pub trailing_text: String,
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A new token was generated.
    Token(StreamToken),
    /// Generation is complete.
    Done(StreamDone),
    /// Generation failed and no successful completion may be cached.
    Error(String),
}

enum StreamTokenDisposition {
    Continue,
    Finished(FinishReason),
    ReceiverDropped,
}

/// Configuration for the live greedy decode rendezvous worker.
///
/// This is a pure execution value: the owning product resolves operator intent,
/// backend defaults, and scheduler ceilings before constructing it. Metal's
/// backend policy uses a small admission delay to collect compatible peers;
/// CUDA drains immediately and defaults to one row per worker pass because the
/// current coalesced CUDA GDN decode path is slower than rowwise scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBatcherConfig {
    /// Maximum compatible rows to execute in one decode forward pass.
    pub max_batch: usize,
    /// Optional admission delay for collecting peers.
    pub wait: std::time::Duration,
    /// Whether one batch may contain rows at different decode positions.
    pub allow_mixed_seq_lens: bool,
}

fn env_flag_value(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "false" | "off" | "no" => Some(false),
        "1" | "true" | "on" | "yes" => Some(true),
        _ => None,
    }
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    env_flag_value(name).unwrap_or(default)
}

fn decode_batcher_rowwise_retry_enabled(backend: &dyn BackendRuntime) -> bool {
    let policy = BackendCapabilityQueries::backend_capabilities(backend).decode_batcher;
    if let Some(env_var) = policy.rowwise_retry_env
        && env_flag_enabled(env_var, false)
    {
        return true;
    }
    decode_hot_path_fallback_policy_for_backend(backend).allows_fallback()
}

fn greedy_token_decode_enabled(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .decode_batcher
        .use_greedy_token_decode
}

fn prefix_cache_split_snapshot_allowed(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .decode_batcher
        .allow_prefix_cache_split_snapshot
}

fn native_support_enabled(support: Support) -> bool {
    matches!(support, Support::Native | Support::NativeWithConstraints)
}

fn paged_decode_graph_replay_request(config: &ModelConfig, max_batch: usize) -> ReplayRequest {
    ReplayRequest::paged_decode_graph_outputs(
        config.hidden_size,
        config.intermediate_size,
        max_batch.max(1),
    )
    .with_dtype(paged_cache_kt_dtype(config.dtype))
}

fn paged_decode_replay_primitive_enabled(
    backend: &dyn BackendRuntime,
    config: &ModelConfig,
    max_batch: usize,
    primitive: ReplayNativePrimitive,
) -> bool {
    let req = paged_decode_graph_replay_request(config, max_batch);
    let support = ReplayBackend::runtime_supports_replay_request(backend, &req);
    let authority = ReplayBackend::runtime_replay_authority(backend);
    native_support_enabled(support) && authority.native_primitive == primitive
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GreedyBatchRoute {
    HipGraph,
    Contiguous,
    Later,
}

fn greedy_batch_route(
    all_greedy: bool,
    cache_is_fp8: bool,
    row_count: usize,
    hip_graph_ready: bool,
) -> GreedyBatchRoute {
    if all_greedy && row_count == 1 && hip_graph_ready {
        GreedyBatchRoute::HipGraph
    } else if all_greedy && !cache_is_fp8 {
        GreedyBatchRoute::Contiguous
    } else {
        GreedyBatchRoute::Later
    }
}

fn should_use_unidentified_single_row_greedy_route(
    batch: usize,
    stable_row_ids_present: bool,
    resident_decode_supported: bool,
) -> bool {
    batch == 1 && !(stable_row_ids_present && resident_decode_supported)
}

fn decode_hot_path_fallback_disabled_context(
    backend: &dyn BackendRuntime,
    operation: &'static str,
) -> String {
    format!(
        "{operation}; fallback policy {:?} for {} decode hot path \
         (set KILN_DECODE_HOT_PATH_DEBUG_FALLBACK=1 to opt in)",
        decode_hot_path_fallback_policy_for_backend(backend),
        BackendIdentity::runtime_name(backend)
    )
}

/// Shared live decode rendezvous for greedy streaming requests.
///
/// Requests keep ownership of stop handling, output routing, block lifetime,
/// and one-row GDN state. At each eligible decode step they temporarily hand a
/// single-token job to this worker; the worker groups same-position jobs and
/// calls `ModelRunner::decode_next_tokens_paged_contiguous_batch_greedy`.
pub struct DecodeBatcher {
    sender: Mutex<Option<mpsc::Sender<DecodeBatchJob>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    counters: Arc<DecodeBatcherCounters>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeBatcherStats {
    pub submitted_jobs: usize,
    pub executed_batches: usize,
    pub executed_rows: usize,
    pub runner_calls: usize,
    pub max_runner_calls_per_token: usize,
    pub max_observed_batch: usize,
    pub runner_busy_jobs: usize,
    pub failed_jobs: usize,
}

impl DecodeBatcherStats {
    /// Phase 8 sentinel budget: a live greedy decode row should normally cost
    /// one runner call, with one extra call allowed for the explicit rowwise
    /// retry path after a failed batched attempt.
    pub const MAX_RUNNER_CALLS_PER_TOKEN_BUDGET: usize = 2;

    pub fn runner_calls_per_token(&self) -> Option<f64> {
        if self.executed_rows == 0 {
            None
        } else {
            Some(self.runner_calls as f64 / self.executed_rows as f64)
        }
    }

    pub const fn runner_call_budget_per_token(&self) -> usize {
        Self::MAX_RUNNER_CALLS_PER_TOKEN_BUDGET
    }

    pub const fn runner_call_budget_exceeded(&self) -> bool {
        self.max_runner_calls_per_token > Self::MAX_RUNNER_CALLS_PER_TOKEN_BUDGET
    }
}

struct DecodeBatcherCounters {
    submitted_jobs: AtomicUsize,
    executed_batches: AtomicUsize,
    executed_rows: AtomicUsize,
    runner_calls: AtomicUsize,
    max_runner_calls_per_token: AtomicUsize,
    max_observed_batch: AtomicUsize,
    runner_busy_jobs: AtomicUsize,
    failed_jobs: AtomicUsize,
}

struct DecodeBatchJob {
    input_token: TokenId,
    seq_len: usize,
    block_table: BlockTable,
    linear_state: LinearAttentionState,
    skip_gdn_state_readback: bool,
    response: mpsc::Sender<DecodeBatchReply>,
}

enum DecodeBatchReply {
    Decoded {
        token: TokenId,
        linear_state: LinearAttentionState,
    },
    RunnerBusy {
        linear_state: LinearAttentionState,
    },
    Failed {
        error: String,
        linear_state: LinearAttentionState,
    },
}

enum DecodeBatcherDecode {
    Decoded(TokenId),
    RunnerBusy,
}

impl DecodeBatcher {
    pub fn spawn(
        runner_lock: Arc<std::sync::RwLock<ModelRunner>>,
        paged_cache: Arc<PagedKvCache>,
        config: DecodeBatcherConfig,
    ) -> Result<Arc<Self>> {
        let (sender, receiver) = mpsc::channel();
        let backend = runner_lock
            .read()
            .map_err(|err| anyhow::anyhow!("failed to acquire model runner for batcher: {err}"))?
            .backend
            .clone();
        let counters = Arc::new(DecodeBatcherCounters {
            submitted_jobs: AtomicUsize::new(0),
            executed_batches: AtomicUsize::new(0),
            executed_rows: AtomicUsize::new(0),
            runner_calls: AtomicUsize::new(0),
            max_runner_calls_per_token: AtomicUsize::new(0),
            max_observed_batch: AtomicUsize::new(0),
            runner_busy_jobs: AtomicUsize::new(0),
            failed_jobs: AtomicUsize::new(0),
        });
        let counters_for_worker = counters.clone();
        let worker = std::thread::Builder::new()
            .name("kiln-decode-batcher".to_string())
            .spawn(move || {
                run_decode_batcher_worker(
                    runner_lock,
                    paged_cache,
                    backend,
                    receiver,
                    config,
                    counters_for_worker,
                );
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn decode batcher worker: {e}"))?;

        Ok(Arc::new(Self {
            sender: Mutex::new(Some(sender)),
            worker: Mutex::new(Some(worker)),
            counters,
        }))
    }

    pub fn max_observed_batch(&self) -> usize {
        self.counters.max_observed_batch.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> DecodeBatcherStats {
        DecodeBatcherStats {
            submitted_jobs: self.counters.submitted_jobs.load(Ordering::Relaxed),
            executed_batches: self.counters.executed_batches.load(Ordering::Relaxed),
            executed_rows: self.counters.executed_rows.load(Ordering::Relaxed),
            runner_calls: self.counters.runner_calls.load(Ordering::Relaxed),
            max_runner_calls_per_token: self
                .counters
                .max_runner_calls_per_token
                .load(Ordering::Relaxed),
            max_observed_batch: self.counters.max_observed_batch.load(Ordering::Relaxed),
            runner_busy_jobs: self.counters.runner_busy_jobs.load(Ordering::Relaxed),
            failed_jobs: self.counters.failed_jobs.load(Ordering::Relaxed),
        }
    }

    /// Close the rendezvous queue and join its worker before accelerator
    /// runtime teardown. The worker owns a model-runner reference, so leaving
    /// it detached can race graph-buffer destruction with HIP/CUDA finalizers.
    pub fn shutdown(&self) -> Result<()> {
        self.sender
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let worker = self
            .worker
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let Some(worker) = worker else {
            return Ok(());
        };
        anyhow::ensure!(
            worker.thread().id() != std::thread::current().id(),
            "decode batcher cannot join itself"
        );
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("decode batcher worker panicked during shutdown"))?;
        Ok(())
    }

    fn decode_next_token_greedy(
        &self,
        input_token: TokenId,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        skip_gdn_state_readback: bool,
    ) -> Result<DecodeBatcherDecode> {
        let (response_tx, response_rx) = mpsc::channel();
        let owned_state = take_linear_attention_state(linear_state);
        let job = DecodeBatchJob {
            input_token,
            seq_len,
            block_table: block_table.clone(),
            linear_state: owned_state,
            skip_gdn_state_readback,
            response: response_tx,
        };
        let sender_guard = self
            .sender
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(sender) = sender_guard.as_ref() else {
            *linear_state = job.linear_state;
            anyhow::bail!("decode batcher worker is shutting down");
        };
        if let Err(err) = sender.send(job) {
            *linear_state = err.0.linear_state;
            anyhow::bail!("decode batcher worker is not running");
        }
        drop(sender_guard);
        self.counters.submitted_jobs.fetch_add(1, Ordering::Relaxed);

        match response_rx.recv() {
            Ok(DecodeBatchReply::Decoded {
                token,
                linear_state: returned_state,
            }) => {
                *linear_state = returned_state;
                Ok(DecodeBatcherDecode::Decoded(token))
            }
            Ok(DecodeBatchReply::RunnerBusy {
                linear_state: returned_state,
            }) => {
                *linear_state = returned_state;
                Ok(DecodeBatcherDecode::RunnerBusy)
            }
            Ok(DecodeBatchReply::Failed {
                error,
                linear_state: returned_state,
            }) => {
                *linear_state = returned_state;
                anyhow::bail!("{error}");
            }
            Err(err) => anyhow::bail!("decode batcher worker disconnected before reply: {err}"),
        }
    }
}

impl Drop for DecodeBatcher {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            tracing::error!(
                event = "decode_batcher_shutdown_failed",
                error = %error,
                "decode batcher failed to join during drop"
            );
        }
    }
}

fn take_linear_attention_state(state: &mut LinearAttentionState) -> LinearAttentionState {
    std::mem::replace(
        state,
        LinearAttentionState {
            recurrent_states: Vec::new(),
            conv_states: Vec::new(),
        },
    )
}

fn materialize_decode_job_resident_states(
    backend: &dyn BackendRuntime,
    jobs: &mut [DecodeBatchJob],
) -> Result<()> {
    for job in jobs {
        job.linear_state
            .materialize_gdn_recurrent_resident_states(backend)?;
    }
    Ok(())
}

fn run_decode_batcher_worker(
    runner_lock: Arc<std::sync::RwLock<ModelRunner>>,
    paged_cache: Arc<PagedKvCache>,
    backend: Arc<dyn BackendRuntime>,
    receiver: mpsc::Receiver<DecodeBatchJob>,
    config: DecodeBatcherConfig,
    counters: Arc<DecodeBatcherCounters>,
) {
    let max_batch = config.max_batch.max(1);
    let allow_mixed_seq_lens = config.allow_mixed_seq_lens;
    let mut deferred = VecDeque::new();
    let mut disconnected = false;

    while !disconnected || !deferred.is_empty() {
        let Some(first) = deferred.pop_front().or_else(|| receiver.recv().ok()) else {
            break;
        };
        let seq_len = first.seq_len;
        let mut jobs = vec![first];

        while jobs.len() < max_batch {
            match receiver.try_recv() {
                Ok(job) if allow_mixed_seq_lens || job.seq_len == seq_len => jobs.push(job),
                Ok(job) => deferred.push_back(job),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if config.wait > std::time::Duration::ZERO && jobs.len() < max_batch && !disconnected {
            let deadline = std::time::Instant::now() + config.wait;
            while jobs.len() < max_batch {
                let now = std::time::Instant::now();
                if now >= deadline {
                    break;
                }
                match receiver.recv_timeout(deadline.saturating_duration_since(now)) {
                    Ok(job) if allow_mixed_seq_lens || job.seq_len == seq_len => jobs.push(job),
                    Ok(job) => deferred.push_back(job),
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        counters
            .max_observed_batch
            .fetch_max(jobs.len(), Ordering::Relaxed);
        counters.executed_batches.fetch_add(1, Ordering::Relaxed);
        counters
            .executed_rows
            .fetch_add(jobs.len(), Ordering::Relaxed);
        process_decode_batch_jobs(
            &runner_lock,
            paged_cache.as_ref(),
            &*backend,
            jobs,
            &counters,
        );
    }
}

fn process_decode_batch_jobs(
    runner_lock: &std::sync::RwLock<ModelRunner>,
    paged_cache: &PagedKvCache,
    fallback_backend: &dyn BackendRuntime,
    mut jobs: Vec<DecodeBatchJob>,
    counters: &DecodeBatcherCounters,
) {
    let runner_guard = match runner_lock.try_read() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::WouldBlock) => {
            counters
                .runner_busy_jobs
                .fetch_add(jobs.len(), Ordering::Relaxed);
            if let Err(err) = materialize_decode_job_resident_states(fallback_backend, &mut jobs) {
                let message = format!(
                    "failed to materialize resident GDN state before runner-busy fallback: {err:#}"
                );
                counters
                    .failed_jobs
                    .fetch_add(jobs.len(), Ordering::Relaxed);
                for job in jobs {
                    let _ = job.response.send(DecodeBatchReply::Failed {
                        error: message.clone(),
                        linear_state: job.linear_state,
                    });
                }
                return;
            }
            for job in jobs {
                let _ = job.response.send(DecodeBatchReply::RunnerBusy {
                    linear_state: job.linear_state,
                });
            }
            return;
        }
        Err(std::sync::TryLockError::Poisoned(err)) => {
            let mut message =
                format!("failed to acquire runner read lock in decode batcher: {err}");
            if let Err(materialize_err) =
                materialize_decode_job_resident_states(fallback_backend, &mut jobs)
            {
                tracing::warn!(
                    error = %materialize_err,
                    "failed to materialize resident GDN state after poisoned runner lock"
                );
                message = format!(
                    "{message}; also failed to materialize resident GDN state: {materialize_err:#}"
                );
            }
            counters
                .failed_jobs
                .fetch_add(jobs.len(), Ordering::Relaxed);
            for job in jobs {
                let _ = job.response.send(DecodeBatchReply::Failed {
                    error: message.clone(),
                    linear_state: job.linear_state,
                });
            }
            return;
        }
    };

    let backend = &*runner_guard.backend;
    let job_count = jobs.len();
    let mut runner_calls_for_jobs = 1usize;
    let rowwise_retry_enabled = decode_batcher_rowwise_retry_enabled(backend);
    let tokens =
        match decode_batch_jobs_with_runner(&runner_guard, paged_cache, &mut jobs, counters) {
            Ok(tokens) => Ok(tokens),
            Err(err) if jobs.len() > 1 && rowwise_retry_enabled => {
                tracing::debug!(
                    batch = jobs.len(),
                    error = %err,
                    "batched greedy decode failed; falling back to rowwise decode jobs"
                );
                let mut tokens = Vec::with_capacity(jobs.len());
                let mut fallback_error = None;
                for idx in 0..jobs.len() {
                    runner_calls_for_jobs += 1;
                    match decode_batch_jobs_with_runner(
                        &runner_guard,
                        paged_cache,
                        &mut jobs[idx..idx + 1],
                        counters,
                    ) {
                        Ok(mut row_tokens) => tokens.push(row_tokens.remove(0)),
                        Err(row_err) => {
                            fallback_error = Some(row_err);
                            break;
                        }
                    }
                }
                match fallback_error {
                    Some(err) => Err(err),
                    None => Ok(tokens),
                }
            }
            Err(err) if jobs.len() > 1 => {
                tracing::debug!(
                    batch = jobs.len(),
                    error = %err,
                    "batched greedy decode failed; rowwise retry disabled"
                );
                Err(err)
            }
            Err(err) => Err(err),
        };
    counters.max_runner_calls_per_token.fetch_max(
        if job_count > 0 && runner_calls_for_jobs > 1 {
            2
        } else {
            usize::from(job_count > 0)
        },
        Ordering::Relaxed,
    );

    match tokens {
        Ok(tokens) => {
            for (job, token) in jobs.into_iter().zip(tokens.into_iter()) {
                if job.skip_gdn_state_readback {
                    job.linear_state
                        .evict_gdn_recurrent_resident_states(backend);
                }
                let _ = job.response.send(DecodeBatchReply::Decoded {
                    token,
                    linear_state: job.linear_state,
                });
            }
        }
        Err(err) => {
            let message = format!("{err:#}");
            if let Err(materialize_err) = materialize_decode_job_resident_states(backend, &mut jobs)
            {
                tracing::warn!(
                    error = %materialize_err,
                    "failed to materialize resident GDN state after decode batch error"
                );
            }
            counters
                .failed_jobs
                .fetch_add(jobs.len(), Ordering::Relaxed);
            for job in jobs {
                let _ = job.response.send(DecodeBatchReply::Failed {
                    error: message.clone(),
                    linear_state: job.linear_state,
                });
            }
        }
    }
}

fn decode_batch_jobs_with_runner(
    runner: &ModelRunner,
    paged_cache: &PagedKvCache,
    jobs: &mut [DecodeBatchJob],
    counters: &DecodeBatcherCounters,
) -> Result<Vec<TokenId>> {
    counters.runner_calls.fetch_add(1, Ordering::Relaxed);
    let profile_stages = profile_decode_batcher_stages_enabled();
    let total_start = profile_stages.then(std::time::Instant::now);
    let stage_start = profile_stages.then(std::time::Instant::now);
    let input_tokens: Vec<TokenId> = jobs.iter().map(|job| job.input_token).collect();
    let seq_lens: Vec<usize> = jobs.iter().map(|job| job.seq_len).collect();
    let block_tables: Vec<BlockTable> = jobs.iter().map(|job| job.block_table.clone()).collect();
    let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
    let skip_gdn_state_readback = skip_final_gdn_state_readback_enabled()
        && jobs.iter().all(|job| job.skip_gdn_state_readback);
    finish_decode_batcher_stage_profile("job_metadata", jobs.len(), stage_start);

    let stage_start = profile_stages.then(std::time::Instant::now);
    let _skip_scope = crate::forward::VulkanSkipGdnStateReadbackScope::new(skip_gdn_state_readback);
    let tokens = if runner.has_linear_attention_layers() {
        let mut linear_states: Vec<&mut LinearAttentionState> =
            jobs.iter_mut().map(|job| &mut job.linear_state).collect();
        runner.decode_next_tokens_paged_contiguous_batch_greedy(
            &input_tokens,
            paged_cache,
            &block_table_refs,
            &seq_lens,
            &mut linear_states,
        )
    } else {
        let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
        runner.decode_next_tokens_paged_contiguous_batch_greedy(
            &input_tokens,
            paged_cache,
            &block_table_refs,
            &seq_lens,
            &mut no_linear_states,
        )
    };
    finish_decode_batcher_stage_profile("runner_call", jobs.len(), stage_start);
    finish_decode_batcher_stage_profile("worker_total", jobs.len(), total_start);
    tokens
}

struct SharedBlockReservation<'a> {
    block_manager: &'a Mutex<BlockManager>,
    block_ids: Vec<u32>,
}

impl SharedBlockReservation<'_> {
    fn release_after_settlement<T>(
        self,
        runner: &ModelRunner,
        boundary: &'static str,
        result: Result<T>,
    ) -> Result<T> {
        self.release_after_settlement_with(boundary, result, || {
            catch_external_yield_sync_panic(&runner.backend_health, boundary, || {
                runner.synchronize_external_yield(boundary)
            })
        })
    }

    fn release_after_settlement_with<T>(
        mut self,
        boundary: &'static str,
        result: Result<T>,
        synchronize: impl FnOnce() -> Result<()>,
    ) -> Result<T> {
        match synchronize() {
            Ok(()) => {
                let block_ids = std::mem::take(&mut self.block_ids);
                if !block_ids.is_empty() {
                    match self.block_manager.lock() {
                        Ok(mut guard) => guard.free_all(&block_ids),
                        Err(error) => tracing::error!(
                            %error,
                            boundary,
                            "failed to lock block manager after settled shared reservation"
                        ),
                    }
                }
                result
            }
            Err(sync_error) => {
                let prior_error = result.as_ref().err().map(|error| format!("{error:#}"));
                std::mem::forget(result);
                match prior_error {
                    Some(prior_error) => Err(sync_error.context(format!(
                        "generation also failed before shared KV release: {prior_error}"
                    ))),
                    None => Err(sync_error),
                }
            }
        }
    }
}

impl Drop for SharedBlockReservation<'_> {
    fn drop(&mut self) {
        if self.block_ids.is_empty() {
            return;
        }
        tracing::error!(
            blocks = self.block_ids.len(),
            "unsettled shared KV reservation dropped; retaining its pages"
        );
    }
}

struct SettlementOutcome<T, O> {
    result: Result<T>,
    owners: O,
}

fn catch_external_yield_sync_panic(
    backend_health: &BackendHealthHandle,
    boundary: &'static str,
    synchronize: impl FnOnce() -> Result<()>,
) -> Result<()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(synchronize)) {
        Ok(result) => result,
        Err(_) => {
            let reason = format!("backend synchronization panicked at {boundary}");
            backend_health.quarantine(reason.clone());
            Err(anyhow::anyhow!(reason))
        }
    }
}

/// Exclusive reservation used by the legacy synchronous paged-stream API.
///
/// Unlike the older ad hoc cleanup branches, this guard retains its pages by
/// default. Only `release_after_settlement*` can return them to the mutable
/// block manager, after backend completion has been proven and the associated
/// GPU owners can be dropped safely.
struct MutableBlockReservation<'a> {
    block_manager: &'a mut BlockManager,
    block_ids: Vec<u32>,
}

impl MutableBlockReservation<'_> {
    fn block_table(&self) -> BlockTable {
        let mut block_table = BlockTable::new();
        for &block_id in &self.block_ids {
            block_table.push(block_id);
        }
        block_table
    }

    fn release_after_settlement<T>(
        self,
        runner: &ModelRunner,
        boundary: &'static str,
        outcome: SettlementOutcome<T, LegacyMutablePagedStreamOwners<'_>>,
    ) -> Result<T> {
        self.release_after_settlement_with(
            boundary,
            outcome,
            || {
                catch_external_yield_sync_panic(&runner.backend_health, boundary, || {
                    runner.synchronize_external_yield(boundary)
                })
            },
            LegacyMutablePagedStreamOwners::quarantine,
        )
    }

    fn release_after_settlement_with<T, O>(
        mut self,
        boundary: &'static str,
        mut outcome: SettlementOutcome<T, O>,
        synchronize: impl FnOnce() -> Result<()>,
        quarantine_owners: impl FnOnce(&mut O),
    ) -> Result<T> {
        match synchronize() {
            Ok(()) => {
                let SettlementOutcome { result, owners } = outcome;
                drop(owners);
                let block_ids = std::mem::take(&mut self.block_ids);
                self.block_manager.free_all(&block_ids);
                result
            }
            Err(sync_error) => {
                let prior_error = outcome
                    .result
                    .as_ref()
                    .err()
                    .map(|error| format!("{error:#}"));
                quarantine_owners(&mut outcome.owners);
                std::mem::forget(outcome);
                tracing::error!(
                    blocks = self.block_ids.len(),
                    boundary,
                    "mutable KV reservation settlement failed; retaining pages and GPU owners"
                );
                std::mem::forget(self);
                match prior_error {
                    Some(prior_error) => Err(sync_error.context(format!(
                        "generation also failed before mutable KV release: {prior_error}"
                    ))),
                    None => Err(sync_error),
                }
            }
        }
    }
}

impl Drop for MutableBlockReservation<'_> {
    fn drop(&mut self) {
        if self.block_ids.is_empty() {
            return;
        }
        tracing::error!(
            blocks = self.block_ids.len(),
            "unsettled mutable KV reservation dropped; retaining its pages"
        );
    }
}

struct LegacyMutablePagedStreamOwners<'a> {
    linear_state: Option<LinearAttentionState>,
    prefill_logits: Option<kiln_tensor::Tensor>,
    pending_decode_logits: Option<kiln_tensor::Tensor>,
    cuda_graph: Option<std::sync::MutexGuard<'a, CudaGraphRunner>>,
}

impl LegacyMutablePagedStreamOwners<'_> {
    fn new() -> Self {
        Self {
            linear_state: None,
            prefill_logits: None,
            pending_decode_logits: None,
            cuda_graph: None,
        }
    }

    fn quarantine(&mut self) {
        // The guard is coordination state, not GPU-owned request state. Keep it
        // through the synchronization attempt, then unlock even when device
        // completion remains unknown so later health checks fail promptly.
        drop(self.cuda_graph.take());
        if let Some(linear_state) = self.linear_state.as_mut() {
            quarantine_linear_attention_state(linear_state);
        }
    }
}

fn lock_block_manager(
    block_manager: &Mutex<BlockManager>,
) -> Result<std::sync::MutexGuard<'_, BlockManager>> {
    block_manager
        .lock()
        .map_err(|e| anyhow::anyhow!("failed to lock block manager: {e}"))
}

// `PagedKvCache` no longer hides behind a `Mutex` — its write methods take
// `&self` and rely on the underlying tensor storage's interior mutability,
// so callers can simply pass the `&PagedKvCache` straight through. This
// helper is kept as a pass-through identity to minimize call-site churn
// during the lock-removal sweep; it can be inlined later.
fn lock_paged_cache(paged_cache: &PagedKvCache) -> Result<&PagedKvCache> {
    Ok(paged_cache)
}

pub fn append_prefix_block_table(cached_blocks: &[u32], allocated_blocks: &[u32]) -> BlockTable {
    let mut block_table = BlockTable::new();
    for &block_id in cached_blocks {
        block_table.push(block_id);
    }
    for &block_id in allocated_blocks {
        block_table.push(block_id);
    }
    block_table
}

/// A reused KV prefix may be extended only when its final block is complete.
/// Otherwise decode would append into a partial block still owned by the
/// prefix cache and corrupt that shared entry for later requests.
fn paged_prefix_reuse_matches_prompt(
    prefix: &PagedPrefixReuse,
    prompt_len: usize,
    block_size: usize,
    params: &SamplingParams,
) -> bool {
    if block_size == 0
        || prefix.cached_tokens == 0
        || prefix.cached_tokens > prompt_len
        || prefix.cached_tokens % block_size != 0
        || prefix.block_ids.len() != prefix.cached_tokens / block_size
    {
        return false;
    }

    if prefix.cached_tokens < prompt_len {
        return true;
    }

    prefix.next_token.as_ref().is_some_and(|next| match next {
        PagedPrefixNextToken::Logits(_) => true,
        PagedPrefixNextToken::GreedyToken(_) => params.is_effectively_greedy(),
    })
}

/// Legacy "lm_head → host sampler" batched path. Used when the backend
/// doesn't expose the fused on-device sampler, when the sampling request is
/// outside the backend kernel's supported envelope, or as the final fallback
/// for mixed shapes that cannot take a greedy or fused sampled path.
fn run_legacy_lm_head_sample_batch(
    backend: &dyn crate::backend::BackendRuntime,
    hidden: &kiln_tensor::Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    params: &[SamplingParams],
    states: &[&mut PagedBatchedDecodeState],
) -> Result<Vec<TokenId>> {
    // (#1082) lm head + sampler are both kt-native — `hidden` arrives kt from
    // the batched decode forward and the sampler takes kt logits directly. No
    // candle bridge.
    let logits = crate::forward::model_forward_head_backend_decode_if(
        Some(backend),
        hidden,
        weights,
        config,
    )
    .context("batched decode lm head")?;
    let mut sampled = Vec::with_capacity(states.len());
    for (idx, params) in params.iter().enumerate() {
        let row = logits
            .narrow(0, idx, 1)
            .with_context(|| format!("batched decode lm head row {idx}"))?;
        let token = if params.temperature == 0.0 {
            greedy_sample(&row)?
        } else {
            let mut row_params = params.clone();
            row_params.seed = states[idx].step_seed;
            sample_with_full_params(&row, &row_params, &states[idx].generated_tokens)?
        };
        sampled.push(token);
    }
    Ok(sampled)
}

fn unique_history_counts_for_batch_sample(history: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut counts: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for &token in history {
        *counts.entry(token).or_default() += 1;
    }
    let mut indices = Vec::with_capacity(counts.len());
    let mut values = Vec::with_capacity(counts.len());
    for (token, count) in counts {
        indices.push(token);
        values.push(count);
    }
    (indices, values)
}

fn sample_seed_for_batch_row(step_seed: Option<u64>, history: &[u32]) -> u64 {
    step_seed.unwrap_or_else(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let history_hash = history.iter().fold(0xCBF29CE484222325u64, |acc, &token| {
            (acc ^ token as u64).wrapping_mul(0x100000001B3)
        });
        nanos.wrapping_add(history_hash)
    })
}

fn run_lm_head_sample_batch_with_contexts(
    backend: &dyn crate::backend::BackendRuntime,
    hidden: &kiln_tensor::Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    params: &[SamplingParams],
    step_seeds: &[Option<u64>],
    generated_tokens: &[Vec<TokenId>],
) -> Result<Vec<TokenId>> {
    anyhow::ensure!(
        params.len() == step_seeds.len() && params.len() == generated_tokens.len(),
        "batched decode sampling context length mismatch"
    );
    let top_k_values: Vec<u32> = params.iter().map(|param| param.top_k).collect();
    let temperature_values: Vec<f32> = params.iter().map(|param| param.temperature).collect();
    if SamplingBackend::runtime_supports_linear_decode_sample_batch(
        backend,
        &top_k_values,
        &temperature_values,
    ) {
        let normed = crate::forward::model_forward_final_norm(hidden, weights, config)
            .context("batched decode final norm for fused sampling")?;
        let repetition_values: Vec<f32> = params
            .iter()
            .map(|param| param.repetition_penalty)
            .collect();
        let presence_values: Vec<f32> = params.iter().map(|param| param.presence_penalty).collect();
        let frequency_values: Vec<f32> =
            params.iter().map(|param| param.frequency_penalty).collect();
        let top_p_values: Vec<f32> = params.iter().map(|param| param.top_p).collect();
        let min_p_values: Vec<f32> = params.iter().map(|param| param.min_p).collect();
        let seed_values: Vec<u64> = step_seeds
            .iter()
            .zip(generated_tokens.iter())
            .map(|(&seed, history)| sample_seed_for_batch_row(seed, history))
            .collect();
        let mut history_rows = Vec::new();
        let mut history_indices = Vec::new();
        let mut history_counts = Vec::new();
        for (row_idx, (param, history)) in params.iter().zip(generated_tokens.iter()).enumerate() {
            if param.is_effectively_greedy()
                || param.token_penalties_are_no_op()
                || history.is_empty()
            {
                continue;
            }
            let (indices, counts) = unique_history_counts_for_batch_sample(history);
            for (idx, count) in indices.into_iter().zip(counts.into_iter()) {
                history_rows.push(row_idx as u32);
                history_indices.push(idx);
                history_counts.push(count);
            }
        }
        if let Some(tokens) = SamplingBackend::runtime_linear_decode_sample_batch(
            backend,
            &normed,
            &weights.embed_tokens_t,
            &history_rows,
            &history_indices,
            &history_counts,
            &repetition_values,
            &presence_values,
            &frequency_values,
            &temperature_values,
            &top_k_values,
            &top_p_values,
            &min_p_values,
            &seed_values,
        )
        .context("fused batched linear_decode_sample failed")?
        {
            return Ok(tokens);
        }
    }
    let logits = crate::forward::model_forward_head_backend_decode_if(
        Some(backend),
        hidden,
        weights,
        config,
    )
    .context("batched decode lm head")?;
    let mut sampled = Vec::with_capacity(params.len());
    for (idx, params) in params.iter().enumerate() {
        let row = logits
            .narrow(0, idx, 1)
            .with_context(|| format!("batched decode lm head row {idx}"))?;
        let token = if params.temperature == 0.0 {
            greedy_sample(&row)?
        } else {
            let mut row_params = params.clone();
            row_params.seed = step_seeds[idx];
            sample_with_full_params(&row, &row_params, &generated_tokens[idx])?
        };
        sampled.push(token);
    }
    Ok(sampled)
}

fn sample_first_decode_token(
    // (#1082) kt-native logits — sampler is kt now.
    logits: &kiln_tensor::Tensor,
    params: &SamplingParams,
) -> Result<TokenId> {
    if params.is_effectively_greedy() {
        Ok(greedy_sample(logits)?)
    } else {
        // First decode token has no generated history yet — penalties
        // become a no-op even when set, which is the correct OpenAI
        // semantics (penalties apply to *generated* tokens only).
        Ok(sample_with_full_params(logits, params, &[])?)
    }
}

fn sample_first_decode_token_with_logprob(
    logits: &kiln_tensor::Tensor,
    params: &SamplingParams,
) -> Result<SampledToken> {
    // First decode token has no generated history. The traced sampler still
    // resolves every configured filter and requires the request's effective
    // seed for stochastic behavior.
    sample_step_with_logprob(logits, params, params.seed, &[])
}

/// Composite per-request emit gate: incremental detokenization + stop
/// holdback. One per streaming generation; finish() drains residue at
/// non-stop exits (a stop can complete inside held bytes).
struct StreamTextGate {
    detok: crate::stream_text::IncrementalDetokenizer,
    stop: crate::stream_text::StopTailGate,
}

impl StreamTextGate {
    fn new(stop_sequences: &[String]) -> Self {
        Self {
            detok: crate::stream_text::IncrementalDetokenizer::new(),
            stop: crate::stream_text::StopTailGate::new(stop_sequences),
        }
    }

    /// Non-stop loop exit: push the detokenizer residual through the stop
    /// gate, then drain the stop holdback. Returns
    /// `(trailing_text, late_stop)` — when `late_stop` is `Some`, a stop
    /// completed inside the held bytes and the caller must override its
    /// finish reason.
    fn finish(
        &mut self,
        tokenizer: &KilnTokenizer,
        tokens: &[TokenId],
    ) -> Result<(String, Option<String>)> {
        let residual = self
            .detok
            .flush(tokenizer, tokens)
            .context("failed to flush streaming detokenizer")?;
        let scan = self.stop.push(&residual);
        if let Some(stop) = scan.matched_stop {
            return Ok((scan.emit, Some(stop)));
        }
        let mut trailing = scan.emit;
        trailing.push_str(&self.stop.flush());
        Ok((trailing, None))
    }
}

fn emit_stream_token(
    tx: &mpsc::Sender<StreamEvent>,
    tokenizer: &KilnTokenizer,
    gate: &mut StreamTextGate,
    generated_tokens: &mut Vec<TokenId>,
    token: TokenId,
) -> Result<StreamTokenDisposition> {
    generated_tokens.push(token);

    // CHECK BEFORE EMIT: the delta passes the stop gate first, so the
    // matched stop never reaches the wire (the pre-gate code emitted the
    // token THEN ran the stop check on the full decoded prefix — pi's
    // stop-marker parsers saw phantom delimiters in every stream).
    // Exactly one StreamEvent::Token per accepted token (text may be ""),
    // keeping completion-token counting and usage exact.
    let delta = match gate.detok.next_delta(tokenizer, generated_tokens) {
        Ok(delta) => delta,
        Err(error) => {
            generated_tokens.pop();
            return Err(error).context("failed to decode streaming token delta");
        }
    };
    let scan = gate.stop.push(&delta);
    let ready_at = Instant::now();
    if tx
        .send(StreamEvent::Token(StreamToken {
            token_id: token,
            text: scan.emit,
            ready_at,
        }))
        .is_err()
    {
        return Ok(StreamTokenDisposition::ReceiverDropped);
    }
    Ok(match scan.matched_stop {
        Some(stop) => StreamTokenDisposition::Finished(FinishReason::StopSequence(stop)),
        None => StreamTokenDisposition::Continue,
    })
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS token.
    Eos,
    /// Reached max_tokens limit.
    MaxTokens,
    /// Hit a stop sequence in the decoded text.
    StopSequence(String),
}

impl ModelRunner {
    pub fn is_eos_token(&self, token: TokenId) -> bool {
        self.eos_token_ids.contains(&token)
    }

    fn should_stop_on_eos(&self, params: &SamplingParams, token: TokenId) -> bool {
        !params.ignore_eos && self.eos_token_ids.contains(&token)
    }

    fn eos_token_ids_for<'a>(&'a self, params: &SamplingParams) -> &'a [TokenId] {
        if params.ignore_eos {
            &[]
        } else {
            &self.eos_token_ids
        }
    }

    pub fn stop_sequence_match(
        &self,
        generated_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<Option<String>> {
        if params.stop.is_empty() {
            return Ok(None);
        }
        let Some(text) = self
            .tokenizer
            .decode(generated_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .ok()
        else {
            return Ok(None);
        };
        Ok(params
            .stop
            .iter()
            .find(|stop_seq| text.contains(stop_seq.as_str()))
            .cloned())
    }

    /// Compatibility constructor for owners that initialize memory governance
    /// separately, such as `kiln-server`.
    ///
    /// Direct accelerator consumers should prefer
    /// [`Self::new_with_initialized_runtime`], which makes the startup contract
    /// explicit and fails before inference when the selected device, probe, or
    /// capacity policy is inconsistent. This constructor never probes.
    pub fn new(weights: GpuWeights, tokenizer: KilnTokenizer, config: ModelConfig) -> Self {
        Self::new_with_options(weights, tokenizer, config, false)
    }

    /// Create a new ModelRunner with explicit CUDA graph control.
    ///
    /// This compatibility constructor does not enable ROCm graphs as a side
    /// effect of its CUDA flag. ROCm owners must supply a typed policy through
    /// [`Self::new_with_runtime_options`].
    pub fn new_with_options(
        weights: GpuWeights,
        tokenizer: KilnTokenizer,
        config: ModelConfig,
        cuda_graphs: bool,
    ) -> Self {
        Self::new_with_runtime_options(
            weights,
            tokenizer,
            config,
            ModelRunnerRuntimeOptions {
                cuda_graphs,
                rocm_graph: RocmGraphExecutionPolicy::disabled(),
                metal_graphs: true,
                max_decode_batch: None,
                streaming_prefill: None,
            },
        )
    }

    /// Create a runner with backend graph eligibility resolved by the owning
    /// product surface. This compatibility constructor never probes or
    /// initializes memory governance; direct accelerator consumers should use
    /// [`Self::new_with_initialized_runtime`].
    pub fn new_with_runtime_options(
        weights: GpuWeights,
        tokenizer: KilnTokenizer,
        config: ModelConfig,
        options: ModelRunnerRuntimeOptions,
    ) -> Self {
        let execution_device = weights.embed_tokens.device();
        let selected_backend = backend::for_device_kt(&execution_device);
        Self::new_with_selected_backend(
            weights,
            tokenizer,
            config,
            options,
            execution_device,
            selected_backend,
        )
    }

    fn new_with_selected_backend(
        weights: GpuWeights,
        tokenizer: KilnTokenizer,
        config: ModelConfig,
        options: ModelRunnerRuntimeOptions,
        execution_device: kiln_tensor::Device,
        selected_backend: Arc<dyn BackendRuntime>,
    ) -> Self {
        let eos_token_ids = tokenizer.eos_token_ids();
        let cuda_graph = CudaGraphRunner::new(&execution_device, options.cuda_graphs);
        let rocm_graph = RocmGraphRunner::new(&execution_device, options.rocm_graph);
        let rocm_graph_telemetry = rocm_graph.telemetry_handle();
        let metal_graph = MetalGraphRunner::new(&execution_device, options.metal_graphs);
        let training_caps =
            TrainingLossBackend::runtime_training_capabilities(selected_backend.as_ref());
        let decode_buffer_max_batch =
            decode_buffer_max_batch(selected_backend.as_ref(), options.max_decode_batch);
        let streaming_prefill = options.streaming_prefill.unwrap_or_else(|| {
            StreamingPrefillExecutionPolicy::for_runtime(selected_backend.as_ref())
        });
        tracing::info!(
            backend = BackendIdentity::runtime_name(selected_backend.as_ref()),
            execution_device = %execution_device.short_name(),
            projection_training = training_caps.projection_training,
            flce_loss = training_caps.flce_loss,
            rmsnorm_training = training_caps.rmsnorm_training,
            resident_activation = training_caps.resident_activation,
            lora_delta_training = training_caps.lora_delta_training,
            sgd_step = training_caps.sgd_step,
            adamw_step = training_caps.adamw_step,
            native_training = training_caps.native_training,
            "Backend training capability profile"
        );
        // Phase A.5: registry + decode-buffer config are deferred to first hot-path
        // access. Building them eagerly here regressed c=1 paged decode by 22%
        // (Validation #4: 42.6 tok/s vs 54.76 baseline). The lazy `OnceLock` keeps
        // construction cheap and matches the production-path warmup contract.
        Self {
            weights,
            tokenizer,
            config,
            eos_token_ids,
            active_lora: None,
            cuda_graph: Mutex::new(cuda_graph),
            rocm_graph: Mutex::new(rocm_graph),
            rocm_graph_telemetry,
            metal_graph: Mutex::new(metal_graph),
            packed_weight_registry: OnceLock::new(),
            decode_buffers: OnceLock::new(),
            decode_buffer_max_batch,
            decode_buffer_config: OnceLock::new(),
            batched_state_cache: Mutex::new(None),
            batched_state_cache_counters: BatchedStateCacheCounters::default(),
            backend: selected_backend,
            streaming_prefill,
            backend_health: BackendHealthHandle::default(),
            memory_runtime: None,
        }
    }

    /// Build a direct-inference runner from an explicitly initialized memory
    /// runtime. This constructor performs no hardware probe; it verifies the
    /// typed binding against the weights and already-published global policy.
    pub fn new_with_initialized_runtime(
        weights: GpuWeights,
        tokenizer: KilnTokenizer,
        config: ModelConfig,
        options: ModelRunnerRuntimeOptions,
        memory_binding: &InferenceMemoryRuntime,
    ) -> Result<Self> {
        memory_binding.validate_weight_device(weights.embed_tokens.device())?;
        let selected_backend = backend::for_explicit_device_kt(memory_binding.device())?;
        let mut runner = Self::new_with_selected_backend(
            weights,
            tokenizer,
            config,
            options,
            memory_binding.device(),
            selected_backend,
        );
        runner.memory_runtime = Some(*memory_binding);
        Ok(runner)
    }

    /// Direct-inference memory binding, or `None` for compatibility owners
    /// that installed process memory governance outside `ModelRunner`.
    pub const fn inference_memory_runtime(&self) -> Option<InferenceMemoryRuntime> {
        self.memory_runtime
    }

    /// Startup-resolved policy governing streaming-prefill dispatch and tiles.
    pub const fn streaming_prefill_policy(&self) -> StreamingPrefillExecutionPolicy {
        self.streaming_prefill
    }

    pub fn backend_health_handle(&self) -> BackendHealthHandle {
        self.backend_health.clone()
    }

    pub fn backend_health_snapshot(&self) -> BackendHealthSnapshot {
        self.backend_health.snapshot()
    }

    pub fn ensure_backend_healthy(&self) -> Result<()> {
        self.backend_health.ensure_healthy()?;
        #[cfg(feature = "rocm")]
        if let kiln_tensor::Device::Rocm(device_index) = self.weights.embed_tokens.device() {
            if kiln_tensor::rocm_cleanup_quarantined(device_index)
                .context("query ROCm cleanup quarantine")?
            {
                let reason =
                    "ROCm synchronization recovery failed; execution and cleanup are quarantined"
                        .to_string();
                self.backend_health.quarantine(reason.clone());
                anyhow::bail!(reason);
            }
        }
        Ok(())
    }

    /// Prove that all backend work submitted so far has completed before
    /// publishing progress or recycling mutable device resources.
    pub fn synchronize_external_yield(&self, boundary: &'static str) -> Result<()> {
        self.ensure_backend_healthy()?;
        let started = std::time::Instant::now();
        let synchronized = self.backend.runtime_synchronize_external_yield();
        let elapsed = started.elapsed();
        self.backend_health
            .record_external_yield_sync(boundary, elapsed, synchronized.is_err());
        if elapsed >= SLOW_EXTERNAL_YIELD_SYNC {
            tracing::warn!(
                event = "slow_backend_external_yield_sync",
                backend = self.backend_name(),
                boundary,
                elapsed_ms = elapsed.as_millis() as u64,
                failed = synchronized.is_err(),
                "backend external-yield synchronization was slow"
            );
        }
        if let Err(err) = synchronized {
            let reason = format!("backend synchronization failed at {boundary}: {err:#}");
            self.backend_health.quarantine(reason.clone());
            anyhow::bail!(reason);
        }
        self.ensure_backend_healthy()?;
        Ok(())
    }

    pub fn backend_name(&self) -> &'static str {
        BackendIdentity::runtime_name(self.backend.as_ref())
    }

    /// Borrow the runner-owned backend runtime.
    ///
    /// Callers performing auxiliary model forwards must reuse this instance:
    /// accelerator backends own long-lived weight caches, resident tensor
    /// registries, and inference policy state that a freshly constructed
    /// backend would discard.
    pub fn backend_runtime(&self) -> &dyn BackendRuntime {
        self.backend.as_ref()
    }

    /// Snapshot batched recurrent-state cache ownership without changing it.
    pub fn batched_state_cache_stats(&self) -> BatchedStateCacheStats {
        let mut stats = self.batched_state_cache_counters.snapshot();
        let cache = self
            .batched_state_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.as_ref() {
            stats.entry_present = true;
            stats.logical_rows = cached.row_ids.len();
            stats.capacity_rows = cached.state.batch_size().unwrap_or(stats.logical_rows);
            stats.resident = cached
                .state
                .has_all_gdn_state_resident_kt(self.backend.as_ref());
        }
        stats
    }

    /// Snapshot direct backend-private recurrent-state owners, including
    /// resumable prefill rows that are absent from the separate batched cache.
    pub fn gdn_recurrent_state_residency_stats(&self) -> GdnRecurrentStateResidencyStats {
        ResidencyBackend::runtime_gdn_recurrent_state_residency_stats(self.backend.as_ref())
    }

    fn evict_cached_batched_state(&self, cached: CachedBatchedState) {
        cached
            .state
            .evict_gdn_state_resident_kt(self.backend.as_ref());
    }

    fn assemble_batched_linear_state(
        &self,
        linear_states: &[&mut LinearAttentionState],
        all_rows_resident: bool,
    ) -> Result<LinearAttentionState> {
        let state_refs: Vec<&LinearAttentionState> =
            linear_states.iter().map(|state| &**state).collect();
        let state = LinearAttentionState::from_batch_rows(&state_refs)?;
        let mut lease = ResidentBatchedStateLease::new(
            Some(state),
            self.backend.as_ref(),
            &self.batched_state_cache_counters,
        );
        if all_rows_resident {
            lease
                .as_ref()
                .expect("resident batched-state lease was just initialized")
                .assemble_gdn_state_resident_batch_rows_kt(&*self.backend, &state_refs)?;
        }
        Ok(lease
            .take()
            .expect("resident batched-state lease still owns assembled state"))
    }

    fn invalidate_batched_state_cache(&self) {
        self.batched_state_cache_counters
            .explicit_invalidation_count
            .fetch_add(1, Ordering::Relaxed);
        let cached = match self.batched_state_cache.lock() {
            Ok(mut cache) => cache.take(),
            Err(poisoned) => {
                tracing::warn!("recovering poisoned batched-state cache during invalidation");
                poisoned.into_inner().take()
            }
        };
        if let Some(cached) = cached {
            self.batched_state_cache_counters
                .explicit_invalidation_eviction_count
                .fetch_add(1, Ordering::Relaxed);
            self.evict_cached_batched_state(cached);
        }
    }

    fn take_batched_state_cache(&self) -> Result<Option<CachedBatchedState>> {
        let mut cache = self
            .batched_state_cache
            .lock()
            .map_err(|error| anyhow::anyhow!("failed to lock batched state cache: {error}"))?;
        let cached = cache.take();
        if cached.is_some() {
            self.batched_state_cache_counters
                .take_hit_count
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.batched_state_cache_counters
                .take_miss_count
                .fetch_add(1, Ordering::Relaxed);
            if self
                .batched_state_cache_counters
                .active_leases
                .load(Ordering::Relaxed)
                > 0
            {
                self.batched_state_cache_counters
                    .take_miss_while_leased_count
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(cached)
    }

    fn prepare_batched_linear_state<'a>(
        &'a self,
        linear_states: &[&mut LinearAttentionState],
        all_rows_resident: bool,
        row_ids: Option<&[u64]>,
    ) -> Result<(ResidentBatchedStateLease<'a>, bool)> {
        let batch = linear_states.len();
        anyhow::ensure!(batch > 0, "batched linear state requires at least one row");
        if let Some(ids) = row_ids {
            anyhow::ensure!(
                ids.len() == batch,
                "batched linear state row-id count mismatch ({} vs {batch})",
                ids.len()
            );
        }
        let state_refs: Vec<&LinearAttentionState> =
            linear_states.iter().map(|state| &**state).collect();

        if let Some(cached) = self.take_batched_state_cache()? {
            let CachedBatchedState {
                state,
                row_ids: cached_row_ids,
            } = cached;
            let mut cached_lease = ResidentBatchedStateLease::new(
                Some(state),
                self.backend.as_ref(),
                &self.batched_state_cache_counters,
            );
            let exact_match = row_ids.is_some_and(|ids| cached_row_ids == ids);
            let capacity = cached_lease
                .as_ref()
                .expect("cached batched-state lease was just initialized")
                .batch_size()?;
            let cached_is_resident = cached_lease
                .as_ref()
                .expect("cached batched-state lease was just initialized")
                .has_all_gdn_state_resident_kt(self.backend.as_ref());

            if exact_match && capacity == batch {
                self.batched_state_cache_counters
                    .exact_reuse_count
                    .fetch_add(1, Ordering::Relaxed);
                return Ok((cached_lease, true));
            }

            if row_ids.is_some() && all_rows_resident && cached_is_resident && capacity >= batch {
                self.batched_state_cache_counters
                    .resident_capacity_reuse_count
                    .fetch_add(1, Ordering::Relaxed);
                let lease = if capacity == batch {
                    cached_lease
                } else {
                    self.batched_state_cache_counters
                        .resident_prefix_view_count
                        .fetch_add(1, Ordering::Relaxed);
                    let view = cached_lease
                        .as_ref()
                        .expect("cached batched-state lease was just initialized")
                        .resident_batch_prefix_view(batch)?;
                    let capacity_state = cached_lease
                        .take()
                        .expect("cached batched-state lease still owns capacity state");
                    ResidentBatchedStateLease::with_capacity_view(
                        view,
                        capacity_state,
                        self.backend.as_ref(),
                        &self.batched_state_cache_counters,
                    )
                };
                if exact_match {
                    self.batched_state_cache_counters
                        .exact_reuse_count
                        .fetch_add(1, Ordering::Relaxed);
                } else {
                    self.batched_state_cache_counters
                        .resident_refresh_count
                        .fetch_add(1, Ordering::Relaxed);
                    lease
                        .as_ref()
                        .expect("resident capacity lease was just initialized")
                        .assemble_gdn_state_resident_batch_rows_kt(
                            self.backend.as_ref(),
                            &state_refs,
                        )?;
                }
                return Ok((lease, exact_match));
            }

            let rejection_counter = if row_ids.is_none() {
                &self
                    .batched_state_cache_counters
                    .rejected_missing_row_ids_count
            } else if !all_rows_resident {
                &self
                    .batched_state_cache_counters
                    .rejected_nonresident_rows_count
            } else if !cached_is_resident {
                &self
                    .batched_state_cache_counters
                    .rejected_nonresident_cache_count
            } else {
                debug_assert!(capacity < batch);
                &self
                    .batched_state_cache_counters
                    .rejected_insufficient_capacity_count
            };
            rejection_counter.fetch_add(1, Ordering::Relaxed);
        }

        self.batched_state_cache_counters
            .fresh_assembly_count
            .fetch_add(1, Ordering::Relaxed);
        let state = self.assemble_batched_linear_state(linear_states, all_rows_resident)?;
        Ok((
            ResidentBatchedStateLease::new(
                Some(state),
                self.backend.as_ref(),
                &self.batched_state_cache_counters,
            ),
            false,
        ))
    }

    fn park_batched_state(&self, state: LinearAttentionState, row_ids: &[u64]) {
        self.batched_state_cache_counters
            .park_count
            .fetch_add(1, Ordering::Relaxed);
        let stale = match self.batched_state_cache.lock() {
            Ok(mut cache) => cache.replace(CachedBatchedState {
                state,
                row_ids: row_ids.to_vec(),
            }),
            Err(poisoned) => {
                tracing::warn!("recovering poisoned batched-state cache while parking state");
                poisoned.into_inner().replace(CachedBatchedState {
                    state,
                    row_ids: row_ids.to_vec(),
                })
            }
        };
        if let Some(stale) = stale {
            self.batched_state_cache_counters
                .park_replacement_eviction_count
                .fetch_add(1, Ordering::Relaxed);
            self.evict_cached_batched_state(stale);
        }
    }

    fn release_batched_decode_state(&self, row_id: u64, state: &LinearAttentionState) {
        state.evict_gdn_state_resident_kt(self.backend.as_ref());
        #[cfg(feature = "vulkan")]
        if let Some(vk_backend) = BackendIdentity::runtime_as_any(self.backend.as_ref())
            .downcast_ref::<crate::backend::vulkan::VulkanBackend>()
        {
            vk_backend.evict_resident_decode_row(row_id);
        }
        // A resident cache owns reusable allocation capacity; its row IDs are
        // only a content fingerprint. Completing one of those rows makes the
        // fingerprint stale, but the next batch safely refreshes the same
        // buffers in place. Nonresident caches cannot do that and are released
        // eagerly as before.
        let take_nonresident_cache = |cache: &mut Option<CachedBatchedState>| {
            let Some(cached) = cache.as_ref() else {
                return None;
            };
            if !cached.row_ids.contains(&row_id) {
                return None;
            }
            if cached
                .state
                .has_all_gdn_state_resident_kt(self.backend.as_ref())
            {
                self.batched_state_cache_counters
                    .completed_row_preservation_count
                    .fetch_add(1, Ordering::Relaxed);
                None
            } else {
                self.batched_state_cache_counters
                    .completed_row_eviction_count
                    .fetch_add(1, Ordering::Relaxed);
                cache.take()
            }
        };
        let cached = match self.batched_state_cache.lock() {
            Ok(mut cache) => take_nonresident_cache(&mut cache),
            Err(poisoned) => {
                tracing::warn!(
                    "recovering poisoned batched-state cache while releasing decode row"
                );
                let mut cache = poisoned.into_inner();
                take_nonresident_cache(&mut cache)
            }
        };
        if let Some(cached) = cached {
            self.evict_cached_batched_state(cached);
        }
    }

    /// Release backend-private ownership for a prefill row that never reached
    /// decode. This covers native token-prefill ownership and resumable GDN
    /// state retained across ordinary prompt chunks.
    pub fn release_paged_batched_prefill_state(&self, state: &PagedBatchedPrefillState) {
        ResidencyBackend::runtime_evict_gdn_prefill_resident_state_owner(&*self.backend, state.id);
        state
            .linear_state
            .evict_gdn_recurrent_resident_states(&*self.backend);
        if state.resident_token_prefill_started {
            self.release_batched_decode_state(state.id, &state.linear_state);
        }
    }

    pub fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilityQueries::backend_capabilities(self.backend.as_ref())
    }

    pub fn training_precision_policy(&self) -> TrainingPrecisionPolicy {
        TrainingLossBackend::runtime_training_precision_policy(self.backend.as_ref())
    }

    pub fn sft_flce_loss_route(&self) -> crate::backend::SftFlceLossRoute {
        TrainingLossBackend::runtime_sft_flce_loss_route(self.backend.as_ref())
    }

    /// Eagerly allocate the backend-resident decode scratch ring when the
    /// backend supports it. This keeps the first live decode request from
    /// paying the pool feasibility/allocation cost on the request path.
    pub fn warm_resident_decode_pool(&self, max_batch: usize) -> Result<bool> {
        self.ensure_backend_healthy()?;
        let ready = ReplayBackend::runtime_decode_resident_pool_ready(
            self.backend.as_ref(),
            self.config.hidden_size,
            self.config.intermediate_size,
            max_batch,
        );
        self.ensure_backend_healthy()?;
        Ok(ready)
    }

    pub fn precompile_backend_startup_kernels(&self) -> Result<()> {
        self.ensure_backend_healthy()?;
        StartupBackend::runtime_precompile_startup_kernels(self.backend.as_ref())?;
        self.ensure_backend_healthy()
    }

    /// Preload backend-specific decode weights into persistent device caches.
    ///
    /// Prewarm is deliberately non-destructive: serving and shared-tape
    /// training retain the authoritative tensors because portable fallback and
    /// backward both read them directly.
    pub fn prewarm_backend_decode_weights(&self) -> Result<()> {
        self.prewarm_backend_decode_weights_with_policy(
            &crate::backend::DecodeWeightPrewarmPolicy::unlimited(),
        )
    }

    pub fn prewarm_backend_decode_weights_with_policy(
        &self,
        policy: &crate::backend::DecodeWeightPrewarmPolicy,
    ) -> Result<()> {
        self.ensure_backend_healthy()?;
        #[cfg(feature = "vulkan")]
        let _durable_vulkan_allocations =
            matches!(self.weights.device_kt(), kiln_tensor::Device::Vulkan(_))
                .then(kiln_vulkan_kernel::buffer_pool::durable_allocation_scope);
        LinearBackend::runtime_prewarm_decode_weights_with_policy(
            self.backend.as_ref(),
            &self.weights,
            policy,
        )?;
        self.ensure_backend_healthy()
    }

    /// Load a LoRA adapter from a PEFT-compatible directory.
    ///
    /// The directory must contain `adapter_config.json` and `adapter_model.safetensors`.
    /// Replaces any previously loaded adapter.
    pub fn load_adapter(&mut self, path: &Path) -> Result<()> {
        self.ensure_backend_healthy()?;
        // (#1082) `LoraWeights::load` is kt-native — pass kt device by value.
        let kt_device = self.weights.embed_tokens.device();
        let num_layers = self.config.num_layers;
        let lora = LoraWeights::load(path, num_layers, kt_device)
            .context("failed to load LoRA adapter")?;
        // Phase 4.1: register the adapter's LoRA tensors in the
        // backend's resident activation registry so the inference
        // path's `add_lora_delta_to_base` dispatches through
        // `lora_delta_resident` (on-device LoRA matmul) instead of
        // candle CPU `compute_lora_delta`. No-op on backends without
        // registry support.
        if let Err(e) = lora.register_with_backend(&*self.backend) {
            tracing::warn!(error = %e, "failed to register LoRA adapter with backend; \
                falling back to candle CPU LoRA delta path");
        }
        // If a previous adapter is loaded, evict it first so the
        // registry doesn't accumulate stale entries.
        if let Some(prev) = self.active_lora.take() {
            prev.evict_from_backend(&*self.backend);
        }
        self.active_lora = Some(lora);
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.rocm_graph.lock() {
            graph
                .invalidate()
                .context("failed to invalidate ROCm graphs after adapter load")?;
        }
        if let Ok(mut graph) = self.metal_graph.lock() {
            graph.invalidate();
        }
        // Adapter swap rewires the matmul weights; any cached batched
        // LinearAttentionState is per-request data (independent of weights)
        // but the cache lifecycle follows the same conservative
        // invalidation rule as `cuda_graph` so we don't try to skip the
        // assemble step across a weight-change boundary.
        self.invalidate_batched_state_cache();
        self.ensure_backend_healthy()
    }

    /// Unload the currently active LoRA adapter, reverting to base model.
    pub fn unload_adapter(&mut self) -> Result<()> {
        self.ensure_backend_healthy()?;
        if let Some(prev) = self.active_lora.take() {
            // Phase 4.1: evict the now-removed adapter's LoRA Vars
            // from the resident registry so they don't leak.
            prev.evict_from_backend(&*self.backend);
        }
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.rocm_graph.lock() {
            graph
                .invalidate()
                .context("failed to invalidate ROCm graphs after adapter unload")?;
        }
        if let Ok(mut graph) = self.metal_graph.lock() {
            graph.invalidate();
        }
        self.invalidate_batched_state_cache();
        self.ensure_backend_healthy()
    }

    /// Returns a reference to the active LoRA weights, if any.
    pub fn active_lora(&self) -> Option<&LoraWeights> {
        self.active_lora.as_ref()
    }

    pub fn packed_weight_registry(&self) -> &GpuPackedWeightRegistry {
        // Phase A.5: lazy build on first access. See `new_with_options` for the
        // 22% c=1 regression that this defers.
        self.packed_weight_registry.get_or_init(|| {
            GpuPackedWeightRegistry::from_gpu_weights(&self.weights)
                .expect("Qwen3.5 packed-weight registry must build from loaded GPU weights")
        })
    }

    pub fn ensure_decode_buffers(&self, batch: usize) -> Result<()> {
        // Phase A.6: lock-free fast path. The buffer is allocated once at the
        // largest configured graph bucket; subsequent decode steps only need a
        // load-acquire on the `OnceLock`, eliminating the ~11% c=1 regression
        // measured in Validation #5 from a per-step `Mutex::lock`.
        if let Some(buffers) = self.decode_buffers.get() {
            return buffers.ensure_batch_fits(batch);
        }
        // Phase A.5: lazy decode-buffer-config build (see `new_with_options`).
        let cfg = self
            .decode_buffer_config
            .get_or_init(|| {
                DecodeBufferConfig::graph_bucket(
                    self.decode_buffer_max_batch,
                    self.config.max_position_embeddings,
                    1,
                    16,
                    DecodeElementType::Bf16,
                )
                .expect("Qwen3.5 decode buffer config must be valid")
            })
            .clone();
        // (#1082) kt-native — DecodeBuffers::allocate takes the kt device directly.
        let kt_device = self.weights.embed_tokens.device();
        let buffers = DecodeBuffers::allocate(cfg, &kt_device)?;
        // If another thread won the race, drop our newly allocated copy
        // harmlessly and fall through to the winner's buffer.
        let _ = self.decode_buffers.set(buffers);
        self.decode_buffers
            .get()
            .expect("decode buffers initialized")
            .ensure_batch_fits(batch)
    }

    /// Atomically swap the active LoRA adapter.
    ///
    /// Pass `Some(lora)` to activate pre-loaded weights, or `None` to revert to
    /// the base model. Designed for use with `RwLock`: load weights outside the
    /// lock, then take a brief write lock to call this method.
    ///
    /// Invalidates any captured CUDA graph since the adapter change alters
    /// weight tensor pointers embedded in the graph.
    pub fn swap_lora(&mut self, lora: Option<LoraWeights>) -> Result<()> {
        self.ensure_backend_healthy()?;
        self.active_lora = lora;
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.rocm_graph.lock() {
            graph
                .invalidate()
                .context("failed to invalidate ROCm graphs after adapter swap")?;
        }
        if let Ok(mut graph) = self.metal_graph.lock() {
            graph.invalidate();
        }
        self.invalidate_batched_state_cache();
        self.ensure_backend_healthy()
    }

    fn snapshot_draft_linear_state(
        &self,
        linear_state: &LinearAttentionState,
        spec_config: &SpeculativeConfig,
    ) -> Result<LinearAttentionState> {
        let draft_linear_layers = self
            .weights
            .linear_attention_layers_in_prefix(spec_config.draft_layers);
        linear_state
            .snapshot_for_decode_rollback_prefix(draft_linear_layers)
            .context("clone draft linear-attention prefix from skip-layer prefill")
    }

    /// Generate text from a prompt string.
    ///
    /// Tokenizes the prompt, runs the autoregressive generation loop,
    /// and decodes the output tokens back to text.
    pub fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens(&prompt_tokens, params)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Create a new KV cache sized for this model.
    fn new_kv_cache(&self, max_seq_len: usize) -> Result<KvCache> {
        // #1082: `embed_tokens.device()` is now a kt `Device` (by value);
        // route through the kt-typed `KvCache::new_kt` so no candle Device
        // import is needed at the call site.
        let dtype = match self.config.dtype {
            kiln_core::config::DType::BF16 => kiln_tensor::DType::BF16,
            kiln_core::config::DType::FP16 => kiln_tensor::DType::F16,
            kiln_core::config::DType::FP32 => kiln_tensor::DType::F32,
        };
        let device = self.weights.embed_tokens.device();
        KvCache::new_kt(
            self.config.num_full_attention_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            max_seq_len,
            dtype,
            &device,
        )
    }

    /// Create a new linear attention state for GDN layers.
    fn new_linear_state(&self) -> Result<LinearAttentionState> {
        // #1082: kt `Device` by value -> pass by reference.
        let device = self.weights.embed_tokens.device();
        LinearAttentionState::new_with_batch_for_inference_runtime(
            &self.config,
            1,
            &device,
            self.backend.as_ref(),
        )
    }

    fn has_linear_attention_layers(&self) -> bool {
        self.weights.layers.iter().any(|layer| {
            matches!(
                layer.attention,
                crate::forward::GpuAttentionWeights::Linear(_)
            )
        })
    }

    pub fn cuda_graph_enabled(&self) -> Result<bool> {
        Ok(self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?
            .is_enabled())
    }

    pub fn rocm_graph_enabled(&self) -> Result<bool> {
        Ok(self
            .rocm_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?
            .is_enabled())
    }

    /// Snapshot ROCm graph configuration, circuit-breaker state, and execution
    /// counters. Counters are lifetime-monotonic for this model runner.
    pub fn rocm_graph_stats(
        &self,
    ) -> std::result::Result<crate::rocm_graph::RocmGraphStats, RocmGraphStatsUnavailable> {
        match self.rocm_graph.try_lock() {
            Ok(runner) => Ok(runner.stats()),
            Err(std::sync::TryLockError::WouldBlock) => Err(RocmGraphStatsUnavailable::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(RocmGraphStatsUnavailable::Poisoned),
        }
    }

    /// Snapshot the currently active capture phase without acquiring the model's
    /// graph-runner lock. This remains responsive during long driver calls.
    pub fn rocm_graph_live_telemetry(&self) -> RocmGraphLiveTelemetry {
        self.rocm_graph_telemetry.snapshot()
    }

    /// Clone the graph telemetry channel for ownership outside a surrounding
    /// `ModelRunner` lock. Server health paths use this to remain responsive
    /// while inference holds the runner for mutation.
    pub fn rocm_graph_telemetry_handle(&self) -> RocmGraphTelemetryHandle {
        self.rocm_graph_telemetry.clone()
    }

    /// Destroy every decode graph before a paged-KV pool replacement can free
    /// the allocation whose pointers were captured. The graph runners also
    /// validate pool identity at replay, but eager invalidation keeps native
    /// graph handles from retaining obsolete allocation state across resize.
    pub fn invalidate_decode_graphs_for_kv_pool_change(&self) -> Result<()> {
        self.cuda_graph
            .lock()
            .map_err(|error| anyhow::anyhow!("failed to lock CUDA graph runner: {error}"))?
            .invalidate();
        self.rocm_graph
            .lock()
            .map_err(|error| anyhow::anyhow!("failed to lock ROCm graph runner: {error}"))?
            .invalidate()
            .context("failed to invalidate ROCm graphs for KV pool change")?;
        self.metal_graph
            .lock()
            .map_err(|error| anyhow::anyhow!("failed to lock Metal graph runner: {error}"))?
            .invalidate();
        Ok(())
    }

    pub fn metal_graph_enabled(&self) -> Result<bool> {
        Ok(self
            .metal_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?
            .is_enabled())
    }

    /// Generate text token-by-token, sending each token to a channel as it is produced.
    ///
    /// Returns an `mpsc::Receiver<StreamEvent>` that yields `Token` events
    /// followed by a final `Done` event.  The generation runs synchronously
    /// on the calling thread (caller should use `spawn_blocking`).
    pub fn generate_streaming(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let (tx, rx) = mpsc::channel();

        let max_total = prompt_tokens.len() + params.max_tokens;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: run forward pass on all prompt tokens at once
        let logits = model_forward_kt_with_policy(
            &*self.backend,
            &prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
            self.streaming_prefill,
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        // Sample first token from the last position's logits
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);

        let mut next_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, step_seed, &[])?
        };

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            // Check for EOS
            if self.should_stop_on_eos(params, next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                next_token,
            )? {
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                StreamTokenDisposition::Finished(reason) => {
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens: generated_tokens.len(),
                        trailing_text: String::new(),
                    }));
                    return Ok(rx);
                }
                StreamTokenDisposition::Continue => {}
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            // Decode step: forward pass on just the new token
            let logits = model_forward_kt_with_policy(
                &*self.backend,
                &[next_token],
                &self.weights,
                &self.config,
                Some(&mut kv_cache),
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                self.streaming_prefill,
            )
            .context("decode forward pass failed")?;
            kv_cache.advance(1);

            next_token = if params.is_effectively_greedy() {
                greedy_sample(&logits)?
            } else {
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        let (trailing_text, late_stop) = gate.finish(&self.tokenizer, &generated_tokens)?;
        let (finish_reason, trailing_text) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, trailing_text),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text,
        }));

        Ok(rx)
    }

    /// Autoregressive generation loop operating on token IDs.
    ///
    /// 1. Prefill: run forward pass on the full prompt to get first next-token logits.
    /// 2. Decode: repeatedly sample a token, run forward on just the new token.
    /// 3. Stop on EOS, max_tokens, or stop sequence.
    pub fn generate_from_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let max_total = prompt_tokens.len() + params.max_tokens;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: run forward pass on all prompt tokens at once
        let logits = model_forward_kt_with_policy(
            &*self.backend,
            prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
            self.streaming_prefill,
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        // Sample first token from the last position's logits
        let mut next_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, step_seed, &[])?
        };

        for _step in 0..params.max_tokens {
            // Advance seed for next step
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            // Check for EOS
            if self.should_stop_on_eos(params, next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            // Check stop sequences against decoded text so far
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            // Decode step: forward pass on just the new token (KV cache has all previous)
            let logits = model_forward_kt_with_policy(
                &*self.backend,
                &[next_token],
                &self.weights,
                &self.config,
                Some(&mut kv_cache),
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                self.streaming_prefill,
            )
            .context("decode forward pass failed")?;
            kv_cache.advance(1);

            // Sample next token from the new logits
            next_token = if params.is_effectively_greedy() {
                greedy_sample(&logits)?
            } else {
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Compute the number of blocks needed for a given number of tokens.
    fn blocks_needed(num_tokens: usize, block_size: usize) -> usize {
        (num_tokens + block_size - 1) / block_size
    }

    /// Initial block capacity for the batching engine.
    ///
    /// Batched decode can grow its per-request block table as generation crosses
    /// block boundaries, so it should not reserve `prompt + max_tokens` up front.
    /// Large OpenAI-compatible clients commonly send very high `max_tokens`;
    /// making the decode block table that large turns every token into a
    /// long-context operation even when the model stops after a tool call.
    fn initial_batched_decode_blocks_needed(
        prompt_tokens: usize,
        max_tokens: usize,
        block_size: usize,
    ) -> usize {
        let initial_tokens = prompt_tokens.saturating_add(usize::from(max_tokens > 0));
        Self::blocks_needed(initial_tokens, block_size)
    }

    /// Generate text from a prompt using paged KV cache backed by a BlockManager.
    ///
    /// This is the memory-efficient path: blocks are allocated on demand from the
    /// shared BlockManager pool and freed when generation completes.
    pub fn generate_paged(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &PagedKvCache,
    ) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_paged(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
            None,
        )?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Autoregressive generation using paged KV cache.
    ///
    /// Allocates blocks from `block_manager` as needed and frees them when done.
    pub fn generate_from_tokens_paged(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &PagedKvCache,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = block_manager.block_size();
        let max_total = prompt_tokens.len() + params.max_tokens;

        // Pre-allocate blocks for the maximum possible sequence length
        let num_blocks = Self::blocks_needed(max_total, block_size);
        let allocated_blocks = block_manager
            .allocate(num_blocks)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut block_table = BlockTable::new();
        for &block_id in &allocated_blocks {
            block_table.push(block_id);
        }

        // Run generation with paged cache; free blocks on completion (or error)
        let result = self.generate_from_tokens_paged_inner(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            cancel,
        );

        // Always free allocated blocks
        block_manager.free_all(&allocated_blocks);

        result
    }

    /// Generate text from a prompt using shared paged-cache state protected by
    /// short-lived mutexes.
    ///
    /// On backends with CUDA graph replay enabled we preserve the existing
    /// whole-request lock scope because the graph state is runner-global.
    /// On non-CUDA desktop paths (Metal / CPU), blocks are reserved up front,
    /// the block manager is released immediately, and the paged cache is locked
    /// only around prefill / decode forward passes so concurrent requests can
    /// interleave between decode steps.
    pub fn generate_paged_shared(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
    ) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_paged_shared(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
            None,
        )?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Same as [`generate_paged_shared`], but optionally reuses a
    /// block-aligned cached prefix and returns a completed prompt snapshot that
    /// the caller may register after successful generation.
    pub fn generate_paged_shared_tokens_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
        cancel: Option<&CancelHandle>,
    ) -> Result<PrefixCachedGenerationOutput> {
        let output = self.generate_from_tokens_paged_interleaved_with_prefix_cache(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
            cached_prefix,
            cancel,
        )?;

        let text = match self
            .tokenizer
            .decode(&output.output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")
        {
            Ok(text) => text,
            Err(decode_err) => {
                if let Err(sync_err) =
                    self.synchronize_external_yield("prefix generation tokenizer failure")
                {
                    std::mem::forget(output);
                    return Err(sync_err.context(format!(
                        "tokenizer decode also failed before synchronization: {decode_err:#}"
                    )));
                }
                if !output.allocated_blocks.is_empty() {
                    let mut bm_guard = lock_block_manager(block_manager)?;
                    bm_guard.free_all(&output.allocated_blocks);
                }
                return Err(decode_err);
            }
        };

        Ok(PrefixCachedGenerationOutput {
            output: GenerationOutput {
                text,
                token_ids: output.output.token_ids,
                finish_reason: output.output.finish_reason,
            },
            registration: output.registration,
            extra_registrations: output.extra_registrations,
            allocated_blocks: output.allocated_blocks,
            prefill_duration: output.prefill_duration,
            decode_duration: output.decode_duration,
        })
    }

    pub fn prepare_paged_batched_decode_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
        capture_prefix_split: bool,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedDecodeState> {
        let start = self.begin_paged_batched_decode_with_prefix_cache(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
            cached_prefix,
            capture_prefix_split,
            cancel,
        )?;
        let state = match start {
            PagedBatchedPrefillStart::Ready(state) => return Ok(state),
            PagedBatchedPrefillStart::Prefilling(state) => state,
        };
        let mut state = Some(state);
        loop {
            match self.advance_paged_batched_prefill(
                &mut state,
                params,
                paged_cache,
                usize::MAX,
                cancel,
            ) {
                Ok(PagedBatchedPrefillProgress {
                    decode_state: Some(state),
                    ..
                }) => return Ok(state),
                Ok(PagedBatchedPrefillProgress {
                    decode_state: None, ..
                }) => {}
                Err(prepare_err) => {
                    if let Err(sync_err) =
                        self.synchronize_external_yield("batched prefill failure cleanup")
                    {
                        std::mem::forget(state.take());
                        return Err(sync_err.context(format!(
                            "batched prefill also failed before synchronization: {prepare_err:#}"
                        )));
                    }
                    if let Some(prefill) = state.as_ref() {
                        self.release_paged_batched_prefill_state(prefill);
                    }
                    let allocated_blocks = state
                        .take()
                        .map(PagedBatchedPrefillState::into_allocated_blocks)
                        .unwrap_or_default();
                    if !allocated_blocks.is_empty() {
                        let mut bm_guard = lock_block_manager(block_manager)?;
                        bm_guard.free_all(&allocated_blocks);
                    }
                    return Err(prepare_err);
                }
            }
        }
    }

    /// Resolve prefix reuse and allocate paged ownership without executing an
    /// unbounded prompt forward. Exact cache hits are immediately ready;
    /// otherwise the returned state must be advanced and externally
    /// synchronized one bounded quantum at a time.
    pub fn begin_paged_batched_decode_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        _paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
        capture_prefix_split: bool,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedPrefillStart> {
        self.begin_paged_batched_decode_with_prefix_cache_and_behavior_logprobs(
            prompt_tokens,
            params,
            block_manager,
            _paged_cache,
            cached_prefix,
            capture_prefix_split,
            false,
            cancel,
        )
    }

    /// Begin paged batched generation with optional exact behavior-policy
    /// log-probability capture. Capture affects only sampling/output metadata;
    /// prefix lookup and KV ownership remain identical to ordinary serving.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_paged_batched_decode_with_prefix_cache_and_behavior_logprobs(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        _paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
        capture_prefix_split: bool,
        capture_behavior_logprobs: bool,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedPrefillStart> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        check_cancelled(cancel)?;

        let block_size = {
            let bm_guard = lock_block_manager(block_manager)?;
            bm_guard.block_size()
        };

        let cached_prefix = cached_prefix.filter(|prefix| {
            paged_prefix_reuse_matches_prompt(prefix, prompt_tokens.len(), block_size, params)
        });

        let cached_blocks = cached_prefix
            .as_ref()
            .map(|prefix| prefix.block_ids.as_slice())
            .unwrap_or(&[]);

        let total_blocks = Self::initial_batched_decode_blocks_needed(
            prompt_tokens.len(),
            params.max_tokens,
            block_size,
        );
        let additional_blocks_needed = total_blocks.saturating_sub(cached_blocks.len());
        let allocated_blocks = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            bm_guard
                .allocate(additional_blocks_needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let block_table = append_prefix_block_table(cached_blocks, &allocated_blocks);

        let prepared = self.begin_paged_batched_decode_with_prefix_blocks(
            prompt_tokens,
            params,
            block_table,
            cached_prefix,
            block_size,
            allocated_blocks.clone(),
            capture_prefix_split,
            capture_behavior_logprobs,
            cancel,
        );

        match prepared {
            Ok(state) => Ok(state),
            Err(prepare_err) => {
                if let Err(sync_err) =
                    self.synchronize_external_yield("batched prefill initialization failure")
                {
                    return Err(sync_err.context(format!(
                        "batched prefill initialization also failed before synchronization: {prepare_err:#}"
                    )));
                }
                if !allocated_blocks.is_empty() {
                    let mut bm_guard = lock_block_manager(block_manager)?;
                    bm_guard.free_all(&allocated_blocks);
                }
                Err(prepare_err)
            }
        }
    }

    /// Same as [`generate_paged_shared`], but accepts an already-tokenized
    /// prompt so API callers do not render/tokenize the same prompt twice.
    ///
    /// The optional `cancel` handle is polled between decode tokens so that
    /// callers (notably `kiln-server`'s `tokio::time::timeout` path) can drain
    /// the still-running blocking work after a request timeout fires, instead
    /// of leaving the closure running with `runner` / `prefix_cache` locks
    /// held — see #664.
    pub fn generate_paged_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let output = self.generate_from_tokens_paged_shared(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
            cancel,
        )?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    fn generate_from_tokens_paged_shared(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let cuda_graph_enabled = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?
            .is_enabled();

        // Phase 12-B'': allocate blocks under a one-shot BlockManager lock and
        // wrap them in `SharedBlockReservation` so the BM lock is released
        // before any forward passes run. The CUDA-graph branch previously held
        // both the BM and the PagedKvCache locks for the entire generation
        // (~2.3 s for a 512-prompt, 128-decode run), which forced concurrent
        // requests onto a serial staircase even with c=8. Phase 12-C removed
        // the global `Mutex<PagedKvCache>` entirely: the cache now uses
        // interior mutability so forward passes can take `&PagedKvCache`
        // concurrently, with disjoint block tables per request providing
        // safety.
        let max_total = prompt_tokens.len() + params.max_tokens;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = if cuda_graph_enabled {
            self.generate_from_tokens_paged_cuda_graph_interleaved(
                prompt_tokens,
                params,
                paged_cache,
                &block_table,
                cancel,
            )
        } else {
            self.generate_from_tokens_paged_interleaved(
                prompt_tokens,
                params,
                paged_cache,
                &block_table,
                cancel,
            )
        };

        reservation.release_after_settlement(self, "direct shared KV release", result)
    }

    fn generate_from_tokens_paged_interleaved_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
        cancel: Option<&CancelHandle>,
    ) -> Result<PrefixCachedGenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = {
            let bm_guard = lock_block_manager(block_manager)?;
            bm_guard.block_size()
        };

        let cached_prefix = cached_prefix.filter(|prefix| {
            paged_prefix_reuse_matches_prompt(prefix, prompt_tokens.len(), block_size, params)
        });

        let cached_blocks = cached_prefix
            .as_ref()
            .map(|prefix| prefix.block_ids.as_slice())
            .unwrap_or(&[]);

        let max_total = prompt_tokens.len() + params.max_tokens;
        let total_blocks = Self::blocks_needed(max_total, block_size);
        let additional_blocks_needed = total_blocks.saturating_sub(cached_blocks.len());
        let allocated_blocks = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            bm_guard
                .allocate(additional_blocks_needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let block_table = append_prefix_block_table(cached_blocks, &allocated_blocks);

        let result = self.generate_from_tokens_paged_interleaved_with_prefix_blocks(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            cached_prefix,
            block_size,
            cancel,
        );

        match result {
            Ok(mut output) => {
                output.allocated_blocks = allocated_blocks;
                Ok(output)
            }
            Err(err) => {
                if let Err(sync_err) =
                    self.synchronize_external_yield("prefix generation failure cleanup")
                {
                    return Err(sync_err.context(format!(
                        "prefix generation also failed before synchronization: {err:#}"
                    )));
                }
                if !allocated_blocks.is_empty() {
                    let mut bm_guard = lock_block_manager(block_manager)?;
                    bm_guard.free_all(&allocated_blocks);
                }
                Err(err)
            }
        }
    }

    fn generate_from_tokens_paged_interleaved_with_prefix_blocks(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cached_prefix: Option<PagedPrefixReuse>,
        block_size: usize,
        cancel: Option<&CancelHandle>,
    ) -> Result<PrefixCachedGenerationOutput> {
        let (cached_tokens, exact_next_token, mut linear_state) = match cached_prefix {
            Some(prefix) => {
                let exact_next_token = if prefix.cached_tokens == prompt_tokens.len() {
                    prefix.next_token
                } else {
                    None
                };
                (prefix.cached_tokens, exact_next_token, prefix.linear_state)
            }
            None => (0, None, self.new_linear_state()?),
        };

        if let Some(next_token) = exact_next_token {
            let decode_start = std::time::Instant::now();
            let output = match next_token {
                PagedPrefixNextToken::Logits(logits) => self.decode_from_prefill_logits(
                    logits,
                    prompt_tokens.len(),
                    params,
                    paged_cache,
                    block_table,
                    &mut linear_state,
                    cancel,
                )?,
                PagedPrefixNextToken::GreedyToken(token) => {
                    anyhow::ensure!(
                        params.is_effectively_greedy(),
                        "greedy cached first token cannot serve non-greedy sampling"
                    );
                    self.decode_from_prefill_token(
                        token,
                        prompt_tokens.len(),
                        params,
                        paged_cache,
                        block_table,
                        &mut linear_state,
                        params.seed,
                        cancel,
                    )?
                }
            };

            return Ok(PrefixCachedGenerationOutput {
                output,
                registration: None,
                extra_registrations: Vec::new(),
                allocated_blocks: Vec::new(),
                prefill_duration: std::time::Duration::ZERO,
                decode_duration: decode_start.elapsed(),
            });
        }

        let prefill_tokens = &prompt_tokens[cached_tokens..];
        anyhow::ensure!(
            !prefill_tokens.is_empty(),
            "non-exact prefix cache hit must leave at least one suffix token"
        );

        let use_greedy_prefill_token = params.is_effectively_greedy()
            && greedy_token_decode_enabled(self.backend.as_ref())
            && !self.streaming_prefill.enabled_for(prefill_tokens.len());
        // Same capability gate as the batching-engine path: the split
        // snapshot is what makes multi-turn strict-prefix lookups possible
        // (RealPrefixCache only serves longer prompts from block-aligned
        // entries), so a backend opting out here opts out of multi-turn
        // prefix caching entirely.
        let split_pos = prefix_cache_split_snapshot_allowed(self.backend.as_ref())
            .then(|| strict_prompt_prefix_split_pos(prompt_tokens.len(), cached_tokens, block_size))
            .flatten();
        let mut prefill_split_snapshot: Option<RollingPrefixSnapshot> = None;
        let prefill_start = std::time::Instant::now();
        let prefill_source = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if self.streaming_prefill.enabled_for(prefill_tokens.len()) {
                if let Some(split_pos) = split_pos {
                    let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                    let _ = model_forward_paged_streaming_with_progress_and_policy(
                        &*self.backend,
                        head_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        cancel,
                        self.streaming_prefill,
                    )
                    .context("prefill forward pass (paged prefix cache, streaming head)")?;
                    prefill_split_snapshot = self
                        .authoritative_prefix_snapshot(
                            &linear_state,
                            "streaming-prefill-split",
                            split_pos,
                        )
                        .context("snapshot linear state at streaming prefill split")?
                        .map(|linear_state| RollingPrefixSnapshot {
                            position: split_pos,
                            linear_state,
                        });

                    let tail_tokens = &prompt_tokens[split_pos..];
                    let logits = model_forward_paged_streaming_with_progress_offset_and_policy(
                        &*self.backend,
                        tail_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        split_pos,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        cancel,
                        head_tokens.len() as u64,
                        self.streaming_prefill,
                    )
                    .context("prefill forward pass (paged prefix cache, streaming tail)")?;
                    if let Some(cancel) = cancel {
                        cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                    }
                    // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                    PrefillSampleSource::Logits(logits)
                } else {
                    let logits = model_forward_paged_streaming_with_progress_and_policy(
                        &*self.backend,
                        prefill_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        cancel,
                        self.streaming_prefill,
                    )
                    .context("prefill forward pass (paged prefix cache, streaming) failed")?;
                    PrefillSampleSource::Logits(logits)
                }
            } else if use_greedy_prefill_token {
                if let Some(split_pos) = split_pos {
                    // Split the prefill at the last block boundary so the
                    // linear-attention state can be snapshotted at the
                    // cross-turn-safe position (mirrors the batching-engine
                    // path). The head pass's logits are discarded.
                    let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                    let _ = model_forward_paged_last_token(
                        &*self.backend,
                        head_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                    .context("greedy prefill forward pass (paged prefix cache, head) failed")?;
                    if let Some(cancel) = cancel {
                        cancel.report_prefill_tokens_completed(head_tokens.len() as u64);
                    }
                    check_cancelled(cancel)?;
                    prefill_split_snapshot = self
                        .authoritative_prefix_snapshot(
                            &linear_state,
                            "greedy-prefill-split",
                            split_pos,
                        )
                        .context("snapshot linear state at greedy prefill split")?
                        .map(|linear_state| RollingPrefixSnapshot {
                            position: split_pos,
                            linear_state,
                        });

                    let tail_tokens = &prompt_tokens[split_pos..];
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    let token = model_forward_paged_last_token_greedy(
                        &*self.backend,
                        tail_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        split_pos,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                    .context("greedy prefill forward pass (paged prefix cache, tail) failed")?;
                    if let Some(cancel) = cancel {
                        cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                    }
                    check_cancelled(cancel)?;
                    PrefillSampleSource::GreedyToken(token)
                } else {
                    let token = model_forward_paged_last_token_greedy(
                        &*self.backend,
                        prefill_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                    .context("greedy prefill forward pass (paged prefix cache) failed")?;
                    if let Some(cancel) = cancel {
                        cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                    }
                    check_cancelled(cancel)?;
                    PrefillSampleSource::GreedyToken(token)
                }
            } else if let Some(split_pos) = split_pos {
                // Split the prefill at the last block boundary so the
                // linear-attention state can be snapshotted at the
                // cross-turn-safe position (mirrors the batching-engine
                // path). The head pass's logits are discarded.
                let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                let _ = model_forward_paged_last_token(
                    &*self.backend,
                    head_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    cached_tokens,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged prefix cache, head) failed")?;
                prefill_split_snapshot = self
                    .authoritative_prefix_snapshot(&linear_state, "prefill-split", split_pos)
                    .context("snapshot linear state at prefill split")?
                    .map(|linear_state| RollingPrefixSnapshot {
                        position: split_pos,
                        linear_state,
                    });
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(head_tokens.len() as u64);
                }

                let tail_tokens = &prompt_tokens[split_pos..];
                let pc_guard = lock_paged_cache(paged_cache)?;
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    tail_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    split_pos,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged prefix cache, tail) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                }
                // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                PrefillSampleSource::Logits(logits)
            } else {
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    prefill_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    cached_tokens,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged prefix cache) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                }
                // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                PrefillSampleSource::Logits(logits)
            }
        };

        let prefill_duration = prefill_start.elapsed();
        let registration = self.completed_prompt_registration(
            prompt_tokens,
            block_table,
            &linear_state,
            block_size,
            Some(prefill_source.cached_next_token()),
        )?;

        let decode_start = std::time::Instant::now();
        let output = match prefill_source {
            PrefillSampleSource::Logits(logits) => self.decode_from_prefill_logits(
                logits,
                prompt_tokens.len(),
                params,
                paged_cache,
                block_table,
                &mut linear_state,
                cancel,
            )?,
            PrefillSampleSource::GreedyToken(token) => self.decode_from_prefill_token(
                token,
                prompt_tokens.len(),
                params,
                paged_cache,
                block_table,
                &mut linear_state,
                params.seed,
                cancel,
            )?,
        };

        let decode_duration = decode_start.elapsed();
        let mut extra_registrations = Vec::new();
        if let Some(reg) = build_extended_registration(
            prompt_tokens,
            &output.token_ids,
            block_table,
            block_size,
            prefill_split_snapshot,
        ) {
            extra_registrations.push(reg);
        }

        Ok(PrefixCachedGenerationOutput {
            output,
            registration,
            extra_registrations,
            allocated_blocks: Vec::new(),
            prefill_duration,
            decode_duration,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_paged_batched_decode_with_prefix_blocks(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_table: BlockTable,
        cached_prefix: Option<PagedPrefixReuse>,
        block_size: usize,
        allocated_blocks: Vec<u32>,
        capture_prefix_split: bool,
        capture_behavior_logprobs: bool,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedPrefillStart> {
        let (cached_tokens, exact_next_token, linear_state) = match cached_prefix {
            Some(prefix) => {
                let exact_next_token = if prefix.cached_tokens == prompt_tokens.len() {
                    prefix.next_token
                } else {
                    None
                };
                (prefix.cached_tokens, exact_next_token, prefix.linear_state)
            }
            None => (0, None, self.new_linear_state()?),
        };

        if let Some(next_token) = exact_next_token {
            let (next_token, next_token_logprob) = match next_token {
                PagedPrefixNextToken::Logits(logits) if capture_behavior_logprobs => {
                    let sampled = sample_first_decode_token_with_logprob(&logits, params)?;
                    (sampled.token_id, Some(sampled.logprob))
                }
                PagedPrefixNextToken::Logits(logits) => {
                    (sample_first_decode_token(&logits, params)?, None)
                }
                PagedPrefixNextToken::GreedyToken(token) => {
                    anyhow::ensure!(
                        params.is_effectively_greedy(),
                        "greedy cached first token cannot serve non-greedy sampling"
                    );
                    (token, capture_behavior_logprobs.then_some(0.0))
                }
            };
            return Ok(PagedBatchedPrefillStart::Ready(PagedBatchedDecodeState {
                block_table,
                linear_state,
                seq_len: prompt_tokens.len(),
                next_token,
                next_token_logprob,
                generated_tokens: Vec::new(),
                step_seed: params.seed,
                capture_behavior_logprobs,
                registration: None,
                allocated_blocks,
                prefill_duration: std::time::Duration::ZERO,
                decode_duration: std::time::Duration::ZERO,
                prompt_tokens: prompt_tokens.to_vec(),
                block_size,
                prefill_split_snapshot: None,
                rolling_snapshot: None,
                prefix_cache_registration_allowed: true,
                id: next_decode_row_id(),
            }));
        }

        anyhow::ensure!(
            cached_tokens < prompt_tokens.len(),
            "prefix cache hit must leave at least one suffix token"
        );
        check_cancelled(cancel)?;
        let capture_prefix_split =
            capture_prefix_split && prefix_cache_split_snapshot_allowed(self.backend.as_ref());
        let split_pos = capture_prefix_split
            .then(|| strict_prompt_prefix_split_pos(prompt_tokens.len(), cached_tokens, block_size))
            .flatten();
        let streaming = self
            .streaming_prefill
            .enabled_for(prompt_tokens.len().saturating_sub(cached_tokens));

        Ok(PagedBatchedPrefillStart::Prefilling(
            PagedBatchedPrefillState {
                block_table,
                linear_state,
                prompt_tokens: prompt_tokens.to_vec(),
                cached_tokens,
                next_position: cached_tokens,
                block_size,
                allocated_blocks,
                split_pos,
                prefill_split_snapshot: None,
                streaming,
                prefill_duration: std::time::Duration::ZERO,
                capture_behavior_logprobs,
                pending_logits: None,
                pending_layer_forward: None,
                pending_chunk_end: None,
                id: next_decode_row_id(),
                resident_token_prefill_started: false,
            },
        ))
    }

    /// Execute at most `max_tokens` prompt tokens. The state remains in the
    /// caller-owned option on every error, allowing the caller to synchronize
    /// and either release or deliberately retain all accelerator ownership.
    pub fn advance_paged_batched_prefill(
        &self,
        prefill: &mut Option<PagedBatchedPrefillState>,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        max_tokens: usize,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedPrefillProgress> {
        self.advance_paged_batched_prefill_with_layer_budget(
            prefill,
            params,
            paged_cache,
            max_tokens,
            usize::MAX,
            cancel,
        )
    }

    /// Execute at most `max_layers` of one `max_tokens` prompt chunk.
    /// `tokens_scheduled` charges a new chunk exactly once; `tokens_processed`
    /// reports only final-layer completion. A retained layer group may
    /// therefore report zero for both token fields while still making layer
    /// progress.
    pub fn advance_paged_batched_prefill_with_layer_budget(
        &self,
        prefill: &mut Option<PagedBatchedPrefillState>,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        max_tokens: usize,
        max_layers: usize,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedPrefillProgress> {
        let owner_id = prefill.as_ref().map(|state| state.id);
        let _resident_scope = GdnPrefillResidentStateScope::new(&*self.backend, owner_id);
        self.advance_paged_batched_prefill_with_layer_budget_inner(
            prefill,
            params,
            paged_cache,
            max_tokens,
            max_layers,
            cancel,
        )
    }

    fn advance_paged_batched_prefill_with_layer_budget_inner(
        &self,
        prefill: &mut Option<PagedBatchedPrefillState>,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        max_tokens: usize,
        max_layers: usize,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedPrefillProgress> {
        anyhow::ensure!(max_layers > 0, "prefill layer quantum must be positive");
        check_cancelled(cancel)?;

        let state = prefill
            .as_mut()
            .context("paged batched prefill state was already consumed")?;
        anyhow::ensure!(
            !state.resident_token_prefill_started,
            "resident token-prefill row cannot resume through the generic paged KV path"
        );
        anyhow::ensure!(
            state.next_position < state.prompt_tokens.len(),
            "paged batched prefill has no remaining prompt tokens"
        );
        // A completed chunk's output was retained only until the caller's
        // external synchronization. An in-flight layer group instead retains
        // its own hidden/position tensors until that chunk finishes.
        if state.pending_layer_forward.is_none() {
            state.pending_logits.take();
        }
        let chunk_start = state.next_position;
        let (chunk_end, chunk_started) = match state.pending_chunk_end {
            Some(chunk_end) => (chunk_end, false),
            None => {
                anyhow::ensure!(max_tokens > 0, "prefill token quantum must be positive");
                let mut chunk_end = chunk_start
                    .saturating_add(max_tokens)
                    .min(state.prompt_tokens.len());
                if let Some(split_pos) = state.split_pos
                    && split_pos > chunk_start
                    && split_pos < chunk_end
                {
                    chunk_end = split_pos;
                }
                state.pending_chunk_end = Some(chunk_end);
                (chunk_end, true)
            }
        };
        let chunk_len = chunk_end.saturating_sub(chunk_start);
        anyhow::ensure!(chunk_len > 0, "prefill quantum made no progress");
        let tokens_scheduled = usize::from(chunk_started).saturating_mul(chunk_len);

        let started = std::time::Instant::now();
        let layer_bounded = max_layers != usize::MAX || state.pending_layer_forward.is_some();
        let forward: Result<(Option<kiln_tensor::Tensor>, usize)> = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let tokens = &state.prompt_tokens[chunk_start..chunk_end];
            if state.streaming && !layer_bounded {
                Ok((
                    Some(
                        model_forward_paged_streaming_with_progress_offset_and_policy(
                            &*self.backend,
                            tokens,
                            &self.weights,
                            &self.config,
                            pc_guard,
                            &state.block_table,
                            chunk_start,
                            Some(&mut state.linear_state),
                            self.active_lora.as_ref(),
                            cancel,
                            chunk_start.saturating_sub(state.cached_tokens) as u64,
                            self.streaming_prefill,
                        )
                        .context("batched-engine chunked streaming prefill failed")?,
                    ),
                    self.weights.layers.len(),
                ))
            } else {
                let progress = model_forward_paged_last_token_layer_group(
                    &*self.backend,
                    tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &state.block_table,
                    chunk_start,
                    &mut state.linear_state,
                    self.active_lora.as_ref(),
                    state.pending_layer_forward.take(),
                    max_layers,
                )
                .context("batched-engine layer-bounded chunked prefill failed")?;
                anyhow::ensure!(
                    progress.layers_processed > 0,
                    "layer-bounded prefill reported no transformer progress"
                );
                state.pending_layer_forward = progress.state;
                Ok((progress.logits, progress.layers_processed))
            }
        };
        state.prefill_duration += started.elapsed();
        let (logits, layers_processed) = forward?;
        if state.pending_layer_forward.is_some() {
            anyhow::ensure!(
                logits.is_none(),
                "partial layer-bounded prefill returned final logits"
            );
            check_cancelled(cancel)?;
            return Ok(PagedBatchedPrefillProgress {
                tokens_scheduled,
                tokens_processed: 0,
                layers_processed,
                decode_state: None,
            });
        }
        let logits = logits.context("completed layer-bounded prefill returned no logits")?;
        state.pending_chunk_end = None;
        state.next_position = chunk_end;
        state.pending_logits = Some(logits);

        if !state.streaming
            && let Some(cancel) = cancel
        {
            cancel.report_prefill_tokens_completed(state.processed_tokens() as u64);
        }
        if state.split_pos == Some(chunk_end) && state.prefill_split_snapshot.is_none() {
            state.prefill_split_snapshot = self
                .authoritative_prefix_snapshot(
                    &state.linear_state,
                    "chunked-prefill-split",
                    chunk_end,
                )
                .context("snapshot linear state at chunked prefill split")?;
        }
        check_cancelled(cancel)?;

        if state.next_position < state.prompt_tokens.len() {
            state
                .linear_state
                .apply_gdn_prefill_resident_state_boundary(&*self.backend, state.id)
                .context("apply resumable prefill GDN state precision boundary")?;
            return Ok(PagedBatchedPrefillProgress {
                tokens_scheduled,
                tokens_processed: chunk_len,
                layers_processed,
                decode_state: None,
            });
        }

        state
            .linear_state
            .materialize_gdn_prefill_resident_states(&*self.backend, state.id)
            .context("materialize resumable prefill GDN state before decode handoff")?;
        let logits = state
            .pending_logits
            .as_ref()
            .context("completed prefill did not retain final logits")?;
        let (next_token, next_token_logprob) = if state.capture_behavior_logprobs {
            let sampled = sample_first_decode_token_with_logprob(logits, params)?;
            (sampled.token_id, Some(sampled.logprob))
        } else {
            (sample_first_decode_token(logits, params)?, None)
        };
        let registration = self.completed_prompt_registration(
            &state.prompt_tokens,
            &state.block_table,
            &state.linear_state,
            state.block_size,
            Some(PagedPrefixNextToken::Logits(logits.clone())),
        )?;

        let mut state = prefill
            .take()
            .context("completed paged batched prefill state disappeared")?;
        let prefill_split_snapshot = match (state.split_pos, state.prefill_split_snapshot.take()) {
            (Some(position), Some(linear_state)) => Some(RollingPrefixSnapshot {
                position,
                linear_state,
            }),
            _ => None,
        };
        Ok(PagedBatchedPrefillProgress {
            tokens_scheduled,
            tokens_processed: chunk_len,
            layers_processed,
            decode_state: Some(PagedBatchedDecodeState {
                block_table: state.block_table,
                linear_state: state.linear_state,
                seq_len: state.prompt_tokens.len(),
                next_token,
                next_token_logprob,
                generated_tokens: Vec::new(),
                step_seed: params.seed,
                capture_behavior_logprobs: state.capture_behavior_logprobs,
                registration,
                allocated_blocks: state.allocated_blocks,
                prefill_duration: state.prefill_duration,
                decode_duration: std::time::Duration::ZERO,
                prompt_tokens: state.prompt_tokens,
                block_size: state.block_size,
                prefill_split_snapshot,
                rolling_snapshot: None,
                prefix_cache_registration_allowed: true,
                id: state.id,
            }),
        })
    }

    /// Advance one prompt token per row through the native resident Vulkan
    /// batch stack.
    ///
    /// The ordinary actor path remains token-chunked and layer-resumable. This
    /// narrower route is eligible only after each row has committed initial KV
    /// state, only for effectively greedy rows without behavior-logprob
    /// capture, and only when at least two rows can enter together. Once a row
    /// enters, it stays eligible as a single-row tail so authority never moves
    /// back from the resident Vulkan KV cache to the now-stale generic cache.
    /// A decline is reported as `Ok(None)` before any state is mutated.
    pub fn advance_paged_batched_prefill_resident_token_batch(
        &self,
        prefills: &mut [&mut Option<PagedBatchedPrefillState>],
        params: &[SamplingParams],
        paged_cache: &PagedKvCache,
        cancels: &[&CancelHandle],
    ) -> Result<Option<Vec<PagedBatchedPrefillProgress>>> {
        let batch = prefills.len();
        anyhow::ensure!(
            params.len() == batch && cancels.len() == batch,
            "resident token-prefill batch metadata length mismatch"
        );
        if batch == 0
            || self.backend.name() != "vulkan"
            || self.active_lora.is_some()
            || !self.config.attn_output_gate
            || !ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
            || !ReplayBackend::runtime_decode_resident_pool_ready(
                self.backend.as_ref(),
                self.config.hidden_size,
                self.config.intermediate_size,
                64,
            )
        {
            return Ok(None);
        }

        let mut any_resident = false;
        for (idx, (prefill, params)) in prefills.iter().zip(params).enumerate() {
            let state = prefill
                .as_ref()
                .with_context(|| format!("resident token-prefill row {idx} has no state"))?;
            any_resident |= state.resident_token_prefill_started;
            if state.next_position == 0
                || state.remaining_tokens() == 0
                || state.pending_layer_forward.is_some()
                || state.pending_chunk_end.is_some()
                || state.capture_behavior_logprobs
                || !params.is_effectively_greedy()
            {
                return Ok(None);
            }
        }
        if batch == 1 && !any_resident {
            return Ok(None);
        }
        for cancel in cancels {
            check_cancelled(Some(cancel))?;
        }

        let input_tokens: Vec<TokenId> = prefills
            .iter()
            .map(|prefill| {
                let state = prefill
                    .as_ref()
                    .expect("resident token-prefill state validated above");
                state.prompt_tokens[state.next_position]
            })
            .collect();
        let block_tables_owned: Vec<BlockTable> = prefills
            .iter()
            .map(|prefill| {
                prefill
                    .as_ref()
                    .expect("resident token-prefill state validated above")
                    .block_table
                    .clone()
            })
            .collect();
        let block_tables: Vec<&BlockTable> = block_tables_owned.iter().collect();
        let seq_lens: Vec<usize> = prefills
            .iter()
            .map(|prefill| {
                prefill
                    .as_ref()
                    .expect("resident token-prefill state validated above")
                    .next_position
            })
            .collect();
        let row_ids: Vec<u64> = prefills
            .iter()
            .map(|prefill| {
                prefill
                    .as_ref()
                    .expect("resident token-prefill state validated above")
                    .id
            })
            .collect();
        // From this point onward a failed or cancelled call may have created
        // backend-private row ownership. Mark it before entering the native
        // stack so every error path releases conservatively.
        for prefill in prefills.iter_mut() {
            prefill
                .as_mut()
                .expect("resident token-prefill state validated above")
                .resident_token_prefill_started = true;
        }
        let mut linear_states: Vec<&mut LinearAttentionState> = prefills
            .iter_mut()
            .map(|prefill| {
                &mut prefill
                    .as_mut()
                    .expect("resident token-prefill state validated above")
                    .linear_state
            })
            .collect();
        let started = std::time::Instant::now();
        let next_tokens = self
            .decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
                &input_tokens,
                paged_cache,
                &block_tables,
                &seq_lens,
                &mut linear_states,
                Some(&row_ids),
            )
            .context("resident token-prefill Vulkan batch failed")?;
        let elapsed = started.elapsed();
        drop(linear_states);
        anyhow::ensure!(
            next_tokens.len() == batch,
            "resident token-prefill returned {} rows for batch {batch}",
            next_tokens.len()
        );

        for cancel in cancels {
            check_cancelled(Some(cancel))?;
        }
        for (prefill, cancel) in prefills.iter_mut().zip(cancels) {
            let state = prefill
                .as_mut()
                .expect("resident token-prefill state validated above");
            state.next_position = state.next_position.saturating_add(1);
            state.prefill_duration += elapsed;
            cancel.report_prefill_tokens_completed(state.processed_tokens() as u64);
        }

        let mut completed = Vec::with_capacity(batch);
        for prefill in prefills.iter() {
            let state = prefill
                .as_ref()
                .expect("resident token-prefill state validated above");
            completed.push(state.next_position == state.prompt_tokens.len());
        }

        let layers_processed = self.weights.layers.len();
        let mut progress = Vec::with_capacity(batch);
        for (idx, ((prefill, next_token), completed)) in prefills
            .iter_mut()
            .zip(next_tokens)
            .zip(completed)
            .enumerate()
        {
            let decode_state = match completed {
                true => {
                    let state = prefill
                        .take()
                        .expect("completed resident token-prefill state disappeared");
                    Some(PagedBatchedDecodeState {
                        block_table: state.block_table,
                        linear_state: state.linear_state,
                        seq_len: state.prompt_tokens.len(),
                        next_token,
                        next_token_logprob: None,
                        generated_tokens: Vec::new(),
                        step_seed: params[idx].seed,
                        capture_behavior_logprobs: false,
                        registration: None,
                        allocated_blocks: state.allocated_blocks,
                        prefill_duration: state.prefill_duration,
                        decode_duration: std::time::Duration::ZERO,
                        prompt_tokens: state.prompt_tokens,
                        block_size: state.block_size,
                        prefill_split_snapshot: None,
                        rolling_snapshot: None,
                        prefix_cache_registration_allowed: false,
                        id: state.id,
                    })
                }
                false => None,
            };
            progress.push(PagedBatchedPrefillProgress {
                tokens_scheduled: 1,
                tokens_processed: 1,
                layers_processed,
                decode_state,
            });
        }
        Ok(Some(progress))
    }

    pub fn paged_batched_decode_step(
        &self,
        states: &mut [&mut PagedBatchedDecodeState],
        params: &[SamplingParams],
        paged_cache: &PagedKvCache,
    ) -> Result<Vec<TokenId>> {
        anyhow::ensure!(
            states.len() == params.len(),
            "decode state length {} != params length {}",
            states.len(),
            params.len()
        );
        anyhow::ensure!(!states.is_empty(), "batched decode step requires rows");

        let row_count = states.len();
        self.ensure_decode_buffers(row_count)?;
        let input_tokens: Vec<TokenId> = states.iter().map(|state| state.next_token).collect();
        let block_tables: Vec<BlockTable> = states
            .iter()
            .map(|state| state.block_table.clone())
            .collect();
        let sequence_lengths: Vec<usize> = states.iter().map(|state| state.seq_len).collect();
        // Collect stable batched-state-cache fingerprint *before* the
        // `linear_states` mutable borrow below — otherwise the borrow
        // checker rejects the immutable `states.iter()`.
        let row_ids: Vec<u64> = states.iter().map(|state| state.id).collect();
        let all_greedy = params.iter().all(|p| p.temperature == 0.0);
        // (#1082) Capture row-0 sampling context BEFORE the `linear_states`
        // mutable borrow so the Vulkan native single-row decode branch below can
        // sample (temperature > 0) without re-borrowing `states[0]` while
        // `linear_states` holds it mutably. Only row 0 matters (the native branch
        // is row_count == 1) and only on Vulkan; one Option copy + one small Vec
        // clone per step.
        #[cfg(feature = "vulkan")]
        let vk_row0_sampling: Option<(Option<u64>, Vec<TokenId>)> = if row_count == 1 {
            Some((states[0].step_seed, states[0].generated_tokens.clone()))
        } else {
            None
        };
        #[cfg(feature = "vulkan")]
        let vk_batch_sampling_contexts: Option<(Vec<Option<u64>>, Vec<Vec<TokenId>>)> =
            if row_count > 1 && !all_greedy {
                Some((
                    states.iter().map(|state| state.step_seed).collect(),
                    states
                        .iter()
                        .map(|state| state.generated_tokens.clone())
                        .collect(),
                ))
            } else {
                None
            };
        let batch_sampling_contexts: Option<(Vec<Option<u64>>, Vec<Vec<TokenId>>)> = if !all_greedy
        {
            Some((
                states.iter().map(|state| state.step_seed).collect(),
                states
                    .iter()
                    .map(|state| state.generated_tokens.clone())
                    .collect(),
            ))
        } else {
            None
        };
        let mut linear_states: Vec<&mut LinearAttentionState> = states
            .iter_mut()
            .map(|state| &mut state.linear_state)
            .collect();

        let started = std::time::Instant::now();

        // Fast path: when all rows are greedy and the cache is non-FP8, route
        // compatible rows through the contiguous-batched
        // primitive. Uniform-position full-attention batches use a single
        // forward pass with fused argmax. CUDA GDN batches may also enter with
        // mixed sequence lengths because their implementation row-loops through
        // the single-row paged greedy path while preserving scheduler-visible
        // batching.
        let common_seq_len = sequence_lengths[0];
        let positions_uniform = sequence_lengths.iter().all(|&n| n == common_seq_len);
        let cache_is_fp8 = lock_paged_cache(paged_cache)?.is_fp8();
        let has_linear_layers = self.has_linear_attention_layers();
        #[cfg(any(feature = "vulkan", feature = "metal"))]
        let decode_batcher_policy =
            BackendCapabilityQueries::backend_capabilities(self.backend.as_ref()).decode_batcher;
        #[cfg(feature = "vulkan")]
        let sampled_contiguous_resident_decode_ready = decode_batcher_policy
            .use_native_sampled_contiguous_decode
            && decode_batcher_policy.sampled_contiguous_decode_requires_resident_decode
            && ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref());
        #[cfg(feature = "metal")]
        let sampled_contiguous_nonresident_decode_ready = decode_batcher_policy
            .use_native_sampled_contiguous_decode
            && !decode_batcher_policy.sampled_contiguous_decode_requires_resident_decode;
        // `model_forward_paged_decode_contiguous_batch_hidden` already handles
        // per-row positions via dyn-seqlen flash attention for full-attn
        // layers, and the GDN layers operate on the batched
        // `LinearAttentionState` regardless. The `positions_uniform` gate was
        // a leftover from before the dyn-seqlen path landed — dropping it
        // routes every bs > 1 greedy decode through the true-batched path
        // (which also batches the LM-head argmax into a single kernel
        // launch instead of `run_legacy_lm_head_sample_batch`'s per-row
        // narrow + argmax loop).
        let _ = positions_uniform;
        let hip_graph_single_row_ready = row_count == 1
            && paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                1,
                ReplayNativePrimitive::HipGraph,
            )
            && self
                .rocm_graph
                .lock()
                .map(|graph| graph.is_enabled())
                .unwrap_or(false);
        let greedy_route = greedy_batch_route(
            all_greedy,
            cache_is_fp8,
            row_count,
            hip_graph_single_row_ready,
        );
        let try_contiguous_batched = greedy_route == GreedyBatchRoute::Contiguous;

        let mut sampled: Option<Vec<TokenId>> = None;
        // Multi-batch CUDA graph fast path.
        if row_count > 1 && try_contiguous_batched && has_linear_layers {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                linear_states.iter_mut().map(|s| &mut **s).collect();
            let graph_result = {
                let mut graph_runner = self
                    .cuda_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;
                if graph_runner.is_batched_enabled() {
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    graph_runner.decode_step_paged_batched(
                        &*self.backend,
                        &input_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_table_refs,
                        &sequence_lengths,
                        &mut linear_state_refs,
                        self.active_lora.as_ref(),
                    )
                } else {
                    Ok(None)
                }
            };
            match graph_result {
                Ok(Some(tokens)) => sampled = Some(tokens),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(
                        batch = row_count,
                        error = %err,
                        "batched CUDA graph path errored; falling back to eager"
                    );
                }
            }
        }

        if sampled.is_none() && try_contiguous_batched {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let result = if has_linear_layers {
                let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                    linear_states.iter_mut().map(|s| &mut **s).collect();
                self.decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut linear_state_refs,
                    Some(&row_ids),
                )
            } else {
                let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                self.decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut no_linear_states,
                    Some(&row_ids),
                )
            };
            match result {
                Ok(tokens) => sampled = Some(tokens),
                Err(err)
                    if !decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend) =>
                {
                    return Err(err).context(decode_hot_path_fallback_disabled_context(
                        &*self.backend,
                        "contiguous-batched decode declined",
                    ));
                }
                Err(err) => {
                    tracing::debug!(
                        batch = row_count,
                        error = %err,
                        "contiguous-batched decode declined; falling back to per-row path"
                    );
                }
            }
        }

        // (#1082) Vulkan single-row decode: route the production serving path
        // (batching engine -> paged_batched_decode_step, row_count==1) through
        // native single-submit resident forwards. Greedy uses the token-only
        // resident argmax entry. Stochastic rows try the resident decode +
        // sampler tail first so they only read back one token; unsupported
        // sampler settings fall back to the older resident-logits path.
        // The row-0 sampling context (seed + generated tokens) was snapshotted
        // into `vk_row0_sampling` before the `linear_states` mutable borrow, so
        // the sampler doesn't re-borrow `states[0]` here.
        // Skipped when the contiguous-batched path above already produced tokens
        // (row > 1).
        #[cfg(feature = "vulkan")]
        if sampled.is_none()
            && row_count == 1
            && ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
        {
            let token = if params[0].temperature == 0.0 {
                let linear_state = if has_linear_layers {
                    Some(&mut *linear_states[0])
                } else {
                    None
                };
                model_forward_paged_next_token_greedy(
                    &*self.backend,
                    input_tokens[0],
                    &self.weights,
                    &self.config,
                    paged_cache,
                    &block_tables[0],
                    sequence_lengths[0],
                    linear_state,
                    self.active_lora.as_ref(),
                    None,
                )
                .context("vulkan resident single-row greedy decode forward failed")?
            } else {
                let (step_seed, generated) = vk_row0_sampling
                    .as_ref()
                    .expect("vk_row0_sampling captured for row_count == 1");
                let sample_result = if !cache_is_fp8
                    && sampled_contiguous_resident_decode_ready
                    && self.active_lora.is_none()
                {
                    let block_table_refs = [&block_tables[0]];
                    if has_linear_layers {
                        let mut linear_state_refs: [&mut LinearAttentionState; 1] =
                            [&mut *linear_states[0]];
                        self.decode_sample_paged_contiguous_batch_with_ids(
                            &input_tokens[..1],
                            paged_cache,
                            &block_table_refs,
                            &sequence_lengths[..1],
                            &mut linear_state_refs,
                            Some(&row_ids[..1]),
                            &params[..1],
                            std::slice::from_ref(step_seed),
                            std::slice::from_ref(generated),
                        )
                    } else {
                        let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                        self.decode_sample_paged_contiguous_batch_with_ids(
                            &input_tokens[..1],
                            paged_cache,
                            &block_table_refs,
                            &sequence_lengths[..1],
                            &mut no_linear_states,
                            Some(&row_ids[..1]),
                            &params[..1],
                            std::slice::from_ref(step_seed),
                            std::slice::from_ref(generated),
                        )
                    }
                } else {
                    Ok(None)
                };
                match sample_result {
                    Ok(Some(tokens)) => *tokens
                        .first()
                        .context("resident single-row sample returned no token")?,
                    Ok(None) => {
                        let logits = model_forward_paged_last_token_resident(
                            &*self.backend,
                            &input_tokens,
                            &self.weights,
                            &self.config,
                            paged_cache,
                            &block_tables[0],
                            sequence_lengths[0],
                            Some(&mut *linear_states[0]),
                            self.active_lora.as_ref(),
                            None,
                        )
                        .context("vulkan resident single-row decode forward failed")?;
                        let mut row_params = params[0].clone();
                        row_params.seed = *step_seed;
                        sample_with_full_params(&logits, &row_params, generated)?
                    }
                    Err(err) => {
                        return Err(err).context(
                            "resident single-row sample decode failed after native path selection",
                        );
                    }
                }
            };
            sampled = Some(vec![token]);
        }

        #[cfg(feature = "vulkan")]
        if sampled.is_none()
            && row_count > 1
            && !all_greedy
            && !cache_is_fp8
            && sampled_contiguous_resident_decode_ready
            && self.active_lora.is_none()
        {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let (step_seeds, generated_tokens) = vk_batch_sampling_contexts
                .as_ref()
                .expect("vk_batch_sampling_contexts captured for non-greedy row_count > 1");
            let sample_result = if has_linear_layers {
                let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                    linear_states.iter_mut().map(|s| &mut **s).collect();
                self.decode_sample_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut linear_state_refs,
                    Some(&row_ids),
                    params,
                    step_seeds,
                    generated_tokens,
                )
            } else {
                let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                self.decode_sample_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut no_linear_states,
                    Some(&row_ids),
                    params,
                    step_seeds,
                    generated_tokens,
                )
            };
            match sample_result {
                Ok(Some(tokens)) => sampled = Some(tokens),
                Ok(None) => {}
                Err(err) => {
                    return Err(err).context(
                        "resident batched sample decode failed after native path selection",
                    );
                }
            }
            if sampled.is_none() {
                let hidden_result = if has_linear_layers {
                    let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                        linear_states.iter_mut().map(|s| &mut **s).collect();
                    self.decode_hidden_paged_contiguous_batch_with_ids(
                        &input_tokens,
                        paged_cache,
                        &block_table_refs,
                        &sequence_lengths,
                        &mut linear_state_refs,
                        Some(&row_ids),
                    )
                } else {
                    let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                    self.decode_hidden_paged_contiguous_batch_with_ids(
                        &input_tokens,
                        paged_cache,
                        &block_table_refs,
                        &sequence_lengths,
                        &mut no_linear_states,
                        Some(&row_ids),
                    )
                };
                match hidden_result {
                    Ok(hidden) => {
                        let tokens = run_lm_head_sample_batch_with_contexts(
                            &*self.backend,
                            &hidden,
                            &self.weights,
                            &self.config,
                            params,
                            step_seeds,
                            generated_tokens,
                        )
                        .context("sample Vulkan resident multi-row hidden batch")?;
                        sampled = Some(tokens);
                    }
                    Err(err)
                        if !decode_hot_path_generic_fallback_enabled_for_backend(
                            &*self.backend,
                        ) =>
                    {
                        return Err(err).context(decode_hot_path_fallback_disabled_context(
                            &*self.backend,
                            "resident batched hidden decode declined",
                        ));
                    }
                    Err(err) => {
                        tracing::debug!(
                            batch = row_count,
                            error = %err,
                            "resident batched hidden decode declined; falling back to generic hidden path"
                        );
                    }
                }
            }
        }

        #[cfg(feature = "metal")]
        if sampled.is_none()
            && !all_greedy
            && !cache_is_fp8
            && sampled_contiguous_nonresident_decode_ready
            && self.active_lora.is_none()
        {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let (step_seeds, generated_tokens) = batch_sampling_contexts
                .as_ref()
                .expect("batch_sampling_contexts captured for non-greedy Metal decode");
            let sample_result = if has_linear_layers {
                let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                    linear_states.iter_mut().map(|s| &mut **s).collect();
                self.decode_sample_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut linear_state_refs,
                    Some(&row_ids),
                    params,
                    step_seeds,
                    generated_tokens,
                )
            } else {
                let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                self.decode_sample_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut no_linear_states,
                    Some(&row_ids),
                    params,
                    step_seeds,
                    generated_tokens,
                )
            };
            match sample_result {
                Ok(Some(tokens)) => sampled = Some(tokens),
                Ok(None) => {}
                Err(err)
                    if !decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend) =>
                {
                    return Err(err).context(decode_hot_path_fallback_disabled_context(
                        &*self.backend,
                        "Metal sampled decode declined",
                    ));
                }
                Err(err) => {
                    tracing::warn!(
                        batch = row_count,
                        error = %err,
                        "Metal sampled decode declined; falling back to eager hidden sample path"
                    );
                }
            }
            if sampled.is_none()
                && !decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend)
            {
                anyhow::bail!(
                    "{}",
                    decode_hot_path_fallback_disabled_context(
                        &*self.backend,
                        "Metal sampled decode did not produce tokens"
                    )
                );
            }
        }

        // ROCm sampled serving batches need a native decode path even when the
        // HIP-graph bs=1 optimization is disabled or inapplicable. Decode the
        // hidden rows through the contiguous batched ROCm path, then sample from
        // those rows outside the transformer hot path. This keeps
        // NativeRequired from silently depending on the generic fallback when
        // concurrent sampled streams coalesce into row_count > 1.
        if sampled.is_none()
            && !all_greedy
            && matches!(
                BackendIdentity::runtime_device(self.backend.as_ref()),
                kiln_tensor::Device::Rocm(_)
            )
            && (row_count > 1
                || !paged_decode_replay_primitive_enabled(
                    self.backend.as_ref(),
                    &self.config,
                    1,
                    ReplayNativePrimitive::HipGraph,
                )
                || !self
                    .rocm_graph
                    .lock()
                    .map(|g| g.is_enabled())
                    .unwrap_or(false))
        {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let (step_seeds, generated_tokens) = batch_sampling_contexts
                .as_ref()
                .context("missing sampling contexts for ROCm sampled batched decode")?;
            let hidden_result = if has_linear_layers {
                let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                    linear_states.iter_mut().map(|s| &mut **s).collect();
                self.decode_hidden_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut linear_state_refs,
                    Some(&row_ids),
                )
            } else {
                let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                self.decode_hidden_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut no_linear_states,
                    Some(&row_ids),
                )
            };
            let hidden = hidden_result.context("ROCm sampled batched hidden decode failed")?;
            sampled = Some(
                run_lm_head_sample_batch_with_contexts(
                    &*self.backend,
                    &hidden,
                    &self.weights,
                    &self.config,
                    params,
                    step_seeds,
                    generated_tokens,
                )
                .context("sample ROCm hidden batch")?,
            );
        }

        // R.9: ROCm HIP-graph single-row decode for the batched/batching-engine
        // path. Gated by the ROCm runner, so when disabled `sampled` stays as
        // set above and the cuda/eager block below runs unchanged. Sampled rows
        // use the hidden-only graph path and keep the stochastic lm-head sampler
        // outside the captured graph.
        if sampled.is_none() && hip_graph_single_row_ready {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if params[0].temperature == 0.0 {
                let token = self
                    .rocm_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?
                    .decode_step_paged_greedy(
                        &*self.backend,
                        input_tokens[0],
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_tables[0],
                        sequence_lengths[0],
                        &mut *linear_states[0],
                        self.active_lora.as_ref(),
                        row_ids[0],
                    )
                    .context("batched decode ROCm graph greedy row failed")?;
                sampled = Some(vec![token]);
            } else {
                let (step_seeds, generated_tokens) = batch_sampling_contexts
                    .as_ref()
                    .context("missing row-0 sampling context for ROCm graph sampled decode")?;
                let hidden = self
                    .rocm_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?
                    .decode_step_paged_hidden(
                        &*self.backend,
                        input_tokens[0],
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_tables[0],
                        sequence_lengths[0],
                        &mut *linear_states[0],
                        self.active_lora.as_ref(),
                        row_ids[0],
                    )
                    .context("batched decode ROCm graph hidden row failed")?;
                let token = if let Some(token) = lm_head_sample_backend_decode_if(
                    Some(&*self.backend),
                    &hidden,
                    &self.weights,
                    &self.config,
                    &params[0],
                    step_seeds[0],
                    &generated_tokens[0],
                )
                .context("fused ROCm graph linear_decode_sample failed")?
                {
                    token
                } else {
                    run_lm_head_sample_batch_with_contexts(
                        &*self.backend,
                        &hidden,
                        &self.weights,
                        &self.config,
                        params,
                        step_seeds,
                        generated_tokens,
                    )?[0]
                };
                sampled = Some(vec![token]);
            }
        }

        let sampled = if let Some(tokens) = sampled {
            tokens
        } else {
            if !decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend) {
                anyhow::bail!(
                    "{}",
                    decode_hot_path_fallback_disabled_context(
                        &*self.backend,
                        "no native batched decode path produced tokens"
                    )
                );
            }
            let pc_guard = lock_paged_cache(paged_cache)?;
            let mut graph_runner = self
                .cuda_graph
                .lock()
                .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;
            if graph_runner.is_enabled() && row_count == 1 {
                let row = graph_runner
                    .decode_step_paged(
                        &*self.backend,
                        input_tokens[0],
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_tables[0],
                        sequence_lengths[0],
                        &mut *linear_states[0],
                        self.active_lora.as_ref(),
                        Some(row_ids[0]),
                    )
                    .context("batched decode CUDA graph row failed")?;
                // #1082: `decode_step_paged` now returns a kt `Tensor` — feed it
                // straight to the kt-typed samplers, no candle->kt bridge.
                let token = if params[0].temperature == 0.0 {
                    greedy_sample(&row)?
                } else {
                    let mut row_params = params[0].clone();
                    row_params.seed = states[0].step_seed;
                    sample_with_full_params(&row, &row_params, &states[0].generated_tokens)?
                };
                vec![token]
            } else {
                let hidden = model_forward_paged_batched_decode_hidden(
                    &*self.backend,
                    &input_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_tables,
                    &sequence_lengths,
                    &mut linear_states,
                    self.active_lora.as_ref(),
                )
                .context("batched decode forward pass failed")?;

                // Single-row non-greedy: try the backend-fused fused
                // sample path first. It does lm_head + penalty + top-k
                // + softmax + min_p + top_p + categorical entirely
                // on-device and reads back only the 4-byte token.
                // Falls back to the legacy "lm_head + host sample"
                // flow when the backend declines.
                if row_count == 1 && params[0].temperature > 0.0 {
                    let row_hidden = hidden
                        .narrow(0, 0, 1)
                        .context("batched decode hidden row 0")?;
                    if let Some(token) = lm_head_sample_backend_decode_if(
                        Some(&*self.backend),
                        &row_hidden,
                        &self.weights,
                        &self.config,
                        &params[0],
                        states[0].step_seed,
                        &states[0].generated_tokens,
                    )
                    .context("fused linear_decode_sample failed")?
                    {
                        vec![token]
                    } else {
                        // Backend declined (top_k > kernel max, dtype
                        // mismatch, etc.) — fall through to the legacy
                        // lm_head + host sampler.
                        run_legacy_lm_head_sample_batch(
                            &*self.backend,
                            &hidden,
                            &self.weights,
                            &self.config,
                            params,
                            states,
                        )?
                    }
                } else {
                    let step_seeds: Vec<Option<u64>> =
                        states.iter().map(|state| state.step_seed).collect();
                    let generated_tokens: Vec<Vec<TokenId>> = states
                        .iter()
                        .map(|state| state.generated_tokens.clone())
                        .collect();
                    run_lm_head_sample_batch_with_contexts(
                        &*self.backend,
                        &hidden,
                        &self.weights,
                        &self.config,
                        params,
                        &step_seeds,
                        &generated_tokens,
                    )?
                }
            }
        };
        let decode_duration = started.elapsed();

        complete_paged_batched_decode_step(
            self.backend.as_ref(),
            &self
                .batched_state_cache_counters
                .resident_prefix_snapshot_suppression_count,
            states,
            decode_duration,
        );

        Ok(sampled)
    }

    /// Decode one step while retaining each selected token's exact effective
    /// behavior-policy log-probability. Trace mode deliberately performs the
    /// LM head and host-visible post-filter sampler instead of accepting a
    /// token-only fused result whose probability cannot be reconstructed.
    pub fn paged_batched_decode_step_with_behavior_logprobs(
        &self,
        states: &mut [&mut PagedBatchedDecodeState],
        params: &[SamplingParams],
        paged_cache: &PagedKvCache,
    ) -> Result<Vec<SampledToken>> {
        anyhow::ensure!(
            states.len() == params.len(),
            "decode state length {} != params length {}",
            states.len(),
            params.len()
        );
        anyhow::ensure!(!states.is_empty(), "batched decode step requires rows");
        anyhow::ensure!(
            states.iter().all(|state| state.capture_behavior_logprobs),
            "behavior-logprob decode received a row that did not opt into capture"
        );

        let row_count = states.len();
        self.ensure_decode_buffers(row_count)?;
        let input_tokens: Vec<TokenId> = states.iter().map(|state| state.next_token).collect();
        let block_tables: Vec<BlockTable> = states
            .iter()
            .map(|state| state.block_table.clone())
            .collect();
        let sequence_lengths: Vec<usize> = states.iter().map(|state| state.seq_len).collect();
        let step_seeds: Vec<Option<u64>> = states.iter().map(|state| state.step_seed).collect();
        let row_ids: Vec<u64> = states.iter().map(|state| state.id).collect();
        let generated_tokens: Vec<Vec<TokenId>> = states
            .iter()
            .map(|state| state.generated_tokens.clone())
            .collect();
        let mut linear_states: Vec<&mut LinearAttentionState> = states
            .iter_mut()
            .map(|state| &mut state.linear_state)
            .collect();

        let started = std::time::Instant::now();
        let hip_graph_single_row_ready = row_count == 1
            && paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                1,
                ReplayNativePrimitive::HipGraph,
            )
            && self
                .rocm_graph
                .lock()
                .map(|graph| graph.is_enabled())
                .unwrap_or(false);
        let hidden = if hip_graph_single_row_ready {
            let pc_guard = lock_paged_cache(paged_cache)?;
            self.rocm_graph
                .lock()
                .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?
                .decode_step_paged_hidden(
                    &*self.backend,
                    input_tokens[0],
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_tables[0],
                    sequence_lengths[0],
                    &mut *linear_states[0],
                    self.active_lora.as_ref(),
                    row_ids[0],
                )
                .context("behavior-logprob ROCm graph hidden row failed")?
        } else {
            let pc_guard = lock_paged_cache(paged_cache)?;
            model_forward_paged_batched_decode_hidden(
                &*self.backend,
                &input_tokens,
                &self.weights,
                &self.config,
                pc_guard,
                &block_tables,
                &sequence_lengths,
                &mut linear_states,
                self.active_lora.as_ref(),
            )
            .context("behavior-logprob batched decode forward pass failed")?
        };
        drop(linear_states);
        let logits = crate::forward::model_forward_head_backend_decode_if(
            Some(&*self.backend),
            &hidden,
            &self.weights,
            &self.config,
        )
        .context("behavior-logprob batched decode lm head")?;
        let mut sampled = Vec::with_capacity(row_count);
        for (idx, param) in params.iter().enumerate() {
            let row = logits
                .narrow(0, idx, 1)
                .with_context(|| format!("behavior-logprob batched decode lm head row {idx}"))?;
            sampled.push(sample_step_with_logprob(
                &row,
                param,
                step_seeds[idx],
                &generated_tokens[idx],
            )?);
        }

        let decode_duration = started.elapsed();
        complete_paged_batched_decode_step(
            self.backend.as_ref(),
            &self
                .batched_state_cache_counters
                .resident_prefix_snapshot_suppression_count,
            states,
            decode_duration,
        );
        Ok(sampled)
    }

    pub fn finish_paged_batched_decode(
        &self,
        state: PagedBatchedDecodeState,
        finish_reason: FinishReason,
    ) -> Result<PrefixCachedGenerationOutput> {
        // This is the common completion boundary for normal, cancelled,
        // disconnected, and failed batching-engine requests. Release the unique
        // decode-row owner before token decoding or other fallible finish work so
        // stale graphs and timelines cannot accumulate in a long-running server.
        match self.rocm_graph.lock() {
            Ok(mut graph) => graph.release_decode_row(state.id),
            Err(poisoned) => {
                tracing::warn!(
                    row_id = state.id,
                    "recovering poisoned ROCm graph lock to release finished decode row"
                );
                poisoned.into_inner().release_decode_row(state.id);
            }
        }
        self.release_batched_decode_state(state.id, &state.linear_state);

        let PagedBatchedDecodeState {
            block_table,
            generated_tokens,
            registration,
            allocated_blocks,
            prefill_duration,
            decode_duration,
            prompt_tokens,
            block_size,
            prefill_split_snapshot,
            rolling_snapshot,
            prefix_cache_registration_allowed,
            ..
        } = state;

        let text = self
            .tokenizer
            .decode(&generated_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        let mut extra_registrations = Vec::new();
        if prefix_cache_registration_allowed {
            if let Some(reg) = build_extended_registration(
                &prompt_tokens,
                &generated_tokens,
                &block_table,
                block_size,
                prefill_split_snapshot,
            ) {
                extra_registrations.push(reg);
            }
            if let Some(reg) = build_extended_registration(
                &prompt_tokens,
                &generated_tokens,
                &block_table,
                block_size,
                rolling_snapshot,
            ) {
                extra_registrations.push(reg);
            }
        }

        Ok(PrefixCachedGenerationOutput {
            output: GenerationOutput {
                text,
                token_ids: generated_tokens,
                finish_reason,
            },
            registration,
            extra_registrations,
            allocated_blocks,
            prefill_duration,
            decode_duration,
        })
    }

    fn completed_prompt_registration(
        &self,
        prompt_tokens: &[TokenId],
        block_table: &BlockTable,
        linear_state: &LinearAttentionState,
        block_size: usize,
        next_token: Option<PagedPrefixNextToken>,
    ) -> Result<Option<PagedPrefixRegistration>> {
        if prompt_tokens.is_empty() || block_size == 0 || prompt_tokens.len() % block_size != 0 {
            return Ok(None);
        }
        let num_prompt_blocks = prompt_tokens.len() / block_size;
        if num_prompt_blocks == 0 || block_table.blocks.len() < num_prompt_blocks {
            return Ok(None);
        }
        let Some(linear_state) =
            self.authoritative_prefix_snapshot(linear_state, "whole-prompt", prompt_tokens.len())?
        else {
            return Ok(None);
        };
        Ok(Some(PagedPrefixRegistration {
            prompt_tokens: prompt_tokens.to_vec(),
            block_ids: block_table.blocks[..num_prompt_blocks].to_vec(),
            linear_state,
            next_token,
        }))
    }

    fn authoritative_prefix_snapshot(
        &self,
        linear_state: &LinearAttentionState,
        snapshot_kind: &'static str,
        position: usize,
    ) -> Result<Option<LinearAttentionState>> {
        capture_authoritative_prefix_snapshot(
            self.backend.as_ref(),
            &self
                .batched_state_cache_counters
                .resident_prefix_snapshot_suppression_count,
            linear_state,
            snapshot_kind,
            position,
        )
    }

    fn decode_from_prefill_logits(
        &self,
        // (#1082) kt-native logits — sampler (greedy_sample/sample_step) is kt.
        logits: kiln_tensor::Tensor,
        seq_len: usize,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        linear_state: &mut LinearAttentionState,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let step_seed = params.seed;

        let sampled = if params.is_effectively_greedy() {
            greedy_sample(&logits)
        } else {
            sample_step(&logits, params, step_seed, &[])
        };
        let next_token = match sampled {
            Ok(token) => token,
            Err(sample_error) => {
                if let Err(sync_error) =
                    self.synchronize_external_yield("direct prefill sampling failure")
                {
                    quarantine_linear_attention_state(linear_state);
                    std::mem::forget(logits);
                    return Err(sync_error.context(format!(
                        "prefill sampling also failed before synchronization: {sample_error:#}"
                    )));
                }
                return Err(sample_error);
            }
        };
        let result = self.decode_from_prefill_token(
            next_token,
            seq_len,
            params,
            paged_cache,
            block_table,
            linear_state,
            step_seed,
            cancel,
        );
        if result.is_err() && self.backend_health.snapshot().quarantined {
            std::mem::forget(logits);
        }
        result
    }

    fn decode_from_prefill_token(
        &self,
        mut next_token: TokenId,
        mut seq_len: usize,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        linear_state: &mut LinearAttentionState,
        mut step_seed: Option<u64>,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let rocm_owner = RocmDecodeOwnerLease::new(&self.rocm_graph, &self.backend_health);
        let result = (|| -> Result<GenerationOutput> {
            let mut generated_tokens: Vec<TokenId> = Vec::new();
            for _step in 0..params.max_tokens {
                check_cancelled(cancel)?;
                if let Some(s) = step_seed.as_mut() {
                    *s = s.wrapping_add(1);
                }

                next_token = params.apply_thinking_budget(&generated_tokens, next_token);
                if self.should_stop_on_eos(params, next_token) {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                    });
                }

                generated_tokens.push(next_token);

                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(GenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }

                let skip_gdn_state_readback = skip_final_gdn_state_readback_enabled()
                    && generated_tokens.len() + 1 >= params.max_tokens;
                next_token = self.decode_next_token_paged_interleaved(
                    params,
                    next_token,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    step_seed,
                    &generated_tokens,
                    rocm_owner.row_id(),
                    skip_gdn_state_readback,
                )?;
                seq_len += 1;
            }

            Ok(GenerationOutput {
                text: String::new(),
                token_ids: generated_tokens,
                finish_reason: FinishReason::MaxTokens,
            })
        })();
        match self.synchronize_external_yield("direct paged decode completion") {
            Ok(()) => result,
            Err(sync_err) => {
                quarantine_linear_attention_state(linear_state);
                std::mem::forget(result);
                Err(sync_err)
            }
        }
    }

    /// Decode one greedy token for multiple compatible paged requests in one
    /// model-forward call.
    ///
    /// This is the scheduler admission primitive for true decode batching: the
    /// caller still owns request readiness, stop handling, and output routing,
    /// while this method owns the row assembly needed to call
    /// `model_forward_paged_decode_contiguous_batch_greedy`.
    ///
    /// Current constraints intentionally mirror the lower-level helper:
    /// non-empty rows, one token per row, one `BlockTable` per row, non-FP8
    /// cache, backend-compatible paged attention windows, and shared base
    /// model/LoRA state for every row. Qwen-style GDN models must pass one
    /// mutable one-row `LinearAttentionState` per row; the method assembles
    /// those into batch state before the forward pass and scatters the updated
    /// rows back afterward. A resident backend must preserve supplied stable
    /// row IDs even for a one-row cohort so backend-private KV seed ownership
    /// and the retained batched recurrent-state cache remain request-scoped.
    pub fn decode_next_tokens_paged_contiguous_batch_greedy(
        &self,
        input_tokens: &[TokenId],
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
    ) -> Result<Vec<TokenId>> {
        // Stable-id-less call site (e.g. tests). Skip the batched-state
        // cache.
        self.decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
            input_tokens,
            paged_cache,
            block_tables,
            seq_lens,
            linear_states,
            None,
        )
    }

    pub fn decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
        &self,
        input_tokens: &[TokenId],
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
        row_ids: Option<&[u64]>,
    ) -> Result<Vec<TokenId>> {
        let _resident_scope = GdnRecurrentResidentStateScope::new(&*self.backend);
        let batch = input_tokens.len();
        let profile_stages = profile_decode_batcher_stages_enabled();
        let total_start = profile_stages.then(std::time::Instant::now);
        anyhow::ensure!(batch > 0, "batched decode requires at least one row");
        anyhow::ensure!(
            block_tables.len() == batch && seq_lens.len() == batch,
            "batched decode metadata length mismatch"
        );
        if let Some(ids) = row_ids {
            anyhow::ensure!(
                ids.len() == batch,
                "batched decode row-id count mismatch ({} vs {batch})",
                ids.len()
            );
        }

        let has_linear_layers = self.has_linear_attention_layers();
        if has_linear_layers {
            anyhow::ensure!(
                linear_states.len() == batch,
                "batched decode requires one LinearAttentionState per row"
            );
        } else {
            anyhow::ensure!(
                linear_states.is_empty(),
                "full-attention-only batched decode does not accept linear states"
            );
        }

        let resident_decode_supported =
            ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref());
        if should_use_unidentified_single_row_greedy_route(
            batch,
            row_ids.is_some(),
            resident_decode_supported,
        ) {
            let stage_start = profile_stages.then(std::time::Instant::now);
            let pc_guard = lock_paged_cache(paged_cache)?;
            #[cfg(feature = "metal")]
            let token = {
                let mut token = None;
                if paged_decode_replay_primitive_enabled(
                    self.backend.as_ref(),
                    &self.config,
                    1,
                    ReplayNativePrimitive::MetalIcb,
                ) && self.active_lora.is_none()
                {
                    let graph_tokens = {
                        let one_tokens = [input_tokens[0]];
                        let one_block_tables = [block_tables[0]];
                        let one_seq_lens = [seq_lens[0]];
                        let linear_state_for_graph = if has_linear_layers {
                            Some(&mut *linear_states[0])
                        } else {
                            None
                        };
                        let mut runner = self.metal_graph.lock().map_err(|e| {
                            anyhow::anyhow!("failed to lock Metal graph runner: {e}")
                        })?;
                        if runner.is_enabled() {
                            runner.decode_step_paged_greedy_batch(
                                &*self.backend,
                                &one_tokens,
                                &self.weights,
                                &self.config,
                                pc_guard,
                                &one_block_tables,
                                &one_seq_lens,
                                linear_state_for_graph,
                                self.active_lora.as_ref(),
                            )?
                        } else {
                            None
                        }
                    };
                    if let Some(graph_tokens) = graph_tokens {
                        anyhow::ensure!(
                            graph_tokens.len() == 1,
                            "Metal graph single-row greedy returned {} tokens",
                            graph_tokens.len()
                        );
                        token = graph_tokens.first().copied();
                    }
                }
                token
            };
            #[cfg(not(feature = "metal"))]
            let token = None;
            let token = (match token {
                Some(token) => Ok(token),
                None => {
                    let linear_state = if has_linear_layers {
                        Some(&mut *linear_states[0])
                    } else {
                        None
                    };
                    model_forward_paged_next_token_greedy(
                        &*self.backend,
                        input_tokens[0],
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_tables[0],
                        seq_lens[0],
                        linear_state,
                        self.active_lora.as_ref(),
                        None,
                    )
                }
            })
            .context("single-row greedy decode forward pass (paged) failed")?;
            finish_decode_batcher_stage_profile("single_forward", batch, stage_start);
            finish_decode_batcher_stage_profile("decode_total", batch, total_start);
            return Ok(vec![token]);
        }

        // #1082 PERF + CRASHER FIX (per-row contiguity partition).
        // The vendored FA2 split-KV paged-decode kernel reads each kBlockN-token
        // K/V tile as ONE physically-contiguous run of pages from a single
        // block_table entry (it never consults the intervening entries — see
        // flash_fwd_kernel.h block_table_idx stride). A fragmented BlockManager
        // free list (concurrent finish->free->re-admit) hands the kernel
        // NON-ADJACENT pages -> CUDA_ERROR_ILLEGAL_ADDRESS / wrong KV. #1445
        // guarded this by forcing the WHOLE batch onto the per-row loop when ANY
        // row was fragmented — but under concurrency the detector fires on the
        // whole batch nearly every step, serializing bs=N into N single-row
        // forwards (the n=64 cliff: 366s p50, 11 tok/s). PARTITION instead:
        // row-loop ONLY the genuinely-fragmented rows and batch the contiguous
        // majority through the fast path. Crash-safe (no non-adjacent pages ever
        // reach the kernel) and a strict superset of #1445's correctness.
        let decode_policy =
            BackendCapabilityQueries::backend_capabilities(self.backend.as_ref()).decode_batcher;
        if decode_policy.partition_noncontiguous_gdn_kv_tiles && has_linear_layers {
            let row_loop_all = gdn_batched_decode_row_loop_debug_enabled();
            let block_size = paged_cache.block_size();
            let noncontig: Vec<bool> = (0..batch)
                .map(|row| {
                    row_loop_all
                        || row_has_noncontiguous_kv_tiles(
                            block_tables[row].blocks.as_slice(),
                            seq_lens[row],
                            block_size,
                        )
                })
                .collect();
            let n_noncontig = noncontig.iter().filter(|&&x| x).count();

            if n_noncontig == batch {
                // Every row fragmented (or debug row-loop-all): the original
                // contiguity-safe per-row loop, unchanged.
                let stage_start = profile_stages.then(std::time::Instant::now);
                let mut tokens = Vec::with_capacity(batch);
                for row in 0..batch {
                    let linear_state =
                        Some(&mut **linear_states.get_mut(row).with_context(|| {
                            format!("missing linear state for CUDA row-loop decode row {row}")
                        })?);
                    let token = {
                        let pc_guard = lock_paged_cache(paged_cache)?;
                        model_forward_paged_next_token_greedy(
                            &*self.backend,
                            input_tokens[row],
                            &self.weights,
                            &self.config,
                            pc_guard,
                            block_tables[row],
                            seq_lens[row],
                            linear_state,
                            self.active_lora.as_ref(),
                            None,
                        )
                        .with_context(|| {
                            format!(
                                "CUDA row-loop greedy decode row {row} forward pass (paged) failed"
                            )
                        })?
                    };
                    tokens.push(token);
                }
                finish_decode_batcher_stage_profile(
                    "cuda_gdn_row_loop_forward",
                    batch,
                    stage_start,
                );
                finish_decode_batcher_stage_profile("decode_total", batch, total_start);
                return Ok(tokens);
            } else if n_noncontig > 0 {
                // MIXED: row-loop only the fragmented rows; batch the contiguous
                // majority through the fast path (recurse on the all-contiguous
                // subset, which falls straight through to it). Scatter back to
                // input order. This is what keeps the fast path alive at n=64 when
                // only a handful of rows hold freshly-recycled pages.
                let stage_start = profile_stages.then(std::time::Instant::now);
                let mut out = vec![0u32; batch];
                // Disjoint partition of the &mut linear states in one pass.
                let mut contig_idx: Vec<usize> = Vec::new();
                let mut contig_states: Vec<&mut LinearAttentionState> = Vec::new();
                let mut noncontig_rows: Vec<(usize, &mut LinearAttentionState)> = Vec::new();
                for (row, ls) in linear_states.iter_mut().enumerate() {
                    if noncontig[row] {
                        noncontig_rows.push((row, &mut **ls));
                    } else {
                        contig_idx.push(row);
                        contig_states.push(&mut **ls);
                    }
                }
                // Fragmented rows: contiguity-safe single-row path.
                for (row, ls) in noncontig_rows.iter_mut() {
                    let token = {
                        let pc_guard = lock_paged_cache(paged_cache)?;
                        model_forward_paged_next_token_greedy(
                            &*self.backend,
                            input_tokens[*row],
                            &self.weights,
                            &self.config,
                            pc_guard,
                            block_tables[*row],
                            seq_lens[*row],
                            Some(&mut **ls),
                            self.active_lora.as_ref(),
                            None,
                        )
                        .with_context(|| {
                            format!(
                                "CUDA mixed-batch fragmented-row greedy decode row {row} failed"
                            )
                        })?
                    };
                    out[*row] = token;
                }
                // Contiguous majority: one fast batched forward via recursion (the
                // all-contiguous subset hits n_noncontig==0 and falls through).
                let contig_tokens: Vec<TokenId> =
                    contig_idx.iter().map(|&i| input_tokens[i]).collect();
                let contig_bts: Vec<&BlockTable> =
                    contig_idx.iter().map(|&i| block_tables[i]).collect();
                let contig_seqlens: Vec<usize> = contig_idx.iter().map(|&i| seq_lens[i]).collect();
                let contig_row_ids: Option<Vec<u64>> =
                    row_ids.map(|r| contig_idx.iter().map(|&i| r[i]).collect());
                let contig_out = self
                    .decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
                        &contig_tokens,
                        paged_cache,
                        &contig_bts,
                        &contig_seqlens,
                        &mut contig_states,
                        contig_row_ids.as_deref(),
                    )
                    .context("CUDA mixed-batch contiguous-subset batched decode failed")?;
                for (k, &row) in contig_idx.iter().enumerate() {
                    out[row] = contig_out[k];
                }
                finish_decode_batcher_stage_profile(
                    "cuda_gdn_partition_forward",
                    batch,
                    stage_start,
                );
                finish_decode_batcher_stage_profile("decode_total", batch, total_start);
                return Ok(out);
            }
            // n_noncontig == 0: all rows contiguous -> fall through to fast path.
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        if has_linear_layers {
            if ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
                && ReplayBackend::runtime_decode_resident_pool_ready(
                    self.backend.as_ref(),
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    64,
                )
            {
                for state in linear_states.iter() {
                    state.ensure_gdn_state_resident_kt(&*self.backend)?;
                }
            }
            let any_resident = linear_states
                .iter()
                .any(|state| state.has_any_gdn_state_resident_kt(&*self.backend));
            let all_resident = any_resident
                && linear_states
                    .iter()
                    .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
            if any_resident && !all_resident {
                anyhow::bail!(
                    "mixed kt-resident GDN state rows are not supported for batched decode"
                );
            }
        }

        let all_rows_resident = has_linear_layers
            && linear_states
                .iter()
                .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
        let (mut batch_state, batched_state_cache_hit) = if has_linear_layers {
            self.prepare_batched_linear_state(linear_states, all_rows_resident, row_ids)?
        } else {
            (
                ResidentBatchedStateLease::new(
                    None,
                    self.backend.as_ref(),
                    &self.batched_state_cache_counters,
                ),
                false,
            )
        };
        if batched_state_cache_hit {
            finish_decode_batcher_stage_profile(
                "batch_state_assemble_cache_hit",
                batch,
                stage_start,
            );
        } else {
            finish_decode_batcher_stage_profile("batch_state_assemble", batch, stage_start);
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        let tokens = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let graph_tokens = if paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                batch,
                ReplayNativePrimitive::MetalIcb,
            ) {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                if runner.is_enabled() {
                    runner.decode_step_paged_greedy_batch(
                        &*self.backend,
                        input_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_tables,
                        seq_lens,
                        batch_state.as_mut(),
                        self.active_lora.as_ref(),
                    )?
                } else {
                    None
                }
            } else {
                None
            };
            match graph_tokens {
                Some(tokens) => tokens,
                None => model_forward_paged_decode_contiguous_batch_greedy_with_ids(
                    &*self.backend,
                    input_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_tables,
                    seq_lens,
                    batch_state.as_mut(),
                    self.active_lora.as_ref(),
                    row_ids,
                )
                .context("batched greedy decode forward pass (paged) failed")?,
            }
        };
        finish_decode_batcher_stage_profile("batched_forward", batch, stage_start);

        if let Some(state) = batch_state.as_ref() {
            let stage_start = profile_stages.then(std::time::Instant::now);
            if fast_batched_linear_state_scatter_enabled() {
                if !state.scatter_gdn_state_resident_batch_rows_kt(&*self.backend, linear_states)? {
                    state.scatter_batch_rows_replace(linear_states)?;
                }
                finish_decode_batcher_stage_profile(
                    "batch_state_scatter_replace",
                    batch,
                    stage_start,
                );
            } else {
                state.scatter_batch_rows(linear_states)?;
                finish_decode_batcher_stage_profile("batch_state_scatter_copy", batch, stage_start);
            }
        }
        // Park the (now updated) batched state back in the cache so the
        // next decode step can reuse its resident allocation. Exact row IDs
        // skip assembly; a different Vulkan-resident row set refreshes the
        // retained maximum-capacity buffers in place. The per-row states are
        // byte-for-byte equivalent to the logical batch prefix right now
        // because we just scattered. We only cache when the caller supplied
        // IDs; otherwise no later call can establish that equivalence.
        if let (Some(state), Some(ids)) = (batch_state.take_for_cache(), row_ids) {
            self.park_batched_state(state, ids);
        }
        finish_decode_batcher_stage_profile("decode_total", batch, total_start);

        Ok(tokens)
    }

    /// Decode multiple compatible paged requests through the transformer stack,
    /// returning final hidden states for caller-owned sampling.
    ///
    /// This mirrors the greedy continuous-batch state assembly/scatter path, but
    /// stops before the LM-head argmax so mixed sampling parameters can still be
    /// handled by the existing sampler.
    #[allow(clippy::too_many_arguments)]
    fn decode_sample_paged_contiguous_batch_with_ids(
        &self,
        input_tokens: &[TokenId],
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
        row_ids: Option<&[u64]>,
        params: &[SamplingParams],
        step_seeds: &[Option<u64>],
        generated_tokens: &[Vec<TokenId>],
    ) -> Result<Option<Vec<TokenId>>> {
        let top_k_values: Vec<u32> = params.iter().map(|param| param.top_k).collect();
        let temperature_values: Vec<f32> = params.iter().map(|param| param.temperature).collect();
        if !SamplingBackend::runtime_supports_linear_decode_sample_batch(
            self.backend.as_ref(),
            &top_k_values,
            &temperature_values,
        ) {
            return Ok(None);
        }

        let _resident_scope = GdnRecurrentResidentStateScope::new(&*self.backend);
        let batch = input_tokens.len();
        let profile_stages = profile_decode_batcher_stages_enabled();
        let total_start = profile_stages.then(std::time::Instant::now);
        anyhow::ensure!(batch > 0, "batched sample decode requires at least one row");
        anyhow::ensure!(
            block_tables.len() == batch
                && seq_lens.len() == batch
                && params.len() == batch
                && step_seeds.len() == batch
                && generated_tokens.len() == batch,
            "batched sample decode metadata length mismatch"
        );

        let mut repetition_values = Vec::with_capacity(batch);
        let mut presence_values = Vec::with_capacity(batch);
        let mut frequency_values = Vec::with_capacity(batch);
        let mut top_p_values = Vec::with_capacity(batch);
        let mut min_p_values = Vec::with_capacity(batch);
        let mut seed_values = Vec::with_capacity(batch);
        let mut history_rows = Vec::new();
        let mut history_indices = Vec::new();
        let mut history_counts = Vec::new();
        for (row_idx, ((param, step_seed), history)) in params
            .iter()
            .zip(step_seeds.iter())
            .zip(generated_tokens.iter())
            .enumerate()
        {
            repetition_values.push(param.repetition_penalty);
            presence_values.push(param.presence_penalty);
            frequency_values.push(param.frequency_penalty);
            top_p_values.push(param.top_p);
            min_p_values.push(param.min_p);
            seed_values.push(sample_seed_for_batch_row(*step_seed, history));
            if param.is_effectively_greedy()
                || param.token_penalties_are_no_op()
                || history.is_empty()
            {
                continue;
            }
            let (indices, counts) = unique_history_counts_for_batch_sample(history);
            for (idx, count) in indices.into_iter().zip(counts.into_iter()) {
                history_rows.push(row_idx as u32);
                history_indices.push(idx);
                history_counts.push(count);
            }
        }

        let has_linear_layers = self.has_linear_attention_layers();
        if has_linear_layers {
            anyhow::ensure!(
                linear_states.len() == batch,
                "batched sample decode requires one LinearAttentionState per row"
            );
        } else {
            anyhow::ensure!(
                linear_states.is_empty(),
                "full-attention-only batched sample decode does not accept linear states"
            );
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        if has_linear_layers {
            if ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
                && ReplayBackend::runtime_decode_resident_pool_ready(
                    self.backend.as_ref(),
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    64,
                )
            {
                for state in linear_states.iter() {
                    state.ensure_gdn_state_resident_kt(&*self.backend)?;
                }
            }
            let any_resident = linear_states
                .iter()
                .any(|state| state.has_any_gdn_state_resident_kt(&*self.backend));
            let all_resident = any_resident
                && linear_states
                    .iter()
                    .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
            if any_resident && !all_resident {
                anyhow::bail!(
                    "mixed kt-resident GDN state rows are not supported for batched sample decode"
                );
            }
        }

        let single_row_direct_state = has_linear_layers && batch == 1;
        let all_rows_resident = has_linear_layers
            && linear_states
                .iter()
                .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
        let (mut batch_state, batched_state_cache_hit) =
            if has_linear_layers && !single_row_direct_state {
                self.prepare_batched_linear_state(linear_states, all_rows_resident, row_ids)?
            } else {
                (
                    ResidentBatchedStateLease::new(
                        None,
                        self.backend.as_ref(),
                        &self.batched_state_cache_counters,
                    ),
                    false,
                )
            };
        if single_row_direct_state {
            finish_decode_batcher_stage_profile(
                "sample_batch_state_direct_row",
                batch,
                stage_start,
            );
        } else if batched_state_cache_hit {
            finish_decode_batcher_stage_profile(
                "sample_batch_state_assemble_cache_hit",
                batch,
                stage_start,
            );
        } else {
            finish_decode_batcher_stage_profile("sample_batch_state_assemble", batch, stage_start);
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        let mut tokens = None;
        #[cfg(feature = "metal")]
        if paged_decode_replay_primitive_enabled(
            self.backend.as_ref(),
            &self.config,
            batch,
            ReplayNativePrimitive::MetalIcb,
        ) && self.active_lora.is_none()
        {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let linear_state_for_graph = if has_linear_layers {
                if single_row_direct_state {
                    Some(&mut *linear_states[0])
                } else {
                    batch_state.as_mut()
                }
            } else {
                None
            };
            let graph_result = {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                runner.decode_step_paged_sample_batch(
                    &*self.backend,
                    input_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_tables,
                    seq_lens,
                    linear_state_for_graph,
                    self.active_lora.as_ref(),
                    &history_rows,
                    &history_indices,
                    &history_counts,
                    &repetition_values,
                    &presence_values,
                    &frequency_values,
                    &temperature_values,
                    &top_k_values,
                    &top_p_values,
                    &min_p_values,
                    &seed_values,
                )
            };
            match graph_result {
                Ok(Some(graph_tokens)) => tokens = Some(graph_tokens),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(
                        batch,
                        error = %err,
                        "Metal graph sampled decode declined; falling back to eager sample decode"
                    );
                }
            }
        }
        if tokens.is_none() {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let linear_state_for_forward = if single_row_direct_state {
                Some(&*linear_states[0])
            } else {
                batch_state.as_ref()
            };
            tokens = model_forward_paged_decode_contiguous_batch_sample_with_ids(
                &*self.backend,
                input_tokens,
                &self.weights,
                &self.config,
                pc_guard,
                block_tables,
                seq_lens,
                linear_state_for_forward,
                self.active_lora.as_ref(),
                row_ids,
                &history_rows,
                &history_indices,
                &history_counts,
                &repetition_values,
                &presence_values,
                &frequency_values,
                &temperature_values,
                &top_k_values,
                &top_p_values,
                &min_p_values,
                &seed_values,
            )
            .context("batched sample decode forward pass (paged) failed")?;
        }
        let Some(tokens) = tokens else {
            return Ok(None);
        };
        finish_decode_batcher_stage_profile("sample_batched_forward", batch, stage_start);

        if let Some(state) = batch_state.as_ref() {
            let stage_start = profile_stages.then(std::time::Instant::now);
            if fast_batched_linear_state_scatter_enabled() {
                if !state.scatter_gdn_state_resident_batch_rows_kt(&*self.backend, linear_states)? {
                    state.scatter_batch_rows_replace(linear_states)?;
                }
                finish_decode_batcher_stage_profile(
                    "sample_batch_state_scatter_replace",
                    batch,
                    stage_start,
                );
            } else {
                state.scatter_batch_rows(linear_states)?;
                finish_decode_batcher_stage_profile(
                    "sample_batch_state_scatter_copy",
                    batch,
                    stage_start,
                );
            }
        }

        if let (Some(state), Some(ids)) = (batch_state.take_for_cache(), row_ids) {
            self.park_batched_state(state, ids);
        }
        finish_decode_batcher_stage_profile("sample_decode_total", batch, total_start);

        Ok(Some(tokens))
    }

    /// Decode multiple compatible paged requests through the transformer stack,
    /// returning final hidden states for caller-owned sampling.
    ///
    /// This mirrors the greedy continuous-batch state assembly/scatter path, but
    /// stops before the LM-head argmax so mixed sampling parameters can still be
    /// handled by the existing sampler.
    fn decode_hidden_paged_contiguous_batch_with_ids(
        &self,
        input_tokens: &[TokenId],
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
        row_ids: Option<&[u64]>,
    ) -> Result<kiln_tensor::Tensor> {
        let _resident_scope = GdnRecurrentResidentStateScope::new(&*self.backend);
        let batch = input_tokens.len();
        let profile_stages = profile_decode_batcher_stages_enabled();
        let total_start = profile_stages.then(std::time::Instant::now);
        anyhow::ensure!(batch > 0, "batched hidden decode requires at least one row");
        anyhow::ensure!(
            block_tables.len() == batch && seq_lens.len() == batch,
            "batched hidden decode metadata length mismatch"
        );

        let has_linear_layers = self.has_linear_attention_layers();
        if has_linear_layers {
            anyhow::ensure!(
                linear_states.len() == batch,
                "batched hidden decode requires one LinearAttentionState per row"
            );
        } else {
            anyhow::ensure!(
                linear_states.is_empty(),
                "full-attention-only batched hidden decode does not accept linear states"
            );
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        if has_linear_layers {
            if ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
                && ReplayBackend::runtime_decode_resident_pool_ready(
                    self.backend.as_ref(),
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    64,
                )
            {
                for state in linear_states.iter() {
                    state.ensure_gdn_state_resident_kt(&*self.backend)?;
                }
            }
            let any_resident = linear_states
                .iter()
                .any(|state| state.has_any_gdn_state_resident_kt(&*self.backend));
            let all_resident = any_resident
                && linear_states
                    .iter()
                    .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
            if any_resident && !all_resident {
                anyhow::bail!(
                    "mixed kt-resident GDN state rows are not supported for batched hidden decode"
                );
            }
        }

        let all_rows_resident = has_linear_layers
            && linear_states
                .iter()
                .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
        let single_row_direct_state = has_linear_layers && batch == 1;
        let (mut batch_state, batched_state_cache_hit) =
            if has_linear_layers && !single_row_direct_state {
                self.prepare_batched_linear_state(linear_states, all_rows_resident, row_ids)?
            } else {
                (
                    ResidentBatchedStateLease::new(
                        None,
                        self.backend.as_ref(),
                        &self.batched_state_cache_counters,
                    ),
                    false,
                )
            };
        if single_row_direct_state {
            finish_decode_batcher_stage_profile(
                "hidden_batch_state_direct_row",
                batch,
                stage_start,
            );
        } else if batched_state_cache_hit {
            finish_decode_batcher_stage_profile(
                "hidden_batch_state_assemble_cache_hit",
                batch,
                stage_start,
            );
        } else {
            finish_decode_batcher_stage_profile("hidden_batch_state_assemble", batch, stage_start);
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        let hidden = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let linear_state_for_forward = if single_row_direct_state {
                Some(&mut *linear_states[0])
            } else {
                batch_state.as_mut()
            };
            model_forward_paged_decode_contiguous_batch_hidden_with_ids(
                &*self.backend,
                input_tokens,
                &self.weights,
                &self.config,
                pc_guard,
                block_tables,
                seq_lens,
                linear_state_for_forward,
                self.active_lora.as_ref(),
                row_ids,
            )
            .context("batched hidden decode forward pass (paged) failed")?
        };
        finish_decode_batcher_stage_profile("hidden_batched_forward", batch, stage_start);

        if let Some(state) = batch_state.as_ref() {
            let stage_start = profile_stages.then(std::time::Instant::now);
            if fast_batched_linear_state_scatter_enabled() {
                if !state.scatter_gdn_state_resident_batch_rows_kt(&*self.backend, linear_states)? {
                    state.scatter_batch_rows_replace(linear_states)?;
                }
                finish_decode_batcher_stage_profile(
                    "hidden_batch_state_scatter_replace",
                    batch,
                    stage_start,
                );
            } else {
                state.scatter_batch_rows(linear_states)?;
                finish_decode_batcher_stage_profile(
                    "hidden_batch_state_scatter_copy",
                    batch,
                    stage_start,
                );
            }
        }

        if let (Some(state), Some(ids)) = (batch_state.take_for_cache(), row_ids) {
            self.park_batched_state(state, ids);
        }
        finish_decode_batcher_stage_profile("hidden_decode_total", batch, total_start);

        Ok(hidden)
    }

    fn decode_next_token_paged_greedy_metal_graph(
        &self,
        input_token: TokenId,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: Option<&mut LinearAttentionState>,
    ) -> Result<Option<TokenId>> {
        #[cfg(feature = "metal")]
        {
            if !paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                1,
                ReplayNativePrimitive::MetalIcb,
            ) || self.active_lora.is_some()
            {
                return Ok(None);
            }

            let pc_guard = lock_paged_cache(paged_cache)?;
            let token_ids = [input_token];
            let block_tables = [block_table];
            let seq_lens = [seq_len];
            let graph_tokens = {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                if !runner.is_enabled() {
                    return Ok(None);
                }
                runner.decode_step_paged_greedy_batch(
                    &*self.backend,
                    &token_ids,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_tables,
                    &seq_lens,
                    linear_state,
                    self.active_lora.as_ref(),
                )?
            };

            if let Some(tokens) = graph_tokens {
                anyhow::ensure!(
                    tokens.len() == 1,
                    "Metal graph single-row greedy returned {} tokens",
                    tokens.len()
                );
                return Ok(tokens.first().copied());
            }
            Ok(None)
        }

        #[cfg(not(feature = "metal"))]
        {
            let _ = (input_token, paged_cache, block_table, seq_len, linear_state);
            Ok(None)
        }
    }

    fn decode_next_token_paged_sample_metal_graph(
        &self,
        params: &SamplingParams,
        input_token: TokenId,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: Option<&mut LinearAttentionState>,
        step_seed: Option<u64>,
        history: &[TokenId],
    ) -> Result<Option<TokenId>> {
        #[cfg(feature = "metal")]
        {
            if params.is_effectively_greedy()
                || !paged_decode_replay_primitive_enabled(
                    self.backend.as_ref(),
                    &self.config,
                    1,
                    ReplayNativePrimitive::MetalIcb,
                )
                || self.active_lora.is_some()
            {
                return Ok(None);
            }
            let top_k = [params.top_k];
            let temperatures = [params.temperature];
            if !SamplingBackend::runtime_supports_linear_decode_sample_batch(
                self.backend.as_ref(),
                &top_k,
                &temperatures,
            ) {
                return Ok(None);
            }

            let mut history_rows = Vec::new();
            let mut history_indices = Vec::new();
            let mut history_counts = Vec::new();
            if !params.token_penalties_are_no_op() && !history.is_empty() {
                let (indices, counts) = unique_history_counts_for_batch_sample(history);
                for (idx, count) in indices.into_iter().zip(counts.into_iter()) {
                    history_rows.push(0);
                    history_indices.push(idx);
                    history_counts.push(count);
                }
            }

            let pc_guard = lock_paged_cache(paged_cache)?;
            let token_ids = [input_token];
            let block_tables = [block_table];
            let seq_lens = [seq_len];
            let repetition_penalties = [params.repetition_penalty];
            let presence_penalties = [params.presence_penalty];
            let frequency_penalties = [params.frequency_penalty];
            let top_p = [params.top_p];
            let min_p = [params.min_p];
            let seeds = [sample_seed_for_batch_row(step_seed, history)];
            let graph_tokens = {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                if !runner.is_enabled() {
                    return Ok(None);
                }
                runner.decode_step_paged_sample_batch(
                    &*self.backend,
                    &token_ids,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_tables,
                    &seq_lens,
                    linear_state,
                    self.active_lora.as_ref(),
                    &history_rows,
                    &history_indices,
                    &history_counts,
                    &repetition_penalties,
                    &presence_penalties,
                    &frequency_penalties,
                    &temperatures,
                    &top_k,
                    &top_p,
                    &min_p,
                    &seeds,
                )?
            };

            if let Some(tokens) = graph_tokens {
                anyhow::ensure!(
                    tokens.len() == 1,
                    "Metal graph single-row sampled decode returned {} tokens",
                    tokens.len()
                );
                return Ok(tokens.first().copied());
            }
            Ok(None)
        }

        #[cfg(not(feature = "metal"))]
        {
            let _ = (
                params,
                input_token,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                step_seed,
                history,
            );
            Ok(None)
        }
    }

    fn decode_next_token_paged_interleaved(
        &self,
        params: &SamplingParams,
        input_token: TokenId,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        step_seed: Option<u64>,
        history: &[TokenId],
        graph_row_id: u64,
        skip_gdn_state_readback: bool,
    ) -> Result<TokenId> {
        let _resident_scope = GdnRecurrentResidentStateScope::new(&*self.backend);
        let _skip_scope =
            crate::forward::VulkanSkipGdnStateReadbackScope::new(skip_gdn_state_readback);
        if params.is_effectively_greedy() && greedy_token_decode_enabled(self.backend.as_ref()) {
            let linear_state_for_graph = if self.has_linear_attention_layers() {
                Some(&mut *linear_state)
            } else {
                None
            };
            if let Some(token) = self
                .decode_next_token_paged_greedy_metal_graph(
                    input_token,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state_for_graph,
                )
                .context("greedy Metal graph decode forward pass (paged) failed")?
            {
                if skip_gdn_state_readback {
                    linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
                }
                return Ok(token);
            }
            let pc_guard = lock_paged_cache(paged_cache)?;
            let token = {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                if runner.is_enabled() {
                    runner.decode_step_paged_greedy(
                        &*self.backend,
                        input_token,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        seq_len,
                        linear_state,
                        self.active_lora.as_ref(),
                    )
                } else {
                    model_forward_paged_next_token_greedy(
                        &*self.backend,
                        input_token,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        seq_len,
                        Some(linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                }
            }
            .context("greedy Metal decode forward pass (paged) failed")?;
            if skip_gdn_state_readback {
                linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
            }
            return Ok(token);
        }

        if !params.is_effectively_greedy()
            && paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                1,
                ReplayNativePrimitive::MetalIcb,
            )
        {
            let linear_state_for_graph = if self.has_linear_attention_layers() {
                Some(&mut *linear_state)
            } else {
                None
            };
            if let Some(token) = self
                .decode_next_token_paged_sample_metal_graph(
                    params,
                    input_token,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state_for_graph,
                    step_seed,
                    history,
                )
                .context("sampled Metal graph decode forward pass (paged) failed")?
            {
                if skip_gdn_state_readback {
                    linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
                }
                return Ok(token);
            }
        }

        // R.9: ROCm HIP-graph decode. On a Rocm device, route the step through
        // the graph runner (capture/replay, with eager fallback). When the
        // runner is disabled by its typed runtime policy, this is skipped entirely
        // and the eager path below runs unchanged.
        if paged_decode_replay_primitive_enabled(
            self.backend.as_ref(),
            &self.config,
            1,
            ReplayNativePrimitive::HipGraph,
        ) {
            let maybe_logits = {
                let mut runner = self
                    .rocm_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?;
                if runner.is_enabled() {
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    Some(
                        runner
                            .decode_step_paged(
                                &*self.backend,
                                input_token,
                                &self.weights,
                                &self.config,
                                pc_guard,
                                block_table,
                                seq_len,
                                linear_state,
                                self.active_lora.as_ref(),
                                graph_row_id,
                            )
                            .context("ROCm graph decode step failed")?,
                    )
                } else {
                    None
                }
            };
            if let Some(logits) = maybe_logits {
                let token = if params.is_effectively_greedy() {
                    greedy_sample(&logits)
                } else {
                    sample_step(&logits, params, step_seed, history)
                }?;
                if skip_gdn_state_readback {
                    linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
                }
                return Ok(token);
            }
        }

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            model_forward_paged(
                &*self.backend,
                &[input_token],
                &self.weights,
                &self.config,
                pc_guard,
                block_table,
                seq_len,
                Some(linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("decode forward pass (paged) failed")?
        };
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let token = if params.is_effectively_greedy() {
            greedy_sample(&logits)
        } else {
            sample_step(&logits, params, step_seed, history)
        }?;
        if skip_gdn_state_readback {
            linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
        }
        Ok(token)
    }

    fn decode_next_token_paged_interleaved_or_batched(
        &self,
        params: &SamplingParams,
        input_token: TokenId,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        step_seed: Option<u64>,
        decode_batcher: Option<&DecodeBatcher>,
        history: &[TokenId],
        graph_row_id: u64,
        skip_gdn_state_readback: bool,
    ) -> Result<TokenId> {
        if params.is_effectively_greedy()
            && let Some(batcher) = decode_batcher
        {
            match batcher.decode_next_token_greedy(
                input_token,
                block_table,
                seq_len,
                linear_state,
                skip_gdn_state_readback,
            )? {
                DecodeBatcherDecode::Decoded(token) => return Ok(token),
                DecodeBatcherDecode::RunnerBusy => {}
            }
        }

        self.decode_next_token_paged_interleaved(
            params,
            input_token,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            step_seed,
            history,
            graph_row_id,
            skip_gdn_state_readback,
        )
    }

    /// Unavailable high-level paged speculative generation entry point.
    ///
    /// Currently returns a stable fail-closed error before inspecting inputs.
    pub fn generate_paged_speculative_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        spec_config: &SpeculativeConfig,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        ensure_speculative_generation_available()?;
        if params.thinking_budget.is_some() {
            return self.generate_paged_shared_tokens(
                prompt_tokens,
                params,
                block_manager,
                paged_cache,
                cancel,
            );
        }
        anyhow::ensure!(
            params.temperature == 0.0,
            "paged skip-layer speculative decode is greedy-only"
        );
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let output = self.generate_from_tokens_paged_speculative_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            spec_config,
            cancel,
        );

        let output = reservation.release_after_settlement(
            self,
            "paged speculative shared KV release",
            output,
        )?;
        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    fn generate_from_tokens_paged_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if self.streaming_prefill.enabled_for(prompt_tokens.len()) {
                model_forward_paged_streaming_with_progress_and_policy(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    cancel,
                    self.streaming_prefill,
                )
                .context("prefill forward pass (paged, streaming) failed")?
            } else {
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prompt_tokens.len() as u64);
                }
                logits
            }
        };
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        let mut next_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, step_seed, &[])?
        };
        let rocm_owner = RocmDecodeOwnerLease::new(&self.rocm_graph, &self.backend_health);

        let result = (|| -> Result<GenerationOutput> {
            for _step in 0..params.max_tokens {
                check_cancelled(cancel)?;
                if let Some(s) = step_seed.as_mut() {
                    *s = s.wrapping_add(1);
                }

                next_token = params.apply_thinking_budget(&generated_tokens, next_token);
                if self.should_stop_on_eos(params, next_token) {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                    });
                }

                generated_tokens.push(next_token);

                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(GenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }

                let skip_gdn_state_readback = skip_final_gdn_state_readback_enabled()
                    && generated_tokens.len() + 1 >= params.max_tokens;
                next_token = self.decode_next_token_paged_interleaved(
                    params,
                    next_token,
                    paged_cache,
                    block_table,
                    seq_len,
                    &mut linear_state,
                    step_seed,
                    &generated_tokens,
                    rocm_owner.row_id(),
                    skip_gdn_state_readback,
                )?;
                seq_len += 1;
            }

            Ok(GenerationOutput {
                text: String::new(),
                token_ids: generated_tokens,
                finish_reason: FinishReason::MaxTokens,
            })
        })();
        match self.synchronize_external_yield("direct interleaved decode completion") {
            Ok(()) => result,
            Err(sync_err) => {
                quarantine_linear_attention_state(&mut linear_state);
                std::mem::forget(logits);
                std::mem::forget(result);
                Err(sync_err)
            }
        }
    }

    /// CUDA-graph variant of the interleaved decode path (Phase 12-B'').
    ///
    /// Mirrors `generate_from_tokens_paged_inner` (the path the old CUDA-graph
    /// branch used) but takes `paged_cache: &PagedKvCache` (Phase 12-C
    /// removed the surrounding `Mutex`; the cache uses interior mutability
    /// for concurrent `&self` writes). The CUDA graph runner mutex is still
    /// acquired per decode step, so that concurrent c=8 requests can
    /// interleave on a per-step granularity rather
    /// than serialising on a generation-lifetime lock. Blocks are still
    /// allocated once up-front by the caller (`generate_from_tokens_paged_shared`)
    /// and freed via `SharedBlockReservation` when the caller drops the
    /// reservation guard, mirroring the non-graph interleaved path.
    fn generate_from_tokens_paged_cuda_graph_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let mut linear_state = self.new_linear_state()?;

        // Prefill: lock the paged cache for one forward pass and drop it
        // before the decode loop starts. The decode loop then re-acquires the
        // cache per step.
        let streaming_prefill = self.streaming_prefill.enabled_for(prompt_tokens.len());
        let prefill_source = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill {
                let logits = model_forward_paged_streaming_with_policy(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    self.streaming_prefill,
                )
                .context("prefill forward pass (paged, streaming) failed")?;
                // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                PrefillSampleSource::Logits(logits)
            } else if params.is_effectively_greedy()
                && greedy_token_decode_enabled(self.backend.as_ref())
            {
                PrefillSampleSource::GreedyToken(
                    model_forward_paged_last_token_greedy(
                        &*self.backend,
                        prompt_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        0,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                    .context("greedy prefill forward pass (paged) failed")?,
                )
            } else {
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prompt_tokens.len() as u64);
                }
                // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                PrefillSampleSource::Logits(logits)
            }
        };

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        let mut next_token = match prefill_source {
            PrefillSampleSource::GreedyToken(token) => token,
            PrefillSampleSource::Logits(logits) => {
                if params.is_effectively_greedy() {
                    greedy_sample(&logits)?
                } else {
                    sample_step(&logits, params, step_seed, &[])?
                }
            }
        };

        for _step in 0..params.max_tokens {
            check_cancelled(cancel)?;
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            if self.should_stop_on_eos(params, next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = if params.is_effectively_greedy()
                && greedy_token_decode_enabled(self.backend.as_ref())
            {
                let linear_state_for_graph = if self.has_linear_attention_layers() {
                    Some(&mut linear_state)
                } else {
                    None
                };
                if let Some(token) = self
                    .decode_next_token_paged_greedy_metal_graph(
                        next_token,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state_for_graph,
                    )
                    .context("greedy Metal graph decode forward pass (paged) failed")?
                {
                    seq_len += 1;
                    token
                } else {
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    let token = model_forward_paged_next_token_greedy(
                        &*self.backend,
                        next_token,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        seq_len,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )?;
                    seq_len += 1;
                    token
                }
            } else {
                // CUDA graph decode step: acquire the graph runner and the
                // paged cache for one step, then drop both before sampling so
                // concurrent requests can interleave on the next step.
                let logits = {
                    let mut graph_runner = self
                        .cuda_graph
                        .lock()
                        .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    graph_runner.decode_step_paged(
                        &*self.backend,
                        next_token,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        seq_len,
                        &mut linear_state,
                        self.active_lora.as_ref(),
                        None,
                    )?
                };
                seq_len += 1;
                // #1082: `decode_step_paged` now returns kt — feed `sample_step`
                // directly, no candle->kt bridge.
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    fn generate_from_tokens_paged_speculative_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        spec_config: &SpeculativeConfig,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if self.streaming_prefill.enabled_for(prompt_tokens.len()) {
                model_forward_paged_streaming_with_policy(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    self.streaming_prefill,
                )
                .context("prefill forward pass (paged skip-layer, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged skip-layer) failed")?
            }
        };
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut base_pos = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut last_token = greedy_sample(&logits)?;

        loop {
            check_cancelled(cancel)?;
            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            if self.should_stop_on_eos(params, last_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(last_token);
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: spec_config.num_speculative_tokens.min(remaining),
                draft_layers: spec_config.draft_layers,
            };

            let result = {
                let pc_guard = lock_paged_cache(paged_cache)?;
                speculative_decode_step_paged_greedy(
                    &*self.backend,
                    last_token,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    base_pos,
                    &mut linear_state,
                    &mut draft_linear_state,
                    &effective_config,
                    params,
                    self.eos_token_ids_for(params),
                    self.active_lora.as_ref(),
                )
                .context("paged skip-layer speculative decode step failed")?
            };
            base_pos += result.base_advance;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                    });
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);
                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(GenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();
            if result.hit_eos {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Inner generation loop using paged KV cache (blocks already allocated).
    fn generate_from_tokens_paged_inner(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let mut linear_state = self.new_linear_state()?;

        // Prefill: forward pass on all prompt tokens (never uses CUDA graphs).
        // The immutable startup policy selects tiled or monolithic execution.
        let streaming_prefill = self.streaming_prefill.enabled_for(prompt_tokens.len());
        let prefill_source = if streaming_prefill {
            let logits = model_forward_paged_streaming_with_policy(
                &*self.backend,
                prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                self.streaming_prefill,
            )
            .context("prefill forward pass (paged, streaming) failed")?;
            // (#1082) kt-native logits — sampler is kt now; no candle bridge.
            PrefillSampleSource::Logits(logits)
        } else if params.is_effectively_greedy()
            && greedy_token_decode_enabled(self.backend.as_ref())
        {
            PrefillSampleSource::GreedyToken(
                model_forward_paged_last_token_greedy(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("greedy prefill forward pass (paged) failed")?,
            )
        } else {
            let logits = model_forward_paged_last_token(
                &*self.backend,
                prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("prefill forward pass (paged) failed")?;
            // (#1082) kt-native logits — sampler is kt now; no candle bridge.
            PrefillSampleSource::Logits(logits)
        };

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        // Acquire the CUDA graph runner for decode steps
        let mut graph_runner = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;

        let mut next_token = match prefill_source {
            PrefillSampleSource::GreedyToken(token) => token,
            PrefillSampleSource::Logits(logits) => {
                if params.is_effectively_greedy() {
                    greedy_sample(&logits)?
                } else {
                    sample_step(&logits, params, step_seed, &[])?
                }
            }
        };

        for _step in 0..params.max_tokens {
            check_cancelled(cancel)?;
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            // Check for EOS
            if self.should_stop_on_eos(params, next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            // Check stop sequences
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = if params.is_effectively_greedy()
                && greedy_token_decode_enabled(self.backend.as_ref())
            {
                let linear_state_for_graph = if self.has_linear_attention_layers() {
                    Some(&mut linear_state)
                } else {
                    None
                };
                if let Some(token) = self
                    .decode_next_token_paged_greedy_metal_graph(
                        next_token,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state_for_graph,
                    )
                    .context("greedy Metal graph decode forward pass (paged) failed")?
                {
                    seq_len += 1;
                    token
                } else {
                    let token = model_forward_paged_next_token_greedy(
                        &*self.backend,
                        next_token,
                        &self.weights,
                        &self.config,
                        paged_cache,
                        block_table,
                        seq_len,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )?;
                    seq_len += 1;
                    token
                }
            } else {
                // Decode step: use CUDA graph runner (captures/replays when enabled)
                let logits = graph_runner.decode_step_paged(
                    &*self.backend,
                    next_token,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    block_table,
                    seq_len,
                    &mut linear_state,
                    self.active_lora.as_ref(),
                    None,
                )?;
                seq_len += 1;
                // #1082: `decode_step_paged` now returns kt — feed `sample_step`
                // directly, no candle->kt bridge.
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Generate text using self-speculative decoding (skip-layer draft).
    ///
    /// The first `spec_config.draft_layers` layers of the model propose candidate
    /// tokens, and the full model verifies them in a single forward pass. Any
    /// speedup is backend- and workload-dependent and requires qualification.
    ///
    /// Currently returns a stable fail-closed error before tokenization or any
    /// accelerator work.
    pub fn generate_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<GenerationOutput> {
        ensure_speculative_generation_available()?;
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_speculative(&prompt_tokens, params, spec_config)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Unavailable speculative generation loop operating on token IDs.
    ///
    /// Currently returns a stable fail-closed error before inspecting inputs or
    /// performing accelerator work.
    pub fn generate_from_tokens_speculative(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<GenerationOutput> {
        ensure_speculative_generation_available()?;
        use rand::SeedableRng;

        if params.thinking_budget.is_some() {
            return self.generate_from_tokens(prompt_tokens, params);
        }
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        // Verification writes the full speculative window (`last_token + k`)
        // before the loop commits accepted tokens, so flat KV needs temporary
        // headroom beyond the user-visible max token budget.
        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: full model forward pass on all prompt tokens
        let logits = model_forward_kt_with_policy(
            &*self.backend,
            prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
            self.streaming_prefill,
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::make_rng::<rand::rngs::StdRng>(),
        };

        // Sample first token from prefill logits
        let mut last_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, params.seed, &[])?
        };

        loop {
            // Check if we've hit max_tokens
            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            // Check for EOS
            if self.should_stop_on_eos(params, last_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(last_token);

            // Check stop sequences
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            // Run one speculative decode step
            let remaining = params.max_tokens - generated_tokens.len();
            let effective_k = spec_config.num_speculative_tokens.min(remaining);
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: effective_k,
                draft_layers: spec_config.draft_layers,
            };

            let result = speculative_decode_step(
                &*self.backend,
                last_token,
                &self.weights,
                &self.config,
                &mut kv_cache,
                &mut linear_state,
                &mut draft_linear_state,
                &effective_config,
                params,
                self.eos_token_ids_for(params),
                &mut rng,
                self.active_lora.as_ref(),
            )
            .context("speculative decode step failed")?;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                    });
                }
                // No tokens accepted and no EOS — shouldn't happen normally,
                // but fall back to sampling from the verification logits.
                // Break to avoid infinite loop.
                break;
            }

            // Add accepted tokens (except the last one which becomes last_token)
            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);

                // Check stop sequences after each token
                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(GenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Unavailable native-MTP speculative text generation entry point.
    ///
    /// Currently returns a stable fail-closed error before tokenization, weight
    /// materialization, or accelerator work.
    pub fn generate_mtp_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<MtpGenerationOutput> {
        ensure_speculative_generation_available()?;
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_mtp_speculative(&prompt_tokens, params)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(MtpGenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
            draft_accepted_count: output.draft_accepted_count,
            total_draft_attempts: output.total_draft_attempts,
        })
    }

    /// Unavailable native-MTP speculative generation over token IDs.
    ///
    /// Currently returns a stable fail-closed error before inspecting inputs,
    /// materializing deferred weights, or allocating caches.
    pub fn generate_from_tokens_mtp_speculative(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<MtpGenerationOutput> {
        ensure_speculative_generation_available()?;
        use rand::SeedableRng;

        if params.thinking_budget.is_some() {
            let output = self.generate_from_tokens(prompt_tokens, params)?;
            return Ok(MtpGenerationOutput {
                text: output.text,
                token_ids: output.token_ids,
                finish_reason: output.finish_reason,
                draft_accepted_count: 0,
                total_draft_attempts: 0,
            });
        }
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        anyhow::ensure!(
            self.weights.mtp.is_some(),
            "generate_mtp_speculative requires the checkpoint to carry mtp.* tensors \
             (Qwen3.5-4B native MTP head)"
        );
        anyhow::ensure!(
            params.temperature == 0.0,
            "generate_mtp_speculative currently only supports greedy decoding (temperature == 0)"
        );

        // Block size matches the kiln-core default + the bench convention
        // (#1082: 16 -> 64 so each FA2 kBlockN=64 tile is one physical page).
        const BLOCK_SIZE: usize = 64;

        let max_total = prompt_tokens.len() + params.max_tokens;
        // (#1082) kt-native paged cache — `PagedKvCacheKt::new` allocates pools
        // on the model's runtime `Device` (kiln is single-GPU).
        let cache_device =
            paged_cache_device(self.backend.as_ref(), &self.weights.embed_tokens.device())?;
        let dtype = paged_cache_kt_dtype(self.config.dtype);

        // Two independent paged caches:
        //   * `base_cache` covers the model's full-attention layers.
        //   * `mtp_cache` is a single-layer cache for the MTP block.
        // Each gets its own block table mapping logical block i -> physical i.
        let num_blocks = Self::blocks_needed(max_total, BLOCK_SIZE);
        let base_cache = PagedKvCache::new(
            self.config.num_full_attention_layers,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            cache_device,
        )?;
        let mtp_cache = PagedKvCache::new(
            1,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            cache_device,
        )?;
        let mut base_block_table = BlockTable::new();
        let mut mtp_block_table = BlockTable::new();
        for i in 0..num_blocks as u32 {
            base_block_table.push(i);
            mtp_block_table.push(i);
        }

        let mut linear_state = self.new_linear_state()?;

        // Prefill: feed the prompt through the base model and capture the
        // post-final-norm last hidden row as the seed `h_prev`.
        let (prefill_logits_kt, h_prev_kt) =
            if self.streaming_prefill.enabled_for(prompt_tokens.len()) {
                model_forward_paged_streaming_last_token_with_last_hidden_with_policy(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &base_cache,
                    &base_block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    self.streaming_prefill,
                )
                .context("mtp streaming prefill forward pass failed")?
            } else {
                model_forward_paged_last_token_with_last_hidden(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &base_cache,
                    &base_block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("mtp prefill forward pass failed")?
            };
        // (#1082) MTP speculative step + speculative.rs are fully kt now —
        // `h_prev`/`prefill_logits` stay kt; no candle bridge.
        let prefill_logits = prefill_logits_kt;
        let mut h_prev = h_prev_kt;

        // The last-row logits drive the first emitted token (same as the
        // skip-layer path).
        let prefill_last = prefill_logits.squeeze(1)?;
        let mut last_token = greedy_sample(&prefill_last)?;

        let mut base_pos = prompt_tokens.len();
        let mut mtp_pos = 0usize;
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut draft_accepted_count: usize = 0;
        let mut total_draft_attempts: usize = 0;

        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::make_rng::<rand::rngs::StdRng>(),
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            if self.should_stop_on_eos(params, last_token) {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            generated_tokens.push(last_token);

            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(MtpGenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                draft_accepted_count,
                                total_draft_attempts,
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            total_draft_attempts += 1;
            let mut replay_prefix =
                Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
            replay_prefix.extend_from_slice(prompt_tokens);
            replay_prefix.extend_from_slice(&generated_tokens);
            crate::mtp_debug::set_h_main_replay_prefix_tokens(&replay_prefix);
            let result = speculative_mtp_decode_step(
                &*self.backend,
                last_token,
                &h_prev,
                &self.weights,
                &self.config,
                &base_cache,
                &base_block_table,
                base_pos,
                &mut linear_state,
                &mtp_cache,
                &mtp_block_table,
                mtp_pos,
                params,
                self.eos_token_ids_for(params),
                &mut rng,
                self.active_lora.as_ref(),
            );
            crate::mtp_debug::clear_h_main_replay_prefix_tokens();
            let result = result.context("mtp speculative decode step failed")?;

            if result.draft_accepted {
                draft_accepted_count += 1;
            }
            base_pos += result.base_advance;
            mtp_pos += result.mtp_advance;
            h_prev = result.new_h_prev;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(MtpGenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                        draft_accepted_count,
                        total_draft_attempts,
                    });
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);

                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(MtpGenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                    draft_accepted_count,
                                    total_draft_attempts,
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(MtpGenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                        draft_accepted_count,
                        total_draft_attempts,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }
        }

        Ok(MtpGenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
            draft_accepted_count,
            total_draft_attempts,
        })
    }

    /// Streaming self-speculative decoding (skip-layer draft).
    ///
    /// Currently returns a stable fail-closed error before tokenization,
    /// channel creation, or accelerator work.
    pub fn generate_streaming_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        ensure_speculative_generation_available()?;
        use rand::SeedableRng;

        if params.thinking_budget.is_some() {
            return self.generate_streaming(prompt, params);
        }
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let (tx, rx) = mpsc::channel();
        // Verification writes the full speculative window (`last_token + k`)
        // before the loop commits accepted tokens, so flat KV needs temporary
        // headroom beyond the user-visible max token budget.
        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        let logits = model_forward_kt_with_policy(
            &*self.backend,
            &prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
            self.streaming_prefill,
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::make_rng::<rand::rngs::StdRng>(),
        };

        let mut last_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, params.seed, &[])?
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.should_stop_on_eos(params, last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                last_token,
            )? {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    let completion_tokens = generated_tokens.len();
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens,
                        trailing_text: String::new(),
                    }));
                    return Ok(rx);
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_k = spec_config.num_speculative_tokens.min(remaining);
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: effective_k,
                draft_layers: spec_config.draft_layers,
            };

            let result = speculative_decode_step(
                &*self.backend,
                last_token,
                &self.weights,
                &self.config,
                &mut kv_cache,
                &mut linear_state,
                &mut draft_linear_state,
                &effective_config,
                params,
                self.eos_token_ids_for(params),
                &mut rng,
                self.active_lora.as_ref(),
            )
            .context("speculative decode step failed")?;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut gate,
                    &mut generated_tokens,
                    token,
                )? {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        let completion_tokens = generated_tokens.len();
                        let _ = tx.send(StreamEvent::Done(StreamDone {
                            finish_reason: reason,
                            completion_tokens,
                            trailing_text: String::new(),
                        }));
                        return Ok(rx);
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens)?;
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }));

        Ok(rx)
    }

    /// Streaming native-MTP speculative decoding.
    ///
    /// Currently returns a stable fail-closed error before tokenization,
    /// deferred-weight materialization, channel creation, or accelerator work.
    pub fn generate_streaming_mtp_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        ensure_speculative_generation_available()?;
        use rand::SeedableRng;

        if params.thinking_budget.is_some() {
            return self.generate_streaming(prompt, params);
        }
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        anyhow::ensure!(
            self.weights.mtp.is_some(),
            "generate_streaming_mtp_speculative requires the checkpoint to carry mtp.* tensors \
             (Qwen3.5-4B native MTP head)"
        );
        anyhow::ensure!(
            params.temperature == 0.0,
            "generate_streaming_mtp_speculative currently only supports greedy decoding \
             (temperature == 0)"
        );

        // #1082: 16 -> 64 so each FA2 kBlockN=64 tile is one physical page.
        const BLOCK_SIZE: usize = 64;

        let max_total = prompt_tokens.len() + params.max_tokens;
        // (#1082) kt-native paged cache — kt `DType` + runtime `Device`.
        let cache_device =
            paged_cache_device(self.backend.as_ref(), &self.weights.embed_tokens.device())?;
        let dtype = paged_cache_kt_dtype(self.config.dtype);

        let num_blocks = Self::blocks_needed(max_total, BLOCK_SIZE);
        let base_cache = PagedKvCache::new(
            self.config.num_full_attention_layers,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            cache_device,
        )?;
        let mtp_cache = PagedKvCache::new(
            1,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            cache_device,
        )?;
        let mut base_block_table = BlockTable::new();
        let mut mtp_block_table = BlockTable::new();
        for i in 0..num_blocks as u32 {
            base_block_table.push(i);
            mtp_block_table.push(i);
        }

        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        let (prefill_logits_kt, h_prev_kt) =
            if self.streaming_prefill.enabled_for(prompt_tokens.len()) {
                model_forward_paged_streaming_last_token_with_last_hidden_with_policy(
                    &*self.backend,
                    &prompt_tokens,
                    &self.weights,
                    &self.config,
                    &base_cache,
                    &base_block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    self.streaming_prefill,
                )
                .context("mtp streaming prefill forward pass failed")?
            } else {
                model_forward_paged_last_token_with_last_hidden(
                    &*self.backend,
                    &prompt_tokens,
                    &self.weights,
                    &self.config,
                    &base_cache,
                    &base_block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("mtp prefill forward pass failed")?
            };
        // (#1082) MTP speculative step + speculative.rs are fully kt now —
        // `h_prev`/`prefill_logits` stay kt; no candle bridge.
        let prefill_logits = prefill_logits_kt;
        let mut h_prev = h_prev_kt;

        let prefill_last = prefill_logits.squeeze(1)?;
        let mut last_token = greedy_sample(&prefill_last)?;

        let mut base_pos = prompt_tokens.len();
        let mut mtp_pos = 0usize;
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::make_rng::<rand::rngs::StdRng>(),
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.should_stop_on_eos(params, last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                last_token,
            )? {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    let completion_tokens = generated_tokens.len();
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens,
                        trailing_text: String::new(),
                    }));
                    return Ok(rx);
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let mut replay_prefix =
                Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
            replay_prefix.extend_from_slice(&prompt_tokens);
            replay_prefix.extend_from_slice(&generated_tokens);
            crate::mtp_debug::set_h_main_replay_prefix_tokens(&replay_prefix);
            let result = speculative_mtp_decode_step(
                &*self.backend,
                last_token,
                &h_prev,
                &self.weights,
                &self.config,
                &base_cache,
                &base_block_table,
                base_pos,
                &mut linear_state,
                &mtp_cache,
                &mtp_block_table,
                mtp_pos,
                params,
                self.eos_token_ids_for(params),
                &mut rng,
                self.active_lora.as_ref(),
            );
            crate::mtp_debug::clear_h_main_replay_prefix_tokens();
            let result = result.context("mtp speculative decode step failed")?;

            base_pos += result.base_advance;
            mtp_pos += result.mtp_advance;
            h_prev = result.new_h_prev;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut gate,
                    &mut generated_tokens,
                    token,
                )? {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        let completion_tokens = generated_tokens.len();
                        let _ = tx.send(StreamEvent::Done(StreamDone {
                            finish_reason: reason,
                            completion_tokens,
                            trailing_text: String::new(),
                        }));
                        return Ok(rx);
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens)?;
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }));

        Ok(rx)
    }

    /// Streaming generation using shared paged-cache state protected by
    /// short-lived mutexes.
    ///
    /// Mirrors [`generate_paged_shared`]: CUDA graph-enabled runtimes keep the
    /// existing whole-request lock scope, while non-CUDA desktop paths reserve
    /// blocks up front and lock the paged cache only around prefill / decode
    /// forward passes.
    pub fn generate_streaming_paged_shared(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        self.generate_from_tokens_streaming_paged_shared(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    /// Streaming variant of [`Self::generate_paged_shared_tokens`].
    pub fn generate_streaming_paged_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        self.generate_from_tokens_streaming_paged_shared(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    /// Same as [`Self::generate_streaming_paged_shared_tokens`], but optionally reuses
    /// a block-aligned cached prefix and returns completed prompt metadata that
    /// the caller may register after successful generation.
    pub fn generate_streaming_paged_shared_tokens_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
    ) -> Result<PrefixCachedStreamingOutput> {
        self.generate_from_tokens_streaming_paged_interleaved_with_prefix_cache(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
            cached_prefix,
        )
    }

    /// Threaded variant of [`Self::generate_streaming_paged_shared_tokens`] that
    /// performs prefill on the calling thread and runs the decode loop on a
    /// spawned `std::thread`. The returned receiver yields tokens as they are
    /// produced, instead of after the entire `max_tokens` loop has completed
    /// (which is the behavior of the legacy `&self` variant — fine for unit
    /// tests but it makes `stream: true` look hung at the HTTP layer because
    /// the receiver only becomes observable when generation finishes).
    ///
    /// Holds an `Arc<RwLock<Self>>` so the spawned worker can re-acquire a
    /// read lock for decode steps without keeping the lock guard alive across
    /// thread boundaries (which `RwLockReadGuard` cannot do).
    ///
    /// `cancel` is required because dropping the receiver does not interrupt
    /// GPU work already running on the worker. Callers must retain a clone,
    /// signal it, and then observe [`ThreadedStreamingOutput::settled`] before
    /// considering a cancelled request quiescent.
    ///
    /// A prefill panic is converted to `Err` after quarantining the backend.
    /// The request lifetime and allocation ownership are intentionally retained
    /// in that case because completion cannot be proven.
    pub fn spawn_streaming_paged_shared_tokens<L>(
        runner_lock: Arc<std::sync::RwLock<Self>>,
        prompt_tokens: Vec<TokenId>,
        params: SamplingParams,
        block_manager: Arc<Mutex<BlockManager>>,
        paged_cache: Arc<PagedKvCache>,
        decode_batcher: Option<Arc<DecodeBatcher>>,
        cancel: CancelHandle,
        worker_lifetime: L,
    ) -> Result<ThreadedStreamingOutput>
    where
        L: Send + 'static,
    {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        check_cancelled(Some(&cancel))?;

        let backend_health = {
            let runner = runner_lock
                .read()
                .map_err(|err| anyhow::anyhow!("failed to acquire runner read lock: {err}"))?;
            runner.ensure_backend_healthy()?;
            runner.backend_health_handle()
        };

        // Allocate the full block reservation up front so the prompt + decode
        // window has its KV cache pages laid out before we hand the receiver
        // back to the caller. The legacy synchronous path uses
        // `SharedBlockReservation` for RAII free-on-drop; here we own the
        // block ids through to the end of the spawned thread instead.
        let max_total = prompt_tokens.len() + params.max_tokens;
        let block_table = {
            let mut bm_guard = lock_block_manager(block_manager.as_ref())?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            block_table
        };
        let ownership = ThreadedPrefillOwnership {
            allocated_blocks: block_table.blocks.clone(),
            block_table,
            linear_state: None,
            post_decode: (),
            worker_lifetime,
        };

        // Run prefill on the calling thread so a malformed prompt fails the
        // request synchronously rather than via an SSE error chunk. The decode
        // loop is what actually benefits from being threaded.
        let (prefill_result, ownership) = run_threaded_prefill_with_panic_fence(
            &backend_health,
            "threaded streaming prefill",
            ownership,
            |ownership| {
                let runner_guard = runner_lock
                    .read()
                    .map_err(|e| anyhow::anyhow!("failed to acquire runner read lock: {e}"))?;
                let result = (|| -> Result<_> {
                    runner_guard.ensure_backend_healthy()?;
                    check_cancelled(Some(&cancel))?;
                    ownership.linear_state = Some(runner_guard.new_linear_state()?);
                    let linear_state = ownership
                        .linear_state
                        .as_mut()
                        .expect("linear state initialized before prefill");
                    let logits = {
                        let pc_guard = lock_paged_cache(paged_cache.as_ref())?;
                        if runner_guard
                            .streaming_prefill
                            .enabled_for(prompt_tokens.len())
                        {
                            model_forward_paged_streaming_with_progress_and_policy(
                                &*runner_guard.backend,
                                &prompt_tokens,
                                &runner_guard.weights,
                                &runner_guard.config,
                                pc_guard,
                                &ownership.block_table,
                                0,
                                Some(linear_state),
                                runner_guard.active_lora.as_ref(),
                                Some(&cancel),
                                runner_guard.streaming_prefill,
                            )
                            .context("prefill forward pass (paged, streaming) failed")?
                        } else {
                            let logits = model_forward_paged_last_token(
                                &*runner_guard.backend,
                                &prompt_tokens,
                                &runner_guard.weights,
                                &runner_guard.config,
                                pc_guard,
                                &ownership.block_table,
                                0,
                                Some(linear_state),
                                runner_guard.active_lora.as_ref(),
                                None,
                            )
                            .context("prefill forward pass (paged) failed")?;
                            cancel.report_prefill_tokens_completed(prompt_tokens.len() as u64);
                            logits
                        }
                    };
                    check_cancelled(Some(&cancel))?;
                    let next_token = sample_first_decode_token(&logits, &params)?;
                    Ok((next_token, logits))
                })();
                let synchronized =
                    runner_guard.synchronize_external_yield("threaded streaming prefill");
                drop(runner_guard);
                match synchronized {
                    Ok(()) => result,
                    Err(err) => {
                        std::mem::forget(result);
                        Err(err)
                    }
                }
            },
        )?;
        let (next_token, logits) = match prefill_result {
            Ok(result) => result,
            Err(err) => {
                if backend_health.snapshot().quarantined {
                    std::mem::forget(ownership);
                } else if !ownership.allocated_blocks.is_empty() {
                    lock_block_manager(block_manager.as_ref())?
                        .free_all(&ownership.allocated_blocks);
                }
                return Err(err);
            }
        };
        drop(logits);
        let ThreadedPrefillOwnership {
            worker_lifetime,
            block_table,
            allocated_blocks,
            linear_state,
            post_decode: (),
        } = ownership;
        let mut linear_state = linear_state.expect("successful prefill initialized linear state");

        let (tx, rx) = mpsc::channel();
        let seq_len = prompt_tokens.len();
        let runner_for_thread = runner_lock;
        let bm_for_thread = block_manager.clone();
        let pc_for_thread = paged_cache;
        let decode_batcher_for_thread = decode_batcher;
        let cleanup = PrefixCachedStreamingCleanup {
            registration: None,
            extra_registrations: Vec::new(),
            allocated_blocks: allocated_blocks.clone(),
        };
        let blocks_for_spawn_failure = allocated_blocks;
        let backend_health_for_thread = backend_health.clone();
        let (settled_tx, settled_rx) = mpsc::channel();
        let spawn_result = std::thread::Builder::new()
            .name("kiln-stream-decode".to_string())
            .spawn(move || {
                let worker_lifetime = worker_lifetime;
                let quarantined = run_prefix_cached_stream_worker(
                    tx,
                    move |tx| {
                        let runner_guard = match runner_for_thread.read() {
                            Ok(guard) => guard,
                            Err(err) => {
                                return PrefixStreamDecodeOutcome::Quarantined(format!(
                                    "failed to acquire runner read lock in decode thread: {err}"
                                ));
                            }
                        };
                        if let Err(err) = runner_guard.ensure_backend_healthy() {
                            return PrefixStreamDecodeOutcome::Quarantined(err.to_string());
                        }
                        let result = runner_guard.run_stream_decode_loop_with_first(
                            tx,
                            next_token,
                            seq_len,
                            &params,
                            pc_for_thread.as_ref(),
                            &block_table,
                            &mut linear_state,
                            decode_batcher_for_thread.as_deref(),
                            Some(&cancel),
                        );
                        match runner_guard.ensure_backend_healthy() {
                            Ok(()) => PrefixStreamDecodeOutcome::Settled(result),
                            Err(err) => PrefixStreamDecodeOutcome::Quarantined(format!(
                                "backend became unhealthy during streaming decode: {err:#}"
                            )),
                        }
                    },
                    move |cleanup| {
                        if cleanup.allocated_blocks.is_empty() {
                            return Ok(());
                        }
                        let mut guard = lock_block_manager(bm_for_thread.as_ref())?;
                        guard.free_all(&cleanup.allocated_blocks);
                        Ok(())
                    },
                    cleanup,
                    &backend_health_for_thread,
                );
                if quarantined {
                    std::mem::forget(worker_lifetime);
                } else {
                    drop(worker_lifetime);
                }
                let _ = settled_tx.send(());
            });

        if let Err(err) = spawn_result {
            if !blocks_for_spawn_failure.is_empty() {
                lock_block_manager(block_manager.as_ref())?.free_all(&blocks_for_spawn_failure);
            }
            return Err(anyhow::anyhow!(
                "failed to spawn streaming decode thread: {err}"
            ));
        }
        Ok(ThreadedStreamingOutput {
            receiver: rx,
            settled: settled_rx,
        })
    }

    /// Threaded variant of
    /// [`Self::generate_streaming_paged_shared_tokens_with_prefix_cache`]. Same
    /// motivation as [`Self::spawn_streaming_paged_shared_tokens`]: hand the
    /// receiver back before decode starts so the SSE layer can stream tokens
    /// in real time. It has the same required cancellation and explicit
    /// settlement contract as the non-prefix variant. Prefill panics quarantine
    /// the backend and retain the request lifetime, allocation metadata, and
    /// prefix-cache finalizer rather than releasing an unproven cache lease.
    pub fn spawn_streaming_paged_shared_tokens_with_prefix_cache<F, L>(
        runner_lock: Arc<std::sync::RwLock<Self>>,
        prompt_tokens: Vec<TokenId>,
        params: SamplingParams,
        block_manager: Arc<Mutex<BlockManager>>,
        paged_cache: Arc<PagedKvCache>,
        cached_prefix: Option<PagedPrefixReuse>,
        decode_batcher: Option<Arc<DecodeBatcher>>,
        cancel: CancelHandle,
        worker_lifetime: L,
        post_decode: F,
    ) -> Result<ThreadedStreamingOutput>
    where
        F: FnOnce(PrefixCachedStreamingCleanup) -> Result<()> + Send + 'static,
        L: Send + 'static,
    {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        check_cancelled(Some(&cancel))?;

        let backend_health = {
            let runner = runner_lock
                .read()
                .map_err(|err| anyhow::anyhow!("failed to acquire runner read lock: {err}"))?;
            runner.ensure_backend_healthy()?;
            runner.backend_health_handle()
        };

        let block_size = {
            let bm_guard = lock_block_manager(block_manager.as_ref())?;
            bm_guard.block_size()
        };

        let cached_prefix = cached_prefix.filter(|prefix| {
            paged_prefix_reuse_matches_prompt(prefix, prompt_tokens.len(), block_size, &params)
        });

        let cached_blocks = cached_prefix
            .as_ref()
            .map(|prefix| prefix.block_ids.clone())
            .unwrap_or_default();
        let cached_tokens = cached_prefix
            .as_ref()
            .map(|prefix| prefix.cached_tokens)
            .unwrap_or(0);

        let max_total = prompt_tokens.len() + params.max_tokens;
        let total_blocks = Self::blocks_needed(max_total, block_size);
        let additional_blocks_needed = total_blocks.saturating_sub(cached_blocks.len());
        let allocated_blocks = {
            let mut bm_guard = lock_block_manager(block_manager.as_ref())?;
            bm_guard
                .allocate(additional_blocks_needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let block_table = append_prefix_block_table(&cached_blocks, &allocated_blocks);
        let ownership = ThreadedPrefillOwnership {
            worker_lifetime,
            post_decode,
            block_table,
            allocated_blocks,
            linear_state: None,
        };

        // Free helper for failure paths so a prefill error does not leak the
        // freshly-allocated suffix blocks (the cached-prefix blocks remain
        // owned by the prefix cache and must not be freed here).
        let free_allocated = |allocated: &[u32]| {
            if allocated.is_empty() {
                return;
            }
            match block_manager.lock() {
                Ok(mut guard) => guard.free_all(allocated),
                Err(e) => tracing::error!(
                    error = %e,
                    "failed to lock block manager to free blocks after prefix-cache prefill error"
                ),
            }
        };

        let (prepared, ownership) = run_threaded_prefill_with_panic_fence(
            &backend_health,
            "prefix-cached threaded streaming prefill",
            ownership,
            |ownership| {
                let runner_guard = runner_lock
                    .read()
                    .map_err(|e| anyhow::anyhow!("failed to acquire runner read lock: {e}"))?;
                let result = (|| -> Result<(
                    TokenId,
                    Option<PagedPrefixRegistration>,
                    Vec<PagedPrefixRegistration>,
                    Option<kiln_tensor::Tensor>,
                )> {
                    runner_guard.ensure_backend_healthy()?;
                    check_cancelled(Some(&cancel))?;
                    let (exact_next_token, linear_state) = match cached_prefix {
                        Some(prefix) => {
                            let exact_next_token =
                                if prefix.cached_tokens == prompt_tokens.len() {
                                    prefix.next_token
                                } else {
                                    None
                                };
                            (exact_next_token, prefix.linear_state)
                        }
                        None => (None, runner_guard.new_linear_state()?),
                    };
                    ownership.linear_state = Some(linear_state);
                    let ThreadedPrefillOwnership {
                        block_table,
                        linear_state,
                        ..
                    } = ownership;
                    let linear_state = linear_state
                        .as_mut()
                        .expect("prefix linear state initialized before prefill");
                    if let Some(next_token) = exact_next_token {
                        return match next_token {
                            PagedPrefixNextToken::Logits(logits) => {
                                let token = sample_first_decode_token(&logits, &params)?;
                                Ok((token, None, Vec::new(), Some(logits)))
                            }
                            PagedPrefixNextToken::GreedyToken(token) => {
                                anyhow::ensure!(
                                    params.temperature == 0.0,
                                    "greedy cached first token cannot serve non-greedy sampling"
                                );
                                Ok((token, None, Vec::new(), None))
                            }
                        };
                    }

                    let prefill_tokens = &prompt_tokens[cached_tokens..];
                    anyhow::ensure!(
                        !prefill_tokens.is_empty(),
                        "non-exact streaming prefix cache hit must leave at least one suffix token"
                    );

                    let split_pos =
                        strict_prompt_prefix_split_pos(prompt_tokens.len(), cached_tokens, block_size);
                    let mut prefill_split_snapshot: Option<RollingPrefixSnapshot> = None;
                    let logits = {
                        let pc_guard = lock_paged_cache(paged_cache.as_ref())?;
                        if runner_guard.streaming_prefill.enabled_for(prefill_tokens.len()) {
                            if let Some(split_pos) = split_pos {
                                let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                                let _ = model_forward_paged_streaming_with_progress_and_policy(
                                    &*runner_guard.backend,
                                    head_tokens,
                                    &runner_guard.weights,
                                    &runner_guard.config,
                                    pc_guard,
                                    &*block_table,
                                    cached_tokens,
                                    Some(&mut *linear_state),
                                    runner_guard.active_lora.as_ref(),
                                    Some(&cancel),
                                    runner_guard.streaming_prefill,
                                )
                                .context(
                                    "prefill forward pass (streaming paged prefix cache head)",
                                )?;
                                prefill_split_snapshot = runner_guard
                                    .authoritative_prefix_snapshot(
                                        &linear_state,
                                        "streaming-prefix-cache-split",
                                        split_pos,
                                    )
                                    .context(
                                        "snapshot linear state at streaming prefix-cache split",
                                    )?
                                    .map(|linear_state| RollingPrefixSnapshot {
                                        position: split_pos,
                                        linear_state,
                                    });

                                let tail_tokens = &prompt_tokens[split_pos..];
                                model_forward_paged_streaming_with_progress_offset_and_policy(
                                    &*runner_guard.backend,
                                    tail_tokens,
                                    &runner_guard.weights,
                                    &runner_guard.config,
                                    pc_guard,
                                    &*block_table,
                                    split_pos,
                                    Some(&mut *linear_state),
                                    runner_guard.active_lora.as_ref(),
                                    Some(&cancel),
                                    head_tokens.len() as u64,
                                    runner_guard.streaming_prefill,
                                )
                                .context(
                                    "prefill forward pass (streaming paged prefix cache tail)",
                                )?
                            } else {
                                model_forward_paged_streaming_with_progress_and_policy(
                                    &*runner_guard.backend,
                                    prefill_tokens,
                                    &runner_guard.weights,
                                    &runner_guard.config,
                                    pc_guard,
                                    &*block_table,
                                    cached_tokens,
                                    Some(&mut *linear_state),
                                    runner_guard.active_lora.as_ref(),
                                    Some(&cancel),
                                    runner_guard.streaming_prefill,
                                )
                                .context(
                                    "prefill forward pass (streaming paged prefix cache) failed",
                                )?
                            }
                        } else {
                            let logits = model_forward_paged_last_token(
                                &*runner_guard.backend,
                                prefill_tokens,
                                &runner_guard.weights,
                                &runner_guard.config,
                                pc_guard,
                                &*block_table,
                                cached_tokens,
                                Some(&mut *linear_state),
                                runner_guard.active_lora.as_ref(),
                                None,
                            )
                            .context("prefill forward pass (paged prefix cache) failed")?;
                            cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                            logits
                        }
                    };
                    check_cancelled(Some(&cancel))?;
                    // (#1082) kt-native logits — next-token store is kt; no bridge.
                    let registration = runner_guard.completed_prompt_registration(
                        &prompt_tokens,
                        &*block_table,
                        &*linear_state,
                        block_size,
                        Some(PagedPrefixNextToken::Logits(logits.clone())),
                    )?;
                    let mut extra_registrations = Vec::new();
                    if let Some(reg) = build_extended_registration(
                        &prompt_tokens,
                        &[],
                        &*block_table,
                        block_size,
                        prefill_split_snapshot,
                    ) {
                        extra_registrations.push(reg);
                    }
                    let next_token = sample_first_decode_token(&logits, &params)?;
                    Ok((
                        next_token,
                        registration,
                        extra_registrations,
                        Some(logits),
                    ))
                })();
                let synchronized = runner_guard
                    .synchronize_external_yield("prefix streaming prefill and first-token sample");
                drop(runner_guard);
                match synchronized {
                    Ok(()) => result,
                    Err(err) => {
                        std::mem::forget(result);
                        Err(err)
                    }
                }
            },
        )?;
        let (next_token, registration, extra_registrations, logits_keepalive) = match prepared {
            Ok(prepared) => prepared,
            Err(err) => {
                if backend_health.snapshot().quarantined {
                    std::mem::forget(ownership);
                } else {
                    free_allocated(&ownership.allocated_blocks);
                }
                return Err(err);
            }
        };
        drop(logits_keepalive);
        let ThreadedPrefillOwnership {
            worker_lifetime,
            post_decode,
            block_table,
            allocated_blocks,
            linear_state,
        } = ownership;
        let mut linear_state =
            linear_state.expect("successful prefix prefill retained linear state");

        let (tx, rx) = mpsc::channel();
        let seq_len = prompt_tokens.len();
        let runner_for_thread = runner_lock;
        let pc_for_thread = paged_cache;
        let decode_batcher_for_thread = decode_batcher;
        let block_table_for_thread = block_table.clone();
        let cleanup = PrefixCachedStreamingCleanup {
            registration,
            extra_registrations,
            allocated_blocks: allocated_blocks.clone(),
        };
        let backend_health_for_thread = backend_health.clone();
        let (settled_tx, settled_rx) = mpsc::channel();

        let spawn_result = std::thread::Builder::new()
            .name("kiln-stream-decode-prefix".to_string())
            .spawn(move || {
                let worker_lifetime = worker_lifetime;
                let quarantined = run_prefix_cached_stream_worker(
                    tx,
                    move |tx| {
                        let runner_guard = match runner_for_thread.read() {
                            Ok(guard) => guard,
                            Err(err) => {
                                return PrefixStreamDecodeOutcome::Quarantined(format!(
                                    "failed to acquire runner read lock in decode thread: {err}"
                                ));
                            }
                        };
                        if let Err(err) = runner_guard.ensure_backend_healthy() {
                            return PrefixStreamDecodeOutcome::Quarantined(err.to_string());
                        }
                        let result = runner_guard.run_stream_decode_loop_with_first(
                            tx,
                            next_token,
                            seq_len,
                            &params,
                            pc_for_thread.as_ref(),
                            &block_table_for_thread,
                            &mut linear_state,
                            decode_batcher_for_thread.as_deref(),
                            Some(&cancel),
                        );
                        match runner_guard.ensure_backend_healthy() {
                            Ok(()) => PrefixStreamDecodeOutcome::Settled(result),
                            Err(err) => PrefixStreamDecodeOutcome::Quarantined(format!(
                                "backend became unhealthy during prefix streaming decode: {err:#}"
                            )),
                        }
                    },
                    post_decode,
                    cleanup,
                    &backend_health_for_thread,
                );
                if quarantined {
                    // Unknown completion means exclusive GPU mutation must
                    // remain blocked for the lifetime of this process.
                    std::mem::forget(worker_lifetime);
                } else {
                    drop(worker_lifetime);
                }
                let _ = settled_tx.send(());
            });

        if let Err(err) = spawn_result {
            free_allocated(&allocated_blocks);
            return Err(anyhow::anyhow!(
                "failed to spawn streaming decode thread: {err}"
            ));
        }

        Ok(ThreadedStreamingOutput {
            receiver: rx,
            settled: settled_rx,
        })
    }

    /// Unavailable high-level paged speculative streaming entry point.
    ///
    /// Currently returns a stable fail-closed error before inspecting inputs.
    pub fn generate_streaming_paged_speculative_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        spec_config: &SpeculativeConfig,
        cancel: Option<&CancelHandle>,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        ensure_speculative_generation_available()?;
        check_cancelled(cancel)?;
        if params.thinking_budget.is_some() {
            anyhow::ensure!(
                cancel.is_none(),
                "cancellable thinking-budget streams must use single-token decode"
            );
            return self.generate_streaming_paged_shared_tokens(
                prompt_tokens,
                params,
                block_manager,
                paged_cache,
            );
        }
        anyhow::ensure!(
            params.temperature == 0.0,
            "paged skip-layer speculative streaming is greedy-only"
        );
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = self.generate_from_tokens_streaming_paged_speculative_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            spec_config,
            cancel,
        );

        reservation.release_after_settlement(
            self,
            "streaming speculative shared KV release",
            result,
        )
    }

    fn generate_from_tokens_streaming_paged_shared(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let cuda_graph_enabled = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?
            .is_enabled();
        if cuda_graph_enabled {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let pc_guard = lock_paged_cache(paged_cache)?;
            return self.generate_from_tokens_streaming_paged_locked(
                prompt_tokens,
                params,
                &mut bm_guard,
                pc_guard,
            );
        }

        let max_total = prompt_tokens.len() + params.max_tokens;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = self.generate_from_tokens_streaming_paged_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
        );

        reservation.release_after_settlement(self, "streaming shared KV release", result)
    }

    fn generate_from_tokens_streaming_paged_interleaved_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
    ) -> Result<PrefixCachedStreamingOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = {
            let bm_guard = lock_block_manager(block_manager)?;
            bm_guard.block_size()
        };

        let cached_prefix = cached_prefix.filter(|prefix| {
            prefix.cached_tokens > 0
                && prefix.cached_tokens < prompt_tokens.len()
                && prefix.cached_tokens % block_size == 0
                && prefix.block_ids.len() == prefix.cached_tokens / block_size
        });

        let cached_blocks = cached_prefix
            .as_ref()
            .map(|prefix| prefix.block_ids.as_slice())
            .unwrap_or(&[]);

        let max_total = prompt_tokens.len() + params.max_tokens;
        let total_blocks = Self::blocks_needed(max_total, block_size);
        let additional_blocks_needed = total_blocks.saturating_sub(cached_blocks.len());
        let allocated_blocks = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            bm_guard
                .allocate(additional_blocks_needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let block_table = append_prefix_block_table(cached_blocks, &allocated_blocks);

        let result = self.generate_from_tokens_streaming_paged_interleaved_with_prefix_blocks(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            cached_prefix,
            block_size,
        );

        match result {
            Ok(mut output) => {
                output.allocated_blocks = allocated_blocks;
                Ok(output)
            }
            Err(err) => {
                if let Err(sync_err) =
                    self.synchronize_external_yield("direct streaming failure cleanup")
                {
                    return Err(sync_err.context(format!(
                        "direct streaming generation also failed before synchronization: {err:#}"
                    )));
                }
                if !allocated_blocks.is_empty() {
                    let mut bm_guard = lock_block_manager(block_manager)?;
                    bm_guard.free_all(&allocated_blocks);
                }
                Err(err)
            }
        }
    }

    fn generate_from_tokens_streaming_paged_interleaved_with_prefix_blocks(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cached_prefix: Option<PagedPrefixReuse>,
        block_size: usize,
    ) -> Result<PrefixCachedStreamingOutput> {
        let cached_tokens = cached_prefix
            .as_ref()
            .map(|prefix| prefix.cached_tokens)
            .unwrap_or(0);
        let mut linear_state = match cached_prefix {
            Some(prefix) => prefix.linear_state,
            None => self.new_linear_state()?,
        };

        let prefill_tokens = &prompt_tokens[cached_tokens..];
        anyhow::ensure!(
            !prefill_tokens.is_empty(),
            "streaming prefix cache hit must leave at least one suffix token"
        );

        let split_pos =
            strict_prompt_prefix_split_pos(prompt_tokens.len(), cached_tokens, block_size);
        let mut prefill_split_snapshot: Option<RollingPrefixSnapshot> = None;
        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if self.streaming_prefill.enabled_for(prefill_tokens.len()) {
                if let Some(split_pos) = split_pos {
                    let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                    let _ = model_forward_paged_streaming_with_policy(
                        &*self.backend,
                        head_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        self.streaming_prefill,
                    )
                    .context("prefill forward pass (streaming paged prefix cache head)")?;
                    prefill_split_snapshot = self
                        .authoritative_prefix_snapshot(
                            &linear_state,
                            "streaming-prefix-cache-split",
                            split_pos,
                        )
                        .context("snapshot linear state at streaming prefix-cache split")?
                        .map(|linear_state| RollingPrefixSnapshot {
                            position: split_pos,
                            linear_state,
                        });

                    let tail_tokens = &prompt_tokens[split_pos..];
                    model_forward_paged_streaming_with_policy(
                        &*self.backend,
                        tail_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        split_pos,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        self.streaming_prefill,
                    )
                    .context("prefill forward pass (streaming paged prefix cache tail)")?
                } else {
                    model_forward_paged_streaming_with_policy(
                        &*self.backend,
                        prefill_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        self.streaming_prefill,
                    )
                    .context("prefill forward pass (streaming paged prefix cache) failed")?
                }
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prefill_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    cached_tokens,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged prefix cache) failed")?
            }
        };
        // (#1082) kt-native logits — next-token store + sampler are both kt;
        // no candle bridge.

        let registration = self.completed_prompt_registration(
            prompt_tokens,
            block_table,
            &linear_state,
            block_size,
            Some(PagedPrefixNextToken::Logits(logits.clone())),
        )?;
        let mut extra_registrations = Vec::new();
        if let Some(reg) = build_extended_registration(
            prompt_tokens,
            &[],
            block_table,
            block_size,
            prefill_split_snapshot,
        ) {
            extra_registrations.push(reg);
        }

        let receiver = self.stream_decode_from_prefill_logits(
            logits,
            prompt_tokens.len(),
            params,
            paged_cache,
            block_table,
            &mut linear_state,
        )?;

        // Legacy synchronous path: receiver is fully populated before return,
        // no decode thread is alive, so the API layer is free to call
        // bm.free_all on the same call frame. No rendezvous channel needed.
        Ok(PrefixCachedStreamingOutput {
            receiver,
            registration,
            extra_registrations,
            allocated_blocks: Vec::new(),
        })
    }

    fn generate_from_tokens_streaming_paged_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if self.streaming_prefill.enabled_for(prompt_tokens.len()) {
                model_forward_paged_streaming_with_policy(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    self.streaming_prefill,
                )
                .context("prefill forward pass (paged, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged) failed")?
            }
        };
        // (#1082) forward returns kt logits; sampler entry is kt now — no bridge.

        self.stream_decode_from_prefill_logits(
            logits,
            prompt_tokens.len(),
            params,
            paged_cache,
            block_table,
            &mut linear_state,
        )
    }

    fn stream_decode_from_prefill_logits(
        &self,
        // (#1082) kt-native logits — sample_first_decode_token is kt.
        logits: kiln_tensor::Tensor,
        seq_len: usize,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        linear_state: &mut LinearAttentionState,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel();
        // Sample the first decode token from prefill logits and run the loop on
        // the calling thread. Used by tests and the synchronous (non-spawned)
        // entry points. The receiver is fully populated by the time we return.
        // Threaded callers should use [`run_stream_decode_loop_with_first`]
        // directly so they can sample the first token before spawning.
        let sampled = sample_first_decode_token(&logits, params);
        if let Err(sync_err) = self.synchronize_external_yield("direct streaming prefill") {
            std::mem::forget(logits);
            std::mem::forget(sampled);
            return Err(sync_err);
        }
        let next_token = sampled?;
        let done = self.run_stream_decode_loop_with_first(
            &tx,
            next_token,
            seq_len,
            params,
            paged_cache,
            block_table,
            linear_state,
            None,
            None,
        )?;
        if let Some(done) = done {
            let _ = tx.send(StreamEvent::Done(done));
        }
        Ok(rx)
    }

    /// Streaming decode loop body, sending each generated token to `tx` as it
    /// is produced. The `next_token` argument is the first token to emit (the
    /// argmax/sample of the prefill logits). The caller owns `tx` so that
    /// threaded callers can spawn the loop and return the receiver to the
    /// async layer immediately, instead of waiting for `max_tokens` decode
    /// steps before the receiver becomes observable.
    pub(crate) fn run_stream_decode_loop_with_first(
        &self,
        tx: &mpsc::Sender<StreamEvent>,
        mut next_token: TokenId,
        mut seq_len: usize,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        linear_state: &mut LinearAttentionState,
        decode_batcher: Option<&DecodeBatcher>,
        cancel: Option<&CancelHandle>,
    ) -> Result<Option<StreamDone>> {
        let rocm_owner = RocmDecodeOwnerLease::new(&self.rocm_graph, &self.backend_health);
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);

        for _step in 0..params.max_tokens {
            check_cancelled(cancel)?;
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            if self.should_stop_on_eos(params, next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                next_token,
            )? {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    finish_reason = reason;
                    break;
                }
                StreamTokenDisposition::ReceiverDropped => {
                    tracing::debug!(
                        event = "direct_decode_receiver_dropped",
                        row_id = rocm_owner.row_id(),
                        generated_tokens = generated_tokens.len(),
                        "direct_decode_receiver_dropped"
                    );
                    return Ok(None);
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let skip_gdn_state_readback = skip_final_gdn_state_readback_enabled()
                && generated_tokens.len() + 1 >= params.max_tokens;
            let decode_result = self.decode_next_token_paged_interleaved_or_batched(
                params,
                next_token,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                step_seed,
                decode_batcher,
                &generated_tokens,
                rocm_owner.row_id(),
                skip_gdn_state_readback,
            );
            self.synchronize_external_yield("direct streaming decode step")?;
            next_token = decode_result?;
            seq_len += 1;
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens)?;
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        Ok(Some(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }))
    }

    fn generate_from_tokens_streaming_paged_speculative_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        spec_config: &SpeculativeConfig,
        cancel: Option<&CancelHandle>,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        check_cancelled(cancel)?;
        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if self.streaming_prefill.enabled_for(prompt_tokens.len()) {
                model_forward_paged_streaming_with_progress_and_policy(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    cancel,
                    self.streaming_prefill,
                )
                .context("prefill forward pass (streaming paged skip-layer, streaming) failed")?
            } else {
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (streaming paged skip-layer) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prompt_tokens.len() as u64);
                }
                logits
            }
        };
        check_cancelled(cancel)?;
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut base_pos = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);
        let mut last_token = greedy_sample(&logits)?;

        loop {
            check_cancelled(cancel)?;
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.should_stop_on_eos(params, last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                last_token,
            )? {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    finish_reason = reason;
                    break;
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: spec_config.num_speculative_tokens.min(remaining),
                draft_layers: spec_config.draft_layers,
            };

            let result = {
                let pc_guard = lock_paged_cache(paged_cache)?;
                speculative_decode_step_paged_greedy(
                    &*self.backend,
                    last_token,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    base_pos,
                    &mut linear_state,
                    &mut draft_linear_state,
                    &effective_config,
                    params,
                    self.eos_token_ids_for(params),
                    self.active_lora.as_ref(),
                )
                .context("streaming paged skip-layer speculative decode step failed")?
            };
            base_pos += result.base_advance;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut gate,
                    &mut generated_tokens,
                    token,
                )? {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        finish_reason = reason;
                        break;
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();
            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens)?;
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }));

        Ok(rx)
    }

    /// Legacy synchronous generation using a mutable paged KV block manager.
    ///
    /// Despite returning a receiver, this method performs prefill and decode on
    /// the calling thread and the receiver is already fully populated when it
    /// returns. New serving integrations should use
    /// [`Self::spawn_streaming_paged_shared_tokens`], which exposes live token
    /// delivery, cancellation, and explicit worker settlement.
    #[deprecated(
        note = "use spawn_streaming_paged_shared_tokens for live streaming and explicit settlement"
    )]
    pub fn generate_streaming_paged(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        self.generate_from_tokens_streaming_paged_locked(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    fn generate_from_tokens_streaming_paged_locked(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        self.ensure_backend_healthy()?;
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = block_manager.block_size();
        let max_total = prompt_tokens.len() + params.max_tokens;

        let num_blocks = Self::blocks_needed(max_total, block_size);
        let allocated_blocks = block_manager
            .allocate(num_blocks)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let reservation = MutableBlockReservation {
            block_manager,
            block_ids: allocated_blocks,
        };
        let block_table = reservation.block_table();

        let (tx, rx) = mpsc::channel();
        let mut owners = LegacyMutablePagedStreamOwners::new();
        let execution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<mpsc::Receiver<StreamEvent>> {
                owners.linear_state = Some(self.new_linear_state()?);

                // The immutable startup policy selects tiled or monolithic prefill.
                owners.prefill_logits = Some(
                    if self.streaming_prefill.enabled_for(prompt_tokens.len()) {
                        model_forward_paged_streaming_with_policy(
                            &*self.backend,
                            prompt_tokens,
                            &self.weights,
                            &self.config,
                            paged_cache,
                            &block_table,
                            0,
                            owners.linear_state.as_mut(),
                            self.active_lora.as_ref(),
                            self.streaming_prefill,
                        )
                    } else {
                        model_forward_paged_last_token(
                            &*self.backend,
                            prompt_tokens,
                            &self.weights,
                            &self.config,
                            paged_cache,
                            &block_table,
                            0,
                            owners.linear_state.as_mut(),
                            self.active_lora.as_ref(),
                            None,
                        )
                    }
                    .context("prefill forward pass (paged) failed")?,
                );
                // (#1082) forward returns kt logits; sampler is kt -- no bridge.

                let mut seq_len = prompt_tokens.len();
                let mut generated_tokens: Vec<TokenId> = Vec::new();
                let mut step_seed = params.seed;
                let mut finish_reason = FinishReason::MaxTokens;
                let mut gate = StreamTextGate::new(&params.stop);

                // Preserve the legacy whole-request CUDA graph lock and replay path.
                owners.cuda_graph = Some(
                    self.cuda_graph
                        .lock()
                        .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?,
                );

                let mut next_token = if params.is_effectively_greedy() {
                    greedy_sample(
                        owners
                            .prefill_logits
                            .as_ref()
                            .expect("prefill logits initialized"),
                    )?
                } else {
                    sample_step(
                        owners
                            .prefill_logits
                            .as_ref()
                            .expect("prefill logits initialized"),
                        params,
                        step_seed,
                        &[],
                    )?
                };

                for _step in 0..params.max_tokens {
                    if let Some(s) = step_seed.as_mut() {
                        *s = s.wrapping_add(1);
                    }

                    next_token = params.apply_thinking_budget(&generated_tokens, next_token);
                    if self.should_stop_on_eos(params, next_token) {
                        finish_reason = FinishReason::Eos;
                        break;
                    }

                    match emit_stream_token(
                        &tx,
                        &self.tokenizer,
                        &mut gate,
                        &mut generated_tokens,
                        next_token,
                    )? {
                        StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                        StreamTokenDisposition::Finished(reason) => {
                            let _ = tx.send(StreamEvent::Done(StreamDone {
                                finish_reason: reason,
                                completion_tokens: generated_tokens.len(),
                                trailing_text: String::new(),
                            }));
                            return Ok(rx);
                        }
                        StreamTokenDisposition::Continue => {}
                    }

                    if generated_tokens.len() >= params.max_tokens {
                        break;
                    }

                    next_token = if params.is_effectively_greedy()
                        && greedy_token_decode_enabled(self.backend.as_ref())
                    {
                        let linear_state_for_graph = if self.has_linear_attention_layers() {
                            owners.linear_state.as_mut()
                        } else {
                            None
                        };
                        match self
                            .decode_next_token_paged_greedy_metal_graph(
                                next_token,
                                paged_cache,
                                &block_table,
                                seq_len,
                                linear_state_for_graph,
                            )
                            .context("greedy Metal graph decode forward pass (paged) failed")?
                        {
                            Some(token) => {
                                seq_len += 1;
                                token
                            }
                            None => {
                                let token = model_forward_paged_next_token_greedy(
                                    &*self.backend,
                                    next_token,
                                    &self.weights,
                                    &self.config,
                                    paged_cache,
                                    &block_table,
                                    seq_len,
                                    owners.linear_state.as_mut(),
                                    self.active_lora.as_ref(),
                                    None,
                                )
                                .context("decode forward pass (paged greedy) failed")?;
                                seq_len += 1;
                                token
                            }
                        }
                    } else {
                        owners.pending_decode_logits = Some(
                            owners
                                .cuda_graph
                                .as_mut()
                                .expect("CUDA graph runner initialized")
                                .decode_step_paged(
                                    &*self.backend,
                                    next_token,
                                    &self.weights,
                                    &self.config,
                                    paged_cache,
                                    &block_table,
                                    seq_len,
                                    owners
                                        .linear_state
                                        .as_mut()
                                        .expect("linear state initialized"),
                                    self.active_lora.as_ref(),
                                    None,
                                )
                                .context("decode forward pass (paged) failed")?,
                        );
                        seq_len += 1;
                        sample_step(
                            owners
                                .pending_decode_logits
                                .as_ref()
                                .expect("decode logits initialized"),
                            params,
                            step_seed,
                            &generated_tokens,
                        )?
                    };
                }

                let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens)?;
                let (finish_reason, gate_trailing) = match late_stop {
                    Some(stop) => (FinishReason::StopSequence(stop), String::new()),
                    None => (finish_reason, gate_trailing),
                };
                let _ = tx.send(StreamEvent::Done(StreamDone {
                    finish_reason,
                    completion_tokens: generated_tokens.len(),
                    trailing_text: gate_trailing,
                }));
                Ok(rx)
            },
        ));
        let result = match execution {
            Ok(result) => result,
            Err(_) => {
                let reason = "legacy mutable paged streaming execution panicked; backend completion and request ownership are unknown";
                self.backend_health.quarantine(reason);
                Err(anyhow::anyhow!(reason))
            }
        };

        reservation.release_after_settlement(
            self,
            "legacy mutable paged streaming release",
            SettlementOutcome { result, owners },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::FallbackPolicy;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    const DECODE_FALLBACK_ENV: &[&str] = &[
        "KILN_DECODE_HOT_PATH_DEBUG_FALLBACK",
        "KILN_METAL_DECODE_BATCH_GENERIC_FALLBACK",
        "KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK",
        "KILN_ROCM_DECODE_BATCH_GENERIC_FALLBACK",
    ];

    #[test]
    fn model_runner_runtime_options_default_to_eager_rocm_execution() {
        let options = ModelRunnerRuntimeOptions::default();
        assert_eq!(options, ModelRunnerRuntimeOptions::eager_only());
        assert_eq!(
            options.rocm_graph,
            RocmGraphExecutionPolicy::disabled(),
            "lazy ROCm capture must require an explicit product policy"
        );
    }

    #[test]
    fn completed_row_preserves_only_resident_batched_state_capacity() {
        let cached_rows = [11, 12, 13];
        assert!(completed_row_invalidates_batched_state_cache(
            &cached_rows,
            12,
            false
        ));
        assert!(!completed_row_invalidates_batched_state_cache(
            &cached_rows,
            12,
            true
        ));
        assert!(!completed_row_invalidates_batched_state_cache(
            &cached_rows,
            99,
            false
        ));
    }

    #[test]
    fn batched_state_cache_counters_track_overlapping_leases() {
        let counters = BatchedStateCacheCounters::default();
        counters.acquire_lease();
        counters.acquire_lease();
        counters
            .take_miss_while_leased_count
            .fetch_add(1, Ordering::Relaxed);

        let overlapping = counters.snapshot();
        assert_eq!(overlapping.active_leases, 2);
        assert_eq!(overlapping.max_active_leases, 2);
        assert_eq!(overlapping.take_miss_while_leased_count, 1);

        counters.release_lease();
        counters.release_lease();
        let drained = counters.snapshot();
        assert_eq!(drained.active_leases, 0);
        assert_eq!(drained.max_active_leases, 2);
    }

    #[test]
    fn speculative_generation_unavailable_reason_is_stable() {
        let error = ensure_speculative_generation_available()
            .expect_err("unqualified speculative generation must fail closed");
        assert_eq!(
            error.to_string(),
            "speculative generation is disabled pending cancellation-safe owner settlement and local accelerator qualification"
        );
    }

    #[test]
    fn inference_memory_binding_accepts_only_exact_or_vulkan_host_weights() {
        let binding = InferenceMemoryRuntime {
            device: kiln_tensor::Device::Vulkan(0),
            selector: kiln_memory::VramProbeSelector::LinuxDrm {
                index: 0,
                vendor: None,
            },
            effective_capacity_bytes: 16 * 1024 * 1024 * 1024,
            governor: kiln_memory::GovernorConfig::default(),
        };
        assert!(binding.is_weight_device_compatible(kiln_tensor::Device::Vulkan(0)));
        assert!(binding.is_weight_device_compatible(kiln_tensor::Device::Cpu));
        assert!(!binding.is_weight_device_compatible(kiln_tensor::Device::Cuda(0)));
        assert!(!binding.is_weight_device_compatible(kiln_tensor::Device::Vulkan(1)));
    }

    fn empty_prefix_stream_cleanup() -> PrefixCachedStreamingCleanup {
        PrefixCachedStreamingCleanup {
            registration: None,
            extra_registrations: Vec::new(),
            allocated_blocks: Vec::new(),
        }
    }

    #[test]
    fn threaded_prefill_panic_fence_quarantines_every_request_owner() {
        struct DropProbe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        struct PrefillOwners {
            _worker_lifetime: DropProbe,
            _allocation_metadata: DropProbe,
            _prefix_lease: DropProbe,
        }

        let worker_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let allocation_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let lease_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let backend_health = BackendHealthHandle::default();
        let result = run_threaded_prefill_with_panic_fence(
            &backend_health,
            "injected threaded prefill",
            PrefillOwners {
                _worker_lifetime: DropProbe(Arc::clone(&worker_drops)),
                _allocation_metadata: DropProbe(Arc::clone(&allocation_drops)),
                _prefix_lease: DropProbe(Arc::clone(&lease_drops)),
            },
            |_| -> Result<()> { panic!("injected prefill panic") },
        );

        let error = match result {
            Ok(_) => panic!("prefill panic must become an error"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("injected threaded prefill panicked")
        );
        assert_eq!(worker_drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            allocation_drops.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(lease_drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        let snapshot = backend_health.snapshot();
        assert!(snapshot.quarantined);
        assert_eq!(
            snapshot.reason.as_deref(),
            Some(
                "injected threaded prefill panicked; backend completion and request ownership are unknown"
            )
        );
    }

    #[test]
    fn prefix_stream_worker_finalizes_before_terminal_event() {
        let finalized = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let finalized_in_cleanup = Arc::clone(&finalized);
        let backend_health = BackendHealthHandle::default();
        let (tx, rx) = mpsc::channel();
        run_prefix_cached_stream_worker(
            tx,
            |_| {
                PrefixStreamDecodeOutcome::Settled(Ok(Some(StreamDone {
                    finish_reason: FinishReason::Eos,
                    completion_tokens: 3,
                    trailing_text: "tail".to_string(),
                })))
            },
            move |_| {
                finalized_in_cleanup.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            },
            empty_prefix_stream_cleanup(),
            &backend_health,
        );

        let StreamEvent::Done(done) = rx.recv().expect("terminal event") else {
            panic!("expected terminal event");
        };
        assert!(finalized.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(done.completion_tokens, 3);
        assert_eq!(done.trailing_text, "tail");
    }

    #[test]
    fn shared_block_reservation_retains_pages_when_settlement_fails() -> anyhow::Result<()> {
        #[derive(Debug)]
        struct DropProbe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let result_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let block_manager = Mutex::new(BlockManager::new(4, 4));
        let block_ids = block_manager.lock().unwrap().allocate(1)?;
        let reservation = SharedBlockReservation {
            block_manager: &block_manager,
            block_ids,
        };

        let error = reservation
            .release_after_settlement_with(
                "injected shared reservation settlement",
                Ok(DropProbe(Arc::clone(&result_drops))),
                || anyhow::bail!("injected sync failure"),
            )
            .expect_err("settlement failure must fail the request");
        assert!(error.to_string().contains("injected sync failure"));
        assert_eq!(result_drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(block_manager.lock().unwrap().num_used(), 1);
        Ok(())
    }

    #[test]
    fn mutable_block_reservation_settles_receiver_drop_outcome_before_release() -> Result<()> {
        #[derive(Debug, PartialEq, Eq)]
        enum TestStreamExit {
            ReceiverDropped,
        }

        struct DropProbe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let owner_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut block_manager = BlockManager::new(4, 4);
        let block_ids = block_manager.allocate(1)?;
        let reservation = MutableBlockReservation {
            block_manager: &mut block_manager,
            block_ids,
        };

        let exit = reservation.release_after_settlement_with(
            "injected mutable reservation success",
            SettlementOutcome {
                result: Ok(TestStreamExit::ReceiverDropped),
                owners: DropProbe(Arc::clone(&owner_drops)),
            },
            || Ok(()),
            |_| {},
        )?;

        assert_eq!(exit, TestStreamExit::ReceiverDropped);
        assert_eq!(owner_drops.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(block_manager.num_used(), 0);
        Ok(())
    }

    #[test]
    fn mutable_block_reservation_releases_after_settled_execution_error() -> Result<()> {
        struct DropProbe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let owner_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut block_manager = BlockManager::new(4, 4);
        let block_ids = block_manager.allocate(1)?;
        let reservation = MutableBlockReservation {
            block_manager: &mut block_manager,
            block_ids,
        };

        let error = reservation
            .release_after_settlement_with(
                "injected mutable reservation execution error",
                SettlementOutcome::<(), _> {
                    result: Err(anyhow::anyhow!("injected execution failure")),
                    owners: DropProbe(Arc::clone(&owner_drops)),
                },
                || Ok(()),
                |_| {},
            )
            .expect_err("a settled execution error must remain an error");

        assert!(error.to_string().contains("injected execution failure"));
        assert_eq!(owner_drops.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(block_manager.num_used(), 0);
        Ok(())
    }

    #[test]
    fn mutable_block_reservation_retains_device_outcome_but_releases_coordination_on_sync_failure()
    -> Result<()> {
        #[derive(Debug)]
        struct DropProbe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        struct TestOwners {
            device: DropProbe,
            coordination: Option<DropProbe>,
            quarantined: Arc<std::sync::atomic::AtomicBool>,
        }

        let device_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let coordination_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let quarantined = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut block_manager = BlockManager::new(4, 4);
        let block_ids = block_manager.allocate(1)?;
        let reservation = MutableBlockReservation {
            block_manager: &mut block_manager,
            block_ids,
        };

        let error = reservation
            .release_after_settlement_with(
                "injected mutable reservation settlement failure",
                SettlementOutcome {
                    result: Ok(DropProbe(Arc::clone(&result_drops))),
                    owners: TestOwners {
                        device: DropProbe(Arc::clone(&device_drops)),
                        coordination: Some(DropProbe(Arc::clone(&coordination_drops))),
                        quarantined: Arc::clone(&quarantined),
                    },
                },
                || anyhow::bail!("injected sync failure"),
                |owners| {
                    owners
                        .quarantined
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    drop(owners.coordination.take());
                    let _retain_device_owner = &owners.device;
                },
            )
            .expect_err("failed settlement must fail the request");

        assert!(error.to_string().contains("injected sync failure"));
        assert!(quarantined.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(
            coordination_drops.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(device_drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(result_drops.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(block_manager.num_used(), 1);
        Ok(())
    }

    #[test]
    fn mutable_block_reservation_sync_panic_latches_quarantine() {
        let backend_health = BackendHealthHandle::default();
        let error = catch_external_yield_sync_panic(
            &backend_health,
            "injected mutable reservation sync panic",
            || -> Result<()> { panic!("injected synchronization panic") },
        )
        .expect_err("synchronization panic must become a quarantine error");

        assert!(
            error
                .to_string()
                .contains("backend synchronization panicked")
        );
        let snapshot = backend_health.snapshot();
        assert!(snapshot.quarantined);
        assert_eq!(
            snapshot.reason.as_deref(),
            Some("backend synchronization panicked at injected mutable reservation sync panic")
        );
    }

    #[test]
    fn prefix_stream_worker_finalizes_on_disconnect_and_decode_error() {
        for decode_error in [false, true] {
            let finalized = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let finalized_in_cleanup = Arc::clone(&finalized);
            let backend_health = BackendHealthHandle::default();
            let (tx, rx) = mpsc::channel();
            let rx = if decode_error {
                Some(rx)
            } else {
                drop(rx);
                None
            };
            run_prefix_cached_stream_worker(
                tx,
                move |_| {
                    if decode_error {
                        PrefixStreamDecodeOutcome::Settled(Err(anyhow::anyhow!(
                            "injected decode failure"
                        )))
                    } else {
                        PrefixStreamDecodeOutcome::Settled(Ok(None))
                    }
                },
                move |_| {
                    finalized_in_cleanup.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(())
                },
                empty_prefix_stream_cleanup(),
                &backend_health,
            );
            assert_eq!(finalized.load(std::sync::atomic::Ordering::SeqCst), 1);
            if decode_error {
                assert!(matches!(rx.unwrap().recv(), Ok(StreamEvent::Error(_))));
            }
            assert!(!backend_health.snapshot().quarantined);
        }
    }

    #[test]
    fn prefix_stream_worker_quarantines_cleanup_after_decode_panic() {
        struct DropProbe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let finalized = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let finalized_in_cleanup = Arc::clone(&finalized);
        let dropped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cleanup_owner = DropProbe(Arc::clone(&dropped));
        let backend_health = BackendHealthHandle::default();
        let (tx, rx) = mpsc::channel();
        run_prefix_cached_stream_worker(
            tx,
            |_| -> PrefixStreamDecodeOutcome { panic!("injected decode panic") },
            move |_| {
                let _keep_alive = &cleanup_owner;
                finalized_in_cleanup.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            },
            empty_prefix_stream_cleanup(),
            &backend_health,
        );
        assert_eq!(finalized.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(dropped.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(matches!(rx.recv(), Ok(StreamEvent::Error(_))));
        assert!(backend_health.snapshot().quarantined);
    }

    #[test]
    fn prefix_stream_worker_latches_quarantine_before_error_and_leaks_decode_state() {
        struct DropProbe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let backend_health = BackendHealthHandle::default();
        let health_for_worker = backend_health.clone();
        let dropped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let probe = DropProbe(Arc::clone(&dropped));
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            run_prefix_cached_stream_worker(
                tx,
                move |_| {
                    let _keep_alive = &probe;
                    PrefixStreamDecodeOutcome::Quarantined("injected unknown completion".into())
                },
                |_| Ok(()),
                empty_prefix_stream_cleanup(),
                &health_for_worker,
            )
        });

        assert!(matches!(rx.recv(), Ok(StreamEvent::Error(_))));
        let snapshot = backend_health.snapshot();
        assert!(snapshot.quarantined);
        assert_eq!(
            snapshot.reason.as_deref(),
            Some("injected unknown completion")
        );
        assert_eq!(dropped.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(worker.join().unwrap());
    }

    #[test]
    fn external_yield_sync_stats_are_bounded_by_static_boundary() {
        let health = BackendHealthHandle::default();
        health.record_external_yield_sync(
            "batched decode step",
            std::time::Duration::from_millis(40),
            false,
        );
        health.record_external_yield_sync(
            "batched decode step",
            std::time::Duration::from_millis(125),
            true,
        );

        assert_eq!(
            health.external_yield_sync_stats(),
            vec![ExternalYieldSyncStats {
                boundary: "batched decode step".to_string(),
                calls: 2,
                failures: 1,
                total_micros: 165_000,
                max_micros: 125_000,
                slow_calls: 1,
            }]
        );
    }

    #[test]
    fn backend_quarantine_latches_first_reason_across_clones() {
        let health = BackendHealthHandle::default();
        let clone = health.clone();
        clone.quarantine("first unknown completion");
        health.quarantine("later failure");

        assert_eq!(
            health.snapshot(),
            BackendHealthSnapshot {
                quarantined: true,
                reason: Some("first unknown completion".to_string()),
            }
        );
        assert!(clone.ensure_healthy().is_err());
    }

    #[test]
    fn streaming_worker_gpu_owner_blocks_writer_through_cleanup() {
        let execution_lock = Arc::new(std::sync::RwLock::new(()));
        let worker_lock = Arc::clone(&execution_lock);
        let backend_health = BackendHealthHandle::default();
        let (guard_ready_tx, guard_ready_rx) = mpsc::channel();
        let (allow_decode_tx, allow_decode_rx) = mpsc::channel();
        let (cleanup_started_tx, cleanup_started_rx) = mpsc::channel();
        let (allow_cleanup_tx, allow_cleanup_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _guard = worker_lock.read().unwrap();
            guard_ready_tx.send(()).unwrap();
            let (tx, _rx) = mpsc::channel();
            run_prefix_cached_stream_worker(
                tx,
                move |_| {
                    allow_decode_rx.recv().unwrap();
                    PrefixStreamDecodeOutcome::Settled(Ok(None))
                },
                move |_| {
                    cleanup_started_tx.send(()).unwrap();
                    allow_cleanup_rx.recv().unwrap();
                    Ok(())
                },
                empty_prefix_stream_cleanup(),
                &backend_health,
            );
        });
        guard_ready_rx.recv().unwrap();

        assert!(
            execution_lock.try_write().is_err(),
            "writer acquired while decode was blocked"
        );

        allow_decode_tx.send(()).unwrap();
        cleanup_started_rx.recv().unwrap();
        assert!(
            execution_lock.try_write().is_err(),
            "writer acquired while cleanup was blocked"
        );

        allow_cleanup_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(
            execution_lock.try_write().is_ok(),
            "writer must acquire after worker cleanup"
        );
    }

    #[test]
    fn decode_row_ids_are_process_unique_across_threads() {
        const THREADS: usize = 8;
        const IDS_PER_THREAD: usize = 128;
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..IDS_PER_THREAD)
                        .map(|_| next_decode_row_id())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let ids: Vec<u64> = workers
            .into_iter()
            .flat_map(|worker| {
                worker
                    .join()
                    .expect("decode owner allocator worker panicked")
            })
            .collect();
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();

        assert_eq!(ids.len(), THREADS * IDS_PER_THREAD);
        assert_eq!(unique.len(), ids.len());
        assert!(ids.into_iter().all(|id| id != 0));
    }

    #[test]
    fn decode_row_id_exhaustion_never_wraps_or_reuses_zero() {
        let counter = std::sync::atomic::AtomicU64::new(u64::MAX);
        assert_eq!(allocate_decode_row_id(&counter), u64::MAX);
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);

        let exhausted = std::panic::catch_unwind(|| allocate_decode_row_id(&counter));
        assert!(
            exhausted.is_err(),
            "exhausted owner namespace must fail closed"
        );
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
    const DECODE_BATCHER_ROWWISE_ENV: &[&str] = &["KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY"];

    struct EnvRestore(Vec<(&'static str, Option<String>)>);

    impl EnvRestore {
        fn clear(keys: &[&'static str]) -> Self {
            let prior = keys
                .iter()
                .map(|&key| (key, std::env::var(key).ok()))
                .collect::<Vec<_>>();
            unsafe {
                for &key in keys {
                    std::env::remove_var(key);
                }
            }
            Self(prior)
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            unsafe {
                for (key, value) in &self.0 {
                    if let Some(value) = value {
                        std::env::set_var(key, value);
                    } else {
                        std::env::remove_var(key);
                    }
                }
            }
        }
    }

    #[test]
    fn single_row_hip_graph_preempts_generic_greedy_batch_route() {
        assert_eq!(
            greedy_batch_route(true, false, 1, true),
            GreedyBatchRoute::HipGraph,
            "BF16 single-row ROCm decode must reach the enabled HIP graph runner"
        );
        assert_eq!(
            greedy_batch_route(true, true, 1, true),
            GreedyBatchRoute::HipGraph,
            "FP8 single-row ROCm decode is also graph-capturable"
        );
        assert_eq!(
            greedy_batch_route(true, false, 1, false),
            GreedyBatchRoute::Contiguous,
            "graphs-off and non-ROCm single-row decode must retain the eager path"
        );
        assert_eq!(
            greedy_batch_route(true, false, 4, true),
            GreedyBatchRoute::Contiguous,
            "multi-row greedy decode must retain true batching"
        );
        assert_eq!(
            greedy_batch_route(false, false, 1, true),
            GreedyBatchRoute::Later,
            "sampled single-row decode is handled by the later HIP graph branch"
        );
    }

    #[test]
    fn resident_single_row_greedy_preserves_stable_row_identity() {
        assert!(should_use_unidentified_single_row_greedy_route(
            1, false, true
        ));
        assert!(should_use_unidentified_single_row_greedy_route(
            1, true, false
        ));
        assert!(!should_use_unidentified_single_row_greedy_route(
            1, true, true
        ));
        assert!(!should_use_unidentified_single_row_greedy_route(
            2, false, true
        ));
    }

    #[derive(Debug)]
    struct NamedTestBackend {
        name: &'static str,
        device: kiln_tensor::Device,
        resident_linear_state: bool,
        resident_recurrent_state: bool,
    }

    impl BackendIdentity for NamedTestBackend {
        fn runtime_name(&self) -> &'static str {
            self.name
        }

        fn runtime_device(&self) -> kiln_tensor::Device {
            self.device
        }

        fn runtime_as_any(&self) -> &dyn std::any::Any {
            &()
        }
    }

    impl StartupBackend for NamedTestBackend {}

    impl crate::backend::ExternalYieldBackend for NamedTestBackend {
        fn runtime_synchronize_external_yield(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    impl crate::backend::AttentionBackend for NamedTestBackend {}

    impl crate::backend::GdnBackend for NamedTestBackend {}

    impl crate::backend::ConvBackend for NamedTestBackend {}

    impl crate::backend::LinearBackend for NamedTestBackend {}

    impl crate::backend::residency::ResidentRegistry for NamedTestBackend {}

    impl crate::backend::ResidencyBackend for NamedTestBackend {
        fn runtime_has_gdn_recurrent_resident_state(&self, _state: &kiln_tensor::Tensor) -> bool {
            self.resident_recurrent_state
        }

        fn runtime_has_linear_attn_gdn_state_kt(&self, _key: kiln_tensor::TensorId) -> bool {
            self.resident_linear_state
        }
    }

    impl crate::backend::SamplingBackend for NamedTestBackend {}

    impl crate::backend::OptimizerBackend for NamedTestBackend {}

    impl crate::backend::PagedKvBackend for NamedTestBackend {}

    impl crate::backend::ReplayBackend for NamedTestBackend {}

    impl crate::backend::TrainingLossBackend for NamedTestBackend {}

    impl BackendRuntime for NamedTestBackend {}

    #[test]
    fn decode_batcher_stats_report_runner_calls_per_token() {
        let stats = DecodeBatcherStats {
            executed_rows: 4,
            runner_calls: 5,
            max_runner_calls_per_token: 2,
            ..DecodeBatcherStats::default()
        };

        assert_eq!(stats.runner_calls_per_token(), Some(1.25));
        assert_eq!(stats.max_runner_calls_per_token, 2);
        assert_eq!(stats.runner_call_budget_per_token(), 2);
        assert!(!stats.runner_call_budget_exceeded());
        assert_eq!(DecodeBatcherStats::default().runner_calls_per_token(), None);

        let exceeded = DecodeBatcherStats {
            max_runner_calls_per_token: 3,
            ..DecodeBatcherStats::default()
        };
        assert!(exceeded.runner_call_budget_exceeded());
    }

    #[test]
    fn test_decode_batcher_default_backend_policy() {
        for (
            backend_name,
            device,
            max_batch,
            wait_micros,
            allow_mixed_seq_lens,
            rowwise_retry_env,
            use_native_sampled_contiguous_decode,
            sampled_contiguous_decode_requires_resident_decode,
            partition_noncontiguous_gdn_kv_tiles,
        ) in [
            (
                "cpu",
                kiln_tensor::Device::Cpu,
                8,
                0,
                false,
                None,
                false,
                false,
                false,
            ),
            (
                "cuda",
                kiln_tensor::Device::Cpu,
                1,
                0,
                false,
                None,
                false,
                false,
                true,
            ),
            (
                "cuda",
                kiln_tensor::Device::Cuda(0),
                1,
                0,
                false,
                None,
                false,
                false,
                true,
            ),
            (
                "metal",
                kiln_tensor::Device::Metal(0),
                8,
                100,
                true,
                None,
                true,
                false,
                false,
            ),
            (
                "vulkan",
                kiln_tensor::Device::Cpu,
                64,
                5_000,
                true,
                Some("KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY"),
                true,
                true,
                false,
            ),
            (
                "vulkan",
                kiln_tensor::Device::Vulkan(0),
                64,
                5_000,
                true,
                Some("KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY"),
                true,
                true,
                false,
            ),
            (
                "rocm",
                kiln_tensor::Device::Rocm(0),
                8,
                0,
                false,
                None,
                false,
                false,
                false,
            ),
        ] {
            let policy = DecodeBatcherPolicy::for_backend(backend_name, device);
            assert!(
                policy.rendezvous_default_enabled,
                "{backend_name} rendezvous enable policy drifted"
            );
            assert_eq!(
                policy.max_batch, max_batch,
                "{backend_name} max batch policy drifted"
            );
            assert_eq!(
                policy.wait_micros, wait_micros,
                "{backend_name} wait policy drifted"
            );
            assert_eq!(
                policy.allow_mixed_seq_lens, allow_mixed_seq_lens,
                "{backend_name} mixed-seq policy drifted"
            );
            assert_eq!(
                policy.rowwise_retry_env, rowwise_retry_env,
                "{backend_name} rowwise retry policy drifted"
            );
            assert_eq!(
                policy.use_native_sampled_contiguous_decode, use_native_sampled_contiguous_decode,
                "{backend_name} sampled contiguous decode policy drifted"
            );
            assert_eq!(
                policy.sampled_contiguous_decode_requires_resident_decode,
                sampled_contiguous_decode_requires_resident_decode,
                "{backend_name} sampled contiguous resident requirement policy drifted"
            );
            assert_eq!(
                policy.partition_noncontiguous_gdn_kv_tiles, partition_noncontiguous_gdn_kv_tiles,
                "{backend_name} GDN KV contiguity partition policy drifted"
            );
        }
    }

    #[test]
    fn decode_batcher_config_preserves_injected_execution_values() {
        let config = DecodeBatcherConfig {
            max_batch: 12,
            wait: std::time::Duration::from_micros(3_500),
            allow_mixed_seq_lens: true,
        };

        assert_eq!(config.max_batch, 12);
        assert_eq!(config.wait, std::time::Duration::from_micros(3_500));
        assert!(config.allow_mixed_seq_lens);
    }

    #[test]
    fn decode_buffer_width_honors_injected_scheduler_requirement() {
        let vulkan = NamedTestBackend {
            name: "vulkan",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: false,
            resident_recurrent_state: false,
        };

        assert_eq!(decode_buffer_max_batch(&vulkan, None), 64);
        assert_eq!(decode_buffer_max_batch(&vulkan, Some(24)), 24);
        assert_eq!(decode_buffer_max_batch(&vulkan, Some(1)), 1);
    }

    #[test]
    fn test_decode_batcher_rowwise_retry_uses_backend_policy() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _fallback_env = EnvRestore::clear(DECODE_FALLBACK_ENV);
        let _rowwise_env = EnvRestore::clear(DECODE_BATCHER_ROWWISE_ENV);

        let vulkan_cpu_sentinel = NamedTestBackend {
            name: "vulkan",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: false,
            resident_recurrent_state: false,
        };
        let metal = NamedTestBackend {
            name: "metal",
            device: kiln_tensor::Device::Metal(0),
            resident_linear_state: false,
            resident_recurrent_state: false,
        };

        assert!(!decode_batcher_rowwise_retry_enabled(&vulkan_cpu_sentinel));
        assert!(!decode_batcher_rowwise_retry_enabled(&metal));

        unsafe {
            std::env::set_var("KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY", "1");
        }
        assert!(decode_batcher_rowwise_retry_enabled(&vulkan_cpu_sentinel));
        assert!(
            !decode_batcher_rowwise_retry_enabled(&metal),
            "Vulkan rowwise retry env should not apply to Metal policy"
        );
    }

    #[test]
    fn test_decode_hot_path_fallback_policy_defaults() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvRestore::clear(DECODE_FALLBACK_ENV);
        for (backend_name, device, expected, debug_env) in [
            (
                "cpu",
                kiln_tensor::Device::Cpu,
                FallbackPolicy::CorrectnessAllowed,
                None,
            ),
            (
                "cuda",
                kiln_tensor::Device::Cuda(0),
                FallbackPolicy::CorrectnessAllowed,
                None,
            ),
            (
                "metal",
                kiln_tensor::Device::Metal(0),
                FallbackPolicy::NativeRequired,
                Some("KILN_METAL_DECODE_BATCH_GENERIC_FALLBACK"),
            ),
            (
                "vulkan",
                kiln_tensor::Device::Vulkan(0),
                FallbackPolicy::NativeRequired,
                Some("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK"),
            ),
            (
                "rocm",
                kiln_tensor::Device::Rocm(0),
                FallbackPolicy::NativeRequired,
                Some("KILN_ROCM_DECODE_BATCH_GENERIC_FALLBACK"),
            ),
        ] {
            let fallback =
                backend::capability::BackendFallbackCapabilities::for_backend(backend_name, device);
            assert_eq!(fallback.decode_hot_path, expected);
            assert_eq!(
                fallback.decode_hot_path_debug_env, debug_env,
                "{backend_name} decode debug fallback env drifted"
            );
            let backend = NamedTestBackend {
                name: backend_name,
                device,
                resident_linear_state: false,
                resident_recurrent_state: false,
            };
            assert_eq!(
                decode_hot_path_fallback_policy_for_backend(&backend),
                expected
            );
        }

        let vulkan_cpu_sentinel = NamedTestBackend {
            name: "vulkan",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: false,
            resident_recurrent_state: false,
        };
        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&vulkan_cpu_sentinel),
            FallbackPolicy::NativeRequired
        );
        assert_eq!(
            backend::capability::BackendFallbackCapabilities::for_backend(
                "vulkan",
                kiln_tensor::Device::Cpu,
            )
            .decode_hot_path_debug_env,
            Some("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK")
        );
    }

    #[test]
    fn test_decode_hot_path_debug_fallback_opt_in_warns_and_counts() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvRestore::clear(DECODE_FALLBACK_ENV);
        unsafe {
            std::env::set_var("KILN_DECODE_HOT_PATH_DEBUG_FALLBACK", "1");
        }
        for (backend_name, device) in [
            ("metal", kiln_tensor::Device::Metal(0)),
            ("vulkan", kiln_tensor::Device::Vulkan(0)),
            ("rocm", kiln_tensor::Device::Rocm(0)),
        ] {
            let backend = NamedTestBackend {
                name: backend_name,
                device,
                resident_linear_state: false,
                resident_recurrent_state: false,
            };
            let policy = decode_hot_path_fallback_policy_for_backend(&backend);
            assert_eq!(policy, FallbackPolicy::WarnAndCount);
            assert!(policy.allows_fallback());
        }
    }

    #[test]
    fn test_decode_hot_path_backend_debug_fallback_uses_policy_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvRestore::clear(DECODE_FALLBACK_ENV);

        let metal = NamedTestBackend {
            name: "metal",
            device: kiln_tensor::Device::Metal(0),
            resident_linear_state: false,
            resident_recurrent_state: false,
        };
        let vulkan = NamedTestBackend {
            name: "vulkan",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: false,
            resident_recurrent_state: false,
        };
        let rocm = NamedTestBackend {
            name: "rocm",
            device: kiln_tensor::Device::Rocm(0),
            resident_linear_state: false,
            resident_recurrent_state: false,
        };

        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&metal),
            FallbackPolicy::NativeRequired
        );
        unsafe {
            std::env::set_var("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK", "1");
        }
        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&vulkan),
            FallbackPolicy::WarnAndCount
        );
        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&metal),
            FallbackPolicy::NativeRequired,
            "Vulkan decode fallback env should not apply to Metal policy"
        );
        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&rocm),
            FallbackPolicy::NativeRequired,
            "Vulkan decode fallback env should not apply to ROCm policy"
        );
    }

    fn block_table_with(blocks: &[u32]) -> BlockTable {
        let mut bt = BlockTable::new();
        bt.blocks = blocks.to_vec();
        bt
    }

    fn empty_linear_state() -> LinearAttentionState {
        LinearAttentionState {
            recurrent_states: Vec::new(),
            conv_states: Vec::new(),
        }
    }

    #[test]
    fn noncontiguous_kv_tiles_detection() {
        // #1082: FA2_KBLOCK_N=64 (hdim256). At block_size=16 → pages_per_chunk
        // = 64/16 = 4. CONTIGUOUS within each 4-page chunk → safe (false).
        let bt_contig = block_table_with(&[100, 101, 102, 103, 104, 105, 106, 107, 200, 201]);
        assert!(
            !batch_has_noncontiguous_kv_tiles(&[&bt_contig], &[160], 16),
            "physically-contiguous pages within a tile must NOT force the row-loop"
        );
        // A gap (999) starting the 2nd 4-page chunk: base=999 then 105 != 1000
        // → non-contiguous (true).
        let bt_frag = block_table_with(&[100, 101, 102, 103, 999, 105, 106, 107]);
        assert!(
            batch_has_noncontiguous_kv_tiles(&[&bt_frag], &[128], 16),
            "a fragmented page inside a tile must force the contiguity-safe row-loop"
        );
        // Chunk BOUNDARY discontinuity (idx 8 starts a new 4-page chunk) is
        // allowed — the kernel re-reads block_table at chunk starts.
        assert!(
            !batch_has_noncontiguous_kv_tiles(&[&bt_contig], &[144], 16),
            "discontinuity at a chunk boundary (every 4 pages) is allowed"
        );
        // bs=1 short row (1 page) is trivially contiguous.
        let bt_one = block_table_with(&[42]);
        assert!(!batch_has_noncontiguous_kv_tiles(&[&bt_one], &[5], 16));
        // Mixed batch: one bad row anywhere → true.
        assert!(batch_has_noncontiguous_kv_tiles(
            &[&bt_contig, &bt_frag],
            &[160, 128],
            16
        ));
        // Only check pages covering live tokens: a fragmented page BEYOND
        // seqused_k is not read by the kernel → not flagged.
        let bt_tail_frag = block_table_with(&[100, 101, 999]);
        assert!(
            !batch_has_noncontiguous_kv_tiles(&[&bt_tail_frag], &[20], 16),
            "fragmentation beyond the live window (seqlen=20 → 2 pages) is not read"
        );
        // #1082 KEY: at the new default block_size=64, pages_per_chunk =
        // FA2_KBLOCK_N/64 = 1, so each FA2 tile is exactly one page and the
        // kernel looks it up independently. Arbitrarily strided (non-adjacent)
        // blocks that WOULD trip at block_size=16 are safe at 64 → the row-loop
        // never fires for FA2 reasons (this is what restores bs=64 concurrent).
        let bt_strided = block_table_with(&[5, 7, 9, 11]);
        assert!(
            !batch_has_noncontiguous_kv_tiles(&[&bt_strided], &[256], 64),
            "block_size>=kBlockN makes every FA2 tile one page → no fragmentation trips"
        );
    }

    #[test]
    fn per_row_contiguity_mask_partitions_mixed_batch() {
        // #1082 partition fix: the batched-decode partition routes ONLY the
        // genuinely-fragmented rows to the per-row loop and batches the
        // contiguous majority through the fast path (vs #1445's all-or-nothing
        // whole-batch serialization). Validate the per-row mask it is built on.
        let bt_contig = block_table_with(&[100, 101, 102, 103, 104, 105, 106, 107]);
        let bt_frag = block_table_with(&[100, 101, 102, 103, 999, 105, 106, 107]);
        let bt_short = block_table_with(&[42]);
        // Per-row helper agrees with the batch wrapper, row by row.
        assert!(!row_has_noncontiguous_kv_tiles(
            bt_contig.blocks.as_slice(),
            128,
            16
        ));
        assert!(row_has_noncontiguous_kv_tiles(
            bt_frag.blocks.as_slice(),
            128,
            16
        ));
        assert!(!row_has_noncontiguous_kv_tiles(
            bt_short.blocks.as_slice(),
            5,
            16
        ));
        // A mixed batch yields a mask that picks out exactly the fragmented row;
        // the partition batches rows 0,2 (fast path) and row-loops only row 1.
        let bts = [&bt_contig, &bt_frag, &bt_short];
        let seqlens = [128usize, 128, 5];
        let mask: Vec<bool> = (0..3)
            .map(|r| row_has_noncontiguous_kv_tiles(bts[r].blocks.as_slice(), seqlens[r], 16))
            .collect();
        assert_eq!(mask, vec![false, true, false]);
        // The batch wrapper is exactly the OR of the per-row mask — so #1445's
        // detector fired on the WHOLE batch for a single bad row; the partition
        // resolves that per-row instead of serializing all of it.
        assert_eq!(
            batch_has_noncontiguous_kv_tiles(&bts, &seqlens, 16),
            mask.iter().any(|&x| x)
        );
    }

    #[test]
    fn extended_registration_requires_snapshot() {
        let bt = block_table_with(&[10, 11, 12, 13]);
        assert!(
            build_extended_registration(&[1, 2, 3, 4, 5], &[6, 7, 8], &bt, 4, None).is_none(),
            "no snapshot → no extended registration"
        );
    }

    #[test]
    fn resident_authority_suppresses_prompt_snapshot_capture() {
        let linear_state = LinearAttentionState {
            recurrent_states: vec![kiln_tensor::Tensor::from_vec(vec![0.0_f32], vec![1]).unwrap()],
            conv_states: vec![kiln_tensor::Tensor::from_vec(vec![0.0_f32], vec![1]).unwrap()],
        };
        let logical_backend = NamedTestBackend {
            name: "logical-test",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: false,
            resident_recurrent_state: false,
        };
        let resident_backend = NamedTestBackend {
            name: "resident-test",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: true,
            resident_recurrent_state: false,
        };
        let recurrent_resident_backend = NamedTestBackend {
            name: "recurrent-resident-test",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: false,
            resident_recurrent_state: true,
        };
        let suppression_count = AtomicU64::new(0);

        assert!(
            capture_authoritative_prefix_snapshot(
                &logical_backend,
                &suppression_count,
                &linear_state,
                "whole-prompt",
                64,
            )
            .unwrap()
            .is_some()
        );
        assert_eq!(suppression_count.load(Ordering::Relaxed), 0);

        assert!(
            capture_authoritative_prefix_snapshot(
                &resident_backend,
                &suppression_count,
                &linear_state,
                "whole-prompt",
                64,
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(suppression_count.load(Ordering::Relaxed), 1);

        assert!(
            capture_authoritative_prefix_snapshot(
                &recurrent_resident_backend,
                &suppression_count,
                &linear_state,
                "prompt-split",
                32,
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(suppression_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn resident_authority_never_publishes_generic_prefix_snapshots() {
        let make_state = |prefix_cache_registration_allowed| PagedBatchedDecodeState {
            block_table: block_table_with(&[10]),
            linear_state: empty_linear_state(),
            seq_len: 3,
            next_token: 7,
            next_token_logprob: None,
            generated_tokens: Vec::new(),
            step_seed: None,
            capture_behavior_logprobs: false,
            registration: None,
            allocated_blocks: vec![10],
            prefill_duration: std::time::Duration::ZERO,
            decode_duration: std::time::Duration::ZERO,
            prompt_tokens: vec![1, 2, 3],
            block_size: 4,
            prefill_split_snapshot: None,
            rolling_snapshot: None,
            prefix_cache_registration_allowed,
            id: 1,
        };
        let mut generic = make_state(true);
        let mut resident_prefill = make_state(false);
        let logical_backend = NamedTestBackend {
            name: "logical-test",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: false,
            resident_recurrent_state: false,
        };
        let suppression_count = AtomicU64::new(0);

        complete_paged_batched_decode_step(
            &logical_backend,
            &suppression_count,
            &mut [&mut generic, &mut resident_prefill],
            std::time::Duration::from_millis(1),
        );

        assert_eq!(generic.seq_len, 4);
        assert!(generic.rolling_snapshot.is_some());
        assert_eq!(resident_prefill.seq_len, 4);
        assert!(resident_prefill.rolling_snapshot.is_none());
        assert_eq!(suppression_count.load(Ordering::Relaxed), 0);

        let mut resident_decode = make_state(true);
        resident_decode.linear_state = LinearAttentionState {
            recurrent_states: vec![kiln_tensor::Tensor::from_vec(vec![0.0_f32], vec![1]).unwrap()],
            conv_states: vec![kiln_tensor::Tensor::from_vec(vec![0.0_f32], vec![1]).unwrap()],
        };
        let resident_backend = NamedTestBackend {
            name: "resident-test",
            device: kiln_tensor::Device::Cpu,
            resident_linear_state: true,
            resident_recurrent_state: false,
        };
        complete_paged_batched_decode_step(
            &resident_backend,
            &suppression_count,
            &mut [&mut resident_decode],
            std::time::Duration::from_millis(1),
        );

        assert_eq!(resident_decode.seq_len, 4);
        assert!(resident_decode.rolling_snapshot.is_none());
        assert_eq!(suppression_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn strict_prompt_prefix_split_is_inside_prompt() {
        assert_eq!(strict_prompt_prefix_split_pos(9, 0, 4), Some(8));
        assert_eq!(
            strict_prompt_prefix_split_pos(8, 0, 4),
            Some(4),
            "block-aligned prompts still need an inside split before the final prompt block"
        );
        assert_eq!(strict_prompt_prefix_split_pos(8, 4, 4), None);
        assert_eq!(strict_prompt_prefix_split_pos(3, 0, 4), None);
    }

    #[test]
    fn paged_prefix_reuse_requires_a_complete_final_block() {
        let greedy = SamplingParams::greedy();
        let sampled = SamplingParams {
            temperature: 0.8,
            ..SamplingParams::default()
        };

        let non_aligned_exact = PagedPrefixReuse {
            cached_tokens: 5,
            block_ids: vec![10, 11],
            linear_state: empty_linear_state(),
            next_token: Some(PagedPrefixNextToken::GreedyToken(7)),
        };
        assert!(!paged_prefix_reuse_matches_prompt(
            &non_aligned_exact,
            5,
            4,
            &greedy
        ));

        let aligned_exact = PagedPrefixReuse {
            cached_tokens: 4,
            block_ids: vec![10],
            linear_state: empty_linear_state(),
            next_token: Some(PagedPrefixNextToken::GreedyToken(7)),
        };
        assert!(paged_prefix_reuse_matches_prompt(
            &aligned_exact,
            4,
            4,
            &greedy
        ));
        assert!(!paged_prefix_reuse_matches_prompt(
            &aligned_exact,
            4,
            4,
            &sampled
        ));
        assert!(paged_prefix_reuse_matches_prompt(
            &aligned_exact,
            9,
            4,
            &sampled
        ));
    }

    #[test]
    fn extended_registration_skipped_when_not_block_aligned() {
        let bt = block_table_with(&[10, 11, 12, 13]);
        let snap = Some(RollingPrefixSnapshot {
            position: 7,
            linear_state: empty_linear_state(),
        });
        assert!(
            build_extended_registration(&[1, 2, 3, 4], &[5, 6, 7], &bt, 4, snap).is_none(),
            "position 7 is not block-aligned at block_size 4"
        );
    }

    #[test]
    fn extended_registration_inside_prompt_covers_prefix() {
        // Prefill-time snapshot at position 4, prompt is 8 tokens, no
        // generation yet. The registration should be a strict-prefix entry
        // covering the first 4 prompt tokens — this is what makes
        // multi-turn lookups hit when the chat template appends a divergent
        // generation-prompt tail to the prompt (the tail is past position
        // 4, so the entry remains valid for subsequent turns).
        let bt = block_table_with(&[10, 11, 12, 13]);
        let snap = Some(RollingPrefixSnapshot {
            position: 4,
            linear_state: empty_linear_state(),
        });
        let reg = build_extended_registration(&[1, 2, 3, 4, 5, 6, 7, 8], &[], &bt, 4, snap)
            .expect("expected strict-prefix registration");
        assert_eq!(reg.prompt_tokens, vec![1, 2, 3, 4]);
        assert_eq!(reg.block_ids, vec![10]);
        assert!(reg.next_token.is_none());
    }

    #[test]
    fn extended_registration_covers_prompt_plus_decoded() {
        let bt = block_table_with(&[10, 11, 12, 13, 14]);
        // 5-token prompt + 7 generated. Snapshot lands at position 12
        // (block-aligned at block_size 4) — covers prompt + 7 generated.
        let snap = Some(RollingPrefixSnapshot {
            position: 12,
            linear_state: empty_linear_state(),
        });
        let reg = build_extended_registration(
            &[1, 2, 3, 4, 5],
            &[10, 11, 12, 13, 14, 15, 16],
            &bt,
            4,
            snap,
        )
        .expect("expected extended registration");
        assert_eq!(
            reg.prompt_tokens,
            vec![1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16]
        );
        assert_eq!(reg.block_ids, vec![10, 11, 12]);
        assert!(reg.next_token.is_none());
    }

    #[test]
    fn extended_registration_truncates_to_last_boundary() {
        let bt = block_table_with(&[10, 11, 12, 13, 14]);
        // 5-token prompt + 11 generated → position 16 if all written, but
        // snapshot was only taken at position 12 (last boundary crossed).
        let snap = Some(RollingPrefixSnapshot {
            position: 12,
            linear_state: empty_linear_state(),
        });
        let reg = build_extended_registration(
            &[1, 2, 3, 4, 5],
            &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            &bt,
            4,
            snap,
        )
        .expect("expected extended registration");
        // Only the 7 generated tokens up to the snapshotted boundary are
        // included; the rest of the generation tail is discarded for the
        // cache entry (no linear-state snapshot for it).
        assert_eq!(reg.prompt_tokens.len(), 12);
        assert_eq!(reg.block_ids.len(), 3);
    }

    #[test]
    fn extended_registration_bails_when_block_table_short() {
        // Snapshot says position 12 (3 blocks) but block table only has 2
        // blocks — bookkeeping bug upstream; refuse to register a corrupt
        // entry instead of indexing out of bounds.
        let bt = block_table_with(&[10, 11]);
        let snap = Some(RollingPrefixSnapshot {
            position: 12,
            linear_state: empty_linear_state(),
        });
        assert!(
            build_extended_registration(&[1, 2, 3, 4, 5], &[6, 7, 8, 9, 10, 11, 12], &bt, 4, snap)
                .is_none(),
            "must not produce a registration referencing missing blocks"
        );
    }
}
