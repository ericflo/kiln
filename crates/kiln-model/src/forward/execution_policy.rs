use super::*;

// ---------------------------------------------------------------------------
// Phase 7: streaming/tiled GDN prefill execution policy.
//
// Startup resolves typed configuration over backend defaults and injects the
// immutable result. CUDA, ROCm, and Metal enable streaming automatically for
// long prompts where tiled prefill materially reduces peak activation memory.
// The recurrent state in `LinearAttentionState` provides the O(1) hand-off
// required for bit-exact agreement with the monolithic path. Vulkan remains
// opt-in because its GDN training path has backend-specific residency constraints.
// ---------------------------------------------------------------------------

/// Fallback tile size for explicit streaming prefill on devices without a
/// device-specific default. Must be a multiple of `GDN_CHUNK_SIZE` (64) so the
/// chunkwise kernel never sees a partial tail chunk from a tile boundary.
pub const STREAMING_PREFILL_DEFAULT_TILE: usize = 8192;
pub const STREAMING_PREFILL_CUDA_DEFAULT_TILE: usize =
    StreamingPrefillBackendPolicy::CUDA_TILE_TOKENS;
pub const STREAMING_PREFILL_CUDA_DEFAULT_THRESHOLD: usize =
    StreamingPrefillBackendPolicy::AUTO_MIN_PROMPT_TOKENS;
pub const STREAMING_PREFILL_ROCM_DEFAULT_TILE: usize =
    StreamingPrefillBackendPolicy::ROCM_TILE_TOKENS;
pub const STREAMING_PREFILL_ROCM_MEDIUM_TILE: usize = STREAMING_PREFILL_ROCM_DEFAULT_TILE;
pub const STREAMING_PREFILL_ROCM_MEDIUM_TILE_MAX_TOKENS: usize = 20_000;
pub const STREAMING_PREFILL_ROCM_DEFAULT_THRESHOLD: usize =
    StreamingPrefillBackendPolicy::ROCM_AUTO_MIN_PROMPT_TOKENS;
pub const DETACHED_FULL_ATTN_CUDA_DEFAULT_TILE: usize = 8192;
// Materialized-score full-attention backends still run exact causal attention,
// but their score/scratch tensors scale with query_tile * key_prefix. Keep the
// automatic cap below the multi-GB-per-score range: ROCm can abort the process
// from the VM heap instead of returning a recoverable allocation error when a
// long-row replay asks for overlarge contiguous score tensors.
pub const DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE: usize =
    DETACHED_FULL_ATTN_CUDA_DEFAULT_TILE;
pub const DETACHED_FULL_ATTN_ROCM_DEFAULT_TILE: usize =
    DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE;
pub const DETACHED_FULL_ATTN_FLASH_DEFAULT_TILE: usize = 65_536;
pub const DETACHED_FULL_ATTN_ROCM_ONLINE_DEFAULT_TILE: usize =
    DETACHED_FULL_ATTN_FLASH_DEFAULT_TILE;
pub(super) const MATERIALIZED_FULL_ATTN_TILE_GRANULARITY: usize = 128;
pub(super) const MATERIALIZED_FULL_ATTN_FORWARD_SCRATCH_BUFFERS: usize = 3;
pub(super) const DEFAULT_FULL_ATTN_SCORE_TILE_MAX_ELEMENTS: usize = 1 << 29;
pub const STREAMING_PREFILL_METAL_DEFAULT_TILE: usize = 2048;
pub const STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD: usize = 2048;
pub const STREAMING_PREFILL_VULKAN_DEFAULT_TILE: usize = STREAMING_PREFILL_METAL_DEFAULT_TILE;
pub const STREAMING_PREFILL_CUDA_TAPE_DEFAULT_TILE: usize = STREAMING_PREFILL_CUDA_DEFAULT_TILE;
pub const STREAMING_PREFILL_ROCM_TAPE_DEFAULT_TILE: usize =
    StreamingPrefillBackendPolicy::ROCM_TILE_TOKENS;
pub const STREAMING_PREFILL_METAL_TAPE_DEFAULT_TILE: usize = STREAMING_PREFILL_METAL_DEFAULT_TILE;
pub const STREAMING_PREFILL_VULKAN_TAPE_DEFAULT_TILE: usize = STREAMING_PREFILL_VULKAN_DEFAULT_TILE;
pub(super) const PAGED_KV_HEAD_MAJOR_READ_MIN_TOKENS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamingPrefillDeviceKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
}

pub(super) fn streaming_prefill_device_kind(device: &Device) -> StreamingPrefillDeviceKind {
    match device {
        Device::Cuda(_) => StreamingPrefillDeviceKind::Cuda,
        Device::Rocm(_) => StreamingPrefillDeviceKind::Rocm,
        Device::Metal(_) => StreamingPrefillDeviceKind::Metal,
        Device::Vulkan(_) => StreamingPrefillDeviceKind::Vulkan,
        _ => StreamingPrefillDeviceKind::Cpu,
    }
}

/// Operator selection for streaming-prefill execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamingPrefillMode {
    /// Apply the owning backend's automatic prompt-length policy.
    Auto,
    /// Stream every non-empty prefill, regardless of prompt length.
    Enabled,
    /// Keep streaming-prefill execution disabled.
    Disabled,
}

/// Fully resolved, immutable streaming-prefill execution policy.
///
/// Startup configuration is resolved into this value once and injected into
/// model execution. The model layer never consults process environment while a
/// request is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamingPrefillExecutionPolicy {
    mode: StreamingPrefillMode,
    auto_dispatch: StreamingPrefillAutoDispatch,
    base_tile_tokens: usize,
    tape_tile_tokens: usize,
    detached_full_attn_tile_tokens: usize,
    detached_full_attn_boundary_tile_tokens: usize,
    detached_full_attn_tape_replay_tile_tokens: usize,
    last_token_lm_head: bool,
}

impl StreamingPrefillExecutionPolicy {
    /// Resolve configured values over backend-owned defaults.
    ///
    /// A threshold override changes the crossover only for backends whose auto
    /// policy already permits streaming. It does not make CPU or Vulkan auto
    /// dispatch streaming. A detached-full-attention override applies to the
    /// ordinary, detached-boundary, and tape-replay variants, matching the
    /// single public configuration field.
    pub const fn resolve(
        backend: StreamingPrefillBackendPolicy,
        mode: StreamingPrefillMode,
        threshold_tokens: Option<usize>,
        base_tile_tokens: Option<usize>,
        tape_tile_tokens: Option<usize>,
        detached_full_attn_tile_tokens: Option<usize>,
        last_token_lm_head: bool,
    ) -> Self {
        let auto_dispatch = match (backend.auto_dispatch, threshold_tokens) {
            (StreamingPrefillAutoDispatch::PromptTokensAtLeast(_), Some(threshold_tokens)) => {
                StreamingPrefillAutoDispatch::PromptTokensAtLeast(threshold_tokens)
            }
            (auto_dispatch, _) => auto_dispatch,
        };
        let configured_base_tile_tokens = base_tile_tokens;
        let base_tile_tokens = match configured_base_tile_tokens {
            Some(tile_tokens) => tile_tokens,
            None => backend.base_tile_tokens,
        };
        let tape_tile_tokens = match tape_tile_tokens {
            Some(tile_tokens) => tile_tokens,
            None => match configured_base_tile_tokens {
                Some(_) => base_tile_tokens,
                None => backend.tape_tile_tokens,
            },
        };
        let (
            detached_full_attn_tile_tokens,
            detached_full_attn_boundary_tile_tokens,
            detached_full_attn_tape_replay_tile_tokens,
        ) = match detached_full_attn_tile_tokens {
            Some(tile_tokens) => (tile_tokens, tile_tokens, tile_tokens),
            None => match configured_base_tile_tokens {
                Some(_) => (base_tile_tokens, base_tile_tokens, base_tile_tokens),
                None => (
                    backend.detached_full_attn_tile_tokens,
                    backend.detached_full_attn_boundary_tile_tokens,
                    backend.detached_full_attn_tape_replay_tile_tokens,
                ),
            },
        };
        Self {
            mode,
            auto_dispatch,
            base_tile_tokens,
            tape_tile_tokens,
            detached_full_attn_tile_tokens,
            detached_full_attn_boundary_tile_tokens,
            detached_full_attn_tape_replay_tile_tokens,
            last_token_lm_head,
        }
    }

    pub fn for_runtime(backend: &dyn BackendRuntime) -> Self {
        Self::from_backend_policy(StreamingPrefillBackendPolicy::for_backend(
            BackendIdentity::runtime_name(backend),
            BackendIdentity::runtime_device(backend),
        ))
    }

    pub fn for_device(device: Device) -> Self {
        Self::from_backend_policy(StreamingPrefillBackendPolicy::for_device(device))
    }

    pub const fn from_backend_policy(backend: StreamingPrefillBackendPolicy) -> Self {
        Self::resolve(
            backend,
            StreamingPrefillMode::Auto,
            None,
            None,
            None,
            None,
            true,
        )
    }

    pub const fn mode(self) -> StreamingPrefillMode {
        self.mode
    }

    pub const fn threshold_tokens(self) -> Option<usize> {
        self.auto_dispatch.minimum_prompt_tokens()
    }

    pub const fn enabled_for(self, seq_len: usize) -> bool {
        match self.mode {
            StreamingPrefillMode::Auto => self.auto_dispatch.enabled_for_prompt_tokens(seq_len),
            StreamingPrefillMode::Enabled => seq_len > 0,
            StreamingPrefillMode::Disabled => false,
        }
    }

    pub const fn base_tile_tokens(self) -> usize {
        self.base_tile_tokens
    }

    pub const fn base_tile_tokens_for(self, _seq_len: usize) -> usize {
        self.base_tile_tokens
    }

    pub const fn tape_tile_tokens(self) -> usize {
        self.tape_tile_tokens
    }

    pub const fn detached_full_attn_tile_tokens(self) -> usize {
        self.detached_full_attn_tile_tokens
    }

    pub const fn detached_full_attn_boundary_tile_tokens(self) -> usize {
        self.detached_full_attn_boundary_tile_tokens
    }

    pub const fn detached_full_attn_tape_replay_tile_tokens(self) -> usize {
        self.detached_full_attn_tape_replay_tile_tokens
    }

    pub const fn last_token_lm_head(self) -> bool {
        self.last_token_lm_head
    }
}

/// Compatibility helper for non-device-aware callers. Startup configuration
/// should instead inject a [`StreamingPrefillExecutionPolicy`].
pub fn streaming_prefill_enabled() -> bool {
    false
}

/// Device-aware streaming prefill policy for production prefill dispatch.
///
/// This compatibility wrapper is env-free and resolves the backend default.
/// Configured production policy should be injected through an explicit-policy
/// forward variant.
pub fn streaming_prefill_enabled_for(device: &Device, seq_len: usize) -> bool {
    StreamingPrefillExecutionPolicy::for_device(*device).enabled_for(seq_len)
}

/// Portable automatic dispatch threshold retained for compatibility callers.
pub fn streaming_prefill_threshold_tokens() -> usize {
    STREAMING_PREFILL_METAL_DEFAULT_THRESHOLD
}

/// Portable tile default retained for compatibility callers.
pub fn streaming_tile_tokens() -> usize {
    STREAMING_PREFILL_DEFAULT_TILE
}

/// Device-aware, env-free backend tile default.
pub fn streaming_tile_tokens_for(device: &Device) -> usize {
    StreamingPrefillExecutionPolicy::for_device(*device).base_tile_tokens()
}

/// Tile size for detached full-attention boundary forwards during
/// checkpointed training. This is separate from GDN streaming because
/// full-attention uses FlashAttention over a query tile and a prefix KV span,
/// while GDN tiles carry recurrent-state and backward-memory constraints.
pub fn detached_full_attn_tile_tokens_for(device: &Device) -> usize {
    StreamingPrefillExecutionPolicy::for_device(*device).detached_full_attn_tile_tokens()
}

pub(super) fn rocm_long_flash_attn_enabled() -> bool {
    crate::rocm_policy::current_rocm_kernel_policy().long_flash_attn
}

#[cfg(feature = "rocm")]
pub(super) fn rocm_native_rectangular_causal_flash_enabled() -> bool {
    crate::rocm_policy::current_rocm_kernel_policy().native_rectangular_causal_flash
}

pub(super) fn long_prefill_leaf_flash_allowed_for_device(
    device: &Device,
    q_len: usize,
    kv_len: usize,
) -> bool {
    let _ = (q_len, kv_len);
    if matches!(
        streaming_prefill_device_kind(device),
        StreamingPrefillDeviceKind::Rocm
    ) {
        // ROCm's long flash/online SDPA route is exact and is the only
        // practical path for large prefix-causal training rows on gfx115x.
        // The immutable qualified profile keeps this route enabled so long
        // rows fit without a per-request or hot-path policy lookup.
        rocm_long_flash_attn_enabled()
    } else {
        true
    }
}

pub(super) fn flash_prefill_allowed_for_shape(
    backend: &dyn BackendRuntime,
    device: &Device,
    dtype: DType,
    head_dim: usize,
    q_len: usize,
    kv_len: usize,
) -> bool {
    if dtype != DType::BF16 || !matches!(head_dim, 128 | 256) {
        return false;
    }
    if !AttentionBackend::runtime_supports_flash_attn_prefill(backend) {
        return false;
    }
    long_prefill_leaf_flash_allowed_for_device(device, q_len, kv_len)
}

pub(super) fn full_attn_score_tile_max_elements() -> usize {
    // Keep materialized score tiles comfortably below common 32-bit indexing
    // boundaries. This is still exact SDPA, just split into more prefix tiles.
    DEFAULT_FULL_ATTN_SCORE_TILE_MAX_ELEMENTS
}

pub(super) fn full_attn_materialized_score_budget_mib() -> usize {
    crate::full_attention_policy::full_attention_score_budget_mib()
}

pub(super) fn full_attn_materialized_scores_for_device(device: &Device) -> bool {
    matches!(
        streaming_prefill_device_kind(device),
        StreamingPrefillDeviceKind::Cuda
            | StreamingPrefillDeviceKind::Rocm
            | StreamingPrefillDeviceKind::Metal
            | StreamingPrefillDeviceKind::Vulkan
    )
}

pub(super) fn full_attn_score_dtype_bytes(dtype: DType) -> usize {
    match dtype {
        // Materialized SDPA paths usually promote scores/softmax scratch even
        // when Q/K/V are BF16/F16. Budget against that larger allocation so the
        // exact tiled path does not ask ROCm to map an overlarge score buffer.
        DType::BF16 | DType::F16 => 4,
        DType::F32 => 4,
        _ => dtype.size_in_bytes().max(1),
    }
}

pub(super) fn full_attn_adaptive_max_tile_tokens(
    device: &Device,
    dtype: DType,
    batch: usize,
    key_prefix_len: usize,
    num_heads: usize,
    base_tile_tokens: usize,
    scratch_buffers: usize,
) -> usize {
    let budget_mb = full_attn_materialized_score_budget_mib();
    full_attn_adaptive_max_tile_tokens_with_budget(
        device,
        dtype,
        batch,
        key_prefix_len,
        num_heads,
        base_tile_tokens,
        scratch_buffers,
        budget_mb,
    )
}

pub(super) fn full_attn_adaptive_max_tile_tokens_with_budget(
    device: &Device,
    dtype: DType,
    batch: usize,
    key_prefix_len: usize,
    num_heads: usize,
    base_tile_tokens: usize,
    scratch_buffers: usize,
    budget_mb: usize,
) -> usize {
    if base_tile_tokens == 0
        || batch == 0
        || key_prefix_len == 0
        || num_heads == 0
        || scratch_buffers == 0
        || !full_attn_materialized_scores_for_device(device)
    {
        return base_tile_tokens;
    }

    let budget_bytes = budget_mb.saturating_mul(1024 * 1024);
    let score_bytes = full_attn_score_dtype_bytes(dtype);
    let denom = batch
        .saturating_mul(num_heads)
        .saturating_mul(key_prefix_len)
        .saturating_mul(score_bytes)
        .saturating_mul(scratch_buffers);
    if denom == 0 {
        return base_tile_tokens;
    }
    let budgeted = (budget_bytes / denom).max(1);
    let score_element_denom = batch
        .saturating_mul(num_heads)
        .saturating_mul(key_prefix_len);
    let budgeted = if score_element_denom == 0 {
        budgeted
    } else {
        let max_by_elements = (full_attn_score_tile_max_elements() / score_element_denom).max(1);
        budgeted.min(max_by_elements)
    };
    let granularity = MATERIALIZED_FULL_ATTN_TILE_GRANULARITY;
    let aligned = if budgeted >= granularity {
        (budgeted / granularity).max(1) * granularity
    } else {
        budgeted
    };
    base_tile_tokens.min(aligned).max(1)
}

pub(super) fn full_attn_adaptive_tile_len(
    device: &Device,
    dtype: DType,
    batch: usize,
    tile_start: usize,
    remaining: usize,
    num_heads: usize,
    base_tile_tokens: usize,
    scratch_buffers: usize,
) -> usize {
    let budget_mb = full_attn_materialized_score_budget_mib();
    full_attn_adaptive_tile_len_with_budget(
        device,
        dtype,
        batch,
        tile_start,
        remaining,
        num_heads,
        base_tile_tokens,
        scratch_buffers,
        budget_mb,
    )
}

pub(super) fn full_attn_adaptive_tile_len_with_budget(
    device: &Device,
    dtype: DType,
    batch: usize,
    tile_start: usize,
    remaining: usize,
    num_heads: usize,
    base_tile_tokens: usize,
    scratch_buffers: usize,
    budget_mb: usize,
) -> usize {
    let mut tile_len = remaining.min(base_tile_tokens.max(1));
    if tile_len <= GDN_CHUNK_SIZE
        || !full_attn_materialized_scores_for_device(device)
        || batch == 0
        || num_heads == 0
        || scratch_buffers == 0
    {
        return tile_len.max(1);
    }

    loop {
        let key_prefix_len = tile_start.saturating_add(tile_len);
        let max_for_prefix = full_attn_adaptive_max_tile_tokens_with_budget(
            device,
            dtype,
            batch,
            key_prefix_len,
            num_heads,
            base_tile_tokens,
            scratch_buffers,
            budget_mb,
        );
        if tile_len <= max_for_prefix || tile_len <= MATERIALIZED_FULL_ATTN_TILE_GRANULARITY {
            return tile_len.max(1);
        }
        let candidate = remaining.min(max_for_prefix).max(1);
        tile_len = if candidate > MATERIALIZED_FULL_ATTN_TILE_GRANULARITY {
            (candidate / MATERIALIZED_FULL_ATTN_TILE_GRANULARITY).max(1)
                * MATERIALIZED_FULL_ATTN_TILE_GRANULARITY
        } else {
            candidate
        };
    }
}

pub(super) fn full_attn_adaptive_tile_plan_summary(
    device: &Device,
    dtype: DType,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    base_tile_tokens: usize,
    scratch_buffers: usize,
    budget_mb: usize,
) -> (usize, usize, usize, usize, Option<u64>) {
    let mut tile_start = 0usize;
    let mut tile_count = 0usize;
    let mut first_tile = 0usize;
    let mut min_tile = usize::MAX;
    let mut max_tile = 0usize;
    let mut peak_scratch_bytes = Some(0u64);
    while tile_start < seq_len {
        let remaining = seq_len - tile_start;
        let tile_len = full_attn_adaptive_tile_len_with_budget(
            device,
            dtype,
            batch,
            tile_start,
            remaining,
            num_heads,
            base_tile_tokens,
            scratch_buffers,
            budget_mb,
        );
        if tile_count == 0 {
            first_tile = tile_len;
        }
        tile_count += 1;
        min_tile = min_tile.min(tile_len);
        max_tile = max_tile.max(tile_len);
        let tile_end = tile_start + tile_len;
        peak_scratch_bytes = peak_scratch_bytes.and_then(|peak| {
            full_attn_materialized_scratch_bytes(
                dtype,
                batch,
                num_heads,
                tile_len,
                tile_end,
                scratch_buffers,
            )
            .map(|tile_bytes| peak.max(tile_bytes))
        });
        tile_start = tile_end;
    }
    if tile_count == 0 {
        min_tile = 0;
    }
    (
        tile_count,
        first_tile,
        min_tile,
        max_tile,
        peak_scratch_bytes,
    )
}

pub(super) fn full_attn_materialized_scratch_bytes(
    dtype: DType,
    batch: usize,
    num_heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    scratch_buffers: usize,
) -> Option<u64> {
    let mut bytes = 1u64;
    for factor in [
        batch,
        num_heads,
        query_tokens,
        key_tokens,
        full_attn_score_dtype_bytes(dtype),
        scratch_buffers,
    ] {
        bytes = bytes.checked_mul(u64::try_from(factor).ok()?)?;
    }
    Some(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FullAttnChunkMode {
    DetachedBoundary,
    TapeReplay,
}

impl FullAttnChunkMode {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::DetachedBoundary => "detached_boundary",
            Self::TapeReplay => "tape_replay",
        }
    }

    pub(super) const fn detach_outputs(self) -> bool {
        matches!(self, Self::DetachedBoundary)
    }

    pub(super) const fn materialized_scratch_buffers(self) -> usize {
        match self {
            // ROCm/Metal/Vulkan materialized SDPA forward paths can hold QK
            // scores, scaled scores, and softmax probabilities concurrently.
            // Budget against all three exact score-sized buffers so adaptive
            // tiles do not ask ROCm to map oversized tensors at long prefixes.
            Self::DetachedBoundary => MATERIALIZED_FULL_ATTN_FORWARD_SCRATCH_BUFFERS,
            // Tape replay also has to survive backward. The fast ROCm route
            // materializes p/dp/ds score-sized tensors around BLASLt matmuls;
            // charging the replay tiler for those buffers lets long-context
            // training choose tiles that hit the fast exact backward instead
            // of falling back to the scalar bounded kernel.
            Self::TapeReplay => 8,
        }
    }

    pub(super) fn flash_tile_guaranteed(
        self,
        backend: &dyn BackendRuntime,
        device: &Device,
        dtype: DType,
        head_dim: usize,
    ) -> bool {
        let kind = streaming_prefill_device_kind(device);
        if !matches!(
            kind,
            StreamingPrefillDeviceKind::Cuda | StreamingPrefillDeviceKind::Rocm
        ) {
            return false;
        }
        if kind == StreamingPrefillDeviceKind::Rocm && !rocm_long_flash_attn_enabled() {
            return false;
        }
        if dtype != DType::BF16 || !matches!(head_dim, 128 | 256) {
            return false;
        }
        if !AttentionBackend::runtime_supports_flash_attn_prefill(backend) {
            return false;
        }
        match self {
            Self::DetachedBoundary => true,
            Self::TapeReplay => {
                #[cfg(any(
                    feature = "cuda",
                    feature = "metal",
                    feature = "vulkan",
                    feature = "rocm"
                ))]
                {
                    crate::tape_forward::tape_scope_active()
                }
                #[cfg(not(any(
                    feature = "cuda",
                    feature = "metal",
                    feature = "vulkan",
                    feature = "rocm"
                )))]
                {
                    false
                }
            }
        }
    }

    pub(super) fn materialized_scratch_buffers_for_tile_plan(
        self,
        backend: &dyn BackendRuntime,
        device: &Device,
        dtype: DType,
        head_dim: usize,
    ) -> usize {
        #[cfg(feature = "rocm")]
        if matches!(
            streaming_prefill_device_kind(device),
            StreamingPrefillDeviceKind::Rocm
        ) && matches!(self, Self::DetachedBoundary)
            && rocm_native_rectangular_causal_flash_enabled()
            && self.flash_tile_guaranteed(backend, device, dtype, head_dim)
        {
            return 0;
        }
        if matches!(
            streaming_prefill_device_kind(device),
            StreamingPrefillDeviceKind::Rocm
        ) {
            return self.materialized_scratch_buffers();
        }
        if self.flash_tile_guaranteed(backend, device, dtype, head_dim) {
            0
        } else {
            self.materialized_scratch_buffers()
        }
    }
}

/// Device-aware tile-size default for active kt-tape replay.
///
/// Checkpointed training records each reverse segment under a fresh tape scope.
/// CUDA/ROCm keep the same tile size as GDN streaming by default: halving the
/// tile reduced per-call scratch but increased cumulative tape residency enough
/// to slow late long-context GDN tiles. Detached full-attention boundary
/// forwards use their own larger tile selector because they are not tape
/// recording and are FlashAttention-backed.
pub fn tape_streaming_tile_tokens_for(device: &Device) -> usize {
    StreamingPrefillExecutionPolicy::for_device(*device).tape_tile_tokens()
}

/// Compatibility default for streaming LM-head execution.
///
/// In streaming mode only the final token's logits are needed for sampling, so
/// the LM head projection is collapsed to a single row per prefill. Production
/// callers inject the resolved value through a streaming execution policy.
pub fn streaming_last_token_lm_head() -> bool {
    true
}
