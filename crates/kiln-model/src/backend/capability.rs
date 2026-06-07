//! Typed capability descriptors for backend diagnostics.
//!
//! This is the Phase 0/1 bridge from bool-only `supports_*` predicates toward
//! request-shaped capability queries. The snapshot intentionally reads the
//! focused backend facets that currently forward to `BackendRuntime`; it is
//! descriptive only and does not change dispatch behavior.

use kiln_graph::ReplayKey;

use super::{
    AttentionBackend, BackendIdentity, BackendRuntime, ConvBackend, FallbackPolicy, GdnBackend,
    PagedKvBackend, ReplayBackend, ResidencyBackend, SamplingBackend, TrainingCapabilities,
    TrainingLossBackend, TrainingPrecisionPolicy,
};

/// Backend answer for a capability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Support {
    Native,
    NativeWithConstraints,
    HostFallbackAllowed,
    Declined,
    Unsupported,
    DisabledByEnv,
    RequiresFeature,
}

impl Support {
    pub const fn from_supports_predicate(supported: bool) -> Self {
        if supported {
            Support::NativeWithConstraints
        } else {
            Support::Declined
        }
    }

    pub const fn is_native(self) -> bool {
        matches!(self, Support::Native | Support::NativeWithConstraints)
    }
}

/// Accumulation precision requested by a matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulAccumulation {
    F32,
}

/// Logical storage layout for a matmul operand or output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulOperandLayout {
    RowMajor,
    ColMajor,
}

impl MatmulOperandLayout {
    /// Stable layout name matching the lower BLAS request descriptors.
    pub const fn blas_name(self) -> &'static str {
        match self {
            Self::RowMajor => "row",
            Self::ColMajor => "col",
        }
    }
}

/// Fused matmul tail requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulEpilogue {
    Identity,
    Bias,
    Relu,
    Gelu,
    Silu,
    BiasSilu,
    BiasGelu,
}

impl MatmulEpilogue {
    /// Stable epilogue name matching the lower BLAS request descriptors.
    pub const fn blas_name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Bias => "bias",
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::Silu => "silu",
            Self::BiasSilu => "bias_silu",
            Self::BiasGelu => "bias_gelu",
        }
    }
}

/// Batch shape collapsed into the logical matmul request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulBatchPolicy {
    Single,
    Batched { batches: usize },
}

impl MatmulBatchPolicy {
    fn from_leading_shape(shape: &[usize]) -> Self {
        let batches = shape.iter().product::<usize>().max(1);
        if batches == 1 {
            Self::Single
        } else {
            Self::Batched { batches }
        }
    }
}

/// Error returned when a rich engine request cannot be projected to BLAS shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulRequestProjectionError {
    IncompatibleShape,
    MixedDTypes,
    UnsupportedAccumulation,
    UnsupportedDType,
    InvalidConcurrentStreams,
}

/// Dependency-free projection of [`MatmulRequest`] onto the lower BLAS request shape.
///
/// `kiln-model` intentionally does not depend on `kiln-blas` or `kiln-rocblas`,
/// so this descriptor mirrors their shared `m/n/k + dtype + layout + epilogue`
/// vocabulary without importing the concrete crate type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatmulBlasRequest {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub dtype: kiln_tensor::DType,
    pub lhs_layout: MatmulOperandLayout,
    pub rhs_layout: MatmulOperandLayout,
    pub out_layout: MatmulOperandLayout,
    pub epilogue: MatmulEpilogue,
    pub batch: MatmulBatchPolicy,
    pub replay_safe: bool,
    pub concurrent_streams: u8,
}

impl MatmulBlasRequest {
    pub const fn dtype_name(&self) -> &'static str {
        self.dtype.short_name()
    }
}

/// Engine-facing matmul capability request.
///
/// This intentionally carries the richer shape/dtype/layout/replay vocabulary
/// from the unification plan, while staying compatible with the narrower
/// BLAS-crate `m/n/k + dtype + layout + epilogue` request direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulRequest {
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
    pub lhs_dtype: kiln_tensor::DType,
    pub rhs_dtype: kiln_tensor::DType,
    pub out_dtype: kiln_tensor::DType,
    pub accumulation: MatmulAccumulation,
    pub lhs_layout: MatmulOperandLayout,
    pub rhs_layout: MatmulOperandLayout,
    pub out_layout: MatmulOperandLayout,
    pub batch: MatmulBatchPolicy,
    pub epilogue: MatmulEpilogue,
    pub replay_safe: bool,
}

impl MatmulRequest {
    pub fn plain(
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        dtype: kiln_tensor::DType,
        replay_safe: bool,
    ) -> Self {
        let batch = lhs_shape
            .len()
            .checked_sub(2)
            .map(|rank_prefix| MatmulBatchPolicy::from_leading_shape(&lhs_shape[..rank_prefix]))
            .unwrap_or(MatmulBatchPolicy::Single);
        Self {
            lhs_shape,
            rhs_shape,
            lhs_dtype: dtype,
            rhs_dtype: dtype,
            out_dtype: dtype,
            accumulation: MatmulAccumulation::F32,
            lhs_layout: MatmulOperandLayout::RowMajor,
            rhs_layout: MatmulOperandLayout::RowMajor,
            out_layout: MatmulOperandLayout::RowMajor,
            batch,
            epilogue: MatmulEpilogue::Identity,
            replay_safe,
        }
    }

    pub fn with_epilogue(mut self, epilogue: MatmulEpilogue) -> Self {
        self.epilogue = epilogue;
        self
    }

    pub fn to_blas_request(
        &self,
        concurrent_streams: u8,
    ) -> Result<MatmulBlasRequest, MatmulRequestProjectionError> {
        if concurrent_streams == 0 {
            return Err(MatmulRequestProjectionError::InvalidConcurrentStreams);
        }
        let (m, n, k) = self
            .logical_mnk()
            .ok_or(MatmulRequestProjectionError::IncompatibleShape)?;
        if self.lhs_dtype != self.rhs_dtype || self.lhs_dtype != self.out_dtype {
            return Err(MatmulRequestProjectionError::MixedDTypes);
        }
        if self.accumulation != MatmulAccumulation::F32 {
            return Err(MatmulRequestProjectionError::UnsupportedAccumulation);
        }
        if !matches!(
            self.lhs_dtype,
            kiln_tensor::DType::F32 | kiln_tensor::DType::BF16 | kiln_tensor::DType::F16
        ) {
            return Err(MatmulRequestProjectionError::UnsupportedDType);
        }

        Ok(MatmulBlasRequest {
            m: m as u64,
            n: n as u64,
            k: k as u64,
            dtype: self.lhs_dtype,
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
            out_layout: self.out_layout,
            epilogue: self.epilogue,
            batch: self.batch,
            replay_safe: self.replay_safe,
            concurrent_streams,
        })
    }

    pub fn logical_mnk(&self) -> Option<(usize, usize, usize)> {
        if !self.has_compatible_shapes() {
            return None;
        }
        let rank = self.lhs_shape.len();
        Some((
            self.lhs_shape[rank - 2],
            self.rhs_shape[rank - 1],
            self.lhs_shape[rank - 1],
        ))
    }

    pub fn rank(&self) -> Option<usize> {
        if self.lhs_shape.len() == self.rhs_shape.len() {
            Some(self.lhs_shape.len())
        } else {
            None
        }
    }

    fn has_compatible_shapes(&self) -> bool {
        let Some(rank) = self.rank() else {
            return false;
        };
        if rank < 2 {
            return false;
        }
        if self.lhs_shape[..rank - 2] != self.rhs_shape[..rank - 2] {
            return false;
        }
        if self.lhs_shape[rank - 1] != self.rhs_shape[rank - 2] {
            return false;
        }
        self.batch == MatmulBatchPolicy::from_leading_shape(&self.lhs_shape[..rank - 2])
    }

    fn has_supported_dtype_contract(&self) -> bool {
        self.lhs_dtype == self.rhs_dtype
            && self.lhs_dtype == self.out_dtype
            && self.accumulation == MatmulAccumulation::F32
            && matches!(
                self.lhs_dtype,
                kiln_tensor::DType::F32 | kiln_tensor::DType::BF16 | kiln_tensor::DType::F16
            )
    }

    fn is_row_major_output(&self) -> bool {
        self.out_layout == MatmulOperandLayout::RowMajor
    }

    fn is_row_major_input(&self) -> bool {
        self.lhs_layout == MatmulOperandLayout::RowMajor
            && self.rhs_layout == MatmulOperandLayout::RowMajor
    }
}

/// Attention operation family being queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionRequestKind {
    FlashPrefill,
    FlashPrefillHeadMajor,
    FlashPagedDecode,
}

/// Logical tensor layout for an attention capability request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionLayout {
    Sdpa,
    HeadMajor,
    PagedKv,
}

impl AttentionLayout {
    const fn for_kind(kind: AttentionRequestKind) -> Self {
        match kind {
            AttentionRequestKind::FlashPrefill => Self::Sdpa,
            AttentionRequestKind::FlashPrefillHeadMajor => Self::HeadMajor,
            AttentionRequestKind::FlashPagedDecode => Self::PagedKv,
        }
    }
}

/// Request descriptor for attention capability queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionRequest {
    pub kind: AttentionRequestKind,
    pub q_shape: Vec<usize>,
    pub k_shape: Vec<usize>,
    pub v_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub layout: AttentionLayout,
    pub q_dtype: kiln_tensor::DType,
    pub k_dtype: kiln_tensor::DType,
    pub v_dtype: kiln_tensor::DType,
    pub batch: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub replay_safe: bool,
}

impl AttentionRequest {
    pub fn flash_prefill(
        q_dtype: kiln_tensor::DType,
        k_dtype: kiln_tensor::DType,
        v_dtype: kiln_tensor::DType,
        batch: usize,
        seq_len: usize,
        head_dim: usize,
        replay_safe: bool,
    ) -> Self {
        let shape = Self::shape_from_dims(batch, seq_len, head_dim);
        Self {
            kind: AttentionRequestKind::FlashPrefill,
            q_shape: shape.clone(),
            k_shape: shape.clone(),
            v_shape: shape.clone(),
            output_shape: shape,
            layout: AttentionLayout::for_kind(AttentionRequestKind::FlashPrefill),
            q_dtype,
            k_dtype,
            v_dtype,
            batch,
            seq_len,
            head_dim,
            replay_safe,
        }
    }

    pub fn with_kind(mut self, kind: AttentionRequestKind) -> Self {
        self.kind = kind;
        self.layout = AttentionLayout::for_kind(kind);
        self
    }

    pub fn with_dims(mut self, batch: usize, seq_len: usize, head_dim: usize) -> Self {
        self.batch = batch;
        self.seq_len = seq_len;
        self.head_dim = head_dim;
        let shape = Self::shape_from_dims(batch, seq_len, head_dim);
        self.q_shape = shape.clone();
        self.k_shape = shape.clone();
        self.v_shape = shape.clone();
        self.output_shape = shape;
        self
    }

    pub fn with_replay_safe(mut self, replay_safe: bool) -> Self {
        self.replay_safe = replay_safe;
        self
    }

    pub fn with_layout(mut self, layout: AttentionLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn with_shapes(
        mut self,
        q_shape: Vec<usize>,
        k_shape: Vec<usize>,
        v_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Self {
        self.q_shape = q_shape;
        self.k_shape = k_shape;
        self.v_shape = v_shape;
        self.output_shape = output_shape;
        self
    }

    pub fn shape_key(&self) -> Vec<Vec<usize>> {
        vec![
            self.q_shape.clone(),
            self.k_shape.clone(),
            self.v_shape.clone(),
            self.output_shape.clone(),
        ]
    }

    fn shape_from_dims(batch: usize, seq_len: usize, head_dim: usize) -> Vec<usize> {
        vec![batch, seq_len, head_dim]
    }
}

/// Linear/decode operation family being queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinearRequestKind {
    DecodeArgmax,
    DecodeArgmaxBatch,
    DecodeSample,
    DecodeSampleBatch,
}

/// Logical tensor layouts for a linear/lm-head capability request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinearLayouts {
    pub input: MatmulOperandLayout,
    pub weight: MatmulOperandLayout,
    pub output: MatmulOperandLayout,
}

impl LinearLayouts {
    pub const ROW_MAJOR: Self = Self {
        input: MatmulOperandLayout::RowMajor,
        weight: MatmulOperandLayout::RowMajor,
        output: MatmulOperandLayout::RowMajor,
    };
}

/// Request descriptor for linear/lm-head capability queries.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearRequest {
    pub kind: LinearRequestKind,
    pub input_shape: Vec<usize>,
    pub weight_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub layout: LinearLayouts,
    pub input_dtype: kiln_tensor::DType,
    pub weight_dtype: kiln_tensor::DType,
    pub output_dtype: kiln_tensor::DType,
    pub batch: usize,
    pub top_k: Vec<u32>,
    pub temperatures: Vec<f32>,
    pub replay_safe: bool,
}

impl LinearRequest {
    pub fn decode_argmax(
        input_dtype: kiln_tensor::DType,
        weight_dtype: kiln_tensor::DType,
        output_dtype: kiln_tensor::DType,
        batch: usize,
        replay_safe: bool,
    ) -> Self {
        Self {
            kind: if batch > 1 {
                LinearRequestKind::DecodeArgmaxBatch
            } else {
                LinearRequestKind::DecodeArgmax
            },
            input_shape: Vec::new(),
            weight_shape: Vec::new(),
            output_shape: Vec::new(),
            layout: LinearLayouts::ROW_MAJOR,
            input_dtype,
            weight_dtype,
            output_dtype,
            batch,
            top_k: Vec::new(),
            temperatures: Vec::new(),
            replay_safe,
        }
    }

    pub fn decode_sample(
        input_dtype: kiln_tensor::DType,
        weight_dtype: kiln_tensor::DType,
        output_dtype: kiln_tensor::DType,
        top_k: Vec<u32>,
        temperatures: Vec<f32>,
        replay_safe: bool,
    ) -> Self {
        let batch = top_k.len().max(1);
        Self {
            kind: if batch > 1 {
                LinearRequestKind::DecodeSampleBatch
            } else {
                LinearRequestKind::DecodeSample
            },
            input_shape: Vec::new(),
            weight_shape: Vec::new(),
            output_shape: Vec::new(),
            layout: LinearLayouts::ROW_MAJOR,
            input_dtype,
            weight_dtype,
            output_dtype,
            batch,
            top_k,
            temperatures,
            replay_safe,
        }
    }

    pub fn with_shapes(
        mut self,
        input_shape: Vec<usize>,
        weight_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Self {
        self.input_shape = input_shape;
        self.weight_shape = weight_shape;
        self.output_shape = output_shape;
        self
    }

    pub fn with_layout(mut self, layout: LinearLayouts) -> Self {
        self.layout = layout;
        self
    }

    pub fn shape_key(&self) -> Vec<Vec<usize>> {
        vec![
            self.input_shape.clone(),
            self.weight_shape.clone(),
            self.output_shape.clone(),
        ]
    }
}

/// Logical resource layout for replay/capture capability queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplayLayout {
    StableResident,
    PagedDecodeGraphOutputs,
}

/// Replay operation family being queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplayRequestKind {
    ResidentDecode,
    PagedDecodeGraphOutputs,
}

impl ReplayRequestKind {
    pub const fn operation_name(self) -> &'static str {
        match self {
            Self::ResidentDecode => "resident_decode",
            Self::PagedDecodeGraphOutputs => "paged_decode_graph_outputs",
        }
    }
}

/// Request descriptor for replay/capture capability queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRequest {
    pub kind: ReplayRequestKind,
    pub replay_shape: Vec<usize>,
    pub layout: ReplayLayout,
    pub max_hidden: usize,
    pub max_intermediate: usize,
    pub max_batch: usize,
    pub dtype: Option<kiln_tensor::DType>,
    pub replay_safe: bool,
}

impl ReplayRequest {
    pub fn resident_decode(max_hidden: usize, max_intermediate: usize, max_batch: usize) -> Self {
        let replay_shape = Self::shape_from_bounds(max_hidden, max_intermediate, max_batch);
        Self {
            kind: ReplayRequestKind::ResidentDecode,
            replay_shape,
            layout: ReplayLayout::StableResident,
            max_hidden,
            max_intermediate,
            max_batch,
            dtype: None,
            replay_safe: true,
        }
    }

    pub fn paged_decode_graph_outputs(
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> Self {
        let replay_shape = Self::shape_from_bounds(max_hidden, max_intermediate, max_batch);
        Self {
            kind: ReplayRequestKind::PagedDecodeGraphOutputs,
            replay_shape,
            layout: ReplayLayout::PagedDecodeGraphOutputs,
            max_hidden,
            max_intermediate,
            max_batch,
            dtype: None,
            replay_safe: true,
        }
    }

    pub fn with_dtype(mut self, dtype: kiln_tensor::DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn with_replay_safe(mut self, replay_safe: bool) -> Self {
        self.replay_safe = replay_safe;
        self
    }

    pub fn with_replay_shape(mut self, replay_shape: Vec<usize>) -> Self {
        self.replay_shape = replay_shape;
        self
    }

    pub fn with_layout(mut self, layout: ReplayLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn shape_key(&self) -> Vec<usize> {
        self.replay_shape.clone()
    }

    pub fn replay_key(&self, backend: kiln_tensor::Backend) -> ReplayKey {
        ReplayKey::new(
            backend,
            self.kind.operation_name(),
            self.shape_key(),
            self.dtype,
            self.max_batch,
            self.replay_safe,
        )
    }

    pub const fn has_valid_bounds(&self) -> bool {
        self.max_hidden > 0 && self.max_intermediate > 0 && self.max_batch > 0
    }

    fn shape_from_bounds(
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> Vec<usize> {
        vec![max_hidden, max_intermediate, max_batch]
    }
}

/// Snapshot of the existing backend capability predicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilitySnapshot {
    pub backend: &'static str,
    pub device: kiln_tensor::Device,
    pub training: TrainingCapabilities,
    pub resident_decode: Support,
    pub resident_activation: Support,
    pub flash_attn_prefill: Support,
    pub flash_attn_paged_decode: Support,
    pub paged_kv_head_major_read: Support,
    pub gdn_recurrent_step: Support,
    pub causal_conv1d_update: Support,
    pub linear_decode_argmax: Support,
}

impl BackendCapabilitySnapshot {
    pub fn from_backend<T>(backend: &T) -> Self
    where
        T: BackendRuntime
            + BackendIdentity
            + AttentionBackend
            + PagedKvBackend
            + GdnBackend
            + ConvBackend
            + SamplingBackend
            + ResidencyBackend
            + TrainingLossBackend
            + ReplayBackend
            + ?Sized,
    {
        Self {
            backend: BackendIdentity::runtime_name(backend),
            device: BackendIdentity::runtime_device(backend),
            training: TrainingLossBackend::runtime_training_capabilities(backend),
            resident_decode: Support::from_supports_predicate(
                ReplayBackend::runtime_supports_resident_decode(backend),
            ),
            resident_activation: Support::from_supports_predicate(
                ResidencyBackend::runtime_supports_resident_activation(backend),
            ),
            flash_attn_prefill: Support::from_supports_predicate(
                AttentionBackend::runtime_supports_flash_attn_prefill(backend),
            ),
            flash_attn_paged_decode: Support::from_supports_predicate(
                AttentionBackend::runtime_supports_flash_attn_paged_decode(backend),
            ),
            paged_kv_head_major_read: Support::from_supports_predicate(
                PagedKvBackend::runtime_supports_paged_kv_head_major_read(backend),
            ),
            gdn_recurrent_step: Support::from_supports_predicate(
                GdnBackend::runtime_supports_gdn_recurrent_step(backend),
            ),
            causal_conv1d_update: Support::from_supports_predicate(
                ConvBackend::runtime_supports_causal_conv1d_update(backend),
            ),
            linear_decode_argmax: Support::from_supports_predicate(
                SamplingBackend::runtime_supports_linear_decode_argmax(backend),
            ),
        }
    }
}

/// Storage and residency capabilities surfaced through one backend descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageCapabilities {
    pub backend: kiln_tensor::Backend,
    pub device: kiln_tensor::Device,
    pub resident_activation: Support,
    pub resident_decode: Support,
    pub kv_cache_device_memory_pressure: bool,
    pub gpu_memory_detection_policy: GpuMemoryDetectionPolicy,
    pub gpu_memory_budget_policy: GpuMemoryBudgetPolicy,
    pub gpu_memory_reclaim_policy: GpuMemoryReclaimPolicy,
    pub kv_sizing_residency_model_multiplier: u64,
    pub kv_auto_block_policy: KvCacheAutoBlockPolicy,
    pub kv_cache_fp8_policy: KvCacheFp8Policy,
}

/// Backend-owned policy for interpreting server GPU-memory detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemoryDetectionPolicy {
    pub detected_total_log_message: Option<&'static str>,
    pub missing_total_warning: Option<&'static str>,
    pub missing_total_fallback_bytes: Option<u64>,
}

/// Backend-owned policy for server GPU-memory budget probes and retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemoryBudgetPolicy {
    pub use_live_memory_snapshot: bool,
    pub cap_kv_blocks_by_live_budget: bool,
    pub retry_kv_allocation_after_reclaim: bool,
}

/// Backend-owned policy for server GPU-memory pressure reclaim hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemoryReclaimPolicy {
    pub reclaimer: GpuMemoryReclaimer,
}

/// Concrete reclaimer selected by [`GpuMemoryReclaimPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryReclaimer {
    None,
    CudaTrimPool,
    RocmTrimPool,
    LoggedNoop { log_message: &'static str },
}

/// Backend-owned policy for server KV auto-sizing block caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheAutoBlockPolicy {
    pub context_window_cap: bool,
    pub static_max_blocks: Option<usize>,
    pub memory_tier_cap: Option<KvCacheMemoryTierBlockCap>,
    pub allow_min_blocks_below_live_budget: bool,
}

/// Memory-tiered block cap for UMA backends where zero-filled KV pools also
/// compete with system memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheMemoryTierBlockCap {
    pub low_memory_bytes_exclusive: u64,
    pub low_max_blocks: usize,
    pub mid_memory_bytes_exclusive: u64,
    pub mid_max_blocks: usize,
    pub high_max_blocks: usize,
}

/// Backend-owned policy for honoring requested FP8 paged-KV cache storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheFp8Policy {
    pub allow_when_requested_by_default: bool,
    pub explicit_enable_env: Option<&'static str>,
    pub disabled_reason: Option<&'static str>,
}

/// Backend-owned server startup and prewarm policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartupCapabilities {
    pub run_inference_prewarm: bool,
    pub require_inference_prewarm_for_health: bool,
    pub precompile_custom_kernels: bool,
    pub native_training_default_enabled: bool,
    pub native_training_env: Option<&'static str>,
    pub decode_weight_prewarm_when_native_training: bool,
}

/// Representative matmul capability probes backed by [`MatmulRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulCapabilities {
    pub rank2_f32: Support,
    pub batched_bf16: Support,
    pub bias_epilogue: Support,
}

/// Representative attention capability probes backed by [`AttentionRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionCapabilities {
    pub flash_prefill: Support,
    pub flash_prefill_head_major: Support,
    pub flash_paged_decode: Support,
    pub flash_prefill_consumes_grouped_kv: bool,
    pub detached_chunked_prefill: Support,
}

/// Focused GDN capability snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdnCapabilities {
    pub recurrent_step: Support,
    pub recurrent_step_f32: Support,
    pub inference_recurrent_state: InferenceRecurrentStatePolicy,
    pub chunk_prep: Support,
    pub chunk_scan: Support,
    pub full_chunk_forward: Support,
    pub gates: Support,
    pub gated_rms_norm: Support,
    pub gated_rms_norm_preserves_tape_residency: bool,
}

/// Backend-owned dtype policy for GDN recurrent state in inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceRecurrentStatePolicy {
    pub bf16: Support,
    pub f16: Support,
}

/// Decode and lm-head capability snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeCapabilities {
    pub resident_decode: Support,
    pub paged_decode_graph_outputs: Support,
    pub mtp_speculative_generation: Support,
    pub speculative_policy: SpeculativeDecodePolicy,
    pub linear_argmax: Support,
    pub linear_argmax_batch: Support,
    pub linear_sample: Support,
    pub linear_sample_batch: Support,
}

/// Backend-owned speculative decode routing thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeDecodePolicy {
    pub mtp_max_prompt_tokens: usize,
    pub long_prompt_skip_layer_min_prompt_tokens: usize,
    pub long_prompt_skip_layer_min_output_tokens: usize,
}

/// Backend-owned defaults for the live decode rendezvous worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBatcherPolicy {
    pub max_batch: usize,
    pub wait_micros: u64,
    pub allow_mixed_seq_lens: bool,
    pub rowwise_retry_env: Option<&'static str>,
    pub require_native_decode_attention: bool,
    pub prefer_direct_paged_decode_attention: bool,
    pub direct_paged_decode_attention_env_gate: DecodeAttentionEnvGate,
    pub allow_prefix_cache_split_snapshot: bool,
    pub paged_decode_requires_contiguous_kv_chunks: bool,
    pub use_greedy_token_decode: bool,
    pub use_native_sampled_contiguous_decode: bool,
    pub sampled_contiguous_decode_requires_resident_decode: bool,
    pub partition_noncontiguous_gdn_kv_tiles: bool,
    pub use_decode_width_prefill_admission: bool,
    pub burst_prefill_admission: bool,
    pub batching_engine_default_enabled: bool,
    pub warm_resident_decode_pool_on_startup: bool,
}

/// Environment gate attached to backend decode-attention routing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeAttentionEnvGate {
    None,
    DisabledWhenSet(&'static str),
    EnabledUnlessOff(&'static str),
}

impl DecodeAttentionEnvGate {
    pub fn allows(self) -> bool {
        match self {
            Self::None => true,
            Self::DisabledWhenSet(name) => std::env::var(name).is_err(),
            Self::EnabledUnlessOff(name) => std::env::var(name)
                .map(|value| {
                    let value = value.trim().to_ascii_lowercase();
                    !(value == "0" || value == "false" || value == "no" || value == "off")
                })
                .unwrap_or(true),
        }
    }
}

/// Replay capability probes backed by [`ReplayRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCapabilities {
    pub resident_decode: Support,
    pub paged_decode_graph_outputs: Support,
    pub authority: ReplayAuthority,
}

/// Which layer currently owns production replay behavior for a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplayProductionAuthority {
    None,
    ModelLevelRunner,
    ModelLevelRunnerWithGraphCrateReplayObject,
    ResidentDecodeCommandBatch,
}

impl ReplayProductionAuthority {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ModelLevelRunner => "model_level_runner",
            Self::ModelLevelRunnerWithGraphCrateReplayObject => {
                "model_level_runner_with_graph_crate_replay_object"
            }
            Self::ResidentDecodeCommandBatch => "resident_decode_command_batch",
        }
    }
}

/// Native backend replay primitive used by production decode replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplayNativePrimitive {
    None,
    CudaGraph,
    HipGraph,
    MetalIcb,
    VulkanCommandBatch,
}

impl ReplayNativePrimitive {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CudaGraph => "CUDA graph",
            Self::HipGraph => "HIP graph",
            Self::MetalIcb => "Metal ICB",
            Self::VulkanCommandBatch => "Vulkan CommandBatch",
        }
    }
}

/// Role played by the `kiln-graph-*` crate for a backend today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplayGraphCrateRole {
    None,
    Scaffold,
    ReplayObject,
    ResidentPlanScaffold,
}

impl ReplayGraphCrateRole {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Scaffold => "scaffold",
            Self::ReplayObject => "replay_object",
            Self::ResidentPlanScaffold => "resident_plan_scaffold",
        }
    }
}

/// Typed replay authority boundary for runtime diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReplayAuthority {
    pub backend: kiln_tensor::Backend,
    pub production_authority: ReplayProductionAuthority,
    pub native_primitive: ReplayNativePrimitive,
    pub graph_crate_role: ReplayGraphCrateRole,
}

impl ReplayAuthority {
    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        let backend = backend_kind_for_runtime(name, device);
        let (production_authority, native_primitive, graph_crate_role) = match backend {
            kiln_tensor::Backend::Cuda => (
                ReplayProductionAuthority::ModelLevelRunner,
                ReplayNativePrimitive::CudaGraph,
                ReplayGraphCrateRole::Scaffold,
            ),
            kiln_tensor::Backend::Rocm => (
                ReplayProductionAuthority::ModelLevelRunner,
                ReplayNativePrimitive::HipGraph,
                ReplayGraphCrateRole::None,
            ),
            kiln_tensor::Backend::Metal => (
                ReplayProductionAuthority::ModelLevelRunnerWithGraphCrateReplayObject,
                ReplayNativePrimitive::MetalIcb,
                ReplayGraphCrateRole::ReplayObject,
            ),
            kiln_tensor::Backend::Vulkan => (
                ReplayProductionAuthority::ResidentDecodeCommandBatch,
                ReplayNativePrimitive::VulkanCommandBatch,
                ReplayGraphCrateRole::ResidentPlanScaffold,
            ),
            kiln_tensor::Backend::Cpu => (
                ReplayProductionAuthority::None,
                ReplayNativePrimitive::None,
                ReplayGraphCrateRole::None,
            ),
            _ => (
                ReplayProductionAuthority::None,
                ReplayNativePrimitive::None,
                ReplayGraphCrateRole::None,
            ),
        };

        Self {
            backend,
            production_authority,
            native_primitive,
            graph_crate_role,
        }
    }
}

/// Backend fallback policy surface used by hot-path callers and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendFallbackCapabilities {
    pub generic_device_op: FallbackPolicy,
    pub decode_hot_path: FallbackPolicy,
    pub decode_hot_path_debug_env: Option<&'static str>,
    pub training_optimizer: FallbackPolicy,
    pub training_optimizer_debug_env: Option<&'static str>,
}

/// Training capability and dtype policy surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendTrainingCapabilities {
    pub hooks: TrainingCapabilities,
    pub precision: TrainingPrecisionPolicy,
    pub server_dispatch: ServerTrainingDispatchPolicy,
}

/// Server-side native training route selected by backend policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServerTrainingNativeRoute {
    SharedKtTape,
    LegacyCudaNative,
}

impl ServerTrainingNativeRoute {
    pub const fn label(self) -> &'static str {
        match self {
            Self::SharedKtTape => "shared_kt_tape",
            Self::LegacyCudaNative => "legacy_cuda_native",
        }
    }
}

/// Backend-owned SFT/GRPO dispatch policy consumed by `kiln-server`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerTrainingDispatchPolicy {
    pub native_route: ServerTrainingNativeRoute,
    pub native_training_env: Option<&'static str>,
    pub native_training_default_enabled: bool,
}

/// One structured runtime capability descriptor for backend diagnostics.
///
/// This is the data-shaped counterpart to [`BackendCapabilityQueries`]. It
/// intentionally composes today's compatibility predicates and request-shaped
/// probes without changing dispatch behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub backend: &'static str,
    pub device: kiln_tensor::Device,
    pub storage: StorageCapabilities,
    pub startup: StartupCapabilities,
    pub matmul: MatmulCapabilities,
    pub attention: AttentionCapabilities,
    pub gdn: GdnCapabilities,
    pub decode: DecodeCapabilities,
    pub decode_batcher: DecodeBatcherPolicy,
    pub training: BackendTrainingCapabilities,
    pub graph_replay: ReplayCapabilities,
    pub fallback: BackendFallbackCapabilities,
}

impl BackendCapabilities {
    pub fn from_backend<T>(backend: &T) -> Self
    where
        T: BackendRuntime
            + BackendIdentity
            + AttentionBackend
            + PagedKvBackend
            + GdnBackend
            + ConvBackend
            + SamplingBackend
            + ResidencyBackend
            + TrainingLossBackend
            + ReplayBackend
            + ?Sized,
    {
        let device = BackendIdentity::runtime_device(backend);
        let name = BackendIdentity::runtime_name(backend);
        let backend_kind = backend_kind_for_runtime(name, device);

        let rank2_f32 =
            MatmulRequest::plain(vec![2, 3], vec![3, 4], kiln_tensor::DType::F32, false);
        let batched_bf16 =
            MatmulRequest::plain(vec![2, 2, 3], vec![2, 3, 4], kiln_tensor::DType::BF16, true);
        let bias_epilogue = rank2_f32.clone().with_epilogue(MatmulEpilogue::Bias);
        let flash_prefill = AttentionRequest::flash_prefill(
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            1,
            16,
            128,
            false,
        );
        let flash_prefill_head_major = flash_prefill
            .clone()
            .with_kind(AttentionRequestKind::FlashPrefillHeadMajor);
        let flash_paged_decode = flash_prefill
            .clone()
            .with_kind(AttentionRequestKind::FlashPagedDecode)
            .with_dims(1, 1, 128)
            .with_replay_safe(true);
        let linear_argmax = LinearRequest::decode_argmax(
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::I64,
            1,
            true,
        )
        .with_shapes(vec![1, 4096], vec![32000, 4096], vec![1]);
        let linear_argmax_batch = LinearRequest::decode_argmax(
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::I64,
            2,
            true,
        )
        .with_shapes(vec![2, 4096], vec![32000, 4096], vec![2]);
        let linear_sample = LinearRequest::decode_sample(
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::I64,
            vec![8],
            vec![1.0],
            true,
        )
        .with_shapes(vec![1, 4096], vec![32000, 4096], vec![1]);
        let linear_sample_batch = LinearRequest::decode_sample(
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::I64,
            vec![8, 8],
            vec![1.0, 1.0],
            true,
        )
        .with_shapes(vec![2, 4096], vec![32000, 4096], vec![2]);
        let resident_replay =
            ReplayRequest::resident_decode(8, 16, 2).with_dtype(kiln_tensor::DType::BF16);
        let paged_replay = ReplayRequest::paged_decode_graph_outputs(8, 16, 2)
            .with_dtype(kiln_tensor::DType::BF16);

        Self {
            backend: name,
            device,
            storage: StorageCapabilities {
                backend: backend_kind,
                device,
                resident_activation: Support::from_supports_predicate(
                    ResidencyBackend::runtime_supports_resident_activation(backend),
                ),
                resident_decode: Support::from_supports_predicate(
                    ReplayBackend::runtime_supports_resident_decode(backend),
                ),
                kv_cache_device_memory_pressure: kv_cache_device_memory_pressure(name, device),
                gpu_memory_detection_policy: GpuMemoryDetectionPolicy::for_backend(name, device),
                gpu_memory_budget_policy: GpuMemoryBudgetPolicy::for_backend(name, device),
                gpu_memory_reclaim_policy: GpuMemoryReclaimPolicy::for_backend(name, device),
                kv_sizing_residency_model_multiplier: kv_sizing_residency_model_multiplier(
                    name, device,
                ),
                kv_auto_block_policy: KvCacheAutoBlockPolicy::for_backend(name, device),
                kv_cache_fp8_policy: KvCacheFp8Policy::for_backend(name, device),
            },
            startup: StartupCapabilities::for_backend(name, device),
            matmul: MatmulCapabilities {
                rank2_f32: BackendCapabilityQueries::supports_matmul_request(backend, &rank2_f32),
                batched_bf16: BackendCapabilityQueries::supports_matmul_request(
                    backend,
                    &batched_bf16,
                ),
                bias_epilogue: BackendCapabilityQueries::supports_matmul_request(
                    backend,
                    &bias_epilogue,
                ),
            },
            attention: AttentionCapabilities {
                flash_prefill: BackendCapabilityQueries::supports_attention_request(
                    backend,
                    &flash_prefill,
                ),
                flash_prefill_head_major: BackendCapabilityQueries::supports_attention_request(
                    backend,
                    &flash_prefill_head_major,
                ),
                flash_paged_decode: BackendCapabilityQueries::supports_attention_request(
                    backend,
                    &flash_paged_decode,
                ),
                flash_prefill_consumes_grouped_kv: flash_prefill_consumes_grouped_kv(name),
                detached_chunked_prefill: detached_chunked_prefill_support(name),
            },
            gdn: GdnCapabilities {
                recurrent_step: Support::from_supports_predicate(
                    GdnBackend::runtime_supports_gdn_recurrent_step(backend),
                ),
                recurrent_step_f32: gdn_recurrent_step_f32_support(
                    name,
                    GdnBackend::runtime_supports_gdn_recurrent_step(backend),
                ),
                inference_recurrent_state: InferenceRecurrentStatePolicy::for_backend(name, device),
                chunk_prep: Support::from_supports_predicate(
                    GdnBackend::runtime_supports_gdn_chunk_prep(backend),
                ),
                chunk_scan: Support::from_supports_predicate(
                    GdnBackend::runtime_supports_gdn_chunk_scan(backend),
                ),
                full_chunk_forward: Support::from_supports_predicate(
                    GdnBackend::runtime_supports_gdn_full_chunk_forward(backend),
                ),
                gates: Support::from_supports_predicate(GdnBackend::runtime_supports_gdn_gates(
                    backend,
                )),
                gated_rms_norm: Support::from_supports_predicate(
                    GdnBackend::runtime_supports_gdn_gated_rms_norm(backend),
                ),
                gated_rms_norm_preserves_tape_residency:
                    gdn_gated_rms_norm_preserves_tape_residency(name, device),
            },
            decode: DecodeCapabilities {
                resident_decode: Support::from_supports_predicate(
                    ReplayBackend::runtime_supports_resident_decode(backend),
                ),
                paged_decode_graph_outputs: BackendCapabilityQueries::supports_replay_request(
                    backend,
                    &paged_replay,
                ),
                mtp_speculative_generation: mtp_speculative_generation_support(name),
                speculative_policy: SpeculativeDecodePolicy::for_backend(name, device),
                linear_argmax: BackendCapabilityQueries::supports_linear_request(
                    backend,
                    &linear_argmax,
                ),
                linear_argmax_batch: BackendCapabilityQueries::supports_linear_request(
                    backend,
                    &linear_argmax_batch,
                ),
                linear_sample: BackendCapabilityQueries::supports_linear_request(
                    backend,
                    &linear_sample,
                ),
                linear_sample_batch: BackendCapabilityQueries::supports_linear_request(
                    backend,
                    &linear_sample_batch,
                ),
            },
            decode_batcher: DecodeBatcherPolicy::for_backend(name, device),
            training: BackendTrainingCapabilities {
                hooks: TrainingLossBackend::runtime_training_capabilities(backend),
                precision: TrainingLossBackend::runtime_training_precision_policy(backend),
                server_dispatch: ServerTrainingDispatchPolicy::for_backend(name, device),
            },
            graph_replay: ReplayCapabilities {
                resident_decode: BackendCapabilityQueries::supports_replay_request(
                    backend,
                    &resident_replay,
                ),
                paged_decode_graph_outputs: BackendCapabilityQueries::supports_replay_request(
                    backend,
                    &paged_replay,
                ),
                authority: ReplayBackend::runtime_replay_authority(backend),
            },
            fallback: BackendFallbackCapabilities::for_backend(name, device),
        }
    }
}

impl StartupCapabilities {
    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        match backend_kind_for_runtime(name, device) {
            kiln_tensor::Backend::Metal => Self {
                run_inference_prewarm: true,
                require_inference_prewarm_for_health: true,
                precompile_custom_kernels: true,
                native_training_default_enabled: false,
                native_training_env: None,
                decode_weight_prewarm_when_native_training: false,
            },
            kiln_tensor::Backend::Vulkan => Self {
                run_inference_prewarm: true,
                require_inference_prewarm_for_health: true,
                precompile_custom_kernels: true,
                native_training_default_enabled: true,
                native_training_env: Some("KILN_VK_NATIVE_TRAINING"),
                decode_weight_prewarm_when_native_training: true,
            },
            kiln_tensor::Backend::Rocm => Self {
                run_inference_prewarm: true,
                require_inference_prewarm_for_health: false,
                precompile_custom_kernels: false,
                native_training_default_enabled: false,
                native_training_env: None,
                decode_weight_prewarm_when_native_training: false,
            },
            _ => Self {
                run_inference_prewarm: false,
                require_inference_prewarm_for_health: false,
                precompile_custom_kernels: false,
                native_training_default_enabled: false,
                native_training_env: None,
                decode_weight_prewarm_when_native_training: false,
            },
        }
    }
}

impl ServerTrainingDispatchPolicy {
    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        match backend_kind_for_runtime(name, device) {
            kiln_tensor::Backend::Cuda => Self {
                native_route: ServerTrainingNativeRoute::LegacyCudaNative,
                native_training_env: Some("KILN_CUDA_NATIVE_TRAINING"),
                native_training_default_enabled: false,
            },
            _ => Self {
                native_route: ServerTrainingNativeRoute::SharedKtTape,
                native_training_env: None,
                native_training_default_enabled: false,
            },
        }
    }

    pub fn native_route_enabled(self) -> bool {
        match self.native_route {
            ServerTrainingNativeRoute::LegacyCudaNative => self
                .native_training_env
                .map(|env| kiln_core::env_flag::env_flag(env, self.native_training_default_enabled))
                .unwrap_or(self.native_training_default_enabled),
            ServerTrainingNativeRoute::SharedKtTape => false,
        }
    }
}

impl SpeculativeDecodePolicy {
    pub const MTP_MAX_PROMPT_TOKENS_DEFAULT: usize = 128;
    pub const LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT: usize = 1024;
    pub const LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL: usize = 4096;
    pub const LONG_PROMPT_SKIP_LAYER_MIN_OUTPUT_TOKENS_DEFAULT: usize = 32;

    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        let long_prompt_skip_layer_min_prompt_tokens = match backend_kind_for_runtime(name, device)
        {
            kiln_tensor::Backend::Metal => Self::LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL,
            _ => Self::LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        };
        Self {
            mtp_max_prompt_tokens: Self::MTP_MAX_PROMPT_TOKENS_DEFAULT,
            long_prompt_skip_layer_min_prompt_tokens,
            long_prompt_skip_layer_min_output_tokens:
                Self::LONG_PROMPT_SKIP_LAYER_MIN_OUTPUT_TOKENS_DEFAULT,
        }
    }
}

impl Default for SpeculativeDecodePolicy {
    fn default() -> Self {
        Self::for_backend("cpu", kiln_tensor::Device::Cpu)
    }
}

impl DecodeBatcherPolicy {
    pub const DEFAULT_MAX_BATCH: usize = 8;
    pub const VULKAN_MAX_BATCH: usize = 64;
    pub const METAL_WAIT_MICROS: u64 = 100;
    pub const VULKAN_WAIT_MICROS: u64 = 5_000;

    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        match backend_kind_for_runtime(name, device) {
            kiln_tensor::Backend::Cuda => Self {
                max_batch: 1,
                wait_micros: 0,
                allow_mixed_seq_lens: false,
                rowwise_retry_env: None,
                require_native_decode_attention: false,
                prefer_direct_paged_decode_attention: true,
                direct_paged_decode_attention_env_gate: DecodeAttentionEnvGate::DisabledWhenSet(
                    "KILN_DISABLE_CUDA_DIRECT_PAGED_DECODE",
                ),
                allow_prefix_cache_split_snapshot: true,
                paged_decode_requires_contiguous_kv_chunks: true,
                use_greedy_token_decode: false,
                use_native_sampled_contiguous_decode: false,
                sampled_contiguous_decode_requires_resident_decode: false,
                partition_noncontiguous_gdn_kv_tiles: true,
                use_decode_width_prefill_admission: true,
                burst_prefill_admission: true,
                batching_engine_default_enabled: true,
                warm_resident_decode_pool_on_startup: false,
            },
            kiln_tensor::Backend::Metal => Self {
                max_batch: Self::DEFAULT_MAX_BATCH,
                wait_micros: Self::METAL_WAIT_MICROS,
                allow_mixed_seq_lens: true,
                rowwise_retry_env: None,
                require_native_decode_attention: false,
                prefer_direct_paged_decode_attention: false,
                direct_paged_decode_attention_env_gate: DecodeAttentionEnvGate::None,
                allow_prefix_cache_split_snapshot: true,
                paged_decode_requires_contiguous_kv_chunks: true,
                use_greedy_token_decode: true,
                use_native_sampled_contiguous_decode: true,
                sampled_contiguous_decode_requires_resident_decode: false,
                partition_noncontiguous_gdn_kv_tiles: false,
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                batching_engine_default_enabled: false,
                warm_resident_decode_pool_on_startup: false,
            },
            kiln_tensor::Backend::Vulkan => Self {
                max_batch: Self::VULKAN_MAX_BATCH,
                wait_micros: Self::VULKAN_WAIT_MICROS,
                allow_mixed_seq_lens: true,
                rowwise_retry_env: Some("KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY"),
                require_native_decode_attention: true,
                prefer_direct_paged_decode_attention: true,
                direct_paged_decode_attention_env_gate: DecodeAttentionEnvGate::None,
                allow_prefix_cache_split_snapshot: true,
                paged_decode_requires_contiguous_kv_chunks: false,
                use_greedy_token_decode: false,
                use_native_sampled_contiguous_decode: true,
                sampled_contiguous_decode_requires_resident_decode: true,
                partition_noncontiguous_gdn_kv_tiles: false,
                use_decode_width_prefill_admission: true,
                burst_prefill_admission: false,
                batching_engine_default_enabled: true,
                warm_resident_decode_pool_on_startup: true,
            },
            kiln_tensor::Backend::Rocm => Self {
                max_batch: Self::DEFAULT_MAX_BATCH,
                wait_micros: 0,
                allow_mixed_seq_lens: false,
                rowwise_retry_env: None,
                require_native_decode_attention: false,
                prefer_direct_paged_decode_attention: true,
                direct_paged_decode_attention_env_gate: DecodeAttentionEnvGate::EnabledUnlessOff(
                    "KILN_ROCM_PAGED_DECODE",
                ),
                allow_prefix_cache_split_snapshot: false,
                paged_decode_requires_contiguous_kv_chunks: true,
                use_greedy_token_decode: false,
                use_native_sampled_contiguous_decode: false,
                sampled_contiguous_decode_requires_resident_decode: false,
                partition_noncontiguous_gdn_kv_tiles: false,
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                batching_engine_default_enabled: true,
                warm_resident_decode_pool_on_startup: false,
            },
            _ => Self {
                max_batch: Self::DEFAULT_MAX_BATCH,
                wait_micros: 0,
                allow_mixed_seq_lens: false,
                rowwise_retry_env: None,
                require_native_decode_attention: false,
                prefer_direct_paged_decode_attention: false,
                direct_paged_decode_attention_env_gate: DecodeAttentionEnvGate::None,
                allow_prefix_cache_split_snapshot: true,
                paged_decode_requires_contiguous_kv_chunks: true,
                use_greedy_token_decode: false,
                use_native_sampled_contiguous_decode: false,
                sampled_contiguous_decode_requires_resident_decode: false,
                partition_noncontiguous_gdn_kv_tiles: false,
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                batching_engine_default_enabled: true,
                warm_resident_decode_pool_on_startup: false,
            },
        }
    }

    pub fn direct_paged_decode_attention_enabled(self) -> bool {
        self.prefer_direct_paged_decode_attention
            && self.direct_paged_decode_attention_env_gate.allows()
    }
}

fn flash_prefill_consumes_grouped_kv(name: &str) -> bool {
    matches!(name, "cuda")
}

fn detached_chunked_prefill_support(name: &str) -> Support {
    match name {
        "cuda" => Support::NativeWithConstraints,
        _ => Support::Declined,
    }
}

fn mtp_speculative_generation_support(name: &str) -> Support {
    match name {
        "cuda" => Support::NativeWithConstraints,
        "metal" if kiln_core::env_flag::env_flag("KILN_ENABLE_METAL_NATIVE_MTP", false) => {
            Support::NativeWithConstraints
        }
        "metal" => Support::DisabledByEnv,
        _ => Support::Declined,
    }
}

fn kv_cache_device_memory_pressure(name: &str, device: kiln_tensor::Device) -> bool {
    matches!(
        backend_kind_for_runtime(name, device),
        kiln_tensor::Backend::Cuda | kiln_tensor::Backend::Rocm
    )
}

fn kv_sizing_residency_model_multiplier(name: &str, device: kiln_tensor::Device) -> u64 {
    match backend_kind_for_runtime(name, device) {
        kiln_tensor::Backend::Vulkan => 2,
        _ => 0,
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

impl GpuMemoryDetectionPolicy {
    pub const DETECTED_TOTAL_LOG_DEFAULT: &'static str = "GPU VRAM detected";
    pub const DETECTED_TOTAL_LOG_METAL: &'static str = "unified memory detected (Apple Silicon)";
    pub const CUDA_MISSING_TOTAL_WARNING: &'static str =
        "CUDA device present but VRAM detection failed";
    pub const METAL_MISSING_TOTAL_WARNING: &'static str =
        "Metal device present but unified memory detection failed";
    pub const CUDA_MISSING_TOTAL_FALLBACK_BYTES: u64 = 24 * GIB;
    pub const METAL_MISSING_TOTAL_FALLBACK_BYTES: u64 = 16 * GIB;

    pub const DETECTED_TOTAL_ONLY: Self = Self {
        detected_total_log_message: None,
        missing_total_warning: None,
        missing_total_fallback_bytes: None,
    };

    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        match backend_kind_for_runtime(name, device) {
            kiln_tensor::Backend::Cuda => Self {
                detected_total_log_message: Some(Self::DETECTED_TOTAL_LOG_DEFAULT),
                missing_total_warning: Some(Self::CUDA_MISSING_TOTAL_WARNING),
                missing_total_fallback_bytes: Some(Self::CUDA_MISSING_TOTAL_FALLBACK_BYTES),
            },
            kiln_tensor::Backend::Metal => Self {
                detected_total_log_message: Some(Self::DETECTED_TOTAL_LOG_METAL),
                missing_total_warning: Some(Self::METAL_MISSING_TOTAL_WARNING),
                missing_total_fallback_bytes: Some(Self::METAL_MISSING_TOTAL_FALLBACK_BYTES),
            },
            _ => Self::DETECTED_TOTAL_ONLY,
        }
    }

    pub fn total_memory_bytes(self, detected_total_bytes: u64) -> u64 {
        if detected_total_bytes > 0 {
            detected_total_bytes
        } else {
            self.missing_total_fallback_bytes
                .unwrap_or(detected_total_bytes)
        }
    }
}

impl GpuMemoryBudgetPolicy {
    pub const DEVICE_MEMORY_AWARE: Self = Self {
        use_live_memory_snapshot: true,
        cap_kv_blocks_by_live_budget: true,
        retry_kv_allocation_after_reclaim: true,
    };

    pub const HOST_MEMORY_ONLY: Self = Self {
        use_live_memory_snapshot: false,
        cap_kv_blocks_by_live_budget: false,
        retry_kv_allocation_after_reclaim: false,
    };

    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        if matches!(
            backend_kind_for_runtime(name, device),
            kiln_tensor::Backend::Cpu
        ) {
            Self::HOST_MEMORY_ONLY
        } else {
            Self::DEVICE_MEMORY_AWARE
        }
    }
}

impl GpuMemoryReclaimPolicy {
    pub const METAL_LOGGED_NOOP_MESSAGE: &'static str =
        "metal reclaimer: UMA, no pool to trim (no-op)";
    pub const VULKAN_LOGGED_NOOP_MESSAGE: &'static str =
        "vulkan reclaimer: cache-drain not yet implemented (no-op)";

    pub const NONE: Self = Self {
        reclaimer: GpuMemoryReclaimer::None,
    };

    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        let reclaimer = match backend_kind_for_runtime(name, device) {
            kiln_tensor::Backend::Cuda => GpuMemoryReclaimer::CudaTrimPool,
            kiln_tensor::Backend::Rocm => GpuMemoryReclaimer::RocmTrimPool,
            kiln_tensor::Backend::Metal => GpuMemoryReclaimer::LoggedNoop {
                log_message: Self::METAL_LOGGED_NOOP_MESSAGE,
            },
            kiln_tensor::Backend::Vulkan => GpuMemoryReclaimer::LoggedNoop {
                log_message: Self::VULKAN_LOGGED_NOOP_MESSAGE,
            },
            kiln_tensor::Backend::Cpu => GpuMemoryReclaimer::None,
            _ => GpuMemoryReclaimer::None,
        };
        Self { reclaimer }
    }
}

impl KvCacheAutoBlockPolicy {
    pub const MEMORY_BUDGET_ONLY: Self = Self {
        context_window_cap: false,
        static_max_blocks: None,
        memory_tier_cap: None,
        allow_min_blocks_below_live_budget: false,
    };

    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        match backend_kind_for_runtime(name, device) {
            kiln_tensor::Backend::Metal => Self {
                context_window_cap: true,
                static_max_blocks: None,
                memory_tier_cap: Some(KvCacheMemoryTierBlockCap {
                    low_memory_bytes_exclusive: 14 * GIB,
                    low_max_blocks: 512,
                    mid_memory_bytes_exclusive: 24 * GIB,
                    mid_max_blocks: 1024,
                    high_max_blocks: 2048,
                }),
                allow_min_blocks_below_live_budget: false,
            },
            kiln_tensor::Backend::Rocm => Self {
                context_window_cap: true,
                static_max_blocks: Some(4096),
                memory_tier_cap: None,
                allow_min_blocks_below_live_budget: true,
            },
            _ => Self::MEMORY_BUDGET_ONLY,
        }
    }

    pub fn runtime_cap_blocks(
        self,
        max_position_embeddings: usize,
        block_size: usize,
        min_blocks: usize,
        total_vram_bytes: u64,
    ) -> usize {
        let mut cap = if self.context_window_cap {
            max_position_embeddings.div_ceil(block_size).max(min_blocks)
        } else {
            usize::MAX
        };
        if let Some(static_max) = self.static_max_blocks {
            cap = cap.min(static_max);
        }
        if let Some(tier) = self.memory_tier_cap {
            cap = cap.min(tier.max_blocks_for_total_vram(total_vram_bytes));
        }
        cap
    }
}

impl KvCacheMemoryTierBlockCap {
    pub fn max_blocks_for_total_vram(self, total_vram_bytes: u64) -> usize {
        if total_vram_bytes < self.low_memory_bytes_exclusive {
            self.low_max_blocks
        } else if total_vram_bytes < self.mid_memory_bytes_exclusive {
            self.mid_max_blocks
        } else {
            self.high_max_blocks
        }
    }
}

impl KvCacheFp8Policy {
    pub const ALLOW_WHEN_REQUESTED: Self = Self {
        allow_when_requested_by_default: true,
        explicit_enable_env: None,
        disabled_reason: None,
    };

    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        match backend_kind_for_runtime(name, device) {
            kiln_tensor::Backend::Metal => Self {
                allow_when_requested_by_default: false,
                explicit_enable_env: Some("KILN_ALLOW_FP8_ON_METAL"),
                disabled_reason: Some("CPU round-trip cost"),
            },
            _ => Self::ALLOW_WHEN_REQUESTED,
        }
    }

    pub fn enabled(self, requested: bool) -> bool {
        if !requested {
            return false;
        }
        self.allow_when_requested_by_default || self.explicit_enable_env_is_truthy()
    }

    pub fn explicit_enable_env_is_truthy(self) -> bool {
        let Some(name) = self.explicit_enable_env else {
            return false;
        };
        matches!(
            std::env::var(name).as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    }
}

fn gdn_gated_rms_norm_preserves_tape_residency(name: &str, device: kiln_tensor::Device) -> bool {
    match backend_kind_for_runtime(name, device) {
        kiln_tensor::Backend::Cuda | kiln_tensor::Backend::Rocm | kiln_tensor::Backend::Metal => {
            true
        }
        _ => false,
    }
}

impl InferenceRecurrentStatePolicy {
    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        match backend_kind_for_runtime(name, device) {
            kiln_tensor::Backend::Cuda => {
                let support = support_unless_env_set("KILN_DISABLE_CUDA_BF16_INFERENCE_STATE");
                Self {
                    bf16: support,
                    f16: support,
                }
            }
            kiln_tensor::Backend::Rocm => {
                let support = support_unless_env_set("KILN_DISABLE_ROCM_BF16_INFERENCE_STATE");
                Self {
                    bf16: support,
                    f16: support,
                }
            }
            kiln_tensor::Backend::Metal => Self {
                bf16: Support::NativeWithConstraints,
                f16: Support::NativeWithConstraints,
            },
            kiln_tensor::Backend::Vulkan => {
                let support = support_unless_env_set("KILN_DISABLE_VULKAN_BF16_INFERENCE_STATE");
                Self {
                    bf16: support,
                    f16: support,
                }
            }
            _ => Self {
                bf16: Support::Declined,
                f16: Support::Declined,
            },
        }
    }

    pub fn supports_dtype(self, dtype: kiln_tensor::DType) -> bool {
        match dtype {
            kiln_tensor::DType::BF16 => self.bf16.is_native(),
            kiln_tensor::DType::F16 => self.f16.is_native(),
            _ => false,
        }
    }
}

fn support_unless_env_set(env_var: &'static str) -> Support {
    if std::env::var(env_var).is_ok() {
        Support::DisabledByEnv
    } else {
        Support::NativeWithConstraints
    }
}

fn gdn_recurrent_step_f32_support(name: &str, recurrent_step_supported: bool) -> Support {
    match name {
        "vulkan" if std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_STEP_F32").is_ok() => {
            Support::DisabledByEnv
        }
        "vulkan" if recurrent_step_supported => Support::NativeWithConstraints,
        _ => Support::Declined,
    }
}

impl BackendFallbackCapabilities {
    pub fn for_backend(name: &str, device: kiln_tensor::Device) -> Self {
        Self {
            generic_device_op: generic_device_op_fallback_policy(name, device),
            decode_hot_path: decode_hot_path_fallback_policy(name, device),
            decode_hot_path_debug_env: decode_hot_path_debug_fallback_env(name, device),
            training_optimizer: training_optimizer_fallback_policy(name, device),
            training_optimizer_debug_env: training_optimizer_debug_fallback_env(name, device),
        }
    }

    pub fn decode_hot_path_debug_fallback_enabled(self) -> bool {
        kiln_core::env_flag::env_flag("KILN_DECODE_HOT_PATH_DEBUG_FALLBACK", false)
            || self
                .decode_hot_path_debug_env
                .map(|env_var| kiln_core::env_flag::env_flag(env_var, false))
                .unwrap_or(false)
    }
}

fn generic_device_op_fallback_policy(name: &str, _device: kiln_tensor::Device) -> FallbackPolicy {
    match name {
        "cpu" => FallbackPolicy::CorrectnessAllowed,
        "cuda" => FallbackPolicy::NativeRequired,
        "metal" | "vulkan" | "rocm" => FallbackPolicy::WarnAndCount,
        _ => FallbackPolicy::ErrorInHotPath,
    }
}

fn decode_hot_path_fallback_policy(name: &str, _device: kiln_tensor::Device) -> FallbackPolicy {
    match name {
        "cpu" | "cuda" => FallbackPolicy::CorrectnessAllowed,
        "metal" | "vulkan" | "rocm" => FallbackPolicy::NativeRequired,
        _ => FallbackPolicy::ErrorInHotPath,
    }
}

fn decode_hot_path_debug_fallback_env(
    name: &str,
    _device: kiln_tensor::Device,
) -> Option<&'static str> {
    match name {
        "metal" => Some("KILN_METAL_DECODE_BATCH_GENERIC_FALLBACK"),
        "vulkan" => Some("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK"),
        "rocm" => Some("KILN_ROCM_DECODE_BATCH_GENERIC_FALLBACK"),
        _ => None,
    }
}

fn training_optimizer_fallback_policy(name: &str, _device: kiln_tensor::Device) -> FallbackPolicy {
    match name {
        "cpu" => FallbackPolicy::CorrectnessAllowed,
        "cuda" | "metal" | "vulkan" | "rocm" => FallbackPolicy::NativeRequired,
        _ => FallbackPolicy::ErrorInHotPath,
    }
}

fn training_optimizer_debug_fallback_env(
    name: &str,
    _device: kiln_tensor::Device,
) -> Option<&'static str> {
    match name {
        "cuda" => Some("KILN_CUDA_TRAINING_OPTIMIZER_FALLBACK"),
        "metal" => Some("KILN_METAL_TRAINING_OPTIMIZER_FALLBACK"),
        "vulkan" => Some("KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK"),
        "rocm" => Some("KILN_ROCM_TRAINING_OPTIMIZER_FALLBACK"),
        _ => None,
    }
}

fn backend_kind_for_runtime(name: &str, device: kiln_tensor::Device) -> kiln_tensor::Backend {
    match name {
        "cpu" => kiln_tensor::Backend::Cpu,
        "cuda" => kiln_tensor::Backend::Cuda,
        "metal" => kiln_tensor::Backend::Metal,
        "vulkan" => kiln_tensor::Backend::Vulkan,
        "rocm" => kiln_tensor::Backend::Rocm,
        _ => device.backend(),
    }
}

/// Request-shaped capability query surface backed by the focused runtime facets.
///
/// This is the compatibility bridge for the target architecture: call sites can
/// start asking request-shaped questions while existing backends keep their
/// current bool predicates and shape gates behind the focused facets.
pub trait BackendCapabilityQueries:
    BackendRuntime
    + BackendIdentity
    + AttentionBackend
    + SamplingBackend
    + ReplayBackend
    + PagedKvBackend
    + GdnBackend
    + ConvBackend
    + ResidencyBackend
    + TrainingLossBackend
{
    fn capability_snapshot(&self) -> BackendCapabilitySnapshot {
        BackendCapabilitySnapshot::from_backend(self)
    }

    fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::from_backend(self)
    }

    fn supports_attention_request(&self, req: &AttentionRequest) -> Support {
        Support::from_supports_predicate(match req.kind {
            AttentionRequestKind::FlashPrefill => {
                AttentionBackend::runtime_supports_flash_attn_prefill(self)
            }
            AttentionRequestKind::FlashPrefillHeadMajor => {
                AttentionBackend::runtime_supports_flash_attn_prefill_head_major(self)
            }
            AttentionRequestKind::FlashPagedDecode => {
                AttentionBackend::runtime_supports_flash_attn_paged_decode(self)
            }
        })
    }

    fn supports_matmul_request(&self, req: &MatmulRequest) -> Support {
        if !req.has_compatible_shapes()
            || !req.has_supported_dtype_contract()
            || !req.is_row_major_output()
            || !req.is_row_major_input()
            || !matches!(
                req.epilogue,
                MatmulEpilogue::Identity | MatmulEpilogue::Bias
            )
        {
            return Support::Unsupported;
        }

        let Some(rank) = req.rank() else {
            return Support::Unsupported;
        };
        let native = match self.name() {
            "cpu" => matches!(
                req.epilogue,
                MatmulEpilogue::Identity | MatmulEpilogue::Bias
            ),
            "cuda" | "rocm" => match req.epilogue {
                MatmulEpilogue::Identity => true,
                MatmulEpilogue::Bias => rank == 2,
                _ => false,
            },
            "metal" => {
                req.lhs_dtype == kiln_tensor::DType::BF16
                    && matches!(req.epilogue, MatmulEpilogue::Identity)
            }
            "vulkan" => {
                matches!(req.epilogue, MatmulEpilogue::Identity)
                    && (req.lhs_dtype == kiln_tensor::DType::F32
                        || req.lhs_dtype == kiln_tensor::DType::BF16 && rank > 2)
            }
            _ => false,
        };

        if native {
            Support::NativeWithConstraints
        } else {
            Support::HostFallbackAllowed
        }
    }

    fn supports_linear_request(&self, req: &LinearRequest) -> Support {
        Support::from_supports_predicate(match req.kind {
            LinearRequestKind::DecodeArgmax => {
                SamplingBackend::runtime_supports_linear_decode_argmax(self)
            }
            LinearRequestKind::DecodeArgmaxBatch => {
                SamplingBackend::runtime_supports_linear_decode_argmax_batch(self)
            }
            LinearRequestKind::DecodeSample => req
                .top_k
                .first()
                .copied()
                .map(|top_k| SamplingBackend::runtime_supports_linear_decode_sample(self, top_k))
                .unwrap_or(false),
            LinearRequestKind::DecodeSampleBatch => {
                SamplingBackend::runtime_supports_linear_decode_sample_batch(
                    self,
                    &req.top_k,
                    &req.temperatures,
                )
            }
        })
    }

    fn supports_replay_request(&self, req: &ReplayRequest) -> Support {
        if !req.replay_safe || !req.has_valid_bounds() {
            return Support::Unsupported;
        }

        ReplayBackend::runtime_supports_replay_request(self, req)
    }

    fn replay_key_for_request(&self, req: &ReplayRequest) -> ReplayKey {
        ReplayBackend::runtime_replay_key_for_request(self, req)
    }
}

impl<T> BackendCapabilityQueries for T where
    T: BackendRuntime
        + BackendIdentity
        + AttentionBackend
        + SamplingBackend
        + ReplayBackend
        + PagedKvBackend
        + GdnBackend
        + ConvBackend
        + ResidencyBackend
        + TrainingLossBackend
        + ?Sized
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_authority_maps_backends_to_native_primitives() {
        for (name, device, production_authority, primitive, graph_crate_role) in [
            (
                "cpu",
                kiln_tensor::Device::Cpu,
                ReplayProductionAuthority::None,
                ReplayNativePrimitive::None,
                ReplayGraphCrateRole::None,
            ),
            (
                "cuda",
                kiln_tensor::Device::Cuda(0),
                ReplayProductionAuthority::ModelLevelRunner,
                ReplayNativePrimitive::CudaGraph,
                ReplayGraphCrateRole::Scaffold,
            ),
            (
                "rocm",
                kiln_tensor::Device::Rocm(0),
                ReplayProductionAuthority::ModelLevelRunner,
                ReplayNativePrimitive::HipGraph,
                ReplayGraphCrateRole::None,
            ),
            (
                "metal",
                kiln_tensor::Device::Metal(0),
                ReplayProductionAuthority::ModelLevelRunnerWithGraphCrateReplayObject,
                ReplayNativePrimitive::MetalIcb,
                ReplayGraphCrateRole::ReplayObject,
            ),
            (
                "vulkan",
                kiln_tensor::Device::Vulkan(0),
                ReplayProductionAuthority::ResidentDecodeCommandBatch,
                ReplayNativePrimitive::VulkanCommandBatch,
                ReplayGraphCrateRole::ResidentPlanScaffold,
            ),
        ] {
            let authority = ReplayAuthority::for_backend(name, device);
            assert_eq!(authority.backend, device.backend());
            assert_eq!(authority.production_authority, production_authority);
            assert_eq!(authority.native_primitive, primitive);
            assert_eq!(authority.graph_crate_role, graph_crate_role);
            assert_ne!(authority.native_primitive.label(), "");
            assert_ne!(authority.production_authority.label(), "");
            assert_ne!(authority.graph_crate_role.label(), "");
        }
    }
}
