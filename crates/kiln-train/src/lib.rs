//! Bounded in-process LoRA training for Kiln.
//!
//! Native SFT is the versioned `native_online_lora_v1` microtrainer: it runs
//! one admitted conversation and one optimizer update at a time against the
//! already-loaded model. It is not a general distributed training framework.
//! GRPO and OPD have their own explicitly versioned contracts in this crate.

pub mod adapter_output;
pub mod adapter_shape;
pub mod checkpoint;
// (#1082) Per-crate candle facade — every `candle_core::` path that
// `trainer.rs` previously held inline (type aliases, generic constructor
// helpers, safetensors I/O shims, `cd_bail!` macro) now lives in this
// module. Keeps `trainer.rs` at one direct candle reference (the
// `CustomOp1` trait impl, which cannot be type-aliased on stable Rust).
pub(crate) mod cd_types;
#[cfg(feature = "cuda")]
pub mod cuda_train;
pub mod diagnostics;
// (#1082) `pub mod echo;` deleted: ECHO's only caller was the OPD candle
// gradient-checkpointing path (`opd_step_forward_backward_candle`), which
// the candle-drop removed. ECHO's FLCE composite has no kt-tape coverage,
// so it dropped out with the candle path. Re-add a kt-native ECHO module
// when the FLCE env-CE term gets a tape adapter.
// (#1082) Candle↔kt boundary for the SFT/FLCE trainer — relocated out of
// `kiln-flce-kernel` so that kernel crate became 100% candle-free (the
// THIRD kernel-crate candle drop, after `kiln-opd-loss-kernel` and
// `kiln-rmsnorm-kernel`). Holds the candle-typed `FlceMatmulProvider` trait,
// the pure-candle Phase A reference, the Phase B candle `CustomOp1`, the
// (#1082 candle-drop) `flce_candle_shim` deleted — the candle FLCE CustomOp +
// `FlceMatmulProvider` (the `KILN_CUDA_FLCE` provider opt-in) are gone; FLCE is
// kt-native via `kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_kt`.
// (#1082) CP-4: GRPO policy-gradient scalar-loss tape root. The
// candle↔kt boundary for the GRPO trainer's tape-authoritative path —
// a single fused tape node taking the full `[1, T, V]` policy logits and
// producing the scalar PG (+ optional KL) loss, whose backward recomputes
// the candle GRPO forward with autograd ON to yield `dL/dlogits`. Mirrors
// `kiln_model::tape_forward::try_tape_cross_entropy_from_logits_cuda` (the
// SFT loss root) and `opd_tape_shim::try_tape_opd_scalar_mean_cuda` (OPD).
pub mod grpo_tape_shim;
pub mod hf_grpo_interop;
pub mod hf_interop;
pub mod hf_interop_bundle;
pub mod logit_cache;
pub mod logit_source;
pub mod long_context_fixture;
pub mod lora_scaling;
pub mod opd;
// (#1082) Candle↔kt boundary for the OPD trainer — relocated out of
// `kiln-opd-loss-kernel` so that kernel crate became 100% candle-free
// (the first kernel-crate candle drop). Holds the pure-candle Phase A
// reference path, the candle `CustomOp1`-based kt-forward-op shim, and
// the kt-tape production-caller adapters. See module docstring. (#1082)
pub mod opd_tape_shim;
pub mod pi_trajectory;
pub mod receipt;
pub mod remote_teacher;
pub mod replay;
pub mod sft_ingestion;
pub mod sft_tape_shim;
// CP-4 substrate pilot — `kiln_autograd::Tape`-based parallel training
// entry. Sits alongside the candle-typed `trainer` module so future PRs
// can extend it to cover the full per-step graph. See module docstring
// + `docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`. (#1082)
pub mod tape_step;
pub mod teacher_identity;
pub mod train_receipt;
pub mod trainer;
pub mod trajectory;
pub mod trajectory_inspect;
pub mod trajectory_mask;

/// Number of full hidden-state boundaries retained by GRPO and OPD when a
/// checkpoint plan has `num_segments` segments. The forward pass stores the
/// embedded input plus every segment output until reverse replay completes.
pub const fn retained_checkpoint_boundary_count(num_segments: usize) -> usize {
    num_segments.saturating_add(1)
}

/// Default sequence length at which SFT starts replaying sparse checkpoint
/// boundaries instead of retaining every segment input.
pub const DEFAULT_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS: usize = 8192;

/// Default memory target for sparse SFT checkpoint-boundary anchors (6 GiB).
pub const DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// Immutable sparse-boundary dispatch for checkpointed SFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointBoundaryRecomputeMode {
    Auto,
    Enabled,
    Disabled,
}

impl CheckpointBoundaryRecomputeMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }
}

impl std::fmt::Display for CheckpointBoundaryRecomputeMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Process-lifetime policy for retaining or replaying checkpointed SFT
/// segment boundaries.
///
/// Configuration resolves GiB and optional values before constructing this
/// type. The runtime stores only validated integral values so admission,
/// execution, and exact-resume identity cannot disagree because of parsing or
/// floating-point conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CheckpointBoundaryPolicy {
    #[serde(rename = "recompute_mode")]
    recompute_mode: CheckpointBoundaryRecomputeMode,
    #[serde(rename = "recompute_threshold_tokens")]
    recompute_threshold_tokens: usize,
    #[serde(rename = "anchor_stride")]
    anchor_stride: Option<usize>,
    #[serde(rename = "cache_target_bytes")]
    cache_target_bytes: u64,
}

impl CheckpointBoundaryPolicy {
    pub const DEFAULT: Self = Self {
        recompute_mode: CheckpointBoundaryRecomputeMode::Auto,
        recompute_threshold_tokens: DEFAULT_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS,
        anchor_stride: None,
        cache_target_bytes: DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES,
    };

    /// Construct a policy after validating every numeric invariant.
    pub fn from_parts(
        recompute_mode: CheckpointBoundaryRecomputeMode,
        recompute_threshold_tokens: usize,
        anchor_stride: Option<usize>,
        cache_target_bytes: u64,
    ) -> Result<Self, InvalidCheckpointBoundaryPolicy> {
        if recompute_threshold_tokens == 0 {
            return Err(InvalidCheckpointBoundaryPolicy::RecomputeThresholdTokens);
        }
        if anchor_stride == Some(0) {
            return Err(InvalidCheckpointBoundaryPolicy::AnchorStride);
        }
        if cache_target_bytes == 0 {
            return Err(InvalidCheckpointBoundaryPolicy::CacheTargetBytes);
        }
        Ok(Self {
            recompute_mode,
            recompute_threshold_tokens,
            anchor_stride,
            cache_target_bytes,
        })
    }

    pub const fn recompute_mode(self) -> CheckpointBoundaryRecomputeMode {
        self.recompute_mode
    }

    pub const fn recompute_threshold_tokens(self) -> usize {
        self.recompute_threshold_tokens
    }

    /// Explicit boundary stride, or `None` when the cache target selects it.
    pub const fn anchor_stride(self) -> Option<usize> {
        self.anchor_stride
    }

    pub const fn cache_target_bytes(self) -> u64 {
        self.cache_target_bytes
    }

    /// Decide whether a sequence uses sparse replay instead of retaining every
    /// checkpoint boundary.
    pub const fn recompute_for(self, seq_len: usize) -> bool {
        match self.recompute_mode {
            CheckpointBoundaryRecomputeMode::Auto => seq_len >= self.recompute_threshold_tokens,
            CheckpointBoundaryRecomputeMode::Enabled => true,
            CheckpointBoundaryRecomputeMode::Disabled => false,
        }
    }

    /// Resolve the sparse anchor stride for one checkpointed SFT shape.
    ///
    /// This preserves the historical policy exactly: an explicit positive
    /// stride wins; a single segment uses stride one; otherwise the cache
    /// target reserves one slot for replay and spreads the remaining anchors
    /// evenly across the segment boundaries.
    pub fn anchor_stride_for_shape(
        self,
        seq_len: usize,
        num_segments: usize,
        hidden_size: usize,
        boundary_bytes_per_elem: usize,
    ) -> usize {
        if let Some(explicit) = self.anchor_stride {
            return explicit;
        }
        if num_segments <= 1 {
            return 1;
        }

        let boundary_bytes = usize_to_u64_saturating(seq_len)
            .saturating_mul(usize_to_u64_saturating(hidden_size))
            .saturating_mul(usize_to_u64_saturating(boundary_bytes_per_elem.max(1)))
            .max(1);
        let max_anchors = usize::try_from((self.cache_target_bytes / boundary_bytes).max(2))
            .unwrap_or(usize::MAX);
        let replay_anchor_slots = max_anchors.saturating_sub(1).max(1);
        num_segments.div_ceil(replay_anchor_slots).max(1)
    }
}

impl Default for CheckpointBoundaryPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidCheckpointBoundaryPolicy {
    RecomputeThresholdTokens,
    AnchorStride,
    CacheTargetBytes,
}

impl std::fmt::Display for InvalidCheckpointBoundaryPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::RecomputeThresholdTokens => {
                "checkpoint boundary recompute threshold tokens must be greater than zero"
            }
            Self::AnchorStride => {
                "checkpoint boundary anchor stride must be greater than zero when set"
            }
            Self::CacheTargetBytes => {
                "checkpoint boundary cache target bytes must be greater than zero"
            }
        })
    }
}

impl std::error::Error for InvalidCheckpointBoundaryPolicy {}

/// Immutable gradient-checkpoint behavior for one native training run.
///
/// `Disabled` retains an optional explicit segment count so resolving both
/// legacy controls does not silently discard either value. The count remains
/// part of checkpoint identity even though execution is disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum GradientCheckpointPolicy {
    Auto,
    ExplicitSegments {
        segments: std::num::NonZeroUsize,
    },
    Disabled {
        segments: Option<std::num::NonZeroUsize>,
    },
}

impl GradientCheckpointPolicy {
    /// Construct a validated policy from typed configuration fields.
    pub fn from_parts(
        segments: Option<usize>,
        disabled: bool,
    ) -> Result<Self, InvalidGradientCheckpointSegments> {
        let segments = segments
            .map(|value| {
                std::num::NonZeroUsize::new(value).ok_or(InvalidGradientCheckpointSegments)
            })
            .transpose()?;
        Ok(Self::from_nonzero_parts(segments, disabled))
    }

    pub const fn from_nonzero_parts(
        segments: Option<std::num::NonZeroUsize>,
        disabled: bool,
    ) -> Self {
        if disabled {
            Self::Disabled { segments }
        } else if let Some(segments) = segments {
            Self::ExplicitSegments { segments }
        } else {
            Self::Auto
        }
    }

    pub const fn explicit_segments(self) -> Option<std::num::NonZeroUsize> {
        match self {
            Self::Auto => None,
            Self::ExplicitSegments { segments } => Some(segments),
            Self::Disabled { segments } => segments,
        }
    }

    pub const fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }

    pub const fn is_disabled(self) -> bool {
        matches!(self, Self::Disabled { .. })
    }
}

impl Default for GradientCheckpointPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidGradientCheckpointSegments;

impl std::fmt::Display for InvalidGradientCheckpointSegments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gradient checkpoint segments must be greater than zero")
    }
}

impl std::error::Error for InvalidGradientCheckpointSegments {}

/// Immutable process-lifetime inputs used to plan a native training run.
///
/// Server and accelerator callers construct this with
/// [`TrainingRuntimeContext::new_for_device`] and pass it through the explicit
/// `*_with_runtime` entry points. Standalone checkpoint planners can use
/// [`TrainingRuntimeContext::standalone`] (or [`Default`]) for physical
/// memory autodetection without claiming a backend identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingRuntimeContext {
    effective_vram: kiln_memory::vram::GpuVramInfo,
    gradient_checkpoint_policy: GradientCheckpointPolicy,
    checkpoint_boundary_policy: CheckpointBoundaryPolicy,
    runtime_device: Option<kiln_tensor::Device>,
    streaming_prefill_policy: Option<kiln_model::forward::StreamingPrefillExecutionPolicy>,
    admitted_sft_loss_route: Option<kiln_model::backend::SftFlceLossRoute>,
}

impl TrainingRuntimeContext {
    /// Build an unbound checkpoint-planning context.
    ///
    /// Accelerator execution rejects this form; use [`Self::new_for_device`]
    /// when the context will authorize a training run.
    pub const fn new(
        effective_vram: kiln_memory::vram::GpuVramInfo,
        gradient_checkpoint_policy: GradientCheckpointPolicy,
    ) -> Self {
        Self {
            effective_vram,
            gradient_checkpoint_policy,
            checkpoint_boundary_policy: CheckpointBoundaryPolicy::DEFAULT,
            runtime_device: None,
            streaming_prefill_policy: None,
            admitted_sft_loss_route: None,
        }
    }

    /// Bind training to the backend device selected by the owning runtime.
    ///
    /// This binding is required when accelerator identity cannot be recovered
    /// from tensor storage, notably Vulkan's CPU-host weight representation.
    pub const fn new_for_device(
        runtime_device: kiln_tensor::Device,
        effective_vram: kiln_memory::vram::GpuVramInfo,
        gradient_checkpoint_policy: GradientCheckpointPolicy,
    ) -> Self {
        Self {
            effective_vram,
            gradient_checkpoint_policy,
            checkpoint_boundary_policy: CheckpointBoundaryPolicy::DEFAULT,
            runtime_device: Some(runtime_device),
            streaming_prefill_policy: None,
            admitted_sft_loss_route: None,
        }
    }

    /// Install the startup-resolved SFT checkpoint-boundary policy.
    pub const fn with_checkpoint_boundary_policy(
        mut self,
        policy: CheckpointBoundaryPolicy,
    ) -> Self {
        self.checkpoint_boundary_policy = policy;
        self
    }

    /// Install the startup-resolved streaming-prefill policy for this run.
    ///
    /// Standalone callers can omit this and receive the selected device's
    /// backend defaults. Server callers use this builder so inference,
    /// admission planning, training, and exact-resume identity all share the
    /// same immutable policy.
    pub const fn with_streaming_prefill_policy(
        mut self,
        policy: kiln_model::forward::StreamingPrefillExecutionPolicy,
    ) -> Self {
        self.streaming_prefill_policy = Some(policy);
        self
    }

    /// Bind one SFT run to the backend loss route used by admission.
    ///
    /// This is job-local typed state, not an operator-selectable policy. The
    /// trainer revalidates it against the execution backend before allocating
    /// trainable parameters, then carries this exact value through every step.
    pub const fn with_admitted_sft_loss_route(
        mut self,
        route: kiln_model::backend::SftFlceLossRoute,
    ) -> Self {
        self.admitted_sft_loss_route = Some(route);
        self
    }

    /// Build a standalone context using automatic device and checkpoint-policy
    /// selection. Explicit policy belongs in [`TrainingRuntimeContext::new`].
    pub fn standalone() -> Self {
        Self::standalone_for_selector(kiln_memory::vram::VramProbeSelector::Auto)
    }

    /// Build a standalone context explicitly bound to `device`.
    pub fn standalone_for_device(device: kiln_tensor::Device) -> Self {
        let mut runtime = Self::standalone_for_selector(device.memory_probe_selector());
        runtime.runtime_device = Some(device);
        runtime
    }

    fn standalone_for_selector(selector: kiln_memory::vram::VramProbeSelector) -> Self {
        Self::new(
            kiln_memory::vram::detect_vram_for(selector),
            GradientCheckpointPolicy::Auto,
        )
    }

    pub(crate) fn standalone_with_effective_vram(
        effective_vram: kiln_memory::vram::GpuVramInfo,
    ) -> Self {
        Self::new(effective_vram, GradientCheckpointPolicy::Auto)
    }

    /// Effective accelerator capacity and physical-memory topology for this run.
    pub const fn effective_vram(&self) -> &kiln_memory::vram::GpuVramInfo {
        &self.effective_vram
    }

    pub const fn gradient_checkpoint_policy(&self) -> GradientCheckpointPolicy {
        self.gradient_checkpoint_policy
    }

    pub const fn checkpoint_boundary_policy(&self) -> CheckpointBoundaryPolicy {
        self.checkpoint_boundary_policy
    }

    /// Backend device selected by the owning runtime, when one was bound.
    pub const fn runtime_device(&self) -> Option<kiln_tensor::Device> {
        self.runtime_device
    }

    /// Explicit startup policy, if the owning runtime installed one.
    pub const fn configured_streaming_prefill_policy(
        &self,
    ) -> Option<kiln_model::forward::StreamingPrefillExecutionPolicy> {
        self.streaming_prefill_policy
    }

    /// The loss route pinned by SFT admission, if this is an admitted run.
    pub const fn admitted_sft_loss_route(&self) -> Option<kiln_model::backend::SftFlceLossRoute> {
        self.admitted_sft_loss_route
    }

    /// Resolve the immutable streaming-prefill policy for `device`.
    ///
    /// This never reads process environment. Compatibility and standalone
    /// contexts fall back to the backend capability contract for the device.
    pub fn resolved_streaming_prefill_policy(
        &self,
        device: kiln_tensor::Device,
    ) -> kiln_model::forward::StreamingPrefillExecutionPolicy {
        self.streaming_prefill_policy.unwrap_or_else(|| {
            kiln_model::forward::StreamingPrefillExecutionPolicy::for_device(device)
        })
    }

    /// Resolve the training device without inferring backend identity from
    /// process features, environment variables, or hardware availability.
    ///
    /// Weight storage and execution must currently name the same device. In
    /// particular, Vulkan serving may use CPU-host weight handles, but the
    /// full-model resident training upload has not passed the production-model
    /// correctness and memory-safety gates. Treating that representation as a
    /// Vulkan training device would turn a known-incomplete multi-GiB upload
    /// into the default server path, so it fails closed here.
    pub fn resolve_device_for_weights(
        &self,
        weight_device: kiln_tensor::Device,
    ) -> anyhow::Result<kiln_tensor::Device> {
        match self.runtime_device {
            Some(runtime_device) if runtime_device == weight_device => Ok(runtime_device),
            Some(kiln_tensor::Device::Vulkan(_)) if weight_device == kiln_tensor::Device::Cpu => {
                anyhow::bail!(
                    "native Vulkan training is unavailable for CPU-host serving weights: the full-model resident Vulkan training substrate is not production-qualified"
                )
            }
            Some(runtime_device) => anyhow::bail!(
                "training runtime device {} does not match model weight device {}",
                runtime_device.short_name(),
                weight_device.short_name(),
            ),
            None if weight_device == kiln_tensor::Device::Cpu => Ok(kiln_tensor::Device::Cpu),
            None => anyhow::bail!(
                "training runtime has no explicit device binding for accelerator-backed weights on {}; construct TrainingRuntimeContext::new_for_device",
                weight_device.short_name(),
            ),
        }
    }

    /// Stable exact-resume identity for every input that can change planning.
    pub fn checkpoint_planning_identity(&self) -> serde_json::Value {
        let streaming_prefill_policy = self
            .streaming_prefill_policy
            .or_else(|| {
                self.runtime_device.map(|device| {
                    kiln_model::forward::StreamingPrefillExecutionPolicy::for_device(device)
                })
            })
            .map(streaming_prefill_policy_identity);
        let mut identity = serde_json::json!({
            "schema": "kiln.training-checkpoint-planning.v3",
            "effective_vram": {
                "total_bytes": self.effective_vram.total_bytes,
                "source": self.effective_vram.source.to_string(),
                "unified": self.effective_vram.unified,
            },
            "gradient_checkpoint_policy": self.gradient_checkpoint_policy,
            "checkpoint_boundary_policy": self.checkpoint_boundary_policy,
            "runtime_device": self.runtime_device.map(kiln_tensor::Device::short_name),
            "streaming_prefill_policy": streaming_prefill_policy,
        });
        if let Some(route) = self.admitted_sft_loss_route {
            let object = identity
                .as_object_mut()
                .expect("checkpoint planning identity is always an object");
            object.insert(
                "schema".to_string(),
                serde_json::json!("kiln.training-checkpoint-planning.v4"),
            );
            object.insert(
                "sft_loss_route".to_string(),
                serde_json::json!(route.as_str()),
            );
        }
        identity
    }

    /// Stable planning identity after resolving the actual training device.
    pub fn checkpoint_planning_identity_for_device(
        &self,
        device: kiln_tensor::Device,
    ) -> serde_json::Value {
        let mut identity = self.checkpoint_planning_identity();
        let object = identity
            .as_object_mut()
            .expect("checkpoint planning identity is always an object");
        object.insert(
            "runtime_device".to_string(),
            serde_json::json!(device.short_name()),
        );
        object.insert(
            "streaming_prefill_policy".to_string(),
            streaming_prefill_policy_identity(self.resolved_streaming_prefill_policy(device)),
        );
        identity
    }
}

fn streaming_prefill_policy_identity(
    policy: kiln_model::forward::StreamingPrefillExecutionPolicy,
) -> serde_json::Value {
    let mode = match policy.mode() {
        kiln_model::forward::StreamingPrefillMode::Auto => "auto",
        kiln_model::forward::StreamingPrefillMode::Enabled => "enabled",
        kiln_model::forward::StreamingPrefillMode::Disabled => "disabled",
    };
    serde_json::json!({
        "mode": mode,
        "threshold_tokens": policy.threshold_tokens(),
        "base_tile_tokens": policy.base_tile_tokens(),
        "tape_tile_tokens": policy.tape_tile_tokens(),
        "detached_full_attn_tile_tokens": policy.detached_full_attn_tile_tokens(),
        "detached_full_attn_boundary_tile_tokens": policy.detached_full_attn_boundary_tile_tokens(),
        "detached_full_attn_tape_replay_tile_tokens": policy.detached_full_attn_tape_replay_tile_tokens(),
        "last_token_lm_head": policy.last_token_lm_head(),
    })
}

impl Default for TrainingRuntimeContext {
    fn default() -> Self {
        Self::standalone()
    }
}

/// Resolve the compatibility-wrapper context from tensor storage only.
///
/// CPU-host weights select CPU in this compatibility path and never acquire a
/// Vulkan identity from hardware presence. Vulkan callers must use a
/// `*_with_runtime` entry point with [`TrainingRuntimeContext::new_for_device`].
pub fn standalone_training_runtime_for_weight_device(
    weight_device: kiln_tensor::Device,
) -> anyhow::Result<TrainingRuntimeContext> {
    Ok(TrainingRuntimeContext::standalone_for_device(weight_device))
}

/// Initialize the process governor before a standalone training run.
/// Server callers have already installed the same immutable policy; standalone
/// examples and library entry points use this explicit startup boundary rather
/// than discovering memory from an attention hot path.
pub fn ensure_memory_governor_for_runtime(
    device: kiln_tensor::Device,
    runtime: &TrainingRuntimeContext,
) -> anyhow::Result<()> {
    if let Some(runtime_device) = runtime.runtime_device() {
        if runtime_device != device {
            anyhow::bail!(
                "training device {} does not match runtime-bound device {}",
                device.short_name(),
                runtime_device.short_name(),
            );
        }
    } else if device.is_gpu() {
        anyhow::bail!(
            "training runtime has no explicit device binding for {}; construct TrainingRuntimeContext::new_for_device",
            device.short_name(),
        );
    }
    let selector = device.memory_probe_selector();
    if selector == kiln_memory::VramProbeSelector::None {
        return Ok(());
    }

    kiln_memory::validate_vram_probe_identity(selector).map_err(|error| {
        anyhow::anyhow!(
            "cannot initialize training runtime for {}: {error}",
            device.short_name()
        )
    })?;

    let runtime_capacity = runtime.effective_vram().total_bytes;
    if runtime_capacity == 0 {
        anyhow::bail!(
            "training runtime for {} has no safe accelerator capacity",
            device.short_name()
        );
    }

    if kiln_memory::MemoryGovernor::try_global_cached_snapshot().is_none() {
        kiln_memory::MemoryGovernor::configure_global(
            selector,
            kiln_memory::GovernorConfig {
                capacity_limit_bytes: Some(runtime_capacity),
                ..kiln_memory::GovernorConfig::default()
            },
        )?;
    } else {
        let configuration = kiln_memory::MemoryGovernor::global_configuration();
        if configuration.selector != selector {
            anyhow::bail!(
                "training device {} does not match the initialized memory governor selector {:?}",
                device.short_name(),
                configuration.selector,
            );
        }
        if configuration.governor.capacity_limit_bytes != Some(runtime_capacity) {
            anyhow::bail!(
                "training runtime capacity {} bytes does not match the initialized memory governor capacity {:?}",
                runtime_capacity,
                configuration.governor.capacity_limit_bytes,
            );
        }
    }

    let governor = kiln_memory::MemoryGovernor::global();
    let published = governor.refresh();
    if published.total_bytes != runtime_capacity || published.observations.probe_failed {
        anyhow::bail!(
            "training memory probe for {} did not publish the runtime-bound {}-byte capacity (published {} bytes, probe_failed={})",
            device.short_name(),
            runtime_capacity,
            published.total_bytes,
            published.observations.probe_failed,
        );
    }
    if !governor.start_sampler() {
        anyhow::bail!("failed to start the training memory sampler");
    }
    Ok(())
}

pub use hf_grpo_interop::{
    HF_TRL_GRPO_CORPUS_IDENTITY_V1, HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP,
    HF_TRL_GRPO_MAX_DATASET_BYTES, HF_TRL_GRPO_MAX_GROUPS, HF_TRL_GRPO_MAX_ROW_BYTES,
    HfTrlGrpoCorpusSummary, HfTrlGrpoExportIdentity, ordered_grpo_corpus_sha256,
    validate_hf_trl_grpo_groups_for_export, validate_hf_trl_grpo_jsonl_for_export,
};
pub use hf_interop::{
    HF_TRL_ADAPTER_CONFIG_FILENAME, HF_TRL_ADAPTER_MODEL_FILENAME, HF_TRL_CHAT_TEMPLATE_FILENAME,
    HF_TRL_DATASET_FILENAME, HF_TRL_ENVIRONMENT_LOCK_FILENAME, HF_TRL_EXECUTED_SCRIPT_FILENAME,
    HF_TRL_EXPORT_MANIFEST_FILENAME, HF_TRL_EXPORT_SCHEMA_VERSION, HF_TRL_EXPORT_TYPE,
    HF_TRL_IMPORT_RECEIPT_FILENAME, HF_TRL_IMPORT_SCHEMA_VERSION, HF_TRL_IMPORT_TYPE,
    HF_TRL_MODEL_CONFIG_FILENAME, HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME,
    HF_TRL_REFERENCE_SCRIPT_FILENAME, HF_TRL_RESULT_MANIFEST_FILENAME,
    HF_TRL_RESULT_SCHEMA_VERSION, HF_TRL_RESULT_TYPE, HF_TRL_SFT_INGESTION_FILENAME,
    HF_TRL_SPLIT_MANIFEST_FILENAME, HF_TRL_TOKENIZER_FILENAME, HF_TRL_TRAINING_TEMPLATE_FILENAME,
    HfTrlConfigValue, HfTrlDataExport, HfTrlDatasetFormat, HfTrlExportManifestV1,
    HfTrlFileIdentity, HfTrlImportReceiptV1, HfTrlInputAdapter, HfTrlModelIdentity,
    HfTrlOutputAdapter, HfTrlResidentModelIdentity, HfTrlSftLabelPolicy, HfTrlSftSelection,
    HfTrlTask, HfTrlTrainerIdentity, HfTrlTrainerKind, HfTrlTrainingResultV1,
    read_hf_trl_export_manifest, read_hf_trl_import_receipt, read_hf_trl_training_result,
};
pub use hf_interop_bundle::{
    HF_TRL_BUNDLE_SUFFIX, HF_TRL_GRPO_ENVIRONMENT_LOCK, HF_TRL_GRPO_REFERENCE_SCRIPT,
    HF_TRL_IMPORT_ENVELOPE_SUFFIX, HF_TRL_IMPORT_MAX_ADAPTER_CONFIG_BYTES,
    HF_TRL_IMPORT_MAX_ARCHIVE_BYTES, HF_TRL_IMPORT_MAX_ARCHIVE_ENTRIES,
    HF_TRL_IMPORT_MAX_AUXILIARY_BYTES, HF_TRL_IMPORT_MAX_EXPANDED_BYTES,
    HF_TRL_IMPORT_MAX_MANIFEST_BYTES, HF_TRL_IMPORT_MAX_SAFETENSORS_HEADER_BYTES,
    HF_TRL_IMPORT_MAX_SCRIPT_BYTES, HF_TRL_IMPORT_MAX_TAR_ZERO_PADDING_BYTES,
    HF_TRL_IMPORTED_ADAPTER_FILES, HF_TRL_SFT_ENVIRONMENT_LOCK, HF_TRL_SFT_REFERENCE_SCRIPT,
    HfTrlGrpoBundleInput, HfTrlGrpoDatasetSource, HfTrlInputAdapterSource, HfTrlSftBundleInput,
    hf_trl_import_envelope_files, validate_hf_trl_import_name, verify_hf_trl_completed_bundle,
    verify_hf_trl_export_bundle, verify_hf_trl_import_envelope, write_hf_trl_grpo_bundle,
    write_hf_trl_import_envelope, write_hf_trl_sft_bundle,
};
pub use logit_cache::{CacheEntry, CacheStats, CachedLogitSource, LogitCache, hash_prefix};
pub use receipt::{
    AdapterReceipt, DiagnosticSummary, PromptSourceDescriptor, RECEIPT_SCHEMA_VERSION,
    TeacherDescriptor,
};
pub use remote_teacher::{
    RemoteProvider, RemoteTeacher, RemoteTeacherConfig, discover_vllm_identity,
    normalize_vllm_completions_url,
};
pub use sft_ingestion::{
    SFT_INGESTION_SCHEMA_V1, SftIngestionReceipt, SftInvalidRowPolicy, SftPreparedDataset,
    SftRejectedRowReceipt, SftRowRejectionReason, prepare_sft_examples, prepare_sft_jsonl,
    verify_prepared_sft_examples,
};
pub use teacher_identity::{
    MAX_TEACHER_IDENTITY_FINGERPRINT_BYTES, MAX_TEACHER_IDENTITY_JSON_BYTES,
    MAX_TEACHER_IDENTITY_NAME_BYTES, MAX_TEACHER_IMPLEMENTATION_BYTES, MAX_TEACHER_MODEL_LEN,
    MAX_TEACHER_PROMPT_LOGPROB_CANDIDATES, MAX_TEACHER_TOP_K, MAX_TEACHER_VOCAB_SIZE,
    TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1, TEACHER_IDENTITY_LOGPROBS_MODE_V1,
    TEACHER_IDENTITY_PROTOCOL_V1, TEACHER_IDENTITY_SCHEMA_V1, TeacherAdapterIdentityV1,
    TeacherIdentityError, TeacherIdentityV1,
};

pub use adapter_output::{
    ADAPTER_MANIFEST_FILENAME, ADAPTER_MANIFEST_SCHEMA_VERSION, ADAPTER_RECEIPT_FILENAME,
    AdapterManifest, AdapterManifestFiles, AdapterOutputReceipt, AdapterRestoreOptions,
    AdapterRestoreReceipt, ResolvedSftOutputLayout, install_adapter_symlink, read_adapter_manifest,
    read_adapter_manifest_from_adapter_dir, resolve_sft_output_layout,
    restore_adapter_from_manifest, validate_adapter_output_dir, validate_install_adapter_name,
    write_adapter_manifest_from_train_receipt, write_adapter_output_receipt,
};
pub use adapter_shape::{
    ALLOW_ADAPTER_SHAPE_CONVERSION_FLAG, BaseAdapterCompatibility, TRAINABLE_TARGET_MODULES,
    resolve_base_adapter_dir, validate_base_adapter_compatibility,
};
pub use diagnostics::{
    DIVERSITY_COLLAPSE_THRESHOLD, DIVERSITY_COLLAPSE_WINDOW, GuardrailDecision, GuardrailTrigger,
    LengthInflationGuardrail, OpdDiagnosticSnapshot, REPETITION_GUARDRAIL_THRESHOLD,
    RolloutSummary, SELF_PLAY_SATURATION_THRESHOLD, SELF_PLAY_SATURATION_WINDOW, build_snapshot,
    repetition_rate, rollout_diversity, truncation_rate,
};
pub use logit_source::{
    DeterministicUniformLogitSource, LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch,
    TopKLogprobs,
};
pub use lora_scaling::{
    ALLOW_HIGH_LORA_SCALE_FLAG, MAX_LORA_ALPHA_OVER_RANK, alpha_over_rank, validate_lora_scaling,
};
pub use opd::{
    AgenticLossInputs, AgenticLossWeights, COLD_START_DEFAULT_EPOCHS, COLD_START_DEFAULT_PROMPTS,
    COLD_START_OVERLAP_THRESHOLD, ColdStartDecision, DistillMergeRequest, DistillMergeSource,
    DistillPumpMode, DistillPumpRequest, DistillRefreshRequest, DistillSelfRequest,
    LoadedOffPolicyDistillationDataset, NewKnowledgeSource,
    OFF_POLICY_DISTILLATION_MANIFEST_SCHEMA_V1, OffPolicyDistillationExample,
    OffPolicyDistillationManifestV1, OffPolicyDistillationSummary, OffPolicyLossBreakdown,
    OpdConfig, OpdLossGranularity, OpdObjective, OpdPrompt, OpdRequest, OpdTrainingMode,
    PreparedOffPolicyDistillation, SelfDistillMode, StableOpdCoefficients, StableOpdLossInputs,
    StableOpdLossOutputs, TeacherActionToken, TeacherTopLogprob, TipTokenClass, cold_start_probe,
    cold_start_probe_default, compose_off_policy_distillation_loss, compute_agentic_loss_weights,
    compute_initial_overlap, compute_stable_opd_loss, default_beta_kl, default_lambda_sft,
    default_lambda_verifier, default_opd_samples_per_prompt, default_opd_top_k,
    default_score_decay_steps, default_score_earliest_weight, default_tip_tool_call_weight,
    default_tip_tool_name_weight, load_off_policy_distillation_dataset,
    load_off_policy_distillation_jsonl, parse_off_policy_distillation_dataset_str,
    parse_off_policy_distillation_jsonl_str, prepare_off_policy_distillation_dataset,
    prepare_off_policy_distillation_dataset_with_identity, resolve_opd_top_k,
};
pub use train_receipt::{
    ADAPTER_CANARY_STATUS_FILENAME, AdapterCanaryCheckReceipt, AdapterCanaryState,
    AdapterCanaryStatusReceipt, AdapterSmokePromptDiagnosis, AdapterSmokePromptDiagnosisReceipt,
    AdapterSmokePromptReceipt, AdapterSmokeTestReceipt, GRPO_POLICY_AUDIT_SCHEMA_V1,
    GrpoImportanceSamplingMetricsReceipt, GrpoKlReferenceMetricsReceipt, GrpoPolicyAuditReceipt,
    GrpoRecordedBehaviorSourceReceipt, GrpoRecordedProvenanceReceipt, OpdReceipt,
    TRAIN_RECEIPT_FILENAME, TRAIN_RECEIPT_SCHEMA_VERSION, TrainFailureReason, TrainReceipt,
    TrainReceiptStatus, read_adapter_canary_status_from_adapter_dir,
};

pub use replay::{
    BaseModel, Lineage, OutcomeRecord, OutcomeStatus, ParentLora, ReplayKind, ReplayLog,
    ReplayRecord, RequestRecord,
};
pub use trainer::CheckpointConfig;

use serde::{Deserialize, Serialize};

/// The canonical chat message used by tokenization, inference, eval, and
/// training. Agentic fields are preserved through SFT, GRPO, and OPD inputs.
pub use kiln_core::tokenizer::ChatMessage;

/// An SFT training example — a conversation with the correct assistant response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SftExample {
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
}

/// Request to run SFT training on submitted examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SftRequest {
    #[serde(default)]
    pub examples: Vec<SftExample>,
    /// Optional server-local JSONL dataset path. Each non-empty line is one
    /// `SftExample`. This keeps large local SFT corpora out of the HTTP
    /// request body while preserving exact per-example training semantics.
    /// Mutually exclusive with `examples` and `dataset`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_path: Option<String>,
    /// Optional name of an uploaded dataset (the eval dataset store) to train
    /// on instead of inline `examples`. The server resolves the name and reads
    /// every row — callers never round-trip rows through the client, so large
    /// datasets train whole (no preview-endpoint truncation) and the request
    /// stays a few bytes. Mutually exclusive with `examples`.
    ///
    /// The magic name `corrections:active` resolves the durable corrections
    /// basket's trainable rows (hand-written ideal answers not yet trained);
    /// the consumed rows flip to `trained_into` when the job COMPLETES.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    #[serde(default)]
    pub config: SftConfig,
    /// Server-owned ingestion evidence. It is not part of the public request
    /// wire; admission creates it after parsing and tokenization, then the
    /// worker verifies it before training.
    #[serde(skip)]
    pub ingestion: Option<SftIngestionReceipt>,
    /// Optional auto-eval hook: when set, the training queue worker enqueues
    /// an eval against the produced adapter once training completes. Lets
    /// callers chain `train → eval` in a single API call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

/// Optimizer selection for training.
///
/// `Muon` (Bernstein-Newhouse 2024 / Jordan et al. 2024) is the
/// **default**: momentum-orthogonalized SGD. It keeps a single per-param
/// heavy-ball momentum buffer, takes a (Nesterov) look-ahead, and — for
/// the rank-2 LoRA A/B weight matrices — projects it onto the nearest
/// semi-orthogonal matrix via a Newton-Schulz iteration before stepping,
/// then rescales by `sqrt(max(rows, cols))` so the update magnitude is
/// shape-independent. Dispatched on-device via `dispatch_muon_step`
/// (fused per-matrix Newton-Schulz kernels — CUDA / ROCm / Vulkan /
/// Metal) when operands are resident; otherwise the `kiln_optim::Muon`
/// CPU reference. It uses one momentum buffer per parameter, while AdamW
/// uses first- and second-moment buffers.
///
/// `AdamW` is decoupled-weight-decay Adam (Loshchilov & Hutter 2019);
/// dispatched on-device via `dispatch_adamw_step` when the backend
/// supports residency. The trainer allocates per-parameter first/second
/// moment tensors at init, registers them in the resident-activation
/// registry alongside the param/grad, and updates all three in-place per
/// step. Select with `{"optimizer": {"kind": "adam_w"}}`.
///
/// `Sgd` is plain stochastic gradient descent (`param -= lr * grad`);
/// dispatched on-device via `dispatch_sgd_step` when the backend
/// supports residency. Select with `{"optimizer": {"kind": "sgd"}}`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum Optimizer {
    Sgd,
    AdamW {
        #[serde(default = "default_beta1")]
        beta1: f32,
        #[serde(default = "default_beta2")]
        beta2: f32,
        #[serde(default = "default_eps")]
        eps: f32,
        #[serde(default = "default_weight_decay")]
        weight_decay: f32,
    },
    /// Momentum-orthogonalized SGD (the default). `momentum` is the
    /// heavy-ball coefficient; `nesterov` toggles the look-ahead;
    /// `ns_iters` is the Newton-Schulz iteration count (paper uses 5);
    /// `weight_decay` is decoupled.
    Muon {
        #[serde(default = "default_muon_momentum")]
        momentum: f32,
        #[serde(default = "default_muon_nesterov")]
        nesterov: bool,
        #[serde(default = "default_muon_ns_iters")]
        ns_iters: u32,
        #[serde(default = "default_muon_weight_decay")]
        weight_decay: f32,
    },
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::Muon {
            momentum: default_muon_momentum(),
            nesterov: default_muon_nesterov(),
            ns_iters: default_muon_ns_iters(),
            weight_decay: default_muon_weight_decay(),
        }
    }
}

impl Optimizer {
    pub const fn kind(self) -> kiln_model::TrainingOptimizerKind {
        match self {
            Self::Sgd => kiln_model::TrainingOptimizerKind::Sgd,
            Self::AdamW { .. } => kiln_model::TrainingOptimizerKind::AdamW,
            Self::Muon { .. } => kiln_model::TrainingOptimizerKind::Muon,
        }
    }

    /// Fail closed on optimizer values that would make an update undefined or
    /// silently non-finite. Public admission calls this before queueing work;
    /// trainer entry points repeat it for direct Rust callers.
    pub fn validate_hyperparameters(&self) -> anyhow::Result<()> {
        match *self {
            Self::Sgd => {}
            Self::AdamW {
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                anyhow::ensure!(
                    beta1.is_finite() && (0.0..1.0).contains(&beta1),
                    "AdamW beta1 must be finite and in [0, 1), got {beta1}"
                );
                anyhow::ensure!(
                    beta2.is_finite() && (0.0..1.0).contains(&beta2),
                    "AdamW beta2 must be finite and in [0, 1), got {beta2}"
                );
                anyhow::ensure!(
                    eps.is_finite() && eps > 0.0,
                    "AdamW eps must be finite and greater than zero, got {eps}"
                );
                anyhow::ensure!(
                    weight_decay.is_finite() && weight_decay >= 0.0,
                    "AdamW weight_decay must be finite and non-negative, got {weight_decay}"
                );
            }
            Self::Muon {
                momentum,
                ns_iters,
                weight_decay,
                ..
            } => {
                anyhow::ensure!(
                    momentum.is_finite() && (0.0..1.0).contains(&momentum),
                    "Muon momentum must be finite and in [0, 1), got {momentum}"
                );
                anyhow::ensure!(
                    (1..=20).contains(&ns_iters),
                    "Muon ns_iters must be in 1..=20, got {ns_iters}"
                );
                anyhow::ensure!(
                    weight_decay.is_finite() && weight_decay >= 0.0,
                    "Muon weight_decay must be finite and non-negative, got {weight_decay}"
                );
            }
        }
        Ok(())
    }
}

fn default_beta1() -> f32 {
    0.9
}
fn default_beta2() -> f32 {
    0.999
}
fn default_eps() -> f32 {
    1e-8
}
fn default_weight_decay() -> f32 {
    0.0
}
fn default_muon_momentum() -> f32 {
    0.95
}
fn default_muon_nesterov() -> bool {
    true
}
fn default_muon_ns_iters() -> u32 {
    5
}
fn default_muon_weight_decay() -> f32 {
    0.0
}

/// Which training mode a learning rate is being resolved for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainMode {
    Sft,
    Grpo,
    Opd,
}

/// Per-optimizer learning-rate default, used when a config omits
/// `learning_rate`.
///
/// AdamW and SGD keep the legacy defaults (SFT 1e-4, GRPO/OPD 1e-5).
/// Muon's orthogonalized, RMS-matched update is larger than AdamW's update,
/// but the optimizer-library 2e-2 default is too hot for LoRA SFT on long
/// correction examples. SFT therefore uses the empirically stable 1e-3
/// band, while GRPO/OPD keep the earlier 2e-3 heuristic. Train receipts
/// record the resolved value, so every run stays auditable either way.
pub fn resolve_learning_rate(optimizer: &Optimizer, mode: TrainMode) -> f64 {
    match (optimizer, mode) {
        (Optimizer::Muon { .. }, TrainMode::Sft) => 1e-3,
        (Optimizer::Muon { .. }, TrainMode::Grpo | TrainMode::Opd) => 2e-3,
        (_, TrainMode::Sft) => 1e-4,
        (_, TrainMode::Grpo | TrainMode::Opd) => 1e-5,
    }
}

/// Warn when an explicit learning rate sits far outside the selected
/// optimizer's band. Returns `None` inside the 50x band; ordinary tuning
/// (a few x either way) never trips it.
pub fn learning_rate_band_warning(explicit: f64, resolved_default: f64) -> Option<String> {
    const BAND_RATIO: f64 = 50.0;
    if !explicit.is_finite() || explicit <= 0.0 || resolved_default <= 0.0 {
        return None;
    }
    let ratio = explicit / resolved_default;
    if ratio > BAND_RATIO {
        Some(format!(
            "learning_rate {explicit:e} is {ratio:.0}x larger than this optimizer's \
             default {resolved_default:e} — the run may diverge; omit learning_rate \
             to use the per-optimizer default"
        ))
    } else if ratio < 1.0 / BAND_RATIO {
        Some(format!(
            "learning_rate {explicit:e} is {:.0}x smaller than this optimizer's \
             default {resolved_default:e} — the run may train too cold to matter; \
             omit learning_rate to use the per-optimizer default",
            1.0 / ratio
        ))
    } else {
        None
    }
}

pub const NATIVE_SFT_PROFILE_V1: &str = "native_online_lora_v1";

/// The only training-shape contract implemented by native SFT.
///
/// One admitted conversation is one microbatch and one optimizer step. The
/// learning rate is constant from the first step, without accumulation,
/// warmup, decay, or gradient clipping. Use HF/TRL directly for configurable
/// or batched training rather than expecting this API to approximate it.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum SftTrainingProfile {
    #[default]
    #[serde(rename = "native_online_lora_v1")]
    NativeOnlineLoraV1,
}

impl std::fmt::Display for SftTrainingProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NativeOnlineLoraV1 => f.write_str(NATIVE_SFT_PROFILE_V1),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SftConfig {
    /// Versioned native training-shape contract. There is intentionally only
    /// one native profile; use HF/TRL for general trainer configuration.
    #[serde(default)]
    pub training_profile: SftTrainingProfile,
    /// How SFT ingestion handles malformed, structurally invalid, or
    /// untokenizable rows. `fail` is the default and rejects the entire
    /// submission. `skip` trains only accepted rows and records stable hashes
    /// for both accepted and rejected rows in the train receipt.
    #[serde(default)]
    pub invalid_row_policy: SftInvalidRowPolicy,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// Learning rate. `None` (the default) resolves per optimizer at run
    /// start — see [`resolve_learning_rate`]. Explicit values are used
    /// verbatim; the train receipt records whichever value actually ran.
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default = "default_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_alpha")]
    pub lora_alpha: f32,
    /// Select the separate post-SFT native-MTP LoRA alignment phase.
    /// Standalone `kiln-train` treats `None` as automatic when the checkpoint
    /// ships `mtp.*` tensors, `Some(false)` as disabled, and `Some(true)` as an
    /// explicit request. Server SFT instead normalizes `None` to `Some(false)`
    /// and rejects `Some(true)` until this phase participates in server GPU
    /// coordination, memory admission, cancellation, and settlement.
    #[serde(default)]
    pub train_mtp: Option<bool>,
    /// If set, continue training from this adapter instead of starting fresh.
    pub base_adapter: Option<String>,
    /// Reserved escape hatch for future explicit shape conversion. Today,
    /// incompatible base adapters still fail before optimizer setup.
    #[serde(default)]
    pub allow_adapter_shape_conversion: bool,
    /// Permit alpha/rank above the default safety limit for deliberate
    /// experiments.
    #[serde(default)]
    pub allow_high_lora_scale: bool,
    /// Name for the output adapter. Auto-generated if not set.
    pub output_name: Option<String>,
    /// Automatically load the resulting adapter when training completes (default true).
    #[serde(default = "default_auto_load")]
    pub auto_load: bool,
    /// Publish an exact resumable checkpoint every N committed optimizer
    /// steps. Cancellation also checkpoints at the next step boundary.
    /// `None` disables periodic checkpoints.
    #[serde(default)]
    pub checkpoint_interval: Option<usize>,
    /// Resume from an immutable `.kiln-checkpoint` directory. This is not a
    /// PEFT adapter path: the checkpoint must contain the exact optimizer,
    /// cursor, RNG, and loop state produced by this training mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_checkpoint: Option<String>,
    /// Internal submit-time resolution of dynamic gradient checkpointing.
    /// None means the trainer should auto-tune from the workload shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_checkpoint_segments: Option<usize>,
    /// Seed selected for LoRA initialization and RNG-dependent steps. If
    /// `None`, the trainer generates one and records the concrete value in
    /// `replay.jsonl` for audit. The seed alone is not a replay guarantee.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optimizer selection. Defaults to Muon (momentum-orthogonalized SGD
    /// with fused on-device Newton-Schulz). AdamW remains available via
    /// `{"optimizer": {"kind": "adam_w"}}` and plain SGD via
    /// `{"optimizer": {"kind": "sgd"}}` for backwards-compatible runs.
    #[serde(default)]
    pub optimizer: Optimizer,
    /// After successful training, run a small base-vs-adapter canary check and
    /// record the result in `train_receipt.json`.
    #[serde(default)]
    pub adapter_smoke_test: bool,
}

fn default_auto_load() -> bool {
    true
}
fn default_epochs() -> usize {
    3
}
fn default_rank() -> usize {
    16
}
fn default_alpha() -> f32 {
    32.0
}

impl SftConfig {
    /// The explicit `learning_rate` when given, else the per-optimizer
    /// default for SFT.
    pub fn effective_learning_rate(&self) -> f64 {
        self.learning_rate
            .unwrap_or_else(|| resolve_learning_rate(&self.optimizer, TrainMode::Sft))
    }

    /// Validate the complete bounded native-SFT contract before expensive
    /// tokenization, queue publication, or GPU ownership.
    pub fn validate_native_contract(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.epochs > 0, "SFT epochs must be greater than zero");
        let learning_rate = self.effective_learning_rate();
        anyhow::ensure!(
            learning_rate.is_finite() && learning_rate > 0.0,
            "SFT learning_rate must be finite and greater than zero, got {learning_rate}"
        );
        let optimizer_learning_rate = learning_rate as f32;
        anyhow::ensure!(
            optimizer_learning_rate.is_finite() && optimizer_learning_rate > 0.0,
            "SFT learning_rate must remain finite and greater than zero when represented as f32, got {learning_rate}"
        );
        validate_lora_scaling(self.lora_rank, self.lora_alpha, self.allow_high_lora_scale)?;
        self.optimizer.validate_hyperparameters()?;
        anyhow::ensure!(
            self.checkpoint_interval != Some(0),
            "SFT checkpoint_interval must be greater than zero"
        );
        anyhow::ensure!(
            self.grad_checkpoint_segments != Some(0),
            "SFT grad_checkpoint_segments must be greater than zero"
        );
        for (field, value) in [
            ("base_adapter", self.base_adapter.as_deref()),
            ("output_name", self.output_name.as_deref()),
            ("resume_checkpoint", self.resume_checkpoint.as_deref()),
        ] {
            anyhow::ensure!(
                value.is_none_or(|value| !value.trim().is_empty()),
                "SFT {field} must not be blank"
            );
        }
        Ok(())
    }
}

impl Default for SftConfig {
    fn default() -> Self {
        Self {
            training_profile: SftTrainingProfile::default(),
            invalid_row_policy: SftInvalidRowPolicy::default(),
            epochs: default_epochs(),
            learning_rate: None,
            lora_rank: default_rank(),
            lora_alpha: default_alpha(),
            train_mtp: None,
            base_adapter: None,
            allow_adapter_shape_conversion: false,
            allow_high_lora_scale: false,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: None,
            resume_checkpoint: None,
            grad_checkpoint_segments: None,
            seed: None,
            optimizer: Optimizer::default(),
            adapter_smoke_test: false,
        }
    }
}

// ScoredCompletion and GrpoGroup are now re-exports of the canonical
// trajectory-aware types defined in `crate::trajectory`. The old field
// shape (`{text, reward}` / `{messages, completions}`) is preserved
// byte-identical, plus an optional `trajectory` field on each rollout
// that ECHO consumes. Legacy callers see no field-name change.
//
// See `docs/plans/echo-integration-plan.md` §2 and §B.1 for the design.

pub use crate::trajectory::{
    AgenticGroup, ROLLOUT_PROVENANCE_SCHEMA_V1, RolloutActionTokenSourceV1, RolloutActionTokenV1,
    RolloutAdapterIdentityV1, RolloutBehaviorPolicyIdentityV1, RolloutChatTemplateInvocationV1,
    RolloutProvenanceV1, RolloutSamplingConfigV1, RolloutThinkingBudgetV1,
    RolloutTokenizerIdentityV1, ScoredRollout, TurnKind, TurnSegment,
    rollout_prompt_messages_sha256, scored_rollout_payload_sha256,
};

/// Legacy alias for [`ScoredRollout`]. Use the canonical name in new code.
pub type ScoredCompletion = ScoredRollout;

/// Legacy alias for [`AgenticGroup`]. Use the canonical name in new code.
pub type GrpoGroup = AgenticGroup;

/// Request to run a GRPO training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoRequest {
    /// Accepts `agentic_groups` as an alias so the /v1/train/agentic
    /// route's JSON body uses the semantically-meaningful name. Legacy
    /// clients posting `groups` continue to deserialize unchanged.
    #[serde(default, alias = "agentic_groups")]
    pub groups: Vec<GrpoGroup>,
    /// Optional server-local JSONL dataset path. Each non-empty line is one
    /// `GrpoGroup`. Used by Vulkan-native GRPO to stream large datasets without
    /// retaining every group in memory.
    #[serde(default)]
    pub dataset_path: Option<String>,
    /// Optional name of an uploaded dataset (the eval dataset store) to train
    /// on instead of inline `groups` / a raw path. The server resolves the
    /// name to its on-disk JSONL and streams it like `dataset_path`. Mutually
    /// exclusive with both `groups` and `dataset_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    #[serde(default)]
    pub config: GrpoConfig,
    /// Optional auto-eval hook (see `SftRequest::post_eval`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

/// Group-relative advantage normalization mode.
///
/// `Vanilla` is the original DeepSeekMath / R1 form: `A_i = (r_i - mean(r)) /
/// (std(r) + eps)`. `DrGrpo` follows arXiv:2503.20783 ("Understanding
/// R1-Zero-Like Training") and drops the `std` normalization to eliminate
/// question-difficulty bias and the numerical blow-up when within-group
/// rewards are nearly uniform.
///
/// Default: `DrGrpo`. Phase 1 ablation on Qwen3.5-4B (the kiln backbone)
/// showed clean training under both modes; dropping std normalization is a
/// strict improvement at the math level (removes the small-std blow-up and
/// the question-difficulty bias from arXiv:2503.20783) with zero compute
/// cost, so we ship Dr. GRPO as the default.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdvantageMode {
    /// `A_i = (r_i - mean(r)) / (std(r) + 1e-8)`. The original DeepSeekMath /
    /// R1 form; retained for ablation and back-compat.
    Vanilla,
    /// `A_i = r_i - mean(r)`. Recommended by Dr. GRPO (arXiv:2503.20783) and
    /// the SimpleRL-Zoo replication (arXiv:2503.18892). The kiln default
    /// from Phase 1 onward.
    #[default]
    DrGrpo,
}

/// How the GRPO surrogate loss is aggregated across the completions in a group.
///
/// `PerSample` is the original DeepSeekMath / R1 form: each completion's loss
/// is the per-token mean, then the group reports the mean across completions,
/// and the optimizer steps once per completion. It systematically
/// under-penalizes long incorrect completions and is the documented source of
/// the GRPO "length-drift" failure mode.
///
/// `TokenLevel` is the DAPO Token-Level Policy Gradient Loss (arXiv:2503.14476):
/// per-token surrogates are accumulated across all completions in the group
/// and divided by the total active completion token count, with a single
/// optimizer step per group. Removes per-sample length bias.
///
/// Default: `TokenLevel`. Phase 1 ablation on Qwen3.5-4B showed a ~2× speedup
/// over `PerSample` (one optimizer step per group, plus dynamic-sampling
/// filtering) with clean loss curves; combined with the literature consensus
/// against per-sample averaging, this becomes the kiln default. The
/// vk-native path falls back to `PerSample` (kernel work pending).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LossAggregation {
    /// Per-completion mean-of-tokens, per-completion optimizer step. The
    /// original DeepSeekMath / R1 form; retained for ablation and as the
    /// vk-native fallback.
    PerSample,
    /// DAPO Token-Level Loss: sum-over-all-tokens-in-group divided by
    /// total active tokens, single optimizer step per group. The kiln
    /// default from Phase 1 onward (candle path).
    #[default]
    TokenLevel,
}

/// How the importance-sampling ratio is computed and clipped.
///
/// `Token` is the historical kiln behavior: the IS ratio is per-token, the
/// PPO surrogate clips per-token, and clipped tokens drop out of the
/// gradient. The DeepSeekMath / R1 default.
///
/// `Sequence` implements GSPO (arXiv:2507.18071): the ratio is computed once
/// per sequence as `(π_θ(y) / π_old(y))^(1/|y|)`, clip and surrogate live at
/// the sequence level, and every token in the sequence sees the same scalar
/// gradient coefficient `s · d_surrogate / |y|`. Originally proposed for MoE
/// (Qwen3-MoE production); kiln's Qwen3.5-4B is dense so the MoE argument
/// does not apply, but the variance-reduction argument on long CoT may.
/// Treat as a controlled experiment until ablated on the actual workload.
///
/// `Cispo` implements CISPO (MiniMax-M1, arXiv:2506.13585): per-token gradient
/// flow with the IS *weight* capped above (not the surrogate). Every token
/// contributes a gradient; there is no lower weight floor. The absolute cap
/// comes from [`GrpoConfig::cispo_max_weight`], not the PPO epsilon fields.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IsLevel {
    /// Historical per-token IS with PPO min(surrogate, clipped_surrogate).
    #[default]
    Token,
    /// GSPO sequence-level IS (arXiv:2507.18071).
    Sequence,
    /// CISPO upper-capped-weight IS (arXiv:2506.13585).
    Cispo,
}

/// Which distribution supplies the denominator for policy importance ratios.
///
/// `Recorded` consumes the exact, post-filter log-probabilities stored in
/// rollout provenance. Training fails closed when any sampled action token is
/// missing that provenance. `NoImportanceCorrection` explicitly fixes the
/// ratio at 1; it never substitutes the KL reference for the behavior policy.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorPolicy {
    /// Explicit on-policy/REINFORCE-style mode. No behavior probabilities are
    /// read and the importance ratio is fixed at 1.
    #[default]
    NoImportanceCorrection,
    /// Use sampled-token log-probabilities from exact rollout provenance.
    Recorded,
}

/// What frozen policy anchors the KL penalty, and how often it refreshes.
///
/// This selection is independent of [`BehaviorPolicy`]. In particular, the
/// base model or an EMA snapshot must never be reused as a stand-in for the
/// behavior distribution that sampled an off-policy rollout.
///
/// `BasePerStep` uses the base model (no LoRA), recomputed for every
/// completion. `None` skips the frozen-policy forward and is valid only when
/// the KL penalty is disabled.
///
/// `Ema` snapshots the LoRA-applied policy every `refresh_every` optimizer
/// steps into a frozen reference. `decay` controls EMA blending across
/// successive snapshots: `decay = 0.0` resets the snapshot at every
/// refresh; `decay = 0.9` keeps 90% of the prior snapshot and blends in
/// 10% of the current policy. This implements the "stronger reference"
/// idea from the post-DAPO line of work — the reference policy tracks
/// actual training progress instead of pulling toward the (potentially
/// pre-reasoning) base model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KlReferencePolicy {
    /// Base model (no LoRA), recomputed per completion. Historical default.
    BasePerStep,
    /// No KL-reference forward. Requires the KL penalty to be disabled.
    None,
    /// EMA snapshot of the LoRA-applied policy.
    Ema {
        #[serde(default = "default_ema_decay")]
        decay: f32,
        #[serde(default = "default_ema_refresh")]
        refresh_every: usize,
    },
}

impl Default for KlReferencePolicy {
    fn default() -> Self {
        KlReferencePolicy::BasePerStep
    }
}

/// Backwards-compatible Rust type name. New code should use
/// [`KlReferencePolicy`] so it cannot be confused with [`BehaviorPolicy`].
#[deprecated(note = "use KlReferencePolicy")]
pub type ReferencePolicy = KlReferencePolicy;

fn default_ema_decay() -> f32 {
    0.0
}
fn default_ema_refresh() -> usize {
    32
}

/// Which KL estimator to use for the per-token penalty.
///
/// This changes only the per-token KL term. Importance correction is selected
/// independently by [`BehaviorPolicy`].
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KlEstimator {
    /// Schulman k1: `KL_t = log_ratio_t`. Gradient-correct (matches the
    /// existing kiln implementation and is the recommended default).
    #[default]
    K1,
    /// Schulman k3: `KL_t = exp(-log_ratio_t) - 1 + log_ratio_t`. Always
    /// non-negative; value-correct but the gradient is biased. DeepSeekMath
    /// uses this form. Implemented on the shared kt tape path and the
    /// Vulkan kernel (`VK_GRPO_KL_MODE_K3`).
    K3,
    /// No KL penalty. The KL-reference forward is skipped. Equivalent in
    /// effect to `kl_coeff = 0` but expresses the intent explicitly.
    None,
}

/// Behavior when reward-variance filtering would leave too few groups.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RewardFilterOnEmpty {
    /// Fail the run with a receipt and reward-filter sidecar.
    #[default]
    Fail,
    /// Ignore the reward filter for this run and train on every group.
    TrainAll,
    /// Skip optimizer work and emit an untrained/base adapter plus receipt.
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    /// Learning rate. `None` (the default) resolves per optimizer at run
    /// start — see [`resolve_learning_rate`]. Explicit values are used
    /// verbatim; the train receipt records whichever value actually ran.
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default = "default_kl_coeff")]
    pub kl_coeff: f64,
    /// Symmetric clip epsilon. When `clip_eps_high` is `None`, both the lower
    /// and upper PPO clip bounds use this value (`[1-ε, 1+ε]`). When
    /// `clip_eps_high` is `Some(h)`, this field provides the lower epsilon
    /// (`1-ε_low`) and `h` provides the upper (`1+ε_high`). DAPO's
    /// "Clip-Higher" recommendation is `clip_epsilon = 0.20`,
    /// `clip_eps_high = Some(0.28)` (arXiv:2503.14476). CISPO does not read
    /// either PPO bound; use `cispo_max_weight` for that objective.
    #[serde(default = "default_clip_eps")]
    pub clip_epsilon: f64,
    /// Upper PPO clip epsilon for the asymmetric Clip-Higher recipe. `None`
    /// (default) preserves symmetric clipping using `clip_epsilon` on both
    /// sides. Ignored when `is_level = "cispo"`.
    #[serde(default)]
    pub clip_eps_high: Option<f64>,
    /// Absolute upper cap for the detached CISPO importance weight.
    ///
    /// This is not a PPO epsilon: `5.0` means `min(ratio, 5.0)`, matching the
    /// MiniMax-M1 definition and TRL's `loss_type="cispo"` interpretation of
    /// `epsilon_high`. CISPO intentionally has no lower weight floor, so
    /// low-probability actions retain their natural (small) gradient weight.
    #[serde(default = "default_cispo_max_weight")]
    pub cispo_max_weight: f64,
    /// Advantage normalization mode. Defaults to `DrGrpo` (drops
    /// std-normalization per arXiv:2503.20783). Set to `Vanilla` for the
    /// historical DeepSeekMath/R1 form.
    #[serde(default)]
    pub advantage_mode: AdvantageMode,
    /// Surrogate-loss aggregation mode. Defaults to `TokenLevel` (the DAPO
    /// Token-Level Loss, arXiv:2503.14476). Set to `PerSample` for the
    /// historical per-completion form.
    #[serde(default)]
    pub loss_aggregation: LossAggregation,
    /// KL penalty estimator. Defaults to `K1` (historical kiln behavior).
    #[serde(default)]
    pub kl_estimator: KlEstimator,
    /// When true (the kiln default since Phase 1), groups whose completions
    /// all share the same reward (and therefore produce a uniformly-zero
    /// advantage vector) are skipped before training. Implements DAPO's
    /// Dynamic Sampling filter (arXiv:2503.14476). Degenerate groups have
    /// no policy-gradient signal under any [`AdvantageMode`]; the only
    /// thing they contribute is a spurious KL pull, so dropping them is
    /// strictly compute-saving and a stability win. Phase 1 ablation on
    /// Qwen3.5-4B observed ~60-70% of humaneval-style groups filtered as
    /// degenerate at typical reward distributions.
    #[serde(default = "default_dynamic_sampling")]
    pub dynamic_sampling: bool,
    /// Importance-sampling level. Defaults to `Token` (historical kiln
    /// behavior). Set to `Sequence` for GSPO or `Cispo` for CISPO.
    #[serde(default)]
    pub is_level: IsLevel,
    /// Importance-correction source. `Recorded` requires exact per-token
    /// rollout provenance; `NoImportanceCorrection` fixes the ratio at 1.
    #[serde(default)]
    pub behavior_policy: BehaviorPolicy,
    /// Frozen policy used only for the KL penalty. The legacy JSON key
    /// `reference_policy` remains accepted as an input alias.
    #[serde(default, alias = "reference_policy")]
    pub kl_reference_policy: KlReferencePolicy,
    /// Phase 3c — selective-KL entropy regulation. When `Some(q)`, only
    /// tokens whose proxy entropy (defined as `-policy_log_prob` for the
    /// selected token) is at or above the `q`-quantile across the active
    /// tokens contribute to the KL penalty. This is a cheap approximation
    /// of the Cui et al. "high-entropy minority tokens drive effective RL"
    /// finding (arXiv:2506.01939): the full Clip-Cov / KL-Cov estimators
    /// require per-token vocab covariance, which doubles the analytic tail
    /// cost; selecting by negative chosen log-prob instead requires no
    /// extra forward work since policy log-probs are already materialized
    /// for the IS ratio. Typical values: `0.8` (apply KL on top-20% tokens
    /// by uncertainty), `None` = full-token KL (historical behavior).
    /// Only meaningful when `kl_estimator != None`.
    #[serde(default)]
    pub entropy_aware_kl_quantile: Option<f32>,
    /// Mean reward threshold above which low-variance groups are considered
    /// saturated. Used only for diagnostics and train receipts.
    #[serde(default = "default_reward_saturation_threshold")]
    pub reward_saturation_threshold: f64,
    /// Population variance threshold used with `reward_saturation_threshold`
    /// to warn about low-signal reward distributions.
    #[serde(default = "default_reward_low_variance_threshold")]
    pub reward_low_variance_threshold: f64,
    /// Drop groups whose population reward variance is below this threshold.
    /// Disabled when both reward-filter variance bounds are unset.
    #[serde(default)]
    pub reward_filter_var_min: Option<f64>,
    /// Drop groups whose population reward variance is above this threshold.
    /// Usually left unset; useful for excluding pathological reward outliers.
    #[serde(default)]
    pub reward_filter_var_max: Option<f64>,
    /// Minimum number of groups that must remain after reward filtering.
    #[serde(default = "default_reward_filter_min_groups")]
    pub reward_filter_min_groups: usize,
    /// Explicit behavior when reward filtering keeps fewer than
    /// `reward_filter_min_groups`.
    #[serde(default)]
    pub reward_filter_on_empty: RewardFilterOnEmpty,
    #[serde(default = "default_rank")]
    pub lora_rank: usize,
    #[serde(default = "default_alpha")]
    pub lora_alpha: f32,
    pub base_adapter: Option<String>,
    /// Reserved escape hatch for future explicit shape conversion. Today,
    /// incompatible base adapters still fail before optimizer setup.
    #[serde(default)]
    pub allow_adapter_shape_conversion: bool,
    /// Permit alpha/rank above the default safety limit for deliberate
    /// experiments.
    #[serde(default)]
    pub allow_high_lora_scale: bool,
    pub output_name: Option<String>,
    /// Automatically load the resulting adapter when training completes (default true).
    #[serde(default = "default_auto_load")]
    pub auto_load: bool,
    /// Publish an exact resumable checkpoint every N committed optimizer
    /// groups. Cancellation also checkpoints at the next group boundary.
    /// `None` disables periodic checkpoints.
    #[serde(default)]
    pub checkpoint_interval: Option<usize>,
    /// Resume from an immutable `.kiln-checkpoint` directory produced by the
    /// same GRPO route. PEFT adapter snapshots are not resumable checkpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_checkpoint: Option<String>,
    /// Internal submit-time resolution of dynamic gradient checkpointing.
    /// None means the trainer should auto-tune from the workload shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_checkpoint_segments: Option<usize>,
    /// Seed selected for LoRA initialization and RNG-dependent steps. If
    /// `None`, the trainer generates one and records the concrete value in
    /// `replay.jsonl` for audit. The seed alone is not a replay guarantee.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optimizer selection — see `SftConfig::optimizer`.
    #[serde(default)]
    pub optimizer: Optimizer,
    /// After successful training, run a small base-vs-adapter canary check and
    /// record the result in `train_receipt.json`.
    #[serde(default)]
    pub adapter_smoke_test: bool,
    /// Composition of per-token training objectives. ECHO is OFF by
    /// default until its env-CE term regains a kt-tape gradient root
    /// (#1082); OPD's slot is reserved but empty. See `LossConfig`.
    #[serde(default)]
    pub loss: LossConfig,
}

fn default_kl_coeff() -> f64 {
    0.1
}
fn default_clip_eps() -> f64 {
    0.2
}
pub(crate) fn default_cispo_max_weight() -> f64 {
    5.0
}
fn default_dynamic_sampling() -> bool {
    true
}
fn default_reward_saturation_threshold() -> f64 {
    crate::train_receipt::DEFAULT_REWARD_SATURATION_THRESHOLD
}
fn default_reward_low_variance_threshold() -> f64 {
    crate::train_receipt::DEFAULT_REWARD_LOW_VARIANCE_THRESHOLD
}
fn default_reward_filter_min_groups() -> usize {
    1
}

// ---- ECHO + LossConfig ----------------------------------------------------
//
// `LossConfig` composes three loss objectives that share one forward pass:
//
//   L_total = L_policy(actions)              [the GRPO surrogate]
//           + λ_echo · L_envCE(observations) [paper: Shrivastava 2026]
//           + λ_opd  · L_revKL(actions)      [paper: Lu 2025; wired in OPD merge]
//
// `LossConfig::default()` has ECHO ON (λ=0.05, paper §3.3): the env-CE
// term regained its gradient root on the fused GRPO tape node (ECHO
// resurrection PR2 — `grpo_tape_shim::EchoEnvSpec`), so default-config
// agentic GRPO on real pi trajectories trains the observation tokens
// again. Rollouts with no env tokens pay nothing (the term contributes
// exactly zero and the receipt's env_ce stays None).
// `opd` is a placeholder None; its shape is reserved so the composition
// stays orthogonal.

/// What positions in an observation segment contribute to the env_mask.
///
/// Paper §3.2: terminal warning prefixes memorize within ~60 steps and stop
/// providing useful gradient. `EnvOnly` (default) strips the harness
/// warning prefix from each Observation; `FullObs` (debug only) covers
/// the full observation including warnings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvMaskMode {
    #[default]
    EnvOnly,
    FullObs,
}

/// Configuration for ECHO — the auxiliary environment cross-entropy loss
/// on tool/observation tokens. See
/// `docs/papers/echo/echo_paper.md` and `docs/plans/echo-integration-plan.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoConfig {
    /// Mixing coefficient for the env-CE term: `L_total = L_policy + λ · L_envCE`.
    /// Paper §3.3 default: 0.05. Productive range: 0.01–0.05. ≥0.1 risks
    /// degrading the policy objective; 0.2 collapses to predictable-output
    /// rollouts (mode collapse).
    #[serde(default = "default_echo_lambda")]
    pub lambda: f64,
    /// Env_only (default) strips the harness warning prefix. FullObs (debug)
    /// covers the full observation including warnings.
    #[serde(default)]
    pub env_mask_mode: EnvMaskMode,
    /// Whether the trajectory_mask's warning_filter is on. Defaults to true.
    /// Set to false for debugging or for environments whose tool output
    /// never includes a `WARNINGS:` prefix.
    #[serde(default = "default_warning_filter")]
    pub warning_filter: bool,
}

fn default_echo_lambda() -> f64 {
    0.05
}
fn default_warning_filter() -> bool {
    true
}
/// ECHO default: ON at λ=0.05 (paper §3.3). The term was forced OFF
/// between the #1082 candle drop (which deleted the only env-CE gradient
/// producer) and resurrection PR2 (which rebuilt it as constant-coefficient
/// `(softmax − onehot)` rows on the fused GRPO tape root). Legacy
/// single-turn rollouts carry no env tokens, so the default costs them
/// exactly nothing.
fn default_echo_some() -> Option<EchoConfig> {
    Some(EchoConfig::default())
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            lambda: default_echo_lambda(),
            env_mask_mode: EnvMaskMode::default(),
            warning_filter: default_warning_filter(),
        }
    }
}

/// Placeholder for OPD's auxiliary term on the LossConfig — populated when
/// the OPD branch rebases on top of this work. The fields here mirror
/// `kiln_train::opd::OpdConfig`'s most relevant knobs; the exact shape will
/// be reconciled during the OPD merge. Default: None.
///
/// Reserving this slot now means `L = L_policy + λ_echo · L_envCE +
/// λ_opd · L_revKL` is structurally encoded in the type system, so OPD
/// composition is mechanical when its branch lands.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpdAuxConfig {
    #[serde(default)]
    pub lambda: f64,
}

/// Composition of per-token training objectives. Each branch contributes to
/// `L_total = L_policy + λ_echo · L_envCE + λ_opd · L_revKL` where inactive
/// branches contribute zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// Action-token policy-gradient term — knobs already on GrpoConfig
    /// (clip_eps, kl_coeff, kl_estimator, advantage_mode, is_level,
    /// behavior_policy, kl_reference_policy, etc.). LossConfig doesn't duplicate them; the
    /// trainer reads them from the surrounding GrpoConfig.
    ///
    /// Observation-token cross-entropy (paper: Shrivastava et al. 2026).
    /// Default: `Some(EchoConfig::default())` — ON at λ=0.05 (resurrection
    /// PR2 rebuilt the env-CE gradient on the fused GRPO tape root after
    /// the #1082 candle drop severed it). Rollouts without environment
    /// tokens pay nothing; set to `null` (or pass `--no-echo`) to train
    /// the action-token policy loss alone.
    #[serde(default = "default_echo_some")]
    pub echo: Option<EchoConfig>,
    /// Action-token reverse-KL to a teacher (OPD; Lu 2025). Default: None.
    /// Wired in by the OPD branch rebase; LossConfig holds the field so the
    /// composition is structurally orthogonal.
    #[serde(default)]
    pub opd: Option<OpdAuxConfig>,
    /// Verifier-free env-only adaptation mode (paper §5.5). When `true`,
    /// the trainer masks out the GRPO policy-gradient term entirely and
    /// trains *only* on ECHO's env-CE objective. Useful for adapting an
    /// already-trained agentic policy on tasks where no programmatic
    /// verifier is available — the model improves purely by learning to
    /// predict the consequences of its own actions. Default: false.
    #[serde(default)]
    pub no_policy_loss: bool,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            echo: default_echo_some(),
            opd: None,
            no_policy_loss: false,
        }
    }
}

impl LossConfig {
    /// Validate this composition against the kt-tape training path BEFORE
    /// any GPU work. The ECHO env-CE term is live again (resurrection PR2:
    /// constant-coefficient rows on the fused GRPO tape root), so
    /// echo-enabled configs with environment tokens TRAIN now. Still
    /// rejected: `no_policy_loss` WITHOUT an enabled ECHO term (nothing
    /// to train on) and the reserved `opd` slot. `no_policy_loss` + ECHO
    /// is the §5.5 verifier-free mode: env-CE rows drive the update while
    /// the policy-gradient coefficients are zeroed.
    /// `has_env_tokens` = the training data carries trajectory
    /// Observation/tool segments that the env mask would cover.
    pub fn validate_for_kt_tape(&self, _has_env_tokens: bool) -> Result<(), String> {
        if self.no_policy_loss && !self.echo_enabled() {
            return Err(
                "loss.no_policy_loss masks the policy-gradient term, so the ECHO \
                 env-CE term must be enabled to provide a gradient source — with \
                 both off there is nothing to train on. Enable loss.echo (the \
                 default) or remove no_policy_loss."
                    .to_string(),
            );
        }
        if self.opd.is_some() {
            return Err(
                "loss.opd composition on GRPO is reserved but not wired — use \
                 POST /v1/train/opd for on-policy distillation instead."
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Lambda for the ECHO term, or 0.0 if ECHO is disabled.
    pub fn echo_lambda(&self) -> f64 {
        self.echo.as_ref().map(|c| c.lambda).unwrap_or(0.0)
    }
    /// Lambda for the OPD auxiliary term, or 0.0 if OPD is disabled.
    pub fn opd_lambda(&self) -> f64 {
        self.opd.as_ref().map(|c| c.lambda).unwrap_or(0.0)
    }
    /// True when the ECHO term is configured to be applied.
    pub fn echo_enabled(&self) -> bool {
        self.echo.is_some() && self.echo_lambda() != 0.0
    }

    /// Apply environment-variable overrides on top of an existing
    /// `LossConfig`. Honors:
    ///
    /// - `KILN_ECHO_ENABLED` — `0`/`false`/`no` → disable ECHO
    ///   (`loss.echo = None`); `1`/`true`/`yes` → enable with current
    ///   `lambda` (or default 0.05 if previously disabled).
    /// - `KILN_ECHO_LAMBDA` — overrides `lambda` on the existing
    ///   `EchoConfig`; if ECHO is disabled and this is set, ECHO is
    ///   re-enabled with the given lambda.
    /// - `KILN_ECHO_ENV_MASK_MODE` — `env_only` (default) | `full_obs`.
    /// - `KILN_ECHO_WARNING_FILTER` — bool, default true.
    ///
    /// Call this from CLI entry points (cuda_grpo_ablation, vk_train CLI,
    /// future kiln-server route handlers) so users can toggle ECHO from
    /// the shell without editing JSON.
    pub fn apply_kiln_echo_env_overrides(&mut self) {
        // KILN_ECHO_ENABLED — explicit disable wins over anything else.
        if let Ok(val) = std::env::var("KILN_ECHO_ENABLED") {
            let v = val.to_lowercase();
            if v == "0" || v == "false" || v == "no" {
                self.echo = None;
            } else if v == "1" || v == "true" || v == "yes" {
                self.echo.get_or_insert_with(EchoConfig::default);
            }
        }

        // The remaining knobs only make sense when ECHO is on. Apply
        // them on the current EchoConfig (creating one if needed when
        // any knob is set).
        let lambda_override = std::env::var("KILN_ECHO_LAMBDA")
            .ok()
            .and_then(|v| v.parse::<f64>().ok());
        let mask_mode_override =
            std::env::var("KILN_ECHO_ENV_MASK_MODE").ok().and_then(|v| {
                match v.to_lowercase().as_str() {
                    "env_only" | "envonly" => Some(EnvMaskMode::EnvOnly),
                    "full_obs" | "fullobs" => Some(EnvMaskMode::FullObs),
                    _ => None,
                }
            });
        let warning_filter_override =
            std::env::var("KILN_ECHO_WARNING_FILTER")
                .ok()
                .and_then(|v| match v.to_lowercase().as_str() {
                    "0" | "false" | "no" => Some(false),
                    "1" | "true" | "yes" => Some(true),
                    _ => None,
                });

        if lambda_override.is_some()
            || mask_mode_override.is_some()
            || warning_filter_override.is_some()
        {
            let cfg = self.echo.get_or_insert_with(EchoConfig::default);
            if let Some(lambda) = lambda_override {
                cfg.lambda = lambda;
            }
            if let Some(mode) = mask_mode_override {
                cfg.env_mask_mode = mode;
            }
            if let Some(wf) = warning_filter_override {
                cfg.warning_filter = wf;
            }
        }
    }
}

impl GrpoConfig {
    /// Resolved PPO clip epsilon bounds `(ε_low, ε_high)`. The PPO clip range
    /// is `[1 - ε_low, 1 + ε_high]`. When `clip_eps_high` is `None`, both
    /// sides use `clip_epsilon` (symmetric, historical behavior).
    pub fn clip_bounds(&self) -> (f64, f64) {
        (
            self.clip_epsilon,
            self.clip_eps_high.unwrap_or(self.clip_epsilon),
        )
    }

    /// The explicit `learning_rate` when given, else the per-optimizer
    /// default for GRPO.
    pub fn effective_learning_rate(&self) -> f64 {
        self.learning_rate
            .unwrap_or_else(|| resolve_learning_rate(&self.optimizer, TrainMode::Grpo))
    }

    /// Whether this configuration needs frozen-policy log-probabilities for
    /// its KL term. A zero coefficient and an explicit `KlEstimator::None`
    /// both disable that work.
    pub fn kl_penalty_enabled(&self) -> bool {
        self.kl_coeff != 0.0 && !matches!(self.kl_estimator, KlEstimator::None)
    }

    /// Validate the independent behavior-policy and KL-reference selections.
    /// This is device-free so API submission, dry-run, and trainer entry can
    /// all reject an incoherent job before allocating accelerator memory.
    pub fn validate_policy_config(&self) -> Result<(), String> {
        if !self.kl_coeff.is_finite() || self.kl_coeff < 0.0 {
            return Err(format!(
                "kl_coeff must be finite and non-negative, got {}",
                self.kl_coeff
            ));
        }
        if !self.clip_epsilon.is_finite() || !(0.0..1.0).contains(&self.clip_epsilon) {
            return Err(format!(
                "clip_epsilon must be finite and within [0, 1), got {}",
                self.clip_epsilon
            ));
        }
        if let Some(clip_high) = self.clip_eps_high {
            if !clip_high.is_finite() || clip_high < 0.0 {
                return Err(format!(
                    "clip_eps_high must be finite and non-negative, got {clip_high}"
                ));
            }
        }
        if !self.cispo_max_weight.is_finite() || self.cispo_max_weight <= 0.0 {
            return Err(format!(
                "cispo_max_weight must be finite and greater than zero, got {}",
                self.cispo_max_weight
            ));
        }
        if self.kl_penalty_enabled() && matches!(self.kl_reference_policy, KlReferencePolicy::None)
        {
            return Err(
                "kl_reference_policy=none requires kl_estimator=none or kl_coeff=0; the behavior policy is never substituted as a KL reference"
                    .to_string(),
            );
        }
        if let KlReferencePolicy::Ema {
            decay,
            refresh_every,
        } = &self.kl_reference_policy
        {
            if !decay.is_finite() || !(0.0..=1.0).contains(decay) {
                return Err(format!(
                    "kl_reference_policy EMA decay must be finite and within [0, 1], got {decay}"
                ));
            }
            if *refresh_every == 0 {
                return Err(
                    "kl_reference_policy EMA refresh_every must be greater than zero".to_string(),
                );
            }
        }
        if let Some(q) = self.entropy_aware_kl_quantile {
            if !q.is_finite() || !(0.0..=1.0).contains(&q) {
                return Err(format!(
                    "entropy_aware_kl_quantile must be finite and within [0, 1], got {q}"
                ));
            }
            if !self.kl_penalty_enabled() {
                return Err("entropy_aware_kl_quantile requires an enabled KL penalty".to_string());
            }
        }
        Ok(())
    }
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            learning_rate: None,
            kl_coeff: default_kl_coeff(),
            clip_epsilon: default_clip_eps(),
            clip_eps_high: None,
            cispo_max_weight: default_cispo_max_weight(),
            advantage_mode: AdvantageMode::default(),
            loss_aggregation: LossAggregation::default(),
            kl_estimator: KlEstimator::default(),
            dynamic_sampling: default_dynamic_sampling(),
            is_level: IsLevel::default(),
            behavior_policy: BehaviorPolicy::default(),
            kl_reference_policy: KlReferencePolicy::default(),
            entropy_aware_kl_quantile: None,
            reward_saturation_threshold: default_reward_saturation_threshold(),
            reward_low_variance_threshold: default_reward_low_variance_threshold(),
            reward_filter_var_min: None,
            reward_filter_var_max: None,
            reward_filter_min_groups: default_reward_filter_min_groups(),
            reward_filter_on_empty: RewardFilterOnEmpty::default(),
            lora_rank: default_rank(),
            lora_alpha: default_alpha(),
            base_adapter: None,
            allow_adapter_shape_conversion: false,
            allow_high_lora_scale: false,
            output_name: None,
            auto_load: default_auto_load(),
            checkpoint_interval: None,
            resume_checkpoint: None,
            grad_checkpoint_segments: None,
            seed: None,
            optimizer: Optimizer::default(),
            adapter_smoke_test: false,
            loss: LossConfig::default(),
        }
    }
}

/// Status of an ongoing training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub job_id: String,
    pub state: TrainingState,
    pub progress: f32,
    pub current_loss: Option<f64>,
    pub adapter_name: Option<String>,
    /// Exact seed materialized before queue publication. Decimal string keeps
    /// the full `u64` value intact in browser clients; absent on legacy jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_seed: Option<String>,
    pub started_at: String,
    pub elapsed_secs: f64,
    /// Wall-clock submit time as Unix milliseconds — survives server restarts
    /// so the /ui can render real timestamps for archived terminal jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submitted_unix_ms: Option<u64>,
    /// Wall-clock terminal-transition time as Unix milliseconds. `None` while
    /// the job is still active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_unix_ms: Option<u64>,
    /// "sft" or "grpo" — populated by the server side; absent on older
    /// payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_type: Option<String>,
    /// Failure detail when `state == failed`; absent otherwise and on
    /// payloads that predate the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// §8.7 promotion-gate verdict, stamped by the eval worker when the
    /// request carried `post_eval.min_accuracy`: whether the adapter was
    /// promoted, demoted to `<name>.failed`, or left unpromoted because
    /// the eval errored. Absent for ungated jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval_verdict: Option<String>,
    /// Machine-readable classification of `post_eval_verdict`:
    /// `promoted | kept | regression | demoted | error`. Additive twin of
    /// the prose verdict so consumers (dashboard pill, scripts) never
    /// classify prose by substring. Absent for ungated jobs and for
    /// verdicts archived before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_outcome: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingState {
    Queued,
    Running,
    Completed,
    Failed,
}

/// Response after submitting a training request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResponse {
    pub job_id: String,
    pub state: TrainingState,
    /// Exact decimal seed that the queued run will use.
    pub effective_seed: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_checkpoint_policy_preserves_disabled_segments() {
        let policy = GradientCheckpointPolicy::from_parts(Some(8), true).unwrap();
        assert!(policy.is_disabled());
        assert_eq!(policy.explicit_segments().map(|value| value.get()), Some(8));
        assert_eq!(
            serde_json::to_value(policy).unwrap(),
            serde_json::json!({"mode": "disabled", "segments": 8})
        );
        assert!(GradientCheckpointPolicy::from_parts(Some(0), false).is_err());
    }

    #[test]
    fn checkpoint_boundary_policy_validates_and_serializes_stably() {
        let policy = CheckpointBoundaryPolicy::default();
        assert_eq!(
            policy.recompute_mode(),
            CheckpointBoundaryRecomputeMode::Auto
        );
        assert_eq!(
            policy.recompute_threshold_tokens(),
            DEFAULT_RECOMPUTE_BOUNDARY_THRESHOLD_TOKENS
        );
        assert_eq!(policy.anchor_stride(), None);
        assert_eq!(
            policy.cache_target_bytes(),
            DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES
        );
        assert_eq!(
            serde_json::to_value(policy).unwrap(),
            serde_json::json!({
                "recompute_mode": "auto",
                "recompute_threshold_tokens": 8192,
                "anchor_stride": null,
                "cache_target_bytes": 6_442_450_944u64,
            })
        );

        assert_eq!(
            CheckpointBoundaryPolicy::from_parts(CheckpointBoundaryRecomputeMode::Auto, 0, None, 1,),
            Err(InvalidCheckpointBoundaryPolicy::RecomputeThresholdTokens)
        );
        assert_eq!(
            CheckpointBoundaryPolicy::from_parts(
                CheckpointBoundaryRecomputeMode::Auto,
                1,
                Some(0),
                1,
            ),
            Err(InvalidCheckpointBoundaryPolicy::AnchorStride)
        );
        assert_eq!(
            CheckpointBoundaryPolicy::from_parts(CheckpointBoundaryRecomputeMode::Auto, 1, None, 0,),
            Err(InvalidCheckpointBoundaryPolicy::CacheTargetBytes)
        );
    }

    #[test]
    fn checkpoint_boundary_policy_preserves_existing_dispatch_and_stride() {
        let automatic = CheckpointBoundaryPolicy::default();
        assert!(!automatic.recompute_for(8191));
        assert!(automatic.recompute_for(8192));

        let enabled = CheckpointBoundaryPolicy::from_parts(
            CheckpointBoundaryRecomputeMode::Enabled,
            8192,
            None,
            DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES,
        )
        .unwrap();
        let disabled = CheckpointBoundaryPolicy::from_parts(
            CheckpointBoundaryRecomputeMode::Disabled,
            8192,
            None,
            DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES,
        )
        .unwrap();
        assert!(enabled.recompute_for(1));
        assert!(!disabled.recompute_for(usize::MAX));

        let cache_limited = CheckpointBoundaryPolicy::from_parts(
            CheckpointBoundaryRecomputeMode::Auto,
            8192,
            None,
            4096,
        )
        .unwrap();
        assert_eq!(cache_limited.anchor_stride_for_shape(1, 8, 512, 2), 3);
        assert_eq!(cache_limited.anchor_stride_for_shape(1, 1, 512, 2), 1);

        let explicit = CheckpointBoundaryPolicy::from_parts(
            CheckpointBoundaryRecomputeMode::Auto,
            8192,
            Some(5),
            1,
        )
        .unwrap();
        assert_eq!(explicit.anchor_stride_for_shape(1, 1, 1, 1), 5);

        let saturated = CheckpointBoundaryPolicy::from_parts(
            CheckpointBoundaryRecomputeMode::Auto,
            1,
            None,
            u64::MAX,
        )
        .unwrap();
        assert_eq!(
            saturated.anchor_stride_for_shape(usize::MAX, usize::MAX, usize::MAX, usize::MAX,),
            usize::MAX,
            "maximal shape arithmetic must saturate instead of wrapping"
        );
    }

    #[test]
    fn training_runtime_device_selectors_are_backend_specific() {
        use kiln_memory::vram::{LinuxDrmVendor, VramProbeSelector};

        assert_eq!(
            kiln_tensor::Device::Cuda(2).memory_probe_selector(),
            VramProbeSelector::Nvidia(2)
        );
        assert_eq!(
            kiln_tensor::Device::Rocm(1).memory_probe_selector(),
            VramProbeSelector::LinuxDrm {
                index: 1,
                vendor: Some(LinuxDrmVendor::Amd),
            }
        );
        assert_eq!(
            kiln_tensor::Device::Vulkan(3).memory_probe_selector(),
            VramProbeSelector::LinuxDrm {
                index: 3,
                vendor: None,
            }
        );
        assert_eq!(
            kiln_tensor::Device::Metal(4).memory_probe_selector(),
            VramProbeSelector::AppleUnified
        );
        assert_eq!(
            kiln_tensor::Device::Cpu.memory_probe_selector(),
            VramProbeSelector::None
        );
    }

    #[test]
    fn explicit_vulkan_runtime_rejects_unqualified_cpu_host_weight_residency() {
        let vram = kiln_memory::vram::GpuVramInfo {
            total_bytes: 16 * 1024 * 1024 * 1024,
            source: kiln_memory::vram::VramSource::ConfigOverride,
            unified: true,
        };
        let vulkan = TrainingRuntimeContext::new_for_device(
            kiln_tensor::Device::Vulkan(0),
            vram,
            GradientCheckpointPolicy::Auto,
        );
        let error = vulkan
            .resolve_device_for_weights(kiln_tensor::Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(error.contains("native Vulkan training is unavailable"));
        assert!(error.contains("not production-qualified"));

        let unbound = TrainingRuntimeContext::new(vram, GradientCheckpointPolicy::Auto);
        assert!(
            unbound
                .resolve_device_for_weights(kiln_tensor::Device::Rocm(0))
                .unwrap_err()
                .to_string()
                .contains("no explicit device binding")
        );
        assert!(
            vulkan
                .resolve_device_for_weights(kiln_tensor::Device::Cuda(0))
                .unwrap_err()
                .to_string()
                .contains("does not match")
        );
    }

    #[test]
    fn training_runtime_streaming_prefill_policy_is_explicit_and_resume_stable() {
        let vram = kiln_memory::vram::GpuVramInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            source: kiln_memory::vram::VramSource::ConfigOverride,
            unified: false,
        };
        let backend =
            kiln_model::StreamingPrefillBackendPolicy::for_device(kiln_tensor::Device::Rocm(0));
        let policy = kiln_model::forward::StreamingPrefillExecutionPolicy::resolve(
            backend,
            kiln_model::forward::StreamingPrefillMode::Enabled,
            Some(4096),
            Some(2048),
            Some(1024),
            Some(8192),
            false,
        );
        let runtime = TrainingRuntimeContext::new_for_device(
            kiln_tensor::Device::Rocm(0),
            vram,
            GradientCheckpointPolicy::Auto,
        )
        .with_streaming_prefill_policy(policy);

        assert_eq!(runtime.configured_streaming_prefill_policy(), Some(policy));
        assert_eq!(
            runtime.resolved_streaming_prefill_policy(kiln_tensor::Device::Rocm(0)),
            policy
        );
        let identity =
            runtime.checkpoint_planning_identity_for_device(kiln_tensor::Device::Rocm(0));
        assert_eq!(identity["schema"], "kiln.training-checkpoint-planning.v3");
        assert_eq!(
            identity["checkpoint_boundary_policy"],
            serde_json::to_value(CheckpointBoundaryPolicy::default()).unwrap()
        );
        assert_eq!(identity["streaming_prefill_policy"]["mode"], "enabled");
        assert_eq!(
            identity["streaming_prefill_policy"]["base_tile_tokens"],
            2048
        );
        assert_eq!(
            identity["streaming_prefill_policy"]["last_token_lm_head"],
            false
        );
    }

    #[test]
    fn admitted_sft_loss_route_is_job_local_exact_resume_identity() {
        let checkpoint_boundary_policy = CheckpointBoundaryPolicy::from_parts(
            CheckpointBoundaryRecomputeMode::Enabled,
            4096,
            Some(2),
            1024 * 1024 * 1024,
        )
        .unwrap();
        let streaming_prefill_policy =
            kiln_model::forward::StreamingPrefillExecutionPolicy::resolve(
                kiln_model::StreamingPrefillBackendPolicy::for_device(kiln_tensor::Device::Cpu),
                kiln_model::forward::StreamingPrefillMode::Enabled,
                Some(64),
                Some(128),
                Some(256),
                Some(512),
                false,
            );
        let runtime = TrainingRuntimeContext::new_for_device(
            kiln_tensor::Device::Cpu,
            kiln_memory::vram::GpuVramInfo {
                total_bytes: 0,
                source: kiln_memory::vram::VramSource::None,
                unified: false,
            },
            GradientCheckpointPolicy::Auto,
        )
        .with_checkpoint_boundary_policy(checkpoint_boundary_policy)
        .with_streaming_prefill_policy(streaming_prefill_policy);
        let unbound = runtime.checkpoint_planning_identity_for_device(kiln_tensor::Device::Cpu);
        assert_eq!(unbound["schema"], "kiln.training-checkpoint-planning.v3");
        assert!(unbound.get("sft_loss_route").is_none());

        let full_logits = runtime
            .with_admitted_sft_loss_route(kiln_model::backend::SftFlceLossRoute::FullLogits)
            .checkpoint_planning_identity_for_device(kiln_tensor::Device::Cpu);
        assert_eq!(
            full_logits["schema"],
            "kiln.training-checkpoint-planning.v4"
        );
        assert_eq!(full_logits["sft_loss_route"], "full_logits");
        assert_eq!(
            full_logits["checkpoint_boundary_policy"],
            serde_json::to_value(checkpoint_boundary_policy).unwrap()
        );
        assert_eq!(
            full_logits["streaming_prefill_policy"]["base_tile_tokens"],
            128
        );

        let kt_flce = runtime
            .with_admitted_sft_loss_route(kiln_model::backend::SftFlceLossRoute::KtTapeFlce)
            .checkpoint_planning_identity_for_device(kiln_tensor::Device::Cpu);
        assert_eq!(kt_flce["sft_loss_route"], "kt_tape_flce");
        assert_ne!(full_logits, kt_flce);
    }

    #[test]
    fn checkpoint_boundary_policy_dimensions_are_exact_resume_identity() {
        let vram = kiln_memory::vram::GpuVramInfo {
            total_bytes: 24 * 1024 * 1024 * 1024,
            source: kiln_memory::vram::VramSource::ConfigOverride,
            unified: false,
        };
        let base_runtime = TrainingRuntimeContext::new_for_device(
            kiln_tensor::Device::Rocm(0),
            vram,
            GradientCheckpointPolicy::Auto,
        );
        let base_identity =
            base_runtime.checkpoint_planning_identity_for_device(kiln_tensor::Device::Rocm(0));
        let variants = [
            CheckpointBoundaryPolicy::from_parts(
                CheckpointBoundaryRecomputeMode::Enabled,
                8192,
                None,
                DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES,
            )
            .unwrap(),
            CheckpointBoundaryPolicy::from_parts(
                CheckpointBoundaryRecomputeMode::Auto,
                4096,
                None,
                DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES,
            )
            .unwrap(),
            CheckpointBoundaryPolicy::from_parts(
                CheckpointBoundaryRecomputeMode::Auto,
                8192,
                Some(4),
                DEFAULT_CHECKPOINT_BOUNDARY_CACHE_TARGET_BYTES,
            )
            .unwrap(),
            CheckpointBoundaryPolicy::from_parts(
                CheckpointBoundaryRecomputeMode::Auto,
                8192,
                None,
                3 * 1024 * 1024 * 1024,
            )
            .unwrap(),
        ];

        for policy in variants {
            let runtime = base_runtime.with_checkpoint_boundary_policy(policy);
            assert_eq!(runtime.checkpoint_boundary_policy(), policy);
            assert_ne!(
                runtime.checkpoint_planning_identity_for_device(kiln_tensor::Device::Rocm(0)),
                base_identity,
                "every checkpoint-boundary policy dimension must be exact-resume identity"
            );
        }
    }

    /// Legacy `TrainingStatus` payloads (pre-`error` archives, older
    /// servers) must keep deserializing, and the field must stay off the
    /// wire when `None` so old dashboards see an unchanged shape.
    #[test]
    fn training_status_error_field_is_optional_and_skipped_when_none() {
        let legacy = serde_json::json!({
            "job_id": "job-1",
            "state": "failed",
            "progress": 0.5,
            "current_loss": null,
            "adapter_name": "a",
            "started_at": "3s ago",
            "elapsed_secs": 3.0
        });
        let status: TrainingStatus = serde_json::from_value(legacy).unwrap();
        assert!(status.error.is_none());
        assert!(status.effective_seed.is_none());

        let none_wire = serde_json::to_value(&status).unwrap();
        assert!(none_wire.get("error").is_none(), "None must be omitted");

        let failed = TrainingStatus {
            error: Some("trainer exploded".to_string()),
            ..status
        };
        let wire = serde_json::to_value(&failed).unwrap();
        assert_eq!(wire["error"], "trainer exploded");
        let back: TrainingStatus = serde_json::from_value(wire).unwrap();
        assert_eq!(back.error.as_deref(), Some("trainer exploded"));
    }

    /// Pin the serde shape of the training-policy defaults. A future default
    /// flip must update this snapshot in the same commit, which forces the
    /// field/enum docs (and CHANGELOG) to be revisited together — the Muon
    /// flip left "Defaults to AdamW" doc-rot behind precisely because nothing
    /// tied the default to a literal.
    #[test]
    fn training_policy_defaults_match_pinned_snapshots() {
        assert_eq!(
            serde_json::to_value(Optimizer::default()).unwrap(),
            serde_json::json!({
                "kind": "muon",
                // f32 fields widen to f64 in serde_json, so pin via casts.
                "momentum": 0.95f32 as f64,
                "nesterov": true,
                "ns_iters": 5,
                "weight_decay": 0.0f32 as f64
            })
        );
        assert_eq!(
            serde_json::to_value(AdvantageMode::default()).unwrap(),
            serde_json::json!("dr_grpo")
        );
        assert_eq!(
            serde_json::to_value(LossAggregation::default()).unwrap(),
            serde_json::json!("token_level")
        );
        assert_eq!(
            serde_json::to_value(KlEstimator::default()).unwrap(),
            serde_json::json!("k1")
        );
        assert_eq!(
            serde_json::to_value(BehaviorPolicy::default()).unwrap(),
            serde_json::json!("no_importance_correction")
        );
        assert_eq!(
            serde_json::to_value(KlReferencePolicy::default()).unwrap(),
            serde_json::json!({"kind": "base_per_step"})
        );
    }

    #[test]
    fn grpo_policy_config_uses_unambiguous_wire_fields_with_legacy_input_alias() {
        let config: GrpoConfig = serde_json::from_value(serde_json::json!({
            "reference_policy": {"kind": "none"},
            "kl_estimator": "none"
        }))
        .unwrap();
        assert_eq!(
            config.behavior_policy,
            BehaviorPolicy::NoImportanceCorrection
        );
        assert_eq!(config.kl_reference_policy, KlReferencePolicy::None);
        assert_eq!(config.cispo_max_weight, 5.0);
        config.validate_policy_config().unwrap();

        let wire = serde_json::to_value(config).unwrap();
        assert!(wire.get("reference_policy").is_none());
        assert_eq!(wire["behavior_policy"], "no_importance_correction");
        assert_eq!(wire["cispo_max_weight"], 5.0);
        assert_eq!(
            wire["kl_reference_policy"],
            serde_json::json!({"kind": "none"})
        );
    }

    #[test]
    fn grpo_policy_config_rejects_invalid_clipping_controls() {
        let invalid_clip_low = [-0.1, 1.0, f64::NAN, f64::INFINITY];
        for clip_epsilon in invalid_clip_low {
            let config = GrpoConfig {
                clip_epsilon,
                ..Default::default()
            };
            assert!(
                config
                    .validate_policy_config()
                    .unwrap_err()
                    .contains("clip_epsilon"),
                "accepted clip_epsilon={clip_epsilon}"
            );
        }

        for clip_eps_high in [-0.1, f64::NAN, f64::INFINITY] {
            let config = GrpoConfig {
                clip_eps_high: Some(clip_eps_high),
                ..Default::default()
            };
            assert!(
                config
                    .validate_policy_config()
                    .unwrap_err()
                    .contains("clip_eps_high"),
                "accepted clip_eps_high={clip_eps_high}"
            );
        }

        for cispo_max_weight in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let config = GrpoConfig {
                is_level: IsLevel::Cispo,
                cispo_max_weight,
                ..Default::default()
            };
            assert!(
                config
                    .validate_policy_config()
                    .unwrap_err()
                    .contains("cispo_max_weight"),
                "accepted cispo_max_weight={cispo_max_weight}"
            );
        }
    }

    #[test]
    fn grpo_policy_config_rejects_incoherent_kl_selection() {
        let missing_reference = GrpoConfig {
            kl_reference_policy: KlReferencePolicy::None,
            ..Default::default()
        };
        assert!(
            missing_reference
                .validate_policy_config()
                .unwrap_err()
                .contains("requires kl_estimator=none or kl_coeff=0")
        );

        let bad_ema = GrpoConfig {
            kl_reference_policy: KlReferencePolicy::Ema {
                decay: 0.5,
                refresh_every: 0,
            },
            ..Default::default()
        };
        assert!(
            bad_ema
                .validate_policy_config()
                .unwrap_err()
                .contains("refresh_every")
        );

        let disabled_kl = GrpoConfig {
            kl_estimator: KlEstimator::None,
            kl_reference_policy: KlReferencePolicy::None,
            ..Default::default()
        };
        disabled_kl.validate_policy_config().unwrap();
        assert!(!disabled_kl.kl_penalty_enabled());
    }

    /// Pin the full per-optimizer learning-rate table. AdamW/SGD are the
    /// unchanged legacy defaults; Muon's SFT value is lower than the raw
    /// optimizer-library default because LoRA SFT diverges at 2e-2 on
    /// correction-style long examples.
    #[test]
    fn learning_rate_resolution_table_snapshot() {
        let adamw: Optimizer = serde_json::from_str(r#"{"kind": "adam_w"}"#).unwrap();
        let muon = Optimizer::default();
        let table = [
            (&muon, TrainMode::Sft, 1e-3),
            (&muon, TrainMode::Grpo, 2e-3),
            (&muon, TrainMode::Opd, 2e-3),
            (&adamw, TrainMode::Sft, 1e-4),
            (&adamw, TrainMode::Grpo, 1e-5),
            (&adamw, TrainMode::Opd, 1e-5),
            (&Optimizer::Sgd, TrainMode::Sft, 1e-4),
            (&Optimizer::Sgd, TrainMode::Grpo, 1e-5),
            (&Optimizer::Sgd, TrainMode::Opd, 1e-5),
        ];
        for (optimizer, mode, expected) in table {
            let resolved = resolve_learning_rate(optimizer, mode);
            assert_eq!(
                resolved, expected,
                "resolve_learning_rate({optimizer:?}, {mode:?})"
            );
        }
    }

    #[test]
    fn learning_rate_serde_back_compat() {
        // Explicit wire values still deserialize (full back-compat) …
        let sft: SftConfig = serde_json::from_str(r#"{"learning_rate": 5e-5}"#).unwrap();
        assert_eq!(sft.learning_rate, Some(5e-5));
        let grpo: GrpoConfig = serde_json::from_str(r#"{"learning_rate": 5e-5}"#).unwrap();
        assert_eq!(grpo.learning_rate, Some(5e-5));
        // … and omitting the field means "resolve per optimizer".
        let sft: SftConfig = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(sft.learning_rate, None);
        let grpo: GrpoConfig = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(grpo.learning_rate, None);
    }

    #[test]
    fn effective_learning_rate_prefers_explicit_value() {
        let mut sft = SftConfig::default();
        assert_eq!(sft.effective_learning_rate(), 1e-3); // Muon default
        sft.learning_rate = Some(3e-4);
        assert_eq!(sft.effective_learning_rate(), 3e-4);

        let mut grpo = GrpoConfig::default();
        assert_eq!(grpo.effective_learning_rate(), 2e-3); // Muon default
        grpo.learning_rate = Some(7e-6);
        assert_eq!(grpo.effective_learning_rate(), 7e-6);

        let adamw_sft = SftConfig {
            optimizer: serde_json::from_str(r#"{"kind": "adam_w"}"#).unwrap(),
            ..SftConfig::default()
        };
        assert_eq!(adamw_sft.effective_learning_rate(), 1e-4);
    }

    #[test]
    fn learning_rate_band_warning_fires_only_past_50x() {
        assert!(learning_rate_band_warning(1e-5, 1e-3).is_some());
        // 100x hot fires too.
        assert!(learning_rate_band_warning(1e-1, 1e-3).is_some());
        // Ordinary tuning (10x either way) stays quiet.
        assert!(learning_rate_band_warning(1e-4, 1e-3).is_none());
        assert!(learning_rate_band_warning(1e-2, 1e-3).is_none());
        // Exactly on band, and degenerate inputs, stay quiet.
        assert!(learning_rate_band_warning(1e-3, 1e-3).is_none());
        assert!(learning_rate_band_warning(0.0, 1e-3).is_none());
        assert!(learning_rate_band_warning(f64::NAN, 1e-3).is_none());
    }

    #[test]
    fn test_sft_config_default_checkpoint_interval_is_none() {
        let config = SftConfig::default();
        assert!(config.checkpoint_interval.is_none());
        assert!(config.resume_checkpoint.is_none());
        assert_eq!(
            config.training_profile,
            SftTrainingProfile::NativeOnlineLoraV1
        );
        assert_eq!(config.invalid_row_policy, SftInvalidRowPolicy::Fail);
        assert_eq!(
            serde_json::to_value(config).unwrap()["training_profile"],
            NATIVE_SFT_PROFILE_V1
        );
    }

    #[test]
    fn native_sft_profile_rejects_unknown_general_trainer_knobs() {
        for (field, value) in [
            ("per_device_train_batch_size", serde_json::json!(8)),
            ("gradient_accumulation_steps", serde_json::json!(4)),
            ("lr_scheduler_type", serde_json::json!("cosine")),
            ("warmup_steps", serde_json::json!(10)),
            ("max_grad_norm", serde_json::json!(1.0)),
        ] {
            let mut config = serde_json::Map::new();
            config.insert(field.to_string(), value);
            let error = serde_json::from_value::<SftConfig>(config.into()).unwrap_err();
            assert!(
                error.to_string().contains("unknown field"),
                "{field}: {error}"
            );
        }
        assert!(
            serde_json::from_value::<SftConfig>(serde_json::json!({
                "training_profile": "general_sft"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<SftConfig>(serde_json::json!({
                "optimizer": {"kind": "adam_w", "amsgrad": true}
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<SftRequest>(serde_json::json!({
                "examples": [],
                "gradient_accumulation_steps": 4
            }))
            .is_err()
        );
    }

    #[test]
    fn native_sft_profile_validates_every_numeric_contract() {
        SftConfig::default().validate_native_contract().unwrap();

        let invalid = [
            SftConfig {
                epochs: 0,
                ..Default::default()
            },
            SftConfig {
                learning_rate: Some(0.0),
                ..Default::default()
            },
            SftConfig {
                learning_rate: Some(f64::MAX),
                ..Default::default()
            },
            SftConfig {
                learning_rate: Some(f64::MIN_POSITIVE),
                ..Default::default()
            },
            SftConfig {
                lora_alpha: f32::NAN,
                ..Default::default()
            },
            SftConfig {
                checkpoint_interval: Some(0),
                ..Default::default()
            },
            SftConfig {
                grad_checkpoint_segments: Some(0),
                ..Default::default()
            },
            SftConfig {
                optimizer: Optimizer::AdamW {
                    beta1: 1.0,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
                ..Default::default()
            },
            SftConfig {
                optimizer: Optimizer::Muon {
                    momentum: 0.95,
                    nesterov: true,
                    ns_iters: 0,
                    weight_decay: 0.0,
                },
                ..Default::default()
            },
            SftConfig {
                output_name: Some("  ".to_string()),
                ..Default::default()
            },
        ];
        for config in invalid {
            assert!(config.validate_native_contract().is_err(), "{config:?}");
        }
    }

    #[test]
    fn sft_invalid_row_policy_and_server_owned_ingestion_are_strict() {
        let config: SftConfig = serde_json::from_str(r#"{"invalid_row_policy":"skip"}"#).unwrap();
        assert_eq!(config.invalid_row_policy, SftInvalidRowPolicy::Skip);
        assert!(serde_json::from_str::<SftConfig>(r#"{"invalid_row_policy":"drop"}"#).is_err());

        assert!(
            serde_json::from_value::<SftRequest>(serde_json::json!({
                "examples": [{"messages": []}],
                "ingestion": {"forged": true}
            }))
            .is_err()
        );
    }

    #[test]
    fn test_grpo_config_default_checkpoint_interval_is_none() {
        let config = GrpoConfig::default();
        assert!(config.checkpoint_interval.is_none());
        assert!(config.resume_checkpoint.is_none());
    }

    #[test]
    fn test_sft_config_deserialize_with_checkpoint_interval() {
        let json = r#"{"checkpoint_interval": 25}"#;
        let config: SftConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.checkpoint_interval, Some(25));
        assert_eq!(config.epochs, 3); // default preserved
    }

    #[test]
    fn test_sft_config_deserialize_without_checkpoint_interval() {
        let json = r#"{"epochs": 5}"#;
        let config: SftConfig = serde_json::from_str(json).unwrap();
        assert!(config.checkpoint_interval.is_none());
        assert_eq!(config.epochs, 5);
    }

    #[test]
    fn test_sft_config_round_trips_resume_checkpoint() {
        let name = "support-checkpoint-step-00000025.kiln-checkpoint";
        let config: SftConfig =
            serde_json::from_value(serde_json::json!({"resume_checkpoint": name})).unwrap();
        assert_eq!(config.resume_checkpoint.as_deref(), Some(name));
        assert_eq!(
            serde_json::to_value(config).unwrap()["resume_checkpoint"],
            name
        );
        assert!(
            serde_json::to_value(SftConfig::default())
                .unwrap()
                .get("resume_checkpoint")
                .is_none(),
            "unset resume state should not add noise to API payloads"
        );
    }

    #[test]
    fn test_grpo_config_deserialize_with_checkpoint_interval() {
        let json = r#"{"checkpoint_interval": 10}"#;
        let config: GrpoConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.checkpoint_interval, Some(10));
        assert_eq!(config.kl_coeff, 0.1); // default preserved
    }

    #[test]
    fn test_grpo_config_round_trips_resume_checkpoint() {
        let name = "reasoning-checkpoint-step-00000010.kiln-checkpoint";
        let config: GrpoConfig =
            serde_json::from_value(serde_json::json!({"resume_checkpoint": name})).unwrap();
        assert_eq!(config.resume_checkpoint.as_deref(), Some(name));
        assert_eq!(
            serde_json::to_value(config).unwrap()["resume_checkpoint"],
            name
        );
        assert!(
            serde_json::to_value(GrpoConfig::default())
                .unwrap()
                .get("resume_checkpoint")
                .is_none()
        );
    }

    #[test]
    fn optimizer_default_is_muon() {
        match Optimizer::default() {
            Optimizer::Muon {
                momentum,
                nesterov,
                ns_iters,
                weight_decay,
            } => {
                assert_eq!(momentum, 0.95);
                assert!(nesterov);
                assert_eq!(ns_iters, 5);
                assert_eq!(weight_decay, 0.0);
            }
            other => panic!("default optimizer should be Muon, got {other:?}"),
        }
    }

    #[test]
    fn optimizer_muon_serializes_with_kind_muon() {
        let opt = Optimizer::default();
        let v: serde_json::Value = serde_json::to_value(opt).unwrap();
        assert_eq!(v["kind"], "muon");
        // momentum is an f32 widened to f64 in JSON — compare with tolerance.
        assert!((v["momentum"].as_f64().unwrap() - 0.95).abs() < 1e-6);
        assert_eq!(v["nesterov"], true);
        assert_eq!(v["ns_iters"], 5);
    }

    #[test]
    fn optimizer_muon_round_trips() {
        let opt = Optimizer::Muon {
            momentum: 0.9,
            nesterov: false,
            ns_iters: 7,
            weight_decay: 0.01,
        };
        let json = serde_json::to_string(&opt).unwrap();
        let back: Optimizer = serde_json::from_str(&json).unwrap();
        assert_eq!(opt, back);
    }

    #[test]
    fn optimizer_muon_minimal_json_fills_defaults() {
        // Bare `{"kind": "muon"}` must fill every Muon field from defaults.
        let opt: Optimizer = serde_json::from_str(r#"{"kind": "muon"}"#).unwrap();
        assert_eq!(opt, Optimizer::default());
    }

    #[test]
    fn optimizer_adamw_and_sgd_still_deserialize() {
        // Back-compat: the old optimizer kinds remain selectable.
        let adamw: Optimizer = serde_json::from_str(r#"{"kind": "adam_w"}"#).unwrap();
        assert!(matches!(adamw, Optimizer::AdamW { .. }));
        let sgd: Optimizer = serde_json::from_str(r#"{"kind": "sgd"}"#).unwrap();
        assert_eq!(sgd, Optimizer::Sgd);
    }

    #[test]
    fn optimizer_hyperparameters_fail_closed_before_execution() {
        assert!(Optimizer::default().validate_hyperparameters().is_ok());
        assert!(Optimizer::Sgd.validate_hyperparameters().is_ok());
        assert!(
            Optimizer::AdamW {
                beta1: f32::NAN,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            }
            .validate_hyperparameters()
            .is_err()
        );
        assert!(
            Optimizer::Muon {
                momentum: 0.95,
                nesterov: true,
                ns_iters: 0,
                weight_decay: 0.0,
            }
            .validate_hyperparameters()
            .is_err()
        );
    }

    #[test]
    fn configs_default_to_muon_optimizer() {
        // Every training-mode config defaults to Muon when no optimizer is given.
        assert!(matches!(
            SftConfig::default().optimizer,
            Optimizer::Muon { .. }
        ));
        assert!(matches!(
            GrpoConfig::default().optimizer,
            Optimizer::Muon { .. }
        ));
        let sft: SftConfig = serde_json::from_str(r#"{"epochs": 2}"#).unwrap();
        assert!(matches!(sft.optimizer, Optimizer::Muon { .. }));
    }

    #[test]
    fn test_grpo_reward_diagnostic_threshold_defaults() {
        let config = GrpoConfig::default();
        assert!(
            (config.reward_saturation_threshold
                - crate::train_receipt::DEFAULT_REWARD_SATURATION_THRESHOLD)
                .abs()
                < 1e-12
        );
        assert!(
            (config.reward_low_variance_threshold
                - crate::train_receipt::DEFAULT_REWARD_LOW_VARIANCE_THRESHOLD)
                .abs()
                < 1e-12
        );
        assert!(config.reward_filter_var_min.is_none());
        assert!(config.reward_filter_var_max.is_none());
        assert_eq!(config.reward_filter_min_groups, 1);
        assert_eq!(config.reward_filter_on_empty, RewardFilterOnEmpty::Fail);
    }

    #[test]
    fn test_grpo_reward_diagnostic_thresholds_deserialize() {
        let json = r#"{
            "reward_saturation_threshold": 0.8,
            "reward_low_variance_threshold": 0.002,
            "reward_filter_var_min": 0.01,
            "reward_filter_var_max": 0.25,
            "reward_filter_min_groups": 3,
            "reward_filter_on_empty": "train-all"
        }"#;
        let config: GrpoConfig = serde_json::from_str(json).unwrap();
        assert!((config.reward_saturation_threshold - 0.8).abs() < 1e-12);
        assert!((config.reward_low_variance_threshold - 0.002).abs() < 1e-12);
        assert_eq!(config.reward_filter_var_min, Some(0.01));
        assert_eq!(config.reward_filter_var_max, Some(0.25));
        assert_eq!(config.reward_filter_min_groups, 3);
        assert_eq!(config.reward_filter_on_empty, RewardFilterOnEmpty::TrainAll);
        assert_eq!(config.kl_coeff, 0.1);
    }

    // ECHO LossConfig + env-var override tests. Env-var manipulation is
    // process-global; each test scrubs the env vars at start to keep the
    // suite hermetic regardless of order. We use the modern `unsafe`
    // wrappers on Rust 1.83+ to make the mutation explicit.
    //
    // ENV_LOCK serializes the env-var-touching tests against each other
    // so cargo test's default parallel runner doesn't race on the
    // process-global env state. Mirrors `trainer::tests::ENV_LOCK`.
    // Acquire via `_env_guard = ENV_LOCK.lock().unwrap_or_else(...)` at
    // the top of every test that calls `clear_kiln_echo_env_vars` or
    // `std::env::set_var`.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_kiln_echo_env_vars() {
        unsafe {
            std::env::remove_var("KILN_ECHO_ENABLED");
            std::env::remove_var("KILN_ECHO_LAMBDA");
            std::env::remove_var("KILN_ECHO_ENV_MASK_MODE");
            std::env::remove_var("KILN_ECHO_WARNING_FILTER");
        }
    }

    #[test]
    fn loss_config_default_has_echo_on_again() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let cfg = LossConfig::default();
        // Resurrection PR2: the env-CE term regained its gradient root on
        // the fused GRPO tape node, so the paper default (λ=0.05) is the
        // out-of-the-box behavior again — and a default config with env
        // tokens VALIDATES.
        let echo = cfg
            .echo
            .as_ref()
            .expect("ECHO defaults ON post-resurrection");
        assert!((echo.lambda - 0.05).abs() < 1e-12, "paper §3.3 default λ");
        assert!(cfg.opd.is_none(), "OPD should be off by default");
        assert!(cfg.validate_for_kt_tape(true).is_ok());
    }

    #[test]
    fn loss_config_validation_rejects_untrainable_compositions() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();

        // ECHO + env tokens trains again (resurrection PR2) — both shapes
        // validate.
        let mut cfg = LossConfig::default();
        cfg.echo = Some(EchoConfig::default());
        assert!(cfg.validate_for_kt_tape(false).is_ok());
        assert!(
            cfg.validate_for_kt_tape(true).is_ok(),
            "echo + env tokens is the flagship agentic shape — must validate"
        );

        // no_policy_loss + ECHO (the default) = §5.5 verifier-free mode:
        // env-CE rows drive while the PG term is masked — VALIDATES.
        let mut cfg = LossConfig::default();
        cfg.no_policy_loss = true;
        assert!(cfg.validate_for_kt_tape(true).is_ok());
        // Without an enabled ECHO term there is nothing to train on.
        cfg.echo = None;
        let err = cfg.validate_for_kt_tape(false).unwrap_err();
        assert!(err.contains("no_policy_loss"), "{err}");
        assert!(err.contains("nothing to train"), "{err}");

        // Reserved OPD slot: point at the real endpoint.
        let mut cfg = LossConfig::default();
        cfg.opd = Some(OpdAuxConfig { lambda: 1.0 });
        let err = cfg.validate_for_kt_tape(false).unwrap_err();
        assert!(err.contains("/v1/train/opd"), "{err}");
    }

    #[test]
    fn loss_config_echo_lambda_returns_zero_when_disabled() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let mut cfg = LossConfig::default();
        cfg.echo = None;
        assert_eq!(cfg.echo_lambda(), 0.0);
        assert!(!cfg.echo_enabled());
    }

    #[test]
    fn kiln_echo_enabled_false_disables_echo() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_ENABLED", "false") };
        // Start from an explicitly-enabled config (the default is now OFF)
        // so the env override has something to disable.
        let mut cfg = LossConfig::default();
        cfg.echo = Some(EchoConfig::default());
        cfg.apply_kiln_echo_env_overrides();
        assert!(cfg.echo.is_none(), "KILN_ECHO_ENABLED=false should disable");
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_lambda_overrides_value() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_LAMBDA", "0.02") };
        let mut cfg = LossConfig::default();
        cfg.apply_kiln_echo_env_overrides();
        assert!(
            (cfg.echo_lambda() - 0.02).abs() < 1e-12,
            "KILN_ECHO_LAMBDA=0.02 should override; got {}",
            cfg.echo_lambda()
        );
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_lambda_re_enables_when_previously_disabled() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_LAMBDA", "0.03") };
        let mut cfg = LossConfig::default();
        cfg.echo = None; // explicitly disabled before override
        cfg.apply_kiln_echo_env_overrides();
        assert!(
            cfg.echo.is_some(),
            "setting LAMBDA env var should also re-enable ECHO"
        );
        assert!((cfg.echo_lambda() - 0.03).abs() < 1e-12);
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_env_mask_mode_overrides() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_ENV_MASK_MODE", "full_obs") };
        let mut cfg = LossConfig::default();
        cfg.apply_kiln_echo_env_overrides();
        let echo = cfg.echo.expect("ECHO should still be on");
        assert_eq!(echo.env_mask_mode, EnvMaskMode::FullObs);
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_warning_filter_overrides() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe { std::env::set_var("KILN_ECHO_WARNING_FILTER", "false") };
        let mut cfg = LossConfig::default();
        cfg.apply_kiln_echo_env_overrides();
        let echo = cfg.echo.expect("ECHO should still be on");
        assert!(!echo.warning_filter);
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_enabled_true_with_lambda_combo() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        unsafe {
            std::env::set_var("KILN_ECHO_ENABLED", "true");
            std::env::set_var("KILN_ECHO_LAMBDA", "0.1");
        }
        let mut cfg = LossConfig::default();
        cfg.echo = None;
        cfg.apply_kiln_echo_env_overrides();
        assert!(cfg.echo.is_some());
        assert!((cfg.echo_lambda() - 0.1).abs() < 1e-12);
        clear_kiln_echo_env_vars();
    }

    #[test]
    fn kiln_echo_no_env_vars_leaves_config_unchanged() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        // Both the (ON again) default and an explicit config survive a
        // pass with no env vars set.
        let mut cfg = LossConfig::default();
        cfg.apply_kiln_echo_env_overrides();
        assert!(cfg.echo.is_some(), "default ON stays ON");

        let mut cfg = LossConfig::default();
        cfg.echo = Some(EchoConfig::default());
        let original_lambda = cfg.echo_lambda();
        cfg.apply_kiln_echo_env_overrides();
        assert!((cfg.echo_lambda() - original_lambda).abs() < 1e-12);
        assert!(cfg.echo.is_some(), "explicit opt-in stays on");
    }

    #[test]
    fn loss_config_no_policy_loss_default_is_false() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let cfg = LossConfig::default();
        assert!(!cfg.no_policy_loss);
    }

    #[test]
    fn loss_config_no_policy_loss_serde_round_trip() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let cfg = LossConfig {
            echo: Some(EchoConfig::default()),
            opd: None,
            no_policy_loss: true,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: LossConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.no_policy_loss);
        assert!((parsed.echo_lambda() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn loss_config_legacy_payload_without_no_policy_loss_defaults_false() {
        // Old payloads that don't include `no_policy_loss` at all still parse.
        let json = r#"{"echo": {"lambda": 0.05}, "opd": null}"#;
        let cfg: LossConfig = serde_json::from_str(json).unwrap();
        assert!(!cfg.no_policy_loss);
    }

    #[test]
    fn echo_config_custom_fields_survive_json_roundtrip() {
        // Every EchoConfig field set away from its default — pin the
        // wire format so a future field rename or default change is a
        // loud test failure rather than a silent compatibility break.
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let custom = EchoConfig {
            lambda: 0.027,
            env_mask_mode: EnvMaskMode::FullObs,
            warning_filter: false,
        };
        let json = serde_json::to_string(&custom).unwrap();
        // The serialized form should mention every non-default field.
        assert!(json.contains("0.027"), "lambda missing from {json}");
        assert!(
            json.contains("full_obs"),
            "env_mask_mode missing from {json}"
        );
        assert!(
            json.contains("\"warning_filter\":false"),
            "warning_filter missing from {json}"
        );

        let parsed: EchoConfig = serde_json::from_str(&json).unwrap();
        assert!((parsed.lambda - 0.027).abs() < 1e-12);
        assert_eq!(parsed.env_mask_mode, EnvMaskMode::FullObs);
        assert!(!parsed.warning_filter);
    }

    #[test]
    fn echo_config_partial_payload_fills_defaults() {
        // HTTP clients may send only `{"lambda": ...}` — the other fields
        // must fall back to their `#[serde(default = ...)]` values.
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_kiln_echo_env_vars();
        let json = r#"{"lambda": 0.012}"#;
        let parsed: EchoConfig = serde_json::from_str(json).unwrap();
        assert!((parsed.lambda - 0.012).abs() < 1e-12);
        assert_eq!(parsed.env_mask_mode, EnvMaskMode::default());
        assert!(parsed.warning_filter, "warning_filter must default to true");
    }

    #[test]
    fn grpo_request_accepts_agentic_groups_alias() {
        // The /v1/train/agentic route reads the same `GrpoRequest` struct
        // as /v1/train/grpo, but clients calling the "agentic" route may
        // semantically prefer `agentic_groups`. Both must parse.
        let body_legacy = r#"{"groups": []}"#;
        let body_agentic = r#"{"agentic_groups": []}"#;
        let parsed_legacy: GrpoRequest = serde_json::from_str(body_legacy).unwrap();
        let parsed_agentic: GrpoRequest = serde_json::from_str(body_agentic).unwrap();
        assert_eq!(parsed_legacy.groups.len(), 0);
        assert_eq!(parsed_agentic.groups.len(), 0);
    }

    #[test]
    fn sft_request_round_trips_agentic_messages() {
        let request: SftRequest = serde_json::from_value(serde_json::json!({
            "examples": [{
                "messages": [
                    {"role": "user", "content": "calculate"},
                    {
                        "role": "assistant",
                        "content": null,
                        "name": "calculator",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": "{\"x\":1}"}
                        }]
                    },
                    {
                        "role": "tool",
                        "content": "1",
                        "name": "calculator",
                        "tool_call_id": "call_1"
                    },
                    {"role": "assistant", "content": "done"}
                ]
            }]
        }))
        .unwrap();

        let messages = &request.examples[0].messages;
        assert_eq!(messages[1].content, "");
        assert_eq!(messages[1].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(messages[2].name.as_deref(), Some("calculator"));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));

        let round_trip: SftRequest =
            serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
        assert_eq!(round_trip.examples[0].messages, *messages);
    }

    /// Capability authors ship `capability.config.json` files with a
    /// `training_phase1_defaults.loss` block that the cap launcher
    /// passes verbatim to `train_tokenized_grpo_group`. Pin the
    /// contract: the shapes used in `pi-terminal-bench-lite` and
    /// `pi-script-fixup` must always deserialize as a valid LossConfig.
    /// If either ever drifts the assertion fires.
    #[test]
    fn cap_config_loss_blocks_deserialize_as_lossconfig() {
        // pi-terminal-bench-lite: ECHO on, OPD off, no_policy_loss absent.
        let tblite = r#"{
            "echo": {
                "lambda": 0.05,
                "env_mask_mode": "env_only",
                "warning_filter": true
            },
            "opd": null
        }"#;
        let parsed: LossConfig = serde_json::from_str(tblite).unwrap();
        assert!((parsed.echo_lambda() - 0.05).abs() < 1e-12);
        assert!(parsed.opd.is_none());
        assert!(!parsed.no_policy_loss);
        let echo = parsed.echo.as_ref().unwrap();
        assert_eq!(echo.env_mask_mode, EnvMaskMode::EnvOnly);
        assert!(echo.warning_filter);

        // pi-script-fixup: same shape plus no_policy_loss = true.
        let fixup = r#"{
            "echo": {
                "lambda": 0.05,
                "env_mask_mode": "env_only",
                "warning_filter": true
            },
            "opd": null,
            "no_policy_loss": true
        }"#;
        let parsed: LossConfig = serde_json::from_str(fixup).unwrap();
        assert!(
            parsed.no_policy_loss,
            "verifier-free cap must parse with no_policy_loss=true"
        );
        assert!((parsed.echo_lambda() - 0.05).abs() < 1e-12);
    }
}
