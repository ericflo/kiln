use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;

use crate::config::{
    BatchingRuntimeConfig, ConfigValueSource, DecodeRuntimeConfig, OperationalRuntimeConfig,
    ResolvedAcceleratorRuntimePolicy, ServingProfileDiagnostics, SpecMethod,
    StreamingPrefillRuntimeConfig,
};
use crate::memory_observability::CachedMemoryGovernorObservation;
use crate::state::{AppState, DirectDecodeRendezvousRuntimeState, ModelBackend, TrainingWorkload};

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const SPECULATIVE_SERVING_UNAVAILABLE_REASON: &str =
    "pending_cancel_safe_local_accelerator_qualification";

fn gib(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB
}

fn prefix_cache_effective_reason(
    requested_enabled: bool,
    effective_enabled: bool,
    inference_device: kiln_tensor::Device,
) -> &'static str {
    if effective_enabled {
        "active"
    } else if !requested_enabled {
        "configuration"
    } else if matches!(inference_device, kiln_tensor::Device::Vulkan(_)) {
        "vulkan_correctness_quarantine"
    } else {
        "backend_unavailable"
    }
}

#[derive(Serialize)]
struct ConfigResponse {
    serving_profile: ServingProfileDiagnostics,
    accelerator_runtime: ResolvedAcceleratorRuntimePolicy,
    /// Nonblocking full graph-cache snapshot governed by the resolved policy.
    rocm_graphs: Option<kiln_model::RocmGraphStats>,
    rocm_graphs_unavailable_reason:
        Option<crate::rocm_graph_observability::RocmGraphUnavailableReason>,
    /// Phase/transient telemetry independent of the model and graph-runner
    /// locks, including in-progress work.
    rocm_graph_telemetry: Option<kiln_model::RocmGraphLiveTelemetry>,
    rocm_graph_telemetry_unavailable_reason:
        Option<crate::rocm_graph_observability::RocmGraphUnavailableReason>,
    cuda_graphs: CudaGraphConfigResponse,
    decode_runtime: DecodeRuntimeConfig,
    batching: BatchingConfigResponse,
    prefix_cache: PrefixCacheConfigResponse,
    streaming_prefill: StreamingPrefillRuntimeConfig,
    speculative: SpeculativeConfig,
    operational: OperationalRuntimeConfig,
    vram: VramConfig,
    kv_cache: KvCacheConfig,
    training: TrainingConfig,
    memory_budget: MemoryBudgetConfig,
    generation: GenerationConfig,
}

#[derive(Serialize)]
pub(crate) struct CudaGraphConfigResponse {
    requested: bool,
    capture_allowed_by_serving_profile: bool,
    effective_policy_enabled: bool,
    max_cached_graphs: usize,
    stable_paged_metadata: bool,
    batched_capture_available: bool,
    restart_required_to_change: bool,
}

pub(crate) fn cuda_graph_config_response(state: &AppState) -> CudaGraphConfigResponse {
    let capture_allowed_by_serving_profile =
        state.serving_profile.runtime_policy().live_graph_capture;
    CudaGraphConfigResponse {
        requested: state.memory_config.cuda_graphs,
        capture_allowed_by_serving_profile,
        effective_policy_enabled: state.memory_config.cuda_graphs
            && capture_allowed_by_serving_profile,
        max_cached_graphs: state.memory_config.cuda_graph_cache_entries,
        stable_paged_metadata: true,
        batched_capture_available: false,
        restart_required_to_change: true,
    }
}

#[derive(Serialize)]
struct BatchingConfigResponse {
    configuration: BatchingRuntimeConfig,
    actor_active: bool,
    direct_decode_rendezvous: DirectDecodeRendezvousRuntimeState,
}

#[derive(Serialize)]
struct PrefixCacheConfigResponse {
    configuration: crate::config::PrefixCacheConfig,
    effective_enabled: bool,
    effective_reason: &'static str,
    effective_max_blocks: usize,
    effective_max_entries: usize,
    effective_max_state_bytes: u64,
}

#[derive(Serialize)]
struct VramConfig {
    probe_selector: String,
    unified: bool,
    physical_capacity_bytes: u64,
    physical_capacity_gib: f64,
    physical_capacity_source: String,
    configured_capacity_bytes: Option<u64>,
    configured_capacity_gib: Option<f64>,
    effective_capacity_bytes: u64,
    effective_capacity_gib: f64,
    effective_capacity_source: String,
    configured_capacity_clamped: bool,
    live: LiveMemoryConfig,
    governor: MemoryGovernorConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    vulkan_buffer_pool: Option<VulkanBufferPoolConfig>,
}

#[derive(Serialize)]
struct VulkanBufferPoolConfig {
    max_retained_bytes: u64,
    max_retained_gib: f64,
    retained_bytes: u64,
    retained_gib: f64,
    free_bytes: u64,
    borrowed_bytes: u64,
    cache_hits: u64,
    cache_misses: u64,
    device_local_cache_misses: u64,
    host_visible_cache_misses: u64,
    eviction_count: u64,
    evicted_bytes: u64,
    uncached_allocation_count: u64,
    uncached_allocated_bytes: u64,
}

#[derive(Serialize)]
struct LiveMemoryConfig {
    total_bytes: u64,
    total_gib: f64,
    used_bytes: u64,
    used_gib: f64,
    available_bytes: u64,
    available_gib: f64,
    effective_capacity_available_bytes: u64,
    effective_capacity_available_gib: f64,
    usable_after_governor_floor_bytes: u64,
    usable_after_governor_floor_gib: f64,
    soft_reserved_bytes: u64,
    soft_reserved_gib: f64,
    pressure: String,
    source: String,
    sample_age_ms: u64,
    sample_max_age_ms: u64,
    sample_stale: bool,
    sampler_required: bool,
    sampler_running: bool,
    sampler_healthy: bool,
    raw_observations: RawMemoryObservations,
}

#[derive(Serialize)]
struct MemoryTierConfig {
    total_bytes: u64,
    total_gib: f64,
    used_bytes: u64,
    used_gib: f64,
    free_bytes: u64,
    free_gib: f64,
}

#[derive(Serialize)]
struct RawMemoryObservations {
    probe_failed: bool,
    driver_total_bytes: Option<u64>,
    driver_used_bytes: Option<u64>,
    driver_free_bytes: Option<u64>,
    driver_vram_total_bytes: Option<u64>,
    driver_vram_used_bytes: Option<u64>,
    driver_gtt_total_bytes: Option<u64>,
    driver_gtt_used_bytes: Option<u64>,
    host_total_bytes: Option<u64>,
    host_available_bytes: Option<u64>,
    cgroup_limit_bytes: Option<u64>,
    cgroup_high_bytes: Option<u64>,
    cgroup_current_bytes: Option<u64>,
    cgroup_remaining_bytes: Option<u64>,
    unified_reserve_bytes: Option<u64>,
    host_backed: Option<MemoryTierConfig>,
}

#[derive(Serialize)]
struct MemoryGovernorConfig {
    floor_bytes: u64,
    floor_gib: f64,
    capacity_limit_bytes: u64,
    capacity_limit_gib: f64,
    probe_ms: u64,
    reclaim_mode_requested: &'static str,
    reclaim_mode_effective: &'static str,
    reclaim_mode_source: ConfigValueSource,
    reclaim_disabled_by_serving_profile: bool,
}

#[derive(Serialize)]
struct KvCacheConfig {
    num_blocks: usize,
    num_blocks_source: &'static str,
    fp8_enabled: bool,
    autoscaler: crate::kv_autoscaler::KvAutoscalerState,
}

#[derive(Serialize)]
struct TrainingConfig {
    runtime_device: Option<String>,
    model_weight_device: String,
    native_training_supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_training_unavailable_reason: Option<String>,
    optimizer_support: Option<TrainingOptimizerSupportConfig>,
    checkpoint_policy: kiln_train::GradientCheckpointPolicy,
    checkpoint_boundary_policy: kiln_train::CheckpointBoundaryPolicy,
    checkpoint_segments: usize,
    checkpoint_segments_source: &'static str,
    checkpointing_enabled: bool,
}

#[derive(Serialize)]
struct TrainingOptimizerSupportConfig {
    schema: TrainingOptimizerSupportSchema,
    backend: String,
    device: String,
    base_weight_dtype: String,
    resolved_lora_parameter_dtype: Option<String>,
    immutable_after_startup: bool,
    rounding_modes: Vec<&'static str>,
    backend_implementation_rounding_modes: Vec<&'static str>,
    optimizer_tuple_kinds: Vec<&'static str>,
    workloads: Vec<TrainingWorkloadSupportConfig>,
    optimizers: Vec<TrainingOptimizerKindConfig>,
}

#[derive(Serialize)]
struct TrainingOptimizerSupportSchema {
    id: &'static str,
    version: u32,
}

#[derive(Serialize)]
struct TrainingOptimizerKindConfig {
    kind: &'static str,
    backend_implementation: TrainingOptimizerImplementationConfig,
    optimizer_tuple: TrainingOptimizerTupleConfig,
}

#[derive(Serialize)]
struct TrainingOptimizerImplementationConfig {
    supported: bool,
    route: &'static str,
    native_device_hook: bool,
    parameter_dtypes: Vec<&'static str>,
}

#[derive(Serialize)]
struct TrainingOptimizerTupleConfig {
    supported: bool,
    unavailable_reason: Option<String>,
    lora_rank: TrainingOptimizerRankConfig,
}

#[derive(Serialize)]
struct TrainingWorkloadSupportConfig {
    workload: &'static str,
    supported: bool,
    unavailable_reason: Option<String>,
    allowed_optimizer_kinds: Vec<&'static str>,
}

#[derive(Serialize)]
struct TrainingOptimizerRankConfig {
    minimum: usize,
    maximum: Option<usize>,
    backend_maximum: Option<usize>,
    model_maximum: usize,
    live_memory_admission_required: bool,
}

#[derive(Serialize)]
struct MemoryBudgetConfig {
    total_vram_bytes: u64,
    total_vram_gib: f64,
    model_bytes: u64,
    model_gib: f64,
    kv_cache_bytes: u64,
    kv_cache_gib: f64,
    training_budget_bytes: u64,
    training_budget_gib: f64,
    inference_memory_fraction: f64,
}

#[derive(Serialize)]
struct GenerationConfig {
    default_thinking_enabled: Option<bool>,
    default_thinking_budget_tokens: Option<usize>,
    default_thinking_budget_ms: Option<u64>,
    fold_reasoning_into_content: bool,
}

#[derive(Serialize)]
struct SpeculativeConfig {
    enabled: bool,
    configured_method: SpecMethod,
    configured_effective_method: SpecMethod,
    serving_effective_method: SpecMethod,
    num_speculative_tokens: usize,
    draft_layers: usize,
    configured_policy_immutable_after_startup: bool,
    serving_routable: bool,
    serving_unavailable_reason: &'static str,
    draft_token_ceiling: usize,
    backend_mtp: SpeculativeBackendMtpConfig,
}

#[derive(Serialize)]
struct SpeculativeBackendMtpConfig {
    support: &'static str,
    native: bool,
}

async fn get_config(State(state): State<AppState>) -> Json<ConfigResponse> {
    let memory_observation =
        CachedMemoryGovernorObservation::capture_global_for(state.vram_probe_selector);

    // Get actual num_blocks from the running backend
    let (num_blocks, num_blocks_source) = match state.backend.as_ref() {
        ModelBackend::Real { block_manager, .. } => {
            let bm = block_manager.lock().unwrap();
            let source = if state.memory_config.num_blocks.is_some() {
                "configured"
            } else {
                "auto"
            };
            (bm.num_blocks(), source)
        }
        ModelBackend::Mock { scheduler, .. } => {
            let sched = scheduler.try_lock();
            match sched {
                Ok(s) => (s.block_manager().num_blocks(), "mock"),
                Err(_) => (0, "unknown"),
            }
        }
    };
    let (prefix_cache_effective_enabled, prefix_cache_stats) = match state.backend.as_ref() {
        ModelBackend::Real { prefix_cache, .. } => {
            let cache = prefix_cache.lock().unwrap();
            (cache.is_enabled(), cache.stats())
        }
        ModelBackend::Mock { scheduler, .. } => match scheduler.try_lock() {
            Ok(scheduler) => {
                let stats = scheduler.prefix_cache_stats();
                (stats.max_blocks > 0, stats)
            }
            Err(_) => (false, kiln_scheduler::PrefixCacheStats::default()),
        },
    };
    let prefix_cache_effective_reason = prefix_cache_effective_reason(
        state.prefix_cache_config.enabled,
        prefix_cache_effective_enabled,
        state.inference_device,
    );

    // Determine checkpoint segments
    let ckpt = kiln_train::CheckpointConfig::from_runtime(
        state.model_config.num_layers,
        &state.training_runtime,
    );
    let segments_source = match state.training_runtime.gradient_checkpoint_policy() {
        kiln_train::GradientCheckpointPolicy::Auto if ckpt.auto_configured => "auto",
        kiln_train::GradientCheckpointPolicy::Auto => "conservative_fallback",
        kiln_train::GradientCheckpointPolicy::ExplicitSegments { .. } => "configured",
        kiln_train::GradientCheckpointPolicy::Disabled { .. } => "disabled",
    };
    // Resolve workload gates before taking the config runner lock. Re-entering
    // the same RwLock while a writer is queued is not guaranteed to make
    // progress on every platform.
    let workload_reasons = [
        TrainingWorkload::Sft,
        TrainingWorkload::Grpo,
        TrainingWorkload::Opd,
        TrainingWorkload::DistillRefresh,
    ]
    .map(|workload| {
        (
            workload,
            state.training_workload_unavailable_reason(workload),
        )
    });
    let optimizer_support = build_training_optimizer_support(&state, &workload_reasons);
    let native_training_unavailable_reason =
        if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
            Some("mock backend does not execute native training".to_string())
        } else if workload_reasons.iter().all(|(_, reason)| reason.is_some()) {
            workload_reasons
                .iter()
                .find_map(|(_, reason)| reason.clone())
        } else if let Some(support) = optimizer_support.as_ref() {
            if support.optimizer_tuple_kinds.is_empty() {
                support
                    .optimizers
                    .iter()
                    .find_map(|optimizer| optimizer.optimizer_tuple.unavailable_reason.clone())
                    .or_else(|| {
                        Some("no optimizer tuple is admitted for the resident weights".to_string())
                    })
            } else {
                None
            }
        } else {
            Some("optimizer support is unavailable for the resident runner".to_string())
        };

    let b = &state.memory_budget;
    let rocm_graph_observation = crate::rocm_graph_observability::observe_rocm_graphs(&state);

    Json(ConfigResponse {
        serving_profile: state.serving_profile.diagnostics(),
        accelerator_runtime: state.accelerator_runtime_policy,
        rocm_graphs: rocm_graph_observation.stats,
        rocm_graphs_unavailable_reason: rocm_graph_observation.stats_unavailable_reason,
        rocm_graph_telemetry: rocm_graph_observation.telemetry,
        rocm_graph_telemetry_unavailable_reason: rocm_graph_observation
            .telemetry_unavailable_reason,
        cuda_graphs: cuda_graph_config_response(&state),
        decode_runtime: state.decode_runtime_config,
        batching: BatchingConfigResponse {
            configuration: state.batching_runtime_config,
            actor_active: matches!(
                state.backend.as_ref(),
                ModelBackend::Real {
                    batching_engine: Some(_),
                    ..
                }
            ),
            direct_decode_rendezvous: state.direct_decode_rendezvous_runtime_state(),
        },
        prefix_cache: PrefixCacheConfigResponse {
            configuration: state.prefix_cache_config.clone(),
            effective_enabled: prefix_cache_effective_enabled,
            effective_reason: prefix_cache_effective_reason,
            effective_max_blocks: prefix_cache_stats.max_blocks,
            effective_max_entries: prefix_cache_stats.max_entries,
            effective_max_state_bytes: prefix_cache_stats.max_state_bytes,
        },
        streaming_prefill: state.streaming_prefill_runtime_config,
        speculative: build_speculative_config(&state),
        operational: state.operational_runtime.as_ref().clone(),
        vram: build_vram_config(&state, memory_observation),
        kv_cache: KvCacheConfig {
            num_blocks,
            num_blocks_source,
            fp8_enabled: match state.backend.as_ref() {
                ModelBackend::Real { paged_cache, .. } => paged_cache.is_fp8(),
                ModelBackend::Mock { .. } => false,
            },
            autoscaler: state.kv_autoscaler,
        },
        training: TrainingConfig {
            runtime_device: state
                .training_runtime
                .runtime_device()
                .map(|device| device.short_name().to_string()),
            model_weight_device: state.model_weight_device.short_name().to_string(),
            native_training_supported: native_training_unavailable_reason.is_none(),
            native_training_unavailable_reason,
            optimizer_support,
            checkpoint_policy: state.training_runtime.gradient_checkpoint_policy(),
            checkpoint_boundary_policy: state.training_runtime.checkpoint_boundary_policy(),
            checkpoint_segments: if ckpt.enabled { ckpt.num_segments } else { 0 },
            checkpoint_segments_source: segments_source,
            checkpointing_enabled: ckpt.enabled,
        },
        memory_budget: MemoryBudgetConfig {
            total_vram_bytes: b.total_vram_bytes,
            total_vram_gib: gib(b.total_vram_bytes),
            model_bytes: b.model_memory_bytes,
            model_gib: gib(b.model_memory_bytes),
            kv_cache_bytes: b.kv_cache_bytes,
            kv_cache_gib: gib(b.kv_cache_bytes),
            training_budget_bytes: b.training_budget_bytes,
            training_budget_gib: gib(b.training_budget_bytes),
            inference_memory_fraction: b.inference_memory_fraction,
        },
        generation: GenerationConfig {
            default_thinking_enabled: state.default_thinking_enabled,
            default_thinking_budget_tokens: state.default_thinking_budget_tokens,
            default_thinking_budget_ms: state.default_thinking_budget_ms,
            fold_reasoning_into_content: state.fold_reasoning_into_content,
        },
    })
}

fn build_training_optimizer_support(
    state: &AppState,
    workload_reasons: &[(TrainingWorkload, Option<String>); 4],
) -> Option<TrainingOptimizerSupportConfig> {
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return None;
    };
    let runner = runner.read().ok()?;
    let capabilities = runner.backend_capabilities();
    let training = capabilities.training;
    let base_weight_device = runner.weights.embed_tokens.device();
    let base_weight_dtype = runner.weights.embed_tokens.dtype();
    let backend_identity_reason = (capabilities.device != base_weight_device).then(|| {
        format!(
            "backend `{}` reports device {} but resident weights are on {}",
            capabilities.backend, capabilities.device, base_weight_device
        )
    });
    let model_lora_rank_ceiling =
        crate::training_preflight::model_lora_rank_ceiling(&state.model_config);
    Some(build_training_optimizer_support_from_capabilities(
        capabilities.backend,
        capabilities.device,
        base_weight_dtype,
        training,
        model_lora_rank_ceiling,
        backend_identity_reason.as_deref(),
        workload_reasons,
    ))
}

fn build_training_optimizer_support_from_capabilities(
    backend: &str,
    device: kiln_tensor::Device,
    base_weight_dtype: kiln_tensor::DType,
    training: kiln_model::BackendTrainingCapabilities,
    model_lora_rank_ceiling: usize,
    optimizer_tuple_unavailable_reason: Option<&str>,
    workload_reasons: &[(TrainingWorkload, Option<String>); 4],
) -> TrainingOptimizerSupportConfig {
    use kiln_model::{TrainingOptimizerKind, TrainingOptimizerRounding};

    const PRODUCT_ROUNDING: TrainingOptimizerRounding = TrainingOptimizerRounding::RoundToNearest;
    let base_weight_dtype_supported = training
        .precision
        .base_weight_dtypes
        .contains(&base_weight_dtype);
    let resolved_lora_parameter_dtype = base_weight_dtype_supported.then(|| {
        training
            .precision
            .lora_parameter_dtype_for_base_weight(base_weight_dtype)
            .short_name()
            .to_string()
    });
    let backend_implementation_rounding_modes = training
        .optimizer
        .rounding_modes
        .iter()
        .map(|rounding| rounding.label())
        .collect();
    let mut optimizer_tuple_kinds = Vec::new();
    let mut optimizers = Vec::new();

    for kind in [
        TrainingOptimizerKind::Muon,
        TrainingOptimizerKind::AdamW,
        TrainingOptimizerKind::Sgd,
    ] {
        let parameter_dtypes = training
            .optimizer
            .parameter_dtypes(kind)
            .iter()
            .map(|dtype| dtype.short_name())
            .collect::<Vec<_>>();
        let implementation_supported =
            !parameter_dtypes.is_empty() && !training.optimizer.rounding_modes.is_empty();
        let implementation_route = if !implementation_supported {
            "unavailable"
        } else if backend == "cpu" && matches!(device, kiln_tensor::Device::Cpu) {
            "portable_reference"
        } else {
            "native_device_hook"
        };
        let minimum = if matches!(kind, TrainingOptimizerKind::Muon) {
            training.optimizer.muon_min_lora_rank.unwrap_or(1).max(1)
        } else {
            1
        };
        let backend_maximum = if matches!(kind, TrainingOptimizerKind::Muon) {
            training.optimizer.muon_max_lora_rank
        } else {
            None
        };
        let maximum = Some(
            backend_maximum
                .map(|maximum| maximum.min(model_lora_rank_ceiling))
                .unwrap_or(model_lora_rank_ceiling),
        );
        let rank_range_is_valid = maximum.is_some_and(|maximum| minimum <= maximum);
        let resolution = if let Some(reason) = optimizer_tuple_unavailable_reason {
            Err(reason.to_string())
        } else if !rank_range_is_valid {
            Err(format!(
                "resident model and backend report no common LoRA rank range: minimum {minimum}, model maximum {model_lora_rank_ceiling}, backend maximum {backend_maximum:?}"
            ))
        } else {
            training
                .resolve_optimizer_request(kind, base_weight_dtype, PRODUCT_ROUNDING, minimum)
                .map(|_| ())
                .map_err(|error| error.to_string())
        };
        let optimizer_tuple_supported = resolution.is_ok();
        if optimizer_tuple_supported {
            optimizer_tuple_kinds.push(kind.label());
        }
        optimizers.push(TrainingOptimizerKindConfig {
            kind: kind.label(),
            backend_implementation: TrainingOptimizerImplementationConfig {
                supported: implementation_supported,
                route: implementation_route,
                native_device_hook: implementation_route == "native_device_hook",
                parameter_dtypes,
            },
            optimizer_tuple: TrainingOptimizerTupleConfig {
                supported: optimizer_tuple_supported,
                unavailable_reason: resolution.err(),
                lora_rank: TrainingOptimizerRankConfig {
                    minimum,
                    maximum,
                    backend_maximum,
                    model_maximum: model_lora_rank_ceiling,
                    live_memory_admission_required: true,
                },
            },
        });
    }

    let workloads = workload_reasons
        .iter()
        .map(
            |(workload, unavailable_reason)| TrainingWorkloadSupportConfig {
                workload: workload.label(),
                supported: unavailable_reason.is_none() && !optimizer_tuple_kinds.is_empty(),
                unavailable_reason: unavailable_reason.clone().or_else(|| {
                    optimizer_tuple_kinds.is_empty().then(|| {
                        "no optimizer tuple is admitted for the resident weights".to_string()
                    })
                }),
                allowed_optimizer_kinds: if unavailable_reason.is_none() {
                    optimizer_tuple_kinds.clone()
                } else {
                    Vec::new()
                },
            },
        )
        .collect();

    TrainingOptimizerSupportConfig {
        schema: TrainingOptimizerSupportSchema {
            id: "kiln.training-optimizer-support",
            version: 1,
        },
        backend: backend.to_string(),
        device: device.short_name().to_string(),
        base_weight_dtype: base_weight_dtype.short_name().to_string(),
        resolved_lora_parameter_dtype,
        immutable_after_startup: true,
        rounding_modes: vec![PRODUCT_ROUNDING.label()],
        backend_implementation_rounding_modes,
        optimizer_tuple_kinds,
        workloads,
        optimizers,
    }
}

fn build_speculative_config(state: &AppState) -> SpeculativeConfig {
    let configured = state.speculative_config;
    let configured_effective_method = configured.effective_method();
    let runtime = state.speculative_runtime_policy;
    let native_mtp = runtime.mtp_support.is_native();
    SpeculativeConfig {
        enabled: configured.enabled,
        configured_method: configured.method,
        configured_effective_method,
        serving_effective_method: SpecMethod::Off,
        num_speculative_tokens: configured.num_speculative_tokens,
        draft_layers: configured.draft_layers,
        configured_policy_immutable_after_startup: true,
        serving_routable: false,
        serving_unavailable_reason: SPECULATIVE_SERVING_UNAVAILABLE_REASON,
        draft_token_ceiling: crate::config::MAX_SPECULATIVE_DRAFT_TOKENS,
        backend_mtp: SpeculativeBackendMtpConfig {
            support: support_name(runtime.mtp_support),
            native: native_mtp,
        },
    }
}

fn support_name(support: kiln_model::Support) -> &'static str {
    use kiln_model::Support;

    match support {
        Support::Native => "native",
        Support::NativeWithConstraints => "native_with_constraints",
        Support::HostFallbackAllowed => "host_fallback_allowed",
        Support::Declined => "declined",
        Support::Unsupported => "unsupported",
        Support::DisabledByEnv => "disabled_by_env",
        Support::RequiresFeature => "requires_feature",
    }
}

fn build_vram_config(state: &AppState, observation: CachedMemoryGovernorObservation) -> VramConfig {
    let live = observation.snapshot;
    let resolution = state.vram_capacity_resolution;
    let vram = resolution.effective;
    let floor_bytes = state.memory_config.governor_config().floor_bytes;
    let effective_capacity_available = live
        .free_bytes
        .min(vram.total_bytes.saturating_sub(live.used_bytes));
    let usable_after_governor_floor = observation
        .available_bytes
        .min(effective_capacity_available.saturating_sub(floor_bytes));
    let requested_reclaim_mode = state.memory_config.reclaim_mode.mode().as_str();
    let reclaim_enabled = state.serving_profile.runtime_policy().allocator_reclaim;

    VramConfig {
        probe_selector: format_probe_selector(state.vram_probe_selector),
        unified: vram.unified,
        physical_capacity_bytes: resolution.physical.total_bytes,
        physical_capacity_gib: gib(resolution.physical.total_bytes),
        physical_capacity_source: resolution.physical.source.to_string(),
        configured_capacity_bytes: resolution.requested_bytes,
        configured_capacity_gib: resolution.requested_bytes.map(gib),
        effective_capacity_bytes: vram.total_bytes,
        effective_capacity_gib: gib(vram.total_bytes),
        effective_capacity_source: vram.source.to_string(),
        configured_capacity_clamped: resolution.clamped,
        live: LiveMemoryConfig {
            total_bytes: live.total_bytes,
            total_gib: gib(live.total_bytes),
            used_bytes: live.used_bytes,
            used_gib: gib(live.used_bytes),
            available_bytes: live.free_bytes,
            available_gib: gib(live.free_bytes),
            effective_capacity_available_bytes: effective_capacity_available,
            effective_capacity_available_gib: gib(effective_capacity_available),
            usable_after_governor_floor_bytes: usable_after_governor_floor,
            usable_after_governor_floor_gib: gib(usable_after_governor_floor),
            soft_reserved_bytes: observation.soft_reserved_bytes,
            soft_reserved_gib: gib(observation.soft_reserved_bytes),
            pressure: format!("{:?}", observation.pressure),
            source: live.source.to_string(),
            sample_age_ms: observation
                .sample_status
                .age
                .as_millis()
                .min(u64::MAX as u128) as u64,
            sample_max_age_ms: observation
                .sample_status
                .max_age
                .as_millis()
                .min(u64::MAX as u128) as u64,
            sample_stale: observation.sample_status.stale,
            sampler_required: observation.sample_status.sampler_required,
            sampler_running: observation.sample_status.sampler_running,
            sampler_healthy: observation.sample_status.healthy,
            raw_observations: RawMemoryObservations {
                probe_failed: live.observations.probe_failed,
                driver_total_bytes: live.observations.driver_total_bytes,
                driver_used_bytes: live.observations.driver_used_bytes,
                driver_free_bytes: live.observations.driver_free_bytes,
                driver_vram_total_bytes: live.observations.driver_vram_total_bytes,
                driver_vram_used_bytes: live.observations.driver_vram_used_bytes,
                driver_gtt_total_bytes: live.observations.driver_gtt_total_bytes,
                driver_gtt_used_bytes: live.observations.driver_gtt_used_bytes,
                host_total_bytes: live.observations.host_total_bytes,
                host_available_bytes: live.observations.host_available_bytes,
                cgroup_limit_bytes: live.observations.cgroup_limit_bytes,
                cgroup_high_bytes: live.observations.cgroup_high_bytes,
                cgroup_current_bytes: live.observations.cgroup_current_bytes,
                cgroup_remaining_bytes: live.observations.cgroup_remaining_bytes,
                unified_reserve_bytes: live.observations.unified_reserve_bytes,
                host_backed: live.observations.host_backed.map(|tier| MemoryTierConfig {
                    total_bytes: tier.total_bytes,
                    total_gib: gib(tier.total_bytes),
                    used_bytes: tier.used_bytes,
                    used_gib: gib(tier.used_bytes),
                    free_bytes: tier.free_bytes,
                    free_gib: gib(tier.free_bytes),
                }),
            },
        },
        governor: MemoryGovernorConfig {
            floor_bytes,
            floor_gib: gib(floor_bytes),
            capacity_limit_bytes: vram.total_bytes,
            capacity_limit_gib: gib(vram.total_bytes),
            probe_ms: state.memory_config.probe_ms,
            reclaim_mode_requested: requested_reclaim_mode,
            reclaim_mode_effective: if reclaim_enabled {
                requested_reclaim_mode
            } else {
                "off"
            },
            reclaim_mode_source: state.memory_config.reclaim_mode.source(),
            reclaim_disabled_by_serving_profile: !reclaim_enabled,
        },
        vulkan_buffer_pool: kiln_model::vulkan_buffer_pool_stats().map(|pool| {
            VulkanBufferPoolConfig {
                max_retained_bytes: pool.max_retained_bytes,
                max_retained_gib: gib(pool.max_retained_bytes),
                retained_bytes: pool.total_bytes,
                retained_gib: gib(pool.total_bytes),
                free_bytes: pool.free_bytes,
                borrowed_bytes: pool.borrowed_bytes(),
                cache_hits: pool.cache_hits,
                cache_misses: pool.cache_misses,
                device_local_cache_misses: pool.device_local_cache_misses,
                host_visible_cache_misses: pool.host_visible_cache_misses,
                eviction_count: pool.eviction_count,
                evicted_bytes: pool.evicted_bytes,
                uncached_allocation_count: pool.uncached_allocation_count,
                uncached_allocated_bytes: pool.uncached_allocated_bytes,
            }
        }),
    }
}

fn format_probe_selector(selector: kiln_memory::vram::VramProbeSelector) -> String {
    use kiln_memory::vram::{LinuxDrmVendor, VramProbeSelector};
    match selector {
        VramProbeSelector::Auto => "auto".to_owned(),
        VramProbeSelector::Nvidia(index) => format!("nvidia:{index}"),
        VramProbeSelector::LinuxDrm {
            index,
            vendor: Some(LinuxDrmVendor::Amd),
        } => format!("linux-drm:amd:{index}"),
        VramProbeSelector::LinuxDrm {
            index,
            vendor: Some(LinuxDrmVendor::Intel),
        } => format!("linux-drm:intel:{index}"),
        VramProbeSelector::LinuxDrm {
            index,
            vendor: None,
        } => format!("linux-drm:any:{index}"),
        VramProbeSelector::AppleUnified => "apple-unified".to_owned(),
        VramProbeSelector::None => "none".to_owned(),
    }
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/config", get(get_config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use kiln_core::config::ModelConfig;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn make_test_state() -> AppState {
        let model_config = ModelConfig::qwen3_5_4b();
        let scheduler = Scheduler::new(SchedulerConfig::default(), 256);
        AppState::new_mock(
            model_config.clone(),
            scheduler,
            Arc::new(MockEngine::new(model_config)),
            crate::api::test_tokenizer(),
            300,
            "Qwen3.5-4B".to_string(),
        )
    }

    #[test]
    fn prefix_cache_reason_uses_logical_inference_device() {
        assert_eq!(
            prefix_cache_effective_reason(true, true, kiln_tensor::Device::Vulkan(0)),
            "active"
        );
        assert_eq!(
            prefix_cache_effective_reason(false, false, kiln_tensor::Device::Vulkan(0)),
            "configuration"
        );
        assert_eq!(
            prefix_cache_effective_reason(true, false, kiln_tensor::Device::Vulkan(0)),
            "vulkan_correctness_quarantine"
        );
        assert_eq!(
            prefix_cache_effective_reason(true, false, kiln_tensor::Device::Cpu),
            "backend_unavailable"
        );
    }

    #[tokio::test]
    async fn config_reports_vulkan_quarantine_with_cpu_resident_weights() {
        let model_config = ModelConfig::qwen3_5_4b();
        let scheduler = Scheduler::new(
            SchedulerConfig {
                prefix_cache_enabled: false,
                ..SchedulerConfig::default()
            },
            256,
        );
        let mut state = AppState::new_mock(
            model_config.clone(),
            scheduler,
            Arc::new(MockEngine::new(model_config)),
            crate::api::test_tokenizer(),
            300,
            "Qwen3.5-4B".to_string(),
        );
        state.inference_device = kiln_tensor::Device::Vulkan(0);
        assert_eq!(state.model_weight_device, kiln_tensor::Device::Cpu);

        let response = routes()
            .with_state(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["prefix_cache"]["configuration"]["enabled"], true);
        assert_eq!(json["prefix_cache"]["effective_enabled"], false);
        assert_eq!(
            json["prefix_cache"]["effective_reason"],
            "vulkan_correctness_quarantine"
        );
        assert_eq!(json["prefix_cache"]["effective_max_blocks"], 0);
        assert_eq!(json["prefix_cache"]["effective_max_entries"], 0);
        assert_eq!(json["prefix_cache"]["effective_max_state_bytes"], 0);
    }

    #[tokio::test]
    async fn config_reports_profile_provenance_and_every_effective_policy() {
        let mut state = make_test_state();
        state.serving_profile = crate::config::ServingProfileSetting::new(
            crate::config::ServingProfile::Experimental,
            crate::config::ConfigValueSource::Environment,
        );
        state.accelerator_runtime_policy = crate::config::AcceleratorRuntimeConfig::default()
            .resolved_policy(state.serving_profile);
        state.default_thinking_budget_tokens = Some(64);
        state.default_thinking_budget_ms = Some(1_500);
        let checkpoint_boundary_policy = kiln_train::CheckpointBoundaryPolicy::from_parts(
            kiln_train::CheckpointBoundaryRecomputeMode::Enabled,
            4096,
            Some(3),
            2 * 1024 * 1024 * 1024,
        )
        .unwrap();
        state.training_runtime = state
            .training_runtime
            .with_checkpoint_boundary_policy(checkpoint_boundary_policy);
        let expected_streaming_prefill =
            serde_json::to_value(state.streaming_prefill_runtime_config).unwrap();
        let expected_checkpoint_boundary_policy =
            serde_json::to_value(checkpoint_boundary_policy).unwrap();
        let app = routes().with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["streaming_prefill"], expected_streaming_prefill);
        assert_eq!(json["serving_profile"]["profile"], "experimental");
        assert_eq!(json["serving_profile"]["source"], "environment");
        assert_eq!(json["serving_profile"]["immutable_after_startup"], true);
        assert_eq!(json["serving_profile"]["request_overrides_allowed"], false);
        assert_eq!(json["cuda_graphs"]["requested"], true);
        assert_eq!(
            json["cuda_graphs"]["capture_allowed_by_serving_profile"],
            true
        );
        assert_eq!(json["cuda_graphs"]["effective_policy_enabled"], true);
        assert_eq!(json["cuda_graphs"]["max_cached_graphs"], 8);
        assert_eq!(json["cuda_graphs"]["stable_paged_metadata"], true);
        assert_eq!(json["cuda_graphs"]["batched_capture_available"], false);
        assert_eq!(json["cuda_graphs"]["restart_required_to_change"], true);
        assert_eq!(
            json["accelerator_runtime"]["schema_id"],
            "kiln.accelerator-runtime-policy.v12"
        );
        assert_eq!(json["accelerator_runtime"]["version"], 12);
        assert_eq!(
            json["accelerator_runtime"]["vulkan_kernel_policy_schema_id"],
            "kiln.vulkan-kernel-policy.v3"
        );
        assert_eq!(
            json["accelerator_runtime"]["vulkan_device_policy_schema_id"],
            "kiln.vulkan-device-policy.v1"
        );
        assert_eq!(
            json["accelerator_runtime"]["kt_api_mode"]["effective"],
            "auto"
        );
        assert_eq!(
            json["accelerator_runtime"]["full_attention_score_budget_mib"]["effective"],
            kiln_model::DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB
        );
        assert!(json["accelerator_runtime"]["vulkan_device_index"]["effective"].is_null());
        assert_eq!(
            json["accelerator_runtime"]["vulkan_validation"]["effective"],
            false
        );
        assert_eq!(
            json["accelerator_runtime"]["cuda_kernel_profile"]["effective"],
            "native_default"
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_graph_cache_max_bytes"]["effective"],
            crate::config::DEFAULT_ROCM_GRAPH_CACHE_MAX_BYTES
        );
        assert!(json["rocm_graphs"].is_null());
        assert_eq!(
            json["rocm_graphs_unavailable_reason"],
            "backend_without_graph_runner"
        );
        assert!(json["rocm_graph_telemetry"].is_null());
        assert_eq!(
            json["rocm_graph_telemetry_unavailable_reason"],
            "backend_without_graph_runner"
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_synchronization_mode"]["effective"],
            "legacy_host_barriers"
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_strided_batched_matmul_mode"]["effective"],
            "auto"
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_bf16_matmul_output_mode"]["effective"],
            "auto"
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_kernel_profile"]["effective"],
            "qualified"
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_graph_mode"]["effective"],
            "lazy_capture_replay"
        );
        assert_eq!(json["generation"]["default_thinking_budget_tokens"], 64);
        assert_eq!(json["generation"]["default_thinking_budget_ms"], 1_500);
        assert_eq!(json["decode_runtime"]["deterministic"]["enabled"], false);
        assert_eq!(json["decode_runtime"]["deterministic"]["source"], "default");
        assert!(json["decode_runtime"]["max_decode_batch"]["configured"].is_null());
        assert_eq!(json["decode_runtime"]["max_decode_batch"]["effective"], 8);
        assert_eq!(
            json["decode_runtime"]["max_decode_batch"]["effective_source"],
            "backend_policy"
        );
        assert_eq!(json["batching"]["actor_active"], false);
        assert_eq!(json["prefix_cache"]["configuration"]["enabled"], true);
        assert!(json["prefix_cache"]["configuration"]["max_blocks"].is_null());
        assert!(json["prefix_cache"]["configuration"]["max_entries"].is_null());
        assert_eq!(json["prefix_cache"]["effective_enabled"], true);
        assert_eq!(json["prefix_cache"]["effective_reason"], "active");
        assert!(
            json["prefix_cache"]["effective_max_blocks"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(json["prefix_cache"]["effective_max_entries"], 0);
        assert_eq!(
            json["batching"]["direct_decode_rendezvous"]["scope"],
            "direct_streaming_greedy_only"
        );
        assert_eq!(
            json["batching"]["direct_decode_rendezvous"]["backend_available"],
            false
        );
        assert_eq!(
            json["batching"]["direct_decode_rendezvous"]["backend_unavailable_reason"],
            "mock_backend"
        );
        assert_eq!(
            json["batching"]["direct_decode_rendezvous"]["actor_active"],
            false
        );
        assert_eq!(
            json["batching"]["direct_decode_rendezvous"]["worker_active"],
            false
        );
        assert_eq!(
            json["batching"]["direct_decode_rendezvous"]["route_available"],
            false
        );
        assert_eq!(
            json["batching"]["configuration"]["mode"]["configured"],
            "auto"
        );
        assert_eq!(
            json["batching"]["configuration"]["mode"]["effective_enabled"],
            false
        );
        assert_eq!(
            json["batching"]["configuration"]["rowwise_decode"]["enabled"],
            false
        );
        assert_eq!(
            json["batching"]["configuration"]["prefix_aware_admission"]["enabled"],
            true
        );
        assert_eq!(
            json["batching"]["configuration"]["prefill_admission_quantum"]["effective"],
            4
        );
        assert_eq!(
            json["batching"]["configuration"]["direct_decode_rendezvous"]["mode"]["configured"],
            "auto"
        );
        assert_eq!(
            json["batching"]["configuration"]["direct_decode_rendezvous"]["mode"]["backend_policy_enabled"],
            false
        );
        assert_eq!(
            json["batching"]["configuration"]["direct_decode_rendezvous"]["mode"]["effective_enabled"],
            false
        );
        assert_eq!(
            json["batching"]["configuration"]["direct_decode_rendezvous"]["max_batch"]["backend_policy"],
            1
        );
        assert_eq!(
            json["batching"]["configuration"]["direct_decode_rendezvous"]["max_batch"]["effective"],
            1
        );
        assert_eq!(
            json["batching"]["configuration"]["direct_decode_rendezvous"]["wait_us"]["effective"],
            0
        );
        assert_eq!(
            json["batching"]["configuration"]["direct_decode_rendezvous"]["mixed_seq_lens"]["effective"],
            false
        );
        assert_eq!(
            json["batching"]["configuration"]["burst_prefill_admission"],
            false
        );
        assert_eq!(
            json["streaming_prefill"]["dispatch"]["configured_mode"],
            "auto"
        );
        assert_eq!(
            json["streaming_prefill"]["dispatch"]["backend_policy"]["policy"],
            "never"
        );
        assert_eq!(
            json["streaming_prefill"]["dispatch"]["effective_source"],
            "backend_policy"
        );
        assert_eq!(
            json["streaming_prefill"]["tile_tokens"]["backend_policy"],
            8192
        );
        assert_eq!(json["streaming_prefill"]["tile_tokens"]["effective"], 8192);
        assert_eq!(json["streaming_prefill"]["immutable_after_startup"], true);
        assert_eq!(
            json["streaming_prefill"]["restart_required_to_change"],
            true
        );
        assert_eq!(json["speculative"]["enabled"], false);
        assert_eq!(json["speculative"]["configured_method"], "off");
        assert_eq!(json["speculative"]["configured_effective_method"], "off");
        assert_eq!(json["speculative"]["serving_effective_method"], "off");
        assert_eq!(
            json["speculative"]["num_speculative_tokens"],
            crate::config::MAX_SPECULATIVE_DRAFT_TOKENS
        );
        assert_eq!(json["speculative"]["draft_layers"], 8);
        assert_eq!(
            json["speculative"]["configured_policy_immutable_after_startup"],
            true
        );
        assert_eq!(json["speculative"]["serving_routable"], false);
        assert_eq!(
            json["speculative"]["serving_unavailable_reason"],
            SPECULATIVE_SERVING_UNAVAILABLE_REASON
        );
        assert_eq!(
            json["speculative"]["draft_token_ceiling"],
            crate::config::MAX_SPECULATIVE_DRAFT_TOKENS
        );
        assert_eq!(json["speculative"]["backend_mtp"]["support"], "unsupported");
        let policy = &json["serving_profile"]["effective_policy"];
        for field in [
            "inference_admission",
            "training_gpu_ownership",
            "adapter_weight_transitions",
            "dynamic_kv_resize",
            "allocator_reclaim",
            "live_graph_capture",
            "vulkan_resident_prefill",
        ] {
            assert_eq!(policy[field], true, "unexpected {field}");
        }
        assert_eq!(policy["exclusive_gpu_behavior"], "writer_priority");
        assert_eq!(json["vram"]["probe_selector"], "none");
        assert_eq!(json["vram"]["physical_capacity_bytes"], 0);
        assert_eq!(json["vram"]["effective_capacity_bytes"], 0);
        assert_eq!(json["vram"]["governor"]["floor_gib"], 1.0);
        assert_eq!(json["vram"]["governor"]["probe_ms"], 500);
        assert_eq!(json["vram"]["governor"]["reclaim_mode_requested"], "off");
        assert_eq!(json["vram"]["governor"]["reclaim_mode_effective"], "off");
        assert_eq!(json["kv_cache"]["autoscaler"]["requested"], true);
        assert_eq!(
            json["kv_cache"]["autoscaler"]["requested_source"],
            "default"
        );
        assert!(json["kv_cache"]["autoscaler"]["force_blocks"].is_null());
        assert_eq!(
            json["kv_cache"]["autoscaler"]["force_blocks_source"],
            "default"
        );
        assert_eq!(json["training"]["runtime_device"], "cpu");
        assert_eq!(json["training"]["model_weight_device"], "cpu");
        assert_eq!(json["training"]["native_training_supported"], false);
        assert_eq!(
            json["training"]["native_training_unavailable_reason"],
            "mock backend does not execute native training"
        );
        assert!(json["training"]["optimizer_support"].is_null());
        assert_eq!(json["training"]["checkpoint_policy"]["mode"], "auto");
        assert_eq!(
            json["training"]["checkpoint_boundary_policy"],
            expected_checkpoint_boundary_policy
        );
        assert!(json["vram"]["live"]["raw_observations"]["driver_total_bytes"].is_null());
    }

    fn training_capabilities_for(
        backend: &str,
        device: kiln_tensor::Device,
        precision: kiln_model::backend::TrainingPrecisionPolicy,
    ) -> kiln_model::BackendTrainingCapabilities {
        kiln_model::BackendTrainingCapabilities {
            hooks: kiln_model::backend::TrainingCapabilities::portable(),
            precision,
            optimizer: kiln_model::TrainingOptimizerSupport::for_backend(backend, device),
            server_dispatch: kiln_model::ServerTrainingDispatchPolicy::for_backend(backend, device),
            acceleration_profile: kiln_model::TrainingAccelerationProfilePolicy::for_backend(
                backend, device,
            ),
        }
    }

    fn supported_workloads() -> [(TrainingWorkload, Option<String>); 4] {
        [
            (TrainingWorkload::Sft, None),
            (TrainingWorkload::Grpo, None),
            (TrainingWorkload::Opd, None),
            (
                TrainingWorkload::DistillRefresh,
                Some(crate::state::DISTILL_REFRESH_COMPOSITE_ADMISSION_UNAVAILABLE.to_string()),
            ),
        ]
    }

    fn unavailable_workloads(reason: &str) -> [(TrainingWorkload, Option<String>); 4] {
        [
            (TrainingWorkload::Sft, Some(reason.to_string())),
            (TrainingWorkload::Grpo, Some(reason.to_string())),
            (TrainingWorkload::Opd, Some(reason.to_string())),
            (TrainingWorkload::DistillRefresh, Some(reason.to_string())),
        ]
    }

    #[test]
    fn optimizer_support_separates_implementation_tuple_and_workload_admission() {
        let device = kiln_tensor::Device::Metal(0);
        let workloads = supported_workloads();
        let support = build_training_optimizer_support_from_capabilities(
            "metal",
            device,
            kiln_tensor::DType::BF16,
            training_capabilities_for(
                "metal",
                device,
                kiln_model::backend::TrainingPrecisionPolicy::metal(),
            ),
            64,
            None,
            &workloads,
        );
        let json = serde_json::to_value(support).unwrap();

        assert_eq!(json["schema"]["id"], "kiln.training-optimizer-support");
        assert_eq!(json["schema"]["version"], 1);
        assert_eq!(json["backend"], "metal");
        assert_eq!(json["device"], "metal:0");
        assert_eq!(json["base_weight_dtype"], "bf16");
        assert_eq!(json["resolved_lora_parameter_dtype"], "bf16");
        assert_eq!(json["immutable_after_startup"], true);
        assert_eq!(
            json["rounding_modes"],
            serde_json::json!(["round_to_nearest"])
        );
        assert_eq!(
            json["optimizer_tuple_kinds"],
            serde_json::json!(["muon", "adam_w"])
        );
        assert_eq!(json["workloads"][0]["workload"], "sft");
        assert_eq!(json["workloads"][0]["supported"], true);
        assert_eq!(
            json["workloads"][0]["allowed_optimizer_kinds"],
            serde_json::json!(["muon", "adam_w"])
        );
        assert_eq!(json["workloads"][3]["workload"], "distill_refresh");
        assert_eq!(json["workloads"][3]["supported"], false);
        assert_eq!(
            json["workloads"][3]["unavailable_reason"],
            crate::state::DISTILL_REFRESH_COMPOSITE_ADMISSION_UNAVAILABLE
        );
        assert_eq!(
            json["workloads"][3]["allowed_optimizer_kinds"],
            serde_json::json!([])
        );

        let optimizers = json["optimizers"].as_array().unwrap();
        let muon = optimizers
            .iter()
            .find(|item| item["kind"] == "muon")
            .unwrap();
        assert_eq!(muon["backend_implementation"]["supported"], true);
        assert_eq!(
            muon["backend_implementation"]["route"],
            "native_device_hook"
        );
        assert_eq!(muon["backend_implementation"]["native_device_hook"], true);
        assert_eq!(muon["optimizer_tuple"]["supported"], true);
        assert_eq!(muon["optimizer_tuple"]["lora_rank"]["minimum"], 2);
        assert_eq!(muon["optimizer_tuple"]["lora_rank"]["maximum"], 32);
        assert_eq!(muon["optimizer_tuple"]["lora_rank"]["backend_maximum"], 32);
        assert_eq!(muon["optimizer_tuple"]["lora_rank"]["model_maximum"], 64);
        assert_eq!(
            muon["optimizer_tuple"]["lora_rank"]["live_memory_admission_required"],
            true
        );
        let sgd = optimizers
            .iter()
            .find(|item| item["kind"] == "sgd")
            .unwrap();
        assert_eq!(sgd["backend_implementation"]["supported"], false);
        assert_eq!(sgd["backend_implementation"]["route"], "unavailable");
        assert_eq!(
            sgd["backend_implementation"]["parameter_dtypes"],
            serde_json::json!([])
        );
        assert_eq!(sgd["optimizer_tuple"]["supported"], false);
        assert!(sgd["optimizer_tuple"]["unavailable_reason"].is_string());
        let adamw = optimizers
            .iter()
            .find(|item| item["kind"] == "adam_w")
            .unwrap();
        assert_eq!(adamw["optimizer_tuple"]["lora_rank"]["maximum"], 64);
        assert!(adamw["optimizer_tuple"]["lora_rank"]["backend_maximum"].is_null());
        assert_eq!(adamw["optimizer_tuple"]["lora_rank"]["model_maximum"], 64);
    }

    #[test]
    fn optimizer_implementation_does_not_inherit_product_rounding_policy() {
        static STOCHASTIC_ONLY: &[kiln_model::TrainingOptimizerRounding] =
            &[kiln_model::TrainingOptimizerRounding::Stochastic];

        let device = kiln_tensor::Device::Metal(0);
        let workloads = supported_workloads();
        let mut capabilities = training_capabilities_for(
            "metal",
            device,
            kiln_model::backend::TrainingPrecisionPolicy::metal(),
        );
        capabilities.optimizer.rounding_modes = STOCHASTIC_ONLY;
        let json = serde_json::to_value(build_training_optimizer_support_from_capabilities(
            "metal",
            device,
            kiln_tensor::DType::BF16,
            capabilities,
            64,
            None,
            &workloads,
        ))
        .unwrap();

        assert_eq!(
            json["rounding_modes"],
            serde_json::json!(["round_to_nearest"])
        );
        assert_eq!(
            json["backend_implementation_rounding_modes"],
            serde_json::json!(["stochastic"])
        );
        for optimizer in json["optimizers"].as_array().unwrap() {
            if optimizer["kind"] == "sgd" {
                assert_eq!(optimizer["backend_implementation"]["supported"], false);
            } else {
                assert_eq!(optimizer["backend_implementation"]["supported"], true);
            }
            assert_eq!(optimizer["optimizer_tuple"]["supported"], false);
        }
        for workload in json["workloads"].as_array().unwrap() {
            assert_eq!(workload["supported"], false);
            assert!(
                workload["allowed_optimizer_kinds"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn optimizer_support_keeps_implementation_facts_when_server_training_is_unavailable() {
        let device = kiln_tensor::Device::Vulkan(0);
        let workloads = unavailable_workloads("resident model weights are CPU-hosted");
        let support = build_training_optimizer_support_from_capabilities(
            "vulkan",
            device,
            kiln_tensor::DType::BF16,
            training_capabilities_for(
                "vulkan",
                device,
                kiln_model::backend::TrainingPrecisionPolicy::vulkan(),
            ),
            64,
            None,
            &workloads,
        );
        let json = serde_json::to_value(support).unwrap();

        assert_eq!(json["resolved_lora_parameter_dtype"], "f32");
        assert_eq!(
            json["optimizer_tuple_kinds"],
            serde_json::json!(["muon", "adam_w", "sgd"])
        );
        for workload in json["workloads"].as_array().unwrap() {
            assert_eq!(workload["supported"], false);
            assert_eq!(workload["allowed_optimizer_kinds"], serde_json::json!([]));
            assert_eq!(
                workload["unavailable_reason"],
                "resident model weights are CPU-hosted"
            );
        }
        for optimizer in json["optimizers"].as_array().unwrap() {
            assert_eq!(optimizer["backend_implementation"]["supported"], true);
            assert_eq!(
                optimizer["backend_implementation"]["native_device_hook"],
                true
            );
            assert_eq!(optimizer["optimizer_tuple"]["supported"], true);
        }
    }

    #[test]
    fn cpu_optimizer_execution_is_labeled_as_portable_not_native() {
        let device = kiln_tensor::Device::Cpu;
        let workloads = unavailable_workloads("cpu tape training is unavailable");
        let support = build_training_optimizer_support_from_capabilities(
            "cpu",
            device,
            kiln_tensor::DType::F32,
            training_capabilities_for(
                "cpu",
                device,
                kiln_model::backend::TrainingPrecisionPolicy::portable(),
            ),
            64,
            None,
            &workloads,
        );
        let json = serde_json::to_value(support).unwrap();

        assert_eq!(
            json["optimizer_tuple_kinds"],
            serde_json::json!(["muon", "adam_w", "sgd"])
        );
        for workload in json["workloads"].as_array().unwrap() {
            assert_eq!(workload["supported"], false);
            assert_eq!(workload["allowed_optimizer_kinds"], serde_json::json!([]));
        }
        for optimizer in json["optimizers"].as_array().unwrap() {
            assert_eq!(optimizer["backend_implementation"]["supported"], true);
            assert_eq!(
                optimizer["backend_implementation"]["route"],
                "portable_reference"
            );
            assert_eq!(
                optimizer["backend_implementation"]["native_device_hook"],
                false
            );
            assert_eq!(optimizer["optimizer_tuple"]["supported"], true);
        }
        let optimizers = json["optimizers"].as_array().unwrap();
        let muon = optimizers
            .iter()
            .find(|item| item["kind"] == "muon")
            .unwrap();
        assert_eq!(muon["optimizer_tuple"]["lora_rank"]["minimum"], 2);
        assert_eq!(muon["optimizer_tuple"]["lora_rank"]["maximum"], 64);
        assert!(muon["optimizer_tuple"]["lora_rank"]["backend_maximum"].is_null());
        for kind in ["adam_w", "sgd"] {
            let optimizer = optimizers.iter().find(|item| item["kind"] == kind).unwrap();
            assert_eq!(optimizer["optimizer_tuple"]["lora_rank"]["minimum"], 1);
            assert_eq!(optimizer["optimizer_tuple"]["lora_rank"]["maximum"], 64);
        }
    }

    #[test]
    fn speculative_snapshot_distinguishes_configured_policy_from_serving_policy() {
        let mut state = make_test_state();
        state.speculative_config = crate::config::SpeculativeDecodingConfig {
            enabled: true,
            method: SpecMethod::Mtp,
            num_speculative_tokens: 4,
            draft_layers: 6,
        };
        state.speculative_runtime_policy =
            crate::state::SpeculativeRuntimePolicy::new(kiln_model::Support::NativeWithConstraints);

        let json = serde_json::to_value(build_speculative_config(&state)).unwrap();
        assert_eq!(json["enabled"], true);
        assert_eq!(json["configured_method"], "mtp");
        assert_eq!(json["configured_effective_method"], "mtp");
        assert_eq!(json["serving_effective_method"], "off");
        assert_eq!(json["num_speculative_tokens"], 4);
        assert_eq!(json["draft_layers"], 6);
        assert_eq!(json["configured_policy_immutable_after_startup"], true);
        assert_eq!(json["serving_routable"], false);
        assert_eq!(
            json["serving_unavailable_reason"],
            SPECULATIVE_SERVING_UNAVAILABLE_REASON
        );
        assert_eq!(json["draft_token_ceiling"], 4);

        assert_eq!(json["backend_mtp"]["support"], "native_with_constraints");
        assert_eq!(json["backend_mtp"]["native"], true);
        assert!(json.get("routing").is_none());
    }

    #[test]
    fn speculative_snapshot_preserves_configured_and_effective_method() {
        let mut state = make_test_state();
        state.speculative_config = crate::config::SpeculativeDecodingConfig {
            enabled: true,
            method: SpecMethod::Off,
            num_speculative_tokens: 4,
            draft_layers: 4,
        };

        let json = serde_json::to_value(build_speculative_config(&state)).unwrap();
        assert_eq!(json["enabled"], true);
        assert_eq!(json["configured_method"], "off");
        assert_eq!(json["configured_effective_method"], "skip_layer");
        assert_eq!(json["serving_effective_method"], "off");
        assert_eq!(json["serving_routable"], false);
    }

    #[test]
    fn speculative_backend_support_names_are_stable_and_exhaustive() {
        use kiln_model::Support;

        for (support, expected) in [
            (Support::Native, "native"),
            (Support::NativeWithConstraints, "native_with_constraints"),
            (Support::HostFallbackAllowed, "host_fallback_allowed"),
            (Support::Declined, "declined"),
            (Support::Unsupported, "unsupported"),
            (Support::DisabledByEnv, "disabled_by_env"),
            (Support::RequiresFeature, "requires_feature"),
        ] {
            assert_eq!(support_name(support), expected);
        }
    }

    #[tokio::test]
    async fn disabled_checkpointing_reports_zero_effective_segments() {
        let mut state = make_test_state();
        state.training_runtime = kiln_train::TrainingRuntimeContext::new_for_device(
            kiln_tensor::Device::Cpu,
            state.vram_info,
            kiln_train::GradientCheckpointPolicy::from_parts(Some(8), true).unwrap(),
        );
        let response = routes()
            .with_state(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["training"]["checkpoint_policy"]["mode"], "disabled");
        assert_eq!(json["training"]["checkpoint_policy"]["segments"], 8);
        assert_eq!(json["training"]["checkpoint_segments"], 0);
        assert_eq!(json["training"]["checkpoint_segments_source"], "disabled");
        assert_eq!(json["training"]["checkpointing_enabled"], false);
    }

    #[test]
    fn vram_diagnostics_keep_capacity_live_and_raw_observations_distinct() {
        use kiln_memory::vram::{
            GpuVramInfo, LinuxDrmVendor, MemorySnapshot, MemorySnapshotObservations,
            MemoryTierSnapshot, VramCapacityResolution, VramProbeSelector, VramSource,
        };

        let gib = 1024 * 1024 * 1024;
        let mut state = make_test_state();
        let physical = GpuVramInfo {
            total_bytes: 24 * gib,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
        };
        state.vram_info = physical;
        state.vram_capacity_resolution = VramCapacityResolution {
            physical,
            requested_bytes: Some(48 * gib),
            effective: physical,
            clamped: true,
        };
        state.vram_probe_selector = VramProbeSelector::LinuxDrm {
            index: 1,
            vendor: Some(LinuxDrmVendor::Amd),
        };
        state.memory_config.floor_gb = 2.0;
        state.memory_config.probe_ms = 750;
        state.memory_config.reclaim_mode = crate::config::MemoryReclaimModeSetting::new(
            kiln_memory::MemoryReclaimMode::Automatic,
            ConfigValueSource::ConfigFile,
        );
        let live = MemorySnapshot {
            total_bytes: 24 * gib,
            used_bytes: 10 * gib,
            free_bytes: 14 * gib,
            source: VramSource::LinuxDrmSysfsUnified,
            unified: true,
            observations: MemorySnapshotObservations {
                probe_failed: false,
                driver_total_bytes: Some(120 * gib),
                driver_used_bytes: Some(4 * gib),
                driver_free_bytes: Some(116 * gib),
                driver_vram_total_bytes: Some(96 * gib),
                driver_vram_used_bytes: Some(3 * gib),
                driver_gtt_total_bytes: Some(24 * gib),
                driver_gtt_used_bytes: Some(gib),
                host_total_bytes: Some(32 * gib),
                host_available_bytes: Some(22 * gib),
                cgroup_limit_bytes: Some(28 * gib),
                cgroup_high_bytes: Some(26 * gib),
                cgroup_current_bytes: Some(6 * gib),
                cgroup_remaining_bytes: Some(22 * gib),
                unified_reserve_bytes: Some(8 * gib),
                host_backed: Some(MemoryTierSnapshot {
                    total_bytes: 16 * gib,
                    used_bytes: 6 * gib,
                    free_bytes: 10 * gib,
                }),
            },
        };

        let json = serde_json::to_value(build_vram_config(
            &state,
            CachedMemoryGovernorObservation {
                snapshot: live,
                available_bytes: 12 * gib,
                soft_reserved_bytes: 0,
                pressure: kiln_memory::MemoryPressure::Moderate,
                sample_status: kiln_memory::CachedSampleStatus {
                    age: std::time::Duration::from_millis(125),
                    max_age: std::time::Duration::from_secs(5),
                    stale: false,
                    sampler_required: true,
                    sampler_running: true,
                    healthy: true,
                },
                automatic_monitor_enabled: false,
                automatic_reclaim: kiln_memory::AutomaticReclaimStats::default(),
            },
        ))
        .unwrap();
        assert_eq!(json["probe_selector"], "linux-drm:amd:1");
        assert_eq!(json["physical_capacity_bytes"], 24 * gib);
        assert_eq!(json["configured_capacity_bytes"], 48 * gib);
        assert_eq!(json["effective_capacity_bytes"], 24 * gib);
        assert_eq!(json["configured_capacity_clamped"], true);
        assert_eq!(json["live"]["available_bytes"], 14 * gib);
        assert_eq!(json["live"]["usable_after_governor_floor_bytes"], 12 * gib);
        assert_eq!(
            json["live"]["raw_observations"]["driver_total_bytes"],
            120 * gib
        );
        assert_eq!(
            json["live"]["raw_observations"]["host_available_bytes"],
            22 * gib
        );
        assert_eq!(json["live"]["sample_age_ms"], 125);
        assert_eq!(json["live"]["sampler_healthy"], true);
        assert_eq!(
            json["live"]["raw_observations"]["host_backed"]["free_bytes"],
            10 * gib
        );
        assert_eq!(json["governor"]["reclaim_mode_requested"], "automatic");
        assert_eq!(json["governor"]["reclaim_mode_effective"], "off");
        assert_eq!(
            json["governor"]["reclaim_disabled_by_serving_profile"],
            true
        );
        assert_eq!(json["governor"]["capacity_limit_bytes"], 24 * gib);
        assert_eq!(json["governor"]["reclaim_mode_source"], "config_file");
    }
}
