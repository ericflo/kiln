use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;

use crate::config::{
    BatchingRuntimeConfig, ConfigValueSource, DecodeRuntimeConfig, ServingProfileDiagnostics,
    SpecMethod,
};
use crate::memory_observability::CachedMemoryGovernorObservation;
use crate::state::{AppState, ModelBackend};

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const SPECULATIVE_SERVING_UNAVAILABLE_REASON: &str =
    "pending_cancel_safe_local_accelerator_qualification";

fn gib(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB
}

#[derive(Serialize)]
struct ConfigResponse {
    serving_profile: ServingProfileDiagnostics,
    decode_runtime: DecodeRuntimeConfig,
    batching: BatchingConfigResponse,
    speculative: SpeculativeConfig,
    vram: VramConfig,
    kv_cache: KvCacheConfig,
    training: TrainingConfig,
    memory_budget: MemoryBudgetConfig,
    generation: GenerationConfig,
}

#[derive(Serialize)]
struct BatchingConfigResponse {
    configuration: BatchingRuntimeConfig,
    actor_active: bool,
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
}

#[derive(Serialize)]
struct TrainingConfig {
    runtime_device: Option<String>,
    model_weight_device: String,
    native_training_supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_training_unavailable_reason: Option<String>,
    checkpoint_policy: kiln_train::GradientCheckpointPolicy,
    checkpoint_segments: usize,
    checkpoint_segments_source: &'static str,
    checkpointing_enabled: bool,
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
    let native_training_unavailable_reason =
        if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
            Some("mock backend does not execute native training".to_string())
        } else {
            state
                .training_runtime
                .resolve_device_for_weights(state.model_weight_device)
                .err()
                .map(|error| error.to_string())
        };

    let b = &state.memory_budget;

    Json(ConfigResponse {
        serving_profile: state.serving_profile.diagnostics(),
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
        },
        speculative: build_speculative_config(&state),
        vram: build_vram_config(&state, memory_observation),
        kv_cache: KvCacheConfig {
            num_blocks,
            num_blocks_source,
            fp8_enabled: match state.backend.as_ref() {
                ModelBackend::Real { paged_cache, .. } => paged_cache.is_fp8(),
                ModelBackend::Mock { .. } => false,
            },
        },
        training: TrainingConfig {
            runtime_device: state
                .training_runtime
                .runtime_device()
                .map(|device| device.short_name().to_string()),
            model_weight_device: state.model_weight_device.short_name().to_string(),
            native_training_supported: native_training_unavailable_reason.is_none(),
            native_training_unavailable_reason,
            checkpoint_policy: state.training_runtime.gradient_checkpoint_policy(),
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

    #[tokio::test]
    async fn config_reports_profile_provenance_and_every_effective_policy() {
        let mut state = make_test_state();
        state.serving_profile = crate::config::ServingProfileSetting::new(
            crate::config::ServingProfile::Experimental,
            crate::config::ConfigValueSource::Environment,
        );
        state.default_thinking_budget_tokens = Some(64);
        state.default_thinking_budget_ms = Some(1_500);
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

        assert_eq!(json["serving_profile"]["profile"], "experimental");
        assert_eq!(json["serving_profile"]["source"], "environment");
        assert_eq!(json["serving_profile"]["immutable_after_startup"], true);
        assert_eq!(json["serving_profile"]["request_overrides_allowed"], false);
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
            json["batching"]["configuration"]["burst_prefill_admission"],
            false
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
        assert_eq!(json["training"]["runtime_device"], "cpu");
        assert_eq!(json["training"]["model_weight_device"], "cpu");
        assert_eq!(json["training"]["native_training_supported"], false);
        assert_eq!(
            json["training"]["native_training_unavailable_reason"],
            "mock backend does not execute native training"
        );
        assert_eq!(json["training"]["checkpoint_policy"]["mode"], "auto");
        assert!(json["vram"]["live"]["raw_observations"]["driver_total_bytes"].is_null());
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

    #[test]
    fn speculative_config_api_reports_only_immutable_policy_and_backend_facts() {
        let source = include_str!("config.rs");
        let section = source
            .split("fn build_speculative_config")
            .nth(1)
            .unwrap()
            .split("fn build_vram_config")
            .next()
            .unwrap();

        assert!(section.contains("state.speculative_config"));
        assert!(section.contains("state.speculative_runtime_policy"));
        assert!(section.contains("serving_effective_method: SpecMethod::Off"));
        assert!(section.contains("serving_routable: false"));
        assert!(section.contains("SPECULATIVE_SERVING_UNAVAILABLE_REASON"));
        assert!(
            section.contains("draft_token_ceiling: crate::config::MAX_SPECULATIVE_DRAFT_TOKENS")
        );
        assert!(!section.contains("runtime.decode_policy"));
        assert!(!section.contains("mtp_max_prompt_tokens"));
        assert!(!section.contains("long_prompt_skip_layer"));
        assert!(!section.contains("std::env"));
        assert!(!section.contains("backend_health_handle()"));
        assert!(!section.contains("available_permits()"));
        assert!(!section.contains("try_global_cached_available_bytes()"));
        assert!(!section.contains("runner"));
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
