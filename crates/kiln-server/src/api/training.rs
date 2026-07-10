//! Training API endpoints — pure Rust, in-process LoRA training.
//!
//! Training requests are enqueued in a FIFO queue and executed sequentially
//! by a background worker. This prevents GPU memory conflicts between
//! concurrent training jobs.

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path as AxumPath, State},
    routing::{delete, get, post},
};

use kiln_train::{
    DistillMergeRequest, DistillPumpRequest, DistillRefreshRequest, DistillSelfRequest, GrpoGroup,
    GrpoRequest, OpdRequest, SftRequest, TrainingResponse, TrainingState, TrainingStatus,
};
use serde::Serialize;

use std::{
    path::{Path, PathBuf},
    sync::atomic::Ordering,
};

use crate::error::ApiError;
use crate::metrics::{TrainingMetricStatus, TrainingMetricType};
use crate::state::{AppState, ModelBackend, TrainingJobInfo, TrainingJobType};
use crate::training_preflight::{
    self, EstimateOptions, WeightResidency, auto_fit_checkpoint_segments,
    available_for_training_bytes, estimate_step_working_set_with_options,
    estimate_vk_native_recompute_working_set, format_oom_message_with_source,
};
use crate::training_queue::{QueueEntry, QueuedJob};
use kiln_memory::vram::VramSource;

struct GrpoSubmissionStats {
    num_groups: Option<usize>,
    total_completions: Option<usize>,
    max_seq_len: usize,
    streaming_dataset: bool,
}

struct SftSubmissionStats {
    num_examples: usize,
    max_seq_len: usize,
    max_supervised_tokens: usize,
    streaming_dataset: bool,
}

struct PreflightAdmission {
    reserved_bytes: u64,
    checkpoint_segments: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct TrainingMemoryAvailability {
    bytes: u64,
    live_bytes: u64,
    allocator_bytes: Option<u64>,
    reclaimable_kv_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct TrainingActivationEstimate {
    bytes_per_elem: usize,
    streaming_gdn_tile_tokens: Option<usize>,
}

fn model_dtype_bytes(dtype: kiln_core::config::DType) -> usize {
    match dtype {
        kiln_core::config::DType::BF16 | kiln_core::config::DType::FP16 => 2,
        kiln_core::config::DType::FP32 => 4,
    }
}

const GDN_TAPE_EFFECTIVE_BYTES_PER_ELEM: usize = 10;

fn training_activation_bytes_per_elem(
    base: usize,
    uses_f32_activations: bool,
    has_linear_attention: bool,
) -> usize {
    if uses_f32_activations {
        return 4;
    }
    if has_linear_attention {
        base.max(GDN_TAPE_EFFECTIVE_BYTES_PER_ELEM)
    } else {
        base
    }
}

fn training_activation_estimate_for_runtime_device(
    base: usize,
    uses_f32_activations: bool,
    has_linear_attention: bool,
    runtime_device: kiln_tensor::Device,
    max_seq_len: usize,
) -> TrainingActivationEstimate {
    // Mirror kiln-train's GDN tape sizing. The server stamps resolved
    // checkpoint segments at submission time, so an optimistic bf16-only
    // preflight would bypass the trainer's more conservative auto-tuner.
    let bytes_per_elem =
        training_activation_bytes_per_elem(base, uses_f32_activations, has_linear_attention);
    let streaming_gdn_tile_tokens = if has_linear_attention {
        kiln_model::forward::streaming_prefill_enabled_for(&runtime_device, max_seq_len)
            .then(|| kiln_model::forward::tape_streaming_tile_tokens_for(&runtime_device))
            .filter(|&tile| tile > 0 && tile < max_seq_len)
    } else {
        None
    };
    TrainingActivationEstimate {
        bytes_per_elem,
        streaming_gdn_tile_tokens,
    }
}

fn training_activation_estimate_for_state(
    state: &AppState,
    max_seq_len: usize,
) -> TrainingActivationEstimate {
    let base = model_dtype_bytes(state.model_config.dtype);
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return TrainingActivationEstimate {
            bytes_per_elem: base,
            streaming_gdn_tile_tokens: None,
        };
    };
    let Ok(runner) = runner.read() else {
        tracing::warn!(
            "model runner lock poisoned while sizing training preflight; using model dtype width"
        );
        return TrainingActivationEstimate {
            bytes_per_elem: base,
            streaming_gdn_tile_tokens: None,
        };
    };
    let capabilities = runner.backend_capabilities();
    let has_linear_attention = state.model_config.num_full_attention_layers
        < state.model_config.num_layers
        || runner
            .weights
            .linear_attention_layers_in_prefix(runner.config.num_layers)
            > 0;
    training_activation_estimate_for_runtime_device(
        base,
        runner
            .training_precision_policy()
            .uses_f32_activations_for_mixed_base_weights(),
        has_linear_attention,
        capabilities.device,
        max_seq_len,
    )
}

fn checkpoint_env_override_present() -> bool {
    std::env::var("KILN_GRAD_CHECKPOINT_SEGMENTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some()
        || std::env::var("KILN_NO_GRAD_CHECKPOINT")
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

fn effective_checkpoint_segments(config: kiln_train::CheckpointConfig) -> usize {
    if config.enabled {
        config.num_segments
    } else {
        1
    }
}

fn auto_mode_reservation_segments(
    cfg: &kiln_core::config::ModelConfig,
    fit_segments: usize,
) -> usize {
    // Auto mode resolves the real checkpoint plan per example/group in
    // kiln-train. Queue admission only needs a conservative reservation for
    // the largest submitted row, so reserve the most checkpointed shape the
    // shared trainer can use instead of pinning every row to that plan.
    cfg.num_layers.max(fit_segments).max(1)
}

fn combine_training_available_bytes(
    live_bytes: u64,
    allocator_bytes: Option<u64>,
    reclaimable_kv_bytes: u64,
    total_bytes: u64,
    floor_bytes: u64,
    vram_source: VramSource,
) -> u64 {
    let base = if matches!(
        vram_source,
        VramSource::LinuxDrmSysfsUnified | VramSource::AppleSilicon
    ) {
        live_bytes
    } else {
        allocator_bytes.map_or(live_bytes, |bytes| live_bytes.max(bytes))
    };
    let with_reclaimable = base.saturating_add(reclaimable_kv_bytes);
    if total_bytes == 0 {
        with_reclaimable
    } else {
        with_reclaimable.min(total_bytes.saturating_sub(floor_bytes))
    }
}

fn dynamic_training_availability(
    state: &AppState,
    vram: &kiln_memory::vram::GpuVramInfo,
    live_available: u64,
) -> TrainingMemoryAvailability {
    let governor = kiln_memory::MemoryGovernor::global();
    let floor_bytes = governor.config().floor_bytes;
    let soft_reserved = governor.soft_reserved_bytes();
    let live_bytes = live_available.saturating_sub(soft_reserved);
    let mut allocator_bytes = None;
    let mut reclaimable_kv_bytes = 0u64;

    if let ModelBackend::Real {
        runner,
        paged_cache,
        batching_engine,
        ..
    } = state.backend.as_ref()
        && let Ok(runner) = runner.read()
    {
        let capabilities = runner.backend_capabilities();
        let device = runner.weights.embed_tokens.device();
        allocator_bytes = crate::device_memory::allocator_safe_available_bytes_with_soft_reserved(
            capabilities.storage.gpu_allocator_memory_probe_policy,
            &device,
            floor_bytes,
            soft_reserved,
        );

        let cache_device_matches_model = paged_cache
            .device()
            .is_some_and(|cache_device| cache_device == device);
        if capabilities.storage.kv_cache_device_memory_pressure
            && batching_engine.is_some()
            && cache_device_matches_model
        {
            let current_blocks = paged_cache.num_blocks();
            let bytes_per_block = paged_cache.bytes_per_block() as u64;
            reclaimable_kv_bytes = current_blocks.saturating_sub(1) as u64 * bytes_per_block;
        }
    }

    let bytes = combine_training_available_bytes(
        live_bytes,
        allocator_bytes,
        reclaimable_kv_bytes,
        vram.total_bytes,
        floor_bytes,
        vram.source,
    );
    TrainingMemoryAvailability {
        bytes,
        live_bytes,
        allocator_bytes,
        reclaimable_kv_bytes,
    }
}

fn validate_grpo_jsonl_submission_head(
    dataset_path: &str,
    tokenizer: Option<&kiln_core::tokenizer::KilnTokenizer>,
) -> Result<GrpoSubmissionStats, ApiError> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(dataset_path).map_err(|e| {
        ApiError::training_invalid_request(format!(
            "failed to open GRPO dataset_path '{dataset_path}': {e}"
        ))
    })?;
    let reader = BufReader::new(file);

    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| {
            ApiError::training_invalid_request(format!(
                "failed to read GRPO dataset_path '{dataset_path}' line {}: {e}",
                idx + 1
            ))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let group: GrpoGroup = serde_json::from_str(trimmed).map_err(|e| {
            ApiError::training_invalid_request(format!(
                "invalid GRPO JSONL group at line {} in '{dataset_path}': {e}",
                idx + 1
            ))
        })?;
        if group.completions.is_empty() {
            return Err(ApiError::training_invalid_request(format!(
                "GRPO JSONL first non-empty group at line {} in '{dataset_path}' has no completions",
                idx + 1
            )));
        }
        return Ok(GrpoSubmissionStats {
            num_groups: None,
            total_completions: None,
            max_seq_len: training_preflight::approximate_max_seq_len_grpo_group(&group, tokenizer),
            streaming_dataset: true,
        });
    }

    Err(ApiError::training_invalid_request(format!(
        "GRPO dataset_path '{dataset_path}' contains no groups"
    )))
}

/// Estimate the per-step working set against the corrected memory
/// budget, and reject the submission with HTTP 413 if it cannot fit.
///
/// `max_seq_len` is approximated upstream by the request-specific
/// helper (`approximate_max_seq_len_sft` / `_grpo`) so this helper
/// stays SFT/GRPO-agnostic.
/// Validate a training submission fits in VRAM AND return the estimated per-step
/// working-set bytes (so the caller can stash it on the queue entry and hold a
/// governor reservation across the job — #24), plus the resolved dynamic
/// checkpoint segment count. Returns zero bytes and no segment override when no
/// estimate is available (no memory signal / zero-length); callers skip the
/// reservation and let the trainer fall back to its own guard.
fn enforce_training_preflight(
    state: &AppState,
    max_seq_len: usize,
    mut options: EstimateOptions,
    lora_rank: usize,
    vk_native_recompute: bool,
) -> Result<PreflightAdmission, ApiError> {
    if max_seq_len == 0 {
        return Ok(PreflightAdmission {
            reserved_bytes: 0,
            checkpoint_segments: None,
        });
    }
    let activation_estimate = training_activation_estimate_for_state(state, max_seq_len);
    if options.activation_bytes_per_elem.is_none() {
        options.activation_bytes_per_elem = Some(activation_estimate.bytes_per_elem);
    }
    if options.streaming_gdn_tile_tokens.is_none() {
        options.streaming_gdn_tile_tokens = activation_estimate.streaming_gdn_tile_tokens;
    }
    let vram = kiln_memory::vram::detect_vram();
    let live_available = available_for_training_bytes(&vram);
    if live_available == u64::MAX {
        // No memory signal at all — let the trainer be the line of
        // defense. Better than rejecting every submission on machines
        // where detection is misconfigured.
        return Ok(PreflightAdmission {
            reserved_bytes: 0,
            checkpoint_segments: None,
        });
    }
    let available = dynamic_training_availability(state, &vram, live_available);
    if available.bytes > live_available {
        tracing::info!(
            live_available_gb = available.live_bytes as f64 / 1e9,
            effective_available_gb = available.bytes as f64 / 1e9,
            allocator_available_gb = available.allocator_bytes.map(|bytes| bytes as f64 / 1e9),
            reclaimable_kv_gb = available.reclaimable_kv_bytes as f64 / 1e9,
            "training preflight using dynamic memory availability"
        );
    }
    let checkpoint_env_override = checkpoint_env_override_present();
    let env_segments = if vk_native_recompute || checkpoint_env_override {
        effective_checkpoint_segments(kiln_train::CheckpointConfig::from_env(
            state.model_config.num_layers,
        ))
    } else {
        1
    };
    // Until the resident registry (Phase 1.2-1.4) lands, weights on
    // Vulkan APUs live in BOTH candle CPU storage and VulkanBuffer
    // caches — same physical RAM on unified memory. The estimator
    // must reflect that or the preflight will accept payloads that
    // ultimately exhaust the host. Once Phase 1.2 is deployed,
    // switch this to WeightResidency::SingleCopy.
    let residency = WeightResidency::for_vram_source(vram.source);
    // Whether the available budget already accounts for the loaded
    // model. On unified APUs we read MemAvailable at submission time
    // so the model is already deducted; including base weights again
    // double-counts and over-rejects every job. On discrete GPUs the
    // available number is a static pre-deduction reserve, so weights
    // ARE still pending and must be counted. `KILN_GPU_MEMORY_GB` is
    // an explicit operator override for this already-loaded server
    // process, so treat it as a post-load budget as well.
    let weights_already_resident = matches!(
        vram.source,
        kiln_memory::vram::VramSource::LinuxDrmSysfsUnified
            | kiln_memory::vram::VramSource::AppleSilicon
            | kiln_memory::vram::VramSource::EnvOverride
    );
    let (num_segments, estimate) = if vk_native_recompute {
        (
            env_segments,
            estimate_vk_native_recompute_working_set(
                &state.model_config,
                max_seq_len,
                lora_rank,
                residency,
                weights_already_resident,
            ),
        )
    } else if checkpoint_env_override {
        (
            env_segments,
            estimate_step_working_set_with_options(
                &state.model_config,
                max_seq_len,
                lora_rank,
                env_segments,
                residency,
                weights_already_resident,
                options,
            ),
        )
    } else {
        auto_fit_checkpoint_segments(
            &state.model_config,
            max_seq_len,
            lora_rank,
            state.model_config.num_layers,
            residency,
            weights_already_resident,
            options,
            available.bytes,
        )
    };
    if estimate.total_bytes > available.bytes {
        let msg = format_oom_message_with_source(
            &estimate,
            available.bytes,
            lora_rank,
            num_segments,
            Some(vram.source),
        );
        return Err(ApiError::training_will_not_fit(msg));
    }
    let checkpoint_segments = if vk_native_recompute || checkpoint_env_override {
        Some(num_segments)
    } else {
        None
    };
    let reserved_bytes = if vk_native_recompute || checkpoint_env_override {
        estimate.total_bytes
    } else {
        let reservation_segments =
            auto_mode_reservation_segments(&state.model_config, num_segments);
        if reservation_segments == num_segments {
            estimate.total_bytes
        } else {
            estimate_step_working_set_with_options(
                &state.model_config,
                max_seq_len,
                lora_rank,
                reservation_segments,
                residency,
                weights_already_resident,
                options,
            )
            .total_bytes
        }
    };
    Ok(PreflightAdmission {
        reserved_bytes,
        checkpoint_segments,
    })
}

fn validate_grpo_submission_source(req: &GrpoRequest) -> Result<(), ApiError> {
    if req.dataset_path.is_some() && !req.groups.is_empty() {
        return Err(ApiError::training_invalid_request(
            "GRPO request must use either groups or dataset_path, not both",
        ));
    }
    if req.dataset_path.is_none() && req.groups.is_empty() {
        return Err(ApiError::training_invalid_request(
            "GRPO request needs either non-empty groups or dataset_path",
        ));
    }
    // Fail fast on loss compositions the kt-tape trainer cannot train
    // (ECHO env-CE with environment tokens, no_policy_loss, reserved OPD
    // slot) — the worker would otherwise dequeue a job guaranteed to die,
    // possibly hours later behind a long queue.
    let has_env_tokens = req.groups.iter().any(|g| {
        g.completions.iter().any(|c| {
            c.trajectory
                .iter()
                .any(|seg| seg.kind == kiln_train::trajectory::TurnKind::Observation)
        })
    });
    req.config
        .loss
        .validate_for_kt_tape(has_env_tokens)
        .map_err(ApiError::training_invalid_request)?;
    Ok(())
}

/// Response for queue listing.
#[derive(serde::Serialize)]
struct QueueResponse {
    /// Currently running job (if any).
    running: Option<TrainingStatus>,
    /// Jobs waiting in the queue.
    queued: Vec<QueueStatusEntry>,
    /// Recently completed/failed jobs.
    completed: Vec<TrainingStatus>,
}

#[derive(serde::Serialize)]
struct QueueStatusEntry {
    job_id: String,
    job_type: TrainingJobType,
    adapter_name: String,
    position: usize,
}

/// Submit-time guard for the trainer's LoRA alpha/rank safety gate.
///
/// The trainer enforces this anyway — but only after the job has been
/// accepted, queued, and run, so the caller sees a 200 and a doomed job
/// instead of a 400. The dashboard's corrections train shipped a
/// rank-8/alpha-32 (ratio 4.0) config that failed every job this way
/// while the UI had already marked the basket trained. Same message as
/// the trainer's so the two surfaces never disagree.
pub(crate) fn validate_lora_scale_at_submit(
    lora_rank: usize,
    lora_alpha: f32,
    allow_high_lora_scale: bool,
) -> Result<(), ApiError> {
    kiln_train::lora_scaling::validate_lora_scaling(lora_rank, lora_alpha, allow_high_lora_scale)
        .map(|_| ())
        .map_err(|e| ApiError::training_invalid_request(format!("{e:#}")))
}

fn validate_opd_loss_at_submit(loss: kiln_train::OpdLossGranularity) -> Result<(), ApiError> {
    if matches!(loss, kiln_train::OpdLossGranularity::FullVocab) {
        return Err(ApiError::training_invalid_request(
            "OPD loss is unavailable: full_vocab has no concrete server-built teacher source; use teacher_top_k"
                .to_string(),
        ));
    }
    match loss.unsupported_reason() {
        Some(reason) => Err(ApiError::training_invalid_request(format!(
            "OPD loss is unavailable: {reason}"
        ))),
        None => Ok(()),
    }
}

fn validate_opd_config_at_submit(config: &kiln_train::OpdConfig) -> Result<(), ApiError> {
    validate_opd_loss_at_submit(config.loss)?;
    config
        .validate_runtime_contract()
        .map_err(|error| ApiError::training_invalid_request(format!("{error:#}")))
}

fn require_off_policy_fixture_mode(
    surface: &str,
    config: &kiln_train::OpdConfig,
) -> Result<(), ApiError> {
    if !matches!(
        config.training_mode,
        kiln_train::opd::OpdTrainingMode::OffPolicy
    ) {
        return Err(ApiError::training_invalid_request(format!(
            "{surface} materializes teacher logits for fixed action sequences and requires config.training_mode=\"off_policy\""
        )));
    }
    Ok(())
}

fn registered_teacher_top_k_limit(
    spec: &super::teachers::TeacherSpec,
    requested_top_k: usize,
) -> usize {
    let configured = spec.max_top_k.unwrap_or(0);
    let provider_limit =
        if matches!(spec.kind, super::teachers::TeacherKind::Remote) && configured == 0 {
            kiln_train::RemoteProvider::Vllm.default_max_top_k()
        } else if configured == 0 {
            requested_top_k
        } else {
            configured
        };
    spec.vocab_size
        .map_or(provider_limit, |vocab_size| provider_limit.min(vocab_size))
}

/// Resolve the user-facing OPD K to a value implemented by the active KT
/// kernels before any job or teacher request is admitted.
fn resolve_opd_top_k_at_submit(
    config: &mut kiln_train::OpdConfig,
    source_max_top_k: usize,
) -> Result<Option<(usize, usize)>, ApiError> {
    let requested = config.top_k;
    let effective = kiln_train::resolve_opd_top_k(requested, source_max_top_k).map_err(|error| {
        ApiError::training_invalid_request(format!(
            "OPD top_k {requested} is not executable with source cap {source_max_top_k}: {error:#}"
        ))
    })?;
    config.top_k = effective;
    Ok((effective != requested).then_some((requested, effective)))
}

fn top_k_adjustment_suffix(adjustment: Option<(usize, usize)>) -> String {
    adjustment.map_or_else(String::new, |(requested, effective)| {
        format!(
            " Requested top_k {requested} was resolved to effective top_k {effective} for the teacher and active OPD kernel."
        )
    })
}

fn validate_self_distill_context_at_submit(
    req: &kiln_train::DistillSelfRequest,
) -> Result<(), ApiError> {
    let require_prompts = || {
        req.prompts.as_deref().filter(|prompts| !prompts.is_empty()).ok_or_else(|| {
            ApiError::training_invalid_request(
                "distill/self: this privileged-information mode requires explicit non-empty `prompts`"
                    .to_string(),
            )
        })
    };
    match req.mode {
        kiln_train::SelfDistillMode::GroundTruthConditioning => {
            let prompts = require_prompts()?;
            let answers = req.ground_truth.as_deref().ok_or_else(|| {
                ApiError::training_invalid_request(
                    "distill/self: ground_truth_conditioning requires `ground_truth`".to_string(),
                )
            })?;
            if answers.len() != prompts.len()
                || answers.iter().any(|answer| answer.trim().is_empty())
            {
                return Err(ApiError::training_invalid_request(format!(
                    "distill/self: ground_truth must contain one non-empty answer per prompt ({} prompts, {} answers)",
                    prompts.len(),
                    answers.len()
                )));
            }
        }
        kiln_train::SelfDistillMode::DocumentAsPi => {
            let prompts = require_prompts()?;
            let documents = req.documents.as_deref().ok_or_else(|| {
                ApiError::training_invalid_request(
                    "distill/self: document_as_pi requires `documents`".to_string(),
                )
            })?;
            if documents.len() != prompts.len()
                || documents.iter().any(|document| document.trim().is_empty())
            {
                return Err(ApiError::training_invalid_request(format!(
                    "distill/self: documents must contain one non-empty context per prompt ({} prompts, {} documents)",
                    prompts.len(),
                    documents.len()
                )));
            }
        }
        kiln_train::SelfDistillMode::Conciseness | kiln_train::SelfDistillMode::ReverseTeacher => {}
    }
    Ok(())
}

fn opd_prompt_has_action(prompt: &kiln_train::opd::OpdPrompt) -> bool {
    prompt
        .messages
        .iter()
        .any(|message| message.role == "assistant" && !message.content.trim().is_empty())
        || prompt.trajectory.iter().any(|segment| {
            matches!(segment.kind, kiln_train::trajectory::TurnKind::Action)
                && !segment.content.trim().is_empty()
        })
}

fn validate_opd_prompts_at_submit(
    surface: &str,
    prompts: &[kiln_train::opd::OpdPrompt],
    require_action: bool,
) -> Result<(), ApiError> {
    if prompts.is_empty() {
        return Err(ApiError::training_invalid_request(format!(
            "{surface} requires at least one prompt"
        )));
    }
    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        if prompt.messages.is_empty() {
            return Err(ApiError::training_invalid_request(format!(
                "{surface} prompt {prompt_idx} has no messages"
            )));
        }
        if require_action && !opd_prompt_has_action(prompt) {
            return Err(ApiError::training_invalid_request(format!(
                "{surface} prompt {prompt_idx} requires a non-empty assistant action for off-policy scoring"
            )));
        }
    }
    Ok(())
}

fn validate_opd_request_at_submit(req: &kiln_train::OpdRequest) -> Result<(), ApiError> {
    if req.prompts.is_empty() == req.dataset_path.is_none() {
        return Err(ApiError::training_invalid_request(
            "OPD request must use exactly one of non-empty prompts or dataset_path".to_string(),
        ));
    }
    if req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "OPD request must specify a teacher alias".to_string(),
        ));
    }
    if let Some(path) = req.dataset_path.as_deref() {
        if path.trim().is_empty() {
            return Err(ApiError::training_invalid_request(
                "OPD dataset_path must be non-empty".to_string(),
            ));
        }
        if !crate::dataset_resolve::is_agent_traces_selector(path)
            && !matches!(
                req.config.training_mode,
                kiln_train::opd::OpdTrainingMode::OffPolicy
            )
        {
            return Err(ApiError::training_invalid_request(
                "OPD teacher-logprob JSONL requires config.training_mode=\"off_policy\"; for on-policy training on pi sessions use an `agent_traces:` selector"
                    .to_string(),
            ));
        }
    } else {
        let require_action = matches!(
            req.config.training_mode,
            kiln_train::opd::OpdTrainingMode::OffPolicy
        );
        validate_opd_prompts_at_submit("OPD", &req.prompts, require_action)?;
    }
    validate_opd_config_at_submit(&req.config)
}

fn validate_distill_merge_at_submit(
    state: &AppState,
    req: &kiln_train::DistillMergeRequest,
) -> Result<(), ApiError> {
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill_merge: name must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    if req.sources.is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill_merge: sources must be non-empty".to_string(),
        ));
    }
    validate_opd_config_at_submit(&req.config)?;
    require_off_policy_fixture_mode("distill_merge", &req.config)?;
    for source in &req.sources {
        super::adapters::validate_adapter_name(&source.adapter)?;
        let dir = state.adapter_dir.join(&source.adapter);
        if !dir.is_dir() {
            return Err(ApiError::training_invalid_request(format!(
                "distill_merge: source adapter {:?} not found at {}",
                source.adapter,
                dir.display()
            )));
        }
    }
    Ok(())
}

fn validate_distill_self_at_submit(req: &kiln_train::DistillSelfRequest) -> Result<(), ApiError> {
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/self: name must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    if matches!(req.mode, kiln_train::SelfDistillMode::ReverseTeacher) {
        return Err(ApiError::training_invalid_request(
            "distill/self: reverse_teacher requires a distinct reverse objective".to_string(),
        ));
    }
    validate_opd_config_at_submit(&req.config)?;
    require_off_policy_fixture_mode("distill/self", &req.config)?;
    let prompts = req.prompts.as_deref().ok_or_else(|| {
        ApiError::training_invalid_request(
            "distill/self requires explicit off-policy prompts with assistant actions".to_string(),
        )
    })?;
    validate_opd_prompts_at_submit("distill/self", prompts, true)?;
    validate_self_distill_context_at_submit(req)
}

fn validate_distill_pump_at_submit(req: &kiln_train::DistillPumpRequest) -> Result<(), ApiError> {
    if req.name.trim().is_empty() || req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/pump requires non-empty name and teacher".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    validate_opd_config_at_submit(&req.config)?;
    let off_policy = matches!(
        req.config.training_mode,
        kiln_train::opd::OpdTrainingMode::OffPolicy
    );
    match &req.mode {
        kiln_train::DistillPumpMode::Examples { examples } => {
            validate_opd_prompts_at_submit("distill/pump", examples, off_policy)?;
        }
        _ if off_policy => {
            return Err(ApiError::training_invalid_request(
                    "distill/pump domain and wide seed modes require on_policy training; off_policy requires explicit examples with assistant actions"
                        .to_string(),
                ));
        }
        _ => {}
    }
    Ok(())
}

fn validate_distill_refresh_at_submit(
    req: &kiln_train::DistillRefreshRequest,
) -> Result<(), ApiError> {
    if req.name.trim().is_empty() || req.behavioural_teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/refresh requires non-empty name and behavioural_teacher".to_string(),
        ));
    }
    validate_opd_config_at_submit(&req.config)?;
    if let kiln_train::NewKnowledgeSource::Inline { examples } = &req.new_data {
        // The first phase is SFT, so every inline example needs an assistant
        // target even when the recovery phase samples on-policy.
        validate_opd_prompts_at_submit("distill/refresh", examples, true)?;
    }
    Ok(())
}

/// Normalize every OPD-class queue payload, including recipes, scheduled
/// self-improvement, and the intent-aware front door that bypass dedicated
/// endpoint handlers.
pub(crate) fn normalize_queued_opd_top_k(
    state: &AppState,
    job: &mut QueuedJob,
) -> Result<Option<(usize, usize)>, ApiError> {
    match job {
        QueuedJob::Opd(req) => {
            validate_opd_request_at_submit(req)?;
            let spec = super::teachers::require_registered_teacher(
                state,
                &req.teacher,
                format!("OPD teacher alias '{}' is not registered", req.teacher),
            )?;
            let prescored = req
                .dataset_path
                .as_deref()
                .is_some_and(|path| !crate::dataset_resolve::is_agent_traces_selector(path));
            let source_limit = if prescored {
                req.config.top_k
            } else {
                registered_teacher_top_k_limit(&spec, req.config.top_k)
            };
            resolve_opd_top_k_at_submit(&mut req.config, source_limit)
        }
        QueuedJob::DistillRefresh(req) => {
            validate_distill_refresh_at_submit(req)?;
            let spec = super::teachers::require_registered_teacher(
                state,
                &req.behavioural_teacher,
                format!(
                    "DistillRefresh: behavioural_teacher alias '{}' is not registered",
                    req.behavioural_teacher
                ),
            )?;
            let source_limit = registered_teacher_top_k_limit(&spec, req.config.top_k);
            resolve_opd_top_k_at_submit(&mut req.config, source_limit)
        }
        QueuedJob::DistillPump(req) => {
            validate_distill_pump_at_submit(req)?;
            let spec = super::teachers::require_registered_teacher(
                state,
                &req.teacher,
                format!(
                    "distill/pump: teacher alias '{}' is not registered",
                    req.teacher
                ),
            )?;
            let source_limit = registered_teacher_top_k_limit(&spec, req.config.top_k);
            resolve_opd_top_k_at_submit(&mut req.config, source_limit)
        }
        QueuedJob::DistillMerge(req) => {
            validate_distill_merge_at_submit(state, req)?;
            let requested = req.config.top_k;
            resolve_opd_top_k_at_submit(&mut req.config, requested)
        }
        QueuedJob::DistillSelf(req) => {
            validate_distill_self_at_submit(req)?;
            let requested = req.config.top_k;
            resolve_opd_top_k_at_submit(&mut req.config, requested)
        }
        QueuedJob::Sft(_) | QueuedJob::Grpo(_) => Ok(None),
    }
}

async fn submit_sft(
    State(state): State<AppState>,
    Json(mut req): Json<SftRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    // Reject new jobs during shutdown
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;

    // Reject when the queue is at its configured cap. This protects the
    // server from queue-exhaustion DoS where a client submits jobs faster
    // than the trainer can drain them. Audit reference: security-audit-v0.1
    // §4 part 1.
    let max_queued = state.max_queued_training_jobs;
    let queued_now = state.training_queue.lock().unwrap().len();
    if queued_now >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }

    // Reject when the tracking map is at its configured cap. The training
    // worker GC's terminal entries on a TTL, but a flood of `Completed` /
    // `Failed` entries could still pin the map and exhaust memory. Audit
    // reference: security-audit-v0.1 §4 part 2.
    let max_tracked = state.max_tracked_jobs;
    let tracked_now = state.training_jobs.read().unwrap().len();
    if tracked_now >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }

    // The corrections feed: dataset "corrections:active" resolves the
    // durable basket's trainable rows (hand-written ideal, not yet
    // trained) server-side. The consumed row ids ride the job and flip
    // to trained_into ON COMPLETION — a failed job leaves the basket
    // intact and re-trainable.
    let mut consumed_correction_ids: Vec<String> = Vec::new();
    if let Some(path) = req.dataset_path.take() {
        let path = path.trim().to_string();
        if !path.is_empty() {
            req.dataset_path = Some(path);
        }
    }
    if req.dataset_path.is_some() && (!req.examples.is_empty() || req.dataset.is_some()) {
        return Err(ApiError::training_invalid_request(
            "SFT request must use exactly one of examples, dataset_path, or dataset",
        ));
    }
    if req.dataset_path.is_none() && req.examples.is_empty() && req.dataset.is_none() {
        return Err(ApiError::training_invalid_request(
            "SFT request needs examples, dataset_path, or dataset",
        ));
    }

    if req.dataset.as_deref() == Some("corrections:active") {
        req.dataset = None;
        if !req.examples.is_empty() {
            return Err(ApiError::training_invalid_request(
                "SFT request must use exactly one of examples, dataset_path, or dataset",
            ));
        }
        let store = super::corrections::CorrectionsStore::for_state(&state);
        let (ids, examples) = store.trainable_rows();
        if examples.is_empty() {
            return Err(ApiError::training_invalid_request(
                "corrections:active has no trainable rows — write an ideal answer \
                 (different from the original) for at least one correction first",
            ));
        }
        consumed_correction_ids = ids;
        req.examples = examples;
    }

    // Train-by-dataset-name: resolve an uploaded dataset (the eval dataset
    // store) into inline examples server-side. The UI/CLI sends just the name,
    // so rows never round-trip through the client and the whole dataset trains
    // (the rows preview endpoint clamps at 5000 — this path has no such cap).
    if let Some(dataset_name) = req.dataset.take() {
        if !req.examples.is_empty() {
            return Err(ApiError::training_invalid_request(
                "SFT request must use exactly one of examples, dataset_path, or dataset",
            ));
        }
        let registry = state
            .dataset_registry
            .as_ref()
            .ok_or_else(ApiError::dataset_registry_unavailable)?;
        let iter = registry.iter_sft(&dataset_name).map_err(|e| match e {
            crate::eval::DatasetError::NotFound(_) => ApiError::dataset_not_found(&dataset_name),
            crate::eval::DatasetError::InvalidName(_) => ApiError::dataset_invalid(&dataset_name),
            other => ApiError::dataset_invalid(format!("{other}")),
        })?;
        req.examples = iter
            .map(|conv| kiln_train::SftExample {
                messages: conv
                    .messages
                    .into_iter()
                    .map(|m| kiln_train::ChatMessage {
                        role: m.role,
                        content: m.content,
                    })
                    .collect(),
            })
            .filter(|ex| !ex.messages.is_empty())
            .collect();
        if req.examples.is_empty() {
            return Err(ApiError::training_invalid_request(format!(
                "dataset '{dataset_name}' contains no usable SFT examples",
            )));
        }
    }

    let stats = if let Some(path) = req.dataset_path.as_deref() {
        let path = Path::new(path);
        let stats = crate::sft_dataset::scan_sft_jsonl_stats(path, Some(state.tokenizer.as_ref()))
            .map_err(|e| {
                ApiError::training_invalid_request(format!(
                    "invalid SFT dataset_path '{}': {e:#}",
                    path.display()
                ))
            })?;
        SftSubmissionStats {
            num_examples: stats.examples,
            max_seq_len: stats.max_seq_len,
            max_supervised_tokens: stats.max_supervised_tokens,
            streaming_dataset: true,
        }
    } else {
        SftSubmissionStats {
            num_examples: req.examples.len(),
            max_seq_len: training_preflight::approximate_max_seq_len_sft(
                &req.examples,
                Some(state.tokenizer.as_ref()),
            ),
            max_supervised_tokens: training_preflight::approximate_max_supervised_tokens_sft(
                &req.examples,
                Some(state.tokenizer.as_ref()),
            ),
            streaming_dataset: false,
        }
    };
    let num_examples = stats.num_examples;
    let job_id = uuid::Uuid::new_v4().to_string();
    if let Some(name) = req.config.output_name.as_deref() {
        super::adapters::validate_adapter_name(name)?;
    }
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("sft-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    // Verify we have real model weights
    if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
        return Err(ApiError::mock_mode_no_training());
    }

    // Working-set preflight: refuse jobs that won't fit in the
    // corrected memory budget. Better than OOM-killing the server
    // partway through the first step.
    let max_seq_len = stats.max_seq_len;
    let max_supervised_tokens = stats.max_supervised_tokens;
    let admission = enforce_training_preflight(
        &state,
        max_seq_len,
        EstimateOptions {
            max_supervised_tokens: Some(max_supervised_tokens),
            recompute_boundaries: training_preflight::recompute_checkpoint_boundaries_for_seq_len(
                max_seq_len,
            ),
            ..Default::default()
        },
        req.config.lora_rank,
        // Vulkan now trains through the shared kt-tape path (segment-
        // checkpointed), not the deleted vk_native recompute fork, so the
        // preflight uses the standard segment-checkpoint working-set estimate.
        false,
    )?;
    req.config.grad_checkpoint_segments = admission.checkpoint_segments;
    let reserved_bytes = admission.reserved_bytes;

    tracing::info!(
        num_examples,
        job_id = %job_id,
        adapter = %adapter_name,
        max_seq_len,
        dataset_path = req.dataset_path.as_deref().unwrap_or(""),
        streaming_dataset = stats.streaming_dataset,
        "SFT training request queued"
    );

    // Register the job in the tracking map
    let info = TrainingJobInfo {
        job_id: job_id.clone(),
        adapter_name: adapter_name.clone(),
        job_type: TrainingJobType::Sft,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids,
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    // Enqueue and publish the tracking record under one admission lock pair.
    let queue_position = admit_training_jobs(
        &state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.clone(),
                reserved_bytes,
                job: QueuedJob::Sft(req),
            },
        )],
    )?;

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!(
            "Queued SFT training with {num_examples} examples (position {queue_position} in queue)"
        ),
    }))
}

async fn submit_grpo(
    State(state): State<AppState>,
    Json(mut req): Json<GrpoRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    // Reject new jobs during shutdown
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;

    // Train-by-dataset-name: resolve an uploaded dataset (the eval dataset
    // store) to its on-disk JSONL and ride the existing dataset_path
    // streaming path. Callers send just the name — no rows round-trip.
    if let Some(dataset_name) = req.dataset.take() {
        if !req.groups.is_empty() || req.dataset_path.is_some() {
            return Err(ApiError::training_invalid_request(
                "GRPO request must use exactly one of groups, dataset_path, or dataset",
            ));
        }
        let registry = state
            .dataset_registry
            .as_ref()
            .ok_or_else(ApiError::dataset_registry_unavailable)?;
        let dir = registry.dataset_dir(&dataset_name).map_err(|e| match e {
            crate::eval::DatasetError::InvalidName(_) => ApiError::dataset_invalid(&dataset_name),
            other => ApiError::dataset_invalid(format!("{other}")),
        })?;
        let path = dir.join("data.jsonl");
        if !path.is_file() {
            return Err(ApiError::dataset_not_found(&dataset_name));
        }
        req.dataset_path = Some(path.to_string_lossy().into_owned());
    }

    // Reject when the queue is at its configured cap. See submit_sft above
    // for the audit reference.
    let max_queued = state.max_queued_training_jobs;
    let queued_now = state.training_queue.lock().unwrap().len();
    if queued_now >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }

    // Reject when the tracking map is at its configured cap. See submit_sft
    // above for the audit reference (security-audit-v0.1 §4 part 2).
    let max_tracked = state.max_tracked_jobs;
    let tracked_now = state.training_jobs.read().unwrap().len();
    if tracked_now >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }

    if let Some(path) = req.dataset_path.take() {
        let path = path.trim().to_string();
        if !path.is_empty() {
            req.dataset_path = Some(path);
        }
    }
    validate_grpo_submission_source(&req)?;

    let stats = if let Some(path) = req.dataset_path.as_deref() {
        validate_grpo_jsonl_submission_head(path, Some(state.tokenizer.as_ref()))?
    } else {
        GrpoSubmissionStats {
            num_groups: Some(req.groups.len()),
            total_completions: Some(req.groups.iter().map(|g| g.completions.len()).sum()),
            max_seq_len: training_preflight::approximate_max_seq_len_grpo(
                &req.groups,
                Some(state.tokenizer.as_ref()),
            ),
            streaming_dataset: false,
        }
    };
    let job_id = uuid::Uuid::new_v4().to_string();
    if let Some(name) = req.config.output_name.as_deref() {
        super::adapters::validate_adapter_name(name)?;
    }
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("grpo-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    if stats.streaming_dataset {
        tracing::info!(
            dataset_path = req.dataset_path.as_deref().unwrap_or_default(),
            max_seq_len_first_group = stats.max_seq_len,
            job_id = %job_id,
            adapter = %adapter_name,
            "streamed GRPO training request queued"
        );
    } else {
        tracing::info!(
            num_groups = stats.num_groups.unwrap_or(0),
            total_completions = stats.total_completions.unwrap_or(0),
            job_id = %job_id,
            adapter = %adapter_name,
            "GRPO training request queued"
        );
    }

    // Verify we have real model weights
    if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
        return Err(ApiError::mock_mode_no_training());
    }

    // Working-set preflight (see submit_sft for rationale). Vulkan trains
    // through the shared kt-tape (segment-checkpointed) path now, not the
    // deleted vk_native recompute fork, so the standard estimate applies.
    let max_seq_len = stats.max_seq_len;
    let admission = enforce_training_preflight(
        &state,
        max_seq_len,
        EstimateOptions {
            recompute_boundaries: training_preflight::recompute_checkpoint_boundaries_for_seq_len(
                max_seq_len,
            ),
            ..Default::default()
        },
        req.config.lora_rank,
        false,
    )?;
    req.config.grad_checkpoint_segments = admission.checkpoint_segments;
    let reserved_bytes = admission.reserved_bytes;

    // Register the job in the tracking map
    let info = TrainingJobInfo {
        job_id: job_id.clone(),
        adapter_name: adapter_name.clone(),
        job_type: TrainingJobType::Grpo,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    // Enqueue and publish the tracking record under one admission lock pair.
    let queue_position = admit_training_jobs(
        &state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.clone(),
                reserved_bytes,
                job: QueuedJob::Grpo(req),
            },
        )],
    )?;

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: if stats.streaming_dataset {
            format!(
                "Queued streamed GRPO training from dataset_path (position {queue_position} in queue)"
            )
        } else {
            let num_groups = stats.num_groups.unwrap_or(0);
            let total_completions = stats.total_completions.unwrap_or(0);
            format!(
                "Queued GRPO training with {num_groups} groups ({total_completions} completions, position {queue_position} in queue)"
            )
        },
    }))
}

/// `POST /v1/train/opd` — submit an On-Policy Distillation training run.
///
/// Mirror of `submit_grpo` adapted to the §3.1 OPD recipe. The request
/// shape is `OpdRequest` (defined in `kiln-train::opd`): a list of
/// prompts, a teacher alias, and an `OpdConfig` whose §6 paper-cited
/// defaults select the bounded executable path (top_k=32, temperature=1.0,
/// top_p=0.9, max_tokens=7K, direct reverse-KL, Stable-OPD off).
///
/// Same queue / hot-swap / auto-load / post-eval semantics as SFT/GRPO.
/// Job tracking via `/v1/train/status`, `/v1/train/queue`, etc.
async fn submit_opd(
    State(state): State<AppState>,
    Json(mut req): Json<OpdRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    // Reject during shutdown.
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;

    // Queue / tracking caps — mirror SFT/GRPO.
    let max_queued = state.max_queued_training_jobs;
    let queued_now = state.training_queue.lock().unwrap().len();
    if queued_now >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }
    let max_tracked = state.max_tracked_jobs;
    let tracked_now = state.training_jobs.read().unwrap().len();
    if tracked_now >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }

    // Trim a blank dataset_path the same way GRPO does.
    if let Some(path) = req.dataset_path.take() {
        let path = path.trim().to_string();
        if !path.is_empty() {
            req.dataset_path = Some(path);
        }
    }
    validate_opd_request_at_submit(&req)?;

    if req.prompts.is_empty() && req.dataset_path.is_none() {
        return Err(ApiError::training_invalid_request(
            "OPD request must include at least one prompt or a dataset_path".to_string(),
        ));
    }
    if req.dataset_path.is_some() && !req.prompts.is_empty() {
        return Err(ApiError::training_invalid_request(
            "OPD request must use either prompts or dataset_path, not both".to_string(),
        ));
    }
    if req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "OPD request must specify a teacher alias (e.g. \"qwen3.6-27b@local\")".to_string(),
        ));
    }
    // A plain-file dataset_path is pre-scored off-policy teacher JSONL;
    // `agent_traces:` selectors are on-policy prompt sources. The worker
    // enforces this too — but at submission the caller can still fix it.
    if let Some(path) = req.dataset_path.as_deref() {
        if !crate::dataset_resolve::is_agent_traces_selector(path)
            && !matches!(
                req.config.training_mode,
                kiln_train::opd::OpdTrainingMode::OffPolicy
            )
        {
            return Err(ApiError::training_invalid_request(
                "OPD dataset_path (teacher-logprob JSONL) requires config.training_mode = \
                 \"off_policy\"; for on-policy training on pi sessions use an \
                 `agent_traces:` selector instead"
                    .to_string(),
            ));
        }
    }
    for (i, prompt) in req.prompts.iter().enumerate() {
        if prompt.messages.is_empty() {
            return Err(ApiError::training_invalid_request(format!(
                "OPD prompt {i} has no messages"
            )));
        }
    }
    if req.config.top_k == 0 {
        return Err(ApiError::training_invalid_request(
            "OPD top_k must be > 0".to_string(),
        ));
    }
    validate_opd_config_at_submit(&req.config)?;
    if req.config.samples_per_prompt == 0 {
        return Err(ApiError::training_invalid_request(
            "OPD samples_per_prompt must be > 0".to_string(),
        ));
    }

    let job_id = uuid::Uuid::new_v4().to_string();
    if let Some(name) = req.config.output_name.as_deref() {
        super::adapters::validate_adapter_name(name)?;
    }
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    // The worker resolves the teacher only at dequeue — a typo'd alias
    // used to enqueue a job guaranteed to fail later, possibly hours
    // later behind a long queue. Fail here with the remediation. (After
    // the pure-input checks above: a malformed request is the caller's
    // first problem, an unregistered teacher the second.)
    let teacher_spec = super::teachers::require_registered_teacher(
        &state,
        &req.teacher,
        format!("OPD teacher alias '{}' is not registered", req.teacher),
    )?;
    let uses_prescored_dataset = req
        .dataset_path
        .as_deref()
        .is_some_and(|path| !crate::dataset_resolve::is_agent_traces_selector(path));
    let source_max_top_k = if uses_prescored_dataset {
        req.config.top_k
    } else {
        registered_teacher_top_k_limit(&teacher_spec, req.config.top_k)
    };
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, source_max_top_k)?;
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("opd-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    tracing::info!(
        num_prompts = req.prompts.len(),
        teacher = %req.teacher,
        loss = ?req.config.loss,
        top_k = req.config.top_k,
        requested_top_k = top_k_adjustment.map(|(requested, _)| requested),
        samples_per_prompt = req.config.samples_per_prompt,
        job_id = %job_id,
        adapter = %adapter_name,
        "OPD training request queued"
    );

    if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
        return Err(ApiError::mock_mode_no_training());
    }

    // Register the job and enqueue. OPD now runs through the real
    // trainer body (kiln_train::opd::opd_train) — same GPU lock /
    // replay / hot-swap / receipt semantics as SFT and GRPO. Wiring
    // OPD into the SFT/GRPO working-set preflight (so the §8.5
    // capacity calc applies) is a follow-up.
    // Working-set reservation (#36): OPD previously skipped the preflight
    // entirely (no VRAM check, no governor reservation). Estimate its footprint
    // — longest prompt + rollout budget — and reserve it like SFT/GRPO, so the
    // KV autoscaler accounts for OPD and a too-large job is rejected instead of
    // OOMing mid-run.
    let opd_max_seq_len = training_preflight::approximate_max_seq_len_opd(
        &req.prompts,
        req.config.max_tokens,
        Some(state.tokenizer.as_ref()),
    );
    let reserved_bytes = enforce_training_preflight(
        &state,
        opd_max_seq_len,
        EstimateOptions {
            max_supervised_tokens: None,
            recompute_boundaries: training_preflight::recompute_checkpoint_boundaries_for_seq_len(
                opd_max_seq_len,
            ),
            ..Default::default()
        },
        req.config.lora_rank,
        false,
    )?
    .reserved_bytes;

    let info = TrainingJobInfo {
        job_id: job_id.clone(),
        adapter_name: adapter_name.clone(),
        job_type: TrainingJobType::Opd,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    let queue_position = admit_training_jobs(
        &state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.clone(),
                reserved_bytes, // #36: OPD working-set reservation (preflight estimate)
                job: QueuedJob::Opd(req),
            },
        )],
    )?;

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!(
            "Queued OPD training (position {queue_position} in queue).{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// `POST /v1/distill/refresh` — §3.6 continual-learning recipe
/// (Lu 2025 instruction-following recovery experiment).
///
/// Body: [`DistillRefreshRequest`]. The runtime mid-trains on
/// `new_data` then OPD-recovers against the prior-self
/// `behavioural_teacher`, gated on dual eval (IF-eval recovery +
/// new-knowledge gain). Same queue / receipt / auto-load semantics
/// as `/v1/train/opd`.
async fn submit_distill_refresh(
    State(state): State<AppState>,
    Json(mut req): Json<DistillRefreshRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    validate_distill_refresh_at_submit(&req)?;
    let max_queued = state.max_queued_training_jobs;
    let queued_now = state.training_queue.lock().unwrap().len();
    if queued_now >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }
    let max_tracked = state.max_tracked_jobs;
    let tracked_now = state.training_jobs.read().unwrap().len();
    if tracked_now >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "DistillRefresh: `name` must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    validate_opd_config_at_submit(&req.config)?;
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    if req.behavioural_teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "DistillRefresh: `behavioural_teacher` alias must be non-empty".to_string(),
        ));
    }
    let teacher_spec = super::teachers::require_registered_teacher(
        &state,
        &req.behavioural_teacher,
        format!(
            "DistillRefresh: behavioural_teacher alias '{}' is not registered",
            req.behavioural_teacher
        ),
    )?;
    let source_max_top_k = registered_teacher_top_k_limit(&teacher_spec, req.config.top_k);
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, source_max_top_k)?;
    if !(0.0..=1.0).contains(&req.require_if_eval_recovery) {
        return Err(ApiError::training_invalid_request(
            "require_if_eval_recovery must be in [0.0, 1.0]".to_string(),
        ));
    }

    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = format!("{}@refresh-{}", req.name, &job_id[..8]);
    let auto_load = req.config.auto_load;

    tracing::info!(
        name = %req.name,
        behavioural_teacher = %req.behavioural_teacher,
        background_chat = %req.background_chat,
        require_if_eval_recovery = req.require_if_eval_recovery,
        require_internal_qa_gain = req.require_internal_qa_gain,
        job_id = %job_id,
        adapter = %adapter_name,
        "distill/refresh request queued"
    );

    if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
        return Err(ApiError::mock_mode_no_training());
    }

    // Preflight BEFORE the tracking insert — a 413 here must not leave an
    // orphaned Queued entry in the jobs map.
    let reserved_bytes = distill_working_set_reservation(
        &state,
        match &req.new_data {
            kiln_train::NewKnowledgeSource::Inline { examples } => examples.as_slice(),
            kiln_train::NewKnowledgeSource::Dataset { .. } => &[],
        },
        &req.config,
    )?;

    let info = TrainingJobInfo {
        job_id: job_id.clone(),
        adapter_name: adapter_name.clone(),
        // Reuse the Opd job type — refresh is structurally an OPD run
        // with extra orchestration. Dashboards group both as OPD-class.
        job_type: TrainingJobType::Opd,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    let queue_position = admit_training_jobs(
        &state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.clone(),
                reserved_bytes,
                job: QueuedJob::DistillRefresh(req),
            },
        )],
    )?;

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!(
            "Queued distill/refresh (position {queue_position} in queue).{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// `POST /v1/adapters/distill_merge` — §3.4 behaviour-space merge.
async fn submit_distill_merge(
    State(state): State<AppState>,
    Json(mut req): Json<DistillMergeRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    validate_distill_merge_at_submit(&state, &req)?;
    if req.sources.is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill_merge: `sources` must be non-empty".to_string(),
        ));
    }
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill_merge: `name` must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    validate_opd_config_at_submit(&req.config)?;
    require_off_policy_fixture_mode("distill_merge", &req.config)?;
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    let requested_top_k = req.config.top_k;
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, requested_top_k)?;
    // A source adapter that doesn't exist on disk is a typo — fail now,
    // not after the job dequeues and silently falls back to the base
    // model for that source's prompts.
    for source in &req.sources {
        super::adapters::validate_adapter_name(&source.adapter)?;
        let dir = state.adapter_dir.join(&source.adapter);
        if !dir.is_dir() {
            return Err(ApiError::training_invalid_request(format!(
                "distill_merge: source adapter `{}` not found at {}",
                source.adapter,
                dir.display()
            )));
        }
    }
    enforce_queue_caps(&state)?;
    let reserved_bytes = distill_working_set_reservation(&state, &[], &req.config)?;
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req.name.clone();
    let auto_load = req.config.auto_load;
    register_and_enqueue_distill(
        &state,
        &job_id,
        &adapter_name,
        auto_load,
        reserved_bytes,
        QueuedJob::DistillMerge(req),
    )?;
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!(
            "Queued distill_merge.{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// `POST /v1/distill/pump` — §3.5 Knowledge Pump.
async fn submit_distill_pump(
    State(state): State<AppState>,
    Json(mut req): Json<DistillPumpRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    validate_distill_pump_at_submit(&req)?;
    if req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/pump: `teacher` alias must be non-empty".to_string(),
        ));
    }
    let teacher_spec = super::teachers::require_registered_teacher(
        &state,
        &req.teacher,
        format!(
            "distill/pump: teacher alias '{}' is not registered",
            req.teacher
        ),
    )?;
    super::adapters::validate_adapter_name(&req.name)?;
    validate_opd_config_at_submit(&req.config)?;
    // The worker overrides config.lora_rank with the request's top-level
    // `rank` when set (training_queue.rs pump arm) — validate the rank
    // that will actually train, not the config default it shadows.
    validate_lora_scale_at_submit(
        req.rank.unwrap_or(req.config.lora_rank),
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    let source_max_top_k = registered_teacher_top_k_limit(&teacher_spec, req.config.top_k);
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, source_max_top_k)?;
    enforce_queue_caps(&state)?;
    let inline_prompts: &[kiln_train::opd::OpdPrompt] = match &req.mode {
        kiln_train::DistillPumpMode::Examples { examples } => examples.as_slice(),
        // Domain/Wide resolve to short seed prompts at run time; the
        // rollout budget dominates the working set.
        _ => &[],
    };
    let reserved_bytes = distill_working_set_reservation(&state, inline_prompts, &req.config)?;
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req.name.clone();
    let auto_load = req.config.auto_load;
    register_and_enqueue_distill(
        &state,
        &job_id,
        &adapter_name,
        auto_load,
        reserved_bytes,
        QueuedJob::DistillPump(req),
    )?;
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!(
            "Queued distill/pump.{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// `POST /v1/distill/self` — §3.12 PI self-distillation.
async fn submit_distill_self(
    State(state): State<AppState>,
    Json(mut req): Json<DistillSelfRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    validate_distill_self_at_submit(&req)?;
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/self: `name` must be non-empty".to_string(),
        ));
    }
    if matches!(req.mode, kiln_train::SelfDistillMode::ReverseTeacher) {
        return Err(ApiError::training_invalid_request(
            "distill/self: `reverse_teacher` is unsupported because it requires a distinct reverse objective; negated logprobs are invalid"
                .to_string(),
        ));
    }
    validate_self_distill_context_at_submit(&req)?;
    validate_opd_config_at_submit(&req.config)?;
    require_off_policy_fixture_mode("distill/self", &req.config)?;
    super::adapters::validate_adapter_name(&req.name)?;
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
    let requested_top_k = req.config.top_k;
    let top_k_adjustment = resolve_opd_top_k_at_submit(&mut req.config, requested_top_k)?;
    enforce_queue_caps(&state)?;
    let reserved_bytes = distill_working_set_reservation(
        &state,
        req.prompts.as_deref().unwrap_or(&[]),
        &req.config,
    )?;
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req.name.clone();
    let auto_load = req.config.auto_load;
    register_and_enqueue_distill(
        &state,
        &job_id,
        &adapter_name,
        auto_load,
        reserved_bytes,
        QueuedJob::DistillSelf(req),
    )?;
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!(
            "Queued distill/self.{}",
            top_k_adjustment_suffix(top_k_adjustment)
        ),
    }))
}

/// Shared submission gate: queue cap, tracked-jobs cap, and mock-mode
/// rejection. Used by the distill_* endpoints and every other surface
/// that enqueues training jobs (front door, recipes, agent endpoints).
/// Submission-time §8.7 gate validation: a post_eval naming a suite that
/// isn't installed must reject NOW, not after training burns GPU-hours
/// and the eval worker discovers the name resolves to nothing (round-4
/// discovery: docs/tests pointed at "agentic-core" while the installed
/// builtin is "qwen3.5-agentic-core" — gated rounds trained forever and
/// never promoted).
pub(crate) fn validate_post_eval_suite(
    state: &AppState,
    post_eval: Option<&kiln_eval::PostEvalConfig>,
) -> Result<(), ApiError> {
    let Some(cfg) = post_eval else {
        return Ok(());
    };
    let Some(registry) = state.suite_registry.as_ref() else {
        // No registry on this state (mock/test shapes) — the eval worker
        // will surface the failure; don't block submission.
        return Ok(());
    };
    if registry.load(&cfg.suite).is_err() {
        let available: Vec<String> = registry.list().into_iter().map(|s| s.name).collect();
        return Err(ApiError::training_invalid_request(format!(
            "post_eval.suite '{}' is not an installed eval suite — available: [{}]",
            cfg.suite,
            available.join(", ")
        )));
    }
    Ok(())
}

pub(crate) fn enforce_queue_caps(state: &AppState) -> Result<(), ApiError> {
    enforce_queue_capacity_for(state, 1)
}

pub(crate) fn enforce_queue_capacity_for(
    state: &AppState,
    additional_jobs: usize,
) -> Result<(), ApiError> {
    // Keep the same lock order as the atomic admission path below. This is
    // only an advisory fast-fail check: admission rechecks both caps while
    // holding write access to the tracking map and exclusive queue access.
    let tracked = state.training_jobs.read().unwrap();
    let queue = state.training_queue.lock().unwrap();
    validate_training_admission_capacity(
        queue.len(),
        state.max_queued_training_jobs,
        tracked.len(),
        state.max_tracked_jobs,
        additional_jobs,
        !matches!(state.backend.as_ref(), ModelBackend::Mock { .. }),
    )
}

fn validate_training_admission_capacity(
    queued: usize,
    max_queued: usize,
    tracked: usize,
    max_tracked: usize,
    additional_jobs: usize,
    training_supported: bool,
) -> Result<(), ApiError> {
    if queued.saturating_add(additional_jobs) > max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }
    if tracked.saturating_add(additional_jobs) > max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }
    if !training_supported {
        return Err(ApiError::mock_mode_no_training());
    }
    Ok(())
}

fn admit_training_jobs_into(
    training_jobs: &crate::state::TrainingJobs,
    training_queue: &crate::training_queue::SharedTrainingQueue,
    max_queued: usize,
    max_tracked: usize,
    training_supported: bool,
    pending: Vec<(TrainingJobInfo, QueueEntry)>,
) -> Result<usize, ApiError> {
    let additional_jobs = pending.len();

    // `list_queue` already uses this order. Holding both guards makes the
    // capacity decision and every insert one transaction with respect to all
    // other API submissions routed through this function.
    let mut tracked = training_jobs.write().unwrap();
    let mut queue = training_queue.lock().unwrap();
    validate_training_admission_capacity(
        queue.len(),
        max_queued,
        tracked.len(),
        max_tracked,
        additional_jobs,
        training_supported,
    )?;

    let mut pending_ids = std::collections::HashSet::with_capacity(additional_jobs);
    for (info, entry) in &pending {
        if info.job_id != entry.job_id {
            return Err(ApiError::internal(format!(
                "training admission job id mismatch: tracking={} queue={}",
                info.job_id, entry.job_id
            )));
        }
        if tracked.contains_key(&info.job_id) || !pending_ids.insert(info.job_id.clone()) {
            return Err(ApiError::internal(format!(
                "training admission duplicate job id: {}",
                info.job_id
            )));
        }
    }

    for (info, entry) in pending {
        tracked.insert(info.job_id.clone(), info);
        queue.push(entry);
    }
    Ok(queue.len())
}

/// Atomically reserve queue/tracking capacity and publish a complete batch.
/// A rejected batch leaves both the tracking map and FIFO unchanged.
pub(crate) fn admit_training_jobs(
    state: &AppState,
    pending: Vec<(TrainingJobInfo, QueueEntry)>,
) -> Result<usize, ApiError> {
    admit_training_jobs_into(
        &state.training_jobs,
        &state.training_queue,
        state.max_queued_training_jobs,
        state.max_tracked_jobs,
        !matches!(state.backend.as_ref(), ModelBackend::Mock { .. }),
        pending,
    )
}

/// Working-set preflight for distill-family jobs: longest inline prompt
/// plus the rollout budget, same estimator SFT/GRPO/OPD use. Distill jobs
/// used to enqueue with `reserved_bytes: 0` — no VRAM check, no governor
/// reservation — on a host that has been hard-crashed by exactly that
/// class of unchecked training allocation before.
fn distill_working_set_reservation(
    state: &AppState,
    inline_prompts: &[kiln_train::opd::OpdPrompt],
    config: &kiln_train::OpdConfig,
) -> Result<u64, ApiError> {
    let max_seq_len = training_preflight::approximate_max_seq_len_opd(
        inline_prompts,
        config.max_tokens,
        Some(state.tokenizer.as_ref()),
    );
    Ok(enforce_training_preflight(
        state,
        max_seq_len,
        EstimateOptions {
            max_supervised_tokens: None,
            recompute_boundaries: training_preflight::recompute_checkpoint_boundaries_for_seq_len(
                max_seq_len,
            ),
            ..Default::default()
        },
        config.lora_rank,
        false,
    )?
    .reserved_bytes)
}

/// Shared registration+enqueue for distill_* endpoints. Same shape as
/// `submit_distill_refresh`/`submit_opd` but inlined for the simpler
/// distill variants.
fn register_and_enqueue_distill(
    state: &AppState,
    job_id: &str,
    adapter_name: &str,
    auto_load: bool,
    reserved_bytes: u64,
    job: QueuedJob,
) -> Result<usize, ApiError> {
    let info = TrainingJobInfo {
        job_id: job_id.to_string(),
        adapter_name: adapter_name.to_string(),
        job_type: TrainingJobType::Opd,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load,
        consumed_correction_ids: Vec::new(),
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    admit_training_jobs(
        state,
        vec![(
            info,
            QueueEntry {
                job_id: job_id.to_string(),
                reserved_bytes,
                job,
            },
        )],
    )
}

fn training_status_from_info(j: &crate::state::TrainingJobInfo) -> TrainingStatus {
    TrainingStatus {
        job_id: j.job_id.clone(),
        state: j.state,
        progress: j.progress,
        current_loss: j.loss,
        adapter_name: Some(j.adapter_name.clone()),
        started_at: format!("{}s ago", j.submitted_at.elapsed().as_secs()),
        elapsed_secs: j.submitted_at.elapsed().as_secs_f64(),
        submitted_unix_ms: Some(j.submitted_unix_ms),
        finished_unix_ms: j.finished_unix_ms,
        job_type: Some(
            match j.job_type {
                TrainingJobType::Sft => "sft",
                TrainingJobType::Grpo => "grpo",
                TrainingJobType::Opd => "opd",
            }
            .into(),
        ),
        error: j.error.clone(),
        post_eval_verdict: j.post_eval_verdict.clone(),
        gate_outcome: j.gate_outcome.clone(),
    }
}

/// GET /v1/train/status — overall training status (list all tracked jobs).
async fn training_status(State(state): State<AppState>) -> Json<Vec<TrainingStatus>> {
    let jobs = state.training_jobs.read().unwrap();
    let statuses: Vec<TrainingStatus> = jobs.values().map(training_status_from_info).collect();
    Json(statuses)
}

/// GET /v1/train/status/:job_id — per-job status.
async fn job_status(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<TrainingStatus>, ApiError> {
    let jobs = state.training_jobs.read().unwrap();
    let job = jobs
        .get(&job_id)
        .ok_or_else(|| ApiError::training_job_not_found(&job_id))?;

    Ok(Json(training_status_from_info(job)))
}

/// GET /v1/train/queue — list queue contents organized by state.
async fn list_queue(State(state): State<AppState>) -> Json<QueueResponse> {
    let jobs = state.training_jobs.read().unwrap();
    let queue = state.training_queue.lock().unwrap();

    let mut running = None;
    let mut completed = Vec::new();

    for j in jobs.values() {
        let status = training_status_from_info(j);
        match j.state {
            TrainingState::Running => running = Some(status),
            TrainingState::Completed | TrainingState::Failed => completed.push(status),
            TrainingState::Queued => {} // handled from queue below
        }
    }

    // Build queued list from the actual queue (preserves FIFO order)
    let queued: Vec<QueueStatusEntry> = queue
        .queue
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let (job_type, adapter_name) = jobs
                .get(&entry.job_id)
                .map(|j| (j.job_type, j.adapter_name.clone()))
                .unwrap_or((TrainingJobType::Sft, "unknown".into()));
            QueueStatusEntry {
                job_id: entry.job_id.clone(),
                job_type,
                adapter_name,
                position: i + 1,
            }
        })
        .collect();

    // Sort completed by most-recently-finished first (falls back to submit
    // time when the terminal-transition timestamp is missing — e.g., an
    // archived entry that pre-dates the `finished_unix_ms` field).
    completed.sort_by(|a, b| {
        let a_t = a
            .finished_unix_ms
            .unwrap_or_else(|| a.submitted_unix_ms.unwrap_or(0));
        let b_t = b
            .finished_unix_ms
            .unwrap_or_else(|| b.submitted_unix_ms.unwrap_or(0));
        b_t.cmp(&a_t)
    });

    Json(QueueResponse {
        running,
        queued,
        completed,
    })
}

/// DELETE /v1/train/queue/:job_id — cancel a queued OR running job.
///
/// Queued: removed from the queue immediately. Running: the job's
/// cooperative cancel flag is set; the trainer aborts at the next step
/// boundary (typically one decode/optimizer step) and the job lands in
/// `Failed` with error "cancelled by user" and receipt failure_reason
/// "cancelled".
async fn cancel_queued_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Check job state; flag running jobs for cooperative cancellation.
    {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| ApiError::training_job_not_found(&job_id))?;
        if job.state == TrainingState::Running {
            job.cancel_requested
                .store(true, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(job_id = %job_id, "cancellation requested for running training job");
            return Ok(Json(serde_json::json!({
                "job_id": job_id,
                "status": "cancelling",
                "message": "stop requested — the trainer aborts at the next step boundary"
            })));
        }
        if job.state != TrainingState::Queued {
            return Err(ApiError::training_job_not_cancellable(
                &job_id,
                format!("{:?}", job.state),
            ));
        }
    }

    // Remove from queue
    let removed = {
        let mut q = state.training_queue.lock().unwrap();
        q.remove(&job_id)
    };

    if removed {
        // Mark as failed (cancelled) in the tracking map
        let metric_type = {
            let mut jobs = state.training_jobs.write().unwrap();
            let jt = jobs.get(&job_id).map(|j| j.job_type);
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = TrainingState::Failed;
                job.error = Some("cancelled while queued".to_string());
                job.finished_at = Some(std::time::Instant::now());
                job.finished_unix_ms = Some(crate::recent_requests::now_unix_ms());
            }
            jt
        };
        if let Some(jt) = metric_type {
            let mt = match jt {
                TrainingJobType::Sft => TrainingMetricType::Sft,
                TrainingJobType::Grpo => TrainingMetricType::Grpo,
                TrainingJobType::Opd => TrainingMetricType::Opd,
            };
            state
                .metrics
                .inc_training(mt, TrainingMetricStatus::Cancelled);
        }
        Ok(Json(serde_json::json!({
            "job_id": job_id,
            "status": "cancelled"
        })))
    } else {
        Err(ApiError::training_job_already_started(&job_id))
    }
}

/// DELETE /v1/train/jobs/:job_id — permanently delete a terminal training
/// job from both the in-memory tracking map and the on-disk archive. Refuses
/// to delete jobs that are still queued / running (use
/// `DELETE /v1/train/queue/:job_id` for those).
async fn delete_archived_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Refuse if the job is still active. The in-memory map is the source
    // of truth for live state; an archived entry will only exist for jobs
    // already in a terminal state.
    {
        let jobs = state.training_jobs.read().unwrap();
        if let Some(job) = jobs.get(&job_id) {
            match job.state {
                TrainingState::Queued | TrainingState::Running => {
                    return Err(ApiError::training_job_not_cancellable(
                        &job_id,
                        format!("{:?}", job.state),
                    ));
                }
                _ => {}
            }
        }
        // Missing from in-memory but present on-disk is also valid — we'll
        // still try to delete the archive file below.
    }

    // Remove from in-memory map (idempotent — missing is fine).
    {
        let mut jobs = state.training_jobs.write().unwrap();
        jobs.remove(&job_id);
    }

    // Delete the on-disk archive file. Missing is fine (already gone).
    let archive_path =
        crate::training_history::archive_dir(&state.adapter_dir).join(format!("{job_id}.json"));
    let removed_file = match std::fs::remove_file(&archive_path) {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => {
            return Err(ApiError::internal(format!(
                "failed to delete archive file {}: {}",
                archive_path.display(),
                e
            )));
        }
    };

    Ok(Json(serde_json::json!({
        "job_id": job_id,
        "status": "deleted",
        "removed_archive_file": removed_file,
    })))
}

/// Rich detail payload exposed at `GET /v1/train/jobs/:job_id`. Flattens
/// `TrainingStatus` so the wire shape stays a superset (no field drift)
/// and adds the curve + back-references the UI's drill-in panel needs.
#[derive(Serialize)]
struct TrainingJobDetail {
    #[serde(flatten)]
    status: TrainingStatus,
    job_type: TrainingJobType,
    epoch: Option<u32>,
    adapter_path: Option<String>,
    auto_load: bool,
    /// Eval job IDs queued by `post_eval`. `None` when no post-eval was
    /// requested; otherwise newest-first.
    linked_eval_job_ids: Vec<String>,
    /// §8.7 gate verdict (promoted / demoted to `.failed` / not measured)
    /// once the post-training eval reaches a terminal state.
    post_eval_verdict: Option<String>,
    /// Time-series of progress samples. Empty until the trainer emits
    /// its first callback.
    loss_history: Vec<crate::state::TrainingLossSample>,
    /// Machine-readable training receipt from the adapter directory when
    /// present. This carries the resolved hyperparameters, data hashes,
    /// token counts, and backend audit trail.
    train_receipt: Option<serde_json::Value>,
    /// Replay request summary from `replay.jsonl`. Large inline datasets are
    /// reduced to counts so the drill-in remains usable.
    replay_request: Option<serde_json::Value>,
    /// Non-fatal metadata read/parse error. Missing metadata is represented
    /// by null fields rather than an error.
    metadata_error: Option<String>,
}

fn training_job_adapter_dir(
    adapter_root: &Path,
    adapter_name: &str,
    adapter_path: Option<&str>,
) -> PathBuf {
    if let Some(path) = adapter_path {
        let path = PathBuf::from(path);
        if path.is_absolute() || path.exists() {
            return path;
        }
    }
    adapter_root.join(adapter_name)
}

fn read_optional_json(path: &Path) -> Result<Option<serde_json::Value>, String> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| format!("parse {}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("read {}: {e}", path.display())),
    }
}

fn summarize_replay_request(mut value: serde_json::Value) -> serde_json::Value {
    if let Some(body) = value
        .as_object_mut()
        .and_then(|obj| obj.get_mut("request_body"))
        .and_then(|body| body.as_object_mut())
    {
        for key in ["examples", "groups", "prompts"] {
            if let Some(rows) = body.remove(key) {
                let count = rows.as_array().map(|rows| rows.len()).unwrap_or(0);
                body.insert(format!("{key}_count"), serde_json::json!(count));
            }
        }
    }
    value
}

fn read_replay_request(
    adapter_dir: &Path,
    job_id: &str,
) -> Result<Option<serde_json::Value>, String> {
    let replay_path = adapter_dir.join("replay.jsonl");
    let file = match std::fs::File::open(&replay_path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read {}: {e}", replay_path.display())),
    };
    use std::io::BufRead;
    let mut first_request = None;
    for (idx, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|e| format!("read {} line {}: {e}", replay_path.display(), idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("parse {} line {}: {e}", replay_path.display(), idx + 1))?;
        if value.get("type").and_then(|v| v.as_str()) != Some("request") {
            continue;
        }
        let summary = summarize_replay_request(value);
        if summary.get("request_id").and_then(|v| v.as_str()) == Some(job_id) {
            return Ok(Some(summary));
        }
        if first_request.is_none() {
            first_request = Some(summary);
        }
    }
    Ok(first_request)
}

fn read_training_job_metadata(
    adapter_dir: &Path,
    job_id: &str,
) -> (
    Option<serde_json::Value>,
    Option<serde_json::Value>,
    Option<String>,
) {
    let mut errors = Vec::new();
    let train_receipt = match read_optional_json(&adapter_dir.join("train_receipt.json")) {
        Ok(value) => value,
        Err(err) => {
            errors.push(err);
            None
        }
    };
    let replay_request = match read_replay_request(adapter_dir, job_id) {
        Ok(value) => value,
        Err(err) => {
            errors.push(err);
            None
        }
    };
    let metadata_error = if errors.is_empty() {
        None
    } else {
        Some(errors.join("; "))
    };
    (train_receipt, replay_request, metadata_error)
}

async fn job_detail(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<TrainingJobDetail>, ApiError> {
    // Build the response inside a tight read-lock scope and drop the
    // guard before serializing. The drill-modal poll runs at 1.5s and
    // clones up to 1024 loss samples here; the trainer's progress
    // callback contends for the WRITE lock on every step. Building
    // outside a `let _ = jobs;` would extend the borrow until end of
    // function.
    let (mut detail, metadata_dir) = {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| ApiError::training_job_not_found(&job_id))?;
        (
            TrainingJobDetail {
                status: training_status_from_info(job),
                job_type: job.job_type,
                epoch: job.epoch,
                adapter_path: job.adapter_path.clone(),
                auto_load: job.auto_load,
                linked_eval_job_ids: job.linked_eval_job_ids.clone(),
                post_eval_verdict: job.post_eval_verdict.clone(),
                loss_history: job.loss_history.clone(),
                train_receipt: None,
                replay_request: None,
                metadata_error: None,
            },
            training_job_adapter_dir(
                &state.adapter_dir,
                &job.adapter_name,
                job.adapter_path.as_deref(),
            ),
        )
    };
    let (train_receipt, replay_request, metadata_error) =
        read_training_job_metadata(&metadata_dir, &job_id);
    detail.train_receipt = train_receipt;
    detail.replay_request = replay_request;
    detail.metadata_error = metadata_error;
    Ok(Json(detail))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/v1/train/sft",
            post(submit_sft).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/v1/train/grpo",
            post(submit_grpo).layer(DefaultBodyLimit::disable()),
        )
        // Canonical alias for /v1/train/grpo after the ECHO trajectory
        // schema landing. The "agentic" name reflects what the endpoint
        // actually trains: multi-turn rollouts with action/observation
        // segments. Both routes serve the same handler; legacy callers
        // keep working unchanged.
        .route(
            "/v1/train/agentic",
            post(submit_grpo).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/v1/train/opd",
            post(submit_opd).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/v1/distill/refresh",
            post(submit_distill_refresh).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/v1/adapters/distill_merge",
            post(submit_distill_merge).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/v1/distill/pump",
            post(submit_distill_pump).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/v1/distill/self",
            post(submit_distill_self).layer(DefaultBodyLimit::disable()),
        )
        .route("/v1/train/status", get(training_status))
        .route("/v1/train/status/{job_id}", get(job_status))
        .route(
            "/v1/train/jobs/{job_id}",
            get(job_detail).delete(delete_archived_job),
        )
        .route("/v1/train/queue", get(list_queue))
        .route("/v1/train/queue/{job_id}", delete(cancel_queued_job))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_train::opd::StableOpdMode;
    use kiln_train::opd::{OpdConfig, OpdPrompt};
    use kiln_train::{
        ChatMessage, GrpoConfig, OpdLossGranularity, OpdObjective, ScoredCompletion, SftConfig,
    };
    use std::sync::{Arc, Barrier, Mutex, RwLock};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn pending_sft_job(job_id: impl Into<String>) -> (TrainingJobInfo, QueueEntry) {
        let job_id = job_id.into();
        let info = TrainingJobInfo {
            job_id: job_id.clone(),
            adapter_name: format!("adapter-{job_id}"),
            job_type: TrainingJobType::Sft,
            state: TrainingState::Queued,
            progress: 0.0,
            loss: None,
            epoch: None,
            adapter_path: None,
            submitted_at: std::time::Instant::now(),
            submitted_unix_ms: crate::recent_requests::now_unix_ms(),
            auto_load: false,
            consumed_correction_ids: Vec::new(),
            finished_at: None,
            finished_unix_ms: None,
            error: None,
            linked_eval_job_ids: Vec::new(),
            post_eval_verdict: None,
            gate_outcome: None,
            loss_history: Vec::new(),
            cancel_requested: Default::default(),
        };
        let request = SftRequest {
            examples: Vec::new(),
            dataset_path: None,
            dataset: None,
            config: SftConfig::default(),
            post_eval: None,
        };
        (
            info,
            QueueEntry {
                job_id,
                reserved_bytes: 0,
                job: QueuedJob::Sft(request),
            },
        )
    }

    #[test]
    fn concurrent_batch_admission_is_capacity_atomic() {
        let tracked: crate::state::TrainingJobs = Arc::new(RwLock::new(Default::default()));
        let queue = crate::training_queue::new_shared_queue();
        let start = Arc::new(Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|batch| {
                let tracked = tracked.clone();
                let queue = queue.clone();
                let start = start.clone();
                std::thread::spawn(move || {
                    let pending = vec![
                        pending_sft_job(format!("batch-{batch}-a")),
                        pending_sft_job(format!("batch-{batch}-b")),
                    ];
                    start.wait();
                    admit_training_jobs_into(&tracked, &queue, 3, 3, true, pending)
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("admission thread panicked"))
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);

        let tracked = tracked.read().unwrap();
        let queue = queue.lock().unwrap();
        assert_eq!(tracked.len(), 2);
        assert_eq!(queue.len(), 2);
        assert!(
            queue
                .queue
                .iter()
                .all(|entry| tracked.contains_key(&entry.job_id))
        );
    }

    #[test]
    fn concurrent_single_submission_cannot_bypass_batch_capacity() {
        let tracked: crate::state::TrainingJobs = Arc::new(RwLock::new(Default::default()));
        let queue = crate::training_queue::new_shared_queue();
        let start = Arc::new(Barrier::new(2));

        let batch_handle = {
            let tracked = tracked.clone();
            let queue = queue.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                admit_training_jobs_into(
                    &tracked,
                    &queue,
                    2,
                    2,
                    true,
                    vec![pending_sft_job("batch-a"), pending_sft_job("batch-b")],
                )
            })
        };
        let single_handle = {
            let tracked = tracked.clone();
            let queue = queue.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                admit_training_jobs_into(
                    &tracked,
                    &queue,
                    2,
                    2,
                    true,
                    vec![pending_sft_job("single")],
                )
            })
        };

        let batch_result = batch_handle.join().expect("batch thread panicked");
        let single_result = single_handle.join().expect("single thread panicked");
        assert_ne!(batch_result.is_ok(), single_result.is_ok());

        let tracked = tracked.read().unwrap();
        let queue = queue.lock().unwrap();
        let expected_len = if batch_result.is_ok() { 2 } else { 1 };
        assert_eq!(tracked.len(), expected_len);
        assert_eq!(queue.len(), expected_len);
        assert!(queue.len() <= 2);
    }

    #[test]
    fn rejected_batch_leaves_tracking_and_queue_unchanged() {
        let tracked: crate::state::TrainingJobs = Arc::new(RwLock::new(Default::default()));
        let queue = crate::training_queue::new_shared_queue();
        let error = admit_training_jobs_into(
            &tracked,
            &queue,
            10,
            1,
            true,
            vec![pending_sft_job("a"), pending_sft_job("b")],
        )
        .unwrap_err();

        assert_eq!(error.code, "training_tracked_full");
        assert!(tracked.read().unwrap().is_empty());
        assert_eq!(queue.lock().unwrap().len(), 0);
    }

    #[test]
    fn single_job_admission_preserves_queue_position_semantics() {
        let tracked: crate::state::TrainingJobs = Arc::new(RwLock::new(Default::default()));
        let queue = crate::training_queue::new_shared_queue();

        let position = admit_training_jobs_into(
            &tracked,
            &queue,
            2,
            2,
            true,
            vec![pending_sft_job("single")],
        )
        .unwrap();

        assert_eq!(position, 1);
        assert!(tracked.read().unwrap().contains_key("single"));
        let queue = queue.lock().unwrap();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.queue.front().unwrap().job_id, "single");
    }

    #[test]
    fn sampled_token_opd_is_rejected_at_submission() {
        let err = validate_opd_loss_at_submit(OpdLossGranularity::SampledToken).unwrap_err();
        assert!(err.to_string().contains("identically zero"), "{err}");
        validate_opd_loss_at_submit(OpdLossGranularity::TeacherTopK).unwrap();
        let err = validate_opd_loss_at_submit(OpdLossGranularity::FullVocab).unwrap_err();
        assert!(err.to_string().contains("no concrete"), "{err}");
    }

    #[test]
    fn unwired_opd_semantics_are_rejected_at_submission() {
        let mut config = OpdConfig::default();
        validate_opd_config_at_submit(&config).unwrap();

        config.objective = OpdObjective::CrossEntropy;
        assert!(
            validate_opd_config_at_submit(&config)
                .unwrap_err()
                .to_string()
                .contains("cross_entropy")
        );

        config = OpdConfig::default();
        config.stable_opd = StableOpdMode::Auto;
        assert!(
            validate_opd_config_at_submit(&config)
                .unwrap_err()
                .to_string()
                .contains("Stable-OPD")
        );
    }

    #[test]
    fn alternate_opd_admission_rejects_empty_and_unscored_off_policy_work() {
        let mut request = OpdRequest {
            prompts: Vec::new(),
            dataset_path: None,
            teacher: "teacher".into(),
            config: OpdConfig::default(),
            post_eval: None,
        };
        assert!(validate_opd_request_at_submit(&request).is_err());

        request.config.training_mode = kiln_train::opd::OpdTrainingMode::OffPolicy;
        request.prompts = vec![OpdPrompt {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "question".into(),
            }],
            teacher_extra_messages: Vec::new(),
            trajectory: Vec::new(),
        }];
        let error = validate_opd_request_at_submit(&request).unwrap_err();
        assert!(error.to_string().contains("assistant action"), "{error}");
    }

    #[test]
    fn fixed_fixture_request_defaults_are_off_policy_and_self_requires_actions() {
        let merge: kiln_train::DistillMergeRequest =
            serde_json::from_str(r#"{"name":"merged","sources":[{"adapter":"source"}]}"#).unwrap();
        assert!(matches!(
            merge.config.training_mode,
            kiln_train::opd::OpdTrainingMode::OffPolicy
        ));

        let mut self_request: kiln_train::DistillSelfRequest = serde_json::from_str(
            r#"{"name":"self","mode":"conciseness","prompts":[{"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer"}]}]}"#,
        )
        .unwrap();
        assert!(matches!(
            self_request.config.training_mode,
            kiln_train::opd::OpdTrainingMode::OffPolicy
        ));
        validate_distill_self_at_submit(&self_request).unwrap();

        self_request.prompts = None;
        let error = validate_distill_self_at_submit(&self_request).unwrap_err();
        assert!(error.to_string().contains("explicit off-policy prompts"));
    }

    #[test]
    fn pump_examples_are_validated_before_enqueue_in_both_training_modes() {
        let mut request: kiln_train::DistillPumpRequest =
            serde_json::from_str(r#"{"name":"pump","teacher":"teacher","mode":{"examples":[]}}"#)
                .unwrap();
        let error = validate_distill_pump_at_submit(&request).unwrap_err();
        assert!(error.to_string().contains("at least one prompt"), "{error}");

        request.mode = kiln_train::DistillPumpMode::Examples {
            examples: vec![OpdPrompt {
                messages: Vec::new(),
                teacher_extra_messages: Vec::new(),
                trajectory: Vec::new(),
            }],
        };
        let error = validate_distill_pump_at_submit(&request).unwrap_err();
        assert!(error.to_string().contains("has no messages"), "{error}");

        request.mode = kiln_train::DistillPumpMode::Examples {
            examples: vec![OpdPrompt {
                messages: vec![ChatMessage {
                    role: "user".into(),
                    content: "question".into(),
                }],
                teacher_extra_messages: Vec::new(),
                trajectory: Vec::new(),
            }],
        };
        validate_distill_pump_at_submit(&request).unwrap();

        request.config.training_mode = kiln_train::opd::OpdTrainingMode::OffPolicy;
        let error = validate_distill_pump_at_submit(&request).unwrap_err();
        assert!(error.to_string().contains("assistant action"), "{error}");
    }

    #[test]
    fn opd_top_k_is_resolved_to_the_executable_kernel_envelope() {
        let mut config = OpdConfig::default();
        assert_eq!(
            resolve_opd_top_k_at_submit(&mut config, 20).unwrap(),
            Some((32, 16))
        );
        assert_eq!(config.top_k, 16);

        config.top_k = 32;
        assert_eq!(resolve_opd_top_k_at_submit(&mut config, 32).unwrap(), None);
        assert_eq!(config.top_k, 32);

        config.top_k = 15;
        let error = resolve_opd_top_k_at_submit(&mut config, 20).unwrap_err();
        assert!(error.to_string().contains("not executable"), "{error}");
    }

    #[test]
    fn stock_vllm_registration_limits_opd_to_its_default_twenty() {
        let mut spec = super::super::teachers::TeacherSpec {
            alias: "remote".into(),
            kind: super::super::teachers::TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "model".into(),
            max_top_k: None,
            vocab_size: Some(1024),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            url: Some("http://vllm.local".into()),
            api_key_env: None,
            notes: None,
            adapter: None,
        };
        assert_eq!(registered_teacher_top_k_limit(&spec, 32), 20);
        spec.max_top_k = Some(32);
        assert_eq!(registered_teacher_top_k_limit(&spec, 32), 32);
    }

    #[test]
    fn self_distill_privileged_modes_require_one_nonempty_context_per_prompt() {
        let prompts = vec![OpdPrompt {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "question".into(),
            }],
            teacher_extra_messages: vec![],
            trajectory: vec![],
        }];
        let mut req = kiln_train::DistillSelfRequest {
            name: "self-test".into(),
            mode: kiln_train::SelfDistillMode::GroundTruthConditioning,
            prompts: Some(prompts.clone()),
            ground_truth: None,
            documents: None,
            config: OpdConfig::default(),
            post_eval: None,
        };
        assert!(validate_self_distill_context_at_submit(&req).is_err());
        req.ground_truth = Some(vec!["  ".into()]);
        assert!(validate_self_distill_context_at_submit(&req).is_err());
        req.ground_truth = Some(vec!["answer".into()]);
        validate_self_distill_context_at_submit(&req).unwrap();

        req.mode = kiln_train::SelfDistillMode::DocumentAsPi;
        req.ground_truth = None;
        assert!(validate_self_distill_context_at_submit(&req).is_err());
        req.documents = Some(vec!["context".into()]);
        validate_self_distill_context_at_submit(&req).unwrap();

        req.mode = kiln_train::SelfDistillMode::Conciseness;
        req.prompts = None;
        req.documents = None;
        validate_self_distill_context_at_submit(&req).unwrap();
    }

    fn grpo_group() -> GrpoGroup {
        GrpoGroup {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "prompt".to_string(),
            }],
            completions: vec![ScoredCompletion {
                text: "completion".to_string(),
                reward: 1.0,
                ..Default::default()
            }],
        }
    }

    fn grpo_req(dataset_path: Option<&str>, groups: Vec<GrpoGroup>) -> GrpoRequest {
        GrpoRequest {
            dataset: None,
            groups,
            dataset_path: dataset_path.map(str::to_string),
            config: GrpoConfig::default(),
            post_eval: None,
        }
    }

    #[test]
    fn server_preflight_activation_width_matches_gdn_trainer_tape() {
        assert_eq!(training_activation_bytes_per_elem(2, false, false), 2);
        assert_eq!(training_activation_bytes_per_elem(2, false, true), 10);
        assert_eq!(training_activation_bytes_per_elem(2, true, true), 4);
    }

    #[test]
    fn server_preflight_streaming_policy_uses_runtime_backend_device() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior_streaming = std::env::var("KILN_STREAMING_PREFILL").ok();
        let prior_streaming_tile = std::env::var("KILN_STREAMING_TILE_TOKENS").ok();
        let prior_tape_tile = std::env::var("KILN_TAPE_STREAMING_TILE_TOKENS").ok();
        unsafe {
            std::env::remove_var("KILN_STREAMING_PREFILL");
            std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            std::env::remove_var("KILN_TAPE_STREAMING_TILE_TOKENS");
        }

        let max_seq_len = 104_412;
        let rocm = training_activation_estimate_for_runtime_device(
            2,
            false,
            true,
            kiln_tensor::Device::Rocm(0),
            max_seq_len,
        );
        let cpu_storage = training_activation_estimate_for_runtime_device(
            2,
            false,
            true,
            kiln_tensor::Device::Cpu,
            max_seq_len,
        );

        unsafe {
            if let Some(value) = prior_streaming {
                std::env::set_var("KILN_STREAMING_PREFILL", value);
            } else {
                std::env::remove_var("KILN_STREAMING_PREFILL");
            }
            if let Some(value) = prior_streaming_tile {
                std::env::set_var("KILN_STREAMING_TILE_TOKENS", value);
            } else {
                std::env::remove_var("KILN_STREAMING_TILE_TOKENS");
            }
            if let Some(value) = prior_tape_tile {
                std::env::set_var("KILN_TAPE_STREAMING_TILE_TOKENS", value);
            } else {
                std::env::remove_var("KILN_TAPE_STREAMING_TILE_TOKENS");
            }
        }

        assert_eq!(rocm.streaming_gdn_tile_tokens, Some(1024));
        assert_eq!(cpu_storage.streaming_gdn_tile_tokens, None);
    }

    #[test]
    fn dynamic_training_available_counts_allocator_and_reclaimable_kv() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            combine_training_available_bytes(
                21 * gb,
                Some(80 * gb),
                8 * gb,
                120 * gb,
                gb,
                VramSource::NvidiaSmi,
            ),
            88 * gb
        );
    }

    #[test]
    fn dynamic_training_available_does_not_trust_allocator_over_live_unified_memory() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            combine_training_available_bytes(
                21 * gb,
                Some(80 * gb),
                8 * gb,
                120 * gb,
                gb,
                VramSource::LinuxDrmSysfsUnified,
            ),
            29 * gb
        );
    }

    #[test]
    fn dynamic_training_available_is_capped_by_total_minus_floor() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            combine_training_available_bytes(
                21 * gb,
                Some(118 * gb),
                8 * gb,
                120 * gb,
                gb,
                VramSource::NvidiaSmi,
            ),
            119 * gb
        );
    }

    #[test]
    fn auto_mode_reservation_uses_max_checkpointed_shape_for_long_rows() {
        let cfg = kiln_core::config::ModelConfig::qwen3_5_4b();
        let gb = 1024 * 1024 * 1024;
        let vram = kiln_memory::vram::GpuVramInfo {
            total_bytes: 120 * gb,
            source: VramSource::LinuxDrmSysfsUnified,
        };
        let options = EstimateOptions {
            max_supervised_tokens: Some(512),
            recompute_boundaries: true,
            activation_bytes_per_elem: Some(10),
            streaming_gdn_tile_tokens: Some(1024),
        };
        let max_seq_len = 104_412;
        let one_segment = training_preflight::estimate_step_working_set_with_options(
            &cfg,
            max_seq_len,
            8,
            1,
            WeightResidency::for_vram_source(vram.source),
            true,
            options,
        );
        let reservation_segments = auto_mode_reservation_segments(&cfg, 1);
        assert_eq!(
            reservation_segments, cfg.num_layers,
            "auto-mode queue accounting should reserve the largest row with maximum checkpointing"
        );
        let reserved = training_preflight::estimate_step_working_set_with_options(
            &cfg,
            max_seq_len,
            8,
            reservation_segments,
            WeightResidency::for_vram_source(vram.source),
            true,
            options,
        );
        assert!(
            reserved.total_bytes < one_segment.total_bytes,
            "reservation should shrink when runtime-style checkpointing engages"
        );
    }

    #[test]
    fn grpo_dataset_path_submission_allows_generic_streaming_route() {
        let req = grpo_req(Some("/tmp/grpo.jsonl"), Vec::new());
        validate_grpo_submission_source(&req).unwrap();
    }

    #[test]
    fn grpo_submission_rejects_ambiguous_or_empty_sources() {
        let both = grpo_req(Some("/tmp/grpo.jsonl"), vec![grpo_group()]);
        let err = validate_grpo_submission_source(&both).unwrap_err();
        assert!(err.message.contains("either groups or dataset_path"));

        let empty = grpo_req(None, Vec::new());
        let err = validate_grpo_submission_source(&empty).unwrap_err();
        assert!(err.message.contains("non-empty groups or dataset_path"));

        let inline = grpo_req(None, vec![grpo_group()]);
        validate_grpo_submission_source(&inline).unwrap();
    }

    /// ECHO env-CE trains again (resurrection PR2), so echo-enabled
    /// submissions with Observation segments now pass validation — the
    /// flagship agentic shape. Still rejected at submission (not at worker
    /// dequeue hours later): no_policy_loss (not yet re-wired) and the
    /// reserved OPD slot.
    #[test]
    fn grpo_submission_validates_loss_configs() {
        // ECHO + a rollout carrying an Observation segment.
        let mut group = grpo_group();
        group.completions[0].trajectory = vec![
            kiln_train::trajectory::TurnSegment {
                role: "assistant".into(),
                content: "running".into(),
                kind: kiln_train::trajectory::TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            kiln_train::trajectory::TurnSegment {
                role: "tool".into(),
                content: "exit 0".into(),
                kind: kiln_train::trajectory::TurnKind::Observation,
                tool_call_id: None,
                warning_prefix_len: None,
            },
        ];
        let mut req = grpo_req(None, vec![group.clone()]);
        req.config.loss.echo = Some(kiln_train::EchoConfig::default());
        validate_grpo_submission_source(&req)
            .expect("echo + observation segments is the flagship agentic shape");

        // Same data WITHOUT echo: also fine — the policy loss trains the
        // trajectory's action tokens.
        let req = grpo_req(None, vec![group]);
        validate_grpo_submission_source(&req).unwrap();

        // ECHO on legacy single-turn rollouts: zero env term, harmless.
        let mut req = grpo_req(None, vec![grpo_group()]);
        req.config.loss.echo = Some(kiln_train::EchoConfig::default());
        validate_grpo_submission_source(&req).unwrap();

        // no_policy_loss + default ECHO = §5.5 verifier-free mode: valid.
        let mut req = grpo_req(None, vec![grpo_group()]);
        req.config.loss.no_policy_loss = true;
        validate_grpo_submission_source(&req).expect("verifier-free mode validates");
        // Without ECHO there is nothing to train on — still rejected.
        req.config.loss.echo = None;
        let err = validate_grpo_submission_source(&req).unwrap_err();
        assert!(err.message.contains("no_policy_loss"), "{}", err.message);
    }

    #[test]
    fn grpo_dataset_path_submission_validation_is_head_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grpo.jsonl");
        let first = serde_json::to_string(&grpo_group()).unwrap();
        std::fs::write(&path, format!("{first}\nthis is not json\n")).unwrap();

        let stats = validate_grpo_jsonl_submission_head(path.to_str().unwrap(), None).unwrap();
        assert!(stats.streaming_dataset);
        assert_eq!(stats.num_groups, None);
        assert_eq!(stats.total_completions, None);
        assert!(stats.max_seq_len > 0);
    }

    fn opd_request_payload() -> OpdRequest {
        OpdRequest {
            prompts: vec![OpdPrompt {
                messages: vec![ChatMessage {
                    role: "user".into(),
                    content: "Solve 5x + 7 = 22".into(),
                }],
                teacher_extra_messages: vec![],
                trajectory: vec![],
            }],
            dataset_path: None,
            teacher: "qwen3.6-27b@local".into(),
            config: OpdConfig::default(),
            post_eval: None,
        }
    }

    #[test]
    fn opd_request_serde_round_trip_carries_grand_plan_defaults() {
        let req = opd_request_payload();
        let json = serde_json::to_string(&req).expect("serialize");
        let parsed: OpdRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.teacher, "qwen3.6-27b@local");
        assert_eq!(parsed.config.top_k, 32);
        assert_eq!(parsed.config.samples_per_prompt, 4);
        assert!((parsed.config.top_p - 0.9).abs() < 1e-9);
        assert!(matches!(
            parsed.config.loss,
            OpdLossGranularity::TeacherTopK
        ));
        assert_eq!(parsed.config.max_tokens, 7168);
    }

    #[test]
    fn opd_request_accepts_dataset_path_in_place_of_prompts() {
        // A streaming-dataset payload — no inline prompts but a
        // `dataset_path` set. The `submit_opd` handler treats this as
        // valid; tested at the wire level.
        let json = r#"{"prompts":[],"dataset_path":"/tmp/opd.jsonl","teacher":"qwen3.6-27b@vllm"}"#;
        let req: OpdRequest = serde_json::from_str(json).unwrap();
        assert!(req.prompts.is_empty());
        assert_eq!(req.dataset_path.as_deref(), Some("/tmp/opd.jsonl"));
        assert_eq!(req.teacher, "qwen3.6-27b@vllm");
    }

    #[test]
    fn opd_request_rejects_unknown_loss_granularity() {
        let json = r#"{"prompts":[],"teacher":"x","config":{"loss":"sampled_lobotomy"}}"#;
        let parsed: Result<OpdRequest, _> = serde_json::from_str(json);
        assert!(
            parsed.is_err(),
            "unknown loss value should fail to deserialize"
        );
    }

    #[test]
    fn distill_refresh_request_minimal_json_parses() {
        let json = r#"{
            "name": "company-assistant",
            "new_data": {"dataset": "q4-2026"},
            "behavioural_teacher": "company-assistant@v17"
        }"#;
        let req: DistillRefreshRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "company-assistant");
        assert_eq!(req.behavioural_teacher, "company-assistant@v17");
        // Defaults populate.
        assert_eq!(req.background_chat, "tulu3");
        assert!((req.require_if_eval_recovery - 0.95).abs() < 1e-9);
        assert!((req.require_internal_qa_gain - 0.05).abs() < 1e-9);
    }
}
