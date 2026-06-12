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

struct GrpoSubmissionStats {
    num_groups: Option<usize>,
    total_completions: Option<usize>,
    max_seq_len: usize,
    streaming_dataset: bool,
}

struct PreflightAdmission {
    reserved_bytes: u64,
    checkpoint_segments: Option<usize>,
}

fn model_dtype_bytes(dtype: kiln_core::config::DType) -> usize {
    match dtype {
        kiln_core::config::DType::BF16 | kiln_core::config::DType::FP16 => 2,
        kiln_core::config::DType::FP32 => 4,
    }
}

fn training_activation_bytes_per_elem_for_state(state: &AppState) -> usize {
    let base = model_dtype_bytes(state.model_config.dtype);
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return base;
    };
    let Ok(runner) = runner.read() else {
        tracing::warn!(
            "model runner lock poisoned while sizing training preflight; using model dtype width"
        );
        return base;
    };
    if runner
        .training_precision_policy()
        .uses_f32_activations_for_mixed_base_weights()
    {
        4
    } else {
        base
    }
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
    if options.activation_bytes_per_elem.is_none() {
        options.activation_bytes_per_elem =
            Some(training_activation_bytes_per_elem_for_state(state));
    }
    let vram = kiln_memory::vram::detect_vram();
    let available = available_for_training_bytes(&vram);
    if available == u64::MAX {
        // No memory signal at all — let the trainer be the line of
        // defense. Better than rejecting every submission on machines
        // where detection is misconfigured.
        return Ok(PreflightAdmission {
            reserved_bytes: 0,
            checkpoint_segments: None,
        });
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
            available,
        )
    };
    if estimate.total_bytes > available {
        let msg = format_oom_message_with_source(
            &estimate,
            available,
            lora_rank,
            num_segments,
            Some(vram.source),
        );
        return Err(ApiError::training_will_not_fit(msg));
    }
    Ok(PreflightAdmission {
        reserved_bytes: estimate.total_bytes,
        checkpoint_segments: Some(num_segments),
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
    if req.dataset.as_deref() == Some("corrections:active") {
        req.dataset = None;
        if !req.examples.is_empty() {
            return Err(ApiError::training_invalid_request(
                "SFT request must use either examples or dataset, not both",
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
                "SFT request must use either examples or dataset, not both",
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

    let num_examples = req.examples.len();
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
    let max_seq_len = training_preflight::approximate_max_seq_len_sft(
        &req.examples,
        Some(state.tokenizer.as_ref()),
    );
    let max_supervised_tokens = training_preflight::approximate_max_supervised_tokens_sft(
        &req.examples,
        Some(state.tokenizer.as_ref()),
    );
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
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(job_id.clone(), info);

    // Enqueue the job
    let queue_position = {
        let mut q = state.training_queue.lock().unwrap();
        q.push(QueueEntry {
            job_id: job_id.clone(),
            reserved_bytes,
            job: QueuedJob::Sft(req),
        });
        q.len() // position = queue length after push (1-indexed)
    };

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
        EstimateOptions::default(),
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
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(job_id.clone(), info);

    // Enqueue the job
    let queue_position = {
        let mut q = state.training_queue.lock().unwrap();
        q.push(QueueEntry {
            job_id: job_id.clone(),
            reserved_bytes,
            job: QueuedJob::Grpo(req),
        });
        q.len()
    };

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
/// defaults match the grand plan exactly (top_k=32, temperature=1.0,
/// top_p=0.9, max_tokens=7K, γ=0, Stable-OPD auto, etc.).
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
    super::teachers::require_registered_teacher(
        &state,
        &req.teacher,
        format!("OPD teacher alias '{}' is not registered", req.teacher),
    )?;
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
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(job_id.clone(), info);

    let queue_position = {
        let mut q = state.training_queue.lock().unwrap();
        q.push(QueueEntry {
            job_id: job_id.clone(),
            reserved_bytes, // #36: OPD working-set reservation (preflight estimate)
            job: QueuedJob::Opd(req),
        });
        q.len()
    };

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!("Queued OPD training (position {queue_position} in queue)."),
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
    Json(req): Json<DistillRefreshRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
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
    super::teachers::require_registered_teacher(
        &state,
        &req.behavioural_teacher,
        format!(
            "DistillRefresh: behavioural_teacher alias '{}' is not registered",
            req.behavioural_teacher
        ),
    )?;
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
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(job_id.clone(), info);

    let queue_position = {
        let mut q = state.training_queue.lock().unwrap();
        q.push(QueueEntry {
            job_id: job_id.clone(),
            reserved_bytes,
            job: QueuedJob::DistillRefresh(req),
        });
        q.len()
    };

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!("Queued distill/refresh (position {queue_position} in queue)."),
    }))
}

/// `POST /v1/adapters/distill_merge` — §3.4 behaviour-space merge.
async fn submit_distill_merge(
    State(state): State<AppState>,
    Json(req): Json<DistillMergeRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
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
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
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
    );
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: "Queued distill_merge.".to_string(),
    }))
}

/// `POST /v1/distill/pump` — §3.5 Knowledge Pump.
async fn submit_distill_pump(
    State(state): State<AppState>,
    Json(req): Json<DistillPumpRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    if req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/pump: `teacher` alias must be non-empty".to_string(),
        ));
    }
    super::teachers::require_registered_teacher(
        &state,
        &req.teacher,
        format!(
            "distill/pump: teacher alias '{}' is not registered",
            req.teacher
        ),
    )?;
    super::adapters::validate_adapter_name(&req.name)?;
    // The worker overrides config.lora_rank with the request's top-level
    // `rank` when set (training_queue.rs pump arm) — validate the rank
    // that will actually train, not the config default it shadows.
    validate_lora_scale_at_submit(
        req.rank.unwrap_or(req.config.lora_rank),
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
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
    );
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: "Queued distill/pump.".to_string(),
    }))
}

/// `POST /v1/distill/self` — §3.12 PI self-distillation.
async fn submit_distill_self(
    State(state): State<AppState>,
    Json(req): Json<DistillSelfRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

    validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    if req.name.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "distill/self: `name` must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    validate_lora_scale_at_submit(
        req.config.lora_rank,
        req.config.lora_alpha,
        req.config.allow_high_lora_scale,
    )?;
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
    );
    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: "Queued distill/self.".to_string(),
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
    let max_queued = state.max_queued_training_jobs;
    if state.training_queue.lock().unwrap().len() >= max_queued {
        return Err(ApiError::training_queue_full(max_queued));
    }
    let max_tracked = state.max_tracked_jobs;
    if state.training_jobs.read().unwrap().len() >= max_tracked {
        return Err(ApiError::training_tracked_full(max_tracked));
    }
    if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
        return Err(ApiError::mock_mode_no_training());
    }
    Ok(())
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
) {
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
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(job_id.to_string(), info);
    let mut q = state.training_queue.lock().unwrap();
    q.push(QueueEntry {
        job_id: job_id.to_string(),
        reserved_bytes,
        job,
    });
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
    use kiln_train::opd::{OpdConfig, OpdPrompt};
    use kiln_train::{ChatMessage, GrpoConfig, OpdLossGranularity, ScoredCompletion};

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
        let json =
            r#"{"prompts":[],"dataset_path":"/tmp/opd.jsonl","teacher":"qwen3.6-27b@openrouter"}"#;
        let req: OpdRequest = serde_json::from_str(json).unwrap();
        assert!(req.prompts.is_empty());
        assert_eq!(req.dataset_path.as_deref(), Some("/tmp/opd.jsonl"));
        assert_eq!(req.teacher, "qwen3.6-27b@openrouter");
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
