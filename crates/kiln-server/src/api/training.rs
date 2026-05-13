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

use kiln_core::env_flag::env_tristate;
use kiln_train::{
    GrpoGroup, GrpoRequest, SftRequest, TrainingResponse, TrainingState, TrainingStatus,
};

use std::sync::atomic::Ordering;

use crate::error::ApiError;
use crate::metrics::{TrainingMetricStatus, TrainingMetricType};
use crate::state::{AppState, ModelBackend, TrainingJobInfo, TrainingJobType};
use crate::training_preflight::{
    self, EstimateOptions, WeightResidency, available_for_training_bytes,
    estimate_step_working_set_with_options, estimate_vk_native_recompute_working_set,
    format_oom_message_with_source,
};
use crate::training_queue::{QueueEntry, QueuedJob};

struct GrpoSubmissionStats {
    num_groups: Option<usize>,
    total_completions: Option<usize>,
    max_seq_len: usize,
    streaming_dataset: bool,
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
fn enforce_training_preflight(
    state: &AppState,
    max_seq_len: usize,
    options: EstimateOptions,
    lora_rank: usize,
    vk_native_recompute: bool,
) -> Result<(), ApiError> {
    if max_seq_len == 0 {
        return Ok(());
    }
    let vram = kiln_core::vram::detect_vram();
    let available = available_for_training_bytes(&vram);
    if available == u64::MAX {
        // No memory signal at all — let the trainer be the line of
        // defense. Better than rejecting every submission on machines
        // where detection is misconfigured.
        return Ok(());
    }
    let num_segments =
        kiln_train::CheckpointConfig::from_env(state.model_config.num_layers).num_segments;
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
        kiln_core::vram::VramSource::LinuxDrmSysfsUnified
            | kiln_core::vram::VramSource::AppleSilicon
            | kiln_core::vram::VramSource::EnvOverride
    );
    let estimate = if vk_native_recompute {
        estimate_vk_native_recompute_working_set(
            &state.model_config,
            max_seq_len,
            lora_rank,
            residency,
            weights_already_resident,
        )
    } else {
        estimate_step_working_set_with_options(
            &state.model_config,
            max_seq_len,
            lora_rank,
            num_segments,
            residency,
            weights_already_resident,
            options,
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
    Ok(())
}

fn vk_native_sft_enabled(state: &AppState) -> bool {
    match env_tristate("KILN_VK_NATIVE_TRAINING") {
        Some(enabled) => enabled,
        None => {
            #[cfg(feature = "vulkan")]
            {
                let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
                    return false;
                };
                runner.read().unwrap().backend_name() == "vulkan"
            }
            #[cfg(not(feature = "vulkan"))]
            {
                let _ = state;
                false
            }
        }
    }
}

fn vk_native_grpo_enabled(state: &AppState) -> bool {
    match env_tristate("KILN_VK_NATIVE_GRPO") {
        Some(enabled) => enabled,
        None => vk_native_sft_enabled(state),
    }
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

async fn submit_sft(
    State(state): State<AppState>,
    Json(req): Json<SftRequest>,
) -> Result<Json<TrainingResponse>, ApiError> {
    // Reject new jobs during shutdown
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }

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

    let num_examples = req.examples.len();
    let job_id = uuid::Uuid::new_v4().to_string();
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
    enforce_training_preflight(
        &state,
        max_seq_len,
        EstimateOptions {
            max_supervised_tokens: Some(max_supervised_tokens),
            recompute_boundaries: training_preflight::recompute_checkpoint_boundaries_for_seq_len(
                max_seq_len,
            ),
        },
        req.config.lora_rank,
        vk_native_sft_enabled(&state)
            && state.model_config.num_full_attention_layers < state.model_config.num_layers,
    )?;

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
        auto_load,
        finished_at: None,
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
    let vk_native_grpo = vk_native_grpo_enabled(&state);
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

    // Working-set preflight (see submit_sft for rationale).
    let max_seq_len = stats.max_seq_len;
    enforce_training_preflight(
        &state,
        max_seq_len,
        EstimateOptions::default(),
        req.config.lora_rank,
        vk_native_grpo,
    )?;

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
        auto_load,
        finished_at: None,
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

/// GET /v1/train/status — overall training status (list all tracked jobs).
async fn training_status(State(state): State<AppState>) -> Json<Vec<TrainingStatus>> {
    let jobs = state.training_jobs.read().unwrap();
    let statuses: Vec<TrainingStatus> = jobs
        .values()
        .map(|j| TrainingStatus {
            job_id: j.job_id.clone(),
            state: j.state,
            progress: j.progress,
            current_loss: j.loss,
            adapter_name: Some(j.adapter_name.clone()),
            started_at: format!("{}s ago", j.submitted_at.elapsed().as_secs()),
            elapsed_secs: j.submitted_at.elapsed().as_secs_f64(),
        })
        .collect();
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

    Ok(Json(TrainingStatus {
        job_id: job.job_id.clone(),
        state: job.state,
        progress: job.progress,
        current_loss: job.loss,
        adapter_name: Some(job.adapter_name.clone()),
        started_at: format!("{}s ago", job.submitted_at.elapsed().as_secs()),
        elapsed_secs: job.submitted_at.elapsed().as_secs_f64(),
    }))
}

/// GET /v1/train/queue — list queue contents organized by state.
async fn list_queue(State(state): State<AppState>) -> Json<QueueResponse> {
    let jobs = state.training_jobs.read().unwrap();
    let queue = state.training_queue.lock().unwrap();

    let mut running = None;
    let mut completed = Vec::new();

    for j in jobs.values() {
        let status = TrainingStatus {
            job_id: j.job_id.clone(),
            state: j.state,
            progress: j.progress,
            current_loss: j.loss,
            adapter_name: Some(j.adapter_name.clone()),
            started_at: format!("{}s ago", j.submitted_at.elapsed().as_secs()),
            elapsed_secs: j.submitted_at.elapsed().as_secs_f64(),
        };
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

    // Sort completed by most recent first
    completed.sort_by(|a, b| a.elapsed_secs.partial_cmp(&b.elapsed_secs).unwrap());

    Json(QueueResponse {
        running,
        queued,
        completed,
    })
}

/// DELETE /v1/train/queue/:job_id — cancel a queued job.
async fn cancel_queued_job(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Check if the job exists and is in Queued state
    {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| ApiError::training_job_not_found(&job_id))?;
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
                job.finished_at = Some(std::time::Instant::now());
            }
            jt
        };
        if let Some(jt) = metric_type {
            let mt = match jt {
                TrainingJobType::Sft => TrainingMetricType::Sft,
                TrainingJobType::Grpo => TrainingMetricType::Grpo,
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

/// Body-size cap for SFT training submissions (audit LOW §1).
/// 64 MiB accommodates long-context training examples that exceed the 2 MiB axum default.
const SFT_BODY_LIMIT: usize = 64 * 1024 * 1024;
/// Body-size cap for GRPO training submissions (audit LOW §1).
/// 64 MiB accommodates batches of scored completions.
const GRPO_BODY_LIMIT: usize = 64 * 1024 * 1024;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/v1/train/sft",
            post(submit_sft).layer(DefaultBodyLimit::max(SFT_BODY_LIMIT)),
        )
        .route(
            "/v1/train/grpo",
            post(submit_grpo).layer(DefaultBodyLimit::max(GRPO_BODY_LIMIT)),
        )
        .route("/v1/train/status", get(training_status))
        .route("/v1/train/status/{job_id}", get(job_status))
        .route("/v1/train/queue", get(list_queue))
        .route("/v1/train/queue/{job_id}", delete(cancel_queued_job))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_train::{ChatMessage, GrpoConfig, ScoredCompletion};

    fn grpo_group() -> GrpoGroup {
        GrpoGroup {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "prompt".to_string(),
            }],
            completions: vec![ScoredCompletion {
                text: "completion".to_string(),
                reward: 1.0,
            }],
        }
    }

    fn grpo_req(dataset_path: Option<&str>, groups: Vec<GrpoGroup>) -> GrpoRequest {
        GrpoRequest {
            groups,
            dataset_path: dataset_path.map(str::to_string),
            config: GrpoConfig::default(),
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
}
