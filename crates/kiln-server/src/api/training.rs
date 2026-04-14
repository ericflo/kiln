//! Training API endpoints — pure Rust, in-process LoRA training.
//!
//! Training requests are enqueued in a FIFO queue and executed sequentially
//! by a background worker. This prevents GPU memory conflicts between
//! concurrent training jobs.

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};

use kiln_train::{GrpoRequest, SftRequest, TrainingResponse, TrainingState, TrainingStatus};

use crate::state::{AppState, ModelBackend, TrainingJobInfo, TrainingJobType};
use crate::training_queue::{QueueEntry, QueuedJob};

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
) -> Result<Json<TrainingResponse>, (StatusCode, String)> {
    let num_examples = req.examples.len();
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("sft-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    tracing::info!(num_examples, job_id = %job_id, adapter = %adapter_name, "SFT training request queued");

    // Verify we have real model weights
    if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "training requires real model weights (not available in mock mode)".to_string(),
        ));
    }

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
    Json(req): Json<GrpoRequest>,
) -> Result<Json<TrainingResponse>, (StatusCode, String)> {
    let num_groups = req.groups.len();
    let total_completions: usize = req.groups.iter().map(|g| g.completions.len()).sum();
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("grpo-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    tracing::info!(num_groups, total_completions, job_id = %job_id, adapter = %adapter_name, "GRPO training request queued");

    // Verify we have real model weights
    if matches!(state.backend.as_ref(), ModelBackend::Mock { .. }) {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "training requires real model weights (not available in mock mode)".to_string(),
        ));
    }

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
        message: format!(
            "Queued GRPO training with {num_groups} groups ({total_completions} completions, position {queue_position} in queue)"
        ),
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
) -> Result<Json<TrainingStatus>, (StatusCode, String)> {
    let jobs = state.training_jobs.read().unwrap();
    let job = jobs.get(&job_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("training job not found: {job_id}"),
        )
    })?;

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
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Check if the job exists and is in Queued state
    {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs.get(&job_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("training job not found: {job_id}"),
            )
        })?;
        if job.state != TrainingState::Queued {
            return Err((
                StatusCode::CONFLICT,
                format!(
                    "cannot cancel job {job_id}: state is {:?} (only queued jobs can be cancelled)",
                    job.state
                ),
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
        let mut jobs = state.training_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id) {
            job.state = TrainingState::Failed;
        }
        Ok(Json(serde_json::json!({
            "job_id": job_id,
            "status": "cancelled"
        })))
    } else {
        Err((
            StatusCode::CONFLICT,
            format!("job {job_id} was not found in queue (may have already started)"),
        ))
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/train/sft", post(submit_sft))
        .route("/v1/train/grpo", post(submit_grpo))
        .route("/v1/train/status", get(training_status))
        .route("/v1/train/status/{job_id}", get(job_status))
        .route("/v1/train/queue", get(list_queue))
        .route("/v1/train/queue/{job_id}", delete(cancel_queued_job))
}
