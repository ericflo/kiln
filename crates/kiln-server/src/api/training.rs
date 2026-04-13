use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use kiln_train::{GrpoRequest, SftRequest, TrainingResponse, TrainingState, TrainingStatus};

use crate::sidecar::{SidecarClient, SidecarResponse};
use crate::state::{AppState, TrainingJobInfo, TrainingJobType};

fn sidecar_or_503(sidecar: &Option<SidecarClient>) -> Result<&SidecarClient, (StatusCode, String)> {
    sidecar.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "training sidecar not configured — start kiln with KILN_SIDECAR_SOCKET to enable training".to_string(),
        )
    })
}

async fn submit_sft(
    State(state): State<AppState>,
    Json(req): Json<SftRequest>,
) -> Result<Json<TrainingResponse>, (StatusCode, String)> {
    let client = sidecar_or_503(&state.sidecar)?;

    let num_examples = req.examples.len();
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("sft-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    tracing::info!(num_examples, job_id = %job_id, "SFT training request received");

    // Serialize examples as JSON values for the sidecar protocol
    let examples: Vec<serde_json::Value> = req
        .examples
        .iter()
        .map(|ex| serde_json::to_value(ex).unwrap_or_default())
        .collect();

    let resp = client
        .submit_sft(
            &job_id,
            examples,
            &adapter_name,
            req.config.epochs,
            req.config.learning_rate,
            req.config.lora_rank,
            req.config.lora_alpha,
        )
        .await
        .map_err(|e| {
            tracing::error!("sidecar error: {e}");
            (StatusCode::BAD_GATEWAY, format!("training sidecar error: {e}"))
        })?;

    match resp {
        SidecarResponse::JobAccepted { ref job_id, .. } => {
            // Track the job in shared state.
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

            Ok(Json(TrainingResponse {
                job_id: job_id.clone(),
                state: TrainingState::Queued,
                message: format!("Queued SFT training with {num_examples} examples"),
            }))
        }
        SidecarResponse::Error { message, .. } => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("sidecar rejected job: {message}"),
        )),
        other => {
            tracing::warn!("unexpected sidecar response: {:?}", other);
            Ok(Json(TrainingResponse {
                job_id,
                state: TrainingState::Queued,
                message: format!("Queued SFT training with {num_examples} examples"),
            }))
        }
    }
}

async fn submit_grpo(
    State(state): State<AppState>,
    Json(req): Json<GrpoRequest>,
) -> Result<Json<TrainingResponse>, (StatusCode, String)> {
    let client = sidecar_or_503(&state.sidecar)?;

    let num_groups = req.groups.len();
    let total_completions: usize = req.groups.iter().map(|g| g.completions.len()).sum();
    let job_id = uuid::Uuid::new_v4().to_string();
    let adapter_name = req
        .config
        .output_name
        .clone()
        .unwrap_or_else(|| format!("grpo-{}", &job_id[..8]));
    let auto_load = req.config.auto_load;

    tracing::info!(num_groups, total_completions, job_id = %job_id, "GRPO training request received");

    let groups: Vec<serde_json::Value> = req
        .groups
        .iter()
        .map(|g| serde_json::to_value(g).unwrap_or_default())
        .collect();

    let resp = client
        .submit_grpo(
            &job_id,
            groups,
            &adapter_name,
            req.config.learning_rate,
            req.config.lora_rank,
            req.config.lora_alpha,
        )
        .await
        .map_err(|e| {
            tracing::error!("sidecar error: {e}");
            (StatusCode::BAD_GATEWAY, format!("training sidecar error: {e}"))
        })?;

    match resp {
        SidecarResponse::JobAccepted { ref job_id, .. } => {
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

            Ok(Json(TrainingResponse {
                job_id: job_id.clone(),
                state: TrainingState::Queued,
                message: format!(
                    "Queued GRPO training with {num_groups} groups ({total_completions} completions)"
                ),
            }))
        }
        SidecarResponse::Error { message, .. } => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("sidecar rejected job: {message}"),
        )),
        other => {
            tracing::warn!("unexpected sidecar response: {:?}", other);
            Ok(Json(TrainingResponse {
                job_id,
                state: TrainingState::Queued,
                message: format!(
                    "Queued GRPO training with {num_groups} groups ({total_completions} completions)"
                ),
            }))
        }
    }
}

/// GET /v1/train/status — overall training status (list all tracked jobs).
async fn training_status(
    State(state): State<AppState>,
) -> Json<Vec<TrainingStatus>> {
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
    // First check cached state.
    let cached = {
        let jobs = state.training_jobs.read().unwrap();
        jobs.get(&job_id).cloned()
    };

    if let Some(job) = cached {
        // For completed/failed jobs, return cached state directly.
        if job.state == TrainingState::Completed || job.state == TrainingState::Failed {
            return Ok(Json(TrainingStatus {
                job_id: job.job_id,
                state: job.state,
                progress: job.progress,
                current_loss: job.loss,
                adapter_name: Some(job.adapter_name),
                started_at: format!("{}s ago", job.submitted_at.elapsed().as_secs()),
                elapsed_secs: job.submitted_at.elapsed().as_secs_f64(),
            }));
        }

        // For active jobs, try to get live status from sidecar.
        if let Some(ref client) = state.sidecar {
            if let Ok(resp) = client.query_status(&job_id).await {
                match resp {
                    SidecarResponse::JobStatus {
                        state: ref sidecar_state,
                        progress,
                        epoch,
                        loss,
                        adapter_path,
                        ..
                    } => {
                        let live_state = match sidecar_state.as_str() {
                            "completed" => TrainingState::Completed,
                            "failed" => TrainingState::Failed,
                            "running" => TrainingState::Running,
                            _ => TrainingState::Queued,
                        };

                        // Update cached state.
                        {
                            let mut jobs = state.training_jobs.write().unwrap();
                            if let Some(j) = jobs.get_mut(&job_id) {
                                j.state = live_state;
                                j.progress = progress;
                                j.epoch = epoch;
                                j.loss = loss;
                                j.adapter_path = adapter_path;
                            }
                        }

                        return Ok(Json(TrainingStatus {
                            job_id: job.job_id,
                            state: live_state,
                            progress,
                            current_loss: loss,
                            adapter_name: Some(job.adapter_name),
                            started_at: format!("{}s ago", job.submitted_at.elapsed().as_secs()),
                            elapsed_secs: job.submitted_at.elapsed().as_secs_f64(),
                        }));
                    }
                    _ => {}
                }
            }
        }

        // Fallback to cached state.
        return Ok(Json(TrainingStatus {
            job_id: job.job_id,
            state: job.state,
            progress: job.progress,
            current_loss: job.loss,
            adapter_name: Some(job.adapter_name),
            started_at: format!("{}s ago", job.submitted_at.elapsed().as_secs()),
            elapsed_secs: job.submitted_at.elapsed().as_secs_f64(),
        }));
    }

    Err((
        StatusCode::NOT_FOUND,
        format!("training job not found: {job_id}"),
    ))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/train/sft", post(submit_sft))
        .route("/v1/train/grpo", post(submit_grpo))
        .route("/v1/train/status", get(training_status))
        .route("/v1/train/status/{job_id}", get(job_status))
}
