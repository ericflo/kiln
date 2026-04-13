//! Training API endpoints — pure Rust, in-process LoRA training.
//!
//! Training runs on a background thread sharing the already-loaded model weights.
//! No Python sidecar, no second model copy.

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use kiln_model::lora_loader::LoraWeights;
use kiln_train::{GrpoRequest, SftRequest, TrainingResponse, TrainingState, TrainingStatus};
use kiln_train::trainer;

use crate::state::{AppState, ModelBackend, TrainingJobInfo, TrainingJobType};

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

    tracing::info!(num_examples, job_id = %job_id, adapter = %adapter_name, "SFT training request received");

    // Extract model weights reference for training
    let runner_arc = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner.clone(),
        ModelBackend::Mock { .. } => {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "training requires real model weights (not available in mock mode)".to_string(),
            ));
        }
    };

    // Register the job
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

    // Spawn training on a background thread
    let training_jobs = state.training_jobs.clone();
    let model_config = state.model_config.clone();
    let tokenizer = state.tokenizer.clone();
    let adapter_dir = state.adapter_dir.clone();
    let active_adapter_name = state.active_adapter_name.clone();
    let job_id_clone = job_id.clone();
    let adapter_name_clone = adapter_name.clone();

    std::thread::spawn(move || {
        // Mark as running
        {
            let mut jobs = training_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id_clone) {
                job.state = TrainingState::Running;
            }
        }

        // Read model weights and config under a brief read lock
        let (weights_ref, _device, num_layers) = {
            let guard = runner_arc.read().unwrap();
            // We need to reference the weights for training. Since training
            // only READS base weights (LoRA params are separate Vars), we can
            // safely hold a read lock for the duration, OR we can extract what
            // we need and drop the lock. For simplicity and to not block
            // inference for the entire training run, we'll hold the read lock
            // only briefly here to get the device and config, then hold it
            // again during the actual forward passes.
            (
                // We need the runner Arc to access weights during training
                runner_arc.clone(),
                guard.weights.embed_tokens.device().clone(),
                guard.config.num_layers,
            )
        };

        // Set up progress callback
        let training_jobs_cb = training_jobs.clone();
        let job_id_cb = job_id_clone.clone();
        let progress_cb = Box::new(move |progress: trainer::TrainingProgress| {
            let mut jobs = training_jobs_cb.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id_cb) {
                job.progress = progress.progress;
                job.loss = Some(progress.loss);
                job.epoch = Some(progress.epoch as u32);
            }
        });

        // Run training — this blocks until complete.
        // We need to hold a read lock on the runner during training because
        // the forward pass needs access to the base model weights.
        let result = {
            let guard = runner_arc.read().unwrap();
            trainer::sft_train(
                &req.examples,
                &req.config,
                &model_config,
                &guard.weights,
                &tokenizer,
                &adapter_dir,
                &adapter_name_clone,
                Some(progress_cb),
            )
        };

        match result {
            Ok(adapter_path) => {
                let path_str = adapter_path.display().to_string();
                tracing::info!(
                    job_id = %job_id_clone,
                    adapter = %adapter_name_clone,
                    path = %path_str,
                    "SFT training completed"
                );

                // Update job state
                {
                    let mut jobs = training_jobs.write().unwrap();
                    if let Some(job) = jobs.get_mut(&job_id_clone) {
                        job.state = TrainingState::Completed;
                        job.progress = 1.0;
                        job.adapter_path = Some(path_str.clone());
                    }
                }

                // Auto-load the adapter if requested
                if auto_load {
                    if let Err(e) = auto_load_adapter(
                        &weights_ref,
                        &active_adapter_name,
                        &adapter_path,
                        &adapter_name_clone,
                        num_layers,
                    ) {
                        tracing::error!(
                            job_id = %job_id_clone,
                            adapter = %adapter_name_clone,
                            "auto-load failed: {e}"
                        );
                    } else {
                        tracing::info!(
                            job_id = %job_id_clone,
                            adapter = %adapter_name_clone,
                            "auto-loaded trained adapter"
                        );
                    }
                }
            }
            Err(e) => {
                tracing::error!(
                    job_id = %job_id_clone,
                    "SFT training failed: {e:#}"
                );
                let mut jobs = training_jobs.write().unwrap();
                if let Some(job) = jobs.get_mut(&job_id_clone) {
                    job.state = TrainingState::Failed;
                }
            }
        }
    });

    Ok(Json(TrainingResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!("Queued SFT training with {num_examples} examples"),
    }))
}

async fn submit_grpo(
    State(_state): State<AppState>,
    Json(req): Json<GrpoRequest>,
) -> Result<Json<TrainingResponse>, (StatusCode, String)> {
    let num_groups = req.groups.len();
    let total_completions: usize = req.groups.iter().map(|g| g.completions.len()).sum();
    let job_id = uuid::Uuid::new_v4().to_string();

    tracing::info!(num_groups, total_completions, job_id = %job_id, "GRPO training request received");

    // GRPO training is not yet implemented in the pure Rust trainer.
    // The types are defined and the endpoint accepts requests, but the
    // actual GRPO training loop (advantage normalization, clipped IS, KL penalty)
    // is Phase 4 work.
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "GRPO training is not yet implemented — SFT training is available via /v1/train/sft".to_string(),
    ))
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

/// Load a LoRA adapter using the two-phase RwLock pattern.
fn auto_load_adapter(
    runner: &std::sync::Arc<std::sync::RwLock<kiln_model::ModelRunner>>,
    active_adapter_name: &std::sync::Arc<std::sync::RwLock<Option<String>>>,
    adapter_path: &std::path::Path,
    adapter_name: &str,
    num_layers: usize,
) -> Result<(), String> {
    // Phase 1: read device under brief read lock
    let device = {
        let guard = runner.read().unwrap();
        guard.weights.embed_tokens.device().clone()
    };

    // Phase 2: load weights outside any lock
    let lora = LoraWeights::load(adapter_path, num_layers, &device)
        .map_err(|e| format!("failed to load adapter: {e}"))?;

    // Brief write lock to swap
    {
        let mut guard = runner.write().unwrap();
        guard.swap_lora(Some(lora));
    }
    *active_adapter_name.write().unwrap() = Some(adapter_name.to_string());

    Ok(())
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/train/sft", post(submit_sft))
        .route("/v1/train/grpo", post(submit_grpo))
        .route("/v1/train/status", get(training_status))
        .route("/v1/train/status/{job_id}", get(job_status))
}
