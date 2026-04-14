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

    // Check GPU memory budget before accepting training job
    if let Err(msg) = state.memory_budget.check_training_feasible(0) {
        return Err((StatusCode::SERVICE_UNAVAILABLE, msg));
    }

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
    let gpu_lock = state.gpu_lock.clone();
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
            (
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
        // Acquire GPU write lock to prevent inference from running simultaneously,
        // avoiding combined VRAM peak that could OOM.
        // The write lock blocks all inference read locks for the duration of training.
        let result = {
            let _gpu_guard = gpu_lock.write().unwrap();
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

    tracing::info!(num_groups, total_completions, job_id = %job_id, adapter = %adapter_name, "GRPO training request received");

    // Check GPU memory budget before accepting training job
    if let Err(msg) = state.memory_budget.check_training_feasible(0) {
        return Err((StatusCode::SERVICE_UNAVAILABLE, msg));
    }

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

    // Spawn training on a background thread
    let training_jobs = state.training_jobs.clone();
    let model_config = state.model_config.clone();
    let tokenizer = state.tokenizer.clone();
    let adapter_dir = state.adapter_dir.clone();
    let active_adapter_name = state.active_adapter_name.clone();
    let gpu_lock = state.gpu_lock.clone();
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

        // Read model config under brief read lock
        let (weights_ref, _device, num_layers) = {
            let guard = runner_arc.read().unwrap();
            (
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

        // Run GRPO training — this blocks until complete.
        // Acquire GPU write lock to prevent inference from running simultaneously.
        let result = {
            let _gpu_guard = gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            trainer::grpo_train(
                &req.groups,
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
                    "GRPO training completed"
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
                    "GRPO training failed: {e:#}"
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
        message: format!("Queued GRPO training with {num_groups} groups ({total_completions} completions)"),
    }))
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
