use axum::{http::StatusCode, routing::{get, post}, Json, Router};

use kiln_train::{
    GrpoRequest, SftRequest, TrainingResponse, TrainingState, TrainingStatus,
};

use crate::state::AppState;

async fn submit_sft(
    Json(req): Json<SftRequest>,
) -> Result<Json<TrainingResponse>, (StatusCode, String)> {
    let num_examples = req.examples.len();
    tracing::info!(num_examples, "SFT training request received");

    // TODO: forward to Python training sidecar
    Ok(Json(TrainingResponse {
        job_id: uuid::Uuid::new_v4().to_string(),
        state: TrainingState::Queued,
        message: format!("Queued SFT training with {num_examples} examples"),
    }))
}

async fn submit_grpo(
    Json(req): Json<GrpoRequest>,
) -> Result<Json<TrainingResponse>, (StatusCode, String)> {
    let num_groups = req.groups.len();
    let total_completions: usize = req.groups.iter().map(|g| g.completions.len()).sum();
    tracing::info!(num_groups, total_completions, "GRPO training request received");

    // TODO: forward to Python training sidecar
    Ok(Json(TrainingResponse {
        job_id: uuid::Uuid::new_v4().to_string(),
        state: TrainingState::Queued,
        message: format!(
            "Queued GRPO training with {num_groups} groups ({total_completions} completions)"
        ),
    }))
}

async fn training_status() -> Json<Option<TrainingStatus>> {
    // TODO: query sidecar for current training status
    Json(None)
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/train/sft", post(submit_sft))
        .route("/v1/train/grpo", post(submit_grpo))
        .route("/v1/train/status", get(training_status))
}
