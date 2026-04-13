use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

use kiln_train::{
    GrpoRequest, SftRequest, TrainingResponse, TrainingState, TrainingStatus,
};

use crate::sidecar::{SidecarClient, SidecarResponse};
use crate::state::AppState;

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
        SidecarResponse::JobAccepted { job_id, .. } => Ok(Json(TrainingResponse {
            job_id,
            state: TrainingState::Queued,
            message: format!("Queued SFT training with {num_examples} examples"),
        })),
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
        SidecarResponse::JobAccepted { job_id, .. } => Ok(Json(TrainingResponse {
            job_id,
            state: TrainingState::Queued,
            message: format!(
                "Queued GRPO training with {num_groups} groups ({total_completions} completions)"
            ),
        })),
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

async fn training_status(
    State(state): State<AppState>,
) -> Result<Json<Option<TrainingStatus>>, (StatusCode, String)> {
    let client = match &state.sidecar {
        Some(c) => c,
        None => return Ok(Json(None)),
    };

    if !client.is_available() {
        return Ok(Json(None));
    }

    // The status endpoint doesn't have a specific job_id in the current API shape.
    // Return None for now — callers should use /v1/train/status/:job_id when we add it.
    Ok(Json(None))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/train/sft", post(submit_sft))
        .route("/v1/train/grpo", post(submit_grpo))
        .route("/v1/train/status", get(training_status))
}
