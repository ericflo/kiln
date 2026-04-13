use axum::{extract::State, routing::get, Json, Router};
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let _ = state; // Will use config for model name later
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo {
            id: "qwen3-4b".to_string(),
            object: "model",
            owned_by: "kiln",
        }],
    })
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/models", get(list_models))
}
