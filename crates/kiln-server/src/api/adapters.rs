use axum::{routing::get, Json, Router};
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
struct AdaptersResponse {
    active: Option<String>,
    available: Vec<String>,
}

async fn list_adapters() -> Json<AdaptersResponse> {
    // TODO: query engine for real adapter state
    Json(AdaptersResponse {
        active: None,
        available: vec![],
    })
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/adapters", get(list_adapters))
}
