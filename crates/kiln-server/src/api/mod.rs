use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

mod health;
mod models;
mod completions;
mod adapters;
mod training;

pub fn router(state: AppState) -> Router {
    Router::new()
        .merge(health::routes())
        .merge(models::routes())
        .merge(completions::routes())
        .merge(adapters::routes())
        .merge(training::routes())
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}
