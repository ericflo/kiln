use axum::Router;
use tower_http::trace::TraceLayer;
use tracing::Span;

use crate::state::AppState;

mod adapters;
pub mod agent_runs;
pub mod agent_traces;
pub(crate) mod cache;
pub(crate) mod completions;
mod config;
pub mod corrections;
mod debug_model_state;
mod eval;
mod health;
mod hf_trl;
mod hf_trl_import;
pub(crate) mod library;
mod metrics;
mod models;
pub mod openenv;
pub(crate) mod pit_of_success;
pub(crate) mod recipes;
pub mod self_improve;
mod stats;
pub(crate) mod teachers;
pub mod terminal;
mod training;
mod ui;

pub(crate) use training::{
    enforce_queued_training_optimizer_admission, enforce_queued_training_workload_admission,
};

#[cfg(test)]
pub(crate) fn test_tokenizer() -> kiln_core::tokenizer::KilnTokenizer {
    let json = br#"{
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1},
            "merges": []
        }
    }"#;
    kiln_core::tokenizer::KilnTokenizer::from_bytes(json).unwrap()
}

pub fn router(state: AppState) -> Router {
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|request: &axum::http::Request<_>| {
            tracing::info_span!(
                "http_request",
                method = %request.method(),
                path = %request.uri().path(),
                status = tracing::field::Empty,
                duration_ms = tracing::field::Empty,
            )
        })
        .on_response(
            |response: &axum::http::Response<_>, latency: std::time::Duration, span: &Span| {
                span.record("status", response.status().as_u16());
                span.record("duration_ms", latency.as_secs_f64() * 1000.0);
                tracing::info!(
                    status = response.status().as_u16(),
                    duration_ms = latency.as_secs_f64() * 1000.0,
                    "response"
                );
            },
        );

    Router::new()
        .merge(health::routes())
        .merge(metrics::routes())
        .merge(models::routes())
        // Inference routes get the durable request/response tap (no-op when
        // [request_log] is disabled or unset on this AppState).
        .merge(
            completions::routes().layer(axum::middleware::from_fn_with_state(
                state.clone(),
                crate::request_log::tap,
            )),
        )
        .merge(adapters::routes())
        .merge(corrections::routes())
        .merge(teachers::routes())
        .merge(recipes::routes())
        .merge(cache::routes())
        .merge(library::routes())
        .merge(agent_runs::routes())
        .merge(agent_traces::routes())
        .merge(openenv::routes())
        .merge(self_improve::routes())
        .merge(pit_of_success::routes())
        .merge(training::routes())
        .merge(hf_trl::routes())
        .merge(hf_trl_import::routes())
        .merge(eval::routes())
        .merge(config::routes())
        .merge(debug_model_state::routes())
        .merge(stats::routes())
        .merge(terminal::routes())
        .merge(ui::routes())
        .with_state(state)
        .layer(trace_layer)
}
