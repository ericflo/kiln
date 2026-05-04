use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::Span;

use crate::state::AppState;

mod adapters;
mod completions;
mod config;
mod health;
mod metrics;
mod models;
mod stats;
mod training;
mod ui;

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
        .merge(completions::routes())
        .merge(adapters::routes())
        .merge(training::routes())
        .merge(config::routes())
        .merge(stats::routes())
        .merge(ui::routes())
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(trace_layer)
}
