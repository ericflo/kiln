//! Live runtime statistics endpoints.
//!
//! Exposes `/v1/stats/decode` (decode tok/s + ITL over a rolling window) and
//! `/v1/stats/recent-requests` (bounded history of recent chat completions).
//! The /ui dashboard polls both every 2 seconds.

use std::time::Instant;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};

use crate::decode_stats::DecodeStatsSnapshot;
use crate::recent_requests::RequestRecord;
use crate::state::AppState;

async fn get_decode_stats(State(state): State<AppState>) -> Json<DecodeStatsSnapshot> {
    let snap = state
        .decode_stats
        .lock()
        .map(|ring| ring.snapshot(Instant::now()))
        .unwrap_or_else(|poisoned| poisoned.into_inner().snapshot(Instant::now()));
    Json(snap)
}

async fn get_recent_requests(State(state): State<AppState>) -> Json<Vec<RequestRecord>> {
    let snap = state
        .recent_requests
        .lock()
        .map(|ring| ring.snapshot())
        .unwrap_or_else(|poisoned| poisoned.into_inner().snapshot());
    Json(snap)
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/stats/decode", get(get_decode_stats))
        .route("/v1/stats/recent-requests", get(get_recent_requests))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use kiln_core::config::ModelConfig;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn make_test_state() -> AppState {
        let config = ModelConfig::qwen3_5_4b();
        let sched_config = SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
        };
        let scheduler = Scheduler::new(sched_config, 256);
        let engine = MockEngine::new(config.clone());
        let tokenizer = crate::api::test_tokenizer();
        AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            300,
            "kiln-test".to_string(),
        )
    }

    #[tokio::test]
    async fn empty_snapshot_serializes_to_zeros() {
        let state = make_test_state();
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stats/decode")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["sample_count"], 0);
        assert_eq!(body["tok_per_sec"], 0.0);
        assert!(body["window_secs"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn recent_requests_endpoint_is_empty_by_default() {
        let state = make_test_state();
        let app = routes().with_state(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stats/recent-requests")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(body.is_array());
        assert_eq!(body.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn recent_requests_endpoint_returns_newest_first() {
        use crate::recent_requests::RequestRecord;
        let state = make_test_state();
        {
            let mut ring = state.recent_requests.lock().unwrap();
            for (i, id) in ["first", "second", "third"].iter().enumerate() {
                ring.record(RequestRecord {
                    id: (*id).to_owned(),
                    timestamp_unix_ms: 1_000 + i as u64,
                    model: "kiln-test".to_owned(),
                    prompt_preview: format!("prompt {id}"),
                    completion_preview: format!("done {id}"),
                    prompt_tokens: 4,
                    completion_tokens: 8,
                    duration_ms: 50 + i as u64 * 10,
                    streamed: i % 2 == 0,
                    finish_reason: "stop".to_owned(),
                });
            }
        }

        let app = routes().with_state(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stats/recent-requests")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(resp.into_body(), 16 * 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let arr = body.as_array().expect("array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["id"], "third");
        assert_eq!(arr[1]["id"], "second");
        assert_eq!(arr[2]["id"], "first");
        assert_eq!(arr[0]["completion_tokens"], 8);
        assert_eq!(arr[0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn snapshot_reflects_recorded_tokens() {
        let state = make_test_state();
        {
            let mut ring = state.decode_stats.lock().unwrap();
            let t0 = Instant::now();
            ring.record_token(t0);
            ring.record_token(t0 + std::time::Duration::from_millis(20));
            ring.record_token(t0 + std::time::Duration::from_millis(40));
        }
        let app = routes().with_state(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stats/decode")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["sample_count"], 2);
        let tok_per_sec = body["tok_per_sec"].as_f64().unwrap();
        assert!(
            (tok_per_sec - 50.0).abs() < 1.0,
            "tok_per_sec should be ~50, got {}",
            tok_per_sec
        );
    }
}
