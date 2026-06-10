//! Integration test: durable request/response log capture.
//!
//! Every inference request — non-streaming, streaming (SSE), and error
//! responses — must land as one JSONL row carrying the wire-format request
//! and response so production traffic is minable and trainable later.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::request_log::{RequestLogConfig, RequestLogger};
use kiln_server::state::AppState;

fn test_tokenizer() -> KilnTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0u32..32 {
        vocab.insert(format!("t{i}"), i);
    }
    let json = json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": [
            {
                "id": 0, "content": "<|endoftext|>",
                "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            },
        ]
    });
    KilnTokenizer::from_bytes(&serde_json::to_vec(&json).unwrap()).unwrap()
}

fn make_state_with_log(dir: &std::path::Path) -> (AppState, Arc<RequestLogger>) {
    let config = ModelConfig::qwen3_5_4b();
    let scheduler = Scheduler::new(
        SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
        },
        256,
    );
    let engine = MockEngine::new(config.clone());
    let mut state = AppState::new_mock(
        config,
        scheduler,
        Arc::new(engine),
        test_tokenizer(),
        300,
        "Qwen3.5-4B".to_string(),
    );
    let logger = RequestLogger::spawn(dir.to_path_buf(), RequestLogConfig::default()).unwrap();
    state.request_log = Some(logger.clone());
    (state, logger)
}

fn read_rows(dir: &std::path::Path) -> Vec<Value> {
    let text = std::fs::read_to_string(dir.join("requests-current.jsonl")).unwrap_or_default();
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

async fn post_json(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("user-agent", "request-log-test")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, bytes.to_vec())
}

#[tokio::test]
async fn non_streaming_chat_completion_is_logged_with_full_bodies() {
    let dir = tempfile::tempdir().unwrap();
    let (state, logger) = make_state_with_log(dir.path());
    let app = api::router(state);

    let (status, _) = post_json(
        &app,
        "/v1/chat/completions",
        json!({ "messages": [{"role": "user", "content": "Hello kiln"}] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    logger.flush();

    let rows = read_rows(dir.path());
    assert_eq!(rows.len(), 1, "expected exactly one log row");
    let row = &rows[0];
    assert_eq!(row["route"], "/v1/chat/completions");
    assert_eq!(row["status"], 200);
    assert_eq!(row["streamed"], false);
    assert_eq!(row["user_agent"], "request-log-test");
    assert_eq!(row["request"]["messages"][0]["content"], "Hello kiln");
    // The mining contract: request.messages + response message form a
    // complete chat transcript.
    let message = &row["response"]["choices"][0]["message"];
    assert_eq!(message["role"], "assistant");
    assert!(
        message["content"].as_str().is_some(),
        "assistant content missing: {row}"
    );
    assert!(row["duration_ms"].is_u64());
}

#[tokio::test]
async fn streaming_rejection_in_mock_mode_is_logged() {
    // The mock backend rejects streaming with 501 — the tap must still
    // record the request and the structured error body.
    let dir = tempfile::tempdir().unwrap();
    let (state, logger) = make_state_with_log(dir.path());
    let app = api::router(state);

    let (status, _) = post_json(
        &app,
        "/v1/chat/completions",
        json!({
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    logger.flush();

    let rows = read_rows(dir.path());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["status"], 501);
    assert_eq!(
        rows[0]["response"]["error"]["code"],
        "streaming_not_supported"
    );
}

#[tokio::test]
async fn sse_responses_are_reassembled_through_the_tap() {
    // The mock backend cannot stream, so exercise the middleware's SSE leg
    // with a stub route emitting real OpenAI-style chunks under the same
    // tap layer the inference routes use.
    use axum::response::IntoResponse;
    use axum::routing::post;

    let dir = tempfile::tempdir().unwrap();
    let (state, logger) = make_state_with_log(dir.path());

    async fn fake_sse() -> axum::response::Response {
        let body = concat!(
            "data: {\"id\":\"chatcmpl-x\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hi \"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"choices\":[{\"delta\":{\"content\":\"there\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        (
            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
            body,
        )
            .into_response()
    }

    let app = axum::Router::new()
        .route("/v1/chat/completions", post(fake_sse))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            kiln_server::request_log::tap,
        ))
        .with_state(state);

    let (status, body) = post_json(
        &app,
        "/v1/chat/completions",
        json!({ "messages": [{"role": "user", "content": "stream"}], "stream": true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(String::from_utf8_lossy(&body).contains("data: [DONE]"));
    logger.flush();

    let rows = read_rows(dir.path());
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row["streamed"], true);
    assert!(
        row.get("stream_interrupted").is_none(),
        "completed stream must not be flagged interrupted: {row}"
    );
    let message = &row["response"]["choices"][0]["message"];
    assert_eq!(message["content"], "Hi there");
    assert_eq!(row["response"]["choices"][0]["finish_reason"], "stop");
    assert_eq!(row["response"]["usage"]["completion_tokens"], 2);
    assert_eq!(row["request"]["messages"][0]["content"], "stream");
}

#[tokio::test]
async fn error_responses_are_logged_too() {
    let dir = tempfile::tempdir().unwrap();
    let (state, logger) = make_state_with_log(dir.path());
    let app = api::router(state);

    // Missing `messages` → 4xx from the handler; the row must still land.
    let (status, _) = post_json(&app, "/v1/chat/completions", json!({ "bogus": true })).await;
    assert!(status.is_client_error(), "expected 4xx, got {status}");
    logger.flush();

    let rows = read_rows(dir.path());
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row["status"].as_u64().unwrap(), status.as_u16() as u64);
    assert_eq!(row["request"]["bogus"], true);
    assert!(
        row["response"].get("error").is_some() || row["response"].get("_raw").is_some(),
        "error body should be captured: {row}"
    );
}
