//! Contract test for the eval-drill → corrections capture surface.
//!
//! The eval drill modal's "Add to corrections" button closes the
//! observe→correct loop for eval failures the same way the recent-requests
//! capture does for live traffic. The dashboard JS (`src/ui/app.js`, served
//! at `/ui/app.js`) is `include_str!`-baked, so a redesign can silently drop
//! the wiring without a compile error — this pins the JS hooks (the drill
//! modal renders its markup from app.js) and the safety property that
//! non-verbatim scorer targets are never pre-seeded as the ideal answer.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::state::AppState;

fn tiny_tokenizer() -> KilnTokenizer {
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

fn make_state() -> AppState {
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
    AppState::new_mock(
        config,
        scheduler,
        Arc::new(engine),
        tiny_tokenizer(),
        300,
        "Qwen3.5-4B".to_string(),
    )
}

#[tokio::test]
async fn ui_carries_eval_drill_correction_capture() {
    let app = api::router(make_state());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/ui/app.js")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let app_js = String::from_utf8(
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();

    // The capture button renders for failing outcomes...
    assert!(
        app_js.contains("data-outcome-correct"),
        "eval drill must offer Add to corrections on failing outcomes"
    );
    // ...through the shared basket-insert helper...
    assert!(app_js.contains("function addCorrectionItem"));
    assert!(app_js.contains("function addCorrectionFromEvalOutcome"));
    // ...and both capture surfaces go through it.
    assert!(
        app_js.matches("addCorrectionItem(").count() >= 3,
        "recent-requests and eval-drill captures must share addCorrectionItem"
    );
    // Safety property: only verbatim-target scorers pre-seed the ideal —
    // a pattern/choice target must never be trained as a reply.
    assert!(
        app_js.contains("scorerKind === 'exact_match' || scorerKind === 'contains'"),
        "ideal pre-seeding must stay restricted to verbatim-target scorers"
    );
}
