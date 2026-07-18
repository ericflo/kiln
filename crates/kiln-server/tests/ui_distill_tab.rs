//! Smoke test for the §3 + §10.6 Distill tab in the embedded dashboard.
//!
//! The UI (`src/ui/{index.html,app.js,...}`) is baked into the binary via
//! `include_str!`. The build path can succeed even if a handler
//! accidentally drops markup. This test mounts the api router against a
//! mock-backend state, fetches `/ui/` and `/ui/app.js`, and verifies that
//! the primary `Distill` tab and every sub-tab pane exist in the returned
//! HTML — the cheap regression net for the OPD UI surfaces.

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

fn mock_state() -> AppState {
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

async fn fetch_ui_route(app: axum::Router, uri: &str) -> String {
    let response = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK, "GET {uri}");
    let body = axum::body::to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    String::from_utf8(body.to_vec()).expect("UI asset is valid utf-8")
}

#[tokio::test]
async fn ui_contains_distill_primary_tab_and_every_sub_tab() {
    let state = mock_state();
    let html = fetch_ui_route(api::router(state.clone()), "/ui/").await;
    let app_js = fetch_ui_route(api::router(state), "/ui/app.js").await;
    let html = html.as_str();

    // Top-level Distill primary-nav tab.
    assert!(
        html.contains(r#"data-page="distill""#),
        "primary-nav missing Distill tab"
    );
    assert!(
        html.contains(r#"id="page-distill""#),
        "main missing page-distill section"
    );
    assert!(
        html.contains(r#"data-distill-tabs"#),
        "Distill page missing sub-tab container"
    );

    // Every sub-tab pane the §3 / §10.6 surfaces map to.
    for sub in &[
        "opd",
        "teachers",
        "recipes",
        "refresh",
        "pump",
        "merge",
        "self",
        "cache",
        "library",
        "traces",
        "preflight",
    ] {
        assert!(
            html.contains(&format!(r#"id="distill-tab-{sub}-pane""#)),
            "Distill page missing sub-tab pane for {sub}"
        );
    }

    // Forms each post to the right endpoint — guards against drift.
    assert!(html.contains(r#"id="opd-form""#), "OPD form missing");
    assert!(html.contains(r#"id="opd-checkpoint-interval""#));
    assert!(html.contains(r#"id="opd-resume-checkpoint""#));
    assert!(html.contains(r#"id="opd-resume-note""#));
    assert!(
        html.contains(r#"id="teacher-form""#),
        "Teacher register form missing"
    );
    assert!(html.contains(r#"id="teacher-credential-id""#));
    assert!(html.contains(r#"id="teacher-register-status""#));
    assert!(!html.contains(r#"id="teacher-api-key-env""#));
    assert!(!html.contains(r#"id="teacher-max-top-k""#));
    assert!(!html.contains(r#"id="teacher-vocab-size""#));
    assert!(
        html.contains(r#"id="distill-refresh-form""#),
        "distill_refresh form missing"
    );
    assert!(
        html.contains(r#"id="distill-pump-form""#),
        "distill_pump form missing"
    );
    assert!(
        html.contains(r#"id="distill-merge-form""#),
        "distill_merge form missing"
    );
    assert!(
        html.contains(r#"id="distill-self-form""#),
        "distill_self form missing"
    );
    assert!(
        html.contains(r#"id="library-publish-form""#),
        "library publish form missing"
    );

    // JS handlers (served at /ui/app.js) reference the actual endpoints.
    for endpoint in &[
        "/v1/train/opd",
        "/v1/teachers",
        "/v1/recipes",
        "/v1/recipes/run",
        "/v1/distill/refresh",
        "/v1/distill/pump",
        "/v1/distill/self",
        "/v1/adapters/distill_merge",
        "/v1/cache/stats",
        "/v1/cache/export",
        "/v1/library",
        "/v1/agent/traces",
        "/v1/agent/traces/discover",
        "/v1/preflight/compatibility",
        "/v1/preflight/tiers",
    ] {
        assert!(
            app_js.contains(endpoint),
            "Distill UI doesn't reference endpoint {endpoint}"
        );
    }

    assert!(
        !app_js.contains("/v1/cache/import"),
        "Distill UI must not expose the removed unsafe cache import route"
    );
    assert!(app_js.contains("t.usable !== true"));
    assert!(app_js.contains("body.config.checkpoint_interval = checkpointInterval"));
    assert!(app_js.contains("body.config.resume_checkpoint = resumeCheckpoint"));
    assert!(html.contains(r#"id="sft-invalid-row-policy""#));
    assert!(html.contains(r#"id="sft-detect-anomaly""#));
    assert!(html.contains(r#"id="grpo-detect-anomaly""#));
    assert!(html.contains(r#"id="opd-detect-anomaly""#));
    assert!(html.contains(r#"id="sft-adapter-smoke-test""#));
    assert!(html.contains(r#"id="sft-adapter-smoke-prompts""#));
    assert!(html.contains(r#"id="grpo-adapter-smoke-test""#));
    assert!(html.contains(r#"id="grpo-adapter-smoke-prompts""#));
    assert!(html.contains(r#"id="grpo-shared-prefix-reference""#));
    assert!(html.contains(r#"id="opd-sampler-segments""#));
    assert!(html.contains(r#"id="opd-rollout-prompt-rendering""#));
    assert!(app_js.contains("invalid_row_policy: invalidRowPolicy"));
    assert!(app_js.contains("config.detect_anomaly = true"));
    assert!(app_js.contains("body.config.detect_anomaly = true"));
    assert!(app_js.contains("config.adapter_smoke_prompts = smokePrompts"));
    assert!(
        app_js.contains("config.shared_prefix_reference = form.shared_prefix_reference.checked")
    );
    assert!(app_js.contains("body.config.sampler_segments = samplerSegments"));
    assert!(app_js.contains("body.config.rollout_prompt_rendering"));
    assert!(app_js.contains("config.invalid_row_policy === 'skip'"));
    assert!(app_js.contains("kind === 'sft' || kind === 'grpo' || kind === 'opd'"));
    assert!(app_js.contains("checkpoint.teacher_identity_revision"));
    assert!(app_js.contains("OPD checkpoint loaded"));
    assert!(app_js.contains("body.credential_id"));
    assert!(!app_js.contains("form.api_key_env"));
    assert!(!app_js.contains("body.max_top_k"));
    assert!(!app_js.contains("body.vocab_size"));
}
