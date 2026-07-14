//! Executable binding between the published OpenAPI path/method inventory and
//! the production Axum router. Field-level schemas are checked by the portable
//! contract tooling; this test proves those operations are actually mounted.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::config::OperationalRuntimeConfig;
use kiln_server::state::AppState;
use serde_json::json;
use tower::ServiceExt;

const OPENAPI: &str = include_str!("../../../contracts/kiln-http-api-v1.openapi.json");
const PROBED_METHODS: &[&str] = &["GET", "POST", "DELETE", "PUT", "PATCH"];

fn test_tokenizer() -> KilnTokenizer {
    let vocab: HashMap<String, u32> = [("a".to_owned(), 0), ("b".to_owned(), 1)]
        .into_iter()
        .collect();
    KilnTokenizer::from_bytes(
        &serde_json::to_vec(&json!({
            "version": "1.0",
            "model": { "type": "BPE", "vocab": vocab, "merges": [] }
        }))
        .unwrap(),
    )
    .unwrap()
}

fn mock_state(root: &std::path::Path) -> AppState {
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
        30,
        "Qwen3.5-4B".to_owned(),
    );
    state.adapter_dir = root.join("adapters");
    std::fs::create_dir_all(&state.adapter_dir).unwrap();
    state.operational_runtime = Arc::new(OperationalRuntimeConfig {
        logit_cache_dir: root.join("logit-cache"),
        pi_sessions_dir: root.join("pi-sessions"),
        ..OperationalRuntimeConfig::default()
    });
    state
}

fn concrete_path(template: &str) -> String {
    template
        .split('/')
        .map(|segment| {
            if segment.starts_with('{') && segment.ends_with('}') {
                "contract-probe"
            } else {
                segment
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn documented_operations() -> BTreeMap<String, BTreeSet<String>> {
    let document: serde_json::Value = serde_json::from_str(OPENAPI).unwrap();
    document["paths"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(path, item)| {
            let methods = item
                .as_object()
                .unwrap()
                .keys()
                .filter_map(|method| {
                    let upper = method.to_ascii_uppercase();
                    PROBED_METHODS.contains(&upper.as_str()).then_some(upper)
                })
                .collect();
            (path.clone(), methods)
        })
        .collect()
}

#[tokio::test]
async fn production_router_matches_openapi_paths_and_methods() {
    let root = tempfile::tempdir().unwrap();
    let app = api::router(mock_state(root.path()));
    let operations = documented_operations();
    assert_eq!(operations.len(), 101, "OpenAPI path accounting drifted");
    assert_eq!(
        operations.values().map(BTreeSet::len).sum::<usize>(),
        112,
        "OpenAPI operation accounting drifted"
    );

    for (template, documented_methods) in operations {
        assert!(
            PROBED_METHODS
                .iter()
                .any(|method| !documented_methods.contains(*method)),
            "{template} needs an unsupported probe method to distinguish a mounted path from 404"
        );
        let path = concrete_path(&template);
        for method in PROBED_METHODS {
            let request = Request::builder()
                .method(Method::from_bytes(method.as_bytes()).unwrap())
                .uri(&path)
                .header("content-type", "application/json")
                .body(if *method == "POST" {
                    // Invalid JSON proves routing without starting valid work in
                    // handlers whose request body would otherwise be actionable.
                    Body::from("{")
                } else {
                    Body::empty()
                })
                .unwrap();
            let response =
                tokio::time::timeout(Duration::from_secs(2), app.clone().oneshot(request))
                    .await
                    .unwrap_or_else(|_| {
                        panic!("{method} {path} timed out during router contract probe")
                    })
                    .unwrap();
            let mounted = response.status() != StatusCode::METHOD_NOT_ALLOWED;
            assert_eq!(
                mounted,
                documented_methods.contains(*method),
                "live router/OpenAPI mismatch for {method} {template} (probe {path}) returned {}",
                response.status()
            );
        }
    }
}
