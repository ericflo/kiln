//! Integration tests for expanded `GET /v1/adapters` registry state.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use serde_json::json;
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::state::AppState;

fn test_tokenizer() -> KilnTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("a".to_string(), 0);
    vocab.insert("b".to_string(), 1);
    let json = json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] }
    });
    KilnTokenizer::from_bytes(&serde_json::to_vec(&json).unwrap()).unwrap()
}

fn make_state(adapter_dir: std::path::PathBuf) -> AppState {
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
        "qwen3.5-4b-kiln".to_string(),
    );
    state.adapter_dir = adapter_dir;
    state
}

fn write_adapter(adapter_dir: &std::path::Path, name: &str, rank: u64, alpha: f64) {
    let path = adapter_dir.join(name);
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(
        path.join("adapter_config.json"),
        serde_json::to_vec_pretty(&json!({
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(path.join("adapter_model.safetensors"), b"weights").unwrap();
}

fn entry_by_name<'a>(body: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    body["available_adapters"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("missing registry entry {name}"))
}

#[tokio::test]
async fn adapters_registry_reports_loaded_available_and_invalid_entries() {
    let tmp = tempfile::tempdir().unwrap();
    write_adapter(tmp.path(), "active-adapter", 8, 16.0);
    write_adapter(tmp.path(), "runtime-only", 4, 6.0);
    write_adapter(tmp.path(), "available-only", 2, 8.0);
    std::fs::create_dir_all(tmp.path().join("invalid-empty")).unwrap();

    let state = make_state(tmp.path().to_path_buf());
    *state.active_adapter_name.write().unwrap() = Some("active-adapter".to_string());
    *state.loaded_adapter_name.write().unwrap() = Some("runtime-only".to_string());
    state
        .adapter_load_errors
        .write()
        .unwrap()
        .insert("invalid-empty".to_string(), "previous load failed".to_string());

    let app = api::router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/adapters")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(body["active_adapter"], "active-adapter");
    assert_eq!(body["active"], "active-adapter");
    assert_eq!(body["loaded_adapter"], "runtime-only");
    assert_eq!(body["loaded_adapters"], json!(["active-adapter", "runtime-only"]));
    assert_eq!(body["adapter_dir"], tmp.path().canonicalize().unwrap().to_str().unwrap());

    let active = entry_by_name(&body, "active-adapter");
    assert_eq!(active["status"], "loaded");
    assert_eq!(active["rank"], 8);
    assert_eq!(active["alpha"], 16.0);
    assert_eq!(active["alpha_over_rank"], 2.0);
    assert_eq!(active["target_modules"], json!(["q_proj", "v_proj"]));
    assert_eq!(active["base_model_name_or_path"], "Qwen/Qwen3.5-4B");
    assert_eq!(active["adapter_model_size_bytes"], 7);
    assert_eq!(active["adapter_model_sha256"].as_str().unwrap().len(), 64);
    assert!(active["path"].as_str().unwrap().ends_with("active-adapter"));

    let runtime_only = entry_by_name(&body, "runtime-only");
    assert_eq!(runtime_only["status"], "loaded");

    let available = entry_by_name(&body, "available-only");
    assert_eq!(available["status"], "available");

    let invalid = entry_by_name(&body, "invalid-empty");
    assert_eq!(invalid["status"], "invalid");
    assert_eq!(invalid["last_load_error"], "previous load failed");
    assert!(invalid["error"].as_str().unwrap().contains("adapter_config.json"));
    assert!(
        invalid["error"]
            .as_str()
            .unwrap()
            .contains("adapter_model.safetensors")
    );
}
