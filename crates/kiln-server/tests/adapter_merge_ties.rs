//! Integration test: POST /v1/adapters/merge accepts `mode == "ties"` and
//! produces a PEFT-compatible merged adapter on disk.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::adapter_merge::{MergeTensor, PeftLora};
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

/// Write a single-tensor PEFT-format adapter (rank=2, q_proj only) where
/// every value of `lora_A` and `lora_B` is `fill`. Used to set up sources
/// for the merge endpoint.
fn write_uniform_adapter(adapter_dir: &std::path::Path, name: &str, fill: f32) {
    let path = adapter_dir.join(name);
    let mut tensors: BTreeMap<String, MergeTensor> = BTreeMap::new();
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        MergeTensor {
            shape: vec![2, 4],
            data: vec![fill; 8],
        },
    );
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        MergeTensor {
            shape: vec![3, 2],
            data: vec![fill; 6],
        },
    );
    let config = json!({
        "r": 2,
        "lora_alpha": 4.0,
        "target_modules": ["q_proj"],
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen3.5-4B"
    });
    let adapter = PeftLora { config, tensors };
    adapter.save(&path).unwrap();
}

#[tokio::test]
async fn test_merge_ties_endpoint_returns_200_and_writes_adapter() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 2.0);
    write_uniform_adapter(tmp.path(), "src-b", 6.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let body = json!({
        "sources": [
            { "name": "src-a", "weight": 0.5 },
            { "name": "src-b", "weight": 0.5 },
        ],
        "output_name": "merged-ties",
        "mode": "ties",
        "density": 1.0,
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/merge")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "ties merge failed: {}",
        String::from_utf8_lossy(&body_bytes)
    );

    let parsed: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(parsed["status"], "merged");
    assert_eq!(parsed["mode"], "ties");
    assert_eq!(parsed["output_name"], "merged-ties");
    assert_eq!(parsed["num_tensors"], 2);

    // Merged adapter should be loadable and have the expected values.
    let merged_dir = tmp.path().join("merged-ties");
    assert!(merged_dir.exists(), "merged adapter dir was not created");
    let loaded = PeftLora::load(&merged_dir).unwrap();
    // density=1.0 + matching positive signs => weighted average of values.
    // (0.5*2 + 0.5*6) / (0.5 + 0.5) = 4.0 across every position in A and B.
    let a = &loaded.tensors
        ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
    for v in &a.data {
        assert!((*v - 4.0).abs() < 1e-6, "expected 4.0 in A, got {v}");
    }
    let b = &loaded.tensors
        ["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];
    for v in &b.data {
        assert!((*v - 4.0).abs() < 1e-6, "expected 4.0 in B, got {v}");
    }
}

#[tokio::test]
async fn test_merge_ties_endpoint_rejects_bad_density() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 1.0);
    write_uniform_adapter(tmp.path(), "src-b", 1.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    // density = 0.0 is excluded (open lower bound).
    let body = json!({
        "sources": [
            { "name": "src-a", "weight": 0.5 },
            { "name": "src-b", "weight": 0.5 },
        ],
        "output_name": "merged-bad",
        "mode": "ties",
        "density": 0.0,
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/merge")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "expected 400, got {}: {}",
        status,
        String::from_utf8_lossy(&body_bytes)
    );
    let msg = String::from_utf8_lossy(&body_bytes);
    assert!(
        msg.contains("density"),
        "expected density-related error, got: {msg}"
    );

    // density > 1.0 also rejected.
    let body = json!({
        "sources": [
            { "name": "src-a", "weight": 0.5 },
            { "name": "src-b", "weight": 0.5 },
        ],
        "output_name": "merged-bad-2",
        "mode": "ties",
        "density": 1.5,
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/merge")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_merge_density_rejected_for_weighted_average() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 1.0);
    write_uniform_adapter(tmp.path(), "src-b", 1.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    // density only applies to TIES mode; passing it with the default mode
    // should be a clean 400 rather than a silent ignore.
    let body = json!({
        "sources": [
            { "name": "src-a", "weight": 0.5 },
            { "name": "src-b", "weight": 0.5 },
        ],
        "output_name": "merged-bad-density",
        "density": 0.5,
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/merge")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "expected 400, got {}: {}",
        status,
        String::from_utf8_lossy(&body_bytes)
    );
}
