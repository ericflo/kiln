//! Integration test: per-request adapter composition (`adapters: [...]` field
//! on `POST /v1/chat/completions`).
//!
//! Verifies that a fresh composition spec triggers a `merge_concat` and writes
//! the result under `<adapter_dir>/.composed/<hash>/`, that a second request
//! with the same spec is served from the cached directory (no remerge), and
//! that mutually-exclusive misuse with `adapter` returns 400.

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

/// BPE tokenizer rich enough to round-trip the ChatML prompt scaffolding
/// (`<|im_start|>user\n…<|im_end|>`) without surfacing an Encode error from
/// the tokenizers crate. Mirrors the helper in real_model_integration.rs so
/// `/v1/chat/completions` succeeds end-to-end in mock mode.
fn test_tokenizer() -> KilnTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0u32..20 {
        vocab.insert(format!("t{i}"), i);
    }
    vocab.insert("<|im_start|>".to_string(), 20);
    vocab.insert("<|im_end|>".to_string(), 21);
    vocab.insert("user".to_string(), 22);
    vocab.insert("assistant".to_string(), 23);
    vocab.insert("\n".to_string(), 24);
    for i in 25u32..32 {
        vocab.insert(format!("x{i}"), i);
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
            {
                "id": 20, "content": "<|im_start|>",
                "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            },
            {
                "id": 21, "content": "<|im_end|>",
                "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            }
        ]
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

/// Write a single-tensor PEFT-format adapter (q_proj only). Same shape as the
/// fixtures in `adapter_merge_concat.rs` so `merge_concat` accepts the pair.
fn write_uniform_adapter(adapter_dir: &std::path::Path, name: &str, rank: usize, fill: f32) {
    let path = adapter_dir.join(name);
    let mut tensors: BTreeMap<String, MergeTensor> = BTreeMap::new();
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        MergeTensor {
            shape: vec![rank, 4],
            data: vec![fill; rank * 4],
        },
    );
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        MergeTensor {
            shape: vec![3, rank],
            data: vec![fill; 3 * rank],
        },
    );
    let config = json!({
        "r": rank,
        "lora_alpha": (rank * 2) as f32,
        "target_modules": ["q_proj"],
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen3.5-4B"
    });
    let adapter = PeftLora { config, tensors };
    adapter.save(&path).unwrap();
}

fn chat_with_adapters(adapters: Value) -> Request<Body> {
    let body = json!({
        "model": "qwen3.5-4b-kiln",
        "messages": [{"role": "user", "content": "t1 t2 t3"}],
        "max_tokens": 4,
        "adapters": adapters,
    });
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// First request synthesizes the composed adapter on disk; second request with
/// the same `adapters` payload reuses the cached directory without remerging.
#[tokio::test]
async fn test_compose_endpoint_caches_synthesized_adapter() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 2, 1.0);
    write_uniform_adapter(tmp.path(), "src-b", 2, 5.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let composed_root = tmp.path().join(".composed");
    assert!(
        !composed_root.exists(),
        "composed cache should not exist yet"
    );

    let payload = json!([
        { "name": "src-a", "scale": 0.5 },
        { "name": "src-b", "scale": 0.5 },
    ]);

    let resp = app.clone().oneshot(chat_with_adapters(payload.clone())).await.unwrap();
    let status = resp.status();
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "first compose request failed: {}",
        String::from_utf8_lossy(&body_bytes)
    );

    // Exactly one cached composition dir, with PEFT-shaped contents.
    let entries: Vec<_> = std::fs::read_dir(&composed_root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "expected one composed cache dir, got {entries:?}"
    );
    let composed_dir = &entries[0];
    assert!(composed_dir.join("adapter_config.json").exists());
    assert!(composed_dir.join("adapter_model.safetensors").exists());

    // Concat-merge of two rank-2 adapters → rank 4 with correct A/B shapes.
    let loaded = PeftLora::load(composed_dir).unwrap();
    assert_eq!(loaded.rank(), Some(4));

    // Capture the safetensors mtime to prove the second request did not
    // overwrite the cached file.
    let mtime_before = std::fs::metadata(composed_dir.join("adapter_model.safetensors"))
        .unwrap()
        .modified()
        .unwrap();

    // Wait long enough that any rewrite would bump the mtime under fs
    // resolution (some filesystems round to seconds).
    std::thread::sleep(std::time::Duration::from_millis(1100));

    let resp2 = app.clone().oneshot(chat_with_adapters(payload)).await.unwrap();
    let status2 = resp2.status();
    let body_bytes2 = axum::body::to_bytes(resp2.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status2,
        StatusCode::OK,
        "second compose request failed: {}",
        String::from_utf8_lossy(&body_bytes2)
    );

    // Cache hit: same dir, untouched file.
    let entries_after: Vec<_> = std::fs::read_dir(&composed_root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(
        entries_after.len(),
        1,
        "expected cache reuse, got {entries_after:?}"
    );
    let mtime_after = std::fs::metadata(composed_dir.join("adapter_model.safetensors"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(
        mtime_before, mtime_after,
        "composed adapter was rewritten on second request — cache miss"
    );
}

/// Scale changes produce a distinct cache directory (different hash).
#[tokio::test]
async fn test_compose_endpoint_distinct_scales_distinct_cache_dirs() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 2, 1.0);
    write_uniform_adapter(tmp.path(), "src-b", 2, 1.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let resp_a = app
        .clone()
        .oneshot(chat_with_adapters(json!([
            { "name": "src-a", "scale": 0.5 },
            { "name": "src-b", "scale": 0.5 },
        ])))
        .await
        .unwrap();
    assert_eq!(resp_a.status(), StatusCode::OK);

    let resp_b = app
        .clone()
        .oneshot(chat_with_adapters(json!([
            { "name": "src-a", "scale": 0.75 },
            { "name": "src-b", "scale": 0.25 },
        ])))
        .await
        .unwrap();
    assert_eq!(resp_b.status(), StatusCode::OK);

    let composed_root = tmp.path().join(".composed");
    let entries: Vec<_> = std::fs::read_dir(&composed_root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(
        entries.len(),
        2,
        "expected two distinct cache dirs (one per scale tuple), got {entries:?}"
    );
}

/// Specifying both `adapter` and `adapters` is a 400.
#[tokio::test]
async fn test_compose_endpoint_rejects_both_adapter_and_adapters() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 2, 1.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let body = json!({
        "model": "qwen3.5-4b-kiln",
        "messages": [{"role": "user", "content": "t1 t2 t3"}],
        "max_tokens": 1,
        "adapter": "src-a",
        "adapters": [{ "name": "src-a", "scale": 1.0 }],
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(parsed["error"]["code"], "invalid_compose_request");
}

/// Empty `adapters: []` is a 400.
#[tokio::test]
async fn test_compose_endpoint_rejects_empty_list() {
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let resp = app
        .clone()
        .oneshot(chat_with_adapters(json!([])))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(parsed["error"]["code"], "invalid_compose_request");
}

/// Missing source adapter surfaces as 404.
#[tokio::test]
async fn test_compose_endpoint_404_when_source_missing() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 2, 1.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let resp = app
        .clone()
        .oneshot(chat_with_adapters(json!([
            { "name": "src-a", "scale": 0.5 },
            { "name": "ghost", "scale": 0.5 },
        ])))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(parsed["error"]["code"], "adapter_not_found");
}
