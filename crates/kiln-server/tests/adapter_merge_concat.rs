//! Integration test: POST /v1/adapters/merge accepts `mode == "concat"` and
//! produces a PEFT-compatible higher-rank merged adapter on disk.

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

/// Write a single-tensor PEFT-format adapter (q_proj only) where every value
/// of `lora_A` and `lora_B` is `fill`. `rank` controls the rank dim of both
/// tensors; in_features is fixed at 4 and out_features at 3.
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

#[tokio::test]
async fn test_merge_concat_endpoint_returns_200_and_writes_higher_rank_adapter() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 2, 1.0);
    write_uniform_adapter(tmp.path(), "src-b", 2, 5.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let body = json!({
        "sources": [
            { "name": "src-a", "weight": 0.5 },
            { "name": "src-b", "weight": 0.5 },
        ],
        "output_name": "merged-concat",
        "mode": "concat",
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
        "concat merge failed: {}",
        String::from_utf8_lossy(&body_bytes)
    );

    let parsed: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(parsed["status"], "merged");
    assert_eq!(parsed["mode"], "concat");
    assert_eq!(parsed["output_name"], "merged-concat");
    assert_eq!(parsed["num_tensors"], 2);

    let merged_dir = tmp.path().join("merged-concat");
    assert!(merged_dir.exists(), "merged adapter dir was not created");
    let loaded = PeftLora::load(&merged_dir).unwrap();

    // Rank grew additively: 2 + 2 = 4.
    assert_eq!(loaded.rank(), Some(4));
    let a = &loaded.tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
    let b = &loaded.tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];
    assert_eq!(a.shape, vec![4, 4]);
    assert_eq!(b.shape, vec![3, 4]);
    // A is row-concat of A_1=1.0 (rows 0..2) and A_2=5.0 (rows 2..4).
    for v in &a.data[0..8] {
        assert!((*v - 1.0).abs() < 1e-6, "A first block: got {v}");
    }
    for v in &a.data[8..16] {
        assert!((*v - 5.0).abs() < 1e-6, "A second block: got {v}");
    }
    // B is column-concat with each block scaled by w_i. With weights 0.5/0.5,
    // every row should be [0.5, 0.5, 2.5, 2.5].
    for j in 0..3 {
        let row = &b.data[j * 4..(j + 1) * 4];
        assert!((row[0] - 0.5).abs() < 1e-6, "B row {j}: {row:?}");
        assert!((row[1] - 0.5).abs() < 1e-6, "B row {j}: {row:?}");
        assert!((row[2] - 2.5).abs() < 1e-6, "B row {j}: {row:?}");
        assert!((row[3] - 2.5).abs() < 1e-6, "B row {j}: {row:?}");
    }
}

#[tokio::test]
async fn test_merge_concat_endpoint_rejects_density() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 2, 1.0);
    write_uniform_adapter(tmp.path(), "src-b", 2, 1.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    // density only applies to TIES — passing it with mode=concat must 400.
    let body = json!({
        "sources": [
            { "name": "src-a", "weight": 0.5 },
            { "name": "src-b", "weight": 0.5 },
        ],
        "output_name": "merged-bad-concat-density",
        "mode": "concat",
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
        "expected 400 for density+concat, got {}: {}",
        status,
        String::from_utf8_lossy(&body_bytes)
    );
    let msg = String::from_utf8_lossy(&body_bytes);
    assert!(
        msg.contains("density"),
        "expected density-related error, got: {msg}"
    );
}

#[tokio::test]
async fn test_merge_concat_endpoint_accepts_different_ranks() {
    let tmp = tempfile::tempdir().unwrap();
    // Different ranks are the whole point of concat.
    write_uniform_adapter(tmp.path(), "src-a", 2, 1.0);
    write_uniform_adapter(tmp.path(), "src-b", 3, 1.0);
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let body = json!({
        "sources": [
            { "name": "src-a", "weight": 1.0 },
            { "name": "src-b", "weight": 1.0 },
        ],
        "output_name": "merged-concat-mixed",
        "mode": "concat",
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
        "concat with mixed ranks failed: {}",
        String::from_utf8_lossy(&body_bytes)
    );

    let merged_dir = tmp.path().join("merged-concat-mixed");
    let loaded = PeftLora::load(&merged_dir).unwrap();
    // 2 + 3 = 5 total rank.
    assert_eq!(loaded.rank(), Some(5));
    let a = &loaded.tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"];
    let b = &loaded.tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"];
    assert_eq!(a.shape, vec![5, 4]);
    assert_eq!(b.shape, vec![3, 5]);
}

#[tokio::test]
async fn test_merge_concat_endpoint_rejects_target_modules_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    write_uniform_adapter(tmp.path(), "src-a", 2, 1.0);
    // Build a second adapter with a different target_modules set.
    let other_path = tmp.path().join("src-b");
    let mut tensors: BTreeMap<String, MergeTensor> = BTreeMap::new();
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        MergeTensor {
            shape: vec![2, 4],
            data: vec![1.0_f32; 8],
        },
    );
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        MergeTensor {
            shape: vec![3, 2],
            data: vec![1.0_f32; 6],
        },
    );
    let other = PeftLora {
        config: json!({
            "r": 2,
            "lora_alpha": 4.0,
            "target_modules": ["k_proj"],
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B"
        }),
        tensors,
    };
    other.save(&other_path).unwrap();

    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let body = json!({
        "sources": [
            { "name": "src-a", "weight": 1.0 },
            { "name": "src-b", "weight": 1.0 },
        ],
        "output_name": "merged-bad-targets",
        "mode": "concat",
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
        "expected 400 for target_modules mismatch, got {}: {}",
        status,
        String::from_utf8_lossy(&body_bytes)
    );
}
