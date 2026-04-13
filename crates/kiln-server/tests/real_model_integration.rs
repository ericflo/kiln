//! Integration test: wire a tiny random-weight ModelRunner into the HTTP server
//! and verify /v1/chat/completions returns real generated text.

use std::collections::HashMap;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};
use tower::ServiceExt; // for `oneshot`

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::{
    GpuAttentionWeights, GpuFfnWeights, GpuFullAttentionWeights, GpuLayerWeights, GpuWeights,
};
use kiln_model::ModelRunner;
use kiln_server::api;
use kiln_server::state::AppState;

/// Create a tiny model config for testing.
fn tiny_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_layers: 1,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 32,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::FP32,
        num_full_attention_layers: 1,
        full_attention_interval: 1,
    }
}

/// Create random GPU weights matching the tiny config.
fn tiny_weights(config: &ModelConfig, device: &Device) -> GpuWeights {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;

    let embed = Tensor::randn(0.0_f32, 0.02, (vocab, h), device).unwrap();
    let final_norm = Tensor::ones((h,), DType::F32, device).unwrap();

    let layer = GpuLayerWeights {
        input_layernorm: Tensor::ones((h,), DType::F32, device).unwrap(),
        post_attention_layernorm: Tensor::ones((h,), DType::F32, device).unwrap(),
        attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
            q_proj: Tensor::randn(0.0_f32, 0.02, (num_heads * head_dim, h), device).unwrap(),
            k_proj: Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, h), device).unwrap(),
            v_proj: Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, h), device).unwrap(),
            o_proj: Tensor::randn(0.0_f32, 0.02, (h, num_heads * head_dim), device).unwrap(),
            q_norm: Tensor::ones((head_dim,), DType::F32, device).unwrap(),
            k_norm: Tensor::ones((head_dim,), DType::F32, device).unwrap(),
        }),
        mlp: GpuFfnWeights {
            gate_proj: Tensor::randn(0.0_f32, 0.02, (inter, h), device).unwrap(),
            up_proj: Tensor::randn(0.0_f32, 0.02, (inter, h), device).unwrap(),
            down_proj: Tensor::randn(0.0_f32, 0.02, (h, inter), device).unwrap(),
        },
    };

    GpuWeights {
        embed_tokens: embed,
        layers: vec![layer],
        final_norm,
    }
}

/// Create a minimal tokenizer for testing.
///
/// The vocab includes ChatML special tokens so that `apply_chat_template`
/// produces a prompt that can be tokenized (each special token maps to its own ID).
fn test_tokenizer() -> KilnTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    // Reserve 0-19 for regular tokens
    for i in 0u32..20 {
        let c = format!("t{i}");
        vocab.insert(c, i);
    }
    // ChatML-related tokens as regular vocab entries so BPE can emit them
    vocab.insert("<|im_start|>".to_string(), 20);
    vocab.insert("<|im_end|>".to_string(), 21);
    vocab.insert("user".to_string(), 22);
    vocab.insert("assistant".to_string(), 23);
    vocab.insert("\n".to_string(), 24);
    // Pad to vocab_size=32
    for i in 25u32..32 {
        vocab.insert(format!("x{i}"), i);
    }

    let json = json!({
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": []
        },
        "added_tokens": [
            {
                "id": 0,
                "content": "<|endoftext|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            },
            {
                "id": 20,
                "content": "<|im_start|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            },
            {
                "id": 21,
                "content": "<|im_end|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ]
    });

    let bytes = serde_json::to_vec(&json).unwrap();
    KilnTokenizer::from_bytes(&bytes).unwrap()
}

#[tokio::test]
async fn test_real_model_chat_completion() {
    let config = tiny_config();
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);

    let runner_tokenizer = test_tokenizer();
    let state_tokenizer = test_tokenizer();

    let runner = ModelRunner::new(weights, runner_tokenizer, config.clone());
    let state = AppState::new_real(config, runner, state_tokenizer);

    let app = api::router(state);

    let body = json!({
        "messages": [{"role": "user", "content": "t1 t2 t3"}],
        "max_tokens": 5,
        "temperature": 0.0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    let status = response.status();
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();

    if status != StatusCode::OK {
        let body_str = String::from_utf8_lossy(&body_bytes);
        panic!("Expected 200, got {status}: {body_str}");
    }

    let resp: Value = serde_json::from_slice(&body_bytes).unwrap();

    // Verify response structure
    assert_eq!(resp["object"], "chat.completion");
    assert!(resp["id"].as_str().unwrap().starts_with("chatcmpl-"));

    let choices = resp["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0]["message"]["role"], "assistant");

    // The model produces random tokens, but the content should be a string
    assert!(choices[0]["message"]["content"].is_string());

    // Verify finish_reason is either "stop" or "length"
    let finish = choices[0]["finish_reason"].as_str().unwrap();
    assert!(
        finish == "stop" || finish == "length",
        "unexpected finish_reason: {finish}"
    );

    // Usage should have completion_tokens > 0 (model generated something)
    let usage = &resp["usage"];
    assert!(usage["completion_tokens"].as_u64().unwrap() > 0);
    assert!(usage["total_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_health_with_real_backend() {
    let config = tiny_config();
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);

    let runner_tokenizer = test_tokenizer();
    let state_tokenizer = test_tokenizer();

    let runner = ModelRunner::new(weights, runner_tokenizer, config.clone());
    let state = AppState::new_real(config, runner, state_tokenizer);

    let app = api::router(state);

    let request = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let resp: Value = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(resp["status"], "ok");
    assert_eq!(resp["backend"], "model");
    // scheduler should be null for real backend
    assert!(resp["scheduler"].is_null());
}
