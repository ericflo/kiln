//! Integration test: wire a tiny random-weight ModelRunner into the HTTP server
//! and verify /v1/chat/completions returns real generated text.

use std::collections::HashMap;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use candle_core::{DType, Device, Tensor};
use serde_json::{Value, json};
use tower::ServiceExt; // for `oneshot`

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::ModelRunner;
use kiln_model::forward::{
    GpuAttentionWeights, GpuFfnWeights, GpuFullAttentionWeights, GpuLayerWeights, GpuWeights,
};
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
        attn_output_gate: false,
        linear_num_key_heads: 0,
        linear_key_head_dim: 0,
        linear_num_value_heads: 0,
        linear_value_head_dim: 0,
        linear_conv_kernel_dim: 0,
        partial_rotary_factor: 1.0,
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
    let embed_t = embed.t().unwrap().contiguous().unwrap();
    let final_norm = Tensor::zeros((h,), DType::F32, device).unwrap();

    let q_proj = Tensor::randn(0.0_f32, 0.02, (num_heads * head_dim, h), device).unwrap();
    let k_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, h), device).unwrap();
    let v_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, h), device).unwrap();
    let o_proj = Tensor::randn(0.0_f32, 0.02, (h, num_heads * head_dim), device).unwrap();
    let q_proj_t = q_proj.t().unwrap().contiguous().unwrap();
    let k_proj_t = k_proj.t().unwrap().contiguous().unwrap();
    let v_proj_t = v_proj.t().unwrap().contiguous().unwrap();
    let o_proj_t = o_proj.t().unwrap().contiguous().unwrap();

    let gate_proj = Tensor::randn(0.0_f32, 0.02, (inter, h), device).unwrap();
    let up_proj = Tensor::randn(0.0_f32, 0.02, (inter, h), device).unwrap();
    let down_proj = Tensor::randn(0.0_f32, 0.02, (h, inter), device).unwrap();
    let gate_proj_t = gate_proj.t().unwrap().contiguous().unwrap();
    let up_proj_t = up_proj.t().unwrap().contiguous().unwrap();
    let down_proj_t = down_proj.t().unwrap().contiguous().unwrap();

    let layer = GpuLayerWeights {
        input_layernorm: Tensor::zeros((h,), DType::F32, device).unwrap(),
        post_attention_layernorm: Tensor::zeros((h,), DType::F32, device).unwrap(),
        attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Tensor::zeros((head_dim,), DType::F32, device).unwrap(),
            k_norm: Tensor::zeros((head_dim,), DType::F32, device).unwrap(),
            q_proj_t,
            k_proj_t,
            v_proj_t,
            o_proj_t,
            q_proj_marlin: None,
        }),
        mlp: GpuFfnWeights {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_t,
            up_proj_t,
            down_proj_t,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
        },
    };

    let rotary_inv_freq = kiln_model::forward::compute_rotary_inv_freq(
        config.rotary_dim(),
        config.rope_theta,
        device,
    )
    .unwrap();

    GpuWeights {
        embed_tokens: embed,
        embed_tokens_t: embed_t,
        layers: vec![layer],
        final_norm,
        rotary_inv_freq,
        mtp: None,
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
    let state = AppState::new_real(
        config,
        runner,
        state_tokenizer,
        device.clone(),
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "qwen3.5-4b-kiln".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );

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
async fn test_real_model_streaming_chat_completion() {
    let config = tiny_config();
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);

    let runner_tokenizer = test_tokenizer();
    let state_tokenizer = test_tokenizer();

    let runner = ModelRunner::new(weights, runner_tokenizer, config.clone());
    let state = AppState::new_real(
        config,
        runner,
        state_tokenizer,
        device.clone(),
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "qwen3.5-4b-kiln".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );

    let app = api::router(state);

    let body = json!({
        "messages": [{"role": "user", "content": "t1 t2 t3"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": true
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    let status = response.status();
    assert_eq!(status, StatusCode::OK, "expected 200 for streaming request");

    // Verify content-type is text/event-stream
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        content_type.contains("text/event-stream"),
        "expected text/event-stream, got {content_type}"
    );

    // Read the full SSE body
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&body_bytes);

    // Parse SSE events: lines starting with "data: "
    let data_lines: Vec<&str> = body_str
        .lines()
        .filter(|line| line.starts_with("data: ") || line.starts_with("data:"))
        .map(|line| {
            line.strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
                .unwrap_or(line)
        })
        .collect();

    assert!(
        data_lines.len() >= 3,
        "expected at least 3 data lines (role + tokens + [DONE]), got {}: {:?}",
        data_lines.len(),
        data_lines
    );

    // First chunk should have role: "assistant"
    let first: Value = serde_json::from_str(data_lines[0])
        .unwrap_or_else(|e| panic!("failed to parse first chunk: {e}\nraw: {}", data_lines[0]));
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert!(first["choices"][0]["finish_reason"].is_null());

    // Middle chunks may contain content, but tiny random weights can also
    // produce an EOS or empty-decoding token immediately. Keep this test about
    // streaming protocol correctness rather than random text quality.
    for line in &data_lines[1..data_lines.len() - 1] {
        if *line == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("failed to parse chunk: {e}\nraw: {line}"));
        assert_eq!(chunk["object"], "chat.completion.chunk");
    }

    // Last line should be [DONE]
    assert_eq!(
        *data_lines.last().unwrap(),
        "[DONE]",
        "stream should end with [DONE]"
    );

    // Second-to-last data line (before [DONE]) should have finish_reason
    let second_to_last = data_lines[data_lines.len() - 2];
    let finish_chunk: Value = serde_json::from_str(second_to_last)
        .unwrap_or_else(|e| panic!("failed to parse finish chunk: {e}\nraw: {second_to_last}"));
    let finish_reason = finish_chunk["choices"][0]["finish_reason"]
        .as_str()
        .expect("finish_reason should be a string");
    assert!(
        finish_reason == "stop" || finish_reason == "length",
        "unexpected finish_reason: {finish_reason}"
    );
}

/// Test that request timeout is configurable via config parameter.
#[tokio::test]
async fn test_request_timeout_configurable() {
    let config = tiny_config();
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);

    let runner_tokenizer = test_tokenizer();
    let state_tokenizer = test_tokenizer();

    let runner = ModelRunner::new(weights, runner_tokenizer, config.clone());
    let state = AppState::new_real(
        config,
        runner,
        state_tokenizer,
        device.clone(),
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        42,
        "qwen3.5-4b-kiln".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );

    assert_eq!(state.request_timeout.as_secs(), 42);
}

/// Test that default request timeout is 300 seconds.
#[tokio::test]
async fn test_default_request_timeout() {
    let config = tiny_config();
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);

    let runner_tokenizer = test_tokenizer();
    let state_tokenizer = test_tokenizer();

    let runner = ModelRunner::new(weights, runner_tokenizer, config.clone());
    let state = AppState::new_real(
        config,
        runner,
        state_tokenizer,
        device.clone(),
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "qwen3.5-4b-kiln".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );

    assert_eq!(state.request_timeout.as_secs(), 300);
}

#[tokio::test]
async fn test_health_with_real_backend() {
    let config = tiny_config();
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);

    let runner_tokenizer = test_tokenizer();
    let state_tokenizer = test_tokenizer();

    let runner = ModelRunner::new(weights, runner_tokenizer, config.clone());
    let state = AppState::new_real(
        config,
        runner,
        state_tokenizer,
        device.clone(),
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "qwen3.5-4b-kiln".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );

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
    let scheduler = &resp["scheduler"];
    assert_eq!(scheduler["waiting"], 0);
    assert_eq!(scheduler["running"], 0);
    assert!(scheduler["blocks_total"].as_u64().unwrap() > 0);
    let checks = resp["checks"].as_array().unwrap();
    assert!(
        checks
            .iter()
            .any(|check| check["name"] == "inference_prewarm_complete" && check["pass"] == true)
    );
}

/// End-to-end: HTTP → axum → ModelRunner → Metal → generate. Runs the tiny
/// random-weight model on `Device::Metal(0)`. Head_dim=4 routes through the
/// portable fallback rather than candle SDPA, so this validates that every
/// op in the non-fused path (embed, RMSNorm, RoPE, QK-norm, naive attention,
/// SwiGLU, sampling) executes on Apple Silicon end-to-end.
///
/// Skipped gracefully when Metal isn't available so the test stays portable
/// on Linux+CUDA hosts.
#[cfg(feature = "metal")]
#[tokio::test]
async fn test_real_model_chat_completion_metal() {
    let Some(device) = kiln_model::backend::metal::try_new_metal() else {
        return;
    };

    let config = tiny_config();
    let weights = tiny_weights(&config, &device);

    let runner_tokenizer = test_tokenizer();
    let state_tokenizer = test_tokenizer();

    let runner = ModelRunner::new(weights, runner_tokenizer, config.clone());
    let state = AppState::new_real(
        config,
        runner,
        state_tokenizer,
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "qwen3.5-4b-kiln".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );

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
    assert_eq!(resp["object"], "chat.completion");
    assert!(resp["choices"][0]["message"]["content"].is_string());
    assert!(resp["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}
