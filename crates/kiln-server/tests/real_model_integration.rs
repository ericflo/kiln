//! Integration test: wire a tiny random-weight ModelRunner into the HTTP server
//! and verify /v1/chat/completions returns real generated text.

use std::collections::HashMap;

use axum::body::Body;
use axum::http::{Request, StatusCode};
// (#1082) Fully kt-typed: `tiny_weights` builds kt `Tensor`s directly into the
// (now kt-typed) GpuWeights / GpuLayerWeights / GpuFullAttentionWeights /
// GpuFfnWeights fields, and `AppState::new_real` takes a kt `Device`. No candle
// anywhere — kiln-server's candle-core dev-dep was dropped accordingly.
use kiln_tensor::{DType, Device, Tensor};
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
use kiln_server::training_queue::QueuedJob;

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
            qkv_proj_t: None,
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
            gate_up_proj_t: None,
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

#[cfg(feature = "vulkan")]
#[tokio::test]
async fn submit_grpo_dataset_path_route_defaults_to_vulkan_streaming_queue() {
    if !kiln_model::backend::vulkan::vulkan_is_available() {
        eprintln!("Vulkan unavailable, skipping route-level native GRPO dataset_path test");
        return;
    }

    let config = tiny_config();
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);

    let runner_tokenizer = test_tokenizer();
    let state_tokenizer = test_tokenizer();

    let runner = ModelRunner::new(weights, runner_tokenizer, config.clone());
    assert_eq!(
        runner.backend_name(),
        "vulkan",
        "bare real backend should select Vulkan by default when Vulkan is available"
    );

    let adapter_dir = tempfile::tempdir().unwrap();
    let state = AppState::new_real(
        config,
        runner,
        state_tokenizer,
        device,
        adapter_dir.path().to_path_buf(),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );
    let state_for_assert = state.clone();
    let app = api::router(state);

    let dataset = tempfile::NamedTempFile::new().unwrap();
    let first = json!({
        "messages": [{"role": "user", "content": "t1"}],
        "completions": [
            {"text": "t2", "reward": 1.0},
            {"text": "t3", "reward": 0.0}
        ]
    });
    std::fs::write(
        dataset.path(),
        format!(
            "{}\nthis is not json and must not be parsed at submit time\n",
            first
        ),
    )
    .unwrap();

    let body = json!({
        "dataset_path": format!("  {}  ", dataset.path().display()),
        "config": {
            "output_name": "api-jsonl-route",
            "auto_load": false,
            "lora_rank": 2
        }
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/train/grpo")
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
    assert_eq!(resp["state"], "queued");
    assert!(
        resp["message"]
            .as_str()
            .unwrap()
            .contains("Queued streamed GRPO training from dataset_path")
    );
    let job_id = resp["job_id"].as_str().unwrap().to_string();

    {
        let jobs = state_for_assert.training_jobs.read().unwrap();
        let job = jobs.get(&job_id).expect("queued job should be tracked");
        assert!(matches!(
            job.job_type,
            kiln_server::state::TrainingJobType::Grpo
        ));
        assert_eq!(job.state, kiln_train::TrainingState::Queued);
        assert_eq!(job.adapter_name, "api-jsonl-route");
    }

    let queued = state_for_assert
        .training_queue
        .lock()
        .unwrap()
        .pop()
        .expect("queued GRPO job");
    assert_eq!(queued.job_id, job_id);
    match queued.job {
        QueuedJob::Grpo(req) => {
            assert_eq!(req.groups.len(), 0);
            assert_eq!(
                req.dataset_path.as_deref(),
                Some(dataset.path().to_str().unwrap()),
                "route should trim and preserve dataset_path for the streaming worker"
            );
        }
        QueuedJob::Sft(_) => panic!("expected queued GRPO job"),
    }
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
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "Qwen3.5-4B".to_string(),
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
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "Qwen3.5-4B".to_string(),
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
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        42,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );

    assert_eq!(state.request_timeout.as_secs(), 42);
}

/// Test that default request timeout is 600 seconds.
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
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        600,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
    );

    assert_eq!(state.request_timeout.as_secs(), 600);
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
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "Qwen3.5-4B".to_string(),
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
    // Availability gate (candle device handle is only used to confirm a Metal
    // GPU exists); the runner + weights are built on the kt `Device::Metal(0)`
    // — the candle→kt forward-flip made every backend seam kt-typed (#1082).
    if kiln_model::backend::metal::try_new_metal().is_none() {
        return;
    }
    let device = Device::Metal(0);

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
        "Qwen3.5-4B".to_string(),
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

/// BF16 variant of the Metal e2e chat test. The projection weights are BF16,
/// so the decode path routes through the **fused BF16 Metal kernels** that
/// the FP32 test (head_dim=4 naive path) never touches: transposed-coop
/// GEMV (Q/K/V/O proj), fused MLP gate+up, and SiLU*mul. This is the
/// end-to-end plumbing gate for the candle-free fused-kernel flip (#1082) —
/// a wrong output shape / buffer wiring would crash or produce non-finite
/// logits and fail generation. Skipped gracefully without a Metal device.
#[cfg(feature = "metal")]
#[tokio::test]
async fn test_real_model_chat_completion_metal_bf16_fused() {
    if kiln_model::backend::metal::try_new_metal().is_none() {
        return;
    }
    let device = Device::Metal(0);

    let mut config = tiny_config();
    config.dtype = kiln_core::config::DType::BF16;

    // Build the standard tiny weights, then cast the projection weights
    // (and their pre-transposed forms) to BF16 so the decode fused kernels'
    // `_supports` gates pass. Norms / rotary / embed stay F32 (the forward
    // casts activations to the BF16 compute dtype).
    let mut weights = tiny_weights(&config, &device);
    let bf16 = |t: &Tensor| kiln_tensor::ops::cast(t, DType::BF16).expect("cast bf16");
    // Cast every weight (embed, norms, projections) to BF16 so the whole
    // forward flows in BF16 — matching the BF16 fused kernels and avoiding
    // F32-activation x BF16-weight matmul mismatches. Rotary tables stay F32
    // (the RoPE op consumes F32 cos/sin tables).
    weights.embed_tokens = bf16(&weights.embed_tokens);
    weights.embed_tokens_t = bf16(&weights.embed_tokens_t);
    weights.final_norm = bf16(&weights.final_norm);
    weights.layers[0].input_layernorm = bf16(&weights.layers[0].input_layernorm);
    weights.layers[0].post_attention_layernorm = bf16(&weights.layers[0].post_attention_layernorm);
    if let GpuAttentionWeights::Full(ref mut attn) = weights.layers[0].attention {
        attn.q_proj = bf16(&attn.q_proj);
        attn.k_proj = bf16(&attn.k_proj);
        attn.v_proj = bf16(&attn.v_proj);
        attn.o_proj = bf16(&attn.o_proj);
        attn.q_norm = bf16(&attn.q_norm);
        attn.k_norm = bf16(&attn.k_norm);
        attn.q_proj_t = bf16(&attn.q_proj_t);
        attn.k_proj_t = bf16(&attn.k_proj_t);
        attn.v_proj_t = bf16(&attn.v_proj_t);
        attn.o_proj_t = bf16(&attn.o_proj_t);
    }
    let mlp = &mut weights.layers[0].mlp;
    mlp.gate_proj = bf16(&mlp.gate_proj);
    mlp.up_proj = bf16(&mlp.up_proj);
    mlp.down_proj = bf16(&mlp.down_proj);
    mlp.gate_proj_t = bf16(&mlp.gate_proj_t);
    mlp.up_proj_t = bf16(&mlp.up_proj_t);
    mlp.down_proj_t = bf16(&mlp.down_proj_t);

    let runner = ModelRunner::new(weights, test_tokenizer(), config.clone());
    let state = AppState::new_real(
        config,
        runner,
        test_tokenizer(),
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters-bf16"),
        &kiln_server::config::MemoryConfig::default(),
        300,
        "Qwen3.5-4B".to_string(),
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
    assert!(resp["choices"][0]["message"]["content"].is_string());
    assert!(resp["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

// ---------------------------------------------------------------------------
// (#1082) End-to-end TRAINING smokes on Device::Metal(0).
//
// Until now nothing exercised SFT/GRPO/OPD on Metal *hardware*: the autograd
// math tests run the kt tape on CPU tensors, and the server route tests only
// assert that a job *queues*. These build the tiny random-weight model on
// Device::Metal(0) and drive a real (tiny) training run through the kt-native
// trainer (`for_device_kt` -> `MetalBackend`), so the forward AND the kt-tape
// backward + AdamW optimizer step all execute Metal kernels. Asserts the run
// completes, writes a `.safetensors` adapter, and does not blow up. Skipped
// gracefully without a Metal device.
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn metal_chat_msg(role: &str, content: &str) -> kiln_train::ChatMessage {
    kiln_train::ChatMessage {
        role: role.to_string(),
        content: content.to_string(),
    }
}

// The former `metal_gpu_guard()` process-global serialization mutex was
// removed (#1082): the shared `MetalCompanion` command-buffer stream is now
// correct under cross-thread concurrency (the deferred-commit pool no longer
// reorders data-dependent ops across command buffers — see the re-architected
// `kiln_tensor::metal_rt::commands`). These GPU-heavy Metal smokes now run
// concurrently with cargo's parallel test threads and remain deterministic,
// which is exactly the production guarantee (server inference races a training
// job on the same GPU).

#[cfg(feature = "metal")]
fn assert_adapter_written(out: &std::path::Path) {
    assert!(out.exists(), "adapter output dir {out:?} should exist");
    let entries: Vec<_> = std::fs::read_dir(out)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        entries.iter().any(|n| n.ends_with(".safetensors")),
        "expected an adapter .safetensors in {out:?}, got {entries:?}"
    );
}

/// Tiny random-weight model with all projections/norms/embeddings cast to BF16.
/// The kt tape-authoritative training adapters are BF16-only
/// (`base_dtype_supports_tape`), so SFT/GRPO/OPD smokes need a BF16 base.
/// Mirrors the cast recipe in `test_real_model_chat_completion_metal_bf16_fused`.
#[cfg(feature = "metal")]
fn tiny_weights_bf16(config: &ModelConfig, device: &Device) -> GpuWeights {
    let mut weights = tiny_weights(config, device);
    let bf16 = |t: &Tensor| kiln_tensor::ops::cast(t, DType::BF16).expect("cast bf16");
    weights.embed_tokens = bf16(&weights.embed_tokens);
    weights.embed_tokens_t = bf16(&weights.embed_tokens_t);
    weights.final_norm = bf16(&weights.final_norm);
    weights.layers[0].input_layernorm = bf16(&weights.layers[0].input_layernorm);
    weights.layers[0].post_attention_layernorm = bf16(&weights.layers[0].post_attention_layernorm);
    if let GpuAttentionWeights::Full(ref mut attn) = weights.layers[0].attention {
        attn.q_proj = bf16(&attn.q_proj);
        attn.k_proj = bf16(&attn.k_proj);
        attn.v_proj = bf16(&attn.v_proj);
        attn.o_proj = bf16(&attn.o_proj);
        attn.q_norm = bf16(&attn.q_norm);
        attn.k_norm = bf16(&attn.k_norm);
        attn.q_proj_t = bf16(&attn.q_proj_t);
        attn.k_proj_t = bf16(&attn.k_proj_t);
        attn.v_proj_t = bf16(&attn.v_proj_t);
        attn.o_proj_t = bf16(&attn.o_proj_t);
    }
    let mlp = &mut weights.layers[0].mlp;
    mlp.gate_proj = bf16(&mlp.gate_proj);
    mlp.up_proj = bf16(&mlp.up_proj);
    mlp.down_proj = bf16(&mlp.down_proj);
    mlp.gate_proj_t = bf16(&mlp.gate_proj_t);
    mlp.up_proj_t = bf16(&mlp.up_proj_t);
    mlp.down_proj_t = bf16(&mlp.down_proj_t);
    weights
}

/// Capture training loss per progress tick into a shared vec, for the
/// loss-decrease assertion.
#[cfg(feature = "metal")]
fn loss_capture_cb() -> (
    std::sync::Arc<std::sync::Mutex<Vec<f64>>>,
    kiln_train::trainer::ProgressCallback,
) {
    let losses = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let sink = losses.clone();
    let cb: kiln_train::trainer::ProgressCallback =
        Box::new(move |p: kiln_train::trainer::TrainingProgress| {
            sink.lock().unwrap().push(p.loss);
        });
    (losses, cb)
}

#[cfg(feature = "metal")]
fn assert_loss_decreases(losses: &[f64]) {
    assert!(!losses.is_empty(), "progress callback recorded no losses");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "all training losses must be finite, got {losses:?}"
    );
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "training loss should decrease over the run, got {losses:?}"
    );
}

#[cfg(feature = "metal")]
#[test]
fn test_real_model_sft_metal() {
    if kiln_model::backend::metal::try_new_metal().is_none() {
        eprintln!("No Metal device — skipping SFT-on-Metal smoke");
        return;
    }
    let device = Device::Metal(0);
    let mut config = tiny_config();
    config.dtype = kiln_core::config::DType::BF16;
    let weights = tiny_weights_bf16(&config, &device);
    let tokenizer = test_tokenizer();

    let examples = vec![
        kiln_train::SftExample {
            messages: vec![
                metal_chat_msg("user", "t1 t2 t3"),
                metal_chat_msg("assistant", "t2 t3 t1"),
            ],
        },
        kiln_train::SftExample {
            messages: vec![
                metal_chat_msg("user", "t3 t1"),
                metal_chat_msg("assistant", "t1 t2 t3"),
            ],
        },
    ];
    let sft_config = kiln_train::SftConfig {
        epochs: 3,
        learning_rate: 1e-3,
        lora_rank: 2,
        lora_alpha: 4.0,
        auto_load: false,
        seed: Some(0),
        ..Default::default()
    };

    let (losses, cb) = loss_capture_cb();
    let adapter_dir = tempfile::tempdir().unwrap();
    let out = kiln_train::trainer::sft_train(
        &examples,
        &sft_config,
        &config,
        &weights,
        &tokenizer,
        adapter_dir.path(),
        "sft-metal-smoke",
        Some(cb),
        None,
    )
    .expect("SFT training on Device::Metal(0) should complete");
    assert_adapter_written(&out);
    assert_loss_decreases(&losses.lock().unwrap());
}

/// Read the `lora_grad_norms` array out of the GRPO `train_receipt.json` the
/// trainer writes to the output dir. Returns `(num_modules, max_mean_norm)`: a
/// non-zero module count proves the tape walked and deposited grads (an
/// empty/severed tape records none), and a strictly-positive max mean-norm
/// proves the deposited grads carried real signal (not an all-zero no-op).
#[cfg(feature = "metal")]
fn receipt_lora_grad_norms(out: &std::path::Path) -> (usize, f64) {
    let receipt_path = out.join("train_receipt.json");
    let json = std::fs::read_to_string(&receipt_path)
        .unwrap_or_else(|e| panic!("read GRPO train receipt {receipt_path:?}: {e}"));
    let v: Value = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("parse GRPO train receipt {receipt_path:?}: {e}"));
    let arr = v["lora_grad_norms"].as_array().cloned().unwrap_or_default();
    let max_mean = arr
        .iter()
        .filter_map(|s| s["mean"].as_f64())
        .fold(0.0f64, f64::max);
    (arr.len(), max_mean)
}

/// GRPO on Metal smoke. Exercises the kt tape-authoritative GRPO producer
/// (`grpo_step_forward_backward_tape_authoritative_kt` + the
/// `grpo_tape_shim` scalar-loss tape root) on a real BF16 tiny model on
/// `Device::Metal(0)`. One group with VARIED rewards (1.0 / 0.0) so the
/// group-relative advantage is non-zero and a genuine policy gradient flows.
/// KL is off (`kl_coeff = 0`) and ECHO is disabled so the per-completion step
/// stays on the tape-authoritative path (no `no_policy_loss`, no env-CE root).
#[cfg(feature = "metal")]
#[test]
fn test_real_model_grpo_metal() {
    if kiln_model::backend::metal::try_new_metal().is_none() {
        eprintln!("No Metal device — skipping GRPO-on-Metal smoke");
        return;
    }
    let device = Device::Metal(0);
    let mut config = tiny_config();
    config.dtype = kiln_core::config::DType::BF16;
    let weights = tiny_weights_bf16(&config, &device);
    let tokenizer = test_tokenizer();

    // A few groups, each a user prompt + two scored completions with DIFFERENT
    // rewards so the within-group advantage is non-degenerate (1.0 vs 0.0).
    // GRPO does a single pass over the groups; with `PerSample` aggregation each
    // completion drives its own optimizer step + loss tick, so the run produces
    // a multi-element loss vector and exercises the producer per completion.
    // Legacy single-turn rollouts (no trajectory) keep ECHO inactive regardless.
    let mk_group = |prompt: &str, win: &str, lose: &str| kiln_train::GrpoGroup {
        messages: vec![metal_chat_msg("user", prompt)],
        completions: vec![
            kiln_train::ScoredRollout::legacy(win.to_string(), 1.0),
            kiln_train::ScoredRollout::legacy(lose.to_string(), 0.0),
        ],
    };
    let groups = vec![
        mk_group("t1 t2 t3", "t2 t3 t1", "t3 t1 t2"),
        mk_group("t3 t1", "t1 t2 t3", "t3 t3 t3"),
        mk_group("t2 t1 t3", "t3 t2 t1", "t1 t1 t1"),
    ];

    let mut grpo_config = kiln_train::GrpoConfig {
        learning_rate: 1e-3,
        kl_coeff: 0.0,
        lora_rank: 2,
        lora_alpha: 4.0,
        auto_load: false,
        seed: Some(0),
        // Each group already has reward variance; disable dynamic sampling so
        // none is filtered as degenerate.
        dynamic_sampling: false,
        // PerSample (one optimizer step per completion). The default TokenLevel
        // aggregation SUMS the per-completion grads within a group before the
        // step; since GRPO advantages are mean-zero within a group, on this
        // near-uniform tiny init the completions' grads (each ~ advantage x a
        // common direction) cancel to a zero group gradient — a real GRPO
        // degeneracy, not a Metal bug. PerSample sidesteps it so each step has a
        // genuine non-zero policy gradient to validate grad flow on Metal.
        loss_aggregation: kiln_train::LossAggregation::PerSample,
        ..kiln_train::GrpoConfig::default()
    };
    // Disable ECHO so the step stays on the kt tape-authoritative path.
    grpo_config.loss.echo = None;
    grpo_config.loss.no_policy_loss = false;

    let (losses, cb) = loss_capture_cb();
    let adapter_dir = tempfile::tempdir().unwrap();
    let out = kiln_train::trainer::grpo_train(
        &groups,
        &grpo_config,
        &config,
        &weights,
        &tokenizer,
        adapter_dir.path(),
        "grpo-metal-smoke",
        Some(cb),
        None,
    )
    .expect("GRPO training on Device::Metal(0) should complete");

    assert_adapter_written(&out);

    let losses = losses.lock().unwrap();
    assert!(!losses.is_empty(), "GRPO progress callback recorded no losses");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "all GRPO training losses must be finite, got {losses:?}"
    );

    // Gradients must have flowed through the kt tape-authoritative GRPO step:
    // the receipt records a per-module grad-norm summary for every LoRA module
    // that received a gradient. An empty list means the tape severed and no
    // grads reached the adapter — the failure mode this smoke guards against.
    // A strictly-positive max mean-norm additionally proves the grads carried
    // real policy-gradient signal (varied rewards => non-zero advantage), not a
    // degenerate all-zero deposit. NOTE: the scalar group loss can be ~0 even
    // when grads are large — a single two-completion group with opposite
    // advantages (+1 / -1) has its per-completion losses cancel in the group
    // mean, while the per-completion gradients are real and non-canceling.
    let (grad_norm_modules, max_mean_norm) = receipt_lora_grad_norms(&out);
    assert!(
        grad_norm_modules > 0,
        "GRPO step recorded no LoRA grad norms — gradients did not flow through \
         the kt tape-authoritative path on Metal (losses were {losses:?})"
    );
    assert!(
        max_mean_norm > 0.0,
        "GRPO LoRA grad norms are all zero — the tape walked but deposited no \
         signal (modules={grad_norm_modules}, losses={losses:?})"
    );
    eprintln!(
        "[GRPO-METAL] losses={losses:?} lora_grad_norm_modules={grad_norm_modules} \
         max_mean_norm={max_mean_norm}"
    );
}

/// OPD (off-policy distillation) on Metal smoke.
///
/// Exercises the kt-native OPD producer path on a real BF16 tiny model on
/// `Device::Metal(0)`: `opd_train` -> `opd_step_forward_backward_tape_authoritative`
/// -> the full Metal forward (`model_forward_no_head`) -> the kt-native OPD
/// scalar-loss tape root (`opd_tape_shim::try_tape_opd_scalar_mean_cuda_kt` ->
/// `kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_via_kt_tape`).
///
/// # End-to-end on Metal (#1082 Metal lane)
///
/// Like SFT (CE) and GRPO (policy-grad), OPD now runs fully on Metal: both
/// op-gaps that previously blocked it are closed.
///
///  (A) FORWARD: `log_softmax_last_dim` has a Metal MSL kernel
///      (`metal_log_softmax_last_axis`), so the kt-native OPD forward
///      (`per_position_forward_kt`) completes on Metal storage.
///
///  (B) BACKWARD: the OPD top-K reverse-KL backward is now a
///      device-agnostic analytic kt-composite
///      (`kt_api::opd_top_k_reverse_kl_phase_b_bwd_composite_kt`, derived
///      from the CUDA kernel's math and validated against finite-difference)
///      — the same "pure-kt analytic backward" pattern SFT's CE uses. The
///      kt-tape backward op (`CudaOpdTopKReverseKlPhaseBBackward::apply`)
///      routes CPU/Metal through the composite and keeps the perf-tuned CUDA
///      FFI kernel for `Device::Cuda(_)`.
///
/// This smoke now runs OPD to completion on Metal: it asserts the full
/// SFT/GRPO-style contract — adapter written, finite losses, and LoRA grads
/// flowed through the kt tape-authoritative path.
#[cfg(feature = "metal")]
#[test]
fn test_real_model_opd_metal() {
    use std::sync::Arc;

    if kiln_model::backend::metal::try_new_metal().is_none() {
        eprintln!("No Metal device — skipping OPD-on-Metal smoke");
        return;
    }
    let device = Device::Metal(0);
    let mut config = tiny_config();
    config.dtype = kiln_core::config::DType::BF16;
    let weights = tiny_weights_bf16(&config, &device);
    let tokenizer = test_tokenizer();

    // Off-policy replay: the student trains on the teacher-authored assistant
    // turn (no student sampling). Short shared-vocab tokens ("t1 t2 t3") keep
    // the rollout tiny.
    let prompts = vec![
        kiln_train::opd::OpdPrompt {
            messages: vec![
                metal_chat_msg("user", "t1 t2 t3"),
                metal_chat_msg("assistant", "t2 t3 t1"),
            ],
            teacher_extra_messages: vec![],
            trajectory: vec![],
        },
        kiln_train::opd::OpdPrompt {
            messages: vec![
                metal_chat_msg("user", "t3 t1"),
                metal_chat_msg("assistant", "t1 t2 t3"),
            ],
            teacher_extra_messages: vec![],
            trajectory: vec![],
        },
    ];

    // A built-in deterministic teacher so no real second model is needed. K=32
    // matches the OPD kernel envelope ({16, 32}) and the OpdConfig default; the
    // tiny model's vocab_size is 32, so the uniform top-K spans the full vocab.
    let teacher: Arc<dyn kiln_train::logit_source::LogitSource> = Arc::new(
        kiln_train::logit_source::DeterministicUniformLogitSource::new(
            "metal-smoke-teacher",
            config.vocab_size,
            32,
        ),
    );

    let mut opd_config = kiln_train::opd::OpdConfig {
        learning_rate: 1e-3,
        lora_rank: 2,
        lora_alpha: 4.0,
        auto_load: false,
        seed: Some(0),
        epochs: 1,
        ..Default::default()
    };
    // Force off-policy: on-policy is preflight-rejected, and off-policy
    // auto-scales samples_per_prompt -> 1 (one step per prompt), keeping the
    // smoke cheap.
    opd_config.training_mode = kiln_train::opd::OpdTrainingMode::OffPolicy;

    let (losses, cb) = loss_capture_cb();
    let adapter_dir = tempfile::tempdir().unwrap();
    let result = kiln_train::opd::opd_train(
        &prompts,
        &opd_config,
        &config,
        &weights,
        &tokenizer,
        teacher,
        adapter_dir.path(),
        "opd-metal-smoke",
        Some(cb),
    );

    match result {
        Ok(out) => {
            // Both op-gaps are closed (Metal `log_softmax` kernel + the
            // device-agnostic kt-composite OPD backward), so OPD runs to
            // completion on Metal — assert the full SFT/GRPO-style contract
            // (adapter written + finite losses + grads flowed).
            assert_adapter_written(&out);
            let losses = losses.lock().unwrap();
            assert!(!losses.is_empty(), "OPD progress callback recorded no losses");
            assert!(
                losses.iter().all(|l| l.is_finite()),
                "all OPD training losses must be finite, got {losses:?}"
            );
            let (grad_norm_modules, max_mean_norm) = receipt_lora_grad_norms(&out);
            assert!(
                grad_norm_modules > 0,
                "OPD step recorded no LoRA grad norms — gradients did not flow \
                 through the kt tape-authoritative path on Metal"
            );
            eprintln!(
                "[OPD-METAL] run completed (Metal OPD forward + composite backward): \
                 losses={losses:?} lora_grad_norm_modules={grad_norm_modules} \
                 max_mean_norm={max_mean_norm}"
            );
        }
        Err(e) => {
            // OPD now runs end-to-end on Metal (forward via the Metal
            // `log_softmax` kernel; backward via the device-agnostic
            // kt-composite). Any error here is a real regression — fail loudly.
            let chain = format!("{e:#}");
            panic!(
                "[OPD-METAL] OPD-on-Metal must now run to completion (both the \
                 forward `log_softmax` Metal kernel and the device-agnostic \
                 kt-composite backward are wired). Got error chain:\n{chain}"
            );
        }
    }
}
