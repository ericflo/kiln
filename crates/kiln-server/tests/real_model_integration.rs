//! Integration test: wire a tiny random-weight ModelRunner into the HTTP server
//! and verify /v1/chat/completions returns real generated text.

use std::collections::HashMap;
use std::sync::{Arc, TryLockError};
use std::time::{Duration, Instant};

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
use kiln_model::forward::{
    GpuAttentionWeights, GpuFfnWeights, GpuFullAttentionWeights, GpuLayerWeights,
    GpuLinearAttentionWeights, GpuWeights, LinearAttentionState,
};
use kiln_model::lora_loader::LoraSourceIdentity;
use kiln_model::{LoraWeights, ModelRunner};
use kiln_server::api;
use kiln_server::state::{AppState, ModelBackend};

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
            qkv_proj_w8: None,
            o_proj_w8: None,
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
            gate_up_proj_w8: None,
            down_proj_w8: None,
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
        lm_head_w8: None,
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

fn synthetic_base_teacher_identity(
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    backend: &str,
) -> Arc<kiln_train::TeacherIdentityV1> {
    let tokenizer_vocab_sha256 = tokenizer
        .vocab_identity_sha256()
        .strip_prefix("sha256:")
        .unwrap()
        .to_string();
    let tokenizer_config_sha256 = tokenizer
        .tokenizer_config_sha256()
        .unwrap()
        .strip_prefix("sha256:")
        .unwrap()
        .to_string();
    Arc::new(
        kiln_train::TeacherIdentityV1::new(
            "Qwen3.5-4B",
            "a".repeat(64),
            tokenizer_vocab_sha256,
            tokenizer_config_sha256,
            None,
            config.vocab_size as u32,
            config.vocab_size.min(256) as u32,
            config.max_position_embeddings as u32,
            65_536,
            format!("kiln-test/{backend}"),
            "d".repeat(64),
        )
        .unwrap(),
    )
}

fn tiny_real_state_with_timeout(config: ModelConfig, request_timeout: Duration) -> AppState {
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);
    let runner = ModelRunner::new(weights, test_tokenizer(), config.clone());
    let state_tokenizer = test_tokenizer();
    let base_teacher_identity =
        synthetic_base_teacher_identity(&config, &state_tokenizer, runner.backend_name());
    let mut state = AppState::new_real(
        config,
        runner,
        state_tokenizer,
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        request_timeout.as_secs().max(1),
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        Some(base_teacher_identity),
    );
    // Production configuration is second-granularity. Integration tests use a
    // shorter duration so lifecycle regressions fail quickly and locally.
    state.request_timeout = request_timeout;
    state
}

fn real_runner(state: &AppState) -> Arc<std::sync::RwLock<ModelRunner>> {
    match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner.clone(),
        ModelBackend::Mock { .. } => panic!("expected real model backend"),
    }
}

fn prompt_logprob_request(prompt: &[u32], top_k: usize) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "prompt": prompt,
                "max_tokens": 0,
                "prompt_logprobs": top_k
            })
            .to_string(),
        ))
        .unwrap()
}

#[tokio::test]
async fn adapter_load_publishes_the_exact_loaded_content_revision() {
    let adapters = tempfile::tempdir().unwrap();
    let adapter_dir = adapters.path().join("revisioned");
    std::fs::create_dir_all(&adapter_dir).unwrap();
    std::fs::write(
        adapter_dir.join("adapter_config.json"),
        br#"{"r":1,"lora_alpha":1.0,"target_modules":[]}"#,
    )
    .unwrap();
    let tensor_bytes = 0.0f32.to_le_bytes();
    let ignored =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &tensor_bytes)
            .unwrap();
    let weights = safetensors::tensor::serialize([("ignored.weight", ignored)], None).unwrap();
    std::fs::write(adapter_dir.join("adapter_model.safetensors"), weights).unwrap();

    let exact_source = LoraSourceIdentity::from_adapter_dir(&adapter_dir).unwrap();
    let expected_revision = exact_source.content_revision();
    let mut state = tiny_real_state_with_timeout(tiny_config(), Duration::from_secs(1));
    state.adapter_dir = adapters.path().to_path_buf();
    let state_for_assert = state.clone();
    let runner = real_runner(&state);
    let app = api::router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/adapters/load")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"revisioned"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["name"], "revisioned");
    assert_eq!(body["content_revision"], expected_revision);

    let published = state_for_assert
        .loaded_adapter_identity()
        .expect("loaded identity published with the runner flip");
    assert_eq!(published.name, "revisioned");
    assert_eq!(published.content_revision, expected_revision);
    let runner = runner.read().unwrap();
    let loaded_source = runner
        .active_lora()
        .and_then(LoraWeights::source_identity)
        .expect("runner retained the exact loader source identity");
    assert_eq!(loaded_source, &exact_source);
    assert_eq!(loaded_source.content_revision(), published.content_revision);
}

#[tokio::test]
async fn quarantined_backend_rejects_training_admission_without_publication() {
    let state = tiny_real_state_with_timeout(tiny_config(), Duration::from_secs(1));
    let state_for_assert = state.clone();
    let backend_health = match state.backend.as_ref() {
        ModelBackend::Real { backend_health, .. } => backend_health.clone(),
        ModelBackend::Mock { .. } => unreachable!("test constructed a real backend"),
    };
    backend_health.quarantine("injected admission quarantine");
    let app = api::router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/train/sft")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"examples":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{json}");
    assert_eq!(json["error"]["code"], "backend_quarantined");
    assert!(state_for_assert.training_jobs.read().unwrap().is_empty());
    assert_eq!(state_for_assert.training_queue.lock().unwrap().len(), 0);
}

#[tokio::test]
async fn queued_training_transition_fails_while_quarantined_reader_is_retained() {
    use kiln_server::state::{TrainingJobInfo, TrainingJobType};
    use kiln_server::training_queue::{QueueEntry, QueuedJob, spawn_training_worker};
    use kiln_train::{SftRequest, TrainingState};

    let mut state = tiny_real_state_with_timeout(tiny_config(), Duration::from_secs(1));
    let adapter_dir = tempfile::tempdir().unwrap();
    state.adapter_dir = adapter_dir.path().to_path_buf();
    let job_id = "quarantine-transition".to_string();
    state.training_jobs.write().unwrap().insert(
        job_id.clone(),
        TrainingJobInfo {
            job_id: job_id.clone(),
            adapter_name: "never-started".to_string(),
            job_type: TrainingJobType::Sft,
            state: TrainingState::Queued,
            progress: 0.0,
            loss: None,
            epoch: None,
            adapter_path: None,
            submitted_at: Instant::now(),
            submitted_unix_ms: 0,
            auto_load: false,
            consumed_correction_ids: Vec::new(),
            finished_at: None,
            finished_unix_ms: None,
            error: None,
            linked_eval_job_ids: Vec::new(),
            post_eval_verdict: None,
            gate_outcome: None,
            cancel_requested: Default::default(),
            loss_history: Vec::new(),
        },
    );
    state.training_queue.lock().unwrap().push(QueueEntry {
        job_id: job_id.clone(),
        reserved_bytes: 0,
        teacher_bindings: Vec::new(),
        job: QueuedJob::Sft(SftRequest {
            examples: Vec::new(),
            dataset_path: None,
            dataset: None,
            config: Default::default(),
            post_eval: None,
        }),
    });

    let retained_inference = state.gpu_lock.clone().read_owned().await;
    match state.backend.as_ref() {
        ModelBackend::Real { backend_health, .. } => {
            backend_health.quarantine("injected queued-transition quarantine")
        }
        ModelBackend::Mock { .. } => unreachable!("test constructed a real backend"),
    }
    spawn_training_worker(state.clone(), state.shutdown.clone());

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let terminal = state
                .training_jobs
                .read()
                .unwrap()
                .get(&job_id)
                .is_some_and(|job| job.state == TrainingState::Failed);
            if terminal {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("queued job must reject without waiting for retained GPU ownership");

    let jobs = state.training_jobs.read().unwrap();
    let failed = jobs.get(&job_id).unwrap();
    assert_eq!(failed.state, TrainingState::Failed);
    assert!(
        failed
            .error
            .as_deref()
            .is_some_and(|error| error.contains("requires restart")),
        "{:?}",
        failed.error
    );
    assert!(state.gpu_lock.try_write().is_err());
    drop(jobs);
    state
        .shutdown
        .store(true, std::sync::atomic::Ordering::Relaxed);
    drop(retained_inference);
}

#[cfg(feature = "vulkan")]
#[tokio::test]
async fn submit_grpo_dataset_path_route_defaults_to_vulkan_streaming_queue() {
    use kiln_server::training_queue::QueuedJob;

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
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        300,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        None,
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
        _ => panic!("expected queued GRPO job"),
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
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        300,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        None,
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
async fn test_real_model_one_token_prompt_logprobs_is_exactly_null() {
    let request_timeout = Duration::from_millis(40);
    let state = tiny_real_state_with_timeout(tiny_config(), request_timeout);
    let state_for_assert = state.clone();
    let expected_fingerprint = state.base_teacher_identity.as_ref().unwrap().fingerprint();
    let gpu_lock = state.gpu_lock.clone();
    let runner = real_runner(&state);
    let held_read = gpu_lock.clone().read_owned().await;
    let app = api::router(state);
    let runner_strong_count = Arc::strong_count(&runner);

    // A one-token prompt has no predecessor to score. Holding a read permit
    // proves this response takes the no-work path before exclusive admission.
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        app.oneshot(prompt_logprob_request(&[7], 2)),
    )
    .await
    .expect("one-token prompt-logprobs request should not wait for admission")
    .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let response_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response_json["system_fingerprint"], expected_fingerprint);
    assert_eq!(
        response_json["choices"][0]["prompt_logprobs"],
        json!([null])
    );
    assert_eq!(
        Arc::strong_count(&runner),
        runner_strong_count,
        "the no-work path must not clone a runner into a blocking scorer"
    );
    assert_eq!(
        state_for_assert
            .metrics
            .active_requests
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );

    drop(held_read);
    drop(
        gpu_lock
            .try_write()
            .expect("one-token response must leave GPU admission writable"),
    );
}

#[tokio::test]
async fn test_real_model_prompt_logprobs_requires_verified_identity() {
    let mut state = tiny_real_state_with_timeout(tiny_config(), Duration::from_secs(1));
    state.base_teacher_identity = None;
    let gpu_lock = state.gpu_lock.clone();
    let app = api::router(state);

    let response = app.oneshot(prompt_logprob_request(&[7], 2)).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let response_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response_json["error"]["code"], "internal_error");
    assert!(
        response_json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no verified base teacher identity")
    );
    drop(
        gpu_lock
            .try_write()
            .expect("identity rejection must happen before GPU admission"),
    );
}

#[tokio::test]
async fn test_real_model_prompt_logprobs_times_out_before_exclusive_admission() {
    let request_timeout = Duration::from_millis(40);
    let state = tiny_real_state_with_timeout(tiny_config(), request_timeout);
    let state_for_assert = state.clone();
    let gpu_lock = state.gpu_lock.clone();
    let runner = real_runner(&state);
    let held_read = gpu_lock.clone().read_owned().await;
    let app = api::router(state);
    let runner_strong_count = Arc::strong_count(&runner);
    let started = Instant::now();

    let response = tokio::time::timeout(
        Duration::from_secs(1),
        app.oneshot(prompt_logprob_request(&[1, 2, 3], 2)),
    )
    .await
    .expect("admission timeout response should settle promptly")
    .unwrap();
    let elapsed = started.elapsed();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::REQUEST_TIMEOUT,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(
        elapsed >= request_timeout,
        "408 arrived before the configured admission deadline: {elapsed:?}"
    );
    let response_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response_json["error"]["code"], "request_timeout");

    assert_eq!(
        Arc::strong_count(&runner),
        runner_strong_count,
        "timed-out admission must not clone a runner into spawn_blocking"
    );
    drop(
        runner
            .try_write()
            .expect("no blocking scorer may hold the runner after admission timeout"),
    );
    assert_eq!(
        state_for_assert
            .metrics
            .active_requests
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "timed-out request must leave no active HTTP lifecycle"
    );
    assert_eq!(
        state_for_assert
            .metrics
            .requests_timeout
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    drop(held_read);
    drop(
        gpu_lock
            .try_write()
            .expect("timed-out exclusive waiter must be removed from GPU admission"),
    );
}

#[tokio::test]
async fn test_real_model_prompt_logprobs_rejects_active_adapter() {
    let state = tiny_real_state_with_timeout(tiny_config(), Duration::from_secs(1));
    let gpu_lock = state.gpu_lock.clone();
    let runner = real_runner(&state);
    runner
        .write()
        .unwrap()
        .swap_lora(Some(LoraWeights {
            layers: Vec::new(),
            mtp: None,
            rank: 1,
            alpha: 1.0,
            scale: 1.0,
            source_identity: None,
        }))
        .unwrap();
    let app = api::router(state);

    let response = app
        .oneshot(prompt_logprob_request(&[1, 2, 3], 2))
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "{}",
        String::from_utf8_lossy(&body)
    );
    let response_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response_json["error"]["code"], "completion_invalid_request");
    assert!(
        response_json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("base-model only")
    );
    drop(
        gpu_lock
            .try_write()
            .expect("adapter rejection must not leak exclusive GPU admission"),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_real_model_prompt_logprobs_timeout_drains_started_scorer() {
    let request_timeout = Duration::from_millis(100);
    let mut config = tiny_config();
    config.max_position_embeddings = 2048;
    let state = tiny_real_state_with_timeout(config, request_timeout);
    let state_for_assert = state.clone();
    let gpu_lock = state.gpu_lock.clone();
    let runner = real_runner(&state);
    let app = api::router(state);
    let runner_strong_count = Arc::strong_count(&runner);
    let prompt = (0..1024)
        .map(|index| (index % 32) as u32)
        .collect::<Vec<_>>();
    let request_started = Instant::now();
    let mut request_task = tokio::spawn(async move {
        app.oneshot(prompt_logprob_request(&prompt, 0))
            .await
            .unwrap()
    });

    // `real_prompt_logprobs` clones the runner only after exclusive admission;
    // the blocking closure then holds a runner read lock for the full forward.
    // Requiring both observations distinguishes the worker from the brief
    // pre-spawn health check and proves the timeout fires after work starts.
    let worker_observed_at = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let worker_owns_runner_clone = Arc::strong_count(&runner) > runner_strong_count;
            let worker_holds_runner_read = match runner.try_write() {
                Ok(guard) => {
                    drop(guard);
                    false
                }
                Err(TryLockError::WouldBlock) => true,
                Err(TryLockError::Poisoned(error)) => {
                    panic!("runner lock poisoned while observing scorer: {error}")
                }
            };
            if worker_owns_runner_clone && worker_holds_runner_read {
                break Instant::now();
            }
            assert!(
                !request_task.is_finished(),
                "prompt-logprobs request settled before a blocking scorer was observed"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("blocking prompt-logprobs scorer should start before its deadline");

    tokio::time::sleep_until(tokio::time::Instant::from_std(
        request_started + request_timeout + Duration::from_millis(20),
    ))
    .await;
    assert!(
        !request_task.is_finished(),
        "HTTP response must remain pending while the timed-out scorer settles"
    );
    assert_eq!(
        state_for_assert
            .metrics
            .active_requests
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "request lifecycle must remain active during scorer settlement"
    );
    assert_eq!(
        state_for_assert
            .metrics
            .requests_timeout
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "timeout metric is recorded only after scorer settlement"
    );
    assert!(
        gpu_lock.try_read().is_err(),
        "started scorer must retain exclusive admission during settlement"
    );

    let response = tokio::time::timeout(Duration::from_secs(10), &mut request_task)
        .await
        .expect("cancelled scorer should settle within the integration-test ceiling")
        .expect("request task should join cleanly");
    let response_elapsed = request_started.elapsed();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::REQUEST_TIMEOUT,
        "{}",
        String::from_utf8_lossy(&body)
    );
    assert!(worker_observed_at >= request_started);
    assert!(
        response_elapsed >= request_timeout,
        "408 arrived before the configured scoring deadline: {response_elapsed:?}"
    );

    // The handler cannot make spawn_blocking stop synchronously. Its timeout
    // branch must cancel and await the worker; immediate write access proves
    // the 408 was withheld until the exclusive permit and runner read released.
    drop(
        gpu_lock
            .try_write()
            .expect("timeout response returned before blocking scorer settlement"),
    );
    drop(
        runner
            .try_write()
            .expect("settled scorer must release its runner read lock"),
    );
    assert_eq!(
        Arc::strong_count(&runner),
        runner_strong_count,
        "settled scorer must drop its captured runner clone"
    );
    assert_eq!(
        state_for_assert
            .metrics
            .active_requests
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        state_for_assert
            .metrics
            .requests_timeout
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    let settlement = match state_for_assert.backend.as_ref() {
        ModelBackend::Real { backend_health, .. } => backend_health
            .external_yield_sync_stats()
            .into_iter()
            .find(|stats| stats.boundary == "prompt-logprobs scoring"),
        ModelBackend::Mock { .. } => unreachable!("test constructed a real backend"),
    }
    .expect("timed-out scorer must settle the backend before returning 408");
    assert_eq!(settlement.calls, 1);
    assert_eq!(settlement.failures, 0);
}

#[tokio::test]
async fn test_real_model_prompt_logprobs_match_full_forward_reference() {
    const TOP_K: usize = 2;
    // The production scorer projects 32 normalized-hidden rows at a time.
    // Crossing that boundary proves the real no-head/chunked-LM-head path,
    // including its final short chunk, rather than only its first iteration.
    const PROMPT_LEN: usize = 35;

    let config = tiny_config();
    let device = Device::Cpu;
    let weights = tiny_weights(&config, &device);
    let reference_weights = weights.clone();

    let full_forward_rows = |prompt_ids: &[u32]| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let backend = kiln_model::backend::for_device_kt(&device);
        let mut linear_state = LinearAttentionState::new_with_batch_for_inference_runtime(
            &config,
            1,
            &device,
            backend.as_ref(),
        )
        .expect("reference inference linear state");
        let logits = kiln_model::forward::model_forward_kt(
            &*backend,
            prompt_ids,
            &reference_weights,
            &config,
            None,
            Some(&mut linear_state),
            None,
        )
        .expect("full-forward reference logits");
        let logprobs = kiln_tensor::ops::log_softmax_last_dim_f32(&logits)
            .expect("F32 reference log-softmax")
            .squeeze(0)
            .expect("reference batch squeeze")
            .to_vec2::<f32>()
            .expect("reference logprobs to host");
        let logits = logits
            .squeeze(0)
            .expect("reference logits batch squeeze")
            .to_vec2::<f32>()
            .expect("reference logits to host");
        (logits, logprobs)
    };
    let ranked_token_ids = |row: &[f32]| -> Vec<usize> {
        let mut ids = (0..row.len()).collect::<Vec<_>>();
        ids.sort_unstable_by(|&left, &right| {
            row[right]
                .partial_cmp(&row[left])
                .expect("reference logits are finite")
                .then_with(|| left.cmp(&right))
        });
        ids
    };

    // A token at position i cannot affect logits row i-1. Choose one observed
    // token inside TOP_K and the next outside TOP_K, then retain those
    // guarantees when the remaining causal suffix is appended.
    let first_token = 1u32;
    let (first_logits, _) = full_forward_rows(&[first_token]);
    let observed_inside_top_k = ranked_token_ids(&first_logits[0])[0] as u32;
    let (second_logits, _) = full_forward_rows(&[first_token, observed_inside_top_k]);
    let second_row_top_k = ranked_token_ids(&second_logits[1]);
    let observed_outside_top_k = (0..config.vocab_size)
        .find(|token_id| !second_row_top_k[..TOP_K].contains(token_id))
        .expect("tiny vocabulary has a token outside top-K")
        as u32;

    let mut prompt_ids = vec![first_token, observed_inside_top_k, observed_outside_top_k];
    while prompt_ids.len() < PROMPT_LEN {
        prompt_ids.push(((prompt_ids.len() * 7 + 3) % config.vocab_size) as u32);
    }
    let (reference_logits, reference_logprobs) = full_forward_rows(&prompt_ids);
    assert_eq!(reference_logits.len(), prompt_ids.len());
    assert_eq!(reference_logprobs.len(), prompt_ids.len());

    let runner = ModelRunner::new(weights, test_tokenizer(), config.clone());
    let state_tokenizer = test_tokenizer();
    let base_teacher_identity =
        synthetic_base_teacher_identity(&config, &state_tokenizer, runner.backend_name());
    let state = AppState::new_real(
        config.clone(),
        runner,
        state_tokenizer,
        device,
        std::path::PathBuf::from("/tmp/kiln-test-adapters"),
        &kiln_server::config::MemoryConfig::default(),
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        300,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        Some(base_teacher_identity),
    );
    let backend_health = match state.backend.as_ref() {
        ModelBackend::Real { backend_health, .. } => backend_health.clone(),
        ModelBackend::Mock { .. } => unreachable!("test constructed a real backend"),
    };
    let app = api::router(state);

    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "prompt": prompt_ids.clone(),
                "max_tokens": 0,
                "prompt_logprobs": TOP_K
            })
            .to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let response_json: Value = serde_json::from_slice(&body).unwrap();
    let rows = response_json["choices"][0]["prompt_logprobs"]
        .as_array()
        .expect("prompt_logprobs array");
    assert_eq!(rows.len(), prompt_ids.len());
    assert!(
        rows[0].is_null(),
        "the first prompt token has no predecessor"
    );

    let mut saw_k = false;
    let mut saw_k_plus_one = false;
    for position in 1..prompt_ids.len() {
        let reference_logits_row = &reference_logits[position - 1];
        let reference_logprob_row = &reference_logprobs[position - 1];
        let ranked = ranked_token_ids(reference_logits_row);
        let observed_id = prompt_ids[position] as usize;
        let observed_rank = ranked
            .iter()
            .position(|&token_id| token_id == observed_id)
            .expect("observed token is in model vocabulary")
            + 1;
        let observed_in_top_k = ranked[..TOP_K].contains(&observed_id);
        let expected_cardinality = TOP_K + usize::from(!observed_in_top_k);

        let row = rows[position].as_object().expect("scored prompt row");
        assert_eq!(row.len(), expected_cardinality, "position {position}");
        saw_k |= row.len() == TOP_K;
        saw_k_plus_one |= row.len() == TOP_K + 1;

        let observed = row
            .get(&observed_id.to_string())
            .unwrap_or_else(|| panic!("position {position} omitted observed token {observed_id}"));
        let actual_logprob = observed["logprob"].as_f64().unwrap() as f32;
        let expected_logprob = reference_logprob_row[observed_id];
        assert!(
            (actual_logprob - expected_logprob).abs() <= 1e-5,
            "position {position} observed token {observed_id}: got {actual_logprob}, expected {expected_logprob}"
        );
        assert_eq!(
            observed["rank"].as_u64().unwrap() as usize,
            observed_rank,
            "position {position} observed-token rank"
        );

        for (rank_index, &top_token_id) in ranked[..TOP_K].iter().enumerate() {
            let top = row.get(&top_token_id.to_string()).unwrap_or_else(|| {
                panic!("position {position} omitted top-K token {top_token_id}")
            });
            assert_eq!(top["rank"].as_u64().unwrap() as usize, rank_index + 1);
            let actual = top["logprob"].as_f64().unwrap() as f32;
            assert!((actual - reference_logprob_row[top_token_id]).abs() <= 1e-5);
        }
    }
    assert!(
        saw_k,
        "prompt must exercise observed-token top-K deduplication"
    );
    assert!(
        saw_k_plus_one,
        "prompt must exercise observed-token inclusion outside top-K"
    );

    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "prompt": prompt_ids.clone(),
                "max_tokens": 0,
                "prompt_logprobs": 0
            })
            .to_string(),
        ))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let response_json: Value = serde_json::from_slice(&body).unwrap();
    let rows = response_json["choices"][0]["prompt_logprobs"]
        .as_array()
        .expect("prompt_logprobs array for K=0");
    assert!(rows[0].is_null());
    for position in 1..prompt_ids.len() {
        let observed_id = prompt_ids[position] as usize;
        let row = rows[position].as_object().expect("K=0 scored prompt row");
        assert_eq!(row.len(), 1, "K=0 position {position}");
        let observed = row
            .get(&observed_id.to_string())
            .unwrap_or_else(|| panic!("K=0 position {position} omitted token {observed_id}"));
        let ranked = ranked_token_ids(&reference_logits[position - 1]);
        let expected_rank = ranked
            .iter()
            .position(|&token_id| token_id == observed_id)
            .unwrap()
            + 1;
        assert_eq!(observed["rank"].as_u64().unwrap() as usize, expected_rank);
        let actual = observed["logprob"].as_f64().unwrap() as f32;
        assert!((actual - reference_logprobs[position - 1][observed_id]).abs() <= 1e-5);
    }
    let settlement = backend_health
        .external_yield_sync_stats()
        .into_iter()
        .find(|stats| stats.boundary == "prompt-logprobs scoring")
        .expect("real prompt scoring must publish its backend settlement boundary");
    assert_eq!(settlement.calls, 2);
    assert_eq!(settlement.failures, 0);
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
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        300,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        None,
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
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        42,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        None,
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
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        600,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        None,
    );

    assert_eq!(state.request_timeout.as_secs(), 600);
}

#[tokio::test]
#[ignore = "constructs a full real-backend AppState (model prewarm) — too heavy for CI runners; run locally with --ignored"]
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
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        300,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        None,
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
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        300,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        None,
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
        kiln_server::batching_engine::ResponseDeliveryPolicy::default(),
        kiln_server::config::BatchTokenBudget::default(),
        300,
        "Qwen3.5-4B".to_string(),
        &kiln_server::config::PrefixCacheConfig::default(),
        None,
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
            kiln_train::trainer::TrainControl::Continue
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

/// Tiny HYBRID config with ONE GDN (linear-attention) layer + one full-attn
/// layer, for the GDN-on-Metal training smoke. `full_attention_interval = 2`
/// makes layer 0 linear/GDN and layer 1 full (`(idx+1) % 2`). The GDN dims are
/// the smallest that still exercise every GDN sub-op: 1 key/value head, head_dim
/// 4, conv kernel 4. The conv channels = `linear_qkv_dim = 2*qk + v = 12`.
#[cfg(feature = "metal")]
fn tiny_gdn_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_layers: 2,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 32,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::BF16,
        // Layer 0 -> linear (GDN), layer 1 -> full attention.
        num_full_attention_layers: 1,
        full_attention_interval: 2,
        attn_output_gate: false,
        linear_num_key_heads: 1,
        linear_key_head_dim: 4,
        linear_num_value_heads: 1,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    }
}

/// BF16 GpuWeights for [`tiny_gdn_config`]: layer 0 is a GDN
/// (`GpuAttentionWeights::Linear`) layer, layer 1 is full attention. Mirrors the
/// GDN weight literal in `kiln_model::forward` / `kiln_train::trainer` test
/// fixtures (all `*_t` transposes materialized, `a_log_gates`/`dt_bias`/`norm`
/// populated). The GDN layer's `in_proj_qkv` LoRA grad path runs through the
/// depthwise causal conv1d, so a non-empty grad-norm receipt proves the conv
/// backward (the composite on Metal) did not sever.
#[cfg(feature = "metal")]
fn tiny_gdn_weights_bf16(config: &ModelConfig, device: &Device) -> GpuWeights {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;

    let bf16 = |t: &Tensor| kiln_tensor::ops::cast(t, DType::BF16).expect("cast bf16");
    let rnd = |shape: &[usize]| bf16(&Tensor::randn(0.0_f32, 0.02, shape, device).unwrap());
    let rnd_t = |shape: &[usize]| {
        let w = Tensor::randn(0.0_f32, 0.02, shape, device).unwrap();
        let wt = w.t().unwrap().contiguous().unwrap();
        (bf16(&w), bf16(&wt))
    };

    let embed = Tensor::randn(0.0_f32, 0.02, (vocab, h), device).unwrap();
    let embed_t = embed.t().unwrap().contiguous().unwrap();
    let embed = bf16(&embed);
    let embed_t = bf16(&embed_t);
    let final_norm = bf16(&Tensor::zeros((h,), DType::F32, device).unwrap());

    let mk_mlp = || {
        let (gate_proj, gate_proj_t) = rnd_t(&[inter, h]);
        let (up_proj, up_proj_t) = rnd_t(&[inter, h]);
        let (down_proj, down_proj_t) = rnd_t(&[h, inter]);
        GpuFfnWeights {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_t,
            up_proj_t,
            down_proj_t,
            gate_up_proj_t: None,
            gate_up_proj_w8: None,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
            down_proj_w8: None,
        }
    };

    // --- Layer 0: GDN (linear attention). ---
    let qkv_dim = config.linear_qkv_dim();
    let v_dim = config.linear_v_dim();
    let nv = config.linear_num_value_heads;
    let (in_proj_qkv, in_proj_qkv_t) = rnd_t(&[qkv_dim, h]);
    let (in_proj_z, in_proj_z_t) = rnd_t(&[v_dim, h]);
    let (out_proj, out_proj_t) = rnd_t(&[h, v_dim]);
    let (in_proj_a, in_proj_a_t) = rnd_t(&[nv, h]);
    let (in_proj_b, in_proj_b_t) = rnd_t(&[nv, h]);
    let a_log = bf16(&Tensor::randn(0.0_f32, 0.5, (nv,), device).unwrap());
    let gdn_layer = GpuLayerWeights {
        input_layernorm: bf16(&Tensor::zeros((h,), DType::F32, device).unwrap()),
        post_attention_layernorm: bf16(&Tensor::zeros((h,), DType::F32, device).unwrap()),
        attention: GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
            in_proj_qkv,
            in_proj_z,
            out_proj,
            in_proj_a,
            in_proj_b,
            conv1d: rnd(&[qkv_dim, 1, config.linear_conv_kernel_dim]),
            norm: Tensor::ones((config.linear_value_head_dim,), DType::F32, device).unwrap(),
            a_log: a_log.clone(),
            a_log_gates: a_log,
            dt_bias: bf16(&Tensor::zeros((nv,), DType::F32, device).unwrap()),
            in_proj_qkv_t,
            in_proj_z_t,
            in_proj_a_t,
            in_proj_b_t,
            in_proj_ab_t: None,
            out_proj_t,
            out_proj_marlin: None,
            in_proj_qkvzab_w8: None,
        }),
        mlp: mk_mlp(),
    };

    // --- Layer 1: full attention. ---
    let (q_proj, q_proj_t) = rnd_t(&[num_heads * head_dim, h]);
    let (k_proj, k_proj_t) = rnd_t(&[num_kv_heads * head_dim, h]);
    let (v_proj, v_proj_t) = rnd_t(&[num_kv_heads * head_dim, h]);
    let (o_proj, o_proj_t) = rnd_t(&[h, num_heads * head_dim]);
    let full_layer = GpuLayerWeights {
        input_layernorm: bf16(&Tensor::zeros((h,), DType::F32, device).unwrap()),
        post_attention_layernorm: bf16(&Tensor::zeros((h,), DType::F32, device).unwrap()),
        attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: bf16(&Tensor::zeros((head_dim,), DType::F32, device).unwrap()),
            k_norm: bf16(&Tensor::zeros((head_dim,), DType::F32, device).unwrap()),
            q_proj_t,
            k_proj_t,
            v_proj_t,
            qkv_proj_t: None,
            qkv_proj_w8: None,
            o_proj_t,
            q_proj_marlin: None,
            o_proj_w8: None,
        }),
        mlp: mk_mlp(),
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
        layers: vec![gdn_layer, full_layer],
        final_norm,
        rotary_inv_freq,
        mtp: None,
        lm_head_w8: None,
    }
}

/// GDN-layer LoRA SFT on Metal smoke (#1082). The conv1d-backward gap this PR
/// closes was one of several Metal GDN-training gaps; with the conv1d-bwd-input
/// composite wired, the `in_proj_qkv` LoRA grad path no longer severs at the
/// depthwise causal conv1d. This builds a tiny hybrid model with ONE GDN layer
/// (`GpuAttentionWeights::Linear`) + one full-attention layer and runs a few SFT
/// steps on `Device::Metal(0)`, asserting LoRA grads flowed (a non-empty,
/// non-zero `lora_grad_norms` receipt).
///
/// # Enabled: the Metal GDN forward + backward now run end-to-end
///
/// The chunkwise GDN forward (`forward.rs` `gdn_chunkwise_recurrence`) reaches
/// `kiln_tensor::ops::{cumsum, compare, where_select}`, all of which now have
/// kiln-owned native MSL kernels (no host round-trip), so the GDN forward
/// completes on `Device::Metal(0)`; the conv1d backward closes the
/// `in_proj_qkv` LoRA grad path. This test runs by default as the end-to-end
/// GDN-training smoke and asserts LoRA grads flowed.
#[cfg(feature = "metal")]
#[test]
fn test_real_model_gdn_sft_metal() {
    if kiln_model::backend::metal::try_new_metal().is_none() {
        eprintln!("No Metal device — skipping GDN-SFT-on-Metal smoke");
        return;
    }
    let device = Device::Metal(0);
    let config = tiny_gdn_config();
    let weights = tiny_gdn_weights_bf16(&config, &device);
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
        learning_rate: Some(1e-3),
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
        "gdn-sft-metal-smoke",
        Some(cb),
        None,
        None,
    )
    .expect("GDN SFT training on Device::Metal(0) should complete");
    assert_adapter_written(&out);

    let losses = losses.lock().unwrap();
    assert!(
        !losses.is_empty(),
        "GDN SFT progress callback recorded no losses"
    );
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "all GDN SFT training losses must be finite, got {losses:?}"
    );

    // The decisive assertion: LoRA grads must have flowed. With the conv1d
    // backward severed (the pre-#1082 Metal behavior), the GDN layer's
    // in_proj_qkv path records no gradient and the receipt's grad-norm list is
    // empty / all-zero. A non-empty list with a strictly-positive max mean-norm
    // proves the conv1d-bwd-input composite carried the GDN gradient on Metal.
    let (grad_norm_modules, max_mean_norm) = receipt_lora_grad_norms(&out);
    assert!(
        grad_norm_modules > 0,
        "GDN SFT step recorded no LoRA grad norms — gradients did not flow \
         through the GDN layer on Metal (the conv1d backward severed). \
         losses={losses:?}"
    );
    assert!(
        max_mean_norm > 0.0,
        "GDN SFT LoRA grad norms are all zero — the tape walked but deposited \
         no signal (modules={grad_norm_modules}, losses={losses:?})"
    );
    eprintln!(
        "[GDN-SFT-METAL] losses={losses:?} lora_grad_norm_modules={grad_norm_modules} \
         max_mean_norm={max_mean_norm:.3e}"
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
        learning_rate: Some(1e-3),
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
        learning_rate: Some(1e-3),
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
    assert!(
        !losses.is_empty(),
        "GRPO progress callback recorded no losses"
    );
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
        learning_rate: Some(1e-3),
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
            assert!(
                !losses.is_empty(),
                "OPD progress callback recorded no losses"
            );
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

// ---------------------------------------------------------------------------
// Multi-turn prefix-cache reuse with a REAL ModelRunner (CPU).
//
// These are the regression tests for the 002af558 class of bug: the prefix
// cache "worked" (entries registered, exact replays hit) while multi-turn
// reuse — the entire point of the cache for agent traffic — was structurally
// impossible because no block-aligned strict-prefix entry ever registered.
// One test drives the batching-engine forward (CUDA/Vulkan/ROCm/CPU serve
// path), one drives the non-batched generate path (Metal's default serve
// path); both run on the CPU backend so they execute on every CI runner.
// ---------------------------------------------------------------------------

use std::sync::{Mutex, RwLock};

use kiln_core::block::BlockManager;
use kiln_core::sampling::{SamplingParams, ThinkingBudget};
use kiln_core::token::TokenId;
use kiln_model::{
    CancelHandle, FinishReason, PagedBatchedPrefillStart, PagedKvCacheKt, PagedPrefixReuse,
    StreamEvent,
};
use kiln_server::batching_engine::{
    DecodeForward, DecodeSlot, EngineRequest, RealDecodeForward, RequestPreparation,
};
use kiln_server::state::{LoadedAdapterIdentity, RealPrefixCache};
use uuid::Uuid;

const PREFIX_TEST_BLOCK_SIZE: usize = 16;
const PREFIX_TEST_NUM_BLOCKS: usize = 64;

/// Tiny HYBRID config (one GDN layer + one full-attention layer) in FP32 for
/// CPU prefix-cache tests. The GDN layer matters: prefix-cache resume must
/// restore the linear-attention state snapshot exactly, and a full-attn-only
/// model would never exercise that path.
fn tiny_gdn_config_f32() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_layers: 2,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        vocab_size: 32,
        max_position_embeddings: 256,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        dtype: kiln_core::config::DType::FP32,
        // Layer 0 -> linear (GDN), layer 1 -> full attention.
        num_full_attention_layers: 1,
        full_attention_interval: 2,
        attn_output_gate: false,
        linear_num_key_heads: 1,
        linear_key_head_dim: 4,
        linear_num_value_heads: 1,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        partial_rotary_factor: 1.0,
    }
}

/// FP32 GpuWeights for [`tiny_gdn_config_f32`]: layer 0 GDN, layer 1 full
/// attention. Same literal as `tiny_gdn_weights_bf16` minus the BF16 casts.
fn tiny_gdn_weights_f32(config: &ModelConfig, device: &Device) -> GpuWeights {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;

    let rnd = |shape: &[usize]| Tensor::randn(0.0_f32, 0.02, shape, device).unwrap();
    let rnd_t = |shape: &[usize]| {
        let w = Tensor::randn(0.0_f32, 0.02, shape, device).unwrap();
        let wt = w.t().unwrap().contiguous().unwrap();
        (w, wt)
    };

    let embed = Tensor::randn(0.0_f32, 0.02, (vocab, h), device).unwrap();
    let embed_t = embed.t().unwrap().contiguous().unwrap();
    let final_norm = Tensor::zeros((h,), DType::F32, device).unwrap();

    let mk_mlp = || {
        let (gate_proj, gate_proj_t) = rnd_t(&[inter, h]);
        let (up_proj, up_proj_t) = rnd_t(&[inter, h]);
        let (down_proj, down_proj_t) = rnd_t(&[h, inter]);
        GpuFfnWeights {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_t,
            up_proj_t,
            down_proj_t,
            gate_up_proj_t: None,
            gate_up_proj_w8: None,
            gate_proj_marlin: None,
            up_proj_marlin: None,
            down_proj_marlin: None,
            down_proj_w8: None,
        }
    };

    // --- Layer 0: GDN (linear attention). ---
    let qkv_dim = config.linear_qkv_dim();
    let v_dim = config.linear_v_dim();
    let nv = config.linear_num_value_heads;
    let (in_proj_qkv, in_proj_qkv_t) = rnd_t(&[qkv_dim, h]);
    let (in_proj_z, in_proj_z_t) = rnd_t(&[v_dim, h]);
    let (out_proj, out_proj_t) = rnd_t(&[h, v_dim]);
    let (in_proj_a, in_proj_a_t) = rnd_t(&[nv, h]);
    let (in_proj_b, in_proj_b_t) = rnd_t(&[nv, h]);
    let a_log = Tensor::randn(0.0_f32, 0.5, (nv,), device).unwrap();
    let gdn_layer = GpuLayerWeights {
        input_layernorm: Tensor::zeros((h,), DType::F32, device).unwrap(),
        post_attention_layernorm: Tensor::zeros((h,), DType::F32, device).unwrap(),
        attention: GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
            in_proj_qkv,
            in_proj_z,
            out_proj,
            in_proj_a,
            in_proj_b,
            conv1d: rnd(&[qkv_dim, 1, config.linear_conv_kernel_dim]),
            norm: Tensor::ones((config.linear_value_head_dim,), DType::F32, device).unwrap(),
            a_log: a_log.clone(),
            a_log_gates: a_log,
            dt_bias: Tensor::zeros((nv,), DType::F32, device).unwrap(),
            in_proj_qkv_t,
            in_proj_z_t,
            in_proj_a_t,
            in_proj_b_t,
            in_proj_ab_t: None,
            out_proj_t,
            out_proj_marlin: None,
            in_proj_qkvzab_w8: None,
        }),
        mlp: mk_mlp(),
    };

    // --- Layer 1: full attention. ---
    let (q_proj, q_proj_t) = rnd_t(&[num_heads * head_dim, h]);
    let (k_proj, k_proj_t) = rnd_t(&[num_kv_heads * head_dim, h]);
    let (v_proj, v_proj_t) = rnd_t(&[num_kv_heads * head_dim, h]);
    let (o_proj, o_proj_t) = rnd_t(&[h, num_heads * head_dim]);
    let full_layer = GpuLayerWeights {
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
            qkv_proj_w8: None,
            o_proj_t,
            q_proj_marlin: None,
            o_proj_w8: None,
        }),
        mlp: mk_mlp(),
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
        layers: vec![gdn_layer, full_layer],
        final_norm,
        rotary_inv_freq,
        mtp: None,
        lm_head_w8: None,
    }
}

fn prefix_test_paged_cache(config: &ModelConfig) -> PagedKvCacheKt {
    PagedKvCacheKt::new(
        config.num_full_attention_layers,
        PREFIX_TEST_NUM_BLOCKS,
        PREFIX_TEST_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        DType::F32,
        Device::Cpu,
    )
    .expect("paged KV cache")
}

#[test]
fn batching_forward_rejects_queued_request_after_same_name_revision_swap() {
    let config = tiny_config();
    let runner = ModelRunner::new(
        tiny_weights(&config, &Device::Cpu),
        test_tokenizer(),
        config.clone(),
    );
    let loaded_adapter = Arc::new(RwLock::new(Some(LoadedAdapterIdentity {
        name: "same-name".to_string(),
        content_revision: "new-revision".to_string(),
    })));
    let forward = RealDecodeForward::new(
        Arc::new(RwLock::new(runner)),
        Arc::new(Mutex::new(BlockManager::new(
            PREFIX_TEST_NUM_BLOCKS,
            PREFIX_TEST_BLOCK_SIZE,
        ))),
        Arc::new(prefix_test_paged_cache(&config)),
        Arc::new(Mutex::new(RealPrefixCache::disabled(
            PREFIX_TEST_BLOCK_SIZE,
        ))),
        Arc::new(tokio::sync::RwLock::new(())),
        loaded_adapter,
    );
    let request = EngineRequest {
        request_id: Uuid::new_v4(),
        prompt_tokens: vec![1, 2, 3],
        sampling: SamplingParams::greedy(),
        adapter: Some(LoadedAdapterIdentity {
            name: "same-name".to_string(),
            content_revision: "old-revision".to_string(),
        }),
        cancel: CancelHandle::new(),
    };

    let error = match forward.prepare_request(&request) {
        Ok(_) => panic!("stale queued request unexpectedly reached prefill"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("adapter revision is stale"),
        "{error:#}"
    );
}

#[test]
#[allow(deprecated)]
fn legacy_mutable_paged_stream_settles_before_return() {
    let config = tiny_config();
    let weights = tiny_weights(&config, &Device::Cpu);
    let runner = ModelRunner::new(weights, test_tokenizer(), config.clone());
    let mut block_manager = BlockManager::new(PREFIX_TEST_NUM_BLOCKS, PREFIX_TEST_BLOCK_SIZE);
    let paged_cache = prefix_test_paged_cache(&config);
    let sampling = SamplingParams {
        max_tokens: 1,
        thinking_budget: Some(
            ThinkingBudget::new(Some(0), None, 1, vec![1])
                .expect("deterministic one-token thinking budget"),
        ),
        ..SamplingParams::greedy()
    };

    let events = runner
        .generate_streaming_paged("<|im_start|>", &sampling, &mut block_manager, &paged_cache)
        .expect("legacy mutable paged stream")
        .into_iter()
        .collect::<Vec<_>>();

    assert!(matches!(
        events.as_slice(),
        [
            StreamEvent::Token(token),
            StreamEvent::Done(done)
        ] if token.token_id == 1 && done.completion_tokens == 1
    ));
    assert_eq!(
        block_manager.num_used(),
        0,
        "the synchronous compatibility API must settle before returning its populated receiver"
    );
}

/// 81 deterministic prompt tokens: long enough that the prefill-split entry
/// (floor((81-1)/16)*16 = 80 tokens) clears the production-style
/// min_register_tokens=64 gate, and non-block-aligned (81 % 16 != 0) like
/// every real chat-template rendering.
fn prefix_test_turn1_prompt() -> Vec<TokenId> {
    (0..81u32).map(|i| (i % 17) + 1).collect()
}

#[test]
fn resumable_paged_prefill_matches_monolithic_cpu() {
    let config = tiny_gdn_config_f32();
    let runner = ModelRunner::new(
        tiny_gdn_weights_f32(&config, &Device::Cpu),
        test_tokenizer(),
        config.clone(),
    );
    let prompt = prefix_test_turn1_prompt();
    let sampling = SamplingParams {
        max_tokens: 1,
        ..SamplingParams::greedy()
    };

    let control_blocks = Mutex::new(BlockManager::new(
        PREFIX_TEST_NUM_BLOCKS,
        PREFIX_TEST_BLOCK_SIZE,
    ));
    let control_cache = prefix_test_paged_cache(&config);
    let control = runner
        .prepare_paged_batched_decode_with_prefix_cache(
            &prompt,
            &sampling,
            &control_blocks,
            &control_cache,
            None,
            true,
            None,
        )
        .expect("monolithic control prefill");
    runner
        .synchronize_external_yield("monolithic CPU prefill test")
        .unwrap();

    let chunked_blocks = Mutex::new(BlockManager::new(
        PREFIX_TEST_NUM_BLOCKS,
        PREFIX_TEST_BLOCK_SIZE,
    ));
    let chunked_cache = prefix_test_paged_cache(&config);
    let start = runner
        .begin_paged_batched_decode_with_prefix_cache(
            &prompt,
            &sampling,
            &chunked_blocks,
            &chunked_cache,
            None,
            true,
            None,
        )
        .expect("begin resumable prefill");
    let PagedBatchedPrefillStart::Prefilling(prefill) = start else {
        panic!("an uncached prompt must require prefill")
    };
    assert_eq!(prefill.processed_tokens(), 0);
    assert_eq!(prefill.remaining_tokens(), prompt.len());

    let mut prefill = Some(prefill);
    let mut chunks = Vec::new();
    let chunked = loop {
        let progress = runner
            .advance_paged_batched_prefill(&mut prefill, &sampling, &chunked_cache, 17, None)
            .expect("advance resumable prefill");
        runner
            .synchronize_external_yield("resumable CPU prefill test quantum")
            .unwrap();
        assert!((1..=17).contains(&progress.tokens_processed));
        chunks.push(progress.tokens_processed);
        if let Some(state) = progress.decode_state {
            break state;
        }
        assert!(prefill.as_ref().unwrap().remaining_tokens() > 0);
    };

    assert!(chunks.len() > 1, "prompt should span multiple quanta");
    assert_eq!(chunks.iter().sum::<usize>(), prompt.len());
    assert_eq!(chunked.next_token, control.next_token);
    assert_eq!(chunked.seq_len, control.seq_len);
    assert_eq!(
        chunked
            .prefill_split_snapshot
            .as_ref()
            .map(|snapshot| snapshot.position),
        control
            .prefill_split_snapshot
            .as_ref()
            .map(|snapshot| snapshot.position)
    );
    assert_eq!(
        chunked
            .prefill_split_snapshot
            .as_ref()
            .map(|snapshot| snapshot.position),
        Some(80)
    );

    control_blocks
        .lock()
        .unwrap()
        .free_all(&control.allocated_blocks);
    chunked_blocks
        .lock()
        .unwrap()
        .free_all(&chunked.allocated_blocks);
    assert_eq!(control_blocks.lock().unwrap().num_used(), 0);
    assert_eq!(chunked_blocks.lock().unwrap().num_used(), 0);
}

#[test]
fn real_resumable_prefill_cancel_and_discard_release_cpu_ownership() {
    let config = tiny_gdn_config_f32();
    let block_manager = Arc::new(Mutex::new(BlockManager::new(
        PREFIX_TEST_NUM_BLOCKS,
        PREFIX_TEST_BLOCK_SIZE,
    )));
    let prefix_cache = Arc::new(Mutex::new(RealPrefixCache::new_with_min_register_tokens(
        true,
        PREFIX_TEST_BLOCK_SIZE,
        32,
        8,
        1024,
        64,
    )));
    let forward = RealDecodeForward::new(
        Arc::new(RwLock::new(ModelRunner::new(
            tiny_gdn_weights_f32(&config, &Device::Cpu),
            test_tokenizer(),
            config.clone(),
        ))),
        block_manager.clone(),
        Arc::new(prefix_test_paged_cache(&config)),
        prefix_cache.clone(),
        Arc::new(tokio::sync::RwLock::new(())),
        Arc::new(RwLock::new(None)),
    );
    let sampling = SamplingParams {
        max_tokens: 1,
        ..SamplingParams::greedy()
    };

    let request = EngineRequest {
        request_id: Uuid::new_v4(),
        prompt_tokens: prefix_test_turn1_prompt(),
        sampling: sampling.clone(),
        adapter: None,
        cancel: CancelHandle::new(),
    };
    let RequestPreparation::Prefilling {
        slot,
        tokens_processed,
    } = forward
        .prepare_request_chunked(&request, 17)
        .expect("begin actor-owned prefill")
    else {
        panic!("uncached prompt unexpectedly became decode-ready")
    };
    assert_eq!(tokens_processed, 0);
    assert!(block_manager.lock().unwrap().num_used() > 0);
    let RequestPreparation::Prefilling {
        slot,
        tokens_processed,
    } = forward
        .advance_prefill(slot, 17, &sampling, &request.cancel)
        .expect("advance actor-owned prefill")
    else {
        panic!("17 tokens unexpectedly completed an 81-token prefill")
    };
    assert_eq!(tokens_processed, 17);
    forward.discard_request(slot);
    assert_eq!(block_manager.lock().unwrap().num_used(), 0);
    let stats = prefix_cache.lock().unwrap().stats();
    assert_eq!(stats.active_leases, 0);
    assert_eq!(stats.pending_release_entries, 0);

    let cancelled = EngineRequest {
        request_id: Uuid::new_v4(),
        prompt_tokens: prefix_test_turn1_prompt(),
        sampling: sampling.clone(),
        adapter: None,
        cancel: CancelHandle::new(),
    };
    let RequestPreparation::Prefilling { slot, .. } = forward
        .prepare_request_chunked(&cancelled, 17)
        .expect("begin cancellable actor-owned prefill")
    else {
        panic!("uncached prompt unexpectedly became decode-ready")
    };
    assert!(block_manager.lock().unwrap().num_used() > 0);
    cancelled.cancel.cancel();
    let error = match forward.advance_prefill(slot, 17, &sampling, &cancelled.cancel) {
        Ok(_) => panic!("cancelled prefill unexpectedly advanced"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("cancelled"), "{error:#}");
    assert_eq!(block_manager.lock().unwrap().num_used(), 0);
    let stats = prefix_cache.lock().unwrap().stats();
    assert_eq!(stats.active_leases, 0);
    assert_eq!(stats.pending_release_entries, 0);
}

/// Batching-engine serve path (CUDA / Vulkan / ROCm / CPU default): a prompt
/// ending inside a KV block must retain only its block-aligned strict prefix.
/// Both an identical retry and a longer second turn must resume from that safe
/// entry instead of sharing and then mutating the prompt's final partial block.
#[test]
fn prefix_cache_multi_turn_hit_through_batching_engine_forward() {
    let config = tiny_config();
    let weights = tiny_weights(&config, &Device::Cpu);
    let runner = ModelRunner::new(weights, test_tokenizer(), config.clone());

    let prefix_cache = Arc::new(Mutex::new(RealPrefixCache::new_with_min_register_tokens(
        true,
        PREFIX_TEST_BLOCK_SIZE,
        32,   // max_blocks
        8,    // max_entries
        1024, // state_bytes_per_entry (accounting only)
        64,   // production REAL_PREFIX_CACHE_MIN_REGISTER_TOKENS
    )));
    let block_manager = Arc::new(Mutex::new(BlockManager::new(
        PREFIX_TEST_NUM_BLOCKS,
        PREFIX_TEST_BLOCK_SIZE,
    )));
    let forward = RealDecodeForward::new(
        Arc::new(RwLock::new(runner)),
        block_manager.clone(),
        Arc::new(prefix_test_paged_cache(&config)),
        prefix_cache.clone(),
        Arc::new(tokio::sync::RwLock::new(())),
        Arc::new(RwLock::new(None)),
    );

    let sampling = SamplingParams {
        max_tokens: 4,
        ..SamplingParams::greedy()
    };
    let turn1 = prefix_test_turn1_prompt();
    let req1 = EngineRequest {
        request_id: Uuid::new_v4(),
        prompt_tokens: turn1.clone(),
        sampling: sampling.clone(),
        adapter: None,
        cancel: CancelHandle::new(),
    };
    let slot1 = forward.prepare_request(&req1).expect("turn-1 prepare");
    let turn1_next_token = match &slot1 {
        DecodeSlot::Real { state, .. } => state.next_token,
        DecodeSlot::Mock { .. } => panic!("real forward returned a mock slot"),
        DecodeSlot::RealPrefill { .. } => panic!("unbounded prepare left prefill pending"),
    };
    forward
        .finish_request(slot1, FinishReason::MaxTokens)
        .expect("turn-1 finish");

    let stats = prefix_cache.lock().unwrap().stats();
    assert_eq!(stats.lookup_hits, 0);
    assert_eq!(stats.lookup_misses, 1, "turn-1 lookup must miss (cold)");
    assert_eq!(
        stats.cached_entries, 1,
        "the non-aligned full prompt must not become an exact entry"
    );
    assert_eq!(
        stats.cached_blocks, 5,
        "the aligned 80-token prefix owns five blocks"
    );
    assert_eq!(block_manager.lock().unwrap().num_used(), 5);

    // An identical retry must select the safe 80-token entry and prefill the
    // one-token suffix into a separately allocated block. It must not take an
    // exact hit on the six-block, partially filled turn-1 table.
    let retry = EngineRequest {
        request_id: Uuid::new_v4(),
        prompt_tokens: turn1.clone(),
        sampling: sampling.clone(),
        adapter: None,
        cancel: CancelHandle::new(),
    };
    let retry_slot = forward
        .prepare_request(&retry)
        .expect("same-prompt safe-prefix prepare");
    match &retry_slot {
        DecodeSlot::Real {
            state,
            prefix_request,
            ..
        } => {
            assert!(
                prefix_request.is_some(),
                "retry must pin the safe prefix entry"
            );
            assert_eq!(state.next_token, turn1_next_token);
            assert_eq!(state.block_table.blocks.len(), 6);
        }
        DecodeSlot::Mock { .. } => panic!("real forward returned a mock slot"),
        DecodeSlot::RealPrefill { .. } => panic!("unbounded prepare left prefill pending"),
    }
    assert_eq!(block_manager.lock().unwrap().num_used(), 6);
    forward
        .finish_request(retry_slot, FinishReason::MaxTokens)
        .expect("same-prompt safe-prefix finish");
    assert_eq!(
        block_manager.lock().unwrap().num_used(),
        5,
        "retry suffix block must be released while cached aligned blocks stay retained"
    );

    // Turn 2: the conversation grew (assistant reply + new user message).
    let mut turn2 = turn1.clone();
    turn2.extend((0..9u32).map(|i| (i % 7) + 2));
    let req2 = EngineRequest {
        request_id: Uuid::new_v4(),
        prompt_tokens: turn2,
        sampling,
        adapter: None,
        cancel: CancelHandle::new(),
    };
    let slot2 = forward.prepare_request(&req2).expect("turn-2 prepare");
    let stats = prefix_cache.lock().unwrap().stats();
    assert_eq!(
        stats.lookup_hits, 2,
        "turn-2 must hit the block-aligned strict-prefix entry"
    );
    assert_eq!(
        stats.hit_tokens, 160,
        "both hits must cover the full split position (floor((81-1)/16)*16)"
    );
    assert!(
        prefix_cache.lock().unwrap().clear().is_empty(),
        "clear must retire, not release, the prefix entry pinned by turn 2"
    );
    let retired = prefix_cache.lock().unwrap().stats();
    assert_eq!(retired.active_leases, 1);
    assert_eq!(retired.pending_release_entries, 1);
    assert!(block_manager.lock().unwrap().num_used() > 0);
    forward
        .finish_request(slot2, FinishReason::MaxTokens)
        .expect("turn-2 finish");
    assert_eq!(
        block_manager.lock().unwrap().num_used(),
        0,
        "final request completion must release the retired prefix and private suffix"
    );
}

/// Non-batched serve path (Metal's default): the non-streaming CPU prefill
/// must register a block-aligned strict-prefix entry (the branch 002af558-era
/// code only covered under streaming prefill), and resuming generation from
/// that entry — paged KV blocks + the GDN linear-attention state snapshot —
/// must produce tokens IDENTICAL to an uncached full prefill. Token
/// equivalence on a hybrid GDN model is the correctness proof that the
/// linear-state snapshot/restore is exact, not just that plumbing connects.
#[test]
fn prefix_cache_nonbatched_path_split_entry_replays_token_identical() {
    let config = tiny_gdn_config_f32();
    let weights = tiny_gdn_weights_f32(&config, &Device::Cpu);
    let runner = ModelRunner::new(weights, test_tokenizer(), config.clone());

    let sampling = SamplingParams {
        max_tokens: 4,
        ..SamplingParams::greedy()
    };
    let turn1 = prefix_test_turn1_prompt();

    let block_manager = Mutex::new(BlockManager::new(
        PREFIX_TEST_NUM_BLOCKS,
        PREFIX_TEST_BLOCK_SIZE,
    ));
    let paged_cache = prefix_test_paged_cache(&config);
    let out1 = runner
        .generate_paged_shared_tokens_with_prefix_cache(
            &turn1,
            &sampling,
            &block_manager,
            &paged_cache,
            None,
            None,
        )
        .expect("turn-1 generation");

    let split_reg = out1
        .extra_registrations
        .iter()
        .find(|reg| {
            reg.prompt_tokens.len() % PREFIX_TEST_BLOCK_SIZE == 0
                && turn1.starts_with(&reg.prompt_tokens)
        })
        .expect(
            "non-streaming CPU prefill must register a block-aligned strict-prefix \
             entry — its absence means the non-batched path lost the split snapshot",
        );
    assert_eq!(split_reg.prompt_tokens.len(), 80);

    // Turn 2 extends the transcript: turn-1 prompt + emitted tokens + new input.
    let mut turn2 = turn1.clone();
    turn2.extend(out1.output.token_ids.iter().copied());
    turn2.extend([3u32, 5, 7, 2, 4]);

    // Control: full prefill on a fresh KV pool, no cache involvement.
    let control_bm = Mutex::new(BlockManager::new(
        PREFIX_TEST_NUM_BLOCKS,
        PREFIX_TEST_BLOCK_SIZE,
    ));
    let control_pc = prefix_test_paged_cache(&config);
    let control = runner
        .generate_paged_shared_tokens_with_prefix_cache(
            &turn2,
            &sampling,
            &control_bm,
            &control_pc,
            None,
            None,
        )
        .expect("control generation");

    // Cached: resume from the split entry's KV blocks + linear-state snapshot
    // (turn-1's blocks are still resident in `paged_cache`; nothing freed them).
    let reuse = PagedPrefixReuse {
        cached_tokens: split_reg.prompt_tokens.len(),
        block_ids: split_reg.block_ids.clone(),
        linear_state: split_reg
            .linear_state
            .snapshot()
            .expect("snapshot registered linear state"),
        next_token: None,
    };
    let cached = runner
        .generate_paged_shared_tokens_with_prefix_cache(
            &turn2,
            &sampling,
            &block_manager,
            &paged_cache,
            Some(reuse),
            None,
        )
        .expect("cached generation");

    assert!(
        !control.output.token_ids.is_empty(),
        "control run must generate at least one token for the equivalence check"
    );
    assert_eq!(
        cached.output.token_ids, control.output.token_ids,
        "generation resumed from the prefix-cache entry (KV blocks + GDN linear \
         state) must be token-identical to an uncached full prefill"
    );
}
