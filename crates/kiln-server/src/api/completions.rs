use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use std::path::Path;

use kiln_core::request::Request;
use kiln_core::sampling::SamplingParams;
use kiln_core::tokenizer::ChatMessage;
use kiln_model::lora_loader::LoraWeights;
use kiln_model::{ModelRunner, StreamEvent};

use crate::error::ApiError;
use crate::metrics::RequestStatus;
use crate::state::{AppState, ModelBackend};

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Kiln extension: which LoRA adapter to use for this request.
    #[serde(default)]
    pub adapter: Option<String>,
}

fn default_model() -> String {
    "qwen3.5-4b".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(deserialize_with = "deserialize_content")]
    pub content: String,
}

/// Accept `content` as either a plain string or an OpenAI-style array of
/// content parts (`[{"type": "text", "text": "..."}, ...]`). Text parts are
/// concatenated in order; non-text parts are ignored since kiln is text-only.
fn deserialize_content<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Content {
        Text(String),
        Parts(Vec<serde_json::Value>),
    }

    match Content::deserialize(deserializer)? {
        Content::Text(s) => Ok(s),
        Content::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                let Some(obj) = part.as_object() else { continue };
                let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if ty == "text" {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        out.push_str(text);
                    }
                }
            }
            Ok(out)
        }
    }
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// OpenAI-compatible streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let start = std::time::Instant::now();
    state.metrics.inc_active();

    let result = chat_completions_inner(&state, req).await;

    state.metrics.dec_active();
    let elapsed = start.elapsed().as_secs_f64();
    state.metrics.observe_duration(elapsed);

    match &result {
        Ok(_) => state.metrics.inc_request(RequestStatus::Ok),
        Err(e) => {
            if e.status == StatusCode::REQUEST_TIMEOUT {
                state.metrics.inc_request(RequestStatus::Timeout);
            } else {
                state.metrics.inc_request(RequestStatus::Error);
            }
        }
    }

    result
}

async fn chat_completions_inner(
    state: &AppState,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    // Convert request messages to ChatMessage for template formatting
    let chat_messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    // Apply chat template and tokenize
    let prompt_text = state
        .tokenizer
        .apply_chat_template(&chat_messages)
        .map_err(ApiError::chat_template_failed)?;

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        max_tokens: req.max_tokens.unwrap_or(2048),
        stop: req.stop.clone().unwrap_or_default(),
        seed: req.seed,
        ..Default::default()
    };

    // Ensure the correct LoRA adapter is active for this request.
    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        ensure_adapter(state, runner, &req.adapter).await?;
    }

    if req.stream {
        match state.backend.as_ref() {
            ModelBackend::Real {
                runner,
                block_manager,
                paged_cache,
            } => {
                generate_real_streaming(
                    state,
                    runner,
                    block_manager,
                    paged_cache,
                    &prompt_text,
                    &sampling,
                    &req,
                )
                .await
            }
            ModelBackend::Mock { .. } => Err(ApiError::streaming_not_supported_mock()),
        }
    } else {
        match state.backend.as_ref() {
            ModelBackend::Real {
                runner,
                block_manager,
                paged_cache,
            } => {
                let resp = generate_real(
                    state,
                    runner,
                    block_manager,
                    paged_cache,
                    &prompt_text,
                    &sampling,
                    &req,
                )
                .await?;
                // Count generated tokens for metrics.
                state
                    .metrics
                    .add_tokens(resp.usage.completion_tokens as u64);
                Ok(Json(resp).into_response())
            }
            ModelBackend::Mock { scheduler, engine } => {
                let resp =
                    generate_mock(state, scheduler, engine, &prompt_text, &sampling, &req).await?;
                state
                    .metrics
                    .add_tokens(resp.usage.completion_tokens as u64);
                Ok(Json(resp).into_response())
            }
        }
    }
}

/// Ensure the correct LoRA adapter is active for the given request.
///
/// Compares `req_adapter` against the currently active adapter name. If they
/// differ, loads/unloads the adapter using the two-phase RwLock pattern
/// (read config, load weights outside lock, write-lock to swap).
async fn ensure_adapter(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    req_adapter: &Option<String>,
) -> Result<(), ApiError> {
    let current = state.active_adapter_name.read().unwrap().clone();
    if *req_adapter == current {
        return Ok(());
    }

    match req_adapter {
        Some(name) => {
            // Resolve adapter path
            let adapter_path = if Path::new(name).is_absolute() {
                std::path::PathBuf::from(name)
            } else {
                state.adapter_dir.join(name)
            };
            if !adapter_path.exists() {
                return Err(ApiError::adapter_not_found(name));
            }

            // Two-phase load: read device/num_layers, load weights, then swap.
            let (device, num_layers) = {
                let guard = runner.read().unwrap();
                (guard.weights.embed_tokens.device().clone(), guard.config.num_layers)
            };

            let runner = runner.clone();
            let adapter_name = name.clone();
            let active_name = state.active_adapter_name.clone();

            tokio::task::spawn_blocking(move || {
                let lora = LoraWeights::load(&adapter_path, num_layers, &device)
                    .map_err(|e| format!("{e}"))?;
                let mut guard = runner.write().unwrap();
                guard.swap_lora(Some(lora));
                *active_name.write().unwrap() = Some(adapter_name);
                Ok::<(), String>(())
            })
            .await
            .map_err(|e| ApiError::internal(format!("join error: {e}")))?
            .map_err(ApiError::adapter_load_failed)?;
        }
        None => {
            // Revert to base model.
            let runner = runner.clone();
            let active_name = state.active_adapter_name.clone();

            tokio::task::spawn_blocking(move || {
                let mut guard = runner.write().unwrap();
                guard.swap_lora(None);
                *active_name.write().unwrap() = None;
            })
            .await
            .map_err(|e| ApiError::internal(format!("join error: {e}")))?;
        }
    }

    Ok(())
}

/// Generate using the real ModelRunner with paged KV cache.
async fn generate_real(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<kiln_model::ModelRunner>>,
    block_manager: &std::sync::Arc<std::sync::Mutex<kiln_core::block::BlockManager>>,
    paged_cache: &std::sync::Arc<std::sync::Mutex<kiln_model::PagedKvCache>>,
    prompt_text: &str,
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, ApiError> {
    // Count prompt tokens for usage stats.
    let prompt_token_count = state
        .tokenizer
        .encode(prompt_text)
        .map_err(ApiError::tokenization_failed)?
        .len();

    // ModelRunner.generate_paged() is CPU-bound; run on a blocking thread to
    // avoid starving the tokio runtime.
    let runner = runner.clone();
    let bm = block_manager.clone();
    let pc = paged_cache.clone();
    let prompt = prompt_text.to_owned();
    let params = sampling.clone();

    let gpu_lock = state.gpu_lock.clone();
    let timeout = state.request_timeout;
    let generation = tokio::task::spawn_blocking(move || {
        // Acquire GPU coordination read lock — allows concurrent inference,
        // but blocks while training holds the write lock.
        let _gpu_guard = gpu_lock.read().unwrap();
        let runner_guard = runner.read().unwrap();
        let mut bm_guard = bm.lock().unwrap();
        let mut pc_guard = pc.lock().unwrap();
        runner_guard.generate_paged(&prompt, &params, &mut bm_guard, &mut pc_guard)
    });

    let output = match tokio::time::timeout(timeout, generation).await {
        Ok(join_result) => join_result
            .map_err(|e| ApiError::internal(format!("join error: {e}")))?
            .map_err(ApiError::generation_failed)?,
        Err(_) => {
            return Err(ApiError::request_timeout(timeout.as_secs()));
        }
    };

    let finish_reason = match output.finish_reason {
        kiln_model::FinishReason::Eos => "stop",
        kiln_model::FinishReason::MaxTokens => "length",
        kiln_model::FinishReason::StopSequence(_) => "stop",
    };

    let now = now_epoch();

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: now,
        model: req.model.clone(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: output.text,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens: output.token_ids.len(),
            total_tokens: prompt_token_count + output.token_ids.len(),
        },
    })
}

/// Generate using the real ModelRunner with SSE streaming and paged KV cache.
///
/// Handles two cancellation paths:
/// 1. **Client disconnect**: When the SSE client drops the connection, the async
///    mpsc `tx.send()` fails. The forwarding task then drops `sync_rx`, which
///    causes `tx.send()` in the generation loop to fail, stopping generation.
/// 2. **Request timeout**: A `tokio::time::sleep` future races against the
///    forwarding loop. On timeout, the task drops `sync_rx`, stopping generation,
///    and sends an error event to the client.
async fn generate_real_streaming(
    _state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<kiln_model::ModelRunner>>,
    block_manager: &std::sync::Arc<std::sync::Mutex<kiln_core::block::BlockManager>>,
    paged_cache: &std::sync::Arc<std::sync::Mutex<kiln_model::PagedKvCache>>,
    prompt_text: &str,
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let runner = runner.clone();
    let bm = block_manager.clone();
    let pc = paged_cache.clone();
    let prompt = prompt_text.to_owned();
    let params = sampling.clone();
    let model = req.model.clone();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_epoch();
    let gpu_lock = _state.gpu_lock.clone();
    let timeout = _state.request_timeout;

    // Use a tokio mpsc channel to bridge sync generation -> async SSE stream.
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(32);

    // Spawn a task that runs the blocking generation and converts to SSE events.
    tokio::task::spawn({
        let id = completion_id.clone();
        let model = model.clone();
        async move {
            // Send initial role chunk
            let role_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            if tx
                .send(Event::default().data(serde_json::to_string(&role_chunk).unwrap()))
                .await
                .is_err()
            {
                return;
            }

            // Run blocking generation with paged KV cache
            let sync_rx = match tokio::task::spawn_blocking(move || {
                // Acquire GPU coordination read lock
                let _gpu_guard = gpu_lock.read().unwrap();
                let runner_guard = runner.read().unwrap();
                let mut bm_guard = bm.lock().unwrap();
                let mut pc_guard = pc.lock().unwrap();
                runner_guard.generate_streaming_paged(&prompt, &params, &mut bm_guard, &mut pc_guard)
            })
            .await
            {
                Ok(Ok(rx)) => rx,
                _ => {
                    let _ = tx.send(Event::default().data("[DONE]")).await;
                    return;
                }
            };

            // Forward token events as SSE, racing against a timeout.
            // When the timeout fires or client disconnects, we drop `sync_rx`,
            // which causes `tx.send()` in the generation loop to fail, stopping
            // generation and freeing KV cache blocks.
            //
            // We wrap `sync_rx.recv()` in `spawn_blocking` so it doesn't block
            // the async runtime. The receiver is moved into each spawn_blocking
            // call and returned along with the result so we can reuse it.
            let mut maybe_rx = Some(sync_rx);
            let mut timed_out = false;
            let deadline = tokio::time::Instant::now() + timeout;

            loop {
                let rx_inner = match maybe_rx.take() {
                    Some(r) => r,
                    None => break,
                };

                // Race: recv the next token vs. request timeout
                let recv_handle = tokio::task::spawn_blocking(move || {
                    let result = rx_inner.recv();
                    (rx_inner, result)
                });

                tokio::select! {
                    join_result = recv_handle => {
                        match join_result {
                            Ok((rx_back, Ok(StreamEvent::Token(token)))) => {
                                maybe_rx = Some(rx_back);
                                let chunk = ChatCompletionChunk {
                                    id: id.clone(),
                                    object: "chat.completion.chunk",
                                    created,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: Some(token.text),
                                        },
                                        finish_reason: None,
                                    }],
                                };
                                if tx
                                    .send(
                                        Event::default()
                                            .data(serde_json::to_string(&chunk).unwrap()),
                                    )
                                    .await
                                    .is_err()
                                {
                                    // Client disconnected — drop rx to stop generation
                                    return;
                                }
                            }
                            Ok((_, Ok(StreamEvent::Done(done)))) => {
                                let finish = match done.finish_reason {
                                    kiln_model::FinishReason::Eos => "stop",
                                    kiln_model::FinishReason::MaxTokens => "length",
                                    kiln_model::FinishReason::StopSequence(_) => "stop",
                                };
                                let chunk = ChatCompletionChunk {
                                    id: id.clone(),
                                    object: "chat.completion.chunk",
                                    created,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: None,
                                        },
                                        finish_reason: Some(finish.to_string()),
                                    }],
                                };
                                let _ = tx
                                    .send(
                                        Event::default()
                                            .data(serde_json::to_string(&chunk).unwrap()),
                                    )
                                    .await;
                                let _ = tx.send(Event::default().data("[DONE]")).await;
                                return;
                            }
                            _ => {
                                // Channel closed or join error
                                let _ = tx.send(Event::default().data("[DONE]")).await;
                                return;
                            }
                        }
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        timed_out = true;
                        // recv_handle is dropped, which will abort the
                        // spawn_blocking task. The sync_rx inside it will
                        // be dropped, causing the generation loop to stop.
                        break;
                    }
                }
            }

            if timed_out {
                let error_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("timeout".to_string()),
                    }],
                };
                let _ = tx
                    .send(
                        Event::default()
                            .data(serde_json::to_string(&error_chunk).unwrap()),
                    )
                    .await;
                let _ = tx.send(Event::default().data("[DONE]")).await;
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(Ok::<_, std::convert::Infallible>);

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Generate using the mock engine + scheduler loop (existing behavior).
async fn generate_mock(
    state: &AppState,
    scheduler: &tokio::sync::Mutex<kiln_scheduler::Scheduler>,
    engine: &std::sync::Arc<dyn kiln_model::engine::Engine>,
    prompt_text: &str,
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, ApiError> {
    let prompt_tokens = state
        .tokenizer
        .encode(prompt_text)
        .map_err(ApiError::tokenization_failed)?;

    let prompt_token_count = prompt_tokens.len();
    let request = Request::new(prompt_tokens, sampling.clone(), req.adapter.clone());
    let request_id = request.id;

    // Add to scheduler
    {
        let mut sched = scheduler.lock().await;
        sched.add_request(request);
    }

    // Run scheduler steps until this request completes.
    let max_steps = 100;
    let mut output_tokens = Vec::new();

    for _ in 0..max_steps {
        let mut sched = scheduler.lock().await;
        let step_output = sched.step();

        if step_output.scheduled.is_empty() {
            break;
        }

        // Build batch input
        let batch = kiln_model::engine::BatchInput {
            token_ids: vec![0; step_output.total_tokens],
            seqlens: step_output
                .scheduled
                .iter()
                .map(|s| s.num_tokens)
                .collect(),
            slot_mapping: vec![0; step_output.total_tokens],
            block_tables: step_output
                .scheduled
                .iter()
                .map(|_| vec![0])
                .collect(),
            is_prefill: step_output.scheduled.iter().map(|s| s.is_prefill).collect(),
            request_ids: step_output
                .scheduled
                .iter()
                .map(|s| s.request_id)
                .collect(),
        };

        let engine_output = engine
            .step(&batch)
            .map_err(ApiError::generation_failed)?;

        for (rid, token, finished) in &engine_output.results {
            if *rid == request_id {
                if let Some(t) = token {
                    output_tokens.push(*t);
                }
            }

            let prefill_processed = step_output
                .scheduled
                .iter()
                .find(|s| s.request_id == *rid && s.is_prefill)
                .map(|s| s.num_tokens);

            let finished = *finished || output_tokens.len() >= 20; // Mock: stop after 20 tokens
            sched.update_request(rid, *token, finished, prefill_processed);
        }

        // Check if our request is done
        if let Some(req) = sched.get_request(&request_id) {
            if matches!(
                req.state,
                kiln_core::request::RequestState::Complete
                    | kiln_core::request::RequestState::Cancelled
            ) {
                break;
            }
        } else {
            break;
        }
    }

    // Decode output tokens
    let completion_text = state
        .tokenizer
        .decode(&output_tokens)
        .unwrap_or_else(|_| format!("[{} tokens, decode failed]", output_tokens.len()));

    let now = now_epoch();

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: now,
        model: req.model.clone(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: completion_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens: output_tokens.len(),
            total_tokens: prompt_token_count + output_tokens.len(),
        },
    })
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_request(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("request should deserialize")
    }

    #[test]
    fn content_accepts_plain_string() {
        let req = parse_request(
            r#"{"messages":[{"role":"user","content":"hello"}]}"#,
        );
        assert_eq!(req.messages[0].content, "hello");
    }

    #[test]
    fn content_accepts_text_parts_array() {
        let req = parse_request(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}]}"#,
        );
        assert_eq!(req.messages[0].content, "hello world");
    }

    #[test]
    fn content_ignores_non_text_parts() {
        let req = parse_request(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"describe: "},{"type":"image_url","image_url":{"url":"https://example.com/a.png"}},{"type":"text","text":"done"}]}]}"#,
        );
        assert_eq!(req.messages[0].content, "describe: done");
    }

    #[test]
    fn content_empty_array_is_empty_string() {
        let req = parse_request(
            r#"{"messages":[{"role":"user","content":[]}]}"#,
        );
        assert_eq!(req.messages[0].content, "");
    }

    #[test]
    fn content_mixed_messages_in_same_request() {
        let req = parse_request(
            r#"{"messages":[{"role":"system","content":"be nice"},{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
        );
        assert_eq!(req.messages[0].content, "be nice");
        assert_eq!(req.messages[1].content, "hi");
    }
}
