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

use kiln_core::request::Request;
use kiln_core::sampling::SamplingParams;
use kiln_core::tokenizer::ChatMessage;
use kiln_model::StreamEvent;

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
    pub content: String,
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
) -> Result<Response, (StatusCode, String)> {
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
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        max_tokens: req.max_tokens.unwrap_or(2048),
        stop: req.stop.clone().unwrap_or_default(),
        seed: req.seed,
        ..Default::default()
    };

    if req.stream {
        match state.backend.as_ref() {
            ModelBackend::Real {
                runner,
                block_manager,
                paged_cache,
            } => {
                generate_real_streaming(
                    &state,
                    runner,
                    block_manager,
                    paged_cache,
                    &prompt_text,
                    &sampling,
                    &req,
                )
                .await
            }
            ModelBackend::Mock { .. } => Err((
                StatusCode::NOT_IMPLEMENTED,
                "streaming not supported with mock backend".to_string(),
            )),
        }
    } else {
        match state.backend.as_ref() {
            ModelBackend::Real {
                runner,
                block_manager,
                paged_cache,
            } => {
                generate_real(
                    &state,
                    runner,
                    block_manager,
                    paged_cache,
                    &prompt_text,
                    &sampling,
                    &req,
                )
                .await
                .map(|json| json.into_response())
            }
            ModelBackend::Mock { scheduler, engine } => {
                generate_mock(&state, scheduler, engine, &prompt_text, &sampling, &req)
                    .await
                    .map(|json| json.into_response())
            }
        }
    }
}

/// Generate using the real ModelRunner with paged KV cache.
async fn generate_real(
    state: &AppState,
    runner: &std::sync::Arc<kiln_model::ModelRunner>,
    block_manager: &std::sync::Arc<std::sync::Mutex<kiln_core::block::BlockManager>>,
    paged_cache: &std::sync::Arc<std::sync::Mutex<kiln_model::PagedKvCache>>,
    prompt_text: &str,
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    // Count prompt tokens for usage stats.
    let prompt_token_count = state
        .tokenizer
        .encode(prompt_text)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .len();

    // ModelRunner.generate_paged() is CPU-bound; run on a blocking thread to
    // avoid starving the tokio runtime.
    let runner = runner.clone();
    let bm = block_manager.clone();
    let pc = paged_cache.clone();
    let prompt = prompt_text.to_owned();
    let params = sampling.clone();

    let output = tokio::task::spawn_blocking(move || {
        let mut bm_guard = bm.lock().unwrap();
        let mut pc_guard = pc.lock().unwrap();
        runner.generate_paged(&prompt, &params, &mut bm_guard, &mut pc_guard)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let finish_reason = match output.finish_reason {
        kiln_model::FinishReason::Eos => "stop",
        kiln_model::FinishReason::MaxTokens => "length",
        kiln_model::FinishReason::StopSequence(_) => "stop",
    };

    let now = now_epoch();

    Ok(Json(ChatCompletionResponse {
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
    }))
}

/// Generate using the real ModelRunner with SSE streaming and paged KV cache.
async fn generate_real_streaming(
    _state: &AppState,
    runner: &std::sync::Arc<kiln_model::ModelRunner>,
    block_manager: &std::sync::Arc<std::sync::Mutex<kiln_core::block::BlockManager>>,
    paged_cache: &std::sync::Arc<std::sync::Mutex<kiln_model::PagedKvCache>>,
    prompt_text: &str,
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
) -> Result<Response, (StatusCode, String)> {
    let runner = runner.clone();
    let bm = block_manager.clone();
    let pc = paged_cache.clone();
    let prompt = prompt_text.to_owned();
    let params = sampling.clone();
    let model = req.model.clone();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_epoch();

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
                let mut bm_guard = bm.lock().unwrap();
                let mut pc_guard = pc.lock().unwrap();
                runner.generate_streaming_paged(&prompt, &params, &mut bm_guard, &mut pc_guard)
            })
            .await
            {
                Ok(Ok(rx)) => rx,
                _ => {
                    let _ = tx.send(Event::default().data("[DONE]")).await;
                    return;
                }
            };

            // Forward token events as SSE
            loop {
                match sync_rx.recv() {
                    Ok(StreamEvent::Token(token)) => {
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
                            return;
                        }
                    }
                    Ok(StreamEvent::Done(done)) => {
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
                    Err(_) => {
                        // Channel closed without Done — send [DONE] anyway
                        let _ = tx.send(Event::default().data("[DONE]")).await;
                        return;
                    }
                }
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
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let prompt_tokens = state
        .tokenizer
        .encode(prompt_text)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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

    Ok(Json(ChatCompletionResponse {
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
    }))
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
