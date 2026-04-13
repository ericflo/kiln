use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use kiln_core::request::Request;
use kiln_core::sampling::SamplingParams;

use crate::state::AppState;

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

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    // TODO: real tokenization. For now, rough estimate: 1 token per 4 chars.
    let prompt_text: String = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");
    let estimated_tokens = prompt_text.len() / 4;
    let prompt_tokens: Vec<u32> = (0..estimated_tokens as u32).collect();

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        max_tokens: req.max_tokens.unwrap_or(2048),
        stop: req.stop.unwrap_or_default(),
        seed: req.seed,
        ..Default::default()
    };

    let request = Request::new(prompt_tokens.clone(), sampling, req.adapter);
    let request_id = request.id;

    // Add to scheduler
    {
        let mut sched = state.scheduler.lock().await;
        sched.add_request(request);
    }

    // Run scheduler steps until this request completes.
    // In the real implementation this will be an async loop driven by the engine.
    // For now with MockEngine, we just step until done.
    let max_steps = 100;
    let mut output_tokens = Vec::new();

    for _ in 0..max_steps {
        let mut sched = state.scheduler.lock().await;
        let step_output = sched.step();

        if step_output.scheduled.is_empty() {
            break;
        }

        // Build batch input (simplified — real impl builds proper ragged batch)
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

        let engine_output = state
            .engine
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

    // Mock: generate some text from the output tokens
    let completion_text = format!("[mock response with {} tokens]", output_tokens.len());

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: now,
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: completion_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: output_tokens.len(),
            total_tokens: prompt_tokens.len() + output_tokens.len(),
        },
    }))
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}
