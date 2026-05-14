//! Generator trait used by the eval executor.
//!
//! The trait splits one logical "evaluate this example N times" call into
//! three phases so each can be amortized correctly:
//!
//! 1. `set_adapter(adapter)` — once per (suite, adapter). Loads the LoRA
//!    (or unloads to the base model). Returns the *previous* active
//!    adapter so the executor can restore on suite-run completion.
//! 2. `prepare(messages, system_prompt, params)` — once per example.
//!    Renders the chat template + tokenizes. Cached via the same prompt /
//!    rendered-prompt caches the regular `/v1/chat/completions` path uses,
//!    so a suite run against an adapter that just answered the same prompt
//!    interactively pays no chat-template cost.
//! 3. `run(prepared, params, completion_index, adapter_label)` — once per
//!    completion (per `n`). Just enqueue + await.
//!
//! The pre-refactor shape (`generate(messages, ..., n_idx)`) collapsed all
//! three into the per-completion call, which paid the chat-template cost
//! `n` times per example and re-acquired the runner's RwLock on every
//! generate to no-op-check the adapter.

use std::sync::Arc;
use std::time::Instant;

use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_eval::scorers::JudgeRunner;
use kiln_eval::{EvalChatMessage, EvalGenerationParams};
use uuid::Uuid;

use crate::api::completions::{Message, encode_prompt_tokens, render_prompt_text};
use crate::batching_engine::{EngineEvent, EngineRequest};
use crate::state::{AppState, ModelBackend};

/// Result of a single generation call — what the executor needs to score
/// the example and record cost metadata.
#[derive(Debug, Clone)]
pub struct EvalCompletion {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub latency_ms: f64,
    /// Adapter that produced the completion (echo of the request — useful
    /// for compare-mode diffs).
    pub adapter: Option<String>,
}

/// Pre-rendered example handed from `prepare` → `run`. Opaque-ish: the
/// executor just plumbs it through.
#[derive(Debug, Clone)]
pub struct PreparedPrompt {
    pub tokens: Vec<TokenId>,
}

/// Generator trait. The executor calls `set_adapter` once per suite run,
/// `prepare` once per example, `run` once per completion.
pub trait EvalGenerator: Send + Sync {
    /// Load `adapter` (or unload to the base model when `None`). Returns
    /// the *previous* active adapter name so the caller can restore after
    /// the suite run. Idempotent: when target == current, this is a
    /// no-op return.
    fn set_adapter(
        &self,
        adapter: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
    >;

    /// Render chat template + tokenize. Called once per example.
    /// `tools` is the effective tool catalogue (per-example override or
    /// suite-level default). When `None` or empty, no `<tools>` block is
    /// rendered into the prompt.
    fn prepare(
        &self,
        messages: &[EvalChatMessage],
        system_prompt: Option<&str>,
        tools: Option<&[serde_json::Value]>,
        params: &EvalGenerationParams,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<PreparedPrompt, String>> + Send + '_>,
    >;

    /// Run a single completion off a prepared prompt. Called n times per
    /// example.
    fn run(
        &self,
        prepared: &PreparedPrompt,
        params: &EvalGenerationParams,
        completion_index: usize,
        adapter_label: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EvalCompletion, String>> + Send + '_>,
    >;
}

/// Live generator backed by the in-process model runner.
pub struct LiveEvalGenerator {
    state: AppState,
}

impl LiveEvalGenerator {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    /// Convert eval messages into the API `Message` shape that
    /// `render_prompt_text` consumes. Carries the optional agentic fields
    /// (`tool_calls`, `name`, `tool_call_id`) through so multi-turn tool
    /// trajectories render correctly via Qwen3.5's chat template.
    fn to_api_messages(
        messages: &[EvalChatMessage],
        system_prompt: Option<&str>,
    ) -> Vec<Message> {
        let mut out = Vec::with_capacity(messages.len() + 1);
        let has_system = messages
            .first()
            .map(|m| m.role.eq_ignore_ascii_case("system"))
            .unwrap_or(false);
        if !has_system
            && let Some(sp) = system_prompt
        {
            out.push(Message {
                role: "system".into(),
                content: sp.to_string(),
                reasoning_content: None,
                tool_calls: None,
                name: None,
                tool_call_id: None,
            });
        }
        for m in messages {
            out.push(Message {
                role: m.role.clone(),
                content: m.content.clone(),
                reasoning_content: None,
                tool_calls: m.tool_calls.clone(),
                name: m.name.clone(),
                tool_call_id: m.tool_call_id.clone(),
            });
        }
        out
    }
}

impl EvalGenerator for LiveEvalGenerator {
    fn set_adapter(
        &self,
        adapter: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
    > {
        let state = self.state.clone();
        let want = adapter.map(str::to_string).filter(|s| !s.is_empty());
        Box::pin(async move {
            let previous = state.active_adapter_name.read().unwrap().clone();
            if want == previous {
                return Ok(previous);
            }
            // RwLock acquires + LoraWeights::load are blocking — punt to
            // a dedicated thread so we don't stall the runtime.
            tokio::task::spawn_blocking(move || -> Result<Option<String>, String> {
                let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
                    return Err("eval requires a real model backend".into());
                };
                let (device, num_layers) = {
                    let guard = runner.read().unwrap();
                    (
                        guard.weights.embed_tokens.device().clone(),
                        guard.config.num_layers,
                    )
                };
                match want.as_ref() {
                    Some(name) => {
                        let path = state.adapter_dir.join(name);
                        if !path.exists() {
                            return Err(format!(
                                "adapter `{name}` not found at {}",
                                path.display()
                            ));
                        }
                        let lora =
                            kiln_model::lora_loader::LoraWeights::load(&path, num_layers, &device)
                                .map_err(|e| format!("loading adapter `{name}`: {e}"))?;
                        runner.write().unwrap().swap_lora(Some(lora));
                    }
                    None => {
                        runner.write().unwrap().unload_adapter();
                    }
                }
                *state.active_adapter_name.write().unwrap() = want;
                state.clear_real_prefix_cache();
                Ok(previous)
            })
            .await
            .map_err(|e| format!("join error: {e}"))?
        })
    }

    fn prepare(
        &self,
        messages: &[EvalChatMessage],
        system_prompt: Option<&str>,
        tools: Option<&[serde_json::Value]>,
        params: &EvalGenerationParams,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<PreparedPrompt, String>> + Send + '_>,
    > {
        let state = self.state.clone();
        let api_messages = Self::to_api_messages(messages, system_prompt);
        let kwargs = params.chat_template_kwargs.clone();
        let tools_owned: Option<Vec<serde_json::Value>> = tools.map(|t| t.to_vec());
        Box::pin(async move {
            // render + tokenize hit `state.rendered_prompt_cache` and
            // `state.prompt_token_cache` respectively, so re-running the
            // same suite gets a free pass on prompt setup. Both helpers
            // also acquire short locks; spawn_blocking keeps us off the
            // runtime.
            tokio::task::spawn_blocking(move || -> Result<PreparedPrompt, String> {
                let prompt = render_prompt_text(
                    &state,
                    &api_messages,
                    tools_owned.as_deref(),
                    None,
                    kwargs.as_ref(),
                )
                .map_err(|e| format!("chat template render: {e:?}"))?;
                let tokens = encode_prompt_tokens(&state, &prompt)
                    .map_err(|e| format!("tokenize: {e:?}"))?;
                Ok(PreparedPrompt { tokens })
            })
            .await
            .map_err(|e| format!("join error: {e}"))?
        })
    }

    fn run(
        &self,
        prepared: &PreparedPrompt,
        params: &EvalGenerationParams,
        completion_index: usize,
        adapter_label: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EvalCompletion, String>> + Send + '_>,
    > {
        let state = self.state.clone();
        let prompt_tokens = prepared.tokens.clone();
        let sampling = build_sampling(params, completion_index);
        let adapter_label = adapter_label.map(str::to_string);
        Box::pin(async move {
            let ModelBackend::Real {
                batching_engine, ..
            } = state.backend.as_ref()
            else {
                return Err("eval requires the real model backend".into());
            };
            let batching_engine = match batching_engine.as_ref() {
                Some(h) => h.clone(),
                None => return Err("batching engine not initialized".into()),
            };
            let request_id = Uuid::new_v4();
            let cancel = kiln_model::CancelHandle::with_prefill_progress_gauge(
                state.metrics.request_prefill_tokens_completed.clone(),
            );
            let started = Instant::now();
            // Inference-side gpu_lock — read lock so concurrent eval calls
            // can fan out across batching slots. Held only across the
            // synchronous enqueue scheduling step (the `RwLockReadGuard`
            // is `!Send` and can't cross an await).
            let active_adapter = state.active_adapter_name.read().unwrap().clone();
            let enqueue_fut = {
                let _gpu_guard = state.gpu_lock.read().unwrap();
                batching_engine.enqueue(EngineRequest {
                    request_id,
                    prompt_tokens: prompt_tokens.clone(),
                    sampling,
                    adapter: active_adapter,
                    cancel: cancel.clone(),
                })
            };
            let mut events = enqueue_fut.await.map_err(|e| format!("enqueue: {e}"))?;
            let timeout = state.request_timeout;
            let collect = async {
                loop {
                    match events.recv().await {
                        Some(EngineEvent::Token(_)) => {}
                        Some(EngineEvent::Done { output }) => break Ok(output),
                        Some(EngineEvent::Error(err)) => break Err(err),
                        None => break Err("batching engine response channel closed".to_string()),
                    }
                }
            };
            let output = match tokio::time::timeout(timeout, collect).await {
                Ok(Ok(o)) => o,
                Ok(Err(e)) => {
                    cancel.clear_prefill_progress();
                    return Err(format!("generation: {e}"));
                }
                Err(_) => {
                    cancel.cancel();
                    let _ = batching_engine.cancel(request_id).await;
                    cancel.clear_prefill_progress();
                    return Err(format!("timeout after {}s", timeout.as_secs()));
                }
            };
            cancel.clear_prefill_progress();
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            Ok(EvalCompletion {
                text: output.text,
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: output.completion_tokens,
                latency_ms: elapsed_ms,
                adapter: adapter_label,
            })
        })
    }
}

fn build_sampling(params: &EvalGenerationParams, completion_index: usize) -> SamplingParams {
    let seed = params.seed.map(|s| s.wrapping_add(completion_index as u64));
    // When the suite/example author didn't set explicit stop strings,
    // default to Qwen3.5's assistant turn terminators. Without this an
    // eval can run past `<|im_end|>` into a second `assistant\n<think>`
    // pseudo-turn and confuse downstream scoring. The base server EOS
    // already terminates on the EOS token, but the extra stop string is
    // a belt-and-suspenders guard against templates that occasionally
    // emit the literal `<|im_end|>` text.
    let stop = if params.stop.is_empty() {
        vec!["<|im_end|>".to_string()]
    } else {
        params.stop.clone()
    };
    // Evals want determinism — start from greedy and only override the
    // fields the eval suite actually set. The Qwen3.5 sampling
    // defaults are tuned for interactive generation, not for verifiable-
    // reward scoring.
    SamplingParams {
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        max_tokens: params.max_tokens,
        stop,
        seed,
        ..SamplingParams::greedy()
    }
}

/// Implementation of `JudgeRunner` that re-enters the live generator for a
/// judge-style synchronous prompt. Used by the `LlmJudge` scorer.
pub struct LiveJudgeRunner<'a> {
    pub generator: &'a LiveEvalGenerator,
    pub runtime: &'a tokio::runtime::Handle,
}

impl<'a> JudgeRunner for LiveJudgeRunner<'a> {
    fn judge(&self, adapter: Option<&str>, prompt: &str) -> Option<String> {
        let messages = vec![EvalChatMessage::new("user", prompt)];
        let params = EvalGenerationParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 64,
            n: 1,
            stop: vec![],
            seed: Some(0xC0FFEE),
            chat_template_kwargs: None,
        };
        let prep_fut = self.generator.prepare(&messages, None, None, &params);
        let prepared = match self.runtime.block_on(prep_fut) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "LLM judge prepare failed");
                return None;
            }
        };
        let _ = self.runtime.block_on(self.generator.set_adapter(adapter));
        let run_fut = self.generator.run(&prepared, &params, 0, adapter);
        match self.runtime.block_on(run_fut) {
            Ok(c) => Some(c.text),
            Err(e) => {
                tracing::warn!(error = %e, "LLM judge run failed");
                None
            }
        }
    }
}

/// Mock generator. Replies with a deterministic, suite-aware string so
/// scorers can be exercised end-to-end without a model.
pub struct MockEvalGenerator {
    pub force_reply: Option<String>,
    pub adapter_replies: std::collections::HashMap<String, String>,
    /// Tracks the "currently loaded" adapter for set_adapter symmetry —
    /// the mock backend never holds real state, but tests want the
    /// previous-adapter return to be plausible.
    active: std::sync::Mutex<Option<String>>,
}

impl MockEvalGenerator {
    pub fn new() -> Self {
        Self {
            force_reply: None,
            adapter_replies: std::collections::HashMap::new(),
            active: std::sync::Mutex::new(None),
        }
    }
    pub fn with_force_reply(mut self, reply: impl Into<String>) -> Self {
        self.force_reply = Some(reply.into());
        self
    }
    pub fn with_adapter_reply(mut self, adapter: &str, reply: impl Into<String>) -> Self {
        self.adapter_replies
            .insert(adapter.to_string(), reply.into());
        self
    }
}

impl Default for MockEvalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalGenerator for MockEvalGenerator {
    fn set_adapter(
        &self,
        adapter: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
    > {
        let want = adapter.map(str::to_string).filter(|s| !s.is_empty());
        Box::pin(async move {
            let mut slot = self.active.lock().unwrap();
            let previous = slot.clone();
            *slot = want;
            Ok(previous)
        })
    }

    fn prepare(
        &self,
        messages: &[EvalChatMessage],
        _system_prompt: Option<&str>,
        _tools: Option<&[serde_json::Value]>,
        _params: &EvalGenerationParams,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<PreparedPrompt, String>> + Send + '_>,
    > {
        // Encode role+content as a synthetic token stream. The mock
        // doesn't actually decode these, but the executor stamps the count
        // into the metrics.
        let approx_tokens: Vec<TokenId> = messages
            .iter()
            .flat_map(|m| std::iter::once(m.role.len() as u32).chain(std::iter::once(m.content.len() as u32)))
            .collect();
        Box::pin(async move {
            Ok(PreparedPrompt {
                tokens: approx_tokens,
            })
        })
    }

    fn run(
        &self,
        prepared: &PreparedPrompt,
        _params: &EvalGenerationParams,
        _completion_index: usize,
        adapter_label: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EvalCompletion, String>> + Send + '_>,
    > {
        let prompt_token_count = prepared.tokens.len();
        let adapter_key = adapter_label.unwrap_or("").to_string();
        let adapter_owned = adapter_label.map(str::to_string);
        let force = self.force_reply.clone();
        let mapped = self.adapter_replies.get(&adapter_key).cloned();
        Box::pin(async move {
            let text = force
                .or(mapped)
                .unwrap_or_else(|| format!("mock-reply [{adapter_key}]"));
            Ok(EvalCompletion {
                text,
                prompt_tokens: prompt_token_count,
                completion_tokens: 1,
                latency_ms: 0.5,
                adapter: adapter_owned,
            })
        })
    }
}

/// Factory: returns the right generator for the current backend.
pub fn generator_from_state(state: AppState) -> Arc<dyn EvalGenerator> {
    match state.backend.as_ref() {
        ModelBackend::Real { .. } => Arc::new(LiveEvalGenerator::new(state)),
        ModelBackend::Mock { .. } => Arc::new(MockEvalGenerator::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_generator_three_phase_round_trip() {
        let g = MockEvalGenerator::new();
        let messages = vec![EvalChatMessage::new("user", "hi")];
        let params = EvalGenerationParams::default();
        let prepared = g.prepare(&messages, None, None, &params).await.unwrap();
        let prev = g.set_adapter(Some("foo")).await.unwrap();
        assert_eq!(prev, None);
        let r = g.run(&prepared, &params, 0, Some("foo")).await.unwrap();
        assert!(r.text.contains("[foo]"));
        // Restoring None returns "foo" as the previous.
        let prev = g.set_adapter(None).await.unwrap();
        assert_eq!(prev.as_deref(), Some("foo"));
    }

    #[tokio::test]
    async fn mock_generator_adapter_specific_reply() {
        let g = MockEvalGenerator::new().with_adapter_reply("alpha", "alpha-answer");
        let prepared = PreparedPrompt {
            tokens: vec![1, 2, 3],
        };
        let params = EvalGenerationParams::default();
        let r = g.run(&prepared, &params, 0, Some("alpha")).await.unwrap();
        assert_eq!(r.text, "alpha-answer");
        let r2 = g.run(&prepared, &params, 0, Some("beta")).await.unwrap();
        assert!(r2.text.contains("[beta]"));
    }

    #[tokio::test]
    async fn mock_generator_force_reply_overrides_adapter_map() {
        let g = MockEvalGenerator::new()
            .with_adapter_reply("alpha", "alpha-answer")
            .with_force_reply("override");
        let prepared = PreparedPrompt { tokens: vec![] };
        let params = EvalGenerationParams::default();
        let r = g.run(&prepared, &params, 0, Some("alpha")).await.unwrap();
        assert_eq!(r.text, "override");
    }

    #[tokio::test]
    async fn set_adapter_returns_previous_for_restore() {
        let g = MockEvalGenerator::new();
        let p1 = g.set_adapter(Some("v1")).await.unwrap();
        assert_eq!(p1, None);
        let p2 = g.set_adapter(Some("v2")).await.unwrap();
        assert_eq!(p2.as_deref(), Some("v1"));
    }
}
