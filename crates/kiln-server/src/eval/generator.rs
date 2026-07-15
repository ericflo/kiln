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

use kiln_core::sampling::{SamplingParams, ThinkingBudget};
use kiln_core::thinking_budget::{
    EffectiveThinkingBudget, ThinkingBudgetDefaults, ThinkingBudgetOverrides, ThinkingBudgetScope,
};
use kiln_core::token::TokenId;
#[cfg(test)]
use kiln_eval::EvalBudgetOverride;
use kiln_eval::scorers::JudgeRunner;
use kiln_eval::{EvalChatMessage, EvalGenerationParams, EvalThinkingBudget};
use uuid::Uuid;

use crate::api::completions::{
    Message, encode_prompt_tokens, render_prompt_text, stop_sequence_conflicts_with_thinking_close,
};
use crate::batching_engine::{EngineEvent, EngineRequest};
use crate::state::{AppState, ModelBackend};

/// Result of a single generation call — what the executor needs to score
/// the example and record cost metadata.
#[derive(Debug, Clone)]
pub struct EvalCompletion {
    /// Eval-normalized text used for scoring. This may restore an opening
    /// `<think>` that was prefilled by the chat template.
    pub text: String,
    /// Exact model continuation before eval-only normalization.
    pub raw_text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub latency_ms: f64,
    /// Adapter that produced the completion (echo of the request — useful
    /// for compare-mode diffs).
    pub adapter: Option<String>,
    pub thinking_budget: EvalThinkingBudget,
}

/// Origin of the effective generation object selected by the executor.
/// Individual inherited budget dimensions still report `server_default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalGenerationSource {
    Suite,
    RunOverride,
    Example,
}

impl EvalGenerationSource {
    fn budget_scope(self) -> ThinkingBudgetScope {
        match self {
            Self::Suite => ThinkingBudgetScope::Suite,
            Self::RunOverride => ThinkingBudgetScope::RunOverride,
            Self::Example => ThinkingBudgetScope::Example,
        }
    }
}

fn resolved_thinking_budget(
    params: &EvalGenerationParams,
    default_tokens: Option<usize>,
    default_ms: Option<u64>,
    generation_source: EvalGenerationSource,
) -> EvalThinkingBudget {
    let effective = EffectiveThinkingBudget::resolve(
        ThinkingBudgetOverrides {
            tokens: params.thinking_budget_tokens,
            time_ms: params.thinking_budget_ms,
        },
        ThinkingBudgetDefaults {
            tokens: default_tokens,
            time_ms: default_ms,
        },
        generation_source.budget_scope(),
    );
    EvalThinkingBudget {
        configured: effective.configured(),
        applied: false,
        max_tokens: effective.max_tokens,
        max_time_ms: effective.max_time_ms,
        tokens_source: effective.tokens_source,
        time_source: effective.time_source,
        outcome: None,
    }
}

/// Pre-rendered example handed from `prepare` → `run`. Opaque-ish: the
/// executor just plumbs it through.
#[derive(Debug, Clone)]
pub struct PreparedPrompt {
    pub tokens: Vec<TokenId>,
    pub starts_in_reasoning: bool,
}

fn normalize_eval_completion(text: String, starts_in_reasoning: bool) -> String {
    if starts_in_reasoning && !text.trim_start().starts_with("<think>") {
        format!("<think>\n{text}")
    } else {
        text
    }
}

/// Generator trait. The executor calls `set_adapter` once per suite run,
/// `prepare` once per example, `run` once per completion.
pub trait EvalGenerator: Send + Sync {
    /// Resolve inherited limits and validate tokenizer/stop/max-token
    /// compatibility before the adapter is loaded or any example runs.
    fn preflight_thinking_budget(
        &self,
        params: &EvalGenerationParams,
        generation_source: EvalGenerationSource,
        starts_in_reasoning: bool,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EvalThinkingBudget, String>> + Send + '_>,
    > {
        let resolved = resolved_thinking_budget(params, None, None, generation_source);
        let _ = starts_in_reasoning;
        Box::pin(async move { Ok(resolved) })
    }

    /// Load `adapter` (or unload to the base model when `None`) for an
    /// eval run. Returns the *previous* active adapter name so the caller
    /// can restore after the suite run.
    ///
    /// CONTRACT: an eval must measure the adapter's CURRENT DISK CONTENT.
    /// Implementations must reload even when target == current name —
    /// the §8.7 gate evaluates an adapter that was just RETRAINED under
    /// its serving name, and a name-equality no-op would score (and then
    /// "promote") the stale in-memory weights.
    fn set_adapter(
        &self,
        adapter: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
    >;

    /// Restore the pre-suite adapter after the run. Unlike
    /// [`Self::set_adapter`], a name match may no-op: the serving
    /// adapter's content did not change while the eval held the runtime.
    fn restore_adapter(
        &self,
        adapter: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
    > {
        self.set_adapter(adapter)
    }

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
        thinking_budget: &EvalThinkingBudget,
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
    fn to_api_messages(messages: &[EvalChatMessage], system_prompt: Option<&str>) -> Vec<Message> {
        let mut out = Vec::with_capacity(messages.len() + 1);
        let has_system = messages
            .first()
            .map(|m| m.role.eq_ignore_ascii_case("system"))
            .unwrap_or(false);
        if !has_system && let Some(sp) = system_prompt {
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

fn eval_target_requires_content_reload(adapter: Option<&str>) -> bool {
    adapter.is_some_and(|name| !name.is_empty())
}

impl EvalGenerator for LiveEvalGenerator {
    fn preflight_thinking_budget(
        &self,
        params: &EvalGenerationParams,
        generation_source: EvalGenerationSource,
        starts_in_reasoning: bool,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EvalThinkingBudget, String>> + Send + '_>,
    > {
        let state = self.state.clone();
        let params = params.clone();
        Box::pin(async move {
            let resolved = resolved_thinking_budget(
                &params,
                state.default_thinking_budget_tokens,
                state.default_thinking_budget_ms,
                generation_source,
            );
            if !starts_in_reasoning || !resolved.configured || params.max_tokens == 0 {
                return Ok(resolved);
            }
            let sampling = build_sampling(&params);
            if sampling
                .stop
                .iter()
                .any(|stop| stop_sequence_conflicts_with_thinking_close(stop))
            {
                return Err(
                    "thinking budgets cannot be combined with a stop sequence that matches part of '</think>'"
                        .to_string(),
                );
            }
            let close_tokens = state
                .tokenizer
                .encode("</think>")
                .map_err(|error| format!("tokenize thinking close tag: {error}"))?;
            let decoded_close = state
                .tokenizer
                .decode(&close_tokens)
                .map_err(|error| format!("decode thinking close tag: {error}"))?;
            if decoded_close != "</think>" {
                return Err(
                    "the active tokenizer cannot reproduce \"</think>\" as a forced token sequence"
                        .to_string(),
                );
            }
            ThinkingBudget::new(
                resolved.max_tokens,
                resolved.max_time_ms.map(std::time::Duration::from_millis),
                params.max_tokens,
                close_tokens,
            )
            .map_err(|error| format!("configure thinking budget: {error}"))?;
            Ok(resolved)
        })
    }

    fn set_adapter(
        &self,
        adapter: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
    > {
        let state = self.state.clone();
        let content_changed = eval_target_requires_content_reload(adapter);
        let want = adapter.map(str::to_string).filter(|s| !s.is_empty());
        Box::pin(async move {
            state
                .ensure_inference_admission_allowed()
                .map_err(|error| format!("eval adapter selection rejected: {error:#}"))?;
            let previous = state.active_adapter_name.read().unwrap().clone();
            // Named eval targets always reload: the §8.7 gate can evaluate an
            // adapter just retrained under its serving name, and a name-only
            // no-op would score stale weights. The immutable base target has
            // no disk adapter content to reload; treating base-to-base as a
            // mutation would incorrectly reject ordinary evals under the
            // stable serving profile.
            //
            // Barrier swap (see `adapter_swap`): a streaming request that's
            // mid-generation when an eval starts finishes on its own
            // weights first. This also keeps the loaded-adapter identity
            // truthful — the old direct swap left it stale, so the next
            // chat request could no-op its adapter check and silently
            // serve on the eval's weights. Per-adapter cache keying means
            // the serving agent's prefix cache survives the round-trip.
            let target = match want.as_ref() {
                Some(name) => crate::adapter_swap::SwapTarget::Named(name.clone()),
                None => crate::adapter_swap::SwapTarget::Base,
            };
            crate::adapter_swap::swap_runtime_adapter(
                &state,
                crate::adapter_swap::SwapRequest {
                    target,
                    content_changed,
                    default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Replace(
                        want.clone(),
                    ),
                    reason: "eval_set_adapter",
                },
            )
            .await?;
            Ok(previous)
        })
    }

    fn restore_adapter(
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
                // Cheap restore: the serving adapter's content did not
                // change while the eval held the runtime.
                return Ok(previous);
            }
            let target = match want.as_ref() {
                Some(name) => crate::adapter_swap::SwapTarget::Named(name.clone()),
                None => crate::adapter_swap::SwapTarget::Base,
            };
            crate::adapter_swap::swap_runtime_adapter(
                &state,
                crate::adapter_swap::SwapRequest {
                    target,
                    content_changed: false,
                    default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Replace(
                        want.clone(),
                    ),
                    reason: "eval_restore_adapter",
                },
            )
            .await?;
            Ok(previous)
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
                Ok(PreparedPrompt {
                    tokens,
                    starts_in_reasoning: prompt.trim_end().ends_with("<think>"),
                })
            })
            .await
            .map_err(|e| format!("join error: {e}"))?
        })
    }

    fn run(
        &self,
        prepared: &PreparedPrompt,
        params: &EvalGenerationParams,
        thinking_budget: &EvalThinkingBudget,
        _completion_index: usize,
        adapter_label: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EvalCompletion, String>> + Send + '_>,
    > {
        let state = self.state.clone();
        let prompt_tokens = prepared.tokens.clone();
        let starts_in_reasoning = prepared.starts_in_reasoning;
        let params = params.clone();
        let mut thinking_budget_record = thinking_budget.clone();
        let adapter_label = adapter_label.map(str::to_string);
        Box::pin(async move {
            state
                .ensure_inference_admission_allowed()
                .map_err(|error| format!("eval generation rejected: {error:#}"))?;
            // The executor materializes a distinct seed for every stable
            // example/completion identity before calling the generator.
            let mut sampling = build_sampling(&params);
            if starts_in_reasoning && sampling.max_tokens > 0 && thinking_budget_record.configured {
                if sampling
                    .stop
                    .iter()
                    .any(|stop| stop_sequence_conflicts_with_thinking_close(stop))
                {
                    return Err(
                        "thinking budgets cannot be combined with a stop sequence that matches part of '</think>'"
                            .to_string(),
                    );
                }
                let close_tokens = state
                    .tokenizer
                    .encode("</think>")
                    .map_err(|error| format!("tokenize thinking close tag: {error}"))?;
                let decoded_close = state
                    .tokenizer
                    .decode(&close_tokens)
                    .map_err(|error| format!("decode thinking close tag: {error}"))?;
                if decoded_close != "</think>" {
                    return Err(
                        "the active tokenizer cannot reproduce \"</think>\" as a forced token sequence"
                            .to_string(),
                    );
                }
                if close_tokens.len() > sampling.max_tokens {
                    return Err(format!(
                        "max_tokens {} is too small for the tokenizer's {}-token </think> close sequence",
                        sampling.max_tokens,
                        close_tokens.len()
                    ));
                }
                sampling.thinking_budget = Some(
                    ThinkingBudget::new(
                        thinking_budget_record.max_tokens,
                        thinking_budget_record
                            .max_time_ms
                            .map(std::time::Duration::from_millis),
                        sampling.max_tokens,
                        close_tokens,
                    )
                    .map_err(|error| format!("configure thinking budget: {error}"))?,
                );
                thinking_budget_record.applied = true;
            }
            let thinking_budget_handle = sampling.thinking_budget.clone();
            // Swap-corruption detection (round-4 discovery item 8): a
            // concurrent chat request's ensure_runtime_adapter can
            // barrier-swap the global weights MID-SUITE — the engine
            // never checks label-vs-loaded, so without this guard the
            // suite scores (and the §8.7 gate then promotes/demotes on)
            // mixed-weight outputs. Detection over prevention: fail the
            // example loudly; the gate's incomplete-run verdict refuses
            // promotion and the operator sees exactly what happened.
            let pinned = state.active_adapter_name.read().unwrap().clone();
            let loaded = state.loaded_adapter_name();
            if pinned != loaded {
                return Err(format!(
                    "eval invalidated by a concurrent adapter swap: the suite pinned                      `{}` but the runtime now serves `{}` — re-run the eval (or gate)                      when live traffic isn't swapping adapters",
                    pinned.as_deref().unwrap_or("base"),
                    loaded.as_deref().unwrap_or("base"),
                ));
            }
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
            // synchronous enqueue scheduling step.
            let active_adapter = state.loaded_adapter_identity();
            let enqueue_fut = {
                state
                    .ensure_backend_healthy()
                    .map_err(|error| format!("{error:#}"))?;
                let _gpu_guard = state.gpu_lock.clone().read_owned().await;
                state
                    .ensure_backend_healthy()
                    .map_err(|error| format!("{error:#}"))?;
                batching_engine.enqueue(EngineRequest {
                    request_id,
                    prompt_tokens: prompt_tokens.clone(),
                    sampling,
                    adapter: active_adapter,
                    capture_behavior_logprobs: false,
                    cancel: cancel.clone(),
                })
            };
            let mut events = enqueue_fut.await.map_err(|e| format!("enqueue: {e}"))?;
            let timeout = state.request_timeout;
            let collect = async {
                loop {
                    match events.recv().await {
                        Some(EngineEvent::Token { .. }) => {}
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
            if let Some(mut status) = thinking_budget_handle.map(|budget| budget.status()) {
                if !status.closed {
                    status.thinking_tokens = output.completion_tokens;
                }
                thinking_budget_record.outcome = Some(status.into());
            }
            let raw_text = output.text;
            Ok(EvalCompletion {
                text: normalize_eval_completion(raw_text.clone(), starts_in_reasoning),
                raw_text,
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: output.completion_tokens,
                latency_ms: elapsed_ms,
                adapter: adapter_label,
                thinking_budget: thinking_budget_record,
            })
        })
    }
}

fn build_sampling(params: &EvalGenerationParams) -> SamplingParams {
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
        seed: params.seed,
        ..SamplingParams::greedy()
    }
}

/// Implementation of `JudgeRunner` that re-enters the live generator for a
/// judge-style synchronous prompt. Used by the `LlmJudge` scorer.
///
/// Contract: `judge()` MUST be called from a blocking thread (the
/// executor's deferred judge pass runs scoring inside `spawn_blocking`) —
/// `Handle::block_on` panics on a tokio worker thread, which is exactly
/// how the previous borrow-based version would have died had anything
/// constructed it (nothing did; the worker always installed the no-op
/// runner, so every judge-scored example silently degraded to Invalid).
///
/// The judge adapter is swapped in by the EXECUTOR, once per suite run —
/// not here. The old per-call `set_adapter` both thrashed the weights for
/// every judge call and never restored them, so the rest of the suite
/// would have run on the judge's weights.
pub struct LiveJudgeRunner {
    generator: Arc<LiveEvalGenerator>,
    runtime: tokio::runtime::Handle,
}

impl LiveJudgeRunner {
    /// Capture the current runtime handle; call from async context (the
    /// eval worker) at construction time.
    pub fn new(state: AppState) -> Self {
        Self {
            generator: Arc::new(LiveEvalGenerator::new(state)),
            runtime: tokio::runtime::Handle::current(),
        }
    }
}

impl JudgeRunner for LiveJudgeRunner {
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
            thinking_budget_tokens: kiln_eval::EvalBudgetOverride::Inherit,
            thinking_budget_ms: kiln_eval::EvalBudgetOverride::Inherit,
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
        // `adapter` is a label for the engine request only — the executor
        // already activated the judge adapter at the batch boundary.
        let budget = match self
            .runtime
            .block_on(self.generator.preflight_thinking_budget(
                &params,
                EvalGenerationSource::Suite,
                prepared.starts_in_reasoning,
            )) {
            Ok(budget) => budget,
            Err(e) => {
                tracing::warn!(error = %e, "LLM judge thinking-budget preflight failed");
                return None;
            }
        };
        let run_fut = self.generator.run(&prepared, &params, &budget, 0, adapter);
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
    /// Every set_adapter target, in order — tests assert swap batching
    /// (e.g. the judge adapter activates once per suite, not per call).
    pub swap_log: std::sync::Mutex<Vec<Option<String>>>,
}

impl MockEvalGenerator {
    pub fn new() -> Self {
        Self {
            force_reply: None,
            adapter_replies: std::collections::HashMap::new(),
            active: std::sync::Mutex::new(None),
            swap_log: std::sync::Mutex::new(Vec::new()),
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
            self.swap_log.lock().unwrap().push(want.clone());
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
            .flat_map(|m| {
                std::iter::once(m.role.len() as u32).chain(std::iter::once(m.content.len() as u32))
            })
            .collect();
        Box::pin(async move {
            Ok(PreparedPrompt {
                tokens: approx_tokens,
                starts_in_reasoning: false,
            })
        })
    }

    fn run(
        &self,
        prepared: &PreparedPrompt,
        _params: &EvalGenerationParams,
        thinking_budget: &EvalThinkingBudget,
        _completion_index: usize,
        adapter_label: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EvalCompletion, String>> + Send + '_>,
    > {
        let prompt_token_count = prepared.tokens.len();
        let adapter_key = adapter_label.unwrap_or("").to_string();
        let adapter_owned = adapter_label.map(str::to_string);
        let thinking_budget = thinking_budget.clone();
        let force = self.force_reply.clone();
        let mapped = self.adapter_replies.get(&adapter_key).cloned();
        Box::pin(async move {
            let text = force
                .or(mapped)
                .unwrap_or_else(|| format!("mock-reply [{adapter_key}]"));
            Ok(EvalCompletion {
                raw_text: text.clone(),
                text,
                prompt_tokens: prompt_token_count,
                completion_tokens: 1,
                latency_ms: 0.5,
                adapter: adapter_owned,
                thinking_budget,
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

    #[test]
    fn only_named_eval_targets_require_content_reload() {
        assert!(!eval_target_requires_content_reload(None));
        assert!(!eval_target_requires_content_reload(Some("")));
        assert!(eval_target_requires_content_reload(Some("adapter-a")));
    }

    #[test]
    fn resolved_budget_tracks_limit_provenance() {
        let params = EvalGenerationParams {
            thinking_budget_tokens: EvalBudgetOverride::Inherit,
            thinking_budget_ms: EvalBudgetOverride::Unlimited,
            ..EvalGenerationParams::default()
        };
        let budget =
            resolved_thinking_budget(&params, Some(17), Some(900), EvalGenerationSource::Example);
        assert!(budget.configured);
        assert_eq!(budget.max_tokens, Some(17));
        assert_eq!(budget.max_time_ms, None);
        assert_eq!(budget.tokens_source, "server_default");
        assert_eq!(budget.time_source, "example_unlimited");
        assert!(!budget.applied);
        assert!(budget.outcome.is_none());
    }

    #[tokio::test]
    async fn mock_generator_three_phase_round_trip() {
        let g = MockEvalGenerator::new();
        let messages = vec![EvalChatMessage::new("user", "hi")];
        let params = EvalGenerationParams::default();
        let prepared = g.prepare(&messages, None, None, &params).await.unwrap();
        let budget = g
            .preflight_thinking_budget(
                &params,
                EvalGenerationSource::Suite,
                prepared.starts_in_reasoning,
            )
            .await
            .unwrap();
        let prev = g.set_adapter(Some("foo")).await.unwrap();
        assert_eq!(prev, None);
        let r = g
            .run(&prepared, &params, &budget, 0, Some("foo"))
            .await
            .unwrap();
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
            starts_in_reasoning: false,
        };
        let params = EvalGenerationParams::default();
        let budget = g
            .preflight_thinking_budget(
                &params,
                EvalGenerationSource::Suite,
                prepared.starts_in_reasoning,
            )
            .await
            .unwrap();
        let r = g
            .run(&prepared, &params, &budget, 0, Some("alpha"))
            .await
            .unwrap();
        assert_eq!(r.text, "alpha-answer");
        let r2 = g
            .run(&prepared, &params, &budget, 0, Some("beta"))
            .await
            .unwrap();
        assert!(r2.text.contains("[beta]"));
    }

    #[tokio::test]
    async fn mock_generator_force_reply_overrides_adapter_map() {
        let g = MockEvalGenerator::new()
            .with_adapter_reply("alpha", "alpha-answer")
            .with_force_reply("override");
        let prepared = PreparedPrompt {
            tokens: vec![],
            starts_in_reasoning: false,
        };
        let params = EvalGenerationParams::default();
        let budget = g
            .preflight_thinking_budget(
                &params,
                EvalGenerationSource::Suite,
                prepared.starts_in_reasoning,
            )
            .await
            .unwrap();
        let r = g
            .run(&prepared, &params, &budget, 0, Some("alpha"))
            .await
            .unwrap();
        assert_eq!(r.text, "override");
    }

    #[test]
    fn reasoning_prefill_is_restored_for_eval_splitting_and_scoring() {
        let normalized = normalize_eval_completion(
            "work through it\n</think>\n\nfinal answer".to_string(),
            true,
        );
        let split = kiln_eval::qwen3::split_thinking(&normalized);
        assert_eq!(split.reasoning, Some("work through it"));
        assert_eq!(split.answer, "final answer");
        assert!(split.had_thinking);
        assert!(!split.unclosed);

        assert_eq!(
            normalize_eval_completion("plain answer".to_string(), false),
            "plain answer"
        );
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
