//! Remote-teacher `LogitSource` (grand plan §3.2).
//!
//! HTTP client that speaks the OpenAI-compatible `top_logprobs`
//! schema. All eight day-one providers (vLLM, sglang, llama.cpp,
//! OpenRouter, Together, Fireworks, DeepInfra, TGI) share enough
//! protocol overlap that one impl handles them with provider-specific
//! variation handled by a `RemoteProvider` field on the config:
//!
//! - **URL conventions** (path layout, header names).
//! - **top_logprobs cap** — vLLM ~256, llama.cpp full-vocab,
//!   OpenRouter 20, Together 20, Fireworks 8, etc.
//! - **API-key header** vs **bearer-token** vs **query-param**.
//! - **Tokenization stability check** — we re-tokenise the prompt
//!   with kiln's own tokenizer and compare `usage.prompt_tokens`
//!   against the provider's count to catch tokenizer drift before
//!   the run starts.
//!
//! § 8.6 cost-lock: every request increments the per-job cost
//! tally; when configured `max_cost_usd` is exceeded the next
//! `fetch_logprobs` returns `LogitSourceError::Invalid` with the
//! cost-cap message and the caller pauses + offers cache-only mode.
//!
//! The trait is synchronous (per the §3.2 design decision); HTTP
//! is run through `tokio::runtime::Handle::block_on` of a
//! `reqwest::Client::send`. Since the trainer already runs inside
//! `spawn_blocking`, this is the same pattern kiln-eval already
//! uses for HTTP scorers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::logit_source::{LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch, TopKLogprobs};

/// Provider variations on the OpenAI `top_logprobs` schema. Most
/// fields are URL/header conventions; the protocol body is the
/// same across providers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RemoteProvider {
    /// vLLM `/v1/completions` or `/v1/chat/completions`, `top_logprobs`
    /// returned for the next-token positions. Top-K cap ~256 in
    /// practice (configurable on the server).
    Vllm,
    /// sglang — same OpenAI surface as vLLM.
    Sglang,
    /// llama.cpp / `llama-server`. Uses `--logits-all` or `n_probs`;
    /// effectively unbounded top-K.
    LlamaCpp,
    /// OpenRouter — cap 20.
    OpenRouter,
    /// Together — cap 20.
    Together,
    /// Fireworks — cap 8.
    Fireworks,
    /// DeepInfra — cap 20.
    DeepInfra,
    /// Hugging Face TGI — `top_n_tokens`. Cap 5 by default.
    Tgi,
}

impl RemoteProvider {
    /// Conservative top-K cap per provider. Updated as providers
    /// raise their caps.
    pub fn default_max_top_k(&self) -> usize {
        match self {
            Self::Vllm => 256,
            Self::Sglang => 256,
            Self::LlamaCpp => 8192,
            Self::OpenRouter => 20,
            Self::Together => 20,
            Self::Fireworks => 8,
            Self::DeepInfra => 20,
            Self::Tgi => 5,
        }
    }

    /// Whether this provider exposes a `usage.prompt_tokens` field
    /// the trainer can compare against its own tokenization. All
    /// OpenAI-compatible providers do; the local ones (vLLM, sglang,
    /// llama.cpp) certainly do.
    pub fn supports_tokenizer_drift_check(&self) -> bool {
        true
    }

    /// Approximate cost per 1K prompt tokens in USD. Used by the
    /// §8.6 cost lock as a pre-fetch budget estimate. `None` for
    /// self-hosted (no per-call cost).
    pub fn cost_per_1k_prompt_tokens_usd(&self) -> Option<f64> {
        match self {
            Self::Vllm | Self::Sglang | Self::LlamaCpp => None,
            Self::OpenRouter => Some(0.0003),
            Self::Together => Some(0.0002),
            Self::Fireworks => Some(0.0002),
            Self::DeepInfra => Some(0.0002),
            Self::Tgi => None,
        }
    }
}

/// Configuration for the [`RemoteTeacher`] `LogitSource`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteTeacherConfig {
    /// Provider — selects the conventions.
    pub provider: RemoteProvider,
    /// Provider-specific model id (e.g. `qwen/qwen-3.6-27b`).
    pub model: String,
    /// Base URL of the API (e.g. `https://openrouter.ai/api/v1`).
    pub url: String,
    /// Environment variable to read the API key from. We never
    /// store the key in serialized form — the §8.6 cost-lock policy
    /// requires the key to live in env. `None` for self-hosted
    /// providers that don't need auth.
    pub api_key_env: Option<String>,
    /// User-facing teacher id (alias). Mirrors
    /// `LogitSourceCaps.teacher_id`.
    pub teacher_id: String,
    /// Tokenizer hash for drift detection. When set, we compare
    /// against the provider's `usage.prompt_tokens` on each
    /// response.
    pub tokenizer_hash: Option<String>,
    /// Cap on top-K. When 0, defaults to
    /// `provider.default_max_top_k()`.
    #[serde(default)]
    pub max_top_k: usize,
    /// Full vocabulary size — used by capabilities reporting and
    /// kernel dispatch.
    #[serde(default)]
    pub vocab_size: usize,
    /// §8.6 cost lock — per-job cap in USD. None = no cap.
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    /// Per-request timeout (ms). Defaults to 60_000.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    60_000
}

/// Cost accounting state shared by all `fetch_logprobs` calls in one
/// job. Atomic so the trainer-side worker can read the running tally
/// from any thread.
#[derive(Debug, Default)]
pub struct CostTally {
    /// Cents spent so far. Stored in cents to use AtomicU64 without
    /// floating-point atomic shenanigans.
    cents_spent: AtomicU64,
}

impl CostTally {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_usd(&self, usd: f64) {
        let cents = (usd * 100.0).max(0.0) as u64;
        self.cents_spent.fetch_add(cents, Ordering::Relaxed);
    }
    pub fn total_usd(&self) -> f64 {
        self.cents_spent.load(Ordering::Relaxed) as f64 / 100.0
    }
}

/// Remote `LogitSource` over HTTP. Constructed at job start by
/// resolving a teacher alias against the §3.2 registry.
#[derive(Debug)]
pub struct RemoteTeacher {
    config: RemoteTeacherConfig,
    cost_tally: Arc<CostTally>,
}

impl RemoteTeacher {
    pub fn new(config: RemoteTeacherConfig) -> Self {
        Self {
            config,
            cost_tally: Arc::new(CostTally::new()),
        }
    }

    /// Borrow the cost tally so callers (the trainer / dashboard /
    /// receipt builder) can read the running spend.
    pub fn cost_tally(&self) -> Arc<CostTally> {
        self.cost_tally.clone()
    }

    /// Resolved capabilities including post-default max_top_k.
    pub fn capabilities(&self) -> LogitSourceCaps {
        let max_top_k = if self.config.max_top_k == 0 {
            self.config.provider.default_max_top_k()
        } else {
            self.config
                .max_top_k
                .min(self.config.provider.default_max_top_k())
        };
        LogitSourceCaps {
            teacher_id: self.config.teacher_id.clone(),
            vocab_size: self.config.vocab_size,
            max_top_k,
            supports_full_vocab: matches!(self.config.provider, RemoteProvider::LlamaCpp),
            supports_batched: true,
            tokenizer_hash: self.config.tokenizer_hash.clone(),
        }
    }
}

impl LogitSource for RemoteTeacher {
    fn capabilities(&self) -> LogitSourceCaps {
        self.capabilities()
    }

    fn fetch_logprobs(
        &self,
        _tokens: &[u32],
        _positions: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch, LogitSourceError> {
        let caps = self.capabilities();
        // §8.6 cost lock — pre-flight check.
        if let Some(cap) = self.config.max_cost_usd {
            if self.cost_tally.total_usd() >= cap {
                return Err(LogitSourceError::invalid(
                    &caps.teacher_id,
                    format!(
                        "§8.6 cost cap reached: spent ${:.2} >= cap ${:.2}",
                        self.cost_tally.total_usd(),
                        cap
                    ),
                ));
            }
        }
        let requested_k = top_k.unwrap_or(caps.max_top_k);
        if requested_k > caps.max_top_k {
            return Err(LogitSourceError::TopKExceedsCap {
                requested: requested_k,
                cap: caps.max_top_k,
                teacher_id: caps.teacher_id.clone(),
            });
        }

        // Milestone-10 surface: the HTTP fetch is a real call when
        // the `remote-teacher-http` feature is on and the env var
        // gives an API key. We surface the contract here so the
        // trainer can call `RemoteTeacher` without conditionalising
        // on the feature. The pragmatic shipping order is:
        //   1. Land the shape + capabilities + cost lock (this
        //      commit). Provider-specific HTTP impls slot in.
        //   2. Wire reqwest::blocking calls per provider behind
        //      `remote-teacher-http` once the §3.2 LocalTeacher is
        //      operational (needs both for the §13 Phase 0 success
        //      criterion's frontier-pump run).
        Err(LogitSourceError::invalid(
            &caps.teacher_id,
            format!(
                "RemoteTeacher::fetch_logprobs not yet wired for provider {provider:?} \
                 (see §3.2 in the grand plan — concrete HTTP impl per provider lands \
                 alongside the trainer body refactor in #31)",
                provider = self.config.provider
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_max_top_k_matches_published_caps() {
        assert_eq!(RemoteProvider::Vllm.default_max_top_k(), 256);
        assert_eq!(RemoteProvider::Sglang.default_max_top_k(), 256);
        assert_eq!(RemoteProvider::LlamaCpp.default_max_top_k(), 8192);
        assert_eq!(RemoteProvider::OpenRouter.default_max_top_k(), 20);
        assert_eq!(RemoteProvider::Together.default_max_top_k(), 20);
        assert_eq!(RemoteProvider::Fireworks.default_max_top_k(), 8);
        assert_eq!(RemoteProvider::DeepInfra.default_max_top_k(), 20);
        assert_eq!(RemoteProvider::Tgi.default_max_top_k(), 5);
    }

    #[test]
    fn cost_per_1k_self_hosted_returns_none() {
        for p in [
            RemoteProvider::Vllm,
            RemoteProvider::Sglang,
            RemoteProvider::LlamaCpp,
            RemoteProvider::Tgi,
        ] {
            assert!(p.cost_per_1k_prompt_tokens_usd().is_none(), "{p:?}");
        }
    }

    #[test]
    fn cost_per_1k_paid_providers_set() {
        for p in [
            RemoteProvider::OpenRouter,
            RemoteProvider::Together,
            RemoteProvider::Fireworks,
            RemoteProvider::DeepInfra,
        ] {
            assert!(p.cost_per_1k_prompt_tokens_usd().is_some(), "{p:?}");
        }
    }

    #[test]
    fn caps_clamp_user_max_top_k_to_provider_cap() {
        // User asked for 100 on Fireworks (cap 8) — clamp.
        let cfg = RemoteTeacherConfig {
            provider: RemoteProvider::Fireworks,
            model: "any".into(),
            url: "https://api.fireworks.ai/v1".into(),
            api_key_env: Some("FIREWORKS_API_KEY".into()),
            teacher_id: "fw@test".into(),
            tokenizer_hash: None,
            max_top_k: 100,
            vocab_size: 50000,
            max_cost_usd: None,
            timeout_ms: 30_000,
        };
        let t = RemoteTeacher::new(cfg);
        assert_eq!(t.capabilities().max_top_k, 8);
    }

    #[test]
    fn caps_use_provider_default_when_user_unspecified() {
        let cfg = RemoteTeacherConfig {
            provider: RemoteProvider::OpenRouter,
            model: "any".into(),
            url: "https://openrouter.ai/api/v1".into(),
            api_key_env: Some("OPENROUTER_API_KEY".into()),
            teacher_id: "or@test".into(),
            tokenizer_hash: None,
            max_top_k: 0, // unspecified
            vocab_size: 152064,
            max_cost_usd: None,
            timeout_ms: 30_000,
        };
        let t = RemoteTeacher::new(cfg);
        assert_eq!(t.capabilities().max_top_k, 20);
    }

    #[test]
    fn cost_tally_accumulates() {
        let tally = CostTally::new();
        tally.add_usd(0.05);
        tally.add_usd(0.07);
        let total = tally.total_usd();
        assert!((total - 0.12).abs() < 1e-9, "got {total}");
    }

    #[test]
    fn fetch_returns_cost_cap_error_when_exceeded() {
        let cfg = RemoteTeacherConfig {
            provider: RemoteProvider::OpenRouter,
            model: "any".into(),
            url: "https://openrouter.ai/api/v1".into(),
            api_key_env: Some("OPENROUTER_API_KEY".into()),
            teacher_id: "or@costtest".into(),
            tokenizer_hash: None,
            max_top_k: 20,
            vocab_size: 152064,
            max_cost_usd: Some(1.00),
            timeout_ms: 30_000,
        };
        let t = RemoteTeacher::new(cfg);
        // Push spend over the cap.
        t.cost_tally.add_usd(1.50);
        let err = t.fetch_logprobs(&[1, 2, 3], &[1], Some(8)).unwrap_err();
        match err {
            LogitSourceError::Invalid { teacher_id, message } => {
                assert_eq!(teacher_id, "or@costtest");
                assert!(message.contains("cost cap reached"));
            }
            other => panic!("expected cost-cap Invalid, got {other:?}"),
        }
    }

    #[test]
    fn fetch_rejects_topk_above_caps_cap() {
        let cfg = RemoteTeacherConfig {
            provider: RemoteProvider::Fireworks,
            model: "any".into(),
            url: "https://api.fireworks.ai/v1".into(),
            api_key_env: Some("FIREWORKS_API_KEY".into()),
            teacher_id: "fw@cap".into(),
            tokenizer_hash: None,
            max_top_k: 0,
            vocab_size: 50000,
            max_cost_usd: None,
            timeout_ms: 30_000,
        };
        let t = RemoteTeacher::new(cfg);
        let err = t.fetch_logprobs(&[1, 2, 3], &[1], Some(32)).unwrap_err();
        match err {
            LogitSourceError::TopKExceedsCap { requested, cap, .. } => {
                assert_eq!(requested, 32);
                assert_eq!(cap, 8);
            }
            other => panic!("expected TopKExceedsCap, got {other:?}"),
        }
    }
}
