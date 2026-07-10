//! Remote-teacher `LogitSource` (grand plan §3.2).
//!
//! HTTP client for strict remote prompt-logprob protocols. The only wired
//! adapter is vLLM's numeric-ID `/v1/completions` `prompt_logprobs` envelope.
//! Other provider names remain representable in configuration but fail before
//! HTTP until they have dedicated request/response adapters and pinned fixtures.
//!
//! - **URL conventions** (path layout, header names).
//! - **top_logprobs cap** — vLLM defaults to 20; an explicit higher config
//!   asserts a matching operator-raised `--max-logprobs` setting.
//! - **API-key header** vs **bearer-token** vs **query-param**.
//! - **Request-shape check** — raw token-ID requests require the provider's
//!   `usage.prompt_tokens` to match exactly. This does not establish tokenizer
//!   or model-content identity; identity-bound caching requires a separate
//!   authoritative handshake.
//!
//! Paid-provider cost accounting is intentionally absent: the only wired
//! provider is self-hosted vLLM. A configured dollar cap is rejected rather
//! than presented as an enforcement mechanism without a billing source.
//!
//! The trait is synchronous (per the §3.2 design decision) and uses
//! `reqwest::blocking` from the trainer's blocking worker.

use std::collections::{HashMap, HashSet};
use std::io::Read;

use serde::{Deserialize, Serialize};

use crate::logit_source::{
    LogitSource, LogitSourceCaps, LogitSourceError, LogprobBatch, TopKLogprobs,
    validate_logit_request, validate_topk_logprob_row, validate_topk_logprobs_batch,
};

/// Provider identities for remote teachers. Only `Vllm` currently has a
/// protocol adapter; all other variants are rejected before network I/O.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RemoteProvider {
    /// vLLM `/v1/completions` numeric-ID `prompt_logprobs`. The upstream
    /// default cap is 20 and can be raised by the server operator.
    Vllm,
    /// sglang. Its echo/logprobs response differs from vLLM prompt_logprobs;
    /// a dedicated adapter is required.
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
            Self::Vllm => 20,
            Self::Sglang => 256,
            Self::LlamaCpp => 8192,
            Self::OpenRouter => 20,
            Self::Together => 20,
            Self::Fireworks => 8,
            Self::DeepInfra => 20,
            Self::Tgi => 5,
        }
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
    /// Base URL of the vLLM server (for example `http://127.0.0.1:8000`).
    pub url: String,
    /// Environment variable to read the API key from. We never
    /// store the key in serialized form. `None` for vLLM servers that do not
    /// require authentication.
    pub api_key_env: Option<String>,
    /// User-facing teacher id (alias). Mirrors
    /// `LogitSourceCaps.teacher_id`.
    pub teacher_id: String,
    /// Expected tokenizer hash. This becomes authoritative only after the
    /// remote identity handshake; prompt-token usage counts validate request
    /// shape but cannot prove that equal numeric IDs have equal semantics.
    pub tokenizer_hash: Option<String>,
    /// Cap on top-K. When 0, defaults to
    /// `provider.default_max_top_k()`.
    #[serde(default)]
    pub max_top_k: usize,
    /// Full vocabulary size — used by capabilities reporting and
    /// kernel dispatch.
    #[serde(default)]
    pub vocab_size: usize,
    /// Reserved paid-provider cost cap. Must be `None` while vLLM is the only
    /// wired provider because no authoritative billing rate exists.
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    /// Per-request timeout (ms). Defaults to 60_000.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    60_000
}

/// Remote `LogitSource` over HTTP. Constructed at job start by
/// resolving a teacher alias against the §3.2 registry.
#[derive(Debug)]
pub struct RemoteTeacher {
    config: RemoteTeacherConfig,
}

impl RemoteTeacher {
    /// Construct a remote teacher only when its protocol adapter and declared
    /// bounds are usable. Unsupported providers fail here, before capabilities
    /// can be observed or any job can be queued.
    pub fn new(config: RemoteTeacherConfig) -> Result<Self, LogitSourceError> {
        if !matches!(config.provider, RemoteProvider::Vllm) {
            return Err(LogitSourceError::invalid(
                &config.teacher_id,
                format!(
                    "RemoteTeacher provider {:?} is not wired; only vLLM numeric-ID prompt_logprobs is supported",
                    config.provider
                ),
            ));
        }
        if config.teacher_id.trim().is_empty() {
            return Err(LogitSourceError::invalid(
                &config.teacher_id,
                "teacher_id must be non-empty",
            ));
        }
        if config.model.trim().is_empty() {
            return Err(LogitSourceError::invalid(
                &config.teacher_id,
                "model must be non-empty",
            ));
        }
        if config.vocab_size == 0 {
            return Err(LogitSourceError::invalid(
                &config.teacher_id,
                "vocab_size must be greater than zero",
            ));
        }
        if config.timeout_ms == 0 {
            return Err(LogitSourceError::invalid(
                &config.teacher_id,
                "timeout_ms must be greater than zero",
            ));
        }
        if config.max_cost_usd.is_some() {
            return Err(LogitSourceError::invalid(
                &config.teacher_id,
                "max_cost_usd is unavailable: vLLM is self-hosted and no metered billing source is wired",
            ));
        }
        vllm_completions_url(&config)?;
        Ok(Self { config })
    }

    /// Resolved capabilities including post-default max_top_k.
    pub fn capabilities(&self) -> LogitSourceCaps {
        let configured_top_k = if self.config.max_top_k == 0 {
            self.config.provider.default_max_top_k()
        } else {
            // A nonzero value is an explicit assertion that the vLLM server
            // was launched with a raised --max-logprobs setting. The identity
            // handshake will replace this trust with a probed capability.
            self.config.max_top_k
        };
        let max_top_k = configured_top_k.min(self.config.vocab_size);
        LogitSourceCaps {
            teacher_id: self.config.teacher_id.clone(),
            vocab_size: self.config.vocab_size,
            max_top_k,
            // No wired remote adapter currently returns LogprobBatch::FullVocab.
            supports_full_vocab: false,
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
        tokens: &[u32],
        positions: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch, LogitSourceError> {
        let caps = self.capabilities();
        let requested_k = top_k.ok_or_else(|| LogitSourceError::FullVocabUnsupported {
            teacher_id: caps.teacher_id.clone(),
        })?;
        validate_logit_request(&caps, tokens, positions, Some(requested_k))?;
        match self.config.provider {
            RemoteProvider::Vllm => {
                fetch_logprobs_vllm(&self.config, &caps, tokens, positions, requested_k)
            }
            _ => Err(LogitSourceError::invalid(
                &caps.teacher_id,
                format!(
                    "RemoteTeacher HTTP impl not yet wired for provider {:?}; \
                     only vLLM numeric-ID prompt_logprobs is supported today",
                    self.config.provider
                ),
            )),
        }
    }
}

#[derive(Debug)]
struct VllmPromptLogprobRequest {
    body: serde_json::Value,
    sent_tokens: Vec<u32>,
}

struct StrictJsonValue(serde_json::Value);

impl<'de> serde::Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonValueVisitor)
    }
}

struct StrictJsonValueVisitor;

impl<'de> serde::de::Visitor<'de> for StrictJsonValueVisitor {
    type Value = StrictJsonValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(serde_json::Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(serde_json::Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(serde_json::Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .map(StrictJsonValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(serde_json::Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(serde_json::Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(serde_json::Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        StrictJsonValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
        while let Some(value) = sequence.next_element::<StrictJsonValue>()? {
            values.push(value.0);
        }
        Ok(StrictJsonValue(serde_json::Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut values = serde_json::Map::with_capacity(object.size_hint().unwrap_or(0));
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object key {key:?}"
                )));
            }
            values.insert(key, object.next_value::<StrictJsonValue>()?.0);
        }
        Ok(StrictJsonValue(serde_json::Value::Object(values)))
    }
}

fn parse_vllm_json_response(
    cfg: &RemoteTeacherConfig,
    body: &str,
) -> Result<serde_json::Value, LogitSourceError> {
    serde_json::from_str::<StrictJsonValue>(body)
        .map(|value| value.0)
        .map_err(|error| {
            LogitSourceError::invalid(&cfg.teacher_id, format!("parse vllm response: {error}"))
        })
}

fn build_vllm_prompt_logprob_request(
    cfg: &RemoteTeacherConfig,
    caps: &LogitSourceCaps,
    tokens: &[u32],
    positions: &[usize],
    top_k: usize,
) -> Result<VllmPromptLogprobRequest, LogitSourceError> {
    validate_logit_request(caps, tokens, positions, Some(top_k))?;

    let mut sent_tokens = tokens.to_vec();
    if positions.iter().any(|&pos| pos == tokens.len() - 1) {
        // Prompt-logprobs position i is produced by logits row i-1. A causal
        // probe exposes the original final logits row without changing it.
        sent_tokens.push(
            *tokens
                .last()
                .expect("request validation rejected empty tokens"),
        );
    }
    // vLLM also requires the one generated-token slot below. The authoritative
    // remote-capability handshake must enforce `sent_tokens.len() + 1` against
    // its model context; until then, a provider context rejection is surfaced.
    sent_tokens.len().checked_add(1).ok_or_else(|| {
        LogitSourceError::invalid(
            &caps.teacher_id,
            "vllm prompt length overflowed while reserving its required generation slot",
        )
    })?;
    let body = serde_json::json!({
        "model": cfg.model,
        "prompt": sent_tokens,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": top_k,
        "n": 1,
        "stream": false,
        "add_special_tokens": false,
    });
    Ok(VllmPromptLogprobRequest { body, sent_tokens })
}

#[derive(Debug, Clone, Copy)]
struct ParsedPromptLogprobCandidate {
    token_id: u32,
    /// Exact JSON number used for semantic response-contract checks. Those
    /// checks must run before conversion can erase a sign or collapse distinct
    /// values into the same f32.
    source_logprob: f64,
    logprob: f32,
    rank: usize,
}

fn invalid_response(cfg: &RemoteTeacherConfig, message: impl Into<String>) -> LogitSourceError {
    LogitSourceError::invalid(
        &cfg.teacher_id,
        format!("invalid vllm response: {}", message.into()),
    )
}

/// Validate a vLLM base URL and return the canonical completions endpoint.
///
/// This is public so API admission and persisted-registry validation use the
/// exact same URL contract as [`RemoteTeacher`] construction.
pub fn normalize_vllm_completions_url(url: &str) -> Result<String, String> {
    let mut url = reqwest::Url::parse(url.trim())
        .map_err(|error| format!("remote teacher URL {url:?} is invalid: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("remote teacher URL must use http or https and include a host".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(
            "remote teacher URL must not embed credentials; configure api_key_env instead"
                .to_string(),
        );
    }
    if url.cannot_be_a_base() || url.query().is_some() || url.fragment().is_some() {
        return Err(
            "remote teacher URL must be an absolute base URL without a query or fragment"
                .to_string(),
        );
    }
    let base_path = url.path().trim_end_matches('/');
    let endpoint_path = if base_path.ends_with("/v1/completions") {
        base_path.to_string()
    } else if base_path.ends_with("/v1") {
        format!("{base_path}/completions")
    } else {
        format!("{base_path}/v1/completions")
    };
    url.set_path(&endpoint_path);
    Ok(url.into())
}

fn vllm_completions_url(cfg: &RemoteTeacherConfig) -> Result<String, LogitSourceError> {
    normalize_vllm_completions_url(&cfg.url)
        .map_err(|message| LogitSourceError::invalid(&cfg.teacher_id, message))
}

fn parse_prompt_logprob_candidate(
    cfg: &RemoteTeacherConfig,
    target_pos: usize,
    raw_id: &str,
    value: &serde_json::Value,
) -> Result<ParsedPromptLogprobCandidate, LogitSourceError> {
    let token_id = raw_id.parse::<u32>().map_err(|_| {
        invalid_response(
            cfg,
            format!("prompt_logprobs[{target_pos}] has non-u32 token key {raw_id:?}"),
        )
    })?;
    if token_id.to_string() != raw_id {
        return Err(invalid_response(
            cfg,
            format!("prompt_logprobs[{target_pos}] token key {raw_id:?} is not canonical decimal"),
        ));
    }
    if token_id as usize >= cfg.vocab_size {
        return Err(invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}] token id {token_id} is outside vocab size {}",
                cfg.vocab_size
            ),
        ));
    }

    let fields = value.as_object().ok_or_else(|| {
        invalid_response(
            cfg,
            format!("prompt_logprobs[{target_pos}][{raw_id:?}] is not an object"),
        )
    })?;
    let logprob_f64 = fields
        .get("logprob")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            invalid_response(
                cfg,
                format!("prompt_logprobs[{target_pos}][{raw_id:?}].logprob is not a JSON number"),
            )
        })?;
    if !logprob_f64.is_finite() {
        return Err(invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}][{raw_id:?}].logprob {logprob_f64:?} is not finite"
            ),
        ));
    }
    if logprob_f64 > 0.0 {
        return Err(invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}][{raw_id:?}].logprob {logprob_f64:?} is positive"
            ),
        ));
    }
    let logprob = logprob_f64 as f32;
    if !logprob.is_finite() {
        return Err(invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}][{raw_id:?}].logprob {logprob_f64:?} is not representable as finite f32"
            ),
        ));
    }

    let rank_u64 = fields
        .get("rank")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            invalid_response(
                cfg,
                format!(
                    "prompt_logprobs[{target_pos}][{raw_id:?}].rank is not an unsigned integer"
                ),
            )
        })?;
    let rank = usize::try_from(rank_u64).map_err(|_| {
        invalid_response(
            cfg,
            format!("prompt_logprobs[{target_pos}][{raw_id:?}].rank {rank_u64} does not fit usize"),
        )
    })?;
    if rank == 0 || rank > cfg.vocab_size {
        return Err(invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}][{raw_id:?}].rank {rank} is outside 1..={}",
                cfg.vocab_size
            ),
        ));
    }
    Ok(ParsedPromptLogprobCandidate {
        token_id,
        source_logprob: logprob_f64,
        logprob,
        rank,
    })
}

fn parse_prompt_logprob_target(
    cfg: &RemoteTeacherConfig,
    caps: &LogitSourceCaps,
    sent_tokens: &[u32],
    prompt_logprobs: &[serde_json::Value],
    logits_row: usize,
    top_k: usize,
) -> Result<Vec<ParsedPromptLogprobCandidate>, LogitSourceError> {
    if top_k == 0 {
        return Err(LogitSourceError::invalid(
            &cfg.teacher_id,
            "remote-teacher top_k must be > 0; observed-only API responses do not provide OPD support",
        ));
    }
    let target_pos = logits_row.checked_add(1).ok_or_else(|| {
        invalid_response(
            cfg,
            format!("logits row {logits_row} overflowed target position"),
        )
    })?;
    let observed_token = *sent_tokens.get(target_pos).ok_or_else(|| {
        invalid_response(
            cfg,
            format!(
                "logits row {logits_row} has no target token at position {target_pos}; final rows require a causal probe"
            ),
        )
    })?;
    let entry = prompt_logprobs.get(target_pos).ok_or_else(|| {
        invalid_response(
            cfg,
            format!("missing prompt_logprobs[{target_pos}] for logits row {logits_row}"),
        )
    })?;
    let object = entry.as_object().ok_or_else(|| {
        invalid_response(
            cfg,
            format!("prompt_logprobs[{target_pos}] is not an object"),
        )
    })?;

    let mut seen_ids = HashSet::with_capacity(object.len());
    let mut candidates = Vec::with_capacity(object.len());
    let mut observed = None;
    let mut response_probability_mass = 0.0f64;
    for (raw_id, value) in object {
        let candidate = parse_prompt_logprob_candidate(cfg, target_pos, raw_id, value)?;
        response_probability_mass += candidate.source_logprob.exp();
        if !seen_ids.insert(candidate.token_id) {
            return Err(invalid_response(
                cfg,
                format!(
                    "prompt_logprobs[{target_pos}] repeats token id {}",
                    candidate.token_id
                ),
            ));
        }
        if candidate.token_id == observed_token {
            observed = Some(candidate);
        }
        candidates.push(candidate);
    }
    const MASS_TOLERANCE: f64 = 32.0 * f32::EPSILON as f64;
    if response_probability_mass > 1.0 + MASS_TOLERANCE {
        return Err(invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}] candidate probability mass {response_probability_mass:.9} exceeds 1"
            ),
        ));
    }

    let observed = observed.ok_or_else(|| {
        invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}] does not contain observed token id {observed_token}"
            ),
        )
    })?;
    let maximum_cardinality = top_k
        .checked_add(1)
        .ok_or_else(|| invalid_response(cfg, "top-k response cardinality overflowed usize"))?;
    if object.len() != top_k && object.len() != maximum_cardinality {
        return Err(invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}] has {} entries; vllm requires top_k {top_k} entries when the observed token is selected or {maximum_cardinality} when it is extra",
                object.len(),
            ),
        ));
    }
    // vLLM concatenates the observed token with torch.topk and then builds a
    // token-ID-keyed dictionary. Cardinality, not observed rank, therefore
    // tells us whether the observed token was selected: ties can give an
    // unselected observed token a rank within 1..=K.
    let observed_is_selected = object.len() == top_k;
    if observed_is_selected && observed.rank > top_k {
        return Err(invalid_response(
            cfg,
            format!(
                "prompt_logprobs[{target_pos}] has {top_k} entries, but observed rank {} is outside the selected top {top_k} and requires {maximum_cardinality} for top_k {top_k}",
                observed.rank
            ),
        ));
    }

    let mut by_rank = HashMap::with_capacity(top_k);
    for candidate in candidates {
        if !observed_is_selected && candidate.token_id == observed_token {
            continue;
        }
        if candidate.rank > top_k {
            return Err(invalid_response(
                cfg,
                format!(
                    "prompt_logprobs[{target_pos}] contains non-observed token {} with rank {} outside top {top_k}",
                    candidate.token_id, candidate.rank
                ),
            ));
        }
        if by_rank.insert(candidate.rank, candidate).is_some() {
            return Err(invalid_response(
                cfg,
                format!(
                    "prompt_logprobs[{target_pos}] contains duplicate top-k rank {}",
                    candidate.rank
                ),
            ));
        }
    }

    let mut ranked = Vec::with_capacity(top_k);
    for rank in 1..=top_k {
        ranked.push(*by_rank.get(&rank).ok_or_else(|| {
            invalid_response(
                cfg,
                format!("prompt_logprobs[{target_pos}] is missing top-k rank {rank}"),
            )
        })?);
    }
    for pair in ranked.windows(2) {
        if pair[0].source_logprob < pair[1].source_logprob {
            return Err(invalid_response(
                cfg,
                format!(
                    "prompt_logprobs[{target_pos}] logprobs increase from rank {} ({}) to rank {} ({})",
                    pair[0].rank, pair[0].source_logprob, pair[1].rank, pair[1].source_logprob
                ),
            ));
        }
    }
    if !observed_is_selected {
        if observed.rank <= top_k {
            let ranked_at_observed = ranked[observed.rank - 1];
            if observed.source_logprob != ranked_at_observed.source_logprob {
                return Err(invalid_response(
                    cfg,
                    format!(
                        "prompt_logprobs[{target_pos}] extra observed token has rank {} but logprob {} does not tie selected rank {} logprob {}",
                        observed.rank,
                        observed.source_logprob,
                        observed.rank,
                        ranked_at_observed.source_logprob
                    ),
                ));
            }
        } else if observed.source_logprob > ranked[top_k - 1].source_logprob {
            return Err(invalid_response(
                cfg,
                format!(
                    "prompt_logprobs[{target_pos}] observed rank {} logprob {} exceeds rank {top_k} logprob {}",
                    observed.rank,
                    observed.source_logprob,
                    ranked[top_k - 1].source_logprob
                ),
            ));
        }
    }
    let row_indices: Vec<u32> = ranked.iter().map(|candidate| candidate.token_id).collect();
    let row_logprobs: Vec<f32> = ranked.iter().map(|candidate| candidate.logprob).collect();
    validate_topk_logprob_row(caps, top_k, logits_row, &row_indices, &row_logprobs)?;
    Ok(ranked)
}

fn parse_vllm_prompt_logprob_response(
    cfg: &RemoteTeacherConfig,
    caps: &LogitSourceCaps,
    sent_tokens: &[u32],
    positions: &[usize],
    top_k: usize,
    parsed: &serde_json::Value,
) -> Result<TopKLogprobs, LogitSourceError> {
    validate_logit_request(caps, sent_tokens, positions, Some(top_k))?;
    if parsed.get("object").and_then(serde_json::Value::as_str) != Some("text_completion") {
        return Err(invalid_response(
            cfg,
            "top-level `object` is not \"text_completion\"",
        ));
    }
    let response_model = parsed
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_response(cfg, "missing string `model`"))?;
    if response_model != cfg.model {
        return Err(invalid_response(
            cfg,
            format!(
                "model {response_model:?} does not match requested model {:?}",
                cfg.model
            ),
        ));
    }
    let usage_prompt_tokens_u64 = parsed
        .get("usage")
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| invalid_response(cfg, "missing integer `usage.prompt_tokens`"))?;
    let usage_prompt_tokens = usize::try_from(usage_prompt_tokens_u64)
        .map_err(|_| invalid_response(cfg, "`usage.prompt_tokens` does not fit usize"))?;
    if usage_prompt_tokens != sent_tokens.len() {
        return Err(invalid_response(
            cfg,
            format!(
                "usage.prompt_tokens is {usage_prompt_tokens}; expected {} sent token ids",
                sent_tokens.len()
            ),
        ));
    }

    let choices = parsed
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid_response(cfg, "missing array `choices`"))?;
    if choices.len() != 1 {
        return Err(invalid_response(
            cfg,
            format!(
                "choices has {} entries; expected exactly one",
                choices.len()
            ),
        ));
    }
    let choice = choices[0]
        .as_object()
        .ok_or_else(|| invalid_response(cfg, "choices[0] is not an object"))?;
    if choice.get("index").and_then(serde_json::Value::as_u64) != Some(0) {
        return Err(invalid_response(
            cfg,
            "choices[0].index is not integer zero",
        ));
    }
    let prompt_logprobs = choice
        .get("prompt_logprobs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid_response(cfg, "missing array `choices[0].prompt_logprobs`"))?;
    if prompt_logprobs.len() != sent_tokens.len() {
        return Err(invalid_response(
            cfg,
            format!(
                "choices[0].prompt_logprobs has {} entries; expected {}",
                prompt_logprobs.len(),
                sent_tokens.len()
            ),
        ));
    }
    if !prompt_logprobs[0].is_null() {
        return Err(invalid_response(
            cfg,
            "choices[0].prompt_logprobs[0] must be null",
        ));
    }

    let mut validated_rows = Vec::with_capacity(sent_tokens.len().saturating_sub(1));
    for logits_row in 0..sent_tokens.len().saturating_sub(1) {
        validated_rows.push(parse_prompt_logprob_target(
            cfg,
            caps,
            sent_tokens,
            prompt_logprobs,
            logits_row,
            top_k,
        )?);
    }

    let expected_flat_len = positions.len().checked_mul(top_k).ok_or_else(|| {
        invalid_response(cfg, "requested position/top-k product overflowed usize")
    })?;
    let mut indices = Vec::with_capacity(expected_flat_len);
    let mut logprobs = Vec::with_capacity(expected_flat_len);
    for &logits_row in positions {
        let row = validated_rows.get(logits_row).ok_or_else(|| {
            invalid_response(
                cfg,
                format!("requested logits row {logits_row} was not present in the response"),
            )
        })?;
        for candidate in row {
            indices.push(candidate.token_id);
            logprobs.push(candidate.logprob);
        }
    }
    let batch = TopKLogprobs {
        indices,
        logprobs,
        top_k,
    };
    validate_topk_logprobs_batch(caps, sent_tokens, positions, top_k, &batch)?;
    Ok(batch)
}

/// vLLM `/v1/completions` `prompt_logprobs` implementation.
///
/// vLLM exposes top-K logprobs at every prompt position via:
/// ```text
/// POST /v1/completions
/// { "model": "...", "prompt": [token ids...],
///   "max_tokens": 1, "temperature": 0,
///   "prompt_logprobs": K }
/// ```
/// Response includes a `prompt_logprobs` array (one entry per prompt
/// position) where each entry is a dict mapping vocab-id → {logprob, ...}.
/// The first entry is `null` (no logprob for the BOS token).
fn fetch_logprobs_vllm(
    cfg: &RemoteTeacherConfig,
    caps: &LogitSourceCaps,
    tokens: &[u32],
    positions: &[usize],
    top_k: usize,
) -> Result<LogprobBatch, LogitSourceError> {
    let prepared = build_vllm_prompt_logprob_request(cfg, caps, tokens, positions, top_k)?;
    if positions.is_empty() {
        let batch = TopKLogprobs {
            indices: Vec::new(),
            logprobs: Vec::new(),
            top_k,
        };
        validate_topk_logprobs_batch(caps, tokens, positions, top_k, &batch)?;
        return Ok(LogprobBatch::TopK(batch));
    }
    let url = vllm_completions_url(cfg)?;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(cfg.timeout_ms))
        .build()
        .map_err(|e| LogitSourceError::transport(&cfg.teacher_id, e.to_string()))?;
    let mut req = client.post(&url).json(&prepared.body);
    if let Some(env_name) = cfg.api_key_env.as_deref() {
        let key = std::env::var(env_name).map_err(|_| {
            LogitSourceError::invalid(
                &cfg.teacher_id,
                format!("configured API-key environment variable {env_name:?} is not set"),
            )
        })?;
        if key.trim().is_empty() {
            return Err(LogitSourceError::invalid(
                &cfg.teacher_id,
                format!("configured API-key environment variable {env_name:?} is empty"),
            ));
        }
        req = req.bearer_auth(key);
    }
    let resp = req
        .send()
        .map_err(|e| LogitSourceError::transport(&cfg.teacher_id, e.to_string()))?;
    let status = resp.status();
    let body_limit = vllm_response_body_limit(prepared.sent_tokens.len(), top_k);
    if resp
        .content_length()
        .is_some_and(|content_length| content_length > body_limit as u64)
    {
        return Err(LogitSourceError::invalid(
            &cfg.teacher_id,
            format!("vllm response Content-Length exceeds bounded limit of {body_limit} bytes"),
        ));
    }
    let body_text = read_bounded_utf8_body(resp, body_limit, &cfg.teacher_id)?;
    if !status.is_success() {
        return Err(LogitSourceError::invalid(
            &cfg.teacher_id,
            format!(
                "vllm {} → {}: {}",
                url,
                status,
                body_text.chars().take(400).collect::<String>()
            ),
        ));
    }
    let parsed = parse_vllm_json_response(cfg, &body_text)?;
    parse_vllm_prompt_logprob_response(cfg, caps, &prepared.sent_tokens, positions, top_k, &parsed)
        .map(LogprobBatch::TopK)
}

fn vllm_response_body_limit(prompt_tokens: usize, top_k: usize) -> usize {
    const MIN_BYTES: usize = 1024 * 1024;
    const HARD_MAX_BYTES: usize = 256 * 1024 * 1024;
    prompt_tokens
        .checked_mul(top_k.saturating_add(1))
        .and_then(|candidates| candidates.checked_mul(128))
        .and_then(|candidate_bytes| {
            prompt_tokens
                .checked_mul(32)
                .and_then(|row_bytes| candidate_bytes.checked_add(row_bytes))
        })
        .and_then(|bytes| bytes.checked_add(64 * 1024))
        .unwrap_or(HARD_MAX_BYTES)
        .clamp(MIN_BYTES, HARD_MAX_BYTES)
}

fn read_bounded_utf8_body(
    reader: impl Read,
    limit: usize,
    teacher_id: &str,
) -> Result<String, LogitSourceError> {
    let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
    reader
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| LogitSourceError::transport(teacher_id, error.to_string()))?;
    if bytes.len() > limit {
        return Err(LogitSourceError::invalid(
            teacher_id,
            format!("vllm response body exceeds bounded limit of {limit} bytes"),
        ));
    }
    String::from_utf8(bytes).map_err(|error| {
        LogitSourceError::invalid(
            teacher_id,
            format!("vllm response body is not valid UTF-8: {error}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vllm_config() -> RemoteTeacherConfig {
        RemoteTeacherConfig {
            provider: RemoteProvider::Vllm,
            model: "teacher-model".into(),
            url: "http://127.0.0.1:9".into(),
            api_key_env: None,
            teacher_id: "teacher@test".into(),
            tokenizer_hash: Some("tokenizer-test".into()),
            max_top_k: 8,
            vocab_size: 128,
            max_cost_usd: None,
            timeout_ms: 10,
        }
    }

    fn candidate(logprob: serde_json::Value, rank: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "logprob": logprob,
            "rank": rank,
            "decoded_token": "unused",
        })
    }

    fn prompt_row(entries: &[(u32, f64, u64)]) -> serde_json::Value {
        let mut row = serde_json::Map::new();
        for &(token_id, logprob, rank) in entries {
            row.insert(
                token_id.to_string(),
                candidate(serde_json::json!(logprob), serde_json::json!(rank)),
            );
        }
        serde_json::Value::Object(row)
    }

    fn valid_response(cfg: &RemoteTeacherConfig, sent_tokens: &[u32]) -> serde_json::Value {
        assert_eq!(sent_tokens, [10, 20, 30]);
        serde_json::json!({
            "object": "text_completion",
            "model": cfg.model,
            "choices": [{
                "index": 0,
                "prompt_logprobs": [
                    null,
                    prompt_row(&[(20, -1.0, 1), (5, -1.2, 2)]),
                    prompt_row(&[(6, -0.8, 1), (7, -1.1, 2), (30, -3.0, 5)]),
                ],
            }],
            "usage": {
                "prompt_tokens": sent_tokens.len(),
                "completion_tokens": 1,
                "total_tokens": sent_tokens.len() + 1,
            },
        })
    }

    fn parse_for(
        cfg: &RemoteTeacherConfig,
        sent_tokens: &[u32],
        positions: &[usize],
        top_k: usize,
        response: &serde_json::Value,
    ) -> Result<TopKLogprobs, LogitSourceError> {
        let caps = RemoteTeacher::new(cfg.clone()).unwrap().capabilities();
        parse_vllm_prompt_logprob_response(cfg, &caps, sent_tokens, positions, top_k, response)
    }

    fn invalid_message(result: Result<TopKLogprobs, LogitSourceError>) -> String {
        match result.unwrap_err() {
            LogitSourceError::Invalid { message, .. } => message,
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    fn target_row_mut(
        response: &mut serde_json::Value,
        target_pos: usize,
    ) -> &mut serde_json::Map<String, serde_json::Value> {
        response["choices"][0]["prompt_logprobs"][target_pos]
            .as_object_mut()
            .unwrap()
    }

    #[test]
    fn request_builder_preserves_rows_and_adds_only_a_final_row_probe() {
        let cfg = vllm_config();
        let caps = RemoteTeacher::new(cfg.clone()).unwrap().capabilities();
        let tokens = [10, 20, 30];

        let ordinary =
            build_vllm_prompt_logprob_request(&cfg, &caps, &tokens, &[0, 1, 0], 2).unwrap();
        assert_eq!(ordinary.sent_tokens, tokens);
        assert_eq!(ordinary.body["prompt"], serde_json::json!(tokens));
        assert_eq!(ordinary.body["n"], 1);
        assert_eq!(ordinary.body["stream"], false);
        assert_eq!(ordinary.body["add_special_tokens"], false);
        assert_eq!(
            ordinary.sent_tokens.len() + ordinary.body["max_tokens"].as_u64().unwrap() as usize,
            4
        );

        let final_row =
            build_vllm_prompt_logprob_request(&cfg, &caps, &tokens, &[2, 0, 2], 2).unwrap();
        assert_eq!(final_row.sent_tokens, [10, 20, 30, 30]);
        assert_eq!(
            final_row.body["prompt"],
            serde_json::json!([10, 20, 30, 30])
        );
        assert_eq!(
            final_row.sent_tokens.len() + final_row.body["max_tokens"].as_u64().unwrap() as usize,
            5
        );
    }

    #[test]
    fn request_builder_rejects_invalid_tokens_positions_vocab_and_k() {
        let cfg = vllm_config();
        let caps = RemoteTeacher::new(cfg.clone()).unwrap().capabilities();
        for (tokens, positions, top_k, needle) in [
            (vec![], vec![], 2, "tokens must not be empty"),
            (vec![10, 128], vec![0], 2, "outside vocab"),
            (vec![10, 20], vec![2], 2, "position"),
            (vec![10, 20], vec![0], 0, "top_k must be greater than zero"),
            (vec![10, 20], vec![0], 129, "exceeds"),
        ] {
            let err = build_vllm_prompt_logprob_request(&cfg, &caps, &tokens, &positions, top_k)
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{err:?} did not contain {needle:?}");
        }

        let mut zero_vocab_cfg = cfg.clone();
        zero_vocab_cfg.vocab_size = 0;
        let err = RemoteTeacher::new(zero_vocab_cfg).unwrap_err().to_string();
        assert!(err.contains("vocab"), "{err}");
    }

    #[test]
    fn fetch_none_requests_full_vocab_instead_of_silently_using_top_k() {
        let teacher = RemoteTeacher::new(vllm_config()).unwrap();
        let err = teacher.fetch_logprobs(&[10, 20], &[0], None).unwrap_err();
        assert!(matches!(err, LogitSourceError::FullVocabUnsupported { .. }));
    }

    #[test]
    fn fetch_rejects_zero_topk_before_http() {
        let teacher = RemoteTeacher::new(vllm_config()).unwrap();
        let error = teacher
            .fetch_logprobs(&[10, 20], &[0], Some(0))
            .unwrap_err()
            .to_string();
        assert!(error.contains("top_k must be greater than zero"), "{error}");
    }

    #[test]
    fn constructor_rejects_sglang_without_using_the_vllm_adapter() {
        let mut config = vllm_config();
        config.provider = RemoteProvider::Sglang;
        let error = RemoteTeacher::new(config).unwrap_err().to_string();
        assert!(error.contains("not wired"), "{error}");
        assert!(error.contains("only vLLM"), "{error}");
    }

    #[test]
    fn parser_rejects_zero_topk_without_indexing_an_empty_support() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let response = valid_response(&cfg, &sent_tokens);
        let error = parse_for(&cfg, &sent_tokens, &[0], 0, &response)
            .unwrap_err()
            .to_string();
        assert!(error.contains("top_k must be greater than zero"), "{error}");
    }

    #[test]
    fn constructor_rejects_unwired_full_vocab_provider() {
        let mut cfg = vllm_config();
        cfg.provider = RemoteProvider::LlamaCpp;
        assert!(RemoteTeacher::new(cfg).is_err());
    }

    #[test]
    fn completions_url_accepts_host_v1_and_complete_endpoint_forms() {
        let mut cfg = vllm_config();
        for (base, expected) in [
            (
                "http://localhost:8000",
                "http://localhost:8000/v1/completions",
            ),
            (
                "http://localhost:8000/",
                "http://localhost:8000/v1/completions",
            ),
            (
                "http://localhost:8000/v1",
                "http://localhost:8000/v1/completions",
            ),
            (
                "http://localhost:8000/api/v1",
                "http://localhost:8000/api/v1/completions",
            ),
            (
                "http://localhost:8000/v1/completions",
                "http://localhost:8000/v1/completions",
            ),
        ] {
            cfg.url = base.into();
            assert_eq!(vllm_completions_url(&cfg).unwrap(), expected);
        }
        cfg.url = "/".into();
        assert!(vllm_completions_url(&cfg).is_err());
        cfg.url = "http://localhost:8000/v1?unexpected=true".into();
        assert!(vllm_completions_url(&cfg).is_err());
        for invalid in [
            "file:///tmp/vllm.sock",
            "ftp://localhost/model",
            "http://",
            "http://user:secret@localhost:8000",
        ] {
            cfg.url = invalid.into();
            assert!(
                vllm_completions_url(&cfg).is_err(),
                "accepted invalid URL {invalid:?}"
            );
        }
    }

    #[test]
    fn response_body_reader_is_bounded_and_utf8_strict() {
        assert_eq!(
            read_bounded_utf8_body(std::io::Cursor::new(b"ok"), 2, "teacher").unwrap(),
            "ok"
        );
        let error = read_bounded_utf8_body(std::io::Cursor::new(b"toolong"), 3, "teacher")
            .unwrap_err()
            .to_string();
        assert!(error.contains("bounded limit"), "{error}");
        let error = read_bounded_utf8_body(std::io::Cursor::new([0xff, 0xfe]), 2, "teacher")
            .unwrap_err()
            .to_string();
        assert!(error.contains("valid UTF-8"), "{error}");
        assert_eq!(vllm_response_body_limit(1, 1), 1024 * 1024);
        assert_eq!(
            vllm_response_body_limit(usize::MAX, usize::MAX),
            256 * 1024 * 1024
        );
    }

    #[test]
    fn configured_missing_api_key_fails_before_http() {
        let mut cfg = vllm_config();
        let env_name = format!(
            "KILN_TEST_REMOTE_TEACHER_KEY_MUST_BE_UNSET_{}",
            std::process::id()
        );
        assert!(std::env::var_os(&env_name).is_none());
        cfg.api_key_env = Some(env_name.clone());
        let teacher = RemoteTeacher::new(cfg).unwrap();
        let error = teacher
            .fetch_logprobs(&[10, 20], &[0], Some(2))
            .unwrap_err()
            .to_string();
        assert!(error.contains(&env_name), "{error}");
        assert!(error.contains("is not set"), "{error}");
    }

    #[test]
    fn empty_positions_return_an_empty_valid_batch_without_http() {
        let teacher = RemoteTeacher::new(vllm_config()).unwrap();
        let batch = teacher.fetch_logprobs(&[10, 20], &[], Some(2)).unwrap();
        let LogprobBatch::TopK(batch) = batch else {
            panic!("expected TopK")
        };
        assert_eq!(batch.top_k, 2);
        assert!(batch.indices.is_empty());
        assert!(batch.logprobs.is_empty());
    }

    #[test]
    fn parser_maps_logits_rows_to_next_prompt_positions_in_requested_order() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let response = valid_response(&cfg, &sent_tokens);
        let batch = parse_for(&cfg, &sent_tokens, &[1, 0, 1], 2, &response).unwrap();
        assert_eq!(batch.top_k, 2);
        assert_eq!(batch.indices, [6, 7, 20, 5, 6, 7]);
        assert_eq!(batch.logprobs, [-0.8, -1.1, -1.0, -1.2, -0.8, -1.1]);
    }

    #[test]
    fn parser_accepts_extra_observed_token_tied_at_the_topk_boundary() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let mut response = valid_response(&cfg, &sent_tokens);
        // vLLM computes the observed rank as count(values > observed) + 1,
        // while torch.topk chooses an arbitrary token among exact ties. The
        // observed token can therefore be extra even though its rank is <= K.
        response["choices"][0]["prompt_logprobs"][1] =
            prompt_row(&[(5, -1.5, 1), (6, -2.0, 2), (20, -2.0, 2)]);

        let batch = parse_for(&cfg, &sent_tokens, &[0], 2, &response).unwrap();
        assert_eq!(batch.indices, [5, 6]);
        assert_eq!(batch.logprobs, [-1.5, -2.0]);
    }

    #[test]
    fn parser_accepts_selected_observed_token_with_sequential_tie_rank() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let mut response = valid_response(&cfg, &sent_tokens);
        // When the observed ID is selected, vLLM's token-ID dictionary keeps
        // the later torch.topk entry and its sequential rank.
        response["choices"][0]["prompt_logprobs"][1] = prompt_row(&[(5, -2.0, 1), (20, -2.0, 2)]);

        let batch = parse_for(&cfg, &sent_tokens, &[0], 2, &response).unwrap();
        assert_eq!(batch.indices, [5, 20]);
        assert_eq!(batch.logprobs, [-2.0, -2.0]);
    }

    #[test]
    fn parser_rejects_extra_observed_in_topk_rank_without_a_tie() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let mut response = valid_response(&cfg, &sent_tokens);
        response["choices"][0]["prompt_logprobs"][1] =
            prompt_row(&[(5, -1.5, 1), (6, -2.0, 2), (20, -2.5, 2)]);

        let message = invalid_message(parse_for(&cfg, &sent_tokens, &[0], 2, &response));
        assert!(
            message.contains("does not tie selected rank 2"),
            "{message}"
        );
    }

    #[test]
    fn parser_rejects_false_observed_tie_hidden_by_f32_rounding() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let mut response = valid_response(&cfg, &sent_tokens);
        let selected_logprob = -2.0_f64;
        let observed_logprob = -2.00000001_f64;
        assert_ne!(selected_logprob, observed_logprob);
        assert_eq!(selected_logprob as f32, observed_logprob as f32);
        response["choices"][0]["prompt_logprobs"][1] = prompt_row(&[
            (5, -1.5, 1),
            (6, selected_logprob, 2),
            (20, observed_logprob, 2),
        ]);

        let message = invalid_message(parse_for(&cfg, &sent_tokens, &[0], 2, &response));
        assert!(
            message.contains("does not tie selected rank 2"),
            "{message}"
        );
    }

    #[test]
    fn strict_json_parser_rejects_identical_duplicate_candidate_keys() {
        let cfg = vllm_config();
        let raw = r#"{
            "object":"text_completion",
            "model":"teacher-model",
            "choices":[{"index":0,"prompt_logprobs":[null,{
                "20":{"logprob":-1.0,"rank":1},
                "20":{"logprob":-1.2,"rank":2}
            }]}],
            "usage":{"prompt_tokens":2}
        }"#;
        let error = parse_vllm_json_response(&cfg, raw).unwrap_err().to_string();
        assert!(
            error.contains("duplicate JSON object key \"20\""),
            "{error}"
        );
    }

    #[test]
    fn parser_accepts_the_causal_probe_for_the_original_final_logits_row() {
        let cfg = vllm_config();
        let caps = RemoteTeacher::new(cfg.clone()).unwrap().capabilities();
        let prepared = build_vllm_prompt_logprob_request(&cfg, &caps, &[10, 20], &[1], 2).unwrap();
        assert_eq!(prepared.sent_tokens, [10, 20, 20]);
        let response = serde_json::json!({
            "object": "text_completion",
            "model": cfg.model,
            "choices": [{
                "index": 0,
                "prompt_logprobs": [
                    null,
                    prompt_row(&[(20, -1.0, 1), (5, -1.2, 2)]),
                    prompt_row(&[(6, -0.8, 1), (7, -1.1, 2), (20, -3.0, 5)]),
                ],
            }],
            "usage": { "prompt_tokens": 3 },
        });
        let batch = parse_vllm_prompt_logprob_response(
            &cfg,
            &caps,
            &prepared.sent_tokens,
            &[1],
            2,
            &response,
        )
        .unwrap();
        assert_eq!(batch.indices, [6, 7]);
    }

    #[test]
    fn parser_rejects_malformed_response_envelopes() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let base = valid_response(&cfg, &sent_tokens);
        let mut cases = Vec::new();

        let mut value = base.clone();
        value["object"] = serde_json::json!("chat.completion");
        cases.push(("wrong object", value, "not \"text_completion\""));
        let mut value = base.clone();
        value.as_object_mut().unwrap().remove("model");
        cases.push(("missing model", value, "missing string `model`"));
        let mut value = base.clone();
        value["model"] = serde_json::json!("wrong-model");
        cases.push(("wrong model", value, "does not match requested model"));
        let mut value = base.clone();
        value["usage"]["prompt_tokens"] = serde_json::json!(2);
        cases.push(("wrong usage", value, "expected 3 sent token ids"));
        let mut value = base.clone();
        value["choices"] = serde_json::json!([]);
        cases.push(("choice count", value, "expected exactly one"));
        let mut value = base.clone();
        value["choices"][0]["index"] = serde_json::json!(1);
        cases.push(("choice index", value, "index is not integer zero"));
        let mut value = base.clone();
        value["choices"][0]["prompt_logprobs"][0] = serde_json::json!({});
        cases.push(("first non-null", value, "must be null"));
        let mut value = base.clone();
        value["choices"][0]["prompt_logprobs"]
            .as_array_mut()
            .unwrap()
            .pop();
        cases.push(("short rows", value, "has 2 entries; expected 3"));
        let mut value = base.clone();
        value["choices"][0]["prompt_logprobs"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::Value::Null);
        cases.push(("long rows", value, "has 4 entries; expected 3"));
        let mut value = base.clone();
        value["choices"][0]["prompt_logprobs"][1] = serde_json::Value::Null;
        cases.push(("null target", value, "prompt_logprobs[1] is not an object"));

        for (name, response, needle) in cases {
            let message = invalid_message(parse_for(&cfg, &sent_tokens, &[0], 2, &response));
            assert!(
                message.contains(needle),
                "{name}: {message:?} did not contain {needle:?}"
            );
        }
    }

    #[test]
    fn parser_rejects_malformed_candidate_ids_values_logprobs_and_ranks() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let base = valid_response(&cfg, &sent_tokens);
        let mut cases = Vec::new();

        let mut value = base.clone();
        target_row_mut(&mut value, 1).insert(
            "01".into(),
            candidate(serde_json::json!(-4.0), serde_json::json!(3)),
        );
        cases.push(("noncanonical id", value, "not canonical decimal"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1).insert(
            "128".into(),
            candidate(serde_json::json!(-4.0), serde_json::json!(3)),
        );
        cases.push(("oov id", value, "outside vocab size"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1).insert("5".into(), serde_json::json!(-1.2));
        cases.push(("non-object", value, "is not an object"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)
            .get_mut("5")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("logprob");
        cases.push(("missing logprob", value, "logprob is not a JSON number"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["logprob"] = serde_json::json!("-1.2");
        cases.push(("string logprob", value, "logprob is not a JSON number"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["logprob"] = serde_json::json!(-1.0e100);
        cases.push(("f32 overflow", value, "not representable as finite f32"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["logprob"] = serde_json::json!(0.1);
        cases.push(("positive logprob", value, "is positive"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["logprob"] = serde_json::json!(1.0e-100);
        cases.push((
            "positive logprob hidden by f32 underflow",
            value,
            "is positive",
        ));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)
            .get_mut("5")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("rank");
        cases.push(("missing rank", value, "rank is not an unsigned integer"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["rank"] = serde_json::json!(2.0);
        cases.push(("float rank", value, "rank is not an unsigned integer"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["rank"] = serde_json::json!(0);
        cases.push(("zero rank", value, "outside 1..=128"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["rank"] = serde_json::json!(129);
        cases.push(("oov rank", value, "outside 1..=128"));

        for (name, response, needle) in cases {
            let message = invalid_message(parse_for(&cfg, &sent_tokens, &[0], 2, &response));
            assert!(
                message.contains(needle),
                "{name}: {message:?} did not contain {needle:?}"
            );
        }
    }

    #[test]
    fn parser_validates_unrequested_prompt_rows_before_returning_requested_data() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let mut response = valid_response(&cfg, &sent_tokens);
        target_row_mut(&mut response, 2).insert(
            "128".into(),
            candidate(serde_json::json!(-4.0), serde_json::json!(3)),
        );
        let message = invalid_message(parse_for(&cfg, &sent_tokens, &[0], 2, &response));
        assert!(
            message.contains("prompt_logprobs[2] token id 128"),
            "{message}"
        );
    }

    #[test]
    fn parser_rejects_observed_cardinality_rank_and_probability_contract_violations() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let base = valid_response(&cfg, &sent_tokens);
        let mut cases = Vec::new();

        let mut value = base.clone();
        let row = target_row_mut(&mut value, 1);
        let observed = row.remove("20").unwrap();
        row.insert("8".into(), observed);
        cases.push((
            "missing observed",
            value,
            "does not contain observed token id 20",
        ));
        let mut value = base.clone();
        target_row_mut(&mut value, 1).insert(
            "8".into(),
            candidate(serde_json::json!(-4.0), serde_json::json!(3)),
        );
        cases.push(("extra non-observed", value, "non-observed token 8"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["rank"] = serde_json::json!(1);
        cases.push(("duplicate top rank", value, "duplicate top-k rank 1"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["20"]["rank"] = serde_json::json!(5);
        cases.push(("wrong K+1 cardinality", value, "requires 3 for top_k 2"));
        let mut value = base.clone();
        target_row_mut(&mut value, 1)["5"]["logprob"] = serde_json::json!(-0.5);
        cases.push(("rank order", value, "logprobs increase"));
        let mut value = base.clone();
        target_row_mut(&mut value, 2)["6"]["logprob"] = serde_json::json!(-1.5);
        target_row_mut(&mut value, 2)["7"]["logprob"] = serde_json::json!(-2.0);
        target_row_mut(&mut value, 2)["30"]["logprob"] = serde_json::json!(-1.9);
        cases.push(("observed exceeds kth", value, "observed rank 5 logprob"));

        for (name, response, needle) in cases {
            let positions = if name == "observed exceeds kth" {
                &[1][..]
            } else {
                &[0][..]
            };
            let message = invalid_message(parse_for(&cfg, &sent_tokens, positions, 2, &response));
            assert!(
                message.contains(needle),
                "{name}: {message:?} did not contain {needle:?}"
            );
        }
    }

    #[test]
    fn parser_rejects_topk_probability_mass_above_one() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let mut response = valid_response(&cfg, &sent_tokens);
        target_row_mut(&mut response, 1)["20"]["logprob"] = serde_json::json!(-0.1);
        target_row_mut(&mut response, 1)["5"]["logprob"] = serde_json::json!(-0.2);
        let error = parse_for(&cfg, &sent_tokens, &[0], 2, &response)
            .unwrap_err()
            .to_string();
        assert!(error.contains("probability mass"), "{error}");
    }

    #[test]
    fn parser_counts_the_extra_observed_candidate_in_probability_mass() {
        let cfg = vllm_config();
        let sent_tokens = [10, 20, 30];
        let mut response = valid_response(&cfg, &sent_tokens);
        target_row_mut(&mut response, 2)["6"]["logprob"] = serde_json::json!(-0.7);
        target_row_mut(&mut response, 2)["7"]["logprob"] = serde_json::json!(-0.8);
        target_row_mut(&mut response, 2)["30"]["logprob"] = serde_json::json!(-0.8);
        let error = parse_for(&cfg, &sent_tokens, &[1], 2, &response)
            .unwrap_err()
            .to_string();
        assert!(error.contains("candidate probability mass"), "{error}");
    }

    #[test]
    fn default_max_top_k_matches_published_caps() {
        assert_eq!(RemoteProvider::Vllm.default_max_top_k(), 20);
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
    fn caps_honor_explicit_operator_raised_vllm_cap() {
        // Explicit vLLM configuration records an operator-raised server cap.
        let cfg = RemoteTeacherConfig {
            provider: RemoteProvider::Vllm,
            model: "any".into(),
            url: "http://127.0.0.1:8000".into(),
            api_key_env: None,
            teacher_id: "vllm@test".into(),
            tokenizer_hash: None,
            max_top_k: 32,
            vocab_size: 50000,
            max_cost_usd: None,
            timeout_ms: 30_000,
        };
        let t = RemoteTeacher::new(cfg).unwrap();
        assert_eq!(t.capabilities().max_top_k, 32);
    }

    #[test]
    fn caps_use_provider_default_when_user_unspecified() {
        let cfg = RemoteTeacherConfig {
            provider: RemoteProvider::Vllm,
            model: "any".into(),
            url: "http://127.0.0.1:8000".into(),
            api_key_env: None,
            teacher_id: "vllm@test".into(),
            tokenizer_hash: None,
            max_top_k: 0, // unspecified
            vocab_size: 152064,
            max_cost_usd: None,
            timeout_ms: 30_000,
        };
        let t = RemoteTeacher::new(cfg).unwrap();
        assert_eq!(t.capabilities().max_top_k, 20);
    }

    #[test]
    fn constructor_rejects_unenforceable_cost_cap() {
        let cfg = RemoteTeacherConfig {
            provider: RemoteProvider::Vllm,
            model: "any".into(),
            url: "http://127.0.0.1:8000".into(),
            api_key_env: None,
            teacher_id: "vllm@costtest".into(),
            tokenizer_hash: None,
            max_top_k: 20,
            vocab_size: 152064,
            max_cost_usd: Some(1.00),
            timeout_ms: 30_000,
        };
        let err = RemoteTeacher::new(cfg).unwrap_err();
        match err {
            LogitSourceError::Invalid {
                teacher_id,
                message,
            } => {
                assert_eq!(teacher_id, "vllm@costtest");
                assert!(message.contains("max_cost_usd"));
                assert!(message.contains("no metered billing source"));
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn fetch_rejects_topk_above_caps_cap() {
        let cfg = RemoteTeacherConfig {
            provider: RemoteProvider::Vllm,
            model: "any".into(),
            url: "http://127.0.0.1:8000".into(),
            api_key_env: None,
            teacher_id: "vllm@cap".into(),
            tokenizer_hash: None,
            max_top_k: 0,
            vocab_size: 50000,
            max_cost_usd: None,
            timeout_ms: 30_000,
        };
        let t = RemoteTeacher::new(cfg).unwrap();
        let err = t.fetch_logprobs(&[1, 2, 3], &[1], Some(32)).unwrap_err();
        match err {
            LogitSourceError::TopKExceedsCap { requested, cap, .. } => {
                assert_eq!(requested, 32);
                assert_eq!(cap, 20);
            }
            other => panic!("expected TopKExceedsCap, got {other:?}"),
        }
    }
}
