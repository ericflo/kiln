//! Canonical trajectory schema for agentic rollouts.
//!
//! See `docs/plans/echo-integration-plan.md` §2 and §B.1 for the design.
//!
//! A trajectory is an ordered sequence of [`TurnSegment`]s. Each segment
//! belongs to a [`TurnKind::Context`] (prompt scaffolding), [`TurnKind::Action`]
//! (assistant generation — target of policy gradient), or
//! [`TurnKind::Observation`] (tool result / environment — target of ECHO's
//! env-CE auxiliary loss).
//!
//! [`ScoredRollout`] attaches a reward to a trajectory. [`AgenticGroup`]
//! groups multiple scored rollouts that share a prompt — the unit of GRPO's
//! group-relative advantage normalization.
//!
//! ## Backwards compatibility
//!
//! Legacy `ScoredCompletion { text: String, reward: f64 }` payloads continue
//! to deserialize because `ScoredRollout` shares those field names. The new
//! optional `trajectory` field is populated only by trajectory-aware
//! emitters; when empty, the rollout behaves exactly like the pre-ECHO
//! single-string completion.
//!
//! `ScoredCompletion` and `GrpoGroup` remain valid type names via
//! `pub type` aliases in `crate::lib`, so call sites that reference them
//! don't need to change. The renames are deliberately *additive*: the same
//! struct, the same field names, plus an optional `trajectory` field.

use serde::{Deserialize, Deserializer, Serialize};

use crate::ChatMessage;

/// Version tag for exact, behavior-policy-bound rollout provenance.
pub const ROLLOUT_PROVENANCE_SCHEMA_V1: &str = "kiln.rollout-provenance.v1";

const MAX_ROLLOUT_IDENTITY_TEXT_BYTES: usize = 256;
const MAX_ROLLOUT_BACKEND_BYTES: usize = 64;
const MAX_ROLLOUT_TOKEN_COUNT: usize = 16_777_216;
const MAX_ROLLOUT_STOP_SEQUENCES: usize = 256;
const MAX_ROLLOUT_STOP_BYTES: usize = 16 * 1024;
const MAX_ROLLOUT_TEMPLATE_TOOLS: usize = 256;
const MAX_ROLLOUT_TEMPLATE_KWARGS: usize = 256;
const MAX_ROLLOUT_TEMPLATE_KEY_BYTES: usize = 256;
const MAX_ROLLOUT_TEMPLATE_INVOCATION_BYTES: usize = 1024 * 1024;

/// Whether an action token came from the model distribution or a deterministic
/// runtime controller such as thinking-budget closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutActionTokenSourceV1 {
    Sampled,
    Forced,
}

/// One exact action-token decision in the full model input sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutActionTokenV1 {
    /// Index into [`RolloutProvenanceV1::input_token_ids`].
    pub sequence_index: usize,
    /// Redundant token ID used to make index drift fail closed.
    pub token_id: u32,
    pub source: RolloutActionTokenSourceV1,
    /// Log-probability under the effective post-filter behavior distribution.
    /// Required for sampled tokens and forbidden for forced tokens.
    pub behavior_logprob: Option<f64>,
}

impl RolloutActionTokenV1 {
    pub fn sampled(sequence_index: usize, token_id: u32, behavior_logprob: f64) -> Self {
        Self {
            sequence_index,
            token_id,
            source: RolloutActionTokenSourceV1::Sampled,
            behavior_logprob: Some(behavior_logprob),
        }
    }

    pub fn forced(sequence_index: usize, token_id: u32) -> Self {
        Self {
            sequence_index,
            token_id,
            source: RolloutActionTokenSourceV1::Forced,
            behavior_logprob: None,
        }
    }
}

/// Content identity of the adapter used by the behavior policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutAdapterIdentityV1 {
    pub name: String,
    pub content_sha256: String,
}

/// Immutable model/runtime identity of the policy that sampled the rollout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutBehaviorPolicyIdentityV1 {
    pub served_model_id: String,
    pub base_model_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<RolloutAdapterIdentityV1>,
    pub inference_config_sha256: String,
    pub implementation: String,
}

/// Exact tokenizer and chat-template identities used to build model inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutTokenizerIdentityV1 {
    pub vocab_sha256: String,
    pub config_sha256: String,
    pub chat_template_sha256: String,
}

/// Non-default inputs supplied to the exact chat-template invocation that
/// produced the rollout prompt. Empty/default values preserve the historical
/// `apply_chat_template(messages)` behavior.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutChatTemplateInvocationV1 {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub template_kwargs: serde_json::Map<String, serde_json::Value>,
}

impl RolloutChatTemplateInvocationV1 {
    fn is_default(&self) -> bool {
        self.tools.is_empty() && self.tool_choice.is_none() && self.template_kwargs.is_empty()
    }

    fn validate(&self) -> Result<(), String> {
        if self.tools.len() > MAX_ROLLOUT_TEMPLATE_TOOLS {
            return Err(format!(
                "rollout template invocation contains {} tools; maximum is {MAX_ROLLOUT_TEMPLATE_TOOLS}",
                self.tools.len()
            ));
        }
        if self.template_kwargs.len() > MAX_ROLLOUT_TEMPLATE_KWARGS {
            return Err(format!(
                "rollout template invocation contains {} template kwargs; maximum is {MAX_ROLLOUT_TEMPLATE_KWARGS}",
                self.template_kwargs.len()
            ));
        }
        if let Some(key) = self
            .template_kwargs
            .keys()
            .find(|key| key.is_empty() || key.len() > MAX_ROLLOUT_TEMPLATE_KEY_BYTES)
        {
            return Err(format!(
                "rollout template kwarg key has {} bytes; expected 1..={MAX_ROLLOUT_TEMPLATE_KEY_BYTES}",
                key.len()
            ));
        }
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize rollout template invocation: {error}"))?;
        if encoded.len() > MAX_ROLLOUT_TEMPLATE_INVOCATION_BYTES {
            return Err(format!(
                "rollout template invocation has {} serialized bytes; maximum is {MAX_ROLLOUT_TEMPLATE_INVOCATION_BYTES}",
                encoded.len()
            ));
        }
        Ok(())
    }
}

/// Effective thinking-budget controls that can replace sampled tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutThinkingBudgetV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_time_ms: Option<u64>,
    pub close_token_ids: Vec<u32>,
}

/// Fully resolved sampling controls used by the behavior policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutSamplingConfigV1 {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub max_tokens: usize,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<RolloutThinkingBudgetV1>,
}

/// Exact, versioned provenance required for off-policy importance correction.
///
/// `input_token_ids` is the complete model sequence at rollout completion,
/// including the original prompt. `action_tokens` identifies model decisions
/// within that sequence; sampled decisions carry behavior log-probabilities,
/// while deterministic runtime insertions are explicit and never masquerade
/// as samples from the model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RolloutProvenanceV1 {
    schema: String,
    pub input_token_ids: Vec<u32>,
    pub prompt_token_count: usize,
    /// Canonical hash of the [`AgenticGroup::messages`] value this rollout
    /// answers. This binds rewards to the intended prompt without attempting
    /// to reconstruct the inference sequence from rendered text.
    pub prompt_messages_sha256: String,
    /// Canonical hash of the rollout's `text` and `trajectory` fields,
    /// excluding the mutable reward.
    pub scored_payload_sha256: String,
    pub action_tokens: Vec<RolloutActionTokenV1>,
    pub behavior_policy: RolloutBehaviorPolicyIdentityV1,
    pub tokenizer: RolloutTokenizerIdentityV1,
    #[serde(
        default,
        skip_serializing_if = "RolloutChatTemplateInvocationV1::is_default"
    )]
    pub template_invocation: RolloutChatTemplateInvocationV1,
    pub sampling: RolloutSamplingConfigV1,
    pub seed: u64,
    pub generation_backend: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RolloutProvenanceV1Wire {
    schema: String,
    input_token_ids: Vec<u32>,
    prompt_token_count: usize,
    prompt_messages_sha256: String,
    scored_payload_sha256: String,
    action_tokens: Vec<RolloutActionTokenV1>,
    behavior_policy: RolloutBehaviorPolicyIdentityV1,
    tokenizer: RolloutTokenizerIdentityV1,
    #[serde(default)]
    template_invocation: RolloutChatTemplateInvocationV1,
    sampling: RolloutSamplingConfigV1,
    seed: u64,
    generation_backend: String,
}

#[derive(Serialize)]
struct ScoredRolloutPayloadIdentityV1<'a> {
    schema: &'static str,
    text: &'a str,
    trajectory: &'a [TurnSegment],
}

/// Canonical prompt identity used by rollout provenance v1.
pub fn rollout_prompt_messages_sha256(messages: &[ChatMessage]) -> Result<String, String> {
    let bytes = serde_json::to_vec(messages)
        .map_err(|error| format!("serialize rollout prompt messages: {error}"))?;
    Ok(kiln_core::config_hashes::sha256_bytes(&bytes))
}

/// Canonical scored-payload identity used by rollout provenance v1. Reward is
/// deliberately excluded so scoring can happen after immutable generation
/// provenance is emitted.
pub fn scored_rollout_payload_sha256(rollout: &ScoredRollout) -> Result<String, String> {
    let identity = ScoredRolloutPayloadIdentityV1 {
        schema: "kiln.scored-rollout-payload.v1",
        text: &rollout.text,
        trajectory: &rollout.trajectory,
    };
    let bytes = serde_json::to_vec(&identity)
        .map_err(|error| format!("serialize scored rollout payload: {error}"))?;
    Ok(kiln_core::config_hashes::sha256_bytes(&bytes))
}

impl RolloutProvenanceV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_token_ids: Vec<u32>,
        prompt_token_count: usize,
        prompt_messages_sha256: String,
        scored_payload_sha256: String,
        action_tokens: Vec<RolloutActionTokenV1>,
        behavior_policy: RolloutBehaviorPolicyIdentityV1,
        tokenizer: RolloutTokenizerIdentityV1,
        sampling: RolloutSamplingConfigV1,
        seed: u64,
        generation_backend: impl Into<String>,
    ) -> Result<Self, String> {
        let provenance = Self {
            schema: ROLLOUT_PROVENANCE_SCHEMA_V1.to_string(),
            input_token_ids,
            prompt_token_count,
            prompt_messages_sha256,
            scored_payload_sha256,
            action_tokens,
            behavior_policy,
            tokenizer,
            template_invocation: RolloutChatTemplateInvocationV1::default(),
            sampling,
            seed,
            generation_backend: generation_backend.into(),
        };
        provenance.validate()?;
        Ok(provenance)
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn sampled_action_tokens(&self) -> impl Iterator<Item = &RolloutActionTokenV1> {
        self.action_tokens
            .iter()
            .filter(|token| token.source == RolloutActionTokenSourceV1::Sampled)
    }

    pub fn with_template_invocation(
        mut self,
        template_invocation: RolloutChatTemplateInvocationV1,
    ) -> Result<Self, String> {
        self.template_invocation = template_invocation;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != ROLLOUT_PROVENANCE_SCHEMA_V1 {
            return Err(format!(
                "unsupported rollout provenance schema {:?}; expected {:?}",
                self.schema, ROLLOUT_PROVENANCE_SCHEMA_V1
            ));
        }
        if self.input_token_ids.is_empty() {
            return Err("rollout provenance input_token_ids must not be empty".to_string());
        }
        if self.input_token_ids.len() > MAX_ROLLOUT_TOKEN_COUNT {
            return Err(format!(
                "rollout provenance contains {} input tokens; maximum is {MAX_ROLLOUT_TOKEN_COUNT}",
                self.input_token_ids.len()
            ));
        }
        if self.prompt_token_count == 0 || self.prompt_token_count > self.input_token_ids.len() {
            return Err(format!(
                "rollout provenance prompt_token_count {} is outside 1..={}",
                self.prompt_token_count,
                self.input_token_ids.len()
            ));
        }
        validate_sha256("prompt_messages_sha256", &self.prompt_messages_sha256)?;
        validate_sha256("scored_payload_sha256", &self.scored_payload_sha256)?;
        if self.action_tokens.is_empty() {
            return Err("rollout provenance action_tokens must not be empty".to_string());
        }
        if self.action_tokens.len() > self.input_token_ids.len() {
            return Err(format!(
                "rollout provenance contains {} action tokens for only {} input tokens",
                self.action_tokens.len(),
                self.input_token_ids.len()
            ));
        }

        let mut previous_index = None;
        let mut sampled_count = 0usize;
        let mut forced_count = 0usize;
        for action in &self.action_tokens {
            if action.sequence_index < self.prompt_token_count
                || action.sequence_index >= self.input_token_ids.len()
            {
                return Err(format!(
                    "rollout action token index {} is outside generated sequence range {}..{}",
                    action.sequence_index,
                    self.prompt_token_count,
                    self.input_token_ids.len()
                ));
            }
            if previous_index.is_some_and(|previous| action.sequence_index <= previous) {
                return Err(format!(
                    "rollout action token indices must be strictly increasing; found {} after {}",
                    action.sequence_index,
                    previous_index.unwrap_or_default()
                ));
            }
            previous_index = Some(action.sequence_index);
            let actual_token_id = self.input_token_ids[action.sequence_index];
            if action.token_id != actual_token_id {
                return Err(format!(
                    "rollout action token at index {} claims id {} but input_token_ids contains {}",
                    action.sequence_index, action.token_id, actual_token_id
                ));
            }
            match (action.source, action.behavior_logprob) {
                (RolloutActionTokenSourceV1::Sampled, Some(logprob))
                    if logprob.is_finite() && logprob <= 1e-6 =>
                {
                    sampled_count += 1;
                }
                (RolloutActionTokenSourceV1::Sampled, Some(logprob)) => {
                    return Err(format!(
                        "sampled rollout action token at index {} has invalid behavior_logprob {logprob}",
                        action.sequence_index
                    ));
                }
                (RolloutActionTokenSourceV1::Sampled, None) => {
                    return Err(format!(
                        "sampled rollout action token at index {} is missing behavior_logprob",
                        action.sequence_index
                    ));
                }
                (RolloutActionTokenSourceV1::Forced, None) => forced_count += 1,
                (RolloutActionTokenSourceV1::Forced, Some(_)) => {
                    return Err(format!(
                        "forced rollout action token at index {} must not carry behavior_logprob",
                        action.sequence_index
                    ));
                }
            }
        }
        if sampled_count == 0 {
            return Err(
                "rollout provenance must contain at least one sampled action token".to_string(),
            );
        }

        validate_identity_text(
            "behavior_policy.served_model_id",
            &self.behavior_policy.served_model_id,
            MAX_ROLLOUT_IDENTITY_TEXT_BYTES,
        )?;
        validate_sha256(
            "behavior_policy.base_model_sha256",
            &self.behavior_policy.base_model_sha256,
        )?;
        validate_sha256(
            "behavior_policy.inference_config_sha256",
            &self.behavior_policy.inference_config_sha256,
        )?;
        validate_identity_text(
            "behavior_policy.implementation",
            &self.behavior_policy.implementation,
            MAX_ROLLOUT_IDENTITY_TEXT_BYTES,
        )?;
        if let Some(adapter) = &self.behavior_policy.adapter {
            validate_identity_text(
                "behavior_policy.adapter.name",
                &adapter.name,
                MAX_ROLLOUT_IDENTITY_TEXT_BYTES,
            )?;
            validate_sha256(
                "behavior_policy.adapter.content_sha256",
                &adapter.content_sha256,
            )?;
        }
        validate_sha256("tokenizer.vocab_sha256", &self.tokenizer.vocab_sha256)?;
        validate_sha256("tokenizer.config_sha256", &self.tokenizer.config_sha256)?;
        validate_sha256(
            "tokenizer.chat_template_sha256",
            &self.tokenizer.chat_template_sha256,
        )?;
        self.template_invocation.validate()?;
        validate_identity_text(
            "generation_backend",
            &self.generation_backend,
            MAX_ROLLOUT_BACKEND_BYTES,
        )?;
        validate_sampling_config(&self.sampling)?;
        if forced_count > 0 {
            let budget = self.sampling.thinking_budget.as_ref().ok_or_else(|| {
                "forced rollout action tokens require sampling.thinking_budget provenance"
                    .to_string()
            })?;
            if let Some(action) = self.action_tokens.iter().find(|action| {
                action.source == RolloutActionTokenSourceV1::Forced
                    && !budget.close_token_ids.contains(&action.token_id)
            }) {
                return Err(format!(
                    "forced rollout action token id {} at index {} is absent from sampling.thinking_budget.close_token_ids",
                    action.token_id, action.sequence_index
                ));
            }
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for RolloutProvenanceV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RolloutProvenanceV1Wire::deserialize(deserializer)?;
        let provenance = Self {
            schema: wire.schema,
            input_token_ids: wire.input_token_ids,
            prompt_token_count: wire.prompt_token_count,
            prompt_messages_sha256: wire.prompt_messages_sha256,
            scored_payload_sha256: wire.scored_payload_sha256,
            action_tokens: wire.action_tokens,
            behavior_policy: wire.behavior_policy,
            tokenizer: wire.tokenizer,
            template_invocation: wire.template_invocation,
            sampling: wire.sampling,
            seed: wire.seed,
            generation_backend: wire.generation_backend,
        };
        provenance.validate().map_err(serde::de::Error::custom)?;
        Ok(provenance)
    }
}

fn validate_sampling_config(config: &RolloutSamplingConfigV1) -> Result<(), String> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(format!(
            "sampling.temperature must be finite and non-negative, got {}",
            config.temperature
        ));
    }
    if !config.top_p.is_finite() || !(0.0..=1.0).contains(&config.top_p) {
        return Err(format!(
            "sampling.top_p must be finite and within [0, 1], got {}",
            config.top_p
        ));
    }
    if !config.min_p.is_finite() || !(0.0..=1.0).contains(&config.min_p) {
        return Err(format!(
            "sampling.min_p must be finite and within [0, 1], got {}",
            config.min_p
        ));
    }
    if config.max_tokens == 0 {
        return Err("sampling.max_tokens must be greater than zero".to_string());
    }
    if !config.repetition_penalty.is_finite() || config.repetition_penalty <= 0.0 {
        return Err(format!(
            "sampling.repetition_penalty must be finite and positive, got {}",
            config.repetition_penalty
        ));
    }
    for (field, value) in [
        ("sampling.presence_penalty", config.presence_penalty),
        ("sampling.frequency_penalty", config.frequency_penalty),
    ] {
        if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
            return Err(format!(
                "{field} must be finite and within [-2, 2], got {value}"
            ));
        }
    }
    if config.stop.len() > MAX_ROLLOUT_STOP_SEQUENCES {
        return Err(format!(
            "sampling.stop has {} entries; maximum is {MAX_ROLLOUT_STOP_SEQUENCES}",
            config.stop.len()
        ));
    }
    let stop_bytes = config.stop.iter().map(String::len).sum::<usize>();
    if stop_bytes > MAX_ROLLOUT_STOP_BYTES {
        return Err(format!(
            "sampling.stop contains {stop_bytes} bytes; maximum is {MAX_ROLLOUT_STOP_BYTES}"
        ));
    }
    if config.stop.iter().any(|stop| stop.is_empty()) {
        return Err("sampling.stop entries must not be empty".to_string());
    }
    if let Some(budget) = &config.thinking_budget {
        if budget.max_tokens.is_none() && budget.max_time_ms.is_none() {
            return Err("sampling.thinking_budget must contain a token or time limit".to_string());
        }
        if budget.close_token_ids.is_empty() {
            return Err("sampling.thinking_budget.close_token_ids must not be empty".to_string());
        }
    }
    Ok(())
}

fn validate_identity_text(field: &str, value: &str, max_bytes: usize) -> Result<(), String> {
    if value.is_empty()
        || value.trim() != value
        || value.len() > max_bytes
        || value.chars().any(char::is_control)
    {
        return Err(format!(
            "{field} must be non-empty, trimmed, control-free, and at most {max_bytes} bytes"
        ));
    }
    Ok(())
}

fn validate_sha256(field: &str, value: &str) -> Result<(), String> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(format!(
            "{field} must use the sha256:<64 lowercase hex> form"
        ));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{field} must use the sha256:<64 lowercase hex> form"
        ));
    }
    Ok(())
}

/// What kind of supervision applies to tokens inside a [`TurnSegment`].
///
/// - [`TurnKind::Context`] — system / user / non-trainable scaffolding;
///   no gradient flows through these tokens.
/// - [`TurnKind::Action`] — assistant-generated tokens; target of policy
///   gradient (and OPD's reverse-KL when configured).
/// - [`TurnKind::Observation`] — tool-result / environment tokens; target
///   of ECHO's env cross-entropy auxiliary loss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnKind {
    #[default]
    Context,
    Action,
    Observation,
}

/// One semantic turn in a trajectory.
///
/// `content` is the raw text the model emitted or saw, *before* chat-template
/// formatting is applied. For assistant tool-call turns this means the raw
/// `<tool_call>...</tool_call>` XML (Qwen XML form); for tool-result turns
/// it is the literal stdout / stderr / file-content bytes.
///
/// `kind` is what kind of supervision applies inside this segment;
/// [`TurnKind`] is the authoritative routing signal — `role` is informational
/// metadata that the chat template uses for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSegment {
    /// Chat role: `"system"` | `"user"` | `"assistant"` | `"tool"` (extensible).
    pub role: String,
    /// Raw content before chat-template formatting.
    pub content: String,
    /// What kind of supervision applies inside this segment.
    #[serde(default)]
    pub kind: TurnKind,
    /// Optional tool-call correlation ID, paired with the corresponding
    /// observation segment when available. Informational; not used by
    /// mask building today.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Bytes at the start of this segment's content that are harness
    /// warnings (e.g. `"WARNINGS:\n- bad tool call format\n"`). Excluded
    /// from `env_mask` when
    /// [`MaskConfig::warning_filter`] is true. Paper §3.2: warnings memorize
    /// in ~60 steps and stop providing useful gradient, while terminal-output
    /// tokens continue to teach.
    ///
    /// [`MaskConfig::warning_filter`]: crate::trajectory_mask::MaskConfig::warning_filter
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning_prefix_len: Option<usize>,
}

impl TurnSegment {
    /// Build a one-segment [`TurnKind::Action`] trajectory from a legacy
    /// `text` string. Used internally to bridge the legacy single-string
    /// completion form into the trajectory schema.
    pub fn legacy_action(text: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content: text,
            kind: TurnKind::Action,
            tool_call_id: None,
            warning_prefix_len: None,
        }
    }
}

/// One scored rollout in a group.
///
/// `text` is the legacy single-string completion text — populated for every
/// rollout, even when `trajectory` is set, so all existing `.text` field
/// access continues to work bit-identically. `reward` is the outcome reward
/// (paper convention: binary 0/1; kiln convention: continuous composite in
/// `[0, 1]`). `trajectory` is the optional canonical multi-turn shape: when
/// non-empty, the trajectory-aware mask builder (and ECHO) consume it; when
/// empty, the rollout is treated as a single-turn legacy completion.
///
/// New emitters should populate both `text` (for compatibility) and
/// `trajectory` (for ECHO). The plain-text `.text` of a multi-turn rollout
/// is conventionally a `<TURN_BREAK>`-joined flattening of the Action
/// segments, matching today's `rollout.py` output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoredRollout {
    /// Legacy single-string completion text. Always populated, even when
    /// `trajectory` is set, so call sites that read `.text` keep working.
    pub text: String,
    /// Outcome reward.
    pub reward: f64,
    /// Optional canonical multi-turn structure. When non-empty, ECHO and the
    /// trajectory-aware masking primitive consume this; the `text` field is
    /// informational/legacy.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trajectory: Vec<TurnSegment>,
    /// Exact generation provenance. Required when training enables recorded
    /// behavior-policy importance correction; legacy data may omit it only
    /// under an explicitly provenance-free training mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<RolloutProvenanceV1>,
}

impl ScoredRollout {
    /// Construct from the legacy `(text, reward)` shape — no trajectory.
    pub fn legacy(text: String, reward: f64) -> Self {
        Self {
            text,
            reward,
            trajectory: Vec::new(),
            provenance: None,
        }
    }

    /// Construct from a trajectory; populates `text` with a
    /// `<TURN_BREAK>`-joined flattening of the Action segments to match
    /// today's `rollout.py` output.
    pub fn from_trajectory(trajectory: Vec<TurnSegment>, reward: f64) -> Self {
        let text = flatten_action_segments(&trajectory);
        Self {
            text,
            reward,
            trajectory,
            provenance: None,
        }
    }

    /// Attach validated exact generation provenance.
    pub fn with_provenance(mut self, provenance: RolloutProvenanceV1) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// True when this rollout carries a multi-turn trajectory ECHO can use.
    pub fn has_trajectory(&self) -> bool {
        !self.trajectory.is_empty()
    }

    /// Borrow the canonical trajectory if present; otherwise synthesize a
    /// one-segment Action trajectory from `text` for use by the masking
    /// primitive.
    ///
    /// Returns a `Cow` so the synthesized trajectory doesn't permanently
    /// mutate the rollout — callers that want to mutate should call
    /// [`Self::ensure_trajectory`] explicitly.
    pub fn effective_trajectory(&self) -> std::borrow::Cow<'_, [TurnSegment]> {
        if self.trajectory.is_empty() {
            std::borrow::Cow::Owned(vec![TurnSegment::legacy_action(self.text.clone())])
        } else {
            std::borrow::Cow::Borrowed(&self.trajectory)
        }
    }

    /// Mutate this rollout so `trajectory` is populated. Idempotent.
    pub fn ensure_trajectory(&mut self) {
        if self.trajectory.is_empty() {
            self.trajectory
                .push(TurnSegment::legacy_action(self.text.clone()));
        }
    }
}

fn flatten_action_segments(trajectory: &[TurnSegment]) -> String {
    let actions: Vec<&str> = trajectory
        .iter()
        .filter(|seg| matches!(seg.kind, TurnKind::Action))
        .map(|seg| seg.content.as_str())
        .collect();
    actions.join("<TURN_BREAK>")
}

/// A group of rollouts sharing a prompt. Group-relative advantage is computed
/// within this set.
///
/// The field name `completions` is preserved for backwards compatibility with
/// the pre-ECHO `GrpoGroup` shape; serde additionally accepts `rollouts` as
/// an alias so canonical forward-looking JSON also deserializes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgenticGroup {
    /// Prompt seen by every rollout in this group (system + user).
    pub messages: Vec<ChatMessage>,
    /// Multiple scored rollouts from the same prompt. Accepts canonical
    /// `rollouts` as a serde alias so new payloads can use either name.
    #[serde(alias = "rollouts")]
    pub completions: Vec<ScoredRollout>,
}

impl AgenticGroup {
    /// Ensure every rollout in the group has a populated `trajectory`. Used
    /// internally when the new mask-builder path is selected.
    pub fn ensure_trajectories(&mut self) {
        for rollout in &mut self.completions {
            rollout.ensure_trajectory();
        }
    }

    /// True when at least one rollout carries a multi-turn trajectory.
    pub fn has_any_trajectory(&self) -> bool {
        self.completions.iter().any(ScoredRollout::has_trajectory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(fill: char) -> String {
        format!("sha256:{}", fill.to_string().repeat(64))
    }

    fn valid_provenance() -> RolloutProvenanceV1 {
        RolloutProvenanceV1::new(
            vec![10, 11, 20, 21],
            2,
            hash('0'),
            hash('1'),
            vec![
                RolloutActionTokenV1::sampled(2, 20, -0.25),
                RolloutActionTokenV1::sampled(3, 21, -1.5),
            ],
            RolloutBehaviorPolicyIdentityV1 {
                served_model_id: "Qwen/Qwen3.5-4B".to_string(),
                base_model_sha256: hash('a'),
                adapter: Some(RolloutAdapterIdentityV1 {
                    name: "reasoning-v3".to_string(),
                    content_sha256: hash('b'),
                }),
                inference_config_sha256: hash('c'),
                implementation: "kiln/0.4.1/rocm".to_string(),
            },
            RolloutTokenizerIdentityV1 {
                vocab_sha256: hash('d'),
                config_sha256: hash('e'),
                chat_template_sha256: hash('f'),
            },
            RolloutSamplingConfigV1 {
                temperature: 0.7,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                max_tokens: 128,
                repetition_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                stop: vec!["<|im_end|>".to_string()],
                thinking_budget: None,
            },
            42,
            "rocm",
        )
        .unwrap()
    }

    #[test]
    fn rollout_provenance_v1_round_trips_canonically() {
        let provenance = valid_provenance();
        let json = serde_json::to_string(&provenance).unwrap();
        let parsed: RolloutProvenanceV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, provenance);
        assert_eq!(parsed.schema(), ROLLOUT_PROVENANCE_SCHEMA_V1);
        assert_eq!(parsed.sampled_action_tokens().count(), 2);
        assert!(json.contains(ROLLOUT_PROVENANCE_SCHEMA_V1));
    }

    #[test]
    fn rollout_provenance_records_template_invocation_and_reads_older_v1() {
        let mut template_kwargs = serde_json::Map::new();
        template_kwargs.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        let provenance = valid_provenance()
            .with_template_invocation(RolloutChatTemplateInvocationV1 {
                tools: vec![serde_json::json!({
                    "type": "function",
                    "function": {"name": "lookup"}
                })],
                tool_choice: Some(serde_json::json!("required")),
                template_kwargs,
            })
            .unwrap();
        let value = serde_json::to_value(&provenance).unwrap();
        assert_eq!(
            value["template_invocation"]["template_kwargs"]["enable_thinking"],
            serde_json::json!(false)
        );
        let parsed: RolloutProvenanceV1 = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(parsed, provenance);

        let mut older_v1 = value;
        older_v1
            .as_object_mut()
            .unwrap()
            .remove("template_invocation");
        let parsed: RolloutProvenanceV1 = serde_json::from_value(older_v1).unwrap();
        assert!(parsed.template_invocation.is_default());
        assert!(
            serde_json::to_value(parsed).unwrap()["template_invocation"].is_null(),
            "default invocation should stay omitted on the wire"
        );
    }

    #[test]
    fn rollout_provenance_rejects_unbounded_template_invocation() {
        let mut template_kwargs = serde_json::Map::new();
        template_kwargs.insert(String::new(), serde_json::json!(true));
        let error = valid_provenance()
            .with_template_invocation(RolloutChatTemplateInvocationV1 {
                template_kwargs,
                ..Default::default()
            })
            .unwrap_err();
        assert!(error.contains("expected 1..="), "{error}");

        let error = valid_provenance()
            .with_template_invocation(RolloutChatTemplateInvocationV1 {
                tools: (0..=MAX_ROLLOUT_TEMPLATE_TOOLS)
                    .map(serde_json::Value::from)
                    .collect(),
                ..Default::default()
            })
            .unwrap_err();
        assert!(error.contains("maximum is 256"), "{error}");
    }

    #[test]
    fn rollout_provenance_payload_hashes_bind_prompt_and_content_but_not_reward() {
        let messages = vec![ChatMessage::new("user", "question")];
        let prompt_hash = rollout_prompt_messages_sha256(&messages).unwrap();
        let changed_prompt_hash = rollout_prompt_messages_sha256(&[ChatMessage::new("user", "other")])
        .unwrap();
        assert_ne!(prompt_hash, changed_prompt_hash);

        let rollout = ScoredRollout::legacy("answer".to_string(), 0.0);
        let mut rescored = rollout.clone();
        rescored.reward = 1.0;
        assert_eq!(
            scored_rollout_payload_sha256(&rollout).unwrap(),
            scored_rollout_payload_sha256(&rescored).unwrap()
        );
        rescored.text.push('!');
        assert_ne!(
            scored_rollout_payload_sha256(&rollout).unwrap(),
            scored_rollout_payload_sha256(&rescored).unwrap()
        );
    }

    #[test]
    fn scored_rollout_round_trips_exact_provenance() {
        let rollout =
            ScoredRollout::legacy("answer".to_string(), 1.0).with_provenance(valid_provenance());
        let json = serde_json::to_string(&rollout).unwrap();
        let parsed: ScoredRollout = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.provenance, rollout.provenance);
    }

    #[test]
    fn rollout_provenance_rejects_unknown_or_misaligned_fields() {
        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["unexpected"] = serde_json::json!(true);
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field"), "{error}");

        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["action_tokens"][0]["token_id"] = serde_json::json!(999);
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("claims id 999"), "{error}");

        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["action_tokens"][1]["sequence_index"] = serde_json::json!(2);
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("strictly increasing"), "{error}");

        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["scored_payload_sha256"] = serde_json::json!("latest");
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sha256:<64 lowercase hex>"), "{error}");
    }

    #[test]
    fn rollout_provenance_rejects_invalid_behavior_logprobs() {
        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["action_tokens"][0]["behavior_logprob"] = serde_json::Value::Null;
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing behavior_logprob"), "{error}");

        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["action_tokens"][0]["behavior_logprob"] = serde_json::json!(0.1);
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid behavior_logprob"), "{error}");

        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["action_tokens"][0]["source"] = serde_json::json!("forced");
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must not carry behavior_logprob"), "{error}");
    }

    #[test]
    fn rollout_provenance_requires_budget_for_forced_tokens() {
        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["action_tokens"][0]["source"] = serde_json::json!("forced");
        value["action_tokens"][0]["behavior_logprob"] = serde_json::Value::Null;
        let error = serde_json::from_value::<RolloutProvenanceV1>(value.clone())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("require sampling.thinking_budget"),
            "{error}"
        );

        value["sampling"]["thinking_budget"] = serde_json::json!({
            "max_tokens": 4,
            "close_token_ids": [999]
        });
        let error = serde_json::from_value::<RolloutProvenanceV1>(value.clone())
            .unwrap_err()
            .to_string();
        assert!(error.contains("absent from"), "{error}");

        value["sampling"]["thinking_budget"]["close_token_ids"] = serde_json::json!([20]);
        let parsed = serde_json::from_value::<RolloutProvenanceV1>(value).unwrap();
        assert_eq!(
            parsed.action_tokens[0].source,
            RolloutActionTokenSourceV1::Forced
        );
    }

    #[test]
    fn rollout_provenance_rejects_invalid_identity_and_sampling() {
        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["behavior_policy"]["base_model_sha256"] = serde_json::json!("model-latest");
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sha256:<64 lowercase hex>"), "{error}");

        let mut value = serde_json::to_value(valid_provenance()).unwrap();
        value["sampling"]["top_p"] = serde_json::json!(1.5);
        let error = serde_json::from_value::<RolloutProvenanceV1>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sampling.top_p"), "{error}");
    }

    #[test]
    fn legacy_scored_completion_text_round_trips() {
        let json = r#"{ "text": "the assistant said this", "reward": 0.85 }"#;
        let rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        assert_eq!(rollout.reward, 0.85);
        assert_eq!(rollout.text, "the assistant said this");
        assert!(rollout.trajectory.is_empty());
        assert!(!rollout.has_trajectory());
    }

    #[test]
    fn canonical_trajectory_form_deserializes_with_text_and_trajectory() {
        let json = r#"{
            "text": "I will read the file<TURN_BREAK>The contents are X",
            "reward": 1.0,
            "trajectory": [
                {"role": "assistant", "content": "I will read the file", "kind": "action"},
                {"role": "tool", "content": "file contents here", "kind": "observation"},
                {"role": "assistant", "content": "The contents are X", "kind": "action"}
            ]
        }"#;
        let rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        assert_eq!(rollout.trajectory.len(), 3);
        assert_eq!(rollout.trajectory[0].kind, TurnKind::Action);
        assert_eq!(rollout.trajectory[1].kind, TurnKind::Observation);
        assert_eq!(rollout.reward, 1.0);
        assert!(rollout.has_trajectory());
    }

    #[test]
    fn agentic_group_accepts_legacy_completions_field_name() {
        let json = r#"{
            "messages": [{"role": "user", "content": "hello"}],
            "completions": [
                {"text": "hi there", "reward": 0.7},
                {"text": "hello!", "reward": 0.9}
            ]
        }"#;
        let group: AgenticGroup = serde_json::from_str(json).unwrap();
        assert_eq!(group.completions.len(), 2);
        assert_eq!(group.completions[0].text, "hi there");
        assert!(!group.has_any_trajectory());
    }

    #[test]
    fn agentic_group_accepts_canonical_rollouts_alias() {
        let json = r#"{
            "messages": [{"role": "user", "content": "x"}],
            "rollouts": [
                {
                    "text": "y",
                    "reward": 1.0,
                    "trajectory": [{"role":"assistant","content":"y","kind":"action"}]
                }
            ]
        }"#;
        let group: AgenticGroup = serde_json::from_str(json).unwrap();
        assert_eq!(group.completions.len(), 1);
        assert_eq!(group.completions[0].text, "y");
        assert!(group.has_any_trajectory());
    }

    #[test]
    fn turn_kind_serde_uses_snake_case() {
        assert_eq!(
            serde_json::to_string(&TurnKind::Action).unwrap(),
            "\"action\""
        );
        assert_eq!(
            serde_json::to_string(&TurnKind::Observation).unwrap(),
            "\"observation\""
        );
        assert_eq!(
            serde_json::to_string(&TurnKind::Context).unwrap(),
            "\"context\""
        );
    }

    #[test]
    fn turn_segment_defaults_to_context_kind() {
        let json = r#"{ "role": "system", "content": "you are helpful" }"#;
        let seg: TurnSegment = serde_json::from_str(json).unwrap();
        assert_eq!(seg.kind, TurnKind::Context);
        assert!(seg.tool_call_id.is_none());
        assert!(seg.warning_prefix_len.is_none());
    }

    #[test]
    fn turn_segment_with_warning_prefix() {
        let json = r#"{
            "role": "tool",
            "content": "WARNINGS:\n- malformed call\n<command_output>ls: cannot access\n</command_output>",
            "kind": "observation",
            "warning_prefix_len": 26
        }"#;
        let seg: TurnSegment = serde_json::from_str(json).unwrap();
        assert_eq!(seg.warning_prefix_len, Some(26));
    }

    #[test]
    fn ensure_trajectory_is_idempotent() {
        let mut rollout = ScoredRollout::legacy("hi".to_string(), 1.0);
        assert!(rollout.trajectory.is_empty());
        rollout.ensure_trajectory();
        rollout.ensure_trajectory();
        assert_eq!(rollout.trajectory.len(), 1);
        assert_eq!(rollout.trajectory[0].content, "hi");
        assert_eq!(rollout.trajectory[0].kind, TurnKind::Action);
    }

    #[test]
    fn legacy_completion_with_zero_reward_round_trips() {
        let json = r#"{ "text": "garbage", "reward": 0.0 }"#;
        let rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        assert_eq!(rollout.reward, 0.0);
        assert_eq!(rollout.text, "garbage");
    }

    #[test]
    fn effective_trajectory_synthesizes_from_text_when_empty() {
        let rollout = ScoredRollout::legacy("only text".to_string(), 1.0);
        let traj = rollout.effective_trajectory();
        assert_eq!(traj.len(), 1);
        assert_eq!(traj[0].kind, TurnKind::Action);
        assert_eq!(traj[0].content, "only text");
        // Original rollout untouched.
        assert!(rollout.trajectory.is_empty());
    }

    #[test]
    fn from_trajectory_flattens_action_segments_to_text() {
        let traj = vec![
            TurnSegment {
                role: "assistant".into(),
                content: "first".into(),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "tool".into(),
                content: "ignored env output".into(),
                kind: TurnKind::Observation,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "assistant".into(),
                content: "second".into(),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
        ];
        let rollout = ScoredRollout::from_trajectory(traj, 0.5);
        assert_eq!(rollout.text, "first<TURN_BREAK>second");
        assert_eq!(rollout.trajectory.len(), 3);
        assert_eq!(rollout.reward, 0.5);
    }

    #[test]
    fn scored_rollout_default_is_empty() {
        let r = ScoredRollout::default();
        assert_eq!(r.text, "");
        assert_eq!(r.reward, 0.0);
        assert!(r.trajectory.is_empty());
    }
}
