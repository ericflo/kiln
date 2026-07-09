use serde::{Deserialize, Serialize};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::token::TokenId;

/// The limit that caused Kiln to take over generation and force the model's
/// closing `</think>` token sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingBudgetTrigger {
    Tokens,
    Time,
    /// The completion-wide `max_tokens` cap was close enough that the closing
    /// sequence had to start immediately in order to fit atomically.
    MaxTokens,
}

impl ThinkingBudgetTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tokens => "tokens",
            Self::Time => "time",
            Self::MaxTokens => "max_tokens",
        }
    }
}

/// Runtime outcome of a thinking budget. The state is shared by clones of a
/// request's [`SamplingParams`], which is important for the batching and
/// streaming paths that clone sampling parameters before decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingBudgetStatus {
    pub trigger: Option<ThinkingBudgetTrigger>,
    pub closed: bool,
    pub thinking_tokens: usize,
    pub elapsed_ms: u64,
}

#[derive(Debug, Default)]
struct ThinkingBudgetRuntime {
    started_at: OnceLock<Instant>,
    state: AtomicU8,
    thinking_tokens: AtomicUsize,
    elapsed_ms: AtomicU64,
}

const BUDGET_ACTIVE: u8 = 0;
const BUDGET_NATURALLY_CLOSED: u8 = 1;
const BUDGET_FORCING_TOKENS: u8 = 2;
const BUDGET_FORCING_TIME: u8 = 3;
const BUDGET_FORCING_MAX_TOKENS: u8 = 4;
const BUDGET_CLOSED_TOKENS: u8 = 5;
const BUDGET_CLOSED_TIME: u8 = 6;
const BUDGET_CLOSED_MAX_TOKENS: u8 = 7;
const BUDGET_PUBLISHING: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ThinkingBudgetConfigError {
    #[error("a thinking budget requires at least one token or time limit")]
    MissingLimit,
    #[error("the thinking close sequence must contain at least one token")]
    EmptyCloseSequence,
    #[error(
        "max completion tokens ({max_completion_tokens}) cannot fit the {close_token_count}-token thinking close sequence"
    )]
    CompletionTooShort {
        max_completion_tokens: usize,
        close_token_count: usize,
    },
}

/// Decode-time policy for closing a reasoning block without terminating the
/// completion. Once a limit is reached, [`ThinkingBudget::apply`] replaces
/// sampled tokens with the configured close sequence. Those tokens therefore
/// enter the model's KV history before ordinary answer generation resumes.
///
/// This type is runtime-only. It is deliberately skipped when
/// [`SamplingParams`] is serialized: API/config layers own the user-facing
/// limits, while this object also contains tokenizer-specific token IDs and
/// request-local timing state.
#[derive(Debug, Clone)]
pub struct ThinkingBudget {
    max_tokens: Option<usize>,
    max_time: Option<Duration>,
    max_completion_tokens: usize,
    close_token_ids: Vec<TokenId>,
    runtime: Arc<ThinkingBudgetRuntime>,
}

impl ThinkingBudget {
    pub fn new(
        max_tokens: Option<usize>,
        max_time: Option<Duration>,
        max_completion_tokens: usize,
        close_token_ids: Vec<TokenId>,
    ) -> Result<Self, ThinkingBudgetConfigError> {
        if max_tokens.is_none() && max_time.is_none() {
            return Err(ThinkingBudgetConfigError::MissingLimit);
        }
        if close_token_ids.is_empty() {
            return Err(ThinkingBudgetConfigError::EmptyCloseSequence);
        }
        if max_completion_tokens < close_token_ids.len() {
            return Err(ThinkingBudgetConfigError::CompletionTooShort {
                max_completion_tokens,
                close_token_count: close_token_ids.len(),
            });
        }
        Ok(Self {
            max_tokens,
            max_time,
            max_completion_tokens,
            close_token_ids,
            runtime: Arc::new(ThinkingBudgetRuntime::default()),
        })
    }

    pub fn max_tokens(&self) -> Option<usize> {
        self.max_tokens
    }

    pub fn max_time(&self) -> Option<Duration> {
        self.max_time
    }

    pub fn close_token_count(&self) -> usize {
        self.close_token_ids.len()
    }

    /// Apply the budget decision at the current token boundary.
    pub fn apply(&self, generated: &[TokenId], sampled: TokenId) -> TokenId {
        self.apply_at(generated, sampled, Instant::now())
    }

    fn apply_at(&self, generated: &[TokenId], sampled: TokenId, now: Instant) -> TokenId {
        let mut state = self.load_state();
        if budget_is_closed(state) {
            return sampled;
        }

        let started_at = *self.runtime.started_at.get_or_init(|| now);
        let elapsed = now.saturating_duration_since(started_at);

        // This normally transitions on the candidate that completes the tag.
        // Keeping a suffix check also makes cloned/restored callers robust if
        // they first observe the controller immediately after a completed tag.
        if generated.ends_with(&self.close_token_ids) {
            let thinking_tokens = generated.len().saturating_sub(self.close_token_ids.len());
            if state == BUDGET_ACTIVE {
                state =
                    self.publish_active_outcome(BUDGET_NATURALLY_CLOSED, thinking_tokens, elapsed);
            } else if budget_is_forcing(state) {
                let closed = closed_state_for(state);
                let _ = self.runtime.state.compare_exchange(
                    state,
                    closed,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                state = self.load_state();
            }
            if budget_is_closed(state) {
                return sampled;
            }
        }

        let close_progress = suffix_prefix_len(generated, &self.close_token_ids);
        let thinking_tokens = generated.len().saturating_sub(close_progress);

        if budget_is_forcing(state) {
            let expected = self.close_token_ids[close_progress];
            if close_progress + 1 == self.close_token_ids.len() {
                let _ = self.runtime.state.compare_exchange(
                    state,
                    closed_state_for(state),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
            return expected;
        }

        let trigger = {
            if self
                .max_tokens
                .is_some_and(|limit| thinking_tokens >= limit)
            {
                Some(ThinkingBudgetTrigger::Tokens)
            } else if self.max_time.is_some_and(|limit| elapsed >= limit) {
                Some(ThinkingBudgetTrigger::Time)
            } else if generated.len().saturating_add(self.close_token_ids.len())
                >= self.max_completion_tokens
            {
                Some(ThinkingBudgetTrigger::MaxTokens)
            } else {
                None
            }
        };

        let Some(trigger) = trigger else {
            // Record a natural close as soon as its final token is selected so
            // telemetry still freezes when the completion ends on that token.
            if sampled == self.close_token_ids[close_progress]
                && close_progress + 1 == self.close_token_ids.len()
            {
                self.publish_active_outcome(BUDGET_NATURALLY_CLOSED, thinking_tokens, elapsed);
            }
            return sampled;
        };
        let expected = self.close_token_ids[close_progress];

        // If the model is naturally producing the close sequence, let it do
        // so and leave the budget outcome untriggered. This makes a natural
        // close win even when it begins exactly on a token/time boundary.
        if sampled == expected {
            if close_progress + 1 == self.close_token_ids.len() {
                self.publish_active_outcome(BUDGET_NATURALLY_CLOSED, thinking_tokens, elapsed);
            }
            return sampled;
        }

        let forcing_state = forcing_state_for(trigger);
        let published_state = if close_progress + 1 == self.close_token_ids.len() {
            closed_state_for(forcing_state)
        } else {
            forcing_state
        };
        self.publish_active_outcome(published_state, thinking_tokens, elapsed);
        expected
    }

    pub fn status(&self) -> ThinkingBudgetStatus {
        let state = self.load_state();
        let trigger = decode_trigger(state);
        let elapsed_ms = if state != BUDGET_ACTIVE {
            self.runtime.elapsed_ms.load(Ordering::Acquire)
        } else {
            self.runtime
                .started_at
                .get()
                .map(|started| started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64)
                .unwrap_or(0)
        };
        ThinkingBudgetStatus {
            trigger,
            closed: budget_is_closed(state),
            thinking_tokens: self.runtime.thinking_tokens.load(Ordering::Acquire),
            elapsed_ms,
        }
    }

    fn publish_active_outcome(&self, outcome: u8, thinking_tokens: usize, elapsed: Duration) -> u8 {
        if self
            .runtime
            .state
            .compare_exchange(
                BUDGET_ACTIVE,
                BUDGET_PUBLISHING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.runtime
                .thinking_tokens
                .store(thinking_tokens, Ordering::Relaxed);
            self.runtime.elapsed_ms.store(
                elapsed.as_millis().min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
            // Release-publish the payload only after both values are complete.
            self.runtime.state.store(outcome, Ordering::Release);
            outcome
        } else {
            self.load_state()
        }
    }

    fn load_state(&self) -> u8 {
        loop {
            let state = self.runtime.state.load(Ordering::Acquire);
            if state != BUDGET_PUBLISHING {
                return state;
            }
            std::hint::spin_loop();
        }
    }
}

fn suffix_prefix_len(generated: &[TokenId], close: &[TokenId]) -> usize {
    let max = generated.len().min(close.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&len| generated.ends_with(&close[..len]))
        .unwrap_or(0)
}

fn forcing_state_for(trigger: ThinkingBudgetTrigger) -> u8 {
    match trigger {
        ThinkingBudgetTrigger::Tokens => BUDGET_FORCING_TOKENS,
        ThinkingBudgetTrigger::Time => BUDGET_FORCING_TIME,
        ThinkingBudgetTrigger::MaxTokens => BUDGET_FORCING_MAX_TOKENS,
    }
}

fn decode_trigger(state: u8) -> Option<ThinkingBudgetTrigger> {
    match state {
        BUDGET_FORCING_TOKENS | BUDGET_CLOSED_TOKENS => Some(ThinkingBudgetTrigger::Tokens),
        BUDGET_FORCING_TIME | BUDGET_CLOSED_TIME => Some(ThinkingBudgetTrigger::Time),
        BUDGET_FORCING_MAX_TOKENS | BUDGET_CLOSED_MAX_TOKENS => {
            Some(ThinkingBudgetTrigger::MaxTokens)
        }
        _ => None,
    }
}

fn budget_is_forcing(state: u8) -> bool {
    matches!(
        state,
        BUDGET_FORCING_TOKENS | BUDGET_FORCING_TIME | BUDGET_FORCING_MAX_TOKENS
    )
}

fn budget_is_closed(state: u8) -> bool {
    matches!(
        state,
        BUDGET_NATURALLY_CLOSED
            | BUDGET_CLOSED_TOKENS
            | BUDGET_CLOSED_TIME
            | BUDGET_CLOSED_MAX_TOKENS
    )
}

fn closed_state_for(state: u8) -> u8 {
    match state {
        BUDGET_FORCING_TOKENS => BUDGET_CLOSED_TOKENS,
        BUDGET_FORCING_TIME => BUDGET_CLOSED_TIME,
        BUDGET_FORCING_MAX_TOKENS => BUDGET_CLOSED_MAX_TOKENS,
        _ => state,
    }
}

/// Parameters controlling how tokens are sampled from the model's output logits.
///
/// Defaults follow Qwen3.5-4B's official "thinking mode for general tasks"
/// recommendation from the model card on Hugging Face:
///
/// > temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
/// > presence_penalty=1.5, repetition_penalty=1.0
///
/// Kiln only targets Qwen3.5-4B, so these defaults are tuned for the
/// model's expected sampling regime out of the box. Use the
/// `qwen3_*` constructors below to switch between Qwen's four
/// recommended profiles. Callers that need bit-exact determinism (evals,
/// benchmarks) should call [`SamplingParams::greedy()`] explicitly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature. 0.0 = greedy.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold. 1.0 = disabled.
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling. 0 = disabled.
    #[serde(default = "default_top_k")]
    pub top_k: u32,

    /// Min-p sampling — drop tokens whose probability is below
    /// `min_p * max_prob`. 0.0 = disabled. Applied after temperature
    /// scaling and after top_k filtering, before top_p truncation.
    #[serde(default)]
    pub min_p: f32,

    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Repetition penalty (1.0 = no penalty). HuggingFace-style:
    /// previously-emitted token logits are divided by `repetition_penalty`
    /// when positive, multiplied when negative. Default 1.0 = no-op.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// OpenAI-style presence penalty (-2.0 ..= 2.0). For each token id
    /// that appeared *at least once* in the generated prefix, subtract
    /// `presence_penalty` from its logit. Default 0.0 = no-op.
    ///
    /// Qwen3.5's recommended thinking-mode default is **1.5**.
    #[serde(default)]
    pub presence_penalty: f32,

    /// OpenAI-style frequency penalty (-2.0 ..= 2.0). For each token id
    /// in the generated prefix, subtract `frequency_penalty * count`
    /// from its logit. Default 0.0 = no-op.
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Stop sequences — generation halts when any of these are produced.
    #[serde(default)]
    pub stop: Vec<String>,

    /// Random seed for reproducibility. None = random.
    #[serde(default)]
    pub seed: Option<u64>,

    /// Request-local forced reasoning closure policy. API/config layers build
    /// this only after rendering the prompt and resolving the tokenizer's
    /// close-tag token sequence.
    #[serde(skip)]
    pub thinking_budget: Option<ThinkingBudget>,
}

fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    0.95
}
fn default_top_k() -> u32 {
    20
}
fn default_max_tokens() -> usize {
    2048
}
fn default_repetition_penalty() -> f32 {
    1.0
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::qwen3_thinking_general()
    }
}

impl SamplingParams {
    /// Bit-exact greedy decoding. Used for eval suites and benchmarks
    /// that need deterministic outputs.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            max_tokens: default_max_tokens(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop: vec![],
            seed: None,
            thinking_budget: None,
        }
    }

    /// Qwen3.5 official "thinking mode for general tasks". The kiln
    /// default — these are the numbers a user gets if they fire up the
    /// playground with no overrides.
    pub fn qwen3_thinking_general() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            max_tokens: default_max_tokens(),
            repetition_penalty: 1.0,
            presence_penalty: 1.5,
            frequency_penalty: 0.0,
            stop: vec![],
            seed: None,
            thinking_budget: None,
        }
    }

    /// Qwen3.5 official "thinking mode for precise coding tasks
    /// (e.g. WebDev)". Lower temperature for code generation,
    /// presence_penalty turned off so the model can re-emit identifiers.
    pub fn qwen3_thinking_coding() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            max_tokens: default_max_tokens(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop: vec![],
            seed: None,
            thinking_budget: None,
        }
    }

    /// Qwen3.5 official "instruct (non-thinking) mode for general tasks".
    /// Pair with `chat_template_kwargs.enable_thinking=false`.
    pub fn qwen3_non_thinking_general() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.0,
            max_tokens: default_max_tokens(),
            repetition_penalty: 1.0,
            presence_penalty: 1.5,
            frequency_penalty: 0.0,
            stop: vec![],
            seed: None,
            thinking_budget: None,
        }
    }

    /// Qwen3.5 official "instruct (non-thinking) mode for reasoning tasks".
    /// Identical to thinking-general values but used without `<think>`.
    pub fn qwen3_non_thinking_reasoning() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            max_tokens: default_max_tokens(),
            repetition_penalty: 1.0,
            presence_penalty: 1.5,
            frequency_penalty: 0.0,
            stop: vec![],
            seed: None,
            thinking_budget: None,
        }
    }

    pub fn values_are_effectively_greedy(temperature: f32, top_k: u32) -> bool {
        temperature == 0.0 || (top_k == 1 && temperature.is_finite() && temperature > 0.0)
    }

    pub fn is_effectively_greedy(&self) -> bool {
        Self::values_are_effectively_greedy(self.temperature, self.top_k)
    }

    pub fn top_p_disables_nucleus_filter(top_p: f32) -> bool {
        top_p <= 0.0 || top_p >= 1.0
    }

    /// True when none of the token-history-dependent penalties are
    /// active. Used by the sampler fast paths to skip the history pass.
    pub fn token_penalties_are_no_op(&self) -> bool {
        self.repetition_penalty == 1.0
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
    }

    /// True when min-p filtering is disabled.
    pub fn min_p_is_disabled(min_p: f32) -> bool {
        !min_p.is_finite() || min_p <= 0.0
    }

    /// Replace a sampled token with the next forced close-tag token when the
    /// active thinking budget has elapsed.
    pub fn apply_thinking_budget(&self, generated: &[TokenId], sampled: TokenId) -> TokenId {
        self.thinking_budget
            .as_ref()
            .map(|budget| budget.apply(generated, sampled))
            .unwrap_or(sampled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_qwen3_thinking_general() {
        let d = SamplingParams::default();
        // The official Qwen3.5 model-card recommendation.
        assert_eq!(d.temperature, 1.0);
        assert_eq!(d.top_p, 0.95);
        assert_eq!(d.top_k, 20);
        assert_eq!(d.min_p, 0.0);
        assert_eq!(d.presence_penalty, 1.5);
        assert_eq!(d.repetition_penalty, 1.0);
        assert_eq!(d.frequency_penalty, 0.0);
    }

    #[test]
    fn greedy_is_deterministic() {
        let g = SamplingParams::greedy();
        assert!(g.is_effectively_greedy());
        assert_eq!(g.temperature, 0.0);
    }

    #[test]
    fn qwen3_presets_match_official_recommendations() {
        // Hugging Face model card for Qwen/Qwen3.5-4B.
        let thinking = SamplingParams::qwen3_thinking_general();
        assert_eq!(thinking.temperature, 1.0);
        assert_eq!(thinking.top_p, 0.95);
        assert_eq!(thinking.top_k, 20);
        assert_eq!(thinking.presence_penalty, 1.5);

        let coding = SamplingParams::qwen3_thinking_coding();
        assert_eq!(coding.temperature, 0.6);
        assert_eq!(coding.top_p, 0.95);
        assert_eq!(coding.top_k, 20);
        assert_eq!(coding.presence_penalty, 0.0);

        let non_thinking = SamplingParams::qwen3_non_thinking_general();
        assert_eq!(non_thinking.temperature, 0.7);
        assert_eq!(non_thinking.top_p, 0.8);
        assert_eq!(non_thinking.top_k, 20);
        assert_eq!(non_thinking.presence_penalty, 1.5);

        let reasoning = SamplingParams::qwen3_non_thinking_reasoning();
        assert_eq!(reasoning.temperature, 1.0);
        assert_eq!(reasoning.top_p, 0.95);
        assert_eq!(reasoning.top_k, 20);
        assert_eq!(reasoning.presence_penalty, 1.5);
    }

    #[test]
    fn token_penalty_predicate_detects_no_op() {
        let default = SamplingParams::default();
        // Default has presence_penalty=1.5 — that's a real penalty.
        assert!(!default.token_penalties_are_no_op());

        let greedy = SamplingParams::greedy();
        assert!(greedy.token_penalties_are_no_op());

        let coding = SamplingParams::qwen3_thinking_coding();
        // Coding mode has all penalties off.
        assert!(coding.token_penalties_are_no_op());
    }

    #[test]
    fn thinking_token_budget_forces_the_full_close_sequence() {
        let budget = ThinkingBudget::new(Some(2), None, 16, vec![90, 91, 92]).unwrap();
        let started = Instant::now();

        assert_eq!(budget.apply_at(&[], 10, started), 10);
        assert_eq!(budget.apply_at(&[10], 11, started), 11);
        assert_eq!(budget.apply_at(&[10, 11], 12, started), 90);
        assert_eq!(budget.apply_at(&[10, 11, 90], 13, started), 91);
        assert_eq!(budget.apply_at(&[10, 11, 90, 91], 14, started), 92);
        assert_eq!(budget.apply_at(&[10, 11, 90, 91, 92], 15, started), 15);

        let status = budget.status();
        assert_eq!(status.trigger, Some(ThinkingBudgetTrigger::Tokens));
        assert!(status.closed);
        assert_eq!(status.thinking_tokens, 2);
    }

    #[test]
    fn thinking_time_budget_starts_at_first_decode_candidate() {
        let budget =
            ThinkingBudget::new(None, Some(Duration::from_millis(25)), 16, vec![90]).unwrap();
        let started = Instant::now();

        assert_eq!(budget.apply_at(&[], 10, started), 10);
        assert_eq!(
            budget.apply_at(&[10], 11, started + Duration::from_millis(24)),
            11
        );
        assert_eq!(
            budget.apply_at(&[10, 11], 12, started + Duration::from_millis(25)),
            90
        );
        let status = budget.status();
        assert_eq!(status.trigger, Some(ThinkingBudgetTrigger::Time));
        assert!(status.closed);
        assert_eq!(status.thinking_tokens, 2);
        assert_eq!(status.elapsed_ms, 25);
    }

    #[test]
    fn natural_thinking_close_wins_on_the_budget_boundary() {
        let budget = ThinkingBudget::new(Some(1), None, 16, vec![90, 91]).unwrap();
        let started = Instant::now();

        assert_eq!(budget.apply_at(&[], 10, started), 10);
        assert_eq!(
            budget.apply_at(&[10], 90, started + Duration::from_millis(4)),
            90
        );
        assert_eq!(
            budget.apply_at(&[10, 90], 91, started + Duration::from_millis(5)),
            91
        );
        assert_eq!(
            budget.apply_at(&[10, 90, 91], 12, started + Duration::from_millis(100)),
            12
        );
        let status = budget.status();
        assert_eq!(status.trigger, None);
        assert!(status.closed);
        assert_eq!(status.thinking_tokens, 1);
        assert_eq!(status.elapsed_ms, 5);
    }

    #[test]
    fn completion_limit_reserves_room_for_an_atomic_close() {
        let budget = ThinkingBudget::new(Some(100), None, 5, vec![90, 91]).unwrap();
        let started = Instant::now();

        assert_eq!(budget.apply_at(&[], 10, started), 10);
        assert_eq!(budget.apply_at(&[10], 11, started), 11);
        assert_eq!(budget.apply_at(&[10, 11], 12, started), 12);
        assert_eq!(budget.apply_at(&[10, 11, 12], 13, started), 90);
        assert_eq!(budget.apply_at(&[10, 11, 12, 90], 13, started), 91);
        assert_eq!(
            budget.status().trigger,
            Some(ThinkingBudgetTrigger::MaxTokens)
        );
    }

    #[test]
    fn thinking_budget_rejects_a_completion_too_short_for_the_close() {
        assert_eq!(
            ThinkingBudget::new(Some(0), None, 1, vec![90, 91]).unwrap_err(),
            ThinkingBudgetConfigError::CompletionTooShort {
                max_completion_tokens: 1,
                close_token_count: 2,
            }
        );
    }

    #[test]
    fn deserialization_uses_qwen3_defaults_for_missing_fields() {
        // Round-trip from JSON with no fields set — the result should
        // match the Qwen3.5 thinking-general profile.
        let v: SamplingParams = serde_json::from_str("{}").unwrap();
        assert_eq!(v.temperature, 1.0);
        assert_eq!(v.top_p, 0.95);
        assert_eq!(v.top_k, 20);
        assert_eq!(v.min_p, 0.0);
        assert_eq!(v.repetition_penalty, 1.0);
        // presence/frequency_penalty default to 0 via #[serde(default)] —
        // they're not part of the JSON default-init for SDK callers, only
        // the Rust `Default::default()` injects the 1.5. Document that.
        assert_eq!(v.presence_penalty, 0.0);
        assert_eq!(v.frequency_penalty, 0.0);
    }
}
