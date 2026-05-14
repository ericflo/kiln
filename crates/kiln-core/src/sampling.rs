use serde::{Deserialize, Serialize};

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
