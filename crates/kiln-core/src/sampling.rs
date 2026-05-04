use serde::{Deserialize, Serialize};

/// Parameters controlling how tokens are sampled from the model's output logits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature. 0.0 = greedy, 1.0 = default.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling. 0 = disabled.
    #[serde(default)]
    pub top_k: u32,

    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Repetition penalty (1.0 = no penalty).
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

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
    1.0
}
fn default_max_tokens() -> usize {
    2048
}
fn default_repetition_penalty() -> f32 {
    1.0
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: 0,
            max_tokens: default_max_tokens(),
            repetition_penalty: default_repetition_penalty(),
            stop: vec![],
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
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
}
