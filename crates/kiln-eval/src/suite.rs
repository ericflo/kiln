//! Eval suite, example, and generation-config types.
//!
//! Suites can be authored two ways:
//!
//! 1. As a single JSON document mapping to `EvalSuite` (suite-level
//!    `generation`, `default_scorer`, and an inline `examples: [...]` array).
//! 2. As a JSONL file where each line is one `EvalExample`. This is the
//!    convenient long-form for large suites; pair it with a `suite.json`
//!    sidecar for the suite-level defaults.

use std::collections::BTreeMap;
use std::path::Path;

pub use kiln_core::thinking_budget::ThinkingBudgetOverride as EvalBudgetOverride;
use serde::{Deserialize, Serialize};

use crate::scorers::Scorer;

/// Sampling / decode params used when running an eval example.
///
/// Mirrors the `ChatCompletionRequest` knobs we care about for eval. The
/// defaults are tuned for verifiable-reward evaluation: greedy decoding
/// (temperature=0) and a tight max-token cap so a misbehaving model can't
/// stall the suite. Set `temperature > 0` and `n > 1` for pass@k metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalGenerationParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Number of completions per example. Most metrics use n=1; pass@k tasks
    /// use n>1 and aggregate via the suite-level `aggregate` choice.
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub stop: Vec<String>,
    /// Optional base seed for deterministic decoding. When `None`, execution
    /// inherits the immutable job seed. Every example/completion derives and
    /// records its own decoder seed from that base.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Maximum reasoning tokens before forced `</think>` closure. Omitted
    /// inherits the live server default; null is unlimited; zero is immediate.
    #[serde(default, skip_serializing_if = "EvalBudgetOverride::is_inherit")]
    pub thinking_budget_tokens: EvalBudgetOverride<usize>,
    /// Reasoning wall-clock budget in milliseconds, with the same tri-state
    /// semantics as `thinking_budget_tokens`.
    #[serde(default, skip_serializing_if = "EvalBudgetOverride::is_inherit")]
    pub thinking_budget_ms: EvalBudgetOverride<u64>,
    /// Optional per-eval chat-template variables (e.g. Qwen's
    /// `enable_thinking=false`). Forwarded into the chat template context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

pub fn default_temperature() -> f32 {
    0.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_top_k() -> u32 {
    0
}
pub fn default_max_tokens() -> usize {
    256
}
fn default_n() -> usize {
    1
}

/// How a suite reduces the completions for one example into one independent
/// statistic. Headline metrics, slices, comparisons, and promotion decisions
/// consume the reduced record rather than treating completions as examples.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvalAggregation {
    /// Exactly one completion per example.
    #[default]
    Single,
    /// Average the `k` scorer values. The reduced example passes when the mean
    /// is at least 0.5.
    MeanAtK { k: usize },
    /// Pass when any of the `k` independently seeded completions passes.
    PassAtK { k: usize },
    /// Pass when strictly more than half of the `k` completions pass. `k` must
    /// be odd so a tie cannot be hidden behind an implementation policy.
    MajorityAtK { k: usize },
}

impl EvalAggregation {
    pub fn k(self) -> usize {
        match self {
            Self::Single => 1,
            Self::MeanAtK { k } | Self::PassAtK { k } | Self::MajorityAtK { k } => k,
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Single => "single".to_string(),
            Self::MeanAtK { k } => format!("mean@{k}"),
            Self::PassAtK { k } => format!("pass@{k}"),
            Self::MajorityAtK { k } => format!("majority@{k}"),
        }
    }
}

impl Default for EvalGenerationParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            max_tokens: default_max_tokens(),
            n: default_n(),
            stop: Vec::new(),
            seed: None,
            thinking_budget_tokens: EvalBudgetOverride::Inherit,
            thinking_budget_ms: EvalBudgetOverride::Inherit,
            chat_template_kwargs: None,
        }
    }
}

/// Canonical core chat message, re-exported under the eval-facing name for API
/// compatibility. Plain and agentic conversations now share one wire schema
/// across inference tokenization, eval, and training.
pub use kiln_core::tokenizer::ChatMessage as EvalChatMessage;

impl Default for EvalExample {
    fn default() -> Self {
        Self {
            id: None,
            messages: Vec::new(),
            target: None,
            aliases: Vec::new(),
            tags: Vec::new(),
            metadata: None,
            scorer: None,
            generation: None,
            weight: 1.0,
            tools: None,
        }
    }
}

/// One eval example: a prompt, an expected answer / scoring target, optional
/// per-example overrides for generation params and scorer.
///
/// `id` is used to refer to the example in `EvalResult` and to make A/B
/// comparison stable across adapters. When omitted, the loader auto-fills it
/// from `sha256(messages || target)[..16]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalExample {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub messages: Vec<EvalChatMessage>,
    /// The expected answer. Interpreted by the scorer: a target like
    /// `"42"` works for exact_match and numeric_tolerance; multiple-choice
    /// uses single letters like `"A"`. JSON-validity ignores the target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Optional alternative correct answers — any match counts. Useful for
    /// paraphrase-tolerant exact match and multiple-correct MCQ.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Optional per-example tags (e.g. `["arithmetic", "easy"]`) for slicing
    /// metrics in the aggregate report.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Optional per-example metadata, surfaced back to the user in the
    /// `EvalResult` so domain-specific dashboards can join in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Override the suite-level scorer for this example only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scorer: Option<Scorer>,
    /// Override the suite-level generation params for this example only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<EvalGenerationParams>,
    /// Optional weight contributing to the aggregate. Defaults to 1.0.
    #[serde(default = "default_weight")]
    pub weight: f32,
    /// Optional tool catalogue for this example. Forwarded into the chat
    /// template so Qwen3.5's `<tools>` system block renders with these
    /// definitions. When `None`, the suite-level `tools` are used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

fn default_weight() -> f32 {
    1.0
}

impl EvalExample {
    /// Returns the stable example ID, computing one from a hash when the
    /// caller did not supply one. Idempotent.
    pub fn resolved_id(&self) -> String {
        if let Some(id) = self.id.as_ref()
            && !id.is_empty()
        {
            return id.clone();
        }
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for m in &self.messages {
            hasher.update(m.role.as_bytes());
            hasher.update([0u8]);
            hasher.update(m.content.as_bytes());
            hasher.update([0u8]);
        }
        if let Some(t) = self.target.as_ref() {
            hasher.update(b"|t|");
            hasher.update(t.as_bytes());
        }
        for a in &self.aliases {
            hasher.update(b"|a|");
            hasher.update(a.as_bytes());
        }
        let digest = hasher.finalize();
        let hex: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
        hex
    }
}

/// A suite: header + examples.
///
/// JSON shape (suite-level defaults + inline examples):
///
/// ```json
/// {
///   "name": "math-200",
///   "description": "200 grade-school arithmetic problems",
///   "default_scorer": {"kind": "numeric_tolerance", "rtol": 0.0, "atol": 0.0},
///   "generation": {"temperature": 0.0, "max_tokens": 64},
///   "examples": [
///     {"messages": [{"role":"user","content":"47 + 138 = ?"}], "target": "185"}
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSuite {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Required: the scorer used for any example without a `scorer` override.
    pub default_scorer: Scorer,
    /// Suite-wide generation defaults. Per-example overrides win.
    #[serde(default)]
    pub generation: EvalGenerationParams,
    /// Canonical per-example reduction. Schema-v1 suites may omit this only
    /// when every effective generation has `n=1`.
    #[serde(default)]
    pub aggregation: EvalAggregation,
    /// Optional system message prepended to every example's messages, if the
    /// example doesn't already start with one. Lets the suite author lock in
    /// task framing without duplicating it per example.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Inline examples. May be empty when examples are streamed from a
    /// sidecar JSONL file (see `EvalSuite::load_jsonl`).
    #[serde(default)]
    pub examples: Vec<EvalExample>,
    /// Suite-author-chosen schema version. Defaults to v1 for backward
    /// compatibility with the first generation of suite files.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Tool catalogue rendered into every example's prompt. Lives on the
    /// suite so authoring agentic evals stays terse — a 50-example suite
    /// with one shared tool set declares the schema once here.
    /// Per-example overrides win.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

impl EvalExample {
    /// Resolve the effective tool catalogue for this example given a
    /// suite-level default. Per-example `tools` win; otherwise the suite
    /// fallback is used.
    pub fn effective_tools<'a>(
        &'a self,
        suite_default: Option<&'a [serde_json::Value]>,
    ) -> Option<&'a [serde_json::Value]> {
        self.tools
            .as_deref()
            .or(suite_default)
            .filter(|t| !t.is_empty())
    }
}

fn default_schema_version() -> u32 {
    1
}

impl EvalSuite {
    /// Load a suite from a JSON file. The file is the full `EvalSuite`.
    pub fn load_json(path: &Path) -> Result<Self, SuiteLoadError> {
        let bytes = std::fs::read(path).map_err(|e| SuiteLoadError::Io(format!("{e}")))?;
        let suite: EvalSuite =
            serde_json::from_slice(&bytes).map_err(|e| SuiteLoadError::Parse(format!("{e}")))?;
        suite.validate()?;
        Ok(suite)
    }

    /// Load a suite split across `header.json` + `examples.jsonl`. The header
    /// carries everything except `examples`; each non-empty line of the JSONL
    /// is one `EvalExample`. Used for large suites that benefit from
    /// line-oriented diffs and streaming reads.
    pub fn load_jsonl(header_path: &Path, examples_path: &Path) -> Result<Self, SuiteLoadError> {
        let bytes = std::fs::read(header_path).map_err(|e| SuiteLoadError::Io(format!("{e}")))?;
        let mut suite: EvalSuite =
            serde_json::from_slice(&bytes).map_err(|e| SuiteLoadError::Parse(format!("{e}")))?;
        if !suite.examples.is_empty() {
            return Err(SuiteLoadError::Parse(
                "header file already contains examples; remove them when pairing with JSONL"
                    .to_string(),
            ));
        }
        let file =
            std::fs::File::open(examples_path).map_err(|e| SuiteLoadError::Io(format!("{e}")))?;
        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(file);
        for (idx, line) in reader.lines().enumerate() {
            let line =
                line.map_err(|e| SuiteLoadError::Io(format!("examples line {}: {e}", idx + 1)))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let ex: EvalExample = serde_json::from_str(trimmed).map_err(|e| {
                SuiteLoadError::Parse(format!("invalid example at line {}: {e}", idx + 1))
            })?;
            suite.examples.push(ex);
        }
        suite.validate()?;
        Ok(suite)
    }

    /// Stable per-suite summary surfaced in the listing API.
    pub fn summary(&self) -> EvalSuiteSummary {
        let mut tags: BTreeMap<String, u32> = BTreeMap::new();
        for ex in &self.examples {
            for t in &ex.tags {
                *tags.entry(t.clone()).or_default() += 1;
            }
        }
        EvalSuiteSummary {
            name: self.name.clone(),
            description: self.description.clone(),
            num_examples: self.examples.len(),
            completions_per_example: self.aggregation.k(),
            aggregation: self.aggregation,
            schema_version: self.schema_version,
            default_scorer_kind: self.default_scorer.kind_label(),
            tags,
        }
    }

    pub fn validate(&self) -> Result<(), SuiteLoadError> {
        if !matches!(self.schema_version, 1 | crate::SUITE_SCHEMA_VERSION) {
            return Err(SuiteLoadError::Parse(format!(
                "unsupported suite schema_version {}; supported versions are 1 and {}",
                self.schema_version,
                crate::SUITE_SCHEMA_VERSION
            )));
        }
        let k = self.aggregation.k();
        if k == 0 || k > 128 {
            return Err(SuiteLoadError::Parse(format!(
                "aggregation {} requires k in 1..=128",
                self.aggregation.label()
            )));
        }
        if matches!(self.aggregation, EvalAggregation::MajorityAtK { .. }) && k.is_multiple_of(2) {
            return Err(SuiteLoadError::Parse(format!(
                "aggregation {} requires an odd k so ties are impossible",
                self.aggregation.label()
            )));
        }
        if self.name.trim().is_empty() {
            return Err(SuiteLoadError::Parse(
                "suite name must be non-empty".to_string(),
            ));
        }
        if self.name.contains('/') || self.name.contains('\\') || self.name.contains("..") {
            return Err(SuiteLoadError::Parse(format!(
                "suite name '{}' must not contain path separators",
                self.name
            )));
        }
        if self.examples.is_empty() {
            return Err(SuiteLoadError::Parse(
                "suite must contain at least one example".to_string(),
            ));
        }
        validate_tools_schema(self.tools.as_deref(), "suite-level tools")?;
        let mut resolved_ids = std::collections::BTreeSet::new();
        for (idx, ex) in self.examples.iter().enumerate() {
            if ex.id.as_deref().is_some_and(|id| id.trim().is_empty()) {
                return Err(SuiteLoadError::Parse(format!(
                    "example {idx} has an empty explicit id"
                )));
            }
            let resolved_id = ex.resolved_id();
            if !resolved_ids.insert(resolved_id.clone()) {
                return Err(SuiteLoadError::Parse(format!(
                    "example {idx} has duplicate resolved id {resolved_id:?}; assign unique explicit ids"
                )));
            }
            if ex.messages.is_empty() {
                return Err(SuiteLoadError::Parse(format!(
                    "example {idx} has empty messages"
                )));
            }
            if !ex.weight.is_finite() || ex.weight < 0.0 {
                return Err(SuiteLoadError::Parse(format!(
                    "example {idx} weight {} must be finite and non-negative",
                    ex.weight
                )));
            }
            let n = ex
                .generation
                .as_ref()
                .map(|generation| generation.n)
                .unwrap_or(self.generation.n);
            if n == 0 || n > 128 {
                return Err(SuiteLoadError::Parse(format!(
                    "example {idx} generation.n {n} must be in 1..=128"
                )));
            }
            if self.schema_version == 1 && (n != 1 || self.aggregation != EvalAggregation::Single) {
                return Err(SuiteLoadError::Parse(format!(
                    "schema-v1 multi-completion results are ambiguous; migrate the suite to schema_version {} and choose an explicit aggregation",
                    crate::SUITE_SCHEMA_VERSION
                )));
            }
            if n != k {
                return Err(SuiteLoadError::Parse(format!(
                    "example {idx} generation.n {n} does not match aggregation {} (k={k})",
                    self.aggregation.label()
                )));
            }
            validate_tools_schema(ex.tools.as_deref(), &format!("example {idx} tools"))?;
        }
        Ok(())
    }
}

/// Shape-check the optional `tools` field. Tools must be an array of
/// objects whose `function.name` is a non-empty string. We're tolerant of
/// extra fields (different SDKs ship different metadata), but we draw the
/// line at missing function names — those would silently break the chat
/// template's `<tools>` rendering and produce useless evals.
fn validate_tools_schema(
    tools: Option<&[serde_json::Value]>,
    context: &str,
) -> Result<(), SuiteLoadError> {
    let Some(tools) = tools else {
        return Ok(());
    };
    if tools.is_empty() {
        return Ok(());
    }
    for (idx, tool) in tools.iter().enumerate() {
        let obj = tool.as_object().ok_or_else(|| {
            SuiteLoadError::Parse(format!(
                "{context}: entry {idx} must be a JSON object, got {tool:?}"
            ))
        })?;
        // Accept both the OpenAI nested shape (`{type, function: {name, …}}`)
        // and a flatter shape (`{name, parameters}`) some upstream datasets
        // use. The chat template treats `tools[*]` as opaque JSON
        // anyway — we just need a discoverable name.
        let name = obj
            .get("function")
            .and_then(|f| f.as_object())
            .and_then(|f| f.get("name"))
            .or_else(|| obj.get("name"))
            .and_then(|v| v.as_str());
        match name {
            Some(n) if !n.trim().is_empty() => {}
            _ => {
                return Err(SuiteLoadError::Parse(format!(
                    "{context}: entry {idx} is missing a non-empty `function.name` (or top-level `name`)"
                )));
            }
        }
    }
    Ok(())
}

/// Errors raised while loading a suite from disk.
#[derive(Debug, thiserror::Error)]
pub enum SuiteLoadError {
    #[error("io: {0}")]
    Io(String),
    #[error("parse: {0}")]
    Parse(String),
}

/// Lightweight projection used by the suite-listing API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSuiteSummary {
    pub name: String,
    pub description: Option<String>,
    pub num_examples: usize,
    pub completions_per_example: usize,
    pub aggregation: EvalAggregation,
    pub schema_version: u32,
    pub default_scorer_kind: &'static str,
    /// Tag → count map (sorted).
    pub tags: BTreeMap<String, u32>,
}

/// Compare-mode spec: same suite, multiple adapters. The server runs the
/// suite once per adapter (sharing prompt tokenization where possible) and
/// emits a head-to-head diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCompareSpec {
    pub suite: String,
    /// Adapter names to compare. An empty string means the base model
    /// (no adapter). Order is preserved in the result.
    pub adapters: Vec<String>,
    /// Optional job-level seed. This materializes one paired seed without
    /// replacing the suite's other generation settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Optional generation override applied to every adapter run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<EvalGenerationParams>,
}

/// Auto-eval hook attached to an `SftRequest` or `GrpoRequest`. When set,
/// the training queue worker enqueues an eval against the produced adapter
/// once training finishes.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PostEvalDataScope {
    /// The suite must be disjoint from the admitted training corpus. Exact,
    /// normalized, and persisted source-provenance overlap reject admission.
    #[default]
    HeldOut,
    /// Explicit diagnostic over training data. Results are descriptive only
    /// and cannot drive an accuracy promotion gate.
    TrainSetEval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEvalConfig {
    /// Name of the registered suite to run.
    pub suite: String,
    /// Whether the suite is required to be held out or is an explicitly
    /// labeled train-set diagnostic. Defaults to the fail-closed held-out
    /// policy.
    #[serde(default)]
    pub data_scope: PostEvalDataScope,
    /// Optional generation override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<EvalGenerationParams>,
    /// When set, enables the server's fixed paired-evidence promotion policy.
    /// The candidate must clear the configured floor by its 95% Wilson lower
    /// bound and satisfy the paired exact-test policy; point accuracy alone
    /// never promotes. A conclusive floor failure is renamed with a `.failed`
    /// suffix, while insufficient evidence is retained but not promoted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_accuracy: Option<f32>,
    /// If true, adds a standalone base-model result for browsing. A gate
    /// always creates its own paired comparison against the previously active
    /// adapter (or base), independent of this display-oriented option.
    #[serde(default)]
    pub include_baseline: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorers::NumericTolerance;

    fn ex(role: &str, content: &str, target: &str) -> EvalExample {
        EvalExample {
            messages: vec![EvalChatMessage::new(role, content)],
            target: Some(target.to_string()),
            ..Default::default()
        }
    }

    fn mk_suite(name: &str, scorer: Scorer, examples: Vec<EvalExample>) -> EvalSuite {
        EvalSuite {
            name: name.into(),
            description: None,
            default_scorer: scorer,
            generation: EvalGenerationParams::default(),
            aggregation: EvalAggregation::Single,
            system_prompt: None,
            examples,
            schema_version: 1,
            tools: None,
        }
    }

    #[test]
    fn post_eval_data_scope_defaults_held_out_and_uses_explicit_wire_label() {
        let default: PostEvalConfig = serde_json::from_value(serde_json::json!({
            "suite": "fixture"
        }))
        .unwrap();
        assert_eq!(default.data_scope, PostEvalDataScope::HeldOut);

        let diagnostic: PostEvalConfig = serde_json::from_value(serde_json::json!({
            "suite": "fixture",
            "data_scope": "train-set-eval"
        }))
        .unwrap();
        assert_eq!(diagnostic.data_scope, PostEvalDataScope::TrainSetEval);
    }

    #[test]
    fn multi_completion_suites_require_v2_and_explicit_matching_aggregation() {
        let mut suite = mk_suite(
            "multi",
            Scorer::ExactMatch {
                case_sensitive: true,
                strip_whitespace: false,
            },
            vec![ex("user", "q", "a")],
        );
        suite.generation.n = 3;
        let error = suite.validate().unwrap_err().to_string();
        assert!(error.contains("schema-v1 multi-completion results are ambiguous"));

        suite.schema_version = crate::SUITE_SCHEMA_VERSION;
        suite.aggregation = EvalAggregation::PassAtK { k: 3 };
        suite.validate().unwrap();

        suite.aggregation = EvalAggregation::MeanAtK { k: 2 };
        assert!(
            suite
                .validate()
                .unwrap_err()
                .to_string()
                .contains("does not match aggregation mean@2")
        );

        suite.aggregation = EvalAggregation::MajorityAtK { k: 2 };
        assert!(
            suite
                .validate()
                .unwrap_err()
                .to_string()
                .contains("requires an odd k")
        );
    }

    #[test]
    fn unsupported_suite_schema_versions_fail_closed() {
        let mut suite = mk_suite(
            "future",
            Scorer::ExactMatch {
                case_sensitive: true,
                strip_whitespace: false,
            },
            vec![ex("user", "q", "a")],
        );
        suite.schema_version = crate::SUITE_SCHEMA_VERSION + 1;
        assert!(
            suite
                .validate()
                .unwrap_err()
                .to_string()
                .contains("unsupported suite schema_version")
        );
    }

    #[test]
    fn resolved_id_is_stable_and_id_overrides_hash() {
        let e = ex("user", "What is 1+1?", "2");
        let id1 = e.resolved_id();
        let id2 = e.resolved_id();
        assert_eq!(id1, id2);
        let mut e2 = e.clone();
        e2.id = Some("custom-id".to_string());
        assert_eq!(e2.resolved_id(), "custom-id");
    }

    #[test]
    fn suite_validation_rejects_duplicate_or_empty_example_identities() {
        let example = EvalExample {
            messages: vec![EvalChatMessage::new("user", "same")],
            ..Default::default()
        };
        let mut suite = mk_suite(
            "identity-test",
            Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            vec![example.clone(), example],
        );
        let error = suite.validate().unwrap_err().to_string();
        assert!(error.contains("duplicate resolved id"), "{error}");

        suite.examples.truncate(1);
        suite.examples[0].id = Some(" ".into());
        let error = suite.validate().unwrap_err().to_string();
        assert!(error.contains("empty explicit id"), "{error}");
    }

    #[test]
    fn suite_validate_rejects_empty_name_and_examples() {
        let suite = mk_suite(
            "",
            Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            vec![ex("user", "x", "y")],
        );
        assert!(suite.validate().is_err());

        let suite = mk_suite(
            "ok",
            Scorer::NumericTolerance(NumericTolerance::default()),
            vec![],
        );
        assert!(suite.validate().is_err());
    }

    #[test]
    fn suite_rejects_tools_without_names() {
        let mut suite = mk_suite(
            "ok",
            Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            vec![ex("user", "x", "y")],
        );
        // Tool missing a name.
        suite.tools = Some(vec![serde_json::json!({"type": "function"})]);
        let err = suite.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("function.name"),
            "expected name-validation error, got: {msg}"
        );
    }

    #[test]
    fn suite_accepts_well_formed_tools() {
        let mut suite = mk_suite(
            "ok",
            Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            vec![ex("user", "x", "y")],
        );
        suite.tools = Some(vec![serde_json::json!({
            "type": "function",
            "function": {"name": "search_web", "parameters": {"type": "object"}}
        })]);
        suite
            .validate()
            .expect("well-formed tool entry should pass");
    }

    #[test]
    fn suite_rejects_bad_name() {
        let suite = mk_suite(
            "bad/name",
            Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            vec![ex("user", "x", "y")],
        );
        assert!(suite.validate().is_err());
    }

    #[test]
    fn summary_counts_tags() {
        let mut e1 = ex("user", "a", "1");
        e1.tags = vec!["math".into(), "easy".into()];
        let mut e2 = ex("user", "b", "2");
        e2.tags = vec!["math".into()];
        let suite = mk_suite(
            "math",
            Scorer::NumericTolerance(NumericTolerance::default()),
            vec![e1, e2],
        );
        let s = suite.summary();
        assert_eq!(s.num_examples, 2);
        assert_eq!(s.tags.get("math"), Some(&2));
        assert_eq!(s.tags.get("easy"), Some(&1));
    }

    #[test]
    fn generation_thinking_budgets_roundtrip_all_three_states() {
        let inherited: EvalGenerationParams = serde_json::from_str("{}").unwrap();
        assert_eq!(
            inherited.thinking_budget_tokens,
            EvalBudgetOverride::Inherit
        );

        let unlimited: EvalGenerationParams =
            serde_json::from_str(r#"{"thinking_budget_tokens":null,"thinking_budget_ms":null}"#)
                .unwrap();
        assert_eq!(
            unlimited.thinking_budget_tokens,
            EvalBudgetOverride::Unlimited
        );
        assert_eq!(unlimited.thinking_budget_ms, EvalBudgetOverride::Unlimited);

        let limited: EvalGenerationParams =
            serde_json::from_str(r#"{"thinking_budget_tokens":0,"thinking_budget_ms":1250}"#)
                .unwrap();
        assert_eq!(
            limited.thinking_budget_tokens,
            EvalBudgetOverride::Limited(0)
        );
        assert_eq!(
            limited.thinking_budget_ms,
            EvalBudgetOverride::Limited(1250)
        );
        let json = serde_json::to_value(&limited).unwrap();
        assert_eq!(json["thinking_budget_tokens"], 0);
        assert_eq!(json["thinking_budget_ms"], 1250);
    }

    #[test]
    fn load_json_and_jsonl_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut suite = mk_suite(
            "math",
            Scorer::NumericTolerance(NumericTolerance::default()),
            vec![ex("user", "1+1?", "2")],
        );
        suite.description = Some("toy".into());
        let json = serde_json::to_string(&suite).unwrap();
        let p = dir.path().join("s.json");
        std::fs::write(&p, &json).unwrap();
        let loaded = EvalSuite::load_json(&p).unwrap();
        assert_eq!(loaded.name, "math");
        assert_eq!(loaded.examples.len(), 1);

        // JSONL split
        let mut header = suite.clone();
        header.examples.clear();
        let header_path = dir.path().join("h.json");
        std::fs::write(&header_path, serde_json::to_string(&header).unwrap()).unwrap();
        let ex_path = dir.path().join("e.jsonl");
        let mut buf = String::new();
        for e in &suite.examples {
            buf.push_str(&serde_json::to_string(e).unwrap());
            buf.push('\n');
        }
        std::fs::write(&ex_path, buf).unwrap();
        let loaded = EvalSuite::load_jsonl(&header_path, &ex_path).unwrap();
        assert_eq!(loaded.examples.len(), 1);
    }
}
