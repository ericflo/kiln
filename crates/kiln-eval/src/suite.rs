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
    /// Optional fixed seed for deterministic decoding. When `None` the server
    /// picks one per request (and records it in the result for replayability).
    #[serde(default)]
    pub seed: Option<u64>,
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
            chat_template_kwargs: None,
        }
    }
}

/// A single chat message (mirrors the API `Message` shape, kept here to
/// keep `kiln-eval` independent of the server crate).
///
/// Carries the optional agentic fields (`tool_calls`, `tool_call_id`,
/// `name`) so eval prompts can re-render multi-turn tool-use trajectories
/// through Qwen3.5's chat template *exactly* as the model would have seen
/// them in production. The fields are serialized only when set, so plain
/// `{role, content}` JSON suites continue to round-trip unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// OpenAI-style assistant tool calls. Each entry is typically
    /// `{"id": "…", "type": "function", "function": {"name": "…", "arguments": "…"}}`.
    /// Forwarded into the chat template so the Qwen3.5 `<tool_call>` XML
    /// renders on prior assistant turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Tool name on `tool`-role messages. Some templates branch on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Which assistant tool-call this `tool`-role message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl EvalChatMessage {
    /// Construct a plain `{role, content}` message (back-compat with the
    /// pre-tools shape). Tests and small fixtures use this everywhere; the
    /// struct-literal form is preferred when any agentic field is set.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }
    }
}

impl Default for EvalChatMessage {
    fn default() -> Self {
        Self::new("", "")
    }
}

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
            let line = line.map_err(|e| {
                SuiteLoadError::Io(format!("examples line {}: {e}", idx + 1))
            })?;
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
            default_scorer_kind: self.default_scorer.kind_label(),
            tags,
        }
    }

    fn validate(&self) -> Result<(), SuiteLoadError> {
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
        for (idx, ex) in self.examples.iter().enumerate() {
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
    /// Optional generation override applied to every adapter run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<EvalGenerationParams>,
}

/// Auto-eval hook attached to an `SftRequest` or `GrpoRequest`. When set,
/// the training queue worker enqueues an eval against the produced adapter
/// once training finishes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEvalConfig {
    /// Name of the registered suite to run.
    pub suite: String,
    /// Optional generation override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<EvalGenerationParams>,
    /// When set and the aggregate accuracy is strictly below this threshold,
    /// the produced adapter is unloaded and renamed with a `.failed` suffix.
    /// Use this as a "promote-only-if-good" gate after fine-tuning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_accuracy: Option<f32>,
    /// If true and `min_accuracy` is set, runs the eval against the base
    /// model as a baseline before scoring the trained adapter. Both results
    /// land on the training job's `EvalRunRef`.
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

    fn mk_suite(
        name: &str,
        scorer: Scorer,
        examples: Vec<EvalExample>,
    ) -> EvalSuite {
        EvalSuite {
            name: name.into(),
            description: None,
            default_scorer: scorer,
            generation: EvalGenerationParams::default(),
            system_prompt: None,
            examples,
            schema_version: 1,
            tools: None,
        }
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
        suite.validate().expect("well-formed tool entry should pass");
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
