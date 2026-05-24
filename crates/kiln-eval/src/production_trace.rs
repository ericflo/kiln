//! Production trace export -> tool-call eval suite.
//!
//! This module consumes source-agnostic JSONL exports from production agent
//! systems and builds a first-class [`EvalSuite`] whose examples ask Kiln to
//! reproduce the recorded assistant tool call for that exact prompt. Any
//! upstream pipeline can write one of the supported row shapes:
//!
//! - `prompt-chosen-jsonl`: `prompt_messages` + `chosen` assistant message.
//! - `openai-jsonl`: `messages` contains prompt history plus the current
//!   assistant response as the final message.
//! - `anthropic-jsonl`: Anthropic-style `messages` + `assistant_response` blocks.
//!
//! Sampling is reservoir-based, so large exports can be streamed without
//! holding every eligible turn in memory. Unlike generic SFT synthesis,
//! dedupe defaults to false: repeated turns are workload frequency signal.

use std::collections::{BTreeMap, HashSet};
use std::io::BufRead;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_core::Rng as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::qwen3::{ParsedToolCall, extract_tool_calls};
use crate::scorers::{ArgsScoring, NameMatch, Scorer};
use crate::suite::{EvalChatMessage, EvalExample, EvalGenerationParams, EvalSuite};
use crate::trajectory::{
    AnthropicBlock, AnthropicMessage, anthropic_turn_to_sft_conversation,
};

/// JSONL shape to expect in the production trace export.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProductionTraceFormat {
    /// Inspect each row and pick the first recognized trace row shape.
    Auto,
    /// Rows with `prompt_messages` plus a separate `chosen` assistant message.
    PromptChosenJsonl,
    /// Rows with OpenAI-style `messages` where the final assistant is target.
    OpenAiJsonl,
    /// Rows with Anthropic-style `messages` plus `assistant_response` blocks.
    AnthropicJsonl,
}

/// Sampling/filtering knobs for production trace suite construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionTraceSampling {
    /// Keep at most this many eligible tool-call turns. `None` keeps all.
    #[serde(default = "default_max_examples")]
    pub max_examples: Option<usize>,
    /// Seed for deterministic reservoir sampling. If omitted, one is
    /// generated and returned in [`ProductionTraceSuiteStats::effective_seed`].
    #[serde(default)]
    pub seed: Option<u64>,
    /// Skip examples whose prompt message payload exceeds this many chars.
    #[serde(default = "default_max_prompt_chars")]
    pub max_prompt_chars: usize,
    /// Skip examples whose canonical target tool-call JSON exceeds this many
    /// chars. Production shell/python calls can be large, so the default is
    /// intentionally much higher than generic answer eval synthesis.
    #[serde(default = "default_max_target_chars")]
    pub max_target_chars: usize,
    /// When true, dedupe by exact `(prompt_messages, target)` hash.
    #[serde(default)]
    pub dedupe: bool,
}

fn default_max_examples() -> Option<usize> {
    Some(500)
}

fn default_max_prompt_chars() -> usize {
    512 * 1024
}

fn default_max_target_chars() -> usize {
    128 * 1024
}

impl Default for ProductionTraceSampling {
    fn default() -> Self {
        Self {
            max_examples: default_max_examples(),
            seed: None,
            max_prompt_chars: default_max_prompt_chars(),
            max_target_chars: default_max_target_chars(),
            dedupe: false,
        }
    }
}

/// Configuration for [`synthesize_production_trace_suite`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionTraceSuiteConfig {
    pub suite_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub input_format: ProductionTraceFormat,
    #[serde(default)]
    pub sampling: ProductionTraceSampling,
    #[serde(default = "default_trace_generation")]
    pub generation: EvalGenerationParams,
    /// Require Qwen3.5-native XML when scoring model outputs. Leave false
    /// when you want to score semantically-correct JSON emissions as pass
    /// while still surfacing the format regression in aggregate metrics.
    #[serde(default)]
    pub require_xml_format: bool,
}

fn default_trace_generation() -> EvalGenerationParams {
    EvalGenerationParams {
        temperature: 0.0,
        max_tokens: 4096,
        ..Default::default()
    }
}

impl ProductionTraceSuiteConfig {
    pub fn new(suite_name: impl Into<String>) -> Self {
        Self {
            suite_name: suite_name.into(),
            description: None,
            input_format: ProductionTraceFormat::Auto,
            sampling: ProductionTraceSampling::default(),
            generation: default_trace_generation(),
            require_xml_format: false,
        }
    }
}

/// Counters emitted after suite construction so callers can audit what the
/// random sample actually represents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProductionTraceSuiteStats {
    pub rows_seen: u64,
    pub rows_parsed: u64,
    pub eligible_tool_turns: u64,
    pub examples_generated: u64,
    pub sample_kept: u64,
    pub skipped_parse_error: u64,
    pub skipped_no_tool_call: u64,
    pub skipped_empty_prompt: u64,
    pub skipped_prompt_too_long: u64,
    pub skipped_target_too_long: u64,
    pub skipped_duplicate: u64,
    pub effective_seed: u64,
    #[serde(default)]
    pub source_format_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub target_tool_histogram: BTreeMap<String, u64>,
    #[serde(default)]
    pub parse_error_examples: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProductionTraceError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("line {line}: {message}")]
    Parse { line: usize, message: String },
    #[error("production trace suite produced no examples")]
    NoExamples,
    #[error("io: {0}")]
    Io(String),
}

impl Default for ProductionTraceFormat {
    fn default() -> Self {
        ProductionTraceFormat::Auto
    }
}

/// Stream a production trace JSONL export and build a tool-call
/// prediction eval suite.
pub fn synthesize_production_trace_suite<R: BufRead>(
    reader: R,
    config: &ProductionTraceSuiteConfig,
) -> Result<(EvalSuite, ProductionTraceSuiteStats), ProductionTraceError> {
    if config.suite_name.trim().is_empty() {
        return Err(ProductionTraceError::InvalidConfig(
            "suite_name must be non-empty".into(),
        ));
    }

    let effective_seed = config.sampling.seed.unwrap_or_else(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let mut h: u64 = 1469598103934665603;
        for b in config.suite_name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        nanos ^ h
    });
    let mut stats = ProductionTraceSuiteStats {
        effective_seed,
        ..Default::default()
    };
    let mut rng = SmallRng::seed_from_u64(effective_seed);
    let mut reservoir: Vec<EvalExample> = Vec::new();
    let mut dedupe_keys: HashSet<String> = HashSet::new();

    for (idx, line_result) in reader.lines().enumerate() {
        let line_no = idx + 1;
        stats.rows_seen += 1;
        let line = line_result.map_err(|e| ProductionTraceError::Io(format!("{e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let turn = match parse_trace_turn(trimmed, line_no, config.input_format) {
            Ok(turn) => turn,
            Err(err) => {
                stats.skipped_parse_error += 1;
                if stats.parse_error_examples.len() < 5 {
                    stats.parse_error_examples.push(format!("{err}"));
                }
                continue;
            }
        };
        stats.rows_parsed += 1;
        *stats
            .source_format_counts
            .entry(turn.source_format.to_string())
            .or_default() += 1;

        let Some((target, target_tools)) = target_tool_call_json(&turn.chosen) else {
            stats.skipped_no_tool_call += 1;
            continue;
        };
        stats.eligible_tool_turns += 1;

        if turn.prompt_messages.is_empty() {
            stats.skipped_empty_prompt += 1;
            continue;
        }
        if prompt_chars(&turn.prompt_messages) > config.sampling.max_prompt_chars {
            stats.skipped_prompt_too_long += 1;
            continue;
        }
        if target.chars().count() > config.sampling.max_target_chars {
            stats.skipped_target_too_long += 1;
            continue;
        }
        if config.sampling.dedupe {
            let key = example_hash(&turn.prompt_messages, &target);
            if !dedupe_keys.insert(key) {
                stats.skipped_duplicate += 1;
                continue;
            }
        }

        let id = stable_example_id(line_no, turn.id.as_deref(), &turn.prompt_messages, &target);
        let mut tags = vec!["production_trace".to_string(), "kind:tool_call".to_string()];
        for tool in &target_tools {
            tags.push(format!("tool:{tool}"));
            *stats.target_tool_histogram.entry(tool.clone()).or_default() += 1;
        }

        let metadata = turn.metadata(&target_tools);
        let example = EvalExample {
            id: Some(id),
            messages: turn.prompt_messages,
            target: Some(target),
            aliases: Vec::new(),
            tags,
            metadata: Some(metadata),
            scorer: None,
            generation: None,
            weight: 1.0,
            tools: if turn.tools.is_empty() {
                None
            } else {
                Some(turn.tools)
            },
        };

        stats.examples_generated += 1;
        match config.sampling.max_examples {
            Some(cap) if reservoir.len() >= cap => {
                let r = (rng.next_u64() % stats.examples_generated) as usize;
                if r < cap {
                    reservoir[r] = example;
                }
            }
            _ => reservoir.push(example),
        }
    }

    stats.sample_kept = reservoir.len() as u64;
    if reservoir.is_empty() {
        return Err(ProductionTraceError::NoExamples);
    }

    let suite_tools = hoist_identical_tools(&mut reservoir);
    let suite = EvalSuite {
        name: config.suite_name.clone(),
        description: config.description.clone().or_else(|| {
            Some(format!(
                "Random sample of production trajectory turns where the recorded assistant response emitted a tool call (seed={}).",
                stats.effective_seed
            ))
        }),
        default_scorer: Scorer::ToolCall {
            name_match: NameMatch::CaseInsensitive,
            args: ArgsScoring::Auto,
            weights: None,
            require_xml_format: config.require_xml_format,
        },
        generation: config.generation.clone(),
        system_prompt: None,
        examples: reservoir,
        schema_version: 1,
        tools: suite_tools,
    };
    Ok((suite, stats))
}

fn parse_trace_turn(
    line: &str,
    line_no: usize,
    requested: ProductionTraceFormat,
) -> Result<TraceTurn, ProductionTraceError> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| ProductionTraceError::Parse {
            line: line_no,
            message: format!("invalid JSON: {e}"),
        })?;
    match requested {
        ProductionTraceFormat::Auto => parse_auto(&value, line_no),
        ProductionTraceFormat::PromptChosenJsonl => parse_prompt_chosen(&value, line_no),
        ProductionTraceFormat::OpenAiJsonl => parse_openai_messages(&value, line_no),
        ProductionTraceFormat::AnthropicJsonl => parse_anthropic_turn(&value, line_no),
    }
}

fn parse_auto(
    value: &serde_json::Value,
    line_no: usize,
) -> Result<TraceTurn, ProductionTraceError> {
    if value.get("prompt_messages").is_some() && value.get("chosen").is_some() {
        return parse_prompt_chosen(value, line_no);
    }
    if value.get("assistant_response").is_some() {
        return parse_anthropic_turn(value, line_no);
    }
    if value.get("messages").is_some() {
        return parse_openai_messages(value, line_no);
    }
    Err(ProductionTraceError::Parse {
        line: line_no,
        message: "unrecognized production trace row shape".into(),
    })
}

fn parse_prompt_chosen(
    value: &serde_json::Value,
    line_no: usize,
) -> Result<TraceTurn, ProductionTraceError> {
    let prompt_values = value
        .get("prompt_messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProductionTraceError::Parse {
            line: line_no,
            message: "`prompt_messages` must be an array".into(),
        })?;
    let mut prompt_messages = Vec::with_capacity(prompt_values.len());
    for (i, msg) in prompt_values.iter().enumerate() {
        prompt_messages.push(value_to_chat_message(msg, line_no, i)?);
    }
    let chosen = value_to_chat_message(
        value.get("chosen").ok_or_else(|| ProductionTraceError::Parse {
            line: line_no,
            message: "missing `chosen` assistant message".into(),
        })?,
        line_no,
        prompt_values.len(),
    )?;
    let tools = normalized_tools(value.get("tools"));
    Ok(TraceTurn::from_export(
        value,
        line_no,
        "prompt_chosen_jsonl",
        prompt_messages,
        chosen,
        tools,
    ))
}

fn parse_openai_messages(
    value: &serde_json::Value,
    line_no: usize,
) -> Result<TraceTurn, ProductionTraceError> {
    let messages = value
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProductionTraceError::Parse {
            line: line_no,
            message: "`messages` must be an array".into(),
        })?;
    let mut parsed = Vec::with_capacity(messages.len());
    for (i, msg) in messages.iter().enumerate() {
        parsed.push(value_to_chat_message(msg, line_no, i)?);
    }
    let chosen_idx = parsed
        .iter()
        .rposition(|m| m.role == "assistant")
        .ok_or_else(|| ProductionTraceError::Parse {
            line: line_no,
            message: "`messages` had no assistant response to score".into(),
        })?;
    let chosen = parsed[chosen_idx].clone();
    let prompt_messages = parsed[..chosen_idx].to_vec();
    let tools = normalized_tools(value.get("tools"));
    Ok(TraceTurn::from_export(
        value,
        line_no,
        "openai_jsonl",
        prompt_messages,
        chosen,
        tools,
    ))
}

#[derive(Debug, Deserialize)]
struct RawAnthropicTurn {
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    assistant_response: Vec<AnthropicBlock>,
    #[serde(default)]
    tools: Vec<serde_json::Value>,
}

fn parse_anthropic_turn(
    value: &serde_json::Value,
    line_no: usize,
) -> Result<TraceTurn, ProductionTraceError> {
    let raw: RawAnthropicTurn =
        serde_json::from_value(value.clone()).map_err(|e| ProductionTraceError::Parse {
            line: line_no,
            message: format!("invalid anthropic trace turn: {e}"),
        })?;
    let tools = raw
        .tools
        .iter()
        .map(normalize_tool_schema)
        .collect::<Vec<_>>();
    let conv = anthropic_turn_to_sft_conversation(
        &raw.messages,
        &raw.assistant_response,
        raw.system_prompt.as_deref(),
        Some(&tools),
    );
    let Some(chosen_idx) = conv.messages.iter().rposition(|m| m.role == "assistant") else {
        return Err(ProductionTraceError::Parse {
            line: line_no,
            message: "anthropic turn had no assistant response to score".into(),
        });
    };
    let chosen = sft_to_chat_message(&conv.messages[chosen_idx]);
    let prompt_messages = conv.messages[..chosen_idx]
        .iter()
        .map(sft_to_chat_message)
        .collect();
    Ok(TraceTurn::from_export(
        value,
        line_no,
        "anthropic_jsonl",
        prompt_messages,
        chosen,
        tools,
    ))
}

fn value_to_chat_message(
    value: &serde_json::Value,
    line_no: usize,
    index: usize,
) -> Result<EvalChatMessage, ProductionTraceError> {
    let obj = value.as_object().ok_or_else(|| ProductionTraceError::Parse {
        line: line_no,
        message: format!("message {index} must be a JSON object"),
    })?;
    let role = obj
        .get("role")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProductionTraceError::Parse {
            line: line_no,
            message: format!("message {index} missing string `role`"),
        })?
        .to_string();
    let content = match obj.get("content") {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    };
    let tool_calls = obj
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .map(|arr| arr.to_vec())
        .filter(|arr| !arr.is_empty());
    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let tool_call_id = obj
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Ok(EvalChatMessage {
        role,
        content,
        tool_calls,
        name,
        tool_call_id,
    })
}

fn sft_to_chat_message(m: &crate::synthesis::SftMessage) -> EvalChatMessage {
    EvalChatMessage {
        role: m.role.clone(),
        content: m.content.clone(),
        tool_calls: m.tool_calls.clone(),
        name: m.name.clone(),
        tool_call_id: m.tool_call_id.clone(),
    }
}

fn target_tool_call_json(chosen: &EvalChatMessage) -> Option<(String, Vec<String>)> {
    let calls = if let Some(tool_calls) = chosen.tool_calls.as_ref().filter(|tc| !tc.is_empty()) {
        let raw = serde_json::json!({ "tool_calls": tool_calls }).to_string();
        extract_tool_calls(&raw)
    } else {
        extract_tool_calls(&chosen.content)
    };
    if calls.is_empty() {
        return None;
    }
    let tool_names = calls.iter().map(|c| c.name.clone()).collect::<Vec<_>>();
    Some((canonical_target_json(&calls), tool_names))
}

fn canonical_target_json(calls: &[ParsedToolCall]) -> String {
    let arr = calls
        .iter()
        .map(ParsedToolCall::to_canonical_json)
        .collect::<Vec<_>>();
    serde_json::to_string(&serde_json::json!({ "tool_calls": arr }))
        .unwrap_or_else(|_| "{\"tool_calls\":[]}".to_string())
}

fn prompt_chars(messages: &[EvalChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| {
            m.role.chars().count()
                + m.content.chars().count()
                + m.tool_calls
                    .as_ref()
                    .map(|tc| serde_json::to_string(tc).unwrap_or_default().chars().count())
                    .unwrap_or(0)
                + m.name.as_ref().map(|s| s.chars().count()).unwrap_or(0)
                + m.tool_call_id
                    .as_ref()
                    .map(|s| s.chars().count())
                    .unwrap_or(0)
        })
        .sum()
}

fn example_hash(messages: &[EvalChatMessage], target: &str) -> String {
    let mut hasher = Sha256::new();
    let bytes = serde_json::to_vec(messages).unwrap_or_default();
    hasher.update(bytes);
    hasher.update([0]);
    hasher.update(target.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn stable_example_id(
    line_no: usize,
    source_id: Option<&str>,
    messages: &[EvalChatMessage],
    target: &str,
) -> String {
    let hash = example_hash(messages, target);
    let suffix = &hash[..12.min(hash.len())];
    match source_id.filter(|s| !s.trim().is_empty()) {
        Some(id) => format!("trace-{line_no}-{id}-{suffix}"),
        None => format!("trace-{line_no}-{suffix}"),
    }
}

fn hoist_identical_tools(examples: &mut [EvalExample]) -> Option<Vec<serde_json::Value>> {
    let mut witness: Option<Vec<serde_json::Value>> = None;
    for ex in examples.iter() {
        let Some(tools) = ex.tools.as_ref().filter(|t| !t.is_empty()) else {
            return None;
        };
        match witness.as_ref() {
            None => witness = Some(tools.clone()),
            Some(prev) if prev == tools => {}
            Some(_) => return None,
        }
    }
    if let Some(tools) = witness.as_ref() {
        for ex in examples {
            ex.tools = None;
        }
        Some(tools.clone())
    } else {
        None
    }
}

fn normalized_tools(value: Option<&serde_json::Value>) -> Vec<serde_json::Value> {
    value
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(normalize_tool_schema).collect())
        .unwrap_or_default()
}

fn normalize_tool_schema(tool: &serde_json::Value) -> serde_json::Value {
    if tool.get("function").is_some() {
        return tool.clone();
    }
    let Some(obj) = tool.as_object() else {
        return tool.clone();
    };
    let Some(name) = obj.get("name").and_then(|v| v.as_str()) else {
        return tool.clone();
    };
    let description = obj
        .get("description")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::String(String::new()));
    let parameters = obj
        .get("input_schema")
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    })
}

#[derive(Debug)]
struct TraceTurn {
    id: Option<String>,
    session_id: Option<String>,
    timestamp: Option<String>,
    model: Option<String>,
    split: Option<String>,
    source_line: usize,
    source_format: &'static str,
    prompt_messages: Vec<EvalChatMessage>,
    chosen: EvalChatMessage,
    tools: Vec<serde_json::Value>,
    source_metadata: serde_json::Map<String, serde_json::Value>,
}

impl TraceTurn {
    fn from_export(
        value: &serde_json::Value,
        line_no: usize,
        source_format: &'static str,
        prompt_messages: Vec<EvalChatMessage>,
        chosen: EvalChatMessage,
        tools: Vec<serde_json::Value>,
    ) -> Self {
        let mut source_metadata = serde_json::Map::new();
        if let Some(meta) = value.get("metadata").and_then(|v| v.as_object()) {
            for (k, v) in meta {
                source_metadata.insert(k.clone(), v.clone());
            }
        }
        Self {
            id: string_field(value, "id"),
            session_id: string_field(value, "session_id"),
            timestamp: string_field(value, "timestamp")
                .or_else(|| source_metadata.get("timestamp").and_then(value_to_string)),
            model: string_field(value, "model"),
            split: string_field(value, "split"),
            source_line: line_no,
            source_format,
            prompt_messages,
            chosen,
            tools,
            source_metadata,
        }
    }

    fn metadata(&self, target_tools: &[String]) -> serde_json::Value {
        let mut obj = self.source_metadata.clone();
        obj.insert("source_line".into(), serde_json::json!(self.source_line));
        obj.insert("source_format".into(), serde_json::json!(self.source_format));
        obj.insert("target_tools".into(), serde_json::json!(target_tools));
        if let Some(id) = self.id.as_deref() {
            obj.insert("turn_id".into(), serde_json::json!(id));
        }
        if let Some(session_id) = self.session_id.as_deref() {
            obj.insert("session_id".into(), serde_json::json!(session_id));
        }
        if let Some(timestamp) = self.timestamp.as_deref() {
            obj.insert("timestamp".into(), serde_json::json!(timestamp));
        }
        if let Some(model) = self.model.as_deref() {
            obj.insert("production_model".into(), serde_json::json!(model));
        }
        if let Some(split) = self.split.as_deref() {
            obj.insert("split".into(), serde_json::json!(split));
        }
        serde_json::Value::Object(obj)
    }
}

fn string_field(value: &serde_json::Value, key: &str) -> Option<String> {
    value.get(key).and_then(value_to_string)
}

fn value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3::extract_first_tool_call;
    use crate::scorers::{NoopJudgeRunner, score_completion};
    use crate::EvalOutcomeKind;

    fn build(input: &str, max_examples: Option<usize>) -> (EvalSuite, ProductionTraceSuiteStats) {
        let mut cfg = ProductionTraceSuiteConfig::new("prod-tool-smoke");
        cfg.sampling.seed = Some(123);
        cfg.sampling.max_examples = max_examples;
        synthesize_production_trace_suite(std::io::Cursor::new(input), &cfg).unwrap()
    }

    #[test]
    fn prompt_chosen_jsonl_preserves_prompt_and_tool_catalogue() {
        let line = serde_json::json!({
            "id": "turn-1",
            "session_id": "sess-1",
            "model": "claude-prod",
            "prompt_messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Read /tmp/a"}
            ],
            "chosen": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{\"file_path\":\"/tmp/a\"}"}
                }]
            },
            "tools": [{
                "type": "function",
                "function": {"name": "Read", "parameters": {"type":"object","properties":{"file_path":{"type":"string"}}}}
            }],
            "metadata": {"timestamp": "2026-05-01T00:00:00Z"}
        });
        let input = format!("{line}\n");
        let (suite, stats) = build(&input, Some(10));
        assert_eq!(stats.rows_seen, 1);
        assert_eq!(stats.eligible_tool_turns, 1);
        assert_eq!(suite.examples.len(), 1);
        assert_eq!(suite.examples[0].messages.len(), 2);
        assert!(suite.tools.as_ref().map_or(false, |t| t.len() == 1));
        let target = suite.examples[0].target.as_deref().unwrap();
        let call = extract_first_tool_call(target).unwrap();
        assert_eq!(call.name, "Read");
        assert_eq!(
            call.arguments.get("file_path"),
            Some(&serde_json::json!("/tmp/a"))
        );
        assert_eq!(
            suite.examples[0].metadata.as_ref().unwrap()["session_id"],
            serde_json::json!("sess-1")
        );
    }

    #[test]
    fn openai_jsonl_uses_final_assistant_as_target() {
        let line = serde_json::json!({
            "id": "turn-2",
            "messages": [
                {"role":"system", "content":"s"},
                {"role":"user", "content":"run ls"},
                {"role":"assistant", "content":"", "tool_calls":[{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"Bash", "arguments":"{\"command\":\"ls -la\"}"}
                }]}
            ],
            "tools": [{"type":"function","function":{"name":"Bash","parameters":{"type":"object"}}}]
        });
        let input = format!("{line}\n");
        let (suite, stats) = build(&input, None);
        assert_eq!(stats.skipped_no_tool_call, 0);
        assert_eq!(suite.examples[0].messages.len(), 2);
        assert_eq!(suite.examples[0].messages.last().unwrap().role, "user");
        assert!(
            suite.examples[0]
                .target
                .as_deref()
                .unwrap()
                .contains("\"Bash\"")
        );
    }

    #[test]
    fn raw_anthropic_jsonl_converts_response_blocks() {
        let line = serde_json::json!({
            "id": "turn-3",
            "system_prompt": "You are a coding agent.",
            "messages": [{"role":"user", "content":"edit file"}],
            "assistant_response": [{
                "type":"tool_use",
                "id":"toolu_3",
                "name":"Edit",
                "input":{"file_path":"/tmp/a", "old_string":"x", "new_string":"y"}
            }],
            "tools": [{
                "name":"Edit",
                "description":"Edit file",
                "input_schema":{"type":"object","properties":{"file_path":{"type":"string"}}}
            }]
        });
        let input = format!("{line}\n");
        let (suite, _) = build(&input, None);
        assert_eq!(suite.examples[0].messages[0].role, "system");
        assert_eq!(suite.examples[0].messages[1].role, "user");
        assert_eq!(
            suite.tools.as_ref().unwrap()[0]["function"]["name"],
            serde_json::json!("Edit")
        );
        let target = suite.examples[0].target.as_deref().unwrap();
        assert!(target.contains("\"old_string\":\"x\""));
    }

    #[test]
    fn generated_suite_scores_qwen_xml_semantically() {
        let line = serde_json::json!({
            "prompt_messages": [{"role":"user", "content":"run pwd"}],
            "chosen": {"role":"assistant", "tool_calls":[{
                "type":"function",
                "function":{"name":"Bash", "arguments":"{\"command\":\"pwd\"}"}
            }]},
            "tools": [{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]
        });
        let (suite, _) = build(&format!("{line}\n"), None);
        let model_output = "<tool_call>\n<function=Bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>";
        let outcome = score_completion(
            &suite.default_scorer,
            &suite.examples[0],
            model_output,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert_eq!(outcome.kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn non_tool_rows_are_skipped() {
        let line = serde_json::json!({
            "prompt_messages": [{"role":"user", "content":"hi"}],
            "chosen": {"role":"assistant", "content":"hello"}
        });
        let mut cfg = ProductionTraceSuiteConfig::new("empty");
        cfg.sampling.seed = Some(1);
        let err = synthesize_production_trace_suite(
            std::io::Cursor::new(format!("{line}\n")),
            &cfg,
        )
        .unwrap_err();
        assert!(matches!(err, ProductionTraceError::NoExamples));
    }

    #[test]
    fn seeded_reservoir_is_stable() {
        let mut rows = String::new();
        for i in 0..20 {
            let line = serde_json::json!({
                "prompt_messages": [{"role":"user", "content": format!("q{i}")}],
                "chosen": {"role":"assistant", "tool_calls":[{
                    "type":"function",
                    "function":{"name":"Bash", "arguments": serde_json::json!({"command": format!("echo {i}")}).to_string()}
                }]}
            });
            rows.push_str(&line.to_string());
            rows.push('\n');
        }
        let (a, _) = build(&rows, Some(5));
        let (b, _) = build(&rows, Some(5));
        let a_ids: Vec<_> = a.examples.iter().map(|e| e.id.clone()).collect();
        let b_ids: Vec<_> = b.examples.iter().map(|e| e.id.clone()).collect();
        assert_eq!(a_ids, b_ids);
    }

    #[test]
    fn dedupe_is_exact_prompt_target_when_enabled() {
        let row = serde_json::json!({
            "prompt_messages": [{"role":"user", "content":"same"}],
            "chosen": {"role":"assistant", "tool_calls":[{
                "type":"function",
                "function":{"name":"Bash", "arguments":"{\"command\":\"pwd\"}"}
            }]}
        });
        let input = format!("{row}\n{row}\n");
        let mut cfg = ProductionTraceSuiteConfig::new("dedupe");
        cfg.sampling.seed = Some(1);
        cfg.sampling.max_examples = None;
        cfg.sampling.dedupe = true;
        let (suite, stats) =
            synthesize_production_trace_suite(std::io::Cursor::new(input), &cfg).unwrap();
        assert_eq!(suite.examples.len(), 1);
        assert_eq!(stats.skipped_duplicate, 1);
    }
}
