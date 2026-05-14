//! Pipeline that turns SFT/GRPO datasets into eval suites.
//!
//! The mental model: an SFT dataset is a list of *trajectories* (chat
//! conversations or GRPO scored groups). Each trajectory can be decomposed
//! into one or more eval *examples* by a `SynthesisStrategy`. The strategies
//! shipped here cover the common cases:
//!
//! - `FinalAssistant`: the prompt is "history up to and including the last
//!   user message", the target is the very last assistant message. Tests
//!   "did the model produce the same final answer?" — works great for
//!   single-shot Q&A and for the *final* answer in a multi-step agent run.
//! - `FirstAssistantTurn`: prompt = system + first user message; target =
//!   the first assistant turn. Cheap and fast — tests the model's
//!   immediate response, not its multi-step planning.
//! - `EveryAssistantTurn`: emits one example per assistant turn in the
//!   trajectory (prompt = history up to that turn). Useful for "next
//!   action prediction" evals over agentic data.
//! - `ToolCallPredict`: a specialization of `EveryAssistantTurn` that
//!   only keeps assistant turns that emit tool calls. The target is the
//!   tool call JSON, the scorer is `json_validity` with the required
//!   structural paths populated automatically.
//!
//! All strategies share a `Sampling` filter (max_examples, max_prompt_chars,
//! random seed) and an auto-detect scorer that picks a sensible default
//! based on each example's target shape.

use std::collections::HashSet;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_core::Rng as _;
use serde::{Deserialize, Serialize};

use crate::scorers::{NumericTolerance, Scorer, contains::ContainsMode};
use crate::suite::{EvalChatMessage, EvalExample, EvalGenerationParams, EvalSuite};

/// One SFT chat message — mirrors the shape of `kiln_train::ChatMessage` to
/// keep this crate independent of the training crate. The loader accepts:
/// - The plain `{role, content}` shape
/// - The OpenAI agentic shape with `tool_calls: [{"function": {"name", "arguments"}}, …]`
///   on assistant turns, and `name` / `tool_call_id` on tool replies
///
/// `tool_calls` is preserved so the synthesis pipeline can build proper
/// tool-call evals (target = the tool call JSON, scored by `ToolCall`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// OpenAI-style assistant tool calls. Each entry is typically
    /// `{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Function name on tool-role messages (some templates branch on it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Identifies which assistant tool call this `tool`-role message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// One SFT example (matches `kiln_train::SftExample` plus optional
/// trajectory metadata the loader doesn't touch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftConversation {
    pub messages: Vec<SftMessage>,
    /// Any unrecognized fields land here — preserved through serialization
    /// so dataset round-trips don't lose anything.
    #[serde(flatten, default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Sampling/filtering knobs applied across every strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sampling {
    /// Cap on the number of examples in the output suite. `None` means
    /// "keep every example the strategy produces" — beware multi-turn
    /// strategies fan out N examples per trajectory.
    #[serde(default)]
    pub max_examples: Option<usize>,
    /// Skip trajectories whose serialized prompt exceeds this many *chars*
    /// (rough proxy for token count — cheap to compute without tokenizing).
    /// Defaults to 32 KiB which fits comfortably in the 32k-token Qwen
    /// context window after chat-template overhead.
    #[serde(default = "default_max_prompt_chars")]
    pub max_prompt_chars: usize,
    /// Skip trajectories whose target exceeds this many *chars*. Long
    /// targets make exact-match evals nearly impossible to pass; use the
    /// LLM-judge scorer for those.
    #[serde(default = "default_max_target_chars")]
    pub max_target_chars: usize,
    /// Optional RNG seed. When set, sampling is deterministic; otherwise a
    /// random seed is picked at synthesis time and emitted in the result.
    #[serde(default)]
    pub seed: Option<u64>,
    /// When true, examples are deduplicated by `(last_user_message, target)`
    /// before sampling. Keeps the output suite small and free of repeats
    /// when the dataset itself was deduped per-trajectory but not per-step.
    #[serde(default = "default_dedupe")]
    pub dedupe: bool,
}

fn default_max_prompt_chars() -> usize {
    32 * 1024
}
fn default_max_target_chars() -> usize {
    4 * 1024
}
fn default_dedupe() -> bool {
    true
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            max_examples: Some(100),
            max_prompt_chars: default_max_prompt_chars(),
            max_target_chars: default_max_target_chars(),
            seed: None,
            dedupe: default_dedupe(),
        }
    }
}

/// Decomposition strategy. Each variant produces zero or more
/// `EvalExample`s from a single `SftConversation`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SynthesisStrategy {
    /// Prompt = history up to the last user message; target = the very
    /// last assistant message in the conversation. Default — works well
    /// for single-shot Q&A AND as an end-to-end check for agent trajectories.
    FinalAssistant,
    /// Prompt = system + first user; target = first assistant turn.
    FirstAssistantTurn,
    /// Emits one example per assistant turn in the trajectory.
    EveryAssistantTurn,
    /// Same as `EveryAssistantTurn` but filters to turns that emit a tool
    /// call (JSON-shaped or fenced ```tool_call blocks).
    ToolCallPredict,
    /// Tests the model's next-step response *after* a tool returns a
    /// result. One example per `(assistant tool_call → tool result → next
    /// assistant)` triple. Prompt includes the preceding tool exchange in
    /// full; target is the next assistant turn (prose, another tool call,
    /// or a final answer). Best Qwen3.5 agentic-reasoning probe.
    ToolResponseFollowup,
    /// Final-answer eval for agent runs: prompt = trajectory up through
    /// the last tool result, target = the closing assistant turn (which
    /// in Qwen3.5 is the "I've finished, here's the result" prose).
    /// Filters out trajectories without any tool exchanges so the test
    /// stays focused on agent endings.
    EndOfTrajectoryAnswer,
}

impl Default for SynthesisStrategy {
    fn default() -> Self {
        SynthesisStrategy::FinalAssistant
    }
}

/// Default scorer selection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScorerChoice {
    /// Pick a scorer per example based on the target's shape:
    /// numeric → numeric_tolerance, JSON-shaped → json_validity, MCQ-shaped
    /// (single letter A-D) → multiple_choice, otherwise → contains with
    /// extracted key phrases. The suite-level default scorer falls back to
    /// `exact_match` for safety.
    AutoDetect,
    /// Use a single fixed scorer for every example.
    Fixed(Scorer),
    /// Use the LLM-as-judge scorer (the executor wires the judge later).
    Judge {
        #[serde(default)]
        judge_adapter: Option<String>,
    },
}

impl Default for ScorerChoice {
    fn default() -> Self {
        ScorerChoice::AutoDetect
    }
}

/// Synthesis configuration body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Name for the resulting `EvalSuite` (used as the on-disk directory
    /// name in the suite registry).
    pub suite_name: String,
    /// Free-form description carried into `EvalSuite::description`.
    #[serde(default)]
    pub description: Option<String>,
    /// How to decompose each trajectory into examples.
    #[serde(default)]
    pub strategy: SynthesisStrategy,
    /// What scorer to use for the generated examples.
    #[serde(default)]
    pub scorer: ScorerChoice,
    /// Generation params on the resulting suite. Defaults to greedy.
    #[serde(default)]
    pub generation: EvalGenerationParams,
    /// Sampling/filtering knobs.
    #[serde(default)]
    pub sampling: Sampling,
    /// Optional system prompt injected into the resulting suite (separate
    /// from any system message inherited from the trajectory).
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// When true, the synthesized examples drop the system message that
    /// came with the trajectory. Useful when the source dataset's system
    /// prompt is huge (multi-KB tool catalogue) and you want a clean eval.
    #[serde(default)]
    pub strip_system_prompt: bool,
}

impl SynthesisConfig {
    /// Sensible defaults for a quick synthesis. Caller still has to set
    /// `suite_name`.
    pub fn new(suite_name: impl Into<String>) -> Self {
        Self {
            suite_name: suite_name.into(),
            description: None,
            strategy: SynthesisStrategy::default(),
            scorer: ScorerChoice::default(),
            generation: EvalGenerationParams::default(),
            sampling: Sampling::default(),
            system_prompt: None,
            strip_system_prompt: false,
        }
    }
}

/// Stats reported back to the caller (and surfaced in the UI) after a
/// synthesis run. Helps the user understand what got filtered out and why.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynthesisStats {
    pub trajectories_seen: u64,
    pub trajectories_used: u64,
    pub examples_generated: u64,
    pub skipped_no_target: u64,
    pub skipped_prompt_too_long: u64,
    pub skipped_target_too_long: u64,
    pub skipped_duplicate: u64,
    pub skipped_strategy_match: u64,
    pub sample_kept: u64,
    pub effective_seed: u64,
    /// Histogram of which auto-detected scorer was assigned per example
    /// (only populated under `ScorerChoice::AutoDetect`).
    #[serde(default)]
    pub auto_scorer_histogram: std::collections::BTreeMap<String, u32>,
}

/// Auto-detect a per-example scorer based on the shape of `target`.
///
/// The classifier is intentionally conservative: it returns a *broad*
/// scorer (lenient `contains` with extracted key phrases) when nothing
/// matches, so even messy targets still produce a runnable eval.
pub fn auto_detect_scorer(target: &str) -> Scorer {
    let t = target.trim();
    if t.is_empty() {
        return Scorer::ExactMatch {
            case_sensitive: false,
            strip_whitespace: true,
        };
    }

    // 1. Multiple choice — bare single letter A-Z, or A-D in parens.
    if is_mcq_label(t) {
        return Scorer::MultipleChoice {
            choices: vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
        };
    }

    // 2. Numeric — target is a single number (or fraction). Use a tight
    //    tolerance for clean integer targets and a loose rtol for floats.
    if let Some(tol) = numeric_tolerance_for(t) {
        return Scorer::NumericTolerance(tol);
    }

    // 3. Tool-call shaped — canonical `{"tool_calls":[...]}` or anything
    //    where a tool-call extractor lands cleanly. ToolCall scorer with
    //    auto args scoring covers both name + argument quality.
    if looks_like_tool_call_target(t) {
        return Scorer::ToolCall {
            name_match: crate::scorers::NameMatch::CaseInsensitive,
            args: crate::scorers::ArgsScoring::Auto,
            weights: None,
        };
    }

    // 4. Code-block — fenced code or fence-less program-shaped content.
    if let Some(lang) = detect_code_language(t) {
        return Scorer::Code {
            language: Some(lang),
            style: crate::scorers::CodeStyle::TokenSimilarity {
                min_jaccard: crate::scorers::default_min_jaccard(),
            },
        };
    }

    // 5. JSON-shaped (non-tool-call) — leading { or [ that parses cleanly.
    if (t.starts_with('{') || t.starts_with('[')) && serde_json::from_str::<serde_json::Value>(t).is_ok() {
        return Scorer::JsonValidity {
            require_object: t.starts_with('{'),
            required_paths: Vec::new(),
        };
    }

    // 6. Short / single-line — exact match (case-insensitive, trim).
    let line_count = t.lines().count();
    if line_count <= 1 && t.chars().count() <= 80 {
        return Scorer::ExactMatch {
            case_sensitive: false,
            strip_whitespace: true,
        };
    }

    // 7. Long-form — contains with extracted key phrases (the 3 most
    //    distinctive 3-word n-grams). Lets the model paraphrase but still
    //    requires the substantive content.
    let phrases = extract_key_phrases(t, 3);
    if phrases.is_empty() {
        return Scorer::ExactMatch {
            case_sensitive: false,
            strip_whitespace: true,
        };
    }
    Scorer::Contains {
        phrases,
        mode: ContainsMode::Any,
        case_sensitive: false,
    }
}

fn looks_like_tool_call_target(s: &str) -> bool {
    let trimmed = s.trim();
    if !(trimmed.starts_with('{') || trimmed.starts_with("```tool_call") || trimmed.starts_with("<tool_call>")) {
        return false;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(obj) = v.as_object() {
            if obj.contains_key("tool_calls") {
                return true;
            }
            if obj.contains_key("name") && obj.contains_key("arguments") {
                return true;
            }
            if obj.contains_key("function") || obj.contains_key("tool_call") {
                return true;
            }
        }
    }
    trimmed.starts_with("```tool_call") || trimmed.starts_with("<tool_call>")
}

fn detect_code_language(s: &str) -> Option<String> {
    // Fenced block detection: ```python\n ... \n```
    if let Some(after) = s.trim_start().strip_prefix("```") {
        let (lang, rest) = after.split_once('\n').unwrap_or((after, ""));
        let lang = lang.trim();
        if !lang.is_empty() && rest.contains("```") {
            return Some(match lang.to_ascii_lowercase().as_str() {
                "py" => "python".into(),
                "rs" => "rust".into(),
                "ts" | "tsx" => "typescript".into(),
                "js" | "mjs" | "cjs" => "javascript".into(),
                other => other.into(),
            });
        }
    }
    None
}

fn is_mcq_label(s: &str) -> bool {
    let trimmed = s.trim_matches(|c: char| c == '(' || c == ')' || c == '.' || c == ' ');
    if trimmed.chars().count() != 1 {
        return false;
    }
    let c = trimmed.chars().next().unwrap();
    c.is_ascii_uppercase()
}

fn numeric_tolerance_for(s: &str) -> Option<NumericTolerance> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace() && *c != ',').collect();
    let parsed: f64 = cleaned.parse().ok()?;
    let integer_only = parsed.fract() == 0.0;
    Some(NumericTolerance {
        atol: if integer_only { 0.0 } else { 1e-6 },
        rtol: if integer_only { 0.0 } else { 1e-3 },
        integer_only,
    })
}

/// Extract up to `k` distinctive multi-word phrases from a long-form target.
/// The heuristic favors *content* tokens (longer words, not stopwords) so
/// the resulting scorer doesn't pass on every fluffy completion.
fn extract_key_phrases(text: &str, k: usize) -> Vec<String> {
    let stop: HashSet<&str> = [
        "the", "and", "for", "with", "that", "this", "from", "you", "are", "not", "but", "have",
        "has", "had", "was", "were", "will", "would", "could", "should", "their", "your", "they",
        "them", "than", "then", "into", "over", "also", "some", "such", "such", "more", "most",
        "much", "many", "very", "just", "like", "what", "which", "when", "where", "while",
    ]
    .into_iter()
    .collect();
    let mut phrases: Vec<String> = Vec::new();
    for line in text.lines() {
        let words: Vec<&str> = line
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
            })
            .filter(|w| {
                w.chars().count() >= 4 && !stop.contains(&w.to_ascii_lowercase().as_str())
            })
            .collect();
        for window in words.windows(3) {
            let phrase = window.join(" ").to_lowercase();
            if !phrases.iter().any(|p| p == &phrase) {
                phrases.push(phrase);
                if phrases.len() >= k {
                    return phrases;
                }
            }
        }
    }
    phrases
}

/// Run synthesis. `conversations` is an iterator of (1-indexed line number,
/// parsed conversation) — the line number is plumbed into stats / errors.
/// Returns `(EvalSuite, SynthesisStats)`.
///
/// Streaming-friendly: we walk the iterator once and apply reservoir
/// sampling for `max_examples` when more examples come through than the
/// requested cap. That way we never have to hold every candidate example in
/// memory before deciding which to keep.
pub fn synthesize_suite<I>(
    conversations: I,
    config: &SynthesisConfig,
) -> Result<(EvalSuite, SynthesisStats), SynthesisError>
where
    I: IntoIterator<Item = SftConversation>,
{
    if config.suite_name.trim().is_empty() {
        return Err(SynthesisError::InvalidConfig(
            "suite_name must be non-empty".into(),
        ));
    }

    let mut stats = SynthesisStats::default();
    let effective_seed = config.sampling.seed.unwrap_or_else(|| {
        // Deterministic-ish: nanos since epoch xor'd with the suite name's hash.
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
    stats.effective_seed = effective_seed;

    let mut rng = SmallRng::seed_from_u64(effective_seed);
    let cap = config.sampling.max_examples;
    let mut reservoir: Vec<EvalExample> = Vec::new();
    let mut dedup_keys: HashSet<String> = HashSet::new();

    for conv in conversations.into_iter() {
        stats.trajectories_seen += 1;
        let candidates = match decompose(&conv, config) {
            Ok(c) => c,
            Err(e) => match e {
                SynthesisError::NoApplicableTurns => {
                    stats.skipped_strategy_match += 1;
                    continue;
                }
                other => return Err(other),
            },
        };

        if candidates.is_empty() {
            stats.skipped_strategy_match += 1;
            continue;
        }
        let mut used = false;

        for mut cand in candidates {
            // Filter checks.
            let target = match cand.target.as_deref() {
                Some(t) if !t.trim().is_empty() => t,
                _ => {
                    stats.skipped_no_target += 1;
                    continue;
                }
            };
            if target.chars().count() > config.sampling.max_target_chars {
                stats.skipped_target_too_long += 1;
                continue;
            }
            let prompt_chars: usize = cand
                .messages
                .iter()
                .map(|m| m.content.chars().count() + m.role.chars().count() + 4)
                .sum();
            if prompt_chars > config.sampling.max_prompt_chars {
                stats.skipped_prompt_too_long += 1;
                continue;
            }
            if config.sampling.dedupe {
                let key = format!(
                    "{}|{}",
                    cand.messages
                        .iter()
                        .rev()
                        .find(|m| m.role == "user")
                        .map(|m| m.content.as_str())
                        .unwrap_or(""),
                    target
                );
                if !dedup_keys.insert(key) {
                    stats.skipped_duplicate += 1;
                    continue;
                }
            }
            used = true;
            stats.examples_generated += 1;
            // For AutoDetect, attach the per-example scorer here so the
            // histogram is derived from the same value (no double-call to
            // auto_detect_scorer) and so candidates that get evicted from
            // the reservoir don't bias the histogram with leftover work.
            if matches!(config.scorer, ScorerChoice::AutoDetect) {
                let scorer = auto_detect_scorer(target);
                let label = scorer.kind_label();
                *stats
                    .auto_scorer_histogram
                    .entry(label.to_string())
                    .or_default() += 1;
                cand.scorer = Some(scorer);
            } else if let Some(label) = scorer_kind_for_stats(&config.scorer, target) {
                *stats
                    .auto_scorer_histogram
                    .entry(label.to_string())
                    .or_default() += 1;
            }

            // Reservoir sampling — keep the candidate if we haven't filled
            // the reservoir yet, else replace a random slot with probability
            // `cap / examples_generated`.
            match cap {
                Some(n) if reservoir.len() >= n => {
                    let r = (rng.next_u64() % stats.examples_generated) as usize;
                    if r < n {
                        reservoir[r] = cand;
                    }
                }
                _ => {
                    reservoir.push(cand);
                }
            }
        }
        if used {
            stats.trajectories_used += 1;
        }
    }

    stats.sample_kept = reservoir.len() as u64;
    if reservoir.is_empty() {
        return Err(SynthesisError::NoExamples);
    }

    let default_scorer = match &config.scorer {
        ScorerChoice::Fixed(s) => s.clone(),
        ScorerChoice::Judge { judge_adapter } => Scorer::LlmJudge {
            judge_adapter: judge_adapter.clone(),
            template: crate::scorers::llm_judge::default_judge_template(),
            score_regex: crate::scorers::llm_judge::default_judge_regex(),
        },
        ScorerChoice::AutoDetect => Scorer::ExactMatch {
            case_sensitive: false,
            strip_whitespace: true,
        },
    };

    let suite = EvalSuite {
        name: config.suite_name.clone(),
        description: config.description.clone(),
        default_scorer,
        generation: config.generation.clone(),
        system_prompt: config.system_prompt.clone(),
        examples: reservoir,
        schema_version: 1,
        tools: None,
    };
    Ok((suite, stats))
}

/// Decompose one trajectory into zero or more candidate `EvalExample`s.
fn decompose(
    conv: &SftConversation,
    config: &SynthesisConfig,
) -> Result<Vec<EvalExample>, SynthesisError> {
    let messages = filter_messages(&conv.messages, config.strip_system_prompt);
    if messages.is_empty() {
        return Err(SynthesisError::NoApplicableTurns);
    }
    match config.strategy {
        SynthesisStrategy::FinalAssistant => {
            // Find the last assistant turn that yields a usable target —
            // either a structured `tool_calls` payload, parseable XML, or
            // non-empty content. This lets agentic trajectories that store
            // tool calls in the raw assistant content (no `tool_calls`
            // field) canonicalize cleanly into JSON targets.
            let last_assistant_idx = messages
                .iter()
                .enumerate()
                .rev()
                .find(|(_, m)| m.role == "assistant" && assistant_target(m).is_some())
                .map(|(i, _)| i)
                .ok_or(SynthesisError::NoApplicableTurns)?;
            let prompt = &messages[..last_assistant_idx];
            if prompt.is_empty() {
                return Err(SynthesisError::NoApplicableTurns);
            }
            // Trim any trailing tool messages so the prompt cleanly ends on
            // an assistant or user turn — the chat template renders better.
            let prompt = trim_trailing_role(prompt, "tool");
            if prompt.is_empty() {
                return Err(SynthesisError::NoApplicableTurns);
            }
            let (target_text, target_kind) = assistant_target(&messages[last_assistant_idx])
                .ok_or(SynthesisError::NoApplicableTurns)?;
            Ok(vec![EvalExample {
                id: None,
                messages: to_chat_messages(prompt),
                target: Some(target_text),
                aliases: Vec::new(),
                tags: vec![
                    "synth:final_assistant".into(),
                    target_kind.tag().to_string(),
                ],
                metadata: trajectory_metadata_with_kind(conv, &target_kind),
                scorer: None,
                generation: None,
                weight: 1.0,
                tools: None,
            }])
        }
        SynthesisStrategy::FirstAssistantTurn => {
            let first_user_idx = messages
                .iter()
                .position(|m| m.role == "user")
                .ok_or(SynthesisError::NoApplicableTurns)?;
            let first_assistant_idx = messages
                .iter()
                .enumerate()
                .skip(first_user_idx)
                .find(|(_, m)| m.role == "assistant" && assistant_target(m).is_some())
                .map(|(i, _)| i)
                .ok_or(SynthesisError::NoApplicableTurns)?;
            let prompt = &messages[..first_assistant_idx];
            let prompt = trim_trailing_role(prompt, "tool");
            if prompt.is_empty() {
                return Err(SynthesisError::NoApplicableTurns);
            }
            let (target_text, target_kind) = assistant_target(&messages[first_assistant_idx])
                .ok_or(SynthesisError::NoApplicableTurns)?;
            Ok(vec![EvalExample {
                id: None,
                messages: to_chat_messages(prompt),
                target: Some(target_text),
                aliases: Vec::new(),
                tags: vec![
                    "synth:first_assistant".into(),
                    target_kind.tag().to_string(),
                ],
                metadata: trajectory_metadata_with_kind(conv, &target_kind),
                scorer: None,
                generation: None,
                weight: 1.0,
                tools: None,
            }])
        }
        SynthesisStrategy::EveryAssistantTurn => Ok(every_assistant_turn(&messages, conv, false)),
        SynthesisStrategy::ToolCallPredict => Ok(every_assistant_turn(&messages, conv, true)),
        SynthesisStrategy::ToolResponseFollowup => Ok(tool_response_followup(&messages, conv)),
        SynthesisStrategy::EndOfTrajectoryAnswer => Ok(end_of_trajectory_answer(&messages, conv)),
    }
}

/// Build one example per `(assistant_with_tool_calls, tool_result(s), next_assistant)`
/// triple. The eval prompt ends on a `tool`-role message, so the model is
/// asked: "given this tool result, what do you do next?".
fn tool_response_followup(
    messages: &[SftMessage],
    conv: &SftConversation,
) -> Vec<EvalExample> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < messages.len() {
        // Look for assistant with tool_calls at `i`.
        let m = &messages[i];
        if m.role != "assistant"
            || !m
                .tool_calls
                .as_ref()
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        {
            i += 1;
            continue;
        }
        // Walk through following `tool` messages.
        let tool_start = i + 1;
        let mut tool_end = tool_start;
        while tool_end < messages.len() && messages[tool_end].role == "tool" {
            tool_end += 1;
        }
        if tool_end == tool_start {
            i += 1;
            continue;
        }
        // Need a subsequent assistant turn to be the target.
        if tool_end >= messages.len() {
            break;
        }
        let next = &messages[tool_end];
        if next.role != "assistant" {
            i = tool_end;
            continue;
        }
        let (target_text, target_kind) = match assistant_target(next) {
            Some(pair) => pair,
            None => {
                i = tool_end;
                continue;
            }
        };
        // Prompt = everything up to and including the tool responses.
        let prompt = &messages[..tool_end];
        if prompt.is_empty() {
            i = tool_end;
            continue;
        }
        out.push(EvalExample {
            id: None,
            messages: to_chat_messages(prompt),
            target: Some(target_text),
            aliases: Vec::new(),
            tags: vec![
                "synth:tool_response_followup".into(),
                format!("synth:turn_{tool_end}"),
                target_kind.tag().to_string(),
            ],
            metadata: trajectory_metadata_with_kind(conv, &target_kind),
            scorer: None,
            generation: None,
            weight: 1.0,
            tools: None,
        });
        i = tool_end + 1;
    }
    out
}

/// Final-answer eval for trajectories that include at least one tool
/// exchange. Prompt = messages[0..last_assistant], target = last assistant
/// turn. Empty when there's no tool turn or no assistant turn after the
/// last tool response.
fn end_of_trajectory_answer(
    messages: &[SftMessage],
    conv: &SftConversation,
) -> Vec<EvalExample> {
    if !messages.iter().any(|m| m.role == "tool") {
        return Vec::new();
    }
    let last_tool = match messages.iter().rposition(|m| m.role == "tool") {
        Some(i) => i,
        None => return Vec::new(),
    };
    // Find the first assistant turn AFTER the last tool result.
    let final_assistant = match messages
        .iter()
        .enumerate()
        .skip(last_tool + 1)
        .find(|(_, m)| m.role == "assistant")
    {
        Some((idx, _)) => idx,
        None => return Vec::new(),
    };
    let (target_text, target_kind) = match assistant_target(&messages[final_assistant]) {
        Some(p) => p,
        None => return Vec::new(),
    };
    let prompt = &messages[..final_assistant];
    if prompt.is_empty() {
        return Vec::new();
    }
    vec![EvalExample {
        id: None,
        messages: to_chat_messages(prompt),
        target: Some(target_text),
        aliases: Vec::new(),
        tags: vec![
            "synth:end_of_trajectory".into(),
            target_kind.tag().to_string(),
        ],
        metadata: trajectory_metadata_with_kind(conv, &target_kind),
        scorer: None,
        generation: None,
        weight: 1.0,
        tools: None,
    }]
}

fn every_assistant_turn(
    messages: &[SftMessage],
    conv: &SftConversation,
    tool_calls_only: bool,
) -> Vec<EvalExample> {
    let mut out = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        if m.role != "assistant" {
            continue;
        }
        let (target_text, target_kind) = match assistant_target(m) {
            Some((text, kind)) => (text, kind),
            None => continue,
        };
        if tool_calls_only && !matches!(target_kind, AssistantTargetKind::ToolCall) {
            continue;
        }
        let prompt = trim_trailing_role(&messages[..i], "tool");
        if prompt.is_empty() {
            continue;
        }
        let base_tag = if tool_calls_only {
            "synth:tool_call_predict"
        } else {
            "synth:every_assistant"
        };
        let mut tags = vec![
            base_tag.to_string(),
            format!("synth:turn_{i}"),
            target_kind.tag().to_string(),
        ];
        if let AssistantTargetKind::Code { ref language } = target_kind {
            if let Some(l) = language.as_deref() {
                tags.push(format!("synth:lang_{l}"));
            }
        }
        out.push(EvalExample {
            id: None,
            messages: to_chat_messages(prompt),
            target: Some(target_text),
            aliases: Vec::new(),
            tags,
            metadata: trajectory_metadata_with_kind(conv, &target_kind),
            scorer: None,
            generation: None,
            weight: 1.0,
            tools: None,
        });
    }
    out
}

/// Resolves the "target" the model should reproduce for an assistant turn.
///
/// Priority:
/// 1. OpenAI-style structured `tool_calls` on the message — canonicalize
///    into a JSON object and tag as `AssistantTargetKind::ToolCall`.
/// 2. Inline JSON / fenced `tool_call` blocks in `content` — same.
/// 3. Fenced code block in `content` — tag as `AssistantTargetKind::Code`
///    so the synthesis layer can promote it to a code-aware scorer.
/// 4. Plain text content — `AssistantTargetKind::Prose`.
fn assistant_target(m: &SftMessage) -> Option<(String, AssistantTargetKind)> {
    if let Some(tc) = m.tool_calls.as_ref().filter(|tc| !tc.is_empty()) {
        let canonical = canonicalize_tool_calls(tc);
        return Some((canonical, AssistantTargetKind::ToolCall));
    }
    let content = m.content.trim();
    if content.is_empty() {
        return None;
    }
    if let Some(json) = extract_tool_call_from_content(content) {
        return Some((json, AssistantTargetKind::ToolCall));
    }
    // Qwen3.5 native XML tool call embedded in `content` (rare in upstream
    // SFT, but appears in trajectories captured directly from Qwen3.5
    // base-model outputs). Canonicalize to the JSON target shape so the
    // tool_call scorer compares structurally.
    let qwen_calls = crate::qwen3::extract_tool_calls(content);
    if !qwen_calls.is_empty() {
        let arr: Vec<serde_json::Value> = qwen_calls
            .iter()
            .map(|c| c.to_canonical_json())
            .collect();
        let canonical = serde_json::json!({"tool_calls": arr});
        return Some((canonical.to_string(), AssistantTargetKind::ToolCall));
    }
    if let Some((language, block)) = extract_first_code_block(content) {
        return Some((block, AssistantTargetKind::Code { language }));
    }
    Some((m.content.clone(), AssistantTargetKind::Prose))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AssistantTargetKind {
    Prose,
    ToolCall,
    Code { language: Option<String> },
}

impl AssistantTargetKind {
    fn tag(&self) -> &'static str {
        match self {
            AssistantTargetKind::Prose => "kind:prose",
            AssistantTargetKind::ToolCall => "kind:tool_call",
            AssistantTargetKind::Code { .. } => "kind:code",
        }
    }
}

/// Produce a canonical `{"tool_calls": [{"name", "arguments"}]}` JSON
/// string from the OpenAI-style structured field. Argument values are
/// kept as JSON values so downstream scorers can inspect them.
fn canonicalize_tool_calls(tc: &[serde_json::Value]) -> String {
    let mut out = Vec::with_capacity(tc.len());
    for call in tc {
        let function = call
            .get("function")
            .cloned()
            .unwrap_or_else(|| call.clone());
        let name = function
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let arguments = function
            .get("arguments")
            .cloned()
            .map(|v| {
                if let Some(s) = v.as_str() {
                    // OpenAI ships arguments as a JSON-encoded string; parse
                    // when possible so equality checks aren't sensitive to
                    // whitespace differences.
                    serde_json::from_str::<serde_json::Value>(s).unwrap_or(serde_json::json!(s))
                } else {
                    v
                }
            })
            .unwrap_or(serde_json::Value::Null);
        out.push(serde_json::json!({
            "name": name,
            "arguments": arguments,
        }));
    }
    serde_json::to_string(&serde_json::json!({ "tool_calls": out }))
        .unwrap_or_else(|_| "{\"tool_calls\":[]}".to_string())
}

/// Look for inline tool-call shapes in plain content. Returns a normalized
/// JSON string when found, so downstream targets are uniform regardless of
/// whether the upstream dataset used `tool_calls` or inline JSON.
fn extract_tool_call_from_content(content: &str) -> Option<String> {
    // Strip optional ```tool_call ... ``` fences.
    let stripped = if let Some(rest) = content.strip_prefix("```tool_call") {
        rest.split("```").next().unwrap_or(rest).trim().to_string()
    } else if content.contains("<tool_call>") {
        content
            .split("<tool_call>")
            .nth(1)?
            .split("</tool_call>")
            .next()?
            .trim()
            .to_string()
    } else {
        content.trim().to_string()
    };
    if !(stripped.starts_with('{') || stripped.starts_with('[')) {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_str(&stripped).ok()?;
    let obj = parsed.as_object()?;
    if !(obj.contains_key("tool_call")
        || obj.contains_key("function")
        || (obj.contains_key("name") && obj.contains_key("arguments")))
    {
        return None;
    }
    // Normalize to the canonical {"tool_calls":[{"name", "arguments"}]} shape.
    let (name, arguments) = if let Some(call) = obj.get("tool_call").and_then(|v| v.as_object()) {
        (
            call.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            call.get("arguments").cloned().unwrap_or(serde_json::Value::Null),
        )
    } else if let Some(function) = obj.get("function").and_then(|v| v.as_object()) {
        (
            function
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            function
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
    } else {
        (
            obj.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            obj.get("arguments").cloned().unwrap_or(serde_json::Value::Null),
        )
    };
    let canonical = serde_json::json!({
        "tool_calls": [{"name": name, "arguments": arguments}],
    });
    Some(canonical.to_string())
}

/// Pull the first fenced code block out of an assistant message. Returns
/// `(language_tag, body)`. Falls back to None when no fence is present.
fn extract_first_code_block(content: &str) -> Option<(Option<String>, String)> {
    let mut chars = content.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c != '`' {
            continue;
        }
        // Need three backticks.
        if !content[i..].starts_with("```") {
            continue;
        }
        let after = &content[i + 3..];
        // Language tag is up to the first newline.
        let (lang_raw, body_with_close) = after.split_once('\n').unwrap_or((after, ""));
        let language = lang_raw.trim();
        let language_opt = if language.is_empty() {
            None
        } else {
            Some(language.to_string())
        };
        let body = body_with_close.split("```").next().unwrap_or("").to_string();
        if body.trim().is_empty() {
            continue;
        }
        // Heuristic: only treat as a "code" target when the language tag
        // looks like a real language or when the body has multiple lines.
        let is_code = body.lines().count() >= 2
            || matches!(
                language_opt.as_deref().unwrap_or(""),
                "python" | "py" | "rust" | "rs" | "ts" | "tsx" | "js" | "go" | "java" | "c"
                | "cpp" | "c++" | "cs" | "kt" | "swift" | "rb" | "php" | "scala" | "sh" | "bash"
                | "zsh" | "html" | "css" | "json" | "yaml" | "yml" | "toml" | "sql"
            );
        if !is_code {
            return None;
        }
        return Some((language_opt, body));
    }
    None
}

fn trajectory_metadata_with_kind(
    conv: &SftConversation,
    kind: &AssistantTargetKind,
) -> Option<serde_json::Value> {
    let base = trajectory_metadata(conv).unwrap_or(serde_json::json!({}));
    let mut obj = base.as_object().cloned().unwrap_or_default();
    obj.insert(
        "synth_kind".into(),
        serde_json::json!(kind.tag().trim_start_matches("kind:")),
    );
    if let AssistantTargetKind::Code { language } = kind {
        if let Some(l) = language.as_deref() {
            obj.insert("synth_language".into(), serde_json::json!(l));
        }
    }
    Some(serde_json::Value::Object(obj))
}

fn filter_messages(messages: &[SftMessage], strip_system: bool) -> Vec<SftMessage> {
    messages
        .iter()
        .filter(|m| !(strip_system && m.role == "system"))
        .cloned()
        .collect()
}

fn trim_trailing_role<'a>(slice: &'a [SftMessage], role: &str) -> &'a [SftMessage] {
    let mut end = slice.len();
    while end > 0 && slice[end - 1].role == role {
        end -= 1;
    }
    &slice[..end]
}

fn to_chat_messages(messages: &[SftMessage]) -> Vec<EvalChatMessage> {
    // Preserve the full agentic shape, including `tool` role replies and
    // assistant `tool_calls`. Qwen3.5's chat template renders these into
    // its native XML form, so prompts that depend on prior tool exchanges
    // (the "next-action prediction" eval flow) get the same input the
    // model saw in production.
    messages
        .iter()
        .map(|m| EvalChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            tool_calls: m.tool_calls.clone(),
            name: m.name.clone(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect()
}

fn trajectory_metadata(conv: &SftConversation) -> Option<serde_json::Value> {
    if conv.extra.is_empty() {
        return None;
    }
    Some(serde_json::Value::Object(conv.extra.clone()))
}

fn scorer_kind_for_stats(choice: &ScorerChoice, target: &str) -> Option<&'static str> {
    match choice {
        ScorerChoice::AutoDetect => Some(auto_detect_scorer(target).kind_label()),
        ScorerChoice::Fixed(s) => Some(s.kind_label()),
        ScorerChoice::Judge { .. } => Some("llm_judge"),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("trajectory had no applicable turns for this strategy")]
    NoApplicableTurns,
    #[error("synthesis produced no examples — every trajectory was filtered out")]
    NoExamples,
}


#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> SftMessage {
        SftMessage {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }
    }

    fn conv(roles_and_content: &[(&str, &str)]) -> SftConversation {
        SftConversation {
            messages: roles_and_content
                .iter()
                .map(|(r, c)| msg(r, c))
                .collect(),
            extra: Default::default(),
        }
    }

    #[test]
    fn auto_scorer_detects_numeric_integer() {
        let s = auto_detect_scorer("185");
        match s {
            Scorer::NumericTolerance(t) => {
                assert!(t.integer_only);
                assert_eq!(t.atol, 0.0);
            }
            other => panic!("expected NumericTolerance, got {other:?}"),
        }
    }

    #[test]
    fn auto_scorer_detects_numeric_float() {
        let s = auto_detect_scorer("3.14159");
        match s {
            Scorer::NumericTolerance(t) => assert!(!t.integer_only),
            other => panic!("expected NumericTolerance, got {other:?}"),
        }
    }

    #[test]
    fn auto_scorer_detects_mcq_letter() {
        let s = auto_detect_scorer("B");
        assert!(matches!(s, Scorer::MultipleChoice { .. }));
    }

    #[test]
    fn auto_scorer_detects_json_object() {
        let s = auto_detect_scorer(r#"{"x": 1}"#);
        match s {
            Scorer::JsonValidity { require_object, .. } => assert!(require_object),
            other => panic!("expected JsonValidity, got {other:?}"),
        }
    }

    #[test]
    fn auto_scorer_short_text_is_exact_match() {
        let s = auto_detect_scorer("Paris");
        assert!(matches!(s, Scorer::ExactMatch { .. }));
    }

    #[test]
    fn auto_scorer_long_text_uses_contains_with_extracted_phrases() {
        let target = "The capital of France is Paris and it is famous for the Eiffel Tower.\nHistorical landmarks attract millions of visitors annually.";
        let s = auto_detect_scorer(target);
        match s {
            Scorer::Contains { phrases, .. } => {
                assert!(!phrases.is_empty());
                assert!(phrases.len() <= 3);
            }
            other => panic!("expected Contains, got {other:?}"),
        }
    }

    #[test]
    fn final_assistant_extracts_last_turn() {
        let c = conv(&[
            ("system", "you are helpful"),
            ("user", "What's 2+2?"),
            ("assistant", "4"),
        ]);
        let cfg = SynthesisConfig::new("test");
        let (suite, stats) = synthesize_suite(vec![c], &cfg).unwrap();
        assert_eq!(stats.examples_generated, 1);
        assert_eq!(suite.examples.len(), 1);
        let ex = &suite.examples[0];
        assert_eq!(ex.target.as_deref(), Some("4"));
        assert_eq!(ex.messages.len(), 2);
        assert_eq!(ex.messages[1].role, "user");
    }

    #[test]
    fn final_assistant_skips_when_no_assistant_turn() {
        let c = conv(&[("system", "..."), ("user", "hi")]);
        let cfg = SynthesisConfig::new("test");
        let res = synthesize_suite(vec![c], &cfg);
        assert!(matches!(res, Err(SynthesisError::NoExamples)));
    }

    #[test]
    fn every_assistant_turn_emits_one_per_assistant() {
        let c = conv(&[
            ("system", "s"),
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
            ("user", "q3"),
            ("assistant", "a3"),
        ]);
        let mut cfg = SynthesisConfig::new("test");
        cfg.strategy = SynthesisStrategy::EveryAssistantTurn;
        let (suite, stats) = synthesize_suite(vec![c], &cfg).unwrap();
        assert_eq!(stats.examples_generated, 3);
        assert_eq!(suite.examples.len(), 3);
    }

    #[test]
    fn first_assistant_turn_keeps_only_first() {
        let c = conv(&[
            ("system", "s"),
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
        ]);
        let mut cfg = SynthesisConfig::new("test");
        cfg.strategy = SynthesisStrategy::FirstAssistantTurn;
        let (suite, _) = synthesize_suite(vec![c], &cfg).unwrap();
        assert_eq!(suite.examples.len(), 1);
        assert_eq!(suite.examples[0].target.as_deref(), Some("a1"));
    }

    #[test]
    fn strip_system_prompt_removes_system_messages_from_prompt() {
        let c = conv(&[
            ("system", "huge system prompt"),
            ("user", "hello"),
            ("assistant", "hi"),
        ]);
        let mut cfg = SynthesisConfig::new("test");
        cfg.strip_system_prompt = true;
        let (suite, _) = synthesize_suite(vec![c], &cfg).unwrap();
        assert!(suite.examples[0]
            .messages
            .iter()
            .all(|m| m.role != "system"));
    }

    #[test]
    fn agentic_trailing_tool_messages_are_trimmed_from_prompt() {
        let c = conv(&[
            ("system", "s"),
            ("user", "q"),
            ("assistant", "a1"),
            ("tool", "tool reply 1"),
            ("tool", "tool reply 2"),
            ("assistant", "final answer"),
        ]);
        let cfg = SynthesisConfig::new("test");
        let (suite, _) = synthesize_suite(vec![c], &cfg).unwrap();
        let prompt = &suite.examples[0].messages;
        // No tool messages in the prompt (we filter them anyway), and the
        // last prompt message should be an assistant turn after tool trim.
        assert!(prompt.iter().all(|m| m.role != "tool"));
        assert_eq!(prompt.last().unwrap().role, "assistant");
        assert_eq!(suite.examples[0].target.as_deref(), Some("final answer"));
    }

    #[test]
    fn dedup_collapses_identical_user_target_pairs() {
        let c1 = conv(&[("user", "q"), ("assistant", "a")]);
        let c2 = c1.clone();
        let cfg = SynthesisConfig::new("test");
        let (suite, stats) = synthesize_suite(vec![c1, c2], &cfg).unwrap();
        assert_eq!(suite.examples.len(), 1);
        assert_eq!(stats.skipped_duplicate, 1);
    }

    #[test]
    fn max_prompt_chars_filters_long_trajectories() {
        let huge = "x".repeat(100_000);
        let c = SftConversation {
            messages: vec![
                msg("user", &huge),
                msg("assistant", "short"),
            ],
            extra: Default::default(),
        };
        let mut cfg = SynthesisConfig::new("test");
        cfg.sampling.max_prompt_chars = 1024;
        let res = synthesize_suite(vec![c], &cfg);
        assert!(matches!(res, Err(SynthesisError::NoExamples)));
    }

    #[test]
    fn max_examples_is_respected_via_reservoir() {
        let cs: Vec<_> = (0..50)
            .map(|i| {
                conv(&[
                    ("user", &format!("q{i}")),
                    ("assistant", &format!("a{i}")),
                ])
            })
            .collect();
        let mut cfg = SynthesisConfig::new("cap");
        cfg.sampling.max_examples = Some(7);
        cfg.sampling.seed = Some(123);
        cfg.sampling.dedupe = false;
        let (suite, stats) = synthesize_suite(cs, &cfg).unwrap();
        assert_eq!(suite.examples.len(), 7);
        assert_eq!(stats.sample_kept, 7);
        assert_eq!(stats.examples_generated, 50);
    }

    #[test]
    fn fixed_scorer_choice_sets_default_only() {
        let c = conv(&[("user", "q"), ("assistant", "a")]);
        let mut cfg = SynthesisConfig::new("test");
        cfg.scorer = ScorerChoice::Fixed(Scorer::Contains {
            phrases: vec!["a".into()],
            mode: ContainsMode::Any,
            case_sensitive: false,
        });
        let (suite, _) = synthesize_suite(vec![c], &cfg).unwrap();
        assert!(matches!(suite.default_scorer, Scorer::Contains { .. }));
        // No per-example overrides in Fixed mode.
        assert!(suite.examples[0].scorer.is_none());
    }

    #[test]
    fn auto_scorer_writes_per_example_overrides() {
        let c1 = conv(&[("user", "q"), ("assistant", "42")]);
        let c2 = conv(&[("user", "q2"), ("assistant", "Paris")]);
        let cfg = SynthesisConfig::new("test");
        let (suite, stats) = synthesize_suite(vec![c1, c2], &cfg).unwrap();
        for ex in &suite.examples {
            assert!(ex.scorer.is_some());
        }
        // Auto-detect histogram has both scorer kinds.
        assert!(stats.auto_scorer_histogram.len() >= 1);
    }

    #[test]
    fn tool_call_predict_only_emits_tool_call_targets() {
        let c = conv(&[
            ("user", "q"),
            ("assistant", "let me think"),
            ("user", "again"),
            ("assistant", r#"{"tool_call": {"name": "search"}}"#),
            ("user", "more"),
            ("assistant", "a regular answer"),
        ]);
        let mut cfg = SynthesisConfig::new("test");
        cfg.strategy = SynthesisStrategy::ToolCallPredict;
        cfg.scorer = ScorerChoice::Fixed(Scorer::JsonValidity {
            require_object: true,
            required_paths: vec![],
        });
        let (suite, _) = synthesize_suite(vec![c], &cfg).unwrap();
        assert_eq!(suite.examples.len(), 1);
        assert!(suite.examples[0].target.as_deref().unwrap().contains("tool_call"));
    }

    #[test]
    fn empty_assistant_content_is_skipped() {
        let c = conv(&[
            ("user", "q"),
            ("assistant", ""),
            ("user", "again"),
            ("assistant", "real"),
        ]);
        let mut cfg = SynthesisConfig::new("test");
        cfg.strategy = SynthesisStrategy::FinalAssistant;
        let (suite, _) = synthesize_suite(vec![c], &cfg).unwrap();
        assert_eq!(suite.examples[0].target.as_deref(), Some("real"));
    }

    fn msg_with_tc(
        role: &str,
        content: &str,
        tool_calls: Option<Vec<serde_json::Value>>,
    ) -> SftMessage {
        SftMessage {
            role: role.into(),
            content: content.into(),
            tool_calls,
            name: None,
            tool_call_id: None,
        }
    }

    fn tool_msg(content: &str) -> SftMessage {
        SftMessage {
            role: "tool".into(),
            content: content.into(),
            tool_calls: None,
            name: Some("get_weather".into()),
            tool_call_id: Some("call_1".into()),
        }
    }

    #[test]
    fn tool_response_followup_emits_per_tool_result() {
        let assistant_tc = vec![serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
        })];
        let conv = SftConversation {
            messages: vec![
                msg("system", "be helpful"),
                msg("user", "weather in paris"),
                msg_with_tc("assistant", "", Some(assistant_tc)),
                tool_msg("18C, cloudy"),
                msg("assistant", "It's 18°C and cloudy in Paris."),
            ],
            extra: Default::default(),
        };
        let mut cfg = SynthesisConfig::new("test");
        cfg.strategy = SynthesisStrategy::ToolResponseFollowup;
        let (suite, _) = synthesize_suite(vec![conv], &cfg).unwrap();
        assert_eq!(suite.examples.len(), 1);
        let ex = &suite.examples[0];
        // Prompt ends on the tool response so the model sees the result.
        assert_eq!(ex.messages.last().unwrap().role, "tool");
        assert_eq!(ex.target.as_deref(), Some("It's 18°C and cloudy in Paris."));
        assert!(ex.tags.iter().any(|t| t == "synth:tool_response_followup"));
    }

    #[test]
    fn end_of_trajectory_answer_requires_a_tool_exchange() {
        // No tool exchange — strategy must skip.
        let plain = conv(&[("user", "q"), ("assistant", "a")]);
        let mut cfg = SynthesisConfig::new("plain");
        cfg.strategy = SynthesisStrategy::EndOfTrajectoryAnswer;
        let res = synthesize_suite(vec![plain], &cfg);
        assert!(matches!(res, Err(SynthesisError::NoExamples)));

        // With a tool exchange the synthesizer keeps the final assistant.
        let assistant_tc = vec![serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"}
        })];
        let agentic = SftConversation {
            messages: vec![
                msg("user", "q"),
                msg_with_tc("assistant", "", Some(assistant_tc)),
                tool_msg("ok"),
                msg("assistant", "Done."),
            ],
            extra: Default::default(),
        };
        let mut cfg = SynthesisConfig::new("agentic");
        cfg.strategy = SynthesisStrategy::EndOfTrajectoryAnswer;
        let (suite, _) = synthesize_suite(vec![agentic], &cfg).unwrap();
        assert_eq!(suite.examples.len(), 1);
        assert_eq!(suite.examples[0].target.as_deref(), Some("Done."));
    }

    #[test]
    fn qwen3_xml_content_canonicalizes_to_tool_call_target() {
        // Some trajectories store assistant tool calls as the raw Qwen3.5
        // XML content (no structured `tool_calls` field). Synthesis must
        // detect and canonicalize so the tool_call scorer can compare.
        let raw_xml = "<tool_call>\n<function=set>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>";
        let c = conv(&[("user", "do it"), ("assistant", raw_xml)]);
        let mut cfg = SynthesisConfig::new("xml-target");
        cfg.strategy = SynthesisStrategy::FinalAssistant;
        let (suite, _) = synthesize_suite(vec![c], &cfg).unwrap();
        let target = suite.examples[0].target.as_deref().unwrap();
        // Target was canonicalized into the JSON tool_calls envelope.
        assert!(target.contains("\"tool_calls\""), "{target}");
        assert!(target.contains("\"set\""));
        assert!(target.contains("\"k\""));
    }

    #[test]
    fn seed_determinism_makes_reservoir_stable() {
        let cs: Vec<_> = (0..30)
            .map(|i| {
                conv(&[
                    ("user", &format!("q{i}")),
                    ("assistant", &format!("a{i}")),
                ])
            })
            .collect();
        let mut cfg = SynthesisConfig::new("test");
        cfg.sampling.max_examples = Some(5);
        cfg.sampling.seed = Some(42);
        cfg.sampling.dedupe = false;
        let (a, _) = synthesize_suite(cs.clone(), &cfg).unwrap();
        let (b, _) = synthesize_suite(cs, &cfg).unwrap();
        let a_targets: Vec<_> = a.examples.iter().map(|e| e.target.clone()).collect();
        let b_targets: Vec<_> = b.examples.iter().map(|e| e.target.clone()).collect();
        assert_eq!(a_targets, b_targets);
    }
}
