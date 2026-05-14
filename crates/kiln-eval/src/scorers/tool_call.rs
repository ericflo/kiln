//! Tool-call scorer.
//!
//! Compares a predicted tool call (extracted from the model's completion)
//! against the trajectory-recorded target. Three things contribute:
//!
//! 1. **Name match** — did the model pick the right function?
//! 2. **Argument structural match** — same set of keys? bonus for nested
//!    structural equality on JSON-valued args.
//! 3. **Argument content quality** — per-arg sub-scorers grade the *value*
//!    of each argument independently. This is the secret sauce: a tool
//!    call whose `query` argument should be a free-form sentence can be
//!    scored by `Contains` or `LlmJudge` over that argument's value; a
//!    `code` argument can be scored by `Scorer::Code`.
//!
//! The final score is a weighted combination of the three. Defaults give
//! 0.4 to name, 0.3 to structure, 0.3 to content quality.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::qwen3::{ParsedToolCall, ToolCallFormat, extract_tool_calls, split_thinking};
use crate::result::EvalOutcomeKind;
use crate::scorers::{
    JudgeRunner, Scorer, ScorerError, bash, code::extract_block, score_completion,
};
use crate::suite::EvalExample;

/// How strictly the predicted tool *name* must match the target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NameMatch {
    /// Exact equality.
    Exact,
    /// Case-insensitive equality.
    CaseInsensitive,
    /// Pass if the predicted name appears in this set. Useful when more
    /// than one tool is acceptable (read_file or read_lines, etc.).
    OneOf { allowed: Vec<String> },
}

impl Default for NameMatch {
    fn default() -> Self {
        NameMatch::CaseInsensitive
    }
}

/// How to score arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArgsScoring {
    /// Score only on whether the argument *keys* match (set equality).
    /// Doesn't look at values. Cheap and useful when arg values are
    /// nondeterministic (timestamps, IDs).
    KeysOnly,
    /// Score on full structural equality of the JSON. Effectively
    /// `JsonValidity` over arguments, with deep canonicalization.
    Structural,
    /// Per-key sub-scorers. The arg's value is converted to a string
    /// (json-stringified for objects/arrays, raw for strings/numbers)
    /// and fed to the named sub-scorer. Missing keys score 0.
    PerKey {
        scorers: BTreeMap<String, Scorer>,
        /// Score for keys present in the prediction but not in the spec.
        /// Defaults to 0 (penalize spurious args).
        #[serde(default)]
        extra_key_penalty: f32,
    },
    /// Auto: structural for non-string args, contains-based for string
    /// args (extracting the top 3 phrases from the target value).
    #[serde(rename = "auto")]
    Auto,
}

impl Default for ArgsScoring {
    fn default() -> Self {
        ArgsScoring::Auto
    }
}

impl ArgsScoring {
    pub(super) fn requires_judge(&self) -> bool {
        match self {
            ArgsScoring::PerKey { scorers, .. } => {
                scorers.values().any(|s| s.requires_judge())
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ToolCallWeights {
    pub name: f32,
    pub structure: f32,
    pub content: f32,
}

/// Pass threshold on the combined (weighted) score. Tuned so that "right
/// tool, mostly-right args" lands as Pass, while "right tool, half-wrong
/// args" lands as Fail.
const TOOL_CALL_PASS_THRESHOLD: f32 = 0.8;
/// Score awarded when the predicted bash command shares the program but
/// not the subcommand (e.g. `git commit` vs `git push`). Partial credit
/// — same family, but the model picked the wrong action.
const BASH_SAME_PROGRAM_DIFF_SUBCOMMAND: f32 = 0.35;
/// Score awarded when the predicted bash command's classification doesn't
/// match the target's at all (e.g. `python_inline` vs `pip_install`).
const BASH_WRONG_FAMILY: f32 = 0.1;

impl Default for ToolCallWeights {
    fn default() -> Self {
        // Equal-ish weighting biased slightly toward picking the right
        // function name. If the model gets the name wrong nothing else
        // matters; the structure/content weights only break ties between
        // candidates that picked the right tool but disagreed on args.
        Self {
            name: 0.4,
            structure: 0.3,
            content: 0.3,
        }
    }
}

pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    name_match: &NameMatch,
    args: &ArgsScoring,
    weights: Option<&ToolCallWeights>,
    require_xml_format: bool,
    judge_runner: &dyn JudgeRunner,
) -> Result<(f32, EvalOutcomeKind, Option<String>), ScorerError> {
    let target_raw = example.target.as_deref().ok_or(ScorerError::MissingTarget {
        kind: "tool_call",
    })?;
    let target_calls = extract_tool_calls(target_raw);
    if target_calls.is_empty() {
        return Err(ScorerError::MissingTarget {
            kind: "tool_call (target had no parseable tool call)",
        });
    }
    let predicted_calls = extract_tool_calls(completion_text);
    // Strict-format mode: if every predicted call is NOT in Qwen3.5
    // native XML, mark the example Invalid. The argument is that a
    // model trained on Qwen3.5's native format should never emit JSON
    // here, so a JSON prediction is a regression worth flagging
    // distinctly from a wrong-tool prediction.
    if require_xml_format
        && !predicted_calls.is_empty()
        && predicted_calls
            .iter()
            .all(|c| c.format != ToolCallFormat::Qwen3Xml)
    {
        return Ok((
            0.0,
            EvalOutcomeKind::Invalid,
            Some(
                "non-XML tool call emitted (require_xml_format=true)".to_string(),
            ),
        ));
    }
    if predicted_calls.is_empty() {
        // No call was emitted. Surface why: thinking-only completions, or
        // a textual refusal, or just a free-form answer where a call was
        // expected. The split_thinking probe lets us mention reasoning
        // length so users can spot the "model thought but never acted"
        // pattern in dashboards.
        let probe = split_thinking(completion_text);
        let detail = if probe.unclosed {
            "no tool_call found (thinking block never closed)".to_string()
        } else if probe.had_thinking && probe.answer.trim().is_empty() {
            "no tool_call found (model emitted thinking but no answer)".to_string()
        } else {
            "no tool_call found in completion".to_string()
        };
        return Ok((0.0, EvalOutcomeKind::Invalid, Some(detail)));
    }

    // Multi-call target: score each call against the target at the same
    // position, then average. Extra predicted calls penalize the score;
    // missing predicted calls show up as misses.
    let pairs = pair_calls(&target_calls, &predicted_calls);
    let pair_count = pairs.len() as f32;
    let weights = weights.copied().unwrap_or_default();
    let total_weight = weights.name + weights.structure + weights.content;
    let total_weight = if total_weight <= 0.0 { 1.0 } else { total_weight };
    let n_w = weights.name / total_weight;
    let s_w = weights.structure / total_weight;
    let c_w = weights.content / total_weight;

    let mut combined_sum = 0.0f32;
    let mut all_name_perfect = true;
    let mut details: Vec<String> = Vec::new();
    for (i, pair) in pairs.iter().enumerate() {
        let (combined, name_perfect, mut detail) =
            score_pair(example, pair, name_match, args, &weights, judge_runner)?;
        if !name_perfect {
            all_name_perfect = false;
        }
        combined_sum += combined;
        // Annotate per-call details with the index when there's more than
        // one. Single-call evals stay terse.
        if pair_count > 1.0 {
            detail = format!("[{}] {}", i, detail);
        }
        details.push(detail);
    }
    let combined_avg = combined_sum / pair_count.max(1.0);

    // Penalty for extra predicted calls beyond the target count: each one
    // costs `0.25 / target_count` of the final score, so a model that fires
    // twice as many tool calls as expected lands in Fail territory even if
    // each call looks plausible.
    let excess = predicted_calls.len().saturating_sub(target_calls.len());
    let excess_penalty = if target_calls.is_empty() {
        0.0
    } else {
        (excess as f32) * (0.25 / target_calls.len() as f32)
    };
    let combined = (combined_avg - excess_penalty).clamp(0.0, 1.0);

    let _ = (n_w, s_w, c_w); // weights consumed inside score_pair
    let kind = if all_name_perfect && combined >= TOOL_CALL_PASS_THRESHOLD {
        EvalOutcomeKind::Pass
    } else {
        EvalOutcomeKind::Fail
    };

    let mut detail = details.join(" || ");
    if excess > 0 {
        detail.push_str(&format!(" || excess_calls={}", excess));
    }
    if predicted_calls
        .iter()
        .any(|c| c.format != ToolCallFormat::Qwen3Xml)
    {
        // Non-blocking diagnostic — many upstream datasets store calls in
        // JSON form, so JSON predictions are still valid output. But for
        // Qwen3.5 the *native* output is XML, so a JSON-only completion is
        // a sign that the model produced a free-form answer rather than
        // using the chat template's tool-call grammar.
        let fmts: Vec<&'static str> = predicted_calls
            .iter()
            .map(|c| match c.format {
                ToolCallFormat::Qwen3Xml => "xml",
                ToolCallFormat::JsonInline => "json",
                ToolCallFormat::OpenAi => "openai",
                ToolCallFormat::Fenced => "fenced",
            })
            .collect();
        detail.push_str(&format!(" || formats=[{}]", fmts.join(",")));
    }

    Ok((combined, kind, Some(detail)))
}

struct CallPair<'a> {
    target: &'a ParsedToolCall,
    predicted: Option<&'a ParsedToolCall>,
}

fn pair_calls<'a>(
    targets: &'a [ParsedToolCall],
    predicted: &'a [ParsedToolCall],
) -> Vec<CallPair<'a>> {
    let mut out = Vec::with_capacity(targets.len());
    for (i, t) in targets.iter().enumerate() {
        out.push(CallPair {
            target: t,
            predicted: predicted.get(i),
        });
    }
    out
}

fn score_pair(
    example: &EvalExample,
    pair: &CallPair<'_>,
    name_match: &NameMatch,
    args: &ArgsScoring,
    weights: &ToolCallWeights,
    judge_runner: &dyn JudgeRunner,
) -> Result<(f32, bool, String), ScorerError> {
    let total_weight = weights.name + weights.structure + weights.content;
    let total_weight = if total_weight <= 0.0 { 1.0 } else { total_weight };
    let n_w = weights.name / total_weight;
    let s_w = weights.structure / total_weight;
    let c_w = weights.content / total_weight;

    let Some(predicted) = pair.predicted else {
        return Ok((
            0.0,
            false,
            format!("missing predicted call (expected `{}`)", pair.target.name),
        ));
    };
    let target_args = serde_json::Value::Object(pair.target.arguments.clone());
    let predicted_args = serde_json::Value::Object(predicted.arguments.clone());
    let (name_score, name_detail) = score_name(&predicted.name, &pair.target.name, name_match);
    let (struct_score, struct_detail) = score_structural(&predicted_args, &target_args);
    let (content_score, content_detail) =
        score_content(example, &predicted_args, &target_args, args, judge_runner)?;
    let combined = name_score * n_w + struct_score * s_w + content_score * c_w;
    let detail = format!(
        "name={:.2} {} | struct={:.2} {} | content={:.2} {}",
        name_score,
        name_detail.as_deref().unwrap_or(""),
        struct_score,
        struct_detail.as_deref().unwrap_or(""),
        content_score,
        content_detail.as_deref().unwrap_or(""),
    );
    Ok((combined, name_score >= 1.0, detail))
}

fn score_name(
    predicted: &str,
    target: &str,
    match_mode: &NameMatch,
) -> (f32, Option<String>) {
    match match_mode {
        NameMatch::Exact => {
            if predicted == target {
                (1.0, None)
            } else {
                (0.0, Some(format!("expected `{target}`, got `{predicted}`")))
            }
        }
        NameMatch::CaseInsensitive => {
            if predicted.eq_ignore_ascii_case(target) {
                (1.0, None)
            } else {
                (0.0, Some(format!("expected `{target}`, got `{predicted}`")))
            }
        }
        NameMatch::OneOf { allowed } => {
            let ok = allowed
                .iter()
                .any(|a| a.eq_ignore_ascii_case(predicted));
            if ok {
                (1.0, None)
            } else {
                (
                    0.0,
                    Some(format!(
                        "expected one of [{}], got `{predicted}`",
                        allowed.join(", ")
                    )),
                )
            }
        }
    }
}

fn score_structural(predicted: &serde_json::Value, target: &serde_json::Value) -> (f32, Option<String>) {
    let p_keys: BTreeSet<&str> = predicted
        .as_object()
        .map(|m| m.keys().map(|s| s.as_str()).collect())
        .unwrap_or_default();
    let t_keys: BTreeSet<&str> = target
        .as_object()
        .map(|m| m.keys().map(|s| s.as_str()).collect())
        .unwrap_or_default();
    if t_keys.is_empty() && p_keys.is_empty() {
        return (1.0, None);
    }
    if t_keys.is_empty() {
        return (
            if p_keys.is_empty() { 1.0 } else { 0.0 },
            Some("target had no args; prediction had some".into()),
        );
    }
    let matched: usize = t_keys.intersection(&p_keys).count();
    let total = t_keys.len().max(1);
    let score = matched as f32 / total as f32;
    let missing: Vec<&&str> = t_keys.difference(&p_keys).collect();
    let extra: Vec<&&str> = p_keys.difference(&t_keys).collect();
    let detail = if missing.is_empty() && extra.is_empty() {
        None
    } else {
        Some(format!(
            "missing=[{}] extra=[{}]",
            missing.iter().copied().copied().collect::<Vec<&str>>().join(","),
            extra.iter().copied().copied().collect::<Vec<&str>>().join(",")
        ))
    };
    (score, detail)
}

fn score_content(
    example: &EvalExample,
    predicted: &serde_json::Value,
    target: &serde_json::Value,
    args: &ArgsScoring,
    judge_runner: &dyn JudgeRunner,
) -> Result<(f32, Option<String>), ScorerError> {
    let target_obj = match target.as_object() {
        Some(o) if !o.is_empty() => o,
        _ => return Ok((1.0, None)),
    };
    let p_obj = predicted.as_object();

    match args {
        ArgsScoring::KeysOnly => {
            // Already covered by structural scoring; content is a no-op.
            Ok((1.0, None))
        }
        ArgsScoring::Structural => {
            let p_canon = canonicalize(predicted);
            let t_canon = canonicalize(target);
            if p_canon == t_canon {
                Ok((1.0, None))
            } else {
                Ok((0.0, Some("args structurally differ".into())))
            }
        }
        ArgsScoring::PerKey {
            scorers,
            extra_key_penalty,
        } => {
            if scorers.is_empty() {
                return Ok((1.0, None));
            }
            let mut sum = 0.0f32;
            let mut details = Vec::new();
            let mut count = 0u32;
            for (key, scorer) in scorers {
                count += 1;
                let pred_value = p_obj.and_then(|m| m.get(key));
                let target_value = target_obj.get(key);
                let target_str = target_value.map(stringify_value).unwrap_or_default();
                let pred_str = pred_value.map(stringify_value).unwrap_or_default();
                // Run the scorer with a synthetic example whose target is the
                // expected arg value. This lets `Scorer::Code` / `Scorer::LlmJudge`
                // recurse cleanly.
                let mut synthetic = example.clone();
                synthetic.target = Some(target_str.clone());
                let out = score_completion(scorer, &synthetic, &pred_str, judge_runner)?;
                sum += out.score;
                if let Some(d) = out.detail {
                    details.push(format!("{key}: {d}"));
                }
            }
            // Penalize extra keys not in the spec.
            let extras = p_obj
                .map(|m| {
                    m.keys()
                        .filter(|k| !scorers.contains_key(*k) && !target_obj.contains_key(*k))
                        .count()
                })
                .unwrap_or(0);
            let extra_penalty = if extras > 0 {
                *extra_key_penalty * extras as f32
            } else {
                0.0
            };
            let mean = if count > 0 { sum / count as f32 } else { 1.0 };
            let final_score = (mean - extra_penalty).clamp(0.0, 1.0);
            Ok((
                final_score,
                if details.is_empty() {
                    None
                } else {
                    Some(details.join(" ; "))
                },
            ))
        }
        ArgsScoring::Auto => {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            let mut details = Vec::new();
            for (key, target_val) in target_obj {
                count += 1;
                let pred_val = p_obj.and_then(|m| m.get(key));
                if pred_val.is_none() {
                    details.push(format!("{key}: missing"));
                    continue;
                }
                let pred_val = pred_val.unwrap();
                let score = score_arg_value_auto(key, target_val, pred_val);
                if score < 1.0 {
                    details.push(format!("{key}: {score:.2}"));
                }
                sum += score;
            }
            let mean = if count > 0 { sum / count as f32 } else { 1.0 };
            Ok((
                mean,
                if details.is_empty() {
                    None
                } else {
                    Some(details.join(" ; "))
                },
            ))
        }
    }
}

/// Score a single argument value in `Auto` mode.
///
/// Special-cases bash-family `command` / `cmd` / `script` keys: introspects
/// both target and prediction, requires the *classification* (program
/// family, e.g. `python_inline`) to match, then sub-scores inline programs
/// with token-similarity over the inner code. This is what lets a "bash
/// tool call wrapping `python3 -c 'import os; print(os.getcwd())'`" be
/// disambiguated from a totally different bash invocation, even when both
/// stringify to similar-looking JSON.
fn score_arg_value_auto(
    key: &str,
    target: &serde_json::Value,
    predicted: &serde_json::Value,
) -> f32 {
    if is_command_key(key) {
        if let (Some(t), Some(p)) = (target.as_str(), predicted.as_str()) {
            return score_bash_command(t, p);
        }
    }
    match (target, predicted) {
        (serde_json::Value::String(t), serde_json::Value::String(p)) => {
            if t == p {
                return 1.0;
            }
            // Maybe the string is a code block — compare via token jaccard.
            if let (Some(t_code), Some(p_code)) = (extract_block(t, None), extract_block(p, None)) {
                return code_token_similarity(&t_code, &p_code);
            }
            let phrases = crate::scorers::contains::naive_key_phrases(t, 5);
            if phrases.is_empty() {
                char_jaccard(t, p)
            } else {
                let any = phrases
                    .iter()
                    .filter(|p_| p.to_ascii_lowercase().contains(&p_.to_ascii_lowercase()))
                    .count();
                let frac = any as f32 / phrases.len() as f32;
                (frac + 0.05).min(1.0)
            }
        }
        (a, b) if a == b => 1.0,
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            let af = a.as_f64().unwrap_or(0.0);
            let bf = b.as_f64().unwrap_or(0.0);
            if (af - bf).abs() <= af.abs() * 1e-3 + 1e-6 {
                1.0
            } else {
                0.0
            }
        }
        (serde_json::Value::Bool(a), serde_json::Value::Bool(b)) => {
            if a == b { 1.0 } else { 0.0 }
        }
        _ => {
            if canonicalize(target) == canonicalize(predicted) {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn is_command_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "command" | "cmd" | "script" | "shell" | "bash" | "code"
    )
}

/// Score two bash command strings. Strategy:
///
/// 1. Introspect both.
/// 2. If classifications differ (e.g. `python_inline` vs `pip_install`),
///    return a low score (0.1) — the model picked the wrong tool entirely.
/// 3. If classifications match but inline languages differ, return 0.2.
/// 4. If both are inline programs of the same language, sub-score the
///    inner code with token similarity (lenient — variable renames OK).
/// 5. Otherwise return token jaccard over the full command tail.
fn score_bash_command(target: &str, predicted: &str) -> f32 {
    let t_intro = bash::introspect(target);
    let p_intro = bash::introspect(predicted);
    if t_intro.classification() != p_intro.classification() {
        if t_intro.program == p_intro.program {
            return BASH_SAME_PROGRAM_DIFF_SUBCOMMAND;
        }
        return BASH_WRONG_FAMILY;
    }
    if let (Some(t_code), Some(p_code)) = (
        t_intro.inline_code.as_deref(),
        p_intro.inline_code.as_deref(),
    ) {
        return code_token_similarity(t_code, p_code);
    }
    // Same family, no inline code — compare argument tails as token sets.
    let t_tokens: BTreeSet<String> = t_intro.tail.iter().cloned().collect();
    let p_tokens: BTreeSet<String> = p_intro.tail.iter().cloned().collect();
    if t_tokens.is_empty() && p_tokens.is_empty() {
        return 1.0;
    }
    let inter = t_tokens.intersection(&p_tokens).count();
    let uni = t_tokens.union(&p_tokens).count().max(1);
    inter as f32 / uni as f32
}

/// Lightweight code similarity used both for arg-level bash inline code
/// and for fenced code-block args. Mirrors the default token-similarity
/// behavior of `Scorer::Code` so behavior stays consistent.
fn code_token_similarity(target: &str, predicted: &str) -> f32 {
    let t_tokens: BTreeSet<String> = code_tokens(target);
    let p_tokens: BTreeSet<String> = code_tokens(predicted);
    if t_tokens.is_empty() && p_tokens.is_empty() {
        return 1.0;
    }
    let inter = t_tokens.intersection(&p_tokens).count();
    let uni = t_tokens.union(&p_tokens).count().max(1);
    inter as f32 / uni as f32
}

fn code_tokens(code: &str) -> BTreeSet<String> {
    code.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_string())
        .collect()
}

fn stringify_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn canonicalize(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut out = serde_json::Map::new();
            for (k, v) in entries {
                out.insert(k.clone(), canonicalize(v));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalize).collect())
        }
        _ => v.clone(),
    }
}

fn char_jaccard(a: &str, b: &str) -> f32 {
    let a_set: BTreeSet<char> = a.chars().collect();
    let b_set: BTreeSet<char> = b.chars().collect();
    let inter = a_set.intersection(&b_set).count();
    let uni = a_set.union(&b_set).count().max(1);
    inter as f32 / uni as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorers::NoopJudgeRunner;
    use crate::suite::EvalChatMessage;

    fn ex(target: &str) -> EvalExample {
        EvalExample {
            messages: vec![EvalChatMessage::new("user", "q")],
            target: Some(target.into()),
            ..Default::default()
        }
    }

    #[test]
    fn name_match_exact_passes_only_on_exact() {
        let (s, _) = score_name("get_weather", "get_weather", &NameMatch::Exact);
        assert_eq!(s, 1.0);
        let (s, _) = score_name("Get_weather", "get_weather", &NameMatch::Exact);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn name_match_case_insensitive() {
        let (s, _) = score_name("Get_Weather", "get_weather", &NameMatch::CaseInsensitive);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn name_match_one_of() {
        let mm = NameMatch::OneOf {
            allowed: vec!["read".into(), "open".into()],
        };
        let (s, _) = score_name("Read", "read", &mm);
        assert_eq!(s, 1.0);
        let (s, _) = score_name("delete", "read", &mm);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn full_score_passes_on_canonical_target() {
        let target = r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris","units":"c"}}]}"#;
        let pred = r#"{"tool_call":{"name":"get_weather","arguments":{"city":"Paris","units":"c"}}}"#;
        let (s, kind, _) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Structural,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert!(s > 0.95, "score was {s}");
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn wrong_name_fails_even_if_args_match() {
        let target = r#"{"tool_calls":[{"name":"read","arguments":{"path":"/x"}}]}"#;
        let pred = r#"{"name":"write","arguments":{"path":"/x"}}"#;
        let (_, kind, _) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Structural,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn missing_args_keys_partial_credit_under_keys_only() {
        let target = r#"{"tool_calls":[{"name":"f","arguments":{"a":1,"b":2}}]}"#;
        let pred = r#"{"name":"f","arguments":{"a":1}}"#;
        let (s, _, _) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::KeysOnly,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        // name=1.0 (0.4) + structural=0.5 (0.3) + content=1.0 (0.3) = 0.85
        assert!(s > 0.8, "score was {s}");
    }

    #[test]
    fn per_key_scorer_grades_inner_prose() {
        let target = r#"{"tool_calls":[{"name":"search","arguments":{"query":"capital of France is Paris"}}]}"#;
        let pred = r#"{"name":"search","arguments":{"query":"Paris is the capital of France"}}"#;
        let mut scorers = BTreeMap::new();
        scorers.insert(
            "query".to_string(),
            Scorer::Contains {
                phrases: vec!["paris".into(), "france".into()],
                mode: crate::scorers::contains::ContainsMode::All,
                case_sensitive: false,
            },
        );
        let (s, _, _) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::PerKey {
                scorers,
                extra_key_penalty: 0.1,
            },
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert!(s > 0.9, "score was {s}");
    }

    #[test]
    fn per_key_scorer_grades_code_args() {
        let target = r#"{"tool_calls":[{"name":"write","arguments":{"path":"main.py","content":"```python\ndef add(a, b):\n    return a + b\n```"}}]}"#;
        let pred = r#"{"name":"write","arguments":{"path":"main.py","content":"```python\ndef add(a,b):\n    return a+b\n```"}}"#;
        let mut scorers = BTreeMap::new();
        scorers.insert(
            "content".to_string(),
            Scorer::Code {
                language: Some("python".into()),
                style: crate::scorers::CodeStyle::TokenSimilarity {
                    min_jaccard: 0.6,
                },
            },
        );
        let (s, _, _) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::PerKey {
                scorers,
                extra_key_penalty: 0.0,
            },
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert!(s > 0.8, "score was {s}");
    }

    #[test]
    fn extract_tool_call_handles_openai_arguments_string() {
        use crate::qwen3::extract_first_tool_call;
        let s = r#"{"tool_calls":[{"function":{"name":"f","arguments":"{\"a\":1}"}}]}"#;
        let tc = extract_first_tool_call(s).unwrap();
        assert_eq!(tc.name, "f");
        assert_eq!(tc.arguments.get("a"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn extract_tool_call_handles_inline_prose_then_json() {
        use crate::qwen3::extract_first_tool_call;
        let s = "let me search.\n{\"name\":\"search\",\"arguments\":{\"q\":\"x\"}}\nokay.";
        let tc = extract_first_tool_call(s).unwrap();
        assert_eq!(tc.name, "search");
        assert_eq!(tc.arguments.get("q"), Some(&serde_json::json!("x")));
    }

    #[test]
    fn extract_tool_call_handles_tool_call_fence() {
        use crate::qwen3::extract_first_tool_call;
        let s = "I'll run a search.\n```tool_call\n{\"name\":\"search\",\"arguments\":{\"q\":\"hi\"}}\n```\n";
        let tc = extract_first_tool_call(s).unwrap();
        assert_eq!(tc.name, "search");
    }

    #[test]
    fn extract_tool_call_handles_qwen3_xml() {
        use crate::qwen3::extract_first_tool_call;
        let s = "<tool_call>\n<function=search>\n<parameter=q>\nhi\n</parameter>\n</function>\n</tool_call>";
        let tc = extract_first_tool_call(s).unwrap();
        assert_eq!(tc.name, "search");
        assert_eq!(tc.arguments.get("q"), Some(&serde_json::json!("hi")));
    }

    #[test]
    fn xml_completion_scores_against_json_target() {
        // Real Qwen3.5 output is XML; suite targets are typically stored
        // as canonical JSON. The scorer must compare the structured form.
        let target = r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}"#;
        let pred = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let (s, kind, _) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Structural,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert!(s > 0.95, "score was {s}");
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn thinking_before_xml_call_is_ignored() {
        // Numeric XML args coerce to JSON numbers (matches what the chat
        // template emits from a source-side int).
        let target = r#"{"tool_calls":[{"name":"f","arguments":{"x":1}}]}"#;
        let pred = "<think>\nI should call f with x=1.\n</think>\n\n<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
        let (s, kind, _) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Structural,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert!(s > 0.95);
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn extra_predicted_call_penalizes_score() {
        let target = r#"{"tool_calls":[{"name":"f","arguments":{}}]}"#;
        // Two XML calls when one was expected.
        let pred = "<tool_call>\n<function=f>\n</function>\n</tool_call>\n<tool_call>\n<function=g>\n</function>\n</tool_call>";
        let (s, _, detail) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Structural,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert!(detail.as_deref().unwrap().contains("excess_calls=1"));
        // Single excess subtracts 0.25/1 = 0.25 from the per-pair score.
        assert!(s < 1.0, "score was {s}");
    }

    #[test]
    fn multi_call_target_pairs_in_order() {
        let target =
            r#"{"tool_calls":[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]}"#;
        let pred = "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>";
        let (_, kind, _) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Structural,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn missing_second_call_drops_score() {
        let target =
            r#"{"tool_calls":[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]}"#;
        let pred = "<tool_call>\n<function=a>\n</function>\n</tool_call>";
        let (s, kind, detail) = score(
            &ex(target),
            pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Structural,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
        assert!(s < 0.6, "score was {s}");
        assert!(detail.as_deref().unwrap().contains("missing predicted call"));
    }

    #[test]
    fn no_tool_call_is_invalid() {
        let (_, kind, _) = score(
            &ex(r#"{"tool_calls":[{"name":"f","arguments":{}}]}"#),
            "plain prose answer",
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Structural,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Invalid);
    }
}
