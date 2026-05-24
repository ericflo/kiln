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
            ArgsScoring::PerKey { scorers, .. } => scorers.values().any(|s| s.requires_judge()),
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
/// Max shell-wrapper recursion while scoring command args. Production agents
/// often wrap the real operation in `bash -lc`, which then invokes Python or
/// another shell. A shallow bound avoids pathological self-similar strings.
const BASH_RECURSION_LIMIT: usize = 4;

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
    let target_raw = example
        .target
        .as_deref()
        .ok_or(ScorerError::MissingTarget { kind: "tool_call" })?;
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
            Some("non-XML tool call emitted (require_xml_format=true)".to_string()),
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
    let total_weight = if total_weight <= 0.0 {
        1.0
    } else {
        total_weight
    };
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
    let total_weight = if total_weight <= 0.0 {
        1.0
    } else {
        total_weight
    };
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

fn score_name(predicted: &str, target: &str, match_mode: &NameMatch) -> (f32, Option<String>) {
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
            let ok = allowed.iter().any(|a| a.eq_ignore_ascii_case(predicted));
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

fn score_structural(
    predicted: &serde_json::Value,
    target: &serde_json::Value,
) -> (f32, Option<String>) {
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
            missing
                .iter()
                .copied()
                .copied()
                .collect::<Vec<&str>>()
                .join(","),
            extra
                .iter()
                .copied()
                .copied()
                .collect::<Vec<&str>>()
                .join(",")
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
            if a == b {
                1.0
            } else {
                0.0
            }
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
/// 2. Recursively unwrap shell wrappers like `bash -lc` when they carry the
///    real command.
/// 3. If classifications differ (e.g. `python_inline` vs `pip_install`),
///    return a low score (0.1) — the model picked the wrong tool entirely.
/// 4. If both are inline programs of the same language, sub-score the
///    inner code with token similarity (lenient — variable renames OK).
/// 5. Otherwise return token jaccard over the full command tail.
fn score_bash_command(target: &str, predicted: &str) -> f32 {
    score_bash_command_depth(target, predicted, 0)
}

fn score_bash_command_depth(target: &str, predicted: &str, depth: usize) -> f32 {
    let t_intro = bash::introspect(target);
    let p_intro = bash::introspect(predicted);
    if depth < BASH_RECURSION_LIMIT {
        if let Some(score) = score_shell_wrapper_pair(&t_intro, &p_intro, target, predicted, depth)
        {
            return score;
        }
    }
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
        if t_intro.inline_language.as_deref() == Some("bash")
            && p_intro.inline_language.as_deref() == Some("bash")
            && depth < BASH_RECURSION_LIMIT
        {
            return code_token_similarity(t_code, p_code).max(score_bash_command_depth(
                t_code,
                p_code,
                depth + 1,
            ));
        }
        if t_intro.inline_language.as_deref() == Some("python")
            && p_intro.inline_language.as_deref() == Some("python")
        {
            return score_python_inline_code(t_code, p_code, depth);
        }
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

fn score_shell_wrapper_pair(
    target_intro: &bash::BashIntrospection,
    predicted_intro: &bash::BashIntrospection,
    target_raw: &str,
    predicted_raw: &str,
    depth: usize,
) -> Option<f32> {
    let target_shell = shell_wrapper_inner(target_intro);
    let predicted_shell = shell_wrapper_inner(predicted_intro);
    match (target_shell, predicted_shell) {
        (Some(t_inner), Some(p_inner)) => {
            let raw_similarity = code_token_similarity(t_inner, p_inner);
            let recursive = score_bash_command_depth(t_inner, p_inner, depth + 1);
            Some(raw_similarity.max(recursive))
        }
        (Some(t_inner), None) => Some(score_bash_command_depth(t_inner, predicted_raw, depth + 1)),
        (None, Some(p_inner)) => Some(score_bash_command_depth(target_raw, p_inner, depth + 1)),
        (None, None) => None,
    }
}

fn shell_wrapper_inner(intro: &bash::BashIntrospection) -> Option<&str> {
    match intro.inline_language.as_deref() {
        Some("bash") => intro.inline_code.as_deref(),
        _ => None,
    }
}

fn score_python_inline_code(target: &str, predicted: &str, depth: usize) -> f32 {
    let base = code_token_similarity(target, predicted);
    let mut semantic_scores = Vec::new();

    let target_commands = extract_python_shell_commands(target);
    let predicted_commands = extract_python_shell_commands(predicted);
    if !target_commands.is_empty() || !predicted_commands.is_empty() {
        semantic_scores.push(score_command_list(
            &target_commands,
            &predicted_commands,
            depth,
        ));
    }

    let target_exec_paths = extract_python_exec_paths(target);
    let predicted_exec_paths = extract_python_exec_paths(predicted);
    if !target_exec_paths.is_empty() || !predicted_exec_paths.is_empty() {
        semantic_scores.push(score_path_list(&target_exec_paths, &predicted_exec_paths));
    }

    if semantic_scores.is_empty() {
        return base;
    }

    let semantic = semantic_scores.iter().sum::<f32>() / semantic_scores.len() as f32;
    (semantic * 0.75 + base * 0.25).clamp(0.0, 1.0)
}

fn score_command_list(target: &[String], predicted: &[String], depth: usize) -> f32 {
    if target.is_empty() && predicted.is_empty() {
        return 1.0;
    }
    if target.is_empty() || predicted.is_empty() {
        return 0.0;
    }
    let total = target.len().max(predicted.len()).max(1);
    let mut sum = 0.0;
    for i in 0..total {
        let Some(t) = target.get(i) else {
            continue;
        };
        let Some(p) = predicted.get(i) else {
            continue;
        };
        sum += score_bash_command_depth(t, p, depth + 1);
    }
    sum / total as f32
}

fn score_path_list(target: &[String], predicted: &[String]) -> f32 {
    if target.is_empty() && predicted.is_empty() {
        return 1.0;
    }
    if target.is_empty() || predicted.is_empty() {
        return 0.0;
    }
    let total = target.len().max(predicted.len()).max(1);
    let mut sum = 0.0;
    for i in 0..total {
        let Some(t) = target.get(i) else {
            continue;
        };
        let Some(p) = predicted.get(i) else {
            continue;
        };
        sum += score_path(t, p);
    }
    sum / total as f32
}

fn score_path(target: &str, predicted: &str) -> f32 {
    let t = normalize_path_like(target);
    let p = normalize_path_like(predicted);
    if t == p {
        return 1.0;
    }
    let t_base = t.rsplit('/').next().unwrap_or(t.as_str());
    let p_base = p.rsplit('/').next().unwrap_or(p.as_str());
    if !t_base.is_empty() && t_base == p_base {
        return 0.8;
    }
    0.0
}

fn normalize_path_like(path: &str) -> String {
    let mut out = path.trim().replace('\\', "/");
    while out.starts_with("./") {
        out = out[2..].to_string();
    }
    while out.contains("//") {
        out = out.replace("//", "/");
    }
    out
}

fn extract_python_shell_commands(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    for pattern in [
        "os.system(",
        "os.popen(",
        "subprocess.run(",
        "subprocess.call(",
        "subprocess.check_output(",
        "subprocess.Popen(",
        "Popen(",
    ] {
        let mut search_from = 0usize;
        while let Some(rel) = code[search_from..].find(pattern) {
            let match_start = search_from + rel;
            if pattern == "Popen(" && has_qualified_python_call_prefix(code, match_start) {
                search_from = match_start + pattern.len();
                continue;
            }
            let args_start = match_start + pattern.len();
            if let Some((args, end)) = extract_balanced_call_args(code, args_start) {
                if let Some(command) = python_command_from_call_args(&args) {
                    out.push(command);
                }
                search_from = end;
            } else {
                break;
            }
        }
    }
    out
}

fn has_qualified_python_call_prefix(code: &str, match_start: usize) -> bool {
    code[..match_start]
        .chars()
        .next_back()
        .map(|ch| ch == '.' || ch == '_' || ch.is_ascii_alphanumeric())
        .unwrap_or(false)
}

fn python_command_from_call_args(args: &str) -> Option<String> {
    let trimmed = args.trim_start();
    if trimmed.starts_with('[') {
        return python_string_list(trimmed)
            .and_then(|parts| python_command_from_string_list(&parts));
    }
    first_python_string_literal(trimmed).map(|(s, _)| s)
}

fn python_command_from_string_list(parts: &[String]) -> Option<String> {
    let program = parts.first()?.as_str();
    if matches!(program, "bash" | "sh" | "zsh") && parts.len() >= 3 && parts[1].contains('c') {
        return Some(parts[2].clone());
    }
    Some(parts.join(" "))
}

fn extract_python_exec_paths(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    for pattern in ["exec(open(", "runpy.run_path(", "Path("] {
        let mut search_from = 0usize;
        while let Some(rel) = code[search_from..].find(pattern) {
            let args_start = search_from + rel + pattern.len();
            if let Some((path, end)) = first_python_string_literal(&code[args_start..]) {
                let suffix = code[args_start + end..].trim_start();
                let suffix = suffix.strip_prefix(')').unwrap_or(suffix).trim_start();
                if pattern != "Path(" || suffix.starts_with(".read_text") {
                    out.push(path);
                }
                search_from = args_start + end;
            } else {
                break;
            }
        }
    }
    out
}

fn extract_balanced_call_args(code: &str, start: usize) -> Option<(String, usize)> {
    let mut depth = 1usize;
    let mut in_string: Option<char> = None;
    let mut escaped = false;
    let mut out = String::new();
    let mut iter = code[start..].char_indices().peekable();
    while let Some((rel, ch)) = iter.next() {
        if let Some(quote) = in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                in_string = Some(ch);
                out.push(ch);
            }
            '(' | '[' | '{' => {
                depth += 1;
                out.push(ch);
            }
            ')' | ']' | '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some((out, start + rel + ch.len_utf8()));
                }
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    None
}

fn python_string_list(input: &str) -> Option<Vec<String>> {
    let mut strings = Vec::new();
    let mut idx = input.find('[')? + 1;
    while idx < input.len() {
        let rest = &input[idx..];
        let skipped = rest.len()
            - rest
                .trim_start_matches(|ch: char| ch.is_whitespace() || ch == ',')
                .len();
        idx += skipped;
        let rest = &input[idx..];
        if rest.starts_with(']') {
            return Some(strings);
        }
        if let Some((s, end)) = python_string_literal_at(input, idx) {
            strings.push(s);
            idx = end;
        } else {
            return None;
        }
    }
    None
}

fn first_python_string_literal(input: &str) -> Option<(String, usize)> {
    let bytes = input.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if let Some(parsed) = python_string_literal_at(input, i) {
            return Some(parsed);
        }
        i += 1;
    }
    None
}

fn python_string_literal_at(input: &str, start: usize) -> Option<(String, usize)> {
    let bytes = input.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    let mut quote_idx = start;
    let ch = bytes[quote_idx] as char;
    if ch != '\'' && ch != '"' {
        if !is_python_string_prefix(ch) {
            return None;
        }
        let mut prefix_len = 0usize;
        while quote_idx < bytes.len()
            && prefix_len < 3
            && is_python_string_prefix(bytes[quote_idx] as char)
        {
            quote_idx += 1;
            prefix_len += 1;
        }
        if quote_idx >= bytes.len() {
            return None;
        }
        let quote = bytes[quote_idx] as char;
        if quote != '\'' && quote != '"' {
            return None;
        }
    }
    parse_python_string_literal(input, quote_idx)
}

fn is_python_string_prefix(ch: char) -> bool {
    matches!(ch, 'r' | 'R' | 'u' | 'U' | 'b' | 'B' | 'f' | 'F')
}

fn parse_python_string_literal(input: &str, quote_idx: usize) -> Option<(String, usize)> {
    let quote = input[quote_idx..].chars().next()?;
    let quote_len = quote.len_utf8();
    let triple_delim = quote.to_string().repeat(3);
    let triple = input[quote_idx..].starts_with(&triple_delim);
    let content_start = quote_idx + if triple { quote_len * 3 } else { quote_len };
    let mut escaped = false;
    let mut out = String::new();
    let mut iter = input[content_start..].char_indices();
    for (rel, ch) in &mut iter {
        let abs = content_start + rel;
        if triple && input[abs..].starts_with(&triple_delim) {
            return Some((out, abs + quote_len * 3));
        }
        if escaped {
            out.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if !triple && ch == quote {
            return Some((out, abs + ch.len_utf8()));
        }
        out.push(ch);
    }
    None
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

    fn score_bash_tool_commands(
        target_command: &str,
        pred_command: &str,
    ) -> (f32, EvalOutcomeKind, Option<String>) {
        let target = serde_json::json!({
            "tool_calls": [{
                "name": "Bash",
                "arguments": {"command": target_command}
            }]
        })
        .to_string();
        let pred = format!(
            "<tool_call>\n<function=Bash>\n<parameter=command>\n{pred_command}\n</parameter>\n</function>\n</tool_call>"
        );
        score(
            &ex(&target),
            &pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Auto,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap()
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
        let target =
            r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris","units":"c"}}]}"#;
        let pred =
            r#"{"tool_call":{"name":"get_weather","arguments":{"city":"Paris","units":"c"}}}"#;
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
                style: crate::scorers::CodeStyle::TokenSimilarity { min_jaccard: 0.6 },
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
    fn auto_args_recursively_score_bash_python_heredoc() {
        let target_command = "bash -lc \"python3 - <<'PY'\nimport os\nprint(os.getcwd())\nPY\"";
        let pred_command = "python3 -c 'import os\nprint(os.getcwd())'";
        let target = serde_json::json!({
            "tool_calls": [{
                "name": "Bash",
                "arguments": {"command": target_command}
            }]
        })
        .to_string();
        let pred = format!(
            "<tool_call>\n<function=Bash>\n<parameter=command>\n{pred_command}\n</parameter>\n</function>\n</tool_call>"
        );
        let (s, kind, detail) = score(
            &ex(&target),
            &pred,
            &NameMatch::CaseInsensitive,
            &ArgsScoring::Auto,
            None,
            false,
            &NoopJudgeRunner,
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass, "{detail:?}");
        assert!(s > 0.95, "score was {s}; detail={detail:?}");
    }

    #[test]
    fn auto_args_fail_wrong_nested_python_body() {
        let target_command = "bash -lc \"python3 - <<'PY'\nimport os\nprint(os.getcwd())\nPY\"";
        let pred_command = "python3 -c 'print(\"hello\")'";
        let (s, kind, _) = score_bash_tool_commands(target_command, pred_command);
        assert_eq!(kind, EvalOutcomeKind::Fail);
        assert!(s < 0.8, "score was {s}");
    }

    #[test]
    fn auto_args_score_python_subprocess_equivalent_shell_command() {
        let target_command = r#"python3 -c 'import os; os.system("grep -R TODO src")'"#;
        let pred_command = r#"python3 -c 'import subprocess; subprocess.run(["bash", "-lc", "grep -R TODO src"], check=True)'"#;
        let (s, kind, detail) = score_bash_tool_commands(target_command, pred_command);
        assert_eq!(kind, EvalOutcomeKind::Pass, "{detail:?}");
        assert!(s > 0.9, "score was {s}; detail={detail:?}");
    }

    #[test]
    fn auto_args_fail_wrong_python_subprocess_shell_command() {
        let target_command = r#"python3 -c 'import os; os.system("grep -R TODO src")'"#;
        let pred_command =
            r#"python3 -c 'import subprocess; subprocess.run(["bash", "-lc", "rm -rf /tmp/x"])'"#;
        let (s, kind, _) = score_bash_tool_commands(target_command, pred_command);
        assert_eq!(kind, EvalOutcomeKind::Fail);
        assert!(s < 0.8, "score was {s}");
    }

    #[test]
    fn auto_args_score_python_exec_equivalent_path() {
        let target_command = r#"python3 -c 'exec(open("scripts/materialize_turn.py").read())'"#;
        let pred_command = r#"python3 -c 'from pathlib import Path; exec(Path("./scripts/materialize_turn.py").read_text())'"#;
        let (s, kind, detail) = score_bash_tool_commands(target_command, pred_command);
        assert_eq!(kind, EvalOutcomeKind::Pass, "{detail:?}");
        assert!(s > 0.9, "score was {s}; detail={detail:?}");
    }

    #[test]
    fn auto_args_fail_python_exec_different_path() {
        let target_command = r#"python3 -c 'exec(open("scripts/materialize_turn.py").read())'"#;
        let pred_command = r#"python3 -c 'exec(open("totally/unrelated_runner.py").read())'"#;
        let (s, kind, _) = score_bash_tool_commands(target_command, pred_command);
        assert_eq!(kind, EvalOutcomeKind::Fail);
        assert!(s < 0.8, "score was {s}");
    }

    #[test]
    fn python_string_list_rejects_dynamic_command_tail() {
        assert!(python_string_list(r#"["bash", "-lc", command]"#).is_none());
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
        let target = r#"{"tool_calls":[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]}"#;
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
        let target = r#"{"tool_calls":[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]}"#;
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
        assert!(
            detail
                .as_deref()
                .unwrap()
                .contains("missing predicted call")
        );
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
