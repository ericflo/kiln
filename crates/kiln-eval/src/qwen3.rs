//! Qwen3.5 chat-format helpers.
//!
//! Kiln only targets Qwen3.5-4B, so the eval system can be *exact* about the
//! wire format the model produces and *exact* about how to interpret model
//! output. This module is the single source of truth for:
//!
//! - **Thinking blocks.** Qwen3.5's chat template prefills `<think>\n` into
//!   the assistant turn whenever `enable_thinking` isn't explicitly false,
//!   so the model continues directly with chain-of-thought and closes with
//!   `</think>` before the actual answer. `split_thinking` separates the
//!   two so scorers always grade the *answer*, not the reasoning. The bare
//!   `disable_thinking=true` path emits an empty `<think>\n\n</think>\n\n`
//!   so the same splitter still works.
//! - **Tool calls.** Qwen3.5's *native* tool-call wire form is XML —
//!   `<tool_call><function=name><parameter=key>value</parameter>…</function></tool_call>`
//!   — NOT the JSON shape OpenAI ships. Every eval that grades tool calling
//!   has to parse that, plus all the other shapes that can plausibly appear
//!   in completions (JSON canonical, fenced ```tool_call```, OpenAI
//!   `tool_calls[*].function`). `extract_tool_calls` is the canonicalizer:
//!   format-in, structured-out, with strong precedence rules so we never
//!   silently score "the JSON I'd mention in my reasoning" as the model's
//!   real tool call.
//! - **Tool responses.** The chat template wraps `tool`-role messages in
//!   `<tool_response>…</tool_response>` inside a user turn. `strip_tool_response_wrapper`
//!   unwraps these for synthesis when we want to feed a tool-response payload
//!   back into an eval prompt.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Opening tag the chat template emits when reasoning is enabled.
pub const THINK_OPEN: &str = "<think>";
/// Closing tag the model emits before its answer.
pub const THINK_CLOSE: &str = "</think>";

/// Result of stripping the `<think>…</think>` block from a model completion.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThinkingSplit<'a> {
    /// The portion of `raw` *inside* the first `<think>…</think>` block.
    /// `None` when no opening tag was found, or when no closing tag was
    /// emitted (which usually means the model never finished thinking —
    /// the answer field will then carry the full raw text so scorers can
    /// still salvage signal).
    pub reasoning: Option<&'a str>,
    /// The completion text *after* `</think>`, ready to be passed to
    /// scorers. When no thinking block was emitted, this is just the raw
    /// completion trimmed of any leading whitespace.
    pub answer: &'a str,
    /// True when an opening `<think>` was observed in `raw` — i.e. the
    /// completion came from a thinking-enabled prompt.
    pub had_thinking: bool,
    /// True when an unclosed thinking block was detected (no `</think>`
    /// followed an opening `<think>`). Scorers can use this to mark the
    /// outcome as Invalid rather than Fail.
    pub unclosed: bool,
}

impl<'a> ThinkingSplit<'a> {
    /// Convenience: return `answer` (the post-`</think>` body).
    pub fn answer(&self) -> &'a str {
        self.answer
    }

    /// Returns `reasoning` if present, else the empty string.
    pub fn reasoning_str(&self) -> &'a str {
        self.reasoning.unwrap_or("")
    }
}

/// Split a model completion into `(reasoning, answer)` around the Qwen3.5
/// `<think>…</think>` boundary.
///
/// Behaviour:
/// - When the completion contains `<think>` followed by `</think>`, returns
///   the contents inside the tags as `reasoning` and the text after the
///   closing tag (left-trimmed of one optional newline) as `answer`.
/// - When the completion contains `<think>` but no `</think>`, marks the
///   split as `unclosed` and returns the raw text after `<think>` as
///   reasoning, with `answer` set to the empty string. The caller decides
///   whether to grade this as Fail or Invalid.
/// - When no opening `<think>` tag is present (e.g. the model was prompted
///   with `enable_thinking=false`, so the template prefilled
///   `<think>\n\n</think>\n\n` and the engine never re-emits an opener),
///   `had_thinking` is `false` and `answer` is the full input.
///
/// The Qwen3.5 server pipeline strips the empty `<think>\n\n</think>\n\n`
/// prefix *before* shipping bytes to clients (kiln-server's reasoning
/// splitter handles that), but eval examples can come from anywhere —
/// upstream datasets, fixture files, recorded responses — so this helper
/// must remain tolerant of every shape.
pub fn split_thinking(raw: &str) -> ThinkingSplit<'_> {
    let trimmed = raw.trim_start();
    let leading_offset = raw.len() - trimmed.len();
    let _ = leading_offset; // kept for future positional info

    let Some(open_idx) = raw.find(THINK_OPEN) else {
        return ThinkingSplit {
            reasoning: None,
            answer: trim_one_leading_newline(trimmed),
            had_thinking: false,
            unclosed: false,
        };
    };

    // Reject any "<think>" buried in normal prose — only accept the opener
    // when it's near the start of the completion (Qwen3.5 always emits it
    // there because the prompt ended with the open tag). Allow up to a few
    // whitespace / leading-newline characters before the open tag so we
    // remain tolerant of formatting variation. A stricter check would just
    // be `raw[..open_idx].trim().is_empty()`.
    if !raw[..open_idx].trim().is_empty() {
        return ThinkingSplit {
            reasoning: None,
            answer: trim_one_leading_newline(trimmed),
            had_thinking: false,
            unclosed: false,
        };
    }

    let body_start = open_idx + THINK_OPEN.len();
    let body = &raw[body_start..];

    let Some(close_off) = body.find(THINK_CLOSE) else {
        // Unclosed — the model never emitted `</think>`. Return the rest as
        // reasoning so scorers can at least see what happened.
        return ThinkingSplit {
            reasoning: Some(trim_thinking_body(body)),
            answer: "",
            had_thinking: true,
            unclosed: true,
        };
    };

    let reasoning_raw = &body[..close_off];
    let after_close = &body[close_off + THINK_CLOSE.len()..];
    ThinkingSplit {
        reasoning: Some(trim_thinking_body(reasoning_raw)),
        answer: trim_one_leading_newline(after_close.trim_start_matches('\n')),
        had_thinking: true,
        unclosed: false,
    }
}

fn trim_one_leading_newline(s: &str) -> &str {
    // Don't trim arbitrary whitespace — answers often start with a leading
    // space (e.g. " Paris"); we only want to strip the single newline the
    // template puts after `</think>`.
    if let Some(rest) = s.strip_prefix("\n\n") {
        rest
    } else if let Some(rest) = s.strip_prefix('\n') {
        rest
    } else {
        s
    }
}

fn trim_thinking_body(s: &str) -> &str {
    s.trim_matches('\n')
}

/// Parsed tool call ready for scoring. Format-agnostic: both Qwen3.5's
/// native XML form and OpenAI's JSON shape canonicalize to this.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedToolCall {
    pub name: String,
    /// Argument values keyed by name. Always a JSON `Value` so structural
    /// comparison and per-key scoring are uniform regardless of upstream
    /// format. Native Qwen3.5 XML parameters are strings unless they parse
    /// as JSON (objects/arrays); `OpenAI {"arguments": "<json string>"}` is
    /// decoded into structured values; canonical `{"arguments": {…}}` is
    /// kept as-is.
    pub arguments: serde_json::Map<String, serde_json::Value>,
    /// Original on-the-wire format — useful for diagnostics and for telling
    /// the user "your model wrote a JSON-shaped tool call but Qwen3.5 native
    /// is XML."
    pub format: ToolCallFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallFormat {
    /// `<tool_call><function=…><parameter=…>…</parameter></function></tool_call>` — Qwen3.5 native.
    Qwen3Xml,
    /// Inline JSON object `{"name": "…", "arguments": {…}}`.
    JsonInline,
    /// OpenAI-style `{"tool_calls": [{"function": {"name", "arguments"}}]}`.
    OpenAi,
    /// Fenced ```` ```tool_call ``` ```` block.
    Fenced,
}

impl ParsedToolCall {
    /// Render the parsed call back into Qwen3.5's native XML form. Useful
    /// when generating synthetic targets so eval-target text matches what
    /// the model would emit.
    pub fn to_qwen3_xml(&self) -> String {
        let mut out = String::new();
        out.push_str("<tool_call>\n<function=");
        out.push_str(&self.name);
        out.push_str(">\n");
        for (k, v) in &self.arguments {
            out.push_str("<parameter=");
            out.push_str(k);
            out.push_str(">\n");
            out.push_str(&format_xml_param_value(v));
            out.push_str("\n</parameter>\n");
        }
        out.push_str("</function>\n</tool_call>");
        out
    }

    /// Render to canonical JSON for storage/comparison. Keys are sorted so
    /// stringified targets are stable across runs.
    pub fn to_canonical_json(&self) -> serde_json::Value {
        let mut sorted: BTreeMap<&str, serde_json::Value> = BTreeMap::new();
        for (k, v) in &self.arguments {
            sorted.insert(k.as_str(), v.clone());
        }
        serde_json::json!({
            "name": self.name,
            "arguments": sorted.into_iter().map(|(k, v)| (k.to_string(), v)).collect::<serde_json::Map<_, _>>(),
        })
    }
}

fn format_xml_param_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Extract every tool call from a Qwen3.5 model completion in order.
/// Tries every wire form Kiln has seen:
///
/// 1. Qwen3.5 native XML: `<tool_call><function=…>…</function></tool_call>`.
/// 2. Canonical JSON: `{"tool_calls": [{"name", "arguments"}]}`.
/// 3. Fenced ```` ```tool_call``` ```` block containing JSON.
/// 4. OpenAI-style `{"tool_calls":[{"function":{"name", "arguments": "<json>"}}]}`.
/// 5. Bare inline JSON object `{"name": "…", "arguments": {…}}`.
///
/// Strips `<think>…</think>` before parsing so reasoning prose can't trip
/// the JSON detector. When the same completion happens to contain both an
/// XML and a JSON tool call (e.g. the model emitted XML *and* mentioned the
/// shape in reasoning), the XML wins because it's Qwen3.5's contractual
/// output.
pub fn extract_tool_calls(raw: &str) -> Vec<ParsedToolCall> {
    let split = split_thinking(raw);
    let scan = if split.had_thinking { split.answer } else { raw };

    let xml = extract_qwen3_xml_tool_calls(scan);
    if !xml.is_empty() {
        return xml;
    }
    if let Some(calls) = extract_fenced_tool_calls(scan) {
        if !calls.is_empty() {
            return calls;
        }
    }
    if let Some(calls) = extract_json_tool_calls(scan) {
        return calls;
    }
    Vec::new()
}

/// Convenience: extract just the *first* tool call. Most scorers grade one
/// call at a time, so this is the common path.
pub fn extract_first_tool_call(raw: &str) -> Option<ParsedToolCall> {
    extract_tool_calls(raw).into_iter().next()
}

/// Parse every `<tool_call>…</tool_call>` block in `text`. Returns an empty
/// vec when none are found.
fn extract_qwen3_xml_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        let Some(open_rel) = text[cursor..].find("<tool_call>") else {
            break;
        };
        let open_abs = cursor + open_rel + "<tool_call>".len();
        let Some(close_rel) = text[open_abs..].find("</tool_call>") else {
            break;
        };
        let close_abs = open_abs + close_rel;
        let body = &text[open_abs..close_abs];
        if let Some(call) = parse_qwen3_xml_body(body) {
            out.push(call);
        }
        cursor = close_abs + "</tool_call>".len();
    }
    out
}

fn parse_qwen3_xml_body(body: &str) -> Option<ParsedToolCall> {
    // Expect: optional whitespace, `<function=NAME>`, parameters, `</function>`
    let trimmed = body.trim();
    let after_fn = trimmed.strip_prefix("<function=")?;
    let name_end = after_fn.find('>')?;
    let name = after_fn[..name_end].trim().to_string();
    let rest = &after_fn[name_end + 1..];
    // Body inside <function=…> … </function>
    let inner_end = rest.find("</function>")?;
    let inner = &rest[..inner_end];

    let mut arguments: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut cur = 0usize;
    while cur < inner.len() {
        let Some(open_rel) = inner[cur..].find("<parameter=") else {
            break;
        };
        let open_abs = cur + open_rel + "<parameter=".len();
        let Some(close_name_rel) = inner[open_abs..].find('>') else {
            break;
        };
        let close_name_abs = open_abs + close_name_rel;
        let param_name = inner[open_abs..close_name_abs].trim().to_string();
        let value_start = close_name_abs + 1;
        let Some(end_rel) = inner[value_start..].find("</parameter>") else {
            break;
        };
        let end_abs = value_start + end_rel;
        // Strip a single leading newline (the template emits `>\n`) and a
        // single trailing newline before `</parameter>`.
        let raw_value = &inner[value_start..end_abs];
        let value = trim_one_leading_newline(raw_value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        arguments.insert(param_name, parse_xml_value(value));
        cur = end_abs + "</parameter>".len();
    }
    Some(ParsedToolCall {
        name,
        arguments,
        format: ToolCallFormat::Qwen3Xml,
    })
}

/// XML parameter values are raw strings. The chat template renders nested
/// objects/lists via `tojson`, but primitives go through `value | string`,
/// so a bool `false` is emitted as the literal token `False` (Python's
/// `str(False)`) and an int `42` as `"42"`. To make structural comparison
/// against JSON targets work, we coerce stringified primitives back into
/// their JSON form when the value is unambiguously a primitive (single
/// short token with no whitespace).
fn parse_xml_value(s: &str) -> serde_json::Value {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return serde_json::Value::String(String::new());
    }
    // Object / array: try strict JSON parse first.
    let first = trimmed.chars().next().unwrap();
    if first == '{' || first == '[' {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return v;
        }
    }
    // Bool / null primitives — accept Python-style ("True"/"False"/"None")
    // and JSON-style ("true"/"false"/"null") tokens emitted by the chat
    // template. Only kick in when the *entire* trimmed body is the token,
    // so prose values that happen to contain "false" mid-sentence stay
    // strings.
    if !trimmed.contains('\n') {
        match trimmed {
            "true" | "True" => return serde_json::Value::Bool(true),
            "false" | "False" => return serde_json::Value::Bool(false),
            "null" | "None" => return serde_json::Value::Null,
            _ => {}
        }
        // Number coercion — only when the *entire* trimmed body is one
        // numeric token. We reject leading-zero non-numbers like "0123"
        // by going through serde_json's parser which preserves precision.
        if looks_like_number(trimmed) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if v.is_number() {
                    return v;
                }
            }
        }
    }
    serde_json::Value::String(s.to_string())
}

fn looks_like_number(s: &str) -> bool {
    let mut chars = s.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if !(first.is_ascii_digit() || first == '-' || first == '+') {
        return false;
    }
    let mut saw_digit = first.is_ascii_digit();
    for c in chars {
        if c.is_ascii_digit() {
            saw_digit = true;
        } else if c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
            // allowed in numbers; we let serde_json validate the full shape
        } else {
            return false;
        }
    }
    saw_digit
}

/// Look for fenced ```` ```tool_call``` ```` blocks. Returns Some(vec) when at
/// least one fence is seen (even if its body fails to parse — the caller
/// then knows the model *tried* to call a tool).
fn extract_fenced_tool_calls(text: &str) -> Option<Vec<ParsedToolCall>> {
    let needle = "```tool_call";
    if !text.contains(needle) {
        return None;
    }
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(open_rel) = text[cursor..].find(needle) {
        let body_start = cursor + open_rel + needle.len();
        let after = &text[body_start..];
        // Skip optional newline / leading whitespace after the fence.
        let after = after.trim_start_matches('\n');
        let Some(close_rel) = after.find("```") else {
            break;
        };
        let body = after[..close_rel].trim();
        if let Some(call) = parse_inline_json_tool_call(body, ToolCallFormat::Fenced) {
            out.push(call);
        }
        cursor = body_start + after.len() - after[close_rel..].len() + 3;
    }
    Some(out)
}

/// Parse any JSON-shaped tool calls anywhere in `text`. Recognized:
/// - `{"tool_calls":[{…}]}` envelope (canonical, possibly with OpenAI's
///   `function.arguments` as a string).
/// - Bare `{"name":"…","arguments":{…}}` or `{"tool_call":{…}}` / `{"function":{…}}`.
fn extract_json_tool_calls(text: &str) -> Option<Vec<ParsedToolCall>> {
    let body = text.trim();
    if body.is_empty() {
        return None;
    }

    // First try a strict parse of the whole text — common case for bench
    // datasets that store the JSON answer verbatim.
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(v) = json_to_tool_calls(&parsed) {
            return Some(v);
        }
    }

    // Otherwise, scan for the first brace-balanced JSON object that looks
    // tool-call-shaped and parse it.
    if let Some(obj) = find_first_tool_call_object(text) {
        if let Some(v) = json_to_tool_calls(&obj) {
            return Some(v);
        }
    }
    None
}

fn json_to_tool_calls(value: &serde_json::Value) -> Option<Vec<ParsedToolCall>> {
    if let Some(arr) = value.get("tool_calls").and_then(|v| v.as_array()) {
        let mut out = Vec::new();
        for entry in arr {
            if let Some(call) = parse_canonical_tool_call_entry(entry) {
                out.push(call);
            }
        }
        if out.is_empty() { None } else { Some(out) }
    } else if let Some(call) = parse_canonical_tool_call_entry(value) {
        Some(vec![call])
    } else {
        None
    }
}

fn parse_canonical_tool_call_entry(value: &serde_json::Value) -> Option<ParsedToolCall> {
    let mut format = ToolCallFormat::JsonInline;
    let value = if let Some(inner) = value.get("function") {
        format = ToolCallFormat::OpenAi;
        inner
    } else if let Some(inner) = value.get("tool_call") {
        inner
    } else {
        value
    };
    let name = value.get("name").and_then(|v| v.as_str())?.to_string();
    let arguments = value
        .get("arguments")
        .or_else(|| value.get("input"))
        .or_else(|| value.get("parameters"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let arguments = decode_argument_blob(arguments);
    let arguments_obj = match arguments {
        serde_json::Value::Object(m) => m,
        // Tool calls without arguments are legal — coerce to empty.
        serde_json::Value::Null => serde_json::Map::new(),
        // Some templates store args as an array (positional). Wrap into
        // `{"args": [...]}` so per-key scorers can still work.
        other => {
            let mut m = serde_json::Map::new();
            m.insert("args".to_string(), other);
            m
        }
    };
    Some(ParsedToolCall {
        name,
        arguments: arguments_obj,
        format,
    })
}

fn parse_inline_json_tool_call(text: &str, format: ToolCallFormat) -> Option<ParsedToolCall> {
    let parsed: serde_json::Value = serde_json::from_str(text).ok()?;
    let mut call = parse_canonical_tool_call_entry(&parsed)?;
    call.format = format;
    Some(call)
}

/// OpenAI's wire form ships `arguments` as a JSON-encoded *string*. Parse it
/// back into a value when possible so per-key scoring sees structured data
/// instead of a string blob.
fn decode_argument_blob(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::String(s) => {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&s) {
                parsed
            } else {
                serde_json::Value::String(s)
            }
        }
        other => other,
    }
}

fn find_first_tool_call_object(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        // Quick lookahead — must mention a tool-call-shaped key within a
        // short window or this is some other JSON.
        let lookahead_end = bytes.len().min(i + 384);
        let head = &text[i..lookahead_end];
        if !(head.contains("\"name\"")
            || head.contains("\"function\"")
            || head.contains("\"tool_call\"")
            || head.contains("\"tool_calls\""))
        {
            i += 1;
            continue;
        }
        // Brace-balanced extraction. JSON-aware: ignore braces inside
        // strings.
        let mut depth = 0i64;
        let mut in_str = false;
        let mut esc = false;
        for j in i..bytes.len() {
            let c = bytes[j];
            if esc {
                esc = false;
                continue;
            }
            if in_str {
                match c {
                    b'\\' => esc = true,
                    b'"' => in_str = false,
                    _ => {}
                }
                continue;
            }
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        let body = &text[i..=j];
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
                            return Some(v);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Unwrap the `<tool_response>…</tool_response>` envelope Qwen3.5's chat
/// template puts around tool replies. Returns `None` when the wrapper isn't
/// present, so callers can pass any content through unchanged.
pub fn strip_tool_response_wrapper(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    let inner = trimmed.strip_prefix("<tool_response>")?;
    let inner = inner.strip_suffix("</tool_response>")?;
    Some(inner.trim_matches('\n'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_thinking_simple_block_separates_reasoning_and_answer() {
        let raw = "<think>\nLet me compute 2+2.\n</think>\n\nThe answer is 4.";
        let s = split_thinking(raw);
        assert!(s.had_thinking);
        assert!(!s.unclosed);
        assert_eq!(s.reasoning, Some("Let me compute 2+2."));
        assert_eq!(s.answer, "The answer is 4.");
    }

    #[test]
    fn split_thinking_no_open_tag_returns_full_text() {
        let raw = "The answer is 4.";
        let s = split_thinking(raw);
        assert!(!s.had_thinking);
        assert_eq!(s.answer, "The answer is 4.");
        assert!(s.reasoning.is_none());
    }

    #[test]
    fn split_thinking_open_no_close_marks_unclosed() {
        let raw = "<think>\nstill thinking";
        let s = split_thinking(raw);
        assert!(s.had_thinking);
        assert!(s.unclosed);
        assert_eq!(s.answer, "");
        assert_eq!(s.reasoning, Some("still thinking"));
    }

    #[test]
    fn split_thinking_open_in_middle_is_ignored() {
        // A `<think>` deep in the prose isn't a thinking opener; it's just
        // content. Don't pull it apart.
        let raw = "I will explain. <think> this is wrong";
        let s = split_thinking(raw);
        assert!(!s.had_thinking);
        assert_eq!(s.answer, raw);
    }

    #[test]
    fn split_thinking_empty_block_works() {
        // enable_thinking=false renders `<think>\n\n</think>\n\n` and the
        // model then emits its answer immediately. Most server pipelines
        // strip this before clients see it, but raw eval fixtures can keep
        // it intact.
        let raw = "<think>\n\n</think>\n\nParis";
        let s = split_thinking(raw);
        assert!(s.had_thinking);
        assert_eq!(s.reasoning, Some(""));
        assert_eq!(s.answer, "Paris");
    }

    #[test]
    fn extract_qwen3_xml_basic() {
        let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=units>\nc\n</parameter>\n</function>\n</tool_call>";
        let calls = extract_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].format, ToolCallFormat::Qwen3Xml);
        assert_eq!(
            calls[0].arguments.get("city"),
            Some(&serde_json::json!("Paris"))
        );
        assert_eq!(
            calls[0].arguments.get("units"),
            Some(&serde_json::json!("c"))
        );
    }

    #[test]
    fn extract_qwen3_xml_with_thinking_prefix() {
        let raw =
            "<think>\nI should call the weather API.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
        let calls = extract_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments.get("city"),
            Some(&serde_json::json!("Tokyo"))
        );
    }

    #[test]
    fn extract_qwen3_xml_multiple_calls() {
        let raw = "<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
        let calls = extract_tool_calls(raw);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        // Bare numeric tokens get coerced to JSON numbers (matches what
        // the chat template produces from a source-side `int` value).
        assert_eq!(calls[1].arguments.get("x"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn extract_qwen3_xml_decodes_nested_json_param() {
        // The chat template renders dict/list args via `tojson`. The model
        // therefore emits them as inline JSON in the parameter body. Decode
        // them so structural scoring works.
        let raw = "<tool_call>\n<function=set>\n<parameter=opts>\n{\"verbose\": true, \"limit\": 5}\n</parameter>\n</function>\n</tool_call>";
        let call = extract_first_tool_call(raw).unwrap();
        let opts = call.arguments.get("opts").unwrap();
        assert!(opts.is_object());
        assert_eq!(opts["verbose"], serde_json::json!(true));
        assert_eq!(opts["limit"], serde_json::json!(5));
    }

    #[test]
    fn extract_json_canonical_envelope() {
        let raw = r#"{"tool_calls":[{"name":"search","arguments":{"q":"hi"}}]}"#;
        let calls = extract_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].format, ToolCallFormat::JsonInline);
        assert_eq!(calls[0].arguments.get("q"), Some(&serde_json::json!("hi")));
    }

    #[test]
    fn extract_json_openai_arguments_string() {
        let raw = r#"{"tool_calls":[{"function":{"name":"f","arguments":"{\"a\":1}"}}]}"#;
        let call = extract_first_tool_call(raw).unwrap();
        assert_eq!(call.name, "f");
        assert_eq!(call.format, ToolCallFormat::OpenAi);
        assert_eq!(call.arguments.get("a"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn extract_fenced_tool_call_block() {
        let raw = "Sure, let me run it.\n```tool_call\n{\"name\":\"search\",\"arguments\":{\"q\":\"hi\"}}\n```\n";
        let calls = extract_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].format, ToolCallFormat::Fenced);
    }

    #[test]
    fn extract_inline_json_after_prose() {
        let raw = "Let me search.\n{\"name\":\"search\",\"arguments\":{\"q\":\"x\"}}\nok";
        let call = extract_first_tool_call(raw).unwrap();
        assert_eq!(call.name, "search");
    }

    #[test]
    fn extract_returns_empty_for_pure_prose() {
        let raw = "I don't think we need a tool for this.";
        assert!(extract_tool_calls(raw).is_empty());
    }

    #[test]
    fn extract_ignores_xml_inside_thinking() {
        // The model might mention the XML form in its reasoning. That
        // shouldn't be scored as the actual emitted call.
        let raw = "<think>\nIf I called <tool_call><function=f></function></tool_call> I would get…\n</think>\n\nNo tool needed.";
        assert!(extract_tool_calls(raw).is_empty());
    }

    #[test]
    fn to_qwen3_xml_roundtrips_simple_call() {
        let call = ParsedToolCall {
            name: "set".into(),
            arguments: serde_json::Map::from_iter([
                ("k".to_string(), serde_json::json!("v")),
                ("n".to_string(), serde_json::json!(7)),
            ]),
            format: ToolCallFormat::Qwen3Xml,
        };
        let rendered = call.to_qwen3_xml();
        // Round-trip: parse the rendered string and confirm equality on
        // name and key set (value types may flatten — XML params are
        // string-shaped for primitives).
        let parsed = extract_first_tool_call(&rendered).unwrap();
        assert_eq!(parsed.name, "set");
        assert_eq!(parsed.arguments.len(), 2);
        assert!(parsed.arguments.contains_key("k"));
        assert!(parsed.arguments.contains_key("n"));
    }

    #[test]
    fn strip_tool_response_wrapper_handles_wrapped_and_unwrapped() {
        assert_eq!(
            strip_tool_response_wrapper("<tool_response>\nhello\n</tool_response>"),
            Some("hello")
        );
        assert_eq!(strip_tool_response_wrapper("plain reply"), None);
    }

    #[test]
    fn xml_wins_over_json_when_both_present() {
        // The model emitted XML; a JSON-shaped string appears later in the
        // completion. Score the XML one.
        let raw = "<tool_call>\n<function=real>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>\n\nNote: don't write `{\"name\":\"fake\"}` instead.";
        let call = extract_first_tool_call(raw).unwrap();
        assert_eq!(call.name, "real");
    }
}
