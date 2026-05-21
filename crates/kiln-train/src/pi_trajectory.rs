//! Pi session JSONL normalization into kiln's canonical trajectory schema.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::ChatMessage;
use crate::trajectory::{ScoredRollout, TurnKind, TurnSegment};

#[derive(Debug, Clone, Default)]
pub struct PiSessionParse {
    pub prompt_messages: Vec<ChatMessage>,
    pub trajectory: Vec<TurnSegment>,
    pub warnings: Vec<String>,
}

pub fn parse_pi_session_jsonl(path: &Path, include_context: bool) -> Result<PiSessionParse> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Ok(parse_pi_session_str(&raw, include_context))
}

pub fn parse_pi_session_str(raw: &str, include_context: bool) -> PiSessionParse {
    let mut values = Vec::new();
    let mut warnings = Vec::new();
    for (line_idx, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => values.push(value),
            Err(err) => warnings.push(format!(
                "skipped malformed JSONL line {}: {err}",
                line_idx + 1
            )),
        }
    }
    parse_pi_session_values(&values, include_context, warnings)
}

pub fn parse_pi_session_values(
    values: &[Value],
    include_context: bool,
    mut warnings: Vec<String>,
) -> PiSessionParse {
    let mut prompt_messages = Vec::new();
    let mut trajectory = Vec::new();
    for value in values {
        if !is_pi_message_event(value) {
            continue;
        }
        let Some(message) = value.get("message").and_then(Value::as_object) else {
            warnings.push("skipped Pi message event with non-object message".to_string());
            continue;
        };
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let content = message.get("content").unwrap_or(&Value::Null);
        match role {
            "system" | "user" => {
                let text = stringify_content(content);
                if text.is_empty() {
                    continue;
                }
                if include_context {
                    trajectory.push(TurnSegment {
                        role: role.to_string(),
                        content: text,
                        kind: TurnKind::Context,
                        tool_call_id: None,
                        warning_prefix_len: None,
                    });
                } else {
                    prompt_messages.push(ChatMessage {
                        role: role.to_string(),
                        content: text,
                    });
                }
            }
            "assistant" => {
                let rendered = render_assistant_content(content, &mut warnings);
                if !rendered.is_empty() {
                    trajectory.push(TurnSegment {
                        role: "assistant".to_string(),
                        content: rendered,
                        kind: TurnKind::Action,
                        tool_call_id: None,
                        warning_prefix_len: None,
                    });
                }
            }
            "tool" | "toolResult" => {
                let (rendered, tool_call_id) = render_tool_result_content(content, &mut warnings);
                if !rendered.is_empty() {
                    trajectory.push(TurnSegment {
                        role: "tool".to_string(),
                        warning_prefix_len: detect_warning_prefix(&rendered),
                        content: rendered,
                        kind: TurnKind::Observation,
                        tool_call_id,
                    });
                }
            }
            other => warnings.push(format!("skipped Pi message with unknown role {other:?}")),
        }
    }
    PiSessionParse {
        prompt_messages,
        trajectory,
        warnings,
    }
}

pub fn is_pi_message_event(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("message") && value.get("message").is_some()
}

pub fn flatten_action_text(trajectory: &[TurnSegment]) -> String {
    trajectory
        .iter()
        .filter(|segment| segment.kind == TurnKind::Action)
        .map(|segment| segment.content.as_str())
        .collect::<Vec<_>>()
        .join("<TURN_BREAK>")
}

pub fn scored_rollout_from_pi_session(
    raw: &str,
    reward: f64,
    include_context: bool,
) -> ScoredRollout {
    let parsed = parse_pi_session_str(raw, include_context);
    let text = {
        let flattened = flatten_action_text(&parsed.trajectory);
        if flattened.is_empty() {
            "(empty)".to_string()
        } else {
            flattened
        }
    };
    ScoredRollout {
        text,
        reward,
        trajectory: parsed.trajectory,
    }
}

pub fn stringify_content(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                if let Some(text) = block.as_str() {
                    Some(text.to_string())
                } else {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn render_assistant_content(content: &Value, warnings: &mut Vec<String>) -> String {
    let Some(blocks) = content.as_array() else {
        return stringify_content(content);
    };
    let mut parts = Vec::new();
    for block in blocks {
        let Some(kind) = block.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "text" => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(text.to_string());
                }
            }
            "thinking" => {
                if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                    parts.push(format!("<think>{text}</think>"));
                }
            }
            "toolCall" => {
                let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                let name_json = serde_json::to_string(name).unwrap_or_else(|_| "\"\"".to_string());
                let args = block.get("input").unwrap_or(&Value::Null);
                let args_json = if args.is_null() {
                    "{}".to_string()
                } else {
                    serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
                };
                parts.push(format!(
                    "<tool_call>{{\"name\": {name_json}, \"arguments\": {args_json}}}</tool_call>"
                ));
            }
            other => warnings.push(format!(
                "skipped unsupported assistant block type {other:?}"
            )),
        }
    }
    parts.join("")
}

fn render_tool_result_content(
    content: &Value,
    warnings: &mut Vec<String>,
) -> (String, Option<String>) {
    let Some(blocks) = content.as_array() else {
        return (stringify_content(content), None);
    };
    let mut parts = Vec::new();
    let mut tool_call_id = None;
    for block in blocks {
        let Some(kind) = block.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "toolResult" => {
                let result_content = block.get("content").unwrap_or(&Value::Null);
                parts.push(stringify_content(result_content));
                if tool_call_id.is_none() {
                    tool_call_id = block
                        .get("toolCallId")
                        .or_else(|| block.get("tool_call_id"))
                        .or_else(|| block.get("id"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string);
                }
            }
            "text" => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(text.to_string());
                }
            }
            other => warnings.push(format!("skipped unsupported tool block type {other:?}")),
        }
    }
    (parts.join(""), tool_call_id)
}

pub fn detect_warning_prefix(text: &str) -> Option<usize> {
    if !text.starts_with("WARNINGS:\n") {
        return None;
    }
    if let Some(idx) = text.find("<command_output>") {
        if idx > 0 {
            return Some(idx);
        }
    }
    if let Some(idx) = text.find("\n\n") {
        if idx > 0 {
            return Some(idx + 2);
        }
    }
    Some("WARNINGS:\n".len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jsonl(lines: &[Value]) -> String {
        let mut out = String::new();
        for line in lines {
            out.push_str(&serde_json::to_string(line).unwrap());
            out.push('\n');
        }
        out
    }

    #[test]
    fn parses_pi_0751_session_like_python_parser() {
        let raw = jsonl(&[
            serde_json::json!({"type":"message","message":{"role":"system","content":[{"type":"text","text":"Python assistant"}]}}),
            serde_json::json!({"type":"message","message":{"role":"user","content":[{"type":"text","text":"Print 42"}]}}),
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"thinking","thinking":"I should use bash"},{"type":"toolCall","name":"bash","input":{"cmd":"python3 -c 'print(42)'"},"id":"c1"}]}}),
            serde_json::json!({"type":"message","message":{"role":"tool","content":[{"type":"toolResult","content":"42\n","toolCallId":"c1"}]}}),
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Done - the program printed 42."}]}}),
        ]);

        let parsed = parse_pi_session_str(&raw, false);

        assert_eq!(parsed.prompt_messages.len(), 2);
        assert_eq!(parsed.trajectory.len(), 3);
        assert_eq!(
            parsed.trajectory.iter().map(|s| s.kind).collect::<Vec<_>>(),
            vec![TurnKind::Action, TurnKind::Observation, TurnKind::Action]
        );
        assert!(parsed.trajectory[0].content.contains("<think>I should use bash</think>"));
        assert!(parsed.trajectory[0].content.contains("<tool_call>"));
        assert!(parsed.trajectory[0].content.contains("\"arguments\""));
        assert_eq!(parsed.trajectory[1].role, "tool");
        assert_eq!(parsed.trajectory[1].content, "42\n");
        assert_eq!(parsed.trajectory[1].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(parsed.trajectory[2].content, "Done - the program printed 42.");
    }

    #[test]
    fn parses_pi_0753_tool_result_role_like_python_parser() {
        let raw = jsonl(&[
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","name":"bash","input":{"cmd":"echo ok"},"id":"c1"}]}}),
            serde_json::json!({"type":"message","message":{"role":"toolResult","content":[{"type":"text","text":"ok\n"}]}}),
        ]);

        let parsed = parse_pi_session_str(&raw, false);

        assert_eq!(parsed.trajectory.len(), 2);
        assert_eq!(parsed.trajectory[1].role, "tool");
        assert_eq!(parsed.trajectory[1].kind, TurnKind::Observation);
        assert_eq!(parsed.trajectory[1].content, "ok\n");
    }

    #[test]
    fn include_context_emits_system_and_user_segments() {
        let raw = jsonl(&[
            serde_json::json!({"type":"message","message":{"role":"system","content":[{"type":"text","text":"sys"}]}}),
            serde_json::json!({"type":"message","message":{"role":"user","content":[{"type":"text","text":"do it"}]}}),
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]}}),
        ]);

        let parsed = parse_pi_session_str(&raw, true);

        assert!(parsed.prompt_messages.is_empty());
        assert_eq!(parsed.trajectory.len(), 3);
        assert_eq!(
            parsed.trajectory.iter().map(|s| s.kind).collect::<Vec<_>>(),
            vec![TurnKind::Context, TurnKind::Context, TurnKind::Action]
        );
    }

    #[test]
    fn warning_prefix_and_flattening_match_python_parser_contract() {
        let warning = "WARNINGS:\n- something bad\n\nactual output";
        let raw = jsonl(&[
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"first"}]}}),
            serde_json::json!({"type":"message","message":{"role":"tool","content":[{"type":"toolResult","content":warning}]}}),
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"second"}]}}),
        ]);

        let parsed = parse_pi_session_str(&raw, false);

        let warning_prefix_len = parsed.trajectory[1].warning_prefix_len.unwrap();
        assert!(warning_prefix_len > 0);
        assert!(warning_prefix_len <= warning.find("actual output").unwrap());
        assert_eq!(flatten_action_text(&parsed.trajectory), "first<TURN_BREAK>second");
        let rollout = scored_rollout_from_pi_session(&raw, 1.0, false);
        assert_eq!(rollout.text, "first<TURN_BREAK>second");
        assert_eq!(rollout.trajectory.len(), parsed.trajectory.len());
        assert_eq!(rollout.trajectory[0].content, parsed.trajectory[0].content);
        assert_eq!(rollout.trajectory[1].warning_prefix_len, Some(warning_prefix_len));
    }
}
