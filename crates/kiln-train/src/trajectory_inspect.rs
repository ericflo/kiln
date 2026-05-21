//! Inspect Pi/kiln trajectory JSONL files through kiln's canonical mask builder.

use std::path::Path;

use anyhow::{Context, Result, bail};
use kiln_core::tokenizer::{ChatMessage as CoreChatMessage, KilnTokenizer};
use serde::Serialize;
use serde_json::Value;

use crate::ChatMessage;
use crate::trajectory::{AgenticGroup, ScoredRollout, TurnKind, TurnSegment};
use crate::trajectory_mask::{MaskConfig, build_masks_from_trajectory};

#[derive(Debug, Clone, Serialize)]
pub struct TrajectoryInspectReport {
    pub path: String,
    pub source_format: String,
    pub rollouts: Vec<RolloutInspection>,
    pub action_tokens: usize,
    pub env_tokens: usize,
    pub context_tokens: usize,
    pub warning_prefix_stripped_bytes: usize,
    pub schema_warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RolloutInspection {
    pub index: usize,
    pub prompt_messages: Vec<ChatMessage>,
    pub rendered_messages: String,
    pub segments: Vec<SegmentInspection>,
    pub action_tokens: usize,
    pub env_tokens: usize,
    pub context_tokens: usize,
    pub warning_prefix_stripped_bytes: usize,
    pub action_preview: String,
    pub env_preview: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentInspection {
    pub index: usize,
    pub role: String,
    pub kind: TurnKind,
    pub content: String,
    pub token_start: Option<usize>,
    pub token_end: Option<usize>,
    pub token_count: usize,
    pub warning_prefix_len: Option<usize>,
    pub warning_prefix_stripped_bytes: usize,
    pub tool_call_id: Option<String>,
}

struct InspectInput {
    source_format: &'static str,
    rollouts: Vec<RolloutInput>,
    warnings: Vec<String>,
}

struct RolloutInput {
    prompt_messages: Vec<ChatMessage>,
    trajectory: Vec<TurnSegment>,
}

pub fn inspect_trajectory_file(
    path: &Path,
    tokenizer: &KilnTokenizer,
    include_context: bool,
    preview_tokens: usize,
) -> Result<TrajectoryInspectReport> {
    let input = load_inspect_input(path, include_context)?;
    let mut rollouts = Vec::with_capacity(input.rollouts.len());
    let mut warnings = input.warnings;

    for (idx, rollout) in input.rollouts.into_iter().enumerate() {
        let inspected = inspect_rollout(idx, rollout, tokenizer, preview_tokens)
            .with_context(|| format!("inspect rollout {idx} from {}", path.display()))?;
        rollouts.push(inspected);
    }

    let action_tokens = rollouts.iter().map(|r| r.action_tokens).sum();
    let env_tokens = rollouts.iter().map(|r| r.env_tokens).sum();
    let context_tokens = rollouts.iter().map(|r| r.context_tokens).sum();
    let warning_prefix_stripped_bytes = rollouts
        .iter()
        .map(|r| r.warning_prefix_stripped_bytes)
        .sum();

    if action_tokens == 0 {
        warnings.push("no trainable action tokens found".to_string());
        bail!(
            "trajectory inspect: no trainable action tokens found in {}",
            path.display()
        );
    }

    Ok(TrajectoryInspectReport {
        path: path.display().to_string(),
        source_format: input.source_format.to_string(),
        rollouts,
        action_tokens,
        env_tokens,
        context_tokens,
        warning_prefix_stripped_bytes,
        schema_warnings: warnings,
    })
}

fn inspect_rollout(
    index: usize,
    input: RolloutInput,
    tokenizer: &KilnTokenizer,
    preview_tokens: usize,
) -> Result<RolloutInspection> {
    let rendered_messages =
        render_full_messages(&input.prompt_messages, &input.trajectory, tokenizer)?;
    let masked = build_masks_from_trajectory(
        &input.trajectory,
        &input.prompt_messages,
        tokenizer,
        &MaskConfig::default(),
    )?;
    let action_tokens = masked.action_mask.iter().filter(|&&active| active).count();
    let env_tokens = masked.env_mask.iter().filter(|&&active| active).count();
    let context_tokens = masked
        .input_ids
        .len()
        .saturating_sub(action_tokens)
        .saturating_sub(env_tokens);
    let warning_prefix_stripped_bytes = input
        .trajectory
        .iter()
        .filter(|segment| matches!(segment.kind, TurnKind::Observation))
        .filter_map(|segment| segment.warning_prefix_len)
        .sum();
    let action_preview = decode_masked_preview(
        &masked.input_ids,
        &masked.action_mask,
        tokenizer,
        preview_tokens,
    );
    let env_preview = decode_masked_preview(
        &masked.input_ids,
        &masked.env_mask,
        tokenizer,
        preview_tokens,
    );

    let mut supervised_spans = masked.segment_spans.iter();
    let mut segments = Vec::with_capacity(input.trajectory.len());
    for (segment_idx, segment) in input.trajectory.iter().enumerate() {
        let span = if matches!(segment.kind, TurnKind::Action | TurnKind::Observation) {
            supervised_spans.next().copied()
        } else {
            None
        };
        let (token_start, token_end, token_count) = match span {
            Some((start, end, _)) => (Some(start), Some(end), end.saturating_sub(start)),
            None => {
                let content_tokens = tokenizer
                    .encode(&segment.content)
                    .map(|ids| ids.len())
                    .unwrap_or(0);
                (None, None, content_tokens)
            }
        };
        let warning_prefix_stripped_bytes = if matches!(segment.kind, TurnKind::Observation) {
            segment
                .warning_prefix_len
                .unwrap_or(0)
                .min(segment.content.len())
        } else {
            0
        };
        segments.push(SegmentInspection {
            index: segment_idx,
            role: segment.role.clone(),
            kind: segment.kind,
            content: segment.content.clone(),
            token_start,
            token_end,
            token_count,
            warning_prefix_len: segment.warning_prefix_len,
            warning_prefix_stripped_bytes,
            tool_call_id: segment.tool_call_id.clone(),
        });
    }

    Ok(RolloutInspection {
        index,
        prompt_messages: input.prompt_messages,
        rendered_messages,
        segments,
        action_tokens,
        env_tokens,
        context_tokens,
        warning_prefix_stripped_bytes,
        action_preview,
        env_preview,
    })
}

fn render_full_messages(
    prompt_messages: &[ChatMessage],
    trajectory: &[TurnSegment],
    tokenizer: &KilnTokenizer,
) -> Result<String> {
    let mut messages: Vec<CoreChatMessage> = prompt_messages
        .iter()
        .map(|message| CoreChatMessage {
            role: message.role.clone(),
            content: message.content.clone(),
            ..Default::default()
        })
        .collect();
    messages.extend(trajectory.iter().map(|segment| CoreChatMessage {
        role: segment.role.clone(),
        content: segment.content.clone(),
        ..Default::default()
    }));
    tokenizer
        .apply_chat_template(&messages)
        .map_err(|err| anyhow::anyhow!("{err}"))
}

fn decode_masked_preview(
    input_ids: &[u32],
    mask: &[bool],
    tokenizer: &KilnTokenizer,
    limit: usize,
) -> String {
    let ids: Vec<u32> = input_ids
        .iter()
        .zip(mask.iter())
        .filter_map(|(&id, &active)| active.then_some(id))
        .take(limit)
        .collect();
    if ids.is_empty() {
        return String::new();
    }
    tokenizer.decode(&ids).unwrap_or_else(|_| {
        ids.iter()
            .map(|id| format!("<{id}>"))
            .collect::<Vec<_>>()
            .join(" ")
    })
}

fn load_inspect_input(path: &Path, include_context: bool) -> Result<InspectInput> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
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
    if values.is_empty() {
        bail!(
            "trajectory inspect: {} contained no JSON records",
            path.display()
        );
    }

    if values.iter().any(is_pi_message_event) {
        let rollouts = vec![parse_pi_session_values(
            &values,
            include_context,
            &mut warnings,
        )];
        return Ok(InspectInput {
            source_format: "pi_session_jsonl",
            rollouts,
            warnings,
        });
    }

    let mut rollouts = Vec::new();
    for value in values {
        if let Ok(rollout) = serde_json::from_value::<ScoredRollout>(value.clone()) {
            rollouts.push(scored_rollout_input(rollout, Vec::new(), &mut warnings));
        } else if let Ok(group) = serde_json::from_value::<AgenticGroup>(value.clone()) {
            for rollout in group.completions {
                rollouts.push(scored_rollout_input(
                    rollout,
                    group.messages.clone(),
                    &mut warnings,
                ));
            }
        } else {
            warnings.push("skipped JSONL record that was neither a Pi message event nor kiln ScoredRollout/AgenticGroup".to_string());
        }
    }
    if rollouts.is_empty() {
        bail!(
            "trajectory inspect: {} contained no inspectable rollouts",
            path.display()
        );
    }
    Ok(InspectInput {
        source_format: "kiln_rollout_jsonl",
        rollouts,
        warnings,
    })
}

fn scored_rollout_input(
    rollout: ScoredRollout,
    prompt_messages: Vec<ChatMessage>,
    warnings: &mut Vec<String>,
) -> RolloutInput {
    let trajectory = if rollout.trajectory.is_empty() {
        warnings.push("legacy text-only ScoredRollout synthesized one Action segment".to_string());
        vec![TurnSegment::legacy_action(rollout.text)]
    } else {
        rollout.trajectory
    };
    RolloutInput {
        prompt_messages,
        trajectory,
    }
}

fn is_pi_message_event(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("message") && value.get("message").is_some()
}

fn parse_pi_session_values(
    values: &[Value],
    include_context: bool,
    warnings: &mut Vec<String>,
) -> RolloutInput {
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
                let rendered = render_assistant_content(content, warnings);
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
                let (rendered, tool_call_id) = render_tool_result_content(content, warnings);
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
    RolloutInput {
        prompt_messages,
        trajectory,
    }
}

fn stringify_content(content: &Value) -> String {
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

fn detect_warning_prefix(text: &str) -> Option<usize> {
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

    fn qwen_shaped_tokenizer() -> KilnTokenizer {
        let mut vocab = String::from("{");
        for b in 0u32..256 {
            let ch = char::from_u32(b).unwrap();
            let key = match ch {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
                _ => ch.to_string(),
            };
            if b > 0 {
                vocab.push(',');
            }
            vocab.push_str(&format!("\"{}\":{}", key, b));
        }
        vocab.push('}');
        let json = format!(
            r#"{{"version": "1.0", "model": {{"type": "BPE", "vocab": {}, "merges": []}}}}"#,
            vocab
        );
        let template = "{% for message in messages -%}\
{% if message.role == 'tool' %}\
{% if loop.previtem is undefined or loop.previtem.role != 'tool' %}<|im_start|>user
{% endif %}<tool_response>
{{ message.content }}
</tool_response>\
{% if loop.last or loop.nextitem.role != 'tool' %}<|im_end|>
{% endif %}\
{% else %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endif %}\
{% endfor %}";
        KilnTokenizer::from_bytes(json.as_bytes())
            .unwrap()
            .with_chat_template(template.to_string())
    }

    fn write_jsonl(lines: &[Value]) -> tempfile::TempPath {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.into_temp_path();
        let mut body = String::new();
        for line in lines {
            body.push_str(&serde_json::to_string(line).unwrap());
            body.push('\n');
        }
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn inspect_pi_0751_session_builds_action_and_env_masks() -> Result<()> {
        let path = write_jsonl(&[
            serde_json::json!({"type":"message","message":{"role":"user","content":[{"type":"text","text":"Print 42"}]}}),
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"thinking","thinking":"use bash"},{"type":"toolCall","name":"bash","input":{"cmd":"python3 -c 'print(42)'"},"id":"c1"}]}}),
            serde_json::json!({"type":"message","message":{"role":"tool","content":[{"type":"toolResult","content":"42\n","toolCallId":"c1"}]}}),
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Done"}]}}),
        ]);

        let report = inspect_trajectory_file(&path, &qwen_shaped_tokenizer(), false, 128)?;

        assert_eq!(report.source_format, "pi_session_jsonl");
        assert_eq!(report.rollouts.len(), 1);
        assert!(report.action_tokens > 0);
        assert!(report.env_tokens > 0);
        assert!(
            report.rollouts[0]
                .rendered_messages
                .contains("<tool_response>")
        );
        assert_eq!(
            report.rollouts[0].segments[1].tool_call_id.as_deref(),
            Some("c1")
        );
        assert!(!report.rollouts[0].action_preview.is_empty());
        Ok(())
    }

    #[test]
    fn inspect_pi_0753_tool_result_role_normalizes_to_tool() -> Result<()> {
        let path = write_jsonl(&[
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"run"}]}}),
            serde_json::json!({"type":"message","message":{"role":"toolResult","content":[{"type":"text","text":"ok\n"}]}}),
        ]);

        let report = inspect_trajectory_file(&path, &qwen_shaped_tokenizer(), false, 16)?;

        assert_eq!(report.rollouts[0].segments[1].role, "tool");
        assert_eq!(report.rollouts[0].segments[1].kind, TurnKind::Observation);
        assert!(report.env_tokens > 0);
        Ok(())
    }

    #[test]
    fn inspect_scored_rollout_jsonl() -> Result<()> {
        let rollout = ScoredRollout::from_trajectory(
            vec![
                TurnSegment {
                    role: "assistant".into(),
                    content: "call".into(),
                    kind: TurnKind::Action,
                    tool_call_id: None,
                    warning_prefix_len: None,
                },
                TurnSegment {
                    role: "tool".into(),
                    content: "result".into(),
                    kind: TurnKind::Observation,
                    tool_call_id: Some("t1".into()),
                    warning_prefix_len: None,
                },
            ],
            1.0,
        );
        let path = write_jsonl(&[serde_json::to_value(rollout)?]);

        let report = inspect_trajectory_file(&path, &qwen_shaped_tokenizer(), false, 16)?;

        assert_eq!(report.source_format, "kiln_rollout_jsonl");
        assert_eq!(report.rollouts[0].segments.len(), 2);
        assert!(report.action_tokens > 0);
        assert!(report.env_tokens > 0);
        Ok(())
    }

    #[test]
    fn inspect_fails_without_action_tokens() {
        let rollout = ScoredRollout::from_trajectory(
            vec![TurnSegment {
                role: "tool".into(),
                content: "result".into(),
                kind: TurnKind::Observation,
                tool_call_id: None,
                warning_prefix_len: None,
            }],
            1.0,
        );
        let path = write_jsonl(&[serde_json::to_value(rollout).unwrap()]);

        let err = inspect_trajectory_file(&path, &qwen_shaped_tokenizer(), false, 16)
            .unwrap_err()
            .to_string();

        assert!(err.contains("no trainable action tokens"));
    }

    #[test]
    fn warning_prefix_is_reported() -> Result<()> {
        let warning = "WARNINGS:\n- bad\n\nreal output";
        let path = write_jsonl(&[
            serde_json::json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"run"}]}}),
            serde_json::json!({"type":"message","message":{"role":"tool","content":[{"type":"toolResult","content":warning}]}}),
        ]);

        let report = inspect_trajectory_file(&path, &qwen_shaped_tokenizer(), false, 16)?;

        assert!(report.warning_prefix_stripped_bytes > 0);
        assert!(report.rollouts[0].segments[1].warning_prefix_len.is_some());
        Ok(())
    }
}
