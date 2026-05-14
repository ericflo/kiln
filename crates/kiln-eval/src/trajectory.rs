//! Anthropic API trajectory → `SftConversation` converter.
//!
//! The clouderic-app trajectory database stores production turns in
//! Anthropic API shape (`{type: "tool_use", name, input}` blocks for
//! assistant tool calls, `{type: "tool_result", tool_use_id, content}`
//! blocks for tool replies). The Python bridge converts to OpenAI shape
//! before feeding kiln SFT; this Rust converter mirrors the same logic
//! so a user can pipe raw Anthropic trajectories straight into kiln-eval
//! synthesis without a Python step.
//!
//! Input is the per-turn pair `(messages, response_blocks)` plus optional
//! `system` / `tools`. Output is a single [`SftConversation`] with
//! OpenAI-style `tool_calls`/`tool_call_id` fields preserved so downstream
//! synthesis can canonicalize cleanly.

use serde::{Deserialize, Serialize};

use crate::synthesis::{SftConversation, SftMessage};

/// One Anthropic message — either a user message containing
/// tool_result/text blocks, or an assistant message containing
/// text/tool_use blocks. The "string content" shortcut Anthropic
/// supports for plain text is also accepted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    /// Plain text content (Anthropic's shortcut form).
    Text(String),
    /// List of typed blocks (canonical form).
    Blocks(Vec<AnthropicBlock>),
}

/// Single block inside an Anthropic message content array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
        #[serde(default)]
        is_error: bool,
    },
    /// Catch-all so unknown block types don't fail the parse.
    #[serde(other)]
    Unknown,
}

/// Input message in Anthropic's "messages" array shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Convert a complete Anthropic-shape turn into a single
/// `SftConversation` ready for synthesis. The trajectory bridge writes
/// (messages → response_blocks) per turn; this fn concatenates them so
/// the resulting conversation ends on the assistant turn that produced
/// `response_blocks`.
///
/// Tool catalog and system prompt are stored in `extra` so downstream
/// synthesis's `auto_pick_tools_from_extra` path lifts them onto the
/// suite automatically.
pub fn anthropic_turn_to_sft_conversation(
    messages: &[AnthropicMessage],
    response_blocks: &[AnthropicBlock],
    system_prompt: Option<&str>,
    tools: Option<&[serde_json::Value]>,
) -> SftConversation {
    let mut out: Vec<SftMessage> = Vec::with_capacity(messages.len() + 2);

    if let Some(sys) = system_prompt.filter(|s| !s.trim().is_empty()) {
        out.push(SftMessage {
            role: "system".into(),
            content: sys.to_string(),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        });
    }

    for m in messages {
        translate_message_into(m, &mut out);
    }

    // Append the assistant's response blocks as the final assistant turn.
    let assistant = blocks_to_assistant_sft_message(response_blocks);
    if assistant_has_payload(&assistant) {
        out.push(assistant);
    }

    let mut extra = serde_json::Map::new();
    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        extra.insert(
            "tools".into(),
            serde_json::Value::Array(tools.to_vec()),
        );
    }

    SftConversation {
        messages: out,
        extra,
    }
}

fn translate_message_into(msg: &AnthropicMessage, out: &mut Vec<SftMessage>) {
    match (msg.role.as_str(), &msg.content) {
        ("user", AnthropicContent::Text(t)) => out.push(SftMessage {
            role: "user".into(),
            content: t.clone(),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }),
        ("assistant", AnthropicContent::Text(t)) => out.push(SftMessage {
            role: "assistant".into(),
            content: t.clone(),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }),
        ("user", AnthropicContent::Blocks(blocks)) => {
            // User-role blocks may contain a mix of `text` and
            // `tool_result` blocks. Split into one SftMessage per
            // semantically distinct part:
            //  - tool_result → role=tool with the corresponding
            //    tool_call_id (so the chat template emits
            //    `<tool_response>` correctly).
            //  - text → user text (concatenated).
            let mut text_parts: Vec<String> = Vec::new();
            for b in blocks {
                match b {
                    AnthropicBlock::Text { text } => text_parts.push(text.clone()),
                    AnthropicBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        let body = tool_result_content_to_string(content);
                        out.push(SftMessage {
                            role: "tool".into(),
                            content: body,
                            tool_calls: None,
                            name: None,
                            tool_call_id: Some(tool_use_id.clone()),
                        });
                    }
                    _ => {}
                }
            }
            if !text_parts.is_empty() {
                out.push(SftMessage {
                    role: "user".into(),
                    content: text_parts.join("\n"),
                    tool_calls: None,
                    name: None,
                    tool_call_id: None,
                });
            }
        }
        ("assistant", AnthropicContent::Blocks(blocks)) => {
            let msg = blocks_to_assistant_sft_message(blocks);
            if assistant_has_payload(&msg) {
                out.push(msg);
            }
        }
        // Unknown roles — drop silently.
        _ => {}
    }
}

fn blocks_to_assistant_sft_message(blocks: &[AnthropicBlock]) -> SftMessage {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    for b in blocks {
        match b {
            AnthropicBlock::Text { text } => text_parts.push(text.clone()),
            AnthropicBlock::ToolUse { id, name, input } => {
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                    }
                }));
            }
            _ => {}
        }
    }
    SftMessage {
        role: "assistant".into(),
        content: text_parts.join("\n"),
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        name: None,
        tool_call_id: None,
    }
}

fn assistant_has_payload(msg: &SftMessage) -> bool {
    !msg.content.is_empty() || msg.tool_calls.as_ref().map_or(false, |t| !t.is_empty())
}

fn tool_result_content_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            // Each entry is typically `{type: "text", text: "..."}`.
            let mut parts: Vec<String> = Vec::new();
            for entry in arr {
                if let Some(text) = entry.get("text").and_then(|v| v.as_str()) {
                    parts.push(text.to_string());
                } else if let Some(s) = entry.as_str() {
                    parts.push(s.to_string());
                } else {
                    parts.push(entry.to_string());
                }
            }
            parts.join("\n")
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_text(t: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Text(t.into()),
        }
    }

    fn assistant_blocks(blocks: Vec<AnthropicBlock>) -> AnthropicMessage {
        AnthropicMessage {
            role: "assistant".into(),
            content: AnthropicContent::Blocks(blocks),
        }
    }

    fn user_blocks(blocks: Vec<AnthropicBlock>) -> AnthropicMessage {
        AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(blocks),
        }
    }

    #[test]
    fn converts_single_turn_with_tool_use() {
        // Real production shape (the same Edit/Bash patterns seen in the
        // user's trajectory DB).
        let messages = vec![user_text("Show me what's in /etc.")];
        let response = vec![
            AnthropicBlock::Text {
                text: "Let me run ls /etc:".into(),
            },
            AnthropicBlock::ToolUse {
                id: "toolu_01".into(),
                name: "Bash".into(),
                input: serde_json::json!({"command": "ls /etc"}),
            },
        ];
        let conv = anthropic_turn_to_sft_conversation(&messages, &response, None, None);
        assert_eq!(conv.messages.len(), 2);
        assert_eq!(conv.messages[0].role, "user");
        assert_eq!(conv.messages[1].role, "assistant");
        let tcs = conv.messages[1].tool_calls.as_ref().expect("tool_calls");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["function"]["name"], "Bash");
        // Arguments are stored as a JSON-encoded string in OpenAI shape
        // (matches what the chat template's normalizer expects).
        let args_str = tcs[0]["function"]["arguments"].as_str().unwrap();
        assert_eq!(args_str, "{\"command\":\"ls /etc\"}");
    }

    #[test]
    fn converts_multi_turn_with_tool_response() {
        let messages = vec![
            user_text("ls /etc"),
            assistant_blocks(vec![AnthropicBlock::ToolUse {
                id: "toolu_99".into(),
                name: "Bash".into(),
                input: serde_json::json!({"command": "ls /etc"}),
            }]),
            user_blocks(vec![AnthropicBlock::ToolResult {
                tool_use_id: "toolu_99".into(),
                content: serde_json::json!("hosts\nresolv.conf"),
                is_error: false,
            }]),
        ];
        let response = vec![AnthropicBlock::Text {
            text: "/etc contains hosts and resolv.conf.".into(),
        }];
        let conv = anthropic_turn_to_sft_conversation(&messages, &response, None, None);
        // user, assistant(tool_call), tool(result), assistant(final)
        assert_eq!(conv.messages.len(), 4);
        assert_eq!(conv.messages[0].role, "user");
        assert_eq!(conv.messages[1].role, "assistant");
        assert!(conv.messages[1].tool_calls.is_some());
        assert_eq!(conv.messages[2].role, "tool");
        assert_eq!(conv.messages[2].content, "hosts\nresolv.conf");
        assert_eq!(
            conv.messages[2].tool_call_id.as_deref(),
            Some("toolu_99")
        );
        assert_eq!(conv.messages[3].role, "assistant");
        assert!(conv.messages[3]
            .content
            .contains("hosts and resolv.conf"));
    }

    #[test]
    fn handles_array_tool_result_content() {
        let messages = vec![user_blocks(vec![AnthropicBlock::ToolResult {
            tool_use_id: "t1".into(),
            content: serde_json::json!([
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"}
            ]),
            is_error: false,
        }])];
        let conv = anthropic_turn_to_sft_conversation(&messages, &[], None, None);
        assert_eq!(conv.messages.len(), 1);
        assert_eq!(conv.messages[0].role, "tool");
        assert_eq!(conv.messages[0].content, "line 1\nline 2");
    }

    #[test]
    fn promotes_tools_to_extra_for_synth_auto_pickup() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "Bash", "parameters": {"type": "object"}}
        })];
        let conv = anthropic_turn_to_sft_conversation(
            &[user_text("x")],
            &[AnthropicBlock::Text { text: "y".into() }],
            None,
            Some(&tools),
        );
        let extra_tools = conv.extra.get("tools").expect("tools in extra");
        assert_eq!(extra_tools.as_array().unwrap().len(), 1);
    }

    #[test]
    fn drops_unknown_block_types_gracefully() {
        // The wire format may evolve — ensure unknown blocks don't fail.
        // Untagged enum variant with `#[serde(other)]` requires no fields.
        let raw = r#"[{"type": "future_image_thing"}, {"type": "text", "text": "ok"}]"#;
        let blocks: Vec<AnthropicBlock> = serde_json::from_str(raw).unwrap();
        let conv = anthropic_turn_to_sft_conversation(
            &[user_text("x")],
            &blocks,
            None,
            None,
        );
        assert_eq!(conv.messages.len(), 2);
        assert_eq!(conv.messages[1].content, "ok");
    }
}
