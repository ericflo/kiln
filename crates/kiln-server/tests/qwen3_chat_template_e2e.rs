//! Chat-template rendering end-to-end against the built-in agentic suite.
//!
//! Pins that:
//! - The built-in `qwen3.5-agentic-core` suite's `tools` catalogue
//!   renders into Qwen3.5's `<tools>` system block.
//! - Multi-turn agentic prompts (user → assistant(tool_call) → tool)
//!   render into Qwen3.5's native `<tool_call><function=…>` and
//!   `<tool_response>` XML form.
//! - The generation prompt always ends inside an open `<think>` block.
//!
//! This is the load-bearing test that ties the eval system's `tools` /
//! agentic plumbing to the actual chat template the model trained on.
//! If anything regresses in the template-rendering path, this test
//! fails loudly before users hit it.

use kiln_core::tokenizer::{ChatMessage, ChatTemplateOptions, KilnTokenizer};
use kiln_eval::qwen3_agentic_core;
use kiln_eval::suite::EvalChatMessage;
use tokenizers::Tokenizer;

const QWEN35_4B_TEMPLATE: &str =
    include_str!("../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");

fn fixture_tokenizer() -> KilnTokenizer {
    // Minimal vocab tokenizer — we only exercise the *template*, not
    // the encoder, so the BPE table can be empty.
    let json = br#"{
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1},
            "merges": []
        }
    }"#;
    let inner = Tokenizer::from_bytes(json.as_slice()).expect("tokenizer json");
    KilnTokenizer::from_inner(inner).with_chat_template(QWEN35_4B_TEMPLATE.to_string())
}

fn eval_to_chat_messages(eval_msgs: &[EvalChatMessage]) -> Vec<ChatMessage> {
    eval_msgs
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            tool_calls: m.tool_calls.clone(),
            name: m.name.clone(),
            tool_call_id: m.tool_call_id.clone(),
            ..Default::default()
        })
        .collect()
}

#[test]
fn builtin_suite_tools_render_into_system_block() {
    let suite = qwen3_agentic_core();
    let tok = fixture_tokenizer();
    let tools = suite.tools.expect("suite has tools").clone();
    let first_example = &suite.examples[0];
    let mut chat = Vec::new();
    if let Some(sp) = suite.system_prompt.as_deref() {
        chat.push(ChatMessage {
            role: "system".into(),
            content: sp.to_string(),
            ..Default::default()
        });
    }
    chat.extend(eval_to_chat_messages(&first_example.messages));

    let prompt = tok
        .apply_chat_template_with_tools(&chat, Some(&tools))
        .expect("chat template render");

    // Tools block must appear with every tool name.
    assert!(prompt.contains("<tools>"), "missing <tools>: {prompt}");
    for tool in &tools {
        let name = tool["function"]["name"].as_str().unwrap();
        // minijinja's `tojson` produces compact (no-space) JSON, but
        // accept either form — different versions / pretty toggles vary.
        let with_space = format!("\"name\": \"{name}\"");
        let without_space = format!("\"name\":\"{name}\"");
        assert!(
            prompt.contains(&with_space) || prompt.contains(&without_space),
            "tool {name} not in <tools>: prompt was {prompt}"
        );
    }
    // Generation prompt must end inside an open `<think>` block — that's
    // what triggers Qwen3.5's reasoning behavior at inference time.
    assert!(
        prompt.ends_with("<|im_start|>assistant\n<think>\n"),
        "expected open <think> at end, got: ...{}",
        &prompt[prompt.len().saturating_sub(80)..]
    );
}

#[test]
fn agentic_followup_example_renders_prior_tool_call_and_response() {
    let suite = qwen3_agentic_core();
    let example = suite
        .examples
        .iter()
        .find(|ex| ex.id.as_deref() == Some("agentic/weather-then-answer"))
        .expect("agentic followup example present in builtin suite");
    assert_eq!(example.messages.len(), 3, "expected user/assistant/tool prompt");

    let tok = fixture_tokenizer();
    let chat = eval_to_chat_messages(&example.messages);
    let tools = suite.tools.as_ref().unwrap();
    let prompt = tok
        .apply_chat_template_with_tools(&chat, Some(tools))
        .expect("multi-turn chat template render");
    // Prior assistant tool call must render in Qwen3.5 native XML.
    assert!(
        prompt.contains("<function=get_weather>"),
        "no `<function=get_weather>` in rendered prompt"
    );
    assert!(
        prompt.contains("<parameter=city>"),
        "no `<parameter=city>` in rendered prompt"
    );
    // Tool response wraps in `<tool_response>` framing.
    assert!(
        prompt.contains("<tool_response>"),
        "no `<tool_response>` in rendered prompt"
    );
    // The tool result body shows up too.
    assert!(prompt.contains("18"), "tool result content missing");
    // Ends with the open thinking block, ready for the model to continue.
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
}

#[test]
fn enable_thinking_false_collapses_thinking_block() {
    // Some users want to evaluate without thinking — confirm the per-
    // example chat_template_kwargs path works end-to-end.
    let suite = qwen3_agentic_core();
    let example = &suite.examples[0];
    let tok = fixture_tokenizer();
    let chat = eval_to_chat_messages(&example.messages);
    let tools = suite.tools.as_ref().unwrap();
    let prompt = tok
        .apply_chat_template_full_with_options(
            &chat,
            Some(tools),
            None,
            ChatTemplateOptions {
                template_kwargs: serde_json::Map::from_iter([(
                    "enable_thinking".to_string(),
                    serde_json::Value::Bool(false),
                )]),
            },
        )
        .expect("enable_thinking=false render");
    assert!(
        prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "enable_thinking=false should pre-close the reasoning block: ...{}",
        &prompt[prompt.len().saturating_sub(80)..]
    );
}
