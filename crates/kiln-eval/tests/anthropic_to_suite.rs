//! End-to-end: Anthropic API trajectory → SftConversation → synthesized
//! EvalSuite → scored against a hand-emitted Qwen3.5 XML completion.
//!
//! This pins the full pipeline that turns production data from a
//! trajectory database into a runnable kiln eval, and confirms the
//! resulting suite scores correctly when a model emits the canonical
//! Qwen3.5 XML form of the same call.

use kiln_eval::scorers::{ArgsScoring, NameMatch, NoopJudgeRunner, Scorer, score_completion};
use kiln_eval::{
    AnthropicBlock, AnthropicContent, AnthropicMessage, EvalOutcomeKind,
    SynthesisConfig, SynthesisStrategy, anthropic_turn_to_sft_conversation, synthesize_suite,
};

/// Build a realistic agentic turn taken straight from the production
/// trajectory DB shape: user asks to edit a file, assistant emits an
/// Edit tool_use block, tool returns a result, assistant follows up.
fn realistic_edit_trajectory() -> (
    Vec<AnthropicMessage>,
    Vec<AnthropicBlock>,
    Vec<serde_json::Value>,
) {
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Edit a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"}
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }
    })];
    let messages = vec![
        AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Text("Rename `dpo-data` to `kto-data` in run_experiment.py.".into()),
        },
        AnthropicMessage {
            role: "assistant".into(),
            content: AnthropicContent::Blocks(vec![
                AnthropicBlock::Text {
                    text: "Updating the upload path:".into(),
                },
                AnthropicBlock::ToolUse {
                    id: "toolu_01YA1iomuNa446p84x5qki57".into(),
                    name: "Edit".into(),
                    input: serde_json::json!({
                        "file_path": "/data/apps/trajectory-trainer/scripts/run_experiment.py",
                        "old_string": "dpo-data",
                        "new_string": "kto-data",
                        "replace_all": false
                    }),
                },
            ]),
        },
        AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![AnthropicBlock::ToolResult {
                tool_use_id: "toolu_01YA1iomuNa446p84x5qki57".into(),
                content: serde_json::json!("File updated successfully."),
                is_error: false,
            }]),
        },
    ];
    let response = vec![AnthropicBlock::Text {
        text: "Done — `run_experiment.py` now uploads to `kto-data`.".into(),
    }];
    (messages, response, tools)
}

#[test]
fn anthropic_turn_to_synthesized_suite_endtoend() {
    let (messages, response, tools) = realistic_edit_trajectory();
    let conv = anthropic_turn_to_sft_conversation(
        &messages,
        &response,
        Some("You are a careful coding assistant."),
        Some(&tools),
    );

    // The converter produces: system, user, assistant(tool_call),
    // tool(result), assistant(final).
    assert_eq!(conv.messages.len(), 5);
    assert_eq!(conv.messages[0].role, "system");
    assert_eq!(conv.messages[2].role, "assistant");
    assert!(conv.messages[2].tool_calls.is_some());
    assert_eq!(conv.messages[3].role, "tool");
    assert_eq!(conv.messages[4].role, "assistant");

    // Synthesize a tool-call-prediction suite — the produced examples
    // should target the Edit tool with the same canonical arguments.
    let mut cfg = SynthesisConfig::new("anthropic-edit-smoke");
    cfg.strategy = SynthesisStrategy::ToolCallPredict;
    let (suite, stats) = synthesize_suite(vec![conv], &cfg).unwrap();
    assert_eq!(stats.examples_generated, 1);
    assert_eq!(suite.examples.len(), 1);
    // Tool catalogue was auto-promoted from `extra`.
    assert!(suite
        .tools
        .as_ref()
        .map_or(false, |t| !t.is_empty()));
    let example = &suite.examples[0];
    let target = example.target.as_deref().unwrap();
    // Compact JSON, no space after the colon.
    assert!(target.contains("\"name\":\"Edit\""), "target was {target}");
    assert!(target.contains("file_path"));
}

#[test]
fn synthesized_target_scores_against_qwen3_xml_completion() {
    let (messages, response, tools) = realistic_edit_trajectory();
    let conv = anthropic_turn_to_sft_conversation(&messages, &response, None, Some(&tools));
    let mut cfg = SynthesisConfig::new("anthropic-edit-score");
    cfg.strategy = SynthesisStrategy::ToolCallPredict;
    let (suite, _) = synthesize_suite(vec![conv], &cfg).unwrap();
    let example = &suite.examples[0];
    let scorer = Scorer::ToolCall {
        name_match: NameMatch::CaseInsensitive,
        args: ArgsScoring::Structural,
        weights: None,
        require_xml_format: false,
    };

    // Model emits the canonical Qwen3.5 XML for the same call.
    let model_output = "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/data/apps/trajectory-trainer/scripts/run_experiment.py\n</parameter>\n<parameter=old_string>\ndpo-data\n</parameter>\n<parameter=new_string>\nkto-data\n</parameter>\n<parameter=replace_all>\nfalse\n</parameter>\n</function>\n</tool_call>";
    let outcome = score_completion(&scorer, example, model_output, &NoopJudgeRunner).unwrap();
    assert_eq!(
        outcome.kind,
        EvalOutcomeKind::Pass,
        "anthropic→synth→qwen3 xml roundtrip should score Pass: {:?}",
        outcome.detail
    );
}

#[test]
fn end_of_trajectory_answer_uses_tool_response_then_assistant() {
    let (messages, response, _) = realistic_edit_trajectory();
    let conv = anthropic_turn_to_sft_conversation(&messages, &response, None, None);
    let mut cfg = SynthesisConfig::new("anthropic-end-of-trajectory");
    cfg.strategy = SynthesisStrategy::EndOfTrajectoryAnswer;
    let (suite, _) = synthesize_suite(vec![conv], &cfg).unwrap();
    assert_eq!(suite.examples.len(), 1);
    let prompt = &suite.examples[0].messages;
    // Prompt ends on the assistant_with_tool_calls + tool_response pair.
    assert!(prompt.iter().any(|m| m.role == "tool"));
    let target = suite.examples[0].target.as_deref().unwrap();
    assert!(target.contains("kto-data"));
}
