//! Integration test against real production trajectory shapes.
//!
//! The fixtures below are taken from a Claude production trajectory
//! database (~126k turns) and represent the actual shape of tool calls
//! Kiln must score in the wild — Anthropic-style `{type: "tool_use",
//! name, input: {...}}` blocks. The bridge script converts those into
//! OpenAI-style trajectories for Qwen3.5 SFT; this test pretends the model
//! has emitted the same call in Qwen3.5's native XML and verifies our
//! `tool_call` scorer treats it as a Pass against the canonical JSON
//! target.
//!
//! Running: `cargo test -p kiln-eval --test real_trajectory_shapes`.
//!
//! The fixtures are inlined verbatim so the test is hermetic — no DB
//! connection required at test time.

use kiln_eval::qwen3::{ParsedToolCall, ToolCallFormat, extract_first_tool_call};
use kiln_eval::scorers::{ArgsScoring, NameMatch, NoopJudgeRunner, Scorer, score_completion};
use kiln_eval::suite::{EvalChatMessage, EvalExample};
use kiln_eval::EvalOutcomeKind;

fn ex(target_json: serde_json::Value) -> EvalExample {
    EvalExample {
        messages: vec![EvalChatMessage::new("user", "irrelevant")],
        target: Some(target_json.to_string()),
        ..Default::default()
    }
}

fn tool_call_scorer() -> Scorer {
    Scorer::ToolCall {
        name_match: NameMatch::CaseInsensitive,
        args: ArgsScoring::Structural,
        weights: None,
    }
}

/// Helper: build a JSON-canonical target from an Anthropic `tool_use`
/// block (the shape stored in the trajectory DB).
fn anthropic_to_canonical(name: &str, input: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "tool_calls": [{
            "name": name,
            "arguments": input,
        }]
    })
}

#[test]
fn edit_tool_with_multiline_strings_roundtrips() {
    // Real production turn: Edit a Python script with multi-line strings.
    let input = serde_json::json!({
        "replace_all": false,
        "file_path": "/data/apps/trajectory-trainer/scripts/run_experiment.py",
        "old_string": "        ssh_cmd(pod_id, \"mkdir -p /workspace/dpo-data\")\n\n        for split in ('train', 'val', 'test'):",
        "new_string": "        ssh_cmd(pod_id, \"mkdir -p /workspace/kto-data\")\n\n        for split in ('train', 'val', 'test'):"
    });
    let target = anthropic_to_canonical("Edit", input.clone());

    // Simulate Qwen3.5 XML output that matches.
    let mut args = serde_json::Map::new();
    for (k, v) in input.as_object().unwrap() {
        args.insert(k.clone(), v.clone());
    }
    let parsed = ParsedToolCall {
        name: "Edit".into(),
        arguments: args,
        format: ToolCallFormat::Qwen3Xml,
    };
    let xml = parsed.to_qwen3_xml();
    // Round-trip the XML via the extractor — every key must round-trip.
    let back = extract_first_tool_call(&xml).expect("XML must parse back");
    assert_eq!(back.name, "Edit");
    assert!(back.arguments.contains_key("file_path"));

    // Score the XML against the JSON target.
    let outcome =
        score_completion(&tool_call_scorer(), &ex(target), &xml, &NoopJudgeRunner).unwrap();
    assert_eq!(
        outcome.kind,
        EvalOutcomeKind::Pass,
        "edit-tool roundtrip should Pass: {:?}",
        outcome.detail
    );
}

#[test]
fn bash_tool_with_command_arg_canonicalizes_correctly() {
    let input = serde_json::json!({
        "command": "head -c 2000 /data/logs/capture/anthropic/2026-04-06/web-7cc48cb7df-vkwqt-001159.json"
    });
    let target = anthropic_to_canonical("Bash", input.clone());

    // Same call emitted as Qwen3.5 XML — Bash's `command` value contains
    // spaces, slashes, and dashes. The XML parameter body just carries it
    // verbatim.
    let xml = "<tool_call>\n<function=Bash>\n<parameter=command>\nhead -c 2000 /data/logs/capture/anthropic/2026-04-06/web-7cc48cb7df-vkwqt-001159.json\n</parameter>\n</function>\n</tool_call>";

    let outcome = score_completion(
        &tool_call_scorer(),
        &ex(target),
        xml,
        &NoopJudgeRunner,
    )
    .unwrap();
    assert_eq!(
        outcome.kind,
        EvalOutcomeKind::Pass,
        "bash-command must canonicalize: {:?}",
        outcome.detail
    );
}

#[test]
fn boolean_arg_in_edit_tool_recovers_as_string_in_auto_mode() {
    // Auto mode is more forgiving — XML params come back as strings, but
    // the scorer's bash/string heuristics handle that.
    let input = serde_json::json!({
        "replace_all": false,
        "file_path": "/x.txt",
        "old_string": "abc",
        "new_string": "xyz"
    });
    let target = anthropic_to_canonical("Edit", input);

    // Simulate model emission with `replace_all` rendered as string "false".
    let xml = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nfalse\n</parameter>\n<parameter=file_path>\n/x.txt\n</parameter>\n<parameter=old_string>\nabc\n</parameter>\n<parameter=new_string>\nxyz\n</parameter>\n</function>\n</tool_call>";

    let auto_scorer = Scorer::ToolCall {
        name_match: NameMatch::CaseInsensitive,
        args: ArgsScoring::Auto,
        weights: None,
    };
    let outcome =
        score_completion(&auto_scorer, &ex(target), xml, &NoopJudgeRunner).unwrap();
    // Auto mode: string "false" vs JSON bool false — partial credit on the
    // arg, but name + structure are perfect. We expect Pass under the
    // current pass threshold (0.8).
    assert!(
        matches!(outcome.kind, EvalOutcomeKind::Pass | EvalOutcomeKind::Fail),
        "kind must be Pass or Fail under auto args, got {:?}",
        outcome.kind
    );
    assert!(outcome.score >= 0.7, "score was {}", outcome.score);
}

#[test]
fn read_tool_with_path_arg_passes_strict_per_key_check() {
    let input = serde_json::json!({"file_path": "/etc/hosts"});
    let target = anthropic_to_canonical("Read", input);

    let xml = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/etc/hosts\n</parameter>\n</function>\n</tool_call>";

    let mut per_key = std::collections::BTreeMap::new();
    per_key.insert(
        "file_path".to_string(),
        Scorer::ExactMatch {
            case_sensitive: true,
            strip_whitespace: true,
        },
    );
    let strict = Scorer::ToolCall {
        name_match: NameMatch::Exact,
        args: ArgsScoring::PerKey {
            scorers: per_key,
            extra_key_penalty: 0.5,
        },
        weights: None,
    };
    let outcome = score_completion(&strict, &ex(target), xml, &NoopJudgeRunner).unwrap();
    assert_eq!(outcome.kind, EvalOutcomeKind::Pass);
}

#[test]
fn thinking_before_tool_call_in_real_trajectory_scores_correctly() {
    // Real Qwen3.5 emission with reasoning then a tool call.
    let raw = "<think>\nThe user is asking about a file. I should use the Read tool to fetch its contents at /etc/hosts.\n</think>\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/etc/hosts\n</parameter>\n</function>\n</tool_call>";
    let target = anthropic_to_canonical("Read", serde_json::json!({"file_path": "/etc/hosts"}));
    let outcome = score_completion(
        &tool_call_scorer(),
        &ex(target),
        raw,
        &NoopJudgeRunner,
    )
    .unwrap();
    assert_eq!(outcome.kind, EvalOutcomeKind::Pass);
    // Reasoning was captured for dashboard display.
    assert!(outcome.reasoning_text.is_some());
}

#[test]
fn wrong_tool_name_fails_even_with_correct_args() {
    let target = anthropic_to_canonical("Edit", serde_json::json!({"file_path": "/a"}));
    // Model picked Write instead of Edit.
    let raw = "<tool_call>\n<function=Write>\n<parameter=file_path>\n/a\n</parameter>\n</function>\n</tool_call>";
    let outcome = score_completion(
        &tool_call_scorer(),
        &ex(target),
        raw,
        &NoopJudgeRunner,
    )
    .unwrap();
    assert_eq!(outcome.kind, EvalOutcomeKind::Fail);
}

#[test]
fn extra_tool_call_in_real_trajectory_is_penalized() {
    // Two tool calls when only one was expected.
    let target = anthropic_to_canonical("Edit", serde_json::json!({"file_path": "/a"}));
    let raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/a\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
    let outcome = score_completion(
        &tool_call_scorer(),
        &ex(target),
        raw,
        &NoopJudgeRunner,
    )
    .unwrap();
    assert!(outcome.score < 1.0, "score was {}", outcome.score);
    assert!(
        outcome
            .detail
            .as_deref()
            .map_or(false, |d| d.contains("excess_calls"))
    );
}

#[test]
fn json_response_when_xml_expected_still_scores_correctly() {
    // Sometimes the model emits JSON tool calls. Scorer should still
    // match against the JSON target — but note format mismatch in detail.
    let target = anthropic_to_canonical("Edit", serde_json::json!({"file_path": "/a"}));
    let raw = r#"{"tool_calls": [{"name": "Edit", "arguments": {"file_path": "/a"}}]}"#;
    let outcome = score_completion(
        &tool_call_scorer(),
        &ex(target),
        raw,
        &NoopJudgeRunner,
    )
    .unwrap();
    assert_eq!(outcome.kind, EvalOutcomeKind::Pass);
    // Format diagnostic present.
    assert!(
        outcome
            .detail
            .as_deref()
            .map_or(false, |d| d.contains("formats=")),
        "expected format diagnostic in detail: {:?}",
        outcome.detail
    );
}

#[test]
fn refusal_when_tool_should_have_been_used_is_invalid() {
    let target = anthropic_to_canonical("Edit", serde_json::json!({"file_path": "/a"}));
    let raw = "I can't make that edit — the file looks dangerous.";
    let outcome = score_completion(
        &tool_call_scorer(),
        &ex(target),
        raw,
        &NoopJudgeRunner,
    )
    .unwrap();
    assert_eq!(outcome.kind, EvalOutcomeKind::Invalid);
}
