//! End-to-end smoke test for the built-in qwen3.5-agentic-core suite.
//!
//! Runs the suite through the live executor against a custom mock
//! generator that emits canonical Qwen3.5 XML replies — verifying:
//!
//! - The suite is well-formed and `validate()`s.
//! - The executor extracts the right per-example tool name into
//!   `pass_rate_by_tool` for tool-scored examples.
//! - The thinking-stripping logic in `score_completion` produces a Pass
//!   when reasoning prefixes a correct answer.
//! - Per-tool breakdown surfaces correctly when examples target multiple
//!   tools.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_eval::scorers::{JudgeRunner, NoopJudgeRunner};
use kiln_eval::{EvalChatMessage, EvalGenerationParams, EvalSuite};
use kiln_server::eval::executor::run_suite_against_adapter;
use kiln_server::eval::generator::{EvalCompletion, EvalGenerator, PreparedPrompt};

/// Returns canned replies indexed by call order. Useful when the suite
/// has a known stable order and the test wants to inject one specific
/// completion per example.
struct OrderedMockGenerator {
    replies: std::sync::Mutex<std::collections::VecDeque<String>>,
    active: std::sync::Mutex<Option<String>>,
}

impl OrderedMockGenerator {
    fn new(replies: Vec<String>) -> Self {
        Self {
            replies: std::sync::Mutex::new(replies.into()),
            active: std::sync::Mutex::new(None),
        }
    }
}

impl EvalGenerator for OrderedMockGenerator {
    fn set_adapter(
        &self,
        adapter: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
    > {
        let want = adapter.map(str::to_string).filter(|s| !s.is_empty());
        Box::pin(async move {
            let mut slot = self.active.lock().unwrap();
            let previous = slot.clone();
            *slot = want;
            Ok(previous)
        })
    }

    fn prepare(
        &self,
        _messages: &[EvalChatMessage],
        _system_prompt: Option<&str>,
        _tools: Option<&[serde_json::Value]>,
        _params: &EvalGenerationParams,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<PreparedPrompt, String>> + Send + '_>,
    > {
        Box::pin(async move { Ok(PreparedPrompt { tokens: vec![1, 2, 3] }) })
    }

    fn run(
        &self,
        _prepared: &PreparedPrompt,
        _params: &EvalGenerationParams,
        _completion_index: usize,
        adapter_label: Option<&str>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EvalCompletion, String>> + Send + '_>,
    > {
        let next = self.replies.lock().unwrap().pop_front();
        let adapter = adapter_label.map(str::to_string);
        Box::pin(async move {
            let text = next.unwrap_or_else(|| "no canned reply".into());
            Ok(EvalCompletion {
                text,
                prompt_tokens: 3,
                completion_tokens: 1,
                latency_ms: 0.1,
                adapter,
            })
        })
    }
}

fn xml_call(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in params {
        s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    s.push_str("</function>\n</tool_call>");
    s
}

#[tokio::test]
async fn builtin_qwen3_agentic_core_validates_and_lists_tools() {
    let suite = kiln_eval::qwen3_agentic_core();
    assert!(suite.examples.len() >= 20);
    assert!(suite.tools.as_ref().map_or(false, |t| t.len() == 4));
    // Round-trip via JSON to confirm the suite is fully serializable.
    let json = serde_json::to_string(&suite).unwrap();
    let parsed: EvalSuite = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, kiln_eval::QWEN3_AGENTIC_CORE);
}

/// Construct a tiny synthetic agentic suite mirroring the production
/// shape (mixed tool calls + no-tool answers + thinking), then run it
/// end-to-end through the executor. This avoids the brittleness of
/// "score every example in the built-in suite" while exercising the same
/// machinery.
#[tokio::test]
async fn executor_e2e_with_realistic_qwen3_xml_replies() {
    // Use a custom suite so we can control the order of replies exactly.
    use kiln_eval::scorers::{ArgsScoring, NameMatch, Scorer};
    use kiln_eval::EvalExample;

    let suite = EvalSuite {
        name: "agentic-smoke".into(),
        description: None,
        default_scorer: Scorer::ToolCall {
            name_match: NameMatch::CaseInsensitive,
            args: ArgsScoring::Structural,
            weights: None,
        },
        generation: EvalGenerationParams::default(),
        system_prompt: None,
        examples: vec![
            EvalExample {
                id: Some("weather-paris".into()),
                messages: vec![EvalChatMessage::new("user", "What's the weather in Paris?")],
                target: Some(
                    r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}"#
                        .into(),
                ),
                tags: vec!["agentic".into()],
                ..Default::default()
            },
            EvalExample {
                id: Some("read-hosts".into()),
                messages: vec![EvalChatMessage::new("user", "Read /etc/hosts.")],
                target: Some(
                    r#"{"tool_calls":[{"name":"Read","arguments":{"path":"/etc/hosts"}}]}"#
                        .into(),
                ),
                tags: vec!["agentic".into()],
                ..Default::default()
            },
            EvalExample {
                id: Some("read-fail-wrong-tool".into()),
                messages: vec![EvalChatMessage::new("user", "Read /etc/hosts.")],
                target: Some(
                    r#"{"tool_calls":[{"name":"Read","arguments":{"path":"/etc/hosts"}}]}"#
                        .into(),
                ),
                tags: vec!["agentic".into()],
                ..Default::default()
            },
            EvalExample {
                id: Some("thinking-passthrough".into()),
                messages: vec![EvalChatMessage::new(
                    "user",
                    "What's the weather in Tokyo?",
                )],
                target: Some(
                    r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Tokyo"}}]}"#
                        .into(),
                ),
                tags: vec!["agentic".into()],
                ..Default::default()
            },
        ],
        schema_version: 1,
        tools: None,
    };

    // Canned replies: 3 correct (XML), 1 with wrong tool name.
    let replies = vec![
        xml_call("get_weather", &[("city", "Paris")]),
        xml_call("Read", &[("path", "/etc/hosts")]),
        // Wrong tool — should fail and *not* increment Read's pass count.
        xml_call("Write", &[("path", "/etc/hosts")]),
        // Thinking prefix — should still pass after strip.
        format!(
            "<think>\nUser is asking about Tokyo's weather. Use get_weather.\n</think>\n\n{}",
            xml_call("get_weather", &[("city", "Tokyo")])
        ),
    ];
    let gen_ =
        Arc::new(OrderedMockGenerator::new(replies)) as Arc<dyn EvalGenerator>;
    let judge: Arc<dyn JudgeRunner> = Arc::new(NoopJudgeRunner);
    let result = run_suite_against_adapter(
        &suite,
        None,
        None,
        gen_,
        None,
        Arc::new(AtomicBool::new(false)),
        judge,
    )
    .await
    .unwrap();

    assert_eq!(result.outcomes.len(), 4);
    assert_eq!(result.metrics.num_pass, 3);
    assert_eq!(result.metrics.num_fail, 1);

    // Per-tool breakdown captures both Read's pass and fail.
    let weather = result
        .metrics
        .pass_rate_by_tool
        .get("get_weather")
        .expect("get_weather should appear in pass_rate_by_tool");
    assert_eq!(weather.num_examples, 2);
    assert_eq!(weather.num_pass, 2);

    let read = result
        .metrics
        .pass_rate_by_tool
        .get("Read")
        .expect("Read should appear in pass_rate_by_tool");
    assert_eq!(read.num_examples, 2);
    assert_eq!(read.num_pass, 1);

    // Reasoning length stats — exactly one example produced a `<think>` block.
    assert_eq!(result.metrics.reasoning_length.num_with_thinking, 1);
    assert!(result.metrics.reasoning_length.max_chars > 10);

    // No unclosed thinking and no non-XML format mismatch in this run.
    assert_eq!(result.metrics.num_unclosed_thinking, 0);
    assert_eq!(result.metrics.num_non_xml_tool_calls, 0);

    // Confusion matrix captures the one Read → Write swap.
    let read_row = result
        .metrics
        .confusion_by_tool
        .get("Read")
        .expect("Read row present in confusion matrix");
    assert_eq!(read_row.get("Read").copied(), Some(1));
    assert_eq!(read_row.get("Write").copied(), Some(1));
    let weather_row = result
        .metrics
        .confusion_by_tool
        .get("get_weather")
        .expect("weather row present in confusion matrix");
    assert_eq!(weather_row.get("get_weather").copied(), Some(2));
}

#[tokio::test]
async fn executor_runs_schema_validation_when_tools_declared() {
    use kiln_eval::scorers::{ArgsScoring, NameMatch, Scorer};
    use kiln_eval::EvalExample;

    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    })];

    let suite = EvalSuite {
        name: "schema-probe".into(),
        description: None,
        default_scorer: Scorer::ToolCall {
            name_match: NameMatch::CaseInsensitive,
            args: ArgsScoring::KeysOnly,
            weights: None,
        },
        generation: EvalGenerationParams::default(),
        system_prompt: None,
        examples: vec![EvalExample {
            id: Some("weather".into()),
            messages: vec![EvalChatMessage::new("user", "weather paris")],
            target: Some(
                r#"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris","units":"c"}}]}"#
                    .into(),
            ),
            ..Default::default()
        }],
        schema_version: 1,
        tools: Some(tools),
    };

    // Model emits the call but invents a `zone` arg and forgets `city`.
    let replies = vec![xml_call(
        "get_weather",
        &[("zone", "Europe"), ("units", "c")],
    )];
    let gen_ = Arc::new(OrderedMockGenerator::new(replies)) as Arc<dyn EvalGenerator>;
    let result = run_suite_against_adapter(
        &suite,
        None,
        None,
        gen_,
        None,
        Arc::new(AtomicBool::new(false)),
        Arc::new(NoopJudgeRunner) as Arc<dyn JudgeRunner>,
    )
    .await
    .unwrap();
    assert_eq!(result.metrics.num_schema_missing_required, 1);
    assert_eq!(result.metrics.num_schema_extra_unknown, 1);
    // The outcome's detail string carries the schema diagnostic so dashboards
    // can show "missing=city, extra=zone" inline next to the failure.
    let detail = result.outcomes[0]
        .detail
        .as_deref()
        .unwrap_or_default();
    assert!(
        detail.contains("missing=city"),
        "outcome detail missing schema note: {detail}"
    );
    assert!(
        detail.contains("extra=zone"),
        "outcome detail missing schema note: {detail}"
    );
}

#[tokio::test]
async fn executor_flags_non_xml_tool_call_in_metrics() {
    use kiln_eval::scorers::{ArgsScoring, NameMatch, Scorer};
    use kiln_eval::EvalExample;

    let suite = EvalSuite {
        name: "format-probe".into(),
        description: None,
        default_scorer: Scorer::ToolCall {
            name_match: NameMatch::CaseInsensitive,
            args: ArgsScoring::Structural,
            weights: None,
        },
        generation: EvalGenerationParams::default(),
        system_prompt: None,
        examples: vec![EvalExample {
            id: Some("json-output".into()),
            messages: vec![EvalChatMessage::new("user", "search for kiln")],
            target: Some(
                r#"{"tool_calls":[{"name":"search","arguments":{"q":"kiln"}}]}"#
                    .into(),
            ),
            ..Default::default()
        }],
        schema_version: 1,
        tools: None,
    };

    // Model emitted JSON instead of XML. Still scores correctly but the
    // `formats=[json]` diagnostic should bump `num_non_xml_tool_calls`.
    let replies =
        vec![r#"{"tool_calls":[{"name":"search","arguments":{"q":"kiln"}}]}"#.to_string()];
    let gen_ = Arc::new(OrderedMockGenerator::new(replies)) as Arc<dyn EvalGenerator>;
    let result = run_suite_against_adapter(
        &suite,
        None,
        None,
        gen_,
        None,
        Arc::new(AtomicBool::new(false)),
        Arc::new(NoopJudgeRunner) as Arc<dyn JudgeRunner>,
    )
    .await
    .unwrap();
    assert_eq!(result.metrics.num_pass, 1);
    assert_eq!(result.metrics.num_non_xml_tool_calls, 1);
}

#[tokio::test]
async fn executor_marks_unclosed_thinking_as_invalid() {
    use kiln_eval::scorers::Scorer;
    use kiln_eval::EvalExample;

    let suite = EvalSuite {
        name: "thinking-overrun".into(),
        description: None,
        default_scorer: Scorer::ExactMatch {
            case_sensitive: false,
            strip_whitespace: true,
        },
        generation: EvalGenerationParams::default(),
        system_prompt: None,
        examples: vec![EvalExample {
            id: Some("ran-out-of-budget".into()),
            messages: vec![EvalChatMessage::new("user", "What is 2+2?")],
            target: Some("4".into()),
            ..Default::default()
        }],
        schema_version: 1,
        tools: None,
    };

    // Model opened `<think>` then ran into max_tokens before closing.
    let replies = vec![
        "<think>\nLet me work through this step by step... 2 plus 2 is".to_string(),
    ];
    let gen_ = Arc::new(OrderedMockGenerator::new(replies)) as Arc<dyn EvalGenerator>;
    let result = run_suite_against_adapter(
        &suite,
        None,
        None,
        gen_,
        None,
        Arc::new(AtomicBool::new(false)),
        Arc::new(NoopJudgeRunner) as Arc<dyn JudgeRunner>,
    )
    .await
    .unwrap();
    assert_eq!(result.metrics.num_invalid, 1);
    assert_eq!(result.metrics.num_unclosed_thinking, 1);
    let outcome = &result.outcomes[0];
    assert!(outcome.unclosed_thinking);
    assert!(outcome.reasoning_text.is_some());
}
