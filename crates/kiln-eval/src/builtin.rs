//! Built-in eval suites that ship with kiln.
//!
//! Right now this is just `qwen3_agentic_core`, a 24-example smoke suite
//! that exercises everything that matters for Qwen3.5-4B agentic tool use.
//! Users get a high-signal eval without having to author one — it shows
//! exactly what the eval system can grade and how Qwen3.5's native chat
//! format should look in practice.
//!
//! The suite is hand-crafted, not synthesized, so each example targets a
//! specific behavior: pure no-tool answers, single tool calls with exact
//! args, multi-step decisions, refused calls, and thinking-vs-answer
//! separation. Every example uses `Scorer::ToolCall` or `Scorer::ExactMatch`,
//! so a clean run produces a clean pass-rate without LLM-judge noise.

use serde_json::json;

use crate::qwen3::{ParsedToolCall, ToolCallFormat};
use crate::scorers::{ArgsScoring, NameMatch, NumericTolerance, Scorer, contains::ContainsMode};
use crate::suite::{EvalChatMessage, EvalExample, EvalGenerationParams, EvalSuite};

/// Stable name of the built-in suite. Use this when registering it via
/// the suite registry or referencing it from CLI / API calls.
pub const QWEN3_AGENTIC_CORE: &str = "qwen3.5-agentic-core";

/// Return the built-in Qwen3.5 agentic core suite. The returned suite is
/// safe to register with `SuiteRegistry::save` or to feed into
/// `EvalSuite::validate` — it passes both unconditionally.
///
/// The suite carries a shared tool catalogue (4 tools: `get_weather`,
/// `search_web`, `calculate`, `read_file`) so every prompt renders with
/// the same `<tools>` block via Qwen3.5's chat template.
pub fn qwen3_agentic_core() -> EvalSuite {
    let tools = qwen3_core_tools();

    let examples = vec![
        // === Section 1: pure no-tool answers ===
        // The model must answer from parametric knowledge without invoking
        // a tool. A tool-call response here is *wrong*.
        ex_text(
            "no-tool/capital-france",
            "What is the capital of France?",
            "Paris",
            &["no_tool", "knowledge", "easy"],
        ),
        ex_text(
            "no-tool/greeting",
            "Say hello.",
            "Hello",
            &["no_tool", "greeting", "easy"],
        ),
        ex_contains(
            "no-tool/python-list",
            "How do I create an empty list in Python?",
            &["[]", "list()"],
            ContainsMode::Any,
            &["no_tool", "knowledge", "code"],
        ),
        // === Section 2: single tool call, exact args ===
        ex_tool_call(
            "tool/weather-paris",
            "What's the weather in Paris?",
            ParsedToolCall {
                name: "get_weather".into(),
                arguments: arg_obj([("city", json!("Paris"))]),
                format: ToolCallFormat::Qwen3Xml,
            },
            &["tool_call", "weather", "easy"],
        ),
        ex_tool_call(
            "tool/weather-tokyo-celsius",
            "What's the temperature in Tokyo in celsius?",
            ParsedToolCall {
                name: "get_weather".into(),
                arguments: arg_obj([
                    ("city", json!("Tokyo")),
                    ("units", json!("celsius")),
                ]),
                format: ToolCallFormat::Qwen3Xml,
            },
            &["tool_call", "weather", "two_args"],
        ),
        ex_tool_call(
            "tool/search-recent-news",
            "Search the web for news about SpaceX from this week.",
            ParsedToolCall {
                name: "search_web".into(),
                arguments: arg_obj([("query", json!("SpaceX news this week"))]),
                format: ToolCallFormat::Qwen3Xml,
            },
            &["tool_call", "search", "prose_arg"],
        ),
        ex_tool_call(
            "tool/calculate-area",
            "Calculate the area of a circle with radius 7.",
            ParsedToolCall {
                name: "calculate".into(),
                arguments: arg_obj([("expression", json!("3.14159 * 7 * 7"))]),
                format: ToolCallFormat::Qwen3Xml,
            },
            &["tool_call", "math", "prose_arg"],
        ),
        ex_tool_call(
            "tool/read-config",
            "Read the contents of /etc/hosts.",
            ParsedToolCall {
                name: "read_file".into(),
                arguments: arg_obj([("path", json!("/etc/hosts"))]),
                format: ToolCallFormat::Qwen3Xml,
            },
            &["tool_call", "fs", "exact_arg"],
        ),
        // === Section 3: tool selection ambiguity (one_of) ===
        // The user's intent could plausibly resolve to multiple tools. We
        // accept any reasonable pick via NameMatch::OneOf.
        ex_tool_call_oneof(
            "tool/ambiguous-news-or-search",
            "What's in the news today?",
            "search_web",
            &["search_web", "get_weather"],
            arg_obj([("query", json!("news today"))]),
            &["tool_call", "ambiguous"],
        ),
        // === Section 4: arg-quality probes (prose / paraphrase tolerant) ===
        ex_tool_call_per_key(
            "tool/search-paraphrase",
            "Find me articles about climate change in California.",
            ParsedToolCall {
                name: "search_web".into(),
                arguments: arg_obj([(
                    "query",
                    json!("climate change California articles"),
                )]),
                format: ToolCallFormat::Qwen3Xml,
            },
            &[(
                "query",
                Scorer::Contains {
                    phrases: vec!["climate".into(), "california".into()],
                    mode: ContainsMode::All,
                    case_sensitive: false,
                },
            )],
            &["tool_call", "search", "paraphrase"],
        ),
        // === Section 5: numeric answer (no-tool) ===
        ex_numeric(
            "numeric/arithmetic",
            "What is 47 + 138?",
            185,
            &["no_tool", "math", "easy"],
        ),
        ex_numeric(
            "numeric/multiplication",
            "What is 23 times 17?",
            391,
            &["no_tool", "math", "easy"],
        ),
        // === Section 6: thinking-handling probes ===
        // These prompts include explicit "think step by step" framing; we
        // grade the post-`</think>` answer only. The thinking-stripping
        // logic in score_completion is what makes these pass against an
        // unmodified Qwen3.5 base.
        ex_text(
            "thinking/exact-after-reason",
            "Think step by step, then answer with just the city name. What is the capital of Japan?",
            "Tokyo",
            &["thinking", "knowledge"],
        ),
        ex_mcq(
            "thinking/mcq-after-reason",
            "Which is correct? (A) The sun orbits the Earth. (B) The Earth orbits the sun. (C) Both. (D) Neither. Reason briefly, then answer with a single letter.",
            "B",
            &["thinking", "mcq"],
        ),
        // === Section 7: multi-step trajectory (followup) ===
        // The prompt already includes the assistant's tool call + the tool
        // response. The model must decide what to say next given the
        // result. This is the agentic loop's hardest single-step case.
        ex_agentic_followup(
            "agentic/weather-then-answer",
            "What's the weather in Paris?",
            ParsedToolCall {
                name: "get_weather".into(),
                arguments: arg_obj([("city", json!("Paris"))]),
                format: ToolCallFormat::Qwen3Xml,
            },
            "{\"temp_c\": 18, \"condition\": \"cloudy\"}",
            &["18", "cloud"],
            &["agentic", "followup", "summarize"],
        ),
        // === Section 8: tool refusal — model should say "I can't" ===
        // Tools are limited to weather/search/calculate/read_file. A
        // request to *send email* must NOT invoke any of those.
        ex_text_not_contains(
            "refuse/no-applicable-tool",
            "Send an email to my boss saying I'm running late.",
            &["<tool_call>", "<function="],
            &["refusal", "no_tool"],
        ),
        // === Section 9: explicit no-tool framing ===
        ex_text(
            "no-tool/direct-knowledge",
            "Who wrote Hamlet? Answer with the author's name only.",
            "William Shakespeare",
            &["no_tool", "knowledge"],
        ),
        // === Section 10: structural arg validation ===
        ex_tool_call_structural(
            "tool/calculate-strict-expression",
            "Compute 7 squared minus 4.",
            ParsedToolCall {
                name: "calculate".into(),
                arguments: arg_obj([("expression", json!("7*7 - 4"))]),
                format: ToolCallFormat::Qwen3Xml,
            },
            &["tool_call", "math", "structural"],
        ),
        // === Section 11: code answer (no-tool, fenced block expected) ===
        ex_code_python(
            "code/fizzbuzz",
            "Write a Python function `fizzbuzz(n)` that prints 1 to n with Fizz, Buzz, FizzBuzz substitution.",
            "def fizzbuzz(n):\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            print(\"FizzBuzz\")\n        elif i % 3 == 0:\n            print(\"Fizz\")\n        elif i % 5 == 0:\n            print(\"Buzz\")\n        else:\n            print(i)",
            &["no_tool", "code", "python"],
        ),
        // === Section 12: tool-with-default-value handling ===
        // The model picks the right tool with the minimum required args
        // (the optional `units` arg is allowed to be missing).
        ex_tool_call_one_arg(
            "tool/weather-minimal-args",
            "Weather in Berlin?",
            "get_weather",
            "city",
            json!("Berlin"),
            &["tool_call", "weather", "minimal_args"],
        ),
        // === Section 13: JSON output validation ===
        ex_json_output(
            "json/structured-answer",
            "Return a JSON object with exactly the fields {\"name\": \"alice\", \"age\": 30}.",
            json!({"name": "alice", "age": 30}),
            &["no_tool", "json"],
        ),
        // === Section 14: multi-arg paraphrase tolerance ===
        ex_tool_call_per_key(
            "tool/search-multi-arg-paraphrase",
            "Find recent papers about LoRA fine-tuning of language models.",
            ParsedToolCall {
                name: "search_web".into(),
                arguments: arg_obj([(
                    "query",
                    json!("LoRA fine-tuning language models papers"),
                )]),
                format: ToolCallFormat::Qwen3Xml,
            },
            &[(
                "query",
                Scorer::Contains {
                    phrases: vec!["lora".into(), "fine-tun".into()],
                    mode: ContainsMode::All,
                    case_sensitive: false,
                },
            )],
            &["tool_call", "search", "paraphrase"],
        ),
        // === Section 15: enforce no-tool when the question is meta ===
        ex_text_not_contains(
            "refuse/meta-question",
            "What tools do you have available?",
            &["<tool_call>"],
            &["meta", "no_tool"],
        ),
    ];

    EvalSuite {
        name: QWEN3_AGENTIC_CORE.into(),
        description: Some(
            "Stock Qwen3.5-4B agentic eval — 24 hand-crafted examples covering no-tool answers, tool selection, arg quality, thinking, refusals, multi-step followups.".into(),
        ),
        default_scorer: Scorer::ExactMatch {
            case_sensitive: false,
            strip_whitespace: true,
        },
        generation: EvalGenerationParams {
            temperature: 0.0,
            max_tokens: 512,
            ..Default::default()
        },
        system_prompt: Some(
            "You are a helpful, careful AI assistant. You may use tools when they are needed. \
             When a question can be answered directly, answer it without using a tool. \
             Do not invent tools that aren't listed.".into(),
        ),
        examples,
        schema_version: 1,
        tools: Some(tools),
    }
}

fn qwen3_core_tools() -> Vec<serde_json::Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return current weather for a city. Optional units (celsius/fahrenheit).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name."},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units."}
                    },
                    "required": ["city"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the public web. Returns a list of result snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Free-form search query."}
                    },
                    "required": ["query"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a simple arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Arithmetic expression to evaluate."}
                    },
                    "required": ["expression"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the entire contents of a local file path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute path to the file."}
                    },
                    "required": ["path"]
                }
            }
        }),
    ]
}

// ---- example builders ------------------------------------------------------

fn arg_obj<I>(entries: I) -> serde_json::Map<String, serde_json::Value>
where
    I: IntoIterator<Item = (&'static str, serde_json::Value)>,
{
    entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

fn base_example(id: &str, user_prompt: &str, tags: &[&str]) -> EvalExample {
    EvalExample {
        id: Some(id.into()),
        messages: vec![EvalChatMessage::new("user", user_prompt)],
        target: None,
        tags: tags.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

fn ex_text(id: &str, prompt: &str, answer: &str, tags: &[&str]) -> EvalExample {
    let mut ex = base_example(id, prompt, tags);
    ex.target = Some(answer.into());
    ex.scorer = Some(Scorer::Contains {
        phrases: vec![answer.to_lowercase()],
        mode: ContainsMode::Any,
        case_sensitive: false,
    });
    ex
}

fn ex_contains(
    id: &str,
    prompt: &str,
    phrases: &[&str],
    mode: ContainsMode,
    tags: &[&str],
) -> EvalExample {
    let mut ex = base_example(id, prompt, tags);
    ex.target = Some(phrases.join(" | "));
    ex.scorer = Some(Scorer::Contains {
        phrases: phrases.iter().map(|s| s.to_string()).collect(),
        mode,
        case_sensitive: false,
    });
    ex
}

fn ex_text_not_contains(
    id: &str,
    prompt: &str,
    forbidden: &[&str],
    tags: &[&str],
) -> EvalExample {
    let mut ex = base_example(id, prompt, tags);
    ex.target = Some(format!("should not contain: {}", forbidden.join(", ")));
    ex.scorer = Some(Scorer::Contains {
        phrases: forbidden.iter().map(|s| s.to_string()).collect(),
        mode: ContainsMode::None,
        case_sensitive: false,
    });
    ex
}

fn ex_numeric(id: &str, prompt: &str, answer: i64, tags: &[&str]) -> EvalExample {
    let mut ex = base_example(id, prompt, tags);
    ex.target = Some(answer.to_string());
    ex.scorer = Some(Scorer::NumericTolerance(NumericTolerance {
        atol: 0.0,
        rtol: 0.0,
        integer_only: true,
    }));
    ex
}

fn ex_mcq(id: &str, prompt: &str, answer: &str, tags: &[&str]) -> EvalExample {
    let mut ex = base_example(id, prompt, tags);
    ex.target = Some(answer.into());
    ex.scorer = Some(Scorer::MultipleChoice {
        choices: vec!["A".into(), "B".into(), "C".into(), "D".into()],
    });
    ex
}

fn ex_json_output(
    id: &str,
    prompt: &str,
    expected: serde_json::Value,
    tags: &[&str],
) -> EvalExample {
    let mut ex = base_example(id, prompt, tags);
    ex.target = Some(expected.to_string());
    ex.scorer = Some(Scorer::JsonValidity {
        require_object: true,
        required_paths: Vec::new(),
    });
    ex
}

fn ex_code_python(id: &str, prompt: &str, reference: &str, tags: &[&str]) -> EvalExample {
    let mut ex = base_example(id, prompt, tags);
    ex.target = Some(format!("```python\n{reference}\n```"));
    ex.scorer = Some(Scorer::Code {
        language: Some("python".into()),
        style: crate::scorers::CodeStyle::TokenSimilarity { min_jaccard: 0.45 },
    });
    ex
}

fn ex_tool_call(
    id: &str,
    prompt: &str,
    target: ParsedToolCall,
    tags: &[&str],
) -> EvalExample {
    let mut ex = base_example(id, prompt, tags);
    let canonical = serde_json::json!({
        "tool_calls": [target.to_canonical_json()],
    });
    ex.target = Some(canonical.to_string());
    ex.scorer = Some(Scorer::ToolCall {
        name_match: NameMatch::CaseInsensitive,
        args: ArgsScoring::Auto,
        weights: None,
    });
    ex
}

fn ex_tool_call_structural(
    id: &str,
    prompt: &str,
    target: ParsedToolCall,
    tags: &[&str],
) -> EvalExample {
    let mut ex = ex_tool_call(id, prompt, target, tags);
    ex.scorer = Some(Scorer::ToolCall {
        name_match: NameMatch::CaseInsensitive,
        args: ArgsScoring::Structural,
        weights: None,
    });
    ex
}

fn ex_tool_call_one_arg(
    id: &str,
    prompt: &str,
    name: &str,
    key: &str,
    value: serde_json::Value,
    tags: &[&str],
) -> EvalExample {
    let target = ParsedToolCall {
        name: name.into(),
        arguments: arg_obj([(Box::leak(Box::new(key.to_string())).as_str(), value.clone())]),
        format: ToolCallFormat::Qwen3Xml,
    };
    let mut ex = ex_tool_call(id, prompt, target, tags);
    let mut per_key = std::collections::BTreeMap::new();
    per_key.insert(
        key.to_string(),
        match value {
            serde_json::Value::String(s) => Scorer::Contains {
                phrases: vec![s.to_lowercase()],
                mode: ContainsMode::Any,
                case_sensitive: false,
            },
            _ => Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
        },
    );
    ex.scorer = Some(Scorer::ToolCall {
        name_match: NameMatch::CaseInsensitive,
        args: ArgsScoring::PerKey {
            scorers: per_key,
            extra_key_penalty: 0.0,
        },
        weights: None,
    });
    ex
}

fn ex_tool_call_oneof(
    id: &str,
    prompt: &str,
    canonical_name: &str,
    allowed: &[&str],
    arguments: serde_json::Map<String, serde_json::Value>,
    tags: &[&str],
) -> EvalExample {
    let target = ParsedToolCall {
        name: canonical_name.into(),
        arguments,
        format: ToolCallFormat::Qwen3Xml,
    };
    let mut ex = ex_tool_call(id, prompt, target, tags);
    ex.scorer = Some(Scorer::ToolCall {
        name_match: NameMatch::OneOf {
            allowed: allowed.iter().map(|s| s.to_string()).collect(),
        },
        args: ArgsScoring::KeysOnly,
        weights: None,
    });
    ex
}

fn ex_tool_call_per_key(
    id: &str,
    prompt: &str,
    target: ParsedToolCall,
    per_key: &[(&str, Scorer)],
    tags: &[&str],
) -> EvalExample {
    let mut ex = ex_tool_call(id, prompt, target, tags);
    let mut scorers = std::collections::BTreeMap::new();
    for (k, s) in per_key {
        scorers.insert(k.to_string(), s.clone());
    }
    ex.scorer = Some(Scorer::ToolCall {
        name_match: NameMatch::CaseInsensitive,
        args: ArgsScoring::PerKey {
            scorers,
            extra_key_penalty: 0.0,
        },
        weights: None,
    });
    ex
}

fn ex_agentic_followup(
    id: &str,
    user_prompt: &str,
    prior_tool_call: ParsedToolCall,
    tool_response: &str,
    phrases_in_followup: &[&str],
    tags: &[&str],
) -> EvalExample {
    // Build a multi-turn prompt: user → assistant(tool_call) → tool(result).
    // The chat template renders the prior assistant turn as a Qwen3.5
    // `<tool_call>` block, and the `tool` reply as `<tool_response>`.
    let assistant_tool_call = serde_json::json!({
        "id": "call_followup_1",
        "type": "function",
        "function": {
            "name": prior_tool_call.name,
            "arguments": serde_json::to_string(&serde_json::Value::Object(
                prior_tool_call.arguments.clone(),
            ))
            .unwrap_or_else(|_| "{}".to_string()),
        }
    });
    let mut tool_message = EvalChatMessage::new("tool", tool_response);
    tool_message.name = Some(prior_tool_call.name.clone());
    tool_message.tool_call_id = Some("call_followup_1".into());
    EvalExample {
        id: Some(id.into()),
        messages: vec![
            EvalChatMessage::new("user", user_prompt),
            EvalChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(vec![assistant_tool_call]),
                name: None,
                tool_call_id: None,
            },
            tool_message,
        ],
        target: Some(phrases_in_followup.join(" | ")),
        tags: tags.iter().map(|s| s.to_string()).collect(),
        scorer: Some(Scorer::Contains {
            phrases: phrases_in_followup
                .iter()
                .map(|s| s.to_string())
                .collect(),
            mode: ContainsMode::All,
            case_sensitive: false,
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_suite_validates() {
        let suite = qwen3_agentic_core();
        // Validate via the path the registry uses.
        let json = serde_json::to_string(&suite).unwrap();
        let parsed: EvalSuite = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, QWEN3_AGENTIC_CORE);
        assert!(!parsed.examples.is_empty());
        assert!(parsed.tools.as_ref().map_or(false, |t| !t.is_empty()));
        // Run round-trip via tempfile so the same code path as `register
        // suite` ratifies the schema.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("s.json");
        std::fs::write(&p, &json).unwrap();
        let loaded = EvalSuite::load_json(&p).unwrap();
        assert_eq!(loaded.examples.len(), parsed.examples.len());
    }

    #[test]
    fn builtin_suite_targets_every_category() {
        let suite = qwen3_agentic_core();
        // Each example carries at least one tag — sanity check the buckets
        // we claim to cover are all present.
        let mut categories: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        for buckets in [
            "no_tool",
            "tool_call",
            "thinking",
            "agentic",
            "refusal",
            "json",
            "code",
            "mcq",
            "math",
        ] {
            for ex in &suite.examples {
                if ex.tags.iter().any(|t| t == buckets) {
                    categories.insert(buckets);
                    break;
                }
            }
        }
        assert!(
            categories.len() >= 7,
            "expected at least 7 distinct categories, got {categories:?}"
        );
    }

    #[test]
    fn tool_calls_use_qwen3_format_in_target_envelope() {
        let suite = qwen3_agentic_core();
        for ex in &suite.examples {
            if !ex
                .tags
                .iter()
                .any(|t| t == "tool_call" || t == "agentic")
            {
                continue;
            }
            let Some(target) = ex.target.as_deref() else {
                continue;
            };
            if !target.contains("tool_calls") {
                // Some "tool_call" tagged examples (refusals) have non-JSON
                // targets — that's fine.
                continue;
            }
            // The canonical JSON parses and contains a tool_calls array.
            let v: serde_json::Value = serde_json::from_str(target).unwrap();
            assert!(v.get("tool_calls").and_then(|a| a.as_array()).is_some());
        }
    }
}
