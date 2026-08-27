use super::batch::*;
use super::*;
use crate::latency_observability::TokenPhaseDurations;
use kiln_core::config::ModelConfig;

struct PromptLogprobDropProbe(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl Drop for PromptLogprobDropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

fn parse_request(json: &str) -> ChatCompletionRequest {
    serde_json::from_str(json).expect("request should deserialize")
}

#[test]
fn rollout_provenance_request_is_opt_in_and_resolves_one_effective_seed() {
    let default = parse_request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    assert!(!default.rollout_provenance);

    let mut generated =
        parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"rollout_provenance":true}"#);
    assert!(generated.rollout_provenance);
    assert!(generated.seed.is_none());
    resolve_rollout_seed(&mut generated);
    let resolved = generated.seed.expect("rollout seed must be resolved");
    resolve_rollout_seed(&mut generated);
    assert_eq!(generated.seed, Some(resolved));

    let mut explicit = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],"rollout_provenance":true,"seed":42}"#,
    );
    resolve_rollout_seed(&mut explicit);
    assert_eq!(explicit.seed, Some(42));
    let sampling = sampling_params_for_chat_request(&explicit);
    assert!(
        deterministic_chat_request_cache_key(&explicit, &sampling)
            .unwrap()
            .is_none(),
        "trace-bearing responses must never use the text-only request cache"
    );
}

#[test]
fn rollout_provenance_reserves_capacity_for_a_sampled_action() {
    let req =
        parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"rollout_provenance":true}"#);
    let mut sampling = SamplingParams {
        max_tokens: 0,
        ..SamplingParams::default()
    };
    let error = validate_rollout_provenance_generation_capacity(&req, &sampling).unwrap_err();
    assert!(error.message.contains("greater than zero"));

    sampling.max_tokens = 2;
    sampling.thinking_budget = Some(
        ThinkingBudget::new(Some(0), None, 2, vec![10, 11])
            .expect("two-token completion can fit only the close sequence"),
    );
    let error = validate_rollout_provenance_generation_capacity(&req, &sampling).unwrap_err();
    assert!(error.message.contains("at least one sampled action"));

    sampling.max_tokens = 3;
    sampling.thinking_budget = Some(
        ThinkingBudget::new(Some(0), None, 3, vec![10, 11])
            .expect("three-token completion leaves one sampled action"),
    );
    validate_rollout_provenance_generation_capacity(&req, &sampling).unwrap();

    let ordinary = parse_request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    sampling.max_tokens = 0;
    validate_rollout_provenance_generation_capacity(&ordinary, &sampling).unwrap();
}

#[tokio::test]
async fn rollout_provenance_rejects_unsupported_paths_before_generation() {
    let (status, body) = chat_post(
            make_batch_test_state(),
            r#"{"messages":[{"role":"user","content":"hi"}],"rollout_provenance":true,"ignore_eos":true}"#,
        )
        .await;
    assert_eq!(status, axum::http::StatusCode::NOT_IMPLEMENTED);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("ignore_eos=true")
    );

    let (status, body) = chat_post(
        make_batch_test_state(),
        r#"{"messages":[{"role":"user","content":"hi"}],"rollout_provenance":true}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["error"]["code"], "rollout_provenance_unavailable");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mock backend")
    );

    let (status, body) = chat_post(
        make_batch_test_state(),
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"rollout_provenance":true}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::NOT_IMPLEMENTED);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stream=true")
    );

    let (status, body) = chat_post(
            make_batch_test_state(),
            r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"type":"function"}]}],"rollout_provenance":true}"#,
        )
        .await;
    assert_eq!(status, axum::http::StatusCode::NOT_IMPLEMENTED);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mock backend")
    );

    let (status, body) = chat_post(
            make_batch_test_state(),
            r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}],"rollout_provenance":true}"#,
        )
        .await;
    assert_eq!(status, axum::http::StatusCode::NOT_IMPLEMENTED);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("tool_choice")
    );
}

#[test]
fn thinking_budget_request_fields_preserve_inherit_unlimited_and_zero() {
    let inherited = parse_request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(inherited.thinking_budget_tokens, BudgetOverride::Inherit);
    assert_eq!(inherited.thinking_budget_ms, BudgetOverride::Inherit);

    let unlimited = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],
                "thinking_budget_tokens":null,"thinking_budget_ms":null}"#,
    );
    assert_eq!(unlimited.thinking_budget_tokens, BudgetOverride::Unlimited);
    assert_eq!(unlimited.thinking_budget_ms, BudgetOverride::Unlimited);

    let immediate = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],
                "thinking_budget_tokens":0,"thinking_budget_ms":0}"#,
    );
    assert_eq!(immediate.thinking_budget_tokens, BudgetOverride::Limited(0));
    assert_eq!(immediate.thinking_budget_ms, BudgetOverride::Limited(0));
}

#[test]
fn thinking_budget_configuration_resolves_defaults_and_request_unlimited() {
    let mut state = make_qwen_template_test_state();
    let tokenizer = kiln_core::tokenizer::KilnTokenizer::from_bytes(
        br#"{
                "version":"1.0",
                "model":{
                    "type":"WordLevel",
                    "vocab":{"</think>":0,"<unk>":1},
                    "unk_token":"<unk>"
                }
            }"#,
    )
    .unwrap();
    state.tokenizer = std::sync::Arc::new(tokenizer);
    state.default_thinking_budget_tokens = Some(12);
    state.default_thinking_budget_ms = Some(250);
    let inherited =
        parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}"#);
    let mut sampling = sampling_params_for_chat_request(&inherited);

    configure_thinking_budget_for_prompt(
        &state,
        &inherited,
        "<|im_start|>assistant\n<think>\n\n",
        &mut sampling,
    )
    .unwrap();
    let budget = sampling.thinking_budget.as_ref().unwrap();
    assert_eq!(budget.max_tokens(), Some(12));
    assert_eq!(
        budget.max_time(),
        Some(std::time::Duration::from_millis(250))
    );

    let request_override = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":64,
                "thinking_budget_tokens":null,"thinking_budget_ms":0}"#,
    );
    let mut sampling = sampling_params_for_chat_request(&request_override);
    configure_thinking_budget_for_prompt(
        &state,
        &request_override,
        "<|im_start|>assistant\n<think>\n",
        &mut sampling,
    )
    .unwrap();
    let budget = sampling.thinking_budget.as_ref().unwrap();
    assert_eq!(budget.max_tokens(), None);
    assert_eq!(budget.max_time(), Some(std::time::Duration::ZERO));
}

#[test]
fn thinking_budget_is_inert_outside_reasoning_and_for_zero_output() {
    let mut state = make_qwen_template_test_state();
    state.default_thinking_budget_tokens = Some(8);
    let req = parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}"#);
    let mut sampling = sampling_params_for_chat_request(&req);
    configure_thinking_budget_for_prompt(&state, &req, "<|im_start|>assistant\n", &mut sampling)
        .unwrap();
    assert!(sampling.thinking_budget.is_none());

    let zero = parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#);
    let metadata =
        chat_completion_metadata_from_prompt(&state, &zero, "<|im_start|>assistant\n<think>\n");
    assert!(metadata.thinking_budget.configured);
    assert!(!metadata.thinking_budget.applied);
}

#[test]
fn recent_thinking_budget_preserves_unresolved_inert_and_final_states() {
    let mut state = make_qwen_template_test_state();
    state.default_thinking_budget_tokens = Some(12);
    state.default_thinking_budget_ms = Some(250);
    let req = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":64,
                "thinking_budget_tokens":32,"thinking_budget_ms":null}"#,
    );

    let unresolved = unresolved_request_thinking_budget(&state, &req);
    assert!(unresolved.configured);
    assert_eq!(unresolved.max_tokens, Some(32));
    assert_eq!(unresolved.max_time_ms, None);
    assert_eq!(unresolved.tokens_source, "request");
    assert_eq!(unresolved.time_source, "request_unlimited");
    assert_eq!(unresolved.applied, None);

    let inert_metadata = thinking_budget_metadata_for_request(&state, &req, false);
    let inert = recent_thinking_budget_from_metadata(&inert_metadata);
    assert_eq!(inert.applied, Some(false));
    assert_eq!(inert.triggered, None);
    assert_eq!(inert.closed, None);

    let active_metadata = thinking_budget_metadata_for_request(&state, &req, true);
    let closed = recent_thinking_budget_with_status(
        &active_metadata,
        Some(ThinkingBudgetStatus {
            trigger: Some(kiln_core::sampling::ThinkingBudgetTrigger::Tokens),
            closed: true,
            thinking_tokens: 32,
            elapsed_ms: 75,
        }),
    );
    assert_eq!(closed.applied, Some(true));
    assert_eq!(closed.triggered, Some(true));
    assert_eq!(closed.trigger.as_deref(), Some("tokens"));
    assert_eq!(closed.closed, Some(true));
    assert_eq!(closed.thinking_tokens, Some(32));
    assert_eq!(closed.thinking_time_ms, Some(75));
}

#[test]
fn thinking_budget_rejects_stop_sequences_that_can_intercept_the_close_tag() {
    for stop in ["", "think", "</think>\n", "think>\n", "prefix</thi"] {
        assert!(
            stop_sequence_conflicts_with_thinking_close(stop),
            "expected conflict for {stop:?}"
        );
    }
    assert!(!stop_sequence_conflicts_with_thinking_close("<|im_end|>"));

    let state = make_qwen_template_test_state();
    let req = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":64,
                "thinking_budget_tokens":8,"stop":["</think>\n"]}"#,
    );
    let mut sampling = sampling_params_for_chat_request(&req);
    let error = configure_thinking_budget_for_prompt(
        &state,
        &req,
        "<|im_start|>assistant\n<think>\n",
        &mut sampling,
    )
    .unwrap_err();
    assert!(error.message.contains("stop sequence"));
    assert!(error.message.contains("</think>\\n"));
}

#[test]
fn thinking_budget_prevalidates_effective_completion_capacity() {
    validate_thinking_budget_completion_capacity(2, 2).unwrap();
    validate_thinking_budget_completion_capacity(64, 2).unwrap();

    let error = validate_thinking_budget_completion_capacity(1, 2).unwrap_err();
    assert!(error.message.contains("effective max_tokens 1"));
    assert!(error.message.contains("2-token"));
    assert!(error.message.contains(REASONING_CLOSE_TAG));
}

#[test]
fn thinking_budget_rejects_negative_request_values() {
    assert!(
        serde_json::from_str::<ChatCompletionRequest>(
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking_budget_tokens":-1}"#,
        )
        .is_err()
    );
    assert!(
        serde_json::from_str::<ChatCompletionRequest>(
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking_budget_ms":-1}"#,
        )
        .is_err()
    );
}

#[test]
fn token_thinking_budget_is_part_of_chat_cache_identity() {
    let a = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],
                "temperature":0,"thinking_budget_tokens":8}"#,
    );
    let b = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],
                "temperature":0,"thinking_budget_tokens":9}"#,
    );
    let key_a = deterministic_chat_request_cache_key(&a, &sampling_params_for_chat_request(&a))
        .unwrap()
        .unwrap();
    let key_b = deterministic_chat_request_cache_key(&b, &sampling_params_for_chat_request(&b))
        .unwrap()
        .unwrap();
    assert_ne!(key_a, key_b);
}

#[test]
fn deterministic_completion_cache_keys_token_budgets_and_rejects_time_budgets() {
    let state = make_qwen_template_test_state();
    let mut token_sampling = SamplingParams::greedy();
    token_sampling.max_tokens = 16;
    token_sampling.thinking_budget =
        Some(ThinkingBudget::new(Some(4), None, token_sampling.max_tokens, vec![90]).unwrap());
    let token_key =
        deterministic_completion_cache_key(&state, &[1, 2, 3], &token_sampling, false).unwrap();
    assert_eq!(token_key.thinking_budget_tokens, Some(4));

    let mut time_sampling = SamplingParams::greedy();
    time_sampling.max_tokens = 16;
    time_sampling.thinking_budget = Some(
        ThinkingBudget::new(
            None,
            Some(std::time::Duration::from_millis(1)),
            time_sampling.max_tokens,
            vec![90],
        )
        .unwrap(),
    );
    assert!(
        deterministic_completion_cache_key(&state, &[1, 2, 3], &time_sampling, false).is_none()
    );
}

#[test]
fn thinking_budget_outcome_serializes_and_survives_cache_conversion() {
    let status = ThinkingBudgetStatus {
        trigger: Some(kiln_core::sampling::ThinkingBudgetTrigger::Tokens),
        closed: true,
        thinking_tokens: 8,
        elapsed_ms: 17,
    };
    let response = ChatCompletionResponse {
        id: "chatcmpl-budget".to_string(),
        object: "chat.completion",
        created: 1,
        model: "test".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: "answer".to_string(),
                reasoning_content: Some("reasoning".to_string()),
                tool_calls: None,
                name: None,
                tool_call_id: None,
            },
            finish_reason: "stop".to_string(),
            thinking_budget: Some(status),
            rollout_provenance: None,
            completion_tokens: 10,
        }],
        usage: Usage {
            prompt_tokens: 4,
            completion_tokens: 10,
            total_tokens: 14,
        },
        metadata: ChatCompletionMetadata {
            thinking_enabled: true,
            thinking_mode: "reasoning".to_string(),
            thinking_source: "request",
            default_thinking_enabled: None,
            final_content_empty: false,
            content_empty_reason: None,
            reasoning_folded_into_content: false,
            thinking_budget: ThinkingBudgetMetadata::default(),
            config_hashes: None,
            performance: None,
        },
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["choices"][0]["thinking_budget"]["triggered"], true);
    assert_eq!(json["choices"][0]["thinking_budget"]["trigger"], "tokens");
    assert_eq!(json["choices"][0]["thinking_budget"]["closed"], true);
    assert_eq!(json["choices"][0]["thinking_budget"]["thinking_tokens"], 8);
    assert_eq!(
        json["choices"][0]["thinking_budget"]["thinking_time_ms"],
        17
    );
    assert_eq!(
        cache_value_from_response(&response)
            .unwrap()
            .thinking_budget_status,
        Some(status)
    );
}

#[test]
fn streaming_finish_metadata_distinguishes_natural_and_unclosed_thinking() {
    let chunk = ChatCompletionChunk {
        id: "chatcmpl-budget".to_string(),
        object: "chat.completion.chunk",
        created: 1,
        model: "test".to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
    };
    let mut metadata = ThinkingBudgetMetadata {
        configured: true,
        max_tokens: Some(16),
        tokens_source: ThinkingBudgetSource::Request,
        applied: true,
        ..Default::default()
    };
    apply_thinking_budget_status_to_metadata(
        &mut metadata,
        ThinkingBudgetStatus {
            trigger: None,
            closed: true,
            thinking_tokens: 5,
            elapsed_ms: 9,
        },
    );
    let natural: serde_json::Value =
        serde_json::from_str(&streaming_finish_chunk_json(&chunk, &metadata, None)).unwrap();
    assert_eq!(natural["metadata"]["thinking_budget"]["triggered"], false);
    assert_eq!(natural["metadata"]["thinking_budget"]["closed"], true);

    let performance = ChatCompletionPerformanceMetadata {
        prompt_tokens: 12,
        completion_tokens: 3,
        ttft_ms: Some(40.0),
        prefill_ms: Some(10.0),
        actor_queue_ms: Some(12.0),
        actor_admission_ms: Some(3.0),
        actor_prefill_wall_ms: Some(20.0),
        resident_prefill_used: Some(true),
        decode_ms: Some(8.0),
        total_latency_ms: 60.0,
        decode_tokens_per_sec: Some(375.0),
        adapter_used: "base".to_string(),
        thinking_mode: "non_reasoning".to_string(),
        finish_reason: "length".to_string(),
        latency: None,
    };
    let timed: serde_json::Value = serde_json::from_str(&streaming_finish_chunk_json(
        &chunk,
        &metadata,
        Some(&performance),
    ))
    .unwrap();
    assert_eq!(timed["metadata"]["performance"]["actor_queue_ms"], 12.0);
    assert_eq!(timed["metadata"]["performance"]["actor_admission_ms"], 3.0);
    assert_eq!(
        timed["metadata"]["performance"]["resident_prefill_used"],
        true
    );
    assert_eq!(
        timed["metadata"]["performance"]["actor_prefill_wall_ms"],
        20.0
    );
    assert_eq!(
        timed["metadata"]["thinking_budget"]["thinking_tokens"],
        natural["metadata"]["thinking_budget"]["thinking_tokens"]
    );

    apply_thinking_budget_status_to_metadata(
        &mut metadata,
        ThinkingBudgetStatus {
            trigger: None,
            closed: false,
            thinking_tokens: 7,
            elapsed_ms: 11,
        },
    );
    let unclosed: serde_json::Value =
        serde_json::from_str(&streaming_finish_chunk_json(&chunk, &metadata, None)).unwrap();
    assert_eq!(unclosed["metadata"]["thinking_budget"]["closed"], false);
}

#[test]
fn finalized_stream_status_preserves_partial_forced_close_telemetry() {
    let forcing = ThinkingBudget::new(Some(0), None, 4, vec![90, 91]).unwrap();
    assert_eq!(forcing.apply(&[], 10), 90);
    let forcing_status = finalized_thinking_budget_status(Some(&forcing), 1).unwrap();
    assert_eq!(
        forcing_status.trigger,
        Some(kiln_core::sampling::ThinkingBudgetTrigger::Tokens)
    );
    assert!(!forcing_status.closed);
    assert_eq!(forcing_status.thinking_tokens, 0);

    let active = ThinkingBudget::new(Some(10), None, 16, vec![90]).unwrap();
    assert_eq!(active.apply(&[], 10), 10);
    let active_status = finalized_thinking_budget_status(Some(&active), 1).unwrap();
    assert_eq!(active_status.trigger, None);
    assert!(!active_status.closed);
    assert_eq!(active_status.thinking_tokens, 1);
}

/// pi always sends tools, and presence_penalty 1.5 punishes every
/// identifier reuse — tools-bearing preset-less requests must resolve
/// to the coding profile, explicit fields must still win, and the
/// cache key must match the resolved sampling by construction.
#[test]
fn tools_bearing_requests_default_to_the_coding_profile() {
    let plain = parse_request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    let p = sampling_params_for_chat_request(&plain);
    assert_eq!(p.temperature, 1.0, "no tools → thinking-general");
    assert_eq!(p.presence_penalty, 1.5);

    let tools = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"run"}}]}"#,
    );
    let p = sampling_params_for_chat_request(&tools);
    assert_eq!(p.temperature, 0.6, "tools → thinking-coding");
    assert_eq!(p.presence_penalty, 0.0);
    assert_eq!(p.top_p, 0.95);

    // Explicit fields override the profile.
    let explicit = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"run"}}],
                "temperature":0.9,"presence_penalty":1.0}"#,
    );
    let p = sampling_params_for_chat_request(&explicit);
    assert_eq!(p.temperature, 0.9);
    assert_eq!(p.presence_penalty, 1.0);

    // An explicit preset wins over the tools heuristic.
    let preset = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"run"}}],
                "sampling_preset":"qwen3-non-thinking-general"}"#,
    );
    let p = sampling_params_for_chat_request(&preset);
    assert_ne!(p.temperature, 0.6);

    // Cache key matches the resolved sampling for the sampled fields.
    let key = chat_request_sampling_for_cache_key(&tools, None);
    let resolved = sampling_params_for_chat_request(&tools);
    assert_eq!(key.temperature, resolved.temperature);
    assert_eq!(key.presence_penalty, resolved.presence_penalty);
    assert_eq!(key.top_k, resolved.top_k);
}

/// Stream-finish stop reconstruction: a tool call terminated by the
/// implicit `</tool_call>` stop (which the emit gates strip) must
/// still parse — and when no call parses, the stop text must NOT
/// leak back into content.
#[test]
fn stream_stop_reconstruction_preserves_tool_calls_without_leaking() {
    let xml = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n";
    let out = stream_assistant_output_with_stop_reconstruction(
        true,
        None,
        &format!("Sure.\n{xml}"),
        Some("</tool_call>"),
        "stop",
    );
    assert!(
        out.tool_calls.is_some(),
        "the reconstructed close tag must let the call parse: {out:?}"
    );
    assert_eq!(out.finish_reason, "tool_calls");
    assert!(!out.content.contains("<tool_call>"), "{:?}", out.content);

    // Plain text + a custom stop: nothing reconstructs, nothing leaks.
    let out = stream_assistant_output_with_stop_reconstruction(
        true,
        None,
        "I should check the file.\n",
        Some("Observation:"),
        "stop",
    );
    assert!(out.tool_calls.is_none());
    assert!(!out.content.contains("Observation:"));
    assert_eq!(out.content, "I should check the file.\n");
}

/// The eager-streaming contract: prose streams immediately, only a
/// possible `<tool_call>` tail (plus the whitespace run before it)
/// holds back, a confirmed tag freezes the wire, and false alarms
/// release every byte.
#[test]
fn tool_call_gate_streams_eagerly_with_tag_holdback() {
    let mut buf = String::new();
    let mut g = ToolCallGate::new(true);

    // Plain prose streams as it arrives — except trailing whitespace,
    // which always holds (a tag could start at the very next byte and
    // the finish path trim_end()s the pre-tag content).
    buf.push_str("Let me check that file. ");
    assert_eq!(&buf[g.advance(&buf)], "Let me check that file.");

    // The held space releases with the next prose; the possible tag
    // start (plus the whitespace before it) holds back.
    buf.push_str("Done.\n<tool");
    assert_eq!(&buf[g.advance(&buf)], " Done.");
    assert!(!g.confirmed());

    // Tag confirms: wire freezes; nothing more emits.
    buf.push_str("_call>\n<function=bash>");
    assert_eq!(g.advance(&buf).len(), 0);
    assert!(g.confirmed());
    buf.push_str("more payload");
    assert_eq!(g.advance(&buf).len(), 0);
    assert!(g.unsent(&buf).starts_with("\n<tool_call>"));

    // False alarm releases all bytes.
    let mut buf = String::new();
    let mut g = ToolCallGate::new(true);
    buf.push_str("compare a <tool");
    let r1 = g.advance(&buf);
    assert_eq!(&buf[r1], "compare a");
    buf.push_str("box> result");
    let r2 = g.advance(&buf);
    assert_eq!(&buf[r2], " <toolbox> result");
    assert!(!g.confirmed());
    assert!(g.unsent(&buf).is_empty());

    // Generics aren't tool tags.
    let mut buf = String::new();
    let mut g = ToolCallGate::new(true);
    buf.push_str("Vec<tool> and a < b hold");
    let r = g.advance(&buf);
    assert_eq!(&buf[r], "Vec<tool> and a < b hold");

    // Disabled gate is a pure pass-through.
    let mut buf = String::new();
    let mut g = ToolCallGate::new(false);
    buf.push_str("anything <tool_call> at all");
    let r = g.advance(&buf);
    assert_eq!(&buf[r], "anything <tool_call> at all");
}

/// stream_options.include_usage: the final chunk shape is the OpenAI
/// contract — empty choices, populated usage, exact totals.
#[test]
fn usage_chunk_matches_openai_contract() {
    let json = usage_chunk_json("chatcmpl-x", 123, "kiln", 100, 25);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["object"], "chat.completion.chunk");
    assert_eq!(v["choices"].as_array().unwrap().len(), 0);
    assert_eq!(v["usage"]["prompt_tokens"], 100);
    assert_eq!(v["usage"]["completion_tokens"], 25);
    assert_eq!(v["usage"]["total_tokens"], 125);

    // The request field parses and defaults off.
    let req = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],
                "stream":true,"stream_options":{"include_usage":true}}"#,
    );
    assert!(req.stream_options.unwrap().include_usage);
    let req = parse_request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    assert!(req.stream_options.is_none());
}

/// OpenAI semantics: the matched stop sequence must never appear in
/// the returned content. Agent harnesses parse on stop markers; a
/// leaked marker is a phantom delimiter.
#[test]
fn matched_stop_sequence_is_stripped_from_content() {
    let stop = kiln_model::FinishReason::StopSequence("Observation:".to_string());
    assert_eq!(
        truncate_at_matched_stop("I should check the file.\nObservation: the file", &stop),
        "I should check the file.\n"
    );
    // Stop text absent from the buffer (already trimmed upstream): no-op.
    assert_eq!(truncate_at_matched_stop("clean text", &stop), "clean text");
    // Other finish reasons: untouched.
    assert_eq!(
        truncate_at_matched_stop("text Observation: x", &kiln_model::FinishReason::Eos),
        "text Observation: x"
    );
}

/// Long pi sessions must hit the OpenAI-style 400 (the signal agent
/// harnesses key auto-compaction on) instead of an opaque 500 from a
/// downstream BlockManager OOM — and a fitting prompt with an
/// oversized max_tokens clamps instead of erroring.
#[test]
fn context_window_rejects_oversized_prompts_and_clamps_max_tokens() {
    let mut sampling = SamplingParams {
        max_tokens: 4096,
        ..SamplingParams::default()
    };

    // No ceiling signal (mock backend): no-op.
    enforce_context_window_with_ceiling(None, &mut sampling, 1_000_000).unwrap();
    assert_eq!(sampling.max_tokens, 4096);

    // Prompt alone over the ceiling → the 400 with the exact code.
    let err = enforce_context_window_with_ceiling(Some(8192), &mut sampling, 9000).unwrap_err();
    assert_eq!(err.code, "context_length_exceeded");
    assert!(err.message.contains("8192"), "{}", err.message);
    assert!(err.message.contains("9000"), "{}", err.message);

    // Prompt fits, sum overflows → clamp to the remaining window.
    enforce_context_window_with_ceiling(Some(8192), &mut sampling, 6000).unwrap();
    assert_eq!(sampling.max_tokens, 8192 - 6000);

    // Already-fitting requests untouched.
    let mut small = SamplingParams {
        max_tokens: 10,
        ..SamplingParams::default()
    };
    enforce_context_window_with_ceiling(Some(8192), &mut small, 6000).unwrap();
    assert_eq!(small.max_tokens, 10);
}

fn make_qwen_template_test_state() -> AppState {
    let config = ModelConfig::qwen3_5_4b();
    let sched_config = kiln_scheduler::SchedulerConfig {
        max_batch_tokens: 8192,
        max_batch_size: 64,
        block_size: 16,
        prefix_cache_enabled: false,
        ..Default::default()
    };
    let scheduler = kiln_scheduler::Scheduler::new(sched_config, 256);
    let engine = kiln_model::engine::MockEngine::new(config.clone());
    let template =
        include_str!("../../../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
    let tokenizer = crate::api::test_tokenizer().with_chat_template(template.to_string());
    AppState::new_mock(
        config,
        scheduler,
        std::sync::Arc::new(engine),
        tokenizer,
        300,
        "kiln-test".to_string(),
    )
}

#[test]
fn completion_usage_counts_terminal_eos_token() {
    assert_eq!(
        completion_usage_tokens(0, &kiln_model::FinishReason::Eos),
        1
    );
    assert_eq!(
        completion_usage_tokens(3, &kiln_model::FinishReason::Eos),
        4
    );
    assert_eq!(
        completion_usage_tokens(3, &kiln_model::FinishReason::MaxTokens),
        3
    );
    assert_eq!(
        completion_usage_tokens(
            3,
            &kiln_model::FinishReason::StopSequence("stop".to_string())
        ),
        3
    );
}

#[test]
fn content_accepts_plain_string() {
    let req = parse_request(r#"{"messages":[{"role":"user","content":"hello"}]}"#);
    assert_eq!(req.messages[0].content, "hello");
}

#[test]
fn chat_adapter_missing_uses_server_default() {
    let req = parse_request(r#"{"messages":[{"role":"user","content":"hello"}]}"#);
    assert_eq!(req.adapter, ChatAdapterSelection::Default);
    assert_eq!(
        req.adapter
            .target_adapter_name(Some("loaded-a".to_string())),
        Some("loaded-a".to_string())
    );
    assert!(!req.adapter.is_explicit());
}

#[tokio::test]
async fn chat_adapter_missing_regression_does_not_unload_active_adapter_http_path() {
    let state = make_batch_test_state();
    *state.active_adapter_name.write().unwrap() = Some("loaded-a".to_string());
    *state.loaded_adapter.write().unwrap() = Some(LoadedAdapterIdentity {
        name: "loaded-a".to_string(),
        content_revision: "a".repeat(64),
    });
    let state_for_assert = state.clone();

    let (status, body) = chat_post(
        state,
        r#"{"messages":[{"role":"user","content":"adapter default"}],"max_tokens":0}"#,
    )
    .await;

    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        state_for_assert
            .active_adapter_name
            .read()
            .unwrap()
            .as_deref(),
        Some("loaded-a"),
        "regression: omitted chat `adapter` must not unload server default"
    );
    assert_eq!(
        state_for_assert.loaded_adapter_name().as_deref(),
        Some("loaded-a"),
        "regression: omitted chat `adapter` must not unload runtime adapter"
    );
}

#[test]
fn chat_adapter_null_selects_base_for_request() {
    let req = parse_request(r#"{"messages":[{"role":"user","content":"hello"}],"adapter":null}"#);
    assert_eq!(req.adapter, ChatAdapterSelection::Base);
    assert_eq!(
        req.adapter
            .target_adapter_name(Some("loaded-a".to_string())),
        None
    );
    assert!(req.adapter.is_explicit());
}

#[test]
fn chat_adapter_empty_selects_base_for_request() {
    let req = parse_request(r#"{"messages":[{"role":"user","content":"hello"}],"adapter":""}"#);
    assert_eq!(req.adapter, ChatAdapterSelection::Base);
    assert_eq!(
        req.adapter
            .target_adapter_name(Some("loaded-a".to_string())),
        None
    );
    assert!(req.adapter.is_explicit());
}

#[test]
fn chat_adapter_name_selects_named_adapter_for_request() {
    let req =
        parse_request(r#"{"messages":[{"role":"user","content":"hello"}],"adapter":"my-adapter"}"#);
    assert_eq!(
        req.adapter,
        ChatAdapterSelection::Named("my-adapter".to_string())
    );
    assert_eq!(
        req.adapter
            .target_adapter_name(Some("loaded-a".to_string())),
        Some("my-adapter".to_string())
    );
    assert!(req.adapter.is_explicit());
}

#[test]
fn chat_adapter_rejects_invalid_value_type() {
    let err = serde_json::from_str::<ChatCompletionRequest>(
        r#"{"messages":[{"role":"user","content":"hello"}],"adapter":7}"#,
    )
    .expect_err("numeric adapter should not deserialize");
    assert!(err.to_string().contains("invalid type"));
}

#[test]
fn chat_adapter_rejects_invalid_name_shape() {
    assert!(validate_compose_name("../escape").is_err());
    assert!(validate_compose_name("/tmp/adapter").is_err());
    assert!(validate_compose_name("valid-name").is_ok());
}

#[test]
fn content_accepts_text_parts_array() {
    let req = parse_request(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}]}"#,
    );
    assert_eq!(req.messages[0].content, "hello world");
}

#[test]
fn content_ignores_non_text_parts() {
    let req = parse_request(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"describe: "},{"type":"image_url","image_url":{"url":"https://example.com/a.png"}},{"type":"text","text":"done"}]}]}"#,
    );
    assert_eq!(req.messages[0].content, "describe: done");
}

#[test]
fn content_empty_array_is_empty_string() {
    let req = parse_request(r#"{"messages":[{"role":"user","content":[]}]}"#);
    assert_eq!(req.messages[0].content, "");
}

#[test]
fn content_mixed_messages_in_same_request() {
    let req = parse_request(
        r#"{"messages":[{"role":"system","content":"be nice"},{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
    );
    assert_eq!(req.messages[0].content, "be nice");
    assert_eq!(req.messages[1].content, "hi");
}

#[test]
fn stop_accepts_string_or_array() {
    let chat_string =
        parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"stop":"END"}"#);
    assert_eq!(chat_string.stop.as_deref(), Some(&["END".to_string()][..]));

    let chat_array =
        parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"stop":["END","DONE"]}"#);
    assert_eq!(
        chat_array.stop.as_deref(),
        Some(&["END".to_string(), "DONE".to_string()][..])
    );

    let batch_string =
        parse_batch_request(r#"{"prompts":[[{"role":"user","content":"hi"}]],"stop":"END"}"#);
    assert_eq!(batch_string.stop.as_deref(), Some(&["END".to_string()][..]));
}

#[test]
fn max_completion_tokens_alias_resolves_like_max_tokens() {
    let alias =
        parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":7}"#);
    assert_eq!(alias.max_tokens, None);
    assert_eq!(alias.max_completion_tokens, Some(7));
    assert_eq!(chat_request_max_tokens(&alias), 7);

    let both = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":3,"max_completion_tokens":7}"#,
    );
    assert_eq!(chat_request_max_tokens(&both), 3);

    let batch_alias = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"hi"}]],"max_completion_tokens":7}"#,
    );
    assert_eq!(batch_alias.max_tokens, None);
    assert_eq!(batch_alias.max_completion_tokens, Some(7));
    assert_eq!(batch_request_max_tokens(&batch_alias), 7);
}

#[test]
fn tools_round_trip_preserved_on_request() {
    let json = r#"{
            "messages":[{"role":"user","content":"run ls"}],
            "tools":[
                {"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object"}}},
                {"type":"function","function":{"name":"Read","description":"Read a file","parameters":{"type":"object"}}}
            ],
            "tool_choice":"auto"
        }"#;
    let req = parse_request(json);
    let tools = req.tools.expect("tools should deserialize");
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["function"]["name"], "Bash");
    assert_eq!(tools[1]["function"]["name"], "Read");
    assert_eq!(
        req.tool_choice.as_ref().and_then(|v| v.as_str()),
        Some("auto")
    );
}

/// CI hardening for kiln#659: pins the FULL chain
/// JSON → `ChatCompletionRequest` → `message_to_chat` → `apply_chat_template_full`
/// against the production bundled Qwen3.5-4B chat template.
///
/// Existing per-layer tests cover only one half each:
///  - `tools_round_trip_preserved_on_request` (above) pins JSON deserialization.
///  - `kiln_core::tokenizer::test_qwen35_4b_chat_template_renders_tools_and_tool_calls`
///    pins rendering with hand-built `ChatMessage` values.
///
/// Neither exercises the seam where production bugs 1 (missing `tojson` filter)
/// and 3 (`arguments` left as JSON-encoded string instead of dict) actually lived
/// in PR #632 and shipped to main. A regression in `message_to_chat` mapping or
/// in how `req.tools.as_deref()` flows through `apply_chat_template_full` would
/// not surface in either of those tests, but it would break this one.
#[test]
fn tools_bearing_chat_completion_renders_via_qwen35_4b_template() {
    // Wire-shape JSON exactly as `/v1/chat/completions` receives. Five
    // turns (system → user → assistant-with-tool_calls → tool → user)
    // exercise the multi-step-tool branch in the Qwen3.5 template, and
    // both tools have non-trivial `parameters.properties` so `tojson`,
    // `|items`, and `|length` filters all run against real data.
    let json = r#"{
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Show me what's in /etc."},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_42", "type": "function", "function": {
                        "name": "Bash",
                        "arguments": "{\"command\": \"ls /etc\"}"
                    }}
                ]},
                {"role": "tool", "name": "Bash", "tool_call_id": "call_42",
                 "content": "hosts\nresolv.conf\nshadow"},
                {"role": "user", "content": "Now read /etc/hosts."}
            ],
            "tools": [
                {"type": "function", "function": {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to run"}
                        },
                        "required": ["command"]
                    }
                }},
                {"type": "function", "function": {
                    "name": "Read",
                    "description": "Read a file from disk",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Filesystem path"}
                        },
                        "required": ["path"]
                    }
                }}
            ],
            "tool_choice": "auto"
        }"#;

    // Step 1: deserialize wire payload (pins serde + content-shape parsing
    // including `content: null` on the assistant tool-calls turn).
    let req = parse_request(json);
    assert_eq!(
        req.messages.len(),
        5,
        "fixture must exercise multi-turn shape"
    );
    assert_eq!(
        req.tools.as_ref().map(|t| t.len()),
        Some(2),
        "tools array must round-trip through deserialization"
    );

    // Step 2: load the production bundled Qwen3.5-4B chat template — the
    // canonical template every kiln user actually hits at runtime. Path is
    // relative to this source file (crates/kiln-server/src/api/...).
    let template =
        include_str!("../../../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
    let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());

    // Step 3: wire EXACTLY as `chat_completions_inner` does (see the
    // `let chat_messages = ... map(message_to_chat) ...` /
    // `apply_chat_template_full(..., req.tools.as_deref(), req.tool_choice.as_ref())`
    // pair near the top of that function). Drift between this test and the
    // production wiring would defeat the point of the smoke test.
    let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();
    let prompt = tok
        .apply_chat_template_full(
            &chat_messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
        )
        .expect(
            "Qwen3.5-4B chat template must render the wire-shape \
                 tools+tool_calls payload without error",
        );

    // Bug 1 (tojson filter on `tools` array): if minijinja lacks the
    // `json` feature, the render fails outright. The `<tools>` block plus
    // both function names appearing prove `tools | tojson` produced
    // valid JSON for both definitions.
    assert!(
        prompt.contains("<tools>"),
        "tools block missing — `tools | tojson` regression? prompt was {prompt:?}"
    );
    assert!(
        prompt.contains("\"Bash\""),
        "Bash tool not serialized into <tools> block: {prompt:?}"
    );
    assert!(
        prompt.contains("\"Read\""),
        "Read tool not serialized into <tools> block: {prompt:?}"
    );

    // Bug 3 (arguments as JSON-encoded string vs dict): the Qwen template
    // iterates `tool_call.arguments | items`. If kiln passes the wire
    // form (`"{\"command\":\"ls /etc\"}"`) through unchanged, minijinja
    // rejects it with "cannot convert value into pairs". The
    // `<parameter=command>` block proves arguments were promoted to a
    // dict before render.
    assert!(
        prompt.contains("<function=Bash>"),
        "prior assistant tool_call did not render in pi-XML form: {prompt:?}"
    );
    assert!(
        prompt.contains("<parameter=command>"),
        "tool_call arguments were not iterated as dict — \
             string-form regression? {prompt:?}"
    );
    assert!(
        prompt.contains("ls /etc"),
        "argument value missing from rendered tool_call: {prompt:?}"
    );

    // `role: "tool"` must wrap as `<tool_response>...</tool_response>` —
    // proves `message_to_chat` propagated `tool_call_id` / `name` so the
    // template's `{%- elif message.role == "tool" %}` branch fires.
    assert!(
        prompt.contains("<tool_response>"),
        "tool response role did not render through message_to_chat wiring: {prompt:?}"
    );
    assert!(
        prompt.contains("hosts"),
        "tool response content did not render inside <tool_response>: {prompt:?}"
    );

    // Follow-up user turn must survive past the template's
    // `multi_step_tool` / `last_query_index` scan; otherwise the template
    // raises "No user query found in messages." and the render errors.
    assert!(
        prompt.contains("Now read /etc/hosts."),
        "follow-up user turn missing — last_query_index regression? {prompt:?}"
    );
}

#[test]
fn qwen35_chat_template_kwargs_disable_thinking() {
    let json = r#"{
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "user", "content": "Answer with exactly one word."}
            ],
            "chat_template_kwargs": {"enable_thinking": false},
            "temperature": 0.0,
            "max_tokens": 12
        }"#;
    let req = parse_request(json);
    let template =
        include_str!("../../../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
    let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());
    let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();

    let prompt = tok
        .apply_chat_template_full_with_options(
            &chat_messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
            chat_template_options_from_kwargs(req.chat_template_kwargs.as_ref()),
        )
        .expect("Qwen3.5 prompt with chat_template_kwargs should render");

    assert!(
        prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "enable_thinking=false should pre-close the reasoning block: {prompt:?}"
    );
    assert!(
        !prompt_starts_in_reasoning(&prompt),
        "pre-closed reasoning prompt should not put generated text into reasoning_content"
    );
    let (reasoning_content, content) = split_reasoning_response("Blue", &prompt);
    assert_eq!(reasoning_content, None);
    assert_eq!(content, "Blue");
}

#[test]
fn qwen35_tool_requests_keep_template_default_thinking() {
    let json = r#"{
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "user", "content": "List files."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            }],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 12
        }"#;
    let req = parse_request(json);
    let template =
        include_str!("../../../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
    let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());
    let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();
    let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
    let normalized_tool_choice =
        normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());

    let prompt = tok
        .apply_chat_template_full_with_options(
            &chat_messages,
            normalized_tools,
            normalized_tool_choice,
            chat_template_options_from_kwargs(normalized_chat_template_kwargs),
        )
        .expect("Qwen3.5 tools prompt should render");

    assert!(
        prompt.ends_with("<|im_start|>assistant\n<think>\n"),
        "tool requests should use Qwen's template default thinking mode: {prompt:?}"
    );
    assert!(
        prompt_starts_in_reasoning(&prompt),
        "default tool requests should still split generated text as reasoning_content"
    );
}

#[test]
fn qwen35_tool_requests_honor_explicit_disable_thinking() {
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"List files."}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "chat_template_kwargs":{"enable_thinking":false},
                "temperature":0.0,
                "max_tokens":12
            }"#,
    );
    let template =
        include_str!("../../../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
    let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());
    let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();
    let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
    let normalized_tool_choice =
        normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());

    let prompt = tok
        .apply_chat_template_full_with_options(
            &chat_messages,
            normalized_tools,
            normalized_tool_choice,
            chat_template_options_from_kwargs(normalized_chat_template_kwargs),
        )
        .expect("Qwen3.5 explicit non-thinking tools prompt should render");

    assert!(
        prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "explicit enable_thinking=false should pre-close thinking: {prompt:?}"
    );
    assert!(
        !prompt_starts_in_reasoning(&prompt),
        "explicit enable_thinking=false should stream generated text as content"
    );
}

#[test]
fn qwen35_no_think_text_is_not_a_control_flag() {
    let json = r#"{
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "user", "content": "Explain what the literal /no_think tag means."}
            ],
            "temperature": 0.0,
            "max_tokens": 12
        }"#;
    let req = parse_request(json);
    let template =
        include_str!("../../../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
    let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());
    let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();

    let prompt = tok
        .apply_chat_template_full_with_options(
            &chat_messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
            chat_template_options_from_kwargs(req.chat_template_kwargs.as_ref()),
        )
        .expect("Qwen3.5 prompt should render");

    assert!(
        prompt.ends_with("<|im_start|>assistant\n<think>\n"),
        "plain prompt text must not disable template thinking: {prompt:?}"
    );
    assert!(
        prompt_starts_in_reasoning(&prompt),
        "open reasoning prompt should still split generated text as reasoning_content"
    );
}

#[test]
fn qwen35_server_default_thinking_can_be_disabled_and_overridden() {
    let mut state = make_qwen_template_test_state();
    state.default_thinking_enabled = Some(false);
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"List files."}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"auto",
                "temperature":0.0,
                "max_tokens":12
            }"#,
    );

    let prompt = render_prompt_text(
        &state,
        &req.messages,
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        req.chat_template_kwargs.as_ref(),
    )
    .expect("Qwen3.5 prompt with server default should render");
    assert!(
        prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "server default thinking=false should pre-close thinking: {prompt:?}"
    );
    let metadata = chat_completion_metadata_from_prompt(&state, &req, &prompt);
    assert!(!metadata.thinking_enabled);
    assert_eq!(metadata.thinking_mode, "non_reasoning");
    assert_eq!(metadata.thinking_source, "server_default");

    let override_req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"List files."}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"auto",
                "chat_template_kwargs":{"enable_thinking":true},
                "temperature":0.0,
                "max_tokens":12
            }"#,
    );
    let override_prompt = render_prompt_text(
        &state,
        &override_req.messages,
        override_req.tools.as_deref(),
        override_req.tool_choice.as_ref(),
        override_req.chat_template_kwargs.as_ref(),
    )
    .expect("Qwen3.5 prompt with request override should render");
    assert!(
        override_prompt.ends_with("<|im_start|>assistant\n<think>\n"),
        "request enable_thinking=true should override server default: {override_prompt:?}"
    );
    let override_metadata =
        chat_completion_metadata_from_prompt(&state, &override_req, &override_prompt);
    assert!(override_metadata.thinking_enabled);
    assert_eq!(override_metadata.thinking_mode, "reasoning");
    assert_eq!(override_metadata.thinking_source, "request");
}

#[test]
fn qwen35_thinking_off_short_prompt_regression_splits_normal_content() {
    let mut state = make_qwen_template_test_state();
    state.default_thinking_enabled = Some(false);
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"Answer with exactly two words."}],
                "temperature":0.0,
                "max_tokens":8
            }"#,
    );

    let prompt = render_prompt_text(
        &state,
        &req.messages,
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        req.chat_template_kwargs.as_ref(),
    )
    .expect("Qwen3.5 non-thinking short prompt should render");
    assert!(
        !prompt_starts_in_reasoning(&prompt),
        "regression: thinking-off prompt must pre-close the reasoning block"
    );
    let (reasoning_content, content) = split_reasoning_response("Quality control", &prompt);
    assert_eq!(
        reasoning_content, None,
        "regression: thinking-off Qwen output should not be hidden in reasoning_content"
    );
    assert_eq!(
        content, "Quality control",
        "regression: thinking-off short prompt should produce normal content"
    );
}

#[test]
fn reasoning_only_output_serializes_reasoning_channel_and_empty_reason_metadata() {
    let state = make_qwen_template_test_state();
    let req = parse_request(
        r#"{"messages":[{"role":"user","content":"think only"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let prompt_text = "<|im_start|>assistant\n<think>\n";
    let assistant_output = assistant_output_from_split_parts(
        &req,
        Some("Need one more step.".to_string()),
        String::new(),
        "length",
    );
    let metadata = chat_completion_metadata_from_prompt_and_output(
        &state,
        &req,
        prompt_text,
        &assistant_output,
    );
    let resp = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion",
        created: 0,
        model: "kiln-test".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: assistant_output.content,
                reasoning_content: assistant_output.reasoning_content,
                tool_calls: assistant_output.tool_calls,
                name: None,
                tool_call_id: None,
            },
            finish_reason: assistant_output.finish_reason,
            thinking_budget: None,
            rollout_provenance: None,
            completion_tokens: 4,
        }],
        usage: Usage {
            prompt_tokens: 3,
            completion_tokens: 4,
            total_tokens: 7,
        },
        metadata,
    };

    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "");
    assert_eq!(
        json["choices"][0]["message"]["reasoning_content"],
        "Need one more step."
    );
    assert_eq!(json["metadata"]["final_content_empty"], true);
    assert_eq!(
        json["metadata"]["content_empty_reason"],
        "reasoning_without_final_content"
    );
    assert_eq!(json["metadata"]["reasoning_folded_into_content"], false);
}

#[test]
fn reasoning_folding_duplicates_reasoning_into_content_without_hiding_channel() {
    let state = make_qwen_template_test_state();
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"think only"}],
                "fold_reasoning_into_content":true,
                "temperature":0.0,
                "max_tokens":4
            }"#,
    );
    let prompt_text = "<|im_start|>assistant\n<think>\n";
    let assistant_output = assistant_output_from_split_parts(
        &req,
        Some("Need one more step.".to_string()),
        String::new(),
        "length",
    );
    let metadata = chat_completion_metadata_from_prompt_and_output(
        &state,
        &req,
        prompt_text,
        &assistant_output,
    );
    let assistant_output = apply_reasoning_content_policy(
        assistant_output,
        fold_reasoning_into_content_for_request(&state, &req),
    );

    assert_eq!(
        assistant_output.content,
        "<think>\nNeed one more step.</think>"
    );
    assert_eq!(
        assistant_output.reasoning_content.as_deref(),
        Some("Need one more step.")
    );
    assert!(metadata.final_content_empty);
    assert_eq!(
        metadata.content_empty_reason,
        Some("reasoning_without_final_content")
    );
    assert!(metadata.reasoning_folded_into_content);
}

#[test]
fn request_reasoning_fold_override_wins_over_server_default() {
    let mut state = make_qwen_template_test_state();
    state.fold_reasoning_into_content = true;
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"think only"}],
                "fold_reasoning_into_content":false,
                "temperature":0.0,
                "max_tokens":4
            }"#,
    );
    let assistant_output = assistant_output_from_split_parts(
        &req,
        Some("Private scratchpad.".to_string()),
        String::new(),
        "length",
    );
    let metadata = chat_completion_metadata_from_prompt_and_output(
        &state,
        &req,
        "<|im_start|>assistant\n<think>\n",
        &assistant_output,
    );
    let assistant_output = apply_reasoning_content_policy(
        assistant_output,
        fold_reasoning_into_content_for_request(&state, &req),
    );

    assert_eq!(assistant_output.content, "");
    assert_eq!(
        assistant_output.reasoning_content.as_deref(),
        Some("Private scratchpad.")
    );
    assert!(!metadata.reasoning_folded_into_content);
}

#[test]
fn folded_reasoning_is_unfolded_before_cache_storage() {
    let content = folded_reasoning_content("Scratchpad", "Final answer");
    assert_eq!(
        response_content_for_cache(&content, Some("Scratchpad")),
        "Final answer"
    );

    let response = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion",
        created: 0,
        model: "kiln-test".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content,
                reasoning_content: Some("Scratchpad".to_string()),
                tool_calls: None,
                name: None,
                tool_call_id: None,
            },
            finish_reason: "stop".to_string(),
            thinking_budget: None,
            rollout_provenance: None,
            completion_tokens: 3,
        }],
        usage: Usage {
            prompt_tokens: 2,
            completion_tokens: 3,
            total_tokens: 5,
        },
        metadata: ChatCompletionMetadata {
            thinking_enabled: true,
            thinking_mode: "reasoning".to_string(),
            thinking_source: "template_default",
            default_thinking_enabled: None,
            final_content_empty: false,
            content_empty_reason: None,
            reasoning_folded_into_content: true,
            thinking_budget: ThinkingBudgetMetadata::default(),
            config_hashes: None,
            performance: None,
        },
    };

    let cached = cache_value_from_response(&response).unwrap();
    assert_eq!(cached.text, "Final answer");
    assert_eq!(cached.reasoning_content.as_deref(), Some("Scratchpad"));
}

#[test]
fn message_tool_calls_round_trip_preserved() {
    let json = r#"{
            "messages":[
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_42","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"ls\"}"}}
                ]},
                {"role":"tool","name":"Bash","tool_call_id":"call_42","content":"file.txt"}
            ]
        }"#;
    let req = parse_request(json);
    // Assistant message with content:null lands as empty string + tool_calls populated.
    assert_eq!(req.messages[0].role, "assistant");
    assert_eq!(req.messages[0].content, "");
    let calls = req.messages[0]
        .tool_calls
        .as_ref()
        .expect("tool_calls present");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "call_42");
    assert_eq!(calls[0]["function"]["name"], "Bash");
    // Tool response with id + name.
    assert_eq!(req.messages[1].role, "tool");
    assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("call_42"));
    assert_eq!(req.messages[1].name.as_deref(), Some("Bash"));
    assert_eq!(req.messages[1].content, "file.txt");
}

#[test]
fn tools_absent_request_keeps_options_none() {
    let req = parse_request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    assert!(
        req.tools.is_none(),
        "tools should default to None when absent"
    );
    assert!(
        req.tool_choice.is_none(),
        "tool_choice should default to None"
    );
    assert!(req.messages[0].tool_calls.is_none());
    assert!(req.messages[0].name.is_none());
    assert!(req.messages[0].tool_call_id.is_none());
}

#[test]
fn message_to_chat_propagates_tool_fields() {
    let m = Message {
        role: "tool".to_string(),
        content: "ok".to_string(),
        reasoning_content: None,
        tool_calls: None,
        name: Some("Bash".to_string()),
        tool_call_id: Some("call_1".to_string()),
    };
    let chat = message_to_chat(&m);
    assert_eq!(chat.role, "tool");
    assert_eq!(chat.content, "ok");
    assert_eq!(chat.name.as_deref(), Some("Bash"));
    assert_eq!(chat.tool_call_id.as_deref(), Some("call_1"));
    assert!(chat.tool_calls.is_none());
}

#[test]
fn message_to_chat_omits_empty_tool_calls() {
    let empty = Message {
        role: "assistant".to_string(),
        content: "ok".to_string(),
        reasoning_content: None,
        tool_calls: Some(Vec::new()),
        name: None,
        tool_call_id: None,
    };
    assert!(
        message_to_chat(&empty).tool_calls.is_none(),
        "empty message tool_calls should render like omitted tool_calls"
    );

    let non_empty = Message {
        role: "assistant".to_string(),
        content: "ok".to_string(),
        reasoning_content: None,
        tool_calls: Some(vec![serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "Lookup", "arguments": "{}"}
        })]),
        name: None,
        tool_call_id: None,
    };
    assert_eq!(
        message_to_chat(&non_empty).tool_calls.unwrap().len(),
        1,
        "non-empty message tool_calls must still reach the template"
    );
}

#[test]
fn qwen_xml_tool_call_output_becomes_openai_tool_calls() {
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"list files"}],
                "tools":[{
                    "type":"function",
                    "function":{
                        "name":"bash",
                        "parameters":{
                            "type":"object",
                            "properties":{"command":{"type":"string"}},
                            "required":["command"]
                        }
                    }
                }],
                "tool_choice":"auto"
            }"#,
    );
    let raw = "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>";
    let output = assistant_output_from_split_parts(&req, None, raw.to_string(), "stop");

    assert_eq!(output.finish_reason, "tool_calls");
    assert_eq!(output.content, "");
    let calls = output.tool_calls.expect("tool_calls should be populated");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(calls[0]["function"]["name"], "bash");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["command"], "ls -la");

    let deltas = tool_call_deltas_from_openai_calls(&calls);
    assert_eq!(deltas[0]["index"], 0);
    assert_eq!(deltas[0]["function"]["name"], "bash");
}

#[test]
fn qwen_xml_tool_call_output_respects_tool_choice_none() {
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"say hi"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"none"
            }"#,
    );
    let raw = "<tool_call>\n<function=bash>\n</function>\n</tool_call>";
    let output = assistant_output_from_split_parts(&req, None, raw.to_string(), "stop");

    assert_eq!(output.finish_reason, "stop");
    assert!(output.tool_calls.is_none());
    assert!(output.content.contains("<tool_call>"));
}

#[test]
fn tool_requests_stop_generation_at_qwen_tool_call_close() {
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"list files"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "stop":["omega","alpha","omega"]
            }"#,
    );
    let sampling = sampling_params_for_chat_request(&req);

    assert_eq!(
        sampling.stop,
        vec![
            "alpha".to_string(),
            "omega".to_string(),
            QWEN_TOOL_CALL_CLOSE_TAG.to_string(),
        ],
        "tools-capable requests should stop as soon as a complete Qwen XML tool call is generated"
    );
}

#[test]
fn qwen_tool_stop_is_not_added_when_tools_are_disabled() {
    let tool_choice_none = parse_request(
        r#"{
                "messages":[{"role":"user","content":"say hi"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"none"
            }"#,
    );
    assert!(
        !sampling_params_for_chat_request(&tool_choice_none)
            .stop
            .contains(&QWEN_TOOL_CALL_CLOSE_TAG.to_string()),
        "tool_choice=none should keep model output text-shaped"
    );

    let explicit_shorter_stop = parse_request(
        r#"{
                "messages":[{"role":"user","content":"say hi"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "stop":"tool_call"
            }"#,
    );
    assert_eq!(
        sampling_params_for_chat_request(&explicit_shorter_stop).stop,
        vec!["tool_call".to_string()],
        "user-supplied stops that would fire inside the close tag should remain authoritative"
    );
}

#[test]
fn qwen_xml_tool_call_after_reasoning_is_not_leaked_as_content() {
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"inspect the repo"}],
                "tools":[{
                    "type":"function",
                    "function":{
                        "name":"bash",
                        "parameters":{
                            "type":"object",
                            "properties":{"command":{"type":"string"}},
                            "required":["command"]
                        }
                    }
                }]
            }"#,
    );
    let prompt_text = "<|im_start|>assistant\n<think>\n";
    let raw = "The user wants a directory listing.</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>";
    let output = assistant_output_from_model_output(&req, raw, prompt_text, "stop");

    assert_eq!(output.finish_reason, "tool_calls");
    assert_eq!(output.content, "");
    assert_eq!(
        output.reasoning_content.as_deref(),
        Some("The user wants a directory listing.")
    );
    assert!(output.tool_calls.is_some());
}

#[test]
fn qwen_xml_tool_call_inside_unclosed_reasoning_is_not_cached_as_text() {
    let req = parse_request(
        r#"{
                "messages":[{"role":"user","content":"inspect the repo"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}]
            }"#,
    );
    let prompt_text = "<|im_start|>assistant\n<think>\n";
    let raw = "Need a command.\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
    let output = assistant_output_from_model_output(&req, raw, prompt_text, "stop");

    assert_eq!(output.finish_reason, "tool_calls");
    assert_eq!(output.content, "");
    assert_eq!(output.reasoning_content.as_deref(), Some("Need a command."));
    assert!(output.tool_calls.is_some());
}

#[tokio::test]
async fn buffered_tool_stream_detects_closed_client_without_sending() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    drop(rx);
    let mut completion_preview_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut content_buf = String::new();

    let ok = emit_or_buffer_reasoning_chunk(
        &tx,
        "chatcmpl-test",
        0,
        "kiln-test",
        ReasoningChunk {
            reasoning: None,
            content: Some("<tool_call>".to_string()),
        },
        &mut completion_preview_buf,
        &mut reasoning_buf,
        &mut content_buf,
        &mut ToolCallGate::new(true),
    )
    .await;

    assert!(
        !ok,
        "buffered tool-call content must still notice dropped SSE clients"
    );
    assert!(content_buf.is_empty());
}

/// Render queued SSE [`Event`]s to wire text the way the streaming
/// handlers' `Sse` response would. The sender must be dropped first
/// so the stream terminates.
async fn sse_body_from_events(rx: tokio::sync::mpsc::Receiver<Event>) -> String {
    use axum::body::to_bytes;

    let stream = ReceiverStream::new(rx).map(Ok::<_, std::convert::Infallible>);
    let resp = Sse::new(stream).into_response();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn sse_data_payloads(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).expect("SSE data line should be JSON"))
        .collect()
}

#[tokio::test]
async fn streaming_generation_errors_survive_a_saturated_event_queue_before_done() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tx.send(Event::default().data("queued-before-error"))
        .await
        .unwrap();
    let terminal = StreamTerminal::default();
    terminal.fail("injected prefill failure");
    drop(tx);

    use axum::body::to_bytes;
    let resp = Sse::new(stream_with_terminal(rx, terminal)).into_response();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    let payloads: Vec<_> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(payloads.len(), 3, "unexpected SSE body: {body}");
    assert_eq!(payloads[0], "queued-before-error");
    let error: serde_json::Value = serde_json::from_str(payloads[1]).unwrap();
    assert_eq!(error["error"]["message"], "injected prefill failure");
    assert_eq!(error["error"]["type"], "server_error");
    assert_eq!(error["error"]["code"], "generation_error");
    assert_eq!(payloads[2], "[DONE]");
}

#[tokio::test]
async fn streaming_task_unwind_signals_cancel_and_clears_prefill_progress() {
    let gauge = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let cancel = CancelHandle::with_prefill_progress_gauge(gauge.clone());
    cancel.report_prefill_tokens_completed(64);
    let cancel_for_task = cancel.clone();

    let task = tokio::spawn(async move {
        let _guard = PrefillProgressGuard::new(cancel_for_task);
        panic!("injected streaming producer panic");
    });
    let error = task.await.expect_err("producer task should panic");
    assert!(error.is_panic());
    assert!(cancel.is_cancelled());
    assert_eq!(
        gauge.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "task unwind must clear its in-flight prefill contribution"
    );
    cancel.report_prefill_tokens_completed(128);
    assert_eq!(cancel.prefill_tokens_completed(), 0);
    assert_eq!(
        gauge.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "a detached prefill must not resurrect progress after cancellation"
    );
}

#[tokio::test]
async fn streaming_success_terminals_survive_a_saturated_event_queue() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tx.send(Event::default().data("queued-before-complete"))
        .await
        .unwrap();

    let (terminal_tx, terminal_rx) = tokio::sync::mpsc::channel(STREAM_TERMINAL_EVENT_CAPACITY);
    terminal_tx
        .send(Event::default().data("finish"))
        .await
        .unwrap();
    terminal_tx
        .send(Event::default().data("usage"))
        .await
        .unwrap();
    terminal_tx
        .send(Event::default().data("[DONE]"))
        .await
        .unwrap();
    drop(terminal_tx);

    let terminal = StreamTerminal::default();
    terminal.complete(drain_terminal_event_buffer(terminal_rx));
    drop(tx);

    use axum::body::to_bytes;
    let response = Sse::new(stream_with_terminal(rx, terminal)).into_response();
    let bytes = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("saturated success stream stalled")
    .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    let payloads: Vec<_> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(
        payloads,
        ["queued-before-complete", "finish", "usage", "[DONE]"]
    );
}

#[tokio::test]
async fn streaming_producer_drop_fails_closed_unless_completed() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let terminal = StreamTerminal::default();
    drop(tx);

    use axum::body::to_bytes;
    let resp = Sse::new(stream_with_terminal(rx, terminal)).into_response();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    let payloads: Vec<_> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(payloads.len(), 2, "unexpected SSE body: {body}");
    let error: serde_json::Value = serde_json::from_str(payloads[0]).unwrap();
    assert_eq!(
        error["error"]["message"],
        "streaming response producer ended without a terminal state"
    );
    assert_eq!(payloads[1], "[DONE]");

    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let terminal = StreamTerminal::default();
    terminal.complete(std::collections::VecDeque::new());
    drop(tx);
    let resp = Sse::new(stream_with_terminal(rx, terminal)).into_response();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    assert!(
        bytes.is_empty(),
        "completed producers add no fallback events"
    );
}

#[tokio::test]
async fn streaming_token_timing_is_explicit_opt_in_and_not_a_chat_chunk() {
    let omitted = parse_request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    let disabled = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],"include_performance":false}"#,
    );
    let enabled = parse_request(
        r#"{"messages":[{"role":"user","content":"hi"}],"include_performance":true}"#,
    );
    assert!(!streaming_token_timing_enabled(&omitted));
    assert!(!streaming_token_timing_enabled(&disabled));
    assert!(streaming_token_timing_enabled(&enabled));

    let request_start = std::time::Instant::now();
    let ready_at = request_start + std::time::Duration::from_millis(12);
    let mut engine_timing = EngineTokenTiming::ready(ready_at, TokenPhaseDurations::default());
    engine_timing.mark_producer_delivered(ready_at + std::time::Duration::from_millis(2));
    let handler_received_at = ready_at + std::time::Duration::from_millis(5);
    let body_enqueued_at = handler_received_at + std::time::Duration::from_millis(3);
    assert!(
        streaming_token_timing_json(
            false,
            1,
            42,
            request_start,
            engine_timing,
            handler_received_at,
            body_enqueued_at,
            None,
        )
        .is_none()
    );
    let timing = streaming_token_timing_json(
        true,
        7,
        4242,
        request_start,
        engine_timing,
        handler_received_at,
        body_enqueued_at,
        None,
    )
    .unwrap();

    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tx.send(Event::default().data(timing)).await.unwrap();
    drop(tx);
    let payloads = sse_data_payloads(&sse_body_from_events(rx).await);
    assert_eq!(payloads.len(), 1);
    let payload = payloads.into_iter().next().unwrap();
    let object = payload.as_object().unwrap();
    assert_eq!(object.len(), 14, "timing payload shape changed: {payload}");
    assert_eq!(payload["object"], "kiln.token_timing");
    assert_eq!(payload["source"], "batching_engine");
    assert_eq!(payload["token_index"], 7);
    assert_eq!(payload["token_id"], 4242);
    assert_eq!(payload["ready_ms"], 12.0);
    assert_eq!(payload["producer_delivered_ms"], 14.0);
    assert_eq!(payload["handler_received_ms"], 17.0);
    assert_eq!(payload["body_enqueued_ms"], 20.0);
    assert_eq!(payload["response_delivery_ms"], 2.0);
    assert_eq!(payload["handler_queue_ms"], 3.0);
    assert_eq!(payload["queue_delay_ms"], 5.0);
    assert_eq!(payload["client_delivery_ms"], 3.0);
    assert!(payload["blocking_phase"].is_null());
    assert!(payload.get("choices").is_none());
}

#[tokio::test]
async fn flush_buffered_stream_tail_salvages_complete_tool_call() {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let mut splitter = ReasoningSplitter::new(false);
    let mut completion_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut content_buf = "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>".to_string();

    let tail = flush_buffered_stream_tail(
        &tx,
        "chatcmpl-test",
        0,
        "kiln-test",
        &mut splitter,
        &mut completion_buf,
        &mut reasoning_buf,
        &mut content_buf,
        &mut ToolCallGate::new(true),
        "timeout",
    )
    .await
    .expect("a complete buffered tool call should be salvaged");
    drop(tx);

    assert_eq!(
        tail.finish_reason, "tool_calls",
        "a complete salvaged call reports tool_calls instead of timeout"
    );
    let calls = tail.tool_calls.as_deref().expect("tool calls parsed");
    assert_eq!(calls[0]["function"]["name"], "bash");

    let payloads = sse_data_payloads(&sse_body_from_events(rx).await);
    assert_eq!(
        payloads.len(),
        1,
        "a complete call with no preamble emits exactly one tool_calls chunk"
    );
    let choice = &payloads[0]["choices"][0];
    assert_eq!(choice["delta"]["tool_calls"][0]["function"]["name"], "bash");
    assert!(
        choice["finish_reason"].is_null(),
        "the caller, not the flush helper, emits the finish chunk"
    );
}

#[tokio::test]
async fn flush_buffered_stream_tail_emits_partial_content_on_timeout() {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let mut splitter = ReasoningSplitter::new(false);
    let mut completion_buf = String::new();
    let mut reasoning_buf = String::new();
    let partial = "Checking files now\n<tool_call>\n<function=bash>\n<parameter=command>\nls";
    let mut content_buf = partial.to_string();

    let tail = flush_buffered_stream_tail(
        &tx,
        "chatcmpl-test",
        0,
        "kiln-test",
        &mut splitter,
        &mut completion_buf,
        &mut reasoning_buf,
        &mut content_buf,
        &mut ToolCallGate::new(true),
        "timeout",
    )
    .await
    .expect("partial buffered content should be salvaged");
    drop(tx);

    assert_eq!(
        tail.finish_reason, "timeout",
        "without a complete tool call the finish reason stays timeout"
    );
    assert!(tail.tool_calls.is_none());
    assert_eq!(tail.preview_source(), partial);

    let payloads = sse_data_payloads(&sse_body_from_events(rx).await);
    assert_eq!(payloads.len(), 1);
    let choice = &payloads[0]["choices"][0];
    assert_eq!(
        choice["delta"]["content"], partial,
        "buffered partial text must reach the client before the finish chunk"
    );
    assert!(choice["finish_reason"].is_null());
}

#[tokio::test]
async fn flush_buffered_stream_tail_drains_reasoning_splitter_tail() {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let mut splitter = ReasoningSplitter::new(true);
    // A token ending in a partial `</think>` prefix leaves bytes
    // pending inside the splitter; the immediate part would already
    // have been emitted by the stream loop.
    let pushed = splitter.push("thinking</thi");
    assert_eq!(pushed.reasoning.as_deref(), Some("thinking"));
    let mut completion_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut content_buf = String::new();

    let tail = flush_buffered_stream_tail(
        &tx,
        "chatcmpl-test",
        0,
        "kiln-test",
        &mut splitter,
        &mut completion_buf,
        &mut reasoning_buf,
        &mut content_buf,
        &mut ToolCallGate::new(true),
        "timeout",
    )
    .await;
    drop(tx);

    assert!(
        tail.is_none(),
        "reasoning-only tails have no buffered content to salvage"
    );
    assert_eq!(reasoning_buf, "</thi");
    let payloads = sse_data_payloads(&sse_body_from_events(rx).await);
    assert_eq!(payloads.len(), 1);
    assert_eq!(
        payloads[0]["choices"][0]["delta"]["reasoning_content"], "</thi",
        "pending splitter bytes must drain to the client on timeout"
    );
}

#[tokio::test]
async fn flush_buffered_stream_tail_skips_unbuffered_streams() {
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let mut splitter = ReasoningSplitter::new(false);
    let mut completion_buf = String::new();
    let mut reasoning_buf = String::new();
    // Without tool-call buffering this content already streamed to
    // the client delta-by-delta; re-emitting it would duplicate it.
    let mut content_buf = "already streamed".to_string();

    let tail = flush_buffered_stream_tail(
        &tx,
        "chatcmpl-test",
        0,
        "kiln-test",
        &mut splitter,
        &mut completion_buf,
        &mut reasoning_buf,
        &mut content_buf,
        &mut ToolCallGate::new(false),
        "timeout",
    )
    .await;
    drop(tx);

    assert!(tail.is_none());
    assert!(
        sse_data_payloads(&sse_body_from_events(rx).await).is_empty(),
        "non-buffered streams must not replay content on timeout"
    );
}

#[tokio::test]
async fn flush_buffered_stream_tail_salvages_content_for_record_after_disconnect() {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    drop(rx);
    let mut splitter = ReasoningSplitter::new(false);
    let mut completion_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut content_buf = "partial answer text".to_string();

    let tail = flush_buffered_stream_tail(
        &tx,
        "chatcmpl-test",
        0,
        "kiln-test",
        &mut splitter,
        &mut completion_buf,
        &mut reasoning_buf,
        &mut content_buf,
        &mut ToolCallGate::new(true),
        "timeout",
    )
    .await
    .expect("salvage must survive a dropped SSE client for the request record");

    assert_eq!(tail.preview_source(), "partial answer text");

    let record_completion = stream_tail_record_completion(Some(tail), &completion_buf);
    assert_eq!(
        record_completion, "partial answer text",
        "the record must keep the full salvaged content, not the capped preview buffer"
    );
}

#[test]
fn tool_call_responses_round_trip_through_chat_caches() {
    let tool_calls = vec![serde_json::json!({
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
    })];
    let resp = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion",
        created: 0,
        model: "kiln-test".to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: String::new(),
                reasoning_content: None,
                tool_calls: Some(tool_calls.clone()),
                name: None,
                tool_call_id: None,
            },
            finish_reason: "tool_calls".to_string(),
            thinking_budget: None,
            rollout_provenance: None,
            completion_tokens: 8,
        }],
        usage: Usage {
            prompt_tokens: 4,
            completion_tokens: 8,
            total_tokens: 12,
        },
        metadata: ChatCompletionMetadata {
            thinking_enabled: false,
            thinking_mode: "non_reasoning".to_string(),
            thinking_source: "template_default",
            default_thinking_enabled: None,
            final_content_empty: true,
            content_empty_reason: Some("tool_call"),
            reasoning_folded_into_content: false,
            thinking_budget: ThinkingBudgetMetadata::default(),
            config_hashes: None,
            performance: None,
        },
    };

    assert_eq!(
        cache_value_from_response(&resp)
            .unwrap()
            .tool_calls
            .as_ref(),
        Some(&tool_calls)
    );
    assert_eq!(
        chat_request_cache_value_from_response(&resp)
            .unwrap()
            .completion
            .tool_calls
            .as_ref(),
        Some(&tool_calls)
    );
    assert_eq!(
        chat_choices_cache_value_from_response(&resp)
            .unwrap()
            .completions[0]
            .tool_calls
            .as_ref(),
        Some(&tool_calls)
    );
}

#[test]
fn batch_cached_chat_layers_preserve_tool_calls() {
    let state = make_batch_test_state();
    let req = parse_batch_request(
        r#"{
                "prompts":[[{"role":"user","content":"inspect"}]],
                "n":1,
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}]
            }"#,
    );
    let tool_calls = vec![serde_json::json!({
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"}
    })];
    let completion = DeterministicCompletionCacheValue {
        text: String::new(),
        reasoning_content: Some("Need the working directory.".to_string()),
        tool_calls: Some(tool_calls.clone()),
        finish_reason: "tool_calls".to_string(),
        completion_tokens: 7,
        thinking_budget_status: None,
    };

    let from_choices = batch_response_from_cached_chat_choices(
        &state,
        &req,
        DeterministicChatChoicesCacheValue {
            prompt_tokens: 11,
            completions: vec![completion.clone()],
        },
    );
    assert_eq!(
        from_choices.completions[0].tool_calls.as_ref(),
        Some(&tool_calls)
    );

    let from_requests = batch_response_from_cached_chat_requests(
        &state,
        &req,
        vec![DeterministicChatRequestCacheValue {
            prompt_tokens: 11,
            completion,
        }],
    );
    assert_eq!(
        from_requests.completions[0].tool_calls.as_ref(),
        Some(&tool_calls)
    );
}

#[test]
fn batch_items_serialize_openai_tool_calls() {
    let raw = "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>";
    let output =
        assistant_output_from_split_parts_with_tool_parsing(true, None, raw.to_string(), "stop");
    let item = BatchCompletionItem {
        prompt_index: 0,
        completion_index: 0,
        text: output.content,
        reasoning_content: output.reasoning_content,
        tool_calls: output.tool_calls,
        finish_reason: output.finish_reason,
        thinking_budget: None,
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 7,
            total_tokens: 12,
        },
    };

    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(json["text"], "");
    assert_eq!(json["finish_reason"], "tool_calls");
    assert_eq!(json["tool_calls"][0]["type"], "function");
    assert_eq!(json["tool_calls"][0]["function"]["name"], "bash");
    let args: serde_json::Value = serde_json::from_str(
        json["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(args["command"], "pwd");
    assert!(
        json.get("reasoning_content").is_none(),
        "batch responses should omit reasoning_content when absent"
    );
}

#[test]
fn batch_items_serialize_reasoning_content_when_present() {
    let item = BatchCompletionItem {
        prompt_index: 0,
        completion_index: 0,
        text: String::new(),
        reasoning_content: Some("Still thinking.".to_string()),
        tool_calls: None,
        finish_reason: "length".to_string(),
        thinking_budget: None,
        usage: Usage {
            prompt_tokens: 5,
            completion_tokens: 7,
            total_tokens: 12,
        },
    };

    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(json["text"], "");
    assert_eq!(json["reasoning_content"], "Still thinking.");
}

#[test]
fn batch_cache_value_preserves_tool_calls() {
    let tool_calls = vec![serde_json::json!({
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
    })];
    let resp = BatchCompletionResponse {
        id: "batchcmpl-test".to_string(),
        object: "batch.completion",
        created: 0,
        model: "kiln-test".to_string(),
        completions: vec![BatchCompletionItem {
            prompt_index: 0,
            completion_index: 0,
            text: String::new(),
            reasoning_content: None,
            tool_calls: Some(tool_calls.clone()),
            finish_reason: "tool_calls".to_string(),
            thinking_budget: None,
            usage: Usage {
                prompt_tokens: 4,
                completion_tokens: 8,
                total_tokens: 12,
            },
        }],
        usage: Usage {
            prompt_tokens: 4,
            completion_tokens: 8,
            total_tokens: 12,
        },
        metadata: BatchCompletionMetadata::default(),
    };
    let cached = cache_value_from_batch_response(&resp);
    let req = parse_batch_request(
        r#"{
                "prompts":[[{"role":"user","content":"list files"}]],
                "thinking_budget_tokens":null,
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}]
            }"#,
    );
    let mut state = make_batch_test_state();
    state.default_thinking_budget_tokens = Some(99);
    let rehydrated = batch_response_from_cached_value(&state, &req, cached);

    assert_eq!(
        rehydrated.completions[0].tool_calls.as_ref(),
        Some(&tool_calls)
    );
    assert_eq!(rehydrated.completions[0].finish_reason, "tool_calls");
    assert_eq!(
        rehydrated.metadata.thinking_budget.tokens_source,
        "request_unlimited"
    );
    assert_eq!(rehydrated.metadata.thinking_budget.max_tokens, None);
}

#[test]
fn batch_tool_call_items_round_trip_through_chat_caches() {
    let tool_calls = vec![serde_json::json!({
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
    })];
    let item = BatchCompletionItem {
        prompt_index: 0,
        completion_index: 0,
        text: String::new(),
        reasoning_content: None,
        tool_calls: Some(tool_calls.clone()),
        finish_reason: "tool_calls".to_string(),
        thinking_budget: None,
        usage: Usage {
            prompt_tokens: 4,
            completion_tokens: 8,
            total_tokens: 12,
        },
    };

    assert_eq!(
        chat_request_cache_value_from_batch_item(&item)
            .unwrap()
            .completion
            .tool_calls
            .as_ref(),
        Some(&tool_calls)
    );
    assert_eq!(
        chat_choices_cache_value_from_batch_items(vec![&item], 1)
            .unwrap()
            .completions[0]
            .tool_calls
            .as_ref(),
        Some(&tool_calls)
    );
}

// ── Batch completion endpoint ───────────────────────────────────

fn parse_batch_request(json: &str) -> BatchCompletionRequest {
    serde_json::from_str(json).expect("batch request should deserialize")
}

#[test]
fn batch_thinking_budgets_preserve_null_and_numeric_overrides() {
    let inherited = parse_batch_request(r#"{"prompts":[[]]}"#);
    assert_eq!(inherited.thinking_budget_tokens, BudgetOverride::Inherit);
    let request = parse_batch_request(
        r#"{"prompts":[[]],"thinking_budget_tokens":null,"thinking_budget_ms":250}"#,
    );
    assert_eq!(request.thinking_budget_tokens, BudgetOverride::Unlimited);
    assert_eq!(request.thinking_budget_ms, BudgetOverride::Limited(250));
}

fn make_batch_test_state() -> AppState {
    let config = ModelConfig::qwen3_5_4b();
    let sched_config = kiln_scheduler::SchedulerConfig {
        max_batch_tokens: 8192,
        max_batch_size: 64,
        block_size: 16,
        prefix_cache_enabled: false,
        ..Default::default()
    };
    let scheduler = kiln_scheduler::Scheduler::new(sched_config, 256);
    let engine = kiln_model::engine::MockEngine::new(config.clone());
    let tokenizer = crate::api::test_tokenizer();
    AppState::new_mock(
        config,
        scheduler,
        std::sync::Arc::new(engine),
        tokenizer,
        300,
        "kiln-test".to_string(),
    )
}

#[test]
fn batch_metadata_reports_effective_budgets_and_provenance() {
    let mut state = make_batch_test_state();
    state.default_thinking_budget_tokens = Some(64);
    state.default_thinking_budget_ms = Some(1_000);

    let inherited = parse_batch_request(r#"{"prompts":[[]]}"#);
    assert_eq!(
        serde_json::to_value(batch_completion_metadata_for_request(&state, &inherited)).unwrap(),
        serde_json::json!({
            "thinking_budget": {
                "configured": true,
                "max_tokens": 64,
                "max_time_ms": 1_000,
                "tokens_source": "server_default",
                "time_source": "server_default"
            }
        })
    );

    let overridden = parse_batch_request(
        r#"{"prompts":[[]],"thinking_budget_tokens":0,"thinking_budget_ms":null}"#,
    );
    assert_eq!(
        serde_json::to_value(batch_completion_metadata_for_request(&state, &overridden)).unwrap(),
        serde_json::json!({
            "thinking_budget": {
                "configured": true,
                "max_tokens": 0,
                "tokens_source": "request",
                "time_source": "request_unlimited"
            }
        })
    );

    state.default_thinking_budget_tokens = None;
    state.default_thinking_budget_ms = None;
    assert_eq!(
        serde_json::to_value(batch_completion_metadata_for_request(&state, &inherited)).unwrap(),
        serde_json::json!({
            "thinking_budget": {
                "configured": false,
                "tokens_source": "unlimited",
                "time_source": "unlimited"
            }
        })
    );
}

fn make_prompt_logprobs_test_state() -> AppState {
    let mut state = make_batch_test_state();
    let vocab = (0..512u32)
        .map(|token_id| (format!("token-{token_id}"), token_id))
        .collect::<std::collections::HashMap<_, _>>();
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] }
    });
    state.tokenizer = std::sync::Arc::new(
        kiln_core::tokenizer::KilnTokenizer::from_bytes(
            &serde_json::to_vec(&tokenizer_json).unwrap(),
        )
        .unwrap(),
    );
    state
}

fn test_teacher_identity() -> kiln_train::TeacherIdentityV1 {
    kiln_train::TeacherIdentityV1::new(
        "kiln-test",
        "a".repeat(64),
        "b".repeat(64),
        "c".repeat(64),
        None,
        512,
        256,
        4096,
        65_536,
        "kiln-test/cpu",
        "d".repeat(64),
    )
    .unwrap()
}

#[test]
fn completion_fingerprint_requires_identity_for_real_backends() {
    let error = canonical_completion_fingerprint(None, true).unwrap_err();
    assert_eq!(error.code, "internal_error");
    assert!(error.message.contains("no verified base teacher identity"));
    assert_eq!(canonical_completion_fingerprint(None, false).unwrap(), None);

    let identity = test_teacher_identity();
    let fingerprint = canonical_completion_fingerprint(Some(&identity), true)
        .unwrap()
        .unwrap();
    assert_eq!(fingerprint, identity.fingerprint());
    assert_eq!(
        kiln_train::TeacherIdentityV1::parse_fingerprint(&fingerprint).unwrap(),
        identity
    );
}

#[test]
fn text_completion_response_serializes_fingerprint_and_mock_null() {
    let identity = test_teacher_identity();
    let response = TextCompletionResponse {
        id: "cmpl-test".to_string(),
        object: "text_completion",
        created: 1,
        model: "kiln-test".to_string(),
        system_fingerprint: Some(identity.fingerprint()),
        choices: Vec::new(),
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 0,
            total_tokens: 1,
        },
    };
    let json = serde_json::to_value(response).unwrap();
    assert_eq!(
        json["system_fingerprint"],
        serde_json::Value::String(identity.fingerprint())
    );

    let mock = serde_json::to_value(TextCompletionResponse {
        id: "cmpl-mock".to_string(),
        object: "text_completion",
        created: 1,
        model: "kiln-test".to_string(),
        system_fingerprint: None,
        choices: Vec::new(),
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 0,
            total_tokens: 1,
        },
    })
    .unwrap();
    assert!(mock["system_fingerprint"].is_null());
}

#[test]
fn slow_request_log_values_respect_threshold_and_redact_prompt_text() {
    let mut state = make_batch_test_state();
    state.slow_request_warn_threshold = Some(std::time::Duration::from_millis(50));
    let mut record = RequestRecord {
        user_agent: None,
        id: "chatcmpl-test".to_string(),
        timestamp_unix_ms: 0,
        model: "kiln-test".to_string(),
        prompt_preview: "secret prompt".to_string(),
        completion_preview: "ok".to_string(),
        prompt_tokens: 12,
        completion_tokens: 4,
        duration_ms: 49,
        streamed: false,
        finish_reason: "stop".to_string(),
        adapter: Some("adapter-a".to_string()),
        max_tokens: Some(8),
        thinking_mode: Some("reasoning".to_string()),
        prefix_cache: Some("hit".to_string()),
        prompt_full: Some("secret full prompt".to_string()),
        ..RequestRecord::default()
    };

    assert!(slow_request_log_values(&state, &record).is_none());

    record.duration_ms = 50;
    let values = slow_request_log_values(&state, &record).unwrap();
    assert_eq!(values.request_id, "chatcmpl-test");
    assert_eq!(values.adapter, "adapter-a");
    assert_eq!(values.prompt_tokens, 12);
    assert_eq!(values.max_output_tokens, 8);
    assert_eq!(values.generated_tokens, 4);
    assert_eq!(values.elapsed_ms, 50);
    assert_eq!(values.threshold_ms, 50);
    assert_eq!(values.thinking_mode, "reasoning");
    assert_eq!(values.prefix_cache, "hit");
    assert_eq!(values.finish_reason, "stop");

    let debug = format!("{values:?}");
    assert!(!debug.contains("secret prompt"));
    assert!(!debug.contains("secret full prompt"));
}

/// Build a minimal request body, invoke the route, and return (status, body).
async fn batch_post(
    state: AppState,
    body_json: &str,
) -> (axum::http::StatusCode, serde_json::Value) {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let app = routes().with_state(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions/batch")
                .header("content-type", "application/json")
                .body(Body::from(body_json.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, body)
}

async fn chat_post(
    state: AppState,
    body_json: &str,
) -> (axum::http::StatusCode, serde_json::Value) {
    use axum::body::to_bytes;

    let resp = chat_post_raw(state, body_json).await;
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, body)
}

async fn chat_post_raw(state: AppState, body_json: &str) -> Response {
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    let app = routes().with_state(state);
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body_json.to_string()))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn completion_post(
    state: AppState,
    body_json: &str,
) -> (axum::http::StatusCode, serde_json::Value) {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let app = routes().with_state(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body_json.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, body)
}

async fn chat_post_text(state: AppState, body_json: &str) -> (axum::http::StatusCode, String) {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let app = routes().with_state(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body_json.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    (status, body)
}

async fn metrics_get_text(state: AppState) -> String {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    let response = crate::api::metrics::routes()
        .with_state(state)
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    String::from_utf8(
        to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}

#[tokio::test]
async fn chat_rejects_zero_n() {
    let (status, body) = chat_post(
        make_batch_test_state(),
        r#"{"messages":[{"role":"user","content":"hi"}],"n":0}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "chat_invalid_request");
}

#[tokio::test]
async fn chat_rejects_streaming_multi_choice() {
    let (status, body) = chat_post(
        make_batch_test_state(),
        r#"{"messages":[{"role":"user","content":"hi"}],"n":2,"stream":true}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "chat_invalid_request");
}

#[tokio::test]
async fn completions_prompt_logprobs_accepts_token_id_prompt() {
    let body = serde_json::json!({
        "model": "kiln-test",
        "prompt": [11, 22, 33],
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 3
    })
    .to_string();

    let (status, json) = completion_post(make_prompt_logprobs_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    assert_eq!(json["object"], "text_completion");
    assert!(json["system_fingerprint"].is_null());
    assert_eq!(json["usage"]["prompt_tokens"], 3);
    assert_eq!(json["usage"]["completion_tokens"], 0);
    let prompt_logprobs = json["choices"][0]["prompt_logprobs"].as_array().unwrap();
    assert_eq!(prompt_logprobs.len(), 3);
    assert!(prompt_logprobs[0].is_null());
    // Mock top-3 is token IDs 0, 1, 2. Actual prompt tokens 22 and 33
    // are outside top-K, so vLLM compatibility requires K+1 entries.
    assert_eq!(prompt_logprobs[1].as_object().unwrap().len(), 4);
    assert_eq!(prompt_logprobs[2].as_object().unwrap().len(), 4);
    assert_eq!(prompt_logprobs[1]["22"]["rank"], 23);
    assert_eq!(prompt_logprobs[2]["33"]["rank"], 34);
    assert!(
        prompt_logprobs[1]
            .as_object()
            .unwrap()
            .values()
            .all(|entry| entry["logprob"].is_number() && entry["rank"].is_number())
    );
}

#[tokio::test]
async fn completions_prompt_logprobs_zero_returns_only_observed_tokens() {
    let body = serde_json::json!({
        "prompt": [11, 22, 33],
        "max_tokens": 0,
        "prompt_logprobs": 0
    })
    .to_string();

    let (status, json) = completion_post(make_prompt_logprobs_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    let prompt_logprobs = json["choices"][0]["prompt_logprobs"].as_array().unwrap();
    assert!(prompt_logprobs[0].is_null());
    assert_eq!(prompt_logprobs[1].as_object().unwrap().len(), 1);
    assert_eq!(prompt_logprobs[2].as_object().unwrap().len(), 1);
    assert_eq!(prompt_logprobs[1]["22"]["rank"], 23);
    assert_eq!(prompt_logprobs[2]["33"]["rank"], 34);
}

#[tokio::test]
async fn mock_prompt_logprobs_obeys_k_or_k_plus_one_cardinality() {
    let body = serde_json::json!({
        "prompt": [9, 1, 4],
        "max_tokens": 1,
        "prompt_logprobs": 2
    })
    .to_string();

    let (status, json) = completion_post(make_prompt_logprobs_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    let prompt_logprobs = json["choices"][0]["prompt_logprobs"].as_array().unwrap();
    // Actual token 1 is top-2; actual token 4 is not.
    assert_eq!(prompt_logprobs[1].as_object().unwrap().len(), 2);
    assert_eq!(prompt_logprobs[1]["1"]["rank"], 2);
    assert_eq!(prompt_logprobs[2].as_object().unwrap().len(), 3);
    assert_eq!(prompt_logprobs[2]["4"]["rank"], 5);
}

#[test]
fn prompt_logprob_map_propagates_token_decode_failure() {
    let mut attempted_token_ids = Vec::new();
    let err = top_k_logprob_map_with_decoder(
        &[-2.0, -0.5, -1.0],
        &[-2.0, -0.5, -1.0],
        3,
        1,
        2,
        &[7, 8],
        |token_id, context| {
            assert_eq!(context, &[7, 8]);
            attempted_token_ids.push(token_id);
            Err(TokenizerError::Decode(
                "injected decode failure".to_string(),
            ))
        },
    )
    .unwrap_err();

    assert_eq!(attempted_token_ids, vec![1]);
    assert_eq!(err.status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(err.code, "tokenization_error");
    assert!(err.message.contains("token id 1"), "{err:?}");
    assert!(err.message.contains("injected decode failure"), "{err:?}");
}

#[test]
fn prompt_logprob_map_rejects_non_model_vocab_width_before_decode() {
    for row in [vec![-1.0; 3], vec![-1.0; 5]] {
        let mut decode_attempts = 0;
        let err = top_k_logprob_map_with_decoder(&row, &row, 4, 0, 2, &[], |_, _| {
            decode_attempts += 1;
            Ok("unused".to_string())
        })
        .unwrap_err();

        assert_eq!(decode_attempts, 0);
        assert_eq!(err.code, "generation_error");
        assert!(err.message.contains(&format!("row width {}", row.len())));
        assert!(err.message.contains("model vocabulary size 4"));
    }
}

#[test]
fn prompt_logprob_map_rejects_every_non_finite_value_before_decode() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut row = vec![-0.25, -0.5, -0.75, -1.0];
        // This position is outside top-1 for both finite and -Inf values.
        row[3] = value;
        let mut decode_attempts = 0;
        let err = top_k_logprob_map_with_decoder(&row, &row, 4, 0, 1, &[], |_, _| {
            decode_attempts += 1;
            Ok("unused".to_string())
        })
        .unwrap_err();

        assert_eq!(decode_attempts, 0, "value {value:?}");
        assert_eq!(err.code, "generation_error");
        assert!(err.message.contains("non-finite value"), "{err:?}");
        assert!(err.message.contains("token id 3"), "{err:?}");
    }
}

#[test]
fn prompt_logprob_map_ties_are_token_id_ordered_and_keep_duplicate_displays() {
    let row = [0.0, -0.0, -0.25, -0.25, -1.0];

    for top_k in 1..=row.len() {
        let map =
            top_k_logprob_map_with_decoder(&row, &row, row.len(), 4, top_k, &[99], |_, context| {
                assert_eq!(context, &[99]);
                Ok("same-display-token".to_string())
            })
            .unwrap();
        let expected_len = top_k + usize::from(top_k < row.len());
        assert_eq!(map.len(), expected_len);
        assert_eq!(map["4"].rank, 5);

        let mut ranked = map
            .iter()
            .filter(|(token_id, _)| token_id.as_str() != "4")
            .map(|(token_id, entry)| {
                assert_eq!(entry.decoded_token, "same-display-token");
                (entry.rank, token_id.parse::<TokenId>().unwrap())
            })
            .collect::<Vec<_>>();
        ranked.sort_unstable();
        assert_eq!(
            ranked.iter().map(|(rank, _)| *rank).collect::<Vec<_>>(),
            (1..=top_k.min(4)).collect::<Vec<_>>()
        );

        let expected_ids = [0, 1, 2, 3][..top_k.min(4)].to_vec();
        assert_eq!(
            ranked
                .into_iter()
                .map(|(_, token_id)| token_id)
                .collect::<Vec<_>>(),
            expected_ids
        );
    }
}

#[test]
fn prompt_logprob_observed_rank_is_ordinal_in_topk_and_full_rank_when_extra() {
    let row = [0.0, -0.25, -0.25, -0.25, -1.0];

    let selected_tie =
        top_k_logprob_map_with_decoder(&row, &row, row.len(), 1, 2, &[], |token_id, _| {
            Ok(format!("token-{token_id}"))
        })
        .unwrap();
    assert_eq!(selected_tie.len(), 2);
    assert_eq!(selected_tie["0"].rank, 1);
    // Token-ID tie breaking selects token 1 and its full tie rank is
    // overwritten by the top-K ordinal, matching vLLM's dictionary result.
    assert_eq!(selected_tie["1"].rank, 2);

    let extra_tie =
        top_k_logprob_map_with_decoder(&row, &row, row.len(), 3, 2, &[], |token_id, _| {
            Ok(format!("token-{token_id}"))
        })
        .unwrap();
    assert_eq!(extra_tie.len(), 3);
    // Four values are >= token 3's value: token IDs 0, 1, 2, and 3.
    assert_eq!(extra_tie["3"].rank, 4);
}

#[test]
fn device_prompt_logprob_selection_preserves_observed_rank_semantics() {
    let selected_row = kiln_tensor::DevicePromptLogprobRow {
        row_max: 0.0,
        log_sum_exp_shifted: 0.5,
        observed_logit: 0.0,
        observed_logprob: -0.5,
        observed_full_rank: 4,
        candidates: vec![
            kiln_tensor::DevicePromptLogprobCandidate {
                token_id: 0,
                logit: 0.0,
                logprob: -0.5,
            },
            kiln_tensor::DevicePromptLogprobCandidate {
                token_id: 1,
                logit: 0.0,
                logprob: -0.5,
            },
        ],
    };
    let selected = select_prompt_logprobs_from_device_row(&selected_row, 5, 1, 2).unwrap();
    assert_eq!(selected.entries.len(), 2);
    assert_eq!(selected.entries[0].token_id, 1);
    assert_eq!(selected.entries[0].rank, 2);

    let extra_row = kiln_tensor::DevicePromptLogprobRow {
        observed_logit: -0.25,
        observed_logprob: -0.75,
        observed_full_rank: 4,
        ..selected_row
    };
    let extra = select_prompt_logprobs_from_device_row(&extra_row, 5, 3, 2).unwrap();
    assert_eq!(extra.entries.len(), 3);
    assert_eq!(extra.entries[0].token_id, 3);
    assert_eq!(extra.entries[0].rank, 4);
    assert_eq!(extra.entries[1].rank, 1);
    assert_eq!(extra.entries[2].rank, 2);
}

#[test]
fn device_prompt_logprob_selection_fails_closed_on_observed_disagreement() {
    let row = kiln_tensor::DevicePromptLogprobRow {
        row_max: 0.0,
        log_sum_exp_shifted: 0.5,
        observed_logit: 0.0,
        observed_logprob: -0.5,
        observed_full_rank: 1,
        candidates: vec![kiln_tensor::DevicePromptLogprobCandidate {
            token_id: 2,
            logit: 0.0,
            logprob: -0.75,
        }],
    };
    let error = select_prompt_logprobs_from_device_row(&row, 4, 2, 1).unwrap_err();
    assert_eq!(error.code, "generation_error");
    assert!(error.message.contains("disagreed"));
}

#[test]
fn prompt_logprob_selection_ranks_logits_before_f32_logsoftmax_collapse() {
    // Subtracting this large LSE in F32 collapses distinct tail logits 9
    // and 8 to the same log-probability. vLLM selects and ranks from the
    // original logits, then attaches the selected F32 log-probabilities.
    let logits = [33_554_432.0, 10.0, 9.0, 8.0];
    let logprobs = [0.0, -33_554_422.0, -33_554_424.0, -33_554_424.0];
    let observed = top_k_logprob_map_with_decoder(
        &logits,
        &logprobs,
        logits.len(),
        2,
        1,
        &[],
        |token_id, _| Ok(format!("token-{token_id}")),
    )
    .unwrap();
    assert_eq!(observed["2"].rank, 3);
    assert_eq!(observed["2"].logprob, logprobs[2]);

    let reordered_logits = [33_554_432.0, 10.0, 8.0, 9.0];
    let selected = top_k_logprob_map_with_decoder(
        &reordered_logits,
        &logprobs,
        reordered_logits.len(),
        0,
        3,
        &[],
        |token_id, _| Ok(format!("token-{token_id}")),
    )
    .unwrap();
    assert!(selected.contains_key("3"));
    assert!(!selected.contains_key("2"));
}

#[test]
fn prompt_logprob_projection_chunks_respect_the_vocabulary_byte_budget() {
    assert_eq!(prompt_logprob_projection_chunk_tokens(1), 32);
    assert_eq!(prompt_logprob_projection_chunk_tokens(248_320), 32);
    assert_eq!(prompt_logprob_projection_chunk_tokens(1_000_000), 8);
    assert_eq!(prompt_logprob_projection_chunk_tokens(usize::MAX), 1);

    for vocab_size in [1, 151_936, 248_320, 1_000_000] {
        let rows = prompt_logprob_projection_chunk_tokens(vocab_size);
        assert!((1..=MAX_PROMPT_LOGPROB_PROJECTION_CHUNK_TOKENS).contains(&rows));
        assert!(
            rows * vocab_size * 2 * std::mem::size_of::<f32>()
                <= PROMPT_LOGPROB_PROJECTION_BYTE_BUDGET
        );
    }
}

#[test]
fn prompt_logprob_panic_fence_releases_settled_ownership() {
    let backend_health = kiln_model::BackendHealthHandle::default();
    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let value = run_prompt_logprob_worker_with_panic_fence(
        &backend_health,
        PromptLogprobDropProbe(drops.clone()),
        |_| Ok(7usize),
    )
    .unwrap();
    assert_eq!(value, 7);
    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);

    let error = run_prompt_logprob_worker_with_panic_fence(
        &backend_health,
        PromptLogprobDropProbe(drops.clone()),
        |_| Err::<(), _>(anyhow::anyhow!("settled scoring error")),
    )
    .unwrap_err();
    assert!(error.to_string().contains("settled scoring error"));
    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert!(!backend_health.snapshot().quarantined);
}

#[test]
fn prompt_logprob_panic_fence_quarantines_and_retains_unknown_ownership() {
    let backend_health = kiln_model::BackendHealthHandle::default();
    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let error = run_prompt_logprob_worker_with_panic_fence(
        &backend_health,
        PromptLogprobDropProbe(drops.clone()),
        |_| -> anyhow::Result<()> { panic!("injected scorer panic") },
    )
    .unwrap_err();

    assert!(error.to_string().contains("GPU ownership are unknown"));
    assert!(backend_health.snapshot().quarantined);
    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[test]
fn prompt_logprob_panic_fence_retains_ownership_after_sync_failure() {
    let backend_health = kiln_model::BackendHealthHandle::default();
    let health_for_work = backend_health.clone();
    let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let error = run_prompt_logprob_worker_with_panic_fence(
        &backend_health,
        PromptLogprobDropProbe(drops.clone()),
        move |_| {
            health_for_work.quarantine("injected prompt-logprobs sync failure");
            Err::<(), _>(anyhow::anyhow!("sync failed"))
        },
    )
    .unwrap_err();

    assert!(error.to_string().contains("sync failed"));
    assert!(backend_health.snapshot().quarantined);
    assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[test]
fn prompt_logprob_candidates_share_only_preceding_actual_context() {
    let row = [0.0, -1.0, -2.0, -3.0];
    let mut calls = Vec::new();
    let context = [41, 42, 43];
    let map = top_k_logprob_map_with_decoder(
        &row,
        &row,
        row.len(),
        3,
        2,
        &context,
        |token_id, received_context| {
            calls.push((token_id, received_context.to_vec()));
            Ok(format!("token-{token_id}"))
        },
    )
    .unwrap();

    assert_eq!(map.len(), 3);
    assert_eq!(calls.len(), 3);
    assert!(calls.iter().all(|(_, seen)| seen == &context));
    assert_eq!(
        calls.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        [3, 0, 1]
    );
}

#[test]
fn prompt_logprob_rows_use_previous_row_for_each_observed_prompt_token() {
    let mut state = make_prompt_logprobs_test_state();
    state.model_config.vocab_size = 4;
    let prompt_tokens = [0, 2, 3];
    let rows = vec![vec![-0.1, -0.2, -3.0, -0.4], vec![-4.0, -0.1, -0.2, -2.0]];

    let output = prompt_logprobs_from_rows(&state, &prompt_tokens, &rows, &rows, 1).unwrap();
    assert!(output[0].is_none());
    let position_one = output[1].as_ref().unwrap();
    let position_two = output[2].as_ref().unwrap();
    assert_eq!(position_one["2"].logprob, -3.0);
    assert_eq!(position_one["2"].rank, 4);
    assert_eq!(position_two["3"].logprob, -2.0);
    assert_eq!(position_two["3"].rank, 3);
}

#[test]
fn prompt_logprob_rows_require_only_rows_that_predict_observed_tokens() {
    let mut state = make_prompt_logprobs_test_state();
    state.model_config.vocab_size = 4;
    let one_token = prompt_logprobs_from_rows(&state, &[0], &[], &[], 1).unwrap();
    assert_eq!(one_token.len(), 1);
    assert!(one_token[0].is_none());

    let prompt_tokens = [0, 1];
    let rows = vec![vec![-0.1, -0.2, -0.3, -0.4]];
    let output = prompt_logprobs_from_rows(&state, &prompt_tokens, &rows, &rows, 1).unwrap();
    assert_eq!(output.len(), 2);
    assert!(output[0].is_none());
    assert!(output[1].is_some());
}

#[tokio::test]
async fn completions_prompt_logprobs_rejects_token_absent_from_tokenizer() {
    let body = serde_json::json!({
        "prompt": [0, 11],
        "max_tokens": 1,
        "prompt_logprobs": 1
    })
    .to_string();

    let (status, json) = completion_post(make_batch_test_state(), &body).await;
    assert_eq!(
        status,
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        "{json}"
    );
    assert_eq!(json["error"]["code"], "tokenization_error");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("token id 11"),
        "{json}"
    );
}

#[tokio::test]
async fn completions_prompt_logprobs_rejects_missing_prompt_logprobs() {
    let body = serde_json::json!({
        "prompt": [11, 22, 33],
        "max_tokens": 1
    })
    .to_string();

    let (status, json) = completion_post(make_batch_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "completion_invalid_request");
}

#[tokio::test]
async fn completions_prompt_logprobs_rejects_topk_above_cap() {
    let body = serde_json::json!({
        "prompt": [11, 22, 33],
        "max_tokens": 1,
        "prompt_logprobs": MAX_COMPLETION_PROMPT_LOGPROBS + 1
    })
    .to_string();

    let (status, json) = completion_post(make_batch_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "completion_invalid_request");
}

#[tokio::test]
async fn completions_prompt_logprobs_rejects_mismatched_model_identity() {
    let body = serde_json::json!({
        "model": "not-the-served-model",
        "prompt": [11, 22],
        "max_tokens": 0,
        "prompt_logprobs": 1
    })
    .to_string();

    let (status, json) = completion_post(make_prompt_logprobs_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "completion_invalid_request");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not served")
    );
}

#[tokio::test]
async fn completions_prompt_logprobs_caps_total_response_candidates() {
    let tokens = vec![7u32; 300];
    let body = serde_json::json!({
        "prompt": tokens,
        "max_tokens": 0,
        "prompt_logprobs": MAX_COMPLETION_PROMPT_LOGPROBS
    })
    .to_string();

    let (status, json) = completion_post(make_prompt_logprobs_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "completion_invalid_request");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("candidates")
    );
}

#[tokio::test]
async fn completions_prompt_logprobs_rejects_out_of_range_token_ids() {
    let state = make_batch_test_state();
    let vocab_size = state.model_config.vocab_size;
    let body = serde_json::json!({
        "prompt": [11, vocab_size, 33],
        "max_tokens": 1,
        "prompt_logprobs": 4
    })
    .to_string();

    let (status, json) = completion_post(state, &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "completion_invalid_request");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("out of range"),
        "{json}"
    );
}

#[tokio::test]
async fn completions_prompt_logprobs_rejects_over_long_prompts() {
    let tokens: Vec<u32> = vec![7; MAX_COMPLETION_PROMPT_TOKENS + 1];
    let body = serde_json::json!({
        "prompt": tokens,
        "max_tokens": 1,
        "prompt_logprobs": 4
    })
    .to_string();

    let (status, json) = completion_post(make_batch_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "completion_invalid_request");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("capped"),
        "{json}"
    );
}

#[tokio::test]
async fn completions_prompt_logprobs_respects_served_model_context_window() {
    let mut state = make_prompt_logprobs_test_state();
    state.model_config.max_position_embeddings = 2;
    let body = serde_json::json!({
        "prompt": [11, 22, 33],
        "max_tokens": 0,
        "prompt_logprobs": 1
    })
    .to_string();

    let (status, json) = completion_post(state, &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "context_length_exceeded");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("maximum context length is 2 tokens"),
        "{json}"
    );
}

#[tokio::test]
async fn completions_prompt_logprobs_accepts_boundary_valid_prompt() {
    // In-range ids at a modest length must pass validation and produce
    // mock logprobs.
    let body = serde_json::json!({
        "prompt": [11, 22, 33],
        "max_tokens": 1,
        "prompt_logprobs": 4
    })
    .to_string();

    let (status, json) = completion_post(make_prompt_logprobs_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    assert_eq!(json["usage"]["prompt_tokens"], 3);
}

#[tokio::test]
async fn chat_response_metadata_reports_server_thinking_default() {
    let mut state = make_batch_test_state();
    state.default_thinking_enabled = Some(false);
    let body = r#"{"messages":[{"role":"user","content":"metadata"}],"max_tokens":0}"#;

    let (status, json) = chat_post(state, body).await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(json["metadata"]["thinking_enabled"], false);
    assert_eq!(json["metadata"]["thinking_mode"], "non_reasoning");
    assert_eq!(json["metadata"]["thinking_source"], "server_default");
    assert_eq!(json["metadata"]["default_thinking_enabled"], false);
}

#[tokio::test]
async fn chat_performance_metadata_is_omitted_by_default() {
    let (status, json) = chat_post(
        make_batch_test_state(),
        r#"{"messages":[{"role":"user","content":"perf default"}],"max_tokens":0}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert!(
        json["metadata"].get("performance").is_none(),
        "performance metadata should be opt-in by default: {json}"
    );
}

#[tokio::test]
async fn chat_config_hash_metadata_can_be_enabled_by_request_flag() {
    let mut state = make_batch_test_state();
    state.config_hashes.effective_config_hash = Some(format!("sha256:{}", "a".repeat(64)));

    let (status, json) = chat_post(
            state,
            r#"{"messages":[{"role":"user","content":"hash request"}],"max_tokens":0,"include_config_hashes":true}"#,
        )
        .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    let hashes = &json["metadata"]["config_hashes"];
    assert!(
        hashes["model_config_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    assert!(
        hashes["tokenizer_config_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    assert!(hashes["chat_template_hash"].is_null());
    assert!(
        hashes["effective_config_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    assert!(
        json["metadata"].get("performance").is_none(),
        "config hash debug metadata must not imply performance metadata"
    );
}

#[tokio::test]
async fn chat_config_hash_metadata_can_be_enabled_by_server_config() {
    let mut state = make_batch_test_state();
    state.chat_config_hash_metadata = true;

    let (status, json) = chat_post(
        state.clone(),
        r#"{"messages":[{"role":"user","content":"hash server"}],"max_tokens":0}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    assert!(json["metadata"]["config_hashes"].is_object());

    let (status, json) = chat_post(
            state,
            r#"{"messages":[{"role":"user","content":"hash server disabled"}],"max_tokens":0,"include_config_hashes":false}"#,
        )
        .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    assert!(
        json["metadata"].get("config_hashes").is_none(),
        "request flag should be able to disable the server default"
    );
}

#[tokio::test]
async fn chat_performance_metadata_can_be_enabled_by_request_flag() {
    let (status, json) = chat_post(
            make_batch_test_state(),
            r#"{"messages":[{"role":"user","content":"perf request"}],"max_tokens":0,"include_performance":true}"#,
        )
        .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    let perf = &json["metadata"]["performance"];
    assert_eq!(perf["prompt_tokens"], json["usage"]["prompt_tokens"]);
    assert_eq!(perf["completion_tokens"], 0);
    assert_eq!(perf["ttft_ms"], 0.0);
    assert_eq!(perf["prefill_ms"], 0.0);
    assert!(perf["actor_queue_ms"].is_null());
    assert!(perf["actor_admission_ms"].is_null());
    assert!(perf["actor_prefill_wall_ms"].is_null());
    assert!(perf["resident_prefill_used"].is_null());
    assert_eq!(perf["decode_ms"], 0.0);
    assert!(perf["total_latency_ms"].as_f64().unwrap() >= 0.0);
    assert_eq!(perf["decode_tokens_per_sec"], 0.0);
    assert_eq!(perf["adapter_used"], "base");
    assert_eq!(perf["thinking_mode"], json["metadata"]["thinking_mode"]);
    assert_eq!(perf["finish_reason"], "length");
}

#[tokio::test]
async fn chat_performance_metadata_can_be_enabled_by_server_config() {
    let mut state = make_batch_test_state();
    state.chat_performance_metadata = true;

    let (status, json) = chat_post(
        state.clone(),
        r#"{"messages":[{"role":"user","content":"perf server"}],"max_tokens":0}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    assert!(json["metadata"]["performance"].is_object());

    let (status, json) = chat_post(
            state,
            r#"{"messages":[{"role":"user","content":"perf server disabled"}],"max_tokens":0,"include_performance":false}"#,
        )
        .await;
    assert_eq!(status, axum::http::StatusCode::OK, "{json}");
    assert!(
        json["metadata"].get("performance").is_none(),
        "request flag should be able to disable the server default"
    );
}

#[tokio::test]
async fn eval_mode_chat_completion_sets_headers_defaults_and_resets_caches() {
    use axum::body::to_bytes;

    let mut state = make_batch_test_state();
    state.eval_mode = true;
    *state.active_adapter_name.write().unwrap() = Some("eval-adapter".to_string());
    let state_for_assert = state.clone();
    let body = r#"{"messages":[{"role":"user","content":"eval direct"}],"max_tokens":2}"#;

    for _ in 0..16 {
        let resp = chat_post_raw(state.clone(), body).await;
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(resp.headers().get("x-kiln-eval-mode").unwrap(), "true");
        assert_eq!(
            resp.headers().get("x-kiln-active-adapter").unwrap(),
            "eval-adapter"
        );
        assert_eq!(resp.headers().get("x-kiln-loaded-adapter").unwrap(), "base");
        assert_eq!(
            resp.headers()
                .get("x-kiln-loaded-adapter-revision")
                .unwrap(),
            "base"
        );
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "chat.completion");
    }

    let recent = state_for_assert.recent_requests.lock().unwrap().snapshot();
    assert_eq!(recent.len(), 16);
    let latest = recent.first().unwrap();
    assert_eq!(latest.temperature, Some(0.0));
    assert_eq!(latest.top_p, Some(1.0));
    assert_eq!(latest.max_tokens, Some(2));
    assert_eq!(latest.thinking_mode.as_deref(), Some("mock"));

    assert_eq!(state_for_assert.completion_cache.lock().unwrap().stats(), 0);
    assert_eq!(
        state_for_assert.chat_request_cache.lock().unwrap().stats(),
        0
    );
    assert_eq!(
        state_for_assert.chat_choices_cache.lock().unwrap().stats(),
        0
    );
    assert_eq!(state_for_assert.batch_cache.lock().unwrap().stats(), 0);
    assert_eq!(
        state_for_assert
            .rendered_prompt_cache
            .lock()
            .unwrap()
            .stats()
            .2,
        0
    );
    assert_eq!(
        state_for_assert
            .prompt_token_cache
            .lock()
            .unwrap()
            .stats()
            .2,
        0
    );
}

#[test]
fn eval_mode_defaults_are_deterministic_and_disable_thinking_unless_overridden() {
    let mut state = make_batch_test_state();
    state.eval_mode = true;
    let mut req: ChatCompletionRequest =
        serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();

    apply_eval_mode_chat_defaults(&state, &mut req);
    assert_eq!(req.temperature, Some(0.0));
    assert_eq!(req.top_p, Some(1.0));
    assert_eq!(req.top_k, Some(0));
    assert_eq!(req.seed, Some(0));
    assert_eq!(
        req.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("enable_thinking")),
        Some(&serde_json::Value::Bool(false))
    );

    let mut explicit: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.7,"chat_template_kwargs":{"enable_thinking":true}}"#,
        )
        .unwrap();
    apply_eval_mode_chat_defaults(&state, &mut explicit);
    assert_eq!(explicit.temperature, Some(0.7));
    assert_eq!(
        explicit
            .chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("enable_thinking")),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn eval_mode_thinking_default_comes_from_model_profile() {
    let mut state = make_batch_test_state();
    state.eval_mode = true;
    state.model_defaults_profile = crate::config::ModelDefaultsProfile {
        eval_mode_default_thinking_enabled: true,
        ..crate::config::ModelDefaultsProfile::qwen3_5_4b()
    };
    let mut req: ChatCompletionRequest =
        serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();

    apply_eval_mode_chat_defaults(&state, &mut req);

    assert_eq!(
        req.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("enable_thinking")),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn batch_request_parses_minimal_shape() {
    let req = parse_batch_request(r#"{"prompts":[[{"role":"user","content":"hi"}]]}"#);
    assert_eq!(req.prompts.len(), 1);
    assert_eq!(req.prompts[0][0].role, "user");
    assert_eq!(req.prompts[0][0].content, "hi");
    assert!(req.n.is_none());
    assert!(req.seed.is_none());
    assert!(req.adapter.is_none());
    assert!(req.adapters.is_none());
}

#[test]
fn batch_request_parses_full_shape() {
    let req = parse_batch_request(
        r#"{
                "prompts":[
                    [{"role":"user","content":"a"}],
                    [{"role":"user","content":"b"}]
                ],
                "n":4,
                "temperature":0.7,
                "top_p":0.95,
                "top_k":40,
                "max_tokens":32,
                "stop":["\n\n"],
                "seed":1234,
                "adapter":"my-adapter",
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"auto"
            }"#,
    );
    assert_eq!(req.prompts.len(), 2);
    assert_eq!(req.n, Some(4));
    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.top_p, Some(0.95));
    assert_eq!(req.top_k, Some(40));
    assert_eq!(req.max_tokens, Some(32));
    assert_eq!(req.stop.as_deref(), Some(&["\n\n".to_string()][..]));
    assert_eq!(req.seed, Some(1234));
    assert_eq!(req.adapter.as_deref(), Some("my-adapter"));
    assert_eq!(req.tools.as_ref().unwrap().len(), 1);
    assert_eq!(req.tool_choice.as_ref().unwrap(), "auto");
}

#[tokio::test]
async fn batch_rejects_empty_prompts() {
    let (status, body) = batch_post(make_batch_test_state(), r#"{"prompts":[]}"#).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "batch_invalid_request");
}

#[tokio::test]
async fn batch_rejects_zero_n() {
    let (status, body) = batch_post(
        make_batch_test_state(),
        r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":0}"#,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "batch_invalid_request");
}

#[tokio::test]
async fn batch_rejects_too_many_outputs() {
    // 65 prompts * 1 = 65 > BATCH_MAX_TOTAL_OUTPUTS (64)
    let prompts: Vec<serde_json::Value> = (0..65)
        .map(|_| serde_json::json!([{"role":"user","content":"hi"}]))
        .collect();
    let req_body = serde_json::json!({"prompts": prompts}).to_string();
    let (status, body) = batch_post(make_batch_test_state(), &req_body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "batch_too_large");
}

#[tokio::test]
async fn batch_rejects_too_many_outputs_via_n_multiplier() {
    // 8 prompts * 9 = 72 > 64 — proves the cap counts the product, not just prompts.len().
    let prompts: Vec<serde_json::Value> = (0..8)
        .map(|_| serde_json::json!([{"role":"user","content":"hi"}]))
        .collect();
    let req_body = serde_json::json!({"prompts": prompts, "n": 9}).to_string();
    let (status, body) = batch_post(make_batch_test_state(), &req_body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "batch_too_large");
}

#[tokio::test]
async fn batch_rejects_adapter_and_adapters_together() {
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"hi"}]],
        "adapter": "single",
        "adapters": [{"name":"a","scale":1.0}]
    })
    .to_string();
    let (status, body) = batch_post(make_batch_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_compose_request");
}

#[tokio::test]
async fn batch_rejects_empty_adapters_list() {
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"hi"}]],
        "adapters": []
    })
    .to_string();
    let (status, body) = batch_post(make_batch_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_compose_request");
}

#[tokio::test]
async fn batch_rejects_oversized_adapters_list() {
    // 17 entries > MAX_COMPOSE_ADAPTERS (16) — caps audit MEDIUM §6 DoS.
    let adapters: Vec<serde_json::Value> = (0..17)
        .map(|_| serde_json::json!({"name": "a", "scale": 1.0}))
        .collect();
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"hi"}]],
        "adapters": adapters,
    })
    .to_string();
    let (status, body) = batch_post(make_batch_test_state(), &body).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_compose_request");
}

#[test]
fn batch_seed_derivation_is_distinct_per_output() {
    // Verifies the documented derivation: per-output seed =
    // base.wrapping_add(prompt_idx * n + completion_idx).
    let base: u64 = 42;
    let n_per = 3;
    let mut seeds = std::collections::HashSet::new();
    for prompt_idx in 0..2usize {
        for completion_idx in 0..n_per {
            let derived = base.wrapping_add((prompt_idx * n_per + completion_idx) as u64);
            assert!(
                seeds.insert(derived),
                "seed {derived} for ({prompt_idx},{completion_idx}) collides with an earlier output"
            );
        }
    }
    assert_eq!(seeds.len(), 2 * n_per);
}

#[test]
fn batch_prompt_groups_coalesce_duplicate_plain_prompts() {
    let req = parse_batch_request(
        r#"{
                "prompts": [
                    [{"role":"user","content":"same"}],
                    [{"role":"user","content":"different"}],
                    [{"role":"user","content":"same"}]
                ],
                "temperature": 0.7
            }"#,
    );

    let groups = batch_prompt_groups(&req.prompts);
    let grouped_indices: Vec<Vec<usize>> = groups
        .iter()
        .map(|group| group.prompt_indices.clone())
        .collect();
    assert_eq!(grouped_indices, vec![vec![0, 2], vec![1]]);
    assert_eq!(
        groups[0].messages.len(),
        1,
        "duplicate prompt group should store one synthesized message vector"
    );
}

#[tokio::test]
async fn batch_duplicate_prompt_grouping_preserves_response_order() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"same"}],
            [{"role":"user","content":"different"}],
            [{"role":"user","content":"same"}]
        ],
        "temperature": 0.7,
        "max_tokens": 2,
        "seed": 9
    })
    .to_string();

    let (status, body) = batch_post(state, &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let prompt_indices: Vec<u64> = body["completions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["prompt_index"].as_u64().unwrap())
        .collect();
    assert_eq!(
        prompt_indices,
        vec![0, 1, 2],
        "grouped duplicate prompts must not reorder the public batch response"
    );
}

#[test]
fn batch_deterministic_clone_gate_requires_explicit_greedy_multi_completion() {
    let greedy_multi = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":2,"temperature":0.0}"#,
    );
    assert!(batch_can_clone_deterministic_completions(&greedy_multi));
    assert!(batch_can_clone_identical_prompt_groups(&greedy_multi));

    let greedy_single = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":1,"temperature":0.0}"#,
    );
    assert!(!batch_can_clone_deterministic_completions(&greedy_single));
    assert!(batch_can_clone_identical_prompt_groups(&greedy_single));

    let sampled_multi = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":2,"temperature":0.7}"#,
    );
    assert!(!batch_can_clone_deterministic_completions(&sampled_multi));
    assert!(!batch_can_clone_identical_prompt_groups(&sampled_multi));

    let default_temperature =
        parse_batch_request(r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":2}"#);
    assert!(!batch_can_clone_deterministic_completions(
        &default_temperature
    ));
    assert!(!batch_can_clone_identical_prompt_groups(
        &default_temperature
    ));
}

#[test]
fn deterministic_completion_cache_key_accepts_replayable_sampling_only() {
    let state = make_batch_test_state();
    let prompt_tokens = vec![1, 2, 3];

    let unseeded_sampled = SamplingParams {
        temperature: 0.7,
        max_tokens: 4,
        ..Default::default()
    };
    assert!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &unseeded_sampled, false)
            .is_none(),
        "unseeded sampled decoding must stay uncached because it is intentionally random"
    );

    let seeded_sampled_1 = SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        max_tokens: 4,
        seed: Some(1),
        ..Default::default()
    };
    let seeded_sampled_2 = SamplingParams {
        seed: Some(2),
        ..seeded_sampled_1.clone()
    };
    let seeded_sampled_different_temperature = SamplingParams {
        temperature: 0.8,
        ..seeded_sampled_1.clone()
    };
    assert!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_sampled_1, false)
            .is_some(),
        "seeded sampled decoding is replayable and can use the completion cache"
    );
    assert_ne!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_sampled_1, false),
        deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_sampled_2, false),
        "sampled decoding must split cache entries by seed"
    );
    assert_ne!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_sampled_1, false),
        deterministic_completion_cache_key(
            &state,
            &prompt_tokens,
            &seeded_sampled_different_temperature,
            false
        ),
        "sampled decoding must split cache entries by temperature"
    );
    let seeded_full_distribution = SamplingParams {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        max_tokens: 4,
        seed: Some(1),
        ..SamplingParams::greedy()
    };
    let seeded_top_p_above_one = SamplingParams {
        top_p: 1.5,
        ..seeded_full_distribution.clone()
    };
    let seeded_top_p_zero = SamplingParams {
        top_p: 0.0,
        ..seeded_full_distribution.clone()
    };
    let seeded_top_p_negative = SamplingParams {
        top_p: -0.5,
        ..seeded_full_distribution.clone()
    };
    let seeded_top_k_disabled = SamplingParams {
        top_k: state.model_config.vocab_size as u32,
        ..seeded_full_distribution.clone()
    };
    assert_eq!(
        deterministic_completion_cache_key(
            &state,
            &prompt_tokens,
            &seeded_full_distribution,
            false
        ),
        deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_top_p_above_one, false),
        "top_p >= 1.0 disables nucleus filtering, so full-distribution seeded sampling should share completion-cache entries"
    );
    assert_eq!(
        deterministic_completion_cache_key(
            &state,
            &prompt_tokens,
            &seeded_full_distribution,
            false
        ),
        deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_top_p_zero, false),
        "top_p=0 disables nucleus filtering, so full-distribution seeded sampling should share completion-cache entries"
    );
    assert_eq!(
        deterministic_completion_cache_key(
            &state,
            &prompt_tokens,
            &seeded_full_distribution,
            false
        ),
        deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_top_p_negative, false),
        "negative top_p disables nucleus filtering, so full-distribution seeded sampling should share completion-cache entries"
    );
    assert_eq!(
        deterministic_completion_cache_key(
            &state,
            &prompt_tokens,
            &seeded_full_distribution,
            false
        ),
        deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_top_k_disabled, false),
        "top_k >= model vocab size is disabled, so full-distribution seeded sampling should share completion-cache entries"
    );

    let greedy_seed_1 = SamplingParams {
        temperature: 0.0,
        top_p: 0.8,
        top_k: 17,
        max_tokens: 4,
        seed: Some(1),
        ..Default::default()
    };
    let greedy_seed_2 = SamplingParams {
        temperature: 0.0,
        top_p: 0.95,
        top_k: 0,
        max_tokens: 4,
        seed: Some(2),
        ..Default::default()
    };

    assert_eq!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &greedy_seed_1, false),
        deterministic_completion_cache_key(&state, &prompt_tokens, &greedy_seed_2, false),
        "greedy decoding is seed/filter-independent, so seed/top-p/top-k must not split cache entries"
    );
    let top_k_one = SamplingParams {
        temperature: 0.7,
        top_p: 0.2,
        top_k: 1,
        max_tokens: 4,
        ..Default::default()
    };
    assert_eq!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &greedy_seed_1, false),
        deterministic_completion_cache_key(&state, &prompt_tokens, &top_k_one, false),
        "top_k=1 is effectively greedy, so seed/top-p/temperature must not split completion-cache entries"
    );
}

#[test]
fn deterministic_completion_cache_key_includes_min_p_and_penalties() {
    let state = make_batch_test_state();
    let prompt_tokens = vec![1, 2, 3];

    // Default carries Qwen3.5-thinking-general: min_p 0.0,
    // presence 1.5, frequency 0.0, repetition 1.0.
    let seeded_base = SamplingParams {
        temperature: 0.7,
        max_tokens: 4,
        seed: Some(1),
        ..Default::default()
    };
    let base_key = deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_base, false);
    assert!(base_key.is_some());

    for (label, changed) in [
        (
            "min_p",
            SamplingParams {
                min_p: 0.05,
                ..seeded_base.clone()
            },
        ),
        (
            "presence_penalty",
            SamplingParams {
                presence_penalty: 0.0,
                ..seeded_base.clone()
            },
        ),
        (
            "frequency_penalty",
            SamplingParams {
                frequency_penalty: 0.5,
                ..seeded_base.clone()
            },
        ),
        (
            "repetition_penalty",
            SamplingParams {
                repetition_penalty: 1.1,
                ..seeded_base.clone()
            },
        ),
    ] {
        assert_ne!(
            base_key,
            deterministic_completion_cache_key(&state, &prompt_tokens, &changed, false),
            "seeded sampling must split completion-cache entries by {label}"
        );
    }

    let min_p_negative = SamplingParams {
        min_p: -1.0,
        ..seeded_base.clone()
    };
    assert_eq!(
        base_key,
        deterministic_completion_cache_key(&state, &prompt_tokens, &min_p_negative, false),
        "min_p <= 0 is disabled, so disabled spellings should share completion-cache entries"
    );

    let presence_zero = SamplingParams {
        presence_penalty: 0.0,
        ..seeded_base.clone()
    };
    let presence_negative_zero = SamplingParams {
        presence_penalty: -0.0,
        ..seeded_base.clone()
    };
    assert_eq!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &presence_zero, false),
        deterministic_completion_cache_key(&state, &prompt_tokens, &presence_negative_zero, false),
        "-0.0 is the same no-op presence penalty as 0.0 and should share completion-cache entries"
    );

    let greedy_default_penalties = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let greedy_active_penalties = SamplingParams {
        temperature: 0.0,
        min_p: 0.2,
        presence_penalty: 0.4,
        frequency_penalty: 0.6,
        repetition_penalty: 1.3,
        max_tokens: 4,
        ..Default::default()
    };
    assert_eq!(
        deterministic_completion_cache_key(
            &state,
            &prompt_tokens,
            &greedy_default_penalties,
            false
        ),
        deterministic_completion_cache_key(&state, &prompt_tokens, &greedy_active_penalties, false),
        "greedy decoding short-circuits min_p and penalties, so they must not split cache entries"
    );
}

#[test]
fn deterministic_chat_cache_keys_normalize_omitted_min_p_and_penalties() {
    // Path A keys from resolved SamplingParams; path B keys from raw
    // request fields via requested_or_default_*. Both must agree on
    // the Qwen3.5-thinking-general defaults (presence 1.5,
    // repetition 1.0, frequency 0.0, min_p 0.0) or seeded batch and
    // chat requests stop sharing cache entries.
    let chat_omitted = parse_request(
        r#"{"messages":[{"role":"user","content":"penalty defaults"}],"n":2,"temperature":0.7,"max_tokens":4,"seed":9}"#,
    );
    let chat_key = deterministic_chat_choices_cache_key(
        &chat_omitted,
        2,
        &sampling_params_for_chat_request(&chat_omitted),
    )
    .unwrap()
    .expect("seeded chat choices request should be cacheable");

    let batch_omitted = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.7,"max_tokens":4,"seed":9}"#,
    );
    let batch_omitted_key =
        deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
            &batch_omitted,
            usize::MAX,
            false,
        )
        .unwrap()
        .expect("seeded single-prompt batch should be cacheable");
    assert_eq!(
        chat_key, batch_omitted_key,
        "request-side defaults must match the resolved sampling defaults across key paths"
    );

    let batch_explicit_defaults = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.7,"max_tokens":4,"seed":9,"min_p":0.0,"presence_penalty":1.5,"frequency_penalty":0.0,"repetition_penalty":1.0}"#,
    );
    assert_eq!(
        Some(&batch_omitted_key),
        deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
            &batch_explicit_defaults,
            usize::MAX,
            false,
        )
        .unwrap()
        .as_ref(),
        "explicitly spelling the default min_p/penalties should not split cache entries"
    );

    let batch_min_p_disabled = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.7,"max_tokens":4,"seed":9,"min_p":-1.0}"#,
    );
    assert_eq!(
        Some(&batch_omitted_key),
        deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
            &batch_min_p_disabled,
            usize::MAX,
            false,
        )
        .unwrap()
        .as_ref(),
        "min_p <= 0 is disabled and should share cache entries with the omitted spelling"
    );

    for (label, body) in [
        (
            "min_p",
            r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.7,"max_tokens":4,"seed":9,"min_p":0.05}"#,
        ),
        (
            "presence_penalty",
            r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.7,"max_tokens":4,"seed":9,"presence_penalty":0.0}"#,
        ),
        (
            "frequency_penalty",
            r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.7,"max_tokens":4,"seed":9,"frequency_penalty":0.5}"#,
        ),
        (
            "repetition_penalty",
            r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.7,"max_tokens":4,"seed":9,"repetition_penalty":1.1}"#,
        ),
    ] {
        let changed = parse_batch_request(body);
        assert_ne!(
            Some(&batch_omitted_key),
            deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
                &changed,
                usize::MAX,
                false,
            )
            .unwrap()
            .as_ref(),
            "seeded requests must split cache entries by {label}"
        );
    }

    // Greedy requests short-circuit min_p and penalties entirely.
    let greedy_omitted = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );
    let greedy_active = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"penalty defaults"}]],"n":2,"temperature":0.0,"max_tokens":4,"min_p":0.2,"presence_penalty":0.4,"frequency_penalty":0.6,"repetition_penalty":1.3}"#,
    );
    assert_eq!(
        deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
            &greedy_omitted,
            usize::MAX,
            false,
        )
        .unwrap(),
        deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
            &greedy_active,
            usize::MAX,
            false,
        )
        .unwrap(),
        "greedy decoding ignores min_p and penalties, so they must not split cache entries"
    );
}

#[tokio::test]
async fn chat_seeded_request_with_different_presence_penalty_misses_cache() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"presence penalty cache split"}],
        "temperature": 0.7,
        "max_tokens": 4,
        "seed": 41
    })
    .to_string();
    let presence_zero_body = serde_json::json!({
        "messages": [{"role":"user","content":"presence penalty cache split"}],
        "temperature": 0.7,
        "max_tokens": 4,
        "seed": 41,
        "presence_penalty": 0.0
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);

    let (status_repeat, repeat) = chat_post(state.clone(), &body).await;
    assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "identical seeded repeat should hit the deterministic cache"
    );

    let (status_changed, changed) = chat_post(state.clone(), &presence_zero_body).await;
    assert_eq!(status_changed, axum::http::StatusCode::OK, "{changed}");
    assert!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed)
            > generated_after_first,
        "turning the default presence penalty off must miss the seeded cache and regenerate"
    );
}

#[test]
fn deterministic_chat_request_cache_key_ignores_stream_flag() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let non_streaming = parse_request(
        r#"{"messages":[{"role":"user","content":"same cached chat"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let streaming = parse_request(
        r#"{"messages":[{"role":"user","content":"same cached chat"}],"temperature":0.0,"max_tokens":4,"stream":true}"#,
    );

    assert_eq!(
        deterministic_chat_request_cache_key(&non_streaming, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&streaming, &sampling).unwrap(),
        "streaming and non-streaming deterministic chat requests share the same cached payload"
    );
}

#[test]
fn ignore_eos_is_typed_forwarded_and_cache_keyed() {
    let ordinary = parse_request(
        r#"{"messages":[{"role":"user","content":"same eos policy"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let ignored = parse_request(
        r#"{"messages":[{"role":"user","content":"same eos policy"}],"temperature":0.0,"max_tokens":4,"ignore_eos":true}"#,
    );
    let ordinary_sampling = sampling_params_for_chat_request(&ordinary);
    let ignored_sampling = sampling_params_for_chat_request(&ignored);
    assert!(!ordinary.ignore_eos);
    assert!(!ordinary_sampling.ignore_eos);
    assert!(ignored.ignore_eos);
    assert!(ignored_sampling.ignore_eos);
    assert_ne!(
        deterministic_chat_request_cache_key(&ordinary, &ordinary_sampling).unwrap(),
        deterministic_chat_request_cache_key(&ignored, &ignored_sampling).unwrap(),
        "different EOS policies must not share deterministic completions"
    );

    let batch: BatchCompletionRequest = serde_json::from_str(
            r#"{"prompts":[[{"role":"user","content":"batch eos policy"}]],"max_tokens":4,"ignore_eos":true}"#,
        )
        .unwrap();
    assert!(batch.ignore_eos);
    assert!(batch_request_sampling_for_cache_key(&batch, None).ignore_eos);
}

#[test]
fn deterministic_chat_request_cache_key_skips_multi_choice_requests() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let single_choice = parse_request(
        r#"{"messages":[{"role":"user","content":"same cached chat"}],"temperature":0.0,"max_tokens":4,"n":1}"#,
    );
    let multi_choice = parse_request(
        r#"{"messages":[{"role":"user","content":"same cached chat"}],"temperature":0.0,"max_tokens":4,"n":2}"#,
    );

    assert!(
        deterministic_chat_request_cache_key(&single_choice, &sampling)
            .unwrap()
            .is_some(),
        "n=1 should keep the single-choice request cache path"
    );
    assert!(
        deterministic_chat_request_cache_key(&multi_choice, &sampling)
            .unwrap()
            .is_none(),
        "the single-choice request cache value cannot represent top-level n>1 responses"
    );
}

#[test]
fn deterministic_chat_choices_cache_key_normalizes_replayable_multi_choice_requests() {
    let greedy_a = parse_request(
        r#"{"messages":[{"role":"user","content":"same greedy choices"}],"n":4,"temperature":0.0,"top_p":0.8,"top_k":17,"max_tokens":4,"seed":1}"#,
    );
    let greedy_b = parse_request(
        r#"{"messages":[{"role":"user","content":"same greedy choices"}],"n":4,"temperature":0.0,"top_p":0.95,"top_k":0,"max_tokens":4,"seed":2}"#,
    );
    let greedy_sampling_a = SamplingParams {
        temperature: 0.0,
        top_p: 0.8,
        top_k: 17,
        max_tokens: 4,
        seed: Some(1),
        ..Default::default()
    };
    let greedy_sampling_b = SamplingParams {
        temperature: 0.0,
        top_p: 0.95,
        top_k: 0,
        max_tokens: 4,
        seed: Some(2),
        ..Default::default()
    };

    assert_eq!(
        deterministic_chat_choices_cache_key(&greedy_a, 4, &greedy_sampling_a).unwrap(),
        deterministic_chat_choices_cache_key(&greedy_b, 4, &greedy_sampling_b).unwrap(),
        "greedy chat n>1 cache keys should ignore seed/top-p/top-k"
    );
    assert_ne!(
        deterministic_chat_choices_cache_key(&greedy_a, 4, &greedy_sampling_a).unwrap(),
        deterministic_chat_choices_cache_key(&greedy_a, 3, &greedy_sampling_a).unwrap(),
        "different n values must not share top-level chat choices cache entries"
    );
    let top_k_one = parse_request(
        r#"{"messages":[{"role":"user","content":"same greedy choices"}],"n":4,"temperature":0.7,"top_p":0.2,"top_k":1,"max_tokens":4}"#,
    );
    let top_k_one_sampling = SamplingParams {
        temperature: 0.7,
        top_p: 0.2,
        top_k: 1,
        max_tokens: 4,
        ..Default::default()
    };
    assert_eq!(
        deterministic_chat_choices_cache_key(&greedy_a, 4, &greedy_sampling_a).unwrap(),
        deterministic_chat_choices_cache_key(&top_k_one, 4, &top_k_one_sampling).unwrap(),
        "top_k=1 is effectively greedy, so it should share chat choices cache entries with temperature=0"
    );

    let sampled_a = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled choices"}],"n":4,"temperature":0.7,"top_p":0.9,"max_tokens":4,"seed":1}"#,
    );
    let sampled_b = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled choices"}],"n":4,"temperature":0.7,"top_p":0.9,"max_tokens":4,"seed":2}"#,
    );
    let sampled_unseeded = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled choices"}],"n":4,"temperature":0.7,"top_p":0.9,"max_tokens":4}"#,
    );
    let sampled_sampling_a = SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 4,
        seed: Some(1),
        ..Default::default()
    };
    let sampled_sampling_b = SamplingParams {
        seed: Some(2),
        ..sampled_sampling_a.clone()
    };
    let sampled_sampling_unseeded = SamplingParams {
        seed: None,
        ..sampled_sampling_a.clone()
    };

    assert_ne!(
        deterministic_chat_choices_cache_key(&sampled_a, 4, &sampled_sampling_a).unwrap(),
        deterministic_chat_choices_cache_key(&sampled_b, 4, &sampled_sampling_b).unwrap(),
        "seeded sampled chat n>1 cache keys should split by base seed"
    );
    let sampled_full_distribution = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled choices full top p"}],"n":4,"temperature":0.7,"top_p":1.0,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_p_above_one = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled choices full top p"}],"n":4,"temperature":0.7,"top_p":1.5,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_p_zero = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled choices full top p"}],"n":4,"temperature":0.7,"top_p":0.0,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_k_disabled = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled choices full top p"}],"n":4,"temperature":0.7,"top_p":1.0,"top_k":248320,"max_tokens":4,"seed":1}"#,
    );
    let sampled_full_distribution_sampling = SamplingParams {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        max_tokens: 4,
        seed: Some(1),
        ..SamplingParams::greedy()
    };
    let sampled_top_p_above_one_sampling = SamplingParams {
        top_p: 1.5,
        ..sampled_full_distribution_sampling.clone()
    };
    let sampled_top_p_zero_sampling = SamplingParams {
        top_p: 0.0,
        ..sampled_full_distribution_sampling.clone()
    };
    let sampled_top_k_disabled_sampling = SamplingParams {
        top_k: ModelConfig::qwen3_5_4b().vocab_size as u32,
        ..sampled_full_distribution_sampling.clone()
    };
    let model_vocab_size = ModelConfig::qwen3_5_4b().vocab_size;
    assert_eq!(
        deterministic_chat_choices_cache_key(
            &sampled_full_distribution,
            4,
            &sampled_full_distribution_sampling
        )
        .unwrap(),
        deterministic_chat_choices_cache_key(
            &sampled_top_p_above_one,
            4,
            &sampled_top_p_above_one_sampling
        )
        .unwrap(),
        "top_p >= 1.0 disables nucleus filtering, so full-distribution seeded chat choices should share cache entries"
    );
    assert_eq!(
        deterministic_chat_choices_cache_key_with_vocab_size(
            &sampled_full_distribution,
            4,
            &sampled_full_distribution_sampling,
            model_vocab_size
        )
        .unwrap(),
        deterministic_chat_choices_cache_key_with_vocab_size(
            &sampled_top_p_zero,
            4,
            &sampled_top_p_zero_sampling,
            model_vocab_size
        )
        .unwrap(),
        "top_p=0 disables nucleus filtering, so full-distribution seeded chat choices should share cache entries"
    );
    assert_eq!(
        deterministic_chat_choices_cache_key_with_vocab_size(
            &sampled_full_distribution,
            4,
            &sampled_full_distribution_sampling,
            model_vocab_size
        )
        .unwrap(),
        deterministic_chat_choices_cache_key_with_vocab_size(
            &sampled_top_k_disabled,
            4,
            &sampled_top_k_disabled_sampling,
            model_vocab_size
        )
        .unwrap(),
        "top_k >= model vocab size is disabled, so full-distribution seeded chat choices should share cache entries"
    );
    assert!(
        deterministic_chat_choices_cache_key(&sampled_unseeded, 4, &sampled_sampling_unseeded)
            .unwrap()
            .is_none(),
        "unseeded sampled chat n>1 requests are intentionally random and must not be cached"
    );
}

#[test]
fn deterministic_chat_request_cache_key_normalizes_equivalent_sampling_fields() {
    let greedy_a = parse_request(
        r#"{"messages":[{"role":"user","content":"same greedy chat"}],"temperature":0.0,"top_p":0.8,"top_k":17,"max_tokens":4,"seed":1}"#,
    );
    let greedy_b = parse_request(
        r#"{"messages":[{"role":"user","content":"same greedy chat"}],"temperature":0.0,"top_p":0.95,"top_k":0,"max_tokens":4,"seed":2}"#,
    );
    let greedy_sampling_a = SamplingParams {
        temperature: 0.0,
        top_p: 0.8,
        top_k: 17,
        max_tokens: 4,
        seed: Some(1),
        ..Default::default()
    };
    let greedy_sampling_b = SamplingParams {
        temperature: 0.0,
        top_p: 0.95,
        top_k: 0,
        max_tokens: 4,
        seed: Some(2),
        ..Default::default()
    };

    assert_eq!(
        deterministic_chat_request_cache_key(&greedy_a, &greedy_sampling_a).unwrap(),
        deterministic_chat_request_cache_key(&greedy_b, &greedy_sampling_b).unwrap(),
        "greedy request-cache keys should ignore seed/top-p/top-k"
    );
    let top_k_one = parse_request(
        r#"{"messages":[{"role":"user","content":"same greedy chat"}],"temperature":0.7,"top_p":0.2,"top_k":1,"max_tokens":4}"#,
    );
    let top_k_one_sampling = SamplingParams {
        temperature: 0.7,
        top_p: 0.2,
        top_k: 1,
        max_tokens: 4,
        ..Default::default()
    };

    assert_eq!(
        deterministic_chat_request_cache_key(&greedy_a, &greedy_sampling_a).unwrap(),
        deterministic_chat_request_cache_key(&top_k_one, &top_k_one_sampling).unwrap(),
        "top_k=1 request-cache keys should normalize to greedy"
    );

    let zero_a = parse_request(
        r#"{"messages":[{"role":"user","content":"same zero chat"}],"temperature":0.7,"top_p":0.8,"top_k":17,"max_tokens":0,"stop":["x"],"seed":1}"#,
    );
    let zero_b = parse_request(
        r#"{"messages":[{"role":"user","content":"same zero chat"}],"temperature":0.2,"top_p":0.95,"top_k":0,"max_tokens":0,"stop":["y"],"seed":2}"#,
    );
    let zero_sampling_a = SamplingParams {
        temperature: 0.7,
        top_p: 0.8,
        top_k: 17,
        max_tokens: 0,
        stop: vec!["x".to_string()],
        seed: Some(1),
        ..Default::default()
    };
    let zero_sampling_b = SamplingParams {
        temperature: 0.2,
        top_p: 0.95,
        top_k: 0,
        max_tokens: 0,
        stop: vec!["y".to_string()],
        seed: Some(2),
        ..Default::default()
    };

    assert_eq!(
        deterministic_chat_request_cache_key(&zero_a, &zero_sampling_a).unwrap(),
        deterministic_chat_request_cache_key(&zero_b, &zero_sampling_b).unwrap(),
        "max_tokens=0 request-cache keys should ignore generation-only sampling fields"
    );

    let sampled_a = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled chat"}],"temperature":0.7,"max_tokens":4,"seed":1}"#,
    );
    let sampled_b = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled chat"}],"temperature":0.7,"max_tokens":4,"seed":2}"#,
    );
    let sampled_sampling_a = SamplingParams {
        temperature: 0.7,
        max_tokens: 4,
        seed: Some(1),
        ..Default::default()
    };
    let sampled_sampling_b = SamplingParams {
        temperature: 0.7,
        max_tokens: 4,
        seed: Some(2),
        ..Default::default()
    };

    assert_ne!(
        deterministic_chat_request_cache_key(&sampled_a, &sampled_sampling_a).unwrap(),
        deterministic_chat_request_cache_key(&sampled_b, &sampled_sampling_b).unwrap(),
        "seeded sampled request-cache keys must still split by seed"
    );
    // top_k=0 explicit in the JSON because the kiln default is 20 —
    // this test is specifically validating top_k=disabled vs
    // top_k>=vocab equivalence, so we have to opt out of the model
    // default to get the "disabled" semantics it asserts on.
    let sampled_full_distribution = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled chat full top p"}],"temperature":0.7,"top_p":1.0,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_p_above_one = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled chat full top p"}],"temperature":0.7,"top_p":1.5,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_p_zero = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled chat full top p"}],"temperature":0.7,"top_p":0.0,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_k_disabled = parse_request(
        r#"{"messages":[{"role":"user","content":"sampled chat full top p"}],"temperature":0.7,"top_p":1.0,"top_k":248320,"max_tokens":4,"seed":1}"#,
    );
    let sampled_full_distribution_sampling = SamplingParams {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        max_tokens: 4,
        seed: Some(1),
        ..SamplingParams::greedy()
    };
    let sampled_top_p_above_one_sampling = SamplingParams {
        top_p: 1.5,
        ..sampled_full_distribution_sampling.clone()
    };
    let sampled_top_p_zero_sampling = SamplingParams {
        top_p: 0.0,
        ..sampled_full_distribution_sampling.clone()
    };
    let sampled_top_k_disabled_sampling = SamplingParams {
        top_k: ModelConfig::qwen3_5_4b().vocab_size as u32,
        ..sampled_full_distribution_sampling.clone()
    };
    let model_vocab_size = ModelConfig::qwen3_5_4b().vocab_size;
    assert_eq!(
        deterministic_chat_request_cache_key(
            &sampled_full_distribution,
            &sampled_full_distribution_sampling
        )
        .unwrap(),
        deterministic_chat_request_cache_key(
            &sampled_top_p_above_one,
            &sampled_top_p_above_one_sampling
        )
        .unwrap(),
        "top_p >= 1.0 disables nucleus filtering, so full-distribution seeded chat requests should share cache entries"
    );
    assert_eq!(
        deterministic_chat_request_cache_key_with_vocab_size(
            &sampled_full_distribution,
            &sampled_full_distribution_sampling,
            model_vocab_size
        )
        .unwrap(),
        deterministic_chat_request_cache_key_with_vocab_size(
            &sampled_top_p_zero,
            &sampled_top_p_zero_sampling,
            model_vocab_size
        )
        .unwrap(),
        "top_p=0 disables nucleus filtering, so full-distribution seeded chat requests should share cache entries"
    );
    assert_eq!(
        deterministic_chat_request_cache_key_with_vocab_size(
            &sampled_full_distribution,
            &sampled_full_distribution_sampling,
            model_vocab_size
        )
        .unwrap(),
        deterministic_chat_request_cache_key_with_vocab_size(
            &sampled_top_k_disabled,
            &sampled_top_k_disabled_sampling,
            model_vocab_size
        )
        .unwrap(),
        "top_k >= model vocab size is disabled, so full-distribution seeded chat requests should share cache entries"
    );
}

#[test]
fn deterministic_chat_request_cache_key_normalizes_max_completion_tokens_alias() {
    let max_tokens = parse_request(
        r#"{"messages":[{"role":"user","content":"same max token alias"}],"temperature":0.0,"max_tokens":4,"seed":1}"#,
    );
    let alias = parse_request(
        r#"{"messages":[{"role":"user","content":"same max token alias"}],"temperature":0.0,"max_completion_tokens":4,"seed":2}"#,
    );
    let max_tokens_sampling = SamplingParams {
        temperature: max_tokens.temperature.unwrap_or(1.0),
        max_tokens: chat_request_max_tokens(&max_tokens),
        seed: max_tokens.seed,
        ..Default::default()
    };
    let alias_sampling = SamplingParams {
        temperature: alias.temperature.unwrap_or(1.0),
        max_tokens: chat_request_max_tokens(&alias),
        seed: alias.seed,
        ..Default::default()
    };

    assert_eq!(
        deterministic_chat_request_cache_key(&max_tokens, &max_tokens_sampling).unwrap(),
        deterministic_chat_request_cache_key(&alias, &alias_sampling).unwrap(),
        "max_completion_tokens should share deterministic request-cache entries with max_tokens"
    );
}

#[test]
fn deterministic_chat_request_cache_key_ignores_default_openai_option_fields() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let plain = parse_request(
        r#"{"messages":[{"role":"user","content":"same default options"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let defaults = parse_request(
        r#"{"messages":[{"role":"user","content":"same default options"}],"temperature":0.0,"max_tokens":4,"n":1,"response_format":{"type":"text"},"parallel_tool_calls":true,"user":"client-a","metadata":{"trace_id":"ignored"},"store":false,"service_tier":"auto","logprobs":false,"top_logprobs":0,"frequency_penalty":0.0,"presence_penalty":0.0,"stream_options":{"include_usage":false}}"#,
    );

    assert_eq!(
        deterministic_chat_request_cache_key(&plain, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&defaults, &sampling).unwrap(),
        "default OpenAI option fields should not split deterministic chat request-cache entries"
    );
}

#[test]
fn deterministic_chat_request_cache_key_includes_chat_template_kwargs() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let default_req = parse_request(
        r#"{"messages":[{"role":"user","content":"same template kwargs"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let empty_kwargs = parse_request(
        r#"{"messages":[{"role":"user","content":"same template kwargs"}],"temperature":0.0,"max_tokens":4,"chat_template_kwargs":{}}"#,
    );
    let no_think = parse_request(
        r#"{"messages":[{"role":"user","content":"same template kwargs"}],"temperature":0.0,"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false}}"#,
    );

    assert_eq!(
        deterministic_chat_request_cache_key(&default_req, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&empty_kwargs, &sampling).unwrap(),
        "empty chat_template_kwargs should normalize to omitted"
    );
    assert_ne!(
        deterministic_chat_request_cache_key(&default_req, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&no_think, &sampling).unwrap(),
        "template kwargs change rendered prompts and must split request-cache entries"
    );
}

#[test]
fn deterministic_chat_request_cache_key_keeps_tool_template_kwargs_explicit() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let omitted_kwargs = parse_request(
        r#"{"messages":[{"role":"user","content":"same tool default kwargs"}],"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto"}"#,
    );
    let explicit_no_think = parse_request(
        r#"{"messages":[{"role":"user","content":"same tool default kwargs"}],"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":false}}"#,
    );
    let explicit_think = parse_request(
        r#"{"messages":[{"role":"user","content":"same tool default kwargs"}],"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":true}}"#,
    );

    assert_ne!(
        deterministic_chat_request_cache_key(&omitted_kwargs, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&explicit_no_think, &sampling).unwrap(),
        "tool requests must not reinterpret omitted kwargs as enable_thinking=false"
    );
    assert_ne!(
        deterministic_chat_request_cache_key(&explicit_no_think, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&explicit_think, &sampling).unwrap(),
        "explicit enable_thinking values must keep separate cache entries"
    );
}

#[test]
fn deterministic_chat_request_cache_key_normalizes_empty_tools() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let omitted_tools = parse_request(
        r#"{"messages":[{"role":"user","content":"same no-op tools"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let empty_tools = parse_request(
        r#"{"messages":[{"role":"user","content":"same no-op tools"}],"tools":[],"temperature":0.0,"max_tokens":4}"#,
    );
    let real_tool = parse_request(
        r#"{"messages":[{"role":"user","content":"same no-op tools"}],"tools":[{"type":"function","function":{"name":"Search","parameters":{"type":"object","properties":{}}}}],"temperature":0.0,"max_tokens":4}"#,
    );

    assert_eq!(
        deterministic_chat_request_cache_key(&omitted_tools, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&empty_tools, &sampling).unwrap(),
        "empty tools should not split request-cache entries from omitted tools"
    );
    assert_ne!(
        deterministic_chat_request_cache_key(&omitted_tools, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&real_tool, &sampling).unwrap(),
        "non-empty tools must still split request-cache entries"
    );
    assert_eq!(
        normalized_tools_option_for_synthetic_request(empty_tools.tools.as_deref()),
        None,
        "synthetic fanout should drop empty tools before cloning"
    );
    assert_eq!(
        normalized_tool_choice_option_for_synthetic_request(
            empty_tools.tools.as_deref(),
            empty_tools.tool_choice.as_ref(),
        ),
        None,
        "synthetic fanout should drop absent tool_choice with empty tools"
    );
}

#[test]
fn deterministic_chat_request_cache_key_normalizes_no_tool_auto_choice() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let omitted_choice = parse_request(
        r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let auto_without_tools = parse_request(
        r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"tool_choice":"auto","temperature":0.0,"max_tokens":4}"#,
    );
    let none_with_empty_tools = parse_request(
        r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"tools":[],"tool_choice":"none","temperature":0.0,"max_tokens":4}"#,
    );
    let required_without_tools = parse_request(
        r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"tool_choice":"required","temperature":0.0,"max_tokens":4}"#,
    );
    let real_tool_auto = parse_request(
        r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"tools":[{"type":"function","function":{"name":"Search","parameters":{"type":"object","properties":{}}}}],"tool_choice":"auto","temperature":0.0,"max_tokens":4}"#,
    );

    let omitted_key = deterministic_chat_request_cache_key(&omitted_choice, &sampling).unwrap();
    assert_eq!(
        omitted_key,
        deterministic_chat_request_cache_key(&auto_without_tools, &sampling).unwrap(),
        "tool_choice=auto without tools should not split request-cache entries"
    );
    assert_eq!(
        omitted_key,
        deterministic_chat_request_cache_key(&none_with_empty_tools, &sampling).unwrap(),
        "tool_choice=none with empty tools should not split request-cache entries"
    );
    assert_ne!(
        omitted_key,
        deterministic_chat_request_cache_key(&required_without_tools, &sampling).unwrap(),
        "tool_choice=required without tools stays distinct because it is not a no-op choice"
    );
    assert_ne!(
        omitted_key,
        deterministic_chat_request_cache_key(&real_tool_auto, &sampling).unwrap(),
        "non-empty tools must still split request-cache entries"
    );
    assert_eq!(
        normalized_tool_choice_option_for_synthetic_request(
            auto_without_tools.tools.as_deref(),
            auto_without_tools.tool_choice.as_ref(),
        ),
        None,
        "synthetic fanout should drop no-tool tool_choice=auto before cloning"
    );
    assert_eq!(
        normalized_tool_choice_option_for_synthetic_request(
            none_with_empty_tools.tools.as_deref(),
            none_with_empty_tools.tool_choice.as_ref(),
        ),
        None,
        "synthetic fanout should drop no-tool tool_choice=none before cloning"
    );
    assert_eq!(
        normalized_tool_choice_option_for_synthetic_request(
            required_without_tools.tools.as_deref(),
            required_without_tools.tool_choice.as_ref(),
        ),
        required_without_tools.tool_choice,
        "synthetic fanout must keep non-no-op required tool_choice"
    );
}

#[test]
fn deterministic_chat_request_cache_key_ignores_input_reasoning_content() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let without_reasoning = parse_request(
        r#"{"messages":[{"role":"user","content":"same rendered prompt"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let with_reasoning = parse_request(
        r#"{"messages":[{"role":"user","content":"same rendered prompt","reasoning_content":"ignored by renderer"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let with_name = parse_request(
        r#"{"messages":[{"role":"user","content":"same rendered prompt","name":"distinct"}],"temperature":0.0,"max_tokens":4}"#,
    );

    assert_eq!(
        deterministic_chat_request_cache_key(&without_reasoning, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&with_reasoning, &sampling).unwrap(),
        "input reasoning_content is not rendered and should not split request-cache entries"
    );
    assert_ne!(
        deterministic_chat_request_cache_key(&without_reasoning, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&with_name, &sampling).unwrap(),
        "fields propagated to the renderer must still split request-cache entries"
    );
}

#[test]
fn deterministic_cache_keys_include_reasoning_fold_mode() {
    let state = make_qwen_template_test_state();
    let prompt_tokens = vec![1, 2, 3];
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let plain = parse_request(
        r#"{"messages":[{"role":"user","content":"same prompt"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let folded = parse_request(
        r#"{"messages":[{"role":"user","content":"same prompt"}],"fold_reasoning_into_content":true,"temperature":0.0,"max_tokens":4}"#,
    );

    assert_ne!(
        deterministic_chat_request_cache_key(&plain, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&folded, &sampling).unwrap(),
        "request-cache entries must split when content folding changes response shape"
    );
    assert_ne!(
        deterministic_chat_choices_cache_key(&plain, 2, &sampling).unwrap(),
        deterministic_chat_choices_cache_key(&folded, 2, &sampling).unwrap(),
        "choices-cache entries must split when content folding changes response shape"
    );
    assert_ne!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &sampling, false),
        deterministic_completion_cache_key(&state, &prompt_tokens, &sampling, true),
        "completion-cache entries must split when content folding changes response shape"
    );

    let batch = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch prompt"}]],"temperature":0.0,"max_tokens":4}"#,
    );
    assert_ne!(
        deterministic_batch_cache_key_with_vocab_size_and_fold(
            &batch,
            1,
            usize::MAX,
            false,
            batch_token_budget_without_server_default(&batch),
        ),
        deterministic_batch_cache_key_with_vocab_size_and_fold(
            &batch,
            1,
            usize::MAX,
            true,
            batch_token_budget_without_server_default(&batch),
        ),
        "batch-cache entries must split when server folding changes response shape"
    );
}

#[test]
fn deterministic_chat_request_cache_key_normalizes_text_content_parts() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let plain = parse_request(
        r#"{"messages":[{"role":"user","content":"same text parts"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let parts = parse_request(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"same "},{"type":"text","text":"text parts"}]}],"temperature":0.0,"max_tokens":4}"#,
    );

    assert_eq!(
        deterministic_chat_request_cache_key(&plain, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&parts, &sampling).unwrap(),
        "equivalent OpenAI text content parts should not split request-cache entries"
    );
}

#[test]
fn deterministic_chat_request_cache_key_ignores_non_text_content_parts() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let plain = parse_request(
        r#"{"messages":[{"role":"user","content":"same visible text"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let parts = parse_request(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"same visible "},{"type":"image_url","image_url":{"url":"https://example.invalid/ignored.png"}},{"type":"input_audio","input_audio":{"data":"ignored","format":"wav"}},{"type":"text","text":"text"}]}],"temperature":0.0,"max_tokens":4}"#,
    );

    assert_eq!(
        deterministic_chat_request_cache_key(&plain, &sampling).unwrap(),
        deterministic_chat_request_cache_key(&parts, &sampling).unwrap(),
        "non-text content parts are ignored by the text-only deserializer and should not split request-cache entries"
    );
}

#[test]
fn deterministic_chat_request_cache_key_normalizes_empty_message_tool_calls() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let omitted = parse_request(
        r#"{"messages":[{"role":"user","content":"same empty message tool calls"},{"role":"assistant","content":"ok"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let empty_tool_calls = parse_request(
        r#"{"messages":[{"role":"user","content":"same empty message tool calls"},{"role":"assistant","content":"ok","tool_calls":[]},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let non_empty_tool_calls = parse_request(
        r#"{"messages":[{"role":"user","content":"same empty message tool calls"},{"role":"assistant","content":"ok","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"{}"}}]},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
    );

    let omitted_key = deterministic_chat_request_cache_key(&omitted, &sampling).unwrap();
    assert_eq!(
        omitted_key,
        deterministic_chat_request_cache_key(&empty_tool_calls, &sampling).unwrap(),
        "empty message tool_calls should not split request-cache entries"
    );
    assert_ne!(
        omitted_key,
        deterministic_chat_request_cache_key(&non_empty_tool_calls, &sampling).unwrap(),
        "non-empty message tool_calls must still split request-cache entries"
    );
}

#[test]
fn deterministic_chat_request_cache_key_normalizes_tool_call_argument_json_strings() {
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        ..Default::default()
    };
    let compact = parse_request(
        r#"{"messages":[{"role":"user","content":"normalize tool args"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"{\"query\":\"cache\",\"limit\":2}"}}]},{"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let whitespace = parse_request(
        r#"{"messages":[{"role":"user","content":"normalize tool args"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"{ \"limit\" : 2, \"query\" : \"cache\" }"}}]},{"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let structured = parse_request(
        r#"{"messages":[{"role":"user","content":"normalize tool args"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":{"limit":2,"query":"cache"}}}]},{"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
    );
    let non_json = parse_request(
        r#"{"messages":[{"role":"user","content":"normalize tool args"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"not-json"}}]},{"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
    );

    let compact_key = deterministic_chat_request_cache_key(&compact, &sampling).unwrap();
    assert_eq!(
        compact_key,
        deterministic_chat_request_cache_key(&whitespace, &sampling).unwrap(),
        "JSON-equivalent tool_call argument strings should not split request-cache entries"
    );
    assert_eq!(
        compact_key,
        deterministic_chat_request_cache_key(&structured, &sampling).unwrap(),
        "parsed and structured tool_call arguments render equivalently"
    );
    assert_ne!(
        compact_key,
        deterministic_chat_request_cache_key(&non_json, &sampling).unwrap(),
        "non-JSON argument strings must stay distinct"
    );
}

#[test]
fn deterministic_cache_keys_normalize_stop_sequence_sets() {
    let state = make_batch_test_state();
    let prompt_tokens = vec![1, 2, 3];
    let stop_a = vec![
        "omega".to_string(),
        "alpha".to_string(),
        "omega".to_string(),
    ];
    let stop_b = vec!["alpha".to_string(), "omega".to_string()];
    let completion_a = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        stop: stop_a.clone(),
        ..Default::default()
    };
    let completion_b = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        stop: stop_b.clone(),
        ..Default::default()
    };
    assert_eq!(
        deterministic_completion_cache_key(&state, &prompt_tokens, &completion_a, false),
        deterministic_completion_cache_key(&state, &prompt_tokens, &completion_b, false),
        "stop sequence order and duplicates should not split completion-cache entries"
    );

    let chat_a = parse_request(
        r#"{"messages":[{"role":"user","content":"same stop set"}],"temperature":0.0,"max_tokens":4,"stop":["omega","alpha","omega"]}"#,
    );
    let chat_b = parse_request(
        r#"{"messages":[{"role":"user","content":"same stop set"}],"temperature":0.0,"max_tokens":4,"stop":["alpha","omega"]}"#,
    );
    assert_eq!(
        deterministic_chat_request_cache_key(&chat_a, &completion_a).unwrap(),
        deterministic_chat_request_cache_key(&chat_b, &completion_b).unwrap(),
        "stop sequence order and duplicates should not split chat request-cache entries"
    );
    let chat_string = parse_request(
        r#"{"messages":[{"role":"user","content":"same stop set"}],"temperature":0.0,"max_tokens":4,"stop":"alpha"}"#,
    );
    let chat_single_list = parse_request(
        r#"{"messages":[{"role":"user","content":"same stop set"}],"temperature":0.0,"max_tokens":4,"stop":["alpha"]}"#,
    );
    let single_stop_sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 4,
        stop: vec!["alpha".to_string()],
        ..Default::default()
    };
    assert_eq!(
        deterministic_chat_request_cache_key(&chat_string, &single_stop_sampling).unwrap(),
        deterministic_chat_request_cache_key(&chat_single_list, &single_stop_sampling).unwrap(),
        "single-string stop should share chat request-cache entries with a one-item stop list"
    );

    let batch_a = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same stop set"}]],"n":1,"temperature":0.0,"max_tokens":4,"stop":["omega","alpha","omega"]}"#,
    );
    let batch_b = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same stop set"}]],"n":1,"temperature":0.0,"max_tokens":4,"stop":["alpha","omega"]}"#,
    );
    assert_eq!(
        deterministic_batch_cache_key(&batch_a, 1),
        deterministic_batch_cache_key(&batch_b, 1),
        "stop sequence order and duplicates should not split batch-cache entries"
    );
    let batch_string = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same stop set"}]],"n":1,"temperature":0.0,"max_tokens":4,"stop":"alpha"}"#,
    );
    let batch_single_list = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same stop set"}]],"n":1,"temperature":0.0,"max_tokens":4,"stop":["alpha"]}"#,
    );
    assert_eq!(
        deterministic_batch_cache_key(&batch_string, 1),
        deterministic_batch_cache_key(&batch_single_list, 1),
        "single-string stop should share batch-cache entries with a one-item stop list"
    );

    assert_eq!(
        normalized_stop_for_cache(&["x".to_string(), String::new()]),
        vec![String::new()],
        "an empty stop sequence dominates other stop strings"
    );
    assert_eq!(
        normalized_stop_for_cache(&[
            "omega".to_string(),
            "alpha-extra".to_string(),
            "alpha".to_string(),
            "omega-tail".to_string(),
        ]),
        vec!["alpha".to_string(), "omega".to_string()],
        "a shorter stop sequence dominates longer stops that start with it"
    );
    assert_eq!(
        normalized_stop_for_cache(&[
            "prefix-needle-suffix".to_string(),
            "needle".to_string(),
            "az".to_string(),
            "xaz".to_string(),
        ]),
        vec!["az".to_string(), "needle".to_string()],
        "a shorter stop sequence dominates longer stops that contain it anywhere"
    );
    assert_eq!(
        normalized_stop_for_generation(Some(&[
            "prefix-needle-suffix".to_string(),
            "needle".to_string(),
        ])),
        vec!["needle".to_string()],
        "fresh generation should use the same canonical stop list as replay keys"
    );
    assert_eq!(
        normalized_stop_option_for_synthetic_request(Some(&[
            "prefix-needle-suffix".to_string(),
            "needle".to_string(),
        ])),
        Some(vec!["needle".to_string()]),
        "synthetic fanout requests should clone only canonical stop strings"
    );
    assert_eq!(
        normalized_stop_option_for_synthetic_request(Some(&[] as &[String])),
        None,
        "synthetic requests should preserve no-stop as None"
    );
}

#[test]
fn deterministic_batch_cache_key_normalizes_equivalent_sampling_fields() {
    let greedy_a = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same greedy batch"}]],"n":1,"temperature":0.0,"top_p":0.8,"top_k":17,"max_tokens":4,"seed":1}"#,
    );
    let greedy_b = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same greedy batch"}]],"n":1,"temperature":0.0,"top_p":0.95,"top_k":0,"max_tokens":4,"seed":2}"#,
    );
    assert_eq!(
        deterministic_batch_cache_key(&greedy_a, 1),
        deterministic_batch_cache_key(&greedy_b, 1),
        "greedy batch-cache keys should ignore seed/top-p/top-k"
    );
    let top_k_one = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same greedy batch"}]],"n":1,"temperature":0.7,"top_p":0.2,"top_k":1,"max_tokens":4}"#,
    );
    assert_eq!(
        deterministic_batch_cache_key(&greedy_a, 1),
        deterministic_batch_cache_key(&top_k_one, 1),
        "top_k=1 batch-cache keys should normalize to greedy"
    );

    let zero_a = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same zero batch"}]],"n":1,"temperature":0.7,"top_p":0.8,"top_k":17,"max_tokens":0,"stop":["x"],"seed":1}"#,
    );
    let zero_b = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same zero batch"}]],"n":1,"temperature":0.2,"top_p":0.95,"top_k":0,"max_tokens":0,"stop":["y"],"seed":2}"#,
    );
    assert_eq!(
        deterministic_batch_cache_key(&zero_a, 1),
        deterministic_batch_cache_key(&zero_b, 1),
        "max_tokens=0 batch-cache keys should ignore generation-only sampling fields"
    );

    let sampled_a = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"sampled batch"}]],"n":1,"temperature":0.7,"max_tokens":4,"seed":1}"#,
    );
    let sampled_b = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"sampled batch"}]],"n":1,"temperature":0.7,"max_tokens":4,"seed":2}"#,
    );
    assert_ne!(
        deterministic_batch_cache_key(&sampled_a, 1),
        deterministic_batch_cache_key(&sampled_b, 1),
        "seeded sampled batch-cache keys must still split by seed"
    );
    let sampled_full_distribution = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"sampled batch full top p"}]],"n":1,"temperature":0.7,"top_p":1.0,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_p_above_one = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"sampled batch full top p"}]],"n":1,"temperature":0.7,"top_p":1.5,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_p_zero = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"sampled batch full top p"}]],"n":1,"temperature":0.7,"top_p":0.0,"top_k":0,"max_tokens":4,"seed":1}"#,
    );
    let sampled_top_k_disabled = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"sampled batch full top p"}]],"n":1,"temperature":0.7,"top_p":1.0,"top_k":248320,"max_tokens":4,"seed":1}"#,
    );
    assert_eq!(
        deterministic_batch_cache_key(&sampled_full_distribution, 1),
        deterministic_batch_cache_key(&sampled_top_p_above_one, 1),
        "top_p >= 1.0 disables nucleus filtering, so full-distribution seeded batches should share cache entries"
    );
    assert_eq!(
        deterministic_batch_cache_key(&sampled_full_distribution, 1),
        deterministic_batch_cache_key(&sampled_top_p_zero, 1),
        "top_p=0 disables nucleus filtering, so full-distribution seeded batches should share cache entries"
    );
    let model_vocab_size = ModelConfig::qwen3_5_4b().vocab_size;
    assert_eq!(
        deterministic_batch_cache_key_with_vocab_size(
            &sampled_full_distribution,
            1,
            model_vocab_size
        ),
        deterministic_batch_cache_key_with_vocab_size(&sampled_top_k_disabled, 1, model_vocab_size),
        "top_k >= model vocab size is disabled, so full-distribution seeded batches should share cache entries"
    );
}

#[test]
fn deterministic_batch_cache_key_normalizes_max_completion_tokens_alias() {
    let max_tokens = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch max token alias"}]],"n":2,"temperature":0.0,"max_tokens":4,"seed":1}"#,
    );
    let alias = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch max token alias"}]],"n":2,"temperature":0.0,"max_completion_tokens":4,"seed":2}"#,
    );

    assert_eq!(
        deterministic_batch_cache_key(&max_tokens, 2),
        deterministic_batch_cache_key(&alias, 2),
        "max_completion_tokens should share deterministic batch-cache entries with max_tokens"
    );
}

#[test]
fn deterministic_batch_cache_key_ignores_default_openai_option_fields() {
    let plain = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch default options"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );
    let defaults = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch default options"}]],"n":2,"temperature":0.0,"max_tokens":4,"response_format":{"type":"text"},"parallel_tool_calls":true,"user":"client-a","metadata":{"trace_id":"ignored"},"store":false,"service_tier":"auto","logprobs":false,"top_logprobs":0,"frequency_penalty":0.0,"presence_penalty":0.0,"stream_options":{"include_usage":false}}"#,
    );

    assert_eq!(
        deterministic_batch_cache_key(&plain, 2),
        deterministic_batch_cache_key(&defaults, 2),
        "default OpenAI option fields should not split deterministic batch-cache entries"
    );
}

#[test]
fn deterministic_batch_cache_key_includes_chat_template_kwargs() {
    let default_req = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch template kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );
    let empty_kwargs = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch template kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"chat_template_kwargs":{}}"#,
    );
    let no_think = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch template kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false}}"#,
    );

    assert_eq!(
        deterministic_batch_cache_key(&default_req, 2),
        deterministic_batch_cache_key(&empty_kwargs, 2),
        "empty batch chat_template_kwargs should normalize to omitted"
    );
    assert_ne!(
        deterministic_batch_cache_key(&default_req, 2),
        deterministic_batch_cache_key(&no_think, 2),
        "template kwargs change rendered prompts and must split batch-cache entries"
    );
}

#[test]
fn deterministic_batch_cache_key_keeps_tool_template_kwargs_explicit() {
    let omitted_kwargs = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch tool default kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto"}"#,
    );
    let explicit_no_think = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch tool default kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":false}}"#,
    );
    let explicit_think = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch tool default kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":true}}"#,
    );

    assert_ne!(
        deterministic_batch_cache_key(&omitted_kwargs, 2),
        deterministic_batch_cache_key(&explicit_no_think, 2),
        "batch tool requests must not reinterpret omitted kwargs as enable_thinking=false"
    );
    assert_ne!(
        deterministic_batch_cache_key(&explicit_no_think, 2),
        deterministic_batch_cache_key(&explicit_think, 2),
        "explicit batch enable_thinking values must keep separate cache entries"
    );
}

#[test]
fn deterministic_batch_cache_key_includes_tools_and_tool_choice() {
    let plain = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch tools"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );
    let with_tools = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch tools"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}]}"#,
    );
    let with_tools_required = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch tools"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"required"}"#,
    );

    assert_ne!(
        deterministic_batch_cache_key(&plain, 2),
        deterministic_batch_cache_key(&with_tools, 2),
        "top-level tools change the rendered batch prompt and must split cache entries"
    );
    assert_ne!(
        deterministic_batch_cache_key(&with_tools, 2),
        deterministic_batch_cache_key(&with_tools_required, 2),
        "tool_choice is template-visible when tools are present"
    );
}

#[test]
fn deterministic_batch_cache_key_normalizes_text_content_parts() {
    let plain = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch text parts"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );
    let parts = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":[{"type":"text","text":"same batch "},{"type":"text","text":"text parts"}]}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );

    assert_eq!(
        deterministic_batch_cache_key(&plain, 2),
        deterministic_batch_cache_key(&parts, 2),
        "equivalent OpenAI text content parts should not split batch-cache entries"
    );
}

#[test]
fn deterministic_batch_cache_key_ignores_non_text_content_parts() {
    let plain = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch visible text"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );
    let parts = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":[{"type":"text","text":"same batch visible "},{"type":"image_url","image_url":{"url":"https://example.invalid/ignored.png"}},{"type":"input_audio","input_audio":{"data":"ignored","format":"wav"}},{"type":"text","text":"text"}]}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );

    assert_eq!(
        deterministic_batch_cache_key(&plain, 2),
        deterministic_batch_cache_key(&parts, 2),
        "non-text content parts are ignored by the text-only deserializer and should not split batch-cache entries"
    );
}

#[test]
fn deterministic_batch_cache_key_ignores_unrendered_message_metadata() {
    let plain = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch metadata"},{"role":"assistant","content":"ok"},{"role":"user","content":"continue"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );
    let metadata = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch metadata","reasoning_content":"ignored by batch renderer"},{"role":"assistant","content":"ok","tool_calls":[]},{"role":"user","content":"continue"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );

    assert_eq!(
        deterministic_batch_cache_key(&plain, 2),
        deterministic_batch_cache_key(&metadata, 2),
        "reasoning_content and empty tool_calls should not split batch-cache entries"
    );
    let non_empty_tool_calls = parse_batch_request(
        r#"{"prompts":[[{"role":"user","content":"same batch metadata"},{"role":"assistant","content":"ok","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"{}"}}]},{"role":"user","content":"continue"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
    );
    assert_ne!(
        deterministic_batch_cache_key(&plain, 2),
        deterministic_batch_cache_key(&non_empty_tool_calls, 2),
        "historical assistant tool calls are rendered and must split batch-cache entries"
    );
}

#[tokio::test]
async fn batch_greedy_n_clones_one_physical_completion() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"hi"}]],
        "n": 3,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 7
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let completions = body["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 3);
    assert_eq!(completions[0]["completion_index"], 0);
    assert_eq!(completions[1]["completion_index"], 1);
    assert_eq!(completions[2]["completion_index"], 2);
    assert_eq!(completions[1]["text"], completions[0]["text"]);
    assert_eq!(completions[2]["text"], completions[0]["text"]);

    let one_completion_tokens = completions[0]["usage"]["completion_tokens"]
        .as_u64()
        .unwrap();
    assert_eq!(
        body["usage"]["completion_tokens"].as_u64().unwrap(),
        one_completion_tokens * 3
    );
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        one_completion_tokens,
        "metrics should count physical model decode work, not cloned logical completions"
    );
    assert_eq!(
        state.recent_requests.lock().unwrap().len(),
        1,
        "only the physical generation should enter the recent-request ring"
    );
}

#[tokio::test]
async fn batch_greedy_duplicate_prompts_clone_one_physical_completion() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"same greedy prompt"}],
            [{"role":"user","content":"same greedy prompt"}]
        ],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 7
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let completions = body["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 4);
    let logical_positions: Vec<(u64, u64)> = completions
        .iter()
        .map(|item| {
            (
                item["prompt_index"].as_u64().unwrap(),
                item["completion_index"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(logical_positions, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);

    let first_text = completions[0]["text"].clone();
    assert!(
        completions.iter().all(|item| item["text"] == first_text),
        "all identical greedy prompt rows should clone the same deterministic text"
    );

    let one_completion_tokens = completions[0]["usage"]["completion_tokens"]
        .as_u64()
        .unwrap();
    assert_eq!(
        body["usage"]["completion_tokens"].as_u64().unwrap(),
        one_completion_tokens * 4
    );
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        one_completion_tokens,
        "duplicate greedy prompt rows should perform one physical generation total"
    );
    assert_eq!(
        state.recent_requests.lock().unwrap().len(),
        1,
        "cloned duplicate prompt rows should not create extra synthetic recent requests"
    );

    let (render_hits, render_misses, _) = state.rendered_prompt_cache.lock().unwrap().stats();
    assert_eq!(render_misses, 1);
    assert_eq!(
        render_hits, 0,
        "cloned duplicate prompt rows should skip render-cache lookups entirely"
    );

    let (token_hits, token_misses, _) = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(token_misses, 1);
    assert_eq!(
        token_hits, 0,
        "cloned duplicate prompt rows should skip token-cache lookups entirely"
    );
}

#[tokio::test]
async fn batch_top_k_one_clones_single_physical_completion() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"top k one greedy batch"}]],
        "n": 4,
        "temperature": 0.7,
        "top_p": 0.2,
        "top_k": 1,
        "max_tokens": 4
    })
    .to_string();
    let repeat_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"top k one greedy batch"}]],
        "n": 4,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 1,
        "max_tokens": 4
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let completions = body["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 4);
    let first_text = completions[0]["text"].clone();
    assert!(
        completions.iter().all(|item| item["text"] == first_text),
        "top_k=1 completions should clone the single greedy physical output"
    );

    let physical_completion_tokens = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(physical_completion_tokens > 0);
    assert_eq!(
        body["usage"]["completion_tokens"].as_u64().unwrap(),
        physical_completion_tokens * 4,
        "top_k=1 batch n usage should count logical completions while model work stays single-output"
    );

    let (render_hits, render_misses, _) = state.rendered_prompt_cache.lock().unwrap().stats();
    assert_eq!((render_hits, render_misses), (0, 1));
    let (token_hits, token_misses, _) = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!((token_hits, token_misses), (0, 1));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    let (status_repeat, repeat) = batch_post(state.clone(), &repeat_body).await;
    assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "unseeded top_k=1 batch repeat should hit the whole-batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_k=1 batch-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_k=1 batch-cache hit should return before prompt-token lookup"
    );
    assert_eq!(body["usage"], repeat["usage"]);
    assert_eq!(body["completions"], repeat["completions"]);
}

#[tokio::test]
async fn batch_top_p_above_one_hits_seeded_full_distribution_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"top p full distribution batch cache"}]],
        "n": 4,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 4,
        "seed": 140
    })
    .to_string();
    let repeat_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"top p full distribution batch cache"}]],
        "n": 4,
        "temperature": 0.7,
        "top_p": 1.5,
        "max_tokens": 4,
        "seed": 140
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_repeat, repeat) = batch_post(state.clone(), &repeat_body).await;
    assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "top_p >= 1.0 seeded batch repeat should hit the whole-batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_p >= 1.0 batch-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_p >= 1.0 batch-cache hit should return before prompt-token lookup"
    );
    assert_eq!(body["usage"], repeat["usage"]);
    assert_eq!(body["completions"], repeat["completions"]);
}

#[tokio::test]
async fn batch_top_p_zero_hits_seeded_full_distribution_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"top p zero full distribution batch cache"}]],
        "n": 4,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 4,
        "seed": 144
    })
    .to_string();
    let repeat_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"top p zero full distribution batch cache"}]],
        "n": 4,
        "temperature": 0.7,
        "top_p": 0.0,
        "max_tokens": 4,
        "seed": 144
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_repeat, repeat) = batch_post(state.clone(), &repeat_body).await;
    assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "top_p=0 seeded batch repeat should hit the whole-batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_p=0 batch-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_p=0 batch-cache hit should return before prompt-token lookup"
    );
    assert_eq!(body["usage"], repeat["usage"]);
    assert_eq!(body["completions"], repeat["completions"]);
}

#[tokio::test]
async fn batch_top_k_oversized_hits_seeded_full_distribution_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let disabled_top_k = state.model_config.vocab_size as u32;
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"top k oversized full distribution batch cache"}]],
        "n": 4,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": disabled_top_k,
        "max_tokens": 4,
        "seed": 148
    })
    .to_string();
    let repeat_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"top k oversized full distribution batch cache"}]],
        "n": 4,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 4,
        "seed": 148
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_repeat, repeat) = batch_post(state.clone(), &repeat_body).await;
    assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "top_k >= vocab seeded batch repeat should hit the whole-batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_k >= vocab batch-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_k >= vocab batch-cache hit should return before prompt-token lookup"
    );
    assert_eq!(body["usage"], repeat["usage"]);
    assert_eq!(body["completions"], repeat["completions"]);
}

#[tokio::test]
async fn chat_multi_choice_greedy_clones_single_physical_completion() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"same greedy chat choices"}],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 7
    })
    .to_string();

    let (status, body) = chat_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 4);
    let choice_indices: Vec<u64> = choices
        .iter()
        .map(|choice| choice["index"].as_u64().unwrap())
        .collect();
    assert_eq!(choice_indices, vec![0, 1, 2, 3]);

    let first_message = choices[0]["message"].clone();
    assert!(
        choices
            .iter()
            .all(|choice| choice["message"] == first_message),
        "all greedy choices should clone the same deterministic message"
    );

    let physical_completion_tokens = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(physical_completion_tokens > 0);
    assert_eq!(
        body["usage"]["completion_tokens"].as_u64().unwrap(),
        physical_completion_tokens * 4,
        "logical chat n usage should count every returned choice"
    );
    assert_eq!(
        state.recent_requests.lock().unwrap().len(),
        1,
        "cloned chat choices should not create extra synthetic recent requests"
    );

    let (render_hits, render_misses, _) = state.rendered_prompt_cache.lock().unwrap().stats();
    assert_eq!(render_misses, 1);
    assert_eq!(
        render_hits, 0,
        "cloned chat choices should skip render-cache lookups entirely"
    );

    let (token_hits, token_misses, _) = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(token_misses, 1);
    assert_eq!(
        token_hits, 0,
        "cloned chat choices should skip token-cache lookups entirely"
    );
}

#[tokio::test]
async fn chat_multi_choice_top_k_one_clones_single_physical_completion() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"top k one greedy chat choices"}],
        "n": 4,
        "temperature": 0.7,
        "top_p": 0.2,
        "top_k": 1,
        "max_tokens": 4
    })
    .to_string();
    let repeat_body = serde_json::json!({
        "messages": [{"role":"user","content":"top k one greedy chat choices"}],
        "n": 4,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 1,
        "max_tokens": 4
    })
    .to_string();

    let (status, body) = chat_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 4);
    let first_message = choices[0]["message"].clone();
    assert!(
        choices
            .iter()
            .all(|choice| choice["message"] == first_message),
        "top_k=1 chat choices should clone the single greedy physical output"
    );

    let physical_completion_tokens = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(physical_completion_tokens > 0);
    assert_eq!(
        body["usage"]["completion_tokens"].as_u64().unwrap(),
        physical_completion_tokens * 4,
        "top_k=1 chat n usage should count logical choices while model work stays single-output"
    );

    let (render_hits, render_misses, _) = state.rendered_prompt_cache.lock().unwrap().stats();
    assert_eq!((render_hits, render_misses), (0, 1));
    let (token_hits, token_misses, _) = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!((token_hits, token_misses), (0, 1));
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    let (status_repeat, repeat) = chat_post(state.clone(), &repeat_body).await;
    assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "unseeded top_k=1 chat n repeat should hit the top-level choices cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_k=1 chat choices hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_k=1 chat choices hit should return before prompt-token lookup"
    );
    assert_eq!(body["usage"], repeat["usage"]);
    assert_eq!(body["choices"], repeat["choices"]);
}

#[tokio::test]
async fn chat_multi_choice_repeated_greedy_hits_top_level_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let first_body = serde_json::json!({
        "messages": [{"role":"user","content":"cache chat n choices"}],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let second_body = serde_json::json!({
        "messages": [{"role":"user","content":"cache chat n choices"}],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &first_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &second_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "second greedy chat n request should hit the top-level choices cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top-level chat n cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top-level chat n cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn chat_multi_choice_repeated_seeded_sampled_hits_top_level_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"cache sampled chat n choices"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    assert_eq!(first["choices"].as_array().unwrap().len(), 3);
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(
        render_stats_after_first,
        (2, 1, 1),
        "first sampled chat n request should render once and reuse the rendered prompt for later choices"
    );
    assert_eq!(
        token_stats_after_first,
        (2, 1, 1),
        "first sampled chat n request should tokenize once and reuse tokens for later choices"
    );
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 3);
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "second seeded sampled chat n request should hit the top-level choices cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top-level sampled chat n cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top-level sampled chat n cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn concurrent_chat_multi_choice_singleflights_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"chat n choices singleflight"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (first, second) = tokio::join!(
        chat_post(state.clone(), &body),
        chat_post(state.clone(), &body)
    );
    let (status_first, first) = first;
    let (status_second, second) = second;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        (2, 1, 1),
        "concurrent duplicate chat n request should render once and reuse for later choices only"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        (2, 1, 1),
        "concurrent duplicate chat n request should tokenize once and reuse for later choices only"
    );
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        first["usage"]["completion_tokens"].as_u64().unwrap(),
        "concurrent duplicate chat n request should do one top-level set of physical completions"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
    assert_ne!(
        first["id"], second["id"],
        "singleflight replay should still get a fresh chat response id"
    );
}

#[tokio::test]
async fn single_prompt_batch_hits_chat_choices_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let chat_body = serde_json::json!({
        "messages": [{"role":"user","content":"share sampled chat n with batch"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let batch_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"share sampled chat n with batch"}]],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
    assert_eq!(chat["choices"].as_array().unwrap().len(), 3);
    let generated_after_chat = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_chat > 0);
    let render_stats_after_chat = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_chat = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 0);

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_chat,
        "equivalent one-prompt batch should hit the chat choices cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chat,
        "batch-from-chat-choices hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chat,
        "batch-from-chat-choices hit should return before prompt-token lookup"
    );
    assert_eq!(
        state.batch_cache.lock().unwrap().stats(),
        1,
        "chat choices hit should also populate the batch cache"
    );

    assert_eq!(
        batch["usage"]["prompt_tokens"].as_u64().unwrap(),
        chat["usage"]["prompt_tokens"].as_u64().unwrap() * 3
    );
    assert_eq!(
        batch["usage"]["completion_tokens"],
        chat["usage"]["completion_tokens"]
    );
    for (choice, completion) in chat["choices"]
        .as_array()
        .unwrap()
        .iter()
        .zip(batch["completions"].as_array().unwrap())
    {
        assert_eq!(completion["text"], choice["message"]["content"]);
        assert_eq!(completion["finish_reason"], choice["finish_reason"]);
    }
}

#[tokio::test]
async fn single_prompt_batch_from_choices_cache_rehydrates_request_cache_before_single_chat_work() {
    let state = make_batch_test_state();
    let chat_n_body = serde_json::json!({
        "messages": [{"role":"user","content":"choices batch rehydrates request single"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 193
    })
    .to_string();
    let batch_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"choices batch rehydrates request single"}]],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 193
    })
    .to_string();
    let chat_one_body = serde_json::json!({
        "messages": [{"role":"user","content":"choices batch rehydrates request single"}],
        "temperature": 0.2,
        "top_p": 0.1,
        "max_tokens": 0,
        "seed": 999
    })
    .to_string();

    let (status_chat_n, chat_n) = chat_post(state.clone(), &chat_n_body).await;
    assert_eq!(status_chat_n, axum::http::StatusCode::OK, "{chat_n}");
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
    let render_stats_after_chat_n = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_chat_n = state.prompt_token_cache.lock().unwrap().stats();

    *state.chat_request_cache.lock().unwrap() = DeterministicChatRequestCache::new(128);
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chat_n,
        "batch choices-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chat_n,
        "batch choices-cache hit should return before prompt-token lookup"
    );
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        1,
        "batch from choices cache should rehydrate the normalized single-chat request entry"
    );

    let (status_chat_one, chat_one) = chat_post(state.clone(), &chat_one_body).await;
    assert_eq!(status_chat_one, axum::http::StatusCode::OK, "{chat_one}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chat_n,
        "rehydrated single-chat request hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chat_n,
        "rehydrated single-chat request hit should return before prompt-token lookup"
    );
    assert_eq!(chat_one["usage"], batch["completions"][0]["usage"]);
    assert_eq!(chat_one["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn single_prompt_batch_populates_chat_choices_cache_before_chat_work() {
    let state = make_batch_test_state();
    let batch_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"share sampled batch n with chat"}]],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let chat_body = serde_json::json!({
        "messages": [{"role":"user","content":"share sampled batch n with chat"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(batch["completions"].as_array().unwrap().len(), 3);
    let generated_after_batch = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_batch > 0);
    let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

    let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_batch,
        "equivalent chat n request should hit the chat choices cache populated by batch"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_batch,
        "chat-from-batch-choices hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_batch,
        "chat-from-batch-choices hit should return before prompt-token lookup"
    );

    assert_eq!(
        chat["usage"]["prompt_tokens"].as_u64().unwrap() * 3,
        batch["usage"]["prompt_tokens"].as_u64().unwrap()
    );
    assert_eq!(
        chat["usage"]["completion_tokens"],
        batch["usage"]["completion_tokens"]
    );
    for (choice, completion) in chat["choices"]
        .as_array()
        .unwrap()
        .iter()
        .zip(batch["completions"].as_array().unwrap())
    {
        assert_eq!(choice["message"]["content"], completion["text"]);
        assert_eq!(choice["finish_reason"], completion["finish_reason"]);
    }
}

#[tokio::test]
async fn batch_repeated_greedy_request_hits_batch_cache_before_completion_cache_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"cache me"}]],
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        generated_after_first > 0,
        "first request should perform physical generation"
    );
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "second identical greedy request should be served from batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch cache hit should return before rendered-prompt cache lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch cache hit should return before prompt-token cache lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(
        first["completions"][0]["text"],
        second["completions"][0]["text"]
    );
    assert_ne!(
        first["id"], second["id"],
        "cached responses should still get a fresh batch response id"
    );
    assert_eq!(
        state.recent_requests.lock().unwrap().len(),
        1,
        "early batch-cache hits should not synthesize per-output recent requests"
    );
}

#[tokio::test]
async fn batch_text_content_parts_hits_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let plain_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch text content parts should be no-op"}]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let parts_body = serde_json::json!({
        "prompts": [[{
            "role":"user",
            "content":[
                {"type":"text","text":"batch text content parts "},
                {"type":"text","text":"should be no-op"}
            ]
        }]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &plain_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &parts_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "equivalent text content parts should reuse the batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch text content parts cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch text content parts cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
}

#[tokio::test]
async fn batch_non_text_content_parts_hits_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let plain_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch non-text content parts should be no-op"}]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let parts_body = serde_json::json!({
        "prompts": [[{
            "role":"user",
            "content":[
                {"type":"text","text":"batch non-text content parts "},
                {"type":"image_url","image_url":{"url":"https://example.invalid/ignored.png"}},
                {"type":"input_audio","input_audio":{"data":"ignored","format":"wav"}},
                {"type":"text","text":"should be no-op"}
            ]
        }]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &plain_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &parts_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "non-text content parts should reuse the batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch non-text content parts cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch non-text content parts cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
}

#[tokio::test]
async fn batch_max_completion_tokens_alias_hits_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let max_tokens_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch max completion tokens alias should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
    let alias_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch max completion tokens alias should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_completion_tokens": 4,
            "seed": 999
        })
        .to_string();

    let (status_first, first) = batch_post(state.clone(), &max_tokens_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &alias_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "max_completion_tokens should reuse the max_tokens batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch max_completion_tokens cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch max_completion_tokens cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
}

#[tokio::test]
async fn batch_stop_string_hits_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let list_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch stop string should be no-op"}]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["never-match-stop"],
        "seed": 123
    })
    .to_string();
    let string_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch stop string should be no-op"}]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": "never-match-stop",
        "seed": 999
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &list_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &string_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "single-string stop should reuse the one-item list batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch stop-string cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch stop-string cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
}

#[tokio::test]
async fn batch_dominated_stop_hits_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let redundant_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch dominated stop should be no-op"}]],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["never-match-stop", "never-match-stop-suffix"],
        "seed": 123
    })
    .to_string();
    let minimal_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch dominated stop should be no-op"}]],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["never-match-stop"],
        "seed": 999
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &redundant_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &minimal_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "dominated stop sequences should reuse the minimal stop batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "dominated-stop batch-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "dominated-stop batch-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
}

#[tokio::test]
async fn batch_substring_dominated_stop_hits_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let redundant_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch substring dominated stop should be no-op"}]],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["prefix-never-match-stop-suffix", "never-match-stop"],
        "seed": 123
    })
    .to_string();
    let minimal_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch substring dominated stop should be no-op"}]],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["never-match-stop"],
        "seed": 999
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &redundant_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &minimal_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "substring-dominated stop sequences should reuse the minimal stop batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "substring-dominated-stop batch-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "substring-dominated-stop batch-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
}

#[tokio::test]
async fn batch_default_openai_option_fields_hit_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let plain_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch default OpenAI options should be no-op"}]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let defaults_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"batch default OpenAI options should be no-op"}]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999,
        "response_format": {"type":"text"},
        "parallel_tool_calls": true,
        "user": "client-a",
        "metadata": {"trace_id":"ignored"},
        "store": false,
        "service_tier": "auto",
        "logprobs": false,
        "top_logprobs": 0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream_options": {"include_usage": false}
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &plain_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &defaults_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "default OpenAI options should reuse the batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch default-option cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch default-option cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
}

#[tokio::test]
async fn batch_unrendered_message_metadata_hits_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let plain_body = serde_json::json!({
        "prompts": [[
            {"role":"user","content":"batch ignored metadata should be no-op"},
            {"role":"assistant","content":"ok"},
            {"role":"user","content":"continue"}
        ]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let metadata_body = serde_json::json!({
        "prompts": [[
            {
                "role":"user",
                "content":"batch ignored metadata should be no-op",
                "reasoning_content":"ignored by batch renderer"
            },
            {"role":"assistant","content":"ok","tool_calls":[]},
            {"role":"user","content":"continue"}
        ]],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &plain_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &metadata_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "batch ignored message metadata should reuse the batch cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch ignored metadata cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch ignored metadata cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
}

#[tokio::test]
async fn batch_single_output_hits_chat_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let chat_body = serde_json::json!({
        "messages": [{"role":"user","content":"cross endpoint chat to batch"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let batch_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"cross endpoint chat to batch"}]],
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
    let generated_after_chat = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_chat > 0);
    let render_stats_after_chat = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_chat = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_chat, (0, 1, 1));
    assert_eq!(token_stats_after_chat, (0, 1, 1));
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_chat,
        "single-output batch should reuse the chat request cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chat,
        "batch should hit chat request cache before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chat,
        "batch should hit chat request cache before prompt-token lookup"
    );
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(chat["usage"], batch["usage"]);
    assert_eq!(
        chat["choices"][0]["message"]["content"],
        batch["completions"][0]["text"]
    );
}

#[tokio::test]
async fn multi_prompt_batch_hits_chat_request_cache_before_fanout_work() {
    let state = make_batch_test_state();
    let chat_a_body = serde_json::json!({
        "messages": [{"role":"user","content":"cross endpoint cached prompt a"}],
        "temperature": 0.7,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let chat_b_body = serde_json::json!({
        "messages": [{"role":"user","content":"cross endpoint cached prompt b"}],
        "temperature": 0.7,
        "max_tokens": 4,
        "seed": 124
    })
    .to_string();
    let batch_body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"cross endpoint cached prompt a"}],
            [{"role":"user","content":"cross endpoint cached prompt b"}]
        ],
        "n": 1,
        "temperature": 0.7,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_a, chat_a) = chat_post(state.clone(), &chat_a_body).await;
    assert_eq!(status_a, axum::http::StatusCode::OK, "{chat_a}");
    let (status_b, chat_b) = chat_post(state.clone(), &chat_b_body).await;
    assert_eq!(status_b, axum::http::StatusCode::OK, "{chat_b}");
    let generated_after_chats = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_chats > 0);
    let render_stats_after_chats = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_chats = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_chats, (0, 2, 2));
    assert_eq!(token_stats_after_chats, (0, 2, 2));
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 2);

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_chats,
        "multi-prompt batch should reuse per-prompt chat request-cache hits"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chats,
        "multi-prompt batch should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chats,
        "multi-prompt batch should return before prompt-token lookup"
    );
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(
        chat_a["choices"][0]["message"]["content"],
        batch["completions"][0]["text"]
    );
    assert_eq!(
        chat_b["choices"][0]["message"]["content"],
        batch["completions"][1]["text"]
    );
    assert_eq!(
        batch["usage"]["prompt_tokens"],
        chat_a["usage"]["prompt_tokens"].as_u64().unwrap()
            + chat_b["usage"]["prompt_tokens"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn multi_prompt_batch_hits_chat_choices_cache_before_fanout_work() {
    let state = make_batch_test_state();
    let chat_a_body = serde_json::json!({
        "messages": [{"role":"user","content":"cross endpoint cached choices prompt a"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 700
    })
    .to_string();
    let chat_b_body = serde_json::json!({
        "messages": [{"role":"user","content":"cross endpoint cached choices prompt b"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 703
    })
    .to_string();
    let batch_body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"cross endpoint cached choices prompt a"}],
            [{"role":"user","content":"cross endpoint cached choices prompt b"}]
        ],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 700
    })
    .to_string();

    let (status_a, chat_a) = chat_post(state.clone(), &chat_a_body).await;
    assert_eq!(status_a, axum::http::StatusCode::OK, "{chat_a}");
    let (status_b, chat_b) = chat_post(state.clone(), &chat_b_body).await;
    assert_eq!(status_b, axum::http::StatusCode::OK, "{chat_b}");
    assert_eq!(chat_a["choices"].as_array().unwrap().len(), 3);
    assert_eq!(chat_b["choices"].as_array().unwrap().len(), 3);
    let generated_after_chats = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_chats > 0);
    let render_stats_after_chats = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_chats = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_chats, (4, 2, 2));
    assert_eq!(token_stats_after_chats, (4, 2, 2));
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 6);
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 2);

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_chats,
        "multi-prompt n batch should reuse per-prompt chat choices-cache hits"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chats,
        "multi-prompt n batch should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chats,
        "multi-prompt n batch should return before prompt-token lookup"
    );
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(batch["completions"].as_array().unwrap().len(), 6);
    assert_eq!(
        batch["usage"]["prompt_tokens"].as_u64().unwrap(),
        (chat_a["usage"]["prompt_tokens"].as_u64().unwrap()
            + chat_b["usage"]["prompt_tokens"].as_u64().unwrap())
            * 3
    );
    assert_eq!(
        batch["usage"]["completion_tokens"].as_u64().unwrap(),
        chat_a["usage"]["completion_tokens"].as_u64().unwrap()
            + chat_b["usage"]["completion_tokens"].as_u64().unwrap()
    );
    for (choice, completion) in chat_a["choices"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&batch["completions"].as_array().unwrap()[0..3])
    {
        assert_eq!(completion["prompt_index"], 0);
        assert_eq!(completion["text"], choice["message"]["content"]);
        assert_eq!(completion["finish_reason"], choice["finish_reason"]);
    }
    for (choice, completion) in chat_b["choices"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&batch["completions"].as_array().unwrap()[3..6])
    {
        assert_eq!(completion["prompt_index"], 1);
        assert_eq!(completion["text"], choice["message"]["content"]);
        assert_eq!(completion["finish_reason"], choice["finish_reason"]);
    }
}

#[tokio::test]
async fn multi_prompt_batch_from_choices_cache_rehydrates_request_cache_before_chat_work() {
    let state = make_batch_test_state();
    let chat_a_body = serde_json::json!({
        "messages": [{"role":"user","content":"multi choices rehydrate request prompt a"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 194
    })
    .to_string();
    let chat_b_body = serde_json::json!({
        "messages": [{"role":"user","content":"multi choices rehydrate request prompt b"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 197
    })
    .to_string();
    let batch_body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"multi choices rehydrate request prompt a"}],
            [{"role":"user","content":"multi choices rehydrate request prompt b"}]
        ],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 194
    })
    .to_string();
    let chat_b_single_body = serde_json::json!({
        "messages": [{"role":"user","content":"multi choices rehydrate request prompt b"}],
        "temperature": 0.2,
        "top_p": 0.1,
        "max_tokens": 0,
        "seed": 999
    })
    .to_string();

    let (status_a, chat_a) = chat_post(state.clone(), &chat_a_body).await;
    assert_eq!(status_a, axum::http::StatusCode::OK, "{chat_a}");
    let (status_b, chat_b) = chat_post(state.clone(), &chat_b_body).await;
    assert_eq!(status_b, axum::http::StatusCode::OK, "{chat_b}");
    let render_stats_after_chats = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_chats = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_chats, (0, 2, 2));
    assert_eq!(token_stats_after_chats, (0, 2, 2));
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 2);
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 2);

    *state.chat_request_cache.lock().unwrap() = DeterministicChatRequestCache::new(128);
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chats,
        "multi-prompt batch choices-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chats,
        "multi-prompt batch choices-cache hit should return before prompt-token lookup"
    );
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        2,
        "multi-prompt batch from choices cache should rehydrate one request entry per prompt"
    );

    let (status_chat, chat_b_single) = chat_post(state.clone(), &chat_b_single_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_b_single}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chats,
        "rehydrated prompt-b request hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chats,
        "rehydrated prompt-b request hit should return before prompt-token lookup"
    );
    assert_eq!(chat_b_single["usage"], batch["completions"][3]["usage"]);
    assert_eq!(chat_b_single["choices"][0]["finish_reason"], "length");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn multi_prompt_batch_populates_chat_choices_cache_before_chat_work() {
    let state = make_batch_test_state();
    let batch_body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"batch populates choices prompt a"}],
            [{"role":"user","content":"batch populates choices prompt b"}]
        ],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 800
    })
    .to_string();
    let chat_b_body = serde_json::json!({
        "messages": [{"role":"user","content":"batch populates choices prompt b"}],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 803
    })
    .to_string();

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(batch["completions"].as_array().unwrap().len(), 6);
    let generated_after_batch = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_batch > 0);
    let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_batch, (0, 2, 2));
    assert_eq!(token_stats_after_batch, (0, 2, 2));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(
        state.chat_choices_cache.lock().unwrap().stats(),
        2,
        "multi-prompt batch should populate one chat choices entry per prompt"
    );

    let (status_chat, chat_b) = chat_post(state.clone(), &chat_b_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_b}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_batch,
        "equivalent chat n request should hit choices cache populated by multi-prompt batch"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_batch,
        "batch-populated chat choices hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_batch,
        "batch-populated chat choices hit should return before prompt-token lookup"
    );
    assert_eq!(
        state.chat_choices_cache.lock().unwrap().stats(),
        2,
        "chat hit should not need to create a new choices-cache entry"
    );
    assert_eq!(
        chat_b["usage"]["prompt_tokens"].as_u64().unwrap(),
        batch["completions"][3]["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap()
    );
    assert_eq!(
        chat_b["usage"]["completion_tokens"].as_u64().unwrap(),
        batch["completions"].as_array().unwrap()[3..6]
            .iter()
            .map(|item| item["usage"]["completion_tokens"].as_u64().unwrap())
            .sum::<u64>()
    );
    for (choice, completion) in chat_b["choices"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&batch["completions"].as_array().unwrap()[3..6])
    {
        assert_eq!(choice["message"]["content"], completion["text"]);
        assert_eq!(choice["finish_reason"], completion["finish_reason"]);
    }
}

#[tokio::test]
async fn chat_hits_request_cache_populated_by_single_output_batch() {
    let state = make_batch_test_state();
    let batch_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"cross endpoint batch to chat"}]],
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let chat_body = serde_json::json!({
        "messages": [{"role":"user","content":"cross endpoint batch to chat"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    let generated_after_batch = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_batch > 0);
    let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_batch, (0, 1, 1));
    assert_eq!(token_stats_after_batch, (0, 1, 1));
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        1,
        "batch generation should populate the equivalent chat request cache"
    );

    let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_batch,
        "chat should reuse the request cache populated by the batch request"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_batch,
        "chat should hit request cache before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_batch,
        "chat should hit request cache before prompt-token lookup"
    );
    assert_eq!(batch["usage"], chat["usage"]);
    assert_eq!(
        batch["completions"][0]["text"],
        chat["choices"][0]["message"]["content"]
    );
}

#[tokio::test]
async fn multi_prompt_zero_batch_populates_chat_request_cache_before_chat_work() {
    let state = make_batch_test_state();
    let batch_body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"zero batch feeds chat prompt a"}],
            [{"role":"user","content":"zero batch feeds chat prompt b"}]
        ],
        "n": 1,
        "temperature": 0.7,
        "max_tokens": 0,
        "seed": 900
    })
    .to_string();
    let chat_b_body = serde_json::json!({
        "messages": [{"role":"user","content":"zero batch feeds chat prompt b"}],
        "temperature": 0.7,
        "max_tokens": 0,
        "seed": 901
    })
    .to_string();

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(batch["completions"].as_array().unwrap().len(), 2);
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_batch, (0, 2, 2));
    assert_eq!(token_stats_after_batch, (0, 2, 2));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        2,
        "zero-token multi-prompt batch should populate one chat request entry per prompt"
    );

    let (status_chat, chat_b) = chat_post(state.clone(), &chat_b_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_b}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "zero-token chat hit should not generate model tokens"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_batch,
        "batch-populated chat request hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_batch,
        "batch-populated chat request hit should return before prompt-token lookup"
    );
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        2,
        "chat hit should not need to create a new request-cache entry"
    );
    assert_eq!(
        chat_b["usage"]["prompt_tokens"],
        batch["completions"][1]["usage"]["prompt_tokens"]
    );
    assert_eq!(chat_b["usage"]["completion_tokens"], 0);
    assert_eq!(
        chat_b["choices"][0]["message"]["content"],
        batch["completions"][1]["text"]
    );
    assert_eq!(
        chat_b["choices"][0]["finish_reason"],
        batch["completions"][1]["finish_reason"]
    );
}

#[tokio::test]
async fn multi_output_zero_batch_populates_chat_request_cache_before_single_chat_work() {
    let state = make_batch_test_state();
    let batch_body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"zero batch n feeds single chat prompt a"}],
            [{"role":"user","content":"zero batch n feeds single chat prompt b"}]
        ],
        "n": 3,
        "temperature": 0.7,
        "max_tokens": 0,
        "seed": 950
    })
    .to_string();
    let chat_b_body = serde_json::json!({
        "messages": [{"role":"user","content":"zero batch n feeds single chat prompt b"}],
        "temperature": 0.7,
        "max_tokens": 0,
        "seed": 955
    })
    .to_string();

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(batch["completions"].as_array().unwrap().len(), 6);
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_batch, (0, 2, 2));
    assert_eq!(token_stats_after_batch, (0, 2, 2));
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        2,
        "zero-token n>1 batch should populate one request entry per prompt after key normalization"
    );
    assert_eq!(
        state.chat_choices_cache.lock().unwrap().stats(),
        2,
        "zero-token n>1 batch should still populate one choices entry per prompt"
    );

    let (status_chat, chat_b) = chat_post(state.clone(), &chat_b_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_b}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "zero-token chat hit should not generate model tokens"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_batch,
        "batch-populated request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_batch,
        "batch-populated request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 2);
    assert_eq!(
        chat_b["usage"]["prompt_tokens"],
        batch["completions"][3]["usage"]["prompt_tokens"]
    );
    assert_eq!(
        chat_b["choices"][0]["message"]["content"],
        batch["completions"][3]["text"]
    );
    assert_eq!(
        chat_b["choices"][0]["finish_reason"],
        batch["completions"][3]["finish_reason"]
    );
}

#[tokio::test]
async fn greedy_multi_output_batch_clones_cached_chat_response_before_prompt_work() {
    let state = make_batch_test_state();
    let chat_body = serde_json::json!({
        "messages": [{"role":"user","content":"cached chat fans out to greedy batch"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let batch_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"cached chat fans out to greedy batch"}]],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
    let generated_after_chat = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_chat > 0);
    let render_stats_after_chat = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_chat = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_chat, (0, 1, 1));
    assert_eq!(token_stats_after_chat, (0, 1, 1));

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_chat,
        "greedy n>1 batch should clone the cached chat response without model work"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_chat,
        "greedy n>1 batch should hit chat request cache before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_chat,
        "greedy n>1 batch should hit chat request cache before prompt-token lookup"
    );

    let completions = batch["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 4);
    for completion in completions {
        assert_eq!(
            completion["text"], chat["choices"][0]["message"]["content"],
            "each logical greedy batch output should clone the cached chat content"
        );
        assert_eq!(completion["usage"], chat["usage"]);
    }
    assert_eq!(
        batch["usage"]["prompt_tokens"],
        4 * chat["usage"]["prompt_tokens"].as_u64().unwrap()
    );
    assert_eq!(
        batch["usage"]["completion_tokens"],
        4 * chat["usage"]["completion_tokens"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn chat_hits_request_cache_populated_by_greedy_multi_output_batch() {
    let state = make_batch_test_state();
    let batch_body = serde_json::json!({
        "prompts": [[{"role":"user","content":"cached greedy batch feeds chat"}]],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let chat_body = serde_json::json!({
        "messages": [{"role":"user","content":"cached greedy batch feeds chat"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
    let generated_after_batch = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_batch > 0);
    let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_batch, (0, 1, 1));
    assert_eq!(token_stats_after_batch, (0, 1, 1));
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        1,
        "the one physical greedy generation should populate the equivalent chat cache"
    );

    let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_batch,
        "chat should reuse the multi-output batch's physical cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_batch,
        "chat should hit request cache before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_batch,
        "chat should hit request cache before prompt-token lookup"
    );

    let completions = batch["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 4);
    assert_eq!(
        completions[0]["text"],
        chat["choices"][0]["message"]["content"]
    );
    assert_eq!(completions[0]["usage"], chat["usage"]);
    assert_eq!(
        batch["usage"]["prompt_tokens"],
        4 * chat["usage"]["prompt_tokens"].as_u64().unwrap()
    );
    assert_eq!(
        batch["usage"]["completion_tokens"],
        4 * chat["usage"]["completion_tokens"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn repeated_multi_output_zero_batch_hits_batch_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"batch cache zero one"}],
            [{"role":"user","content":"batch cache zero two"}]
        ],
        "n": 2,
        "temperature": 0.7,
        "max_tokens": 0
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 2, 2));
    assert_eq!(token_stats_after_first, (0, 2, 2));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = batch_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch cache hit should return before rendered-prompt cache lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch cache hit should return before prompt-token cache lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
    assert_ne!(
        first["id"], second["id"],
        "cached batch responses should still get a fresh batch id"
    );
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn batch_cache_hit_rehydrates_chat_request_cache_before_chat_work() {
    let state = make_batch_test_state();
    let batch_body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"batch hit rehydrates request one"}],
            [{"role":"user","content":"batch hit rehydrates request two"}]
        ],
        "n": 1,
        "temperature": 0.7,
        "max_tokens": 0,
        "seed": 187
    })
    .to_string();
    let chat_two_body = serde_json::json!({
        "messages": [{"role":"user","content":"batch hit rehydrates request two"}],
        "temperature": 0.2,
        "top_p": 0.1,
        "max_tokens": 0,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 2, 2));
    assert_eq!(token_stats_after_first, (0, 2, 2));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 2);

    *state.chat_request_cache.lock().unwrap() = DeterministicChatRequestCache::new(128);
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

    let (status_second, second) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch cache rehydration should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch cache rehydration should return before prompt-token lookup"
    );
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        2,
        "batch cache hit should rehydrate one request-cache entry per prompt"
    );

    let (status_chat, chat_two) = chat_post(state.clone(), &chat_two_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_two}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "rehydrated request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "rehydrated request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(chat_two["usage"], second["completions"][1]["usage"]);
    assert_eq!(chat_two["choices"][0]["finish_reason"], "length");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn batch_cache_hit_rehydrates_chat_choices_cache_before_chat_n_work() {
    let state = make_batch_test_state();
    let batch_body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"batch hit rehydrates choices one"}],
            [{"role":"user","content":"batch hit rehydrates choices two"}]
        ],
        "n": 3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 188
    })
    .to_string();
    let chat_two_body = serde_json::json!({
        "messages": [{"role":"user","content":"batch hit rehydrates choices two"}],
        "n": 3,
        "temperature": 0.2,
        "top_p": 0.1,
        "max_tokens": 0,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    assert_eq!(first["completions"].as_array().unwrap().len(), 6);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 2, 2));
    assert_eq!(token_stats_after_first, (0, 2, 2));
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 2);

    *state.chat_choices_cache.lock().unwrap() = DeterministicChatChoicesCache::new(64);
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 0);

    let (status_second, second) = batch_post(state.clone(), &batch_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "batch cache choices rehydration should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "batch cache choices rehydration should return before prompt-token lookup"
    );
    assert_eq!(
        state.chat_choices_cache.lock().unwrap().stats(),
        2,
        "batch cache hit should rehydrate one choices-cache entry per prompt"
    );

    let (status_chat, chat_two) = chat_post(state.clone(), &chat_two_body).await;
    assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_two}");
    assert_eq!(chat_two["choices"].as_array().unwrap().len(), 3);
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "rehydrated choices-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "rehydrated choices-cache hit should return before prompt-token lookup"
    );
    assert_eq!(
        chat_two["usage"]["prompt_tokens"],
        second["completions"][3]["usage"]["prompt_tokens"]
    );
    assert_eq!(
        chat_two["usage"]["completion_tokens"].as_u64().unwrap(),
        second["completions"].as_array().unwrap()[3..6]
            .iter()
            .map(|item| item["usage"]["completion_tokens"].as_u64().unwrap())
            .sum::<u64>()
    );
}

#[tokio::test]
async fn concurrent_multi_output_greedy_batch_singleflights_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"batch singleflight one"}],
            [{"role":"user","content":"batch singleflight two"}]
        ],
        "n": 2,
        "temperature": 0.0,
        "max_tokens": 2
    })
    .to_string();

    let (first, second) = tokio::join!(
        batch_post(state.clone(), &body),
        batch_post(state.clone(), &body)
    );
    let (status_first, first) = first;
    let (status_second, second) = second;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        (0, 2, 2),
        "concurrent duplicate batch should do prompt rendering once per unique prompt"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        (0, 2, 2),
        "concurrent duplicate batch should tokenize once per unique prompt"
    );
    assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["completions"], second["completions"]);
    assert_ne!(
        first["id"], second["id"],
        "singleflight replay should still get a fresh batch id"
    );
}

#[tokio::test]
async fn batch_repeated_seeded_sampled_request_hits_completion_cache() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"cache seeded sample"}]],
        "n": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_first, first) = batch_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);

    let (status_second, second) = batch_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "second identical seeded sampled batch request should be served from full completion cache"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(
        first["completions"][0]["text"],
        second["completions"][0]["text"]
    );
}

#[tokio::test]
async fn chat_repeated_greedy_request_hits_completion_cache() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"cache me once"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123,
        "thinking_budget_tokens": 64,
        "thinking_budget_ms": null
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);

    let (status_second, second) = chat_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "second identical greedy chat request should be served from full completion cache"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(
        first["choices"][0]["message"]["content"],
        second["choices"][0]["message"]["content"]
    );
    assert_ne!(
        first["id"], second["id"],
        "cached chat responses should still get a fresh response id"
    );
    let recent = state.recent_requests.lock().unwrap().snapshot();
    assert_eq!(recent.len(), 2);
    assert!(recent.iter().all(|record| {
        record.thinking_budget.as_ref().is_some_and(|budget| {
            budget.configured
                && budget.max_tokens == Some(64)
                && budget.tokens_source == "request"
                && budget.time_source == "request_unlimited"
                && budget.applied == Some(false)
        })
    }));
}

#[tokio::test]
async fn chat_repeated_seeded_sampled_request_hits_completion_cache() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"cache sampled once"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);

    let (status_second, second) = chat_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "second identical seeded sampled chat request should be served from full completion cache"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(
        first["choices"][0]["message"]["content"],
        second["choices"][0]["message"]["content"]
    );
}

#[tokio::test]
async fn chat_repeated_unseeded_sampled_request_does_not_use_completion_cache() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"do not cache random sample"}],
        "temperature": 0.7,
        "max_tokens": 4
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);

    let (status_second, second) = chat_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed)
            > generated_after_first,
        "unseeded sampled requests should keep performing physical generation"
    );
}

#[tokio::test]
async fn chat_streaming_repeated_greedy_request_uses_completion_cache() {
    let state = make_batch_test_state();
    let base = serde_json::json!({
        "messages": [{"role":"user","content":"cache me as a stream"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    });

    let (status_first, first) = chat_post(state.clone(), &base.to_string()).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let mut stream_body_json = base;
    stream_body_json["stream"] = serde_json::Value::Bool(true);
    let (status_second, body) = chat_post_text(state.clone(), &stream_body_json.to_string()).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{body}");
    assert!(
        body.contains("chat.completion.chunk") && body.contains("[DONE]"),
        "cached streaming response should be SSE-shaped: {body}"
    );
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "cached streaming response should not perform model generation"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "streaming request-cache hit should return before rendered-prompt cache lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "streaming request-cache hit should return before prompt-token cache lookup"
    );
    assert_eq!(state.recent_requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn chat_streaming_completion_cache_hit_populates_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base = serde_json::json!({
        "messages": [{"role":"user","content":"stream lower cache should promote"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    });
    let req = parse_request(&base.to_string());
    let sampling = SamplingParams {
        temperature: requested_or_default_temperature(req.temperature),
        top_p: requested_or_default_top_p(req.top_p),
        top_k: requested_or_default_top_k(req.top_k),
        max_tokens: chat_request_max_tokens(&req),
        stop: normalized_stop_for_generation(req.stop.as_deref()),
        seed: req.seed,
        ..Default::default()
    };
    let prompt_text = render_prompt_text(
        &state,
        &req.messages,
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        req.chat_template_kwargs.as_ref(),
    )
    .unwrap();
    let prompt_tokens = encode_prompt_tokens(&state, &prompt_text).unwrap();
    let completion_cache_key =
        deterministic_completion_cache_key(&state, &prompt_tokens, &sampling, false)
            .expect("greedy request should be deterministic");
    state
        .completion_cache
        .lock()
        .unwrap()
        .insert_complete_value(
            completion_cache_key,
            DeterministicCompletionCacheValue {
                text: "cached lower-layer stream".to_string(),
                reasoning_content: None,
                tool_calls: None,
                finish_reason: "stop".to_string(),
                completion_tokens: 3,
                thinking_budget_status: None,
            },
        );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        (0, 1, 1)
    );
    assert_eq!(state.prompt_token_cache.lock().unwrap().stats(), (0, 1, 1));
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

    let mut stream_body = base.clone();
    stream_body["stream"] = serde_json::Value::Bool(true);
    let (status_stream, stream) = chat_post_text(state.clone(), &stream_body.to_string()).await;
    assert_eq!(status_stream, axum::http::StatusCode::OK, "{stream}");
    assert!(stream.contains("cached lower-layer stream"));
    assert!(stream.contains("[DONE]"));
    let render_stats_after_stream = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_stream = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(
        render_stats_after_stream,
        (1, 1, 1),
        "streaming lower-cache hit still has to render before promotion"
    );
    assert_eq!(
        token_stats_after_stream,
        (1, 1, 1),
        "streaming lower-cache hit still has to tokenize before promotion"
    );
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        1,
        "streaming completion-cache hit should promote to chat request cache"
    );

    let (status_second, second) = chat_post(state.clone(), &base.to_string()).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        second["choices"][0]["message"]["content"],
        "cached lower-layer stream"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_stream,
        "promoted request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_stream,
        "promoted request-cache hit should return before prompt-token lookup"
    );
}

#[tokio::test]
async fn chat_zero_max_tokens_returns_without_generation() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"do not decode"}],
        "temperature": 0.7,
        "max_tokens": 0
    })
    .to_string();

    let (status, body) = chat_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 0);
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "max_tokens=0 should not enter model generation"
    );
    assert_eq!(state.recent_requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn recent_request_and_metrics_expose_effective_inert_budget() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"bounded but no output"}],
        "max_tokens": 0,
        "thinking_budget_tokens": 64,
        "thinking_budget_ms": null
    })
    .to_string();

    let (status, response) = chat_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{response}");
    let recent = state.recent_requests.lock().unwrap().snapshot();
    let budget = recent[0].thinking_budget.as_ref().unwrap();
    assert!(budget.configured);
    assert_eq!(budget.max_tokens, Some(64));
    assert_eq!(budget.max_time_ms, None);
    assert_eq!(budget.tokens_source, "request");
    assert_eq!(budget.time_source, "request_unlimited");
    assert_eq!(budget.applied, Some(false));
    assert_eq!(budget.triggered, None);
    assert_eq!(budget.closed, None);

    let body = metrics_get_text(state).await;
    assert!(
        body.contains(
            "kiln_thinking_budget_source_total{dimension=\"tokens\",source=\"request\"} 1"
        )
    );
    assert!(body.contains(
        "kiln_thinking_budget_source_total{dimension=\"time\",source=\"request_unlimited\"} 1"
    ));
    assert!(body.contains("kiln_thinking_budget_outcomes_total{outcome=\"inert\"} 1"));
    assert!(body.contains("kiln_thinking_budget_effective_tokens_count 1"));
    assert!(body.contains("kiln_thinking_budget_effective_tokens_sum 64"));
    assert!(body.contains("kiln_thinking_budget_effective_seconds_count 0"));
}

#[tokio::test]
async fn streaming_setup_failure_records_applied_budget_as_interrupted() {
    let mut state = make_qwen_template_test_state();
    let template =
        include_str!("../../../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
    let tokenizer = kiln_core::tokenizer::KilnTokenizer::from_bytes(
        br#"{
                "version":"1.0",
                "model":{
                    "type":"WordLevel",
                    "vocab":{"</think>":0,"<unk>":1},
                    "unk_token":"<unk>"
                }
            }"#,
    )
    .unwrap()
    .with_chat_template(template.to_string());
    state.tokenizer = std::sync::Arc::new(tokenizer);
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"stream from mock"}],
        "stream": true,
        "max_tokens": 64,
        "thinking_budget_tokens": 32,
        "thinking_budget_ms": null
    })
    .to_string();

    let (status, response) = chat_post(state.clone(), &body).await;
    assert_eq!(
        status,
        axum::http::StatusCode::NOT_IMPLEMENTED,
        "{response}"
    );
    assert_eq!(response["error"]["code"], "streaming_not_supported");
    let recent = state.recent_requests.lock().unwrap().snapshot();
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].finish_reason, "error");
    let budget = recent[0].thinking_budget.as_ref().unwrap();
    assert!(budget.configured);
    assert_eq!(budget.max_tokens, Some(32));
    assert_eq!(budget.applied, Some(true));
    assert_eq!(budget.triggered, None);
    assert_eq!(budget.closed, None);

    let metrics = metrics_get_text(state).await;
    assert!(metrics.contains("kiln_thinking_budget_outcomes_total{outcome=\"interrupted\"} 1"));
}

#[tokio::test]
async fn chat_zero_max_completion_tokens_alias_returns_without_generation() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"do not decode alias"}],
        "temperature": 0.7,
        "max_completion_tokens": 0
    })
    .to_string();

    let (status, body) = chat_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 0);
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "max_completion_tokens=0 should not enter model generation"
    );
    assert_eq!(state.recent_requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn repeated_zero_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"request cache avoids prompt work"}],
        "temperature": 0.7,
        "max_tokens": 0
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "chat request cache hit should return before rendered-prompt cache lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "chat request cache hit should return before prompt-token cache lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
    assert_ne!(
        first["id"], second["id"],
        "cached chat request should still get a fresh response id"
    );
}

#[tokio::test]
async fn multi_choice_zero_chat_populates_request_cache_before_single_chat_work() {
    let state = make_batch_test_state();
    let multi_body = serde_json::json!({
        "messages": [{"role":"user","content":"zero chat n feeds single chat"}],
        "n": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 100
    })
    .to_string();
    let single_body = serde_json::json!({
        "messages": [{"role":"user","content":"zero chat n feeds single chat"}],
        "temperature": 0.2,
        "top_p": 0.1,
        "max_tokens": 0,
        "seed": 999
    })
    .to_string();

    let (status_multi, multi) = chat_post(state.clone(), &multi_body).await;
    assert_eq!(status_multi, axum::http::StatusCode::OK, "{multi}");
    assert_eq!(multi["choices"].as_array().unwrap().len(), 4);
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let render_stats_after_multi = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_multi = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_multi, (0, 1, 1));
    assert_eq!(token_stats_after_multi, (0, 1, 1));
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        1,
        "zero-token chat n should populate one normalized request-cache entry"
    );
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

    let (status_single, single) = chat_post(state.clone(), &single_body).await;
    assert_eq!(status_single, axum::http::StatusCode::OK, "{single}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "zero-token single chat hit should not generate model tokens"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_multi,
        "chat n-populated request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_multi,
        "chat n-populated request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
    assert_eq!(multi["usage"], single["usage"]);
    assert_eq!(multi["choices"][0], single["choices"][0]);
}

#[tokio::test]
async fn chat_choices_cache_hit_rehydrates_request_cache_before_single_chat_work() {
    let state = make_batch_test_state();
    let multi_body = serde_json::json!({
        "messages": [{"role":"user","content":"choices hit feeds single chat"}],
        "n": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 199
    })
    .to_string();
    let single_body = serde_json::json!({
        "messages": [{"role":"user","content":"choices hit feeds single chat"}],
        "temperature": 0.2,
        "top_p": 0.1,
        "max_tokens": 0,
        "seed": 999
    })
    .to_string();

    let (status_multi, multi) = chat_post(state.clone(), &multi_body).await;
    assert_eq!(status_multi, axum::http::StatusCode::OK, "{multi}");
    assert_eq!(multi["choices"].as_array().unwrap().len(), 4);
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let render_stats_after_populate = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_populate = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_populate, (0, 1, 1));
    assert_eq!(token_stats_after_populate, (0, 1, 1));
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    *state.chat_request_cache.lock().unwrap() = DeterministicChatRequestCache::new(128);
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

    let (status_hit, hit) = chat_post(state.clone(), &multi_body).await;
    assert_eq!(status_hit, axum::http::StatusCode::OK, "{hit}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_populate,
        "choices-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_populate,
        "choices-cache hit should return before prompt-token lookup"
    );
    assert_eq!(multi["usage"], hit["usage"]);
    assert_eq!(multi["choices"], hit["choices"]);
    assert_eq!(
        state.chat_request_cache.lock().unwrap().stats(),
        1,
        "choices-cache hit should rehydrate normalized request-cache entries"
    );

    let (status_single, single) = chat_post(state.clone(), &single_body).await;
    assert_eq!(status_single, axum::http::StatusCode::OK, "{single}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_populate,
        "rehydrated request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_populate,
        "rehydrated request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "zero-token single chat hit should not generate model tokens"
    );
    assert_eq!(hit["usage"], single["usage"]);
    assert_eq!(hit["choices"][0], single["choices"][0]);
}

#[tokio::test]
async fn multi_choice_zero_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let single_body = serde_json::json!({
        "messages": [{"role":"user","content":"single chat feeds zero chat n"}],
        "temperature": 0.2,
        "top_p": 0.1,
        "max_tokens": 0,
        "seed": 999
    })
    .to_string();
    let multi_body = serde_json::json!({
        "messages": [{"role":"user","content":"single chat feeds zero chat n"}],
        "n": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 0,
        "seed": 202
    })
    .to_string();

    let (status_single, single) = chat_post(state.clone(), &single_body).await;
    assert_eq!(status_single, axum::http::StatusCode::OK, "{single}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let render_stats_after_single = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_single = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_single, (0, 1, 1));
    assert_eq!(token_stats_after_single, (0, 1, 1));
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 0);

    let (status_multi, multi) = chat_post(state.clone(), &multi_body).await;
    assert_eq!(status_multi, axum::http::StatusCode::OK, "{multi}");
    assert_eq!(multi["choices"].as_array().unwrap().len(), 4);
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_single,
        "zero-token chat n should hit request cache before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_single,
        "zero-token chat n should hit request cache before prompt-token lookup"
    );
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "zero-token chat n request-cache hit should not generate model tokens"
    );
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
    assert_eq!(single["usage"], multi["usage"]);
    assert_eq!(single["choices"][0], multi["choices"][0]);

    let (status_repeat, repeat) = chat_post(state.clone(), &multi_body).await;
    assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_single,
        "request-cache synthesized choices should seed the choices cache"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_single,
        "choices-cache repeat should still avoid prompt-token lookup"
    );
    assert_eq!(multi["usage"], repeat["usage"]);
    assert_eq!(multi["choices"], repeat["choices"]);
}

#[tokio::test]
async fn top_k_one_chat_hits_greedy_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let greedy_body = serde_json::json!({
        "messages": [{"role":"user","content":"top k one request cache"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 1
    })
    .to_string();
    let top_k_one_body = serde_json::json!({
        "messages": [{"role":"user","content":"top k one request cache"}],
        "temperature": 0.7,
        "top_p": 0.2,
        "top_k": 1,
        "max_tokens": 4
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &greedy_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &top_k_one_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "top_k=1 should hit the equivalent greedy request cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_k=1 request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_k=1 request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn top_p_above_one_chat_hits_seeded_full_distribution_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"top p full distribution request cache"}],
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 4,
        "seed": 140
    })
    .to_string();
    let repeat_body = serde_json::json!({
        "messages": [{"role":"user","content":"top p full distribution request cache"}],
        "temperature": 0.7,
        "top_p": 1.5,
        "max_tokens": 4,
        "seed": 140
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &repeat_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "top_p >= 1.0 seeded chat repeat should hit the equivalent request cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_p >= 1.0 request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_p >= 1.0 request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn top_p_zero_chat_hits_seeded_full_distribution_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"top p zero full distribution request cache"}],
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 4,
        "seed": 144
    })
    .to_string();
    let repeat_body = serde_json::json!({
        "messages": [{"role":"user","content":"top p zero full distribution request cache"}],
        "temperature": 0.7,
        "top_p": 0.0,
        "max_tokens": 4,
        "seed": 144
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &repeat_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "top_p=0 seeded chat repeat should hit the equivalent request cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_p=0 request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_p=0 request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn top_k_oversized_chat_hits_seeded_full_distribution_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let disabled_top_k = state.model_config.vocab_size as u32;
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"top k oversized full distribution request cache"}],
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": disabled_top_k,
        "max_tokens": 4,
        "seed": 148
    })
    .to_string();
    let repeat_body = serde_json::json!({
        "messages": [{"role":"user","content":"top k oversized full distribution request cache"}],
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 4,
        "seed": 148
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &repeat_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "top_k >= vocab seeded chat repeat should hit the equivalent request cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "top_k >= vocab request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "top_k >= vocab request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn empty_tools_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [{"role":"user","content":"empty tools should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let empty_tools_body = serde_json::json!({
        "messages": [{"role":"user","content":"empty tools should be no-op"}],
        "tools": [],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &empty_tools_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "empty-tools request should reuse the no-tools cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "empty-tools request cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "empty-tools request cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn no_tool_auto_choice_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [{"role":"user","content":"auto tool choice should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let no_op_choice_body = serde_json::json!({
        "messages": [{"role":"user","content":"auto tool choice should be no-op"}],
        "tools": [],
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &no_op_choice_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "no-tool auto choice should reuse the no-tools cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "no-tool auto choice request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "no-tool auto choice request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn no_tool_none_choice_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [{"role":"user","content":"none tool choice should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let no_op_choice_body = serde_json::json!({
        "messages": [{"role":"user","content":"none tool choice should be no-op"}],
        "tools": [],
        "tool_choice": "none",
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &no_op_choice_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "no-tool none choice should reuse the no-tools cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "no-tool none choice request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "no-tool none choice request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn no_tool_none_choice_chat_multi_choice_hits_choices_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [{"role":"user","content":"none tool choice chat n should be no-op"}],
        "n": 4,
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let no_op_choice_body = serde_json::json!({
        "messages": [{"role":"user","content":"none tool choice chat n should be no-op"}],
        "n": 4,
        "tools": [],
        "tool_choice": "none",
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &no_op_choice_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "no-tool none choice should reuse the no-tools chat choices cache"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "no-tool none choices-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "no-tool none choices-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn input_reasoning_content_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [{"role":"user","content":"input reasoning content should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let reasoning_body = serde_json::json!({
        "messages": [{
            "role":"user",
            "content":"input reasoning content should be no-op",
            "reasoning_content":"ignored by prompt rendering"
        }],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &reasoning_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "input reasoning_content should reuse the rendered-equivalent cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "input reasoning_content request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "input reasoning_content request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn text_content_parts_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [{"role":"user","content":"text content parts should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let parts_body = serde_json::json!({
        "messages": [{
            "role":"user",
            "content":[
                {"type":"text","text":"text content parts "},
                {"type":"text","text":"should be no-op"}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &parts_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "equivalent text content parts should reuse the cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "text content parts request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "text content parts request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn non_text_content_parts_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [{"role":"user","content":"non-text content parts should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let parts_body = serde_json::json!({
        "messages": [{
            "role":"user",
            "content":[
                {"type":"text","text":"non-text content parts "},
                {"type":"image_url","image_url":{"url":"https://example.invalid/ignored.png"}},
                {"type":"input_audio","input_audio":{"data":"ignored","format":"wav"}},
                {"type":"text","text":"should be no-op"}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &parts_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "non-text content parts should reuse the cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "non-text content parts request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "non-text content parts request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn max_completion_tokens_alias_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let max_tokens_body = serde_json::json!({
        "messages": [{"role":"user","content":"max completion tokens alias should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let alias_body = serde_json::json!({
        "messages": [{"role":"user","content":"max completion tokens alias should be no-op"}],
        "temperature": 0.0,
        "max_completion_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &max_tokens_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &alias_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "max_completion_tokens should reuse the max_tokens cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "max_completion_tokens request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "max_completion_tokens request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn stop_string_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let list_body = serde_json::json!({
        "messages": [{"role":"user","content":"stop string should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["never-match-stop"],
        "seed": 123
    })
    .to_string();
    let string_body = serde_json::json!({
        "messages": [{"role":"user","content":"stop string should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": "never-match-stop",
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &list_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &string_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "single-string stop should reuse the one-item list cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "stop-string request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "stop-string request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn dominated_stop_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let redundant_body = serde_json::json!({
        "messages": [{"role":"user","content":"dominated stop should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["never-match-stop", "never-match-stop-suffix"],
        "seed": 123
    })
    .to_string();
    let minimal_body = serde_json::json!({
        "messages": [{"role":"user","content":"dominated stop should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["never-match-stop"],
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &redundant_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &minimal_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "dominated stop sequences should reuse the minimal stop cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "dominated-stop request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "dominated-stop request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn substring_dominated_stop_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let redundant_body = serde_json::json!({
        "messages": [{"role":"user","content":"substring dominated stop should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["prefix-never-match-stop-suffix", "never-match-stop"],
        "seed": 123
    })
    .to_string();
    let minimal_body = serde_json::json!({
        "messages": [{"role":"user","content":"substring dominated stop should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "stop": ["never-match-stop"],
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &redundant_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

    let (status_second, second) = chat_post(state.clone(), &minimal_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "substring-dominated stop sequences should reuse the minimal stop cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "substring-dominated-stop request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "substring-dominated-stop request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn default_openai_option_fields_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let plain_body = serde_json::json!({
        "messages": [{"role":"user","content":"default OpenAI options should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let defaults_body = serde_json::json!({
        "messages": [{"role":"user","content":"default OpenAI options should be no-op"}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999,
        "n": 1,
        "response_format": {"type":"text"},
        "parallel_tool_calls": true,
        "user": "client-a",
        "metadata": {"trace_id":"ignored"},
        "store": false,
        "service_tier": "auto",
        "logprobs": false,
        "top_logprobs": 0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream_options": {"include_usage": false}
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &plain_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &defaults_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "default OpenAI options should reuse the cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "default-option request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "default-option request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn empty_message_tool_calls_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [
            {"role":"user","content":"empty message tool calls should be no-op"},
            {"role":"assistant","content":"ok"},
            {"role":"user","content":"continue"}
        ],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let empty_tool_calls_body = serde_json::json!({
        "messages": [
            {"role":"user","content":"empty message tool calls should be no-op"},
            {"role":"assistant","content":"ok","tool_calls":[]},
            {"role":"user","content":"continue"}
        ],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &empty_tool_calls_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "empty message tool_calls should reuse the rendered-equivalent cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "empty message tool_calls request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "empty message tool_calls request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn tool_call_argument_json_string_chat_hits_request_cache_before_prompt_work() {
    let state = make_batch_test_state();
    let base_body = serde_json::json!({
        "messages": [
            {"role":"user","content":"tool call argument JSON should be canonical"},
            {"role":"assistant","content":null,"tool_calls":[{
                "id":"call_1",
                "type":"function",
                "function":{"name":"Lookup","arguments":"{\"query\":\"cache\",\"limit\":2}"}
            }]},
            {"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},
            {"role":"user","content":"continue"}
        ],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 123
    })
    .to_string();
    let whitespace_args_body = serde_json::json!({
        "messages": [
            {"role":"user","content":"tool call argument JSON should be canonical"},
            {"role":"assistant","content":null,"tool_calls":[{
                "id":"call_1",
                "type":"function",
                "function":{"name":"Lookup","arguments":"{ \"limit\" : 2, \"query\" : \"cache\" }"}
            }]},
            {"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},
            {"role":"user","content":"continue"}
        ],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 999
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &base_body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let generated_after_first = state
        .metrics
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(generated_after_first > 0);
    let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
    let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(render_stats_after_first, (0, 1, 1));
    assert_eq!(token_stats_after_first, (0, 1, 1));

    let (status_second, second) = chat_post(state.clone(), &whitespace_args_body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        generated_after_first,
        "JSON-equivalent tool_call arguments should reuse the cached response"
    );
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        render_stats_after_first,
        "tool_call argument JSON request-cache hit should return before rendered-prompt lookup"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        token_stats_after_first,
        "tool_call argument JSON request-cache hit should return before prompt-token lookup"
    );
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
}

#[tokio::test]
async fn concurrent_zero_chat_singleflights_before_prompt_work() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"chat request singleflight"}],
        "temperature": 0.0,
        "max_tokens": 0
    })
    .to_string();

    let (first, second) = tokio::join!(
        chat_post(state.clone(), &body),
        chat_post(state.clone(), &body)
    );
    let (status_first, first) = first;
    let (status_second, second) = second;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
    assert_eq!(
        state.rendered_prompt_cache.lock().unwrap().stats(),
        (0, 1, 1),
        "concurrent identical chat requests should render once"
    );
    assert_eq!(
        state.prompt_token_cache.lock().unwrap().stats(),
        (0, 1, 1),
        "concurrent identical chat requests should tokenize once"
    );
    assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
    assert_eq!(first["usage"], second["usage"]);
    assert_eq!(first["choices"], second["choices"]);
    assert_ne!(first["id"], second["id"]);
}

#[tokio::test]
async fn repeated_chat_prompt_reuses_prompt_token_cache() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"tokenize this once"}],
        "temperature": 0.7,
        "max_tokens": 1
    })
    .to_string();

    let (status_first, first) = chat_post(state.clone(), &body).await;
    assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
    let (status_second, second) = chat_post(state.clone(), &body).await;
    assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");

    let (render_hits, render_misses, render_entries) =
        state.rendered_prompt_cache.lock().unwrap().stats();
    assert_eq!(
        render_misses, 1,
        "first chat request should miss rendered-prompt cache"
    );
    assert_eq!(
        render_hits, 1,
        "second identical chat request should hit rendered-prompt cache"
    );
    assert_eq!(render_entries, 1);

    let (token_hits, token_misses, token_entries) =
        state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(
        token_misses, 1,
        "first rendered prompt should miss token cache"
    );
    assert_eq!(
        token_hits, 1,
        "second identical rendered prompt should hit token cache"
    );
    assert_eq!(token_entries, 1);
}

#[tokio::test]
async fn chat_streaming_zero_max_tokens_returns_sse_without_generation() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "messages": [{"role":"user","content":"do not stream decode"}],
        "temperature": 0.0,
        "max_tokens": 0,
        "stream": true
    })
    .to_string();

    let (status, body) = chat_post_text(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("\"finish_reason\":\"length\""));
    assert!(body.contains("[DONE]"));
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "streaming max_tokens=0 should not enter model generation"
    );
    assert_eq!(state.recent_requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn batch_zero_max_tokens_returns_without_generation() {
    let mut state = make_batch_test_state();
    state.default_thinking_budget_tokens = Some(64);
    state.default_thinking_budget_ms = Some(1_000);
    let body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"zero one"}],
            [{"role":"user","content":"zero two"}]
        ],
        "n": 2,
        "temperature": 0.7,
        "max_tokens": 0,
        "thinking_budget_tokens": 0,
        "thinking_budget_ms": null
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let completions = body["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 4);
    assert!(completions.iter().all(|item| item["text"] == ""));
    assert!(
        completions
            .iter()
            .all(|item| item["finish_reason"] == "length")
    );
    assert_eq!(body["usage"]["completion_tokens"], 0);
    assert_eq!(
        body["metadata"]["thinking_budget"],
        serde_json::json!({
            "configured": true,
            "max_tokens": 0,
            "tokens_source": "request",
            "time_source": "request_unlimited"
        })
    );
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "batch max_tokens=0 should not enter model generation"
    );
}

#[tokio::test]
async fn batch_zero_max_completion_tokens_alias_returns_without_generation() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"zero alias one"}],
            [{"role":"user","content":"zero alias two"}]
        ],
        "n": 2,
        "temperature": 0.7,
        "max_completion_tokens": 0
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    let completions = body["completions"].as_array().unwrap();
    assert_eq!(completions.len(), 4);
    assert!(completions.iter().all(|item| item["text"] == ""));
    assert!(
        completions
            .iter()
            .all(|item| item["finish_reason"] == "length")
    );
    assert_eq!(body["usage"]["completion_tokens"], 0);
    assert_eq!(
        state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "batch max_completion_tokens=0 should not enter model generation"
    );
}

#[tokio::test]
async fn duplicate_batch_zero_prompts_skip_repeated_render_and_tokenize() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [
            [{"role":"user","content":"same token prompt"}],
            [{"role":"user","content":"same token prompt"}]
        ],
        "n": 1,
        "temperature": 0.7,
        "max_tokens": 0
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    assert_eq!(body["completions"].as_array().unwrap().len(), 2);
    assert_eq!(body["completions"][0]["prompt_index"], 0);
    assert_eq!(body["completions"][1]["prompt_index"], 1);

    let (render_hits, render_misses, render_entries) =
        state.rendered_prompt_cache.lock().unwrap().stats();
    assert_eq!(
        render_misses, 1,
        "duplicate zero-token batch prompts should render once"
    );
    assert_eq!(
        render_hits, 0,
        "duplicate zero-token batch prompts should skip repeated render lookups"
    );
    assert_eq!(render_entries, 1);

    let (token_hits, token_misses, token_entries) =
        state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(
        token_misses, 1,
        "duplicate zero-token batch prompts should tokenize once"
    );
    assert_eq!(
        token_hits, 0,
        "duplicate zero-token batch prompts should skip repeated token lookups"
    );
    assert_eq!(token_entries, 1);
}

#[tokio::test]
async fn batch_multi_sample_prepares_prompt_once_per_group() {
    let state = make_batch_test_state();
    let body = serde_json::json!({
        "prompts": [[{"role":"user","content":"same sampled prompt"}]],
        "n": 3,
        "temperature": 0.7,
        "max_tokens": 1,
        "seed": 282
    })
    .to_string();

    let (status, body) = batch_post(state.clone(), &body).await;
    assert_eq!(status, axum::http::StatusCode::OK, "{body}");
    assert_eq!(body["completions"].as_array().unwrap().len(), 3);

    let (render_hits, render_misses, render_entries) =
        state.rendered_prompt_cache.lock().unwrap().stats();
    assert_eq!(
        (render_hits, render_misses, render_entries),
        (0, 1, 1),
        "sampled n>1 batch should prepare the shared prompt once"
    );

    let (token_hits, token_misses, token_entries) =
        state.prompt_token_cache.lock().unwrap().stats();
    assert_eq!(
        (token_hits, token_misses, token_entries),
        (0, 1, 1),
        "sampled n>1 batch should tokenize the shared prompt once"
    );
}

#[test]
fn runtime_headers_preserve_the_revision_bound_to_the_response() {
    let state = make_batch_test_state();
    let old = Some(LoadedAdapterIdentity {
        name: "same-name".to_string(),
        content_revision: "old-revision".to_string(),
    });
    let response =
        response_with_loaded_adapter_identity(Response::new(axum::body::Body::empty()), &old);
    *state.loaded_adapter.write().unwrap() = Some(LoadedAdapterIdentity {
        name: "same-name".to_string(),
        content_revision: "new-revision".to_string(),
    });

    let response = response_with_runtime_headers(&state, response);
    assert_eq!(
        response
            .headers()
            .get("x-kiln-loaded-adapter-revision")
            .unwrap(),
        "old-revision"
    );
}

#[test]
fn cache_owner_key_match_includes_the_purge_generation() {
    let state = make_batch_test_state();
    let adapter = Some(LoadedAdapterIdentity {
        name: "same-name".to_string(),
        content_revision: "same-revision".to_string(),
    });
    let old_key = state.deterministic_cache_key(adapter.clone(), "request".to_string());
    let claim_id = match state.chat_request_cache.lock().unwrap().claim(&old_key) {
        DeterministicChatRequestCacheClaim::Owner(claim_id) => claim_id,
        _ => panic!("first request should own the cache claim"),
    };
    let owner =
        ChatRequestCacheOwnerGuard::new(state.chat_request_cache.clone(), old_key, claim_id);
    state.purge_adapter_caches(&Some("same-name".to_string()));
    let rebound = state.deterministic_cache_key(adapter, "request".to_string());

    assert!(!owner.matches_key(&rebound));
}

#[test]
fn batch_response_object_field_is_batch_completion() {
    // Lock down the discriminator string clients will key on so we don't
    // accidentally rename it.
    let resp = BatchCompletionResponse {
        id: "batchcmpl-test".to_string(),
        object: "batch.completion",
        created: 0,
        model: "kiln-test".to_string(),
        completions: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
        metadata: BatchCompletionMetadata::default(),
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["object"], "batch.completion");
}
