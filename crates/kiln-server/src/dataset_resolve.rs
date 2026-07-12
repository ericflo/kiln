//! Symbolic dataset selectors → concrete training prompts.
//!
//! The §10 agentic loop ("use pi all week, train on it Saturday") needs a
//! bridge between the §10.3 Agent Trace Layer index and the training
//! pipeline. This module is that bridge: it resolves the
//! `agent_traces:<filter>` selectors used by `/v1/agent/self_improve`,
//! `/v1/agent/judge_distill`, recipe `dataset:` sources, and OPD
//! `dataset_path` into `OpdPrompt`s built from the user's actual pi
//! sessions, and resolves bare names against the uploaded eval dataset
//! registry.
//!
//! Every resolution failure is actionable: it says which index/registry was
//! consulted, why zero prompts qualified, and what to run next — never a
//! placeholder corpus. (Before this module existed, an unresolvable
//! selector either failed at worker dequeue or silently fell through to a
//! three-generic-prompt seed bank, producing "judge" adapters trained on
//! "describe an interesting fact about this topic".)

use std::path::Path;

use kiln_train::ChatMessage;
use kiln_train::opd::OpdPrompt;
use kiln_train::trajectory::{TurnKind, TurnSegment};

use crate::api::agent_traces::{AgentTrace, AgentTraceIndex};

/// Selector scheme prefix for the agent-trace index.
pub const AGENT_TRACES_PREFIX: &str = "agent_traces:";

/// How far back `agent_traces:weekly` reaches.
const WEEKLY_WINDOW_MS: i64 = 7 * 24 * 3600 * 1000;
/// Judge-corpus caps: most-recent actions per trace, total prompts, and
/// character budgets so a single huge session can't produce a 100 MB
/// request body (the corpus is embedded into the queued pump request).
const MAX_JUDGE_TURNS_PER_TRACE: usize = 8;
const MAX_JUDGE_PROMPTS: usize = 256;
const JUDGE_CONTEXT_CHAR_BUDGET: usize = 3000;
const JUDGE_SEGMENT_CHAR_CAP: usize = 600;
const JUDGE_TURN_CHAR_CAP: usize = 2000;

/// Filters over the persisted agent-trace index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentTraceFilter {
    /// Every indexed trace with a usable prompt scaffold.
    All,
    /// Sessions that look like they went well: ended exit-0 (or unknown),
    /// not forked, no follow-up re-attempt.
    Successful,
    /// `Successful` restricted to the last 7 days — the
    /// `kiln self-improve` Saturday default.
    Weekly,
    /// (turn, context) judge-scoring prompts from *all* traces — failures
    /// and forks are exactly the discrimination signal a judge needs.
    JudgeTurns,
    /// `Successful` scaffolds re-prompted under conciseness pressure
    /// (§10.6.4 CRISP).
    Crisp,
}

pub fn is_agent_traces_selector(selector: &str) -> bool {
    selector.starts_with(AGENT_TRACES_PREFIX)
}

pub fn parse_agent_trace_filter(selector: &str) -> Result<AgentTraceFilter, String> {
    let suffix = selector.strip_prefix(AGENT_TRACES_PREFIX).ok_or_else(|| {
        format!("selector {selector:?} does not start with {AGENT_TRACES_PREFIX:?}")
    })?;
    match suffix {
        "all" => Ok(AgentTraceFilter::All),
        // `user-pi` is the spelling the day-one `learn-from-my-pi-history`
        // recipe shipped with — keep it working.
        "successful" | "user-pi" => Ok(AgentTraceFilter::Successful),
        "weekly" => Ok(AgentTraceFilter::Weekly),
        "judge_turns" => Ok(AgentTraceFilter::JudgeTurns),
        "crisp" => Ok(AgentTraceFilter::Crisp),
        other => Err(format!(
            "unknown agent_traces filter {other:?} — valid filters: all, successful, \
             user-pi, weekly, judge_turns, crisp"
        )),
    }
}

/// Resolve an `agent_traces:<filter>` selector against the index persisted
/// at `<adapter_dir>/agent_traces.json`. Errors are user-facing and carry
/// the remediation.
pub fn resolve_agent_trace_prompts(
    adapter_dir: &Path,
    selector: &str,
    now_unix_ms: i64,
) -> Result<Vec<OpdPrompt>, String> {
    let filter = parse_agent_trace_filter(selector)?;
    let index_path = adapter_dir.join("agent_traces.json");
    if !index_path.is_file() {
        return Err(format!(
            "no agent-trace index at {} — run `POST /v1/agent/traces/discover` (or open \
             the dashboard's Agent Traces panel) to index your pi sessions first",
            index_path.display()
        ));
    }
    let index = AgentTraceIndex::load_from_path(&index_path);
    if index.traces.is_empty() {
        return Err(format!(
            "agent-trace index at {} is empty — use pi pointed at this Kiln (try `kiln \
             pi-setup` or the dashboard's pi terminal), then re-run `POST \
             /v1/agent/traces/discover`",
            index_path.display()
        ));
    }
    let total = index.traces.len();

    // Most-recent-first so prompt caps keep the freshest sessions.
    let mut traces: Vec<&AgentTrace> = index.traces.values().collect();
    traces.sort_by(|a, b| b.last_event_at.cmp(&a.last_event_at));

    let prompts = match filter {
        AgentTraceFilter::All => scaffold_prompts(traces.iter().copied()),
        AgentTraceFilter::Successful => {
            scaffold_prompts(traces.iter().copied().filter(|t| trace_is_successful(t)))
        }
        AgentTraceFilter::Weekly => scaffold_prompts(traces.iter().copied().filter(|t| {
            trace_is_successful(t) && trace_within_ms(t, now_unix_ms, WEEKLY_WINDOW_MS)
        })),
        AgentTraceFilter::JudgeTurns => judge_turn_prompts(&traces),
        AgentTraceFilter::Crisp => {
            scaffold_prompts(traces.iter().copied().filter(|t| trace_is_successful(t)))
                .into_iter()
                .map(with_conciseness_pressure)
                .collect()
        }
    };

    if prompts.is_empty() {
        let scaffoldless = index
            .traces
            .values()
            .filter(|t| t.prompt_messages.is_empty())
            .count();
        if scaffoldless == total {
            return Err(format!(
                "agent-trace index has {total} trace(s) but none carry a prompt scaffold — \
                 the index predates prompt capture. Re-run `POST /v1/agent/traces/discover` \
                 to refresh it"
            ));
        }
        return Err(format!(
            "agent-trace index has {total} trace(s) but none qualify under \
             {selector:?} (successful = ended exit-0, not forked, no follow-up attempt; \
             weekly additionally = within the last 7 days) — try `agent_traces:all`, or \
             re-run `POST /v1/agent/traces/discover` to pick up newer sessions"
        ));
    }
    Ok(prompts)
}

/// Resolve a bare dataset name against the uploaded eval dataset registry
/// into OPD prompt scaffolds (one prompt per SFT conversation).
pub fn resolve_registry_opd_prompts(
    registry: Option<&crate::eval::DatasetRegistry>,
    name: &str,
) -> Result<Vec<OpdPrompt>, String> {
    let registry = registry.ok_or_else(|| {
        "dataset registry unavailable (server started without an eval root)".to_string()
    })?;
    // `iter_sft` surfaces a missing dataset as a bare IO error — probe the
    // data file first so absence gets the actionable message.
    let not_found = || {
        format!(
            "dataset {name:?} not found in the eval dataset registry — upload it via \
             `POST /v1/eval/datasets` or use an `agent_traces:` selector"
        )
    };
    let dir = registry
        .dataset_dir(name)
        .map_err(|e| format!("dataset {name:?}: {e}"))?;
    if !dir.join("data.jsonl").is_file() {
        return Err(not_found());
    }
    let iter = registry.iter_sft(name).map_err(|e| match e {
        crate::eval::DatasetError::NotFound(_) => not_found(),
        other => format!("dataset {name:?}: {other}"),
    })?;
    let prompts: Vec<OpdPrompt> = iter
        .map(|conv| OpdPrompt {
            messages: conv.messages,
            teacher_extra_messages: Vec::new(),
            trajectory: Vec::new(),
        })
        .filter(|p| !p.messages.is_empty())
        .collect();
    if prompts.is_empty() {
        return Err(format!(
            "dataset {name:?} contains no usable conversations (every row was empty or \
             malformed)"
        ));
    }
    Ok(prompts)
}

/// Resolve a bare dataset name against the registry into SFT examples.
/// Mirrors `submit_sft`'s train-by-dataset-name path for recipe steps,
/// which enqueue directly without passing through `submit_sft`.
pub fn resolve_registry_sft_examples(
    registry: Option<&crate::eval::DatasetRegistry>,
    name: &str,
) -> Result<Vec<kiln_train::SftExample>, String> {
    let prompts = resolve_registry_opd_prompts(registry, name)?;
    Ok(prompts
        .into_iter()
        .map(|p| kiln_train::SftExample {
            messages: p.messages,
        })
        .collect())
}

/// Unified entry: `agent_traces:<filter>` selectors hit the trace index;
/// anything else is treated as an uploaded-dataset name.
pub fn resolve_opd_dataset_selector(
    selector: &str,
    adapter_dir: &Path,
    registry: Option<&crate::eval::DatasetRegistry>,
    now_unix_ms: i64,
) -> Result<Vec<OpdPrompt>, String> {
    if is_agent_traces_selector(selector) {
        resolve_agent_trace_prompts(adapter_dir, selector, now_unix_ms)
    } else {
        resolve_registry_opd_prompts(registry, selector)
    }
}

fn trace_is_successful(trace: &AgentTrace) -> bool {
    trace.outcome.ended_with_exit_0 != Some(false)
        && trace.outcome.has_followup_attempt != Some(true)
        && !trace.forked
}

fn trace_within_ms(trace: &AgentTrace, now_unix_ms: i64, window_ms: i64) -> bool {
    let Some(ts) = trace.last_event_at.as_deref() else {
        return false;
    };
    let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(ts) else {
        return false;
    };
    let age_ms = now_unix_ms.saturating_sub(parsed.timestamp_millis());
    (0..=window_ms).contains(&age_ms)
}

/// Task scaffolds for re-rolling: one prompt per trace, carrying the
/// leading system/user context the session started from. The student
/// samples fresh rollouts from these; the (judge) teacher scores them —
/// §10.6.2 step 1, "samples agent rollouts on the week's pi tasks".
fn scaffold_prompts<'a>(traces: impl Iterator<Item = &'a AgentTrace>) -> Vec<OpdPrompt> {
    traces
        .filter(|t| !t.prompt_messages.is_empty())
        .map(|t| OpdPrompt {
            messages: t.prompt_messages.clone(),
            teacher_extra_messages: Vec::new(),
            trajectory: Vec::new(),
        })
        .collect()
}

/// §10.6.1 (turn, context) judge-scoring corpus: for each assistant action
/// in each trace, a prompt asking the model to score that turn given the
/// session context that preceded it.
fn judge_turn_prompts(traces: &[&AgentTrace]) -> Vec<OpdPrompt> {
    let mut prompts = Vec::new();
    for trace in traces {
        if prompts.len() >= MAX_JUDGE_PROMPTS {
            break;
        }
        let action_count = trace
            .trajectory
            .iter()
            .filter(|s| s.kind == TurnKind::Action)
            .count();
        if action_count == 0 {
            continue;
        }
        // Keep the most recent N actions per trace.
        let skip_actions = action_count.saturating_sub(MAX_JUDGE_TURNS_PER_TRACE);

        let mut transcript: Vec<String> = trace
            .prompt_messages
            .iter()
            .map(|m| render_transcript_line(&m.role, &m.content))
            .collect();
        let mut actions_seen = 0usize;
        for seg in &trace.trajectory {
            if seg.kind == TurnKind::Action {
                actions_seen += 1;
                if actions_seen > skip_actions && prompts.len() < MAX_JUDGE_PROMPTS {
                    prompts.push(judge_prompt(&transcript, seg));
                }
            }
            transcript.push(render_transcript_line(&seg.role, &seg.content));
        }
    }
    prompts
}

fn render_transcript_line(role: &str, content: &str) -> String {
    format!(
        "[{role}] {}",
        truncate_middle(content, JUDGE_SEGMENT_CHAR_CAP)
    )
}

fn judge_prompt(transcript: &[String], action: &TurnSegment) -> OpdPrompt {
    // Tail of the transcript closest to the candidate turn, within budget.
    let mut context_lines: Vec<&str> = Vec::new();
    let mut budget = JUDGE_CONTEXT_CHAR_BUDGET;
    for line in transcript.iter().rev() {
        let cost = line.chars().count() + 1;
        if cost > budget {
            break;
        }
        budget -= cost;
        context_lines.push(line.as_str());
    }
    context_lines.reverse();
    let context = if context_lines.is_empty() {
        "(session start)".to_string()
    } else {
        context_lines.join("\n")
    };
    OpdPrompt {
        messages: vec![
            ChatMessage::new(
                "system",
                "You are a strict turn judge for terminal coding agents. Score \
                          the candidate assistant turn on five axes — tool_correctness, \
                          goal_progress, reasoning_quality, terseness, \
                          instruction_following — each 0-10, and respond with only a \
                          JSON object like {\"tool_correctness\": 7, \"goal_progress\": 5, \
                          \"reasoning_quality\": 6, \"terseness\": 8, \
                          \"instruction_following\": 9}. Judge only the candidate turn, \
                          not the rest of the session.",
            ),
            ChatMessage::new(
                "user",
                format!(
                    "# Session context\n{context}\n\n# Candidate assistant turn\n{}\n\n\
                     Score the candidate turn now.",
                    truncate_middle(&action.content, JUDGE_TURN_CHAR_CAP)
                ),
            ),
        ],
        teacher_extra_messages: Vec::new(),
        trajectory: Vec::new(),
    }
}

/// §10.6.4 CRISP: same task, explicit conciseness pressure. Folded into an
/// existing system message when present so the chat template still sees a
/// single system turn.
fn with_conciseness_pressure(mut prompt: OpdPrompt) -> OpdPrompt {
    const PRESSURE: &str = "Be maximally concise: shortest correct tool calls, no filler \
                            prose, the minimum tokens that still solve the task correctly.";
    match prompt.messages.first_mut() {
        Some(first) if first.role == "system" => {
            first.content = format!("{}\n\n{PRESSURE}", first.content);
        }
        _ => prompt
            .messages
            .insert(0, ChatMessage::new("system", PRESSURE)),
    }
    prompt
}

fn truncate_middle(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let keep = max_chars.saturating_sub(16) / 2;
    let head: String = text.chars().take(keep).collect();
    let tail: String = text.chars().skip(count - keep).collect();
    format!("{head}\n[… {} chars …]\n{tail}", count - 2 * keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::agent_traces::TraceOutcome;
    use std::collections::BTreeMap;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage::new(role, content)
    }

    fn seg(role: &str, kind: TurnKind, content: &str) -> TurnSegment {
        TurnSegment {
            role: role.to_string(),
            content: content.to_string(),
            kind,
            tool_call_id: None,
            warning_prefix_len: None,
        }
    }

    fn trace(id: &str, last_event_at: &str, success: bool) -> AgentTrace {
        AgentTrace {
            id: id.to_string(),
            working_dir: "/tmp/proj".to_string(),
            num_turns: 4,
            num_tool_calls: 1,
            outcome: TraceOutcome {
                ended_with_exit_0: Some(success),
                user_edited_agent_files: Vec::new(),
                has_followup_attempt: Some(false),
            },
            first_event_at: Some(last_event_at.to_string()),
            last_event_at: Some(last_event_at.to_string()),
            forked: false,
            parent_id: None,
            tool_manifest_sha: None,
            prompt_messages: vec![
                msg("system", "You are pi."),
                msg("user", &format!("Fix the failing test in {id}")),
            ],
            trajectory: vec![
                seg("assistant", TurnKind::Action, "Looking at the test."),
                seg("tool", TurnKind::Observation, "FAILED tests/x.rs"),
                seg("assistant", TurnKind::Action, "Patching."),
            ],
        }
    }

    fn write_index(dir: &Path, traces: &[AgentTrace]) {
        let map: BTreeMap<String, AgentTrace> =
            traces.iter().map(|t| (t.id.clone(), t.clone())).collect();
        std::fs::write(
            dir.join("agent_traces.json"),
            serde_json::to_vec_pretty(&map).unwrap(),
        )
        .unwrap();
    }

    const NOW_MS: i64 = 1_780_000_000_000; // 2026-06-08-ish

    fn rfc3339_ms_ago(ms: i64) -> String {
        chrono::DateTime::from_timestamp_millis(NOW_MS - ms)
            .unwrap()
            .to_rfc3339()
    }

    #[test]
    fn filter_parsing_accepts_known_filters_and_aliases() {
        assert_eq!(
            parse_agent_trace_filter("agent_traces:weekly").unwrap(),
            AgentTraceFilter::Weekly
        );
        assert_eq!(
            parse_agent_trace_filter("agent_traces:user-pi").unwrap(),
            AgentTraceFilter::Successful
        );
        let err = parse_agent_trace_filter("agent_traces:bogus").unwrap_err();
        assert!(err.contains("weekly"), "lists valid filters: {err}");
    }

    #[test]
    fn missing_index_error_names_discover_endpoint() {
        let dir = tempfile::tempdir().unwrap();
        let err =
            resolve_agent_trace_prompts(dir.path(), "agent_traces:weekly", NOW_MS).unwrap_err();
        assert!(err.contains("/v1/agent/traces/discover"), "{err}");
    }

    #[test]
    fn weekly_filters_by_success_and_recency() {
        let dir = tempfile::tempdir().unwrap();
        let recent_ok = trace("recent-ok", &rfc3339_ms_ago(3_600_000), true);
        let recent_fail = trace("recent-fail", &rfc3339_ms_ago(3_600_000), false);
        let old_ok = trace("old-ok", &rfc3339_ms_ago(30 * 24 * 3_600_000), true);
        write_index(dir.path(), &[recent_ok, recent_fail, old_ok]);

        let prompts =
            resolve_agent_trace_prompts(dir.path(), "agent_traces:weekly", NOW_MS).unwrap();
        assert_eq!(prompts.len(), 1);
        assert!(prompts[0].messages[1].content.contains("recent-ok"));
        // Re-roll scaffolds carry no replay trajectory.
        assert!(prompts[0].trajectory.is_empty());

        let all = resolve_agent_trace_prompts(dir.path(), "agent_traces:all", NOW_MS).unwrap();
        assert_eq!(all.len(), 3);

        let successful =
            resolve_agent_trace_prompts(dir.path(), "agent_traces:successful", NOW_MS).unwrap();
        assert_eq!(successful.len(), 2, "old-ok still qualifies as successful");
    }

    #[test]
    fn forked_and_followed_up_traces_are_not_successful() {
        let dir = tempfile::tempdir().unwrap();
        let mut forked = trace("forked", &rfc3339_ms_ago(1000), true);
        forked.forked = true;
        let mut followup = trace("followup", &rfc3339_ms_ago(1000), true);
        followup.outcome.has_followup_attempt = Some(true);
        write_index(dir.path(), &[forked, followup]);
        let err =
            resolve_agent_trace_prompts(dir.path(), "agent_traces:weekly", NOW_MS).unwrap_err();
        assert!(err.contains("none qualify"), "{err}");
        assert!(
            err.contains("agent_traces:all"),
            "offers remediation: {err}"
        );
    }

    #[test]
    fn scaffoldless_index_says_rerun_discover() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = trace("old-format", &rfc3339_ms_ago(1000), true);
        t.prompt_messages.clear();
        write_index(dir.path(), &[t]);
        let err = resolve_agent_trace_prompts(dir.path(), "agent_traces:all", NOW_MS).unwrap_err();
        assert!(err.contains("predates prompt capture"), "{err}");
    }

    #[test]
    fn judge_turns_builds_turn_context_pairs_from_all_traces() {
        let dir = tempfile::tempdir().unwrap();
        let ok = trace("ok", &rfc3339_ms_ago(1000), true);
        let failed = trace("failed", &rfc3339_ms_ago(2000), false);
        write_index(dir.path(), &[ok, failed]);

        let prompts =
            resolve_agent_trace_prompts(dir.path(), "agent_traces:judge_turns", NOW_MS).unwrap();
        // 2 actions per trace × 2 traces (failures included — they're the
        // discrimination signal).
        assert_eq!(prompts.len(), 4);
        for p in &prompts {
            assert_eq!(p.messages[0].role, "system");
            assert!(p.messages[0].content.contains("tool_correctness"));
            assert!(p.messages[1].content.contains("# Candidate assistant turn"));
        }
        // The second action's context includes the first action and the
        // tool observation that followed it.
        let second = &prompts[1].messages[1].content;
        assert!(second.contains("Looking at the test."), "{second}");
        assert!(second.contains("FAILED tests/x.rs"), "{second}");
        assert!(second.contains("Patching."), "{second}");
    }

    #[test]
    fn crisp_prompts_fold_pressure_into_existing_system_message() {
        let dir = tempfile::tempdir().unwrap();
        write_index(dir.path(), &[trace("t", &rfc3339_ms_ago(1000), true)]);
        let prompts =
            resolve_agent_trace_prompts(dir.path(), "agent_traces:crisp", NOW_MS).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].messages[0].role, "system");
        assert!(prompts[0].messages[0].content.starts_with("You are pi."));
        assert!(prompts[0].messages[0].content.contains("maximally concise"));
        // No double system message.
        assert_eq!(
            prompts[0]
                .messages
                .iter()
                .filter(|m| m.role == "system")
                .count(),
            1
        );
    }

    #[test]
    fn registry_resolution_reads_sft_datasets() {
        let dir = tempfile::tempdir().unwrap();
        let registry = crate::eval::DatasetRegistry::new(dir.path().join("datasets"));
        let row = serde_json::json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"}
                    }]
                },
                {
                    "role": "tool",
                    "content": "world",
                    "name": "lookup",
                    "tool_call_id": "call_1"
                },
                {"role": "assistant", "content": "done"}
            ]
        });
        let body = format!("{row}\n");
        registry
            .create(
                "my-data",
                crate::eval::DatasetFormat::SftChat,
                None,
                body.as_bytes(),
            )
            .unwrap();
        let prompts = resolve_registry_opd_prompts(Some(&registry), "my-data").unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].messages[0].content, "hello");
        assert_eq!(prompts[0].messages[1].content, "");
        assert_eq!(prompts[0].messages[1].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(prompts[0].messages[2].name.as_deref(), Some("lookup"));
        assert_eq!(
            prompts[0].messages[2].tool_call_id.as_deref(),
            Some("call_1")
        );

        let examples = resolve_registry_sft_examples(Some(&registry), "my-data").unwrap();
        assert_eq!(examples[0].messages, prompts[0].messages);

        let err = resolve_registry_opd_prompts(Some(&registry), "nope").unwrap_err();
        assert!(err.contains("/v1/eval/datasets"), "{err}");

        let err = resolve_registry_opd_prompts(None, "my-data").unwrap_err();
        assert!(err.contains("registry unavailable"), "{err}");
    }

    #[test]
    fn unified_selector_routes_by_prefix() {
        let dir = tempfile::tempdir().unwrap();
        write_index(dir.path(), &[trace("t", &rfc3339_ms_ago(1000), true)]);
        let registry = crate::eval::DatasetRegistry::new(dir.path().join(".eval/datasets"));
        registry
            .create(
                "uploaded",
                crate::eval::DatasetFormat::SftChat,
                None,
                b"{\"messages\":[{\"role\":\"user\",\"content\":\"from registry\"}]}\n",
            )
            .unwrap();

        let from_traces =
            resolve_opd_dataset_selector("agent_traces:all", dir.path(), Some(&registry), NOW_MS)
                .unwrap();
        assert!(
            from_traces[0].messages[1]
                .content
                .contains("Fix the failing test")
        );

        let from_registry =
            resolve_opd_dataset_selector("uploaded", dir.path(), Some(&registry), NOW_MS).unwrap();
        assert_eq!(from_registry[0].messages[0].content, "from registry");
    }

    #[test]
    fn truncate_middle_keeps_head_and_tail() {
        let long = "a".repeat(100) + &"b".repeat(100);
        let cut = truncate_middle(&long, 50);
        assert!(cut.starts_with("aaa"));
        assert!(cut.ends_with("bbb"));
        assert!(cut.contains("chars …"));
        assert!(truncate_middle("short", 50) == "short");
    }
}
