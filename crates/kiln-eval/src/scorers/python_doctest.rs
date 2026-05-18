//! Python doctest scorer.
//!
//! The humaneval / python-algo eval datasets ship the test cases inside the
//! prompt's docstring as `>>> call_to_function(args)` / `expected_value`
//! pairs. This scorer parses those out of the user message, extracts the
//! Python function the model generated from a fenced code block, runs the
//! function under a Python subprocess against each doctest assertion, and
//! returns a fractional pass rate.
//!
//! Design choices:
//!
//! - The scorer reaches into [`EvalExample::messages`] to find the prompt
//!   text — every other built-in scorer only inspects the completion +
//!   `target`, but doctest-style verification structurally needs the
//!   problem statement.
//! - Execution is a single short-lived `python3` subprocess per completion,
//!   with stdin used to feed the generated function plus a small harness.
//!   The harness compares `repr(actual)` to the expected string from the
//!   docstring after both have been stripped of trailing whitespace, so the
//!   classic `>>> f()\n5` form matches whether the function returned `5` or
//!   `5L` / `5.0` etc. — same `repr` normalization Python's stdlib doctest
//!   module applies.
//! - Each subprocess gets a hard wall-clock timeout (default 5 s). On
//!   timeout the score is 0 and the outcome is `Error`. The subprocess is
//!   killed with SIGKILL via `Child::kill` so a runaway `while True:`
//!   doesn't leak.
//! - We never hit the network or write to anywhere outside the spawned
//!   subprocess's own memory. The harness is self-contained text fed on
//!   stdin.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::result::EvalOutcomeKind;
use crate::scorers::ScorerError;
use crate::suite::EvalExample;

/// Default execution time budget for a single completion's doctests.
pub const DEFAULT_TIMEOUT_SECONDS: f32 = 5.0;

/// Default Python interpreter command. Overridable via the scorer config
/// or the `KILN_DOCTEST_PYTHON` environment variable.
pub const DEFAULT_PYTHON_BIN: &str = "python3";

/// One extracted doctest pair: a callable expression and the expected
/// result (already stripped of leading `>>> ` and trailing whitespace).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Doctest {
    pub call: String,
    pub expected: String,
}

/// Parse `>>> expr\n<expected>` pairs out of a docstring-bearing prompt.
///
/// Recognizes the classical Python doctest layout: any line beginning with
/// `>>>` (after optional indentation) starts a doctest; subsequent lines that
/// begin with `... ` (continuation) are appended to the same call; the first
/// non-empty line that does NOT begin with `>>> ` / `... ` is the expected
/// result. An empty result line (e.g. `>>> f()\n\n>>> g()\n...`) is treated
/// as expected output `None` — the canonical "function returned None and
/// printed nothing" doctest convention.
pub(crate) fn parse_doctests(prompt: &str) -> Vec<Doctest> {
    let mut out = Vec::new();
    let mut lines = prompt.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let Some(call_start) = trimmed.strip_prefix(">>> ") else {
            // also accept `>>>` with no trailing space + empty body
            if trimmed.trim_end() == ">>>" {
                continue;
            }
            continue;
        };
        let mut call = call_start.trim_end().to_string();
        // Greedily fold `... ` continuation lines into the call.
        while let Some(next) = lines.peek() {
            let nt = next.trim_start();
            if let Some(cont) = nt.strip_prefix("... ").or_else(|| nt.strip_prefix("...")) {
                call.push('\n');
                call.push_str(cont.trim_end());
                lines.next();
            } else {
                break;
            }
        }
        // The expected line: the next non-doctest-prefixed line. If it's
        // blank or a new `>>>`, expected is `None` (the function returned
        // None or printed nothing).
        let expected = match lines.peek() {
            None => "None".to_string(),
            Some(next) => {
                let nt = next.trim_end();
                let nt_lead = next.trim_start();
                if nt_lead.starts_with(">>> ")
                    || nt_lead == ">>>"
                    || nt_lead.starts_with("```")
                    || nt.is_empty()
                {
                    "None".to_string()
                } else {
                    let consumed = lines.next().unwrap();
                    consumed.trim_end().trim_start().to_string()
                }
            }
        };
        out.push(Doctest { call, expected });
    }
    out
}

/// Extract the first python (or generic) fenced code block from the
/// completion. Falls back to the whole completion when no fence is present
/// (covers models that emit raw code without markdown).
pub(crate) fn extract_python_block(completion: &str) -> String {
    let mut iter = completion.split("```");
    // First piece is text BEFORE the first fence — drop.
    if iter.next().is_none() {
        return completion.trim().to_string();
    }
    let Some(fenced) = iter.next() else {
        return completion.trim().to_string();
    };
    // The fenced segment may start with the language tag on its first line.
    let body = match fenced.split_once('\n') {
        Some((tag, rest)) => {
            let lower = tag.trim().to_lowercase();
            if lower.is_empty() || lower == "python" || lower == "py" || lower == "python3" {
                rest
            } else {
                // Other-language tag (e.g. `bash`) — still take the body so
                // we at least try to score; the harness will fail to import.
                rest
            }
        }
        None => fenced,
    };
    body.trim_end().trim_end_matches('`').trim_end().to_string()
}

/// Internal harness assembled per call.
fn build_harness(user_code: &str, doctests: &[Doctest]) -> String {
    // We test by `repr(actual) == expected` after a strip. Python's stdlib
    // doctest does the same thing under "no special doctest directives"
    // mode. We catch user exceptions per-test and report them as failures.
    let mut harness = String::from(
        "import sys, json, traceback\n_kdt_results = []\n",
    );
    harness.push_str("try:\n");
    for line in user_code.lines() {
        harness.push_str("    ");
        harness.push_str(line);
        harness.push('\n');
    }
    if user_code.is_empty() {
        harness.push_str("    pass\n");
    }
    harness.push_str("except Exception:\n");
    harness.push_str("    _kdt_results.append({\"ok\": False, \"detail\": \"import_error: \" + traceback.format_exc()[-200:]})\n");
    harness.push_str("    print(json.dumps(_kdt_results))\n");
    harness.push_str("    sys.exit(0)\n");
    for (i, dt) in doctests.iter().enumerate() {
        // `call` is a Python EXPRESSION — emit verbatim. Multi-line calls
        // (from `... ` continuations) are joined with embedded newlines,
        // which need to stay readable as Python; wrap in parens to make
        // multi-line expressions parse.
        let call_expr = if dt.call.contains('\n') {
            format!("(\n{}\n)", dt.call)
        } else {
            dt.call.clone()
        };
        // `expected` goes INSIDE a Python string literal — escape the
        // backslashes and double quotes only.
        let expected_lit = dt.expected.replace('\\', "\\\\").replace('"', "\\\"");
        // `_kdt_safe_*` field values use json.dumps later so they survive
        // whatever weirdness is inside the actual/repr output.
        harness.push_str(&format!(
            "try:\n    _kdt_actual = {call_expr}\n    _kdt_expected = \"{expected_lit}\"\n    _kdt_ok = repr(_kdt_actual).strip() == _kdt_expected.strip()\n    _kdt_results.append({{\"i\": {i}, \"ok\": _kdt_ok, \"expected\": _kdt_expected, \"actual\": repr(_kdt_actual)[:200]}})\nexcept Exception as e:\n    _kdt_results.append({{\"i\": {i}, \"ok\": False, \"expected\": \"{expected_lit}\", \"err\": traceback.format_exc()[-300:]}})\n",
            call_expr = call_expr,
            i = i,
            expected_lit = expected_lit
        ));
    }
    harness.push_str("print(json.dumps(_kdt_results))\n");
    harness
}

/// Outcome of running the harness for one completion.
pub(crate) struct DoctestRunOutcome {
    pub passed: usize,
    pub total: usize,
    pub timed_out: bool,
    pub spawn_error: Option<String>,
    pub first_failure_detail: Option<String>,
}

/// Run `python3` against the harness assembled from `user_code` + `doctests`.
/// `timeout` caps the wall-clock budget; an overrun returns `timed_out=true`.
///
/// The child is polled via `try_wait` so we can call `kill` when the
/// deadline expires — this handles infinite-loop / `while True:` user
/// code correctly. stdout is then drained from the (now-closed) pipe.
pub(crate) fn run_doctests(
    python_bin: &str,
    user_code: &str,
    doctests: &[Doctest],
    timeout: Duration,
) -> DoctestRunOutcome {
    let harness = build_harness(user_code, doctests);
    let mut child = match Command::new(python_bin)
        .arg("-I") // isolated mode: no PYTHONPATH, no user site
        .arg("-S") // no site-packages init
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return DoctestRunOutcome {
                passed: 0,
                total: doctests.len(),
                timed_out: false,
                spawn_error: Some(format!("spawn {python_bin}: {e}")),
                first_failure_detail: None,
            };
        }
    };
    // Feed the harness on stdin in a thread so a blocked stdin write can't
    // deadlock the parent. Drop the writer afterward so the child sees EOF.
    if let Some(mut stdin) = child.stdin.take() {
        let bytes = harness.into_bytes();
        thread::spawn(move || {
            let _ = stdin.write_all(&bytes);
            // stdin is dropped here, closing the pipe so the child sees EOF.
        });
    }
    // Poll for child exit until the deadline, killing on overrun.
    let deadline = Instant::now() + timeout;
    let timed_out;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => {
                timed_out = false;
                break;
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                return DoctestRunOutcome {
                    passed: 0,
                    total: doctests.len(),
                    timed_out: false,
                    spawn_error: Some(format!("try_wait: {e}")),
                    first_failure_detail: None,
                };
            }
        }
    }
    // Drain stdout/stderr now that the child has exited.
    let mut stdout_buf = Vec::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_end(&mut stdout_buf);
    }
    if timed_out {
        return DoctestRunOutcome {
            passed: 0,
            total: doctests.len(),
            timed_out: true,
            spawn_error: None,
            first_failure_detail: Some(format!("timeout after {:.1}s", timeout.as_secs_f32())),
        };
    }
    let stdout = String::from_utf8_lossy(&stdout_buf);
    let parsed: Vec<serde_json::Value> = serde_json::from_str(stdout.trim()).unwrap_or_default();
    let mut passed = 0usize;
    let mut first_failure: Option<String> = None;
    for entry in &parsed {
        let ok = entry.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        if ok {
            passed += 1;
        } else if first_failure.is_none() {
            let exp = entry
                .get("expected")
                .and_then(|v| v.as_str())
                .unwrap_or("<no expected>");
            let actual = entry
                .get("actual")
                .or_else(|| entry.get("err"))
                .or_else(|| entry.get("detail"))
                .and_then(|v| v.as_str())
                .unwrap_or("<no actual>");
            first_failure = Some(format!("expected {exp} got {actual}"));
        }
    }
    DoctestRunOutcome {
        passed,
        total: doctests.len(),
        timed_out: false,
        spawn_error: None,
        first_failure_detail: first_failure,
    }
}

/// Score entry point. Returns `(score, kind, detail)` matching the rest of
/// the scorer surface in `crate::scorers`.
pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    timeout_seconds: f32,
    python_bin: &str,
) -> Result<(f32, EvalOutcomeKind, Option<String>), ScorerError> {
    let timeout = Duration::from_secs_f32(timeout_seconds.max(0.1));

    // Find the prompt's last user message — that's where humaneval-style
    // problems put the function signature + docstring + doctests.
    let prompt_text = example
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let doctests = parse_doctests(&prompt_text);
    if doctests.is_empty() {
        return Ok((
            0.0,
            EvalOutcomeKind::Invalid,
            Some("no doctest examples found in prompt".to_string()),
        ));
    }
    let user_code = extract_python_block(completion_text);
    if user_code.trim().is_empty() {
        return Ok((
            0.0,
            EvalOutcomeKind::Invalid,
            Some("no python code block in completion".to_string()),
        ));
    }
    let outcome = run_doctests(python_bin, &user_code, &doctests, timeout);
    if let Some(err) = outcome.spawn_error {
        return Ok((0.0, EvalOutcomeKind::Error, Some(err)));
    }
    if outcome.timed_out {
        return Ok((
            0.0,
            EvalOutcomeKind::Error,
            outcome.first_failure_detail,
        ));
    }
    let score = if outcome.total == 0 {
        0.0
    } else {
        outcome.passed as f32 / outcome.total as f32
    };
    let kind = if outcome.total > 0 && outcome.passed == outcome.total {
        EvalOutcomeKind::Pass
    } else {
        EvalOutcomeKind::Fail
    };
    let detail = if matches!(kind, EvalOutcomeKind::Pass) {
        Some(format!("{}/{} doctests passed", outcome.passed, outcome.total))
    } else {
        Some(format!(
            "{}/{} doctests passed{}",
            outcome.passed,
            outcome.total,
            outcome
                .first_failure_detail
                .map(|d| format!(" ({d})"))
                .unwrap_or_default()
        ))
    };
    Ok((score, kind, detail))
}

/// Resolve the python binary to use, honoring the env override.
pub(crate) fn resolve_python_bin(configured: Option<&str>) -> String {
    if let Some(c) = configured {
        if !c.is_empty() {
            return c.to_string();
        }
    }
    std::env::var("KILN_DOCTEST_PYTHON").unwrap_or_else(|_| DEFAULT_PYTHON_BIN.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::EvalChatMessage;

    fn user(content: &str) -> EvalExample {
        EvalExample {
            id: None,
            messages: vec![EvalChatMessage::new("user", content)],
            target: None,
            aliases: Vec::new(),
            tags: Vec::new(),
            metadata: None,
            scorer: None,
            generation: None,
            weight: 1.0,
            tools: None,
        }
    }

    #[test]
    fn parse_doctests_extracts_call_expected_pairs() {
        let prompt = "    >>> add(2, 3)\n    5\n    >>> add(5, 7)\n    12\n";
        let parsed = parse_doctests(prompt);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].call, "add(2, 3)");
        assert_eq!(parsed[0].expected, "5");
        assert_eq!(parsed[1].call, "add(5, 7)");
        assert_eq!(parsed[1].expected, "12");
    }

    #[test]
    fn parse_doctests_handles_empty_result_as_none() {
        let prompt = "    >>> longest([])\n\n    >>> longest(['a', 'b'])\n    'a'\n";
        let parsed = parse_doctests(prompt);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].expected, "None");
        assert_eq!(parsed[1].expected, "'a'");
    }

    #[test]
    fn extract_python_block_pulls_first_fenced_block() {
        let completion =
            "Here you go:\n```python\ndef add(x, y):\n    return x + y\n```\nAll done!";
        let code = extract_python_block(completion);
        assert!(code.contains("def add"));
        assert!(!code.contains("Here you go"));
    }

    #[test]
    fn extract_python_block_falls_back_to_raw_completion() {
        let completion = "def add(x, y):\n    return x + y\n";
        let code = extract_python_block(completion);
        assert!(code.contains("def add"));
    }

    #[test]
    #[ignore = "requires python3 on PATH"]
    fn score_passes_when_all_doctests_match() {
        let example = user(
            "Complete the function:\n```python\ndef add(x, y):\n    \"\"\"Add.\n    >>> add(2, 3)\n    5\n    >>> add(-1, 1)\n    0\n    \"\"\"\n```",
        );
        let completion = "```python\ndef add(x, y):\n    return x + y\n```";
        let (score, kind, _) = score(&example, completion, 5.0, "python3").unwrap();
        assert_eq!(score, 1.0);
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    #[ignore = "requires python3 on PATH"]
    fn score_fails_when_some_doctests_miss() {
        let example = user(
            "```python\ndef add(x, y):\n    \"\"\"Add.\n    >>> add(2, 3)\n    5\n    >>> add(0, 0)\n    0\n    \"\"\"\n```",
        );
        // Wrong: ignores y
        let completion = "```python\ndef add(x, y):\n    return x\n```";
        let (score, kind, _) = score(&example, completion, 5.0, "python3").unwrap();
        assert!(score < 1.0);
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    #[ignore = "requires python3 on PATH"]
    fn score_returns_error_on_runaway_loop() {
        let example = user(
            "```python\ndef f():\n    \"\"\"\n    >>> f()\n    1\n    \"\"\"\n```",
        );
        let completion = "```python\ndef f():\n    while True:\n        pass\n```";
        let (score, kind, _) = score(&example, completion, 0.5, "python3").unwrap();
        assert_eq!(score, 0.0);
        assert_eq!(kind, EvalOutcomeKind::Error);
    }
}
