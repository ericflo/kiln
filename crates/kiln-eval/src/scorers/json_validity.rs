//! JSON-validity scorer.
//!
//! Three escalating bars are supported:
//! - Plain validity (default): the completion parses as JSON.
//! - `require_object`: the parsed value must be an object.
//! - `required_paths`: each JSON Pointer in this list must resolve.
//!
//! When the example has a `target`, the target is parsed as canonical JSON
//! and compared structurally (after sorting object keys). Two JSON documents
//! that differ only in key order or whitespace still match.

use crate::result::EvalOutcomeKind;
use crate::suite::EvalExample;

pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    require_object: bool,
    required_paths: &[String],
) -> (f32, EvalOutcomeKind, Option<String>) {
    let trimmed = completion_text.trim();
    let candidate = extract_json_body(trimmed);
    let parsed: serde_json::Value = match serde_json::from_str(candidate) {
        Ok(v) => v,
        Err(e) => {
            return (
                0.0,
                EvalOutcomeKind::Invalid,
                Some(format!("parse error: {e}")),
            );
        }
    };

    if require_object && !parsed.is_object() {
        return (
            0.0,
            EvalOutcomeKind::Fail,
            Some(format!("expected object, got {}", value_kind(&parsed))),
        );
    }
    for path in required_paths {
        if parsed.pointer(path).is_none() {
            return (
                0.0,
                EvalOutcomeKind::Fail,
                Some(format!("missing required path `{path}`")),
            );
        }
    }
    if let Some(target) = example.target.as_deref() {
        let target_value: serde_json::Value = match serde_json::from_str(target) {
            Ok(v) => v,
            Err(e) => {
                return (
                    0.0,
                    EvalOutcomeKind::Invalid,
                    Some(format!("target itself failed to parse: {e}")),
                );
            }
        };
        if canonicalize(&parsed) == canonicalize(&target_value) {
            (1.0, EvalOutcomeKind::Pass, None)
        } else {
            (
                0.0,
                EvalOutcomeKind::Fail,
                Some("parsed JSON did not match target".into()),
            )
        }
    } else {
        (1.0, EvalOutcomeKind::Pass, None)
    }
}

/// Best-effort extraction of a JSON body from a free-form completion. The
/// model often wraps JSON in ```json fences or includes leading prose; we
/// grab the substring from the first `{` or `[` to the matching closing
/// bracket. Falls back to the input as-is when no bracket is found.
fn extract_json_body(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut start: Option<usize> = None;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'{' || *b == b'[' {
            start = Some(i);
            break;
        }
    }
    let Some(start_idx) = start else {
        return s;
    };
    let opener = bytes[start_idx];
    let closer = if opener == b'{' { b'}' } else { b']' };
    let mut depth = 0i64;
    let mut in_str = false;
    let mut escape = false;
    for (i, b) in bytes.iter().enumerate().skip(start_idx) {
        let c = *b;
        if escape {
            escape = false;
            continue;
        }
        if in_str {
            if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        if c == b'"' {
            in_str = true;
            continue;
        }
        if c == opener {
            depth += 1;
        } else if c == closer {
            depth -= 1;
            if depth == 0 {
                return &s[start_idx..=i];
            }
        }
    }
    &s[start_idx..]
}

fn value_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn canonicalize(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));
            let mut out = serde_json::Map::new();
            for (k, v) in entries {
                out.insert(k.clone(), canonicalize(v));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize).collect())
        }
        _ => v.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::EvalChatMessage;

    fn ex(target: Option<&str>) -> EvalExample {
        EvalExample {
            messages: vec![EvalChatMessage::new("user", "q")],
            target: target.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn valid_json_no_target_passes() {
        let (_, kind, _) = score(&ex(None), r#"{"a":1}"#, false, &[]);
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn invalid_json_marked_invalid() {
        let (_, kind, _) = score(&ex(None), "not json", false, &[]);
        assert_eq!(kind, EvalOutcomeKind::Invalid);
    }

    #[test]
    fn require_object_rejects_arrays() {
        let (_, kind, _) = score(&ex(None), "[1,2]", true, &[]);
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn required_paths_check() {
        let (_, kind, _) = score(
            &ex(None),
            r#"{"tool_call": {"name": "x"}}"#,
            false,
            &["/tool_call/name".into()],
        );
        assert_eq!(kind, EvalOutcomeKind::Pass);
        let (_, kind, _) = score(
            &ex(None),
            r#"{"other": 1}"#,
            false,
            &["/tool_call/name".into()],
        );
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn target_compared_canonicalized() {
        let (_, kind, _) = score(
            &ex(Some(r#"{"b":2,"a":1}"#)),
            r#"{"a":1,"b":2}"#,
            true,
            &[],
        );
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn extract_json_strips_fences_and_prose() {
        let out = extract_json_body("Here you go:\n```json\n{\"a\":1}\n```");
        assert_eq!(out.trim(), "{\"a\":1}");
    }

    #[test]
    fn extract_handles_escaped_braces_in_strings() {
        let inp = r#"prelude { "msg": "} hi {" } trailer"#;
        let body = extract_json_body(inp);
        assert!(body.starts_with('{'));
        assert!(body.ends_with('}'));
        let parsed: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(parsed["msg"], "} hi {");
    }
}
