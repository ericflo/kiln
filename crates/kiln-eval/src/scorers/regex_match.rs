//! Regex scorer with optional capture-group comparison.

use regex::RegexBuilder;

use crate::result::EvalOutcomeKind;
use crate::scorers::ScorerError;
use crate::suite::EvalExample;

pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    pattern: &str,
    capture_group: Option<usize>,
    case_sensitive: bool,
) -> Result<(f32, EvalOutcomeKind, Option<String>), ScorerError> {
    let re = RegexBuilder::new(pattern)
        .case_insensitive(!case_sensitive)
        .build()
        .map_err(|e| ScorerError::InvalidRegex {
            pattern: pattern.to_string(),
            msg: format!("{e}"),
        })?;
    let caps = match re.captures(completion_text) {
        Some(c) => c,
        None => {
            return Ok((
                0.0,
                EvalOutcomeKind::Fail,
                Some(format!("pattern `{pattern}` did not match")),
            ));
        }
    };

    let Some(group_idx) = capture_group else {
        // Plain match-vs-no-match.
        return Ok((1.0, EvalOutcomeKind::Pass, None));
    };
    // Capture-group mode: compare the captured slice against the example target.
    let Some(matched) = caps.get(group_idx) else {
        return Ok((
            0.0,
            EvalOutcomeKind::Invalid,
            Some(format!("capture group {group_idx} did not bind")),
        ));
    };
    let target = example.target.as_deref().ok_or(ScorerError::MissingTarget {
        kind: "regex",
    })?;
    let actual = matched.as_str();
    let mut candidates: Vec<&str> = std::iter::once(target).collect();
    for a in &example.aliases {
        candidates.push(a.as_str());
    }
    let matched_target = if case_sensitive {
        candidates.iter().any(|c| *c == actual)
    } else {
        candidates
            .iter()
            .any(|c| c.eq_ignore_ascii_case(actual))
    };
    if matched_target {
        Ok((1.0, EvalOutcomeKind::Pass, None))
    } else {
        Ok((
            0.0,
            EvalOutcomeKind::Fail,
            Some(format!("expected `{target}`, captured `{actual}`")),
        ))
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
    fn plain_match_passes_without_capture() {
        let (_, kind, _) =
            score(&ex(None), "the answer is 42", r"answer is \d+", None, true).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn capture_group_compares_to_target() {
        let (_, kind, _) =
            score(&ex(Some("42")), "answer: 42", r"answer:\s*(\d+)", Some(1), true).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
        let (_, kind, _) =
            score(&ex(Some("43")), "answer: 42", r"answer:\s*(\d+)", Some(1), true).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn missing_group_is_invalid() {
        let (_, kind, _) =
            score(&ex(Some("x")), "answer: 42", r"answer:\s*\d+", Some(1), true).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Invalid);
    }

    #[test]
    fn bad_pattern_is_scorer_error() {
        let err = score(&ex(None), "x", r"(", None, true).unwrap_err();
        assert!(matches!(err, ScorerError::InvalidRegex { .. }));
    }
}
