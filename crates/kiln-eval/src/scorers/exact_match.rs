//! Exact-match scorer with normalization knobs.

use unicode_normalization::UnicodeNormalization;

use crate::result::EvalOutcomeKind;
use crate::scorers::ScorerError;
use crate::suite::EvalExample;

pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    case_sensitive: bool,
    strip_whitespace: bool,
) -> Result<(f32, EvalOutcomeKind, Option<String>), ScorerError> {
    let target = example
        .target
        .as_deref()
        .ok_or(ScorerError::MissingTarget {
            kind: "exact_match",
        })?;
    let candidates: Vec<String> = std::iter::once(target.to_string())
        .chain(example.aliases.iter().cloned())
        .map(|s| normalize(&s, case_sensitive, strip_whitespace))
        .collect();
    let actual = normalize(completion_text, case_sensitive, strip_whitespace);
    if candidates.iter().any(|t| t == &actual) {
        Ok((1.0, EvalOutcomeKind::Pass, None))
    } else {
        let preview = preview_for_detail(&actual);
        let target_preview = preview_for_detail(&candidates[0]);
        Ok((
            0.0,
            EvalOutcomeKind::Fail,
            Some(format!("expected `{target_preview}`, got `{preview}`")),
        ))
    }
}

fn normalize(s: &str, case_sensitive: bool, strip_whitespace: bool) -> String {
    let nfc: String = s.nfc().collect();
    let stripped = if strip_whitespace {
        nfc.trim().to_string()
    } else {
        nfc
    };
    if case_sensitive {
        stripped
    } else {
        stripped.to_lowercase()
    }
}

fn preview_for_detail(s: &str) -> String {
    if s.chars().count() <= 32 {
        s.to_string()
    } else {
        let head: String = s.chars().take(32).collect();
        format!("{head}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::EvalChatMessage;

    fn ex(target: &str) -> EvalExample {
        EvalExample {
            messages: vec![EvalChatMessage::new("user", "q")],
            target: Some(target.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn case_insensitive_passes_with_strip() {
        let (score, kind, _) = score(&ex("Paris"), "  paris\n", false, true).unwrap();
        assert_eq!(score, 1.0);
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn case_sensitive_fails_on_case_mismatch() {
        let (_, kind, _) = score(&ex("Paris"), "paris", true, true).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn aliases_accepted() {
        let mut e = ex("Paris");
        e.aliases = vec!["The City of Light".into()];
        let (_, kind, _) = score(&e, "the city of light", false, true).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn missing_target_is_error() {
        let mut e = ex("x");
        e.target = None;
        let err = score(&e, "x", false, true).unwrap_err();
        assert!(matches!(err, ScorerError::MissingTarget { .. }));
    }

    #[test]
    fn nfc_normalizes_unicode() {
        // "é" expressed two ways: precomposed vs combining
        let target = "café";
        let candidate = "cafe\u{0301}"; // 'e' + combining acute
        let (_, kind, _) = score(&ex(target), candidate, false, true).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }
}
