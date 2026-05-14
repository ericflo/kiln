//! Multiple-choice scorer.
//!
//! Reduces the completion to a single answer label by:
//! 1. Looking for `"Answer: X"`, `"answer is X"`, `"The answer is X"`
//! 2. Looking for a leading `"X)"`, `"(X)"`, `"X."`
//! 3. Falling back to the first run of letters that matches one of `choices`.
//!
//! Designed to match what an MCQ-tuned model actually emits, not the
//! Pythia-style "first token is the letter" assumption (which only holds for
//! very specific prompt formats).

use crate::result::EvalOutcomeKind;
use crate::scorers::ScorerError;
use crate::suite::EvalExample;

pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    choices: &[String],
) -> Result<(f32, EvalOutcomeKind, Option<String>), ScorerError> {
    let target = example.target.as_deref().ok_or(ScorerError::MissingTarget {
        kind: "multiple_choice",
    })?;
    let target_norm = target.trim().to_uppercase();
    let mut valid: Vec<String> = choices.iter().map(|c| c.trim().to_uppercase()).collect();
    if !valid.contains(&target_norm) {
        valid.push(target_norm.clone());
    }
    // Aliases let suites name multiple correct answers (e.g. both "B" and "b").
    let alias_set: Vec<String> = example
        .aliases
        .iter()
        .map(|a| a.trim().to_uppercase())
        .collect();
    let accepted: Vec<&str> = std::iter::once(target_norm.as_str())
        .chain(alias_set.iter().map(|s| s.as_str()))
        .collect();

    let detected = detect_answer(completion_text, &valid);
    match detected {
        Some(letter) => {
            if accepted.iter().any(|t| *t == letter) {
                Ok((1.0, EvalOutcomeKind::Pass, None))
            } else {
                Ok((
                    0.0,
                    EvalOutcomeKind::Fail,
                    Some(format!("expected `{target}`, got `{letter}`")),
                ))
            }
        }
        None => Ok((
            0.0,
            EvalOutcomeKind::Invalid,
            Some("no answer letter detected".into()),
        )),
    }
}

fn detect_answer(text: &str, valid: &[String]) -> Option<String> {
    let upper = text.to_uppercase();
    // Look for the strongest signal first.
    for marker in ["ANSWER:", "ANSWER IS", "FINAL ANSWER:", "FINAL ANSWER IS"] {
        if let Some(idx) = upper.find(marker) {
            let tail = &upper[idx + marker.len()..];
            if let Some(letter) = next_letter(tail, valid) {
                return Some(letter);
            }
        }
    }
    // Leading patterns like "B)", "(B)", "B." — check the trimmed start
    let stripped = upper.trim_start();
    if let Some(letter) = next_letter(stripped, valid) {
        return Some(letter);
    }
    // Final fallback: any standalone letter token anywhere.
    next_letter(&upper, valid)
}

fn next_letter(s: &str, valid: &[String]) -> Option<String> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip non-alphanumerics.
        while i < bytes.len() && !bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        if start == i {
            break;
        }
        let token = &s[start..i];
        // Single-letter answers and multi-character labels like "AB" both supported.
        if valid.iter().any(|c| c == token) {
            return Some(token.to_string());
        }
        // Special case: the marker "X" inside parentheses is followed by a `)`.
        // We've already split on non-alnum, so the bare letter is what we see.
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::EvalChatMessage;

    fn ex(target: &str) -> EvalExample {
        EvalExample {
            id: None,
            messages: vec![EvalChatMessage {
                role: "user".into(),
                content: "q".into(),
            }],
            target: Some(target.to_string()),
            aliases: Vec::new(),
            tags: Vec::new(),
            metadata: None,
            scorer: None,
            generation: None,
            weight: 1.0,
        }
    }

    fn choices() -> Vec<String> {
        vec!["A".into(), "B".into(), "C".into(), "D".into()]
    }

    #[test]
    fn answer_colon_pattern() {
        let (_, kind, _) = score(&ex("B"), "Answer: B because...", &choices()).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn parenthesized_leading_letter() {
        let (_, kind, _) = score(&ex("C"), "(C) is correct.", &choices()).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn dotted_leading_letter() {
        let (_, kind, _) = score(&ex("A"), "A. The answer is A.", &choices()).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn wrong_letter_fails() {
        let (_, kind, _) = score(&ex("A"), "Answer: B", &choices()).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn no_letter_is_invalid() {
        let (_, kind, _) = score(&ex("A"), "I am not sure.", &choices()).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Invalid);
    }

    #[test]
    fn multi_char_label_supported() {
        let choices = vec!["AB".into(), "CD".into()];
        let (_, kind, _) = score(&ex("AB"), "Answer: AB", &choices).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn aliases_accept_alternates() {
        let mut e = ex("A");
        e.aliases = vec!["B".into()];
        let (_, kind, _) = score(&e, "Answer: B", &choices()).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }
}
