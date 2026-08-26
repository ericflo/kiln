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
    let target = example
        .target
        .as_deref()
        .ok_or(ScorerError::MissingTarget {
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
    // Strongest signal: explicit answer markers. Models revise themselves
    // ("the answer is A... no wait, the answer is C"), so the LAST marker
    // occurrence wins, not the first.
    let mut marker_hit: Option<(usize, String)> = None;
    for marker in ["ANSWER:", "ANSWER IS", "FINAL ANSWER:", "FINAL ANSWER IS"] {
        for (idx, _) in upper.match_indices(marker) {
            let tail = &upper[idx + marker.len()..];
            // Colon forms ("Answer: X") are deliberate label slots — take
            // the first valid token verbatim. Prose forms ("answer is …")
            // need the article guard: "the answer is A bit tricky" must
            // not read the article as choice A.
            let letter = if marker.ends_with(':') {
                next_letter(tail, valid, false)
            } else {
                next_letter(tail, valid, true)
            };
            if let Some(letter) = letter
                && marker_hit.as_ref().is_none_or(|(pos, _)| idx >= *pos)
            {
                marker_hit = Some((idx, letter));
            }
        }
    }
    if let Some((_, letter)) = marker_hit {
        return Some(letter);
    }
    // Leading patterns like "B)", "(B)", "B." — by definition only the
    // FIRST token can be "leading"; scanning further belongs to the
    // fallback below (where the article guard applies).
    if let Some(letter) = first_token(upper.trim_start(), valid) {
        return Some(letter);
    }
    // Final fallback: the LAST standalone valid token anywhere — a model
    // that weighs options ("I considered A but the right choice is D")
    // states its conclusion last.
    next_letter_from_end(&upper, valid)
}

/// True when a single-letter candidate at `..end` reads as English ("A"
/// article / "I" pronoun) rather than an answer: i.e. it is immediately
/// followed by whitespace and another word. Punctuation or end-of-text
/// after the letter means it stands alone as an answer.
fn looks_like_article(s: &str, token: &str, end: usize) -> bool {
    if token != "A" && token != "I" {
        return false;
    }
    let rest = &s[end..];
    let trimmed = rest.trim_start();
    trimmed.len() < rest.len() && trimmed.starts_with(|c: char| c.is_ascii_alphanumeric())
}

/// First valid token scanning forward. With `guard_articles`, skips
/// "A"/"I" hits that read as English words.
fn next_letter(s: &str, valid: &[String], guard_articles: bool) -> Option<String> {
    for (token, end) in tokens(s) {
        if valid.iter().any(|c| c == token) {
            if guard_articles && looks_like_article(s, token, end) {
                continue;
            }
            return Some(token.to_string());
        }
    }
    None
}

/// Last valid token in the text, with the article guard always on.
fn next_letter_from_end(s: &str, valid: &[String]) -> Option<String> {
    let mut last: Option<String> = None;
    for (token, end) in tokens(s) {
        if valid.iter().any(|c| c == token) && !looks_like_article(s, token, end) {
            last = Some(token.to_string());
        }
    }
    last
}

/// The first alphanumeric run only — for leading-pattern detection.
fn first_token(s: &str, valid: &[String]) -> Option<String> {
    let (token, _) = tokens(s).next()?;
    valid.iter().any(|c| c == token).then(|| token.to_string())
}

/// Iterator over (token, end_byte_offset) alphanumeric runs.
fn tokens(s: &str) -> impl Iterator<Item = (&str, usize)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < bytes.len() && !bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        (start < i).then(|| (&s[start..i], i))
    })
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
    fn last_marker_wins_over_self_revision() {
        let (_, kind, _) = score(
            &ex("C"),
            "The answer is A... wait, re-reading the question, the answer is C.",
            &choices(),
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn article_a_after_answer_is_does_not_match() {
        let (_, kind, _) = score(
            &ex("C"),
            "Well, the answer is a bit tricky here, but it's C",
            &choices(),
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn deliberation_resolves_to_final_mention() {
        let (_, kind, detail) = score(
            &ex("D"),
            "I considered A but the correct choice is D",
            &choices(),
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass, "{detail:?}");
    }

    #[test]
    fn leading_pattern_only_matches_first_token() {
        // "It seems B..." has no leading-letter pattern; the fallback picks
        // the last standalone mention.
        let (_, kind, _) = score(&ex("B"), "It seems clear that B is right", &choices()).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn roman_numeral_choices_work_with_pronoun_guard() {
        let roman: Vec<String> = vec!["I".into(), "II".into(), "III".into()];
        let (_, kind, _) = score(&ex("II"), "The answer is II", &roman).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
        // A bare English "I think..." must not read the pronoun as choice I.
        let (_, kind, _) = score(&ex("III"), "I think the answer is III", &roman).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn answer_colon_a_with_following_word_still_matches() {
        // Colon form is a deliberate label slot — no article guard.
        let (_, kind, _) = score(&ex("A"), "Answer: A because it is prime", &choices()).unwrap();
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
