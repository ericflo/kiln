//! Substring containment scorer.

use serde::{Deserialize, Serialize};

use crate::result::EvalOutcomeKind;

/// Extract up to `k` distinctive lower-cased multi-word phrases from
/// arbitrary text. Public so the tool-call / json-validity scorers can
/// reuse the same heuristic without duplicating the stopword list.
pub fn naive_key_phrases(text: &str, k: usize) -> Vec<String> {
    let stop: std::collections::HashSet<&str> = [
        "the", "and", "for", "with", "that", "this", "from", "you", "are", "not", "but", "have",
        "has", "had", "was", "were", "will", "would", "could", "should", "their", "your", "they",
        "them", "than", "then", "into", "over", "also", "some", "such", "more", "most",
        "much", "many", "very", "just", "like", "what", "which", "when", "where", "while",
    ]
    .into_iter()
    .collect();
    let mut phrases: Vec<String> = Vec::new();
    for line in text.lines() {
        let words: Vec<&str> = line
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| {
                w.chars().count() >= 4 && !stop.contains(&w.to_ascii_lowercase().as_str())
            })
            .collect();
        for window in words.windows(3) {
            let phrase = window.join(" ").to_lowercase();
            if !phrases.iter().any(|p| p == &phrase) {
                phrases.push(phrase);
                if phrases.len() >= k {
                    return phrases;
                }
            }
        }
    }
    phrases
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContainsMode {
    /// Pass when any phrase appears anywhere in the completion.
    #[default]
    Any,
    /// Pass only when every phrase appears.
    All,
    /// Pass only when none of the phrases appear (useful as a guard for
    /// safety / undesired-content evals).
    None,
}

pub(super) fn score(
    completion_text: &str,
    phrases: &[String],
    mode: ContainsMode,
    case_sensitive: bool,
) -> (f32, EvalOutcomeKind, Option<String>) {
    if phrases.is_empty() {
        return (0.0, EvalOutcomeKind::Invalid, Some("no phrases configured".into()));
    }
    let haystack = if case_sensitive {
        completion_text.to_string()
    } else {
        completion_text.to_lowercase()
    };

    let matches: Vec<bool> = phrases
        .iter()
        .map(|p| {
            let needle = if case_sensitive {
                p.clone()
            } else {
                p.to_lowercase()
            };
            haystack.contains(&needle)
        })
        .collect();
    let num_hits = matches.iter().filter(|b| **b).count();

    match mode {
        ContainsMode::Any => {
            if num_hits > 0 {
                (1.0, EvalOutcomeKind::Pass, None)
            } else {
                (
                    0.0,
                    EvalOutcomeKind::Fail,
                    Some(format!("none of {} phrases matched", phrases.len())),
                )
            }
        }
        ContainsMode::All => {
            if num_hits == phrases.len() {
                (1.0, EvalOutcomeKind::Pass, None)
            } else {
                (
                    num_hits as f32 / phrases.len() as f32,
                    EvalOutcomeKind::Fail,
                    Some(format!("{num_hits}/{} matched", phrases.len())),
                )
            }
        }
        ContainsMode::None => {
            if num_hits == 0 {
                (1.0, EvalOutcomeKind::Pass, None)
            } else {
                (
                    0.0,
                    EvalOutcomeKind::Fail,
                    Some(format!("matched {num_hits} disallowed phrase(s)")),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_mode_passes_on_any_hit() {
        let (_, kind, _) =
            score("hello world", &["world".into(), "missing".into()], ContainsMode::Any, true);
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn all_mode_requires_every_hit() {
        let (s, kind, _) =
            score("hello world", &["hello".into(), "missing".into()], ContainsMode::All, true);
        assert!(matches!(kind, EvalOutcomeKind::Fail));
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn none_mode_penalizes_disallowed_content() {
        let (_, kind, _) =
            score("safe text", &["unsafe".into()], ContainsMode::None, true);
        assert_eq!(kind, EvalOutcomeKind::Pass);
        let (_, kind2, _) =
            score("very unsafe", &["unsafe".into()], ContainsMode::None, true);
        assert_eq!(kind2, EvalOutcomeKind::Fail);
    }

    #[test]
    fn case_insensitive_mode_lowercases_both_sides() {
        let (_, kind, _) =
            score("Hello WORLD", &["world".into()], ContainsMode::Any, false);
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn empty_phrases_is_invalid() {
        let (_, kind, _) = score("x", &[], ContainsMode::Any, true);
        assert_eq!(kind, EvalOutcomeKind::Invalid);
    }
}
