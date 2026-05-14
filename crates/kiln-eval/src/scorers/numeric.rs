//! Numeric tolerance scorer.
//!
//! Extracts the *last* signed number from the completion (after standard
//! cleanup — commas as thousands separators, leading `$`, trailing units) and
//! compares it to the example target. Accept iff
//! `|got - target| <= atol + rtol * |target|`.

use serde::{Deserialize, Serialize};

use crate::result::EvalOutcomeKind;
use crate::scorers::ScorerError;
use crate::suite::EvalExample;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct NumericTolerance {
    #[serde(default)]
    pub atol: f64,
    #[serde(default)]
    pub rtol: f64,
    /// When true, the scorer only accepts integers. Useful for arithmetic
    /// suites where "2" is the only correct answer and "2.0" should also pass.
    #[serde(default)]
    pub integer_only: bool,
}

impl Default for NumericTolerance {
    fn default() -> Self {
        Self {
            atol: 0.0,
            rtol: 0.0,
            integer_only: false,
        }
    }
}

pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    tol: &NumericTolerance,
) -> Result<(f32, EvalOutcomeKind, Option<String>), ScorerError> {
    let target_raw = example.target.as_deref().ok_or(ScorerError::MissingTarget {
        kind: "numeric_tolerance",
    })?;
    let target = match parse_number(target_raw) {
        Some(v) => v,
        None => {
            return Err(ScorerError::MissingTarget {
                kind: "numeric_tolerance (unparseable target)",
            });
        }
    };

    let got = match extract_last_number(completion_text) {
        Some(v) => v,
        None => {
            return Ok((
                0.0,
                EvalOutcomeKind::Invalid,
                Some("no number found in completion".into()),
            ));
        }
    };

    if tol.integer_only && got.fract() != 0.0 {
        return Ok((
            0.0,
            EvalOutcomeKind::Fail,
            Some(format!("expected integer, got {got}")),
        ));
    }

    let tolerance = tol.atol + tol.rtol * target.abs();
    if (got - target).abs() <= tolerance {
        Ok((1.0, EvalOutcomeKind::Pass, None))
    } else {
        Ok((
            0.0,
            EvalOutcomeKind::Fail,
            Some(format!("expected {target}, got {got}")),
        ))
    }
}

fn parse_number(s: &str) -> Option<f64> {
    let cleaned: String = s
        .chars()
        .filter(|c| {
            c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')
        })
        .collect();
    cleaned.parse::<f64>().ok()
}

fn extract_last_number(text: &str) -> Option<f64> {
    // Strip common decorations. Commas inside a number are thousands
    // separators (`1,234.50`) and must be removed without replacing them
    // with whitespace, or "1,234" would parse as "234" only.
    let scrubbed: String = text
        .chars()
        .filter_map(|c| match c {
            ',' => None,
            '$' | '%' => Some(' '),
            other => Some(other),
        })
        .collect();
    let mut last: Option<f64> = None;
    let bytes = scrubbed.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let is_start = (c == b'-' || c == b'+')
            && i + 1 < bytes.len()
            && (bytes[i + 1].is_ascii_digit() || bytes[i + 1] == b'.');
        if c.is_ascii_digit() || c == b'.' || is_start {
            let start = i;
            if c == b'-' || c == b'+' {
                i += 1;
            }
            let mut saw_digit = false;
            let mut saw_dot = false;
            let mut saw_exp = false;
            while i < bytes.len() {
                let ch = bytes[i];
                if ch.is_ascii_digit() {
                    saw_digit = true;
                    i += 1;
                } else if ch == b'.' && !saw_dot && !saw_exp {
                    saw_dot = true;
                    i += 1;
                } else if (ch == b'e' || ch == b'E') && saw_digit && !saw_exp {
                    saw_exp = true;
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
                        i += 1;
                    }
                } else {
                    break;
                }
            }
            if saw_digit {
                if let Ok(val) = std::str::from_utf8(&bytes[start..i])
                    .unwrap_or("")
                    .parse::<f64>()
                {
                    last = Some(val);
                }
            }
        } else {
            i += 1;
        }
    }
    last
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

    #[test]
    fn last_number_extracted() {
        assert_eq!(extract_last_number("the answer is 42"), Some(42.0));
        assert_eq!(extract_last_number("Final: -3.14"), Some(-3.14));
        assert_eq!(extract_last_number("got 1, then 2, finally 3"), Some(3.0));
        assert_eq!(extract_last_number("price is $1,234.50"), Some(1234.50));
        assert_eq!(extract_last_number("no number here"), None);
        assert_eq!(extract_last_number("1.5e3 kg"), Some(1500.0));
    }

    #[test]
    fn pass_within_atol() {
        let (_, kind, _) = score(
            &ex("42"),
            "answer 41.5",
            &NumericTolerance {
                atol: 1.0,
                rtol: 0.0,
                integer_only: false,
            },
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn rtol_scales_with_magnitude() {
        let (_, kind, _) = score(
            &ex("1000"),
            "the answer is 1009",
            &NumericTolerance {
                atol: 0.0,
                rtol: 0.01,
                integer_only: false,
            },
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn no_number_is_invalid() {
        let (_, kind, _) =
            score(&ex("42"), "I don't know", &NumericTolerance::default()).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Invalid);
    }

    #[test]
    fn integer_only_rejects_floats() {
        let (_, kind, _) = score(
            &ex("3"),
            "3.5",
            &NumericTolerance {
                atol: 1.0,
                rtol: 0.0,
                integer_only: true,
            },
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }
}
