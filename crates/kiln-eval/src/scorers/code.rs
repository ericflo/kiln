//! Code scorer.
//!
//! Extracts the first matching fenced code block (or top-level program in
//! "looks like code" content) from both the completion and the target, then
//! scores their relationship under one of four styles:
//!
//! - `AnyBlock` — pass if the completion contains *any* code block in the
//!   declared language. Lenient sanity check.
//! - `ExactBlock { strip_comments }` — strict equality after normalization.
//! - `TokenSimilarity { min_jaccard }` — pass when the Jaccard similarity
//!   of code-token sets clears the threshold (default 0.6). The default
//!   for "great by default" usage: lets the model paraphrase variable
//!   names while still catching missing logic.
//! - `LineCoverage { min_coverage }` — pass when the fraction of target
//!   *non-trivial* lines (≥4 chars, not pure comment) that appear in the
//!   completion clears `min_coverage`. Useful for "did you call all the
//!   right APIs" without insisting on perfect formatting.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::result::EvalOutcomeKind;
use crate::scorers::ScorerError;
use crate::suite::EvalExample;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CodeStyle {
    /// Pass if the completion contains any code block of the declared
    /// language. Useful as a precondition.
    AnyBlock,
    /// Strict block equality after normalization (whitespace + optional
    /// comment stripping).
    ExactBlock {
        #[serde(default)]
        strip_comments: bool,
    },
    /// Jaccard on code-token sets.
    TokenSimilarity {
        #[serde(default = "default_min_jaccard")]
        min_jaccard: f32,
    },
    /// Line-coverage: fraction of target lines present in the completion.
    LineCoverage {
        #[serde(default = "default_min_coverage")]
        min_coverage: f32,
    },
}

pub fn default_min_jaccard() -> f32 {
    0.6
}
fn default_min_coverage() -> f32 {
    0.7
}

impl Default for CodeStyle {
    fn default() -> Self {
        CodeStyle::TokenSimilarity {
            min_jaccard: default_min_jaccard(),
        }
    }
}

pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    language: Option<&str>,
    style: &CodeStyle,
) -> Result<(f32, EvalOutcomeKind, Option<String>), ScorerError> {
    let target = example.target.as_deref().ok_or(ScorerError::MissingTarget {
        kind: "code",
    })?;
    let target_block = extract_block(target, language);
    let completion_block = extract_block(completion_text, language);

    match style {
        CodeStyle::AnyBlock => {
            if completion_block.is_some() {
                Ok((1.0, EvalOutcomeKind::Pass, None))
            } else {
                Ok((
                    0.0,
                    EvalOutcomeKind::Fail,
                    Some(format!(
                        "no code block (language={}) found in completion",
                        language.unwrap_or("any")
                    )),
                ))
            }
        }
        CodeStyle::ExactBlock { strip_comments } => {
            let lang_for_strip = language;
            let t = normalize(target_block.as_deref().unwrap_or(target), *strip_comments, lang_for_strip);
            let p = normalize(
                completion_block.as_deref().unwrap_or(completion_text),
                *strip_comments,
                lang_for_strip,
            );
            if t == p {
                Ok((1.0, EvalOutcomeKind::Pass, None))
            } else {
                Ok((
                    0.0,
                    EvalOutcomeKind::Fail,
                    Some("code blocks differ after normalization".into()),
                ))
            }
        }
        CodeStyle::TokenSimilarity { min_jaccard } => {
            let t_tokens = tokens(target_block.as_deref().unwrap_or(target));
            let p_tokens = tokens(completion_block.as_deref().unwrap_or(completion_text));
            let score = jaccard(&t_tokens, &p_tokens);
            let kind = if score >= *min_jaccard {
                EvalOutcomeKind::Pass
            } else {
                EvalOutcomeKind::Fail
            };
            Ok((
                score,
                kind,
                Some(format!(
                    "jaccard={score:.2} (threshold {min_jaccard:.2})"
                )),
            ))
        }
        CodeStyle::LineCoverage { min_coverage } => {
            let t_lines: BTreeSet<String> = significant_lines(target_block.as_deref().unwrap_or(target));
            let p_lines: BTreeSet<String> = significant_lines(completion_block.as_deref().unwrap_or(completion_text));
            if t_lines.is_empty() {
                return Ok((
                    1.0,
                    EvalOutcomeKind::Pass,
                    Some("target had no significant lines".into()),
                ));
            }
            let hit = t_lines.intersection(&p_lines).count();
            let coverage = hit as f32 / t_lines.len() as f32;
            let kind = if coverage >= *min_coverage {
                EvalOutcomeKind::Pass
            } else {
                EvalOutcomeKind::Fail
            };
            Ok((
                coverage,
                kind,
                Some(format!(
                    "coverage={coverage:.2} ({hit}/{} lines, threshold {min_coverage:.2})",
                    t_lines.len()
                )),
            ))
        }
    }
}

/// Extract the first code block matching `language`. When `language` is
/// `None`, accepts any fenced block. Also accepts content without fences
/// when the entire string looks like code (heuristic: ≥2 lines with at
/// least one common-keyword token).
pub fn extract_block(text: &str, language: Option<&str>) -> Option<String> {
    let lang_norm = language.map(normalize_lang);
    let mut search_from = 0usize;
    while let Some(rel) = text[search_from..].find("```") {
        let start = search_from + rel;
        let after = &text[start + 3..];
        let (lang_raw, rest) = after.split_once('\n').unwrap_or((after, ""));
        let block_lang = normalize_lang(lang_raw.trim());
        let body = rest.split("```").next().unwrap_or("").to_string();
        let matches = match (&lang_norm, block_lang.as_str()) {
            (Some(want), have) => want == have || (want == "node" && have == "javascript"),
            (None, _) => true,
        };
        if matches && !body.trim().is_empty() {
            return Some(body);
        }
        // Advance past this fence's close.
        let close_offset = rest.find("```").map(|o| o + 3).unwrap_or(rest.len());
        search_from = start + 3 + lang_raw.len() + 1 + close_offset;
        if search_from >= text.len() {
            break;
        }
    }
    // Fence-less fallback: when the content already looks code-shaped and
    // no language is required, return the trimmed body verbatim.
    if language.is_none() && looks_like_code(text) {
        return Some(text.to_string());
    }
    None
}

fn normalize_lang(s: &str) -> String {
    match s.trim().to_ascii_lowercase().as_str() {
        "py" => "python".into(),
        "rs" => "rust".into(),
        "ts" | "tsx" => "typescript".into(),
        "js" | "jsx" | "mjs" | "cjs" => "javascript".into(),
        "rb" => "ruby".into(),
        "kt" => "kotlin".into(),
        "yml" => "yaml".into(),
        "sh" => "bash".into(),
        other => other.into(),
    }
}

fn looks_like_code(text: &str) -> bool {
    let lines = text.lines().collect::<Vec<_>>();
    if lines.len() < 2 {
        return false;
    }
    let keywords = [
        "def ", "fn ", "function ", "class ", "import ", "from ", "return ", "let ", "const ",
        "var ", "if ", "else", "for ", "while ", "use ", "pub ", "package ", "struct ",
    ];
    let hits = lines
        .iter()
        .filter(|l| keywords.iter().any(|k| l.contains(k)))
        .count();
    hits >= 1 && lines.iter().any(|l| l.starts_with(' ') || l.contains('{') || l.contains(';'))
}

fn normalize(code: &str, strip_comments: bool, language: Option<&str>) -> String {
    let mut out = String::with_capacity(code.len());
    for line in code.lines() {
        let line = if strip_comments {
            strip_line_comment(line, language)
        } else {
            line.to_string()
        };
        let trimmed = line.trim_end();
        if !trimmed.is_empty() {
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    out
}

fn strip_line_comment(line: &str, language: Option<&str>) -> String {
    let marker = match language.map(normalize_lang) {
        Some(ref l) if l == "python" || l == "ruby" || l == "bash" || l == "yaml" => "#",
        Some(ref l) if l == "rust" || l == "javascript" || l == "typescript"
            || l == "go" || l == "java" || l == "c" || l == "cpp" => "//",
        _ => return line.to_string(),
    };
    if let Some(idx) = line.find(marker) {
        // Best-effort: don't strip when the marker is inside a string.
        let before = &line[..idx];
        let single = before.chars().filter(|c| *c == '\'').count();
        let double = before.chars().filter(|c| *c == '"').count();
        if single % 2 == 0 && double % 2 == 0 {
            return before.trim_end().to_string();
        }
    }
    line.to_string()
}

fn tokens(code: &str) -> BTreeSet<String> {
    code.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_string())
        .collect()
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count();
    let uni = a.union(b).count();
    if uni == 0 {
        0.0
    } else {
        inter as f32 / uni as f32
    }
}

fn significant_lines(code: &str) -> BTreeSet<String> {
    code.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| {
            l.len() >= 4
                && !l.starts_with('#')
                && !l.starts_with("//")
                && !l.starts_with("/*")
                && !l.starts_with('*')
                && !l.starts_with("--")
        })
        .collect()
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
            target: Some(target.into()),
            aliases: vec![],
            tags: vec![],
            metadata: None,
            scorer: None,
            generation: None,
            weight: 1.0,
        }
    }

    #[test]
    fn extract_block_with_language_filters() {
        let text = "Here's some code:\n```python\nprint(1)\n```\nand also\n```rust\nfn main() {}\n```";
        let py = extract_block(text, Some("python")).unwrap();
        assert!(py.contains("print(1)"));
        let rs = extract_block(text, Some("rust")).unwrap();
        assert!(rs.contains("fn main"));
        // js falls through to None.
        assert!(extract_block(text, Some("javascript")).is_none());
    }

    #[test]
    fn any_block_style_passes_on_presence() {
        let target = "```python\nprint(1)\n```";
        let pred = "Here you go:\n```python\nprint(2)\n```";
        let (_, kind, _) = score(&ex(target), pred, Some("python"), &CodeStyle::AnyBlock).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
        let (_, kind, _) =
            score(&ex(target), "no code", Some("python"), &CodeStyle::AnyBlock).unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn exact_block_normalizes_whitespace() {
        let target = "```python\ndef add(a, b):\n    return a + b\n```";
        let pred = "```python\n\ndef add(a, b):\n    return a + b\n\n```";
        let (_, kind, _) = score(
            &ex(target),
            pred,
            Some("python"),
            &CodeStyle::ExactBlock {
                strip_comments: false,
            },
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn exact_block_can_strip_python_comments() {
        let target = "```python\ndef f():\n    return 1\n```";
        let pred = "```python\ndef f():\n    # a comment\n    return 1\n```";
        let (_, kind, _) = score(
            &ex(target),
            pred,
            Some("python"),
            &CodeStyle::ExactBlock {
                strip_comments: true,
            },
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn token_similarity_pass_on_paraphrased_code() {
        let target = "```python\ndef multiply(a, b):\n    return a * b\n```";
        let pred = "```python\ndef multiply(x, y):\n    return x * y\n```";
        let (s, kind, _) = score(
            &ex(target),
            pred,
            Some("python"),
            &CodeStyle::TokenSimilarity { min_jaccard: 0.3 },
        )
        .unwrap();
        assert!(s > 0.3, "score was {s}");
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn line_coverage_passes_when_target_lines_present() {
        let target = "```python\nimport os\nimport sys\nprint(os.getcwd())\n```";
        let pred = "```python\nimport os\nimport sys\nimport json\nprint(os.getcwd())\n```";
        let (_, kind, _) = score(
            &ex(target),
            pred,
            Some("python"),
            &CodeStyle::LineCoverage { min_coverage: 0.8 },
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn line_coverage_fails_on_missing_lines() {
        let target = "```python\nimport os\nimport sys\nimport json\n```";
        let pred = "```python\nimport os\n```";
        let (_, kind, _) = score(
            &ex(target),
            pred,
            Some("python"),
            &CodeStyle::LineCoverage { min_coverage: 0.8 },
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn extract_block_falls_back_on_fenceless_code() {
        let text = "def add(a, b):\n    return a + b";
        let extracted = extract_block(text, None).unwrap();
        assert!(extracted.contains("def add"));
    }

    #[test]
    fn js_alias_matches_node_language() {
        let text = "```javascript\nconst x = 1;\n```";
        let extracted = extract_block(text, Some("node")).unwrap();
        assert!(extracted.contains("const x"));
    }
}
