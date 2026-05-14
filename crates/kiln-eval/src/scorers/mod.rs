//! Scorers — pluggable functions from `(completion_text, target)` to an
//! `ExampleOutcome`. Every scorer is serializable so suites can declare them
//! in JSON.
//!
//! Adding a new scorer:
//!
//! 1. Add a variant to `Scorer`.
//! 2. Add a `kind_label()` arm.
//! 3. Implement the scoring logic in a sibling module under
//!    `crates/kiln-eval/src/scorers/<name>.rs` and dispatch in `score_completion`.
//! 4. Add unit tests covering pass / fail / invalid / edge cases.

use serde::{Deserialize, Serialize};

use crate::result::{EvalOutcomeKind, ExampleOutcome};
use crate::suite::EvalExample;

pub(crate) mod bash;
pub(crate) mod code;
pub mod contains;
pub(crate) mod exact_match;
pub(crate) mod json_validity;
pub mod llm_judge;
pub(crate) mod multiple_choice;
pub(crate) mod numeric;
pub(crate) mod regex_match;
pub(crate) mod tool_call;

pub use code::{CodeStyle, default_min_jaccard};
pub use numeric::NumericTolerance;
pub use tool_call::{ArgsScoring, NameMatch};

/// Built-in scorer variants. Custom scoring (anything that doesn't fit) can
/// be expressed as `Composite` or by sending the completion to an external
/// judge via `LlmJudge`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Scorer {
    /// Exact textual equality after optional normalization.
    ExactMatch {
        #[serde(default = "default_true")]
        case_sensitive: bool,
        #[serde(default = "default_true")]
        strip_whitespace: bool,
    },
    /// Substring containment. `phrases` lists the strings to look for;
    /// `mode` controls how matches combine.
    Contains {
        phrases: Vec<String>,
        #[serde(default)]
        mode: contains::ContainsMode,
        #[serde(default = "default_true")]
        case_sensitive: bool,
    },
    /// Regex match; an example passes when the pattern matches the
    /// completion. When `capture_group` is set, the captured slice is
    /// compared to the target instead of the full completion.
    Regex {
        pattern: String,
        #[serde(default)]
        capture_group: Option<usize>,
        #[serde(default = "default_true")]
        case_sensitive: bool,
    },
    /// JSON validity. The completion passes when `serde_json::from_str`
    /// succeeds. When `target` is set on the example, the parsed JSON must
    /// equal `serde_json::from_str(target)` after canonicalization.
    JsonValidity {
        #[serde(default)]
        require_object: bool,
        /// Optional list of JSON Pointer paths (`/foo/bar`) that must exist
        /// in the parsed output. Used to score structured tool-call shapes
        /// without requiring exact equality.
        #[serde(default)]
        required_paths: Vec<String>,
    },
    /// Multiple-choice scoring. The example target is the correct
    /// answer label (e.g. `"A"`). The model's output is reduced to the
    /// first letter / first answer-token and compared.
    MultipleChoice {
        #[serde(default = "default_choices")]
        choices: Vec<String>,
    },
    /// Numeric tolerance: extracts the last numeric token from the
    /// completion and accepts when it's within `atol + rtol * |target|`.
    NumericTolerance(NumericTolerance),
    /// LLM-judge scorer: produces a score by asking another model to grade
    /// the completion. The judge call itself is dispatched by the executor —
    /// the kiln-eval crate only carries the prompt template + parsing rules.
    LlmJudge {
        /// Adapter or base-model name used by the judge.
        #[serde(default)]
        judge_adapter: Option<String>,
        /// Prompt template with `{question}`, `{answer}`, `{target}`
        /// placeholders. Defaults to a binary "did the response correctly
        /// answer the question?" template.
        #[serde(default = "llm_judge::default_judge_template")]
        template: String,
        /// Regex used to pull a score out of the judge's reply. Must capture
        /// either a 0/1 binary outcome or a `[0, 1]` float in group 1.
        #[serde(default = "llm_judge::default_judge_regex")]
        score_regex: String,
    },
    /// Tool-call scorer: scores predicted tool calls against a target
    /// (extracted from `tool_calls` on the SFT trajectory). Both the
    /// chosen function name AND the structural shape of arguments are
    /// scored; individual string-valued arguments can be graded with
    /// nested scorers (great for prose-in-args and code-in-args).
    ToolCall {
        #[serde(default)]
        name_match: NameMatch,
        #[serde(default)]
        args: ArgsScoring,
        /// If set, the per-call score is the mean of name and arg sub-scores
        /// weighted by these values. Default (None) gives equal weight.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        weights: Option<tool_call::ToolCallWeights>,
    },
    /// Code-output scorer: extracts the first fenced code block from the
    /// completion (and from the target) and scores their relationship.
    Code {
        /// Optional language tag — when set, only fences with a matching
        /// tag are considered. `python` / `py` are interchangeable.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        /// What kind of code comparison to run.
        #[serde(default)]
        style: CodeStyle,
    },
    /// Composite: every sub-scorer must pass for the outcome to be Pass.
    /// Score is the mean of sub-scores. Use this for "JSON valid AND
    /// numeric answer correct" kinds of metrics.
    All { scorers: Vec<Scorer> },
    /// Composite: any sub-scorer passing counts as Pass. Score is the max.
    Any { scorers: Vec<Scorer> },
}

fn default_true() -> bool {
    true
}
fn default_choices() -> Vec<String> {
    vec!["A".into(), "B".into(), "C".into(), "D".into()]
}

impl Scorer {
    /// Stable label used in metric breakdowns. Borrowed from the variant name
    /// so dashboards can group runs without parsing the JSON shape.
    pub fn kind_label(&self) -> &'static str {
        match self {
            Scorer::ExactMatch { .. } => "exact_match",
            Scorer::Contains { .. } => "contains",
            Scorer::Regex { .. } => "regex",
            Scorer::JsonValidity { .. } => "json_validity",
            Scorer::MultipleChoice { .. } => "multiple_choice",
            Scorer::NumericTolerance(_) => "numeric_tolerance",
            Scorer::LlmJudge { .. } => "llm_judge",
            Scorer::ToolCall { .. } => "tool_call",
            Scorer::Code { .. } => "code",
            Scorer::All { .. } => "all",
            Scorer::Any { .. } => "any",
        }
    }

    /// Returns true if this scorer (or any nested scorer in a composite) is
    /// an LLM judge. The executor uses this to short-circuit dispatch into
    /// its judge sub-runner without inspecting the full enum tree at the
    /// call site.
    pub fn requires_judge(&self) -> bool {
        match self {
            Scorer::LlmJudge { .. } => true,
            Scorer::All { scorers } | Scorer::Any { scorers } => {
                scorers.iter().any(|s| s.requires_judge())
            }
            Scorer::ToolCall { args, .. } => args.requires_judge(),
            _ => false,
        }
    }
}

/// Scorer-level errors. Most scorers degrade to `EvalOutcomeKind::Invalid`
/// rather than erroring; this enum is for "the suite itself is broken"
/// classes (bad regex, missing target where required, etc.).
#[derive(Debug, thiserror::Error)]
pub enum ScorerError {
    #[error("invalid regex `{pattern}`: {msg}")]
    InvalidRegex { pattern: String, msg: String },
    #[error("scorer `{kind}` requires `target` on every example")]
    MissingTarget { kind: &'static str },
    #[error("scorer `{kind}` is not runnable offline; provide a JudgeRunner")]
    NeedsJudge { kind: &'static str },
}

/// Apply a scorer to a single completion, returning a populated
/// `ExampleOutcome` (minus the `tags` / `prompt_tokens` / latency fields
/// which the executor fills in at the call site).
///
/// `judge_runner` is used to handle `LlmJudge` scorers. If the scorer doesn't
/// need a judge, this argument is ignored. The default offline path passes
/// `&NoopJudgeRunner` and the LLM-judge sub-scorer falls back to an `Invalid`
/// outcome — that way the kiln-eval crate stays GPU-free.
pub fn score_completion(
    scorer: &Scorer,
    example: &EvalExample,
    completion_text: &str,
    judge_runner: &dyn JudgeRunner,
) -> Result<ExampleOutcome, ScorerError> {
    let example_id = example.resolved_id();
    let (score, kind, detail) = match scorer {
        Scorer::ExactMatch {
            case_sensitive,
            strip_whitespace,
        } => exact_match::score(example, completion_text, *case_sensitive, *strip_whitespace)?,
        Scorer::Contains {
            phrases,
            mode,
            case_sensitive,
        } => contains::score(completion_text, phrases, *mode, *case_sensitive),
        Scorer::Regex {
            pattern,
            capture_group,
            case_sensitive,
        } => regex_match::score(example, completion_text, pattern, *capture_group, *case_sensitive)?,
        Scorer::JsonValidity {
            require_object,
            required_paths,
        } => json_validity::score(example, completion_text, *require_object, required_paths),
        Scorer::MultipleChoice { choices } => {
            multiple_choice::score(example, completion_text, choices)?
        }
        Scorer::NumericTolerance(tol) => numeric::score(example, completion_text, tol)?,
        Scorer::LlmJudge {
            judge_adapter,
            template,
            score_regex,
        } => llm_judge::score(
            example,
            completion_text,
            judge_adapter.as_deref(),
            template,
            score_regex,
            judge_runner,
        )?,
        Scorer::ToolCall {
            name_match,
            args,
            weights,
        } => tool_call::score(
            example,
            completion_text,
            name_match,
            args,
            weights.as_ref(),
            judge_runner,
        )?,
        Scorer::Code { language, style } => {
            code::score(example, completion_text, language.as_deref(), style)?
        }
        Scorer::All { scorers } => {
            if scorers.is_empty() {
                (0.0, EvalOutcomeKind::Invalid, Some("empty composite".into()))
            } else {
                let mut sum = 0.0f32;
                let mut all_pass = true;
                let mut details = Vec::new();
                let mut any_error = false;
                for s in scorers {
                    let outcome = score_completion(s, example, completion_text, judge_runner)?;
                    sum += outcome.score;
                    if !matches!(outcome.kind, EvalOutcomeKind::Pass) {
                        all_pass = false;
                    }
                    if matches!(outcome.kind, EvalOutcomeKind::Error) {
                        any_error = true;
                    }
                    if let Some(d) = outcome.detail {
                        details.push(format!("{}: {d}", s.kind_label()));
                    }
                }
                let kind = if any_error {
                    EvalOutcomeKind::Error
                } else if all_pass {
                    EvalOutcomeKind::Pass
                } else {
                    EvalOutcomeKind::Fail
                };
                (
                    sum / scorers.len() as f32,
                    kind,
                    if details.is_empty() {
                        None
                    } else {
                        Some(details.join("; "))
                    },
                )
            }
        }
        Scorer::Any { scorers } => {
            if scorers.is_empty() {
                (0.0, EvalOutcomeKind::Invalid, Some("empty composite".into()))
            } else {
                let mut best = 0.0f32;
                let mut any_pass = false;
                let mut details = Vec::new();
                for s in scorers {
                    let outcome = score_completion(s, example, completion_text, judge_runner)?;
                    if outcome.score > best {
                        best = outcome.score;
                    }
                    if matches!(outcome.kind, EvalOutcomeKind::Pass) {
                        any_pass = true;
                    }
                    if let Some(d) = outcome.detail {
                        details.push(format!("{}: {d}", s.kind_label()));
                    }
                }
                let kind = if any_pass {
                    EvalOutcomeKind::Pass
                } else {
                    EvalOutcomeKind::Fail
                };
                (
                    best,
                    kind,
                    if details.is_empty() {
                        None
                    } else {
                        Some(details.join("; "))
                    },
                )
            }
        }
    };

    Ok(ExampleOutcome {
        example_id,
        completion_index: 0,
        completion_text: completion_text.to_string(),
        kind,
        score,
        detail,
        prompt_tokens: None,
        completion_tokens: None,
        latency_ms: None,
        tags: example.tags.clone(),
    })
}

/// Trait the executor uses to dispatch LlmJudge scoring back into the model
/// runner. Implementations may be synchronous (offline mock) or call into the
/// in-process completion API on the server. The crate ships a default
/// no-op implementation so unit tests can run without a model.
pub trait JudgeRunner: Send + Sync {
    /// Submit a single judge prompt. Returns the judge's reply text or `None`
    /// when the implementation cannot serve a judge call (e.g. base server in
    /// mock mode). When the runner returns `None`, the scorer downgrades the
    /// example to `EvalOutcomeKind::Invalid` rather than failing the run.
    fn judge(&self, adapter: Option<&str>, prompt: &str) -> Option<String>;
}

/// Default no-op judge runner — every call returns `None`. The eval engine
/// in kiln-server provides a `LiveJudgeRunner` backed by the model runner.
pub struct NoopJudgeRunner;

impl JudgeRunner for NoopJudgeRunner {
    fn judge(&self, _adapter: Option<&str>, _prompt: &str) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::EvalChatMessage;

    fn ex(target: Option<&str>) -> EvalExample {
        EvalExample {
            id: None,
            messages: vec![EvalChatMessage {
                role: "user".into(),
                content: "q".into(),
            }],
            target: target.map(str::to_string),
            aliases: Vec::new(),
            tags: Vec::new(),
            metadata: None,
            scorer: None,
            generation: None,
            weight: 1.0,
        }
    }

    #[test]
    fn kind_labels_cover_every_variant() {
        let cases: &[Scorer] = &[
            Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            Scorer::Contains {
                phrases: vec!["x".into()],
                mode: contains::ContainsMode::Any,
                case_sensitive: true,
            },
            Scorer::Regex {
                pattern: ".*".into(),
                capture_group: None,
                case_sensitive: true,
            },
            Scorer::JsonValidity {
                require_object: false,
                required_paths: Vec::new(),
            },
            Scorer::MultipleChoice {
                choices: default_choices(),
            },
            Scorer::NumericTolerance(NumericTolerance::default()),
            Scorer::LlmJudge {
                judge_adapter: None,
                template: llm_judge::default_judge_template(),
                score_regex: llm_judge::default_judge_regex(),
            },
            Scorer::All { scorers: vec![] },
            Scorer::Any { scorers: vec![] },
        ];
        for s in cases {
            assert!(!s.kind_label().is_empty());
        }
    }

    #[test]
    fn all_scorer_requires_pass_on_every_subscorer() {
        // Use Contains+Contains so we can test "both phrases present" vs
        // "only one present" without ExactMatch's whole-string requirement
        // accidentally fighting Contains's substring requirement.
        let scorer = Scorer::All {
            scorers: vec![
                Scorer::Contains {
                    phrases: vec!["hello".into()],
                    mode: contains::ContainsMode::Any,
                    case_sensitive: false,
                },
                Scorer::Contains {
                    phrases: vec!["x".into()],
                    mode: contains::ContainsMode::Any,
                    case_sensitive: true,
                },
            ],
        };
        let e = ex(Some("hello"));
        let out = score_completion(&scorer, &e, "hello x", &NoopJudgeRunner).unwrap();
        assert_eq!(out.kind, EvalOutcomeKind::Pass);
        let out = score_completion(&scorer, &e, "hello", &NoopJudgeRunner).unwrap();
        assert_eq!(out.kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn any_scorer_passes_on_first_pass() {
        let scorer = Scorer::Any {
            scorers: vec![
                Scorer::ExactMatch {
                    case_sensitive: false,
                    strip_whitespace: true,
                },
                Scorer::Contains {
                    phrases: vec!["x".into()],
                    mode: contains::ContainsMode::Any,
                    case_sensitive: true,
                },
            ],
        };
        let e = ex(Some("hello"));
        let out = score_completion(&scorer, &e, "we mention x", &NoopJudgeRunner).unwrap();
        assert_eq!(out.kind, EvalOutcomeKind::Pass);
        let out = score_completion(&scorer, &e, "no match here", &NoopJudgeRunner).unwrap();
        assert_eq!(out.kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn judge_scorer_with_noop_runner_is_invalid() {
        let scorer = Scorer::LlmJudge {
            judge_adapter: None,
            template: llm_judge::default_judge_template(),
            score_regex: llm_judge::default_judge_regex(),
        };
        let e = ex(Some("hi"));
        let out = score_completion(&scorer, &e, "hi", &NoopJudgeRunner).unwrap();
        assert_eq!(out.kind, EvalOutcomeKind::Invalid);
    }

    #[test]
    fn requires_judge_is_recursive() {
        let inner = Scorer::All {
            scorers: vec![
                Scorer::ExactMatch {
                    case_sensitive: false,
                    strip_whitespace: true,
                },
                Scorer::LlmJudge {
                    judge_adapter: None,
                    template: llm_judge::default_judge_template(),
                    score_regex: llm_judge::default_judge_regex(),
                },
            ],
        };
        assert!(inner.requires_judge());
        let flat = Scorer::Contains {
            phrases: vec!["x".into()],
            mode: contains::ContainsMode::Any,
            case_sensitive: true,
        };
        assert!(!flat.requires_judge());
    }
}
