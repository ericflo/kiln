//! Evaluation framework for Kiln.
//!
//! This crate is the data + scoring layer of the eval system. It is
//! intentionally pure (no GPU, no network, no model runner) so it can be
//! built and tested on any host. The kiln-server crate provides the HTTP
//! surface and runs eval suites against the live `ModelRunner`.
//!
//! The mental model:
//!
//! - An `EvalSuite` is a named collection of `EvalExample`s plus suite-wide
//!   defaults (generation params, scorer).
//! - An `EvalExample` is a chat-style prompt (the same `messages` shape that
//!   `/v1/chat/completions` accepts) with a target answer and an optional
//!   per-example scorer override.
//! - A `Scorer` turns the model's actual completion text into a score in
//!   `[0.0, 1.0]` and an outcome label. Scorers are pluggable; the built-in
//!   variants cover the verifiable-reward space that matters for LoRA
//!   evaluation (exact match, regex, JSON validity, multiple choice,
//!   numeric tolerance, substring contains).
//! - An `EvalResult` is the per-example record plus aggregate metrics.
//!
//! The server layer plumbs `EvalSuite` → batched generation → `Scorer` →
//! `EvalResult` so callers see the same shape whether they kick off an eval
//! over HTTP, the CLI, or the post-training auto-eval hook.

pub mod result;
pub mod scorers;
pub mod suite;
pub mod synthesis;

pub use result::{
    AggregateMetrics, EvalJobState, EvalOutcomeKind, EvalProgress, EvalResult, ExampleOutcome,
    LatencyStats, ScorerBreakdown, SuiteResult,
};
pub use scorers::{
    ArgsScoring, CodeStyle, NameMatch, NumericTolerance, Scorer, ScorerError, score_completion,
};
pub use suite::{
    EvalChatMessage, EvalCompareSpec, EvalExample, EvalGenerationParams, EvalSuite,
    EvalSuiteSummary, PostEvalConfig, default_max_tokens, default_temperature,
};
pub use synthesis::{
    Sampling, ScorerChoice, SftConversation, SftMessage, SynthesisConfig, SynthesisError,
    SynthesisStats, SynthesisStrategy, auto_detect_scorer, synthesize_suite,
};

/// Schema version of suite JSON files. Bumped when an incompatible field is
/// added. Suite files without `schema_version` are assumed to be v1 so existing
/// JSONL on disk keeps loading.
pub const SUITE_SCHEMA_VERSION: u32 = 1;
