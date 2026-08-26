//! Result types: per-example outcomes, suite-level aggregates, and the
//! `EvalJobState` lifecycle exposed via the HTTP API.

use std::collections::BTreeMap;

pub use kiln_core::thinking_budget::ThinkingBudgetOutcome as EvalThinkingBudgetOutcome;
pub use kiln_core::thinking_budget::ThinkingBudgetRecord as EvalThinkingBudget;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::suite::EvalAggregation;

pub const EVAL_RESULT_SCHEMA_VERSION: u32 = 2;

/// Serde adapter for exact `u64` values on JSON-facing provenance fields.
/// JavaScript cannot represent every `u64` as a number, so new result fields
/// serialize as decimal strings while deserialization accepts either form.
pub mod u64_decimal {
    use serde::{Deserialize, Deserializer, Serializer, de};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Wire {
        Decimal(String),
        Number(u64),
    }

    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        match Wire::deserialize(deserializer)? {
            Wire::Decimal(value) => value.parse().map_err(de::Error::custom),
            Wire::Number(value) => Ok(value),
        }
    }
}

pub mod optional_u64_decimal {
    use serde::{Deserialize, Deserializer, Serializer, de};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Wire {
        Decimal(String),
        Number(u64),
    }

    pub fn serialize<S>(value: &Option<u64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(value) => serializer.serialize_some(&value.to_string()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        match Option::<Wire>::deserialize(deserializer)? {
            Some(Wire::Decimal(value)) => value.parse().map(Some).map_err(de::Error::custom),
            Some(Wire::Number(value)) => Ok(Some(value)),
            None => Ok(None),
        }
    }
}

/// Version tag for the deterministic eval seed derivation contract.
pub const EVAL_SEED_DERIVATION_V1: &str = "kiln.eval-seed.v1";

/// Derive one stable decoder seed from a job/example/completion identity.
///
/// The example ID, rather than its ordinal or suite name, keeps a filtered
/// re-run on the same seed as the original job. Callers must persist both the
/// job seed and this derived value so the contract remains auditable if a
/// future schema changes the derivation.
pub fn derive_eval_completion_seed(
    effective_seed: u64,
    example_id: &str,
    completion_index: usize,
) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(EVAL_SEED_DERIVATION_V1.as_bytes());
    hasher.update([0]);
    hasher.update(effective_seed.to_le_bytes());
    hasher.update((example_id.len() as u64).to_le_bytes());
    hasher.update(example_id.as_bytes());
    hasher.update((completion_index as u64).to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

/// Lifecycle of an eval job (mirrors the training-job state machine so
/// dashboards can render both with one code path).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalJobState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// What happened to a single example.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalOutcomeKind {
    /// Scorer awarded a positive score (typically 1.0 for binary metrics).
    Pass,
    /// Scorer awarded zero or a below-threshold score.
    Fail,
    /// Scorer rejected the output (e.g. unparseable for numeric tolerance,
    /// invalid JSON for json_validity). Counted distinctly so suites can
    /// surface "what fraction of outputs even tried to follow the format".
    Invalid,
    /// Generation did not produce any output (timeout, cancellation, OOM,
    /// rejected by sampler). Counted as a failure for headline accuracy but
    /// surfaced separately for triage.
    Error,
}

/// Per-example record returned in the `EvalResult`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleOutcome {
    pub example_id: String,
    pub completion_index: usize,
    /// Exact decoder seed derived for this example/completion. Absent only on
    /// legacy results written before eval seeds were materialized at enqueue.
    #[serde(
        default,
        with = "optional_u64_decimal",
        skip_serializing_if = "Option::is_none"
    )]
    pub generation_seed: Option<u64>,
    pub completion_text: String,
    /// Exact decoder text before eval-only normalization. In particular, a
    /// prompt prefilled with `<think>` does not make the decoder repeat that
    /// opening tag, so `completion_text` restores it for scoring while this
    /// field preserves the model's actual continuation for audit/replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_completion_text: Option<String>,
    /// Resolved thinking-budget limits, provenance, and runtime outcome for
    /// this completion. Absent on legacy results and generations that failed
    /// before a completion was produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<EvalThinkingBudget>,
    pub kind: EvalOutcomeKind,
    /// Score in `[0.0, 1.0]`. Most scorers emit binary 0/1 outcomes; numeric
    /// tolerance and LLM-judge can emit continuous values.
    pub score: f32,
    /// Scorer-specific commentary surfaced to the user (e.g. "expected 185,
    /// got 186"). Free-form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Prompt tokens (resolved by tokenizer) attributable to this example,
    /// useful for cost dashboards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<usize>,
    /// Wall-clock latency for the generation that produced this completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    /// Tags inherited from the example (echoed so consumers don't need to
    /// re-join against the suite).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Metadata inherited from the example. Production trace suites use this
    /// for source_path/source_line/session/model provenance so raw eval
    /// results remain auditable without joining against the source suite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Qwen3.5 `<think>…</think>` reasoning extracted from the raw completion,
    /// when present. Stored separately so dashboards can show "the model
    /// thought for 432 chars before answering" without re-parsing on every
    /// render. `None` means the completion contained no thinking block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<String>,
    /// True when the model emitted an opening `<think>` that was never
    /// closed by `</think>`. These outcomes typically grade as Invalid.
    #[serde(default, skip_serializing_if = "is_false")]
    pub unclosed_thinking: bool,
}

/// One independent example statistic reduced from its raw completions.
/// Aggregate metrics and all decisions consume this shape; `ExampleOutcome`
/// remains the replay/audit record for each generated completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregatedExampleOutcome {
    pub example_id: String,
    pub kind: EvalOutcomeKind,
    pub score: f32,
    pub completion_indices: Vec<usize>,
    pub representative_completion_index: usize,
    pub num_pass: u32,
    pub num_fail: u32,
    pub num_invalid: u32,
    pub num_error: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EvalAggregationError {
    #[error("invalid aggregation definition: {0}")]
    InvalidDefinition(String),
    #[error(
        "example {example_id:?} has {actual} completions; aggregation {aggregation} requires exactly {expected}"
    )]
    CompletionCount {
        example_id: String,
        aggregation: String,
        expected: usize,
        actual: usize,
    },
    #[error("example {example_id:?} has completion indices {actual:?}; expected {expected:?}")]
    CompletionIndices {
        example_id: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("example {example_id:?} completion {completion_index} has invalid score {score}")]
    InvalidScore {
        example_id: String,
        completion_index: usize,
        score: String,
    },
    #[error("example {example_id:?} completions disagree on tags or metadata")]
    InconsistentExampleMetadata { example_id: String },
}

/// Reduce every complete group of raw completions to one example statistic.
/// Input order is preserved by first appearance, while completion order is
/// canonicalized and validated as the exact sequence `0..k`.
pub fn aggregate_example_outcomes(
    outcomes: &[ExampleOutcome],
    aggregation: EvalAggregation,
) -> Result<Vec<AggregatedExampleOutcome>, EvalAggregationError> {
    let k = aggregation.k();
    if k == 0 || k > 128 {
        return Err(EvalAggregationError::InvalidDefinition(format!(
            "{} requires k in 1..=128",
            aggregation.label()
        )));
    }
    if matches!(aggregation, EvalAggregation::MajorityAtK { .. }) && k.is_multiple_of(2) {
        return Err(EvalAggregationError::InvalidDefinition(format!(
            "{} requires an odd k so ties are impossible",
            aggregation.label()
        )));
    }

    let mut order = Vec::new();
    let mut groups: BTreeMap<String, Vec<&ExampleOutcome>> = BTreeMap::new();
    for outcome in outcomes {
        if !groups.contains_key(&outcome.example_id) {
            order.push(outcome.example_id.clone());
        }
        groups
            .entry(outcome.example_id.clone())
            .or_default()
            .push(outcome);
    }

    let expected_indices: Vec<usize> = (0..k).collect();
    let mut reduced = Vec::with_capacity(groups.len());
    for example_id in order {
        let mut group = groups.remove(&example_id).unwrap_or_default();
        if group.len() != k {
            return Err(EvalAggregationError::CompletionCount {
                example_id,
                aggregation: aggregation.label(),
                expected: k,
                actual: group.len(),
            });
        }
        group.sort_by_key(|outcome| outcome.completion_index);
        let actual_indices: Vec<usize> = group
            .iter()
            .map(|outcome| outcome.completion_index)
            .collect();
        if actual_indices != expected_indices {
            return Err(EvalAggregationError::CompletionIndices {
                example_id,
                expected: expected_indices.clone(),
                actual: actual_indices,
            });
        }

        let first = group[0];
        if group
            .iter()
            .any(|outcome| outcome.tags != first.tags || outcome.metadata != first.metadata)
        {
            return Err(EvalAggregationError::InconsistentExampleMetadata { example_id });
        }
        for outcome in &group {
            if !outcome.score.is_finite() || !(0.0..=1.0).contains(&outcome.score) {
                return Err(EvalAggregationError::InvalidScore {
                    example_id,
                    completion_index: outcome.completion_index,
                    score: outcome.score.to_string(),
                });
            }
        }

        let num_pass = group
            .iter()
            .filter(|outcome| outcome.kind == EvalOutcomeKind::Pass)
            .count() as u32;
        let num_fail = group
            .iter()
            .filter(|outcome| outcome.kind == EvalOutcomeKind::Fail)
            .count() as u32;
        let num_invalid = group
            .iter()
            .filter(|outcome| outcome.kind == EvalOutcomeKind::Invalid)
            .count() as u32;
        let num_error = group
            .iter()
            .filter(|outcome| outcome.kind == EvalOutcomeKind::Error)
            .count() as u32;
        let fallback_kind = if num_fail > 0 {
            EvalOutcomeKind::Fail
        } else if num_invalid > 0 {
            EvalOutcomeKind::Invalid
        } else {
            EvalOutcomeKind::Error
        };

        let (kind, score, representative_completion_index) = match aggregation {
            EvalAggregation::Single => (first.kind, first.score, 0),
            EvalAggregation::MeanAtK { .. } => {
                let score = group.iter().map(|outcome| outcome.score).sum::<f32>() / k as f32;
                let kind = if num_pass + num_fail == 0 {
                    fallback_kind
                } else if score >= 0.5 {
                    EvalOutcomeKind::Pass
                } else {
                    EvalOutcomeKind::Fail
                };
                let representative = group
                    .iter()
                    .min_by(|left, right| {
                        (left.score - score)
                            .abs()
                            .total_cmp(&(right.score - score).abs())
                            .then_with(|| left.completion_index.cmp(&right.completion_index))
                    })
                    .map(|outcome| outcome.completion_index)
                    .unwrap_or(0);
                (kind, score, representative)
            }
            EvalAggregation::PassAtK { .. } => {
                if let Some(passing) = group
                    .iter()
                    .find(|outcome| outcome.kind == EvalOutcomeKind::Pass)
                {
                    (EvalOutcomeKind::Pass, 1.0, passing.completion_index)
                } else {
                    (fallback_kind, 0.0, 0)
                }
            }
            EvalAggregation::MajorityAtK { .. } => {
                if num_pass as usize > k / 2 {
                    let representative = group
                        .iter()
                        .find(|outcome| outcome.kind == EvalOutcomeKind::Pass)
                        .map(|outcome| outcome.completion_index)
                        .unwrap_or(0);
                    (EvalOutcomeKind::Pass, 1.0, representative)
                } else {
                    let representative = group
                        .iter()
                        .find(|outcome| outcome.kind != EvalOutcomeKind::Pass)
                        .map(|outcome| outcome.completion_index)
                        .unwrap_or(0);
                    (fallback_kind, 0.0, representative)
                }
            }
        };

        reduced.push(AggregatedExampleOutcome {
            example_id,
            kind,
            score,
            completion_indices: expected_indices.clone(),
            representative_completion_index,
            num_pass,
            num_fail,
            num_invalid,
            num_error,
            tags: first.tags.clone(),
            metadata: first.metadata.clone(),
        });
    }
    Ok(reduced)
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Latency stats expressed as p50/p90/p99 in milliseconds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
}

impl LatencyStats {
    /// Compute percentiles from a slice of samples. Empty input produces an
    /// all-zeros stat block (callers can detect this via `samples.is_empty()`).
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut sorted: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 0 {
            return Self::default();
        }
        let p = |q: f64| {
            let idx = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
            sorted[idx]
        };
        let mean = sorted.iter().sum::<f64>() / (n as f64);
        Self {
            p50_ms: p(0.5),
            p90_ms: p(0.9),
            p99_ms: p(0.99),
            mean_ms: mean,
            max_ms: *sorted.last().unwrap_or(&0.0),
        }
    }
}

/// Per-scorer-kind breakdown — useful when a suite mixes scorers (e.g. an
/// answer-correctness scorer plus a JSON-format scorer applied to the same
/// outputs via per-example overrides).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScorerBreakdown {
    pub scorer_kind: String,
    pub num_examples: u32,
    pub mean_score: f32,
    pub pass_rate: f32,
}

/// Wilson-score confidence interval for a binomial pass rate.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct PassRateConfidenceInterval {
    pub confidence_level: f32,
    pub lower: f32,
    pub upper: f32,
}

/// Aggregate metrics over a single suite run against a single adapter.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Independent examples after the suite's declared reduction.
    pub num_examples: u32,
    /// Raw generated completions retained for audit and cost accounting.
    #[serde(default)]
    pub num_completions: u32,
    pub num_pass: u32,
    pub num_fail: u32,
    pub num_invalid: u32,
    pub num_error: u32,
    pub accuracy: f32,
    /// 95% Wilson interval around [`Self::accuracy`]. Production trace evals
    /// are random samples; this makes sample uncertainty visible alongside
    /// the point estimate.
    #[serde(default)]
    pub accuracy_confidence_interval: PassRateConfidenceInterval,
    pub mean_score: f32,
    /// Weighted mean over `EvalExample::weight`.
    pub weighted_mean_score: f32,
    pub latency: LatencyStats,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    /// Total wall-clock seconds the run took (start → finish).
    pub elapsed_secs: f64,
    /// Pass rate sliced by example tag. Useful for "easy/medium/hard" splits.
    pub pass_rate_by_tag: BTreeMap<String, f32>,
    /// Count-aware tag breakdown with confidence intervals. `pass_rate_by_tag`
    /// is retained for backward compatibility and quick dashboards.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tag_breakdown: BTreeMap<String, TagBreakdown>,
    /// Per-scorer-kind breakdown when the suite mixes scorers.
    pub by_scorer: Vec<ScorerBreakdown>,
    /// Per-tool-name pass rate, surfaced only when the suite exercises
    /// tool calls. The key is the target tool name (e.g. "get_weather");
    /// the value is `(num_examples_targeting_this_tool, passed)`. Lets
    /// users immediately spot "the model nails Read but flubs Edit".
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub pass_rate_by_tool: BTreeMap<String, ToolBreakdown>,
    /// Distribution of reasoning lengths (chars in the `<think>…</think>`
    /// block). Zero when no example produced a reasoning trace.
    #[serde(default)]
    pub reasoning_length: ReasoningLengthStats,
    /// Number of completions that emitted an opening `<think>` but never
    /// closed it. Surfaced separately because these are a distinct class
    /// of failure (generation timed out / hit max_tokens inside reasoning).
    #[serde(default)]
    pub num_unclosed_thinking: u32,
    /// Number of completions whose tool call wasn't in Qwen3.5's native
    /// XML form. Non-fatal but a useful "the model regressed from XML to
    /// JSON" signal during fine-tuning.
    #[serde(default)]
    pub num_non_xml_tool_calls: u32,
    /// Tool-confusion matrix: `target_tool → predicted_tool → count`.
    /// Populated for any example whose scorer was `ToolCall`. A target on
    /// `Read` whose model emitted `Write` shows up as
    /// `confusion_by_tool["Read"]["Write"] = 1`. Lets users see "the model
    /// confuses Edit/Write 4 times" in one glance.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub confusion_by_tool: BTreeMap<String, BTreeMap<String, u32>>,
    /// Number of predicted tool calls that missed a required arg in the
    /// declared schema. Only meaningful when the suite has a `tools`
    /// catalogue. A passing scorer might still log a schema violation if
    /// the *target* itself was missing the required field (rare — that's
    /// a data-quality bug worth surfacing).
    #[serde(default)]
    pub num_schema_missing_required: u32,
    /// Number of predicted tool calls with arg keys not declared on the
    /// tool. Usually the model hallucinating extra kwargs.
    #[serde(default)]
    pub num_schema_extra_unknown: u32,
}

/// Per-tool aggregate counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolBreakdown {
    pub num_examples: u32,
    pub num_pass: u32,
    pub pass_rate: f32,
    #[serde(default)]
    pub confidence_interval: PassRateConfidenceInterval,
}

/// Per-tag aggregate counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TagBreakdown {
    pub num_examples: u32,
    pub num_pass: u32,
    pub pass_rate: f32,
    #[serde(default)]
    pub confidence_interval: PassRateConfidenceInterval,
}

/// Compact summary of reasoning-block lengths across a suite run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningLengthStats {
    pub num_with_thinking: u32,
    pub mean_chars: f64,
    pub p50_chars: u32,
    pub p90_chars: u32,
    pub max_chars: u32,
}

impl AggregateMetrics {
    /// Compute aggregate metrics from a vector of per-example outcomes and
    /// the example weights they map to. Latency samples are read from
    /// `ExampleOutcome::latency_ms` where present.
    ///
    /// `target_tool_by_example` is an optional map from example_id to the
    /// target tool name (parsed from the example's `target` field by the
    /// executor). When present, the result includes a per-tool pass-rate
    /// breakdown — invaluable for agentic-suite dashboards.
    pub fn compute(
        aggregated_outcomes: &[AggregatedExampleOutcome],
        outcomes: &[ExampleOutcome],
        weights: &BTreeMap<String, f32>,
        tags_by_example: &BTreeMap<String, Vec<String>>,
        scorer_kind_by_example: &BTreeMap<String, &'static str>,
        elapsed_secs: f64,
    ) -> Self {
        Self::compute_with_tools(
            aggregated_outcomes,
            outcomes,
            weights,
            tags_by_example,
            scorer_kind_by_example,
            &BTreeMap::new(),
            &BTreeMap::new(),
            elapsed_secs,
        )
    }

    /// Same as [`Self::compute`] but takes per-example target tool names
    /// + predicted tool names so the aggregate carries per-tool pass-rate
    /// breakdown AND a tool-confusion matrix.
    ///
    /// `predicted_tool_by_outcome` is keyed by `(example_id,
    /// completion_index)` so n>1 runs don't conflate predictions.
    /// `schema_violations_by_outcome` carries the (missing, extra) tally
    /// per outcome — those are populated by the executor only when the
    /// suite declared a `tools` catalogue.
    pub fn compute_with_tools(
        aggregated_outcomes: &[AggregatedExampleOutcome],
        outcomes: &[ExampleOutcome],
        weights: &BTreeMap<String, f32>,
        tags_by_example: &BTreeMap<String, Vec<String>>,
        scorer_kind_by_example: &BTreeMap<String, &'static str>,
        target_tool_by_example: &BTreeMap<String, String>,
        predicted_tool_by_outcome: &BTreeMap<(String, usize), String>,
        elapsed_secs: f64,
    ) -> Self {
        Self::compute_with_tools_full(
            aggregated_outcomes,
            outcomes,
            weights,
            tags_by_example,
            scorer_kind_by_example,
            target_tool_by_example,
            predicted_tool_by_outcome,
            &BTreeMap::new(),
            elapsed_secs,
        )
    }

    /// Most-comprehensive aggregator: takes per-outcome schema-violation
    /// counts in addition to the basic per-tool maps. Keyed the same way
    /// as `predicted_tool_by_outcome`.
    pub fn compute_with_tools_full(
        aggregated_outcomes: &[AggregatedExampleOutcome],
        outcomes: &[ExampleOutcome],
        weights: &BTreeMap<String, f32>,
        tags_by_example: &BTreeMap<String, Vec<String>>,
        scorer_kind_by_example: &BTreeMap<String, &'static str>,
        target_tool_by_example: &BTreeMap<String, String>,
        predicted_tool_by_outcome: &BTreeMap<(String, usize), String>,
        schema_violations_by_outcome: &BTreeMap<(String, usize), (u32, u32)>,
        elapsed_secs: f64,
    ) -> Self {
        let mut num_pass = 0u32;
        let mut num_fail = 0u32;
        let mut num_invalid = 0u32;
        let mut num_error = 0u32;
        let mut sum_score = 0.0f32;
        let mut sum_weighted = 0.0f32;
        let mut sum_weights = 0.0f32;
        let mut latencies = Vec::with_capacity(outcomes.len());
        let mut prompt_tokens: u64 = 0;
        let mut completion_tokens: u64 = 0;

        // pass-rate by tag
        let mut tag_pass: BTreeMap<String, (u32, u32)> = BTreeMap::new();
        // by scorer kind
        let mut scorer_acc: BTreeMap<&'static str, (u32, f32, u32)> = BTreeMap::new();
        // per-tool pass-rate
        let mut tool_pass: BTreeMap<String, (u32, u32)> = BTreeMap::new();
        // confusion matrix: target_tool → predicted_tool → count
        let mut confusion: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
        // reasoning-length samples
        let mut reasoning_lens: Vec<u32> = Vec::new();
        let mut num_unclosed_thinking = 0u32;
        // qwen3-format diagnostic — extracted from the scorer's detail.
        let mut num_non_xml_tool_calls = 0u32;
        // schema-violation tallies sourced from the per-outcome map.
        let mut num_schema_missing_required = 0u32;
        let mut num_schema_extra_unknown = 0u32;

        for out in aggregated_outcomes {
            match out.kind {
                EvalOutcomeKind::Pass => num_pass += 1,
                EvalOutcomeKind::Fail => num_fail += 1,
                EvalOutcomeKind::Invalid => num_invalid += 1,
                EvalOutcomeKind::Error => num_error += 1,
            }
            sum_score += out.score;
            let weight = weights.get(&out.example_id).copied().unwrap_or(1.0);
            sum_weighted += out.score * weight;
            sum_weights += weight;
            if let Some(tags) = tags_by_example.get(&out.example_id) {
                for tag in tags {
                    let entry = tag_pass.entry(tag.clone()).or_insert((0, 0));
                    entry.0 += 1;
                    if matches!(out.kind, EvalOutcomeKind::Pass) {
                        entry.1 += 1;
                    }
                }
            }
            if let Some(tool) = target_tool_by_example.get(&out.example_id) {
                let entry = tool_pass.entry(tool.clone()).or_insert((0, 0));
                entry.0 += 1;
                if matches!(out.kind, EvalOutcomeKind::Pass) {
                    entry.1 += 1;
                }
                let predicted = predicted_tool_by_outcome
                    .get(&(out.example_id.clone(), out.representative_completion_index))
                    .cloned()
                    .unwrap_or_else(|| "<none>".to_string());
                *confusion
                    .entry(tool.clone())
                    .or_default()
                    .entry(predicted)
                    .or_insert(0) += 1;
            }
            if let Some(kind) = scorer_kind_by_example.get(&out.example_id) {
                let entry = scorer_acc.entry(*kind).or_insert((0, 0.0, 0));
                entry.0 += 1;
                entry.1 += out.score;
                if matches!(out.kind, EvalOutcomeKind::Pass) {
                    entry.2 += 1;
                }
            }
        }

        // Raw completions remain the source for cost, latency, reasoning, and
        // schema diagnostics. They never contribute independent samples to
        // accuracy, slices, confidence intervals, or scorer breakdowns.
        for out in outcomes {
            if let Some(lat) = out.latency_ms {
                latencies.push(lat);
            }
            if let Some(p) = out.prompt_tokens {
                prompt_tokens = prompt_tokens.saturating_add(p as u64);
            }
            if let Some(c) = out.completion_tokens {
                completion_tokens = completion_tokens.saturating_add(c as u64);
            }
            if out.unclosed_thinking {
                num_unclosed_thinking += 1;
            }
            if let Some(text) = out.reasoning_text.as_deref() {
                reasoning_lens.push(text.chars().count() as u32);
            }
            if let Some(detail) = out.detail.as_deref()
                && detail.contains("formats=")
                && !detail.contains("formats=[xml")
                && !detail.contains("formats=[xml,")
            {
                num_non_xml_tool_calls += 1;
            }
            if let Some((missing, extra)) =
                schema_violations_by_outcome.get(&(out.example_id.clone(), out.completion_index))
            {
                num_schema_missing_required += missing;
                num_schema_extra_unknown += extra;
            }
        }

        let num_examples = aggregated_outcomes.len() as u32;
        let accuracy = if num_examples > 0 {
            num_pass as f32 / num_examples as f32
        } else {
            0.0
        };
        let mean_score = if num_examples > 0 {
            sum_score / num_examples as f32
        } else {
            0.0
        };
        let weighted_mean_score = if sum_weights > 0.0 {
            sum_weighted / sum_weights
        } else {
            0.0
        };
        let pass_rate_by_tag = tag_pass
            .iter()
            .map(|(tag, (n, p))| {
                let rate = if *n > 0 { *p as f32 / *n as f32 } else { 0.0 };
                (tag.clone(), rate)
            })
            .collect();
        let tag_breakdown = tag_pass
            .into_iter()
            .map(|(tag, (n, p))| {
                let rate = if n > 0 { p as f32 / n as f32 } else { 0.0 };
                (
                    tag,
                    TagBreakdown {
                        num_examples: n,
                        num_pass: p,
                        pass_rate: rate,
                        confidence_interval: pass_rate_confidence_interval(p, n),
                    },
                )
            })
            .collect();
        let by_scorer = scorer_acc
            .into_iter()
            .map(|(kind, (n, sum, pass))| ScorerBreakdown {
                scorer_kind: kind.to_string(),
                num_examples: n,
                mean_score: if n > 0 { sum / n as f32 } else { 0.0 },
                pass_rate: if n > 0 { pass as f32 / n as f32 } else { 0.0 },
            })
            .collect();

        let pass_rate_by_tool = tool_pass
            .into_iter()
            .map(|(tool, (n, p))| {
                let rate = if n > 0 { p as f32 / n as f32 } else { 0.0 };
                (
                    tool,
                    ToolBreakdown {
                        num_examples: n,
                        num_pass: p,
                        pass_rate: rate,
                        confidence_interval: pass_rate_confidence_interval(p, n),
                    },
                )
            })
            .collect();
        let reasoning_length = compute_reasoning_stats(&reasoning_lens);

        Self {
            num_examples,
            num_completions: outcomes.len() as u32,
            num_pass,
            num_fail,
            num_invalid,
            num_error,
            accuracy,
            accuracy_confidence_interval: pass_rate_confidence_interval(num_pass, num_examples),
            mean_score,
            weighted_mean_score,
            latency: LatencyStats::from_samples(&latencies),
            total_prompt_tokens: prompt_tokens,
            total_completion_tokens: completion_tokens,
            elapsed_secs,
            pass_rate_by_tag,
            tag_breakdown,
            by_scorer,
            pass_rate_by_tool,
            reasoning_length,
            num_unclosed_thinking,
            num_non_xml_tool_calls,
            confusion_by_tool: confusion,
            num_schema_missing_required,
            num_schema_extra_unknown,
        }
    }
}

fn compute_reasoning_stats(samples: &[u32]) -> ReasoningLengthStats {
    if samples.is_empty() {
        return ReasoningLengthStats::default();
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p = |q: f64| -> u32 {
        let idx = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
        sorted[idx]
    };
    let total: u64 = sorted.iter().map(|x| *x as u64).sum();
    ReasoningLengthStats {
        num_with_thinking: n as u32,
        mean_chars: total as f64 / n as f64,
        p50_chars: p(0.5),
        p90_chars: p(0.9),
        max_chars: *sorted.last().unwrap_or(&0),
    }
}

/// Compute the canonical 95% Wilson interval used by eval aggregation and
/// promotion gates. Keeping this public prevents policy consumers from
/// reimplementing confidence arithmetic differently.
pub fn pass_rate_confidence_interval(
    num_pass: u32,
    num_examples: u32,
) -> PassRateConfidenceInterval {
    if num_examples == 0 {
        return PassRateConfidenceInterval {
            confidence_level: 0.95,
            lower: 0.0,
            upper: 0.0,
        };
    }
    let z = 1.959_963_984_540_054_f64;
    let n = num_examples as f64;
    let phat = num_pass as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = phat + z2 / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    PassRateConfidenceInterval {
        confidence_level: 0.95,
        lower: ((center - margin) / denom).clamp(0.0, 1.0) as f32,
        upper: ((center + margin) / denom).clamp(0.0, 1.0) as f32,
    }
}

/// Progress snapshot used while a job is `Running`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalProgress {
    pub examples_completed: u32,
    pub examples_total: u32,
    pub running_accuracy: f32,
    pub running_mean_score: f32,
}

/// Full result for a single (suite, adapter) run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteResult {
    pub suite_name: String,
    /// Adapter the model was running under. `None` means base model.
    pub adapter: Option<String>,
    pub aggregation: EvalAggregation,
    pub metrics: AggregateMetrics,
    /// Raw completion records used for audit and deterministic replay.
    pub outcomes: Vec<ExampleOutcome>,
    /// Exactly one decision/statistic per example. Metrics and promotion
    /// consume this collection exclusively.
    pub aggregated_outcomes: Vec<AggregatedExampleOutcome>,
    /// ISO-8601 timestamps.
    pub started_at: String,
    pub finished_at: String,
    /// Stable hash of the suite content (header + examples) for replay
    /// auditing — different suite revisions produce different hashes.
    pub suite_hash: String,
    /// Hash of the suite, optional run-level generation override, and every
    /// example's resolved thinking-budget limits/provenance. Unlike
    /// `suite_hash`, this changes when inherited server defaults change.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub effective_generation_hash: String,
    /// Self-validating suite, generation, model/scorer, environment, and raw
    /// completion identity. Absent only on legacy or explicitly synthetic
    /// results that predate the replay contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_record: Option<crate::replay::EvalReplayRecordV1>,
}

/// Top-level eval result that may contain multiple suite runs (one per
/// adapter when compare-mode is used).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    #[serde(
        serialize_with = "serialize_result_schema_version",
        deserialize_with = "deserialize_result_schema_version"
    )]
    pub schema_version: u32,
    pub job_id: String,
    pub state: EvalJobState,
    /// Exact base-weight artifacts resident when the job was admitted. Absent
    /// only for mock/synthetic generators and legacy archived jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_weight_shard_manifest: Option<kiln_core::model_provenance::BaseWeightShardManifest>,
    /// Startup-owned process/runtime envelope captured before queue publication.
    /// Absent only for synthetic generators and legacy archived jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_provenance: Option<kiln_core::execution_provenance::ExecutionProvenanceV1>,
    /// One immutable seed materialized before the job enters the queue.
    /// Example/completion seeds are derived from it with `seed_derivation`.
    /// Both fields are absent only on legacy archived jobs.
    #[serde(
        default,
        with = "optional_u64_decimal",
        skip_serializing_if = "Option::is_none"
    )]
    pub effective_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_derivation: Option<String>,
    /// Immutable source record expected by a strict replay job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_expectation: Option<crate::replay::EvalReplayExpectationV1>,
    /// Byte-for-byte comparison result, populated when a replay job reaches a
    /// terminal state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_verdict: Option<crate::replay::EvalReplayVerdict>,
    /// One entry per adapter requested. Single-adapter runs have len == 1.
    pub runs: Vec<SuiteResult>,
    /// Live progress for the currently-running adapter (cleared once the
    /// job terminates). Useful for streaming dashboards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<EvalProgress>,
    /// Free-form error message when `state == Failed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl EvalResult {
    /// Compute the flip-diff between the first two `runs`. Returns `None`
    /// when there are fewer than two runs (single-adapter result).
    ///
    /// The diff is computed against `runs[0]` as baseline and `runs[1]`
    /// as candidate. Each entry in the returned diff is keyed by the suite's
    /// canonical per-example reduction. Useful for surfacing "training
    /// improved 12 examples and regressed 3" at a glance.
    pub fn flip_diff(&self) -> Option<FlipDiff> {
        if self.runs.len() < 2 {
            return None;
        }
        let baseline = &self.runs[0];
        let candidate = &self.runs[1];
        let mut bmap: BTreeMap<String, EvalOutcomeKind> = BTreeMap::new();
        if baseline.aggregation != candidate.aggregation {
            return None;
        }
        for o in &baseline.aggregated_outcomes {
            bmap.insert(o.example_id.clone(), o.kind);
        }
        let mut diff = FlipDiff::default();
        for o in &candidate.aggregated_outcomes {
            let prior = match bmap.get(&o.example_id) {
                Some(k) => *k,
                None => continue,
            };
            match (prior, o.kind) {
                (EvalOutcomeKind::Pass, EvalOutcomeKind::Pass) => diff.both_pass += 1,
                (EvalOutcomeKind::Pass, _) => diff.regressed.push(o.example_id.clone()),
                (_, EvalOutcomeKind::Pass) => diff.improved.push(o.example_id.clone()),
                _ => diff.both_fail += 1,
            }
        }
        diff.baseline = baseline
            .adapter
            .clone()
            .unwrap_or_else(|| "<base>".to_string());
        diff.candidate = candidate
            .adapter
            .clone()
            .unwrap_or_else(|| "<base>".to_string());
        Some(diff)
    }
}

fn deserialize_result_schema_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let version = u32::deserialize(deserializer)?;
    if version != EVAL_RESULT_SCHEMA_VERSION {
        return Err(serde::de::Error::custom(format!(
            "unsupported eval result schema_version {version}; expected {EVAL_RESULT_SCHEMA_VERSION}; legacy multi-completion results are ambiguous and must be rerun"
        )));
    }
    Ok(version)
}

fn serialize_result_schema_version<S>(version: &u32, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if *version != EVAL_RESULT_SCHEMA_VERSION {
        return Err(serde::ser::Error::custom(format!(
            "cannot serialize eval result schema_version {version}; expected {EVAL_RESULT_SCHEMA_VERSION}"
        )));
    }
    serializer.serialize_u32(*version)
}

/// Paired sign test over a [`FlipDiff`]'s discordant examples. The
/// concordant pairs (both_pass / both_fail) carry no information about
/// which adapter is better; under the null hypothesis each discordant
/// example improves or regresses with probability 1/2.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SignTest {
    pub improved: u32,
    pub regressed: u32,
    /// Two-sided exact binomial p-value. 1.0 when there are no discordant
    /// pairs (no evidence either way).
    pub p_value: f64,
}

impl SignTest {
    /// Conventional 5% significance gate used by the CLI and dashboard
    /// verdicts.
    pub fn significant(&self) -> bool {
        self.p_value < 0.05
    }
}

/// Two-sided exact binomial sign test: probability of seeing a split at
/// least as lopsided as (b, c) under p=1/2. Log-space so large suites
/// don't underflow.
pub fn sign_test(improved: u32, regressed: u32) -> SignTest {
    let n = improved + regressed;
    let p_value = if n == 0 {
        1.0
    } else {
        let k = improved.min(regressed);
        let ln2 = std::f64::consts::LN_2;
        // ln C(n, i) built incrementally; tail = sum_{i<=k} C(n,i) / 2^n.
        let mut ln_terms: Vec<f64> = Vec::with_capacity(k as usize + 1);
        let mut ln_c = 0.0_f64; // ln C(n, 0)
        for i in 0..=k {
            ln_terms.push(ln_c - f64::from(n) * ln2);
            ln_c += (f64::from(n - i)).ln() - (f64::from(i + 1)).ln();
        }
        let max = ln_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let tail: f64 = ln_terms.iter().map(|t| (t - max).exp()).sum::<f64>() * max.exp();
        (2.0 * tail).min(1.0)
    };
    SignTest {
        improved,
        regressed,
        p_value,
    }
}

/// Pass↔Fail flip diff between two adapter runs of the same suite.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlipDiff {
    pub baseline: String,
    pub candidate: String,
    /// Examples that passed under baseline but not under candidate
    /// (regressions caused by the new adapter).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regressed: Vec<String>,
    /// Examples that didn't pass under baseline but pass under candidate
    /// (improvements from the new adapter).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub improved: Vec<String>,
    /// Examples that passed under both runs.
    pub both_pass: u32,
    /// Examples that failed under both runs.
    pub both_fail: u32,
}

impl FlipDiff {
    /// Sign test over this diff's discordant examples.
    pub fn significance(&self) -> SignTest {
        sign_test(self.improved.len() as u32, self.regressed.len() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(id: &str, idx: usize, kind: EvalOutcomeKind, score: f32, lat: f64) -> ExampleOutcome {
        ExampleOutcome {
            example_id: id.into(),
            completion_index: idx,
            generation_seed: None,
            completion_text: format!("c-{id}-{idx}"),
            raw_completion_text: None,
            thinking_budget: None,
            kind,
            score,
            detail: None,
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            latency_ms: Some(lat),
            tags: Vec::new(),
            metadata: None,
            reasoning_text: None,
            unclosed_thinking: false,
        }
    }

    #[test]
    fn latency_stats_basic() {
        let lat = LatencyStats::from_samples(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        assert!((lat.p50_ms - 30.0).abs() < 0.1);
        assert!(lat.p90_ms >= 40.0);
        assert!((lat.mean_ms - 30.0).abs() < 0.001);
        assert_eq!(lat.max_ms, 50.0);
    }

    #[test]
    fn eval_seed_derivation_has_a_pinned_v1_golden() {
        assert_eq!(
            derive_eval_completion_seed(42, "example-7", 3),
            1_961_911_680_962_343_893
        );
        assert_ne!(
            derive_eval_completion_seed(42, "example-7", 3),
            derive_eval_completion_seed(42, "example-7", 4)
        );
        assert_ne!(
            derive_eval_completion_seed(42, "example-7", 3),
            derive_eval_completion_seed(43, "example-7", 3)
        );
    }

    #[test]
    fn aggregation_known_answers_cover_all_reducers() {
        let raw = vec![
            mk("x", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
            mk("x", 1, EvalOutcomeKind::Fail, 0.0, 1.0),
            mk("x", 2, EvalOutcomeKind::Pass, 0.8, 1.0),
        ];

        let mean = aggregate_example_outcomes(&raw, EvalAggregation::MeanAtK { k: 3 }).unwrap();
        assert_eq!(mean.len(), 1);
        assert!((mean[0].score - 0.6).abs() < 1e-6);
        assert_eq!(mean[0].kind, EvalOutcomeKind::Pass);
        assert_eq!(mean[0].representative_completion_index, 2);

        let pass = aggregate_example_outcomes(&raw, EvalAggregation::PassAtK { k: 3 }).unwrap();
        assert_eq!(pass[0].score, 1.0);
        assert_eq!(pass[0].kind, EvalOutcomeKind::Pass);
        assert_eq!(pass[0].representative_completion_index, 0);

        let majority =
            aggregate_example_outcomes(&raw, EvalAggregation::MajorityAtK { k: 3 }).unwrap();
        assert_eq!(majority[0].score, 1.0);
        assert_eq!(majority[0].kind, EvalOutcomeKind::Pass);
        assert_eq!(majority[0].num_pass, 2);
        assert_eq!(majority[0].num_fail, 1);
    }

    #[test]
    fn aggregation_truth_table_matches_definitions() {
        for k in 1..=9 {
            for mask in 0usize..(1usize << k) {
                let pass_count = mask.count_ones() as usize;
                let raw: Vec<_> = (0..k)
                    .map(|index| {
                        let pass = mask & (1 << index) != 0;
                        mk(
                            "x",
                            index,
                            if pass {
                                EvalOutcomeKind::Pass
                            } else {
                                EvalOutcomeKind::Fail
                            },
                            if pass { 1.0 } else { 0.0 },
                            1.0,
                        )
                    })
                    .collect();

                let mean =
                    aggregate_example_outcomes(&raw, EvalAggregation::MeanAtK { k }).unwrap();
                assert!((mean[0].score - pass_count as f32 / k as f32).abs() < 1e-6);
                assert_eq!(mean[0].kind == EvalOutcomeKind::Pass, pass_count * 2 >= k);

                let pass =
                    aggregate_example_outcomes(&raw, EvalAggregation::PassAtK { k }).unwrap();
                assert_eq!(pass[0].kind == EvalOutcomeKind::Pass, pass_count > 0);

                if k % 2 == 1 {
                    let majority =
                        aggregate_example_outcomes(&raw, EvalAggregation::MajorityAtK { k })
                            .unwrap();
                    assert_eq!(
                        majority[0].kind == EvalOutcomeKind::Pass,
                        pass_count > k / 2
                    );
                }
            }
        }
    }

    #[test]
    fn aggregation_rejects_incomplete_duplicate_and_nonfinite_groups() {
        assert!(matches!(
            aggregate_example_outcomes(&[], EvalAggregation::PassAtK { k: 0 }),
            Err(EvalAggregationError::InvalidDefinition(_))
        ));
        assert!(matches!(
            aggregate_example_outcomes(&[], EvalAggregation::MajorityAtK { k: 2 }),
            Err(EvalAggregationError::InvalidDefinition(_))
        ));

        let incomplete = vec![mk("x", 0, EvalOutcomeKind::Pass, 1.0, 1.0)];
        assert!(matches!(
            aggregate_example_outcomes(&incomplete, EvalAggregation::PassAtK { k: 2 }),
            Err(EvalAggregationError::CompletionCount { .. })
        ));

        let duplicate = vec![
            mk("x", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
            mk("x", 0, EvalOutcomeKind::Fail, 0.0, 1.0),
        ];
        assert!(matches!(
            aggregate_example_outcomes(&duplicate, EvalAggregation::PassAtK { k: 2 }),
            Err(EvalAggregationError::CompletionIndices { .. })
        ));

        let nonfinite = vec![mk("x", 0, EvalOutcomeKind::Pass, f32::NAN, 1.0)];
        assert!(matches!(
            aggregate_example_outcomes(&nonfinite, EvalAggregation::Single),
            Err(EvalAggregationError::InvalidScore { .. })
        ));
    }

    #[test]
    fn eval_result_rejects_missing_and_legacy_schema_versions() {
        let missing = serde_json::json!({
            "job_id": "legacy",
            "state": "completed",
            "runs": []
        });
        assert!(serde_json::from_value::<EvalResult>(missing).is_err());

        let legacy = serde_json::json!({
            "schema_version": 1,
            "job_id": "legacy",
            "state": "completed",
            "runs": []
        });
        let error = serde_json::from_value::<EvalResult>(legacy)
            .unwrap_err()
            .to_string();
        assert!(error.contains("legacy multi-completion results are ambiguous"));

        let invalid_for_write = EvalResult {
            schema_version: 1,
            job_id: "legacy".into(),
            state: EvalJobState::Completed,
            base_weight_shard_manifest: None,
            execution_provenance: None,
            effective_seed: None,
            seed_derivation: None,
            replay_expectation: None,
            replay_verdict: None,
            runs: Vec::new(),
            progress: None,
            error: None,
        };
        assert!(serde_json::to_value(invalid_for_write).is_err());
    }

    fn mk_run(adapter: &str, outcomes: Vec<ExampleOutcome>) -> SuiteResult {
        let aggregated_outcomes =
            aggregate_example_outcomes(&outcomes, EvalAggregation::Single).unwrap();
        SuiteResult {
            suite_name: "test".into(),
            adapter: Some(adapter.into()),
            aggregation: EvalAggregation::Single,
            metrics: AggregateMetrics::default(),
            outcomes,
            aggregated_outcomes,
            started_at: "2026-05-14T00:00:00Z".into(),
            finished_at: "2026-05-14T00:00:01Z".into(),
            suite_hash: "deadbeef".into(),
            effective_generation_hash: "feedface".into(),
            replay_record: None,
        }
    }

    #[test]
    fn outcome_serializes_raw_text_and_thinking_budget_provenance() {
        let mut outcome = mk("a", 0, EvalOutcomeKind::Pass, 1.0, 2.0);
        outcome.generation_seed = Some(73);
        outcome.raw_completion_text = Some("decoder continuation".into());
        outcome.thinking_budget = Some(EvalThinkingBudget {
            configured: true,
            applied: true,
            max_tokens: Some(12),
            max_time_ms: None,
            tokens_source: "server_default".into(),
            time_source: "example_unlimited".into(),
            outcome: Some(EvalThinkingBudgetOutcome::new(
                Some(kiln_core::sampling::ThinkingBudgetTrigger::Tokens),
                true,
                12,
                41,
            )),
        });

        let json = serde_json::to_value(&outcome).unwrap();
        assert_eq!(json["generation_seed"], "73");
        assert_eq!(json["raw_completion_text"], "decoder continuation");
        assert_eq!(json["thinking_budget"]["max_tokens"], 12);
        assert_eq!(json["thinking_budget"]["trigger"], "tokens");
        assert_eq!(json["thinking_budget"]["closed"], true);

        let legacy = serde_json::json!({
            "example_id": "legacy",
            "completion_index": 0,
            "completion_text": "answer",
            "kind": "pass",
            "score": 1.0
        });
        let decoded: ExampleOutcome = serde_json::from_value(legacy).unwrap();
        assert!(decoded.raw_completion_text.is_none());
        assert!(decoded.thinking_budget.is_none());
        assert!(decoded.generation_seed.is_none());

        for wire_seed in [
            serde_json::json!(u64::MAX),
            serde_json::json!(u64::MAX.to_string()),
        ] {
            let mut value =
                serde_json::to_value(mk("seeded", 0, EvalOutcomeKind::Pass, 1.0, 1.0)).unwrap();
            value["generation_seed"] = wire_seed;
            let decoded: ExampleOutcome = serde_json::from_value(value).unwrap();
            assert_eq!(decoded.generation_seed, Some(u64::MAX));
        }
    }

    #[test]
    fn sign_test_matches_known_exact_values() {
        // b=9, c=2: n=11, k=2 -> 2 * (1+11+55)/2^11 = 134/2048.
        let t = sign_test(9, 2);
        assert!((t.p_value - 134.0 / 2048.0).abs() < 1e-12, "{}", t.p_value);
        assert!(!t.significant());
        // b=15, c=3: n=18, k=3 -> 2 * (1+18+153+816)/2^18.
        let t = sign_test(15, 3);
        assert!(
            (t.p_value - 2.0 * 988.0 / 262144.0).abs() < 1e-12,
            "{}",
            t.p_value
        );
        assert!(t.significant());
        // No discordant pairs: no evidence.
        assert_eq!(sign_test(0, 0).p_value, 1.0);
        // Symmetric in b/c.
        assert_eq!(sign_test(9, 2).p_value, sign_test(2, 9).p_value);
        // Even split: p clamps to 1.0.
        assert_eq!(sign_test(5, 5).p_value, 1.0);
        // Large n must not underflow to 0-vs-NaN.
        let t = sign_test(900, 700);
        assert!(t.p_value > 0.0 && t.p_value <= 1.0, "{}", t.p_value);
    }

    #[test]
    fn flip_diff_classifies_improvements_and_regressions() {
        let baseline = mk_run(
            "v0",
            vec![
                mk("a", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
                mk("b", 0, EvalOutcomeKind::Fail, 0.0, 1.0),
                mk("c", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
                mk("d", 0, EvalOutcomeKind::Fail, 0.0, 1.0),
            ],
        );
        let candidate = mk_run(
            "v1",
            vec![
                // a: pass→fail = regression
                mk("a", 0, EvalOutcomeKind::Fail, 0.0, 1.0),
                // b: fail→pass = improvement
                mk("b", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
                // c: pass→pass
                mk("c", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
                // d: fail→fail
                mk("d", 0, EvalOutcomeKind::Invalid, 0.0, 1.0),
            ],
        );
        let result = EvalResult {
            schema_version: EVAL_RESULT_SCHEMA_VERSION,
            job_id: "j1".into(),
            state: EvalJobState::Completed,
            base_weight_shard_manifest: None,
            execution_provenance: None,
            effective_seed: None,
            seed_derivation: None,
            replay_expectation: None,
            replay_verdict: None,
            runs: vec![baseline, candidate],
            progress: None,
            error: None,
        };
        let diff = result.flip_diff().expect("two runs → flip diff");
        assert_eq!(diff.baseline, "v0");
        assert_eq!(diff.candidate, "v1");
        assert_eq!(diff.both_pass, 1);
        assert_eq!(diff.both_fail, 1);
        assert_eq!(diff.improved, vec!["b".to_string()]);
        assert_eq!(diff.regressed, vec!["a".to_string()]);
    }

    #[test]
    fn flip_diff_returns_none_for_single_run() {
        let result = EvalResult {
            schema_version: EVAL_RESULT_SCHEMA_VERSION,
            job_id: "j2".into(),
            state: EvalJobState::Completed,
            base_weight_shard_manifest: None,
            execution_provenance: None,
            effective_seed: None,
            seed_derivation: None,
            replay_expectation: None,
            replay_verdict: None,
            runs: vec![mk_run(
                "v0",
                vec![mk("a", 0, EvalOutcomeKind::Pass, 1.0, 1.0)],
            )],
            progress: None,
            error: None,
        };
        assert!(result.flip_diff().is_none());
    }

    #[test]
    fn aggregate_basic_counts_pass_and_score() {
        let outcomes = vec![
            mk("a", 0, EvalOutcomeKind::Pass, 1.0, 10.0),
            mk("b", 0, EvalOutcomeKind::Fail, 0.0, 20.0),
            mk("c", 0, EvalOutcomeKind::Invalid, 0.0, 5.0),
        ];
        let mut weights = BTreeMap::new();
        weights.insert("a".into(), 2.0);
        weights.insert("b".into(), 1.0);
        weights.insert("c".into(), 1.0);
        let aggregated = aggregate_example_outcomes(&outcomes, EvalAggregation::Single).unwrap();
        let metrics = AggregateMetrics::compute(
            &aggregated,
            &outcomes,
            &weights,
            &BTreeMap::new(),
            &BTreeMap::new(),
            1.5,
        );
        assert_eq!(metrics.num_examples, 3);
        assert_eq!(metrics.num_completions, 3);
        assert_eq!(metrics.num_pass, 1);
        assert_eq!(metrics.num_fail, 1);
        assert_eq!(metrics.num_invalid, 1);
        assert!((metrics.accuracy - 1.0 / 3.0).abs() < 1e-6);
        assert!(metrics.accuracy_confidence_interval.lower < metrics.accuracy);
        assert!(metrics.accuracy_confidence_interval.upper > metrics.accuracy);
        assert_eq!(metrics.accuracy_confidence_interval.confidence_level, 0.95);
        // weighted: 2*1 + 1*0 + 1*0 / 4 = 0.5
        assert!((metrics.weighted_mean_score - 0.5).abs() < 1e-6);
        assert_eq!(metrics.total_completion_tokens, 60);
    }

    #[test]
    fn aggregate_tag_passrates_use_declared_example_reduction() {
        let outcomes = vec![
            mk("a", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
            mk("a", 1, EvalOutcomeKind::Fail, 0.0, 1.0),
            mk("b", 0, EvalOutcomeKind::Fail, 0.0, 1.0),
            mk("b", 1, EvalOutcomeKind::Fail, 0.0, 1.0),
        ];
        let mut tags = BTreeMap::new();
        tags.insert("a".to_string(), vec!["easy".to_string()]);
        tags.insert("b".to_string(), vec!["easy".to_string()]);
        let aggregated =
            aggregate_example_outcomes(&outcomes, EvalAggregation::PassAtK { k: 2 }).unwrap();
        let m = AggregateMetrics::compute(
            &aggregated,
            &outcomes,
            &BTreeMap::new(),
            &tags,
            &BTreeMap::new(),
            1.0,
        );
        assert_eq!(m.pass_rate_by_tag.get("easy").copied(), Some(0.5));
        let easy = m.tag_breakdown.get("easy").unwrap();
        assert_eq!(easy.num_examples, 2);
        assert_eq!(easy.num_pass, 1);
        assert!((easy.pass_rate - 0.5).abs() < 1e-6);
        assert!(easy.confidence_interval.lower < 0.5);
        assert!(easy.confidence_interval.upper > 0.5);
    }

    #[test]
    fn aggregate_tool_breakdown_has_confidence_interval() {
        let outcomes = vec![
            mk("a", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
            mk("b", 0, EvalOutcomeKind::Fail, 0.0, 1.0),
        ];
        let mut target_tools = BTreeMap::new();
        target_tools.insert("a".to_string(), "Bash".to_string());
        target_tools.insert("b".to_string(), "Bash".to_string());
        let aggregated = aggregate_example_outcomes(&outcomes, EvalAggregation::Single).unwrap();
        let m = AggregateMetrics::compute_with_tools(
            &aggregated,
            &outcomes,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &target_tools,
            &BTreeMap::new(),
            1.0,
        );
        let bash = m.pass_rate_by_tool.get("Bash").unwrap();
        assert_eq!(bash.num_examples, 2);
        assert_eq!(bash.num_pass, 1);
        assert!((bash.pass_rate - 0.5).abs() < 1e-6);
        assert!(bash.confidence_interval.lower < 0.5);
        assert!(bash.confidence_interval.upper > 0.5);
    }

    #[test]
    fn aggregate_serializes_confidence_interval_fields() {
        let outcomes = vec![
            mk("a", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
            mk("b", 0, EvalOutcomeKind::Fail, 0.0, 1.0),
        ];
        let aggregated = aggregate_example_outcomes(&outcomes, EvalAggregation::Single).unwrap();
        let m = AggregateMetrics::compute(
            &aggregated,
            &outcomes,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            1.0,
        );
        let json = serde_json::to_value(&m).unwrap();
        assert!(
            (json["accuracy_confidence_interval"]["confidence_level"]
                .as_f64()
                .unwrap()
                - 0.95)
                .abs()
                < 1e-6
        );
        assert!(
            json["accuracy_confidence_interval"]["lower"]
                .as_f64()
                .unwrap()
                < 0.5
        );
        assert!(
            json["accuracy_confidence_interval"]["upper"]
                .as_f64()
                .unwrap()
                > 0.5
        );
    }
}
