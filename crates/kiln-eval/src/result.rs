//! Result types: per-example outcomes, suite-level aggregates, and the
//! `EvalJobState` lifecycle exposed via the HTTP API.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
    pub completion_text: String,
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
    pub num_examples: u32,
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
        outcomes: &[ExampleOutcome],
        weights: &BTreeMap<String, f32>,
        tags_by_example: &BTreeMap<String, Vec<String>>,
        scorer_kind_by_example: &BTreeMap<String, &'static str>,
        elapsed_secs: f64,
    ) -> Self {
        Self::compute_with_tools(
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
        outcomes: &[ExampleOutcome],
        weights: &BTreeMap<String, f32>,
        tags_by_example: &BTreeMap<String, Vec<String>>,
        scorer_kind_by_example: &BTreeMap<String, &'static str>,
        target_tool_by_example: &BTreeMap<String, String>,
        predicted_tool_by_outcome: &BTreeMap<(String, usize), String>,
        elapsed_secs: f64,
    ) -> Self {
        Self::compute_with_tools_full(
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

        for out in outcomes {
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
            if let Some(lat) = out.latency_ms {
                latencies.push(lat);
            }
            if let Some(p) = out.prompt_tokens {
                prompt_tokens = prompt_tokens.saturating_add(p as u64);
            }
            if let Some(c) = out.completion_tokens {
                completion_tokens = completion_tokens.saturating_add(c as u64);
            }
            // tag pass-rate uses the FIRST completion per example to avoid
            // double-counting under n>1. We mark the first occurrence by
            // tracking example_id seen.
            if out.completion_index == 0 {
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
                        .get(&(out.example_id.clone(), out.completion_index))
                        .cloned()
                        .unwrap_or_else(|| "<none>".to_string());
                    *confusion
                        .entry(tool.clone())
                        .or_default()
                        .entry(predicted)
                        .or_insert(0) += 1;
                }
            }
            if out.unclosed_thinking {
                num_unclosed_thinking += 1;
            }
            if let Some(text) = out.reasoning_text.as_deref() {
                reasoning_lens.push(text.chars().count() as u32);
            }
            // Scorer detail of the form "... formats=[json,...]" marks a
            // non-Qwen3.5-XML emission. Cheap textual check — the canonical
            // alternative is to plumb a typed format flag through the
            // outcome but the textual check is good enough for an
            // aggregate stat.
            if let Some(detail) = out.detail.as_deref() {
                if detail.contains("formats=")
                    && !detail.contains("formats=[xml")
                    && !detail.contains("formats=[xml,")
                {
                    num_non_xml_tool_calls += 1;
                }
            }
            if let Some((missing, extra)) =
                schema_violations_by_outcome.get(&(out.example_id.clone(), out.completion_index))
            {
                num_schema_missing_required += missing;
                num_schema_extra_unknown += extra;
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

        let num_examples = outcomes.len() as u32;
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

fn pass_rate_confidence_interval(num_pass: u32, num_examples: u32) -> PassRateConfidenceInterval {
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
    pub metrics: AggregateMetrics,
    pub outcomes: Vec<ExampleOutcome>,
    /// ISO-8601 timestamps.
    pub started_at: String,
    pub finished_at: String,
    /// Stable hash of the suite content (header + examples) for replay
    /// auditing — different suite revisions produce different hashes.
    pub suite_hash: String,
}

/// Top-level eval result that may contain multiple suite runs (one per
/// adapter when compare-mode is used).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub job_id: String,
    pub state: EvalJobState,
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
    /// as candidate. Each entry in the returned diff is keyed by
    /// `example_id` (using the *first* completion only — pass@k metrics
    /// pre-aggregate before this point). Useful for surfacing "training
    /// improved 12 examples and regressed 3" at a glance.
    pub fn flip_diff(&self) -> Option<FlipDiff> {
        if self.runs.len() < 2 {
            return None;
        }
        let baseline = &self.runs[0];
        let candidate = &self.runs[1];
        let mut bmap: BTreeMap<String, EvalOutcomeKind> = BTreeMap::new();
        for o in &baseline.outcomes {
            if o.completion_index == 0 {
                bmap.insert(o.example_id.clone(), o.kind);
            }
        }
        let mut diff = FlipDiff::default();
        for o in &candidate.outcomes {
            if o.completion_index != 0 {
                continue;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(id: &str, idx: usize, kind: EvalOutcomeKind, score: f32, lat: f64) -> ExampleOutcome {
        ExampleOutcome {
            example_id: id.into(),
            completion_index: idx,
            completion_text: format!("c-{id}-{idx}"),
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

    fn mk_run(adapter: &str, outcomes: Vec<ExampleOutcome>) -> SuiteResult {
        SuiteResult {
            suite_name: "test".into(),
            adapter: Some(adapter.into()),
            metrics: AggregateMetrics::default(),
            outcomes,
            started_at: "2026-05-14T00:00:00Z".into(),
            finished_at: "2026-05-14T00:00:01Z".into(),
            suite_hash: "deadbeef".into(),
        }
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
            job_id: "j1".into(),
            state: EvalJobState::Completed,
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
            job_id: "j2".into(),
            state: EvalJobState::Completed,
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
        let metrics =
            AggregateMetrics::compute(&outcomes, &weights, &BTreeMap::new(), &BTreeMap::new(), 1.5);
        assert_eq!(metrics.num_examples, 3);
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
    fn aggregate_tag_passrates_use_first_completion() {
        let outcomes = vec![
            mk("a", 0, EvalOutcomeKind::Pass, 1.0, 1.0),
            mk("a", 1, EvalOutcomeKind::Fail, 0.0, 1.0),
            mk("b", 0, EvalOutcomeKind::Fail, 0.0, 1.0),
        ];
        let mut tags = BTreeMap::new();
        tags.insert("a".to_string(), vec!["easy".to_string()]);
        tags.insert("b".to_string(), vec!["easy".to_string()]);
        let m =
            AggregateMetrics::compute(&outcomes, &BTreeMap::new(), &tags, &BTreeMap::new(), 1.0);
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
        let m = AggregateMetrics::compute_with_tools(
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
        let m = AggregateMetrics::compute(
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
