//! Background eval worker — pulls jobs from `state.eval_queue` and runs
//! them through the executor.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use kiln_eval::scorers::JudgeRunner;
use kiln_eval::{EvalJobState, EvalProgress};

use crate::eval::executor::{
    EvalReplayEnvironment, noop_judge_runner, run_suite_against_adapter_with_replay,
};
use crate::eval::generator::generator_from_state;
use crate::eval::queue::{EvalQueueEntry, QueuedEvalJob};
use crate::state::AppState;
use crate::training_queue::ShutdownFlag;

/// Spawn the eval worker. Mirrors `spawn_training_worker` — polls the
/// queue every 250ms, runs jobs serially, exits cleanly on shutdown.
pub fn spawn_eval_worker(state: AppState, shutdown: ShutdownFlag) {
    tokio::spawn(async move {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                tracing::info!("eval worker shutting down");
                break;
            }
            gc_eval_jobs(&state);
            let entry = {
                let mut q = state.eval_queue.lock().unwrap();
                q.pop()
            };
            if let Some(entry) = entry {
                run_one_job(state.clone(), entry).await;
            } else {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
        }
    });
}

/// Evict terminal (`Completed` / `Failed` / `Cancelled`) entries from
/// `state.eval_jobs` past TTL only when the map exceeds
/// `max_tracked_eval_jobs`. Oldest-by-finish-time first. Mirrors
/// `training_queue::gc_tracked_jobs` (cap-driven, not TTL-driven).
pub fn gc_eval_jobs(state: &AppState) -> usize {
    let cap = state.max_tracked_eval_jobs;
    let ttl = state.tracked_job_ttl;
    let now = std::time::Instant::now();
    let mut jobs = state.eval_jobs.write().unwrap();
    if jobs.len() <= cap {
        return 0;
    }
    let mut candidates: Vec<(String, std::time::Instant)> =
        jobs.iter()
            .filter_map(|(id, j)| match (j.state, j.finished_at) {
                (
                    EvalJobState::Completed | EvalJobState::Failed | EvalJobState::Cancelled,
                    Some(t),
                ) if now.saturating_duration_since(t) >= ttl => Some((id.clone(), t)),
                _ => None,
            })
            .collect();
    candidates.sort_by_key(|(_, t)| *t);
    let want_to_remove = jobs.len().saturating_sub(cap);
    let mut removed = 0;
    for (id, _) in candidates.into_iter().take(want_to_remove) {
        jobs.remove(&id);
        removed += 1;
    }
    if removed > 0 {
        tracing::debug!(
            removed,
            remaining = jobs.len(),
            cap,
            "evicted oldest terminal eval jobs past TTL to honor max_tracked_eval_jobs cap"
        );
    }
    removed
}

fn evaluate_replay_verdict(
    expectation: &kiln_eval::EvalReplayExpectationV1,
    runs: &[kiln_eval::SuiteResult],
) -> kiln_eval::EvalReplayVerdict {
    let error = |message: String| kiln_eval::EvalReplayVerdict {
        status: kiln_eval::EvalReplayStatus::Error,
        source_job_id: expectation.source_job_id.clone(),
        source_run_index: expectation.source_run_index,
        expected_record_sha256: expectation.expected_record_sha256.clone(),
        actual_record_sha256: None,
        expected_raw_completion_set_sha256: expectation.expected_raw_completion_set_sha256.clone(),
        actual_raw_completion_set_sha256: None,
        message,
    };
    if let Err(error_message) = expectation.validate() {
        return error(format!("invalid replay expectation: {error_message}"));
    }
    let [run] = runs else {
        return error(format!(
            "strict replay expected exactly one run, observed {}",
            runs.len()
        ));
    };
    let Some(record) = run.replay_record.as_ref() else {
        return error("replay run did not produce an identity record".to_string());
    };
    if let Err(error_message) = record.validate_strict_replay(&run.outcomes) {
        return error(format!(
            "replay run identity is incomplete: {error_message}"
        ));
    }
    let matched = record.record_sha256 == expectation.expected_record_sha256
        && record.raw_completion_set_sha256 == expectation.expected_raw_completion_set_sha256;
    kiln_eval::EvalReplayVerdict {
        status: if matched {
            kiln_eval::EvalReplayStatus::Matched
        } else {
            kiln_eval::EvalReplayStatus::Mismatch
        },
        source_job_id: expectation.source_job_id.clone(),
        source_run_index: expectation.source_run_index,
        expected_record_sha256: expectation.expected_record_sha256.clone(),
        actual_record_sha256: Some(record.record_sha256.clone()),
        expected_raw_completion_set_sha256: expectation.expected_raw_completion_set_sha256.clone(),
        actual_raw_completion_set_sha256: Some(record.raw_completion_set_sha256.clone()),
        message: if matched {
            "all replay identities and raw decoder completions matched byte-for-byte".to_string()
        } else {
            "replay completed but at least one identity or raw decoder completion differed"
                .to_string()
        },
    }
}

fn failed_replay_verdict(
    expectation: &kiln_eval::EvalReplayExpectationV1,
    message: String,
) -> kiln_eval::EvalReplayVerdict {
    kiln_eval::EvalReplayVerdict {
        status: kiln_eval::EvalReplayStatus::Error,
        source_job_id: expectation.source_job_id.clone(),
        source_run_index: expectation.source_run_index,
        expected_record_sha256: expectation.expected_record_sha256.clone(),
        actual_record_sha256: None,
        expected_raw_completion_set_sha256: expectation.expected_raw_completion_set_sha256.clone(),
        actual_raw_completion_set_sha256: None,
        message,
    }
}

async fn run_one_job(state: AppState, entry: EvalQueueEntry) {
    let generator = generator_from_state(state.clone());
    run_one_job_with_generator(state, entry, generator).await
}

/// `run_one_job` body with the generator injected — the seam the
/// cancellation test uses to gate generation deterministically.
async fn run_one_job_with_generator(
    state: AppState,
    entry: EvalQueueEntry,
    generator: Arc<dyn crate::eval::generator::EvalGenerator>,
) {
    let job_id = entry.job_id.clone();
    let inference_admission = state
        .ensure_inference_admission_allowed()
        .map_err(|error| format!("eval rejected before execution: {error:#}"));
    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Mark as running, stamp started_at, and install the cancellation flag
    // so DELETE /v1/eval/jobs/{id} can actually stop the executor.
    {
        let mut jobs = state.eval_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id) {
            if matches!(job.state, EvalJobState::Cancelled | EvalJobState::Failed) {
                tracing::info!(job_id = %job_id, "skipping cancelled eval job");
                return;
            }
            job.state = EvalJobState::Running;
            job.started_at_iso = Some(chrono::Utc::now().to_rfc3339());
            job.cancel_flag = Some(cancel_flag.clone());
        } else {
            tracing::warn!(job_id = %job_id, "eval job not found in tracking map");
            return;
        }
    }

    // Live judge on the real backend: LlmJudge scorers re-enter the model
    // through the executor's deferred judge pass (batch adapter swap +
    // blocking-thread scoring). Mock mode keeps the no-op runner — judge
    // scorers degrade to Invalid there, honestly.
    let judge_runner: Arc<dyn JudgeRunner> = match state.backend.as_ref() {
        crate::state::ModelBackend::Real { .. } => {
            Arc::new(crate::eval::generator::LiveJudgeRunner::new(state.clone()))
        }
        crate::state::ModelBackend::Mock { .. } => noop_judge_runner(),
    };

    let progress_state = state.eval_jobs.clone();
    let progress_job_id = job_id.clone();
    let progress_cb: crate::eval::executor::ProgressCallback = Box::new(move |p: EvalProgress| {
        let mut jobs = progress_state.write().unwrap();
        if let Some(job) = jobs.get_mut(&progress_job_id) {
            job.progress = p;
        }
    });

    let result = match inference_admission {
        Ok(()) => {
            run_job(
                &state,
                &entry.job,
                entry.effective_seed,
                entry.replay_source_record.clone(),
                generator,
                judge_runner,
                Some(progress_cb),
                cancel_flag.clone(),
            )
            .await
        }
        Err(error) => Err(error),
    };

    let now_iso = chrono::Utc::now().to_rfc3339();
    let now_instant = std::time::Instant::now();

    let was_cancelled = cancel_flag.load(std::sync::atomic::Ordering::Relaxed);
    let archive_snapshot = match result {
        Ok(runs) => {
            let headline = runs.iter().last().map(|r| r.metrics.accuracy);
            let mut jobs = state.eval_jobs.write().unwrap();
            jobs.get_mut(&job_id).map(|job| {
                // A cancelled run still archives its partial outcomes, but
                // the terminal state must stay Cancelled — not flip back to
                // Completed and erase what the user asked for.
                if was_cancelled || matches!(job.state, EvalJobState::Cancelled) {
                    job.state = EvalJobState::Cancelled;
                } else {
                    job.state = EvalJobState::Completed;
                }
                job.finished_runs = runs;
                job.replay_verdict = job
                    .replay_expectation
                    .as_ref()
                    .map(|expectation| evaluate_replay_verdict(expectation, &job.finished_runs));
                job.headline_accuracy = headline;
                job.finished_at_iso = Some(now_iso);
                job.finished_at = Some(now_instant);
                job.cancel_flag = None;
                job.clone()
            })
        }
        Err(err) => {
            tracing::error!(job_id = %job_id, error = %err, "eval job failed");
            let mut jobs = state.eval_jobs.write().unwrap();
            jobs.get_mut(&job_id).map(|job| {
                job.state = EvalJobState::Failed;
                job.error = Some(err);
                job.replay_verdict = job.replay_expectation.as_ref().map(|expectation| {
                    failed_replay_verdict(
                        expectation,
                        job.error
                            .clone()
                            .unwrap_or_else(|| "replay execution failed".to_string()),
                    )
                });
                job.finished_at_iso = Some(now_iso);
                job.finished_at = Some(now_instant);
                job.cancel_flag = None;
                job.clone()
            })
        }
    };
    if let Some(snapshot) = archive_snapshot {
        if let Err(e) = crate::eval_history::save(&state.adapter_dir, &snapshot) {
            tracing::warn!(error = %e, job_id = %job_id, "failed to archive terminal eval job");
        }
        crate::eval_history::prune_to_max(
            &state.adapter_dir,
            crate::eval_history::MAX_ARCHIVED_JOBS,
        );
        // §8.7: this job may carry a promotion gate (post-training eval
        // with min_accuracy). Apply the verdict now that the run is
        // terminal.
        apply_post_eval_gate(&state, &snapshot).await;

        // Eval results become signals: fire the completion webhook AFTER
        // the gate so the payload carries the promotion verdict. The gate
        // stamps its verdict on the LINKED TRAINING job; re-read it here.
        if let Some(ref url) = state.eval_webhook_url {
            let gate_verdict = snapshot.post_eval_gate.as_ref().and_then(|gate| {
                let jobs = state.training_jobs.read().unwrap();
                jobs.get(&gate.training_job_id)
                    .and_then(|j| j.post_eval_verdict.clone())
            });
            let event = serde_json::json!({
                "event": "eval_completed",
                "job_id": snapshot.job_id,
                "suite": snapshot.suite_name,
                "adapters": snapshot.adapters,
                "status": match snapshot.state {
                    EvalJobState::Completed => "completed",
                    EvalJobState::Cancelled => "cancelled",
                    _ => "failed",
                },
                "headline_accuracy": snapshot.headline_accuracy,
                "gate_verdict": gate_verdict,
                "error": snapshot.error,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            });
            crate::training_queue::fire_webhook_json(url.clone(), event);
        }
    }

    // Back-linking from eval-completion → training-job is intentionally
    // NOT done here. `enqueue_post_training_eval` already populates the
    // training job's `linked_eval_job_ids` at queue time, so the
    // training-side dashboard can link to the eval the moment it lands
    // in the queue (not after it finishes). Pushing again here would
    // duplicate every ID.
}

/// Fixed, versioned policy for automatic post-eval promotion. The minimum is
/// deliberately not a request knob: allowing each caller to weaken the
/// evidence threshold would turn a safety gate back into a point-estimate
/// footgun. Wilson bounds still force larger suites when the requested
/// accuracy floor is high (for example, 20/20 is not enough to prove 0.90).
const POST_EVAL_PROMOTION_POLICY_VERSION: &str = "paired_wilson_v1";
const POST_EVAL_SUITE_POLICY: &str = "single_versioned_suite";
const POST_EVAL_MIN_PAIRED_EXAMPLES: u32 = 20;
const POST_EVAL_EXACT_TEST_ALPHA: f64 = 0.05;

fn gate_outcome_priority(outcome: &str) -> u8 {
    match outcome {
        "error" => 6,
        "demoted" | "regression" => 5,
        "inconclusive" => 4,
        "promoted" => 3,
        "kept" => 2,
        // Preserve an unknown archived value rather than letting a newer
        // success silently overwrite a classification this binary does not
        // understand.
        _ => 7,
    }
}

fn aggregate_gate_outcome(existing: Option<&str>, new_outcome: &str) -> String {
    match existing {
        Some(existing) if gate_outcome_priority(existing) >= gate_outcome_priority(new_outcome) => {
            existing.to_string()
        }
        _ => new_outcome.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatisticalGateDecision {
    Pass,
    Regression,
    Demoted,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PairedAggregateSummary {
    paired_examples: u32,
    improved: u32,
    regressed: u32,
    tied: u32,
    baseline_pass: u32,
    candidate_pass: u32,
}

fn paired_aggregate_flips(
    baseline: &kiln_eval::SuiteResult,
    candidate: &kiln_eval::SuiteResult,
) -> Result<PairedAggregateSummary, String> {
    if baseline.suite_name != candidate.suite_name {
        return Err(format!(
            "suite-name mismatch: baseline {:?} vs candidate {:?}",
            baseline.suite_name, candidate.suite_name
        ));
    }
    if baseline.suite_hash != candidate.suite_hash {
        return Err(format!(
            "suite-hash mismatch: baseline {} vs candidate {}",
            baseline.suite_hash, candidate.suite_hash
        ));
    }
    if baseline.effective_generation_hash != candidate.effective_generation_hash {
        return Err(format!(
            "generation-hash mismatch: baseline {} vs candidate {}",
            baseline.effective_generation_hash, candidate.effective_generation_hash
        ));
    }
    if baseline.aggregation != candidate.aggregation {
        return Err(format!(
            "aggregation mismatch: baseline {} vs candidate {}",
            baseline.aggregation.label(),
            candidate.aggregation.label()
        ));
    }
    if baseline.aggregated_outcomes.len() != candidate.aggregated_outcomes.len() {
        return Err(format!(
            "example-count mismatch: baseline {} vs candidate {}",
            baseline.aggregated_outcomes.len(),
            candidate.aggregated_outcomes.len()
        ));
    }
    if baseline.aggregated_outcomes.is_empty() {
        return Err("paired suite produced zero independent examples".to_string());
    }
    let mut candidate_by_id = std::collections::BTreeMap::new();
    for outcome in &candidate.aggregated_outcomes {
        if !outcome.score.is_finite() {
            return Err(format!(
                "candidate example {:?} has non-finite score",
                outcome.example_id
            ));
        }
        if candidate_by_id
            .insert(outcome.example_id.as_str(), outcome)
            .is_some()
        {
            return Err(format!(
                "candidate contains duplicate example {:?}",
                outcome.example_id
            ));
        }
    }
    let mut baseline_ids = std::collections::BTreeSet::new();
    let mut improved = 0u32;
    let mut regressed = 0u32;
    let mut baseline_pass = 0u32;
    let mut candidate_pass = 0u32;
    for baseline_outcome in &baseline.aggregated_outcomes {
        if !baseline_outcome.score.is_finite() {
            return Err(format!(
                "baseline example {:?} has non-finite score",
                baseline_outcome.example_id
            ));
        }
        if !baseline_ids.insert(baseline_outcome.example_id.as_str()) {
            return Err(format!(
                "baseline contains duplicate example {:?}",
                baseline_outcome.example_id
            ));
        }
        let Some(candidate_outcome) = candidate_by_id.get(baseline_outcome.example_id.as_str())
        else {
            return Err(format!(
                "candidate is missing example {:?}",
                baseline_outcome.example_id
            ));
        };
        if candidate_outcome.score > baseline_outcome.score {
            improved += 1;
        } else if candidate_outcome.score < baseline_outcome.score {
            regressed += 1;
        }
        if baseline_outcome.kind == kiln_eval::EvalOutcomeKind::Pass {
            baseline_pass += 1;
        }
        if candidate_outcome.kind == kiln_eval::EvalOutcomeKind::Pass {
            candidate_pass += 1;
        }
    }
    let paired_examples = u32::try_from(baseline.aggregated_outcomes.len())
        .map_err(|_| "paired example count exceeds u32".to_string())?;
    Ok(PairedAggregateSummary {
        paired_examples,
        improved,
        regressed,
        tied: paired_examples - improved - regressed,
        baseline_pass,
        candidate_pass,
    })
}

struct StatisticalGateResult {
    decision: StatisticalGateDecision,
    evidence: kiln_train::PostEvalGateEvidence,
    reason: String,
}

fn evaluate_paired_promotion(
    eval_job_id: &str,
    gate: &crate::eval::queue::PostEvalGate,
    baseline: &kiln_eval::SuiteResult,
    candidate: &kiln_eval::SuiteResult,
) -> Result<StatisticalGateResult, String> {
    let paired = paired_aggregate_flips(baseline, candidate)?;
    let sign_test = kiln_eval::result::sign_test(paired.improved, paired.regressed);
    let baseline_ci = kiln_eval::result::pass_rate_confidence_interval(
        paired.baseline_pass,
        paired.paired_examples,
    );
    let candidate_ci = kiln_eval::result::pass_rate_confidence_interval(
        paired.candidate_pass,
        paired.paired_examples,
    );
    let baseline_accuracy = paired.baseline_pass as f32 / paired.paired_examples as f32;
    let candidate_accuracy = paired.candidate_pass as f32 / paired.paired_examples as f32;
    let relative_recovery_lower_bound = gate.relative_recovery.map(|_| {
        if baseline_ci.upper > 0.0 {
            candidate_ci.lower / baseline_ci.upper
        } else {
            1.0
        }
    });
    let absolute_gain_lower_bound = gate
        .absolute_gain
        .map(|_| candidate_ci.lower - baseline_ci.upper);

    let (decision, reason) = if paired.paired_examples < POST_EVAL_MIN_PAIRED_EXAMPLES {
        (
            StatisticalGateDecision::Inconclusive,
            format!(
                "only {} paired examples; policy requires at least {}",
                paired.paired_examples, POST_EVAL_MIN_PAIRED_EXAMPLES
            ),
        )
    } else if paired.regressed > paired.improved && sign_test.p_value < POST_EVAL_EXACT_TEST_ALPHA {
        (
            StatisticalGateDecision::Regression,
            format!(
                "candidate is significantly worse (improved {}, regressed {}, exact two-sided p={:.6})",
                paired.improved, paired.regressed, sign_test.p_value
            ),
        )
    } else if candidate_ci.upper < gate.min_accuracy {
        (
            StatisticalGateDecision::Demoted,
            format!(
                "candidate 95% Wilson upper bound {:.3} is below accuracy floor {:.3}",
                candidate_ci.upper, gate.min_accuracy
            ),
        )
    } else if candidate_ci.lower < gate.min_accuracy {
        (
            StatisticalGateDecision::Inconclusive,
            format!(
                "candidate 95% Wilson lower bound {:.3} does not reach accuracy floor {:.3}",
                candidate_ci.lower, gate.min_accuracy
            ),
        )
    } else if let (Some(required), Some(observed)) =
        (gate.relative_recovery, relative_recovery_lower_bound)
    {
        if observed < required {
            (
                StatisticalGateDecision::Inconclusive,
                format!("relative-recovery lower bound {observed:.3} does not reach {required:.3}"),
            )
        } else if let (Some(required_gain), Some(observed_gain)) =
            (gate.absolute_gain, absolute_gain_lower_bound)
        {
            if observed_gain < required_gain {
                (
                    StatisticalGateDecision::Inconclusive,
                    format!(
                        "absolute-gain lower bound {observed_gain:+.3} does not reach {required_gain:+.3}"
                    ),
                )
            } else {
                (
                    StatisticalGateDecision::Pass,
                    "all paired confidence requirements passed".to_string(),
                )
            }
        } else {
            (
                StatisticalGateDecision::Pass,
                "all paired confidence requirements passed".to_string(),
            )
        }
    } else if let (Some(required_gain), Some(observed_gain)) =
        (gate.absolute_gain, absolute_gain_lower_bound)
    {
        if observed_gain < required_gain {
            (
                StatisticalGateDecision::Inconclusive,
                format!(
                    "absolute-gain lower bound {observed_gain:+.3} does not reach {required_gain:+.3}"
                ),
            )
        } else {
            (
                StatisticalGateDecision::Pass,
                "all paired confidence requirements passed".to_string(),
            )
        }
    } else if paired.improved > paired.regressed && sign_test.p_value < POST_EVAL_EXACT_TEST_ALPHA {
        (
            StatisticalGateDecision::Pass,
            "exact paired improvement and accuracy confidence floor passed".to_string(),
        )
    } else {
        (
            StatisticalGateDecision::Inconclusive,
            format!(
                "no significant paired improvement (improved {}, regressed {}, exact two-sided p={:.6})",
                paired.improved, paired.regressed, sign_test.p_value
            ),
        )
    };

    Ok(StatisticalGateResult {
        decision,
        evidence: kiln_train::PostEvalGateEvidence {
            policy_version: POST_EVAL_PROMOTION_POLICY_VERSION.to_string(),
            suite_policy: POST_EVAL_SUITE_POLICY.to_string(),
            eval_job_id: eval_job_id.to_string(),
            suite_name: candidate.suite_name.clone(),
            suite_hash: candidate.suite_hash.clone(),
            effective_generation_hash: candidate.effective_generation_hash.clone(),
            baseline_adapter: baseline.adapter.clone(),
            candidate_adapter: gate.adapter_name.clone(),
            aggregation: candidate.aggregation.label().to_string(),
            minimum_paired_examples: POST_EVAL_MIN_PAIRED_EXAMPLES,
            paired_examples: paired.paired_examples,
            improved: paired.improved,
            regressed: paired.regressed,
            tied: paired.tied,
            exact_sign_test_p_value: sign_test.p_value,
            exact_sign_test_alpha: POST_EVAL_EXACT_TEST_ALPHA,
            baseline_accuracy,
            baseline_accuracy_lower_bound: baseline_ci.lower,
            baseline_accuracy_upper_bound: baseline_ci.upper,
            candidate_accuracy,
            candidate_accuracy_lower_bound: candidate_ci.lower,
            candidate_accuracy_upper_bound: candidate_ci.upper,
            accuracy_confidence_level: candidate_ci.confidence_level,
            minimum_accuracy: gate.min_accuracy,
            required_relative_recovery: gate.relative_recovery,
            relative_recovery_lower_bound,
            required_absolute_gain: gate.absolute_gain,
            absolute_gain_lower_bound,
            outcome: String::new(),
        },
        reason,
    })
}

async fn apply_post_eval_gate(state: &AppState, snapshot: &crate::eval::queue::EvalJobInfo) {
    let Some(gate) = snapshot.post_eval_gate.clone() else {
        return;
    };
    let job_id = snapshot.job_id.clone();

    let stamp_verdict =
        |outcome: crate::state::GateOutcome,
         verdict: String,
         mut evidence: Option<kiln_train::PostEvalGateEvidence>| {
            tracing::info!(
                eval_job = %job_id,
                training_job = %gate.training_job_id,
                adapter = %gate.adapter_name,
                outcome = %outcome.as_str(),
                verdict = %verdict,
                "post-eval gate verdict"
            );
            let snapshot = {
                let mut jobs = state.training_jobs.write().unwrap();
                jobs.get_mut(&gate.training_job_id).map(|job| {
                    let new_outcome = outcome.as_str();
                    let aggregate_outcome =
                        aggregate_gate_outcome(job.gate_outcome.as_deref(), new_outcome);
                    job.post_eval_verdict = Some(if aggregate_outcome == new_outcome {
                        verdict
                    } else {
                        format!(
                            "{} retained across independent gate evidence; latest result: {verdict}",
                            aggregate_outcome.to_uppercase()
                        )
                    });
                    // Machine-readable twin of the prose verdict — persisted
                    // together everywhere the verdict persists so consumers
                    // never classify prose by substring.
                    job.gate_outcome = Some(aggregate_outcome);
                    if let Some(mut evidence) = evidence.take() {
                        evidence.outcome = outcome.as_str().to_string();
                        if let Some(existing) = job
                            .post_eval_gate_evidence
                            .iter_mut()
                            .find(|prior| prior.eval_job_id == evidence.eval_job_id)
                        {
                            *existing = evidence;
                        } else {
                            job.post_eval_gate_evidence.push(evidence);
                        }
                    }
                    job.clone()
                })
            };
            // Re-archive: finalize_job persisted the terminal job BEFORE the
            // gate eval ran, so without this re-save the verdict only lived
            // in memory and a restart showed the gated job verdict-less
            // (round-5 quick win).
            if let Some(job) = snapshot {
                if let Err(e) = crate::training_history::save(&state.adapter_dir, &job) {
                    tracing::warn!(
                        error = %e,
                        training_job = %gate.training_job_id,
                        "failed to persist gate verdict to training history"
                    );
                }
            }
        };

    if snapshot.state != EvalJobState::Completed {
        stamp_verdict(
            crate::state::GateOutcome::Error,
            format!(
                "post-eval did not complete (state: {:?}) — adapter `{}` left on disk, NOT promoted",
                snapshot.state, gate.adapter_name
            ),
            None,
        );
        return;
    }

    let Some(candidate_run) = snapshot
        .finished_runs
        .iter()
        .find(|run| run.adapter.as_deref() == Some(gate.adapter_name.as_str()))
    else {
        stamp_verdict(
            crate::state::GateOutcome::Error,
            format!(
                "post-eval produced no run for adapter `{}` — NOT promoted",
                gate.adapter_name
            ),
            None,
        );
        return;
    };
    let Some(baseline_run) = snapshot
        .finished_runs
        .iter()
        .find(|run| run.adapter.as_deref() != Some(gate.adapter_name.as_str()))
    else {
        stamp_verdict(
            crate::state::GateOutcome::Error,
            format!(
                "post-eval produced no paired baseline for adapter `{}` — NOT promoted",
                gate.adapter_name
            ),
            None,
        );
        return;
    };

    let statistical = match evaluate_paired_promotion(&job_id, &gate, baseline_run, candidate_run) {
        Ok(result) => result,
        Err(error) => {
            stamp_verdict(
                crate::state::GateOutcome::Error,
                format!("post-eval paired evidence is invalid: {error} — NOT promoted"),
                None,
            );
            return;
        }
    };
    let accuracy = statistical.evidence.candidate_accuracy;
    match statistical.decision {
        StatisticalGateDecision::Regression => {
            stamp_verdict(
                crate::state::GateOutcome::Regression,
                format!(
                    "REGRESSION: `{}` vs `{}`: {} — NOT promoted",
                    gate.adapter_name,
                    baseline_run.adapter.as_deref().unwrap_or("base"),
                    statistical.reason
                ),
                Some(statistical.evidence),
            );
            return;
        }
        StatisticalGateDecision::Inconclusive => {
            stamp_verdict(
                crate::state::GateOutcome::Inconclusive,
                format!(
                    "INCONCLUSIVE: `{}`: {}; adapter left on disk, NOT promoted",
                    gate.adapter_name, statistical.reason
                ),
                Some(statistical.evidence),
            );
            return;
        }
        StatisticalGateDecision::Pass => {
            if gate.auto_load_on_pass {
                let adapter_dir = state.adapter_dir.join(&gate.adapter_name);
                match crate::adapter_swap::swap_runtime_adapter(
                    state,
                    crate::adapter_swap::SwapRequest {
                        target: crate::adapter_swap::SwapTarget::Resolved {
                            active_name: gate.adapter_name.clone(),
                            dir: adapter_dir,
                        },
                        content_changed: true,
                        default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Replace(Some(
                            gate.adapter_name.clone(),
                        )),
                        reason: "post_eval_gate_promotion",
                    },
                )
                .await
                {
                    Ok(_) => {
                        stamp_verdict(
                            crate::state::GateOutcome::Promoted,
                            format!(
                                "PASSED: {}; accuracy {accuracy:.3}, 95% lower bound {:.3} >= {:.3}; adapter `{}` promoted to active",
                                statistical.reason,
                                statistical.evidence.candidate_accuracy_lower_bound,
                                gate.min_accuracy,
                                gate.adapter_name
                            ),
                            Some(statistical.evidence.clone()),
                        );
                    }
                    // The gate itself passed, but the system failed to apply
                    // the promotion — that is an operational error, not a
                    // measured success or failure.
                    Err(e) => stamp_verdict(
                        crate::state::GateOutcome::Error,
                        format!(
                            "STATISTICAL GATE PASSED: accuracy {accuracy:.3}, but promotion failed: {e}"
                        ),
                        Some(statistical.evidence.clone()),
                    ),
                }
            } else {
                stamp_verdict(
                    crate::state::GateOutcome::Kept,
                    format!(
                        "PASSED: {}; accuracy {accuracy:.3}, 95% lower bound {:.3} >= {:.3}; adapter `{}` kept (auto_load not requested)",
                        statistical.reason,
                        statistical.evidence.candidate_accuracy_lower_bound,
                        gate.min_accuracy,
                        gate.adapter_name
                    ),
                    Some(statistical.evidence),
                );
            }
            return;
        }
        StatisticalGateDecision::Demoted => {}
    }

    let demotion_evidence = statistical.evidence;

    // Conclusively failed the accuracy floor. Own the same revision barrier as load, delete, upload,
    // and training publication until the serving name has been removed. This
    // makes the loaded check, optional unload, default clear, rename, and cache
    // purge one serialized transaction.
    let serial = match crate::adapter_swap::adapter_mutation_guard(state).await {
        Ok(serial) => serial,
        Err(error) => {
            stamp_verdict(
                crate::state::GateOutcome::Error,
                format!(
                    "FAILED: accuracy {accuracy:.3} < {:.3}, but adapter `{}` could not be demoted: {error}",
                    gate.min_accuracy, gate.adapter_name
                ),
                Some(demotion_evidence.clone()),
            );
            return;
        }
    };
    let currently_loaded = state.loaded_adapter_name();
    if currently_loaded.as_deref() == Some(gate.adapter_name.as_str()) {
        if let Err(error) = crate::adapter_swap::swap_runtime_adapter_locked(
            state,
            crate::adapter_swap::SwapRequest {
                target: crate::adapter_swap::SwapTarget::Base,
                content_changed: false,
                default_adapter: crate::adapter_swap::DefaultAdapterUpdate::ClearIf(
                    gate.adapter_name.clone(),
                ),
                reason: "post_eval_gate_demotion",
            },
            &serial,
        )
        .await
        {
            stamp_verdict(
                crate::state::GateOutcome::Error,
                format!(
                    "FAILED: accuracy {accuracy:.3} < {:.3}, but loaded adapter `{}` could not be swapped away: {error}",
                    gate.min_accuracy, gate.adapter_name
                ),
                Some(demotion_evidence.clone()),
            );
            return;
        }
    } else {
        crate::adapter_swap::apply_default_update(
            state,
            &crate::adapter_swap::DefaultAdapterUpdate::ClearIf(gate.adapter_name.clone()),
        );
    }

    let src = state.adapter_dir.join(&gate.adapter_name);
    let mut dst = state
        .adapter_dir
        .join(format!("{}.failed", gate.adapter_name));
    if dst.exists() {
        dst = state.adapter_dir.join(format!(
            "{}.failed-{}",
            gate.adapter_name,
            crate::recent_requests::now_unix_ms()
        ));
    }
    if let Err(error) = std::fs::rename(&src, &dst) {
        stamp_verdict(
            crate::state::GateOutcome::Error,
            format!(
                "FAILED: accuracy {accuracy:.3} < {:.3}, but adapter `{}` could not be renamed to .failed: {error}",
                gate.min_accuracy, gate.adapter_name
            ),
            Some(demotion_evidence.clone()),
        );
        return;
    }
    state.purge_adapter_caches(&Some(gate.adapter_name.clone()));
    drop(serial);
    let rename_note = format!(
        "renamed to `{}`",
        dst.file_name().unwrap_or_default().to_string_lossy()
    );
    stamp_verdict(
        crate::state::GateOutcome::Demoted,
        format!(
            "FAILED: {}; accuracy {accuracy:.3}, 95% upper bound {:.3} < {:.3}; adapter `{}` NOT promoted, {rename_note}",
            statistical.reason,
            demotion_evidence.candidate_accuracy_upper_bound,
            gate.min_accuracy,
            gate.adapter_name
        ),
        Some(demotion_evidence),
    );
}

async fn run_job(
    state: &AppState,
    job: &QueuedEvalJob,
    effective_seed: u64,
    replay_source_record: Option<Arc<kiln_eval::EvalReplayRecordV1>>,
    generator: Arc<dyn crate::eval::generator::EvalGenerator>,
    judge_runner: Arc<dyn JudgeRunner>,
    progress: Option<crate::eval::executor::ProgressCallback>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
) -> Result<Vec<kiln_eval::SuiteResult>, String> {
    let replay_environment = EvalReplayEnvironment {
        execution_provenance_sha256: state
            .execution_provenance
            .as_deref()
            .map(|provenance| provenance.provenance_sha256.clone()),
        base_weight_manifest_sha256: state
            .base_weight_shard_manifest
            .as_deref()
            .map(|manifest| manifest.aggregate_sha256.clone()),
    };
    match job {
        QueuedEvalJob::Registered {
            suite_name,
            adapter,
            generation_override,
        } => {
            let suite = state
                .suite_registry
                .as_ref()
                .ok_or_else(|| "no suite registry configured".to_string())?
                .load(suite_name)
                .map_err(|e| format!("{e}"))?;
            let r = run_suite_against_adapter_with_replay(
                &suite,
                adapter.as_deref(),
                generation_override.as_ref(),
                effective_seed,
                generator,
                progress,
                cancel_flag,
                judge_runner,
                replay_environment,
                replay_source_record,
            )
            .await
            .map_err(|e| format!("{e}"))?;
            Ok(vec![r])
        }
        QueuedEvalJob::Inline {
            suite,
            adapter,
            generation_override,
        } => {
            let r = run_suite_against_adapter_with_replay(
                suite,
                adapter.as_deref(),
                generation_override.as_ref(),
                effective_seed,
                generator,
                progress,
                cancel_flag,
                judge_runner,
                replay_environment,
                replay_source_record,
            )
            .await
            .map_err(|e| format!("{e}"))?;
            Ok(vec![r])
        }
        QueuedEvalJob::Compare(spec) => {
            let suite = state
                .suite_registry
                .as_ref()
                .ok_or_else(|| "no suite registry configured".to_string())?
                .load(&spec.suite)
                .map_err(|e| format!("{e}"))?;
            let mut runs = Vec::with_capacity(spec.adapters.len());
            // The single `progress` callback owns the progress slot on
            // EvalJobInfo. Cloning per-adapter would race; instead we reuse
            // it across adapters and let the slot reflect the most recent
            // adapter's running accuracy. Without this the UI's compare
            // panel shows zero progress for the entire job.
            let mut progress_slot = progress;
            for adapter in &spec.adapters {
                // A cancelled compare job must not keep swapping adapters
                // for runs nobody asked to finish.
                if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let adapter_opt = if adapter.is_empty() {
                    None
                } else {
                    Some(adapter.as_str())
                };
                let r = run_suite_against_adapter_with_replay(
                    &suite,
                    adapter_opt,
                    spec.generation.as_ref(),
                    effective_seed,
                    generator.clone(),
                    progress_slot.take(),
                    cancel_flag.clone(),
                    judge_runner.clone(),
                    replay_environment.clone(),
                    replay_source_record.clone(),
                )
                .await
                .map_err(|e| format!("compare ({}): {e}", adapter))?;
                runs.push(r);
            }
            Ok(runs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::queue::{EvalJobInfo, EvalQueueEntry, EvalSubmissionKind};
    use kiln_eval::scorers::Scorer;
    use kiln_eval::{EvalChatMessage, EvalExample, EvalGenerationParams, EvalSuite};
    use tower::ServiceExt;

    fn aggregate_outcome(
        id: &str,
        kind: kiln_eval::EvalOutcomeKind,
        score: f32,
    ) -> kiln_eval::AggregatedExampleOutcome {
        kiln_eval::AggregatedExampleOutcome {
            example_id: id.into(),
            kind,
            score,
            completion_indices: vec![0, 1, 2],
            representative_completion_index: 1,
            num_pass: if kind == kiln_eval::EvalOutcomeKind::Pass {
                2
            } else {
                1
            },
            num_fail: if kind == kiln_eval::EvalOutcomeKind::Pass {
                1
            } else {
                2
            },
            num_invalid: 0,
            num_error: 0,
            tags: Vec::new(),
            metadata: None,
        }
    }

    fn raw_outcome(id: &str, kind: kiln_eval::EvalOutcomeKind) -> kiln_eval::ExampleOutcome {
        kiln_eval::ExampleOutcome {
            example_id: id.into(),
            completion_index: 0,
            generation_seed: None,
            completion_text: String::new(),
            raw_completion_text: None,
            thinking_budget: None,
            kind,
            score: if kind == kiln_eval::EvalOutcomeKind::Pass {
                1.0
            } else {
                0.0
            },
            detail: None,
            prompt_tokens: None,
            completion_tokens: None,
            latency_ms: None,
            tags: Vec::new(),
            metadata: None,
            reasoning_text: None,
            unclosed_thinking: false,
        }
    }

    fn replay_run(completion: &str) -> kiln_eval::SuiteResult {
        let suite = gate_suite(completion);
        let outcome = kiln_eval::ExampleOutcome {
            example_id: "e0".into(),
            completion_index: 0,
            generation_seed: Some(kiln_eval::derive_eval_completion_seed(7, "e0", 0)),
            completion_text: completion.into(),
            raw_completion_text: Some(completion.into()),
            thinking_budget: Some(kiln_eval::EvalThinkingBudget::default()),
            kind: kiln_eval::EvalOutcomeKind::Pass,
            score: 1.0,
            detail: None,
            prompt_tokens: Some(1),
            completion_tokens: Some(1),
            latency_ms: Some(1.0),
            tags: Vec::new(),
            metadata: None,
            reasoning_text: None,
            unclosed_thinking: false,
        };
        let record = kiln_eval::EvalReplayRecordV1::new(
            suite.clone(),
            None,
            7,
            vec![kiln_eval::EvalThinkingBudget::default()],
            Some(kiln_eval::EvalModelTargetIdentity::base()),
            Vec::new(),
            Some(format!("sha256:{}", "a".repeat(64))),
            Some(format!("sha256:{}", "b".repeat(64))),
            std::slice::from_ref(&outcome),
        )
        .unwrap();
        kiln_eval::SuiteResult {
            suite_name: suite.name,
            adapter: None,
            aggregation: suite.aggregation,
            metrics: kiln_eval::AggregateMetrics::default(),
            outcomes: vec![outcome],
            aggregated_outcomes: Vec::new(),
            started_at: "2026-07-20T00:00:00Z".into(),
            finished_at: "2026-07-20T00:00:01Z".into(),
            suite_hash: record.suite_sha256.clone(),
            effective_generation_hash: record.effective_generation_sha256.clone(),
            replay_record: Some(record),
        }
    }

    #[test]
    fn strict_replay_verdict_distinguishes_match_mismatch_and_incomplete_runs() {
        let source = replay_run("x");
        let expectation = kiln_eval::EvalReplayExpectationV1::new(
            "source-job".into(),
            0,
            source.replay_record.as_ref().unwrap(),
        );

        let matched = evaluate_replay_verdict(&expectation, std::slice::from_ref(&source));
        assert_eq!(matched.status, kiln_eval::EvalReplayStatus::Matched);
        matched.validate(&expectation).unwrap();

        let changed = replay_run("y");
        let mismatched = evaluate_replay_verdict(&expectation, &[changed]);
        assert_eq!(mismatched.status, kiln_eval::EvalReplayStatus::Mismatch);
        mismatched.validate(&expectation).unwrap();

        let error = evaluate_replay_verdict(&expectation, &[]);
        assert_eq!(error.status, kiln_eval::EvalReplayStatus::Error);
        error.validate(&expectation).unwrap();
    }

    fn paired_run(
        adapter: &str,
        aggregate: kiln_eval::AggregatedExampleOutcome,
        raw: kiln_eval::ExampleOutcome,
    ) -> kiln_eval::SuiteResult {
        kiln_eval::SuiteResult {
            suite_name: "paired".into(),
            adapter: Some(adapter.into()),
            aggregation: kiln_eval::EvalAggregation::MajorityAtK { k: 3 },
            metrics: kiln_eval::AggregateMetrics::default(),
            outcomes: vec![raw],
            aggregated_outcomes: vec![aggregate],
            started_at: "2026-07-14T00:00:00Z".into(),
            finished_at: "2026-07-14T00:00:01Z".into(),
            suite_hash: "suite".into(),
            effective_generation_hash: "generation".into(),
            replay_record: None,
        }
    }

    #[test]
    fn paired_promotion_flips_use_reduced_examples_not_completion_zero() {
        let baseline = paired_run(
            "base",
            aggregate_outcome("e1", kiln_eval::EvalOutcomeKind::Fail, 0.0),
            raw_outcome("e1", kiln_eval::EvalOutcomeKind::Pass),
        );
        let candidate = paired_run(
            "candidate",
            aggregate_outcome("e1", kiln_eval::EvalOutcomeKind::Pass, 1.0),
            raw_outcome("e1", kiln_eval::EvalOutcomeKind::Fail),
        );

        assert_eq!(
            paired_aggregate_flips(&baseline, &candidate),
            Ok(PairedAggregateSummary {
                paired_examples: 1,
                improved: 1,
                regressed: 0,
                tied: 0,
                baseline_pass: 0,
                candidate_pass: 1,
            })
        );
    }

    #[test]
    fn independent_gate_outcomes_retain_the_strongest_failure() {
        assert_eq!(
            aggregate_gate_outcome(Some("regression"), "kept"),
            "regression"
        );
        assert_eq!(
            aggregate_gate_outcome(Some("kept"), "inconclusive"),
            "inconclusive"
        );
        assert_eq!(
            aggregate_gate_outcome(Some("inconclusive"), "demoted"),
            "demoted"
        );
        assert_eq!(aggregate_gate_outcome(Some("regression"), "error"), "error");
        assert_eq!(aggregate_gate_outcome(Some("kept"), "promoted"), "promoted");
    }

    struct PanicIfInvokedGenerator;

    impl crate::eval::generator::EvalGenerator for PanicIfInvokedGenerator {
        fn set_adapter(
            &self,
            _adapter: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
        > {
            panic!("maintenance worker invoked eval adapter selection")
        }

        fn prepare(
            &self,
            _messages: &[EvalChatMessage],
            _system_prompt: Option<&str>,
            _tools: Option<&[serde_json::Value]>,
            _params: &EvalGenerationParams,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<crate::eval::generator::PreparedPrompt, String>,
                    > + Send
                    + '_,
            >,
        > {
            panic!("maintenance worker invoked eval prompt preparation")
        }

        fn run(
            &self,
            _prepared: &crate::eval::generator::PreparedPrompt,
            _params: &EvalGenerationParams,
            _thinking_budget: &kiln_eval::EvalThinkingBudget,
            _completion_index: usize,
            _adapter_label: Option<&str>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::eval::EvalCompletion, String>>
                    + Send
                    + '_,
            >,
        > {
            panic!("maintenance worker invoked eval generation")
        }
    }

    fn big_suite(n: usize) -> EvalSuite {
        EvalSuite {
            name: "cancel-me".into(),
            description: None,
            default_scorer: Scorer::Contains {
                phrases: vec!["mock".into()],
                mode: Default::default(),
                case_sensitive: false,
            },
            generation: EvalGenerationParams::default(),
            aggregation: kiln_eval::EvalAggregation::Single,
            system_prompt: None,
            examples: (0..n)
                .map(|i| EvalExample {
                    id: Some(format!("e{i}")),
                    messages: vec![EvalChatMessage::new("user", format!("ping {i}"))],
                    target: None,
                    ..Default::default()
                })
                .collect(),
            schema_version: 1,
            tools: None,
        }
    }

    /// Gate wrapper: first example completes, then generation parks on a
    /// Notify until the test has cancelled through the real DELETE route.
    struct GatedGenerator {
        inner: crate::eval::MockEvalGenerator,
        reached: Arc<tokio::sync::Notify>,
        resume: Arc<tokio::sync::Notify>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl crate::eval::generator::EvalGenerator for GatedGenerator {
        fn set_adapter(
            &self,
            adapter: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<Option<String>, String>> + Send + '_>,
        > {
            self.inner.set_adapter(adapter)
        }

        fn prepare(
            &self,
            messages: &[EvalChatMessage],
            system_prompt: Option<&str>,
            tools: Option<&[serde_json::Value]>,
            params: &EvalGenerationParams,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<crate::eval::generator::PreparedPrompt, String>,
                    > + Send
                    + '_,
            >,
        > {
            self.inner.prepare(messages, system_prompt, tools, params)
        }

        fn run(
            &self,
            prepared: &crate::eval::generator::PreparedPrompt,
            params: &EvalGenerationParams,
            thinking_budget: &kiln_eval::EvalThinkingBudget,
            completion_index: usize,
            adapter_label: Option<&str>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::eval::EvalCompletion, String>>
                    + Send
                    + '_,
            >,
        > {
            let call = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let fut = self.inner.run(
                prepared,
                params,
                thinking_budget,
                completion_index,
                adapter_label,
            );
            let reached = self.reached.clone();
            let resume = self.resume.clone();
            Box::pin(async move {
                if call == 1 {
                    // One outcome is recorded; tell the test we are mid-run
                    // and wait for it to fire the cancellation.
                    reached.notify_one();
                    resume.notified().await;
                }
                fut.await
            })
        }
    }

    fn gate_test_state() -> (crate::state::AppState, tempfile::TempDir) {
        let adapter_dir = tempfile::tempdir().unwrap();
        let config = kiln_core::config::ModelConfig::qwen3_5_4b();
        let scheduler = kiln_scheduler::Scheduler::new(
            kiln_scheduler::SchedulerConfig {
                max_batch_tokens: 8192,
                max_batch_size: 64,
                block_size: 16,
                prefix_cache_enabled: false,
                ..Default::default()
            },
            256,
        );
        let engine = kiln_model::engine::MockEngine::new(config.clone());
        let mut state = crate::state::AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            crate::api::test_tokenizer(),
            300,
            "kiln-test".to_string(),
        );
        state.adapter_dir = adapter_dir.path().to_path_buf();
        (state, adapter_dir)
    }

    #[tokio::test]
    async fn maintenance_rejects_injected_eval_before_generator_invocation() {
        let (mut state, _dir) = gate_test_state();
        state.serving_profile = crate::config::ServingProfileSetting::new(
            crate::config::ServingProfile::Maintenance,
            crate::config::ConfigValueSource::ConfigFile,
        );
        let suite = gate_suite("unused");
        let job_id = "maintenance-injected-eval".to_string();
        state.eval_jobs.write().unwrap().insert(
            job_id.clone(),
            EvalJobInfo::queued(
                job_id.clone(),
                suite.name.clone(),
                vec![None],
                EvalSubmissionKind::OnDemand,
                None,
                17,
            ),
        );
        let entry = EvalQueueEntry {
            job_id: job_id.clone(),
            effective_seed: 17,
            replay_source_record: None,
            job: QueuedEvalJob::Inline {
                suite: Box::new(suite),
                adapter: None,
                generation_override: None,
            },
        };

        run_one_job_with_generator(state.clone(), entry, Arc::new(PanicIfInvokedGenerator)).await;

        let jobs = state.eval_jobs.read().unwrap();
        let job = jobs.get(&job_id).unwrap();
        assert_eq!(job.state, kiln_eval::EvalJobState::Failed);
        assert!(
            job.error
                .as_deref()
                .is_some_and(|error| error.contains("disables inference admission")),
            "{:?}",
            job.error
        );
    }

    fn seed_training_job(state: &crate::state::AppState, job_id: &str, adapter: &str) {
        state.training_jobs.write().unwrap().insert(
            job_id.to_string(),
            crate::state::TrainingJobInfo {
                job_id: job_id.to_string(),
                adapter_name: adapter.to_string(),
                job_type: crate::state::TrainingJobType::Sft,
                effective_seed: Some(17),
                state: kiln_train::TrainingState::Completed,
                progress: 1.0,
                loss: None,
                epoch: None,
                adapter_path: None,
                submitted_at: std::time::Instant::now(),
                submitted_unix_ms: crate::recent_requests::now_unix_ms(),
                auto_load: true,
                consumed_correction_ids: Vec::new(),
                training_data: None,
                finished_at: None,
                finished_unix_ms: None,
                error: None,
                linked_eval_job_ids: Vec::new(),
                post_eval_verdict: None,
                gate_outcome: None,
                post_eval_gate_evidence: Vec::new(),
                loss_history: Vec::new(),
                cancel_requested: Default::default(),
            },
        );
    }

    /// One-example suite whose scorer passes iff the mock generator's
    /// reply contains `phrase`.
    fn gate_suite(phrase: &str) -> EvalSuite {
        EvalSuite {
            name: "gate-suite".into(),
            description: None,
            default_scorer: Scorer::Contains {
                phrases: vec![phrase.into()],
                mode: Default::default(),
                case_sensitive: false,
            },
            generation: EvalGenerationParams::default(),
            aggregation: kiln_eval::EvalAggregation::Single,
            system_prompt: None,
            examples: vec![EvalExample {
                id: Some("e0".into()),
                messages: vec![EvalChatMessage::new("user", "ping")],
                target: None,
                ..Default::default()
            }],
            schema_version: 1,
            tools: None,
        }
    }

    fn synthetic_gate_run(
        adapter: Option<&str>,
        pass: bool,
        examples: u32,
    ) -> kiln_eval::SuiteResult {
        synthetic_gate_run_with_passes(adapter, examples, if pass { examples } else { 0 })
    }

    fn synthetic_gate_run_with_passes(
        adapter: Option<&str>,
        examples: u32,
        pass_count: u32,
    ) -> kiln_eval::SuiteResult {
        assert!(pass_count <= examples);
        kiln_eval::SuiteResult {
            suite_name: "gate-suite".into(),
            adapter: adapter.map(str::to_string),
            aggregation: kiln_eval::EvalAggregation::Single,
            metrics: kiln_eval::AggregateMetrics::default(),
            outcomes: Vec::new(),
            aggregated_outcomes: (0..examples)
                .map(|index| {
                    let pass = index < pass_count;
                    aggregate_outcome(
                        &format!("e{index}"),
                        if pass {
                            kiln_eval::EvalOutcomeKind::Pass
                        } else {
                            kiln_eval::EvalOutcomeKind::Fail
                        },
                        if pass { 1.0 } else { 0.0 },
                    )
                })
                .collect(),
            started_at: "2026-07-20T00:00:00Z".into(),
            finished_at: "2026-07-20T00:00:01Z".into(),
            suite_hash: "sha256:gate-suite".into(),
            effective_generation_hash: "sha256:gate-generation".into(),
            replay_record: None,
        }
    }

    fn test_gate(min_accuracy: f32) -> crate::eval::queue::PostEvalGate {
        crate::eval::queue::PostEvalGate {
            min_accuracy,
            relative_recovery: None,
            absolute_gain: None,
            adapter_name: "candidate".into(),
            training_job_id: "train".into(),
            auto_load_on_pass: false,
        }
    }

    #[test]
    fn paired_policy_requires_significant_improvement_not_a_tied_point_pass() {
        let baseline = synthetic_gate_run(None, true, 40);
        let candidate = synthetic_gate_run(Some("candidate"), true, 40);
        let result = evaluate_paired_promotion("eval", &test_gate(0.9), &baseline, &candidate)
            .expect("valid paired evidence");

        assert_eq!(result.decision, StatisticalGateDecision::Inconclusive);
        assert_eq!(
            (result.evidence.improved, result.evidence.regressed),
            (0, 0)
        );
        assert_eq!(result.evidence.exact_sign_test_p_value, 1.0);
        assert!(result.evidence.candidate_accuracy_lower_bound >= 0.9);
    }

    #[test]
    fn paired_policy_rejects_significant_regression() {
        let baseline = synthetic_gate_run(None, true, 40);
        let candidate = synthetic_gate_run(Some("candidate"), false, 40);
        let result = evaluate_paired_promotion("eval", &test_gate(0.0), &baseline, &candidate)
            .expect("valid paired evidence");

        assert_eq!(result.decision, StatisticalGateDecision::Regression);
        assert_eq!(
            (result.evidence.improved, result.evidence.regressed),
            (0, 40)
        );
        assert!(result.evidence.exact_sign_test_p_value < POST_EVAL_EXACT_TEST_ALPHA);
    }

    #[test]
    fn paired_policy_does_not_promote_when_only_point_accuracy_clears_floor() {
        let baseline = synthetic_gate_run(None, false, 40);
        let candidate = synthetic_gate_run_with_passes(Some("candidate"), 40, 38);
        let result = evaluate_paired_promotion("eval", &test_gate(0.9), &baseline, &candidate)
            .expect("valid paired evidence");

        assert_eq!(result.evidence.candidate_accuracy, 0.95);
        assert!(result.evidence.candidate_accuracy_lower_bound < 0.9);
        assert!(result.evidence.candidate_accuracy_upper_bound >= 0.9);
        assert_eq!(result.decision, StatisticalGateDecision::Inconclusive);
    }

    #[test]
    fn paired_policy_uses_confidence_bound_for_relative_recovery() {
        let baseline = synthetic_gate_run(None, true, 40);
        let candidate = synthetic_gate_run(Some("candidate"), true, 40);
        let mut gate = test_gate(0.0);
        gate.relative_recovery = Some(0.9);
        let result = evaluate_paired_promotion("eval", &gate, &baseline, &candidate)
            .expect("valid paired evidence");

        assert_eq!(result.decision, StatisticalGateDecision::Pass);
        assert!(result.evidence.relative_recovery_lower_bound.unwrap() >= 0.9);
        assert_eq!(result.evidence.exact_sign_test_p_value, 1.0);
    }

    async fn run_gated_job(
        state: &crate::state::AppState,
        gate: crate::eval::queue::PostEvalGate,
        baseline_pass: bool,
        candidate_pass: bool,
        examples: u32,
    ) {
        let job_id = format!("eval-{}", gate.adapter_name);
        let mut info = EvalJobInfo::queued(
            job_id.clone(),
            "gate-suite".into(),
            vec![None, Some(gate.adapter_name.clone())],
            EvalSubmissionKind::PostTraining,
            Some(gate.training_job_id.clone()),
            17,
        );
        info.post_eval_gate = Some(gate.clone());
        info.state = kiln_eval::EvalJobState::Completed;
        info.finished_runs = vec![
            synthetic_gate_run(None, baseline_pass, examples),
            synthetic_gate_run(Some(&gate.adapter_name), candidate_pass, examples),
        ];
        state
            .eval_jobs
            .write()
            .unwrap()
            .insert(job_id, info.clone());
        apply_post_eval_gate(state, &info).await;
    }

    fn verdict_of(state: &crate::state::AppState, training_job: &str) -> String {
        state
            .training_jobs
            .read()
            .unwrap()
            .get(training_job)
            .and_then(|j| j.post_eval_verdict.clone())
            .expect("verdict stamped")
    }

    /// The machine-readable twin stamped next to the prose verdict.
    fn outcome_of(state: &crate::state::AppState, training_job: &str) -> String {
        state
            .training_jobs
            .read()
            .unwrap()
            .get(training_job)
            .and_then(|j| j.gate_outcome.clone())
            .expect("gate_outcome stamped")
    }

    fn evidence_of(
        state: &crate::state::AppState,
        training_job: &str,
    ) -> kiln_train::PostEvalGateEvidence {
        state
            .training_jobs
            .read()
            .unwrap()
            .get(training_job)
            .and_then(|job| job.post_eval_gate_evidence.first().cloned())
            .expect("post_eval_gate_evidence stamped")
    }

    /// Gate FAIL: the adapter directory is renamed `<name>.failed` (the
    /// documented PostEvalConfig::min_accuracy contract) and the verdict
    /// lands on the training job. Before this, min_accuracy was parsed,
    /// documented — and read by no code at all.
    #[tokio::test]
    async fn gate_failure_renames_adapter_and_stamps_verdict() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-1", "gated");
        std::fs::create_dir(state.adapter_dir.join("gated")).unwrap();
        *state.active_adapter_name.write().unwrap() = Some("gated".to_string());
        *state.loaded_adapter.write().unwrap() = Some(crate::state::LoadedAdapterIdentity {
            name: "request-override".to_string(),
            content_revision: "c".repeat(64),
        });

        run_gated_job(
            &state,
            crate::eval::queue::PostEvalGate {
                min_accuracy: 0.9,
                relative_recovery: None,
                absolute_gain: None,
                adapter_name: "gated".into(),
                training_job_id: "train-1".into(),
                auto_load_on_pass: false,
            },
            false,
            false,
            40,
        )
        .await;

        assert!(
            !state.adapter_dir.join("gated").exists(),
            "failed adapter must not stay under its serving name"
        );
        assert!(
            state.adapter_dir.join("gated.failed").exists(),
            "failed adapter renamed per the documented contract"
        );
        let verdict = verdict_of(&state, "train-1");
        assert!(verdict.contains("FAILED"), "{verdict}");
        assert!(verdict.contains("gated.failed"), "{verdict}");
        assert_eq!(outcome_of(&state, "train-1"), "demoted");
        let evidence = evidence_of(&state, "train-1");
        assert_eq!(evidence.policy_version, "paired_wilson_v1");
        assert_eq!(evidence.paired_examples, 40);
        assert_eq!(evidence.outcome, "demoted");
        assert!(evidence.candidate_accuracy_upper_bound < 0.9);
        assert!(
            state.active_adapter_name.read().unwrap().is_none(),
            "demotion must clear a rejected server default even when a request override is loaded"
        );
        assert_eq!(
            state.loaded_adapter_name().as_deref(),
            Some("request-override"),
            "demotion must not unload a different physically loaded revision"
        );
    }

    /// Gate PASS without a deferred auto-load: adapter stays under its
    /// name, verdict records the pass.
    #[tokio::test]
    async fn gate_pass_keeps_adapter_and_stamps_verdict() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-2", "good");
        std::fs::create_dir(state.adapter_dir.join("good")).unwrap();

        run_gated_job(
            &state,
            crate::eval::queue::PostEvalGate {
                min_accuracy: 0.9,
                relative_recovery: None,
                absolute_gain: None,
                adapter_name: "good".into(),
                training_job_id: "train-2".into(),
                auto_load_on_pass: false,
            },
            false,
            true,
            40,
        )
        .await;

        assert!(state.adapter_dir.join("good").exists());
        assert!(!state.adapter_dir.join("good.failed").exists());
        let verdict = verdict_of(&state, "train-2");
        assert!(verdict.contains("PASSED"), "{verdict}");
        // A pass without a requested promotion is `kept` — distinct from
        // `promoted` so the dashboard never paints it as a warning by
        // substring-sniffing the prose.
        assert_eq!(outcome_of(&state, "train-2"), "kept");
        let evidence = evidence_of(&state, "train-2");
        assert_eq!((evidence.improved, evidence.regressed), (40, 0));
        assert!(evidence.exact_sign_test_p_value < 0.05);
        assert!(evidence.candidate_accuracy_lower_bound >= 0.9);
        assert_eq!(evidence.outcome, "kept");
    }

    #[tokio::test]
    async fn gate_with_too_few_pairs_is_inconclusive_and_keeps_serving_name() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-small", "small");
        std::fs::create_dir(state.adapter_dir.join("small")).unwrap();

        run_gated_job(
            &state,
            crate::eval::queue::PostEvalGate {
                min_accuracy: 0.5,
                relative_recovery: None,
                absolute_gain: None,
                adapter_name: "small".into(),
                training_job_id: "train-small".into(),
                auto_load_on_pass: true,
            },
            false,
            true,
            POST_EVAL_MIN_PAIRED_EXAMPLES - 1,
        )
        .await;

        assert_eq!(outcome_of(&state, "train-small"), "inconclusive");
        assert!(state.adapter_dir.join("small").exists());
        assert!(!state.adapter_dir.join("small.failed").exists());
        assert!(state.active_adapter_name.read().unwrap().is_none());
        let evidence = evidence_of(&state, "train-small");
        assert_eq!(evidence.paired_examples, 19);
        assert_eq!(evidence.minimum_paired_examples, 20);
        assert_eq!(evidence.outcome, "inconclusive");
    }

    /// An errored eval must leave the adapter on disk and NOT promote it —
    /// an unmeasured adapter never starts serving.
    #[tokio::test]
    async fn gate_on_errored_eval_does_not_promote_or_rename() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-3", "unmeasured");
        std::fs::create_dir(state.adapter_dir.join("unmeasured")).unwrap();

        // Registered suite with no registry configured → run_job errors →
        // terminal state Failed.
        let job_id = "eval-unmeasured".to_string();
        let mut info = EvalJobInfo::queued(
            job_id.clone(),
            "missing-suite".into(),
            vec![Some("unmeasured".into())],
            EvalSubmissionKind::PostTraining,
            Some("train-3".into()),
            17,
        );
        info.post_eval_gate = Some(crate::eval::queue::PostEvalGate {
            min_accuracy: 0.5,
            relative_recovery: None,
            absolute_gain: None,
            adapter_name: "unmeasured".into(),
            training_job_id: "train-3".into(),
            auto_load_on_pass: true,
        });
        state
            .eval_jobs
            .write()
            .unwrap()
            .insert(job_id.clone(), info);
        let entry = EvalQueueEntry {
            job_id,
            effective_seed: 17,
            replay_source_record: None,
            job: QueuedEvalJob::Registered {
                suite_name: "missing-suite".into(),
                adapter: Some("unmeasured".into()),
                generation_override: None,
            },
        };
        let generator = Arc::new(crate::eval::MockEvalGenerator::new())
            as Arc<dyn crate::eval::generator::EvalGenerator>;
        run_one_job_with_generator(state.clone(), entry, generator).await;

        assert!(state.adapter_dir.join("unmeasured").exists());
        assert!(!state.adapter_dir.join("unmeasured.failed").exists());
        assert!(
            state.active_adapter_name.read().unwrap().is_none(),
            "unmeasured adapter must not be promoted"
        );
        let verdict = verdict_of(&state, "train-3");
        assert!(verdict.contains("did not complete"), "{verdict}");
        assert!(verdict.contains("NOT promoted"), "{verdict}");
        assert_eq!(outcome_of(&state, "train-3"), "error");
    }

    /// The stamped verdict + machine-readable outcome must survive all the
    /// way to the API payload shape: GET /v1/train/queue serializes the
    /// completed job with BOTH `post_eval_verdict` (prose, for humans) and
    /// `gate_outcome` (string enum, for the dashboard pill). Before
    /// `gate_outcome` existed the UI classified the prose by substring and
    /// a PASSED gate could render as a warning.
    #[tokio::test]
    async fn gate_outcome_reaches_train_queue_api_payload() {
        let (state, _dir) = gate_test_state();
        seed_training_job(&state, "train-api", "api-gated");
        std::fs::create_dir(state.adapter_dir.join("api-gated")).unwrap();

        run_gated_job(
            &state,
            crate::eval::queue::PostEvalGate {
                min_accuracy: 0.9,
                relative_recovery: None,
                absolute_gain: None,
                adapter_name: "api-gated".into(),
                training_job_id: "train-api".into(),
                auto_load_on_pass: false,
            },
            false,
            true,
            40,
        )
        .await;

        let app = crate::api::router(state.clone());
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/v1/train/queue")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let job = payload["completed"]
            .as_array()
            .expect("completed array")
            .iter()
            .find(|j| j["job_id"] == "train-api")
            .expect("gated training job present in /v1/train/queue");
        assert_eq!(job["gate_outcome"], "kept", "payload: {job}");
        assert_eq!(
            job["post_eval_gate_evidence"][0]["policy_version"], "paired_wilson_v1",
            "payload: {job}"
        );
        assert_eq!(
            job["post_eval_gate_evidence"][0]["paired_examples"], 40,
            "payload: {job}"
        );
        assert!(
            job["post_eval_verdict"]
                .as_str()
                .is_some_and(|v| v.contains("PASSED")),
            "payload: {job}"
        );
    }

    /// DELETE /v1/eval/jobs/{id} mid-run must stop the executor at the next
    /// example boundary and the terminal state must STAY Cancelled with the
    /// partial outcomes recorded — the worker used to clobber it back to
    /// Completed and the flag was never wired, so cancellation did nothing.
    #[tokio::test]
    async fn cancel_running_job_stops_early_and_stays_cancelled() {
        // Archive writes go under adapter_dir — point it at a tempdir so
        // the test never litters the repo tree (a stray adapters/.kiln-jobs
        // breaks the health adapter-count test).
        let adapter_dir = tempfile::tempdir().unwrap();
        let state = {
            let config = kiln_core::config::ModelConfig::qwen3_5_4b();
            let scheduler = kiln_scheduler::Scheduler::new(
                kiln_scheduler::SchedulerConfig {
                    max_batch_tokens: 8192,
                    max_batch_size: 64,
                    block_size: 16,
                    prefix_cache_enabled: false,
                    ..Default::default()
                },
                256,
            );
            let engine = kiln_model::engine::MockEngine::new(config.clone());
            crate::state::AppState::new_mock(
                config,
                scheduler,
                Arc::new(engine),
                crate::api::test_tokenizer(),
                300,
                "kiln-test".to_string(),
            )
        };
        let mut state = state;
        state.adapter_dir = adapter_dir.path().to_path_buf();
        let state = state;
        let total_examples = 8usize;
        let job_id = "job-cancel".to_string();
        state.eval_jobs.write().unwrap().insert(
            job_id.clone(),
            EvalJobInfo::queued(
                job_id.clone(),
                "cancel-me".into(),
                vec![None],
                EvalSubmissionKind::OnDemand,
                None,
                17,
            ),
        );
        let entry = EvalQueueEntry {
            job_id: job_id.clone(),
            effective_seed: 17,
            replay_source_record: None,
            job: QueuedEvalJob::Inline {
                suite: Box::new(big_suite(total_examples)),
                adapter: None,
                generation_override: None,
            },
        };

        let reached = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        let generator = Arc::new(GatedGenerator {
            inner: crate::eval::MockEvalGenerator::new(),
            reached: reached.clone(),
            resume: resume.clone(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }) as Arc<dyn crate::eval::generator::EvalGenerator>;

        let worker = tokio::spawn(run_one_job_with_generator(state.clone(), entry, generator));

        // Deterministic: the gate fires after the first outcome landed and
        // the second example is mid-generation.
        reached.notified().await;
        {
            let jobs = state.eval_jobs.read().unwrap();
            let job = jobs.get(&job_id).unwrap();
            assert_eq!(job.state, EvalJobState::Running);
            assert!(job.cancel_flag.is_some(), "worker must install the handle");
        }

        // Cancel through the real route, exactly as the dashboard does.
        let app = crate::api::router(state.clone());
        let response = app
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/eval/jobs/{job_id}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        resume.notify_one();

        worker.await.unwrap();

        let jobs = state.eval_jobs.read().unwrap();
        let job = jobs.get(&job_id).expect("job should stay tracked");
        assert_eq!(
            job.state,
            EvalJobState::Cancelled,
            "worker must not clobber Cancelled back to Completed"
        );
        assert_eq!(job.finished_runs.len(), 1, "partial run should be recorded");
        let outcomes = job.finished_runs[0].outcomes.len();
        assert!(
            outcomes >= 1 && outcomes < total_examples,
            "executor should stop early, got {outcomes}/{total_examples} outcomes"
        );
        assert!(job.finished_at_iso.is_some(), "terminal timestamps stamped");
        assert!(
            job.cancel_flag.is_none(),
            "handle cleared at terminal state"
        );
    }
}
