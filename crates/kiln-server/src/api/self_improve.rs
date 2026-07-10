//! §10.6 self-distillation engine — "the centerpiece" of the grand
//! plan's agentic deployment story.
//!
//! Three endpoints:
//!
//! - `POST /v1/agent/judge_distill` — §10.6.1. Distil a small judge
//!   LoRA from the 27B teacher's multi-axis scoring of (turn,
//!   context) pairs collected via the §10.3 agent traces.
//!   One-time investment.
//!
//! - `POST /v1/agent/self_improve` — §10.6.2. The perpetual loop.
//!   Score the week's rollouts with the local judge LoRA, run GRPO
//!   with judge-derived advantages, optional CRISP terseness pass
//!   on top of successful trajectories (§10.6.4). Stable-OPD
//!   safeguards active.
//!
//! - `POST /v1/agent/judge_drift_check` — §10.6.3. Sample a slice
//!   (~50 trajectories), re-score with the 27B teacher, compare
//!   judge agreement. When agreement < 80% on contested cases, the
//!   trainer auto-triggers judge re-distillation.
//!
//! Per §10.6.6: "the §3.5 Knowledge Pump distils a static slice of
//! 27B into a static LoRA. The self-distillation engine is dynamic.
//! It captures the user's actual workflow, refreshes against the
//! latest 27B periodically, and compounds week over week."

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use kiln_train::{DistillPumpMode, DistillPumpRequest, OpdConfig, OpdRequest, TrainingResponse};
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;

use crate::error::ApiError;
use crate::state::{AppState, TrainingJobInfo, TrainingJobType};
use crate::training_queue::{QueueEntry, QueuedJob};
use kiln_train::TrainingState;

/// §10.6.1 turn-judge distillation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeDistillRequest {
    /// Output adapter name. Default `judge-pi-v1`.
    #[serde(default = "default_judge_name")]
    pub name: String,
    /// Teacher alias (typically Qwen3.6-27B).
    #[serde(default = "default_judge_teacher")]
    pub teacher: String,
    /// Whether to include public pi-share-hf sessions (default
    /// `false` — privacy default).
    #[serde(default)]
    pub include_pi_share: bool,
    /// Per-job OPD config; rank defaults to 16 per §10.6.1 (small
    /// judges work — "judging is easier than generating").
    #[serde(default = "default_judge_config")]
    pub config: OpdConfig,
    /// §8.7 promotion gate for the judge adapter (same contract as
    /// SelfImproveRequest.post_eval).
    #[serde(default)]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

fn default_judge_name() -> String {
    "judge-pi-v1".to_string()
}
fn default_judge_teacher() -> String {
    "qwen3.6-27b@local".to_string()
}
fn default_judge_config() -> OpdConfig {
    let mut c = OpdConfig::default();
    c.lora_rank = 16;
    c
}

#[derive(Debug, Serialize)]
struct JudgeDistillResponse {
    job_id: String,
    state: TrainingState,
    message: String,
}

async fn judge_distill(
    State(state): State<AppState>,
    Json(req): Json<JudgeDistillRequest>,
) -> Result<Json<JudgeDistillResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    if req.teacher.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "judge_distill: `teacher` alias must be non-empty".to_string(),
        ));
    }
    super::adapters::validate_adapter_name(&req.name)?;
    // The pump resolves the teacher only when the worker dequeues the job
    // (training_queue), so an unknown alias used to enqueue a job that was
    // guaranteed to fail later. Fail fast with the remediation instead.
    require_registered_teacher(
        &state,
        &req.teacher,
        format!(
            "judge_distill: teacher alias '{}' is not registered",
            req.teacher
        ),
    )?;
    // §10.6.1: the judge corpus is (turn, context) pairs from the user's
    // OWN indexed pi sessions — resolved here, at submission (and before
    // the queue/mock gates, so data problems surface with their
    // remediation first). Before this, the corpus silently fell through
    // to generic seed prompts and reported success.
    let (pump_req, num_pairs) = build_judge_pump_request(&state.adapter_dir, &req)?;
    let mut queued = QueuedJob::DistillPump(pump_req);
    let top_k_adjustment = super::training::normalize_queued_opd_top_k(&state, &mut queued)?;
    super::training::validate_post_eval_suite(&state, req.post_eval.as_ref())?;
    super::training::enforce_queue_caps(&state)?;

    let job_id = uuid::Uuid::new_v4().to_string();
    super::training::admit_training_jobs(
        &state,
        vec![prepare_agent_job(&job_id, &req.name, queued)],
    )?;
    let top_k_note = top_k_adjustment.map_or_else(String::new, |(requested, effective)| {
        format!(" Effective top_k was resolved from {requested} to {effective}.")
    });
    Ok(Json(JudgeDistillResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!(
            "Queued judge distillation for '{}' on {num_pairs} (turn, context) pairs \
             from your indexed pi sessions. Per §10.6.1 this is the one-time \
             investment — pay once, judge approximates 27B at <1% inference cost.{top_k_note}",
            req.name,
        ),
    }))
}

/// §10.6.2 self-improve loop request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfImproveRequest {
    /// Agent adapter to improve. Default `pi-coder-current`.
    #[serde(default = "default_agent_name")]
    pub agent: String,
    /// Judge LoRA alias to use for scoring rollouts.
    #[serde(default = "default_judge_name")]
    pub judge: String,
    /// Whether to engage the §10.6.4 CRISP terseness pass on top of
    /// successful rollouts. Default `true` (§6 pit-of-success).
    #[serde(default = "default_crisp_enabled")]
    pub crisp: bool,
    /// Per-job OPD config.
    #[serde(default)]
    pub config: OpdConfig,
    /// §8.7 promotion gate for the produced adapter. When set with
    /// `min_accuracy`, auto-load is DEFERRED until the eval passes and a
    /// failing adapter is demoted to `<name>.failed` — the same contract
    /// explicit training requests get. Without it the flywheel's own
    /// products would auto-load unmeasured.
    #[serde(default)]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

impl Default for SelfImproveRequest {
    /// The same defaults an empty POST body gets (serde field defaults) —
    /// the scheduler's no-config case must match `{}` exactly.
    fn default() -> Self {
        serde_json::from_str("{}").expect("empty SelfImproveRequest parses")
    }
}

fn default_agent_name() -> String {
    "pi-coder-current".to_string()
}
fn default_crisp_enabled() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct SelfImproveResponse {
    pub job_ids: Vec<String>,
    pub state: TrainingState,
    pub message: String,
}

async fn self_improve(
    State(state): State<AppState>,
    Json(req): Json<SelfImproveRequest>,
) -> Result<Json<SelfImproveResponse>, ApiError> {
    submit_self_improve(&state, req).map(Json)
}

/// The §10.6.2 submission body, callable without an HTTP request — the
/// `[agent] self_improve_interval_hours` scheduler drives the same code
/// path the POST handler does, so validation, teacher gating, and queue
/// caps apply identically to scheduled rounds.
pub fn submit_self_improve(
    state: &AppState,
    req: SelfImproveRequest,
) -> Result<SelfImproveResponse, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    if req.agent.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "self_improve: `agent` adapter must be non-empty".to_string(),
        ));
    }
    // The derived `{agent}-improve` / `{agent}-crisp` output names are safe
    // iff the agent name itself is a safe single segment.
    super::adapters::validate_adapter_name(&req.agent)?;
    // Both phases use the judge as their distillation teacher, resolved only
    // when the worker dequeues the job — so an unregistered judge used to
    // enqueue jobs that were guaranteed to fail later.
    require_registered_teacher(
        state,
        &req.judge,
        format!(
            "self_improve: judge '{}' must be registered as a teacher alias \
             (it scores rollouts and serves as the distillation teacher)",
            req.judge
        ),
    )?;
    // Resolve the week's tasks from the §10.3 trace index NOW — an empty
    // or stale index fails here with the remediation, not at worker
    // dequeue hours later, and before the queue/mock gates so data
    // problems surface first. (The worker re-resolves the same selector
    // at run time so sessions captured between submission and dequeue
    // are included.)
    let (opd_phase, crisp_pump, num_tasks) = build_self_improve_jobs(&state.adapter_dir, &req)?;
    let mut opd_phase = QueuedJob::Opd(opd_phase);
    let opd_top_k_adjustment = super::training::normalize_queued_opd_top_k(state, &mut opd_phase)?;
    let mut crisp_pump = crisp_pump.map(QueuedJob::DistillPump);
    let crisp_top_k_adjustment = match crisp_pump.as_mut() {
        Some(job) => super::training::normalize_queued_opd_top_k(state, job)?,
        None => None,
    };
    let additional_jobs = 1 + usize::from(crisp_pump.is_some());
    super::training::enforce_queue_capacity_for(state, additional_jobs)?;

    // §10.6.2: score with judge → GRPO → CRISP pass. Each phase
    // queues independently. The trainer body (#31) wires the
    // judge-scored advantages into the GRPO step.
    let mut job_ids = Vec::new();
    let mut pending = Vec::with_capacity(additional_jobs);

    let opd_job = uuid::Uuid::new_v4().to_string();
    pending.push(prepare_agent_job(
        &opd_job,
        &format!("{}-improve", req.agent),
        opd_phase,
    ));
    job_ids.push(opd_job);

    if let Some(crisp_pump) = crisp_pump {
        let crisp_job = uuid::Uuid::new_v4().to_string();
        pending.push(prepare_agent_job(
            &crisp_job,
            &format!("{}-crisp", req.agent),
            crisp_pump,
        ));
        job_ids.push(crisp_job);
    }
    super::training::admit_training_jobs(state, pending)?;

    let mut top_k_notes = Vec::new();
    if let Some((requested, effective)) = opd_top_k_adjustment {
        top_k_notes.push(format!("OPD {requested}->{effective}"));
    }
    if let Some((requested, effective)) = crisp_top_k_adjustment {
        top_k_notes.push(format!("CRISP {requested}->{effective}"));
    }
    let top_k_note = (!top_k_notes.is_empty())
        .then(|| format!(" Effective top_k: {}.", top_k_notes.join(", ")))
        .unwrap_or_default();

    Ok(SelfImproveResponse {
        job_ids,
        state: TrainingState::Queued,
        message: format!(
            "§10.6.2 self_improve queued on {num_tasks} task(s) from this week's pi \
             sessions: agent={}, judge={}, crisp={}. Phase 1 = judge-scored GRPO; \
             Phase 2 = CRISP terseness pass.{top_k_note}",
            req.agent, req.judge, req.crisp,
        ),
    })
}

/// §10.6.3 judge drift check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeDriftCheckRequest {
    /// Judge LoRA alias.
    #[serde(default = "default_judge_name")]
    pub judge: String,
    /// 27B teacher alias to compare against.
    #[serde(default = "default_judge_teacher")]
    pub teacher: String,
    /// Sample size — number of trajectories to re-score with the
    /// teacher. Default 50 per §10.6.3.
    #[serde(default = "default_drift_sample_size")]
    pub sample_size: usize,
    /// Agreement threshold below which auto-refresh fires.
    /// Default 0.80 per §10.6.3.
    #[serde(default = "default_drift_threshold")]
    pub agreement_threshold: f64,
}

fn default_drift_sample_size() -> usize {
    50
}
fn default_drift_threshold() -> f64 {
    0.80
}

/// The actual scoring + comparison is a real GPU run that lands with #31.
/// Until then the endpoint validates its inputs (so callers wire up the
/// judge/teacher/thresholds correctly today) and returns an honest 501 —
/// never a fake "no drift" success.
async fn judge_drift_check(
    State(state): State<AppState>,
    Json(req): Json<JudgeDriftCheckRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    super::adapters::validate_adapter_name(&req.judge)?;
    let judge_dir = state.adapter_dir.join(&req.judge);
    if !judge_dir.is_dir() {
        return Err(ApiError::adapter_not_found(judge_dir.display()));
    }
    require_registered_teacher(
        &state,
        &req.teacher,
        format!(
            "judge_drift_check: teacher alias '{}' is not registered",
            req.teacher
        ),
    )?;
    if req.sample_size < 1 {
        return Err(ApiError::training_invalid_request(
            "judge_drift_check: `sample_size` must be >= 1".to_string(),
        ));
    }
    if !(req.agreement_threshold > 0.0 && req.agreement_threshold <= 1.0) {
        return Err(ApiError::training_invalid_request(format!(
            "judge_drift_check: `agreement_threshold` must be in (0.0, 1.0], got {}",
            req.agreement_threshold
        )));
    }
    Err(ApiError::drift_check_not_implemented())
}

/// §10.6.1 corpus + pump construction, factored out of the handler so the
/// wiring is unit-testable without a real (non-mock) backend. Returns the
/// pump request and the corpus size.
fn build_judge_pump_request(
    adapter_dir: &std::path::Path,
    req: &JudgeDistillRequest,
) -> Result<(DistillPumpRequest, usize), ApiError> {
    super::agent_traces::ensure_agent_trace_index(adapter_dir);
    let judge_prompts = crate::dataset_resolve::resolve_agent_trace_prompts(
        adapter_dir,
        "agent_traces:judge_turns",
        crate::recent_requests::now_unix_ms() as i64,
    )
    .map_err(|e| ApiError::training_invalid_request(format!("judge_distill: {e}")))?;
    let num_pairs = judge_prompts.len();
    let pump_req = DistillPumpRequest {
        name: req.name.clone(),
        teacher: req.teacher.clone(),
        mode: DistillPumpMode::Examples {
            examples: judge_prompts,
        },
        rank: Some(req.config.lora_rank),
        rollout_budget: 50_000,
        use_cache: true,
        config: req.config.clone(),
        post_eval: req.post_eval.clone(),
    };
    Ok((pump_req, num_pairs))
}

/// §10.6.2 phase construction: the weekly on-policy OPD phase (validated
/// here, re-resolved by the worker at dequeue) and the optional §10.6.4
/// CRISP pump on resolved conciseness prompts. Returns the task count for
/// the response message.
#[allow(clippy::type_complexity)]
fn build_self_improve_jobs(
    adapter_dir: &std::path::Path,
    req: &SelfImproveRequest,
) -> Result<(OpdRequest, Option<DistillPumpRequest>, usize), ApiError> {
    let now_ms = crate::recent_requests::now_unix_ms() as i64;
    // Canonical onboarding: build the trace index from the default pi
    // sessions dir when it doesn't exist yet, instead of hard-failing
    // and demanding a manual POST /v1/agent/traces/discover.
    super::agent_traces::ensure_agent_trace_index(adapter_dir);
    let weekly = crate::dataset_resolve::resolve_agent_trace_prompts(
        adapter_dir,
        "agent_traces:weekly",
        now_ms,
    )
    .map_err(|e| ApiError::training_invalid_request(format!("self_improve: {e}")))?;
    let num_tasks = weekly.len();

    // Warm start: when the agent's adapter already exists, this week's
    // training CONTINUES from it instead of restarting from the base
    // model. Without this, every self_improve cycle threw away the
    // previous cycles' learning — the flywheel spun but nothing
    // compounded ("your model gets better every time you use it"
    // requires the weeks to stack). An explicit base_adapter in the
    // request still wins; a missing adapter dir trains from base (first
    // cycle).
    let warm_start_base = |config: &kiln_train::OpdConfig| -> Option<String> {
        if config.base_adapter.is_some() {
            return config.base_adapter.clone();
        }
        // The round's OUTPUT is "{agent}-improve" (register_agent_job
        // below) — so that is what the NEXT round must continue from, or
        // every scheduled cycle silently retrains from base and
        // overwrites last week's product (round-4 discovery caught the
        // original checking only the bare agent name, which nothing
        // creates). Fall back to the bare agent dir for operators who
        // maintain one by hand.
        let improve_name = format!("{}-improve", req.agent);
        if adapter_dir.join(&improve_name).is_dir() {
            return Some(improve_name);
        }
        adapter_dir
            .join(&req.agent)
            .is_dir()
            .then(|| req.agent.clone())
    };

    // Phase 1: the student re-rolls the week's pi tasks on-policy with
    // the judge as the scoring teacher. The selector resolves in the
    // worker via the same path that just validated it above.
    let mut opd_config = req.config.clone();
    opd_config.base_adapter = warm_start_base(&req.config);
    let opd_phase = OpdRequest {
        prompts: Vec::new(),
        dataset_path: Some("agent_traces:weekly".to_string()),
        teacher: req.judge.clone(),
        config: opd_config,
        post_eval: req.post_eval.clone(),
    };

    // Phase 2: the same successful sessions re-prompted under
    // conciseness pressure (§10.6.4).
    let crisp_pump = if req.crisp {
        let crisp_prompts = crate::dataset_resolve::resolve_agent_trace_prompts(
            adapter_dir,
            "agent_traces:crisp",
            now_ms,
        )
        .map_err(|e| {
            ApiError::training_invalid_request(format!("self_improve (crisp phase): {e}"))
        })?;
        let mut crisp_config = req.config.clone();
        crisp_config.output_name = Some(format!("{}-crisp", req.agent));
        crisp_config.base_adapter = warm_start_base(&req.config);
        Some(DistillPumpRequest {
            name: format!("{}-crisp", req.agent),
            teacher: req.judge.clone(),
            mode: DistillPumpMode::Examples {
                examples: crisp_prompts,
            },
            rank: None,
            rollout_budget: 10_000,
            use_cache: true,
            config: crisp_config,
            post_eval: req.post_eval.clone(),
        })
    } else {
        None
    };

    Ok((opd_phase, crisp_pump, num_tasks))
}

use super::teachers::require_registered_teacher;

fn prepare_agent_job(
    job_id: &str,
    adapter_name: &str,
    job: QueuedJob,
) -> (TrainingJobInfo, QueueEntry) {
    let info = TrainingJobInfo {
        job_id: job_id.to_string(),
        adapter_name: adapter_name.to_string(),
        job_type: TrainingJobType::Opd,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: crate::recent_requests::now_unix_ms(),
        auto_load: true,
        consumed_correction_ids: Vec::new(),
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    (
        info,
        QueueEntry {
            job_id: job_id.to_string(),
            reserved_bytes: 0,
            job,
        },
    )
}

// Avoid unused warnings on the Response type alias.
#[allow(dead_code)]
type _Response = TrainingResponse;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/agent/judge_distill", post(judge_distill))
        .route("/v1/agent/self_improve", post(self_improve))
        .route("/v1/agent/judge_drift_check", post(judge_drift_check))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judge_distill_request_defaults_match_section_10_6_1() {
        let json = r#"{}"#;
        let req: JudgeDistillRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "judge-pi-v1");
        assert_eq!(req.teacher, "qwen3.6-27b@local");
        assert!(!req.include_pi_share);
        // §10.6.1: rank=16 default
        assert_eq!(req.config.lora_rank, 16);
    }

    #[test]
    fn self_improve_request_defaults() {
        let req: SelfImproveRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(req.agent, "pi-coder-current");
        assert_eq!(req.judge, "judge-pi-v1");
        assert!(req.crisp, "§10.6.4 CRISP pass is on by default");
    }

    /// The scheduler's no-config request must equal an empty POST body,
    /// and an [agent.self_improve] table must deserialize through the
    /// same shape the endpoint accepts.
    #[test]
    fn scheduler_request_defaults_match_empty_post_body() {
        let from_default = SelfImproveRequest::default();
        let from_empty: SelfImproveRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(from_default.agent, from_empty.agent);
        assert_eq!(from_default.judge, from_empty.judge);
        assert_eq!(from_default.crisp, from_empty.crisp);
        assert!(from_default.post_eval.is_none());

        let toml_like = serde_json::json!({
            "agent": "pi-coder-current",
            "judge": "judge-pi-v1",
            "post_eval": {"suite": "qwen3.5-agentic-core", "min_accuracy": 0.7}
        });
        let req: SelfImproveRequest = serde_json::from_value(toml_like).unwrap();
        assert_eq!(req.post_eval.unwrap().min_accuracy, Some(0.7));
    }

    #[test]
    fn self_improve_can_disable_crisp() {
        let req: SelfImproveRequest = serde_json::from_str(r#"{"crisp": false}"#).unwrap();
        assert!(!req.crisp);
    }

    #[test]
    fn judge_drift_check_defaults_match_section_10_6_3() {
        let req: JudgeDriftCheckRequest = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(req.judge, "judge-pi-v1");
        assert_eq!(req.teacher, "qwen3.6-27b@local");
        assert_eq!(req.sample_size, 50);
        assert!((req.agreement_threshold - 0.80).abs() < 1e-9);
    }

    use kiln_train::ChatMessage;
    use kiln_train::trajectory::{TurnKind, TurnSegment};

    fn seeded_trace_index(dir: &std::path::Path) {
        let now = chrono::Utc::now().to_rfc3339();
        let trace = crate::api::agent_traces::AgentTrace {
            id: "session-a".into(),
            working_dir: "/home/user/proj".into(),
            num_turns: 4,
            num_tool_calls: 1,
            outcome: crate::api::agent_traces::TraceOutcome {
                ended_with_exit_0: Some(true),
                user_edited_agent_files: Vec::new(),
                has_followup_attempt: Some(false),
            },
            first_event_at: Some(now.clone()),
            last_event_at: Some(now),
            forked: false,
            parent_id: None,
            tool_manifest_sha: None,
            prompt_messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: "You are pi.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Fix the flaky test".into(),
                },
            ],
            trajectory: vec![
                TurnSegment {
                    role: "assistant".into(),
                    content: "Reading the test.".into(),
                    kind: TurnKind::Action,
                    tool_call_id: None,
                    warning_prefix_len: None,
                },
                TurnSegment {
                    role: "tool".into(),
                    content: "FAILED tests/flaky.rs".into(),
                    kind: TurnKind::Observation,
                    tool_call_id: None,
                    warning_prefix_len: None,
                },
                TurnSegment {
                    role: "assistant".into(),
                    content: "Pinning the seed.".into(),
                    kind: TurnKind::Action,
                    tool_call_id: None,
                    warning_prefix_len: None,
                },
            ],
        };
        let mut map = std::collections::BTreeMap::new();
        map.insert(trace.id.clone(), trace);
        std::fs::write(
            dir.join("agent_traces.json"),
            serde_json::to_vec_pretty(&map).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn judge_pump_carries_turn_context_pairs_from_the_index() {
        let dir = tempfile::tempdir().unwrap();
        seeded_trace_index(dir.path());
        let req: JudgeDistillRequest = serde_json::from_str("{}").unwrap();

        let (pump, num_pairs) = build_judge_pump_request(dir.path(), &req).unwrap();

        assert_eq!(num_pairs, 2, "one judge prompt per assistant action");
        let DistillPumpMode::Examples { examples } = &pump.mode else {
            panic!("judge pump must carry resolved Examples");
        };
        assert!(examples[0].messages[0].content.contains("tool_correctness"));
        assert!(
            examples[1].messages[1]
                .content
                .contains("FAILED tests/flaky.rs"),
            "later turns see the observations that preceded them"
        );
    }

    #[test]
    fn judge_pump_without_index_carries_discover_remediation() {
        let dir = tempfile::tempdir().unwrap();
        // Point auto-discovery away from the developer's real ~/.pi
        // sessions so the missing-index path actually exercises. Restored
        // after the call; tests in this mod run on separate adapter dirs.
        let prev = std::env::var("KILN_PI_SESSIONS_DIR").ok();
        unsafe { std::env::set_var("KILN_PI_SESSIONS_DIR", "/nonexistent/pi/sessions") };
        let req: JudgeDistillRequest = serde_json::from_str("{}").unwrap();
        let result = build_judge_pump_request(dir.path(), &req);
        match prev {
            Some(v) => unsafe { std::env::set_var("KILN_PI_SESSIONS_DIR", v) },
            None => unsafe { std::env::remove_var("KILN_PI_SESSIONS_DIR") },
        }
        let msg = format!("{:?}", result.unwrap_err());
        assert!(msg.contains("/v1/agent/traces/discover"), "{msg}");
    }

    #[test]
    fn self_improve_jobs_resolve_weekly_tasks_and_crisp_examples() {
        let dir = tempfile::tempdir().unwrap();
        seeded_trace_index(dir.path());
        let req: SelfImproveRequest = serde_json::from_str("{}").unwrap();

        let (opd, crisp, num_tasks) = build_self_improve_jobs(dir.path(), &req).unwrap();

        assert_eq!(num_tasks, 1);
        assert_eq!(opd.dataset_path.as_deref(), Some("agent_traces:weekly"));
        assert_eq!(opd.teacher, "judge-pi-v1");
        let crisp = crisp.expect("crisp defaults on");
        let DistillPumpMode::Examples { examples } = &crisp.mode else {
            panic!("crisp pump must carry resolved Examples");
        };
        assert_eq!(examples.len(), 1);
        assert!(
            examples[0].messages[0]
                .content
                .contains("maximally concise"),
            "conciseness pressure folded into the system turn"
        );
        assert!(
            examples[0].messages[1]
                .content
                .contains("Fix the flaky test")
        );
    }

    /// The flywheel must COMPOUND: when the agent's adapter already
    /// exists, both phases continue from it; on the first cycle (no
    /// adapter yet) they train from base; an explicit base_adapter in
    /// the request always wins.
    #[test]
    fn self_improve_warm_starts_from_existing_agent_adapter() {
        let dir = tempfile::tempdir().unwrap();
        seeded_trace_index(dir.path());
        let req: SelfImproveRequest = serde_json::from_str("{}").unwrap();

        // First cycle: no adapter dir → base model.
        let (opd, crisp, _) = build_self_improve_jobs(dir.path(), &req).unwrap();
        assert!(opd.config.base_adapter.is_none());
        assert!(crisp.unwrap().config.base_adapter.is_none());

        // Later cycles: the agent adapter exists → warm start.
        std::fs::create_dir(dir.path().join("pi-coder-current")).unwrap();
        let (opd, crisp, _) = build_self_improve_jobs(dir.path(), &req).unwrap();
        assert_eq!(opd.config.base_adapter.as_deref(), Some("pi-coder-current"));
        assert_eq!(
            crisp.unwrap().config.base_adapter.as_deref(),
            Some("pi-coder-current")
        );

        // THE COMPOUNDING CASE (round-4 discovery): each round's output
        // is "{agent}-improve" — when it exists, the next round must
        // continue from IT, not the bare agent dir (and never from
        // base). Without this, the weekly scheduler retrained from base
        // and overwrote last week's product forever.
        std::fs::create_dir(dir.path().join("pi-coder-current-improve")).unwrap();
        let (opd, crisp, _) = build_self_improve_jobs(dir.path(), &req).unwrap();
        assert_eq!(
            opd.config.base_adapter.as_deref(),
            Some("pi-coder-current-improve")
        );
        assert_eq!(
            crisp.unwrap().config.base_adapter.as_deref(),
            Some("pi-coder-current-improve")
        );

        // Explicit base_adapter wins over the warm-start default.
        let req: SelfImproveRequest =
            serde_json::from_str(r#"{"config": {"base_adapter": "my-pinned-base"}}"#).unwrap();
        let (opd, _, _) = build_self_improve_jobs(dir.path(), &req).unwrap();
        assert_eq!(opd.config.base_adapter.as_deref(), Some("my-pinned-base"));
    }

    /// The flywheel's own products honor the §8.7 gate: a post_eval on
    /// the request rides into every produced job.
    #[test]
    fn self_improve_post_eval_gate_rides_into_all_phases() {
        let dir = tempfile::tempdir().unwrap();
        seeded_trace_index(dir.path());
        let req: SelfImproveRequest = serde_json::from_str(
            r#"{"post_eval": {"suite": "qwen3.5-agentic-core", "min_accuracy": 0.7}}"#,
        )
        .unwrap();
        let (opd, crisp, _) = build_self_improve_jobs(dir.path(), &req).unwrap();
        assert_eq!(
            opd.post_eval.as_ref().unwrap().suite,
            "qwen3.5-agentic-core"
        );
        assert_eq!(opd.post_eval.as_ref().unwrap().min_accuracy, Some(0.7));
        assert_eq!(
            crisp.unwrap().post_eval.as_ref().unwrap().suite,
            "qwen3.5-agentic-core"
        );
    }

    #[test]
    fn self_improve_jobs_skip_crisp_when_disabled() {
        let dir = tempfile::tempdir().unwrap();
        seeded_trace_index(dir.path());
        let req: SelfImproveRequest = serde_json::from_str(r#"{"crisp": false}"#).unwrap();
        let (_, crisp, _) = build_self_improve_jobs(dir.path(), &req).unwrap();
        assert!(crisp.is_none());
    }
}
