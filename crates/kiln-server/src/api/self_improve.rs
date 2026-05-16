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

    // §10.6.1 internally is a `distill_pump` against the judge_traces
    // canonical corpus. We construct the pump request server-side so
    // the agentic surface stays minimal.
    let pump_req = DistillPumpRequest {
        name: req.name.clone(),
        teacher: req.teacher.clone(),
        mode: DistillPumpMode::Domain {
            domain: "judge_traces".to_string(),
        },
        rank: Some(req.config.lora_rank),
        rollout_budget: 50_000,
        use_cache: true,
        config: req.config.clone(),
        post_eval: None,
    };

    let job_id = uuid::Uuid::new_v4().to_string();
    register_agent_job(&state, &job_id, &req.name, QueuedJob::DistillPump(pump_req));
    Ok(Json(JudgeDistillResponse {
        job_id,
        state: TrainingState::Queued,
        message: format!(
            "Queued judge distillation for '{}'. Per §10.6.1 this is the one-time \
             investment — pay once, judge approximates 27B at <1% inference cost.",
            req.name
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
}

fn default_agent_name() -> String {
    "pi-coder-current".to_string()
}
fn default_crisp_enabled() -> bool {
    true
}

#[derive(Debug, Serialize)]
struct SelfImproveResponse {
    job_ids: Vec<String>,
    state: TrainingState,
    message: String,
}

async fn self_improve(
    State(state): State<AppState>,
    Json(req): Json<SelfImproveRequest>,
) -> Result<Json<SelfImproveResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    if req.agent.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "self_improve: `agent` adapter must be non-empty".to_string(),
        ));
    }

    // §10.6.2: score with judge → GRPO → CRISP pass. Each phase
    // queues independently. The trainer body (#31) wires the
    // judge-scored advantages into the GRPO step.
    let mut job_ids = Vec::new();

    // Phase 1: agent GRPO step (queued as OPD with the judge as the
    // teacher — the loss kernel scores per-token KL against the
    // judge's distribution).
    let opd_phase = OpdRequest {
        prompts: Vec::new(),
        dataset_path: Some("agent_traces:weekly".to_string()),
        teacher: req.judge.clone(),
        config: req.config.clone(),
        post_eval: None,
    };
    let opd_job = uuid::Uuid::new_v4().to_string();
    register_agent_job(
        &state,
        &opd_job,
        &format!("{}-improve", req.agent),
        QueuedJob::Opd(opd_phase),
    );
    job_ids.push(opd_job);

    // Phase 2: optional CRISP terseness pass.
    if req.crisp {
        let mut crisp_config = req.config.clone();
        crisp_config.output_name = Some(format!("{}-crisp", req.agent));
        let crisp_job = uuid::Uuid::new_v4().to_string();
        let crisp_pump = DistillPumpRequest {
            name: format!("{}-crisp", req.agent),
            teacher: req.judge.clone(),
            mode: DistillPumpMode::Domain {
                domain: "crisp_terseness".to_string(),
            },
            rank: None,
            rollout_budget: 10_000,
            use_cache: true,
            config: crisp_config,
            post_eval: None,
        };
        register_agent_job(
            &state,
            &crisp_job,
            &format!("{}-crisp", req.agent),
            QueuedJob::DistillPump(crisp_pump),
        );
        job_ids.push(crisp_job);
    }

    Ok(Json(SelfImproveResponse {
        job_ids,
        state: TrainingState::Queued,
        message: format!(
            "§10.6.2 self_improve queued: agent={}, judge={}, crisp={}. \
             Phase 1 = judge-scored GRPO; Phase 2 = CRISP terseness pass.",
            req.agent, req.judge, req.crisp
        ),
    }))
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

#[derive(Debug, Serialize)]
struct JudgeDriftCheckResponse {
    /// Whether a refresh was triggered as a result of this check.
    refresh_triggered: bool,
    /// Refresh job id, when triggered.
    refresh_job_id: Option<String>,
    /// Observed agreement rate (None when the check couldn't run —
    /// e.g. judge or teacher not registered).
    observed_agreement: Option<f64>,
    message: String,
}

async fn judge_drift_check(
    State(state): State<AppState>,
    Json(req): Json<JudgeDriftCheckRequest>,
) -> Result<Json<JudgeDriftCheckResponse>, ApiError> {
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    // The actual scoring + comparison is a real GPU run that lands
    // with #31. For now the endpoint shape is established and the
    // refresh-trigger path is wired so the dashboard can poll on
    // the same schedule.
    let _ = req;
    let _ = state;
    Ok(Json(JudgeDriftCheckResponse {
        refresh_triggered: false,
        refresh_job_id: None,
        observed_agreement: None,
        message: "§10.6.3 drift check shape established. Full scoring + comparison \
                  + auto-refresh trigger wires alongside the trainer body (#31)."
            .to_string(),
    }))
}

fn register_agent_job(
    state: &AppState,
    job_id: &str,
    adapter_name: &str,
    job: QueuedJob,
) {
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
        finished_at: None,
        finished_unix_ms: None,
        linked_eval_job_ids: Vec::new(),
        loss_history: Vec::new(),
    };
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(job_id.to_string(), info);
    state.training_queue.lock().unwrap().push(QueueEntry {
        job_id: job_id.to_string(),
        job,
    });
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

    #[test]
    fn self_improve_can_disable_crisp() {
        let req: SelfImproveRequest =
            serde_json::from_str(r#"{"crisp": false}"#).unwrap();
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
}
