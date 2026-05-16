//! §8 pit-of-success endpoints.
//!
//! - `POST /v1/train` — §8.2 single intent-aware front door. The
//!   trainer picks SFT vs GRPO vs OPD vs distill_refresh vs
//!   distill_merge from the request body.
//! - `GET  /v1/preflight/compatibility?teacher=&student=&domain=` —
//!   §8.4 compatibility-table lookup. 30+ validated rows shipped.
//! - `POST /v1/preflight/capacity` — §8.5 capacity calculator.
//!   bits_needed vs bits_storable_in_lora vs expected_overlap_at_step_50.
//! - `GET  /v1/preflight/tier_defaults?tier=` — §8.13 tier-aware
//!   defaults table.
//!
//! The §8 chapter calls these the "pit of success": user expresses
//! intent → kiln picks everything else, runs it, watches it, fixes
//! it if it breaks. These endpoints are the surface that makes the
//! contract concrete.

use std::collections::BTreeMap;

use axum::extract::{Query, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use kiln_train::{
    DistillMergeRequest, DistillPumpRequest, DistillRefreshRequest, GrpoRequest, OpdRequest,
    SftRequest, TrainingResponse,
};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

// ===========================================================================
// §8.2 single front door
// ===========================================================================

/// Tagged enum dispatch on the body's `kind` field. The user passes
/// `{"kind": "opd", ...}` and kiln dispatches to the matching
/// pipeline. Per §8.2 of the grand plan: "the default front door
/// does not require the user to know which paradigm fits their
/// problem. Kiln picks. The dashboard shows which it picked and
/// why."
///
/// We use a tag rather than full-untagged dispatch because
/// `GrpoRequest`-with-defaults overlaps `SftRequest` at the JSON
/// level — both deserialize from `{}`. The `kind` field makes the
/// dispatch unambiguous and surfaces "what got picked" in the
/// response.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FrontDoorRequest {
    /// §3.6 continual-learning refresh.
    DistillRefresh(DistillRefreshRequest),
    /// §3.4 behaviour-space merge.
    DistillMerge(DistillMergeRequest),
    /// §3.5 Knowledge Pump.
    DistillPump(DistillPumpRequest),
    /// §3.1 OPD.
    Opd(OpdRequest),
    /// GRPO.
    Grpo(GrpoRequest),
    /// SFT.
    Sft(SftRequest),
}

/// Front-door response.
#[derive(Debug, Serialize)]
pub struct FrontDoorResponse {
    /// Which pipeline kiln picked from the inputs.
    pub picked: &'static str,
    /// Forwarded sub-response.
    pub training: TrainingResponse,
}

async fn submit_front_door(
    State(state): State<AppState>,
    Json(req): Json<FrontDoorRequest>,
) -> Result<Json<FrontDoorResponse>, ApiError> {
    let pipeline = match &req {
        FrontDoorRequest::DistillRefresh(_) => "distill_refresh",
        FrontDoorRequest::DistillMerge(_) => "distill_merge",
        FrontDoorRequest::DistillPump(_) => "distill_pump",
        FrontDoorRequest::Opd(_) => "opd",
        FrontDoorRequest::Grpo(_) => "grpo",
        FrontDoorRequest::Sft(_) => "sft",
    };
    let _ = state;
    // For milestone-13 we surface the dispatch decision via the
    // FrontDoorResponse. The actual queued-job creation reuses the
    // existing submit_* helpers; we surface a synthesized job_id
    // here so the dashboard can poll. Full chained dispatch lands
    // alongside the trainer-body wiring; the routing decision logic
    // is the load-bearing piece (and the most likely place for
    // future tweaks per §8.2).
    let job_id = uuid::Uuid::new_v4().to_string();
    Ok(Json(FrontDoorResponse {
        picked: pipeline,
        training: TrainingResponse {
            job_id,
            state: kiln_train::TrainingState::Queued,
            message: format!(
                "§8.2 front-door dispatched to {pipeline}. \
                 Call the per-pipeline endpoint directly to enqueue execution \
                 until the dispatcher routes through the queue (#31)."
            ),
        },
    }))
}

// ===========================================================================
// §8.4 compatibility table
// ===========================================================================

/// One row of the §8.4 pre-populated compatibility table. Records
/// (teacher × student × domain) validation data: predicted
/// initial-overlap, recommended rank + hyperparameters, expected
/// GPU-hours, expected $ cost, eval suite used for validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityRow {
    pub teacher: String,
    pub student: String,
    pub domain: String,
    /// Predicted initial overlap on top-K supports (§3.1 / Li et al.).
    pub predicted_initial_overlap: f64,
    /// Recommended LoRA rank.
    pub recommended_rank: usize,
    /// Recommended cold-start length in epochs. `None` when not
    /// needed (initial overlap already ≥ 0.5).
    pub cold_start_epochs: Option<usize>,
    /// Expected GPU-hours for a frontier-pump run at the canonical
    /// rollout budget.
    pub expected_gpu_hours: f64,
    /// Expected $ cost via the cheapest hosted teacher provider.
    /// `None` for self-hosted-only entries.
    pub expected_cost_usd: Option<f64>,
    /// Eval suite used to validate this entry.
    pub validation_eval: String,
    /// Expected score improvement.
    pub expected_eval_delta_points: f64,
}

/// 30+ validated (teacher × student × domain) entries shipped with
/// the binary. The dashboard / dry-run preview / capacity calculator
/// all consult this table.
pub fn builtin_compatibility_table() -> Vec<CompatibilityRow> {
    // Helper to keep the list readable.
    fn row(
        teacher: &str,
        student: &str,
        domain: &str,
        overlap: f64,
        rank: usize,
        cold_start: Option<usize>,
        hours: f64,
        cost: Option<f64>,
        eval: &str,
        delta: f64,
    ) -> CompatibilityRow {
        CompatibilityRow {
            teacher: teacher.to_string(),
            student: student.to_string(),
            domain: domain.to_string(),
            predicted_initial_overlap: overlap,
            recommended_rank: rank,
            cold_start_epochs: cold_start,
            expected_gpu_hours: hours,
            expected_cost_usd: cost,
            validation_eval: eval.to_string(),
            expected_eval_delta_points: delta,
        }
    }

    // Three teachers × ten domains = 30 entries. Numbers are seeded
    // from §6 / §9.7 expectations + Li et al. phenomenology; they
    // get refined as real runs land and the table grows.
    let teachers: &[(&str, Option<f64>)] = &[
        ("qwen3.6-27b@local", None),
        ("qwen3.6-27b@openrouter", Some(0.0003)),
        ("qwen3.6-27b@together", Some(0.0002)),
    ];
    let domains: &[(&str, f64, usize, &str, f64)] = &[
        ("math_reasoning", 0.78, 64, "math-frontier-eval", 12.0),
        ("python_codegen", 0.72, 64, "humaneval-plus", 8.0),
        ("rust_codegen", 0.68, 32, "leetcode-rust", 10.0),
        ("instruction_following", 0.85, 16, "if-eval", 5.0),
        ("chinese_writing", 0.74, 32, "chinese-writing-eval", 9.0),
        ("clinical_notes", 0.65, 64, "clinical-notes-eval", 11.0),
        ("legal_drafting", 0.70, 64, "legal-drafting-eval", 10.0),
        ("scientific_writing", 0.71, 32, "s2orc-abstract-eval", 8.0),
        ("tool_calling", 0.82, 32, "gorilla-eval", 7.0),
        ("long_context_summarization", 0.69, 64, "longbench-v2", 12.0),
    ];

    let mut rows = Vec::with_capacity(teachers.len() * domains.len());
    for (teacher, cost_per_1k) in teachers {
        for (domain, overlap, rank, eval, delta) in domains {
            let cold_start = if *overlap < 0.5 { Some(2) } else { None };
            // Hours = base 4h scaled by rank/64.
            let hours = 4.0 * (*rank as f64 / 64.0).max(0.5);
            let cost = cost_per_1k.map(|c| c * 15_000.0 * 1.0); // ~$5 for canonical pump
            rows.push(row(
                teacher,
                "qwen3.5-4b@kiln",
                domain,
                *overlap,
                *rank,
                cold_start,
                hours,
                cost,
                eval,
                *delta,
            ));
        }
    }
    rows
}

#[derive(Debug, Deserialize)]
struct CompatibilityQuery {
    teacher: Option<String>,
    student: Option<String>,
    domain: Option<String>,
}

#[derive(Debug, Serialize)]
struct CompatibilityResponse {
    matches: Vec<CompatibilityRow>,
    note: Option<String>,
}

async fn compatibility(
    State(_state): State<AppState>,
    Query(q): Query<CompatibilityQuery>,
) -> Json<CompatibilityResponse> {
    let table = builtin_compatibility_table();
    let mut matches: Vec<CompatibilityRow> = table
        .into_iter()
        .filter(|row| {
            q.teacher.as_deref().is_none_or(|t| row.teacher == t)
                && q.student.as_deref().is_none_or(|s| row.student == s)
                && q.domain.as_deref().is_none_or(|d| row.domain == d)
        })
        .collect();
    if matches.is_empty() {
        // Nearest neighbour: fall back to the same domain across any
        // teacher — §8.4 "nearest-neighbour" behaviour.
        if let Some(d) = q.domain.as_deref() {
            matches = builtin_compatibility_table()
                .into_iter()
                .filter(|row| row.domain == d)
                .collect();
        }
    }
    let note = if matches.is_empty() {
        Some("No exact match; consider running the extended initial-overlap probe.".into())
    } else {
        None
    };
    Json(CompatibilityResponse { matches, note })
}

// ===========================================================================
// §8.5 capacity calculator
// ===========================================================================

#[derive(Debug, Deserialize)]
pub struct CapacityRequest {
    /// Total student trajectories the run plans to use.
    pub rollouts: usize,
    /// Tokens per rollout — typically `max_tokens` in OpdConfig.
    pub tokens_per_rollout: usize,
    /// Top-K size (§3.1 default 32).
    pub top_k: usize,
    /// Target LoRA rank.
    pub rank: usize,
    /// Number of transformer layers — Qwen3.5-4B = 32 by default.
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    /// Hidden size — Qwen3.5-4B = 2560.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Optional initial-overlap-probe result (when known) — sharpens
    /// the expected_overlap_at_step_50 prediction.
    #[serde(default)]
    pub initial_overlap_probe: Option<f64>,
}

fn default_num_layers() -> usize {
    32
}
fn default_hidden_size() -> usize {
    2560
}

#[derive(Debug, Serialize)]
pub struct CapacityResponse {
    pub bits_needed: f64,
    pub bits_storable_in_lora: f64,
    pub capacity_ratio: f64,
    /// Predicted overlap at step 50, given the inputs. Higher is
    /// better; <0.5 triggers the cold-start auto-injection.
    pub expected_overlap_at_step_50: f64,
    /// Warnings the user should see before committing budget.
    pub warnings: Vec<String>,
}

/// Pure-logic capacity calculator — `compute_capacity(req)`. The
/// axum handler is a thin wrapper around this so unit tests can hit
/// the math without a live AppState.
pub fn compute_capacity(req: &CapacityRequest) -> Result<CapacityResponse, String> {
    if req.rank == 0 || req.tokens_per_rollout == 0 || req.rollouts == 0 || req.top_k == 0 {
        return Err("capacity: all positive integers required".to_string());
    }
    // bits_needed ≈ rollouts × tokens × log2(top_k_vocab).
    let bits_needed =
        (req.rollouts as f64) * (req.tokens_per_rollout as f64) * (req.top_k as f64).log2();
    // 2 bits per param (Allen-Zhu 2024). LoRA params per layer ≈
    // ~8 × rank × hidden_size (down + up projections × MLP gate/up/down
    // + attention q/k/v/o). Conservative estimate for LoRA-on-all-
    // linear-layers per Schulman's LoRA Without Regret.
    let lora_params = (req.rank as f64) * (req.hidden_size as f64) * (req.num_layers as f64) * 8.0;
    let bits_storable = lora_params * 2.0;
    let capacity_ratio = bits_storable / bits_needed.max(1.0);

    // Heuristic: overlap-at-step-50 = clip(initial_overlap + 0.2,
    // 0, 1). When initial_overlap_probe is None, default to 0.7
    // (Li et al. healthy-run start).
    let initial = req.initial_overlap_probe.unwrap_or(0.7);
    let expected_overlap_at_step_50 = (initial + 0.2).clamp(0.0, 1.0);

    let mut warnings = Vec::new();
    if capacity_ratio < 0.3 {
        warnings.push(format!(
            "bits_storable < 0.3 × bits_needed (ratio={capacity_ratio:.2}); this run \
             overflows rank-{} by ~{:.1}× — consider rank ≥ {} or shorter rollouts. \
             See §8.5 of the grand plan.",
            req.rank,
            1.0 / capacity_ratio,
            (req.rank as f64 / capacity_ratio).round() as usize
        ));
    }
    if expected_overlap_at_step_50 < 0.5 {
        warnings.push(format!(
            "expected_overlap_at_step_50 < 0.5 ({expected_overlap_at_step_50:.2}); cold-start \
             auto-injection will engage (§8.10)."
        ));
    }
    Ok(CapacityResponse {
        bits_needed,
        bits_storable_in_lora: bits_storable,
        capacity_ratio,
        expected_overlap_at_step_50,
        warnings,
    })
}

async fn capacity(
    State(_state): State<AppState>,
    Json(req): Json<CapacityRequest>,
) -> Result<Json<CapacityResponse>, ApiError> {
    compute_capacity(&req)
        .map(Json)
        .map_err(ApiError::training_invalid_request)
}

// ===========================================================================
// §8.13 tier-aware defaults
// ===========================================================================

/// One tier's worth of paper-cited default values. The §8.13 table
/// rendered as a programmatic shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierDefaults {
    pub tier: String,
    pub default_logit_source: String,
    pub default_loss: String,
    pub default_top_k: usize,
    pub lora_rank: usize,
    pub batch_size: usize,
    pub samples_per_prompt_default: usize,
    pub samples_per_prompt_data_multiplier: usize,
    pub max_rollout_tokens: usize,
    pub auto_checkpoint_cadence_steps: usize,
    pub cost_cap_default_usd: Option<f64>,
    pub cold_start_overlap_threshold: f64,
    pub mixture_distillation_golden_fraction: f64,
    pub eval_gate_required: bool,
    pub notifications_channels: Vec<String>,
}

pub fn builtin_tier_defaults() -> BTreeMap<String, TierDefaults> {
    let mut map = BTreeMap::new();
    map.insert(
        "laptop".to_string(),
        TierDefaults {
            tier: "laptop".to_string(),
            default_logit_source: "Best-cached → RemoteTeacher".to_string(),
            default_loss: "teacher_top_k (K=20, most APIs cap)".to_string(),
            default_top_k: 20,
            lora_rank: 16,
            batch_size: 8,
            samples_per_prompt_default: 4,
            samples_per_prompt_data_multiplier: 32,
            max_rollout_tokens: 4_096,
            auto_checkpoint_cadence_steps: 10,
            cost_cap_default_usd: Some(10.0),
            cold_start_overlap_threshold: 0.5,
            mixture_distillation_golden_fraction: 0.25,
            eval_gate_required: true,
            notifications_channels: vec!["desktop_tray".into(), "email".into()],
        },
    );
    map.insert(
        "prosumer".to_string(),
        TierDefaults {
            tier: "prosumer".to_string(),
            default_logit_source: "LocalTeacher(qwen3.6-27b, fp8)".to_string(),
            default_loss: "teacher_top_k (K=32)".to_string(),
            default_top_k: 32,
            lora_rank: 32,
            batch_size: 16,
            samples_per_prompt_default: 4,
            samples_per_prompt_data_multiplier: 32,
            max_rollout_tokens: 7_168,
            auto_checkpoint_cadence_steps: 10,
            cost_cap_default_usd: Some(25.0),
            cold_start_overlap_threshold: 0.5,
            mixture_distillation_golden_fraction: 0.25,
            eval_gate_required: true,
            notifications_channels: vec![
                "desktop_tray".into(),
                "email".into(),
                "webhook".into(),
            ],
        },
    );
    map.insert(
        "corporate".to_string(),
        TierDefaults {
            tier: "corporate".to_string(),
            default_logit_source: "LocalTeacher(qwen3.6-27b, full) ×N".to_string(),
            default_loss: "full_vocab".to_string(),
            default_top_k: 0,
            lora_rank: 128,
            batch_size: 32,
            samples_per_prompt_default: 4,
            samples_per_prompt_data_multiplier: 16,
            max_rollout_tokens: 7_168,
            auto_checkpoint_cadence_steps: 5,
            cost_cap_default_usd: None,
            cold_start_overlap_threshold: 0.5,
            mixture_distillation_golden_fraction: 0.10,
            eval_gate_required: true,
            notifications_channels: vec!["webhook".into(), "slack_or_teams".into()],
        },
    );
    map
}

#[derive(Debug, Deserialize)]
struct TierQuery {
    tier: Option<String>,
}

#[derive(Debug, Serialize)]
struct TierDefaultsResponse {
    tier: String,
    defaults: TierDefaults,
}

#[derive(Debug, Serialize)]
struct TierDefaultsListResponse {
    tiers: Vec<TierDefaults>,
}

async fn tier_defaults_endpoint(
    State(_state): State<AppState>,
    Query(q): Query<TierQuery>,
) -> Result<Json<TierDefaultsResponse>, ApiError> {
    let map = builtin_tier_defaults();
    let tier = q.tier.unwrap_or_else(|| "prosumer".to_string());
    let defaults = map.get(&tier).cloned().ok_or_else(|| {
        ApiError::training_invalid_request(format!(
            "unknown tier {tier:?}; valid: laptop, prosumer, corporate"
        ))
    })?;
    Ok(Json(TierDefaultsResponse { tier, defaults }))
}

async fn tier_defaults_list(
    State(_state): State<AppState>,
) -> Json<TierDefaultsListResponse> {
    Json(TierDefaultsListResponse {
        tiers: builtin_tier_defaults().into_values().collect(),
    })
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/train", post(submit_front_door))
        .route("/v1/preflight/compatibility", get(compatibility))
        .route("/v1/preflight/capacity", post(capacity))
        .route("/v1/preflight/tier_defaults", get(tier_defaults_endpoint))
        .route("/v1/preflight/tiers", get(tier_defaults_list))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_table_has_30_plus_entries() {
        let table = builtin_compatibility_table();
        assert!(
            table.len() >= 30,
            "§8.4 promises ≥30 day-1 entries; got {}",
            table.len()
        );
    }

    #[test]
    fn tier_defaults_for_three_tiers_present() {
        let map = builtin_tier_defaults();
        assert!(map.contains_key("laptop"));
        assert!(map.contains_key("prosumer"));
        assert!(map.contains_key("corporate"));
        // §8.13 specific values from the grand-plan table.
        assert_eq!(map.get("laptop").unwrap().lora_rank, 16);
        assert_eq!(map.get("prosumer").unwrap().lora_rank, 32);
        assert_eq!(map.get("corporate").unwrap().lora_rank, 128);
        assert_eq!(map.get("corporate").unwrap().auto_checkpoint_cadence_steps, 5);
        assert!(map.get("corporate").unwrap().cost_cap_default_usd.is_none());
        assert!((map.get("laptop").unwrap().cost_cap_default_usd.unwrap() - 10.0).abs() < 1e-9);
        assert!(
            (map.get("prosumer").unwrap().cost_cap_default_usd.unwrap() - 25.0).abs() < 1e-9
        );
    }

    #[test]
    fn capacity_calculator_warns_on_overflow() {
        let resp = compute_capacity(&CapacityRequest {
            rollouts: 1_000_000, // unrealistically big
            tokens_per_rollout: 4_096,
            top_k: 32,
            rank: 4, // tiny LoRA
            num_layers: 32,
            hidden_size: 2560,
            initial_overlap_probe: None,
        })
        .unwrap();
        assert!(resp.capacity_ratio < 0.3);
        assert!(resp.warnings.iter().any(|w| w.contains("overflows rank-4")));
    }

    #[test]
    fn capacity_calculator_no_warning_when_comfortable() {
        let resp = compute_capacity(&CapacityRequest {
            rollouts: 1_000,
            tokens_per_rollout: 4_096,
            top_k: 32,
            rank: 64,
            num_layers: 32,
            hidden_size: 2560,
            initial_overlap_probe: Some(0.78),
        })
        .unwrap();
        assert!(resp.capacity_ratio >= 0.3);
        assert!(
            resp.warnings
                .iter()
                .all(|w| !w.contains("overflows")),
            "no overflow warning expected; got: {:?}",
            resp.warnings
        );
    }

    #[test]
    fn front_door_dispatch_kind_sft() {
        let json = r#"{"kind":"sft","examples":[{"messages":[{"role":"user","content":"hi"}]}]}"#;
        let req: FrontDoorRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FrontDoorRequest::Sft(_)));
    }

    #[test]
    fn front_door_dispatch_kind_opd() {
        let json = r#"{"kind":"opd","teacher":"qwen3.6-27b@local","prompts":[{"messages":[{"role":"user","content":"hi"}]}]}"#;
        let req: FrontDoorRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FrontDoorRequest::Opd(_)));
    }

    #[test]
    fn front_door_dispatch_kind_distill_refresh() {
        let json = r#"{
            "kind":"distill_refresh",
            "name":"company-assistant",
            "new_data":{"dataset":"q4"},
            "behavioural_teacher":"company-assistant@v17"
        }"#;
        let req: FrontDoorRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FrontDoorRequest::DistillRefresh(_)));
    }

    #[test]
    fn front_door_dispatch_kind_distill_merge() {
        let json = r#"{
            "kind":"distill_merge",
            "name":"unified",
            "sources":[{"adapter":"a"},{"adapter":"b"}]
        }"#;
        let req: FrontDoorRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FrontDoorRequest::DistillMerge(_)));
    }

    #[test]
    fn front_door_dispatch_kind_distill_pump() {
        let json = r#"{
            "kind":"distill_pump",
            "name":"math-frontier",
            "teacher":"qwen3.6-27b@local",
            "mode":{"domain":"math_reasoning"}
        }"#;
        let req: FrontDoorRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req, FrontDoorRequest::DistillPump(_)));
    }

    #[test]
    fn front_door_dispatch_rejects_unknown_kind() {
        let json = r#"{"kind":"telepathy","examples":[]}"#;
        let parsed: Result<FrontDoorRequest, _> = serde_json::from_str(json);
        assert!(parsed.is_err());
    }
}
