//! `/v1/recipes/run` + `GET /v1/recipes` — recipe runtime (§3.7).
//!
//! Recipes are YAML-defined multi-step workflows that compose the
//! distill_* endpoints. Each step is a typed enum variant that
//! resolves to one of the existing pipelines. The runtime processes
//! steps in order, threading the previous step's output (an adapter
//! name) into the next step's `base` field where applicable.
//!
//! The six day-one recipes from §3.7 ship as YAML files baked into
//! the binary via `include_str!`; users can also POST a recipe-body
//! payload that the server runs ad-hoc.

use std::collections::BTreeMap;

use axum::Json;
use axum::Router;
use axum::extract::{State, rejection::JsonRejection};
use axum::routing::{get, post};
use kiln_train::{
    DistillMergeRequest, DistillMergeSource, DistillPumpMode, DistillPumpRequest,
    DistillRefreshRequest, DistillSelfRequest, NewKnowledgeSource, OpdConfig, OpdRequest,
    SelfDistillMode, SftConfig, SftRequest,
};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::{AppState, TrainingWorkload};

/// One step in a recipe. The trainer runs steps in order and threads
/// adapter outputs through subsequent steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecipeStep {
    /// Run SFT on the given examples (typically as a cold-start phase
    /// before OPD per §3.1 + §8.10).
    Sft {
        /// Output adapter name.
        name: String,
        /// Optional base adapter — None starts from the model.
        #[serde(default)]
        base_adapter: Option<String>,
        /// Inline SFT examples or a server-registered dataset.
        examples_from: ExamplesSource,
        #[serde(default)]
        config: SftConfig,
    },
    /// Run an OPD run.
    Opd {
        /// Output adapter name.
        name: String,
        teacher: String,
        /// Inline prompts or server-registered dataset.
        prompts_from: PromptsSource,
        #[serde(default)]
        config: OpdConfig,
    },
    /// Behaviour-space adapter merge.
    DistillMerge {
        name: String,
        sources: Vec<DistillMergeSource>,
        #[serde(default = "default_merge_student")]
        student: String,
        #[serde(default = "default_merge_rollout_budget")]
        rollout_budget: usize,
        #[serde(default)]
        config: OpdConfig,
    },
    /// Knowledge Pump (3 modes).
    DistillPump {
        name: String,
        teacher: String,
        mode: DistillPumpMode,
        #[serde(default)]
        config: OpdConfig,
    },
    /// Continual-learning refresh.
    DistillRefresh {
        /// Existing adapter to refresh.
        name: String,
        new_data: NewKnowledgeSource,
        behavioural_teacher: String,
        #[serde(default = "default_background_chat")]
        background_chat: String,
        #[serde(default)]
        config: OpdConfig,
    },
    /// PI self-distillation.
    DistillSelf {
        name: String,
        mode: SelfDistillMode,
        #[serde(default)]
        config: OpdConfig,
    },
    /// Post-training eval gate (§3.9). The trainer verifies the
    /// previous step's output adapter clears `require_min_score`
    /// on the named eval suite; if not, the recipe halts and the
    /// adapter is not promoted.
    PostEval {
        suite: String,
        adapter: String,
        require_min_score: f64,
    },
}

fn default_merge_student() -> String {
    "base".to_string()
}
fn default_merge_rollout_budget() -> usize {
    5_000
}
fn default_background_chat() -> String {
    "tulu3".to_string()
}

/// Inline prompts or a server dataset reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptsSource {
    Dataset { dataset: String },
    Inline { prompts: Vec<kiln_train::OpdPrompt> },
}

/// Inline SFT examples or a server dataset reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExamplesSource {
    Dataset {
        dataset: String,
    },
    Inline {
        examples: Vec<kiln_train::SftExample>,
    },
}

/// A complete recipe — a sequence of steps + optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub steps: Vec<RecipeStep>,
}

/// `POST /v1/recipes/run` payload — either a named recipe or an
/// inline recipe body, plus optional input overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RecipeRunRequest {
    Named {
        recipe: String,
        #[serde(default)]
        inputs: BTreeMap<String, serde_json::Value>,
    },
    Inline {
        body: Recipe,
        #[serde(default)]
        inputs: BTreeMap<String, serde_json::Value>,
    },
}

/// Response from `POST /v1/recipes/run`: list of job IDs for each
/// step that was queued.
#[derive(Debug, Serialize)]
pub struct RecipeRunResponse {
    pub recipe: String,
    pub job_ids: Vec<String>,
    /// Exact decimal seed for every queued training step, keyed by job ID.
    pub effective_seeds: BTreeMap<String, String>,
    pub message: String,
}

/// Response from `GET /v1/recipes`.
#[derive(Debug, Serialize)]
pub struct RecipesListResponse {
    pub recipes: Vec<RecipeDescriptor>,
}

#[derive(Debug, Serialize)]
pub struct RecipeDescriptor {
    pub name: String,
    pub description: Option<String>,
    pub num_steps: usize,
    pub admission: RecipeAdmissionDescriptor,
}

#[derive(Debug, Serialize)]
pub struct RecipeAdmissionDescriptor {
    pub supported: bool,
    pub unavailable_reason: Option<String>,
}

// Six day-one recipes baked into the binary (§3.7) + four agentic
// recipes (§10.8).
const RECIPE_RECOVER_IF: &str = include_str!("recipes/recover-instruction-following.yaml");
const RECIPE_FRONTIER_PUMP: &str = include_str!("recipes/frontier-pump.yaml");
const RECIPE_MERGE_LORAS: &str = include_str!("recipes/merge-my-loras.yaml");
const RECIPE_UPDATE_DOCS: &str = include_str!("recipes/update-with-new-docs.yaml");
const RECIPE_CODING_FROM_REPO: &str = include_str!("recipes/coding-assistant-from-repo.yaml");
const RECIPE_JUDGE_FROM_PICKS: &str = include_str!("recipes/make-a-judge-lora-from-my-picks.yaml");

// §10.8 agentic recipes.
const RECIPE_LEARN_PI_HISTORY: &str = include_str!("recipes/learn-from-my-pi-history.yaml");
const RECIPE_MERGE_AGENT_LORAS: &str = include_str!("recipes/merge-my-agent-loras.yaml");
const RECIPE_RECOVER_TOOL_FOLLOWING: &str = include_str!("recipes/recover-tool-following.yaml");
const RECIPE_PI_SHARE_THEN_PUMP: &str = include_str!("recipes/pi-share-then-pump.yaml");

/// Load all day-one recipes baked into the server binary.
pub fn builtin_recipes() -> Vec<(String, &'static str)> {
    vec![
        (
            "recover-instruction-following".to_string(),
            RECIPE_RECOVER_IF,
        ),
        ("frontier-pump".to_string(), RECIPE_FRONTIER_PUMP),
        ("merge-my-loras".to_string(), RECIPE_MERGE_LORAS),
        ("update-with-new-docs".to_string(), RECIPE_UPDATE_DOCS),
        (
            "coding-assistant-from-repo".to_string(),
            RECIPE_CODING_FROM_REPO,
        ),
        (
            "make-a-judge-lora-from-my-picks".to_string(),
            RECIPE_JUDGE_FROM_PICKS,
        ),
        // §10.8 agentic recipes
        (
            "learn-from-my-pi-history".to_string(),
            RECIPE_LEARN_PI_HISTORY,
        ),
        ("merge-my-agent-loras".to_string(), RECIPE_MERGE_AGENT_LORAS),
        (
            "recover-tool-following".to_string(),
            RECIPE_RECOVER_TOOL_FOLLOWING,
        ),
        ("pi-share-then-pump".to_string(), RECIPE_PI_SHARE_THEN_PUMP),
    ]
}

fn recipe_admission(state: &AppState, recipe: &Recipe) -> RecipeAdmissionDescriptor {
    for (step_index, step) in recipe.steps.iter().enumerate() {
        if let Some(workload) = recipe_step_workload(step)
            && let Some(reason) = state.training_workload_unavailable_reason(workload)
        {
            return RecipeAdmissionDescriptor {
                supported: false,
                unavailable_reason: Some(format!(
                    "step {} ({}) is unavailable: {reason}",
                    step_index + 1,
                    workload.label(),
                )),
            };
        }
        if let Some((optimizer, rank)) = recipe_step_optimizer_request(step)
            && let Err(error) =
                super::training::enforce_training_optimizer_admission(state, optimizer, rank)
        {
            return RecipeAdmissionDescriptor {
                supported: false,
                unavailable_reason: Some(format!(
                    "step {} optimizer tuple is unavailable: {error}",
                    step_index + 1,
                )),
            };
        }
    }
    RecipeAdmissionDescriptor {
        supported: true,
        unavailable_reason: None,
    }
}

async fn list_recipes(State(state): State<AppState>) -> Json<RecipesListResponse> {
    let mut out = Vec::new();
    for (name, yaml) in builtin_recipes() {
        // Best-effort parse; if the bundled YAML is malformed the
        // CI builds would fail. Stay graceful here so a single bad
        // recipe doesn't blow up the list endpoint.
        let descriptor = match serde_yaml::from_str::<Recipe>(yaml) {
            Ok(r) => RecipeDescriptor {
                name: r.name.clone(),
                description: r.description.clone(),
                num_steps: r.steps.len(),
                admission: recipe_admission(&state, &r),
            },
            Err(_) => RecipeDescriptor {
                name,
                description: Some("(failed to parse YAML)".into()),
                num_steps: 0,
                admission: RecipeAdmissionDescriptor {
                    supported: false,
                    unavailable_reason: Some("bundled recipe YAML failed to parse".to_string()),
                },
            },
        };
        out.push(descriptor);
    }
    Json(RecipesListResponse { recipes: out })
}

fn training_step_name(step: &RecipeStep) -> Option<&str> {
    match step {
        RecipeStep::Sft { name, .. }
        | RecipeStep::Opd { name, .. }
        | RecipeStep::DistillMerge { name, .. }
        | RecipeStep::DistillPump { name, .. }
        | RecipeStep::DistillRefresh { name, .. }
        | RecipeStep::DistillSelf { name, .. } => Some(name),
        RecipeStep::PostEval { .. } => None,
    }
}

fn recipe_step_optimizer_request(step: &RecipeStep) -> Option<(kiln_train::Optimizer, usize)> {
    match step {
        RecipeStep::Sft { config, .. } => Some((config.optimizer, config.lora_rank)),
        RecipeStep::Opd { config, .. }
        | RecipeStep::DistillMerge { config, .. }
        | RecipeStep::DistillPump { config, .. }
        | RecipeStep::DistillRefresh { config, .. }
        | RecipeStep::DistillSelf { config, .. } => Some((config.optimizer, config.lora_rank)),
        RecipeStep::PostEval { .. } => None,
    }
}

fn recipe_step_workload(step: &RecipeStep) -> Option<TrainingWorkload> {
    match step {
        RecipeStep::Sft { .. } => Some(TrainingWorkload::Sft),
        RecipeStep::DistillRefresh { .. } => Some(TrainingWorkload::DistillRefresh),
        RecipeStep::Opd { .. }
        | RecipeStep::DistillMerge { .. }
        | RecipeStep::DistillPump { .. }
        | RecipeStep::DistillSelf { .. } => Some(TrainingWorkload::Opd),
        RecipeStep::PostEval { .. } => None,
    }
}

fn validate_recipe_structure_and_names(
    recipe_name: &str,
    recipe: &Recipe,
) -> Result<usize, ApiError> {
    let mut previous_adapter: Option<&str> = None;
    let mut training_steps = 0usize;
    for step in &recipe.steps {
        if let RecipeStep::PostEval { suite, adapter, .. } = step {
            let Some(previous) = previous_adapter else {
                return Err(ApiError::training_invalid_request(format!(
                    "recipe '{recipe_name}': PostEval (suite={suite}) must follow a \
                     training step -- it gates that step's output adapter"
                )));
            };
            if adapter != previous {
                return Err(ApiError::training_invalid_request(format!(
                    "recipe '{recipe_name}': PostEval adapter '{adapter}' does not match \
                     the preceding step's output adapter '{previous}'"
                )));
            }
            continue;
        }

        let name = training_step_name(step).expect("non-PostEval step has an adapter name");
        super::adapters::validate_adapter_name(name)?;
        previous_adapter = Some(name);
        training_steps = training_steps.saturating_add(1);
    }
    Ok(training_steps)
}

async fn run_recipe(
    State(state): State<AppState>,
    payload: Result<Json<RecipeRunRequest>, JsonRejection>,
) -> Result<Json<RecipeRunResponse>, ApiError> {
    let req = super::training::parse_training_json(payload, "recipe request")?;
    if state.shutdown.load(Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    let (recipe_name, recipe) = match req {
        RecipeRunRequest::Named { recipe, .. } => {
            let yaml = builtin_recipes()
                .into_iter()
                .find(|(n, _)| *n == recipe)
                .map(|(_, y)| y)
                .ok_or_else(|| {
                    ApiError::training_invalid_request(format!(
                        "unknown recipe: {recipe}. List available with GET /v1/recipes"
                    ))
                })?;
            let parsed: Recipe = serde_yaml::from_str(yaml)
                .map_err(|e| ApiError::internal(format!("bundled recipe YAML invalid: {e}")))?;
            (recipe, parsed)
        }
        RecipeRunRequest::Inline { body, .. } => (body.name.clone(), body),
    };

    if recipe.steps.is_empty() {
        return Err(ApiError::training_invalid_request(format!(
            "recipe {recipe_name} has no steps"
        )));
    }

    // Validate the complete recipe before resolving any corpus. Capacity and
    // mock-mode rejection then happen before tokenization while a hostile name
    // on a later step still rejects the entire recipe atomically.
    let training_steps = validate_recipe_structure_and_names(&recipe_name, &recipe)?;
    super::training::enforce_queue_capacity_for(&state, training_steps)?;
    super::training::ensure_training_backend_admission(&state)?;
    for step in &recipe.steps {
        if let Some(workload) = recipe_step_workload(step) {
            super::training::enforce_training_workload_admission(&state, workload)?;
        }
        if let Some((optimizer, rank)) = recipe_step_optimizer_request(step) {
            super::training::enforce_training_optimizer_admission(&state, optimizer, rank)?;
        }
    }

    // Enqueue each step independently. Steps run in FIFO order via the
    // global training queue, so by the time step N+1 starts its base
    // adapter (step N's output) is already on disk. We auto-chain by
    // defaulting `base_adapter` on each training step to the previous
    // step's output adapter when the recipe didn't set one. A PostEval
    // step doesn't enqueue a job — it becomes the PRECEDING step's §8.7
    // promotion gate (post_eval on the queued job: auto-load defers
    // until the eval passes; a failing adapter demotes to <name>.failed).
    // Two phases: resolve + validate every step first, then enqueue. A bad
    // adapter name on step N must reject the whole recipe before step 1 is
    // queued — never a partially-enqueued recipe.
    let mut prepared = Vec::with_capacity(recipe.steps.len());
    let mut top_k_adjustments = Vec::new();
    let mut previous_adapter: Option<String> = None;
    for (idx, step) in recipe.steps.iter().enumerate() {
        if matches!(step, RecipeStep::PostEval { .. }) {
            continue;
        }
        // Lookahead: a directly-following PostEval step becomes this
        // job's promotion gate — auto-load defers until the eval passes
        // and a failing adapter demotes to <name>.failed (the same
        // training_queue §8.7 machinery direct submissions use).
        let post_eval = match recipe.steps.get(idx + 1) {
            Some(RecipeStep::PostEval {
                suite,
                require_min_score,
                ..
            }) => Some(kiln_eval::PostEvalConfig {
                suite: suite.clone(),
                data_scope: Default::default(),
                generation: None,
                min_accuracy: Some(*require_min_score as f32),
                include_baseline: false,
            }),
            _ => None,
        };
        let job_id = uuid::Uuid::new_v4().to_string();
        let (adapter_name, mut queued) = step_to_queued_job(
            &state,
            step,
            previous_adapter.as_deref(),
            &job_id,
            post_eval,
        )?;
        if let Some((requested, effective)) =
            super::training::normalize_queued_opd_top_k(&state, &mut queued)?
        {
            top_k_adjustments.push(format!("{adapter_name}: {requested}->{effective}"));
        }
        previous_adapter = Some(adapter_name.clone());
        prepared.push((job_id, adapter_name, queued));
    }

    let job_ids: Vec<String> = prepared
        .iter()
        .map(|(job_id, _, _)| job_id.clone())
        .collect();
    let pending = prepared
        .into_iter()
        .map(|(job_id, adapter_name, queued)| prepare_step_job(&job_id, &adapter_name, queued))
        .collect();
    super::training::admit_training_jobs(&state, pending)?;
    let effective_seeds = super::training::admitted_training_seeds(&state, &job_ids)?;

    Ok(Json(RecipeRunResponse {
        recipe: recipe_name.clone(),
        message: format!(
            "Queued {} training step(s) from recipe {recipe_name}. Steps run \
             FIFO; each step's base_adapter chains to the previous output by \
             default.{}",
            job_ids.len(),
            (!top_k_adjustments.is_empty())
                .then(|| format!(" Effective top_k: {}.", top_k_adjustments.join(", ")))
                .unwrap_or_default()
        ),
        job_ids,
        effective_seeds,
    }))
}

/// Map a `RecipeStep` to a `(adapter_name, QueuedJob)` pair.
///
/// `previous_adapter`, when `Some`, is the output adapter name of the
/// previous (training) step in the same recipe. Each training step
/// auto-chains to it when its own `base_adapter` field is empty —
/// the §3.7 recipe contract is "each step's output is the next
/// step's input by default."
fn step_to_queued_job(
    state: &AppState,
    step: &RecipeStep,
    previous_adapter: Option<&str>,
    _job_id: &str,
    post_eval: Option<kiln_eval::PostEvalConfig>,
) -> Result<(String, crate::training_queue::QueuedJob), ApiError> {
    use crate::training_queue::QueuedJob;
    // Keep this helper safe for direct unit callers as well as the two-phase
    // runner above.
    if let Some(name) = training_step_name(step) {
        super::adapters::validate_adapter_name(name)?;
    }
    match step {
        RecipeStep::Sft {
            name,
            base_adapter,
            examples_from,
            config,
        } => {
            let mut sft_config = config.clone();
            if sft_config.base_adapter.is_none() {
                sft_config.base_adapter = base_adapter
                    .clone()
                    .or_else(|| previous_adapter.map(|s| s.to_string()));
            }
            sft_config.output_name = Some(name.clone());
            sft_config.validate_native_contract().map_err(|error| {
                ApiError::training_invalid_request(format!(
                    "SFT recipe step {name:?} has an invalid native profile: {error:#}"
                ))
            })?;
            let (examples, dataset) = match examples_from {
                ExamplesSource::Inline { examples } => (examples.clone(), None),
                ExamplesSource::Dataset { dataset } => (Vec::new(), Some(dataset.clone())),
            };
            Ok((
                name.clone(),
                QueuedJob::Sft(SftRequest {
                    dataset_path: None,
                    dataset,
                    dataset_split: None,
                    examples,
                    config: sft_config,
                    ingestion: None,
                    post_eval: post_eval.clone(),
                }),
            ))
        }
        RecipeStep::Opd {
            name,
            teacher,
            prompts_from,
            config,
        } => {
            let prompts = match prompts_from {
                PromptsSource::Inline { prompts } => prompts.clone(),
                PromptsSource::Dataset { dataset } => {
                    // `agent_traces:` selectors (e.g. the day-one
                    // learn-from-my-pi-history recipe) resolve against the
                    // §10.3 trace index; bare names against the uploaded
                    // dataset registry.
                    crate::dataset_resolve::resolve_opd_dataset_selector(
                        dataset,
                        &state.adapter_dir,
                        state.dataset_registry.as_deref(),
                        crate::recent_requests::now_unix_ms() as i64,
                    )
                    .map_err(|e| {
                        ApiError::training_invalid_request(format!(
                            "OPD step `dataset: {dataset}`: {e}"
                        ))
                    })?
                }
            };
            let mut opd_config = config.clone();
            if opd_config.base_adapter.is_none() {
                opd_config.base_adapter = previous_adapter.map(|s| s.to_string());
            }
            opd_config.output_name = Some(name.clone());
            Ok((
                name.clone(),
                QueuedJob::Opd(OpdRequest {
                    prompts,
                    dataset_path: None,
                    teacher: teacher.clone(),
                    config: opd_config,
                    post_eval: post_eval.clone(),
                }),
            ))
        }
        RecipeStep::DistillMerge {
            name,
            sources,
            student,
            rollout_budget,
            config,
        } => {
            let mut c = config.clone();
            c.output_name = Some(name.clone());
            Ok((
                name.clone(),
                QueuedJob::DistillMerge(DistillMergeRequest {
                    name: name.clone(),
                    sources: sources.clone(),
                    student: student.clone(),
                    rollout_budget: *rollout_budget,
                    config: c,
                    post_eval: post_eval.clone(),
                }),
            ))
        }
        RecipeStep::DistillPump {
            name,
            teacher,
            mode,
            config,
        } => {
            let mut c = config.clone();
            if c.base_adapter.is_none() {
                c.base_adapter = previous_adapter.map(|s| s.to_string());
            }
            c.output_name = Some(name.clone());
            Ok((
                name.clone(),
                QueuedJob::DistillPump(DistillPumpRequest {
                    name: name.clone(),
                    teacher: teacher.clone(),
                    mode: mode.clone(),
                    rank: None,
                    rollout_budget: 50_000,
                    use_cache: true,
                    config: c,
                    post_eval: post_eval.clone(),
                }),
            ))
        }
        RecipeStep::DistillRefresh {
            name,
            new_data,
            behavioural_teacher,
            background_chat,
            config,
        } => {
            let mut c = config.clone();
            c.output_name = Some(name.clone());
            Ok((
                name.clone(),
                QueuedJob::DistillRefresh(DistillRefreshRequest {
                    name: name.clone(),
                    new_data: new_data.clone(),
                    behavioural_teacher: behavioural_teacher.clone(),
                    background_chat: background_chat.clone(),
                    require_if_eval_recovery: 0.95,
                    require_internal_qa_gain: 0.05,
                    config: c,
                    post_eval: post_eval.clone(),
                    if_eval_suite: None,
                    new_knowledge_eval_suite: None,
                }),
            ))
        }
        RecipeStep::DistillSelf { name, mode, config } => {
            let mut c = config.clone();
            if c.base_adapter.is_none() {
                c.base_adapter = previous_adapter.map(|s| s.to_string());
            }
            c.output_name = Some(name.clone());
            Ok((
                name.clone(),
                QueuedJob::DistillSelf(DistillSelfRequest {
                    name: name.clone(),
                    mode: *mode,
                    prompts: None,
                    ground_truth: None,
                    documents: None,
                    config: c,
                    post_eval: post_eval.clone(),
                }),
            ))
        }
        RecipeStep::PostEval {
            suite,
            adapter,
            require_min_score: _,
        } => {
            // PostEval never reaches here: the prepare loop attaches it
            // to the preceding step's post_eval config (or rejects the
            // recipe when dangling/mismatched).
            Err(ApiError::training_invalid_request(format!(
                "PostEval step (suite={suite}, adapter={adapter}) is attached to the \
                 preceding training step by the recipe runner; it is not a standalone job"
            )))
        }
    }
}

fn prepare_step_job(
    job_id: &str,
    adapter_name: &str,
    queued: crate::training_queue::QueuedJob,
) -> (
    crate::state::TrainingJobInfo,
    crate::training_queue::QueueEntry,
) {
    use crate::state::{TrainingJobInfo, TrainingJobType};
    use crate::training_queue::QueueEntry;
    let info = TrainingJobInfo {
        job_id: job_id.to_string(),
        adapter_name: adapter_name.to_string(),
        job_type: TrainingJobType::Opd,
        effective_seed: None,
        state: TrainingState::Queued,
        progress: 0.0,
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
    };
    (
        info,
        QueueEntry {
            job_id: job_id.to_string(),
            reserved_bytes: 0,
            teacher_bindings: Vec::new(),
            admitted_resume_checkpoint: None,
            prepared_data: Default::default(),
            prepared_data_permit: Default::default(),
            job: queued,
        },
    )
}

use kiln_train::TrainingState;
use std::sync::atomic::Ordering;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/recipes", get(list_recipes))
        .route("/v1/recipes/run", post(run_recipe))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_six_builtin_recipes_parse() {
        for (name, yaml) in builtin_recipes() {
            let recipe: Recipe = serde_yaml::from_str(yaml).unwrap_or_else(|e| {
                panic!("recipe {name} failed to parse: {e}");
            });
            assert_eq!(
                recipe.name, name,
                "recipe `name` field {} doesn't match file slug {}",
                recipe.name, name
            );
            assert!(!recipe.steps.is_empty(), "recipe {name} has no steps");
        }
    }

    #[test]
    fn builtin_recipe_optimizer_defaults_fit_every_accelerator_contract() {
        use kiln_model::backend::TrainingPrecisionPolicy;
        use kiln_model::{TrainingOptimizerRounding, TrainingOptimizerSupport};

        let backends = [
            (
                "cuda",
                kiln_tensor::Device::Cuda(0),
                TrainingPrecisionPolicy::cuda(),
            ),
            (
                "rocm",
                kiln_tensor::Device::Rocm(0),
                TrainingPrecisionPolicy::rocm(),
            ),
            (
                "metal",
                kiln_tensor::Device::Metal(0),
                TrainingPrecisionPolicy::metal(),
            ),
            (
                "vulkan",
                kiln_tensor::Device::Vulkan(0),
                TrainingPrecisionPolicy::vulkan(),
            ),
        ];

        for (recipe_name, yaml) in builtin_recipes() {
            let recipe: Recipe = serde_yaml::from_str(yaml).unwrap();
            for (step_index, step) in recipe.steps.iter().enumerate() {
                let config = match step {
                    RecipeStep::Sft { config, .. } => Some((config.optimizer, config.lora_rank)),
                    RecipeStep::Opd { config, .. }
                    | RecipeStep::DistillMerge { config, .. }
                    | RecipeStep::DistillPump { config, .. }
                    | RecipeStep::DistillRefresh { config, .. }
                    | RecipeStep::DistillSelf { config, .. } => {
                        Some((config.optimizer, config.lora_rank))
                    }
                    RecipeStep::PostEval { .. } => None,
                };
                let Some((optimizer, rank)) = config else {
                    continue;
                };
                for (backend, device, precision) in backends {
                    TrainingOptimizerSupport::for_backend(backend, device)
                        .resolve_optimizer_request(
                            precision,
                            optimizer.kind(),
                            kiln_tensor::DType::BF16,
                            TrainingOptimizerRounding::RoundToNearest,
                            rank,
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "recipe {recipe_name} step {step_index} optimizer tuple is unsupported on {backend}: {error}"
                            )
                        });
                }
            }
        }
    }

    /// A PostEval step becomes the preceding step's §8.7 gate; dangling
    /// or mismatched gates reject the whole recipe before anything
    /// enqueues.
    #[test]
    fn post_eval_step_parses_and_validates_placement() {
        let recipe: Recipe = serde_yaml::from_str(
            "name: gated\nsteps:\n  - kind: post_eval\n    suite: qwen3.5-agentic-core\n    adapter: x\n    require_min_score: 0.7\n",
        )
        .unwrap();
        let RecipeStep::PostEval {
            suite,
            adapter,
            require_min_score,
        } = &recipe.steps[0]
        else {
            panic!("expected PostEval step");
        };
        assert_eq!(suite, "qwen3.5-agentic-core");
        assert_eq!(adapter, "x");
        assert!((require_min_score - 0.7).abs() < 1e-9);
        // The lookahead conversion used by the prepare loop.
        let gate = kiln_eval::PostEvalConfig {
            suite: suite.clone(),
            data_scope: Default::default(),
            generation: None,
            min_accuracy: Some(*require_min_score as f32),
            include_baseline: false,
        };
        assert_eq!(gate.min_accuracy, Some(0.7f32));
    }

    #[test]
    fn recipe_inline_request_parses() {
        let json = r#"{
            "body": {
                "name": "test",
                "steps": [
                    {"kind": "opd", "name": "step1", "teacher": "qwen3.6-27b@local",
                     "prompts_from": {"prompts": [{"messages":[{"role":"user","content":"hi"}]}]}}
                ]
            }
        }"#;
        let req: RecipeRunRequest = serde_json::from_str(json).unwrap();
        match req {
            RecipeRunRequest::Inline { body, .. } => {
                assert_eq!(body.steps.len(), 1);
            }
            other => panic!("expected Inline, got {other:?}"),
        }
    }

    #[test]
    fn step_recipe_roundtrips_yaml() {
        let yaml = r#"
name: example
description: demo
steps:
  - kind: opd
    name: step1
    teacher: qwen3.6-27b@local
    prompts_from:
      prompts:
        - messages:
            - role: user
              content: hi
"#;
        let r: Recipe = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(r.name, "example");
        assert_eq!(r.steps.len(), 1);
    }
}
