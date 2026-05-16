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
use axum::extract::State;
use axum::routing::{get, post};
use kiln_train::{
    DistillMergeRequest, DistillMergeSource, DistillPumpMode, DistillPumpRequest,
    DistillRefreshRequest, DistillSelfRequest, NewKnowledgeSource, OpdConfig, OpdRequest,
    SelfDistillMode, SftConfig, SftRequest,
};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

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
    Dataset { dataset: String },
    Inline { examples: Vec<kiln_train::SftExample> },
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
}

// Six day-one recipes baked into the binary (§3.7).
const RECIPE_RECOVER_IF: &str = include_str!("recipes/recover-instruction-following.yaml");
const RECIPE_FRONTIER_PUMP: &str = include_str!("recipes/frontier-pump.yaml");
const RECIPE_MERGE_LORAS: &str = include_str!("recipes/merge-my-loras.yaml");
const RECIPE_UPDATE_DOCS: &str = include_str!("recipes/update-with-new-docs.yaml");
const RECIPE_CODING_FROM_REPO: &str =
    include_str!("recipes/coding-assistant-from-repo.yaml");
const RECIPE_JUDGE_FROM_PICKS: &str =
    include_str!("recipes/make-a-judge-lora-from-my-picks.yaml");

/// Load all day-one recipes baked into the server binary.
pub fn builtin_recipes() -> Vec<(String, &'static str)> {
    vec![
        ("recover-instruction-following".to_string(), RECIPE_RECOVER_IF),
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
    ]
}

async fn list_recipes(State(_state): State<AppState>) -> Json<RecipesListResponse> {
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
            },
            Err(_) => RecipeDescriptor {
                name,
                description: Some("(failed to parse YAML)".into()),
                num_steps: 0,
            },
        };
        out.push(descriptor);
    }
    Json(RecipesListResponse { recipes: out })
}

async fn run_recipe(
    State(state): State<AppState>,
    Json(req): Json<RecipeRunRequest>,
) -> Result<Json<RecipeRunResponse>, ApiError> {
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
            let parsed: Recipe = serde_yaml::from_str(yaml).map_err(|e| {
                ApiError::internal(format!("bundled recipe YAML invalid: {e}"))
            })?;
            (recipe, parsed)
        }
        RecipeRunRequest::Inline { body, .. } => (body.name.clone(), body),
    };

    if recipe.steps.is_empty() {
        return Err(ApiError::training_invalid_request(format!(
            "recipe {recipe_name} has no steps"
        )));
    }

    // Enqueue each step independently. Step ordering / dependency
    // (one step's output becoming the next's base) is honoured by
    // the trainer body when it lands (#31); here we land the multi-
    // job enqueue so the surface works end-to-end.
    let mut job_ids = Vec::with_capacity(recipe.steps.len());
    for step in &recipe.steps {
        let job_id = uuid::Uuid::new_v4().to_string();
        let (adapter_name, queued) = step_to_queued_job(step, &job_id)?;
        // Register tracking entry, mirroring submit_* helpers.
        register_step_job(&state, &job_id, &adapter_name, queued);
        job_ids.push(job_id);
    }

    Ok(Json(RecipeRunResponse {
        recipe: recipe_name.clone(),
        message: format!(
            "Queued {} step(s) from recipe {recipe_name}. Note: cross-step \
             adapter chaining lands with #31.",
            job_ids.len()
        ),
        job_ids,
    }))
}

/// Map a `RecipeStep` to a `(adapter_name, QueuedJob)` pair.
fn step_to_queued_job(
    step: &RecipeStep,
    _job_id: &str,
) -> Result<(String, crate::training_queue::QueuedJob), ApiError> {
    use crate::training_queue::QueuedJob;
    match step {
        RecipeStep::Sft {
            name,
            base_adapter,
            examples_from,
            config,
        } => {
            let examples = match examples_from {
                ExamplesSource::Inline { examples } => examples.clone(),
                ExamplesSource::Dataset { dataset } => {
                    return Err(ApiError::training_invalid_request(format!(
                        "SFT step with `dataset: {dataset}` requires server-side dataset \
                         resolution (wired with #31)"
                    )));
                }
            };
            let mut sft_config = config.clone();
            if sft_config.base_adapter.is_none() {
                sft_config.base_adapter = base_adapter.clone();
            }
            sft_config.output_name = Some(name.clone());
            Ok((
                name.clone(),
                QueuedJob::Sft(SftRequest {
                    examples,
                    config: sft_config,
                    post_eval: None,
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
                    return Err(ApiError::training_invalid_request(format!(
                        "OPD step with `dataset: {dataset}` requires server-side dataset \
                         resolution (wired with #31)"
                    )));
                }
            };
            let mut opd_config = config.clone();
            opd_config.output_name = Some(name.clone());
            Ok((
                name.clone(),
                QueuedJob::Opd(OpdRequest {
                    prompts,
                    dataset_path: None,
                    teacher: teacher.clone(),
                    config: opd_config,
                    post_eval: None,
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
                    post_eval: None,
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
                    post_eval: None,
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
                    post_eval: None,
                }),
            ))
        }
        RecipeStep::DistillSelf { name, mode, config } => {
            let mut c = config.clone();
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
                    post_eval: None,
                }),
            ))
        }
        RecipeStep::PostEval {
            suite,
            adapter,
            require_min_score: _,
        } => {
            // Eval steps don't enqueue training jobs; for milestone-9
            // we surface them as "no-op" entries in the response so
            // the user sees the eval was acknowledged. Real
            // post-eval gating lives in the trainer body (#31) which
            // will run the eval inline and halt the recipe on
            // failure.
            Err(ApiError::training_invalid_request(format!(
                "PostEval step (suite={suite}, adapter={adapter}) is handled inline by the \
                 trainer body (#31); add as a config field instead of a standalone step \
                 in pre-#31 recipes"
            )))
        }
    }
}

fn register_step_job(
    state: &AppState,
    job_id: &str,
    adapter_name: &str,
    queued: crate::training_queue::QueuedJob,
) {
    use crate::state::{TrainingJobInfo, TrainingJobType};
    use crate::training_queue::QueueEntry;
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
        auto_load: true,
        finished_at: None,
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
        job: queued,
    });
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
            assert!(
                !recipe.steps.is_empty(),
                "recipe {name} has no steps"
            );
        }
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
