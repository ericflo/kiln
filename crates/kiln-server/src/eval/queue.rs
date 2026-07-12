//! Eval-job queue and per-job tracking state.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use kiln_eval::{EvalCompareSpec, EvalJobState, EvalProgress, EvalResult, EvalSuite, SuiteResult};
use serde::{Deserialize, Serialize};

/// What a queued eval job actually contains. Either:
///
/// - `Suite { suite, adapter }` — run a registered suite against a single
///   adapter (`None` = base model).
/// - `Inline { suite, adapters }` — run an inline (non-registered) suite,
///   optionally across multiple adapters in compare mode.
/// - `Compare { suite, adapters }` — run a registered suite against every
///   adapter in `adapters` and emit a head-to-head diff.
pub enum QueuedEvalJob {
    /// Run a registered suite against a single adapter (or the base model
    /// when `adapter` is `None`).
    Registered {
        suite_name: String,
        adapter: Option<String>,
        generation_override: Option<kiln_eval::EvalGenerationParams>,
    },
    /// Run an inline (caller-supplied) suite against the active adapter or a
    /// pinned one. Used for one-shot evals that don't get registered.
    Inline {
        suite: Box<EvalSuite>,
        adapter: Option<String>,
        generation_override: Option<kiln_eval::EvalGenerationParams>,
    },
    /// Run a (registered) suite against multiple adapters. The result has
    /// one `SuiteResult` per adapter in input order.
    Compare(EvalCompareSpec),
}

impl QueuedEvalJob {
    /// Suite-of-record name for logging and dashboards (the inline form gets
    /// its own `suite.name`).
    pub fn suite_name(&self) -> &str {
        match self {
            QueuedEvalJob::Registered { suite_name, .. } => suite_name,
            QueuedEvalJob::Inline { suite, .. } => &suite.name,
            QueuedEvalJob::Compare(spec) => &spec.suite,
        }
    }
}

/// What kind of submission produced this job — used by the metrics layer to
/// distinguish on-demand evals from post-training auto-evals.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalSubmissionKind {
    OnDemand,
    PostTraining,
    Compare,
}

/// Entry stored in `state.eval_queue`. The submission kind and the source
/// training job ID live on the matching `EvalJobInfo`; the queue entry
/// only carries the routing info the worker needs to start work.
pub struct EvalQueueEntry {
    pub job_id: String,
    /// Immutable seed resolved before this entry becomes visible to the
    /// worker. Kept beside the payload so execution cannot observe a mutable
    /// tracking-map value.
    pub effective_seed: u64,
    pub job: QueuedEvalJob,
}

/// Immutable identifiers returned by the central eval admission path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalEnqueueReceipt {
    pub job_id: String,
    pub effective_seed: u64,
}

fn now_instant_default() -> std::time::Instant {
    std::time::Instant::now()
}

/// §8.7 promotion gate carried by a post-training eval job. When the run
/// completes, the eval worker applies the verdict: accuracy >= threshold
/// promotes (auto-loads) the adapter when training asked for that;
/// accuracy below it renames the adapter dir to `<name>.failed` and the
/// adapter is never promoted — "kiln must not make the model worse and
/// silently keep serving it."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostEvalGate {
    /// Aggregate-accuracy floor from `PostEvalConfig::min_accuracy`.
    pub min_accuracy: f32,
    /// distill_refresh §6.4: minimum FRACTIONAL recovery vs the baseline
    /// run (`new/baseline >= x`). `None` for ordinary gates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative_recovery: Option<f32>,
    /// distill_refresh §6.4: minimum ABSOLUTE gain vs the baseline run
    /// (`new - baseline >= x`). `None` for ordinary gates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absolute_gain: Option<f32>,
    /// The adapter under judgment (the training job's output).
    pub adapter_name: String,
    /// Training job to stamp the verdict on.
    pub training_job_id: String,
    /// Training requested auto-load, which was DEFERRED until this gate
    /// passes (§8.7: while the verdict is pending, the prior adapter
    /// stays active).
    pub auto_load_on_pass: bool,
}

/// Tracked eval job — mirrors `TrainingJobInfo` so the UI can render both
/// with one code path. Stored under `state.eval_jobs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalJobInfo {
    pub job_id: String,
    pub suite_name: String,
    pub adapters: Vec<Option<String>>,
    pub submission_kind: EvalSubmissionKind,
    /// Immutable snapshot of the resident base-weight artifacts at admission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_weight_shard_manifest: Option<kiln_core::model_provenance::BaseWeightShardManifest>,
    /// Immutable job seed. `None` is reserved for archives written before
    /// eval seed materialization was introduced.
    #[serde(
        default,
        with = "kiln_eval::result::optional_u64_decimal",
        skip_serializing_if = "Option::is_none"
    )]
    pub effective_seed: Option<u64>,
    pub state: EvalJobState,
    pub progress: EvalProgress,
    /// One per finished adapter (compare-mode jobs accumulate as they go).
    pub finished_runs: Vec<SuiteResult>,
    /// Last-known accuracy across the most-recently-finished run — surfaced
    /// here so list views don't have to walk `finished_runs`.
    pub headline_accuracy: Option<f32>,
    /// Free-form failure message when `state == Failed`.
    pub error: Option<String>,
    pub source_training_job_id: Option<String>,
    /// ISO-8601 timestamp when the job was queued.
    pub submitted_at_iso: String,
    /// ISO-8601 timestamp when the job entered Running.
    pub started_at_iso: Option<String>,
    /// ISO-8601 timestamp when the job entered a terminal state.
    pub finished_at_iso: Option<String>,
    #[serde(skip, default = "now_instant_default")]
    pub submitted_at: std::time::Instant,
    #[serde(skip, default)]
    pub finished_at: Option<std::time::Instant>,
    /// Cooperative cancellation handle for the executor while the job is
    /// Running. The cancel endpoint sets it; the worker installs it when the
    /// job starts. Skipped from serde (runtime-only).
    #[serde(skip, default)]
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// §8.7 promotion gate — set by `enqueue_post_training_eval` on the
    /// adapter's run (never the baseline run) when the training request
    /// carried `post_eval.min_accuracy`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval_gate: Option<PostEvalGate>,
}

impl EvalJobInfo {
    /// Construct a freshly-queued job info. Centralizes the `Instant::now`
    /// + `chrono::Utc::now()` + default-state plumbing so submission sites
    /// only have to specify what's actually distinct about the job.
    pub fn queued(
        job_id: String,
        suite_name: String,
        adapters: Vec<Option<String>>,
        submission_kind: EvalSubmissionKind,
        source_training_job_id: Option<String>,
        effective_seed: u64,
    ) -> Self {
        Self {
            job_id,
            suite_name,
            adapters,
            submission_kind,
            base_weight_shard_manifest: None,
            effective_seed: Some(effective_seed),
            state: EvalJobState::Queued,
            progress: EvalProgress::default(),
            finished_runs: Vec::new(),
            headline_accuracy: None,
            error: None,
            source_training_job_id,
            submitted_at_iso: chrono::Utc::now().to_rfc3339(),
            started_at_iso: None,
            finished_at_iso: None,
            submitted_at: std::time::Instant::now(),
            finished_at: None,
            cancel_flag: None,
            post_eval_gate: None,
        }
    }

    /// Snapshot the tracked-job into the public `EvalResult` shape returned
    /// by `GET /v1/eval/jobs/:id`.
    pub fn to_result(&self) -> EvalResult {
        EvalResult {
            job_id: self.job_id.clone(),
            state: self.state,
            base_weight_shard_manifest: self.base_weight_shard_manifest.clone(),
            effective_seed: self.effective_seed,
            seed_derivation: self
                .effective_seed
                .map(|_| kiln_eval::EVAL_SEED_DERIVATION_V1.to_string()),
            runs: self.finished_runs.clone(),
            progress: if matches!(self.state, EvalJobState::Running | EvalJobState::Queued) {
                Some(self.progress.clone())
            } else {
                None
            },
            error: self.error.clone(),
        }
    }
}

/// FIFO eval queue, sibling of `TrainingQueue`. We keep them separate so an
/// eval can run while training is queued behind it (both still serialize
/// at the GPU-coordination lock when needed).
pub struct EvalQueue {
    pub(crate) queue: VecDeque<EvalQueueEntry>,
}

impl EvalQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
    pub fn push(&mut self, entry: EvalQueueEntry) {
        self.queue.push_back(entry);
    }
    pub fn pop(&mut self) -> Option<EvalQueueEntry> {
        self.queue.pop_front()
    }
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    /// Remove a queued job by ID. Returns true if found and removed.
    pub fn remove(&mut self, job_id: &str) -> bool {
        let before = self.queue.len();
        self.queue.retain(|e| e.job_id != job_id);
        self.queue.len() < before
    }
}

pub type SharedEvalQueue = Arc<std::sync::Mutex<EvalQueue>>;
pub type EvalJobs = Arc<std::sync::RwLock<HashMap<String, EvalJobInfo>>>;

pub fn new_shared_eval_queue() -> SharedEvalQueue {
    Arc::new(std::sync::Mutex::new(EvalQueue::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_eval::scorers::Scorer;
    use kiln_eval::{EvalChatMessage, EvalExample, EvalGenerationParams};

    fn dummy_suite() -> EvalSuite {
        EvalSuite {
            name: "t".into(),
            description: None,
            default_scorer: Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            generation: EvalGenerationParams::default(),
            system_prompt: None,
            examples: vec![EvalExample {
                messages: vec![EvalChatMessage::new("user", "x")],
                target: Some("x".into()),
                ..Default::default()
            }],
            schema_version: 1,
            tools: None,
        }
    }

    #[test]
    fn queue_fifo() {
        let mut q = EvalQueue::new();
        q.push(EvalQueueEntry {
            job_id: "a".into(),
            effective_seed: 1,
            job: QueuedEvalJob::Inline {
                suite: Box::new(dummy_suite()),
                adapter: None,
                generation_override: None,
            },
        });
        q.push(EvalQueueEntry {
            job_id: "b".into(),
            effective_seed: 2,
            job: QueuedEvalJob::Inline {
                suite: Box::new(dummy_suite()),
                adapter: None,
                generation_override: None,
            },
        });
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop().unwrap().job_id, "a");
        assert_eq!(q.pop().unwrap().job_id, "b");
        assert!(q.pop().is_none());
    }

    #[test]
    fn queue_remove_works_for_pending_and_missing() {
        let mut q = EvalQueue::new();
        q.push(EvalQueueEntry {
            job_id: "a".into(),
            effective_seed: 1,
            job: QueuedEvalJob::Inline {
                suite: Box::new(dummy_suite()),
                adapter: None,
                generation_override: None,
            },
        });
        assert!(q.remove("a"));
        assert!(!q.remove("a"));
        assert!(q.is_empty());
    }

    #[test]
    fn suite_name_returns_correct_value_for_each_variant() {
        let reg = QueuedEvalJob::Registered {
            suite_name: "registered".into(),
            adapter: None,
            generation_override: None,
        };
        assert_eq!(reg.suite_name(), "registered");
        let inline = QueuedEvalJob::Inline {
            suite: Box::new(dummy_suite()),
            adapter: None,
            generation_override: None,
        };
        assert_eq!(inline.suite_name(), "t");
        let cmp = QueuedEvalJob::Compare(EvalCompareSpec {
            suite: "cmp".into(),
            adapters: vec!["a".into(), "b".into()],
            seed: None,
            generation: None,
        });
        assert_eq!(cmp.suite_name(), "cmp");
    }

    #[test]
    fn legacy_archived_job_without_seed_remains_readable_and_honest() {
        let info = EvalJobInfo::queued(
            "legacy".into(),
            "t".into(),
            vec![None],
            EvalSubmissionKind::OnDemand,
            None,
            17,
        );
        let mut value = serde_json::to_value(info).unwrap();
        value.as_object_mut().unwrap().remove("effective_seed");
        let decoded: EvalJobInfo = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.effective_seed, None);
        let public = decoded.to_result();
        assert_eq!(public.effective_seed, None);
        assert_eq!(public.seed_derivation, None);
    }
}
