//! Eval subsystem — registry, queue, worker, executor.
//!
//! Eval lives next to training: both queue against the same GPU, share the
//! model runner, and use the same JSON-shaped job-status surface. The two
//! paths differ in concurrency contract — eval acquires the **read** side of
//! `state.gpu_lock` (so it can run concurrently with regular inference), and
//! never holds it across the suite-wide loop, only per-generation.

pub mod datasets;
pub mod executor;
pub mod generator;
pub mod judgments;
pub mod queue;
pub mod registry;
pub mod rerun;
pub mod synthesis_driver;
pub mod util;
pub mod worker;

pub use datasets::{DatasetError, DatasetFormat, DatasetManifest, DatasetRegistry, DatasetStats};
pub use executor::{EvalExecutionError, run_suite_against_adapter};
pub use generator::{EvalCompletion, EvalGenerator, MockEvalGenerator, generator_from_state};
pub use judgments::{
    CompilationError, JudgmentError, JudgmentManifest, JudgmentMessage, JudgmentRow, JudgmentStore,
    JudgmentWinner, build_validation_suite, compile_judgments_to_sft, format_judge_prompt,
};
pub use queue::{
    EvalJobInfo, EvalJobs, EvalQueue, EvalQueueEntry, EvalSubmissionKind, QueuedEvalJob,
    SharedEvalQueue, new_shared_eval_queue,
};
pub use registry::{SuiteRegistry, SuiteRegistryError};
pub use rerun::rerun_filtered_suite;
pub use synthesis_driver::{
    SharedDatasetRegistry, SynthesisDriverError, SynthesisOutcome, SynthesisPreview,
    preview_synthesis, synthesize_and_save,
};
pub use worker::spawn_eval_worker;
