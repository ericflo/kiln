//! Driver that ties the dataset registry + the `kiln_eval::synthesis`
//! pipeline + the on-disk suite registry into a single user-facing
//! operation.
//!
//! Two operations:
//!
//! - `preview_synthesis(dataset, config, head_n)` — synthesize only the
//!   first `head_n` conversations and return the candidate examples
//!   without persisting anything. Used by the UI preview to show users
//!   what they'll get *before* they commit.
//! - `synthesize_and_save(dataset, config)` — full-dataset synthesis +
//!   register the result in the suite registry. Returns both the suite
//!   summary and the synthesis stats so the caller can render a
//!   "synthesized N examples from M trajectories" line.

use std::sync::Arc;

use kiln_eval::synthesis::{
    SynthesisConfig, SynthesisError, SynthesisStats, synthesize_suite,
};
use kiln_eval::{EvalExample, EvalSuiteSummary};
use serde::Serialize;

use crate::eval::datasets::{DatasetError, DatasetRegistry};
use crate::eval::registry::{SuiteRegistry, SuiteRegistryError};

#[derive(Debug, thiserror::Error)]
pub enum SynthesisDriverError {
    #[error("dataset: {0}")]
    Dataset(#[from] DatasetError),
    #[error("synthesis: {0}")]
    Synthesis(#[from] SynthesisError),
    #[error("registry: {0}")]
    Registry(#[from] SuiteRegistryError),
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthesisPreview {
    pub examples: Vec<EvalExample>,
    pub stats: SynthesisStats,
    pub suite_name: String,
    /// Snapshot of the default scorer that *would* be persisted on commit.
    /// Useful for the UI to show "we'd grade these as `numeric_tolerance`".
    pub default_scorer_kind: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthesisOutcome {
    pub suite: EvalSuiteSummary,
    pub stats: SynthesisStats,
}

pub fn preview_synthesis(
    datasets: &DatasetRegistry,
    dataset_name: &str,
    config: &SynthesisConfig,
    head_n: usize,
) -> Result<SynthesisPreview, SynthesisDriverError> {
    let convs = datasets.head_sft(dataset_name, head_n.max(1))?;
    let (suite, stats) = synthesize_suite(convs, config)?;
    // Truncate examples for preview UI — the SUIte file itself is
    // already small at this stage but the front-end only renders 5.
    let preview = SynthesisPreview {
        examples: suite.examples.into_iter().take(10).collect(),
        stats,
        suite_name: config.suite_name.clone(),
        default_scorer_kind: suite.default_scorer.kind_label(),
    };
    Ok(preview)
}

pub fn synthesize_and_save(
    datasets: &DatasetRegistry,
    suites: &SuiteRegistry,
    dataset_name: &str,
    config: &SynthesisConfig,
    force: bool,
) -> Result<SynthesisOutcome, SynthesisDriverError> {
    let convs = datasets.iter_sft(dataset_name)?;
    let (mut suite, stats) = synthesize_suite(convs, config)?;
    // Tag the suite metadata with where it came from so users can audit
    // later.
    let mut description = config.description.clone().unwrap_or_default();
    if !description.is_empty() {
        description.push_str("\n\n");
    }
    description.push_str(&format!(
        "Synthesized from dataset `{dataset_name}` using strategy `{:?}` ({} examples from {} trajectories, seed={}).",
        config.strategy, stats.examples_generated, stats.trajectories_used, stats.effective_seed
    ));
    suite.description = Some(description);
    suites.save(&suite, force)?;
    Ok(SynthesisOutcome {
        suite: suite.summary(),
        stats,
    })
}

/// Convenience alias so the API layer can hold a `Arc` without dragging the
/// concrete type around.
pub type SharedDatasetRegistry = Arc<DatasetRegistry>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::datasets::DatasetFormat;
    use kiln_eval::synthesis::{Sampling, SynthesisStrategy};
    use tempfile::tempdir;

    fn rows() -> String {
        let r = |role: &str, content: &str| {
            serde_json::json!({"role": role, "content": content})
        };
        let convs = vec![
            serde_json::json!({"messages": [r("user", "1+1?"), r("assistant", "2")]}),
            serde_json::json!({"messages": [r("user", "2+2?"), r("assistant", "4")]}),
            serde_json::json!({"messages": [r("user", "3+3?"), r("assistant", "6")]}),
        ];
        convs
            .into_iter()
            .map(|c| serde_json::to_string(&c).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn cfg(name: &str) -> SynthesisConfig {
        SynthesisConfig {
            suite_name: name.into(),
            description: None,
            strategy: SynthesisStrategy::FinalAssistant,
            scorer: kiln_eval::synthesis::ScorerChoice::AutoDetect,
            generation: Default::default(),
            sampling: Sampling {
                max_examples: Some(10),
                max_prompt_chars: 1_000_000,
                max_target_chars: 1_000_000,
                seed: Some(42),
                dedupe: true,
            },
            system_prompt: None,
            strip_system_prompt: false,
        }
    }

    fn registries(dir: &std::path::Path) -> (DatasetRegistry, SuiteRegistry) {
        std::fs::create_dir_all(dir.join("datasets")).unwrap();
        std::fs::create_dir_all(dir.join("suites")).unwrap();
        (
            DatasetRegistry::new(dir.join("datasets")),
            SuiteRegistry::new(dir.join("suites")),
        )
    }

    #[test]
    fn preview_returns_examples_without_persisting() {
        let d = tempdir().unwrap();
        let (datasets, _suites) = registries(d.path());
        datasets
            .create("math", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap();
        let preview = preview_synthesis(&datasets, "math", &cfg("preview-suite"), 2).unwrap();
        // Preview head_n=2 + 2 example conversations → 2 examples
        assert_eq!(preview.examples.len(), 2);
        assert_eq!(preview.suite_name, "preview-suite");
    }

    #[test]
    fn synthesize_and_save_writes_suite() {
        let d = tempdir().unwrap();
        let (datasets, suites) = registries(d.path());
        datasets
            .create("math", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap();
        let outcome = synthesize_and_save(&datasets, &suites, "math", &cfg("real-suite"), false).unwrap();
        assert_eq!(outcome.suite.name, "real-suite");
        assert!(outcome.stats.examples_generated >= 1);
        let listed = suites.list();
        assert_eq!(listed.len(), 1);
    }

    #[test]
    fn synthesize_force_overwrites() {
        let d = tempdir().unwrap();
        let (datasets, suites) = registries(d.path());
        datasets
            .create("math", DatasetFormat::SftChat, None, rows().as_bytes())
            .unwrap();
        synthesize_and_save(&datasets, &suites, "math", &cfg("once"), false).unwrap();
        let err = synthesize_and_save(&datasets, &suites, "math", &cfg("once"), false).unwrap_err();
        assert!(matches!(err, SynthesisDriverError::Registry(_)));
        synthesize_and_save(&datasets, &suites, "math", &cfg("once"), true).unwrap();
    }
}
