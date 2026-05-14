//! Re-run helper: take a registered suite and a list of example IDs,
//! and produce a new inline `EvalSuite` containing only those examples.
//! The UI uses this for "re-run failed examples" — pull only the ones
//! that didn't pass and queue an inline-suite eval against the same
//! adapter, no need for a new on-disk suite.

use std::collections::HashSet;

use kiln_eval::EvalSuite;

use crate::eval::registry::{SuiteRegistry, SuiteRegistryError};

#[derive(Debug, thiserror::Error)]
pub enum RerunError {
    #[error("registry: {0}")]
    Registry(#[from] SuiteRegistryError),
    #[error("no example IDs requested")]
    Empty,
    #[error("none of the requested example IDs matched the suite")]
    NoMatches,
}

/// Build an inline `EvalSuite` containing the subset of `suite_name`'s
/// examples whose `resolved_id()` is in `example_ids`. Carries every
/// other suite-level setting through unchanged so the re-run scores the
/// exact same way the original did.
pub fn rerun_filtered_suite(
    suites: &SuiteRegistry,
    suite_name: &str,
    example_ids: &[String],
) -> Result<EvalSuite, RerunError> {
    if example_ids.is_empty() {
        return Err(RerunError::Empty);
    }
    let suite = suites.load(suite_name)?;
    let want: HashSet<&str> = example_ids.iter().map(String::as_str).collect();
    let mut filtered: Vec<_> = suite
        .examples
        .iter()
        .filter(|ex| want.contains(ex.resolved_id().as_str()))
        .cloned()
        .collect();
    if filtered.is_empty() {
        return Err(RerunError::NoMatches);
    }
    // Stable order: same as the source suite (already preserved by .iter()).
    // Tag-tag the rerun so downstream metrics distinguish it.
    for ex in &mut filtered {
        ex.tags.push("synth:rerun".into());
    }
    Ok(EvalSuite {
        name: format!("{suite_name}-rerun"),
        description: Some(format!(
            "Filtered re-run of `{suite_name}` ({} of {} examples)",
            filtered.len(),
            suite.examples.len()
        )),
        default_scorer: suite.default_scorer.clone(),
        generation: suite.generation.clone(),
        system_prompt: suite.system_prompt.clone(),
        examples: filtered,
        schema_version: suite.schema_version,
        tools: suite.tools.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_eval::scorers::{NumericTolerance, Scorer};
    use kiln_eval::{EvalChatMessage, EvalExample, EvalGenerationParams};
    use tempfile::tempdir;

    fn mk_suite(name: &str, n: usize) -> EvalSuite {
        let examples = (0..n)
            .map(|i| EvalExample {
                id: Some(format!("e{i}")),
                messages: vec![EvalChatMessage::new("user", format!("Q{i}"))],
                target: Some(format!("{i}")),
                ..Default::default()
            })
            .collect();
        EvalSuite {
            name: name.into(),
            description: None,
            default_scorer: Scorer::NumericTolerance(NumericTolerance::default()),
            generation: EvalGenerationParams::default(),
            system_prompt: None,
            examples,
            schema_version: 1,
            tools: None,
        }
    }

    fn registry() -> (tempfile::TempDir, SuiteRegistry) {
        let dir = tempdir().unwrap();
        let reg = SuiteRegistry::new(dir.path().to_path_buf());
        (dir, reg)
    }

    #[test]
    fn rerun_picks_only_requested_ids() {
        let (_dir, reg) = registry();
        reg.save(&mk_suite("math", 5), false).unwrap();
        let r = rerun_filtered_suite(&reg, "math", &["e1".into(), "e3".into()]).unwrap();
        let ids: Vec<_> = r.examples.iter().map(|e| e.id.clone().unwrap()).collect();
        assert_eq!(ids, vec!["e1", "e3"]);
        assert_eq!(r.name, "math-rerun");
        assert!(r.examples.iter().all(|e| e.tags.contains(&"synth:rerun".to_string())));
    }

    #[test]
    fn rerun_empty_ids_errors() {
        let (_dir, reg) = registry();
        reg.save(&mk_suite("math", 2), false).unwrap();
        assert!(matches!(
            rerun_filtered_suite(&reg, "math", &[]),
            Err(RerunError::Empty)
        ));
    }

    #[test]
    fn rerun_unknown_suite_errors() {
        let (_dir, reg) = registry();
        let err = rerun_filtered_suite(&reg, "missing", &["x".into()]).unwrap_err();
        assert!(matches!(err, RerunError::Registry(_)));
    }

    #[test]
    fn rerun_no_matches_errors() {
        let (_dir, reg) = registry();
        reg.save(&mk_suite("math", 3), false).unwrap();
        let err = rerun_filtered_suite(&reg, "math", &["e99".into()]).unwrap_err();
        assert!(matches!(err, RerunError::NoMatches));
    }

    #[test]
    fn rerun_carries_through_default_scorer_and_generation() {
        let (_dir, reg) = registry();
        let mut suite = mk_suite("math", 2);
        suite.system_prompt = Some("You are a calculator.".into());
        suite.generation.temperature = 0.0;
        reg.save(&suite, false).unwrap();
        let r = rerun_filtered_suite(&reg, "math", &["e0".into()]).unwrap();
        assert_eq!(r.system_prompt.as_deref(), Some("You are a calculator."));
        assert_eq!(r.generation.temperature, 0.0);
    }
}
