//! Filesystem-backed registry of eval suites.
//!
//! Suites live under `<eval_dir>/<name>/`. Each suite directory contains
//! either a single `suite.json` document (inline examples) or a
//! `suite.json` header file plus an `examples.jsonl` sidecar that the
//! `EvalSuite::load_jsonl` loader walks line by line. Loading is best-effort
//! per suite — a malformed suite is logged at WARN and skipped, so one bad
//! file doesn't take the whole registry offline.

use std::path::{Path, PathBuf};

use kiln_eval::{EvalSuite, EvalSuiteSummary, suite::SuiteLoadError};

/// Disk layout: `<eval_dir>/<name>/{suite.json, examples.jsonl?}`.
pub struct SuiteRegistry {
    root: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum SuiteRegistryError {
    #[error("suite not found: {0}")]
    NotFound(String),
    #[error("invalid suite name: {0}")]
    InvalidName(String),
    #[error("suite already exists: {0}")]
    AlreadyExists(String),
    #[error("io: {0}")]
    Io(String),
    #[error("load: {0}")]
    Load(String),
}

impl From<SuiteLoadError> for SuiteRegistryError {
    fn from(e: SuiteLoadError) -> Self {
        SuiteRegistryError::Load(format!("{e}"))
    }
}

impl SuiteRegistry {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Load a suite by name. Tries `suite.json` first; falls back to
    /// `suite.json` + `examples.jsonl` JSONL mode.
    pub fn load(&self, name: &str) -> Result<EvalSuite, SuiteRegistryError> {
        Self::validate_name(name)?;
        let dir = self.root.join(name);
        if !dir.exists() {
            return Err(SuiteRegistryError::NotFound(name.to_string()));
        }
        let header = dir.join("suite.json");
        let jsonl = dir.join("examples.jsonl");
        if !header.exists() {
            return Err(SuiteRegistryError::NotFound(format!(
                "{name}/suite.json"
            )));
        }
        if jsonl.exists() {
            Ok(EvalSuite::load_jsonl(&header, &jsonl)?)
        } else {
            Ok(EvalSuite::load_json(&header)?)
        }
    }

    /// Save a suite to disk. `force` overwrites an existing suite.
    pub fn save(&self, suite: &EvalSuite, force: bool) -> Result<PathBuf, SuiteRegistryError> {
        Self::validate_name(&suite.name)?;
        let dir = self.root.join(&suite.name);
        if dir.exists() {
            if !force {
                return Err(SuiteRegistryError::AlreadyExists(suite.name.clone()));
            }
        } else {
            std::fs::create_dir_all(&dir).map_err(|e| SuiteRegistryError::Io(format!("{e}")))?;
        }
        let header = dir.join("suite.json");
        let json = serde_json::to_string_pretty(suite)
            .map_err(|e| SuiteRegistryError::Io(format!("serialize: {e}")))?;
        // Atomic write: write to a tempfile in the same directory and rename
        // over the destination so a crash mid-write doesn't corrupt the file.
        let tmp = dir.join("suite.json.tmp");
        std::fs::write(&tmp, json).map_err(|e| SuiteRegistryError::Io(format!("{e}")))?;
        std::fs::rename(&tmp, &header).map_err(|e| SuiteRegistryError::Io(format!("{e}")))?;
        // Existing examples.jsonl sidecars are cleared when we save a new
        // header that already carries inline examples (otherwise the
        // loader would concatenate them).
        let jsonl = dir.join("examples.jsonl");
        if !suite.examples.is_empty() && jsonl.exists() {
            let _ = std::fs::remove_file(&jsonl);
        }
        Ok(header)
    }

    /// Delete a registered suite.
    pub fn delete(&self, name: &str) -> Result<(), SuiteRegistryError> {
        Self::validate_name(name)?;
        let dir = self.root.join(name);
        if !dir.exists() {
            return Err(SuiteRegistryError::NotFound(name.to_string()));
        }
        std::fs::remove_dir_all(&dir).map_err(|e| SuiteRegistryError::Io(format!("{e}")))?;
        Ok(())
    }

    /// Walk the registry directory and return a summary for every loadable
    /// suite. Malformed entries are logged at WARN and skipped.
    pub fn list(&self) -> Vec<EvalSuiteSummary> {
        let mut out = Vec::new();
        let read_dir = match std::fs::read_dir(&self.root) {
            Ok(rd) => rd,
            Err(_) => return out,
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            // Skip hidden directories.
            if name.starts_with('.') {
                continue;
            }
            match self.load(&name) {
                Ok(s) => out.push(s.summary()),
                Err(e) => {
                    tracing::warn!(suite = %name, error = %e, "skipping malformed suite");
                }
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    fn validate_name(name: &str) -> Result<(), SuiteRegistryError> {
        if !crate::eval::util::is_valid_segment_name(name) {
            return Err(SuiteRegistryError::InvalidName(name.to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_eval::scorers::Scorer;
    use kiln_eval::{EvalChatMessage, EvalExample, EvalGenerationParams};

    fn mk_suite(name: &str) -> EvalSuite {
        EvalSuite {
            name: name.into(),
            description: None,
            default_scorer: Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            generation: EvalGenerationParams::default(),
            system_prompt: None,
            examples: vec![EvalExample {
                id: Some("e1".into()),
                messages: vec![EvalChatMessage::new("user", "x")],
                target: Some("y".into()),
                tags: vec!["smoke".into()],
                ..Default::default()
            }],
            schema_version: 1,
            tools: None,
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let reg = SuiteRegistry::new(dir.path().to_path_buf());
        let suite = mk_suite("math");
        reg.save(&suite, false).unwrap();
        let loaded = reg.load("math").unwrap();
        assert_eq!(loaded.name, "math");
        assert_eq!(loaded.examples.len(), 1);
    }

    #[test]
    fn duplicate_save_fails_without_force() {
        let dir = tempfile::tempdir().unwrap();
        let reg = SuiteRegistry::new(dir.path().to_path_buf());
        let suite = mk_suite("dup");
        reg.save(&suite, false).unwrap();
        let err = reg.save(&suite, false).unwrap_err();
        assert!(matches!(err, SuiteRegistryError::AlreadyExists(_)));
        reg.save(&suite, true).unwrap();
    }

    #[test]
    fn list_returns_summaries() {
        let dir = tempfile::tempdir().unwrap();
        let reg = SuiteRegistry::new(dir.path().to_path_buf());
        reg.save(&mk_suite("a"), false).unwrap();
        reg.save(&mk_suite("b"), false).unwrap();
        let summaries = reg.list();
        let names: Vec<_> = summaries.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(summaries[0].tags.get("smoke"), Some(&1));
    }

    #[test]
    fn invalid_names_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let reg = SuiteRegistry::new(dir.path().to_path_buf());
        for bad in &["", "..", "a/b", "a\\b"] {
            let mut s = mk_suite("ok");
            s.name = (*bad).into();
            let err = reg.save(&s, false).unwrap_err();
            assert!(matches!(err, SuiteRegistryError::InvalidName(_)));
        }
    }

    #[test]
    fn delete_removes_suite() {
        let dir = tempfile::tempdir().unwrap();
        let reg = SuiteRegistry::new(dir.path().to_path_buf());
        reg.save(&mk_suite("doomed"), false).unwrap();
        reg.delete("doomed").unwrap();
        assert!(matches!(
            reg.load("doomed").unwrap_err(),
            SuiteRegistryError::NotFound(_)
        ));
    }

    #[test]
    fn list_skips_malformed_suite_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let reg = SuiteRegistry::new(dir.path().to_path_buf());
        reg.save(&mk_suite("good"), false).unwrap();
        // Create a malformed sibling.
        let bad = dir.path().join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("suite.json"), "{ not json ").unwrap();
        let summaries = reg.list();
        let names: Vec<_> = summaries.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["good"]);
    }
}
