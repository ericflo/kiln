//! Judgments — append-only datasets of user A/B preferences over model
//! completions, plus the compiler that turns them into SFT data for
//! training a local judge LoRA. No frontier-LLM calls anywhere in this
//! pipeline; the judge is always a kiln-local adapter (or the base model).
//!
//! Disk layout: `<adapter_dir>/.eval/judgments/<name>/`
//! - `judgments.jsonl` — one `JudgmentRow` per line.
//! - `manifest.json` — counts + provenance.
//!
//! The cycle:
//!
//! 1. Pick a prompt set + 2 adapters (or 1 adapter + 2 temperatures).
//! 2. `kiln serve` generates pairs of completions via the existing batched
//!    completion API.
//! 3. User clicks A / B / Tie / Skip in the UI; each click POSTs one
//!    `JudgmentRow` here.
//! 4. `compile_to_sft(name)` writes a sister SFT dataset whose target is
//!    `"A"` / `"B"` / `"tie"` — train a judge LoRA on it.
//! 5. Use the judge LoRA via `Scorer::LlmJudge { judge_adapter }`. Held-out
//!    rows score the judge LoRA's accuracy.
//! 6. When the LoRA mis-judges, `remove_row` + retrain refines it. The
//!    dataset is the asset; the LoRA is derived.

use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::eval::datasets::{DatasetFormat, DatasetRegistry};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JudgmentWinner {
    A,
    B,
    Tie,
    Skip,
}

/// Chat message shape persisted on judgment rows. Aliased to
/// `kiln_eval::EvalChatMessage` so we don't carry two parallel definitions
/// of the same `{role, content}` shape — the judgment compiler builds
/// `EvalExample` from these directly downstream.
pub type JudgmentMessage = kiln_eval::EvalChatMessage;

/// One row in the judgments dataset. Stable wire shape — version this if
/// you ever need to change it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentRow {
    /// Stable ID per judgment (allows correction edits).
    pub id: String,
    /// The exact messages the user was shown (with the model's reply
    /// stripped — that lives in `response_a` / `response_b`).
    pub prompt: Vec<JudgmentMessage>,
    /// Adapter that produced `response_a` (empty string = base model).
    pub adapter_a: Option<String>,
    pub adapter_b: Option<String>,
    pub response_a: String,
    pub response_b: String,
    pub winner: JudgmentWinner,
    /// Free-form user note explaining the pick. Carried into SFT
    /// distillation so the judge LoRA learns the *why*, not just the *what*.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Tags the user attached (`"prose"`, `"tool_call"`, `"code_style"`, …)
    /// — used to slice metrics in the UI.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    pub submitted_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentManifest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub num_rows: u64,
    pub created_at: String,
    pub updated_at: String,
    /// Winner histogram. Pure telemetry, also helps the UI spot bias
    /// ("you've picked A in 90% of cases — is something off?").
    #[serde(default)]
    pub winner_histogram: std::collections::BTreeMap<String, u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum JudgmentError {
    #[error("io: {0}")]
    Io(String),
    #[error("invalid name: {0}")]
    InvalidName(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("already exists: {0}")]
    AlreadyExists(String),
}

pub struct JudgmentStore {
    root: PathBuf,
}

impl JudgmentStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn ensure_root(&self) -> Result<(), JudgmentError> {
        std::fs::create_dir_all(&self.root).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        Ok(())
    }

    pub fn dir(&self, name: &str) -> Result<PathBuf, JudgmentError> {
        validate_name(name)?;
        Ok(self.root.join(name))
    }

    pub fn create(&self, name: &str, description: Option<String>) -> Result<JudgmentManifest, JudgmentError> {
        self.ensure_root()?;
        let dir = self.dir(name)?;
        if dir.exists() {
            return Err(JudgmentError::AlreadyExists(name.to_string()));
        }
        std::fs::create_dir_all(&dir).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        let now = chrono::Utc::now().to_rfc3339();
        // Touch the data file.
        std::fs::write(dir.join("judgments.jsonl"), "")
            .map_err(|e| JudgmentError::Io(format!("{e}")))?;
        let manifest = JudgmentManifest {
            name: name.to_string(),
            description,
            num_rows: 0,
            created_at: now.clone(),
            updated_at: now,
            winner_histogram: Default::default(),
        };
        self.write_manifest(name, &manifest)?;
        Ok(manifest)
    }

    pub fn append(&self, name: &str, row: &JudgmentRow) -> Result<JudgmentManifest, JudgmentError> {
        let dir = self.dir(name)?;
        if !dir.exists() {
            return Err(JudgmentError::NotFound(name.to_string()));
        }
        let path = dir.join("judgments.jsonl");
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| JudgmentError::Io(format!("{e}")))?;
        let line = serde_json::to_string(row).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        writeln!(f, "{line}").map_err(|e| JudgmentError::Io(format!("{e}")))?;
        f.sync_all().map_err(|e| JudgmentError::Io(format!("{e}")))?;
        let mut manifest = self.load_manifest(name)?;
        manifest.num_rows += 1;
        manifest.updated_at = chrono::Utc::now().to_rfc3339();
        let key = match row.winner {
            JudgmentWinner::A => "a",
            JudgmentWinner::B => "b",
            JudgmentWinner::Tie => "tie",
            JudgmentWinner::Skip => "skip",
        };
        *manifest
            .winner_histogram
            .entry(key.to_string())
            .or_default() += 1;
        self.write_manifest(name, &manifest)?;
        Ok(manifest)
    }

    pub fn remove(&self, name: &str, judgment_id: &str) -> Result<JudgmentManifest, JudgmentError> {
        let dir = self.dir(name)?;
        if !dir.exists() {
            return Err(JudgmentError::NotFound(name.to_string()));
        }
        let path = dir.join("judgments.jsonl");
        let tmp = dir.join("judgments.jsonl.tmp");
        let f = std::fs::File::open(&path).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        let mut out = std::fs::File::create(&tmp).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        use std::io::{BufRead, BufReader};
        let mut removed = 0u32;
        for line in BufReader::new(f).lines() {
            let line = line.map_err(|e| JudgmentError::Io(format!("{e}")))?;
            if line.trim().is_empty() {
                continue;
            }
            let row: JudgmentRow = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => {
                    writeln!(out, "{line}").map_err(|e| JudgmentError::Io(format!("{e}")))?;
                    continue;
                }
            };
            if row.id == judgment_id {
                removed += 1;
                continue;
            }
            writeln!(out, "{line}").map_err(|e| JudgmentError::Io(format!("{e}")))?;
        }
        out.sync_all().map_err(|e| JudgmentError::Io(format!("{e}")))?;
        drop(out);
        std::fs::rename(&tmp, &path).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        if removed == 0 {
            return Err(JudgmentError::NotFound(format!("{name}/{judgment_id}")));
        }
        let mut manifest = self.load_manifest(name)?;
        manifest.num_rows = manifest.num_rows.saturating_sub(removed as u64);
        manifest.updated_at = chrono::Utc::now().to_rfc3339();
        self.write_manifest(name, &manifest)?;
        Ok(manifest)
    }

    pub fn delete(&self, name: &str) -> Result<(), JudgmentError> {
        let dir = self.dir(name)?;
        if !dir.exists() {
            return Err(JudgmentError::NotFound(name.to_string()));
        }
        std::fs::remove_dir_all(&dir).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        Ok(())
    }

    pub fn list(&self) -> Vec<JudgmentManifest> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.root) else {
            return out;
        };
        for entry in rd.flatten() {
            let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            if let Ok(m) = self.load_manifest(&name) {
                out.push(m);
            }
        }
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        out
    }

    pub fn load_manifest(&self, name: &str) -> Result<JudgmentManifest, JudgmentError> {
        let path = self.dir(name)?.join("manifest.json");
        let bytes = std::fs::read(&path).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        serde_json::from_slice(&bytes).map_err(|e| JudgmentError::Io(format!("manifest parse: {e}")))
    }

    fn write_manifest(&self, name: &str, manifest: &JudgmentManifest) -> Result<(), JudgmentError> {
        let dir = self.dir(name)?;
        let path = dir.join("manifest.json");
        let tmp = dir.join("manifest.json.tmp");
        let body = serde_json::to_string_pretty(manifest)
            .map_err(|e| JudgmentError::Io(format!("manifest serialize: {e}")))?;
        std::fs::write(&tmp, body).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        std::fs::rename(&tmp, &path).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        Ok(())
    }

    pub fn iter_rows(&self, name: &str) -> Result<JudgmentIter, JudgmentError> {
        let path = self.dir(name)?.join("judgments.jsonl");
        let f = std::fs::File::open(&path).map_err(|e| JudgmentError::Io(format!("{e}")))?;
        Ok(JudgmentIter {
            reader: std::io::BufReader::new(f),
        })
    }
}

pub struct JudgmentIter {
    reader: std::io::BufReader<std::fs::File>,
}

impl Iterator for JudgmentIter {
    type Item = JudgmentRow;
    fn next(&mut self) -> Option<Self::Item> {
        use std::io::BufRead;
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).ok()?;
            if n == 0 {
                return None;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(row) = serde_json::from_str::<JudgmentRow>(trimmed) {
                return Some(row);
            }
        }
    }
}

/// Compile a judgments dataset into an SFT chat dataset suitable for
/// training a judge LoRA. Each row becomes one SFT example where the
/// assistant target is `A` / `B` / `tie` and the user message is the
/// pairwise judging prompt.
///
/// The compiled dataset is written into the same `DatasetRegistry` that
/// powers training, under the name `<judgment_name>-sft`. Train it via
/// the regular `/v1/train/sft` flow.
pub fn compile_judgments_to_sft(
    judgments: &JudgmentStore,
    datasets: &DatasetRegistry,
    name: &str,
    output_dataset: &str,
    include_skips: bool,
) -> Result<u64, CompilationError> {
    let rows = judgments
        .iter_rows(name)
        .map_err(|e| CompilationError::Judgment(format!("{e}")))?;
    let mut compiled = String::new();
    let mut compiled_rows = 0u64;
    for row in rows {
        if matches!(row.winner, JudgmentWinner::Skip) && !include_skips {
            continue;
        }
        let winner_label = match row.winner {
            JudgmentWinner::A => "A",
            JudgmentWinner::B => "B",
            JudgmentWinner::Tie => "Tie",
            JudgmentWinner::Skip => continue,
        };
        let user_prompt = format_judge_prompt(&row);
        let target_text = if let Some(note) = row.note.as_deref().filter(|n| !n.is_empty()) {
            format!("Winner: {winner_label}\nReason: {note}")
        } else {
            format!("Winner: {winner_label}")
        };
        let sft = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are an impartial judge. Pick the better assistant reply. Output `Winner: A`, `Winner: B`, or `Winner: Tie`. Optionally add `Reason:` on the next line."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": target_text},
            ],
        });
        compiled.push_str(&sft.to_string());
        compiled.push('\n');
        compiled_rows += 1;
    }
    if compiled_rows == 0 {
        return Err(CompilationError::Empty);
    }
    datasets
        .create(
            output_dataset,
            DatasetFormat::SftChat,
            Some(format!(
                "Compiled from {compiled_rows} judgments in `{name}` for training a local judge LoRA."
            )),
            compiled.as_bytes(),
        )
        .map_err(|e| CompilationError::Dataset(format!("{e}")))?;
    Ok(compiled_rows)
}

#[derive(Debug, thiserror::Error)]
pub enum CompilationError {
    #[error("judgment store: {0}")]
    Judgment(String),
    #[error("dataset registry: {0}")]
    Dataset(String),
    #[error("no judgments to compile (all skipped?)")]
    Empty,
}

/// Format a single judgment row into the standard pairwise-judging prompt
/// used in the SFT compilation. Shared with the live judge runner so the
/// model sees the same shape at training and inference time.
pub fn format_judge_prompt(row: &JudgmentRow) -> String {
    let mut prompt = String::new();
    prompt.push_str("Compare the following two assistant replies to this conversation:\n\n");
    prompt.push_str("# Conversation\n");
    for msg in &row.prompt {
        prompt.push_str(&format!("[{}] {}\n", msg.role, msg.content));
    }
    prompt.push_str("\n# Reply A\n");
    prompt.push_str(&row.response_a);
    prompt.push_str("\n\n# Reply B\n");
    prompt.push_str(&row.response_b);
    prompt.push_str("\n\nWhich reply is better? Reply with `Winner: A`, `Winner: B`, or `Winner: Tie`.");
    prompt
}

/// Evaluate a judge LoRA against the held-out portion of a judgments
/// dataset. Returns accuracy as a pure number — the caller (the HTTP API
/// or the auto-eval pipeline) can wrap it into the standard `EvalResult`.
///
/// Held-out = the last `holdout_n` rows. Order in the JSONL is the order
/// the user submitted them, so the most recent judgments naturally form
/// the validation set (a common pattern when the user is bootstrapping a
/// judge LoRA and wants to test against fresh data).
pub fn build_validation_suite(
    judgments: &JudgmentStore,
    name: &str,
    holdout_n: usize,
) -> Result<kiln_eval::EvalSuite, JudgmentError> {
    let all: Vec<JudgmentRow> = judgments.iter_rows(name)?.collect();
    if all.len() < 2 {
        return Err(JudgmentError::NotFound(format!(
            "{name}: need at least 2 judgments to build a validation suite"
        )));
    }
    let split = all.len().saturating_sub(holdout_n.max(1));
    let holdout = &all[split..];
    let examples: Vec<kiln_eval::EvalExample> = holdout
        .iter()
        .map(|row| {
            let target = match row.winner {
                JudgmentWinner::A => "A".to_string(),
                JudgmentWinner::B => "B".to_string(),
                JudgmentWinner::Tie => "Tie".to_string(),
                JudgmentWinner::Skip => return None,
            };
            Some(kiln_eval::EvalExample {
                id: Some(row.id.clone()),
                messages: vec![
                    kiln_eval::EvalChatMessage::new(
                        "system",
                        "You are an impartial judge. Pick the better assistant reply. Output `Winner: A`, `Winner: B`, or `Winner: Tie`.",
                    ),
                    kiln_eval::EvalChatMessage::new("user", format_judge_prompt(row)),
                ],
                target: Some(target),
                tags: row.tags.clone(),
                scorer: Some(kiln_eval::Scorer::Regex {
                    pattern: r"(?i)Winner:\s*(A|B|Tie)".into(),
                    capture_group: Some(1),
                    case_sensitive: false,
                }),
                generation: Some(kiln_eval::EvalGenerationParams {
                    temperature: 0.0,
                    max_tokens: 64,
                    ..Default::default()
                }),
                ..Default::default()
            })
        })
        .filter_map(|e| e)
        .collect();
    if examples.is_empty() {
        return Err(JudgmentError::NotFound(format!(
            "{name}: no scorable judgments in hold-out (all skipped?)"
        )));
    }
    Ok(kiln_eval::EvalSuite {
        name: format!("judge-validate-{name}"),
        description: Some(format!(
            "Held-out validation suite generated from `{name}` ({} examples)",
            examples.len()
        )),
        default_scorer: kiln_eval::Scorer::Regex {
            pattern: r"(?i)Winner:\s*(A|B|Tie)".into(),
            capture_group: Some(1),
            case_sensitive: false,
        },
        generation: kiln_eval::EvalGenerationParams {
            temperature: 0.0,
            max_tokens: 64,
            ..Default::default()
        },
        system_prompt: None,
        examples,
        schema_version: 1,
        tools: None,
    })
}

fn validate_name(name: &str) -> Result<(), JudgmentError> {
    if !crate::eval::util::is_valid_segment_name(name) {
        return Err(JudgmentError::InvalidName(name.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use JudgmentMessage;

    fn row(id: &str, winner: JudgmentWinner) -> JudgmentRow {
        JudgmentRow {
            id: id.into(),
            prompt: vec![JudgmentMessage::new("user", "Q?")],
            adapter_a: None,
            adapter_b: Some("v1".into()),
            response_a: "answer A".into(),
            response_b: "answer B".into(),
            winner,
            note: Some("A is clearer".into()),
            tags: vec!["prose".into()],
            submitted_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    #[test]
    fn create_and_append() {
        let d = tempfile::tempdir().unwrap();
        let store = JudgmentStore::new(d.path().to_path_buf());
        let m = store.create("prose-judge", None).unwrap();
        assert_eq!(m.num_rows, 0);
        let m = store.append("prose-judge", &row("j1", JudgmentWinner::A)).unwrap();
        assert_eq!(m.num_rows, 1);
        assert_eq!(m.winner_histogram.get("a"), Some(&1));
    }

    #[test]
    fn remove_by_id() {
        let d = tempfile::tempdir().unwrap();
        let store = JudgmentStore::new(d.path().to_path_buf());
        store.create("p", None).unwrap();
        store.append("p", &row("j1", JudgmentWinner::A)).unwrap();
        store.append("p", &row("j2", JudgmentWinner::B)).unwrap();
        let m = store.remove("p", "j1").unwrap();
        assert_eq!(m.num_rows, 1);
        let err = store.remove("p", "nope").unwrap_err();
        assert!(matches!(err, JudgmentError::NotFound(_)));
    }

    #[test]
    fn compile_emits_sft_dataset() {
        let d = tempfile::tempdir().unwrap();
        let store = JudgmentStore::new(d.path().join("judgments"));
        let datasets = DatasetRegistry::new(d.path().join("datasets"));
        store.create("p", None).unwrap();
        store.append("p", &row("j1", JudgmentWinner::A)).unwrap();
        store.append("p", &row("j2", JudgmentWinner::B)).unwrap();
        store.append("p", &row("j3", JudgmentWinner::Tie)).unwrap();
        store.append("p", &row("j4", JudgmentWinner::Skip)).unwrap();
        let n = compile_judgments_to_sft(&store, &datasets, "p", "p-sft", false).unwrap();
        assert_eq!(n, 3);
        let listed = datasets.list();
        assert!(listed.iter().any(|m| m.name == "p-sft"));
    }

    #[test]
    fn validation_suite_uses_recent_rows_as_holdout() {
        let d = tempfile::tempdir().unwrap();
        let store = JudgmentStore::new(d.path().to_path_buf());
        store.create("p", None).unwrap();
        for i in 0..10 {
            let win = if i % 2 == 0 {
                JudgmentWinner::A
            } else {
                JudgmentWinner::B
            };
            store.append("p", &row(&format!("j{i}"), win)).unwrap();
        }
        let suite = build_validation_suite(&store, "p", 3).unwrap();
        assert_eq!(suite.examples.len(), 3);
        // Most-recent 3 rows are j7, j8, j9 (B, A, B).
        assert_eq!(suite.examples[0].id.as_deref(), Some("j7"));
        assert_eq!(suite.examples[1].id.as_deref(), Some("j8"));
        assert_eq!(suite.examples[2].id.as_deref(), Some("j9"));
    }

    #[test]
    fn duplicate_create_rejected() {
        let d = tempfile::tempdir().unwrap();
        let store = JudgmentStore::new(d.path().to_path_buf());
        store.create("p", None).unwrap();
        let err = store.create("p", None).unwrap_err();
        assert!(matches!(err, JudgmentError::AlreadyExists(_)));
    }
}
