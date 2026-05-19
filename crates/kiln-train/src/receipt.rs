//! Reproducibility receipts (grand plan §8.11).
//!
//! Every adapter kiln produces ships with a JSON receipt that records
//! exactly enough information to rebuild it:
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "adapter": "math-frontier",
//!   "produced_at": "2026-05-15T18:43:11Z",
//!   "kiln_version": "0.42.7",
//!   "kernel_versions": { "cuda": "...", "vulkan": "...", "metal": "..." },
//!   "seed": 4218,
//!   "teacher": {
//!     "alias": "qwen3.6-27b@openrouter",
//!     "model_id": "qwen/qwen-3.6-27b",
//!     "model_version_hash": "sha256:...",
//!     "snapshot_url": "..."
//!   },
//!   "prompts": {
//!     "source": "kiln-canonical:math_reasoning:v3",
//!     "manifest_hash": "sha256:..."
//!   },
//!   "hyperparameters": { ... },
//!   "diagnostic_summary": {
//!     "overlap_ratio_final": 0.92,
//!     "rep_rate_max": 0.01,
//!     "guardrail_triggers": []
//!   },
//!   "post_eval": { "math-frontier-eval": 0.71 }
//! }
//! ```
//!
//! Written to `<adapter_dir>/receipt.json` whenever a training job
//! completes. The CLI subcommand `kiln distill verify <adapter>`
//! re-runs the recipe and bit-checks (deterministic kernel paths,
//! same hardware) or evaluations-equivalent-within-1% (same recipe,
//! hardware drift).
//!
//! # Why this module lives in kiln-train, not kiln-server
//!
//! Receipts are produced at the moment the adapter is finalised —
//! which is the trainer's job. The server's role is to expose them
//! at `GET /v1/adapters/{name}/receipt` (and consume them at
//! `kiln distill verify`); the trainer is what _writes_ them.
//!
//! Schema versioning: `schema_version` starts at 1. Every additive
//! change keeps it; any breaking change bumps it and the CLI verify
//! command knows how to read older versions.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Current receipt schema version. Bumped only on breaking schema
/// changes; additive fields are accepted at the older version.
pub const RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Top-level reproducibility receipt for an adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterReceipt {
    /// Schema version. `1` at first release.
    pub schema_version: u32,
    /// Adapter name on disk.
    pub adapter: String,
    /// UTC RFC3339 timestamp at which the adapter was finalised.
    pub produced_at: String,
    /// Kiln binary version (`CARGO_PKG_VERSION`).
    pub kiln_version: String,
    /// Per-engine kernel revision strings — typically Cargo
    /// version of `kiln-opd-loss-kernel`, plus optional CUDA/Vulkan/
    /// Metal build IDs.
    pub kernel_versions: BTreeMap<String, String>,
    /// Deterministic seed used by the trainer (LoRA init + RNG).
    pub seed: u64,
    /// Source kind for this adapter. `sft`, `grpo`, `opd`,
    /// `distill_merge`, `distill_pump`, `distill_refresh`,
    /// `distill_self`, `judge_distill`, `recipe`.
    pub source_kind: String,
    /// Teacher description. `None` for SFT / GRPO adapters that
    /// didn't use a teacher.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub teacher: Option<TeacherDescriptor>,
    /// Prompt-source description (the seed corpus). `None` for
    /// adapters whose source is implicit (e.g. behaviour-space
    /// merges that reuse stored source prompts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptSourceDescriptor>,
    /// All trainer hyperparameters that affect output (lr, rank,
    /// alpha, top_k, top_p, temperature, etc.). Stored as a free
    /// JSON object so future config additions don't break older
    /// receipts.
    pub hyperparameters: serde_json::Value,
    /// Coarse summary of the run-time diagnostics. The full
    /// snapshot stream is in the adapter's run log; this is the
    /// post-mortem readable at a glance.
    pub diagnostic_summary: DiagnosticSummary,
    /// Eval scores against any post-training eval suites the run
    /// invoked. `name -> score`.
    pub post_eval: BTreeMap<String, f64>,
}

/// Teacher description for the receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherDescriptor {
    /// `/v1/teachers` alias used at run time
    /// (e.g. `qwen3.6-27b@openrouter`).
    pub alias: String,
    /// Provider's model id (e.g. `qwen/qwen-3.6-27b`).
    pub model_id: String,
    /// SHA256 hash of the teacher's weights or provider snapshot
    /// fingerprint. `None` when running against a hosted endpoint
    /// that doesn't expose this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version_hash: Option<String>,
    /// URL pointing at the teacher snapshot (HuggingFace, internal
    /// blob store, etc.). `None` when not applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot_url: Option<String>,
}

/// Prompt source description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSourceDescriptor {
    /// Canonical reference: `kiln-canonical:math_reasoning:v3`,
    /// `user-uploaded:<job_id>`, `pi-share-hf:<adapter_id>`, etc.
    pub source: String,
    /// SHA256 hash of the prompt-manifest file. The manifest itself
    /// lives at `<adapter_dir>/prompts.jsonl` for in-repo
    /// reproducibility; the hash here lets us detect drift.
    pub manifest_hash: String,
    /// Number of prompts in the source.
    pub count: usize,
}

/// Coarse run-time diagnostic summary embedded in the receipt.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiagnosticSummary {
    /// Overlap ratio at the end of training (if measured). Higher
    /// is healthier; Li et al. report 0.91 for successful runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlap_ratio_final: Option<f64>,
    /// Maximum RepRate observed across the run (the Luo et al. §3.2
    /// detector). Healthy = 0.0; 0.05+ indicates Stable-OPD engaged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rep_rate_max: Option<f64>,
    /// Names of guardrails that fired during the run (e.g.
    /// `["LengthInflation@step120"]`). Empty in successful runs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guardrail_triggers: Vec<String>,
    /// Final loss (mean KL for OPD, mean CE for SFT, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_loss: Option<f64>,
    /// ECHO summary: present only when the run trained with
    /// `LossConfig.echo = Some(...)` and at least one rollout
    /// carried env-observation tokens. Paper §3.1 / §5.2 reporting
    /// conventions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub echo: Option<EchoDiagnosticSummary>,
}

/// Receipt-grade ECHO summary. Captures the headline diagnostics from
/// docs/papers/echo/echo_paper.md §5.2 — env-token CE at the start and
/// end of training. A large drop (e.g. 30%+) is the paper's evidence
/// that the model "learned terminal dynamics."
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EchoDiagnosticSummary {
    /// Lambda used for the env-CE term (paper §3.3 default 0.05).
    pub lambda: f64,
    /// Env-token cross-entropy at the start of training, averaged
    /// across the env-positions of the first GRPO group. Lower = the
    /// model already predicts terminal output well; useful as a
    /// baseline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_ce_initial: Option<f64>,
    /// Env-token cross-entropy at the end of training. Paper §5.2:
    /// ECHO sharply lowers this relative to GRPO-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_ce_final: Option<f64>,
    /// Fraction of env-CE drop: `(env_ce_initial - env_ce_final) /
    /// env_ce_initial`. Direct comparison to paper Figure 3.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_ce_drop_pct: Option<f64>,
    /// Running λ·L_env / L_grpo ratio at the end of training. Useful
    /// for the "auto-anneal is working" check — paper §3.3 says ECHO
    /// self-anneals as the model learns terminal structure, so this
    /// ratio should shrink over time (or at least stay bounded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lambda_effective_final: Option<f64>,
    /// Total number of env-token gradients that contributed during the
    /// run (Σ |O'| across all GRPO groups). Useful for the "ECHO
    /// actually fired" smoke check.
    #[serde(default)]
    pub env_tokens_supervised: u64,
}

impl AdapterReceipt {
    /// Build a fresh receipt with the current UTC timestamp + the
    /// `CARGO_PKG_VERSION` for `kiln-train`. Most fields take
    /// reasonable defaults that callers should override.
    pub fn new(adapter: impl Into<String>, source_kind: impl Into<String>, seed: u64) -> Self {
        let mut kernel_versions = BTreeMap::new();
        kernel_versions.insert(
            "kiln-opd-loss-kernel".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );
        Self {
            schema_version: RECEIPT_SCHEMA_VERSION,
            adapter: adapter.into(),
            produced_at: chrono::Utc::now().to_rfc3339(),
            kiln_version: env!("CARGO_PKG_VERSION").to_string(),
            kernel_versions,
            seed,
            source_kind: source_kind.into(),
            teacher: None,
            prompts: None,
            hyperparameters: serde_json::Value::Object(Default::default()),
            diagnostic_summary: DiagnosticSummary::default(),
            post_eval: BTreeMap::new(),
        }
    }

    /// Builder helper.
    pub fn with_teacher(mut self, t: TeacherDescriptor) -> Self {
        self.teacher = Some(t);
        self
    }

    /// Builder helper.
    pub fn with_prompts(mut self, p: PromptSourceDescriptor) -> Self {
        self.prompts = Some(p);
        self
    }

    /// Builder helper for the free-form hyperparameters object.
    pub fn with_hyperparameters(mut self, value: serde_json::Value) -> Self {
        self.hyperparameters = value;
        self
    }

    /// Builder helper for diagnostic summary.
    pub fn with_diagnostic_summary(mut self, s: DiagnosticSummary) -> Self {
        self.diagnostic_summary = s;
        self
    }

    /// Builder helper for a post-eval score.
    pub fn with_eval_score(mut self, name: impl Into<String>, score: f64) -> Self {
        self.post_eval.insert(name.into(), score);
        self
    }

    /// Builder helper for a kernel-version entry. Caller can stamp
    /// commit-hash or git rev when those are tracked.
    pub fn with_kernel_version(
        mut self,
        engine: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        self.kernel_versions.insert(engine.into(), version.into());
        self
    }

    /// Serialize to pretty JSON and write to `<adapter_dir>/receipt.json`.
    /// The directory is created if it does not exist. Replaces any
    /// existing file at that path.
    pub fn write_to_adapter_dir(&self, adapter_dir: &Path) -> Result<std::path::PathBuf> {
        std::fs::create_dir_all(adapter_dir).with_context(|| {
            format!(
                "create adapter dir {} for receipt",
                adapter_dir.display()
            )
        })?;
        let path = adapter_dir.join("receipt.json");
        let json = serde_json::to_string_pretty(self).context("serialize receipt")?;
        std::fs::write(&path, json).with_context(|| {
            format!("write receipt {}", path.display())
        })?;
        Ok(path)
    }

    /// Read a receipt from `<adapter_dir>/receipt.json`. Returns
    /// `None` when no receipt is present (older adapters).
    pub fn read_from_adapter_dir(adapter_dir: &Path) -> Result<Option<Self>> {
        let path = adapter_dir.join("receipt.json");
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).with_context(|| {
            format!("read receipt {}", path.display())
        })?;
        let receipt: AdapterReceipt = serde_json::from_slice(&bytes).with_context(|| {
            format!("deserialize receipt {}", path.display())
        })?;
        Ok(Some(receipt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn receipt_round_trips_through_json() {
        let r = AdapterReceipt::new("math-frontier", "opd", 4218)
            .with_teacher(TeacherDescriptor {
                alias: "qwen3.6-27b@openrouter".into(),
                model_id: "qwen/qwen-3.6-27b".into(),
                model_version_hash: Some("sha256:dead".into()),
                snapshot_url: None,
            })
            .with_prompts(PromptSourceDescriptor {
                source: "kiln-canonical:math_reasoning:v3".into(),
                manifest_hash: "sha256:beef".into(),
                count: 100_000,
            })
            .with_hyperparameters(serde_json::json!({
                "loss": "teacher_top_k",
                "top_k": 32,
                "temperature": 1.0,
            }))
            .with_eval_score("math-frontier-eval", 0.71);

        let s = serde_json::to_string_pretty(&r).unwrap();
        let parsed: AdapterReceipt = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.schema_version, RECEIPT_SCHEMA_VERSION);
        assert_eq!(parsed.adapter, r.adapter);
        assert_eq!(parsed.seed, r.seed);
        assert_eq!(
            parsed.teacher.unwrap().alias,
            "qwen3.6-27b@openrouter"
        );
        let p = parsed.prompts.unwrap();
        assert_eq!(p.count, 100_000);
        assert_eq!(parsed.post_eval.get("math-frontier-eval"), Some(&0.71));
    }

    #[test]
    fn receipt_write_and_read_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let r = AdapterReceipt::new("test-adapter", "sft", 17);
        let path = r.write_to_adapter_dir(dir.path())?;
        assert!(path.exists());
        let loaded = AdapterReceipt::read_from_adapter_dir(dir.path())?
            .expect("receipt exists after write");
        assert_eq!(loaded.adapter, r.adapter);
        assert_eq!(loaded.source_kind, "sft");
        Ok(())
    }

    #[test]
    fn read_returns_none_when_missing() -> Result<()> {
        let dir = tempdir()?;
        // No receipt written.
        assert!(AdapterReceipt::read_from_adapter_dir(dir.path())?.is_none());
        Ok(())
    }

    #[test]
    fn omits_optional_fields_when_unset() {
        let r = AdapterReceipt::new("minimal", "sft", 0);
        let s = serde_json::to_string(&r).unwrap();
        // teacher/prompts/snapshot_url should not appear.
        assert!(!s.contains("\"teacher\""));
        assert!(!s.contains("\"prompts\""));
        assert!(!s.contains("\"snapshot_url\""));
    }

    #[test]
    fn default_diagnostic_summary_serializes_clean() {
        let r = AdapterReceipt::new("clean", "opd", 0);
        let s = serde_json::to_string(&r).unwrap();
        // Defaults: no overlap_ratio_final, no rep_rate_max, no
        // guardrail_triggers, no final_loss. The summary object is
        // present but empty.
        assert!(s.contains("\"diagnostic_summary\":{}"));
    }

    #[test]
    fn echo_diagnostic_summary_round_trips_with_all_fields() {
        // Pin the wire format for the ECHO diagnostics that
        // capability authors will read from receipt.json. All fields
        // set away from default; round-trip must preserve them.
        let summary = EchoDiagnosticSummary {
            lambda: 0.05,
            env_ce_initial: Some(4.21),
            env_ce_final: Some(0.83),
            env_ce_drop_pct: Some(80.3),
            lambda_effective_final: Some(0.07),
            env_tokens_supervised: 24576,
        };
        let json = serde_json::to_string(&summary).unwrap();
        // Every non-Option field should be present; every populated
        // Option should be present (skipped only when None).
        assert!(json.contains("\"lambda\":0.05"), "lambda missing: {json}");
        assert!(json.contains("\"env_ce_initial\":4.21"), "env_ce_initial missing: {json}");
        assert!(json.contains("\"env_ce_final\":0.83"), "env_ce_final missing: {json}");
        assert!(json.contains("\"env_ce_drop_pct\":80.3"), "drop_pct missing: {json}");
        assert!(json.contains("\"lambda_effective_final\":0.07"), "lambda_effective missing: {json}");
        assert!(json.contains("\"env_tokens_supervised\":24576"), "env_tokens_supervised missing: {json}");

        let parsed: EchoDiagnosticSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, summary);
    }

    #[test]
    fn echo_diagnostic_summary_skips_none_fields() {
        // A run with only λ tracked (e.g. before env_ce sampling lands)
        // should produce a JSON without the unset Option fields.
        let summary = EchoDiagnosticSummary {
            lambda: 0.05,
            env_ce_initial: None,
            env_ce_final: None,
            env_ce_drop_pct: None,
            lambda_effective_final: None,
            env_tokens_supervised: 0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"lambda\":0.05"));
        assert!(!json.contains("env_ce_initial"), "should skip None: {json}");
        assert!(!json.contains("env_ce_final"));
        assert!(!json.contains("env_ce_drop_pct"));
        assert!(!json.contains("lambda_effective_final"));
        // env_tokens_supervised is u64, not Option — always present.
        assert!(json.contains("\"env_tokens_supervised\":0"));
    }

    #[test]
    fn echo_section_embeds_in_diagnostic_summary() {
        // The ECHO summary sits under diagnostic_summary.echo. Verify
        // a populated receipt round-trips with the ECHO field intact.
        let echo = EchoDiagnosticSummary {
            lambda: 0.05,
            env_ce_initial: Some(4.21),
            env_ce_final: Some(0.83),
            env_ce_drop_pct: Some(80.3),
            lambda_effective_final: Some(0.07),
            env_tokens_supervised: 24576,
        };
        let mut summary = DiagnosticSummary::default();
        summary.echo = Some(echo.clone());
        let receipt = AdapterReceipt::new("echo-adapter", "grpo", 4218)
            .with_diagnostic_summary(summary);
        let s = serde_json::to_string(&receipt).unwrap();
        assert!(s.contains("\"echo\""), "echo summary missing: {s}");
        let parsed: AdapterReceipt = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.diagnostic_summary.echo, Some(echo));
    }
}
