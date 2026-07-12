//! Versioned, self-verifying handoff between Kiln and HF/TRL.
//!
//! An export manifest binds the selected corpus, model/tokenizer/template
//! identity, optional input adapter, and the exact reference script. External
//! training produces a separate result manifest that binds its package
//! versions, effective configuration, and PEFT output back to that export.
//! Importers validate both manifests and every referenced byte before
//! publishing an adapter.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned};

use crate::{ROLLOUT_PROVENANCE_SCHEMA_V1, SftInvalidRowPolicy};

pub const HF_TRL_EXPORT_SCHEMA_VERSION: u32 = 1;
pub const HF_TRL_EXPORT_TYPE: &str = "kiln.hf-trl-export.v1";
pub const HF_TRL_RESULT_SCHEMA_VERSION: u32 = 1;
pub const HF_TRL_RESULT_TYPE: &str = "kiln.hf-trl-result.v1";

pub const HF_TRL_EXPORT_MANIFEST_FILENAME: &str = "kiln_hf_export.json";
pub const HF_TRL_RESULT_MANIFEST_FILENAME: &str = "kiln_hf_result.json";
pub const HF_TRL_DATASET_FILENAME: &str = "train.jsonl";
pub const HF_TRL_MODEL_CONFIG_FILENAME: &str = "kiln_model_config.json";
pub const HF_TRL_TOKENIZER_FILENAME: &str = "tokenizer.json";
pub const HF_TRL_CHAT_TEMPLATE_FILENAME: &str = "chat_template.jinja";
pub const HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME: &str = "kiln_training_chat_template.jinja";
pub const HF_TRL_TRAINING_TEMPLATE_FILENAME: &str = "training_chat_template.jinja";
pub const HF_TRL_REFERENCE_SCRIPT_FILENAME: &str = "train.py";
pub const HF_TRL_EXECUTED_SCRIPT_FILENAME: &str = "executed_train.py";
pub const HF_TRL_ENVIRONMENT_LOCK_FILENAME: &str = "requirements.lock";
pub const HF_TRL_SPLIT_MANIFEST_FILENAME: &str = "split_manifest.json";
pub const HF_TRL_SFT_INGESTION_FILENAME: &str = "sft_ingestion.json";
pub const HF_TRL_ADAPTER_CONFIG_FILENAME: &str = "adapter_config.json";
pub const HF_TRL_ADAPTER_MODEL_FILENAME: &str = "adapter_model.safetensors";

const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_RELATIVE_PATH_BYTES: usize = 512;
const MAX_TEXT_BYTES: usize = 512;
const MAX_EFFECTIVE_CONFIG_BYTES: usize = 64 * 1024;
const MAX_EFFECTIVE_CONFIG_ENTRIES: usize = 256;
const MAX_CONFIG_LIST_ITEMS: usize = 256;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HfTrlTask {
    Sft,
    Grpo,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HfTrlDatasetFormat {
    SftMessagesJsonl,
    GrpoGroupsJsonl,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HfTrlTrainerKind {
    TrlSftTrainer,
    TrlGrpoTrainer,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HfTrlSftLabelPolicy {
    /// TRL derives labels from `{% generation %}` template spans.
    AssistantOnlyGenerationSpans,
}

/// Cross-language-stable value used in the effective trainer configuration.
///
/// JSON floating-point formatting is not byte-identical between Rust and
/// Python for every exponent. Decimal values therefore travel as exact text;
/// the remaining variants have unambiguous JSON encodings in both runtimes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum HfTrlConfigValue {
    Boolean(bool),
    Integer(i64),
    Unsigned(u64),
    Decimal(String),
    Text(String),
    TextList(Vec<String>),
}

impl HfTrlConfigValue {
    fn validate(&self, field: &str) -> Result<()> {
        match self {
            Self::Boolean(_) | Self::Integer(_) | Self::Unsigned(_) => Ok(()),
            Self::Decimal(value) => {
                validate_text(field, value)?;
                let parsed = value
                    .parse::<f64>()
                    .with_context(|| format!("HF/TRL {field} is not a decimal number"))?;
                ensure!(parsed.is_finite(), "HF/TRL {field} decimal must be finite");
                Ok(())
            }
            Self::Text(value) => validate_text(field, value),
            Self::TextList(values) => {
                ensure!(
                    values.len() <= MAX_CONFIG_LIST_ITEMS,
                    "HF/TRL {field} contains more than {MAX_CONFIG_LIST_ITEMS} entries"
                );
                for (index, value) in values.iter().enumerate() {
                    validate_text(&format!("{field}[{index}]"), value)?;
                }
                Ok(())
            }
        }
    }
}

impl HfTrlTrainerKind {
    fn task(self) -> HfTrlTask {
        match self {
            Self::TrlSftTrainer => HfTrlTask::Sft,
            Self::TrlGrpoTrainer => HfTrlTask::Grpo,
        }
    }
}

/// Exact identity of one regular file inside an interoperability bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlFileIdentity {
    pub relative_path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

impl HfTrlFileIdentity {
    pub fn from_bytes(relative_path: impl Into<String>, bytes: &[u8]) -> Result<Self> {
        let identity = Self {
            relative_path: relative_path.into(),
            size_bytes: u64::try_from(bytes.len())
                .context("interop artifact length exceeds u64")?,
            sha256: crate::train_receipt::sha256_bytes(bytes),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn from_file(root: &Path, relative_path: impl Into<String>) -> Result<Self> {
        let relative_path = relative_path.into();
        validate_relative_path(&relative_path)?;
        let path = resolve_regular_bundle_file(root, &relative_path)?;
        let size_bytes = path
            .metadata()
            .with_context(|| format!("stat interoperability artifact {}", path.display()))?
            .len();
        let sha256 = crate::train_receipt::sha256_file(&path)?;
        let identity = Self {
            relative_path,
            size_bytes,
            sha256,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<()> {
        validate_relative_path(&self.relative_path)?;
        ensure!(
            self.size_bytes > 0,
            "interoperability artifact {:?} must not be empty",
            self.relative_path
        );
        validate_sha256("interoperability artifact sha256", &self.sha256)
    }

    pub fn verify(&self, root: &Path) -> Result<()> {
        self.validate()?;
        let path = resolve_regular_bundle_file(root, &self.relative_path)?;
        let actual_size = path
            .metadata()
            .with_context(|| format!("stat interoperability artifact {}", path.display()))?
            .len();
        ensure!(
            actual_size == self.size_bytes,
            "interoperability artifact {:?} size differs: manifest={}, actual={actual_size}",
            self.relative_path,
            self.size_bytes
        );
        let actual_sha256 = crate::train_receipt::sha256_file(&path)?;
        ensure!(
            actual_sha256 == self.sha256,
            "interoperability artifact {:?} SHA-256 differs: manifest={}, actual={actual_sha256}",
            self.relative_path,
            self.sha256
        );
        Ok(())
    }
}

/// Model bytes and template configuration an external trainer must use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlModelIdentity {
    pub served_model_id: String,
    pub base_weight_shard_manifest: kiln_core::model_provenance::BaseWeightShardManifest,
    pub tokenizer_vocab_sha256: String,
    pub model_config: HfTrlFileIdentity,
    pub tokenizer: HfTrlFileIdentity,
    pub chat_template: HfTrlFileIdentity,
    /// Exact effective template identity recorded by native Kiln artifacts.
    pub native_training_chat_template: HfTrlFileIdentity,
    /// Exact generation-marked template passed to HF/TRL.
    pub trl_training_chat_template: HfTrlFileIdentity,
}

impl HfTrlModelIdentity {
    fn validate(&self) -> Result<()> {
        validate_text("served_model_id", &self.served_model_id)?;
        self.base_weight_shard_manifest
            .validate()
            .context("validate HF/TRL base-weight identity")?;
        validate_sha256("tokenizer_vocab_sha256", &self.tokenizer_vocab_sha256)?;
        self.model_config.validate()?;
        self.tokenizer.validate()?;
        self.chat_template.validate()?;
        self.native_training_chat_template.validate()?;
        self.trl_training_chat_template.validate()?;
        ensure_exact_path(&self.model_config, HF_TRL_MODEL_CONFIG_FILENAME)?;
        ensure_exact_path(&self.tokenizer, HF_TRL_TOKENIZER_FILENAME)?;
        ensure_exact_path(&self.chat_template, HF_TRL_CHAT_TEMPLATE_FILENAME)?;
        ensure_exact_path(
            &self.native_training_chat_template,
            HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME,
        )?;
        ensure_exact_path(
            &self.trl_training_chat_template,
            HF_TRL_TRAINING_TEMPLATE_FILENAME,
        )?;
        Ok(())
    }

    fn verify_files(&self, root: &Path) -> Result<()> {
        self.model_config.verify(root)?;
        self.tokenizer.verify(root)?;
        self.chat_template.verify(root)?;
        self.native_training_chat_template.verify(root)?;
        self.trl_training_chat_template.verify(root)
    }
}

/// Compact SFT row-selection evidence. Per-row hashes remain in Kiln's train
/// receipt; the handoff binds their ordered aggregate and the exact exported
/// JSONL bytes without making large datasets produce large manifests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlSftSelection {
    pub invalid_row_policy: SftInvalidRowPolicy,
    pub label_policy: HfTrlSftLabelPolicy,
    pub rows_read: u64,
    pub rows_kept: u64,
    pub rows_rejected: u64,
    pub kept_corpus_sha256: String,
    pub ingestion_receipt: HfTrlFileIdentity,
}

impl HfTrlSftSelection {
    pub fn from_ingestion(
        ingestion: &crate::SftIngestionReceipt,
        ingestion_receipt: HfTrlFileIdentity,
    ) -> Result<Self> {
        ingestion.validate()?;
        let selection = Self {
            invalid_row_policy: ingestion.invalid_row_policy,
            label_policy: HfTrlSftLabelPolicy::AssistantOnlyGenerationSpans,
            rows_read: u64::try_from(ingestion.rows_read).context("SFT rows_read exceeds u64")?,
            rows_kept: u64::try_from(ingestion.rows_kept).context("SFT rows_kept exceeds u64")?,
            rows_rejected: u64::try_from(ingestion.rows_rejected)
                .context("SFT rows_rejected exceeds u64")?,
            kept_corpus_sha256: ingestion.kept_corpus_sha256.clone(),
            ingestion_receipt,
        };
        selection.validate()?;
        Ok(selection)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.rows_kept > 0,
            "HF/TRL SFT export contains no kept rows"
        );
        let expected_rows_read = self
            .rows_kept
            .checked_add(self.rows_rejected)
            .context("HF/TRL SFT row counts overflow u64")?;
        ensure!(
            self.rows_read == expected_rows_read,
            "HF/TRL SFT row counts are inconsistent"
        );
        validate_sha256("kept_corpus_sha256", &self.kept_corpus_sha256)?;
        self.ingestion_receipt.validate()?;
        ensure_exact_path(&self.ingestion_receipt, HF_TRL_SFT_INGESTION_FILENAME)
    }

    fn verify_files(&self, root: &Path) -> Result<crate::SftIngestionReceipt> {
        self.ingestion_receipt.verify(root)?;
        let path = resolve_regular_bundle_file(root, &self.ingestion_receipt.relative_path)?;
        let file = std::fs::File::open(&path)
            .with_context(|| format!("open HF/TRL SFT ingestion receipt {}", path.display()))?;
        let ingestion: crate::SftIngestionReceipt = serde_json::from_reader(file)
            .with_context(|| format!("parse HF/TRL SFT ingestion receipt {}", path.display()))?;
        ingestion
            .validate()
            .context("validate HF/TRL SFT ingestion receipt")?;
        ensure!(
            ingestion.invalid_row_policy == self.invalid_row_policy
                && u64::try_from(ingestion.rows_read).ok() == Some(self.rows_read)
                && u64::try_from(ingestion.rows_kept).ok() == Some(self.rows_kept)
                && u64::try_from(ingestion.rows_rejected).ok() == Some(self.rows_rejected)
                && ingestion.kept_corpus_sha256 == self.kept_corpus_sha256,
            "HF/TRL SFT selection differs from its full ingestion receipt"
        );
        Ok(ingestion)
    }
}

/// Dataset bytes and selection/provenance policy exported to HF/TRL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlDataExport {
    pub source_name: String,
    pub format: HfTrlDatasetFormat,
    pub row_count: u64,
    pub ordered_corpus_sha256: String,
    pub dataset: HfTrlFileIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sft_selection: Option<HfTrlSftSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_provenance_schema: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_manifest: Option<HfTrlFileIdentity>,
}

impl HfTrlDataExport {
    fn validate(&self, task: HfTrlTask) -> Result<()> {
        validate_text("source_name", &self.source_name)?;
        ensure!(self.row_count > 0, "HF/TRL export contains no rows");
        validate_sha256("ordered_corpus_sha256", &self.ordered_corpus_sha256)?;
        self.dataset.validate()?;
        ensure_exact_path(&self.dataset, HF_TRL_DATASET_FILENAME)?;
        if let Some(split) = self.split_manifest.as_ref() {
            split.validate()?;
            ensure_exact_path(split, HF_TRL_SPLIT_MANIFEST_FILENAME)?;
        }
        match task {
            HfTrlTask::Sft => {
                ensure!(
                    self.format == HfTrlDatasetFormat::SftMessagesJsonl,
                    "SFT interoperability export must use sft_messages_jsonl"
                );
                let selection = self
                    .sft_selection
                    .as_ref()
                    .context("SFT interoperability export is missing row-selection evidence")?;
                selection.validate()?;
                ensure!(
                    self.row_count == selection.rows_kept,
                    "SFT exported row_count differs from rows_kept"
                );
                ensure!(
                    self.ordered_corpus_sha256 == selection.kept_corpus_sha256,
                    "SFT ordered corpus identity differs from row-selection evidence"
                );
                ensure!(
                    self.rollout_provenance_schema.is_none(),
                    "SFT interoperability export must not declare rollout provenance"
                );
            }
            HfTrlTask::Grpo => {
                ensure!(
                    self.format == HfTrlDatasetFormat::GrpoGroupsJsonl,
                    "GRPO interoperability export must use grpo_groups_jsonl"
                );
                ensure!(
                    self.sft_selection.is_none(),
                    "GRPO interoperability export must not contain SFT selection evidence"
                );
                ensure!(
                    self.rollout_provenance_schema.as_deref() == Some(ROLLOUT_PROVENANCE_SCHEMA_V1),
                    "GRPO interoperability export requires exact {ROLLOUT_PROVENANCE_SCHEMA_V1} records"
                );
            }
        }
        Ok(())
    }

    fn verify_files(&self, root: &Path, task: HfTrlTask) -> Result<()> {
        self.dataset.verify(root)?;
        if let Some(split) = self.split_manifest.as_ref() {
            split.verify(root)?;
        }
        if task == HfTrlTask::Sft {
            let selection = self
                .sft_selection
                .as_ref()
                .context("SFT interoperability export is missing row-selection evidence")?;
            let ingestion = selection.verify_files(root)?;
            ensure!(
                ingestion.source == self.source_name,
                "HF/TRL SFT source_name differs from its ingestion receipt"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlInputAdapter {
    pub name: String,
    pub config: HfTrlFileIdentity,
    pub model: HfTrlFileIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kiln_manifest: Option<HfTrlFileIdentity>,
}

impl HfTrlInputAdapter {
    fn validate(&self) -> Result<()> {
        validate_text("input adapter name", &self.name)?;
        self.config.validate()?;
        self.model.validate()?;
        ensure_exact_path(&self.config, "input_adapter/adapter_config.json")?;
        ensure_exact_path(&self.model, "input_adapter/adapter_model.safetensors")?;
        if let Some(manifest) = self.kiln_manifest.as_ref() {
            manifest.validate()?;
            ensure_exact_path(manifest, "input_adapter/adapter_manifest.json")?;
        }
        Ok(())
    }

    fn verify_files(&self, root: &Path) -> Result<()> {
        self.config.verify(root)?;
        self.model.verify(root)?;
        if let Some(manifest) = self.kiln_manifest.as_ref() {
            manifest.verify(root)?;
        }
        Ok(())
    }
}

/// Immutable identity of one Kiln-to-HF/TRL export directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlExportManifestV1 {
    pub schema_version: u32,
    pub manifest_type: String,
    pub task: HfTrlTask,
    pub source_execution_provenance: kiln_core::execution_provenance::ExecutionProvenanceV1,
    pub model: HfTrlModelIdentity,
    pub data: HfTrlDataExport,
    pub reference_script: HfTrlFileIdentity,
    pub environment_lock: HfTrlFileIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_adapter: Option<HfTrlInputAdapter>,
    pub export_sha256: String,
}

impl HfTrlExportManifestV1 {
    pub fn new(
        task: HfTrlTask,
        source_execution_provenance: kiln_core::execution_provenance::ExecutionProvenanceV1,
        model: HfTrlModelIdentity,
        data: HfTrlDataExport,
        reference_script: HfTrlFileIdentity,
        environment_lock: HfTrlFileIdentity,
        input_adapter: Option<HfTrlInputAdapter>,
    ) -> Result<Self> {
        let mut manifest = Self {
            schema_version: HF_TRL_EXPORT_SCHEMA_VERSION,
            manifest_type: HF_TRL_EXPORT_TYPE.to_string(),
            task,
            source_execution_provenance,
            model,
            data,
            reference_script,
            environment_lock,
            input_adapter,
            export_sha256: empty_sha256(),
        };
        manifest.validate_fields()?;
        manifest.export_sha256 = manifest.compute_sha256()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_fields()?;
        validate_sha256("export_sha256", &self.export_sha256)?;
        let expected = self.compute_sha256()?;
        ensure!(
            self.export_sha256 == expected,
            "HF/TRL export manifest digest differs: manifest={}, expected={expected}",
            self.export_sha256
        );
        Ok(())
    }

    pub fn verify_files(&self, root: &Path) -> Result<()> {
        self.validate()?;
        self.model.verify_files(root)?;
        self.data.verify_files(root, self.task)?;
        self.reference_script.verify(root)?;
        self.environment_lock.verify(root)?;
        if let Some(adapter) = self.input_adapter.as_ref() {
            adapter.verify_files(root)?;
        }
        Ok(())
    }

    fn validate_fields(&self) -> Result<()> {
        ensure!(
            self.schema_version == HF_TRL_EXPORT_SCHEMA_VERSION,
            "unsupported HF/TRL export schema_version {}",
            self.schema_version
        );
        ensure!(
            self.manifest_type == HF_TRL_EXPORT_TYPE,
            "invalid HF/TRL export manifest_type {:?}",
            self.manifest_type
        );
        self.source_execution_provenance
            .validate()
            .context("validate source execution provenance in HF/TRL export")?;
        self.model.validate()?;
        let source_model = &self.source_execution_provenance.model;
        ensure!(
            source_model.model_config_sha256 == self.model.model_config.sha256
                && source_model.tokenizer_config_sha256 == self.model.tokenizer.sha256
                && source_model.tokenizer_vocab_sha256 == self.model.tokenizer_vocab_sha256
                && source_model.chat_template_sha256.as_deref()
                    == Some(self.model.chat_template.sha256.as_str())
                && source_model.training_chat_template_sha256.as_deref()
                    == Some(self.model.native_training_chat_template.sha256.as_str()),
            "HF/TRL model artifacts differ from source execution provenance"
        );
        self.data.validate(self.task)?;
        self.reference_script.validate()?;
        ensure_exact_path(&self.reference_script, HF_TRL_REFERENCE_SCRIPT_FILENAME)?;
        self.environment_lock.validate()?;
        ensure_exact_path(&self.environment_lock, HF_TRL_ENVIRONMENT_LOCK_FILENAME)?;
        if let Some(adapter) = self.input_adapter.as_ref() {
            adapter.validate()?;
        }
        Ok(())
    }

    fn compute_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        struct Identity<'a> {
            schema_version: u32,
            manifest_type: &'a str,
            task: HfTrlTask,
            source_execution_provenance: &'a kiln_core::execution_provenance::ExecutionProvenanceV1,
            model: &'a HfTrlModelIdentity,
            data: &'a HfTrlDataExport,
            reference_script: &'a HfTrlFileIdentity,
            environment_lock: &'a HfTrlFileIdentity,
            #[serde(skip_serializing_if = "Option::is_none")]
            input_adapter: &'a Option<HfTrlInputAdapter>,
        }
        canonical_json_sha256(&Identity {
            schema_version: self.schema_version,
            manifest_type: &self.manifest_type,
            task: self.task,
            source_execution_provenance: &self.source_execution_provenance,
            model: &self.model,
            data: &self.data,
            reference_script: &self.reference_script,
            environment_lock: &self.environment_lock,
            input_adapter: &self.input_adapter,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlTrainerIdentity {
    pub kind: HfTrlTrainerKind,
    pub python_version: String,
    pub torch_version: String,
    pub transformers_version: String,
    pub trl_version: String,
    pub peft_version: String,
    /// Preserved bytes of the script that actually executed.
    pub script: HfTrlFileIdentity,
}

impl HfTrlTrainerIdentity {
    fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("python_version", self.python_version.as_str()),
            ("torch_version", self.torch_version.as_str()),
            ("transformers_version", self.transformers_version.as_str()),
            ("trl_version", self.trl_version.as_str()),
            ("peft_version", self.peft_version.as_str()),
        ] {
            validate_text(field, value)?;
        }
        self.script.validate()?;
        ensure_exact_path(&self.script, HF_TRL_EXECUTED_SCRIPT_FILENAME)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlOutputAdapter {
    pub config: HfTrlFileIdentity,
    pub model: HfTrlFileIdentity,
}

impl HfTrlOutputAdapter {
    fn validate(&self) -> Result<()> {
        self.config.validate()?;
        self.model.validate()?;
        ensure_exact_path(&self.config, HF_TRL_ADAPTER_CONFIG_FILENAME)?;
        ensure_exact_path(&self.model, HF_TRL_ADAPTER_MODEL_FILENAME)
    }

    fn verify_files(&self, root: &Path) -> Result<()> {
        self.config.verify(root)?;
        self.model.verify(root)
    }
}

/// External trainer output required by validated PEFT import.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HfTrlTrainingResultV1 {
    pub schema_version: u32,
    pub result_type: String,
    pub export_sha256: String,
    pub task: HfTrlTask,
    pub trainer: HfTrlTrainerIdentity,
    #[serde(deserialize_with = "deserialize_unique_effective_config")]
    pub effective_config: BTreeMap<String, HfTrlConfigValue>,
    pub output_adapter: HfTrlOutputAdapter,
    pub result_sha256: String,
}

impl HfTrlTrainingResultV1 {
    pub fn new(
        export_sha256: String,
        task: HfTrlTask,
        trainer: HfTrlTrainerIdentity,
        effective_config: BTreeMap<String, HfTrlConfigValue>,
        output_adapter: HfTrlOutputAdapter,
    ) -> Result<Self> {
        let mut result = Self {
            schema_version: HF_TRL_RESULT_SCHEMA_VERSION,
            result_type: HF_TRL_RESULT_TYPE.to_string(),
            export_sha256,
            task,
            trainer,
            effective_config,
            output_adapter,
            result_sha256: empty_sha256(),
        };
        result.validate_fields()?;
        result.result_sha256 = result.compute_sha256()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_fields()?;
        validate_sha256("result_sha256", &self.result_sha256)?;
        let expected = self.compute_sha256()?;
        ensure!(
            self.result_sha256 == expected,
            "HF/TRL result manifest digest differs: manifest={}, expected={expected}",
            self.result_sha256
        );
        Ok(())
    }

    pub fn validate_against_export(&self, export: &HfTrlExportManifestV1) -> Result<()> {
        self.validate()?;
        export.validate()?;
        ensure!(
            self.export_sha256 == export.export_sha256,
            "HF/TRL result references export {}, but bundle contains {}",
            self.export_sha256,
            export.export_sha256
        );
        ensure!(
            self.task == export.task,
            "HF/TRL result task differs from its export"
        );
        ensure!(
            self.trainer.kind.task() == self.task,
            "HF/TRL trainer kind does not match result task"
        );
        Ok(())
    }

    /// Whether the preserved executed script exactly matches Kiln's exported
    /// reference. Custom scripts remain importable, but their distinct bytes
    /// stay visible in the result identity.
    pub fn uses_exported_reference_script(&self, export: &HfTrlExportManifestV1) -> Result<bool> {
        self.validate_against_export(export)?;
        Ok(self.trainer.script.sha256 == export.reference_script.sha256)
    }

    pub fn verify_files(&self, root: &Path) -> Result<()> {
        self.validate()?;
        self.trainer.script.verify(root)?;
        self.output_adapter.verify_files(root)
    }

    fn validate_fields(&self) -> Result<()> {
        ensure!(
            self.schema_version == HF_TRL_RESULT_SCHEMA_VERSION,
            "unsupported HF/TRL result schema_version {}",
            self.schema_version
        );
        ensure!(
            self.result_type == HF_TRL_RESULT_TYPE,
            "invalid HF/TRL result_type {:?}",
            self.result_type
        );
        validate_sha256("result export_sha256", &self.export_sha256)?;
        self.trainer.validate()?;
        ensure!(
            self.trainer.kind.task() == self.task,
            "HF/TRL trainer kind does not match result task"
        );
        ensure!(
            !self.effective_config.is_empty(),
            "HF/TRL result effective_config must not be empty"
        );
        ensure!(
            self.effective_config.len() <= MAX_EFFECTIVE_CONFIG_ENTRIES,
            "HF/TRL result effective_config contains more than {MAX_EFFECTIVE_CONFIG_ENTRIES} entries"
        );
        for (key, value) in &self.effective_config {
            validate_text("effective_config key", key)?;
            value.validate(&format!("effective_config.{key}"))?;
        }
        let config_bytes = canonical_json_bytes(&self.effective_config)?;
        ensure!(
            config_bytes.len() <= MAX_EFFECTIVE_CONFIG_BYTES,
            "HF/TRL result effective_config exceeds {MAX_EFFECTIVE_CONFIG_BYTES} bytes"
        );
        self.output_adapter.validate()
    }

    fn compute_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        struct Identity<'a> {
            schema_version: u32,
            result_type: &'a str,
            export_sha256: &'a str,
            task: HfTrlTask,
            trainer: &'a HfTrlTrainerIdentity,
            effective_config: &'a BTreeMap<String, HfTrlConfigValue>,
            output_adapter: &'a HfTrlOutputAdapter,
        }
        canonical_json_sha256(&Identity {
            schema_version: self.schema_version,
            result_type: &self.result_type,
            export_sha256: &self.export_sha256,
            task: self.task,
            trainer: &self.trainer,
            effective_config: &self.effective_config,
            output_adapter: &self.output_adapter,
        })
    }
}

pub fn read_hf_trl_export_manifest(root: &Path) -> Result<HfTrlExportManifestV1> {
    let manifest: HfTrlExportManifestV1 = read_manifest(root, HF_TRL_EXPORT_MANIFEST_FILENAME)?;
    manifest.validate()?;
    Ok(manifest)
}

pub fn read_hf_trl_training_result(root: &Path) -> Result<HfTrlTrainingResultV1> {
    let result: HfTrlTrainingResultV1 = read_manifest(root, HF_TRL_RESULT_MANIFEST_FILENAME)?;
    result.validate()?;
    Ok(result)
}

fn deserialize_unique_effective_config<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, HfTrlConfigValue>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueConfigVisitor;

    impl<'de> serde::de::Visitor<'de> for UniqueConfigVisitor {
        type Value = BTreeMap<String, HfTrlConfigValue>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("an effective configuration object with unique keys")
        }

        fn visit_map<A>(self, mut access: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut values = BTreeMap::new();
            while let Some((key, value)) = access.next_entry::<String, HfTrlConfigValue>()? {
                if values.insert(key.clone(), value).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate effective_config key {key:?}"
                    )));
                }
            }
            Ok(values)
        }
    }

    deserializer.deserialize_map(UniqueConfigVisitor)
}

fn read_manifest<T: DeserializeOwned>(root: &Path, relative_path: &str) -> Result<T> {
    let path = resolve_regular_bundle_file(root, relative_path)?;
    let size = path
        .metadata()
        .with_context(|| format!("stat interoperability manifest {}", path.display()))?
        .len();
    ensure!(
        size <= MAX_MANIFEST_BYTES,
        "interoperability manifest {} exceeds {MAX_MANIFEST_BYTES} bytes",
        path.display()
    );
    let bytes = std::fs::read(&path)
        .with_context(|| format!("read interoperability manifest {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse interoperability manifest {}", path.display()))
}

fn ensure_exact_path(identity: &HfTrlFileIdentity, expected: &str) -> Result<()> {
    ensure!(
        identity.relative_path == expected,
        "interoperability artifact path {:?} must be {expected:?}",
        identity.relative_path
    );
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<()> {
    ensure!(!value.is_empty(), "interoperability artifact path is empty");
    ensure!(
        value.len() <= MAX_RELATIVE_PATH_BYTES,
        "interoperability artifact path exceeds {MAX_RELATIVE_PATH_BYTES} bytes"
    );
    ensure!(
        !value.contains('\\'),
        "interoperability artifact path contains a backslash: {value:?}"
    );
    ensure!(
        !value.chars().any(char::is_control),
        "interoperability artifact path contains a control character: {value:?}"
    );
    ensure!(
        value
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != ".."),
        "interoperability artifact path is not normalized: {value:?}"
    );
    let path = Path::new(value);
    ensure!(
        !path.is_absolute()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "interoperability artifact path is not normalized: {value:?}"
    );
    Ok(())
}

fn resolve_regular_bundle_file(root: &Path, relative_path: &str) -> Result<std::path::PathBuf> {
    validate_relative_path(relative_path)?;
    let root_meta = std::fs::symlink_metadata(root)
        .with_context(|| format!("stat interoperability bundle root {}", root.display()))?;
    ensure!(
        root_meta.file_type().is_dir() && !root_meta.file_type().is_symlink(),
        "interoperability bundle root must be a real directory: {}",
        root.display()
    );
    let root = root
        .canonicalize()
        .with_context(|| format!("resolve interoperability bundle root {}", root.display()))?;
    let mut current = root;
    let components = Path::new(relative_path).components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(segment) = component else {
            unreachable!("relative path was validated")
        };
        current.push(segment);
        let metadata = std::fs::symlink_metadata(&current)
            .with_context(|| format!("stat interoperability path {}", current.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "interoperability path must not contain symlinks: {}",
            current.display()
        );
        if index + 1 == components.len() {
            ensure!(
                metadata.file_type().is_file(),
                "interoperability artifact is not a regular file: {}",
                current.display()
            );
        } else {
            ensure!(
                metadata.file_type().is_dir(),
                "interoperability artifact parent is not a directory: {}",
                current.display()
            );
        }
    }
    Ok(current)
}

fn validate_text(field: &str, value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.trim() == value
            && value.len() <= MAX_TEXT_BYTES
            && !value.chars().any(char::is_control),
        "HF/TRL {field} must be non-empty, trimmed, bounded text"
    );
    Ok(())
}

fn validate_sha256(field: &str, value: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        bail!("HF/TRL {field} must use sha256:<64 lowercase hex>")
    };
    ensure!(
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "HF/TRL {field} must use sha256:<64 lowercase hex>"
    );
    Ok(())
}

fn empty_sha256() -> String {
    format!("sha256:{}", "0".repeat(64))
}

fn canonical_json_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(crate::train_receipt::sha256_bytes(&canonical_json_bytes(
        value,
    )?))
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut value = serde_json::to_value(value).context("serialize HF/TRL identity")?;
    canonicalize_json(&mut value);
    serde_json::to_vec(&value).context("encode canonical HF/TRL identity")
}

fn canonicalize_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                canonicalize_json(value);
            }
        }
        serde_json::Value::Object(object) => {
            let mut entries = std::mem::take(object).into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (key, mut value) in entries {
                canonicalize_json(&mut value);
                object.insert(key, value);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_core::execution_provenance::{
        ExecutionBackendIdentity, ExecutionBuildIdentity, ExecutionConfigurationIdentity,
        ExecutionKernelIdentity, ExecutionModelIdentity, ExecutionPrecisionIdentity,
        ExecutionProvenanceV1,
    };
    use kiln_core::model_provenance::{BaseWeightShardIdentity, BaseWeightShardManifest};
    use serde_json::json;

    fn write(root: &Path, relative: &str, bytes: &[u8]) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn artifact(root: &Path, relative: &str) -> HfTrlFileIdentity {
        HfTrlFileIdentity::from_file(root, relative).unwrap()
    }

    fn source_execution(model: &HfTrlModelIdentity) -> ExecutionProvenanceV1 {
        let hash = |label: &[u8]| crate::train_receipt::sha256_bytes(label);
        ExecutionProvenanceV1::new(
            ExecutionBackendIdentity {
                name: "cpu".to_string(),
                device: "test-device".to_string(),
                numerical_runtime_sha256: hash(b"runtime"),
            },
            ExecutionBuildIdentity {
                package_version: "0.4.1".to_string(),
                target: "test-target".to_string(),
                executable_sha256: hash(b"executable"),
                git_commit: None,
                source_tree_sha256: None,
                source_dirty: None,
            },
            ExecutionModelIdentity {
                model_config_sha256: model.model_config.sha256.clone(),
                tokenizer_vocab_sha256: model.tokenizer_vocab_sha256.clone(),
                tokenizer_config_sha256: model.tokenizer.sha256.clone(),
                chat_template_sha256: Some(model.chat_template.sha256.clone()),
                training_chat_template_sha256: Some(
                    model.native_training_chat_template.sha256.clone(),
                ),
            },
            ExecutionPrecisionIdentity {
                inference_dtype: "f32".to_string(),
                training_policy: "f32".to_string(),
            },
            ExecutionKernelIdentity::new(
                BTreeMap::from([("test_kernel".to_string(), "v1".to_string())]),
                Vec::new(),
            )
            .unwrap(),
            ExecutionConfigurationIdentity {
                effective_server_config_sha256: hash(b"server-config"),
                effective_environment_sha256: hash(b"environment"),
            },
        )
        .unwrap()
    }

    fn export_bundle(root: &Path) -> HfTrlExportManifestV1 {
        for (path, bytes) in [
            (HF_TRL_MODEL_CONFIG_FILENAME, b"{}".as_slice()),
            (
                HF_TRL_TOKENIZER_FILENAME,
                b"{\"version\":\"1.0\"}".as_slice(),
            ),
            (HF_TRL_CHAT_TEMPLATE_FILENAME, b"{{ messages }}".as_slice()),
            (
                HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME,
                b"{{ messages }}".as_slice(),
            ),
            (
                HF_TRL_TRAINING_TEMPLATE_FILENAME,
                b"{% generation %}{{ messages }}{% endgeneration %}".as_slice(),
            ),
            (
                HF_TRL_REFERENCE_SCRIPT_FILENAME,
                b"print('train')\n".as_slice(),
            ),
            (
                HF_TRL_ENVIRONMENT_LOCK_FILENAME,
                b"transformers==5.13.1\ntrl==1.8.0\n".as_slice(),
            ),
            (
                HF_TRL_DATASET_FILENAME,
                b"{\"messages\":[],\"completions\":[]}\n".as_slice(),
            ),
        ] {
            write(root, path, bytes);
        }
        let base_weight_shard_manifest = BaseWeightShardManifest::new(vec![
            BaseWeightShardIdentity::new(
                "model.safetensors",
                4,
                crate::train_receipt::sha256_bytes(b"base"),
            )
            .unwrap(),
        ])
        .unwrap();
        let model = HfTrlModelIdentity {
            served_model_id: "tiny-model".to_string(),
            base_weight_shard_manifest,
            tokenizer_vocab_sha256: crate::train_receipt::sha256_bytes(b"vocab"),
            model_config: artifact(root, HF_TRL_MODEL_CONFIG_FILENAME),
            tokenizer: artifact(root, HF_TRL_TOKENIZER_FILENAME),
            chat_template: artifact(root, HF_TRL_CHAT_TEMPLATE_FILENAME),
            native_training_chat_template: artifact(root, HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME),
            trl_training_chat_template: artifact(root, HF_TRL_TRAINING_TEMPLATE_FILENAME),
        };
        let source_execution_provenance = source_execution(&model);
        let data = HfTrlDataExport {
            source_name: "rollouts".to_string(),
            format: HfTrlDatasetFormat::GrpoGroupsJsonl,
            row_count: 1,
            ordered_corpus_sha256: crate::train_receipt::sha256_bytes(b"ordered-rollouts"),
            dataset: artifact(root, HF_TRL_DATASET_FILENAME),
            sft_selection: None,
            rollout_provenance_schema: Some(ROLLOUT_PROVENANCE_SCHEMA_V1.to_string()),
            split_manifest: None,
        };
        HfTrlExportManifestV1::new(
            HfTrlTask::Grpo,
            source_execution_provenance,
            model,
            data,
            artifact(root, HF_TRL_REFERENCE_SCRIPT_FILENAME),
            artifact(root, HF_TRL_ENVIRONMENT_LOCK_FILENAME),
            None,
        )
        .unwrap()
    }

    fn sft_export_bundle(root: &Path) -> HfTrlExportManifestV1 {
        let base = export_bundle(root);
        let tokenizer = kiln_core::tokenizer::KilnTokenizer::from_bytes(
            br#"{
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1},
                    "merges": []
                }
            }"#,
        )
        .unwrap()
        .with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        );
        let prepared = crate::prepare_sft_examples(
            [crate::SftExample {
                messages: vec![
                    crate::ChatMessage::new("user", "a"),
                    crate::ChatMessage::new("assistant", "b"),
                ],
            }],
            &tokenizer,
            SftInvalidRowPolicy::Fail,
            "inline",
            None,
        )
        .unwrap();
        let mut dataset = serde_json::to_vec(&prepared.examples[0]).unwrap();
        dataset.push(b'\n');
        write(root, HF_TRL_DATASET_FILENAME, &dataset);
        write(
            root,
            HF_TRL_SFT_INGESTION_FILENAME,
            &serde_json::to_vec_pretty(&prepared.ingestion).unwrap(),
        );
        let selection = HfTrlSftSelection::from_ingestion(
            &prepared.ingestion,
            artifact(root, HF_TRL_SFT_INGESTION_FILENAME),
        )
        .unwrap();
        let data = HfTrlDataExport {
            source_name: prepared.ingestion.source.clone(),
            format: HfTrlDatasetFormat::SftMessagesJsonl,
            row_count: 1,
            ordered_corpus_sha256: prepared.ingestion.kept_corpus_sha256.clone(),
            dataset: artifact(root, HF_TRL_DATASET_FILENAME),
            sft_selection: Some(selection),
            rollout_provenance_schema: None,
            split_manifest: None,
        };
        HfTrlExportManifestV1::new(
            HfTrlTask::Sft,
            base.source_execution_provenance,
            base.model,
            data,
            base.reference_script,
            base.environment_lock,
            None,
        )
        .unwrap()
    }

    fn training_result(root: &Path, export: &HfTrlExportManifestV1) -> HfTrlTrainingResultV1 {
        write(root, HF_TRL_ADAPTER_CONFIG_FILENAME, b"{\"r\":8}");
        write(root, HF_TRL_ADAPTER_MODEL_FILENAME, b"adapter");
        write(root, HF_TRL_EXECUTED_SCRIPT_FILENAME, b"print('train')\n");
        let trainer = HfTrlTrainerIdentity {
            kind: HfTrlTrainerKind::TrlGrpoTrainer,
            python_version: "3.13.5".to_string(),
            torch_version: "2.13.0".to_string(),
            transformers_version: "5.13.1".to_string(),
            trl_version: "1.8.0".to_string(),
            peft_version: "0.18.0".to_string(),
            script: artifact(root, HF_TRL_EXECUTED_SCRIPT_FILENAME),
        };
        HfTrlTrainingResultV1::new(
            export.export_sha256.clone(),
            HfTrlTask::Grpo,
            trainer,
            BTreeMap::from([
                (
                    "learning_rate".to_string(),
                    HfTrlConfigValue::Decimal("0.0001".to_string()),
                ),
                (
                    "gradient_accumulation_steps".to_string(),
                    HfTrlConfigValue::Unsigned(4),
                ),
            ]),
            HfTrlOutputAdapter {
                config: artifact(root, HF_TRL_ADAPTER_CONFIG_FILENAME),
                model: artifact(root, HF_TRL_ADAPTER_MODEL_FILENAME),
            },
        )
        .unwrap()
    }

    #[test]
    fn export_and_result_bind_every_referenced_file() {
        let dir = tempfile::tempdir().unwrap();
        let export = export_bundle(dir.path());
        export.validate().unwrap();
        export.verify_files(dir.path()).unwrap();

        write(
            dir.path(),
            HF_TRL_ENVIRONMENT_LOCK_FILENAME,
            b"tampered lock\n",
        );
        let error = export.verify_files(dir.path()).unwrap_err();
        assert!(error.to_string().contains("differs"), "{error:#}");
        write(
            dir.path(),
            HF_TRL_ENVIRONMENT_LOCK_FILENAME,
            b"transformers==5.13.1\ntrl==1.8.0\n",
        );

        let result = training_result(dir.path(), &export);
        result.validate_against_export(&export).unwrap();
        assert!(result.uses_exported_reference_script(&export).unwrap());
        result.verify_files(dir.path()).unwrap();

        write(dir.path(), HF_TRL_ADAPTER_MODEL_FILENAME, b"tampered");
        let error = result.verify_files(dir.path()).unwrap_err();
        assert!(error.to_string().contains("differs"), "{error:#}");
    }

    #[test]
    fn sft_export_verifies_full_ingestion_receipt() {
        let dir = tempfile::tempdir().unwrap();
        let mut export = sft_export_bundle(dir.path());
        export.verify_files(dir.path()).unwrap();

        let receipt_path = dir.path().join(HF_TRL_SFT_INGESTION_FILENAME);
        let mut receipt: crate::SftIngestionReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path).unwrap()).unwrap();
        receipt.source = "dataset_path".to_string();
        write(
            dir.path(),
            HF_TRL_SFT_INGESTION_FILENAME,
            &serde_json::to_vec_pretty(&receipt).unwrap(),
        );
        export
            .data
            .sft_selection
            .as_mut()
            .unwrap()
            .ingestion_receipt = artifact(dir.path(), HF_TRL_SFT_INGESTION_FILENAME);
        export.export_sha256 = export.compute_sha256().unwrap();
        let error = export.verify_files(dir.path()).unwrap_err();
        assert!(error.to_string().contains("source_name"), "{error:#}");
    }

    #[test]
    fn manifest_digests_reject_semantic_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let mut export = export_bundle(dir.path());
        let mut export_identity = serde_json::to_value(&export).unwrap();
        export_identity
            .as_object_mut()
            .unwrap()
            .remove("export_sha256");
        assert_eq!(
            canonical_json_sha256(&export_identity).unwrap(),
            export.export_sha256
        );
        export.data.row_count = 2;
        let error = export.validate().unwrap_err();
        assert!(error.to_string().contains("digest differs"), "{error:#}");

        let export = export_bundle(dir.path());
        let mut result = training_result(dir.path(), &export);
        let mut result_identity = serde_json::to_value(&result).unwrap();
        result_identity
            .as_object_mut()
            .unwrap()
            .remove("result_sha256");
        assert_eq!(
            canonical_json_sha256(&result_identity).unwrap(),
            result.result_sha256
        );
        result.effective_config.insert(
            "learning_rate".to_string(),
            HfTrlConfigValue::Decimal("0.0002".to_string()),
        );
        let error = result.validate().unwrap_err();
        assert!(error.to_string().contains("digest differs"), "{error:#}");
    }

    #[test]
    fn result_records_custom_script_and_rejects_a_different_task() {
        let dir = tempfile::tempdir().unwrap();
        let export = export_bundle(dir.path());
        let mut result = training_result(dir.path(), &export);
        write(
            dir.path(),
            HF_TRL_EXECUTED_SCRIPT_FILENAME,
            b"print('custom train')\n",
        );
        result.trainer.script = artifact(dir.path(), HF_TRL_EXECUTED_SCRIPT_FILENAME);
        result.result_sha256 = result.compute_sha256().unwrap();
        result.validate_against_export(&export).unwrap();
        result.verify_files(dir.path()).unwrap();
        assert!(!result.uses_exported_reference_script(&export).unwrap());

        result.task = HfTrlTask::Sft;
        result.result_sha256 = result.compute_sha256().unwrap();
        let error = result.validate().unwrap_err();
        assert!(error.to_string().contains("trainer kind"), "{error:#}");
    }

    #[test]
    fn strict_read_rejects_unknown_fields_and_symlink_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let export = export_bundle(dir.path());
        let mut value = serde_json::to_value(&export).unwrap();
        value["ignored_general_trainer_knob"] = json!(true);
        write(
            dir.path(),
            HF_TRL_EXPORT_MANIFEST_FILENAME,
            &serde_json::to_vec(&value).unwrap(),
        );
        let error = read_hf_trl_export_manifest(dir.path()).unwrap_err();
        assert!(format!("{error:#}").contains("unknown field"), "{error:#}");

        let result = training_result(dir.path(), &export);
        let value = serde_json::to_string(&result).unwrap();
        let duplicate = value.replacen(
            "\"effective_config\":{",
            "\"effective_config\":{\"learning_rate\":{\"kind\":\"decimal\",\"value\":\"0.0001\"},",
            1,
        );
        write(
            dir.path(),
            HF_TRL_RESULT_MANIFEST_FILENAME,
            duplicate.as_bytes(),
        );
        let error = read_hf_trl_training_result(dir.path()).unwrap_err();
        assert!(
            format!("{error:#}").contains("duplicate effective_config key"),
            "{error:#}"
        );

        #[cfg(unix)]
        {
            std::fs::remove_file(dir.path().join(HF_TRL_DATASET_FILENAME)).unwrap();
            std::os::unix::fs::symlink(
                dir.path().join(HF_TRL_TOKENIZER_FILENAME),
                dir.path().join(HF_TRL_DATASET_FILENAME),
            )
            .unwrap();
            let error = export.verify_files(dir.path()).unwrap_err();
            assert!(error.to_string().contains("symlink"), "{error:#}");
        }
    }

    #[test]
    fn paths_and_task_specific_data_are_fail_closed() {
        assert!(HfTrlFileIdentity::from_bytes("empty", b"").is_err());
        assert!(HfTrlFileIdentity::from_bytes("../adapter", b"x").is_err());
        assert!(HfTrlFileIdentity::from_bytes("C:\\adapter", b"x").is_err());
        assert!(HfTrlFileIdentity::from_bytes("a//b", b"x").is_err());
        assert!(HfTrlFileIdentity::from_bytes("a/./b", b"x").is_err());

        let invalid_decimal = HfTrlConfigValue::Decimal("NaN".to_string());
        assert!(invalid_decimal.validate("learning_rate").is_err());

        let dir = tempfile::tempdir().unwrap();
        let mut provenance_mismatch = export_bundle(dir.path());
        provenance_mismatch.model.tokenizer.sha256 =
            crate::train_receipt::sha256_bytes(b"other-tokenizer");
        provenance_mismatch.export_sha256 = provenance_mismatch.compute_sha256().unwrap();
        let error = provenance_mismatch.validate().unwrap_err();
        assert!(error.to_string().contains("source execution"), "{error:#}");

        let mut export = export_bundle(dir.path());
        export.data.rollout_provenance_schema = None;
        export.export_sha256 = export.compute_sha256().unwrap();
        let error = export.validate().unwrap_err();
        assert!(error.to_string().contains(ROLLOUT_PROVENANCE_SCHEMA_V1));
    }
}
