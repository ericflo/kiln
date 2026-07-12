//! Atomic construction of immutable HF/TRL handoff directories.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::DirBuilderExt;

use anyhow::{Context, Result, bail, ensure};
use uuid::Uuid;

use crate::adapter_output::{ADAPTER_MANIFEST_FILENAME, read_adapter_manifest};
use crate::{
    HF_TRL_ADAPTER_CONFIG_FILENAME, HF_TRL_ADAPTER_MODEL_FILENAME, HF_TRL_CHAT_TEMPLATE_FILENAME,
    HF_TRL_DATASET_FILENAME, HF_TRL_ENVIRONMENT_LOCK_FILENAME, HF_TRL_EXECUTED_SCRIPT_FILENAME,
    HF_TRL_EXPORT_MANIFEST_FILENAME, HF_TRL_IMPORT_RECEIPT_FILENAME, HF_TRL_MODEL_CONFIG_FILENAME,
    HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME, HF_TRL_REFERENCE_SCRIPT_FILENAME,
    HF_TRL_RESULT_MANIFEST_FILENAME, HF_TRL_SFT_INGESTION_FILENAME, HF_TRL_SPLIT_MANIFEST_FILENAME,
    HF_TRL_TOKENIZER_FILENAME, HF_TRL_TRAINING_TEMPLATE_FILENAME, HfTrlDataExport,
    HfTrlDatasetFormat, HfTrlExportManifestV1, HfTrlFileIdentity, HfTrlInputAdapter,
    HfTrlModelIdentity, HfTrlSftSelection, HfTrlTask, HfTrlTrainingResultV1,
    ROLLOUT_PROVENANCE_SCHEMA_V1, SftPreparedDataset,
};

pub const HF_TRL_BUNDLE_SUFFIX: &str = ".kiln-hf";
pub const HF_TRL_IMPORT_ENVELOPE_SUFFIX: &str = ".kiln-hf-import";
pub const HF_TRL_IMPORT_MAX_ARCHIVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const HF_TRL_IMPORT_MAX_EXPANDED_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const HF_TRL_IMPORT_MAX_ARCHIVE_ENTRIES: usize = 32;
pub const HF_TRL_IMPORT_MAX_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;
pub const HF_TRL_IMPORT_MAX_SCRIPT_BYTES: u64 = 16 * 1024 * 1024;
pub const HF_TRL_IMPORT_MAX_ADAPTER_CONFIG_BYTES: u64 = 1024 * 1024;
pub const HF_TRL_IMPORT_MAX_AUXILIARY_BYTES: u64 = 512 * 1024 * 1024;
pub const HF_TRL_IMPORT_MAX_SAFETENSORS_HEADER_BYTES: u64 = 16 * 1024 * 1024;
pub const HF_TRL_IMPORT_MAX_TAR_ZERO_PADDING_BYTES: u64 = 10 * 1024;
pub const HF_TRL_IMPORTED_ADAPTER_FILES: [&str; 6] = [
    HF_TRL_ADAPTER_CONFIG_FILENAME,
    HF_TRL_ADAPTER_MODEL_FILENAME,
    HF_TRL_EXECUTED_SCRIPT_FILENAME,
    HF_TRL_EXPORT_MANIFEST_FILENAME,
    HF_TRL_RESULT_MANIFEST_FILENAME,
    HF_TRL_IMPORT_RECEIPT_FILENAME,
];

/// Validate the adapter name used as both an import archive root and a
/// resident adapter directory name.
pub fn validate_hf_trl_import_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= 128
            && name.bytes().enumerate().all(|(index, byte)| {
                byte.is_ascii_alphanumeric() || (index > 0 && matches!(byte, b'-' | b'_' | b'.'))
            })
            && !name.contains(".."),
        "adapter name must be 1..=128 ASCII bytes, start with an alphanumeric character, contain only alphanumerics, '-', '_' or '.', and not contain '..'"
    );
    Ok(())
}

pub const HF_TRL_SFT_REFERENCE_SCRIPT: &[u8] =
    include_bytes!("../../../scripts/hf_trl/train_sft.py");
pub const HF_TRL_SFT_ENVIRONMENT_LOCK: &[u8] =
    include_bytes!("../../../scripts/hf_trl/requirements-sft.lock");
/// Task-specific name for the shared, task-aware pinned reference runner.
pub const HF_TRL_GRPO_REFERENCE_SCRIPT: &[u8] = HF_TRL_SFT_REFERENCE_SCRIPT;
/// Task-specific name for the shared pinned HF/TRL/PEFT environment.
pub const HF_TRL_GRPO_ENVIRONMENT_LOCK: &[u8] = HF_TRL_SFT_ENVIRONMENT_LOCK;

/// Optional Kiln PEFT adapter to copy into an HF/TRL handoff.
#[derive(Debug, Clone, Copy)]
pub struct HfTrlInputAdapterSource<'a> {
    pub name: &'a str,
    pub directory: &'a Path,
}

/// Already-admitted SFT data and immutable resident identities needed to
/// construct one handoff. Callers retain ownership of every input while the
/// synchronous snapshot is written.
pub struct HfTrlSftBundleInput<'a> {
    pub served_model_id: &'a str,
    pub model_config: &'a kiln_core::config::ModelConfig,
    pub tokenizer: &'a kiln_core::tokenizer::KilnTokenizer,
    pub base_weight_shard_manifest: &'a kiln_core::model_provenance::BaseWeightShardManifest,
    pub source_execution_provenance: &'a kiln_core::execution_provenance::ExecutionProvenanceV1,
    pub prepared: &'a SftPreparedDataset,
    pub reference_script: &'a [u8],
    pub environment_lock: &'a [u8],
    pub split_manifest: Option<&'a [u8]>,
    pub input_adapter: Option<HfTrlInputAdapterSource<'a>>,
}

/// Canonical GRPO groups to snapshot into an immutable handoff.
#[derive(Debug, Clone, Copy)]
pub enum HfTrlGrpoDatasetSource<'a> {
    Groups {
        source_name: &'a str,
        groups: &'a [crate::GrpoGroup],
    },
    Jsonl {
        source_name: &'a str,
        path: &'a Path,
    },
}

impl<'a> HfTrlGrpoDatasetSource<'a> {
    fn source_name(self) -> &'a str {
        match self {
            Self::Groups { source_name, .. } | Self::Jsonl { source_name, .. } => source_name,
        }
    }
}

/// Exact resident identities and rollout source required to construct one
/// immutable GRPO handoff.
pub struct HfTrlGrpoBundleInput<'a> {
    pub served_model_id: &'a str,
    pub model_config: &'a kiln_core::config::ModelConfig,
    pub tokenizer: &'a kiln_core::tokenizer::KilnTokenizer,
    pub base_weight_shard_manifest: &'a kiln_core::model_provenance::BaseWeightShardManifest,
    pub source_execution_provenance: &'a kiln_core::execution_provenance::ExecutionProvenanceV1,
    pub dataset: HfTrlGrpoDatasetSource<'a>,
    pub reference_script: &'a [u8],
    pub environment_lock: &'a [u8],
    pub split_manifest: Option<&'a [u8]>,
    pub input_adapter: Option<HfTrlInputAdapterSource<'a>>,
}

#[derive(Clone, Copy)]
struct HfTrlCommonBundleInput<'a> {
    served_model_id: &'a str,
    model_config: &'a kiln_core::config::ModelConfig,
    tokenizer: &'a kiln_core::tokenizer::KilnTokenizer,
    base_weight_shard_manifest: &'a kiln_core::model_provenance::BaseWeightShardManifest,
    source_execution_provenance: &'a kiln_core::execution_provenance::ExecutionProvenanceV1,
    reference_script: &'a [u8],
    environment_lock: &'a [u8],
    split_manifest: Option<&'a [u8]>,
    input_adapter: Option<HfTrlInputAdapterSource<'a>>,
}

impl<'a> From<&HfTrlSftBundleInput<'a>> for HfTrlCommonBundleInput<'a> {
    fn from(input: &HfTrlSftBundleInput<'a>) -> Self {
        Self {
            served_model_id: input.served_model_id,
            model_config: input.model_config,
            tokenizer: input.tokenizer,
            base_weight_shard_manifest: input.base_weight_shard_manifest,
            source_execution_provenance: input.source_execution_provenance,
            reference_script: input.reference_script,
            environment_lock: input.environment_lock,
            split_manifest: input.split_manifest,
            input_adapter: input.input_adapter,
        }
    }
}

impl<'a> From<&HfTrlGrpoBundleInput<'a>> for HfTrlCommonBundleInput<'a> {
    fn from(input: &HfTrlGrpoBundleInput<'a>) -> Self {
        Self {
            served_model_id: input.served_model_id,
            model_config: input.model_config,
            tokenizer: input.tokenizer,
            base_weight_shard_manifest: input.base_weight_shard_manifest,
            source_execution_provenance: input.source_execution_provenance,
            reference_script: input.reference_script,
            environment_lock: input.environment_lock,
            split_manifest: input.split_manifest,
            input_adapter: input.input_adapter,
        }
    }
}

struct WrittenCommonArtifacts {
    model: HfTrlModelIdentity,
    reference_script: HfTrlFileIdentity,
    environment_lock: HfTrlFileIdentity,
    split_manifest: Option<HfTrlFileIdentity>,
    input_adapter: Option<HfTrlInputAdapter>,
}

/// Build and atomically publish one immutable SFT handoff directory.
///
/// `target` must end in `.kiln-hf` and must not already exist. The function
/// writes a sibling staging directory, fsyncs and validates the complete
/// bundle, renames it into place, and fsyncs the parent directory.
pub fn write_hf_trl_sft_bundle(
    target: &Path,
    input: HfTrlSftBundleInput<'_>,
) -> Result<HfTrlExportManifestV1> {
    let (parent, target, basename) = prepare_target(target)?;
    validate_sft_input(&input)?;

    let staging = parent.join(format!(".{basename}.incomplete-{}", Uuid::new_v4()));
    create_private_directory(&staging)
        .with_context(|| format!("create HF/TRL staging directory {}", staging.display()))?;
    let guard = StagingGuard::new(staging.clone());
    let common = write_common_artifacts(&staging, HfTrlCommonBundleInput::from(&input))?;
    write_sft_dataset(&staging.join(HF_TRL_DATASET_FILENAME), input.prepared)?;
    write_pretty_json(
        &staging.join(HF_TRL_SFT_INGESTION_FILENAME),
        &input.prepared.ingestion,
    )?;
    let selection = HfTrlSftSelection::from_ingestion(
        &input.prepared.ingestion,
        HfTrlFileIdentity::from_file(&staging, HF_TRL_SFT_INGESTION_FILENAME)?,
    )?;
    let data = HfTrlDataExport {
        source_name: input.prepared.ingestion.source.clone(),
        format: HfTrlDatasetFormat::SftMessagesJsonl,
        row_count: u64::try_from(input.prepared.examples.len())
            .context("HF/TRL SFT row count exceeds u64")?,
        ordered_corpus_sha256: input.prepared.ingestion.kept_corpus_sha256.clone(),
        dataset: HfTrlFileIdentity::from_file(&staging, HF_TRL_DATASET_FILENAME)?,
        sft_selection: Some(selection),
        rollout_provenance_schema: None,
        split_manifest: common.split_manifest,
    };
    let manifest = HfTrlExportManifestV1::new(
        HfTrlTask::Sft,
        input.source_execution_provenance.clone(),
        common.model,
        data,
        common.reference_script,
        common.environment_lock,
        common.input_adapter,
    )?;
    publish_verified_bundle(parent, target, staging, guard, manifest)
}

/// Build and atomically publish one immutable, provenance-complete GRPO
/// handoff. JSONL inputs must already use Kiln's canonical compact row
/// encoding; in-memory groups are serialized into that encoding.
pub fn write_hf_trl_grpo_bundle(
    target: &Path,
    input: HfTrlGrpoBundleInput<'_>,
) -> Result<HfTrlExportManifestV1> {
    let (parent, target, basename) = prepare_target(target)?;
    validate_grpo_input(&input)?;

    let staging = parent.join(format!(".{basename}.incomplete-{}", Uuid::new_v4()));
    create_private_directory(&staging)
        .with_context(|| format!("create HF/TRL staging directory {}", staging.display()))?;
    let guard = StagingGuard::new(staging.clone());
    let common = write_common_artifacts(&staging, HfTrlCommonBundleInput::from(&input))?;
    let dataset = write_grpo_dataset(&staging.join(HF_TRL_DATASET_FILENAME), input.dataset)?;
    let data = HfTrlDataExport {
        source_name: input.dataset.source_name().to_string(),
        format: HfTrlDatasetFormat::GrpoGroupsJsonl,
        row_count: dataset.row_count,
        ordered_corpus_sha256: dataset.ordered_corpus_sha256,
        dataset: HfTrlFileIdentity::from_file(&staging, HF_TRL_DATASET_FILENAME)?,
        sft_selection: None,
        rollout_provenance_schema: Some(ROLLOUT_PROVENANCE_SCHEMA_V1.to_string()),
        split_manifest: common.split_manifest,
    };
    let manifest = HfTrlExportManifestV1::new(
        HfTrlTask::Grpo,
        input.source_execution_provenance.clone(),
        common.model,
        data,
        common.reference_script,
        common.environment_lock,
        common.input_adapter,
    )?;
    publish_verified_bundle(parent, target, staging, guard, manifest)
}

/// Validate a pristine Kiln-created export, including its exact recursive
/// file set. External training copies add result artifacts and therefore do
/// not satisfy this pre-training bundle check.
pub fn verify_hf_trl_export_bundle(root: &Path) -> Result<HfTrlExportManifestV1> {
    let manifest = crate::read_hf_trl_export_manifest(root)?;
    ensure_exact_file_set(root, &manifest)?;
    manifest.verify_files(root)?;
    Ok(manifest)
}

/// Validate a completed external-training copy of a Kiln export.
///
/// The original export remains closed and byte-identical. Exactly four result
/// artifacts are added at its root: the preserved executed script, PEFT
/// configuration and weights, and the result manifest published last.
pub fn verify_hf_trl_completed_bundle(
    root: &Path,
) -> Result<(HfTrlExportManifestV1, HfTrlTrainingResultV1)> {
    let export = crate::read_hf_trl_export_manifest(root)?;
    let result = crate::read_hf_trl_training_result(root)?;
    result.validate_against_export(&export)?;
    let mut expected = expected_export_files(&export);
    expected.extend(expected_result_files());
    ensure_exact_files(root, &expected, "completed bundle")?;
    export.verify_files(root)?;
    result.verify_files(root)?;
    Ok((export, result))
}

/// Validate the minimal upload envelope used by the PEFT import API.
///
/// This deliberately excludes dataset, split, environment-lock, reference
/// runner, and input-adapter bytes. The source export/result identities remain
/// self-verifying, while the included model/tokenizer/template files let the
/// receiving server compare exact bytes to its resident model.
pub fn verify_hf_trl_import_envelope(
    root: &Path,
) -> Result<(HfTrlExportManifestV1, HfTrlTrainingResultV1)> {
    let export = crate::read_hf_trl_export_manifest(root)?;
    let result = crate::read_hf_trl_training_result(root)?;
    result.validate_against_export(&export)?;
    let expected = hf_trl_import_envelope_files(&export);
    ensure_exact_files(root, &expected, "import envelope")?;
    export.verify_model_files(root)?;
    result.verify_files(root)?;
    Ok((export, result))
}

/// Materialize a minimal, immutable import envelope from a fully verified
/// completed bundle. The target must be new and end in
/// `.kiln-hf-import`; publication is a same-filesystem atomic rename.
pub fn write_hf_trl_import_envelope(
    source: &Path,
    target: &Path,
) -> Result<(HfTrlExportManifestV1, HfTrlTrainingResultV1)> {
    let (export, result) = verify_hf_trl_completed_bundle(source)?;
    let (parent, target, basename) =
        prepare_target_with_suffix(target, HF_TRL_IMPORT_ENVELOPE_SUFFIX, "import envelope")?;
    let staging = parent.join(format!(".{basename}.incomplete-{}", Uuid::new_v4()));
    create_private_directory(&staging).with_context(|| {
        format!(
            "create HF/TRL import-envelope staging directory {}",
            staging.display()
        )
    })?;
    let mut guard = StagingGuard::new(staging.clone());

    for relative in hf_trl_import_envelope_files(&export) {
        copy_regular_file(&source.join(&relative), &staging.join(&relative))?;
    }
    let verified = verify_hf_trl_import_envelope(&staging)?;
    ensure!(
        verified.0 == export && verified.1 == result,
        "HF/TRL import envelope identity changed during materialization"
    );
    sync_directory_tree(&staging)?;
    kiln_resource::atomic_rename_noreplace(&staging, &target).with_context(|| {
        format!(
            "publish HF/TRL import envelope {} -> {}",
            staging.display(),
            target.display()
        )
    })?;
    sync_directory(&parent)?;
    guard.disarm();
    let published = verify_hf_trl_import_envelope(&target)?;
    ensure!(
        published.0 == export && published.1 == result,
        "published HF/TRL import envelope identity differs"
    );
    Ok(published)
}

fn create_private_directory(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    builder.mode(0o700);
    builder.create(path)
}

fn write_common_artifacts(
    staging: &Path,
    input: HfTrlCommonBundleInput<'_>,
) -> Result<WrittenCommonArtifacts> {
    let model_config_value =
        serde_json::to_value(input.model_config).context("serialize HF/TRL model config")?;
    let model_config_bytes =
        serde_json::to_vec(&model_config_value).context("encode canonical HF/TRL model config")?;
    write_new_synced_file(
        &staging.join(HF_TRL_MODEL_CONFIG_FILENAME),
        &model_config_bytes,
    )?;
    let tokenizer_bytes = input
        .tokenizer
        .tokenizer_config_json()
        .map_err(|error| anyhow::anyhow!("serialize HF/TRL tokenizer: {error}"))?;
    write_new_synced_file(
        &staging.join(HF_TRL_TOKENIZER_FILENAME),
        tokenizer_bytes.as_bytes(),
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_CHAT_TEMPLATE_FILENAME),
        input
            .tokenizer
            .chat_template()
            .context("HF/TRL export requires a configured chat template")?
            .as_bytes(),
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME),
        input
            .tokenizer
            .training_chat_template()
            .context("HF/TRL export requires a native training template")?
            .as_bytes(),
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_TRAINING_TEMPLATE_FILENAME),
        input
            .tokenizer
            .trl_training_chat_template()
            .context("HF/TRL export requires a TRL training template")?
            .as_bytes(),
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_REFERENCE_SCRIPT_FILENAME),
        input.reference_script,
    )?;
    write_new_synced_file(
        &staging.join(HF_TRL_ENVIRONMENT_LOCK_FILENAME),
        input.environment_lock,
    )?;
    let split_manifest = if let Some(bytes) = input.split_manifest {
        write_new_synced_file(&staging.join(HF_TRL_SPLIT_MANIFEST_FILENAME), bytes)?;
        Some(HfTrlFileIdentity::from_file(
            staging,
            HF_TRL_SPLIT_MANIFEST_FILENAME,
        )?)
    } else {
        None
    };
    let input_adapter = input
        .input_adapter
        .map(|source| copy_input_adapter(staging, source))
        .transpose()?;
    let model = HfTrlModelIdentity {
        served_model_id: input.served_model_id.to_string(),
        base_weight_shard_manifest: input.base_weight_shard_manifest.clone(),
        tokenizer_vocab_sha256: input.tokenizer.vocab_identity_sha256(),
        model_config: HfTrlFileIdentity::from_file(staging, HF_TRL_MODEL_CONFIG_FILENAME)?,
        tokenizer: HfTrlFileIdentity::from_file(staging, HF_TRL_TOKENIZER_FILENAME)?,
        chat_template: HfTrlFileIdentity::from_file(staging, HF_TRL_CHAT_TEMPLATE_FILENAME)?,
        native_training_chat_template: HfTrlFileIdentity::from_file(
            staging,
            HF_TRL_NATIVE_TRAINING_TEMPLATE_FILENAME,
        )?,
        trl_training_chat_template: HfTrlFileIdentity::from_file(
            staging,
            HF_TRL_TRAINING_TEMPLATE_FILENAME,
        )?,
    };
    Ok(WrittenCommonArtifacts {
        model,
        reference_script: HfTrlFileIdentity::from_file(staging, HF_TRL_REFERENCE_SCRIPT_FILENAME)?,
        environment_lock: HfTrlFileIdentity::from_file(staging, HF_TRL_ENVIRONMENT_LOCK_FILENAME)?,
        split_manifest,
        input_adapter,
    })
}

fn publish_verified_bundle(
    parent: PathBuf,
    target: PathBuf,
    staging: PathBuf,
    mut guard: StagingGuard,
    manifest: HfTrlExportManifestV1,
) -> Result<HfTrlExportManifestV1> {
    write_pretty_json(&staging.join(HF_TRL_EXPORT_MANIFEST_FILENAME), &manifest)?;
    ensure_exact_file_set(&staging, &manifest)?;
    manifest.verify_files(&staging)?;
    let loaded = crate::read_hf_trl_export_manifest(&staging)?;
    ensure!(
        loaded == manifest,
        "published HF/TRL manifest differs after serialization"
    );
    sync_directory_tree(&staging)?;
    kiln_resource::atomic_rename_noreplace(&staging, &target).with_context(|| {
        format!(
            "publish HF/TRL bundle {} -> {}",
            staging.display(),
            target.display()
        )
    })?;
    sync_directory(&parent)?;
    guard.disarm();
    let published = verify_hf_trl_export_bundle(&target)?;
    ensure!(
        published == manifest,
        "published HF/TRL bundle identity differs after rename"
    );
    Ok(manifest)
}

fn validate_common_input(input: HfTrlCommonBundleInput<'_>) -> Result<()> {
    ensure!(
        !input.served_model_id.is_empty()
            && input.served_model_id.trim() == input.served_model_id
            && !input.served_model_id.chars().any(char::is_control),
        "HF/TRL served model ID must be non-empty, trimmed, and contain no control characters"
    );
    input
        .base_weight_shard_manifest
        .validate()
        .context("validate HF/TRL base-weight source")?;
    input
        .source_execution_provenance
        .validate()
        .context("validate HF/TRL source execution provenance")?;
    input
        .tokenizer
        .chat_template()
        .context("HF/TRL export requires a configured chat template")?;
    input
        .tokenizer
        .training_chat_template()
        .context("HF/TRL export requires a native training template")?;
    input
        .tokenizer
        .trl_training_chat_template()
        .context("HF/TRL export requires a TRL training template")?;
    ensure!(
        !input.reference_script.is_empty(),
        "HF/TRL reference script must not be empty"
    );
    let reference_script = std::str::from_utf8(input.reference_script)
        .context("HF/TRL reference script must be UTF-8 text")?;
    ensure!(
        !reference_script.contains('\0'),
        "HF/TRL reference script must not contain NUL"
    );
    ensure!(
        !input.environment_lock.is_empty(),
        "HF/TRL environment lock must not be empty"
    );
    let environment_lock = std::str::from_utf8(input.environment_lock)
        .context("HF/TRL environment lock must be UTF-8 text")?;
    ensure!(
        !environment_lock.contains('\0'),
        "HF/TRL environment lock must not contain NUL"
    );
    if let Some(split) = input.split_manifest {
        ensure!(!split.is_empty(), "HF/TRL split manifest must not be empty");
        let split = serde_json::from_slice::<serde_json::Value>(split)
            .context("parse HF/TRL split manifest as JSON")?;
        ensure!(
            split.is_object(),
            "HF/TRL split manifest must be a JSON object"
        );
    }
    if let Some(adapter) = input.input_adapter {
        ensure!(
            !adapter.name.is_empty()
                && adapter.name.trim() == adapter.name
                && !adapter.name.chars().any(char::is_control),
            "HF/TRL input-adapter name must be non-empty, trimmed, and contain no control characters"
        );
    }
    Ok(())
}

fn validate_sft_input(input: &HfTrlSftBundleInput<'_>) -> Result<()> {
    validate_common_input(HfTrlCommonBundleInput::from(input))?;
    input
        .prepared
        .ingestion
        .validate()
        .context("validate HF/TRL SFT ingestion receipt")?;
    let (max_seq_len, max_supervised_tokens) = crate::verify_prepared_sft_examples(
        &input.prepared.examples,
        input.tokenizer,
        &input.prepared.ingestion,
    )?;
    ensure!(
        max_seq_len == input.prepared.max_seq_len
            && max_supervised_tokens == input.prepared.max_supervised_tokens,
        "HF/TRL prepared SFT sizing differs from revalidation"
    );
    let trl_template = input
        .tokenizer
        .trl_training_chat_template()
        .context("HF/TRL SFT export requires a TRL training template")?;
    ensure!(
        contains_generation_tag(trl_template, false) && contains_generation_tag(trl_template, true),
        "HF/TRL SFT export requires generation/endgeneration spans in its TRL template"
    );
    Ok(())
}

fn validate_grpo_input(input: &HfTrlGrpoBundleInput<'_>) -> Result<()> {
    validate_common_input(HfTrlCommonBundleInput::from(input))?;
    let source_name = input.dataset.source_name();
    ensure!(
        !source_name.is_empty()
            && source_name.len() <= 512
            && source_name.trim() == source_name
            && !source_name.chars().any(char::is_control),
        "HF/TRL GRPO source name must be 1..=512 bytes, trimmed, and contain no control characters"
    );
    match input.dataset {
        HfTrlGrpoDatasetSource::Groups { groups, .. } => ensure!(
            !groups.is_empty()
                && u64::try_from(groups.len()).ok().is_some_and(|count| {
                    count <= crate::hf_grpo_interop::HF_TRL_GRPO_MAX_GROUPS
                }),
            "HF/TRL GRPO group source must contain 1..={} groups",
            crate::hf_grpo_interop::HF_TRL_GRPO_MAX_GROUPS
        ),
        HfTrlGrpoDatasetSource::Jsonl { path, .. } => {
            open_regular_grpo_source(path)?;
        }
    }
    Ok(())
}

fn open_regular_grpo_source(path: &Path) -> Result<File> {
    let before = fs::symlink_metadata(path)
        .with_context(|| format!("inspect HF/TRL GRPO source {}", path.display()))?;
    ensure!(
        before.file_type().is_file() && !before.file_type().is_symlink(),
        "HF/TRL GRPO JSONL source must be a regular file"
    );
    let file =
        File::open(path).with_context(|| format!("open HF/TRL GRPO source {}", path.display()))?;
    let opened = file
        .metadata()
        .with_context(|| format!("inspect opened HF/TRL GRPO source {}", path.display()))?;
    ensure!(
        opened.is_file(),
        "opened HF/TRL GRPO JSONL source must be a regular file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            before.dev() == opened.dev() && before.ino() == opened.ino(),
            "HF/TRL GRPO JSONL source changed while being opened"
        );
    }
    ensure!(
        opened.len() > 0 && opened.len() <= crate::hf_grpo_interop::HF_TRL_GRPO_MAX_DATASET_BYTES,
        "HF/TRL GRPO JSONL source must contain 1..={} bytes",
        crate::hf_grpo_interop::HF_TRL_GRPO_MAX_DATASET_BYTES
    );
    Ok(file)
}

struct WrittenGrpoDataset {
    row_count: u64,
    ordered_corpus_sha256: String,
}

struct BoundedGrpoRow {
    bytes: Vec<u8>,
    max_bytes: u64,
}

impl BoundedGrpoRow {
    fn new(max_bytes: u64) -> Self {
        Self {
            bytes: Vec::new(),
            max_bytes,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for BoundedGrpoRow {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next_len = u64::try_from(self.bytes.len())
            .ok()
            .and_then(|length| {
                u64::try_from(bytes.len())
                    .ok()
                    .and_then(|add| length.checked_add(add))
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "HF/TRL GRPO row length overflow",
                )
            })?;
        if next_len > self.max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "HF/TRL GRPO row exceeds {} bytes",
                    self.max_bytes.saturating_add(1)
                ),
            ));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|error| io::Error::other(format!("reserve HF/TRL GRPO row: {error}")))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn write_grpo_dataset(
    path: &Path,
    source: HfTrlGrpoDatasetSource<'_>,
) -> Result<WrittenGrpoDataset> {
    let mut output = create_new_file(path)?;
    let mut digest = crate::hf_grpo_interop::GrpoCorpusDigest::new();
    let mut row_count = 0u64;
    let mut written_bytes = 0u64;
    match source {
        HfTrlGrpoDatasetSource::Groups { groups, .. } => {
            for group in groups {
                let mut row =
                    BoundedGrpoRow::new(crate::hf_grpo_interop::HF_TRL_GRPO_MAX_ROW_BYTES - 1);
                serde_json::to_writer(&mut row, group)
                    .with_context(|| format!("serialize HF/TRL GRPO row {}", row_count + 1))?;
                let row = row.into_inner();
                append_grpo_row(
                    &mut output,
                    &mut digest,
                    &mut row_count,
                    &mut written_bytes,
                    &row,
                )?;
            }
        }
        HfTrlGrpoDatasetSource::Jsonl { path: source, .. } => {
            let source_file = open_regular_grpo_source(source)?;
            let mut reader = BufReader::with_capacity(256 * 1024, source_file);
            let mut row = Vec::new();
            while crate::hf_grpo_interop::read_canonical_grpo_row(
                &mut reader,
                row_count + 1,
                &mut row,
            )?
            .is_some()
            {
                append_grpo_row(
                    &mut output,
                    &mut digest,
                    &mut row_count,
                    &mut written_bytes,
                    &row,
                )?;
            }
        }
    }
    ensure!(row_count > 0, "HF/TRL GRPO dataset contains no groups");
    output
        .sync_all()
        .with_context(|| format!("sync HF/TRL GRPO dataset {}", path.display()))?;
    Ok(WrittenGrpoDataset {
        row_count,
        ordered_corpus_sha256: digest.finish(),
    })
}

fn append_grpo_row(
    output: &mut File,
    digest: &mut crate::hf_grpo_interop::GrpoCorpusDigest,
    row_count: &mut u64,
    written_bytes: &mut u64,
    row: &[u8],
) -> Result<()> {
    let row_bytes = u64::try_from(row.len())
        .context("HF/TRL GRPO row length exceeds u64")?
        .checked_add(1)
        .context("HF/TRL GRPO row length overflow")?;
    ensure!(
        row_bytes <= crate::hf_grpo_interop::HF_TRL_GRPO_MAX_ROW_BYTES,
        "HF/TRL GRPO row {} exceeds {} bytes",
        *row_count + 1,
        crate::hf_grpo_interop::HF_TRL_GRPO_MAX_ROW_BYTES
    );
    *row_count = row_count
        .checked_add(1)
        .context("HF/TRL GRPO row-count overflow")?;
    ensure!(
        *row_count <= crate::hf_grpo_interop::HF_TRL_GRPO_MAX_GROUPS,
        "HF/TRL GRPO dataset exceeds {} groups",
        crate::hf_grpo_interop::HF_TRL_GRPO_MAX_GROUPS
    );
    *written_bytes = written_bytes
        .checked_add(row_bytes)
        .context("HF/TRL GRPO dataset-size overflow")?;
    ensure!(
        *written_bytes <= crate::hf_grpo_interop::HF_TRL_GRPO_MAX_DATASET_BYTES,
        "HF/TRL GRPO dataset exceeds {} bytes",
        crate::hf_grpo_interop::HF_TRL_GRPO_MAX_DATASET_BYTES
    );
    output
        .write_all(row)
        .context("write canonical HF/TRL GRPO row")?;
    output
        .write_all(b"\n")
        .context("write HF/TRL GRPO row delimiter")?;
    digest.observe(row)
}

fn contains_generation_tag(template: &str, end: bool) -> bool {
    let name = if end { "endgeneration" } else { "generation" };
    [
        format!("{{% {name} %}}"),
        format!("{{%- {name} %}}"),
        format!("{{% {name} -%}}"),
        format!("{{%- {name} -%}}"),
    ]
    .iter()
    .any(|tag| template.contains(tag))
}

fn prepare_target(target: &Path) -> Result<(PathBuf, PathBuf, String)> {
    prepare_target_with_suffix(target, HF_TRL_BUNDLE_SUFFIX, "bundle")
}

fn prepare_target_with_suffix(
    target: &Path,
    suffix: &str,
    kind: &str,
) -> Result<(PathBuf, PathBuf, String)> {
    let basename = target
        .file_name()
        .and_then(|name| name.to_str())
        .context("HF/TRL target must have a UTF-8 basename")?;
    ensure!(
        basename.ends_with(suffix)
            && basename.len() > suffix.len()
            && !basename.chars().any(char::is_control)
            && !basename.contains('\\'),
        "HF/TRL {kind} target basename must be non-empty and end in {suffix}"
    );
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create HF/TRL target parent {}", parent.display()))?;
    let parent = parent
        .canonicalize()
        .with_context(|| format!("resolve HF/TRL target parent {}", parent.display()))?;
    let target = parent.join(basename);
    match fs::symlink_metadata(&target) {
        Ok(_) => bail!("refusing to overwrite HF/TRL {kind} {}", target.display()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stat HF/TRL {kind} target {}", target.display()));
        }
    }
    Ok((parent, target, basename.to_string()))
}

fn copy_input_adapter(
    staging: &Path,
    source: HfTrlInputAdapterSource<'_>,
) -> Result<HfTrlInputAdapter> {
    let root_meta = fs::symlink_metadata(source.directory).with_context(|| {
        format!(
            "stat HF/TRL input adapter directory {}",
            source.directory.display()
        )
    })?;
    ensure!(
        root_meta.file_type().is_dir() && !root_meta.file_type().is_symlink(),
        "HF/TRL input adapter source must be a real directory"
    );
    let destination = staging.join("input_adapter");
    fs::create_dir(&destination)
        .with_context(|| format!("create HF/TRL input adapter dir {}", destination.display()))?;
    copy_regular_file(
        &source.directory.join("adapter_config.json"),
        &destination.join("adapter_config.json"),
    )?;
    copy_regular_file(
        &source.directory.join("adapter_model.safetensors"),
        &destination.join("adapter_model.safetensors"),
    )?;
    let source_manifest = source.directory.join(ADAPTER_MANIFEST_FILENAME);
    let kiln_manifest = match fs::symlink_metadata(&source_manifest) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "HF/TRL input adapter manifest must be a regular file"
            );
            let parsed = read_adapter_manifest(&source_manifest)?;
            ensure!(
                parsed.adapter_name == source.name,
                "HF/TRL input adapter name differs from its Kiln manifest"
            );
            copy_regular_file(
                &source_manifest,
                &destination.join(ADAPTER_MANIFEST_FILENAME),
            )?;
            Some(HfTrlFileIdentity::from_file(
                staging,
                "input_adapter/adapter_manifest.json",
            )?)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "stat HF/TRL input adapter manifest {}",
                    source_manifest.display()
                )
            });
        }
    };
    sync_directory(&destination)?;
    Ok(HfTrlInputAdapter {
        name: source.name.to_string(),
        config: HfTrlFileIdentity::from_file(staging, "input_adapter/adapter_config.json")?,
        model: HfTrlFileIdentity::from_file(staging, "input_adapter/adapter_model.safetensors")?,
        kiln_manifest,
    })
}

fn copy_regular_file(source: &Path, target: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)
        .with_context(|| format!("stat HF/TRL source artifact {}", source.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "HF/TRL source artifact must be a regular file: {}",
        source.display()
    );
    let mut reader = File::open(source)
        .with_context(|| format!("open HF/TRL source artifact {}", source.display()))?;
    let mut writer = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(target)
        .with_context(|| format!("create HF/TRL artifact {}", target.display()))?;
    io::copy(&mut reader, &mut writer).with_context(|| {
        format!(
            "copy HF/TRL artifact {} -> {}",
            source.display(),
            target.display()
        )
    })?;
    writer
        .sync_all()
        .with_context(|| format!("sync HF/TRL artifact {}", target.display()))
}

fn write_sft_dataset(path: &Path, prepared: &SftPreparedDataset) -> Result<()> {
    let mut file = create_new_file(path)?;
    for (index, example) in prepared.examples.iter().enumerate() {
        serde_json::to_writer(&mut file, example)
            .with_context(|| format!("serialize HF/TRL SFT row {}", index + 1))?;
        file.write_all(b"\n")
            .with_context(|| format!("write HF/TRL SFT row {}", index + 1))?;
    }
    file.sync_all()
        .with_context(|| format!("sync HF/TRL dataset {}", path.display()))
}

fn write_pretty_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize HF/TRL JSON artifact {}", path.display()))?;
    write_new_synced_file(path, &bytes)
}

fn write_new_synced_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = create_new_file(path)?;
    file.write_all(bytes)
        .with_context(|| format!("write HF/TRL artifact {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync HF/TRL artifact {}", path.display()))
}

fn create_new_file(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create HF/TRL artifact parent {}", parent.display()))?;
    }
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("create HF/TRL artifact {}", path.display()))
}

fn ensure_exact_file_set(root: &Path, manifest: &HfTrlExportManifestV1) -> Result<()> {
    ensure_exact_files(root, &expected_export_files(manifest), "export bundle")
}

fn expected_export_files(manifest: &HfTrlExportManifestV1) -> BTreeSet<String> {
    let mut expected = BTreeSet::from([
        HF_TRL_EXPORT_MANIFEST_FILENAME.to_string(),
        manifest.model.model_config.relative_path.clone(),
        manifest.model.tokenizer.relative_path.clone(),
        manifest.model.chat_template.relative_path.clone(),
        manifest
            .model
            .native_training_chat_template
            .relative_path
            .clone(),
        manifest
            .model
            .trl_training_chat_template
            .relative_path
            .clone(),
        manifest.data.dataset.relative_path.clone(),
        manifest.reference_script.relative_path.clone(),
        manifest.environment_lock.relative_path.clone(),
    ]);
    if let Some(selection) = manifest.data.sft_selection.as_ref() {
        expected.insert(selection.ingestion_receipt.relative_path.clone());
    }
    if let Some(split) = manifest.data.split_manifest.as_ref() {
        expected.insert(split.relative_path.clone());
    }
    if let Some(adapter) = manifest.input_adapter.as_ref() {
        expected.insert(adapter.config.relative_path.clone());
        expected.insert(adapter.model.relative_path.clone());
        if let Some(kiln_manifest) = adapter.kiln_manifest.as_ref() {
            expected.insert(kiln_manifest.relative_path.clone());
        }
    }
    expected
}

fn expected_result_files() -> BTreeSet<String> {
    BTreeSet::from([
        HF_TRL_RESULT_MANIFEST_FILENAME.to_string(),
        HF_TRL_EXECUTED_SCRIPT_FILENAME.to_string(),
        HF_TRL_ADAPTER_CONFIG_FILENAME.to_string(),
        HF_TRL_ADAPTER_MODEL_FILENAME.to_string(),
    ])
}

/// Exact relative paths transported by the corpus-free import envelope.
pub fn hf_trl_import_envelope_files(export: &HfTrlExportManifestV1) -> BTreeSet<String> {
    BTreeSet::from([
        HF_TRL_EXPORT_MANIFEST_FILENAME.to_string(),
        HF_TRL_RESULT_MANIFEST_FILENAME.to_string(),
        export.model.model_config.relative_path.clone(),
        export.model.tokenizer.relative_path.clone(),
        export.model.chat_template.relative_path.clone(),
        export
            .model
            .native_training_chat_template
            .relative_path
            .clone(),
        export
            .model
            .trl_training_chat_template
            .relative_path
            .clone(),
        HF_TRL_EXECUTED_SCRIPT_FILENAME.to_string(),
        HF_TRL_ADAPTER_CONFIG_FILENAME.to_string(),
        HF_TRL_ADAPTER_MODEL_FILENAME.to_string(),
    ])
}

fn ensure_exact_files(root: &Path, expected: &BTreeSet<String>, kind: &str) -> Result<()> {
    let actual = collect_relative_files(root)?;
    ensure!(
        actual == *expected,
        "HF/TRL {kind} file set differs: expected {expected:?}, found {actual:?}"
    );
    Ok(())
}

fn collect_relative_files(root: &Path) -> Result<BTreeSet<String>> {
    fn visit(root: &Path, directory: &Path, files: &mut BTreeSet<String>) -> Result<()> {
        for entry in fs::read_dir(directory)
            .with_context(|| format!("read HF/TRL directory {}", directory.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "HF/TRL staging tree contains a symlink: {}",
                path.display()
            );
            if metadata.file_type().is_dir() {
                visit(root, &path, files)?;
            } else if metadata.file_type().is_file() {
                let relative = path
                    .strip_prefix(root)
                    .expect("visited path must remain below root")
                    .to_str()
                    .context("HF/TRL artifact path must be UTF-8")?
                    .replace(std::path::MAIN_SEPARATOR, "/");
                ensure!(
                    files.insert(relative.clone()),
                    "duplicate HF/TRL artifact path {relative:?}"
                );
            } else {
                bail!("unsupported HF/TRL staging entry {}", path.display());
            }
        }
        Ok(())
    }

    let mut files = BTreeSet::new();
    visit(root, root, &mut files)?;
    Ok(files)
}

fn sync_directory_tree(root: &Path) -> Result<()> {
    let mut directories = vec![root.to_path_buf()];
    let mut cursor = 0;
    while cursor < directories.len() {
        let current = directories[cursor].clone();
        cursor += 1;
        for entry in fs::read_dir(&current)
            .with_context(|| format!("read HF/TRL directory {} for sync", current.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                directories.push(entry.path());
            }
        }
    }
    for directory in directories.into_iter().rev() {
        sync_directory(&directory)?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("open HF/TRL directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("sync HF/TRL directory {}", path.display()))
}

struct StagingGuard {
    path: PathBuf,
    armed: bool,
}

impl StagingGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use kiln_core::config::ModelConfig;
    use kiln_core::execution_provenance::{
        ExecutionBackendIdentity, ExecutionBuildIdentity, ExecutionConfigurationIdentity,
        ExecutionKernelIdentity, ExecutionModelIdentity, ExecutionPrecisionIdentity,
        ExecutionProvenanceV1,
    };
    use kiln_core::model_provenance::{BaseWeightShardIdentity, BaseWeightShardManifest};
    use kiln_core::tokenizer::KilnTokenizer;

    use super::*;
    use crate::{
        ChatMessage, HfTrlConfigValue, HfTrlOutputAdapter, HfTrlTrainerIdentity, HfTrlTrainerKind,
        SftExample, SftInvalidRowPolicy, prepare_sft_examples,
    };

    #[test]
    fn embedded_reference_assets_are_pinned_task_aware_executable_text() {
        let script = std::str::from_utf8(HF_TRL_SFT_REFERENCE_SCRIPT).unwrap();
        let lock = std::str::from_utf8(HF_TRL_SFT_ENVIRONMENT_LOCK).unwrap();
        assert_eq!(HF_TRL_GRPO_REFERENCE_SCRIPT, HF_TRL_SFT_REFERENCE_SCRIPT);
        assert_eq!(HF_TRL_GRPO_ENVIRONMENT_LOCK, HF_TRL_SFT_ENVIRONMENT_LOCK);
        assert!(script.starts_with("#!/usr/bin/env python3\n"));
        assert!(script.contains("assistant_only_loss=True"));
        assert!(script.contains("class _RecordedRolloutSource"));
        assert!(script.contains("output[\"old_per_token_logps\"] = recorded.detach()"));
        assert!(script.contains("\"env_mask\": env_mask"));
        assert!(script.contains("kiln.hf-trl-result.v1"));
        for package in [
            "torch==2.13.0",
            "transformers==5.13.1",
            "trl==1.8.0",
            "peft==0.19.1",
        ] {
            assert!(lock.lines().any(|line| line == package));
        }
    }

    #[test]
    fn in_memory_grpo_row_serialization_is_bounded_before_growth() {
        let mut row = BoundedGrpoRow::new(3);
        row.write_all(b"abc").unwrap();
        let error = row.write_all(b"d").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(row.into_inner(), b"abc");
    }

    struct Fixture {
        model_config: ModelConfig,
        tokenizer: KilnTokenizer,
        base_weights: BaseWeightShardManifest,
        provenance: ExecutionProvenanceV1,
        prepared: SftPreparedDataset,
    }

    impl Fixture {
        fn input(&self) -> HfTrlSftBundleInput<'_> {
            HfTrlSftBundleInput {
                served_model_id: "test/qwen",
                model_config: &self.model_config,
                tokenizer: &self.tokenizer,
                base_weight_shard_manifest: &self.base_weights,
                source_execution_provenance: &self.provenance,
                prepared: &self.prepared,
                reference_script: b"print('train')\n",
                environment_lock: b"transformers==5.13.1\ntrl==1.8.0\n",
                split_manifest: Some(b"{\"schema\":\"test.split.v1\"}"),
                input_adapter: None,
            }
        }

        fn grpo_input<'a>(
            &'a self,
            dataset: HfTrlGrpoDatasetSource<'a>,
        ) -> HfTrlGrpoBundleInput<'a> {
            HfTrlGrpoBundleInput {
                served_model_id: "test/qwen",
                model_config: &self.model_config,
                tokenizer: &self.tokenizer,
                base_weight_shard_manifest: &self.base_weights,
                source_execution_provenance: &self.provenance,
                dataset,
                reference_script: b"print('train-grpo')\n",
                environment_lock: b"transformers==5.13.1\ntrl==1.8.0\n",
                split_manifest: Some(b"{\"schema\":\"test.split.v1\"}"),
                input_adapter: None,
            }
        }
    }

    fn fixture() -> Fixture {
        let tokenizer = KilnTokenizer::from_bytes(
            br#"{
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "a": 1, "b": 2},
                    "unk_token": "[UNK]"
                },
                "pre_tokenizer": {"type": "Whitespace"}
            }"#,
        )
        .unwrap()
        .with_chat_template(
            include_str!("../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja").to_string(),
        );
        let prepared = prepare_sft_examples(
            [SftExample {
                messages: vec![
                    ChatMessage::new("user", "a"),
                    ChatMessage::new("assistant", "b"),
                ],
            }],
            &tokenizer,
            SftInvalidRowPolicy::Fail,
            "inline",
            None,
        )
        .unwrap();
        let model_config = ModelConfig::qwen3_5_4b();
        let base_weights = BaseWeightShardManifest::new(vec![
            BaseWeightShardIdentity::new(
                "model.safetensors",
                4,
                kiln_core::config_hashes::sha256_bytes(b"base"),
            )
            .unwrap(),
        ])
        .unwrap();
        let hash = |bytes: &[u8]| kiln_core::config_hashes::sha256_bytes(bytes);
        let provenance = ExecutionProvenanceV1::new(
            ExecutionBackendIdentity {
                name: "rocm".to_string(),
                device: "test-gpu".to_string(),
                numerical_runtime_sha256: hash(b"runtime"),
            },
            ExecutionBuildIdentity {
                package_version: "0.4.1".to_string(),
                target: "test-target".to_string(),
                executable_sha256: hash(b"executable"),
                git_commit: Some("a".repeat(40)),
                source_tree_sha256: Some(hash(b"source")),
                source_dirty: Some(false),
            },
            ExecutionModelIdentity {
                model_config_sha256: kiln_core::config_hashes::sha256_json_serializable(
                    &model_config,
                )
                .unwrap(),
                tokenizer_vocab_sha256: tokenizer.vocab_identity_sha256(),
                tokenizer_config_sha256: tokenizer.tokenizer_config_sha256().unwrap(),
                chat_template_sha256: tokenizer.chat_template_sha256(),
                training_chat_template_sha256: tokenizer.training_chat_template_sha256(),
            },
            ExecutionPrecisionIdentity {
                inference_dtype: "bf16".to_string(),
                training_policy: "bf16".to_string(),
            },
            ExecutionKernelIdentity::new(
                BTreeMap::from([("test_kernel".to_string(), "v1".to_string())]),
                vec!["rocm".to_string()],
            )
            .unwrap(),
            ExecutionConfigurationIdentity {
                effective_server_config_sha256: hash(b"server-config"),
                effective_environment_sha256: hash(b"environment"),
            },
        )
        .unwrap();
        Fixture {
            model_config,
            tokenizer,
            base_weights,
            provenance,
            prepared,
        }
    }

    fn grpo_groups(fixture: &Fixture) -> Vec<crate::GrpoGroup> {
        let messages = vec![ChatMessage::new("user", "a")];
        let prompt_text = fixture.tokenizer.apply_chat_template(&messages).unwrap();
        let prompt_ids = fixture.tokenizer.encode(&prompt_text).unwrap();
        let completion_token = fixture.tokenizer.encode("b").unwrap()[0];
        let prompt_sha256 = crate::rollout_prompt_messages_sha256(&messages).unwrap();
        let behavior_policy = crate::RolloutBehaviorPolicyIdentityV1 {
            served_model_id: "test/qwen".to_string(),
            base_model_sha256: fixture.base_weights.aggregate_sha256.clone(),
            adapter: None,
            inference_config_sha256: kiln_core::config_hashes::sha256_bytes(b"inference-config"),
            implementation: "kiln-test/rocm".to_string(),
        };
        let tokenizer = crate::RolloutTokenizerIdentityV1 {
            vocab_sha256: fixture.tokenizer.vocab_identity_sha256(),
            config_sha256: fixture.tokenizer.tokenizer_config_sha256().unwrap(),
            chat_template_sha256: fixture.tokenizer.chat_template_sha256().unwrap(),
        };
        let completions = [(0.0, 7u64), (1.0, 8u64)]
            .into_iter()
            .map(|(reward, seed)| {
                let rollout = crate::ScoredRollout::legacy("b".to_string(), reward);
                let payload_sha256 = crate::scored_rollout_payload_sha256(&rollout).unwrap();
                let mut input_token_ids = prompt_ids.clone();
                input_token_ids.push(completion_token);
                let provenance = crate::RolloutProvenanceV1::new(
                    input_token_ids,
                    prompt_ids.len(),
                    prompt_sha256.clone(),
                    payload_sha256,
                    vec![crate::RolloutActionTokenV1::sampled(
                        prompt_ids.len(),
                        completion_token,
                        -0.5,
                    )],
                    behavior_policy.clone(),
                    tokenizer.clone(),
                    crate::RolloutSamplingConfigV1 {
                        temperature: 0.7,
                        top_p: 0.95,
                        top_k: 20,
                        min_p: 0.0,
                        max_tokens: 1,
                        repetition_penalty: 1.0,
                        presence_penalty: 0.0,
                        frequency_penalty: 0.0,
                        stop: Vec::new(),
                        thinking_budget: None,
                    },
                    seed,
                    "rocm",
                )
                .unwrap();
                rollout.with_provenance(provenance)
            })
            .collect();
        vec![crate::GrpoGroup {
            messages,
            completions,
        }]
    }

    fn add_completed_result(root: &Path, export: &HfTrlExportManifestV1) -> HfTrlTrainingResultV1 {
        fs::write(
            root.join(HF_TRL_EXECUTED_SCRIPT_FILENAME),
            b"print('executed')\n",
        )
        .unwrap();
        fs::write(
            root.join(HF_TRL_ADAPTER_CONFIG_FILENAME),
            br#"{"peft_type":"LORA","r":8,"lora_alpha":16,"target_modules":["q_proj"]}"#,
        )
        .unwrap();
        fs::write(
            root.join(HF_TRL_ADAPTER_MODEL_FILENAME),
            b"test-adapter-weights",
        )
        .unwrap();
        let trainer = HfTrlTrainerIdentity {
            kind: HfTrlTrainerKind::TrlSftTrainer,
            python_version: "3.13.5".to_string(),
            torch_version: "2.13.0".to_string(),
            transformers_version: "5.13.1".to_string(),
            trl_version: "1.8.0".to_string(),
            peft_version: "0.19.1".to_string(),
            script: HfTrlFileIdentity::from_file(root, HF_TRL_EXECUTED_SCRIPT_FILENAME).unwrap(),
        };
        let result = HfTrlTrainingResultV1::new(
            export.export_sha256.clone(),
            HfTrlTask::Sft,
            trainer,
            BTreeMap::from([("seed".to_string(), HfTrlConfigValue::Unsigned(42))]),
            HfTrlOutputAdapter {
                config: HfTrlFileIdentity::from_file(root, HF_TRL_ADAPTER_CONFIG_FILENAME).unwrap(),
                model: HfTrlFileIdentity::from_file(root, HF_TRL_ADAPTER_MODEL_FILENAME).unwrap(),
            },
        )
        .unwrap();
        write_pretty_json(&root.join(HF_TRL_RESULT_MANIFEST_FILENAME), &result).unwrap();
        result
    }

    fn incomplete_entries(parent: &Path) -> Vec<String> {
        fs::read_dir(parent)
            .unwrap()
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().into_string().ok()?;
                name.contains(".incomplete-").then_some(name)
            })
            .collect()
    }

    #[test]
    fn publishes_a_revalidated_immutable_sft_bundle() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("export.kiln-hf");
        let fixture = fixture();
        let manifest = write_hf_trl_sft_bundle(&target, fixture.input()).unwrap();

        assert_eq!(manifest.task, HfTrlTask::Sft);
        assert_eq!(manifest.data.row_count, 1);
        assert!(target.is_dir());
        assert!(incomplete_entries(directory.path()).is_empty());
        assert_eq!(
            crate::read_hf_trl_export_manifest(&target).unwrap(),
            manifest
        );
        manifest.verify_files(&target).unwrap();
        let dataset = fs::read_to_string(target.join(HF_TRL_DATASET_FILENAME)).unwrap();
        assert_eq!(dataset.lines().count(), 1);
        assert_eq!(
            serde_json::from_str::<SftExample>(dataset.trim()).unwrap(),
            fixture.prepared.examples[0]
        );

        let error = write_hf_trl_sft_bundle(&target, fixture.input()).unwrap_err();
        assert!(error.to_string().contains("overwrite"), "{error:#}");

        fs::write(target.join(HF_TRL_DATASET_FILENAME), b"tampered\n").unwrap();
        let error = manifest.verify_files(&target).unwrap_err();
        assert!(error.to_string().contains("differs"), "{error:#}");
    }

    #[test]
    fn publishes_a_deeply_verified_immutable_grpo_bundle() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("grpo.kiln-hf");
        let fixture = fixture();
        let mut groups = grpo_groups(&fixture);
        let expected_row = serde_json::to_vec(&groups[0]).unwrap();
        let manifest = write_hf_trl_grpo_bundle(
            &target,
            fixture.grpo_input(HfTrlGrpoDatasetSource::Groups {
                source_name: "rollouts",
                groups: &groups,
            }),
        )
        .unwrap();

        assert_eq!(manifest.task, HfTrlTask::Grpo);
        assert_eq!(manifest.data.row_count, 1);
        assert_eq!(
            manifest.data.rollout_provenance_schema.as_deref(),
            Some(ROLLOUT_PROVENANCE_SCHEMA_V1)
        );
        assert!(target.is_dir());
        assert!(incomplete_entries(directory.path()).is_empty());
        assert_eq!(verify_hf_trl_export_bundle(&target).unwrap(), manifest);
        let summary = crate::hf_grpo_interop::verify_hf_trl_grpo_corpus(
            &target,
            &manifest.model,
            &manifest.data,
            manifest.input_adapter.as_ref(),
        )
        .unwrap();
        assert_eq!(summary.row_count, 1);
        assert_eq!(summary.completions_per_group, 2);
        assert_eq!(summary.completion_count, 2);
        assert_eq!(summary.sampled_action_tokens, 2);
        assert_eq!(summary.forced_action_tokens, 0);

        let mut expected_dataset = expected_row;
        expected_dataset.push(b'\n');
        assert_eq!(
            fs::read(target.join(HF_TRL_DATASET_FILENAME)).unwrap(),
            expected_dataset
        );
        groups[0].completions[0].reward = 0.25;
        assert_eq!(
            fs::read(target.join(HF_TRL_DATASET_FILENAME)).unwrap(),
            expected_dataset
        );

        let error = write_hf_trl_grpo_bundle(
            &target,
            fixture.grpo_input(HfTrlGrpoDatasetSource::Groups {
                source_name: "rollouts",
                groups: &groups,
            }),
        )
        .unwrap_err();
        assert!(error.to_string().contains("overwrite"), "{error:#}");
        assert!(incomplete_entries(directory.path()).is_empty());

        fs::write(target.join(HF_TRL_DATASET_FILENAME), b"tampered\n").unwrap();
        let error = verify_hf_trl_export_bundle(&target).unwrap_err();
        assert!(error.to_string().contains("differs"), "{error:#}");
    }

    #[test]
    fn canonical_jsonl_and_in_memory_grpo_sources_have_identical_identity() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = fixture();
        let groups = grpo_groups(&fixture);
        let memory_target = directory.path().join("memory.kiln-hf");
        let memory_manifest = write_hf_trl_grpo_bundle(
            &memory_target,
            fixture.grpo_input(HfTrlGrpoDatasetSource::Groups {
                source_name: "rollouts",
                groups: &groups,
            }),
        )
        .unwrap();
        let dataset = fs::read(memory_target.join(HF_TRL_DATASET_FILENAME)).unwrap();
        let source = directory.path().join("rollouts.jsonl");
        fs::write(&source, &dataset).unwrap();

        let jsonl_target = directory.path().join("jsonl.kiln-hf");
        let jsonl_manifest = write_hf_trl_grpo_bundle(
            &jsonl_target,
            fixture.grpo_input(HfTrlGrpoDatasetSource::Jsonl {
                source_name: "rollouts",
                path: &source,
            }),
        )
        .unwrap();
        assert_eq!(jsonl_manifest, memory_manifest);
        assert_eq!(
            fs::read(jsonl_target.join(HF_TRL_DATASET_FILENAME)).unwrap(),
            dataset
        );

        fs::write(&source, b"changed after publication\n").unwrap();
        assert_eq!(
            fs::read(jsonl_target.join(HF_TRL_DATASET_FILENAME)).unwrap(),
            dataset
        );
        assert_eq!(
            verify_hf_trl_export_bundle(&jsonl_target).unwrap(),
            jsonl_manifest
        );
        assert!(incomplete_entries(directory.path()).is_empty());
    }

    #[test]
    fn grpo_bundle_rejects_group_width_the_pinned_runner_cannot_replay() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = fixture();
        let mut groups = grpo_groups(&fixture);
        groups.push(groups[0].clone());
        let extra_completion = groups[1].completions[0].clone();
        groups[1].completions.push(extra_completion);
        let target = directory.path().join("heterogeneous.kiln-hf");
        let error = write_hf_trl_grpo_bundle(
            &target,
            fixture.grpo_input(HfTrlGrpoDatasetSource::Groups {
                source_name: "heterogeneous",
                groups: &groups,
            }),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("uniform group width"));
        assert!(!target.exists());
    }

    #[test]
    fn invalid_grpo_sources_fail_closed_without_publication_debris() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = fixture();
        let groups = grpo_groups(&fixture);
        let canonical = serde_json::to_vec(&groups[0]).unwrap();
        let cases = [
            (
                "noncanonical",
                [canonical.as_slice(), b" \n"].concat(),
                "not canonical compact JSON",
            ),
            ("missing-lf", canonical.clone(), "end every row with LF"),
            (
                "blank-row",
                [canonical.as_slice(), b"\n\n"].concat(),
                "blank row",
            ),
        ];
        for (name, bytes, expected_error) in cases {
            let source = directory.path().join(format!("{name}.jsonl"));
            fs::write(&source, bytes).unwrap();
            let target = directory.path().join(format!("{name}.kiln-hf"));
            let error = write_hf_trl_grpo_bundle(
                &target,
                fixture.grpo_input(HfTrlGrpoDatasetSource::Jsonl {
                    source_name: "rollouts",
                    path: &source,
                }),
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected_error), "{error:#}");
            assert!(!target.exists());
            assert!(incomplete_entries(directory.path()).is_empty());
        }

        let mut invalid_groups = groups;
        invalid_groups[0].completions[0].provenance = None;
        let target = directory.path().join("invalid-group.kiln-hf");
        let error = write_hf_trl_grpo_bundle(
            &target,
            fixture.grpo_input(HfTrlGrpoDatasetSource::Groups {
                source_name: "rollouts",
                groups: &invalid_groups,
            }),
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("missing exact rollout provenance"),
            "{error:#}"
        );
        assert!(!target.exists());
        assert!(incomplete_entries(directory.path()).is_empty());

        #[cfg(unix)]
        {
            let source = directory.path().join("canonical.jsonl");
            fs::write(&source, [canonical.as_slice(), b"\n"].concat()).unwrap();
            let symlink = directory.path().join("rollouts-symlink.jsonl");
            std::os::unix::fs::symlink(&source, &symlink).unwrap();
            let target = directory.path().join("symlink.kiln-hf");
            let error = write_hf_trl_grpo_bundle(
                &target,
                fixture.grpo_input(HfTrlGrpoDatasetSource::Jsonl {
                    source_name: "rollouts",
                    path: &symlink,
                }),
            )
            .unwrap_err();
            assert!(error.to_string().contains("regular file"), "{error:#}");
            assert!(!target.exists());
            assert!(incomplete_entries(directory.path()).is_empty());
        }
    }

    #[test]
    fn input_drift_fails_without_publishing_or_leaving_staging() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("drift.kiln-hf");
        let mut fixture = fixture();
        fixture.prepared.examples[0].messages[1].content = "changed".to_string();

        let error = write_hf_trl_sft_bundle(&target, fixture.input()).unwrap_err();
        assert!(error.to_string().contains("hash differs"), "{error:#}");
        assert!(!target.exists());
        assert!(incomplete_entries(directory.path()).is_empty());
    }

    #[test]
    fn snapshots_optional_input_adapter_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let adapter = directory.path().join("adapter");
        fs::create_dir(&adapter).unwrap();
        fs::write(
            adapter.join("adapter_config.json"),
            b"{\"peft_type\":\"LORA\"}",
        )
        .unwrap();
        fs::write(adapter.join("adapter_model.safetensors"), b"adapter").unwrap();
        let target = directory.path().join("adapter-export.kiln-hf");
        let fixture = fixture();
        let mut input = fixture.input();
        input.split_manifest = None;
        input.input_adapter = Some(HfTrlInputAdapterSource {
            name: "adapter",
            directory: &adapter,
        });

        let manifest = write_hf_trl_sft_bundle(&target, input).unwrap();
        let exported = manifest.input_adapter.as_ref().unwrap();
        assert_eq!(exported.name, "adapter");
        assert!(exported.kiln_manifest.is_none());
        assert_eq!(
            fs::read(target.join(&exported.config.relative_path)).unwrap(),
            b"{\"peft_type\":\"LORA\"}"
        );
        assert_eq!(
            fs::read(target.join(&exported.model.relative_path)).unwrap(),
            b"adapter"
        );
        manifest.verify_files(&target).unwrap();
    }

    #[test]
    fn rejects_bad_suffix_split_and_symlinked_adapter_artifact() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = fixture();
        assert!(
            write_hf_trl_sft_bundle(&directory.path().join("export"), fixture.input()).is_err()
        );

        let target = directory.path().join("split.kiln-hf");
        let mut bad_split = fixture.input();
        bad_split.split_manifest = Some(b"not-json");
        assert!(write_hf_trl_sft_bundle(&target, bad_split).is_err());
        assert!(!target.exists());
        assert!(incomplete_entries(directory.path()).is_empty());

        #[cfg(unix)]
        {
            let adapter = directory.path().join("adapter");
            fs::create_dir(&adapter).unwrap();
            fs::write(adapter.join("real-config.json"), b"{}").unwrap();
            std::os::unix::fs::symlink(
                adapter.join("real-config.json"),
                adapter.join("adapter_config.json"),
            )
            .unwrap();
            fs::write(adapter.join("adapter_model.safetensors"), b"adapter").unwrap();
            let mut with_adapter = fixture.input();
            with_adapter.split_manifest = None;
            with_adapter.input_adapter = Some(HfTrlInputAdapterSource {
                name: "adapter",
                directory: &adapter,
            });
            let error = write_hf_trl_sft_bundle(&target, with_adapter).unwrap_err();
            assert!(error.to_string().contains("regular file"), "{error:#}");
            assert!(!target.exists());
            assert!(incomplete_entries(directory.path()).is_empty());
        }
    }

    #[test]
    fn pristine_verifier_rejects_undeclared_files() {
        let fixture = fixture();
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("verified.kiln-hf");
        write_hf_trl_sft_bundle(&target, fixture.input()).unwrap();
        verify_hf_trl_export_bundle(&target).unwrap();

        fs::write(target.join("undeclared.txt"), b"not in the manifest").unwrap();
        let error = verify_hf_trl_export_bundle(&target).unwrap_err();
        assert!(error.to_string().contains("file set differs"), "{error:#}");
    }

    #[test]
    fn completed_bundle_materializes_a_minimal_atomic_import_envelope() {
        let fixture = fixture();
        let directory = tempfile::tempdir().unwrap();
        let completed = directory.path().join("completed.kiln-hf");
        let export = write_hf_trl_sft_bundle(&completed, fixture.input()).unwrap();
        let result = add_completed_result(&completed, &export);

        assert!(verify_hf_trl_export_bundle(&completed).is_err());
        assert_eq!(
            verify_hf_trl_completed_bundle(&completed).unwrap(),
            (export.clone(), result.clone())
        );

        let envelope = directory.path().join("completed.kiln-hf-import");
        assert_eq!(
            write_hf_trl_import_envelope(&completed, &envelope).unwrap(),
            (export.clone(), result.clone())
        );
        assert_eq!(
            verify_hf_trl_import_envelope(&envelope).unwrap(),
            (export, result)
        );
        assert!(envelope.join(HF_TRL_MODEL_CONFIG_FILENAME).is_file());
        assert!(envelope.join(HF_TRL_ADAPTER_MODEL_FILENAME).is_file());
        for excluded in [
            HF_TRL_DATASET_FILENAME,
            HF_TRL_SFT_INGESTION_FILENAME,
            HF_TRL_SPLIT_MANIFEST_FILENAME,
            HF_TRL_ENVIRONMENT_LOCK_FILENAME,
            HF_TRL_REFERENCE_SCRIPT_FILENAME,
        ] {
            assert!(
                !envelope.join(excluded).exists(),
                "import envelope leaked {excluded}"
            );
        }
        assert!(incomplete_entries(directory.path()).is_empty());
        let error = write_hf_trl_import_envelope(&completed, &envelope).unwrap_err();
        assert!(error.to_string().contains("overwrite"), "{error:#}");
    }

    #[test]
    fn completed_and_import_verifiers_reject_drift_and_extra_files() {
        let fixture = fixture();
        let directory = tempfile::tempdir().unwrap();
        let completed = directory.path().join("drift.kiln-hf");
        let export = write_hf_trl_sft_bundle(&completed, fixture.input()).unwrap();
        add_completed_result(&completed, &export);
        let envelope = directory.path().join("drift.kiln-hf-import");
        write_hf_trl_import_envelope(&completed, &envelope).unwrap();

        fs::write(
            envelope.join(HF_TRL_ADAPTER_MODEL_FILENAME),
            b"different adapter",
        )
        .unwrap();
        let error = verify_hf_trl_import_envelope(&envelope).unwrap_err();
        assert!(error.to_string().contains("differs"), "{error:#}");

        let envelope = directory.path().join("extra.kiln-hf-import");
        write_hf_trl_import_envelope(&completed, &envelope).unwrap();
        fs::write(envelope.join("dataset.jsonl"), b"private row\n").unwrap();
        let error = verify_hf_trl_import_envelope(&envelope).unwrap_err();
        assert!(error.to_string().contains("file set differs"), "{error:#}");

        fs::remove_file(completed.join(HF_TRL_RESULT_MANIFEST_FILENAME)).unwrap();
        assert!(verify_hf_trl_completed_bundle(&completed).is_err());
    }
}
