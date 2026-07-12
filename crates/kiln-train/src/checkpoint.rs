//! Durable, exact-resume training checkpoint contract.
//!
//! A resumable checkpoint is an immutable directory containing a strict
//! versioned manifest and every state artifact needed to continue a native
//! training run. It is deliberately not a PEFT adapter export: PEFT snapshots
//! contain serving weights, while this format also owns optimizer, scheduler,
//! cursor, RNG, and reference state.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const TRAINING_CHECKPOINT_SCHEMA_VERSION: u32 = 1;
pub const TRAINING_CHECKPOINT_MANIFEST_TYPE: &str = "kiln.training-checkpoint.v1";
pub const TRAINING_CHECKPOINT_MANIFEST_FILENAME: &str = "checkpoint_manifest.json";
pub const TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL: &str = ".incomplete";
pub const TRAINING_CHECKPOINT_DIRECTORY_SUFFIX: &str = ".kiln-checkpoint";
pub(crate) const BASE_WEIGHT_SHARD_MANIFEST_AUXILIARY_KEY: &str = "base_weight_shard_manifest";

const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARTIFACTS: usize = 64;
const MAX_IDENTIFIER_BYTES: usize = 255;
const MAX_RELATIVE_PATH_BYTES: usize = 1024;

/// Parse and verify the base-weight provenance embedded in an exact checkpoint.
/// Legacy checkpoints that recorded only one aggregate are deliberately not
/// accepted for exact resume because they cannot identify the constituent
/// shard artifacts.
pub(crate) fn validated_checkpoint_base_weight_manifest(
    auxiliary_state: &Value,
) -> Result<kiln_core::model_provenance::BaseWeightShardManifest> {
    let value = auxiliary_state
        .get(BASE_WEIGHT_SHARD_MANIFEST_AUXILIARY_KEY)
        .with_context(|| {
            format!(
                "checkpoint has no {BASE_WEIGHT_SHARD_MANIFEST_AUXILIARY_KEY}; exact resume requires per-shard base-weight provenance"
            )
        })?;
    let manifest: kiln_core::model_provenance::BaseWeightShardManifest =
        serde_json::from_value(value.clone())
            .context("parse checkpoint base-weight shard manifest")?;
    manifest
        .validate()
        .context("validate checkpoint base-weight shard manifest")?;
    let aggregate = auxiliary_state
        .get("base_model_weights_sha256")
        .and_then(Value::as_str)
        .context("checkpoint has no base_model_weights_sha256 aggregate")?;
    ensure!(
        aggregate == manifest.aggregate_sha256,
        "checkpoint base-model aggregate {aggregate} differs from its shard manifest {}",
        manifest.aggregate_sha256
    );
    Ok(manifest)
}

pub(crate) fn validate_checkpoint_base_weight_resume_binding(
    checkpoint_auxiliary_state: &Value,
    current_auxiliary_state: &Value,
) -> Result<()> {
    let checkpoint = validated_checkpoint_base_weight_manifest(checkpoint_auxiliary_state)?;
    let current = validated_checkpoint_base_weight_manifest(current_auxiliary_state)
        .context("current run has no valid base-weight shard provenance")?;
    ensure!(
        checkpoint
            .content_equivalent(&current)
            .context("compare checkpoint and current base-weight shard manifests")?,
        "resume checkpoint base-weight shard content differs from this run"
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingKind {
    Sft,
    Grpo,
    Opd,
    CapabilityDistillation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingCheckpointKind {
    Resumable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum CheckpointFileRole {
    AdapterParameters,
    OptimizerState,
    ReferenceState,
    EmaState,
    RewardNormalizationState,
    RngState,
    DataOrder,
    LossHistory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointArtifact {
    pub relative_path: String,
    pub role: CheckpointFileRole,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointFile {
    pub role: CheckpointFileRole,
    pub size_bytes: u64,
    /// Lowercase, unprefixed SHA-256 of the exact artifact bytes.
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointProgress {
    /// Number of fully committed optimizer steps.
    pub global_step: u64,
    pub total_steps: u64,
    /// Zero-based epoch containing the next item to process.
    pub epoch_index: u64,
    /// Zero-based cursor into `data_order` for the next item to process.
    pub cursor_in_epoch: u64,
    /// Exact item order for `epoch_index`. Empty only after all steps finish.
    pub data_order: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointData {
    pub source_kind: String,
    pub content_sha256: String,
    pub item_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointRngState {
    /// Stable algorithm or derivation name, such as `chacha12` or
    /// `kiln.epoch-order.v1`.
    pub algorithm: String,
    pub seed: u64,
    /// Number of draws consumed, or the deterministic counter used to derive
    /// the next draw. Zero is valid for counter-derived streams.
    pub position: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointOptimizer {
    pub kind: String,
    pub step: u64,
    pub hyperparameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointScheduler {
    pub kind: String,
    pub step: u64,
    pub state: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointPrecision {
    pub parameter_dtype: String,
    pub optimizer_state_dtype: String,
    pub activation_dtype: String,
    pub gradient_dtype: String,
    pub stochastic_rounding: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointStateFiles {
    pub adapter_parameters: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ema_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_normalization_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_history: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointManifest {
    pub schema_version: u32,
    pub manifest_type: String,
    pub checkpoint_kind: TrainingCheckpointKind,
    pub checkpoint_id: String,
    pub training_kind: TrainingKind,
    pub adapter_name: String,
    pub created_at: String,
    /// Exact resolved configuration, after defaults and server policy.
    pub effective_config: Value,
    pub precision_policy: TrainingCheckpointPrecision,
    pub progress: TrainingCheckpointProgress,
    pub data: TrainingCheckpointData,
    /// Every independent RNG/counter stream, keyed by a bounded stable name.
    pub rng_states: BTreeMap<String, TrainingCheckpointRngState>,
    pub optimizer: TrainingCheckpointOptimizer,
    pub scheduler: TrainingCheckpointScheduler,
    pub state_files: TrainingCheckpointStateFiles,
    /// Backend-independent reference/EMA/sampler details that do not need a
    /// tensor artifact. Changes to this structure require a schema bump.
    pub auxiliary_state: Value,
    /// Populated by [`write_training_checkpoint_atomic`]. Callers must pass an
    /// empty map so checksums cannot be supplied or trusted externally.
    pub files: BTreeMap<String, CheckpointFile>,
}

impl TrainingCheckpointManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        checkpoint_id: impl Into<String>,
        training_kind: TrainingKind,
        adapter_name: impl Into<String>,
        effective_config: Value,
        precision_policy: TrainingCheckpointPrecision,
        progress: TrainingCheckpointProgress,
        data: TrainingCheckpointData,
        rng_states: BTreeMap<String, TrainingCheckpointRngState>,
        optimizer: TrainingCheckpointOptimizer,
        scheduler: TrainingCheckpointScheduler,
        state_files: TrainingCheckpointStateFiles,
        auxiliary_state: Value,
    ) -> Self {
        Self {
            schema_version: TRAINING_CHECKPOINT_SCHEMA_VERSION,
            manifest_type: TRAINING_CHECKPOINT_MANIFEST_TYPE.to_string(),
            checkpoint_kind: TrainingCheckpointKind::Resumable,
            checkpoint_id: checkpoint_id.into(),
            training_kind,
            adapter_name: adapter_name.into(),
            created_at: Utc::now().to_rfc3339(),
            effective_config,
            precision_policy,
            progress,
            data,
            rng_states,
            optimizer,
            scheduler,
            state_files,
            auxiliary_state,
            files: BTreeMap::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == TRAINING_CHECKPOINT_SCHEMA_VERSION,
            "unsupported training checkpoint schema_version {}; expected {}",
            self.schema_version,
            TRAINING_CHECKPOINT_SCHEMA_VERSION
        );
        ensure!(
            self.manifest_type == TRAINING_CHECKPOINT_MANIFEST_TYPE,
            "invalid training checkpoint manifest_type {:?}",
            self.manifest_type
        );
        validate_identifier("checkpoint_id", &self.checkpoint_id)?;
        validate_identifier("adapter_name", &self.adapter_name)?;
        chrono::DateTime::parse_from_rfc3339(&self.created_at)
            .context("training checkpoint created_at must be RFC 3339")?;
        ensure!(
            self.effective_config.is_object(),
            "training checkpoint effective_config must be a JSON object"
        );
        ensure!(
            self.auxiliary_state.is_object(),
            "training checkpoint auxiliary_state must be a JSON object"
        );
        ensure!(
            self.progress.global_step <= self.progress.total_steps,
            "training checkpoint global_step exceeds total_steps"
        );
        ensure!(
            self.progress.cursor_in_epoch <= self.progress.data_order.len() as u64,
            "training checkpoint cursor_in_epoch exceeds data_order length"
        );
        if self.progress.global_step < self.progress.total_steps {
            ensure!(
                !self.progress.data_order.is_empty(),
                "in-progress training checkpoint has an empty data_order"
            );
        }
        ensure!(
            self.data.item_count > 0,
            "training checkpoint data item_count must be positive"
        );
        if !self.progress.data_order.is_empty() {
            ensure!(
                self.progress.data_order.len() as u64 == self.data.item_count,
                "training checkpoint data_order length {} does not match data item_count {}",
                self.progress.data_order.len(),
                self.data.item_count
            );
            let mut unique = BTreeSet::new();
            for &item in &self.progress.data_order {
                ensure!(
                    item < self.data.item_count,
                    "training checkpoint data_order item {item} exceeds item_count {}",
                    self.data.item_count
                );
                ensure!(
                    unique.insert(item),
                    "training checkpoint data_order contains duplicate item {item}"
                );
            }
        }
        validate_sha256("data.content_sha256", &self.data.content_sha256)?;
        validate_nonempty("data.source_kind", &self.data.source_kind)?;
        validate_nonempty("optimizer.kind", &self.optimizer.kind)?;
        validate_nonempty("scheduler.kind", &self.scheduler.kind)?;
        ensure!(
            self.optimizer.hyperparameters.is_object(),
            "training checkpoint optimizer.hyperparameters must be a JSON object"
        );
        ensure!(
            self.scheduler.state.is_object(),
            "training checkpoint scheduler.state must be a JSON object"
        );
        ensure!(
            self.optimizer.step == self.progress.global_step,
            "optimizer step {} does not match global_step {}",
            self.optimizer.step,
            self.progress.global_step
        );
        ensure!(
            self.scheduler.step == self.progress.global_step,
            "scheduler step {} does not match global_step {}",
            self.scheduler.step,
            self.progress.global_step
        );
        for (name, state) in &self.rng_states {
            validate_identifier("rng state name", name)?;
            validate_nonempty("rng algorithm", &state.algorithm)?;
            if let Some(path) = &state.state_file {
                validate_relative_artifact_path(path)?;
            }
        }
        for (field, value) in [
            ("parameter_dtype", &self.precision_policy.parameter_dtype),
            (
                "optimizer_state_dtype",
                &self.precision_policy.optimizer_state_dtype,
            ),
            ("activation_dtype", &self.precision_policy.activation_dtype),
            ("gradient_dtype", &self.precision_policy.gradient_dtype),
        ] {
            validate_nonempty(field, value)?;
        }
        ensure!(
            self.precision_policy.stochastic_rounding.is_object(),
            "training checkpoint stochastic_rounding must be a JSON object"
        );
        ensure!(
            !self.files.is_empty(),
            "training checkpoint contains no files"
        );
        ensure!(
            self.files.len() <= MAX_ARTIFACTS,
            "training checkpoint has too many files ({} > {MAX_ARTIFACTS})",
            self.files.len()
        );
        for (path, file) in &self.files {
            validate_relative_artifact_path(path)?;
            validate_sha256(&format!("files[{path:?}].sha256"), &file.sha256)?;
        }
        self.validate_state_file_roles()
    }

    fn validate_state_file_roles(&self) -> Result<()> {
        self.require_role(
            &self.state_files.adapter_parameters,
            CheckpointFileRole::AdapterParameters,
            "state_files.adapter_parameters",
        )?;
        for (path, role, field) in [
            (
                self.state_files.optimizer_state.as_deref(),
                CheckpointFileRole::OptimizerState,
                "state_files.optimizer_state",
            ),
            (
                self.state_files.reference_state.as_deref(),
                CheckpointFileRole::ReferenceState,
                "state_files.reference_state",
            ),
            (
                self.state_files.ema_state.as_deref(),
                CheckpointFileRole::EmaState,
                "state_files.ema_state",
            ),
            (
                self.state_files.reward_normalization_state.as_deref(),
                CheckpointFileRole::RewardNormalizationState,
                "state_files.reward_normalization_state",
            ),
            (
                self.state_files.loss_history.as_deref(),
                CheckpointFileRole::LossHistory,
                "state_files.loss_history",
            ),
        ] {
            if let Some(path) = path {
                self.require_role(path, role, field)?;
            }
        }
        if let Some(path) = self.optimizer.state_file.as_deref() {
            self.require_role(
                path,
                CheckpointFileRole::OptimizerState,
                "optimizer.state_file",
            )?;
            ensure!(
                self.state_files.optimizer_state.as_deref() == Some(path),
                "optimizer.state_file and state_files.optimizer_state disagree"
            );
        } else {
            ensure!(
                self.state_files.optimizer_state.is_none(),
                "state_files.optimizer_state is present but optimizer.state_file is absent"
            );
        }
        for state in self.rng_states.values() {
            if let Some(path) = &state.state_file {
                self.require_role(path, CheckpointFileRole::RngState, "rng state_file")?;
            }
        }
        Ok(())
    }

    fn require_role(&self, path: &str, expected: CheckpointFileRole, field: &str) -> Result<()> {
        validate_relative_artifact_path(path)?;
        let file = self
            .files
            .get(path)
            .with_context(|| format!("{field} references untracked artifact {path:?}"))?;
        ensure!(
            file.role == expected,
            "{field} artifact {path:?} has role {:?}, expected {:?}",
            file.role,
            expected
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ValidatedTrainingCheckpoint {
    pub root: PathBuf,
    pub manifest: TrainingCheckpointManifest,
}

impl ValidatedTrainingCheckpoint {
    pub fn artifact_path(&self, relative_path: &str) -> Result<PathBuf> {
        validate_relative_artifact_path(relative_path)?;
        ensure!(
            self.manifest.files.contains_key(relative_path),
            "artifact {relative_path:?} is not tracked by the checkpoint manifest"
        );
        Ok(self.root.join(relative_path))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckpointPublicationStage {
    StagingCreated,
    SentinelSynced,
    ArtifactsWritten,
    ManifestAndArtifactsSynced,
    ReadyToPublish,
    Published,
    ParentDirectorySynced,
}

impl CheckpointPublicationStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::StagingCreated => "staging_created",
            Self::SentinelSynced => "sentinel_synced",
            Self::ArtifactsWritten => "artifacts_written",
            Self::ManifestAndArtifactsSynced => "manifest_and_artifacts_synced",
            Self::ReadyToPublish => "ready_to_publish",
            Self::Published => "published",
            Self::ParentDirectorySynced => "parent_directory_synced",
        }
    }

    #[cfg(test)]
    fn is_published(self) -> bool {
        matches!(self, Self::Published | Self::ParentDirectorySynced)
    }
}

fn observe_checkpoint_publication_stage<O>(
    observe: &mut O,
    stage: CheckpointPublicationStage,
) -> Result<()>
where
    O: FnMut(CheckpointPublicationStage) -> Result<()>,
{
    observe(stage).with_context(|| {
        format!(
            "checkpoint publication interrupted at durable stage {}",
            stage.as_str()
        )
    })
}

/// Stage and publish one immutable resumable checkpoint directory.
///
/// `write_artifacts` writes every declared artifact beneath the supplied
/// staging directory. The final path remains absent until all files are
/// checksummed, the strict manifest is synced, and the incomplete sentinel is
/// removed. Existing checkpoints are never overwritten.
pub fn write_training_checkpoint_atomic<F>(
    target: &Path,
    manifest: TrainingCheckpointManifest,
    artifacts: &[CheckpointArtifact],
    write_artifacts: F,
) -> Result<PathBuf>
where
    F: FnOnce(&Path) -> Result<()>,
{
    write_training_checkpoint_atomic_observed(target, manifest, artifacts, write_artifacts, |_| {
        Ok(())
    })
}

fn write_training_checkpoint_atomic_observed<F, O>(
    target: &Path,
    mut manifest: TrainingCheckpointManifest,
    artifacts: &[CheckpointArtifact],
    write_artifacts: F,
    mut observe: O,
) -> Result<PathBuf>
where
    F: FnOnce(&Path) -> Result<()>,
    O: FnMut(CheckpointPublicationStage) -> Result<()>,
{
    ensure!(
        manifest.files.is_empty(),
        "checkpoint manifest files must be empty before writing"
    );
    validate_checkpoint_target(target)?;
    ensure!(
        !target.exists(),
        "refusing to overwrite immutable training checkpoint {}",
        target.display()
    );
    ensure!(
        !artifacts.is_empty(),
        "training checkpoint must declare at least one artifact"
    );
    ensure!(
        artifacts.len() <= MAX_ARTIFACTS,
        "training checkpoint has too many artifacts ({} > {MAX_ARTIFACTS})",
        artifacts.len()
    );

    let mut planned = BTreeMap::new();
    for artifact in artifacts {
        validate_relative_artifact_path(&artifact.relative_path)?;
        ensure!(
            planned
                .insert(artifact.relative_path.clone(), artifact.role)
                .is_none(),
            "duplicate checkpoint artifact {:?}",
            artifact.relative_path
        );
    }

    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create checkpoint parent {}", parent.display()))?;
    let basename = target
        .file_name()
        .and_then(|name| name.to_str())
        .context("training checkpoint target must have a UTF-8 basename")?;
    let staging = parent.join(format!(".{basename}.incomplete-{}", Uuid::new_v4()));
    fs::create_dir(&staging)
        .with_context(|| format!("create checkpoint staging directory {}", staging.display()))?;
    let mut guard = StagingGuard::new(staging.clone());
    observe_checkpoint_publication_stage(&mut observe, CheckpointPublicationStage::StagingCreated)?;

    let sentinel_path = staging.join(TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL);
    write_new_synced_file(
        &sentinel_path,
        format!("{}\n", manifest.checkpoint_id).as_bytes(),
    )?;
    sync_directory(&staging)?;
    observe_checkpoint_publication_stage(&mut observe, CheckpointPublicationStage::SentinelSynced)?;

    write_artifacts(&staging).context("write training checkpoint artifacts")?;
    observe_checkpoint_publication_stage(
        &mut observe,
        CheckpointPublicationStage::ArtifactsWritten,
    )?;

    let actual = collect_files(&staging)?;
    let expected: BTreeSet<_> = planned.keys().cloned().collect();
    let allowed: BTreeSet<_> = expected
        .iter()
        .cloned()
        .chain([TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL.to_string()])
        .collect();
    ensure!(
        actual == allowed,
        "checkpoint staging files do not match declaration: expected {:?}, found {:?}",
        allowed,
        actual
    );

    for (relative, role) in planned {
        let path = staging.join(&relative);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("stat checkpoint artifact {}", path.display()))?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "checkpoint artifact is not a regular file: {}",
            path.display()
        );
        ensure_not_hardlinked(&path, &metadata)?;
        let sha256 = sha256_file(&path)?;
        manifest.files.insert(
            relative,
            CheckpointFile {
                role,
                size_bytes: metadata.len(),
                sha256,
            },
        );
    }
    manifest.validate()?;

    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).context("serialize training checkpoint manifest")?;
    ensure!(
        manifest_bytes.len() as u64 <= MAX_MANIFEST_BYTES,
        "training checkpoint manifest exceeds {MAX_MANIFEST_BYTES} bytes"
    );
    write_new_synced_file(
        &staging.join(TRAINING_CHECKPOINT_MANIFEST_FILENAME),
        &manifest_bytes,
    )?;
    for relative in manifest.files.keys() {
        File::open(staging.join(relative))
            .with_context(|| format!("open checkpoint artifact {relative:?} for sync"))?
            .sync_all()
            .with_context(|| format!("sync checkpoint artifact {relative:?}"))?;
    }
    observe_checkpoint_publication_stage(
        &mut observe,
        CheckpointPublicationStage::ManifestAndArtifactsSynced,
    )?;
    fs::remove_file(&sentinel_path)
        .with_context(|| format!("remove checkpoint sentinel {}", sentinel_path.display()))?;
    sync_directory_tree(&staging)?;
    observe_checkpoint_publication_stage(&mut observe, CheckpointPublicationStage::ReadyToPublish)?;

    fs::rename(&staging, target).with_context(|| {
        format!(
            "publish training checkpoint {} -> {}",
            staging.display(),
            target.display()
        )
    })?;
    observe_checkpoint_publication_stage(&mut observe, CheckpointPublicationStage::Published)?;
    sync_directory(parent)?;
    observe_checkpoint_publication_stage(
        &mut observe,
        CheckpointPublicationStage::ParentDirectorySynced,
    )?;
    guard.disarm();
    Ok(target.to_path_buf())
}

/// Read and validate checkpoint metadata without hashing state artifacts.
///
/// This is intended for low-cost discovery surfaces that may poll while a job
/// is running. It still rejects non-directories, symlinks, incomplete writes,
/// oversized or hard-linked manifests, unknown fields, and invalid manifest
/// semantics. Call [`load_training_checkpoint`] before restoring any state;
/// that path additionally validates the complete file set and every checksum.
pub fn read_training_checkpoint_manifest(path: &Path) -> Result<TrainingCheckpointManifest> {
    validate_checkpoint_target(path)?;
    let root_meta = fs::symlink_metadata(path)
        .with_context(|| format!("stat training checkpoint {}", path.display()))?;
    ensure!(
        root_meta.file_type().is_dir() && !root_meta.file_type().is_symlink(),
        "training checkpoint is not a regular directory: {}",
        path.display()
    );
    ensure!(
        !path.join(TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL).exists(),
        "training checkpoint is incomplete: {}",
        path.display()
    );
    let manifest_path = path.join(TRAINING_CHECKPOINT_MANIFEST_FILENAME);
    let metadata = fs::symlink_metadata(&manifest_path).with_context(|| {
        format!(
            "missing training checkpoint manifest {}; PEFT adapter snapshots are not resumable checkpoints",
            manifest_path.display()
        )
    })?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "training checkpoint manifest is not a regular file"
    );
    ensure_not_hardlinked(&manifest_path, &metadata)?;
    ensure!(
        metadata.len() <= MAX_MANIFEST_BYTES,
        "training checkpoint manifest exceeds {MAX_MANIFEST_BYTES} bytes"
    );
    let bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "read training checkpoint manifest {}",
            manifest_path.display()
        )
    })?;
    let manifest: TrainingCheckpointManifest =
        serde_json::from_slice(&bytes).context("parse strict training checkpoint manifest")?;
    manifest.validate()?;
    Ok(manifest)
}

/// Load a checkpoint only after strict manifest, file-set, and checksum
/// validation. Incomplete staging directories and PEFT adapter directories are
/// rejected rather than partially interpreted as resumable state.
pub fn load_training_checkpoint(path: &Path) -> Result<ValidatedTrainingCheckpoint> {
    let manifest = read_training_checkpoint_manifest(path)?;

    let actual = collect_files(path)?;
    let expected: BTreeSet<_> = manifest
        .files
        .keys()
        .cloned()
        .chain([TRAINING_CHECKPOINT_MANIFEST_FILENAME.to_string()])
        .collect();
    ensure!(
        actual == expected,
        "checkpoint files do not match manifest: expected {:?}, found {:?}",
        expected,
        actual
    );
    for (relative, expected_file) in &manifest.files {
        let artifact = path.join(relative);
        let metadata = fs::symlink_metadata(&artifact)
            .with_context(|| format!("stat checkpoint artifact {}", artifact.display()))?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "checkpoint artifact is not a regular file: {}",
            artifact.display()
        );
        ensure_not_hardlinked(&artifact, &metadata)?;
        ensure!(
            metadata.len() == expected_file.size_bytes,
            "checkpoint artifact size mismatch for {relative:?}: expected {}, found {}",
            expected_file.size_bytes,
            metadata.len()
        );
        let actual_hash = sha256_file(&artifact)?;
        ensure!(
            actual_hash == expected_file.sha256,
            "checkpoint artifact checksum mismatch for {relative:?}: expected {}, found {}",
            expected_file.sha256,
            actual_hash
        );
    }
    Ok(ValidatedTrainingCheckpoint {
        root: path.to_path_buf(),
        manifest,
    })
}

fn validate_checkpoint_target(path: &Path) -> Result<()> {
    ensure!(
        !path.as_os_str().is_empty(),
        "training checkpoint target is empty"
    );
    let basename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("training checkpoint target must have a UTF-8 basename")?;
    ensure!(
        basename.ends_with(TRAINING_CHECKPOINT_DIRECTORY_SUFFIX),
        "path is not a resumable training checkpoint: directory must end with {:?}",
        TRAINING_CHECKPOINT_DIRECTORY_SUFFIX
    );
    Ok(())
}

fn validate_identifier(field: &str, value: &str) -> Result<()> {
    validate_nonempty(field, value)?;
    ensure!(
        value.len() <= MAX_IDENTIFIER_BYTES,
        "{field} exceeds {MAX_IDENTIFIER_BYTES} bytes"
    );
    ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')),
        "{field} contains unsupported characters: {value:?}"
    );
    Ok(())
}

fn validate_nonempty(field: &str, value: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{field} must not be empty");
    Ok(())
}

fn validate_sha256(field: &str, value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "{field} must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn validate_relative_artifact_path(value: &str) -> Result<()> {
    ensure!(!value.is_empty(), "checkpoint artifact path is empty");
    ensure!(
        value.len() <= MAX_RELATIVE_PATH_BYTES,
        "checkpoint artifact path exceeds {MAX_RELATIVE_PATH_BYTES} bytes"
    );
    ensure!(
        !value.contains('\\'),
        "checkpoint artifact path contains a backslash: {value:?}"
    );
    ensure!(
        value != TRAINING_CHECKPOINT_MANIFEST_FILENAME
            && value != TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL,
        "reserved checkpoint artifact path {value:?}"
    );
    let path = Path::new(value);
    ensure!(
        !path.is_absolute(),
        "checkpoint artifact path is absolute: {value:?}"
    );
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "checkpoint artifact path is not normalized: {value:?}"
    );
    Ok(())
}

fn collect_files(root: &Path) -> Result<BTreeSet<String>> {
    let mut files = BTreeSet::new();
    collect_files_recursive(root, root, &mut files)?;
    Ok(files)
}

fn collect_files_recursive(
    root: &Path,
    current: &Path,
    files: &mut BTreeSet<String>,
) -> Result<()> {
    for entry in fs::read_dir(current)
        .with_context(|| format!("read checkpoint directory {}", current.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("stat checkpoint entry {}", path.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "checkpoint entries may not be symlinks: {}",
            path.display()
        );
        if metadata.file_type().is_dir() {
            collect_files_recursive(root, &path, files)?;
        } else if metadata.file_type().is_file() {
            let relative = path
                .strip_prefix(root)
                .context("checkpoint entry escaped root")?
                .to_str()
                .context("checkpoint artifact path must be UTF-8")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            validate_relative_artifact_path(&relative).or_else(|error| {
                if relative == TRAINING_CHECKPOINT_MANIFEST_FILENAME
                    || relative == TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL
                {
                    Ok(())
                } else {
                    Err(error)
                }
            })?;
            files.insert(relative);
        } else {
            bail!("unsupported checkpoint entry type: {}", path.display());
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hash checkpoint artifact {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let mut encoded = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(encoded)
}

fn write_new_synced_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create checkpoint artifact parent {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("create checkpoint file {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write checkpoint file {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync checkpoint file {}", path.display()))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("open directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("sync directory {}", path.display()))
}

fn sync_directory_tree(root: &Path) -> Result<()> {
    let mut directories = vec![root.to_path_buf()];
    let mut cursor = 0;
    while cursor < directories.len() {
        let current = directories[cursor].clone();
        cursor += 1;
        for entry in fs::read_dir(&current)
            .with_context(|| format!("read checkpoint directory {} for sync", current.display()))?
        {
            let entry = entry?;
            let metadata = fs::symlink_metadata(entry.path())?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "checkpoint entries may not be symlinks: {}",
                entry.path().display()
            );
            if metadata.file_type().is_dir() {
                directories.push(entry.path());
            }
        }
    }
    for directory in directories.iter().rev() {
        sync_directory(directory)?;
    }
    Ok(())
}

fn ensure_not_hardlinked(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            metadata.nlink() == 1,
            "checkpoint files may not be hard links: {}",
            path.display()
        );
    }
    #[cfg(not(unix))]
    let _ = (path, metadata);
    Ok(())
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
    use super::*;

    fn base_weight_manifest(
        filename: &str,
        byte: u8,
    ) -> kiln_core::model_provenance::BaseWeightShardManifest {
        kiln_core::model_provenance::BaseWeightShardManifest::new(vec![
            kiln_core::model_provenance::BaseWeightShardIdentity::from_digest(
                filename, 11, [byte; 32],
            )
            .unwrap(),
        ])
        .unwrap()
    }

    fn base_weight_auxiliary_state(
        manifest: &kiln_core::model_provenance::BaseWeightShardManifest,
    ) -> Value {
        serde_json::json!({
            "base_model_weights_sha256": manifest.aggregate_sha256,
            BASE_WEIGHT_SHARD_MANIFEST_AUXILIARY_KEY: manifest,
        })
    }

    fn hash(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn manifest() -> TrainingCheckpointManifest {
        TrainingCheckpointManifest::new(
            "sft-step-2",
            TrainingKind::Sft,
            "demo-adapter",
            serde_json::json!({"epochs": 2, "seed": 7}),
            TrainingCheckpointPrecision {
                parameter_dtype: "f32".into(),
                optimizer_state_dtype: "f32".into(),
                activation_dtype: "f32".into(),
                gradient_dtype: "f32".into(),
                stochastic_rounding: serde_json::json!({"enabled": false}),
            },
            TrainingCheckpointProgress {
                global_step: 2,
                total_steps: 4,
                epoch_index: 1,
                cursor_in_epoch: 0,
                data_order: vec![1, 0],
            },
            TrainingCheckpointData {
                source_kind: "inline-sft".into(),
                content_sha256: hash(0x11),
                item_count: 2,
            },
            BTreeMap::from([(
                "data-order".into(),
                TrainingCheckpointRngState {
                    algorithm: "kiln.epoch-order.v1".into(),
                    seed: 7,
                    position: 1,
                    state_file: None,
                },
            )]),
            TrainingCheckpointOptimizer {
                kind: "adam-w".into(),
                step: 2,
                hyperparameters: serde_json::json!({"lr": 0.001}),
                state_file: Some("optimizer.safetensors".into()),
            },
            TrainingCheckpointScheduler {
                kind: "constant".into(),
                step: 2,
                state: serde_json::json!({"lr": 0.001}),
            },
            TrainingCheckpointStateFiles {
                adapter_parameters: "adapter.safetensors".into(),
                optimizer_state: Some("optimizer.safetensors".into()),
                reference_state: None,
                ema_state: None,
                reward_normalization_state: None,
                loss_history: Some("losses.json".into()),
            },
            serde_json::json!({"reference_policy": "base"}),
        )
    }

    fn artifacts() -> Vec<CheckpointArtifact> {
        vec![
            CheckpointArtifact {
                relative_path: "adapter.safetensors".into(),
                role: CheckpointFileRole::AdapterParameters,
            },
            CheckpointArtifact {
                relative_path: "optimizer.safetensors".into(),
                role: CheckpointFileRole::OptimizerState,
            },
            CheckpointArtifact {
                relative_path: "losses.json".into(),
                role: CheckpointFileRole::LossHistory,
            },
        ]
    }

    #[test]
    fn exact_resume_requires_valid_per_shard_base_weight_provenance() {
        let manifest = base_weight_manifest("model.safetensors", 0x11);
        let valid = base_weight_auxiliary_state(&manifest);
        assert_eq!(
            validated_checkpoint_base_weight_manifest(&valid).unwrap(),
            manifest
        );

        let aggregate_only = serde_json::json!({
            "base_model_weights_sha256": manifest.aggregate_sha256,
        });
        let error = validated_checkpoint_base_weight_manifest(&aggregate_only)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exact resume requires per-shard"));

        let mut inconsistent = valid;
        inconsistent["base_model_weights_sha256"] =
            Value::String(format!("sha256:{}", "0".repeat(64)));
        let error = validated_checkpoint_base_weight_manifest(&inconsistent)
            .unwrap_err()
            .to_string();
        assert!(error.contains("differs from its shard manifest"));
    }

    #[test]
    fn exact_resume_compares_shard_content_not_audit_filenames() {
        let original = base_weight_manifest("model.safetensors", 0x11);
        let renamed = base_weight_manifest("relocated.safetensors", 0x11);
        validate_checkpoint_base_weight_resume_binding(
            &base_weight_auxiliary_state(&original),
            &base_weight_auxiliary_state(&renamed),
        )
        .unwrap();

        let changed = base_weight_manifest("model.safetensors", 0x12);
        let error = validate_checkpoint_base_weight_resume_binding(
            &base_weight_auxiliary_state(&original),
            &base_weight_auxiliary_state(&changed),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("shard content differs"));
    }

    fn write_artifacts(root: &Path) -> Result<()> {
        fs::write(root.join("adapter.safetensors"), b"adapter")?;
        fs::write(root.join("optimizer.safetensors"), b"optimizer")?;
        fs::write(root.join("losses.json"), b"[1.0,0.5]")?;
        Ok(())
    }

    fn publication_stages() -> [CheckpointPublicationStage; 7] {
        [
            CheckpointPublicationStage::StagingCreated,
            CheckpointPublicationStage::SentinelSynced,
            CheckpointPublicationStage::ArtifactsWritten,
            CheckpointPublicationStage::ManifestAndArtifactsSynced,
            CheckpointPublicationStage::ReadyToPublish,
            CheckpointPublicationStage::Published,
            CheckpointPublicationStage::ParentDirectorySynced,
        ]
    }

    fn publication_stage_named(name: &str) -> Option<CheckpointPublicationStage> {
        publication_stages()
            .into_iter()
            .find(|stage| stage.as_str() == name)
    }

    #[test]
    fn injected_fault_at_every_publication_stage_is_absent_or_valid() -> Result<()> {
        for fault_stage in publication_stages() {
            let temp = tempfile::tempdir()?;
            let target = temp.path().join("fault.kiln-checkpoint");
            let error = format!(
                "{:#}",
                write_training_checkpoint_atomic_observed(
                    &target,
                    manifest(),
                    &artifacts(),
                    write_artifacts,
                    |stage| {
                        if stage == fault_stage {
                            bail!("injected fault at {}", stage.as_str());
                        }
                        Ok(())
                    },
                )
                .unwrap_err()
            );
            assert!(
                error.contains(&format!("injected fault at {}", fault_stage.as_str())),
                "wrong error at {fault_stage:?}: {error}"
            );

            if fault_stage.is_published() {
                load_training_checkpoint(&target).with_context(|| {
                    format!("published target must remain valid after {fault_stage:?}")
                })?;
            } else {
                assert!(
                    !target.exists(),
                    "pre-publish fault {fault_stage:?} exposed the final basename"
                );
                assert_eq!(
                    fs::read_dir(temp.path())?.count(),
                    0,
                    "in-process cleanup leaked staging after {fault_stage:?}"
                );
                write_training_checkpoint_atomic(
                    &target,
                    manifest(),
                    &artifacts(),
                    write_artifacts,
                )?;
                load_training_checkpoint(&target).with_context(|| {
                    format!("retry after {fault_stage:?} did not produce a valid checkpoint")
                })?;
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn process_kill_at_every_publication_stage_is_absent_or_valid() -> Result<()> {
        use std::os::unix::process::ExitStatusExt as _;
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};

        const CHILD_ROOT_ENV: &str = "KILN_CHECKPOINT_KILL_CHILD_ROOT";
        const CHILD_STAGE_ENV: &str = "KILN_CHECKPOINT_KILL_CHILD_STAGE";
        const TARGET_BASENAME: &str = "crash.kiln-checkpoint";
        const READY_MARKER: &str = "publication-stage-ready";

        if let Some(root) = std::env::var_os(CHILD_ROOT_ENV) {
            let root = PathBuf::from(root);
            let stage_name = std::env::var(CHILD_STAGE_ENV)
                .context("checkpoint kill child omitted publication stage")?;
            let stop_stage = publication_stage_named(&stage_name)
                .with_context(|| format!("unknown checkpoint publication stage {stage_name:?}"))?;
            let marker = root.join(READY_MARKER);
            let result = write_training_checkpoint_atomic_observed(
                &root.join(TARGET_BASENAME),
                manifest(),
                &artifacts(),
                write_artifacts,
                |stage| {
                    if stage == stop_stage {
                        write_new_synced_file(&marker, format!("{}\n", stage.as_str()).as_bytes())?;
                        sync_directory(&root)?;
                        loop {
                            std::thread::sleep(Duration::from_secs(60));
                        }
                    }
                    Ok(())
                },
            );
            bail!(
                "checkpoint kill child passed stop stage {stop_stage:?} without blocking: {result:?}"
            );
        }

        for stop_stage in publication_stages() {
            let temp = tempfile::tempdir()?;
            let case_root = temp.path().join(stop_stage.as_str());
            fs::create_dir(&case_root)?;
            let target = case_root.join(TARGET_BASENAME);
            let marker = case_root.join(READY_MARKER);
            let mut child = Command::new(std::env::current_exe()?)
                .arg("process_kill_at_every_publication_stage_is_absent_or_valid")
                .arg("--test-threads=1")
                .env(CHILD_ROOT_ENV, &case_root)
                .env(CHILD_STAGE_ENV, stop_stage.as_str())
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .with_context(|| format!("spawn checkpoint kill child for {stop_stage:?}"))?;

            let deadline = Instant::now() + Duration::from_secs(10);
            while !marker.is_file() {
                if let Some(status) = child.try_wait()? {
                    bail!("checkpoint kill child exited before {stop_stage:?} marker: {status}");
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    bail!("checkpoint kill child timed out before {stop_stage:?} marker");
                }
                std::thread::sleep(Duration::from_millis(10));
            }

            child.kill()?;
            let status = child.wait()?;
            assert_eq!(
                status.signal(),
                Some(9),
                "child at {stop_stage:?} did not terminate by SIGKILL: {status}"
            );

            if stop_stage.is_published() {
                load_training_checkpoint(&target).with_context(|| {
                    format!("post-rename kill at {stop_stage:?} left an invalid target")
                })?;
                assert!(
                    fs::read_dir(&case_root)?
                        .filter_map(std::result::Result::ok)
                        .all(|entry| !entry.file_name().to_string_lossy().starts_with('.')),
                    "post-rename kill at {stop_stage:?} retained staging"
                );
                continue;
            }

            assert!(
                !target.exists(),
                "pre-rename kill at {stop_stage:?} exposed the final basename"
            );
            let staging_prefix = format!(".{TARGET_BASENAME}.incomplete-");
            let staging = fs::read_dir(&case_root)?
                .filter_map(std::result::Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .is_some_and(|name| name.to_string_lossy().starts_with(&staging_prefix))
                })
                .collect::<Vec<_>>();
            assert_eq!(
                staging.len(),
                1,
                "pre-rename kill at {stop_stage:?} did not retain exactly one orphan staging directory"
            );
            assert!(
                load_training_checkpoint(&staging[0]).is_err(),
                "orphan staging at {stop_stage:?} must never be accepted as resumable"
            );
            let sentinel_exists = staging[0]
                .join(TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL)
                .exists();
            assert_eq!(
                sentinel_exists,
                matches!(
                    stop_stage,
                    CheckpointPublicationStage::SentinelSynced
                        | CheckpointPublicationStage::ArtifactsWritten
                        | CheckpointPublicationStage::ManifestAndArtifactsSynced
                ),
                "unexpected sentinel state after {stop_stage:?}"
            );

            write_training_checkpoint_atomic(&target, manifest(), &artifacts(), write_artifacts)?;
            load_training_checkpoint(&target).with_context(|| {
                format!("retry around orphan staging after {stop_stage:?} was invalid")
            })?;
        }
        Ok(())
    }

    #[test]
    fn atomic_round_trip_is_explicitly_resumable() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let target = temp.path().join("demo-step-2.kiln-checkpoint");
        write_training_checkpoint_atomic(&target, manifest(), &artifacts(), |staging| {
            assert!(
                staging
                    .join(TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL)
                    .is_file()
            );
            assert!(!target.exists());
            write_artifacts(staging)
        })?;
        assert!(
            !target
                .join(TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL)
                .exists()
        );
        let loaded = load_training_checkpoint(&target)?;
        assert_eq!(
            loaded.manifest.checkpoint_kind,
            TrainingCheckpointKind::Resumable
        );
        assert_eq!(loaded.manifest.progress.global_step, 2);
        assert_eq!(loaded.manifest.files.len(), 3);
        assert_eq!(
            loaded.artifact_path("adapter.safetensors")?,
            target.join("adapter.safetensors")
        );
        Ok(())
    }

    #[test]
    fn corruption_fails_checksum_validation() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let target = temp.path().join("corrupt.kiln-checkpoint");
        write_training_checkpoint_atomic(&target, manifest(), &artifacts(), write_artifacts)?;
        fs::write(target.join("optimizer.safetensors"), b"tampered!")?;
        assert_eq!(
            read_training_checkpoint_manifest(&target)?
                .progress
                .global_step,
            2,
            "metadata discovery deliberately avoids reading large state artifacts"
        );
        let error = load_training_checkpoint(&target).unwrap_err().to_string();
        assert!(error.contains("checksum mismatch") || error.contains("size mismatch"));
        Ok(())
    }

    #[test]
    fn incomplete_and_peft_directories_are_not_resumable() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let incomplete = temp.path().join("partial.kiln-checkpoint");
        fs::create_dir(&incomplete)?;
        fs::write(
            incomplete.join(TRAINING_CHECKPOINT_INCOMPLETE_SENTINEL),
            b"partial",
        )?;
        assert!(
            load_training_checkpoint(&incomplete)
                .unwrap_err()
                .to_string()
                .contains("incomplete")
        );

        let peft = temp.path().join("peft-adapter");
        fs::create_dir(&peft)?;
        fs::write(peft.join("adapter_config.json"), b"{}")?;
        fs::write(peft.join("adapter_model.safetensors"), b"weights")?;
        let error = load_training_checkpoint(&peft).unwrap_err().to_string();
        assert!(
            error.contains("not a resumable training checkpoint")
                && error.contains(TRAINING_CHECKPOINT_DIRECTORY_SUFFIX),
            "PEFT rejection must name the canonical checkpoint suffix: {error}"
        );
        Ok(())
    }

    #[test]
    fn failed_write_never_publishes_and_cleans_staging() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let target = temp.path().join("failed.kiln-checkpoint");
        let error = format!(
            "{:#}",
            write_training_checkpoint_atomic(&target, manifest(), &artifacts(), |root| {
                fs::write(root.join("adapter.safetensors"), b"adapter")?;
                bail!("injected write failure")
            })
            .unwrap_err()
        );
        assert!(error.contains("injected write failure"));
        assert!(!target.exists());
        assert_eq!(fs::read_dir(temp.path())?.count(), 0);
        Ok(())
    }

    #[test]
    fn immutable_checkpoint_cannot_be_overwritten() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let target = temp.path().join("immutable.kiln-checkpoint");
        write_training_checkpoint_atomic(&target, manifest(), &artifacts(), write_artifacts)?;
        let error = write_training_checkpoint_atomic(&target, manifest(), &artifacts(), |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("refusing to overwrite"));
        Ok(())
    }

    #[test]
    fn undeclared_files_and_path_traversal_fail_closed() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let target = temp.path().join("extra.kiln-checkpoint");
        let error = write_training_checkpoint_atomic(&target, manifest(), &artifacts(), |root| {
            write_artifacts(root)?;
            fs::write(root.join("surprise.bin"), b"surprise")?;
            Ok(())
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("do not match declaration"));

        let mut bad = artifacts();
        bad[0].relative_path = "../adapter.safetensors".into();
        let error = write_training_checkpoint_atomic(
            &temp.path().join("traversal.kiln-checkpoint"),
            manifest(),
            &bad,
            |_| Ok(()),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not normalized"));
        Ok(())
    }

    #[test]
    fn invalid_data_order_fails_before_publish() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let target = temp.path().join("bad-order.kiln-checkpoint");
        let mut bad_manifest = manifest();
        bad_manifest.progress.data_order = vec![0, 0];
        let error =
            write_training_checkpoint_atomic(&target, bad_manifest, &artifacts(), write_artifacts)
                .unwrap_err()
                .to_string();
        assert!(error.contains("duplicate item"));
        assert!(!target.exists());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlink_artifact_fails_closed() -> Result<()> {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir()?;
        let outside = temp.path().join("outside");
        fs::write(&outside, b"outside")?;
        let target = temp.path().join("symlink.kiln-checkpoint");
        let error = write_training_checkpoint_atomic(&target, manifest(), &artifacts(), |root| {
            symlink(&outside, root.join("adapter.safetensors"))?;
            fs::write(root.join("optimizer.safetensors"), b"optimizer")?;
            fs::write(root.join("losses.json"), b"[]")?;
            Ok(())
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("symlinks"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn hardlink_artifact_fails_closed() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let outside = temp.path().join("outside");
        fs::write(&outside, b"outside")?;
        let target = temp.path().join("hardlink.kiln-checkpoint");
        let error = write_training_checkpoint_atomic(&target, manifest(), &artifacts(), |root| {
            fs::hard_link(&outside, root.join("adapter.safetensors"))?;
            fs::write(root.join("optimizer.safetensors"), b"optimizer")?;
            fs::write(root.join("losses.json"), b"[]")?;
            Ok(())
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("hard links"));
        Ok(())
    }

    #[test]
    fn strict_manifest_rejects_unknown_fields() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let target = temp.path().join("strict.kiln-checkpoint");
        write_training_checkpoint_atomic(&target, manifest(), &artifacts(), write_artifacts)?;
        let manifest_path = target.join(TRAINING_CHECKPOINT_MANIFEST_FILENAME);
        let mut value: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        value["surprise"] = Value::Bool(true);
        fs::write(&manifest_path, serde_json::to_vec_pretty(&value)?)?;
        let error = format!("{:#}", load_training_checkpoint(&target).unwrap_err());
        assert!(error.contains("unknown field"));
        Ok(())
    }
}
