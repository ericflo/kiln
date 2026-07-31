//! Publication boundary for native trainer evidence retained by OpenEnv runs.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kiln_train::TrainingState;

use super::{AppState, OpenEnvArtifact, OpenEnvRunRequest, OpenEnvTrainingStatus};

const MAX_OPENENV_TRAINING_EVIDENCE_BYTES: usize = 4 * 1024 * 1024;

pub(super) fn ensure_openenv_training_evidence(
    state: &AppState,
    run_id: &str,
    request: &OpenEnvRunRequest,
    training: &OpenEnvTrainingStatus,
) -> Result<()> {
    let current = state
        .openenv_runs
        .get(run_id)
        .context("OpenEnv run disappeared before training evidence publication")?;
    if ["train_receipt", "adapter_manifest"]
        .into_iter()
        .all(|kind| {
            current
                .artifacts
                .iter()
                .any(|artifact| artifact.kind == kind)
        })
    {
        return Ok(());
    }
    let training_artifacts = publish_openenv_training_evidence(state, run_id, request, training)?;
    state.openenv_runs.update(run_id, |status| {
        status.training = Some(training.clone());
        status.artifacts.retain(|artifact| {
            artifact.kind != "train_receipt" && artifact.kind != "adapter_manifest"
        });
        status.artifacts.extend(training_artifacts);
    })?;
    Ok(())
}

/// Copy the native trainer's small forensic records into the run-owned
/// content-addressed bundle. The adapter directory remains authoritative for
/// model weights, while the OpenEnv run retains immutable evidence even if a
/// later adapter lifecycle operation moves or removes that directory.
pub(super) fn publish_openenv_training_evidence(
    state: &AppState,
    run_id: &str,
    request: &OpenEnvRunRequest,
    training: &OpenEnvTrainingStatus,
) -> Result<Vec<OpenEnvArtifact>> {
    anyhow::ensure!(
        training.state == TrainingState::Completed,
        "OpenEnv training evidence cannot be published before trainer completion"
    );
    let admitted = training
        .training_data
        .as_ref()
        .context("completed OpenEnv trainer omitted admitted corpus provenance")?;
    let admitted_openenv = admitted
        .openenv
        .as_ref()
        .context("completed OpenEnv trainer omitted semantic OpenEnv corpus provenance")?;
    admitted_openenv
        .validate()
        .map_err(anyhow::Error::msg)
        .context("validate admitted OpenEnv corpus provenance before evidence publication")?;
    let output_adapter = request
        .output_adapter
        .as_deref()
        .context("OpenEnv train request omitted output_adapter")?;
    let adapter_dir = PathBuf::from(
        training
            .adapter_path
            .as_deref()
            .context("completed OpenEnv trainer omitted adapter_path")?,
    );

    let receipt_path = adapter_dir.join(kiln_train::TRAIN_RECEIPT_FILENAME);
    let (receipt_sha256, receipt_bytes) = verified_training_evidence_bytes(&receipt_path)?;
    let exact_receipt: kiln_train::TrainReceipt = serde_json::from_slice(&receipt_bytes)
        .context("parse bounded verified OpenEnv train receipt")?;
    exact_receipt
        .validate()
        .context("validate bounded verified OpenEnv train receipt")?;
    anyhow::ensure!(
        exact_receipt.status == kiln_train::TrainReceiptStatus::Success,
        "completed OpenEnv trainer published a non-success train receipt"
    );
    anyhow::ensure!(
        exact_receipt.adapter_name == output_adapter,
        "OpenEnv train receipt adapter {:?} differs from requested output adapter {:?}",
        exact_receipt.adapter_name,
        output_adapter
    );
    anyhow::ensure!(
        exact_receipt.training_data.sha256.as_deref()
            == Some(admitted.admitted_corpus_sha256.as_str()),
        "OpenEnv train receipt corpus digest differs from admission"
    );
    anyhow::ensure!(
        exact_receipt.training_data.openenv.as_ref() == Some(admitted_openenv),
        "OpenEnv train receipt semantic corpus lineage differs from admission"
    );

    let manifest_path = adapter_dir.join(kiln_train::ADAPTER_MANIFEST_FILENAME);
    let (manifest_sha256, manifest_bytes) = verified_training_evidence_bytes(&manifest_path)?;
    let exact_manifest: kiln_train::AdapterManifest = serde_json::from_slice(&manifest_bytes)
        .context("parse bounded verified OpenEnv adapter manifest")?;
    exact_manifest
        .validate()
        .context("validate bounded verified OpenEnv adapter manifest")?;
    anyhow::ensure!(
        exact_manifest.schema_version == kiln_train::ADAPTER_MANIFEST_SCHEMA_VERSION
            && exact_manifest.manifest_type == "kiln_adapter_manifest",
        "completed OpenEnv trainer published an unsupported adapter manifest envelope"
    );
    anyhow::ensure!(
        exact_manifest.adapter_name == output_adapter,
        "OpenEnv adapter manifest adapter {:?} differs from requested output adapter {:?}",
        exact_manifest.adapter_name,
        output_adapter
    );
    anyhow::ensure!(
        exact_manifest.files.train_receipt.as_deref() == Some(kiln_train::TRAIN_RECEIPT_FILENAME),
        "OpenEnv adapter manifest does not name the canonical train receipt"
    );
    anyhow::ensure!(
        exact_manifest.receipt_hash.as_deref() == Some(receipt_sha256.as_str()),
        "OpenEnv adapter manifest receipt hash differs from train_receipt.json"
    );
    anyhow::ensure!(
        exact_manifest.training_data_hash.as_deref()
            == Some(admitted.admitted_corpus_sha256.as_str()),
        "OpenEnv adapter manifest corpus digest differs from admission"
    );
    anyhow::ensure!(
        exact_manifest.openenv_training_data.as_ref() == Some(admitted_openenv),
        "OpenEnv adapter manifest semantic corpus lineage differs from admission"
    );

    let run_dir = state.openenv_runs.run_dir(run_id);
    publish_training_evidence_bytes(
        &run_dir.join(kiln_train::TRAIN_RECEIPT_FILENAME),
        &receipt_bytes,
        &receipt_sha256,
    )?;
    publish_training_evidence_bytes(
        &run_dir.join(kiln_train::ADAPTER_MANIFEST_FILENAME),
        &manifest_bytes,
        &manifest_sha256,
    )?;
    let prefix = format!("/v1/openenv/runs/{run_id}/artifacts");
    Ok(vec![
        OpenEnvArtifact {
            kind: "train_receipt".into(),
            url: format!("{prefix}/train_receipt"),
            sha256: receipt_sha256,
            bytes: receipt_bytes.len(),
        },
        OpenEnvArtifact {
            kind: "adapter_manifest".into(),
            url: format!("{prefix}/adapter_manifest"),
            sha256: manifest_sha256,
            bytes: manifest_bytes.len(),
        },
    ])
}

fn verified_training_evidence_bytes(path: &Path) -> Result<(String, Vec<u8>)> {
    let (sha256, bytes) = crate::openenv_replay::bounded_artifact_metadata_with_limit(
        path,
        MAX_OPENENV_TRAINING_EVIDENCE_BYTES,
    )
    .with_context(|| format!("bind native training evidence {}", path.display()))?;
    let source =
        crate::openenv_replay::open_verified_artifact(path, &sha256, bytes).with_context(|| {
            format!(
                "reopen verified native training evidence {}",
                path.display()
            )
        })?;
    let mut contents = Vec::with_capacity(bytes);
    source
        .take((MAX_OPENENV_TRAINING_EVIDENCE_BYTES as u64).saturating_add(1))
        .read_to_end(&mut contents)
        .with_context(|| format!("read verified native training evidence {}", path.display()))?;
    anyhow::ensure!(
        contents.len() == bytes,
        "native training evidence {} changed length while reading",
        path.display()
    );
    Ok((sha256, contents))
}

fn publish_training_evidence_bytes(path: &Path, bytes: &[u8], expected_sha256: &str) -> Result<()> {
    let parent = path
        .parent()
        .context("OpenEnv training evidence path has no parent")?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("stage OpenEnv training evidence beside {}", path.display()))?;
    staged
        .write_all(bytes)
        .with_context(|| format!("write staged OpenEnv training evidence {}", path.display()))?;
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("sync staged OpenEnv training evidence {}", path.display()))?;
    staged
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish OpenEnv training evidence {}", path.display()))?;
    let (actual_sha256, actual_bytes) =
        crate::openenv_replay::bounded_artifact_metadata_with_limit(
            path,
            MAX_OPENENV_TRAINING_EVIDENCE_BYTES,
        )
        .with_context(|| {
            format!(
                "verify published OpenEnv training evidence {}",
                path.display()
            )
        })?;
    anyhow::ensure!(
        actual_sha256 == expected_sha256 && actual_bytes == bytes.len(),
        "published OpenEnv training evidence {} differs from its native source",
        path.display()
    );
    Ok(())
}
