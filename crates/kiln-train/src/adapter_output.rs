//! Adapter output receipts and install helpers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const ADAPTER_RECEIPT_FILENAME: &str = "adapter_receipt.json";
pub const ADAPTER_MANIFEST_FILENAME: &str = "adapter_manifest.json";
pub const ADAPTER_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterOutputReceipt {
    pub schema_version: u32,
    pub adapter_name: String,
    pub adapter_dir: String,
    pub adapter_config: String,
    pub adapter_model: String,
    pub rank: usize,
    pub alpha: f32,
    pub alpha_over_rank: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub installed_adapter_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdapterManifest {
    pub schema_version: u32,
    pub manifest_type: String,
    pub adapter_name: String,
    pub safetensors_hash: String,
    pub config_hash: String,
    pub receipt_hash: Option<String>,
    pub parent_adapter: Option<String>,
    pub model_config_hash: Option<String>,
    pub kiln_commit: Option<String>,
    pub training_data_hash: Option<String>,
    pub training_data_source: Option<String>,
    pub training_data_path: Option<String>,
    pub files: AdapterManifestFiles,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdapterManifestFiles {
    pub adapter_model: String,
    pub adapter_config: String,
    pub train_receipt: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AdapterRestoreOptions {
    pub manifest_path: PathBuf,
    pub adapter_dir: PathBuf,
    pub adapter_name: Option<String>,
    pub overwrite: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdapterRestoreReceipt {
    pub status: String,
    pub manifest_path: String,
    pub source_adapter_dir: String,
    pub restored_adapter_dir: String,
    pub adapter_name: String,
    pub copied_files: Vec<String>,
    pub verified_hashes: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct AdapterOutputConfig {
    r: usize,
    lora_alpha: f32,
}

pub fn validate_adapter_output_dir(adapter_dir: &Path) -> Result<PathBuf> {
    if !adapter_dir.exists() || !adapter_dir.is_dir() {
        bail!("adapter directory does not exist: {}", adapter_dir.display());
    }
    let resolved = adapter_dir
        .canonicalize()
        .with_context(|| format!("resolve adapter dir {}", adapter_dir.display()))?;
    let mut missing = Vec::new();
    if !resolved.join("adapter_config.json").is_file() {
        missing.push("adapter_config.json");
    }
    if !resolved.join("adapter_model.safetensors").is_file() {
        missing.push("adapter_model.safetensors");
    }
    if !missing.is_empty() {
        let mut detail = format!(
            "{} is missing required file(s): {}",
            resolved.display(),
            missing.join(", ")
        );
        if let Some(child) = find_single_nested_adapter_dir(&resolved) {
            detail.push_str(&format!(
                "; found nested adapter directory at {}",
                child.display()
            ));
        }
        bail!(detail);
    }
    Ok(resolved)
}

pub fn write_adapter_output_receipt(
    adapter_dir: &Path,
    adapter_name: &str,
    installed_adapter_dir: Option<&Path>,
) -> Result<PathBuf> {
    let resolved = validate_adapter_output_dir(adapter_dir)?;
    let adapter_config_path = resolved.join("adapter_config.json");
    let adapter_config: AdapterOutputConfig = serde_json::from_slice(
        &std::fs::read(&adapter_config_path)
            .with_context(|| format!("read {}", adapter_config_path.display()))?,
    )
    .with_context(|| format!("parse {}", adapter_config_path.display()))?;
    let alpha_over_rank =
        crate::lora_scaling::alpha_over_rank(adapter_config.r, adapter_config.lora_alpha)?;
    let installed = installed_adapter_dir.map(|path| path.display().to_string());
    let receipt = AdapterOutputReceipt {
        schema_version: 1,
        adapter_name: adapter_name.to_string(),
        adapter_dir: resolved.display().to_string(),
        adapter_config: adapter_config_path.display().to_string(),
        adapter_model: resolved
            .join("adapter_model.safetensors")
            .display()
            .to_string(),
        rank: adapter_config.r,
        alpha: adapter_config.lora_alpha,
        alpha_over_rank,
        installed_adapter_dir: installed,
    };
    let path = resolved.join(ADAPTER_RECEIPT_FILENAME);
    let json = serde_json::to_string_pretty(&receipt).context("serialize adapter output receipt")?;
    std::fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

pub fn write_adapter_manifest_from_train_receipt(
    adapter_dir: &Path,
    receipt: &crate::train_receipt::TrainReceipt,
) -> Result<Option<PathBuf>> {
    if receipt.status != crate::train_receipt::TrainReceiptStatus::Success {
        return Ok(None);
    }
    if !adapter_dir.join("adapter_config.json").is_file()
        || !adapter_dir.join("adapter_model.safetensors").is_file()
    {
        return Ok(None);
    }

    let resolved = validate_adapter_output_dir(adapter_dir)?;
    let manifest = build_adapter_manifest_from_train_receipt(&resolved, receipt)?;
    let path = resolved.join(ADAPTER_MANIFEST_FILENAME);
    let json = serde_json::to_string_pretty(&manifest).context("serialize adapter manifest")?;
    std::fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
    Ok(Some(path))
}

pub fn build_adapter_manifest_from_train_receipt(
    adapter_dir: &Path,
    receipt: &crate::train_receipt::TrainReceipt,
) -> Result<AdapterManifest> {
    let resolved = validate_adapter_output_dir(adapter_dir)?;
    let adapter_model = resolved.join("adapter_model.safetensors");
    let adapter_config = resolved.join("adapter_config.json");
    let train_receipt = resolved.join(crate::train_receipt::TRAIN_RECEIPT_FILENAME);
    let receipt_hash = if train_receipt.is_file() {
        Some(crate::train_receipt::sha256_file(&train_receipt)?)
    } else {
        None
    };

    Ok(AdapterManifest {
        schema_version: ADAPTER_MANIFEST_SCHEMA_VERSION,
        manifest_type: "kiln_adapter_manifest".to_string(),
        adapter_name: receipt.adapter_name.clone(),
        safetensors_hash: crate::train_receipt::sha256_file(&adapter_model)?,
        config_hash: crate::train_receipt::sha256_file(&adapter_config)?,
        receipt_hash,
        parent_adapter: parent_adapter_from_receipt(receipt),
        model_config_hash: receipt
            .model
            .config_hash
            .clone()
            .or_else(|| receipt.config_hashes.model_config_hash.clone()),
        kiln_commit: receipt.kiln.git_commit.clone(),
        training_data_hash: receipt.training_data.sha256.clone(),
        training_data_source: Some(receipt.training_data.source.clone()),
        training_data_path: receipt.training_data.path.clone(),
        files: AdapterManifestFiles {
            adapter_model: "adapter_model.safetensors".to_string(),
            adapter_config: "adapter_config.json".to_string(),
            train_receipt: train_receipt
                .is_file()
                .then(|| crate::train_receipt::TRAIN_RECEIPT_FILENAME.to_string()),
        },
    })
}

pub fn read_adapter_manifest(path: &Path) -> Result<AdapterManifest> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn read_adapter_manifest_from_adapter_dir(
    adapter_dir: &Path,
) -> Result<Option<AdapterManifest>> {
    let path = adapter_dir.join(ADAPTER_MANIFEST_FILENAME);
    if !path.is_file() {
        return Ok(None);
    }
    read_adapter_manifest(&path).map(Some)
}

pub fn restore_adapter_from_manifest(
    options: AdapterRestoreOptions,
) -> Result<AdapterRestoreReceipt> {
    let manifest_path = options
        .manifest_path
        .canonicalize()
        .with_context(|| format!("resolve manifest {}", options.manifest_path.display()))?;
    let manifest = read_adapter_manifest(&manifest_path)?;
    let source_dir = manifest_path
        .parent()
        .context("adapter manifest must have a parent directory")?
        .to_path_buf();
    let adapter_name = options
        .adapter_name
        .unwrap_or_else(|| manifest.adapter_name.clone());
    validate_install_adapter_name(&adapter_name)?;

    std::fs::create_dir_all(&options.adapter_dir)
        .with_context(|| format!("create adapter dir {}", options.adapter_dir.display()))?;
    let target_root = options
        .adapter_dir
        .canonicalize()
        .with_context(|| format!("resolve adapter dir {}", options.adapter_dir.display()))?;
    let final_path = target_root.join(&adapter_name);
    if final_path.exists() || final_path.symlink_metadata().is_ok() {
        if options.overwrite {
            remove_path(&final_path)?;
        } else {
            bail!(
                "refusing to overwrite existing adapter {}; pass --overwrite to replace it",
                final_path.display()
            );
        }
    }

    let tmp_path = target_root.join(format!(
        ".restore-tmp-{}-{}",
        std::process::id(),
        adapter_name
    ));
    if tmp_path.exists() || tmp_path.symlink_metadata().is_ok() {
        remove_path(&tmp_path)?;
    }
    std::fs::create_dir_all(&tmp_path)
        .with_context(|| format!("create temporary restore dir {}", tmp_path.display()))?;

    let mut copied_files = Vec::new();
    copy_manifest_file(
        &source_dir,
        &tmp_path,
        &manifest.files.adapter_config,
        "adapter_config.json",
        &mut copied_files,
    )?;
    copy_manifest_file(
        &source_dir,
        &tmp_path,
        &manifest.files.adapter_model,
        "adapter_model.safetensors",
        &mut copied_files,
    )?;
    if let Some(train_receipt) = manifest.files.train_receipt.as_deref() {
        copy_manifest_file(
            &source_dir,
            &tmp_path,
            train_receipt,
            crate::train_receipt::TRAIN_RECEIPT_FILENAME,
            &mut copied_files,
        )?;
    }
    std::fs::copy(&manifest_path, tmp_path.join(ADAPTER_MANIFEST_FILENAME)).with_context(|| {
        format!(
            "copy {} to {}",
            manifest_path.display(),
            tmp_path.join(ADAPTER_MANIFEST_FILENAME).display()
        )
    })?;
    copied_files.push(ADAPTER_MANIFEST_FILENAME.to_string());

    let mut verified_hashes = BTreeMap::new();
    verify_manifest_hash(
        &tmp_path.join("adapter_config.json"),
        &manifest.config_hash,
        "config_hash",
        &mut verified_hashes,
    )?;
    verify_manifest_hash(
        &tmp_path.join("adapter_model.safetensors"),
        &manifest.safetensors_hash,
        "safetensors_hash",
        &mut verified_hashes,
    )?;
    if let Some(expected) = manifest.receipt_hash.as_deref() {
        verify_manifest_hash(
            &tmp_path.join(crate::train_receipt::TRAIN_RECEIPT_FILENAME),
            expected,
            "receipt_hash",
            &mut verified_hashes,
        )?;
    }
    validate_adapter_output_dir(&tmp_path)?;

    std::fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "move restored adapter {} to {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    let restored = validate_adapter_output_dir(&final_path)?;
    Ok(AdapterRestoreReceipt {
        status: "ok".to_string(),
        manifest_path: manifest_path.display().to_string(),
        source_adapter_dir: source_dir.display().to_string(),
        restored_adapter_dir: restored.display().to_string(),
        adapter_name,
        copied_files,
        verified_hashes,
    })
}

pub fn install_adapter_symlink(
    source_adapter_dir: &Path,
    install_adapter_dir: &Path,
    install_adapter_name: &str,
) -> Result<PathBuf> {
    let source = validate_adapter_output_dir(source_adapter_dir)?;
    validate_install_adapter_name(install_adapter_name)?;
    std::fs::create_dir_all(install_adapter_dir)
        .with_context(|| format!("create install dir {}", install_adapter_dir.display()))?;
    let install_root = install_adapter_dir
        .canonicalize()
        .with_context(|| format!("resolve install dir {}", install_adapter_dir.display()))?;
    let final_path = install_root.join(install_adapter_name);
    let tmp_path = install_root.join(format!(
        ".install-tmp-{}-{}",
        std::process::id(),
        install_adapter_name
    ));
    if tmp_path.exists() || tmp_path.symlink_metadata().is_ok() {
        remove_path(&tmp_path)?;
    }

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&source, &tmp_path)
            .with_context(|| format!("create adapter symlink {}", tmp_path.display()))?;
        validate_adapter_output_dir(&tmp_path).with_context(|| {
            format!(
                "validate temporary adapter install {} before replacing {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        if final_path.exists() || final_path.symlink_metadata().is_ok() {
            let meta = final_path
                .symlink_metadata()
                .with_context(|| format!("stat existing install {}", final_path.display()))?;
            if meta.file_type().is_dir() && !meta.file_type().is_symlink() {
                remove_path(&tmp_path)?;
                bail!(
                    "refusing to replace existing adapter directory {}; delete it first or choose another --install-adapter-name",
                    final_path.display()
                );
            }
        }
        std::fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "atomically replace adapter symlink {} with {}",
                final_path.display(),
                tmp_path.display()
            )
        })?;
    }

    #[cfg(not(unix))]
    {
        copy_dir_recursive(&source, &tmp_path)?;
        validate_adapter_output_dir(&tmp_path).with_context(|| {
            format!(
                "validate temporary adapter install {} before replacing {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
        if final_path.exists() || final_path.symlink_metadata().is_ok() {
            remove_path(&tmp_path)?;
            bail!(
                "refusing to replace existing adapter path {} on non-Unix platforms",
                final_path.display()
            );
        }
        std::fs::rename(&tmp_path, &final_path).with_context(|| {
            format!(
                "move temporary adapter install {} to {}",
                tmp_path.display(),
                final_path.display()
            )
        })?;
    }

    validate_adapter_output_dir(&final_path).with_context(|| {
        format!(
            "validate installed adapter {} after install",
            final_path.display()
        )
    })?;
    Ok(final_path)
}

pub fn validate_install_adapter_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
    {
        bail!(
            "install adapter name must be non-empty, path-safe, and contain no separators; got {name:?}"
        );
    }
    Ok(())
}

fn parent_adapter_from_receipt(receipt: &crate::train_receipt::TrainReceipt) -> Option<String> {
    receipt
        .config
        .get("base_adapter")
        .and_then(|v| v.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .or_else(|| receipt.adapters.base.path.clone())
}

fn copy_manifest_file(
    source_dir: &Path,
    target_dir: &Path,
    manifest_name: &str,
    output_name: &str,
    copied_files: &mut Vec<String>,
) -> Result<()> {
    if manifest_name.contains('/') || manifest_name.contains('\\') {
        bail!("manifest file entry must be a plain filename: {manifest_name:?}");
    }
    let source = source_dir.join(manifest_name);
    let target = target_dir.join(output_name);
    std::fs::copy(&source, &target)
        .with_context(|| format!("copy {} to {}", source.display(), target.display()))?;
    copied_files.push(output_name.to_string());
    Ok(())
}

fn verify_manifest_hash(
    path: &Path,
    expected: &str,
    label: &str,
    verified_hashes: &mut BTreeMap<String, String>,
) -> Result<()> {
    let actual = crate::train_receipt::sha256_file(path)?;
    if !hashes_equal(&actual, expected) {
        bail!(
            "{label} mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    verified_hashes.insert(label.to_string(), actual);
    Ok(())
}

fn hashes_equal(actual: &str, expected: &str) -> bool {
    actual == expected
        || actual.strip_prefix("sha256:") == Some(expected)
        || expected.strip_prefix("sha256:") == Some(actual)
}

fn find_single_nested_adapter_dir(parent: &Path) -> Option<PathBuf> {
    let mut matches = Vec::new();
    let entries = std::fs::read_dir(parent).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir()
            && path.join("adapter_config.json").is_file()
            && path.join("adapter_model.safetensors").is_file()
        {
            matches.push(path);
        }
    }
    if matches.len() == 1 {
        matches.pop()
    } else {
        None
    }
}

fn remove_path(path: &Path) -> Result<()> {
    let meta = path
        .symlink_metadata()
        .with_context(|| format!("stat {}", path.display()))?;
    if meta.file_type().is_dir() && !meta.file_type().is_symlink() {
        std::fs::remove_dir_all(path).with_context(|| format!("remove dir {}", path.display()))
    } else {
        std::fs::remove_file(path).with_context(|| format!("remove file {}", path.display()))
    }
}

#[cfg(not(unix))]
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).with_context(|| format!("create dir {}", dst.display()))?;
    for entry in std::fs::read_dir(src).with_context(|| format!("read dir {}", src.display()))? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let meta = entry.metadata()?;
        if meta.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else if meta.is_file() {
            std::fs::copy(&src_path, &dst_path).with_context(|| {
                format!("copy {} to {}", src_path.display(), dst_path.display())
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_core::config::ModelConfig;
    use kiln_core::tokenizer::KilnTokenizer;

    fn write_minimal_adapter(path: &Path) {
        std::fs::create_dir_all(path).unwrap();
        std::fs::write(
            path.join("adapter_config.json"),
            serde_json::json!({
                "r": 2,
                "lora_alpha": 4.0,
                "target_modules": ["q_proj"],
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(path.join("adapter_model.safetensors"), "weights").unwrap();
    }

    fn minimal_tokenizer() -> Result<KilnTokenizer> {
        let json = br#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1},
                "unk_token": "[UNK]"
            }
        }"#;
        KilnTokenizer::from_bytes(json).map_err(|err| anyhow::anyhow!("{err}"))
    }

    fn manifest_test_receipt(adapter_name: &str) -> crate::train_receipt::TrainReceipt {
        let model = ModelConfig::qwen3_5_4b();
        let tokenizer = minimal_tokenizer().unwrap();
        let mut receipt = crate::train_receipt::TrainReceipt::new(
            adapter_name,
            "grpo",
            &model,
            &tokenizer,
            crate::train_receipt::HyperparameterReceipt {
                mode: "grpo".to_string(),
                rank: 2,
                alpha: 4.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed: Some(11),
            },
            serde_json::json!({"base_adapter": "parent-v1"}),
        );
        receipt.kiln.git_commit = Some("abc123".to_string());
        receipt.model.config_hash = Some("sha256:model-config".to_string());
        receipt.training_data = crate::train_receipt::TrainingDataReceipt {
            source: "jsonl_grpo_groups".to_string(),
            path: Some("/data/groups.jsonl".to_string()),
            sha256: Some("sha256:training-data".to_string()),
        };
        receipt
    }

    fn write_train_receipt(adapter: &Path, receipt: &crate::train_receipt::TrainReceipt) {
        std::fs::write(
            adapter.join(crate::train_receipt::TRAIN_RECEIPT_FILENAME),
            serde_json::to_string_pretty(receipt).unwrap(),
        )
        .unwrap();
    }

    #[cfg(unix)]
    fn canonical_symlink_target(path: &Path) -> PathBuf {
        let target = std::fs::read_link(path).unwrap();
        let resolved = if target.is_absolute() {
            target
        } else {
            path.parent().unwrap().join(target)
        };
        resolved.canonicalize().unwrap()
    }

    #[test]
    fn adapter_output_receipt_records_canonical_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let adapter = tmp.path().join("actual");
        write_minimal_adapter(&adapter);

        let receipt_path = write_adapter_output_receipt(&adapter, "actual", None).unwrap();
        let receipt: AdapterOutputReceipt =
            serde_json::from_slice(&std::fs::read(&receipt_path).unwrap()).unwrap();

        assert_eq!(receipt.adapter_name, "actual");
        assert_eq!(
            receipt.adapter_dir,
            adapter.canonicalize().unwrap().display().to_string()
        );
        assert_eq!(receipt.rank, 2);
        assert_eq!(receipt.alpha, 4.0);
        assert_eq!(receipt.alpha_over_rank, 2.0);
        assert!(receipt.adapter_config.ends_with("adapter_config.json"));
        assert!(receipt.adapter_model.ends_with("adapter_model.safetensors"));
    }

    #[test]
    fn adapter_manifest_records_training_provenance_and_hashes() {
        let tmp = tempfile::tempdir().unwrap();
        let adapter = tmp.path().join("actual");
        write_minimal_adapter(&adapter);
        let receipt = manifest_test_receipt("actual");
        write_train_receipt(&adapter, &receipt);

        let manifest_path =
            write_adapter_manifest_from_train_receipt(&adapter, &receipt).unwrap().unwrap();
        let manifest = read_adapter_manifest(&manifest_path).unwrap();

        assert_eq!(manifest.schema_version, ADAPTER_MANIFEST_SCHEMA_VERSION);
        assert_eq!(manifest.manifest_type, "kiln_adapter_manifest");
        assert_eq!(manifest.adapter_name, "actual");
        assert_eq!(manifest.parent_adapter.as_deref(), Some("parent-v1"));
        assert_eq!(manifest.kiln_commit.as_deref(), Some("abc123"));
        assert_eq!(manifest.model_config_hash.as_deref(), Some("sha256:model-config"));
        assert_eq!(manifest.training_data_hash.as_deref(), Some("sha256:training-data"));
        assert_eq!(manifest.files.adapter_model, "adapter_model.safetensors");
        assert_eq!(manifest.files.adapter_config, "adapter_config.json");
        assert_eq!(
            manifest.files.train_receipt.as_deref(),
            Some(crate::train_receipt::TRAIN_RECEIPT_FILENAME)
        );
        assert!(manifest.safetensors_hash.starts_with("sha256:"));
        assert!(manifest.config_hash.starts_with("sha256:"));
        assert!(manifest.receipt_hash.as_deref().unwrap().starts_with("sha256:"));
    }

    #[test]
    fn restore_adapter_from_manifest_copies_and_verifies_files() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        let registry = tmp.path().join("registry");
        write_minimal_adapter(&source);
        let receipt = manifest_test_receipt("source-adapter");
        write_train_receipt(&source, &receipt);
        let manifest_path =
            write_adapter_manifest_from_train_receipt(&source, &receipt).unwrap().unwrap();

        let restore = restore_adapter_from_manifest(AdapterRestoreOptions {
            manifest_path,
            adapter_dir: registry.clone(),
            adapter_name: Some("restored".to_string()),
            overwrite: false,
        })
        .unwrap();

        let restored = registry.join("restored");
        assert_eq!(restore.status, "ok");
        assert_eq!(restore.adapter_name, "restored");
        assert!(restored.join("adapter_config.json").is_file());
        assert!(restored.join("adapter_model.safetensors").is_file());
        assert!(restored.join(crate::train_receipt::TRAIN_RECEIPT_FILENAME).is_file());
        assert!(restored.join(ADAPTER_MANIFEST_FILENAME).is_file());
        assert!(restore.verified_hashes.contains_key("config_hash"));
        assert!(restore.verified_hashes.contains_key("safetensors_hash"));
        assert!(restore.verified_hashes.contains_key("receipt_hash"));
    }

    #[test]
    fn restore_adapter_from_manifest_rejects_hash_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        let registry = tmp.path().join("registry");
        write_minimal_adapter(&source);
        let receipt = manifest_test_receipt("source-adapter");
        write_train_receipt(&source, &receipt);
        let manifest_path =
            write_adapter_manifest_from_train_receipt(&source, &receipt).unwrap().unwrap();
        std::fs::write(source.join("adapter_model.safetensors"), "corrupt").unwrap();

        let err = restore_adapter_from_manifest(AdapterRestoreOptions {
            manifest_path,
            adapter_dir: registry,
            adapter_name: None,
            overwrite: false,
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("safetensors_hash mismatch"));
    }

    #[test]
    fn adapter_output_validation_reports_nested_adapter() {
        let tmp = tempfile::tempdir().unwrap();
        let parent = tmp.path().join("parent");
        write_minimal_adapter(&parent.join("child"));

        let err = validate_adapter_output_dir(&parent).unwrap_err().to_string();

        assert!(err.contains("adapter_config.json"));
        assert!(err.contains("adapter_model.safetensors"));
        assert!(err.contains("nested adapter directory"));
    }

    #[cfg(unix)]
    #[test]
    fn install_adapter_symlink_replaces_existing_symlink_after_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let source_one = tmp.path().join("source-one");
        let source_two = tmp.path().join("source-two");
        let registry = tmp.path().join("registry");
        write_minimal_adapter(&source_one);
        write_minimal_adapter(&source_two);

        let installed = install_adapter_symlink(&source_one, &registry, "agent").unwrap();
        assert_eq!(
            canonical_symlink_target(&installed),
            source_one.canonicalize().unwrap()
        );

        let installed = install_adapter_symlink(&source_two, &registry, "agent").unwrap();
        assert_eq!(
            canonical_symlink_target(&installed),
            source_two.canonicalize().unwrap()
        );
        assert!(validate_adapter_output_dir(&installed).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn install_adapter_symlink_refuses_existing_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        let registry = tmp.path().join("registry");
        write_minimal_adapter(&source);
        write_minimal_adapter(&registry.join("agent"));

        let err = install_adapter_symlink(&source, &registry, "agent")
            .unwrap_err()
            .to_string();

        assert!(err.contains("refusing to replace existing adapter directory"));
    }
}
