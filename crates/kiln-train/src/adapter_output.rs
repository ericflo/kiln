//! Adapter output receipts and install helpers.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const ADAPTER_RECEIPT_FILENAME: &str = "adapter_receipt.json";

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
        assert_eq!(std::fs::read_link(&installed).unwrap(), source_one);

        let installed = install_adapter_symlink(&source_two, &registry, "agent").unwrap();
        assert_eq!(std::fs::read_link(&installed).unwrap(), source_two);
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
