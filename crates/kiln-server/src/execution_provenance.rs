//! Production construction of the process-lifetime execution provenance.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use kiln_core::execution_provenance::{
    ExecutionBackendIdentity, ExecutionBuildIdentity, ExecutionConfigurationIdentity,
    ExecutionKernelIdentity, ExecutionModelIdentity, ExecutionPrecisionIdentity,
    ExecutionProvenanceV1,
};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::backend::TrainingPrecisionPolicy;

use crate::config::KilnConfig;

pub fn build_execution_provenance(
    config: &KilnConfig,
    model_config: &kiln_core::config::ModelConfig,
    tokenizer: &KilnTokenizer,
    backend_name: &str,
    device: kiln_tensor::Device,
    executable_sha256: &str,
    numerical_runtime_sha256: &str,
    training_precision: TrainingPrecisionPolicy,
) -> Result<ExecutionProvenanceV1> {
    let hashes =
        kiln_core::config_hashes::ConfigHashes::from_model_tokenizer(model_config, tokenizer, None);
    let model_config_sha256 = hashes
        .model_config_hash
        .context("serialize model configuration for execution provenance")?;
    let tokenizer_config_sha256 = hashes
        .tokenizer_config_hash
        .context("serialize tokenizer configuration for execution provenance")?;
    let effective_server_config_sha256 = kiln_core::config_hashes::sha256_json_serializable(config)
        .context("serialize effective server configuration for execution provenance")?;
    let effective_environment_sha256 =
        kiln_core::config_hashes::sha256_json_serializable(&effective_kiln_environment())
            .context("serialize effective KILN environment for execution provenance")?;
    let (git_commit, source_dirty) = detect_source_revision();
    let source_tree_sha256 = std::env::var("KILN_SOURCE_TREE_HASH")
        .ok()
        .filter(|value| !value.trim().is_empty());

    let kernels = ExecutionKernelIdentity::new(kernel_versions(), compiled_features())
        .context("construct execution kernel identity")?;
    ExecutionProvenanceV1::new(
        ExecutionBackendIdentity {
            name: backend_name.to_string(),
            device: device.short_name(),
            numerical_runtime_sha256: normalize_sha256(numerical_runtime_sha256),
        },
        ExecutionBuildIdentity {
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            target: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            executable_sha256: normalize_sha256(executable_sha256),
            git_commit,
            source_tree_sha256,
            source_dirty,
        },
        ExecutionModelIdentity {
            model_config_sha256,
            tokenizer_vocab_sha256: tokenizer.vocab_identity_sha256(),
            tokenizer_config_sha256,
            chat_template_sha256: hashes.chat_template_hash,
            training_chat_template_sha256: hashes.training_chat_template_hash,
        },
        ExecutionPrecisionIdentity {
            inference_dtype: match model_config.dtype {
                kiln_core::config::DType::BF16 => "bf16",
                kiln_core::config::DType::FP16 => "f16",
                kiln_core::config::DType::FP32 => "f32",
            }
            .to_string(),
            training_policy: training_precision.name.to_string(),
        },
        kernels,
        ExecutionConfigurationIdentity {
            effective_server_config_sha256,
            effective_environment_sha256,
        },
    )
    .context("construct execution provenance")
}

fn normalize_sha256(value: &str) -> String {
    if value.starts_with("sha256:") {
        value.to_string()
    } else {
        format!("sha256:{value}")
    }
}

fn effective_kiln_environment() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| key.starts_with("KILN_"))
        .map(|(key, value)| {
            let value = if environment_key_is_sensitive(&key) {
                "<redacted-present>".to_string()
            } else {
                value
            };
            (key, value)
        })
        .collect()
}

fn environment_key_is_sensitive(key: &str) -> bool {
    key.split('_').any(|part| {
        matches!(
            part,
            "AUTH"
                | "CREDENTIAL"
                | "CREDENTIALS"
                | "KEY"
                | "PASSWORD"
                | "PRIVATE"
                | "SECRET"
                | "TOKEN"
        )
    })
}

fn kernel_versions() -> BTreeMap<String, String> {
    let version = env!("CARGO_PKG_VERSION").to_string();
    [
        "kiln-conv1d-kernel",
        "kiln-flash-attn",
        "kiln-flce-kernel",
        "kiln-gdn-kernel",
        "kiln-model",
        "kiln-opd-loss-kernel",
        "kiln-rmsnorm-kernel",
        "kiln-tensor",
        "kiln-train",
    ]
    .into_iter()
    .map(|name| (name.to_string(), version.clone()))
    .collect()
}

fn compiled_features() -> Vec<String> {
    let mut features = Vec::new();
    if cfg!(feature = "cuda") {
        features.push("cuda".to_string());
    }
    if cfg!(feature = "rocm") {
        features.push("rocm".to_string());
    }
    if cfg!(feature = "vulkan") {
        features.push("vulkan".to_string());
    }
    if cfg!(feature = "metal") {
        features.push("metal".to_string());
    }
    if cfg!(feature = "nvtx") {
        features.push("nvtx".to_string());
    }
    if let Some(arches) = option_env!("KILN_ROCM_ARCHS")
        && !arches.trim().is_empty()
    {
        features.push(format!("rocm-archs={}", arches.trim()));
    }
    features
}

fn detect_source_revision() -> (Option<String>, Option<bool>) {
    if let Some(commit) = std::env::var("KILN_COMMIT")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return (Some(commit), None);
    }
    let Some(root) = source_root() else {
        return (None, None);
    };
    let commit = git_output(&root, &["rev-parse", "HEAD"]);
    let dirty = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty());
    (commit, dirty)
}

fn source_root() -> Option<PathBuf> {
    std::env::var_os("KILN_REPO_ROOT")
        .map(PathBuf::from)
        .or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(2)
                .map(Path::to_path_buf)
        })
        .filter(|root| root.is_dir())
}

fn git_output(root: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
pub(crate) fn test_execution_provenance() -> ExecutionProvenanceV1 {
    use kiln_core::execution_provenance::{
        ExecutionBackendIdentity, ExecutionBuildIdentity, ExecutionConfigurationIdentity,
        ExecutionKernelIdentity, ExecutionModelIdentity, ExecutionPrecisionIdentity,
    };

    let hash = |byte: char| format!("sha256:{}", byte.to_string().repeat(64));
    ExecutionProvenanceV1::new(
        ExecutionBackendIdentity {
            name: "test".into(),
            device: "cpu".into(),
            numerical_runtime_sha256: hash('1'),
        },
        ExecutionBuildIdentity {
            package_version: env!("CARGO_PKG_VERSION").into(),
            target: "linux-x86_64".into(),
            executable_sha256: hash('2'),
            git_commit: Some("test-commit".into()),
            source_tree_sha256: Some(hash('3')),
            source_dirty: Some(false),
        },
        ExecutionModelIdentity {
            model_config_sha256: hash('4'),
            tokenizer_vocab_sha256: hash('5'),
            tokenizer_config_sha256: hash('6'),
            chat_template_sha256: Some(hash('7')),
            training_chat_template_sha256: Some(hash('8')),
        },
        ExecutionPrecisionIdentity {
            inference_dtype: "f32".into(),
            training_policy: "cpu_f32_reference".into(),
        },
        ExecutionKernelIdentity::new(
            BTreeMap::from([("kiln-model".into(), env!("CARGO_PKG_VERSION").into())]),
            Vec::new(),
        )
        .unwrap(),
        ExecutionConfigurationIdentity {
            effective_server_config_sha256: hash('8'),
            effective_environment_sha256: hash('9'),
        },
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> KilnTokenizer {
        KilnTokenizer::from_bytes(
            br#"{
                "version":"1.0","truncation":null,"padding":null,
                "added_tokens":[],"normalizer":null,
                "pre_tokenizer":{"type":"Whitespace"},
                "post_processor":null,"decoder":null,
                "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn production_builder_records_complete_bounded_execution_envelope() {
        let provenance = build_execution_provenance(
            &KilnConfig::default(),
            &kiln_core::config::ModelConfig::qwen3_5_4b(),
            &tokenizer(),
            "cpu",
            kiln_tensor::Device::Cpu,
            &"1".repeat(64),
            &"2".repeat(64),
            TrainingPrecisionPolicy::portable(),
        )
        .unwrap();

        provenance.validate().unwrap();
        assert_eq!(provenance.backend.name, "cpu");
        assert_eq!(provenance.backend.device, "cpu");
        assert_eq!(provenance.precision.training_policy, "cpu_f32_reference");
        assert_eq!(provenance.kernels.versions.len(), 9);
        assert!(provenance.build.executable_sha256.starts_with("sha256:"));
        assert!(
            provenance
                .configuration
                .effective_server_config_sha256
                .starts_with("sha256:")
        );
    }

    #[test]
    fn execution_environment_redacts_secret_values_without_hiding_numeric_tokens() {
        assert!(environment_key_is_sensitive("KILN_VLLM_API_KEY"));
        assert!(environment_key_is_sensitive("KILN_REMOTE_ACCESS_TOKEN"));
        assert!(!environment_key_is_sensitive(
            "KILN_MAX_PREFILL_TOKENS_PER_CYCLE"
        ));
        assert!(!environment_key_is_sensitive("KILN_TOKENIZER_PATH"));
    }
}
