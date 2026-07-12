//! Canonical identity of the executable and numerical environment behind work.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const EXECUTION_PROVENANCE_SCHEMA_VERSION: u32 = 1;
pub const EXECUTION_PROVENANCE_TYPE: &str = "kiln.execution-provenance.v1";
pub const KERNEL_CONTRACT_TYPE: &str = "kiln.kernel-contract.v1";

const MAX_TEXT_BYTES: usize = 512;
const MAX_KERNEL_ENTRIES: usize = 64;
const MAX_FEATURES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct ExecutionProvenanceError {
    message: String,
}

impl ExecutionProvenanceError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionBackendIdentity {
    pub name: String,
    pub device: String,
    pub numerical_runtime_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionBuildIdentity {
    pub package_version: String,
    pub target: String,
    pub executable_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_tree_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dirty: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionModelIdentity {
    pub model_config_sha256: String,
    pub tokenizer_vocab_sha256: String,
    pub tokenizer_config_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPrecisionIdentity {
    pub inference_dtype: String,
    pub training_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionKernelIdentity {
    pub contract_type: String,
    pub versions: BTreeMap<String, String>,
    pub compiled_features: Vec<String>,
    pub contract_sha256: String,
}

impl ExecutionKernelIdentity {
    pub fn new(
        versions: BTreeMap<String, String>,
        mut compiled_features: Vec<String>,
    ) -> Result<Self, ExecutionProvenanceError> {
        compiled_features.sort();
        compiled_features.dedup();
        validate_kernel_fields(&versions, &compiled_features)?;
        let contract_sha256 = kernel_contract_sha256(&versions, &compiled_features)?;
        Ok(Self {
            contract_type: KERNEL_CONTRACT_TYPE.to_string(),
            versions,
            compiled_features,
            contract_sha256,
        })
    }

    fn validate(&self) -> Result<(), ExecutionProvenanceError> {
        if self.contract_type != KERNEL_CONTRACT_TYPE {
            return Err(ExecutionProvenanceError::new(format!(
                "invalid execution kernel contract_type {:?}",
                self.contract_type
            )));
        }
        validate_kernel_fields(&self.versions, &self.compiled_features)?;
        if !is_strictly_sorted_unique(&self.compiled_features) {
            return Err(ExecutionProvenanceError::new(
                "execution compiled_features must be strictly sorted and unique",
            ));
        }
        validate_sha256("execution kernel contract_sha256", &self.contract_sha256)?;
        let expected = kernel_contract_sha256(&self.versions, &self.compiled_features)?;
        if self.contract_sha256 != expected {
            return Err(ExecutionProvenanceError::new(format!(
                "execution kernel contract digest mismatch: manifest has {}, expected {expected}",
                self.contract_sha256
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfigurationIdentity {
    pub effective_server_config_sha256: String,
    pub effective_environment_sha256: String,
}

/// Immutable, self-verifying execution envelope captured at server startup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProvenanceV1 {
    pub schema_version: u32,
    pub provenance_type: String,
    pub backend: ExecutionBackendIdentity,
    pub build: ExecutionBuildIdentity,
    pub model: ExecutionModelIdentity,
    pub precision: ExecutionPrecisionIdentity,
    pub kernels: ExecutionKernelIdentity,
    pub configuration: ExecutionConfigurationIdentity,
    pub provenance_sha256: String,
}

impl ExecutionProvenanceV1 {
    pub fn new(
        backend: ExecutionBackendIdentity,
        build: ExecutionBuildIdentity,
        model: ExecutionModelIdentity,
        precision: ExecutionPrecisionIdentity,
        kernels: ExecutionKernelIdentity,
        configuration: ExecutionConfigurationIdentity,
    ) -> Result<Self, ExecutionProvenanceError> {
        let mut provenance = Self {
            schema_version: EXECUTION_PROVENANCE_SCHEMA_VERSION,
            provenance_type: EXECUTION_PROVENANCE_TYPE.to_string(),
            backend,
            build,
            model,
            precision,
            kernels,
            configuration,
            provenance_sha256: prefixed_sha256(&[]),
        };
        provenance.validate_fields()?;
        provenance.provenance_sha256 = provenance.compute_sha256()?;
        Ok(provenance)
    }

    pub fn validate(&self) -> Result<(), ExecutionProvenanceError> {
        self.validate_fields()?;
        validate_sha256("execution provenance_sha256", &self.provenance_sha256)?;
        let expected = self.compute_sha256()?;
        if self.provenance_sha256 != expected {
            return Err(ExecutionProvenanceError::new(format!(
                "execution provenance digest mismatch: manifest has {}, expected {expected}",
                self.provenance_sha256
            )));
        }
        Ok(())
    }

    fn validate_fields(&self) -> Result<(), ExecutionProvenanceError> {
        if self.schema_version != EXECUTION_PROVENANCE_SCHEMA_VERSION {
            return Err(ExecutionProvenanceError::new(format!(
                "unsupported execution provenance schema_version {}; expected {}",
                self.schema_version, EXECUTION_PROVENANCE_SCHEMA_VERSION
            )));
        }
        if self.provenance_type != EXECUTION_PROVENANCE_TYPE {
            return Err(ExecutionProvenanceError::new(format!(
                "invalid execution provenance_type {:?}",
                self.provenance_type
            )));
        }
        for (label, value) in [
            ("backend.name", self.backend.name.as_str()),
            ("backend.device", self.backend.device.as_str()),
            ("build.package_version", self.build.package_version.as_str()),
            ("build.target", self.build.target.as_str()),
            (
                "precision.inference_dtype",
                self.precision.inference_dtype.as_str(),
            ),
            (
                "precision.training_policy",
                self.precision.training_policy.as_str(),
            ),
        ] {
            validate_text(label, value)?;
        }
        if let Some(commit) = self.build.git_commit.as_deref() {
            validate_text("build.git_commit", commit)?;
        }
        for (label, value) in [
            (
                "backend.numerical_runtime_sha256",
                self.backend.numerical_runtime_sha256.as_str(),
            ),
            (
                "build.executable_sha256",
                self.build.executable_sha256.as_str(),
            ),
            (
                "model.model_config_sha256",
                self.model.model_config_sha256.as_str(),
            ),
            (
                "model.tokenizer_vocab_sha256",
                self.model.tokenizer_vocab_sha256.as_str(),
            ),
            (
                "model.tokenizer_config_sha256",
                self.model.tokenizer_config_sha256.as_str(),
            ),
            (
                "configuration.effective_server_config_sha256",
                self.configuration.effective_server_config_sha256.as_str(),
            ),
            (
                "configuration.effective_environment_sha256",
                self.configuration.effective_environment_sha256.as_str(),
            ),
        ] {
            validate_sha256(label, value)?;
        }
        if let Some(hash) = self.build.source_tree_sha256.as_deref() {
            validate_sha256("build.source_tree_sha256", hash)?;
        }
        if let Some(hash) = self.model.chat_template_sha256.as_deref() {
            validate_sha256("model.chat_template_sha256", hash)?;
        }
        self.kernels.validate()?;
        Ok(())
    }

    fn compute_sha256(&self) -> Result<String, ExecutionProvenanceError> {
        #[derive(Serialize)]
        struct IdentityFields<'a> {
            schema_version: u32,
            provenance_type: &'a str,
            backend: &'a ExecutionBackendIdentity,
            build: &'a ExecutionBuildIdentity,
            model: &'a ExecutionModelIdentity,
            precision: &'a ExecutionPrecisionIdentity,
            kernels: &'a ExecutionKernelIdentity,
            configuration: &'a ExecutionConfigurationIdentity,
        }
        let bytes = serde_json::to_vec(&IdentityFields {
            schema_version: self.schema_version,
            provenance_type: &self.provenance_type,
            backend: &self.backend,
            build: &self.build,
            model: &self.model,
            precision: &self.precision,
            kernels: &self.kernels,
            configuration: &self.configuration,
        })
        .map_err(|error| {
            ExecutionProvenanceError::new(format!("serialize execution provenance: {error}"))
        })?;
        Ok(prefixed_sha256(&bytes))
    }
}

fn validate_kernel_fields(
    versions: &BTreeMap<String, String>,
    features: &[String],
) -> Result<(), ExecutionProvenanceError> {
    if versions.is_empty() || versions.len() > MAX_KERNEL_ENTRIES {
        return Err(ExecutionProvenanceError::new(format!(
            "execution kernel versions must contain 1..={MAX_KERNEL_ENTRIES} entries"
        )));
    }
    if features.len() > MAX_FEATURES {
        return Err(ExecutionProvenanceError::new(format!(
            "execution compiled_features exceeds the {MAX_FEATURES}-entry limit"
        )));
    }
    for (name, version) in versions {
        validate_text("kernel version name", name)?;
        validate_text("kernel version value", version)?;
    }
    for feature in features {
        validate_text("compiled feature", feature)?;
    }
    Ok(())
}

fn kernel_contract_sha256(
    versions: &BTreeMap<String, String>,
    features: &[String],
) -> Result<String, ExecutionProvenanceError> {
    let bytes =
        serde_json::to_vec(&(KERNEL_CONTRACT_TYPE, versions, features)).map_err(|error| {
            ExecutionProvenanceError::new(format!("serialize kernel contract: {error}"))
        })?;
    Ok(prefixed_sha256(&bytes))
}

fn validate_text(label: &str, value: &str) -> Result<(), ExecutionProvenanceError> {
    if value.is_empty()
        || value.trim() != value
        || value.len() > MAX_TEXT_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(ExecutionProvenanceError::new(format!(
            "execution {label} must be non-empty, trimmed, bounded text"
        )));
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), ExecutionProvenanceError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(ExecutionProvenanceError::new(format!(
            "execution {label} must use sha256:<64 lowercase hex>"
        )));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ExecutionProvenanceError::new(format!(
            "execution {label} must use sha256:<64 lowercase hex>"
        )));
    }
    Ok(())
}

fn is_strictly_sorted_unique(values: &[String]) -> bool {
    values
        .windows(2)
        .all(|pair| pair[0].as_str() < pair[1].as_str())
}

fn prefixed_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn provenance() -> ExecutionProvenanceV1 {
        ExecutionProvenanceV1::new(
            ExecutionBackendIdentity {
                name: "rocm".into(),
                device: "rocm:0".into(),
                numerical_runtime_sha256: hash('1'),
            },
            ExecutionBuildIdentity {
                package_version: "0.4.1".into(),
                target: "linux-x86_64".into(),
                executable_sha256: hash('2'),
                git_commit: Some("abc123".into()),
                source_tree_sha256: Some(hash('3')),
                source_dirty: Some(false),
            },
            ExecutionModelIdentity {
                model_config_sha256: hash('4'),
                tokenizer_vocab_sha256: hash('5'),
                tokenizer_config_sha256: hash('6'),
                chat_template_sha256: Some(hash('7')),
            },
            ExecutionPrecisionIdentity {
                inference_dtype: "bf16".into(),
                training_policy: "rocm-native".into(),
            },
            ExecutionKernelIdentity::new(
                BTreeMap::from([("kiln-model".into(), "0.4.1".into())]),
                vec!["rocm".into(), "gfx1151".into()],
            )
            .unwrap(),
            ExecutionConfigurationIdentity {
                effective_server_config_sha256: hash('8'),
                effective_environment_sha256: hash('9'),
            },
        )
        .unwrap()
    }

    #[test]
    fn execution_provenance_is_canonical_and_self_verifying() {
        let provenance = provenance();
        provenance.validate().unwrap();
        let encoded = serde_json::to_vec(&provenance).unwrap();
        let decoded: ExecutionProvenanceV1 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, provenance);
        assert!(provenance.provenance_sha256.starts_with("sha256:"));
        assert_eq!(
            provenance.kernels.compiled_features,
            vec!["gfx1151", "rocm"]
        );
    }

    #[test]
    fn execution_provenance_rejects_tampering_and_unknown_fields() {
        let mut changed_device = provenance();
        changed_device.backend.device = "rocm:1".into();
        assert!(changed_device.validate().is_err());

        let mut changed_kernels = provenance();
        changed_kernels
            .kernels
            .versions
            .insert("extra".into(), "1".into());
        assert!(changed_kernels.validate().is_err());

        let encoded = serde_json::to_string(&provenance()).unwrap();
        assert!(
            serde_json::from_str::<ExecutionProvenanceV1>(
                &encoded.replace("\"backend\":", "\"unknown\":true,\"backend\":")
            )
            .is_err()
        );
    }
}
