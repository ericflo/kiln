use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::ModelConfig;
use crate::tokenizer::KilnTokenizer;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigHashes {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_config_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_chat_template_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_config_hash: Option<String>,
    #[serde(
        default,
        alias = "kiln_env_config_hash",
        skip_serializing_if = "Option::is_none"
    )]
    pub effective_config_hash: Option<String>,
}

impl ConfigHashes {
    pub fn from_model_tokenizer(
        model_config: &ModelConfig,
        tokenizer: &KilnTokenizer,
        effective_config_hash: Option<String>,
    ) -> Self {
        Self {
            tokenizer_config_hash: tokenizer.tokenizer_config_sha256().ok(),
            chat_template_hash: tokenizer.chat_template_sha256(),
            training_chat_template_hash: tokenizer.training_chat_template_sha256(),
            model_config_hash: sha256_json_serializable(model_config),
            effective_config_hash,
        }
    }
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("sha256:{}", hex_digest(digest.as_slice()))
}

pub fn sha256_json_value(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    sha256_bytes(&bytes)
}

pub fn sha256_json_serializable<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_value(value)
        .ok()
        .map(|value| sha256_json_value(&value))
}

pub fn effective_config_hash<T: Serialize>(config: &T) -> Option<String> {
    sha256_json_serializable(config)
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_hash_depends_only_on_the_serialized_config() {
        let first = effective_config_hash(&serde_json::json!({"port": 8420}));
        let repeated = effective_config_hash(&serde_json::json!({"port": 8420}));
        let changed = effective_config_hash(&serde_json::json!({"port": 8421}));

        assert_eq!(first, repeated);
        assert_ne!(first, changed);
    }

    #[test]
    fn legacy_environment_hash_field_migrates_to_effective_config_hash() {
        let digest = format!("sha256:{}", "a".repeat(64));
        let hashes: ConfigHashes = serde_json::from_value(serde_json::json!({
            "kiln_env_config_hash": digest,
        }))
        .unwrap();
        let encoded = serde_json::to_value(hashes).unwrap();

        assert_eq!(encoded["effective_config_hash"], digest);
        assert!(encoded.get("kiln_env_config_hash").is_none());
    }
}
