use std::collections::BTreeMap;

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
    pub model_config_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kiln_env_config_hash: Option<String>,
}

impl ConfigHashes {
    pub fn from_model_tokenizer(
        model_config: &ModelConfig,
        tokenizer: &KilnTokenizer,
        kiln_env_config_hash: Option<String>,
    ) -> Self {
        Self {
            tokenizer_config_hash: tokenizer.tokenizer_config_sha256().ok(),
            chat_template_hash: tokenizer.chat_template_sha256(),
            model_config_hash: sha256_json_serializable(model_config),
            kiln_env_config_hash,
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

pub fn kiln_env_config_hash<T: Serialize>(config: &T) -> Option<String> {
    let env = kiln_env_vars();
    serde_json::to_value(serde_json::json!({
        "effective_config": config,
        "kiln_env": env,
    }))
    .ok()
    .map(|value| sha256_json_value(&value))
}

pub fn kiln_env_vars() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| key.starts_with("KILN_"))
        .collect()
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
