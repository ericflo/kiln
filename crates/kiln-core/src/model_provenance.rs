//! Stable identity for the exact safetensors shards behind a loaded model.

use sha2::{Digest, Sha256};
use thiserror::Error;

use serde::{Deserialize, Serialize};

pub const BASE_WEIGHT_SHARD_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const BASE_WEIGHT_SHARD_MANIFEST_TYPE: &str = "kiln.base-weight-shards.v1";
pub const BASE_WEIGHT_AGGREGATE_ALGORITHM: &str = "kiln.base-model-content.v1";
const MAX_BASE_WEIGHT_SHARDS: usize = 4096;
const MAX_SHARD_FILENAME_BYTES: usize = 255;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct ModelProvenanceError {
    message: String,
}

impl ModelProvenanceError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Exact identity of one logical safetensors shard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BaseWeightShardIdentity {
    /// Path-safe filename from the model's safetensors index or directory.
    pub filename: String,
    /// Exact byte length hashed by the loader.
    pub size_bytes: u64,
    /// Lowercase `sha256:` digest of the complete shard bytes.
    pub sha256: String,
}

impl BaseWeightShardIdentity {
    pub fn new(
        filename: impl Into<String>,
        size_bytes: u64,
        sha256: impl Into<String>,
    ) -> Result<Self, ModelProvenanceError> {
        let identity = Self {
            filename: filename.into(),
            size_bytes,
            sha256: sha256.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn from_digest(
        filename: impl Into<String>,
        size_bytes: u64,
        digest: [u8; 32],
    ) -> Result<Self, ModelProvenanceError> {
        Self::new(filename, size_bytes, prefixed_sha256(&digest))
    }

    pub fn validate(&self) -> Result<(), ModelProvenanceError> {
        validate_shard_filename(&self.filename)?;
        if self.size_bytes == 0 {
            return Err(ModelProvenanceError::new(format!(
                "base-weight shard {:?} must not be empty",
                self.filename
            )));
        }
        decode_prefixed_sha256("base-weight shard sha256", &self.sha256)?;
        Ok(())
    }
}

/// Canonical, portable manifest for all base-weight shard bytes.
///
/// Shards are sorted by filename for stable serialization. The aggregate
/// remains independent of path, filename, and index order so identical weight
/// bytes retain the existing model-content identity after relocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BaseWeightShardManifest {
    pub schema_version: u32,
    pub manifest_type: String,
    pub aggregate_algorithm: String,
    pub aggregate_sha256: String,
    pub total_size_bytes: u64,
    pub shards: Vec<BaseWeightShardIdentity>,
}

impl BaseWeightShardManifest {
    pub fn new(mut shards: Vec<BaseWeightShardIdentity>) -> Result<Self, ModelProvenanceError> {
        shards.sort_by(|left, right| left.filename.cmp(&right.filename));
        validate_shards(&shards)?;
        let total_size_bytes = total_size_bytes(&shards)?;
        let aggregate_sha256 = aggregate_sha256(&shards)?;
        Ok(Self {
            schema_version: BASE_WEIGHT_SHARD_MANIFEST_SCHEMA_VERSION,
            manifest_type: BASE_WEIGHT_SHARD_MANIFEST_TYPE.to_string(),
            aggregate_algorithm: BASE_WEIGHT_AGGREGATE_ALGORITHM.to_string(),
            aggregate_sha256,
            total_size_bytes,
            shards,
        })
    }

    pub fn validate(&self) -> Result<(), ModelProvenanceError> {
        if self.schema_version != BASE_WEIGHT_SHARD_MANIFEST_SCHEMA_VERSION {
            return Err(ModelProvenanceError::new(format!(
                "unsupported base-weight shard manifest schema_version {}; expected {}",
                self.schema_version, BASE_WEIGHT_SHARD_MANIFEST_SCHEMA_VERSION
            )));
        }
        if self.manifest_type != BASE_WEIGHT_SHARD_MANIFEST_TYPE {
            return Err(ModelProvenanceError::new(format!(
                "invalid base-weight shard manifest_type {:?}",
                self.manifest_type
            )));
        }
        if self.aggregate_algorithm != BASE_WEIGHT_AGGREGATE_ALGORITHM {
            return Err(ModelProvenanceError::new(format!(
                "unsupported base-weight aggregate_algorithm {:?}",
                self.aggregate_algorithm
            )));
        }
        validate_shards(&self.shards)?;
        let expected_total = total_size_bytes(&self.shards)?;
        if self.total_size_bytes != expected_total {
            return Err(ModelProvenanceError::new(format!(
                "base-weight total_size_bytes mismatch: manifest has {}, shards total {expected_total}",
                self.total_size_bytes
            )));
        }
        decode_prefixed_sha256("base-weight aggregate_sha256", &self.aggregate_sha256)?;
        let expected_aggregate = aggregate_sha256(&self.shards)?;
        if self.aggregate_sha256 != expected_aggregate {
            return Err(ModelProvenanceError::new(format!(
                "base-weight aggregate_sha256 mismatch: manifest has {}, expected {expected_aggregate}",
                self.aggregate_sha256
            )));
        }
        Ok(())
    }

    /// Whether two valid manifests identify the same multiset of shard bytes.
    ///
    /// Filenames are retained for auditability but deliberately excluded from
    /// content identity so relocating or renaming identical shards does not
    /// invalidate an exact training resume.
    pub fn content_equivalent(&self, other: &Self) -> Result<bool, ModelProvenanceError> {
        self.validate()?;
        other.validate()?;
        Ok(self.aggregate_algorithm == other.aggregate_algorithm
            && self.aggregate_sha256 == other.aggregate_sha256)
    }
}

fn validate_shards(shards: &[BaseWeightShardIdentity]) -> Result<(), ModelProvenanceError> {
    if shards.is_empty() {
        return Err(ModelProvenanceError::new(
            "base-weight shard manifest must contain at least one shard",
        ));
    }
    if shards.len() > MAX_BASE_WEIGHT_SHARDS {
        return Err(ModelProvenanceError::new(format!(
            "base-weight shard manifest exceeds the {MAX_BASE_WEIGHT_SHARDS}-shard safety limit"
        )));
    }

    let mut previous: Option<&str> = None;
    for shard in shards {
        shard.validate()?;
        if previous.is_some_and(|name| name >= shard.filename.as_str()) {
            return Err(ModelProvenanceError::new(format!(
                "base-weight shard filenames must be unique and strictly sorted; {:?} follows {:?}",
                shard.filename, previous
            )));
        }
        previous = Some(&shard.filename);
    }
    Ok(())
}

fn validate_shard_filename(filename: &str) -> Result<(), ModelProvenanceError> {
    let bytes = filename.as_bytes();
    if bytes.is_empty()
        || bytes.len() > MAX_SHARD_FILENAME_BYTES
        || !filename.ends_with(".safetensors")
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(ModelProvenanceError::new(format!(
            "base-weight shard filename must be one portable .safetensors filename: {filename:?}"
        )));
    }
    Ok(())
}

fn total_size_bytes(shards: &[BaseWeightShardIdentity]) -> Result<u64, ModelProvenanceError> {
    shards.iter().try_fold(0u64, |total, shard| {
        total
            .checked_add(shard.size_bytes)
            .ok_or_else(|| ModelProvenanceError::new("base-weight shard byte total exceeds u64"))
    })
}

fn aggregate_sha256(shards: &[BaseWeightShardIdentity]) -> Result<String, ModelProvenanceError> {
    let mut records = shards
        .iter()
        .map(|shard| {
            Ok((
                decode_prefixed_sha256("base-weight shard sha256", &shard.sha256)?,
                shard.size_bytes,
            ))
        })
        .collect::<Result<Vec<_>, ModelProvenanceError>>()?;
    records.sort_unstable();

    let mut hasher = Sha256::new();
    hasher.update(b"kiln.base-model-content.v1\0");
    hasher.update((records.len() as u64).to_le_bytes());
    for (digest, size_bytes) in records {
        hasher.update(size_bytes.to_le_bytes());
        hasher.update(digest);
    }
    Ok(prefixed_sha256(&hasher.finalize().into()))
}

fn decode_prefixed_sha256(label: &str, value: &str) -> Result<[u8; 32], ModelProvenanceError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(ModelProvenanceError::new(format!(
            "{label} must use the sha256:<64 lowercase hex> form"
        )));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ModelProvenanceError::new(format!(
            "{label} must use the sha256:<64 lowercase hex> form"
        )));
    }
    let mut digest = [0u8; 32];
    for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        digest[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(digest)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("validated lowercase hexadecimal digit"),
    }
}

fn prefixed_sha256(digest: &[u8; 32]) -> String {
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

    fn shard(filename: &str, size_bytes: u64, byte: u8) -> BaseWeightShardIdentity {
        BaseWeightShardIdentity::from_digest(filename, size_bytes, [byte; 32]).unwrap()
    }

    #[test]
    fn manifest_is_canonical_and_path_independent() {
        let manifest = BaseWeightShardManifest::new(vec![
            shard("model-00002-of-00002.safetensors", 22, 0x22),
            shard("model-00001-of-00002.safetensors", 11, 0x11),
        ])
        .unwrap();
        assert_eq!(manifest.total_size_bytes, 33);
        assert_eq!(
            manifest
                .shards
                .iter()
                .map(|shard| shard.filename.as_str())
                .collect::<Vec<_>>(),
            vec![
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );

        let renamed = BaseWeightShardManifest::new(vec![
            shard("a.safetensors", 11, 0x11),
            shard("b.safetensors", 22, 0x22),
        ])
        .unwrap();
        assert_eq!(
            manifest.aggregate_sha256,
            "sha256:813979f1d3dd938e49874427651a97344cd1af5409e75aceaa741707616cd7a5"
        );
        assert_eq!(manifest.aggregate_sha256, renamed.aggregate_sha256);
        assert!(manifest.content_equivalent(&renamed).unwrap());
        manifest.validate().unwrap();
    }

    #[test]
    fn content_equivalence_rejects_byte_size_and_multiplicity_changes() {
        let original = BaseWeightShardManifest::new(vec![
            shard("first.safetensors", 11, 0x11),
            shard("second.safetensors", 22, 0x22),
        ])
        .unwrap();
        let changed_bytes = BaseWeightShardManifest::new(vec![
            shard("first.safetensors", 11, 0x11),
            shard("second.safetensors", 22, 0x23),
        ])
        .unwrap();
        let changed_size = BaseWeightShardManifest::new(vec![
            shard("first.safetensors", 12, 0x11),
            shard("second.safetensors", 22, 0x22),
        ])
        .unwrap();
        let duplicate = BaseWeightShardManifest::new(vec![
            shard("first.safetensors", 11, 0x11),
            shard("copy.safetensors", 11, 0x11),
            shard("second.safetensors", 22, 0x22),
        ])
        .unwrap();

        assert!(!original.content_equivalent(&changed_bytes).unwrap());
        assert!(!original.content_equivalent(&changed_size).unwrap());
        assert!(!original.content_equivalent(&duplicate).unwrap());
    }

    #[test]
    fn manifest_validation_rejects_tampering_and_ambiguous_names() {
        assert!(BaseWeightShardManifest::new(Vec::new()).is_err());
        assert!(BaseWeightShardIdentity::from_digest("../model.safetensors", 1, [0; 32]).is_err());
        assert!(
            BaseWeightShardIdentity::from_digest("nested/model.safetensors", 1, [0; 32]).is_err()
        );
        assert!(BaseWeightShardIdentity::from_digest("model.SAFETENSORS", 1, [0; 32]).is_err());
        assert!(BaseWeightShardIdentity::from_digest("model.safetensors", 0, [0; 32]).is_err());

        let mut manifest =
            BaseWeightShardManifest::new(vec![shard("model.safetensors", 1, 0x11)]).unwrap();
        manifest.total_size_bytes = 2;
        assert!(manifest.validate().is_err());
        manifest.total_size_bytes = 1;
        manifest.aggregate_sha256 = format!("sha256:{}", "0".repeat(64));
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn strict_json_round_trip_preserves_manifest() {
        let manifest =
            BaseWeightShardManifest::new(vec![shard("model.safetensors", u64::MAX, 0xff)]).unwrap();
        let encoded = serde_json::to_string(&manifest).unwrap();
        let decoded: BaseWeightShardManifest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, manifest);
        assert!(
            serde_json::from_str::<BaseWeightShardManifest>(
                &encoded.replace("\"shards\":", "\"unknown\":true,\"shards\":")
            )
            .is_err()
        );
    }
}
