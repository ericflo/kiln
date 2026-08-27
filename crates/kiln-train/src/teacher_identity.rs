//! Authenticated, canonical identity for a remote prompt-logprob teacher.
//!
//! A served model name is only an alias. It does not identify the tokenizer,
//! base weights, adapter, or inference settings that produced a logprob row.
//! This module defines the bounded, versioned document that Kiln pins at
//! teacher registration and binds to every subsequent scoring response.

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const TEACHER_IDENTITY_SCHEMA_V1: &str = "kiln.teacher-identity.v1";
pub const TEACHER_IDENTITY_PROTOCOL_V1: &str = "vllm.prompt-logprobs.numeric-token-ids.causal.v1";
pub const TEACHER_IDENTITY_LOGPROBS_MODE_V1: &str = "raw_logprobs";
pub const TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1: &str = "kiln-teacher-v1";

pub const MAX_TEACHER_IDENTITY_JSON_BYTES: usize = 4 * 1024;
pub const MAX_TEACHER_IDENTITY_FINGERPRINT_BYTES: usize = 6 * 1024;
pub const MAX_TEACHER_IDENTITY_NAME_BYTES: usize = 256;
pub const MAX_TEACHER_IMPLEMENTATION_BYTES: usize = 256;
pub const MAX_TEACHER_VOCAB_SIZE: u32 = 16_777_216;
pub const MAX_TEACHER_TOP_K: u32 = 65_536;
pub const MAX_TEACHER_MODEL_LEN: u32 = 16_777_216;
pub const MAX_TEACHER_PROMPT_LOGPROB_CANDIDATES: u32 = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TeacherIdentityError {
    #[error("teacher identity JSON is {actual} bytes; maximum is {max}")]
    CanonicalJsonTooLarge { actual: usize, max: usize },
    #[error("teacher identity fingerprint is {actual} bytes; maximum is {max}")]
    FingerprintTooLarge { actual: usize, max: usize },
    #[error("invalid teacher identity field `{field}`: {reason}")]
    InvalidField { field: &'static str, reason: String },
    #[error("invalid teacher identity JSON: {message}")]
    InvalidJson { message: String },
    #[error("teacher identity JSON is not the canonical compact encoding")]
    NonCanonicalJson,
    #[error("invalid teacher identity fingerprint: {reason}")]
    InvalidFingerprint { reason: &'static str },
    #[error("teacher identity fingerprint digest does not match its document")]
    DigestMismatch,
}

/// Identity of a statically loaded adapter served by the teacher.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TeacherAdapterIdentityV1 {
    name: String,
    weights_sha256: String,
    config_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TeacherAdapterIdentityV1Wire {
    name: String,
    weights_sha256: String,
    config_sha256: String,
}

impl TeacherAdapterIdentityV1 {
    pub fn new(
        name: impl Into<String>,
        weights_sha256: impl Into<String>,
        config_sha256: impl Into<String>,
    ) -> Result<Self, TeacherIdentityError> {
        let identity = Self {
            name: name.into(),
            weights_sha256: weights_sha256.into(),
            config_sha256: config_sha256.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    fn validate(&self) -> Result<(), TeacherIdentityError> {
        validate_bounded_text("adapter.name", &self.name, MAX_TEACHER_IDENTITY_NAME_BYTES)?;
        validate_sha256("adapter.weights_sha256", &self.weights_sha256)?;
        validate_sha256("adapter.config_sha256", &self.config_sha256)?;
        Ok(())
    }
}

impl<'de> Deserialize<'de> for TeacherAdapterIdentityV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TeacherAdapterIdentityV1Wire::deserialize(deserializer)?;
        Self::new(wire.name, wire.weights_sha256, wire.config_sha256)
            .map_err(serde::de::Error::custom)
    }
}

/// Complete semantic identity of a remote numeric prompt-logprob source.
///
/// Fields are private so an identity cannot be mutated after validation. The
/// declaration order is the normative canonical JSON key order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TeacherIdentityV1 {
    schema: String,
    protocol: String,
    served_model_id: String,
    base_model_sha256: String,
    tokenizer_vocab_sha256: String,
    tokenizer_config_sha256: String,
    adapter: Option<TeacherAdapterIdentityV1>,
    vocab_size: u32,
    max_top_k: u32,
    max_model_len: u32,
    max_prompt_logprob_candidates: u32,
    logprobs_mode: String,
    implementation: String,
    inference_config_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TeacherIdentityV1Wire {
    schema: String,
    protocol: String,
    served_model_id: String,
    base_model_sha256: String,
    tokenizer_vocab_sha256: String,
    tokenizer_config_sha256: String,
    adapter: Option<TeacherAdapterIdentityV1>,
    vocab_size: u32,
    max_top_k: u32,
    max_model_len: u32,
    max_prompt_logprob_candidates: u32,
    logprobs_mode: String,
    implementation: String,
    inference_config_sha256: String,
}

impl TeacherIdentityV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        served_model_id: impl Into<String>,
        base_model_sha256: impl Into<String>,
        tokenizer_vocab_sha256: impl Into<String>,
        tokenizer_config_sha256: impl Into<String>,
        adapter: Option<TeacherAdapterIdentityV1>,
        vocab_size: u32,
        max_top_k: u32,
        max_model_len: u32,
        max_prompt_logprob_candidates: u32,
        implementation: impl Into<String>,
        inference_config_sha256: impl Into<String>,
    ) -> Result<Self, TeacherIdentityError> {
        let identity = Self {
            schema: TEACHER_IDENTITY_SCHEMA_V1.to_owned(),
            protocol: TEACHER_IDENTITY_PROTOCOL_V1.to_owned(),
            served_model_id: served_model_id.into(),
            base_model_sha256: base_model_sha256.into(),
            tokenizer_vocab_sha256: tokenizer_vocab_sha256.into(),
            tokenizer_config_sha256: tokenizer_config_sha256.into(),
            adapter,
            vocab_size,
            max_top_k,
            max_model_len,
            max_prompt_logprob_candidates,
            logprobs_mode: TEACHER_IDENTITY_LOGPROBS_MODE_V1.to_owned(),
            implementation: implementation.into(),
            inference_config_sha256: inference_config_sha256.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn parse_canonical_json(input: &[u8]) -> Result<Self, TeacherIdentityError> {
        if input.len() > MAX_TEACHER_IDENTITY_JSON_BYTES {
            return Err(TeacherIdentityError::CanonicalJsonTooLarge {
                actual: input.len(),
                max: MAX_TEACHER_IDENTITY_JSON_BYTES,
            });
        }

        let identity: Self =
            serde_json::from_slice(input).map_err(|error| TeacherIdentityError::InvalidJson {
                message: error.to_string(),
            })?;
        let canonical = identity.canonical_json_bytes();
        if canonical.as_slice() != input {
            return Err(TeacherIdentityError::NonCanonicalJson);
        }
        Ok(identity)
    }

    pub fn parse_fingerprint(input: &str) -> Result<Self, TeacherIdentityError> {
        if input.len() > MAX_TEACHER_IDENTITY_FINGERPRINT_BYTES {
            return Err(TeacherIdentityError::FingerprintTooLarge {
                actual: input.len(),
                max: MAX_TEACHER_IDENTITY_FINGERPRINT_BYTES,
            });
        }

        let mut parts = input.split('.');
        let prefix = parts.next().unwrap_or_default();
        let encoded = parts
            .next()
            .ok_or(TeacherIdentityError::InvalidFingerprint {
                reason: "missing canonical document",
            })?;
        let claimed_digest = parts
            .next()
            .ok_or(TeacherIdentityError::InvalidFingerprint {
                reason: "missing digest",
            })?;
        if parts.next().is_some() {
            return Err(TeacherIdentityError::InvalidFingerprint {
                reason: "expected exactly three dot-separated components",
            });
        }
        if prefix != TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1 {
            return Err(TeacherIdentityError::InvalidFingerprint {
                reason: "unsupported fingerprint prefix",
            });
        }
        if !is_lower_sha256(claimed_digest) {
            return Err(TeacherIdentityError::InvalidFingerprint {
                reason: "digest must be exactly 64 lowercase hexadecimal characters",
            });
        }

        let canonical = decode_base64url_no_pad(encoded)?;
        let actual_digest = sha256_hex(&canonical);
        if actual_digest != claimed_digest {
            return Err(TeacherIdentityError::DigestMismatch);
        }
        Self::parse_canonical_json(&canonical)
    }

    pub fn canonical_json(&self) -> String {
        // All fields are validated UTF-8 values and serde_json's in-memory
        // serializer cannot fail for this fixed data model.
        serde_json::to_string(self).expect("validated teacher identity serializes")
    }

    pub fn canonical_json_bytes(&self) -> Vec<u8> {
        self.canonical_json().into_bytes()
    }

    pub fn fingerprint(&self) -> String {
        let canonical = self.canonical_json_bytes();
        format!(
            "{}.{}.{}",
            TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1,
            encode_base64url_no_pad(&canonical),
            sha256_hex(&canonical)
        )
    }

    /// SHA-256 of the complete canonical identity document.
    pub fn identity_digest_sha256(&self) -> String {
        sha256_hex(&self.canonical_json_bytes())
    }

    /// Alias for [`Self::identity_digest_sha256`] used at protocol boundaries.
    pub fn identity_digest(&self) -> String {
        self.identity_digest_sha256()
    }

    /// Compact revision used to bind requests, responses, receipts, and cache
    /// entries. It changes whenever any semantic identity field changes.
    pub fn content_revision(&self) -> String {
        self.identity_digest_sha256()
    }

    /// Derive the identity of the same loaded base/runtime with one static
    /// adapter. The base identity remains immutable; only the adapter tuple is
    /// replaced, so callers cannot accidentally drop a tokenizer, runtime, or
    /// inference-contract field while publishing a loaded LoRA revision.
    pub fn with_static_adapter(
        &self,
        adapter: TeacherAdapterIdentityV1,
    ) -> Result<Self, TeacherIdentityError> {
        Self::new(
            self.served_model_id.clone(),
            self.base_model_sha256.clone(),
            self.tokenizer_vocab_sha256.clone(),
            self.tokenizer_config_sha256.clone(),
            Some(adapter),
            self.vocab_size,
            self.max_top_k,
            self.max_model_len,
            self.max_prompt_logprob_candidates,
            self.implementation.clone(),
            self.inference_config_sha256.clone(),
        )
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn protocol(&self) -> &str {
        &self.protocol
    }

    pub fn served_model_id(&self) -> &str {
        &self.served_model_id
    }

    pub fn base_model_sha256(&self) -> &str {
        &self.base_model_sha256
    }

    pub fn tokenizer_vocab_sha256(&self) -> &str {
        &self.tokenizer_vocab_sha256
    }

    pub fn tokenizer_config_sha256(&self) -> &str {
        &self.tokenizer_config_sha256
    }

    pub fn adapter(&self) -> Option<&TeacherAdapterIdentityV1> {
        self.adapter.as_ref()
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn max_top_k(&self) -> u32 {
        self.max_top_k
    }

    pub fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    pub fn max_prompt_logprob_candidates(&self) -> u32 {
        self.max_prompt_logprob_candidates
    }

    pub fn logprobs_mode(&self) -> &str {
        &self.logprobs_mode
    }

    pub fn implementation(&self) -> &str {
        &self.implementation
    }

    pub fn inference_config_sha256(&self) -> &str {
        &self.inference_config_sha256
    }

    fn validate(&self) -> Result<(), TeacherIdentityError> {
        validate_exact("schema", &self.schema, TEACHER_IDENTITY_SCHEMA_V1)?;
        validate_exact("protocol", &self.protocol, TEACHER_IDENTITY_PROTOCOL_V1)?;
        validate_bounded_text(
            "served_model_id",
            &self.served_model_id,
            MAX_TEACHER_IDENTITY_NAME_BYTES,
        )?;
        validate_sha256("base_model_sha256", &self.base_model_sha256)?;
        validate_sha256("tokenizer_vocab_sha256", &self.tokenizer_vocab_sha256)?;
        validate_sha256("tokenizer_config_sha256", &self.tokenizer_config_sha256)?;
        if let Some(adapter) = &self.adapter {
            adapter.validate()?;
        }
        validate_range("vocab_size", self.vocab_size, 1, MAX_TEACHER_VOCAB_SIZE)?;
        validate_range("max_top_k", self.max_top_k, 1, MAX_TEACHER_TOP_K)?;
        if self.max_top_k > self.vocab_size {
            return Err(invalid_field("max_top_k", "must not exceed vocab_size"));
        }
        validate_range(
            "max_model_len",
            self.max_model_len,
            1,
            MAX_TEACHER_MODEL_LEN,
        )?;
        validate_range(
            "max_prompt_logprob_candidates",
            self.max_prompt_logprob_candidates,
            1,
            MAX_TEACHER_PROMPT_LOGPROB_CANDIDATES,
        )?;
        let one_row_max = self.max_top_k.saturating_add(1).min(self.vocab_size);
        if self.max_prompt_logprob_candidates < one_row_max {
            return Err(invalid_field(
                "max_prompt_logprob_candidates",
                format!("must fit one maximum-K response row ({one_row_max} candidates)"),
            ));
        }
        validate_exact(
            "logprobs_mode",
            &self.logprobs_mode,
            TEACHER_IDENTITY_LOGPROBS_MODE_V1,
        )?;
        validate_bounded_text(
            "implementation",
            &self.implementation,
            MAX_TEACHER_IMPLEMENTATION_BYTES,
        )?;
        validate_sha256("inference_config_sha256", &self.inference_config_sha256)?;

        let canonical_len = serde_json::to_vec(self)
            .expect("validated teacher identity serializes")
            .len();
        if canonical_len > MAX_TEACHER_IDENTITY_JSON_BYTES {
            return Err(TeacherIdentityError::CanonicalJsonTooLarge {
                actual: canonical_len,
                max: MAX_TEACHER_IDENTITY_JSON_BYTES,
            });
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for TeacherIdentityV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TeacherIdentityV1Wire::deserialize(deserializer)?;
        let identity = Self {
            schema: wire.schema,
            protocol: wire.protocol,
            served_model_id: wire.served_model_id,
            base_model_sha256: wire.base_model_sha256,
            tokenizer_vocab_sha256: wire.tokenizer_vocab_sha256,
            tokenizer_config_sha256: wire.tokenizer_config_sha256,
            adapter: wire.adapter,
            vocab_size: wire.vocab_size,
            max_top_k: wire.max_top_k,
            max_model_len: wire.max_model_len,
            max_prompt_logprob_candidates: wire.max_prompt_logprob_candidates,
            logprobs_mode: wire.logprobs_mode,
            implementation: wire.implementation,
            inference_config_sha256: wire.inference_config_sha256,
        };
        identity.validate().map_err(serde::de::Error::custom)?;
        Ok(identity)
    }
}

fn validate_exact(
    field: &'static str,
    actual: &str,
    expected: &'static str,
) -> Result<(), TeacherIdentityError> {
    if actual != expected {
        return Err(invalid_field(
            field,
            format!("must be exactly `{expected}`"),
        ));
    }
    Ok(())
}

fn validate_bounded_text(
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), TeacherIdentityError> {
    if value.is_empty() {
        return Err(invalid_field(field, "must not be empty"));
    }
    if value.len() > max_bytes {
        return Err(invalid_field(
            field,
            format!("is {} bytes; maximum is {max_bytes}", value.len()),
        ));
    }
    if value.trim() != value {
        return Err(invalid_field(
            field,
            "must not have leading or trailing whitespace",
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(invalid_field(field, "must not contain control characters"));
    }
    Ok(())
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), TeacherIdentityError> {
    if !is_lower_sha256(value) {
        return Err(invalid_field(
            field,
            "must be exactly 64 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

/// SHA-256 digest shape check: exactly 64 lowercase hexadecimal characters.
/// Shared with the logit-cache module for prefix-digest validation.
pub(crate) fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_range(
    field: &'static str,
    value: u32,
    min: u32,
    max: u32,
) -> Result<(), TeacherIdentityError> {
    if !(min..=max).contains(&value) {
        return Err(invalid_field(
            field,
            format!("must be between {min} and {max}, inclusive"),
        ));
    }
    Ok(())
}

fn invalid_field(field: &'static str, reason: impl Into<String>) -> TeacherIdentityError {
    TeacherIdentityError::InvalidField {
        field,
        reason: reason.into(),
    }
}

fn sha256_hex(input: &[u8]) -> String {
    let digest = Sha256::digest(input);
    let mut output = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn encode_base64url_no_pad(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut output = String::with_capacity((input.len() * 4).div_ceil(3));
    let mut chunks = input.chunks_exact(3);
    for chunk in &mut chunks {
        output.push(ALPHABET[(chunk[0] >> 2) as usize] as char);
        output.push(ALPHABET[(((chunk[0] & 0x03) << 4) | (chunk[1] >> 4)) as usize] as char);
        output.push(ALPHABET[(((chunk[1] & 0x0f) << 2) | (chunk[2] >> 6)) as usize] as char);
        output.push(ALPHABET[(chunk[2] & 0x3f) as usize] as char);
    }
    match chunks.remainder() {
        [] => {}
        [first] => {
            output.push(ALPHABET[(first >> 2) as usize] as char);
            output.push(ALPHABET[((first & 0x03) << 4) as usize] as char);
        }
        [first, second] => {
            output.push(ALPHABET[(first >> 2) as usize] as char);
            output.push(ALPHABET[(((first & 0x03) << 4) | (second >> 4)) as usize] as char);
            output.push(ALPHABET[((second & 0x0f) << 2) as usize] as char);
        }
        _ => unreachable!("chunks_exact remainder is shorter than three bytes"),
    }
    output
}

fn decode_base64url_no_pad(input: &str) -> Result<Vec<u8>, TeacherIdentityError> {
    if input.is_empty() {
        return Err(TeacherIdentityError::InvalidFingerprint {
            reason: "canonical document is empty",
        });
    }
    if input.len() % 4 == 1 {
        return Err(TeacherIdentityError::InvalidFingerprint {
            reason: "invalid base64url length",
        });
    }
    let decoded_len = (input.len() / 4)
        .checked_mul(3)
        .and_then(|len| {
            len.checked_add(match input.len() % 4 {
                0 => 0,
                2 => 1,
                3 => 2,
                _ => unreachable!(),
            })
        })
        .ok_or(TeacherIdentityError::InvalidFingerprint {
            reason: "base64url length overflow",
        })?;
    if decoded_len > MAX_TEACHER_IDENTITY_JSON_BYTES {
        return Err(TeacherIdentityError::CanonicalJsonTooLarge {
            actual: decoded_len,
            max: MAX_TEACHER_IDENTITY_JSON_BYTES,
        });
    }

    let encoded = input.as_bytes();
    let mut output = Vec::with_capacity(decoded_len);
    let full_len = encoded.len() / 4 * 4;
    for chunk in encoded[..full_len].chunks_exact(4) {
        let a = decode_base64url_char(chunk[0])?;
        let b = decode_base64url_char(chunk[1])?;
        let c = decode_base64url_char(chunk[2])?;
        let d = decode_base64url_char(chunk[3])?;
        output.push((a << 2) | (b >> 4));
        output.push((b << 4) | (c >> 2));
        output.push((c << 6) | d);
    }

    match &encoded[full_len..] {
        [] => {}
        [a, b] => {
            let a = decode_base64url_char(*a)?;
            let b = decode_base64url_char(*b)?;
            if b & 0x0f != 0 {
                return Err(TeacherIdentityError::InvalidFingerprint {
                    reason: "non-canonical base64url trailing bits",
                });
            }
            output.push((a << 2) | (b >> 4));
        }
        [a, b, c] => {
            let a = decode_base64url_char(*a)?;
            let b = decode_base64url_char(*b)?;
            let c = decode_base64url_char(*c)?;
            if c & 0x03 != 0 {
                return Err(TeacherIdentityError::InvalidFingerprint {
                    reason: "non-canonical base64url trailing bits",
                });
            }
            output.push((a << 2) | (b >> 4));
            output.push((b << 4) | (c >> 2));
        }
        _ => unreachable!("base64url remainder cannot have one byte"),
    }
    debug_assert_eq!(output.len(), decoded_len);
    Ok(output)
}

fn decode_base64url_char(byte: u8) -> Result<u8, TeacherIdentityError> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'-' => Ok(62),
        b'_' => Ok(63),
        _ => Err(TeacherIdentityError::InvalidFingerprint {
            reason: "canonical document must use unpadded URL-safe base64",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn identity(adapter: bool) -> TeacherIdentityV1 {
        TeacherIdentityV1::new(
            "teacher/model",
            A,
            B,
            C,
            adapter.then(|| TeacherAdapterIdentityV1::new("math-lora", D, E).unwrap()),
            248_320,
            20,
            262_144,
            1_000_000,
            "vllm/0.10.2+kiln-identity/1",
            F,
        )
        .unwrap()
    }

    fn fingerprint_for_json_with_digest(json: &str, digest: &str) -> String {
        format!(
            "{TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1}.{}.{digest}",
            encode_base64url_no_pad(json.as_bytes())
        )
    }

    #[test]
    fn canonical_json_has_exact_schema_order_and_null_adapter() {
        let actual = identity(false).canonical_json();
        let expected = format!(
            concat!(
                "{{\"schema\":\"kiln.teacher-identity.v1\",",
                "\"protocol\":\"vllm.prompt-logprobs.numeric-token-ids.causal.v1\",",
                "\"served_model_id\":\"teacher/model\",",
                "\"base_model_sha256\":\"{}\",",
                "\"tokenizer_vocab_sha256\":\"{}\",",
                "\"tokenizer_config_sha256\":\"{}\",",
                "\"adapter\":null,",
                "\"vocab_size\":248320,\"max_top_k\":20,\"max_model_len\":262144,",
                "\"max_prompt_logprob_candidates\":1000000,",
                "\"logprobs_mode\":\"raw_logprobs\",",
                "\"implementation\":\"vllm/0.10.2+kiln-identity/1\",",
                "\"inference_config_sha256\":\"{}\"}}"
            ),
            A, B, C, F
        );
        assert_eq!(actual, expected);
        assert_eq!(
            TeacherIdentityV1::parse_canonical_json(actual.as_bytes()).unwrap(),
            identity(false)
        );
    }

    #[test]
    fn adapter_and_identity_round_trip_through_fingerprint() {
        for expected in [identity(false), identity(true)] {
            let fingerprint = expected.fingerprint();
            assert!(!fingerprint.contains('='));
            assert!(fingerprint.len() <= MAX_TEACHER_IDENTITY_FINGERPRINT_BYTES);
            let parsed = TeacherIdentityV1::parse_fingerprint(&fingerprint).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.identity_digest(), parsed.identity_digest_sha256());
            assert_eq!(parsed.content_revision(), parsed.identity_digest_sha256());
            assert_eq!(parsed.content_revision().len(), 64);
        }

        let with_adapter = identity(true);
        let adapter = with_adapter.adapter().unwrap();
        assert_eq!(adapter.name(), "math-lora");
        assert_eq!(adapter.weights_sha256(), D);
        assert_eq!(adapter.config_sha256(), E);
    }

    #[test]
    fn static_adapter_derivation_preserves_every_base_runtime_field() {
        let base = identity(false);
        let adapter = TeacherAdapterIdentityV1::new("math-lora", D, E).unwrap();
        let derived = base.with_static_adapter(adapter.clone()).unwrap();

        assert_eq!(derived.adapter(), Some(&adapter));
        assert_eq!(derived.served_model_id(), base.served_model_id());
        assert_eq!(derived.base_model_sha256(), base.base_model_sha256());
        assert_eq!(
            derived.tokenizer_vocab_sha256(),
            base.tokenizer_vocab_sha256()
        );
        assert_eq!(
            derived.tokenizer_config_sha256(),
            base.tokenizer_config_sha256()
        );
        assert_eq!(derived.vocab_size(), base.vocab_size());
        assert_eq!(derived.max_top_k(), base.max_top_k());
        assert_eq!(derived.max_model_len(), base.max_model_len());
        assert_eq!(
            derived.max_prompt_logprob_candidates(),
            base.max_prompt_logprob_candidates()
        );
        assert_eq!(derived.implementation(), base.implementation());
        assert_eq!(
            derived.inference_config_sha256(),
            base.inference_config_sha256()
        );
        assert_ne!(derived.content_revision(), base.content_revision());
    }

    #[test]
    fn all_accessors_report_the_authenticated_values() {
        let value = identity(false);
        assert_eq!(value.schema(), TEACHER_IDENTITY_SCHEMA_V1);
        assert_eq!(value.protocol(), TEACHER_IDENTITY_PROTOCOL_V1);
        assert_eq!(value.served_model_id(), "teacher/model");
        assert_eq!(value.base_model_sha256(), A);
        assert_eq!(value.tokenizer_vocab_sha256(), B);
        assert_eq!(value.tokenizer_config_sha256(), C);
        assert!(value.adapter().is_none());
        assert_eq!(value.vocab_size(), 248_320);
        assert_eq!(value.max_top_k(), 20);
        assert_eq!(value.max_model_len(), 262_144);
        assert_eq!(value.max_prompt_logprob_candidates(), 1_000_000);
        assert_eq!(value.logprobs_mode(), TEACHER_IDENTITY_LOGPROBS_MODE_V1);
        assert_eq!(value.implementation(), "vllm/0.10.2+kiln-identity/1");
        assert_eq!(value.inference_config_sha256(), F);
    }

    #[test]
    fn digest_rejects_mutation_of_every_identity_field() {
        let original = identity(true);
        let json = original.canonical_json();
        let digest = original.identity_digest_sha256();
        let replacements = [
            (
                "\"schema\":\"kiln.teacher-identity.v1\"",
                "\"schema\":\"kiln.teacher-identity.v2\"",
            ),
            (
                "\"protocol\":\"vllm.prompt-logprobs.numeric-token-ids.causal.v1\"",
                "\"protocol\":\"vllm.prompt-logprobs.numeric-token-ids.causal.v2\"",
            ),
            (
                "\"served_model_id\":\"teacher/model\"",
                "\"served_model_id\":\"teacher/other\"",
            ),
            (
                &format!("\"base_model_sha256\":\"{A}\""),
                &format!("\"base_model_sha256\":\"{F}\""),
            ),
            (
                &format!("\"tokenizer_vocab_sha256\":\"{B}\""),
                &format!("\"tokenizer_vocab_sha256\":\"{F}\""),
            ),
            (
                &format!("\"tokenizer_config_sha256\":\"{C}\""),
                &format!("\"tokenizer_config_sha256\":\"{F}\""),
            ),
            (
                "\"adapter\":{\"name\":\"math-lora\"",
                "\"adapter\":{\"name\":\"other-lora\"",
            ),
            ("\"vocab_size\":248320", "\"vocab_size\":248321"),
            ("\"max_top_k\":20", "\"max_top_k\":21"),
            ("\"max_model_len\":262144", "\"max_model_len\":262145"),
            (
                "\"max_prompt_logprob_candidates\":1000000",
                "\"max_prompt_logprob_candidates\":1000001",
            ),
            (
                "\"logprobs_mode\":\"raw_logprobs\"",
                "\"logprobs_mode\":\"normalized\"",
            ),
            (
                "\"implementation\":\"vllm/0.10.2+kiln-identity/1\"",
                "\"implementation\":\"vllm/0.10.3+kiln-identity/1\"",
            ),
            (
                &format!("\"inference_config_sha256\":\"{F}\""),
                &format!("\"inference_config_sha256\":\"{A}\""),
            ),
        ];

        for (needle, replacement) in replacements {
            let mutated = json.replacen(needle, replacement, 1);
            assert_ne!(mutated, json, "test replacement must change the document");
            let error = TeacherIdentityV1::parse_fingerprint(&fingerprint_for_json_with_digest(
                &mutated, &digest,
            ))
            .unwrap_err();
            assert_eq!(error, TeacherIdentityError::DigestMismatch, "{needle}");
        }
    }

    #[test]
    fn digest_rejects_mutation_of_each_adapter_revision_field() {
        let original = identity(true);
        let json = original.canonical_json();
        let digest = original.identity_digest_sha256();
        for (field, before, after) in [("weights_sha256", D, A), ("config_sha256", E, A)] {
            let needle = format!("\"{field}\":\"{before}\"");
            let replacement = format!("\"{field}\":\"{after}\"");
            let mutated = json.replacen(&needle, &replacement, 1);
            assert_eq!(
                TeacherIdentityV1::parse_fingerprint(&fingerprint_for_json_with_digest(
                    &mutated, &digest,
                ))
                .unwrap_err(),
                TeacherIdentityError::DigestMismatch
            );
        }
    }

    #[test]
    fn validates_every_fixed_and_sha_field() {
        let canonical = identity(true).canonical_json();
        let invalid_replacements = [
            ("kiln.teacher-identity.v1", "kiln.teacher-identity.v2"),
            (
                "vllm.prompt-logprobs.numeric-token-ids.causal.v1",
                "vllm.prompt-logprobs.numeric-token-ids.causal.v2",
            ),
            (
                A,
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            ),
            (
                B,
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
            (
                C,
                "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
            ),
            (
                D,
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddD",
            ),
            (
                E,
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            ),
            ("raw_logprobs", "normalized_logprobs"),
            (
                F,
                "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffF",
            ),
        ];
        for (before, after) in invalid_replacements {
            let mutated = canonical.replacen(before, after, 1);
            assert!(
                matches!(
                    TeacherIdentityV1::parse_canonical_json(mutated.as_bytes()),
                    Err(TeacherIdentityError::InvalidJson { .. })
                ),
                "replacement {before:?} -> {after:?} must be rejected"
            );
        }
    }

    #[test]
    fn validates_text_and_numeric_bounds() {
        assert!(TeacherIdentityV1::new("", A, B, C, None, 10, 1, 1, 10, "impl", F).is_err());
        assert!(TeacherIdentityV1::new(" model", A, B, C, None, 10, 1, 1, 10, "impl", F).is_err());
        assert!(
            TeacherIdentityV1::new("model\nname", A, B, C, None, 10, 1, 1, 10, "impl", F).is_err()
        );
        assert!(
            TeacherIdentityV1::new(
                "x".repeat(MAX_TEACHER_IDENTITY_NAME_BYTES + 1),
                A,
                B,
                C,
                None,
                10,
                1,
                1,
                10,
                "impl",
                F,
            )
            .is_err()
        );
        assert!(TeacherIdentityV1::new("model", A, B, C, None, 10, 1, 1, 10, "", F).is_err());
        assert!(
            TeacherIdentityV1::new(
                "model",
                A,
                B,
                C,
                None,
                10,
                1,
                1,
                10,
                "x".repeat(MAX_TEACHER_IMPLEMENTATION_BYTES + 1),
                F,
            )
            .is_err()
        );

        for (vocab, top_k, model_len, candidates) in [
            (0, 1, 1, 1),
            (MAX_TEACHER_VOCAB_SIZE + 1, 1, 1, 1),
            (10, 0, 1, 1),
            (10, 11, 1, 1),
            (MAX_TEACHER_TOP_K + 1, MAX_TEACHER_TOP_K + 1, 1, 1),
            (10, 1, 0, 1),
            (10, 1, MAX_TEACHER_MODEL_LEN + 1, 1),
            (10, 1, 1, 0),
            (10, 1, 1, MAX_TEACHER_PROMPT_LOGPROB_CANDIDATES + 1),
        ] {
            assert!(
                TeacherIdentityV1::new(
                    "model", A, B, C, None, vocab, top_k, model_len, candidates, "impl", F,
                )
                .is_err(),
                "must reject vocab={vocab}, top_k={top_k}, model_len={model_len}, candidates={candidates}"
            );
        }
    }

    #[test]
    fn validates_adapter_fields_and_unknown_or_duplicate_keys() {
        assert!(TeacherAdapterIdentityV1::new("", D, E).is_err());
        assert!(TeacherAdapterIdentityV1::new(" adapter", D, E).is_err());
        assert!(TeacherAdapterIdentityV1::new("adapter", A.to_uppercase(), E).is_err());
        assert!(TeacherAdapterIdentityV1::new("adapter", D, "short").is_err());

        let json = identity(true).canonical_json();
        let duplicate = json.replacen("{", "{\"schema\":\"kiln.teacher-identity.v1\",", 1);
        assert!(matches!(
            TeacherIdentityV1::parse_canonical_json(duplicate.as_bytes()),
            Err(TeacherIdentityError::InvalidJson { .. })
        ));

        let unknown = json.replacen(
            "\"inference_config_sha256\"",
            "\"unknown\":true,\"inference_config_sha256\"",
            1,
        );
        assert!(matches!(
            TeacherIdentityV1::parse_canonical_json(unknown.as_bytes()),
            Err(TeacherIdentityError::InvalidJson { .. })
        ));

        let adapter_unknown = json.replacen(
            "\"weights_sha256\"",
            "\"unknown\":true,\"weights_sha256\"",
            1,
        );
        assert!(matches!(
            TeacherIdentityV1::parse_canonical_json(adapter_unknown.as_bytes()),
            Err(TeacherIdentityError::InvalidJson { .. })
        ));

        let adapter_duplicate = json.replacen(
            "\"name\":\"math-lora\"",
            "\"name\":\"math-lora\",\"name\":\"math-lora\"",
            1,
        );
        assert!(matches!(
            TeacherIdentityV1::parse_canonical_json(adapter_duplicate.as_bytes()),
            Err(TeacherIdentityError::InvalidJson { .. })
        ));
    }

    #[test]
    fn canonical_parser_rejects_malformed_and_noncanonical_json() {
        for malformed in [b"".as_slice(), b"{".as_slice(), b"[]".as_slice()] {
            assert!(matches!(
                TeacherIdentityV1::parse_canonical_json(malformed),
                Err(TeacherIdentityError::InvalidJson { .. })
            ));
        }

        let json = identity(false).canonical_json();
        let whitespace = format!(" {json}");
        assert_eq!(
            TeacherIdentityV1::parse_canonical_json(whitespace.as_bytes()).unwrap_err(),
            TeacherIdentityError::NonCanonicalJson
        );
        let escaped = json.replacen("teacher/model", "teacher\\u002fmodel", 1);
        assert_eq!(
            TeacherIdentityV1::parse_canonical_json(escaped.as_bytes()).unwrap_err(),
            TeacherIdentityError::NonCanonicalJson
        );
        let reordered = json.replacen(
            concat!(
                "\"schema\":\"kiln.teacher-identity.v1\",",
                "\"protocol\":\"vllm.prompt-logprobs.numeric-token-ids.causal.v1\""
            ),
            concat!(
                "\"protocol\":\"vllm.prompt-logprobs.numeric-token-ids.causal.v1\",",
                "\"schema\":\"kiln.teacher-identity.v1\""
            ),
            1,
        );
        assert_eq!(
            TeacherIdentityV1::parse_canonical_json(reordered.as_bytes()).unwrap_err(),
            TeacherIdentityError::NonCanonicalJson
        );
        let missing_nullable_adapter = json.replacen("\"adapter\":null,", "", 1);
        assert_eq!(
            TeacherIdentityV1::parse_canonical_json(missing_nullable_adapter.as_bytes())
                .unwrap_err(),
            TeacherIdentityError::NonCanonicalJson
        );

        let oversized = vec![b' '; MAX_TEACHER_IDENTITY_JSON_BYTES + 1];
        assert!(matches!(
            TeacherIdentityV1::parse_canonical_json(&oversized),
            Err(TeacherIdentityError::CanonicalJsonTooLarge { .. })
        ));
    }

    #[test]
    fn fingerprint_parser_is_exact_and_bounded() {
        let valid = identity(false).fingerprint();
        let mut parts = valid.split('.');
        let _prefix = parts.next().unwrap();
        let encoded = parts.next().unwrap();
        let digest = parts.next().unwrap();

        for invalid in [
            valid.replacen(TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1, "kiln-teacher-v2", 1),
            format!("{valid}.extra"),
            format!("{TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1}.{encoded}"),
            format!("{TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1}.{encoded}=.{digest}"),
            format!(
                "{TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1}.{encoded}.{}",
                digest.to_uppercase()
            ),
            format!("{TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1}.{encoded}.short"),
            format!("{TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1}.AB.{A}"),
        ] {
            assert!(
                TeacherIdentityV1::parse_fingerprint(&invalid).is_err(),
                "must reject {invalid}"
            );
        }

        let mut bad_digest = digest.to_owned();
        bad_digest.replace_range(0..1, if &bad_digest[0..1] == "0" { "1" } else { "0" });
        assert_eq!(
            TeacherIdentityV1::parse_fingerprint(&format!(
                "{TEACHER_IDENTITY_FINGERPRINT_PREFIX_V1}.{encoded}.{bad_digest}"
            ))
            .unwrap_err(),
            TeacherIdentityError::DigestMismatch
        );

        let oversized = "x".repeat(MAX_TEACHER_IDENTITY_FINGERPRINT_BYTES + 1);
        assert!(matches!(
            TeacherIdentityV1::parse_fingerprint(&oversized),
            Err(TeacherIdentityError::FingerprintTooLarge { .. })
        ));
    }

    #[test]
    fn fingerprint_rejects_noncanonical_json_even_with_matching_digest() {
        let json = format!(" {}", identity(false).canonical_json());
        let digest = sha256_hex(json.as_bytes());
        assert_eq!(
            TeacherIdentityV1::parse_fingerprint(
                &fingerprint_for_json_with_digest(&json, &digest,)
            )
            .unwrap_err(),
            TeacherIdentityError::NonCanonicalJson
        );
    }

    #[test]
    fn base64url_codec_is_canonical_for_all_remainders() {
        for input in [
            b"a".as_slice(),
            b"ab".as_slice(),
            b"abc".as_slice(),
            b"abcd".as_slice(),
        ] {
            let encoded = encode_base64url_no_pad(input);
            assert!(!encoded.contains('='));
            assert_eq!(decode_base64url_no_pad(&encoded).unwrap(), input);
        }
        assert!(decode_base64url_no_pad("A/").is_err());
        assert!(decode_base64url_no_pad("AB").is_err());
        assert!(decode_base64url_no_pad("AAB").is_err());
        assert!(decode_base64url_no_pad("A").is_err());
    }

    #[test]
    fn is_lower_sha256_enforces_exactly_64_lowercase_hex_bytes() {
        let valid = "0123456789abcdef".repeat(4);
        assert_eq!(valid.len(), 64);
        assert!(is_lower_sha256(&valid));

        // Length divergence: 63 and 65 bytes are rejected even when every
        // byte is otherwise valid.
        assert!(!is_lower_sha256(&valid[..63]));
        assert!(!is_lower_sha256(&format!("{valid}f")));

        // Uppercase hex is rejected, fully and mixed.
        assert!(!is_lower_sha256(&"0123456789ABCDEF".repeat(4)));
        assert!(!is_lower_sha256(&"A123456789abcdef".repeat(4)));

        // Non-hex letters are rejected in either case.
        assert!(!is_lower_sha256(&"g123456789abcdef".repeat(4)));
        assert!(!is_lower_sha256(&"z123456789abcdef".repeat(4)));
        assert!(!is_lower_sha256(&"G123456789abcdef".repeat(4)));

        // Non-ASCII is rejected even when the byte length is exactly 64:
        // each two-byte char displaces two hex chars.
        assert!(!is_lower_sha256(&format!("{}\u{e9}", "a".repeat(62))));
        assert!(!is_lower_sha256(&"\u{e9}".repeat(32)));
    }

    #[test]
    fn is_lower_sha256_matches_the_legacy_checkpoint_predicate() {
        // Round 80 routed checkpoint.rs through this helper, replacing its
        // inline `value.len() == 64 && value.bytes().all(|byte|
        // byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())` check.
        // This test locks the behavioral identity against that removed
        // expression across the full ASCII byte space, so a future edit
        // cannot let the two shapes drift.
        let legacy = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        };
        let base: Vec<char> = vec!['0'; 64];
        for byte in 0u8..=127 {
            let mut chars = base.clone();
            chars[0] = byte as char;
            let case: String = chars.into_iter().collect();
            assert_eq!(
                is_lower_sha256(&case),
                legacy(&case),
                "divergence at ASCII byte {byte:#04x}"
            );
        }
        // Length boundary around exactly 64.
        for case in ["0".repeat(63), format!("{}0", "0".repeat(64))] {
            assert_eq!(
                is_lower_sha256(&case),
                legacy(&case),
                "divergence at length boundary: len={}",
                case.len()
            );
        }
    }
}
