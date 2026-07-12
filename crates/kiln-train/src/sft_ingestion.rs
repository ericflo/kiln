//! Canonical SFT row admission and provenance.
//!
//! Every SFT transport is reduced to this contract before GPU ownership. Row
//! identity is derived from canonical parsed content, so JSON whitespace and
//! object-key order cannot make inline, named-dataset, and JSONL submissions
//! select different training rows.

use std::io::BufRead;

use anyhow::{Context, Result, bail};
use kiln_core::tokenizer::KilnTokenizer;
use serde::{Deserialize, Serialize};

use crate::SftExample;

pub const SFT_INGESTION_SCHEMA_V1: &str = "kiln.sft-ingestion.v1";
const PARSED_ROW_HASH_DOMAIN: &[u8] = b"kiln.sft-parsed-row.v1\0";
const RAW_ROW_HASH_DOMAIN: &[u8] = b"kiln.sft-raw-row.v1\0";
const KEPT_CORPUS_HASH_DOMAIN: &[u8] = b"kiln.sft-kept-corpus.v1\0";

/// Submission behavior when any non-blank SFT row is unusable.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SftInvalidRowPolicy {
    /// Reject the complete submission before queue publication.
    #[default]
    Fail,
    /// Omit invalid rows and record their content hashes in the receipt.
    Skip,
}

impl std::fmt::Display for SftInvalidRowPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fail => f.write_str("fail"),
            Self::Skip => f.write_str("skip"),
        }
    }
}

/// Stable, bounded classification for an SFT row rejection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SftRowRejectionReason {
    InvalidJson,
    EmptyMessages,
    InvalidRole,
    TokenizationError,
}

impl std::fmt::Display for SftRowRejectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidJson => f.write_str("invalid_json"),
            Self::EmptyMessages => f.write_str("empty_messages"),
            Self::InvalidRole => f.write_str("invalid_role"),
            Self::TokenizationError => f.write_str("tokenization_error"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SftRejectedRowReceipt {
    /// One-based position among non-blank rows in the submitted corpus.
    pub row_index: usize,
    /// Canonical parsed-row hash, or a domain-separated raw-row hash when the
    /// row could not be parsed as JSON.
    pub row_sha256: String,
    pub reason: SftRowRejectionReason,
}

/// Immutable evidence for the rows selected by SFT ingestion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SftIngestionReceipt {
    pub schema: String,
    /// `inline`, `dataset_path`, `named_dataset`, `corrections`, `recipe`, or
    /// `rust_api`. It describes transport only and does not affect row hashes.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_locator: Option<String>,
    pub invalid_row_policy: SftInvalidRowPolicy,
    pub rows_read: usize,
    pub rows_kept: usize,
    pub rows_rejected: usize,
    /// Ordered and duplicate-preserving canonical identities of trained rows.
    pub kept_row_hashes: Vec<String>,
    pub rejected_rows: Vec<SftRejectedRowReceipt>,
    /// Ordered aggregate over `kept_row_hashes`; equal across transports for
    /// the same selected corpus.
    pub kept_corpus_sha256: String,
}

impl SftIngestionReceipt {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema == SFT_INGESTION_SCHEMA_V1,
            "unsupported SFT ingestion schema {:?}",
            self.schema
        );
        anyhow::ensure!(
            matches!(
                self.source.as_str(),
                "inline" | "dataset_path" | "named_dataset" | "corrections" | "recipe" | "rust_api"
            ),
            "unsupported SFT ingestion source {:?}",
            self.source
        );
        anyhow::ensure!(
            self.source_locator
                .as_deref()
                .is_none_or(|locator| !locator.trim().is_empty()),
            "SFT ingestion source_locator must not be blank"
        );
        anyhow::ensure!(
            self.rows_read == self.rows_kept.saturating_add(self.rows_rejected),
            "SFT ingestion row counts are inconsistent"
        );
        anyhow::ensure!(
            self.rows_kept > 0,
            "SFT ingestion receipt contains no kept rows"
        );
        anyhow::ensure!(
            self.kept_row_hashes.len() == self.rows_kept,
            "SFT ingestion kept-row count does not match its hash list"
        );
        anyhow::ensure!(
            self.rejected_rows.len() == self.rows_rejected,
            "SFT ingestion rejected-row count does not match its receipt list"
        );
        for (index, hash) in self.kept_row_hashes.iter().enumerate() {
            validate_sha256(&format!("kept_row_hashes[{index}]"), hash)?;
        }
        let mut prior = 0usize;
        for (index, row) in self.rejected_rows.iter().enumerate() {
            anyhow::ensure!(
                row.row_index > prior && row.row_index <= self.rows_read,
                "SFT ingestion rejected_rows[{index}].row_index is out of order or range"
            );
            prior = row.row_index;
            validate_sha256(
                &format!("rejected_rows[{index}].row_sha256"),
                &row.row_sha256,
            )?;
        }
        validate_sha256("kept_corpus_sha256", &self.kept_corpus_sha256)?;
        anyhow::ensure!(
            self.kept_corpus_sha256 == kept_corpus_sha256(&self.kept_row_hashes),
            "SFT ingestion kept-corpus hash does not match its row hashes"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftPreparedDataset {
    pub examples: Vec<SftExample>,
    pub ingestion: SftIngestionReceipt,
    pub max_seq_len: usize,
    pub max_supervised_tokens: usize,
}

enum CandidateRow {
    Parsed(SftExample),
    Rejected {
        row_sha256: String,
        reason: SftRowRejectionReason,
        detail: String,
    },
}

pub fn prepare_sft_examples<I>(
    examples: I,
    tokenizer: &KilnTokenizer,
    policy: SftInvalidRowPolicy,
    source: impl Into<String>,
    source_locator: Option<String>,
) -> Result<SftPreparedDataset>
where
    I: IntoIterator<Item = SftExample>,
{
    prepare_candidates(
        examples.into_iter().map(CandidateRow::Parsed),
        tokenizer,
        policy,
        source.into(),
        source_locator,
    )
}

/// Parse and validate JSONL without silently losing malformed rows. Blank lines
/// are transport whitespace and do not count as corpus rows.
pub fn prepare_sft_jsonl<R: BufRead>(
    mut reader: R,
    tokenizer: &KilnTokenizer,
    policy: SftInvalidRowPolicy,
    source: impl Into<String>,
    source_locator: Option<String>,
) -> Result<SftPreparedDataset> {
    let mut candidates = Vec::new();
    let mut buffer = Vec::new();
    let mut physical_line = 0usize;
    loop {
        buffer.clear();
        let read = reader
            .read_until(b'\n', &mut buffer)
            .with_context(|| format!("read SFT JSONL line {}", physical_line + 1))?;
        if read == 0 {
            break;
        }
        physical_line += 1;
        while buffer
            .last()
            .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
        {
            buffer.pop();
        }
        let trimmed = trim_ascii_whitespace(&buffer);
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_slice::<SftExample>(trimmed) {
            Ok(example) => candidates.push(CandidateRow::Parsed(example)),
            Err(error) => candidates.push(CandidateRow::Rejected {
                row_sha256: domain_sha256(RAW_ROW_HASH_DOMAIN, trimmed),
                reason: SftRowRejectionReason::InvalidJson,
                detail: format!("physical line {physical_line}: {error}"),
            }),
        }
    }
    prepare_candidates(candidates, tokenizer, policy, source.into(), source_locator)
}

/// Verify that materialized examples still match a server-owned ingestion
/// receipt. This is the queue-time mutation and caller-spoofing guard.
pub fn verify_prepared_sft_examples(
    examples: &[SftExample],
    tokenizer: &KilnTokenizer,
    ingestion: &SftIngestionReceipt,
) -> Result<(usize, usize)> {
    ingestion.validate()?;
    anyhow::ensure!(
        examples.len() == ingestion.rows_kept,
        "materialized SFT example count {} differs from ingestion receipt {}",
        examples.len(),
        ingestion.rows_kept
    );
    let mut max_seq_len = 0usize;
    let mut max_supervised_tokens = 0usize;
    for (index, (example, expected_hash)) in
        examples.iter().zip(&ingestion.kept_row_hashes).enumerate()
    {
        let actual_hash = canonical_sft_row_sha256(example)?;
        anyhow::ensure!(
            actual_hash == *expected_hash,
            "materialized SFT row {} hash differs from ingestion receipt",
            index + 1
        );
        let (input_ids, label_mask) =
            validate_example(example, tokenizer).map_err(|(_, detail)| {
                anyhow::anyhow!(
                    "materialized SFT row {} no longer passes ingestion validation: {detail}",
                    index + 1
                )
            })?;
        max_seq_len = max_seq_len.max(input_ids.len());
        max_supervised_tokens =
            max_supervised_tokens.max(label_mask.into_iter().filter(|active| *active).count());
    }
    Ok((max_seq_len, max_supervised_tokens))
}

fn prepare_candidates<I>(
    candidates: I,
    tokenizer: &KilnTokenizer,
    policy: SftInvalidRowPolicy,
    source: String,
    source_locator: Option<String>,
) -> Result<SftPreparedDataset>
where
    I: IntoIterator<Item = CandidateRow>,
{
    let mut examples = Vec::new();
    let mut kept_row_hashes = Vec::new();
    let mut rejected_rows = Vec::new();
    let mut rows_read = 0usize;
    let mut max_seq_len = 0usize;
    let mut max_supervised_tokens = 0usize;

    for candidate in candidates {
        rows_read = rows_read.saturating_add(1);
        let row_index = rows_read;
        let outcome = match candidate {
            CandidateRow::Parsed(example) => {
                let row_sha256 = canonical_sft_row_sha256(&example)?;
                match validate_example(&example, tokenizer) {
                    Ok((input_ids, label_mask)) => {
                        max_seq_len = max_seq_len.max(input_ids.len());
                        max_supervised_tokens = max_supervised_tokens
                            .max(label_mask.into_iter().filter(|active| *active).count());
                        kept_row_hashes.push(row_sha256);
                        examples.push(example);
                        None
                    }
                    Err((reason, detail)) => Some((row_sha256, reason, detail)),
                }
            }
            CandidateRow::Rejected {
                row_sha256,
                reason,
                detail,
            } => Some((row_sha256, reason, detail)),
        };
        if let Some((row_sha256, reason, detail)) = outcome {
            if policy == SftInvalidRowPolicy::Fail {
                bail!("SFT row {row_index} ({row_sha256}) rejected as {reason}: {detail}");
            }
            tracing::warn!(
                row_index,
                row_sha256,
                reason = %reason,
                "skipping invalid SFT row under explicit skip policy"
            );
            rejected_rows.push(SftRejectedRowReceipt {
                row_index,
                row_sha256,
                reason,
            });
        }
    }

    if examples.is_empty() {
        bail!(
            "SFT corpus contains no trainable rows (read {rows_read}, rejected {})",
            rejected_rows.len()
        );
    }
    let ingestion = SftIngestionReceipt {
        schema: SFT_INGESTION_SCHEMA_V1.to_string(),
        source,
        source_locator,
        invalid_row_policy: policy,
        rows_read,
        rows_kept: examples.len(),
        rows_rejected: rejected_rows.len(),
        kept_corpus_sha256: kept_corpus_sha256(&kept_row_hashes),
        kept_row_hashes,
        rejected_rows,
    };
    ingestion.validate()?;
    Ok(SftPreparedDataset {
        examples,
        ingestion,
        max_seq_len,
        max_supervised_tokens,
    })
}

fn validate_example(
    example: &SftExample,
    tokenizer: &KilnTokenizer,
) -> std::result::Result<(Vec<u32>, Vec<bool>), (SftRowRejectionReason, String)> {
    if example.messages.is_empty() {
        return Err((
            SftRowRejectionReason::EmptyMessages,
            "messages must not be empty".to_string(),
        ));
    }
    if let Some((index, role)) = example
        .messages
        .iter()
        .enumerate()
        .find_map(|(index, message)| {
            (!matches!(
                message.role.as_str(),
                "system" | "user" | "assistant" | "tool"
            ))
            .then_some((index, message.role.as_str()))
        })
    {
        return Err((
            SftRowRejectionReason::InvalidRole,
            format!("message {} has unsupported role {role:?}", index + 1),
        ));
    }
    crate::trainer::tokenize_for_training(example, tokenizer).map_err(|error| {
        (
            SftRowRejectionReason::TokenizationError,
            format!("{error:#}"),
        )
    })
}

fn canonical_sft_row_sha256(example: &SftExample) -> Result<String> {
    let value = serde_json::to_value(example).context("serialize canonical SFT row")?;
    let canonical = canonicalize_json(value);
    let encoded = serde_json::to_vec(&canonical).context("encode canonical SFT row")?;
    Ok(domain_sha256(PARSED_ROW_HASH_DOMAIN, &encoded))
}

fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonicalize_json).collect())
        }
        serde_json::Value::Object(values) => {
            let mut entries = values.into_iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, canonicalize_json(value)))
                    .collect(),
            )
        }
        other => other,
    }
}

fn kept_corpus_sha256(hashes: &[String]) -> String {
    let mut encoded = Vec::with_capacity(
        KEPT_CORPUS_HASH_DOMAIN.len() + hashes.iter().map(String::len).sum::<usize>(),
    );
    encoded.extend_from_slice(KEPT_CORPUS_HASH_DOMAIN);
    for hash in hashes {
        encoded.extend_from_slice(&(hash.len() as u64).to_be_bytes());
        encoded.extend_from_slice(hash.as_bytes());
    }
    kiln_core::config_hashes::sha256_bytes(&encoded)
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut encoded = Vec::with_capacity(domain.len() + bytes.len());
    encoded.extend_from_slice(domain);
    encoded.extend_from_slice(bytes);
    kiln_core::config_hashes::sha256_bytes(&encoded)
}

fn validate_sha256(field: &str, value: &str) -> Result<()> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        bail!("{field} must use sha256:<64 lowercase hex>");
    };
    anyhow::ensure!(
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} must use sha256:<64 lowercase hex>"
    );
    Ok(())
}

fn trim_ascii_whitespace(mut bytes: &[u8]) -> &[u8] {
    while bytes.first().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[1..];
    }
    while bytes.last().is_some_and(u8::is_ascii_whitespace) {
        bytes = &bytes[..bytes.len() - 1];
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> KilnTokenizer {
        KilnTokenizer::from_bytes(
            br#"{
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1},
                    "merges": []
                }
            }"#,
        )
        .unwrap()
        .with_chat_template(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        )
    }

    fn corpus() -> Vec<SftExample> {
        vec![
            SftExample {
                messages: vec![
                    crate::ChatMessage::new("user", "a"),
                    crate::ChatMessage::new("assistant", "b"),
                ],
            },
            SftExample { messages: vec![] },
            SftExample {
                messages: vec![
                    crate::ChatMessage::new("user", "b"),
                    crate::ChatMessage::new("assistant", "a"),
                ],
            },
        ]
    }

    #[test]
    fn invalid_row_policy_defaults_to_fail_and_uses_snake_case() {
        assert_eq!(SftInvalidRowPolicy::default(), SftInvalidRowPolicy::Fail);
        assert_eq!(
            serde_json::to_string(&SftInvalidRowPolicy::Skip).unwrap(),
            "\"skip\""
        );
        assert!(serde_json::from_str::<SftInvalidRowPolicy>("\"drop\"").is_err());
    }

    #[test]
    fn canonical_json_orders_nested_tool_call_keys() {
        let left: SftExample = serde_json::from_value(serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "id": "1", "function": {
                    "arguments": "{\"x\":1}", "name": "calc"
                }}]
            }]
        }))
        .unwrap();
        let right: SftExample = serde_json::from_str(
            r#"{"messages":[{"tool_calls":[{"function":{"name":"calc","arguments":"{\"x\":1}"},"id":"1","type":"function"}],"content":"","role":"assistant"}]}"#,
        )
        .unwrap();
        assert_eq!(
            canonical_sft_row_sha256(&left).unwrap(),
            canonical_sft_row_sha256(&right).unwrap()
        );
    }

    #[test]
    fn same_corpus_selects_same_rows_for_every_sft_transport() {
        let tokenizer = tokenizer();
        let inline = prepare_sft_examples(
            corpus(),
            &tokenizer,
            SftInvalidRowPolicy::Skip,
            "inline",
            None,
        )
        .unwrap();
        let jsonl = corpus()
            .into_iter()
            .map(|row| serde_json::to_string(&row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let streamed = prepare_sft_jsonl(
            std::io::Cursor::new(format!("\n{jsonl}\n\n")),
            &tokenizer,
            SftInvalidRowPolicy::Skip,
            "dataset_path",
            Some("/data/train.jsonl".to_string()),
        )
        .unwrap();

        for source in ["named_dataset", "corrections", "recipe", "rust_api"] {
            let prepared = prepare_sft_examples(
                corpus(),
                &tokenizer,
                SftInvalidRowPolicy::Skip,
                source,
                Some(source.to_string()),
            )
            .unwrap();
            assert_eq!(prepared.examples, inline.examples, "source={source}");
            assert_eq!(
                prepared.ingestion.kept_row_hashes, inline.ingestion.kept_row_hashes,
                "source={source}"
            );
            assert_eq!(
                prepared.ingestion.rejected_rows, inline.ingestion.rejected_rows,
                "source={source}"
            );
            assert_eq!(
                prepared.ingestion.kept_corpus_sha256, inline.ingestion.kept_corpus_sha256,
                "source={source}"
            );
        }
        assert_eq!(streamed.examples, inline.examples);
        assert_eq!(
            streamed.ingestion.kept_row_hashes,
            inline.ingestion.kept_row_hashes
        );
        assert_eq!(
            streamed.ingestion.rejected_rows,
            inline.ingestion.rejected_rows
        );
        assert_eq!(
            streamed.ingestion.kept_corpus_sha256,
            inline.ingestion.kept_corpus_sha256
        );
        assert_eq!(inline.ingestion.rows_read, 3);
        assert_eq!(inline.ingestion.rows_kept, 2);
        assert_eq!(inline.ingestion.rows_rejected, 1);
    }

    #[test]
    fn fail_policy_rejects_the_same_semantic_row_for_inline_and_jsonl() {
        let tokenizer = tokenizer();
        let expected_hash = canonical_sft_row_sha256(&corpus()[1]).unwrap();
        let inline_error = prepare_sft_examples(
            corpus(),
            &tokenizer,
            SftInvalidRowPolicy::Fail,
            "inline",
            None,
        )
        .unwrap_err()
        .to_string();
        let jsonl = corpus()
            .into_iter()
            .map(|row| serde_json::to_string(&row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let jsonl_error = prepare_sft_jsonl(
            std::io::Cursor::new(jsonl),
            &tokenizer,
            SftInvalidRowPolicy::Fail,
            "dataset_path",
            None,
        )
        .unwrap_err()
        .to_string();
        for error in [inline_error, jsonl_error] {
            assert!(error.contains("SFT row 2"), "{error}");
            assert!(error.contains(&expected_hash), "{error}");
            assert!(error.contains("empty_messages"), "{error}");
        }
    }

    #[test]
    fn skip_policy_hashes_malformed_json_without_retaining_content() {
        let valid = serde_json::to_string(&corpus()[0]).unwrap();
        let prepared = prepare_sft_jsonl(
            std::io::Cursor::new(format!("{valid}\nnot-json\n")),
            &tokenizer(),
            SftInvalidRowPolicy::Skip,
            "dataset_path",
            None,
        )
        .unwrap();
        assert_eq!(prepared.ingestion.rows_kept, 1);
        assert_eq!(prepared.ingestion.rows_rejected, 1);
        assert_eq!(
            prepared.ingestion.rejected_rows[0].reason,
            SftRowRejectionReason::InvalidJson
        );
        let encoded = serde_json::to_string(&prepared.ingestion).unwrap();
        assert!(!encoded.contains("not-json"));
    }

    #[test]
    fn verification_detects_row_and_manifest_mutation() {
        let prepared = prepare_sft_examples(
            vec![corpus()[0].clone()],
            &tokenizer(),
            SftInvalidRowPolicy::Fail,
            "inline",
            None,
        )
        .unwrap();
        let mut changed = prepared.examples.clone();
        changed[0].messages[1].content = "a".to_string();
        assert!(
            verify_prepared_sft_examples(&changed, &tokenizer(), &prepared.ingestion)
                .unwrap_err()
                .to_string()
                .contains("hash differs")
        );

        let mut tampered = prepared.ingestion.clone();
        tampered.rows_read += 1;
        assert!(tampered.validate().is_err());

        let mut tampered = prepared.ingestion.clone();
        tampered.source = "unknown_transport".to_string();
        assert!(
            tampered
                .validate()
                .unwrap_err()
                .to_string()
                .contains("unsupported SFT ingestion source")
        );

        let mut tampered = prepared.ingestion;
        tampered.source_locator = Some("  ".to_string());
        assert!(
            tampered
                .validate()
                .unwrap_err()
                .to_string()
                .contains("source_locator must not be blank")
        );

        tampered.source_locator = None;
        tampered.rows_read = 0;
        tampered.rows_kept = 0;
        tampered.kept_row_hashes.clear();
        tampered.kept_corpus_sha256 = kept_corpus_sha256(&[]);
        assert!(
            tampered
                .validate()
                .unwrap_err()
                .to_string()
                .contains("no kept rows")
        );
    }
}
