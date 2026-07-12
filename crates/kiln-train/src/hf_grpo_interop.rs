//! Strict corpus validation for the HF/TRL GRPO handoff.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::lora_loader::LoraSourceIdentity;
use sha2::{Digest, Sha256};

use crate::{
    BehaviorPolicy, GrpoConfig, GrpoGroup, HfTrlDataExport, HfTrlDatasetFormat, HfTrlInputAdapter,
    HfTrlModelIdentity, ROLLOUT_PROVENANCE_SCHEMA_V1, RolloutAdapterIdentityV1,
    RolloutBehaviorPolicyIdentityV1, RolloutTokenizerIdentityV1, rollout_prompt_messages_sha256,
    scored_rollout_payload_sha256,
};

pub const HF_TRL_GRPO_CORPUS_IDENTITY_V1: &str = "kiln.hf-trl-grpo-corpus.v1";
pub const HF_TRL_GRPO_MAX_DATASET_BYTES: u64 = 64 * 1024 * 1024 * 1024;
pub const HF_TRL_GRPO_MAX_ROW_BYTES: u64 = 256 * 1024 * 1024;
pub const HF_TRL_GRPO_MAX_GROUPS: u64 = 10_000_000;
pub const HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP: usize = 1024;

const CORPUS_DIGEST_DOMAIN: &[u8] = b"kiln.hf-trl-grpo-corpus.v1\0";
const MAX_TOKENIZER_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_CHAT_TEMPLATE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfTrlGrpoCorpusSummary {
    pub row_count: u64,
    pub completion_count: u64,
    pub sampled_action_tokens: u64,
    pub forced_action_tokens: u64,
    pub max_sequence_tokens: u64,
    pub ordered_corpus_sha256: String,
    pub behavior_policy: RolloutBehaviorPolicyIdentityV1,
}

pub(crate) struct GrpoCorpusDigest {
    digest: Sha256,
    rows: u64,
}

impl GrpoCorpusDigest {
    pub(crate) fn new() -> Self {
        let mut digest = Sha256::new();
        digest.update(CORPUS_DIGEST_DOMAIN);
        Self { digest, rows: 0 }
    }

    pub(crate) fn observe(&mut self, canonical_row: &[u8]) -> Result<()> {
        let row_index = self.rows;
        self.rows = self
            .rows
            .checked_add(1)
            .context("GRPO row-count overflow")?;
        let row_len = u64::try_from(canonical_row.len()).context("GRPO row size exceeds u64")?;
        self.digest.update(row_index.to_le_bytes());
        self.digest.update(row_len.to_le_bytes());
        self.digest.update(canonical_row);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> String {
        self.digest.update(self.rows.to_le_bytes());
        prefixed_hex(&self.digest.finalize())
    }
}

/// Compute the domain-separated ordered identity of canonical compact JSON
/// rows. The dataset file digest separately binds newline framing.
pub fn ordered_grpo_corpus_sha256<'a>(rows: impl IntoIterator<Item = &'a [u8]>) -> Result<String> {
    let mut digest = GrpoCorpusDigest::new();
    for row in rows {
        digest.observe(row)?;
    }
    Ok(digest.finish())
}

pub(crate) fn read_canonical_grpo_row<R: BufRead>(
    reader: &mut R,
    row_position: u64,
    row: &mut Vec<u8>,
) -> Result<Option<GrpoGroup>> {
    row.clear();
    let mut limited = reader.by_ref().take(HF_TRL_GRPO_MAX_ROW_BYTES + 1);
    let read = limited
        .read_until(b'\n', row)
        .with_context(|| format!("read HF/TRL GRPO row {row_position}"))?;
    if read == 0 {
        return Ok(None);
    }
    ensure!(
        u64::try_from(read)
            .ok()
            .is_some_and(|read| read <= HF_TRL_GRPO_MAX_ROW_BYTES),
        "HF/TRL GRPO row {row_position} exceeds {HF_TRL_GRPO_MAX_ROW_BYTES} bytes"
    );
    ensure!(
        row.last() == Some(&b'\n'),
        "HF/TRL GRPO dataset must end every row with LF, including the final row"
    );
    row.pop();
    ensure!(
        !row.is_empty(),
        "HF/TRL GRPO dataset contains a blank row at position {row_position}"
    );
    let group: GrpoGroup = serde_json::from_slice(row)
        .with_context(|| format!("parse canonical HF/TRL GRPO row {row_position}"))?;
    let canonical = serde_json::to_vec(&group)
        .with_context(|| format!("serialize canonical HF/TRL GRPO row {row_position}"))?;
    ensure!(
        canonical.as_slice() == row.as_slice(),
        "HF/TRL GRPO row {row_position} is not canonical compact JSON or contains unknown/duplicate fields"
    );
    Ok(Some(group))
}

/// Verify the canonical GRPO JSONL corpus referenced by an already validated
/// export manifest. Callers must verify the model, dataset, and optional input
/// adapter file identities immediately before invoking this function.
pub(crate) fn verify_hf_trl_grpo_corpus(
    root: &Path,
    model: &HfTrlModelIdentity,
    data: &HfTrlDataExport,
    input_adapter: Option<&HfTrlInputAdapter>,
) -> Result<HfTrlGrpoCorpusSummary> {
    ensure!(
        data.format == HfTrlDatasetFormat::GrpoGroupsJsonl,
        "HF/TRL GRPO corpus must use grpo_groups_jsonl"
    );
    ensure!(
        data.rollout_provenance_schema.as_deref() == Some(ROLLOUT_PROVENANCE_SCHEMA_V1),
        "HF/TRL GRPO corpus requires exact {ROLLOUT_PROVENANCE_SCHEMA_V1} records"
    );
    ensure!(
        data.dataset.size_bytes <= HF_TRL_GRPO_MAX_DATASET_BYTES,
        "HF/TRL GRPO dataset is {} bytes; maximum is {HF_TRL_GRPO_MAX_DATASET_BYTES}",
        data.dataset.size_bytes
    );

    let tokenizer_bytes = read_regular_artifact(
        root,
        &model.tokenizer.relative_path,
        model.tokenizer.size_bytes,
        MAX_TOKENIZER_ARTIFACT_BYTES,
        "tokenizer",
    )?;
    let chat_template_bytes = read_regular_artifact(
        root,
        &model.chat_template.relative_path,
        model.chat_template.size_bytes,
        MAX_CHAT_TEMPLATE_BYTES,
        "chat template",
    )?;
    let chat_template = String::from_utf8(chat_template_bytes)
        .context("HF/TRL GRPO chat template must be UTF-8")?;
    let tokenizer = KilnTokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| anyhow::anyhow!("parse HF/TRL GRPO tokenizer: {error}"))?
        .with_chat_template(chat_template);
    ensure!(
        tokenizer.vocab_identity_sha256() == model.tokenizer_vocab_sha256,
        "HF/TRL GRPO tokenizer vocabulary differs from the export manifest"
    );
    ensure!(
        tokenizer
            .tokenizer_config_sha256()
            .map_err(|error| anyhow::anyhow!("hash HF/TRL GRPO tokenizer config: {error}"))?
            == model.tokenizer.sha256,
        "HF/TRL GRPO tokenizer configuration differs from the export manifest"
    );
    ensure!(
        tokenizer
            .chat_template_sha256()
            .context("HF/TRL GRPO tokenizer has no chat template")?
            == model.chat_template.sha256,
        "HF/TRL GRPO chat template differs from the export manifest"
    );

    let expected_adapter = expected_behavior_adapter(input_adapter)?;
    let dataset_path = root.join(&data.dataset.relative_path);
    let metadata = std::fs::symlink_metadata(&dataset_path)
        .with_context(|| format!("inspect HF/TRL GRPO dataset {}", dataset_path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "HF/TRL GRPO dataset must be a regular file"
    );
    ensure!(
        metadata.len() == data.dataset.size_bytes,
        "HF/TRL GRPO dataset size changed after identity verification"
    );

    let file = File::open(&dataset_path)
        .with_context(|| format!("open HF/TRL GRPO dataset {}", dataset_path.display()))?;
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    let mut row = Vec::new();
    let mut corpus_digest = GrpoCorpusDigest::new();
    let mut row_count = 0u64;
    let mut completion_count = 0u64;
    let mut sampled_action_tokens = 0u64;
    let mut forced_action_tokens = 0u64;
    let mut max_sequence_tokens = 0u64;
    let mut behavior_policy: Option<RolloutBehaviorPolicyIdentityV1> = None;

    loop {
        let Some(group) = read_canonical_grpo_row(&mut reader, row_count + 1, &mut row)? else {
            break;
        };
        row_count = row_count
            .checked_add(1)
            .context("GRPO row-count overflow")?;
        ensure!(
            row_count <= HF_TRL_GRPO_MAX_GROUPS,
            "HF/TRL GRPO dataset exceeds {HF_TRL_GRPO_MAX_GROUPS} groups"
        );
        corpus_digest.observe(&row)?;

        let group_summary = validate_group(
            &group,
            row_count,
            model,
            expected_adapter.as_ref(),
            &tokenizer,
        )?;
        completion_count = completion_count
            .checked_add(group_summary.completion_count)
            .context("GRPO completion-count overflow")?;
        sampled_action_tokens = sampled_action_tokens
            .checked_add(group_summary.sampled_action_tokens)
            .context("GRPO sampled-token count overflow")?;
        forced_action_tokens = forced_action_tokens
            .checked_add(group_summary.forced_action_tokens)
            .context("GRPO forced-token count overflow")?;
        max_sequence_tokens = max_sequence_tokens.max(group_summary.max_sequence_tokens);
        match behavior_policy.as_ref() {
            Some(expected) => ensure!(
                expected == &group_summary.behavior_policy,
                "HF/TRL GRPO row {row_count} uses a different behavior-policy identity"
            ),
            None => behavior_policy = Some(group_summary.behavior_policy),
        }
    }

    ensure!(row_count > 0, "HF/TRL GRPO dataset contains no groups");
    ensure!(
        row_count == data.row_count,
        "HF/TRL GRPO row count {row_count} differs from manifest value {}",
        data.row_count
    );
    let ordered_corpus_sha256 = corpus_digest.finish();
    ensure!(
        ordered_corpus_sha256 == data.ordered_corpus_sha256,
        "HF/TRL GRPO ordered corpus identity differs: manifest={}, computed={ordered_corpus_sha256}",
        data.ordered_corpus_sha256
    );

    Ok(HfTrlGrpoCorpusSummary {
        row_count,
        completion_count,
        sampled_action_tokens,
        forced_action_tokens,
        max_sequence_tokens,
        ordered_corpus_sha256,
        behavior_policy: behavior_policy.context("HF/TRL GRPO corpus has no behavior policy")?,
    })
}

struct GroupSummary {
    completion_count: u64,
    sampled_action_tokens: u64,
    forced_action_tokens: u64,
    max_sequence_tokens: u64,
    behavior_policy: RolloutBehaviorPolicyIdentityV1,
}

fn validate_group(
    group: &GrpoGroup,
    row_index: u64,
    model: &HfTrlModelIdentity,
    expected_adapter: Option<&RolloutAdapterIdentityV1>,
    tokenizer: &KilnTokenizer,
) -> Result<GroupSummary> {
    ensure!(
        !group.messages.is_empty(),
        "HF/TRL GRPO row {row_index} has no prompt messages"
    );
    ensure!(
        (2..=HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP).contains(&group.completions.len()),
        "HF/TRL GRPO row {row_index} must contain 2..={HF_TRL_GRPO_MAX_COMPLETIONS_PER_GROUP} completions"
    );
    let prompt_sha256 = rollout_prompt_messages_sha256(&group.messages)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("hash HF/TRL GRPO row {row_index} prompt"))?;
    let expected_tokenizer = RolloutTokenizerIdentityV1 {
        vocab_sha256: model.tokenizer_vocab_sha256.clone(),
        config_sha256: model.tokenizer.sha256.clone(),
        chat_template_sha256: model.chat_template.sha256.clone(),
    };
    let mut behavior_policy: Option<RolloutBehaviorPolicyIdentityV1> = None;
    let mut sampled_action_tokens = 0u64;
    let mut forced_action_tokens = 0u64;
    let mut max_sequence_tokens = 0u64;

    for (completion_index, completion) in group.completions.iter().enumerate() {
        ensure!(
            completion.reward.is_finite(),
            "HF/TRL GRPO row {row_index} completion {completion_index} has a non-finite reward"
        );
        let provenance = completion.provenance.as_ref().with_context(|| {
            format!(
                "HF/TRL GRPO row {row_index} completion {completion_index} is missing exact rollout provenance"
            )
        })?;
        provenance
            .validate()
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "validate HF/TRL GRPO row {row_index} completion {completion_index} provenance"
                )
            })?;
        ensure!(
            provenance.prompt_messages_sha256 == prompt_sha256,
            "HF/TRL GRPO row {row_index} completion {completion_index} prompt identity differs from its provenance"
        );
        let payload_sha256 = scored_rollout_payload_sha256(completion)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!("hash HF/TRL GRPO row {row_index} completion {completion_index}")
            })?;
        ensure!(
            provenance.scored_payload_sha256 == payload_sha256,
            "HF/TRL GRPO row {row_index} completion {completion_index} scored payload differs from its provenance"
        );
        ensure!(
            provenance.tokenizer == expected_tokenizer,
            "HF/TRL GRPO row {row_index} completion {completion_index} tokenizer identity differs from the export"
        );
        ensure!(
            provenance.behavior_policy.served_model_id == model.served_model_id,
            "HF/TRL GRPO row {row_index} completion {completion_index} served model differs from the export"
        );
        ensure!(
            provenance.behavior_policy.base_model_sha256
                == model.base_weight_shard_manifest.aggregate_sha256,
            "HF/TRL GRPO row {row_index} completion {completion_index} base model differs from the export"
        );
        ensure!(
            provenance.behavior_policy.adapter.as_ref() == expected_adapter,
            "HF/TRL GRPO row {row_index} completion {completion_index} behavior adapter differs from the exported input adapter"
        );
        match behavior_policy.as_ref() {
            Some(expected) => ensure!(
                expected == &provenance.behavior_policy,
                "HF/TRL GRPO row {row_index} mixes behavior-policy identities"
            ),
            None => behavior_policy = Some(provenance.behavior_policy.clone()),
        }
        for action in &provenance.action_tokens {
            match action.source {
                crate::RolloutActionTokenSourceV1::Sampled => {
                    sampled_action_tokens = sampled_action_tokens
                        .checked_add(1)
                        .context("GRPO sampled-token count overflow")?;
                }
                crate::RolloutActionTokenSourceV1::Forced => {
                    forced_action_tokens = forced_action_tokens
                        .checked_add(1)
                        .context("GRPO forced-token count overflow")?;
                }
            }
        }
        max_sequence_tokens = max_sequence_tokens.max(
            u64::try_from(provenance.input_token_ids.len())
                .context("GRPO sequence length exceeds u64")?,
        );
    }

    let config = GrpoConfig {
        behavior_policy: BehaviorPolicy::Recorded,
        ..GrpoConfig::default()
    };
    crate::trainer::validate_grpo_group_policy_data(group, &config, tokenizer).with_context(
        || format!("replay HF/TRL GRPO row {row_index} through the production tokenizer/mask path"),
    )?;

    Ok(GroupSummary {
        completion_count: u64::try_from(group.completions.len())
            .context("GRPO completion count exceeds u64")?,
        sampled_action_tokens,
        forced_action_tokens,
        max_sequence_tokens,
        behavior_policy: behavior_policy.context("HF/TRL GRPO group has no behavior policy")?,
    })
}

fn expected_behavior_adapter(
    input_adapter: Option<&HfTrlInputAdapter>,
) -> Result<Option<RolloutAdapterIdentityV1>> {
    input_adapter
        .map(|adapter| {
            let weights_sha256 = strip_sha256(&adapter.model.sha256, "input adapter weights")?;
            let config_sha256 = strip_sha256(&adapter.config.sha256, "input adapter config")?;
            let identity = LoraSourceIdentity::from_verified_peft_digests(
                adapter.model.size_bytes,
                weights_sha256,
                config_sha256,
            )
            .context("derive HF/TRL GRPO input-adapter content identity")?;
            Ok(RolloutAdapterIdentityV1 {
                name: adapter.name.clone(),
                content_sha256: format!("sha256:{}", identity.content_revision()),
            })
        })
        .transpose()
}

fn strip_sha256<'a>(digest: &'a str, field: &str) -> Result<&'a str> {
    digest
        .strip_prefix("sha256:")
        .with_context(|| format!("HF/TRL {field} digest is not sha256-prefixed"))
}

fn read_regular_artifact(
    root: &Path,
    relative: &str,
    expected_bytes: u64,
    max_bytes: u64,
    label: &str,
) -> Result<Vec<u8>> {
    ensure!(
        expected_bytes <= max_bytes,
        "HF/TRL GRPO {label} is {expected_bytes} bytes; maximum is {max_bytes}"
    );
    let path = root.join(relative);
    let metadata = std::fs::symlink_metadata(&path)
        .with_context(|| format!("inspect HF/TRL GRPO {label} {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "HF/TRL GRPO {label} must be a regular file"
    );
    ensure!(
        metadata.len() == expected_bytes,
        "HF/TRL GRPO {label} size changed after identity verification"
    );
    let mut bytes = Vec::with_capacity(
        usize::try_from(expected_bytes).context("HF/TRL GRPO artifact exceeds address space")?,
    );
    File::open(&path)
        .with_context(|| format!("open HF/TRL GRPO {label} {}", path.display()))?
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read HF/TRL GRPO {label} {}", path.display()))?;
    ensure!(
        u64::try_from(bytes.len()).ok() == Some(expected_bytes),
        "HF/TRL GRPO {label} changed while being read"
    );
    Ok(bytes)
}

fn prefixed_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity("sha256:".len() + bytes.len() * 2);
    output.push_str("sha256:");
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_corpus_digest_matches_cross_language_golden() {
        assert_eq!(
            ordered_grpo_corpus_sha256([b"{\"a\":1}".as_slice(), b"[]".as_slice()]).unwrap(),
            "sha256:3f495e234c3946d423ff971d5ade7fc11e7139a5a24cd3ea35a36cba2c6bdeb3"
        );
        assert_ne!(
            ordered_grpo_corpus_sha256([b"[]".as_slice(), b"{\"a\":1}".as_slice()]).unwrap(),
            "sha256:3f495e234c3946d423ff971d5ade7fc11e7139a5a24cd3ea35a36cba2c6bdeb3"
        );
    }
}
