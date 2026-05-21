//! Reproducible synthetic long-context GRPO fixtures.
//!
//! These fixtures mirror the compaction-shaped rollouts used by
//! `long_context_grpo_bench`, but live in the library so tests, examples, and
//! diagnostics can all build the same input instead of hand-rolling variants.

use anyhow::{Context, Result};
use kiln_core::tokenizer::KilnTokenizer;
use serde::{Deserialize, Serialize};

use crate::trajectory_mask::{MaskConfig, build_masks_from_trajectory};
use crate::{ChatMessage, GrpoGroup, ScoredRollout, TurnKind, TurnSegment};

pub const LONG_CONTEXT_FIXTURE_SCHEMA_VERSION: u32 = 1;
pub const LONG_CONTEXT_FIXTURE_TYPE: &str = "kiln_synthetic_long_context_grpo_fixture";
pub const LONG_CONTEXT_FIXTURE_VERIFY_INSTRUCTION: &str =
    "Inspect this synthetic trace and answer with the exact phrase: pi_compaction retained facts.";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticLongContextFixture {
    pub schema_version: u32,
    pub fixture_type: String,
    pub requested_seq_len: usize,
    pub observed_seq_len: usize,
    pub completions: usize,
    pub adapter_verify_prompt: String,
    pub group: GrpoGroup,
}

impl SyntheticLongContextFixture {
    pub fn new(
        requested_seq_len: usize,
        observed_seq_len: usize,
        completions: usize,
        group: GrpoGroup,
    ) -> Self {
        let adapter_verify_prompt = adapter_verify_prompt_for_group(&group);
        Self {
            schema_version: LONG_CONTEXT_FIXTURE_SCHEMA_VERSION,
            fixture_type: LONG_CONTEXT_FIXTURE_TYPE.to_string(),
            requested_seq_len,
            observed_seq_len,
            completions,
            adapter_verify_prompt,
            group,
        }
    }
}

pub fn synthetic_long_context_fixture_for_length(
    tokenizer: &KilnTokenizer,
    target_len: usize,
    completions: usize,
) -> Result<SyntheticLongContextFixture> {
    let group = synthetic_long_context_group_for_length(tokenizer, target_len, completions)?;
    let observed_seq_len = observed_long_context_seq_len(tokenizer, &group)?;
    Ok(SyntheticLongContextFixture::new(
        target_len,
        observed_seq_len,
        completions,
        group,
    ))
}

pub fn synthetic_long_context_group_for_length(
    tokenizer: &KilnTokenizer,
    target_len: usize,
    completions: usize,
) -> Result<GrpoGroup> {
    anyhow::ensure!(
        completions >= 2,
        "long-context fixture needs at least two completions for non-degenerate GRPO advantages"
    );
    let messages = synthetic_long_context_prompt_messages();
    let mut high = 1usize;
    while rollout_seq_len_for_repeats(tokenizer, &messages, high)? < target_len {
        high = high.saturating_mul(2);
        anyhow::ensure!(
            high <= target_len.saturating_mul(32).max(1_024),
            "could not synthesize a rollout near {target_len} tokens"
        );
    }

    let mut low = 0usize;
    while low < high {
        let mid = low + (high - low) / 2;
        if rollout_seq_len_for_repeats(tokenizer, &messages, mid)? < target_len {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    let completions = (0..completions)
        .map(|idx| {
            let reward = if idx % 2 == 0 { 1.0 } else { 0.0 };
            synthetic_long_context_rollout(low, idx, reward)
        })
        .collect();
    Ok(GrpoGroup {
        messages,
        completions,
    })
}

pub fn observed_long_context_seq_len(
    tokenizer: &KilnTokenizer,
    group: &GrpoGroup,
) -> Result<usize> {
    let mut max_len = 0usize;
    for rollout in &group.completions {
        max_len = max_len.max(rollout_seq_len(tokenizer, &group.messages, rollout)?);
    }
    Ok(max_len)
}

pub fn synthetic_long_context_prompt_messages() -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a concise agent compressing long terminal traces.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Inspect the synthetic trace and produce the final compact answer."
                .to_string(),
        },
    ]
}

pub fn synthetic_long_context_rollout(
    repeats: usize,
    completion_idx: usize,
    reward: f64,
) -> ScoredRollout {
    let observation_unit = format!(
        "trace_line={completion_idx} status=ok metric=pi_compaction payload=abcdefghijklmnopqrstuvwxyz0123456789\n"
    );
    let observation = observation_unit.repeat(repeats);
    ScoredRollout::from_trajectory(
        vec![
            TurnSegment {
                role: "assistant".to_string(),
                content: format!(
                    "<tool_call>\n{{\"cmd\":\"inspect_trace\",\"completion\":{completion_idx}}}\n</tool_call>"
                ),
                kind: TurnKind::Action,
                tool_call_id: Some(format!("inspect-{completion_idx}")),
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "tool".to_string(),
                content: observation,
                kind: TurnKind::Observation,
                tool_call_id: Some(format!("inspect-{completion_idx}")),
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "assistant".to_string(),
                content: format!(
                    "Final compact answer for completion {completion_idx}: retain causal facts and discard repeated trace noise."
                ),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
        ],
        reward,
    )
}

pub fn adapter_verify_prompt_for_group(group: &GrpoGroup) -> String {
    let trace = group
        .completions
        .first()
        .and_then(|rollout| {
            rollout
                .effective_trajectory()
                .iter()
                .find(|segment| segment.kind == TurnKind::Observation)
                .map(|segment| segment.content.clone())
        })
        .unwrap_or_default();
    format!("{LONG_CONTEXT_FIXTURE_VERIFY_INSTRUCTION}\n\n{trace}")
}

fn rollout_seq_len_for_repeats(
    tokenizer: &KilnTokenizer,
    messages: &[ChatMessage],
    repeats: usize,
) -> Result<usize> {
    let rollout = synthetic_long_context_rollout(repeats, 0, 1.0);
    rollout_seq_len(tokenizer, messages, &rollout)
}

fn rollout_seq_len(
    tokenizer: &KilnTokenizer,
    messages: &[ChatMessage],
    rollout: &ScoredRollout,
) -> Result<usize> {
    let trajectory = rollout.effective_trajectory();
    let masked = build_masks_from_trajectory(
        trajectory.as_ref(),
        messages,
        tokenizer,
        &MaskConfig::default(),
    )
    .context("build masks for synthetic long-context fixture")?;
    Ok(masked.input_ids.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_tokenizer() -> Result<KilnTokenizer> {
        let mut vocab = String::from("{");
        for b in 0u32..256 {
            let ch = char::from_u32(b).context("invalid byte vocab char")?;
            let key = match ch {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
                c => c.to_string(),
            };
            if b > 0 {
                vocab.push(',');
            }
            vocab.push_str(&format!("\"{}\":{}", key, b));
        }
        vocab.push('}');
        let json = format!(
            r#"{{"version": "1.0", "model": {{"type": "BPE", "vocab": {}, "merges": []}}}}"#,
            vocab
        );
        let template = "{% for message in messages -%}\
{% if message.role == 'tool' %}\
{% if loop.previtem is undefined or loop.previtem.role != 'tool' %}<|im_start|>user
{% endif %}<tool_response>
{{ message.content }}
</tool_response>\
{% if loop.last or loop.nextitem.role != 'tool' %}<|im_end|>
{% endif %}\
{% else %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endif %}\
{% endfor %}";
        Ok(KilnTokenizer::from_bytes(json.as_bytes())
            .map_err(|err| anyhow::anyhow!("{err}"))?
            .with_chat_template(template.to_string()))
    }

    #[test]
    fn synthetic_fixture_is_reproducible_and_serializable() -> Result<()> {
        let tokenizer = synthetic_tokenizer()?;
        let a = synthetic_long_context_fixture_for_length(&tokenizer, 512, 2)?;
        let b = synthetic_long_context_fixture_for_length(&tokenizer, 512, 2)?;

        assert_eq!(a.requested_seq_len, 512);
        assert!(a.observed_seq_len >= 512);
        assert_eq!(a.observed_seq_len, b.observed_seq_len);
        assert_eq!(
            a.group.completions[0].trajectory[1].content,
            b.group.completions[0].trajectory[1].content
        );
        assert!(a.adapter_verify_prompt.contains("pi_compaction"));
        assert!(a.adapter_verify_prompt.contains("trace_line=0"));

        let json = serde_json::to_string(&a)?;
        let decoded: SyntheticLongContextFixture = serde_json::from_str(&json)?;
        assert_eq!(decoded.fixture_type, LONG_CONTEXT_FIXTURE_TYPE);
        assert_eq!(decoded.group.completions.len(), 2);
        Ok(())
    }
}
