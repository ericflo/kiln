//! Build `(input_ids, action_mask, env_mask)` from a [`Trajectory`].
//!
//! Generalizes the SFT pattern in
//! [`crate::trainer::label_mask_from_rendered_assistant_spans`] from one
//! role (assistant) to multiple (assistant + tool / observation), and adds
//! the paper §3.2 harness-warning-prefix exclusion for env spans.
//!
//! See `docs/plans/echo-integration-plan.md` §3.2 and §B.2 for the design.
//!
//! [`Trajectory`]: crate::trajectory

use anyhow::{Context, Result};
use kiln_core::tokenizer::{ChatMessage as CoreChatMessage, KilnTokenizer};

use crate::ChatMessage;
use crate::trajectory::{TurnKind, TurnSegment};

// ---- Public types ----------------------------------------------------------

/// How env-CE masks are built from a trajectory.
#[derive(Clone, Debug)]
pub struct MaskConfig {
    /// Whether to strip the harness `WARNINGS:\n- ...` prefix from env_mask.
    /// Paper §3.2: warning tokens are low-entropy and memorize in ~60 steps,
    /// while terminal-output tokens keep providing useful gradient. Default
    /// `true`.
    pub warning_filter: bool,

    /// `EnvOnly` (default) — `env_mask` covers only the tool-output bytes
    /// (after warning_prefix_len). `FullObs` (debug) — env_mask covers the
    /// full observation including warnings.
    pub env_mask_mode: EnvMaskMode,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            warning_filter: true,
            env_mask_mode: EnvMaskMode::EnvOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EnvMaskMode {
    #[default]
    EnvOnly,
    FullObs,
}

/// The output of [`build_masks_from_trajectory`]: tokens plus the two masks
/// ECHO requires, plus per-segment token spans for diagnostics.
#[derive(Clone, Debug)]
pub struct MaskedRollout {
    pub input_ids: Vec<u32>,
    /// True at positions the model generated (Action segments).
    pub action_mask: Vec<bool>,
    /// True at positions the environment produced (Observation segments).
    pub env_mask: Vec<bool>,
    /// Per-segment token span `(token_start_inclusive, token_end_exclusive,
    /// kind)` for diagnostics and per-segment loss attribution. One entry
    /// per non-Context segment in the trajectory (Context segments
    /// contribute no gradient, so we don't bother recording them).
    pub segment_spans: Vec<(usize, usize, TurnKind)>,
}

impl MaskedRollout {
    /// `|O|` — the total number of tokens that belong to Observation
    /// segments before warning-filter trimming. Used by ECHO's length
    /// normalization (paper §3.1: divide by `|O|`, not by `|O'|`).
    pub fn total_obs_len(&self) -> usize {
        self.segment_spans
            .iter()
            .filter(|(_, _, k)| *k == TurnKind::Observation)
            .map(|(s, e, _)| e.saturating_sub(*s))
            .sum()
    }

    /// Sanity check: action_mask and env_mask must be disjoint at every
    /// position. A position cannot simultaneously be a model-generated token
    /// and an environment-produced token. Returns the offending index on
    /// failure.
    pub fn assert_masks_disjoint(&self) -> Result<()> {
        for (i, (&a, &e)) in self
            .action_mask
            .iter()
            .zip(self.env_mask.iter())
            .enumerate()
        {
            anyhow::ensure!(
                !(a && e),
                "action_mask and env_mask overlap at token index {i}"
            );
        }
        Ok(())
    }
}

// ---- The main entry point --------------------------------------------------

/// Build `(input_ids, action_mask, env_mask, segment_spans)` from a
/// trajectory.
///
/// `prompt_messages` is the system + user scaffolding seen by every rollout
/// in the group (the `AgenticGroup::messages` field). It is *not* part of the
/// trajectory; it is prepended to the message list before chat-template
/// rendering so that the rendered prompt prefix matches what the model saw.
///
/// `trajectory` is the per-rollout sequence of `TurnSegment`s. Each segment
/// is rendered as a `<|im_start|>{role}\n{content}<|im_end|>\n` ChatML block
/// by `tokenizer.apply_chat_template`. The function then scans the rendered
/// text for those delimiters and marks the tokens overlapping each segment's
/// rendered byte range in the corresponding mask.
///
/// Defaults to the byte-search strategy (mirrors SFT's
/// `label_mask_from_rendered_assistant_spans`); falls back to cumulative
/// prefix re-tokenization (mirrors `label_mask_by_prefix_tokenization`)
/// when the byte-search can't account for every expected segment.
pub fn build_masks_from_trajectory(
    trajectory: &[TurnSegment],
    prompt_messages: &[ChatMessage],
    tokenizer: &KilnTokenizer,
    cfg: &MaskConfig,
) -> Result<MaskedRollout> {
    // Step 1: assemble the full ChatMessage list — prompt scaffold first,
    // then every trajectory segment.
    let mut full_messages: Vec<CoreChatMessage> = prompt_messages
        .iter()
        .map(|m| CoreChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            ..Default::default()
        })
        .collect();
    full_messages.extend(trajectory.iter().map(|seg| CoreChatMessage {
        role: seg.role.clone(),
        content: seg.content.clone(),
        ..Default::default()
    }));

    // Step 2: render the full conversation once.
    let full_text = tokenizer
        .apply_chat_template(&full_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("apply_chat_template on full trajectory")?;
    let (input_ids, offsets) = tokenizer
        .encode_with_offsets(&full_text)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("encode_with_offsets on rendered trajectory")?;

    anyhow::ensure!(
        !input_ids.is_empty(),
        "build_masks_from_trajectory: empty tokenization result"
    );

    let seq_len = input_ids.len();
    let mut action_mask = vec![false; seq_len];
    let mut env_mask = vec![false; seq_len];
    let mut segment_spans: Vec<(usize, usize, TurnKind)> = Vec::new();

    // Step 3: try the byte-search strategy first (cheap, robust for ChatML).
    let byte_search_result = byte_search_strategy(
        trajectory,
        &full_text,
        &offsets,
        seq_len,
        cfg,
        &mut action_mask,
        &mut env_mask,
        &mut segment_spans,
    );

    if byte_search_result.is_err()
        || segment_spans.len() != count_supervised_segments(trajectory)
    {
        // Step 3b: fall back to cumulative-prefix re-tokenization (mirrors
        // SFT's label_mask_by_prefix_tokenization).
        action_mask.iter_mut().for_each(|b| *b = false);
        env_mask.iter_mut().for_each(|b| *b = false);
        segment_spans.clear();
        prefix_tokenization_strategy(
            trajectory,
            prompt_messages,
            tokenizer,
            cfg,
            seq_len,
            &mut action_mask,
            &mut env_mask,
            &mut segment_spans,
        )?;
    }

    let masked = MaskedRollout {
        input_ids,
        action_mask,
        env_mask,
        segment_spans,
    };
    masked
        .assert_masks_disjoint()
        .context("trajectory mask invariant violated")?;
    Ok(masked)
}

// ---- Internal helpers ------------------------------------------------------

fn count_supervised_segments(trajectory: &[TurnSegment]) -> usize {
    trajectory
        .iter()
        .filter(|s| matches!(s.kind, TurnKind::Action | TurnKind::Observation))
        .count()
}

/// Byte-search strategy: scan the rendered full text for ChatML role markers
/// in order, and mark the tokens whose byte offsets overlap each segment's
/// content range.
fn byte_search_strategy(
    trajectory: &[TurnSegment],
    full_text: &str,
    offsets: &[(usize, usize)],
    seq_len: usize,
    cfg: &MaskConfig,
    action_mask: &mut [bool],
    env_mask: &mut [bool],
    segment_spans: &mut Vec<(usize, usize, TurnKind)>,
) -> Result<()> {
    const MESSAGE_END: &str = "<|im_end|>";

    let mut cursor = 0usize;
    for seg in trajectory {
        // Only Action and Observation segments need masks; Context segments
        // are still rendered into the prefix but contribute no gradient.
        if matches!(seg.kind, TurnKind::Context) {
            // Advance the cursor past this Context segment by finding the
            // next role marker for the next supervised segment, but we don't
            // need to mark anything. The role marker is rendered as
            // `<|im_start|>{role}\n`. Skip past one role's worth.
            let role_marker = format!("<|im_start|>{}\n", seg.role);
            let pos = full_text[cursor..]
                .find(&role_marker)
                .map(|p| cursor + p + role_marker.len())
                .unwrap_or(cursor);
            let end = full_text[pos..]
                .find(MESSAGE_END)
                .map(|p| pos + p + MESSAGE_END.len())
                .unwrap_or(pos);
            cursor = end;
            continue;
        }

        let role_marker = format!("<|im_start|>{}\n", seg.role);
        let role_start = full_text[cursor..]
            .find(&role_marker)
            .with_context(|| {
                format!(
                    "byte-search: could not locate role marker {:?} after cursor {}",
                    role_marker, cursor
                )
            })?;
        let content_start = cursor + role_start + role_marker.len();
        let content_end = full_text[content_start..]
            .find(MESSAGE_END)
            .with_context(|| {
                format!(
                    "byte-search: could not locate {} after content start {}",
                    MESSAGE_END, content_start
                )
            })?;
        let content_end_abs = content_start + content_end;

        // Apply warning_filter for Observation segments.
        let effective_start = match seg.kind {
            TurnKind::Observation if cfg.warning_filter => {
                let trim = seg.warning_prefix_len.unwrap_or(0);
                content_start.saturating_add(trim).min(content_end_abs)
            }
            _ => content_start,
        };

        // Mark the tokens overlapping [effective_start, content_end_abs).
        let mut span_token_start = seq_len;
        let mut span_token_end = 0usize;
        let target_mask: &mut [bool] = match seg.kind {
            TurnKind::Action => action_mask,
            TurnKind::Observation if cfg.env_mask_mode == EnvMaskMode::FullObs => env_mask,
            TurnKind::Observation => env_mask,
            TurnKind::Context => unreachable!("filtered above"),
        };
        let span_lo = match seg.kind {
            TurnKind::Observation if cfg.env_mask_mode == EnvMaskMode::FullObs => content_start,
            _ => effective_start,
        };
        for (idx, &(tok_start, tok_end)) in offsets.iter().enumerate() {
            if idx >= seq_len || tok_start == tok_end {
                continue;
            }
            if tok_start < content_end_abs && tok_end > span_lo {
                target_mask[idx] = true;
                if idx < span_token_start {
                    span_token_start = idx;
                }
                if idx + 1 > span_token_end {
                    span_token_end = idx + 1;
                }
            }
        }
        if span_token_end > span_token_start {
            segment_spans.push((span_token_start, span_token_end, seg.kind));
        }

        // Move cursor past this segment (after the <|im_end|>).
        cursor = content_end_abs + MESSAGE_END.len();
    }

    Ok(())
}

/// Cumulative-prefix re-tokenization fallback: render the conversation
/// prefix-by-prefix and use the token-id length deltas to determine each
/// segment's exact token boundaries. Robust to custom chat templates that
/// don't use the ChatML `<|im_start|>{role}\n...<|im_end|>` pattern.
///
/// Mirrors `label_mask_by_prefix_tokenization` (trainer.rs:2475) but
/// generalizes to two roles (Action + Observation).
fn prefix_tokenization_strategy(
    trajectory: &[TurnSegment],
    prompt_messages: &[ChatMessage],
    tokenizer: &KilnTokenizer,
    _cfg: &MaskConfig,
    seq_len: usize,
    action_mask: &mut [bool],
    env_mask: &mut [bool],
    segment_spans: &mut Vec<(usize, usize, TurnKind)>,
) -> Result<()> {
    // Build the same prefix message list and walk it segment by segment,
    // capturing the token-count delta for each supervised segment.
    let mut prefix_messages: Vec<CoreChatMessage> = prompt_messages
        .iter()
        .map(|m| CoreChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            ..Default::default()
        })
        .collect();

    let render = |msgs: &[CoreChatMessage]| -> Result<Vec<u32>> {
        if msgs.is_empty() {
            return Ok(Vec::new());
        }
        let txt = tokenizer
            .apply_chat_template(msgs)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        tokenizer
            .encode(&txt)
            .map_err(|e| anyhow::anyhow!("{e}"))
    };

    for seg in trajectory {
        let before_ids = render(&prefix_messages)?;
        prefix_messages.push(CoreChatMessage {
            role: seg.role.clone(),
            content: seg.content.clone(),
            ..Default::default()
        });
        let after_ids = render(&prefix_messages)?;

        if matches!(seg.kind, TurnKind::Context) {
            continue;
        }

        let start = before_ids.len();
        let end = after_ids.len().min(seq_len);
        if end <= start {
            continue;
        }

        // For the fallback path we don't have byte-level information for
        // the warning_prefix_len trimming, so we mark the entire content
        // span. This is a precision degradation in the rare-edge-case
        // path; the common byte-search path handles warning_filter
        // correctly. Phase 0 unit tests pin this behavior.
        let target_mask: &mut [bool] = match seg.kind {
            TurnKind::Action => action_mask,
            TurnKind::Observation => env_mask,
            TurnKind::Context => unreachable!(),
        };
        for i in start..end {
            target_mask[i] = true;
        }
        segment_spans.push((start, end, seg.kind));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure-math test that doesn't need a real tokenizer: feed
    /// `assert_masks_disjoint` masks that overlap and verify it catches it.
    #[test]
    fn assert_masks_disjoint_rejects_overlap() {
        let masked = MaskedRollout {
            input_ids: vec![0, 1, 2, 3],
            action_mask: vec![true, true, false, false],
            env_mask: vec![false, true, true, false],
            segment_spans: vec![],
        };
        let err = masked.assert_masks_disjoint().unwrap_err();
        assert!(err.to_string().contains("overlap at token index 1"));
    }

    #[test]
    fn assert_masks_disjoint_accepts_proper_partition() {
        let masked = MaskedRollout {
            input_ids: vec![0, 1, 2, 3, 4],
            action_mask: vec![true, true, false, false, false],
            env_mask: vec![false, false, true, true, false],
            segment_spans: vec![(0, 2, TurnKind::Action), (2, 4, TurnKind::Observation)],
        };
        masked.assert_masks_disjoint().unwrap();
    }

    #[test]
    fn total_obs_len_sums_observation_span_widths() {
        let masked = MaskedRollout {
            input_ids: vec![],
            action_mask: vec![],
            env_mask: vec![],
            segment_spans: vec![
                (0, 5, TurnKind::Action),
                (5, 12, TurnKind::Observation),
                (12, 15, TurnKind::Action),
                (15, 20, TurnKind::Observation),
            ],
        };
        // 12-5 + 20-15 = 7 + 5 = 12
        assert_eq!(masked.total_obs_len(), 12);
    }

    #[test]
    fn count_supervised_segments_ignores_context() {
        let traj = vec![
            TurnSegment {
                role: "system".into(),
                content: "x".into(),
                kind: TurnKind::Context,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "assistant".into(),
                content: "y".into(),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "tool".into(),
                content: "z".into(),
                kind: TurnKind::Observation,
                tool_call_id: None,
                warning_prefix_len: None,
            },
        ];
        assert_eq!(count_supervised_segments(&traj), 2);
    }
}
