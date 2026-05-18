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
///
/// Two chat-template patterns to handle (verified against
/// `/workspace/qwen3.5-4b/chat_template.jinja` on 2026-05-18):
///
/// - **Standard ChatML role** (`system` / `user` / `assistant`):
///   `<|im_start|>{role}\n{content}<|im_end|>`
/// - **Qwen tool result** (`role == "tool"`): rendered *inside* a
///   `<|im_start|>user` block as `<tool_response>\n{content}\n</tool_response>`.
///   The masker therefore looks for the `<tool_response>` / `</tool_response>`
///   pair, not for a `<|im_start|>tool` marker (which Qwen3.5 does not emit).
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
    const TOOL_RESPONSE_OPEN: &str = "<tool_response>\n";
    const TOOL_RESPONSE_CLOSE: &str = "\n</tool_response>";

    let mut cursor = 0usize;
    for seg in trajectory {
        // Only Action and Observation segments need masks; Context segments
        // are still rendered into the prefix but contribute no gradient.
        if matches!(seg.kind, TurnKind::Context) {
            // Advance the cursor past this Context segment.
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

        // For Observation segments, look for the <tool_response>...</tool_response>
        // wrapper (Qwen-style) rather than a <|im_start|>tool marker.
        let (content_start, content_end_abs, advance_to) = if matches!(seg.kind, TurnKind::Observation) {
            let open_rel = full_text[cursor..]
                .find(TOOL_RESPONSE_OPEN)
                .with_context(|| {
                    format!(
                        "byte-search: could not locate {:?} for tool result after cursor {}",
                        TOOL_RESPONSE_OPEN, cursor
                    )
                })?;
            let content_start = cursor + open_rel + TOOL_RESPONSE_OPEN.len();
            let close_rel = full_text[content_start..]
                .find(TOOL_RESPONSE_CLOSE)
                .with_context(|| {
                    format!(
                        "byte-search: could not locate {:?} after tool-response content start {}",
                        TOOL_RESPONSE_CLOSE, content_start
                    )
                })?;
            let content_end_abs = content_start + close_rel;
            // Advance past the closing </tool_response>; surrounding
            // <|im_end|> handled on the next iteration if any.
            let advance = content_end_abs + TOOL_RESPONSE_CLOSE.len();
            (content_start, content_end_abs, advance)
        } else {
            // Action / standard role: <|im_start|>{role}\n{content}<|im_end|>
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
            let content_end_rel = full_text[content_start..]
                .find(MESSAGE_END)
                .with_context(|| {
                    format!(
                        "byte-search: could not locate {} after content start {}",
                        MESSAGE_END, content_start
                    )
                })?;
            let content_end_abs = content_start + content_end_rel;
            let advance = content_end_abs + MESSAGE_END.len();
            (content_start, content_end_abs, advance)
        };

        // Apply warning_filter for Observation segments. `effective_start`
        // is where the env_mask is allowed to begin; `content_start` is the
        // full semantic span start (used by segment_spans and by total_obs_len).
        let effective_start = match seg.kind {
            TurnKind::Observation if cfg.warning_filter => {
                let trim = seg.warning_prefix_len.unwrap_or(0);
                content_start.saturating_add(trim).min(content_end_abs)
            }
            _ => content_start,
        };

        // The mask span — what gets marked in action_mask / env_mask —
        // depends on env_mask_mode for Observation segments. EnvOnly
        // (default) respects warning_filter; FullObs (debug) covers the
        // full content including warnings.
        let mask_span_lo = match seg.kind {
            TurnKind::Observation if cfg.env_mask_mode == EnvMaskMode::FullObs => content_start,
            _ => effective_start,
        };

        let target_mask: &mut [bool] = match seg.kind {
            TurnKind::Action => action_mask,
            TurnKind::Observation => env_mask,
            TurnKind::Context => unreachable!("filtered above"),
        };

        // First pass: mark target_mask over [mask_span_lo, content_end_abs).
        for (idx, &(tok_start, tok_end)) in offsets.iter().enumerate() {
            if idx >= seq_len || tok_start == tok_end {
                continue;
            }
            if tok_start < content_end_abs && tok_end > mask_span_lo {
                target_mask[idx] = true;
            }
        }

        // Second pass: compute the FULL semantic token span for this segment
        // (covering the entire content range, *not* the warning-filtered
        // subset). segment_spans drives total_obs_len() which is |O| for
        // paper §3.1 length normalization — must equal the full observation
        // length regardless of warning_filter.
        let mut full_span_start = seq_len;
        let mut full_span_end = 0usize;
        for (idx, &(tok_start, tok_end)) in offsets.iter().enumerate() {
            if idx >= seq_len || tok_start == tok_end {
                continue;
            }
            if tok_start < content_end_abs && tok_end > content_start {
                if idx < full_span_start {
                    full_span_start = idx;
                }
                if idx + 1 > full_span_end {
                    full_span_end = idx + 1;
                }
            }
        }
        if full_span_end > full_span_start {
            segment_spans.push((full_span_start, full_span_end, seg.kind));
        }

        cursor = advance_to;
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
    use kiln_core::tokenizer::KilnTokenizer;

    /// Minimal byte-level tokenizer with a Qwen-shaped chat template. The
    /// vocab is single-byte ASCII / UTF-8 bytes so each byte gets its own
    /// token, which makes the offset → token mapping trivial to predict and
    /// asserting precise span boundaries straightforward.
    ///
    /// The template renders:
    ///   <|im_start|>{role}\n{content}<|im_end|>\n
    /// for all roles EXCEPT tool, which renders inside a user block as:
    ///   <|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n
    /// matching the actual Qwen3.5-4B template at
    /// /workspace/qwen3.5-4b/chat_template.jinja (verified 2026-05-18).
    fn qwen_shaped_tokenizer() -> KilnTokenizer {
        // Build a vocab that maps every byte 0..255 to a single token id.
        // BPE with no merges treats each input byte as its own token, so
        // encode_with_offsets returns (id_at_byte_n, (n, n+1)) — exactly
        // what we want for precise byte→token mapping assertions.
        let mut vocab = String::from("{");
        for b in 0u32..256 {
            // Most printable bytes go in directly; control bytes use \\u escapes.
            let ch = char::from_u32(b).unwrap();
            let key = match ch {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
                _ => ch.to_string(),
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
        // Explicit Jinja with no whitespace control — `{{- ... -}}` would
        // strip newlines and make the test layout unpredictable. The
        // tradeoff is that we emit extra newlines vs the real Qwen
        // template; the byte-search code accepts both since it locates
        // anchors by string-find rather than position.
        //
        // Per-iteration output for non-tool roles:
        //     <|im_start|>{role}\n{content}<|im_end|>\n
        // Per-iteration output for tool roles:
        //     <|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n
        //
        // For two consecutive tool turns (no intervening non-tool), the
        // wrapping <|im_start|>user...<|im_end|> envelope only opens on
        // the first and closes on the last — matching real Qwen.
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
        KilnTokenizer::from_bytes(json.as_bytes())
            .unwrap()
            .with_chat_template(template.to_string())
    }

    fn ctx(role: &str, content: &str) -> TurnSegment {
        TurnSegment {
            role: role.into(),
            content: content.into(),
            kind: TurnKind::Context,
            tool_call_id: None,
            warning_prefix_len: None,
        }
    }

    fn act(content: &str) -> TurnSegment {
        TurnSegment {
            role: "assistant".into(),
            content: content.into(),
            kind: TurnKind::Action,
            tool_call_id: None,
            warning_prefix_len: None,
        }
    }

    fn obs(content: &str, warning_prefix_len: Option<usize>) -> TurnSegment {
        TurnSegment {
            role: "tool".into(),
            content: content.into(),
            kind: TurnKind::Observation,
            tool_call_id: None,
            warning_prefix_len,
        }
    }

    /// Render a trajectory through the chat template and return both the
    /// rendered text and the per-segment expected byte ranges (so the test
    /// asserts the masker matched what the chat template actually emitted).
    fn rendered_segments(tokenizer: &KilnTokenizer, traj: &[TurnSegment]) -> String {
        let core_msgs: Vec<kiln_core::tokenizer::ChatMessage> = traj
            .iter()
            .map(|s| kiln_core::tokenizer::ChatMessage {
                role: s.role.clone(),
                content: s.content.clone(),
                ..Default::default()
            })
            .collect();
        tokenizer.apply_chat_template(&core_msgs).unwrap()
    }

    /// Helper: count tokens whose byte offsets fall fully inside a given
    /// byte range `[lo, hi)`. With the byte-level tokenizer above, each
    /// byte is one token, so this is just `hi - lo`.
    fn count_tokens_in_range(rendered: &str, content: &str) -> usize {
        // The byte-level tokenizer makes `content.len()` the exact token count.
        let _ = rendered;
        content.len()
    }

    #[test]
    fn build_masks_assistant_only_marks_action_tokens() {
        let tok = qwen_shaped_tokenizer();
        let traj = vec![
            ctx("system", "sys"),
            ctx("user", "ask"),
            act("REPLY"),
        ];
        let result =
            build_masks_from_trajectory(&traj, &[], &tok, &MaskConfig::default()).unwrap();
        let n_action_true = result.action_mask.iter().filter(|&&b| b).count();
        let n_env_true = result.env_mask.iter().filter(|&&b| b).count();
        assert_eq!(n_action_true, 5, "REPLY is 5 bytes/tokens");
        assert_eq!(n_env_true, 0, "no observation segments");
        assert_eq!(result.total_obs_len(), 0);
    }

    #[test]
    fn build_masks_four_turn_pi_session() {
        let tok = qwen_shaped_tokenizer();
        let rendered = rendered_segments(
            &tok,
            &[
                ctx("system", "sys"),
                ctx("user", "ask"),
                act("RUN"),
                obs("OUTPUT", None),
                act("FINAL"),
            ],
        );
        // Sanity: the chat template did render <tool_response> wrapping.
        assert!(rendered.contains("<tool_response>\nOUTPUT\n</tool_response>"));

        let traj = vec![
            ctx("system", "sys"),
            ctx("user", "ask"),
            act("RUN"),
            obs("OUTPUT", None),
            act("FINAL"),
        ];
        let result =
            build_masks_from_trajectory(&traj, &[], &tok, &MaskConfig::default()).unwrap();

        // The two Action segments contribute RUN (3) + FINAL (5) = 8 action tokens.
        let n_action_true = result.action_mask.iter().filter(|&&b| b).count();
        assert_eq!(n_action_true, "RUN".len() + "FINAL".len());

        // The Observation segment contributes OUTPUT (6) env tokens.
        let n_env_true = result.env_mask.iter().filter(|&&b| b).count();
        assert_eq!(n_env_true, "OUTPUT".len());

        // Masks must be disjoint.
        result.assert_masks_disjoint().unwrap();

        // total_obs_len equals the env token count when there's no warning trim.
        assert_eq!(result.total_obs_len(), "OUTPUT".len());

        // Three supervised segments expected (2 Action + 1 Observation).
        assert_eq!(result.segment_spans.len(), 3);
        let kinds: Vec<_> = result.segment_spans.iter().map(|(_, _, k)| *k).collect();
        assert_eq!(
            kinds,
            vec![TurnKind::Action, TurnKind::Observation, TurnKind::Action]
        );
    }

    #[test]
    fn build_masks_warning_filter_trims_env_span() {
        let tok = qwen_shaped_tokenizer();
        // "WARNINGS:\n- bad\n<command_output>real</command_output>"
        // warning_prefix_len = len("WARNINGS:\n- bad\n") = 16
        let raw = "WARNINGS:\n- bad\n<command_output>real</command_output>";
        let warning_prefix_len = raw.find("<command_output>").unwrap();
        assert_eq!(warning_prefix_len, 16);

        let traj = vec![
            ctx("system", "s"),
            ctx("user", "u"),
            act("A"),
            obs(raw, Some(warning_prefix_len)),
        ];

        // First, env_mask_mode = EnvOnly (default) with warning_filter on.
        let on_cfg = MaskConfig::default();
        let on = build_masks_from_trajectory(&traj, &[], &tok, &on_cfg).unwrap();
        let n_env_on = on.env_mask.iter().filter(|&&b| b).count();
        // Warning_filter trims off the warning prefix; remaining bytes are
        // raw.len() - warning_prefix_len. Tokenizer maps each byte to one
        // token in this fixture, so the count is exactly that.
        assert_eq!(n_env_on, raw.len() - warning_prefix_len);

        // Now disable warning_filter and verify the env_mask grows by
        // exactly the warning prefix length.
        let off_cfg = MaskConfig {
            warning_filter: false,
            env_mask_mode: EnvMaskMode::EnvOnly,
        };
        let off = build_masks_from_trajectory(&traj, &[], &tok, &off_cfg).unwrap();
        let n_env_off = off.env_mask.iter().filter(|&&b| b).count();
        assert!(
            n_env_off > n_env_on,
            "warning_filter=false must mark more env tokens than warning_filter=true (off={n_env_off}, on={n_env_on})"
        );
        assert_eq!(
            n_env_off - n_env_on,
            warning_prefix_len,
            "the delta between off and on must equal the warning_prefix_len"
        );

        // total_obs_len doesn't change with warning_filter — it's |O|, not |O'|.
        assert_eq!(on.total_obs_len(), off.total_obs_len());
        // total_obs_len equals the off-case env count (which covers the full content).
        assert_eq!(on.total_obs_len(), n_env_off);
    }

    #[test]
    fn build_masks_consecutive_tool_observations_share_user_block() {
        let tok = qwen_shaped_tokenizer();
        let traj = vec![
            ctx("system", "s"),
            ctx("user", "u"),
            act("call1"),
            obs("res1", None),
            obs("res2", None),  // second tool result without an intervening assistant
            act("done"),
        ];
        let result =
            build_masks_from_trajectory(&traj, &[], &tok, &MaskConfig::default()).unwrap();

        // Both env segments should be marked.
        let n_env = result.env_mask.iter().filter(|&&b| b).count();
        assert_eq!(n_env, "res1".len() + "res2".len());
        // Plus two Action segments.
        let n_act = result.action_mask.iter().filter(|&&b| b).count();
        assert_eq!(n_act, "call1".len() + "done".len());
        result.assert_masks_disjoint().unwrap();
    }

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
