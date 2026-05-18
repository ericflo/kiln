//! Canonical trajectory schema for agentic rollouts.
//!
//! See `docs/plans/echo-integration-plan.md` §2 and §B.1 for the design.
//!
//! A trajectory is an ordered sequence of [`TurnSegment`]s. Each segment
//! belongs to a [`TurnKind::Context`] (prompt scaffolding), [`TurnKind::Action`]
//! (assistant generation — target of policy gradient), or
//! [`TurnKind::Observation`] (tool result / environment — target of ECHO's
//! env-CE auxiliary loss).
//!
//! [`ScoredRollout`] attaches a reward to a trajectory. [`AgenticGroup`]
//! groups multiple scored rollouts that share a prompt — the unit of GRPO's
//! group-relative advantage normalization.
//!
//! ## Backwards compatibility
//!
//! The legacy `ScoredCompletion { text: String, reward: f64 }` and
//! `GrpoGroup { messages, completions }` payloads continue to deserialize.
//! `ScoredRollout` accepts the legacy `text` field via `#[serde(alias)]` and
//! falls back to a one-segment [`TurnKind::Action`] trajectory when only
//! `text` is supplied. `AgenticGroup::rollouts` accepts `completions` as a
//! serde alias.
//!
//! New emitters should populate `trajectory` directly.

use serde::{Deserialize, Serialize};

use crate::ChatMessage;

/// What kind of supervision applies to tokens inside a [`TurnSegment`].
///
/// - [`TurnKind::Context`] — system / user / non-trainable scaffolding;
///   no gradient flows through these tokens.
/// - [`TurnKind::Action`] — assistant-generated tokens; target of policy
///   gradient (and OPD's reverse-KL when configured).
/// - [`TurnKind::Observation`] — tool-result / environment tokens; target
///   of ECHO's env cross-entropy auxiliary loss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnKind {
    #[default]
    Context,
    Action,
    Observation,
}

/// One semantic turn in a trajectory.
///
/// `content` is the raw text the model emitted or saw, *before* chat-template
/// formatting is applied. For assistant tool-call turns this means the raw
/// `<tool_call>...</tool_call>` XML (Qwen XML form); for tool-result turns
/// it is the literal stdout / stderr / file-content bytes.
///
/// `kind` is what kind of supervision applies inside this segment;
/// [`TurnKind`] is the authoritative routing signal — `role` is informational
/// metadata that the chat template uses for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSegment {
    /// Chat role: `"system"` | `"user"` | `"assistant"` | `"tool"` (extensible).
    pub role: String,
    /// Raw content before chat-template formatting.
    pub content: String,
    /// What kind of supervision applies inside this segment.
    #[serde(default)]
    pub kind: TurnKind,
    /// Optional tool-call correlation ID, paired with the corresponding
    /// observation segment when available. Informational; not used by
    /// mask building today.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Bytes at the start of this segment's content that are harness
    /// warnings (e.g. `"WARNINGS:\n- bad tool call format\n"`). Excluded
    /// from `env_mask` when [`MaskConfig::warning_filter`] is true.
    /// Paper §3.2: warnings memorize in ~60 steps and stop providing useful
    /// gradient, while terminal-output tokens continue to teach.
    ///
    /// [`MaskConfig::warning_filter`]: crate::trajectory_mask::MaskConfig::warning_filter
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning_prefix_len: Option<usize>,
}

impl TurnSegment {
    /// Convenience: build a one-shot [`TurnKind::Action`] segment for a
    /// single-turn rollout (the legacy `text` shape).
    pub fn legacy_action(text: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content: text,
            kind: TurnKind::Action,
            tool_call_id: None,
            warning_prefix_len: None,
        }
    }
}

/// One scored rollout in a group.
///
/// In the canonical (new) form, [`Self::trajectory`] is populated and
/// [`Self::text`] is `None`. The legacy single-string form is supported via
/// the `text` field; deserialize-time logic synthesizes a one-segment
/// [`TurnKind::Action`] trajectory in [`Self::ensure_trajectory`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredRollout {
    /// Canonical multi-turn structure.
    #[serde(default)]
    pub trajectory: Vec<TurnSegment>,
    /// Outcome reward (paper convention: binary 0/1; kiln convention:
    /// continuous composite in `[0, 1]`).
    pub reward: f64,
    /// Legacy single-string completion text. When `trajectory` is empty and
    /// `text` is present, the rollout is treated as a one-segment Action
    /// trajectory. New emitters should populate `trajectory` directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl ScoredRollout {
    /// Synthesize a legacy text-only payload into a one-segment Action
    /// trajectory. Idempotent — does nothing if `trajectory` is already
    /// populated.
    pub fn ensure_trajectory(&mut self) {
        if self.trajectory.is_empty() {
            if let Some(text) = self.text.take() {
                self.trajectory.push(TurnSegment::legacy_action(text));
            }
        }
    }

    /// Convenience constructor for the legacy single-string form.
    pub fn legacy(text: String, reward: f64) -> Self {
        Self {
            trajectory: vec![TurnSegment::legacy_action(text)],
            reward,
            text: None,
        }
    }
}

/// A group of rollouts sharing a prompt. Group-relative advantage is computed
/// within this set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticGroup {
    /// Prompt seen by every rollout in this group (system + user).
    pub messages: Vec<ChatMessage>,
    /// Multiple scored rollouts from the same prompt. Accepts the legacy
    /// field name `completions` via serde alias.
    #[serde(alias = "completions")]
    pub rollouts: Vec<ScoredRollout>,
}

impl AgenticGroup {
    /// Materialize legacy single-string completions into Action trajectories
    /// across every rollout in this group. Idempotent.
    pub fn ensure_trajectories(&mut self) {
        for rollout in &mut self.rollouts {
            rollout.ensure_trajectory();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_scored_completion_text_round_trips() {
        let json = r#"{ "text": "the assistant said this", "reward": 0.85 }"#;
        let mut rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        assert_eq!(rollout.reward, 0.85);
        assert!(rollout.trajectory.is_empty());
        assert_eq!(rollout.text.as_deref(), Some("the assistant said this"));

        rollout.ensure_trajectory();
        assert_eq!(rollout.trajectory.len(), 1);
        assert_eq!(rollout.trajectory[0].role, "assistant");
        assert_eq!(rollout.trajectory[0].content, "the assistant said this");
        assert_eq!(rollout.trajectory[0].kind, TurnKind::Action);
        assert!(rollout.text.is_none());
    }

    #[test]
    fn canonical_trajectory_form_deserializes_unchanged() {
        let json = r#"{
            "trajectory": [
                {"role": "assistant", "content": "I will read the file", "kind": "action"},
                {"role": "tool", "content": "file contents here", "kind": "observation"}
            ],
            "reward": 1.0
        }"#;
        let rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        assert_eq!(rollout.trajectory.len(), 2);
        assert_eq!(rollout.trajectory[0].kind, TurnKind::Action);
        assert_eq!(rollout.trajectory[1].kind, TurnKind::Observation);
        assert_eq!(rollout.reward, 1.0);
    }

    #[test]
    fn agentic_group_accepts_legacy_completions_alias() {
        let json = r#"{
            "messages": [{"role": "user", "content": "hello"}],
            "completions": [
                {"text": "hi there", "reward": 0.7},
                {"text": "hello!", "reward": 0.9}
            ]
        }"#;
        let mut group: AgenticGroup = serde_json::from_str(json).unwrap();
        assert_eq!(group.rollouts.len(), 2);
        assert!(group.rollouts[0].trajectory.is_empty());

        group.ensure_trajectories();
        assert_eq!(group.rollouts[0].trajectory.len(), 1);
        assert_eq!(group.rollouts[0].trajectory[0].kind, TurnKind::Action);
        assert_eq!(group.rollouts[1].trajectory[0].content, "hello!");
    }

    #[test]
    fn agentic_group_accepts_canonical_rollouts_field() {
        let json = r#"{
            "messages": [{"role": "user", "content": "x"}],
            "rollouts": [
                {"trajectory": [{"role":"assistant","content":"y","kind":"action"}], "reward": 1.0}
            ]
        }"#;
        let group: AgenticGroup = serde_json::from_str(json).unwrap();
        assert_eq!(group.rollouts.len(), 1);
        assert_eq!(group.rollouts[0].trajectory[0].content, "y");
    }

    #[test]
    fn turn_kind_serde_uses_snake_case() {
        assert_eq!(
            serde_json::to_string(&TurnKind::Action).unwrap(),
            "\"action\""
        );
        assert_eq!(
            serde_json::to_string(&TurnKind::Observation).unwrap(),
            "\"observation\""
        );
        assert_eq!(
            serde_json::to_string(&TurnKind::Context).unwrap(),
            "\"context\""
        );
    }

    #[test]
    fn turn_segment_defaults_to_context_kind() {
        let json = r#"{ "role": "system", "content": "you are helpful" }"#;
        let seg: TurnSegment = serde_json::from_str(json).unwrap();
        assert_eq!(seg.kind, TurnKind::Context);
        assert!(seg.tool_call_id.is_none());
        assert!(seg.warning_prefix_len.is_none());
    }

    #[test]
    fn turn_segment_with_warning_prefix() {
        let json = r#"{
            "role": "tool",
            "content": "WARNINGS:\n- malformed call\n<command_output>ls: cannot access\n</command_output>",
            "kind": "observation",
            "warning_prefix_len": 26
        }"#;
        let seg: TurnSegment = serde_json::from_str(json).unwrap();
        assert_eq!(seg.warning_prefix_len, Some(26));
    }

    #[test]
    fn ensure_trajectory_is_idempotent() {
        let mut rollout = ScoredRollout::legacy("hi".to_string(), 1.0);
        let trajectory_len_before = rollout.trajectory.len();
        rollout.ensure_trajectory();
        rollout.ensure_trajectory();
        assert_eq!(rollout.trajectory.len(), trajectory_len_before);
    }

    #[test]
    fn legacy_completion_with_zero_reward_round_trips() {
        let json = r#"{ "text": "garbage", "reward": 0.0 }"#;
        let mut rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        rollout.ensure_trajectory();
        assert_eq!(rollout.reward, 0.0);
        assert_eq!(rollout.trajectory[0].content, "garbage");
    }
}
