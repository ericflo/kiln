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
//! Legacy `ScoredCompletion { text: String, reward: f64 }` payloads continue
//! to deserialize because `ScoredRollout` shares those field names. The new
//! optional `trajectory` field is populated only by trajectory-aware
//! emitters; when empty, the rollout behaves exactly like the pre-ECHO
//! single-string completion.
//!
//! `ScoredCompletion` and `GrpoGroup` remain valid type names via
//! `pub type` aliases in `crate::lib`, so call sites that reference them
//! don't need to change. The renames are deliberately *additive*: the same
//! struct, the same field names, plus an optional `trajectory` field.

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
    /// from `env_mask` when
    /// [`MaskConfig::warning_filter`] is true. Paper §3.2: warnings memorize
    /// in ~60 steps and stop providing useful gradient, while terminal-output
    /// tokens continue to teach.
    ///
    /// [`MaskConfig::warning_filter`]: crate::trajectory_mask::MaskConfig::warning_filter
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning_prefix_len: Option<usize>,
}

impl TurnSegment {
    /// Build a one-segment [`TurnKind::Action`] trajectory from a legacy
    /// `text` string. Used internally to bridge the legacy single-string
    /// completion form into the trajectory schema.
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
/// `text` is the legacy single-string completion text — populated for every
/// rollout, even when `trajectory` is set, so all existing `.text` field
/// access continues to work bit-identically. `reward` is the outcome reward
/// (paper convention: binary 0/1; kiln convention: continuous composite in
/// `[0, 1]`). `trajectory` is the optional canonical multi-turn shape: when
/// non-empty, the trajectory-aware mask builder (and ECHO) consume it; when
/// empty, the rollout is treated as a single-turn legacy completion.
///
/// New emitters should populate both `text` (for compatibility) and
/// `trajectory` (for ECHO). The plain-text `.text` of a multi-turn rollout
/// is conventionally a `<TURN_BREAK>`-joined flattening of the Action
/// segments, matching today's `rollout.py` output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoredRollout {
    /// Legacy single-string completion text. Always populated, even when
    /// `trajectory` is set, so call sites that read `.text` keep working.
    pub text: String,
    /// Outcome reward.
    pub reward: f64,
    /// Optional canonical multi-turn structure. When non-empty, ECHO and the
    /// trajectory-aware masking primitive consume this; the `text` field is
    /// informational/legacy.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trajectory: Vec<TurnSegment>,
}

impl ScoredRollout {
    /// Construct from the legacy `(text, reward)` shape — no trajectory.
    pub fn legacy(text: String, reward: f64) -> Self {
        Self {
            text,
            reward,
            trajectory: Vec::new(),
        }
    }

    /// Construct from a trajectory; populates `text` with a
    /// `<TURN_BREAK>`-joined flattening of the Action segments to match
    /// today's `rollout.py` output.
    pub fn from_trajectory(trajectory: Vec<TurnSegment>, reward: f64) -> Self {
        let text = flatten_action_segments(&trajectory);
        Self {
            text,
            reward,
            trajectory,
        }
    }

    /// True when this rollout carries a multi-turn trajectory ECHO can use.
    pub fn has_trajectory(&self) -> bool {
        !self.trajectory.is_empty()
    }

    /// Borrow the canonical trajectory if present; otherwise synthesize a
    /// one-segment Action trajectory from `text` for use by the masking
    /// primitive.
    ///
    /// Returns a `Cow` so the synthesized trajectory doesn't permanently
    /// mutate the rollout — callers that want to mutate should call
    /// [`Self::ensure_trajectory`] explicitly.
    pub fn effective_trajectory(&self) -> std::borrow::Cow<'_, [TurnSegment]> {
        if self.trajectory.is_empty() {
            std::borrow::Cow::Owned(vec![TurnSegment::legacy_action(self.text.clone())])
        } else {
            std::borrow::Cow::Borrowed(&self.trajectory)
        }
    }

    /// Mutate this rollout so `trajectory` is populated. Idempotent.
    pub fn ensure_trajectory(&mut self) {
        if self.trajectory.is_empty() {
            self.trajectory
                .push(TurnSegment::legacy_action(self.text.clone()));
        }
    }
}

fn flatten_action_segments(trajectory: &[TurnSegment]) -> String {
    let actions: Vec<&str> = trajectory
        .iter()
        .filter(|seg| matches!(seg.kind, TurnKind::Action))
        .map(|seg| seg.content.as_str())
        .collect();
    actions.join("<TURN_BREAK>")
}

/// A group of rollouts sharing a prompt. Group-relative advantage is computed
/// within this set.
///
/// The field name `completions` is preserved for backwards compatibility with
/// the pre-ECHO `GrpoGroup` shape; serde additionally accepts `rollouts` as
/// an alias so canonical forward-looking JSON also deserializes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgenticGroup {
    /// Prompt seen by every rollout in this group (system + user).
    pub messages: Vec<ChatMessage>,
    /// Multiple scored rollouts from the same prompt. Accepts canonical
    /// `rollouts` as a serde alias so new payloads can use either name.
    #[serde(alias = "rollouts")]
    pub completions: Vec<ScoredRollout>,
}

impl AgenticGroup {
    /// Ensure every rollout in the group has a populated `trajectory`. Used
    /// internally when the new mask-builder path is selected.
    pub fn ensure_trajectories(&mut self) {
        for rollout in &mut self.completions {
            rollout.ensure_trajectory();
        }
    }

    /// True when at least one rollout carries a multi-turn trajectory.
    pub fn has_any_trajectory(&self) -> bool {
        self.completions.iter().any(ScoredRollout::has_trajectory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_scored_completion_text_round_trips() {
        let json = r#"{ "text": "the assistant said this", "reward": 0.85 }"#;
        let rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        assert_eq!(rollout.reward, 0.85);
        assert_eq!(rollout.text, "the assistant said this");
        assert!(rollout.trajectory.is_empty());
        assert!(!rollout.has_trajectory());
    }

    #[test]
    fn canonical_trajectory_form_deserializes_with_text_and_trajectory() {
        let json = r#"{
            "text": "I will read the file<TURN_BREAK>The contents are X",
            "reward": 1.0,
            "trajectory": [
                {"role": "assistant", "content": "I will read the file", "kind": "action"},
                {"role": "tool", "content": "file contents here", "kind": "observation"},
                {"role": "assistant", "content": "The contents are X", "kind": "action"}
            ]
        }"#;
        let rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        assert_eq!(rollout.trajectory.len(), 3);
        assert_eq!(rollout.trajectory[0].kind, TurnKind::Action);
        assert_eq!(rollout.trajectory[1].kind, TurnKind::Observation);
        assert_eq!(rollout.reward, 1.0);
        assert!(rollout.has_trajectory());
    }

    #[test]
    fn agentic_group_accepts_legacy_completions_field_name() {
        let json = r#"{
            "messages": [{"role": "user", "content": "hello"}],
            "completions": [
                {"text": "hi there", "reward": 0.7},
                {"text": "hello!", "reward": 0.9}
            ]
        }"#;
        let group: AgenticGroup = serde_json::from_str(json).unwrap();
        assert_eq!(group.completions.len(), 2);
        assert_eq!(group.completions[0].text, "hi there");
        assert!(!group.has_any_trajectory());
    }

    #[test]
    fn agentic_group_accepts_canonical_rollouts_alias() {
        let json = r#"{
            "messages": [{"role": "user", "content": "x"}],
            "rollouts": [
                {
                    "text": "y",
                    "reward": 1.0,
                    "trajectory": [{"role":"assistant","content":"y","kind":"action"}]
                }
            ]
        }"#;
        let group: AgenticGroup = serde_json::from_str(json).unwrap();
        assert_eq!(group.completions.len(), 1);
        assert_eq!(group.completions[0].text, "y");
        assert!(group.has_any_trajectory());
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
        assert!(rollout.trajectory.is_empty());
        rollout.ensure_trajectory();
        rollout.ensure_trajectory();
        assert_eq!(rollout.trajectory.len(), 1);
        assert_eq!(rollout.trajectory[0].content, "hi");
        assert_eq!(rollout.trajectory[0].kind, TurnKind::Action);
    }

    #[test]
    fn legacy_completion_with_zero_reward_round_trips() {
        let json = r#"{ "text": "garbage", "reward": 0.0 }"#;
        let rollout: ScoredRollout = serde_json::from_str(json).unwrap();
        assert_eq!(rollout.reward, 0.0);
        assert_eq!(rollout.text, "garbage");
    }

    #[test]
    fn effective_trajectory_synthesizes_from_text_when_empty() {
        let rollout = ScoredRollout::legacy("only text".to_string(), 1.0);
        let traj = rollout.effective_trajectory();
        assert_eq!(traj.len(), 1);
        assert_eq!(traj[0].kind, TurnKind::Action);
        assert_eq!(traj[0].content, "only text");
        // Original rollout untouched.
        assert!(rollout.trajectory.is_empty());
    }

    #[test]
    fn from_trajectory_flattens_action_segments_to_text() {
        let traj = vec![
            TurnSegment {
                role: "assistant".into(),
                content: "first".into(),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "tool".into(),
                content: "ignored env output".into(),
                kind: TurnKind::Observation,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "assistant".into(),
                content: "second".into(),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
        ];
        let rollout = ScoredRollout::from_trajectory(traj, 0.5);
        assert_eq!(rollout.text, "first<TURN_BREAK>second");
        assert_eq!(rollout.trajectory.len(), 3);
        assert_eq!(rollout.reward, 0.5);
    }

    #[test]
    fn scored_rollout_default_is_empty() {
        let r = ScoredRollout::default();
        assert_eq!(r.text, "");
        assert_eq!(r.reward, 0.0);
        assert!(r.trajectory.is_empty());
    }
}
