//! Streaming text gates: incremental detokenization and stop-sequence
//! holdback.
//!
//! Two composable, allocation-light gates that sit between raw generated
//! token ids and the SSE wire:
//!
//! 1. [`IncrementalDetokenizer`] — HF/vLLM-style two-offset incremental
//!    decode. Fixes the U+FFFD mojibake that per-token decode produces
//!    when a multi-byte character spans token boundaries, and replaces the
//!    O(n²) full-prefix re-decode with a bounded window.
//! 2. [`StopTailGate`] — check-before-emit stop-sequence gating with a
//!    rolling tail holdback (the longest suffix that is still a proper
//!    prefix of any stop). The matched stop NEVER reaches the wire; pi's
//!    stop-marker parsers stop seeing phantom delimiters mid-stream.
//!
//! Both follow the server's `ReasoningSplitter` contract: `push` returns
//! what is safe to emit now, `flush` drains the holdback at end-of-stream
//! so no bytes are silently dropped on non-stop exits.

use kiln_core::token::TokenId;
use kiln_core::tokenizer::KilnTokenizer;

/// HF/vLLM two-offset incremental detokenizer.
///
/// Call [`Self::next_delta`] after appending each newly accepted token to
/// the full generated sequence. Text is emitted only once its UTF-8
/// characters are complete; a token that splits a multi-byte character
/// yields `""` and the character arrives whole with the next token.
#[derive(Debug, Default)]
pub struct IncrementalDetokenizer {
    /// Start of the decode window (tokens before this are already final).
    prefix_offset: usize,
    /// End of the already-emitted tokens within the window.
    read_offset: usize,
}

impl IncrementalDetokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The new complete-character text after the latest token, or `""`
    /// while the tail is an incomplete UTF-8 sequence.
    pub fn next_delta(&mut self, tokenizer: &KilnTokenizer, tokens: &[TokenId]) -> String {
        let prefix_text = tokenizer
            .decode(&tokens[self.prefix_offset..self.read_offset])
            .unwrap_or_default();
        let new_text = match tokenizer.decode(&tokens[self.prefix_offset..]) {
            Ok(t) => t,
            Err(_) => {
                // Decode failure: fall back to single-token decode (the
                // pre-gate behavior) and resync so nothing re-emits.
                self.prefix_offset = tokens.len().saturating_sub(1);
                self.read_offset = tokens.len();
                return tokenizer
                    .decode(&tokens[tokens.len().saturating_sub(1)..])
                    .unwrap_or_default();
            }
        };
        if new_text.len() <= prefix_text.len() || new_text.ends_with('\u{FFFD}') {
            // No growth, or the newest token split a multi-byte character —
            // hold until it completes.
            return String::new();
        }
        // Byte-level BPE keeps the decoded prefix stable; guard anyway
        // (mirrors the strip_prefix fallback the server used per token).
        let delta = match new_text.strip_prefix(prefix_text.as_str()) {
            Some(d) => d.to_string(),
            None => tokenizer
                .decode(&tokens[self.read_offset..])
                .unwrap_or_default(),
        };
        self.prefix_offset = self.read_offset;
        self.read_offset = tokens.len();
        delta
    }

    /// End-of-stream residual. May legitimately contain U+FFFD when
    /// generation stopped mid-character.
    pub fn flush(&mut self, tokenizer: &KilnTokenizer, tokens: &[TokenId]) -> String {
        let prefix_text = tokenizer
            .decode(&tokens[self.prefix_offset..self.read_offset])
            .unwrap_or_default();
        let new_text = tokenizer
            .decode(&tokens[self.prefix_offset..])
            .unwrap_or_default();
        self.prefix_offset = tokens.len();
        self.read_offset = tokens.len();
        new_text
            .strip_prefix(prefix_text.as_str())
            .unwrap_or("")
            .to_string()
    }
}

/// Result of pushing a delta through the [`StopTailGate`].
#[derive(Debug, Default)]
pub struct StopScan {
    /// Text safe to emit now (stop-clean).
    pub emit: String,
    /// The stop sequence that just matched, if any. After a match the
    /// gate self-mutes: every later push emits nothing.
    pub matched_stop: Option<String>,
}

/// Check-before-emit stop-sequence gate with a rolling tail holdback.
///
/// Invariant: the un-emitted tail always retains every suffix that could
/// still grow into a stop, so no proper prefix of a stop is ever emitted —
/// any stop that completes is found entirely within `pending` and the
/// truncation is exact.
#[derive(Debug)]
pub struct StopTailGate {
    stops: Vec<String>,
    pending: String,
    matched: Option<String>,
}

impl StopTailGate {
    pub fn new(stop_sequences: &[String]) -> Self {
        Self {
            // An empty stop would match instantly — filter, mirroring the
            // generation-side normalization.
            stops: stop_sequences
                .iter()
                .filter(|s| !s.is_empty())
                .cloned()
                .collect(),
            pending: String::new(),
            matched: None,
        }
    }

    pub fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// Append `delta` and return what is now safe to emit. The EARLIEST
    /// positional match wins (ties → first in the stop list), matching
    /// OpenAI semantics rather than list-order scanning.
    pub fn push(&mut self, delta: &str) -> StopScan {
        if self.matched.is_some() {
            return StopScan::default();
        }
        if self.stops.is_empty() {
            return StopScan {
                emit: delta.to_string(),
                matched_stop: None,
            };
        }
        self.pending.push_str(delta);

        // Earliest full occurrence across all stops.
        let mut best: Option<(usize, usize)> = None; // (byte idx, stop idx)
        for (si, s) in self.stops.iter().enumerate() {
            if let Some(i) = self.pending.find(s.as_str()) {
                if best.map(|(bi, _)| i < bi).unwrap_or(true) {
                    best = Some((i, si));
                }
            }
        }
        if let Some((idx, si)) = best {
            let emit = self.pending[..idx].to_string();
            let stop = self.stops[si].clone();
            self.pending.clear();
            self.matched = Some(stop.clone());
            return StopScan {
                emit,
                matched_stop: Some(stop),
            };
        }

        // Holdback: the longest suffix of `pending` that is a PROPER
        // prefix of any stop (char-boundary iteration).
        let mut hold = 0usize;
        for s in &self.stops {
            for k in (1..s.len()).rev() {
                if k > self.pending.len() || k <= hold {
                    continue;
                }
                if s.is_char_boundary(k) && self.pending.ends_with(&s[..k]) {
                    hold = k;
                    break;
                }
            }
        }
        let emit_end = self.pending.len() - hold;
        let emit = self.pending[..emit_end].to_string();
        self.pending.drain(..emit_end);
        StopScan {
            emit,
            matched_stop: None,
        }
    }

    /// Drain the holdback at end-of-stream when no stop matched (`""`
    /// after a match — the held text WAS the stop).
    pub fn flush(&mut self) -> String {
        if self.matched.is_some() {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fixture_tokenizer() -> KilnTokenizer {
        // Byte-level BPE with a vocab that includes a CJK char split
        // across two tokens: "好" = E5 A5 BD; GPT-2 byte alphabet maps
        // E5→"å", A5→"¥", BD→"½".
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("A".into(), 0);
        vocab.insert("ĠB".into(), 1); // " B"
        vocab.insert("å¥".into(), 2); // bytes E5 A5 (first 2/3 of 好)
        vocab.insert("½".into(), 3); // byte BD (last 1/3 of 好)
        vocab.insert("ĠObservation".into(), 4);
        vocab.insert(":".into(), 5);
        vocab.insert("ĠC".into(), 6);
        let json = serde_json::json!({
            "version": "1.0",
            "model": { "type": "BPE", "vocab": vocab, "merges": [] },
            "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
            "added_tokens": []
        });
        KilnTokenizer::from_bytes(&serde_json::to_vec(&json).unwrap()).unwrap()
    }

    /// CJK split-token: the first token ends mid-character — nothing
    /// emits (no U+FFFD ever); the character arrives whole with the
    /// second token.
    #[test]
    fn detokenizer_holds_split_multibyte_chars() {
        let tok = fixture_tokenizer();
        let mut d = IncrementalDetokenizer::new();
        let mut tokens: Vec<TokenId> = Vec::new();

        tokens.push(2); // E5 A5 — incomplete
        let delta = d.next_delta(&tok, &tokens);
        assert_eq!(delta, "", "mid-character bytes must be held");

        tokens.push(3); // BD — completes 好
        let delta = d.next_delta(&tok, &tokens);
        assert_eq!(delta, "好");
        assert!(!delta.contains('\u{FFFD}'));

        // Clean end: nothing left.
        assert_eq!(d.flush(&tok, &tokens), "");
    }

    #[test]
    fn detokenizer_flush_returns_residue_when_stream_ends_mid_char() {
        let tok = fixture_tokenizer();
        let mut d = IncrementalDetokenizer::new();
        let tokens: Vec<TokenId> = vec![0, 2]; // "A" + incomplete bytes
        assert_eq!(d.next_delta(&tok, &tokens[..1]), "A");
        assert_eq!(d.next_delta(&tok, &tokens), "", "held");
        let residue = d.flush(&tok, &tokens);
        assert!(
            residue.contains('\u{FFFD}'),
            "mid-char end legitimately surfaces the replacement char: {residue:?}"
        );
    }

    #[test]
    fn detokenizer_ascii_passthrough() {
        let tok = fixture_tokenizer();
        let mut d = IncrementalDetokenizer::new();
        let mut tokens: Vec<TokenId> = Vec::new();
        let mut out = String::new();
        for t in [0u32, 1, 6] {
            tokens.push(t);
            out.push_str(&d.next_delta(&tok, &tokens));
        }
        assert_eq!(out, "A B C");
    }

    /// The required stop="Observation:" case: the marker never reaches
    /// the emitted stream, split across deltas or in one delta.
    #[test]
    fn stop_gate_observation_marker_never_emits() {
        let mut g = StopTailGate::new(&["Observation:".to_string()]);
        let mut out = String::new();
        let s1 = g.push("I should check\nObs");
        out.push_str(&s1.emit);
        assert!(s1.matched_stop.is_none());
        let s2 = g.push("ervation:");
        out.push_str(&s2.emit);
        assert_eq!(s2.matched_stop.as_deref(), Some("Observation:"));
        // Post-match pushes are muted.
        let s3 = g.push(" extra");
        assert_eq!(s3.emit, "");
        assert_eq!(out, "I should check\n");
        assert_eq!(g.flush(), "", "the held text WAS the stop");

        // Single-delta variant.
        let mut g = StopTailGate::new(&["Observation:".to_string()]);
        let s = g.push("text before\nObservation: the file");
        assert_eq!(s.emit, "text before\n");
        assert_eq!(s.matched_stop.as_deref(), Some("Observation:"));
    }

    #[test]
    fn stop_gate_false_alarm_releases_all_bytes() {
        let mut g = StopTailGate::new(&["Observation:".to_string()]);
        let mut out = String::new();
        out.push_str(&g.push("Observ").emit);
        out.push_str(&g.push("able systems").emit);
        out.push_str(&g.flush());
        assert_eq!(out, "Observable systems");
    }

    #[test]
    fn stop_gate_earliest_match_wins_and_multibyte_stops_hold_on_boundaries() {
        let mut g = StopTailGate::new(&["ZZ".to_string(), "B".to_string()]);
        let s = g.push("A B then ZZ");
        assert_eq!(s.emit, "A ");
        assert_eq!(s.matched_stop.as_deref(), Some("B"), "earliest position wins");

        // Multi-byte stop: holdback never splits a char boundary.
        let mut g = StopTailGate::new(&["。end".to_string()]);
        let s = g.push("sentence。");
        assert!(s.matched_stop.is_none());
        assert_eq!(s.emit, "sentence");
        let s2 = g.push("end");
        assert_eq!(s2.matched_stop.as_deref(), Some("。end"));
    }

    #[test]
    fn stop_gate_empty_stops_filtered_and_inactive_passthrough() {
        let mut g = StopTailGate::new(&["".to_string()]);
        assert!(!g.is_active());
        let s = g.push("anything");
        assert_eq!(s.emit, "anything");
        assert!(s.matched_stop.is_none());
    }

    /// Property: while the gate is live, no emitted prefix could have
    /// been the start of a stop that later completes.
    #[test]
    fn stop_gate_never_emits_a_live_stop_prefix() {
        let stop = "</tool_call>".to_string();
        let mut g = StopTailGate::new(&[stop.clone()]);
        let mut out = String::new();
        for chunk in ["abc</to", "ol_", "call>tail"] {
            out.push_str(&g.push(chunk).emit);
        }
        assert_eq!(out, "abc");
        assert_eq!(g.matched(), Some("</tool_call>"));
    }
}
