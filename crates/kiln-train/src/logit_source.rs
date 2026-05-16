//! Logit-source abstraction for on-policy distillation.
//!
//! See `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`
//! §3.2 for the design. Every concrete teacher — local model, hosted HTTP
//! API, on-disk cache, federated peer — implements the [`LogitSource`]
//! trait. The trainer asks the source what it can deliver via
//! [`LogitSource::capabilities`] and picks the OPD loss granularity from
//! there: `full_vocab` when a teacher can produce full-vocab logprobs,
//! `teacher_top_k` otherwise.
//!
//! The trait is **synchronous** by design. Kiln's training loop already
//! runs inside a `tokio::task::spawn_blocking` worker — making the trait
//! async would force every implementor to either run a nested runtime or
//! use `async-trait`, both of which would complicate the call site without
//! buying performance. Async I/O (e.g. hitting OpenRouter) is wrapped via
//! `tokio::runtime::Handle::block_on` inside the implementation, exactly
//! like the existing kiln-eval HTTP scorer pattern.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Capability self-description for a [`LogitSource`].
///
/// The trainer reads this once at the start of an OPD run to choose the
/// loss granularity (`sampled_token` / `teacher_top_k` / `full_vocab`)
/// and to validate that the user-requested top-K is achievable. See
/// §3.2 of the grand plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitSourceCaps {
    /// Stable identifier for the teacher this source serves
    /// (e.g. `"qwen3.6-27b@local"`, `"qwen/qwen-3.6-27b@openrouter"`).
    /// Used in cache keys (§3.3) and reproducibility receipts (§8.11).
    pub teacher_id: String,

    /// Full vocabulary size of the teacher. Provided so the trainer can
    /// validate that `teacher_topk_indices` from
    /// [`LogitSource::fetch_logprobs`] are in range.
    pub vocab_size: usize,

    /// Maximum top-K the source can return. Many hosted-API teachers cap
    /// at 5–20; a local teacher returns up to `vocab_size`. Callers
    /// requesting more than this from `fetch_logprobs` get a
    /// [`LogitSourceError::TopKExceedsCap`].
    pub max_top_k: usize,

    /// `true` when the source can return the full vocabulary's logprobs
    /// at each position. Required for the DeepSeek-V4-style
    /// `full_vocab` loss path (§5.1.2 of the deepseek_v4 paper).
    pub supports_full_vocab: bool,

    /// `true` when the source can be queried at every position of a
    /// trajectory in one batched call (rather than per-token). Local
    /// teachers and vLLM/sglang APIs are batched; some hosted endpoints
    /// only return the assistant tokens.
    pub supports_batched: bool,

    /// Tokenizer hash. Used to detect silent tokenizer drift between
    /// teacher and student when the kiln student is Qwen3.5-4B and the
    /// teacher is *supposed* to share that tokenizer (e.g. Qwen3.6-27B).
    /// Mismatch triggers a guardrail before the first request is sent.
    pub tokenizer_hash: Option<String>,
}

/// Per-position top-K logprobs returned by a [`LogitSource`].
///
/// `indices` and `logprobs` are both flattened in row-major order, length
/// `num_positions * top_k`. The position order matches the `positions`
/// slice passed to [`LogitSource::fetch_logprobs`]. `logprobs` are full-
/// vocab `log_softmax` values at the indexed positions (NOT renormalised
/// over the K support — the OPD loss kernel does that renormalisation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKLogprobs {
    /// Flattened `[num_positions, top_k]` u32 vocabulary indices.
    pub indices: Vec<u32>,
    /// Flattened `[num_positions, top_k]` f32 log-probabilities at those
    /// indices.
    pub logprobs: Vec<f32>,
    /// `K` — the support size. `indices.len() == positions.len() * top_k`.
    pub top_k: usize,
}

/// Output of a logprob query — either top-K (the §3.2 default for hosted
/// teachers) or full-vocab (used by §5.1.2 corporate-tier multi-teacher
/// consolidation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogprobBatch {
    /// Top-K logprobs at each queried position.
    TopK(TopKLogprobs),
    /// Full-vocab logprobs at each queried position. `logprobs` is
    /// `[num_positions, vocab_size]` flattened row-major.
    FullVocab {
        logprobs: Vec<f32>,
        vocab_size: usize,
    },
}

impl LogprobBatch {
    /// Number of (position, k) entries.
    pub fn flat_len(&self) -> usize {
        match self {
            Self::TopK(t) => t.indices.len(),
            Self::FullVocab { logprobs, .. } => logprobs.len(),
        }
    }

    /// `K` for top-K, vocab_size for full-vocab.
    pub fn support_size(&self) -> usize {
        match self {
            Self::TopK(t) => t.top_k,
            Self::FullVocab { vocab_size, .. } => *vocab_size,
        }
    }
}

/// Errors a [`LogitSource`] can return.
#[derive(Debug, Error)]
pub enum LogitSourceError {
    /// The caller requested top-K higher than the source's `max_top_k`.
    /// Mitigation: the trainer should clamp to `caps.max_top_k` or fall
    /// back to a `sampled_token` loss (§3.1).
    #[error("top_k {requested} exceeds source cap {cap} for teacher {teacher_id:?}")]
    TopKExceedsCap {
        requested: usize,
        cap: usize,
        teacher_id: String,
    },

    /// Caller requested full-vocab logprobs from a source that doesn't
    /// support them.
    #[error("source {teacher_id:?} does not support full-vocab logprobs")]
    FullVocabUnsupported { teacher_id: String },

    /// Tokenizer hash mismatch detected. Critical guardrail (§3.9
    /// `TokenizerDrift` rule).
    #[error(
        "tokenizer drift: source {teacher_id:?} expected hash {expected:?}, got {actual:?}"
    )]
    TokenizerDrift {
        teacher_id: String,
        expected: String,
        actual: String,
    },

    /// Generic transport/I/O failure. Carries the underlying error.
    #[error("source {teacher_id:?} transport error: {message}")]
    Transport { teacher_id: String, message: String },

    /// Catch-all for invalid-input cases (mismatched shapes, etc.).
    #[error("source {teacher_id:?} invalid input: {message}")]
    Invalid { teacher_id: String, message: String },
}

impl LogitSourceError {
    pub fn transport(teacher_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Transport {
            teacher_id: teacher_id.into(),
            message: message.into(),
        }
    }

    pub fn invalid(teacher_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Invalid {
            teacher_id: teacher_id.into(),
            message: message.into(),
        }
    }
}

/// A source of teacher logprobs for on-policy distillation.
///
/// The contract:
///
/// 1. The trainer constructs a [`LogitSource`] implementation once at the
///    start of an OPD run.
/// 2. For each rollout batch, the trainer calls
///    [`LogitSource::fetch_logprobs`] with the rollout's token IDs and
///    the positions where it needs teacher signal (typically the
///    assistant tokens; prompt tokens are masked).
/// 3. The source returns a [`LogprobBatch`] — either top-K (the §3.2
///    default) or full-vocab (corporate-tier). The order of entries
///    matches the order of `positions`.
///
/// Implementations:
///
/// * [`LocalTeacher`] — loads a second model into the same kiln process
///   (or attaches to one already loaded), runs a forward pass, extracts
///   top-K via `torch.topk`-equivalent on the resulting logits. Default
///   for the prosumer / corporate tiers (§2.2 / §2.3).
/// * `RemoteTeacher` (next milestone) — speaks the OpenAI-compatible
///   `top_logprobs` schema. Adapters per provider (vLLM, sglang,
///   llama.cpp, OpenRouter, Together, Fireworks, …).
/// * `CachedTeacher` (next milestone) — wraps another source, answers
///   from local + community logit cache before falling through (§3.3).
pub trait LogitSource: Send + Sync + Debug {
    /// Self-describing capability set. Called once at the start of an
    /// OPD run to pick the loss granularity.
    fn capabilities(&self) -> LogitSourceCaps;

    /// Fetch teacher logprobs at the given positions in the given
    /// sequence.
    ///
    /// * `tokens` — full tokenized rollout including the prompt prefix.
    ///   The source produces logits for *every* position in `tokens` but
    ///   only returns logprobs at the positions in `positions`. This
    ///   matches the kiln trainer convention (prompt tokens never
    ///   contribute to the loss).
    /// * `positions` — 0-indexed positions in `tokens` where the trainer
    ///   wants teacher signal. Typically the assistant-token positions.
    /// * `top_k` — `Some(K)` requests top-K logprobs; `None` requests
    ///   full-vocab logprobs (only valid if `caps.supports_full_vocab`).
    ///
    /// Returns a [`LogprobBatch`] in `positions` order.
    fn fetch_logprobs(
        &self,
        tokens: &[u32],
        positions: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch, LogitSourceError>;
}

// ---------------------------------------------------------------------------
// LocalTeacher
// ---------------------------------------------------------------------------
//
// The local-teacher implementation loads a second model (typically
// Qwen3.6-27B) into the same kiln process — or accepts an already-loaded
// model handle — and runs a forward pass to produce per-position logits.
// Top-K extraction happens on the device.
//
// Milestone-2 scope: a *placeholder* `LocalTeacher` that the trainer can
// construct in tests without a real GPU model. The real
// `Model + Tokenizer + KV-cache` integration lands when the
// `/v1/train/opd` endpoint is wired in #22, because the server-side
// model loader is what owns the second model. The placeholder here lets
// the trainer scaffolding compile and be unit-tested against a known
// in-memory fixture.

/// A logit source backed by a fixed in-memory dictionary of
/// (position-key, top-K logprobs) entries. Used by trainer unit tests
/// and as the reference for the §3.3 `CachedTeacher` design — the
/// production `LocalTeacher` (and `RemoteTeacher`) will produce data
/// in the same shape.
#[derive(Debug, Clone)]
pub struct FixtureLogitSource {
    caps: LogitSourceCaps,
    /// For each (tokens hash, position) we precompute the answer.
    /// Indexing convention: the trainer queries `(tokens, positions)` and
    /// we look up each `positions[i]` keyed by `(hash, positions[i])`.
    entries: std::collections::HashMap<(u64, usize), (Vec<u32>, Vec<f32>)>,
    top_k: usize,
}

impl FixtureLogitSource {
    /// Build a fixture source where every (tokens, position) query
    /// returns the same uniform-over-K top-K. Useful as a smoke fixture
    /// for the trainer wiring.
    pub fn uniform_topk(
        teacher_id: impl Into<String>,
        vocab_size: usize,
        top_k: usize,
    ) -> Self {
        Self {
            caps: LogitSourceCaps {
                teacher_id: teacher_id.into(),
                vocab_size,
                max_top_k: top_k,
                supports_full_vocab: false,
                supports_batched: true,
                tokenizer_hash: None,
            },
            entries: std::collections::HashMap::new(),
            top_k,
        }
    }

    /// Insert an entry: at position `pos` of any tokens whose hash is
    /// `tokens_hash`, return the given top-K indices and logprobs.
    pub fn insert(
        &mut self,
        tokens_hash: u64,
        pos: usize,
        indices: Vec<u32>,
        logprobs: Vec<f32>,
    ) {
        assert_eq!(indices.len(), self.top_k);
        assert_eq!(logprobs.len(), self.top_k);
        self.entries.insert((tokens_hash, pos), (indices, logprobs));
    }

    /// Compute the simple FNV-style hash of a `&[u32]` we use as the
    /// fixture key. Public so tests can produce the same hash.
    pub fn hash_tokens(tokens: &[u32]) -> u64 {
        // FNV-1a 64-bit
        let mut h: u64 = 0xcbf29ce484222325;
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

impl LogitSource for FixtureLogitSource {
    fn capabilities(&self) -> LogitSourceCaps {
        self.caps.clone()
    }

    fn fetch_logprobs(
        &self,
        tokens: &[u32],
        positions: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch, LogitSourceError> {
        let teacher_id = self.caps.teacher_id.clone();
        let requested_k = top_k.unwrap_or(self.top_k);
        if requested_k > self.caps.max_top_k {
            return Err(LogitSourceError::TopKExceedsCap {
                requested: requested_k,
                cap: self.caps.max_top_k,
                teacher_id,
            });
        }
        if top_k.is_none() && !self.caps.supports_full_vocab {
            return Err(LogitSourceError::FullVocabUnsupported { teacher_id });
        }

        let tokens_hash = Self::hash_tokens(tokens);
        let mut indices = Vec::with_capacity(positions.len() * requested_k);
        let mut logprobs = Vec::with_capacity(positions.len() * requested_k);
        for &pos in positions {
            let entry = self.entries.get(&(tokens_hash, pos)).ok_or_else(|| {
                LogitSourceError::invalid(
                    &teacher_id,
                    format!("no fixture entry for (hash={tokens_hash:#x}, pos={pos})"),
                )
            })?;
            indices.extend_from_slice(&entry.0[..requested_k]);
            logprobs.extend_from_slice(&entry.1[..requested_k]);
        }
        Ok(LogprobBatch::TopK(TopKLogprobs {
            indices,
            logprobs,
            top_k: requested_k,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_returns_inserted_entries() {
        let mut src = FixtureLogitSource::uniform_topk("test-teacher", 64, 4);
        let tokens = vec![10u32, 20, 30, 40];
        let h = FixtureLogitSource::hash_tokens(&tokens);
        src.insert(h, 1, vec![5, 6, 7, 8], vec![-0.1, -0.2, -0.3, -0.4]);
        src.insert(h, 2, vec![9, 10, 11, 12], vec![-0.5, -0.6, -0.7, -0.8]);

        let batch = src.fetch_logprobs(&tokens, &[1, 2], Some(4)).unwrap();
        match batch {
            LogprobBatch::TopK(t) => {
                assert_eq!(t.top_k, 4);
                assert_eq!(t.indices, vec![5, 6, 7, 8, 9, 10, 11, 12]);
                assert_eq!(
                    t.logprobs,
                    vec![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
                );
            }
            _ => panic!("expected top-k batch"),
        }
    }

    #[test]
    fn fixture_rejects_overlarge_k() {
        let src = FixtureLogitSource::uniform_topk("test-teacher", 64, 4);
        let err = src
            .fetch_logprobs(&[1, 2, 3], &[1], Some(8))
            .unwrap_err();
        match err {
            LogitSourceError::TopKExceedsCap {
                requested,
                cap,
                ..
            } => {
                assert_eq!(requested, 8);
                assert_eq!(cap, 4);
            }
            _ => panic!("expected TopKExceedsCap"),
        }
    }

    #[test]
    fn caps_round_trip_through_serde() {
        let caps = LogitSourceCaps {
            teacher_id: "qwen3.6-27b@local".into(),
            vocab_size: 152_064,
            max_top_k: 32,
            supports_full_vocab: true,
            supports_batched: true,
            tokenizer_hash: Some("sha256:abcd".into()),
        };
        let json = serde_json::to_string(&caps).unwrap();
        let parsed: LogitSourceCaps = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.teacher_id, caps.teacher_id);
        assert_eq!(parsed.max_top_k, caps.max_top_k);
        assert!(parsed.supports_full_vocab);
    }
}
