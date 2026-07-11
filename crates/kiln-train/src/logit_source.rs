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
    /// Mitigation: the trainer should clamp to `caps.max_top_k` or use a
    /// source with a wider validated support.
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
    #[error("tokenizer drift: source {teacher_id:?} expected hash {expected:?}, got {actual:?}")]
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

/// Validate the request-side invariants shared by every [`LogitSource`].
///
/// Duplicate and out-of-order positions are valid: implementations must return
/// one row for each requested position in exactly the caller-provided order.
pub fn validate_logit_request(
    caps: &LogitSourceCaps,
    tokens: &[u32],
    positions: &[usize],
    top_k: Option<usize>,
) -> Result<(), LogitSourceError> {
    let invalid = |message: String| LogitSourceError::invalid(&caps.teacher_id, message);

    if caps.vocab_size == 0 {
        return Err(invalid("vocab_size must be greater than zero".into()));
    }
    if tokens.is_empty() {
        return Err(invalid("tokens must not be empty".into()));
    }
    for (token_index, &token_id) in tokens.iter().enumerate() {
        if token_id as usize >= caps.vocab_size {
            return Err(invalid(format!(
                "token {token_index} has id {token_id}, outside vocab_size {}",
                caps.vocab_size
            )));
        }
    }
    for (row_index, &position) in positions.iter().enumerate() {
        if position >= tokens.len() {
            return Err(invalid(format!(
                "positions[{row_index}]={position} is outside token length {}",
                tokens.len()
            )));
        }
    }

    match top_k {
        Some(0) => Err(invalid("top_k must be greater than zero".into())),
        Some(requested) if requested > caps.max_top_k => Err(LogitSourceError::TopKExceedsCap {
            requested,
            cap: caps.max_top_k,
            teacher_id: caps.teacher_id.clone(),
        }),
        Some(requested) if requested > caps.vocab_size => Err(invalid(format!(
            "top_k {requested} exceeds vocab_size {}",
            caps.vocab_size
        ))),
        Some(_) => Ok(()),
        None if !caps.supports_full_vocab => Err(LogitSourceError::FullVocabUnsupported {
            teacher_id: caps.teacher_id.clone(),
        }),
        None => Ok(()),
    }
}

/// Validate one sparse logprob row against a source's advertised vocabulary.
///
/// The support must have exactly `expected_top_k` unique token IDs. Values are
/// full-vocabulary logprobs, so every value is finite and non-positive and the
/// probability mass represented by the sparse support cannot exceed one.
pub fn validate_topk_logprob_row(
    caps: &LogitSourceCaps,
    expected_top_k: usize,
    row_index: usize,
    indices: &[u32],
    logprobs: &[f32],
) -> Result<(), LogitSourceError> {
    let invalid = |message| LogitSourceError::invalid(&caps.teacher_id, message);

    if expected_top_k == 0 {
        return Err(invalid(format!("row {row_index} declares top_k 0")));
    }
    if expected_top_k > caps.max_top_k {
        return Err(invalid(format!(
            "row {row_index} top_k {expected_top_k} exceeds source cap {}",
            caps.max_top_k
        )));
    }
    if expected_top_k > caps.vocab_size {
        return Err(invalid(format!(
            "row {row_index} top_k {expected_top_k} exceeds vocab_size {}",
            caps.vocab_size
        )));
    }
    if indices.len() != expected_top_k || logprobs.len() != expected_top_k {
        return Err(invalid(format!(
            "row {row_index} declares top_k {expected_top_k} but has {} indices and {} logprobs",
            indices.len(),
            logprobs.len()
        )));
    }

    let mut seen = std::collections::HashSet::with_capacity(expected_top_k);
    let mut probability_mass = 0.0f64;
    let mut previous_logprob = None;
    for (candidate_index, (&token_id, &logprob)) in indices.iter().zip(logprobs.iter()).enumerate()
    {
        if token_id as usize >= caps.vocab_size {
            return Err(invalid(format!(
                "row {row_index} candidate {candidate_index} has token id {token_id}, outside vocab_size {}",
                caps.vocab_size
            )));
        }
        if !seen.insert(token_id) {
            return Err(invalid(format!(
                "row {row_index} contains duplicate token id {token_id}"
            )));
        }
        if !logprob.is_finite() || logprob > 0.0 {
            return Err(invalid(format!(
                "row {row_index} candidate {candidate_index} has invalid logprob {logprob}"
            )));
        }
        if let Some(previous) = previous_logprob {
            if logprob > previous {
                return Err(invalid(format!(
                    "row {row_index} logprobs are not non-increasing at candidate {candidate_index}: {logprob} follows {previous}"
                )));
            }
        }
        previous_logprob = Some(logprob);
        probability_mass += (logprob as f64).exp();
    }

    const MASS_TOLERANCE: f64 = 32.0 * f32::EPSILON as f64;
    if probability_mass > 1.0 + MASS_TOLERANCE {
        return Err(invalid(format!(
            "row {row_index} sparse probability mass {probability_mass:.9} exceeds 1"
        )));
    }
    Ok(())
}

/// Validate the exact shape and every row of a top-K logprob response.
///
/// A batch contains exactly one row per item in `positions`; this is the
/// structural part of the position-order contract. Implementations remain
/// responsible for placing each semantic row in the corresponding slot.
pub fn validate_topk_logprobs_batch(
    caps: &LogitSourceCaps,
    tokens: &[u32],
    positions: &[usize],
    requested_top_k: usize,
    batch: &TopKLogprobs,
) -> Result<(), LogitSourceError> {
    validate_logit_request(caps, tokens, positions, Some(requested_top_k))?;
    if batch.top_k != requested_top_k {
        return Err(LogitSourceError::invalid(
            &caps.teacher_id,
            format!(
                "response declares top_k {}, expected {requested_top_k}",
                batch.top_k
            ),
        ));
    }
    let flat_len = positions
        .len()
        .checked_mul(requested_top_k)
        .ok_or_else(|| {
            LogitSourceError::invalid(&caps.teacher_id, "top-K response shape overflows usize")
        })?;
    if batch.indices.len() != flat_len || batch.logprobs.len() != flat_len {
        return Err(LogitSourceError::invalid(
            &caps.teacher_id,
            format!(
                "response for {} positions at top_k {requested_top_k} requires {flat_len} values, got {} indices and {} logprobs",
                positions.len(),
                batch.indices.len(),
                batch.logprobs.len()
            ),
        ));
    }

    for row_index in 0..positions.len() {
        let start = row_index * requested_top_k;
        let end = start + requested_top_k;
        validate_topk_logprob_row(
            caps,
            requested_top_k,
            row_index,
            &batch.indices[start..end],
            &batch.logprobs[start..end],
        )?;
    }
    Ok(())
}

/// Validate a full-vocabulary logprob response, including normalization.
pub fn validate_full_vocab_logprobs_batch(
    caps: &LogitSourceCaps,
    tokens: &[u32],
    positions: &[usize],
    batch: &LogprobBatch,
) -> Result<(), LogitSourceError> {
    validate_logit_request(caps, tokens, positions, None)?;
    let LogprobBatch::FullVocab {
        logprobs,
        vocab_size,
    } = batch
    else {
        return Err(LogitSourceError::invalid(
            &caps.teacher_id,
            "full-vocab request returned a top-K response",
        ));
    };
    if *vocab_size != caps.vocab_size {
        return Err(LogitSourceError::invalid(
            &caps.teacher_id,
            format!(
                "full-vocab response declares vocab_size {vocab_size}, expected {}",
                caps.vocab_size
            ),
        ));
    }
    let flat_len = positions
        .len()
        .checked_mul(caps.vocab_size)
        .ok_or_else(|| {
            LogitSourceError::invalid(
                &caps.teacher_id,
                "full-vocab response shape overflows usize",
            )
        })?;
    if logprobs.len() != flat_len {
        return Err(LogitSourceError::invalid(
            &caps.teacher_id,
            format!(
                "full-vocab response for {} positions requires {flat_len} values, got {}",
                positions.len(),
                logprobs.len()
            ),
        ));
    }

    const MASS_TOLERANCE: f64 = 64.0 * f32::EPSILON as f64;
    for (row_index, row) in logprobs.chunks_exact(caps.vocab_size).enumerate() {
        let mut probability_mass = 0.0f64;
        for (token_id, &logprob) in row.iter().enumerate() {
            if !logprob.is_finite() || logprob > 0.0 {
                return Err(LogitSourceError::invalid(
                    &caps.teacher_id,
                    format!(
                        "full-vocab row {row_index} token {token_id} has invalid logprob {logprob}"
                    ),
                ));
            }
            probability_mass += (logprob as f64).exp();
        }
        if (probability_mass - 1.0).abs() > MASS_TOLERANCE {
            return Err(LogitSourceError::invalid(
                &caps.teacher_id,
                format!(
                    "full-vocab row {row_index} probability mass {probability_mass:.9} is not normalized"
                ),
            ));
        }
    }
    Ok(())
}

/// Convert target-token positions into the causal logits rows that predict them.
///
/// A target token at position `q` is scored by logits row `q - 1`. Callers must
/// provide a strictly increasing, unique list of targets that exist in the
/// original token sequence; token zero has no preceding causal row.
pub fn target_token_positions_to_logits_rows(
    teacher_id: &str,
    token_len: usize,
    target_positions: &[usize],
) -> Result<Vec<usize>, LogitSourceError> {
    let mut logits_rows = Vec::with_capacity(target_positions.len());
    let mut previous = None;
    for (index, &target_position) in target_positions.iter().enumerate() {
        if target_position == 0 {
            return Err(LogitSourceError::invalid(
                teacher_id,
                format!("target_positions[{index}] is zero and has no preceding logits row"),
            ));
        }
        if target_position >= token_len {
            return Err(LogitSourceError::invalid(
                teacher_id,
                format!(
                    "target_positions[{index}]={target_position} is outside token length {token_len}"
                ),
            ));
        }
        if let Some(previous) = previous {
            if target_position <= previous {
                return Err(LogitSourceError::invalid(
                    teacher_id,
                    format!(
                        "target positions must be strictly increasing and unique: {target_position} follows {previous}"
                    ),
                ));
            }
        }
        previous = Some(target_position);
        logits_rows.push(target_position - 1);
    }
    Ok(logits_rows)
}

/// A source of teacher logprobs for on-policy distillation.
///
/// The contract:
///
/// 1. The trainer constructs a [`LogitSource`] implementation once at the
///    start of an OPD run.
/// 2. For each rollout batch, the trainer calls
///    [`LogitSource::fetch_logprobs`] with the rollout's token IDs and
///    the causal logits rows that predict its active target tokens. Callers
///    convert a target-token index `q` to row `q - 1` before this boundary.
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
/// * `RemoteTeacher` speaks the validated OpenAI-compatible prompt-logprob
///   schema for explicitly supported providers.
/// * `CachedTeacher` wraps another source and answers from the local logit
///   cache before falling through (§3.3).
pub trait LogitSource: Send + Sync + Debug {
    /// Self-describing capability set. Called once at the start of an
    /// OPD run to pick the loss granularity.
    fn capabilities(&self) -> LogitSourceCaps;

    /// Complete content/protocol identity when this source can prove one.
    /// Cache and receipt code must treat `None` as unverified rather than
    /// reconstructing an identity from aliases or legacy capability fields.
    fn authoritative_teacher_identity(&self) -> Option<&crate::TeacherIdentityV1> {
        None
    }

    /// Stable identity of the exact numeric source used by training.
    ///
    /// Model-backed sources inherit the canonical teacher revision. Fixed
    /// fixtures override this with a digest of their scored rows, which lets
    /// exact resume verify composite and precomputed teachers without
    /// pretending they are one model identity.
    fn authoritative_content_revision(&self) -> Option<String> {
        self.authoritative_teacher_identity()
            .map(|identity| format!("sha256:{}", identity.content_revision()))
    }

    /// Fetch teacher logprobs at the given positions in the given
    /// sequence.
    ///
    /// * `tokens` — full tokenized rollout including the prompt prefix.
    ///   The source produces logits for *every* position in `tokens` but
    ///   only returns logprobs at the positions in `positions`. This
    ///   matches the kiln trainer convention (prompt tokens never
    ///   contribute to the loss).
    /// * `positions` — 0-indexed causal logits-row indices. Row `p`
    ///   conditions on `tokens[..=p]` and predicts `tokens[p + 1]` when that
    ///   target exists. The final row `p == tokens.len() - 1` is the generic
    ///   next-token distribution; remote sources may need to append a probe
    ///   token to obtain it. Duplicate and out-of-order rows are allowed.
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
    identity: Option<crate::TeacherIdentityV1>,
    /// For each exact token sequence and position we precompute the answer.
    /// Indexing convention: the trainer queries `(tokens, positions)` and
    /// we look up each `positions[i]` under the complete sequence. A digest
    /// alone is not sufficient here: fixtures may carry externally supplied
    /// numeric training targets, so even a deliberate hash collision must not
    /// alias two examples.
    entries:
        std::collections::HashMap<Vec<u32>, std::collections::HashMap<usize, (Vec<u32>, Vec<f32>)>>,
    top_k: usize,
}

impl FixtureLogitSource {
    /// Build a fixture source where every (tokens, position) query
    /// returns the same uniform-over-K top-K. Useful as a smoke fixture
    /// for the trainer wiring.
    pub fn uniform_topk(teacher_id: impl Into<String>, vocab_size: usize, top_k: usize) -> Self {
        Self {
            caps: LogitSourceCaps {
                teacher_id: teacher_id.into(),
                vocab_size,
                max_top_k: top_k,
                supports_full_vocab: false,
                supports_batched: true,
                tokenizer_hash: None,
            },
            identity: None,
            entries: std::collections::HashMap::new(),
            top_k,
        }
    }

    /// Attach provenance to a fixture loaded from a verified, pre-scored
    /// artifact. Ordinary synthetic fixtures remain explicitly unverified.
    pub fn with_authoritative_identity(
        mut self,
        identity: crate::TeacherIdentityV1,
    ) -> Result<Self, LogitSourceError> {
        if identity.vocab_size() as usize != self.caps.vocab_size {
            return Err(LogitSourceError::invalid(
                &self.caps.teacher_id,
                format!(
                    "fixture identity vocab_size {} does not match fixture vocab_size {}",
                    identity.vocab_size(),
                    self.caps.vocab_size
                ),
            ));
        }
        if (identity.max_top_k() as usize) < self.top_k {
            return Err(LogitSourceError::invalid(
                &self.caps.teacher_id,
                format!(
                    "fixture top_k {} exceeds identity max_top_k {}",
                    self.top_k,
                    identity.max_top_k()
                ),
            ));
        }
        self.caps.tokenizer_hash = Some(identity.tokenizer_vocab_sha256().to_owned());
        self.identity = Some(identity);
        Ok(self)
    }

    /// Insert one exact-sequence fixture row.
    ///
    /// Repeating an identical row is idempotent. Reusing the same sequence and
    /// position with different numeric targets is rejected so duplicate
    /// examples cannot silently replace provenance-bound training data.
    pub fn insert(
        &mut self,
        tokens: &[u32],
        pos: usize,
        indices: Vec<u32>,
        logprobs: Vec<f32>,
    ) -> Result<(), LogitSourceError> {
        validate_logit_request(&self.caps, tokens, &[pos], Some(self.top_k))?;
        validate_topk_logprob_row(&self.caps, self.top_k, pos, &indices, &logprobs)?;

        let rows = self.entries.entry(tokens.to_vec()).or_default();
        if let Some((existing_indices, existing_logprobs)) = rows.get(&pos) {
            let same_logprobs = existing_logprobs.len() == logprobs.len()
                && existing_logprobs
                    .iter()
                    .zip(&logprobs)
                    .all(|(left, right)| left.to_bits() == right.to_bits());
            if existing_indices == &indices && same_logprobs {
                return Ok(());
            }
            return Err(LogitSourceError::invalid(
                &self.caps.teacher_id,
                format!(
                    "conflicting fixture rows for the same exact token sequence at position {pos}"
                ),
            ));
        }
        rows.insert(pos, (indices, logprobs));
        Ok(())
    }
}

impl LogitSource for FixtureLogitSource {
    fn capabilities(&self) -> LogitSourceCaps {
        self.caps.clone()
    }

    fn authoritative_teacher_identity(&self) -> Option<&crate::TeacherIdentityV1> {
        self.identity.as_ref()
    }

    fn authoritative_content_revision(&self) -> Option<String> {
        #[derive(Serialize)]
        struct FixtureRevisionRow<'a> {
            tokens: &'a [u32],
            position: usize,
            indices: &'a [u32],
            logprob_bits: Vec<u32>,
        }

        #[derive(Serialize)]
        struct FixtureRevision<'a> {
            schema: &'static str,
            capabilities: &'a LogitSourceCaps,
            top_k: usize,
            rows: Vec<FixtureRevisionRow<'a>>,
        }

        let mut sequences = self.entries.iter().collect::<Vec<_>>();
        sequences.sort_by(|(left, _), (right, _)| left.cmp(right));
        let mut rows = Vec::new();
        for (tokens, positions) in sequences {
            let mut positions = positions.iter().collect::<Vec<_>>();
            positions.sort_by_key(|(position, _)| **position);
            rows.extend(
                positions
                    .into_iter()
                    .map(|(position, (indices, logprobs))| FixtureRevisionRow {
                        tokens,
                        position: *position,
                        indices,
                        logprob_bits: logprobs.iter().map(|value| value.to_bits()).collect(),
                    }),
            );
        }
        let descriptor = FixtureRevision {
            schema: "kiln.fixture-logit-source.v1",
            capabilities: &self.caps,
            top_k: self.top_k,
            rows,
        };
        serde_json::to_vec(&descriptor)
            .ok()
            .map(|bytes| crate::train_receipt::sha256_bytes(&bytes))
    }

    fn fetch_logprobs(
        &self,
        tokens: &[u32],
        positions: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch, LogitSourceError> {
        let teacher_id = self.caps.teacher_id.clone();
        validate_logit_request(&self.caps, tokens, positions, top_k)?;
        let requested_k = top_k.unwrap_or(self.top_k);

        let rows = self.entries.get(tokens).ok_or_else(|| {
            LogitSourceError::invalid(
                &teacher_id,
                "no fixture entries for the exact token sequence",
            )
        })?;
        let mut indices = Vec::with_capacity(positions.len() * requested_k);
        let mut logprobs = Vec::with_capacity(positions.len() * requested_k);
        for &pos in positions {
            let entry = rows.get(&pos).ok_or_else(|| {
                LogitSourceError::invalid(
                    &teacher_id,
                    format!("no fixture entry for the exact token sequence at position {pos}"),
                )
            })?;
            validate_topk_logprob_row(&self.caps, self.top_k, pos, &entry.0, &entry.1)?;
            indices.extend_from_slice(&entry.0[..requested_k]);
            logprobs.extend_from_slice(&entry.1[..requested_k]);
        }
        let batch = TopKLogprobs {
            indices,
            logprobs,
            top_k: requested_k,
        };
        validate_topk_logprobs_batch(&self.caps, tokens, positions, requested_k, &batch)?;
        Ok(LogprobBatch::TopK(batch))
    }
}

/// A simple deterministic `LogitSource` that synthesises a uniform
/// top-K answer from a hash of `(teacher_id, tokens, pos)`. Used as
/// the milestone-13 trainer-wire-up fallback when no real teacher is
/// resolved — the math path is exercised end-to-end (forward → loss
/// → backward → optimizer step), and the synthetic uniform target
/// still produces a real (nonzero) gradient into the LoRA params.
///
/// Production runs should swap this for `RemoteTeacher` (HTTP) or a
/// `LocalTeacher` (in-process model). The trait is the same.
#[derive(Debug)]
pub struct DeterministicUniformLogitSource {
    caps: LogitSourceCaps,
    top_k: usize,
}

impl DeterministicUniformLogitSource {
    pub fn new(teacher_id: impl Into<String>, vocab_size: usize, top_k: usize) -> Self {
        Self {
            caps: LogitSourceCaps {
                teacher_id: teacher_id.into(),
                vocab_size,
                max_top_k: top_k,
                supports_full_vocab: false,
                supports_batched: true,
                tokenizer_hash: None,
            },
            top_k,
        }
    }
}

impl LogitSource for DeterministicUniformLogitSource {
    fn capabilities(&self) -> LogitSourceCaps {
        self.caps.clone()
    }

    fn authoritative_content_revision(&self) -> Option<String> {
        #[derive(Serialize)]
        struct DeterministicUniformRevision<'a> {
            schema: &'static str,
            algorithm: &'static str,
            capabilities: &'a LogitSourceCaps,
            top_k: usize,
        }

        serde_json::to_vec(&DeterministicUniformRevision {
            schema: "kiln.deterministic-logit-source.v1",
            algorithm: "fnv1a64-tokens-position-uniform-topk-v1",
            capabilities: &self.caps,
            top_k: self.top_k,
        })
        .ok()
        .map(|bytes| crate::train_receipt::sha256_bytes(&bytes))
    }

    fn fetch_logprobs(
        &self,
        tokens: &[u32],
        positions: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch, LogitSourceError> {
        let teacher_id = self.caps.teacher_id.clone();
        validate_logit_request(&self.caps, tokens, positions, top_k)?;
        let requested_k = top_k.unwrap_or(self.top_k);
        let vocab = self.caps.vocab_size as u32;
        if vocab < requested_k as u32 {
            return Err(LogitSourceError::invalid(
                &teacher_id,
                format!("vocab_size {vocab} < top_k {requested_k}"),
            ));
        }
        // FNV-1a hash of tokens, mixed with position — deterministic
        // and cheap. The K picks step forward by 1 (mod vocab) from
        // the seed to guarantee uniqueness within a position.
        let mut indices = Vec::with_capacity(positions.len() * requested_k);
        let mut logprobs = Vec::with_capacity(positions.len() * requested_k);
        let logp = -(requested_k as f32).ln();
        for &pos in positions {
            let mut h: u64 = 0xcbf29ce484222325;
            for &t in tokens {
                h ^= t as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= pos as u64;
            h = h.wrapping_mul(0x100000001b3);
            let seed = (h % vocab as u64) as u32;
            let mut seen = std::collections::HashSet::new();
            let mut produced = 0usize;
            let mut idx = seed;
            while produced < requested_k {
                if seen.insert(idx) {
                    indices.push(idx);
                    logprobs.push(logp);
                    produced += 1;
                }
                idx = (idx + 1) % vocab;
            }
        }
        let batch = TopKLogprobs {
            indices,
            logprobs,
            top_k: requested_k,
        };
        validate_topk_logprobs_batch(&self.caps, tokens, positions, requested_k, &batch)?;
        Ok(LogprobBatch::TopK(batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validation_caps() -> LogitSourceCaps {
        LogitSourceCaps {
            teacher_id: "validator-test".into(),
            vocab_size: 8,
            max_top_k: 4,
            supports_full_vocab: false,
            supports_batched: true,
            tokenizer_hash: None,
        }
    }

    #[test]
    fn request_validation_rejects_invalid_tokens_positions_and_k() {
        let caps = validation_caps();
        assert!(validate_logit_request(&caps, &[], &[], Some(2)).is_err());
        assert!(validate_logit_request(&caps, &[0, 8], &[0], Some(2)).is_err());
        assert!(validate_logit_request(&caps, &[0, 1], &[2], Some(2)).is_err());
        assert!(validate_logit_request(&caps, &[0, 1], &[0], Some(0)).is_err());
        assert!(matches!(
            validate_logit_request(&caps, &[0, 1], &[0], Some(5)),
            Err(LogitSourceError::TopKExceedsCap { .. })
        ));
        assert!(matches!(
            validate_logit_request(&caps, &[0, 1], &[0], None),
            Err(LogitSourceError::FullVocabUnsupported { .. })
        ));

        let mut zero_vocab = caps.clone();
        zero_vocab.vocab_size = 0;
        assert!(validate_logit_request(&zero_vocab, &[0], &[0], Some(1)).is_err());

        let mut vocab_bound = caps;
        vocab_bound.max_top_k = 16;
        assert!(validate_logit_request(&vocab_bound, &[0], &[0], Some(9)).is_err());
    }

    #[test]
    fn request_validation_allows_empty_and_duplicate_position_lists() {
        let caps = validation_caps();
        validate_logit_request(&caps, &[0, 1, 2], &[], Some(2)).unwrap();
        validate_logit_request(&caps, &[0, 1, 2], &[2, 0, 2], Some(2)).unwrap();
    }

    #[test]
    fn topk_row_validation_rejects_malformed_sparse_support() {
        let caps = validation_caps();
        validate_topk_logprob_row(&caps, 2, 0, &[1, 2], &[-0.5, -1.5]).unwrap();

        assert!(validate_topk_logprob_row(&caps, 2, 0, &[1], &[-1.0]).is_err());
        assert!(validate_topk_logprob_row(&caps, 2, 0, &[1, 1], &[-1.0, -1.0]).is_err());
        assert!(validate_topk_logprob_row(&caps, 2, 0, &[1, 8], &[-1.0, -1.0]).is_err());
        assert!(validate_topk_logprob_row(&caps, 2, 0, &[1, 2], &[-2.0, -1.0]).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.01] {
            assert!(validate_topk_logprob_row(&caps, 2, 0, &[1, 2], &[-1.0, bad]).is_err());
        }
        assert!(validate_topk_logprob_row(&caps, 2, 0, &[1, 2], &[0.0, 0.0]).is_err());
    }

    #[test]
    fn topk_batch_validation_enforces_declared_k_and_row_count() {
        let caps = validation_caps();
        let tokens = [0, 1, 2];
        let positions = [2, 0];
        let valid = TopKLogprobs {
            indices: vec![1, 2, 3, 4],
            logprobs: vec![-1.0, -1.5, -1.0, -1.5],
            top_k: 2,
        };
        validate_topk_logprobs_batch(&caps, &tokens, &positions, 2, &valid).unwrap();

        let mut wrong_declared_k = valid.clone();
        wrong_declared_k.top_k = 3;
        assert!(
            validate_topk_logprobs_batch(&caps, &tokens, &positions, 2, &wrong_declared_k).is_err()
        );
        let mut short = valid;
        short.indices.pop();
        assert!(validate_topk_logprobs_batch(&caps, &tokens, &positions, 2, &short).is_err());
    }

    #[test]
    fn full_vocab_validation_enforces_variant_shape_values_and_mass() {
        let mut caps = validation_caps();
        caps.vocab_size = 4;
        caps.supports_full_vocab = true;
        let tokens = [0, 1, 2];
        let positions = [0, 2];
        let uniform = -(4.0f32).ln();
        let valid = LogprobBatch::FullVocab {
            logprobs: vec![uniform; 8],
            vocab_size: 4,
        };
        validate_full_vocab_logprobs_batch(&caps, &tokens, &positions, &valid).unwrap();

        let wrong_variant = LogprobBatch::TopK(TopKLogprobs {
            indices: vec![0, 1, 0, 1],
            logprobs: vec![-1.0, -1.5, -1.0, -1.5],
            top_k: 2,
        });
        assert!(
            validate_full_vocab_logprobs_batch(&caps, &tokens, &positions, &wrong_variant).is_err()
        );
        let wrong_vocab = LogprobBatch::FullVocab {
            logprobs: vec![uniform; 8],
            vocab_size: 8,
        };
        assert!(
            validate_full_vocab_logprobs_batch(&caps, &tokens, &positions, &wrong_vocab).is_err()
        );
        let short = LogprobBatch::FullVocab {
            logprobs: vec![uniform; 7],
            vocab_size: 4,
        };
        assert!(validate_full_vocab_logprobs_batch(&caps, &tokens, &positions, &short).is_err());
        for malformed_row in [
            vec![f32::NAN, uniform, uniform, uniform],
            vec![0.1, uniform, uniform, uniform],
            vec![0.0; 4],
            vec![-10.0; 4],
        ] {
            let malformed = LogprobBatch::FullVocab {
                logprobs: malformed_row,
                vocab_size: 4,
            };
            assert!(validate_full_vocab_logprobs_batch(&caps, &tokens, &[0], &malformed).is_err());
        }
    }

    #[test]
    fn target_positions_convert_to_preceding_logits_rows() {
        assert_eq!(
            target_token_positions_to_logits_rows("teacher", 8, &[1, 3, 7]).unwrap(),
            vec![0, 2, 6]
        );
        assert!(
            target_token_positions_to_logits_rows("teacher", 0, &[])
                .unwrap()
                .is_empty()
        );
        for invalid in [&[0][..], &[1, 1], &[2, 1], &[1, 8]] {
            assert!(target_token_positions_to_logits_rows("teacher", 8, invalid).is_err());
        }
    }

    #[test]
    fn fixture_returns_inserted_entries() {
        let mut src = FixtureLogitSource::uniform_topk("test-teacher", 64, 4);
        let tokens = vec![10u32, 20, 30, 40];
        src.insert(&tokens, 1, vec![5, 6, 7, 8], vec![-1.5, -1.6, -1.7, -1.8])
            .unwrap();
        src.insert(
            &tokens,
            2,
            vec![9, 10, 11, 12],
            vec![-2.0, -2.1, -2.2, -2.3],
        )
        .unwrap();

        let batch = src.fetch_logprobs(&tokens, &[1, 2], Some(4)).unwrap();
        match batch {
            LogprobBatch::TopK(t) => {
                assert_eq!(t.top_k, 4);
                assert_eq!(t.indices, vec![5, 6, 7, 8, 9, 10, 11, 12]);
                assert_eq!(
                    t.logprobs,
                    vec![-1.5, -1.6, -1.7, -1.8, -2.0, -2.1, -2.2, -2.3]
                );
            }
            _ => panic!("expected top-k batch"),
        }
    }

    #[test]
    fn fixture_rejects_conflicting_duplicate_rows() {
        let mut src = FixtureLogitSource::uniform_topk("test-teacher", 64, 2);
        let tokens = [10u32, 20, 30];
        src.insert(&tokens, 1, vec![5, 6], vec![-1.0, -2.0])
            .unwrap();
        src.insert(&tokens, 1, vec![5, 6], vec![-1.0, -2.0])
            .unwrap();

        let error = src
            .insert(&tokens, 1, vec![5, 7], vec![-1.0, -2.0])
            .unwrap_err();
        assert!(error.to_string().contains("conflicting fixture rows"));
        let batch = src.fetch_logprobs(&tokens, &[1], Some(2)).unwrap();
        let LogprobBatch::TopK(batch) = batch else {
            panic!("expected top-K fixture row")
        };
        assert_eq!(batch.indices, vec![5, 6]);
    }

    #[test]
    fn fixture_keys_rows_by_exact_tokens() {
        let mut src = FixtureLogitSource::uniform_topk("test-teacher", 64, 2);
        let first = [1u32, 2, 3];
        let second = [1u32, 2, 4];
        src.insert(&first, 1, vec![5, 6], vec![-1.0, -2.0]).unwrap();
        src.insert(&second, 1, vec![7, 8], vec![-1.0, -2.0])
            .unwrap();

        let LogprobBatch::TopK(first_row) = src.fetch_logprobs(&first, &[1], Some(2)).unwrap()
        else {
            panic!("expected top-K fixture row")
        };
        let LogprobBatch::TopK(second_row) = src.fetch_logprobs(&second, &[1], Some(2)).unwrap()
        else {
            panic!("expected top-K fixture row")
        };
        assert_eq!(first_row.indices, vec![5, 6]);
        assert_eq!(second_row.indices, vec![7, 8]);
    }

    #[test]
    fn fixture_content_revision_is_order_independent_and_covers_every_row() {
        let first_tokens = [1u32, 2, 3];
        let second_tokens = [4u32, 5, 6];
        let mut forward = FixtureLogitSource::uniform_topk("revision-test", 64, 2);
        forward
            .insert(&first_tokens, 1, vec![5, 6], vec![-1.0, -2.0])
            .unwrap();
        forward
            .insert(&second_tokens, 0, vec![7, 8], vec![-1.25, -2.25])
            .unwrap();

        let mut reverse = FixtureLogitSource::uniform_topk("revision-test", 64, 2);
        reverse
            .insert(&second_tokens, 0, vec![7, 8], vec![-1.25, -2.25])
            .unwrap();
        reverse
            .insert(&first_tokens, 1, vec![5, 6], vec![-1.0, -2.0])
            .unwrap();

        let revision = forward.authoritative_content_revision().unwrap();
        assert!(revision.starts_with("sha256:"));
        assert_eq!(revision, reverse.authoritative_content_revision().unwrap());

        reverse
            .insert(&first_tokens, 0, vec![9, 10], vec![-1.5, -2.5])
            .unwrap();
        assert_ne!(revision, reverse.authoritative_content_revision().unwrap());
    }

    #[test]
    fn fixture_rejects_overlarge_k() {
        let src = FixtureLogitSource::uniform_topk("test-teacher", 64, 4);
        let err = src.fetch_logprobs(&[1, 2, 3], &[1], Some(8)).unwrap_err();
        match err {
            LogitSourceError::TopKExceedsCap { requested, cap, .. } => {
                assert_eq!(requested, 8);
                assert_eq!(cap, 4);
            }
            _ => panic!("expected TopKExceedsCap"),
        }
    }

    #[test]
    fn deterministic_uniform_returns_kunique_indices_per_position() {
        let src = DeterministicUniformLogitSource::new("test-det", 256, 8);
        let batch = src.fetch_logprobs(&[1, 2, 3, 4], &[0, 3], Some(8)).unwrap();
        match batch {
            LogprobBatch::TopK(t) => {
                assert_eq!(t.top_k, 8);
                assert_eq!(t.indices.len(), 16);
                // First position's 8 indices are unique.
                let mut first: std::collections::HashSet<u32> =
                    t.indices.iter().take(8).copied().collect();
                assert_eq!(first.len(), 8);
                // Second position's 8 indices are unique.
                let second: std::collections::HashSet<u32> =
                    t.indices.iter().skip(8).copied().collect();
                assert_eq!(second.len(), 8);
                // Uniform log-prob = -ln(K).
                let expected_lp = -(8f32).ln();
                for &lp in &t.logprobs {
                    assert!((lp - expected_lp).abs() < 1e-5);
                }
                // Position differentiation: row-0 and row-1 indices
                // should be distinguishable for different positions
                // (not identical sets).
                first.retain(|i| second.contains(i));
                assert!(first.len() < 8, "row 0 and row 1 should differ");
            }
            _ => panic!("expected TopK batch"),
        }
    }

    #[test]
    fn deterministic_uniform_is_deterministic() {
        let src_a = DeterministicUniformLogitSource::new("det", 256, 4);
        let src_b = DeterministicUniformLogitSource::new("det", 256, 4);
        let a = src_a
            .fetch_logprobs(&[7, 11, 13], &[1, 2], Some(4))
            .unwrap();
        let b = src_b
            .fetch_logprobs(&[7, 11, 13], &[1, 2], Some(4))
            .unwrap();
        match (a, b) {
            (LogprobBatch::TopK(ta), LogprobBatch::TopK(tb)) => {
                assert_eq!(ta.indices, tb.indices);
                assert_eq!(ta.logprobs, tb.logprobs);
            }
            _ => panic!("expected TopK"),
        }
    }

    #[test]
    fn deterministic_uniform_revision_covers_its_algorithm_contract() {
        let first = DeterministicUniformLogitSource::new("det", 256, 4);
        let same = DeterministicUniformLogitSource::new("det", 256, 4);
        let different = DeterministicUniformLogitSource::new("det", 256, 8);

        assert_eq!(
            first.authoritative_content_revision(),
            same.authoritative_content_revision()
        );
        assert_ne!(
            first.authoritative_content_revision(),
            different.authoritative_content_revision()
        );
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
