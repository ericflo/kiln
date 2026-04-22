//! Optional per-step instrumentation for native MTP (Multi-Token Prediction)
//! speculative decode.
//!
//! Off by default — every call site early-outs cheaply when `KILN_MTP_DEBUG`
//! is unset. Enable with `KILN_MTP_DEBUG=1` to emit one `tracing::info!` line
//! per draft and per verify pass containing:
//!
//! - `h_prev` L2 norm at MTP draft entry (sanity check on tensor magnitudes)
//! - top-K MTP draft logits as `(token_id, logit)` pairs (default K=5)
//! - top-K base verify-pos0 logits (the prediction the draft is graded against)
//! - the chosen draft token, the target token at pos 0, and accept/reject
//!
//! By default only the first `KILN_MTP_DEBUG_MAX_CALLS` (default 16) draft
//! steps log; set the env var to `0` for unlimited. The counter is a single
//! process-wide `AtomicUsize`, so a fresh process resets to zero. Use
//! [`reset_counter`] to re-arm without restarting (handy from tests).
//!
//! This module exists to support Phase B of the low-α root-cause investigation
//! tracked in `PROFILING.md` ("MTP α=0.411 vs 0.72 floor"). The intended
//! workflow is: enable on a pod, capture a 16-step trace, diff against an HF
//! reference run on the same prompt, and either ship a Tier 1 fix (if a
//! tensor or layout bug appears) or document the runtime evidence in a Tier 2
//! diagnostic update to `PROFILING.md`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static C7_SDPA_CAPTURE_ARMED_THREADS: AtomicUsize = AtomicUsize::new(0);
static MTP_FP32_HEAD_ARMED_THREADS: AtomicUsize = AtomicUsize::new(0);
// Phase B7a: track which mtp_pos values have already been dumped so a single
// process can capture multiple positions. Initialized lazily on first use.
static DUMP_DONE_POSITIONS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

thread_local! {
    /// Phase B7b: thread-local sink for sub-op taps captured inside the MTP
    /// inner transformer block. `None` = disabled (every `capture_subop` call
    /// is a no-op). `Some(buf)` = currently capturing into `buf`.
    ///
    /// Driven by [`arm_subop_capture`] / [`drain_subop_capture`]. Lives in TLS
    /// because the MTP debug call sites all run on the inference thread that
    /// also owns the transformer block; cross-thread capture is not supported
    /// (and not needed — the dump is single-call, single-thread).
    static SUBOP_CAPTURE: RefCell<Option<Vec<(String, Vec<usize>, Vec<f32>)>>> =
        const { RefCell::new(None) };

    /// Phase B10: separate thread-local sink for base-model per-layer hidden
    /// states (`h_layer_*` + `h_post_final_norm`, renamed from
    /// `h_pre_final_norm` in Phase C18). Kept distinct from
    /// [`SUBOP_CAPTURE`] so the base-model forward can run while the MTP
    /// inner-block capture window is closed (and vice versa), and so each
    /// buffer can be drained without conflation.
    ///
    /// Driven by [`arm_h_main_capture`] / [`drain_h_main_capture`] /
    /// [`capture_h_main_tap`].
    static H_MAIN_CAPTURE: RefCell<Option<Vec<(String, Vec<usize>, Vec<f32>)>>> =
        const { RefCell::new(None) };

    /// Phase B11: thread-local slot for the base-model input token ids that
    /// fed the forward pass which produced the currently-armed h_main
    /// capture. Populated in `model_forward_paged_with_last_hidden` right
    /// after [`arm_h_main_capture`] via [`stash_h_main_prompt_tokens`], and
    /// drained in `mtp_forward_step`'s dump block so the tokens can be
    /// serialized alongside the taps (see [`write_mtp_dump`]). The HF
    /// reference (`scripts/mtp_h_main_reference_dump.py`) prefers these
    /// tokens over its canonical greeting, guaranteeing both sides replay
    /// the exact same prompt during the per-tap bisect.
    static H_MAIN_PROMPT_TOKENS: RefCell<Option<Vec<u32>>> =
        const { RefCell::new(None) };

    /// Phase B11b: thread-local sink for layer-0 GDN sub-op taps. Kept
    /// distinct from [`H_MAIN_CAPTURE`] so the per-layer boundary taps and
    /// the fine-grained layer-0 bisect taps can be drained independently.
    /// Entries are stored as `(name, shape, host_f32)` so the GPU scratch
    /// is released at capture time — same pattern as B10's h_main slot.
    ///
    /// Driven by [`arm_b11_layer0_capture`] / [`drain_b11_layer0_capture`] /
    /// [`capture_b11_layer0_tap`], and gated on
    /// `KILN_MTP_DUMP_B11_TAPS=1` so production decode pays zero cost.
    static B11_LAYER0_CAPTURE: RefCell<Option<Vec<(String, Vec<usize>, Vec<f32>)>>> =
        const { RefCell::new(None) };

    /// Phase B12: thread-local sink for layer-31 GQA sub-op taps. Distinct
    /// from B10/B11 sinks so each phase's capture window can be armed and
    /// drained independently. Same `(name, shape, host_f32)` shape as the
    /// other phase sinks; same release-at-capture-time pattern.
    ///
    /// Driven by [`arm_b12_gqa_capture`] / [`drain_b12_gqa_capture`] /
    /// [`capture_b12_gqa_tap`], and gated on `KILN_MTP_DUMP_B12_GQA_TAPS=1`
    /// so production decode pays zero cost.
    static B12_GQA_CAPTURE: RefCell<Option<Vec<(String, Vec<usize>, Vec<f32>)>>> =
        const { RefCell::new(None) };

    /// Phase B12: thread-local "current absolute layer index" slot. Set by
    /// the main transformer layer loop around each layer's forward call so
    /// that deep inside `gqa_attention_paged` / `transformer_block_paged`
    /// the capture call sites can gate on `abs_layer_idx == 31` without
    /// needing an explicit signature-change plumb through every call site.
    ///
    /// `None` = no layer-scoped capture in flight (production path,
    /// MTP-inner-block path). `Some(i)` = base-model layer `i` is currently
    /// executing on this thread.
    ///
    /// Driven by [`enter_b12_layer_scope`] / [`exit_b12_layer_scope`] and
    /// read via [`current_b12_layer_is_31`].
    static B12_CURRENT_LAYER: RefCell<Option<usize>> =
        const { RefCell::new(None) };

    /// Phase C6: thread-local sink for the 5 pre-RoPE MTP-input taps
    /// captured inside `mtp_forward_step`. The pre-RoPE bisect localizes
    /// drift to one of `token_emb`, `norm_emb`, `norm_h`, `concat`, or
    /// `fused` before the inner transformer block applies RoPE. Kept
    /// distinct from B11/B12 sinks so the MTP-inner capture window can
    /// be drained independently of the base-model sinks.
    ///
    /// Driven by [`arm_pre_rope_capture`] / [`drain_pre_rope_capture`] /
    /// [`capture_pre_rope_tap`], and gated on `KILN_MTP_DUMP_PRE_ROPE=1`
    /// so production decode pays zero cost.
    static PRE_ROPE_CAPTURE: RefCell<Option<Vec<(String, Vec<usize>, Vec<f32>)>>> =
        const { RefCell::new(None) };

    /// Phase C7: thread-local sink for the 7 SDPA-internal taps captured
    /// inside the MTP inner GQA attention. Extends the C6 pre-RoPE bisect
    /// one layer deeper: C6 localized drift to `post_attn_raw` (SDPA
    /// output) at `mtp_pos=2` despite pre-RoPE inputs matching
    /// cos_sim≥0.9999. C7 bisects inside SDPA — `pre_sdpa_q/k/v`,
    /// `causal_mask`, `attn_scores_pre_softmax`, `attn_probs`, `attn_out`
    /// — so we can pin the first tensor to drop below cos_sim=1 inside
    /// the attention step itself. Kept distinct from the C6 / B11 / B12
    /// sinks so the MTP-inner SDPA capture window can be armed and
    /// drained independently.
    ///
    /// Driven by [`arm_c7_sdpa_capture`] / [`drain_c7_sdpa_capture`] /
    /// [`capture_c7_sdpa_tap`], and gated on `KILN_MTP_DUMP_C7_SDPA=1`
    /// so production decode pays zero cost.
    static C7_SDPA_CAPTURE: RefCell<Option<Vec<(String, Vec<usize>, Vec<f32>)>>> =
        const { RefCell::new(None) };

    /// Phase C14: thread-local sink for the 3 post-MTP-transformer-block taps
    /// captured inside `mtp_forward_step`. Extends the C13 pre-projection
    /// splice verdict (splice inputs clear, cos ≥ 0.9999928) downstream to
    /// cover the MTP transformer block output (`post_block`), final
    /// normalization (`post_norm`), and tied lm_head logits (`logits`). The
    /// three tensors are already visible at the outer `mtp_forward_step`
    /// scope (no deep-plumb needed), so capture sites live next to the
    /// existing dump taps. Kept distinct from C6/C7 sinks so the post-block
    /// capture window can be drained independently of the pre-projection
    /// bisect. Serialization format matches C6/C7: `(name, shape, host_f32)`
    /// triples, emitted as `c14__<name>` F32 tensors in the dump.
    ///
    /// Driven by [`arm_c14_post_block_capture`] /
    /// [`drain_c14_post_block_capture`] / [`capture_c14_post_block_tap`],
    /// and gated on `KILN_MTP_DUMP_C14_POST_BLOCK=1` (OR-composed with the
    /// C13 splice meta-flag) so production decode pays zero cost.
    static C14_POST_BLOCK_CAPTURE: RefCell<Option<Vec<(String, Vec<usize>, Vec<f32>)>>> =
        const { RefCell::new(None) };

    /// Phase C8: thread-local flag that switches the MTP inner GQA attention
    /// to single-token self-attention (kv_len = 1, no paged-cache history).
    /// Set by [`arm_mtp_single_token_self_attn`] inside `mtp_forward_step`
    /// and cleared by [`disarm_mtp_single_token_self_attn`] right after the
    /// inner block returns. Read by `gqa_attention_paged` to:
    ///   * Skip the paged KV cache write/read for the MTP layer
    ///   * Bypass the fused paged-decode flash-attention kernel (which would
    ///     attend over the entire MTP cache history, defeating the fix)
    /// Always defaults to `false`, so non-MTP paths and pre-C8 callers see
    /// the legacy paged-cache behavior unchanged.
    static MTP_SINGLE_TOKEN_SELF_ATTN_ARMED: RefCell<bool> =
        const { RefCell::new(false) };

    /// Phase C12: thread-local flag that forces projection matmuls inside
    /// the MTP draft head to compute in f32. Set by [`arm_mtp_fp32_head`]
    /// inside `mtp_forward_step` when `KILN_MTP_FP32_HEAD=1` is active, and
    /// cleared by [`disarm_mtp_fp32_head`] right after the inner block
    /// returns. Read by `linear_with_lora_t` (and by the fused `mtp.fc`
    /// matmul path in `mtp_forward_step` itself, which also honors the
    /// existing `KILN_MTP_FC_FP32_ACCUM` flag) to promote inputs + weights
    /// to f32, matmul, then downcast back to the input dtype. This is the
    /// C12 kill switch: if α materially recovers with the flag on, the
    /// residual bf16 accumulation inside the MTP head's q/k/v/o + fc
    /// matmuls contributes meaningfully to α suppression. If α is
    /// unchanged, the null result exonerates bf16-accumulation in MTP.
    /// Always defaults to `false`, so non-MTP paths and pre-C12 callers
    /// see the legacy bf16 path unchanged.
    static MTP_FP32_HEAD_ARMED: RefCell<bool> =
        const { RefCell::new(false) };
}

const DEFAULT_MAX_CALLS: usize = 16;

/// True when `KILN_MTP_DEBUG=1` (or `true`) is set.
pub fn is_enabled() -> bool {
    std::env::var("KILN_MTP_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Returns true if instrumentation is enabled AND we are still under the
/// per-process call cap. Increments the call counter as a side effect, so each
/// invocation moves the budget forward by one.
///
/// The cap defaults to 16 to keep traces small enough to read by eye; set
/// `KILN_MTP_DEBUG_MAX_CALLS=0` for unlimited (noisy but useful when chasing
/// a specific reproducer that needs many tokens).
pub fn should_log() -> bool {
    if !is_enabled() {
        return false;
    }
    let max = std::env::var("KILN_MTP_DEBUG_MAX_CALLS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_CALLS);
    if max == 0 {
        return true;
    }
    let n = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    n < max
}

/// Reset the per-process call counter to zero. Test-only convenience; not
/// reachable from outside this crate's instrumentation paths.
pub fn reset_counter() {
    CALL_COUNT.store(0, Ordering::Relaxed);
}

/// Extract the top-K `(token_id, logit)` pairs from a flattenable logits
/// tensor. Accepts shapes `[V]`, `[1, V]`, or `[1, 1, V]` — anything that
/// flattens to a 1-D vocab-sized vector.
pub fn top_k_logits(logits: &Tensor, k: usize) -> Result<Vec<(u32, f32)>> {
    let vals = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut idx: Vec<(u32, f32)> = vals
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    Ok(idx)
}

/// Compute the L2 norm of a tensor as a single f32. Diagnostic check on
/// tensor magnitudes — useful for catching the "all zeros" or "explosion"
/// failure modes that would silently kill draft accuracy.
pub fn tensor_l2_norm(t: &Tensor) -> Result<f32> {
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(v.iter().map(|x| x * x).sum::<f32>().sqrt())
}

/// True when `KILN_MTP_SWAP_FC_NORMS=1` (or `true`) is set. When enabled,
/// `mtp_forward_step` swaps which RMSNorm weight is applied to which half of
/// the `fc` input — i.e. `pre_fc_norm_hidden` is applied to the embedding
/// half and `pre_fc_norm_embedding` is applied to the `h_prev` half. This is
/// the Phase B2 secondary-hypothesis A/B test from `PROFILING.md`: if α
/// jumps with the swap on, the loader pairs the wrong norm tensor to the
/// wrong half of the fused `fc` input. Read on every call (no caching) so
/// runs can A/B by toggling the env var between processes.
pub fn is_swap_fc_norms_enabled() -> bool {
    std::env::var("KILN_MTP_SWAP_FC_NORMS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// True when `KILN_MTP_FC_FP32_ACCUM=1` (or `true`) is set. When enabled,
/// `mtp_forward_step` promotes the `mtp.fc` matmul inputs to f32, performs
/// the `[1, 2H] @ [2H, H]` fused projection in f32, and casts the result
/// back to bf16. This is the Phase C9 opt-in falsification toggle: if α
/// materially lifts with the flag on, the residual bf16 matmul accumulation
/// noise at `fc_output` (max|Δ| ~1.6e-2, within the BF16 error budget)
/// contributes to downstream top1 flips; if α is unchanged, the null
/// result confirms that the bf16 variance captured by the C6 comparator
/// is a benign accumulation artifact and not the dominant α-suppressing
/// signal. Read on every call (no caching) so runs can A/B by toggling
/// the env var between processes.
pub fn is_mtp_fc_fp32_accum_enabled() -> bool {
    std::env::var("KILN_MTP_FC_FP32_ACCUM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// True when `KILN_MTP_FP32_HEAD=1` (or `true`) is set. When enabled,
/// `mtp_forward_step` arms the [`MTP_FP32_HEAD_ARMED`] TLS flag before the
/// MTP inner transformer block runs, which causes `linear_with_lora_t` to
/// promote projection inputs + weights to f32, matmul in f32, and cast the
/// result back to the input dtype for the q/k/v/o projections inside the
/// block. The arm flag also forces the `mtp.fc` matmul in
/// `mtp_forward_step` onto its fp32 accumulation path (same behaviour as
/// `KILN_MTP_FC_FP32_ACCUM=1`), so a single flag covers both halves of the
/// Phase C12 kill switch: q/k/v/o and fc_input/fc_output.
///
/// Subtlety: the MTP head is not Marlin-packed in the current build (see
/// `mtp_forward_step` docstring), so "whichever are currently Marlin-packed"
/// resolves to the empty set today. The flag still runs the intended
/// experiment because the test is really whether bf16 accumulation in the
/// MTP projection matmuls is the α-suppressor, regardless of whether those
/// matmuls go through the Marlin path or the straight BF16 broadcast_matmul
/// path. If α recovers with the flag on, Marlin-packing the MTP head is
/// expected to regress α once it lands — and a fused fp32 accumulation
/// variant of Marlin would be the follow-up. If α is unchanged, the signal
/// is elsewhere and Marlin can be added to MTP without worry.
pub fn is_mtp_fp32_head_enabled() -> bool {
    std::env::var("KILN_MTP_FP32_HEAD")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Path to which `mtp_forward_step` should dump all 8 Phase B6 numerical
/// bisect taps on each targeted MTP draft call, if set. Unset → no dump.
///
/// Phase B7a: when the path contains the literal substring `{pos}`, it is
/// substituted with the current `mtp_pos` value (e.g. `dump_pos{pos}.st`
/// becomes `dump_pos0.st`, `dump_pos1.st`, `dump_pos2.st`). This lets a
/// single run capture multiple positions for monotonic-degradation tests.
/// If the substring is absent, the path is used as-is — but the dump still
/// fires once per position in `KILN_MTP_DUMP_POS`, so include `{pos}` to
/// avoid each later position overwriting the prior file.
///
/// Paired with a Python reference implementation (see
/// `scripts/mtp_reference_dump.py`) that produces identically-named
/// safetensors files for the same fixed prompt + seed. A post-hoc
/// per-tap `allclose` sweep (see `scripts/mtp_compare.py`) localizes
/// the first divergence and narrows the remaining runtime hypotheses.
pub fn dump_path() -> Option<String> {
    std::env::var("KILN_MTP_DUMP_PATH").ok().filter(|s| !s.is_empty())
}

/// Path with `{pos}` substituted. Falls through to `dump_path()` when
/// the placeholder is absent (callers that capture multiple positions
/// should always include `{pos}` to avoid overwriting prior dumps).
pub fn dump_path_for_pos(mtp_pos: usize) -> Option<String> {
    let raw = dump_path()?;
    Some(raw.replace("{pos}", &mtp_pos.to_string()))
}

/// Set of `mtp_pos` values for which a dump should be written. Defaults
/// to `{0}` for backwards compatibility with Phase B6 (which always
/// dumped pos 0). Set `KILN_MTP_DUMP_POS=0,1,2` to capture three
/// positions in one process.
pub fn target_dump_positions() -> HashSet<usize> {
    let raw = std::env::var("KILN_MTP_DUMP_POS").unwrap_or_default();
    if raw.trim().is_empty() {
        let mut s = HashSet::new();
        s.insert(0);
        return s;
    }
    raw.split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

/// Returns `true` if this is the first call for `mtp_pos` AND that
/// position is targeted by `KILN_MTP_DUMP_POS`. Idempotent within the
/// process — each (pos) value fires at most once.
pub fn try_consume_dump_slot_for_pos(mtp_pos: usize) -> bool {
    if dump_path().is_none() {
        return false;
    }
    if !target_dump_positions().contains(&mtp_pos) {
        return false;
    }
    let mut guard = DUMP_DONE_POSITIONS
        .lock()
        .expect("DUMP_DONE_POSITIONS mutex poisoned");
    let set = guard.get_or_insert_with(HashSet::new);
    set.insert(mtp_pos)
}

/// Reset the dump latch. Test-only helper.
#[cfg(test)]
fn reset_dump_latch() {
    let mut guard = DUMP_DONE_POSITIONS.lock().unwrap();
    *guard = None;
}

/// True when `KILN_MTP_DUMP_SUBOPS=1` (or `true`) is set. Phase B7b
/// opt-in: when enabled and `mtp_forward_step` is about to dump for the
/// targeted `mtp_pos`, the inner transformer block records each named
/// sub-op tensor (post_q_proj, post_qk_norm, post_rope, …) into a
/// thread-local buffer that is appended to the safetensors file.
pub fn is_dump_subops_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_SUBOPS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Begin a sub-op capture window. Subsequent [`capture_subop`] calls
/// from the same thread record into a fresh buffer until
/// [`drain_subop_capture`] is called.
pub fn arm_subop_capture() {
    SUBOP_CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Drain the captured sub-op tensors and disarm. Returns whatever was
/// recorded since the matching [`arm_subop_capture`] call, in order.
pub fn drain_subop_capture() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    SUBOP_CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

/// Record one named tap if a capture window is currently open on this
/// thread. Cheap no-op (single TLS access + borrow) when the window is
/// closed, which is the production case. The tensor is materialized to
/// host F32 immediately so the GPU scratch is free to be reused.
///
/// Unlike `write_mtp_dump`, this is best-effort: serialization errors
/// are swallowed (returned via `Result` so callers using `?` get the
/// usual propagation, but production callers wrap in `let _ = ...`).
pub fn capture_subop(name: &str, t: &Tensor) -> Result<()> {
    let armed = SUBOP_CAPTURE.with(|c| c.borrow().is_some());
    if !armed {
        return Ok(());
    }
    let (shape, flat) = tensor_to_f32_host(t)
        .with_context(|| format!("capture_subop `{name}`: tensor→f32 host copy"))?;
    SUBOP_CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push((name.to_string(), shape, flat));
        }
    });
    Ok(())
}

// -----------------------------------------------------------------------------
// Phase B10: base-model per-layer hidden-state capture
// -----------------------------------------------------------------------------

/// Boundary layer indices captured when `KILN_MTP_DUMP_HIDDEN_STATES=1`.
/// Spans the full 32-layer Qwen3.5-4B stack: three GDN samples (0, 8, 16)
/// plus two GQA samples (23, 31). Layer 31's output is still the
/// pre-final-norm hidden state; the comparator also writes the
/// `h_post_final_norm` tap (Phase C18) so the final-layer → final_norm →
/// h_main handoff can be verified independently.
pub const B10_BOUNDARY_LAYERS: [usize; 5] = [0, 8, 16, 23, 31];

/// True when `KILN_MTP_DUMP_HIDDEN_STATES=1` (or `true`) is set. Phase B10
/// opt-in: when enabled alongside `KILN_MTP_DUMP_PATH`, the base-model
/// forward records the last-row hidden state at each boundary layer and at
/// the post-final-norm site (Phase C18), and appends them to the MTP dump
/// safetensors under names `h_layer_0`, `h_layer_8`, `h_layer_16`,
/// `h_layer_23`, `h_layer_31`, and `h_post_final_norm`.
pub fn is_dump_hidden_states_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_HIDDEN_STATES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// True if the Phase B10 base-model hidden-state capture window is currently
/// open on this thread AND the given layer index is one of the boundary
/// layers. Call sites use this to gate the (relatively cheap) per-layer
/// tensor-narrow + host copy so disarmed runs pay zero cost.
///
/// Phase B12: when `KILN_MTP_DUMP_B12_GQA_TAPS=1` is also set, this returns
/// true for every layer in [`B12_GQA_LAYERS`] in addition to the B10
/// boundary set, so the comparator gets full coverage of the 8 GQA layers.
pub fn should_capture_hidden_state_for_layer(layer_idx: usize) -> bool {
    let armed = H_MAIN_CAPTURE.with(|c| c.borrow().is_some());
    if !armed {
        return false;
    }
    if B10_BOUNDARY_LAYERS.contains(&layer_idx) {
        return true;
    }
    if is_dump_b12_gqa_taps_enabled() && B12_GQA_LAYERS.contains(&layer_idx) {
        return true;
    }
    false
}

/// True if the Phase B10 hidden-state capture window is currently armed on
/// this thread. Used to gate `h_post_final_norm` capture (Phase C18; not
/// indexed by layer).
pub fn is_h_main_capture_armed() -> bool {
    H_MAIN_CAPTURE.with(|c| c.borrow().is_some())
}

/// Begin a B10 base-model hidden-state capture window. Subsequent
/// [`capture_h_main_tap`] calls from the same thread record into a fresh
/// buffer until [`drain_h_main_capture`] is called. Does nothing if
/// [`is_dump_hidden_states_enabled`] is false — so callers can just invoke
/// this unconditionally around the base-model forward.
pub fn arm_h_main_capture() {
    if !is_dump_hidden_states_enabled() {
        return;
    }
    H_MAIN_CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Drain the captured h_main taps and disarm. Returns whatever was recorded
/// since the matching [`arm_h_main_capture`] call, in order.
pub fn drain_h_main_capture() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    H_MAIN_CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

/// Record one named h_main tap if a capture window is currently open on
/// this thread. Cheap no-op (single TLS access + borrow) when the window
/// is closed, which is the production case. The tensor is materialized to
/// host F32 immediately so the GPU scratch is free to be reused.
pub fn capture_h_main_tap(name: &str, t: &Tensor) -> Result<()> {
    let armed = H_MAIN_CAPTURE.with(|c| c.borrow().is_some());
    if !armed {
        return Ok(());
    }
    let (shape, flat) = tensor_to_f32_host(t)
        .with_context(|| format!("capture_h_main_tap `{name}`: tensor→f32 host copy"))?;
    H_MAIN_CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push((name.to_string(), shape, flat));
        }
    });
    Ok(())
}

/// Phase B11: stash the base-model prompt tokens for the currently-armed
/// h_main capture. No-op when the capture window is closed (matches the
/// semantics of [`arm_h_main_capture`] — callers can invoke unconditionally).
pub fn stash_h_main_prompt_tokens(tokens: &[u32]) {
    if !is_h_main_capture_armed() {
        return;
    }
    H_MAIN_PROMPT_TOKENS.with(|c| *c.borrow_mut() = Some(tokens.to_vec()));
}

/// Phase B11: drain the stashed prompt tokens. Returns an empty vector when
/// nothing was stashed (e.g. h_main capture never armed, or the prior dump
/// pass already drained). The empty-case is what `write_mtp_dump` passes on
/// to skip serializing a `prompt_tokens` tensor.
pub fn drain_h_main_prompt_tokens() -> Vec<u32> {
    H_MAIN_PROMPT_TOKENS.with(|c| c.borrow_mut().take().unwrap_or_default())
}

// -----------------------------------------------------------------------------
// Phase B11b: layer-0 GDN sub-op taps
// -----------------------------------------------------------------------------

/// True when `KILN_MTP_DUMP_B11_TAPS=1` (or `true`) is set. Opt-in for the
/// layer-0 GDN sub-op bisect: when enabled alongside `KILN_MTP_DUMP_PATH`
/// *and* `KILN_MTP_DUMP_HIDDEN_STATES=1`, the base-model forward records the
/// 11 named layer-0 taps listed in [`b11_tap_names`] and appends them to the
/// MTP dump safetensors under names `b11__<name>`.
pub fn is_dump_b11_taps_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_B11_TAPS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Canonical ordered list of Phase B11b layer-0 GDN sub-op tap names. The
/// comparator (`scripts/mtp_compare.py --b11`) expects this exact order in
/// its output table, and the HF reference (`scripts/mtp_h_main_reference_dump.py`)
/// emits mirrored `b11__<name>` tensors in the same order.
///
/// Order follows the forward graph at layer 0:
/// 1. `tok_embed` — `embed_tokens(input_ids)` output at layer 0 entry
/// 2. `layer_0_post_input_norm` — output of the layer's pre-GDN input LayerNorm
/// 3. `gdn_in_proj` — `in_proj_qkvz` / `in_proj_ba` output, concatenated
/// 4. `gdn_conv` — causal conv1d output (prefill path)
/// 5. `gdn_qk_norm_q` — L2-normalized query after `q_l2norm`
/// 6. `gdn_qk_norm_k` — L2-normalized key after `k_l2norm`
/// 7. `gdn_gate_beta` — sigmoid-gated `beta` after the beta projection
/// 8. `gdn_gate_g` — softplus-gated `g` after the A/gate projection
/// 9. `gdn_recur_out` — output of `fused_recurrent_gated_delta_rule` / chunked eqvt
/// 10. `gdn_gated_norm` — output of the `GatedRMSNorm` / gated pre-out-proj norm
/// 11. `gdn_out_proj` — output of the final `out_proj` linear
pub const B11_TAP_NAMES: &[&str] = &[
    "tok_embed",
    "layer_0_post_input_norm",
    "gdn_in_proj",
    "gdn_conv",
    "gdn_qk_norm_q",
    "gdn_qk_norm_k",
    "gdn_gate_beta",
    "gdn_gate_g",
    "gdn_recur_out",
    "gdn_gated_norm",
    "gdn_out_proj",
];

/// True if the Phase B11b layer-0 capture window is currently armed on this
/// thread. Call sites use this to gate the (relatively cheap) host copy so
/// disarmed runs pay zero cost.
pub fn is_b11_layer0_capture_armed() -> bool {
    B11_LAYER0_CAPTURE.with(|c| c.borrow().is_some())
}

/// True if a B11b capture should happen for `layer_idx`. Currently only
/// layer 0 is captured (matches the task brief — the B10 boundary scan
/// already localized the divergence to layer 0). Exposed as a helper so
/// forward.rs can guard the cheap arithmetic / host copy per call site.
pub fn should_capture_b11_tap_for_layer(layer_idx: usize) -> bool {
    layer_idx == 0 && is_b11_layer0_capture_armed()
}

/// Begin a B11b layer-0 capture window. Subsequent [`capture_b11_layer0_tap`]
/// calls from the same thread record into a fresh buffer until
/// [`drain_b11_layer0_capture`] is called. Does nothing if
/// [`is_dump_b11_taps_enabled`] is false — so callers can invoke this
/// unconditionally around the base-model forward.
pub fn arm_b11_layer0_capture() {
    if !is_dump_b11_taps_enabled() {
        return;
    }
    B11_LAYER0_CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Drain the captured B11b layer-0 taps and disarm. Returns whatever was
/// recorded since the matching [`arm_b11_layer0_capture`] call, in order.
pub fn drain_b11_layer0_capture() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    B11_LAYER0_CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

/// Record one named B11b layer-0 tap if a capture window is currently open
/// on this thread. Cheap no-op (single TLS access + borrow) when the window
/// is closed, which is the production case. The tensor is materialized to
/// host F32 immediately so the GPU scratch is free to be reused — same
/// pattern as [`capture_h_main_tap`].
pub fn capture_b11_layer0_tap(name: &str, t: &Tensor) -> Result<()> {
    let armed = B11_LAYER0_CAPTURE.with(|c| c.borrow().is_some());
    if !armed {
        return Ok(());
    }
    let (shape, flat) = tensor_to_f32_host(t)
        .with_context(|| format!("capture_b11_layer0_tap `{name}`: tensor→f32 host copy"))?;
    B11_LAYER0_CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push((name.to_string(), shape, flat));
        }
    });
    Ok(())
}

// -----------------------------------------------------------------------------
// Phase B12: layer 24..31 GQA per-layer + layer-31 sub-op taps
// -----------------------------------------------------------------------------

/// Layer indices captured under Phase B12 in addition to the standard B10
/// boundary set. Covers the 8 full-attention (GQA) layers at the tail of the
/// Qwen3.5-4B stack so the comparator can localize a sub-percentile cos_sim
/// drift to a single layer.
///
/// Layer 31 is already in [`B10_BOUNDARY_LAYERS`]; including it here too is
/// harmless because [`should_capture_hidden_state_for_layer`] uses set
/// membership semantics.
pub const B12_GQA_LAYERS: [usize; 8] = [24, 25, 26, 27, 28, 29, 30, 31];

/// True when `KILN_MTP_DUMP_B12_GQA_TAPS=1` (or `true`) is set. Opt-in for
/// the Phase B12 layer-24..31 + layer-31 GQA sub-op bisect: when enabled
/// alongside `KILN_MTP_DUMP_PATH` *and* `KILN_MTP_DUMP_HIDDEN_STATES=1`,
/// the base-model forward records:
///
/// - one `h_layer_<idx>` tap per layer in [`B12_GQA_LAYERS`] (in addition
///   to the B10 boundary set), via the existing h_main capture path;
/// - the named layer-31 GQA sub-op taps listed in [`B12_GQA_TAP_NAMES`],
///   appended to the safetensors dump under names `b12__<name>`.
pub fn is_dump_b12_gqa_taps_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_B12_GQA_TAPS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Canonical ordered list of Phase B12 layer-31 GQA sub-op tap names. The
/// comparator (`scripts/mtp_compare.py --b12`) expects this exact order in
/// its output table, and the HF reference (`scripts/mtp_h_main_reference_dump.py`)
/// emits mirrored `b12__<name>` tensors in the same order.
///
/// Order follows the forward graph at layer 31:
/// 1. `post_input_norm` — output of the pre-attention input LayerNorm
/// 2. `q_proj` — output of `q_proj` linear (pre-reshape, pre-norm)
/// 3. `k_proj` — output of `k_proj` linear (pre-reshape, pre-norm)
/// 4. `v_proj` — output of `v_proj` linear (pre-reshape)
/// 5. `qk_norm_q` — output of the per-head q RMSNorm (`q_norm`)
/// 6. `qk_norm_k` — output of the per-head k RMSNorm (`k_norm`)
/// 7. `rope_q` — q after rotary position embedding application
/// 8. `rope_k` — k after rotary position embedding application
/// 9. `attn_out` — output of the SDPA / FlashAttention call
/// 10. `o_proj` — output of `o_proj` linear (post-attention output projection)
/// 11. `post_attn_norm` — output of the post-attention LayerNorm (pre-MLP)
/// 12. `mlp_gate` — output of the MLP gate projection
/// 13. `mlp_up` — output of the MLP up projection
/// 14. `mlp_down` — output of the MLP down projection
pub const B12_GQA_TAP_NAMES: &[&str] = &[
    "post_input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_norm_q",
    "qk_norm_k",
    "rope_q",
    "rope_k",
    "attn_out",
    "o_proj",
    "post_attn_norm",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
];

/// True if the Phase B12 layer-31 GQA capture window is currently armed on
/// this thread. Call sites use this to gate the (relatively cheap) host
/// copy so disarmed runs pay zero cost.
pub fn is_b12_gqa_capture_armed() -> bool {
    B12_GQA_CAPTURE.with(|c| c.borrow().is_some())
}

/// True if a B12 sub-op capture should happen for `layer_idx`. Currently
/// only layer 31 is captured at sub-op granularity (the per-layer h_layer_*
/// taps for layers 24..30 already provide the per-layer comparator data).
/// Exposed as a helper so forward.rs can guard the per-call host copies.
pub fn should_capture_b12_gqa_tap_for_layer(layer_idx: usize) -> bool {
    layer_idx == 31 && is_b12_gqa_capture_armed()
}

/// Record that base-model layer `abs_layer_idx` is the one currently being
/// executed on this thread. Set by the main transformer loop (and only the
/// main transformer loop — not the MTP inner block path) right before the
/// layer's forward call. Paired with [`exit_b12_layer_scope`] on the
/// immediately following line so that the window is tightly scoped.
///
/// Safe to call unconditionally — it's a single TLS slot write. Call sites
/// do not need to check `is_dump_b12_gqa_taps_enabled()` first, because the
/// downstream [`capture_b12_gqa_tap`] gates already short-circuit when the
/// capture window is not armed.
pub fn enter_b12_layer_scope(abs_layer_idx: usize) {
    B12_CURRENT_LAYER.with(|c| *c.borrow_mut() = Some(abs_layer_idx));
}

/// Clear the "current layer index" slot. Paired with
/// [`enter_b12_layer_scope`]; must be called even on the error path so the
/// slot does not leak into the next iteration. In practice call sites use
/// this immediately after the layer's forward call returns (success or
/// error) via a `?`-before-exit pattern.
pub fn exit_b12_layer_scope() {
    B12_CURRENT_LAYER.with(|c| *c.borrow_mut() = None);
}

/// True when the current layer scope on this thread is layer 31 AND a B12
/// capture window is armed. This is the gate used by every
/// [`capture_b12_gqa_tap`] call site in `gqa_attention_paged` /
/// `transformer_block_paged` so that only the layer-31 forward pass emits
/// sub-op taps. Layers 24..30 are covered by the per-layer h_layer_*
/// boundary taps the B10 sink already records.
pub fn current_b12_layer_is_31() -> bool {
    if !is_b12_gqa_capture_armed() {
        return false;
    }
    B12_CURRENT_LAYER.with(|c| *c.borrow() == Some(31))
}

/// Begin a B12 layer-31 GQA capture window. Subsequent
/// [`capture_b12_gqa_tap`] calls from the same thread record into a fresh
/// buffer until [`drain_b12_gqa_capture`] is called. Does nothing if
/// [`is_dump_b12_gqa_taps_enabled`] is false — so callers can invoke this
/// unconditionally around the base-model forward.
pub fn arm_b12_gqa_capture() {
    if !is_dump_b12_gqa_taps_enabled() {
        return;
    }
    B12_GQA_CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Drain the captured B12 layer-31 GQA taps and disarm. Returns whatever
/// was recorded since the matching [`arm_b12_gqa_capture`] call, in order.
pub fn drain_b12_gqa_capture() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    B12_GQA_CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

/// Record one named B12 layer-31 GQA tap if a capture window is currently
/// open on this thread. Cheap no-op (single TLS access + borrow) when the
/// window is closed, which is the production case. The tensor is
/// materialized to host F32 immediately so the GPU scratch is free to be
/// reused — same pattern as [`capture_h_main_tap`] /
/// [`capture_b11_layer0_tap`].
pub fn capture_b12_gqa_tap(name: &str, t: &Tensor) -> Result<()> {
    let armed = B12_GQA_CAPTURE.with(|c| c.borrow().is_some());
    if !armed {
        return Ok(());
    }
    let (shape, flat) = tensor_to_f32_host(t)
        .with_context(|| format!("capture_b12_gqa_tap `{name}`: tensor→f32 host copy"))?;
    B12_GQA_CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push((name.to_string(), shape, flat));
        }
    });
    Ok(())
}

// -----------------------------------------------------------------------------
// Phase C6: pre-RoPE MTP-input bisect taps
// -----------------------------------------------------------------------------

/// Canonical ordered list of Phase C6 pre-RoPE tap names. The comparator
/// (`scripts/mtp_compare.py --c6`) expects this exact order in its output
/// table, and the HF reference (`scripts/mtp_reference_dump.py`) emits
/// mirrored `c6__<name>` tensors in the same order.
///
/// Order follows the forward graph inside `mtp_forward_step`, top-down, up
/// to the moment the inner transformer block begins to apply RoPE:
/// 1. `token_emb` — `embed_tokens(draft_token)` output, unsqueezed to [1,1,H]
/// 2. `norm_emb` — `rms_norm(token_emb, pre_fc_norm_embedding)` (pre-fc norm on embed half)
/// 3. `norm_h` — `rms_norm(h_prev, pre_fc_norm_hidden)` (pre-fc norm on hidden half)
/// 4. `concat` — `Tensor::cat([norm_emb, norm_h], dim=2)` post-contiguous
/// 5. `fused` — `concat @ mtp.fc_t` (output of the fused-input linear)
pub const C6_TAP_NAMES: &[&str] = &[
    "token_emb",
    "norm_emb",
    "norm_h",
    "concat",
    "fused",
];

/// True when `KILN_MTP_DUMP_PRE_ROPE=1` (or `true`) is set. Opt-in for the
/// Phase C6 pre-RoPE MTP-input bisect: when enabled alongside
/// `KILN_MTP_DUMP_PATH`, `mtp_forward_step` captures the 5 named tensors in
/// [`C6_TAP_NAMES`] and appends them to the MTP dump safetensors under names
/// `c6__<name>`. Naming-space is distinct from B11/B12 so the existing dump
/// format is unchanged when this flag is unset.
pub fn is_dump_pre_rope_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_PRE_ROPE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// True if the Phase C6 pre-RoPE capture window is currently armed on this
/// thread. Call sites use this to gate the host copy so disarmed runs pay
/// zero cost.
pub fn is_pre_rope_capture_armed() -> bool {
    PRE_ROPE_CAPTURE.with(|c| c.borrow().is_some())
}

/// Begin a C6 pre-RoPE capture window. Subsequent [`capture_pre_rope_tap`]
/// calls from the same thread record into a fresh buffer until
/// [`drain_pre_rope_capture`] is called. Does nothing if
/// [`is_dump_pre_rope_enabled`] is false — so callers can invoke this
/// unconditionally around the MTP inner-block path.
pub fn arm_pre_rope_capture() {
    // Phase C13: the splice meta-flag also drives pre-RoPE capture. Using the
    // effective check here lets callers invoke this unconditionally without
    // needing to set `KILN_MTP_DUMP_PRE_ROPE=1` when splice is on.
    if !is_dump_pre_rope_effectively_enabled() {
        return;
    }
    PRE_ROPE_CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Drain the captured C6 pre-RoPE taps and disarm. Returns whatever was
/// recorded since the matching [`arm_pre_rope_capture`] call, in order.
pub fn drain_pre_rope_capture() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    PRE_ROPE_CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

/// Record one named pre-RoPE tap if a capture window is currently open on
/// this thread. Cheap no-op (single TLS access + borrow) when the window
/// is closed, which is the production case. The tensor is materialized to
/// host F32 immediately so the GPU scratch is free to be reused — same
/// pattern as [`capture_h_main_tap`] / [`capture_b11_layer0_tap`].
pub fn capture_pre_rope_tap(name: &str, t: &Tensor) -> Result<()> {
    let armed = PRE_ROPE_CAPTURE.with(|c| c.borrow().is_some());
    if !armed {
        return Ok(());
    }
    let (shape, flat) = tensor_to_f32_host(t)
        .with_context(|| format!("capture_pre_rope_tap `{name}`: tensor→f32 host copy"))?;
    PRE_ROPE_CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push((name.to_string(), shape, flat));
        }
    });
    Ok(())
}

// -----------------------------------------------------------------------------
// Phase C13: pre-projection splice dump meta-flag
// -----------------------------------------------------------------------------
//
// `KILN_MTP_DUMP_SPLICE=1` is a convenience meta-flag that composes over the
// existing `write_mtp_dump` pipeline. It does three things:
//
//   1. Forces pre-RoPE capture on (so `c6__*` taps are emitted without
//      requiring `KILN_MTP_DUMP_PRE_ROPE=1` to also be set).
//   2. Replaces the one-shot (per process, per `mtp_pos`) latch with an
//      N-step counter (`KILN_MTP_DUMP_SPLICE_MAX_STEPS`, default 8) so up
//      to N draft steps per targeted position land in their own files.
//   3. Defaults `mtp_pos` targets to `{0, 2}` when `KILN_MTP_DUMP_SPLICE_POS`
//      is unset (largest-`n` MTP positions from C5 attribution data).
//
// The dump path accepts `{pos}` and `{step}` placeholders; the pair uniquely
// identifies each file. No behavior change when the flag is unset.

/// True when `KILN_MTP_DUMP_SPLICE=1` (or `true`) is set.
pub fn is_dump_splice_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_SPLICE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Maximum number of splice-dump steps captured per targeted position.
/// Defaults to 8. Set `KILN_MTP_DUMP_SPLICE_MAX_STEPS=N` to override.
pub fn splice_max_steps() -> usize {
    std::env::var("KILN_MTP_DUMP_SPLICE_MAX_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(8)
}

/// Set of `mtp_pos` values for which the splice dump should fire. Defaults
/// to `{0, 2}` when `KILN_MTP_DUMP_SPLICE_POS` is unset. Parses
/// `KILN_MTP_DUMP_SPLICE_POS=0,1,2` as a comma-separated list.
pub fn splice_target_positions() -> HashSet<usize> {
    let raw = std::env::var("KILN_MTP_DUMP_SPLICE_POS").unwrap_or_default();
    if raw.trim().is_empty() {
        let mut s = HashSet::new();
        s.insert(0);
        s.insert(2);
        return s;
    }
    raw.split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

/// Per-position step counters. `None` until the first splice slot is taken.
static SPLICE_STEP_COUNTS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

/// Returns `Some(step)` (0-indexed) when the splice lane is enabled, a
/// `KILN_MTP_DUMP_PATH` is set, `mtp_pos` is targeted, and the per-position
/// counter has not yet reached [`splice_max_steps`]. Advances the counter.
/// Returns `None` in any other case (splice disabled, position not targeted,
/// cap reached, env unset, mutex poisoned).
pub fn try_consume_splice_slot(mtp_pos: usize) -> Option<usize> {
    if !is_dump_splice_enabled() {
        return None;
    }
    if dump_path().is_none() {
        return None;
    }
    if !splice_target_positions().contains(&mtp_pos) {
        return None;
    }
    let max = splice_max_steps();
    let mut guard = SPLICE_STEP_COUNTS.lock().ok()?;
    let map = guard.get_or_insert_with(HashMap::new);
    let count = map.entry(mtp_pos).or_insert(0);
    if *count >= max {
        return None;
    }
    let idx = *count;
    *count += 1;
    Some(idx)
}

/// Dump path with `{pos}` and optional `{step}` substituted. Falls back to
/// [`dump_path_for_pos`] when `step` is `None`. Callers targeting multiple
/// steps per position should include `{step}` in the template to avoid
/// later steps overwriting earlier ones.
pub fn dump_path_for_pos_and_step(mtp_pos: usize, step: Option<usize>) -> Option<String> {
    let raw = dump_path()?;
    let mut path = raw.replace("{pos}", &mtp_pos.to_string());
    if let Some(s) = step {
        path = path.replace("{step}", &s.to_string());
    }
    Some(path)
}

/// OR-composition: pre-RoPE capture should fire when either the explicit
/// `KILN_MTP_DUMP_PRE_ROPE=1` flag is set or the C13 splice meta-flag is
/// set. Use this in the forward dump wiring so splice mode does not need
/// callers to set both env vars.
pub fn is_dump_pre_rope_effectively_enabled() -> bool {
    is_dump_pre_rope_enabled() || is_dump_splice_enabled()
}

/// Test-only: reset the splice step counters between unit tests.
#[cfg(test)]
fn reset_splice_counters() {
    let mut guard = SPLICE_STEP_COUNTS.lock().unwrap();
    *guard = None;
}

// -----------------------------------------------------------------------------
// Phase C7: SDPA-internal MTP attention bisect taps
// -----------------------------------------------------------------------------

/// Canonical ordered list of Phase C7 SDPA-internal tap names. The comparator
/// (`scripts/mtp_compare.py --c7`) expects this exact order in its output
/// table, and the HF reference (`scripts/mtp_reference_dump.py`) emits
/// mirrored `c7__<name>` tensors in the same order.
///
/// Order follows the forward graph inside the GQA attention block used by
/// the MTP inner transformer, top-down, from the moment Q/K/V enter SDPA
/// through the raw attention output:
/// 1. `pre_sdpa_q` — Q tensor handed to SDPA (post-RoPE, post-transpose to `[batch, heads, seq_len, head_dim]`)
/// 2. `pre_sdpa_k` — K tensor handed to SDPA (unexpanded GQA form, `[batch, num_kv_heads, kv_len, head_dim]`)
/// 3. `pre_sdpa_v` — V tensor handed to SDPA (unexpanded GQA form, `[batch, num_kv_heads, kv_len, head_dim]`)
/// 4. `causal_mask` — causal mask applied to scores (empty / no-op for decode, q_len=1 attends to everything)
/// 5. `attn_scores_pre_softmax` — `Q @ K^T / sqrt(d)` before softmax, reshaped to canonical `[batch, num_heads, q_len, kv_len]`
/// 6. `attn_probs` — post-softmax attention probabilities, same canonical shape as scores
/// 7. `attn_out` — raw SDPA output (self-check against existing `post_attn_raw` tap)
pub const C7_SDPA_TAP_NAMES: &[&str] = &[
    "pre_sdpa_q",
    "pre_sdpa_k",
    "pre_sdpa_v",
    "causal_mask",
    "attn_scores_pre_softmax",
    "attn_probs",
    "attn_out",
];

/// True when `KILN_MTP_DUMP_C7_SDPA=1` (or `true`) is set. Opt-in for the
/// Phase C7 SDPA-internal bisect: when enabled alongside `KILN_MTP_DUMP_PATH`,
/// the GQA attention call inside the MTP inner block captures the 7 named
/// tensors in [`C7_SDPA_TAP_NAMES`] and appends them to the MTP dump
/// safetensors under names `c7__<name>`. Naming-space is distinct from
/// B11/B12/C6 so the existing dump format is unchanged when this flag is
/// unset. Also serves as a signal to bypass the fused paged-decode kernel
/// path, which runs a black-box CUDA kernel that doesn't materialize the
/// SDPA intermediates we need to bisect.
pub fn is_dump_c7_sdpa_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_C7_SDPA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// True if the Phase C7 SDPA capture window is currently armed on this
/// thread. Call sites inside `gqa_attention_paged` use this to gate both
/// the host copy (so disarmed runs pay zero cost) AND the fused-kernel
/// bypass (so the grouped-decode Candle path runs and materializes scores
/// / probs as real tensors the capture can see).
pub fn is_c7_sdpa_capture_armed() -> bool {
    if C7_SDPA_CAPTURE_ARMED_THREADS.load(Ordering::Acquire) == 0 {
        return false;
    }
    C7_SDPA_CAPTURE.with(|c| c.borrow().is_some())
}

/// Begin a C7 SDPA capture window. Subsequent [`capture_c7_sdpa_tap`]
/// calls from the same thread record into a fresh buffer until
/// [`drain_c7_sdpa_capture`] is called. Does nothing if
/// [`is_dump_c7_sdpa_enabled`] is false — so callers can invoke this
/// unconditionally around the MTP inner-block path.
pub fn arm_c7_sdpa_capture() {
    if !is_dump_c7_sdpa_enabled() {
        return;
    }
    let was_armed = C7_SDPA_CAPTURE.with(|c| {
        let mut capture = c.borrow_mut();
        let was_armed = capture.is_some();
        *capture = Some(Vec::new());
        was_armed
    });
    if !was_armed {
        C7_SDPA_CAPTURE_ARMED_THREADS.fetch_add(1, Ordering::Release);
    }
}

/// Drain the captured C7 SDPA taps and disarm. Returns whatever was
/// recorded since the matching [`arm_c7_sdpa_capture`] call, in order.
pub fn drain_c7_sdpa_capture() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let (was_armed, drained) = C7_SDPA_CAPTURE.with(|c| {
        let mut capture = c.borrow_mut();
        let was_armed = capture.is_some();
        (was_armed, capture.take().unwrap_or_default())
    });
    if was_armed {
        C7_SDPA_CAPTURE_ARMED_THREADS.fetch_sub(1, Ordering::Release);
    }
    drained
}

/// Phase C8: returns true when the current thread is executing inside an
/// `mtp_forward_step` call that has armed single-token self-attention for
/// the MTP inner block. `gqa_attention_paged` checks this to:
///   * Skip the paged KV cache write (the cache is never read in this mode)
///   * Skip the paged KV cache read (use per-step K/V scratch directly)
///   * Bypass the fused paged-decode flash-attention kernel (which reads
///     over the full cache history, defeating the kv_len=1 contract)
///
/// This implements the Phase C8 fix for the MTP SDPA kv_len mismatch
/// localized in Phase C7. The Qwen3-Next reference contract — see
/// `scripts/mtp_reference_dump.py` and the HF / vLLM
/// `Qwen3NextMultiTokenPredictor` — runs the MTP inner attention as a
/// single-token self-attention over the just-fused hidden state: Q·K^T
/// yields a 1×1 scalar, softmax = 1.0, and the output equals V.
pub fn is_mtp_single_token_self_attn_armed() -> bool {
    MTP_SINGLE_TOKEN_SELF_ATTN_ARMED.with(|c| *c.borrow())
}

/// Arm single-token self-attention for the MTP inner block on this thread.
/// Idempotent within a single `mtp_forward_step` call; the matching
/// [`disarm_mtp_single_token_self_attn`] is required and runs in both the
/// success and the early-error paths inside `mtp_forward_step`.
pub fn arm_mtp_single_token_self_attn() {
    MTP_SINGLE_TOKEN_SELF_ATTN_ARMED.with(|c| *c.borrow_mut() = true);
}

/// Disarm single-token self-attention for the MTP inner block on this
/// thread. Always runs after the inner block returns (success or error)
/// so the next `mtp_forward_step` call (or any non-MTP attention call on
/// this thread) sees the legacy paged-cache behavior.
pub fn disarm_mtp_single_token_self_attn() {
    MTP_SINGLE_TOKEN_SELF_ATTN_ARMED.with(|c| *c.borrow_mut() = false);
}

/// True if the Phase C12 fp32-head TLS flag is currently armed on this
/// thread. Read by `linear_with_lora_t` to decide whether to promote the
/// projection matmul to f32. Cheap single TLS access; the production path
/// pays one atomic load when no thread is inside the fp32-head window. The
/// expensive TLS check is only needed while at least one thread has armed the
/// kill switch. Always `false` outside the window bracketed by
/// [`arm_mtp_fp32_head`] / [`disarm_mtp_fp32_head`] inside `mtp_forward_step`.
pub fn is_mtp_fp32_head_armed() -> bool {
    if MTP_FP32_HEAD_ARMED_THREADS.load(Ordering::Acquire) == 0 {
        return false;
    }
    MTP_FP32_HEAD_ARMED.with(|c| *c.borrow())
}

/// Arm fp32 projection matmuls for the MTP inner block on this thread.
/// Idempotent within a single `mtp_forward_step` call; the matching
/// [`disarm_mtp_fp32_head`] is required and runs in both the success and
/// the early-error paths inside `mtp_forward_step`.
pub fn arm_mtp_fp32_head() {
    let was_armed = MTP_FP32_HEAD_ARMED.with(|c| {
        let mut armed = c.borrow_mut();
        let was_armed = *armed;
        *armed = true;
        was_armed
    });
    if !was_armed {
        MTP_FP32_HEAD_ARMED_THREADS.fetch_add(1, Ordering::AcqRel);
    }
}

/// Disarm fp32 projection matmuls for the MTP inner block on this thread.
/// Always runs after the inner block returns (success or error) so the
/// next `mtp_forward_step` call (or any non-MTP projection call on this
/// thread) sees the legacy bf16 matmul behavior.
pub fn disarm_mtp_fp32_head() {
    let was_armed = MTP_FP32_HEAD_ARMED.with(|c| {
        let mut armed = c.borrow_mut();
        let was_armed = *armed;
        *armed = false;
        was_armed
    });
    if was_armed {
        let _ = MTP_FP32_HEAD_ARMED_THREADS.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |count| Some(count.saturating_sub(1)),
        );
    }
}

/// Record one named SDPA-internal tap if a capture window is currently
/// open on this thread. Cheap no-op (single TLS access + borrow) when the
/// window is closed, which is the production case. The tensor is
/// materialized to host F32 immediately so the GPU scratch is free to be
/// reused — same pattern as [`capture_pre_rope_tap`] /
/// [`capture_b11_layer0_tap`].
pub fn capture_c7_sdpa_tap(name: &str, t: &Tensor) -> Result<()> {
    let armed = C7_SDPA_CAPTURE.with(|c| c.borrow().is_some());
    if !armed {
        return Ok(());
    }
    let (shape, flat) = tensor_to_f32_host(t)
        .with_context(|| format!("capture_c7_sdpa_tap `{name}`: tensor→f32 host copy"))?;
    C7_SDPA_CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push((name.to_string(), shape, flat));
        }
    });
    Ok(())
}

// -----------------------------------------------------------------------------
// Phase C14: post-MTP-transformer-block splice dump taps
// -----------------------------------------------------------------------------

/// Canonical ordered list of Phase C14 post-block tap names. The comparator
/// (`scripts/c14_hf_reference_dump.py`) expects this exact order in its
/// output table, and the HF reference emits mirrored `c14__<name>` tensors
/// in the same order.
///
/// Order follows the forward graph inside `mtp_forward_step`, top-down, from
/// the moment the MTP transformer block returns through the tied lm_head:
/// 1. `post_block` — output of the MTP inner transformer block (pre-norm,
///    the same tensor exposed by the legacy `post_layer` tap but under the
///    C14 prefix for the splice-mode comparator)
/// 2. `post_norm` — output of `mtp.final_layernorm` (pre-lm_head)
/// 3. `logits` — output of the tied lm_head on `post_norm` (pre-softmax)
pub const C14_TAP_NAMES: &[&str] = &["post_block", "post_norm", "logits"];

/// True when `KILN_MTP_DUMP_C14_POST_BLOCK=1` (or `true`) is set. Opt-in for
/// the Phase C14 post-block MTP-output bisect: when enabled alongside
/// `KILN_MTP_DUMP_PATH`, `mtp_forward_step` captures the 3 named tensors in
/// [`C14_TAP_NAMES`] and appends them to the MTP dump safetensors under
/// names `c14__<name>`. Naming-space is distinct from B11/B12/C6/C7 so the
/// existing dump format is unchanged when this flag is unset.
pub fn is_dump_c14_post_block_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_C14_POST_BLOCK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// OR-composition: post-block capture should fire when either the explicit
/// `KILN_MTP_DUMP_C14_POST_BLOCK=1` flag is set or the C13 splice meta-flag
/// is set. Use this in the forward dump wiring so splice mode does not need
/// callers to set both env vars — identical pattern to
/// [`is_dump_pre_rope_effectively_enabled`].
pub fn is_dump_c14_post_block_effectively_enabled() -> bool {
    is_dump_c14_post_block_enabled() || is_dump_splice_enabled()
}

/// True if the Phase C14 post-block capture window is currently armed on
/// this thread. Call sites use this to gate the host copy so disarmed runs
/// pay zero cost.
pub fn is_c14_post_block_capture_armed() -> bool {
    C14_POST_BLOCK_CAPTURE.with(|c| c.borrow().is_some())
}

/// Begin a C14 post-block capture window. Subsequent
/// [`capture_c14_post_block_tap`] calls from the same thread record into a
/// fresh buffer until [`drain_c14_post_block_capture`] is called. Does
/// nothing if [`is_dump_c14_post_block_effectively_enabled`] is false — so
/// callers can invoke this unconditionally around the MTP inner-block path.
pub fn arm_c14_post_block_capture() {
    if !is_dump_c14_post_block_effectively_enabled() {
        return;
    }
    C14_POST_BLOCK_CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
}

/// Drain the captured C14 post-block taps and disarm. Returns whatever was
/// recorded since the matching [`arm_c14_post_block_capture`] call, in order.
pub fn drain_c14_post_block_capture() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    C14_POST_BLOCK_CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default())
}

/// Record one named post-block tap if a capture window is currently open on
/// this thread. Cheap no-op (single TLS access + borrow) when the window is
/// closed, which is the production case. The tensor is materialized to host
/// F32 immediately so the GPU scratch is free to be reused — same pattern as
/// [`capture_pre_rope_tap`] / [`capture_c7_sdpa_tap`].
pub fn capture_c14_post_block_tap(name: &str, t: &Tensor) -> Result<()> {
    let armed = C14_POST_BLOCK_CAPTURE.with(|c| c.borrow().is_some());
    if !armed {
        return Ok(());
    }
    let (shape, flat) = tensor_to_f32_host(t).with_context(|| {
        format!("capture_c14_post_block_tap `{name}`: tensor→f32 host copy")
    })?;
    C14_POST_BLOCK_CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push((name.to_string(), shape, flat));
        }
    });
    Ok(())
}

/// Convert a tensor to a contiguous host float32 `Vec<f32>` plus shape.
/// Used by [`write_mtp_dump`] to serialize taps uniformly regardless of
/// whether the source is BF16 on CUDA or F32 on CPU.
fn tensor_to_f32_host(t: &Tensor) -> Result<(Vec<usize>, Vec<f32>)> {
    let shape = t.dims().to_vec();
    let flat = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok((shape, flat))
}

/// Write the Phase B6 per-tap dump to the path from `KILN_MTP_DUMP_PATH`.
///
/// Output format is `safetensors` (same format kiln's loader already
/// supports, so the Python side can read it with the stock `safetensors`
/// package). All tensors are written as `F32` regardless of source dtype —
/// precision loss from BF16→F32 is one-directional (upcast) and keeps the
/// comparison numerically faithful.
///
/// The named taps correspond one-to-one with the 8 steps in the task brief:
///
/// | Tap name          | Meaning                                              |
/// | ----------------- | ---------------------------------------------------- |
/// | `h_main`          | Last-hidden-state from main model at position t.     |
/// | `tok_embed`       | `embed_tokens(draft_token_t-1)` (pre-norm).          |
/// | `fc_input`        | `concat([norm_emb, norm_h])` (input to `mtp.fc`).    |
/// | `fc_output`       | Post-`mtp.fc` linear projection.                     |
/// | `pre_layer`       | Input to the MTP inner transformer block (== `fc_output`). |
/// | `post_layer`      | Output of the MTP inner transformer block.           |
/// | `post_final_ln`   | Output of `mtp.final_layernorm` (pre-LM head).       |
/// | `mtp_logits`      | Output of `lm_head` on `post_final_ln`.              |
///
/// Plus metadata: `draft_token_id`, `mtp_pos`, `swap_fc_norms`.
///
/// Phase B7b: pass `extra_subops` to append per-sub-op taps captured via
/// the [`arm_subop_capture`] / [`drain_subop_capture`] / [`capture_subop`]
/// trio. These are written under their captured names alongside the
/// static taps; an empty slice makes this behave identically to Phase B6.
///
/// Phase B11: `prompt_tokens` carries the base-model input token ids from
/// the forward pass that produced the h_main value (see
/// [`stash_h_main_prompt_tokens`] / [`drain_h_main_prompt_tokens`]). When
/// non-empty, it is serialized as a flat I32 tensor named `prompt_tokens`
/// plus a 1-element I32 scalar `meta__prompt_tokens_len`. The HF reference
/// (`scripts/mtp_h_main_reference_dump.py`) prefers these tokens over its
/// canonical greeting so both sides replay the exact same prompt during
/// the per-tap bisect. Pass `&[]` (or legacy callers using `&extra_subops`
/// before this param) to skip the prompt-tokens emission.
///
/// Phase B11b: `b11_taps` carries the layer-0 GDN sub-op taps captured via
/// [`arm_b11_layer0_capture`] / [`capture_b11_layer0_tap`] /
/// [`drain_b11_layer0_capture`]. Each entry is serialized as F32 under the
/// name `b11__<name>` so the comparator can select them with a single
/// prefix match. When the slice is empty, the dump bytes are bit-identical
/// to the legacy format (preserved by
/// [`write_and_reparse_mtp_dump_round_trips`]). Note: the task spec called
/// for BF16 serialization; F32 is used here for parity with the existing
/// taps / subops pipeline (single host-materialization path, no `half`
/// dependency). The comparator upcasts HF reference tensors to F32 anyway,
/// so precision is not lost.
///
/// Phase B12: `b12_taps` carries the layer-31 GQA sub-op taps captured via
/// [`arm_b12_gqa_capture`] / [`capture_b12_gqa_tap`] /
/// [`drain_b12_gqa_capture`]. Each entry is serialized as F32 under the
/// name `b12__<name>` (same scheme as B11). When the slice is empty, the
/// dump bytes are bit-identical to the pre-B12 format, so legacy callers
/// passing `&[]` for both `b11_taps` and `b12_taps` produce the historical
/// safetensors layout.
///
/// Phase C6: `c6_taps` carries the pre-RoPE MTP-input taps captured via
/// [`arm_pre_rope_capture`] / [`capture_pre_rope_tap`] /
/// [`drain_pre_rope_capture`]. Each entry is serialized as F32 under the
/// name `c6__<name>` so the comparator can select them with a single
/// prefix match. When the slice is empty, the dump bytes are bit-identical
/// to the pre-C6 format (see `write_and_reparse_mtp_dump_round_trips`).
///
/// Phase C7: `c7_taps` carries the SDPA-internal MTP attention taps
/// captured via [`arm_c7_sdpa_capture`] / [`capture_c7_sdpa_tap`] /
/// [`drain_c7_sdpa_capture`]. Each entry is serialized as F32 under the
/// name `c7__<name>` (same scheme as C6). When the slice is empty, the
/// dump bytes are bit-identical to the pre-C7 format, so legacy callers
/// passing `&[]` for `c7_taps` produce the historical safetensors layout.
///
/// Phase C14: `c14_taps` carries the post-MTP-transformer-block taps
/// captured via [`arm_c14_post_block_capture`] /
/// [`capture_c14_post_block_tap`] / [`drain_c14_post_block_capture`]. Each
/// entry is serialized as F32 under the name `c14__<name>` (same scheme as
/// C6/C7). When the slice is empty, the dump bytes are bit-identical to
/// the pre-C14 format, so legacy callers passing `&[]` for `c14_taps`
/// produce the historical safetensors layout.
pub fn write_mtp_dump(
    path: &str,
    draft_token_id: u32,
    mtp_pos: usize,
    base_pos: usize,
    swap_fc_norms: bool,
    taps: &[(&str, &Tensor)],
    extra_subops: &[(String, Vec<usize>, Vec<f32>)],
    prompt_tokens: &[u32],
    b11_taps: &[(String, Vec<usize>, Vec<f32>)],
    b12_taps: &[(String, Vec<usize>, Vec<f32>)],
    c6_taps: &[(String, Vec<usize>, Vec<f32>)],
    c7_taps: &[(String, Vec<usize>, Vec<f32>)],
    c14_taps: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    // Materialize every tensor to a host byte buffer first so the
    // TensorViews we build below can borrow from a stable backing store.
    // Capacity: static taps + subops + 4 meta + optional (prompt_tokens, len)
    // + b11 taps + b12 taps + c6 taps + c7 taps + c14 taps.
    let prompt_reserve = if prompt_tokens.is_empty() { 0 } else { 2 };
    let mut backings: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> = Vec::with_capacity(
        taps.len()
            + extra_subops.len()
            + 4
            + prompt_reserve
            + b11_taps.len()
            + b12_taps.len()
            + c6_taps.len()
            + c7_taps.len()
            + c14_taps.len(),
    );
    for (name, t) in taps {
        let (shape, flat) = tensor_to_f32_host(t)
            .with_context(|| format!("dump tap `{name}`: tensor→f32 host copy"))?;
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for v in &flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        backings.push(((*name).to_string(), shape, Dtype::F32, bytes));
    }
    for (name, shape, flat) in extra_subops {
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for v in flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        backings.push((name.clone(), shape.clone(), Dtype::F32, bytes));
    }

    // Metadata as 1-element I32 tensors so the Python side can read them
    // with one loader. Widened to i32 because safetensors 0.5 lacks U32 in
    // its serialization path on some builds; I32 is always available.
    let dti = draft_token_id as i32;
    backings.push((
        "meta__draft_token_id".into(),
        vec![1],
        Dtype::I32,
        dti.to_le_bytes().to_vec(),
    ));
    let mpi = mtp_pos as i32;
    backings.push((
        "meta__mtp_pos".into(),
        vec![1],
        Dtype::I32,
        mpi.to_le_bytes().to_vec(),
    ));
    // Phase C3: emit absolute base position so the HF reference can compute
    // `abs_pos = base_pos + mtp_pos` when applying RoPE inside the MTP inner
    // block. Without this the reference used bare `mtp_pos` for RoPE, which
    // diverged from kiln's absolute-position threading (PR #284 / Phase B8).
    let bpi = base_pos as i32;
    backings.push((
        "meta__base_pos".into(),
        vec![1],
        Dtype::I32,
        bpi.to_le_bytes().to_vec(),
    ));
    let swf = if swap_fc_norms { 1i32 } else { 0i32 };
    backings.push((
        "meta__swap_fc_norms".into(),
        vec![1],
        Dtype::I32,
        swf.to_le_bytes().to_vec(),
    ));

    // Phase B11: serialize the base-model prompt tokens so the HF reference
    // can replay the exact same input instead of its canonical fallback.
    if !prompt_tokens.is_empty() {
        let len_i32 = prompt_tokens.len() as i32;
        backings.push((
            "meta__prompt_tokens_len".into(),
            vec![1],
            Dtype::I32,
            len_i32.to_le_bytes().to_vec(),
        ));
        let mut pt_bytes = Vec::with_capacity(prompt_tokens.len() * 4);
        for &tok in prompt_tokens {
            let as_i32 = tok as i32;
            pt_bytes.extend_from_slice(&as_i32.to_le_bytes());
        }
        backings.push((
            "prompt_tokens".into(),
            vec![prompt_tokens.len()],
            Dtype::I32,
            pt_bytes,
        ));
    }

    // Phase B11b: serialize the layer-0 GDN sub-op taps under `b11__<name>`
    // so the comparator (`scripts/mtp_compare.py --b11`) can select them
    // with a single prefix match. When `b11_taps` is empty the loop is a
    // no-op and the on-disk dump is bit-identical to the legacy layout.
    for (name, shape, flat) in b11_taps {
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for v in flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        backings.push((format!("b11__{name}"), shape.clone(), Dtype::F32, bytes));
    }

    // Phase B12: serialize the layer-31 GQA sub-op taps under `b12__<name>`
    // (same scheme as B11). When `b12_taps` is empty the loop is a no-op
    // and the on-disk dump is bit-identical to the pre-B12 layout.
    for (name, shape, flat) in b12_taps {
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for v in flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        backings.push((format!("b12__{name}"), shape.clone(), Dtype::F32, bytes));
    }

    // Phase C6: serialize the pre-RoPE MTP-input taps under `c6__<name>` so
    // the comparator (`scripts/mtp_compare.py --c6`) can select them with a
    // single prefix match. When `c6_taps` is empty the loop is a no-op and
    // the on-disk dump is bit-identical to the pre-C6 layout.
    for (name, shape, flat) in c6_taps {
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for v in flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        backings.push((format!("c6__{name}"), shape.clone(), Dtype::F32, bytes));
    }

    // Phase C7: serialize the SDPA-internal MTP attention taps under
    // `c7__<name>` so the comparator (`scripts/mtp_compare.py --c7`) can
    // select them with a single prefix match. When `c7_taps` is empty the
    // loop is a no-op and the on-disk dump is bit-identical to the pre-C7
    // layout.
    for (name, shape, flat) in c7_taps {
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for v in flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        backings.push((format!("c7__{name}"), shape.clone(), Dtype::F32, bytes));
    }

    // Phase C14: serialize the post-MTP-transformer-block taps under
    // `c14__<name>` (same scheme as C6/C7). When `c14_taps` is empty the
    // loop is a no-op and the on-disk dump is bit-identical to the pre-C14
    // layout.
    for (name, shape, flat) in c14_taps {
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for v in flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        backings.push((format!("c14__{name}"), shape.clone(), Dtype::F32, bytes));
    }

    let mut views: Vec<(String, TensorView)> = Vec::with_capacity(backings.len());
    for (name, shape, dtype, bytes) in &backings {
        let view = TensorView::new(*dtype, shape.clone(), bytes.as_slice())
            .map_err(|e| anyhow::anyhow!("safetensors TensorView::new for `{name}`: {e:?}"))?;
        views.push((name.clone(), view));
    }

    let serialized = safetensors::serialize(views, &None)
        .map_err(|e| anyhow::anyhow!("safetensors::serialize MTP dump: {e:?}"))?;
    std::fs::write(path, serialized).with_context(|| format!("write MTP dump to {path}"))?;
    Ok(())
}

/// Format a top-K list as `"[(id=42, l=12.34), (id=1, l=11.20), ...]"`.
pub fn format_top_k(top: &[(u32, f32)]) -> String {
    let parts: Vec<String> = top
        .iter()
        .map(|(id, l)| format!("(id={}, l={:.3})", id, l))
        .collect();
    format!("[{}]", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn top_k_picks_largest_in_descending_order() {
        let t = Tensor::new(&[1.0_f32, 5.0, 3.0, 4.0, 2.0][..], &Device::Cpu).unwrap();
        let top = top_k_logits(&t, 3).unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1); // logit 5.0
        assert_eq!(top[1].0, 3); // logit 4.0
        assert_eq!(top[2].0, 2); // logit 3.0
    }

    #[test]
    fn l2_norm_matches_hand_calculation() {
        let t = Tensor::new(&[3.0_f32, 4.0][..], &Device::Cpu).unwrap();
        let n = tensor_l2_norm(&t).unwrap();
        assert!((n - 5.0).abs() < 1e-5);
    }

    #[test]
    fn format_top_k_renders_compact_pairs() {
        let s = format_top_k(&[(7, 1.5), (2, 0.5)]);
        assert_eq!(s, "[(id=7, l=1.500), (id=2, l=0.500)]");
    }

    #[test]
    fn should_log_is_false_when_env_unset() {
        // SAFETY: tests are #[cfg(test)] and run with cargo's default serial
        // dispatch within a process. We only mutate KILN_MTP_DEBUG here, so a
        // single set/unset is fine for this check.
        unsafe {
            std::env::remove_var("KILN_MTP_DEBUG");
        }
        reset_counter();
        assert!(!should_log());
    }

    #[test]
    fn dump_slot_fires_once_per_position() {
        // SAFETY: single-threaded test; scoped env mutation.
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_slot_once.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_PATH", &tmp_s);
            std::env::remove_var("KILN_MTP_DUMP_POS");
        }
        reset_dump_latch();
        assert_eq!(dump_path().as_deref(), Some(tmp_s.as_str()));
        // Default targets only pos 0.
        assert!(try_consume_dump_slot_for_pos(0));
        assert!(!try_consume_dump_slot_for_pos(0));
        assert!(!try_consume_dump_slot_for_pos(1));
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_PATH");
        }
    }

    #[test]
    fn dump_slot_supports_multi_position_targets() {
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_multi_pos.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_PATH", &tmp_s);
            std::env::set_var("KILN_MTP_DUMP_POS", "0,1,2");
        }
        reset_dump_latch();
        let positions = target_dump_positions();
        assert!(positions.contains(&0));
        assert!(positions.contains(&1));
        assert!(positions.contains(&2));
        assert!(!positions.contains(&3));
        // Each targeted position fires exactly once.
        assert!(try_consume_dump_slot_for_pos(0));
        assert!(!try_consume_dump_slot_for_pos(0));
        assert!(try_consume_dump_slot_for_pos(1));
        assert!(try_consume_dump_slot_for_pos(2));
        // A non-targeted position never fires.
        assert!(!try_consume_dump_slot_for_pos(3));
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_PATH");
            std::env::remove_var("KILN_MTP_DUMP_POS");
        }
    }

    #[test]
    fn dump_path_for_pos_substitutes_placeholder() {
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_PATH", "/tmp/dump_pos{pos}.safetensors");
        }
        assert_eq!(
            dump_path_for_pos(0).as_deref(),
            Some("/tmp/dump_pos0.safetensors")
        );
        assert_eq!(
            dump_path_for_pos(2).as_deref(),
            Some("/tmp/dump_pos2.safetensors")
        );
        // Without `{pos}`, the path is returned as-is (caller should always
        // include `{pos}` when capturing multiple positions).
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_PATH", "/tmp/dump.safetensors");
        }
        assert_eq!(
            dump_path_for_pos(1).as_deref(),
            Some("/tmp/dump.safetensors")
        );
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_PATH");
        }
    }

    // ---- Phase C13 splice meta-flag ---------------------------------------

    #[test]
    fn splice_disabled_by_default_and_consumes_no_slot() {
        // SAFETY: single-test scoped env mutation; callers downstream remove
        // vars on success.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_SPLICE");
            std::env::remove_var("KILN_MTP_DUMP_PATH");
            std::env::remove_var("KILN_MTP_DUMP_SPLICE_POS");
            std::env::remove_var("KILN_MTP_DUMP_SPLICE_MAX_STEPS");
        }
        reset_splice_counters();
        assert!(!is_dump_splice_enabled());
        assert_eq!(try_consume_splice_slot(0), None);
        assert_eq!(try_consume_splice_slot(2), None);
        assert!(!is_dump_pre_rope_effectively_enabled());
    }

    #[test]
    fn splice_defaults_to_positions_0_and_2_with_8_steps() {
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_SPLICE", "1");
            std::env::set_var(
                "KILN_MTP_DUMP_PATH",
                "/tmp/c13_splice_pos{pos}_step{step}.safetensors",
            );
            std::env::remove_var("KILN_MTP_DUMP_SPLICE_POS");
            std::env::remove_var("KILN_MTP_DUMP_SPLICE_MAX_STEPS");
        }
        reset_splice_counters();

        assert!(is_dump_splice_enabled());
        assert!(is_dump_pre_rope_effectively_enabled());
        let targets = splice_target_positions();
        assert!(targets.contains(&0));
        assert!(targets.contains(&2));
        assert!(!targets.contains(&1));
        assert_eq!(splice_max_steps(), 8);

        // Eight slots fire per targeted position, then stop.
        for step in 0..8 {
            assert_eq!(try_consume_splice_slot(0), Some(step));
        }
        assert_eq!(try_consume_splice_slot(0), None);
        // Position 2 is independent and also gets 8 slots.
        for step in 0..8 {
            assert_eq!(try_consume_splice_slot(2), Some(step));
        }
        assert_eq!(try_consume_splice_slot(2), None);
        // Position 1 is never targeted under defaults.
        assert_eq!(try_consume_splice_slot(1), None);

        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_SPLICE");
            std::env::remove_var("KILN_MTP_DUMP_PATH");
        }
    }

    #[test]
    fn splice_respects_custom_positions_and_max_steps() {
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_SPLICE", "1");
            std::env::set_var("KILN_MTP_DUMP_PATH", "/tmp/c13_custom.safetensors");
            std::env::set_var("KILN_MTP_DUMP_SPLICE_POS", "3,5");
            std::env::set_var("KILN_MTP_DUMP_SPLICE_MAX_STEPS", "2");
        }
        reset_splice_counters();

        let targets = splice_target_positions();
        assert!(targets.contains(&3));
        assert!(targets.contains(&5));
        assert!(!targets.contains(&0));
        assert!(!targets.contains(&2));
        assert_eq!(splice_max_steps(), 2);

        assert_eq!(try_consume_splice_slot(3), Some(0));
        assert_eq!(try_consume_splice_slot(3), Some(1));
        assert_eq!(try_consume_splice_slot(3), None);
        assert_eq!(try_consume_splice_slot(0), None);
        assert_eq!(try_consume_splice_slot(5), Some(0));

        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_SPLICE");
            std::env::remove_var("KILN_MTP_DUMP_PATH");
            std::env::remove_var("KILN_MTP_DUMP_SPLICE_POS");
            std::env::remove_var("KILN_MTP_DUMP_SPLICE_MAX_STEPS");
        }
    }

    #[test]
    fn splice_requires_dump_path_to_fire() {
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_SPLICE", "1");
            std::env::remove_var("KILN_MTP_DUMP_PATH");
            std::env::remove_var("KILN_MTP_DUMP_SPLICE_POS");
            std::env::remove_var("KILN_MTP_DUMP_SPLICE_MAX_STEPS");
        }
        reset_splice_counters();
        assert!(is_dump_splice_enabled());
        // No DUMP_PATH → no slot, even when the meta-flag is on.
        assert_eq!(try_consume_splice_slot(0), None);
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_SPLICE");
        }
    }

    #[test]
    fn dump_path_for_pos_and_step_substitutes_both_placeholders() {
        unsafe {
            std::env::set_var(
                "KILN_MTP_DUMP_PATH",
                "/tmp/c13/mtp_pos-{pos}/step-{step}.safetensors",
            );
        }
        assert_eq!(
            dump_path_for_pos_and_step(0, Some(3)).as_deref(),
            Some("/tmp/c13/mtp_pos-0/step-3.safetensors")
        );
        assert_eq!(
            dump_path_for_pos_and_step(2, Some(7)).as_deref(),
            Some("/tmp/c13/mtp_pos-2/step-7.safetensors")
        );
        // With step=None the `{step}` placeholder is left alone (callers
        // that don't know their step should not use `{step}`).
        assert_eq!(
            dump_path_for_pos_and_step(0, None).as_deref(),
            Some("/tmp/c13/mtp_pos-0/step-{step}.safetensors")
        );
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_PATH");
        }
    }

    #[test]
    fn subop_capture_records_then_drains() {
        let a = Tensor::new(&[1.0_f32, 2.0, 3.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0_f32, 20.0][..], &Device::Cpu).unwrap();
        // Disarmed: capture is a silent no-op.
        capture_subop("post_q_proj", &a).unwrap();
        assert!(drain_subop_capture().is_empty());
        // Armed: records in order.
        arm_subop_capture();
        capture_subop("post_q_proj", &a).unwrap();
        capture_subop("post_k_proj", &b).unwrap();
        let drained = drain_subop_capture();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "post_q_proj");
        assert_eq!(drained[0].1, vec![3]);
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, "post_k_proj");
        // Drain disarms.
        capture_subop("post_v_proj", &a).unwrap();
        assert!(drain_subop_capture().is_empty());
    }

    #[test]
    fn fp32_head_arm_is_thread_local_and_idempotent() {
        disarm_mtp_fp32_head();
        assert!(!is_mtp_fp32_head_armed());

        arm_mtp_fp32_head();
        assert!(is_mtp_fp32_head_armed());

        arm_mtp_fp32_head();
        assert!(is_mtp_fp32_head_armed());

        disarm_mtp_fp32_head();
        assert!(!is_mtp_fp32_head_armed());

        disarm_mtp_fp32_head();
        assert!(!is_mtp_fp32_head_armed());
    }

    #[test]
    fn h_main_capture_records_then_drains() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_HIDDEN_STATES", "1");
        }
        let a = Tensor::new(&[1.0_f32, 2.0, 3.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0_f32, 20.0][..], &Device::Cpu).unwrap();
        // Disarmed: capture is a silent no-op.
        capture_h_main_tap("h_layer_0", &a).unwrap();
        assert!(drain_h_main_capture().is_empty());
        // Armed: records in order.
        arm_h_main_capture();
        assert!(is_h_main_capture_armed());
        assert!(should_capture_hidden_state_for_layer(0));
        assert!(should_capture_hidden_state_for_layer(31));
        assert!(!should_capture_hidden_state_for_layer(5));
        capture_h_main_tap("h_layer_0", &a).unwrap();
        capture_h_main_tap("h_post_final_norm", &b).unwrap();
        let drained = drain_h_main_capture();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "h_layer_0");
        assert_eq!(drained[0].1, vec![3]);
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, "h_post_final_norm");
        // Drain disarms.
        assert!(!is_h_main_capture_armed());
        capture_h_main_tap("h_layer_16", &a).unwrap();
        assert!(drain_h_main_capture().is_empty());
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_HIDDEN_STATES");
        }
    }

    #[test]
    fn h_main_arm_is_noop_when_env_unset() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_HIDDEN_STATES");
        }
        arm_h_main_capture();
        assert!(!is_h_main_capture_armed());
        let a = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        capture_h_main_tap("h_layer_0", &a).unwrap();
        assert!(drain_h_main_capture().is_empty());
    }

    #[test]
    fn stash_h_main_prompt_tokens_requires_armed_capture() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_HIDDEN_STATES");
        }
        // Disarmed: stash is a silent no-op.
        stash_h_main_prompt_tokens(&[1, 2, 3]);
        assert!(drain_h_main_prompt_tokens().is_empty());

        // Armed: stash round-trips through drain.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_HIDDEN_STATES", "1");
        }
        arm_h_main_capture();
        stash_h_main_prompt_tokens(&[10, 20, 30, 40]);
        let drained = drain_h_main_prompt_tokens();
        assert_eq!(drained, vec![10, 20, 30, 40]);
        // Second drain returns empty (moved out).
        assert!(drain_h_main_prompt_tokens().is_empty());
        // Clean up capture state.
        let _ = drain_h_main_capture();
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_HIDDEN_STATES");
        }
    }

    #[test]
    fn write_mtp_dump_emits_prompt_tokens_when_provided() {
        use safetensors::SafeTensors;

        let a = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_with_prompt.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        let prompt = [7u32, 11, 13, 17, 19];
        write_mtp_dump(
            &tmp_s,
            /* draft_token_id = */ 42,
            /* mtp_pos = */ 0,
            /* base_pos = */ 0,
            /* swap_fc_norms = */ false,
            &[("h_main", &a)],
            &[],
            &prompt,
            /* b11_taps = */ &[],
            /* b12_taps = */ &[],
            /* c6_taps = */ &[],
            /* c7_taps = */ &[],
            /* c14_taps = */ &[],
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        assert!(names.contains(&"prompt_tokens"));
        assert!(names.contains(&"meta__prompt_tokens_len"));

        let pt = st.tensor("prompt_tokens").unwrap();
        assert_eq!(pt.dtype(), safetensors::Dtype::I32);
        assert_eq!(pt.shape(), &[prompt.len()]);
        let data = pt.data();
        for (i, &tok) in prompt.iter().enumerate() {
            let slice: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().unwrap();
            assert_eq!(i32::from_le_bytes(slice), tok as i32);
        }

        let len_meta = st.tensor("meta__prompt_tokens_len").unwrap();
        assert_eq!(len_meta.dtype(), safetensors::Dtype::I32);
        let len_val = i32::from_le_bytes(len_meta.data()[0..4].try_into().unwrap());
        assert_eq!(len_val, prompt.len() as i32);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn b10_boundary_layers_span_full_stack() {
        // Guardrail against accidental future edits that would narrow the
        // bisect span. Order matters only for the comparator output, but
        // the set must include layers 0 and 31 so the comparator can verify
        // both the embedding-adjacent and pre-final-norm ends of the stack.
        assert!(B10_BOUNDARY_LAYERS.contains(&0));
        assert!(B10_BOUNDARY_LAYERS.contains(&31));
        assert_eq!(B10_BOUNDARY_LAYERS.len(), 5);
    }

    #[test]
    fn write_and_reparse_mtp_dump_round_trips() {
        use safetensors::SafeTensors;

        let a = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[0.5_f32, -0.25][..], &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_round_trip.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        write_mtp_dump(
            &tmp_s,
            /* draft_token_id = */ 561,
            /* mtp_pos = */ 0,
            /* base_pos = */ 0,
            /* swap_fc_norms = */ false,
            &[("h_main", &a), ("mtp_logits", &b)],
            &[("post_q_proj".to_string(), vec![2], vec![0.1_f32, 0.2])],
            /* prompt_tokens = */ &[],
            /* b11_taps = */ &[],
            /* b12_taps = */ &[],
            /* c6_taps = */ &[],
            /* c7_taps = */ &[],
            /* c14_taps = */ &[],
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        assert!(names.contains(&"h_main"));
        assert!(names.contains(&"mtp_logits"));
        assert!(names.contains(&"meta__draft_token_id"));
        assert!(names.contains(&"meta__mtp_pos"));
        assert!(names.contains(&"meta__base_pos"));
        assert!(names.contains(&"meta__swap_fc_norms"));
        // With prompt_tokens = &[], neither the tensor nor its length meta
        // should be serialized.
        assert!(!names.contains(&"prompt_tokens"));
        assert!(!names.contains(&"meta__prompt_tokens_len"));
        // With b11_taps = &[], no b11__* tensors should appear.
        assert!(!names.iter().any(|n| n.starts_with("b11__")));
        // With b12_taps = &[], no b12__* tensors should appear.
        assert!(!names.iter().any(|n| n.starts_with("b12__")));
        // With c6_taps = &[], no c6__* tensors should appear.
        assert!(!names.iter().any(|n| n.starts_with("c6__")));
        // With c7_taps = &[], no c7__* tensors should appear.
        assert!(!names.iter().any(|n| n.starts_with("c7__")));
        // With c14_taps = &[], no c14__* tensors should appear.
        assert!(!names.iter().any(|n| n.starts_with("c14__")));

        let h = st.tensor("h_main").unwrap();
        assert_eq!(h.dtype(), safetensors::Dtype::F32);
        assert_eq!(h.shape(), &[4]);
        let meta = st.tensor("meta__draft_token_id").unwrap();
        assert_eq!(meta.dtype(), safetensors::Dtype::I32);
        let v = i32::from_le_bytes(meta.data()[0..4].try_into().unwrap());
        assert_eq!(v, 561);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn b11_tap_names_enumerate_all_eleven_layer0_subops() {
        // Guardrail against accidental future edits that would drop a tap
        // from the layer-0 bisect span. Both the Rust capture sites and the
        // Python HF reference dump iterate this list; keeping it fixed at
        // 11 entries is part of the B11b contract.
        assert_eq!(B11_TAP_NAMES.len(), 11);
        assert_eq!(B11_TAP_NAMES[0], "tok_embed");
        assert_eq!(B11_TAP_NAMES[10], "gdn_out_proj");
    }

    #[test]
    fn b11_layer0_capture_records_then_drains() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_B11_TAPS", "1");
        }
        let a = Tensor::new(&[1.0_f32, 2.0, 3.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0_f32, 20.0][..], &Device::Cpu).unwrap();
        // Disarmed: capture is a silent no-op.
        capture_b11_layer0_tap("tok_embed", &a).unwrap();
        assert!(drain_b11_layer0_capture().is_empty());
        assert!(!is_b11_layer0_capture_armed());
        assert!(!should_capture_b11_tap_for_layer(0));

        // Armed: records in order.
        arm_b11_layer0_capture();
        assert!(is_b11_layer0_capture_armed());
        assert!(should_capture_b11_tap_for_layer(0));
        assert!(!should_capture_b11_tap_for_layer(1));
        capture_b11_layer0_tap("tok_embed", &a).unwrap();
        capture_b11_layer0_tap("gdn_out_proj", &b).unwrap();
        let drained = drain_b11_layer0_capture();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "tok_embed");
        assert_eq!(drained[0].1, vec![3]);
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, "gdn_out_proj");
        assert_eq!(drained[1].1, vec![2]);
        assert_eq!(drained[1].2, vec![10.0, 20.0]);

        // Drain disarms.
        assert!(!is_b11_layer0_capture_armed());
        capture_b11_layer0_tap("gdn_conv", &a).unwrap();
        assert!(drain_b11_layer0_capture().is_empty());

        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B11_TAPS");
        }
    }

    #[test]
    fn b11_layer0_arm_is_noop_when_env_unset() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B11_TAPS");
        }
        arm_b11_layer0_capture();
        assert!(!is_b11_layer0_capture_armed());
        assert!(!should_capture_b11_tap_for_layer(0));
        let a = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        capture_b11_layer0_tap("tok_embed", &a).unwrap();
        assert!(drain_b11_layer0_capture().is_empty());
    }

    #[test]
    fn write_mtp_dump_emits_b11_taps_when_provided() {
        use safetensors::SafeTensors;

        let h = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_with_b11.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        let b11 = vec![
            (
                "tok_embed".to_string(),
                vec![2],
                vec![0.11_f32, 0.22],
            ),
            (
                "gdn_out_proj".to_string(),
                vec![3],
                vec![0.31_f32, 0.32, 0.33],
            ),
        ];
        write_mtp_dump(
            &tmp_s,
            /* draft_token_id = */ 99,
            /* mtp_pos = */ 0,
            /* base_pos = */ 0,
            /* swap_fc_norms = */ false,
            &[("h_main", &h)],
            &[],
            /* prompt_tokens = */ &[],
            &b11,
            /* b12_taps = */ &[],
            /* c6_taps = */ &[],
            /* c7_taps = */ &[],
            /* c14_taps = */ &[],
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        // Both taps appear under the b11__<name> namespace.
        assert!(names.contains(&"b11__tok_embed"));
        assert!(names.contains(&"b11__gdn_out_proj"));

        let te = st.tensor("b11__tok_embed").unwrap();
        assert_eq!(te.dtype(), safetensors::Dtype::F32);
        assert_eq!(te.shape(), &[2]);
        let bytes = te.data();
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((v0 - 0.11).abs() < 1e-6);
        assert!((v1 - 0.22).abs() < 1e-6);

        let op = st.tensor("b11__gdn_out_proj").unwrap();
        assert_eq!(op.dtype(), safetensors::Dtype::F32);
        assert_eq!(op.shape(), &[3]);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn b12_gqa_layers_span_full_attention_tail() {
        // Guardrail: the 8 GQA layers at the end of the Qwen3.5-4B stack are
        // 24..=31 inclusive. Comparator + reference both iterate this list.
        assert_eq!(B12_GQA_LAYERS.len(), 8);
        assert_eq!(B12_GQA_LAYERS[0], 24);
        assert_eq!(B12_GQA_LAYERS[7], 31);
    }

    #[test]
    fn b12_gqa_tap_names_enumerate_all_fourteen_subops() {
        // Guardrail against accidental future edits that would drop a tap
        // from the layer-31 GQA bisect span. Both the Rust capture sites and
        // the Python HF reference dump iterate this list.
        assert_eq!(B12_GQA_TAP_NAMES.len(), 14);
        assert_eq!(B12_GQA_TAP_NAMES[0], "post_input_norm");
        assert_eq!(B12_GQA_TAP_NAMES[13], "mlp_down");
    }

    #[test]
    fn b12_gqa_capture_records_then_drains() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_B12_GQA_TAPS", "1");
        }
        let a = Tensor::new(&[1.0_f32, 2.0, 3.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0_f32, 20.0][..], &Device::Cpu).unwrap();
        // Disarmed: capture is a silent no-op.
        capture_b12_gqa_tap("post_input_norm", &a).unwrap();
        assert!(drain_b12_gqa_capture().is_empty());
        assert!(!is_b12_gqa_capture_armed());
        assert!(!should_capture_b12_gqa_tap_for_layer(31));

        // Armed: records in order, only fires on layer 31.
        arm_b12_gqa_capture();
        assert!(is_b12_gqa_capture_armed());
        assert!(should_capture_b12_gqa_tap_for_layer(31));
        assert!(!should_capture_b12_gqa_tap_for_layer(30));
        capture_b12_gqa_tap("post_input_norm", &a).unwrap();
        capture_b12_gqa_tap("mlp_down", &b).unwrap();
        let drained = drain_b12_gqa_capture();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "post_input_norm");
        assert_eq!(drained[0].1, vec![3]);
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, "mlp_down");
        assert_eq!(drained[1].1, vec![2]);
        assert_eq!(drained[1].2, vec![10.0, 20.0]);

        // Drain disarms.
        assert!(!is_b12_gqa_capture_armed());
        capture_b12_gqa_tap("q_proj", &a).unwrap();
        assert!(drain_b12_gqa_capture().is_empty());

        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B12_GQA_TAPS");
        }
    }

    #[test]
    fn b12_gqa_arm_is_noop_when_env_unset() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B12_GQA_TAPS");
        }
        arm_b12_gqa_capture();
        assert!(!is_b12_gqa_capture_armed());
        assert!(!should_capture_b12_gqa_tap_for_layer(31));
        let a = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        capture_b12_gqa_tap("post_input_norm", &a).unwrap();
        assert!(drain_b12_gqa_capture().is_empty());
    }

    #[test]
    fn write_mtp_dump_emits_b12_taps_when_provided() {
        use safetensors::SafeTensors;

        let h = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_with_b12.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        let b12 = vec![
            (
                "post_input_norm".to_string(),
                vec![2],
                vec![0.41_f32, 0.42],
            ),
            (
                "mlp_down".to_string(),
                vec![3],
                vec![0.51_f32, 0.52, 0.53],
            ),
        ];
        write_mtp_dump(
            &tmp_s,
            /* draft_token_id = */ 77,
            /* mtp_pos = */ 0,
            /* base_pos = */ 0,
            /* swap_fc_norms = */ false,
            &[("h_main", &h)],
            &[],
            /* prompt_tokens = */ &[],
            /* b11_taps = */ &[],
            &b12,
            /* c6_taps = */ &[],
            /* c7_taps = */ &[],
            /* c14_taps = */ &[],
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        // Both taps appear under the b12__<name> namespace.
        assert!(names.contains(&"b12__post_input_norm"));
        assert!(names.contains(&"b12__mlp_down"));
        // No b11__ taps when b11_taps is empty.
        assert!(!names.iter().any(|n| n.starts_with("b11__")));

        let pin = st.tensor("b12__post_input_norm").unwrap();
        assert_eq!(pin.dtype(), safetensors::Dtype::F32);
        assert_eq!(pin.shape(), &[2]);
        let bytes = pin.data();
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((v0 - 0.41).abs() < 1e-6);
        assert!((v1 - 0.42).abs() < 1e-6);

        let md = st.tensor("b12__mlp_down").unwrap();
        assert_eq!(md.dtype(), safetensors::Dtype::F32);
        assert_eq!(md.shape(), &[3]);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn b12_layer_scope_disarmed_returns_false_regardless_of_scope() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B12_GQA_TAPS");
        }
        // Without arming, setting layer 31 must still return false so that
        // production runs (where capture is disabled) short-circuit the
        // per-call capture gate.
        enter_b12_layer_scope(31);
        assert!(!current_b12_layer_is_31());
        exit_b12_layer_scope();
    }

    #[test]
    fn b12_layer_scope_armed_without_scope_returns_false() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_B12_GQA_TAPS", "1");
        }
        arm_b12_gqa_capture();
        assert!(is_b12_gqa_capture_armed());
        // Armed but no enter_b12_layer_scope call: the gate returns false so
        // no capture fires for the MTP inner-block path which never enters a
        // layer scope.
        assert!(!current_b12_layer_is_31());
        let _ = drain_b12_gqa_capture();
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B12_GQA_TAPS");
        }
    }

    #[test]
    fn b12_layer_scope_armed_non_31_layer_returns_false() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_B12_GQA_TAPS", "1");
        }
        arm_b12_gqa_capture();
        for layer in [0_usize, 23, 24, 30, 32, 63] {
            enter_b12_layer_scope(layer);
            assert!(
                !current_b12_layer_is_31(),
                "layer {layer} must not trigger layer-31 capture",
            );
            exit_b12_layer_scope();
        }
        let _ = drain_b12_gqa_capture();
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B12_GQA_TAPS");
        }
    }

    #[test]
    fn b12_layer_scope_armed_layer_31_returns_true() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_B12_GQA_TAPS", "1");
        }
        arm_b12_gqa_capture();
        enter_b12_layer_scope(31);
        assert!(current_b12_layer_is_31());
        exit_b12_layer_scope();
        let _ = drain_b12_gqa_capture();
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B12_GQA_TAPS");
        }
    }

    #[test]
    fn b12_layer_scope_exit_clears_slot() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_B12_GQA_TAPS", "1");
        }
        arm_b12_gqa_capture();
        enter_b12_layer_scope(31);
        assert!(current_b12_layer_is_31());
        exit_b12_layer_scope();
        // After exit, the TLS slot is cleared, so a subsequent check must
        // return false — the next layer's forward must not leak capture.
        assert!(!current_b12_layer_is_31());
        let _ = drain_b12_gqa_capture();
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_B12_GQA_TAPS");
        }
    }

    #[test]
    fn c6_tap_names_enumerate_all_five_pre_rope_taps() {
        // Guardrail against accidental future edits that would drop a tap
        // from the pre-RoPE bisect span. Both the Rust capture sites and
        // the Python HF reference dump iterate this list.
        assert_eq!(C6_TAP_NAMES.len(), 5);
        assert_eq!(C6_TAP_NAMES[0], "token_emb");
        assert_eq!(C6_TAP_NAMES[1], "norm_emb");
        assert_eq!(C6_TAP_NAMES[2], "norm_h");
        assert_eq!(C6_TAP_NAMES[3], "concat");
        assert_eq!(C6_TAP_NAMES[4], "fused");
    }

    #[test]
    fn c6_pre_rope_capture_records_then_drains() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_PRE_ROPE", "1");
        }
        let a = Tensor::new(&[1.0_f32, 2.0, 3.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0_f32, 20.0][..], &Device::Cpu).unwrap();
        // Disarmed: capture is a silent no-op.
        capture_pre_rope_tap("token_emb", &a).unwrap();
        assert!(drain_pre_rope_capture().is_empty());
        assert!(!is_pre_rope_capture_armed());

        // Armed: records in order.
        arm_pre_rope_capture();
        assert!(is_pre_rope_capture_armed());
        capture_pre_rope_tap("token_emb", &a).unwrap();
        capture_pre_rope_tap("fused", &b).unwrap();
        let drained = drain_pre_rope_capture();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "token_emb");
        assert_eq!(drained[0].1, vec![3]);
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, "fused");
        assert_eq!(drained[1].1, vec![2]);
        assert_eq!(drained[1].2, vec![10.0, 20.0]);

        // Drain disarms.
        assert!(!is_pre_rope_capture_armed());
        capture_pre_rope_tap("norm_emb", &a).unwrap();
        assert!(drain_pre_rope_capture().is_empty());

        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_PRE_ROPE");
        }
    }

    #[test]
    fn c6_pre_rope_arm_is_noop_when_env_unset() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_PRE_ROPE");
        }
        arm_pre_rope_capture();
        assert!(!is_pre_rope_capture_armed());
        let a = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        capture_pre_rope_tap("token_emb", &a).unwrap();
        assert!(drain_pre_rope_capture().is_empty());
    }

    #[test]
    fn write_mtp_dump_emits_c6_taps_when_provided() {
        use safetensors::SafeTensors;

        let h = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_with_c6.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        let c6 = vec![
            (
                "token_emb".to_string(),
                vec![2],
                vec![0.61_f32, 0.62],
            ),
            (
                "fused".to_string(),
                vec![3],
                vec![0.71_f32, 0.72, 0.73],
            ),
        ];
        write_mtp_dump(
            &tmp_s,
            /* draft_token_id = */ 55,
            /* mtp_pos = */ 0,
            /* base_pos = */ 0,
            /* swap_fc_norms = */ false,
            &[("h_main", &h)],
            &[],
            /* prompt_tokens = */ &[],
            /* b11_taps = */ &[],
            /* b12_taps = */ &[],
            &c6,
            /* c7_taps = */ &[],
            /* c14_taps = */ &[],
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        // Both taps appear under the c6__<name> namespace.
        assert!(names.contains(&"c6__token_emb"));
        assert!(names.contains(&"c6__fused"));
        // No b11__ / b12__ taps when those slices are empty.
        assert!(!names.iter().any(|n| n.starts_with("b11__")));
        assert!(!names.iter().any(|n| n.starts_with("b12__")));

        let te = st.tensor("c6__token_emb").unwrap();
        assert_eq!(te.dtype(), safetensors::Dtype::F32);
        assert_eq!(te.shape(), &[2]);
        let bytes = te.data();
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((v0 - 0.61).abs() < 1e-6);
        assert!((v1 - 0.62).abs() < 1e-6);

        let fu = st.tensor("c6__fused").unwrap();
        assert_eq!(fu.dtype(), safetensors::Dtype::F32);
        assert_eq!(fu.shape(), &[3]);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn c7_tap_names_enumerate_all_seven_sdpa_taps() {
        // Guardrail against accidental future edits that would drop a tap
        // from the SDPA-internal bisect span. Both the Rust capture sites
        // and the Python HF reference dump iterate this list; the
        // comparator relies on exact ordering for its verdict table.
        assert_eq!(C7_SDPA_TAP_NAMES.len(), 7);
        assert_eq!(C7_SDPA_TAP_NAMES[0], "pre_sdpa_q");
        assert_eq!(C7_SDPA_TAP_NAMES[1], "pre_sdpa_k");
        assert_eq!(C7_SDPA_TAP_NAMES[2], "pre_sdpa_v");
        assert_eq!(C7_SDPA_TAP_NAMES[3], "causal_mask");
        assert_eq!(C7_SDPA_TAP_NAMES[4], "attn_scores_pre_softmax");
        assert_eq!(C7_SDPA_TAP_NAMES[5], "attn_probs");
        assert_eq!(C7_SDPA_TAP_NAMES[6], "attn_out");
    }

    #[test]
    fn c7_sdpa_capture_records_then_drains() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_C7_SDPA", "1");
        }
        let a = Tensor::new(&[1.0_f32, 2.0, 3.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0_f32, 20.0][..], &Device::Cpu).unwrap();
        // Disarmed: capture is a silent no-op.
        capture_c7_sdpa_tap("pre_sdpa_q", &a).unwrap();
        assert!(drain_c7_sdpa_capture().is_empty());
        assert!(!is_c7_sdpa_capture_armed());

        // Armed: records in order.
        arm_c7_sdpa_capture();
        assert!(is_c7_sdpa_capture_armed());
        capture_c7_sdpa_tap("pre_sdpa_q", &a).unwrap();
        capture_c7_sdpa_tap("attn_out", &b).unwrap();
        let drained = drain_c7_sdpa_capture();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "pre_sdpa_q");
        assert_eq!(drained[0].1, vec![3]);
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, "attn_out");
        assert_eq!(drained[1].1, vec![2]);
        assert_eq!(drained[1].2, vec![10.0, 20.0]);

        // Drain disarms.
        assert!(!is_c7_sdpa_capture_armed());
        capture_c7_sdpa_tap("attn_probs", &a).unwrap();
        assert!(drain_c7_sdpa_capture().is_empty());

        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_C7_SDPA");
        }
    }

    #[test]
    fn c7_sdpa_arm_is_noop_when_env_unset() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_C7_SDPA");
        }
        arm_c7_sdpa_capture();
        assert!(!is_c7_sdpa_capture_armed());
        let a = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        capture_c7_sdpa_tap("pre_sdpa_q", &a).unwrap();
        assert!(drain_c7_sdpa_capture().is_empty());
    }

    #[test]
    fn write_mtp_dump_emits_c7_taps_when_provided() {
        use safetensors::SafeTensors;

        let h = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_with_c7.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        let c7 = vec![
            (
                "pre_sdpa_q".to_string(),
                vec![2],
                vec![0.71_f32, 0.72],
            ),
            (
                "attn_out".to_string(),
                vec![3],
                vec![0.81_f32, 0.82, 0.83],
            ),
        ];
        write_mtp_dump(
            &tmp_s,
            /* draft_token_id = */ 33,
            /* mtp_pos = */ 0,
            /* base_pos = */ 0,
            /* swap_fc_norms = */ false,
            &[("h_main", &h)],
            &[],
            /* prompt_tokens = */ &[],
            /* b11_taps = */ &[],
            /* b12_taps = */ &[],
            /* c6_taps = */ &[],
            &c7,
            /* c14_taps = */ &[],
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        // Both taps appear under the c7__<name> namespace.
        assert!(names.contains(&"c7__pre_sdpa_q"));
        assert!(names.contains(&"c7__attn_out"));
        // No b11__ / b12__ / c6__ taps when those slices are empty.
        assert!(!names.iter().any(|n| n.starts_with("b11__")));
        assert!(!names.iter().any(|n| n.starts_with("b12__")));
        assert!(!names.iter().any(|n| n.starts_with("c6__")));

        let q = st.tensor("c7__pre_sdpa_q").unwrap();
        assert_eq!(q.dtype(), safetensors::Dtype::F32);
        assert_eq!(q.shape(), &[2]);
        let bytes = q.data();
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((v0 - 0.71).abs() < 1e-6);
        assert!((v1 - 0.72).abs() < 1e-6);

        let out = st.tensor("c7__attn_out").unwrap();
        assert_eq!(out.dtype(), safetensors::Dtype::F32);
        assert_eq!(out.shape(), &[3]);

        let _ = std::fs::remove_file(&tmp);
    }

    // -------------------------------------------------------------------------
    // Phase C14: post-MTP-transformer-block splice dump taps
    // -------------------------------------------------------------------------

    #[test]
    fn c14_tap_names_enumerate_all_three_post_block_taps() {
        // Guardrail: the post-MTP-transformer-block splice window is exactly
        // three taps (output of the MTP block before final norm, after final
        // norm, and post-lm_head logits). Both the Rust capture sites and the
        // Python HF reference dump iterate this list.
        assert_eq!(C14_TAP_NAMES.len(), 3);
        assert_eq!(C14_TAP_NAMES[0], "post_block");
        assert_eq!(C14_TAP_NAMES[1], "post_norm");
        assert_eq!(C14_TAP_NAMES[2], "logits");
    }

    #[test]
    fn c14_post_block_capture_records_then_drains() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::set_var("KILN_MTP_DUMP_C14_POST_BLOCK", "1");
        }
        let a = Tensor::new(&[1.0_f32, 2.0, 3.0][..], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0_f32, 20.0][..], &Device::Cpu).unwrap();
        // Disarmed: capture is a silent no-op.
        capture_c14_post_block_tap("post_block", &a).unwrap();
        assert!(drain_c14_post_block_capture().is_empty());
        assert!(!is_c14_post_block_capture_armed());

        // Armed: records in order.
        arm_c14_post_block_capture();
        assert!(is_c14_post_block_capture_armed());
        capture_c14_post_block_tap("post_block", &a).unwrap();
        capture_c14_post_block_tap("logits", &b).unwrap();
        let drained = drain_c14_post_block_capture();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "post_block");
        assert_eq!(drained[0].1, vec![3]);
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, "logits");
        assert_eq!(drained[1].1, vec![2]);
        assert_eq!(drained[1].2, vec![10.0, 20.0]);

        // Drain disarms.
        assert!(!is_c14_post_block_capture_armed());
        capture_c14_post_block_tap("post_norm", &a).unwrap();
        assert!(drain_c14_post_block_capture().is_empty());

        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_C14_POST_BLOCK");
        }
    }

    #[test]
    fn c14_post_block_arm_is_noop_when_env_unset() {
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_C14_POST_BLOCK");
            std::env::remove_var("KILN_MTP_DUMP_SPLICE");
        }
        arm_c14_post_block_capture();
        assert!(!is_c14_post_block_capture_armed());
        let a = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        capture_c14_post_block_tap("post_block", &a).unwrap();
        assert!(drain_c14_post_block_capture().is_empty());
    }

    #[test]
    fn c14_post_block_splice_flag_enables_capture() {
        // OR-composition contract: setting the C13 splice meta-flag alone is
        // sufficient to enable C14 capture, without needing a separate
        // explicit per-phase opt-in. This is what lets a single
        // KILN_MTP_DUMP_SPLICE=1 invocation drive the full splice-bisect
        // window (C6 + C7 + C14) end-to-end.
        //
        // SAFETY: single-threaded test; scoped env mutation.
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_C14_POST_BLOCK");
            std::env::set_var("KILN_MTP_DUMP_SPLICE", "1");
        }
        assert!(!is_dump_c14_post_block_enabled());
        assert!(is_dump_c14_post_block_effectively_enabled());
        arm_c14_post_block_capture();
        assert!(is_c14_post_block_capture_armed());
        let _ = drain_c14_post_block_capture();
        unsafe {
            std::env::remove_var("KILN_MTP_DUMP_SPLICE");
        }
    }

    #[test]
    fn write_mtp_dump_emits_c14_taps_when_provided() {
        use safetensors::SafeTensors;

        let h = Tensor::new(&[1.0_f32, 2.0][..], &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("kiln_mtp_dump_with_c14.safetensors");
        let tmp_s = tmp.to_string_lossy().into_owned();
        let c14 = vec![
            (
                "post_block".to_string(),
                vec![2],
                vec![0.91_f32, 0.92],
            ),
            (
                "logits".to_string(),
                vec![3],
                vec![1.01_f32, 1.02, 1.03],
            ),
        ];
        write_mtp_dump(
            &tmp_s,
            /* draft_token_id = */ 22,
            /* mtp_pos = */ 0,
            /* base_pos = */ 0,
            /* swap_fc_norms = */ false,
            &[("h_main", &h)],
            &[],
            /* prompt_tokens = */ &[],
            /* b11_taps = */ &[],
            /* b12_taps = */ &[],
            /* c6_taps = */ &[],
            /* c7_taps = */ &[],
            &c14,
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        // Both taps appear under the c14__<name> namespace.
        assert!(names.contains(&"c14__post_block"));
        assert!(names.contains(&"c14__logits"));
        // No b11__ / b12__ / c6__ / c7__ taps when those slices are empty.
        assert!(!names.iter().any(|n| n.starts_with("b11__")));
        assert!(!names.iter().any(|n| n.starts_with("b12__")));
        assert!(!names.iter().any(|n| n.starts_with("c6__")));
        assert!(!names.iter().any(|n| n.starts_with("c7__")));

        let pb = st.tensor("c14__post_block").unwrap();
        assert_eq!(pb.dtype(), safetensors::Dtype::F32);
        assert_eq!(pb.shape(), &[2]);
        let bytes = pb.data();
        let v0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let v1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert!((v0 - 0.91).abs() < 1e-6);
        assert!((v1 - 0.92).abs() < 1e-6);

        let lg = st.tensor("c14__logits").unwrap();
        assert_eq!(lg.dtype(), safetensors::Dtype::F32);
        assert_eq!(lg.shape(), &[3]);

        let _ = std::fs::remove_file(&tmp);
    }
}
