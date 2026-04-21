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
use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
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
    /// states (`h_layer_*` + `h_pre_final_norm`). Kept distinct from
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
/// plus two GQA samples (23, 31). Layer 31's output IS the pre-final-norm
/// hidden state, but the comparator still writes a separate `h_pre_final_norm`
/// tap so the final-layer → h_main handoff can be verified independently.
pub const B10_BOUNDARY_LAYERS: [usize; 5] = [0, 8, 16, 23, 31];

/// True when `KILN_MTP_DUMP_HIDDEN_STATES=1` (or `true`) is set. Phase B10
/// opt-in: when enabled alongside `KILN_MTP_DUMP_PATH`, the base-model
/// forward records the last-row hidden state at each boundary layer and at
/// the pre-final-norm site, and appends them to the MTP dump safetensors
/// under names `h_layer_0`, `h_layer_8`, `h_layer_16`, `h_layer_23`,
/// `h_layer_31`, and `h_pre_final_norm`.
pub fn is_dump_hidden_states_enabled() -> bool {
    std::env::var("KILN_MTP_DUMP_HIDDEN_STATES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// True if the Phase B10 base-model hidden-state capture window is currently
/// open on this thread AND the given layer index is one of the boundary
/// layers. Call sites use this to gate the (relatively cheap) per-layer
/// tensor-narrow + host copy so disarmed runs pay zero cost.
pub fn should_capture_hidden_state_for_layer(layer_idx: usize) -> bool {
    let armed = H_MAIN_CAPTURE.with(|c| c.borrow().is_some());
    armed && B10_BOUNDARY_LAYERS.contains(&layer_idx)
}

/// True if the Phase B10 hidden-state capture window is currently armed on
/// this thread. Used to gate `h_pre_final_norm` capture (which is not
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
pub fn write_mtp_dump(
    path: &str,
    draft_token_id: u32,
    mtp_pos: usize,
    swap_fc_norms: bool,
    taps: &[(&str, &Tensor)],
    extra_subops: &[(String, Vec<usize>, Vec<f32>)],
    prompt_tokens: &[u32],
    b11_taps: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    // Materialize every tensor to a host byte buffer first so the
    // TensorViews we build below can borrow from a stable backing store.
    // Capacity: static taps + subops + 3 meta + optional (prompt_tokens, len)
    // + b11 taps.
    let prompt_reserve = if prompt_tokens.is_empty() { 0 } else { 2 };
    let mut backings: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> = Vec::with_capacity(
        taps.len() + extra_subops.len() + 3 + prompt_reserve + b11_taps.len(),
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
        capture_h_main_tap("h_pre_final_norm", &b).unwrap();
        let drained = drain_h_main_capture();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "h_layer_0");
        assert_eq!(drained[0].1, vec![3]);
        assert_eq!(drained[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(drained[1].0, "h_pre_final_norm");
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
            /* swap_fc_norms = */ false,
            &[("h_main", &a)],
            &[],
            &prompt,
            /* b11_taps = */ &[],
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
            /* swap_fc_norms = */ false,
            &[("h_main", &a), ("mtp_logits", &b)],
            &[("post_q_proj".to_string(), vec![2], vec![0.1_f32, 0.2])],
            /* prompt_tokens = */ &[],
            /* b11_taps = */ &[],
        )
        .unwrap();

        let raw = std::fs::read(&tmp).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let names: Vec<&str> = st.names().into_iter().map(|s| s.as_str()).collect();
        assert!(names.contains(&"h_main"));
        assert!(names.contains(&"mtp_logits"));
        assert!(names.contains(&"meta__draft_token_id"));
        assert!(names.contains(&"meta__mtp_pos"));
        assert!(names.contains(&"meta__swap_fc_norms"));
        // With prompt_tokens = &[], neither the tensor nor its length meta
        // should be serialized.
        assert!(!names.contains(&"prompt_tokens"));
        assert!(!names.contains(&"meta__prompt_tokens_len"));
        // With b11_taps = &[], no b11__* tensors should appear.
        assert!(!names.iter().any(|n| n.starts_with("b11__")));

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
            /* swap_fc_norms = */ false,
            &[("h_main", &h)],
            &[],
            /* prompt_tokens = */ &[],
            &b11,
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
}
