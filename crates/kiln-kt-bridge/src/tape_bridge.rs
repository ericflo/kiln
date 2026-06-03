//! kt-native tape scopes for tape-authoritative backward (#1082).
//!
//! After the candle drop (#1082), training is fully kt-native: the
//! differentiable leaves are kt `Parameter`s, the forward records onto a
//! thread-local `kiln_autograd::Tape`, and the backward walks that tape
//! directly — there is no candle `GradStore` round-trip anymore.
//!
//! This module provides the per-thread IO-mapping scope plus two
//! tape-authoritative backward drivers:
//!
//! * [`with_io_mapping_scope`] — opens ONLY the IO-mapping scope so adapters'
//!   [`register_input_mapping_kt`] calls become live, while the caller drives
//!   the tape walk itself.
//! * [`with_tape_authoritative_scope_kt`] — runs `forward()` under a fresh
//!   thread-local tape, seeds `dL/dL = 1` at the kt scalar loss (the recorded
//!   tape root), walks the tape, and returns the per-input kt gradients keyed
//!   by their recorded deposit ids.
//! * [`with_tape_segment_backward_scope`] — per-segment variant for gradient
//!   checkpointing: seeds an arbitrary segment OUTPUT with an externally
//!   supplied upstream gradient instead of a loss root.
//!
//! The deposit map (`IoMappingScope::kt_to_candle_input`) carries the kt
//! input id → deposit id pairs registered during forward; LoRA-`Parameter`
//! deposits are tagged with [`KT_PARAM_DEPOSIT_TAG`] (see [`register_input_mapping_kt`])
//! so producers can distinguish them via [`decode_kt_param_deposit`].

#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan", feature = "rocm"))]

use std::cell::RefCell;
use std::collections::HashMap;

use kiln_autograd::{with_thread_local_tape, Tape};
use kiln_tensor::TensorId as KtTensorId;

use crate::BridgeError;

/// One side of the bridge's per-thread mapping state.
///
/// Tracks the kt TensorId → deposit-id pairs registered by tape adapters
/// during forward, so the tape-authoritative backward walker can deposit
/// each kt-side input gradient under its recorded deposit id.
#[derive(Debug, Default)]
struct IoMappingScope {
    /// `kt_input_id → deposit_id(s)` for inputs registered by adapters.
    /// Used by the tape-authoritative backward to know which key to insert
    /// each kt-side input gradient under in the returned grad map.
    ///
    /// #1082 CP-4: a `kt_input_id` may fan out to MULTIPLE deposit ids now
    /// that the bridge helpers CHAIN (a recorded kt tensor is reused across
    /// island boundaries, so the same kt feeds several adapters whose
    /// inputs differ). Mapping kt → set of ids keeps every correspondence;
    /// the backward deposit visits all of them so a LoRA param's grad still
    /// lands under its id even when the same kt also feeds an intermediate
    /// island input.
    ///
    /// (#1082) LoRA-`Parameter` deposits are tagged with
    /// [`KT_PARAM_DEPOSIT_TAG`] via [`register_input_mapping_kt`]; producers
    /// decode them with [`decode_kt_param_deposit`].
    kt_to_candle_input: HashMap<u64, Vec<usize>>,
}

thread_local! {
    /// Active bridge scope on this thread. `Some` only inside a
    /// `with_io_mapping_scope` block. Outside the scope,
    /// `register_input_mapping_kt` is a no-op.
    static BRIDGE_SCOPE: RefCell<Option<IoMappingScope>> = const { RefCell::new(None) };
}

/// Namespace tag OR'd onto kt-leaf deposit ids stored in the (shared,
/// `usize`-keyed) `kt_to_candle_input` deposit map by
/// [`register_input_mapping_kt`].
///
/// # Why this is REQUIRED (#1082 candle-drop grad-shape regression)
///
/// Historically the deposit map mixed ids from TWO independent id namespaces:
/// the (now-removed) candle-keyed `register_input_mapping` stored **candle**
/// `TensorId` raws (frozen base weights, intermediate activations, frozen
/// RMSNorm weights, …) while [`register_input_mapping_kt`] stores **kt**
/// `TensorId` raws (the LoRA `Parameter` leaves the optimiser actually trains).
/// candle's `TensorId` (`AtomicUsize::new(1)`) and kt's `TensorId`
/// (`AtomicU64::new(1)`) are *separate* process-global counters that BOTH start
/// at 1 and increment by 1, so their raw values overlap heavily within a single
/// process.
///
/// Before the candle-drop, LoRA leaves were candle `Var`s registered via the
/// candle-keyed path, so the deposit map held candle ids ONLY — one id space,
/// no collisions. The flip moved LoRA leaves to kt `Parameter` +
/// `register_input_mapping_kt`, which silently injected kt ids into the same
/// `usize` map. A candle id colliding with a kt LoRA-param id then overwrote
/// that param's slot in the per-scope `out` deposit map, delivering a frozen
/// tensor's grad to a LoRA param (observed: a frozen RMSNorm `[hidden]` grad
/// landing on the `in_proj_z` LoRA-B `[out_features, rank]` param → optimizer
/// shape mismatch `[32] != [32, 4]`).
///
/// The candle-keyed path is fully removed now (#1082), so only kt-param
/// deposits flow through this map. The tag is retained because [`decode_kt_param_deposit`]
/// is still how producers distinguish a genuine kt LoRA-param deposit from any
/// other untagged entry. Setting bit 63 on every kt-leaf deposit keeps kt-param
/// deposits in a range disjoint from any bare id (both counters stay far below
/// `1 << 63` for any realistic process).
pub const KT_PARAM_DEPOSIT_TAG: u64 = 1u64 << 63;

/// Decode a deposit-map key back into a kt LoRA-param `TensorId` raw, but ONLY
/// when the key carries [`KT_PARAM_DEPOSIT_TAG`] (i.e. it was stored by
/// [`register_input_mapping_kt`]). Returns `None` for any untagged entry.
///
/// This is the read side of the namespace fix: a producer iterating the
/// per-scope deposit map calls this on each key; `Some(param_raw)` means the
/// entry is a genuine kt LoRA-param grad keyed by `param.tensor_id().as_raw()`,
/// while `None` means an untagged entry that must be ignored. This makes the
/// producer's `param_raw_ids.contains(..)` match collision-proof: an id equal to
/// a param id is untagged, so it decodes to `None` and is skipped.
pub fn decode_kt_param_deposit(key_raw: u64) -> Option<u64> {
    if key_raw & KT_PARAM_DEPOSIT_TAG != 0 {
        Some(key_raw & !KT_PARAM_DEPOSIT_TAG)
    } else {
        None
    }
}

/// Register one input mapping pair where the differentiable leaf is itself a
/// kt tensor (e.g. a LoRA `Var` after the #1082 forward flip made
/// `LoraProjectionWeights` hold `kiln_tensor::Tensor`).
///
/// Records into the `kt_to_candle_input` deposit map with the stored "deposit"
/// id namespaced by [`KT_PARAM_DEPOSIT_TAG`] (bit 63) so a kt-param id can never
/// collide with any other id in the shared `usize`-keyed deposit map (see the
/// tag's docs for the collision incident). This variant stores `kt_leaf_id | TAG`.
/// Producers read it back via [`decode_kt_param_deposit`] (strips the tag,
/// returns the kt leaf id) and match it against `param.tensor_id().as_raw()`.
/// On a 64-bit target `usize == u64`, so the tagged `u64` round-trips through
/// the `usize`-keyed map losslessly.
pub fn register_input_mapping_kt(kt_id: KtTensorId, deposit_kt_id: KtTensorId) {
    BRIDGE_SCOPE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let Some(scope) = borrow.as_mut() else {
            return;
        };
        let kt_raw = kt_id.as_raw();
        // Namespace the kt-leaf deposit id so it cannot alias a candle id that
        // happens to share the same raw counter value (#1082 collision fix).
        let deposit_raw = (deposit_kt_id.as_raw() | KT_PARAM_DEPOSIT_TAG) as usize;
        let ids = scope.kt_to_candle_input.entry(kt_raw).or_default();
        if !ids.contains(&deposit_raw) {
            ids.push(deposit_raw);
        }
    });
}

/// Open ONLY the IO-mapping scope for the duration of `f`. The adapters'
/// `register_input_mapping_kt` calls become live, and the caller drives the
/// tape walk itself (the tape-authoritative path) via
/// [`with_tape_authoritative_scope_kt`] or [`with_tape_segment_backward_scope`].
/// Panics on a nested scope; the scope is cleared on return (including on
/// panic). (#1082 CP-4 endgame.)
pub fn with_io_mapping_scope<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            BRIDGE_SCOPE.with(|cell| {
                *cell.borrow_mut() = None;
            });
        }
    }
    BRIDGE_SCOPE.with(|cell| {
        let mut b = cell.borrow_mut();
        assert!(
            b.is_none(),
            "tape_bridge: nested bridge scopes are not supported. A scope is \
             already active on this thread."
        );
        *b = Some(IoMappingScope::default());
    });
    let _guard = Guard;
    f()
}

/// kt-loss variant of [`with_tape_authoritative_scope`] (#1082 DoD-100 step 14
/// keystone). Identical seeding + grad-extraction, but the `forward` closure
/// returns a **kt** scalar loss instead of a candle one — so there is NO candle
/// loss round-trip and NO `candle_output_kt` id-resolution: the kt loss IS the
/// recorded tape root, seeded directly. This is the seam that lets the
/// GRPO/OPD/cross_entropy adapters stop copying their kt scalar loss back to
/// candle purely so this scope could resolve it.
///
/// The returned grad map is identical in shape/semantics to
/// [`with_tape_authoritative_scope`]'s (keyed by the recorded input ids — the
/// LoRA-`Parameter` deposits are `KT_PARAM_DEPOSIT_TAG`-tagged via
/// `register_input_mapping_kt`, so callers decode them with
/// `decode_kt_param_deposit` exactly as today). The grad DEPOSIT was already
/// kt-native; only the LOSS crosses to kt here.
///
/// # Contract
///
/// `forward` returns `(T, kt loss)` where the kt loss MUST be a tape node
/// recorded in this scope (e.g. the kt cross_entropy / GRPO / OPD scalar-loss
/// adapter recorded it). The tape walk seeds `dL/dL = 1` at the loss node.
pub fn with_tape_authoritative_scope_kt<T, F>(
    forward: F,
) -> Result<(T, kiln_tensor::Tensor, HashMap<usize, kiln_tensor::Tensor>), BridgeError>
where
    F: FnOnce() -> Result<(T, kiln_tensor::Tensor), BridgeError>,
{
    with_io_mapping_scope(|| {
        let (forward_res, tape): (Result<(T, kiln_tensor::Tensor), BridgeError>, Tape) =
            with_thread_local_tape(forward);
        let (payload, loss_kt) = forward_res?;

        // The kt loss IS the tape root — seed dL/dL = 1 directly (no candle
        // round-trip, no `candle_output_kt` resolution).
        let seed = kiln_tensor::ops::ones_like(&loss_kt)
            .map_err(|e| BridgeError::new(format!("tape_bridge: ones_like(loss_kt): {e}")))?;
        let mut seeds: HashMap<KtTensorId, kiln_tensor::Tensor> = HashMap::new();
        seeds.insert(loss_kt.id(), seed);
        let kt_grads = tape
            .backward_with_seeds(seeds, |a, b| kiln_tensor::ops::add(a, b))
            .map_err(|e| {
                BridgeError::new(format!("tape_bridge: tape-authoritative(kt) backward walk: {e}"))
            })?;

        // Same grad-map build as the candle variant: deposit each recorded kt
        // input grad under every mapped key (the LoRA-Parameter deposits are
        // KT_PARAM_DEPOSIT_TAG-tagged via `register_input_mapping_kt`).
        let input_map: Vec<(u64, usize)> = BRIDGE_SCOPE.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|s| {
                    s.kt_to_candle_input
                        .iter()
                        .flat_map(|(k, vs)| vs.iter().map(move |v| (*k, *v)))
                        .collect()
                })
                .unwrap_or_default()
        });
        let mut out: HashMap<usize, kiln_tensor::Tensor> = HashMap::new();
        for (kt_in_raw, mapped_raw) in input_map {
            if let Some(g) = kt_grads.get(KtTensorId::from_raw(kt_in_raw)) {
                out.insert(mapped_raw, g.clone());
            }
        }
        Ok((payload, loss_kt, out))
    })
}

/// Per-segment tape-authoritative backward for gradient checkpointing (#1082).
///
/// Like [`with_tape_authoritative_scope`], but seeds the backward at an
/// arbitrary segment OUTPUT with an externally-supplied upstream gradient
/// (instead of looking up a loss adapter output and seeding `dL/dL = 1`). This
/// is the kt-tape replacement for the legacy candle gradient-checkpointing
/// reverse, which was grad-severed by the flip (candle `.backward()` cannot
/// trace through the now-kt-internal `model_forward_segment`).
///
/// # Contract
///
/// `forward` runs ONE checkpoint segment under a fresh thread-local tape (so
/// only that segment's activations are recorded — memory stays bounded to a
/// single segment) and must return the segment's kt OUTPUT tensor. The tape is
/// then seeded with `{ seg_output.id() : upstream_grad }` and walked.
///
/// Returns `(kt_grads, candle_input_grads)`:
/// * `kt_grads` — the full [`kiln_autograd::GradStore`] (keyed by `KtTensorId`).
///   The caller reads the segment-INPUT grad (`kt_grads.get(seg_input.id())`)
///   to chain into the previous segment.
/// * `candle_input_grads` — `candle_input_id_raw -> kt grad`, the same
///   candle-id-keyed map [`with_tape_authoritative_scope`] returns, so the
///   caller can pick out the LoRA `Var` grads for this segment by candle id.
///
/// # Errors
/// * `forward()` errors (propagated).
/// * Tape walk errors.
pub fn with_tape_segment_backward_scope<F>(
    upstream_grad: kiln_tensor::Tensor,
    forward: F,
) -> Result<(kiln_autograd::GradStore, HashMap<usize, kiln_tensor::Tensor>), BridgeError>
where
    F: FnOnce() -> Result<kiln_tensor::Tensor, BridgeError>,
{
    with_io_mapping_scope(|| {
        let (forward_res, tape): (Result<kiln_tensor::Tensor, BridgeError>, Tape) =
            with_thread_local_tape(forward);
        let seg_output = forward_res?;

        // Seed the upstream grad at the segment output id and walk the tape.
        // The seed dtype is matched to the segment output dtype by the caller.
        let mut seeds: HashMap<KtTensorId, kiln_tensor::Tensor> = HashMap::new();
        seeds.insert(seg_output.id(), upstream_grad);
        let kt_grads = tape
            .backward_with_seeds(seeds, |a, b| kiln_tensor::ops::add(a, b))
            .map_err(|e| {
                BridgeError::new(format!(
                    "tape_bridge: with_tape_segment_backward_scope backward walk: {e}"
                ))
            })?;

        // Map each recorded kt input grad to its candle input id(s) (a kt input
        // may fan out to several candle ids across islands — deposit each).
        let input_map: Vec<(u64, usize)> = BRIDGE_SCOPE.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|s| {
                    s.kt_to_candle_input
                        .iter()
                        .flat_map(|(k, vs)| vs.iter().map(move |v| (*k, *v)))
                        .collect()
                })
                .unwrap_or_default()
        });
        let mut candle_input_grads: HashMap<usize, kiln_tensor::Tensor> = HashMap::new();
        for (kt_in_raw, candle_in_raw) in input_map {
            if let Some(g) = kt_grads.get(KtTensorId::from_raw(kt_in_raw)) {
                candle_input_grads.insert(candle_in_raw, g.clone());
            }
        }
        Ok((kt_grads, candle_input_grads))
    })
}

/// True iff a bridge scope is active on the current thread.
///
/// Adapters can use this to know whether registering is worth the
/// HashMap insert (e.g. skip the kt-side `id()` call entirely when
/// the bridge is off). At today's call rates the bridge-off cost is
/// negligible (a single `RefCell::borrow` + early return inside the
/// register call), so most adapters skip this check.
pub fn bridge_scope_active() -> bool {
    BRIDGE_SCOPE.with(|cell| cell.borrow().is_some())
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_scope_inactive_by_default() {
        assert!(
            !bridge_scope_active(),
            "no scope must be active at test entry"
        );
    }

    /// #1082 collision-fix invariant: a kt-leaf deposit id round-trips through
    /// the namespace tag, while a bare candle id (no tag) decodes to `None` —
    /// even when its raw value equals a kt-param id. This is what prevents a
    /// frozen tensor's candle-keyed grad from aliasing a LoRA param's slot in
    /// the shared `usize`-keyed deposit map (the `[32] != [32, 4]` AdamW shape
    /// mismatch).
    #[test]
    fn kt_param_deposit_tag_roundtrips_and_rejects_candle_ids() {
        // A kt-param deposit raw round-trips: encode (OR tag) then decode.
        for raw in [1u64, 2, 32, 12_345, (1u64 << 40) - 1] {
            let encoded = raw | KT_PARAM_DEPOSIT_TAG;
            assert_eq!(
                decode_kt_param_deposit(encoded),
                Some(raw),
                "tagged kt-param deposit {encoded:#x} must decode back to {raw}"
            );
        }
        // A bare candle id (no tag) is NOT a kt-param deposit — even if its raw
        // value collides with a kt-param id like 32. Decoding rejects it, so the
        // producer's `param_raw_ids.contains(..)` never sees it.
        for candle_raw in [1u64, 2, 32, 12_345] {
            assert_eq!(
                decode_kt_param_deposit(candle_raw),
                None,
                "untagged candle id {candle_raw} must NOT decode as a kt-param deposit"
            );
        }
        // The tag occupies bit 63 only — decoding strips exactly that bit.
        assert_eq!(KT_PARAM_DEPOSIT_TAG, 1u64 << 63);
    }

    // The full end-to-end bridge tests live in
    // `kiln-kt-bridge/tests/tape_bridge_e2e.rs` (gated on
    // `feature = "cuda"`). They require a live CUDA device and
    // exercise the bridge through the production `try_tape_*_cuda`
    // adapters.
}
