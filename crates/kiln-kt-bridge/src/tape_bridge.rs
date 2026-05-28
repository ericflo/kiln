//! `Tape` → candle `GradStore` bridge — CP-4 backward-integration (#1082).
//!
//! # The disjoint-walker problem
//!
//! The STOP doc at
//! [`docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`]
//! documented that `kiln_autograd::Tape::backward` and
//! `candle_core::Tensor::backward()` walk **two disjoint graphs**:
//!
//! * `Tape` is keyed on `kiln_tensor::TensorId` (u64), records
//!   `BoxedBackwardOp` nodes, and walks them in reverse insertion
//!   order to produce a `kiln_autograd::GradStore`.
//! * candle's autograd is keyed on `candle_core::TensorId` (usize),
//!   records `BackpropOp` nodes inside the `Tensor` itself, and
//!   walks them via `loss.backward()` to produce a
//!   `candle_core::backprop::GradStore`.
//!
//! Wave-12/13 (#1082) landed the thread-local `Tape` substrate plus
//! the three `try_tape_{rms_norm,matmul,silu}_cuda` adapters that
//! record onto the active tape and return a *fresh* candle Tensor
//! (built via `kt_tensor_to_candle_cuda_copy`). Those fresh candle
//! Tensors have no backprop lineage in candle's view — they are
//! "leaf" inputs to whatever candle ops consume them downstream.
//!
//! That's the key insight that makes this bridge possible:
//!
//! 1. After the bridge wrapper's forward returns, the candle graph
//!    has the candle-typed outputs of each tape adapter as **leaf
//!    inputs** to downstream candle ops. The loss flows back through
//!    candle's `BackpropOp` graph to those leaves and `loss.backward()`
//!    produces a candle `GradStore` containing `dL/d(tape_output)` for
//!    each tape-adapter return value.
//!
//! 2. The bridge tracks `(candle_id ⟷ kt_id)` pairs at adapter
//!    boundaries (set up by `try_tape_*_cuda` while recording on the
//!    tape). For each tape-output candle TensorId, it pulls the
//!    candle grad from `loss.backward()`'s `GradStore`, converts it
//!    to a kt-typed grad, and seeds `Tape::backward_with_seeds`.
//!
//! 3. The kt-side walker produces per-input kt grads. The bridge
//!    converts each one back to a candle Tensor and inserts it into
//!    the candle `GradStore` under the matched candle input TensorId.
//!
//! End-to-end: a parameter that flows through one or more tape
//! adapters now has its `dL/dparam` populated in the same candle
//! `GradStore` that `loss.backward()` returns. The optimizer step
//! (which iterates the candle `GradStore` keyed on each param's
//! candle TensorId) sees the bridged grad transparently.
//!
//! # The bridge's three pieces
//!
//! 1. [`TapeBridgeScope`] — a thread-local RAII guard that opens
//!    *both* a `kiln_autograd::tape_scope::with_thread_local_tape`
//!    scope **and** a `(candle_id ⟷ kt_id)` mapping scope on entry.
//!    Recording adapters register pairs via [`register_input_mapping`]
//!    / [`register_output_mapping`] during forward.
//!
//! 2. [`with_tape_scope_emit_to_grad_store`] — the user-facing
//!    wrapper. Runs `forward()` inside a `TapeBridgeScope`, asks
//!    `forward()` to produce the loss tensor, calls
//!    `loss.backward()` on candle's side, walks the tape with
//!    candle-derived seeds, and merges the resulting per-input kt
//!    grads back into the candle `GradStore`.
//!
//! 3. [`register_input_mapping`] / [`register_output_mapping`] —
//!    called by the `try_tape_*_cuda` adapters as they record. Stores
//!    `(kt_input_id → candle_input_id)` pairs and
//!    `(kt_output_id → candle_output_id)` pairs in the active bridge
//!    scope's thread-locals.
//!
//! # Production-safety
//!
//! Off by default — the `TapeBridgeScope` is only opened when the
//! caller explicitly wraps `forward()` in
//! `with_tape_scope_emit_to_grad_store`. Outside that scope,
//! `register_input_mapping` is a no-op and the tape adapters work
//! exactly as today (record onto the tape, returned candle tensor
//! has no candle backprop lineage). The existing
//! `kt_forward_op`-shim path stays the production code path until
//! a future PR opts in to the bridge at the training-step root.
//!
//! # Out of scope for this PR
//!
//! * Wiring `kiln-train::trainer` callers onto the bridge. That's a
//!   downstream consumer PR — once the bridge proves it works
//!   end-to-end (forward + loss + bridged backward = candle grads
//!   that match a candle-only backward to within fp tolerance), the
//!   trainer's `loss.backward()` calls can opt in one at a time.
//! * Non-CUDA tape adapters. The current
//!   `try_tape_{rms_norm,matmul,silu}_cuda` triplet covers BF16
//!   CUDA only; the bridge interface is generic but the
//!   `kt_tensor_to_candle_cuda_copy` / `kt_tensor_from_candle_cuda_copy`
//!   helpers it depends on are CUDA-only today.

#![cfg(feature = "cuda")]

use std::cell::RefCell;
use std::collections::HashMap;

use candle_core::backprop::GradStore as CandleGradStore;
use candle_core::Tensor as CandleTensor;
use candle_core::TensorId as CandleTensorId;
use kiln_autograd::{
    with_active_tape, with_thread_local_tape, InjectGradientBackward, Tape,
};
use kiln_tensor::TensorId as KtTensorId;

use crate::BridgeError;

/// One side of the bridge's per-thread mapping state.
///
/// Tracks the candle⟷kt TensorId pairs registered by tape adapters
/// during forward, so the bridge's backward walker can:
///
/// * Convert candle output-seed grads → kt seed grads keyed on the
///   recorded `kt_output_id`.
/// * Convert kt input grads → candle grads keyed on the recorded
///   `candle_input_id`.
#[derive(Debug, Default)]
struct IoMappingScope {
    /// `kt_input_id → candle_input_id` for inputs registered by
    /// adapters. Used by the bridge backward to know which candle
    /// `TensorId` to insert each kt-side input gradient under in the
    /// candle `GradStore`.
    ///
    /// Multiple adapters may register the same `kt_input_id` (e.g.
    /// the same `kt_x_id` borrowed twice from the same candle source
    /// `x`); the mapping value must always match for the same key.
    /// We assert this on insert so a wiring bug surfaces at record
    /// time, not silently at emit time.
    kt_to_candle_input: HashMap<u64, usize>,

    /// `kt_output_id → candle_output_id` for outputs returned by the
    /// adapters. The bridge backward pulls each candle output's grad
    /// from the candle `GradStore` and seeds the tape with the
    /// corresponding `kt_output_id` entry.
    kt_to_candle_output: HashMap<u64, usize>,
}

thread_local! {
    /// Active bridge scope on this thread. `Some` only inside a
    /// `with_tape_scope_emit_to_grad_store` block. Outside the scope,
    /// `register_input_mapping` is a no-op.
    static BRIDGE_SCOPE: RefCell<Option<IoMappingScope>> = const { RefCell::new(None) };
}

/// Register one input mapping pair.
///
/// Called from `try_tape_*_cuda` adapters as they record onto the
/// active tape: for each candle `Tensor` input that gets borrowed
/// (or copied) into a kt Tensor, the adapter pairs the kt side's
/// TensorId with the candle side's TensorId so the bridge backward
/// knows where to deposit the eventual gradient.
///
/// No-ops cleanly when no bridge scope is active — callers can
/// register unconditionally and the cost is one `RefCell::borrow_mut`
/// + early return when the scope is off.
pub fn register_input_mapping(kt_id: KtTensorId, candle_id: CandleTensorId) {
    BRIDGE_SCOPE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let Some(scope) = borrow.as_mut() else {
            return;
        };
        let kt_raw = kt_id.as_raw();
        let candle_raw = candle_id.as_raw();
        if let Some(prev) = scope.kt_to_candle_input.insert(kt_raw, candle_raw) {
            // Same kt input recorded twice. Allowed iff the candle
            // ID matches — that's the "single candle tensor borrowed
            // by two adapters" case. Different candle IDs would mean
            // the kt-bridge's TensorId allocator handed out the same
            // u64 for two distinct kt borrows of two distinct candle
            // sources, which would be a wiring bug.
            assert_eq!(
                prev, candle_raw,
                "tape_bridge: kt input id {kt_raw} already mapped to candle id \
                 {prev}; new candle id {candle_raw} disagrees — this is an \
                 adapter-side wiring bug. The same kt TensorId must not pair \
                 with two distinct candle TensorIds in one bridge scope."
            );
        }
    });
}

/// Register one output mapping pair.
///
/// Called from `try_tape_*_cuda` adapters as they finish recording
/// and prepare to return a fresh candle Tensor copied from the kt
/// output. The kt-side output's TensorId is what the tape recorded
/// the `output_id` field under; the candle-side TensorId is the new
/// candle Tensor's `.id()` that the adapter is about to return to
/// the caller (and which downstream candle ops will reference).
pub fn register_output_mapping(kt_id: KtTensorId, candle_id: CandleTensorId) {
    BRIDGE_SCOPE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let Some(scope) = borrow.as_mut() else {
            return;
        };
        let kt_raw = kt_id.as_raw();
        let candle_raw = candle_id.as_raw();
        // Outputs are unique per adapter call: the kt-bridge
        // allocates a fresh `KtTensorId::next()` and the candle
        // Tensor's `.id()` is freshly minted by `Tensor::zeros`. A
        // duplicate key here would mean two adapter calls each
        // recorded the same kt output id, which is impossible by
        // construction. Panic to make it loud if it ever happens.
        let prev = scope.kt_to_candle_output.insert(kt_raw, candle_raw);
        assert!(
            prev.is_none(),
            "tape_bridge: duplicate kt output id {kt_raw} registered with \
             candle id {candle_raw} (previous was {prev:?}). Two adapters \
             cannot share an output id."
        );
    });
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

/// User-facing wrapper: run `forward()` inside a tape scope *and* a
/// bridge mapping scope, then merge tape-derived gradients into the
/// candle `GradStore` produced by `loss.backward()`.
///
/// # Contract
///
/// `forward` is given no arguments and must return:
/// * The candle `Tensor` loss (or any candle tensor whose
///   `.backward()` should drive the candle side of the graph). The
///   bridge calls `.backward()` on it.
/// * Any application-specific payload `T` the caller wants to keep
///   (e.g. the loss value, model outputs, etc.). Returned to the
///   caller as the first tuple element of the bridge's result.
///
/// The bridge:
///
/// 1. Opens a thread-local `Tape` scope and a `BRIDGE_SCOPE` mapping
///    scope.
/// 2. Runs `forward()`. While it runs, any `try_tape_*_cuda` adapter
///    that successfully records onto the active tape also registers
///    its IO mappings via `register_{input,output}_mapping`.
/// 3. Calls `loss.backward()` on the returned candle loss.
/// 4. Walks the tape with seeds taken from the resulting candle
///    `GradStore` (one seed per tape-output that has a candle grad).
/// 5. Merges the per-kt-input grads back into the candle `GradStore`
///    keyed on the matched candle input TensorIds.
/// 6. Closes both scopes (RAII via the inner functions).
///
/// Returns `(forward_payload, candle_grad_store)`. The candle
/// `GradStore` now contains:
///
/// * The grads candle itself produced for every candle-typed Tensor
///   that was tracked through the candle graph (unchanged from a
///   plain `loss.backward()` call).
/// * **Plus** the grads tape adapters produced for every candle
///   input TensorId registered via `register_input_mapping`. If a
///   candle input ID already had a candle-side grad (because the
///   same Tensor flowed through both tape and candle paths), the
///   tape grad is **added** to it via candle's `Tensor::add`.
///
/// # Errors
///
/// Returns a `BridgeError` if:
/// * `forward()` errors. We propagate verbatim.
/// * `loss.backward()` errors.
/// * Any tape output's candle grad is missing from the
///   `GradStore` (means the candle graph never consumed the tape
///   output — likely a wiring bug in the caller's forward, not the
///   bridge).
/// * Tape walk errors (e.g. anti-pattern 16 version drift).
/// * kt → candle grad copy fails.
pub fn with_tape_scope_emit_to_grad_store<T, F>(
    forward: F,
) -> Result<(T, CandleGradStore), BridgeError>
where
    F: FnOnce() -> Result<(T, CandleTensor), BridgeError>,
{
    // Open the IO mapping scope first. We pair it with the tape
    // scope (opened inside the closure via `with_thread_local_tape`)
    // because adapters need both: the tape scope tells them "record
    // onto this tape" and the mapping scope tells them "stash this
    // (kt, candle) pair for the bridge".
    BRIDGE_SCOPE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_some() {
            panic!(
                "tape_bridge: nested bridge scopes are not supported. \
                 A `with_tape_scope_emit_to_grad_store` is already active \
                 on this thread."
            );
        }
        *borrow = Some(IoMappingScope::default());
    });

    // Always close the bridge scope on the way out, even on the
    // error paths. We pull the scope's contents out at the end of
    // the success path; on errors we just drop it.
    struct ScopeGuard;
    impl Drop for ScopeGuard {
        fn drop(&mut self) {
            BRIDGE_SCOPE.with(|cell| {
                let _ = cell.borrow_mut().take();
            });
        }
    }
    let _guard = ScopeGuard;

    // Run `forward()` inside the tape scope. `with_thread_local_tape`
    // panics on nested scopes (same contract as our bridge scope), so
    // a nested bridge call panics either way.
    let (forward_result, tape): (Result<(T, CandleTensor), BridgeError>, Tape) =
        with_thread_local_tape(forward);

    let (payload, loss) = forward_result?;

    // Candle-side backward. Produces a candle `GradStore` containing
    // grads for every candle-tracked tensor whose lineage flows into
    // `loss`. The tape adapters' fresh candle outputs DO appear as
    // grads in this store because they're inputs to downstream
    // candle ops that flow into the loss.
    let mut candle_grad_store: CandleGradStore = loss
        .backward()
        .map_err(|e| BridgeError::new(format!("tape_bridge: loss.backward(): {e}")))?;

    // Pull the IO mapping scope's contents out for the backward walk.
    let scope = BRIDGE_SCOPE
        .with(|cell| cell.borrow_mut().as_mut().map(std::mem::take))
        .expect("tape_bridge: scope guard dropped before backward emit");

    // Fast-path: no adapters recorded → no bridge work to do.
    // (Empty tape means no IO mappings either; assert in debug.)
    if tape.is_empty() {
        debug_assert!(
            scope.kt_to_candle_input.is_empty() && scope.kt_to_candle_output.is_empty(),
            "tape_bridge: empty tape but non-empty mapping scope — \
             an adapter registered a mapping without recording onto the \
             tape, which is a wiring bug."
        );
        return Ok((payload, candle_grad_store));
    }

    // For each tape-output we registered, fetch its candle-side
    // gradient from `candle_grad_store` and prepare a kt-side seed
    // for the tape walker.
    let mut kt_seeds: HashMap<KtTensorId, kiln_tensor::Tensor> = HashMap::new();
    for (kt_out_raw, candle_out_raw) in &scope.kt_to_candle_output {
        // Reconstruct the candle TensorId. `candle_core::TensorId`
        // doesn't expose a `from_raw(usize)`, but it does provide
        // `get_id` on `GradStore`. We use the raw `usize` payload
        // we recorded at adapter time.
        let candle_grad = lookup_candle_grad_by_raw(&candle_grad_store, *candle_out_raw)
            .ok_or_else(|| {
                BridgeError::new(format!(
                    "tape_bridge: candle GradStore missing grad for tape-output \
                     candle id {candle_out_raw} (kt id {kt_out_raw}). The candle \
                     graph never propagated a gradient to this leaf — likely the \
                     tape-output candle Tensor was not consumed by any candle op \
                     flowing into the loss."
                ))
            })?;

        // Copy the candle grad to a kt-typed tensor on the same
        // device. The borrow adapter is zero-copy + contiguous-only;
        // candle grads may be non-contiguous (broadcast-reductions),
        // so use the copy adapter for safety. (`.contiguous()` is a
        // no-op if already contig.)
        let candle_grad_c = candle_grad
            .contiguous()
            .map_err(|e| BridgeError::new(format!("tape_bridge: contiguous candle grad: {e}")))?;
        let kt_grad = crate::kt_tensor_from_candle_cuda_copy(&candle_grad_c).map_err(|e| {
            BridgeError::new(format!(
                "tape_bridge: candle → kt grad copy failed for tape output \
                 (kt id {kt_out_raw}, candle id {candle_out_raw}): {e}"
            ))
        })?;
        kt_seeds.insert(KtTensorId::from_raw(*kt_out_raw), kt_grad);
    }

    // Walk the tape with the seed map. The walker produces kt grads
    // for every kt TensorId that flowed through a recorded backward
    // op (typically: all the kt input ids registered + intermediate
    // node outputs).
    let kt_grad_store = tape
        .backward_with_seeds(kt_seeds, |a, b| kiln_tensor::ops::add(a, b))
        .map_err(|e| BridgeError::new(format!("tape_bridge: Tape::backward_with_seeds: {e}")))?;

    // For each kt input we registered, look up its grad and copy it
    // back into the candle GradStore under the matched candle id.
    for (kt_in_raw, candle_in_raw) in &scope.kt_to_candle_input {
        let kt_input_id = KtTensorId::from_raw(*kt_in_raw);
        let Some(kt_grad) = kt_grad_store.get(kt_input_id) else {
            // No grad for this kt input. Common case: the same kt
            // TensorId was registered as both an input to one
            // adapter and an output of another (chained adapters);
            // the walker may have consumed the seed without
            // producing a leaf grad. The candle side handles its
            // own; we just skip.
            continue;
        };

        let candle_grad = crate::kt_tensor_to_candle_cuda_copy(kt_grad).map_err(|e| {
            BridgeError::new(format!(
                "tape_bridge: kt → candle grad copy failed for kt input \
                 (kt id {kt_in_raw}, candle id {candle_in_raw}): {e}"
            ))
        })?;

        // Merge into the candle GradStore. If the candle side
        // already has a grad for this id (the same Tensor flowed
        // through both tape and candle paths), accumulate; else,
        // skip with a soft warning (no `TensorId::from_raw` on
        // candle's side — see `insert_or_add_by_raw` doc).
        insert_or_add_by_raw(&mut candle_grad_store, *candle_in_raw, candle_grad)?;
    }

    Ok((payload, candle_grad_store))
}

/// Look up a candle grad in the `GradStore` by the raw `usize` id we
/// recorded earlier. Candle's `TensorId(usize)` has private `new()`
/// but `as_raw() -> usize` is public; we iterate `get_ids()` to find
/// the matching id and then call `get_id`.
///
/// (`GradStore::get_id` takes `TensorId`; we can't reconstruct one
/// directly. Iterating is fine — the store is small.)
fn lookup_candle_grad_by_raw(
    store: &CandleGradStore,
    target_raw: usize,
) -> Option<&CandleTensor> {
    let id = store
        .get_ids()
        .find(|id| id.as_raw() == target_raw)
        .copied()?;
    store.get_id(id)
}

/// Insert-or-accumulate a candle grad into the `GradStore` under a
/// raw `usize` id. Iterates `get_ids()` to find the matching id; if
/// found, replaces with `existing + grad`; else errors.
///
/// **Limitation**: candle's `TensorId(usize)` has no `from_raw`
/// constructor. We can `insert_id(id, grad)` only when we already
/// have a `TensorId` value in hand. Iterating `get_ids()` gives us
/// that value when the entry exists. When it doesn't exist, we
/// cannot synthesize one. This is fine for the bridge's primary use
/// case: the candle input tensor was tracked by candle's graph (i.e.
/// `track_op() == true`, typically via wrapping in a `Var`), so its
/// TensorId is already in the store after `loss.backward()`. If the
/// candle backward never visited the input, we return an error so
/// the wiring bug surfaces rather than dropping the gradient
/// silently.
fn insert_or_add_by_raw(
    store: &mut CandleGradStore,
    target_raw: usize,
    grad: CandleTensor,
) -> Result<(), BridgeError> {
    let existing_id = store
        .get_ids()
        .find(|id| id.as_raw() == target_raw)
        .copied();
    if let Some(id) = existing_id {
        // Add to the existing grad.
        let existing = store.get_id(id).expect("id found above must still exist");
        let summed = existing.add(&grad).map_err(|e| {
            BridgeError::new(format!(
                "tape_bridge: candle Tensor::add for accumulating tape grad \
                 onto candle grad (raw id {target_raw}): {e}"
            ))
        })?;
        store.insert_id(id, summed);
        Ok(())
    } else {
        // No matching id in the candle store. The candle backward
        // walk never touched this leaf — usually because the input
        // was a candle Tensor with `.track_op() == false` (the
        // production caller paths explicitly disable tracking on
        // weight tensors to avoid building a candle graph for the
        // kt-replaced ops).
        //
        // Without a `TensorId::from_raw(usize)` constructor on
        // candle's side, we cannot synthesise a key. The kt grad is
        // real and correct, but the candle GradStore cannot store
        // it. Return an error so the caller knows to wrap the
        // input in a candle `Var` (which sets track_op = true)
        // before passing it to the tape adapter.
        Err(BridgeError::new(format!(
            "tape_bridge: candle GradStore has no entry for raw id {target_raw}; \
             cannot insert tape-derived grad without a `TensorId::from_raw` \
             accessor on candle's side. Common cause: the candle input tensor \
             has `track_op() == false` so the candle backward walk never \
             allocated a slot for it. Workaround: wrap the input in a candle \
             `Var` (which sets track_op = true) before passing it to the tape \
             adapter; the bridge will then find an existing slot and \
             accumulate into it."
        )))
    }
}

/// Candle `CustomOp1` used by [`inject_gradient_kt`] to inject a
/// precomputed gradient into candle's backward walk for `arg`. Forward
/// returns a scalar F32 zero (same placeholder shape as the trainer's
/// historical `InjectTensorGradient`). Backward returns the held
/// `upstream` tensor (with `to_device + to_dtype` matched to `arg`),
/// exactly mirroring the contract of the in-trainer
/// `InjectTensorGradient::bwd`.
///
/// # Why this is Option-2 substrate
///
/// The previous substrate variant (commit `9b2eda8e`) returned
/// `zeros_like(arg)` from `bwd` and relied on the bridge's post-hoc
/// `insert_or_add_by_raw` to overwrite `grads[arg.id()]` with the
/// upstream value AFTER `loss.backward()` returned. That works when
/// `arg` IS the queried `Var` (the parity test's `arg_var.as_tensor()`
/// case) but produces wrong upstream-`Var` grads when `arg` is an
/// intermediate of further candle ops: candle's backward walk consumes
/// `grads[arg.id()]` DURING the walk, before the post-hoc update fires,
/// so all upstream `Var`s end up with grads derived from zeros.
///
/// Option 2 closes that gap by returning the upstream tensor directly
/// from `bwd`. The kt-tape recording (see [`inject_gradient_kt`]) is
/// preserved as a side channel for migration tracking / future
/// `kiln-kt-bridge`-only execution but no longer drives the GradStore
/// population — candle's own walk now produces correct grads for any
/// downstream `Var`.
///
/// See [`docs/inject-grad-flip-blocked-2026-05-28.md`] for the original
/// substrate-design diagnosis and the Option-1 vs Option-2 tradeoff.
#[derive(Clone)]
struct InjectGradientCandleShim {
    /// The precomputed gradient to emit as `arg`'s grad during
    /// candle's backward walk. Lives here so `bwd` doesn't have to
    /// reach into a thread-local / lookup table.
    upstream: candle_core::Tensor,
}

impl std::fmt::Debug for InjectGradientCandleShim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InjectGradientCandleShim")
            .field("upstream_dtype", &self.upstream.dtype())
            .field("upstream_dims", &self.upstream.dims())
            .finish()
    }
}

impl candle_core::CustomOp1 for InjectGradientCandleShim {
    fn name(&self) -> &'static str {
        "kiln-inject-gradient-candle-shim"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Ok((
            candle_core::CpuStorage::F32(vec![0.0]),
            candle_core::Shape::from(()),
        ))
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        let device = storage.device();
        let out_slice = device.clone_htod(&[0.0f32])?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out_slice, device.clone()),
            candle_core::Shape::from(()),
        ))
    }

    fn bwd(
        &self,
        arg: &candle_core::Tensor,
        _res: &candle_core::Tensor,
        _grad_res: &candle_core::Tensor,
    ) -> candle_core::Result<Option<candle_core::Tensor>> {
        // Mirror `InjectTensorGradient::bwd` byte-for-byte. Shape guard
        // first (cross-checked at record time by
        // `InjectGradientBackward::new_validated`, but we keep it here
        // too so the candle path surfaces a typed error if a future
        // refactor hands us a mismatched arg).
        if self.upstream.dims() != arg.dims() {
            candle_core::bail!(
                "InjectGradientCandleShim shape mismatch: upstream {:?}, arg {:?}",
                self.upstream.dims(),
                arg.dims()
            );
        }
        let upstream = self.upstream.to_device(arg.device())?;
        let grad = if upstream.dtype() == arg.dtype() {
            upstream
        } else {
            upstream.to_dtype(arg.dtype())?
        };
        Ok(Some(grad))
    }
}

/// Adapter for the kt-tape replacement of
/// `kiln-train::trainer::InjectTensorGradient` (#1082, CP-4).
///
/// # What this does
///
/// The candle path (still live in `trainer.rs`):
///
/// ```ignore
/// let injected = arg.apply_op1(InjectTensorGradient { upstream })?;
/// let grads = injected.backward()?;
/// // grads.get(arg) == upstream (after to_device + to_dtype)
/// ```
///
/// This function builds the kt-tape equivalent:
///
/// 1. Borrows the candle `arg` zero-copy as a kt tensor.
/// 2. Copies the candle `upstream` to a kt tensor on `arg`'s device,
///    promoting/demoting dtype to match `arg` exactly as candle's
///    `InjectTensorGradient::bwd` does via `to_device` + `to_dtype`.
/// 3. Records an [`InjectGradientBackward`] node on the active tape
///    with the borrowed `arg_kt` as its sole input. The recorded node's
///    output id is a fresh `KtTensorId`.
/// 4. Records the kt-side [`InjectGradientBackward`] on the active
///    tape **if one is open** — this is a side channel for
///    migration tracking and future kiln-kt-bridge-only execution.
///    Under Option 2 the tape recording does NOT drive GradStore
///    population (the candle CustomOp1 below does); IO mappings are
///    intentionally NOT registered for this adapter to avoid the
///    upstream-doubled `2*upstream` bug.
/// 5. Returns a candle Tensor produced by `arg.apply_op1(shim)` where
///    the shim's `bwd` returns the precomputed `upstream` (matched
///    to `arg.device()` + `arg.dtype()`), exactly mirroring the
///    trainer's historical `InjectTensorGradient::bwd` contract.
///    Callers run `injected.backward()` on it; `grads[arg.id()]`
///    receives `upstream` and candle propagates it through any
///    upstream ops above `arg`.
///
/// # Errors
///
/// * `arg` is not on CUDA, not contiguous, or has a dtype that doesn't
///   round-trip through `candle_dtype_to_kt`.
/// * `upstream`'s shape doesn't match `arg`'s shape.
/// * The kt allocations or memcpys fail.
///
/// Note: an inactive tape is **not** an error under Option 2 — the
/// candle CustomOp1 path produces the correct gradient on its own.
/// The kt-tape recording is a no-op when no scope is active.
///
/// # Production-ready
///
/// This adapter is bit-equivalent to the historical
/// `InjectTensorGradient::apply_op1` for any `arg` (Var or
/// intermediate of further candle ops) under Option 2 substrate
/// (commit `e2f8723c`). The trainer.rs `InjectTensorGradient` call
/// sites can flip onto this adapter; the historical
/// `InjectTensorGradient` struct + `impl candle_core::CustomOp1`
/// becomes dead code afterward.
pub fn inject_gradient_kt(
    arg: &CandleTensor,
    upstream: &CandleTensor,
) -> Result<CandleTensor, BridgeError> {
    use crate::{
        candle_cuda_device_with_stream_no_event_tracking, kt_tensor_from_candle_cuda_borrow,
        kt_tensor_from_candle_cuda_copy,
    };

    if arg.dims() != upstream.dims() {
        return Err(BridgeError::new(format!(
            "tape_bridge::inject_gradient_kt: arg shape {:?} != upstream shape {:?}",
            arg.dims(),
            upstream.dims()
        )));
    }

    // Borrow arg zero-copy as kt. This validates contiguity + CUDA +
    // dtype roundtrip. Any layout/dtype/device mismatch surfaces as
    // a typed BridgeError instead of degrading silently.
    let arg_kt = kt_tensor_from_candle_cuda_borrow(arg)?;

    // Convert upstream to a kt-typed grad matching arg's device and
    // dtype. The candle CustomOp1's `bwd` does:
    //   let upstream = self.upstream.to_device(arg.device())?;
    //   if upstream.dtype() != arg.dtype() { upstream.to_dtype(arg.dtype())? }
    // We mirror that on the candle side first (so the dtype/device
    // adjustments compose cleanly with candle's existing converters),
    // then copy into a fresh kt tensor.
    let candle_arg_device = arg.device().clone();
    let candle_arg_dtype = arg.dtype();
    let upstream_on_dev = upstream
        .to_device(&candle_arg_device)
        .map_err(|e| BridgeError::new(format!(
            "tape_bridge::inject_gradient_kt: upstream.to_device(arg.device()): {e}"
        )))?;
    let upstream_typed = if upstream_on_dev.dtype() == candle_arg_dtype {
        upstream_on_dev
    } else {
        upstream_on_dev.to_dtype(candle_arg_dtype).map_err(|e| {
            BridgeError::new(format!(
                "tape_bridge::inject_gradient_kt: upstream.to_dtype(arg.dtype()): {e}"
            ))
        })?
    };
    // contiguous() is a no-op on already-contig tensors; it materializes
    // any narrow/transpose layout the caller might pass in (the q/gate
    // tiled paths slice upstream out of a `narrow(1, tile_start, ...)`,
    // which is non-trailing-axis and therefore not necessarily contig).
    let upstream_typed_c = upstream_typed.contiguous().map_err(|e| {
        BridgeError::new(format!(
            "tape_bridge::inject_gradient_kt: upstream.contiguous(): {e}"
        ))
    })?;
    let injected_kt = kt_tensor_from_candle_cuda_copy(&upstream_typed_c)?;

    // Build the BackwardOp. `new_validated` cross-checks shape + dtype
    // between arg_kt and injected_kt — same contract as the candle
    // CustomOp1's bwd shape guard, but enforced at record time.
    let backward_op = InjectGradientBackward::new_validated(&arg_kt, injected_kt)
        .map_err(|e| BridgeError::new(format!(
            "tape_bridge::inject_gradient_kt: InjectGradientBackward::new_validated: {e}"
        )))?;

    // Allocate the kt-side placeholder output: scalar zero F32 on the
    // same device as arg. Matches the candle CustomOp1's
    // `cuda_fwd`/`cpu_fwd` shape (Shape::from(())) and value (0.0f32).
    // Mirror the kt-side allocator used by the rmsnorm/matmul/silu
    // tape adapters so the kt-side output tensor has a fresh
    // `KtTensorId` and lives on the same CUDA device as arg_kt.
    let arg_kt_storage = arg_kt
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CudaStorage>()
        .ok_or_else(|| {
            BridgeError::new("tape_bridge::inject_gradient_kt: arg_kt storage must be Cuda")
        })?;
    let out_kt =
        crate::alloc_cuda_tensor(arg_kt_storage, kiln_tensor::DType::F32, vec![])?;

    // Record the backward on the active tape, if one is open. Under
    // Option 2 the kt-tape recording is a **side channel** for
    // migration tracking + future kiln-kt-bridge-only execution; the
    // candle CustomOp1 below produces the GradStore population on its
    // own. No active tape is no longer an error — the call is still
    // valid (just side-channel-free).
    //
    // We deliberately DO NOT call `register_input_mapping` /
    // `register_output_mapping` for this adapter. Under Option 2:
    //   * The candle backward (via the shim's `bwd`) emits
    //     `upstream` directly into `grads[arg.id()]`.
    //   * If we registered the output mapping, the bridge's
    //     `with_tape_scope_emit_to_grad_store` walker would seed the
    //     kt-tape node and then `insert_or_add_by_raw` the same
    //     `upstream` value into `grads[arg.id()]` AGAIN — a
    //     `upstream + upstream = 2*upstream` double-count bug.
    //   * Without the mappings, the tape recording exists but the
    //     walker doesn't traverse the InjectGradient node (no seed
    //     entry point), so GradStore stays correct.
    let _ = with_active_tape(|tape: &mut Tape| {
        tape.record(&out_kt, &[&arg_kt], Box::new(backward_op));
    });

    // Build the candle-side output. The shim's `bwd` returns the
    // precomputed `upstream` directly (Option 2; commit e2f8723c),
    // mirroring `InjectTensorGradient::bwd` byte-for-byte. Candle's
    // `loss.backward()` walks from the scalar leaf back to `arg.id()`
    // and populates the GradStore with `upstream`, then propagates
    // through any upstream candle ops above `arg`.
    //
    // The candle device construction above clones `arg.device()`
    // verbatim; we don't need the graph-capturable stream helper here
    // because this placeholder isn't part of the hot decode CUDA
    // graph. Reference the helper so callers can grep for it as the
    // pattern they would use if they ever needed graph-capturable
    // scalars on this path.
    let _ = candle_cuda_device_with_stream_no_event_tracking;
    let candle_out = arg
        .apply_op1(InjectGradientCandleShim {
            upstream: upstream_typed_c.clone(),
        })
        .map_err(|e| {
            BridgeError::new(format!(
                "tape_bridge::inject_gradient_kt: arg.apply_op1(shim): {e}"
            ))
        })?;

    Ok(candle_out)
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

    // The full end-to-end bridge tests live in
    // `kiln-kt-bridge/tests/tape_bridge_e2e.rs` (gated on
    // `feature = "cuda"`). They require a live CUDA device and
    // exercise the bridge through the production `try_tape_*_cuda`
    // adapters.
}
