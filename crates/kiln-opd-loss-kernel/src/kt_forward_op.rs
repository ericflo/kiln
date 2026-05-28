//! Production-caller migration to `KtForwardOp` for OPD per-position
//! reverse-KL ((#1082) — see `docs/CANDLE_REMOVAL_PLAN.md`).
//!
//! # What this module wires together
//!
//! The OPD trainer (`crates/kiln-train/src/opd.rs:1207`) previously
//! called [`crate::opd_top_k_reverse_kl_phase_a_per_position`]
//! directly. Phase A is a pure-candle composite (`index_select` →
//! `matmul` → `log_softmax_last` → `exp` → broadcast subtract →
//! multiply → `sum`); it relies on candle's autograd graph to back-
//! propagate `mean_kl.backward()` into the LoRA Vars that produced
//! `student_hidden`.
//!
//! [`opd_top_k_reverse_kl_per_position_via_kt_forward_op`] replaces
//! that candle composite with a **single** candle `CustomOp1` —
//! [`kiln_kt_bridge::forward_op::KtForwardOp1`] (commit `095f1c74`) —
//! whose forward closure runs the same per-position candle composite
//! as a leaf operation (so the resulting candle autograd graph has
//! one node per OPD call instead of ~8) and whose backward closure
//! calls the kt-typed CUDA backward
//! ([`crate::opd_top_k_reverse_kl_phase_b_bwd_kt`]) on the same fused
//! `kiln_opd_topk_kl_bwd_*` FFI symbols the Phase B candle path
//! already uses. The gradient comes from the same fused CUDA kernel
//! as the Phase B `OpdLossCustomOp::bwd` body (migrated to the kt
//! bridge in commit `0c1be227`).
//!
//! # Forward closure runs the full kt entry
//!
//! The forward closure runs the kt-typed
//! [`crate::opd_top_k_reverse_kl_per_position_kt`] composite end-to-
//! end on CUDA. The kt entry's `head_t.index_select(1, ...)` gather
//! along axis 1 is now supported by the kt-tensor CUDA substrate
//! (see `crates/kiln-tensor/src/ops/index_select.rs::cuda_fwd` and the
//! `kiln_index_select_axis_n_kernel` in
//! `crates/kiln-tensor/csrc/index_select.cu` — both landed earlier in
//! (#1082)). The forward closure borrows the candle CUDA tensors into
//! kt (`kt_tensor_from_candle_cuda_borrow`, zero-copy), runs the kt
//! per-position composite, and copies the `[T_active]` F32 result
//! back to candle (`kt_tensor_to_candle_cuda_copy`). The backward
//! closure already runs through the fused kt CUDA kernel
//! (`opd_top_k_reverse_kl_phase_b_bwd_kt`), so the full kt round-trip
//! for OPD per-position is now in place.
//!
//! # Envelope and fallback
//!
//! The kt-typed backward is gated to `K ∈ {16, 32}` and
//! `dtype ∈ {F32, BF16}` (matching `phase_b::cuda_kernel_supports`).
//! It also requires `hidden.dtype() == head_t.dtype()`. When ANY of
//! the following hold we fall back to the candle Phase A path so the
//! production caller stays correct for the full input envelope it
//! supports today:
//!
//! - `KILN_DISABLE_OPD_KT_FORWARD_OP=1` (kill switch — same convention
//!   as `KILN_DISABLE_OPD_BWD_KT_BRIDGE`, `KILN_DISABLE_OPD_LOSS_KERNEL`,
//!   `KILN_DISABLE_RMSNORM_KERNEL`, etc.).
//! - `hidden` is not on CUDA (the shim is CUDA-only by construction).
//! - `top_k` is not 16 or 32.
//! - `hidden.dtype()` is not F32 or BF16.
//! - `hidden.dtype() != head_t.dtype()`.
//! - `active_count == 0` (no rows to compute; matches the empty-
//!   tensor short-circuit in Phase A / Phase B per-position).
//!
//! Falling back preserves the `mean_kl.backward()` autograd chain
//! the trainer relies on — Phase A's autograd graph is built off
//! `hidden`'s LoRA-Var parents in exactly the same way the kt-shim
//! graph is.
//!
//! # Why this lives in `kiln-opd-loss-kernel` and not `kiln-train`
//!
//! Three reasons:
//!
//! 1. The kt-typed forward + backward entries
//!    ([`crate::opd_top_k_reverse_kl_per_position_kt`],
//!    [`crate::opd_top_k_reverse_kl_phase_b_bwd_kt`]) and the
//!    `cuda_kernel_supports` envelope check are crate-internal here.
//! 2. The kill switch + envelope check are kernel-policy concerns;
//!    pushing them into the trainer would duplicate them per call
//!    site (currently 1, but `trainer.rs::opd_train` may grow more).
//! 3. The trainer crate (`kiln-train`) doesn't depend on
//!    `kiln-kt-bridge`. Keeping the shim plumbing here means the
//!    trainer's `Cargo.toml` doesn't need to learn about the bridge.
//!
//! The call site change in `kiln-train` is then a one-line swap:
//! `opd_top_k_reverse_kl_phase_a_per_position(...)` →
//! `opd_top_k_reverse_kl_per_position_via_kt_forward_op(...)`.
//!
//! # Numerical contract
//!
//! Up to floating-point associativity in the matmul and the log-
//! softmax / KL reductions, the forward output matches Phase A to
//! ≤ 1e-4 in F32 and ≤ 5e-2 in BF16 (the bf16 bound absorbs
//! rounding in the fused projection). The backward d_hidden matches
//! the candle `cuda_kernel_backward` to the same tolerances — it's
//! the same FFI symbols (`kiln_opd_topk_kl_bwd_{bf16,f32}`).

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::opd_top_k_reverse_kl_phase_a_per_position;

/// Read the `KILN_DISABLE_OPD_KT_FORWARD_OP` kill switch. When set
/// (`1` / `true` / `yes` / `TRUE`), the production caller falls back
/// to the candle Phase A per-position path. Same convention as
/// `KILN_DISABLE_OPD_BWD_KT_BRIDGE` (commit `0c1be227`),
/// `KILN_DISABLE_RMSNORM_KERNEL`, `KILN_DISABLE_FUSED_CONV1D`, etc.
pub fn kt_forward_op_disabled() -> bool {
    std::env::var("KILN_DISABLE_OPD_KT_FORWARD_OP")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// Returns `true` when the `(top_k, dtype, head_t.dtype())` triple
/// is in the fused kt-bwd envelope AND `hidden` is on CUDA. The
/// envelope is the intersection of [`crate::phase_b::cuda_kernel_supports`]
/// (the candle-side check) and the dtype-matching constraint from
/// [`crate::opd_top_k_reverse_kl_phase_b_bwd_kt`].
fn shim_envelope_ok(hidden: &Tensor, head_t: &Tensor, top_k: usize) -> bool {
    if !matches!(hidden.device(), Device::Cuda(_)) {
        return false;
    }
    if top_k != 16 && top_k != 32 {
        return false;
    }
    let h_dt = hidden.dtype();
    let dtype_ok = matches!(h_dt, DType::F32 | DType::BF16);
    if !dtype_ok {
        return false;
    }
    if h_dt != head_t.dtype() {
        return false;
    }
    true
}

/// kt-shim per-position OPD reverse-KL with candle-autograd integration.
///
/// Behavioral envelope:
/// - CUDA + `(top_k, dtype) in {16, 32} × {F32, BF16}` + matching
///   `head_t` dtype → routes through [`KtForwardOp1`] over the
///   kt-typed fused forward+backward.
/// - Anything outside the envelope → falls through to
///   [`opd_top_k_reverse_kl_phase_a_per_position`] (the pure-candle
///   reference path). The autograd chain through
///   `mean_kl.backward()` is preserved in either case.
///
/// The signature mirrors [`opd_top_k_reverse_kl_phase_a_per_position`]
/// so the kiln-train call site is a one-line swap.
///
/// [`KtForwardOp1`]: kiln_kt_bridge::forward_op::KtForwardOp1
pub fn opd_top_k_reverse_kl_per_position_via_kt_forward_op(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    // Kill switch + non-CUDA fallback path: identical to the legacy
    // candle Phase A entry the trainer used before this migration.
    if kt_forward_op_disabled() || !shim_envelope_ok(hidden, head_t, top_k) {
        return opd_top_k_reverse_kl_phase_a_per_position(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        );
    }

    #[cfg(feature = "cuda")]
    {
        return cuda_via_kt_forward_op(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        );
    }

    // Non-cuda builds: shim_envelope_ok() returned true above only if
    // hidden.device() is CUDA — but candle without the `cuda` feature
    // can't have a CUDA device, so we'll never reach here. Leave a
    // belt-and-suspenders fallback so the function is still
    // well-typed.
    #[cfg(not(feature = "cuda"))]
    {
        opd_top_k_reverse_kl_phase_a_per_position(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        )
    }
}

// ---------------------------------------------------------------------------
// CUDA fast path: KtForwardOp1 over kt-typed forward + backward.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn cuda_via_kt_forward_op(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    use crate::kt_api::{OpdLossOutputKt, opd_top_k_reverse_kl_phase_b_bwd_kt};
    use kiln_kt_bridge::forward_op::KtForwardOp1;
    use kiln_kt_bridge::{
        kt_tensor_from_candle_cuda_borrow, kt_tensor_to_candle_cuda_copy,
    };
    use std::sync::Arc;

    // ----- Short-circuit on no active rows ---------------------------------
    //
    // Phase A returns `Tensor::zeros((0,), DType::F32, device)` when
    // `active_count == 0`. The kt forward closure would also handle
    // this internally (it returns an empty `[0]` F32 KtTensor via
    // `empty_per_position()`), and the shim wires that into candle
    // storage cleanly. But going through `apply_op1` on a hidden that
    // has no contribution to the loss attaches a dead branch to the
    // autograd graph; the trainer's `mean_kl.backward()` would then
    // try to evaluate that branch, producing an empty grad scatter.
    // Cheap and safer to defer to Phase A's short-circuit.
    let active_count = label_mask.iter().filter(|&&m| m).count();
    if active_count == 0 {
        return opd_top_k_reverse_kl_phase_a_per_position(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        );
    }

    // ----- Force-contiguous `hidden` --------------------------------------
    //
    // `apply_op1` passes the input layout through to the CustomOp's
    // `cuda_fwd` hook; the kt-bridge borrow path requires contiguous
    // storage. Same defensive contiguous() that `phase_b::apply_op` does.
    let hidden_contig = hidden
        .contiguous()
        .context("force-contiguous hidden for OPD kt-shim")?;

    // ----- Captured shim state (cloned into the closures) -----------------
    //
    // `head_t` is frozen (transposed LM head). Both closures need it.
    // `teacher_topk_indices` / `teacher_topk_logprobs` / `label_mask`
    // are also captured by value so the shim outlives the call frame
    // — `KtForwardOp1` is held by `Arc<Box<dyn CustomOp1>>` inside
    // candle's autograd graph until `mean_kl.backward()` is invoked
    // (the parents drop the op when the graph is collected).
    let head_t_owned = head_t.clone();
    let head_t_owned_bwd = head_t.clone();
    let indices_fwd = teacher_topk_indices.to_vec();
    let indices_bwd = teacher_topk_indices.to_vec();
    let logprobs_fwd = teacher_topk_logprobs.to_vec();
    let logprobs_bwd = teacher_topk_logprobs.to_vec();
    let mask_fwd = label_mask.to_vec();
    let mask_bwd = label_mask.to_vec();

    // ----- Forward closure ------------------------------------------------
    //
    // Computes the per-position reverse-KL via the kt-typed forward
    // [`crate::opd_top_k_reverse_kl_per_position_kt`]. The forward
    // borrows the candle CUDA tensors into kt (zero-copy), runs the
    // kt composite (`index_select` along axis 0 of `hidden` AND axis 1
    // of `head_t`, matmul, log_softmax, KL reduction), and copies the
    // result back to a candle CUDA tensor.
    //
    // This swap (over the previous candle composite) is unblocked by
    // the kt-tensor axis-N `IndexSelectOp::cuda_fwd` substrate change
    // earlier in (#1082). Until that landed, the axis-1 gather of
    // `head_t` fell through to `cpu_fwd`, which couldn't downcast
    // CUDA storage. With axis-N CUDA gather available, the kt entry
    // runs end-to-end on the GPU and the OPD per-position path is now
    // fully kt-typed (forward + backward both go through the fused
    // kt FFI symbols).
    let forward = move |hidden_in: &Tensor| -> candle_core::Result<Tensor> {
        // Force-contiguous both inputs before borrowing into kt; the
        // borrow path requires contiguous storage (same constraint
        // the backward closure enforces).
        let hidden_c = hidden_in.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: contiguous hidden: {e}"
            ))
        })?;
        let head_t_c = head_t_owned.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: contiguous head_t: {e}"
            ))
        })?;

        let hidden_kt = kt_tensor_from_candle_cuda_borrow(&hidden_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: borrow hidden: {e}"
            ))
        })?;
        let head_t_kt = kt_tensor_from_candle_cuda_borrow(&head_t_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: borrow head_t: {e}"
            ))
        })?;

        let per_position_kt = crate::opd_top_k_reverse_kl_per_position_kt(
            &hidden_kt,
            &head_t_kt,
            &indices_fwd,
            &logprobs_fwd,
            &mask_fwd,
            top_k,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: kt per-position call: {e}"
            ))
        })?;

        // The kt entry returns `[T_active]` F32. The trailing
        // `sum_axis(1)` in `per_position_forward_kt` can yield a
        // non-contiguous view in some regimes, so be defensive
        // before the copy-back.
        let per_position_kt_c = if per_position_kt.is_contiguous() {
            per_position_kt
        } else {
            per_position_kt.contiguous().map_err(|e| {
                candle_core::Error::Msg(format!(
                    "opd kt-shim fwd: contiguous per_position: {e}"
                ))
            })?
        };

        let per_position = kt_tensor_to_candle_cuda_copy(&per_position_kt_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: copy-back per_position: {e}"
            ))
        })?;

        per_position.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: contiguous per-position: {e}"
            ))
        })
    };

    // ----- Backward closure ----------------------------------------------
    //
    // For PerPosition output mode, the kt bwd takes `grad_loss`
    // shape `[T_active]` (F32 contiguous on CUDA) and returns
    // `d_hidden` shape `[1, T, H]` in the input dtype. The kt entry
    // handles the active-row gather of `hidden`, the teacher tensor
    // uploads, the fused FFI dispatch, and the `scatter_add` from
    // `[T_active, H]` back into `[1, T, H]`.
    //
    // The shim passes us (`arg=hidden, res=per_position_kl,
    // grad_res=grad_per_position_kl`); we ignore `res` since the
    // backward doesn't depend on the forward output value.
    let backward = move |arg: &Tensor,
                         _res: &Tensor,
                         grad_res: &Tensor|
          -> candle_core::Result<Option<Tensor>> {
        let hidden_c = arg.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: contiguous hidden: {e}"
            ))
        })?;
        let head_t_c = head_t_owned_bwd.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: contiguous head_t: {e}"
            ))
        })?;
        // `grad_res` comes in as F32 already (the per-position KL is
        // F32 from `opd_top_k_reverse_kl_per_position_kt`'s output
        // dtype), but cast defensively in case an upstream candle op
        // re-typed the gradient (e.g. mixed-precision wrappers).
        let grad_res_f32 = grad_res.to_dtype(DType::F32).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: cast grad to F32: {e}"
            ))
        })?;
        let grad_res_c = grad_res_f32.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: contiguous grad: {e}"
            ))
        })?;

        let hidden_kt = kt_tensor_from_candle_cuda_borrow(&hidden_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: borrow hidden: {e}"
            ))
        })?;
        let head_t_kt = kt_tensor_from_candle_cuda_borrow(&head_t_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: borrow head_t: {e}"
            ))
        })?;
        let grad_res_kt = kt_tensor_from_candle_cuda_borrow(&grad_res_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: borrow grad: {e}"
            ))
        })?;

        let d_hidden_kt = opd_top_k_reverse_kl_phase_b_bwd_kt(
            &hidden_kt,
            &head_t_kt,
            &indices_bwd,
            &logprobs_bwd,
            &mask_bwd,
            &grad_res_kt,
            top_k,
            OpdLossOutputKt::PerPosition,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: kt bwd call: {e}"
            ))
        })?;

        // `scatter_add` builds the `[1, T, H]` output contiguous from
        // a freshly zero-allocated buffer, but be defensive — the
        // bridge copy-back requires contiguous storage.
        let d_hidden_kt_c = if d_hidden_kt.is_contiguous() {
            d_hidden_kt
        } else {
            d_hidden_kt.contiguous().map_err(|e| {
                candle_core::Error::Msg(format!(
                    "opd kt-shim bwd: contiguous d_hidden: {e}"
                ))
            })?
        };

        let d_hidden = kt_tensor_to_candle_cuda_copy(&d_hidden_kt_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: copy-back d_hidden: {e}"
            ))
        })?;

        Ok(Some(d_hidden))
    };

    // ----- Apply ----------------------------------------------------------
    let op = KtForwardOp1::new("kiln-opd-loss-kt-forward-op", forward, backward);
    let _ = device; // unused; the device is implicit in `hidden_contig`'s storage.
    hidden_contig
        .apply_op1_arc(Arc::new(Box::new(op)))
        .context("apply OPD kt-forward-op to hidden")
}

// ---------------------------------------------------------------------------
// Unit tests (cfg-test only). End-to-end CUDA parity lives in
// `kiln-opd-loss-kernel/tests/cuda_kt_forward_op_parity.rs` next to
// the existing `cuda_opd_bwd_kt_bridge_wiring_parity` test suite.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate `KILN_DISABLE_OPD_KT_FORWARD_OP` so
    // they don't race against each other. Cargo runs tests in threads
    // within a single process; without a lock, `kill_switch_on` can
    // set the env to "1" in the middle of `kill_switch_default_off`'s
    // sequence of `set_var` + `assert!(!kt_forward_op_disabled())`
    // calls, causing a flaky failure (observed on macOS CI run
    // 26572612427, commit 449fb6b1, #1082). The mutex is held for the
    // entire test body — including the prior-value capture and the
    // restore — so the env var is logically owned by exactly one
    // test at a time.
    //
    // We use `parking_lot`-style poison handling (i.e. `.unwrap()` on
    // a poisoned lock would normally propagate the panic from the
    // previous test, but here we want to clear poison so a panic in
    // one test doesn't cascade-fail the next): we explicitly recover
    // from poisoning since the only shared state is the env var
    // itself, which both tests fully restore at the end of their
    // happy paths. If a prior test panicked mid-mutation, the env
    // may still hold a stale value, but each test re-establishes its
    // own starting condition (remove_var / set_var) before asserting.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poisoned| {
            // A previous test panicked while holding the lock. Clear
            // the poison and proceed — each test below re-establishes
            // its own starting env state before asserting.
            ENV_LOCK.clear_poison();
            poisoned.into_inner()
        })
    }

    #[test]
    fn kill_switch_default_off() {
        let _guard = env_lock();
        // Sanity: the default env should not trigger the kill switch.
        // (The harness may have set KILN_DISABLE_OPD_KT_FORWARD_OP=0,
        // which also reads as off.)
        // SAFETY: we don't set the env here — just check that an
        // unset (or "0") env yields false.
        let prior = std::env::var("KILN_DISABLE_OPD_KT_FORWARD_OP").ok();
        // SAFETY: env modification is intra-test and serialized via
        // `ENV_LOCK` against the other test in this module that
        // mutates the same var. The lock is held for the entire test
        // body (including the restore), so no concurrent reader in
        // this binary observes a partially-updated env.
        unsafe {
            std::env::remove_var("KILN_DISABLE_OPD_KT_FORWARD_OP");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", "0");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", "false");
        }
        assert!(!kt_forward_op_disabled());

        // Restore the prior value.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_OPD_KT_FORWARD_OP"),
            }
        }
    }

    #[test]
    fn kill_switch_on() {
        let _guard = env_lock();
        let prior = std::env::var("KILN_DISABLE_OPD_KT_FORWARD_OP").ok();
        for v in ["1", "true", "yes", "TRUE", "Yes"] {
            unsafe {
                std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", v);
            }
            assert!(
                kt_forward_op_disabled(),
                "expected disabled for env={v}"
            );
        }
        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_OPD_KT_FORWARD_OP"),
            }
        }
    }
}
