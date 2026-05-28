//! Tape-based single-step training entry — CP-4 substrate pilot for `kiln-train`.
//!
//! Phase 6a/CP-4 of the candle-removal plan (#1082) — see
//! [`docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`] and
//! [`docs/CANDLE_REMOVAL_PLAN.md`] §"Top 3 next-tasks" #1.
//!
//! # Why this module exists
//!
//! Three kt-tape pilots already landed:
//!
//! * `kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_tape` (`895162ca`)
//! * `kiln_flce_kernel::fused_linear_cross_entropy_phase_b_via_kt_tape`
//! * `kiln_opd_loss_kernel::*_via_kt_tape`
//!
//! Each is a parallel entry alongside the existing candle-CustomOp shim.
//! None of them can be promoted to the production caller because
//! `kiln-train` drives backward through
//! `candle_core::Tensor::backward()` (the candle `BackpropOp` /
//! `GradStore` walker), while the pilots record onto
//! `kiln_autograd::Tape` (a disjoint walker). The two graph systems are
//! mutually-incompatible until `kiln-train` itself adopts the
//! `kiln_autograd::Tape` substrate.
//!
//! This module is **the first `kiln-train` training entry built
//! entirely on `kiln_autograd::Tape`** — no `candle_core::Tensor`, no
//! `loss.backward()`, no `candle_core::backprop::GradStore`. It runs a
//! single end-to-end supervised-learning step (forward + scalar loss +
//! tape backward + SGD parameter update) on CPU `kiln_tensor::Tensor`s,
//! proves the substrate composes for the kiln-train crate, and exposes
//! the gradient store so callers can verify per-parameter gradient
//! flow.
//!
//! It is **not** the production training entry — the trainer's real
//! per-step loop in `crate::trainer` still drives `loss.backward()`.
//! Promoting any of the three kt-tape kernel pilots to the production
//! caller requires this substrate to first cover the full per-step
//! graph (matmul + rmsnorm + FLCE + OPD + LoRA fan-in). That is multi-
//! PR work; this commit lands the smallest end-to-end proof point so
//! the next PR has a concrete substrate to extend, not a green-field.
//!
//! # What this module proves
//!
//! 1. `kiln-train` *can* depend on `kiln_autograd` without breaking
//!    the candle-typed trainer (no shared symbols, no feature-gating
//!    conflicts).
//! 2. A real supervised-learning step composes end-to-end on
//!    `kiln_tensor::Tensor` + `kiln_autograd::Tape` + `kiln_tensor::ops`.
//! 3. The `GradStore` returned by `Tape::backward` exposes per-input
//!    gradients keyed on `TensorId` — the same key kiln-param /
//!    kiln-optim use for the optimizer state slot — so the substrate
//!    plugs cleanly into the existing parameter machinery once the
//!    trainer's forward path is ported.
//! 4. The step makes monotone progress against the analytic optimum
//!    when iterated — confirming the gradient signs and magnitudes
//!    are correct, not just present.
//!
//! # Substrate gaps surfaced by this pilot
//!
//! The implementation deliberately stays on the smallest viable
//! footprint. The following gaps remain before this pilot can be
//! extended to the production trainer's per-step graph. Each is
//! documented inline at the call site that hits it, so the next PR
//! has a punch list rather than a green-field design problem:
//!
//! * **No `kiln_param::Parameter` integration in this module.** The
//!   step takes raw `&Tensor` inputs and returns the new parameter
//!   tensors directly. The Parameter slot-coherence story (anti-
//!   pattern 11) lives in `kiln_optim`'s integration tests; folding
//!   `Parameter`s in here would couple two substrate ports together
//!   and double the PR's blast radius. The next PR (real training
//!   step over LoRA parameters) MUST thread `Parameter` through.
//! * **No optimizer state.** Pure SGD. Adam/AdamW state lives in
//!   `kiln-optim` and is already kt-typed; wiring it up is a separate
//!   substrate concern.
//! * **CPU-only F32.** The kernel pilots (rmsnorm/FLCE/OPD via kt-tape)
//!   are CUDA + BF16. Bridging CPU substrate + GPU kernels through
//!   `Tape::record` is straightforward (the kernel pilots' recorded
//!   `BackwardOp` already handles BF16/F32 internally), but exercising
//!   that path requires a CUDA device — out of scope for this pilot.
//! * **No `kiln_model::forward` path.** The production trainer feeds
//!   `loss.backward()` on the output of `model_forward_paged` (the
//!   full transformer forward). Porting that forward to `&mut Tape`
//!   is the bulk of CP-4 — every `rms_norm` / `matmul` / `silu` /
//!   `embedding` site needs a `tape.record` companion. This pilot
//!   establishes the substrate; the model-forward port lands in
//!   subsequent PRs.
//!
//! # Numerical contract
//!
//! Single training step: `pred = x @ w + b ; err = pred - target ;
//! loss = sum(err * err)`. Gradients computed by `Tape::backward` are
//! analytic — `d_w = 2 X^T (Xw - y)`, `d_b = 2 (Xw - y)` for unbiased
//! MSE. The step takes a learning rate `lr` and applies `new = old -
//! lr * grad` per-element on CPU `F32`.
//!
//! # Sibling pilots for context
//!
//! * `kiln_autograd::tests::training_loop_descent` — proves the same
//!   end-to-end recipe works on the *kiln-autograd side*. This module
//!   lifts that recipe into `kiln-train`'s namespace so the next PRs
//!   can extend it inside the crate boundary where the production
//!   trainer lives.
//! * `kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_tape` — the parallel
//!   kt-tape pilot for the rmsnorm kernel. Once this `kiln-train`
//!   substrate covers the per-step graph, the production
//!   `kiln_model::forward::rms_norm` can flip to call the kt-tape
//!   entry (CP-4 final step).

use anyhow::{Context, Result};
use kiln_autograd::{
    AddBackward, GradStore, MatmulBackward, MulBackward, ReduceBackward, ReduceKind, ReduceScope,
    SiluBackward, SubBackward, Tape,
};
use kiln_tensor::{CpuStorage, DType, Layout, Storage, Tensor, TensorId};
use kiln_tensor::ops::{add, matmul, mul, sub, sum_all};
use std::sync::Arc;

/// Inputs to [`linear_step_via_tape`].
///
/// Shapes (CPU F32, contiguous):
/// * `x`: `[n, k]` — features (`n` rows, `k` input dims)
/// * `target`: `[n, m]` — regression targets
/// * `w`: `[k, m]` — weight (the parameter being trained)
/// * `b`: `[n, m]` — bias broadcast across all rows. Kept as a
///   per-row tensor instead of a `[m]` broadcast because the substrate
///   doesn't yet have a broadcast-aware `add` + matching
///   `BroadcastToBackward`. The Phase 6a substrate's broadcast support
///   is already wired in `kiln_autograd::BroadcastToBackward`; folding
///   it into this entry is the next-PR extension, not a blocker for
///   the substrate proof.
///
/// Each input's `TensorId` is preserved through the step so the
/// returned [`StepOutput::grads`] keys cleanly against the original
/// tensor ids.
#[derive(Debug)]
pub struct LinearStepInputs<'a> {
    /// Feature matrix `[n, k]`. F32 CPU.
    pub x: &'a Tensor,
    /// Regression targets `[n, m]`. F32 CPU.
    pub target: &'a Tensor,
    /// Weight `[k, m]`. F32 CPU. Trainable.
    pub w: &'a Tensor,
    /// Bias `[n, m]`. F32 CPU. Trainable.
    ///
    /// Per-row to avoid the broadcast-backward dependency in this
    /// pilot (see struct-level doc).
    pub b: &'a Tensor,
    /// Learning rate for the SGD update. Applied as `new = old - lr * grad`.
    pub lr: f32,
}

/// Outputs of [`linear_step_via_tape`].
///
/// * `loss` — scalar F32 tensor holding the sum-of-squared-errors loss
///   *before* the parameter update. Useful for the caller's logging /
///   convergence assertion.
/// * `loss_value` — the same scalar surfaced as a plain `f32` for
///   ergonomics (no need for the caller to re-read the tensor).
/// * `new_w` / `new_b` — the parameters after one SGD step.
/// * `grads` — the [`GradStore`] returned by `Tape::backward`. Keys
///   include `x.id()`, `target.id()`, `w.id()`, `b.id()`, plus the
///   intermediate output ids. Callers usually only consult
///   `grads.get(w.id())` and `grads.get(b.id())`. Returned by value
///   so tests can introspect downstream grads, not just the parameter
///   grads.
#[derive(Debug)]
pub struct StepOutput {
    /// Pre-update loss as a CPU F32 scalar.
    pub loss: Tensor,
    /// Pre-update loss as `f32` (cached read from `loss`).
    pub loss_value: f32,
    /// Post-step weight `[k, m]`.
    pub new_w: Tensor,
    /// Post-step bias `[n, m]`.
    pub new_b: Tensor,
    /// All gradients produced by the backward walk, keyed on
    /// `TensorId`. The store is keyed on the *forward input ids*
    /// (`x.id()`, `target.id()`, `w.id()`, `b.id()`) plus any
    /// intermediate-output ids the walker created.
    pub grads: GradStore,
}

/// Run one full forward + backward + SGD step on CPU `kiln_tensor::Tensor`s
/// via `kiln_autograd::Tape`.
///
/// This is the CP-4 substrate's first concrete training entry inside the
/// `kiln-train` crate. It does NOT replace any existing trainer path;
/// it sits alongside the candle-typed trainer as a proof point that
/// `kiln_autograd::Tape` can carry a real training step end-to-end from
/// inside the same crate that today owns `loss.backward()` for the
/// production model.
///
/// # Forward
///
/// ```text
///     pred_raw = x @ w           # [n, m]
///     pred     = pred_raw + b    # [n, m]
///     err      = pred - target   # [n, m]
///     sq       = err * err       # [n, m]
///     loss     = sum_all(sq)     # scalar
/// ```
///
/// Each op records onto a freshly-allocated `Tape`. The tape is
/// dropped at the end of the function — there is no notion of
/// "graph reuse across steps" yet (that's an optimizer-state concern
/// not addressed here).
///
/// # Backward
///
/// `Tape::backward(loss.id(), seed=1.0, accumulator=add)` walks the
/// nodes in reverse-insertion order and produces a [`GradStore`] keyed
/// on `TensorId`. The walker's accumulator is
/// `kiln_tensor::ops::add` (the canonical kt-side gradient
/// accumulator).
///
/// # SGD step
///
/// `new = old - lr * grad`, element-wise on the CPU F32 storage. The
/// resulting tensors are fresh `kiln_tensor::Tensor`s with new
/// `TensorId`s; the caller is responsible for binding them back into
/// any `Parameter` slot it holds. See the module-level doc for the
/// gap analysis on `Parameter` integration.
///
/// # Errors
///
/// Returns `Err` if any of:
/// * shape mismatch (caught by `matmul` / `add` / `sub` / `mul` op
///   validators)
/// * non-F32 dtype
/// * non-CPU device (this entry is CPU-only; the kernel pilots cover
///   the GPU envelope)
/// * any intermediate op fails (e.g. out-of-memory on the CPU
///   allocator)
pub fn linear_step_via_tape(inputs: LinearStepInputs<'_>) -> Result<StepOutput> {
    let LinearStepInputs {
        x,
        target,
        w,
        b,
        lr,
    } = inputs;

    // ---- Forward + tape recording ----
    let mut tape = Tape::new();

    // pred_raw = x @ w
    let pred_raw = matmul(x, w).context("tape_step: matmul(x, w) forward")?;
    tape.record(
        &pred_raw,
        &[x, w],
        Box::new(MatmulBackward {
            a: x.clone(),
            b: w.clone(),
        }),
    );

    // pred = pred_raw + b
    let pred = add(&pred_raw, b).context("tape_step: add(pred_raw, b) forward")?;
    tape.record(&pred, &[&pred_raw, b], Box::new(AddBackward));

    // err = pred - target
    let err = sub(&pred, target).context("tape_step: sub(pred, target) forward")?;
    tape.record(&err, &[&pred, target], Box::new(SubBackward));

    // sq = err * err
    let sq = mul(&err, &err).context("tape_step: mul(err, err) forward")?;
    tape.record(
        &sq,
        &[&err, &err],
        Box::new(MulBackward {
            a: err.clone(),
            b: err.clone(),
        }),
    );

    // loss = sum_all(sq)
    let loss = sum_all(&sq).context("tape_step: sum_all(sq) forward")?;
    tape.record(
        &loss,
        &[&sq],
        Box::new(ReduceBackward {
            input_shape: sq.shape().to_vec(),
            dtype: sq.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    let loss_value = scalar_f32(&loss).context("tape_step: read pre-update loss scalar")?;

    // ---- Backward ----
    let seed = Tensor::from_slice(&[1.0_f32], vec![])
        .context("tape_step: build scalar seed gradient (1.0)")?;
    let grads = tape
        .backward(loss.id(), seed, |a, b| add(a, b))
        .context("tape_step: Tape::backward walk")?;

    let d_w = grads
        .get(w.id())
        .context("tape_step: GradStore missing d_w")?;
    let d_b = grads
        .get(b.id())
        .context("tape_step: GradStore missing d_b")?;

    // ---- SGD update ----
    let new_w = sgd_step_cpu(w, d_w, lr).context("tape_step: SGD step on w")?;
    let new_b = sgd_step_cpu(b, d_b, lr).context("tape_step: SGD step on b")?;

    Ok(StepOutput {
        loss,
        loss_value,
        new_w,
        new_b,
        grads,
    })
}

// ---------------------------------------------------------------------------
// 2-layer MLP via tape — second substrate proof point exercising more of the
// `kiln_autograd` backward-op coverage (Silu activation + two distinct
// matmuls + per-parameter SGD update). Demonstrates that the substrate
// handles the multi-layer chain a real transformer block needs:
//
//     h = silu(x @ w1)
//     pred = h @ w2
//     loss = sum((pred - target)²)
//
// This is the smallest non-trivial composition that:
//   * exercises 2 separate `MatmulBackward` records keyed on different
//     `TensorId`s (proves the GradStore correctly disambiguates parameters
//     when one tensor flows through multiple ops),
//   * exercises a non-linear activation through `SiluBackward` (proves the
//     substrate's backward-op trait works for closures that capture forward
//     inputs, not just elementwise binary ops),
//   * pushes loss down across iterated steps (proves gradient flow
//     composes through both layers, not just the final layer).
//
// Two layers is the minimum that distinguishes "substrate can do MSE
// regression" (1-layer linear) from "substrate can do deep learning".
// Future PRs extend the same recipe with rmsnorm + linear + silu + linear
// (the actual MLP shape in `kiln_model::forward`'s `mlp_block`).
// ---------------------------------------------------------------------------

/// Inputs to [`mlp_step_via_tape`].
///
/// Shapes (CPU F32, contiguous):
/// * `x`: `[n, k_in]` — features
/// * `target`: `[n, k_out]` — regression targets
/// * `w1`: `[k_in, k_hidden]` — first-layer weight, trainable
/// * `w2`: `[k_hidden, k_out]` — second-layer weight, trainable
///
/// No bias terms in this pilot — the unbiased MLP keeps the broadcast-
/// backward dependency out of scope for this commit. Future PRs add the
/// bias path via `BroadcastToBackward` once the
/// `tape_step::LinearStepInputs` precedent is extended.
#[derive(Debug)]
pub struct MlpStepInputs<'a> {
    /// Feature matrix `[n, k_in]`. F32 CPU.
    pub x: &'a Tensor,
    /// Regression targets `[n, k_out]`. F32 CPU.
    pub target: &'a Tensor,
    /// First-layer weight `[k_in, k_hidden]`. F32 CPU. Trainable.
    pub w1: &'a Tensor,
    /// Second-layer weight `[k_hidden, k_out]`. F32 CPU. Trainable.
    pub w2: &'a Tensor,
    /// Learning rate for the SGD update.
    pub lr: f32,
}

/// Outputs of [`mlp_step_via_tape`].
///
/// Same structure as [`StepOutput`] but with two updated weights instead
/// of one weight + one bias. The `grads` store still exposes per-input
/// gradients keyed on `TensorId` — including the hidden-activation grad,
/// which future PRs use as the boundary for inserting recompute /
/// activation-checkpoint logic.
#[derive(Debug)]
pub struct MlpStepOutput {
    /// Pre-update loss as a CPU F32 scalar.
    pub loss: Tensor,
    /// Pre-update loss as `f32` (cached read from `loss`).
    pub loss_value: f32,
    /// Post-step first-layer weight `[k_in, k_hidden]`.
    pub new_w1: Tensor,
    /// Post-step second-layer weight `[k_hidden, k_out]`.
    pub new_w2: Tensor,
    /// All gradients produced by the backward walk.
    pub grads: GradStore,
}

/// Run one full forward + backward + SGD step for a 2-layer MLP via
/// `kiln_autograd::Tape`.
///
/// # Forward
///
/// ```text
///     h_pre = x @ w1        # [n, k_hidden]
///     h     = silu(h_pre)   # [n, k_hidden]
///     pred  = h @ w2        # [n, k_out]
///     err   = pred - target # [n, k_out]
///     sq    = err * err     # [n, k_out]
///     loss  = sum_all(sq)   # scalar
/// ```
///
/// Each op records onto a freshly-allocated `Tape`. After the backward
/// walk, the `GradStore` contains `d_w1` (keyed on `w1.id()`) and
/// `d_w2` (keyed on `w2.id()`); both are applied via the same
/// CPU-F32 SGD helper as [`linear_step_via_tape`].
///
/// # Errors
///
/// Same envelope as [`linear_step_via_tape`]: shape mismatches caught
/// at the op-validator level, non-F32 dtypes rejected, non-CPU device
/// unsupported (extending to GPU lands once Parameter integration
/// brings the device-aware constructors in).
pub fn mlp_step_via_tape(inputs: MlpStepInputs<'_>) -> Result<MlpStepOutput> {
    let MlpStepInputs {
        x,
        target,
        w1,
        w2,
        lr,
    } = inputs;

    let mut tape = Tape::new();

    // h_pre = x @ w1
    let h_pre = matmul(x, w1).context("mlp_step: matmul(x, w1) forward")?;
    tape.record(
        &h_pre,
        &[x, w1],
        Box::new(MatmulBackward {
            a: x.clone(),
            b: w1.clone(),
        }),
    );

    // h = silu(h_pre)
    let h = kiln_tensor::ops::silu(&h_pre).context("mlp_step: silu(h_pre) forward")?;
    tape.record(
        &h,
        &[&h_pre],
        Box::new(SiluBackward { x: h_pre.clone() }),
    );

    // pred = h @ w2
    let pred = matmul(&h, w2).context("mlp_step: matmul(h, w2) forward")?;
    tape.record(
        &pred,
        &[&h, w2],
        Box::new(MatmulBackward {
            a: h.clone(),
            b: w2.clone(),
        }),
    );

    // err = pred - target
    let err = sub(&pred, target).context("mlp_step: sub(pred, target) forward")?;
    tape.record(&err, &[&pred, target], Box::new(SubBackward));

    // sq = err * err
    let sq = mul(&err, &err).context("mlp_step: mul(err, err) forward")?;
    tape.record(
        &sq,
        &[&err, &err],
        Box::new(MulBackward {
            a: err.clone(),
            b: err.clone(),
        }),
    );

    // loss = sum_all(sq)
    let loss = sum_all(&sq).context("mlp_step: sum_all(sq) forward")?;
    tape.record(
        &loss,
        &[&sq],
        Box::new(ReduceBackward {
            input_shape: sq.shape().to_vec(),
            dtype: sq.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    let loss_value = scalar_f32(&loss).context("mlp_step: read pre-update loss scalar")?;

    // Backward.
    let seed = Tensor::from_slice(&[1.0_f32], vec![])
        .context("mlp_step: build scalar seed gradient")?;
    let grads = tape
        .backward(loss.id(), seed, |a, b| add(a, b))
        .context("mlp_step: Tape::backward walk")?;

    let d_w1 = grads
        .get(w1.id())
        .context("mlp_step: GradStore missing d_w1")?;
    let d_w2 = grads
        .get(w2.id())
        .context("mlp_step: GradStore missing d_w2")?;

    let new_w1 = sgd_step_cpu(w1, d_w1, lr).context("mlp_step: SGD step on w1")?;
    let new_w2 = sgd_step_cpu(w2, d_w2, lr).context("mlp_step: SGD step on w2")?;

    Ok(MlpStepOutput {
        loss,
        loss_value,
        new_w1,
        new_w2,
        grads,
    })
}

// ---------------------------------------------------------------------------
// CPU F32 helpers — keep this file self-contained so the substrate proof
// doesn't depend on Parameter / optimizer wiring landing first.
//
// These helpers are intentionally **not** re-exposed from `kiln-train`'s
// public surface. They exist only so the pilot can run end-to-end without
// pulling in `kiln-optim`. A real `kt-tape`-based optimizer step (Adam /
// AdamW over `kiln_param::Parameter`s) is the next-PR extension.
// ---------------------------------------------------------------------------

/// Read a scalar `kiln_tensor::Tensor` (shape `[]` or any all-1
/// shape) as `f32`. Returns an error if the tensor is not CPU F32.
fn scalar_f32(t: &Tensor) -> Result<f32> {
    if t.dtype() != DType::F32 {
        anyhow::bail!(
            "tape_step::scalar_f32: expected F32 tensor, got {:?}",
            t.dtype()
        );
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .context("tape_step::scalar_f32: tensor not on CPU storage")?;
    let bytes = cpu.as_bytes();
    if bytes.len() < 4 {
        anyhow::bail!(
            "tape_step::scalar_f32: storage has only {} bytes, need >= 4",
            bytes.len()
        );
    }
    Ok(f32::from_le_bytes(bytes[..4].try_into().unwrap()))
}

/// Read a full CPU F32 tensor's contents as `Vec<f32>`. Used by the
/// SGD helper. The clone is intentional — the SGD update produces a
/// fresh tensor, so reading the source into an owned `Vec` lets the
/// helper apply the elementwise update without `Arc::make_mut` /
/// `Storage` interior-mutability concerns.
fn read_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    if t.dtype() != DType::F32 {
        anyhow::bail!(
            "tape_step::read_f32_vec: expected F32 tensor, got {:?}",
            t.dtype()
        );
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .context("tape_step::read_f32_vec: tensor not on CPU storage")?;
    let bytes = cpu.as_bytes();
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// `new = old - lr * grad`, element-wise on CPU F32 storage. Returns
/// a fresh tensor with a new `TensorId` so it is not aliased to the
/// input `param`.
///
/// This deliberately does NOT mutate `param` in place — anti-pattern
/// 16 forbids in-place mutation during a backward that hasn't been
/// cleared. The caller's training loop is expected to either rebind
/// to the new tensors or roll its own `Parameter` slot-coherence
/// shim. Both options are downstream from the substrate work proven
/// here.
fn sgd_step_cpu(param: &Tensor, grad: &Tensor, lr: f32) -> Result<Tensor> {
    let p = read_f32_vec(param).context("sgd_step_cpu: read param")?;
    let g = read_f32_vec(grad).context("sgd_step_cpu: read grad")?;
    anyhow::ensure!(
        p.len() == g.len(),
        "sgd_step_cpu: param/grad element-count mismatch: {} vs {}",
        p.len(),
        g.len()
    );
    let updated: Vec<f32> = p
        .iter()
        .zip(g.iter())
        .map(|(&pv, &gv)| pv - lr * gv)
        .collect();
    let bytes: Vec<u8> = updated
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    let cpu = CpuStorage::from_bytes(DType::F32, bytes)
        .context("sgd_step_cpu: build CpuStorage for new tensor")?;
    let storage: Storage = Arc::new(cpu);
    let new = Tensor::from_parts(
        storage,
        Layout::contiguous(param.shape().to_vec()),
        TensorId::next(),
    )
    .context("sgd_step_cpu: assemble new Tensor")?;
    Ok(new)
}

// ---------------------------------------------------------------------------
// Tests — pure CPU, no GPU envelope. Substrate proof point only.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_inputs() -> (Tensor, Tensor, Tensor, Tensor) {
        // y_i = 2 x1_i + 3 x2_i + 1 (the +1 is absorbed into the constant
        // column to keep the system well-conditioned per the sibling
        // `kiln_autograd::tests::training_loop_descent` analysis).
        let n_samples = 16;
        let mut x_data = Vec::with_capacity(n_samples * 3);
        let mut y_data = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let x1 = if i % 2 == 0 { 1.0_f32 } else { -1.0 };
            let x2 = if (i / 2) % 2 == 0 { 1.0_f32 } else { -1.0 };
            x_data.push(x1);
            x_data.push(x2);
            x_data.push(1.0); // constant column = bias slot
            y_data.push(2.0 * x1 + 3.0 * x2 + 1.0);
        }
        let x = Tensor::from_slice(&x_data, vec![n_samples, 3]).unwrap();
        let target = Tensor::from_slice(&y_data, vec![n_samples, 1]).unwrap();
        let w = Tensor::from_slice(&[0.0_f32, 0.0, 0.0], vec![3, 1]).unwrap();
        let b = Tensor::from_slice(&[0.0_f32; 16], vec![n_samples, 1]).unwrap();
        (x, target, w, b)
    }

    /// One step on a fresh init produces a non-zero `d_w` keyed on
    /// `w.id()`. This is the *minimum* substrate proof: the tape
    /// records, the walker runs, the GradStore exposes the parameter
    /// gradient. If this test passes, the kt-tape substrate is wired
    /// far enough to start porting real trainer code onto it.
    #[test]
    fn single_step_produces_parameter_gradient() {
        let (x, target, w, b) = build_inputs();
        let out = linear_step_via_tape(LinearStepInputs {
            x: &x,
            target: &target,
            w: &w,
            b: &b,
            lr: 0.01,
        })
        .expect("substrate step returned Err");

        // GradStore must expose a w-grad keyed on the *original* w.id().
        let d_w = out
            .grads
            .get(w.id())
            .expect("GradStore missing w.id() — substrate did not key gradient on input id");
        let d_w_vec = read_f32_vec(d_w).expect("read d_w as F32");
        assert_eq!(d_w_vec.len(), 3, "d_w element count must match w shape");

        // The gradient must be non-zero — w starts at zero so the
        // initial residual is just `-target`, and `d_w = 2 * X^T *
        // (Xw - y) = -2 * X^T * y`. With the synthetic dataset the
        // analytic gradient has nontrivial magnitude in every slot.
        let any_nonzero = d_w_vec.iter().any(|&v| v.abs() > 1e-6);
        assert!(
            any_nonzero,
            "d_w was all-near-zero on a non-converged init: {d_w_vec:?}"
        );

        // The post-update w must differ from the pre-update w by
        // lr * d_w. (Loose tolerance — we're checking the SGD step
        // composed, not the exact arithmetic.)
        let new_w_vec = read_f32_vec(&out.new_w).expect("read new_w");
        for (i, &new) in new_w_vec.iter().enumerate() {
            let expected = 0.0 - 0.01 * d_w_vec[i];
            let delta = (new - expected).abs();
            assert!(
                delta < 1e-5,
                "new_w[{i}] = {new} but expected {expected} (lr=0.01 * d_w[{i}]={})",
                d_w_vec[i],
            );
        }

        // Pre-update loss must be positive (w starts at zeros, so the
        // residual is non-trivial).
        assert!(
            out.loss_value > 0.0,
            "pre-update loss must be positive but was {}",
            out.loss_value
        );
    }

    /// Iterate the tape step. Confirms the substrate composes ACROSS
    /// steps — the gradients keyed on the *new* `w.id()` flow into a
    /// further-reduced loss, etc. Mirrors the sibling
    /// `kiln_autograd::tests::training_loop_descent` test but at the
    /// `kiln-train` boundary so any regression in the crate's
    /// dependency wiring (Cargo.toml feature flags, re-export
    /// surface) surfaces here.
    #[test]
    fn iterated_steps_drive_loss_down() {
        let (x, target, w0, b0) = build_inputs();
        let mut w = w0;
        let mut b = b0;
        let mut losses = Vec::with_capacity(50);
        // 200 steps × lr=0.005 — per the sibling test's analysis,
        // X^T X = 16·I so the per-step error factor is (1 - 2·lr·16) =
        // (1 - 0.16) = 0.84 on every mode. 200 steps drives the
        // residual to ~0.84^200 ≈ 1.5e-15 — far below the 5%
        // assertion below.
        for _ in 0..200 {
            let out = linear_step_via_tape(LinearStepInputs {
                x: &x,
                target: &target,
                w: &w,
                b: &b,
                lr: 0.005,
            })
            .expect("substrate step returned Err mid-loop");
            losses.push(out.loss_value);
            w = out.new_w;
            b = out.new_b;
        }
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first * 0.05,
            "iterated substrate steps did not descend: first={first}, last={last}"
        );
    }

    /// Substrate exposes per-step gradients on *every* recorded input,
    /// not just the trained parameter. This is the property that lets
    /// future PRs slot the substrate behind a model-forward whose
    /// intermediate activations need backward signal too (e.g. LoRA
    /// fan-in, ECHO env-CE).
    #[test]
    fn grad_store_keys_include_all_inputs() {
        let (x, target, w, b) = build_inputs();
        let out = linear_step_via_tape(LinearStepInputs {
            x: &x,
            target: &target,
            w: &w,
            b: &b,
            lr: 0.005,
        })
        .expect("substrate step returned Err");

        // The walker records grads keyed on input ids, even when the
        // caller isn't going to update them (x, target are leaves
        // here). The store therefore contains at least 4 entries —
        // x, target, w, b — plus any intermediates the walker
        // produces.
        assert!(
            out.grads.contains(w.id()),
            "GradStore missing w.id() (the trained parameter id)"
        );
        assert!(
            out.grads.contains(b.id()),
            "GradStore missing b.id() (the trained parameter id)"
        );
        // d_target should also be populated — `sub(pred, target)`'s
        // backward returns a non-`None` grad for the right-hand side.
        // This is the explicit hook future PRs will use to plumb
        // gradient signal into intermediate forward results.
        assert!(
            out.grads.contains(target.id()),
            "GradStore missing target.id() — substrate did not record a grad for the sub() RHS",
        );
    }

    /// Substrate is callable from inside `kiln-train` without any
    /// candle imports in the public boundary of *this* module. Guard
    /// against future regressions that would re-introduce a
    /// candle_core dependency in the tape-based path. Compile-only
    /// — if the file builds, this assertion holds.
    #[test]
    fn module_does_not_depend_on_candle_in_signatures() {
        // Type-level smoke check: the API uses kiln_tensor::Tensor
        // throughout. Constructing it via `kiln_tensor::Tensor::from_slice`
        // and feeding it into `linear_step_via_tape` is enough to
        // prove no candle types leak into the call boundary.
        let x = Tensor::from_slice(&[1.0_f32, 0.0], vec![1, 2]).unwrap();
        let target = Tensor::from_slice(&[3.0_f32], vec![1, 1]).unwrap();
        let w = Tensor::from_slice(&[1.0_f32, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::from_slice(&[0.0_f32], vec![1, 1]).unwrap();
        let _ = linear_step_via_tape(LinearStepInputs {
            x: &x,
            target: &target,
            w: &w,
            b: &b,
            lr: 0.01,
        })
        .expect("substrate step returned Err on smoke-shape inputs");
    }

    // -------------------------------------------------------------------
    // MLP-step tests — exercise 2-layer composition through silu.
    // -------------------------------------------------------------------

    /// Build a small XOR-like dataset that's only learnable through a
    /// hidden non-linearity. 2-dim features `(x1, x2) ∈ {-1, +1}²`,
    /// target = `(x1 XOR x2)` mapped to ±1. A linear model can't
    /// represent XOR — so if the MLP test asserts loss descent, the
    /// silu non-linearity in the substrate is doing real work.
    fn build_xor_dataset() -> (Tensor, Tensor) {
        let xs = [
            (-1.0_f32, -1.0_f32, -1.0_f32),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
        ];
        // Replicate each row 4× so the per-step gradient signal is
        // strong enough to descend with vanilla SGD inside 200 steps
        // — XOR's tiny dataset is sensitive to mini-batch averaging.
        let n_replicas = 4;
        let mut x_data = Vec::with_capacity(xs.len() * n_replicas * 2);
        let mut y_data = Vec::with_capacity(xs.len() * n_replicas);
        for _ in 0..n_replicas {
            for &(x1, x2, y) in &xs {
                x_data.push(x1);
                x_data.push(x2);
                y_data.push(y);
            }
        }
        let n = xs.len() * n_replicas;
        let x = Tensor::from_slice(&x_data, vec![n, 2]).unwrap();
        let target = Tensor::from_slice(&y_data, vec![n, 1]).unwrap();
        (x, target)
    }

    /// MLP forward + backward + SGD step produces non-zero gradients
    /// on BOTH weight tensors. This is the multi-layer substrate
    /// proof: the GradStore correctly disambiguates `w1` and `w2`
    /// even though they share the same dtype + similar shape.
    #[test]
    fn mlp_step_produces_per_layer_gradients() {
        let (x, target) = build_xor_dataset();
        // 2 → 4 → 1 MLP. Init w1/w2 with small alternating values
        // so silu's gradient isn't stuck at zero on a symmetric init.
        let w1 = Tensor::from_slice(
            &[0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            vec![2, 4],
        )
        .unwrap();
        let w2 = Tensor::from_slice(&[0.1_f32, -0.1, 0.2, -0.2], vec![4, 1]).unwrap();

        let out = mlp_step_via_tape(MlpStepInputs {
            x: &x,
            target: &target,
            w1: &w1,
            w2: &w2,
            lr: 0.01,
        })
        .expect("MLP substrate step returned Err");

        let d_w1 = out
            .grads
            .get(w1.id())
            .expect("MLP GradStore missing d_w1 — substrate did not record first-layer matmul");
        let d_w2 = out
            .grads
            .get(w2.id())
            .expect("MLP GradStore missing d_w2 — substrate did not record second-layer matmul");

        let d_w1_vec = read_f32_vec(d_w1).unwrap();
        let d_w2_vec = read_f32_vec(d_w2).unwrap();
        assert_eq!(d_w1_vec.len(), 8, "d_w1 shape mismatch");
        assert_eq!(d_w2_vec.len(), 4, "d_w2 shape mismatch");

        // Both layers must have non-zero gradients on a non-converged
        // init. If d_w1 is zero, the gradient signal didn't flow
        // back through silu — meaning the SiluBackward record on
        // the tape isn't running.
        let any_d_w1 = d_w1_vec.iter().any(|&v| v.abs() > 1e-6);
        let any_d_w2 = d_w2_vec.iter().any(|&v| v.abs() > 1e-6);
        assert!(any_d_w1, "d_w1 was all-near-zero: {d_w1_vec:?}");
        assert!(any_d_w2, "d_w2 was all-near-zero: {d_w2_vec:?}");

        // Loss must be positive on a non-converged init.
        assert!(
            out.loss_value > 0.0,
            "pre-update MLP loss must be positive but was {}",
            out.loss_value
        );
    }

    /// Iterated MLP steps drive XOR loss down. XOR is *not* linearly
    /// separable — if this passes, the substrate is genuinely
    /// composing the hidden non-linearity, not just shuttling
    /// gradients through a linear chain.
    ///
    /// This is the strongest substrate-completeness assertion in
    /// this module: it proves the kt-tape substrate can drive
    /// gradient-based learning of a function that the linear-only
    /// substrate (the existing `linear_step_via_tape` tests) cannot
    /// represent.
    #[test]
    fn mlp_step_can_learn_xor() {
        let (x, target) = build_xor_dataset();
        // Slightly larger hidden = 8 to give the optimizer headroom
        // — XOR with hidden=4 and vanilla SGD is on the convergence
        // boundary and can stall in local minima depending on the
        // init. Hidden=8 with a small random init reliably descends.
        let mut w1 = Tensor::from_slice(
            &[
                0.12_f32, -0.31, 0.05, 0.22, -0.18, 0.07, 0.41, -0.09, -0.27, 0.33, 0.16, -0.04,
                0.29, -0.21, 0.11, -0.36,
            ],
            vec![2, 8],
        )
        .unwrap();
        let mut w2 = Tensor::from_slice(
            &[0.13_f32, -0.07, 0.21, -0.19, 0.08, -0.25, 0.17, -0.11],
            vec![8, 1],
        )
        .unwrap();

        let mut losses = Vec::with_capacity(500);
        for _ in 0..500 {
            let out = mlp_step_via_tape(MlpStepInputs {
                x: &x,
                target: &target,
                w1: &w1,
                w2: &w2,
                lr: 0.02,
            })
            .expect("MLP substrate step returned Err mid-loop");
            losses.push(out.loss_value);
            w1 = out.new_w1;
            w2 = out.new_w2;
        }
        let first = losses[0];
        let last = *losses.last().unwrap();
        // Loose tolerance — XOR with vanilla SGD + small dataset is
        // not the fastest descent in the world. Require >= 30%
        // reduction, which is well above noise and confirms the
        // gradient signal is non-trivial across both layers.
        assert!(
            last < first * 0.7,
            "MLP substrate didn't descend on XOR: first={first}, last={last}"
        );
    }

    /// MLP GradStore exposes gradients keyed on **every leaf input**
    /// (x, target, w1, w2). This is the multi-layer analogue of the
    /// 1-layer `grad_store_keys_include_all_inputs` test, and the
    /// property future PRs build on for parameter-update plumbing.
    ///
    /// Substrate behavior surfaced by this test: `Tape::backward`
    /// drains intermediate-output grad entries as it walks (the
    /// walker calls `grads.remove(&node.output_id)` before computing
    /// per-input grads), so the final GradStore is keyed on leaf
    /// `TensorId`s plus the loss seed — not on intermediate
    /// activations. That's a deliberate design choice in
    /// `Tape::backward`; activation-checkpoint / selective-recompute
    /// hooks for the intermediate ids live in the per-`TapeNode`
    /// state, not in the post-walk store. (Documented for the next
    /// PR: porting `kiln_model::forward` ops onto the tape needs to
    /// hold references to intermediate `TapeNode`s if it wants
    /// recompute control, not lean on the GradStore.)
    #[test]
    fn mlp_grad_store_keys_include_both_layer_parameters() {
        let (x, target) = build_xor_dataset();
        let w1 = Tensor::from_slice(
            &[0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            vec![2, 4],
        )
        .unwrap();
        let w2 = Tensor::from_slice(&[0.1_f32, -0.1, 0.2, -0.2], vec![4, 1]).unwrap();

        let out = mlp_step_via_tape(MlpStepInputs {
            x: &x,
            target: &target,
            w1: &w1,
            w2: &w2,
            lr: 0.01,
        })
        .unwrap();

        // Every leaf input from the forward must have a grad keyed
        // on its original `TensorId`. The walker disambiguates `w1`
        // and `w2` correctly even though both feed the same
        // `MatmulBackward` op type.
        assert!(out.grads.contains(x.id()), "GradStore missing x.id()");
        assert!(
            out.grads.contains(target.id()),
            "GradStore missing target.id()"
        );
        assert!(
            out.grads.contains(w1.id()),
            "GradStore missing w1.id() — substrate failed to disambiguate first matmul"
        );
        assert!(
            out.grads.contains(w2.id()),
            "GradStore missing w2.id() — substrate failed to disambiguate second matmul"
        );

        // Substrate property: final GradStore is keyed on leaves +
        // loss seed; intermediates are drained during the walk.
        // (See test docstring.) For 4 leaves + 1 loss seed -> 5 is
        // the natural count, but the walker may leave a few
        // intermediates whose grads weren't consumed because their
        // producer op short-circuited (e.g. err appearing as both
        // inputs of the mul). The floor of >= 4 just guarantees
        // every leaf is keyed.
        assert!(
            out.grads.len() >= 4,
            "MLP GradStore had only {} entries; expected >= 4 (the 4 leaves)",
            out.grads.len()
        );
    }
}
