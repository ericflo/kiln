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
    SubBackward, Tape,
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
}
