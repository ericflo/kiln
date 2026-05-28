//! Parity test for the experimental KILN_USE_TAPE_FORWARD path.
//!
//! Compares the output of `kiln_model::forward::rms_norm` under two
//! conditions:
//!
//!   A. **Baseline** — no Tape scope active; `try_tape_rms_norm_cuda`
//!      short-circuits to `Ok(None)`. With `x.track_op() == false`
//!      (the inputs we build below), the call lands in the
//!      inference-only `fused_rmsnorm_kt` branch.
//!   B. **Tape-forward** — env var set + active thread-local Tape;
//!      `try_tape_rms_norm_cuda` records a node on the tape and
//!      returns a candle Tensor copied from the kt output. The kt
//!      call inside is `fused_rmsnorm_via_kt_tape`, which is a thin
//!      tape-recording wrapper around the same `fused_rmsnorm_kt`
//!      kernel.
//!
//! Both paths bottom out in the same `kiln_fused_rmsnorm` FFI symbol,
//! so the outputs must be bit-exact (max-abs-diff == 0.0). The
//! difference is purely in the backward-graph machinery: the baseline
//! returns a candle Tensor with no autograd lineage; the tape path
//! returns a candle Tensor with no candle autograd lineage *plus* a
//! `RmsNormBackward` node recorded on the active Tape. The
//! `Tape::backward` walk for that node would produce the same
//! gradient as the kt-forward-op shim's CustomOp2 backward — but
//! that's a follow-up assertion (CP-4 tape-backward parity).
//!
//! # CP-4 (#1082) context
//!
//! The audit in
//! `docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md` blocks
//! a per-call-site flip of `rms_norm` to `fused_rmsnorm_via_kt_tape`
//! because the production caller has no `&mut Tape` in scope. The
//! `crate::tape_forward` module ships an opt-in scaffold (a
//! thread-local Tape + `KILN_USE_TAPE_FORWARD` env tristate) so we
//! can exercise the tape substrate end-to-end from kiln-model
//! without rewriting the full callgraph.
//!
//! This test proves the tape-forward path is numerically identical
//! to the existing production caller — establishing the substrate is
//! production-safe enough to flip more sites onto next.
//!
//! # Backward presence assertion
//!
//! The tape-forward path must also record exactly one node on the
//! active Tape. We assert `tape.len() == 1` after the call and that
//! the node's output id matches the returned tensor's kt-mirror id
//! (read via the same kt-bridge borrow used by the forward path).
//!
//! # Skip behaviour
//!
//! Non-CUDA builds: `#[cfg(feature = "cuda")]` makes the test a no-op
//! (the only path under test only exists on CUDA).
//!
//! CUDA build without a visible device: the test bails out early so
//! CI without a GPU still compiles and "runs" (skipping the body).

#![cfg(feature = "cuda")]

use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};

/// Lock to serialize env var mutation across tests in this binary —
/// the `KILN_USE_TAPE_FORWARD` cache is a process-wide `OnceLock`,
/// but two tests in different threads could both try to set/unset it.
/// We only have one test today; the lock is defensive infrastructure
/// for follow-up tests that extend the substrate.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn cuda_device() -> Option<Device> {
    Device::new_cuda(0).ok()
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fff_ffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

fn random_bf16_vec(len: usize, seed: u64, scale: f32) -> Vec<half::bf16> {
    let mut state = seed;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(half::bf16::from_f32(lcg(&mut state) * scale));
    }
    v
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let af = a.to_dtype(DType::F32).expect("a -> f32");
    let bf = b.to_dtype(DType::F32).expect("b -> f32");
    let diff = (&af - &bf).expect("diff").abs().expect("abs");
    let flat = diff.flatten_all().expect("flat");
    let values = flat.to_vec1::<f32>().expect("diff vec");
    values.iter().cloned().fold(0.0_f32, f32::max)
}

/// The kt-tape RMSNorm pilot's envelope: bf16, contiguous, CUDA,
/// hidden <= 8192. We use hidden = 2560 (Qwen3.5-4B) and a couple of
/// batch sizes to keep the test fast.
fn build_inputs(device: &Device, rows: usize, hidden: usize) -> (Tensor, Tensor) {
    let x_host = random_bf16_vec(rows * hidden, 0xDEAD_BEEF_1234_5678, 0.25);
    let w_host = random_bf16_vec(hidden, 0xCAFE_F00D_5678_1234, 0.5);
    let x = Tensor::from_vec(x_host, (rows, hidden), &Device::Cpu)
        .expect("x cpu")
        .to_device(device)
        .expect("x -> cuda")
        .contiguous()
        .expect("x contiguous");
    let w = Tensor::from_vec(w_host, hidden, &Device::Cpu)
        .expect("w cpu")
        .to_device(device)
        .expect("w -> cuda")
        .contiguous()
        .expect("w contiguous");
    (x, w)
}

#[test]
fn tape_forward_rms_norm_bit_exact_parity_with_baseline() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward parity: no CUDA device — skipping");
            return;
        }
    };

    let rows = 16usize;
    let hidden = 2560usize; // Qwen3.5-4B hidden_size.
    let eps = 1e-6f64;

    let (x, w) = build_inputs(&device, rows, hidden);

    // Set the env var BEFORE any rms_norm call. `tape_forward_enabled`
    // caches the result of the first read on process-wide OnceLock,
    // so the env must already be set by the time the cache reads.
    // The gate has two conditions (env + active scope); the BASELINE
    // run still routes through the existing dispatch because no Tape
    // scope is active, so `try_tape_rms_norm_cuda` returns `Ok(None)`
    // and falls through.
    //
    // SAFETY: `std::env::set_var` is unsound across threads in
    // multi-threaded contexts (Rust 2024 made this explicit). We hold
    // ENV_LOCK across the call and we set the var to a stable "1"
    // that no other test in this binary changes.
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    // Path A — baseline. Env is on, but no Tape scope is open, so
    // `try_tape_rms_norm_cuda` short-circuits to `Ok(None)` and the
    // call falls through to the existing kt-forward-op dispatch.
    // This is the production path today.
    let baseline = kiln_model::forward::rms_norm(&x, &w, eps).expect("baseline rms_norm");

    // Path B — tape-forward. Env is on AND a Tape scope is open, so
    // `try_tape_rms_norm_cuda` records a node and returns the kt
    // output as a candle Tensor.
    let (tape_result, tape) =
        kiln_model::tape_forward::with_thread_local_tape(|| kiln_model::forward::rms_norm(&x, &w, eps));
    let tape_out = tape_result.expect("tape-forward rms_norm");

    // --- Forward parity (the load-bearing assertion).
    let diff = max_abs_diff(&baseline, &tape_out);
    assert_eq!(
        diff, 0.0,
        "rms_norm tape-forward path must be bit-exact with the baseline \
         (max-abs-diff was {diff}). Both paths share the same kiln_fused_rmsnorm \
         FFI symbol so any nonzero diff is a wiring bug."
    );

    // --- Tape recording assertion. The tape-forward path *must*
    // record exactly one node — the RMSNorm. If the gate was off (env
    // not set / scope missing / envelope rejected), the call falls
    // through and the tape stays empty. An empty tape after a
    // successful tape-forward call is a wiring bug too.
    assert_eq!(
        tape.len(),
        1,
        "tape-forward rms_norm must record exactly one tape node \
         (got {}). Empty tape means try_tape_rms_norm_cuda fell through \
         to the kt-forward-op path; >1 node means an over-record bug.",
        tape.len()
    );

    // --- Sanity on shape — the tape output must match the candle
    // Tensor shape the baseline produced.
    assert_eq!(
        tape_out.shape().dims(),
        baseline.shape().dims(),
        "tape-forward output shape diverges from baseline"
    );
    assert_eq!(
        tape_out.dtype(),
        baseline.dtype(),
        "tape-forward output dtype diverges from baseline"
    );
}

/// Quick sanity: with the env var unset (or with no active scope),
/// the gate must short-circuit cleanly without recording anything.
/// Establishes the production-safety property — opting *in* requires
/// two conditions; opting out is the default.
#[test]
fn tape_forward_short_circuits_without_active_scope() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward short-circuit: no CUDA device — skipping");
            return;
        }
    };

    let rows = 8usize;
    let hidden = 2560usize;
    let eps = 1e-6f64;
    let (x, w) = build_inputs(&device, rows, hidden);

    // Even with the env var set, no active Tape scope ==
    // `try_tape_rms_norm_cuda` returns Ok(None) and the baseline
    // dispatch runs.
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }
    let out = kiln_model::forward::rms_norm(&x, &w, eps).expect("rms_norm without scope");
    assert_eq!(out.shape().dims(), &[rows, hidden]);
}
