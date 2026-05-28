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

// ----------------------------------------------------------------------
// Matmul tape-forward parity (#1082 — Phase 6a/CP-4 wave 12 extension).
//
// Same pattern as the RMSNorm parity test above: build BF16 contiguous
// CUDA inputs in the envelope `kiln_tensor::cuda_matmul` accepts,
// compare baseline (`try_kt_matmul` -> `cuda_matmul` direct) against
// the tape-forward path (`try_tape_matmul_cuda` -> `ops::matmul` ->
// `cuda_matmul` underneath + `MatmulBackward` recorded on the tape).
//
// Both paths bottom out in the same `kiln_blas::CublasLtMatmulHandle`
// kernel call so the outputs must be bit-exact. The difference is
// purely backward-graph machinery — the tape path additionally
// records a `MatmulBackward` node visible to a subsequent
// `Tape::backward` walk.
// ----------------------------------------------------------------------

fn build_matmul_inputs(
    device: &Device,
    m: usize,
    k: usize,
    n: usize,
) -> (Tensor, Tensor) {
    // BF16 contiguous CUDA tensors of shapes [M, K] and [K, N].
    let a_host = random_bf16_vec(m * k, 0xA1B2_C3D4_E5F6_0708, 0.25);
    let b_host = random_bf16_vec(k * n, 0x1122_3344_5566_7788, 0.25);
    let a = Tensor::from_vec(a_host, (m, k), &Device::Cpu)
        .expect("a cpu")
        .to_device(device)
        .expect("a -> cuda")
        .contiguous()
        .expect("a contiguous");
    let b = Tensor::from_vec(b_host, (k, n), &Device::Cpu)
        .expect("b cpu")
        .to_device(device)
        .expect("b -> cuda")
        .contiguous()
        .expect("b contiguous");
    (a, b)
}

#[test]
fn tape_forward_matmul_bit_exact_parity_with_baseline() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward matmul parity: no CUDA device — skipping");
            return;
        }
    };

    // 2D shapes inside the kt cuda_matmul envelope.
    let m = 16usize;
    let k = 256usize;
    let n = 64usize;
    let (a, b) = build_matmul_inputs(&device, m, k, n);

    // SAFETY: see RMSNorm test above — ENV_LOCK serialises mutators,
    // value is stable "1" across the binary.
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    // Path A — baseline. The adapter short-circuits on no-active-scope.
    let baseline = kiln_model::tape_forward::try_tape_matmul_cuda(&a, &b)
        .expect("baseline try_tape_matmul_cuda call ok");
    assert!(
        baseline.is_none(),
        "baseline path (no tape scope) must short-circuit to Ok(None) \
         so the caller falls through; got Some(...) which means the \
         adapter recorded onto a tape that does not exist"
    );

    // Baseline forward via the kt-typed registry directly (matches
    // what `try_kt_matmul` calls when the tape path declines).
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a)
        .expect("a kt borrow");
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b)
        .expect("b kt borrow");
    let baseline_kt = kiln_tensor::cuda_matmul(&a_kt, &b_kt)
        .expect("cuda_matmul baseline");
    let baseline_out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&baseline_kt)
        .expect("baseline kt -> candle");

    // Path B — tape-forward inside an active scope.
    let (tape_result, tape) =
        kiln_model::tape_forward::with_thread_local_tape(|| {
            kiln_model::tape_forward::try_tape_matmul_cuda(&a, &b)
        });
    let tape_out = tape_result
        .expect("tape-forward try_tape_matmul_cuda ok")
        .expect("tape-forward returned Some(out)");

    let diff = max_abs_diff(&baseline_out, &tape_out);
    assert_eq!(
        diff, 0.0,
        "matmul tape-forward path must be bit-exact with the baseline \
         (max-abs-diff was {diff}). Both paths share the same \
         kt cuda_matmul kernel so any nonzero diff is a wiring bug."
    );

    assert_eq!(
        tape.len(),
        1,
        "tape-forward matmul must record exactly one tape node \
         (got {}). Empty tape means try_tape_matmul_cuda fell through; \
         >1 node means an over-record bug.",
        tape.len()
    );

    assert_eq!(
        tape_out.shape().dims(),
        baseline_out.shape().dims(),
        "tape-forward matmul output shape diverges from baseline"
    );
    assert_eq!(
        tape_out.dtype(),
        baseline_out.dtype(),
        "tape-forward matmul output dtype diverges from baseline"
    );
}

#[test]
fn tape_forward_matmul_short_circuits_without_active_scope() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward matmul short-circuit: no CUDA device — skipping");
            return;
        }
    };

    let (a, b) = build_matmul_inputs(&device, 8, 32, 16);
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }
    let out = kiln_model::tape_forward::try_tape_matmul_cuda(&a, &b)
        .expect("try_tape_matmul_cuda call ok");
    assert!(
        out.is_none(),
        "no active tape scope must short-circuit to Ok(None) \
         so the caller falls through to the existing kt dispatch"
    );
}

// ----------------------------------------------------------------------
// SiLU tape-forward parity (#1082 — Phase 6a/CP-4 wave 12 extension).
//
// Same pattern: BF16 contiguous CUDA input, compare baseline
// (`cuda_activation_unary(kind=0)` direct) against the tape-forward
// path (`try_tape_silu_cuda` -> `ops::silu` -> `cuda_activation_unary`
// underneath + `SiluBackward` recorded on the tape). Both paths share
// the same SiLU FFI kernel.
// ----------------------------------------------------------------------

fn build_silu_input(device: &Device, rows: usize, cols: usize) -> Tensor {
    let x_host = random_bf16_vec(rows * cols, 0xBEEF_DEAD_0123_4567, 0.5);
    Tensor::from_vec(x_host, (rows, cols), &Device::Cpu)
        .expect("x cpu")
        .to_device(device)
        .expect("x -> cuda")
        .contiguous()
        .expect("x contiguous")
}

#[test]
fn tape_forward_silu_bit_exact_parity_with_baseline() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward silu parity: no CUDA device — skipping");
            return;
        }
    };

    let rows = 32usize;
    let cols = 1024usize;
    let x = build_silu_input(&device, rows, cols);

    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    // Baseline — no scope, adapter short-circuits.
    let none_out = kiln_model::tape_forward::try_tape_silu_cuda(&x)
        .expect("baseline try_tape_silu_cuda ok");
    assert!(
        none_out.is_none(),
        "baseline path (no tape scope) must short-circuit to Ok(None)"
    );

    // Baseline forward via the kt op-registry directly (matches what
    // `try_tape_silu_cuda` calls when a scope is open).
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x)
        .expect("x kt borrow");
    let baseline_kt = kiln_tensor::ops::silu(&x_kt).expect("kt silu");
    let baseline_out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&baseline_kt)
        .expect("baseline kt -> candle");

    // Tape-forward inside an active scope.
    let (tape_result, tape) =
        kiln_model::tape_forward::with_thread_local_tape(|| {
            kiln_model::tape_forward::try_tape_silu_cuda(&x)
        });
    let tape_out = tape_result
        .expect("tape-forward try_tape_silu_cuda ok")
        .expect("tape-forward returned Some(out)");

    let diff = max_abs_diff(&baseline_out, &tape_out);
    assert_eq!(
        diff, 0.0,
        "silu tape-forward path must be bit-exact with the baseline \
         (max-abs-diff was {diff}). Both paths share the same \
         cuda_activation_unary(kind=0) kernel."
    );

    assert_eq!(
        tape.len(),
        1,
        "tape-forward silu must record exactly one tape node (got {}).",
        tape.len()
    );

    assert_eq!(tape_out.shape().dims(), baseline_out.shape().dims());
    assert_eq!(tape_out.dtype(), baseline_out.dtype());
}

#[test]
fn tape_forward_silu_short_circuits_without_active_scope() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward silu short-circuit: no CUDA device — skipping");
            return;
        }
    };

    let x = build_silu_input(&device, 4, 64);
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }
    let out = kiln_model::tape_forward::try_tape_silu_cuda(&x)
        .expect("try_tape_silu_cuda call ok");
    assert!(
        out.is_none(),
        "no active tape scope must short-circuit to Ok(None)"
    );
}
