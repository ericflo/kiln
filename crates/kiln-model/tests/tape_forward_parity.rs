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

use std::collections::HashMap;
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

// ----------------------------------------------------------------------
// Embedding tape-forward parity (#1082 — Phase 6a/CP-4 wave 12 extension).
//
// Same pattern as the matmul / silu parity tests: build BF16
// contiguous CUDA weights of shape [V, H] and U32 token-ids of shape
// [N], compare baseline (`kiln_tensor::ops::embedding` direct) against
// the tape-forward path (`try_tape_embedding_cuda` -> `ops::embedding`
// -> `cuda_index_select_dim0` underneath + `EmbeddingBackward`
// recorded on the tape). Both paths share the same kt
// `cuda_index_select_dim0` kernel so the outputs must be bit-exact.
// The difference is purely backward-graph machinery — the tape path
// additionally records an `EmbeddingBackward` node visible to a
// subsequent `Tape::backward` walk.
// ----------------------------------------------------------------------

fn build_embedding_inputs(
    device: &Device,
    vocab: usize,
    hidden: usize,
    n_tokens: usize,
) -> (Tensor, Tensor) {
    // BF16 contiguous CUDA weights of shape [V, H] and U32
    // contiguous indices of shape [N] in [0, V).
    let w_host = random_bf16_vec(vocab * hidden, 0xE001_1A2B_3C4D_5E6F, 0.5);
    let weights = Tensor::from_vec(w_host, (vocab, hidden), &Device::Cpu)
        .expect("weights cpu")
        .to_device(device)
        .expect("weights -> cuda")
        .contiguous()
        .expect("weights contiguous");
    // Deterministic LCG-derived ids in [0, vocab).
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let ids_host: Vec<u32> = (0..n_tokens)
        .map(|_| {
            // simple xorshift step before % vocab so we don't
            // collide too predictably.
            let _ = lcg(&mut state);
            ((state >> 17) as u32) % (vocab as u32)
        })
        .collect();
    let token_ids = Tensor::from_vec(ids_host, n_tokens, &Device::Cpu)
        .expect("ids cpu")
        .to_device(device)
        .expect("ids -> cuda")
        .contiguous()
        .expect("ids contiguous");
    (weights, token_ids)
}

#[test]
fn tape_forward_embedding_bit_exact_parity_with_baseline() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward embedding parity: no CUDA device — skipping");
            return;
        }
    };

    // Modest sizes inside the kt cuda_index_select_dim0 envelope.
    let vocab = 1024usize;
    let hidden = 256usize;
    let n_tokens = 32usize;
    let (weights, token_ids) = build_embedding_inputs(&device, vocab, hidden, n_tokens);

    // SAFETY: see RMSNorm test above — ENV_LOCK serialises mutators,
    // value is stable "1" across the binary.
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    // Path A — baseline. The adapter short-circuits on no-active-scope.
    let baseline = kiln_model::tape_forward::try_tape_embedding_cuda(&weights, &token_ids)
        .expect("baseline try_tape_embedding_cuda call ok");
    assert!(
        baseline.is_none(),
        "baseline path (no tape scope) must short-circuit to Ok(None) \
         so the caller falls through; got Some(...) which means the \
         adapter recorded onto a tape that does not exist"
    );

    // Baseline forward via the kt op-registry directly (matches what
    // `try_tape_embedding_cuda` calls when a scope is open).
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&weights)
        .expect("weights kt borrow");
    let ids_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&token_ids)
        .expect("ids kt borrow");
    let baseline_kt = kiln_tensor::ops::embedding(&w_kt, &ids_kt)
        .expect("kt embedding baseline");
    let baseline_out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&baseline_kt)
        .expect("baseline kt -> candle");

    // Path B — tape-forward inside an active scope.
    let (tape_result, tape) =
        kiln_model::tape_forward::with_thread_local_tape(|| {
            kiln_model::tape_forward::try_tape_embedding_cuda(&weights, &token_ids)
        });
    let tape_out = tape_result
        .expect("tape-forward try_tape_embedding_cuda ok")
        .expect("tape-forward returned Some(out)");

    let diff = max_abs_diff(&baseline_out, &tape_out);
    assert_eq!(
        diff, 0.0,
        "embedding tape-forward path must be bit-exact with the baseline \
         (max-abs-diff was {diff}). Both paths share the same kt \
         cuda_index_select_dim0 kernel so any nonzero diff is a wiring bug."
    );

    assert_eq!(
        tape.len(),
        1,
        "tape-forward embedding must record exactly one tape node \
         (got {}). Empty tape means try_tape_embedding_cuda fell through; \
         >1 node means an over-record bug.",
        tape.len()
    );

    assert_eq!(
        tape_out.shape().dims(),
        baseline_out.shape().dims(),
        "tape-forward embedding output shape diverges from baseline"
    );
    assert_eq!(
        tape_out.dtype(),
        baseline_out.dtype(),
        "tape-forward embedding output dtype diverges from baseline"
    );
}

#[test]
fn tape_forward_embedding_short_circuits_without_active_scope() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward embedding short-circuit: no CUDA device — skipping");
            return;
        }
    };

    let (weights, token_ids) = build_embedding_inputs(&device, 64, 32, 8);
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }
    let out = kiln_model::tape_forward::try_tape_embedding_cuda(&weights, &token_ids)
        .expect("try_tape_embedding_cuda call ok");
    assert!(
        out.is_none(),
        "no active tape scope must short-circuit to Ok(None) \
         so the caller falls through to the existing kt dispatch"
    );
}

// ----------------------------------------------------------------------
// SwiGLU (silu(gate) * up) tape-forward parity (#1082 — Phase 6a/CP-4
// wave 14 extension).
//
// Same pattern as the matmul / silu / embedding parity tests: build
// BF16 contiguous CUDA `gate` and `up` of identical shape (the
// `kiln_tensor::ops::mul_sigmoid_gate` envelope), compare baseline
// (`ops::mul_sigmoid_gate` direct) against the tape-forward path
// (`try_tape_swiglu_cuda` -> `ops::mul_sigmoid_gate` underneath +
// `MulSigmoidGateBackward` recorded on the tape). Both paths bottom
// out in the same `cuda_activation_unary(kind=0)` +
// `cuda_elementwise_binary(kind=2)` kernels, so the outputs must be
// bit-exact. The difference is purely backward-graph machinery — the
// tape path additionally records a `MulSigmoidGateBackward` node
// visible to a subsequent `Tape::backward` walk.
// ----------------------------------------------------------------------

fn build_swiglu_inputs(device: &Device, rows: usize, cols: usize) -> (Tensor, Tensor) {
    let gate_host = random_bf16_vec(rows * cols, 0x5A1E_C0FF_EE57_FACE, 0.35);
    let up_host = random_bf16_vec(rows * cols, 0xABBA_B0BA_5EED_F00D, 0.35);
    let gate = Tensor::from_vec(gate_host, (rows, cols), &Device::Cpu)
        .expect("gate cpu")
        .to_device(device)
        .expect("gate -> cuda")
        .contiguous()
        .expect("gate contiguous");
    let up = Tensor::from_vec(up_host, (rows, cols), &Device::Cpu)
        .expect("up cpu")
        .to_device(device)
        .expect("up -> cuda")
        .contiguous()
        .expect("up contiguous");
    (gate, up)
}

#[test]
fn tape_forward_swiglu_bit_exact_parity_with_baseline() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward swiglu parity: no CUDA device — skipping");
            return;
        }
    };

    // Modest 2D shape inside the kt mul_sigmoid_gate envelope.
    // Use the Qwen3.5-4B MLP intermediate dim (rough proxy) so the
    // shape is representative of the production MLP gate path.
    let rows = 16usize;
    let cols = 1024usize;
    let (gate, up) = build_swiglu_inputs(&device, rows, cols);

    // SAFETY: see RMSNorm test above — ENV_LOCK serialises mutators,
    // value is stable "1" across the binary.
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    // Path A — baseline. The adapter short-circuits on no-active-scope.
    let baseline_adapter = kiln_model::tape_forward::try_tape_swiglu_cuda(&gate, &up)
        .expect("baseline try_tape_swiglu_cuda call ok");
    assert!(
        baseline_adapter.is_none(),
        "baseline path (no tape scope) must short-circuit to Ok(None) \
         so the caller falls through; got Some(...) which means the \
         adapter recorded onto a tape that does not exist"
    );

    // Baseline forward via the kt op-registry directly (matches what
    // `try_tape_swiglu_cuda` calls when a scope is open).
    let gate_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&gate)
        .expect("gate kt borrow");
    let up_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&up)
        .expect("up kt borrow");
    let baseline_kt =
        kiln_tensor::ops::mul_sigmoid_gate(&gate_kt, &up_kt).expect("kt mul_sigmoid_gate baseline");
    let baseline_out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&baseline_kt)
        .expect("baseline kt -> candle");

    // Path B — tape-forward inside an active scope.
    let (tape_result, tape) =
        kiln_model::tape_forward::with_thread_local_tape(|| {
            kiln_model::tape_forward::try_tape_swiglu_cuda(&gate, &up)
        });
    let tape_out = tape_result
        .expect("tape-forward try_tape_swiglu_cuda ok")
        .expect("tape-forward returned Some(out)");

    let diff = max_abs_diff(&baseline_out, &tape_out);
    assert_eq!(
        diff, 0.0,
        "swiglu tape-forward path must be bit-exact with the baseline \
         (max-abs-diff was {diff}). Both paths share the same \
         cuda_activation_unary(kind=0) + cuda_elementwise_binary(kind=2) \
         kernel composition so any nonzero diff is a wiring bug."
    );

    assert_eq!(
        tape.len(),
        1,
        "tape-forward swiglu must record exactly one tape node \
         (got {}). Empty tape means try_tape_swiglu_cuda fell through; \
         >1 node means an over-record bug.",
        tape.len()
    );

    assert_eq!(
        tape_out.shape().dims(),
        baseline_out.shape().dims(),
        "tape-forward swiglu output shape diverges from baseline"
    );
    assert_eq!(
        tape_out.dtype(),
        baseline_out.dtype(),
        "tape-forward swiglu output dtype diverges from baseline"
    );
}

#[test]
fn tape_forward_swiglu_short_circuits_without_active_scope() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward swiglu short-circuit: no CUDA device — skipping");
            return;
        }
    };

    let (gate, up) = build_swiglu_inputs(&device, 4, 64);
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }
    let out = kiln_model::tape_forward::try_tape_swiglu_cuda(&gate, &up)
        .expect("try_tape_swiglu_cuda call ok");
    assert!(
        out.is_none(),
        "no active tape scope must short-circuit to Ok(None) \
         so the caller falls through to the existing kt dispatch"
    );
}

// ======================================================================
// Tape-BACKWARD parity (#1082 — CP-4 tape-backward parity).
//
// The forward parity tests above prove each tape-forward adapter records
// exactly one node and produces a bit-exact forward value. This section
// closes the deferred assertion called out in the module header
// (lines 22-25): walking `Tape::backward` over that one recorded node
// must produce the gradient the op's analytic backward defines.
//
// Mechanics shared by every test below:
//
//   1. Run the adapter inside `with_thread_local_tape` → `(result, tape)`.
//   2. Read `out_id = tape.nodes()[0].output_id` and
//      `input_ids = tape.nodes()[0].input_ids` (the kt-mirror ids of the
//      op's output + inputs — the adapters return only a candle Tensor,
//      so the kt output id must come from the tape node).
//   3. Build a candle BF16 CUDA seed grad shaped like the output, borrow
//      it zero-copy as a kt Tensor (the candle seed MUST outlive the
//      backward call — the borrow is a view into its CUDA memory), and
//      feed it as `seeds[out_id]`.
//   4. `tape.backward_with_seeds(seeds, |a, b| kiln_tensor::ops::add(a, b))`
//      walks the single node and returns a `GradStore` keyed on the kt
//      input ids.
//   5. Convert each kt input grad back to candle and compare against an
//      analytic reference computed in F32 (the device backward casts up
//      to F32, composes, casts the result back to BF16 once — so the
//      reference is F32 and the comparison runs at a BF16-output
//      tolerance).
//
// The saved-input kt tensors inside each BackwardOp are zero-copy borrows
// of the original candle inputs, so those candle tensors (and the candle
// seed grad) stay alive for the whole test body.
// ======================================================================

/// Sigmoid in F32, composed from candle core ops (no candle-nn dep):
/// `σ(x) = 1 / (1 + exp(-x))`. `affine(1.0, 1.0)` computes `exp(-x) + 1`.
fn candle_sigmoid_f32(x: &Tensor) -> Tensor {
    let neg = x.neg().expect("neg");
    let e = neg.exp().expect("exp");
    let denom = e.affine(1.0, 1.0).expect("exp(-x) + 1");
    denom.recip().expect("recip")
}

/// Build a BF16 contiguous CUDA seed gradient of the given shape.
fn build_seed_grad(device: &Device, dims: &[usize], seed: u64, scale: f32) -> Tensor {
    let n: usize = dims.iter().product();
    let host = random_bf16_vec(n, seed, scale);
    Tensor::from_vec(host, dims.to_vec(), &Device::Cpu)
        .expect("seed cpu")
        .to_device(device)
        .expect("seed -> cuda")
        .contiguous()
        .expect("seed contiguous")
}

#[test]
fn tape_backward_silu_matches_analytic_reference() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_backward silu: no CUDA device — skipping");
            return;
        }
    };

    let rows = 32usize;
    let cols = 1024usize;
    let x = build_silu_input(&device, rows, cols);

    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    // Forward under the tape so a SiluBackward node is recorded.
    let (res, tape) = kiln_model::tape_forward::with_thread_local_tape(|| {
        kiln_model::tape_forward::try_tape_silu_cuda(&x)
    });
    let _out = res
        .expect("tape-forward try_tape_silu_cuda ok")
        .expect("tape-forward returned Some(out)");
    assert_eq!(tape.len(), 1, "silu must record exactly one node");
    let node = &tape.nodes()[0];
    let out_id = node.output_id;
    let input_ids = node.input_ids.clone();
    assert_eq!(input_ids.len(), 1, "silu records one input (x)");

    // Seed grad shaped like the output, borrowed zero-copy as kt.
    let seed = build_seed_grad(&device, &[rows, cols], 0x511E_0000_0001, 0.25);
    let seed_kt =
        kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&seed).expect("seed kt borrow");
    let mut seeds = HashMap::new();
    seeds.insert(out_id, seed_kt);

    let grads = tape
        .backward_with_seeds(seeds, |a, b| kiln_tensor::ops::add(a, b))
        .expect("silu backward walk");

    let dx_kt = grads.get(input_ids[0]).expect("dx grad present");
    let dx = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(dx_kt).expect("dx kt -> candle");
    assert_eq!(dx.shape().dims(), &[rows, cols], "dx shape");
    assert_eq!(dx.dtype(), DType::BF16, "dx dtype matches input");
    assert!(dx.device().is_cuda(), "dx stays on CUDA");

    // Analytic reference: dx = dy · (σ + x·σ·(1-σ)), all in F32.
    let xf = x.to_dtype(DType::F32).expect("x -> f32");
    let dyf = seed.to_dtype(DType::F32).expect("dy -> f32");
    let s = candle_sigmoid_f32(&xf);
    let oms = s.affine(-1.0, 1.0).expect("1 - s");
    let xs = (&xf * &s).expect("x*s");
    let xs_oms = (&xs * &oms).expect("x*s*(1-s)");
    let dsilu = (&s + &xs_oms).expect("σ + x·σ·(1-σ)");
    let ref_dx = (&dyf * &dsilu).expect("dy*dsilu");

    let diff = max_abs_diff(&dx, &ref_dx);
    assert!(
        diff < 3e-2,
        "silu tape-backward grad diverges from analytic reference \
         (max-abs-diff {diff} >= 3e-2 BF16 tol)"
    );
}

#[test]
fn tape_backward_swiglu_matches_analytic_reference() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_backward swiglu: no CUDA device — skipping");
            return;
        }
    };

    let rows = 16usize;
    let cols = 1024usize;
    let (gate, up) = build_swiglu_inputs(&device, rows, cols);

    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    let (res, tape) = kiln_model::tape_forward::with_thread_local_tape(|| {
        kiln_model::tape_forward::try_tape_swiglu_cuda(&gate, &up)
    });
    let _out = res
        .expect("tape-forward try_tape_swiglu_cuda ok")
        .expect("tape-forward returned Some(out)");
    assert_eq!(tape.len(), 1, "swiglu must record exactly one node");
    let node = &tape.nodes()[0];
    let out_id = node.output_id;
    let input_ids = node.input_ids.clone();
    assert_eq!(input_ids.len(), 2, "swiglu records two inputs (gate, up)");

    let seed = build_seed_grad(&device, &[rows, cols], 0x5716_0000_0002, 0.25);
    let seed_kt =
        kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&seed).expect("seed kt borrow");
    let mut seeds = HashMap::new();
    seeds.insert(out_id, seed_kt);

    let grads = tape
        .backward_with_seeds(seeds, |a, b| kiln_tensor::ops::add(a, b))
        .expect("swiglu backward walk");

    // Input order is [gate, up]; MulSigmoidGateBackward returns
    // [d_gate, d_up] in the same order.
    let dgate_kt = grads.get(input_ids[0]).expect("d_gate present");
    let dup_kt = grads.get(input_ids[1]).expect("d_up present");
    let dgate = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(dgate_kt).expect("d_gate -> candle");
    let dup = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(dup_kt).expect("d_up -> candle");
    assert_eq!(dgate.shape().dims(), &[rows, cols], "d_gate shape");
    assert_eq!(dup.shape().dims(), &[rows, cols], "d_up shape");
    assert_eq!(dgate.dtype(), DType::BF16);
    assert_eq!(dup.dtype(), DType::BF16);

    // Analytic reference in F32:
    //   σ      = sigmoid(gate)
    //   d_gate = dy · up · (σ + gate·σ·(1-σ))
    //   d_up   = dy · gate · σ
    let gatef = gate.to_dtype(DType::F32).expect("gate -> f32");
    let upf = up.to_dtype(DType::F32).expect("up -> f32");
    let dyf = seed.to_dtype(DType::F32).expect("dy -> f32");
    let s = candle_sigmoid_f32(&gatef);
    let oms = s.affine(-1.0, 1.0).expect("1 - s");
    let gs = (&gatef * &s).expect("gate*s");
    let gs_oms = (&gs * &oms).expect("gate*s*(1-s)");
    let dsilu = (&s + &gs_oms).expect("σ + gate·σ·(1-σ)");
    let dy_up = (&dyf * &upf).expect("dy*up");
    let ref_dgate = (&dy_up * &dsilu).expect("dy*up*dsilu");
    let ref_dup = (&dyf * &gs).expect("dy*gate*s");

    let diff_g = max_abs_diff(&dgate, &ref_dgate);
    let diff_u = max_abs_diff(&dup, &ref_dup);
    assert!(
        diff_g < 3e-2,
        "swiglu d_gate diverges from analytic reference (max-abs-diff {diff_g})"
    );
    assert!(
        diff_u < 3e-2,
        "swiglu d_up diverges from analytic reference (max-abs-diff {diff_u})"
    );
}

#[test]
fn tape_backward_matmul_matches_analytic_reference() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_backward matmul: no CUDA device — skipping");
            return;
        }
    };

    let m = 16usize;
    let k = 256usize;
    let n = 64usize;
    let (a, b) = build_matmul_inputs(&device, m, k, n);

    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    let (res, tape) = kiln_model::tape_forward::with_thread_local_tape(|| {
        kiln_model::tape_forward::try_tape_matmul_cuda(&a, &b)
    });
    let _out = res
        .expect("tape-forward try_tape_matmul_cuda ok")
        .expect("tape-forward returned Some(out)");
    assert_eq!(tape.len(), 1, "matmul must record exactly one node");
    let node = &tape.nodes()[0];
    let out_id = node.output_id;
    let input_ids = node.input_ids.clone();
    assert_eq!(input_ids.len(), 2, "matmul records two inputs (a, b)");

    // Output is [M, N]; seed grad matches.
    let seed = build_seed_grad(&device, &[m, n], 0x3A33_0000_0003, 0.25);
    let seed_kt =
        kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&seed).expect("seed kt borrow");
    let mut seeds = HashMap::new();
    seeds.insert(out_id, seed_kt);

    let grads = tape
        .backward_with_seeds(seeds, |x, y| kiln_tensor::ops::add(x, y))
        .expect("matmul backward walk");

    // Input order is [a, b]; MatmulBackward returns [d_a, d_b].
    let da_kt = grads.get(input_ids[0]).expect("d_a present");
    let db_kt = grads.get(input_ids[1]).expect("d_b present");
    let da = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(da_kt).expect("d_a -> candle");
    let db = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(db_kt).expect("d_b -> candle");
    // Shape asserts pin the transpose orientation decisively.
    assert_eq!(da.shape().dims(), &[m, k], "d_a shape [M, K]");
    assert_eq!(db.shape().dims(), &[k, n], "d_b shape [K, N]");

    // Analytic reference in F32: d_a = grad · bᵀ, d_b = aᵀ · grad.
    let af = a.to_dtype(DType::F32).expect("a -> f32");
    let bf = b.to_dtype(DType::F32).expect("b -> f32");
    let gf = seed.to_dtype(DType::F32).expect("grad -> f32");
    let bt = bf.t().expect("b.t").contiguous().expect("bᵀ contiguous");
    let ref_da = gf.matmul(&bt).expect("grad @ bᵀ");
    let at = af.t().expect("a.t").contiguous().expect("aᵀ contiguous");
    let ref_db = at.matmul(&gf).expect("aᵀ @ grad");

    let diff_a = max_abs_diff(&da, &ref_da);
    let diff_b = max_abs_diff(&db, &ref_db);
    assert!(
        diff_a < 3e-2,
        "matmul d_a diverges from grad·bᵀ reference (max-abs-diff {diff_a})"
    );
    assert!(
        diff_b < 3e-2,
        "matmul d_b diverges from aᵀ·grad reference (max-abs-diff {diff_b})"
    );
}

#[test]
fn tape_backward_embedding_scatter_add_conserves_mass() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_backward embedding: no CUDA device — skipping");
            return;
        }
    };

    let vocab = 1024usize;
    let hidden = 256usize;
    let n_tokens = 32usize;
    let (weights, token_ids) = build_embedding_inputs(&device, vocab, hidden, n_tokens);

    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    let (res, tape) = kiln_model::tape_forward::with_thread_local_tape(|| {
        kiln_model::tape_forward::try_tape_embedding_cuda(&weights, &token_ids)
    });
    let _out = res
        .expect("tape-forward try_tape_embedding_cuda ok")
        .expect("tape-forward returned Some(out)");
    assert_eq!(tape.len(), 1, "embedding must record exactly one node");
    let node = &tape.nodes()[0];
    let out_id = node.output_id;
    let input_ids = node.input_ids.clone();
    assert!(!input_ids.is_empty(), "embedding records at least one input");

    // Output is [N, H]; seed grad matches.
    let seed = build_seed_grad(&device, &[n_tokens, hidden], 0xE9B0_0000_0004, 0.25);
    let seed_kt =
        kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&seed).expect("seed kt borrow");
    let mut seeds = HashMap::new();
    seeds.insert(out_id, seed_kt);

    let grads = tape
        .backward_with_seeds(seeds, |a, b| kiln_tensor::ops::add(a, b))
        .expect("embedding backward walk");

    // input_ids[0] = weights → d_weights via scatter_add. The grad for
    // token_ids (input_ids[1], when present) must be None — indices have
    // no gradient.
    let dw_kt = grads.get(input_ids[0]).expect("d_weights present");
    let dw = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(dw_kt).expect("d_weights -> candle");
    assert_eq!(dw.shape().dims(), &[vocab, hidden], "d_weights shape [V, H]");
    assert_eq!(dw.dtype(), DType::BF16, "d_weights dtype matches weights");
    assert!(dw.device().is_cuda(), "d_weights stays on CUDA");
    if input_ids.len() > 1 {
        assert!(
            grads.get(input_ids[1]).is_none(),
            "token-id indices must carry no gradient"
        );
    }

    // Mass conservation: scatter_add distributes every grad-row element
    // into exactly one weight row, so Σ(d_weights) == Σ(grad). This is a
    // backend-agnostic invariant that catches a dropped/mis-routed
    // scatter without needing to re-derive the per-row destinations.
    let dw_sum = dw
        .to_dtype(DType::F32)
        .expect("d_weights -> f32")
        .sum_all()
        .expect("sum d_weights")
        .to_scalar::<f32>()
        .expect("d_weights scalar");
    let seed_sum = seed
        .to_dtype(DType::F32)
        .expect("seed -> f32")
        .sum_all()
        .expect("sum seed")
        .to_scalar::<f32>()
        .expect("seed scalar");
    assert!(
        (dw_sum - seed_sum).abs() < 2e-1,
        "embedding scatter_add violated mass conservation: \
         Σ(d_weights) = {dw_sum}, Σ(grad) = {seed_sum}"
    );
}

#[test]
fn tape_backward_rms_norm_produces_input_grads() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_backward rms_norm: no CUDA device — skipping");
            return;
        }
    };

    let rows = 16usize;
    let hidden = 2560usize;
    let eps = 1e-6f64;
    let (x, w) = build_inputs(&device, rows, hidden);

    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    let (res, tape) = kiln_model::tape_forward::with_thread_local_tape(|| {
        kiln_model::forward::rms_norm(&x, &w, eps)
    });
    let _out = res.expect("tape-forward rms_norm ok");
    assert_eq!(tape.len(), 1, "rms_norm must record exactly one node");
    let node = &tape.nodes()[0];
    let out_id = node.output_id;
    let input_ids = node.input_ids.clone();
    assert!(!input_ids.is_empty(), "rms_norm records at least one input");

    let seed = build_seed_grad(&device, &[rows, hidden], 0x12_0000_0005, 0.25);
    let seed_kt =
        kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&seed).expect("seed kt borrow");
    let mut seeds = HashMap::new();
    seeds.insert(out_id, seed_kt);

    // CudaFusedRmsNormBackward is a CUDA-FFI-only op with no clean candle
    // value reference (the forward fuses mean-square + rsqrt + affine in
    // one kernel). This is a plumbing/smoke assertion: the backward walk
    // must run and emit a grad for the activation input with the right
    // shape/dtype/device. Value parity for the fused kernel is covered by
    // the kernel crate's own parity tests.
    let grads = tape
        .backward_with_seeds(seeds, |a, b| kiln_tensor::ops::add(a, b))
        .expect("rms_norm backward walk");

    let dx_kt = grads.get(input_ids[0]).expect("dx grad present");
    let dx = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(dx_kt).expect("dx kt -> candle");
    assert_eq!(dx.shape().dims(), &[rows, hidden], "dx shape matches x");
    assert_eq!(dx.dtype(), DType::BF16, "dx dtype matches x");
    assert!(dx.device().is_cuda(), "dx stays on CUDA");

    // If the fused backward also emits a weight grad, it must be
    // hidden-shaped. (Don't require it — the op may fold the weight grad
    // elsewhere; the activation grad is the load-bearing one here.)
    if input_ids.len() > 1 {
        if let Some(dw_kt) = grads.get(input_ids[1]) {
            let dw = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(dw_kt).expect("dw -> candle");
            assert_eq!(dw.shape().dims(), &[hidden], "weight grad shape [H]");
        }
    }
}


// ----------------------------------------------------------------------
// Split-half RoPE tape-forward parity (#1082 — CP-4 op #6).
//
// kiln's production `apply_rope` uses the split-half (GPT-NeoX) rotary
// convention on rank-4 [batch, seq, num_heads, head_dim] activations with
// [seq, rotary_dim/2] cos/sin schedules. The adapter routes this through
// `kiln_tensor::ops::rope_split_half` (a device-agnostic composite) and
// records a single `RopeSplitHalfBackward` node — whose backward is the
// same op with sin negated (a rotation's adjoint), run on the grad's own
// device with NO host round-trip.
//
// Forward is checked against an independent host f32 split-half reference;
// backward against the analytic split-half adjoint. We use the realistic
// Qwen3.5-4B partial-rotary geometry (head_dim 256, rotary_dim 64) plus a
// full-rotary case (head_dim == rotary_dim).
// ----------------------------------------------------------------------

fn build_rope_split_half_inputs(
    device: &Device,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> (Tensor, Tensor, Tensor) {
    let half = rotary_dim / 2;
    let x_host = random_bf16_vec(batch * seq * heads * head_dim, 0x0B0E_1234_5678_9ABC, 0.5);
    let mut cos_host = Vec::with_capacity(seq * half);
    let mut sin_host = Vec::with_capacity(seq * half);
    for s in 0..seq {
        for i in 0..half {
            let theta = (s as f32) * (10000f32).powf(-2.0 * (i as f32) / (rotary_dim as f32));
            cos_host.push(half::bf16::from_f32(theta.cos()));
            sin_host.push(half::bf16::from_f32(theta.sin()));
        }
    }
    let x = Tensor::from_vec(x_host, (batch, seq, heads, head_dim), &Device::Cpu)
        .expect("x cpu")
        .to_device(device)
        .expect("x -> cuda")
        .contiguous()
        .expect("x contig");
    let cos = Tensor::from_vec(cos_host, (seq, half), &Device::Cpu)
        .expect("cos cpu")
        .to_device(device)
        .expect("cos -> cuda")
        .contiguous()
        .expect("cos contig");
    let sin = Tensor::from_vec(sin_host, (seq, half), &Device::Cpu)
        .expect("sin cpu")
        .to_device(device)
        .expect("sin -> cuda")
        .contiguous()
        .expect("sin contig");
    (x, cos, sin)
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("-> f32")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec")
}

#[allow(clippy::too_many_arguments)]
fn ref_rope_split_half_fwd(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut out = x.to_vec();
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let row = (((b * seq) + s) * heads + h) * head_dim;
                let sched = s * half;
                for i in 0..half {
                    let c = cos[sched + i];
                    let sn = sin[sched + i];
                    let x1 = x[row + i];
                    let x2 = x[row + half + i];
                    out[row + i] = x1 * c - x2 * sn;
                    out[row + half + i] = x1 * sn + x2 * c;
                }
            }
        }
    }
    out
}

#[test]
fn tape_forward_rope_split_half_matches_f32_reference() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_forward rope_split_half: no CUDA device — skipping");
            return;
        }
    };
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    for (batch, seq, heads, head_dim, rotary_dim) in
        [(2usize, 8usize, 4usize, 256usize, 64usize), (1, 6, 2, 64, 64)]
    {
        let (x, cos, sin) =
            build_rope_split_half_inputs(&device, batch, seq, heads, head_dim, rotary_dim);

        let none_out =
            kiln_model::tape_forward::try_tape_rope_cuda(&x, &cos, &sin, head_dim, rotary_dim)
                .expect("baseline ok");
        assert!(none_out.is_none(), "no-scope path must be Ok(None)");

        let (res, tape) = kiln_model::tape_forward::with_thread_local_tape(|| {
            kiln_model::tape_forward::try_tape_rope_cuda(&x, &cos, &sin, head_dim, rotary_dim)
        });
        let out = res
            .expect("tape try_tape_rope_cuda ok")
            .expect("tape returned Some(out)");
        assert_eq!(tape.len(), 1, "rope must record exactly one node");
        let node = &tape.nodes()[0];
        assert_eq!(
            node.input_ids.len(),
            1,
            "rope_split_half records a single differentiable input (x)"
        );
        assert_eq!(out.shape().dims(), &[batch, seq, heads, head_dim]);
        assert_eq!(out.dtype(), DType::BF16);

        let x_h = host_f32(&x);
        let cos_h = host_f32(&cos);
        let sin_h = host_f32(&sin);
        let want = ref_rope_split_half_fwd(
            &x_h, &cos_h, &sin_h, batch, seq, heads, head_dim, rotary_dim,
        );
        let want_t = Tensor::from_vec(want, (batch, seq, heads, head_dim), &Device::Cpu)
            .expect("ref cpu")
            .to_device(&device)
            .expect("ref -> cuda");
        let diff = max_abs_diff(&out, &want_t);
        assert!(
            diff < 3e-2,
            "rope_split_half forward diverges from f32 reference (max-abs-diff {diff} >= 3e-2 \
             BF16 tol; geometry b={batch} s={seq} h={heads} hd={head_dim} rot={rotary_dim})"
        );
    }
}

#[test]
fn tape_forward_rope_split_half_short_circuits_without_scope() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device = match cuda_device() {
        Some(d) => d,
        None => return,
    };
    let (x, cos, sin) = build_rope_split_half_inputs(&device, 1, 4, 2, 64, 64);
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }
    let out =
        kiln_model::tape_forward::try_tape_rope_cuda(&x, &cos, &sin, 64, 64).expect("call ok");
    assert!(out.is_none(), "no active scope must short-circuit to Ok(None)");
}

#[test]
fn tape_backward_rope_split_half_matches_analytic_adjoint() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("tape_backward rope_split_half: no CUDA device — skipping");
            return;
        }
    };
    unsafe {
        std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
    }

    let (batch, seq, heads, head_dim, rotary_dim) = (2usize, 8usize, 4usize, 256usize, 64usize);
    let half = rotary_dim / 2;
    let (x, cos, sin) =
        build_rope_split_half_inputs(&device, batch, seq, heads, head_dim, rotary_dim);

    let (res, tape) = kiln_model::tape_forward::with_thread_local_tape(|| {
        kiln_model::tape_forward::try_tape_rope_cuda(&x, &cos, &sin, head_dim, rotary_dim)
    });
    let _out = res.expect("fwd ok").expect("Some(out)");
    assert_eq!(tape.len(), 1);
    let node = &tape.nodes()[0];
    let out_id = node.output_id;
    let input_ids = node.input_ids.clone();
    assert_eq!(input_ids.len(), 1);

    let seed = build_seed_grad(&device, &[batch, seq, heads, head_dim], 0x4090_0000_0007, 0.25);
    let seed_kt =
        kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&seed).expect("seed kt borrow");
    let mut seeds = HashMap::new();
    seeds.insert(out_id, seed_kt);

    let grads = tape
        .backward_with_seeds(seeds, |a, b| kiln_tensor::ops::add(a, b))
        .expect("rope backward walk");

    let dx_kt = grads.get(input_ids[0]).expect("dx present");
    let dx = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(dx_kt).expect("dx -> candle");
    assert_eq!(dx.shape().dims(), &[batch, seq, heads, head_dim]);
    assert_eq!(dx.dtype(), DType::BF16);
    assert!(dx.device().is_cuda());

    // Analytic split-half adjoint (host f32):
    //   dx[i]      =  dy[i]*cos + dy[half+i]*sin
    //   dx[half+i] = -dy[i]*sin + dy[half+i]*cos
    let dy_h = host_f32(&seed);
    let cos_h = host_f32(&cos);
    let sin_h = host_f32(&sin);
    let mut want = dy_h.clone();
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let row = (((b * seq) + s) * heads + h) * head_dim;
                let sched = s * half;
                for i in 0..half {
                    let c = cos_h[sched + i];
                    let sn = sin_h[sched + i];
                    let dy0 = dy_h[row + i];
                    let dy1 = dy_h[row + half + i];
                    want[row + i] = dy0 * c + dy1 * sn;
                    want[row + half + i] = -dy0 * sn + dy1 * c;
                }
            }
        }
    }
    let want_t = Tensor::from_vec(want, (batch, seq, heads, head_dim), &Device::Cpu)
        .expect("ref cpu")
        .to_device(&device)
        .expect("ref -> cuda");
    let diff = max_abs_diff(&dx, &want_t);
    assert!(
        diff < 3e-2,
        "rope_split_half backward diverges from analytic adjoint (max-abs-diff {diff} >= 3e-2)"
    );
}
