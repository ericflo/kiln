//! Parity test for the fused GDN gates CUDA kernel.
//!
//! The kernel in `kiln_gdn_kernel::gdn_gates_bf16_kt` replaces the 8-op
//! candle chain in `kiln-model::forward::gated_deltanet_forward` Step 6:
//!
//!   beta = sigmoid(b)                                  // bf16
//!   g    = -exp(A_log) * softplus(a + dt_bias)         // bf16
//!
//! This test constructs a small `[B, T, nv]` workload, runs the fused
//! kernel, runs a pure-Rust F32 host reference path side-by-side, and
//! asserts element-wise closeness within a 1e-2 absolute tolerance in
//! bf16 (the same budget `marlin_w4a16_gemm` uses).
//!
//! Mirrors the algorithmic oracle `naive_gdn_gate` in
//! `fla/ops/gated_delta_rule/gate.py`, which computes
//! `-A_log.float().exp() * F.softplus(g + dt_bias)` in F32 and casts at
//! the end. We do the same in the reference path so the test is
//! independent of the kernel internals.
//!
//! Gracefully no-ops on non-CUDA hosts so that `cargo test` on a
//! CPU-only dev box doesn't fail.
//!
//! Phase 7 candle-removal (#1082): migrated off candle Tensors to the
//! kt-typed surface (`gdn_gates_bf16_kt` / `gdn_gates_supports_kt`).
//! Inputs are constructed via `Tensor::cuda_from_slice`; outputs are
//! pulled back via `cuda_to_host_copy`.
//!
//! CUDA-only: `Tensor::cuda_from_slice` / `cuda_to_host_copy` /
//! `primary_cuda_context` are the cuda-storage substrate helpers and
//! don't exist on the ROCm build. The backend-neutral ROCm parity
//! coverage lives in `rocm_gdn_parity.rs` (gated on `feature = "rocm"`).
#![cfg(feature = "cuda")]

use half::bf16;

use kiln_gdn_kernel::{gdn_gates_bf16_kt, gdn_gates_supports_kt};
use kiln_tensor::{CpuStorage, DType, Tensor, cuda_to_host_copy};

fn cuda_available() -> bool {
    kiln_tensor::primary_cuda_context(0).is_ok()
}

fn lcg_seed(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fff_ffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg_seed(&mut s) * scale).collect()
}

fn softplus_f32(x: f32) -> f32 {
    // Match torch.nn.functional.softplus default threshold (20.0), same
    // as the kernel's `stable_softplus` and FLA's reference.
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference path: the exact F32 recipe described in the kernel docs.
/// We reimplement it here in pure Rust so the test isn't tautological —
/// this is the *algorithmic* oracle, not a second copy of the kernel.
fn reference_host(
    a_host: &[f32],
    b_host: &[f32],
    a_log_host: &[f32],
    dt_bias_host: &[f32],
    rows: usize,
    nv: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut beta = vec![0.0f32; rows * nv];
    let mut g = vec![0.0f32; rows * nv];
    for r in 0..rows {
        for h in 0..nv {
            let idx = r * nv + h;
            beta[idx] = sigmoid_f32(b_host[idx]);
            let a_biased = a_host[idx] + dt_bias_host[h];
            let sp = softplus_f32(a_biased);
            let neg_decay = -(a_log_host[h].exp());
            g[idx] = sp * neg_decay;
        }
    }
    (beta, g)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Convert host F32 values into BF16 for kernel ingest.
fn to_bf16_vec(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|&v| bf16::from_f32(v)).collect()
}

/// Read a CUDA-resident BF16 kt-Tensor back to host F32.
fn read_bf16_host_as_f32(t: &Tensor) -> Vec<f32> {
    let host = cuda_to_host_copy(t).expect("cuda → host copy");
    assert_eq!(host.dtype(), DType::BF16);
    let cpu = host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .expect("CpuStorage");
    let bytes = cpu.as_bytes();
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
        out.push(v);
    }
    out
}

fn run_case(b: usize, t: usize, nv: usize, seed: u64, label: &str) {
    let rows = b * t;

    // Host tensors. Activations near N(0, 0.25); per-head params near
    // N(0, 0.1) so softplus / exp don't blow up.
    let a_host = fill(seed ^ 0xA1, rows * nv, 0.5);
    let b_host = fill(seed ^ 0xB2, rows * nv, 0.5);
    let a_log_host = fill(seed ^ 0xA_106, nv, 0.2);
    let dt_bias_host = fill(seed ^ 0xDE_B1A5, nv, 0.2);

    // Convert host F32 → BF16 and upload to CUDA via the kt substrate
    // helper (`Tensor::cuda_from_slice`, #1082 candle-free constructor).
    let a_bf16 = to_bf16_vec(&a_host);
    let b_bf16 = to_bf16_vec(&b_host);
    let a_log_bf16 = to_bf16_vec(&a_log_host);
    let dt_bias_bf16 = to_bf16_vec(&dt_bias_host);

    let a = Tensor::cuda_from_slice(&a_bf16, vec![b, t, nv], 0).expect("upload a");
    let b_in = Tensor::cuda_from_slice(&b_bf16, vec![b, t, nv], 0).expect("upload b");
    let a_log = Tensor::cuda_from_slice(&a_log_bf16, vec![nv], 0).expect("upload a_log");
    let dt_bias = Tensor::cuda_from_slice(&dt_bias_bf16, vec![nv], 0).expect("upload dt_bias");

    assert!(
        gdn_gates_supports_kt(&a, &b_in, &a_log, &dt_bias),
        "{label}: envelope check failed"
    );

    let (beta, g) = gdn_gates_bf16_kt(&a, &b_in, &a_log, &dt_bias).expect("fused gates kernel");

    // Pull back to host F32 for comparison.
    let beta_host: Vec<f32> = read_bf16_host_as_f32(&beta);
    let g_host: Vec<f32> = read_bf16_host_as_f32(&g);

    let (beta_ref, g_ref) = reference_host(&a_host, &b_host, &a_log_host, &dt_bias_host, rows, nv);

    let beta_err = max_abs_diff(&beta_host, &beta_ref);
    let g_err = max_abs_diff(&g_host, &g_ref);

    // bf16 has ~3e-3 relative resolution; 1e-2 absolute is the same
    // budget the marlin parity test uses and leaves room for the final
    // F32 -> bf16 round-trip on both sides.
    let tol = 1e-2f32;
    println!(
        "[{label}] shape=[{b},{t},{nv}] rows={rows} beta_max_abs={beta_err:.3e} g_max_abs={g_err:.3e}"
    );
    assert!(
        beta_err < tol,
        "{label}: beta max_abs_diff {beta_err} >= {tol}"
    );
    assert!(g_err < tol, "{label}: g max_abs_diff {g_err} >= {tol}");
}

#[test]
fn gdn_gates_parity_vs_host_reference() {
    if !cuda_available() {
        eprintln!("Skipping gdn_gates parity test: no CUDA device");
        return;
    }

    // Decode-shape (B=1, T=1) and prefill-shape (B=1, T=32) across a
    // couple of head counts inside the envelope. Qwen3.5-4B GDN layers
    // use nv in the 32-128 range, so we exercise both ends.
    run_case(1, 1, 32, 0xDEAD_BEEF, "decode/nv=32");
    run_case(1, 1, 128, 0xCAFE_F00D, "decode/nv=128");
    run_case(2, 32, 64, 0x0FAC_E0FF, "prefill/B=2,T=32,nv=64");
    run_case(1, 128, 128, 0x5EED_BEEF, "prefill/T=128,nv=128");
}
