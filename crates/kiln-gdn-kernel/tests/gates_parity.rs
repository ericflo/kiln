//! Parity test for the fused GDN gates CUDA kernel.
//!
//! The kernel in `kiln_gdn_kernel::gdn_gates` replaces the 8-op candle
//! chain in `kiln-model::forward::gated_deltanet_forward` Step 6:
//!
//!   beta = sigmoid(b)                                  // bf16
//!   g    = -exp(A_log) * softplus(a + dt_bias)         // bf16
//!
//! This test constructs a small `[B, T, nv]` workload, runs the fused
//! kernel, runs the exact candle-op reference path side-by-side, and
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

use candle_core::{DType, Device, Tensor};
use kiln_gdn_kernel::{gdn_gates, gdn_gates_supports};

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
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference path: the exact candle chain used today in
/// `gated_deltanet_forward` Step 6. We reimplement the F32 recipe here
/// so the test isn't tautological — this is the *algorithmic* oracle,
/// not a second copy of the kernel.
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

fn run_case(device: &Device, b: usize, t: usize, nv: usize, seed: u64, label: &str) {
    let rows = b * t;

    // Host tensors. Activations near N(0, 0.25); per-head params near
    // N(0, 0.1) so softplus / exp don't blow up.
    let a_host = fill(seed ^ 0xA1, rows * nv, 0.5);
    let b_host = fill(seed ^ 0xB2, rows * nv, 0.5);
    let a_log_host = fill(seed ^ 0xA_106, nv, 0.2);
    let dt_bias_host = fill(seed ^ 0xDE_B1A5, nv, 0.2);

    // Build CUDA bf16 tensors the kernel expects.
    let a_cpu = Tensor::from_vec(a_host.clone(), (b, t, nv), &Device::Cpu).unwrap();
    let a = a_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let b_cpu = Tensor::from_vec(b_host.clone(), (b, t, nv), &Device::Cpu).unwrap();
    let b_in = b_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let a_log_cpu = Tensor::from_vec(a_log_host.clone(), (nv,), &Device::Cpu).unwrap();
    let a_log = a_log_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let dt_bias_cpu = Tensor::from_vec(dt_bias_host.clone(), (nv,), &Device::Cpu).unwrap();
    let dt_bias = dt_bias_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();

    assert!(
        gdn_gates_supports(&a, &b_in, &a_log, &dt_bias),
        "{label}: envelope check failed"
    );

    let (beta, g) = gdn_gates(&a, &b_in, &a_log, &dt_bias).expect("fused gates kernel");

    // Pull back to host F32 for comparison.
    let beta_host: Vec<f32> = beta
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let g_host: Vec<f32> = g
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let (beta_ref, g_ref) =
        reference_host(&a_host, &b_host, &a_log_host, &dt_bias_host, rows, nv);

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
fn gdn_gates_parity_vs_candle_reference() {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping gdn_gates parity test: no CUDA device ({e})");
            return;
        }
    };

    // Decode-shape (B=1, T=1) and prefill-shape (B=1, T=32) across a
    // couple of head counts inside the envelope. Qwen3.5-4B GDN layers
    // use nv in the 32-128 range, so we exercise both ends.
    run_case(&device, 1, 1, 32, 0xDEAD_BEEF, "decode/nv=32");
    run_case(&device, 1, 1, 128, 0xCAFE_F00D, "decode/nv=128");
    run_case(&device, 2, 32, 64, 0xFACE_0FF, "prefill/B=2,T=32,nv=64");
    run_case(&device, 1, 128, 128, 0x5EED_BEEF, "prefill/T=128,nv=128");
}
