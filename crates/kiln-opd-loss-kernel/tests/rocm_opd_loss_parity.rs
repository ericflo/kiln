//! Phase R.7 — wave-size correctness + parity test for the fused OPD
//! top-K reverse-KL BACKWARD kernel (`csrc/opd_topk_kl.cu`).
//!
//! The kernel's per-logit dot product reduces over the hidden dimension `H`,
//! and the renormalisation (log-softmax over the K support) is a length-K
//! block reduction. Both were 32-lane `__shfl_xor_sync` butterflies that are
//! BROKEN on AMD wave64 (HIP static_asserts the mask; even native shuffles
//! mangle cross-32-lane offsets). They are now wave-agnostic shared-memory
//! block reductions. A wave64 bug compiles cleanly and only manifests
//! numerically, so this sweeps `H` across the 32/64-lane wavefront boundary
//! widths {31,32,33,63,64,65,127,128,129,256,1024} and compares the ROCm
//! FFI backward against the device-agnostic analytic kt-composite run on CPU
//! (itself finite-difference validated in the crate's unit tests).
//!
//! Run wave32: `cargo test -p kiln-opd-loss-kernel --features rocm --test rocm_opd_loss_parity`
//! Run wave64: rebuild with `KILN_ROCM_WAVE64=1` set so hipcc emits
//! `-mwavefrontsize64`.
#![cfg(feature = "rocm")]

use kiln_opd_loss_kernel::kt_api::{
    OpdLossOutputKt, opd_top_k_reverse_kl_phase_b_bwd_composite_kt,
    opd_top_k_reverse_kl_phase_b_bwd_kt,
};
use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.7 OPD-loss backward parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-1, 1) from a u64 seed-stream.
fn next_val(s: &mut u64) -> f32 {
    *s = s
        .wrapping_add(0xDEAD_BEEF)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    ((*s as u32 % 2048) as f32 - 1024.0) / 1024.0
}

/// Build the (hidden, head_t, indices, logprobs, label_mask, grad) test
/// fixture deterministically for a given (seq_len, hidden_size, vocab_size,
/// top_k) and seed. Active positions alternate so the scatter path is
/// exercised (some rows zero).
struct Fixture {
    hidden: Vec<f32>,
    head_t: Vec<f32>,
    indices: Vec<u32>,
    logprobs: Vec<f32>,
    label_mask: Vec<bool>,
    grad_per_pos: Vec<f32>,
    grad_scalar: Vec<f32>,
    active_count: usize,
}

fn build_fixture(
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    top_k: usize,
    seed: u64,
) -> Fixture {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);

    let hidden: Vec<f32> = (0..seq_len * hidden_size)
        .map(|_| next_val(&mut s))
        .collect();
    let head_t: Vec<f32> = (0..hidden_size * vocab_size)
        .map(|_| next_val(&mut s))
        .collect();

    let label_mask: Vec<bool> = (0..seq_len).map(|i| i % 2 == 0).collect();
    let active_count = label_mask.iter().filter(|m| **m).count();

    // K unique indices < vocab_size per active row.
    let mut indices = Vec::with_capacity(active_count * top_k);
    let mut logprobs = Vec::with_capacity(active_count * top_k);
    for row in 0..active_count {
        let mut row_idx: Vec<u32> = Vec::with_capacity(top_k);
        let mut k = (row as u32).wrapping_mul(7);
        while row_idx.len() < top_k {
            let idx = k % (vocab_size as u32);
            if !row_idx.contains(&idx) {
                row_idx.push(idx);
            }
            k = k.wrapping_add(1);
        }
        indices.extend(row_idx);
        for _ in 0..top_k {
            // Teacher logprobs: any reals; the renorm inside fwd/bwd handles
            // normalisation.
            logprobs.push(-(next_val(&mut s).abs()) * 4.0);
        }
    }

    let grad_per_pos: Vec<f32> = (0..active_count).map(|_| next_val(&mut s)).collect();
    let grad_scalar = vec![next_val(&mut s)];

    Fixture {
        hidden,
        head_t,
        indices,
        logprobs,
        label_mask,
        grad_per_pos,
        grad_scalar,
        active_count,
    }
}

/// Read any tensor (CPU or ROCm) back to a host `Vec<f32>`.
fn host_f32(t: &Tensor) -> Vec<f32> {
    let cpu = match t.device() {
        Device::Cpu => t.clone(),
        Device::Rocm(_) => kiln_tensor::rocm_to_host_copy(t).expect("rocm_to_host_copy"),
        other => panic!("host_f32: unexpected device {other}"),
    };
    cpu.to_vec::<f32>().expect("to_vec f32")
}

/// One parity check: run the ROCm FFI backward and the CPU analytic composite
/// on identical data and assert d_hidden agreement.
fn check_parity(
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    top_k: usize,
    mode: OpdLossOutputKt,
    seed: u64,
) {
    let fx = build_fixture(seq_len, hidden_size, vocab_size, top_k, seed);

    let (grad_vals, grad_shape): (&[f32], Vec<usize>) = match mode {
        OpdLossOutputKt::ScalarMean => (&fx.grad_scalar, vec![1]),
        OpdLossOutputKt::PerPosition => (&fx.grad_per_pos, vec![fx.active_count]),
    };

    // ---- CPU reference via the analytic kt-composite (FD-validated). ----
    let h_cpu = Tensor::from_vec_on(
        Device::Cpu,
        fx.hidden.clone(),
        vec![1, seq_len, hidden_size],
    )
    .expect("cpu hidden");
    let w_cpu = Tensor::from_vec_on(
        Device::Cpu,
        fx.head_t.clone(),
        vec![hidden_size, vocab_size],
    )
    .expect("cpu head_t");
    let g_cpu =
        Tensor::from_vec_on(Device::Cpu, grad_vals.to_vec(), grad_shape.clone()).expect("cpu grad");
    let ref_d_hidden = opd_top_k_reverse_kl_phase_b_bwd_composite_kt(
        &h_cpu,
        &w_cpu,
        &fx.indices,
        &fx.logprobs,
        &fx.label_mask,
        &g_cpu,
        top_k,
        mode,
    )
    .expect("cpu composite backward");
    let reference = host_f32(&ref_d_hidden);

    // ---- ROCm FFI backward. ----
    let h_rocm = Tensor::from_vec_on(
        Device::Rocm(0),
        fx.hidden.clone(),
        vec![1, seq_len, hidden_size],
    )
    .expect("rocm hidden");
    let w_rocm = Tensor::from_vec_on(
        Device::Rocm(0),
        fx.head_t.clone(),
        vec![hidden_size, vocab_size],
    )
    .expect("rocm head_t");
    let g_rocm =
        Tensor::from_vec_on(Device::Rocm(0), grad_vals.to_vec(), grad_shape).expect("rocm grad");
    let rocm_d_hidden = opd_top_k_reverse_kl_phase_b_bwd_kt(
        &h_rocm,
        &w_rocm,
        &fx.indices,
        &fx.logprobs,
        &fx.label_mask,
        &g_rocm,
        top_k,
        mode,
    )
    .unwrap_or_else(|e| panic!("rocm backward (H={hidden_size}, K={top_k}): {e}"));
    let got = host_f32(&rocm_d_hidden);

    assert_eq!(
        got.len(),
        reference.len(),
        "shape mismatch (H={hidden_size}, K={top_k})"
    );
    assert_eq!(
        rocm_d_hidden.shape(),
        &[1, seq_len, hidden_size],
        "d_hidden shape (H={hidden_size}, K={top_k})"
    );

    let mut max_abs = 0.0f32;
    for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
        let diff = (g - r).abs();
        let tol = 1e-3 + 3e-3 * r.abs();
        max_abs = max_abs.max(diff);
        assert!(
            diff <= tol,
            "d_hidden mismatch at elem {i} (H={hidden_size}, K={top_k}, mode={mode:?}): \
             got {g} ref {r} diff {diff} tol {tol} \
             (a wave64 reduction bug shows up exactly here)",
        );
    }
    eprintln!(
        "OPD bwd parity OK: H={hidden_size} K={top_k} mode={mode:?} max_abs_diff={max_abs:.2e}"
    );
}

/// Sweep the dot-product reduction width (hidden_size H) across the 32/64-lane
/// wavefront boundary widths for both K ∈ {16, 32} and both output modes.
#[test]
fn opd_backward_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    // The kernel reduces over H — these straddle the 32/64-lane boundary.
    let hidden_widths = [31usize, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024];
    let seq_len = 6usize; // 3 active positions (alternating mask)
    let vocab_size = 257usize; // odd, > K, exercises the gather

    let mut seed = 0xA17C_u64;
    for &h in &hidden_widths {
        for &k in &[16usize, 32] {
            for &mode in &[OpdLossOutputKt::PerPosition, OpdLossOutputKt::ScalarMean] {
                seed = seed.wrapping_add(1);
                check_parity(seq_len, h, vocab_size, k, mode, seed);
            }
        }
    }
    eprintln!(
        "OPD top-K reverse-KL backward CPU-vs-ROCm parity passed across \
         hidden widths {hidden_widths:?} x K{{16,32}} x {{PerPosition,ScalarMean}}"
    );
}

/// A larger active-token count + wide H so the K-support length-K block
/// reductions and the scatter path both run across the 64-lane boundary.
#[test]
fn opd_backward_parity_many_active_positions() {
    if no_rocm() {
        return;
    }
    // seq_len 16 -> 8 active rows; H=129 strides the reduction loop.
    check_parity(16, 129, 512, 32, OpdLossOutputKt::PerPosition, 0xBEEF);
    check_parity(16, 129, 512, 16, OpdLossOutputKt::ScalarMean, 0xCAFE);
    eprintln!("OPD backward parity passed for many-active-position scatter case");
}
