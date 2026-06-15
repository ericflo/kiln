//! ROCm parity sweep for the fused RMSNorm kernels (Phase R.7).
//!
//! Validates the wave-size-fixed `fused_rmsnorm_kt` (forward) and
//! `fused_rmsnorm_backward_kt` (backward) against a CPU F32 reference across
//! a sweep of last-axis widths that straddle the wavefront boundary
//! ({31,32,33,63,64,65,127,128,129,256,1024}) — the widths most likely to
//! expose a wave32-vs-wave64 reduction bug (lanes 32-63 dropping out, a
//! half-populated final wave, etc.).
//!
//! The orchestrator runs this under BOTH wave32 and wave64 (KILN_ROCM_WAVE64);
//! the kernels reduce via the wave-agnostic shared-memory `kiln_block_reduce_*`
//! helpers, so both must match the reference.
//!
//! Skips cleanly (returns Ok) when no AMD GPU / HIP runtime is present.
#![cfg(feature = "rocm")]

use half::bf16;

use kiln_rmsnorm_kernel::{fused_rmsnorm_backward_kt, fused_rmsnorm_kt};
use kiln_tensor::{DType, Tensor};

/// Last-axis widths straddling the 32/64-lane wavefront boundary.
const HIDDEN_SWEEP: &[usize] = &[31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024];

fn rocm_available() -> bool {
    kiln_tensor::rocm_is_available()
}

/// Deterministic pseudo-random f32 pattern in roughly [-1, 1).
fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..n {
        s = s
            .wrapping_add(0xDEAD_BEEF)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        out.push(((s as u32 % 1024) as f32 - 512.0) / 512.0);
    }
    out
}

fn pattern_bf16(n: usize, seed: u64) -> Vec<bf16> {
    pattern(n, seed).into_iter().map(bf16::from_f32).collect()
}

/// Build a contiguous BF16 ROCm tensor from host data.
fn rocm_bf16(data: &[bf16], shape: Vec<usize>) -> Tensor {
    let cpu = Tensor::from_slice(data, shape).expect("cpu from_slice");
    kiln_tensor::host_to_rocm_copy(&cpu, 0).expect("host_to_rocm_copy")
}

/// Read a ROCm BF16 tensor back to a host f32 Vec.
fn read_bf16_f32(t: &Tensor) -> Vec<f32> {
    assert_eq!(t.dtype(), DType::BF16, "expected BF16 readback");
    let host = kiln_tensor::rocm_to_host_copy(t).expect("rocm_to_host_copy");
    host.to_vec::<bf16>()
        .expect("to_vec bf16")
        .into_iter()
        .map(|b| b.to_f32())
        .collect()
}

/// CPU F32 reference for Qwen3.5-style RMSNorm:
///   out[r,j] = (1 + w[j]) * x[r,j] * rsqrt(mean_j(x[r,:]^2) + eps)
/// computed in f32 over BF16-rounded inputs (matches the kernel: it reads BF16,
/// accumulates in F32, writes BF16).
fn rmsnorm_ref(x: &[bf16], w: &[bf16], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut sum_sq = 0.0f32;
        for &v in row {
            let f = v.to_f32();
            sum_sq += f * f;
        }
        let rms_inv = (sum_sq / hidden as f32 + eps).sqrt().recip();
        for j in 0..hidden {
            let xj = row[j].to_f32();
            let wj = w[j].to_f32();
            out[r * hidden + j] = (1.0 + wj) * xj * rms_inv;
        }
    }
    out
}

/// Compare two f32 vectors with a combined abs+rel tolerance suited to BF16
/// round-trip (the kernel output is BF16, ~3 decimal digits).
fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    let mut max_abs = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = 3e-2 * w.abs().max(1.0) + 3e-2;
        if diff > tol {
            panic!("{ctx}: element {i} got {g} want {w} (|diff| {diff} > tol {tol})");
        }
        max_abs = max_abs.max(diff);
    }
    eprintln!("{ctx}: OK (max |diff| = {max_abs})");
}

#[test]
fn rocm_fused_rmsnorm_forward_parity_sweep() {
    if !rocm_available() {
        eprintln!("ROCm not available; skipping rocm_fused_rmsnorm_forward_parity_sweep");
        return;
    }

    let rows = 5usize;
    let eps = 1e-6f32;

    for &hidden in HIDDEN_SWEEP {
        let x_host = pattern_bf16(rows * hidden, 0x1000 + hidden as u64);
        let w_host = pattern_bf16(hidden, 0x2000 + hidden as u64);

        let x = rocm_bf16(&x_host, vec![rows, hidden]);
        let w = rocm_bf16(&w_host, vec![hidden]);

        let out = fused_rmsnorm_kt(&x, &w, eps).expect("fused_rmsnorm_kt");
        assert_eq!(out.shape(), &[rows, hidden]);
        assert_eq!(out.dtype(), DType::BF16);

        let got = read_bf16_f32(&out);
        let want = rmsnorm_ref(&x_host, &w_host, rows, hidden, eps);
        assert_close(&got, &want, &format!("forward hidden={hidden}"));
    }
}

#[test]
fn rocm_fused_rmsnorm_long_context_shape_is_finite() {
    if std::env::var("KILN_RUN_LONG_ROCM_RMSNORM").is_err() {
        eprintln!("set KILN_RUN_LONG_ROCM_RMSNORM=1 to run the long-context RMSNorm regression");
        return;
    }
    if !rocm_available() {
        eprintln!("ROCm not available; skipping rocm_fused_rmsnorm_long_context_shape_is_finite");
        return;
    }

    let rows = 40_314usize;
    let hidden = 2_560usize;
    let eps = 1e-6f32;
    let x_host = pattern_bf16(rows * hidden, 0x6A09_E667);
    let w_host = pattern_bf16(hidden, 0xBB67_AE85);
    let x = rocm_bf16(&x_host, vec![1, rows, hidden]);
    let w = rocm_bf16(&w_host, vec![hidden]);

    let out = fused_rmsnorm_kt(&x, &w, eps).expect("long-context fused_rmsnorm_kt");
    assert_eq!(out.shape(), &[1, rows, hidden]);
    assert_eq!(out.dtype(), DType::BF16);

    let got = read_bf16_f32(&out);
    let bad = got
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
        .map(|(i, v)| (i, *v));
    assert!(bad.is_none(), "long-context RMSNorm non-finite at {bad:?}");

    for &row in &[0usize, 1_411, 4_095, 4_096, 25_987, rows - 1] {
        let row_start = row * hidden;
        let want_row = rmsnorm_ref(
            &x_host[row_start..row_start + hidden],
            &w_host,
            1,
            hidden,
            eps,
        );
        assert_close(
            &got[row_start..row_start + hidden],
            &want_row,
            &format!("long-context forward row={row}"),
        );
    }
}

#[test]
fn rocm_fused_rmsnorm_backward_parity_sweep() {
    if !rocm_available() {
        eprintln!("ROCm not available; skipping rocm_fused_rmsnorm_backward_parity_sweep");
        return;
    }

    let rows = 4usize;
    let eps = 1e-6f32;

    for &hidden in HIDDEN_SWEEP {
        let x_host = pattern_bf16(rows * hidden, 0x3000 + hidden as u64);
        let w_host = pattern_bf16(hidden, 0x4000 + hidden as u64);
        let dy_host = pattern_bf16(rows * hidden, 0x5000 + hidden as u64);

        let x = rocm_bf16(&x_host, vec![rows, hidden]);
        let w = rocm_bf16(&w_host, vec![hidden]);
        let dy = rocm_bf16(&dy_host, vec![rows, hidden]);

        let (grad_x, grad_w_partial) =
            fused_rmsnorm_backward_kt(&x, &w, &dy, eps).expect("fused_rmsnorm_backward_kt");
        assert_eq!(grad_x.shape(), &[rows, hidden]);
        assert_eq!(grad_x.dtype(), DType::BF16);
        // grad_w_partial is F32 [rows, hidden]; the cross-row-reduced grad_w
        // lives in its first `hidden` slots (see csrc/fused_rmsnorm_bwd.cu).
        assert_eq!(grad_w_partial.shape(), &[rows, hidden]);
        assert_eq!(grad_w_partial.dtype(), DType::F32);

        let got_gx = read_bf16_f32(&grad_x);

        // CPU F32 reference for grad_x and grad_w, mirroring the kernel math:
        //   rms_inv = rsqrt(mean(x^2) + eps)
        //   c       = (1/H) * rms_inv^2 * sum_j((1+w_j) x_ij g_ij)
        //   grad_x  = rms_inv * ((1+w) g - x c)
        //   grad_w  = sum_i x_ij rms_inv_i g_ij
        let (want_gx, want_gw) = rmsnorm_bwd_ref(&x_host, &w_host, &dy_host, rows, hidden, eps);
        assert_close(
            &got_gx,
            &want_gx,
            &format!("backward grad_x hidden={hidden}"),
        );

        // grad_w lives in the first `hidden` F32 slots of the partial buffer.
        let gw_partial_host = kiln_tensor::rocm_to_host_copy(&grad_w_partial)
            .expect("rocm_to_host_copy grad_w_partial");
        let gw_partial = gw_partial_host.to_vec::<f32>().expect("to_vec f32");
        let got_gw = &gw_partial[..hidden];
        assert_close(
            got_gw,
            &want_gw,
            &format!("backward grad_w hidden={hidden}"),
        );
    }
}

/// CPU F32 reference for the RMSNorm backward. Returns `(grad_x, grad_w)`.
fn rmsnorm_bwd_ref(
    x: &[bf16],
    w: &[bf16],
    dy: &[bf16],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut grad_x = vec![0.0f32; rows * hidden];
    let mut grad_w = vec![0.0f32; hidden];
    for r in 0..rows {
        let xr = &x[r * hidden..(r + 1) * hidden];
        let gr = &dy[r * hidden..(r + 1) * hidden];

        let mut sum_sq = 0.0f32;
        for &v in xr {
            let f = v.to_f32();
            sum_sq += f * f;
        }
        let rms_inv = (sum_sq / hidden as f32 + eps).sqrt().recip();

        let mut sum_xgw = 0.0f32;
        for j in 0..hidden {
            let xj = xr[j].to_f32();
            let wj = w[j].to_f32();
            let gj = gr[j].to_f32();
            sum_xgw += (1.0 + wj) * xj * gj;
        }
        let c = sum_xgw / hidden as f32 * rms_inv * rms_inv;

        for j in 0..hidden {
            let xj = xr[j].to_f32();
            let wj = w[j].to_f32();
            let gj = gr[j].to_f32();
            // grad_x written as BF16 by the kernel, so compare in BF16 precision.
            let dx = rms_inv * ((1.0 + wj) * gj - xj * c);
            grad_x[r * hidden + j] = bf16::from_f32(dx).to_f32();
            grad_w[j] += xj * rms_inv * gj;
        }
    }
    (grad_x, grad_w)
}
