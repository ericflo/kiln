//! ROCm parity tests for the GDN kernels (Phase R.7).
//!
//! Validates two of the hipcc-compiled GDN kernels against CPU F32 references:
//!
//!   1. `gdn_forward_substitution_kt` — the namesake fused chunk forward-sub
//!      kernel (`csrc/gdn_fwd_sub.cu`). Per-thread, no cross-lane reduction;
//!      validates the baseline ROCm device-pointer seam end-to-end.
//!
//!   2. `gdn_gated_rms_norm_bf16_kt` — the fused gated RMSNorm
//!      (`csrc/gdn_gated_rms_norm.cu`). Its per-row last-axis reduction uses a
//!      WAVE-AGNOSTIC shared-memory tree (`__shared__ float scratch[kHidden]` +
//!      a `stride >>= 1` loop), so it is correct on wave32 AND wave64; this test
//!      is the on-hardware proof. The GDN reduction kernels are fixed at
//!      hidden==128 (a single 128-thread block per row), so rather than the
//!      generic last-axis width sweep we sweep the row count (the dimension that
//!      actually varies) and pin hidden==128. The reduction-fixed decode path
//!      (`recurrent_gdn_fwd.cu`'s `block_reduce_sum_128`, converted to
//!      `kiln_block_reduce_sum`) is likewise a fixed-128 block reduction; the
//!      gated-RMSNorm here exercises the same wave-agnostic shared-memory tree
//!      pattern that fix relies on.
//!
//! The orchestrator runs this under BOTH wave32 and wave64 (KILN_ROCM_WAVE64);
//! both must match the CPU reference. Skips cleanly when no AMD GPU / HIP
//! runtime is present.
#![cfg(feature = "rocm")]

use half::bf16;

use kiln_gdn_kernel::{
    gdn_forward_substitution_f32_kt, gdn_forward_substitution_kt, gdn_gated_rms_norm_bf16_kt,
    gdn_gated_rms_norm_supports_kt, gdn_solve_tri_transpose_f32_kt,
};
use kiln_tensor::{DType, Tensor};

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

/// Build a contiguous F32 ROCm tensor from host data.
fn rocm_f32(data: &[f32], shape: Vec<usize>) -> Tensor {
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

fn read_f32(t: &Tensor) -> Vec<f32> {
    assert_eq!(t.dtype(), DType::F32, "expected F32 readback");
    let host = kiln_tensor::rocm_to_host_copy(t).expect("rocm_to_host_copy");
    host.to_vec::<f32>().expect("to_vec f32")
}

/// Combined abs+rel tolerance suited to BF16 round-trip (~3 decimal digits) plus
/// a sequential accumulation chain (forward-substitution feeds back its own
/// outputs, so error grows with chunk_size).
fn assert_close(got: &[f32], want: &[f32], abs_extra: f32, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    let mut max_abs = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = 5e-2 * w.abs().max(1.0) + abs_extra;
        if diff > tol {
            panic!("{ctx}: element {i} got {g} want {w} (|diff| {diff} > tol {tol})");
        }
        max_abs = max_abs.max(diff);
    }
    eprintln!("{ctx}: OK (max |diff| = {max_abs})");
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// ---------------------------------------------------------------------------
// 1) gdn_forward_substitution_kt
// ---------------------------------------------------------------------------

/// CPU F32 forward-substitution reference, BF16-rounding intermediate W rows the
/// way the kernel does (it writes each W row to shared memory as BF16 before the
/// next row reads it back).
///   W[t, d] = beta[t] * (V_prime[t, d] - sum_{i<t} A_strict[t, i] * W[i, d])
fn fwd_sub_ref(
    a_strict: &[bf16], // [B*H, C, C]
    v_prime: &[bf16],  // [B*H, C, dv]
    beta: &[bf16],     // [B*H, C]
    bh: usize,
    c: usize,
    dv: usize,
) -> Vec<f32> {
    let mut w_out = vec![0.0f32; bh * c * dv];
    for blk in 0..bh {
        let a_base = blk * c * c;
        let v_base = blk * c * dv;
        let b_base = blk * c;
        let w_base = blk * c * dv;
        // BF16-rounded W rows used as the feedback term (matches the kernel's
        // shared-memory BF16 W store).
        let mut w_bf = vec![0.0f32; c * dv];
        for t in 0..c {
            let beta_t = beta[b_base + t].to_f32();
            for d in 0..dv {
                let mut acc = 0.0f32;
                for i in 0..t {
                    let a = a_strict[a_base + t * c + i].to_f32();
                    acc += a * w_bf[i * dv + d];
                }
                let vp = v_prime[v_base + t * dv + d].to_f32();
                let w_val = beta_t * (vp - acc);
                let w_rounded = bf16::from_f32(w_val).to_f32();
                w_bf[t * dv + d] = w_rounded;
                w_out[w_base + t * dv + d] = w_rounded;
            }
        }
    }
    w_out
}

fn run_fwd_sub(batch: usize, heads: usize, c: usize, dv: usize, seed: u64) {
    let bh = batch * heads;
    let a_host = pattern_bf16(bh * c * c, seed ^ 0xA);
    let v_host = pattern_bf16(bh * c * dv, seed ^ 0xB);
    // beta in (0, 1) like a sigmoid gate.
    let beta_host: Vec<bf16> = pattern(bh * c, seed ^ 0xC)
        .into_iter()
        .map(|v| bf16::from_f32(sigmoid(v)))
        .collect();

    let a = rocm_bf16(&a_host, vec![batch, heads, c, c]);
    let v = rocm_bf16(&v_host, vec![batch, heads, c, dv]);
    let beta = rocm_bf16(&beta_host, vec![batch, heads, c]);

    let w = gdn_forward_substitution_kt(&a, &v, &beta).expect("gdn_forward_substitution_kt");
    assert_eq!(w.shape(), &[batch, heads, c, dv]);
    assert_eq!(w.dtype(), DType::BF16);

    let got = read_bf16_f32(&w);
    let want = fwd_sub_ref(&a_host, &v_host, &beta_host, bh, c, dv);
    // The feedback chain accumulates error across `c` sequential rows; widen the
    // absolute floor accordingly.
    assert_close(
        &got,
        &want,
        2e-2 + 1e-3 * c as f32,
        &format!("fwd_sub bh={bh} c={c} dv={dv}"),
    );
}

#[test]
fn rocm_gdn_forward_substitution_parity() {
    if !rocm_available() {
        eprintln!("ROCm not available; skipping rocm_gdn_forward_substitution_parity");
        return;
    }
    // Production envelope: c<=128, dv<=1024, dv==blockDim.x. Sweep a few shapes.
    run_fwd_sub(1, 1, 16, 128, 0x1111);
    run_fwd_sub(2, 4, 32, 128, 0x2222);
    run_fwd_sub(1, 8, 64, 128, 0x3333);
}

fn fwd_sub_ref_f32(
    a_strict: &[f32],
    v_prime: &[f32],
    beta: &[f32],
    bh: usize,
    c: usize,
    dv: usize,
) -> Vec<f32> {
    let mut w_out = vec![0.0f32; bh * c * dv];
    for blk in 0..bh {
        let a_base = blk * c * c;
        let v_base = blk * c * dv;
        let b_base = blk * c;
        let w_base = blk * c * dv;
        for t in 0..c {
            let beta_t = beta[b_base + t];
            for d in 0..dv {
                let mut acc = 0.0f32;
                for i in 0..t {
                    acc += a_strict[a_base + t * c + i] * w_out[w_base + i * dv + d];
                }
                w_out[w_base + t * dv + d] = beta_t * (v_prime[v_base + t * dv + d] - acc);
            }
        }
    }
    w_out
}

fn solve_tri_transpose_ref_f32(
    a_strict: &[f32],
    beta: &[f32],
    dw: &[f32],
    bh: usize,
    c: usize,
    dv: usize,
) -> Vec<f32> {
    let mut dr = vec![0.0f32; bh * c * dv];
    for blk in 0..bh {
        let a_base = blk * c * c;
        let b_base = blk * c;
        let v_base = blk * c * dv;
        for t in (0..c).rev() {
            for d in 0..dv {
                let mut acc = 0.0f32;
                for i in (t + 1)..c {
                    acc +=
                        beta[b_base + i] * a_strict[a_base + i * c + t] * dr[v_base + i * dv + d];
                }
                dr[v_base + t * dv + d] = dw[v_base + t * dv + d] - acc;
            }
        }
    }
    dr
}

fn assert_close_f32(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    let mut max_abs = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = 2e-4 * w.abs().max(1.0);
        if diff > tol {
            panic!("{ctx}: element {i} got {g} want {w} (|diff| {diff} > tol {tol})");
        }
        max_abs = max_abs.max(diff);
    }
    eprintln!("{ctx}: OK (max |diff| = {max_abs})");
}

fn run_fwd_sub_f32(batch: usize, heads: usize, c: usize, dv: usize, seed: u64) {
    let bh = batch * heads;
    let a_host = pattern(bh * c * c, seed ^ 0xA)
        .into_iter()
        .map(|v| v * 0.05)
        .collect::<Vec<_>>();
    let v_host = pattern(bh * c * dv, seed ^ 0xB);
    let beta_host = pattern(bh * c, seed ^ 0xC)
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();

    let a = rocm_f32(&a_host, vec![batch, heads, c, c]);
    let v = rocm_f32(&v_host, vec![batch, heads, c, dv]);
    let beta = rocm_f32(&beta_host, vec![batch, heads, c]);

    let w =
        gdn_forward_substitution_f32_kt(&a, &v, &beta).expect("gdn_forward_substitution_f32_kt");
    assert_eq!(w.shape(), &[batch, heads, c, dv]);
    assert_eq!(w.dtype(), DType::F32);

    let got = read_f32(&w);
    let want = fwd_sub_ref_f32(&a_host, &v_host, &beta_host, bh, c, dv);
    assert_close_f32(&got, &want, &format!("fwd_sub_f32 bh={bh} c={c} dv={dv}"));
}

fn run_solve_tri_transpose_f32(batch: usize, heads: usize, c: usize, dv: usize, seed: u64) {
    let bh = batch * heads;
    let a_host = pattern(bh * c * c, seed ^ 0xA)
        .into_iter()
        .map(|v| v * 0.05)
        .collect::<Vec<_>>();
    let beta_host = pattern(bh * c, seed ^ 0xB)
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();
    let dw_host = pattern(bh * c * dv, seed ^ 0xC);

    let a = rocm_f32(&a_host, vec![batch, heads, c, c]);
    let beta = rocm_f32(&beta_host, vec![batch, heads, c]);
    let dw = rocm_f32(&dw_host, vec![batch, heads, c, dv]);

    let dr =
        gdn_solve_tri_transpose_f32_kt(&a, &beta, &dw).expect("gdn_solve_tri_transpose_f32_kt");
    assert_eq!(dr.shape(), &[batch, heads, c, dv]);
    assert_eq!(dr.dtype(), DType::F32);

    let got = read_f32(&dr);
    let want = solve_tri_transpose_ref_f32(&a_host, &beta_host, &dw_host, bh, c, dv);
    assert_close_f32(
        &got,
        &want,
        &format!("solve_tri_transpose_f32 bh={bh} c={c} dv={dv}"),
    );
}

#[test]
fn rocm_gdn_forward_substitution_f32_parity() {
    if !rocm_available() {
        eprintln!("ROCm not available; skipping rocm_gdn_forward_substitution_f32_parity");
        return;
    }
    run_fwd_sub_f32(1, 1, 16, 128, 0x4444);
    run_fwd_sub_f32(1, 8, 64, 128, 0x5555);
}

#[test]
fn rocm_gdn_solve_tri_transpose_f32_parity() {
    if !rocm_available() {
        eprintln!("ROCm not available; skipping rocm_gdn_solve_tri_transpose_f32_parity");
        return;
    }
    run_solve_tri_transpose_f32(1, 1, 16, 128, 0x6666);
    run_solve_tri_transpose_f32(1, 8, 64, 128, 0x7777);
}

// ---------------------------------------------------------------------------
// 2) gdn_gated_rms_norm_bf16_kt  (wave-agnostic last-axis reduction)
// ---------------------------------------------------------------------------

/// CPU F32 reference: `out = (x / rms(x)) * weight * silu(z)` over the last axis.
fn gated_rms_ref(
    x: &[bf16],
    z: &[bf16],
    w: &[bf16],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let row = r * hidden;
        let mut sum_sq = 0.0f32;
        for h in 0..hidden {
            let v = x[row + h].to_f32();
            sum_sq += v * v;
        }
        let rms_inv = (sum_sq / hidden as f32 + eps).sqrt().recip();
        for h in 0..hidden {
            let xn = x[row + h].to_f32() * rms_inv * w[h].to_f32();
            out[row + h] = bf16::from_f32(xn * silu(z[row + h].to_f32())).to_f32();
        }
    }
    out
}

#[test]
fn rocm_gdn_gated_rms_norm_parity_sweep() {
    if !rocm_available() {
        eprintln!("ROCm not available; skipping rocm_gdn_gated_rms_norm_parity_sweep");
        return;
    }
    // The GDN gated-RMSNorm kernel is specialized to hidden==128 (one 128-thread
    // block per row). Sweep the row count — the dimension that actually varies —
    // including counts that straddle the 32/64-lane wavefront boundary, so the
    // wave-agnostic shared-memory reduction is exercised under both wave modes.
    let hidden = 128usize;
    let eps = 1e-6f32;
    for &rows in &[1usize, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256] {
        let x_host = pattern_bf16(rows * hidden, 0x100 + rows as u64);
        // wider scale on x/z to stress the reduction dynamic range.
        let z_host: Vec<bf16> = pattern(rows * hidden, 0x200 + rows as u64)
            .into_iter()
            .map(|v| bf16::from_f32(v * 2.0))
            .collect();
        let w_host: Vec<bf16> = pattern(hidden, 0x300 + rows as u64)
            .into_iter()
            .map(|v| bf16::from_f32(v + 1.0))
            .collect();

        let x = rocm_bf16(&x_host, vec![rows, hidden]);
        let z = rocm_bf16(&z_host, vec![rows, hidden]);
        let w = rocm_bf16(&w_host, vec![hidden]);

        assert!(
            gdn_gated_rms_norm_supports_kt(&x, &z, &w),
            "rows={rows}: envelope check failed"
        );

        let out = gdn_gated_rms_norm_bf16_kt(&x, &z, &w, eps).expect("gdn_gated_rms_norm_bf16_kt");
        assert_eq!(out.shape(), &[rows, hidden]);
        assert_eq!(out.dtype(), DType::BF16);

        let got = read_bf16_f32(&out);
        let want = gated_rms_ref(&x_host, &z_host, &w_host, rows, hidden, eps);
        assert_close(
            &got,
            &want,
            5e-3,
            &format!("gated_rms rows={rows} hidden={hidden}"),
        );
    }
}
