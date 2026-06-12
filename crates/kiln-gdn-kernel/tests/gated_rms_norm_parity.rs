//! Parity test for the fused GDN gated RMSNorm CUDA kernel.
//!
//! The kernel in `kiln_gdn_kernel::gdn_gated_rms_norm_bf16_kt` fuses
//! the candle chain `to_f32 -> rms_norm -> silu -> mul -> bf16` used by
//! `kiln-model` in the `kiln/gdn/gated_norm` NVTX range.
//!
//! Phase 7 candle-removal (#1082): migrated off candle Tensors to the
//! kt-typed surface (`gdn_gated_rms_norm_bf16_kt` /
//! `gdn_gated_rms_norm_supports_kt`). Inputs are constructed via
//! `Tensor::cuda_from_slice`; the reference path is a pure-Rust F32
//! host loop (the kernel's documented algorithm). Outputs are pulled
//! back via `cuda_to_host_copy` and compared element-wise in F32.
//!
//! CUDA-only: `Tensor::cuda_from_slice` / `cuda_to_host_copy` /
//! `primary_cuda_context` are the cuda-storage substrate helpers and
//! don't exist on the ROCm build. The backend-neutral ROCm parity
//! coverage lives in `rocm_gdn_parity.rs` (gated on `feature = "rocm"`),
//! which exercises the same fused gated-RMSNorm kernel.
#![cfg(feature = "cuda")]

use half::bf16;

use kiln_gdn_kernel::{
    gdn_gated_rms_norm_bf16_f32_weight_kt, gdn_gated_rms_norm_bf16_kt,
    gdn_gated_rms_norm_bwd_bf16_f32_weight_kt, gdn_gated_rms_norm_bwd_bf16_kt,
    gdn_gated_rms_norm_bwd_supports_kt, gdn_gated_rms_norm_f32_weight_supports_kt,
    gdn_gated_rms_norm_supports_kt, gdn_l2_norm_scale_bwd_bf16_kt,
    gdn_l2_norm_scale_bwd_supports_kt,
};
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
    let mut state = seed;
    (0..n).map(|_| lcg_seed(&mut state) * scale).collect()
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

/// Pure-Rust F32 host reference: the *algorithmic* oracle for the
/// fused gated RMSNorm kernel. Computes
/// `(x / rms(x)) * weight * silu(z)` row by row over the last axis.
fn reference_host(
    x_host: &[f32],
    z_host: &[f32],
    weight_host: &[f32],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let row_off = r * hidden;
        // RMS over the last axis (mean of squares + eps, then sqrt).
        let mut sum_sq = 0.0f32;
        for h in 0..hidden {
            let v = x_host[row_off + h];
            sum_sq += v * v;
        }
        let variance = sum_sq / hidden as f32;
        let rms_inv = 1.0 / (variance + eps).sqrt();
        for h in 0..hidden {
            let xn = x_host[row_off + h] * rms_inv * weight_host[h];
            out[row_off + h] = xn * silu_f32(z_host[row_off + h]);
        }
    }
    out
}

fn to_bf16_vec(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|&v| bf16::from_f32(v)).collect()
}

fn bf16_vec_to_f32(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|v| v.to_f32()).collect()
}

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

fn read_f32_host(t: &Tensor) -> Vec<f32> {
    let host = cuda_to_host_copy(t).expect("cuda -> host copy");
    assert_eq!(host.dtype(), DType::F32);
    let cpu = host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .expect("CpuStorage");
    cpu.as_bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn reference_bwd_host(
    dout_host: &[f32],
    x_host: &[f32],
    z_host: &[f32],
    weight_host: &[f32],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; rows * hidden];
    let mut dz = vec![0.0f32; rows * hidden];
    let mut dw = vec![0.0f32; hidden];
    for r in 0..rows {
        let row_off = r * hidden;
        let mut sum_sq = 0.0f32;
        for h in 0..hidden {
            let x = x_host[row_off + h];
            sum_sq += x * x;
        }
        let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();

        let mut s = 0.0f32;
        for h in 0..hidden {
            let idx = row_off + h;
            let gate = silu_f32(z_host[idx]);
            let d_normed = dout_host[idx] * gate;
            s += d_normed * x_host[idx] * weight_host[h];
        }

        let rms_inv3 = rms_inv * rms_inv * rms_inv;
        for h in 0..hidden {
            let idx = row_off + h;
            let x = x_host[idx];
            let z = z_host[idx];
            let w = weight_host[h];
            let dout = dout_host[idx];
            let sig = sigmoid_f32(z);
            let gate = z * sig;
            let d_normed = dout * gate;
            let normed = x * w * rms_inv;
            let silu_grad = sig * (1.0 + z * (1.0 - sig));
            dx[idx] = d_normed * w * rms_inv - x * s * (rms_inv3 / hidden as f32);
            dz[idx] = dout * normed * silu_grad;
            dw[h] += d_normed * x * rms_inv;
        }
    }
    (dx, dz, dw)
}

fn reference_l2_bwd_host(
    dout_host: &[f32],
    x_host: &[f32],
    rows: usize,
    hidden: usize,
    scale: f32,
    eps: f32,
) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let row_off = r * hidden;
        let mut sum_sq = 0.0f32;
        let mut s = 0.0f32;
        for h in 0..hidden {
            let idx = row_off + h;
            let x = x_host[idx];
            let dy = dout_host[idx];
            sum_sq += x * x;
            s += dy * x;
        }
        let inv_n = 1.0 / (sum_sq + eps).sqrt();
        let inv_n3 = inv_n * inv_n * inv_n;
        for h in 0..hidden {
            let idx = row_off + h;
            dx[idx] = scale * (dout_host[idx] * inv_n - x_host[idx] * s * inv_n3);
        }
    }
    dx
}

fn run_case(batch: usize, seq_len: usize, heads: usize, hidden: usize, seed: u64, label: &str) {
    // The kt API takes 2D `[rows, hidden]` inputs — the candle path's
    // 4D `(B, T, H, hidden)` collapses to the same row count.
    let rows = batch * seq_len * heads;
    let elems = rows * hidden;

    let x_host = fill(seed ^ 0xA11C_E5, elems, 2.0);
    let z_host = fill(seed ^ 0x6A7E, elems, 4.0);
    let w_host: Vec<f32> = fill(seed ^ 0xBEEF, hidden, 0.5)
        .into_iter()
        .map(|v| v + 1.0)
        .collect();

    let x_bf16 = to_bf16_vec(&x_host);
    let z_bf16 = to_bf16_vec(&z_host);
    let w_bf16 = to_bf16_vec(&w_host);
    let x_ref = bf16_vec_to_f32(&x_bf16);
    let z_ref = bf16_vec_to_f32(&z_bf16);
    let w_ref = bf16_vec_to_f32(&w_bf16);

    let x = Tensor::cuda_from_slice(&x_bf16, vec![rows, hidden], 0).expect("upload x");
    let z = Tensor::cuda_from_slice(&z_bf16, vec![rows, hidden], 0).expect("upload z");
    let weight = Tensor::cuda_from_slice(&w_bf16, vec![hidden], 0).expect("upload weight");
    let weight_f32 = Tensor::cuda_from_slice(&w_host, vec![hidden], 0).expect("upload f32 weight");

    assert!(
        gdn_gated_rms_norm_supports_kt(&x, &z, &weight),
        "{label}: envelope check failed"
    );
    assert!(
        gdn_gated_rms_norm_f32_weight_supports_kt(&x, &z, &weight_f32),
        "{label}: f32-weight envelope check failed"
    );

    let fused = gdn_gated_rms_norm_bf16_kt(&x, &z, &weight, 1e-6).expect("fused gated RMSNorm");
    let fused_f32_weight = gdn_gated_rms_norm_bf16_f32_weight_kt(&x, &z, &weight_f32, 1e-6)
        .expect("fused gated RMSNorm f32 weight");

    assert_eq!(fused.shape(), &[rows, hidden]);
    assert_eq!(fused.dtype(), DType::BF16);
    assert_eq!(fused_f32_weight.shape(), &[rows, hidden]);
    assert_eq!(fused_f32_weight.dtype(), DType::BF16);

    // BF16-round-trip the host reference so the comparison sees the
    // same precision the kernel writes.
    let ref_f32 = reference_host(&x_ref, &z_ref, &w_ref, rows, hidden, 1e-6);
    let ref_bf16_round_tripped: Vec<f32> = ref_f32
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect();
    let got_f32 = read_bf16_host_as_f32(&fused);

    let abs: Vec<f32> = got_f32
        .iter()
        .zip(ref_bf16_round_tripped.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let max = abs.iter().cloned().fold(0.0f32, f32::max);
    let mean = if abs.is_empty() {
        0.0
    } else {
        abs.iter().sum::<f32>() / abs.len() as f32
    };

    println!(
        "[{label}] shape=[{batch},{seq_len},{heads},{hidden}] max_abs={max:.3e} mean_abs={mean:.3e}"
    );
    assert!(max < 5e-3, "{label}: max_abs_diff {max} exceeds tolerance");
    assert!(
        mean < 5e-4,
        "{label}: mean_abs_diff {mean} exceeds tolerance"
    );

    let ref_f32_weight = reference_host(&x_ref, &z_ref, &w_host, rows, hidden, 1e-6);
    let ref_f32_weight_bf16: Vec<f32> = ref_f32_weight
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect();
    let got_f32_weight = read_bf16_host_as_f32(&fused_f32_weight);
    let abs_f32_weight: Vec<f32> = got_f32_weight
        .iter()
        .zip(ref_f32_weight_bf16.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let max_f32_weight = abs_f32_weight.iter().cloned().fold(0.0f32, f32::max);
    let mean_f32_weight = if abs_f32_weight.is_empty() {
        0.0
    } else {
        abs_f32_weight.iter().sum::<f32>() / abs_f32_weight.len() as f32
    };
    println!(
        "[{label} f32-weight] shape=[{batch},{seq_len},{heads},{hidden}] max_abs={max_f32_weight:.3e} mean_abs={mean_f32_weight:.3e}"
    );
    assert!(
        max_f32_weight < 5e-3,
        "{label}: f32-weight max_abs_diff {max_f32_weight} exceeds tolerance"
    );
    assert!(
        mean_f32_weight < 5e-4,
        "{label}: f32-weight mean_abs_diff {mean_f32_weight} exceeds tolerance"
    );
}

fn run_bwd_case(batch: usize, seq_len: usize, heads: usize, hidden: usize, seed: u64, label: &str) {
    let rows = batch * seq_len * heads;
    let elems = rows * hidden;

    let x_host = fill(seed ^ 0xA11C_E5, elems, 2.0);
    let z_host = fill(seed ^ 0x6A7E, elems, 4.0);
    let dout_host = fill(seed ^ 0xD06, elems, 1.5);
    let w_host: Vec<f32> = fill(seed ^ 0xBEEF, hidden, 0.5)
        .into_iter()
        .map(|v| v + 1.0)
        .collect();

    let x_bf16 = to_bf16_vec(&x_host);
    let z_bf16 = to_bf16_vec(&z_host);
    let dout_bf16 = to_bf16_vec(&dout_host);
    let w_bf16 = to_bf16_vec(&w_host);
    let x_ref = bf16_vec_to_f32(&x_bf16);
    let z_ref = bf16_vec_to_f32(&z_bf16);
    let dout_ref = bf16_vec_to_f32(&dout_bf16);
    let w_ref = bf16_vec_to_f32(&w_bf16);

    let x = Tensor::cuda_from_slice(&x_bf16, vec![rows, hidden], 0).expect("upload x");
    let z = Tensor::cuda_from_slice(&z_bf16, vec![rows, hidden], 0).expect("upload z");
    let dout = Tensor::cuda_from_slice(&dout_bf16, vec![rows, hidden], 0).expect("upload dout");
    let weight = Tensor::cuda_from_slice(&w_bf16, vec![hidden], 0).expect("upload weight");
    let weight_f32 = Tensor::cuda_from_slice(&w_host, vec![hidden], 0).expect("upload f32 weight");

    assert!(
        gdn_gated_rms_norm_bwd_supports_kt(&dout, &x, &z, &weight),
        "{label}: backward envelope check failed"
    );
    assert!(
        gdn_gated_rms_norm_bwd_supports_kt(&dout, &x, &z, &weight_f32),
        "{label}: f32-weight backward envelope check failed"
    );

    let grads = gdn_gated_rms_norm_bwd_bf16_kt(&dout, &x, &z, &weight, 1e-6).expect("fused bwd");
    let grads_f32_weight =
        gdn_gated_rms_norm_bwd_bf16_f32_weight_kt(&dout, &x, &z, &weight_f32, 1e-6)
            .expect("fused bwd f32 weight");

    let (ref_dx, ref_dz, ref_dw) =
        reference_bwd_host(&dout_ref, &x_ref, &z_ref, &w_ref, rows, hidden, 1e-6);
    let ref_dx_bf16: Vec<f32> = ref_dx.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let ref_dz_bf16: Vec<f32> = ref_dz.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();

    let got_dx = read_bf16_host_as_f32(&grads.dx);
    let got_dz = read_bf16_host_as_f32(&grads.dz);
    let got_dw = read_f32_host(&grads.dw);

    let max_abs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let dx_max = max_abs(&got_dx, &ref_dx_bf16);
    let dz_max = max_abs(&got_dz, &ref_dz_bf16);
    let dw_max = max_abs(&got_dw, &ref_dw);

    println!(
        "[{label} bwd] shape=[{batch},{seq_len},{heads},{hidden}] dx_max={dx_max:.3e} dz_max={dz_max:.3e} dw_max={dw_max:.3e}"
    );
    assert!(
        dx_max < 6e-3,
        "{label}: dx max_abs_diff {dx_max} exceeds tolerance"
    );
    assert!(
        dz_max < 6e-3,
        "{label}: dz max_abs_diff {dz_max} exceeds tolerance"
    );
    assert!(
        dw_max < 2e-2,
        "{label}: dw max_abs_diff {dw_max} exceeds tolerance"
    );

    let (ref_dx_f32_weight, ref_dz_f32_weight, ref_dw_f32_weight) =
        reference_bwd_host(&dout_ref, &x_ref, &z_ref, &w_host, rows, hidden, 1e-6);
    let ref_dx_f32_weight_bf16: Vec<f32> = ref_dx_f32_weight
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect();
    let ref_dz_f32_weight_bf16: Vec<f32> = ref_dz_f32_weight
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect();
    let got_dx_f32_weight = read_bf16_host_as_f32(&grads_f32_weight.dx);
    let got_dz_f32_weight = read_bf16_host_as_f32(&grads_f32_weight.dz);
    let got_dw_f32_weight = read_f32_host(&grads_f32_weight.dw);

    let dx_f32_weight_max = max_abs(&got_dx_f32_weight, &ref_dx_f32_weight_bf16);
    let dz_f32_weight_max = max_abs(&got_dz_f32_weight, &ref_dz_f32_weight_bf16);
    let dw_f32_weight_max = max_abs(&got_dw_f32_weight, &ref_dw_f32_weight);

    println!(
        "[{label} bwd f32-weight] shape=[{batch},{seq_len},{heads},{hidden}] dx_max={dx_f32_weight_max:.3e} dz_max={dz_f32_weight_max:.3e} dw_max={dw_f32_weight_max:.3e}"
    );
    assert!(
        dx_f32_weight_max < 6e-3,
        "{label}: f32-weight dx max_abs_diff {dx_f32_weight_max} exceeds tolerance"
    );
    assert!(
        dz_f32_weight_max < 6e-3,
        "{label}: f32-weight dz max_abs_diff {dz_f32_weight_max} exceeds tolerance"
    );
    assert!(
        dw_f32_weight_max < 2e-2,
        "{label}: f32-weight dw max_abs_diff {dw_f32_weight_max} exceeds tolerance"
    );
}

fn run_l2_bwd_case(
    batch: usize,
    seq_len: usize,
    heads: usize,
    hidden: usize,
    scale: f32,
    seed: u64,
    label: &str,
) {
    let rows = batch * seq_len * heads;
    let elems = rows * hidden;

    let x_host = fill(seed ^ 0xA11C_E5, elems, 2.0);
    let dout_host = fill(seed ^ 0xD06, elems, 1.5);

    let x_bf16 = to_bf16_vec(&x_host);
    let dout_bf16 = to_bf16_vec(&dout_host);
    let x_ref = bf16_vec_to_f32(&x_bf16);
    let dout_ref = bf16_vec_to_f32(&dout_bf16);

    let x = Tensor::cuda_from_slice(&x_bf16, vec![rows, hidden], 0).expect("upload x");
    let dout = Tensor::cuda_from_slice(&dout_bf16, vec![rows, hidden], 0).expect("upload dout");

    assert!(
        gdn_l2_norm_scale_bwd_supports_kt(&dout, &x),
        "{label}: l2 backward envelope check failed"
    );

    let dx = gdn_l2_norm_scale_bwd_bf16_kt(&dout, &x, scale, 1e-6).expect("fused l2 bwd");

    let ref_dx = reference_l2_bwd_host(&dout_ref, &x_ref, rows, hidden, scale, 1e-6);
    let ref_dx_bf16: Vec<f32> = ref_dx.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let got_dx = read_bf16_host_as_f32(&dx);

    let dx_max = got_dx
        .iter()
        .zip(ref_dx_bf16.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    println!(
        "[{label} l2-bwd] shape=[{batch},{seq_len},{heads},{hidden}] scale={scale:.6} dx_max={dx_max:.3e}"
    );
    assert!(
        dx_max < 6e-3,
        "{label}: l2 dx max_abs_diff {dx_max} exceeds tolerance"
    );
}

#[test]
fn gdn_gated_rms_norm_parity_vs_host_reference() {
    if !cuda_available() {
        eprintln!("Skipping gdn_gated_rms_norm parity test: no CUDA device");
        return;
    }

    run_case(1, 1, 32, 128, 0xCAFE_F00D, "decode");
    run_case(1, 64, 32, 128, 0xDEAD_BEEF, "prefill/T=64");
}

#[test]
fn gdn_gated_rms_norm_backward_parity_vs_host_reference() {
    if !cuda_available() {
        eprintln!("Skipping gdn_gated_rms_norm backward parity test: no CUDA device");
        return;
    }

    run_bwd_case(1, 1, 32, 128, 0xC0FF_EE, "decode");
    run_bwd_case(1, 16, 32, 128, 0xF00D_BAAD, "prefill/T=16");
}

#[test]
fn gdn_l2_norm_scale_backward_parity_vs_host_reference() {
    if !cuda_available() {
        eprintln!("Skipping gdn_l2_norm_scale backward parity test: no CUDA device");
        return;
    }

    run_l2_bwd_case(1, 1, 32, 128, 1.0, 0xC0DE_CAFE, "decode/k");
    run_l2_bwd_case(
        1,
        16,
        32,
        128,
        1.0 / (128.0f32).sqrt(),
        0xBAD5_EED,
        "prefill/q",
    );
}
