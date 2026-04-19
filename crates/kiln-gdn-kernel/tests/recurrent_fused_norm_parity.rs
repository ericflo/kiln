//! Parity test for the fused GDN recurrent-step-with-norms CUDA kernel.
//!
//! `kiln_gdn_kernel::gdn_recurrent_forward_fused_norm` absorbs three
//! originally-separate stages from `kiln-model::forward::gated_deltanet_forward`
//! into one CUDA launch at decode-time (`seq_len == 1`):
//!
//!   Step 5 — qk_norm:
//!       q_normed = l2_normalize(q_raw) * q_scale
//!       k_normed = l2_normalize(k_raw)
//!   Step 7 — gdn_recurrent_step:
//!       (decay, delta, state-update, out_acc)
//!   Step 8 — gated_rms_norm:
//!       rms_inv = rsqrt(mean(attn_out^2) + rms_eps)
//!       out     = attn_out * rms_inv * gamma * silu(z)
//!
//! The algorithmic oracle in `reference_host` reimplements the same math
//! in F32 on the host so the test does not re-run the kernel to check
//! itself. The reference is deliberately independent: it uses naive
//! `for` loops, not candle ops, and keeps accumulators in F32.
//!
//! Gracefully no-ops on non-CUDA hosts.

use candle_core::{DType, Device, Tensor};
use kiln_gdn_kernel::{
    gdn_recurrent_forward_fused_norm, gdn_recurrent_forward_fused_norm_supports,
};

fn lcg_step(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fff_ffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg_step(&mut s) * scale).collect()
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

fn bf16_round_trip(x: f32) -> f32 {
    // Match the kernel's bf16 round-trip (truncation): load as bf16 then
    // cast back to f32. `half::bf16::from_f32` matches __float2bfloat16's
    // round-to-nearest-even semantics used by the CUDA kernel.
    half::bf16::from_f32(x).to_f32()
}

/// Reference path for the fused kernel. Inputs and outputs are host F32
/// but values are round-tripped through bf16 at the exact same points
/// the kernel does, so the test compares round-trip-adjusted F32 tensors.
///
/// Shapes:
///   q_raw, k_raw:  `[rows, dk]`
///   v, z:          `[rows, dv]`
///   beta, g:       `[rows]`
///   gamma:         `[dv]`
///   state (in+out):`[rows, dk, dv]`  (row-major: [i][j] → state[row*dk*dv + i*dv + j])
///   out:           `[rows, dv]`
#[allow(clippy::too_many_arguments)]
fn reference_host(
    q_raw: &[f32],
    k_raw: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    z: &[f32],
    gamma: &[f32],
    state: &mut [f32],
    rows: usize,
    dk: usize,
    dv: usize,
    q_scale: f32,
    l2_eps: f32,
    rms_eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dv];

    for r in 0..rows {
        // ---- Phase 1: L2-normalize q_raw and k_raw (F32 reduction) ----
        let mut q_sumsq = 0.0f32;
        let mut k_sumsq = 0.0f32;
        let q_row = &q_raw[r * dk..(r + 1) * dk];
        let k_row = &k_raw[r * dk..(r + 1) * dk];
        for i in 0..dk {
            q_sumsq += q_row[i] * q_row[i];
            k_sumsq += k_row[i] * k_row[i];
        }
        let inv_q = q_scale * (q_sumsq + l2_eps).sqrt().recip();
        let inv_k = (k_sumsq + l2_eps).sqrt().recip();
        let q_norm: Vec<f32> = q_row.iter().map(|&x| x * inv_q).collect();
        let k_norm: Vec<f32> = k_row.iter().map(|&x| x * inv_k).collect();

        // ---- Phase 3: recurrent step ----
        let decay = g[r].exp();
        let beta_t = beta[r];
        let v_row = &v[r * dv..(r + 1) * dv];
        let state_row_offset = r * dk * dv;

        // Decayed state + per-column v_pred
        let mut v_pred = vec![0.0f32; dv];
        let mut decayed = vec![0.0f32; dk * dv];
        for i in 0..dk {
            for j in 0..dv {
                let d = decay * state[state_row_offset + i * dv + j];
                decayed[i * dv + j] = d;
                v_pred[j] += k_norm[i] * d;
            }
        }

        // delta[j] = beta * (v[j] - v_pred[j])
        let mut delta = vec![0.0f32; dv];
        for j in 0..dv {
            delta[j] = beta_t * (v_row[j] - v_pred[j]);
        }

        // New state = decayed + k_norm ⊗ delta; out_acc[j] = sum_i q_norm[i] * new_state[i, j]
        let mut out_acc = vec![0.0f32; dv];
        for i in 0..dk {
            for j in 0..dv {
                let new_s = decayed[i * dv + j] + k_norm[i] * delta[j];
                // The kernel writes new_s back to state after a bf16 round-trip.
                let new_s_bf = bf16_round_trip(new_s);
                state[state_row_offset + i * dv + j] = new_s_bf;
                out_acc[j] += q_norm[i] * new_s_bf;
            }
        }

        // ---- Phase 4+5: RMSNorm + gate + final bf16 write ----
        let mut rms_sumsq = 0.0f32;
        for j in 0..dv {
            rms_sumsq += out_acc[j] * out_acc[j];
        }
        let rms_inv = (rms_sumsq / dv as f32 + rms_eps).sqrt().recip();

        for j in 0..dv {
            let gate = silu_f32(z[r * dv + j]);
            let y = out_acc[j] * rms_inv * gamma[j] * gate;
            out[r * dv + j] = bf16_round_trip(y);
        }
    }

    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    device: &Device,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    seed: u64,
    label: &str,
) {
    let rows = batch * heads;
    let q_scale = 1.0 / (dk as f32).sqrt();
    let l2_eps = 1e-6f32;
    let rms_eps = 1e-6f32;

    // Host tensors: activations near N(0, 0.25); state / gate weights smaller.
    let q_host = fill(seed ^ 0x51, rows * dk, 1.0);
    let k_host = fill(seed ^ 0x52, rows * dk, 1.0);
    let v_host = fill(seed ^ 0x53, rows * dv, 1.0);
    let beta_host = fill(seed ^ 0x54, rows, 0.5);
    let g_host = fill(seed ^ 0x55, rows, 0.2);
    let z_host = fill(seed ^ 0x56, rows * dv, 1.0);
    let gamma_host = fill(seed ^ 0x57, dv, 0.5);
    let state_host_orig = fill(seed ^ 0x58, rows * dk * dv, 0.3);

    // First round-trip both the state and the pre-norm inputs through bf16,
    // so the reference starts from the same lossy values the kernel sees.
    let state_host_bf: Vec<f32> = state_host_orig.iter().copied().map(bf16_round_trip).collect();
    let q_bf: Vec<f32> = q_host.iter().copied().map(bf16_round_trip).collect();
    let k_bf: Vec<f32> = k_host.iter().copied().map(bf16_round_trip).collect();
    let v_bf: Vec<f32> = v_host.iter().copied().map(bf16_round_trip).collect();
    let beta_bf: Vec<f32> = beta_host.iter().copied().map(bf16_round_trip).collect();
    let g_bf: Vec<f32> = g_host.iter().copied().map(bf16_round_trip).collect();
    let z_bf: Vec<f32> = z_host.iter().copied().map(bf16_round_trip).collect();
    let gamma_bf: Vec<f32> = gamma_host.iter().copied().map(bf16_round_trip).collect();

    // Build candle tensors on CUDA.
    let q = Tensor::from_vec(q_bf.clone(), (batch, heads, dk), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let k = Tensor::from_vec(k_bf.clone(), (batch, heads, dk), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let v = Tensor::from_vec(v_bf.clone(), (batch, heads, dv), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let beta = Tensor::from_vec(beta_bf.clone(), (batch, heads), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let g = Tensor::from_vec(g_bf.clone(), (batch, heads), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let z = Tensor::from_vec(z_bf.clone(), (batch, heads, dv), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let gamma = Tensor::from_vec(gamma_bf.clone(), (dv,), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let mut state = Tensor::from_vec(state_host_bf.clone(), (batch, heads, dk, dv), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();

    assert!(
        gdn_recurrent_forward_fused_norm_supports(&q, &k, &v, &beta, &g, &z, &gamma, &state),
        "{label}: envelope check failed"
    );

    let out = gdn_recurrent_forward_fused_norm(
        &q, &k, &v, &beta, &g, &z, &gamma, &mut state, q_scale, l2_eps, rms_eps,
    )
    .expect("fused kernel");

    let out_host: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let state_host_after: Vec<f32> = state
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Reference.
    let mut state_ref = state_host_bf;
    let out_ref = reference_host(
        &q_bf,
        &k_bf,
        &v_bf,
        &beta_bf,
        &g_bf,
        &z_bf,
        &gamma_bf,
        &mut state_ref,
        rows,
        dk,
        dv,
        q_scale,
        l2_eps,
        rms_eps,
    );

    let out_err = max_abs_diff(&out_host, &out_ref);
    let state_err = max_abs_diff(&state_host_after, &state_ref);

    // bf16 has ~3e-3 relative resolution. The fused kernel accumulates through
    // dv=128 in F32 before the final bf16 write, matching the reference's
    // precision. 1.5e-2 absolute is the same budget existing kiln kernels use
    // (gates_parity, marlin_w4a16_gemm).
    let tol = 1.5e-2f32;
    println!(
        "[{label}] shape=[B={batch},H={heads},dk={dk},dv={dv}] out_err={out_err:.3e} state_err={state_err:.3e}"
    );
    assert!(
        out_err < tol,
        "{label}: out max_abs_diff {out_err:.3e} >= {tol:.3e}"
    );
    assert!(
        state_err < tol,
        "{label}: state max_abs_diff {state_err:.3e} >= {tol:.3e}"
    );
}

#[test]
fn gdn_recurrent_fused_norm_parity_decode_qwen_shape() {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "Skipping gdn_recurrent_fused_norm parity test: no CUDA device ({e})"
            );
            return;
        }
    };

    // Qwen3.5-4B GDN decode shape: B=1, nv=16, dk=dv=128.
    run_case(&device, 1, 16, 128, 128, 0xDEAD_BEEF, "decode/qwen3_5_4b");
}

#[test]
fn gdn_recurrent_fused_norm_parity_small_shape() {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "Skipping gdn_recurrent_fused_norm parity test: no CUDA device ({e})"
            );
            return;
        }
    };

    // Small shape to catch boundary bugs on warp boundaries (dv=32 = 1 warp).
    run_case(&device, 2, 4, 64, 32, 0xCAFE_F00D, "small/B=2,H=4,dk=64,dv=32");
}

#[test]
fn gdn_recurrent_fused_norm_parity_asymmetric_shape() {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "Skipping gdn_recurrent_fused_norm parity test: no CUDA device ({e})"
            );
            return;
        }
    };

    // dk != dv (dk=256 uses the MAX_DK=256 template; dv=128 stays at 4 warps).
    run_case(&device, 1, 8, 256, 128, 0xFACE_0FF, "asymmetric/dk=256,dv=128");
}
