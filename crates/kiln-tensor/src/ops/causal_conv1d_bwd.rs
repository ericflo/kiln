//! Device-agnostic backward (w.r.t. input) for the depthwise causal conv1d
//! used by the GDN (Gated DeltaNet) linear-attention layers.
//!
//! # Forward recap
//!
//! The production prefill forward (`causal_conv1d_prefill` in `kiln-model`) and
//! the CUDA kernel `causal_depthwise_conv1d_f32` compute, in a `[rows, channels]`
//! layout where `rows` is the contiguous causal (time) axis:
//!
//! ```text
//! out[r, c] = sum_{j=0}^{K-1} weight[c, j] * x_padded[r + j, c]
//! ```
//!
//! where `x_padded` is `x` left-padded with the `K-1` previous-input conv-state
//! rows (so `x_padded[p] = x[p - (K-1)]` for `p >= K-1`). Equivalently, against
//! the un-padded input `x` itself:
//!
//! ```text
//! out[r, c] = sum_{k=0}^{K-1} weight[c, k] * x[r - (K-1) + k, c]   (x[<0] := pad)
//! ```
//!
//! # Backward w.r.t. input
//!
//! Differentiating `out` w.r.t. an input row `s`, the only `out` rows that read
//! `x[s]` are `r = s + (K-1) - k` for `k = 0..K-1` (with weight `weight[c, k]`).
//! Hence the transpose/correlation:
//!
//! ```text
//! din[s, c] = sum_{k=0}^{K-1} weight[c, k] * grad_out[s + (K-1) - k, c]
//! ```
//!
//! valid only where `0 <= s + (K-1) - k < rows` (the conv-state contribution is
//! a separate, non-differentiable boundary in LoRA training, so it is dropped —
//! exactly matching the CUDA `causal_depthwise_conv1d_bwd_input_f32` kernel,
//! which clamps `out_row` to `[0, rows)`).
//!
//! Substituting `m = (K-1) - k` (so `k = (K-1) - m`, `m = 0..K-1`):
//!
//! ```text
//! din[s, c] = sum_{m=0}^{K-1} weight[c, (K-1)-m] * grad_out[s + m, c]
//! ```
//!
//! a *future-looking* correlation. Right-padding `grad_out` with `K-1` zero rows
//! along the time axis (`grad_padded`, shape `[rows + K-1, channels]`) makes
//! every `grad_padded[s + m]` in-range for `s in 0..rows`, so:
//!
//! ```text
//! din[s, c] = sum_{m=0}^{K-1} weight_rev[c, m] * grad_padded[s + m, c]
//! ```
//!
//! This mirrors the forward's narrow+broadcast-mul accumulation loop and is
//! built purely from device-agnostic `kiln_tensor` ops (pad / narrow /
//! broadcast-mul / add), so it runs on CPU, Metal, and CUDA alike — no FFI.
//!
//! # Correctness gate
//!
//! Validated against a central finite-difference of `sum(forward)` in the
//! `causal_conv1d_bwd_input_matches_finite_difference` unit test below (CPU,
//! deterministic).

use crate::{DType, Result, Tensor, bail};

/// CPU/Metal/CUDA composite for the depthwise causal conv1d input gradient.
///
/// - `grad_out`: `[rows, channels]` — the conv output gradient (rows = time).
/// - `weight`: `[channels, kernel]`.
/// - returns: `din` of shape `[rows, channels]`, same dtype/device as `grad_out`.
///
/// Numerically matches the CUDA `causal_depthwise_conv1d_bwd_input_f32` kernel
/// (state-row contribution clamped out, as in that kernel).
pub fn causal_depthwise_conv1d_bwd_input_composite(
    grad_out: &Tensor,
    weight: &Tensor,
    kernel: usize,
) -> Result<Tensor> {
    if kernel < 2 {
        bail!("causal_conv1d_bwd_input_composite: kernel must be >= 2, got {kernel}");
    }
    let (rows, channels) = grad_out.dims2().map_err(|e| {
        crate::Error::Msg(format!(
            "causal_conv1d_bwd_input_composite: grad_out must be [rows, channels]: {e}"
        ))
    })?;
    let (wc, wk) = weight.dims2().map_err(|e| {
        crate::Error::Msg(format!(
            "causal_conv1d_bwd_input_composite: weight must be [channels, kernel]: {e}"
        ))
    })?;
    if wc != channels || wk != kernel {
        bail!(
            "causal_conv1d_bwd_input_composite: weight {:?} != [{channels}, {kernel}]",
            weight.dims()
        );
    }

    // Compute in F32 (the CUDA kernel is F32); cast back to grad_out's dtype at
    // the end so the grad-dtype-follows-tensor invariant holds.
    let out_dtype = grad_out.dtype();
    let grad_f32 = if out_dtype == DType::F32 {
        grad_out.clone()
    } else {
        grad_out.to_dtype(DType::F32)?
    };
    let weight_f32 = if weight.dtype() == DType::F32 {
        weight.contiguous()?
    } else {
        weight.to_dtype(DType::F32)?.contiguous()?
    };

    // Right-pad grad_out with (K-1) zero rows on the time axis (axis 0).
    let k_minus_1 = kernel - 1;
    let grad_padded = crate::ops::pad(&grad_f32.contiguous()?, &[(0, k_minus_1), (0, 0)], 0.0)?;

    // din[s, c] = sum_{m=0}^{K-1} weight[c, (K-1)-m] * grad_padded[s + m, c]
    let mut din: Option<Tensor> = None;
    for m in 0..kernel {
        // grad_padded[s + m] for s in 0..rows  ->  [rows, channels]
        let g_slice = grad_padded.narrow(0, m, rows)?;
        // weight column (K-1)-m, shaped [1, channels] to broadcast across rows.
        let w_col = weight_f32
            .narrow(1, k_minus_1 - m, 1)? // [channels, 1]
            .transpose(0, 1)? // [1, channels]
            .contiguous()?;
        let term = g_slice.broadcast_mul(&w_col)?;
        din = Some(match din {
            None => term,
            Some(acc) => acc.add(&term)?,
        });
    }
    let din = din.expect("kernel >= 2 guarantees at least one accumulation term");

    if din.dtype() == out_dtype {
        Ok(din)
    } else {
        din.to_dtype(out_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    /// Reference forward in the `[rows, channels]` convention, matching
    /// `causal_conv1d_prefill` / the CUDA forward kernel with ZERO conv-state
    /// (the differentiable boundary): `out[r,c] = sum_k w[c,k] * x[r-(K-1)+k, c]`
    /// with `x[<0] = 0`. Accumulated in f64 so the finite-difference reference is
    /// not polluted by f32 round-off near the FD step size.
    fn forward_ref(
        x: &[f64],
        weight: &[f64],
        rows: usize,
        channels: usize,
        kernel: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; rows * channels];
        let state_rows = kernel - 1;
        for r in 0..rows {
            for c in 0..channels {
                let mut acc = 0.0f64;
                for k in 0..kernel {
                    // padded_row = r + k; input_row = padded_row - state_rows
                    let padded_row = r as isize + k as isize;
                    let input_row = padded_row - state_rows as isize;
                    if input_row >= 0 && (input_row as usize) < rows {
                        acc += x[input_row as usize * channels + c] * weight[c * kernel + k];
                    }
                }
                out[r * channels + c] = acc;
            }
        }
        out
    }

    #[test]
    fn causal_conv1d_bwd_input_matches_finite_difference() {
        // Deterministic small random inputs.
        let rows = 7usize;
        let channels = 5usize;
        let kernel = 4usize;

        // Simple LCG for reproducible pseudo-random values in [-1, 1].
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };

        let x: Vec<f32> = (0..rows * channels).map(|_| next()).collect();
        let weight: Vec<f32> = (0..channels * kernel).map(|_| next()).collect();
        // Random upstream gradient (so we test the full vector-Jacobian product,
        // not just sum-reduction).
        let grad_out_host: Vec<f32> = (0..rows * channels).map(|_| next()).collect();

        // Analytic gradient via the composite (CPU, f32 — exactly what ships).
        let grad_out_t = Tensor::from_vec(grad_out_host.clone(), vec![rows, channels]).unwrap();
        let weight_t = Tensor::from_vec(weight.clone(), vec![channels, kernel]).unwrap();
        let din_t =
            causal_depthwise_conv1d_bwd_input_composite(&grad_out_t, &weight_t, kernel).unwrap();
        let din_analytic: Vec<f32> = din_t.to_vec::<f32>().unwrap();

        // Central finite-difference of L(x) = sum_{r,c} grad_out[r,c] * fwd(x)[r,c]
        // (so dL/dx[s] is exactly the input gradient under upstream grad_out).
        // The FD reference is computed in f64 so the comparison probes the
        // analytic derivation, not f32 round-off at the FD step size.
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let weight64: Vec<f64> = weight.iter().map(|&v| v as f64).collect();
        let grad64: Vec<f64> = grad_out_host.iter().map(|&v| v as f64).collect();
        let h = 1e-3f64;
        let loss = |xv: &[f64]| -> f64 {
            let out = forward_ref(xv, &weight64, rows, channels, kernel);
            out.iter()
                .zip(grad64.iter())
                .map(|(o, g)| o * g)
                .sum::<f64>()
        };

        let mut din_fd = vec![0.0f32; rows * channels];
        for i in 0..rows * channels {
            let mut xp = x64.clone();
            let mut xm = x64.clone();
            xp[i] += h;
            xm[i] -= h;
            din_fd[i] = ((loss(&xp) - loss(&xm)) / (2.0 * h)) as f32;
        }

        // Tight relative tolerance.
        let mut max_abs_err = 0.0f32;
        let mut max_rel_err = 0.0f32;
        for i in 0..rows * channels {
            let a = din_analytic[i];
            let f = din_fd[i];
            let abs_err = (a - f).abs();
            let denom = f.abs().max(a.abs()).max(1e-4);
            let rel_err = abs_err / denom;
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);
        }
        assert!(
            max_rel_err < 1e-2,
            "conv1d bwd-input composite disagrees with finite difference: \
             max_abs_err={max_abs_err:.3e}, max_rel_err={max_rel_err:.3e}\n\
             analytic={din_analytic:?}\nfd={din_fd:?}"
        );
    }

    /// The composite must reproduce the CUDA kernel's exact index convention:
    /// `din[s,c] = sum_k w[c,k] * grad_out[s+(K-1)-k, c]` (state rows clamped).
    #[test]
    fn causal_conv1d_bwd_input_matches_explicit_index_loop() {
        let rows = 6usize;
        let channels = 3usize;
        let kernel = 4usize;
        let state_rows = kernel - 1;

        let mut seed = 0xdead_beef_cafe_babeu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        let grad_out: Vec<f32> = (0..rows * channels).map(|_| next()).collect();
        let weight: Vec<f32> = (0..channels * kernel).map(|_| next()).collect();

        // Reference: the exact CUDA bwd-input kernel loop.
        let mut din_ref = vec![0.0f32; rows * channels];
        for input_row in 0..rows {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for j in 0..kernel {
                    let out_row = state_rows as isize + input_row as isize - j as isize;
                    if out_row >= 0 && (out_row as usize) < rows {
                        acc += grad_out[out_row as usize * channels + c] * weight[c * kernel + j];
                    }
                }
                din_ref[input_row * channels + c] = acc;
            }
        }

        let grad_out_t = Tensor::from_vec(grad_out, vec![rows, channels]).unwrap();
        let weight_t = Tensor::from_vec(weight, vec![channels, kernel]).unwrap();
        let din_t =
            causal_depthwise_conv1d_bwd_input_composite(&grad_out_t, &weight_t, kernel).unwrap();
        let din: Vec<f32> = din_t.to_vec::<f32>().unwrap();

        for i in 0..rows * channels {
            assert!(
                (din[i] - din_ref[i]).abs() < 1e-5,
                "index {i}: composite {} != kernel-loop {}",
                din[i],
                din_ref[i]
            );
        }
    }
}
