//! `addmm` — fused bias + matmul (`beta * c + alpha * (a @ b)`).
//!
//! PyTorch parity with `torch.addmm(c, a, b, alpha=1.0, beta=1.0)`.
//! `c` is broadcast over the output (typically a bias vector). Routes
//! through the existing `matmul` reference and the
//! `add` / `mul_scalar` elementwise ops, so the parity contract
//! inherits from those.
//!
//! Used by: linear-layer forward (already covered by [`crate::ops::linear`]
//! but `addmm` is the lower-level primitive RL/SFT trainers reach for
//! directly), attention output projection with bias, debug/eval
//! receipts.

use crate::{bail, ops::matmul, Result, Tensor};

/// `out = beta * c + alpha * (a @ b)`.
///
/// Shape requirements:
/// - `a`: `[M, K]`
/// - `b`: `[K, N]`
/// - `c`: shape that's broadcastable to `[M, N]`. For now we accept
///   exact `[M, N]` or rank-1 `[N]` (the common bias case).
pub fn addmm(c: &Tensor, a: &Tensor, b: &Tensor, alpha: f32, beta: f32) -> Result<Tensor> {
    if a.rank() != 2 || b.rank() != 2 {
        bail!(
            "addmm: a and b must be rank-2, got {} and {}",
            a.rank(),
            b.rank()
        );
    }
    let m = a.shape()[0];
    let k_a = a.shape()[1];
    let k_b = b.shape()[0];
    let n = b.shape()[1];
    if k_a != k_b {
        bail!(
            "addmm: inner dim mismatch — a is [{m}, {k_a}], b is [{k_b}, {n}]"
        );
    }

    // Compute the product first.
    let ab = matmul(a, b)?;
    let ab_scaled = if (alpha - 1.0).abs() < f32::EPSILON {
        ab
    } else {
        crate::ops::mul_scalar(&ab, alpha)?
    };

    // Broadcast c to [M, N].
    let c_bn: Tensor = match c.rank() {
        2 => {
            if c.shape() != [m, n] {
                bail!(
                    "addmm: c shape {:?} does not match [{m}, {n}]",
                    c.shape()
                );
            }
            c.clone()
        }
        1 => {
            if c.shape() != [n] {
                bail!(
                    "addmm: rank-1 c must have shape [{n}], got {:?}",
                    c.shape()
                );
            }
            // Broadcast [N] → [M, N] via repeat axis 0.
            crate::ops::repeat(&c.reshape(vec![1, n])?, 0, m)?
        }
        other => bail!("addmm: c must be rank-1 or rank-2, got rank {other}"),
    };
    let c_scaled = if (beta - 1.0).abs() < f32::EPSILON {
        c_bn
    } else {
        crate::ops::mul_scalar(&c_bn, beta)?
    };
    crate::ops::add(&c_scaled, &ab_scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuStorage, Tensor};

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn addmm_default_alpha_beta() {
        // a = [[1, 2], [3, 4]], b = [[5, 6], [7, 8]] → ab = [[19, 22], [43, 50]]
        // c = [[1, 1], [1, 1]] → out = c + ab = [[20, 23], [44, 51]]
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let y = addmm(&c, &a, &b, 1.0, 1.0).unwrap();
        assert_eq!(read_f32(&y), vec![20.0, 23.0, 44.0, 51.0]);
    }

    #[test]
    fn addmm_alpha_scales_product() {
        // alpha=2 doubles the product term.
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let y = addmm(&c, &a, &b, 2.0, 1.0).unwrap();
        assert_eq!(read_f32(&y), vec![38.0, 44.0, 86.0, 100.0]);
    }

    #[test]
    fn addmm_beta_scales_bias() {
        // beta=0 zeroes the bias term, so output is just alpha * (a @ b).
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = Tensor::from_slice(&[100.0f32, 100.0, 100.0, 100.0], vec![2, 2]).unwrap();
        let y = addmm(&c, &a, &b, 1.0, 0.0).unwrap();
        assert_eq!(read_f32(&y), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn addmm_rank1_bias_broadcasts() {
        // c is [N] = [2]; broadcasts to every row.
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let y = addmm(&c, &a, &b, 1.0, 1.0).unwrap();
        // ab = [[19, 22], [43, 50]]; broadcast bias = [[10, 20], [10, 20]]
        assert_eq!(read_f32(&y), vec![29.0, 42.0, 53.0, 70.0]);
    }

    #[test]
    fn addmm_inner_dim_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0], vec![2, 1]).unwrap();
        let c = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let e = addmm(&c, &a, &b, 1.0, 1.0).unwrap_err();
        assert!(e.to_string().contains("inner dim"));
    }

    #[test]
    fn addmm_c_shape_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::from_slice(&[3.0f32, 4.0], vec![2, 1]).unwrap();
        // c should be [1, 1] or [1] but provide [2, 2]
        let c = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let e = addmm(&c, &a, &b, 1.0, 1.0).unwrap_err();
        assert!(e.to_string().contains("does not match"));
    }

    #[test]
    fn addmm_rank2_a_required() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let c = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let e = addmm(&c, &a, &b, 1.0, 1.0).unwrap_err();
        assert!(e.to_string().contains("rank-2"));
    }
}
