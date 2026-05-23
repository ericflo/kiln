//! `log_softmax_last_dim` — log of softmax along the trailing axis.
//!
//! Numerically stable formulation: `log_softmax = x - log_sum_exp(x)`
//! after the max-subtraction step. Used together with `nll_loss` as
//! the two-step alternative to `cross_entropy`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn log_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    if x.rank() == 0 {
        bail!("log_softmax_last_dim: input must have rank ≥ 1");
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "log_softmax_last_dim: dtype must be F32/BF16/F16, got {}",
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("log_softmax_last_dim: input must be contiguous");
    }
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let last = *shape.last().unwrap();
    let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    let per = dtype.size_in_bytes();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("log_softmax: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let mut out = vec![0u8; outer * last * per];

    for r in 0..outer {
        // Load row as F32.
        let mut row = Vec::with_capacity(last);
        for i in 0..last {
            let idx = r * last + i;
            row.push(match dtype {
                DType::F32 => f32::from_le_bytes(bytes[idx * 4..idx * 4 + 4].try_into().unwrap()),
                DType::BF16 => half::bf16::from_le_bytes(
                    bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            });
        }
        // Stable log-sum-exp.
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_e = 0.0_f32;
        for &v in &row {
            sum_e += (v - m).exp();
        }
        let lse = m + sum_e.ln();
        // log_softmax_i = x_i - lse
        for i in 0..last {
            let y = row[i] - lse;
            let idx = r * last + i;
            match dtype {
                DType::F32 => out[idx * 4..idx * 4 + 4].copy_from_slice(&y.to_le_bytes()),
                DType::BF16 => out[idx * 2..idx * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
                DType::F16 => out[idx * 2..idx * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
                _ => unreachable!(),
            }
        }
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(shape), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "idx {i}: got {x}, want {y}");
        }
    }

    #[test]
    fn log_softmax_uniform() {
        // x = zeros[3]; softmax = [1/3]*3; log_softmax = log(1/3) each.
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let y = read_f32(&log_softmax_last_dim(&x).unwrap());
        approx(&y, &[(1.0_f32 / 3.0).ln(); 3], 1e-5);
    }

    #[test]
    fn log_softmax_sums_to_log_probs() {
        // The result exponentiated should sum to 1 per row.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = read_f32(&log_softmax_last_dim(&x).unwrap());
        let probs_sum: f32 = y.iter().map(|v| v.exp()).sum();
        assert!((probs_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn log_softmax_compose_with_nll_matches_cross_entropy() {
        use crate::ops::{cross_entropy, nll_loss};
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5], vec![2, 3]).unwrap();
        let targets = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let ce = cross_entropy(&logits, &targets).unwrap();
        let log_p = log_softmax_last_dim(&logits).unwrap();
        let nll = nll_loss(&log_p, &targets).unwrap();
        // ce and nll(log_softmax) should match within float tolerance.
        let cpu_ce = ce.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let cpu_nll = nll.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let ce_v = f32::from_le_bytes(cpu_ce.as_bytes()[..4].try_into().unwrap());
        let nll_v = f32::from_le_bytes(cpu_nll.as_bytes()[..4].try_into().unwrap());
        assert!((ce_v - nll_v).abs() < 1e-5, "ce={ce_v}, nll={nll_v}");
    }

    #[test]
    fn log_softmax_numerically_stable_with_large_logits() {
        // x = [1000, 1001, 1002] — overflow without max-subtraction.
        let x = Tensor::from_slice(&[1000.0f32, 1001.0, 1002.0], vec![1, 3]).unwrap();
        let y = read_f32(&log_softmax_last_dim(&x).unwrap());
        for v in &y {
            assert!(v.is_finite(), "log_softmax produced non-finite: {v}");
        }
    }

    #[test]
    fn log_softmax_bf16_round_trip() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![1, 3]).unwrap();
        let y = log_softmax_last_dim(&x).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }
}
