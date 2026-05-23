//! Additional loss functions: `mse_loss`, `nll_loss`, `l1_loss`,
//! `huber_loss`.
//!
//! `cross_entropy` lives in its own file (it's the workhorse for
//! language modeling); these are the rest of the common losses.
//!
//! All take rank-2 (or higher) inputs and return a rank-0 scalar
//! (mean reduction). `none` and `sum` reductions can be obtained by
//! composing with the relevant elementwise op + reduce_axis directly.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

fn validate_pair(a: &Tensor, b: &Tensor, name: &str) -> Result<()> {
    if a.shape() != b.shape() {
        bail!("{name}: shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    }
    if a.dtype() != b.dtype() {
        bail!("{name}: dtype mismatch");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", a.dtype());
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("{name}: inputs must be contiguous");
    }
    Ok(())
}

fn load_pair_f32(a: &Tensor, b: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
    let a_bytes = a
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("losses: storage must be CpuStorage"))?
        .as_bytes();
    let b_bytes = b
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("losses: storage must be CpuStorage"))?
        .as_bytes();
    let n = a.element_count();
    let dtype = a.dtype();
    let mut av = Vec::with_capacity(n);
    let mut bv = Vec::with_capacity(n);
    for i in 0..n {
        av.push(match dtype {
            DType::F32 => f32::from_le_bytes(a_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(a_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(a_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            _ => unreachable!(),
        });
        bv.push(match dtype {
            DType::F32 => f32::from_le_bytes(b_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(b_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(b_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            _ => unreachable!(),
        });
    }
    Ok((av, bv))
}

fn scalar_tensor(dtype: DType, v: f32) -> Result<Tensor> {
    let bytes = match dtype {
        DType::F32 => v.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(Vec::<usize>::new()), TensorId::next())
}

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    validate_pair(pred, target, "mse_loss")?;
    let (p, t) = load_pair_f32(pred, target)?;
    let n = p.len() as f32;
    let sum: f32 = p
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    scalar_tensor(pred.dtype(), sum / n)
}

pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    validate_pair(pred, target, "l1_loss")?;
    let (p, t) = load_pair_f32(pred, target)?;
    let n = p.len() as f32;
    let sum: f32 = p.iter().zip(t.iter()).map(|(&a, &b)| (a - b).abs()).sum();
    scalar_tensor(pred.dtype(), sum / n)
}

/// Huber loss with a configurable boundary `delta`. Returns the mean.
/// Quadratic when `|err| < delta`, linear (Lipschitz) otherwise.
pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f32) -> Result<Tensor> {
    if delta <= 0.0 {
        bail!("huber_loss: delta must be > 0, got {delta}");
    }
    validate_pair(pred, target, "huber_loss")?;
    let (p, t) = load_pair_f32(pred, target)?;
    let n = p.len() as f32;
    let sum: f32 = p
        .iter()
        .zip(t.iter())
        .map(|(&a, &b)| {
            let e = (a - b).abs();
            if e < delta {
                0.5 * e * e
            } else {
                delta * (e - 0.5 * delta)
            }
        })
        .sum();
    scalar_tensor(pred.dtype(), sum / n)
}

/// NLL loss for soft probabilities given log_probs + targets.
/// `log_probs: [B, V]`, `targets: [B]` (I64/U32). Returns -mean of
/// log_probs at the target indices.
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    if log_probs.rank() != 2 {
        bail!(
            "nll_loss: log_probs must be rank-2 [B, V], got {:?}",
            log_probs.shape()
        );
    }
    if targets.rank() != 1 {
        bail!(
            "nll_loss: targets must be rank-1 [B], got {:?}",
            targets.shape()
        );
    }
    if log_probs.shape()[0] != targets.shape()[0] {
        bail!(
            "nll_loss: batch mismatch: log_probs.B={} vs targets.B={}",
            log_probs.shape()[0],
            targets.shape()[0]
        );
    }
    if !matches!(targets.dtype(), DType::I64 | DType::U32) {
        bail!(
            "nll_loss: targets dtype must be I64/U32, got {}",
            targets.dtype()
        );
    }
    if !matches!(log_probs.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "nll_loss: log_probs dtype must be F32/BF16/F16, got {}",
            log_probs.dtype()
        );
    }
    let dtype = log_probs.dtype();
    let batch = log_probs.shape()[0];
    let vocab = log_probs.shape()[1];

    let lp_bytes = log_probs
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("nll_loss: log_probs storage must be CpuStorage"))?
        .as_bytes();
    let t_bytes = targets
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("nll_loss: targets storage must be CpuStorage"))?
        .as_bytes();

    let mut sum = 0.0_f32;
    for b in 0..batch {
        let tid = match targets.dtype() {
            DType::I64 => {
                i64::from_le_bytes(t_bytes[b * 8..b * 8 + 8].try_into().unwrap()) as i64
            }
            DType::U32 => u32::from_le_bytes(t_bytes[b * 4..b * 4 + 4].try_into().unwrap()) as i64,
            _ => unreachable!(),
        };
        if tid < 0 || tid as usize >= vocab {
            bail!("nll_loss: target {tid} out of range (vocab={vocab}) at row {b}");
        }
        let idx = b * vocab + tid as usize;
        let lp = match dtype {
            DType::F32 => f32::from_le_bytes(lp_bytes[idx * 4..idx * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(lp_bytes[idx * 2..idx * 2 + 2].try_into().unwrap())
                    .to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(lp_bytes[idx * 2..idx * 2 + 2].try_into().unwrap())
                    .to_f32()
            }
            _ => unreachable!(),
        };
        sum += -lp;
    }
    scalar_tensor(dtype, sum / batch as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32(t: &Tensor) -> f32 {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
    }

    #[test]
    fn mse_loss_known() {
        // pred = [1, 2]; target = [2, 4]; err = [1, 4]; mean = 2.5
        let p = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let t = Tensor::from_slice(&[2.0f32, 4.0], vec![2]).unwrap();
        assert!((scalar_f32(&mse_loss(&p, &t).unwrap()) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn l1_loss_known() {
        // pred = [1, 2, 3]; target = [3, 4, 5]; mean(|err|) = (2+2+2)/3 = 2
        let p = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let t = Tensor::from_slice(&[3.0f32, 4.0, 5.0], vec![3]).unwrap();
        assert!((scalar_f32(&l1_loss(&p, &t).unwrap()) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn huber_loss_quadratic_regime() {
        // delta = 1.0; small errors → quadratic.
        // pred = [1, 1]; target = [1.5, 0.5]; err = [-0.5, 0.5];
        // both |err| < delta → 0.5*e² = 0.125 each; mean = 0.125.
        let p = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let t = Tensor::from_slice(&[1.5f32, 0.5], vec![2]).unwrap();
        let h = scalar_f32(&huber_loss(&p, &t, 1.0).unwrap());
        assert!((h - 0.125).abs() < 1e-6);
    }

    #[test]
    fn huber_loss_linear_regime() {
        // delta = 1.0; large errors → linear.
        // pred = [0]; target = [3]; err = 3 → delta*(e - 0.5*delta) = 1*(3-0.5) = 2.5
        let p = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let t = Tensor::from_slice(&[3.0f32], vec![1]).unwrap();
        let h = scalar_f32(&huber_loss(&p, &t, 1.0).unwrap());
        assert!((h - 2.5).abs() < 1e-6);
    }

    #[test]
    fn huber_delta_zero_errors() {
        let p = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let t = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = huber_loss(&p, &t, 0.0).unwrap_err();
        assert!(e.to_string().contains("delta"));
    }

    #[test]
    fn nll_loss_uniform_log_probs() {
        // log_probs = log(1/3) for every position; vocab=3.
        // loss = -log(1/3) = ln(3) ≈ 1.0986.
        let lp = (1.0_f32 / 3.0).ln();
        let log_probs = Tensor::from_slice(&[lp, lp, lp, lp, lp, lp], vec![2, 3]).unwrap();
        let targets = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let loss = scalar_f32(&nll_loss(&log_probs, &targets).unwrap());
        assert!((loss - 3.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn nll_loss_perfect_prediction_zero_loss() {
        // log_probs[b, target[b]] = 0 (prob 1); rest = -inf-style
        // (we use a large negative value but only the target index
        // is read).
        let log_probs = Tensor::from_slice(
            &[0.0f32, -100.0, -100.0, -100.0, -100.0, 0.0],
            vec![2, 3],
        )
        .unwrap();
        let targets = Tensor::from_slice(&[0i64, 2], vec![2]).unwrap();
        let loss = scalar_f32(&nll_loss(&log_probs, &targets).unwrap());
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn nll_loss_rank_mismatch_errors() {
        let lp = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let t = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let e = nll_loss(&lp, &t).unwrap_err();
        assert!(e.to_string().contains("rank-2"));
    }
}
