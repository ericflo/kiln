//! `CrossEntropyBackward` — gradient of the softmax + NLL loss.
//!
//! Forward (from `kiln_tensor::ops::cross_entropy`):
//! - `logits: [B, V]` F32/BF16/F16
//! - `targets: [B]` I64 or U32
//! - `loss: scalar = mean_b(log_sum_exp(logits[b]) - logits[b, target[b]])`
//!
//! Backward: given `grad_output: scalar = dLoss/dLoss = 1` (or any
//! upstream multiplier), produces:
//!
//! ```text
//! d_logits[b, v] = grad_output * (softmax(logits[b])[v] - 1_{v == target[b]}) / B
//! d_targets      = None   (non-differentiable)
//! ```
//!
//! The gradient is **the canonical "shift-by-one-hot" form** used in
//! every cross-entropy backward implementation — clean, F32-stable,
//! numerically conditioned by the same `log_sum_exp` max-subtraction
//! used in the forward.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct CrossEntropyBackward {
    /// Saved `logits` from the forward pass.
    pub logits: Tensor,
    /// Saved `targets` from the forward pass.
    pub targets: Tensor,
}

impl BackwardOp for CrossEntropyBackward {
    fn name(&self) -> &'static str {
        "cross_entropy_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if self.logits.rank() != 2 {
            bail!(
                "CrossEntropyBackward: logits must be rank-2 [B, V], got {:?}",
                self.logits.shape()
            );
        }
        if !self.logits.is_contiguous() {
            bail!("CrossEntropyBackward: logits must be contiguous");
        }
        if !matches!(self.logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "CrossEntropyBackward: logits dtype must be F32/BF16/F16, got {}",
                self.logits.dtype()
            );
        }
        if !matches!(self.targets.dtype(), DType::I64 | DType::U32) {
            bail!(
                "CrossEntropyBackward: targets dtype must be I64/U32, got {}",
                self.targets.dtype()
            );
        }
        if grad_output.rank() != 0 {
            bail!(
                "CrossEntropyBackward: grad_output must be scalar (rank 0), got rank {}",
                grad_output.rank()
            );
        }

        let batch = self.logits.shape()[0];
        let vocab = self.logits.shape()[1];

        let logits_cpu = self
            .logits
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("CrossEntropyBackward: logits storage must be CpuStorage"))?;
        let logits_bytes = logits_cpu.as_bytes();
        let dtype = self.logits.dtype();
        let per = dtype.size_in_bytes();

        // Read grad_output.
        let go_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("CrossEntropyBackward: grad_output storage must be CpuStorage"))?;
        let go_bytes = go_cpu.as_bytes();
        if grad_output.dtype() != dtype {
            bail!(
                "CrossEntropyBackward: grad_output dtype {} must match logits dtype {}",
                grad_output.dtype(),
                dtype
            );
        }
        let go_val = match dtype {
            DType::F32 => f32::from_le_bytes(go_bytes[..4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(go_bytes[..2].try_into().unwrap()).to_f32(),
            DType::F16 => half::f16::from_le_bytes(go_bytes[..2].try_into().unwrap()).to_f32(),
            _ => unreachable!(),
        };

        // Read targets.
        let t_cpu = self
            .targets
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("CrossEntropyBackward: targets storage must be CpuStorage"))?;
        let t_bytes = t_cpu.as_bytes();
        let targets: Vec<u64> = match self.targets.dtype() {
            DType::I64 => (0..batch)
                .map(|i| {
                    i64::from_le_bytes(t_bytes[i * 8..i * 8 + 8].try_into().unwrap()) as u64
                })
                .collect(),
            DType::U32 => (0..batch)
                .map(|i| {
                    u32::from_le_bytes(t_bytes[i * 4..i * 4 + 4].try_into().unwrap()) as u64
                })
                .collect(),
            _ => unreachable!(),
        };

        let scale = go_val / batch as f32;
        let mut dx = vec![0.0f32; batch * vocab];

        for b in 0..batch {
            // Load row as F32.
            let mut row = Vec::with_capacity(vocab);
            let start = b * vocab * per;
            for v in 0..vocab {
                row.push(match dtype {
                    DType::F32 => f32::from_le_bytes(
                        logits_bytes[start + v * 4..start + v * 4 + 4].try_into().unwrap(),
                    ),
                    DType::BF16 => half::bf16::from_le_bytes(
                        logits_bytes[start + v * 2..start + v * 2 + 2].try_into().unwrap(),
                    )
                    .to_f32(),
                    DType::F16 => half::f16::from_le_bytes(
                        logits_bytes[start + v * 2..start + v * 2 + 2].try_into().unwrap(),
                    )
                    .to_f32(),
                    _ => unreachable!(),
                });
            }
            // Softmax (numerically stable).
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let target = targets[b];
            if target as usize >= vocab {
                bail!(
                    "CrossEntropyBackward: target {target} out of range (vocab={vocab})"
                );
            }
            for v in 0..vocab {
                let p = exps[v] / sum_e;
                let one_hot = if v as u64 == target { 1.0 } else { 0.0 };
                dx[b * vocab + v] = scale * (p - one_hot);
            }
        }

        // Store d_logits back into the input dtype.
        let mut out = vec![0u8; dx.len() * per];
        match dtype {
            DType::F32 => {
                for (i, v) in dx.iter().enumerate() {
                    out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            DType::BF16 => {
                for (i, v) in dx.iter().enumerate() {
                    out[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
                }
            }
            DType::F16 => {
                for (i, v) in dx.iter().enumerate() {
                    out[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        let cpu = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu);
        let d_logits = Tensor::from_parts(
            storage,
            Layout::contiguous(self.logits.shape().to_vec()),
            TensorId::next(),
        )?;

        Ok(vec![Some(d_logits), None /* targets non-differentiable */])
    }
    fn requires_input(&self, idx: usize) -> bool {
        // logits and targets are both saved on the struct; the tape
        // doesn't need to preserve them separately.
        idx == 0 || idx == 1
    }
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

    fn read_bf16(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect()
    }

    #[test]
    fn ce_backward_zero_logits_uniform_softmax() {
        // logits = zeros[1, 3] → softmax = [1/3, 1/3, 1/3].
        // target = 0 → d_logits = (1/B) * (softmax - [1,0,0])
        //                       = ( 1/3 - 1, 1/3, 1/3 )
        //                       = (-2/3,    1/3, 1/3 )
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let targets = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
        let bo = CrossEntropyBackward {
            logits,
            targets,
        };
        let grads = bo.apply(&grad).unwrap();
        let d = read_f32(grads[0].as_ref().unwrap());
        assert!((d[0] - (-2.0 / 3.0)).abs() < 1e-6);
        assert!((d[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((d[2] - (1.0 / 3.0)).abs() < 1e-6);
        // targets grad is None.
        assert!(grads[1].is_none());
    }

    #[test]
    fn ce_backward_batch_average() {
        // logits [2, 2], targets [0, 1]. d_logits[b] = (softmax - one_hot) / 2.
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let targets = Tensor::from_slice(&[0i64, 1], vec![2]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
        let bo = CrossEntropyBackward { logits, targets };
        let d = read_f32(bo.apply(&grad).unwrap()[0].as_ref().unwrap());
        // Each row: softmax = [0.5, 0.5]. divide by batch=2.
        // Row 0 (target 0): [(0.5-1)/2, 0.5/2] = [-0.25, 0.25]
        // Row 1 (target 1): [0.5/2, (0.5-1)/2] = [0.25, -0.25]
        assert!((d[0] - (-0.25)).abs() < 1e-6);
        assert!((d[1] - 0.25).abs() < 1e-6);
        assert!((d[2] - 0.25).abs() < 1e-6);
        assert!((d[3] - (-0.25)).abs() < 1e-6);
    }

    #[test]
    fn ce_backward_respects_grad_output_scaling() {
        // grad_output = 2.0 should double the gradient.
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let targets = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let grad = Tensor::from_slice(&[2.0f32], vec![]).unwrap();
        let bo = CrossEntropyBackward { logits, targets };
        let d = read_f32(bo.apply(&grad).unwrap()[0].as_ref().unwrap());
        assert!((d[0] - (-4.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn ce_backward_bf16_path() {
        let lv: Vec<half::bf16> = (0..3).map(|_| half::bf16::ZERO).collect();
        let logits = Tensor::from_slice(&lv, vec![1, 3]).unwrap();
        let targets = Tensor::from_slice(&[0u32], vec![1]).unwrap();
        let grad = Tensor::from_slice(&[half::bf16::ONE], vec![]).unwrap();
        let bo = CrossEntropyBackward { logits, targets };
        let d_t = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d_t.dtype(), DType::BF16);
        let d = read_bf16(&d_t);
        assert!((d[0] - (-2.0 / 3.0)).abs() < 1e-2);
    }

    #[test]
    fn ce_backward_rejects_non_scalar_grad() {
        let logits = Tensor::from_slice(&[0.0f32, 0.0], vec![1, 2]).unwrap();
        let targets = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let bad = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = CrossEntropyBackward { logits, targets };
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("scalar (rank 0)"));
    }

    #[test]
    fn op_metadata() {
        let logits = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let targets = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let bo = CrossEntropyBackward { logits, targets };
        assert_eq!(bo.name(), "cross_entropy_backward");
        assert_eq!(bo.input_count(), 2);
    }
}
