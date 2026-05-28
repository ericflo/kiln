//! `CrossEntropyKtBackward` — DEVICE-AGNOSTIC gradient of the softmax+NLL
//! cross-entropy loss (`kiln_tensor::ops::cross_entropy`).
//!
//! Forward: `logits [B, V]` (F32/BF16/F16), `targets [B]` (I64/U32),
//! `loss = mean_b(log_sum_exp(logits[b]) - logits[b, target[b]])`.
//!
//! Backward (canonical shift-by-one-hot):
//!
//! ```text
//! d_logits[b, v] = grad_output * (softmax(logits[b])[v] - 1_{v == target[b]}) / B
//! d_targets      = None
//! ```
//!
//! # Why a second CE backward (vs `CrossEntropyBackward`)
//!
//! The legacy `CrossEntropyBackward` downcasts its saved `logits`/`targets`
//! and `grad_output` to `CpuStorage` and runs a host-side softmax loop —
//! so it cannot execute on a CUDA-resident `[B, V]` logits tensor (the
//! tape walk surfaces the saved CUDA tensors directly). For the #1082
//! tape-forward loss adapter we need the gradient to stay on-device.
//!
//! This op composes existing **device-agnostic** core ops
//! (`softmax_last_dim`, `one_hot`, `sub`, `mul_scalar`) so it runs on
//! CPU/CUDA/Vulkan/Metal with the `[B, V]` activations never leaving the
//! device. The ONLY host touch is reading the rank-0 scalar grad
//! multiplier (one element) — there is no `mul` by a scalar *tensor* in
//! the kt op surface, so we read the scalar and fold it into `mul_scalar`.

use kiln_tensor::ops::{mul_scalar, one_hot, softmax_last_dim, sub};
use kiln_tensor::{bail, CpuStorage, DType, Device, Error, Result, Tensor};

use crate::BackwardOp;

#[derive(Debug)]
pub struct CrossEntropyKtBackward {
    /// Saved `logits [B, V]` from the forward pass.
    pub logits: Tensor,
    /// Saved `targets [B]` (I64/U32 class indices).
    pub targets: Tensor,
}

impl BackwardOp for CrossEntropyKtBackward {
    fn name(&self) -> &'static str {
        "cross_entropy_kt_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if self.logits.rank() != 2 {
            bail!(
                "CrossEntropyKtBackward: logits must be rank-2 [B, V], got {:?}",
                self.logits.shape()
            );
        }
        if self.targets.rank() != 1 {
            bail!(
                "CrossEntropyKtBackward: targets must be rank-1 [B], got {:?}",
                self.targets.shape()
            );
        }
        let batch = self.logits.shape()[0];
        let vocab = self.logits.shape()[1];
        if self.targets.shape()[0] != batch {
            bail!(
                "CrossEntropyKtBackward: targets [{}] must match logits batch {batch}",
                self.targets.shape()[0]
            );
        }

        // p = softmax(logits) over the vocab axis (device-resident).
        let sm = softmax_last_dim(&self.logits)?;
        // one_hot(targets) in the same dtype, so `sub` is exact-shape/dtype.
        // NOTE: kt `one_hot` has no device-resident kernel yet — it
        // D2H-copies the indices and materializes on host. Move the result
        // to the logits' device so the subtraction stays on-device. For
        // large vocab a follow-up should replace one_hot with an axis-0
        // flat `scatter_add` (the CUDA scatter path supports axis=0) to
        // avoid the [B, V] host materialization entirely.
        let oh = one_hot(&self.targets, vocab, sm.dtype())?;
        let oh = if oh.device() == sm.device() {
            oh
        } else {
            oh.to_device(sm.device())?
        };
        // (p - one_hot), device-resident.
        let diff = sub(&sm, &oh)?;
        // Scalar grad multiplier — the one unavoidable host touch (1 elem).
        let g = read_scalar_f32(grad_output)?;
        let d_logits = mul_scalar(&diff, g / (batch as f32))?;

        Ok(vec![Some(d_logits), None])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // cos/sin-style: backward reads the SAVED logits/targets (struct
        // fields), not the tape's retained inputs.
        false
    }
}

/// Read a rank-0/1-element tensor as an `f32`, moving to host if needed.
/// This is O(1) — only the scalar grad multiplier, never the `[B, V]`
/// activations.
fn read_scalar_f32(t: &Tensor) -> Result<f32> {
    let host = if t.device() == Device::Cpu {
        t.clone()
    } else {
        t.to_device(Device::Cpu)?
    };
    let cpu = host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| {
            Error::from_str("CrossEntropyKtBackward: grad_output must materialize to CpuStorage")
        })?;
    let bytes = cpu.as_bytes();
    Ok(match t.dtype() {
        DType::F32 => f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        DType::BF16 => half::bf16::from_le_bytes(bytes[0..2].try_into().unwrap()).to_f32(),
        DType::F16 => half::f16::from_le_bytes(bytes[0..2].try_into().unwrap()).to_f32(),
        other => bail!(
            "CrossEntropyKtBackward: grad_output dtype must be F32/BF16/F16, got {other}"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::ops::cross_entropy;
    use kiln_tensor::{CpuStorage, Tensor};

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn analytic(logits: &[f32], targets: &[u32], batch: usize, vocab: usize, g: f32) -> Vec<f32> {
        let mut out = vec![0f32; batch * vocab];
        for b in 0..batch {
            let row = &logits[b * vocab..(b + 1) * vocab];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = row.iter().map(|&x| (x - m).exp()).sum();
            for v in 0..vocab {
                let sm = (row[v] - m).exp() / denom;
                let oh = if v == targets[b] as usize { 1.0 } else { 0.0 };
                out[b * vocab + v] = g * (sm - oh) / (batch as f32);
            }
        }
        out
    }

    #[test]
    fn matches_analytic_gradient() {
        let (batch, vocab) = (4usize, 8usize);
        let logits: Vec<f32> = (0..batch * vocab).map(|i| ((i % 5) as f32) * 0.5 - 1.0).collect();
        let targets: Vec<u32> = (0..batch).map(|b| ((b * 3 + 1) % vocab) as u32).collect();
        let lt = Tensor::from_slice(&logits, vec![batch, vocab]).unwrap();
        let tt = Tensor::from_slice(&targets, vec![batch]).unwrap();
        let bo = CrossEntropyKtBackward {
            logits: lt,
            targets: tt,
        };
        // grad_output = scalar 1.0
        let g = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
        let grads = bo.apply(&g).unwrap();
        assert_eq!(grads.len(), 2);
        assert!(grads[1].is_none(), "targets non-differentiable");
        let got = read_f32(grads[0].as_ref().unwrap());
        let want = analytic(&logits, &targets, batch, vocab, 1.0);
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "idx {i}: got {a}, want {b}");
        }
    }

    #[test]
    fn finite_difference() {
        // d(loss)/d(logits) via central differences vs the analytic op.
        let (batch, vocab) = (3usize, 6usize);
        let logits: Vec<f32> = (0..batch * vocab).map(|i| 0.2 * (i as f32) - 0.5).collect();
        let targets: Vec<u32> = (0..batch).map(|b| ((b * 2) % vocab) as u32).collect();
        let tt = Tensor::from_slice(&targets, vec![batch]).unwrap();
        let bo = CrossEntropyKtBackward {
            logits: Tensor::from_slice(&logits, vec![batch, vocab]).unwrap(),
            targets: tt.clone(),
        };
        let g = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
        let dx = read_f32(bo.apply(&g).unwrap()[0].as_ref().unwrap());

        let loss = |lv: &[f32]| -> f32 {
            let lt = Tensor::from_slice(lv, vec![batch, vocab]).unwrap();
            let l = cross_entropy(&lt, &tt).unwrap();
            read_f32(&l)[0]
        };
        let step = 1e-3;
        for j in 0..batch * vocab {
            let mut up = logits.clone();
            up[j] += step;
            let mut dn = logits.clone();
            dn[j] -= step;
            let fd = (loss(&up) - loss(&dn)) / (2.0 * step);
            assert!((dx[j] - fd).abs() < 2e-3, "idx {j}: analytic {} vs fd {fd}", dx[j]);
        }
    }

    #[test]
    fn op_metadata() {
        let bo = CrossEntropyKtBackward {
            logits: Tensor::from_slice(&[0.0f32, 1.0], vec![1, 2]).unwrap(),
            targets: Tensor::from_slice(&[0u32], vec![1]).unwrap(),
        };
        assert_eq!(bo.name(), "cross_entropy_kt_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(!bo.requires_input(0));
    }
}
