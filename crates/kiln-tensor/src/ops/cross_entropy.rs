//! `cross_entropy` — softmax + negative-log-likelihood loss.
//!
//! The canonical training loss for next-token prediction. Mirrors
//! `candle_nn::loss::cross_entropy` and `kiln-flce-kernel`'s
//! FLCE (fused linear cross-entropy) at the Phase 6b backward target.
//!
//! # Semantics
//!
//! Given:
//! - `logits: Tensor` — shape `[batch, vocab]`, F32 / BF16 / F16
//! - `targets: Tensor` — shape `[batch]`, dtype I64 or U32
//!
//! Returns:
//! - `loss: Tensor` — rank-0 scalar = `mean_over_batch(-log P(target))`
//!
//! Where `P` is the softmax of logits along the vocab axis.
//!
//! Numerically stable formulation:
//!
//! ```text
//! for each batch b:
//!     m = max_v logits[b, v]
//!     log_sum_exp = m + log(sum_v exp(logits[b, v] - m))
//!     loss_b = log_sum_exp - logits[b, targets[b]]
//! loss = mean_b loss_b
//! ```
//!
//! # Determinism
//!
//! `Constructive`. Per-batch fixed-tree reduction (the `max` + `sum`
//! passes) + a final F32 mean over the batch. Bit-identical at the
//! same input dtype.

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Cross-entropy loss op.
#[derive(Debug, Default, Clone, Copy)]
pub struct CrossEntropyOp;

impl DeviceOp2 for CrossEntropyOp {
    fn name(&self) -> &'static str {
        "cross_entropy"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, logits: &Tensor, targets: &Tensor) -> Result<Option<Tensor>> {
        // Defer to the canonical validation so shape/dtype errors are
        // surfaced as Errs (instead of silently falling back to the
        // CPU path, which would mask any CUDA-specific bug).
        validate(logits, targets)?;
        Ok(Some(crate::cuda_cross_entropy_loss(logits, targets)?))
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, logits: &Tensor, targets: &Tensor) -> Result<Option<Tensor>> {
        // Same precondition gates as cuda_fwd: validate shape / dtype
        // first so contract errors surface as Errs. After validation,
        // soft-fall through to the CPU path if either input lives on
        // a non-Vulkan device — the kernel only operates on
        // VkTensor-resident inputs.
        validate(logits, targets)?;
        if !matches!(logits.device(), crate::Device::Vulkan(_))
            || !matches!(targets.device(), crate::Device::Vulkan(_))
        {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Vulkan): implement
        // `crate::vulkan_cross_entropy_loss(logits, targets)`
        // analogous to `crate::cuda_cross_entropy_loss` above. Until
        // that wrapper lands, fall through to the CPU path (numerics-
        // correct, performance-wrong).
        // Candidate implementations:
        //   1. Reuse `kiln-vulkan-kernel::vk_ops::flce` — the existing
        //      "fused linear cross-entropy" kernel is the production
        //      training-loss hot path (logits=W@x fused with CE on
        //      device, never materializing the full vocab×batch
        //      logits tensor). For the simpler non-fused case (where
        //      `logits` is already materialized), a thin wrapper can
        //      skip the matmul half and reuse only the CE reduction.
        //   2. New stand-alone SPIR-V shader: per-batch threadgroup
        //      reduction over the vocab axis — first pass finds row
        //      max, second pass computes log-sum-exp + indexes
        //      `targets[b]` for the target logit. Final batch-mean is
        //      a single-pass reduction in the same kernel via shared
        //      memory. Reuse the row-reduction pattern from
        //      `vk_ops::softmax::dispatch_softmax_fwd`.
        //   3. Dtype matrix:
        //      - `logits`: F32 supported via existing VkDType::F32;
        //        BF16 supported via VkDType::Bf16 (verify flce kernel
        //        accepts BF16 logits — production training is BF16).
        //        F16 needs a new VkDType::F16 variant before any
        //        F16-native shader can land.
        //      - `targets`: I64 or U32 (see cpu_fwd's
        //        read_target_ids). Vulkan compute storage buffers
        //        are u32-aligned; I64 requires two-u32 packing.
        //        Prefer U32 targets at the dispatch boundary on
        //        Vulkan to avoid the packing cost.
        Ok(None)
    }

    fn cpu_fwd(&self, logits: &Tensor, targets: &Tensor) -> Result<Option<Tensor>> {
        validate(logits, targets)?;
        let logits_cpu = downcast_cpu(logits, "logits")?;
        let targets_cpu = downcast_cpu(targets, "targets")?;

        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];

        // Load targets as u64 (rejecting negatives on the I64 path).
        let target_ids = read_target_ids(targets, targets_cpu, batch)?;

        let dtype = logits.dtype();
        let per = dtype.size_in_bytes();

        let mut loss_sum = 0.0_f32;
        for b in 0..batch {
            let row = load_logits_row_f32(dtype, logits_cpu.as_bytes(), b, vocab, per)?;
            let max_logit = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            // If row is all -inf, the mean is undefined — bail explicitly.
            if !max_logit.is_finite() {
                bail!(
                    "CrossEntropyOp: row {b} has no finite logits (all -inf?); \
                     loss is undefined"
                );
            }
            let log_sum_exp = max_logit
                + row.iter().map(|&v| (v - max_logit).exp()).sum::<f32>().ln();
            let target = target_ids[b];
            if target >= vocab as u64 {
                bail!(
                    "CrossEntropyOp: target {target} out of range (vocab={vocab}) at position {b}"
                );
            }
            let target_logit = row[target as usize];
            loss_sum += log_sum_exp - target_logit;
        }
        let loss = loss_sum / batch as f32;

        let bytes = match dtype {
            DType::F32 => loss.to_le_bytes().to_vec(),
            DType::BF16 => half::bf16::from_f32(loss).to_le_bytes().to_vec(),
            DType::F16 => half::f16::from_f32(loss).to_le_bytes().to_vec(),
            _ => unreachable!("validate rejects"),
        };
        let cpu = CpuStorage::from_bytes(dtype, bytes)?;
        let storage: Storage = Arc::new(cpu);
        Ok(Some(Tensor::from_parts(
            storage,
            Layout::contiguous(Vec::<usize>::new()),
            TensorId::next(),
        )?))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// `loss = -mean_b log P(target[b])` where `P = softmax(logits)`.
///
/// `logits: [batch, vocab]`, `targets: [batch]` (I64 or U32).
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    dispatch2(&CrossEntropyOp, logits, targets)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(logits: &Tensor, targets: &Tensor) -> Result<()> {
    if logits.rank() != 2 {
        bail!(
            "CrossEntropyOp: logits must be rank-2 [batch, vocab], got shape {:?}",
            logits.shape()
        );
    }
    if targets.rank() != 1 {
        bail!(
            "CrossEntropyOp: targets must be rank-1 [batch], got shape {:?}",
            targets.shape()
        );
    }
    if logits.shape()[0] != targets.shape()[0] {
        bail!(
            "CrossEntropyOp: batch mismatch — logits has batch={}, targets has batch={}",
            logits.shape()[0],
            targets.shape()[0]
        );
    }
    if !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "CrossEntropyOp: logits dtype must be F32/BF16/F16, got {}",
            logits.dtype()
        );
    }
    if !matches!(targets.dtype(), DType::I64 | DType::U32) {
        bail!(
            "CrossEntropyOp: targets dtype must be I64/U32, got {}",
            targets.dtype()
        );
    }
    if !logits.is_contiguous() || !targets.is_contiguous() {
        bail!("CrossEntropyOp: both inputs must be contiguous");
    }
    if logits.shape()[0] == 0 {
        bail!("CrossEntropyOp: batch dim is 0 — mean is undefined");
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("CrossEntropyOp: {label} storage must be CpuStorage")))
}

fn read_target_ids(targets: &Tensor, cpu: &CpuStorage, batch: usize) -> Result<Vec<u64>> {
    let bytes = cpu.as_bytes();
    let mut out = Vec::with_capacity(batch);
    match targets.dtype() {
        DType::I64 => {
            if bytes.len() < batch * 8 {
                bail!(
                    "CrossEntropyOp: targets buffer too small ({} < {})",
                    bytes.len(),
                    batch * 8
                );
            }
            for i in 0..batch {
                let v = i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
                if v < 0 {
                    bail!("CrossEntropyOp: negative target {v} at position {i}");
                }
                out.push(v as u64);
            }
        }
        DType::U32 => {
            if bytes.len() < batch * 4 {
                bail!(
                    "CrossEntropyOp: targets buffer too small ({} < {})",
                    bytes.len(),
                    batch * 4
                );
            }
            for i in 0..batch {
                let v = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                out.push(v as u64);
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

fn load_logits_row_f32(
    dtype: DType,
    bytes: &[u8],
    row: usize,
    vocab: usize,
    per: usize,
) -> Result<Vec<f32>> {
    let start = row * vocab * per;
    let end = start + vocab * per;
    if bytes.len() < end {
        bail!(
            "CrossEntropyOp: logits buffer too small for row {row} (have {}, need {})",
            bytes.len(),
            end
        );
    }
    let raw = &bytes[start..end];
    let mut out = Vec::with_capacity(vocab);
    match dtype {
        DType::F32 => {
            for i in 0..vocab {
                out.push(f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()));
            }
        }
        DType::BF16 => {
            for i in 0..vocab {
                out.push(
                    half::bf16::from_le_bytes(raw[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        DType::F16 => {
            for i in 0..vocab {
                out.push(
                    half::f16::from_le_bytes(raw[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_scalar_f32(t: &Tensor) -> f32 {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[0..4].try_into().unwrap())
    }

    fn approx(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    #[test]
    fn ce_perfect_prediction_is_close_to_zero() {
        // Logits with a single +∞-ish peak at the target → log P ≈ 0
        // → loss ≈ 0.
        let logits = Tensor::from_slice(
            &[100.0f32, 0.0, 0.0, 100.0],
            vec![2, 2],
        )
        .unwrap();
        let targets = Tensor::from_slice(&[0i64, 1], vec![2]).unwrap();
        let loss = cross_entropy(&logits, &targets).unwrap();
        assert!(loss.rank() == 0);
        let v = read_scalar_f32(&loss);
        assert!(v < 1e-10, "loss should be near 0, got {v}");
    }

    #[test]
    fn ce_uniform_logits_gives_log_vocab() {
        // Uniform logits → P = 1/vocab → -log(1/vocab) = log(vocab).
        let vocab = 4;
        let batch = 3;
        let logits =
            Tensor::from_slice(&vec![0.0f32; batch * vocab], vec![batch, vocab]).unwrap();
        let targets = Tensor::from_slice(&[0i64, 1, 2], vec![batch]).unwrap();
        let loss = cross_entropy(&logits, &targets).unwrap();
        let v = read_scalar_f32(&loss);
        let expected = (vocab as f32).ln();
        assert!(approx(v, expected, 1e-6), "expected {expected}, got {v}");
    }

    #[test]
    fn ce_handles_extreme_logits_no_overflow() {
        // Logits include very large values; max-subtraction should
        // prevent overflow.
        let logits = Tensor::from_slice(&[1000.0f32, -1000.0, 0.0], vec![1, 3]).unwrap();
        let targets = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let loss = cross_entropy(&logits, &targets).unwrap();
        let v = read_scalar_f32(&loss);
        assert!(v.is_finite(), "loss should be finite, got {v}");
        // log P(0) ≈ 0 since one logit dominates.
        assert!(v < 1e-6);
    }

    #[test]
    fn ce_bf16_path_within_tolerance() {
        // Uniform logits in BF16: -log(1/4) = log 4.
        let vocab = 4;
        let logits_bf16: Vec<half::bf16> = vec![half::bf16::from_f32(0.0); vocab];
        let logits = Tensor::from_slice(&logits_bf16, vec![1, vocab]).unwrap();
        let targets = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let loss = cross_entropy(&logits, &targets).unwrap();
        let cpu = loss
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap();
        let v = half::bf16::from_le_bytes(cpu.as_bytes()[0..2].try_into().unwrap()).to_f32();
        let expected = (vocab as f32).ln();
        assert!(approx(v, expected, 1e-2));
    }

    #[test]
    fn ce_u32_targets_path() {
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let targets = Tensor::from_slice(&[0u32, 1], vec![2]).unwrap();
        let loss = cross_entropy(&logits, &targets).unwrap();
        let v = read_scalar_f32(&loss);
        let expected = (2.0_f32).ln();
        assert!(approx(v, expected, 1e-6));
    }

    #[test]
    fn ce_rejects_out_of_range_target() {
        let logits = Tensor::from_slice(&[0.0f32, 0.0], vec![1, 2]).unwrap();
        let targets = Tensor::from_slice(&[5i64], vec![1]).unwrap();
        let e = cross_entropy(&logits, &targets).unwrap_err();
        assert!(e.to_string().contains("out of range"));
    }

    #[test]
    fn ce_rejects_negative_i64_target() {
        let logits = Tensor::from_slice(&[0.0f32, 0.0], vec![1, 2]).unwrap();
        let targets = Tensor::from_slice(&[-1i64], vec![1]).unwrap();
        let e = cross_entropy(&logits, &targets).unwrap_err();
        assert!(e.to_string().contains("negative target"));
    }

    #[test]
    fn ce_rejects_rank_mismatch() {
        let logits = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let targets = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let e = cross_entropy(&logits, &targets).unwrap_err();
        assert!(e.to_string().contains("rank-2"));
    }

    #[test]
    fn ce_rejects_batch_mismatch() {
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let targets = Tensor::from_slice(&[0i64, 1, 0], vec![3]).unwrap();
        let e = cross_entropy(&logits, &targets).unwrap_err();
        assert!(e.to_string().contains("batch mismatch"));
    }

    #[test]
    fn ce_rejects_bad_target_dtype() {
        let logits = Tensor::from_slice(&[0.0f32, 0.0], vec![1, 2]).unwrap();
        let targets = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let e = cross_entropy(&logits, &targets).unwrap_err();
        assert!(e.to_string().contains("I64/U32"));
    }

    #[test]
    fn op_metadata() {
        let op = CrossEntropyOp;
        assert_eq!(op.name(), "cross_entropy");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
    }
}
