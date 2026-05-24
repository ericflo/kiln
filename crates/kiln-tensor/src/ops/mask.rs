//! Mask helpers for the attention path: `masked_fill` + `causal_mask`.
//!
//! Replaces candle's `Tensor::where_cond` + `Tensor::tril` /
//! `Tensor::triu` at the pre-softmax mask call sites in attention.
//! Mirrors `kiln-vulkan-kernel::vk_ops::mask` for cross-backend
//! parity testing.
//!
//! # masked_fill
//!
//! ```text
//! out[i] = if mask[i] != 0 { fill_value } else { x[i] }
//! ```
//!
//! `x`, `mask`, `out` share shape. `mask` dtype is `U8`. `fill_value`
//! is `f32` (cast to `x`'s dtype on store).
//!
//! The "fill with -inf" pattern that nullifies positions before
//! softmax: `masked_fill(scores, mask, f32::NEG_INFINITY)`.
//!
//! # causal_mask
//!
//! Returns a `[seq_len, seq_len]` U8 mask where `mask[i, j] = 1`
//! iff `j > i` (positions strictly in the future). Apply via
//! `masked_fill(scores, causal_mask(seq_len), f32::NEG_INFINITY)`.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; no reduction.

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Masked-fill op handle. Carries the fill value.
#[derive(Debug, Clone, Copy)]
pub struct MaskedFillOp {
    fill_value: f32,
}

impl MaskedFillOp {
    pub const fn new(fill_value: f32) -> Self {
        MaskedFillOp { fill_value }
    }
    pub const fn fill_value(self) -> f32 {
        self.fill_value
    }
}

impl DeviceOp2 for MaskedFillOp {
    fn name(&self) -> &'static str {
        "masked_fill"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor, mask: &Tensor) -> Result<Option<Tensor>> {
        if x.shape() != mask.shape() {
            bail!(
                "MaskedFillOp: shape mismatch {:?} vs mask {:?}",
                x.shape(),
                mask.shape()
            );
        }
        if mask.dtype() != DType::U8 {
            bail!(
                "MaskedFillOp: mask dtype must be U8, got {}",
                mask.dtype()
            );
        }
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!("MaskedFillOp: x dtype must be F32/BF16/F16, got {}", x.dtype());
        }
        if !x.is_contiguous() || !mask.is_contiguous() {
            bail!("MaskedFillOp: x and mask must be contiguous");
        }

        let dtype = x.dtype();
        let per = dtype.size_in_bytes();
        let n = x.element_count();
        let x_cpu = downcast_cpu(x, "x")?;
        let m_cpu = downcast_cpu(mask, "mask")?;
        let x_bytes = x_cpu.as_bytes();
        let m_bytes = m_cpu.as_bytes();
        let mut out = vec![0u8; n * per];

        match dtype {
            DType::F32 => {
                for i in 0..n {
                    let v = if m_bytes[i] != 0 {
                        self.fill_value
                    } else {
                        f32::from_le_bytes(x_bytes[i * 4..i * 4 + 4].try_into().unwrap())
                    };
                    out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            DType::BF16 => {
                for i in 0..n {
                    let v = if m_bytes[i] != 0 {
                        self.fill_value
                    } else {
                        half::bf16::from_le_bytes(x_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                            .to_f32()
                    };
                    out[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                }
            }
            DType::F16 => {
                for i in 0..n {
                    let v = if m_bytes[i] != 0 {
                        self.fill_value
                    } else {
                        half::f16::from_le_bytes(x_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                            .to_f32()
                    };
                    out[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }

        let cpu = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu);
        Ok(Some(Tensor::from_parts(
            storage,
            Layout::contiguous(x.shape().to_vec()),
            TensorId::next(),
        )?))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor, mask: &Tensor) -> Result<Option<Tensor>> {
        // Validate shape / dtype / contiguous — same contract as cpu_fwd.
        // Return Ok(None) for unsupported configurations so the
        // dispatcher falls through to CPU; bail on hard errors that
        // would also fail CPU dispatch.
        if x.shape() != mask.shape() {
            bail!(
                "MaskedFillOp: shape mismatch {:?} vs mask {:?}",
                x.shape(),
                mask.shape()
            );
        }
        if mask.dtype() != DType::U8 {
            bail!(
                "MaskedFillOp: mask dtype must be U8, got {}",
                mask.dtype()
            );
        }
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() || !mask.is_contiguous() {
            return Ok(None);
        }
        // Only CUDA-resident storage on both inputs hits the kernel
        // path; otherwise fall through to CPU.
        if !matches!(x.device(), crate::Device::Cuda(_))
            || !matches!(mask.device(), crate::Device::Cuda(_))
        {
            return Ok(None);
        }

        let out = crate::cuda_masked_fill(x, mask, self.fill_value)?;
        Ok(Some(out))
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, x: &Tensor, mask: &Tensor) -> Result<Option<Tensor>> {
        // Same precondition gates as cuda_fwd. Hard-error on shape
        // mismatch and mask-dtype since those would also fail CPU
        // dispatch; soft-fall through on unsupported value dtypes /
        // non-contiguous / non-Vulkan-resident storage so the
        // dispatcher routes through CPU.
        if x.shape() != mask.shape() {
            bail!(
                "MaskedFillOp: shape mismatch {:?} vs mask {:?}",
                x.shape(),
                mask.shape()
            );
        }
        if mask.dtype() != DType::U8 {
            bail!(
                "MaskedFillOp: mask dtype must be U8, got {}",
                mask.dtype()
            );
        }
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() || !mask.is_contiguous() {
            return Ok(None);
        }
        if !matches!(x.device(), crate::Device::Vulkan(_))
            || !matches!(mask.device(), crate::Device::Vulkan(_))
        {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Vulkan): implement
        // `crate::vulkan_masked_fill(x, mask, self.fill_value)`
        // analogous to `crate::cuda_masked_fill` above. Until that
        // wrapper lands, fall through to the CPU path (numerics-
        // correct, performance-wrong).
        // Candidate implementations:
        //   1. New SPIR-V compute shader added to
        //      `kiln-vulkan-kernel::vk_ops::mask`. The existing
        //      `vk_causal_mask_inplace` already writes -inf into a
        //      scores tensor based on a position predicate; the
        //      `masked_fill` variant generalizes it to an arbitrary
        //      U8 mask buffer and arbitrary `fill_value`. Pure
        //      pointwise — one work-item per element, no reductions.
        //   2. Reuse `vk_ops::elementwise`'s where-style ternary if
        //      present; otherwise this is a thin new shader.
        //   3. Dtype matrix: `VkDType` currently exposes only F32 and
        //      Bf16. F16 needs a new variant added in
        //      `kiln-vulkan-kernel::vk_tensor::VkDType` before any
        //      F16-native shader can land. Mask dtype is fixed at U8
        //      (matches cpu_fwd); the shader must read U8 via a
        //      uint8 storage buffer (Vulkan `VK_KHR_8bit_storage`).
        Ok(None)
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// `out[i] = mask[i] != 0 ? fill_value : x[i]`. Shape-matched inputs.
pub fn masked_fill(x: &Tensor, mask: &Tensor, fill_value: f32) -> Result<Tensor> {
    dispatch2(&MaskedFillOp::new(fill_value), x, mask)
}

/// Construct a `[seq_len, seq_len]` U8 causal mask: `mask[i, j] = 1`
/// iff `j > i` (future positions).
///
/// Apply via `masked_fill(scores, causal_mask(seq_len), f32::NEG_INFINITY)`
/// to zero out future positions in attention pre-softmax.
pub fn causal_mask(seq_len: usize) -> Result<Tensor> {
    let mut bytes = vec![0u8; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                bytes[i * seq_len + j] = 1;
            }
        }
    }
    let cpu = CpuStorage::from_bytes(DType::U8, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(
        storage,
        Layout::contiguous(vec![seq_len, seq_len]),
        TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("MaskedFillOp: {label} storage must be CpuStorage")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn masked_fill_replaces_masked_positions() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let m = Tensor::from_slice(&[0u8, 1, 0, 1], vec![4]).unwrap();
        let out = masked_fill(&x, &m, -99.0).unwrap();
        assert_eq!(read_f32(&out), vec![1.0, -99.0, 3.0, -99.0]);
    }

    #[test]
    fn masked_fill_with_neg_inf_for_pre_softmax() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let m = Tensor::from_slice(&[0u8, 1, 0], vec![3]).unwrap();
        let out = masked_fill(&x, &m, f32::NEG_INFINITY).unwrap();
        let got = read_f32(&out);
        assert_eq!(got[0], 1.0);
        assert!(got[1].is_infinite() && got[1].is_sign_negative());
        assert_eq!(got[2], 3.0);
    }

    #[test]
    fn causal_mask_4x4_is_upper_triangular() {
        let m = causal_mask(4).unwrap();
        assert_eq!(m.shape(), &[4, 4]);
        assert_eq!(m.dtype(), DType::U8);
        let cpu = m.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        // Row-major, j > i is masked:
        //   row 0: [0, 1, 1, 1]
        //   row 1: [0, 0, 1, 1]
        //   row 2: [0, 0, 0, 1]
        //   row 3: [0, 0, 0, 0]
        assert_eq!(
            bytes,
            &[0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        );
    }

    #[test]
    fn causal_mask_then_masked_fill_full_pipeline() {
        // Toy scores: [1, 1, 1, 1] x4 -> apply causal mask + masked_fill
        // -> last row keeps all; first row keeps only [0].
        let n = 4;
        let scores: Vec<f32> = vec![1.0; n * n];
        let scores = Tensor::from_slice(&scores, vec![n, n]).unwrap();
        let m = causal_mask(n).unwrap();
        let out = masked_fill(&scores, &m, f32::NEG_INFINITY).unwrap();
        let got = read_f32(&out);
        // Row 0: [1, -inf, -inf, -inf]
        assert_eq!(got[0], 1.0);
        assert!(got[1].is_infinite() && got[1] < 0.0);
        // Row 3: all 1s
        for j in 0..n {
            assert_eq!(got[3 * n + j], 1.0);
        }
    }

    #[test]
    fn rejects_shape_mismatch() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let m = Tensor::from_slice(&[0u8, 1, 0], vec![3]).unwrap();
        let e = masked_fill(&x, &m, 0.0).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn rejects_non_u8_mask() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let m = Tensor::from_slice(&[1u32, 0], vec![2]).unwrap();
        let e = masked_fill(&x, &m, 0.0).unwrap_err();
        assert!(e.to_string().contains("U8"));
    }

    #[test]
    fn bf16_path() {
        let xv: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&xv, vec![3]).unwrap();
        let m = Tensor::from_slice(&[0u8, 1, 0], vec![3]).unwrap();
        let out = masked_fill(&x, &m, -50.0).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        let v1 = half::bf16::from_le_bytes(bytes[2..4].try_into().unwrap()).to_f32();
        // BF16 has limited precision; -50.0 is exactly representable.
        assert!((v1 - -50.0).abs() < 1e-2);
    }

    #[test]
    fn op_metadata() {
        let op = MaskedFillOp::new(-99.0);
        assert_eq!(op.name(), "masked_fill");
        assert!(op.determinism().is_constructive());
        assert_eq!(op.fill_value(), -99.0);
    }
}
