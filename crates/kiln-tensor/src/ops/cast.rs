//! `cast` — dtype conversion between supported numeric dtypes.
//!
//! Replaces candle's `Tensor::to_dtype` at the call sites that promote
//! BF16 → F32 (e.g. attention scores promoting before softmax for
//! numerical stability) and demote F32 → BF16 (e.g. casting an FP32
//! reference back to BF16 storage for the next layer).
//!
//! # Supported casts
//!
//! Numeric ↔ numeric (FP32 / BF16 / FP16) — round-to-nearest-even per
//! IEEE-754.
//!
//! Integer round-trip (U32 ↔ I64) — exact within range; out-of-range
//! errors.
//!
//! Numeric → integer / integer → numeric is intentionally **not**
//! supported here: those are application-level conversions
//! (`f32 -> i64` for token-id sampling, etc.) that should live behind
//! a typed sampler API rather than a generic tensor cast.
//!
//! # Determinism
//!
//! `Constructive`. Round-to-nearest-even is bit-deterministic across
//! runs; the only ULP variance is the inherent precision loss of
//! the target dtype, which is a property of the conversion, not the
//! schedule.

use crate::{
    bail, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result, Storage,
    Tensor, TensorId,
};
use std::sync::Arc;

/// Dtype conversion op. Carries the target dtype.
#[derive(Debug, Clone, Copy)]
pub struct CastOp {
    target: DType,
}

impl CastOp {
    pub const fn new(target: DType) -> Self {
        CastOp { target }
    }
    pub const fn target(self) -> DType {
        self.target
    }
}

impl DeviceOp1 for CastOp {
    fn name(&self) -> &'static str {
        "cast"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if !x.is_contiguous() {
            bail!("CastOp: input must be contiguous");
        }
        let from = x.dtype();
        let to = self.target;
        if from == to {
            // No-op fast path. Clone is cheap (storage Arc bumped).
            return Ok(Some(x.clone()));
        }

        let cpu = x
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| {
                Error::Msg(format!(
                    "CastOp: storage must be CpuStorage on CPU (from {from} to {to}, device {:?}, \
                     contiguous {}) — an unsupported CUDA cast pair fell back to the CPU branch",
                    x.device(),
                    x.is_contiguous()
                ))
            })?;
        let bytes = cpu.as_bytes();
        let n = x.element_count();

        let out_bytes = match (from, to) {
            // Float <-> Float
            (DType::F32, DType::BF16) => f32_to_bf16(bytes, n),
            (DType::F32, DType::F16) => f32_to_f16(bytes, n),
            (DType::BF16, DType::F32) => bf16_to_f32(bytes, n),
            (DType::BF16, DType::F16) => bf16_to_f16(bytes, n),
            (DType::F16, DType::F32) => f16_to_f32(bytes, n),
            (DType::F16, DType::BF16) => f16_to_bf16(bytes, n),
            // Integer round-trip — exact in range.
            (DType::U32, DType::I64) => u32_to_i64(bytes, n),
            (DType::I64, DType::U32) => i64_to_u32(bytes, n)?,
            // U8 (boolean masks from compare ops) <-> float triple. The
            // forward direction is the load-bearing one: turning a U8
            // comparison mask into a {0.0, 1.0} float multiplier (e.g. the
            // strictly-lower-triangular GDN mask). The reverse is exact for
            // the {0.0, 1.0} masks it round-trips and saturates/truncates
            // otherwise so the conversion is total.
            (DType::U8, DType::F32) => u8_to_f32(bytes, n),
            (DType::U8, DType::BF16) => u8_to_bf16(bytes, n),
            (DType::U8, DType::F16) => u8_to_f16(bytes, n),
            (DType::F32, DType::U8) => f32_to_u8(bytes, n),
            (DType::BF16, DType::U8) => bf16_to_u8(bytes, n),
            (DType::F16, DType::U8) => f16_to_u8(bytes, n),
            // Anything else: deliberately not supported here.
            _ => bail!(
                "CastOp: conversion {from} -> {to} is not supported \
                 (numeric↔numeric and U32↔I64 only)"
            ),
        };

        let cpu_out = CpuStorage::from_bytes(to, out_bytes)?;
        let storage: Storage = Arc::new(cpu_out);
        let out = Tensor::from_parts(
            storage,
            Layout::contiguous(x.shape().to_vec()),
            TensorId::next(),
        )?;
        Ok(Some(out))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        let from = x.dtype();
        let to = self.target;
        // CUDA path covers F32 ↔ BF16 ↔ F16. Integer round-trips
        // stay on the CPU fallback (their few call sites have host
        // data anyway).
        let cuda_supported = matches!(
            (from, to),
            (DType::F32, DType::BF16)
                | (DType::F32, DType::F16)
                | (DType::BF16, DType::F32)
                | (DType::BF16, DType::F16)
                | (DType::F16, DType::F32)
                | (DType::F16, DType::BF16)
                | (DType::F32, DType::F32)
                | (DType::BF16, DType::BF16)
                | (DType::F16, DType::F16)
        );
        if !cuda_supported {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        Ok(Some(crate::cuda_cast(x, to)?))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Phase 4 substrate-op landing: dispatch through
        // `crate::metal_cast` which wraps candle's production Metal
        // `to_dtype` kernel.
        //
        // Gate on:
        //   - float triple (F32/BF16/F16) — integer round-trips stay on CPU
        //   - contiguous input
        //   - storage must be Metal-backed
        //   - same-dtype is a no-op handled by cpu_fwd's fast path
        let from = x.dtype();
        let to = self.target;
        if from == to {
            return Ok(None);
        }
        let metal_supported = matches!(
            (from, to),
            (DType::F32, DType::BF16)
                | (DType::F32, DType::F16)
                | (DType::BF16, DType::F32)
                | (DType::BF16, DType::F16)
                | (DType::F16, DType::F32)
                | (DType::F16, DType::BF16)
                // U8 boolean masks <-> float triple (e.g. GDN triangular mask
                // -> float multiplier). Mirrors the CPU pairs above.
                | (DType::U8, DType::F32)
                | (DType::U8, DType::BF16)
                | (DType::U8, DType::F16)
                | (DType::F32, DType::U8)
                | (DType::BF16, DType::U8)
                | (DType::F16, DType::U8)
        );
        if !metal_supported {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        if x.storage()
            .as_any()
            .downcast_ref::<crate::MetalStorage>()
            .is_none()
        {
            return Ok(None);
        }
        Ok(Some(crate::metal_cast(x, to)?))
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Phase 4 substrate-op landing: dispatch through
        // `crate::vulkan_cast` which wraps the production
        // `vk_cast_f32_to_bf16_no_grad` and `vk_cast_bf16_to_f32_no_grad`
        // SPIR-V shaders.
        //
        // Gate on:
        //   - F32 <-> BF16 dtype pair (only shaders available today;
        //     F16 round-trips and integer casts fall through to CPU)
        //   - contiguous input
        //   - storage must be Vulkan-backed (otherwise fall back to
        //     CPU gracefully rather than have the wrapper raise)
        //   - same-dtype is a no-op handled by cpu_fwd's fast path
        //
        // Current bridge D2H+H2D round-trips bytes through the host;
        // see `vulkan_storage.rs::vulkan_cast` rustdoc for the
        // zero-copy follow-up plan.
        let from = x.dtype();
        let to = self.target;
        if from == to {
            // Same-dtype: let cpu_fwd handle the storage-Arc clone
            // fast path so dispatch falls through cleanly.
            return Ok(None);
        }
        let vulkan_supported = matches!(
            (from, to),
            (DType::F32, DType::BF16) | (DType::BF16, DType::F32)
        );
        if !vulkan_supported {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        if x.storage()
            .as_any()
            .downcast_ref::<crate::VulkanStorage>()
            .is_none()
        {
            return Ok(None);
        }
        Ok(Some(crate::vulkan_cast(x, to)?))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Convenience: dispatch a cast to `target`.
pub fn cast(x: &Tensor, target: DType) -> Result<Tensor> {
    crate::dispatch1(&CastOp::new(target), x)
}

// ----------------------------------------------------------------------
// Per-pair byte conversions. All assume contiguous, length-checked input.
// ----------------------------------------------------------------------

fn f32_to_bf16(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

fn f32_to_f16(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

fn bf16_to_f32(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bf16_to_f16(bytes: &[u8], n: usize) -> Vec<u8> {
    // Go through f32 to avoid bit-pattern surprises.
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

fn f16_to_f32(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f16_to_bf16(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
        out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    out
}

fn u32_to_i64(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 8);
    for i in 0..n {
        let v = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        out.extend_from_slice(&(v as i64).to_le_bytes());
    }
    out
}

fn i64_to_u32(bytes: &[u8], n: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let v = i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
        if !(0..=u32::MAX as i64).contains(&v) {
            bail!(
                "CastOp: I64 value {v} at index {i} out of range for U32 (0..={})",
                u32::MAX
            );
        }
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    Ok(out)
}

// --- U8 (boolean masks) <-> float triple ---
// U8 is 1 byte per element; the float targets are width 4 (F32) or 2 (BF16/F16).

fn u8_to_f32(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        out.extend_from_slice(&(bytes[i] as f32).to_le_bytes());
    }
    out
}

fn u8_to_bf16(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        out.extend_from_slice(&half::bf16::from_f32(bytes[i] as f32).to_le_bytes());
    }
    out
}

fn u8_to_f16(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        out.extend_from_slice(&half::f16::from_f32(bytes[i] as f32).to_le_bytes());
    }
    out
}

/// Clamp a finite-or-NaN `f32` to a `u8`, truncating toward zero and
/// saturating at 0/255 (matches the C/CUDA `(unsigned char)` cast guard rails
/// used elsewhere; NaN -> 0).
#[inline]
fn f32_to_u8_sat(v: f32) -> u8 {
    if v.is_nan() {
        0
    } else {
        v.clamp(0.0, 255.0) as u8
    }
}

fn f32_to_u8(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        out.push(f32_to_u8_sat(v));
    }
    out
}

fn bf16_to_u8(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
        out.push(f32_to_u8_sat(v));
    }
    out
}

fn f16_to_u8(bytes: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
        out.push(f32_to_u8_sat(v));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn cast_identity_returns_clone() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = cast(&x, DType::F32).unwrap();
        // Same storage Arc (no copy).
        assert!(Arc::ptr_eq(x.storage(), y.storage()));
    }

    #[test]
    fn cast_f32_to_bf16_then_back() {
        let x = Tensor::from_slice(&[1.0f32, 2.5, -3.75, 0.0], vec![4]).unwrap();
        let y = cast(&x, DType::BF16).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        let z = cast(&y, DType::F32).unwrap();
        let got = read_f32(&z);
        // Round-trip through BF16 is lossy but exact for these specific
        // values (they all fit in BF16's 7-bit mantissa exactly).
        assert_eq!(got, vec![1.0, 2.5, -3.75, 0.0]);
    }

    #[test]
    fn cast_f32_to_f16_then_back() {
        let x = Tensor::from_slice(&[1.0f32, 0.5, -0.25], vec![3]).unwrap();
        let y = cast(&x, DType::F16).unwrap();
        let z = cast(&y, DType::F32).unwrap();
        assert_eq!(read_f32(&z), vec![1.0, 0.5, -0.25]);
    }

    #[test]
    fn cast_bf16_to_f16_through_f32() {
        let xv: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&xv, vec![3]).unwrap();
        let y = cast(&x, DType::F16).unwrap();
        assert_eq!(y.dtype(), DType::F16);
        let cpu = y.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        for (i, &expected) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            let v = half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn cast_u32_to_i64_exact() {
        let x = Tensor::from_slice(&[0u32, 1, u32::MAX], vec![3]).unwrap();
        let y = cast(&x, DType::I64).unwrap();
        assert_eq!(y.dtype(), DType::I64);
        let cpu = y.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<i64> = bytemuck::cast_slice::<u8, i64>(cpu.as_bytes()).to_vec();
        assert_eq!(back, vec![0i64, 1, u32::MAX as i64]);
    }

    #[test]
    fn cast_i64_to_u32_in_range() {
        let x = Tensor::from_slice(&[0i64, 42, u32::MAX as i64], vec![3]).unwrap();
        let y = cast(&x, DType::U32).unwrap();
        assert_eq!(y.dtype(), DType::U32);
        let cpu = y.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<u32> = bytemuck::cast_slice::<u8, u32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, vec![0u32, 42, u32::MAX]);
    }

    #[test]
    fn cast_i64_to_u32_out_of_range_errors() {
        let x = Tensor::from_slice(&[-1i64], vec![1]).unwrap();
        let e = cast(&x, DType::U32).unwrap_err();
        assert!(e.to_string().contains("out of range"));
    }

    #[test]
    fn cast_u8_mask_to_f32() {
        // Boolean comparison masks (from `gt`/`ge`) are U8; converting them to
        // a {0.0, 1.0} float multiplier is the load-bearing case.
        let x = Tensor::from_slice(&[0u8, 1, 0, 1, 1], vec![5]).unwrap();
        let y = cast(&x, DType::F32).unwrap();
        assert_eq!(y.dtype(), DType::F32);
        assert_eq!(read_f32(&y), vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn cast_u8_mask_to_bf16_and_f16() {
        let x = Tensor::from_slice(&[0u8, 1, 1, 0], vec![4]).unwrap();
        let yb = cast(&x, DType::BF16).unwrap();
        assert_eq!(yb.dtype(), DType::BF16);
        let cpu = yb.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        for (i, &e) in [0.0f32, 1.0, 1.0, 0.0].iter().enumerate() {
            let v = half::bf16::from_le_bytes(cpu.as_bytes()[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
            assert_eq!(v, e);
        }
        let yf = cast(&x, DType::F16).unwrap();
        assert_eq!(yf.dtype(), DType::F16);
    }

    #[test]
    fn cast_f32_to_u8_roundtrips_mask_and_saturates() {
        // Exact {0,1} round-trip plus saturation/truncation guard rails.
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.9, -3.0, 300.0, f32::NAN], vec![6]).unwrap();
        let y = cast(&x, DType::U8).unwrap();
        assert_eq!(y.dtype(), DType::U8);
        let cpu = y.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        // truncate toward zero, clamp to [0,255], NaN -> 0.
        assert_eq!(cpu.as_bytes(), &[0u8, 1, 2, 0, 255, 0]);
    }

    #[test]
    fn cast_unsupported_pair_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = cast(&x, DType::I64).unwrap_err();
        assert!(e.to_string().contains("not supported"));
    }

    #[test]
    fn op_metadata() {
        let op = CastOp::new(DType::BF16);
        assert_eq!(op.name(), "cast");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
        assert_eq!(op.target(), DType::BF16);
    }
}
