//! `sinh`, `cosh`, `atanh` — hyperbolic primitives.
//!
//! `tanh` is in `activation.rs` because it's used as a nonlinearity.
//! These three complete the hyperbolic family for general math use:
//!
//! ```text
//! sinh(x)  = (e^x - e^-x) / 2
//! cosh(x)  = (e^x + e^-x) / 2
//! atanh(x) = 0.5 * ln((1 + x) / (1 - x))   (defined for |x| < 1)
//! ```
//!
//! # CUDA wiring (#1082)
//!
//! All three route through the shared `cuda_activation_unary`
//! kernel — sinh/cosh as kinds 10/11 (#1082 base) and atanh as
//! kind 21 (#1082 follow-up).

use std::sync::Arc;

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperKind {
    Sinh,
    Cosh,
    Atanh,
}

impl HyperKind {
    const fn name(self) -> &'static str {
        match self {
            HyperKind::Sinh => "sinh",
            HyperKind::Cosh => "cosh",
            HyperKind::Atanh => "atanh",
        }
    }

    fn apply_f32(self, v: f32) -> f32 {
        match self {
            HyperKind::Sinh => v.sinh(),
            HyperKind::Cosh => v.cosh(),
            HyperKind::Atanh => v.atanh(),
        }
    }

    /// CUDA kernel kind tag matching `KIND_SINH`/`KIND_COSH`/
    /// `KIND_ATANH` in `csrc/activation.cu` (#1082).
    #[cfg(feature = "cuda")]
    const fn cuda_kind_tag(self) -> i32 {
        match self {
            HyperKind::Sinh => 10,
            HyperKind::Cosh => 11,
            HyperKind::Atanh => 21,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct HyperOp {
    kind: HyperKind,
}

impl DeviceOp1 for HyperOp {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind.name())?;
        let t = cpu_apply(self.kind, x)?;
        Ok(Some(t))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.kind.name())?;
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        Ok(Some(crate::cuda_activation_unary(x, self.kind.cuda_kind_tag())?))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd so future MSL
        // kernel work can drop in without changing the call site.
        validate(x, self.kind.name())?;
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Metal): once a Metal hyperbolic kernel
        // ships, dispatch via `self.kind.cuda_kind_tag()` (or a
        // Metal-specific tag) and route through
        // `crate::metal_activation_unary`. Until then, fall through
        // to CPU (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. Custom MSL kernel: per-element pointwise; switch on
        //      `kind_tag` selecting sinh / cosh / asinh / acosh /
        //      atanh. (tanh is already in the activation kernel.)
        //   2. MPS Graph: per-primitive (`sinh(_:)`, `cosh(_:)`, etc.)
        //      bound by kind.
        Ok(None)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd / metal_fwd.
        validate(x, self.kind.name())?;
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        // TODO(#1082, phase 4 Vulkan): once a Vulkan hyperbolic
        // kernel ships, dispatch via `self.kind.cuda_kind_tag()` and
        // route through `crate::vulkan_activation_unary`. Until
        // then, fall through to CPU (numerics-correct,
        // performance-wrong).
        // Candidate implementations:
        //   1. SPIR-V compute shader: per-element pointwise; switch
        //      on `kind_tag` (push-constant or pipeline-specialized)
        //      selecting sinh / cosh / asinh / acosh / atanh.
        //   2. Dtype matrix gap: `VkDType` exposes F32 / BF16 today;
        //      F16 / extended-precision paths need widening or a
        //      cast wrapper at the dispatch boundary.
        Ok(None)
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

fn validate(x: &Tensor, name: &str) -> Result<()> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("{name}: input must be contiguous");
    }
    Ok(())
}

fn cpu_apply(kind: HyperKind, x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("hyperbolic: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = x.element_count();
    let per = dtype.size_in_bytes();
    let mut out = vec![0u8; n * per];
    for i in 0..n {
        let v = match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        };
        let y = kind.apply_f32(v);
        match dtype {
            DType::F32 => out[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes()),
            DType::BF16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
            DType::F16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(x.shape().to_vec()), TensorId::next())
}

pub fn sinh(x: &Tensor) -> Result<Tensor> {
    dispatch1(&HyperOp { kind: HyperKind::Sinh }, x)
}

pub fn cosh(x: &Tensor) -> Result<Tensor> {
    dispatch1(&HyperOp { kind: HyperKind::Cosh }, x)
}

pub fn atanh(x: &Tensor) -> Result<Tensor> {
    dispatch1(&HyperOp { kind: HyperKind::Atanh }, x)
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

    #[test]
    fn sinh_known_values() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, -1.0], vec![3]).unwrap();
        let y = read_f32(&sinh(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - 1.1752).abs() < 1e-3);
        assert!((y[2] + 1.1752).abs() < 1e-3);
    }

    #[test]
    fn cosh_known_values() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, -1.0], vec![3]).unwrap();
        let y = read_f32(&cosh(&x).unwrap());
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - 1.5431).abs() < 1e-3);
        assert!((y[2] - 1.5431).abs() < 1e-3); // cosh is even
    }

    #[test]
    fn hyperbolic_identity() {
        // cosh²(x) - sinh²(x) = 1
        let x = Tensor::from_slice(&[0.3f32, 1.5, -2.7], vec![3]).unwrap();
        let s = read_f32(&sinh(&x).unwrap());
        let c = read_f32(&cosh(&x).unwrap());
        for i in 0..3 {
            let id = c[i] * c[i] - s[i] * s[i];
            assert!((id - 1.0).abs() < 1e-3, "i={i}: id={id}");
        }
    }

    #[test]
    fn atanh_known_values() {
        // atanh(0)=0, atanh(0.5) ≈ 0.549306, atanh(-0.5) ≈ -0.549306
        let x = Tensor::from_slice(&[0.0f32, 0.5, -0.5], vec![3]).unwrap();
        let y = read_f32(&atanh(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - 0.5493061).abs() < 1e-5);
        assert!((y[2] + 0.5493061).abs() < 1e-5);
    }

    #[test]
    fn atanh_inverse_of_tanh() {
        // atanh(tanh(x)) ≈ x for |x| < ~5 (tanh saturates after that)
        let xs = [-1.5f32, -0.5, 0.0, 0.5, 1.5];
        for &xv in &xs {
            let t = xv.tanh();
            let x = Tensor::from_slice(&[t], vec![1]).unwrap();
            let y = read_f32(&atanh(&x).unwrap());
            assert!((y[0] - xv).abs() < 1e-4, "atanh(tanh({xv}))={}", y[0]);
        }
    }
}
