//! `log2`, `log10`, `log1p`, `exp2`, `expm1` — log/exp variants.
//!
//! Useful for:
//! - **log2 / log10**: information-theoretic quantities (entropy in
//!   bits, ratios in dB)
//! - **log1p**: numerically stable `log(1 + x)` for small `x`
//! - **exp2 / expm1**: inverse counterparts; `expm1` is the
//!   numerically stable `exp(x) - 1`.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; no reduction; bit-identical at the
//! same input dtype.
//!
//! # CUDA wiring (#1082)
//!
//! `log2`, `log10`, `log1p` route through the shared
//! `cuda_activation_unary` kernel (kind tags 15/16/17 — see
//! `csrc/activation.cu`). `exp2` / `expm1` stay CPU-only for now —
//! they don't appear in the model hot path and the kernel kind
//! tags can be added in a follow-up if a call site needs them.

use std::sync::Arc;

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogExpKind {
    Log2,
    Log10,
    Log1p,
    Exp2,
    Expm1,
}

impl LogExpKind {
    pub const fn name(self) -> &'static str {
        match self {
            LogExpKind::Log2 => "log2",
            LogExpKind::Log10 => "log10",
            LogExpKind::Log1p => "log1p",
            LogExpKind::Exp2 => "exp2",
            LogExpKind::Expm1 => "expm1",
        }
    }

    pub fn apply_f32(self, v: f32) -> f32 {
        match self {
            LogExpKind::Log2 => v.log2(),
            LogExpKind::Log10 => v.log10(),
            LogExpKind::Log1p => v.ln_1p(),
            LogExpKind::Exp2 => v.exp2(),
            LogExpKind::Expm1 => v.exp_m1(),
        }
    }

    /// CUDA kernel kind tag matching the `KIND_*` constants in
    /// `csrc/activation.cu` (#1082). `Exp2` and `Expm1` are CPU-only
    /// today — they don't have kind tags in the shared kernel.
    #[cfg(feature = "cuda")]
    const fn cuda_kind_tag(self) -> Option<i32> {
        match self {
            LogExpKind::Log2 => Some(15),
            LogExpKind::Log10 => Some(16),
            LogExpKind::Log1p => Some(17),
            LogExpKind::Exp2 | LogExpKind::Expm1 => None,
        }
    }
}

/// `DeviceOp1` adapter for the log/exp-variant family. CPU path is
/// the canonical reference; CUDA forward path routes through
/// `cuda_activation_unary` for `log2 / log10 / log1p` (#1082).
#[derive(Debug, Clone, Copy)]
struct LogExpOp {
    kind: LogExpKind,
}

impl DeviceOp1 for LogExpOp {
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
        match self.kind.cuda_kind_tag() {
            Some(tag) => Ok(Some(crate::cuda_activation_unary(x, tag)?)),
            None => Ok(None),
        }
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

fn cpu_apply(kind: LogExpKind, x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("log_variants: storage must be CpuStorage"))?;
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

pub fn log2(x: &Tensor) -> Result<Tensor> {
    dispatch1(&LogExpOp { kind: LogExpKind::Log2 }, x)
}
pub fn log10(x: &Tensor) -> Result<Tensor> {
    dispatch1(&LogExpOp { kind: LogExpKind::Log10 }, x)
}
pub fn log1p(x: &Tensor) -> Result<Tensor> {
    dispatch1(&LogExpOp { kind: LogExpKind::Log1p }, x)
}
pub fn exp2(x: &Tensor) -> Result<Tensor> {
    dispatch1(&LogExpOp { kind: LogExpKind::Exp2 }, x)
}
pub fn expm1(x: &Tensor) -> Result<Tensor> {
    dispatch1(&LogExpOp { kind: LogExpKind::Expm1 }, x)
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
    fn log2_powers_of_two() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 4.0, 8.0, 16.0], vec![5]).unwrap();
        let y = read_f32(&log2(&x).unwrap());
        for (i, (g, e)) in y.iter().zip([0.0, 1.0, 2.0, 3.0, 4.0]).enumerate() {
            assert!((g - e).abs() < 1e-5, "idx {i}: got {g}, want {e}");
        }
    }

    #[test]
    fn log10_powers_of_ten() {
        let x = Tensor::from_slice(&[1.0f32, 10.0, 100.0, 1000.0], vec![4]).unwrap();
        let y = read_f32(&log10(&x).unwrap());
        for (i, (g, e)) in y.iter().zip([0.0, 1.0, 2.0, 3.0]).enumerate() {
            assert!((g - e).abs() < 1e-5, "idx {i}: got {g}, want {e}");
        }
    }

    #[test]
    fn log1p_small_values_stable() {
        // log1p(0) = 0; log1p(1) = ln(2); log1p(-0.5) = ln(0.5).
        let x = Tensor::from_slice(&[0.0f32, 1.0, -0.5], vec![3]).unwrap();
        let y = read_f32(&log1p(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - 2.0f32.ln()).abs() < 1e-5);
        assert!((y[2] - 0.5f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn exp2_powers() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, -1.0], vec![4]).unwrap();
        let y = read_f32(&exp2(&x).unwrap());
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[1] - 2.0).abs() < 1e-6);
        assert!((y[2] - 4.0).abs() < 1e-6);
        assert!((y[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn expm1_zero_is_zero() {
        let x = Tensor::from_slice(&[0.0f32, 1.0], vec![2]).unwrap();
        let y = read_f32(&expm1(&x).unwrap());
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - (std::f32::consts::E - 1.0)).abs() < 1e-5);
    }

    #[test]
    fn kind_names() {
        assert_eq!(LogExpKind::Log2.name(), "log2");
        assert_eq!(LogExpKind::Log10.name(), "log10");
        assert_eq!(LogExpKind::Log1p.name(), "log1p");
        assert_eq!(LogExpKind::Exp2.name(), "exp2");
        assert_eq!(LogExpKind::Expm1.name(), "expm1");
    }
}
