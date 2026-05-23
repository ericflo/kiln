//! Elementwise unary arithmetic primitives: `abs`, `neg`, `exp`,
//! `ln`, `sqrt`.
//!
//! Distinct from `activation.rs` (silu/sigmoid/gelu) — those are
//! nonlinearities used by neural nets; these are math primitives
//! used everywhere (loss functions, regularizers, sampling, custom
//! kernels).
//!
//! # Semantics
//!
//! Pointwise; all five operate on F32/BF16/F16. F32 internal compute
//! regardless of input dtype.
//!
//! - `abs(x)  = |x|`
//! - `neg(x)  = -x`
//! - `exp(x)  = e^x`
//! - `ln(x)   = log_e(x)` (returns NaN for x ≤ 0)
//! - `sqrt(x) = √x` (returns NaN for x < 0)
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; no reduction; bit-identical at the
//! same input dtype.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryArithKind {
    Abs,
    Neg,
    Exp,
    Ln,
    Sqrt,
}

impl UnaryArithKind {
    pub const fn name(self) -> &'static str {
        match self {
            UnaryArithKind::Abs => "abs",
            UnaryArithKind::Neg => "neg",
            UnaryArithKind::Exp => "exp",
            UnaryArithKind::Ln => "ln",
            UnaryArithKind::Sqrt => "sqrt",
        }
    }

    pub fn apply_f32(self, x: f32) -> f32 {
        match self {
            UnaryArithKind::Abs => x.abs(),
            UnaryArithKind::Neg => -x,
            UnaryArithKind::Exp => x.exp(),
            UnaryArithKind::Ln => x.ln(),
            UnaryArithKind::Sqrt => x.sqrt(),
        }
    }
}

fn apply(kind: UnaryArithKind, x: &Tensor) -> Result<Tensor> {
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "{}: dtype must be F32/BF16/F16, got {}",
            kind.name(),
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("{}: input must be contiguous", kind.name());
    }
    let dtype = x.dtype();
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("unary_arith: storage must be CpuStorage"))?;
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

pub fn abs(x: &Tensor) -> Result<Tensor> {
    apply(UnaryArithKind::Abs, x)
}
pub fn neg(x: &Tensor) -> Result<Tensor> {
    apply(UnaryArithKind::Neg, x)
}
pub fn exp(x: &Tensor) -> Result<Tensor> {
    apply(UnaryArithKind::Exp, x)
}
pub fn ln(x: &Tensor) -> Result<Tensor> {
    apply(UnaryArithKind::Ln, x)
}
pub fn sqrt(x: &Tensor) -> Result<Tensor> {
    apply(UnaryArithKind::Sqrt, x)
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
            assert!(
                (x - y).abs() < tol,
                "idx {i}: got {x}, want {y} (tol {tol})"
            );
        }
    }

    #[test]
    fn abs_negatives_become_positive() {
        let x = Tensor::from_slice(&[-1.0f32, 2.0, -3.0, 0.0], vec![4]).unwrap();
        let y = abs(&x).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn neg_flips_sign() {
        let x = Tensor::from_slice(&[1.0f32, -2.0, 0.0], vec![3]).unwrap();
        let y = neg(&x).unwrap();
        assert_eq!(read_f32(&y), vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn exp_known_values() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, -1.0], vec![3]).unwrap();
        let y = read_f32(&exp(&x).unwrap());
        approx(&y, &[1.0, std::f32::consts::E, 1.0 / std::f32::consts::E], 1e-5);
    }

    #[test]
    fn ln_known_values() {
        let x = Tensor::from_slice(&[1.0f32, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E], vec![3]).unwrap();
        let y = read_f32(&ln(&x).unwrap());
        approx(&y, &[0.0, 1.0, 2.0], 1e-5);
    }

    #[test]
    fn sqrt_known_values() {
        let x = Tensor::from_slice(&[0.0f32, 1.0, 4.0, 9.0], vec![4]).unwrap();
        let y = read_f32(&sqrt(&x).unwrap());
        approx(&y, &[0.0, 1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn ln_of_negative_is_nan() {
        let x = Tensor::from_slice(&[-1.0f32], vec![1]).unwrap();
        let y = read_f32(&ln(&x).unwrap());
        assert!(y[0].is_nan());
    }

    #[test]
    fn sqrt_of_negative_is_nan() {
        let x = Tensor::from_slice(&[-1.0f32], vec![1]).unwrap();
        let y = read_f32(&sqrt(&x).unwrap());
        assert!(y[0].is_nan());
    }

    #[test]
    fn bf16_round_trips_through_each_op() {
        for kind in [
            UnaryArithKind::Abs,
            UnaryArithKind::Neg,
            UnaryArithKind::Exp,
            UnaryArithKind::Ln,
            UnaryArithKind::Sqrt,
        ] {
            let bf: Vec<half::bf16> = [1.0f32, 2.0]
                .iter()
                .map(|&v| half::bf16::from_f32(v))
                .collect();
            let x = Tensor::from_slice(&bf, vec![2]).unwrap();
            let y = apply(kind, &x).unwrap();
            assert_eq!(y.dtype(), DType::BF16, "kind {}", kind.name());
        }
    }

    #[test]
    fn shape_preserved() {
        let x = Tensor::from_slice(&[1.0f32; 12], vec![2, 3, 2]).unwrap();
        let y = abs(&x).unwrap();
        assert_eq!(y.shape(), &[2, 3, 2]);
    }

    #[test]
    fn rejects_bad_dtype() {
        let x = Tensor::from_slice(&[1u32, 2], vec![2]).unwrap();
        let e = abs(&x).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }

    #[test]
    fn kind_names() {
        assert_eq!(UnaryArithKind::Abs.name(), "abs");
        assert_eq!(UnaryArithKind::Neg.name(), "neg");
        assert_eq!(UnaryArithKind::Exp.name(), "exp");
        assert_eq!(UnaryArithKind::Ln.name(), "ln");
        assert_eq!(UnaryArithKind::Sqrt.name(), "sqrt");
    }
}
