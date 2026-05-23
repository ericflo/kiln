//! Backwards for the unary-arith op family: abs, neg, exp, ln, sqrt.
//!
//! | Forward | Backward |
//! |---------|----------|
//! | `y = abs(x)`  | `dx = dy * sign(x)` — saves `x` |
//! | `y = neg(x)`  | `dx = -dy` — no saved state |
//! | `y = exp(x)`  | `dx = dy * y` — saves `y` |
//! | `y = ln(x)`   | `dx = dy / x` — saves `x` |
//! | `y = sqrt(x)` | `dx = dy / (2*y)` — saves `y` |
//!
//! `sign(0) = 0` for `abs` (the standard subgradient pick); cleaner
//! than `sign(x).signum()` at exact zero.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

fn validate_same(a: &Tensor, b: &Tensor, name: &str) -> Result<()> {
    if a.shape() != b.shape() {
        bail!("{name}: shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    }
    if a.dtype() != b.dtype() {
        bail!("{name}: dtype mismatch: {} vs {}", a.dtype(), b.dtype());
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("{name}: inputs must be contiguous");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", a.dtype());
    }
    Ok(())
}

fn load_f32(t: &Tensor) -> Result<Vec<f32>> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("unary_arith_backward: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let dtype = t.dtype();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        });
    }
    Ok(out)
}

fn store_f32(dtype: DType, shape: &[usize], data: &[f32]) -> Result<Tensor> {
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; data.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
}

// ----------------------------------------------------------------------
// Abs / Ln (save `x`)
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct AbsBackward {
    pub x: Tensor,
}

impl BackwardOp for AbsBackward {
    fn name(&self) -> &'static str {
        "abs_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "AbsBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| {
                let s = if xi > 0.0 {
                    1.0
                } else if xi < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                dyi * s
            })
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct LnBackward {
    pub x: Tensor,
}

impl BackwardOp for LnBackward {
    fn name(&self) -> &'static str {
        "ln_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "LnBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| dyi / xi)
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

// ----------------------------------------------------------------------
// Neg (no saved state)
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct NegBackward;

impl BackwardOp for NegBackward {
    fn name(&self) -> &'static str {
        "neg_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if !matches!(grad_output.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "NegBackward: dtype must be F32/BF16/F16, got {}",
                grad_output.dtype()
            );
        }
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = dy.iter().map(|&v| -v).collect();
        let out = store_f32(grad_output.dtype(), grad_output.shape(), &dx)?;
        Ok(vec![Some(out)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

// ----------------------------------------------------------------------
// Exp / Sqrt (save `y`)
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct ExpBackward {
    pub y: Tensor,
}

impl BackwardOp for ExpBackward {
    fn name(&self) -> &'static str {
        "exp_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.y, grad_output, "ExpBackward")?;
        let y = load_f32(&self.y)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = y
            .iter()
            .zip(dy.iter())
            .map(|(&yi, &dyi)| dyi * yi)
            .collect();
        Ok(vec![Some(store_f32(self.y.dtype(), self.y.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false // saves y, not x
    }
}

#[derive(Debug)]
pub struct SqrtBackward {
    pub y: Tensor,
}

impl BackwardOp for SqrtBackward {
    fn name(&self) -> &'static str {
        "sqrt_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.y, grad_output, "SqrtBackward")?;
        let y = load_f32(&self.y)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = y
            .iter()
            .zip(dy.iter())
            .map(|(&yi, &dyi)| dyi / (2.0 * yi))
            .collect();
        Ok(vec![Some(store_f32(self.y.dtype(), self.y.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "len mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "idx {i}: got {x}, want {y} (tol {tol})"
            );
        }
    }

    #[test]
    fn abs_backward_routes_sign() {
        // x = [-2, 0, 3]; dy = [1, 1, 1]; dx = [-1, 0, 1]
        let x = Tensor::from_slice(&[-2.0f32, 0.0, 3.0], vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let bo = AbsBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        assert_eq!(dx, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn neg_backward_negates() {
        let dy = Tensor::from_slice(&[2.0f32, -3.0, 0.5], vec![3]).unwrap();
        let dx = load_f32(NegBackward.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        assert_eq!(dx, vec![-2.0, 3.0, -0.5]);
    }

    #[test]
    fn exp_backward_uses_saved_y() {
        // y = e^[0, 1, -1] ≈ [1, e, 1/e]; dy = ones; dx = y.
        let y = Tensor::from_slice(
            &[1.0f32, std::f32::consts::E, 1.0 / std::f32::consts::E],
            vec![3],
        )
        .unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![3]).unwrap();
        let bo = ExpBackward { y };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(
            &dx,
            &[1.0, std::f32::consts::E, 1.0 / std::f32::consts::E],
            1e-6,
        );
    }

    #[test]
    fn ln_backward_one_over_x() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 4.0], vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![3]).unwrap();
        let bo = LnBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[1.0, 0.5, 0.25], 1e-6);
    }

    #[test]
    fn sqrt_backward_one_over_two_y() {
        // y = √[1, 4, 9] = [1, 2, 3]; dx = dy / (2y) = [0.5, 0.25, 0.1667]
        let y = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![3]).unwrap();
        let bo = SqrtBackward { y };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[0.5, 0.25, 1.0 / 6.0], 1e-5);
    }

    #[test]
    fn op_metadata() {
        let one = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        assert_eq!(AbsBackward { x: one.clone() }.name(), "abs_backward");
        assert_eq!(NegBackward.name(), "neg_backward");
        assert_eq!(ExpBackward { y: one.clone() }.name(), "exp_backward");
        assert_eq!(LnBackward { x: one.clone() }.name(), "ln_backward");
        assert_eq!(SqrtBackward { y: one.clone() }.name(), "sqrt_backward");
        assert_eq!(AbsBackward { x: one.clone() }.input_count(), 1);
        assert!(AbsBackward { x: one.clone() }.requires_input(0));
        assert!(!NegBackward.requires_input(0));
    }
}
