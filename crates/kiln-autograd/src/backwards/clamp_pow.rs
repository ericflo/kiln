//! Backwards for `clamp` and `pow`.
//!
//! - **ClampBackward** — `dx = dy` where `lo < x < hi`, else `0`.
//!   Saves `x` plus the bounds. The boundary convention follows
//!   PyTorch: the gradient at `x == lo` or `x == hi` is treated as
//!   `0` (a subgradient pick; the "passthrough" alternative is also
//!   valid but less common).
//! - **PowBackward** — `dx = dy * p * x^(p-1)`. Saves `x` and `p`.

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
        .ok_or_else(|| Error::from_str("clamp_pow_backward: storage must be CpuStorage"))?;
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

#[derive(Debug)]
pub struct ClampBackward {
    pub x: Tensor,
    pub lo: f32,
    pub hi: f32,
}

impl BackwardOp for ClampBackward {
    fn name(&self) -> &'static str {
        "clamp_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "ClampBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| {
                if xi > self.lo && xi < self.hi {
                    dyi
                } else {
                    0.0
                }
            })
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct PowBackward {
    pub x: Tensor,
    pub p: f32,
}

impl BackwardOp for PowBackward {
    fn name(&self) -> &'static str {
        "pow_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "PowBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| dyi * self.p * xi.powf(self.p - 1.0))
            .collect();
        Ok(vec![Some(store_f32(self.x.dtype(), self.x.shape(), &dx)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn clamp_backward_passes_through_when_in_range() {
        // x = [0.5, -0.5]; lo=-1, hi=1 → both in range → dx = dy.
        let x = Tensor::from_slice(&[0.5f32, -0.5], vec![2]).unwrap();
        let dy = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let bo = ClampBackward { x, lo: -1.0, hi: 1.0 };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        assert_eq!(dx, vec![10.0, 20.0]);
    }

    #[test]
    fn clamp_backward_zeros_outside_range() {
        // x = [-5, -1, 0, 1, 5]; lo=-1, hi=1.
        // -5 < lo → 0; -1 == lo → 0; 0 in range → dy; 1 == hi → 0; 5 > hi → 0.
        let x = Tensor::from_slice(&[-5.0f32, -1.0, 0.0, 1.0, 5.0], vec![5]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0], vec![5]).unwrap();
        let bo = ClampBackward { x, lo: -1.0, hi: 1.0 };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        assert_eq!(dx, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn pow_backward_square() {
        // y = x²; dy/dx = 2x. dx = dy * 2x.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let bo = PowBackward { x, p: 2.0 };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[2.0, 4.0, 6.0], 1e-5);
    }

    #[test]
    fn pow_backward_cube() {
        // y = x³; dy/dx = 3x². dx = dy * 3x².
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let bo = PowBackward { x, p: 3.0 };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[3.0, 12.0], 1e-5);
    }

    #[test]
    fn pow_backward_one_half_via_sqrt_derivative() {
        // y = x^(1/2); dy/dx = (1/2) * x^(-1/2) = 1/(2 * √x).
        let x = Tensor::from_slice(&[1.0f32, 4.0, 9.0], vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![3]).unwrap();
        let bo = PowBackward { x, p: 0.5 };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[0.5, 0.25, 1.0 / 6.0], 1e-5);
    }

    #[test]
    fn pow_backward_finite_difference() {
        use kiln_tensor::ops::pow;
        let x_data = vec![1.5f32, -0.7, 2.3];
        let x = Tensor::from_slice(&x_data, vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![3]).unwrap();
        let bo = PowBackward { x: x.clone(), p: 2.0 };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        let loss = |x_vec: &[f32]| -> f32 {
            let xt = Tensor::from_slice(x_vec, vec![3]).unwrap();
            load_f32(&pow(&xt, 2.0).unwrap()).unwrap().iter().sum()
        };
        let step = 1e-3;
        let mut fd = Vec::with_capacity(3);
        for i in 0..3 {
            let mut up = x_data.clone();
            up[i] += step;
            let mut dn = x_data.clone();
            dn[i] -= step;
            fd.push((loss(&up) - loss(&dn)) / (2.0 * step));
        }
        approx(&dx, &fd, 1e-3);
    }

    #[test]
    fn op_metadata() {
        let one = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let cb = ClampBackward {
            x: one.clone(),
            lo: -1.0,
            hi: 1.0,
        };
        assert_eq!(cb.name(), "clamp_backward");
        assert_eq!(cb.input_count(), 1);
        assert!(cb.requires_input(0));

        let pb = PowBackward { x: one, p: 2.0 };
        assert_eq!(pb.name(), "pow_backward");
        assert_eq!(pb.input_count(), 1);
    }
}
