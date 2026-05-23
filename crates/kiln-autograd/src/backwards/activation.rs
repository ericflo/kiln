//! Backwards for elementwise activations + softmax.
//!
//! - **SigmoidBackward** — `dx = dy * y * (1 - y)`. Saves forward `y`.
//! - **SiluBackward** — `dx = dy * (σ + x*σ*(1-σ))` where `σ = sigmoid(x)`.
//!   Saves `x`.
//! - **GeluBackward** — `dx = dy * d(gelu)/dx` (tanh approximation).
//!   Saves `x` and recomputes `tanh(arg)` in backward.
//! - **SoftmaxLastDimBackward** — `dx_i = y_i * (dy_i - sum_j(y_j * dy_j))`
//!   per row over the trailing axis. Saves forward `y`.
//!
//! All three operate on F32/BF16/F16 inputs. Internal compute is F32 to
//! match the kiln-tensor forward F32-promotion idiom; outputs are
//! cast back to the input dtype on store.
//!
//! Softmax backward is implemented at byte level rather than as a
//! composition of `mul + sum_axis + sub + mul` because kiln-tensor
//! Phase 1.x does not yet support broadcast; expressing
//! `dy - sum_j(y*dy)` requires expanding the per-row sum back to the
//! full shape, which the broadcast-less op surface cannot do
//! ergonomically. The byte-level path is the reference until Phase
//! 2.x adds an explicit `expand` op.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

// ----------------------------------------------------------------------
// Helpers (shared with future backward ops).
// ----------------------------------------------------------------------

fn validate_same(a: &Tensor, b: &Tensor, name: &str) -> Result<()> {
    if a.shape() != b.shape() {
        bail!(
            "{name}: shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
    }
    if a.dtype() != b.dtype() {
        bail!(
            "{name}: dtype mismatch: {} vs {}",
            a.dtype(),
            b.dtype()
        );
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
        .ok_or_else(|| Error::from_str("activation_backward: storage must be CpuStorage"))?;
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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ----------------------------------------------------------------------
// SigmoidBackward
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct SigmoidBackward {
    /// Forward output `y = sigmoid(x)`. Same shape as input.
    pub y: Tensor,
}

impl BackwardOp for SigmoidBackward {
    fn name(&self) -> &'static str {
        "sigmoid_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.y, grad_output, "SigmoidBackward")?;
        let y = load_f32(&self.y)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = y
            .iter()
            .zip(dy.iter())
            .map(|(&yi, &dyi)| dyi * yi * (1.0 - yi))
            .collect();
        let out = store_f32(self.y.dtype(), self.y.shape(), &dx)?;
        Ok(vec![Some(out)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // Backward needs saved forward output `y`, not the input.
        false
    }
}

// ----------------------------------------------------------------------
// SiluBackward
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct SiluBackward {
    /// Forward input `x`. Same shape as output.
    pub x: Tensor,
}

impl BackwardOp for SiluBackward {
    fn name(&self) -> &'static str {
        "silu_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "SiluBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| {
                let s = sigmoid(xi);
                dyi * (s + xi * s * (1.0 - s))
            })
            .collect();
        let out = store_f32(self.x.dtype(), self.x.shape(), &dx)?;
        Ok(vec![Some(out)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

// ----------------------------------------------------------------------
// GeluBackward
// ----------------------------------------------------------------------

/// GELU backward (tanh approximation).
///
/// Let `c = √(2/π)`, `arg = c * (x + 0.044715*x³)`, `t = tanh(arg)`.
/// Then `gelu(x) = 0.5 * x * (1 + t)` and:
///
/// ```text
/// d(gelu)/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t²) * c * (1 + 3*0.044715*x²)
/// ```
///
/// `dx = dy * d(gelu)/dx`. Saves `x`; recomputes `t` in backward.
#[derive(Debug)]
pub struct GeluBackward {
    /// Forward input `x` — same shape as output.
    pub x: Tensor,
}

impl BackwardOp for GeluBackward {
    fn name(&self) -> &'static str {
        "gelu_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.x, grad_output, "GeluBackward")?;
        let x = load_f32(&self.x)?;
        let dy = load_f32(grad_output)?;
        const C: f32 = 0.7978845608_f32; // √(2/π)
        let dx: Vec<f32> = x
            .iter()
            .zip(dy.iter())
            .map(|(&xi, &dyi)| {
                let arg = C * (xi + 0.044715 * xi * xi * xi);
                let t = arg.tanh();
                let dgdx =
                    0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * C * (1.0 + 3.0 * 0.044715 * xi * xi);
                dyi * dgdx
            })
            .collect();
        let out = store_f32(self.x.dtype(), self.x.shape(), &dx)?;
        Ok(vec![Some(out)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

// ----------------------------------------------------------------------
// SoftmaxLastDimBackward
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct SoftmaxLastDimBackward {
    /// Forward output `y = softmax(x)` over the trailing axis.
    pub y: Tensor,
}

impl BackwardOp for SoftmaxLastDimBackward {
    fn name(&self) -> &'static str {
        "softmax_last_dim_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        validate_same(&self.y, grad_output, "SoftmaxLastDimBackward")?;
        let shape = self.y.shape();
        let last = *shape.last().unwrap();
        let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let y = load_f32(&self.y)?;
        let dy = load_f32(grad_output)?;
        let mut dx = vec![0.0f32; y.len()];
        for r in 0..outer {
            let base = r * last;
            // s = sum_j y_j * dy_j over this row.
            let mut s = 0.0f32;
            for j in 0..last {
                s += y[base + j] * dy[base + j];
            }
            // dx_i = y_i * (dy_i - s)
            for i in 0..last {
                dx[base + i] = y[base + i] * (dy[base + i] - s);
            }
        }
        let out = store_f32(self.y.dtype(), shape, &dx)?;
        Ok(vec![Some(out)])
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

    // ─── SigmoidBackward ─────────────────────────────────────────

    #[test]
    fn sigmoid_backward_at_zero() {
        // sigmoid(0) = 0.5 → dx = dy * 0.5 * 0.5 = dy * 0.25
        let y = Tensor::from_slice(&[0.5f32, 0.5, 0.5], vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 4.0], vec![3]).unwrap();
        let bo = SigmoidBackward { y };
        let grads = bo.apply(&dy).unwrap();
        let dx = grads[0].as_ref().unwrap();
        approx(&load_f32(dx).unwrap(), &[0.25, 0.5, 1.0], 1e-6);
    }

    #[test]
    fn sigmoid_backward_finite_difference() {
        // f(x) = sigmoid(x); f'(x) = y * (1 - y).
        // Pick x = ln(3) so sigmoid(x) = 0.75 → f' = 0.75 * 0.25 = 0.1875.
        let y_val = 3.0f32 / 4.0;
        let y = Tensor::from_slice(&[y_val], vec![1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = SigmoidBackward { y };
        let grads = bo.apply(&dy).unwrap();
        approx(&load_f32(grads[0].as_ref().unwrap()).unwrap(), &[0.1875], 1e-6);
    }

    #[test]
    fn sigmoid_backward_shape_preserved() {
        let y = Tensor::from_slice(&[0.5f32; 6], vec![2, 3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 6], vec![2, 3]).unwrap();
        let bo = SigmoidBackward { y };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(dx.shape(), &[2, 3]);
    }

    // ─── SiluBackward ────────────────────────────────────────────

    #[test]
    fn silu_backward_at_zero() {
        // x = 0 → σ = 0.5; dx = dy * (0.5 + 0 * ...) = dy * 0.5.
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 4.0], vec![3]).unwrap();
        let bo = SiluBackward { x };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        approx(&load_f32(&dx).unwrap(), &[0.5, 1.0, 2.0], 1e-6);
    }

    #[test]
    fn silu_backward_finite_difference() {
        // At x = 1: σ ≈ 0.7311; derivative = σ + x*σ*(1-σ) =
        // 0.7311 + 1 * 0.7311 * 0.2689 ≈ 0.7311 + 0.1966 ≈ 0.9277.
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = SiluBackward { x };
        let dx_val = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap()[0];
        assert!((dx_val - 0.9277).abs() < 1e-3, "got {dx_val}");
    }

    // ─── SoftmaxLastDimBackward ──────────────────────────────────

    #[test]
    fn softmax_backward_balanced_input_with_constant_grad() {
        // Forward y = softmax([0, 0, 0]) = [1/3, 1/3, 1/3].
        // With dy = [1, 1, 1], s = sum(y_i) = 1 ... wait,
        // s = sum_j (y_j * dy_j) = 1/3 + 1/3 + 1/3 = 1.
        // dx_i = y_i * (dy_i - 1) = 1/3 * (1 - 1) = 0 for all i.
        let y = Tensor::from_slice(&[1.0f32 / 3.0; 3], vec![1, 3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![1, 3]).unwrap();
        let bo = SoftmaxLastDimBackward { y };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        approx(&load_f32(&dx).unwrap(), &[0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn softmax_backward_finite_difference() {
        // y = [0.5, 0.3, 0.2]; dy = [1, 0, 0].
        // s = 0.5 * 1 + 0.3 * 0 + 0.2 * 0 = 0.5
        // dx_0 = 0.5 * (1 - 0.5) = 0.25
        // dx_1 = 0.3 * (0 - 0.5) = -0.15
        // dx_2 = 0.2 * (0 - 0.5) = -0.10
        let y = Tensor::from_slice(&[0.5f32, 0.3, 0.2], vec![1, 3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let bo = SoftmaxLastDimBackward { y };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        approx(&load_f32(&dx).unwrap(), &[0.25, -0.15, -0.10], 1e-6);
    }

    #[test]
    fn softmax_backward_batched_rows_independent() {
        // Two rows, different gradients → backward should treat each row
        // independently (the s-reduction is per-row).
        let y_row0 = [0.5f32, 0.5];
        let y_row1 = [0.8f32, 0.2];
        let y_data: Vec<f32> = y_row0.iter().chain(y_row1.iter()).copied().collect();
        let dy_data: Vec<f32> = [1.0f32, 0.0, 0.0, 1.0].to_vec();
        let y = Tensor::from_slice(&y_data, vec![2, 2]).unwrap();
        let dy = Tensor::from_slice(&dy_data, vec![2, 2]).unwrap();
        let bo = SoftmaxLastDimBackward { y };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        // Row 0: s = 0.5 * 1 + 0.5 * 0 = 0.5
        //   dx = 0.5 * (1 - 0.5) = 0.25, 0.5 * (0 - 0.5) = -0.25
        // Row 1: s = 0.8 * 0 + 0.2 * 1 = 0.2
        //   dx = 0.8 * (0 - 0.2) = -0.16, 0.2 * (1 - 0.2) = 0.16
        approx(&load_f32(&dx).unwrap(), &[0.25, -0.25, -0.16, 0.16], 1e-6);
    }

    #[test]
    fn softmax_backward_shape_dtype_mismatch_errors() {
        let y = Tensor::from_slice(&[0.5f32, 0.5], vec![1, 2]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![1, 3]).unwrap();
        let e = SoftmaxLastDimBackward { y }.apply(&dy).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }

    #[test]
    fn gelu_backward_at_zero() {
        // arg(0) = 0, tanh(0) = 0; d/dx = 0.5 * (1 + 0) + 0 = 0.5.
        let x = Tensor::from_slice(&[0.0f32, 0.0], vec![2]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bo = GeluBackward { x };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        approx(&dx, &[0.5, 1.0], 1e-6);
    }

    #[test]
    fn gelu_backward_finite_difference() {
        use kiln_tensor::ops::gelu;
        let x_data = vec![0.5f32, -1.5, 2.0];
        let x = Tensor::from_slice(&x_data, vec![3]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 3], vec![3]).unwrap();
        let bo = GeluBackward { x: x.clone() };
        let dx = load_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap()).unwrap();
        let loss = |xv: &[f32]| -> f32 {
            let xt = Tensor::from_slice(xv, vec![3]).unwrap();
            let y = gelu(&xt).unwrap();
            load_f32(&y).unwrap().iter().sum()
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
        approx(&dx, &fd, 5e-3);
    }

    #[test]
    fn op_metadata() {
        let one = Tensor::from_slice(&[0.5f32], vec![1]).unwrap();
        assert_eq!(SigmoidBackward { y: one.clone() }.name(), "sigmoid_backward");
        assert_eq!(SiluBackward { x: one.clone() }.name(), "silu_backward");
        assert_eq!(GeluBackward { x: one.clone() }.name(), "gelu_backward");
        assert_eq!(
            SoftmaxLastDimBackward { y: one.clone() }.name(),
            "softmax_last_dim_backward"
        );

        let sb = SigmoidBackward { y: one.clone() };
        assert_eq!(sb.input_count(), 1);
        assert!(!sb.requires_input(0));

        let lb = SiluBackward { x: one.clone() };
        assert_eq!(lb.input_count(), 1);
        assert!(lb.requires_input(0));
    }

    #[test]
    fn bf16_path_round_trips() {
        // BF16 → F32 compute → BF16 store.
        let yv: Vec<half::bf16> = [0.5f32, 0.3, 0.2]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let y = Tensor::from_slice(&yv, vec![1, 3]).unwrap();
        let dyv: Vec<half::bf16> = [1.0f32, 0.0, 0.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let dy = Tensor::from_slice(&dyv, vec![1, 3]).unwrap();
        let bo = SoftmaxLastDimBackward { y };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(dx.dtype(), DType::BF16);
        // BF16 tolerance is much looser.
        approx(&load_f32(&dx).unwrap(), &[0.25, -0.15, -0.10], 5e-3);
    }
}
