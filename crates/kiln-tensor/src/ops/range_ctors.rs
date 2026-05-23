//! Tensor constructors: `linspace` and `arange`.
//!
//! - `linspace(start, stop, n)` — `n` evenly-spaced values from
//!   `start` to `stop` (inclusive). `n >= 1` required.
//! - `arange(start, stop, step)` — values `start, start+step, …` up
//!   to (but not including) `stop`. `step != 0` required.
//!
//! Useful for positional encoding generation, range masking,
//! attention bias construction.
//!
//! Non-differentiable (constructors; no input tensors).

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

fn build(dtype: DType, values: &[f32]) -> Result<Tensor> {
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "range constructor: dtype must be F32/BF16/F16, got {dtype}"
        );
    }
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; values.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(vec![values.len()]), TensorId::next())
}

/// `n` evenly-spaced values from `start` to `stop` (inclusive).
///
/// `n == 1` → `[start]`. `n >= 2` → `[start, …, stop]`.
pub fn linspace(start: f32, stop: f32, n: usize, dtype: DType) -> Result<Tensor> {
    if n == 0 {
        bail!("linspace: n must be > 0");
    }
    let mut vals = Vec::with_capacity(n);
    if n == 1 {
        vals.push(start);
    } else {
        let step = (stop - start) / (n - 1) as f32;
        for i in 0..n {
            vals.push(start + (i as f32) * step);
        }
        // Force exact endpoint to avoid floating-point drift on the
        // last sample (e.g. linspace(0, 1, 11) should end exactly 1.0).
        if let Some(last) = vals.last_mut() {
            *last = stop;
        }
    }
    build(dtype, &vals)
}

/// Half-open range `[start, start+step, start+2*step, …)` up to (but
/// not including) `stop`. `step` must be non-zero. Output dtype is
/// F32 by default.
pub fn arange(start: f32, stop: f32, step: f32, dtype: DType) -> Result<Tensor> {
    if step == 0.0 {
        bail!("arange: step must be non-zero");
    }
    if step.is_nan() || start.is_nan() || stop.is_nan() {
        bail!("arange: start/stop/step must not be NaN");
    }
    let mut vals = Vec::new();
    if step > 0.0 {
        let mut v = start;
        while v < stop {
            vals.push(v);
            v += step;
        }
    } else {
        let mut v = start;
        while v > stop {
            vals.push(v);
            v += step;
        }
    }
    build(dtype, &vals)
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

    // ─── linspace ──────────────────────────────────────────────

    #[test]
    fn linspace_inclusive_endpoints() {
        let t = linspace(0.0, 1.0, 5, DType::F32).unwrap();
        approx(&read_f32(&t), &[0.0, 0.25, 0.5, 0.75, 1.0], 1e-6);
    }

    #[test]
    fn linspace_n_equals_1_returns_start() {
        let t = linspace(2.0, 5.0, 1, DType::F32).unwrap();
        assert_eq!(read_f32(&t), vec![2.0]);
    }

    #[test]
    fn linspace_descending() {
        // start > stop is fine — produces a descending sequence.
        let t = linspace(1.0, 0.0, 3, DType::F32).unwrap();
        approx(&read_f32(&t), &[1.0, 0.5, 0.0], 1e-6);
    }

    #[test]
    fn linspace_n_zero_errors() {
        let e = linspace(0.0, 1.0, 0, DType::F32).unwrap_err();
        assert!(e.to_string().contains("n must be > 0"));
    }

    #[test]
    fn linspace_bf16_dtype() {
        let t = linspace(0.0, 1.0, 3, DType::BF16).unwrap();
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn linspace_endpoint_is_exact() {
        // Floating-point drift in the loop should not affect the
        // final sample — it's pinned to `stop`.
        let t = linspace(0.0, 1.0, 11, DType::F32).unwrap();
        let v = read_f32(&t);
        assert_eq!(v[10], 1.0);
    }

    // ─── arange ────────────────────────────────────────────────

    #[test]
    fn arange_ascending() {
        let t = arange(0.0, 5.0, 1.0, DType::F32).unwrap();
        assert_eq!(read_f32(&t), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn arange_descending() {
        let t = arange(5.0, 0.0, -1.0, DType::F32).unwrap();
        assert_eq!(read_f32(&t), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn arange_fractional_step() {
        let t = arange(0.0, 1.0, 0.25, DType::F32).unwrap();
        approx(&read_f32(&t), &[0.0, 0.25, 0.5, 0.75], 1e-6);
    }

    #[test]
    fn arange_step_zero_errors() {
        let e = arange(0.0, 1.0, 0.0, DType::F32).unwrap_err();
        assert!(e.to_string().contains("step"));
    }

    #[test]
    fn arange_empty_when_start_equals_stop() {
        let t = arange(1.0, 1.0, 1.0, DType::F32).unwrap();
        assert!(read_f32(&t).is_empty());
    }

    #[test]
    fn arange_bf16_dtype() {
        let t = arange(0.0, 4.0, 1.0, DType::BF16).unwrap();
        assert_eq!(t.dtype(), DType::BF16);
        assert_eq!(t.shape(), &[4]);
    }

    #[test]
    fn arange_nan_errors() {
        let e = arange(f32::NAN, 1.0, 1.0, DType::F32).unwrap_err();
        assert!(e.to_string().contains("NaN"));
    }
}
