//! Reductions: `sum_all`, `mean_all`, `sum_axis`, `mean_axis`.
//!
//! Foundation for the Phase 6b matmul-bwd dW gradient accumulation
//! pattern (`dW = sum_over_batch(x^T @ dC)` collapses to a sum
//! reduction over the leading batch axes), and for the residual /
//! checkpoint integrity checks Phase 9's bench-gate runs.
//!
//! # F32-promoted reduction
//!
//! Regardless of input dtype, the reduction sum accumulates in F32;
//! the result casts back to the input dtype on store. This matches
//! `kiln-rmsnorm-kernel/src/lib.rs:5024-5029`'s tolerance note that
//! "the BF16 forward is bit-stable in the reduction order (fixed-tree)
//! and the cast-back is the only source of per-element ULP variation."
//!
//! # Determinism
//!
//! `Constructive`. Reductions walk a fixed iteration order (the
//! shape's row-major traversal); same input → same bit output at
//! the same dtype.

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Reduction kind: sum vs mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    Sum,
    Mean,
}

impl ReductionKind {
    pub const fn name(self) -> &'static str {
        match self {
            ReductionKind::Sum => "sum",
            ReductionKind::Mean => "mean",
        }
    }
}

/// Reduction scope: all-axes (scalar output) vs single-axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionScope {
    /// Sum/mean over every element. Output is rank-0 (scalar).
    All,
    /// Sum/mean over a single axis. Output shape = input shape with
    /// the reduced axis removed (keepdim=false).
    Axis(usize),
}

/// Reduction op handle.
#[derive(Debug, Clone, Copy)]
pub struct ReduceOp {
    kind: ReductionKind,
    scope: ReductionScope,
}

impl ReduceOp {
    pub const fn new(kind: ReductionKind, scope: ReductionScope) -> Self {
        ReduceOp { kind, scope }
    }
    pub const fn kind(self) -> ReductionKind {
        self.kind
    }
    pub const fn scope(self) -> ReductionScope {
        self.scope
    }
}

impl DeviceOp1 for ReduceOp {
    fn name(&self) -> &'static str {
        match (self.kind, self.scope) {
            (ReductionKind::Sum, ReductionScope::All) => "sum_all",
            (ReductionKind::Mean, ReductionScope::All) => "mean_all",
            (ReductionKind::Sum, ReductionScope::Axis(_)) => "sum_axis",
            (ReductionKind::Mean, ReductionScope::Axis(_)) => "mean_axis",
        }
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "ReduceOp({}): dtype must be F32/BF16/F16, got {}",
                self.name(),
                x.dtype()
            );
        }
        if !x.is_contiguous() {
            bail!("ReduceOp({}): input must be contiguous", self.name());
        }

        let dtype = x.dtype();
        let xv = load_all_f32(x)?;

        match self.scope {
            ReductionScope::All => {
                let sum: f32 = xv.iter().sum();
                let value = match self.kind {
                    ReductionKind::Sum => sum,
                    ReductionKind::Mean => {
                        if xv.is_empty() {
                            bail!(
                                "ReduceOp({}): cannot compute mean of zero-element tensor",
                                self.name()
                            );
                        }
                        sum / xv.len() as f32
                    }
                };
                build_scalar_tensor(dtype, value)
            }
            ReductionScope::Axis(axis) => {
                if axis >= x.rank() {
                    bail!(
                        "ReduceOp({}): axis {axis} out of bounds (rank {})",
                        self.name(),
                        x.rank()
                    );
                }
                reduce_axis(x, axis, self.kind, &xv).map(Some)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // CUDA backend currently supports only sum/mean over the
        // trailing axis. Other scopes/kinds fall through to the CPU
        // path via `Ok(None)`.
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        let rank = x.rank();
        if rank == 0 {
            return Ok(None);
        }
        // CUDA backend supports sum/mean over any single axis
        // (issue #1082 extended this from last-axis-only). `All`
        // still falls through to CPU.
        let axis = match self.scope {
            ReductionScope::Axis(a) => a,
            ReductionScope::All => return Ok(None),
        };
        if axis >= rank {
            return Ok(None);
        }
        let axis_dim = x.shape()[axis];
        if axis_dim == 0 {
            return Ok(None);
        }
        let out = match self.kind {
            ReductionKind::Sum => crate::cuda_sum_axis(x, axis)?,
            ReductionKind::Mean => crate::cuda_mean_axis(x, axis)?,
        };
        Ok(Some(out))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// `out = sum(x)` — scalar output.
pub fn sum_all(x: &Tensor) -> Result<Tensor> {
    dispatch1(&ReduceOp::new(ReductionKind::Sum, ReductionScope::All), x)
}

/// `out = mean(x)` — scalar output.
pub fn mean_all(x: &Tensor) -> Result<Tensor> {
    dispatch1(&ReduceOp::new(ReductionKind::Mean, ReductionScope::All), x)
}

/// `out = sum(x, axis=A)` — shape `x.shape` with axis A removed.
pub fn sum_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    dispatch1(
        &ReduceOp::new(ReductionKind::Sum, ReductionScope::Axis(axis)),
        x,
    )
}

/// `out = mean(x, axis=A)` — shape `x.shape` with axis A removed.
pub fn mean_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    dispatch1(
        &ReduceOp::new(ReductionKind::Mean, ReductionScope::Axis(axis)),
        x,
    )
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn load_all_f32(x: &Tensor) -> Result<Vec<f32>> {
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("ReduceOp: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = x.element_count();
    let mut out = Vec::with_capacity(n);
    match x.dtype() {
        DType::F32 => {
            for i in 0..n {
                out.push(f32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
            }
        }
        DType::BF16 => {
            for i in 0..n {
                out.push(
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        DType::F16 => {
            for i in 0..n {
                out.push(
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        _ => unreachable!("validate rejects non-float dtypes"),
    }
    Ok(out)
}

fn build_scalar_tensor(dtype: DType, value: f32) -> Result<Option<Tensor>> {
    let bytes = match dtype {
        DType::F32 => value.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(value).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Ok(Some(Tensor::from_parts(
        storage,
        Layout::contiguous(Vec::<usize>::new()),
        TensorId::next(),
    )?))
}

fn reduce_axis(
    x: &Tensor,
    axis: usize,
    kind: ReductionKind,
    xv: &[f32],
) -> Result<Tensor> {
    let shape = x.shape();
    let rank = shape.len();
    let reduced_dim = shape[axis];

    // Outer = product of axes [0..axis]; inner = product of axes [axis+1..rank].
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    // Defensive: outer or inner = 0 means zero-element tensor; for
    // axis-reduction this still has well-defined output shape.
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_size = outer * inner;
    let mut out_f32 = vec![0.0_f32; out_size];

    // Standard 3-loop traversal: for each (outer, inner), sum over
    // the reduced axis.
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0_f32;
            for r in 0..reduced_dim {
                let idx = (o * reduced_dim + r) * inner + i;
                acc += xv[idx];
            }
            out_f32[o * inner + i] = acc;
        }
    }

    if matches!(kind, ReductionKind::Mean) {
        if reduced_dim == 0 {
            return Err(Error::from_str(
                "ReduceOp(mean_axis): reduced dim is 0; would divide by zero",
            ));
        }
        let inv = 1.0_f32 / reduced_dim as f32;
        for v in out_f32.iter_mut() {
            *v *= inv;
        }
    }

    // Store back at input dtype.
    let dtype = x.dtype();
    let per = dtype.size_in_bytes();
    let mut out_bytes = vec![0u8; out_size * per];
    match dtype {
        DType::F32 => {
            for (i, v) in out_f32.iter().enumerate() {
                out_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, v) in out_f32.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, v) in out_f32.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32_scalar(t: &Tensor) -> f32 {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[0..4].try_into().unwrap())
    }

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn sum_all_simple() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let s = sum_all(&x).unwrap();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.dtype(), DType::F32);
        assert!((read_f32_scalar(&s) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn mean_all_simple() {
        let x = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let m = mean_all(&x).unwrap();
        assert_eq!(m.rank(), 0);
        assert!((read_f32_scalar(&m) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn sum_axis_2d_axis0() {
        // [[1, 2, 3], [4, 5, 6]] sum over axis 0 -> [5, 7, 9]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = sum_axis(&x, 0).unwrap();
        assert_eq!(s.shape(), &[3]);
        let got = read_f32(&s);
        for (g, e) in got.iter().zip([5.0_f32, 7.0, 9.0].iter()) {
            assert!((g - e).abs() < 1e-6);
        }
    }

    #[test]
    fn sum_axis_2d_axis1() {
        // [[1, 2, 3], [4, 5, 6]] sum over axis 1 -> [6, 15]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = sum_axis(&x, 1).unwrap();
        assert_eq!(s.shape(), &[2]);
        let got = read_f32(&s);
        for (g, e) in got.iter().zip([6.0_f32, 15.0].iter()) {
            assert!((g - e).abs() < 1e-6);
        }
    }

    #[test]
    fn mean_axis_2d() {
        // [[2, 4, 6], [8, 10, 12]] mean over axis 1 -> [4, 10]
        let x = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3]).unwrap();
        let m = mean_axis(&x, 1).unwrap();
        assert_eq!(m.shape(), &[2]);
        let got = read_f32(&m);
        for (g, e) in got.iter().zip([4.0_f32, 10.0].iter()) {
            assert!((g - e).abs() < 1e-6);
        }
    }

    #[test]
    fn sum_axis_3d_middle_axis() {
        // shape [2, 3, 2]; sum over axis 1.
        // Inputs: 12 values 0..12; reshape conceptually:
        // batch 0: [[0,1], [2,3], [4,5]] -> sum axis 1 = [6, 9]
        // batch 1: [[6,7], [8,9], [10,11]] -> sum axis 1 = [24, 27]
        // out shape [2, 2] = [6, 9, 24, 27]
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&v, vec![2, 3, 2]).unwrap();
        let s = sum_axis(&x, 1).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        let got = read_f32(&s);
        assert_eq!(got, vec![6.0, 9.0, 24.0, 27.0]);
    }

    #[test]
    fn sum_all_bf16_promotes() {
        let xv: Vec<half::bf16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&xv, vec![4]).unwrap();
        let s = sum_all(&x).unwrap();
        assert_eq!(s.dtype(), DType::BF16);
        let cpu = s.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let v = half::bf16::from_le_bytes(cpu.as_bytes()[0..2].try_into().unwrap()).to_f32();
        assert!((v - 10.0).abs() < 1e-2);
    }

    #[test]
    fn mean_all_zero_elements_errors() {
        let x = Tensor::zeros_cpu(vec![0], DType::F32);
        let e = mean_all(&x).unwrap_err();
        assert!(e.to_string().contains("zero-element"));
    }

    #[test]
    fn axis_out_of_bounds_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = sum_axis(&x, 5).unwrap_err();
        assert!(e.to_string().contains("out of bounds"));
    }

    #[test]
    fn op_metadata() {
        let s = ReduceOp::new(ReductionKind::Sum, ReductionScope::All);
        assert_eq!(s.name(), "sum_all");
        assert!(s.determinism().is_constructive());

        let m = ReduceOp::new(ReductionKind::Mean, ReductionScope::Axis(0));
        assert_eq!(m.name(), "mean_axis");
        assert_eq!(m.kind(), ReductionKind::Mean);
        match m.scope() {
            ReductionScope::Axis(0) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn reduction_kind_name_strings() {
        assert_eq!(ReductionKind::Sum.name(), "sum");
        assert_eq!(ReductionKind::Mean.name(), "mean");
    }
}
