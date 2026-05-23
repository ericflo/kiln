//! Matrix multiplication — CPU reference path.
//!
//! Standard `[..., M, K] @ [..., K, N] = [..., M, N]` matmul.
//! **Naive O(M·N·K) triple-loop**: this is the canonical numerical
//! reference every other backend's parity test compares against
//! (`kiln-blas` cublasLt path, `kiln-mps` MPSMatrixMultiplication,
//! `kiln-vulkan-blas` compute pipelines).
//!
//! # Why naive
//!
//! Per the issue's Phase 1 bullet:
//!
//! > CPU storage: `Vec<T>` — designated as the **canonical numerical
//! > reference** every backend's per-op parity test compares against.
//! > **Also a real tested backend** ...
//!
//! Speed isn't the goal here — correctness is. The naive loop has
//! deterministic iteration order, F32 accumulation, and no
//! algorithm-dependent ULP variance. Phase 2's `kiln-blas` CUDA path
//! tunes against this on the same shape.
//!
//! Future optimization (block-blocked + SIMD) lands in a separate PR
//! once we have profiling evidence that CPU matmul is a workload
//! bottleneck (it isn't today — GPU is the production path).
//!
//! # Determinism
//!
//! `Constructive`. Fixed iteration order over (m, n, k); each output
//! element is computed by exactly one inner loop. Bit-identical at
//! the same input dtype.
//!
//! # F32 accumulation
//!
//! For BF16/F16 inputs, the inner loop accumulates in F32 regardless,
//! then casts back on store. This matches the production GPU path's
//! `CUBLAS_COMPUTE_32F` flag and the established
//! `forward.rs:3454,3517` F32-promotion idiom.

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// 2-D or batched-3-D matrix multiplication.
#[derive(Debug, Default, Clone, Copy)]
pub struct MatmulOp;

impl DeviceOp2 for MatmulOp {
    fn name(&self) -> &'static str {
        "matmul"
    }

    fn determinism(&self) -> Determinism {
        // Forward is constructive (fixed-loop). Backward's dW path is
        // tolerance-bounded (per-column atomicAdd on some backends);
        // bwd() returns None today and lands under kiln-autograd.
        Determinism::Constructive
    }

    fn cpu_fwd(&self, a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
        let (a_shape, b_shape) = (a.shape(), b.shape());
        validate(a, b)?;

        // Resolve batched dims (the leading axes), M, N, K.
        let a_rank = a.rank();
        let m = a_shape[a_rank - 2];
        let k = a_shape[a_rank - 1];
        let n = b_shape[b.rank() - 1];

        let dtype = a.dtype();
        let a_cpu = downcast_cpu(a, "a")?;
        let b_cpu = downcast_cpu(b, "b")?;
        let a_bytes = a_cpu.as_bytes();
        let b_bytes = b_cpu.as_bytes();

        // Batch axes are the rank-2 leading dims. Both inputs share
        // the same leading shape (we've validated that already).
        let batch_shape: Vec<usize> = a_shape[..a_rank - 2].to_vec();
        let batch: usize = batch_shape.iter().product::<usize>().max(1);

        let per = dtype.size_in_bytes();
        let a_stride_batch = m * k * per;
        let b_stride_batch = k * n * per;
        let out_stride_batch = m * n * per;
        let mut out_bytes = vec![0u8; batch * out_stride_batch];

        for batch_i in 0..batch {
            let a_off = batch_i * a_stride_batch;
            let b_off = batch_i * b_stride_batch;
            let o_off = batch_i * out_stride_batch;
            matmul_2d_into(
                dtype,
                &a_bytes[a_off..a_off + a_stride_batch],
                &b_bytes[b_off..b_off + b_stride_batch],
                m,
                n,
                k,
                &mut out_bytes[o_off..o_off + out_stride_batch],
            )?;
        }

        let mut out_shape = batch_shape;
        out_shape.push(m);
        out_shape.push(n);
        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let out = Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())?;
        Ok(Some(out))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Dispatch `MatmulOp`. Mirrors candle's `Tensor::matmul`.
///
/// Shapes:
/// - 2-D: `[M, K] @ [K, N] = [M, N]`
/// - 3-D batched: `[B, M, K] @ [B, K, N] = [B, M, N]`
/// - Higher rank: leading dims must match exactly; trailing two are
///   matmul.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    dispatch2(&MatmulOp, a, b)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(a: &Tensor, b: &Tensor) -> Result<()> {
    let (ar, br) = (a.rank(), b.rank());
    if ar < 2 || br < 2 {
        bail!(
            "MatmulOp: both inputs must have rank ≥ 2, got a.rank={ar}, b.rank={br}"
        );
    }
    if ar != br {
        bail!(
            "MatmulOp: rank mismatch: a={ar}, b={br} (Phase 1.18 requires equal ranks)"
        );
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..ar - 2 {
        if a_shape[axis] != b_shape[axis] {
            bail!(
                "MatmulOp: leading axis {axis} mismatch: a={}, b={}",
                a_shape[axis],
                b_shape[axis]
            );
        }
    }
    let k_a = a_shape[ar - 1];
    let k_b = b_shape[br - 2];
    if k_a != k_b {
        bail!(
            "MatmulOp: contraction dim mismatch: a.shape[..,K]={k_a} vs b.shape[..,K,N]={k_b}"
        );
    }
    if a.dtype() != b.dtype() {
        bail!(
            "MatmulOp: dtype mismatch: a={}, b={} (Phase 1.18 requires equal dtypes)",
            a.dtype(),
            b.dtype()
        );
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "MatmulOp: dtype must be F32/BF16/F16, got {}",
            a.dtype()
        );
    }
    if !a.is_contiguous() {
        bail!("MatmulOp: a must be contiguous");
    }
    if !b.is_contiguous() {
        bail!("MatmulOp: b must be contiguous");
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("MatmulOp: {label} storage must be CpuStorage")))
}

fn matmul_2d_into(
    dtype: DType,
    a: &[u8],
    b: &[u8],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [u8],
) -> Result<()> {
    // F32 accumulation regardless of input dtype.
    let mut acc = vec![0.0f32; m * n];
    for mi in 0..m {
        for ki in 0..k {
            let a_v = read_one_f32(dtype, a, mi * k + ki);
            // Inner: out[mi, *] += a[mi, ki] * b[ki, *]
            for ni in 0..n {
                let b_v = read_one_f32(dtype, b, ki * n + ni);
                acc[mi * n + ni] += a_v * b_v;
            }
        }
    }
    // Cast back to dtype.
    match dtype {
        DType::F32 => {
            for i in 0..m * n {
                out[i * 4..i * 4 + 4].copy_from_slice(&acc[i].to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..m * n {
                out[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(acc[i]).to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..m * n {
                out[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(acc[i]).to_le_bytes());
            }
        }
        _ => unreachable!("validate() rejects non-float dtypes"),
    }
    Ok(())
}

fn read_one_f32(dtype: DType, bytes: &[u8], i: usize) -> f32 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
        DType::BF16 => {
            half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
        }
        DType::F16 => {
            half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    fn approx(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    #[test]
    fn matmul_2d_2x3_times_3x2() {
        // A = [[1, 2, 3], [4, 5, 6]]   B = [[7, 8], [9, 10], [11, 12]]
        // C = A @ B = [[58, 64], [139, 154]]
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let got = read_f32(&c);
        assert_eq!(got, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_identity_left() {
        // I @ B = B for I = identity(M).
        let identity = Tensor::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0],
            vec![2, 2],
        )
        .unwrap();
        let b = Tensor::from_slice(&[3.0f32, 7.0, 5.0, 11.0], vec![2, 2]).unwrap();
        let c = matmul(&identity, &b).unwrap();
        assert_eq!(read_f32(&c), vec![3.0, 7.0, 5.0, 11.0]);
    }

    #[test]
    fn matmul_3d_batched() {
        // Batch=2; each batch is a [2, 2] @ [2, 2].
        // Batch 0: [[1, 0], [0, 1]] @ [[5, 6], [7, 8]] = [[5, 6], [7, 8]]
        // Batch 1: [[2, 0], [0, 2]] @ [[5, 6], [7, 8]] = [[10, 12], [14, 16]]
        let a = Tensor::from_slice(
            &[
                1.0f32, 0.0, 0.0, 1.0, // batch 0
                2.0, 0.0, 0.0, 2.0, // batch 1
            ],
            vec![2, 2, 2],
        )
        .unwrap();
        let b = Tensor::from_slice(
            &[
                5.0f32, 6.0, 7.0, 8.0, // batch 0
                5.0, 6.0, 7.0, 8.0, // batch 1
            ],
            vec![2, 2, 2],
        )
        .unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(
            read_f32(&c),
            vec![5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        );
    }

    #[test]
    fn matmul_bf16_path_promotes_to_f32() {
        // [2, 2] @ [2, 2] in BF16. F32 accumulation should keep ULP error small.
        let a_bf16: Vec<half::bf16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let b_bf16: Vec<half::bf16> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let a = Tensor::from_slice(&a_bf16, vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&b_bf16, vec![2, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.dtype(), DType::BF16);
        let cpu = c.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (i, &e) in expected.iter().enumerate() {
            let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
            assert!(approx(v, e, 5e-1), "bf16 matmul[{i}]={v}, expected {e}");
        }
    }

    #[test]
    fn matmul_rejects_rank_mismatch() {
        let a = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let b = Tensor::zeros_cpu(vec![2, 3, 4], DType::F32);
        let e = matmul(&a, &b).unwrap_err();
        assert!(e.to_string().contains("rank mismatch"));
    }

    #[test]
    fn matmul_rejects_contraction_mismatch() {
        let a = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let b = Tensor::zeros_cpu(vec![5, 6], DType::F32);
        let e = matmul(&a, &b).unwrap_err();
        assert!(e.to_string().contains("contraction"));
    }

    #[test]
    fn matmul_rejects_leading_axis_mismatch() {
        let a = Tensor::zeros_cpu(vec![2, 3, 4], DType::F32);
        let b = Tensor::zeros_cpu(vec![3, 4, 5], DType::F32);
        let e = matmul(&a, &b).unwrap_err();
        assert!(e.to_string().contains("leading axis"));
    }

    #[test]
    fn matmul_rejects_bad_dtype() {
        let a = Tensor::from_slice(&[1u32, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[1u32, 2, 3, 4], vec![2, 2]).unwrap();
        let e = matmul(&a, &b).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }

    #[test]
    fn op_metadata() {
        let op = MatmulOp;
        assert_eq!(op.name(), "matmul");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
    }
}
