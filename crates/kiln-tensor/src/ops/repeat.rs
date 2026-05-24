//! `repeat` — tile a tensor `n` times along an axis.
//!
//! ```text
//! repeat(x, axis, n).shape[axis] = x.shape[axis] * n
//! ```
//!
//! Different from `broadcast_to` (which only expands size-1 axes).
//! `repeat` tiles **arbitrary-size** axes; the output is
//! conceptually `concat([x, x, …, x] /* n copies */, axis)`.
//!
//! Used for:
//! - **Sequence broadcasting** — replicate a `[1, S, D]` tensor to
//!   `[B, S, D]` (could also use broadcast_to for size-1 axes; this
//!   is the general case)
//! - **Attention bias spreading**
//! - **MoE expert input duplication** when the same hidden state
//!   feeds multiple experts
//! - **`addmm` bias broadcast** — `addmm` repeats a `[1, N]` bias
//!   to `[M, N]` via `repeat(_, 0, M)`. That call site benefits
//!   directly from the CUDA dispatch path below.
//!
//! # Dispatch
//!
//! Wrapped as a [`DeviceOp1`] so CUDA tensors route through
//! `cuda_index_select_dim0` (axis=0 only). Non-zero-axis repeats
//! fall through to CPU until a more general kernel lands.

use std::sync::Arc;

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};

/// Repeat op — tiles `x` `n` times along `axis`.
#[derive(Debug, Clone, Copy)]
pub struct RepeatOp {
    axis: usize,
    n: usize,
}

impl RepeatOp {
    pub const fn new(axis: usize, n: usize) -> Self {
        RepeatOp { axis, n }
    }
    pub const fn axis(self) -> usize {
        self.axis
    }
    pub const fn n(self) -> usize {
        self.n
    }
}

impl DeviceOp1 for RepeatOp {
    fn name(&self) -> &'static str {
        "repeat"
    }

    fn determinism(&self) -> Determinism {
        // Fixed iteration order, no atomics. Each output element
        // is a byte-copy of a specific input element.
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.axis, self.n)?;

        let dtype = x.dtype();
        let per = dtype.size_in_bytes();
        let shape = x.shape();
        let outer: usize = shape[..self.axis].iter().product::<usize>().max(1);
        let axis_in = shape[self.axis];
        let inner: usize = shape[self.axis + 1..].iter().product::<usize>().max(1);

        let cpu = x
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("repeat: storage must be CpuStorage"))?;
        let bytes = cpu.as_bytes();
        let axis_out = axis_in * self.n;
        let mut out = vec![0u8; outer * axis_out * inner * per];

        // For each outer slab, copy the axis-block `n` times into
        // the output's expanded axis dimension.
        for o in 0..outer {
            for rep in 0..self.n {
                let src_start = o * axis_in * inner * per;
                let src_end = src_start + axis_in * inner * per;
                let dst_start = (o * axis_out + rep * axis_in) * inner * per;
                let dst_end = dst_start + axis_in * inner * per;
                out[dst_start..dst_end].copy_from_slice(&bytes[src_start..src_end]);
            }
        }
        let mut out_shape = shape.to_vec();
        out_shape[self.axis] = axis_out;
        let cpu_out = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu_out);
        Ok(Some(Tensor::from_parts(
            storage,
            Layout::contiguous(out_shape),
            TensorId::next(),
        )?))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, self.axis, self.n)?;

        // n=1 is identity — no copy needed, just hand back a fresh
        // tensor id over the same storage. Skips index allocation.
        if self.n == 1 {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }

        // We currently only fast-path axis=0. For arbitrary axes,
        // fall through to CPU until a generic kernel ships.
        if self.axis != 0 {
            return Ok(None);
        }
        if x.dtype().is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }

        let axis_in = x.shape()[0];
        if axis_in == 0 {
            // Empty axis-0 — produce an empty output via reshape
            // (preserving zero rows, just bumping axis-0 to 0 * n = 0).
            let mut out_shape = x.shape().to_vec();
            out_shape[0] = 0;
            return Ok(Some(x.reshape(out_shape)?));
        }

        // Build tiled indices `[0,1,...,d-1, 0,1,...,d-1, ...]` (n
        // copies) on CPU, ship to CUDA, gather.
        let total = axis_in * self.n;
        let mut indices_host: Vec<u32> = Vec::with_capacity(total);
        for _ in 0..self.n {
            for i in 0..axis_in as u32 {
                indices_host.push(i);
            }
        }
        let indices_cpu = Tensor::from_slice(&indices_host, vec![total])?;

        let x_storage = x
            .storage()
            .as_any()
            .downcast_ref::<crate::CudaStorage>()
            .ok_or_else(|| crate::Error::from_str("repeat::cuda_fwd: x must be CUDA storage"))?;
        let candle_device = x_storage.candle_device().clone();
        let device_index = match x.device() {
            crate::Device::Cuda(i) => i,
            _ => return Ok(None),
        };

        let indices_cuda =
            crate::host_to_cuda_copy(&indices_cpu, candle_device, device_index)?;
        Ok(Some(crate::cuda_index_select_dim0(x, &indices_cuda)?))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd so future MSL
        // kernel work can drop in without changing the call site:
        //   - validate(x, axis, n) (rank-bounded, n >= 1)
        //   - n == 1 is identity (cheap reshape, no copy)
        //   - axis == 0 only (matches the CUDA tiled-gather strategy)
        //   - non-packed dtype, contiguous input
        validate(x, self.axis, self.n)?;
        if self.n == 1 {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }
        if self.axis != 0 {
            return Ok(None);
        }
        if x.dtype().is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        let axis_in = x.shape()[0];
        if axis_in == 0 {
            let mut out_shape = x.shape().to_vec();
            out_shape[0] = 0;
            return Ok(Some(x.reshape(out_shape)?));
        }
        // TODO(#1082, phase 4 Metal): implement
        // `crate::metal_repeat_dim0(x, n)` (or reuse a generic
        // repeat kernel analogous to the CUDA tiled-gather above).
        // Until that kernel lands, fall through to the CPU path
        // (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. Custom MSL kernel: one threadgroup per output row,
        //      one thread per inner-element; out[i, ...] copies
        //      from in[i % axis_in, ...]. Single dispatch, no
        //      index buffer round-trip.
        //   2. Reuse `metal_index_select_dim0` once it lands —
        //      build a tiled-index buffer
        //      ([0..d-1] repeated n times) and gather. Mirrors the
        //      CUDA strategy.
        //   3. MPS Graph: `tile(_:withMultiplier:)` for one-shot
        //      dispatch. Higher per-call overhead than a custom
        //      kernel.
        Ok(None)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd / metal_fwd:
        //   - validate(x, axis, n) (rank-bounded, n >= 1)
        //   - n == 1 is identity (cheap reshape, no copy)
        //   - axis == 0 only
        //   - non-packed dtype, contiguous input
        validate(x, self.axis, self.n)?;
        if self.n == 1 {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }
        if self.axis != 0 {
            return Ok(None);
        }
        if x.dtype().is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        let axis_in = x.shape()[0];
        if axis_in == 0 {
            let mut out_shape = x.shape().to_vec();
            out_shape[0] = 0;
            return Ok(Some(x.reshape(out_shape)?));
        }
        // TODO(#1082, phase 4 Vulkan): implement
        // `crate::vulkan_repeat_dim0(x, n)` analogous to the CUDA
        // tiled-gather above. Until that wrapper lands, fall through
        // to CPU (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. SPIR-V compute shader: one workgroup per output row,
        //      one invocation per inner-element; out[i, ...] copies
        //      from in[i % axis_in, ...].
        //   2. Reuse `vulkan_index_select_dim0` once it lands —
        //      build a tiled-index buffer and gather. Mirrors the
        //      CUDA strategy.
        //   3. Dtype matrix gap: `VkDType` exposes F32 / BF16 today;
        //      other dtypes need widening or a cast wrapper at the
        //      dispatch boundary.
        Ok(None)
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Convenience: dispatch `RepeatOp(axis, n)` on `x`.
pub fn repeat(x: &Tensor, axis: usize, n: usize) -> Result<Tensor> {
    dispatch1(&RepeatOp::new(axis, n), x)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(x: &Tensor, axis: usize, n: usize) -> Result<()> {
    if x.rank() == 0 {
        bail!("repeat: input must have rank ≥ 1");
    }
    if axis >= x.rank() {
        bail!(
            "repeat: axis {axis} out of range for rank-{} input",
            x.rank()
        );
    }
    if n == 0 {
        bail!("repeat: n must be > 0");
    }
    if x.dtype().is_packed() {
        bail!("repeat: packed dtype {} not supported", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("repeat: input must be contiguous");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn repeat_rank1_simple() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = repeat(&x, 0, 3).unwrap();
        assert_eq!(y.shape(), &[9]);
        assert_eq!(
            read_f32(&y),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn repeat_rank2_axis_0() {
        // [[1,2],[3,4]] repeated 2x along axis 0 → 4 rows.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = repeat(&x, 0, 2).unwrap();
        assert_eq!(y.shape(), &[4, 2]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn repeat_rank2_axis_1() {
        // [[1,2],[3,4]] repeated 2x along axis 1 → 4 cols per row.
        // Layout: out[r, :] = [a, b, a, b]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = repeat(&x, 1, 2).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn repeat_n_one_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = repeat(&x, 0, 1).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_rank3_middle_axis() {
        // [B=2, H=1, D=2] repeated 3x at axis 1 → [B=2, H=3, D=2].
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 1, 2]).unwrap();
        let y = repeat(&x, 1, 3).unwrap();
        assert_eq!(y.shape(), &[2, 3, 2]);
        // Batch 0: [1, 2] x 3 = [1,2,1,2,1,2]
        // Batch 1: [3, 4] x 3 = [3,4,3,4,3,4]
        assert_eq!(
            read_f32(&y),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn repeat_n_zero_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = repeat(&x, 0, 0).unwrap_err();
        assert!(e.to_string().contains("n must be > 0"));
    }

    #[test]
    fn repeat_axis_out_of_range_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = repeat(&x, 5, 2).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn repeat_bf16_round_trip() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![2]).unwrap();
        let y = repeat(&x, 0, 3).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.shape(), &[6]);
    }

    #[test]
    fn repeat_op_metadata() {
        let op = RepeatOp::new(0, 3);
        assert_eq!(op.name(), "repeat");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
        assert_eq!(op.axis(), 0);
        assert_eq!(op.n(), 3);
    }
}
