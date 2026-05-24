//! `flip` — reverse the order of elements along given axes.
//!
//! `flip(x, axes)` reverses `x` along each axis in `axes`. PyTorch /
//! NumPy parity with `torch.flip(x, dims)` / `np.flip(x, axes)`.
//!
//! # Dispatch
//!
//! Wrapped as a [`DeviceOp1`] so CUDA tensors route through
//! `cuda_index_select_dim0` (for the single-axis `axes=[0]` case)
//! instead of falling through the byte-level CPU loop. Multi-axis
//! or non-zero axis flips fall through to CPU; the bytes are read
//! from CPU storage only in that path.

use std::sync::Arc;

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, Determinism, DeviceOp1, Layout, Result, Storage,
    Tensor, TensorId,
};

/// Flip op — reverses element order along each axis in `axes`.
#[derive(Debug, Clone)]
pub struct FlipOp {
    axes: Vec<usize>,
}

impl FlipOp {
    pub fn new(axes: &[usize]) -> Self {
        FlipOp { axes: axes.to_vec() }
    }
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
}

impl DeviceOp1 for FlipOp {
    fn name(&self) -> &'static str {
        "flip"
    }

    fn determinism(&self) -> Determinism {
        // Each output element is a byte-copy of a specific input
        // element — fixed iteration order, no atomics.
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, &self.axes)?;

        if self.axes.is_empty() {
            // No-op clone with a fresh tensor id.
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }

        let dtype = x.dtype();
        let shape: Vec<usize> = x.shape().to_vec();
        let rank = shape.len();
        let per = dtype.size_in_bytes();
        let n_elem: usize = shape.iter().product();
        let mut out = vec![0u8; n_elem * per];

        // Compute strides (== input strides since shape unchanged).
        let mut strides = vec![1usize; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        let cpu = x.storage();
        let cpu = cpu
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| crate::Error::from_str("flip: storage must be CpuStorage"))?;
        let src = cpu.as_bytes();

        // For each linear input index, decode to coord, flip selected
        // axes, re-encode, copy one element.
        let mut coord = vec![0usize; rank];
        for in_idx in 0..n_elem {
            let mut rem = in_idx;
            for d in 0..rank {
                coord[d] = rem / strides[d];
                rem %= strides[d];
            }
            for &a in &self.axes {
                coord[a] = shape[a] - 1 - coord[a];
            }
            let mut out_off = 0usize;
            for d in 0..rank {
                out_off += coord[d] * strides[d];
            }
            let src_byte = in_idx * per;
            let dst_byte = out_off * per;
            out[dst_byte..dst_byte + per].copy_from_slice(&src[src_byte..src_byte + per]);
        }

        let cpu_out = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu_out);
        Ok(Some(Tensor::from_parts(
            storage,
            Layout::contiguous(shape),
            TensorId::next(),
        )?))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        validate(x, &self.axes)?;

        // No-op fast path: empty axes ⇒ identity reshape (no data
        // movement, stays on device).
        if self.axes.is_empty() {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }

        // We currently only route through CUDA when flipping a
        // single axis-0 reversal. Multi-axis and non-zero-axis flips
        // fall through to CPU until the more general kernel lands.
        if self.axes.len() != 1 || self.axes[0] != 0 {
            return Ok(None);
        }
        if x.dtype().is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }

        let n = x.shape()[0];
        if n == 0 {
            // Empty axis-0 — produce an empty output via reshape.
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }

        // Build reversed indices [n-1, n-2, ..., 1, 0] on CPU as U32,
        // then ship to the same CUDA device as `x` and gather.
        let indices_host: Vec<u32> = (0..n as u32).rev().collect();
        let indices_cpu = Tensor::from_slice(&indices_host, vec![n])?;

        // Resolve the candle CudaDevice from `x`'s storage so the
        // indices land on the matching device.
        let x_storage = x
            .storage()
            .as_any()
            .downcast_ref::<crate::CudaStorage>()
            .ok_or_else(|| crate::Error::from_str("flip::cuda_fwd: x must be CUDA storage"))?;
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
        //   - axes validate cleanly (rank-bounded, no duplicates)
        //   - empty axes is identity (cheap reshape)
        //   - single-axis axis==0 only (matches the CUDA reversed-
        //     gather strategy); multi-axis / non-zero-axis flips
        //     fall through to CPU
        //   - non-packed dtype, contiguous input
        validate(x, &self.axes)?;
        if self.axes.is_empty() {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }
        if self.axes.len() != 1 || self.axes[0] != 0 {
            return Ok(None);
        }
        if x.dtype().is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        let n = x.shape()[0];
        if n == 0 {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }
        // TODO(#1082, phase 4 Metal): implement
        // `crate::metal_flip_dim0(x)` (or reuse a generic flip kernel
        // analogous to the CUDA reversed-gather above). Until that
        // kernel lands, fall through to the CPU path
        // (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. Custom MSL kernel: one threadgroup per output row, one
        //      thread per inner-element; output[i, ...] copies from
        //      input[n - 1 - i, ...]. Single kernel, no index buffer.
        //   2. Reuse `metal_index_select_dim0` once it lands — build a
        //      reversed-index buffer (`n-1..0`) and gather. Mirrors
        //      the CUDA strategy and avoids writing a flip-specific
        //      kernel.
        //   3. MPS Graph: `flip(_:axes:)` for one-shot dispatch.
        //      Heavier per-call overhead than a custom kernel.
        Ok(None)
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd / metal_fwd:
        //   - axes validate cleanly
        //   - empty axes is identity (cheap reshape)
        //   - single-axis axis==0 only (matches the CUDA strategy)
        //   - non-packed dtype, contiguous input
        validate(x, &self.axes)?;
        if self.axes.is_empty() {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }
        if self.axes.len() != 1 || self.axes[0] != 0 {
            return Ok(None);
        }
        if x.dtype().is_packed() {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        let n = x.shape()[0];
        if n == 0 {
            return Ok(Some(x.reshape(x.shape().to_vec())?));
        }
        // TODO(#1082, phase 4 Vulkan): implement
        // `crate::vulkan_flip_dim0(x)` (or reuse a generic flip
        // wrapper analogous to the CUDA reversed-gather above).
        // Until that wrapper lands, fall through to CPU
        // (numerics-correct, performance-wrong).
        // Candidate implementations:
        //   1. SPIR-V compute shader: one workgroup per output row,
        //      one invocation per inner-element; output[i, ...]
        //      copies from input[n - 1 - i, ...]. Single dispatch,
        //      no index buffer round-trip.
        //   2. Reuse `vulkan_index_select_dim0` once it lands —
        //      build a reversed-index buffer and gather. Mirrors
        //      the CUDA strategy.
        //   3. Dtype matrix gap: `VkDType` exposes F32 / BF16 today;
        //      F16 / I64 / U32 inputs need either widening or a
        //      cast wrapper at the dispatch boundary.
        Ok(None)
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Convenience: dispatch `FlipOp` on the given axes.
pub fn flip(x: &Tensor, axes: &[usize]) -> Result<Tensor> {
    dispatch1(&FlipOp::new(axes), x)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(x: &Tensor, axes: &[usize]) -> Result<()> {
    let rank = x.rank();
    for &a in axes {
        if a >= rank {
            bail!("flip: axis {a} out of bounds for rank {rank}");
        }
    }
    let mut seen = vec![false; rank];
    for &a in axes {
        if seen[a] {
            bail!("flip: duplicate axis {a} in {axes:?}");
        }
        seen[a] = true;
    }
    if x.dtype().is_packed() {
        bail!("flip: packed dtype {} not supported", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("flip: input must be contiguous");
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
    fn flip_rank1() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = flip(&x, &[0]).unwrap();
        assert_eq!(read_f32(&y), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn flip_axis_0_2d() {
        // [[1,2],[3,4],[5,6]] flip axis=0 → [[5,6],[3,4],[1,2]]
        let x = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        )
        .unwrap();
        let y = flip(&x, &[0]).unwrap();
        assert_eq!(read_f32(&y), vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn flip_axis_1_2d() {
        // [[1,2,3],[4,5,6]] flip axis=1 → [[3,2,1],[6,5,4]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let y = flip(&x, &[1]).unwrap();
        assert_eq!(read_f32(&y), vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn flip_both_axes_2d() {
        // [[1,2],[3,4]] flip both → [[4,3],[2,1]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = flip(&x, &[0, 1]).unwrap();
        assert_eq!(read_f32(&y), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn flip_empty_axes_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = flip(&x, &[]).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn flip_axis_oob_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = flip(&x, &[5]).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn flip_duplicate_axis_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = flip(&x, &[0, 0]).unwrap_err();
        assert!(e.to_string().contains("duplicate"));
    }

    #[test]
    fn flip_twice_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let once = flip(&x, &[0]).unwrap();
        let twice = flip(&once, &[0]).unwrap();
        assert_eq!(read_f32(&twice), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn flip_3d_middle_axis() {
        // shape [2, 3, 2], flip axis=1.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, vec![2, 3, 2]).unwrap();
        let y = flip(&x, &[1]).unwrap();
        let expected: Vec<f32> = vec![
            4.0, 5.0, 2.0, 3.0, 0.0, 1.0, // first slab axis1 reversed
            10.0, 11.0, 8.0, 9.0, 6.0, 7.0, // second slab axis1 reversed
        ];
        assert_eq!(read_f32(&y), expected);
    }

    #[test]
    fn flip_op_metadata() {
        let op = FlipOp::new(&[0]);
        assert_eq!(op.name(), "flip");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
        assert_eq!(op.axes(), &[0]);
    }

    #[test]
    fn flip_bf16_axis_0() {
        // Round-trip through bf16 storage on axis 0.
        let bf: Vec<half::bf16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![4]).unwrap();
        let y = flip(&x, &[0]).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.shape(), &[4]);
    }
}
