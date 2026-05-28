//! `index_select` — gather a tensor's slices along `axis` using
//! integer `indices`.
//!
//! Generalized form of [`crate::ops::embedding`] (which is hard-coded
//! to axis 0). Migration target for candle's `Tensor::index_select`
//! at the call sites that select along a non-0 axis (e.g. picking
//! `n_heads` rows from a per-head bias tensor).
//!
//! # Semantics
//!
//! Given:
//! - `input: Tensor` — any rank ≥ 1 with shape `[..., D_axis, ...]`
//! - `axis: usize` — the axis to gather along
//! - `indices: Tensor` — rank ≥ 1, dtype I64 or U32
//!
//! Produces:
//! - `out: Tensor` — shape `[..., indices.shape..., ...]`, replacing
//!   `D_axis` in input.shape with the full shape of `indices`.
//!
//! `axis=0` matches embedding behavior exactly when `indices` is
//! rank-1; embedding is the axis-0 specialization preserved for
//! ergonomics + a slightly faster inner loop.

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Index-gather op handle. Carries the axis the gather operates over.
#[derive(Debug, Clone, Copy)]
pub struct IndexSelectOp {
    axis: usize,
}

impl IndexSelectOp {
    pub const fn new(axis: usize) -> Self {
        IndexSelectOp { axis }
    }
    pub const fn axis(self) -> usize {
        self.axis
    }
}

impl DeviceOp2 for IndexSelectOp {
    fn name(&self) -> &'static str {
        "index_select"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, input: &Tensor, indices: &Tensor) -> Result<Option<Tensor>> {
        validate(input, indices, self.axis)?;

        let dtype = input.dtype();
        let per = dtype.size_in_bytes();
        if per == 0 || dtype.is_packed() {
            bail!(
                "IndexSelectOp: packed dtype {dtype} for input is not supported"
            );
        }

        let shape = input.shape();
        let axis_dim = shape[self.axis];

        let outer: usize = shape[..self.axis].iter().product();
        let inner: usize = shape[self.axis + 1..].iter().product();

        let ids = read_indices(indices)?;
        let n_indices = ids.len();

        // Output shape = shape[..axis] ++ indices.shape ++ shape[axis+1..]
        let mut out_shape: Vec<usize> = shape[..self.axis].to_vec();
        out_shape.extend_from_slice(indices.shape());
        out_shape.extend_from_slice(&shape[self.axis + 1..]);

        let block_bytes = inner * per;
        let in_cpu = downcast_cpu(input, "input")?;
        let in_bytes = in_cpu.as_bytes();
        if in_bytes.len() < outer * axis_dim * block_bytes {
            bail!(
                "IndexSelectOp: input bytes {} < required {}",
                in_bytes.len(),
                outer * axis_dim * block_bytes
            );
        }
        let mut out_bytes = vec![0u8; outer * n_indices * block_bytes];

        // For each outer slot, copy n_indices blocks of `inner` elements
        // from the indexed positions on `axis`.
        for o in 0..outer {
            for (out_pos, &id) in ids.iter().enumerate() {
                if id as usize >= axis_dim {
                    bail!(
                        "IndexSelectOp: index {id} out of range (axis dim {axis_dim}) at position {out_pos}"
                    );
                }
                let src = (o * axis_dim + id as usize) * block_bytes;
                let dst = (o * n_indices + out_pos) * block_bytes;
                out_bytes[dst..dst + block_bytes].copy_from_slice(&in_bytes[src..src + block_bytes]);
            }
        }

        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let layout = Layout::contiguous(out_shape);
        Ok(Some(Tensor::from_parts(storage, layout, TensorId::next())?))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, input: &Tensor, indices: &Tensor) -> Result<Option<Tensor>> {
        // The CUDA substrate exposes two paths:
        //   - `cuda_index_select_dim0` — legacy fast path along axis 0
        //     (one block per output row, full row-byte copy).
        //   - `cuda_index_select_axis_n` — generic axis-N gather
        //     (2-D grid: (ids_dim, left_size); per-block row copy).
        //
        // Both require U32 indices and contiguous, non-packed inputs.
        // I64 indices and packed dtypes still fall through to CPU.
        if input.dtype().is_packed() {
            return Ok(None);
        }
        if !input.is_contiguous() || !indices.is_contiguous() {
            return Ok(None);
        }
        if indices.dtype() != DType::U32 {
            return Ok(None);
        }
        // Storage check — both src and indices must be CUDA-backed. If
        // either side is on a different backend the dispatcher would
        // normally route us elsewhere, but be defensive in case a
        // mixed-storage tensor pair sneaks through.
        if input
            .storage()
            .as_any()
            .downcast_ref::<crate::CudaStorage>()
            .is_none()
            || indices
                .storage()
                .as_any()
                .downcast_ref::<crate::CudaStorage>()
                .is_none()
        {
            return Ok(None);
        }
        if self.axis == 0 {
            // Preserve the dim0 fast path. Multi-D indices flatten →
            // gather → reshape so the legacy kernel can run its
            // single-axis blocks.
            if indices.rank() == 1 {
                Ok(Some(crate::cuda_index_select_dim0(input, indices)?))
            } else {
                let n_indices = indices.element_count();
                let flat_ids = indices.reshape(vec![n_indices])?;
                let gathered = crate::cuda_index_select_dim0(input, &flat_ids)?;
                // gathered shape: [n_indices, ...rest_of_input_dims_after_dim0]
                // Final shape: [..indices.shape, ...rest]
                let mut out_shape = indices.shape().to_vec();
                out_shape.extend_from_slice(&input.shape()[1..]);
                Ok(Some(gathered.reshape(out_shape)?))
            }
        } else {
            // Generic axis-N kernel handles arbitrary axis and the
            // multi-D indices case directly (output shape is
            // `src.shape[..axis] ++ indices.shape ++ src.shape[axis+1..]`).
            Ok(Some(crate::cuda_index_select_axis_n(
                input,
                self.axis,
                indices,
            )?))
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, input: &Tensor, indices: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd so future MSL
        // kernel work can drop in without changing the call site:
        //   - axis == 0 only (dim0 fast path; generic-axis lands later)
        //   - non-packed input dtype
        //   - contiguous input and indices
        //   - U32 indices (matches cuda_fwd; I64 needs a cast kernel)
        //   - F32/BF16/F16 input dtype (candle Metal index_select
        //     supports these)
        //   - storage must be Metal-backed on both sides
        //
        // Phase 4 substrate-op landing: dispatch through
        // `crate::metal_index_select_dim0` which wraps candle's
        // production Metal `call_index_select` kernel (the same path
        // `candle::Tensor::index_select(ids, 0)` takes). Apple Silicon
        // UMA: no host bounce — kt's MetalStorage::buffer is shared
        // with the candle wrapper via Arc<metal::Buffer>.
        if self.axis != 0 {
            return Ok(None);
        }
        if input.dtype().is_packed() {
            return Ok(None);
        }
        if !input.is_contiguous() || !indices.is_contiguous() {
            return Ok(None);
        }
        if indices.dtype() != DType::U32 {
            return Ok(None);
        }
        if !matches!(input.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if input
            .storage()
            .as_any()
            .downcast_ref::<crate::MetalStorage>()
            .is_none()
            || indices
                .storage()
                .as_any()
                .downcast_ref::<crate::MetalStorage>()
                .is_none()
        {
            return Ok(None);
        }
        Ok(Some(crate::metal_index_select_dim0(input, indices)?))
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, input: &Tensor, indices: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cuda_fwd / metal_fwd:
        //   - axis == 0 only
        //   - non-packed input dtype
        //   - contiguous input and indices
        //   - U32 indices (matches cuda_fwd; I64 needs a cast kernel)
        //   - F32 input dtype (the `vk_embedding_lookup_f32` shader is
        //     F32-only today; BF16 has `vk_embedding_lookup_bf16` but
        //     emits F32 output, which doesn't match our same-dtype
        //     index_select contract — wire that separately when needed)
        //   - storage must be Vulkan-backed on both sides
        //
        // Phase 4 substrate-op landing: dispatch through
        // `crate::vulkan_index_select_dim0` which wraps the production
        // `vk_embedding_lookup_f32` shader. Current bridge D2H+H2D
        // round-trips bytes through the host; see
        // `vulkan_storage.rs::vulkan_index_select_dim0` rustdoc for
        // the zero-copy follow-up plan.
        if self.axis != 0 {
            return Ok(None);
        }
        if input.dtype().is_packed() {
            return Ok(None);
        }
        if !input.is_contiguous() || !indices.is_contiguous() {
            return Ok(None);
        }
        if indices.dtype() != DType::U32 {
            return Ok(None);
        }
        if !matches!(input.dtype(), DType::F32) {
            return Ok(None);
        }
        if input
            .storage()
            .as_any()
            .downcast_ref::<crate::VulkanStorage>()
            .is_none()
            || indices
                .storage()
                .as_any()
                .downcast_ref::<crate::VulkanStorage>()
                .is_none()
        {
            return Ok(None);
        }
        Ok(Some(crate::vulkan_index_select_dim0(input, indices)?))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Dispatch `IndexSelectOp` on the given axis.
pub fn index_select(input: &Tensor, axis: usize, indices: &Tensor) -> Result<Tensor> {
    dispatch2(&IndexSelectOp::new(axis), input, indices)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(input: &Tensor, indices: &Tensor, axis: usize) -> Result<()> {
    if input.rank() == 0 {
        bail!("IndexSelectOp: input must have rank ≥ 1");
    }
    if axis >= input.rank() {
        bail!(
            "IndexSelectOp: axis {axis} out of bounds (input rank {})",
            input.rank()
        );
    }
    if indices.rank() == 0 {
        bail!("IndexSelectOp: indices must have rank ≥ 1");
    }
    if !matches!(indices.dtype(), DType::I64 | DType::U32) {
        bail!(
            "IndexSelectOp: indices dtype must be I64/U32, got {}",
            indices.dtype()
        );
    }
    if !input.is_contiguous() {
        bail!("IndexSelectOp: input must be contiguous");
    }
    if !indices.is_contiguous() {
        bail!("IndexSelectOp: indices must be contiguous");
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("IndexSelectOp: {label} storage must be CpuStorage")))
}

fn read_indices(t: &Tensor) -> Result<Vec<u64>> {
    let cpu = downcast_cpu(t, "indices")?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::I64 => {
            if bytes.len() < n * 8 {
                bail!(
                    "IndexSelectOp: indices buffer too small ({} < {})",
                    bytes.len(),
                    n * 8
                );
            }
            for i in 0..n {
                let v = i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
                if v < 0 {
                    bail!("IndexSelectOp: negative index {v} at position {i}");
                }
                out.push(v as u64);
            }
        }
        DType::U32 => {
            if bytes.len() < n * 4 {
                bail!(
                    "IndexSelectOp: indices buffer too small ({} < {})",
                    bytes.len(),
                    n * 4
                );
            }
            for i in 0..n {
                let v = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                out.push(v as u64);
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn axis_0_matches_embedding_semantics() {
        // [[10, 20], [30, 40], [50, 60]] index_select axis 0 indices=[2, 0]
        // -> [[50, 60], [10, 20]]
        let x = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]).unwrap();
        let ids = Tensor::from_slice(&[2i64, 0], vec![2]).unwrap();
        let out = index_select(&x, 0, &ids).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(read_f32(&out), vec![50.0, 60.0, 10.0, 20.0]);
    }

    #[test]
    fn axis_1_gathers_columns() {
        // [[1, 2, 3], [4, 5, 6]] index_select axis 1 indices=[2, 0]
        // -> [[3, 1], [6, 4]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let ids = Tensor::from_slice(&[2i64, 0], vec![2]).unwrap();
        let out = index_select(&x, 1, &ids).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(read_f32(&out), vec![3.0, 1.0, 6.0, 4.0]);
    }

    #[test]
    fn axis_1_3d_gathers_middle_axis() {
        // shape [2, 3, 2], index_select axis 1 indices=[2, 0]
        // -> shape [2, 2, 2]
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&v, vec![2, 3, 2]).unwrap();
        let ids = Tensor::from_slice(&[2i64, 0], vec![2]).unwrap();
        let out = index_select(&x, 1, &ids).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2]);
        // Batch 0: original [[0,1],[2,3],[4,5]] -> picked [[4,5],[0,1]]
        // Batch 1: original [[6,7],[8,9],[10,11]] -> picked [[10,11],[6,7]]
        assert_eq!(
            read_f32(&out),
            vec![4.0, 5.0, 0.0, 1.0, 10.0, 11.0, 6.0, 7.0]
        );
    }

    #[test]
    fn multi_dim_indices_broadcast_into_axis() {
        // [[1,2,3],[4,5,6]] index_select axis 1 indices=[[0,2],[1,1]]
        // -> shape [2, 2, 2]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let ids = Tensor::from_slice(&[0i64, 2, 1, 1], vec![2, 2]).unwrap();
        let out = index_select(&x, 1, &ids).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2]);
        // Row 0 (orig [1,2,3]) at indices [[0,2],[1,1]] -> [[1,3],[2,2]]
        // Row 1 (orig [4,5,6]) at indices [[0,2],[1,1]] -> [[4,6],[5,5]]
        assert_eq!(
            read_f32(&out),
            vec![1.0, 3.0, 2.0, 2.0, 4.0, 6.0, 5.0, 5.0]
        );
    }

    #[test]
    fn u32_indices_path() {
        let x = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let ids = Tensor::from_slice(&[2u32, 0, 1], vec![3]).unwrap();
        let out = index_select(&x, 0, &ids).unwrap();
        assert_eq!(read_f32(&out), vec![30.0, 10.0, 20.0]);
    }

    #[test]
    fn rejects_out_of_range_index() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let ids = Tensor::from_slice(&[5i64], vec![1]).unwrap();
        let e = index_select(&x, 0, &ids).unwrap_err();
        assert!(e.to_string().contains("out of range"));
    }

    #[test]
    fn rejects_axis_out_of_bounds() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let ids = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let e = index_select(&x, 5, &ids).unwrap_err();
        assert!(e.to_string().contains("axis 5 out of bounds"));
    }

    #[test]
    fn rejects_bad_index_dtype() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let ids = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let e = index_select(&x, 0, &ids).unwrap_err();
        assert!(e.to_string().contains("I64/U32"));
    }

    #[test]
    fn op_metadata() {
        let op = IndexSelectOp::new(1);
        assert_eq!(op.name(), "index_select");
        assert!(op.determinism().is_constructive());
        assert_eq!(op.axis(), 1);
    }

    /// CUDA parity for `index_select` across all axes of a rank-4 tensor.
    /// Compares the CUDA path (which now routes through the axis-N
    /// kernel for `axis != 0`) against the CPU reference for bit-exact
    /// equality on F32. (#1082 axis-N gather extension)
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_index_select_axis_parity_f32() {
        // Skip if no CUDA device available at runtime.
        if crate::primary_cuda_device(0).is_err() {
            return;
        }

        // rank-4 source `[A, B, C, D]` with distinct values so axis
        // confusion shows up as a mismatch instead of a coincidence.
        let a = 3usize;
        let b = 4usize;
        let c = 5usize;
        let d = 2usize;
        let n = a * b * c * d;
        let values: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 7.0).collect();
        let cpu_x = Tensor::from_slice(&values, vec![a, b, c, d]).unwrap();
        let cuda_x = cpu_x
            .to_device(crate::Device::Cuda(0))
            .unwrap();

        let dims = [a, b, c, d];
        // For each axis pick a few interesting indices (out-of-order +
        // duplicates) of length K.
        let test_cases: [(usize, Vec<u32>); 4] = [
            (0, vec![2, 0, 1, 0]),
            (1, vec![3, 0, 2, 1, 2]),
            (2, vec![4, 1, 0, 3, 2, 4]),
            (3, vec![1, 0, 1, 0, 1]),
        ];

        for (axis, idx_vec) in test_cases.iter() {
            let cpu_ids = Tensor::from_slice(idx_vec.as_slice(), vec![idx_vec.len()]).unwrap();
            let cuda_ids = cpu_ids
                .to_device(crate::Device::Cuda(0))
                .unwrap();

            let cpu_out = index_select(&cpu_x, *axis, &cpu_ids).unwrap();
            let cuda_out = index_select(&cuda_x, *axis, &cuda_ids).unwrap();

            // Shape parity
            let expected_shape: Vec<usize> = dims[..*axis]
                .iter()
                .copied()
                .chain(std::iter::once(idx_vec.len()))
                .chain(dims[*axis + 1..].iter().copied())
                .collect();
            assert_eq!(
                cpu_out.shape(),
                expected_shape.as_slice(),
                "cpu shape mismatch for axis {axis}"
            );
            assert_eq!(
                cuda_out.shape(),
                expected_shape.as_slice(),
                "cuda shape mismatch for axis {axis}"
            );

            // Value parity: D2H-copy the CUDA output and compare bytes.
            let cuda_out_cpu = cuda_out
                .to_device(crate::Device::Cpu)
                .unwrap();
            let cpu_vec = read_f32(&cpu_out);
            let cuda_vec = read_f32(&cuda_out_cpu);
            assert_eq!(
                cpu_vec.len(),
                cuda_vec.len(),
                "len mismatch for axis {axis}"
            );
            for (i, (c, g)) in cpu_vec.iter().zip(cuda_vec.iter()).enumerate() {
                assert_eq!(
                    c.to_bits(),
                    g.to_bits(),
                    "axis {axis} element {i}: cpu={c} cuda={g}"
                );
            }
        }
    }

    /// CUDA parity for axis-N gather with multi-D indices: the kernel
    /// path should treat indices.shape as a flattened K and produce
    /// `src.shape[..axis] ++ indices.shape ++ src.shape[axis+1..]`.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_index_select_axis_n_multi_d_indices() {
        // Skip if no CUDA device available at runtime.
        if crate::primary_cuda_device(0).is_err() {
            return;
        }

        // `[2, 3]` source, gather axis 1 with `[[0,2],[1,1]]` indices.
        // Expected output shape `[2, 2, 2]` (matches the CPU test
        // `multi_dim_indices_broadcast_into_axis`).
        let cpu_x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cuda_x = cpu_x
            .to_device(crate::Device::Cuda(0))
            .unwrap();
        let cpu_ids = Tensor::from_slice(&[0u32, 2, 1, 1], vec![2, 2]).unwrap();
        let cuda_ids = cpu_ids
            .to_device(crate::Device::Cuda(0))
            .unwrap();

        let cuda_out = index_select(&cuda_x, 1, &cuda_ids).unwrap();
        let cuda_out_cpu = cuda_out
            .to_device(crate::Device::Cpu)
            .unwrap();
        assert_eq!(cuda_out.shape(), &[2, 2, 2]);
        // Row 0 (orig [1,2,3]) at indices [[0,2],[1,1]] -> [[1,3],[2,2]]
        // Row 1 (orig [4,5,6]) at indices [[0,2],[1,1]] -> [[4,6],[5,5]]
        assert_eq!(
            read_f32(&cuda_out_cpu),
            vec![1.0, 3.0, 2.0, 2.0, 4.0, 6.0, 5.0, 5.0]
        );
    }

    /// CUDA parity for axis-N gather on BF16. Compares against the CPU
    /// reference via bytewise equality.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_index_select_axis_parity_bf16() {
        use half::bf16;
        // Skip if no CUDA device available at runtime.
        if crate::primary_cuda_device(0).is_err() {
            return;
        }

        // rank-3 BF16 source `[B, H, K]` with distinct values.
        let bdim = 2usize;
        let h = 4usize;
        let k = 6usize;
        let n = bdim * h * k;
        let values: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32 * 0.25 - 1.0)).collect();
        let cpu_x = Tensor::from_slice(&values, vec![bdim, h, k]).unwrap();
        let cuda_x = cpu_x
            .to_device(crate::Device::Cuda(0))
            .unwrap();

        // Gather along axis 1 (heads).
        let ids_vec = vec![3u32, 0, 2, 1];
        let cpu_ids = Tensor::from_slice(ids_vec.as_slice(), vec![ids_vec.len()]).unwrap();
        let cuda_ids = cpu_ids
            .to_device(crate::Device::Cuda(0))
            .unwrap();

        let cpu_out = index_select(&cpu_x, 1, &cpu_ids).unwrap();
        let cuda_out = index_select(&cuda_x, 1, &cuda_ids).unwrap();

        assert_eq!(cpu_out.shape(), &[bdim, ids_vec.len(), k]);
        assert_eq!(cuda_out.shape(), &[bdim, ids_vec.len(), k]);

        let cuda_out_cpu = cuda_out
            .to_device(crate::Device::Cpu)
            .unwrap();
        let cpu_bytes = cpu_out
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap()
            .as_bytes()
            .to_vec();
        let cuda_bytes = cuda_out_cpu
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap()
            .as_bytes()
            .to_vec();
        assert_eq!(cpu_bytes, cuda_bytes, "bf16 axis-1 bytes differ");
    }
}
