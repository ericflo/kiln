//! `NarrowBackward` — gradient of `narrow(x, axis, offset, length)`.
//!
//! Forward is a zero-copy view that slices `x` along `axis` to
//! `[offset .. offset+length]`. Backward embeds the upstream gradient
//! into a zero-filled tensor with the original `x.shape`, writing it
//! at the same `[offset .. offset+length]` slice.
//!
//! ```text
//! d_x[..., offset+i, ...] = grad_output[..., i, ...]   for i in 0..length
//! d_x[..., j, ...]        = 0                          otherwise
//! ```
//!
//! This is the "zero-pad" gradient pattern — sometimes called the
//! `pad_with_zero_at_offset` adjoint.

use std::sync::Arc;

use kiln_tensor::{bail, CpuStorage, Error, Layout, Result, Storage, Tensor, TensorId};

use crate::BackwardOp;

#[derive(Debug)]
pub struct NarrowBackward {
    /// `axis` from the forward call.
    pub axis: usize,
    /// `offset` from the forward call.
    pub offset: usize,
    /// `length` from the forward call (== grad axis size).
    pub length: usize,
    /// Saved `x.shape` from the forward (target shape for `d_x`).
    pub source_shape: Vec<usize>,
}

impl BackwardOp for NarrowBackward {
    fn name(&self) -> &'static str {
        "narrow_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let rank = self.source_shape.len();
        if self.axis >= rank {
            bail!(
                "NarrowBackward: axis {} out of bounds for rank {}",
                self.axis,
                rank
            );
        }
        if grad_output.rank() != rank {
            bail!(
                "NarrowBackward: grad rank {} != source rank {}",
                grad_output.rank(),
                rank
            );
        }
        let g_shape = grad_output.shape();
        for d in 0..rank {
            if d == self.axis {
                if g_shape[d] != self.length {
                    bail!(
                        "NarrowBackward: grad shape[{d}] = {} != length {}",
                        g_shape[d],
                        self.length
                    );
                }
            } else if g_shape[d] != self.source_shape[d] {
                bail!(
                    "NarrowBackward: grad shape[{d}] = {} != source shape[{d}] = {}",
                    g_shape[d],
                    self.source_shape[d]
                );
            }
        }
        let source_axis_len = self.source_shape[self.axis];
        if self.offset + self.length > source_axis_len {
            bail!(
                "NarrowBackward: offset {} + length {} > source axis length {}",
                self.offset,
                self.length,
                source_axis_len
            );
        }
        let dtype = grad_output.dtype();
        if dtype.is_packed() {
            bail!("NarrowBackward: packed dtype {dtype} not supported");
        }
        if !grad_output.is_contiguous() {
            bail!("NarrowBackward: grad must be contiguous");
        }
        let go_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("NarrowBackward: grad storage must be CpuStorage"))?;
        let go_bytes = go_cpu.as_bytes();
        let per = dtype.size_in_bytes();

        // Compute element strides for both shapes (row-major).
        let mut g_strides = vec![1usize; rank];
        let mut x_strides = vec![1usize; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            g_strides[d] = g_strides[d + 1] * g_shape[d + 1];
            x_strides[d] = x_strides[d + 1] * self.source_shape[d + 1];
        }

        let source_total: usize = self.source_shape.iter().product();
        let mut out_bytes = vec![0u8; source_total * per];

        // Iterate the grad shape and copy into the corresponding dest
        // offset (axis index shifted by self.offset).
        let n_grad: usize = g_shape.iter().product();
        let mut coord = vec![0usize; rank];
        for src_idx in 0..n_grad {
            let mut rem = src_idx;
            for d in 0..rank {
                coord[d] = rem / g_strides[d];
                rem %= g_strides[d];
            }
            let mut dst_off = 0usize;
            for d in 0..rank {
                let c = if d == self.axis {
                    coord[d] + self.offset
                } else {
                    coord[d]
                };
                dst_off += c * x_strides[d];
            }
            let src_byte = src_idx * per;
            let dst_byte = dst_off * per;
            out_bytes[dst_byte..dst_byte + per]
                .copy_from_slice(&go_bytes[src_byte..src_byte + per]);
        }

        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let d_x = Tensor::from_parts(
            storage,
            Layout::contiguous(self.source_shape.clone()),
            TensorId::next(),
        )?;
        Ok(vec![Some(d_x)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // The source is not needed at backward time — only its shape
        // (saved on the struct).
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::DType;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn narrow_backward_1d_zero_pads() {
        // Forward: x.shape=[5], narrow(0, 1, 2) -> [2].
        // Grad in: [10, 20] -> d_x = [0, 10, 20, 0, 0].
        let grad = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let bo = NarrowBackward {
            axis: 0,
            offset: 1,
            length: 2,
            source_shape: vec![5],
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[5]);
        assert_eq!(read_f32(&d), vec![0.0, 10.0, 20.0, 0.0, 0.0]);
    }

    #[test]
    fn narrow_backward_2d_axis_0() {
        // Forward: x.shape=[4, 3], narrow(0, 1, 2) -> [2, 3].
        // Grad [[1,2,3],[4,5,6]] -> d_x rows 1 and 2 are set.
        let grad = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let bo = NarrowBackward {
            axis: 0,
            offset: 1,
            length: 2,
            source_shape: vec![4, 3],
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[4, 3]);
        assert_eq!(
            read_f32(&d),
            vec![
                0.0, 0.0, 0.0, // row 0
                1.0, 2.0, 3.0, // row 1
                4.0, 5.0, 6.0, // row 2
                0.0, 0.0, 0.0, // row 3
            ]
        );
    }

    #[test]
    fn narrow_backward_2d_axis_1_chunk() {
        // FLCE chunk pattern: x.shape=[B=2, V=6], narrow(1, 2, 3) -> [2, 3].
        // Grad [[1,2,3],[4,5,6]] -> middle cols of d_x.
        let grad = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let bo = NarrowBackward {
            axis: 1,
            offset: 2,
            length: 3,
            source_shape: vec![2, 6],
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[2, 6]);
        assert_eq!(
            read_f32(&d),
            vec![
                0.0, 0.0, 1.0, 2.0, 3.0, 0.0, // row 0
                0.0, 0.0, 4.0, 5.0, 6.0, 0.0, // row 1
            ]
        );
    }

    #[test]
    fn narrow_backward_full_span_identity() {
        // narrow(0, 0, N) == identity. Backward should be identity.
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = NarrowBackward {
            axis: 0,
            offset: 0,
            length: 3,
            source_shape: vec![3],
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&d), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn narrow_backward_oob_offset_errors() {
        let grad = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bo = NarrowBackward {
            axis: 0,
            offset: 3,
            length: 2,
            source_shape: vec![4],
        };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("> source axis length"));
    }

    #[test]
    fn narrow_backward_grad_shape_mismatch_errors() {
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = NarrowBackward {
            axis: 0,
            offset: 0,
            length: 2,
            source_shape: vec![4],
        };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("length 2"));
    }

    #[test]
    fn narrow_backward_bf16() {
        let bf: Vec<half::bf16> = [10.0f32, 20.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let grad = Tensor::from_slice(&bf, vec![2]).unwrap();
        let bo = NarrowBackward {
            axis: 0,
            offset: 1,
            length: 2,
            source_shape: vec![4],
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.dtype(), DType::BF16);
        let bytes = d
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap()
            .as_bytes();
        let vals: Vec<f32> = (0..4)
            .map(|i| {
                half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            })
            .collect();
        assert_eq!(vals, vec![0.0, 10.0, 20.0, 0.0]);
    }

    #[test]
    fn op_metadata() {
        let bo = NarrowBackward {
            axis: 0,
            offset: 0,
            length: 1,
            source_shape: vec![1],
        };
        assert_eq!(bo.name(), "narrow_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(!bo.requires_input(0));
    }
}
