//! `interpolate_1d` — linear resampling along the last axis.
//!
//! Given a rank-N tensor of shape `[..., L_in]`, returns shape
//! `[..., L_out]` where each output position is a linear blend of
//! the two nearest input positions.
//!
//! Two modes for the alignment of endpoints:
//! - [`AlignCorners::Yes`]: `out[0] == in[0]` and `out[L_out-1] ==
//!   in[L_in-1]`. Cleanest for "scale these positions to that
//!   length" use cases.
//! - [`AlignCorners::No`]: pixel-center mapping (PyTorch's default
//!   `align_corners=False`). Slightly different boundary handling.
//!
//! PyTorch parity with `torch.nn.functional.interpolate(x,
//! size=L_out, mode='linear', align_corners=…)`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlignCorners {
    Yes,
    No,
}

pub fn interpolate_1d(x: &Tensor, target_len: usize, align: AlignCorners) -> Result<Tensor> {
    if x.rank() == 0 {
        bail!("interpolate_1d: input must have rank >= 1");
    }
    if target_len == 0 {
        bail!("interpolate_1d: target_len must be > 0");
    }
    let dtype = x.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!("interpolate_1d: dtype must be F32/BF16/F16, got {dtype}");
    }
    if !x.is_contiguous() {
        bail!("interpolate_1d: input must be contiguous");
    }
    let mut out_shape: Vec<usize> = x.shape().to_vec();
    let last = *out_shape.last().unwrap();
    if last == 0 {
        bail!("interpolate_1d: last axis must be > 0");
    }
    *out_shape.last_mut().unwrap() = target_len;
    let outer: usize = x.shape()[..x.rank() - 1].iter().product::<usize>().max(1);

    // Read all input as f32 for the math; pack back to input dtype.
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("interpolate_1d: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n_in = outer * last;
    let mut input = vec![0.0f32; n_in];
    match dtype {
        DType::F32 => {
            for i in 0..n_in {
                input[i] = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        DType::BF16 => {
            for i in 0..n_in {
                input[i] = half::bf16::from_le_bytes(
                    bytes[i * 2..i * 2 + 2].try_into().unwrap(),
                )
                .to_f32();
            }
        }
        DType::F16 => {
            for i in 0..n_in {
                input[i] = half::f16::from_le_bytes(
                    bytes[i * 2..i * 2 + 2].try_into().unwrap(),
                )
                .to_f32();
            }
        }
        _ => unreachable!(),
    }

    let mut output = vec![0.0f32; outer * target_len];

    for o in 0..outer {
        let row_in = &input[o * last..(o + 1) * last];
        let row_out = &mut output[o * target_len..(o + 1) * target_len];
        if target_len == 1 {
            // Edge case: target length 1 — copy the first element
            // (PyTorch's behavior for size=1 linear interp).
            row_out[0] = row_in[0];
            continue;
        }
        for j in 0..target_len {
            // Source position in floating-point input-coordinate space.
            let src_pos = match align {
                AlignCorners::Yes => {
                    if last == 1 {
                        0.0
                    } else {
                        (j as f32) * (last as f32 - 1.0) / (target_len as f32 - 1.0)
                    }
                }
                AlignCorners::No => {
                    // Pixel-center mapping: src = (j + 0.5) * (L_in/L_out) - 0.5.
                    let s = (j as f32 + 0.5) * (last as f32 / target_len as f32) - 0.5;
                    s.clamp(0.0, last as f32 - 1.0)
                }
            };
            let lo = src_pos.floor() as usize;
            let hi = (lo + 1).min(last - 1);
            let frac = src_pos - lo as f32;
            row_out[j] = row_in[lo] * (1.0 - frac) + row_in[hi] * frac;
        }
    }

    let per = dtype.size_in_bytes();
    let mut out_bytes = vec![0u8; output.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in output.iter().enumerate() {
                out_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in output.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in output.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
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

    #[test]
    fn upsample_align_corners_endpoints_preserved() {
        // [0, 1, 2] → length 5; endpoints stay 0 and 2.
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0], vec![3]).unwrap();
        let y = interpolate_1d(&x, 5, AlignCorners::Yes).unwrap();
        let v = read_f32(&y);
        assert!((v[0] - 0.0).abs() < 1e-5);
        assert!((v[4] - 2.0).abs() < 1e-5);
        // Midpoint is 1.0.
        assert!((v[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn downsample_align_corners() {
        // [0, 1, 2, 3, 4] → length 3 → [0, 2, 4].
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], vec![5]).unwrap();
        let y = interpolate_1d(&x, 3, AlignCorners::Yes).unwrap();
        let v = read_f32(&y);
        assert!((v[0] - 0.0).abs() < 1e-5);
        assert!((v[1] - 2.0).abs() < 1e-5);
        assert!((v[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn same_length_is_identity_with_align_corners() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = interpolate_1d(&x, 4, AlignCorners::Yes).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn target_len_one_returns_first_element() {
        let x = Tensor::from_slice(&[5.0f32, 6.0, 7.0], vec![3]).unwrap();
        let y = interpolate_1d(&x, 1, AlignCorners::Yes).unwrap();
        assert_eq!(read_f32(&y), vec![5.0]);
    }

    #[test]
    fn preserves_leading_axes() {
        // shape [2, 3] → [2, 6]; each row interpolated independently.
        let x = Tensor::from_slice(
            &[0.0f32, 1.0, 2.0, 10.0, 11.0, 12.0],
            vec![2, 3],
        )
        .unwrap();
        let y = interpolate_1d(&x, 6, AlignCorners::Yes).unwrap();
        assert_eq!(y.shape(), &[2, 6]);
        let v = read_f32(&y);
        // Row 0 endpoints.
        assert!((v[0] - 0.0).abs() < 1e-5);
        assert!((v[5] - 2.0).abs() < 1e-5);
        // Row 1 endpoints.
        assert!((v[6] - 10.0).abs() < 1e-5);
        assert!((v[11] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn align_corners_no_yields_pixel_center_blend() {
        // [0, 1, 2, 3] → length 2 with align_corners=No.
        // src for j=0: (0+0.5)*(4/2) - 0.5 = 0.5; j=1: (1+0.5)*(4/2) - 0.5 = 2.5
        // y[0] = 0*0.5 + 1*0.5 = 0.5; y[1] = 2*0.5 + 3*0.5 = 2.5.
        let x = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], vec![4]).unwrap();
        let y = interpolate_1d(&x, 2, AlignCorners::No).unwrap();
        let v = read_f32(&y);
        assert!((v[0] - 0.5).abs() < 1e-4);
        assert!((v[1] - 2.5).abs() < 1e-4);
    }

    #[test]
    fn zero_target_len_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = interpolate_1d(&x, 0, AlignCorners::Yes).unwrap_err();
        assert!(e.to_string().contains("target_len"));
    }

    #[test]
    fn rank_zero_errors() {
        let x = Tensor::from_slice::<f32>(&[1.0], vec![]).unwrap();
        let e = interpolate_1d(&x, 5, AlignCorners::Yes).unwrap_err();
        assert!(e.to_string().contains("rank"));
    }

    #[test]
    fn bf16_round_trip() {
        let x = Tensor::from_slice(
            &[
                half::bf16::from_f32(0.0),
                half::bf16::from_f32(2.0),
                half::bf16::from_f32(4.0),
            ],
            vec![3],
        )
        .unwrap();
        let y = interpolate_1d(&x, 5, AlignCorners::Yes).unwrap();
        let cpu = y.storage();
        let cpu = cpu.as_any().downcast_ref::<CpuStorage>().unwrap();
        let v: Vec<f32> = cpu
            .as_bytes()
            .chunks(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        assert!((v[0] - 0.0).abs() < 1e-2);
        assert!((v[2] - 2.0).abs() < 1e-2);
        assert!((v[4] - 4.0).abs() < 1e-2);
    }
}
