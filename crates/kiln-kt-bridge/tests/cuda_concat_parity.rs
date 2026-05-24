//! Parity test: kt CUDA concat (`ConcatOp::cuda_fwd` / `cuda_concat`
//! / the `ops::concat` dispatch on all-CUDA inputs) vs kt CPU
//! reference (`ops::concat` on CPU tensors).
//!
//! Phase 4 substrate validation for the per-axis concat kernel in
//! `crates/kiln-tensor/csrc/concat.cu`. The CPU reference performs a
//! byte-wise per-outer-slab copy; the CUDA kernel does the same on
//! device, so parity should be exact for all dtypes.



use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_concat, ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

/// Build a kt-CPU reference: run `ops::concat` on CPU tensors with
/// the same per-input shapes + same F32 data, return the flat F32
/// output vector.
fn cpu_reference_f32(inputs: &[(Vec<usize>, Vec<f32>)], axis: usize) -> Vec<f32> {
    let kt_tensors: Vec<Tensor> = inputs
        .iter()
        .map(|(shape, data)| Tensor::from_slice(data, shape.clone()).unwrap())
        .collect();
    let refs: Vec<&Tensor> = kt_tensors.iter().collect();
    let y = ops::concat(&refs, axis).unwrap();
    let cpu_storage = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let n: usize = y.shape().iter().product();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

#[test]
fn cuda_concat_two_rank2_axis_0_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Axis-0 concat of [[1,2,3],[4,5,6]] and [[7,8,9]] → [[1..9]].
    let a_data: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (7..=9).map(|i| i as f32).collect();
    let a_cd = CandleTensor::from_vec(a_data.clone(), (2, 3), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (1, 3), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = cuda_concat(&[&a_kt, &b_kt], 0).expect("cuda_concat");
    assert_eq!(out_kt.shape(), &[3, 3]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((9,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = cpu_reference_f32(
        &[
            (vec![2, 3], a_data.clone()),
            (vec![1, 3], b_data.clone()),
        ],
        0,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_concat_two_rank2_axis_1_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Axis-1 concat of [[1,2],[3,4]] and [[5],[6]] → [[1,2,5],[3,4,6]].
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0];
    let a_cd = CandleTensor::from_vec(a_data.clone(), (2, 2), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (2, 1), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = cuda_concat(&[&a_kt, &b_kt], 1).expect("cuda_concat");
    assert_eq!(out_kt.shape(), &[2, 3]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((6,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = cpu_reference_f32(
        &[
            (vec![2, 2], a_data.clone()),
            (vec![2, 1], b_data.clone()),
        ],
        1,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_concat_three_rank3_axis_middle_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Three rank-3 tensors concatenated on axis 1.
    // a: [B=2, 1, 4], b: [B=2, 2, 4], c: [B=2, 1, 4] → out: [B=2, 4, 4].
    let a_data = pattern_f32(2 * 1 * 4, 11);
    let b_data = pattern_f32(2 * 2 * 4, 13);
    let c_data = pattern_f32(2 * 1 * 4, 17);
    let a_cd = CandleTensor::from_vec(a_data.clone(), (2, 1, 4), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (2, 2, 4), &dev).unwrap();
    let c_cd = CandleTensor::from_vec(c_data.clone(), (2, 1, 4), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();
    let c_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&c_cd).unwrap();

    let out_kt = cuda_concat(&[&a_kt, &b_kt, &c_kt], 1).expect("cuda_concat 3-way");
    assert_eq!(out_kt.shape(), &[2, 4, 4]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n: usize = out_kt.shape().iter().product();
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = cpu_reference_f32(
        &[
            (vec![2, 1, 4], a_data),
            (vec![2, 2, 4], b_data),
            (vec![2, 1, 4], c_data),
        ],
        1,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_concat_bf16_qkv_pattern() {
    // QKV-style fused concat: hidden 2560 + 256 + 256 = 3072.
    // Realistic shape that exercises both the 1D axis and the BF16
    // path. Tight equality expected since the kernel is byte-wise.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let batch = 4usize;
    let q_dim = 32usize;
    let k_dim = 8usize;
    let v_dim = 8usize;
    let q_data = pattern_f32(batch * q_dim, 71);
    let k_data = pattern_f32(batch * k_dim, 73);
    let v_data = pattern_f32(batch * v_dim, 79);
    let q_cd = CandleTensor::from_vec(q_data.clone(), (batch, q_dim), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let k_cd = CandleTensor::from_vec(k_data.clone(), (batch, k_dim), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let v_cd = CandleTensor::from_vec(v_data.clone(), (batch, v_dim), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let q_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&q_cd).unwrap();
    let k_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&k_cd).unwrap();
    let v_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&v_cd).unwrap();

    let out_kt = cuda_concat(&[&q_kt, &k_kt, &v_kt], 1).expect("cuda_concat bf16");
    assert_eq!(out_kt.shape(), &[batch, q_dim + k_dim + v_dim]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let n: usize = out_kt.shape().iter().product();
    let got_bf16_back: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // For the reference, mimic what the CUDA path stores byte-for-byte:
    // the BF16 quantization happened at upload time (`to_dtype`), so
    // the kt-CPU reference must concat BF16 bytes too. Build the
    // reference in BF16, then read back via F32.
    let bf16_to_f32 = |v: &f32| -> f32 { half::bf16::from_f32(*v).to_f32() };
    let q_bf16: Vec<f32> = q_data.iter().map(bf16_to_f32).collect();
    let k_bf16: Vec<f32> = k_data.iter().map(bf16_to_f32).collect();
    let v_bf16: Vec<f32> = v_data.iter().map(bf16_to_f32).collect();

    // Per-row interleave: row r has q_dim Qs, then k_dim Ks, then v_dim Vs.
    let total_per_row = q_dim + k_dim + v_dim;
    let mut want = vec![0f32; batch * total_per_row];
    for r in 0..batch {
        for i in 0..q_dim {
            want[r * total_per_row + i] = q_bf16[r * q_dim + i];
        }
        for i in 0..k_dim {
            want[r * total_per_row + q_dim + i] = k_bf16[r * k_dim + i];
        }
        for i in 0..v_dim {
            want[r * total_per_row + q_dim + k_dim + i] = v_bf16[r * v_dim + i];
        }
    }

    assert_eq!(want, got_bf16_back);
}

#[test]
fn cuda_concat_dispatches_through_ops_concat() {
    // End-to-end dispatch: ops::concat on all-CUDA inputs should pick
    // the CUDA fast path and produce identical results to the byte-
    // wise CPU reference.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a_data: Vec<f32> = (0..6).map(|i| (i + 100) as f32).collect();
    let b_data: Vec<f32> = (0..3).map(|i| (i + 200) as f32).collect();
    let a_cd = CandleTensor::from_vec(a_data.clone(), (2, 3), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (1, 3), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let out_kt = ops::concat(&[&a_kt, &b_kt], 0).expect("dispatch");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((9,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = cpu_reference_f32(
        &[
            (vec![2, 3], a_data.clone()),
            (vec![1, 3], b_data.clone()),
        ],
        0,
    );
    assert_eq!(want, got);
}

#[test]
fn cuda_concat_op_handle_cuda_fwd_explicit() {
    // Same as above but invoked through the explicit
    // `ConcatOp::cuda_fwd` surface to lock down that handle path.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let a_data: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (4..8).map(|i| i as f32).collect();
    let a_cd = CandleTensor::from_vec(a_data.clone(), (4,), &dev).unwrap();
    let b_cd = CandleTensor::from_vec(b_data.clone(), (4,), &dev).unwrap();
    let a_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&a_cd).unwrap();
    let b_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&b_cd).unwrap();

    let op = ops::ConcatOp::new(0);
    let out_kt = op
        .cuda_fwd(&[&a_kt, &b_kt])
        .expect("cuda_fwd")
        .expect("Some(out)");
    assert_eq!(out_kt.shape(), &[8]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want: Vec<f32> = (0..8).map(|i| i as f32).collect();
    assert_eq!(want, got);
}
