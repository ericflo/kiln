//! Parity test: kt CUDA `cuda_cumsum_axis` vs kt CPU `ops::cumsum`.
//!
//! Phase 6 substrate validation (#1082). Confirms the scan kernel in
//! `csrc/scan_axis.cu` (kind=0) produces inclusive prefix sums along
//! the trailing axis matching the canonical CPU reference (kt's own
//! triple-loop `cumsum`).

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_cumsum_axis, ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        // Small-magnitude values keep BF16 accumulation reasonable.
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 4096.0;
        out.push(f);
    }
    out
}

fn cpu_cumsum_f32(data: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    // Use kt's own CPU `cumsum` as the canonical reference.
    let x = Tensor::from_slice(data, vec![n_rows, n_cols]).unwrap();
    let y = ops::cumsum(&x, 1).unwrap();
    let cpu_storage = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for i in 0..(n_rows * n_cols) {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

fn run_cumsum_parity(n_rows: usize, n_cols: usize, dtype: CandleDType, tolerance: f32) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = n_rows * n_cols;
    let data = pattern(n, 13);

    let x_cd = CandleTensor::from_vec(data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    // Last axis = 1 for a [n_rows, n_cols] tensor.
    let out_kt = cuda_cumsum_axis(&x_kt, 1).expect("cuda_cumsum_axis");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let ref_vec = cpu_cumsum_f32(&data, n_rows, n_cols);

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(ref_vec.len(), got_vec.len());
    let mut max_abs = 0.0f32;
    for (i, (a, b)) in ref_vec.iter().zip(got_vec.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > tolerance * 10.0 {
            panic!(
                "rows={n_rows} cols={n_cols} dtype={dtype:?} index={i} ref={a} got={b} diff={d}"
            );
        }
    }
    assert!(
        max_abs < tolerance,
        "rows={n_rows} cols={n_cols} dtype={dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

#[test]
fn cuda_cumsum_f32_1_row_32_cols() {
    // Small row that fits in a single warp's worth of threads.
    run_cumsum_parity(1, 32, CandleDType::F32, 1e-4);
}

#[test]
fn cuda_cumsum_f32_4_rows_128_cols() {
    run_cumsum_parity(4, 128, CandleDType::F32, 1e-4);
}

#[test]
fn cuda_cumsum_f32_8_rows_512_cols() {
    run_cumsum_parity(8, 512, CandleDType::F32, 1e-3);
}

#[test]
fn cuda_cumsum_f32_2_rows_2048_cols() {
    // Larger row size — exercises the per-thread contiguous-chunk
    // strided path when n_cols > MAX_THREADS.
    run_cumsum_parity(2, 2048, CandleDType::F32, 1e-2);
}

#[test]
fn cuda_cumsum_bf16_4_rows_256_cols() {
    run_cumsum_parity(4, 256, CandleDType::BF16, 0.5);
}

#[test]
fn cuda_cumsum_bf16_2_rows_1024_cols() {
    run_cumsum_parity(2, 1024, CandleDType::BF16, 1.0);
}

#[test]
fn cuda_cumsum_f16_4_rows_256_cols() {
    run_cumsum_parity(4, 256, CandleDType::F16, 0.5);
}

#[test]
fn cuda_cumsum_rejects_non_last_axis() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let x_cd = CandleTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    // axis=0 is not the last axis for a rank-2 tensor; expect an error.
    let res = cuda_cumsum_axis(&x_kt, 0);
    assert!(res.is_err(), "expected non-last-axis to be rejected");
}

#[test]
fn cuda_cumsum_known_values() {
    // Small, hand-checkable input.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
    let x_cd = CandleTensor::from_vec(data, (2, 4), &dev).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let out_kt = cuda_cumsum_axis(&x_kt, 1).unwrap();
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();
    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((8,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Row 0: [1,3,6,10]. Row 1: [1,2,3,4].
    let expected = [1.0f32, 3.0, 6.0, 10.0, 1.0, 2.0, 3.0, 4.0];
    for (a, b) in got.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "got={a} expected={b}");
    }
}
