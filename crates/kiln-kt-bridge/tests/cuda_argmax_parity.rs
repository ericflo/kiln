//! Parity test: kt CUDA argmax_last_axis vs kt CPU argmax_last_dim.
//!
//! Phase 4 substrate validation. Confirms the kernel in
//! `csrc/argmax_last_axis.cu` produces per-row argmax indices
//! matching the canonical CPU reference (kt's own naive triple-loop
//! at `crates/kiln-tensor/src/ops/argmax.rs`).

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

fn cpu_reference_i64(data: &[f32], n_rows: usize, n_cols: usize) -> Vec<i64> {
    // kt CPU argmax via ops::argmax_last_dim.
    let x = Tensor::from_slice(data, vec![n_rows, n_cols]).unwrap();
    let y = ops::argmax_last_dim(&x).unwrap();
    let cpu_storage = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu_storage.as_bytes();
    let mut out = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        out.push(i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()));
    }
    out
}

fn run_argmax_parity(n_rows: usize, n_cols: usize, dtype: CandleDType) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n = n_rows * n_cols;
    let data = pattern(n, 17);

    let x_cd = CandleTensor::from_vec(data.clone(), (n_rows, n_cols), &dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::argmax_last_dim(&x_kt).expect("argmax");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(out_kt.dtype(), kiln_tensor::DType::I64);
    assert_eq!(out_kt.shape(), &[n_rows]);

    // Pull GPU result back to host as i64.
    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt).unwrap();
    let got_vec: Vec<i64> = got_cd
        .reshape((n_rows,))
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();

    // Reference: kt CPU argmax over the original F32 data. For BF16/F16
    // we recompute the reference at the *cast* dtype to match the GPU
    // input precisely (no precision skew on tie-cases).
    let ref_data: Vec<f32> = match dtype {
        CandleDType::F32 => data.clone(),
        CandleDType::BF16 => data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect(),
        CandleDType::F16 => data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect(),
        _ => panic!("unsupported dtype for test"),
    };
    let ref_vec = cpu_reference_i64(&ref_data, n_rows, n_cols);

    assert_eq!(ref_vec.len(), got_vec.len());
    for (row, (a, b)) in ref_vec.iter().zip(got_vec.iter()).enumerate() {
        assert_eq!(
            a, b,
            "rows={n_rows} cols={n_cols} dtype={dtype:?} row={row} cpu={a} cuda={b}"
        );
    }
}

#[test]
fn cuda_argmax_f32_4_rows_512_cols() {
    run_argmax_parity(4, 512, CandleDType::F32);
}

#[test]
fn cuda_argmax_bf16_8_rows_128_cols() {
    run_argmax_parity(8, 128, CandleDType::BF16);
}

#[test]
fn cuda_argmax_bf16_2_rows_2048_cols() {
    // Larger row size — exercises the strided per-thread reduction
    // when n_cols > MAX_THREADS.
    run_argmax_parity(2, 2048, CandleDType::BF16);
}

#[test]
fn cuda_argmax_ties_break_to_lowest_index() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // Two equal max values at idx 0 and idx 3; both equal 5.0.
    // Lowest-index wins -> argmax = 0.
    let data: Vec<f32> = vec![5.0, 1.0, -2.0, 5.0, 3.0, 4.0];
    let x_cd = CandleTensor::from_vec(data, (1, 6), &dev)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = ops::argmax_last_dim(&x_kt).expect("argmax");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_vec: Vec<i64> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((1,))
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    assert_eq!(got_vec, vec![0i64]);
}
