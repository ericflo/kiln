//! End-to-end smoke + parity tests for `CublasLtMatmulHandle`.
//!
//! These tests are gated on `--features cublaslt` and require an
//! actual CUDA device to run. They're skipped (compile-only) on
//! hosts without CUDA — `kiln-blas` deliberately keeps the
//! handle behind a feature flag so the default `cargo test` runs
//! everywhere.
//!
//! # Coverage
//!
//! - `handle_creates_and_destroys` — sanity: cublasLtCreate + Destroy
//!   succeed on the local A6000.
//! - `bf16_matmul_matches_candle` — `[64, 128] @ [128, 96]` matmul
//!   in BF16; compares against candle's `Tensor::matmul` element-wise
//!   under the Phase 0 BF16 tolerance band.
//! - `algo_cache_hit_skips_heuristic_and_matches` — second call at
//!   the same shape reuses the cached algo blob; the underlying
//!   cublasLt matmul still produces a byte-identical result.

#![cfg(feature = "cublaslt")]

use std::sync::{Arc, Mutex};

use candle_core::backend::BackendDevice;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_blas::{AlgoCache, CublasLtMatmulHandle, Epilogue, MatmulLayout, MatmulRequest};

/// Helper: build a row-major MatmulRequest.
fn req_bf16(m: u64, n: u64, k: u64) -> MatmulRequest {
    MatmulRequest {
        m,
        n,
        k,
        dtype: "bf16".to_string(),
        a_layout: MatmulLayout::RowMajor,
        b_layout: MatmulLayout::RowMajor,
        c_layout: MatmulLayout::RowMajor,
        epilogue: Epilogue::Identity,
        concurrent_streams: 1,
    }
}

/// Resolve a CUDA device; bail (skip) the test if CUDA isn't available
/// (e.g., running tests under `cargo test` on a CPU-only host with the
/// feature accidentally enabled).
fn try_cuda_device() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

/// Run the matmul: returns the result vector + the chosen algo id.
/// Scopes all storage guards so the output tensor can be re-used.
fn run_matmul(
    handle: &CublasLtMatmulHandle,
    cuda: &Arc<candle_core::CudaDevice>,
    a_bf16: &CandleTensor,
    b_bf16: &CandleTensor,
    out_bf16: &CandleTensor,
    req: &MatmulRequest,
) -> (i32, u64) {
    let stream = cuda.cuda_stream();
    let (a_storage, _) = a_bf16.storage_and_layout();
    let (b_storage, _) = b_bf16.storage_and_layout();
    let (out_storage, _) = out_bf16.storage_and_layout();
    let a_cuda = match &*a_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => panic!("a not on CUDA"),
    };
    let b_cuda = match &*b_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => panic!("b not on CUDA"),
    };
    let out_cuda = match &*out_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => panic!("out not on CUDA"),
    };
    let a_slice = a_cuda.as_cuda_slice::<half::bf16>().unwrap();
    let b_slice = b_cuda.as_cuda_slice::<half::bf16>().unwrap();
    let out_slice = out_cuda.as_cuda_slice::<half::bf16>().unwrap();
    let (a_ptr, _ag) = a_slice.device_ptr(&stream);
    let (b_ptr, _bg) = b_slice.device_ptr(&stream);
    let (out_ptr, _og) = out_slice.device_ptr(&stream);
    let stream_ptr = stream.cu_stream();
    let outcome = unsafe {
        handle
            .matmul(
                stream_ptr,
                req,
                a_ptr as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                std::ptr::null(),
            )
            .expect("matmul")
    };
    (outcome.algo_blob.algo_id, outcome.bytes_written)
}

#[test]
fn handle_creates_and_destroys() {
    let Some(dev) = try_cuda_device() else {
        eprintln!("CUDA device not available; skipping");
        return;
    };
    let cuda = match dev {
        CandleDevice::Cuda(c) => Arc::new(c),
        _ => unreachable!(),
    };
    let cache = Arc::new(Mutex::new(AlgoCache::new()));
    let handle = CublasLtMatmulHandle::new(cuda, 0, cache, None).expect("handle create");
    // Make sure debug doesn't panic.
    let s = format!("{handle:?}");
    assert!(s.contains("CublasLtMatmulHandle"));
}

#[test]
fn bf16_matmul_matches_candle() {
    let Some(dev) = try_cuda_device() else {
        eprintln!("CUDA device not available; skipping");
        return;
    };

    let m = 64usize;
    let n = 96usize;
    let k = 128usize;

    // Build A, B via candle (BF16). Deterministic pattern.
    let mut a_vec = Vec::with_capacity(m * k);
    for i in 0..(m * k) {
        a_vec.push(((i % 17) as f32 - 8.0) / 16.0);
    }
    let mut b_vec = Vec::with_capacity(k * n);
    for i in 0..(k * n) {
        b_vec.push(((i % 23) as f32 - 11.0) / 16.0);
    }
    let a_f32 = CandleTensor::from_vec(a_vec, (m, k), &dev).unwrap();
    let b_f32 = CandleTensor::from_vec(b_vec, (k, n), &dev).unwrap();
    let a_bf16 = a_f32.to_dtype(CandleDType::BF16).unwrap();
    let b_bf16 = b_f32.to_dtype(CandleDType::BF16).unwrap();

    // Candle reference path.
    let ref_bf16 = a_bf16.matmul(&b_bf16).unwrap();
    let ref_vec: Vec<f32> = ref_bf16
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Build the handle.
    let cuda = match dev {
        CandleDevice::Cuda(c) => Arc::new(c),
        _ => unreachable!(),
    };
    let cache = Arc::new(Mutex::new(AlgoCache::new()));
    let handle = CublasLtMatmulHandle::new(Arc::clone(&cuda), 0, Arc::clone(&cache), None)
        .expect("handle create");

    // Allocate the kt-side output buffer via candle.
    let out_bf16 = CandleTensor::zeros(
        (m, n),
        CandleDType::BF16,
        &CandleDevice::Cuda((*cuda).clone()),
    )
    .unwrap();

    let req = req_bf16(m as u64, n as u64, k as u64);
    let (algo_id, bytes_written) = run_matmul(&handle, &cuda, &a_bf16, &b_bf16, &out_bf16, &req);
    assert!(algo_id >= 0, "algo_id should be set ({algo_id})");
    assert_eq!(bytes_written, (m * n * 2) as u64);

    // Sync + read back. Use BackendDevice::synchronize on the device.
    cuda.synchronize().unwrap();

    let got_vec: Vec<f32> = out_bf16
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(ref_vec.len(), got_vec.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let abs = (a - b).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        let denom = a.abs().max(1.0);
        let rel = abs / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    // BF16 tolerance: small-shape BF16 matmul has ~7-bit mantissa.
    // 5e-2 absolute is generous but well below the "algorithm bug"
    // threshold.
    assert!(max_abs < 5e-2, "max_abs = {max_abs} (expected < 0.05)");
    assert!(max_rel < 5e-2, "max_rel = {max_rel} (expected < 0.05)");

    // Cache populated with the chosen algo.
    let cache_guard = cache.lock().unwrap();
    assert_eq!(cache_guard.len(), 1, "algo cache should have one entry");
}

#[test]
fn algo_cache_hit_skips_heuristic_and_matches() {
    let Some(dev) = try_cuda_device() else {
        eprintln!("CUDA device not available; skipping");
        return;
    };

    let m = 32usize;
    let n = 64usize;
    let k = 48usize;

    let mut a_vec = Vec::with_capacity(m * k);
    for i in 0..(m * k) {
        a_vec.push(((i % 13) as f32 - 6.0) / 13.0);
    }
    let mut b_vec = Vec::with_capacity(k * n);
    for i in 0..(k * n) {
        b_vec.push(((i % 19) as f32 - 9.0) / 19.0);
    }
    let a_bf16 = CandleTensor::from_vec(a_vec, (m, k), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let b_bf16 = CandleTensor::from_vec(b_vec, (k, n), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();

    let cuda = match dev.clone() {
        CandleDevice::Cuda(c) => Arc::new(c),
        _ => unreachable!(),
    };
    let cache = Arc::new(Mutex::new(AlgoCache::new()));
    let handle = CublasLtMatmulHandle::new(Arc::clone(&cuda), 0, Arc::clone(&cache), None)
        .expect("handle create");

    let req = req_bf16(m as u64, n as u64, k as u64);

    // Two fresh output buffers — we want to compare the actual bytes
    // written by each call independently.
    let out1 = CandleTensor::zeros(
        (m, n),
        CandleDType::BF16,
        &CandleDevice::Cuda((*cuda).clone()),
    )
    .unwrap();
    let (algo1, _) = run_matmul(&handle, &cuda, &a_bf16, &b_bf16, &out1, &req);
    cuda.synchronize().unwrap();
    let v1: Vec<f32> = out1
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let out2 = CandleTensor::zeros(
        (m, n),
        CandleDType::BF16,
        &CandleDevice::Cuda((*cuda).clone()),
    )
    .unwrap();
    let (algo2, _) = run_matmul(&handle, &cuda, &a_bf16, &b_bf16, &out2, &req);
    cuda.synchronize().unwrap();
    let v2: Vec<f32> = out2
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((m * n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Both calls picked the same algo (cache hit on second).
    assert_eq!(algo1, algo2);

    // Outputs are bit-identical run-to-run.
    assert_eq!(v1, v2);

    // Cache has exactly one entry.
    assert_eq!(cache.lock().unwrap().len(), 1);

    // Workspace pool saw two calls.
    let pool = handle.workspace_pool();
    assert_eq!(pool.call_count, 2);
}
