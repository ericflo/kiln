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
//! - `bf16_matmul_matches_cpu_reference` — `[64, 128] @ [128, 96]`
//!   matmul in BF16; compares against a hand-rolled CPU FP32
//!   reference (BF16 inputs upcast to FP32 for the reference path)
//!   under the Phase 0 BF16 tolerance band.
//! - `algo_cache_hit_skips_heuristic_and_matches` — second call at
//!   the same shape reuses the cached algo blob; the underlying
//!   cublasLt matmul still produces a byte-identical result.
//!
//! #1082: the previous incarnation of this file used candle as both
//! the CUDA allocator and the matmul reference. It's now fully
//! candle-free: cudarc owns allocation + H2D/D2H, and the reference
//! matmul is a tiny CPU triple-loop. This was the last candle import
//! in kiln-blas.

#![cfg(feature = "cublaslt")]

use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaContext, DevicePtr};
use half::bf16;

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

/// Resolve a cudarc `CudaContext`; bail (skip) the test if CUDA isn't
/// available (e.g., running tests under `cargo test` on a CPU-only
/// host with the feature accidentally enabled).
fn try_cuda_context() -> Option<Arc<CudaContext>> {
    CudaContext::new(0).ok()
}

/// Hand-rolled CPU reference matmul: `[m, k] @ [k, n] = [m, n]`,
/// row-major, BF16 inputs upcast to FP32 for the inner accumulator
/// then rounded back to BF16. The output is also returned in FP32
/// so the parity test can compare on a common dtype.
///
/// This is the candle-free replacement for the old
/// `candle.matmul(...).to_dtype(F32)` reference path — the
/// arithmetic is identical (FP32 accumulate over BF16 multiplicands)
/// and BF16 rounding is bit-equivalent to candle's behavior at the
/// 5e-2 tolerance band used by this test.
fn cpu_matmul_bf16_to_f32(a: &[bf16], b: &[bf16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let av = a[i * k + p].to_f32();
                let bv = b[p * n + j].to_f32();
                acc += av * bv;
            }
            // Round to BF16 then back to FP32 to match the dtype path
            // cublasLt takes on the GPU side (BF16 outputs are stored
            // in BF16, so any consumer reading the output upcasts to
            // FP32 via the same rounding step).
            out[i * n + j] = bf16::from_f32(acc).to_f32();
        }
    }
    out
}

/// Copy a host slice of BF16 onto the device, returning the
/// `CudaSlice<u8>` byte-view (cublasLt sees raw bytes).
fn htod_bf16(ctx: &Arc<CudaContext>, host: &[bf16]) -> cudarc::driver::CudaSlice<u8> {
    // SAFETY: bf16 has the same byte layout as u16 (2 bytes, native
    // endianness on supported targets). We reinterpret the host
    // slice as `&[u8]` of length `host.len() * 2` to hand to cudarc's
    // `clone_htod`. The pointer is valid for `len * 2` bytes.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * std::mem::size_of::<bf16>())
    };
    let stream = ctx.default_stream();
    stream
        .clone_htod(bytes)
        .expect("clone_htod (host -> device)")
}

/// Copy a device `CudaSlice<u8>` back to host as a `Vec<bf16>`.
fn dtoh_bf16(ctx: &Arc<CudaContext>, slice: &cudarc::driver::CudaSlice<u8>, count: usize) -> Vec<bf16> {
    let stream = ctx.default_stream();
    let mut host = vec![0u8; count * std::mem::size_of::<bf16>()];
    stream
        .memcpy_dtoh(slice, &mut host)
        .expect("memcpy_dtoh (device -> host)");
    // SAFETY: same layout reinterpret as in `htod_bf16` — `host.len()`
    // is `count * 2` by construction, so the bf16 slice has exactly
    // `count` elements.
    let bf16_slice: &[bf16] = unsafe {
        std::slice::from_raw_parts(host.as_ptr() as *const bf16, count)
    };
    bf16_slice.to_vec()
}

/// Run the matmul against the handle. Returns `(algo_id, bytes_written, output_host)`.
fn run_matmul(
    handle: &CublasLtMatmulHandle,
    ctx: &Arc<CudaContext>,
    a_dev: &cudarc::driver::CudaSlice<u8>,
    b_dev: &cudarc::driver::CudaSlice<u8>,
    out_dev: &cudarc::driver::CudaSlice<u8>,
    m: usize,
    n: usize,
    req: &MatmulRequest,
) -> (i32, u64, Vec<bf16>) {
    let stream = ctx.default_stream();
    let (a_ptr, _ag): (CUdeviceptr, _) = a_dev.device_ptr(&stream);
    let (b_ptr, _bg): (CUdeviceptr, _) = b_dev.device_ptr(&stream);
    let (out_ptr, _og): (CUdeviceptr, _) = out_dev.device_ptr(&stream);
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
    // `memcpy_dtoh` on the same stream synchronizes-on-completion
    // (cudarc 0.19 CudaStream semantics), so no separate
    // `stream.synchronize()` is required between the matmul and
    // the D2H read-back.
    let host = dtoh_bf16(ctx, out_dev, m * n);
    (outcome.algo_blob.algo_id, outcome.bytes_written, host)
}

#[test]
fn handle_creates_and_destroys() {
    let Some(ctx) = try_cuda_context() else {
        eprintln!("CUDA device not available; skipping");
        return;
    };
    let cache = Arc::new(Mutex::new(AlgoCache::new()));
    let handle = CublasLtMatmulHandle::new_ctx(ctx, 0, cache, None).expect("handle create");
    // Make sure debug doesn't panic.
    let s = format!("{handle:?}");
    assert!(s.contains("CublasLtMatmulHandle"));
}

#[test]
fn bf16_matmul_matches_cpu_reference() {
    let Some(ctx) = try_cuda_context() else {
        eprintln!("CUDA device not available; skipping");
        return;
    };

    let m = 64usize;
    let n = 96usize;
    let k = 128usize;

    // Build A, B as deterministic BF16 patterns. Same byte-level
    // values as the legacy candle-based test so the parity band
    // (5e-2 abs / rel) stays meaningful.
    let mut a_vec_f32 = Vec::with_capacity(m * k);
    for i in 0..(m * k) {
        a_vec_f32.push(((i % 17) as f32 - 8.0) / 16.0);
    }
    let mut b_vec_f32 = Vec::with_capacity(k * n);
    for i in 0..(k * n) {
        b_vec_f32.push(((i % 23) as f32 - 11.0) / 16.0);
    }
    let a_bf16: Vec<bf16> = a_vec_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let b_bf16: Vec<bf16> = b_vec_f32.iter().map(|v| bf16::from_f32(*v)).collect();

    // CPU reference path.
    let ref_vec = cpu_matmul_bf16_to_f32(&a_bf16, &b_bf16, m, k, n);

    // Upload inputs, allocate output.
    let a_dev = htod_bf16(&ctx, &a_bf16);
    let b_dev = htod_bf16(&ctx, &b_bf16);
    let out_dev = ctx
        .default_stream()
        .alloc_zeros::<u8>(m * n * std::mem::size_of::<bf16>())
        .expect("alloc_zeros out");

    // Build the handle.
    let cache = Arc::new(Mutex::new(AlgoCache::new()));
    let handle = CublasLtMatmulHandle::new_ctx(Arc::clone(&ctx), 0, Arc::clone(&cache), None)
        .expect("handle create");

    let req = req_bf16(m as u64, n as u64, k as u64);
    let (algo_id, bytes_written, got_bf16) =
        run_matmul(&handle, &ctx, &a_dev, &b_dev, &out_dev, m, n, &req);
    assert!(algo_id >= 0, "algo_id should be set ({algo_id})");
    assert_eq!(bytes_written, (m * n * 2) as u64);

    let got_vec: Vec<f32> = got_bf16.iter().map(|v| v.to_f32()).collect();

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
    let Some(ctx) = try_cuda_context() else {
        eprintln!("CUDA device not available; skipping");
        return;
    };

    let m = 32usize;
    let n = 64usize;
    let k = 48usize;

    let mut a_vec_f32 = Vec::with_capacity(m * k);
    for i in 0..(m * k) {
        a_vec_f32.push(((i % 13) as f32 - 6.0) / 13.0);
    }
    let mut b_vec_f32 = Vec::with_capacity(k * n);
    for i in 0..(k * n) {
        b_vec_f32.push(((i % 19) as f32 - 9.0) / 19.0);
    }
    let a_bf16: Vec<bf16> = a_vec_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let b_bf16: Vec<bf16> = b_vec_f32.iter().map(|v| bf16::from_f32(*v)).collect();

    let a_dev = htod_bf16(&ctx, &a_bf16);
    let b_dev = htod_bf16(&ctx, &b_bf16);

    let cache = Arc::new(Mutex::new(AlgoCache::new()));
    let handle = CublasLtMatmulHandle::new_ctx(Arc::clone(&ctx), 0, Arc::clone(&cache), None)
        .expect("handle create");

    let req = req_bf16(m as u64, n as u64, k as u64);

    // Two fresh output buffers — we want to compare the actual bytes
    // written by each call independently.
    let out1_dev = ctx
        .default_stream()
        .alloc_zeros::<u8>(m * n * std::mem::size_of::<bf16>())
        .expect("alloc_zeros out1");
    let (algo1, _, got1) = run_matmul(&handle, &ctx, &a_dev, &b_dev, &out1_dev, m, n, &req);

    let out2_dev = ctx
        .default_stream()
        .alloc_zeros::<u8>(m * n * std::mem::size_of::<bf16>())
        .expect("alloc_zeros out2");
    let (algo2, _, got2) = run_matmul(&handle, &ctx, &a_dev, &b_dev, &out2_dev, m, n, &req);

    // Both calls picked the same algo (cache hit on second).
    assert_eq!(algo1, algo2);

    // Outputs are bit-identical run-to-run.
    assert_eq!(got1, got2);

    // Cache has exactly one entry.
    assert_eq!(cache.lock().unwrap().len(), 1);

    let stats = handle.algo_cache_stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.inserts, 1);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.lookups(), 2);
    assert_eq!(stats.hit_rate(), Some(0.5));

    // Workspace pool saw two calls.
    let pool = handle.workspace_pool();
    assert_eq!(pool.call_count, 2);
}
