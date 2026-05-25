//! Parity test: kt CUDA `transpose(d0, d1)` / `permute(axes)` followed
//! by `.contiguous()` produces byte-identical output to the CPU
//! reference (`.transpose(...).contiguous()` on a `CpuStorage`-backed
//! kt-Tensor).
//!
//! # Finding (#1082 substrate)
//!
//! `Tensor::transpose` and `Tensor::permute` are zero-copy layout ops:
//! they only permute `Layout::shape` and `Layout::strides` (see
//! `crates/kiln-tensor/src/layout.rs::Layout::{transpose,permute}`).
//! The transposed tensor shares storage with its parent; no kernel
//! launch is needed for the view itself.
//!
//! The *materializing* step is `.contiguous()`. For CUDA storage
//! `Tensor::contiguous()` dispatches to `cuda_storage::cuda_contiguous`
//! (`crates/kiln-tensor/src/cuda_storage.rs`), whose backing kernel
//! `kiln_contiguous_copy_async` (`crates/kiln-tensor/csrc/contiguous.cu`)
//! is fully **stride-aware**: per logical output element it unflattens
//! the linear index against the output shape and accumulates the
//! source byte offset using arbitrary input `strides_e[d]` plus the
//! layout's `start_offset`. Rank up to 8, every non-packed dtype.
//!
//! Therefore a transposed view + `.contiguous()` already materializes
//! the transposed layout into a contiguous CUDA buffer correctly —
//! **no dedicated `cuda_transpose` / `cuda_permute` kernel is needed**,
//! and no separate op-layer `cuda_fwd` plumbing exists for transpose
//! or permute (they live as `Tensor` methods, not `Op` nodes).
//!
//! This test locks in that property by sweeping shape × dtype ×
//! axis-swap and asserting byte-identical output between CUDA and CPU
//! reference paths.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_to_host_copy, CpuStorage, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

/// Deterministic small-magnitude pattern; magnitudes ≤ 1.0 so BF16/F16
/// quantization is exactly the same on CPU vs CUDA paths (both go
/// through `half::{bf16,f16}::from_f32`).
fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEAD_BEEF).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

fn cpu_kt(data: &[f32], shape: &[usize], dtype: CandleDType) -> Tensor {
    // Build a CPU kt-Tensor at the matching dtype so the
    // transpose-then-contiguous reference quantizes the *same* way
    // the CUDA path does.
    let x = Tensor::from_slice(data, shape.to_vec()).unwrap();
    match dtype {
        CandleDType::F32 => x,
        CandleDType::BF16 => kiln_tensor::ops::cast(&x, kiln_tensor::DType::BF16).unwrap(),
        CandleDType::F16 => kiln_tensor::ops::cast(&x, kiln_tensor::DType::F16).unwrap(),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn cuda_kt(data: &[f32], shape: &[usize], dtype: CandleDType, dev: &CandleDevice) -> Tensor {
    let cd = match shape {
        [a] => CandleTensor::from_vec(data.to_vec(), (*a,), dev).unwrap(),
        [a, b] => CandleTensor::from_vec(data.to_vec(), (*a, *b), dev).unwrap(),
        [a, b, c] => CandleTensor::from_vec(data.to_vec(), (*a, *b, *c), dev).unwrap(),
        [a, b, c, d] => CandleTensor::from_vec(data.to_vec(), (*a, *b, *c, *d), dev).unwrap(),
        other => panic!("unsupported test shape {other:?}"),
    };
    let cd = cd.to_dtype(dtype).unwrap();
    kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap()
}

fn host_bytes(t: &Tensor) -> Vec<u8> {
    // Force materialization, copy to host, then return raw bytes.
    let contig = t.contiguous().expect("contiguous");
    match contig.device() {
        kiln_tensor::Device::Cpu => {
            let cpu = contig
                .storage()
                .as_any()
                .downcast_ref::<CpuStorage>()
                .expect("CpuStorage");
            cpu.as_bytes().to_vec()
        }
        kiln_tensor::Device::Cuda(_) => {
            let host = cuda_to_host_copy(&contig).expect("D2H");
            let cpu = host
                .storage()
                .as_any()
                .downcast_ref::<CpuStorage>()
                .expect("CpuStorage");
            cpu.as_bytes().to_vec()
        }
        other => panic!("unexpected device {other}"),
    }
}

fn run_transpose_parity(shape: Vec<usize>, dtype: CandleDType, d0: usize, d1: usize) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern(n, 31);

    // CUDA path: transpose view → contiguous (kernel-materialized).
    let kt_cuda = cuda_kt(&data, &shape, dtype, &dev);
    let transposed_cuda = kt_cuda.transpose(d0, d1).expect("transpose");

    // Layout invariants — the view shares storage and is generally
    // non-contiguous (unless one of the axes is size 1).
    let mut expected_shape = shape.clone();
    expected_shape.swap(d0, d1);
    assert_eq!(transposed_cuda.shape(), expected_shape.as_slice());

    let cuda_bytes = host_bytes(&transposed_cuda);

    // CPU reference: same transpose, then contiguous via the CPU
    // backend's stride-aware byte-copy walk in `Tensor::contiguous()`.
    let kt_cpu = cpu_kt(&data, &shape, dtype);
    let transposed_cpu = kt_cpu.transpose(d0, d1).expect("cpu transpose");
    assert_eq!(transposed_cpu.shape(), expected_shape.as_slice());
    let cpu_bytes = host_bytes(&transposed_cpu);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(
        cuda_bytes.len(),
        cpu_bytes.len(),
        "byte-length mismatch for shape={shape:?} dtype={dtype:?} swap=({d0},{d1})"
    );
    for (i, (a, b)) in cpu_bytes.iter().zip(cuda_bytes.iter()).enumerate() {
        assert_eq!(
            a, b,
            "shape={shape:?} dtype={dtype:?} swap=({d0},{d1}) byte {i}: cpu={a} cuda={b}"
        );
    }
}

fn run_permute_parity(shape: Vec<usize>, dtype: CandleDType, axes: Vec<usize>) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern(n, 47);

    let kt_cuda = cuda_kt(&data, &shape, dtype, &dev);
    let permuted_cuda = kt_cuda.permute(&axes).expect("permute");

    let expected_shape: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();
    assert_eq!(permuted_cuda.shape(), expected_shape.as_slice());

    let cuda_bytes = host_bytes(&permuted_cuda);

    let kt_cpu = cpu_kt(&data, &shape, dtype);
    let permuted_cpu = kt_cpu.permute(&axes).expect("cpu permute");
    let cpu_bytes = host_bytes(&permuted_cpu);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(
        cuda_bytes.len(),
        cpu_bytes.len(),
        "byte-length mismatch for shape={shape:?} dtype={dtype:?} axes={axes:?}"
    );
    for (i, (a, b)) in cpu_bytes.iter().zip(cuda_bytes.iter()).enumerate() {
        assert_eq!(
            a, b,
            "shape={shape:?} dtype={dtype:?} axes={axes:?} byte {i}: cpu={a} cuda={b}"
        );
    }
}

// --- transpose: rank 2, simple matrix transpose ---

#[test]
fn cuda_transpose_rank2_f32() {
    run_transpose_parity(vec![5, 7], CandleDType::F32, 0, 1);
}

#[test]
fn cuda_transpose_rank2_bf16() {
    run_transpose_parity(vec![6, 4], CandleDType::BF16, 0, 1);
}

#[test]
fn cuda_transpose_rank2_f16() {
    run_transpose_parity(vec![8, 3], CandleDType::F16, 0, 1);
}

// --- transpose: rank 3, swap adjacent and non-adjacent axes ---

#[test]
fn cuda_transpose_rank3_swap01_f32() {
    run_transpose_parity(vec![3, 5, 4], CandleDType::F32, 0, 1);
}

#[test]
fn cuda_transpose_rank3_swap02_f32() {
    run_transpose_parity(vec![3, 5, 4], CandleDType::F32, 0, 2);
}

#[test]
fn cuda_transpose_rank3_swap12_bf16() {
    run_transpose_parity(vec![3, 5, 4], CandleDType::BF16, 1, 2);
}

// --- transpose: rank 4, model-shape-style ---

#[test]
fn cuda_transpose_rank4_swap12_bf16() {
    // Realistic [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
    run_transpose_parity(vec![2, 4, 3, 8], CandleDType::BF16, 1, 2);
}

#[test]
fn cuda_transpose_rank4_swap_inner_f32() {
    run_transpose_parity(vec![2, 4, 3, 8], CandleDType::F32, 2, 3);
}

// --- transpose: same-axis is identity ---

#[test]
fn cuda_transpose_identity_axis_f32() {
    run_transpose_parity(vec![4, 6, 5], CandleDType::F32, 1, 1);
}

// --- transpose twice = back to original (contiguous round-trip) ---

#[test]
fn cuda_transpose_double_swap_round_trip_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let shape = vec![4, 5, 3];
    let data = pattern(60, 99);
    let kt = cuda_kt(&data, &shape, CandleDType::F32, &dev);
    let twice = kt.transpose(0, 2).unwrap().transpose(0, 2).unwrap();
    assert_eq!(twice.shape(), shape.as_slice());
    let got = host_bytes(&twice);
    let expected = host_bytes(&kt);
    assert_eq!(got, expected, "double-transpose should round-trip byte-identically");
}

// --- permute: full axis reorder ---

#[test]
fn cuda_permute_rank3_reverse_f32() {
    run_permute_parity(vec![3, 4, 5], CandleDType::F32, vec![2, 1, 0]);
}

#[test]
fn cuda_permute_rank3_cyclic_bf16() {
    run_permute_parity(vec![3, 4, 5], CandleDType::BF16, vec![1, 2, 0]);
}

#[test]
fn cuda_permute_rank4_arbitrary_f16() {
    run_permute_parity(vec![2, 3, 4, 5], CandleDType::F16, vec![3, 0, 2, 1]);
}

#[test]
fn cuda_permute_identity_f32() {
    run_permute_parity(vec![4, 5, 3], CandleDType::F32, vec![0, 1, 2]);
}

// --- structural: a transposed view is zero-copy (Arc-shared storage) ---

#[test]
fn cuda_transpose_view_shares_storage() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let kt = cuda_kt(&data, &[2, 3, 4], CandleDType::F32, &dev);
    let view = kt.transpose(0, 2).unwrap();

    // Storage Arc pointer must match — the view did NOT allocate or
    // launch a kernel. This is the contract that lets the model code
    // chain `transpose().reshape().matmul()` without per-call kernel
    // dispatch.
    let a = kt.storage();
    let b = view.storage();
    assert!(
        std::sync::Arc::ptr_eq(a, b),
        "Tensor::transpose must be zero-copy (Arc-shared storage)"
    );

    // Shape and strides reflect the swap.
    assert_eq!(view.shape(), &[4, 3, 2]);
    let kt_strides = kt.strides().to_vec();
    let view_strides = view.strides().to_vec();
    assert_eq!(
        view_strides,
        vec![kt_strides[2], kt_strides[1], kt_strides[0]],
        "strides must be permuted to match the swap"
    );
}
