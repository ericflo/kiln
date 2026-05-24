//! Parity test: kt CUDA chunk / split_with_sizes
//! (`ChunkOp::cuda_fwd` / `SplitWithSizesOp::cuda_fwd` / `ops::chunk`
//! / `ops::split_with_sizes` on CUDA inputs) vs kt CPU reference.
//!
//! Chunk and split are pure layout/shape ops — they construct new
//! tensors with adjusted `Layout` over the parent's storage `Arc`,
//! returning **zero-copy views**. The "CUDA fwd" path is therefore
//! identical to the CPU path, but the test validates that:
//!
//! 1. The view's logical shape matches what the CPU reference
//!    produces.
//! 2. After materializing each view to a candle host vector (via
//!    `.contiguous()?` on the kt-Tensor, which routes through
//!    `cuda_contiguous` for CUDA storage), the data matches.
//! 3. The dispatch through `ops::chunk` / `ops::split_with_sizes`
//!    works on CUDA tensors without panicking or returning CPU
//!    storage.



use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_contiguous, ops, Device, Tensor};

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

/// Materialize a kt-Tensor (which may be a non-contiguous view *or*
/// a contiguous view over a Borrowed CudaStorage) into a flat F32
/// vector.
///
/// Why not `kt.contiguous()`? `Tensor::contiguous` short-circuits and
/// returns the input as-is when `is_contiguous()` is true. Chunks of
/// a 1-D tensor (axis=0 splits) ARE contiguous views, so
/// `.contiguous()` leaves their storage as `SliceOwner::Borrowed`,
/// and `kt_tensor_to_candle_cuda_copy` panics because
/// `CudaStorage::slice()` doesn't support borrowed storage.
///
/// Routing through `cuda_contiguous` directly forces a fresh Owned
/// CUDA allocation and a kernel-side stride-aware copy, which is what
/// the kt→candle adapter requires.
fn materialize_to_f32(kt: &Tensor) -> Vec<f32> {
    let owned = cuda_contiguous(kt).unwrap();
    let n: usize = owned.shape().iter().product();
    kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&owned)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// CPU reference: run `ops::chunk` on a CPU kt-Tensor and return the
/// per-output flat F32 vectors and per-output shapes.
fn cpu_chunk_reference(
    data: &[f32],
    shape: Vec<usize>,
    n_chunks: usize,
    axis: usize,
) -> Vec<(Vec<usize>, Vec<f32>)> {
    let t = Tensor::from_slice(data, shape).unwrap();
    let parts = ops::chunk(&t, n_chunks, axis).unwrap();
    parts
        .into_iter()
        .map(|p| {
            let contig = p.contiguous().unwrap();
            let cpu_storage = contig
                .storage()
                .as_any()
                .downcast_ref::<kiln_tensor::CpuStorage>()
                .unwrap();
            let bytes = cpu_storage.as_bytes();
            let n: usize = contig.shape().iter().product();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
            }
            (contig.shape().to_vec(), out)
        })
        .collect()
}

fn cpu_split_reference(
    data: &[f32],
    shape: Vec<usize>,
    sizes: &[usize],
    axis: usize,
) -> Vec<(Vec<usize>, Vec<f32>)> {
    let t = Tensor::from_slice(data, shape).unwrap();
    let parts = ops::split_with_sizes(&t, sizes, axis).unwrap();
    parts
        .into_iter()
        .map(|p| {
            let contig = p.contiguous().unwrap();
            let cpu_storage = contig
                .storage()
                .as_any()
                .downcast_ref::<kiln_tensor::CpuStorage>()
                .unwrap();
            let bytes = cpu_storage.as_bytes();
            let n: usize = contig.shape().iter().product();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
            }
            (contig.shape().to_vec(), out)
        })
        .collect()
}

#[test]
fn cuda_chunk_even_axis_0_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // [6] split into 3 → three length-2 chunks.
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let cd = CandleTensor::from_vec(data.clone(), (6,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();
    assert!(matches!(kt.device(), Device::Cuda(_)));

    let parts = ops::chunk(&kt, 3, 0).expect("chunk");
    assert_eq!(parts.len(), 3);
    for p in &parts {
        // Views still report CUDA device — they're zero-copy over the
        // parent's CUDA storage Arc.
        assert!(matches!(p.device(), Device::Cuda(_)));
        assert_eq!(p.shape(), &[2]);
    }

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let want = cpu_chunk_reference(&data, vec![6], 3, 0);
    for (i, p) in parts.iter().enumerate() {
        let got = materialize_to_f32(p);
        assert_eq!(want[i].0, p.shape());
        assert_eq!(want[i].1, got, "chunk {i}");
    }
}

#[test]
fn cuda_chunk_uneven_axis_0_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // [7] split into 3 → 3, 2, 2.
    let data: Vec<f32> = (0..7).map(|i| i as f32).collect();
    let cd = CandleTensor::from_vec(data.clone(), (7,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let parts = ops::chunk(&kt, 3, 0).expect("chunk uneven");
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].shape(), &[3]);
    assert_eq!(parts[1].shape(), &[2]);
    assert_eq!(parts[2].shape(), &[2]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let want = cpu_chunk_reference(&data, vec![7], 3, 0);
    for (i, p) in parts.iter().enumerate() {
        let got = materialize_to_f32(p);
        assert_eq!(want[i].1, got, "uneven chunk {i}");
    }
}

#[test]
fn cuda_chunk_inner_axis_bf16() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // [2, 4] split inner into 2 → two [2, 2] views.
    let data = pattern_f32(2 * 4, 91);
    let cd = CandleTensor::from_vec(data.clone(), (2, 4), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let parts = ops::chunk(&kt, 2, 1).expect("chunk inner");
    assert_eq!(parts.len(), 2);
    for p in &parts {
        assert_eq!(p.shape(), &[2, 2]);
    }

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Reference: BF16-quantize the F32 data, then chunk over BF16.
    let bf16_data: Vec<f32> = data.iter().map(|v| half::bf16::from_f32(*v).to_f32()).collect();
    // Layout for chunk-axis-1 on [2,4]:
    //   row 0: [b0,b1, b2,b3] → first chunk gets [b0,b1], second [b2,b3]
    //   row 1: [b4,b5, b6,b7] → first chunk gets [b4,b5], second [b6,b7]
    let want0: Vec<f32> = vec![bf16_data[0], bf16_data[1], bf16_data[4], bf16_data[5]];
    let want1: Vec<f32> = vec![bf16_data[2], bf16_data[3], bf16_data[6], bf16_data[7]];
    assert_eq!(want0, materialize_to_f32(&parts[0]));
    assert_eq!(want1, materialize_to_f32(&parts[1]));
}

#[test]
fn cuda_split_with_sizes_qkv_pattern_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    // QKV pattern: [B*T=4, hidden=24] → q(16), k(4), v(4) along axis 1.
    let batch = 4usize;
    let q_dim = 16usize;
    let k_dim = 4usize;
    let v_dim = 4usize;
    let hidden = q_dim + k_dim + v_dim;
    let data = pattern_f32(batch * hidden, 101);
    let cd = CandleTensor::from_vec(data.clone(), (batch, hidden), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let parts = ops::split_with_sizes(&kt, &[q_dim, k_dim, v_dim], 1).expect("split");
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].shape(), &[batch, q_dim]);
    assert_eq!(parts[1].shape(), &[batch, k_dim]);
    assert_eq!(parts[2].shape(), &[batch, v_dim]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let want = cpu_split_reference(
        &data,
        vec![batch, hidden],
        &[q_dim, k_dim, v_dim],
        1,
    );
    for (i, p) in parts.iter().enumerate() {
        let got = materialize_to_f32(p);
        assert_eq!(want[i].1, got, "split part {i}");
    }
}

#[test]
fn cuda_chunk_op_handle_explicit() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cd = CandleTensor::from_vec(data.clone(), (12,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let op = ops::ChunkOp::new(4, 0);
    let parts = op
        .cuda_fwd(&kt)
        .expect("cuda_fwd")
        .expect("Some(parts)");
    assert_eq!(parts.len(), 4);
    for p in &parts {
        assert_eq!(p.shape(), &[3]);
    }

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let mut all = Vec::new();
    for p in &parts {
        all.extend(materialize_to_f32(p));
    }
    assert_eq!(all, data);
}

#[test]
fn cuda_split_with_sizes_op_handle_explicit() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let cd = CandleTensor::from_vec(data.clone(), (10,), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    let op = ops::SplitWithSizesOp::new(0);
    let parts = op
        .cuda_fwd(&kt, &[3, 2, 5])
        .expect("cuda_fwd")
        .expect("Some(parts)");
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].shape(), &[3]);
    assert_eq!(parts[1].shape(), &[2]);
    assert_eq!(parts[2].shape(), &[5]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    assert_eq!(materialize_to_f32(&parts[0]), vec![0.0, 1.0, 2.0]);
    assert_eq!(materialize_to_f32(&parts[1]), vec![3.0, 4.0]);
    assert_eq!(materialize_to_f32(&parts[2]), vec![5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn cuda_chunk_concat_round_trip() {
    // Chunk a tensor then concat the chunks back — should reconstruct
    // the original bit-for-bit. Exercises both CUDA paths together.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let batch = 2usize;
    let hidden = 12usize;
    let data = pattern_f32(batch * hidden, 113);
    let cd = CandleTensor::from_vec(data.clone(), (batch, hidden), &dev).unwrap();
    let kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cd).unwrap();

    // Split into 4 along axis 1.
    let parts = ops::chunk(&kt, 4, 1).expect("chunk");
    assert_eq!(parts.len(), 4);

    // Each view needs to be contiguous before concat (the concat op
    // requires contiguous inputs).
    let contig_parts: Vec<Tensor> = parts.iter().map(|p| p.contiguous().unwrap()).collect();
    let refs: Vec<&Tensor> = contig_parts.iter().collect();

    let rebuilt = ops::concat(&refs, 1).expect("concat back");
    assert_eq!(rebuilt.shape(), kt.shape());

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got = materialize_to_f32(&rebuilt);
    assert_eq!(data, got);
}
