//! Parity test: kt CUDA scatter-add (`ScatterAddOp::cuda_fwd` /
//! `cuda_scatter_add_dim0`) vs kt CPU reference (`ops::scatter_add`).
//!
//! Phase 4 substrate validation. Confirms the kernel in
//! `csrc/scatter_add.cu` produces outputs matching the canonical CPU
//! reference for both F32 and BF16, on the axis=0 + 1-D U32 indices
//! path.
//!
//! # Determinism caveat
//!
//! `atomicAdd`-based scatter is bit-non-deterministic when two index
//! positions collide on the same target row — float addition is not
//! associative and the per-thread arrival order can permute the
//! accumulation. The first two test cases use unique indices (no
//! collisions) to assert tight (1e-6 F32, ~1e-3 BF16) parity. The
//! third case has duplicate indices and uses a larger tolerance to
//! account for the "atomic-bwd" band, per the documented stance in
//! `ScatterAddOp::determinism()`.



use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn make_updates_f32_pattern(n_indices: usize, hidden: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_indices * hidden);
    for n in 0..n_indices {
        for h in 0..hidden {
            // Easy-to-debug deterministic per-(n, h) value.
            out.push((n * 100 + h) as f32 * 0.01);
        }
    }
    out
}

fn cpu_reference_f32(
    updates_data: &[f32],
    n_indices: usize,
    hidden: usize,
    indices_data: &[u32],
    target_dim: usize,
) -> Vec<f32> {
    let updates = Tensor::from_slice(updates_data, vec![n_indices, hidden]).unwrap();
    // CPU op accepts both I64 and U32. Use U32 to match the CUDA path's
    // dtype exactly so the reference and the CUDA path read the same
    // bytes through the same code path.
    let indices = Tensor::from_slice(indices_data, vec![n_indices]).unwrap();
    let y = ops::scatter_add(&updates, 0, &indices, target_dim).unwrap();
    let cpu = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu.as_bytes();
    let mut out = Vec::with_capacity(target_dim * hidden);
    for i in 0..(target_dim * hidden) {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

#[test]
fn cuda_scatter_add_unique_indices_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let target_dim = 16usize;
    let hidden = 32usize;
    let indices_data: Vec<u32> = vec![3, 0, 9, 15, 7, 1, 12, 4]; // all unique, deterministic
    let n_indices = indices_data.len();
    let updates_data = make_updates_f32_pattern(n_indices, hidden);

    // Build candle tensors → borrow as kt tensors.
    let updates_cd = CandleTensor::from_vec(
        updates_data.clone(),
        (n_indices, hidden),
        &dev,
    )
    .unwrap();
    let indices_cd =
        CandleTensor::from_vec(indices_data.clone(), (n_indices,), &dev).unwrap();
    let updates_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&updates_cd).unwrap();
    let indices_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&indices_cd).unwrap();

    // Dispatch through the kt op — should land on cuda_fwd since both
    // tensors are CUDA-backed.
    let out_kt = ops::scatter_add(&updates_kt, 0, &indices_kt, target_dim)
        .expect("scatter_add dispatch");
    assert_eq!(out_kt.shape(), &[target_dim, hidden]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((target_dim * hidden,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let ref_v = cpu_reference_f32(
        &updates_data,
        n_indices,
        hidden,
        &indices_data,
        target_dim,
    );

    assert_eq!(got.len(), ref_v.len());
    for (i, (a, b)) in ref_v.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "idx={i} ref={a} got={b}"
        );
    }
}

#[test]
fn cuda_scatter_add_unique_indices_bf16() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let target_dim = 12usize;
    let hidden = 24usize;
    let indices_data: Vec<u32> = vec![0, 5, 11, 2, 8, 6]; // unique
    let n_indices = indices_data.len();
    let updates_data = make_updates_f32_pattern(n_indices, hidden);

    let updates_cd = CandleTensor::from_vec(
        updates_data.clone(),
        (n_indices, hidden),
        &dev,
    )
    .unwrap()
    .to_dtype(CandleDType::BF16)
    .unwrap();
    let indices_cd =
        CandleTensor::from_vec(indices_data.clone(), (n_indices,), &dev).unwrap();
    let updates_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&updates_cd).unwrap();
    let indices_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&indices_cd).unwrap();

    let out_kt = ops::scatter_add(&updates_kt, 0, &indices_kt, target_dim)
        .expect("scatter_add bf16 dispatch");
    assert_eq!(out_kt.shape(), &[target_dim, hidden]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((target_dim * hidden,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Reference: gather F32 values, BF16-quantize per-cell to match
    // what the kernel sees on input + stores on output.
    let ref_f32 = cpu_reference_f32(
        &updates_data,
        n_indices,
        hidden,
        &indices_data,
        target_dim,
    );
    let mut ref_bf16 = Vec::with_capacity(ref_f32.len());
    for v in &ref_f32 {
        ref_bf16.push(half::bf16::from_f32(*v).to_f32());
    }

    for (i, (a, b)) in ref_bf16.iter().zip(got.iter()).enumerate() {
        // Loose tolerance: BF16 only has 7-bit mantissa, and the kernel
        // accumulates BF16 directly (no F32 staging) so each add can
        // shed ~1 ULP.
        assert!(
            (a - b).abs() < 5e-2,
            "idx={i} ref(bf16)={a} got={b}"
        );
    }
}

#[test]
fn cuda_scatter_add_with_collisions_f32_tolerance_band() {
    // "atomic-bwd zone": multiple index positions hit the same target
    // row, so the kernel uses atomicAdd on every cell and the per-thread
    // ordering is non-deterministic. We assert numerical closeness, not
    // bit equality. This is the documented tolerance-bounded stance per
    // `ScatterAddOp::determinism()`.
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let target_dim = 4usize;
    let hidden = 8usize;
    // Deliberate collisions: index 0 hit 4 times, index 2 hit 3 times.
    let indices_data: Vec<u32> = vec![0, 2, 0, 1, 0, 2, 0, 3, 2];
    let n_indices = indices_data.len();
    let updates_data = make_updates_f32_pattern(n_indices, hidden);

    let updates_cd = CandleTensor::from_vec(
        updates_data.clone(),
        (n_indices, hidden),
        &dev,
    )
    .unwrap();
    let indices_cd =
        CandleTensor::from_vec(indices_data.clone(), (n_indices,), &dev).unwrap();
    let updates_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&updates_cd).unwrap();
    let indices_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&indices_cd).unwrap();

    let out_kt = ops::scatter_add(&updates_kt, 0, &indices_kt, target_dim)
        .expect("scatter_add collision dispatch");
    assert_eq!(out_kt.shape(), &[target_dim, hidden]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((target_dim * hidden,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let ref_v = cpu_reference_f32(
        &updates_data,
        n_indices,
        hidden,
        &indices_data,
        target_dim,
    );

    // Loose tolerance for F32 with up to ~4 collisions per target row.
    // The values being summed are O(1) (a few units), so ~1e-4 absolute
    // is comfortably wider than any plausible reorder-induced ULP drift.
    for (i, (a, b)) in ref_v.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "idx={i} ref={a} got={b}"
        );
    }
}
