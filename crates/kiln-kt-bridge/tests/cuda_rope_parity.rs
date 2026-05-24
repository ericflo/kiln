//! Parity test: kt CUDA RoPE (`RopeOp::cuda_fwd` / `cuda_rope`) vs
//! kt CPU reference (`ops::rope::rope`).
//!
//! Phase 1 substrate validation for #1082. Confirms the kernel in
//! `csrc/rope.cu` produces the same per-position 2-D rotation as the
//! canonical CPU implementation, across F32 / BF16 / F16 dtypes,
//! several shapes (including partial-rotary), and matching cos / sin
//! dtypes against the input.

use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{cuda_rope, ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E3779B97F4A7C15);
        // Range roughly [-1, 1).
        let f = ((s as u32 % 2048) as f32 - 1024.0) / 1024.0;
        out.push(f);
    }
    out
}

fn read_cpu_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu.as_bytes();
    let dtype = t.dtype();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match dtype {
        kiln_tensor::DType::F32 => {
            for i in 0..n {
                out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
            }
        }
        kiln_tensor::DType::BF16 => {
            for i in 0..n {
                let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(v.to_f32());
            }
        }
        kiln_tensor::DType::F16 => {
            for i in 0..n {
                let v = half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(v.to_f32());
            }
        }
        other => panic!("read_cpu_f32: unsupported dtype {other}"),
    }
    out
}

/// Build a CPU kt-Tensor from f32 data, cast to `dtype` via candle then
/// copy back to f32 bytes for kt; OR simpler: build directly via
/// kiln_tensor::Tensor::from_slice for F32 and use candle's cast for
/// BF16/F16 then materialize a kt CPU tensor of that dtype.
fn make_cpu_tensor(data: &[f32], shape: Vec<usize>, dtype: kiln_tensor::DType) -> Tensor {
    match dtype {
        kiln_tensor::DType::F32 => Tensor::from_slice(data, shape).unwrap(),
        kiln_tensor::DType::BF16 => {
            let bytes: Vec<u8> = data
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect();
            let cpu = kiln_tensor::CpuStorage::from_bytes(dtype, bytes).unwrap();
            let storage: kiln_tensor::Storage = std::sync::Arc::new(cpu);
            Tensor::from_parts(
                storage,
                kiln_tensor::Layout::contiguous(shape),
                kiln_tensor::TensorId::next(),
            )
            .unwrap()
        }
        kiln_tensor::DType::F16 => {
            let bytes: Vec<u8> = data
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect();
            let cpu = kiln_tensor::CpuStorage::from_bytes(dtype, bytes).unwrap();
            let storage: kiln_tensor::Storage = std::sync::Arc::new(cpu);
            Tensor::from_parts(
                storage,
                kiln_tensor::Layout::contiguous(shape),
                kiln_tensor::TensorId::next(),
            )
            .unwrap()
        }
        other => panic!("make_cpu_tensor: unsupported dtype {other}"),
    }
}

/// Build a candle CUDA tensor from f32 data + target dtype. Used to
/// then borrow as a kt-Tensor for the CUDA path.
fn make_cuda_candle(
    dev: &CandleDevice,
    data: &[f32],
    shape: &[usize],
    dtype: CandleDType,
) -> CandleTensor {
    let shape_tuple: Vec<usize> = shape.to_vec();
    CandleTensor::from_vec(data.to_vec(), shape_tuple.as_slice(), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn run_rope_parity(
    shape: Vec<usize>,
    rotary_dim: usize,
    dtype_pair: (kiln_tensor::DType, CandleDType),
    tolerance: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let (kt_dtype, candle_dtype) = dtype_pair;
    assert!(shape.len() >= 2);
    let head_dim = *shape.last().unwrap();
    let seq = shape[shape.len() - 2];
    let leading: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);
    let n = leading * seq * head_dim;

    // Generate x data.
    let x_data = pattern(n, 17);

    // Generate cos / sin: rank-2 [seq, rotary_dim/2], random-ish values
    // but inside [-1, 1] so the rotation behaves like sin/cos.
    let pair_count = rotary_dim / 2;
    let cs_n = seq * pair_count;
    let raw_cs = pattern(cs_n * 2, 31);
    // Use sin/cos pairs derived from angles to keep them coherent.
    // angle[s, i] = raw_cs[s*pair_count + i] * pi
    let mut cos_data = Vec::with_capacity(cs_n);
    let mut sin_data = Vec::with_capacity(cs_n);
    for s_idx in 0..seq {
        for i in 0..pair_count {
            let angle = raw_cs[s_idx * pair_count + i] * std::f32::consts::PI;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }

    // --- CPU reference ---
    let x_cpu = make_cpu_tensor(&x_data, shape.clone(), kt_dtype);
    let cos_cpu = make_cpu_tensor(&cos_data, vec![seq, pair_count], kt_dtype);
    let sin_cpu = make_cpu_tensor(&sin_data, vec![seq, pair_count], kt_dtype);
    let y_cpu = ops::rope::rope(&x_cpu, &cos_cpu, &sin_cpu, rotary_dim).unwrap();
    let ref_vec = read_cpu_f32(&y_cpu);

    // --- CUDA path ---
    let x_cd = make_cuda_candle(&dev, &x_data, &shape, candle_dtype);
    let cos_cd = make_cuda_candle(&dev, &cos_data, &[seq, pair_count], candle_dtype);
    let sin_cd = make_cuda_candle(&dev, &sin_data, &[seq, pair_count], candle_dtype);

    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();
    let cos_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&cos_cd).unwrap();
    let sin_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&sin_cd).unwrap();

    let out_kt = cuda_rope(&x_kt, &cos_kt, &sin_kt, rotary_dim).expect("cuda_rope");
    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    // Copy CUDA result back to host as F32 for comparison.
    let got_cd = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap();
    let got_vec: Vec<f32> = got_cd.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert_eq!(ref_vec.len(), got_vec.len());
    let mut max_abs = 0.0f32;
    for (a, b) in ref_vec.iter().zip(got_vec.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tolerance,
        "shape={shape:?} rotary_dim={rotary_dim} dtype={kt_dtype:?} max_abs={max_abs} > {tolerance}"
    );
}

#[test]
fn cuda_rope_f32_2d_full_rotary() {
    // [seq=4, head_dim=4], rotary_dim=4 (full).
    run_rope_parity(
        vec![4, 4],
        4,
        (kiln_tensor::DType::F32, CandleDType::F32),
        1e-5,
    );
}

#[test]
fn cuda_rope_f32_3d_batched() {
    // [batch=2, seq=8, head_dim=16], rotary_dim=16 (full).
    run_rope_parity(
        vec![2, 8, 16],
        16,
        (kiln_tensor::DType::F32, CandleDType::F32),
        1e-5,
    );
}

#[test]
fn cuda_rope_f32_partial_rotary() {
    // [batch=2, seq=8, head_dim=16], rotary_dim=8 (partial — half of head_dim).
    run_rope_parity(
        vec![2, 8, 16],
        8,
        (kiln_tensor::DType::F32, CandleDType::F32),
        1e-5,
    );
}

#[test]
fn cuda_rope_f32_4d_qwen_shape() {
    // [batch=1, heads=4, seq=8, head_dim=16], rotary_dim=16.
    // Mirrors the (B, H, T, D) attention layout used at the call site.
    run_rope_parity(
        vec![1, 4, 8, 16],
        16,
        (kiln_tensor::DType::F32, CandleDType::F32),
        1e-5,
    );
}

#[test]
fn cuda_rope_bf16_3d() {
    run_rope_parity(
        vec![2, 8, 16],
        16,
        (kiln_tensor::DType::BF16, CandleDType::BF16),
        // BF16 has ~7 bits of mantissa; tolerance reflects that, plus
        // the kernel's --use_fast_math floating-point slack.
        2e-2,
    );
}

#[test]
fn cuda_rope_bf16_partial_rotary() {
    run_rope_parity(
        vec![2, 8, 16],
        8,
        (kiln_tensor::DType::BF16, CandleDType::BF16),
        2e-2,
    );
}

#[test]
fn cuda_rope_f16_3d() {
    run_rope_parity(
        vec![2, 8, 16],
        16,
        (kiln_tensor::DType::F16, CandleDType::F16),
        // F16: ~10 bits of mantissa.
        5e-3,
    );
}

#[test]
fn cuda_rope_f32_large_qwen_like() {
    // Closer-to-real Qwen3.5-4B shape: [B=1, H=16, T=32, D=128], rotary_dim=32.
    run_rope_parity(
        vec![1, 16, 32, 128],
        32,
        (kiln_tensor::DType::F32, CandleDType::F32),
        1e-5,
    );
}
