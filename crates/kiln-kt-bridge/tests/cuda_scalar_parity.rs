//! Parity test: kt CUDA `add_scalar` / `sub_scalar` / `mul_scalar` /
//! `div_scalar` vs kt CPU reference.
//!
//! Issue #1082 — sub-phase: ScalarOp cuda_fwd wiring.
//! Confirms the kernel in `crates/kiln-tensor/csrc/scalar_op.cu`
//! produces output that matches the canonical CPU per-element loop in
//! `crates/kiln-tensor/src/ops/scalar.rs`.

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

#[derive(Debug, Clone, Copy)]
enum ScalarOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

fn run_kt_cpu(kind: ScalarOpKind, data: &[f32], shape: Vec<usize>, c: f32) -> Vec<f32> {
    let x = Tensor::from_slice(data, shape).unwrap();
    let y = match kind {
        ScalarOpKind::Add => ops::add_scalar(&x, c).unwrap(),
        ScalarOpKind::Sub => ops::sub_scalar(&x, c).unwrap(),
        ScalarOpKind::Mul => ops::mul_scalar(&x, c).unwrap(),
        ScalarOpKind::Div => ops::div_scalar(&x, c).unwrap(),
    };
    let cpu = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu.as_bytes();
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

fn run_scalar_parity(
    kind: ScalarOpKind,
    shape: Vec<usize>,
    dtype: CandleDType,
    c: f32,
    tolerance: f32,
) {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };
    let n: usize = shape.iter().product();
    let data = pattern(n, 41);

    // Build candle CUDA tensor with the target dtype, then borrow as kt.
    let x_cd = match shape.as_slice() {
        [a] => CandleTensor::from_vec(data.clone(), (*a,), &dev).unwrap(),
        [a, b] => CandleTensor::from_vec(data.clone(), (*a, *b), &dev).unwrap(),
        [a, b, c2] => {
            CandleTensor::from_vec(data.clone(), (*a, *b, *c2), &dev).unwrap()
        }
        other => panic!("unsupported test shape {other:?}"),
    };
    let x_cd = x_cd.to_dtype(dtype).unwrap();
    let x_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_cd).unwrap();

    let out_kt = match kind {
        ScalarOpKind::Add => ops::add_scalar(&x_kt, c).expect("add_scalar dispatch"),
        ScalarOpKind::Sub => ops::sub_scalar(&x_kt, c).expect("sub_scalar dispatch"),
        ScalarOpKind::Mul => ops::mul_scalar(&x_kt, c).expect("mul_scalar dispatch"),
        ScalarOpKind::Div => ops::div_scalar(&x_kt, c).expect("div_scalar dispatch"),
    };
    assert_eq!(out_kt.shape(), shape.as_slice());

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got_vec: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((n,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // CPU reference: cast the input through the same dtype lossy path
    // first so BF16/F16 reference values reflect the storage precision
    // before applying the scalar op.
    let cast_data: Vec<f32> = match dtype {
        CandleDType::F32 => data.clone(),
        CandleDType::BF16 => data
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect(),
        CandleDType::F16 => data
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect(),
        _ => panic!("unsupported dtype {dtype:?}"),
    };
    let ref_vec = run_kt_cpu(kind, &cast_data, shape.clone(), c);

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
        "kind={kind:?} shape={shape:?} dtype={dtype:?} c={c} max_abs={max_abs} > {tolerance}"
    );
}

// ----------------------------------------------------------------------
// F32 parity — tight tolerance.
// ----------------------------------------------------------------------

#[test]
fn cuda_add_scalar_f32_small() {
    run_scalar_parity(ScalarOpKind::Add, vec![1024], CandleDType::F32, 0.5, 1e-6);
}

#[test]
fn cuda_sub_scalar_f32_2d() {
    run_scalar_parity(ScalarOpKind::Sub, vec![16, 64], CandleDType::F32, -0.25, 1e-6);
}

#[test]
fn cuda_mul_scalar_f32_3d() {
    run_scalar_parity(ScalarOpKind::Mul, vec![4, 8, 32], CandleDType::F32, 2.5, 1e-6);
}

#[test]
fn cuda_div_scalar_f32_negative() {
    run_scalar_parity(ScalarOpKind::Div, vec![512], CandleDType::F32, -1.5, 1e-6);
}

// ----------------------------------------------------------------------
// BF16 parity — looser tolerance reflects 8-bit mantissa.
// ----------------------------------------------------------------------

#[test]
fn cuda_add_scalar_bf16() {
    run_scalar_parity(ScalarOpKind::Add, vec![8, 128], CandleDType::BF16, 0.5, 1e-2);
}

#[test]
fn cuda_sub_scalar_bf16() {
    run_scalar_parity(ScalarOpKind::Sub, vec![512], CandleDType::BF16, 0.125, 1e-2);
}

#[test]
fn cuda_mul_scalar_bf16() {
    run_scalar_parity(ScalarOpKind::Mul, vec![4, 64], CandleDType::BF16, 0.5, 1e-2);
}

#[test]
fn cuda_div_scalar_bf16() {
    run_scalar_parity(ScalarOpKind::Div, vec![2, 32, 32], CandleDType::BF16, 2.0, 1e-2);
}

// ----------------------------------------------------------------------
// F16 parity.
// ----------------------------------------------------------------------

#[test]
fn cuda_mul_scalar_f16() {
    run_scalar_parity(ScalarOpKind::Mul, vec![8, 128], CandleDType::F16, 0.25, 1e-2);
}

#[test]
fn cuda_add_scalar_f16() {
    run_scalar_parity(ScalarOpKind::Add, vec![1024], CandleDType::F16, 1.0, 1e-2);
}

// ----------------------------------------------------------------------
// Empty + large + boundary shapes.
// ----------------------------------------------------------------------

#[test]
fn cuda_mul_scalar_large_f32() {
    // Just over one block to exercise the multi-block grid path.
    run_scalar_parity(ScalarOpKind::Mul, vec![257 * 1024], CandleDType::F32, 0.5, 1e-6);
}

#[test]
fn cuda_add_scalar_zero_scalar() {
    // c=0 → identity for Add. Confirms FP zero is handled.
    run_scalar_parity(ScalarOpKind::Add, vec![32, 32], CandleDType::F32, 0.0, 1e-7);
}

#[test]
fn cuda_mul_scalar_one_scalar() {
    // c=1 → identity for Mul.
    run_scalar_parity(ScalarOpKind::Mul, vec![64, 16], CandleDType::F32, 1.0, 1e-7);
}
