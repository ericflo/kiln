#![cfg(feature = "metal")]

//! Metal parity tests — the **first** on-hardware validation of the
//! kt-native Metal substrate (#1082).
//!
//! Every test A/Bs a kt Metal op against the kt CPU reference op on
//! identical inputs. This is the canonical-reference contract from the
//! epic's anti-pattern #6 ("after candle is removed, the CPU kt-tensor
//! path is the canonical numerical reference every other backend
//! compares against"), realized for Metal.
//!
//! Unlike the `cuda_*_parity.rs` suite these are **candle-free**: they
//! build Metal inputs via `Tensor::from_vec_on(Device::Metal(0), ..)`
//! and read Metal outputs back via `Tensor::to_vec` — the host-I/O
//! foundation this same PR adds. No `kiln_kt_bridge` dependency.
//!
//! Skips gracefully (prints + returns) when no Metal device is present,
//! so the suite is a no-op on CI runners without a GPU.

use kiln_tensor::{ops, DType, Device, Tensor};

/// `Device::Metal(0)` if a Metal device is reachable, else `None`.
fn metal() -> Option<Device> {
    // `primary_metal_companion` enumerates `Device::all()`; Ok ⇒ present.
    kiln_tensor::primary_metal_companion(0).ok().map(|_| Device::Metal(0))
}

/// Deterministic pseudo-random f32 pattern in roughly [-1, 1].
fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..n {
        s = s.wrapping_add(0xDEAD_BEEF).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        out.push(((s >> 33) as u32 % 2048) as f32 / 1024.0 - 1.0);
    }
    out
}

/// Strictly-positive f32 pattern (for weights / rmsnorm scale).
fn pattern_pos(n: usize, seed: u64) -> Vec<f32> {
    pattern(n, seed).into_iter().map(|v| v.abs() + 0.25).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Build the same data on CPU and Metal, returning both tensors.
fn pair(data: &[f32], shape: &[usize], dev: Device) -> (Tensor, Tensor) {
    let cpu = Tensor::from_vec(data.to_vec(), shape.to_vec()).unwrap();
    let met = Tensor::from_vec_on(dev, data.to_vec(), shape.to_vec()).unwrap();
    (cpu, met)
}

// ----------------------------------------------------------------------
// Foundation: host → Metal → host round-trips
// ----------------------------------------------------------------------

#[test]
fn roundtrip_f32_contiguous() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let data = pattern(4096, 1);
    let t = Tensor::from_vec_on(dev, data.clone(), vec![64, 64]).unwrap();
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.shape().to_vec(), vec![64, 64]);
    let got = t.to_vec::<f32>().unwrap();
    assert_eq!(got, data, "host→Metal→host must be bit-identical for F32");
}

#[test]
fn roundtrip_strided_view() {
    // A narrow()-then-read must gather only the addressed elements, not
    // the whole backing buffer — exercises the strided host-gather path.
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let data = pattern(48, 7);
    let t = Tensor::from_vec_on(dev, data.clone(), vec![6, 8]).unwrap();
    // rows [2..5) → logical [3,8]
    let view = t.narrow(0, 2, 3).unwrap();
    let got = view.to_vec::<f32>().unwrap();
    let want: Vec<f32> = data[16..40].to_vec();
    assert_eq!(got, want, "narrowed Metal view readback mismatch");
}

#[test]
fn roundtrip_bf16() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let data = pattern(256, 3);
    let cpu = Tensor::from_vec(data.clone(), vec![16, 16]).unwrap();
    let cpu_bf16 = ops::cast(&cpu, DType::BF16).unwrap();
    let want = cpu_bf16.to_vec::<half::bf16>().unwrap();
    // Round-trip the bf16 bytes through Metal.
    let met = cpu_bf16.to_device(dev).unwrap();
    assert_eq!(met.dtype(), DType::BF16);
    let got = met.to_vec::<half::bf16>().unwrap();
    assert_eq!(got, want, "host→Metal→host must be bit-identical for BF16");
}

// ----------------------------------------------------------------------
// Op parity: kt Metal vs kt CPU reference
// ----------------------------------------------------------------------

#[test]
fn softmax_last_axis_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    for (rows, cols, seed) in [(8usize, 128usize, 10u64), (1, 2560, 11), (32, 64, 12)] {
        let data = pattern(rows * cols, seed);
        let (cpu, met) = pair(&data, &[rows, cols], dev);
        let want = ops::softmax_last_dim(&cpu).unwrap().to_vec::<f32>().unwrap();
        let got = kiln_tensor::metal_softmax_last_axis(&met)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let d = max_abs_diff(&want, &got);
        assert!(d < 1e-5, "softmax f32 [{rows},{cols}] max|Δ|={d}");
    }
}

#[test]
fn rmsnorm_last_axis_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (rows, hidden) = (16usize, 256usize);
    let x = pattern(rows * hidden, 20);
    let w = pattern_pos(hidden, 21);
    let (cpu_x, met_x) = pair(&x, &[rows, hidden], dev);
    let (cpu_w, met_w) = pair(&w, &[hidden], dev);
    let eps = 1e-6f32;
    let want = ops::rms_norm(&cpu_x, &cpu_w, eps).unwrap().to_vec::<f32>().unwrap();
    let got = kiln_tensor::metal_rmsnorm_last_axis(&met_x, &met_w, eps)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let d = max_abs_diff(&want, &got);
    assert!(d < 1e-4, "rmsnorm f32 max|Δ|={d}");
}

#[test]
fn layernorm_last_axis_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (rows, hidden) = (16usize, 256usize);
    let x = pattern(rows * hidden, 30);
    let w = pattern_pos(hidden, 31);
    let b = pattern(hidden, 32);
    let (cpu_x, met_x) = pair(&x, &[rows, hidden], dev);
    let (cpu_w, met_w) = pair(&w, &[hidden], dev);
    let (cpu_b, met_b) = pair(&b, &[hidden], dev);
    let eps = 1e-5f32;
    let want = ops::layer_norm(&cpu_x, &cpu_w, &cpu_b, eps)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let got = kiln_tensor::metal_layernorm_last_axis(&met_x, &met_w, &met_b, eps)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let d = max_abs_diff(&want, &got);
    assert!(d < 1e-4, "layernorm f32 max|Δ|={d}");
}

#[test]
fn cast_f32_to_bf16() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let data = pattern(1024, 40);
    let (cpu, met) = pair(&data, &[1024], dev);
    let want = ops::cast(&cpu, DType::BF16)
        .unwrap()
        .to_vec::<half::bf16>()
        .unwrap();
    let got = kiln_tensor::metal_cast(&met, DType::BF16)
        .unwrap()
        .to_vec::<half::bf16>()
        .unwrap();
    assert_eq!(want, got, "cast f32→bf16 must match CPU bit-for-bit");
}

#[test]
fn elementwise_add_mul_sub_div_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let n = 1024usize;
    let a = pattern(n, 50);
    let b = pattern_pos(n, 51); // positive so div is well-behaved
    let (cpu_a, met_a) = pair(&a, &[n], dev);
    let (cpu_b, met_b) = pair(&b, &[n], dev);
    // 0=Add, 1=Sub, 2=Mul, 3=Div
    for (tag, name, cpu_ref) in [
        (0i32, "add", ops::add(&cpu_a, &cpu_b).unwrap()),
        (1, "sub", ops::sub(&cpu_a, &cpu_b).unwrap()),
        (2, "mul", ops::mul(&cpu_a, &cpu_b).unwrap()),
        (3, "div", ops::div(&cpu_a, &cpu_b).unwrap()),
    ] {
        let want = cpu_ref.to_vec::<f32>().unwrap();
        let got = kiln_tensor::metal_elementwise_binary(&met_a, &met_b, tag)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let d = max_abs_diff(&want, &got);
        assert!(d < 1e-5, "elementwise {name} f32 max|Δ|={d}");
    }
}

#[test]
fn activation_silu_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let data = pattern(2048, 60);
    let (cpu, met) = pair(&data, &[2048], dev);
    let want = ops::silu(&cpu).unwrap().to_vec::<f32>().unwrap();
    // kind_tag 0 = silu
    let got = kiln_tensor::metal_activation_unary(&met, 0)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let d = max_abs_diff(&want, &got);
    assert!(d < 1e-5, "silu f32 max|Δ|={d}");
}
