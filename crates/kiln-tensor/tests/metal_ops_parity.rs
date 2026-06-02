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

fn bf16_to_f32_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .unwrap()
        .to_vec::<half::bf16>()
        .unwrap()
        .into_iter()
        .map(half::bf16::to_f32)
        .collect()
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
fn log_softmax_last_axis_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    for (rows, cols, seed) in [(8usize, 128usize, 13u64), (1, 2560, 14), (32, 64, 15)] {
        let data = pattern(rows * cols, seed);
        let (cpu, met) = pair(&data, &[rows, cols], dev);
        // CPU reference (canonical) vs Metal kernel via the same op
        // entry point — ops::log_softmax_last_dim routes Metal storage
        // through the kiln-owned MSL kernel.
        let want = ops::log_softmax_last_dim(&cpu).unwrap().to_vec::<f32>().unwrap();
        let got = kiln_tensor::metal_log_softmax_last_axis(&met)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let d = max_abs_diff(&want, &got);
        assert!(d < 1e-5, "log_softmax f32 [{rows},{cols}] max|Δ|={d}");
        // Also exercise the dispatch path (ops::log_softmax_last_dim on
        // Metal storage must produce the same numbers).
        let got_dispatch = ops::log_softmax_last_dim(&met).unwrap().to_vec::<f32>().unwrap();
        let dd = max_abs_diff(&want, &got_dispatch);
        assert!(dd < 1e-5, "log_softmax f32 dispatch [{rows},{cols}] max|Δ|={dd}");
    }
}

#[test]
fn log_softmax_last_axis_bf16() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    for (rows, cols, seed) in [(8usize, 128usize, 16u64), (4, 256, 17)] {
        let data = pattern(rows * cols, seed);
        let cpu_f32 = Tensor::from_vec(data.clone(), vec![rows, cols]).unwrap();
        let met_f32 = Tensor::from_vec_on(dev, data, vec![rows, cols]).unwrap();
        let cpu = ops::cast(&cpu_f32, DType::BF16).unwrap();
        let met = ops::cast(&met_f32, DType::BF16).unwrap();
        let want = bf16_to_f32_vec(&ops::log_softmax_last_dim(&cpu).unwrap());
        let got_t = kiln_tensor::metal_log_softmax_last_axis(&met).unwrap();
        assert_eq!(got_t.dtype(), DType::BF16);
        let got = bf16_to_f32_vec(&got_t);
        let d = max_abs_diff(&want, &got);
        // BF16 round-trips through float exp/log; loosen the tolerance.
        assert!(d < 5e-2, "log_softmax bf16 [{rows},{cols}] max|Δ|={d}");
    }
}

#[test]
fn cumsum_axis_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // (rows, cols, axis, seed): trailing-axis scan, outer-axis scan, single row.
    for (rows, cols, axis, seed) in [
        (8usize, 128usize, 1usize, 21u64),
        (8, 128, 0, 22),
        (1, 2560, 1, 23),
        (64, 4, 0, 24),
    ] {
        let data = pattern(rows * cols, seed);
        let (cpu, met) = pair(&data, &[rows, cols], dev);
        // CPU reference vs the Metal kernel through the same op entry point
        // (`ops::cumsum` routes Metal storage to the kiln-owned MSL scan).
        let want = ops::cumsum(&cpu, axis).unwrap().to_vec::<f32>().unwrap();
        let got = ops::cumsum(&met, axis).unwrap().to_vec::<f32>().unwrap();
        let d = max_abs_diff(&want, &got);
        // Identical sequential F32 accumulation on both sides ⇒ ~bit-exact.
        assert!(d < 1e-3, "cumsum f32 [{rows},{cols}] axis {axis} max|Δ|={d}");
    }
}

#[test]
fn compare_all_kinds_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (rows, cols) = (8usize, 96usize);
    // `a` ranges over [-1,1]; `b` is a quantized copy so eq/ne hit both
    // branches and lt/le/gt/ge cover all orderings.
    let a = pattern(rows * cols, 31);
    let b: Vec<f32> = a
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 3 == 0 { v } else { (v * 4.0).round() / 4.0 })
        .collect();
    let (cpu_a, met_a) = pair(&a, &[rows, cols], dev);
    let (cpu_b, met_b) = pair(&b, &[rows, cols], dev);
    // Each closure routes Metal storage to the kiln-owned MSL compare kernel
    // (U8 mask output) through the same `ops::` entry point as the CPU side.
    type CmpFn = fn(&Tensor, &Tensor) -> kiln_tensor::Result<Tensor>;
    let kinds: [(&str, CmpFn); 6] = [
        ("eq", ops::eq),
        ("ne", ops::ne),
        ("lt", ops::lt),
        ("le", ops::le),
        ("gt", ops::gt),
        ("ge", ops::ge),
    ];
    for (name, f) in kinds {
        let want = f(&cpu_a, &cpu_b).unwrap();
        let got = f(&met_a, &met_b).unwrap();
        assert_eq!(got.dtype(), DType::U8, "compare {name}: output dtype");
        assert_eq!(got.shape(), &[rows, cols], "compare {name}: output shape");
        let want_u8 = want.to_vec::<u8>().unwrap();
        let got_u8 = got.to_vec::<u8>().unwrap();
        // Exact U8-mask equality — the boolean op is computed identically
        // (F32-promoted compare) on both sides.
        let mismatches = want_u8.iter().zip(&got_u8).filter(|(x, y)| x != y).count();
        assert_eq!(
            mismatches, 0,
            "compare {name} [{rows},{cols}] mask mismatch count={mismatches}"
        );
        // Sanity: the test data exercises both 0 and 1 outcomes.
        assert!(
            want_u8.iter().any(|&v| v == 1) && want_u8.iter().any(|&v| v == 0),
            "compare {name}: test data should hit both true and false"
        );
    }
}

#[test]
fn where_select_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (rows, cols) = (8usize, 96usize);
    let t = pattern(rows * cols, 41);
    let f = pattern(rows * cols, 42);
    // Deterministic U8 mask exercising both branches (some 0, some 1).
    let mask: Vec<u8> = (0..rows * cols).map(|i| if (i * 7 + 3) % 5 < 2 { 1 } else { 0 }).collect();
    let (cpu_t, met_t) = pair(&t, &[rows, cols], dev);
    let (cpu_f, met_f) = pair(&f, &[rows, cols], dev);
    let cpu_m = Tensor::from_vec(mask.clone(), vec![rows, cols]).unwrap();
    let met_m = Tensor::from_vec_on(dev, mask.clone(), vec![rows, cols]).unwrap();
    // CPU reference vs the Metal kernel through the same `ops::where_select`
    // entry point (Metal storage → kiln-owned MSL byte-wise select).
    let want = ops::where_select(&cpu_m, &cpu_t, &cpu_f).unwrap().to_vec::<f32>().unwrap();
    let got = ops::where_select(&met_m, &met_t, &met_f).unwrap().to_vec::<f32>().unwrap();
    let d = max_abs_diff(&want, &got);
    // Byte-wise select copies the chosen operand bit-for-bit ⇒ exact.
    assert!(d == 0.0, "where_select f32 [{rows},{cols}] max|Δ|={d}");
    // Sanity: the mask hit both branches.
    assert!(mask.iter().any(|&m| m == 1) && mask.iter().any(|&m| m == 0));
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
fn cast_u8_mask_to_float_triple_and_back() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // Boolean-style mask (the GDN triangular mask shape): some 0, some 1.
    let n = 1024usize;
    let mask: Vec<u8> = (0..n).map(|i| if (i * 7 + 3) % 5 < 2 { 1 } else { 0 }).collect();
    let cpu_u8 = Tensor::from_vec(mask.clone(), vec![n]).unwrap();
    let met_u8 = Tensor::from_vec_on(dev, mask.clone(), vec![n]).unwrap();
    assert_eq!(met_u8.dtype(), DType::U8);

    // U8 -> F32 / BF16 / F16: every backend value must equal the CPU cast.
    for to in [DType::F32, DType::BF16, DType::F16] {
        let want = ops::cast(&cpu_u8, to).unwrap();
        let got = kiln_tensor::metal_cast(&met_u8, to).unwrap();
        assert_eq!(got.dtype(), to, "cast u8→{to}: output dtype");
        let want_f = want.to_device(Device::Cpu).unwrap();
        let got_f = got.to_device(Device::Cpu).unwrap();
        let (w, g): (Vec<f32>, Vec<f32>) = match to {
            DType::F32 => (
                want_f.to_vec::<f32>().unwrap(),
                got_f.to_vec::<f32>().unwrap(),
            ),
            DType::BF16 => (
                want_f.to_vec::<half::bf16>().unwrap().into_iter().map(|v| v.to_f32()).collect(),
                got_f.to_vec::<half::bf16>().unwrap().into_iter().map(|v| v.to_f32()).collect(),
            ),
            DType::F16 => (
                want_f.to_vec::<half::f16>().unwrap().into_iter().map(|v| v.to_f32()).collect(),
                got_f.to_vec::<half::f16>().unwrap().into_iter().map(|v| v.to_f32()).collect(),
            ),
            _ => unreachable!(),
        };
        assert_eq!(w, g, "cast u8→{to} must match CPU bit-for-bit");
        // Mask semantics preserved: value 1.0 where mask==1, 0.0 elsewhere.
        for (i, &m) in mask.iter().enumerate() {
            assert_eq!(g[i], m as f32, "cast u8→{to} value at {i}");
        }
    }

    // F32 -> U8 round-trip: the {0,1} mask must come back exactly.
    let f32_mask = ops::cast(&cpu_u8, DType::F32).unwrap();
    let met_f32 = f32_mask.to_device(dev).unwrap();
    let want_back = ops::cast(&f32_mask, DType::U8).unwrap().to_vec::<u8>().unwrap();
    let got_back = kiln_tensor::metal_cast(&met_f32, DType::U8)
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap()
        .to_vec::<u8>()
        .unwrap();
    assert_eq!(want_back, mask, "f32→u8 must restore the mask");
    assert_eq!(got_back, want_back, "metal f32→u8 must match CPU");
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

/// `metal_sdpa_last_axis` parity vs the kt CPU SDPA reference — the op
/// `flash_attn_prefill` switched to when the candle `sdpa` symbol was
/// dropped (#1082). Validates both Metal SDPA kernels at their real
/// selection boundary (mirrors candle's `Sdpa::metal_fwd`):
///   - q_seq > 8 -> `call_sdpa_full`, which applies causal masking;
///   - q_seq <= 8 -> `call_sdpa_vector`, the MLX vector kernel (decode
///     shape), which does NOT causal-mask — same as the candle path, so
///     this is checked non-causal.
#[test]
fn sdpa_last_axis_f32() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (bs, heads, hd) = (1usize, 2usize, 64usize); // head_dim 64 is whitelisted

    // Full-kernel causal path (q_seq > 8): must match the CPU causal ref.
    {
        let seq = 16usize;
        let n = bs * heads * seq * hd;
        let shape = [bs, heads, seq, hd];
        let (cpu_q, met_q) = pair(&pattern(n, 70), &shape, dev);
        let (cpu_k, met_k) = pair(&pattern(n, 71), &shape, dev);
        let (cpu_v, met_v) = pair(&pattern(n, 72), &shape, dev);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let got = kiln_tensor::metal_sdpa_last_axis(&met_q, &met_k, &met_v, scale, true)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let want = ops::causal_scaled_dot_product_attention(&cpu_q, &cpu_k, &cpu_v)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let d = max_abs_diff(&want, &got);
        assert!(d < 1e-3, "full causal sdpa f32 (q_seq=16) max|Δ|={d}");
    }

    // Vector-kernel path at its design point — q_seq=1 decode (1 query
    // attending to k_seq keys), non-causal. This is the shape the MLX
    // vector kernel (call_sdpa_vector) is built for; it must match the
    // CPU non-causal reference. (candle uses the same kernel for
    // q_seq<=8; multi-query q_seq>1 is an MLX-vector quirk shared with
    // candle and not naive-comparable — see the full-kernel block above
    // for the q_seq>8 prefill path that DOES causal-mask.)
    {
        let (q_seq, k_seq) = (1usize, 8usize);
        let qn = bs * heads * q_seq * hd;
        let kn = bs * heads * k_seq * hd;
        let cpu_q = Tensor::from_vec(pattern(qn, 73), vec![bs, heads, q_seq, hd]).unwrap();
        let met_q = Tensor::from_vec_on(dev, pattern(qn, 73), vec![bs, heads, q_seq, hd]).unwrap();
        let cpu_k = Tensor::from_vec(pattern(kn, 74), vec![bs, heads, k_seq, hd]).unwrap();
        let met_k = Tensor::from_vec_on(dev, pattern(kn, 74), vec![bs, heads, k_seq, hd]).unwrap();
        let cpu_v = Tensor::from_vec(pattern(kn, 75), vec![bs, heads, k_seq, hd]).unwrap();
        let met_v = Tensor::from_vec_on(dev, pattern(kn, 75), vec![bs, heads, k_seq, hd]).unwrap();
        let scale = 1.0f32 / (hd as f32).sqrt();
        let got = kiln_tensor::metal_sdpa_last_axis(&met_q, &met_k, &met_v, scale, false)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let want = ops::scaled_dot_product_attention(&cpu_q, &cpu_k, &cpu_v)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let d = max_abs_diff(&want, &got);
        assert!(d < 1e-3, "vector sdpa f32 (q_seq=1 decode, non-causal) max|Δ|={d}");
    }
}

/// Matrix-core **tiled flash-attention prefill** parity (#1082). At
/// `q_seq >= 16` `metal_sdpa_last_axis` routes to the simdgroup_matrix
/// (`steel`) kernel; this gates that kernel vs the kt CPU SDPA reference
/// across the prefill design space: q_seq in {16,32,64}, causal AND
/// non-causal, GQA (Hq>Hkv) AND MHA, and head_dim in {64,128}. GQA is
/// referenced by expanding K/V kv-heads to Hq with `repeat_interleave`
/// (query head h uses kv head h/(Hq/Hkv)), which exactly matches the
/// kernel's `kv_h = h/gqa` map. tol max|Δ| < 2e-3.
#[test]
fn sdpa_steel_prefill_parity() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // (label, bs, hq, hkv, q_seq, k_seq, head_dim, causal)
    let cases: &[(&str, usize, usize, usize, usize, usize, usize, bool)] = &[
        ("q16 causal MHA D64", 1, 2, 2, 16, 16, 64, true),
        ("q16 noncausal MHA D64", 1, 2, 2, 16, 24, 64, false),
        ("q32 causal MHA D64", 1, 2, 2, 32, 32, 64, true),
        ("q32 noncausal MHA D128", 1, 2, 2, 32, 40, 128, false),
        ("q64 causal MHA D128", 1, 4, 4, 64, 64, 128, true),
        ("q16 causal GQA D64", 1, 4, 2, 16, 16, 64, true),
        ("q32 noncausal GQA D64", 2, 4, 2, 32, 48, 64, false),
        ("q64 causal GQA D128", 1, 8, 2, 64, 64, 128, true),
        ("q32 causal GQA D128 b2", 2, 4, 2, 32, 32, 128, true),
        ("q48 noncausal GQA D128", 1, 4, 2, 48, 96, 128, false),
    ];
    for &(label, bs, hq, hkv, q_seq, k_seq, hd, causal) in cases {
        let gqa = hq / hkv;
        let qn = bs * hq * q_seq * hd;
        let kn = bs * hkv * k_seq * hd;
        let cpu_q = Tensor::from_vec(pattern(qn, 80), vec![bs, hq, q_seq, hd]).unwrap();
        let met_q = Tensor::from_vec_on(dev, pattern(qn, 80), vec![bs, hq, q_seq, hd]).unwrap();
        let cpu_k = Tensor::from_vec(pattern(kn, 81), vec![bs, hkv, k_seq, hd]).unwrap();
        let met_k = Tensor::from_vec_on(dev, pattern(kn, 81), vec![bs, hkv, k_seq, hd]).unwrap();
        let cpu_v = Tensor::from_vec(pattern(kn, 82), vec![bs, hkv, k_seq, hd]).unwrap();
        let met_v = Tensor::from_vec_on(dev, pattern(kn, 82), vec![bs, hkv, k_seq, hd]).unwrap();

        let scale = 1.0f32 / (hd as f32).sqrt();
        let got = kiln_tensor::metal_sdpa_last_axis(&met_q, &met_k, &met_v, scale, causal)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        // CPU reference: expand kv-heads to Hq for GQA, then plain (causal)
        // SDPA. (Non-causal CPU SDPA does NOT require q_seq==k_seq; causal
        // does, so causal cases use q_seq==k_seq above.)
        let (ck, cv) = if gqa > 1 {
            (
                ops::repeat_interleave(&cpu_k, 1, gqa).unwrap(),
                ops::repeat_interleave(&cpu_v, 1, gqa).unwrap(),
            )
        } else {
            (cpu_k, cpu_v)
        };
        let want = if causal {
            ops::causal_scaled_dot_product_attention(&cpu_q, &ck, &cv).unwrap()
        } else {
            ops::scaled_dot_product_attention(&cpu_q, &ck, &cv).unwrap()
        }
        .to_vec::<f32>()
        .unwrap();

        let d = max_abs_diff(&want, &got);
        assert!(d < 2e-3, "steel sdpa [{label}] max|Δ|={d}");
    }
}

/// Qwen3.5-4B production prefill shape that previously compiled the
/// oversized `BF16 D=256 BQ=32 BK=16` Steel variant. This test must enter
/// the matrix-core prefill path and compile/run a BF16 D=256 GQA kernel on
/// real Metal hardware.
#[test]
fn sdpa_steel_prefill_qwen_bf16_d256_parity() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (bs, hq, hkv, q_seq, k_seq, hd) = (1usize, 16usize, 4usize, 32usize, 32usize, 256usize);
    let gqa = hq / hkv;
    let qn = bs * hq * q_seq * hd;
    let kn = bs * hkv * k_seq * hd;
    let q_shape = [bs, hq, q_seq, hd];
    let kv_shape = [bs, hkv, k_seq, hd];

    let (cpu_q_f32, met_q_f32) = pair(&pattern(qn, 83), &q_shape, dev);
    let (cpu_k_f32, met_k_f32) = pair(&pattern(kn, 84), &kv_shape, dev);
    let (cpu_v_f32, met_v_f32) = pair(&pattern(kn, 85), &kv_shape, dev);
    let cpu_q = ops::cast(&cpu_q_f32, DType::BF16).unwrap();
    let cpu_k = ops::cast(&cpu_k_f32, DType::BF16).unwrap();
    let cpu_v = ops::cast(&cpu_v_f32, DType::BF16).unwrap();
    let met_q = ops::cast(&met_q_f32, DType::BF16).unwrap();
    let met_k = ops::cast(&met_k_f32, DType::BF16).unwrap();
    let met_v = ops::cast(&met_v_f32, DType::BF16).unwrap();

    let scale = 1.0f32 / (hd as f32).sqrt();
    let got_t = kiln_tensor::metal_sdpa_last_axis(&met_q, &met_k, &met_v, scale, true).unwrap();
    assert_eq!(got_t.dtype(), DType::BF16);
    let got = bf16_to_f32_vec(&got_t);

    let ck = ops::repeat_interleave(&cpu_k, 1, gqa).unwrap();
    let cv = ops::repeat_interleave(&cpu_v, 1, gqa).unwrap();
    let want_t = ops::causal_scaled_dot_product_attention(&cpu_q, &ck, &cv).unwrap();
    assert_eq!(want_t.dtype(), DType::BF16);
    let want = bf16_to_f32_vec(&want_t);

    let d = max_abs_diff(&want, &got);
    assert!(d < 3e-2, "steel sdpa [Qwen BF16 D256 GQA] max|Δ|={d}");
}

/// **GQA-shared-KV decode** parity (#1082). At `q_seq < 16` (decode)
/// `metal_sdpa_last_axis` routes to the simd_sum/split-K kernel, which the
/// GQA-shared-KV rewrite restructured: ONE threadgroup handles a whole GQA
/// group (the `GF = Hq/Hkv` q-heads that share a kv-head), loading each
/// K[kj]/V[kj] once and reusing it for all GF heads (GF independent
/// online-softmax states + GF independent split-K combines). This gates the
/// GF>1 sharing + combine vs the kt CPU SDPA reference across decode shapes:
/// q_seq=1, GF in {4} (Hq=8/Hkv=2 and Hq=32/Hkv=8), Sk in {8,64,512}, D in
/// {64,128}, non-causal AND causal. GQA is referenced by expanding kv-heads
/// to Hq with `repeat_interleave` (query head h uses kv head h/GF), matching
/// the kernel's group map. tol max|Δ| < 2e-3.
#[test]
fn sdpa_gqa_decode_parity() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // (label, bs, hq, hkv, k_seq, head_dim, causal) — q_seq is always 1 (decode).
    let cases: &[(&str, usize, usize, usize, usize, usize, bool)] = &[
        ("GF4 Hq8 Sk8 D64 nc", 1, 8, 2, 8, 64, false),
        ("GF4 Hq8 Sk8 D64 causal", 1, 8, 2, 8, 64, true),
        ("GF4 Hq8 Sk64 D64 nc", 1, 8, 2, 64, 64, false),
        ("GF4 Hq8 Sk64 D128 causal", 1, 8, 2, 64, 128, true),
        ("GF4 Hq8 Sk512 D128 nc", 1, 8, 2, 512, 128, false),
        ("GF4 Hq8 Sk512 D128 causal", 1, 8, 2, 512, 128, true),
        ("GF4 Hq32 Sk8 D128 nc", 1, 32, 8, 8, 128, false),
        ("GF4 Hq32 Sk64 D64 causal", 1, 32, 8, 64, 64, true),
        ("GF4 Hq32 Sk512 D128 nc", 1, 32, 8, 512, 128, false),
        ("GF4 Hq32 Sk512 D128 causal", 1, 32, 8, 512, 128, true),
        // batch > 1 to exercise the b dimension of the new grid.
        ("GF4 Hq8 Sk512 D128 b2 nc", 2, 8, 2, 512, 128, false),
    ];
    for &(label, bs, hq, hkv, k_seq, hd, causal) in cases {
        let gf = hq / hkv;
        let q_seq = 1usize;
        let qn = bs * hq * q_seq * hd;
        let kn = bs * hkv * k_seq * hd;
        let cpu_q = Tensor::from_vec(pattern(qn, 90), vec![bs, hq, q_seq, hd]).unwrap();
        let met_q = Tensor::from_vec_on(dev, pattern(qn, 90), vec![bs, hq, q_seq, hd]).unwrap();
        let cpu_k = Tensor::from_vec(pattern(kn, 91), vec![bs, hkv, k_seq, hd]).unwrap();
        let met_k = Tensor::from_vec_on(dev, pattern(kn, 91), vec![bs, hkv, k_seq, hd]).unwrap();
        let cpu_v = Tensor::from_vec(pattern(kn, 92), vec![bs, hkv, k_seq, hd]).unwrap();
        let met_v = Tensor::from_vec_on(dev, pattern(kn, 92), vec![bs, hkv, k_seq, hd]).unwrap();

        let scale = 1.0f32 / (hd as f32).sqrt();
        let got = kiln_tensor::metal_sdpa_last_axis(&met_q, &met_k, &met_v, scale, causal)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        // CPU reference: expand kv-heads to Hq for GQA, then non-causal SDPA.
        // For a SINGLE decode query (q_seq=1) the causal limit is
        //   key_limit = (k_seq + qi + 1) - q_seq = k_seq,
        // i.e. the one query attends ALL keys — so the causal kernel result
        // equals the non-causal one (and the CPU causal helper, which
        // requires q_seq==k_seq, can't be used here anyway). We exercise the
        // kernel's `causal` branch but reference against the non-causal CPU
        // SDPA in both cases.
        let ck = ops::repeat_interleave(&cpu_k, 1, gf).unwrap();
        let cv = ops::repeat_interleave(&cpu_v, 1, gf).unwrap();
        let want = ops::scaled_dot_product_attention(&cpu_q, &ck, &cv)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        let d = max_abs_diff(&want, &got);
        assert!(d < 2e-3, "gqa-decode sdpa [{label}] (causal={causal}) max|Δ|={d}");
    }
}

/// kiln matrix-core GEMM (`metal_matmul` via `MatmulOp::metal_fwd`) vs the kt
/// CPU matmul reference, at Qwen3.5-4B shapes + M-tail/edge cases (#1082).
/// This is the on-M1 correctness gate for the simdgroup_float8x8 GEMM that
/// replaced the prefill/bs>1 CPU host-fallback. Running it also compiles the
/// kiln_gemm_bf16 MSL on the real Metal compiler.
#[test]
fn matmul_matrix_core_f32_parity() {
    let Some(dev) = metal() else { eprintln!("no Metal device; skipping"); return; };
    // (M, K, N): Qwen projections/MLP + M-tail (not mult of tile=64) + M=1.
    let cases = [
        (16usize, 256usize, 256usize),   // small, clean
        (17, 2560, 256),                 // M-tail (17 % 64 != 0), Qwen K
        (64, 2560, 4096),                // prefill QKV-ish
        (8, 2560, 18432),                // gate||up wide-N
        (1, 2560, 4096),                 // M=1 through the GEMM
        (100, 512, 2560),                // M not mult of 64, down-proj-ish
        (16, 256, 100),                  // N-tail (100 % 64 != 0): b_full=false, store_safe
        (33, 2560, 130),                 // M-tail + N-tail together
        (64, 100, 128),                  // K=100 % 16 != 0: naive-kernel fallback path
        (40, 130, 96),                   // K-tail + N-tail through the fallback
    ];
    for (m, k, n) in cases {
        let a = pattern(m * k, 100 + m as u64);
        let b = pattern(k * n, 200 + n as u64);
        // BF16 on both CPU and Metal from identical data.
        let a_cpu = ops::cast(&Tensor::from_vec(a.clone(), vec![m, k]).unwrap(), DType::BF16).unwrap();
        let b_cpu = ops::cast(&Tensor::from_vec(b.clone(), vec![k, n]).unwrap(), DType::BF16).unwrap();
        let a_met = ops::cast(&Tensor::from_vec_on(dev, a, vec![m, k]).unwrap(), DType::BF16).unwrap();
        let b_met = ops::cast(&Tensor::from_vec_on(dev, b, vec![k, n]).unwrap(), DType::BF16).unwrap();

        // CPU reference (matmul cpu_fwd) and Metal (matmul metal_fwd -> kiln GEMM).
        let want_t = ops::matmul(&a_cpu, &b_cpu).unwrap();
        let got_t = ops::matmul(&a_met, &b_met).unwrap();
        assert_eq!(got_t.device(), dev, "metal matmul must stay on Metal");
        assert_eq!(got_t.shape().to_vec(), vec![m, n], "output shape");

        let want: Vec<f32> = ops::cast(&want_t, DType::F32).unwrap().to_vec::<f32>().unwrap();
        let got: Vec<f32> = ops::cast(&got_t, DType::F32).unwrap().to_vec::<f32>().unwrap();
        let (d, mref) = {
            let mut md = 0.0f32; let mut mr = 0.0f32;
            for (g, w) in got.iter().zip(&want) { md = md.max((g - w).abs()); mr = mr.max(w.abs()); }
            (md, mr)
        };
        // BF16 inputs + F32 accumulate over K; bound relative to result magnitude.
        assert!(d < 0.02 * mref.max(1.0), "matmul [{m},{k}]x[{k},{n}] max|Δ|={d} (ref max {mref})");
    }
}

/// Batched matmul parity — exercises the kiln GEMM's `tid.z` per-batch
/// dispatch + per-batch element strides (a_bs/b_bs/c_bs), which the 2D
/// cases above never touch. `[B,M,K] @ [B,K,N] -> [B,M,N]`.
#[test]
fn matmul_matrix_core_batched_parity() {
    let Some(dev) = metal() else { eprintln!("no Metal device; skipping"); return; };
    // (B, M, K, N): full tiles, M/N-tail, and a 3-leading-dim batch.
    let cases = [
        (vec![4usize], 64usize, 256usize, 128usize), // clean batch
        (vec![3], 17, 512, 100),                      // batched + M-tail + N-tail
        (vec![2, 2], 8, 256, 64),                     // 2 leading batch dims
    ];
    for (bdims, m, k, n) in cases {
        let batch: usize = bdims.iter().product();
        let a = pattern(batch * m * k, 300 + m as u64);
        let b = pattern(batch * k * n, 400 + n as u64);
        let mut a_shape = bdims.clone();
        a_shape.extend_from_slice(&[m, k]);
        let mut b_shape = bdims.clone();
        b_shape.extend_from_slice(&[k, n]);

        let a_cpu = ops::cast(&Tensor::from_vec(a.clone(), a_shape.clone()).unwrap(), DType::BF16).unwrap();
        let b_cpu = ops::cast(&Tensor::from_vec(b.clone(), b_shape.clone()).unwrap(), DType::BF16).unwrap();
        let a_met = ops::cast(&Tensor::from_vec_on(dev, a, a_shape).unwrap(), DType::BF16).unwrap();
        let b_met = ops::cast(&Tensor::from_vec_on(dev, b, b_shape).unwrap(), DType::BF16).unwrap();

        let want_t = ops::matmul(&a_cpu, &b_cpu).unwrap();
        let got_t = ops::matmul(&a_met, &b_met).unwrap();
        assert_eq!(got_t.device(), dev, "metal batched matmul must stay on Metal");
        let mut out_shape = bdims.clone();
        out_shape.extend_from_slice(&[m, n]);
        assert_eq!(got_t.shape().to_vec(), out_shape, "batched output shape");

        let want: Vec<f32> = ops::cast(&want_t, DType::F32).unwrap().to_vec::<f32>().unwrap();
        let got: Vec<f32> = ops::cast(&got_t, DType::F32).unwrap().to_vec::<f32>().unwrap();
        let d = max_abs_diff(&got, &want);
        let mref = want.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
        assert!(d < 0.02 * mref.max(1.0), "batched matmul {bdims:?} [{m},{k}]x[{k},{n}] max|Δ|={d} (ref {mref})");
    }
}

/// index_select dim0 parity — gates the kiln-owned MSL gather kernel
/// (`metal_kernels::index_select_dim0`, replacing candle's
/// `call_index_select`). Gather is an exact copy → demand bit-exact equality.
#[test]
fn index_select_dim0_parity() {
    let Some(dev) = metal() else { eprintln!("no Metal device; skipping"); return; };
    // (vocab, hidden, ids): repeats, last-row, row_len=1, multi-row.
    let cases: [(usize, usize, Vec<u32>); 4] = [
        (32, 64, vec![0, 5, 31, 5, 12, 0]),  // repeats + last valid row
        (10, 1, vec![3, 3, 9, 0]),           // row_len == 1
        (100, 128, vec![99, 0, 50, 50, 1]),  // embedding-ish wide row
        (152064, 8, vec![151000, 0, 42]),    // Qwen vocab scale
    ];
    for (vocab, hidden, ids) in cases {
        let w = pattern(vocab * hidden, 700 + vocab as u64);
        let w_cpu = Tensor::from_vec(w.clone(), vec![vocab, hidden]).unwrap();
        let w_met = Tensor::from_vec_on(dev, w, vec![vocab, hidden]).unwrap();
        let ids_cpu = Tensor::from_vec(ids.clone(), vec![ids.len()]).unwrap();
        let ids_met = Tensor::from_vec_on(dev, ids.clone(), vec![ids.len()]).unwrap();

        let want = ops::index_select(&w_cpu, 0, &ids_cpu).unwrap();
        let got = ops::index_select(&w_met, 0, &ids_met).unwrap();
        assert_eq!(got.device(), dev, "index_select must stay on Metal");
        assert_eq!(got.shape().to_vec(), vec![ids.len(), hidden], "output shape");

        let want_v: Vec<f32> = want.to_vec::<f32>().unwrap();
        let got_v: Vec<f32> = got.to_vec::<f32>().unwrap();
        // Exact copy — no rounding; bit-for-bit equality.
        assert_eq!(got_v, want_v, "index_select [{vocab},{hidden}] ids={ids:?}");
    }
}
