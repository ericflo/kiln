//! Phase R.6 — hipBLASLt dense GEMM parity vs a CPU reference, on a real AMD
//! GPU. Covers F32 + BF16, decode-skinny (M=1), square, tall, and batched
//! shapes, plus the fused-bias epilogue. Normal developer runs skip when no
//! ROCm device is present; `KILN_QUALIFICATION=1` makes that a test failure.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_matmul_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{DType, Device, Tensor};

fn qualification_required(value: Option<&str>) -> bool {
    value == Some("1")
}

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        if qualification_required(std::env::var("KILN_QUALIFICATION").ok().as_deref()) {
            panic!("ROCm device unavailable while KILN_QUALIFICATION=1");
        }
        eprintln!("no ROCm device available; skipping R.6 matmul parity test");
        true
    } else {
        false
    }
}

/// Reference row-major matmul C[m,n] = sum_k A[m,k] * B[k,n], f32 accumulate.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Reference row-major matmul C[m,n] = A[k,m]^T * B[k,n], f32 accumulate.
fn cpu_lhs_t_matmul(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[p * m + i] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Reference row-major matmul C[m,n] = A[m,k] * B[n,k]^T, f32 accumulate.
fn cpu_rhs_t_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn val(i: usize, scale: f32) -> f32 {
    (((i * 37 + 11) % 97) as f32 / 97.0 - 0.5) * scale
}

fn check_close(got: &[f32], want: &[f32], rtol: f32, atol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff <= atol + rtol * w.abs(),
            "{label}: idx {i} got {g} want {w} diff {diff}"
        );
    }
}

fn cpu_matmul_cell(a: &[f32], b: &[f32], k: usize, n: usize, row: usize, col: usize) -> f32 {
    let mut acc = 0.0f32;
    for p in 0..k {
        acc += a[row * k + p] * b[p * n + col];
    }
    acc
}

#[test]
fn matmul_f32_shapes() {
    if no_rocm() {
        return;
    }
    // (m, k, n) — decode-skinny M=1, square, tall, wide-K.
    for &(m, k, n) in &[
        (1, 64, 64),
        (1, 2560, 4096),
        (16, 16, 16),
        (32, 128, 64),
        (128, 256, 512),
        (7, 65, 33),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| val(i, 1.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| val(i + 5, 1.0)).collect();
        let want = cpu_matmul(&a, &b, m, k, n);

        let ta = Tensor::from_vec_on(Device::Rocm(0), a, vec![m, k]).expect("a");
        let tb = Tensor::from_vec_on(Device::Rocm(0), b, vec![k, n]).expect("b");
        let tc = kiln_tensor::rocm_matmul(&ta, &tb)
            .unwrap_or_else(|e| panic!("matmul {m}x{k}x{n}: {e}"));
        let got = kiln_tensor::rocm_to_host_copy(&tc)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        check_close(&got, &want, 1e-4, 1e-4, &format!("f32 {m}x{k}x{n}"));
    }
}

#[test]
fn matmul_flce_active_rows_chunk_shape_f32() {
    if no_rocm() {
        return;
    }

    let (m, k, n) = (1_014usize, 2_560usize, 4_096usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| val(i.wrapping_mul(17).wrapping_add(3), 0.25))
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| val(i.wrapping_mul(31).wrapping_add(11), 0.20))
        .collect();

    let ta = Tensor::from_vec_on(Device::Rocm(0), a.clone(), vec![m, k]).expect("a");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b.clone(), vec![k, n]).expect("b");
    let tc = kiln_tensor::ops::matmul(&ta, &tb)
        .unwrap_or_else(|e| panic!("FLCE-shaped f32 matmul {m}x{k}x{n}: {e}"));
    assert_eq!(tc.shape(), &[m, n]);
    assert_eq!(tc.dtype(), DType::F32);
    let got = kiln_tensor::rocm_to_host_copy(&tc)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        got.iter().all(|v| v.is_finite()),
        "FLCE-shaped f32 matmul produced a non-finite output"
    );

    for &(row, col) in &[
        (0usize, 0usize),
        (1, 17),
        (104, 0),
        (104, 2_047),
        (104, 4_095),
        (513, 1_001),
        (1_013, 4_095),
    ] {
        let want = cpu_matmul_cell(&a, &b, k, n, row, col);
        let got = got[row * n + col];
        let diff = (got - want).abs();
        let tol = 5e-3 + 2e-4 * want.abs();
        assert!(
            diff <= tol,
            "FLCE-shaped f32 matmul mismatch row={row} col={col}: got {got} want {want} diff {diff} tol {tol}"
        );
    }
}

#[test]
fn matmul_bf16_shapes() {
    if no_rocm() {
        return;
    }
    use half::bf16;
    for &(m, k, n) in &[(1, 2560, 4096), (32, 256, 256), (64, 128, 512)] {
        let a_f: Vec<f32> = (0..m * k).map(|i| val(i, 1.0)).collect();
        let b_f: Vec<f32> = (0..k * n).map(|i| val(i + 5, 1.0)).collect();
        // Reference computed from the bf16-rounded inputs (matches device precision).
        let a_bf: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
        let b_bf: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();
        let a_r: Vec<f32> = a_bf.iter().map(|x| x.to_f32()).collect();
        let b_r: Vec<f32> = b_bf.iter().map(|x| x.to_f32()).collect();
        let want = cpu_matmul(&a_r, &b_r, m, k, n);

        let ta = Tensor::from_vec_on(Device::Rocm(0), a_bf, vec![m, k]).expect("a");
        let tb = Tensor::from_vec_on(Device::Rocm(0), b_bf, vec![k, n]).expect("b");
        let tc = kiln_tensor::rocm_matmul(&ta, &tb)
            .unwrap_or_else(|e| panic!("bf16 matmul {m}x{k}x{n}: {e}"));
        let got_bf = kiln_tensor::rocm_to_host_copy(&tc)
            .unwrap()
            .to_vec::<bf16>()
            .unwrap();
        let got: Vec<f32> = got_bf.iter().map(|x| x.to_f32()).collect();
        // bf16 GEMM tolerance scales with K (accumulation rounding).
        check_close(
            &got,
            &want,
            3e-2,
            (k as f32) * 1e-3,
            &format!("bf16 {m}x{k}x{n}"),
        );
    }
}

#[test]
fn matmul_with_bias_f32() {
    if no_rocm() {
        return;
    }
    let (m, k, n) = (8, 64, 32);
    let a: Vec<f32> = (0..m * k).map(|i| val(i, 1.0)).collect();
    let b: Vec<f32> = (0..k * n).map(|i| val(i + 5, 1.0)).collect();
    let bias: Vec<f32> = (0..n).map(|i| val(i + 3, 2.0)).collect();
    let mut want = cpu_matmul(&a, &b, m, k, n);
    for i in 0..m {
        for j in 0..n {
            want[i * n + j] += bias[j];
        }
    }
    let ta = Tensor::from_vec_on(Device::Rocm(0), a, vec![m, k]).expect("a");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b, vec![k, n]).expect("b");
    let tbias = Tensor::from_vec_on(Device::Rocm(0), bias, vec![n]).expect("bias");
    let tc = kiln_tensor::rocm_matmul_with_bias(&ta, &tb, &tbias).expect("matmul_with_bias");
    let got = kiln_tensor::rocm_to_host_copy(&tc)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    check_close(&got, &want, 1e-4, 1e-4, "f32 matmul+bias");
}

#[test]
fn matmul_batched_f32() {
    if no_rocm() {
        return;
    }
    let (batch, m, k, n) = (3, 4, 16, 8);
    let a: Vec<f32> = (0..batch * m * k).map(|i| val(i, 1.0)).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| val(i + 5, 1.0)).collect();
    let mut want = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let ca = cpu_matmul(
            &a[bi * m * k..(bi + 1) * m * k],
            &b[bi * k * n..(bi + 1) * k * n],
            m,
            k,
            n,
        );
        want[bi * m * n..(bi + 1) * m * n].copy_from_slice(&ca);
    }
    let ta = Tensor::from_vec_on(Device::Rocm(0), a, vec![batch, m, k]).expect("a");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b, vec![batch, k, n]).expect("b");
    let tc = kiln_tensor::rocm_matmul(&ta, &tb).expect("batched matmul");
    let got = kiln_tensor::rocm_to_host_copy(&tc)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    check_close(&got, &want, 1e-4, 1e-4, "f32 batched");
}

#[test]
fn matmul_lhs_transposed_f32_and_bf16() {
    if no_rocm() {
        return;
    }

    let (k, m, n) = (257usize, 19usize, 23usize);
    let a_f: Vec<f32> = (0..k * m).map(|i| val(i, 1.0)).collect();
    let b_f: Vec<f32> = (0..k * n).map(|i| val(i + 5, 1.0)).collect();

    let want_f32 = cpu_lhs_t_matmul(&a_f, &b_f, k, m, n);
    let ta = Tensor::from_vec_on(Device::Rocm(0), a_f.clone(), vec![k, m]).expect("a f32");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b_f.clone(), vec![k, n]).expect("b f32");
    let tc = kiln_tensor::ops::matmul_lhs_transposed(&ta, &tb)
        .unwrap_or_else(|e| panic!("lhs-transposed f32 matmul: {e}"));
    assert_eq!(tc.shape(), &[m, n]);
    let got = kiln_tensor::rocm_to_host_copy(&tc)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    check_close(&got, &want_f32, 1e-4, 1e-4, "lhs-transposed f32");

    use half::bf16;
    let a_bf: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
    let b_bf: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();
    let a_r: Vec<f32> = a_bf.iter().map(|x| x.to_f32()).collect();
    let b_r: Vec<f32> = b_bf.iter().map(|x| x.to_f32()).collect();
    let want_bf16 = cpu_lhs_t_matmul(&a_r, &b_r, k, m, n);

    let ta = Tensor::from_vec_on(Device::Rocm(0), a_bf, vec![k, m]).expect("a bf16");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b_bf, vec![k, n]).expect("b bf16");
    let tc = kiln_tensor::ops::matmul_lhs_transposed(&ta, &tb)
        .unwrap_or_else(|e| panic!("lhs-transposed bf16 matmul: {e}"));
    assert_eq!(tc.shape(), &[m, n]);
    let got_bf = kiln_tensor::rocm_to_host_copy(&tc)
        .unwrap()
        .to_vec::<bf16>()
        .unwrap();
    let got: Vec<f32> = got_bf.iter().map(|x| x.to_f32()).collect();
    check_close(
        &got,
        &want_bf16,
        3e-2,
        (k as f32) * 1e-3,
        "lhs-transposed bf16",
    );
}

#[test]
fn matmul_rhs_transposed_f32_and_bf16() {
    if no_rocm() {
        return;
    }

    let (m, k, n) = (19usize, 257usize, 23usize);
    let a_f: Vec<f32> = (0..m * k).map(|i| val(i, 1.0)).collect();
    let b_f: Vec<f32> = (0..n * k).map(|i| val(i + 5, 1.0)).collect();

    let want_f32 = cpu_rhs_t_matmul(&a_f, &b_f, m, k, n);
    let ta = Tensor::from_vec_on(Device::Rocm(0), a_f.clone(), vec![m, k]).expect("a f32");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b_f.clone(), vec![n, k]).expect("b f32");
    let tc = kiln_tensor::ops::matmul_rhs_transposed(&ta, &tb)
        .unwrap_or_else(|e| panic!("rhs-transposed f32 matmul: {e}"));
    assert_eq!(tc.shape(), &[m, n]);
    let got = kiln_tensor::rocm_to_host_copy(&tc)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    check_close(&got, &want_f32, 1e-4, 1e-4, "rhs-transposed f32");

    use half::bf16;
    let a_bf: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
    let b_bf: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();
    let a_r: Vec<f32> = a_bf.iter().map(|x| x.to_f32()).collect();
    let b_r: Vec<f32> = b_bf.iter().map(|x| x.to_f32()).collect();
    let want_bf16 = cpu_rhs_t_matmul(&a_r, &b_r, m, k, n);

    let ta = Tensor::from_vec_on(Device::Rocm(0), a_bf, vec![m, k]).expect("a bf16");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b_bf, vec![n, k]).expect("b bf16");
    let tc = kiln_tensor::ops::matmul_rhs_transposed(&ta, &tb)
        .unwrap_or_else(|e| panic!("rhs-transposed bf16 matmul: {e}"));
    assert_eq!(tc.shape(), &[m, n]);
    let got_bf = kiln_tensor::rocm_to_host_copy(&tc)
        .unwrap()
        .to_vec::<bf16>()
        .unwrap();
    let got: Vec<f32> = got_bf.iter().map(|x| x.to_f32()).collect();
    check_close(
        &got,
        &want_bf16,
        3e-2,
        (k as f32) * 1e-3,
        "rhs-transposed bf16",
    );
}

#[test]
fn matmul_bf16_inputs_f32_output() {
    if no_rocm() {
        return;
    }

    use half::bf16;
    let (m, k, n) = (17usize, 129usize, 19usize);
    let a_f: Vec<f32> = (0..m * k).map(|i| val(i, 1.0)).collect();
    let b_f: Vec<f32> = (0..k * n).map(|i| val(i + 13, 1.0)).collect();
    let a_bf: Vec<bf16> = a_f.iter().map(|&x| bf16::from_f32(x)).collect();
    let b_bf: Vec<bf16> = b_f.iter().map(|&x| bf16::from_f32(x)).collect();
    let a_r: Vec<f32> = a_bf.iter().map(|x| x.to_f32()).collect();
    let b_r: Vec<f32> = b_bf.iter().map(|x| x.to_f32()).collect();
    let want = cpu_matmul(&a_r, &b_r, m, k, n);

    let ta = Tensor::from_vec_on(Device::Rocm(0), a_bf, vec![m, k]).expect("a bf16");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b_bf, vec![k, n]).expect("b bf16");
    let tc = kiln_tensor::rocm_matmul_to_dtype(&ta, &tb, DType::F32)
        .unwrap_or_else(|e| panic!("bf16->f32 rocm matmul: {e}"));
    assert_eq!(tc.shape(), &[m, n]);
    assert_eq!(tc.dtype(), DType::F32);
    let got = kiln_tensor::rocm_to_host_copy(&tc)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    check_close(&got, &want, 3e-2, (k as f32) * 1e-3, "bf16->f32 matmul");
}

#[test]
fn matmul_bf16_inputs_f32_output_attention_chunk_shapes() {
    if no_rocm() {
        return;
    }

    use half::bf16;
    for &(m, k, n) in &[
        (32usize, 256usize, 32usize),
        (32, 256, 64),
        (32, 256, 96),
        (32, 256, 128),
        (32, 256, 160),
        (32, 256, 192),
        (32, 256, 224),
        (14, 256, 238),
    ] {
        let a_bf: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32(val(i, 1.0))).collect();
        let b_bf: Vec<bf16> = (0..k * n)
            .map(|i| bf16::from_f32(val(i + 13, 1.0)))
            .collect();
        let a_f32: Vec<f32> = a_bf.iter().map(|value| value.to_f32()).collect();
        let b_f32: Vec<f32> = b_bf.iter().map(|value| value.to_f32()).collect();
        let want = cpu_matmul(&a_f32, &b_f32, m, k, n);

        let ta = Tensor::from_vec_on(Device::Rocm(0), a_bf, vec![m, k]).expect("attention q");
        let tb = Tensor::from_vec_on(Device::Rocm(0), b_bf, vec![k, n]).expect("attention k");
        let tc = kiln_tensor::rocm_matmul_to_dtype(&ta, &tb, DType::F32)
            .unwrap_or_else(|error| panic!("attention bf16->f32 {m}x{k}x{n}: {error}"));
        let got = kiln_tensor::rocm_to_host_copy(&tc)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        check_close(
            &got,
            &want,
            3e-2,
            (k as f32) * 1e-3,
            &format!("attention bf16->f32 {m}x{k}x{n}"),
        );
    }
}
