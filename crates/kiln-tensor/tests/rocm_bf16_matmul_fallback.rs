#![cfg(feature = "rocm")]

use half::bf16;
use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping ROCm BF16 fallback matmul test");
        true
    } else {
        false
    }
}

fn val(i: usize, scale: f32) -> f32 {
    (((i * 37 + 11) % 97) as f32 / 97.0 - 0.5) * scale
}

fn cpu_matmul_sample(
    a: &[bf16],
    b: &[bf16],
    m: usize,
    k: usize,
    n: usize,
    i: usize,
    j: usize,
) -> f32 {
    assert!(i < m);
    assert!(j < n);
    let mut acc = 0.0f32;
    for p in 0..k {
        acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
    }
    acc
}

fn assert_bf16_close(got: bf16, want: f32, label: &str) {
    let got = got.to_f32();
    let want_bf16 = bf16::from_f32(want).to_f32();
    let diff = (got - want_bf16).abs();
    assert!(
        diff <= 0.05,
        "{label}: got {got} want {want_bf16} raw_want {want} diff {diff}"
    );
}

#[test]
fn fallback_matches_cpu_for_q_gate_like_chunks() {
    if no_rocm() {
        return;
    }

    for &(m, k, n) in &[(17, 257, 31), (64, 2560, 1024), (129, 2560, 512)] {
        let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32(val(i, 0.125))).collect();
        let b: Vec<bf16> = (0..k * n)
            .map(|i| bf16::from_f32(val(i + 5, 0.125)))
            .collect();

        let ta = Tensor::from_vec_on(Device::Rocm(0), a.clone(), vec![m, k]).expect("a");
        let tb = Tensor::from_vec_on(Device::Rocm(0), b.clone(), vec![k, n]).expect("b");
        let tc = kiln_tensor::rocm_bf16_matmul_bf16_out(&ta, &tb)
            .unwrap_or_else(|e| panic!("fallback matmul {m}x{k}x{n}: {e}"));
        assert_eq!(tc.shape(), &[m, n]);
        let got = kiln_tensor::rocm_to_host_copy(&tc)
            .expect("copy")
            .to_vec::<bf16>()
            .expect("to_vec");

        for &(i, j) in &[
            (0usize, 0usize),
            (0, n - 1),
            (m / 2, n / 2),
            (m - 1, 0),
            (m - 1, n - 1),
        ] {
            let want = cpu_matmul_sample(&a, &b, m, k, n, i, j);
            assert_bf16_close(got[i * n + j], want, &format!("{m}x{k}x{n} [{i},{j}]"));
        }
    }
}

#[test]
fn fallback_large_q_gate_tile_is_finite_and_nonzero() {
    if no_rocm() {
        return;
    }

    let (m, k, n) = (512usize, 2560usize, 1024usize);
    let a: Vec<bf16> = (0..m * k).map(|i| bf16::from_f32(val(i, 0.125))).collect();
    let b: Vec<bf16> = (0..k * n)
        .map(|i| bf16::from_f32(val(i + 5, 0.125)))
        .collect();
    let ta = Tensor::from_vec_on(Device::Rocm(0), a, vec![m, k]).expect("a");
    let tb = Tensor::from_vec_on(Device::Rocm(0), b, vec![k, n]).expect("b");
    let tc = kiln_tensor::rocm_bf16_matmul_bf16_out(&ta, &tb)
        .unwrap_or_else(|e| panic!("fallback large matmul: {e}"));
    let got = kiln_tensor::rocm_to_host_copy(&tc)
        .expect("copy")
        .to_vec::<bf16>()
        .expect("to_vec");

    let mut nonzero = 0usize;
    for (idx, x) in got.iter().enumerate() {
        let f = x.to_f32();
        assert!(f.is_finite(), "non-finite output at {idx}: {f}");
        if f != 0.0 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero > got.len() / 2,
        "unexpectedly sparse fallback output: {nonzero}/{}",
        got.len()
    );
}
