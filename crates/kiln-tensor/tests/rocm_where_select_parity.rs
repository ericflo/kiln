#![cfg(feature = "rocm")]
//! Parity test for the ROCm `where_select` kernel (Phase R.5).
//!
//! `out[i] = mask[i] != 0 ? t[i] : f[i]`, elementwise over contiguous inputs.
//! Compares the ROCm result against a CPU reference. Skips entirely when no
//! AMD device is present.

use kiln_tensor::{Device, Tensor, rocm_where_select};

/// CPU reference: out[i] = mask[i] != 0 ? t[i] : f[i].
fn cpu_ref(mask: &[u8], t: &[f32], f: &[f32]) -> Vec<f32> {
    (0..t.len())
        .map(|i| if mask[i] != 0 { t[i] } else { f[i] })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], shape: &[usize]) {
    assert_eq!(got.len(), want.len(), "len mismatch for shape {shape:?}");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = 1e-5 + 1e-4 * w.abs();
        assert!(
            diff <= tol,
            "shape {shape:?} idx {i}: got {g} want {w} (diff {diff} > tol {tol})"
        );
    }
}

fn run_case(shape: Vec<usize>) {
    let n: usize = shape.iter().product();
    let dev = Device::Rocm(0);

    // Deterministic, sign-varied inputs; mask alternates / mixes patterns.
    let mask: Vec<u8> = (0..n)
        .map(|i| {
            if (i % 3 == 0) || (i % 7 == 0) {
                1u8
            } else {
                0u8
            }
        })
        .collect();
    let t: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let f: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.25 + 1.0).collect();

    let want = cpu_ref(&mask, &t, &f);

    let mask_dev = Tensor::from_vec_on(dev, mask, shape.clone()).expect("mask to device");
    let t_dev = Tensor::from_vec_on(dev, t, shape.clone()).expect("t to device");
    let f_dev = Tensor::from_vec_on(dev, f, shape.clone()).expect("f to device");

    let out_dev = rocm_where_select(&mask_dev, &t_dev, &f_dev).expect("rocm_where_select");
    assert_eq!(out_dev.shape(), shape.as_slice());

    let out_host = kiln_tensor::rocm_to_host_copy(&out_dev).expect("copy back to host");
    let got = out_host.to_vec::<f32>().expect("host to_vec");

    assert_close(&got, &want, &shape);
}

#[test]
fn rocm_where_select_f32_parity() {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("skipping rocm_where_select_f32_parity: no ROCm device");
        return;
    }
    // Elementwise op: a couple of shapes (incl. non-multiple-of-block sizes)
    // exercise the masked-select path and the tail of the grid.
    run_case(vec![1]);
    run_case(vec![7]);
    run_case(vec![257]);
    run_case(vec![4, 65]);
    run_case(vec![3, 256]);
    run_case(vec![2, 3, 100]);
}
