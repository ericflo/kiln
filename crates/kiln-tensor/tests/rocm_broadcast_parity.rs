#![cfg(feature = "rocm")]
//! ROCm `broadcast_to` parity for the direct broadcast kernel.
//!
//! These shapes mirror long-context inference broadcasts such as RoPE tables
//! expanded across batch/head axes. The old ROCm path flattened these into very
//! large `index_select_dim0` launches.

use kiln_tensor::{ops::broadcast_to, rocm_is_available, rocm_to_host_copy, Device, Tensor};

fn to_host_f32(t: &Tensor) -> Vec<f32> {
    let host = rocm_to_host_copy(t).expect("rocm_to_host_copy");
    host.to_vec::<f32>().expect("to_vec")
}

fn iota(n: usize, base: f32) -> Vec<f32> {
    (0..n).map(|i| base + (i as f32 * 0.25)).collect()
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn cpu_broadcast_ref(src: &[f32], in_shape: &[usize], target_shape: &[usize]) -> Vec<f32> {
    let out_n: usize = target_shape.iter().product();
    let in_strides = contiguous_strides(in_shape);
    let out_strides = contiguous_strides(target_shape);
    let mut out = vec![0.0; out_n];

    for (linear, dst) in out.iter_mut().enumerate() {
        let mut src_linear = 0usize;
        for axis in 0..target_shape.len() {
            let coord = (linear / out_strides[axis]) % target_shape[axis];
            let src_coord = if in_shape[axis] == 1 { 0 } else { coord };
            src_linear += src_coord * in_strides[axis];
        }
        *dst = src[src_linear];
    }
    out
}

fn run_case(in_shape: Vec<usize>, target_shape: Vec<usize>) {
    if !rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }

    let n_in: usize = in_shape.iter().product();
    let data = iota(n_in, 1.0);
    let x = Tensor::from_vec_on(Device::Rocm(0), data.clone(), in_shape.clone())
        .expect("from_vec_on Rocm");
    let y = broadcast_to(&x, &target_shape).expect("broadcast_to Rocm");
    assert_eq!(y.shape(), target_shape.as_slice());

    let got = to_host_f32(&y);
    let expected = cpu_broadcast_ref(&data, &in_shape, &target_shape);
    assert_eq!(got, expected, "shape {in_shape:?} -> {target_shape:?}");
}

#[test]
fn broadcast_rope_table_like_axes() {
    run_case(vec![1, 4096, 1, 32], vec![1, 4096, 28, 32]);
}

#[test]
fn broadcast_long_table_like_axes() {
    run_case(vec![1, 8192, 1, 16], vec![1, 8192, 8, 16]);
}

#[test]
fn broadcast_multiple_size_one_axes() {
    run_case(vec![1, 7, 1, 5], vec![3, 7, 11, 5]);
}
