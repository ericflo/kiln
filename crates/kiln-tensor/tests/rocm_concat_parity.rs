#![cfg(feature = "rocm")]
//! ROCm `concat` parity test (Phase R.5).
//!
//! Builds inputs on `Device::Rocm(0)`, runs `rocm_concat`, copies the result
//! back to host, and compares against a CPU reference computed in-test. Concat
//! is a byte-wise copy (not a reduction), so a handful of representative
//! shapes/axes/arities cover it — no wavefront-boundary sweep needed.

use kiln_tensor::{Device, Tensor, rocm_concat, rocm_is_available, rocm_to_host_copy};

/// Read a ROCm tensor back to a host `Vec<f32>`.
fn to_host_f32(t: &Tensor) -> Vec<f32> {
    let host = rocm_to_host_copy(t).expect("rocm_to_host_copy");
    let cpu = host
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .expect("host storage is CpuStorage");
    cpu.as_bytes()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// CPU reference concat over f32 vectors of given shapes along `axis`.
/// Mirrors the per-outer-slab copy in `ops/concat.rs`.
fn cpu_concat_ref(inputs: &[(Vec<f32>, Vec<usize>)], axis: usize) -> (Vec<f32>, Vec<usize>) {
    let rank = inputs[0].1.len();
    let mut out_shape = inputs[0].1.clone();
    let axis_total: usize = inputs.iter().map(|(_, s)| s[axis]).sum();
    out_shape[axis] = axis_total;

    let outer: usize = out_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = out_shape[axis + 1..].iter().product::<usize>().max(1);

    let _ = rank;
    let mut out = vec![0.0f32; outer * axis_total * inner];
    for o in 0..outer {
        let mut axis_off = 0usize;
        for (vals, shape) in inputs {
            let t_axis = shape[axis];
            let src_start = o * t_axis * inner;
            let dst_start = (o * axis_total + axis_off) * inner;
            let len = t_axis * inner;
            out[dst_start..dst_start + len].copy_from_slice(&vals[src_start..src_start + len]);
            axis_off += t_axis;
        }
    }
    (out, out_shape)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        let tol = 1e-5 + 1e-4 * e.abs();
        assert!(
            diff <= tol,
            "mismatch at {i}: got {a}, want {e} (diff {diff} > tol {tol})"
        );
    }
}

/// Run one concat case on ROCm and compare to the CPU reference.
fn run_case(inputs: &[(Vec<f32>, Vec<usize>)], axis: usize) {
    let dev_tensors: Vec<Tensor> = inputs
        .iter()
        .map(|(vals, shape)| {
            Tensor::from_vec_on(Device::Rocm(0), vals.clone(), shape.clone())
                .expect("from_vec_on Rocm")
        })
        .collect();
    let refs: Vec<&Tensor> = dev_tensors.iter().collect();

    let out = rocm_concat(&refs, axis).expect("rocm_concat");
    let (expected, expected_shape) = cpu_concat_ref(inputs, axis);

    assert_eq!(
        out.shape(),
        expected_shape.as_slice(),
        "output shape mismatch"
    );
    let got = to_host_f32(&out);
    assert_close(&got, &expected);
}

fn iota(n: usize, base: f32) -> Vec<f32> {
    (0..n).map(|i| base + i as f32).collect()
}

#[test]
fn concat_rank1_axis0_two_inputs() {
    if !rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }
    run_case(&[(iota(3, 1.0), vec![3]), (iota(2, 100.0), vec![2])], 0);
}

#[test]
fn concat_rank2_axis0() {
    if !rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }
    // [2,2] + [1,2] along axis 0 = [3,2]
    run_case(
        &[(iota(4, 1.0), vec![2, 2]), (iota(2, 100.0), vec![1, 2])],
        0,
    );
}

#[test]
fn concat_rank2_axis1() {
    if !rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }
    // [2,2] + [2,1] along axis 1 = [2,3]  (exercises inner-byte offsets)
    run_case(
        &[(iota(4, 1.0), vec![2, 2]), (iota(2, 100.0), vec![2, 1])],
        1,
    );
}

#[test]
fn concat_rank3_mid_axis_three_inputs() {
    if !rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }
    // [2,1,3] + [2,2,3] + [2,1,3] along axis 1 = [2,4,3]
    run_case(
        &[
            (iota(2 * 3, 1.0), vec![2, 1, 3]),
            (iota(2 * 2 * 3, 100.0), vec![2, 2, 3]),
            (iota(2 * 3, 1000.0), vec![2, 1, 3]),
        ],
        1,
    );
}

#[test]
fn concat_wide_inner_axis0() {
    if !rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }
    // Larger inner dimension so inner_bytes spans many bytes per element-slab.
    let inner = 257usize;
    run_case(
        &[
            (iota(3 * inner, 1.0), vec![3, inner]),
            (iota(5 * inner, 5000.0), vec![5, inner]),
        ],
        0,
    );
}

#[test]
fn concat_single_input_identity() {
    if !rocm_is_available() {
        eprintln!("skip: no ROCm device");
        return;
    }
    run_case(&[(iota(6, 1.0), vec![2, 3])], 0);
}
