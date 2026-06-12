//! Phase R.3 smoke tests — Tensor <-> ROCm device round-trips on a real AMD
//! GPU. Skips cleanly when no ROCm device is present (toolchain-less / CI hosts
//! without an AMD GPU), mirroring the cuda/metal availability-probe pattern.
//!
//! Run with: `cargo test -p kiln-tensor --features rocm --test rocm_storage_smoke`
#![cfg(feature = "rocm")]

use kiln_tensor::{DType, Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.3 smoke test");
        true
    } else {
        false
    }
}

#[test]
fn zeros_on_rocm_then_host() {
    if no_rocm() {
        return;
    }
    let t = Tensor::zeros_on(Device::Rocm(0), vec![4, 8], DType::F32).expect("zeros_on rocm");
    assert_eq!(t.device(), Device::Rocm(0));
    let host = kiln_tensor::rocm_to_host_copy(&t).expect("rocm_to_host_copy");
    assert_eq!(host.device(), Device::Cpu);
    let v = host.to_vec::<f32>().expect("to_vec");
    assert_eq!(v.len(), 32);
    assert!(
        v.iter().all(|&x| x == 0.0),
        "zeros_on must produce all zeros"
    );
}

#[test]
fn from_vec_on_rocm_roundtrip() {
    if no_rocm() {
        return;
    }
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
    let t =
        Tensor::from_vec_on(Device::Rocm(0), data.clone(), vec![4, 8]).expect("from_vec_on rocm");
    assert_eq!(t.device(), Device::Rocm(0));
    let host = kiln_tensor::rocm_to_host_copy(&t).expect("rocm_to_host_copy");
    let got = host.to_vec::<f32>().expect("to_vec");
    assert_eq!(got, data, "H2D -> D2H must round-trip exactly");
}

#[test]
fn rocm_contiguous_materializes_transpose() {
    if no_rocm() {
        return;
    }
    // [[0,1,2],[3,4,5]] on device, transpose to a non-contiguous [3,2] view,
    // then force a contiguous copy through the hipcc contiguity kernel.
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![2, 3]).expect("from_vec_on");
    let tt = t.transpose(0, 1).expect("transpose"); // [3,2], non-contiguous
    assert_eq!(tt.device(), Device::Rocm(0));
    let host = kiln_tensor::rocm_to_host_copy(&tt).expect("to host (forces contiguous kernel)");
    let got = host.to_vec::<f32>().expect("to_vec");
    // transposed [[0,3],[1,4],[2,5]] -> flat [0,3,1,4,2,5]
    assert_eq!(got, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}
