//! Phase R.5 — CPU-vs-ROCm parity for the masked_fill kernel.
//!
//! `masked_fill` is elementwise (`out[i] = mask[i] ? fill : x[i]`), so there is
//! no cross-lane reduction and no wave-size hazard; a couple of shapes suffice.
//! Skips when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_masked_fill_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 masked_fill parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-5, 5) for index `i`.
fn val(i: usize) -> f32 {
    ((i * 137 + 7) % 1000) as f32 / 100.0 - 5.0
}

/// Deterministic mask byte (0 or 1) for index `i`.
fn mask_byte(i: usize) -> u8 {
    ((i * 53 + 11).is_multiple_of(3)) as u8
}

#[test]
fn masked_fill_parity() {
    if no_rocm() {
        return;
    }
    let fill_value = -1.0e30f32;
    // A handful of shapes including a non-multiple-of-block size and a 2D shape.
    let shapes: [Vec<usize>; 5] = [vec![1], vec![257], vec![6, 64], vec![5, 333], vec![4, 7, 9]];

    for shape in &shapes {
        let n: usize = shape.iter().product();

        let x_data: Vec<f32> = (0..n).map(val).collect();
        let mask_data: Vec<u8> = (0..n).map(mask_byte).collect();

        // CPU reference.
        let reference: Vec<f32> = (0..n)
            .map(|i| {
                if mask_data[i] != 0 {
                    fill_value
                } else {
                    x_data[i]
                }
            })
            .collect();

        // Device path.
        let x = Tensor::from_vec_on(Device::Rocm(0), x_data, shape.clone())
            .unwrap_or_else(|e| panic!("from_vec_on x (shape={shape:?}): {e}"));
        let mask = Tensor::from_vec_on(Device::Rocm(0), mask_data, shape.clone())
            .unwrap_or_else(|e| panic!("from_vec_on mask (shape={shape:?}): {e}"));

        let y = kiln_tensor::rocm_masked_fill(&x, &mask, fill_value)
            .unwrap_or_else(|e| panic!("rocm_masked_fill (shape={shape:?}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (shape={shape:?}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "shape {shape:?}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - rf).abs();
            assert!(
                diff <= 1e-5 + 1e-4 * rf.abs(),
                "masked_fill mismatch at shape={shape:?} idx={i}: got {g} ref {rf} diff {diff}"
            );
        }
    }
    eprintln!("masked_fill CPU-vs-ROCm parity passed across shapes");
}

#[test]
fn causal_mask_fill_parity_with_prefix_offset() {
    if no_rocm() {
        return;
    }
    let fill_value = -1.0e30f32;
    let bh = 3usize;
    let sq = 5usize;
    let sk = 11usize;
    let shape = vec![bh, sq, sk];
    let n = bh * sq * sk;
    let x_data: Vec<f32> = (0..n).map(val).collect();

    let reference: Vec<f32> = (0..n)
        .map(|idx| {
            let col = idx % sk;
            let row = (idx / sk) % sq;
            if col > row + (sk - sq) {
                fill_value
            } else {
                x_data[idx]
            }
        })
        .collect();

    let x = Tensor::from_vec_on(Device::Rocm(0), x_data, shape.clone())
        .unwrap_or_else(|e| panic!("from_vec_on x (shape={shape:?}): {e}"));
    let y = kiln_tensor::rocm_causal_mask_fill(&x, sq, sk, fill_value)
        .unwrap_or_else(|e| panic!("rocm_causal_mask_fill (shape={shape:?}): {e}"));
    let host = kiln_tensor::rocm_to_host_copy(&y)
        .unwrap_or_else(|e| panic!("rocm_to_host_copy (shape={shape:?}): {e}"));
    let got = host.to_vec::<f32>().expect("to_vec");

    assert_eq!(got.len(), reference.len());
    for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
        let diff = (g - rf).abs();
        assert!(
            diff <= 1e-5 + 1e-4 * rf.abs(),
            "causal_mask_fill mismatch idx={i}: got {g} ref {rf} diff {diff}"
        );
    }
}
