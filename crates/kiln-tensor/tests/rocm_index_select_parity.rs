//! Phase R.5 — CPU-vs-ROCm parity for the `index_select` gather kernel.
//!
//! `index_select` is a byte-copy gather (no cross-lane reductions), so this is
//! the elementwise variant of the R.5 parity harness: a handful of shapes
//! across both the dim0 fast path and the generic axis-N path, plus the
//! out-of-range-index contract (those output slices stay zero). Skips when no
//! ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_index_select_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 index_select parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random f32 in ~[-5, 5) for a flat element index.
fn val(i: usize) -> f32 {
    (((i * 131 + 7) % 1000) as f32) / 100.0 - 5.0
}

/// CPU reference gather along axis 0: out row `o` is `src` row `indices[o]`,
/// or all-zero when `indices[o]` is out of range.
fn cpu_dim0(src: &[f32], src_n_rows: usize, inner: usize, indices: &[u32]) -> Vec<f32> {
    let mut out = vec![0.0f32; indices.len() * inner];
    for (o, &id) in indices.iter().enumerate() {
        let id = id as usize;
        if id >= src_n_rows {
            continue;
        }
        out[o * inner..(o + 1) * inner].copy_from_slice(&src[id * inner..(id + 1) * inner]);
    }
    out
}

/// CPU reference gather along an arbitrary axis. `src` is row-major with shape
/// `[left_size, src_dim, right_size]` (already flattened around `axis`).
fn cpu_axis_n(
    src: &[f32],
    left_size: usize,
    src_dim: usize,
    right_size: usize,
    indices: &[u32],
) -> Vec<f32> {
    let ids_dim = indices.len();
    let mut out = vec![0.0f32; left_size * ids_dim * right_size];
    for left in 0..left_size {
        for (o, &id) in indices.iter().enumerate() {
            let id = id as usize;
            if id >= src_dim {
                continue;
            }
            let src_off = (left * src_dim + id) * right_size;
            let dst_off = (left * ids_dim + o) * right_size;
            out[dst_off..dst_off + right_size].copy_from_slice(&src[src_off..src_off + right_size]);
        }
    }
    out
}

fn assert_close(got: &[f32], reference: &[f32], ctx: &str) {
    assert_eq!(got.len(), reference.len(), "{ctx}: length mismatch");
    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        let diff = (g - r).abs();
        assert!(
            diff <= 1e-5 + 1e-4 * r.abs(),
            "{ctx}: mismatch at idx={i}: got {g} ref {r} diff {diff}"
        );
    }
}

#[test]
fn index_select_dim0_parity() {
    if no_rocm() {
        return;
    }
    // (src_n_rows, inner) shapes — inner=1 (pure row vector) up to a wide slice.
    let cases = [(4usize, 1usize), (5, 3), (8, 16), (10, 64), (3, 1000)];
    // Index lists exercise: identity, repeats, reverse, and one out-of-range id.
    for &(src_n_rows, inner) in &cases {
        let src: Vec<f32> = (0..src_n_rows * inner).map(val).collect();

        let mut idx_sets: Vec<Vec<u32>> = vec![
            (0..src_n_rows as u32).collect(),
            vec![0, 0, (src_n_rows - 1) as u32, 1],
        ];
        // Out-of-range index → that output row must stay zero.
        idx_sets.push(vec![0, src_n_rows as u32, 1]);

        for indices in &idx_sets {
            let reference = cpu_dim0(&src, src_n_rows, inner, indices);

            let src_t = Tensor::from_vec_on(Device::Rocm(0), src.clone(), vec![src_n_rows, inner])
                .unwrap_or_else(|e| panic!("from_vec_on src: {e}"));
            let idx_t = Tensor::from_vec_on(Device::Rocm(0), indices.clone(), vec![indices.len()])
                .unwrap_or_else(|e| panic!("from_vec_on indices: {e}"));

            let out = kiln_tensor::rocm_index_select_dim0(&src_t, &idx_t)
                .unwrap_or_else(|e| panic!("rocm_index_select_dim0: {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&out)
                .unwrap_or_else(|e| panic!("rocm_to_host_copy: {e}"));
            let got = host.to_vec::<f32>().expect("to_vec");

            assert_close(
                &got,
                &reference,
                &format!("dim0 src_n_rows={src_n_rows} inner={inner} ids={indices:?}"),
            );
        }
    }
    eprintln!("index_select_dim0 CPU-vs-ROCm parity passed");
}

#[test]
fn index_select_axis_n_parity() {
    if no_rocm() {
        return;
    }
    // (left_size, src_dim, right_size) for the generic axis-N path, plus the
    // axis we feed the wrapper (computed from left dims below).
    // We build a 3-D src [left, src_dim, right] and gather along axis 1.
    let cases = [
        (1usize, 4usize, 1usize),
        (2, 5, 3),
        (3, 6, 8),
        (2, 8, 64),
        (1, 4, 1000),
    ];
    for &(left_size, src_dim, right_size) in &cases {
        let src: Vec<f32> = (0..left_size * src_dim * right_size).map(val).collect();

        let mut idx_sets: Vec<Vec<u32>> = vec![
            (0..src_dim as u32).collect(),
            vec![0, src_dim as u32 - 1, 0, 1],
        ];
        // Out-of-range index → that output slice must stay zero.
        idx_sets.push(vec![0, src_dim as u32, 1]);

        for indices in &idx_sets {
            let reference = cpu_axis_n(&src, left_size, src_dim, right_size, indices);

            // src shape [left_size, src_dim, right_size]; gather along axis 1.
            let src_t = Tensor::from_vec_on(
                Device::Rocm(0),
                src.clone(),
                vec![left_size, src_dim, right_size],
            )
            .unwrap_or_else(|e| panic!("from_vec_on src: {e}"));
            let idx_t = Tensor::from_vec_on(Device::Rocm(0), indices.clone(), vec![indices.len()])
                .unwrap_or_else(|e| panic!("from_vec_on indices: {e}"));

            let out = kiln_tensor::rocm_index_select_axis_n(&src_t, 1, &idx_t)
                .unwrap_or_else(|e| panic!("rocm_index_select_axis_n: {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&out)
                .unwrap_or_else(|e| panic!("rocm_to_host_copy: {e}"));
            let got = host.to_vec::<f32>().expect("to_vec");

            assert_close(
                &got,
                &reference,
                &format!(
                    "axis_n left={left_size} src_dim={src_dim} right={right_size} ids={indices:?}"
                ),
            );
        }
    }
    eprintln!("index_select_axis_n CPU-vs-ROCm parity passed");
}
