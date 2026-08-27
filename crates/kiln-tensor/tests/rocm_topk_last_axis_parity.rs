//! R.5b — CPU-vs-ROCm parity for the on-device sampling top-k kernel.
//!
//! `rocm_topk_last_axis` keeps the full `[V]` logits row resident and returns
//! only the `k` `(value, index)` pairs. The ranking MUST match the host-sort
//! fallback `topk_via_host_sort` in `kiln-model/src/sampling.rs` exactly:
//! descending value, ties broken by LOWEST index. The per-pass argmax reduction
//! uses 32-lane subgroups + `kiln_shfl_xor` (offset capped at 16, never crossing
//! 32 lanes via shuffle), so it is wave32/64-correct.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_topk_last_axis_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5b topk parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-8, 8) for index i.
fn val(i: usize) -> f32 {
    (((i * 2_654_435_761usize.wrapping_rem(7919) + 13) % 1600) as f32) / 100.0 - 8.0
}

/// CPU reference: top-k of `data` by (descending value, ties → lower index).
fn cpu_topk(data: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    idx.sort_by(|&a, &b| {
        data[b]
            .partial_cmp(&data[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b)) // tie → lower index first
    });
    idx.into_iter()
        .take(k)
        .map(|i| (i as u32, data[i]))
        .collect()
}

#[test]
fn topk_matches_cpu_across_widths_and_k() {
    if no_rocm() {
        return;
    }
    // Widths spanning < one block, exactly one, and many blocks; plus a
    // vocab-scale row. k values covering the Qwen3.5 default (20) and edges.
    for &w in &[37usize, 256, 1024, 4096, 152_064] {
        let data: Vec<f32> = (0..w).map(val).collect();
        for &k in &[1usize, 5, 20, 64] {
            if k > w {
                continue;
            }
            let t = Tensor::from_vec_on(Device::Rocm(0), data.clone(), vec![w])
                .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
            let (values, indices) = kiln_tensor::rocm_topk_last_axis(&t, k)
                .unwrap_or_else(|e| panic!("rocm_topk_last_axis (w={w}, k={k}): {e}"));
            let expect = cpu_topk(&data, k);
            assert_eq!(values.len(), k, "value count (w={w}, k={k})");
            assert_eq!(indices.len(), k, "index count (w={w}, k={k})");
            for (rank, (ei, ev)) in expect.iter().enumerate() {
                assert_eq!(
                    indices[rank], *ei,
                    "index mismatch at rank {rank} (w={w}, k={k}): got {} want {ei}",
                    indices[rank]
                );
                assert!(
                    (values[rank] - *ev).abs() < 1e-5,
                    "value mismatch at rank {rank} (w={w}, k={k}): got {} want {ev}",
                    values[rank]
                );
            }
        }
    }
}

#[test]
fn topk_breaks_ties_to_lowest_index() {
    if no_rocm() {
        return;
    }
    // All-equal row: the top-k must be the FIRST k indices in order.
    let w = 512usize;
    let data = vec![std::f32::consts::PI; w];
    let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![w]).expect("from_vec_on tie");
    let (_values, indices) = kiln_tensor::rocm_topk_last_axis(&t, 8).expect("rocm_topk tie");
    let got: Vec<u32> = indices;
    assert_eq!(
        got,
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        "ties must pick lowest indices in order"
    );
}
