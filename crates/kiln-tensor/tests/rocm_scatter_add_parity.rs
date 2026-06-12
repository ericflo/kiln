//! Phase R.5b — scatter_add (dim0 atomic scatter) CPU-vs-ROCm parity.
//!
//! `scatter_add.cu` is the embedding-backward primitive:
//!   out[indices[i], j] += updates[i, j]   (out pre-zeroed)
//!
//! The hazard this test guards is bf16 atomics: HIP has NO native
//! `atomicAdd(__hip_bfloat16*)`, so the kernel routes bf16 through a
//! CAS-on-dword helper (`kiln_atomic_add_bf16`). The CUDA kernel's stock
//! `#else` non-atomic read-modify-write is taken under hipcc unless the helper
//! is wired in, and it produces WRONG results exactly when several indices
//! COLLIDE on the same target row — the common case for embedding-backward.
//! Every case below therefore packs heavy collisions so a non-atomic bf16 path
//! would visibly diverge from the CPU reference.
//!
//! F32 uses the native `atomicAdd(float*)` path. F16 is not supported by the
//! kernel and falls through to the host path at the op layer (not exercised
//! here).
//!
//! Run: cargo test -p kiln-tensor --features rocm --test rocm_scatter_add_parity
#![cfg(feature = "rocm")]

use kiln_tensor::ops::scatter_add;
use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5b scatter_add parity test");
        true
    } else {
        false
    }
}

/// CPU reference: zero an `[target_dim, inner]` buffer, then add each updates
/// row into the target row named by `indices`. F32 accumulation.
fn cpu_scatter_add(updates: &[f32], indices: &[u32], target_dim: usize, inner: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; target_dim * inner];
    for (i, &id) in indices.iter().enumerate() {
        let id = id as usize;
        for j in 0..inner {
            out[id * inner + j] += updates[i * inner + j];
        }
    }
    out
}

fn val(i: usize, j: usize) -> f32 {
    (((i * 37 + j * 13 + 5) % 100) as f32) / 10.0 - 5.0
}

#[test]
fn scatter_add_f32_collisions_parity() {
    if no_rocm() {
        return;
    }
    // (n_indices, target_dim, inner). Indices are generated mod target_dim so
    // MANY rows collide — the exact regime where a non-atomic add breaks.
    let cases = [
        (8usize, 3usize, 1usize),
        (64, 4, 1),
        (128, 5, 7),
        (256, 8, 33), // inner straddles the 32-lane boundary
        (1000, 16, 64),
        (2048, 7, 65),
    ];

    for &(n_indices, target_dim, inner) in &cases {
        let updates: Vec<f32> = (0..n_indices)
            .flat_map(|i| (0..inner).map(move |j| val(i, j)))
            .collect();
        let indices: Vec<u32> = (0..n_indices).map(|i| (i % target_dim) as u32).collect();

        let reference = cpu_scatter_add(&updates, &indices, target_dim, inner);

        let upd_t = Tensor::from_vec_on(Device::Rocm(0), updates, vec![n_indices, inner])
            .unwrap_or_else(|e| panic!("from_vec_on updates: {e}"));
        let idx_t = Tensor::from_vec_on(Device::Rocm(0), indices, vec![n_indices])
            .unwrap_or_else(|e| panic!("from_vec_on indices: {e}"));

        let out = scatter_add(&upd_t, 0, &idx_t, target_dim)
            .unwrap_or_else(|e| panic!("scatter_add f32 ({n_indices},{target_dim},{inner}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&out).unwrap();
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len());
        for (k, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - rf).abs();
            assert!(
                diff <= 1e-3 + 1e-3 * rf.abs(),
                "f32 scatter_add mismatch case=({n_indices},{target_dim},{inner}) idx={k}: \
                 got {g} ref {rf} diff {diff} (a lost atomic add shows up here)"
            );
        }
    }
    eprintln!("scatter_add F32 collision parity passed");
}

#[test]
fn scatter_add_bf16_collisions_parity() {
    if no_rocm() {
        return;
    }
    // bf16 is the load-bearing case: validates the CAS-on-dword atomic. Values
    // kept small + collision counts modest so bf16's ~3-digit precision still
    // lands within tolerance after accumulation.
    let cases = [
        (8usize, 3usize, 1usize),
        (64, 4, 1),
        (128, 5, 8),
        (256, 8, 33),
        (512, 16, 64),
    ];

    for &(n_indices, target_dim, inner) in &cases {
        // Small magnitudes so the running F32-in-bf16 sum stays representable.
        let updates_f: Vec<f32> = (0..n_indices)
            .flat_map(|i| (0..inner).map(move |j| (((i + j) % 7) as f32) * 0.25))
            .collect();
        let indices: Vec<u32> = (0..n_indices).map(|i| (i % target_dim) as u32).collect();

        let reference = cpu_scatter_add(&updates_f, &indices, target_dim, inner);

        let updates_bf: Vec<half::bf16> =
            updates_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let upd_t = Tensor::from_vec_on(Device::Rocm(0), updates_bf, vec![n_indices, inner])
            .unwrap_or_else(|e| panic!("from_vec_on bf16 updates: {e}"));
        let idx_t = Tensor::from_vec_on(Device::Rocm(0), indices, vec![n_indices])
            .unwrap_or_else(|e| panic!("from_vec_on indices: {e}"));

        let out = scatter_add(&upd_t, 0, &idx_t, target_dim)
            .unwrap_or_else(|e| panic!("scatter_add bf16 ({n_indices},{target_dim},{inner}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&out).unwrap();
        let got = host.to_vec::<half::bf16>().expect("to_vec bf16");

        assert_eq!(got.len(), reference.len());
        for (k, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            let gf = g.to_f32();
            // Tolerance scales with the number of collisions on this row (each
            // accumulation rounds to bf16). Use a relative band tied to the
            // magnitude plus a small absolute floor.
            let tol = 0.06 + 0.08 * rf.abs();
            let diff = (gf - rf).abs();
            assert!(
                diff <= tol,
                "bf16 scatter_add mismatch case=({n_indices},{target_dim},{inner}) idx={k}: \
                 got {gf} ref {rf} diff {diff} tol {tol} \
                 (a NON-atomic bf16 add — the HIP #else branch — diverges far past this)"
            );
        }
    }
    eprintln!("scatter_add BF16 collision parity passed (CAS-on-dword atomic correct)");
}

#[test]
fn scatter_add_f32_known_small() {
    if no_rocm() {
        return;
    }
    // out has target_dim=2, inner=2. indices [0,1,0] → row0 gets rows 0+2, row1 gets row1.
    let updates = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0];
    let indices = vec![0u32, 1, 0];
    let upd_t = Tensor::from_vec_on(Device::Rocm(0), updates, vec![3, 2]).unwrap();
    let idx_t = Tensor::from_vec_on(Device::Rocm(0), indices, vec![3]).unwrap();
    let out = scatter_add(&upd_t, 0, &idx_t, 2).unwrap();
    let got = kiln_tensor::rocm_to_host_copy(&out)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    // row0 = [1+10, 2+20] = [11, 22]; row1 = [3, 4].
    assert_eq!(got, vec![11.0, 22.0, 3.0, 4.0]);
}
