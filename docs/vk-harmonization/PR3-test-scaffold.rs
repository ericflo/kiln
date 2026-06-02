//! PR3 test scaffold — `MatmulOp::vulkan_fwd` + VulkanStorage<->VkTensor
//! zero-copy bridge parity / reachability tests. (#1082, PR3)
//!
//! ============================ WIP — DO NOT COMPILE AS-IS ====================
//! This file is a DROP-IN STARTING POINT for the implementer. It will NOT
//! build until PR3 lands:
//!   - `kiln_tensor::vulkan_matmul` must be implemented + re-exported
//!     (PR3 §4.1).
//!   - `MatmulOp::vulkan_fwd` must exist (PR3 §4.2) so `ops::matmul` routes
//!     to Vulkan.
//!   - `Tensor::from_vec_on(Device::Vulkan(0), ..)` and
//!     `to_device(Device::Cpu)` for Vulkan exist as of PR2 — those parts
//!     are real today.
//!
//! Intended final location:
//!     crates/kiln-tensor/tests/vulkan_matmul_parity.rs
//!
//! Build/run (BOUNDED — named test target only, never the full suite):
//!     CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target \
//!       cargo test -p kiln-tensor --features vulkan --test vulkan_matmul_parity
//!
//! Every device test gates behind `vulkan()` and PASSES (returns early) when
//! no Vulkan device is present — same idiom as `vk_matmul_parity.rs:26` and
//! `metal_ops_parity.rs`'s `metal()`. No training loop, no model, no
//! unbounded work: each test issues at most a handful of single GEMM
//! dispatches over small shapes.
//! ===========================================================================

#![cfg(feature = "vulkan")]
// Remove this once PR3 is implemented and the tests are wired to the real
// `ops::matmul` Vulkan path. Until then we mark the whole module ignored so
// CI does not attempt to run an unimplemented path.
#![allow(dead_code, unused_imports)]

use kiln_tensor::{ops, DType, Device, Tensor};

// ---------------------------------------------------------------------------
// Device probe + helpers (mirror metal_ops_parity.rs / vk_matmul_parity.rs)
// ---------------------------------------------------------------------------

/// `Device::Vulkan(0)` if a Vulkan device is reachable, else `None`.
/// `primary_vulkan_device(0)` is the PR2 cached-device entry point; Ok ⇒
/// a device exists.
fn vulkan() -> Option<Device> {
    kiln_tensor::primary_vulkan_device(0).ok().map(|_| Device::Vulkan(0))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0_f32, |m, &x| m.max(x.abs()))
}

/// Deterministic, smooth, bounded test pattern (no RNG; reproducible).
fn pattern(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as u64 + seed) as f32) * 0.013).sin() * 0.5)
        .collect()
}

/// Naive F32 reference GEMM `[M,K] @ [K,N] -> [M,N]` (single batch).
/// This is the same canonical reference `MatmulOp::cpu_fwd` computes; we
/// keep a local copy so the test does not depend on CPU-path internals.
fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for mi in 0..m {
        for ki in 0..k {
            let av = a[mi * k + ki];
            for ni in 0..n {
                c[mi * n + ni] += av * b[ki * n + ni];
            }
        }
    }
    c
}

/// K-scaled F32 acceptance threshold (PR3 §6). F32-in/F32-out, so much
/// tighter than the Metal BF16 gate.
fn matmul_threshold(k: usize) -> f32 {
    if k <= 64 {
        1e-5
    } else if k <= 4096 {
        1e-4
    } else {
        1e-3
    }
}

/// Read a Vulkan (or CPU) tensor back to a flat f32 Vec via the PR2 D2H copy.
fn to_vec_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.to_device(Device::Cpu).expect("to_device(Cpu)");
    cpu.to_vec::<f32>().expect("to_vec::<f32>")
}

// ---------------------------------------------------------------------------
// 2-D forward parity at real LoRA shapes (PR3 §7.2, §7.4)
// ---------------------------------------------------------------------------

#[test]
fn vulkan_matmul_f32_parity() {
    let Some(dev) = vulkan() else {
        eprintln!("no Vulkan device; skipping vulkan_matmul_f32_parity");
        return;
    };
    // (M, K, N): LoRA x@A.T and h@B.T, plus tile-boundary + Qwen-K cases.
    let cases = [
        (16usize, 64usize, 8usize), // x@A.T : batch=16, in=64, rank=8
        (16, 8, 64),                // h@B.T : rank=8, out=64
        (17, 33, 19),               // non-tile-aligned (kernel tile masking)
        (1, 2560, 16),              // M=1 decode-ish, Qwen K
        (8, 2560, 16),              // small-batch, Qwen K
        (4, 128, 4096),             // wide-N out-projection
    ];
    for (m, k, n) in cases {
        let a = pattern(m * k, 100 + m as u64);
        let b = pattern(k * n, 200 + n as u64);

        let a_cpu = Tensor::from_vec(a.clone(), vec![m, k]).unwrap();
        let b_cpu = Tensor::from_vec(b.clone(), vec![k, n]).unwrap();
        let a_vk = Tensor::from_vec_on(dev, a.clone(), vec![m, k]).unwrap();
        let b_vk = Tensor::from_vec_on(dev, b.clone(), vec![k, n]).unwrap();

        let want_t = ops::matmul(&a_cpu, &b_cpu).unwrap();
        let got_t = ops::matmul(&a_vk, &b_vk).unwrap();

        // Result must stay on Vulkan (proves zero-copy on-device GEMM, no
        // accidental host fallback). Mirrors metal_ops_parity.rs:583.
        assert_eq!(got_t.device(), dev, "vulkan matmul must stay on Vulkan");
        assert_eq!(got_t.shape().to_vec(), vec![m, n], "output shape");

        let want = to_vec_f32(&want_t);
        let got = to_vec_f32(&got_t);
        let d = max_abs_diff(&got, &want);
        let tol = matmul_threshold(k);
        let rel_ok = d < 1e-4 * max_abs(&want).max(1.0);
        assert!(
            d < tol || (k > 4096 && rel_ok),
            "matmul [{m},{k}]x[{k},{n}] max|Δ|={d} (tol {tol}, refmax {})",
            max_abs(&want)
        );
    }
}

// ---------------------------------------------------------------------------
// Batched forward parity, incl. the leading-dim flatten path (PR3 §4.2, §7.2)
// ---------------------------------------------------------------------------

#[test]
fn vulkan_matmul_batched_f32_parity() {
    let Some(dev) = vulkan() else {
        eprintln!("no Vulkan device; skipping vulkan_matmul_batched_f32_parity");
        return;
    };
    // (batch_dims, M, K, N) — last case has TWO leading dims (rank-4),
    // exercising the flatten-to-rank-3 path in vulkan_matmul.
    let cases: [(Vec<usize>, usize, usize, usize); 3] = [
        (vec![2], 8, 64, 16),
        (vec![3], 17, 33, 19),
        (vec![2, 2], 4, 32, 8),
    ];
    for (bdims, m, k, n) in cases {
        let batch: usize = bdims.iter().product();
        let a = pattern(batch * m * k, 300 + m as u64);
        let b = pattern(batch * k * n, 400 + n as u64);

        let mut a_shape = bdims.clone();
        a_shape.extend_from_slice(&[m, k]);
        let mut b_shape = bdims.clone();
        b_shape.extend_from_slice(&[k, n]);

        let a_cpu = Tensor::from_vec(a.clone(), a_shape.clone()).unwrap();
        let b_cpu = Tensor::from_vec(b.clone(), b_shape.clone()).unwrap();
        let a_vk = Tensor::from_vec_on(dev, a.clone(), a_shape.clone()).unwrap();
        let b_vk = Tensor::from_vec_on(dev, b.clone(), b_shape.clone()).unwrap();

        let want_t = ops::matmul(&a_cpu, &b_cpu).unwrap();
        let got_t = ops::matmul(&a_vk, &b_vk).unwrap();

        assert_eq!(got_t.device(), dev, "batched vulkan matmul stays on Vulkan");
        let mut out_shape = bdims.clone();
        out_shape.extend_from_slice(&[m, n]);
        assert_eq!(got_t.shape().to_vec(), out_shape, "batched output shape");

        let want = to_vec_f32(&want_t);
        let got = to_vec_f32(&got_t);
        let d = max_abs_diff(&got, &want);
        let tol = matmul_threshold(k);
        assert!(d < tol, "batched matmul {bdims:?} [{m},{k}]x[{k},{n}] max|Δ|={d} (tol {tol})");
    }
}

// ---------------------------------------------------------------------------
// LoRA composition parity: delta = (x @ A.T) @ B.T  (PR3 §7.4)
// Uses ops::matmul end-to-end on Vulkan against the naive CPU reference.
// FD is not needed for the forward; backward is out of PR3 scope
// (MatmulOp::bwd is None). This is a forward-composition parity check.
// ---------------------------------------------------------------------------

#[test]
fn vulkan_lora_delta_forward_parity() {
    let Some(dev) = vulkan() else {
        eprintln!("no Vulkan device; skipping vulkan_lora_delta_forward_parity");
        return;
    };
    let batch = 16usize;
    let in_features = 64usize;
    let rank = 8usize;
    let out_features = 64usize;

    let x = pattern(batch * in_features, 11);
    let a = pattern(rank * in_features, 22); // A: [rank, in]
    let b = pattern(out_features * rank, 33); // B: [out, rank]

    // ---- reference: (x @ A.T) @ B.T on CPU via naive ----
    let mut a_t = vec![0.0_f32; in_features * rank]; // [in, rank]
    for r in 0..rank {
        for c in 0..in_features {
            a_t[c * rank + r] = a[r * in_features + c];
        }
    }
    let mut b_t = vec![0.0_f32; rank * out_features]; // [rank, out]
    for r in 0..out_features {
        for c in 0..rank {
            b_t[c * out_features + r] = b[r * rank + c];
        }
    }
    let h = naive_matmul(&x, &a_t, batch, rank, in_features);
    let want = naive_matmul(&h, &b_t, batch, out_features, rank);

    // ---- Vulkan via ops::matmul (transposes materialized on host, then
    // uploaded — PR3 matmul is contiguous-only so we feed packed A.T / B.T) ----
    let x_vk = Tensor::from_vec_on(dev, x, vec![batch, in_features]).unwrap();
    let a_t_vk = Tensor::from_vec_on(dev, a_t, vec![in_features, rank]).unwrap();
    let b_t_vk = Tensor::from_vec_on(dev, b_t, vec![rank, out_features]).unwrap();

    let h_vk = ops::matmul(&x_vk, &a_t_vk).unwrap();
    assert_eq!(h_vk.device(), dev);
    let delta_vk = ops::matmul(&h_vk, &b_t_vk).unwrap();
    assert_eq!(delta_vk.device(), dev, "LoRA delta stays on Vulkan");
    assert_eq!(delta_vk.shape().to_vec(), vec![batch, out_features]);

    let got = to_vec_f32(&delta_vk);
    let d = max_abs_diff(&got, &want);
    // Two chained matmuls, K=64 then K=8 → small; tight bound.
    assert!(d < 1e-4, "LoRA delta forward max|Δ|={d}");
}

// ---------------------------------------------------------------------------
// Zero-offset guard (GOTCHA #2 / PR3 §7.2) — IGNORED until the op's chosen
// offset behavior is pinned. Implementer: keep ONE of the two asserts.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "WIP PR3: pin the start_offset!=0 behavior (reject vs auto-contiguous) before enabling"]
fn vulkan_matmul_nonzero_offset_behavior() {
    let Some(dev) = vulkan() else { return };
    let m = 4;
    let k = 4;
    let n = 4;
    // Build a [2*m, k] tensor on Vulkan, narrow rows [m..2m) so the view is
    // contiguous BUT carries start_offset = m*k.
    let a_full = Tensor::from_vec_on(dev, pattern(2 * m * k, 7), vec![2 * m, k]).unwrap();
    let a_view = a_full.narrow(0, m, m).unwrap(); // contiguous, start_offset=m*k
    let b = Tensor::from_vec_on(dev, pattern(k * n, 9), vec![k, n]).unwrap();

    let res = ops::matmul(&a_view, &b);
    // EITHER (reject): the bridge errors on nonzero offset.
    //   assert!(res.is_err(), "zero-copy bridge must reject nonzero start_offset");
    // OR (auto-contiguous): the op materializes a zero-offset buffer first
    // and the result is correct vs naive on the narrowed rows.
    let got_t = res.expect("auto-contiguous path");
    let a_rows: Vec<f32> = {
        let full = to_vec_f32(&a_full);
        full[m * k..2 * m * k].to_vec()
    };
    let want = naive_matmul(&a_rows, &to_vec_f32(&b), m, n, k);
    let d = max_abs_diff(&to_vec_f32(&got_t), &want);
    assert!(d < 1e-5, "narrowed matmul max|Δ|={d}");
}

// ---------------------------------------------------------------------------
// Zero-size output (PR3 §7.2)
// ---------------------------------------------------------------------------

#[test]
fn vulkan_matmul_zero_size() {
    let Some(dev) = vulkan() else { return };
    let k = 8;
    let n = 4;
    let a = Tensor::from_vec_on(dev, Vec::<f32>::new(), vec![0, k]).unwrap();
    let b = Tensor::from_vec_on(dev, pattern(k * n, 5), vec![k, n]).unwrap();
    let got = ops::matmul(&a, &b).unwrap();
    assert_eq!(got.shape().to_vec(), vec![0, n], "empty matmul output shape");
    assert_eq!(got.device(), dev);
}

// ---------------------------------------------------------------------------
// Bridge byte-exact round-trip (GOTCHA #1 / PR3 §7.3) — IGNORED until the
// bridge fns are reachable from a test. Two options for the implementer:
//   (a) make vk_tensor_from_kt_storage / kt_tensor_from_vk `pub` under the
//       vulkan feature, or
//   (b) exercise the bridge transitively (an identity-ish op) and assert the
//       readback equals the upload.
// As written this uses an identity matmul (A @ I == A) to drive the bridge
// end-to-end WITHOUT widening visibility — flip off `#[ignore]` once
// `MatmulOp::vulkan_fwd` lands.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "WIP PR3: enable once MatmulOp::vulkan_fwd lands (drives bridge via identity matmul)"]
fn bridge_roundtrip_preserves_bytes_via_identity() {
    let Some(dev) = vulkan() else { return };
    let m = 5;
    let k = 5;
    let data = pattern(m * k, 13);
    // Identity [k, k].
    let mut ident = vec![0.0_f32; k * k];
    for i in 0..k {
        ident[i * k + i] = 1.0;
    }
    let a = Tensor::from_vec_on(dev, data.clone(), vec![m, k]).unwrap();
    let id = Tensor::from_vec_on(dev, ident, vec![k, k]).unwrap();
    let out = ops::matmul(&a, &id).unwrap(); // A @ I == A
    assert_eq!(out.device(), dev);
    let got = to_vec_f32(&out);
    // A @ I is exact in F32 (sum of one nonzero product per output element).
    let d = max_abs_diff(&got, &data);
    assert!(d == 0.0 || d < 1e-6, "A@I round-trip max|Δ|={d}");
}

// ---------------------------------------------------------------------------
// BF16 must not error on Vulkan — host fallback (PR3 §3 resolution A, §7.1).
// Pure-CPU-input variant so it runs even on machines without a GPU: a BF16
// CPU tensor under the `vulkan` feature must still matmul correctly via the
// op's dtype gate + CPU path. The on-Vulkan-storage BF16 fallback is the
// IGNORED device variant below.
// ---------------------------------------------------------------------------

#[test]
fn vulkan_feature_bf16_cpu_matmul_still_works() {
    // No device needed: CPU BF16 tensors exercise the dtype gate / CPU path.
    let a = ops::cast(
        &Tensor::from_vec(pattern(2 * 3, 1), vec![2, 3]).unwrap(),
        DType::BF16,
    )
    .unwrap();
    let b = ops::cast(
        &Tensor::from_vec(pattern(3 * 2, 2), vec![3, 2]).unwrap(),
        DType::BF16,
    )
    .unwrap();
    let c = ops::matmul(&a, &b).unwrap();
    assert_eq!(c.shape().to_vec(), vec![2, 2]);
    assert_eq!(c.dtype(), DType::BF16);
}

#[test]
#[ignore = "WIP PR3: enable once vulkan_fwd's BF16 host fallback is wired (must not error)"]
fn vulkan_storage_bf16_matmul_host_fallback() {
    let Some(dev) = vulkan() else { return };
    // BF16 data uploaded to Vulkan storage; matmul must NOT error — it host-
    // bounces to the CPU reference and returns to Vulkan (PR3 §3 res. A).
    let a = ops::cast(
        &Tensor::from_vec_on(dev, pattern(2 * 3, 1), vec![2, 3]).unwrap(),
        DType::BF16,
    )
    .unwrap();
    let b = ops::cast(
        &Tensor::from_vec_on(dev, pattern(3 * 2, 2), vec![3, 2]).unwrap(),
        DType::BF16,
    )
    .unwrap();
    let c = ops::matmul(&a, &b).expect("BF16 Vulkan matmul must not error");
    assert_eq!(c.shape().to_vec(), vec![2, 2]);
    // Device may be Vulkan (returned post-fallback) — assert it does not panic
    // and shape/dtype are right; exact device depends on the chosen wiring.
}
