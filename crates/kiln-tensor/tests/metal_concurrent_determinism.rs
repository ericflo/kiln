#![cfg(feature = "metal")]

//! Concurrent-determinism gate for the Metal substrate (#1082).
//!
//! The SAME deterministic Metal computation MUST produce byte-identical
//! F32 results whether run single-threaded or concurrently from many
//! threads. This regression-tests the cross-thread command-buffer-pool
//! race in `metal_rt::commands` (a host read of tensor T racing another
//! thread's deferred-commit / global flush, so the read can observe T's
//! output buffer before the GPU write that produced it has completed).
//!
//! Each thread runs a fixed, deterministic op chain (matmul -> softmax,
//! plus a couple of elementwise ops) and reads the result back to host.
//! Every thread / iteration must match a single-threaded reference bit
//! for bit. Run with `--test-threads=8` to maximize pool contention.
//!
//! Skips gracefully when no Metal device is present.

use std::sync::Arc;
use std::thread;

use kiln_tensor::{Device, Tensor, ops};

fn metal() -> Option<Device> {
    kiln_tensor::primary_metal_companion(0)
        .ok()
        .map(|_| Device::Metal(0))
}

/// Deterministic pseudo-random f32 pattern in roughly [-1, 1].
fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for _ in 0..n {
        s = s
            .wrapping_add(0xDEAD_BEEF)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        out.push(((s >> 33) as u32 % 2048) as f32 / 1024.0 - 1.0);
    }
    out
}

/// A fixed, deterministic Metal compute that **chains** many ops on-GPU
/// with NO intermediate host reads — exactly the pattern training/inference
/// forward passes use. Only the final result is read back to host.
///
/// Each op in the chain consumes the previous op's freshly-written output
/// buffer as its input. With the deferred-commit command-buffer pool, the
/// producer and consumer can land on DIFFERENT pool entries (different
/// `MTLCommandBuffer`s). Metal serializes command buffers on one queue in
/// COMMIT order, but the pool's deferred commit can order the consumer's
/// buffer before the producer's under cross-thread contention — so the
/// consumer reads a stale / not-yet-written input. That makes the final
/// result diverge non-deterministically. Bit-for-bit reproducible when the
/// dependency ordering is correct.
fn metal_compute(dev: Device) -> Vec<f32> {
    let m = 64usize;
    let k = 64usize;
    let n = 64usize;
    let a = pattern(m * k, 1);
    let b = pattern(k * n, 2);
    let w = b_row(dev, n, 0);

    let a_met = Tensor::from_vec_on(dev, a, vec![m, k]).unwrap();
    let b_met = Tensor::from_vec_on(dev, b, vec![k, n]).unwrap();

    // Long on-GPU dependency chain, no host reads until the very end:
    //   x0 = A @ B
    //   x1 = softmax(x0)        (consumes x0's buffer)
    //   x2 = rmsnorm(x1, w)     (consumes x1's buffer)
    //   x3 = x2 @ B             (consumes x2's buffer)
    //   ... repeated ...
    let mut x = ops::matmul(&a_met, &b_met).unwrap();
    for _round in 0..24 {
        let sm = kiln_tensor::metal_softmax_last_axis(&x).unwrap();
        let rms = kiln_tensor::metal_rmsnorm_last_axis(&sm, &w, 1e-5).unwrap();
        x = ops::matmul(&rms, &b_met).unwrap();
    }
    let sm = kiln_tensor::metal_softmax_last_axis(&x).unwrap();
    sm.to_vec::<f32>().unwrap()
}

/// A deterministic strictly-positive weight row of length `n` for rmsnorm.
fn b_row(dev: Device, n: usize, seed: u64) -> Tensor {
    let w: Vec<f32> = pattern(n, 7 + seed)
        .into_iter()
        .map(|v| v.abs() + 0.25)
        .collect();
    Tensor::from_vec_on(dev, w, vec![n]).unwrap()
}

/// Bit-identical comparison (no epsilon — determinism, not parity).
fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn concurrent_metal_results_are_deterministic() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };

    // Single-threaded reference, computed twice to confirm the op chain is
    // itself deterministic before we go concurrent.
    let reference = metal_compute(dev);
    let reference2 = metal_compute(dev);
    assert!(
        bits_eq(&reference, &reference2),
        "single-threaded compute is not even self-deterministic"
    );

    let reference = Arc::new(reference);
    const THREADS: usize = 8;
    const ITERS: usize = 50;

    let mut handles = Vec::new();
    for tid in 0..THREADS {
        let reference = Arc::clone(&reference);
        handles.push(thread::spawn(move || {
            for it in 0..ITERS {
                let got = metal_compute(dev);
                if !bits_eq(&reference, &got) {
                    // Find the first divergent index for a useful message.
                    let mut first = None;
                    for (i, (r, g)) in reference.iter().zip(&got).enumerate() {
                        if r.to_bits() != g.to_bits() {
                            first = Some((i, *r, *g));
                            break;
                        }
                    }
                    return Err(format!(
                        "thread {tid} iter {it}: result diverged from single-threaded \
                         reference (len {} vs {}); first diff {:?}",
                        reference.len(),
                        got.len(),
                        first
                    ));
                }
            }
            Ok(())
        }));
    }

    let mut failures = Vec::new();
    for h in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => failures.push(e),
            Err(_) => failures.push("thread panicked".to_string()),
        }
    }
    assert!(
        failures.is_empty(),
        "concurrent Metal results diverged:\n{}",
        failures.join("\n")
    );
}
