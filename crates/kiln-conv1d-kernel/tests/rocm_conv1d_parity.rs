//! Phase R.7 — CPU-vs-ROCm parity for the vendored `causal_conv1d_update` /
//! `causal_conv1d_prefill` kernels.
//!
//! These kernels are per-thread (one thread per (batch, channel)) with a small
//! `__shared__` entry-state cache in the prefill path — NO cross-lane (warp
//! shuffle) reductions, so they are not a wave-size hazard in the strict sense.
//! But the update launcher tiles the channel axis into 64-thread blocks
//! (`grid.y = ceil(C / 64)`) and the prefill launcher picks blockDim from
//! seq_len, so we sweep the channel count `C` and the prefill `seq_len` across
//! the 32/64-lane boundary widths {31,32,33,63,64,65,127,128,129,256,1024} to
//! catch any tiling / shared-memory boundary bug. Compared to a scalar CPU
//! reference of the exact dot-product + state-update + SiLU the kernel computes.
//!
//! Skips when no ROCm device is present.
//!
//! ## Serialization
//!
//! Both `#[test]`s launch conv kernels on `Device::Rocm(0)`'s default stream and
//! then `rocm_to_host_copy` the results back. cargo's default harness runs the
//! two tests on separate threads concurrently; sharing the one device stream
//! across threads lets one test's host read-back observe a still-zeroed (freshly
//! allocated) output buffer of the other — a "got 0" data hazard, not a kernel
//! bug (each test passes in isolation and with `--test-threads=1`). We guard the
//! GPU section with a process-wide mutex so the device work is serialized while
//! the (cheap) CPU reference + asserts still run unsynchronized.
//!
//! Run: `cargo test -p kiln-conv1d-kernel --features rocm --test rocm_conv1d_parity`
#![cfg(feature = "rocm")]

use std::sync::{Mutex, MutexGuard};

use half::bf16;

use kiln_conv1d_kernel::{causal_conv1d_prefill_kt, causal_conv1d_update_kt};
use kiln_tensor::{Device, Tensor};

const KW: usize = 4; // the compiled specialisation

/// Process-wide lock serializing the device-touching tests so concurrent
/// harness threads don't race on `Device::Rocm(0)`'s shared default stream.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the GPU serialization lock, recovering from a poisoned mutex (a prior
/// test panicking inside the guarded section must not cascade into spurious
/// failures for the other test).
fn gpu_guard() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.7 conv1d parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-1, 1) for index i+seed.
fn val(i: usize, seed: u64) -> f32 {
    let mut s = (i as u64)
        .wrapping_add(seed)
        .wrapping_mul(0x9E3779B97F4A7C15);
    s ^= s >> 29;
    ((s % 2048) as f32 - 1024.0) / 1024.0
}

fn bf16_vec(n: usize, seed: u64) -> Vec<bf16> {
    (0..n).map(|i| bf16::from_f32(val(i, seed))).collect()
}

/// bf16 round-trip a host f32 the same way the kernel sees it (the device
/// stores bf16; the kernel widens to f32). Reference math must start from the
/// already-rounded value to match the kernel bit-for-bit (within tol).
fn rt(x: bf16) -> f32 {
    x.to_f32()
}

fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    let host = kiln_tensor::rocm_to_host_copy(t).expect("rocm_to_host_copy");
    host.to_vec::<f32>().expect("to_vec f32")
}

/// One (batch, channel) tiling boundary sweep of `causal_conv1d_update_kt`.
#[test]
fn conv1d_update_parity_channel_sweep() {
    if no_rocm() {
        return;
    }
    // Serialize device work against the sibling prefill test (see module docs).
    let _gpu = gpu_guard();
    let channel_widths = [1usize, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024];
    let batch = 2usize;

    for &c in &channel_widths {
        // x: bf16 [B, C, 1]
        let x_h = bf16_vec(batch * c, 11);
        // weight: bf16 [C, K]
        let w_h = bf16_vec(c * KW, 22);
        // conv_state: f32 [B, C, K-1]
        let cs_h: Vec<f32> = (0..batch * c * (KW - 1)).map(|i| val(i, 33)).collect();

        let x = Tensor::from_vec_on(Device::Rocm(0), x_h.clone(), vec![batch, c, 1])
            .unwrap_or_else(|e| panic!("x from_vec_on (C={c}): {e}"));
        let w = Tensor::from_vec_on(Device::Rocm(0), w_h.clone(), vec![c, KW])
            .unwrap_or_else(|e| panic!("w from_vec_on (C={c}): {e}"));
        let cs = Tensor::from_vec_on(Device::Rocm(0), cs_h.clone(), vec![batch, c, KW - 1])
            .unwrap_or_else(|e| panic!("cs from_vec_on (C={c}): {e}"));

        let out = causal_conv1d_update_kt(&x, &w, &cs, KW)
            .unwrap_or_else(|e| panic!("causal_conv1d_update_kt (C={c}): {e}"));
        assert_eq!(out.shape(), &[batch, c, 1], "out shape (C={c})");
        let got = host_f32(&out);

        // CPU reference: window = [state[0], state[1], state[2], x], dot with
        // the K weights, then SiLU. (silu=1 in the kt wrapper.)
        let mut reference = Vec::with_capacity(batch * c);
        // Updated state reference for the in-place check below.
        let mut ref_state = Vec::with_capacity(batch * c * (KW - 1));
        for b in 0..batch {
            for ch in 0..c {
                let srow = (b * c + ch) * (KW - 1);
                let mut window = [0.0f32; KW];
                window[..(KW - 1)].copy_from_slice(&cs_h[srow..srow + (KW - 1)]);
                window[KW - 1] = rt(x_h[b * c + ch]);
                let mut acc = 0.0f32;
                for i in 0..KW {
                    acc += window[i] * rt(w_h[ch * KW + i]);
                }
                reference.push(silu(acc));
                // new state: drop oldest, append newest.
                for i in 0..KW - 1 {
                    ref_state.push(window[i + 1]);
                }
            }
        }

        assert_eq!(got.len(), reference.len(), "len (C={c})");
        for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - r).abs();
            assert!(
                diff <= 3e-3 + 3e-3 * r.abs(),
                "update mismatch at C={c} idx={i}: got {g} ref {r} diff {diff}"
            );
        }

        // conv_state is mutated in place — verify the updated state too.
        let got_state = host_f32(&cs);
        assert_eq!(got_state.len(), ref_state.len(), "state len (C={c})");
        for (i, (g, r)) in got_state.iter().zip(ref_state.iter()).enumerate() {
            let diff = (g - r).abs();
            assert!(
                diff <= 3e-3 + 3e-3 * r.abs(),
                "state mismatch at C={c} idx={i}: got {g} ref {r} diff {diff}"
            );
        }
    }
    eprintln!("conv1d_update CPU-vs-ROCm parity passed across channel widths {channel_widths:?}");
}

/// `causal_conv1d_prefill_kt` parity across a seq_len boundary sweep (the
/// prefill launcher's blockDim is chosen from seq_len, so the block-stride loop
/// + shared entry-state must be correct at the 32/64/128/256 boundaries).
#[test]
fn conv1d_prefill_parity_seqlen_sweep() {
    if no_rocm() {
        return;
    }
    // Serialize device work against the sibling update test (see module docs).
    let _gpu = gpu_guard();
    let seq_lens = [2usize, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024];
    let batch = 2usize;
    let channels = 5usize;

    for &t in &seq_lens {
        let x_h = bf16_vec(batch * channels * t, 101);
        let w_h = bf16_vec(channels * KW, 202);
        let cs_h: Vec<f32> = (0..batch * channels * (KW - 1))
            .map(|i| val(i, 303))
            .collect();

        let x = Tensor::from_vec_on(Device::Rocm(0), x_h.clone(), vec![batch, channels, t])
            .unwrap_or_else(|e| panic!("x from_vec_on (T={t}): {e}"));
        let w = Tensor::from_vec_on(Device::Rocm(0), w_h.clone(), vec![channels, KW])
            .unwrap_or_else(|e| panic!("w from_vec_on (T={t}): {e}"));
        let cs = Tensor::from_vec_on(Device::Rocm(0), cs_h.clone(), vec![batch, channels, KW - 1])
            .unwrap_or_else(|e| panic!("cs from_vec_on (T={t}): {e}"));

        let out = causal_conv1d_prefill_kt(&x, &w, &cs, KW)
            .unwrap_or_else(|e| panic!("causal_conv1d_prefill_kt (T={t}): {e}"));
        assert_eq!(out.shape(), &[batch, channels, t], "out shape (T={t})");
        let got = host_f32(&out);

        // CPU reference: for each (b, ch), the causal conv over the padded
        // sequence [entry_state(K-1) || x(0..T)], window of width K ending at t.
        let mut reference = Vec::with_capacity(batch * channels * t);
        for b in 0..batch {
            for ch in 0..channels {
                let bc = b * channels + ch;
                let srow = bc * (KW - 1);
                let entry: [f32; KW - 1] = {
                    let mut e = [0.0f32; KW - 1];
                    e.copy_from_slice(&cs_h[srow..srow + (KW - 1)]);
                    e
                };
                let mut wrow = [0.0f32; KW];
                for i in 0..KW {
                    wrow[i] = rt(w_h[ch * KW + i]);
                }
                for ti in 0..t {
                    let mut acc = 0.0f32;
                    for (j, &w) in wrow.iter().enumerate().take(KW) {
                        let padded = ti + j;
                        let v = if padded < KW - 1 {
                            entry[padded]
                        } else {
                            rt(x_h[bc * t + (padded - (KW - 1))])
                        };
                        acc += v * w;
                    }
                    reference.push(silu(acc));
                }
            }
        }

        assert_eq!(got.len(), reference.len(), "len (T={t})");
        for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - r).abs();
            assert!(
                diff <= 3e-3 + 3e-3 * r.abs(),
                "prefill mismatch at T={t} idx={i}: got {g} ref {r} diff {diff}"
            );
        }
    }
    eprintln!("conv1d_prefill CPU-vs-ROCm parity passed across seq_lens {seq_lens:?}");
}
