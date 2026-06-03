//! Phase R.8 — CPU-vs-ROCm parity for the W4A16 Marlin GEMM composite.
//!
//! Marlin's CUDA GEMM is inline `mma.sync` PTX that cannot be hipified, so the
//! ROCm lane (`marlin_w4a16_gemm_kt` under `--features rocm`) is a
//! correctness-first composite: it reads the packed int4 weights + per-group
//! scales back to host, dequantizes them to a dense F16 `[k, n]` weight (the
//! exact inverse of `pack::quantize_and_pack`), uploads it, and runs a plain
//! dense `rocm_matmul`.
//!
//! This test builds a random fp32 weight, quantizes+packs it on the host (the
//! same packer the model uses), uploads the F16 activations / U32 packed
//! weights / F16 permuted scales to `Device::Rocm(0)`, dispatches the
//! composite, and compares the F16 result against a CPU reference matmul of
//! `a_f16 @ dequant_weight` — where `dequant_weight` is the *exact* round-trip
//! the packer reports. The tolerance is a bf16-ish accumulation band (the
//! matmul accumulates k F16 products; the CPU reference accumulates the same
//! products in f32, so we allow an rtol that scales with k).
//!
//! We sweep both Marlin group sizes (`128` and per-column `-1`) and a couple of
//! Marlin-legal shapes (k%128==0, n%256==0). Skips when no ROCm device.
//!
//! Run: `cargo test -p kiln-marlin-gemm --no-default-features --features rocm \
//!       --test rocm_marlin_parity`
//! and  `KILN_ROCM_WAVE64=1 cargo test -p kiln-marlin-gemm \
//!       --no-default-features --features rocm --test rocm_marlin_parity`
#![cfg(feature = "rocm")]

use std::sync::{Mutex, MutexGuard};

use half::f16;

use kiln_marlin_gemm::{marlin_w4a16_gemm_kt, pack};
use kiln_tensor::{DType, Device, Tensor};

/// Process-wide lock so concurrent harness threads don't race on
/// `Device::Rocm(0)`'s shared default stream (same hazard the conv1d parity
/// test documents).
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_guard() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.8 marlin parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-0.5, 0.5).
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fffffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

/// Reference row-major matmul C[m,n] = sum_k A[m,k] * B[k,n], f32 accumulate,
/// with both operands taken through f16 first so the reference sees the same
/// rounded inputs the GPU GEMM does.
fn cpu_matmul_f16(a: &[f16], b: &[f16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn run_case(m: usize, k: usize, n: usize, groupsize: i64) {
    // Random weight [k, n] and activations [m, k].
    let mut wstate = 0xC0FFEE_5EED_u64 ^ ((k as u64) << 17) ^ ((n as u64) << 3);
    let mut weight = vec![0.0f32; k * n];
    for v in weight.iter_mut() {
        *v = lcg(&mut wstate) * 0.5;
    }
    let mut astate = 0xDEAD_BEEF_u64 ^ ((m as u64) << 11) ^ (groupsize as u64);
    let mut acts = vec![0.0f32; m * k];
    for v in acts.iter_mut() {
        *v = lcg(&mut astate) * 0.25;
    }

    // Host pack — the exact packer the model uses. `dequant_f32` is the
    // round-tripped weight the GEMM effectively multiplies against.
    let (b_packed_i32, scales_f16, dequant_f32) =
        pack::quantize_and_pack(&weight, k, n, groupsize);
    let b_packed_u32: Vec<u32> = b_packed_i32.iter().map(|&x| x as u32).collect();

    // Activations as F16 (the kernel's native dtype).
    let acts_f16: Vec<f16> = acts.iter().map(|&v| f16::from_f32(v)).collect();
    // Dequantized weight as F16 — exactly what the composite uploads, so the
    // reference and the device share rounded inputs.
    let dequant_f16: Vec<f16> = dequant_f32.iter().map(|&v| f16::from_f32(v)).collect();

    // Upload operands to the ROCm device.
    let a_kt = Tensor::from_vec_on(Device::Rocm(0), acts_f16.clone(), vec![m, k])
        .unwrap_or_else(|e| panic!("a from_vec_on (g={groupsize}): {e}"));
    let b_kt =
        Tensor::from_vec_on(Device::Rocm(0), b_packed_u32, vec![k / 16, n * 16 / 8])
            .unwrap_or_else(|e| panic!("b from_vec_on (g={groupsize}): {e}"));
    let s_rows = if groupsize == -1 {
        1
    } else {
        k / groupsize as usize
    };
    let s_kt = Tensor::from_vec_on(Device::Rocm(0), scales_f16, vec![s_rows, n])
        .unwrap_or_else(|e| panic!("s from_vec_on (g={groupsize}): {e}"));

    // Dispatch the composite.
    let c_kt = marlin_w4a16_gemm_kt(&a_kt, &b_kt, &s_kt, groupsize as i32)
        .unwrap_or_else(|e| panic!("marlin_w4a16_gemm_kt (g={groupsize}): {e}"));
    assert_eq!(c_kt.shape(), &[m, n], "out shape (g={groupsize})");
    assert_eq!(c_kt.dtype(), DType::F16, "out dtype (g={groupsize})");

    let got_f16 = kiln_tensor::rocm_to_host_copy(&c_kt)
        .unwrap_or_else(|e| panic!("rocm_to_host_copy (g={groupsize}): {e}"))
        .to_vec::<f16>()
        .unwrap_or_else(|e| panic!("to_vec f16 (g={groupsize}): {e}"));
    let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();

    // CPU reference: a_f16 @ dequant_f16, f32 accumulate.
    let want = cpu_matmul_f16(&acts_f16, &dequant_f16, m, k, n);

    // F16 GEMM accumulates k products; allow a band that scales with k. The
    // dense rocm_matmul may accumulate in f16 or f32 depending on the
    // hipBLASLt algo, so we use a generous bf16-ish relative tolerance plus a
    // small absolute floor — well within what a W4A16 forward needs.
    assert_eq!(got.len(), want.len(), "len (g={groupsize})");
    let rtol = 2e-2f32;
    let atol = 1e-2f32 + 2e-3f32 * (k as f32).sqrt();
    let mut max_abs_diff = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        assert!(
            diff <= atol + rtol * w.abs(),
            "marlin parity mismatch g={groupsize} m={m} k={k} n={n} idx={i}: \
             got {g} want {w} diff {diff} (atol {atol} rtol {rtol})"
        );
    }
    eprintln!(
        "marlin R.8 parity OK g={groupsize} m={m} k={k} n={n} max_abs_diff={max_abs_diff:.4}"
    );
}

#[test]
fn marlin_w4a16_parity_grouped() {
    if no_rocm() {
        return;
    }
    let _gpu = gpu_guard();
    // Marlin-legal shapes: k%128==0, n%256==0. groupsize=128 needs k%128==0.
    run_case(16, 128, 256, 128);
    run_case(8, 256, 512, 128);
    run_case(1, 256, 256, 128); // decode-skinny M=1
}

#[test]
fn marlin_w4a16_parity_per_column() {
    if no_rocm() {
        return;
    }
    let _gpu = gpu_guard();
    // groupsize == -1 → per-column quantization (one scale row).
    run_case(16, 128, 256, -1);
    run_case(4, 256, 512, -1);
}
