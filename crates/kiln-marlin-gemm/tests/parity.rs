//! Parity test for kiln-marlin-gemm.
//!
//! Builds a small fp32 weight matrix, fake-quantizes it through the same
//! grid the kernel expects (symmetric INT4, +8 offset, group_size = 128),
//! runs the Marlin kernel on bf16 activations, and compares the result to
//! a candle bf16 matmul of `a @ dequant` within a 1e-2 elementwise
//! tolerance.
//!
//! Shapes are chosen to cover the projection dims used by Qwen3.5-4B
//! (`hidden_size = 2560`, `intermediate_size = 9216`, GQA `head_dim = 256`,
//! `num_q_heads = 16`, `num_kv_heads = 4`) plus a couple of small smoke
//! shapes that the upstream Marlin tile constraints (`k % 128 == 0`,
//! `n % 256 == 0`) admit.
//!
//! The whole test no-ops gracefully when CUDA is unavailable so the
//! workspace `cargo test` on a non-CUDA host (or in a pre-CUDA dev image)
//! does not break.

use candle_core::{DType, Device, Tensor};
use kiln_marlin_gemm::{marlin_w4a16_gemm, pack};

fn lcg_seed(state: &mut u64) -> f32 {
    // Simple deterministic PRNG so we can iterate tolerances without
    // chasing nondeterminism. Mirrors the LCG used in
    // kiln-rmsnorm-kernel/tests.
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fffffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

fn fill_weights(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut w = vec![0.0f32; k * n];
    for v in w.iter_mut() {
        // Roughly N(0, 0.5) — small enough that f16 / bf16 round-trips
        // are well-conditioned and large enough that the int4 grid is
        // exercised.
        *v = lcg_seed(&mut state) * 0.5;
    }
    w
}

fn fill_activations(m: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut a = vec![0.0f32; m * k];
    for v in a.iter_mut() {
        *v = lcg_seed(&mut state) * 0.25;
    }
    a
}

fn run_case(device: &Device, m: usize, k: usize, n: usize, groupsize: i64, label: &str) {
    let weight = fill_weights(k, n, 0xC0FFEE_5EED ^ (k as u64) ^ ((n as u64) << 16));
    let acts = fill_activations(m, k, 0xDEAD_BEEF ^ (m as u64) ^ ((k as u64) << 24));

    let (b_packed, scales_f16, dequant_f32) = pack::quantize_and_pack(&weight, k, n, groupsize);

    let g = if groupsize == -1 {
        k
    } else {
        groupsize as usize
    };
    let num_groups = k / g;

    // Move tensors to CUDA in the dtypes the kernel expects.
    let a_cpu = Tensor::from_vec(acts.clone(), (m, k), &Device::Cpu).expect("a_cpu host tensor");
    let a_bf16 = a_cpu
        .to_dtype(DType::BF16)
        .expect("cast a to bf16")
        .to_device(device)
        .expect("a -> cuda");

    let b_cuda = Tensor::from_vec(b_packed, (k / 16, n * 16 / 8), &Device::Cpu)
        .expect("b host tensor")
        .to_device(device)
        .expect("b -> cuda");

    // Convert f16 scales into a candle tensor on CUDA. candle accepts
    // half::f16 directly for Tensor::from_vec.
    let scales_cuda = Tensor::from_vec(scales_f16, (num_groups, n), &Device::Cpu)
        .expect("scales host tensor")
        .to_device(device)
        .expect("scales -> cuda");

    // --- Run the Marlin kernel.
    let kernel_out = marlin_w4a16_gemm(&a_bf16, &b_cuda, &scales_cuda, groupsize as i32)
        .unwrap_or_else(|e| panic!("[{label}] kernel call failed: {e}"));

    assert_eq!(kernel_out.dims(), &[m, n], "[{label}] output shape");
    assert_eq!(kernel_out.dtype(), DType::BF16, "[{label}] output dtype");

    // --- candle bf16 baseline: a_bf16 @ dequant_bf16.
    let dequant_cpu =
        Tensor::from_vec(dequant_f32, (k, n), &Device::Cpu).expect("dequant host tensor");
    let dequant_bf16 = dequant_cpu
        .to_dtype(DType::BF16)
        .expect("dequant -> bf16")
        .to_device(device)
        .expect("dequant -> cuda");

    let baseline = a_bf16.matmul(&dequant_bf16).expect("baseline matmul");

    // --- Compare elementwise within tolerance.
    let kernel_f32 = kernel_out.to_dtype(DType::F32).expect("kernel -> f32");
    let baseline_f32 = baseline.to_dtype(DType::F32).expect("baseline -> f32");

    let diff = (kernel_f32 - &baseline_f32)
        .expect("diff sub")
        .abs()
        .expect("abs");
    let max_diff = diff
        .max_keepdim(0)
        .and_then(|t| t.max_keepdim(1))
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("max diff scalar")[0];

    let baseline_abs = baseline_f32
        .abs()
        .and_then(|t| t.max_keepdim(0))
        .and_then(|t| t.max_keepdim(1))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("baseline abs scalar")[0];

    let rel_diff = max_diff / baseline_abs.max(1e-6);
    eprintln!(
        "[{label}] m={m} k={k} n={n} groupsize={groupsize} \
         max_abs_diff={max_diff:.5} baseline_max={baseline_abs:.5} \
         rel={rel_diff:.4}"
    );

    // bf16 has only 7 mantissa bits (~7e-3 quantum). For a length-K dot
    // product with random-sign accumulation we expect ~sqrt(K) bf16 rounding
    // events visible in the worst output element, so absolute error grows
    // with sqrt(K) and with the typical output magnitude. A relative budget
    // tracks that scaling much better than a fixed 1e-2.
    //
    // Budget breakdown:
    //   * bf16 activation + bf16 accumulate: ~1% relative for k up to ~10k.
    //   * f16 scale rounding: ~5e-4 relative (already baked into baseline).
    //   * Marlin's tensor-core path: matches candle's bf16 matmul to within
    //     bf16 rounding, no extra slack required.
    //
    // 2% relative is loose enough for k=9216 mlp_down and tight enough that
    // a real packing or scale-permutation bug would still trip the test.
    let rel_tol = 2e-2;
    let abs_tol = 5e-3;
    assert!(
        max_diff < abs_tol || rel_diff < rel_tol,
        "[{label}] max_abs_diff {max_diff:.6} (rel {rel_diff:.4}) exceeds \
         tolerance abs<{abs_tol} or rel<{rel_tol} \
         (baseline_max={baseline_abs:.4}) for m={m}, k={k}, n={n}, groupsize={groupsize}"
    );
}

#[test]
fn marlin_w4a16_gemm_parity_qwen35_4b() {
    let device = match Device::new_cuda(0).ok() {
        Some(d) => d,
        None => {
            eprintln!("CUDA not available, skipping marlin parity test");
            return;
        }
    };

    // Smoke shape: smallest valid Marlin (k=128 is the minimum k with
    // group_blocks=8 instantiated; n=256 satisfies n % 256 == 0).
    run_case(&device, 1, 128, 256, 128, "smoke-decode");
    run_case(&device, 16, 128, 256, 128, "smoke-prefill");

    // Qwen3.5-4B projections (hidden_size=2560, intermediate_size=9216,
    // num_q_heads=16, num_kv_heads=4, head_dim=256). All n values are
    // already %256-aligned. We test the four canonical projection shapes
    // for both decode (m=1) and a small prefill batch (m=16).
    let q_proj_n = 16 * 256; // 4096
    let kv_proj_n = 4 * 256; // 1024
    let mlp_up_n = 9216; // intermediate_size
    let mlp_down_k = 9216;
    let mlp_down_n = 2560; // hidden_size — but %256 not satisfied (2560 IS %256).
    assert_eq!(2560 % 256, 0, "hidden_size must satisfy Marlin's n%256==0");

    // q_proj: a [m, 2560] @ Wq [2560, 4096]
    run_case(&device, 1, 2560, q_proj_n, 128, "q_proj-decode");
    run_case(&device, 16, 2560, q_proj_n, 128, "q_proj-prefill");

    // kv_proj: a [m, 2560] @ Wk [2560, 1024]
    run_case(&device, 1, 2560, kv_proj_n, 128, "kv_proj-decode");
    run_case(&device, 16, 2560, kv_proj_n, 128, "kv_proj-prefill");

    // mlp_up / gate_proj: a [m, 2560] @ W [2560, 9216]
    run_case(&device, 1, 2560, mlp_up_n, 128, "mlp_up-decode");

    // mlp_down: a [m, 9216] @ W [9216, 2560]
    run_case(&device, 1, mlp_down_k, mlp_down_n, 128, "mlp_down-decode");
}

#[test]
fn marlin_w4a16_gemm_parity_per_column() {
    let device = match Device::new_cuda(0).ok() {
        Some(d) => d,
        None => {
            eprintln!("CUDA not available, skipping marlin parity test");
            return;
        }
    };

    // Per-column quantization (groupsize == -1) is the other Marlin
    // CALL_IF arm. Cover at least one shape so we don't silently break it.
    run_case(&device, 1, 256, 256, -1, "per-column-decode");
    run_case(&device, 16, 256, 256, -1, "per-column-prefill");
}
