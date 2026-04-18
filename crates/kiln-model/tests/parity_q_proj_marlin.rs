//! Parity test for the Marlin W4A16 q_proj forward path.
//!
//! Builds a synthetic q_proj weight matrix, fake-quantizes it via
//! `kiln_marlin_gemm::pack::quantize_and_pack`, and drives
//! `kiln_model::forward::q_proj_matmul` once with the packed Marlin tuple
//! populated and once with the legacy bf16 pre-transposed weight. The two
//! outputs must agree within the kernel's documented tolerance
//! (`rel_err <= 2%` or `abs_err <= 5e-3`).
//!
//! Shapes match Qwen3.5-4B's q_proj (`hidden=2560`, `num_q_heads=16`,
//! `head_dim=256` → `out=4096`, `group_size=128`) so this exercises exactly
//! the configuration that ships under `KILN_W4A16=1`.
//!
//! The test no-ops when CUDA is unavailable so it does not break
//! `cargo test` on non-CUDA hosts.

#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use kiln_marlin_gemm::pack;
use kiln_model::forward::{q_proj_matmul, GpuFullAttentionWeights};

fn lcg_seed(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fffffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| lcg_seed(&mut state) * scale).collect()
}

fn make_attn_weights_q_only(
    device: &Device,
    k: usize,
    n: usize,
    weight_f32: &[f32],
    b_packed: Vec<i32>,
    scales_f16: Vec<half::f16>,
    dequant_f32: &[f32],
    groupsize: usize,
) -> GpuFullAttentionWeights {
    // Legacy bf16 path uses the dequantized weight (so both paths see the
    // same effective matrix) transposed to [in, out] contiguous.
    let dequant_bf16 = Tensor::from_vec(dequant_f32.to_vec(), (k, n), &Device::Cpu)
        .expect("dequant host tensor")
        .to_dtype(DType::BF16)
        .expect("dequant -> bf16")
        .to_device(device)
        .expect("dequant -> cuda")
        .contiguous()
        .expect("dequant contiguous");

    // The pre-transposed tensor the legacy matmul expects: `[in, out]`.
    // Our synthetic dequant_f32 is already laid out [k=in, n=out], so no
    // transpose is needed.
    let q_proj_t = dequant_bf16.clone();

    // The untransposed q_proj tensor (unused on the Marlin path but the
    // struct requires it). Shape [out, in].
    let weight_oi: Vec<f32> = {
        // Transpose weight [k, n] row-major to [n, k] row-major.
        let mut out = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                out[j * k + i] = weight_f32[i * n + j];
            }
        }
        out
    };
    let q_proj = Tensor::from_vec(weight_oi, (n, k), &Device::Cpu)
        .expect("q_proj host tensor")
        .to_dtype(DType::BF16)
        .expect("q_proj -> bf16")
        .to_device(device)
        .expect("q_proj -> cuda")
        .contiguous()
        .expect("q_proj contiguous");

    // Marlin tuple.
    let b_cuda = Tensor::from_vec(b_packed, (k / 16, n * 16 / 8), &Device::Cpu)
        .expect("b_packed host tensor")
        .to_device(device)
        .expect("b_packed -> cuda");
    let num_groups = k / groupsize;
    let scales_cuda = Tensor::from_vec(scales_f16, (num_groups, n), &Device::Cpu)
        .expect("scales host tensor")
        .to_device(device)
        .expect("scales -> cuda");

    // Placeholder non-q tensors: the q_proj_matmul helper does not read any
    // of these, but the struct must be fully populated. Use small zeros so
    // allocation stays trivial.
    let zero = Tensor::zeros((1,), DType::BF16, device).expect("zero placeholder");

    GpuFullAttentionWeights {
        q_proj,
        k_proj: zero.clone(),
        v_proj: zero.clone(),
        o_proj: zero.clone(),
        q_norm: zero.clone(),
        k_norm: zero.clone(),
        q_proj_t,
        k_proj_t: zero.clone(),
        v_proj_t: zero.clone(),
        o_proj_t: zero,
        q_proj_marlin: Some((b_cuda, scales_cuda)),
    }
}

fn max_abs_and_rel(a: &Tensor, b: &Tensor) -> (f32, f32) {
    let a_f32 = a.to_dtype(DType::F32).expect("a -> f32");
    let b_f32 = b.to_dtype(DType::F32).expect("b -> f32");
    let diff = (&a_f32 - &b_f32)
        .expect("diff sub")
        .abs()
        .expect("abs");
    let max_diff = diff
        .flatten_all()
        .and_then(|t| t.max(0))
        .and_then(|t| t.to_scalar::<f32>())
        .expect("max_diff scalar");
    let base_abs = b_f32
        .abs()
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.max(0))
        .and_then(|t| t.to_scalar::<f32>())
        .expect("baseline abs scalar");
    (max_diff, max_diff / base_abs.max(1e-6))
}

fn run_case(device: &Device, m: usize, k: usize, n: usize, groupsize: usize, label: &str) {
    // Build synthetic weights and activations. Same LCG scheme as the
    // kiln-marlin-gemm parity test so values stay modest and bf16-friendly.
    let weight_f32 = fill(0xC0FFEE_5EED ^ (k as u64) ^ ((n as u64) << 16), k * n, 0.5);
    let acts_f32 = fill(0xDEAD_BEEF ^ (m as u64) ^ ((k as u64) << 24), m * k, 0.25);

    let (b_packed, scales_f16, dequant_f32) =
        pack::quantize_and_pack(&weight_f32, k, n, groupsize as i64);

    let attn = make_attn_weights_q_only(
        device,
        k,
        n,
        &weight_f32,
        b_packed,
        scales_f16,
        &dequant_f32,
        groupsize,
    );

    // Activation tensor shaped [batch, seq_len, hidden] — the helper
    // expects a 3D tensor (dims3). Use batch=1, seq_len=m.
    let x = Tensor::from_vec(acts_f32.clone(), (1, m, k), &Device::Cpu)
        .expect("x host tensor")
        .to_dtype(DType::BF16)
        .expect("x -> bf16")
        .to_device(device)
        .expect("x -> cuda");

    // 1) Marlin path: q_proj_marlin = Some (as set above).
    let y_marlin = q_proj_matmul(&x, &attn, None, 0.0).expect("q_proj_matmul Marlin");
    assert_eq!(y_marlin.dims(), &[1, m, n], "[{label}] Marlin output shape");
    assert_eq!(y_marlin.dtype(), DType::BF16, "[{label}] Marlin output dtype");

    // 2) bf16 baseline: override q_proj_marlin = None on a clone.
    let mut attn_bf16 = attn;
    attn_bf16.q_proj_marlin = None;
    let y_bf16 = q_proj_matmul(&x, &attn_bf16, None, 0.0).expect("q_proj_matmul bf16");
    assert_eq!(y_bf16.dims(), &[1, m, n], "[{label}] bf16 output shape");

    let (max_diff, rel_diff) = max_abs_and_rel(&y_marlin, &y_bf16);
    eprintln!(
        "[{label}] m={m} k={k} n={n} groupsize={groupsize} \
         max_abs_diff={max_diff:.5} rel={rel_diff:.4}"
    );

    // bf16 has ~7e-3 quantum; Marlin's packing + f16 scales + tensor-core
    // accumulate matches the kernel-crate parity test tolerance.
    let rel_tol = 2e-2;
    let abs_tol = 5e-3;
    assert!(
        max_diff < abs_tol || rel_diff < rel_tol,
        "[{label}] max_abs_diff {max_diff:.6} (rel {rel_diff:.4}) exceeds \
         tolerance abs<{abs_tol} or rel<{rel_tol} for m={m}, k={k}, n={n}"
    );
}

#[test]
fn q_proj_marlin_parity_qwen35_4b() {
    let device = match Device::new_cuda(0).ok() {
        Some(d) => d,
        None => {
            eprintln!("CUDA not available, skipping q_proj Marlin parity test");
            return;
        }
    };

    // Qwen3.5-4B q_proj: hidden_size=2560, num_q_heads=16, head_dim=256 -> 4096.
    // group_size=128 (GPTQ / Marlin canonical).
    run_case(&device, 1, 2560, 4096, 128, "q_proj-decode");
    run_case(&device, 16, 2560, 4096, 128, "q_proj-prefill");
}
