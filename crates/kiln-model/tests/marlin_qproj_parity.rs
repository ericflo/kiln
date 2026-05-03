//! Parity test for the Marlin W4A16 q_proj wiring in [`kiln_model::forward::q_proj_forward`].
//!
//! This test builds a synthetic BF16 q_proj weight matched to Qwen3.5-4B's
//! q_proj shape (hidden=2560, num_heads=16, head_dim=256, attn_output_gate,
//! so `n = 16 * 256 * 2 = 8192`), runs one forward with the Marlin path
//! disabled (BF16 baseline via `linear_with_lora_t`) and one with the Marlin
//! path enabled (`MarlinPackedProj` populated), and reports cosine similarity
//! and max-abs-diff.
//!
//! The Marlin kernel is CUDA-only; on non-CUDA builds the test is ignored
//! via `#[cfg_attr(not(feature = "cuda"), ignore)]` so the workspace
//! `cargo test` on a CPU host does not break. On a CUDA build without a
//! visible device we also bail out gracefully to keep CI runnable.
//!
//! Tolerance: cosine similarity >= 0.9995. The tolerance target matches the
//! Phase 6 wiring spec (see PROFILING.md + Phase 6 task description).
//! `kiln-marlin-gemm`'s own parity test already asserts tighter elementwise
//! bounds against a dequant baseline; this test confirms the forward plumbing
//! preserves that fidelity end to end.

#[cfg(feature = "cuda")]
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "cuda")]
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) as u32) & 0x7fff_ffff;
    (bits as f32 / (i32::MAX as f32)) - 0.5
}

#[cfg(feature = "cuda")]
fn random_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    let mut v = vec![0.0f32; len];
    for x in v.iter_mut() {
        *x = lcg(&mut state) * scale;
    }
    v
}

#[cfg(feature = "cuda")]
fn cosine_similarity(a: &Tensor, b: &Tensor) -> f32 {
    let af = a.to_dtype(DType::F32).expect("a -> f32");
    let bf = b.to_dtype(DType::F32).expect("b -> f32");
    let af = af.flatten_all().expect("a flat");
    let bf = bf.flatten_all().expect("b flat");
    let dot = (&af * &bf).expect("dot mul").sum_all().expect("dot sum");
    let na = (&af * &af).expect("a sq").sum_all().expect("a nrm sum");
    let nb = (&bf * &bf).expect("b sq").sum_all().expect("b nrm sum");
    let dot = dot.to_vec0::<f32>().expect("dot scalar");
    let na = na.to_vec0::<f32>().expect("a nrm scalar").sqrt();
    let nb = nb.to_vec0::<f32>().expect("b nrm scalar").sqrt();
    let denom = (na * nb).max(1e-12);
    dot / denom
}

#[cfg(feature = "cuda")]
fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let af = a.to_dtype(DType::F32).expect("a -> f32");
    let bf = b.to_dtype(DType::F32).expect("b -> f32");
    let diff = (&af - &bf).expect("diff").abs().expect("abs");
    let flat = diff.flatten_all().expect("flat");
    let values = flat.to_vec1::<f32>().expect("diff vec");
    values.iter().cloned().fold(0.0_f32, f32::max)
}

#[cfg(feature = "cuda")]
fn run_parity(device: &Device, m: usize, k: usize, n: usize) {
    use kiln_model::forward::q_proj_forward;
    use kiln_model::marlin_proj::MarlinPackedProj;

    // --- Build a synthetic f32 weight of shape [k, n] on host and run the
    // Marlin packer directly. This gives us the matched pair `(dequant, packed)`
    // used by the kernel's own parity test: any residual error between a BF16
    // matmul against `dequant` and the Marlin kernel against `packed` is the
    // kernel's own rounding error, not the INT4 quantization gap.
    let weight_host = random_vec(k * n, 0x9E37_79B1_1234_5678 ^ (k as u64), 0.25);
    let groupsize: i32 = 128;
    let (b_packed_vec, scales_vec, dequant_f32) =
        kiln_marlin_gemm::pack::quantize_and_pack(&weight_host, k, n, groupsize as i64);

    // Baseline q_proj_t is the dequantized weight in [k, n] layout, on device.
    let q_proj_t = Tensor::from_vec(dequant_f32, (k, n), &Device::Cpu)
        .expect("dequant cpu tensor")
        .to_dtype(DType::BF16)
        .expect("dequant -> bf16")
        .to_device(device)
        .expect("dequant -> cuda")
        .contiguous()
        .expect("q_proj_t contiguous");
    // q_proj (untransposed) is only read for shape; the forward path only
    // touches q_proj_t and q_proj_marlin, so alias it here.
    let q_proj = q_proj_t.clone();

    // --- Build a synthetic activation of shape [1, m, k] (the forward path
    // consumes `[batch, seq, hidden]`).
    let acts_host = random_vec(m * k, 0xDEAD_BEEF_F00D_BABE ^ (m as u64), 0.25);
    let acts_cpu = Tensor::from_vec(acts_host, (1, m, k), &Device::Cpu).expect("acts cpu tensor");
    let x = acts_cpu
        .to_dtype(DType::BF16)
        .expect("acts -> bf16")
        .to_device(device)
        .expect("acts -> cuda");

    // Construct MarlinPackedProj from the same packed/scales as above.
    let b_packed = Tensor::from_vec(b_packed_vec, (k / 16, n * 16 / 8), &Device::Cpu)
        .expect("b_packed cpu tensor")
        .to_device(device)
        .expect("b_packed -> cuda");
    let num_groups = k / groupsize as usize;
    let scales = Tensor::from_vec(scales_vec, (num_groups, n), &Device::Cpu)
        .expect("scales cpu tensor")
        .to_device(device)
        .expect("scales -> cuda");
    let packed = MarlinPackedProj {
        b_packed,
        scales,
        groupsize,
        k,
        n,
    };

    // Dummy norm tensors just so we can construct a GpuFullAttentionWeights.
    // q_proj_forward itself only reads q_proj_t and q_proj_marlin, so the
    // other fields can alias q_proj and zero-filled norm vectors.
    let q_norm = Tensor::ones((1,), DType::BF16, device).expect("q_norm ones");
    let k_norm = q_norm.clone();

    let baseline_weights = kiln_model::forward::GpuFullAttentionWeights {
        q_proj: q_proj.clone(),
        k_proj: q_proj.clone(),
        v_proj: q_proj.clone(),
        o_proj: q_proj.clone(),
        q_norm: q_norm.clone(),
        k_norm: k_norm.clone(),
        q_proj_t: q_proj_t.clone(),
        k_proj_t: q_proj_t.clone(),
        v_proj_t: q_proj_t.clone(),
        o_proj_t: q_proj_t.clone(),
        q_proj_marlin: None, // KILN_W4A16 unset path
    };
    let marlin_weights = kiln_model::forward::GpuFullAttentionWeights {
        q_proj: q_proj.clone(),
        k_proj: q_proj.clone(),
        v_proj: q_proj.clone(),
        o_proj: q_proj.clone(),
        q_norm,
        k_norm,
        q_proj_t: q_proj_t.clone(),
        k_proj_t: q_proj_t.clone(),
        v_proj_t: q_proj_t.clone(),
        o_proj_t: q_proj_t.clone(),
        q_proj_marlin: Some(packed), // KILN_W4A16 set path
    };

    // --- Baseline (KILN_W4A16 unset): BF16 broadcast_matmul via q_proj_t.
    let baseline =
        q_proj_forward(&x, &baseline_weights, None, 0.0).expect("baseline q_proj_forward");
    // --- Marlin (KILN_W4A16 set): W4A16 kernel + (optional) LoRA delta.
    let marlin = q_proj_forward(&x, &marlin_weights, None, 0.0).expect("marlin q_proj_forward");

    assert_eq!(baseline.dims(), marlin.dims(), "shape mismatch");

    let cos = cosine_similarity(&baseline, &marlin);
    let mad = max_abs_diff(&baseline, &marlin);
    eprintln!("m={m} k={k} n={n} cosine_similarity={cos:.6} max_abs_diff={mad:.5}");

    // The Phase 6 wiring spec: cosine similarity >= 0.9995.
    assert!(
        cos >= 0.9995,
        "cosine similarity {cos:.6} < 0.9995 (max_abs_diff {mad:.5}) for m={m} k={k} n={n}"
    );
}

#[test]
#[cfg_attr(not(feature = "cuda"), ignore)]
fn q_proj_marlin_parity_qwen35_4b() {
    #[cfg(feature = "cuda")]
    {
        let device = match Device::new_cuda(0).ok() {
            Some(d) => d,
            None => {
                eprintln!("CUDA not available, skipping marlin q_proj parity test");
                return;
            }
        };

        // Qwen3.5-4B q_proj with attn_output_gate fused: k=hidden=2560,
        // n = num_heads * head_dim * 2 = 16 * 256 * 2 = 8192.
        // This satisfies Marlin's k%128 (2560/128=20) and n%256 (8192/256=32).
        let k = 2560;
        let n = 8192;

        // Decode (batch=1, seqlen=1).
        run_parity(&device, 1, k, n);
        // Prefill-ish (seqlen=128) to exercise the 3D reshape path in
        // marlin_proj::matmul_bf16.
        run_parity(&device, 128, k, n);
    }

    // On non-CUDA builds the test is #[ignore]'d and this body is stripped.
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("cuda feature off, no-op");
    }
}
